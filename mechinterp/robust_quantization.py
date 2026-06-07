import os
import json
import random
import gc
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens import TopKSAEConfig, TopKSAE
from datasets import load_dataset
from transformers import AutoTokenizer

LAYER = 2
D_IN = 768
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOTTLENECK_SIZES = [512, 1024]
K_VALUES = [32, 64, 128]
RESIDUAL_BITWIDTHS = [2, 4]
CONTEXT_SIZE = 128

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Setup] Seed set to {seed}")

def load_sae(d_sae):
    run_name = f"sae_distilgpt2_l{LAYER}_{d_sae}"
    save_path = os.path.join("weights", f"{run_name}_final.pt")
    
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"[Error] SAE weights not found at {save_path}")
        
    cfg = TopKSAEConfig(d_in=D_IN, d_sae=d_sae, k=int(d_sae * 0.1), device=DEVICE)
    sae = TopKSAE(cfg)
    sae.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    sae.eval()
    return sae

def get_token_stream(tokenizer, skip_tokens=0, max_tokens=None):
    print("[stream] initializing dataset stream...")
    dataset = load_dataset(
        "Skylion007/openwebtext", 
        split="train", 
        streaming=True, 
        trust_remote_code=True
    )
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    token_buffer = []
    tokens_skipped = 0
    tokens_yielded = 0
    skipping_done = (skip_tokens == 0)
    
    for doc in dataset:
        tokens = tokenizer.encode(doc["text"], add_special_tokens=False)
        if not skipping_done:
            if tokens_skipped + len(tokens) <= skip_tokens:
                tokens_skipped += len(tokens)
                continue
            else:
                leftover = skip_tokens - tokens_skipped
                tokens = tokens[leftover:]
                tokens_skipped = skip_tokens
                skipping_done = True
                
        token_buffer.extend(tokens)
        while len(token_buffer) >= (CONTEXT_SIZE - 1):
            chunk = [bos_id] + token_buffer[:CONTEXT_SIZE - 1]
            token_buffer = token_buffer[CONTEXT_SIZE - 1:]
            yield chunk
            tokens_yielded += CONTEXT_SIZE
            if max_tokens and tokens_yielded >= max_tokens:
                return

def linear_cka(X, Y):
    X_c = X.to(torch.float64)
    Y_c = Y.to(torch.float64)
    
    X_c = X_c - X_c.mean(dim=0, keepdim=True)
    Y_c = Y_c - Y_c.mean(dim=0, keepdim=True)
    
    X_c = X_c / (torch.linalg.norm(X_c, ord='fro') + 1e-8)
    Y_c = Y_c / (torch.linalg.norm(Y_c, ord='fro') + 1e-8)
    
    num = (torch.linalg.norm(X_c.t() @ Y_c, ord='fro') ** 2).item()
    den = (torch.linalg.norm(X_c.t() @ X_c, ord='fro') * torch.linalg.norm(Y_c.t() @ Y_c, ord='fro')).item()
    return num / den if den > 0 else 0.0

def compute_sds_score(Z_full, Z_preserved, Uk):
    numerator = torch.linalg.norm((Z_full - Z_preserved) @ Uk, ord='fro') ** 2
    denominator = torch.linalg.norm(Z_full @ Uk, ord='fro') ** 2
    return (numerator / denominator).item() if denominator > 0 else 0.0

def static_signed_quantize(tensor, bitwidth, max_val):
    qmin = -(1 << (bitwidth - 1))
    qmax = (1 << (bitwidth - 1)) - 1
    delta = max_val / float(qmax)
    delta = max(delta, 1e-8)
    
    quantized = torch.clamp(torch.round(tensor / delta), qmin, qmax)
    return quantized * delta

def make_subspace_quant_hook(sae, U_k, mu, bitwidth_res, max_val_imp, max_val_res):
    def hook_fn(resid, hook):
        orig_shape = resid.shape
        x_flat = resid.view(-1, resid.size(-1)).to(torch.float32)
        
        z = sae.encode(x_flat)
        z_centered = z - mu
        
        P = z_centered @ U_k
        z_imp = P @ U_k.T
        z_res = z_centered - z_imp
        
        P_tilde = static_signed_quantize(P, bitwidth=8, max_val=max_val_imp)
        z_imp_tilde = P_tilde @ U_k.T
        z_res_tilde = static_signed_quantize(z_res, bitwidth=bitwidth_res, max_val=max_val_res)
        

        z_preserved = z_imp_tilde + z_res_tilde + mu
        
        x_rec = sae.decode(z_preserved).to(resid.dtype)
        return x_rec.view(orig_shape)
    return hook_fn

def main():
    set_seed(42)
    print("[Initialization] Loading model, tokenizer, and validation datasets...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = HookedTransformer.from_pretrained("distilgpt2", device=DEVICE)
    model.eval()
    
    CALIBRATION_TOKENS = 4096 * 50  
    calib_stream = get_token_stream(tokenizer, skip_tokens=0, max_tokens=CALIBRATION_TOKENS)
    calib_chunks = list(tqdm(calib_stream, total=CALIBRATION_TOKENS//CONTEXT_SIZE, desc="Pre-fetching Calib Chunks"))
    
    sds_batch = torch.load("sds_tokens_4096.pt", map_location=DEVICE, weights_only=True)
    ppl_tokens_cpu = torch.load("ppl_tokens_2M.pt", map_location="cpu", weights_only=True)

    with torch.no_grad():
        _, cache = model.run_with_cache(sds_batch, names_filter=[f"blocks.{LAYER}.hook_resid_post"])
        X_orig = cache[f"blocks.{LAYER}.hook_resid_post"].view(-1, D_IN).float()
    
    for d_sae in BOTTLENECK_SIZES:
        print(f"\n==================================================")
        print(f"PROCESSING SAE BOTTLENECK SIZE: m = {d_sae}")
        print(f"==================================================")
        
        try:
            sae = load_sae(d_sae)
            V_master = torch.load(f"U_k_m{d_sae}.pt", map_location=DEVICE, weights_only=True).float()
            Z_baseline = torch.load(f"bottleneck_activations_m{d_sae}_4096_fp.pt", map_location=DEVICE, weights_only=True).float()
        except FileNotFoundError as e:
            print(f"[Warning] Required assets missing for m={d_sae}. Skipping. Details: {e}")
            continue

        mu = Z_baseline.mean(dim=0, keepdim=True)
        Z_centered = Z_baseline - mu
        
        print(f"[Calibration] Extracting unquantized activations from 200k calibration tokens (m={d_sae})...")
        Z_calib_list = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            for i in range(0, len(calib_chunks), 32):
                batch_list = calib_chunks[i:i+32]
                tokens = torch.tensor(batch_list, dtype=torch.long, device=DEVICE)
                _, cache = model.run_with_cache(tokens, names_filter=[f"blocks.{LAYER}.hook_resid_post"])
                x = cache[f"blocks.{LAYER}.hook_resid_post"]
                z = sae.encode(x.to(torch.float32))
                Z_calib_list.append(z.cpu()) 
        Z_calib = torch.cat(Z_calib_list, dim=0).view(-1, d_sae)
        del Z_calib_list
        gc.collect()

        print("[Baseline] Evaluating Full-Precision SAE benchmarks...")
        with torch.no_grad():
            X_rec_baseline = sae.decode(Z_baseline).view(-1, D_IN).float()
            baseline_mse = F.mse_loss(X_orig, X_rec_baseline).item()
            baseline_cka = linear_cka(X_orig, X_rec_baseline)
            
            total_nll, total_tokens = 0.0, 0
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                for i in range(0, len(ppl_tokens_cpu), 32):
                    chunk = ppl_tokens_cpu[i:i+32].to(DEVICE)
                    loss = model.run_with_hooks(
                        chunk, return_type="loss",
                        fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", lambda r, hook: sae.decode(sae.encode(r.to(torch.float32))).to(r.dtype))]
                    )
                    cnt = chunk.numel() - chunk.size(0)
                    total_nll += loss.item() * cnt
                    total_tokens += cnt
            baseline_ppl = np.exp(total_nll / total_tokens)
        
        results_json = {
            "bottleneck_size": d_sae,
            "baseline": {"perplexity": baseline_ppl, "mse": baseline_mse, "cka": baseline_cka},
            "subspace_preserving_quantization": {}
        }
    
        grid_elements = [(k, b_res) for k in K_VALUES for b_res in RESIDUAL_BITWIDTHS]
        
        for k, bitwidth_res in tqdm(grid_elements, desc=f"Evaluating Subspaces (m={d_sae})"):
            k_key = f"k_{k}"
            config_key = f"8bit_important_{bitwidth_res}bit_residual"
            
            if k_key not in results_json["subspace_preserving_quantization"]:
                results_json["subspace_preserving_quantization"][k_key] = {}
                
            U_k = V_master[:, :k]
            
            Z_calib_centered = Z_calib - mu.cpu()
            U_k_cpu = U_k.cpu()
            
            P_calib = Z_calib_centered @ U_k_cpu
            Z_res_calib = Z_calib_centered - (P_calib @ U_k_cpu.T)
            
            max_val_imp = torch.max(torch.abs(P_calib)).item()
            max_val_res = torch.max(torch.abs(Z_res_calib)).item()
            
            del Z_calib_centered, U_k_cpu, P_calib, Z_res_calib

            P = Z_centered @ U_k
            Z_imp = P @ U_k.T
            Z_res = Z_centered - Z_imp
            
            P_tilde = static_signed_quantize(P, bitwidth=8, max_val=max_val_imp)
            Z_imp_tilde = P_tilde @ U_k.T
            Z_res_tilde = static_signed_quantize(Z_res, bitwidth=bitwidth_res, max_val=max_val_res)
            
            Z_preserved = Z_imp_tilde + Z_res_tilde + mu
            
            with torch.no_grad():
                X_rec_quant = sae.decode(Z_preserved).view(-1, D_IN).float()
                eval_mse = F.mse_loss(X_orig, X_rec_quant).item()
                eval_cka = linear_cka(X_orig, X_rec_quant)
                
                sds_profile = {}
                for eval_k in K_VALUES:
                    U_eval = V_master[:, :eval_k]
                    sds_profile[str(eval_k)] = compute_sds_score(Z_baseline, Z_preserved, U_eval)
            
            total_nll, total_tokens = 0.0, 0
            patch_hook = make_subspace_quant_hook(sae, U_k, mu, bitwidth_res, max_val_imp, max_val_res)
            
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                for i in range(0, len(ppl_tokens_cpu), 32):
                    chunk = ppl_tokens_cpu[i:i+32].to(DEVICE)
                    loss = model.run_with_hooks(
                        chunk, return_type="loss",
                        fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", patch_hook)]
                    )
                    cnt = chunk.numel() - chunk.size(0)
                    total_nll += loss.item() * cnt
                    total_tokens += cnt
            
            eval_ppl = np.exp(total_nll / total_tokens)
            results_json["subspace_preserving_quantization"][k_key][config_key] = {
                "perplexity": eval_ppl,
                "mse": eval_mse,
                "cka": eval_cka,
                "sds_scores": sds_profile
            }
            
            del patch_hook
            del U_k, P, Z_imp, Z_res, P_tilde, Z_imp_tilde, Z_res_tilde, Z_preserved, X_rec_quant
            torch.cuda.empty_cache()
            gc.collect()

        output_path = f"metrics_m{d_sae}_task4.json"
        with open(output_path, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"[Success] Completed evaluations written to {output_path}")
        
        del sae, V_master, Z_baseline, mu, Z_centered, Z_calib  
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()