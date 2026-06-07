import os
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import TopKSAEConfig, TopKSAE
import gc
from tqdm import tqdm

LAYER = 2
D_IN = 768
CONTEXT_SIZE = 128
WEIGHTS_DIR = "weights"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENS_TO_SKIP_FOR_EVAL = 500_000_000  
CALIBRATION_TOKENS      = 4096 * 50   
HELD_OUT_TOTAL_TOKENS   = 2_000_000   
BITWIDTHS = [8, 6, 4, 2]
BOTTLENECK_SIZES = [512, 1024]
K_VALUES_SDS = [32, 64, 128]
TOPK_FRAC = 0.10


def get_token_stream(tokenizer, skip_tokens=0, max_tokens=None):
    print("[stream] initializing dataset stream...")
    dataset = load_dataset(
        "Skylion007/openwebtext", 
        split="train", 
        streaming=True, 
        trust_remote_code=True
    )
    print("[stream] dataset stream ready")
    
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    token_buffer = []
    tokens_skipped = 0
    tokens_yielded = 0
    skipping_done = (skip_tokens == 0)
    
    docs_seen = 0
    last_log = time.time()
    for doc in dataset:
        docs_seen += 1
        tokens = tokenizer.encode(doc["text"], add_special_tokens=False)
        
        if not skipping_done:
            if tokens_skipped + len(tokens) <= skip_tokens:
                tokens_skipped += len(tokens)
                if time.time() - last_log >= 30:
                    print(
                        f"[stream] skipping docs={docs_seen} skipped={tokens_skipped}/{skip_tokens} "
                        f"buffer={len(token_buffer)}"
                    )
                    last_log = time.time()
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

        if time.time() - last_log >= 30:
            print(
                f"[stream] docs={docs_seen} skipped={tokens_skipped} yielded={tokens_yielded} "
                f"buffer={len(token_buffer)}"
            )
            last_log = time.time()


def get_q_ranges(bitwidth):
    
    return 0, (1 << bitwidth) - 1

def compute_calibration_stats(model, sae, calib_chunks):
    global_min = torch.inf
    global_max = -torch.inf
    feature_min = torch.full((sae.cfg.d_sae,), torch.inf, device=DEVICE)
    feature_max = torch.full((sae.cfg.d_sae,), -torch.inf, device=DEVICE)
    
    batch_size_seqs = 32
    total_batches = (len(calib_chunks) + batch_size_seqs - 1) // batch_size_seqs
    
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        with tqdm(total=total_batches, desc="Calibration", unit="batch") as pbar:
            for i in range(0, len(calib_chunks), batch_size_seqs):
                batch_list = calib_chunks[i:i + batch_size_seqs]
                if not batch_list: break
                    
                tokens = torch.tensor(batch_list, dtype=torch.long, device=DEVICE)
                _, cache = model.run_with_cache(tokens, names_filter=[f"blocks.{LAYER}.hook_resid_post"])
                x = cache[f"blocks.{LAYER}.hook_resid_post"]
                z = sae.encode(x.to(torch.float32))
                
                global_min = min(global_min, z.min().item())
                global_max = max(global_max, z.max().item())
                
                z_flat = z.reshape(-1, sae.cfg.d_sae)
                feat_min_batch = z_flat.min(dim=0).values
                feat_max_batch = z_flat.max(dim=0).values
                
                feature_min = torch.minimum(feature_min, feat_min_batch)
                feature_max = torch.maximum(feature_max, feat_max_batch)
                
                pbar.update(1)

    return {
        "global_min": global_min,
        "global_max": global_max,
        "feature_min": feature_min,
        "feature_max": feature_max
    }

def quantize_activations(z, method, bitwidth, calib_stats):
    qmin, qmax = get_q_ranges(bitwidth)
    
    if method == "per_tensor":
        delta = calib_stats["global_max"] / float(qmax)
        delta = max(delta, 1e-8)
        
    elif method == "per_feature":
        delta = calib_stats["feature_max"] / float(qmax)
        delta = torch.clamp(delta, min=1e-8).view(1, 1, -1)
    
    z_hat = torch.clamp(torch.round(z / delta), qmin, qmax)
    z_tilde = z_hat * delta
    
    if isinstance(delta, torch.Tensor):
        delta = delta.squeeze().tolist()
        
    return z_tilde, delta

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


def compute_subspace_basis(Z_full_precision, k_values=K_VALUES_SDS):
    Z_centered = Z_full_precision - Z_full_precision.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(Z_centered.float(), full_matrices=False)
    V = Vh.mH 
    return {str(k): V[:, :k] for k in k_values if k <= V.shape[1]}

def compute_sds(Z, Z_hat, Uk_dict):
    sds_scores = {}
    for k_str, Uk in Uk_dict.items():
        numerator = torch.linalg.norm((Z - Z_hat) @ Uk, ord='fro') ** 2
        denominator = torch.linalg.norm(Z @ Uk, ord='fro') ** 2
        sds_scores[k_str] = (numerator / denominator).item() if denominator > 0 else 0.0
        
    return sds_scores

def evaluate_fp_sae_baseline(model, sae, ppl_chunks, sds_batch):
    total_nll = 0.0
    total_tokens = 0
    batch_size_seqs = 32
    hook_storage = {}

    def fp_hook_fast(resid, hook=None):
        z = sae.encode(resid.to(torch.float32))
        return sae.decode(z).to(resid.dtype)

    def fp_hook_with_save(resid, hook=None):
        hook_storage['x_orig'] = resid.detach().clone()
        z = sae.encode(resid.to(torch.float32))
        hook_storage['z'] = z.detach().clone()
        return sae.decode(z).to(resid.dtype)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        total_batches = (len(ppl_chunks) + batch_size_seqs - 1) // batch_size_seqs
        for i in range(0, len(ppl_chunks), batch_size_seqs):
            batch_list = ppl_chunks[i:i + batch_size_seqs]
            if not batch_list: break
            
            tokens = torch.tensor(batch_list, dtype=torch.long, device=DEVICE)
            loss = model.run_with_hooks(
                tokens, return_type="loss",
                fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", fp_hook_fast)]
            )
            token_count = tokens.numel() - tokens.size(0)
            total_nll += loss.item() * token_count
            total_tokens += token_count
            
        perplexity = np.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')        
        
        _ = model.run_with_hooks(
            sds_batch, return_type="logits", 
            fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", fp_hook_with_save)]
        )
        
        X_orig = hook_storage['x_orig']
        Z_fp = hook_storage['z']
        X_rec = sae.decode(Z_fp).to(X_orig.dtype)

        X_orig_flat = X_orig.view(-1, D_IN)
        X_rec_flat = X_rec.view(-1, D_IN)
        Z_flat = Z_fp.view(-1, sae.cfg.d_sae)

    mse = F.mse_loss(X_orig_flat.float(), X_rec_flat.float()).item()
    cka = linear_cka(X_orig_flat.float(), X_rec_flat.float())

    return {
        "perplexity": perplexity,
        "mse": mse,
        "cka": cka
    }, Z_flat 

def evaluate_quantization(model, sae, ppl_chunks, sds_batch, method, bitwidth, calib_stats, Uk_dict):
    qmin, qmax = get_q_ranges(bitwidth)
    total_nll = 0.0
    total_tokens = 0
    batch_size_seqs = 32 
    hook_storage = {}
    
    def quantize_hook_fast(resid, hook=None):
        z = sae.encode(resid.to(torch.float32))
        z_q, _ = quantize_activations(z, method, bitwidth, calib_stats)
        return sae.decode(z_q).to(resid.dtype)

    def quantize_hook_with_save(resid, hook=None):
        hook_storage['x_orig'] = resid.detach().clone()
        z = sae.encode(resid.to(torch.float32))
        hook_storage['z'] = z.detach().clone()
        z_q, step_size = quantize_activations(z, method, bitwidth, calib_stats)
        hook_storage['z_q'] = z_q.detach().clone()
        hook_storage['step_size'] = step_size
        return sae.decode(z_q).to(resid.dtype)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        total_batches = (len(ppl_chunks) + batch_size_seqs - 1) // batch_size_seqs
        with tqdm(total=total_batches, desc="Perplexity", unit="batch") as pbar:
            for i in range(0, len(ppl_chunks), batch_size_seqs):
                batch_list = ppl_chunks[i:i + batch_size_seqs]
                if not batch_list: break
                
                tokens = torch.tensor(batch_list, dtype=torch.long, device=DEVICE)
                loss = model.run_with_hooks(
                    tokens, return_type="loss",
                    fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", quantize_hook_fast)]
                )
                token_count = tokens.numel() - tokens.size(0)
                total_nll += loss.item() * token_count
                total_tokens += token_count
                
                avg_nll = total_nll / total_tokens
                pbar.set_postfix({"avg_nll": f"{avg_nll:.4f}"})
                pbar.update(1)
            
        perplexity = np.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')        
        
        _ = model.run_with_hooks(
            sds_batch, return_type="logits", 
            fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", quantize_hook_with_save)]
        )
        
        X_orig = hook_storage['x_orig']
        Z = hook_storage['z']
        Z_q = hook_storage['z_q']
        step_size = hook_storage['step_size']
        X_rec = sae.decode(Z_q).to(X_orig.dtype)

        X_orig_flat = X_orig.view(-1, D_IN)
        X_rec_flat = X_rec.view(-1, D_IN)
        Z_flat = Z.view(-1, sae.cfg.d_sae)
        Z_q_flat = Z_q.view(-1, sae.cfg.d_sae)

    mse = F.mse_loss(X_orig_flat.float(), X_rec_flat.float()).item()
    cka = linear_cka(X_orig_flat.float(), X_rec_flat.float())
    sds = compute_sds(Z_flat, Z_q_flat, Uk_dict) 

    return {
        "quant_step": step_size,
        "perplexity": perplexity,
        "mse": mse,
        "cka": cka,
        "sds_scores": sds
    }, Z_q_flat

def main():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = HookedTransformer.from_pretrained("distilgpt2", device=DEVICE)
    model.eval()

    print("--- PRE-FETCHING DATASETS (RUNS ONCE) ---")
    print("Gathering Calibration Stream (First ~200k tokens)...")
    calib_stream = get_token_stream(tokenizer, skip_tokens=0, max_tokens=CALIBRATION_TOKENS)
    calib_chunks = list(tqdm(calib_stream, total=CALIBRATION_TOKENS//CONTEXT_SIZE, desc="Calib chunks"))

    print(f"Pre-fetching 2M held-out tokens (Skipping {TOKENS_TO_SKIP_FOR_EVAL} tokens)...")
    total_chunks = HELD_OUT_TOTAL_TOKENS // CONTEXT_SIZE
    held_out_stream = get_token_stream(tokenizer, skip_tokens=TOKENS_TO_SKIP_FOR_EVAL, max_tokens=HELD_OUT_TOTAL_TOKENS)
    held_out_chunks = list(tqdm(held_out_stream, total=total_chunks, desc="Held-out chunks", unit="chunk"))

    sds_chunks = held_out_chunks[:32]
    umap_chunks = held_out_chunks[32:111]
    ppl_chunks = held_out_chunks[111:]

    sds_batch = torch.tensor(sds_chunks, dtype=torch.long, device=DEVICE)
    umap_batch = torch.tensor(umap_chunks, dtype=torch.long, device=DEVICE)
    torch.save(sds_batch.cpu(), "sds_tokens_4096.pt")
    ppl_batch = torch.tensor(ppl_chunks, dtype=torch.long)
    torch.save(ppl_batch, "ppl_tokens_2M.pt")
    
    print("--- DATASET PRE-FETCH COMPLETE ---\n")

    for d_sae in BOTTLENECK_SIZES:
        print(f"\n==============================================")
        print(f"Processing SAE Bottleneck Size: {d_sae}")
        print(f"==============================================")
        
        k = int(d_sae * TOPK_FRAC)
        run_name = f"sae_distilgpt2_l{LAYER}_{d_sae}"
        save_path = os.path.join(WEIGHTS_DIR, f"{run_name}_final.pt")
        
        if not os.path.exists(save_path):
            print(f"Skipping {d_sae}, weights not found at {save_path}")
            continue

        cfg = TopKSAEConfig(
            d_in=D_IN, d_sae=d_sae, k=k, normalize_activations="expected_average_only_in",
            apply_b_dec_to_input=True, dtype="float32", device=DEVICE,
        )
        sae = TopKSAE(cfg)
        sae.load_state_dict(torch.load(save_path, map_location=DEVICE))
        sae.eval()

        print("Computing Calibration Statistics using SAE...")
        calib_stats = compute_calibration_stats(model, sae, calib_chunks)
        
        print("Computing Full-Precision SAE baseline...")
        baseline_metrics, Z_flat_baseline = evaluate_fp_sae_baseline(model, sae, ppl_chunks, sds_batch)
        Uk_dict = compute_subspace_basis(Z_flat_baseline) 
        torch.save(Z_flat_baseline.cpu(), f"bottleneck_activations_m{d_sae}_4096_fp.pt")
        
        print("Extracting and saving FP UMAP bottleneck geometry...")
        umap_activations = []
        with torch.no_grad():
            for i in tqdm(range(0, len(umap_batch), 32), desc="UMAP encoding", unit="batch"):
                b = umap_batch[i:i+32]
                _, cache = model.run_with_cache(b, names_filter=[f"blocks.{LAYER}.hook_resid_post"])
                x = cache[f"blocks.{LAYER}.hook_resid_post"]
                z = sae.encode(x.to(torch.float32))
                umap_activations.append(z.cpu())
                
        umap_tensor = torch.cat(umap_activations, dim=0).view(-1, sae.cfg.d_sae)
        torch.save(umap_tensor, f"umap_activations_m{d_sae}_10112_fp.pt")
        del umap_activations, umap_tensor
        
        results_json = {
            "bottleneck_size": d_sae,
            "baseline": baseline_metrics, 
            "calibration_stats": {
                "per_tensor": {"min": calib_stats["global_min"], "max": calib_stats["global_max"]},
                "per_feature": {
                    "feature_min": calib_stats["feature_min"].tolist(),
                    "feature_max": calib_stats["feature_max"].tolist()
                }
            },
            "bitwidth_configuration": {}, 
            "metrics": {}
        }

        methods = ["per_tensor", "per_feature"]
        for method in methods:
            results_json["metrics"][method] = {}
            for bit in BITWIDTHS:
                print(f"Evaluating: {method} at {bit}-bit")
                
                if f"{bit}_bit" not in results_json["bitwidth_configuration"]:
                    qmin, qmax = get_q_ranges(bit)
                    results_json["bitwidth_configuration"][f"{bit}_bit"] = {"qmin": qmin, "qmax": qmax}
                
                metrics, Z_q_flat = evaluate_quantization(model, sae, ppl_chunks, sds_batch, method, bit, calib_stats, Uk_dict)
                results_json["metrics"][method][f"{bit}_bit"] = metrics
                torch.save(Z_q_flat.cpu(), f"bottleneck_activations_m{d_sae}_4096_{method}_{bit}bit.pt")
                del Z_q_flat
                
                if bit in BITWIDTHS:
                    quant_acts = []
                    with torch.no_grad():
                        for i in range(0, len(umap_batch), 32):
                            b = umap_batch[i:i+32]
                            _, cache = model.run_with_cache(b, names_filter=[f"blocks.{LAYER}.hook_resid_post"])
                            x = cache[f"blocks.{LAYER}.hook_resid_post"]
                            z = sae.encode(x.to(torch.float32))
                            z_q, _ = quantize_activations(z, method, bit, calib_stats)
                            quant_acts.append(z_q.cpu())
                    
                    quant_tensor = torch.cat(quant_acts, dim=0).view(-1, sae.cfg.d_sae)
                    torch.save(quant_tensor, f"umap_activations_m{d_sae}_10112_{method}_{bit}bit.pt")
                    del quant_acts, quant_tensor
                
                torch.cuda.empty_cache()
                gc.collect()

        json_path = f"metrics_m{d_sae}.json"
        with open(json_path, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"Successfully wrote evaluations to {json_path}")
        
        del sae, Z_flat_baseline, Uk_dict
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()