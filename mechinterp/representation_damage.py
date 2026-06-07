import os
import json
import torch
import gc
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import TopKSAEConfig, TopKSAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 2 
D_IN = 768
CONTEXT_SIZE = 128
BATCH_SIZE_SEQS = 32
BOTTLENECK_SIZES = [512, 1024]
METHODS = ["per_tensor", "per_feature"]
BITWIDTHS = [8, 6, 4, 2]
CAUSAL_CONFIG = {
    "causal_method": 3, 
    "subset_tokens": 50000, 
    "top_n_ablate": 50, 
    "k_tokens": 20,      
    "num_bins_kl": 100   
}

def get_q_ranges(bitwidth):
    return 0, (1 << bitwidth) - 1

def load_metrics_stats(d_sae):
    path = f"metrics_m{d_sae}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Please run Task 2 first.")
    with open(path, "r") as f:
        data = json.load(f)
    
    try:
        calib = data["calibration_stats"]
        stats = {
            "global_max": float(calib["per_tensor"]["max"]),
            "feature_max": torch.tensor(calib["per_feature"]["feature_max"], device=DEVICE)
        }
    except KeyError as e:
        raise KeyError(f"Unexpected JSON structure in {path}. Missing key: {e}.") from e
    return stats

def quantize_on_the_fly(Z, method, bitwidth, stats):
    qmin, qmax = get_q_ranges(bitwidth)
    if method == "per_tensor":
        delta = max(stats["global_max"] / float(qmax), 1e-8)
    else:
        delta = torch.clamp(stats["feature_max"] / float(qmax), min=1e-8)
        
    Z_hat = torch.clamp(torch.round(Z / delta), qmin, qmax)
    return Z_hat * delta

def compute_kl_divergence(p_counts, q_counts):
    epsilon = 1e-8
    p_probs = (p_counts + epsilon) / (p_counts.sum(dim=1, keepdim=True) + epsilon * p_counts.shape[1])
    q_probs = (q_counts + epsilon) / (q_counts.sum(dim=1, keepdim=True) + epsilon * q_counts.shape[1])
    kl = torch.sum(p_probs * torch.log(p_probs / q_probs), dim=1)
    return kl

def streaming_representation_damage(model, sae, d_sae, dataset, stats, tokenizer):
    print(f"\n--- Step 2: Streaming Representation Damage (m={d_sae}) ---")
    
    k_top = CAUSAL_CONFIG["k_tokens"]
    num_bins = CAUSAL_CONFIG["num_bins_kl"]
    max_val = stats["global_max"]
    
    l2_scores = {f"{m}_{b}b": torch.zeros(d_sae, device=DEVICE) for m in METHODS for b in BITWIDTHS}
    hist_fp = torch.zeros((d_sae, num_bins), device=DEVICE)
    hist_q = {f"{m}_{b}b": torch.zeros((d_sae, num_bins), device=DEVICE) for m in METHODS for b in BITWIDTHS}
    
    top_vals = torch.full((d_sae, k_top), -1.0, device=DEVICE)
    top_contexts = torch.zeros((d_sae, k_top, 9), dtype=torch.long, device=DEVICE)
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    def hook_fn(resid, hook):
        Z = sae.encode(resid.to(torch.float32))
        Z_flat = Z.view(-1, d_sae)
        
        bin_indices = torch.clamp((Z_flat / max_val) * num_bins, 0, num_bins - 1).long().t()
        hist_fp.scatter_add_(1, bin_indices, torch.ones_like(bin_indices, dtype=torch.float32))
        
        Z_t = Z_flat.t()
        chunk_vals, chunk_idx = torch.topk(Z_t, min(k_top, Z_t.shape[1]), dim=1)
        
        b_idx = chunk_idx // CONTEXT_SIZE
        s_idx = chunk_idx % CONTEXT_SIZE
        
        padded_tokens = torch.nn.functional.pad(tokens, (4, 4), value=pad_id)
        idx_offsets = torch.arange(9, device=DEVICE)
        window_idx = s_idx.unsqueeze(-1) + idx_offsets 
        b_idx_expanded = b_idx.unsqueeze(-1).expand(-1, -1, 9)
        
        chunk_contexts = padded_tokens[b_idx_expanded, window_idx]
        
        combined_vals = torch.cat([top_vals, chunk_vals], dim=1)
        combined_contexts = torch.cat([top_contexts, chunk_contexts], dim=1)
        
        new_top_vals, new_top_idx = torch.topk(combined_vals, k_top, dim=1)
        top_vals.copy_(new_top_vals)
        
        expanded_new_top_idx = new_top_idx.unsqueeze(-1).expand(-1, -1, 9)
        top_contexts.copy_(torch.gather(combined_contexts, 1, expanded_new_top_idx))
        
        for method in METHODS:
            for bit in BITWIDTHS:
                key = f"{method}_{bit}b"
                Z_q = quantize_on_the_fly(Z_flat, method, bit, stats)
                
                diff_sq = (Z_flat - Z_q) ** 2
                l2_scores[key] += diff_sq.sum(dim=0)
                
                bin_idx_q = torch.clamp((Z_q / max_val) * num_bins, 0, num_bins - 1).long().t()
                hist_q[key].scatter_add_(1, bin_idx_q, torch.ones_like(bin_idx_q, dtype=torch.float32))
                
                del Z_q, diff_sq, bin_idx_q
                
        del Z, Z_flat, Z_t, bin_indices, chunk_vals, chunk_idx, \
            combined_vals, new_top_vals, new_top_idx, expanded_new_top_idx, \
            padded_tokens, b_idx, s_idx, window_idx, b_idx_expanded, \
            chunk_contexts, combined_contexts
        return resid

    total_batches = (len(dataset) + BATCH_SIZE_SEQS - 1) // BATCH_SIZE_SEQS
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), BATCH_SIZE_SEQS), total=total_batches, desc="Streaming 2M Tokens"):
            batch = dataset[i : i + BATCH_SIZE_SEQS]
            tokens = batch.clone().detach().to(DEVICE).long()
            
            _ = model.run_with_hooks(
                tokens, return_type=None,
                fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", hook_fn)]
            )
            
            del batch, tokens
            torch.cuda.empty_cache()
            
    final_metrics = {
        "l2": {}, "kl": {}, 
        "top_damaged_lists": {
            "l2_ranked": {}, "kl_ranked": {}, "borda_ranked": {}
        }
    }
    total_tokens = len(dataset) * CONTEXT_SIZE
    
    for key in l2_scores.keys():
        l2_tensor = torch.sqrt(l2_scores[key] / total_tokens)
        kl_tensor = compute_kl_divergence(hist_fp, hist_q[key])
        
        final_metrics["l2"][key] = l2_tensor.tolist()
        final_metrics["kl"][key] = kl_tensor.tolist()
        
        l2_sorted = torch.argsort(l2_tensor, descending=True, stable=True)
        final_metrics["top_damaged_lists"]["l2_ranked"][key] = l2_sorted.tolist()
        
        kl_sorted = torch.argsort(kl_tensor, descending=True, stable=True)
        final_metrics["top_damaged_lists"]["kl_ranked"][key] = kl_sorted.tolist()
        
        l2_pos = torch.zeros_like(l2_sorted)
        l2_pos[l2_sorted] = torch.arange(d_sae, device=DEVICE)
        
        kl_pos = torch.zeros_like(kl_sorted)
        kl_pos[kl_sorted] = torch.arange(d_sae, device=DEVICE)
        
        borda_scores = l2_pos + kl_pos
        borda_sorted = torch.argsort(borda_scores, descending=False, stable=True)
        final_metrics["top_damaged_lists"]["borda_ranked"][key] = borda_sorted.tolist()

    top_n = CAUSAL_CONFIG["top_n_ablate"]
    union_damaged_ids = set()
    
    for rank_method in final_metrics["top_damaged_lists"].values():
        for lst in rank_method.values():
            union_damaged_ids.update(lst[:top_n])
            
    top_tokens_text = {}
    for neuron_idx in union_damaged_ids:
        contexts = top_contexts[neuron_idx].tolist()
        vals = top_vals[neuron_idx].tolist()
        
        valid_contexts = []
        for i in range(k_top):
            if vals[i] > 0.0:
                left_tokens = contexts[i][:4]
                mid_token = contexts[i][4:5]  
                right_tokens = contexts[i][5:]
                
                left_str = tokenizer.decode(left_tokens)
                mid_str = tokenizer.decode(mid_token)
                right_str = tokenizer.decode(right_tokens)
                ctx_str = f"{left_str}<< {mid_str} >>{right_str}".replace("Ġ", " ")
                valid_contexts.append(f"{ctx_str} (act: {vals[i]:.2f})")
                
        top_tokens_text[str(neuron_idx)] = valid_contexts

    return final_metrics, top_tokens_text

def causal_importance(model, sae, d_sae, dataset):
    print(f"\n--- Step 3: Causal Importance (Toggle {CAUSAL_CONFIG['causal_method']}) ---")
    method = CAUSAL_CONFIG["causal_method"]
    
    if method == 1:
        print("!! WARNING: Toggle 1 (Brute Force) selected. Running single-neuron ablations "
              f"over the full 2M-token dataset for {d_sae} neurons. This will be extremely slow.")
        subset = dataset
    elif method == 2:
        subset = dataset[: CAUSAL_CONFIG["subset_tokens"] // CONTEXT_SIZE]
    elif method == 3:
        subset = dataset[: CAUSAL_CONFIG["subset_tokens"] // CONTEXT_SIZE]
        
    scores = torch.zeros(d_sae, device=DEVICE)

    if method in [1, 2]:
        def get_ppl(ablate_idx=None):
            total_nll, total_t = 0.0, 0
            def ablate_hook(resid, hook):
                z = sae.encode(resid.to(torch.float32))
                if ablate_idx is not None:
                    z[:, :, ablate_idx] = 0.0
                return sae.decode(z).to(resid.dtype)

            with torch.no_grad():
                for i in range(0, len(subset), BATCH_SIZE_SEQS):
                    tokens = subset[i:i+BATCH_SIZE_SEQS].to(DEVICE).long()
                    loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", ablate_hook)])
                    count = tokens.numel() - tokens.size(0)
                    total_nll += loss.item() * count
                    total_t += count
                    del tokens
                    torch.cuda.empty_cache()
            return total_nll / total_t

        baseline_nll = get_ppl()
        for neuron in tqdm(range(d_sae), desc="Single-Neuron Ablations"):
            scores[neuron] = get_ppl(neuron) - baseline_nll
            
    elif method == 3:
        model.eval()
        model.requires_grad_(False)
        attr_accum = torch.zeros(d_sae, device=DEVICE)
        
        def attr_hook(resid, hook):
            with torch.no_grad():
                z = sae.encode(resid.to(torch.float32)).detach()
            z.requires_grad_(True)
            hook.ctx['z'] = z
            return sae.decode(z).to(resid.dtype)
            
        for i in tqdm(range(0, len(subset), BATCH_SIZE_SEQS), desc="Gradient Attribution"):
            tokens = subset[i:i+BATCH_SIZE_SEQS].to(DEVICE).long()
            loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", attr_hook)])
    
            loss.backward()
            
            ctx_z = model.hook_dict[f"blocks.{LAYER}.hook_resid_post"].ctx['z']
            if ctx_z.grad is None:
                raise RuntimeError("ctx_z.grad is None after backward(). The gradient graph is broken.")
            
            batch_attr = torch.abs(ctx_z * ctx_z.grad)
            attr_accum += batch_attr.sum(dim=(0, 1))
            
            del tokens, loss, ctx_z, batch_attr
            model.hook_dict[f"blocks.{LAYER}.hook_resid_post"].ctx.clear()
            torch.cuda.empty_cache()
            
        scores = attr_accum / (len(subset) * CONTEXT_SIZE)
    
    _, ranked_indices = torch.sort(scores, descending=True)
    return scores, ranked_indices

def evaluate_alignment_and_ablations(model, sae, d_sae, dataset, damage_metrics, ranked_causal):
    print("\n--- Step 4: Group Ablations & Alignment ---")
    N = CAUSAL_CONFIG["top_n_ablate"]
    causal_list = ranked_causal.tolist()
    
    def get_ranks(ranked_indices):
        ranks = np.zeros(d_sae)
        ranks[ranked_indices] = np.arange(d_sae)
        return ranks
        
    causal_ranks = get_ranks(causal_list)
    subset = dataset[: CAUSAL_CONFIG["subset_tokens"] // CONTEXT_SIZE]
    
    def eval_group(ablate_indices):
        total_nll, total_t = 0.0, 0
        def grp_hook(resid, hook):
            z = sae.encode(resid.to(torch.float32))
            if len(ablate_indices) > 0:
                z[:, :, ablate_indices] = 0.0
            return sae.decode(z).to(resid.dtype)

        with torch.no_grad():
            for i in tqdm(range(0, len(subset), BATCH_SIZE_SEQS), desc=f"Eval ({len(ablate_indices)} ablations)", leave=False):
                tokens = subset[i:i+BATCH_SIZE_SEQS].to(DEVICE).long()
                loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", grp_hook)])
                count = tokens.numel() - tokens.size(0)
                total_nll += loss.item() * count
                total_t += count
                del tokens
                torch.cuda.empty_cache()
        return total_nll / total_t

    baseline_nll = eval_group([])
    
    random_spikes = []
    for _ in range(3):
        rand_idx = torch.randperm(d_sae)[:N].tolist()
        random_spikes.append(eval_group(rand_idx) - baseline_nll)
    avg_random_spike = sum(random_spikes) / 3.0
    
    eval_results = {
        "random_N_delta": avg_random_spike,
        "configurations": {}
    }
    
    configs = list(damage_metrics["top_damaged_lists"]["l2_ranked"].keys())
    ranking_methods = ["l2_ranked", "kl_ranked", "borda_ranked"]
    
    for config in configs:
        eval_results["configurations"][config] = {}
        for rank_method in ranking_methods:
            damaged_list = damage_metrics["top_damaged_lists"][rank_method][config]
            
            overlap_10 = len(set(damaged_list[:10]).intersection(set(causal_list[:10]))) / 10.0
            overlap_25 = len(set(damaged_list[:25]).intersection(set(causal_list[:25]))) / 25.0
            overlap_50 = len(set(damaged_list[:50]).intersection(set(causal_list[:50]))) / 50.0
            
            damaged_ranks = get_ranks(damaged_list)
            corr, _ = spearmanr(damaged_ranks, causal_ranks)
            
            nll_spike = eval_group(damaged_list[:N]) - baseline_nll
            
            eval_results["configurations"][config][rank_method] = {
                "overlap_10": overlap_10 * 100,
                "overlap_25": overlap_25 * 100,
                "overlap_50": overlap_50 * 100,
                "spearman_corr": float(corr),
                "nll_spike": nll_spike
            }
            
    return eval_results

def main():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = HookedTransformer.from_pretrained("distilgpt2", device=DEVICE)
    model.eval()

    print("Loading 2M Token Dataset...")
    dataset = torch.load("ppl_tokens_2M.pt", map_location="cpu", weights_only=True) 

    master_json = {}

    for d_sae in BOTTLENECK_SIZES:
        master_json[str(d_sae)] = {}
        
        run_name = f"sae_distilgpt2_l{LAYER}_{d_sae}"
        save_path = os.path.join("weights", f"{run_name}_final.pt")
        cfg = TopKSAEConfig(d_in=D_IN, d_sae=d_sae, k=int(d_sae*0.1), device=DEVICE)
        sae = TopKSAE(cfg)
        sae.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
        sae.eval()
        sae.requires_grad_(False)

        stats = load_metrics_stats(d_sae)
        
        damage_metrics, top_tokens = streaming_representation_damage(model, sae, d_sae, dataset, stats, tokenizer)
        
        causal_scores, ranked_causal = causal_importance(model, sae, d_sae, dataset)
        
        if CAUSAL_CONFIG["causal_method"] == 3:
            torch.save(causal_scores.cpu(), f"attributions_m{d_sae}.pt")
            print(f"Saved: attributions_m{d_sae}.pt")

        ablation_metrics = evaluate_alignment_and_ablations(
            model, sae, d_sae, dataset, damage_metrics, ranked_causal
        )

        master_json[str(d_sae)] = {
            "l2_divergence_scores": damage_metrics["l2"],
            "kl_divergence_scores": damage_metrics["kl"],
            "top_damaged_neurons_lists": damage_metrics["top_damaged_lists"],
            "top_activating_tokens": top_tokens,
            "perplexity_impact_scores": causal_scores.tolist(),
            "most_impacting_neurons_list": ranked_causal.tolist(),
            "random_N_delta": ablation_metrics["random_N_delta"],
            "configurations": ablation_metrics["configurations"]
        }
        
        del sae, stats, damage_metrics, causal_scores, ranked_causal, top_tokens
        gc.collect()
        torch.cuda.empty_cache()

    method = CAUSAL_CONFIG["causal_method"]
    output_filename = f"task3_metrics_method{method}.json"
    
    with open(output_filename, "w") as f:
        json.dump(master_json, f, indent=4)
    print(f"\nSuccessfully wrote all outputs to {output_filename}")
    
if __name__ == "__main__":
    main()