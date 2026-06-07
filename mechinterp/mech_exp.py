import os
import json
import torch
import random
import gc
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
from sae_lens import TopKSAEConfig, TopKSAE

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
LAYER        = 2       
D_IN         = 768

def safe_float(x):
    f = float(x)
    return None if (np.isnan(f) or np.isinf(f)) else f

def spectral_entropy_rank(sv):
    sv_np = np.array([x for x in sv if x is not None])
    sv_np = sv_np[sv_np > 0]
    if sv_np.size == 0:
        return 0.0
        
    p = sv_np / sv_np.sum()
    H = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(H))

def calc_collapse_ratio(rank_fp, rank_q):
    return 1.0 - (rank_q / max(rank_fp, 1e-6))

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
        
    cfg = TopKSAEConfig(d_in=D_IN, d_sae=d_sae, k=int(d_sae*0.1), device=DEVICE)
    sae = TopKSAE(cfg)
    sae.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    sae.eval() 
    
    return sae

def main():
    set_seed(42)
    
    bottleneck_sizes = [512, 1024]
    final_metrics = {}

    spectral_file = "spectral_analysis_metrics.json"
    spectral_data = {}
    if os.path.exists(spectral_file):
        with open(spectral_file, "r") as f:
            spectral_data = json.load(f)
        print("[Info] Loaded spectral_analysis_metrics.json successfully.")
    else:
        print(f"[Warning] {spectral_file} not found. Spectral extraction will output None.")
        
    task3_file = "task3_metrics_method2.json"
    task3_data = {}
    if os.path.exists(task3_file):
        with open(task3_file, "r") as f:
            task3_data = json.load(f)
        print("[Info] Loaded task3_metrics_method2.json successfully.")
    else:
        print(f"[Warning] {task3_file} not found. Sparsity fragility correlations will fail.")

    for d_sae in tqdm(bottleneck_sizes, desc="Processing SAEs"):
        key = f"m{d_sae}"
        print(f"\n--- Analyzing d_sae = {d_sae} ---")
        
        attr_file = f"attributions_{key}.pt"
        if not os.path.exists(attr_file):
            print(f"[Error] Missing {attr_file}. Please ensure Task 3 outputs are in the directory.")
            continue
            
        fisher_scores = torch.load(attr_file, map_location="cpu")
        assert fisher_scores.shape == (d_sae,), f"Expected shape ({d_sae},), got {fisher_scores.shape}."
        print(f"[Step 1] Loaded {attr_file} (Shape: {fisher_scores.shape})")

        sae = load_sae(d_sae)
        W_enc = sae.W_enc.detach().float()
        
        assert W_enc.shape == (D_IN, d_sae), f"W_enc shape mismatch: expected ({D_IN}, {d_sae}), got {W_enc.shape}"
        
        jacobian_norms = torch.norm(W_enc, p=2, dim=0)
        print(f"[Step 2] Computed Jacobian norms (Shape: {jacobian_norms.shape})")
        
        fisher_np = fisher_scores.detach().cpu().numpy()
        jacobian_np = jacobian_norms.detach().cpu().numpy()
        
        fragility_metrics = {}
        
        corr_fish_jac, p_fish_jac = spearmanr(fisher_np, jacobian_np)
        fragility_metrics["sensitivity_vs_importance_spearman"] = safe_float(corr_fish_jac)
        fragility_metrics["sensitivity_vs_importance_pvalue"] = safe_float(p_fish_jac)
        
        top50_fish = np.argsort(fisher_np)[::-1][:50]
        top50_jac = np.argsort(jacobian_np)[::-1][:50]
        overlap_count = len(set(top50_fish).intersection(set(top50_jac)))
        fragility_metrics["top50_sensitivity_importance_overlap_pct"] = safe_float((overlap_count / 50.0) * 100.0)
        
        QUANT_CONFIGS = [
            "per_tensor_8b", "per_tensor_6b", "per_tensor_4b", "per_tensor_2b",
            "per_feature_8b", "per_feature_6b", "per_feature_4b", "per_feature_2b"
        ]
        
        if str(d_sae) in task3_data:
            l2_scores_dict = task3_data[str(d_sae)].get("l2_divergence_scores", {})
            kl_scores_dict = task3_data[str(d_sae)].get("kl_divergence_scores", {})
            
            for config in QUANT_CONFIGS:
                l2_damage = l2_scores_dict.get(config, [])
                kl_damage = kl_scores_dict.get(config, [])
                
                if not l2_damage or not kl_damage:
                    continue
                
                l2_damage_np = np.array(l2_damage)
                kl_damage_np = np.array(kl_damage)
                
                corr_jac_l2, p_jac_l2 = spearmanr(jacobian_np, l2_damage_np)
                corr_fish_l2, p_fish_l2 = spearmanr(fisher_np, l2_damage_np)
                
                corr_jac_kl, p_jac_kl = spearmanr(jacobian_np, kl_damage_np)
                corr_fish_kl, p_fish_kl = spearmanr(fisher_np, kl_damage_np)

                top50_l2_damaged = np.argsort(l2_damage_np)[::-1][:50]
                top50_kl_damaged = np.argsort(kl_damage_np)[::-1][:50]
                overlap_l2_dmg_imp = len(set(top50_l2_damaged).intersection(set(top50_fish)))
                overlap_kl_dmg_imp = len(set(top50_kl_damaged).intersection(set(top50_fish)))
                overlap_l2_dmg_sens = len(set(top50_l2_damaged).intersection(set(top50_jac)))
                overlap_kl_dmg_sens = len(set(top50_kl_damaged).intersection(set(top50_jac)))
                print(f'top 50 fish: {set(top50_fish)}')
                print(f'top 50 jac: {set(top50_jac)}')
                print(f'top 50 l2 damaged: {set(top50_l2_damaged)}')
                print(f'top 50 kl damaged: {set(top50_kl_damaged)}')
                
                fragility_metrics[config] = {
                    "sensitivity_vs_l2_damage_spearman": safe_float(corr_jac_l2),
                    "sensitivity_vs_l2_damage_pvalue": safe_float(p_jac_l2),
                    "importance_vs_l2_damage_spearman": safe_float(corr_fish_l2),
                    "importance_vs_l2_damage_pvalue": safe_float(p_fish_l2),
                    "sensitivity_vs_kl_damage_spearman": safe_float(corr_jac_kl),
                    "sensitivity_vs_kl_damage_pvalue": safe_float(p_jac_kl),
                    "importance_vs_kl_damage_spearman": safe_float(corr_fish_kl),
                    "importance_vs_kl_damage_pvalue": safe_float(p_fish_kl),
                    "top50_l2_damage_importance_overlap_pct": safe_float((overlap_l2_dmg_imp / 50.0) * 100.0),
                    "top50_kl_damage_importance_overlap_pct": safe_float((overlap_kl_dmg_imp / 50.0) * 100.0),
                    "top50_l2_damage_sensitivity_overlap_pct": safe_float((overlap_l2_dmg_sens / 50.0) * 100.0),
                    "top50_kl_damage_sensitivity_overlap_pct": safe_float((overlap_kl_dmg_sens / 50.0) * 100.0)
                }
        print(f"[Step 3] Calculated fragility metrics across {len(fragility_metrics)-2} configurations.")

        sv_orig = []
        configurations_data = {}
        
        json_key = str(d_sae) 
        
        if json_key in spectral_data:
            model_spectral = spectral_data[json_key]
            
            sv_orig = model_spectral.get("fp_singular_values", [])
            eff_rank_fp = spectral_entropy_rank(sv_orig) 
            
            for config_name, config_metrics in tqdm(model_spectral.items(), desc=f"Spectral configs (m={d_sae})", leave=False):
                if config_name == "fp_singular_values":
                    continue 
                
                clean_config_name = config_name.replace("bit", "b")
                
                eff_rank_q = spectral_entropy_rank(config_metrics.get("quantized_singular_values", []))
                
                if eff_rank_fp > 0 and eff_rank_q > 0:
                    collapse_ratio = calc_collapse_ratio(eff_rank_fp, eff_rank_q)
                else:
                    collapse_ratio = None

                pa_cos = config_metrics.get("principal_angles_cos", {})
                rotations_by_k = {}
                
                for k_str, k_vals in pa_cos.items():
                    if k_vals:
                        clean_vals = [x for x in k_vals if x is not None]
                        if clean_vals:
                            angles_deg = [np.degrees(np.arccos(np.clip(c, -1.0, 1.0))) for c in clean_vals]
                            rotations_by_k[f"mean_rotation_deg_k{k_str}"] = safe_float(np.mean(angles_deg))
                            rotations_by_k[f"max_rotation_deg_k{k_str}"] = safe_float(np.max(angles_deg))
                            rotations_by_k[f"min_principal_cos_k{k_str}"] = safe_float(np.min(clean_vals))

                sds_by_k = config_metrics.get("sds", {})
                clean_sds = {k: safe_float(v) for k, v in sds_by_k.items() if v is not None}

                configurations_data[clean_config_name] = {
                    "singular_values_quantised": config_metrics.get("quantized_singular_values", []),
                    "effective_rank_fp": eff_rank_fp,
                    "effective_rank_q": eff_rank_q,
                    "low_variance_collapse_ratio": safe_float(collapse_ratio),
                    "principal_angles": pa_cos,
                    "rotations_by_k": rotations_by_k,
                    "sds_by_k": clean_sds
                }
            print(f"[Step 4] Extracted original singular values and {len(configurations_data)} quantization configurations.")
        else:
            print(f"[Warning] No spectral data found for {json_key} in JSON.")

        pt_save_path = f"jacobian_norms_{key}.pt"
        torch.save(jacobian_norms, pt_save_path)
        print(f"[Step 5] Saved {pt_save_path}")
        
        final_metrics[key] = {
            "sparsity_fragility": fragility_metrics,
            "spectral_evidence": {
                "singular_values_original": sv_orig,
                "configurations": configurations_data
            }
        }
        
        del sae, W_enc, jacobian_norms, fisher_scores
        gc.collect()

    json_save_path = "task4_part1_metrics.json"
    
    if not final_metrics:
        print(f"\n[Error] No bottleneck sizes were successfully processed. Metrics not saved.")
    else:
        with open(json_save_path, "w") as f:
            json.dump(final_metrics, f, indent=4)
        print(f"\n[Success] Successfully saved all structural metrics to {json_save_path}")

if __name__ == "__main__":
    main()