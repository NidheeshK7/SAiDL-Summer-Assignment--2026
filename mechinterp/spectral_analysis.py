import os
import json
import torch
import gc
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOTTLENECK_SIZES = [512, 1024]
METHODS = ["per_tensor", "per_feature"]
BITWIDTHS = [8, 6, 4, 2]
K_VALUES = [32, 64, 128]

def main():
    print(f"Starting Spectral Analysis using device: {DEVICE}")
    results_json = {}
    for d_sae in BOTTLENECK_SIZES:
        print(f"\n==============================================")
        print(f"Processing Bottleneck Size: {d_sae}")
        print(f"==============================================")
        
        results_json[str(d_sae)] = {}
        fp_path = f"bottleneck_activations_m{d_sae}_4096_fp.pt"
        
        if not os.path.exists(fp_path):
            print(f"[Warning] Full-precision file {fp_path} not found. Skipping {d_sae}.")
            continue

        Z = torch.load(fp_path, map_location=DEVICE, weights_only=True).float()
        print(f"[Diagnostic] Z shape:    {Z.shape}")
        print(f"[Diagnostic] Z sparsity: {(Z == 0).float().mean():.3f}")
        print(f"[Diagnostic] Z range:    [{Z.min():.4f}, {Z.max():.4f}]")

        Z_centered = Z - Z.mean(dim=0, keepdim=True)
        
        _, S_fp, Vh_fp = torch.linalg.svd(Z_centered, full_matrices=False)
        V_fp = Vh_fp.T
        
        torch.save(V_fp.cpu(), f"U_k_m{d_sae}.pt")
        print(f"Saved: U_k_m{d_sae}.pt")
        
        results_json[str(d_sae)]["fp_singular_values"] = S_fp.tolist()
        
        configs = [(m, b) for m in METHODS for b in BITWIDTHS]

        for method, bit in tqdm(configs, desc=f"Quantized Configs (m={d_sae})", unit="tensor"):
            q_path = f"bottleneck_activations_m{d_sae}_4096_{method}_{bit}bit.pt"
            
            if not os.path.exists(q_path):
                continue
                
            config_key = f"{method}_{bit}bit"
            results_json[str(d_sae)][config_key] = {
                "quantized_singular_values": [],
                "principal_angles_cos": {},
                "sds": {}
            }
            
            Z_hat = torch.load(q_path, map_location=DEVICE, weights_only=True).float()
            Z_hat_centered = Z_hat - Z_hat.mean(dim=0, keepdim=True)
            
            _, S_q, Vh_q = torch.linalg.svd(Z_hat_centered, full_matrices=False)
            V_q = Vh_q.T
            
            results_json[str(d_sae)][config_key]["quantized_singular_values"] = S_q.tolist()

            Z_diff = Z - Z_hat

            for k in K_VALUES:
                if k > d_sae:
                    continue
                    
                U_k_fp = V_fp[:, :k]
                U_k_q = V_q[:, :k]
                M = U_k_fp.T @ U_k_q
                S_angles = torch.linalg.svdvals(M)
                S_angles = torch.clamp(S_angles, 0.0, 1.0)
                
                results_json[str(d_sae)][config_key]["principal_angles_cos"][str(k)] = S_angles.tolist()
                
                numerator_mat = Z_diff @ U_k_fp
                denominator_mat = Z @ U_k_fp
                
                num = torch.linalg.norm(numerator_mat, ord='fro') ** 2
                den = torch.linalg.norm(denominator_mat, ord='fro') ** 2
                
                sds = (num / den).item() if den.item() > 0 else 0.0
                results_json[str(d_sae)][config_key]["sds"][str(k)] = sds
                
                del M, numerator_mat, denominator_mat, S_angles
                
            del Z_hat, Z_hat_centered, S_q, Vh_q, V_q, Z_diff
            gc.collect()
            torch.cuda.empty_cache()

        del Z, Z_centered, S_fp, Vh_fp, V_fp
        gc.collect()
        torch.cuda.empty_cache()

    output_file = "spectral_analysis_metrics.json"
    with open(output_file, "w") as f:
        json.dump(results_json, f, indent=4)
    print(f"\nSuccessfully wrote all spectral metrics to {output_file}")

if __name__ == "__main__":
    main()