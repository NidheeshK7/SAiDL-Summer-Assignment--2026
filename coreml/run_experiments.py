import torch
import gc
import traceback
import wandb
from train import main

def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()

def run_sweep():
    experiments = []
    # Section 1: Establish the Baseline
    experiments.append({
        "run_name": "part1_baseline_mha_sinusoidal_len1024_v4",
        "attn_type": "mha",
        "pos_type": "sinusoidal",
        "train_context_length": 1024
    })
    # Section 2: Attention Variants
    attention_types = ['gqa', 'swa', 'linear']
    context_lengths = [512, 1024, 2048]
    for attn in attention_types:
        for seq_len in context_lengths:
            run_name = f"part2_attn_{attn}_sinusoidal_len{seq_len}_v4"
            
            exp_cfg = {
                "run_name": run_name,
                "attn_type": attn,
                "pos_type": "sinusoidal",
                "train_context_length": seq_len
            }
            if seq_len >= 2048:
                exp_cfg["batch_size"] = 1
                exp_cfg["accumulation_steps"] = 8
                
            experiments.append(exp_cfg)

    # Section 3: Positional Variants (Training Phase)
    position_types = ['rope', 'alibi', 'rel_pos']
    extrapolation_train_length = 512
    
    for pos in position_types:
        run_name = f"part3_pos_mha_{pos}_len{extrapolation_train_length}_v4"
        experiments.append({
            "run_name": run_name,
            "attn_type": "mha",
            "pos_type": pos,
            "train_context_length": extrapolation_train_length
        })
        
    # Section 4: Convolution + Attention Hybrids   
    BEST_ATTN = "swa"          
    BEST_POS  = "rel_pos"         
 
    hybrid_types    = ["interleaved", "gated_ffn"]
    hybrid_contexts = [2048]
    kernel_sizes    = [7]     
 
    for hybrid_type in hybrid_types:
        for seq_len in hybrid_contexts:
            for kernel_size in kernel_sizes:
                run_name = (
                    f"part4_hybrid_{hybrid_type}_{BEST_ATTN}_{BEST_POS}"
                    f"_k{kernel_size}_len{seq_len}"
                )
                experiments.append({
                    "run_name":        run_name,
                    "attn_type":       BEST_ATTN,
                    "pos_type":        BEST_POS,
                    "hybrid_type":     hybrid_type,
                    "conv_kernel_size": kernel_size,
                    "train_context_length": seq_len,
                    "use_hybrid":      True,    
                })
    # Section 5: Bonus - Attention Free Transformer (AFT) Variants
    aft_variants = ['aft_simple', 'aft_full', 'aft_local', 'aft_conv']
    
    seq_len  = 2048       

    for aft in aft_variants:
        run_name = f"bonus_{aft}_sinusoidal_len{seq_len}"
        experiments.append({
            "run_name": run_name,
            "attn_type": aft,
            "pos_type": "sinusoidal", 
            "train_context_length": seq_len,
            "use_hybrid": False,   
            "aft_bias_dim": 128,
            "aft_local_window": 32,
            "aft_conv_kernel": 16
        })

    total_experiments = len(experiments)
    
    print(f"Starting Master Sweep. Total configurations to test: {total_experiments}")
    print("=" * 60)

    for idx, exp_cfg in enumerate(experiments):
        print(f"\n[{idx + 1}/{total_experiments}] Launching Experiment: {exp_cfg['run_name']}")
        
        try:
            clear_vram() 
            main(is_dry_run=False, override_cfg=exp_cfg)
            print(f"[SUCCESS] Finished {exp_cfg['run_name']}")

        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                print(f"[OOM ERROR] 8GB VRAM limit exceeded for {exp_cfg['run_name']}.")
                print("Gracefully skipping to the next configuration...")
                
                if wandb.run is not None:
                    wandb.finish(exit_code=1)
            else:
                print(f"[FATAL ERROR] Non-OOM exception occurred in {exp_cfg['run_name']}:")
                traceback.print_exc()
                
                if wandb.run is not None:
                    wandb.finish(exit_code=1)
                break 

if __name__ == "__main__":
    run_sweep()