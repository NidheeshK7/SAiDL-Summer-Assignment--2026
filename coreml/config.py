import json
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

def get_config(is_dry_run=False):
    
    config = {
        "vocab_size": 50257,    
        "emb_dim": 768,         
        "n_heads": 12,          
        "n_layers": 6,          
        "drop_rate": 0.1,       
        "qkv_bias": False,

        "attn_type": "mha",     
        "hybrid_type": "interleaved",
        "conv_kernel_size": 7,
        
        "aft_bias_dim": 128,          
        "aft_local_window": 32,      
        "aft_conv_kernel": 16,        
        
        "pos_type": "sinusoidal",   

        "num_kv_groups": 4,         
        "sliding_window_size": 256,  
        "rel_clip": 16,             

        "train_context_length": 512, 
        "eval_context_lengths": [512, 1024, 2048, 4096], 
        
        "learning_rate": 5e-4,
        "min_lr": 1e-6,
        "warmup_ratio": 0.1,  
        "weight_decay": 0.01,
        "batch_size": 2,       
        "accumulation_steps": 4,
        "model_folder": str(WEIGHTS_DIR),
        "run_name": "baseline_mha_standard_pos", 
        "use_wandb": True,
        "wandb_project": "long-context-coreml",
    }
    if is_dry_run:
        config.update({
            "num_epochs": 1,
            "batch_size": 2,       
            "eval_freq": 5,         
            "eval_iter": 2          
        })
    else:
        config.update({
            "num_epochs": 5,        
            "eval_checkpoints_per_run": 50,      
            "eval_iter": 20          
        })

    return config

def get_weights_file_path(config, epoch: str, prefix="best"):
    model_filename = f"{config['run_name']}_{prefix}_{epoch}.pt"
    return str(Path(config['model_folder']) / model_filename)