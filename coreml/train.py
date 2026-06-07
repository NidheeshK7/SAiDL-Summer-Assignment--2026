import time
import math
import os
import torch
import wandb
from pathlib import Path
from tqdm import tqdm
from config import get_config, get_weights_file_path
from model import GPTModel, generate_text_simple_cached
from data_setup import get_wikitext2_dataloaders
from HybridModel import HybridGPTModel

def set_seed(seed=123):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calc_loss_batch(input_batch, target_batch, model, device, autocast_context):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    with autocast_context:
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def measure_inference_performance(model, device, context_size, batch_size, num_tokens_to_generate=50, use_cache = True):
    model.eval()

    dummy_input = torch.randint(0, 50257, (batch_size, context_size)).to(device)
    
    with torch.no_grad():
        _ = generate_text_simple_cached(model, dummy_input, max_new_tokens=5, context_size=context_size, use_cache=use_cache)
        
    num_runs = 10
    runtimes = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize(device)
            start_time = time.time()
            
            _ = generate_text_simple_cached(model, dummy_input, max_new_tokens=num_tokens_to_generate, context_size=context_size, use_cache=use_cache)
            
            torch.cuda.synchronize(device)
            end_time = time.time()
            runtimes.append(end_time - start_time)
            
    avg_time = sum(runtimes) / len(runtimes)
    total_tokens_generated = batch_size * num_tokens_to_generate
    
    throughput_tok_sec = total_tokens_generated / avg_time
    
    latency_sec_tok = avg_time / num_tokens_to_generate
    
    model.train()
    return latency_sec_tok, throughput_tok_sec

def evaluate_model(model, train_loader, val_loader, device, eval_iter, autocast_context, cfg):
    model.eval()
    config_prefix = f"[{cfg['attn_type'].upper()}-{cfg['pos_type'].upper()}]"
    context_str = f"L_test={cfg['train_context_length']}"
    
    with torch.no_grad():
        train_loss = 0.
        num_train_batches = 0
        for i, (input_batch, target_batch) in enumerate(train_loader):
            if i >= eval_iter: break
            loss = calc_loss_batch(input_batch, target_batch, model, device, autocast_context)
            train_loss += loss.item()
            num_train_batches += 1
        train_loss /= max(1, num_train_batches)

        val_loss = 0.
        num_val_batches = 0
        val_pbar = tqdm(val_loader, desc=f"{config_prefix} {context_str} | Validating", leave=False)
        for i, (input_batch, target_batch) in enumerate(val_pbar):
            if i >= eval_iter: break
            loss = calc_loss_batch(input_batch, target_batch, model, device, autocast_context)
            val_loss += loss.item()
            num_val_batches += 1
            max_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            val_pbar.set_postfix({
                "Val Loss": f"{(val_loss / (i + 1)):.4f}", 
                "Mem(GB)": f"{max_mem_gb:.2f}"
            })
            
        val_loss /= max(1, num_val_batches)
        
    model.train()
    return train_loss, val_loss

def train_model(model, train_loader, val_loader, optimizer, scaler, device, cfg, start_epoch=0, global_step=0, best_val_loss=float('inf')):
    autocast_context = torch.autocast(device_type='cuda', dtype=torch.float16)
    
    peak_lr = cfg.get("learning_rate", 5e-4)
    min_lr = cfg.get("min_lr", 1e-6)
    accumulation_steps = cfg.get("accumulation_steps", 1)
    total_training_steps = (len(train_loader) // accumulation_steps) * cfg["num_epochs"]
    eval_checkpoints = cfg.get("eval_checkpoints_per_run", 10)
    eval_freq = max(1, total_training_steps // eval_checkpoints)
    warmup_ratio = cfg.get("warmup_ratio", 0.1)
    warmup_steps = int(warmup_ratio * total_training_steps)
    lr_increment = (peak_lr - min_lr) / max(1, warmup_steps)
    
    print("Starting training loop...")
    for epoch in range(start_epoch, cfg["num_epochs"]):
        model.train()
        epoch_start_time = time.time()
        
        config_prefix = f"[{cfg['attn_type'].upper()}-{cfg['pos_type'].upper()}]"
        context_str = f"L_train={cfg['train_context_length']}"
        pbar = tqdm(train_loader, desc=f"{config_prefix} {context_str} | Epoch {epoch+1}")
        
        optimizer.zero_grad()
        
        for i, (input_batch, target_batch) in enumerate(pbar):
            batch_start_time = time.time()

            if global_step < warmup_steps:
                lr = min_lr + global_step * lr_increment  
            else:
                progress = ((global_step - warmup_steps) / max(1, (total_training_steps - warmup_steps)))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            loss = calc_loss_batch(input_batch, target_batch, model, device, autocast_context)
            scaled_loss = loss / accumulation_steps
            
            scaler.scale(scaled_loss).backward()
            
            is_accumulation_boundary = (i + 1) % accumulation_steps == 0
            is_last_batch = (i + 1) == len(train_loader)
            
            step_occurred = False
            
            if is_accumulation_boundary or is_last_batch:
                torch.cuda.synchronize(device)
                batch_time = time.time() - batch_start_time
                scaler.unscale_(optimizer)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                global_step += 1 
                step_occurred = True

            if step_occurred:
                torch.cuda.synchronize(device)
                batch_time = time.time() - batch_start_time
                throughput = (input_batch.numel() * accumulation_steps) / batch_time
            else:
                batch_time = 0.0
                throughput = 0.0
            max_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Mem(GB)": f"{max_mem_gb:.2f}",
                "Tok/s": f"{throughput:.0f}"
            })
            
            log_freq = max(1, total_training_steps // 100)

            if step_occurred and cfg["use_wandb"] and global_step % log_freq == 0:
                wandb.log({
                    "train/step_loss": loss.item(),
                    "train/learning_rate": lr,
                    "train/grad_norm": grad_norm.item(),
                    "perf/training_throughput_tok_sec": throughput,
                    "perf/peak_gpu_mem_MB": max_mem_gb * 1024,
                    "global_step": global_step
                })

            if step_occurred and global_step % eval_freq == 0:
                pbar.set_description(f"{config_prefix} {context_str} | Epoch {epoch+1} [Evaluating...]")
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, cfg["eval_iter"], autocast_context, cfg
                )
                
                val_ppl = math.exp(val_loss)
                train_ppl = math.exp(train_loss)
               

                pbar.set_description(f"{config_prefix} {context_str} | Epoch {epoch+1}")
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Val Loss": f"{val_loss:.4f}",
                    "Val PPL": f"{val_ppl:.2f}",
                    "Mem(GB)": f"{max_mem_gb:.2f}",
                })
                tqdm.write(f"Ep {epoch+1} | Step {global_step:05d} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | LR: {lr:.2e}")
                
                if cfg["use_wandb"]:
                    wandb.log({
                        "eval/train_loss": train_loss,
                        "eval/val_loss": val_loss,
                        "eval/train_perplexity": train_ppl,
                        "eval/val_perplexity": val_ppl,
                        "global_step": global_step
                    })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = get_weights_file_path(cfg, "best")
                    
                    checkpoint = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(), 
                        'best_val_loss': best_val_loss,
                        'wandb_id': wandb.run.id if cfg["use_wandb"] else None
                    }
                    torch.save(checkpoint, save_path)
                    tqdm.write(f"--> Saved new best model to {save_path}")

        epoch_duration = time.time() - epoch_start_time
        if cfg["use_wandb"]:
            wandb.log({"perf/epoch_duration_sec": epoch_duration, "epoch": epoch + 1})
            
    return model

def main(is_dry_run=False, resume_checkpoint_path=None, override_cfg=None):
    set_seed(123)
    cfg = get_config(is_dry_run=is_dry_run)
    
    if override_cfg is not None:
        cfg.update(override_cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Initializing {cfg['attn_type'].upper()} Model with {cfg['pos_type'].upper()} embeddings...")
    
    use_hybrid = cfg.get("hybrid_type") is not None and cfg.get("use_hybrid", False)
    if use_hybrid:
        model = HybridGPTModel(cfg).to(device)
        print(f"  → Using HybridGPTModel with hybrid_type='{cfg['hybrid_type']}'")
    else:
        model = GPTModel(cfg).to(device)

    
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version available: {torch.version.cuda}")
    else:
        print("Running on CPU")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.get("learning_rate", 5e-4), 
        weight_decay=cfg.get("weight_decay", 0.1)
    )
    scaler = torch.amp.GradScaler('cuda')
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    run_id = wandb.util.generate_id()

    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        print(f"Loading checkpoint from {resume_checkpoint_path}...")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:            
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        best_val_loss = checkpoint['best_val_loss']
        run_id = checkpoint.get('wandb_id', run_id)
        print(f"Resumed successfully. Starting from epoch {start_epoch+1}, step {global_step}")

    if cfg["use_wandb"]:
        wandb.init(
            project=cfg["wandb_project"],
            name=cfg["run_name"],
            config=cfg,
            id=run_id,
            resume="allow" 
        )

    train_loader, val_loader = get_wikitext2_dataloaders(
        batch_size=cfg["batch_size"],
        max_length=cfg["train_context_length"],
        stride=cfg["train_context_length"]
    )

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        cfg=cfg,
        start_epoch=start_epoch,
        global_step=global_step,
        best_val_loss=best_val_loss
    )
    
    print("Measuring final inference performance...")
    is_aft = cfg.get("attn_type", "").startswith("aft")
    should_use_cache = not (use_hybrid or is_aft)
    
    inf_latency, inf_throughput = measure_inference_performance(
        model, device, cfg["train_context_length"], cfg["batch_size"], use_cache=should_use_cache
    )
    print(f"Inference Latency: {inf_latency:.4f} s/tok | Throughput: {inf_throughput:.1f} tok/s")
    if cfg["use_wandb"]:
        wandb.log({
            "perf/final_inference_latency_sec_per_tok": inf_latency,
            "perf/final_inference_throughput_tok_sec": inf_throughput,
        })

    if cfg["use_wandb"]:
        wandb.finish()

if __name__ == "__main__":
    main(is_dry_run=False, resume_checkpoint_path=None)