import torch
import math
import gc
from pathlib import Path
from tqdm import tqdm
import wandb
from config import get_config, get_weights_file_path
from model import GPTModel
from data_setup import get_wikitext2_dataloaders
from train import calc_loss_batch

def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()

def evaluate_extrapolation(model, val_loader, device, eval_iter=100):
    model.eval()
    autocast_context = torch.autocast(device_type='cuda', dtype=torch.float16)
    
    val_loss = 0.
    steps = 0
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(tqdm(val_loader, desc="Evaluating", leave=False)):
            if i >= eval_iter: 
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device, autocast_context)
            val_loss += loss.item()
            steps += 1
            
    return val_loss / steps if steps > 0 else float('inf')

def run_extrapolation_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Extrapolation Evaluation on {device}...")
    
    position_types = ['rope', 'alibi', 'rel_pos']
    test_lengths = [512, 1024, 2048]
    results = {pos: {} for pos in position_types}
    losses  = {pos: {} for pos in position_types}
    
    for pos in position_types:
        print(f"\n{'='*60}\nTesting Positional Variant: {pos.upper()}\n{'='*60}")

        run_name = f"part3_pos_mha_{pos}_len512_v4"
        cfg = get_config(is_dry_run=False)
        cfg.update({
            "run_name": run_name,
            "attn_type": "mha",
            "pos_type": pos,
            "train_context_length": 512
        })

        weights_path = get_weights_file_path(cfg, "best") 

        if not Path(weights_path).exists():
            print(f"[WARNING] Checkpoint not found for {pos.upper()} at {weights_path}. Skipping.")
            continue

        print(f"Loading weights from {weights_path}...")
        model = GPTModel(cfg).to(device)

        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        for test_len in test_lengths:
            print(f"\n--- Evaluating Context Length: {test_len} ---")

            fixed_batch_size = cfg["batch_size"]
            print(f"Using fixed Batch Size of {fixed_batch_size} for consistent evaluation.")

            _, val_loader = get_wikitext2_dataloaders(
                batch_size=fixed_batch_size, 
                max_length=test_len, 
                stride=test_len
            )

            val_loss = evaluate_extrapolation(model, val_loader, device, eval_iter=cfg["eval_iter"])
            ppl = math.exp(val_loss) if val_loss < 700 else float('inf')
            results[pos][test_len] = ppl
            losses[pos][test_len]  = val_loss
            print(f"{pos.upper()} @ L={test_len} -> Validation Perplexity: {ppl:.2f}")

            clear_vram()

        del model
        clear_vram()

    cfg_for_wandb = get_config(is_dry_run=False)
    wandb.init(
        project=cfg_for_wandb["wandb_project"],
        name="part3_extrapolation_summary",
        config={
            "train_context_length": 512,
            "test_lengths": test_lengths,
            "position_types": position_types,
            "eval_batch_size": cfg_for_wandb["batch_size"]
        }
    )

    wandb.define_metric("test_context_length")
    wandb.define_metric("extrap/*", step_metric="test_context_length")

    table = wandb.Table(columns=["pos_type", "test_length", "val_perplexity"])

    for test_len in test_lengths:
        log_dict = {"test_context_length": test_len}
        for pos in position_types:
            if test_len in results[pos]:
                ppl_val = results[pos][test_len]
                log_dict[f"extrap/{pos}/val_perplexity"] = ppl_val
                log_dict[f"extrap/{pos}/val_loss"] = losses[pos][test_len]
                table.add_data(pos, test_len, ppl_val)
        wandb.log(log_dict, step=test_len)

    wandb.log({"extrapolation_results_table": table})
    wandb.finish()

    print("\n" + "="*60)
    print(" EXTRAPOLATION TEST RESULTS (VALIDATION PERPLEXITY)")
    print("="*60)

    header = f"{'Positional Model':<18} | " + " | ".join([f"L_{l:<5}" for l in test_lengths])
    print(header)
    print("-" * len(header))

    for pos in position_types:
        if not results[pos]:
            continue
        row = f"{pos.upper():<18} | "
        vals = []
        for l in test_lengths:
            v = results[pos].get(l, None)
            vals.append(f"{v:<7.2f}" if v is not None else f"{'N/A':<7}")
        row += " | ".join(vals)
        print(row)

    print("="*60 + "\n")

if __name__ == "__main__":
    run_extrapolation_test()