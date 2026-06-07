import os
import torch
import wandb
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    TopKTrainingSAEConfig,
    LoggingConfig,
)

LAYER         = 2
D_IN          = 768
CONTEXT_SIZE  = 128
BATCH_SIZE    = 4096
TOTAL_TOKENS  = BATCH_SIZE * 100_000
LR            = 1e-4
TOPK_FRAC     = 0.10
WEIGHTS_DIR   = "weights"
WANDB_PROJECT = "mechinterp_sae"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(WEIGHTS_DIR, exist_ok=True)

for d_sae in [512, 1024]:
    k        = int(d_sae * TOPK_FRAC)
    run_name = f"sae_distilgpt2_l{LAYER}_{d_sae}"

    sae_cfg = TopKTrainingSAEConfig(
        d_in                  = D_IN,
        d_sae                 = d_sae,
        k                     = k,
        normalize_activations = "expected_average_only_in",
        apply_b_dec_to_input  = True,
        dtype                 = "float32",
        device                = DEVICE,
    )

    cfg = LanguageModelSAERunnerConfig(
        sae                      = sae_cfg,
        model_name               = "distilgpt2",
        hook_name                = f"blocks.{LAYER}.hook_resid_post",
        dataset_path             = "Skylion007/openwebtext",
        dataset_trust_remote_code= True,
        is_dataset_tokenized     = False,
        streaming                = True,
        context_size             = CONTEXT_SIZE,
        prepend_bos              = True,
        lr                       = LR,
        adam_beta1               = 0.9,
        adam_beta2               = 0.999,
        train_batch_size_tokens  = BATCH_SIZE,
        training_tokens          = TOTAL_TOKENS,
        n_batches_in_buffer      = 32,
        store_batch_size_prompts = 32,
        dtype                    = "float32",
        autocast_lm              = True,
        device                   = DEVICE,
        checkpoint_path          = os.path.join(WEIGHTS_DIR, run_name),
        n_checkpoints            = 5,
        logger                   = LoggingConfig(
            log_to_wandb  = True,
            wandb_project = WANDB_PROJECT,
            run_name      = run_name,
        ),
        seed                     = 42,
    )

    sae = LanguageModelSAETrainingRunner(cfg).run()

    save_path = os.path.join(WEIGHTS_DIR, f"{run_name}_final.pt")
    torch.save(sae.state_dict(), save_path)
    print(f"Saved: {save_path}")

    if wandb.run is not None:
        wandb.finish()