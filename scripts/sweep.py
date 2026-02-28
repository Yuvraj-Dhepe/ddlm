#!/usr/bin/env python3
"""
Hyperparameter sweep script using Optuna + WandB.

This script runs hyperparameter optimization for the Diffusion Language Model
using Optuna with Weights & Biases integration for tracking.
"""

from typing import Any, Dict

import numpy as np
import optuna
import torch
from accelerate import Accelerator
from optuna.integration import WeightsAndBiasesCallback
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

import wandb

# Import from package
from config.config import DiffusionLMConfig, get_training_config
from data.data import (
    create_data_loaders,
    create_hf_tokenizer,
    load_datasets,
    train_tokenizer_from_scratch,
)
from models.model import DiffusionTransformerLM
from utils.diffusion_utils import diffusion_loss


def objective(trial: optuna.Trial, base_config: Dict[str, Any]) -> float:
    """
    Objective function for Optuna optimization.

    Args:
        trial: Optuna trial object
        base_config: Base configuration dictionary

    Returns:
        Validation loss to minimize
    """
    # Suggest hyperparameters to optimize
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    n_layers = trial.suggest_int("n_layers", 4, 12)
    d_model = trial.suggest_categorical("d_model", [256, 384, 512])
    # Ensure d_model is divisible by n_heads - use only valid divisors
    # 256: 4, 8 are valid (256 % 6 = 4)
    # 384: 4, 6, 8 are all valid
    # 512: 4, 8 are valid (512 % 6 = 2)
    if d_model == 256:
        n_heads = trial.suggest_categorical("n_heads", [4, 8])
    elif d_model == 384:
        n_heads = trial.suggest_categorical("n_heads", [4, 6, 8])
    else:  # d_model == 512
        n_heads = trial.suggest_categorical("n_heads", [4, 8])
    diffusion_steps = trial.suggest_int("diffusion_steps", 32, 128)

    # Override base config with suggested values
    cfg = base_config.copy()
    cfg["LR"] = lr
    cfg["BATCH_SIZE"] = batch_size
    cfg["N_LAYERS"] = n_layers
    cfg["N_HEADS"] = n_heads
    cfg["D_MODEL"] = d_model
    cfg["D_FF"] = 4 * d_model
    cfg["DIFFUSION_STEPS"] = diffusion_steps

    # Use a subset for faster sweeps
    cfg["TRAIN_EXAMPLES"] = min(cfg["TRAIN_EXAMPLES"], 10000)
    cfg["VAL_EXAMPLES"] = min(cfg["VAL_EXAMPLES"], 1000)
    cfg["TRAIN_STEPS"] = min(cfg["TRAIN_STEPS"], 500)

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="bf16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "fp16"
    )

    # Log trial params to wandb
    wandb.config.update(
        {
            "learning_rate": lr,
            "batch_size": batch_size,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_model": d_model,
            "diffusion_steps": diffusion_steps,
        },
        allow_val_change=True,
    )

    model = None
    optimizer = None
    train_loader = None
    val_loader = None

    try:
        # Load data
        train_ds, val_ds = load_datasets(
            cfg["TRAIN_EXAMPLES"], cfg["VAL_EXAMPLES"]
        )

        # Train tokenizer (use cached if available)
        tokenizer_file = train_tokenizer_from_scratch(
            train_ds, cfg["VOCAB_SIZE"], cfg["TOKENIZER_TRAIN_EXAMPLES"]
        )
        tokenizer = create_hf_tokenizer(tokenizer_file)

        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_ds, val_ds, tokenizer, cfg["SEQ_LEN"], cfg["BATCH_SIZE"]
        )

        # Create model
        model_cfg = DiffusionLMConfig(
            vocab_size=len(tokenizer),
            seq_len=cfg["SEQ_LEN"],
            d_model=cfg["D_MODEL"],
            n_layers=cfg["N_LAYERS"],
            n_heads=cfg["N_HEADS"],
            d_ff=cfg["D_FF"],
            dropout=0.1,
            diffusion_steps=cfg["DIFFUSION_STEPS"],
        )
        model = DiffusionTransformerLM(model_cfg)

        # Training setup
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"]
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg["WARMUP_STEPS"],
            num_training_steps=cfg["TRAIN_STEPS"],
        )

        # Prepare with accelerator
        model, optimizer, train_loader, val_loader, scheduler = (
            accelerator.prepare(
                model, optimizer, train_loader, val_loader, scheduler
            )
        )

        model.train()
        train_iter = iter(train_loader)

        # Training loop
        total_loss = 0.0
        for step in tqdm(range(cfg["TRAIN_STEPS"]), disable=True):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            loss = (
                diffusion_loss(
                    model, batch, tokenizer, T=cfg["DIFFUSION_STEPS"]
                )
                / cfg["GRAD_ACCUM"]
            )
            accelerator.backward(loss)

            if (step + 1) % cfg["GRAD_ACCUM"] == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * cfg["GRAD_ACCUM"]

            # Log to wandb
            if (step + 1) % 50 == 0:
                avg_loss = total_loss / (step + 1)
                wandb.log({"train/loss": avg_loss, "step": step})

        # Evaluate on validation set
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 10:  # Evaluate on 10 batches
                    break
                loss = diffusion_loss(
                    model, batch, tokenizer, T=cfg["DIFFUSION_STEPS"]
                )
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        wandb.log({"val/loss": val_loss})

        return val_loss

    except Exception as e:
        print(f"Trial failed with error: {e}")
        wandb.log({"error": str(e)})
        return float("inf")
    finally:
        # Cleanup (only if variables were assigned)
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        if train_loader is not None:
            del train_loader
        if val_loader is not None:
            del val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_sweep(
    project_name: str = "ddlm-sweeps",
    n_trials: int = 5,
    run_mode: str = "quick",
) -> None:
    """
    Run hyperparameter sweep.

    Args:
        project_name: WandB project name
        n_trials: Number of trials to run
        run_mode: Base config mode (quick or budget_100)
    """
    # Get base configuration
    base_config = get_training_config(run_mode)

    # Initialize wandb for optuna
    wandb.init(project=project_name, job_type="sweep")

    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        study_name=f"ddlm-{run_mode}-sweep",
    )

    # Add wandb callback
    wandb_callback = WeightsAndBiasesCallback(
        wandb_kwargs={"project": project_name}
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_config),
        n_trials=n_trials,
        callbacks=[wandb_callback],
    )

    # Print best results
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Log best params
    wandb.log(
        {"best_val_loss": study.best_value, "best_params": study.best_params}
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run hyperparameter sweep")
    parser.add_argument(
        "--project", type=str, default="ddlm-sweeps", help="WandB project name"
    )
    parser.add_argument(
        "--trials", type=int, default=20, help="Number of trials"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "budget_100"],
        help="Base config mode",
    )

    args = parser.parse_args()

    run_sweep(
        project_name=args.project, n_trials=args.trials, run_mode=args.mode
    )
