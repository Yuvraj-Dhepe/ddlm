#!/usr/bin/env python3
"""
Hyperparameter sweep script using Optuna + WandB.

This script runs hyperparameter optimization for the Diffusion Language Model
using Optuna with Weights & Biases integration for tracking.
"""

import os
import shutil
import tempfile
from typing import Any, Dict, Optional

import click
import numpy as np
import optuna
import torch
import yaml
from accelerate import Accelerator
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
from generation.generate import chat_prompt, diffusion_generate
from models.model import DiffusionTransformerLM
from utils.diffusion_utils import diffusion_loss
from visualization.classic_render import create_classic_gif


def suggest_from_config(
    trial: optuna.Trial, param_name: str, param_config: Dict[str, Any]
) -> Any:
    """Suggest a hyperparameter value based on the YAML configuration."""
    ptype = param_config.get("type")
    if ptype == "float":
        return trial.suggest_float(
            param_name,
            param_config["low"],
            param_config["high"],
            log=param_config.get("log", False),
        )
    elif ptype == "int":
        return trial.suggest_int(
            param_name,
            param_config["low"],
            param_config["high"],
            log=param_config.get("log", False),
        )
    elif ptype == "categorical":
        return trial.suggest_categorical(param_name, param_config["choices"])
    else:
        raise ValueError(f"Unknown parameter type: {ptype}")


def train_model_with_config(
    cfg: Dict[str, Any],
    base_config: Dict[str, Any],
    is_final: bool = False,
) -> tuple[DiffusionTransformerLM, Any, Any, Any, float]:
    """
    Train a model with the given configuration.

    Returns:
        model, tokenizer, accelerator, final_cfg
    """
    # Only limit for quick mode trials - use full settings for budget_100
    # The base_config determines the training budget
    if not is_final:
        # Keep full budget settings - no limiting applied
        pass

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="bf16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "fp16"
    )

    # Load data
    train_ds, val_ds = load_datasets(
        cfg["TRAIN_EXAMPLES"], cfg["VAL_EXAMPLES"]
    )

    # Train tokenizer
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
    # Log training loss every N steps to avoid too much logging
    log_interval = max(
        1, cfg["TRAIN_STEPS"] // 100
    )  # ~100 log entries per training
    # Log validation loss every M steps (less frequent than training to save time)
    val_log_interval = max(
        1, cfg["TRAIN_STEPS"] // 20
    )  # ~20 val logs per training

    for step in tqdm(
        range(cfg["TRAIN_STEPS"]), disable=not accelerator.is_main_process
    ):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        loss = (
            diffusion_loss(
                model,
                batch,
                tokenizer,
                T=cfg["DIFFUSION_STEPS"],
                schedule=cfg.get("MASK_SCHEDULE", "linear"),
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

        # Log training loss every step (already computed, so cheap)
        wandb.log(
            {
                "train/loss": loss.item() * cfg["GRAD_ACCUM"],
                "train/lr": scheduler.get_last_lr()[0],
                "step": step + 1,
            }
        )

        # Log validation loss periodically to monitor overfitting
        if (step + 1) % val_log_interval == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for i, val_batch in enumerate(val_loader):
                    if i >= 5:  # Quick eval on 5 batches
                        break
                    val_loss = diffusion_loss(
                        model,
                        val_batch,
                        tokenizer,
                        T=cfg["DIFFUSION_STEPS"],
                        schedule=cfg.get("MASK_SCHEDULE", "linear"),
                    )
                    val_losses.append(val_loss.item())
            model.train()

            if val_losses:
                avg_val_loss = np.mean(val_losses)
                wandb.log(
                    {
                        "val/loss": avg_val_loss,
                        "step": step + 1,
                    }
                )

    # Evaluate on validation set
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 10:  # Evaluate on 10 batches
                break
            loss = diffusion_loss(
                model,
                batch,
                tokenizer,
                T=cfg["DIFFUSION_STEPS"],
                schedule=cfg.get("MASK_SCHEDULE", "linear"),
            )
            val_losses.append(loss.item())

    val_loss = np.mean(val_losses)

    return model, tokenizer, accelerator, cfg, val_loss


def run_generation_and_render(
    model: DiffusionTransformerLM,
    tokenizer,
    cfg: Dict[str, Any],
    accelerator: Accelerator,
    user_prompt: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run generation and save GIFs to WandB and locally."""
    model.eval()

    prompt_text = chat_prompt(user_prompt)

    final_text, frames = diffusion_generate(
        model=accelerator.unwrap_model(model),
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        seq_len=cfg["SEQ_LEN"],
        max_new_tokens=128,
        diffusion_steps=cfg["DIFFUSION_STEPS"],
        temperature=0.7,
        top_k=40,
        record_steps=True,
    )

    # Log generated text
    wandb.log({"generated_text": final_text})

    # Save GIFs to a temporary directory and log to WandB
    with tempfile.TemporaryDirectory() as tmpdir:
        # Classic GIF
        classic_path = os.path.join(tmpdir, "inference_classic.gif")
        create_classic_gif(
            frames, cfg["DIFFUSION_STEPS"], user_prompt, classic_path
        )

        # Log GIFs to WandB
        wandb.log(
            {
                "inference_gif": wandb.Image(classic_path),
            }
        )

        # Also save GIF locally if output_dir provided
        if output_dir:
            local_gif_path = os.path.join(output_dir, "inference.gif")
            create_classic_gif(
                frames, cfg["DIFFUSION_STEPS"], user_prompt, local_gif_path
            )

    model.train()
    return {"generated_text_length": len(final_text)}


def objective(
    trial: optuna.Trial,
    base_config: Dict[str, Any],
    sweep_config: Dict[str, Any],
    study_name: str,
    prompt: str,
) -> float:
    """
    Objective function for Optuna optimization.
    """
    # Suggest hyperparameters first
    suggested_params = {}
    for param_name, param_config in sweep_config.get("parameters", {}).items():
        suggested_params[param_name] = suggest_from_config(
            trial, param_name, param_config
        )

    # Start wandb run for this trial - let wandb generate unique name
    wandb.init(
        project=sweep_config.get("project", "ddlm-sweeps"),
        config={
            "trial_number": trial.number,
            **suggested_params,
        },
    )

    # Extract params - all values must come from the YAML file
    lr = suggested_params["learning_rate"]
    batch_size = suggested_params["batch_size"]
    n_layers = suggested_params["n_layers"]
    d_model = suggested_params["d_model"]
    n_heads = suggested_params["n_heads"]
    diffusion_steps = suggested_params["diffusion_steps"]
    mask_schedule = suggested_params.get("mask_schedule", "linear")

    # Ensure d_model is divisible by n_heads
    if d_model % n_heads != 0:
        wandb.finish()
        raise optuna.TrialPruned()

    # Build config
    cfg = base_config.copy()
    cfg["LR"] = lr
    cfg["BATCH_SIZE"] = batch_size
    cfg["N_LAYERS"] = n_layers
    cfg["N_HEADS"] = n_heads
    cfg["D_MODEL"] = d_model
    cfg["D_FF"] = 4 * d_model
    cfg["DIFFUSION_STEPS"] = diffusion_steps
    cfg["MASK_SCHEDULE"] = mask_schedule

    try:
        # Train model
        model, tokenizer, accelerator, final_cfg, val_loss = (
            train_model_with_config(cfg, base_config, is_final=False)
        )

        # Save model with unique run ID
        run_id = wandb.run.id
        output_dir = os.path.join("sweeps", study_name, f"trial_{run_id}")
        os.makedirs(output_dir, exist_ok=True)

        # Save model weights
        model_path = os.path.join(output_dir, "model.pt")
        torch.save(accelerator.unwrap_model(model).state_dict(), model_path)

        # Save tokenizer
        tokenizer_path = os.path.join(output_dir, "tokenizer.json")
        tokenizer_file = "tokenizer_from_scratch/tokenizer.json"
        if os.path.exists(tokenizer_file):
            shutil.copy(tokenizer_file, tokenizer_path)

        print(f"Saved trial {trial.number} model to {model_path}")
        print(f"Saved tokenizer to {tokenizer_path}")

        # Run generation and create GIF
        if prompt:
            print(f"Running generation for trial {trial.number}...")
            run_generation_and_render(
                model, tokenizer, final_cfg, accelerator, prompt, output_dir
            )

        # Log model path to wandb
        wandb.log(
            {
                "model_path": model_path,
                "tokenizer_path": tokenizer_path,
                "val_loss": val_loss,
            }
        )

        # Cleanup
        del model
        del tokenizer
        del accelerator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        wandb.finish()
        return val_loss

    except optuna.TrialPruned:
        wandb.finish()
        raise
    except Exception as e:
        print(f"Trial failed with error: {e}")
        wandb.finish()
        return float("inf")


@click.command()
@click.option(
    "--sweep-config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the YAML sweep configuration file",
)
@click.option(
    "--trials", type=int, required=True, help="Number of trials to run"
)
@click.option(
    "--run-mode",
    type=click.Choice(["quick", "budget_100"]),
    required=True,
    help="Base config mode",
)
@click.option(
    "--user-prompt",
    type=str,
    default=None,
    help="Prompt for generation during sweep (overrides YAML config)",
)
def run_sweep(
    sweep_config: str,
    trials: int,
    run_mode: str,
    user_prompt: str,
) -> None:
    """
    Run hyperparameter sweep using a YAML configuration file.
    """
    # Load sweep configuration
    with open(sweep_config, "r") as f:
        sweep_cfg = yaml.safe_load(f)

    # Validate required fields
    if "project" not in sweep_cfg:
        raise ValueError("YAML config must contain 'project' field")
    if "name" not in sweep_cfg:
        raise ValueError("YAML config must contain 'name' field")
    if "metric" not in sweep_cfg or "goal" not in sweep_cfg.get("metric", {}):
        raise ValueError("YAML config must contain 'metric.goal' field")
    if "parameters" not in sweep_cfg:
        raise ValueError("YAML config must contain 'parameters' field")

    project_name = sweep_cfg["project"]
    study_name = sweep_cfg["name"]

    # Get prompt from YAML or CLI
    prompt = user_prompt if user_prompt else sweep_cfg.get("prompt")
    if not prompt:
        raise ValueError(
            "YAML config must contain 'prompt' field or --user-prompt must be provided"
        )

    # Get base configuration
    base_config = get_training_config(run_mode)

    # Create Optuna study with GridSampler to ensure all combinations are tried
    # Build the search space from the YAML parameters
    search_space = {}
    for param_name, param_config in sweep_cfg.get("parameters", {}).items():
        if param_config.get("type") == "categorical":
            search_space[param_name] = param_config["choices"]

    # Use GridSampler to exhaustively try all combinations
    study = optuna.create_study(
        direction=sweep_cfg["metric"]["goal"],
        sampler=optuna.samplers.GridSampler(search_space),
        study_name=study_name,
    )

    # Run optimization (wandb is handled manually in each trial)
    study.optimize(
        lambda trial: objective(
            trial, base_config, sweep_cfg, study_name, prompt
        ),
        n_trials=trials,
    )

    # Print best results
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"\nAll trial models saved in: sweeps/{study_name}/trial_<run_id>/")


if __name__ == "__main__":
    run_sweep()
