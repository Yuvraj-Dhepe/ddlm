"""
Training module for the Diffusion Language Model.

This module contains the training loop and evaluation functions.
"""

import json
import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from models.model import DiffusionTransformerLM
from utils.diffusion_utils import diffusion_loss


def eval_loss(
    model: DiffusionTransformerLM,
    val_loader: DataLoader,
    tokenizer: PreTrainedTokenizerFast,
    diffusion_steps: int,
    accelerator: Accelerator,
    n_batches: int = 20,
    mask_schedule: str = "linear",
) -> float:
    """
    Evaluate validation loss.

    Args:
        model: The model.
        val_loader: Validation data loader.
        tokenizer: Tokenizer.
        diffusion_steps (int): Number of diffusion steps.
        accelerator: Accelerator instance.
        n_batches (int): Number of batches to evaluate.
        mask_schedule (str): Mask schedule type.

    Returns:
        float: Average loss.
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break

            loss = diffusion_loss(model, batch, tokenizer, T=diffusion_steps, schedule=mask_schedule)

            # gather across processes -> always make it 1D
            gathered = accelerator.gather(loss.detach().float().reshape(1))

            # now gathered is shape [world_size] (or [1] on single GPU)
            losses.append(gathered.cpu())

    model.train()

    if len(losses) == 0:
        return float("nan")

    losses = torch.cat(losses)  # safe: all are 1D tensors
    return losses.mean().item()


def train_model(
    model: DiffusionTransformerLM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: PreTrainedTokenizerFast,
    cfg: Dict[str, Any],
    accelerator: Accelerator,
    mask_schedule: str = "linear",
    wandb_run: Optional[Any] = None,
) -> None:
    """
    Train the diffusion model.

    Args:
        model: The model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        tokenizer: Tokenizer.
        cfg (dict): Training configuration.
        accelerator: Accelerator instance.
        mask_schedule (str): Mask schedule type - 'linear', 'cosine', 'quadratic',
                            'sqrt', 'inv_sqrt', 'sigmoid', 'warmup', 'constant', 'cosine_inv'
        wandb_run: Optional wandb run object for logging (if None and wandb available, will log locally)
    """
    # Determine if we should log to wandb
    log_to_wandb = WANDB_AVAILABLE and wandb_run is not None
    os.makedirs("outputs", exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"]
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["WARMUP_STEPS"],
        num_training_steps=cfg["TRAIN_STEPS"],
    )

    model, optimizer, train_loader, val_loader, scheduler = (
        accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )
    )

    model.train()
    pbar = tqdm(
        range(cfg["TRAIN_STEPS"]), disable=not accelerator.is_main_process
    )
    running = []
    
    if accelerator.is_main_process:
        print(f"\n=== Training with mask schedule: {mask_schedule} ===")

    train_iter = iter(train_loader)

    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        loss = (
            diffusion_loss(model, batch, tokenizer, T=cfg["DIFFUSION_STEPS"], schedule=mask_schedule)
            / cfg["GRAD_ACCUM"]
        )
        accelerator.backward(loss)

        if (step + 1) % cfg["GRAD_ACCUM"] == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running.append(loss.item() * cfg["GRAD_ACCUM"])

        # Log training loss every step (already computed, so cheap)
        if accelerator.is_main_process and log_to_wandb:
            current_loss = loss.item() * cfg["GRAD_ACCUM"]
            wandb.log({
                "train/loss": current_loss,
                "train/lr": scheduler.get_last_lr()[0],
                "step": step + 1,
            })
        
        if (step + 1) % 50 == 0 and accelerator.is_main_process:
            avg_loss = np.mean(running[-50:])
            pbar.set_description(
                f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
            )

        # Log validation loss more frequently to monitor overfitting
        if (step + 1) % 100 == 0 and accelerator.is_main_process:
            val_l = eval_loss(
                model,
                val_loader,
                tokenizer,
                cfg["DIFFUSION_STEPS"],
                accelerator,
                n_batches=10,
                mask_schedule=mask_schedule,
            )
            print(f"\nStep {step + 1} | val_loss ~ {val_l:.4f}")
            # Log validation loss to wandb if available
            if log_to_wandb:
                wandb.log({
                    "val/loss": val_l,
                    "step": step + 1,
                })

    if accelerator.is_main_process:
        OUT_DIR = "outputs"
        torch.save(
            accelerator.unwrap_model(model).state_dict(),
            os.path.join(OUT_DIR, "model.pt"),
        )
        # Include mask_schedule in saved config
        cfg_with_schedule = {**cfg, "MASK_SCHEDULE": mask_schedule}
        with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
            json.dump(cfg_with_schedule, f, indent=2)
        tokenizer.save_pretrained(os.path.join(OUT_DIR, "tokenizer"))
        print("Saved final checkpoint to:", OUT_DIR)

        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(running)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig("outputs/loss_plot.jpg", dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved loss plot to: outputs/loss_plot.jpg")
