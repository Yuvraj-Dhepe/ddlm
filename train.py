"""
Training module for the Diffusion Language Model.

This module contains the training loop and evaluation functions.
"""

import os
import json
from typing import Dict, Any
import numpy as np
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
from transformers import PreTrainedTokenizerFast, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from diffusion_utils import diffusion_loss
from model import DiffusionTransformerLM


def eval_loss(model: DiffusionTransformerLM, val_loader: DataLoader, tokenizer: PreTrainedTokenizerFast, diffusion_steps: int, accelerator: Accelerator, n_batches: int = 20) -> float:
    """
    Evaluate validation loss.

    Args:
        model: The model.
        val_loader: Validation data loader.
        tokenizer: Tokenizer.
        diffusion_steps (int): Number of diffusion steps.
        accelerator: Accelerator instance.
        n_batches (int): Number of batches to evaluate.

    Returns:
        float: Average loss.
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break

            loss = diffusion_loss(model, batch, tokenizer, T=diffusion_steps)

            # gather across processes -> always make it 1D
            gathered = accelerator.gather(loss.detach().float().reshape(1))

            # now gathered is shape [world_size] (or [1] on single GPU)
            losses.append(gathered.cpu())

    model.train()

    if len(losses) == 0:
        return float("nan")

    losses = torch.cat(losses)   # safe: all are 1D tensors
    return losses.mean().item()


def train_model(model: DiffusionTransformerLM, train_loader: DataLoader, val_loader: DataLoader, tokenizer: PreTrainedTokenizerFast, cfg: Dict[str, Any], accelerator: Accelerator) -> None:
    """
    Train the diffusion model.

    Args:
        model: The model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        tokenizer: Tokenizer.
        cfg (dict): Training configuration.
        accelerator: Accelerator instance.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['LR'], weight_decay=cfg['WEIGHT_DECAY'])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg['WARMUP_STEPS'],
        num_training_steps=cfg['TRAIN_STEPS'],
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    model.train()
    pbar = tqdm(range(cfg['TRAIN_STEPS']), disable=not accelerator.is_main_process)
    running = []

    train_iter = iter(train_loader)

    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        loss = diffusion_loss(model, batch, tokenizer, T=cfg['DIFFUSION_STEPS']) / cfg['GRAD_ACCUM']
        accelerator.backward(loss)

        if (step + 1) % cfg['GRAD_ACCUM'] == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running.append(loss.item() * cfg['GRAD_ACCUM'])

        if (step + 1) % 50 == 0 and accelerator.is_main_process:
            pbar.set_description(f"loss={np.mean(running[-50:]):.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        if (step + 1) % 500 == 0 and accelerator.is_main_process:
            val_l = eval_loss(model, val_loader, tokenizer, cfg['DIFFUSION_STEPS'], accelerator, n_batches=10)
            print(f"\nStep {step+1} | val_loss ~ {val_l:.4f}")

    if accelerator.is_main_process:
        OUT_DIR = "checkpoints/final"
        os.makedirs(OUT_DIR, exist_ok=True)
        torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(OUT_DIR, "model.pt"))
        with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        tokenizer.save_pretrained(os.path.join(OUT_DIR, "tokenizer"))
        print("Saved final checkpoint to:", OUT_DIR)