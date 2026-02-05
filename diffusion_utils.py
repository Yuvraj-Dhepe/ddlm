"""
Diffusion utilities module.

This module contains functions for masking, corruption, and loss calculation in the diffusion process.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from model import DiffusionTransformerLM


def mask_ratio_schedule(t: torch.Tensor, T: int) -> torch.Tensor:
    """
    Linear masking ratio schedule.

    Args:
        t: Timestep tensor.
        T (int): Total diffusion steps.

    Returns:
        Tensor: Masking ratio.
    """
    # Linear schedule: ratio = t/T
    return t.float() / float(T)


def corrupt_with_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    t: torch.Tensor,
    tokenizer: PreTrainedTokenizerFast,
    T: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Corrupt input by masking tokens based on timestep.

    Args:
        input_ids: Input token IDs.
        attention_mask: Attention mask.
        t: Timestep.
        tokenizer: Tokenizer with special token IDs.
        T (int): Total steps.

    Returns:
        tuple: (noisy_ids, labels, mask_positions)
    """
    # Returns noisy_ids, labels, mask_positions
    B, L = input_ids.shape
    ratio = mask_ratio_schedule(t, T).unsqueeze(1)  # [B,1]

    can_mask = attention_mask.clone()
    can_mask &= (
        (input_ids != tokenizer.bos_token_id)
        & (input_ids != tokenizer.eos_token_id)
        & (input_ids != tokenizer.pad_token_id)
    )

    rand = torch.rand((B, L), device=input_ids.device)
    mask_positions = (rand < ratio) & can_mask

    noisy = input_ids.clone()
    noisy[mask_positions] = tokenizer.mask_token_id

    labels = torch.full_like(input_ids, -100)
    labels[mask_positions] = input_ids[mask_positions]

    return noisy, labels, mask_positions


def diffusion_loss(
    model: DiffusionTransformerLM,
    batch: Dict[str, torch.Tensor],
    tokenizer: PreTrainedTokenizerFast,
    T: int,
) -> torch.Tensor:
    """
    Compute diffusion loss.

    Args:
        model: The diffusion model.
        batch: Batch data.
        tokenizer: Tokenizer.
        T (int): Total diffusion steps.

    Returns:
        Tensor: Loss.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    B = input_ids.size(0)
    t = torch.randint(1, T + 1, (B,), device=input_ids.device)

    noisy_ids, labels, _ = corrupt_with_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        t=t,
        tokenizer=tokenizer,
        T=T,
    )

    logits = model(noisy_ids, timesteps=t, attention_mask=attention_mask)  # [B,L,V]
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    return loss
