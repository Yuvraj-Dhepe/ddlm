"""
Diffusion utilities module.

This module contains functions for masking, corruption, and loss calculation in the diffusion process.
"""

from enum import Enum
from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from models.model import DiffusionTransformerLM


class MaskScheduleType(Enum):
    """Available masking schedule types."""

    LINEAR = "linear"  # ratio = t/T (current default)
    COSINE = "cosine"  # ratio = 0.5 * (1 + cos(pi * t/T))
    QUADRATIC = "quadratic"  # ratio = (t/T)^2
    SQRT = "sqrt"  # ratio = sqrt(t/T)
    INV_SQRT = "inv_sqrt"  # ratio = 1 - sqrt(1 - t/T)
    SIGMOID = "sigmoid"  # ratio = 1 / (1 + exp(-k * (t/T - 0.5)))
    WARMUP = "warmup"  # ratio = (t/T)^3 (slow start, faster end)
    CONSTANT = "constant"  # Fixed ratio throughout
    COSINE_INV = (
        "cosine_inv"  # ratio = 0.5 * (1 - cos(pi * t/T)) - more early masking
    )


def get_mask_schedule_fn(
    schedule_type: str,
) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """
    Get the masking schedule function based on schedule type.

    Args:
        schedule_type: One of 'linear', 'cosine', 'quadratic', 'sqrt',
                       'inv_sqrt', 'sigmoid', 'warmup', 'constant', 'cosine_inv'

    Returns:
        A function that computes mask ratio from timestep t and total steps T

    Note on schedules:
    - LINEAR: Linear increase from 0 to 1 - uniform coverage
    - COSINE: Slower start, accelerates mid-range - recommended by many papers
    - QUADRATIC: Even more aggressive late-stage masking
    - SQRT: More gradual early masking, accelerates later
    - INV_SQRT: Inverse sqrt - more gradual than linear early on
    - SIGMOID: S-curve - gradual start and end, steep middle
    - WARMUP: Very slow start, accelerates dramatically
    - CONSTANT: Fixed ratio (uses T/2 as reference for ~50% mask at T)
    - COSINE_INV: More masking early, slows down later
    """
    schedule_type = schedule_type.lower()

    def linear_schedule(t: torch.Tensor, T: int) -> torch.Tensor:
        """Linear schedule: ratio = t/T"""
        return t.float() / float(T)

    def cosine_schedule(t: torch.Tensor, T: int) -> torch.Tensor:
        """Cosine schedule: ratio = 0.5 * (1 + cos(pi * t/T))

        Starts at ~0, peaks at T/2 with ~1.0, back to ~0 at T
        Actually this is one minus cosine, so it's: ratio increases from 0 to 1
        """
        pi = torch.tensor(torch.pi)
        return 0.5 * (1 + torch.cos(pi * t.float() / float(T)))

    def quadratic_schedule(t: torch.Tensor, T: int) -> torch.Tensor:
        """Quadratic schedule: ratio = (t/T)^2"""
        ratio = t.float() / float(T)
        return ratio * ratio

    def sqrt_schedule(t: torch.Tensor, T: int) -> torch.Tensor:
        """Square root schedule: ratio = sqrt(t/T)"""
        ratio = t.float() / float(T)
        return torch.sqrt(ratio)

    def inv_sqrt_schedule(t: torch.Tensor, T: int) -> torch.Tensor:
        """Inverse square root schedule: ratio = 1 - sqrt(1 - t/T)"""
        ratio = t.float() / float(T)
        return 1 - torch.sqrt(1 - ratio)

    def sigmoid_schedule(
        t: torch.Tensor, T: int, k: float = 10.0
    ) -> torch.Tensor:
        """Sigmoid schedule: ratio = 1 / (1 + exp(-k * (t/T - 0.5)))"""
        ratio = t.float() / float(T)
        return 1.0 / (1.0 + torch.exp(-k * (ratio - 0.5)))

    def warmup_schedule(t: torch.Tensor, T: int) -> torch.Tensor:
        """Warmup (cubic) schedule: ratio = (t/T)^3"""
        ratio = t.float() / float(T)
        return ratio * ratio * ratio

    def constant_schedule(t: torch.Tensor, T: int) -> torch.Tensor:
        """Constant schedule: fixed ratio of ~0.5 at any timestep"""
        return torch.full_like(t.float(), 0.5)

    def cosine_inv_schedule(t: torch.Tensor, T: int) -> torch.Tensor:
        """Cosine inverse schedule: ratio = 0.5 * (1 - cos(pi * t/T))

        More aggressive early masking, gentler later.
        """
        pi = torch.tensor(torch.pi)
        return 0.5 * (1 - torch.cos(pi * t.float() / float(T)))

    schedules = {
        "linear": linear_schedule,
        "cosine": cosine_schedule,
        "quadratic": quadratic_schedule,
        "sqrt": sqrt_schedule,
        "inv_sqrt": inv_sqrt_schedule,
        "sigmoid": sigmoid_schedule,
        "warmup": warmup_schedule,
        "constant": constant_schedule,
        "cosine_inv": cosine_inv_schedule,
    }

    if schedule_type not in schedules:
        raise ValueError(
            f"Unknown schedule type: {schedule_type}. "
            f"Available: {list(schedules.keys())}"
        )

    return schedules[schedule_type]


# Default schedule type
DEFAULT_MASK_SCHEDULE = "linear"


def mask_ratio_schedule(
    t: torch.Tensor, T: int, schedule: str = DEFAULT_MASK_SCHEDULE
) -> torch.Tensor:
    """
    Compute masking ratio based on timestep and schedule type.

    Args:
        t: Timestep tensor.
        T (int): Total diffusion steps.
        schedule (str): Schedule type - 'linear', 'cosine', 'quadratic',
                        'sqrt', 'inv_sqrt', 'sigmoid', 'warmup', 'constant', 'cosine_inv'

    Returns:
        Tensor: Masking ratio.
    """
    schedule_fn = get_mask_schedule_fn(schedule)
    return schedule_fn(t, T)


def corrupt_with_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    t: torch.Tensor,
    tokenizer: PreTrainedTokenizerFast,
    T: int,
    schedule: str = DEFAULT_MASK_SCHEDULE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Corrupt input by masking tokens based on timestep.

    Args:
        input_ids: Input token IDs.
        attention_mask: Attention mask.
        t: Timestep.
        tokenizer: Tokenizer with special token IDs.
        T (int): Total steps.
        schedule (str): Mask schedule type - 'linear', 'cosine', 'quadratic',
                        'sqrt', 'inv_sqrt', 'sigmoid', 'warmup', 'constant', 'cosine_inv'

    Returns:
        tuple: (noisy_ids, labels, mask_positions)
    """
    # Returns noisy_ids, labels, mask_positions
    B, L = input_ids.shape
    ratio = mask_ratio_schedule(t, T, schedule).unsqueeze(1)  # [B,1]

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
    schedule: str = DEFAULT_MASK_SCHEDULE,
) -> torch.Tensor:
    """
    Compute diffusion loss.

    Args:
        model: The diffusion model.
        batch: Batch data.
        tokenizer: Tokenizer.
        T (int): Total diffusion steps.
        schedule (str): Mask schedule type - 'linear', 'cosine', 'quadratic',
                        'sqrt', 'inv_sqrt', 'sigmoid', 'warmup', 'constant', 'cosine_inv'

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
        schedule=schedule,
    )

    logits = model(
        noisy_ids, timesteps=t, attention_mask=attention_mask
    )  # [B,L,V]
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    return loss
