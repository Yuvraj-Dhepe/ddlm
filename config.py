"""
Configuration module for the Diffusion Language Model training.

This module contains the training parameters based on the run mode and the DiffusionLMConfig dataclass.
"""

from typing import Dict, Any
from dataclasses import dataclass


def get_training_config(run_mode: str) -> Dict[str, Any]:
    """
    Get training configuration parameters based on the run mode.

    Args:
        run_mode (str): Either "quick" or "budget_100".

    Returns:
        dict: Dictionary containing all training parameters.

    Raises:
        ValueError: If run_mode is not "quick" or "budget_100".
    """
    # =======================
    # 0) Choose a run profile
    # =======================

    # You can always override individual values later.

    if run_mode == "quick":
        # Small + fast: good for verifying everything end-to-end
        TRAIN_EXAMPLES = 50_000
        VAL_EXAMPLES   = 2_000
        TOKENIZER_TRAIN_EXAMPLES = 30_000

        SEQ_LEN = 256
        VOCAB_SIZE = 8_000

        D_MODEL = 384
        N_LAYERS = 6
        N_HEADS = 6
        D_FF = 4 * D_MODEL

        DIFFUSION_STEPS = 64

        TRAIN_STEPS = 2_000
        BATCH_SIZE = 32
        GRAD_ACCUM = 1
        LR = 3e-4
        WEIGHT_DECAY = 0.1
        WARMUP_STEPS = 200

    elif run_mode == "budget_100":
        # Heavier: better quality, uses more compute
        TRAIN_EXAMPLES = 1000_000
        VAL_EXAMPLES   = 10_000
        TOKENIZER_TRAIN_EXAMPLES = 150_000

        SEQ_LEN = 256
        VOCAB_SIZE = 26_000

        D_MODEL = 512
        N_LAYERS = 10
        N_HEADS = 8
        D_FF = 4 * D_MODEL

        DIFFUSION_STEPS = 128

        TRAIN_STEPS = 50000
        BATCH_SIZE = 32
        GRAD_ACCUM = 2
        LR = 2e-4
        WEIGHT_DECAY = 0.1
        WARMUP_STEPS = 1_000

    else:
        raise ValueError("RUN_MODE must be 'quick' or 'budget_100'")

    print("RUN_MODE:", run_mode)

    return {
        'TRAIN_EXAMPLES': TRAIN_EXAMPLES,
        'VAL_EXAMPLES': VAL_EXAMPLES,
        'TOKENIZER_TRAIN_EXAMPLES': TOKENIZER_TRAIN_EXAMPLES,
        'SEQ_LEN': SEQ_LEN,
        'VOCAB_SIZE': VOCAB_SIZE,
        'D_MODEL': D_MODEL,
        'N_LAYERS': N_LAYERS,
        'N_HEADS': N_HEADS,
        'D_FF': D_FF,
        'DIFFUSION_STEPS': DIFFUSION_STEPS,
        'TRAIN_STEPS': TRAIN_STEPS,
        'BATCH_SIZE': BATCH_SIZE,
        'GRAD_ACCUM': GRAD_ACCUM,
        'LR': LR,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'WARMUP_STEPS': WARMUP_STEPS,
    }


@dataclass
class DiffusionLMConfig:
    """
    Configuration for the Diffusion Language Model.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        seq_len (int): Maximum sequence length.
        d_model (int): Model dimension.
        n_layers (int): Number of transformer layers.
        n_heads (int): Number of attention heads.
        d_ff (int): Feed-forward dimension.
        dropout (float): Dropout rate.
        diffusion_steps (int): Number of diffusion steps.
    """
    vocab_size: int
    seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    diffusion_steps: int