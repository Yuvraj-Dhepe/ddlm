"""
Generation module for diffusion sampling.

This module contains the function for generating text using diffusion sampling.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from model import DiffusionTransformerLM


def diffusion_generate(
    model: DiffusionTransformerLM,
    tokenizer: PreTrainedTokenizerFast,
    prompt_text: str,
    seq_len: int,
    max_new_tokens: int = 128,
    diffusion_steps: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
    record_steps: bool = True,
) -> Tuple[str, List[Tuple[int, str]]]:
    """
    Generate text using diffusion sampling.

    Args:
        model: The diffusion model.
        tokenizer: Tokenizer.
        prompt_text (str): Prompt text.
        seq_len (int): Sequence length.
        max_new_tokens (int): Maximum new tokens.
        diffusion_steps (int): Number of diffusion steps.
        temperature (float): Sampling temperature.
        top_k (int): Top-k sampling.
        record_steps (bool): Whether to record intermediate steps.

    Returns:
        tuple: (final_text, frames)
    """
    model.eval()
    device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(
        0
    )  # [1, Lp]

    Lp = prompt_ids.size(1)
    L = min(seq_len, Lp + max_new_tokens)
    gen_len = L - Lp

    x = torch.full((1, L), tokenizer.mask_token_id, dtype=torch.long, device=device)
    x[:, :Lp] = prompt_ids[:, :Lp]

    fixed = torch.zeros((1, L), dtype=torch.bool, device=device)
    fixed[:, :Lp] = True

    attention_mask = torch.ones((1, L), dtype=torch.bool, device=device)

    frames = []

    def sample_from_logits(logits):
        if temperature != 1.0:
            logits = logits / temperature

        if top_k and top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(-1, topk_idx, topk_vals)
            logits = filtered

        probs = F.softmax(logits, dim=-1)
        flat = probs.view(-1, probs.size(-1))
        sampled = torch.multinomial(flat, num_samples=1).view(1, L)
        sampled_prob = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)  # [1,L]
        return sampled, sampled_prob

    for s in range(diffusion_steps, 0, -1):
        t = torch.tensor([s], device=device, dtype=torch.long)
        logits = model(x, timesteps=t, attention_mask=attention_mask)
        sampled, conf = sample_from_logits(logits)

        update_pos = ~fixed
        x[update_pos] = sampled[update_pos]

        next_ratio = float(s - 1) / float(diffusion_steps)
        target_masks = int(math.ceil(gen_len * next_ratio))

        gen_positions = torch.arange(L, device=device) >= Lp
        candidates = gen_positions & (~fixed[0])
        cand_idx = torch.where(candidates)[0]

        if target_masks > 0 and cand_idx.numel() > 0:
            cand_conf = conf[0, cand_idx]
            k = min(target_masks, cand_idx.numel())
            _, low_idx = torch.topk(cand_conf, k=k, largest=False)
            remask_positions = cand_idx[low_idx]
            x[0, remask_positions] = tokenizer.mask_token_id

        if record_steps:
            decoded = tokenizer.decode(x[0].tolist())
            decoded = decoded.replace("[MASK]", "█")
            frames.append((s, decoded))

    final = tokenizer.decode(x[0].tolist())
    model.train()
    return final, frames


def chat_prompt(user_msg: str, system_msg: Optional[str] = None) -> str:
    """
    Create a chat prompt.

    Args:
        user_msg (str): User message.
        system_msg (str): System message.

    Returns:
        str: Formatted prompt.
    """
    parts = []
    if system_msg:
        parts.append(f"<|system|>\n{system_msg}\n")
    parts.append(f"<|user|>\n{user_msg}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)
