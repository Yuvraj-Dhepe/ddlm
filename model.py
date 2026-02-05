"""
Model module defining the Diffusion Language Model.

This module contains the DiffusionTransformerLM class, a bidirectional Transformer for masked denoising.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DiffusionLMConfig
from typing import Optional


class DiffusionTransformerLM(nn.Module):
    """
    Diffusion Language Model using a bidirectional Transformer.

    Args:
        cfg (DiffusionLMConfig): Model configuration.
    """
    def __init__(self, cfg: DiffusionLMConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.time_emb = nn.Embedding(cfg.diffusion_steps + 1, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Tie weights (optional; common in LMs)
        self.lm_head.weight = self.tok_emb.weight

        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, input_ids, timesteps, attention_mask=None):
        # input_ids: [B, L]
        # timesteps: [B] integer diffusion step in [1..T]
        # attention_mask: [B, L] bool, True for non-pad tokens

        B, L = input_ids.shape
        if L > self.cfg.seq_len:
            raise ValueError(f"Sequence length {L} > cfg.seq_len {self.cfg.seq_len}")

        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)  # [1, L]
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        t_emb = self.time_emb(timesteps).unsqueeze(1)  # [B, 1, D]
        x = x + t_emb
        x = self.drop(x)

        if attention_mask is None:
            src_key_padding_mask = None
        else:
            src_key_padding_mask = ~attention_mask  # invert: True = pad/ignore

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, L, V]
        return logits