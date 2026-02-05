"""
Main script for training and generating with the Diffusion Language Model.

This script orchestrates the training, generation, and rendering processes using command-line arguments.
"""

import click
from typing import Optional
import torch
from config import get_training_config, DiffusionLMConfig
from data import load_datasets, train_tokenizer_from_scratch, create_hf_tokenizer, create_data_loaders
from model import DiffusionTransformerLM
from train import train_model
from generate import diffusion_generate, chat_prompt
from render import create_gif, create_cool_gif
from accelerate import Accelerator


@click.command()
@click.option('--run-mode', default='quick', help='Run mode: quick or budget_100')
@click.option('--train/--no-train', default=True, help='Whether to train the model')
@click.option('--generate/--no-generate', default=True, help='Whether to generate text')
@click.option('--render/--no-render', default=True, help='Whether to render GIF')
@click.option('--user-prompt', default="Once upon a time", help='User prompt for generation')
def main(run_mode: str, train: bool, generate: bool, render: bool, user_prompt: str) -> None:
    """
    Main function to run the diffusion LM pipeline.

    Args:
        run_mode (str): Run mode.
        train (bool): Whether to train.
        generate (bool): Whether to generate.
        render (bool): Whether to render.
        user_prompt (str): Prompt for generation.
    """
    cfg = get_training_config(run_mode)

    accelerator = Accelerator(mixed_precision="bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16")

    if train:
        # Load data
        train_ds, val_ds = load_datasets(cfg['TRAIN_EXAMPLES'], cfg['VAL_EXAMPLES'])

        # Train tokenizer
        tokenizer_file = train_tokenizer_from_scratch(train_ds, cfg['VOCAB_SIZE'], cfg['TOKENIZER_TRAIN_EXAMPLES'])
        tokenizer = create_hf_tokenizer(tokenizer_file)

        # Create data loaders
        train_loader, val_loader = create_data_loaders(train_ds, val_ds, tokenizer, cfg['SEQ_LEN'], cfg['BATCH_SIZE'])

        # Create model
        model_cfg = DiffusionLMConfig(
            vocab_size=len(tokenizer),
            seq_len=cfg['SEQ_LEN'],
            d_model=cfg['D_MODEL'],
            n_layers=cfg['N_LAYERS'],
            n_heads=cfg['N_HEADS'],
            d_ff=cfg['D_FF'],
            dropout=0.1,
            diffusion_steps=cfg['DIFFUSION_STEPS'],
        )
        model = DiffusionTransformerLM(model_cfg)

        # Train
        train_model(model, train_loader, val_loader, tokenizer, cfg, accelerator)

    if generate:
        # Load model and tokenizer if not trained
        if not train:
            # Assume checkpoint exists
            import json
            with open("checkpoints/final/config.json", "r") as f:
                cfg = json.load(f)
            model_cfg = DiffusionLMConfig(**cfg)
            model = DiffusionTransformerLM(model_cfg)
            model.load_state_dict(torch.load("checkpoints/final/model.pt"))
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast(tokenizer_file="checkpoints/final/tokenizer/tokenizer.json")

        prompt_text = chat_prompt(user_prompt)

        final_text, frames = diffusion_generate(
            model=accelerator.unwrap_model(model),
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            seq_len=cfg['SEQ_LEN'],
            max_new_tokens=128,
            diffusion_steps=cfg['DIFFUSION_STEPS'],
            temperature=1.0,
            top_k=50,
            record_steps=True,
        )

        print("Final decoded (raw):\n")
        print(final_text[:1000])
        print("\nRecorded frames:", len(frames))

    if render:
        if not generate:
            raise ValueError("Cannot render without generating")
        create_gif(frames, cfg['DIFFUSION_STEPS'], user_prompt, "inference.gif")
        create_cool_gif(frames, cfg['DIFFUSION_STEPS'], user_prompt, "inference_cool.gif")


if __name__ == "__main__":
    main()
