"""
Main script for training and generating with the Diffusion Language Model.

This script orchestrates the training, generation, and rendering processes using command-line arguments.
"""

import click
import torch
from accelerate import Accelerator

from classic_render import create_classic_gif
from config import DiffusionLMConfig, get_training_config
from data import (
    create_data_loaders,
    create_hf_tokenizer,
    load_datasets,
    train_tokenizer_from_scratch,
)
from generate import chat_prompt, diffusion_generate
from model import DiffusionTransformerLM
from render import create_cool_gif, create_gif
from train import train_model


@click.command()
@click.option("--run-mode", default="quick", help="Run mode: quick or budget_100")
@click.option("--train/--no-train", default=True, help="Whether to train the model")
@click.option("--generate/--no-generate", default=True, help="Whether to generate text")
@click.option("--render/--no-render", default=True, help="Whether to render GIF")
@click.option(
    "--user-prompt", default="Once upon a time", help="User prompt for generation"
)
def main(
    run_mode: str, train: bool, generate: bool, render: bool, user_prompt: str
) -> None:
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

    accelerator = Accelerator(
        mixed_precision="bf16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "fp16"
    )

    if train:
        # Load data
        train_ds, val_ds = load_datasets(cfg["TRAIN_EXAMPLES"], cfg["VAL_EXAMPLES"])

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

        # Train
        train_model(model, train_loader, val_loader, tokenizer, cfg, accelerator)

    if generate:
        # Load model and tokenizer if not trained
        if not train:
            # Assume checkpoint exists
            import json

            from transformers import PreTrainedTokenizerFast

            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file="outputs/tokenizer/tokenizer.json"
            )
            tokenizer.pad_token = "[PAD]"
            tokenizer.unk_token = "[UNK]"
            tokenizer.bos_token = "[BOS]"
            tokenizer.eos_token = "[EOS]"
            tokenizer.mask_token = "[MASK]"

            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        "<|user|>",
                        "<|assistant|>",
                        "<|system|>",
                        "<|end|>",
                    ]
                }
            )

            try:
                with open("outputs/config.json", "r") as f:
                    cfg = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                # Fallback to default config if config.json is missing or empty
                cfg = get_training_config("quick")
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
            try:
                model.load_state_dict(torch.load("outputs/model.pt"))
            except (FileNotFoundError, RuntimeError):
                print(
                    "Model checkpoint not found or corrupted. Please train the model first."
                )
                return
            model = accelerator.prepare(model)

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

        print("Final decoded (raw):\n")
        print(final_text[:1000])
        print("\nRecorded frames:", len(frames))

    if render:
        if not generate:
            raise ValueError("Cannot render without generating")
        create_gif(frames, cfg["DIFFUSION_STEPS"], user_prompt, "outputs/inference.gif")
        create_cool_gif(
            frames, cfg["DIFFUSION_STEPS"], user_prompt, "outputs/inference_cool.gif"
        )
        create_classic_gif(
            frames, cfg["DIFFUSION_STEPS"], user_prompt, "outputs/inference_classic.gif"
        )


if __name__ == "__main__":
    main()
