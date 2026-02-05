# Modular Diffusion Language Model from TinyStories

This project implements a **discrete diffusion language model** (a masked-denoising Transformer) trained from random initialization on a small slice of TinyStories. It performs **diffusion sampling** and exports **terminal-looking inference GIFs**.

The notebook has been broken down into modular Python files for better organization and maintainability.

## Overview

Autoregressive (AR) LMs generate **left-to-right**:

> token₁ → token₂ → token₃ → …

A masked diffusion LM generates by **iteratively denoising a whole sequence in parallel**:

1. Start from a sequence that is mostly (or entirely) `[MASK]`.
2. Predict tokens for all masked positions.
3. Keep the most confident predictions, re-mask the uncertain ones.
4. Repeat for many steps until no masks remain.

This gives the "cool" effect where the output looks like it's being *edited into existence*.

## File Structure

### `config.py`
- **Purpose**: Contains configuration parameters and the `DiffusionLMConfig` dataclass.
- **Functions**:
  - `get_training_config(run_mode)`: Returns training parameters based on run mode ("quick" or "budget_100").
- **Details**: Defines model hyperparameters, training settings, and dataset sizes. The run mode controls whether to use a quick setup for testing or a more intensive budget setup for better quality.

### `data.py`
- **Purpose**: Handles dataset loading, tokenizer training, and data preparation.
- **Functions**:
  - `load_datasets(train_examples, val_examples)`: Loads TinyStories train and validation datasets.
  - `train_tokenizer_from_scratch(train_ds, vocab_size, tokenizer_train_examples)`: Trains a Byte-level BPE tokenizer.
  - `create_hf_tokenizer(tokenizer_file)`: Creates a HuggingFace tokenizer from the trained tokenizer.
  - `format_as_chat(story_text)`: Formats story text into chat format.
  - `TokenBlockDataset`: Iterable dataset for token blocks.
  - `create_data_loaders(train_ds, val_ds, tokenizer, seq_len, batch_size)`: Creates training and validation data loaders.
- **Details**: Prepares the data by loading TinyStories, training a custom tokenizer with special tokens like `[MASK]`, `[BOS]`, etc., and creating batched datasets for training.

### `model.py`
- **Purpose**: Defines the Diffusion Language Model architecture.
- **Classes**:
  - `DiffusionTransformerLM`: A bidirectional Transformer with token embeddings, position embeddings, time-step embeddings, and a Transformer encoder.
- **Details**: Implements the core model that predicts original tokens at masked positions during training. Uses a standard Transformer architecture with modifications for diffusion.

### `diffusion_utils.py`
- **Purpose**: Contains utilities for diffusion corruption and loss calculation.
- **Functions**:
  - `mask_ratio_schedule(t, T)`: Linear masking ratio schedule.
  - `corrupt_with_mask(input_ids, attention_mask, t, tokenizer, T)`: Corrupts input by masking tokens.
  - `diffusion_loss(model, batch, tokenizer, T)`: Computes diffusion loss.
- **Details**: Handles the diffusion process by masking tokens based on timestep and calculating the loss for training the model to denoise.

### `train.py`
- **Purpose**: Manages the training loop and evaluation.
- **Functions**:
  - `eval_loss(model, val_loader, tokenizer, diffusion_steps, accelerator, n_batches)`: Evaluates validation loss.
  - `train_model(model, train_loader, val_loader, tokenizer, cfg, accelerator)`: Runs the full training loop.
- **Details**: Uses Accelerate for distributed training, implements gradient accumulation, learning rate scheduling, and saves the final checkpoint.

### `generate.py`
- **Purpose**: Handles text generation using diffusion sampling.
- **Functions**:
  - `diffusion_generate(model, tokenizer, prompt_text, seq_len, ...)`: Generates text by iteratively denoising.
  - `chat_prompt(user_msg, system_msg)`: Creates chat-formatted prompts.
- **Details**: Implements the sampling process where the model starts with masked sequences and progressively unmasks confident predictions.

### `render.py`
- **Purpose**: Creates terminal-style GIFs from generation frames.
- **Functions**:
  - `render_terminal_frame(lines, ...)`: Renders a basic terminal frame.
  - `render_terminal_frame_neon(lines, ...)`: Renders a neon-style terminal frame.
  - `create_gif(frames, diffusion_steps, user_prompt, gif_path)`: Creates a GIF from frames.
  - `create_cool_gif(...)`: Creates a neon-themed GIF.
- **Details**: Uses PIL to draw terminal-like interfaces showing the diffusion process step-by-step, with options for different styles.

### `main.py`
- **Purpose**: Orchestrates the entire pipeline using command-line arguments.
- **Features**:
  - Uses Click for CLI with options for run mode, training, generation, and rendering.
  - Loads configurations, prepares data, trains model, generates text, and renders GIFs.
- **Usage**:
  ```bash
  python main.py --run-mode quick --train --generate --render --user-prompt "Once upon a time"
  ```

## Installation and Dependencies

Install required packages (from the original notebook):

```bash
pip install -U datasets tokenizers accelerate tqdm numpy einops imageio pillow transformers hf_transfer
```

## Training

To train the model:

```bash
python main.py --run-mode budget_100 --train
```

This will train on 1M examples with a larger model.

For quick testing:

```bash
python main.py --run-mode quick --train
```

## Generation and Rendering

After training, generate and render:

```bash
python main.py --no-train --generate --render --user-prompt "Write a short story about a cat."
```

This creates `inference.gif` and `inference_cool.gif`.

## Deployment

Two deployment scripts are provided for running the pipeline on remote machines:

### Shell Script (`deploy_and_run.sh`)
```bash
./deploy_and_run.sh user@remote-server.com ~/.ssh/my_key [run_mode] [user_prompt]
```

### Python Fabric Script (`deploy_fabric.py`)
```bash
uv run python deploy_fabric.py --address user@remote-server.com --ssh-key ~/.ssh/my_key --run-mode quick --user-prompt "Once upon a time"
```

Both scripts will:
- Push the codebase to the remote machine
- Install uv and dependencies
- Run training, generation, and GIF creation
