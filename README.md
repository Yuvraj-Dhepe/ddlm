# Modular Diffusion Language Model from TinyStories

This project implements a **discrete diffusion language model** (a masked-denoising Transformer) trained from random initialization on a small slice of TinyStories. It performs **diffusion sampling** and exports **terminal-looking inference GIFs**.

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

```
ddlm/
├── main.py                      # Entry point
├── scripts/
│   └── sweep.py                 # Hyperparameter sweep script
├── src/
│   ├── config/
│   │   └── config.py            # Configuration and hyperparameters
│   ├── data/
│   │   └── data.py              # Dataset loading, tokenizer, data loaders
│   ├── models/
│   │   └── model.py             # DiffusionTransformerLM architecture
│   ├── training/
│   │   └── train.py             # Training loop and evaluation
│   ├── generation/
│   │   └── generate.py          # Diffusion sampling for generation
│   ├── utils/
│   │   └── diffusion_utils.py   # Corruption and loss functions
│   └── visualization/
│       ├── render.py            # Basic and neon GIF rendering
│       └── classic_render.py   # Modern minimal GIF rendering
├── sweeps/                      # Sweep outputs
├── tokenizer_from_scratch/      # Trained tokenizer files
└── docs/
    └── DDLM_code_explnation.md  # Comprehensive documentation
```

## Usage

### Training

```bash
# Quick mode (for testing)
python main.py --run-mode quick --train

# Full training (better quality)
python main.py --run-mode budget_100 --train
```

### Generation

```bash
# Generate with custom prompt
python main.py --no-train --generate --user-prompt "Once upon a time in a distant land"

# Generate and create GIFs
python main.py --no-train --generate --render --user-prompt "Write a story about a brave knight"
```

### Full Pipeline

```bash
# Train, generate, and render in one command
python main.py --run-mode quick --train --generate --render --user-prompt "A curious cat"
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--run-mode` | Training mode: "quick" or "budget_100" | "quick" |
| `--train/--no-train` | Whether to train the model | True |
| `--generate/--no-generate` | Whether to generate text | True |
| `--render/--no-render` | Whether to render GIFs | True |
| `--user-prompt` | Prompt for generation | "Once upon a time" |

## Hyperparameter Sweeps

Run hyperparameter optimization using Optuna + Weights & Biases:

```bash
python scripts/sweep.py \
  --sweep-config src/config/sweeps/sweep.yaml \
  --trials 3 \
  --run-mode quick \
  --user-prompt "Once upon a time"
```

Configuration is defined in `src/config/sweeps/sweep.yaml`. Each trial saves:
- `model.pt` - Model weights
- `tokenizer.json` - Trained tokenizer  
- `inference.gif` - Generation visualization

Output: `sweeps/{study_name}/trial_{run_id}/`

## Installation

```bash
pip install -U datasets tokenizers accelerate tqdm numpy einops imageio pillow transformers hf_transfer
```

## Deployment

### Shell Script
```bash
./deploy_and_run.sh user@remote-server.com ~/.ssh/my_key [run_mode] [user_prompt]
```

### Python Fabric Script
```bash
# Training pipeline
uv run python deploy_fabric.py \
  --address user@remote-server.com \
  --ssh-key ~/.ssh/my_key \
  --run-mode quick \
  --user-prompt "Once upon a time"

# Hyperparameter sweep
uv run python deploy_fabric.py \
  --address user@remote-server.com \
  --ssh-key ~/.ssh/my_key \
  --run-mode quick \
  --run-sweep \
  --sweep-trials 5 \
  --wandb-api-key YOUR_WANDB_KEY \
  --user-prompt "Once upon a time"
```

**Options:**
| Option | Description |
|--------|-------------|
| `--address` | SSH address (user@host) |
| `--ssh-key` | Path to SSH private key |
| `--run-mode` | "quick" or "budget_100" |
| `--user-prompt` | Prompt for generation |
| `--run-sweep` | Run hyperparameter sweep instead of training |
| `--sweep-trials` | Number of sweep trials (default: 3) |
| `--wandb-api-key` | W&B API key for remote login (optional) |

## Documentation

For detailed documentation on model architecture, data pipeline, training process, and inference, see [`docs/DDLM_code_explnation.md`](docs/DDLM_code_explnation.md).
