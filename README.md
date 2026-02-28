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
| `--mask-schedule` | Mask schedule: linear, cosine, quadratic, sqrt, inv_sqrt, sigmoid, warmup, constant, cosine_inv | "linear" |

## Hyperparameter Sweeps

Run hyperparameter optimization using Optuna + Weights & Biases:

```bash
# Quick mode sweep (3 trials - one per mask schedule)
python scripts/sweep.py \
  --sweep-config src/config/sweeps/sweep_quick.yaml \
  --trials 3 \
  --run-mode quick \
  --user-prompt "Once upon a time"

# Budget_100 mode sweep (9 trials - one per mask schedule)
python scripts/sweep.py \
  --sweep-config src/config/sweeps/sweep.yaml \
  --trials 3 \
  --run-mode budget_100 \
  --user-prompt "Once upon a time"
```

**Available sweep configs:**
- `src/config/sweeps/sweep.yaml` - budget_100 settings (larger model, more steps)
- `src/config/sweeps/sweep_quick.yaml` - quick settings (smaller model, fewer steps)

Both configs vary only the `mask_schedule` parameter across 9 different schedules:
- linear, cosine, quadratic, sqrt, inv_sqrt, sigmoid, warmup, constant, cosine_inv

Each trial saves:
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
# Training pipeline (quick mode)
uv run python deploy_fabric.py \
  --address user@remote-server.com \
  --ssh-key ~/.ssh/my_key \
  --run-mode quick \
  --user-prompt "Once upon a time"

# Training pipeline (budget_100 mode)
uv run python deploy_fabric.py \
  --address user@remote-server.com \
  --ssh-key ~/.ssh/my_key \
  --run-mode budget_100 \
  --user-prompt "Once upon a time"

# Hyperparameter sweep (auto-selects sweep config based on run-mode)
# quick mode -> sweep_quick.yaml (3 trials)
# budget_100 mode -> sweep.yaml (3 trials)
uv run python deploy_fabric.py \
  --address shadeform@149.36.1.177 \
  --ssh-key ~/.ssh/sform_key \
  --run-mode quick \
  --run-sweep \
  --sweep-trials 3 \
  --wandb-api-key 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 \
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
| `--sweep-trials` | Number of sweep trials (default: 3, use 9 for full mask schedule sweep) |
| `--wandb-api-key` | W&B API key for remote login (optional) |

**Note:** When `--run-sweep` is enabled, the script automatically selects:
- `sweep_quick.yaml` for `quick` mode
- `sweep.yaml` for `budget_100` mode

## Documentation

For detailed documentation on model architecture, data pipeline, training process, and inference, see [`docs/DDLM_code_explnation.md`](docs/DDLM_code_explnation.md).
