# Diffusion Language Model (DDLM) - Comprehensive Documentation

This document provides a complete, flow-based explanation of the Diffusion Language Model implementation. It covers the model architecture, data pipeline, and inference process in detail.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Model Architecture](#model-architecture)
   - [Core Components](#core-components)
   - [Forward Pass](#forward-pass)
   - [Weight Tying](#weight-tying)
3. [Data Pipeline](#data-pipeline)
   - [Dataset: TinyStories](#dataset-tinystories)
   - [Tokenizer Training](#tokenizer-training)
   - [Data Preparation](#data-preparation)
4. [Training Process](#training-process)
   - [Diffusion Corruption](#diffusion-corruption)
   - [Loss Computation](#loss-computation)
   - [Training Loop](#training-loop)
5. [Inference Process](#inference-process)
   - [Initialization](#initialization)
   - [Iterative Denoising](#iterative-denoising)
   - [Sampling Strategy](#sampling-strategy)
6. [Visualization](#visualization)
7. [Configuration](#configuration)
8. [Usage](#usage)

---

## High-Level Overview

This project implements a **discrete diffusion language model** trained from scratch on the TinyStories dataset. Unlike traditional autoregressive language models that generate tokens left-to-right, this model generates text by **iteratively denoising a masked sequence**.

### How Diffusion Generation Works

**Autoregressive (AR) Generation:**
```
token₁ → token₂ → token₃ → token₄ → ...
```

**Diffusion Generation:**
1. Start with a sequence filled with `[MASK]` tokens
2. At each step, predict the original token for all masked positions
3. Keep confident predictions, re-mask uncertain ones
4. Repeat until all tokens are revealed

This creates the characteristic "editing into existence" effect visible in the generated GIFs.

---

## Model Architecture

The model is implemented in [`src/models/model.py`](src/models/model.py) as the `DiffusionTransformerLM` class.

### Core Components

#### 1. Token Embeddings (`self.tok_emb`)

```python
self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
```

- **Purpose**: Maps token IDs to dense vectors of dimension `d_model`
- **Input**: Token IDs of shape `[batch_size, seq_len]`
- **Output**: Dense vectors of shape `[batch_size, seq_len, d_model]`

#### 2. Position Embeddings (`self.pos_emb`)

```python
self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
```

- **Purpose**: Provides positional information to the model
- **Key Insight**: Unlike causal language models, this uses **bidirectional** position embeddings since the model sees the entire sequence at once
- **Input**: Position indices `[0, 1, 2, ..., seq_len-1]`
- **Output**: Position vectors of shape `[seq_len, d_model]`

#### 3. Time (Timestep) Embeddings (`self.time_emb`)

```python
self.time_emb = nn.Embedding(cfg.diffusion_steps + 1, cfg.d_model)
```

- **Purpose**: Encodes the current diffusion timestep into the model
- **Why needed**: The model needs to know "how corrupted" the input is to predict the denoising step appropriately
- **Timestep range**: `1` (most corrupted) to `T` (least corrupted), with `0` representing clean data
- **Input**: Timestep scalar (e.g., `32` for step 32 out of 64)
- **Output**: Time vector of shape `[d_model]`, broadcast to all positions

#### 3.1 Deep Dive: Time Embeddings in Diffusion Language Models

* **What does time embedding represent?** It represents the current "timestep" or "noise level" in the diffusion process.
* **How is it different from positional embedding?** Positional embedding tells the model *where* a token is in the sequence (spatial). Time embedding tells the model *how corrupted* the sequence is (temporal/noise level).
* **What is it supposed to capture conceptually?** It captures the stage of the denoising process. At high timesteps (lots of noise), the model should make broad, structural guesses. At low timesteps (little noise), it should make fine-grained, local corrections.
* **Does it encode the amount of noise present at a timestep?** Yes, indirectly. The model learns to associate a specific timestep embedding with the typical amount of masking present at that step.
* **How do we determine the size of the time embedding matrix?** The size is `[diffusion_steps + 1, d_model]`. We need an embedding for each possible timestep from 1 to `T`, plus an optional 0 step (clean data). The dimension `d_model` matches the token and position embeddings so they can be added together.
* **Is the number of timesteps `T` a hyperparameter?** Yes. A larger `T` means the denoising process is broken down into smaller, easier steps, which can improve generation quality but makes inference slower.
* **Noise Schedule and Generalization:** The model learns to denoise a specific corruption distribution per timestep based on the training schedule (e.g., linear masking). If we change the masking or noise schedule at inference, the model might perform poorly because it receives inputs with a noise level it doesn't associate with the given timestep embedding. Furthermore, the model is not robust to unseen timesteps because the time embeddings are learned discretely (`nn.Embedding`).

#### 4. Transformer Encoder

```python
enc_layer = nn.TransformerEncoderLayer(
    d_model=cfg.d_model,
    nhead=cfg.n_heads,
    dim_feedforward=cfg.d_ff,
    dropout=cfg.dropout,
    batch_first=True,
    activation="gelu",
    norm_first=True,  # Pre-LN architecture for better training stability
)
self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
```

- **Architecture**: Bidirectional Transformer Encoder (not a decoder!)
- **Key difference from AR models**: Every token can attend to every other token in both directions. 
  > Basically we use all the values from attention matrix instead of lower diagonal part of the matrix
- **Parameters**:
  - `d_model`: Model dimension (e.g., 384 for quick mode)
  - `n_heads`: Number of attention heads (e.g., 6)
  - `d_ff`: Feed-forward dimension (typically 4 × d_model)
  - `n_layers`: Number of layers (e.g., 6)
  - `norm_first=True`: Pre-LayerNorm for better gradient flow

**Deep Dive: Encoder Layers and Multi-Head Attention**
* **`enc_layer` vs `encoder`**: `enc_layer` (`nn.TransformerEncoderLayer`) defines a single Transformer block (which includes multi-head self-attention, feed-forward network, layer normalization, and dropout). `encoder` (`nn.TransformerEncoder`) is a stack of multiple such blocks (defined by `num_layers=cfg.n_layers`).
* **Multi-Head Attention**: Each `enc_layer` contains multiple attention heads (`N_HEADS`), meaning it performs independent attention mechanisms operating in parallel.

**Deep Dive: Sequence Length and Context**
* **`SEQ_LEN` vs Context Length**: `SEQ_LEN` (Sequence Length) is conceptually the same as context length in large language models. It defines the maximum number of tokens the model can process in a single forward pass for every element in the batch. This helps define the fixed size of the "canvas" we are denoising.
* **Architectural Constraint**: Context length is purely architectural due to fixed-size positional embeddings (`nn.Embedding(cfg.seq_len, cfg.d_model)`). The model cannot process sequences longer than `SEQ_LEN` without architectural changes.
* **Can sequence length (`L`) be smaller than `SEQ_LEN`?** Yes. During inference, we might want to generate a sequence shorter than the maximum `seq_len`. During training, if we have variable-length sequences, we can process smaller batches to save compute. However, in this specific implementation, the `TokenBlockDataset` yields fixed-size blocks of exactly `seq_len`, so `L` will equal `seq_len` during training.

#### 5. Output Layers

```python
self.ln_f = nn.LayerNorm(cfg.d_model)
self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
```

- **LayerNorm**: Final layer normalization
- **LM Head**: Linear projection from hidden dimension to vocabulary logits

### Forward Pass

The forward method is defined in [`model.py:49-75`](src/models/model.py:49):

```python
def forward(self, input_ids, timesteps, attention_mask=None):
    # input_ids: [B, L] - token IDs
    # timesteps: [B] - diffusion step (1 to T)
    # attention_mask: [B, L] - True for valid tokens, False for padding
    
    B, L = input_ids.shape
    
    # 1. Token embeddings + Position embeddings
    pos = torch.arange(L, device=input_ids.device).unsqueeze(0)  # [1, L]
    x = self.tok_emb(input_ids) + self.pos_emb(pos)
    
    # 2. Add timestep embedding (broadcast across sequence)
    t_emb = self.time_emb(timesteps).unsqueeze(1)  # [B, 1, D]
    x = x + t_emb
    x = self.drop(x)
    
    # 3. Create attention mask (invert: True = ignore/pad)
    if attention_mask is None:
        src_key_padding_mask = None
    else:
        src_key_padding_mask = ~attention_mask
    
    # 4. Pass through bidirectional Transformer
    x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
    
    # 5. Final normalization and projection to vocabulary
    x = self.ln_f(x)
    logits = self.lm_head(x)  # [B, L, vocab_size]
    
    return logits
```

**Deep Dive: Forward Pass Step-by-Step**
1. **Input**: Receives `input_ids` (corrupted tokens) and `timesteps` (current noise level).
2. **Embeddings**: Converts token IDs to vectors (`tok_emb`). Generates position indices (`torch.arange`) and converts them to vectors (`pos_emb`). Adds them together.
3. **Time Conditioning**: Converts `timesteps` to vectors (`time_emb`). Adds this time vector to every token embedding in the sequence. Now each token knows *what* it is, *where* it is, and *how noisy* the overall sequence is.
4. **Transformer**: Passes the sequence through the bidirectional Transformer encoder. Every token attends to every other token to build context.
5. **Output**: Passes the final hidden states through a LayerNorm and a Linear head (`lm_head`) to get logits (scores) for every word in the vocabulary for every position.

**Key insight**: The model outputs logits for **every position** simultaneously, predicting what token should be at each position given the corrupted input and timestep.

#### Deep Dive: Implementation Details and Code Tricks

* **Positional Embeddings and `torch.arange`**:
  * **What is the use?** `torch.arange(L)` creates a 1D tensor containing a sequence of integers from `0` to `L-1`.
  * **Why pass `pos` to `pos_emb`?** The model has no inherent sense of token order (it treats the input as a "bag of words"). We pass these position indices to the `pos_emb` to retrieve a unique vector for each position.
  * **What is the output?** The output is a tensor of shape `[1, L, d_model]`. Each token at index `i` gets the `i`-th positional embedding vector added to it.
  * **Are embeddings randomly generated?** Initially, yes. When `time_emb` and `pos_emb` are created, their weights are initialized randomly. During training, these weights are updated via backpropagation so the model *learns* the optimal representations for each position and timestep.
* **Attention Masking (`key_padding_mask`)**:
  * **Why is it inverted?** PyTorch's `nn.TransformerEncoder` expects a `src_key_padding_mask` where `True` means "ignore this position" (it's padding) and `False` means "attend to this position". Standard attention masks usually use `True` for valid tokens and `False` for padding, so we invert it (`~attention_mask`).
  * **Is it None during training?** In this specific implementation, because `TokenBlockDataset` yields blocks of exactly `seq_len` without padding, `attention_mask` is effectively all `True`, so the inverted mask is all `False`. It doesn't mask anything out during training.
  * **Is it useful for diffusion?** It *is* useful if you have variable-length sequences with padding tokens (`[PAD]`). You want the model to ignore padding tokens when computing attention. It's also crucial during inference if you process sequences shorter than `seq_len`. Note that we *do not* use a causal mask (like in GPT) because diffusion models are bidirectional.
* **Essential PyTorch Modules Used:**
  * `nn.Embedding`: A simple lookup table that stores embeddings of a fixed dictionary and size. Used for tokens, positions, and timesteps.
  * `nn.LayerNorm`: Normalizes the activations of the previous layer for each given example in a batch independently. Helps stabilize and speed up training.
  * `nn.Linear`: Applies a linear transformation to the incoming data. Used for the final language modeling head.

### Weight Tying

```python
# Tie weights between embedding and output projection
self.lm_head.weight = self.tok_emb.weight
```

This is a common optimization in language models that reduces parameters and often improves performance.

---

## Data Pipeline

The data pipeline is implemented in [`src/data/data.py`](src/data/data.py).

### Dataset: TinyStories

```python
train_ds = load_dataset("roneneldan/TinyStories", split=f"train[:{train_examples}]")
val_ds = load_dataset("roneneldan/TinyStories", split=f"validation[:{val_examples}]")
```

- **Source**: HuggingFace `roneneldan/TinyStories` dataset
- **Content**: Short, simple stories designed for small language models
- **Why TinyStories**: 
  - Clean, simple language
  - Good for training and evaluating small models
  - Contains diverse narratives in a compact format

### Tokenizer Training

The tokenizer is trained from scratch using Byte-Level BPE (Byte Pair Encoding).

#### Special Tokens

```python
SPECIAL_TOKENS = [
    "[PAD]",      # Padding token
    "[UNK]",      # Unknown token
    "[BOS]",      # Beginning of sequence
    "[EOS]",      # End of sequence
    "[MASK]",     # Mask token (crucial for diffusion!)
    "<|user|>",   # Chat: user turn
    "<|assistant|>",  # Chat: assistant turn
    "<|system|>", # Chat: system message
    "<|end|>",    # Chat: end marker
]
```

#### Training Process

```python
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.normalizer = NFKC()  # Unicode normalization
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

trainer = BpeTrainer(
    vocab_size=vocab_size,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS,
)

tokenizer.train_from_iterator(tokenizer_training_iterator(...), trainer=trainer)
```

**Key points**:
- **Byte-level**: Works at the byte level, enabling handling of any text
- **BPE**: Merges frequent byte pairs to create subword tokens
- **Vocabulary size**: 8,000 for quick mode, 26,000 for budget_100

**Deep Dive: Custom Tokenizer**
* **Why train our own tokenizer?** We need specific special tokens like `[MASK]` for the diffusion process, and chat formatting tokens (`<|user|>`, `<|assistant|>`, etc.). While we could add these to an existing tokenizer, training from scratch ensures the vocabulary is perfectly tailored to our specific dataset (TinyStories) and special token needs.
* **Can we reuse an existing tokenizer?** Yes, we could reuse an existing tokenizer (like GPT-2's or Llama's) and add our special tokens to it. However, their vocabularies might be unnecessarily large or not optimized for the simple language of TinyStories.
* **How does vocabulary size affect the model?** A larger vocabulary means the model has to choose from more options at each step, making the classification task harder and requiring a larger `lm_head` (more parameters). A smaller vocabulary makes prediction easier but might require longer sequences to express the same meaning.
* **Does diffusion modeling impose special requirements?** The main requirement is the `[MASK]` token, which represents the fully corrupted state. The tokenizer must handle this token correctly.

#### Post-Processing

```python
bos_id = tokenizer.token_to_id("[BOS]")
eos_id = tokenizer.token_to_id("[EOS]")
tokenizer.post_processor = TemplateProcessing(
    single="[BOS] $A [EOS]",  # Automatically add BOS/EOS to single sequences
    special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
)
```

### Data Preparation

#### Chat Formatting

```python
def format_as_chat(story_text: str) -> str:
    return f"<|user|>\nWrite a short story.\n<|assistant|>\n{story_text}\n<|end|>\n"
```

This formats each story as a chat interaction, which helps the model learn to follow instructions.

**Deep Dive: Chat Formatting**
* **Why use `format_as_chat`?** It structures the data into a prompt-response format. This teaches the model to act as an assistant that responds to instructions, rather than just a text completion engine.
* **Is it necessary during pre-training?** No, standard pre-training usually just uses raw text. However, in this project, we are training from scratch directly on chat-formatted data. This is a form of "instruction pre-training" where the model learns language and instruction-following simultaneously.
* **Effect on Distribution**: The model learns the specific distribution of the chat format (e.g., it learns that `<|user|>` is followed by an instruction, then `<|assistant|>`, then the response). If we were to pretrain without chat format but fine-tune with it (the standard approach for models like Llama), the model would first learn general language modeling, then learn the specific chat format and behavior during fine-tuning.

#### Token Block Dataset

```python
class TokenBlockDataset(IterableDataset):
    def __init__(self, hf_ds, tokenizer, seq_len, shuffle=False, seed=0):
        # Accumulates tokens from multiple stories
        # Yields fixed-size blocks of tokens
```

**How it works**:
1. Iterate through dataset stories
2. Encode each story with tokenizer
3. Accumulate tokens in a buffer
4. When buffer has enough tokens, yield a block of `seq_len` tokens
5. Continue until all stories are processed

**Why this approach**:
- Efficient: No wasted tokens
- Simple: No complex packing needed
- Iterable: Memory efficient for large datasets

#### DataLoader Creation

```python
def collate_blocks(batch):
    # Why torch.stack instead of concatenation?
    # torch.stack takes a list of 1D tensors and stacks them along a *new* dimension
    # (creating a 2D batch [B, L]). Concatenation would join them along an *existing* dimension.
    input_ids = torch.stack(batch, dim=0)  # [B, L]
    attention_mask = input_ids != tokenizer.pad_token_id
    return {"input_ids": input_ids, "attention_mask": attention_mask}

train_loader = DataLoader(train_blocks, batch_size=batch_size, collate_fn=collate_blocks)
```

---

## Training Process

The training code is in [`src/training/train.py`](src/training/train.py).

### Diffusion Corruption

The diffusion process is implemented in [`src/utils/diffusion_utils.py`](src/utils/diffusion_utils.py).

#### Masking Schedule

```python
def mask_ratio_schedule(t: torch.Tensor, T: int) -> torch.Tensor:
    # Linear schedule: ratio = t/T
    return t.float() / float(T)
```

At timestep `t` (out of total `T` steps), approximately `t/T` of the tokens should be masked.

#### Corruption Function

```python
def corrupt_with_mask(input_ids, attention_mask, t, tokenizer, T):
    B, L = input_ids.shape
    ratio = mask_ratio_schedule(t, T).unsqueeze(1)  # [B, 1]
    
    # Don't mask special tokens (BOS, EOS, PAD)
    can_mask = attention_mask.clone()
    can_mask &= (input_ids != tokenizer.bos_token_id)
    can_mask &= (input_ids != tokenizer.eos_token_id)
    can_mask &= (input_ids != tokenizer.pad_token_id)
    
    # Randomly select positions to mask
    rand = torch.rand((B, L), device=input_ids.device)
    mask_positions = (rand < ratio) & can_mask
    
    # Apply masking
    noisy = input_ids.clone()
    noisy[mask_positions] = tokenizer.mask_token_id
    
    # Labels: -100 for unmasked (ignore in loss), original token for masked
    labels = torch.full_like(input_ids, -100)
    labels[mask_positions] = input_ids[mask_positions]
    
    return noisy, labels, mask_positions
```

**Key insight**: We use `-100` as the ignore index for cross-entropy loss, so only masked positions contribute to the loss.

### Loss Computation

```python
def diffusion_loss(model, batch, tokenizer, T):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    
    B = input_ids.size(0)
    
    # Sample random timesteps for each example in batch
    t = torch.randint(1, T + 1, (B,), device=input_ids.device)
    
    # Corrupt input with random masking
    noisy_ids, labels, _ = corrupt_with_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        t=t,
        tokenizer=tokenizer,
        T=T,
    )
    
    # Get model predictions
    logits = model(noisy_ids, timesteps=t, attention_mask=attention_mask)
    # logits: [B, L, vocab_size]
    
    # Compute cross-entropy loss (only on masked positions)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    
    return loss
```

**Training objective**: Given a corrupted sequence at timestep `t`, predict the original (uncorrupted) tokens at masked positions.

**Deep Dive: Loss Computation Trick**
* **`ignore_index=-100`**: `F.cross_entropy` computes the cross entropy loss but ignores targets with the value `-100`. This is a crucial trick because we only want to compute loss on the *masked* tokens, not the unmasked ones. If we didn't use `ignore_index=-100`, the model would try to predict the unmasked tokens as well, which is trivial (it can just copy them) and would dominate the loss, preventing it from learning to denoise the masked tokens.

### Training Loop

```python
def train_model(model, train_loader, val_loader, tokenizer, cfg, accelerator):
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg["LR"], 
        weight_decay=cfg["WEIGHT_DECAY"]
    )
    
    # Learning rate scheduler (cosine with warmup)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["WARMUP_STEPS"],
        num_training_steps=cfg["TRAIN_STEPS"],
    )
    
    # Prepare with Accelerate (distributed training, mixed precision)
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(...)
    
    for step in range(cfg["TRAIN_STEPS"]):
        # Get batch
        batch = next(train_iter)
        
        # Compute loss
        loss = diffusion_loss(model, batch, tokenizer, T=cfg["DIFFUSION_STEPS"])
        loss = loss / cfg["GRAD_ACCUM"]  # Gradient accumulation
        
        # Backward
        accelerator.backward(loss)
        
        # Update weights every GRAD_ACCUM steps
        if (step + 1) % cfg["GRAD_ACCUM"] == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Logging and validation
        if (step + 1) % 500 == 0:
            val_loss = eval_loss(model, val_loader, tokenizer, ...)
    
    # Save checkpoint
    torch.save(accelerator.unwrap_model(model).state_dict(), "outputs/model.pt")
```

**Key features**:
- **Gradient accumulation**: Allows larger effective batch sizes
- **Mixed precision**: BF16 or FP16 for faster training
- **Distributed**: Uses Accelerate for multi-GPU training
- **Warmup**: Gradual learning rate increase at the start

#### Deep Dive: Optimization and Training Mechanics

* **Accelerator Utilities**:
  * **`accelerator.is_main_process`**: When training on multiple GPUs, the script runs simultaneously on all GPUs. This flag is `True` only for the primary GPU. We use it to ensure things like saving checkpoints or logging only happen once.
  * **`accelerator.gather`**: In distributed training, each GPU computes results for its own slice of the batch. `gather` collects these tensors from all GPUs and concatenates them onto the main process to compute global metrics.
* **Gradient Clipping (`clip_grad_norm_`)**: It limits the maximum magnitude of the gradients during backpropagation. If gradients get too large (exploding gradients), the model weights can update too drastically, causing instability. Clipping ensures stable training.
* **How `diffusion_loss` works totally**:
  1. Takes a batch of clean token sequences.
  2. Samples a random timestep `t` for each sequence.
  3. Corrupts the sequences by replacing a fraction of tokens (determined by `t`) with `[MASK]`.
  4. Passes the corrupted sequences and timesteps to the model.
  5. The model predicts logits for all positions.
  6. Computes the cross-entropy loss *only* between the model's predictions and the original tokens at the *masked* positions (using `ignore_index=-100`).
* **Gradient Accumulation**: It's a technique to simulate a larger batch size when you don't have enough GPU memory. Instead of updating the model weights after every forward/backward pass, we accumulate the gradients over multiple smaller batches. We only call `optimizer.step()` after processing `GRAD_ACCUM` batches.
* **Why divide loss before backward?** (`loss = loss / cfg["GRAD_ACCUM"]`) When we accumulate gradients, PyTorch simply adds them up. Dividing the loss ensures the accumulated gradients have the correct magnitude (equivalent to the average over the larger simulated batch).
* **Effect on Training**: Memory usage stays low (equivalent to the small batch size). The number of optimizer steps is reduced by a factor of `GRAD_ACCUM`. If we had enough memory to increase the actual batch size instead, training would be faster because we could process more data in parallel.

#### Deep Dive: Learning Rate Scheduling

* **Cosine Learning Rate**: It's a schedule where the learning rate starts high (after a warmup period) and gradually decreases following the shape of a cosine curve, ending near zero. "Annealing" refers to this process of gradually reducing the learning rate over time.
* **Why use cosine?** The smooth, gradual decay allows the model to settle into a good local minimum without the sudden shocks of step decay.
* **Exploration vs Exploitation**:
  * **Exploration**: Early in training, a high learning rate allows the model to take large steps, exploring the loss landscape and escaping poor local minima.
  * **Exploitation**: Later in training, a low learning rate allows the model to take small, precise steps to fine-tune its weights and settle into the deepest part of the current minimum.
  * **Balance**: Cosine annealing provides a smooth transition. It spends a good amount of time exploring (the top, flat part of the curve), then smoothly transitions into exploitation (the steep drop and long tail).

---

## Inference Process

The generation code is in [`src/generation/generate.py`](src/generation/generate.py).

### Initialization

```python
def diffusion_generate(model, tokenizer, prompt_text, seq_len, max_new_tokens, ...):
    model.eval()
    device = next(model.parameters()).device
    
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_ids = torch.tensor(prompt_ids, device=device).unsqueeze(0)  # [1, Lp]
    
    Lp = prompt_ids.size(1)
    L = min(seq_len, Lp + max_new_tokens)
    gen_len = L - Lp
    
    # Initialize with all MASK tokens
    x = torch.full((1, L), tokenizer.mask_token_id, dtype=torch.long, device=device)
    x[:, :Lp] = prompt_ids[:, :Lp]  # Fill in prompt
    
    # Track which positions are fixed (from prompt)
    fixed = torch.zeros((1, L), dtype=torch.bool, device=device)
    fixed[:, :Lp] = True
    
    attention_mask = torch.ones((1, L), dtype=torch.bool, device=device)
```

**Key insight**: We start with a sequence that's mostly `[MASK]` tokens, with the prompt filled in at the beginning.

### Iterative Denoising

The main loop runs from `diffusion_steps` down to `1`:

```python
for s in range(diffusion_steps, 0, -1):
    t = torch.tensor([s], device=device, dtype=torch.long)
    
    # Get model predictions for all positions
    logits = model(x, timesteps=t, attention_mask=attention_mask)
    # logits: [1, L, vocab_size]
    
    # Sample tokens from logits
    sampled, conf = sample_from_logits(logits)
    
    # Update unmasked positions with sampled tokens
    update_pos = ~fixed  # Positions that are not fixed (not from prompt)
    x[update_pos] = sampled[update_pos]
    
    # Calculate how many positions should remain masked
    next_ratio = float(s - 1) / float(diffusion_steps)
    target_masks = int(math.ceil(gen_len * next_ratio))
    
    # Re-mask the least confident predictions
    gen_positions = torch.arange(L, device=device) >= Lp
    candidates = gen_positions & (~fixed[0])
    cand_idx = torch.where(candidates)[0]
    
    if target_masks > 0 and cand_idx.numel() > 0:
        cand_conf = conf[0, cand_idx]
        k = min(target_masks, cand_idx.numel())
        _, low_idx = torch.topk(cand_conf, k=k, largest=False)  # Least confident
        remask_positions = cand_idx[low_idx]
        x[0, remask_positions] = tokenizer.mask_token_id
    
    # Record frame for GIF
    if record_steps:
        decoded = tokenizer.decode(x[0].tolist())
        frames.append((s, decoded))
```

**Deep Dive: How `diffusion_generate` works step-by-step**
1. **Initialize**: Start with a sequence of length `L` filled entirely with `[MASK]` tokens (except for the user prompt, which is fixed).
2. **Loop**: Iterate from `t = diffusion_steps` down to `1`.
3. **Predict**: Pass the current sequence and timestep `t` to the model. The model predicts the original token for *every* masked position.
4. **Sample**: Convert the model's output logits to probabilities and sample a token for each position. Also, record the model's confidence (probability) for its prediction.
5. **Update**: Replace all unmasked positions with the newly sampled tokens.
6. **Re-mask**: Calculate how many tokens should remain masked at the *next* step (`t-1`). Find the newly predicted tokens that the model was *least confident* about, and replace them back with `[MASK]`.
7. **Repeat**: Go to step 3 with the new, slightly less masked sequence. By `t=1`, 0 tokens are re-masked, and the generation is complete.

### Sampling Strategy

```python
def sample_from_logits(logits, temperature=1.0, top_k=0):
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature
    
    # Apply top-k filtering
    if top_k > 0:
        topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(-1, topk_idx, topk_vals)
        logits = filtered
    
    # Convert to probabilities and sample
    probs = F.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs.view(-1), num_samples=1).view(1, L)
    
    return sampled, probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
```

**Key aspects**:
- **Temperature**: Controls randomness (lower = more deterministic)
- **Top-k**: Limits sampling to top k tokens (reduces unlikely tokens)
- **Confidence tracking**: Keeps track of prediction confidence to decide which tokens to re-mask

#### Deep Dive: Logits and Sampling

* **What exactly are logits?** Logits are the raw, unnormalized scores output by the final linear layer of the model (`lm_head`). They can be any real number. They do not have a direct probabilistic meaning; higher scores just mean the model thinks that token is more likely.
* **Relation to Probabilities**: Logits are converted to probabilities using the Softmax function, which squashes the scores into a range between 0 and 1, ensuring they sum to 1. The model outputs logits instead of probabilities for numerical stability during loss computation.
* **Temperature**: Temperature is a hyperparameter used to scale the logits before applying Softmax (`logits = logits / temperature`).
  * **Higher Temperature (> 1.0)**: Logits are divided by a large number, making them smaller and closer to zero. The differences between scores are reduced. The Softmax output becomes "flatter" (more uniform), increasing randomness and diversity in sampling.
  * **Lower Temperature (< 1.0)**: Logits are divided by a fraction, making them larger. The differences between scores are magnified. The Softmax output becomes "sharper" (more confident), making the model more deterministic.

### Progressive Unmasking

The key to diffusion generation is the **progressive unmasking strategy**:

| Step | Masked % | Description |
|------|----------|-------------|
| T (start) | ~100% | Almost all tokens are masked |
| T/2 | ~50% | Half of generation is revealed |
| 1 (end) | 0% | All tokens revealed |

At each step:
1. Model predicts all masked positions
2. High-confidence predictions are kept
3. Low-confidence predictions are re-masked
4. Repeat until no masks remain

---

## Visualization

The visualization code creates terminal-style GIFs showing the diffusion process:

### Render Styles

1. **Basic** (`render.py`): Simple dark terminal with white text
2. **Neon** (`render.py`): Cyberpunk-style with cyan/magenta accents
3. **Classic** (`classic_render.py`): Modern minimal AGI aesthetic with electric blue

### GIF Creation

```python
def create_gif(frames, diffusion_steps, user_prompt, gif_path):
    gif_frames = []
    for s, decoded in frames:
        # Format as chat
        lines = make_chat_lines(user_prompt, decoded)
        lines.insert(2, f"(diffusion step {s:03d}/{diffusion_steps:03d})")
        
        # Render frame
        img = render_terminal_frame(lines)
        gif_frames.append(np.array(img))
    
    # Save as GIF
    imageio.mimsave(gif_path, gif_frames, duration=0.08)
```

---

## Configuration

Configuration is defined in [`src/config/config.py`](src/config/config.py).

### Run Modes

#### Quick Mode (for testing)
```python
TRAIN_EXAMPLES = 50_000
VAL_EXAMPLES = 2_000
SEQ_LEN = 256
VOCAB_SIZE = 8_000
D_MODEL = 384
N_LAYERS = 6
N_HEADS = 6
DIFFUSION_STEPS = 64
TRAIN_STEPS = 2_000
BATCH_SIZE = 32
```

#### Budget 100 Mode (for better quality)
```python
TRAIN_EXAMPLES = 1_000_000
VAL_EXAMPLES = 10_000
SEQ_LEN = 256
VOCAB_SIZE = 26_000
D_MODEL = 512
N_LAYERS = 10
N_HEADS = 8
DIFFUSION_STEPS = 128
TRAIN_STEPS = 50_000
BATCH_SIZE = 32
GRAD_ACCUM = 2
```

---

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

---

## File Structure Summary

```
ddlm/
├── main.py                      # Entry point, orchestrates everything
├── src/
│   ├── config/
│   │   └── config.py           # Configuration and hyperparameters
│   ├── data/
│   │   └── data.py             # Dataset loading, tokenizer, data loaders
│   ├── models/
│   │   └── model.py            # DiffusionTransformerLM architecture
│   ├── training/
│   │   └── train.py            # Training loop and evaluation
│   ├── generation/
│   │   └── generate.py         # Diffusion sampling for generation
│   ├── utils/
│   │   └── diffusion_utils.py  # Corruption and loss functions
│   └── visualization/
│       ├── render.py           # Basic and neon GIF rendering
│       └── classic_render.py   # Modern minimal GIF rendering
├── outputs/                     # Generated files (model, GIFs, etc.)
└── tokenizer_from_scratch/     # Trained tokenizer files
```

---

## Key Concepts Summary

1. **Bidirectional Transformer**: Unlike autoregressive models, this model sees the entire sequence at once, enabling better context understanding.

2. **Time Embeddings**: The model receives the diffusion timestep as input, allowing it to adjust its predictions based on how corrupted the input is.

3. **Masked Denoising**: Training involves predicting original tokens from corrupted (masked) inputs at random timesteps.

4. **Progressive Unmasking**: Generation starts with all tokens masked and progressively reveals high-confidence predictions while re-masking uncertain ones.

5. **Special Tokens**: The `[MASK]` token is crucial - it's used both during training (for corruption) and generation (as the starting state).

---

This documentation should provide a comprehensive understanding of how the Diffusion Language Model works from data preparation through training to inference. For more details, examine the source code comments and the original research papers on discrete diffusion language models.
