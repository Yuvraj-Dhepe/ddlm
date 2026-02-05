"""
Data module for loading datasets, training tokenizer, and preparing data loaders.

This module handles the TinyStories dataset loading, tokenizer training from scratch, and creation of token-block datasets for training.
"""

import os
import random
from typing import Tuple, Optional
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from torch.utils.data import IterableDataset, DataLoader


def load_datasets(train_examples: int, val_examples: int) -> Tuple[Dataset, Dataset]:
    """
    Load TinyStories datasets.

    Args:
        train_examples (int): Number of training examples to load.
        val_examples (int): Number of validation examples to load.

    Returns:
        tuple: (train_ds, val_ds)
    """
    # !pip uninstall numpy -y --quiet
    # !pip install numpy==1.23.5 --quiet
    train_ds = load_dataset("roneneldan/TinyStories", split=f"train[:{train_examples}]")
    val_ds   = load_dataset("roneneldan/TinyStories", split=f"validation[:{val_examples}]")

    print(train_ds, val_ds)
    print("\nExample:\n", train_ds[0]["text"][:500])
    return train_ds, val_ds


def train_tokenizer_from_scratch(train_ds: Dataset, vocab_size: int, tokenizer_train_examples: int) -> str:
    """
    Train a Byte-level BPE tokenizer from scratch.

    Args:
        train_ds: Training dataset.
        vocab_size (int): Vocabulary size.
        tokenizer_train_examples (int): Number of examples to use for training tokenizer.

    Returns:
        str: Path to the saved tokenizer file.
    """
    SPECIAL_TOKENS = [
        "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]",
        "<|user|>", "<|assistant|>", "<|system|>", "<|end|>",
    ]

    def tokenizer_training_iterator(ds, n_examples):
        for i in range(min(n_examples, len(ds))):
            story = ds[i]["text"].strip()
            yield f"<|user|>\nWrite a short story.\n<|assistant|>\n{story}\n<|end|>\n"

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )

    print("Training tokenizer...")
    tokenizer.train_from_iterator(
        tokenizer_training_iterator(train_ds, tokenizer_train_examples),
        trainer=trainer
    )

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
    )
    tokenizer.decoder = ByteLevelDecoder()

    TOKENIZER_DIR = "tokenizer_from_scratch"
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    TOKENIZER_FILE = os.path.join(TOKENIZER_DIR, "tokenizer.json")
    tokenizer.save(TOKENIZER_FILE)

    print("Saved tokenizer to:", TOKENIZER_FILE)
    print("Vocab size:", tokenizer.get_vocab_size())
    return TOKENIZER_FILE


def create_hf_tokenizer(tokenizer_file: str) -> PreTrainedTokenizerFast:
    """
    Create a HuggingFace PreTrainedTokenizerFast from the trained tokenizer.

    Args:
        tokenizer_file (str): Path to the tokenizer JSON file.

    Returns:
        PreTrainedTokenizerFast: The tokenizer.
    """
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)

    hf_tokenizer.pad_token  = "[PAD]"
    hf_tokenizer.unk_token  = "[UNK]"
    hf_tokenizer.bos_token  = "[BOS]"
    hf_tokenizer.eos_token  = "[EOS]"
    hf_tokenizer.mask_token = "[MASK]"

    hf_tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]
    })

    PAD_ID  = hf_tokenizer.pad_token_id
    MASK_ID = hf_tokenizer.mask_token_id
    BOS_ID  = hf_tokenizer.bos_token_id
    EOS_ID  = hf_tokenizer.eos_token_id

    print("PAD_ID:", PAD_ID, "MASK_ID:", MASK_ID, "BOS_ID:", BOS_ID, "EOS_ID:", EOS_ID)
    print("Example encoding:", hf_tokenizer.encode("Hello world!")[:20])
    return hf_tokenizer


def format_as_chat(story_text: str) -> str:
    """
    Format a story text into a chat format.

    Args:
        story_text (str): The story text.

    Returns:
        str: Formatted chat text.
    """
    story_text = story_text.strip()
    return f"<|user|>\nWrite a short story.\n<|assistant|>\n{story_text}\n<|end|>\n"


class TokenBlockDataset(IterableDataset):
    """
    Iterable dataset for token blocks.

    Args:
        hf_ds: HuggingFace dataset.
        tokenizer: Tokenizer.
        seq_len (int): Sequence length.
        shuffle (bool): Whether to shuffle.
        seed (int): Random seed.
    """
    def __init__(self, hf_ds, tokenizer, seq_len, shuffle=False, seed=0):
        self.hf_ds = hf_ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        indices = list(range(len(self.hf_ds)))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)

        buffer = []
        for idx in indices:
            text = format_as_chat(self.hf_ds[idx]["text"])
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            buffer.extend(ids)

            while len(buffer) >= self.seq_len:
                block = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                yield torch.tensor(block, dtype=torch.long)


def create_data_loaders(train_ds: Dataset, val_ds: Dataset, tokenizer: PreTrainedTokenizerFast, seq_len: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.

    Args:
        train_ds: Training dataset.
        val_ds: Validation dataset.
        tokenizer: Tokenizer.
        seq_len (int): Sequence length.
        batch_size (int): Batch size.

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_blocks = TokenBlockDataset(train_ds, tokenizer, seq_len, shuffle=True, seed=42)
    val_blocks   = TokenBlockDataset(val_ds,   tokenizer, seq_len, shuffle=False)

    def collate_blocks(batch):
        input_ids = torch.stack(batch, dim=0)  # [B, L]
        attention_mask = (input_ids != tokenizer.pad_token_id)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    train_loader = DataLoader(train_blocks, batch_size=batch_size, collate_fn=collate_blocks)
    val_loader   = DataLoader(val_blocks,   batch_size=batch_size, collate_fn=collate_blocks)

    b = next(iter(train_loader))
    print({k: v.shape for k, v in b.items()})
    print("Decoded snippet:\n", tokenizer.decode(b["input_ids"][0][:120].tolist()))
    return train_loader, val_loader