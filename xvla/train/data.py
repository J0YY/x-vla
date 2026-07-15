"""Token data loading for Stage-1 language pretraining.

Two backends:

* ``prepare_tinyshakespeare`` — downloads the ~1MB char corpus and tokenizes it
  with tiktoken (gpt2). Good for a fast smoke test of the four-way ablation.
* ``TokenBinDataset`` — memory-maps a flat ``uint16`` token ``.bin`` file
  (nanoGPT / fineweb layout) for the full-scale ablation.
"""

from __future__ import annotations

import os
from urllib.request import urlopen

import numpy as np
import torch

TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def prepare_tinyshakespeare(data_dir: str) -> tuple[str, str, int]:
    """Download + tokenize tiny-shakespeare into train/val .bin files.

    Returns (train_bin, val_bin, vocab_size).
    """
    import tiktoken

    os.makedirs(data_dir, exist_ok=True)
    txt_path = os.path.join(data_dir, "tinyshakespeare.txt")
    if not os.path.exists(txt_path):
        with urlopen(TINYSHAKESPEARE_URL) as resp:
            text = resp.read().decode("utf-8")
        with open(txt_path, "w") as f:
            f.write(text)
    else:
        with open(txt_path) as f:
            text = f.read()

    enc = tiktoken.get_encoding("gpt2")
    ids = np.array(enc.encode_ordinary(text), dtype=np.uint16)
    n = len(ids)
    split = int(n * 0.9)
    train_bin = os.path.join(data_dir, "ts_train.bin")
    val_bin = os.path.join(data_dir, "ts_val.bin")
    ids[:split].tofile(train_bin)
    ids[split:].tofile(val_bin)
    # gpt2 vocab is 50257; pad to a multiple of 64 for efficiency (50304).
    return train_bin, val_bin, 50304


class TokenBinDataset:
    """Random contiguous-block sampler over a flat uint16 token file."""

    def __init__(self, bin_path: str, seq_len: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        if len(self.data) <= seq_len + 1:
            raise ValueError(f"{bin_path} too small ({len(self.data)} tokens)")

    def batch(self, batch_size: int, device, generator: torch.Generator | None = None):
        hi = len(self.data) - self.seq_len - 1
        ix = torch.randint(hi, (batch_size,), generator=generator)
        x = torch.stack([
            torch.from_numpy(self.data[i : i + self.seq_len].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i + 1 : i + 1 + self.seq_len].astype(np.int64))
            for i in ix
        ])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
