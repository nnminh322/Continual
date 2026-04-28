"""
Shared LLaMA route embedding extraction for runtime SRT.

Ported from routing_analysis/extract_embeddings_llama.py so that support
signatures and query routing use the same tokenizer settings and pooling:
  - add_special_tokens=False
  - left padding via tokenizer.padding_side
  - last hidden state at the last non-padding token
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def build_superni_source_prompt(sample: dict) -> str:
    instance = sample["Instance"]
    return instance["instruction"].format(instance["sentence"])


@torch.no_grad()
def extract_route_embeddings_from_texts(
    encoder_frozen,
    tokenizer,
    texts: Sequence[str],
    batch_size: int,
    max_length: int,
    device,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    dev = torch.device(device)
    all_embs = []
    step = max(1, int(batch_size))

    for start in range(0, len(texts), step):
        batch_texts = texts[start : start + step]
        enc = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        enc = {key: value.to(dev) for key, value in enc.items()}
        pooled = encoder_frozen(enc["input_ids"], enc["attention_mask"])
        all_embs.append(pooled.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)