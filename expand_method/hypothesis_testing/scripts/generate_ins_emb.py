#!/usr/bin/env python3
"""
Generate ins_emb_single.pkl — identical to SMoLoRA scripts/SMoLoRA/ins_gen.py.

Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim) with mean pooling.
7 instruction embeddings, one per ins_type (0–6):

  0: ScienceQA   — "Answer with the option's letter from the given choices directly."
  1: TextVQA    — "Answer the question using a single word or phrase."
  2: Flickr30k  — "What is happening in the presented picture?\\nPlease describe it in one complete sentence."
  3: ImageNet   — "What is the object in the image?\\nAnswer the question using a single word or phrase."
  4: GQA        — "Answer the question using a single word or phrase."
  5: VQAv2      — "Answer the question using a single word or phrase."
  6: Place365   — "What is the background of the image?\\nAnswer the question using a single word or phrase."

Usage:
    python scripts/generate_ins_emb.py \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --output ins_emb_single.pkl
"""
import argparse
import pickle
from pathlib import Path

import torch


def mean_pooling(model_output, attention_mask):
    """Mean pooling over token embeddings (same as SMoLoRA ins_gen.py)."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_instructions(instructions, model_name):
    """Encode instruction strings using Sentence-BERT."""
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    encoded = tokenizer(instructions, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    embeddings = mean_pooling(output, encoded["attention_mask"])
    return embeddings


INSTRUCTIONS = [
    # ins_type 0: ScienceQA
    "Answer with the option's letter from the given choices directly.",
    # ins_type 1: TextVQA
    "Answer the question using a single word or phrase.",
    # ins_type 2: Flickr30k
    "What is happening in the presented picture?\nPlease describe it in one complete sentence.",
    # ins_type 3: ImageNet
    "What is the object in the image?\nAnswer the question using a single word or phrase.",
    # ins_type 4: GQA
    "Answer the question using a single word or phrase.",
    # ins_type 5: VQAv2
    "Answer the question using a single word or phrase.",
    # ins_type 6: Place365
    "What is the background of the image?\nAnswer the question using a single word or phrase.",
]


def main():
    parser = argparse.ArgumentParser(description="Generate ins_emb_single.pkl")
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-BERT model name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ins_emb_single.pkl",
        help="Output pickle path",
    )
    args = parser.parse_args()

    print(f"Encoding {len(INSTRUCTIONS)} instructions with {args.model}...")
    embeddings = encode_instructions(INSTRUCTIONS, args.model)
    print(f"  Shape: {embeddings.shape}")  # expected: (7, 384)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"  Saved to {out_path}")

    # Also save as numpy for easy inspection
    np_path = out_path.with_suffix(".npy")
    import numpy as np
    np.save(np_path, embeddings.numpy())
    print(f"  Also saved numpy to {np_path}")


if __name__ == "__main__":
    main()
