"""
SRT Hypothesis Testing Framework

Validates SRT (Smart Routing Token) theory on SMoLoRA and HiDe-LLaVA continual
learning methods.

Components:
    - srt_router: Core Pooled Mahalanobis router (SRT Thm 4)
    - embedding_extractors: CLIP ViT and Sentence-BERT embeddings
    - experiments: Option A (routing accuracy) and Option B (end-to-end) evaluation
    - scoring: Task accuracy scoring functions following original code

Usage:
    # Option A: Routing accuracy (no trained model needed)
    python experiments/smolora/if_router/routing_accuracy.py --ins_emb path/to/ins_emb.pkl

    # Option B: End-to-end (after training)
    python experiments/run_all.py --model_path /path/to/checkpoint --routing_mode srt
"""

__version__ = "0.1.0"