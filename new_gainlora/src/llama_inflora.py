"""
FrozenLlamaExtractor — FROZEN backbone embedding extractor for SRT signatures.

This module provides FrozenLlamaExtractor: a wrapper that exposes the SAME
embedding space as the frozen (pre-finetuned) LLaMA backbone.

IMPORTANT: Always use THIS extractor (not the adapted model) for:
  1. SRT signature extraction  (add_task / _compute_and_store_signature)
  2. SRT routing at inference  (forward pass in LlamaModel.forward)

Using the adapted model would give a different embedding space → wrong distances.

Matches routing_analysis/extract_embeddings_llama.py:
  - Layer: hidden_states[-1] (last decoder layer)
  - Pool:  last non-padding token per sample (NOT mean pooling)
"""

import torch
import torch.nn as nn
from typing import Optional


class FrozenLlamaExtractor(nn.Module):
    """
    Frozen LLaMA decoder embedding extractor.

    Wraps the core LlamaModel to expose:
      forward(input_ids, attention_mask) → pooled embeddings (B, d)

    Pooling: last non-padding token per sample.
    Last token = position of the last non-masked (non-padding) token.

    The wrapped model is kept in eval() mode with ALL gradients disabled.
    """

    def __init__(self, llama_model: nn.Module):
        super().__init__()
        self.llama_model = llama_model
        # Freeze everything
        for param in self.llama_model.parameters():
            param.requires_grad = False
        self.llama_model.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract frozen embeddings from the last decoder layer.

        Args:
            input_ids: (B, L) token IDs
            attention_mask: (B, L) 1=real token, 0=padding

        Returns:
            pooled: (B, d) — last non-padding token embedding per sample
        """
        with torch.no_grad():
            out = self.llama_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Last decoder layer hidden states
            hidden = out.hidden_states[-1]  # (B, L, d)

            # Pool: last non-padding token per sample
            if attention_mask is None:
                # No padding → last token is at position L-1
                seq_lens = torch.full(
                    (hidden.size(0),), hidden.size(1) - 1,
                    dtype=torch.long, device=hidden.device
                )
            else:
                # Last real token position (masked positions have 0)
                seq_lens = attention_mask.long().sum(dim=1) - 1  # (B,)

            B = hidden.size(0)
            pooled = hidden[torch.arange(B, device=hidden.device), seq_lens]  # (B, d)

        return pooled  # (B, d), no gradients
