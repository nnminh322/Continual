"""
FrozenLlamaExtractor
====================
Wraps a frozen HF LlamaModel (decoder-only) to extract the last non-padding
token hidden state from the final transformer layer.

Used for SRT task routing: embeddings MUST come from the FROZEN pretrained
backbone (no LoRA), so that routing signatures {μ_t, Σ_t} are zero-drift.

Embedding extraction:
  1. Run frozen decoder with output_hidden_states=True, use_cache=False
  2. Take hidden_states[-1]  →  (B, seq, d)  final layer output
  3. Pick the last non-padding position per sample:
       lengths = attention_mask.sum(dim=1) - 1
       h = hidden_states[-1][arange(B), lengths]  →  (B, d)

This matches cl_trainer_srt.py Case 4 (bare LLaMA last-token extraction).
"""

import torch
import torch.nn as nn


class FrozenLlamaExtractor(nn.Module):
    """Frozen LLaMA decoder → (B, hidden_size) last non-padding token embedding."""

    def __init__(self, hf_llama_model: nn.Module):
        super().__init__()
        self.decoder = hf_llama_model
        self.decoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, seq_len)  left-padded source token ids
            attention_mask: (B, seq_len)  1 = real token, 0 = padding

        Returns:
            (B, hidden_size)  float32 embeddings
        """
        out = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        # hidden_states: tuple of len (num_layers + 1), each (B, seq, d)
        h = out.hidden_states[-1]  # (B, seq, d) — final layer

        # Last non-padding position (0-indexed)
        lengths = attention_mask.sum(dim=1) - 1   # (B,)
        lengths = lengths.clamp(min=0)
        batch_idx = torch.arange(h.size(0), device=h.device)
        pooled = h[batch_idx, lengths]             # (B, d)

        return pooled.float()
