"""
Drop-in helpers for integrating TurboQuant with HuggingFace models.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from turboquant_kv.config import TurboQuantConfig
from turboquant_kv.cache import QuantizedKVCache


def wrap_model_kv_cache(
    model: nn.Module,
    config: TurboQuantConfig,
    max_seq_len: int = 4096,
) -> nn.Module:
    """Monkey-patch a HuggingFace model to use QuantizedKVCache.

    This replaces the model's default KV cache mechanism with a TurboQuant
    compressed cache. Currently supports LLaMA-style models (LlamaForCausalLM).

    Args:
        model: A HuggingFace causal LM model.
        config: TurboQuant configuration.
        max_seq_len: Maximum sequence length for pre-allocation.

    Returns:
        The modified model (same object, mutated in place).
    """
    try:
        from transformers import PreTrainedModel
    except ImportError:
        raise ImportError(
            "transformers is required for wrap_model_kv_cache. "
            "Install with: pip install turboquant-kv[transformers]"
        )

    if not isinstance(model, PreTrainedModel):
        raise TypeError(f"Expected a HuggingFace PreTrainedModel, got {type(model)}")

    # Extract model config
    model_config = model.config
    num_layers = getattr(model_config, "num_hidden_layers", None)
    num_heads = getattr(model_config, "num_key_value_heads",
                        getattr(model_config, "num_attention_heads", None))
    head_dim = getattr(model_config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(model_config, "hidden_size", None)
        n_heads = getattr(model_config, "num_attention_heads", None)
        if hidden_size and n_heads:
            head_dim = hidden_size // n_heads

    if any(v is None for v in [num_layers, num_heads, head_dim]):
        raise ValueError(
            "Could not infer model architecture. "
            f"num_layers={num_layers}, num_heads={num_heads}, head_dim={head_dim}"
        )

    device = next(model.parameters()).device

    cache = QuantizedKVCache(
        config=config,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        device=str(device),
    )

    # Attach cache to the model
    model._turboquant_cache = cache
    model._turboquant_config = config

    return model


class TurboQuantAttention(nn.Module):
    """Drop-in replacement attention layer that uses quantized KV cache.

    This is a stub for future implementation. The full version will:
    1. Accept Q, K, V projections as input
    2. Quantize K, V on the fly into the QuantizedKVCache
    3. Compute attention scores from packed keys
    4. Compute attention output from packed values
    """

    def __init__(
        self,
        config: TurboQuantConfig,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 4096,
        layer_id: int = 0,
        device: str = "cuda",
    ):
        super().__init__()
        self.config = config
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.layer_id = layer_id

        self.cache = QuantizedKVCache(
            config=config,
            num_layers=1,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, num_heads, seq_q, head_dim)
            key: (batch, num_heads, seq_new, head_dim)
            value: (batch, num_heads, seq_new, head_dim)
            attention_mask: Optional (batch, 1, seq_q, seq_kv) mask.

        Returns:
            Output: (batch, num_heads, seq_q, head_dim)
        """
        batch = query.shape[0]

        # Append new KV pairs (handle each batch item separately)
        for b in range(batch):
            self.cache.append(0, key[b], value[b])

        # Compute attention scores
        outputs = []
        for b in range(batch):
            logits = self.cache.attention_scores(0, query[b])  # (num_heads, seq_q, seq_kv)

            if attention_mask is not None:
                logits = logits + attention_mask[b]

            weights = torch.softmax(logits, dim=-1)
            out = self.cache.attention_values(0, weights)  # (num_heads, seq_q, head_dim)
            outputs.append(out)

        return torch.stack(outputs, dim=0)
