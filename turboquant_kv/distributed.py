"""
Distributed (tensor-parallel) quantized KV cache.

Shards the KV cache across multiple GPUs by splitting attention heads.
Each GPU stores packed codes for its local shard of heads. Since each GPU
owns disjoint heads, no communication is needed for attention computation.

Usage:
    cache = DistributedQuantizedKVCache(
        config, num_layers=32, max_seq_len=8192,
        num_heads=32, head_dim=128, tp_size=4, tp_rank=rank,
        device=f"cuda:{rank}",
    )
    # or equivalently:
    cache = DistributedQuantizedKVCache.from_config(
        config, tp_size=4, tp_rank=rank,
        num_layers=32, max_seq_len=8192,
        num_heads=32, head_dim=128,
    )
"""

from __future__ import annotations

from typing import Optional

import torch

from turboquant_kv.config import TurboQuantConfig
from turboquant_kv.cache import QuantizedKVCache


class DistributedQuantizedKVCache:
    """Tensor-parallel quantized KV cache.

    Splits attention heads evenly across ``tp_size`` GPUs. Each rank stores
    only ``num_heads // tp_size`` heads worth of packed KV codes.

    Because standard multi-head attention computes each head independently,
    no all-reduce or cross-GPU communication is needed for either
    ``attention_scores`` or ``attention_values``.

    The caller (i.e. the model's TP attention layer) is expected to pass
    key/value/query tensors that are already sliced to the local heads for
    this rank.

    Attributes:
        local_num_heads: Number of heads owned by this rank.
        tp_size: Total number of tensor-parallel ranks.
        tp_rank: This rank's index in [0, tp_size).
        cache: The underlying ``QuantizedKVCache`` for local heads.
    """

    def __init__(
        self,
        config: TurboQuantConfig,
        num_layers: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        device: str = "cuda",
    ):
        """Initialize a distributed quantized KV cache.

        Args:
            config: TurboQuant configuration (bit widths, mode, etc.).
            num_layers: Number of transformer layers.
            max_seq_len: Maximum sequence length to pre-allocate.
            num_heads: *Total* number of attention heads (before sharding).
            head_dim: Dimension of each attention head.
            tp_size: Number of tensor-parallel ranks.
            tp_rank: This rank's index (0-based).
            device: Torch device string for this rank (e.g. "cuda:0").

        Raises:
            ValueError: If num_heads is not divisible by tp_size.
            ValueError: If tp_rank is out of range.
        """
        if num_heads % tp_size != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by tp_size ({tp_size})"
            )
        if not (0 <= tp_rank < tp_size):
            raise ValueError(
                f"tp_rank ({tp_rank}) must be in [0, {tp_size})"
            )

        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.num_heads = num_heads
        self.local_num_heads = num_heads // tp_size
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.device = device

        # Each rank creates a QuantizedKVCache for its local head shard only
        self.cache = QuantizedKVCache(
            config,
            num_layers,
            max_seq_len,
            self.local_num_heads,
            head_dim,
            device,
        )

    @classmethod
    def from_config(
        cls,
        config: TurboQuantConfig,
        tp_size: int,
        tp_rank: int,
        num_layers: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: Optional[str] = None,
    ) -> "DistributedQuantizedKVCache":
        """Construct from config with automatic device placement.

        If ``device`` is None, defaults to ``cuda:{tp_rank}``.

        This constructor is convenient when using ``torch.distributed``:
        the caller can pass ``tp_rank = torch.distributed.get_rank()``
        and ``tp_size = torch.distributed.get_world_size()``.

        Args:
            config: TurboQuant configuration.
            tp_size: Number of tensor-parallel ranks.
            tp_rank: This rank's index.
            num_layers: Number of transformer layers.
            max_seq_len: Maximum sequence length.
            num_heads: Total number of attention heads.
            head_dim: Dimension per head.
            device: Torch device string; defaults to ``cuda:{tp_rank}``.

        Returns:
            A configured ``DistributedQuantizedKVCache`` instance.
        """
        if device is None:
            device = f"cuda:{tp_rank}"
        return cls(
            config, num_layers, max_seq_len, num_heads, head_dim,
            tp_size=tp_size, tp_rank=tp_rank, device=device,
        )

    def append(self, layer_id: int, key: torch.Tensor, value: torch.Tensor):
        """Append new KV pairs for a layer (local heads only).

        The caller is responsible for slicing key/value to the local heads
        belonging to this TP rank before calling this method. In a standard
        tensor-parallel transformer, the attention layer already produces
        per-rank key/value tensors.

        Args:
            layer_id: Layer index in [0, num_layers).
            key: (num_heads_local, seq_new, head_dim) or
                 (batch, num_heads_local, seq_new, head_dim).
            value: Same shape as key.
        """
        self.cache.append(layer_id, key, value)

    def attention_scores(self, layer_id: int, query: torch.Tensor) -> torch.Tensor:
        """Compute attention logits from packed keys (local heads only).

        No cross-GPU communication is needed since each rank owns disjoint
        heads.

        Args:
            query: (batch, num_heads_local, seq_q, head_dim) or
                   (num_heads_local, seq_q, head_dim).

        Returns:
            Attention logits with the same leading dimensions and shape
            (..., seq_q, seq_kv).
        """
        return self.cache.attention_scores(layer_id, query)

    def attention_values(self, layer_id: int, attn_weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of values from packed storage (local heads only).

        No all-reduce is needed for TP attention since heads are independent.

        Args:
            attn_weights: (batch, num_heads_local, seq_q, seq_kv) or
                          (num_heads_local, seq_q, seq_kv).

        Returns:
            Output with same leading dimensions and shape (..., head_dim).
        """
        return self.cache.attention_values(layer_id, attn_weights)

    @property
    def seq_lens(self) -> torch.Tensor:
        """Per-layer sequence lengths (delegated to underlying cache)."""
        return self.cache.seq_lens

    def memory_bytes(self) -> int:
        """Memory usage on this rank in bytes."""
        return self.cache.memory_bytes()

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs full fp16 KV cache (this rank's shard)."""
        return self.cache.compression_ratio

    def __repr__(self) -> str:
        return (
            f"DistributedQuantizedKVCache("
            f"tp_rank={self.tp_rank}/{self.tp_size}, "
            f"local_heads={self.local_num_heads}, "
            f"layers={self.num_layers}, "
            f"max_seq={self.max_seq_len}, "
            f"device={self.device})"
        )
