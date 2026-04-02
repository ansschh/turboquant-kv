"""
TurboQuant vLLM out-of-tree plugin.

Provides a custom KV cache backend for vLLM that uses TurboQuant quantization
to compress keys and values on the fly, reducing memory footprint during inference.

Usage:
    # Via vLLM CLI:
    --kv-cache-dtype turboquant

    # Programmatically:
    from turboquant_kv.vllm_plugin import TurboQuantKVCacheConfig, register
    register()

Entry point registration (pyproject.toml / setup.cfg):
    [project.entry-points."vllm.general_plugins"]
    turboquant = "turboquant_kv.vllm_plugin:register"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from turboquant_kv.config import TurboQuantConfig
from turboquant_kv.cache import QuantizedKVCache

# ---------------------------------------------------------------------------
# Conditional vLLM imports — gracefully degrade if vLLM is absent
# ---------------------------------------------------------------------------

_VLLM_AVAILABLE = False

try:
    from vllm.attention.backends.abstract import (
        AttentionBackend,
        AttentionImpl,
        AttentionMetadata,
        AttentionType,
    )
    from vllm.attention.layer import Attention
    _VLLM_AVAILABLE = True
except ImportError:
    # Provide stub base classes so the module can still be imported and
    # the concrete classes can be defined (they just cannot be used).
    class AttentionBackend:  # type: ignore[no-redef]
        pass

    class AttentionImpl:  # type: ignore[no-redef]
        pass

    class AttentionMetadata:  # type: ignore[no-redef]
        pass

    class AttentionType:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class TurboQuantKVCacheConfig:
    """Configuration for TurboQuant KV cache in a vLLM deployment.

    This wraps the core TurboQuantConfig with vLLM-specific parameters
    (page/block layout, GPU block counts, etc.).

    Attributes:
        key_bits: Quantization bit-width for keys (default 4).
        value_bits: Quantization bit-width for values (default 2).
        mode: ``"mse"`` (Algorithm 1) or ``"prod"`` (Algorithm 2).
        rotation: Rotation method — ``"dense_qr"`` or ``"rht"``.
        block_size: vLLM paged-attention block size (tokens per block).
        num_gpu_blocks: Number of GPU blocks allocated by the engine.
        num_cpu_blocks: Number of CPU (swap) blocks.
        protected_layers: First *N* layers stored at full precision.
        seed: Random seed for rotation matrices.
    """

    key_bits: int = 4
    value_bits: int = 2
    mode: str = "mse"
    rotation: str = "dense_qr"
    block_size: int = 16
    num_gpu_blocks: int = 0
    num_cpu_blocks: int = 0
    protected_layers: int = 0
    seed: int = 42

    def to_turboquant_config(self) -> TurboQuantConfig:
        """Convert to the core TurboQuantConfig."""
        return TurboQuantConfig(
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            mode=self.mode,
            rotation=self.rotation,
            protected_layers=self.protected_layers,
            quantize_on_append=True,
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# Paged KV cache wrapper
# ---------------------------------------------------------------------------


class TurboQuantPagedKVCache:
    """Paged KV cache backed by TurboQuant quantization.

    vLLM allocates KV storage in fixed-size *blocks* (pages).  Each block
    holds ``block_size`` tokens for every layer/head.  This class mirrors
    that layout but stores quantised (packed) representations, yielding
    significant memory savings.

    The quantize-on-append pattern works as follows:
      1. ``append()`` receives freshly-computed K/V tensors for new tokens.
      2. Each tensor is quantised immediately via the TurboQuant pipeline
         (rotate -> Lloyd-Max quantise -> bit-pack).
      3. Packed codes and norms are written into the paged block table.
      4. On read, blocks are dequantised to produce the dense K/V matrices
         required by the attention kernel.
    """

    def __init__(
        self,
        config: TurboQuantKVCacheConfig,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> None:
        self.config = config
        self.tq_config = config.to_turboquant_config()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.block_size = config.block_size

        # Total capacity in tokens
        self._total_blocks = config.num_gpu_blocks
        self._max_tokens = self._total_blocks * self.block_size

        # Underlying QuantizedKVCache — one large cache pre-allocated to
        # accommodate all blocks across all layers.
        if self._max_tokens > 0:
            self._cache = QuantizedKVCache(
                config=self.tq_config,
                num_layers=num_layers,
                max_seq_len=self._max_tokens,
                num_heads=num_heads,
                head_dim=head_dim,
                device=device,
            )
        else:
            self._cache: Optional[QuantizedKVCache] = None

        # Block table: maps (seq_id, block_idx) -> global token offset.
        # Managed externally by vLLM's block manager; we just expose the
        # packed storage through standard read/write helpers.
        self._block_offsets: Dict[int, List[int]] = {}  # seq_id -> [offsets]

    # ------------------------------------------------------------------ #
    # Core read / write API
    # ------------------------------------------------------------------ #

    def allocate(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        """(Re-)allocate cache storage.  Called by vLLM's cache engine."""
        self.config.num_gpu_blocks = num_gpu_blocks
        self.config.num_cpu_blocks = num_cpu_blocks
        self._total_blocks = num_gpu_blocks
        self._max_tokens = num_gpu_blocks * self.block_size

        self._cache = QuantizedKVCache(
            config=self.tq_config,
            num_layers=num_layers,
            max_seq_len=self._max_tokens,
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
        )

    def append(
        self,
        layer_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Quantise and append K/V for new tokens into the cache.

        Args:
            layer_id: Transformer layer index.
            key: ``(num_heads, num_new_tokens, head_dim)`` or
                 ``(batch, num_heads, num_new_tokens, head_dim)``.
            value: Same shape as *key*.
        """
        if self._cache is None:
            raise RuntimeError(
                "TurboQuantPagedKVCache has not been allocated. "
                "Call allocate() first."
            )
        self._cache.append(layer_id, key, value)

    def get_keys(self, layer_id: int) -> torch.Tensor:
        """Dequantise and return all cached keys for *layer_id*.

        Returns:
            ``(num_heads, seq_len, head_dim)`` float tensor.
        """
        if self._cache is None:
            return torch.zeros(
                self.num_heads, 0, self.head_dim,
                dtype=torch.float32, device=self.device,
            )
        return self._cache._dequantize_keys(layer_id)

    def get_values(self, layer_id: int) -> torch.Tensor:
        """Dequantise and return all cached values for *layer_id*.

        Returns:
            ``(num_heads, seq_len, head_dim)`` float tensor.
        """
        if self._cache is None:
            return torch.zeros(
                self.num_heads, 0, self.head_dim,
                dtype=torch.float32, device=self.device,
            )
        return self._cache._dequantize_values(layer_id)

    def attention_scores(
        self,
        layer_id: int,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention logits from packed key storage.

        Args:
            query: ``(batch, num_heads, seq_q, head_dim)`` or
                   ``(num_heads, seq_q, head_dim)``.

        Returns:
            Scaled dot-product logits with the same leading dimensions.
        """
        if self._cache is None:
            raise RuntimeError("Cache not allocated.")
        return self._cache.attention_scores(layer_id, query)

    def attention_values(
        self,
        layer_id: int,
        attn_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted value sum from packed value storage.

        Args:
            attn_weights: Softmax attention weights.

        Returns:
            Attention output tensor.
        """
        if self._cache is None:
            raise RuntimeError("Cache not allocated.")
        return self._cache.attention_values(layer_id, attn_weights)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def get_seq_length(self, layer_id: int) -> int:
        """Number of cached tokens for *layer_id*."""
        if self._cache is None:
            return 0
        return int(self._cache.seq_lens[layer_id].item())

    def memory_bytes(self) -> int:
        """Total memory consumed by packed storage (bytes)."""
        if self._cache is None:
            return 0
        return self._cache.memory_bytes()

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs. fp16 KV cache."""
        if self._cache is None:
            return 0.0
        return self._cache.compression_ratio


# ---------------------------------------------------------------------------
# vLLM attention implementation
# ---------------------------------------------------------------------------


class TurboQuantAttentionImpl(AttentionImpl):
    """Attention implementation that operates on a TurboQuant paged KV cache.

    This is the compute kernel invoked by vLLM's ``Attention`` layer when
    the TurboQuant backend is selected.  It:
      1. Appends incoming K/V to the quantised cache.
      2. Computes Q*K^T scores from packed storage.
      3. Applies masking / softmax.
      4. Computes the weighted V sum from packed storage.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "turboquant",
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        if not _VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is required for TurboQuantAttentionImpl. "
                "Install with: pip install vllm"
            )
        super().__init__()  # type: ignore[call-arg]
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: TurboQuantPagedKVCache,
        attn_metadata: AttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: Any = None,
        output: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run one attention step through the TurboQuant KV cache.

        In vLLM the ``kv_cache`` argument is the cache object produced by
        the cache engine.  We expect a ``TurboQuantPagedKVCache``.

        Args:
            query: ``(num_tokens, num_heads, head_dim)``.
            key: ``(num_tokens, num_kv_heads, head_dim)`` — new tokens.
            value: Same shape as *key*.
            kv_cache: The paged KV cache instance.
            attn_metadata: vLLM-provided metadata (slot mapping, etc.).
            k_scale / v_scale: Optional FP8 scales (unused here).
            attn_type: Attention variant indicator.
            output: Optional pre-allocated output buffer.

        Returns:
            ``(num_tokens, num_heads, head_dim)`` attention output.
        """
        # Determine the layer_id from metadata if available, else default 0
        layer_id: int = getattr(attn_metadata, "layer_id", 0)

        num_tokens = query.shape[0]

        # Reshape to (num_heads, num_tokens, head_dim) for the cache API
        key_for_cache = key.transpose(0, 1)    # (num_kv_heads, num_tokens, head_dim)
        value_for_cache = value.transpose(0, 1)

        # 1. Append new KV to quantised cache
        kv_cache.append(layer_id, key_for_cache, value_for_cache)

        # 2. Compute attention scores from packed keys
        # query shape for attention_scores: (num_heads, seq_q, head_dim)
        query_for_attn = query.transpose(0, 1)  # (num_heads, num_tokens, head_dim)
        logits = kv_cache.attention_scores(layer_id, query_for_attn)
        # logits: (num_heads, num_tokens, seq_kv)

        # 3. Apply softmax
        attn_weights = torch.softmax(logits, dim=-1)

        # 4. Compute weighted values from packed value storage
        attn_output = kv_cache.attention_values(layer_id, attn_weights)
        # attn_output: (num_heads, num_tokens, head_dim)

        # Reshape back to (num_tokens, num_heads, head_dim)
        attn_output = attn_output.transpose(0, 1).to(query.dtype)

        if output is not None:
            output.copy_(attn_output)
            return output
        return attn_output


# ---------------------------------------------------------------------------
# vLLM attention backend
# ---------------------------------------------------------------------------


class TurboQuantAttentionBackend(AttentionBackend):
    """vLLM attention backend that routes through TurboQuant quantised KV.

    Register this backend so that vLLM uses TurboQuant compression when
    ``--kv-cache-dtype turboquant`` is specified.
    """

    @staticmethod
    def get_name() -> str:
        return "turboquant"

    @staticmethod
    def get_impl_cls() -> Type[TurboQuantAttentionImpl]:
        return TurboQuantAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type[AttentionMetadata]:
        return AttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        """Return the logical shape for one cache layer.

        For TurboQuant the actual storage is managed internally by
        ``TurboQuantPagedKVCache``; we report the *logical* shape that
        vLLM uses for bookkeeping.
        """
        # (2 for K+V, num_blocks, block_size, num_kv_heads, head_size)
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        """Head dimensions we support — essentially any reasonable size."""
        return [32, 64, 96, 128, 160, 192, 224, 256]


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


_REGISTERED = False


def register() -> None:
    """Register TurboQuant as a vLLM KV cache backend.

    This function is the entry point discovered by vLLM's plugin system
    (via the ``vllm.general_plugins`` entry-point group).

    It is safe to call multiple times — registration is idempotent.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    if not _VLLM_AVAILABLE:
        # Silently skip if vLLM is not installed — the plugin simply
        # won't be available at runtime.
        return

    try:
        # vLLM >=0.5: register via the backend registry
        from vllm.attention.selector import _Backend  # noqa: F401
        from vllm.attention import selector as _selector

        if hasattr(_selector, "register_backend"):
            _selector.register_backend(
                "turboquant", TurboQuantAttentionBackend
            )
            _REGISTERED = True
            return
    except (ImportError, AttributeError):
        pass

    try:
        # vLLM >=0.4: older registry path
        from vllm.attention.backends import backend_registry  # type: ignore[attr-defined]

        if hasattr(backend_registry, "register"):
            backend_registry.register(
                "turboquant", TurboQuantAttentionBackend
            )
            _REGISTERED = True
            return
    except (ImportError, AttributeError):
        pass

    # If neither path works, the backend class is still importable and
    # usable directly — users can wire it up manually.
    _REGISTERED = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "TurboQuantKVCacheConfig",
    "TurboQuantPagedKVCache",
    "TurboQuantAttentionBackend",
    "TurboQuantAttentionImpl",
    "register",
]
