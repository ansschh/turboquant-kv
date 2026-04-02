"""
HuggingFace Transformers integration for TurboQuant KV cache compression.

Provides ``TurboQuantCache``, a ``DynamicCache`` subclass (Transformers 4.36+)
that quantises KV pairs on the fly during generation.

Usage::

    from turboquant_kv.hf_integration import TurboQuantCache

    cache = TurboQuantCache(key_bits=4, value_bits=2)
    output = model.generate(input_ids, past_key_values=cache, max_new_tokens=100)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from turboquant_kv.config import TurboQuantConfig
from turboquant_kv.reference import (
    make_rotation_matrix,
    lloyd_max_codebook,
    quantize_mse,
    dequantize_mse,
    quantize_prod,
    dequantize_prod,
    _make_qjl_matrix,
)

# ---------------------------------------------------------------------------
# Conditional transformers import
# ---------------------------------------------------------------------------

_TRANSFORMERS_AVAILABLE = False
_DynamicCache: type = object  # fallback base

try:
    from transformers import DynamicCache as _HFDynamicCache

    _DynamicCache = _HFDynamicCache  # type: ignore[misc]
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Per-layer quantised storage
# ---------------------------------------------------------------------------


class _LayerQuantStore:
    """Internal storage for one transformer layer's quantised KV pairs.

    Stores packed codes + norms in lists that grow token-by-token (append
    semantics matching Transformers' DynamicCache).
    """

    def __init__(
        self,
        tq_config: TurboQuantConfig,
        num_heads: int,
        head_dim: int,
        device: torch.device,
    ) -> None:
        self.tq_config = tq_config
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.seq_len: int = 0

        # Precompute rotation / codebook for this layer
        self._key_rotation = make_rotation_matrix(
            head_dim, seed=tq_config.seed, method=tq_config.rotation,
        ).to(device)
        self._val_rotation = make_rotation_matrix(
            head_dim, seed=tq_config.seed + 1, method=tq_config.rotation,
        ).to(device)

        self._is_prod = tq_config.mode == "prod"

        if self._is_prod:
            self._key_codebook = lloyd_max_codebook(tq_config.key_bits - 1, head_dim)
            self._val_codebook = lloyd_max_codebook(tq_config.value_bits - 1, head_dim)
            self._key_S = _make_qjl_matrix(head_dim, head_dim, tq_config.seed, device)
            self._val_S = _make_qjl_matrix(head_dim, head_dim, tq_config.seed + 1, device)
        else:
            self._key_codebook = lloyd_max_codebook(tq_config.key_bits, head_dim)
            self._val_codebook = lloyd_max_codebook(tq_config.value_bits, head_dim)

        # Accumulated packed tensors — one list entry per append call.
        # Each entry has shape (num_heads, num_new_tokens, packed_dim) / (num_heads, num_new_tokens).
        self._key_packed_chunks: List[torch.Tensor] = []
        self._key_norms_chunks: List[torch.Tensor] = []
        self._val_packed_chunks: List[torch.Tensor] = []
        self._val_norms_chunks: List[torch.Tensor] = []

        # Prod-mode extras
        if self._is_prod:
            self._key_signs_chunks: List[torch.Tensor] = []
            self._key_res_norms_chunks: List[torch.Tensor] = []
            self._val_signs_chunks: List[torch.Tensor] = []
            self._val_res_norms_chunks: List[torch.Tensor] = []

    # ------------------------------------------------------------------ #
    # Append (quantise-on-write)
    # ------------------------------------------------------------------ #

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Quantise and store new KV tokens.

        Args:
            key: ``(num_heads, num_new_tokens, head_dim)`` float tensor.
            value: Same shape as *key*.
        """
        num_heads, seq_new, head_dim = key.shape
        cfg = self.tq_config

        all_k_packed: List[torch.Tensor] = []
        all_k_norms: List[torch.Tensor] = []
        all_v_packed: List[torch.Tensor] = []
        all_v_norms: List[torch.Tensor] = []

        if self._is_prod:
            all_k_signs: List[torch.Tensor] = []
            all_k_res: List[torch.Tensor] = []
            all_v_signs: List[torch.Tensor] = []
            all_v_res: List[torch.Tensor] = []

        for h in range(num_heads):
            k_h = key[h]   # (seq_new, head_dim)
            v_h = value[h]

            if self._is_prod:
                kp, ks, kr, kn = quantize_prod(
                    k_h, cfg.key_bits, dim=head_dim,
                    rotation=self._key_rotation, codebook=self._key_codebook,
                    S_matrix=self._key_S, seed=cfg.seed,
                    rotation_method=cfg.rotation,
                )
                vp, vs, vr, vn = quantize_prod(
                    v_h, cfg.value_bits, dim=head_dim,
                    rotation=self._val_rotation, codebook=self._val_codebook,
                    S_matrix=self._val_S, seed=cfg.seed + 1,
                    rotation_method=cfg.rotation,
                )
                all_k_signs.append(ks)
                all_k_res.append(kr)
                all_v_signs.append(vs)
                all_v_res.append(vr)
            else:
                kp, kn = quantize_mse(
                    k_h, cfg.key_bits, dim=head_dim,
                    rotation=self._key_rotation, codebook=self._key_codebook,
                    seed=cfg.seed, rotation_method=cfg.rotation,
                )
                vp, vn = quantize_mse(
                    v_h, cfg.value_bits, dim=head_dim,
                    rotation=self._val_rotation, codebook=self._val_codebook,
                    seed=cfg.seed + 1, rotation_method=cfg.rotation,
                )

            all_k_packed.append(kp)
            all_k_norms.append(kn)
            all_v_packed.append(vp)
            all_v_norms.append(vn)

        # Stack across heads -> (num_heads, seq_new, packed_dim)
        self._key_packed_chunks.append(torch.stack(all_k_packed, dim=0))
        self._key_norms_chunks.append(torch.stack(all_k_norms, dim=0))
        self._val_packed_chunks.append(torch.stack(all_v_packed, dim=0))
        self._val_norms_chunks.append(torch.stack(all_v_norms, dim=0))

        if self._is_prod:
            self._key_signs_chunks.append(torch.stack(all_k_signs, dim=0))
            self._key_res_norms_chunks.append(torch.stack(all_k_res, dim=0))
            self._val_signs_chunks.append(torch.stack(all_v_signs, dim=0))
            self._val_res_norms_chunks.append(torch.stack(all_v_res, dim=0))

        self.seq_len += seq_new

    # ------------------------------------------------------------------ #
    # Dequantise (read path)
    # ------------------------------------------------------------------ #

    def _dequantize_all(
        self,
        packed_chunks: List[torch.Tensor],
        norms_chunks: List[torch.Tensor],
        bits: int,
        rotation: torch.Tensor,
        codebook: Tuple[torch.Tensor, torch.Tensor],
        seed: int,
        signs_chunks: Optional[List[torch.Tensor]] = None,
        res_norms_chunks: Optional[List[torch.Tensor]] = None,
        S_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dequantise concatenated chunks -> (num_heads, total_seq, head_dim)."""
        if not packed_chunks:
            return torch.zeros(
                self.num_heads, 0, self.head_dim,
                dtype=torch.float32, device=self.device,
            )

        packed_cat = torch.cat(packed_chunks, dim=1)   # (H, total_seq, packed_dim)
        norms_cat = torch.cat(norms_chunks, dim=1)     # (H, total_seq)

        if self._is_prod and signs_chunks is not None and res_norms_chunks is not None:
            signs_cat = torch.cat(signs_chunks, dim=1)
            res_cat = torch.cat(res_norms_chunks, dim=1)
        else:
            signs_cat = None
            res_cat = None

        heads: List[torch.Tensor] = []
        for h in range(self.num_heads):
            if self._is_prod and signs_cat is not None and res_cat is not None:
                vec = dequantize_prod(
                    packed_cat[h], signs_cat[h], res_cat[h], norms_cat[h],
                    bits, self.head_dim,
                    rotation=rotation, codebook=codebook,
                    S_matrix=S_matrix, seed=seed,
                    rotation_method=self.tq_config.rotation,
                )
            else:
                vec = dequantize_mse(
                    packed_cat[h], norms_cat[h],
                    bits, self.head_dim,
                    rotation=rotation, codebook=codebook,
                    seed=seed, rotation_method=self.tq_config.rotation,
                )
            heads.append(vec)

        return torch.stack(heads, dim=0)

    def get_keys(self) -> torch.Tensor:
        """Dequantise all cached keys -> (num_heads, seq_len, head_dim)."""
        return self._dequantize_all(
            self._key_packed_chunks,
            self._key_norms_chunks,
            self.tq_config.key_bits if not self._is_prod else self.tq_config.key_bits,
            self._key_rotation,
            self._key_codebook,
            self.tq_config.seed,
            signs_chunks=getattr(self, "_key_signs_chunks", None),
            res_norms_chunks=getattr(self, "_key_res_norms_chunks", None),
            S_matrix=getattr(self, "_key_S", None),
        )

    def get_values(self) -> torch.Tensor:
        """Dequantise all cached values -> (num_heads, seq_len, head_dim)."""
        return self._dequantize_all(
            self._val_packed_chunks,
            self._val_norms_chunks,
            self.tq_config.value_bits if not self._is_prod else self.tq_config.value_bits,
            self._val_rotation,
            self._val_codebook,
            self.tq_config.seed + 1,
            signs_chunks=getattr(self, "_val_signs_chunks", None),
            res_norms_chunks=getattr(self, "_val_res_norms_chunks", None),
            S_matrix=getattr(self, "_val_S", None),
        )


# ---------------------------------------------------------------------------
# TurboQuantCache — HuggingFace DynamicCache subclass
# ---------------------------------------------------------------------------


class TurboQuantCache(_DynamicCache):  # type: ignore[misc]
    """HuggingFace-compatible KV cache with TurboQuant compression.

    Drop-in replacement for ``DynamicCache`` (Transformers 4.36+).  Keys
    and values are quantised on every ``update()`` call and dequantised
    transparently when the attention layer reads them back.

    Example::

        cache = TurboQuantCache(key_bits=4, value_bits=2)
        output = model.generate(input_ids, past_key_values=cache, max_new_tokens=100)
    """

    def __init__(
        self,
        key_bits: int = 4,
        value_bits: int = 2,
        mode: str = "mse",
        rotation: str = "dense_qr",
        protected_layers: int = 0,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        if _TRANSFORMERS_AVAILABLE:
            super().__init__(**kwargs)
        self.tq_config = TurboQuantConfig(
            key_bits=key_bits,
            value_bits=value_bits,
            mode=mode,
            rotation=rotation,
            protected_layers=protected_layers,
            quantize_on_append=True,
            seed=seed,
        )
        self._protected_layers = protected_layers

        # Per-layer quantised stores (created lazily on first update)
        self._layer_stores: Dict[int, _LayerQuantStore] = {}

        # Per-layer full-precision storage for protected layers
        self._fp_key_cache: Dict[int, List[torch.Tensor]] = {}
        self._fp_value_cache: Dict[int, List[torch.Tensor]] = {}

    # ------------------------------------------------------------------ #
    # DynamicCache interface
    # ------------------------------------------------------------------ #

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantise and store new K/V, then return full (dequantised) KV.

        Args:
            key_states: ``(batch, num_kv_heads, seq_new, head_dim)``.
            value_states: Same shape.
            layer_idx: Transformer layer index.
            cache_kwargs: Extra kwargs (unused).

        Returns:
            Tuple of ``(all_keys, all_values)`` each shaped
            ``(batch, num_kv_heads, total_seq, head_dim)``.
        """
        batch_size, num_heads, seq_new, head_dim = key_states.shape
        device = key_states.device

        # Protected layers: store at full precision (no quantisation)
        if layer_idx < self._protected_layers:
            if layer_idx not in self._fp_key_cache:
                self._fp_key_cache[layer_idx] = []
                self._fp_value_cache[layer_idx] = []
            self._fp_key_cache[layer_idx].append(key_states)
            self._fp_value_cache[layer_idx].append(value_states)
            all_keys = torch.cat(self._fp_key_cache[layer_idx], dim=2)
            all_values = torch.cat(self._fp_value_cache[layer_idx], dim=2)
            return all_keys, all_values

        # Quantised path
        # Lazily create per-layer store
        if layer_idx not in self._layer_stores:
            self._layer_stores[layer_idx] = _LayerQuantStore(
                tq_config=self.tq_config,
                num_heads=num_heads,
                head_dim=head_dim,
                device=device,
            )

        store = self._layer_stores[layer_idx]

        # Process each batch item independently (quantisation is per-head,
        # and the cache is logically per-sequence).
        # For simplicity we handle batch=1 natively and loop for batch>1.
        results_k: List[torch.Tensor] = []
        results_v: List[torch.Tensor] = []

        for b in range(batch_size):
            if batch_size == 1:
                # Get dequantized OLD tokens BEFORE appending new ones
                if store.seq_len > 0:
                    k_old = store.get_keys()    # (num_heads, old_seq, head_dim)
                    v_old = store.get_values()
                else:
                    k_old = torch.zeros(num_heads, 0, head_dim, device=device, dtype=key_states.dtype)
                    v_old = torch.zeros(num_heads, 0, head_dim, device=device, dtype=value_states.dtype)

                # Append new tokens to quantized store
                store.append(key_states[b], value_states[b])

                # Concatenate: old (dequantized) + new (exact, no quantization error)
                # This prevents the current token from suffering quantization loss
                k_new_exact = key_states[b]    # (num_heads, seq_new, head_dim)
                v_new_exact = value_states[b]
                k_all = torch.cat([k_old.to(k_new_exact.dtype), k_new_exact], dim=1)
                v_all = torch.cat([v_old.to(v_new_exact.dtype), v_new_exact], dim=1)
                results_k.append(k_all)
                results_v.append(v_all)
            else:
                tmp_store_key = f"_batch_store_{layer_idx}_{b}"
                if not hasattr(self, tmp_store_key):
                    setattr(
                        self,
                        tmp_store_key,
                        _LayerQuantStore(
                            tq_config=self.tq_config,
                            num_heads=num_heads,
                            head_dim=head_dim,
                            device=device,
                        ),
                    )
                bstore: _LayerQuantStore = getattr(self, tmp_store_key)
                if bstore.seq_len > 0:
                    k_old = bstore.get_keys()
                    v_old = bstore.get_values()
                else:
                    k_old = torch.zeros(num_heads, 0, head_dim, device=device, dtype=key_states.dtype)
                    v_old = torch.zeros(num_heads, 0, head_dim, device=device, dtype=value_states.dtype)
                bstore.append(key_states[b], value_states[b])
                k_all = torch.cat([k_old.to(key_states.dtype), key_states[b]], dim=1)
                v_all = torch.cat([v_old.to(value_states.dtype), value_states[b]], dim=1)
                results_k.append(k_all)
                results_v.append(v_all)

        # Stack back to (batch, num_heads, total_seq, head_dim)
        all_keys = torch.stack(results_k, dim=0).to(key_states.dtype)
        all_values = torch.stack(results_v, dim=0).to(value_states.dtype)

        return all_keys, all_values

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the current sequence length for *layer_idx*."""
        if layer_idx < self._protected_layers:
            chunks = self._fp_key_cache.get(layer_idx, [])
            if not chunks:
                return 0
            return sum(c.shape[2] for c in chunks)
        store = self._layer_stores.get(layer_idx)
        if store is None:
            return 0
        return store.seq_len

    def get_max_length(self) -> Optional[int]:
        """Maximum cache length (None = unbounded)."""
        return None

    def get_usable_length(
        self, new_seq_length: int, layer_idx: int = 0
    ) -> int:
        """Return usable length, accounting for already cached tokens."""
        return self.get_seq_length(layer_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder cache for beam search.

        For quantised layers this requires dequantising, reordering, and
        re-quantising.  Protected (full-precision) layers are reordered
        directly.
        """
        # Protected layers
        for layer_idx in list(self._fp_key_cache.keys()):
            new_k = [c.index_select(0, beam_idx.to(c.device)) for c in self._fp_key_cache[layer_idx]]
            new_v = [c.index_select(0, beam_idx.to(c.device)) for c in self._fp_value_cache[layer_idx]]
            self._fp_key_cache[layer_idx] = new_k
            self._fp_value_cache[layer_idx] = new_v

        # Quantised layers: beam search with quantised cache is expensive.
        # We log a warning and skip reorder for quantised layers (beam
        # search quality may degrade; greedy / sampling is recommended).
        if self._layer_stores:
            import warnings
            warnings.warn(
                "TurboQuantCache.reorder_cache: beam search reordering is "
                "not fully supported for quantised layers. Results may be "
                "incorrect. Use greedy or sampling decoding instead.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(
        cls,
        key_bits: int = 4,
        value_bits: int = 2,
        mode: str = "mse",
        rotation: str = "dense_qr",
        protected_layers: int = 0,
        seed: int = 42,
    ) -> "TurboQuantCache":
        """Construct a ``TurboQuantCache`` from explicit parameters.

        This is a convenience factory; the constructor accepts the same
        arguments directly.

        Returns:
            A new ``TurboQuantCache`` instance.
        """
        return cls(
            key_bits=key_bits,
            value_bits=value_bits,
            mode=mode,
            rotation=rotation,
            protected_layers=protected_layers,
            seed=seed,
        )

    @property
    def seen_tokens(self) -> int:
        """Total tokens seen (across all layers, take max)."""
        lengths = [self.get_seq_length(i) for i in range(max(len(self._layer_stores), 1))]
        if self._fp_key_cache:
            lengths.extend(
                self.get_seq_length(i) for i in self._fp_key_cache
            )
        return max(lengths) if lengths else 0

    def memory_bytes(self) -> int:
        """Estimated memory used by quantised storage (bytes).

        Does *not* include protected-layer full-precision storage.
        """
        total = 0
        for store in self._layer_stores.values():
            for chunks in (store._key_packed_chunks, store._val_packed_chunks):
                for c in chunks:
                    total += c.nelement() * c.element_size()
            for chunks in (store._key_norms_chunks, store._val_norms_chunks):
                for c in chunks:
                    total += c.nelement() * c.element_size()
        return total

    def __len__(self) -> int:
        """Number of layers with cached data."""
        return len(self._layer_stores) + len(self._fp_key_cache)

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Index access for compatibility with code that treats cache as a list.

        Returns dequantised (key, value) for the given layer, each
        ``(1, num_heads, seq_len, head_dim)`` with a leading batch dim.
        """
        if layer_idx < self._protected_layers and layer_idx in self._fp_key_cache:
            k = torch.cat(self._fp_key_cache[layer_idx], dim=2)
            v = torch.cat(self._fp_value_cache[layer_idx], dim=2)
            return k, v

        store = self._layer_stores.get(layer_idx)
        if store is None:
            raise IndexError(f"No cached data for layer {layer_idx}")
        k = store.get_keys().unsqueeze(0)   # (1, H, S, D)
        v = store.get_values().unsqueeze(0)
        return k, v

    def __iter__(self):
        """Iterate over layers, yielding (key, value) tuples."""
        all_indices = sorted(
            set(self._layer_stores.keys()) | set(self._fp_key_cache.keys())
        )
        for idx in all_indices:
            yield self[idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "TurboQuantCache",
]
