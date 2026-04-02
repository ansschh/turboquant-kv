"""
QuantizedKVCache — compressed KV cache using TurboQuant.

Initial pure-Python implementation. The dequantize steps inside attention_scores
and attention_values are isolated so they can be replaced by fused CUDA kernels.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from turboquant_kv.config import TurboQuantConfig
from turboquant_kv.reference import (
    make_rotation_matrix,
    lloyd_max_codebook,
    quantize_mse,
    dequantize_mse,
    quantize_prod,
    dequantize_prod,
    pack_codes,
    unpack_codes,
    _make_qjl_matrix,
)


class QuantizedKVCache:
    """Compressed KV cache using TurboQuant quantization.

    Pre-allocates packed storage for keys and values across all layers.
    Supports append-on-the-fly quantization with per-layer configuration.
    """

    def __init__(
        self,
        config: TurboQuantConfig,
        num_layers: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: str = "cuda",
    ):
        self.config = config
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = torch.device(device)

        # Track how many tokens have been appended per layer
        self.seq_lens = torch.zeros(num_layers, dtype=torch.long)

        # Pre-compute rotation matrices and codebooks (one per head_dim, shared)
        self._key_rotation = make_rotation_matrix(
            head_dim, seed=config.seed, method=config.rotation
        ).to(self.device)
        self._val_rotation = make_rotation_matrix(
            head_dim, seed=config.seed + 1, method=config.rotation
        ).to(self.device)

        self._key_codebook = lloyd_max_codebook(config.key_bits, head_dim)
        self._val_codebook = lloyd_max_codebook(config.value_bits, head_dim)

        # For outlier channels: separate codebook at +1 bit
        self._has_outliers = config.outlier_channels > 0
        if self._has_outliers:
            self._key_codebook_outlier = lloyd_max_codebook(config.key_bits + 1, head_dim)
            self._val_codebook_outlier = lloyd_max_codebook(config.value_bits + 1, head_dim)

        # For prod mode: QJL matrices
        self._is_prod = config.mode == "prod"
        if self._is_prod:
            self._key_S = _make_qjl_matrix(head_dim, head_dim, config.seed, self.device)
            self._val_S = _make_qjl_matrix(head_dim, head_dim, config.seed + 1, self.device)
            # Codebooks for prod are at bits-1
            self._key_codebook_prod = lloyd_max_codebook(config.key_bits - 1, head_dim)
            self._val_codebook_prod = lloyd_max_codebook(config.value_bits - 1, head_dim)

        # Pre-allocate packed storage per layer
        # Keys
        key_bytes_per_token = config.key_bits * ((head_dim + 7) // 8)
        val_bytes_per_token = config.value_bits * ((head_dim + 7) // 8)

        # Storage: list of per-layer tensors
        # Each: (num_heads, max_seq_len, bytes_per_token) for packed codes
        # Plus norms: (num_heads, max_seq_len)
        self._key_packed = [
            torch.zeros(num_heads, max_seq_len, key_bytes_per_token,
                        dtype=torch.uint8, device=self.device)
            for _ in range(num_layers)
        ]
        self._key_norms = [
            torch.zeros(num_heads, max_seq_len, dtype=torch.float32, device=self.device)
            for _ in range(num_layers)
        ]
        self._val_packed = [
            torch.zeros(num_heads, max_seq_len, val_bytes_per_token,
                        dtype=torch.uint8, device=self.device)
            for _ in range(num_layers)
        ]
        self._val_norms = [
            torch.zeros(num_heads, max_seq_len, dtype=torch.float32, device=self.device)
            for _ in range(num_layers)
        ]

        # Prod mode extra storage
        if self._is_prod:
            key_sign_bytes = (head_dim + 7) // 8
            val_sign_bytes = (head_dim + 7) // 8
            self._key_qjl_signs = [
                torch.zeros(num_heads, max_seq_len, key_sign_bytes,
                            dtype=torch.uint8, device=self.device)
                for _ in range(num_layers)
            ]
            self._key_res_norms = [
                torch.zeros(num_heads, max_seq_len, dtype=torch.float32, device=self.device)
                for _ in range(num_layers)
            ]
            self._val_qjl_signs = [
                torch.zeros(num_heads, max_seq_len, val_sign_bytes,
                            dtype=torch.uint8, device=self.device)
                for _ in range(num_layers)
            ]
            self._val_res_norms = [
                torch.zeros(num_heads, max_seq_len, dtype=torch.float32, device=self.device)
                for _ in range(num_layers)
            ]

        # Protected layers: store full-precision KV
        self._protected = config.protected_layers > 0
        if self._protected:
            self._key_fp = [
                torch.zeros(num_heads, max_seq_len, head_dim,
                            dtype=torch.float16, device=self.device)
                for _ in range(config.protected_layers)
            ]
            self._val_fp = [
                torch.zeros(num_heads, max_seq_len, head_dim,
                            dtype=torch.float16, device=self.device)
                for _ in range(config.protected_layers)
            ]

    def _is_protected_layer(self, layer_id: int) -> bool:
        return self._protected and layer_id < self.config.protected_layers

    def append(self, layer_id: int, key: torch.Tensor, value: torch.Tensor):
        """Append new KV pairs for a layer.

        Args:
            layer_id: Layer index.
            key: (batch, num_heads, seq_len_new, head_dim) or (num_heads, seq_len_new, head_dim).
            value: Same shape as key.
        """
        # Normalize input shape to (num_heads, seq_len_new, head_dim)
        if key.dim() == 4:
            # (batch, num_heads, seq_len_new, head_dim) - take batch=0
            key = key[0]
            value = value[0]
        assert key.dim() == 3, f"Expected 3D key, got {key.dim()}D"

        num_heads, seq_new, head_dim = key.shape
        pos = self.seq_lens[layer_id].item()

        if self._is_protected_layer(layer_id):
            # Store full precision
            self._key_fp[layer_id][:, pos:pos + seq_new, :] = key.half()
            self._val_fp[layer_id][:, pos:pos + seq_new, :] = value.half()
            self.seq_lens[layer_id] += seq_new
            return

        # Quantize each head independently
        for h in range(num_heads):
            k_h = key[h]  # (seq_new, head_dim)
            v_h = value[h]

            if self._is_prod:
                # Prod mode quantization
                k_packed, k_signs, k_res_norms, k_norms = quantize_prod(
                    k_h, self.config.key_bits, dim=head_dim,
                    rotation=self._key_rotation, codebook=self._key_codebook_prod,
                    S_matrix=self._key_S, seed=self.config.seed,
                    rotation_method=self.config.rotation,
                )
                v_packed, v_signs, v_res_norms, v_norms = quantize_prod(
                    v_h, self.config.value_bits, dim=head_dim,
                    rotation=self._val_rotation, codebook=self._val_codebook_prod,
                    S_matrix=self._val_S, seed=self.config.seed + 1,
                    rotation_method=self.config.rotation,
                )
                self._key_qjl_signs[layer_id][h, pos:pos + seq_new] = k_signs
                self._key_res_norms[layer_id][h, pos:pos + seq_new] = k_res_norms
                self._val_qjl_signs[layer_id][h, pos:pos + seq_new] = v_signs
                self._val_res_norms[layer_id][h, pos:pos + seq_new] = v_res_norms
            else:
                # MSE mode quantization
                k_packed, k_norms = quantize_mse(
                    k_h, self.config.key_bits, dim=head_dim,
                    rotation=self._key_rotation, codebook=self._key_codebook,
                    seed=self.config.seed, rotation_method=self.config.rotation,
                )
                v_packed, v_norms = quantize_mse(
                    v_h, self.config.value_bits, dim=head_dim,
                    rotation=self._val_rotation, codebook=self._val_codebook,
                    seed=self.config.seed + 1, rotation_method=self.config.rotation,
                )

            self._key_packed[layer_id][h, pos:pos + seq_new] = k_packed
            self._key_norms[layer_id][h, pos:pos + seq_new] = k_norms
            self._val_packed[layer_id][h, pos:pos + seq_new] = v_packed
            self._val_norms[layer_id][h, pos:pos + seq_new] = v_norms

        self.seq_lens[layer_id] += seq_new

    def _dequantize_keys(self, layer_id: int) -> torch.Tensor:
        """Dequantize all keys for a layer. Returns (num_heads, seq_len, head_dim)."""
        seq_len = self.seq_lens[layer_id].item()
        if seq_len == 0:
            return torch.zeros(self.num_heads, 0, self.head_dim,
                               dtype=torch.float32, device=self.device)

        if self._is_protected_layer(layer_id):
            return self._key_fp[layer_id][:, :seq_len, :].float()

        keys = []
        for h in range(self.num_heads):
            packed = self._key_packed[layer_id][h, :seq_len]
            norms = self._key_norms[layer_id][h, :seq_len]
            if self._is_prod:
                signs = self._key_qjl_signs[layer_id][h, :seq_len]
                res_norms = self._key_res_norms[layer_id][h, :seq_len]
                k_h = dequantize_prod(
                    packed, signs, res_norms, norms,
                    self.config.key_bits, self.head_dim,
                    rotation=self._key_rotation, codebook=self._key_codebook_prod,
                    S_matrix=self._key_S, seed=self.config.seed,
                    rotation_method=self.config.rotation,
                )
            else:
                k_h = dequantize_mse(
                    packed, norms, self.config.key_bits, self.head_dim,
                    rotation=self._key_rotation, codebook=self._key_codebook,
                    seed=self.config.seed, rotation_method=self.config.rotation,
                )
            keys.append(k_h)
        return torch.stack(keys, dim=0)  # (num_heads, seq_len, head_dim)

    def _dequantize_values(self, layer_id: int) -> torch.Tensor:
        """Dequantize all values for a layer. Returns (num_heads, seq_len, head_dim)."""
        seq_len = self.seq_lens[layer_id].item()
        if seq_len == 0:
            return torch.zeros(self.num_heads, 0, self.head_dim,
                               dtype=torch.float32, device=self.device)

        if self._is_protected_layer(layer_id):
            return self._val_fp[layer_id][:, :seq_len, :].float()

        values = []
        for h in range(self.num_heads):
            packed = self._val_packed[layer_id][h, :seq_len]
            norms = self._val_norms[layer_id][h, :seq_len]
            if self._is_prod:
                signs = self._val_qjl_signs[layer_id][h, :seq_len]
                res_norms = self._val_res_norms[layer_id][h, :seq_len]
                v_h = dequantize_prod(
                    packed, signs, res_norms, norms,
                    self.config.value_bits, self.head_dim,
                    rotation=self._val_rotation, codebook=self._val_codebook_prod,
                    S_matrix=self._val_S, seed=self.config.seed + 1,
                    rotation_method=self.config.rotation,
                )
            else:
                v_h = dequantize_mse(
                    packed, norms, self.config.value_bits, self.head_dim,
                    rotation=self._val_rotation, codebook=self._val_codebook,
                    seed=self.config.seed + 1, rotation_method=self.config.rotation,
                )
            values.append(v_h)
        return torch.stack(values, dim=0)

    def attention_scores(self, layer_id: int, query: torch.Tensor) -> torch.Tensor:
        """Compute attention logits from packed keys.

        For the Python reference: dequantizes keys then computes dot products.
        CUDA path will avoid full dequantization.

        Args:
            query: (batch, num_heads, seq_q, head_dim) or (num_heads, seq_q, head_dim).

        Returns:
            Attention logits: (batch, num_heads, seq_q, seq_kv).
        """
        has_batch = query.dim() == 4
        if has_batch:
            batch = query.shape[0]
            query = query.reshape(-1, query.shape[-2], query.shape[-1])
            # query: (batch * num_heads, seq_q, head_dim)
            query = query.reshape(batch, self.num_heads, -1, self.head_dim)
        else:
            query = query.unsqueeze(0)  # add batch dim

        keys = self._dequantize_keys(layer_id)  # (num_heads, seq_kv, head_dim)
        keys = keys.unsqueeze(0).expand(query.shape[0], -1, -1, -1)

        # (batch, num_heads, seq_q, head_dim) @ (batch, num_heads, head_dim, seq_kv)
        logits = torch.matmul(query.float(), keys.transpose(-2, -1))

        scale = 1.0 / math.sqrt(self.head_dim)
        logits = logits * scale

        if not has_batch:
            logits = logits.squeeze(0)

        return logits

    def attention_values(self, layer_id: int, attn_weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of values from packed storage.

        Args:
            attn_weights: (batch, num_heads, seq_q, seq_kv) or (num_heads, seq_q, seq_kv).

        Returns:
            Output: same leading dims + (head_dim,).
        """
        has_batch = attn_weights.dim() == 4
        if not has_batch:
            attn_weights = attn_weights.unsqueeze(0)

        values = self._dequantize_values(layer_id)  # (num_heads, seq_kv, head_dim)
        values = values.unsqueeze(0).expand(attn_weights.shape[0], -1, -1, -1)

        # (batch, num_heads, seq_q, seq_kv) @ (batch, num_heads, seq_kv, head_dim)
        output = torch.matmul(attn_weights.float(), values)

        if not has_batch:
            output = output.squeeze(0)

        return output

    def memory_bytes(self) -> int:
        """Report actual memory usage of packed storage in bytes."""
        total = 0
        for layer_id in range(self.num_layers):
            seq_len = self.seq_lens[layer_id].item()
            if seq_len == 0:
                continue

            if self._is_protected_layer(layer_id):
                # fp16: 2 bytes per element
                total += 2 * self.num_heads * seq_len * self.head_dim * 2  # keys + values
                continue

            key_bytes = self.config.key_bits * ((self.head_dim + 7) // 8)
            val_bytes = self.config.value_bits * ((self.head_dim + 7) // 8)

            # Packed codes + norms (4 bytes each)
            per_token = key_bytes + val_bytes + 4 + 4  # packed + key_norm + val_norm
            if self._is_prod:
                sign_bytes = (self.head_dim + 7) // 8
                per_token += 2 * sign_bytes + 4 + 4  # key+val signs + res norms

            total += self.num_heads * seq_len * per_token

        return total

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs full fp16 KV cache."""
        fp16_bytes = 0
        for layer_id in range(self.num_layers):
            seq_len = self.seq_lens[layer_id].item()
            if seq_len == 0:
                continue
            # fp16: 2 bytes per element, keys + values
            fp16_bytes += 2 * self.num_heads * seq_len * self.head_dim * 2

        actual = self.memory_bytes()
        if actual == 0:
            return 0.0
        return fp16_bytes / actual
