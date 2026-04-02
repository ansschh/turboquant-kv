"""
ops.py — Python wrappers that try compiled CUDA/CPU ops, fall back to reference.

Each function mirrors the reference.py API but dispatches to the C++ extension
when available, avoiding full decompression in the hot path.
"""

from __future__ import annotations

from typing import Tuple

import torch

from turboquant_kv.reference import (
    pack_codes as _pack_codes_ref,
    unpack_codes as _unpack_codes_ref,
    quantize_mse as _quantize_mse_ref,
    dequantize_mse as _dequantize_mse_ref,
    make_rotation_matrix,
    lloyd_max_codebook,
)


def _has_turboquant_ops() -> bool:
    """Check if compiled C++ ops are available."""
    try:
        torch.ops.turboquant.pack_codes
        return True
    except (AttributeError, RuntimeError):
        return False


# Cache the check result (evaluated once at first call)
_OPS_AVAILABLE = None


def _check_ops():
    global _OPS_AVAILABLE
    if _OPS_AVAILABLE is None:
        _OPS_AVAILABLE = _has_turboquant_ops()
    return _OPS_AVAILABLE


# -------------------------------------------------------------------
# pack_codes / unpack_codes
# -------------------------------------------------------------------


def pack_codes(codes: torch.Tensor, bit_width: int) -> torch.Tensor:
    """Pack quantization codes into compact uint8 representation."""
    if _check_ops():
        try:
            return torch.ops.turboquant.pack_codes(codes, bit_width)
        except (RuntimeError, AttributeError):
            pass
    return _pack_codes_ref(codes, bit_width)


def unpack_codes(packed: torch.Tensor, bit_width: int, dim: int) -> torch.Tensor:
    """Unpack codes from packed uint8 representation."""
    if _check_ops():
        try:
            return torch.ops.turboquant.unpack_codes(packed, bit_width, dim)
        except (RuntimeError, AttributeError):
            pass
    return _unpack_codes_ref(packed, bit_width, dim)


# -------------------------------------------------------------------
# rotate_and_quantize
# -------------------------------------------------------------------


def rotate_and_quantize(
    vectors: torch.Tensor,
    rotation: torch.Tensor,
    boundaries: torch.Tensor,
    bit_width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused rotation + quantization + packing.

    Args:
        vectors: (N, d) float tensor.
        rotation: (d, d) orthogonal matrix.
        boundaries: (2^b - 1,) decision boundaries.
        bit_width: Number of bits per code.

    Returns:
        packed_codes: (N, packed_dim) uint8.
        norms: (N,) float32.
    """
    if _check_ops():
        try:
            return torch.ops.turboquant.rotate_and_quantize(
                vectors, rotation, boundaries, bit_width
            )
        except (RuntimeError, AttributeError):
            pass

    # Fall back to reference implementation
    return _quantize_mse_ref(
        vectors, bit_width, dim=vectors.shape[-1],
        rotation=rotation, codebook=(boundaries, None),
    )


# -------------------------------------------------------------------
# attention_scores_packed
# -------------------------------------------------------------------


def _has_triton() -> bool:
    """Check if Triton is available."""
    try:
        import triton
        return True
    except ImportError:
        return False


_TRITON_AVAILABLE = None


def _check_triton():
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is None:
        _TRITON_AVAILABLE = _has_triton()
    return _TRITON_AVAILABLE


def attention_scores_packed(
    query: torch.Tensor,
    packed_keys: torch.Tensor,
    key_norms: torch.Tensor,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    bit_width: int,
) -> torch.Tensor:
    """Compute attention scores directly from packed key codes.

    Args:
        query: (batch, heads, head_dim) float tensor.
        packed_keys: (seq_len, packed_dim) uint8 packed codes.
        key_norms: (seq_len,) float32 norms.
        centroids: (2^b,) float32 centroid values.
        rotation: (head_dim, head_dim) orthogonal matrix.
        bit_width: Number of bits per code.

    Returns:
        scores: (batch, heads, seq_len) float32.
    """
    # Try optimized Triton v2 first (cuBLAS-parity, 4-bit only)
    if query.is_cuda and _check_triton() and bit_width == 4:
        try:
            from .triton_kernels import triton_attention_scores_v2
            return triton_attention_scores_v2(
                query, packed_keys, key_norms, centroids, rotation, bit_width
            )
        except Exception:
            pass

    # Try general Triton kernel
    if query.is_cuda and _check_triton():
        try:
            from .triton_kernels import triton_attention_scores
            return triton_attention_scores(
                query, packed_keys, key_norms, centroids, rotation, bit_width
            )
        except Exception:
            pass

    # Try compiled C++ ops
    if _check_ops():
        try:
            return torch.ops.turboquant.attention_scores_packed(
                query, packed_keys, key_norms, centroids, rotation, bit_width
            )
        except (RuntimeError, AttributeError):
            pass

    # Fall back: unpack, lookup centroids, matmul
    head_dim = query.shape[-1]
    codes = _unpack_codes_ref(packed_keys, bit_width, head_dim)  # (seq_len, head_dim)
    centroid_vals = centroids.to(device=query.device, dtype=torch.float32)[codes.long()]

    # Rotate query
    rot = rotation.to(device=query.device, dtype=torch.float32)
    q_rot = query.float() @ rot.t()  # (batch, heads, head_dim)

    # (batch, heads, head_dim) @ (head_dim, seq_len) -> (batch, heads, seq_len)
    scores = torch.matmul(q_rot, centroid_vals.t().float())
    scores = scores * key_norms.to(device=query.device).unsqueeze(0).unsqueeze(0)

    return scores


# -------------------------------------------------------------------
# attention_values_packed
# -------------------------------------------------------------------


def attention_values_packed(
    attn_weights: torch.Tensor,
    packed_values: torch.Tensor,
    value_norms: torch.Tensor,
    centroids: torch.Tensor,
    rotation_T: torch.Tensor,
    bit_width: int,
) -> torch.Tensor:
    """Compute attention output from packed value codes.

    Args:
        attn_weights: (batch, heads, seq_len) float tensor.
        packed_values: (seq_len, packed_dim) uint8 packed codes.
        value_norms: (seq_len,) float32 norms.
        centroids: (2^b,) float32 centroid values.
        rotation_T: (head_dim, head_dim) transpose of rotation matrix.
        bit_width: Number of bits per code.

    Returns:
        output: (batch, heads, head_dim) float32.
    """
    if _check_ops():
        try:
            return torch.ops.turboquant.attention_values_packed(
                attn_weights, packed_values, value_norms, centroids, rotation_T, bit_width
            )
        except (RuntimeError, AttributeError):
            pass

    # Fall back: unpack, lookup centroids, weighted sum, inverse rotate
    head_dim = rotation_T.shape[0]
    codes = _unpack_codes_ref(packed_values, bit_width, head_dim)
    centroid_vals = centroids.to(device=attn_weights.device, dtype=torch.float32)[codes.long()]

    # Scale by norms: (seq_len, head_dim) * (seq_len, 1)
    scaled = centroid_vals * value_norms.to(device=attn_weights.device).unsqueeze(-1)

    # Weighted sum: (batch, heads, seq_len) @ (seq_len, head_dim) -> (batch, heads, head_dim)
    rotated_out = torch.matmul(attn_weights.float(), scaled.float())

    # Inverse rotation: (batch, heads, head_dim) @ (head_dim, head_dim)
    rot_T = rotation_T.to(device=attn_weights.device, dtype=torch.float32)
    output = torch.matmul(rotated_out, rot_T)

    return output
