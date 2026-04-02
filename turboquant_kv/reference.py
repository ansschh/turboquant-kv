"""
TurboQuant reference implementation — exact paper algorithms (arXiv:2504.19874).

Algorithm 1: TurboQuant_MSE — minimum MSE scalar quantization on the rotated sphere.
Algorithm 2: TurboQuant_Prod — unbiased inner-product preserving quantization via QJL correction.

All functions operate on torch tensors and support both CPU and CUDA.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple

import torch
import numpy as np
from scipy.stats import norm as scipy_norm


# ---------------------------------------------------------------------------
# Codebook: Lloyd-Max for N(0, 1) then scale by 1/sqrt(d)
# ---------------------------------------------------------------------------


def _lloyd_max_codebook_unscaled(bits: int, max_iter: int = 200, tol: float = 1e-12):
    """Compute Lloyd-Max codebook for the standard normal distribution.

    Returns (boundaries, centroids) as numpy float64 arrays.
    The boundaries array has (2^bits - 1) entries; centroids has 2^bits entries.
    """
    n_levels = 1 << bits
    centroids = np.linspace(-3.0, 3.0, n_levels)

    for _ in range(max_iter):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        edges = np.concatenate([[-np.inf], boundaries, [np.inf]])
        new_centroids = np.zeros(n_levels, dtype=np.float64)

        for i in range(n_levels):
            lo, hi = edges[i], edges[i + 1]
            prob = scipy_norm.cdf(hi) - scipy_norm.cdf(lo)
            if prob < 1e-15:
                new_centroids[i] = centroids[i]
            else:
                # E[X | lo < X < hi] = (phi(lo) - phi(hi)) / prob
                new_centroids[i] = (scipy_norm.pdf(lo) - scipy_norm.pdf(hi)) / prob

        if np.max(np.abs(new_centroids - centroids)) < tol:
            break
        centroids = new_centroids

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return boundaries, centroids


@lru_cache(maxsize=16)
def lloyd_max_codebook(bits: int, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lloyd-Max codebook for Beta distribution on the unit sphere.

    In high dimension d, each rotated coordinate follows N(0, 1/d).
    So we build the standard-normal Lloyd-Max codebook and scale by 1/sqrt(d).

    Returns:
        boundaries: shape (2^bits - 1,) float32 tensor
        centroids:  shape (2^bits,)     float32 tensor
    """
    boundaries, centroids = _lloyd_max_codebook_unscaled(bits)
    scale = 1.0 / math.sqrt(dim)
    return (
        torch.tensor(boundaries * scale, dtype=torch.float32),
        torch.tensor(centroids * scale, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Rotation matrices
# ---------------------------------------------------------------------------


def make_rotation_matrix(dim: int, seed: int = 42, method: str = "dense_qr") -> torch.Tensor:
    """Generate an orthogonal rotation matrix.

    Args:
        dim: Dimension of the rotation.
        seed: Random seed for reproducibility.
        method: "dense_qr" for exact paper method, "rht" for fast Hadamard-based.

    Returns:
        Orthogonal matrix of shape (dim, dim), float32.
    """
    if method == "dense_qr":
        return _make_dense_qr_rotation(dim, seed)
    elif method == "rht":
        return _make_rht_matrix(dim, seed)
    else:
        raise ValueError(f"Unknown rotation method: {method}")


def _make_dense_qr_rotation(dim: int, seed: int) -> torch.Tensor:
    """Exact paper rotation: QR decomposition of random Gaussian matrix."""
    rng = np.random.RandomState(seed)
    G = rng.randn(dim, dim).astype(np.float32)
    Q, R = np.linalg.qr(G)
    # Make Q unique by enforcing positive diagonal of R
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[None, :]
    return torch.from_numpy(Q.copy())


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _walsh_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard transform along the last dimension.

    x: (..., n) where n is a power of 2.
    Returns: (..., n) transformed tensor (unnormalized).
    """
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, "Last dim must be power of 2"
    h = 1
    result = x.clone()
    while h < n:
        # Process pairs at distance h
        result_view = result.view(*result.shape[:-1], n // (2 * h), 2, h)
        a = result_view[..., 0, :].clone()
        b = result_view[..., 1, :].clone()
        result_view[..., 0, :] = a + b
        result_view[..., 1, :] = a - b
        result = result_view.view(*x.shape)
        h <<= 1
    return result


def _make_rht_matrix(dim: int, seed: int) -> torch.Tensor:
    """Fast randomized Hadamard transform (RHT) as rotation.

    Pads to power of 2, applies random sign flips + normalized WHT.
    Returns: (dim, dim) orthogonal matrix.

    NOTE: This constructs the matrix explicitly for compatibility with the
    rest of the pipeline. For large dims, the implicit form should be used.
    """
    rng = torch.Generator().manual_seed(seed)
    padded = _next_power_of_2(dim)

    # Random sign vector
    signs = torch.where(
        torch.randint(0, 2, (padded,), generator=rng).bool(),
        torch.ones(padded),
        -torch.ones(padded),
    )
    D = torch.diag(signs)  # (padded, padded)

    # Build Hadamard matrix
    H = torch.ones(1, 1)
    while H.shape[0] < padded:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    H = H / math.sqrt(padded)  # normalize

    # RHT = H @ D, then crop to (dim, dim)
    M = (H @ D)[:dim, :dim]
    # Re-orthogonalize the cropped matrix via QR
    Q, R = torch.linalg.qr(M)
    signs_r = torch.sign(torch.diag(R))
    signs_r[signs_r == 0] = 1.0
    Q = Q * signs_r.unsqueeze(0)
    return Q.float()


def make_rht_rotation(dim: int, seed: int = 42) -> torch.Tensor:
    """Public API for fast Hadamard-based rotation matrix."""
    return _make_rht_matrix(dim, seed)


# ---------------------------------------------------------------------------
# Bit packing / unpacking
# ---------------------------------------------------------------------------


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack quantization codes into a compact uint8 representation.

    Args:
        codes: (N, d) uint8 tensor with values in [0, 2^bits).
        bits: Number of bits per code.

    Returns:
        Packed uint8 tensor of shape (N, bits * ceil(d / 8)).
    """
    codes = codes.to(torch.uint8)
    n, d = codes.shape
    bytes_per_plane = (d + 7) // 8
    device = codes.device

    planes = []
    for i in range(bits):
        # Extract bit i from each code
        bit_plane = ((codes >> i) & 1).to(torch.uint8)

        # Pad to multiple of 8
        if d % 8 != 0:
            pad = 8 - (d % 8)
            bit_plane = torch.nn.functional.pad(bit_plane, (0, pad))

        # Pack bits: reshape into groups of 8 and combine
        bit_plane = bit_plane.view(n, -1, 8)
        packed_plane = torch.zeros(n, bit_plane.shape[1], dtype=torch.uint8, device=device)
        for b in range(8):
            packed_plane |= bit_plane[:, :, b] << (7 - b)

        planes.append(packed_plane)

    return torch.cat(planes, dim=1)


def unpack_codes(packed: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Unpack codes from packed uint8 representation.

    Args:
        packed: (N, bits * ceil(dim / 8)) uint8 tensor.
        bits: Number of bits per code.
        dim: Original dimension.

    Returns:
        (N, dim) uint8 tensor of code indices.
    """
    n = packed.shape[0]
    bytes_per_plane = (dim + 7) // 8
    device = packed.device

    codes = torch.zeros(n, dim, dtype=torch.uint8, device=device)

    for i in range(bits):
        packed_plane = packed[:, i * bytes_per_plane : (i + 1) * bytes_per_plane]

        # Unpack each byte into 8 bits
        bit_plane = torch.zeros(n, bytes_per_plane * 8, dtype=torch.uint8, device=device)
        for b in range(8):
            bit_plane[:, b::8] = (packed_plane >> (7 - b)) & 1

        # Crop to original dim and set bit i
        codes |= bit_plane[:, :dim] << i

    return codes


# ---------------------------------------------------------------------------
# TurboQuant_MSE (Algorithm 1)
# ---------------------------------------------------------------------------


def quantize_mse(
    vectors: torch.Tensor,
    bits: int,
    dim: Optional[int] = None,
    rotation: Optional[torch.Tensor] = None,
    codebook: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    seed: int = 42,
    rotation_method: str = "dense_qr",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """TurboQuant_MSE quantization (Algorithm 1).

    Args:
        vectors: (..., d) float tensor to quantize.
        bits: Bit width b.
        dim: Dimension override (default: vectors.shape[-1]).
        rotation: Pre-computed (d, d) orthogonal matrix; computed if None.
        codebook: Pre-computed (boundaries, centroids) tuple; computed if None.
        seed: Random seed for rotation matrix.
        rotation_method: "dense_qr" or "rht".

    Returns:
        packed_codes: (N, bits * ceil(d/8)) uint8 packed code tensor.
        norms: (N,) float32 norm tensor.
    """
    orig_shape = vectors.shape
    d = dim if dim is not None else orig_shape[-1]
    vectors_flat = vectors.reshape(-1, d)
    n = vectors_flat.shape[0]
    device = vectors.device

    # 1. Extract norms and normalize to unit sphere
    norms = torch.norm(vectors_flat, dim=-1)
    unit = vectors_flat / torch.clamp(norms, min=1e-10).unsqueeze(-1)

    # 2. Rotate
    if rotation is None:
        rotation = make_rotation_matrix(d, seed=seed, method=rotation_method)
    rotation = rotation.to(device=device, dtype=vectors.dtype)
    rotated = unit @ rotation.t()  # (N, d)

    # 3. Quantize each coordinate using Lloyd-Max codebook
    if codebook is None:
        codebook = lloyd_max_codebook(bits, d)
    boundaries, centroids = codebook
    boundaries = boundaries.to(device=device, dtype=vectors.dtype)

    # searchsorted: find which bin each coordinate falls into
    # boundaries: (2^bits - 1,), rotated: (N, d)
    codes = torch.searchsorted(boundaries, rotated.contiguous())
    codes = codes.to(torch.uint8)

    # 4. Pack
    packed = pack_codes(codes, bits)

    return packed, norms


def dequantize_mse(
    packed_codes: torch.Tensor,
    norms: torch.Tensor,
    bits: int,
    dim: int,
    rotation: Optional[torch.Tensor] = None,
    codebook: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    seed: int = 42,
    rotation_method: str = "dense_qr",
) -> torch.Tensor:
    """TurboQuant_MSE dequantization (Algorithm 1 inverse).

    Args:
        packed_codes: (N, bits * ceil(dim/8)) uint8 packed codes.
        norms: (N,) float32 norms.
        bits: Bit width b.
        dim: Original dimension d.
        rotation: Pre-computed (d, d) orthogonal matrix.
        codebook: Pre-computed (boundaries, centroids) tuple.
        seed: Random seed for rotation matrix.
        rotation_method: "dense_qr" or "rht".

    Returns:
        Reconstructed vectors of shape (N, dim).
    """
    device = packed_codes.device
    dtype = norms.dtype

    # 1. Unpack codes
    codes = unpack_codes(packed_codes, bits, dim)  # (N, dim) uint8

    # 2. Lookup centroids
    if codebook is None:
        codebook = lloyd_max_codebook(bits, dim)
    _, centroids = codebook
    centroids = centroids.to(device=device, dtype=dtype)
    reconstructed_rotated = centroids[codes.long()]  # (N, dim)

    # 3. Inverse rotation
    if rotation is None:
        rotation = make_rotation_matrix(dim, seed=seed, method=rotation_method)
    rotation = rotation.to(device=device, dtype=dtype)
    reconstructed_unit = reconstructed_rotated @ rotation  # Pi^T = Pi.t(), so x = y @ Pi

    # 4. Scale by norms
    reconstructed = reconstructed_unit * norms.unsqueeze(-1)

    return reconstructed


# ---------------------------------------------------------------------------
# TurboQuant_Prod (Algorithm 2)
# ---------------------------------------------------------------------------


def _make_qjl_matrix(dim: int, n_sign_bits: int, seed: int, device: torch.device) -> torch.Tensor:
    """Generate the QJL random Gaussian projection matrix S.

    Args:
        dim: Input dimension d.
        n_sign_bits: Number of sign bits (typically = d).
        seed: Random seed.
        device: Torch device.

    Returns:
        S: (n_sign_bits, dim) float32 random Gaussian matrix (not normalized).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed + 12345)
    S = torch.randn(n_sign_bits, dim, generator=gen, dtype=torch.float32)
    return S.to(device)


def quantize_prod(
    vectors: torch.Tensor,
    bits: int,
    dim: Optional[int] = None,
    rotation: Optional[torch.Tensor] = None,
    codebook: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    S_matrix: Optional[torch.Tensor] = None,
    seed: int = 42,
    rotation_method: str = "dense_qr",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """TurboQuant_Prod quantization (Algorithm 2).

    Uses TurboQuant_MSE at (bits-1) for the base, then QJL sign bits for residual.

    Args:
        vectors: (..., d) float tensor to quantize.
        bits: Total bit budget b. MSE uses b-1 bits; 1 bit for QJL sign per coordinate.
        dim: Dimension override.
        rotation: Pre-computed rotation matrix.
        codebook: Pre-computed codebook for the MSE part (at bits-1).
        S_matrix: Pre-computed QJL projection matrix (d, d).
        seed: Random seed.
        rotation_method: "dense_qr" or "rht".

    Returns:
        mse_packed: Packed MSE codes at (bits-1) bits.
        qjl_signs: (N, ceil(d/8)) packed sign bits.
        residual_norms: (N,) float32 residual norms.
        norms: (N,) float32 original norms.
    """
    orig_shape = vectors.shape
    d = dim if dim is not None else orig_shape[-1]
    vectors_flat = vectors.reshape(-1, d)
    n = vectors_flat.shape[0]
    device = vectors.device
    dtype = vectors.dtype

    # Extract norms and normalize
    norms = torch.norm(vectors_flat, dim=-1)
    unit = vectors_flat / torch.clamp(norms, min=1e-10).unsqueeze(-1)

    # MSE quantization at (bits - 1)
    mse_bits = bits - 1
    if mse_bits < 1:
        raise ValueError(f"Prod mode requires bits >= 2, got {bits}")

    if codebook is None:
        codebook = lloyd_max_codebook(mse_bits, d)

    mse_packed, _ = quantize_mse(
        unit, mse_bits, dim=d, rotation=rotation, codebook=codebook,
        seed=seed, rotation_method=rotation_method,
    )

    # Dequantize MSE to get reconstruction
    mse_recon = dequantize_mse(
        mse_packed, torch.ones(n, device=device, dtype=dtype),
        mse_bits, d, rotation=rotation, codebook=codebook,
        seed=seed, rotation_method=rotation_method,
    )

    # Compute residual on unit sphere
    residual = unit - mse_recon
    residual_norms = torch.norm(residual, dim=-1)

    # QJL: project residual and take signs
    if S_matrix is None:
        S_matrix = _make_qjl_matrix(d, d, seed, device)
    S_matrix = S_matrix.to(device=device, dtype=dtype)

    projected = residual @ S_matrix.t()  # (N, d)
    sign_bits = (projected >= 0).to(torch.uint8)  # (N, d)

    # Pack sign bits
    qjl_signs = pack_codes(sign_bits.unsqueeze(-1) if sign_bits.dim() == 1 else sign_bits, bits=1)
    # pack_codes expects (N, d) with 1 bit => (N, ceil(d/8))

    return mse_packed, qjl_signs, residual_norms, norms


def dequantize_prod(
    mse_packed: torch.Tensor,
    qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
    norms: torch.Tensor,
    bits: int,
    dim: int,
    rotation: Optional[torch.Tensor] = None,
    codebook: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    S_matrix: Optional[torch.Tensor] = None,
    seed: int = 42,
    rotation_method: str = "dense_qr",
) -> torch.Tensor:
    """TurboQuant_Prod dequantization (Algorithm 2 inverse).

    Args:
        mse_packed: Packed MSE codes at (bits-1) bits.
        qjl_signs: (N, ceil(dim/8)) packed sign bits.
        residual_norms: (N,) residual norms.
        norms: (N,) original vector norms.
        bits: Total bit budget.
        dim: Original dimension.
        rotation: Pre-computed rotation matrix.
        codebook: Pre-computed codebook for MSE part.
        S_matrix: Pre-computed QJL matrix (dim, dim).
        seed: Random seed.
        rotation_method: "dense_qr" or "rht".

    Returns:
        Reconstructed vectors of shape (N, dim).
    """
    device = mse_packed.device
    dtype = norms.dtype
    n = norms.shape[0]

    mse_bits = bits - 1
    if codebook is None:
        codebook = lloyd_max_codebook(mse_bits, dim)

    # Reconstruct MSE part (unit vectors)
    mse_recon = dequantize_mse(
        mse_packed, torch.ones(n, device=device, dtype=dtype),
        mse_bits, dim, rotation=rotation, codebook=codebook,
        seed=seed, rotation_method=rotation_method,
    )

    # Unpack sign bits
    sign_codes = unpack_codes(qjl_signs, bits=1, dim=dim)  # (N, dim) in {0, 1}
    signs = 2.0 * sign_codes.float() - 1.0  # map to {-1, +1}

    # QJL residual reconstruction
    if S_matrix is None:
        S_matrix = _make_qjl_matrix(dim, dim, seed, device)
    S_matrix = S_matrix.to(device=device, dtype=dtype)

    # Reconstruction: sqrt(pi/2) / d * ||r|| * S^T @ sign(S @ r)
    scale = math.sqrt(math.pi / 2.0) / dim
    residual_correction = scale * residual_norms.unsqueeze(-1) * (signs @ S_matrix)  # (N, dim)

    # Combine and scale by original norm
    unit_recon = mse_recon + residual_correction
    return unit_recon * norms.unsqueeze(-1)
