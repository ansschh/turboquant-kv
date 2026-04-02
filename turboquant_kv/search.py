"""
TurboQuantIndex — vector search using TurboQuant compression.

Brute-force search over TurboQuant-compressed vectors. Supports both MSE and
Prod modes, incremental insertion, and save/load to disk.
"""

from __future__ import annotations

import io
import struct
from typing import Optional, Tuple

import torch

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


HEADER_FORMAT = "<BII"  # bit_width(u8), dim(u32), n_vectors(u32)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# Mode byte: 0 = mse, 1 = prod
MODE_MSE = 0
MODE_PROD = 1


class TurboQuantIndex:
    """Compressed vector index for approximate nearest neighbor search.

    Uses TurboQuant to compress database vectors and performs brute-force
    search over the compressed representation.
    """

    def __init__(
        self,
        dim: int,
        bit_width: int = 3,
        mode: str = "mse",
        rotation: str = "dense_qr",
        seed: int = 42,
    ):
        self.dim = dim
        self.bit_width = bit_width
        self.mode = mode
        self.rotation_method = rotation
        self.seed = seed

        # Pre-compute rotation and codebook
        self._rotation = make_rotation_matrix(dim, seed=seed, method=rotation)

        if mode == "mse":
            self._codebook = lloyd_max_codebook(bit_width, dim)
        elif mode == "prod":
            self._codebook = lloyd_max_codebook(bit_width - 1, dim)
            self._S_matrix = _make_qjl_matrix(dim, dim, seed, torch.device("cpu"))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Storage (initialized empty)
        self._packed_codes: Optional[torch.Tensor] = None
        self._norms: Optional[torch.Tensor] = None
        self.n_vectors: int = 0

        # Prod-mode extra storage
        self._qjl_signs: Optional[torch.Tensor] = None
        self._residual_norms: Optional[torch.Tensor] = None

    @classmethod
    def from_vectors(
        cls,
        vectors: torch.Tensor,
        bit_width: int = 3,
        mode: str = "mse",
        rotation: str = "dense_qr",
        seed: int = 42,
    ) -> "TurboQuantIndex":
        """Create an index from a batch of vectors.

        Args:
            vectors: (N, dim) float tensor.
            bit_width: Bits per coordinate.
            mode: "mse" or "prod".
            rotation: "dense_qr" or "rht".
            seed: Random seed.

        Returns:
            Populated TurboQuantIndex.
        """
        dim = vectors.shape[-1]
        idx = cls(dim=dim, bit_width=bit_width, mode=mode, rotation=rotation, seed=seed)
        idx.add(vectors)
        return idx

    def add(self, vectors: torch.Tensor):
        """Add vectors to the index.

        Args:
            vectors: (N, dim) or (dim,) float tensor.
        """
        if vectors.dim() == 1:
            vectors = vectors.unsqueeze(0)
        assert vectors.shape[-1] == self.dim

        if self.mode == "mse":
            packed, norms = quantize_mse(
                vectors, self.bit_width, dim=self.dim,
                rotation=self._rotation, codebook=self._codebook,
                seed=self.seed, rotation_method=self.rotation_method,
            )
            if self._packed_codes is None:
                self._packed_codes = packed
                self._norms = norms
            else:
                self._packed_codes = torch.cat([self._packed_codes, packed], dim=0)
                self._norms = torch.cat([self._norms, norms], dim=0)

        elif self.mode == "prod":
            mse_packed, qjl_signs, res_norms, norms = quantize_prod(
                vectors, self.bit_width, dim=self.dim,
                rotation=self._rotation, codebook=self._codebook,
                S_matrix=self._S_matrix, seed=self.seed,
                rotation_method=self.rotation_method,
            )
            if self._packed_codes is None:
                self._packed_codes = mse_packed
                self._norms = norms
                self._qjl_signs = qjl_signs
                self._residual_norms = res_norms
            else:
                self._packed_codes = torch.cat([self._packed_codes, mse_packed], dim=0)
                self._norms = torch.cat([self._norms, norms], dim=0)
                self._qjl_signs = torch.cat([self._qjl_signs, qjl_signs], dim=0)
                self._residual_norms = torch.cat([self._residual_norms, res_norms], dim=0)

        self.n_vectors += vectors.shape[0]

    def search(self, queries: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Brute-force search over compressed vectors.

        Args:
            queries: (Q, dim) or (dim,) float tensor.
            k: Number of nearest neighbors.

        Returns:
            scores: (Q, k) inner product scores.
            indices: (Q, k) indices of nearest neighbors.
        """
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)
        assert self.n_vectors > 0, "Index is empty"

        if self.mode == "mse":
            return self._search_mse(queries, k)
        else:
            return self._search_prod(queries, k)

    def _search_mse(self, queries: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """MSE mode search: rotate query, dot with centroid lookup table."""
        device = queries.device
        rotation = self._rotation.to(device=device, dtype=queries.dtype)
        _, centroids = self._codebook
        centroids = centroids.to(device=device, dtype=queries.dtype)

        # Rotate queries
        q_rot = queries @ rotation.t()  # (Q, dim)

        # Unpack database codes and lookup centroids
        codes = unpack_codes(self._packed_codes.to(device), self.bit_width, self.dim)
        centroid_vals = centroids[codes.long()]  # (N, dim)

        # Compute IP scores: (Q, dim) @ (dim, N) = (Q, N)
        scores = q_rot @ centroid_vals.t()
        # Scale by database norms
        scores = scores * self._norms.to(device).unsqueeze(0)

        k = min(k, self.n_vectors)
        top_idx = torch.topk(scores, k, dim=-1)
        return top_idx.values, top_idx.indices

    def _search_prod(self, queries: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prod mode search: MSE scores + QJL residual correction."""
        device = queries.device
        dtype = queries.dtype

        # Dequantize all vectors (brute force for now)
        recon = dequantize_prod(
            self._packed_codes.to(device),
            self._qjl_signs.to(device),
            self._residual_norms.to(device),
            self._norms.to(device),
            self.bit_width, self.dim,
            rotation=self._rotation.to(device),
            codebook=self._codebook,
            S_matrix=self._S_matrix.to(device),
            seed=self.seed,
            rotation_method=self.rotation_method,
        )

        scores = queries.float() @ recon.t()  # (Q, N)

        k = min(k, self.n_vectors)
        top_idx = torch.topk(scores, k, dim=-1)
        return top_idx.values, top_idx.indices

    def save(self, path: str):
        """Save index to a binary file.

        Format: header + mode_byte + packed_codes + norms [+ qjl_signs + residual_norms]
        """
        mode_byte = MODE_MSE if self.mode == "mse" else MODE_PROD
        header = struct.pack(HEADER_FORMAT, self.bit_width, self.dim, self.n_vectors)

        with open(path, "wb") as f:
            f.write(header)
            f.write(struct.pack("<B", mode_byte))
            f.write(self._packed_codes.cpu().numpy().tobytes())
            f.write(self._norms.cpu().numpy().tobytes())
            if self.mode == "prod":
                f.write(self._qjl_signs.cpu().numpy().tobytes())
                f.write(self._residual_norms.cpu().numpy().tobytes())

    @classmethod
    def load(cls, path: str, rotation: str = "dense_qr", seed: int = 42) -> "TurboQuantIndex":
        """Load index from a binary file."""
        import numpy as np

        with open(path, "rb") as f:
            header = struct.unpack(HEADER_FORMAT, f.read(HEADER_SIZE))
            bit_width, dim, n_vectors = header
            mode_byte = struct.unpack("<B", f.read(1))[0]
            mode = "mse" if mode_byte == MODE_MSE else "prod"

            if mode == "mse":
                packed_bytes = bit_width * ((dim + 7) // 8)
            else:
                packed_bytes = (bit_width - 1) * ((dim + 7) // 8)

            packed = np.frombuffer(f.read(packed_bytes * n_vectors), dtype=np.uint8)
            packed = torch.from_numpy(packed.copy()).reshape(n_vectors, -1)

            norms = np.frombuffer(f.read(n_vectors * 4), dtype=np.float32)
            norms = torch.from_numpy(norms.copy())

            qjl_signs = None
            residual_norms = None
            if mode == "prod":
                sign_bytes = (dim + 7) // 8
                signs_raw = np.frombuffer(f.read(sign_bytes * n_vectors), dtype=np.uint8)
                qjl_signs = torch.from_numpy(signs_raw.copy()).reshape(n_vectors, -1)

                res_raw = np.frombuffer(f.read(n_vectors * 4), dtype=np.float32)
                residual_norms = torch.from_numpy(res_raw.copy())

        idx = cls(dim=dim, bit_width=bit_width, mode=mode, rotation=rotation, seed=seed)
        idx._packed_codes = packed
        idx._norms = norms
        idx._qjl_signs = qjl_signs
        idx._residual_norms = residual_norms
        idx.n_vectors = n_vectors

        return idx
