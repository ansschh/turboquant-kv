"""
Entropy-coded storage for TurboQuant codes.

Implements Huffman coding for compressed on-disk storage of quantization
code indices, as described in Section 3.1 of arXiv:2504.19874
("Entropy Encoding Codebook Pointers").

The codeword probabilities are known a priori from the Beta distribution
(equivalently, N(0,1) in high dimension) and the Lloyd-Max Voronoi cells.
At b=4 the entropy is ~3.8 bits (vs 4 fixed-width), yielding ~5% savings.

This module is for STORAGE/DISK only. The in-memory attention kernels
continue to use fixed-width bit-packed codes for bandwidth-optimal GPU
access. Entropy coding is applied when saving/loading indexes to disk.
"""

from __future__ import annotations

import heapq
import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import norm as scipy_norm

from turboquant_kv.reference import _lloyd_max_codebook_unscaled


# ---------------------------------------------------------------------------
# Codeword probability computation
# ---------------------------------------------------------------------------


def compute_codeword_probabilities(bits: int, dim: int) -> np.ndarray:
    """Compute the probability of each codeword under the source distribution.

    Each rotated coordinate follows N(0, 1/sqrt(d)), but since Lloyd-Max
    boundaries and centroids are computed for N(0,1) then scaled by
    1/sqrt(d), we can compute probabilities directly from the unscaled
    N(0,1) boundaries.

    For each of the 2^b codewords, p_k = Phi(boundary_{k+1}) - Phi(boundary_k),
    where boundaries include -inf and +inf at the ends.

    Args:
        bits: Quantization bit width b.
        dim: Vector dimension d (used to derive the codebook, but the
             probabilities depend only on the standard-normal Voronoi
             cells, which are dimension-independent).

    Returns:
        Array of shape (2^bits,) with probabilities summing to 1.
    """
    n_levels = 1 << bits
    boundaries, _ = _lloyd_max_codebook_unscaled(bits)

    # Extend boundaries with -inf and +inf
    edges = np.concatenate([[-np.inf], boundaries, [np.inf]])
    probs = np.zeros(n_levels, dtype=np.float64)
    for i in range(n_levels):
        probs[i] = scipy_norm.cdf(edges[i + 1]) - scipy_norm.cdf(edges[i])

    # Normalize to handle floating-point drift
    probs /= probs.sum()
    return probs


# ---------------------------------------------------------------------------
# Huffman tree and coder
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _HuffmanNode:
    """Internal node for priority-queue-based Huffman tree construction."""
    freq: float
    # Tie-breaking counter to ensure stable ordering of equal-frequency nodes
    counter: int = field(compare=True)
    symbol: Optional[int] = field(default=None, compare=False)
    left: Optional["_HuffmanNode"] = field(default=None, compare=False, repr=False)
    right: Optional["_HuffmanNode"] = field(default=None, compare=False, repr=False)


class HuffmanCoder:
    """Huffman coder built from known codeword probabilities.

    Constructs an optimal prefix-free binary code from the probability
    distribution of TurboQuant code indices. Since the distribution is
    fixed (determined by bits and the Lloyd-Max quantizer), the codebook
    can be shared between encoder and decoder without transmitting it.

    Args:
        bits: Quantization bit width.
        dim: Vector dimension (for computing probabilities).

    Attributes:
        codebook: Dict mapping symbol (int) -> bitstring (str of '0'/'1').
        probs: Array of codeword probabilities.
    """

    def __init__(self, bits: int, dim: int):
        self.bits = bits
        self.dim = dim
        self.probs = compute_codeword_probabilities(bits, dim)
        self.n_levels = 1 << bits

        # Build Huffman tree
        self._root = self._build_tree()
        self.codebook: Dict[int, str] = {}
        self._build_codebook(self._root, "")

        # Ensure all symbols have a code (even zero-prob ones get long codes)
        for i in range(self.n_levels):
            if i not in self.codebook:
                # Assign a very long code for essentially impossible symbols
                self.codebook[i] = "0" * (self.bits + 8)

        # Build reverse lookup for decoding
        self._decode_table: Dict[str, int] = {
            code: sym for sym, code in self.codebook.items()
        }

    def _build_tree(self) -> _HuffmanNode:
        """Build a Huffman tree from the codeword probabilities."""
        counter = 0
        heap: List[_HuffmanNode] = []
        for i in range(self.n_levels):
            if self.probs[i] > 0:
                heapq.heappush(heap, _HuffmanNode(self.probs[i], counter, symbol=i))
                counter += 1

        # Edge case: single symbol
        if len(heap) == 1:
            node = heapq.heappop(heap)
            root = _HuffmanNode(node.freq, counter, left=node)
            return root

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = _HuffmanNode(
                left.freq + right.freq, counter, left=left, right=right
            )
            counter += 1
            heapq.heappush(heap, merged)

        return heap[0]

    def _build_codebook(self, node: _HuffmanNode, prefix: str):
        """Recursively assign codes by traversing the Huffman tree."""
        if node.symbol is not None:
            self.codebook[node.symbol] = prefix if prefix else "0"
            return
        if node.left is not None:
            self._build_codebook(node.left, prefix + "0")
        if node.right is not None:
            self._build_codebook(node.right, prefix + "1")

    def encode(self, codes: torch.Tensor) -> bytes:
        """Encode a tensor of uint8 code indices to a compressed bitstream.

        The output format is:
            [4 bytes: n_symbols as uint32 LE] [packed bits, padded to byte boundary]

        Args:
            codes: 1-D tensor of uint8 code indices in [0, 2^bits).

        Returns:
            Compressed byte string.
        """
        codes_np = codes.detach().cpu().numpy().astype(np.uint8).ravel()
        n_symbols = len(codes_np)

        # Build bitstring
        bits_list = []
        for sym in codes_np:
            bits_list.append(self.codebook[int(sym)])
        bitstring = "".join(bits_list)

        # Pack into bytes
        n_bits = len(bitstring)
        # Pad to byte boundary
        pad = (8 - n_bits % 8) % 8
        bitstring += "0" * pad

        packed = bytearray()
        for i in range(0, len(bitstring), 8):
            packed.append(int(bitstring[i:i + 8], 2))

        # Header: n_symbols (4 bytes LE) + n_bits (4 bytes LE)
        header = struct.pack("<II", n_symbols, n_bits)
        return bytes(header) + bytes(packed)

    def decode(self, data: bytes, n_symbols: Optional[int] = None) -> torch.Tensor:
        """Decode a compressed bitstream back to uint8 code indices.

        Args:
            data: Compressed byte string as produced by ``encode()``.
            n_symbols: Number of symbols to decode. If None, read from header.

        Returns:
            1-D uint8 tensor of code indices.
        """
        # Parse header
        header_size = 8
        stored_n, stored_nbits = struct.unpack("<II", data[:header_size])
        if n_symbols is None:
            n_symbols = stored_n

        # Unpack bits
        packed_bytes = data[header_size:]
        bitstring = "".join(format(b, "08b") for b in packed_bytes)
        bitstring = bitstring[:stored_nbits]  # remove padding

        # Decode using prefix-free property
        result = []
        pos = 0
        current = ""
        decoded = 0
        while decoded < n_symbols and pos < len(bitstring):
            current += bitstring[pos]
            pos += 1
            if current in self._decode_table:
                result.append(self._decode_table[current])
                current = ""
                decoded += 1

        return torch.tensor(result, dtype=torch.uint8)

    def compression_ratio(self) -> float:
        """Theoretical compression ratio (fixed-width bits / entropy).

        Returns:
            Ratio > 1 means Huffman is smaller; e.g. 1.05 means ~5% savings.
        """
        entropy = self.entropy()
        if entropy == 0:
            return 1.0
        return self.bits / entropy

    def entropy(self) -> float:
        """Shannon entropy of the source distribution in bits."""
        h = 0.0
        for p in self.probs:
            if p > 0:
                h -= p * math.log2(p)
        return h

    def avg_code_length(self) -> float:
        """Average Huffman code length in bits."""
        avg = 0.0
        for i in range(self.n_levels):
            avg += self.probs[i] * len(self.codebook[i])
        return avg

    def __repr__(self) -> str:
        return (
            f"HuffmanCoder(bits={self.bits}, dim={self.dim}, "
            f"entropy={self.entropy():.3f}, "
            f"avg_len={self.avg_code_length():.3f})"
        )


# ---------------------------------------------------------------------------
# Entropy-packed storage
# ---------------------------------------------------------------------------


class EntropyPackedStorage:
    """Compressed storage for TurboQuant codes using Huffman coding.

    Wraps a ``HuffmanCoder`` to compress/decompress code tensors for disk
    storage. The in-memory attention path should still use the fixed-width
    ``pack_codes``/``unpack_codes`` from ``reference.py``.

    Example::

        storage = EntropyPackedStorage.from_codes(codes, bits=4, dim=128)
        print(f"Saved {storage.savings_pct:.1f}% vs fixed-width")
        recovered = storage.to_codes()
        assert (recovered == codes).all()

    Attributes:
        compressed_data: The raw compressed bytes.
        shape: Original tensor shape.
        bits: Quantization bit width.
        dim: Vector dimension.
    """

    def __init__(
        self,
        compressed_data: bytes,
        shape: Tuple[int, ...],
        bits: int,
        dim: int,
        coder: HuffmanCoder,
    ):
        self._data = compressed_data
        self.shape = shape
        self.bits = bits
        self.dim = dim
        self._coder = coder

    @classmethod
    def from_codes(
        cls,
        codes: torch.Tensor,
        bits: int,
        dim: int,
    ) -> "EntropyPackedStorage":
        """Compress a tensor of code indices using Huffman coding.

        Args:
            codes: Tensor of uint8 code indices with values in [0, 2^bits).
                   Can be any shape; will be flattened for compression.
            bits: Quantization bit width.
            dim: Vector dimension (needed for probability computation).

        Returns:
            An ``EntropyPackedStorage`` instance holding the compressed data.
        """
        coder = HuffmanCoder(bits, dim)
        flat = codes.detach().cpu().to(torch.uint8).reshape(-1)
        compressed = coder.encode(flat)
        return cls(compressed, tuple(codes.shape), bits, dim, coder)

    def to_codes(self) -> torch.Tensor:
        """Decompress and return the original code tensor.

        Returns:
            uint8 tensor with the original shape.
        """
        flat = self._coder.decode(self._data)
        return flat.reshape(self.shape)

    @property
    def nbytes(self) -> int:
        """Actual compressed size in bytes (including header)."""
        return len(self._data)

    @property
    def fixed_width_nbytes(self) -> int:
        """Size that fixed-width bit packing would use for the same data.

        This counts just the raw code bits (no header overhead), matching
        what the bit-plane packed format uses.
        """
        n_elements = 1
        for s in self.shape:
            n_elements *= s
        return (n_elements * self.bits + 7) // 8

    @property
    def savings_pct(self) -> float:
        """Percentage saved vs fixed-width bit packing.

        Positive means Huffman is smaller. Accounts for the 8-byte header.
        """
        fixed = self.fixed_width_nbytes
        if fixed == 0:
            return 0.0
        return 100.0 * (1.0 - self.nbytes / fixed)

    def __repr__(self) -> str:
        return (
            f"EntropyPackedStorage("
            f"shape={self.shape}, bits={self.bits}, "
            f"compressed={self.nbytes} bytes, "
            f"savings={self.savings_pct:.1f}%)"
        )
