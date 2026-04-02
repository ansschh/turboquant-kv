"""
Micro-benchmark: measure quantize/dequantize throughput at various (d, b, N) settings.

Usage:
    python -m benchmarks.micro.bench_quantize
    python benchmarks/micro/bench_quantize.py
"""

import sys
import time
from pathlib import Path

import torch

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from turboquant_kv.reference import (
    quantize_mse,
    dequantize_mse,
    quantize_prod,
    dequantize_prod,
    make_rotation_matrix,
    lloyd_max_codebook,
)


def bench_quantize_mse(n: int, d: int, bits: int, device: str, n_warmup: int = 3, n_iters: int = 10):
    """Benchmark MSE quantize throughput."""
    vectors = torch.randn(n, d, device=device)
    rotation = make_rotation_matrix(d, seed=42).to(device)
    codebook = lloyd_max_codebook(bits, d)

    # Warmup
    for _ in range(n_warmup):
        quantize_mse(vectors, bits, dim=d, rotation=rotation, codebook=codebook)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        packed, norms = quantize_mse(vectors, bits, dim=d, rotation=rotation, codebook=codebook)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters

    throughput = n / elapsed
    return elapsed, throughput


def bench_dequantize_mse(n: int, d: int, bits: int, device: str, n_warmup: int = 3, n_iters: int = 10):
    """Benchmark MSE dequantize throughput."""
    vectors = torch.randn(n, d, device=device)
    rotation = make_rotation_matrix(d, seed=42).to(device)
    codebook = lloyd_max_codebook(bits, d)

    packed, norms = quantize_mse(vectors, bits, dim=d, rotation=rotation, codebook=codebook)

    # Warmup
    for _ in range(n_warmup):
        dequantize_mse(packed, norms, bits, d, rotation=rotation, codebook=codebook)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        recon = dequantize_mse(packed, norms, bits, d, rotation=rotation, codebook=codebook)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters

    throughput = n / elapsed
    return elapsed, throughput


def bench_quantize_prod(n: int, d: int, bits: int, device: str, n_warmup: int = 3, n_iters: int = 10):
    """Benchmark Prod quantize throughput."""
    vectors = torch.randn(n, d, device=device)

    # Warmup
    for _ in range(n_warmup):
        quantize_prod(vectors, bits, dim=d)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        quantize_prod(vectors, bits, dim=d)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters

    throughput = n / elapsed
    return elapsed, throughput


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    configs = [
        # (N, d, bits)
        (1000, 64, 2),
        (1000, 64, 3),
        (1000, 64, 4),
        (1000, 128, 2),
        (1000, 128, 3),
        (1000, 128, 4),
        (10000, 128, 3),
        (10000, 128, 4),
        (1000, 256, 3),
        (1000, 256, 4),
    ]

    print(f"{'Mode':<12} {'N':>6} {'d':>4} {'bits':>4} {'time_ms':>10} {'vecs/sec':>12}")
    print("-" * 55)

    for n, d, bits in configs:
        elapsed, throughput = bench_quantize_mse(n, d, bits, device)
        print(f"{'quant_mse':<12} {n:>6} {d:>4} {bits:>4} {elapsed*1000:>10.2f} {throughput:>12.0f}")

        elapsed, throughput = bench_dequantize_mse(n, d, bits, device)
        print(f"{'dequant_mse':<12} {n:>6} {d:>4} {bits:>4} {elapsed*1000:>10.2f} {throughput:>12.0f}")

        if bits >= 2:
            elapsed, throughput = bench_quantize_prod(n, d, bits, device)
            print(f"{'quant_prod':<12} {n:>6} {d:>4} {bits:>4} {elapsed*1000:>10.2f} {throughput:>12.0f}")

        print()


if __name__ == "__main__":
    main()
