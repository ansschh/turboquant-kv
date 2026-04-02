"""
Quality benchmark: reproduce Figure 3 from the TurboQuant paper.

Plots MSE and inner product distortion vs bit width, compared against
theoretical bounds from the paper.

Usage:
    python -m benchmarks.quality.bench_distortion
    python benchmarks/quality/bench_distortion.py
"""

import sys
import math
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from turboquant_kv.reference import (
    quantize_mse,
    dequantize_mse,
    quantize_prod,
    dequantize_prod,
)


def measure_mse_distortion(n: int, d: int, bits: int, seed: int = 0) -> float:
    """Measure average normalized MSE for unit-norm vectors."""
    torch.manual_seed(seed)
    vectors = torch.randn(n, d)
    norms = torch.norm(vectors, dim=-1, keepdim=True)
    vectors_unit = vectors / norms

    packed, quant_norms = quantize_mse(vectors_unit, bits, dim=d)
    recon = dequantize_mse(packed, quant_norms, bits, d)

    mse = ((vectors_unit - recon) ** 2).sum(dim=-1)  # per-vector MSE
    return mse.mean().item()


def measure_ip_distortion(n: int, d: int, bits: int, mode: str = "mse", seed: int = 0) -> float:
    """Measure inner product distortion: E[(true_ip - approx_ip)^2]."""
    torch.manual_seed(seed)
    x = torch.randn(n, d)
    y = torch.randn(n, d)

    # Normalize
    x_norm = x / torch.norm(x, dim=-1, keepdim=True)
    y_norm = y / torch.norm(y, dim=-1, keepdim=True)

    if mode == "mse":
        packed, norms = quantize_mse(x_norm, bits, dim=d)
        x_recon = dequantize_mse(packed, norms, bits, d)
    elif mode == "prod":
        mse_packed, signs, res_norms, norms = quantize_prod(x_norm, bits, dim=d)
        x_recon = dequantize_prod(mse_packed, signs, res_norms, norms, bits, d)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    true_ip = (x_norm * y_norm).sum(dim=-1)
    approx_ip = (x_recon * y_norm).sum(dim=-1)

    ip_err = ((true_ip - approx_ip) ** 2).mean().item()
    return ip_err


def theoretical_mse_bound(bits: int) -> float:
    """Theoretical MSE bound: sqrt(3*pi)/2 * 4^{-b}."""
    return math.sqrt(3 * math.pi) / 2.0 * (4.0 ** (-bits))


def main():
    n = 2000
    d = 128
    bit_widths = [1, 2, 3, 4, 5, 6]

    print("=" * 70)
    print("TurboQuant Distortion Benchmark (Figure 3 reproduction)")
    print(f"N={n}, d={d}")
    print("=" * 70)

    # MSE distortion
    print(f"\n{'bits':>4} {'MSE_empirical':>14} {'MSE_bound':>12} {'ratio':>8}")
    print("-" * 42)
    mse_empirical = []
    mse_bounds = []

    for b in bit_widths:
        emp = measure_mse_distortion(n, d, b)
        bound = theoretical_mse_bound(b)
        ratio = emp / bound if bound > 0 else float('inf')
        mse_empirical.append(emp)
        mse_bounds.append(bound)
        print(f"{b:>4} {emp:>14.6f} {bound:>12.6f} {ratio:>8.3f}")

    # IP distortion
    print(f"\n{'bits':>4} {'IP_err_MSE':>14} {'IP_err_Prod':>14}")
    print("-" * 36)

    for b in bit_widths:
        ip_mse = measure_ip_distortion(n, d, b, mode="mse")
        if b >= 2:
            ip_prod = measure_ip_distortion(n, d, b, mode="prod")
        else:
            ip_prod = float('nan')
        print(f"{b:>4} {ip_mse:>14.6f} {ip_prod:>14.6f}")

    # Try to plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # MSE distortion plot
        ax = axes[0]
        ax.semilogy(bit_widths, mse_empirical, "o-", label="Empirical MSE")
        ax.semilogy(bit_widths, mse_bounds, "s--", label="Theoretical bound")
        ax.set_xlabel("Bit width (b)")
        ax.set_ylabel("MSE distortion")
        ax.set_title("MSE Distortion vs Bit Width")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # IP distortion plot
        ax = axes[1]
        ip_mse_vals = []
        ip_prod_vals = []
        for b in bit_widths:
            ip_mse_vals.append(measure_ip_distortion(n, d, b, mode="mse"))
            if b >= 2:
                ip_prod_vals.append(measure_ip_distortion(n, d, b, mode="prod"))
            else:
                ip_prod_vals.append(float('nan'))

        ax.semilogy(bit_widths, ip_mse_vals, "o-", label="MSE mode")
        valid_bits = [b for b in bit_widths if b >= 2]
        valid_prod = [v for b, v in zip(bit_widths, ip_prod_vals) if b >= 2]
        ax.semilogy(valid_bits, valid_prod, "s-", label="Prod mode")
        ax.set_xlabel("Bit width (b)")
        ax.set_ylabel("IP distortion (MSE of inner products)")
        ax.set_title("Inner Product Distortion vs Bit Width")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = Path(__file__).parent / "distortion_plot.png"
        plt.savefig(out_path, dpi=150)
        print(f"\nPlot saved to {out_path}")

    except ImportError:
        print("\nmatplotlib not available; skipping plot generation.")


if __name__ == "__main__":
    main()
