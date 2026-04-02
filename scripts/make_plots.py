"""Generate clean, minimal benchmark plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "../results")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
})

PUR = "#7c6faa"
PUR_L = "#b8b0d4"
GRY = "#c0c0c0"
GRY_D = "#888888"
RED = "#cc4444"
GRN = "#55a555"
BG = "#f7f7f7"


def plot_recall():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor("white")

    # GloVe-200
    ax1.set_facecolor(BG)
    m1 = ["TQ 4b", "TQ 3b", "TQ 2b", "PQ m=8"]
    v1 = [0.998, 0.971, 0.863, 0.314]
    c1 = [PUR, PUR, PUR_L, GRY]
    bars1 = ax1.bar(range(len(m1)), v1, color=c1, width=0.65, edgecolor="white", linewidth=1.2)
    for b, v in zip(bars1, v1):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.1%}",
                 ha="center", fontsize=9.5, color="#333")
    ax1.set_xticks(range(len(m1)))
    ax1.set_xticklabels(m1, fontsize=9.5)
    ax1.set_ylim(0, 1.1)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.set_title("GloVe-200", fontsize=13)
    ax1.set_ylabel("Recall@1@10", fontsize=10)

    # SIFT-1M
    ax2.set_facecolor(BG)
    m2 = ["TQ 4b", "TQ 3b", "TQ 2b", "PQ m=32", "PQ m=16", "PQ m=8"]
    v2 = [0.810, 0.547, 0.338, 0.777, 0.504, 0.259]
    c2 = [PUR, PUR, PUR_L, GRY, GRY, GRY]
    bars2 = ax2.bar(range(len(m2)), v2, color=c2, width=0.65, edgecolor="white", linewidth=1.2)
    for b, v in zip(bars2, v2):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.1%}",
                 ha="center", fontsize=9, color="#333")
    ax2.set_xticks(range(len(m2)))
    ax2.set_xticklabels(m2, fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.set_title("SIFT-1M", fontsize=13)

    fig.subplots_adjust(wspace=0.25)
    fig.savefig(os.path.join(OUT, "recall_bars.png"), dpi=200)
    plt.close(fig)
    print("recall_bars.png")


def plot_quant_time():
    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(BG)

    methods = ["TQ 4-bit", "TQ 3-bit", "TQ 2-bit", "FlatIP", "PQ m=32", "PQ m=16", "PQ m=8"]
    times = [0.00, 0.00, 0.00, 0.34, 29.4, 27.7, 34.5]
    colors = [GRN, GRN, GRN, GRY, RED, RED, RED]

    bars = ax.barh(range(len(methods)), times, color=colors, height=0.6,
                   edgecolor="white", linewidth=1.2)
    for b, t in zip(bars, times):
        label = "< 0.01s" if t < 0.01 else f"{t:.1f}s"
        x = 0.5 if t < 0.01 else t + 0.5
        color = GRN if t < 0.01 else "#333"
        ax.text(x, b.get_y() + b.get_height()/2, label, va="center", fontsize=9.5, color=color)

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Build time (seconds)", fontsize=10)
    ax.set_title("Index Build Time (SIFT-1M)", fontsize=13)
    fig.savefig(os.path.join(OUT, "quant_time_pro.png"), dpi=200)
    plt.close(fig)
    print("quant_time_pro.png")


def plot_kernel():
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor(BG)

    methods = ["TQ 4-bit\nTriton", "cuBLAS\nfp32", "cuBLAS\nfp16"]
    times = [2.70, 2.76, 18.2]
    data_gb = [0.68, 5.12, 2.56]

    bars = ax1.bar(range(len(methods)), times,
                   color=[PUR, GRY, GRY_D], width=0.55, edgecolor="white", linewidth=1.2)
    for b, t in zip(bars, times):
        ax1.text(b.get_x() + b.get_width()/2, t + 0.4, f"{t:.2f} ms",
                 ha="center", fontsize=10, color="#333")

    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.set_ylabel("Latency (ms)", fontsize=10)
    ax1.set_ylim(0, 23)

    ax2 = ax1.twinx()
    ax2.plot(range(len(methods)), data_gb, color=GRN, marker="o", markersize=9,
             linewidth=2.2, markerfacecolor="white", markeredgewidth=2.2, markeredgecolor=GRN)
    ax2.set_ylabel("Data read (GB)", fontsize=10, color=GRN)
    ax2.tick_params(axis="y", labelcolor=GRN)
    ax2.set_ylim(0, 6.5)

    ax1.set_title("Kernel Performance (H100, 10M vectors)", fontsize=13)
    fig.savefig(os.path.join(OUT, "kernel_throughput.png"), dpi=200)
    plt.close(fig)
    print("kernel_throughput.png")


def plot_kv():
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(BG)

    configs = ["K4/V4", "K4/V3", "K4/V2", "K3/V3"]
    sims = [99.89, 99.89, 99.44, 99.60]
    colors = [PUR, PUR, PUR_L, PUR_L]

    bars = ax.bar(range(len(configs)), sims, color=colors, width=0.55,
                  edgecolor="white", linewidth=1.2)
    for b, v in zip(bars, sims):
        ax.text(b.get_x() + b.get_width()/2, v + 0.03, f"{v:.2f}%",
                ha="center", fontsize=10, color="#333")

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylim(98.8, 100.15)
    ax.set_ylabel("Cosine similarity (%)", fontsize=10)
    ax.set_title("KV Cache Quality (TinyLlama-1.1B, 256 tokens)", fontsize=12)
    fig.savefig(os.path.join(OUT, "kv_quality.png"), dpi=200)
    plt.close(fig)
    print("kv_quality.png")


if __name__ == "__main__":
    plot_recall()
    plot_quant_time()
    plot_kernel()
    plot_kv()
