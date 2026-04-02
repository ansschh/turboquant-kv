"""Generate professional benchmark plots for README."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "../results")
os.makedirs(OUT, exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

PURPLE = "#7c6faa"
PURPLE_LIGHT = "#b8b0d4"
GREEN = "#5cb85c"
GREEN_LIGHT = "#8fd48f"
GRAY = "#bfbfbf"
GRAY_DARK = "#8c8c8c"
RED = "#d9534f"
BG = "#fafafa"


# =====================================================================
# Plot 1: Recall@1@10 bar chart (GloVe-200 + SIFT-1M side by side)
# =====================================================================

def plot_recall_bars():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("white")

    datasets = {
        "GloVe-200": {
            "TurboQuant\n4-bit": 0.998,
            "TurboQuant\n3-bit": 0.971,
            "TurboQuant\n2-bit": 0.863,
            "FAISS PQ\nm=8": 0.314,
        },
        "SIFT-1M": {
            "TurboQuant\n4-bit": 0.810,
            "TurboQuant\n3-bit": 0.547,
            "TurboQuant\n2-bit": 0.338,
            "FAISS PQ\nm=32": 0.777,
            "FAISS PQ\nm=16": 0.504,
            "FAISS PQ\nm=8": 0.259,
        },
    }

    for ax, (ds_name, data) in zip(axes, datasets.items()):
        ax.set_facecolor(BG)
        methods = list(data.keys())
        vals = list(data.values())
        colors = [PURPLE if "TurboQuant" in m else GRAY for m in methods]

        bars = ax.bar(range(len(methods)), vals, color=colors, width=0.7,
                      edgecolor="white", linewidth=1.5)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Recall@1@10", fontsize=12)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

        ax.set_title(ds_name, fontsize=16, fontweight="bold", pad=12)
        ax.axhline(1.0, color=GRAY_DARK, linewidth=0.8, linestyle=":", alpha=0.5)

    fig.suptitle("Nearest Neighbor Recall", fontsize=18, fontweight="bold", y=1.02)
    fig.text(0.5, 0.98, "Recall@1@10 on standard benchmarks. Higher is better. Zero training time for TurboQuant.",
             ha="center", fontsize=11, color="#666666")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "recall_bars.png"), dpi=200)
    plt.close(fig)
    print("Saved: recall_bars.png")


# =====================================================================
# Plot 2: Quantization time comparison
# =====================================================================

def plot_quant_time():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(BG)

    methods = ["TurboQuant 4-bit", "TurboQuant 3-bit", "TurboQuant 2-bit",
               "FAISS FlatIP", "FAISS PQ m=32", "FAISS PQ m=16", "FAISS PQ m=8"]
    times = [0.00, 0.00, 0.00, 0.34, 29.4, 27.7, 34.5]
    colors = [GREEN if "TurboQuant" in m else GRAY if "Flat" in m else RED for m in methods]

    bars = ax.barh(range(len(methods)), times, color=colors, height=0.65,
                   edgecolor="white", linewidth=1.5)

    for bar, t, m in zip(bars, times, methods):
        if t < 0.01:
            ax.text(0.5, bar.get_y() + bar.get_height()/2,
                    "< 0.01s", va="center", fontsize=11, fontweight="bold", color=GREEN)
        else:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{t:.1f}s", va="center", fontsize=11, fontweight="bold")

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Index Build Time (seconds)", fontsize=12)
    ax.set_title("Quantization Time", fontsize=16, fontweight="bold", pad=12)
    fig.text(0.5, -0.02, "SIFT-1M (1M vectors, d=128). TurboQuant is data-oblivious: no training required.",
             ha="center", fontsize=11, color="#666666")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "quant_time_pro.png"), dpi=200)
    plt.close(fig)
    print("Saved: quant_time_pro.png")


# =====================================================================
# Plot 3: Kernel throughput (dual axis: time + compression)
# =====================================================================

def plot_kernel_throughput():
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor(BG)

    methods = ["TurboQuant\n4-bit (Triton)", "cuBLAS\nFlatIP fp32", "cuBLAS\nFlatIP fp16"]
    times = [2.70, 2.76, 18.2]
    data_read = [680, 5120, 2560]  # MB
    compression = [8, 1, 2]

    bars = ax1.bar(range(len(methods)), times,
                   color=[PURPLE, GRAY, GRAY_DARK], width=0.6,
                   edgecolor="white", linewidth=1.5)

    for bar, t in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{t:.2f} ms", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, fontsize=11)
    ax1.set_ylabel("Query Latency (ms)", fontsize=12)
    ax1.set_ylim(0, 22)

    # Second axis: data read
    ax2 = ax1.twinx()
    ax2.plot(range(len(methods)), [d/1000 for d in data_read],
             color=GREEN, marker="o", markersize=10, linewidth=2.5,
             markerfacecolor="white", markeredgewidth=2.5, markeredgecolor=GREEN)
    ax2.set_ylabel("Data Read (GB)", fontsize=12, color=GREEN)
    ax2.tick_params(axis="y", labelcolor=GREEN)
    ax2.set_ylim(0, 6)

    ax1.set_title("Kernel Performance", fontsize=16, fontweight="bold", pad=12)
    fig.text(0.5, -0.02, "H100, 10M vectors, d=128. TurboQuant matches cuBLAS while reading 8x less data.",
             ha="center", fontsize=11, color="#666666")

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=PURPLE, label="Query Latency (ms)"),
        Line2D([0], [0], color=GREEN, marker="o", markersize=8, label="Data Read (GB)",
               markerfacecolor="white", markeredgewidth=2),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kernel_throughput.png"), dpi=200)
    plt.close(fig)
    print("Saved: kernel_throughput.png")


# =====================================================================
# Plot 4: KV Cache quality (cosine similarity)
# =====================================================================

def plot_kv_quality():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(BG)

    configs = ["K4/V4", "K4/V3", "K4/V2", "K3/V3"]
    cos_sims = [99.89, 99.89, 99.44, 99.60]
    kl_divs = [0.007, 0.005, 0.047, 0.022]

    bars = ax.bar(range(len(configs)), cos_sims,
                  color=[PURPLE, PURPLE, PURPLE_LIGHT, PURPLE_LIGHT],
                  width=0.6, edgecolor="white", linewidth=1.5)

    for bar, v, kl in zip(bars, cos_sims, kl_divs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.5,
                f"KL={kl:.3f}", ha="center", va="top", fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=12)
    ax.set_ylim(98.5, 100.2)
    ax.set_ylabel("Cosine Similarity (%)", fontsize=12)
    ax.set_title("KV Cache Quality", fontsize=16, fontweight="bold", pad=12)
    fig.text(0.5, -0.02, "TinyLlama-1.1B, 256 tokens. Logit cosine similarity vs full-precision baseline.",
             ha="center", fontsize=11, color="#666666")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kv_quality.png"), dpi=200)
    plt.close(fig)
    print("Saved: kv_quality.png")


if __name__ == "__main__":
    plot_recall_bars()
    plot_quant_time()
    plot_kernel_throughput()
    plot_kv_quality()
    print("\nAll plots generated.")
