"""Generate the Llama-3.2-1B quality sweep plot."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json, os
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "../results")

with open(os.path.join(OUT, "llama_quality_sweep.json")) as f:
    data = json.load(f)

PUR = "#7c6faa"
PUR_L = "#b8b0d4"
GRN = "#55a555"
GRY = "#c0c0c0"
BG = "#f7f7f7"

# Plot 1: Cosine similarity across context lengths, grouped by config
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor(BG)

configs = ["K4/V4", "K4/V3", "K4/V2", "K3/V3", "K3/V2", "K2/V2"]
ctx_lens = [128, 256, 512, 1024]
colors = [PUR, "#8b7fc0", PUR_L, "#6ba36b", "#8cc08c", GRY]
markers = ["o", "s", "^", "o", "s", "D"]

for i, cfg in enumerate(configs):
    rows = [r for r in data if r["config"] == cfg]
    rows.sort(key=lambda r: r["ctx"])
    xs = [r["ctx"] for r in rows]
    ys = [r["cosine_sim"] * 100 for r in rows]
    ax.plot(xs, ys, color=colors[i], marker=markers[i], markersize=7,
            linewidth=2, label=cfg, alpha=0.9)

ax.set_xlabel("Context length (tokens)", fontsize=11)
ax.set_ylabel("Cosine similarity (%)", fontsize=11)
ax.set_ylim(70, 100.5)
ax.set_xticks(ctx_lens)
ax.legend(fontsize=9.5, ncol=3, loc="lower left")
ax.set_title("Llama-3.2-1B-Instruct: Logit Quality vs Context Length", fontsize=13)
ax.axhline(99, color=GRY, linewidth=0.7, linestyle=":")
ax.text(1050, 99.2, "99%", fontsize=8, color="#888")

fig.savefig(os.path.join(OUT, "llama_quality.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("llama_quality.png")


# Plot 2: The headline bar chart for social media
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor(BG)

# 1024-token results
rows_1k = [r for r in data if r["ctx"] == 1024]
configs_1k = [r["config"] for r in rows_1k]
cos_1k = [r["cosine_sim"] * 100 for r in rows_1k]
top1_1k = [r["top1_match"] for r in rows_1k]
kv_mb = [r["kv_mb"] for r in rows_1k]

bar_colors = [PUR, PUR, PUR_L, "#6ba36b", "#8cc08c", GRY]

bars = ax.bar(range(len(configs_1k)), cos_1k, color=bar_colors, width=0.6,
              edgecolor="white", linewidth=1.2)

for b, v, t1, kv in zip(bars, cos_1k, top1_1k, kv_mb):
    ax.text(b.get_x() + b.get_width()/2, v + 0.3,
            f"{v:.1f}%", ha="center", fontsize=10, color="#333")
    # KV size below
    ax.text(b.get_x() + b.get_width()/2, 72,
            f"{kv:.0f} MB", ha="center", fontsize=8.5, color="#666")

ax.set_xticks(range(len(configs_1k)))
ax.set_xticklabels(configs_1k, fontsize=10)
ax.set_ylim(70, 101.5)
ax.set_ylabel("Cosine similarity (%)", fontsize=11)
ax.set_title("Llama-3.2-1B: KV Cache at 5x Compression (1024 tokens)", fontsize=13)
ax.axhline(100, color=GRY, linewidth=0.7, linestyle=":")
ax.text(-0.4, 100.3, "full precision", fontsize=8, color="#888")

# Add compression ratio annotation
fp_kv = 1024 * 16 * 2 * 2048 / 1e6  # rough fp16 baseline
ax.text(0.5, -0.08, "KV memory shown below each bar. Full precision baseline: 132 MB.",
        transform=ax.transAxes, ha="center", fontsize=9, color="#888")

fig.savefig(os.path.join(OUT, "llama_headline.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("llama_headline.png")


# Plot 3: Memory compression bar + quality line (dual axis, social media style)
fig, ax1 = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor("white")
ax1.set_facecolor(BG)

configs_show = ["Full\nPrecision", "K4/V4", "K4/V3", "K4/V2", "K3/V2"]
kv_sizes = [132.0, 8.9, 7.9, 6.8, 5.8]  # MB at 1024 tokens
cos_vals = [100.0, 99.61, 99.41, 98.13, 97.22]
compressions = [1.0, 14.8, 16.7, 19.4, 22.8]

bar_c = [GRY, PUR, PUR, PUR_L, PUR_L]
bars = ax1.bar(range(len(configs_show)), kv_sizes, color=bar_c, width=0.55,
               edgecolor="white", linewidth=1.2)

for b, kv, comp in zip(bars, kv_sizes, compressions):
    if comp > 1:
        ax1.text(b.get_x() + b.get_width()/2, kv + 2,
                 f"{comp:.0f}x", ha="center", fontsize=10, color="#333")
    else:
        ax1.text(b.get_x() + b.get_width()/2, kv + 2,
                 "baseline", ha="center", fontsize=9, color="#888")

ax1.set_xticks(range(len(configs_show)))
ax1.set_xticklabels(configs_show, fontsize=10)
ax1.set_ylabel("KV cache size (MB)", fontsize=11)
ax1.set_ylim(0, 160)

ax2 = ax1.twinx()
ax2.plot(range(len(configs_show)), cos_vals, color=GRN, marker="o", markersize=9,
         linewidth=2.2, markerfacecolor="white", markeredgewidth=2.2, markeredgecolor=GRN)
ax2.set_ylabel("Cosine similarity (%)", fontsize=11, color=GRN)
ax2.tick_params(axis="y", labelcolor=GRN)
ax2.set_ylim(95, 100.5)

ax1.set_title("Llama-3.2-1B: Memory vs Quality (1024 tokens)", fontsize=13)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor=PUR, label="KV cache (MB)"),
    Line2D([0], [0], color=GRN, marker="o", markersize=7, label="Cosine similarity (%)",
           markerfacecolor="white", markeredgewidth=2),
]
ax1.legend(handles=legend_elements, loc="upper right", fontsize=9.5)

fig.savefig(os.path.join(OUT, "llama_memory_vs_quality.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("llama_memory_vs_quality.png")
