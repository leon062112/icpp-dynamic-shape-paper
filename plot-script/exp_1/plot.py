"""
Experiment 1 – Plotting

Figure 1: Shape × Config Latency Heatmap
  - X-axis: config index (sorted by latency on largest M)
  - Y-axis: shapes (M values)
  - Color:  latency normalized to per-shape best
  - Mark each shape's best config

Figure 2: Static Tuning Performance Loss
  - X-axis: shapes
  - Bars:   per-shape best latency vs static baseline latency
  - Annotate performance loss %
"""

import json
import os
import sys

import matplotlib
import matplotlib.font_manager as font_manager
matplotlib.use("Agg")
import matplotlib.pyplot as plt

font_path = '/System/Library/Fonts/Supplemental/Times New Roman.ttf'
font_manager.fontManager.addfont(font_path)
plt.rcParams.update({'font.family': 'Times New Roman'})

import matplotlib.ticker as ticker
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_results(path=None):
    if path is None:
        path = os.path.join(SCRIPT_DIR, "benchmark_results.json")
    with open(path) as f:
        return json.load(f)


def _short_label(cfg_str):
    """Turn 'BM64_BN128_BK32_w4_s3' into '64×128×32\nw4 s3'."""
    parts = cfg_str.split("_")
    bm = parts[0].replace("BM", "")
    bn = parts[1].replace("BN", "")
    bk = parts[2].replace("BK", "")
    w = parts[3]                       # 'w4'
    s = parts[4]                       # 's3'
    return f"{bm}×{bn}×{bk}\n{w} {s}"


def plot_heatmap(data, save_dir=None, top_k=20):
    """Figure 1: Top-K Config × Shape Latency Heatmap.

    Only keeps the *top_k* most competitive configs (union of per-shape
    top-5 + globally best), sorted by average normalised latency.
    Each cell shows the normalised slowdown vs. per-shape best;
    the per-shape winner is marked with a star.
    """
    if save_dir is None:
        save_dir = SCRIPT_DIR

    M_values = data["M_values"]
    config_names = data["configs"]
    results = data["results"]

    # ── common configs across all shapes ──
    common_configs = set(config_names)
    for m in M_values:
        common_configs &= set(results[str(m)].keys())
    common_configs = sorted(common_configs)
    if not common_configs:
        print("ERROR: No configs have results for all shapes!")
        return
    print(f"Configs with data for all {len(M_values)} shapes: {len(common_configs)}")

    # ── select Top-K configs ──
    # Strategy: union of top-5 per shape (by raw latency) to guarantee every
    # per-shape best is included, then pad with globally-best (by avg norm).
    selected = set()
    for m in M_values:
        lats = [(cfg, results[str(m)][cfg]) for cfg in common_configs]
        lats.sort(key=lambda x: x[1])
        for cfg, _ in lats[:5]:
            selected.add(cfg)

    # If still under top_k, fill with globally-best by avg normalised latency
    if len(selected) < top_k:
        # compute per-shape min for normalisation
        ps_min = {}
        for m in M_values:
            ps_min[m] = min(results[str(m)][c] for c in common_configs)
        avg_norm = {}
        for cfg in common_configs:
            avg_norm[cfg] = np.mean(
                [results[str(m)][cfg] / ps_min[m] for m in M_values]
            )
        for cfg, _ in sorted(avg_norm.items(), key=lambda x: x[1]):
            selected.add(cfg)
            if len(selected) >= top_k:
                break

    selected = sorted(selected)
    n_sel = len(selected)
    print(f"Selected Top-{n_sel} configs for heatmap")

    # ── build matrix ──
    lat_matrix = np.zeros((len(M_values), n_sel))
    for i, m in enumerate(M_values):
        for j, cfg in enumerate(selected):
            lat_matrix[i, j] = results[str(m)][cfg]

    # normalise to per-shape best (among ALL configs, not just selected)
    ps_best_all = np.array([
        min(results[str(m)][c] for c in common_configs) for m in M_values
    ])
    norm_matrix = lat_matrix / ps_best_all[:, None]

    # sort columns by average normalised latency (best first)
    col_order = np.argsort(norm_matrix.mean(axis=0))
    norm_matrix = norm_matrix[:, col_order]
    lat_matrix = lat_matrix[:, col_order]
    selected = [selected[i] for i in col_order]

    # per-shape best column index (within the selected set)
    best_col = np.argmin(lat_matrix, axis=1)

    # ── plot ──
    fig, ax = plt.subplots(figsize=(max(12, n_sel * 0.7), 4.5))
    im = ax.imshow(
        norm_matrix, aspect="auto", cmap="YlOrRd",
        vmin=1.0, vmax=min(float(norm_matrix.max()), 2.5),
        interpolation="nearest",
    )

    # cell text: show normalised value
    for i in range(len(M_values)):
        for j in range(n_sel):
            v = norm_matrix[i, j]
            color = "white" if v > 1.8 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=14, color=color) # 字体大小调整标注: 热力图内数字的值

    # star on per-shape best
    for i, bj in enumerate(best_col):
        ax.plot(bj, i, marker="*", color="#1565C0", markersize=15, zorder=5)

    ax.set_yticks(range(len(M_values)))
    ax.set_yticklabels([str(m) for m in M_values], fontsize=16) # 字体大小调整标注: y轴刻度标签的字体大小
    ax.set_xticks(range(n_sel))
    ax.set_xticklabels([_short_label(c) for c in selected],
                       fontsize=14, rotation=50, ha="center")
    # ax.set_xlabel("Autotuning Configuration  (BLOCK_M×N×K, warps, stages)", fontsize=12)
    ax.set_ylabel("Shape (M)", fontsize=16)
    # 使用图例对象展示和 heatmap 中完全相同的五角星作为头部提示
    ax.plot([], [], marker="*", color="#1565C0", markersize=15, linestyle="None", label="Per-shape Best Configuration")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False, fontsize=16, handletextpad=0.2)
    

    cbar = fig.colorbar(im, ax=ax, shrink=1.0, pad=0.02)
    cbar.set_label("Normalised Latency", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    fig.tight_layout()
    path = os.path.join(save_dir, "3-moti-heatmap.pdf")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)

    return selected, lat_matrix, norm_matrix


def plot_performance_loss(data, save_dir=None):
    """Figure 2: Static Tuning Performance Loss."""
    if save_dir is None:
        save_dir = SCRIPT_DIR

    M_values = data["M_values"]
    results = data["results"]
    config_names = data["configs"]

    # Common configs across all shapes
    common_configs = set(config_names)
    for m in M_values:
        common_configs &= set(results[str(m)].keys())
    common_configs = sorted(common_configs)

    if not common_configs:
        print("ERROR: No common configs!")
        return

    # Per-shape best latency
    per_shape_best_lat = []
    per_shape_best_cfg = []
    for m in M_values:
        lats = {cfg: results[str(m)][cfg] for cfg in common_configs}
        best_cfg = min(lats, key=lats.get)
        per_shape_best_lat.append(lats[best_cfg])
        per_shape_best_cfg.append(best_cfg)

    # Static baseline: best config on largest M
    largest_M = max(M_values)
    lats_largest = {cfg: results[str(largest_M)][cfg] for cfg in common_configs}
    static_cfg = min(lats_largest, key=lats_largest.get)
    print(f"Static baseline config (best on M={largest_M}): {static_cfg}")

    static_lat = []
    for m in M_values:
        static_lat.append(results[str(m)][static_cfg])

    per_shape_best_lat = np.array(per_shape_best_lat)
    static_lat = np.array(static_lat)
    loss_pct = (static_lat - per_shape_best_lat) / per_shape_best_lat * 100

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(M_values))
    width = 0.3

    bars1 = ax.bar(x - width / 2, per_shape_best_lat, width,
                   label="Per-shape Best Config", color="#e0f2d7", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, static_lat, width,
                   label=f"Static (Best Config on M={largest_M})", color="#afc8ea", edgecolor="black", linewidth=0.5)

    # Annotate loss %
    for i, (pct, sl) in enumerate(zip(loss_pct, static_lat)):
        ax.annotate(f"+{pct:.1f}%",
                    xy=(x[i] + width / 2, sl),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=16, color="black", fontweight="bold") # 字体大小调整标注: 注释百分比的字体大小

    ax.set_ylim(0, 0.027)
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in M_values], fontsize=18) # 字体大小调整标注: x轴刻度标签(即横轴M的值)的字体大小
    ax.set_ylabel("Latency (ms)", fontsize=18) # 字体大小调整标注: y轴标题的字体大小
    ax.legend(fontsize=18, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False) # 字体大小调整标注: 图例的字体大小
    ax.tick_params(axis="y", labelsize=18) # 字体大小调整标注: y轴刻度标签(即坐标值)的字体大小
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, "figure2_perf_loss.pdf")
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".pdf", ".png"), dpi=200)
    print(f"Saved: {path}")
    plt.close(fig)

    # Print table
    print("\n--- Performance Loss Summary ---")
    print(f"{'M':>6s}  {'Per-shape best (ms)':>20s}  {'Static (ms)':>12s}  {'Loss':>8s}  {'Best config':>50s}")
    for i, m in enumerate(M_values):
        print(f"{m:>6d}  {per_shape_best_lat[i]:>20.4f}  {static_lat[i]:>12.4f}  "
              f"{loss_pct[i]:>+7.1f}%  {per_shape_best_cfg[i]:>50s}")
    print(f"\nMax loss: {loss_pct.max():.1f}% at M={M_values[np.argmax(loss_pct)]}")
    print(f"Avg loss: {loss_pct.mean():.1f}%")


def main():
    data = load_results()
    print(f"Loaded results: {len(data['M_values'])} shapes × {len(data['configs'])} configs\n")

    plot_heatmap(data)
    print()
    plot_performance_loss(data)
    print()


if __name__ == "__main__":
    main()
