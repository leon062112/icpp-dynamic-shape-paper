"""
Experiment 3 – Plotting

Figure 4: Hardware Bottleneck Migration across Dynamic Shapes
  2×2 layout: Normalized latency, Occupancy, L2 Hit Rate, DRAM Throughput
  Each subplot: Config-A solid, Config-B dashed
"""

import json
import os

import matplotlib
import matplotlib.font_manager as font_manager
matplotlib.use("Agg")
import matplotlib.pyplot as plt

font_path = '/System/Library/Fonts/Supplemental/Times New Roman.ttf'
font_manager.fontManager.addfont(font_path)
plt.rcParams.update({'font.family': 'Times New Roman'})

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COLOR_A = "#e0f2d7"
COLOR_B = "#afc8ea"


def load_results(path=None):
    if path is None:
        path = os.path.join(SCRIPT_DIR, "ncu_results.json")
    with open(path) as f:
        return json.load(f)


def _extract(results_list, metric_key):
    return [r["metrics"].get(metric_key, 0) for r in results_list]


def _get_M_values(results_list):
    return [r["M"] for r in results_list]


def plot_bottleneck_migration(data):
    config_names = list(data.keys())
    assert len(config_names) == 2
    name_a, name_b = config_names
    res_a, res_b = data[name_a], data[name_b]
    M_vals = _get_M_values(res_a)

    label_a = "Config-A (128x128x32, w4 s3)"
    label_b = "Config-B (32x32x64, w4 s5)"

    x = np.arange(len(M_vals))
    x_labels = [str(m) for m in M_vals]

    latency_a = np.array(_extract(res_a, "latency_us"), dtype=float)
    latency_b = np.array(_extract(res_b, "latency_us"), dtype=float)
    best_latency = np.minimum(latency_a, latency_b)
    norm_latency_a = latency_a / best_latency
    norm_latency_b = latency_b / best_latency

    metrics_info = [
        ("normalized_latency", "Normalized Latency", "(a)", norm_latency_a, norm_latency_b),
        ("occupancy_pct", "Occupancy (%)", "(b)", None, None),
        ("l2_hit_rate_pct", "L2 Hit Rate (%)", "(c)", None, None),
        ("dram_throughput_pct", "DRAM Throughput (%)", "(d)", None, None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.8))
    axes = axes.ravel()

    for idx, (metric_key, ylabel, tag, preset_a, preset_b) in enumerate(metrics_info):
        ax = axes[idx]
        if preset_a is None:
            vals_a = np.array(_extract(res_a, metric_key), dtype=float)
            vals_b = np.array(_extract(res_b, metric_key), dtype=float)
        else:
            vals_a = preset_a
            vals_b = preset_b

        ax.plot(x, vals_a, "o-", color=COLOR_A, linewidth=2.2, markersize=6,
                label=label_a, zorder=3)
        ax.plot(x, vals_b, "s--", color=COLOR_B, linewidth=2.2, markersize=6,
                label=label_b, zorder=3)

        ax.fill_between(x, vals_a, vals_b, alpha=0.08, color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10, rotation=30)
        ax.set_xlabel("M", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{tag} {ylabel}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.margins(x=0.08, y=0.12)

        for i, xoff in [(0, 5), (len(M_vals) - 1, -22)]:
            va_a, va_b = vals_a[i], vals_b[i]
            off_a = (xoff, 5) if va_a >= va_b else (xoff, -12)
            off_b = (xoff, -12) if va_a >= va_b else (xoff, 5)
            ha = "left" if i == 0 else "right"
            ax.annotate(f"{va_a:.1f}", xy=(x[i], va_a), xytext=off_a,
                        textcoords="offset points", fontsize=9, ha=ha,
                        color=COLOR_A, fontweight="bold")
            ax.annotate(f"{va_b:.1f}", xy=(x[i], va_b), xytext=off_b,
                        textcoords="offset points", fontsize=9, ha=ha,
                        color=COLOR_B, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=11, loc="upper center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.tight_layout(pad=0.8, rect=(0, 0, 1, 0.95))

    path = os.path.join(SCRIPT_DIR, "figure4_bottleneck_migration.pdf")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def print_summary_table(data):
    """Print a formatted table for paper inclusion."""
    config_names = list(data.keys())
    name_a, name_b = config_names

    print("\n" + "=" * 90)
    print("Table: Hardware Metrics Summary")
    print("=" * 90)
    header = (f"{'M':>6s} | {'Lat-A(us)':>10s} {'Lat-B(us)':>10s} {'A/B':>6s} | "
              f"{'Occ-A':>6s} {'Occ-B':>6s} | {'L2-A':>6s} {'L2-B':>6s} | "
              f"{'DRAM-A':>7s} {'DRAM-B':>7s} | {'Stall-A':>8s} {'Stall-B':>8s}")
    print(header)
    print("-" * 95)

    for i in range(len(data[name_a])):
        ra = data[name_a][i]["metrics"]
        rb = data[name_b][i]["metrics"]
        M = data[name_a][i]["M"]
        la, lb = ra.get("latency_us", 0), rb.get("latency_us", 0)
        ratio = la / lb if lb > 0 else 0
        print(f"{M:>6d} | {la:>10.2f} {lb:>10.2f} {ratio:>6.2f} | "
              f"{ra.get('occupancy_pct',0):>6.2f} {rb.get('occupancy_pct',0):>6.2f} | "
              f"{ra.get('l2_hit_rate_pct',0):>6.2f} {rb.get('l2_hit_rate_pct',0):>6.2f} | "
              f"{ra.get('dram_throughput_pct',0):>7.2f} {rb.get('dram_throughput_pct',0):>7.2f} | "
              f"{ra.get('stall_long_scoreboard',0):>8.1f} {rb.get('stall_long_scoreboard',0):>8.1f}")


def main():
    data = load_results()
    plot_bottleneck_migration(data)
    print_summary_table(data)


if __name__ == "__main__":
    main()
