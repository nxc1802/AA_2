import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# LaTeX / publication aesthetic styling
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "grid.alpha": 0.4,
    "grid.linestyle": "--"
})


def plot_figure2_asr_vs_k(output_dir: str):
    """Figure 2: ASR vs K curve comparing CASA against competitors."""
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)

    k_vals = [1, 2, 4, 8, 16, 32, 64]

    # Benchmark numbers from official paper table
    attacks = {
        "CASA (Ours SOTA)": ([23.37, 39.70, 64.14, 81.43, 94.02, 99.47, 100.00], "#d62728", "o", "-"),
        "SPGD (ICML'19)": ([14.51, 32.66, 60.09, 86.23, 98.08, 100.00, 100.00], "#1f77b4", "s", "--"),
        "Sigma-Zero (NeurIPS'24)": ([12.91, 25.93, 44.72, 72.25, 94.88, 100.00, 100.00], "#2ca02c", "^", "-."),
        "Sparse-RS (ECCV'20)": ([29.88, 58.48, 84.31, 97.55, 100.00, 100.00, 100.00], "#ff7f0e", "D", ":"),
        "CornerSearch (ECCV'20)": ([30.95, 61.15, 77.37, 85.59, 90.29, 92.53, 92.64], "#9467bd", "v", ":"),
        "SparseFool (CVPR'19)": ([3.20, 6.08, 13.87, 27.43, 51.33, 72.25, 87.62], "#8c564b", "x", "-."),
        "PGD0 (Greedy)": ([2.88, 4.06, 8.22, 15.90, 27.85, 42.48, 58.38], "#7f7f7f", "*", "--"),
    }

    for name, (asr, color, marker, ls) in attacks.items():
        is_ours = "CASA" in name
        lw = 2.8 if is_ours else 1.8
        ms = 7 if is_ours else 5
        ax.plot(k_vals, asr, label=name, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=ms)

    ax.set_xscale("log", base=2)
    ax.set_xticks(k_vals)
    ax.set_xticklabels([str(k) for k in k_vals])
    ax.set_xlabel("Spatial Perturbation Budget $K$ (Pixels)")
    ax.set_ylabel("Conditional Attack Success Rate (%)")
    ax.set_title("Attack Success Rate ($ASR@K$) vs. Budget $K$ on CIFAR-10")
    ax.grid(True)
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()

    out_png = os.path.join(output_dir, "figure2_asr_vs_k.png")
    out_pdf = os.path.join(output_dir, "figure2_asr_vs_k.pdf")
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[Plot] Saved Figure 2 to {out_png}")


def plot_figure3_actual_l0(output_dir: str):
    """Figure 3: Actual L0 vs K Budget showing Drop-and-Repair compression."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)

    k_vals = [1, 2, 4, 8, 16, 32, 64]
    budget_line = k_vals
    casa_mean_l0 = [1.00, 1.52, 2.48, 4.09, 10.50, 26.16, 58.00]
    casa_median_l0 = [1.0, 2.0, 3.0, 4.0, 10.0, 26.0, 58.0]

    ax.plot(k_vals, budget_line, label="Budget Upper Bound ($K$)", color="gray", linestyle="--", linewidth=1.5)
    ax.plot(k_vals, casa_mean_l0, label="CASA Mean Actual $L_0$", color="#d62728", marker="o", linewidth=2.4)
    ax.plot(k_vals, casa_median_l0, label="CASA Median Actual $L_0$", color="#ff7f0e", marker="s", linestyle="-.", linewidth=2.0)

    ax.fill_between(k_vals, casa_mean_l0, budget_line, color="#d62728", alpha=0.12, label="Drop-and-Repair $L_0$ Savings")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(k_vals)
    ax.set_yticks(k_vals)
    ax.set_xticklabels([str(k) for k in k_vals])
    ax.set_yticklabels([str(k) for k in k_vals])
    ax.set_xlabel("Allowed Budget $K$ (Pixels)")
    ax.set_ylabel("Achieved Spatial $L_0$ Perturbation")
    ax.set_title("Perturbation Sparsity Compression via Drop-and-Repair")
    ax.grid(True)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()

    out_png = os.path.join(output_dir, "figure3_actual_l0_vs_k.png")
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[Plot] Saved Figure 3 to {out_png}")


def plot_figure4_efficiency(output_dir: str):
    """Figure 4: ASR vs Computational Cost (Wall-clock and queries)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)

    methods = ["CornerSearch", "Sparse-RS", "GSE", "PGD0", "SparseFool", "Sigma-Zero", "SPGD", "CASA"]
    runtime = [17337.0, 11664.0, 1756.0, 504.0, 468.0, 282.0, 183.0, 391.4]
    asr_k16 = [90.29, 100.0, 8.75, 27.85, 51.33, 94.88, 98.08, 94.02]
    colors = ["#9467bd", "#ff7f0e", "#e377c2", "#7f7f7f", "#8c564b", "#2ca02c", "#1f77b4", "#d62728"]

    for m, r, a, c in zip(methods, runtime, asr_k16, colors):
        size = 140 if "CASA" in m else 80
        ax.scatter(r, a, color=c, s=size, label=m, edgecolors="black", zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Runtime for 1,000 Images (Seconds, log-scale)")
    ax.set_ylabel("Attack Success Rate at $K=16$ (%)")
    ax.set_title("Efficiency Frontier: ASR@16 vs. Wall-Clock Runtime")
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()

    out_png = os.path.join(output_dir, "figure4_efficiency_frontier.png")
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[Plot] Saved Figure 4 to {out_png}")


def plot_figure5_ablation(output_dir: str):
    """Figure 5: Ablation Study Progression."""
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)

    stages = [
        "Base (V1)",
        "Iter 1\n(+GCT)",
        "Iter 2\n(+Spatial NMS)",
        "Iter 3\n(+Extremal Init)",
        "Iter 4\n(+Adaptive Swap)",
        "Iter 5\n(Full SOTA)"
    ]
    asr_k1 = [0.43, 18.04, 19.42, 21.77, 23.37, 23.37]
    asr_k4 = [5.34, 54.22, 58.06, 62.01, 62.43, 64.14]
    asr_k16 = [20.81, 88.69, 91.04, 92.74, 93.38, 94.02]

    x = np.arange(len(stages))
    width = 0.25

    ax.bar(x - width, asr_k1, width, label="K=1", color="#3b528b", alpha=0.9)
    ax.bar(x, asr_k4, width, label="K=4", color="#21918c", alpha=0.9)
    ax.bar(x + width, asr_k16, width, label="K=16", color="#5ec962", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Ablation Progression of CASA Components across Budgets")
    ax.grid(True, axis="y")
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()

    out_png = os.path.join(output_dir, "figure5_ablation_progression.png")
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[Plot] Saved Figure 5 to {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Generate Paper Figures for CASA")
    parser.add_argument("--output-dir", type=str, default="result/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating publication figures in {args.output_dir} ...")

    plot_figure2_asr_vs_k(args.output_dir)
    plot_figure3_actual_l0(args.output_dir)
    plot_figure4_efficiency(args.output_dir)
    plot_figure5_ablation(args.output_dir)
    print("\nAll figures generated successfully!")


if __name__ == "__main__":
    main()
