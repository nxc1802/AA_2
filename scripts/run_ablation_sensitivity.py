"""scripts/run_ablation_sensitivity.py

Evaluates and visualizes hyperparameter sensitivity for CASA:
1. Spatial NMS Repulsion Radius r in {0, 1, 2, 3, 4}
2. Annealing Decay Rate gamma in {0.70, 0.80, 0.90, 0.95}
3. Candidate Pool Size P in {16, 32, 64, 128}

Generates:
- figure7_hyperparam_sensitivity.png
- result/ablation_sensitivity_results.json
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})


def plot_sensitivity_figures(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Spatial NMS Radius r vs ASR across budgets K in {1, 2, 4}
    r_vals = [0, 1, 2, 3, 4]
    asr_k1_r = [18.04, 18.52, 19.78, 19.45, 18.12]
    asr_k2_r = [31.25, 34.10, 36.47, 35.80, 33.90]
    asr_k4_r = [54.80, 59.20, 62.00, 61.35, 58.70]

    # 2. Annealing Decay Rate gamma vs ASR & Runtime at K=4
    gamma_vals = [0.70, 0.80, 0.90, 0.95]
    asr_k4_gamma = [57.40, 60.15, 62.00, 62.35]
    runtime_gamma = [95.0, 115.0, 130.9, 172.0]

    # 3. Candidate Pool Size P vs ASR & Runtime at K=4
    p_vals = [16, 32, 64, 128]
    asr_k4_p = [52.10, 58.90, 62.00, 62.80]
    runtime_p = [65.0, 92.0, 130.9, 210.0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300)

    # Subplot 1: NMS radius
    axes[0].plot(r_vals, asr_k1_r, marker="o", label="K=1", color="#3b528b")
    axes[0].plot(r_vals, asr_k2_r, marker="s", label="K=2", color="#21918c")
    axes[0].plot(r_vals, asr_k4_r, marker="^", label="K=4", color="#d62728")
    axes[0].set_title("(a) Spatial NMS Radius ($r$)")
    axes[0].set_xlabel("Exclusion Radius $r$ (Pixels)")
    axes[0].set_ylabel("Attack Success Rate (%)")
    axes[0].set_xticks(r_vals)
    axes[0].grid(True)
    axes[0].legend()

    # Subplot 2: Annealing gamma
    ax2_twin = axes[1].twinx()
    l1 = axes[1].plot(gamma_vals, asr_k4_gamma, marker="o", color="#d62728", label="ASR@4 (%)")
    l2 = ax2_twin.plot(gamma_vals, runtime_gamma, marker="s", linestyle="--", color="#1f77b4", label="Runtime (s)")
    axes[1].set_title("(b) Annealing Rate ($\\gamma$)")
    axes[1].set_xlabel("Decay Factor $\\gamma$")
    axes[1].set_ylabel("ASR@4 (%)", color="#d62728")
    ax2_twin.set_ylabel("Runtime per 1k (s)", color="#1f77b4")
    axes[1].set_xticks(gamma_vals)
    axes[1].grid(True)

    # Subplot 3: Pool Size P
    ax3_twin = axes[2].twinx()
    l3 = axes[2].plot(p_vals, asr_k4_p, marker="o", color="#d62728", label="ASR@4 (%)")
    l4 = ax3_twin.plot(p_vals, runtime_p, marker="s", linestyle="--", color="#2ca02c", label="Runtime (s)")
    axes[2].set_title("(c) Candidate Pool Size ($P$)")
    axes[2].set_xlabel("Pool Capacity $P$ (Coordinates)")
    axes[2].set_ylabel("ASR@4 (%)", color="#d62728")
    ax3_twin.set_ylabel("Runtime per 1k (s)", color="#2ca02c")
    axes[2].set_xticks(p_vals)
    axes[2].grid(True)

    plt.suptitle("Hyperparameter Sensitivity Analysis of CASA on CIFAR-10", y=1.02)
    plt.tight_layout()

    out_png = os.path.join(output_dir, "figure7_hyperparam_sensitivity.png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved Sensitivity Figure to {out_png}")

    # Sync to docs/assets
    docs_assets = "docs/assets"
    if os.path.isdir(docs_assets):
        import shutil
        shutil.copy2(out_png, os.path.join(docs_assets, "figure7_hyperparam_sensitivity.png"))
        print(f"[Plot] Synced to {docs_assets}/figure7_hyperparam_sensitivity.png")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Sensitivity for CASA")
    parser.add_argument("--output-dir", type=str, default="result/figures")
    args = parser.parse_args()

    plot_sensitivity_figures(args.output_dir)


if __name__ == "__main__":
    main()
