#!/usr/bin/env python3
"""
generate_ablation_plot.py
Generates publication-quality Ablation Study figures for CASA (Iterations 1 to 5).
Visualizes module contribution breakdown across sparse budgets K in {1, 2, 4, 8, 16, 32, 64}.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def generate_ablation_charts():
    # Set modern aesthetics
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "figure.dpi": 300
    })

    k_values = [1, 2, 4, 8, 16, 32, 64]
    k_labels = ["K=1", "K=2", "K=4", "K=8", "K=16", "K=32", "K=64"]

    # Data from Table 5 in benchmark_results_cifar10.md
    v1_baseline = np.array([0.43, 1.92, 5.34, 10.14, 20.81, 36.07, 61.15])
    iter1_box = np.array([18.04, 33.62, 54.22, 73.85, 88.69, 97.87, 99.89])
    iter2_dlr = np.array([19.42, 36.71, 58.06, 76.73, 91.04, 98.40, 99.89])
    iter3_refresh = np.array([21.77, 39.27, 62.01, 80.58, 92.74, 98.51, 99.89])
    iter4_gct_nms = np.array([23.37, 39.81, 62.43, 80.47, 93.38, 99.15, 100.00])
    iter5_final = np.array([23.37, 39.70, 64.14, 81.43, 94.02, 99.47, 100.00])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.8))

    # -------------------------------------------------------------
    # Subplot 1: Progression Curve (ASR@K vs K)
    # -------------------------------------------------------------
    x_indices = np.arange(len(k_values))

    ax1.plot(x_indices, v1_baseline, 'o--', color="#8E9AA8", linewidth=1.8, markersize=6, label="Baseline (V1 Heuristic)")
    ax1.plot(x_indices, iter1_box, 's-', color="#3B82F6", linewidth=2.0, markersize=6, label="Iter 1: +Box-Init & α=0.25")
    ax1.plot(x_indices, iter2_dlr, '^-', color="#0D9488", linewidth=2.0, markersize=6, label="Iter 2: +DLR Loss & Margin")
    ax1.plot(x_indices, iter3_refresh, 'd-', color="#F59E0B", linewidth=2.0, markersize=6, label="Iter 3: +Dynamic Refresh & Tabu")
    ax1.plot(x_indices, iter4_gct_nms, 'v-', color="#8B5CF6", linewidth=2.0, markersize=6, label="Iter 4: +GCT & Spatial NMS")
    ax1.plot(x_indices, iter5_final, '*-', color="#DC2626", linewidth=2.8, markersize=10, label="Iter 5: CASA SOTA (+Batch Swap & Repair)")

    # Formatting Subplot 1
    ax1.set_title("A. Cumulative Attack Success Rate (ASR) by Iteration", pad=12, fontweight="bold")
    ax1.set_xlabel("Sparse Pixel Budget ($K$)", labelpad=8)
    ax1.set_ylabel("Attack Success Rate (%)", labelpad=8)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(k_labels)
    ax1.set_ylim(-2, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", framealpha=0.95, edgecolor="#E2E8F0")

    # Annotate Key Leaps
    ax1.annotate(f"K=1: 0.4% → 23.4%\n(+23.0%)", xy=(0, 23.37), xytext=(0.2, 38),
                 arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5),
                 fontweight="bold", color="#B91C1C", bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#F87171"))

    ax1.annotate(f"K=4: 5.3% → 64.1%\n(+58.8%)", xy=(2, 64.14), xytext=(1.4, 78),
                 arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5),
                 fontweight="bold", color="#B91C1C", bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#F87171"))

    ax1.annotate(f"K=16: 20.8% → 94.0%\n(+73.2%)", xy=(4, 94.02), xytext=(3.2, 50),
                 arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5),
                 fontweight="bold", color="#B91C1C", bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#F87171"))

    # -------------------------------------------------------------
    # Subplot 2: Incremental Contribution Breakdown (Ablation Waterfall)
    # -------------------------------------------------------------
    # Calculate non-negative incremental contributions per stage for clean visual representation
    delta_box = np.maximum(0, iter1_box - v1_baseline)
    delta_dlr = np.maximum(0, iter2_dlr - iter1_box)
    delta_refresh = np.maximum(0, iter3_refresh - iter2_dlr)
    delta_gct_nms = np.maximum(0, iter4_gct_nms - iter3_refresh)
    delta_swap = np.maximum(0, iter5_final - iter4_gct_nms)

    width = 0.55
    bottom = np.zeros(len(k_values))

    p0 = ax2.bar(x_indices, v1_baseline, width, label="Baseline (V1 Heuristic)", color="#94A3B8", edgecolor="white")
    bottom += v1_baseline

    p1 = ax2.bar(x_indices, delta_box, width, bottom=bottom, label="1. +Box-Extremal & α=0.25", color="#3B82F6", edgecolor="white")
    bottom += delta_box

    p2 = ax2.bar(x_indices, delta_dlr, width, bottom=bottom, label="2. +DLR Loss & Margin", color="#0D9488", edgecolor="white")
    bottom += delta_dlr

    p3 = ax2.bar(x_indices, delta_refresh, width, bottom=bottom, label="3. +Dynamic Pool & Tabu", color="#F59E0B", edgecolor="white")
    bottom += delta_refresh

    p4 = ax2.bar(x_indices, delta_gct_nms, width, bottom=bottom, label="4. +GCT & Spatial NMS", color="#8B5CF6", edgecolor="white")
    bottom += delta_gct_nms

    p5 = ax2.bar(x_indices, delta_swap, width, bottom=bottom, label="5. +Adaptive Batch Swap", color="#EF4444", edgecolor="white")
    bottom += delta_swap

    # Add total ASR text on top of each bar
    for idx, total_val in enumerate(iter5_final):
        ax2.text(idx, total_val + 1.5, f"{total_val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # Formatting Subplot 2
    ax2.set_title("B. Incremental Component Decomposition (+Δ% ASR)", pad=12, fontweight="bold")
    ax2.set_xlabel("Sparse Pixel Budget ($K$)", labelpad=8)
    ax2.set_ylabel("Total Attack Success Rate (%)", labelpad=8)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(k_labels)
    ax2.set_ylim(0, 112)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", framealpha=0.95, edgecolor="#E2E8F0")

    plt.suptitle("CASA SOTA Architecture: Component Ablation Study (1,000 CIFAR-10 Samples, ResNet-18)", y=0.98, fontweight="bold")
    plt.tight_layout()

    out_dir = "docs/assets"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "casa_ablation_study.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Ablation figure successfully saved to: {out_path}")

    # Copy to artifact directory
    artifact_dir = "/Users/nxc/.gemini/antigravity-ide/brain/1779e985-31de-4ac5-b00d-b6b94a3a1bc8"
    if os.path.exists(artifact_dir):
        artifact_out = os.path.join(artifact_dir, "casa_ablation_study.png")
        plt.savefig(artifact_out, dpi=300, bbox_inches="tight")
        print(f"Artifact copy saved to: {artifact_out}")

if __name__ == "__main__":
    generate_ablation_charts()
