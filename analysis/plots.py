"""
analysis/plots.py

Generates the figures for the paper from saved results JSON files.

Produces:
  1. Radar chart — per-shift metric profile
  2. SSIM vs suppression scatter — the key "silent failure" plot
  3. Spectral degradation bar chart
  4. Metric disagreement heatmap across shift types
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List


SHIFT_ORDER = [
    "baseline",
    "mask_equispaced",
    "mask_magic",
    "accel_6x",
    "accel_8x",
    "accel_16x",
    "contrast_T1",
    "contrast_T1POST",
    "contrast_FLAIR",
    "anatomy_brain",
]

SHIFT_LABELS = {
    "baseline":        "Baseline\n(4×, random)",
    "mask_equispaced": "Mask:\nEquispaced",
    "mask_magic":      "Mask:\nMagic",
    "accel_6x":        "Accel 6×",
    "accel_8x":        "Accel 8×",
    "accel_16x":       "Accel 16×",
    "contrast_T1":     "Contrast:\nT1",
    "contrast_T1POST": "Contrast:\nT1POST",
    "contrast_FLAIR":  "Contrast:\nFLAIR",
    "anatomy_brain":   "Anatomy:\nBrain",
}

# Severity color gradient: blue (mild) → red (severe)
SHIFT_COLORS = {
    "baseline":        "#2c7bb6",
    "mask_equispaced": "#74add1",
    "mask_magic":      "#abd9e9",
    "accel_6x":        "#fee090",
    "accel_8x":        "#fdae61",
    "accel_16x":       "#f46d43",
    "contrast_T1":     "#a6d96a",
    "contrast_T1POST": "#66bd63",
    "contrast_FLAIR":  "#1a9850",
    "anatomy_brain":   "#d73027",
}


def load_results(results_dir: str) -> Dict[str, Dict]:
    """Load per-shift JSON files and compute mean metrics."""
    results_dir = Path(results_dir)
    aggregated = {}
    for shift in SHIFT_ORDER:
        fpath = results_dir / f"{shift}.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            records = json.load(f)
        # Average over all slices (skip non-numeric fields)
        numeric_keys = [k for k in records[0] if isinstance(records[0][k], (int, float))]
        aggregated[shift] = {
            k: float(np.nanmean([r[k] for r in records]))
            for k in numeric_keys
        }
    return aggregated


def plot_metric_bar(aggregated: Dict, metric: str, ax, ylabel: str, title: str, higher_is_better=True):
    """Generic bar chart for a single metric across shifts."""
    shifts  = [s for s in SHIFT_ORDER if s in aggregated]
    values  = [aggregated[s][metric] for s in shifts]
    colors  = [SHIFT_COLORS[s] for s in shifts]
    labels  = [SHIFT_LABELS[s] for s in shifts]

    bars = ax.bar(range(len(shifts)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(shifts)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Baseline reference line
    if "baseline" in aggregated:
        ax.axhline(aggregated["baseline"][metric], color="black",
                   linestyle="--", linewidth=1, alpha=0.6, label="Baseline")
        ax.legend(fontsize=7)


def plot_ssim_vs_suppression(aggregated: Dict, ax):
    """
    Key scatter plot: SSIM (x) vs suppression_ratio (y).

    If a model appears high-SSIM but also high-suppression,
    that's the "silent failure" finding — SSIM is lying.
    """
    for shift in SHIFT_ORDER:
        if shift not in aggregated:
            continue
        x = aggregated[shift].get("ssim", np.nan)
        y = aggregated[shift].get("suppression_ratio", np.nan)
        if np.isnan(x) or np.isnan(y):
            continue
        ax.scatter(x, y, color=SHIFT_COLORS[shift], s=80, zorder=3,
                   label=SHIFT_LABELS[shift].replace("\n", " "))
        ax.annotate(SHIFT_LABELS[shift].split("\n")[0],
                    (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7)

    ax.set_xlabel("SSIM ↑  (higher = better reconstruction)", fontsize=9)
    ax.set_ylabel("Suppression Ratio ↑  (higher = more fine-detail loss)", fontsize=9)
    ax.set_title("Silent Failure: SSIM vs. Feature Suppression", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Quadrant annotation
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.text(0.02, 0.98, "← High SSIM, High Suppression\n   (Model looks fine, but isn't)",
            transform=ax.transAxes, fontsize=7, va="top", color="firebrick",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose", alpha=0.7))


def plot_spectral_degradation(aggregated: Dict, ax):
    """
    Grouped bar: HF ratio and LF ratio side by side per shift.
    Illustrates whether failure is hallucination (LF↑) or suppression (HF↓).
    """
    shifts = [s for s in SHIFT_ORDER if s in aggregated]
    x      = np.arange(len(shifts))
    w      = 0.35

    hf = [aggregated[s].get("hf_power_ratio", np.nan) for s in shifts]
    lf = [aggregated[s].get("lf_power_ratio", np.nan) for s in shifts]

    ax.bar(x - w/2, hf, w, label="HF power ratio", color="#4393c3", alpha=0.85)
    ax.bar(x + w/2, lf, w, label="LF power ratio", color="#d6604d", alpha=0.85)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="Perfect (ratio=1)")

    ax.set_xticks(x)
    ax.set_xticklabels([SHIFT_LABELS[s].replace("\n", " ") for s in shifts], fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Power Ratio (pred / target)", fontsize=9)
    ax.set_title("Spectral Degradation: High vs. Low Frequency Content", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate_all_figures(results_dir: str, output_dir: str):
    """Main entry point: load results and render all paper figures."""
    results_dir = Path(results_dir)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated = load_results(results_dir)
    if not aggregated:
        print("No results found. Run run_experiments.py first.")
        return

    # ---- Figure 1: Core metric panel --------------------------------- #
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Distribution Shift Robustness in Deep Learning MRI Reconstruction",
                 fontsize=13, fontweight="bold", y=1.01)

    plot_metric_bar(aggregated, "ssim",              axes[0, 0], "SSIM",  "SSIM Across Shifts")
    plot_metric_bar(aggregated, "nmse",              axes[0, 1], "NMSE",  "NMSE Across Shifts", higher_is_better=False)
    plot_metric_bar(aggregated, "edge_preservation", axes[1, 0], "Edge Corr.", "Edge Preservation")
    plot_metric_bar(aggregated, "metric_disagreement", axes[1, 1], "Disagreement Index",
                    "SSIM vs. LPIPS Disagreement")

    plt.tight_layout()
    fig.savefig(output_dir / "fig1_metric_panel.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "fig1_metric_panel.png", bbox_inches="tight", dpi=150)
    print(f"Saved fig1_metric_panel")

    # ---- Figure 2: Silent failure scatter ---------------------------- #
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    plot_ssim_vs_suppression(aggregated, ax2)
    plt.tight_layout()
    fig2.savefig(output_dir / "fig2_silent_failure.pdf", bbox_inches="tight", dpi=300)
    fig2.savefig(output_dir / "fig2_silent_failure.png", bbox_inches="tight", dpi=150)
    print(f"Saved fig2_silent_failure")

    # ---- Figure 3: Spectral degradation ------------------------------ #
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    plot_spectral_degradation(aggregated, ax3)
    plt.tight_layout()
    fig3.savefig(output_dir / "fig3_spectral.pdf", bbox_inches="tight", dpi=300)
    fig3.savefig(output_dir / "fig3_spectral.png", bbox_inches="tight", dpi=150)
    print(f"Saved fig3_spectral")

    plt.close("all")
    print(f"\nAll figures written to {output_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--output_dir",  default="figures/")
    args = p.parse_args()
    generate_all_figures(args.results_dir, args.output_dir)
