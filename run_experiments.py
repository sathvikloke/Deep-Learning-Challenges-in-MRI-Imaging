"""
run_experiments.py

Main entry point. Runs all distribution shift experiments and saves results.

Usage:
    python run_experiments.py --data_dir /path/to/fastmri --challenge multicoil --split val
"""

import argparse
import json
import os
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from shifts.mask_shift import MaskShiftGenerator
from shifts.contrast_shift import ContrastShiftLoader
from shifts.anatomy_shift import AnatomyShiftLoader
from shifts.accel_shift import AccelShiftGenerator
from models.reconstructor import load_reconstructor
from analysis.metrics import compute_all_metrics
from utils.io import load_kspace, save_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory of fastMRI data")
    parser.add_argument("--challenge", type=str, default="multicoil",
                        choices=["singlecoil", "multicoil"])
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--model", type=str, default="varnet",
                        choices=["varnet", "unet"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint. If None, uses pretrained.")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--max_slices", type=int, default=200,
                        help="Max slices to evaluate (for quick runs)")
    parser.add_argument("--device", type=str, default="mps",
                        choices=["mps", "cuda", "cpu"])
    return parser.parse_args()


def run_shift_experiment(shift_name, dataloader, model, device, max_slices, output_dir):
    """Run inference under one shift condition and collect metrics."""
    print(f"\n{'='*60}")
    print(f"Running shift: {shift_name}")
    print(f"{'='*60}")

    all_metrics = []
    n = 0

    for batch in tqdm(dataloader, desc=shift_name):
        if n >= max_slices:
            break

        kspace = batch["kspace"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)
        metadata = batch.get("metadata", {})

        with torch.no_grad():
            reconstruction = model(kspace, mask)

        # Compute metrics: pixel-level + spectral + (optionally) downstream
        metrics = compute_all_metrics(
            reconstruction=reconstruction,
            target=target,
            metadata=metadata
        )
        metrics["shift"] = shift_name
        all_metrics.append(metrics)
        n += kspace.shape[0]

    # Save per-shift results
    out_path = Path(output_dir) / f"{shift_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    avg = {k: float(np.mean([m[k] for m in all_metrics if isinstance(m[k], (int, float))]))
           for k in all_metrics[0] if k != "shift"}
    print(f"  Results for {shift_name}:")
    for k, v in avg.items():
        print(f"    {k}: {v:.4f}")

    return all_metrics


def main():
    args = parse_args()
    device = torch.device(args.device if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    # Load reconstruction model
    model = load_reconstructor(
        model_name=args.model,
        checkpoint=args.checkpoint,
        challenge=args.challenge,
        device=device
    )
    model.eval()

    data_dir = Path(args.data_dir)
    results = {}

    # ------------------------------------------------------------------ #
    # 1. BASELINE — in-distribution                                        #
    # ------------------------------------------------------------------ #
    from shifts.mask_shift import MaskShiftGenerator
    baseline_loader = MaskShiftGenerator(
        data_dir=data_dir / args.split,
        challenge=args.challenge,
        mask_type="random",         # trained distribution
        center_fraction=0.08,
        acceleration=4,
    ).get_dataloader(batch_size=1)

    results["baseline"] = run_shift_experiment(
        "baseline", baseline_loader, model, device, args.max_slices, args.output_dir
    )

    # ------------------------------------------------------------------ #
    # 2. MASK SHIFTS                                                       #
    # ------------------------------------------------------------------ #
    for mask_type in ["equispaced", "magic"]:
        loader = MaskShiftGenerator(
            data_dir=data_dir / args.split,
            challenge=args.challenge,
            mask_type=mask_type,
            center_fraction=0.08,
            acceleration=4,
        ).get_dataloader(batch_size=1)
        results[f"mask_{mask_type}"] = run_shift_experiment(
            f"mask_{mask_type}", loader, model, device, args.max_slices, args.output_dir
        )

    # ------------------------------------------------------------------ #
    # 3. ACCELERATION SHIFTS                                               #
    # ------------------------------------------------------------------ #
    for accel in [6, 8, 16]:
        loader = AccelShiftGenerator(
            data_dir=data_dir / args.split,
            challenge=args.challenge,
            acceleration=accel,
            center_fraction=max(0.01, 0.08 / (accel / 4)),
        ).get_dataloader(batch_size=1)
        results[f"accel_{accel}x"] = run_shift_experiment(
            f"accel_{accel}x", loader, model, device, args.max_slices, args.output_dir
        )

    # ------------------------------------------------------------------ #
    # 4. CONTRAST SHIFTS (brain only)                                      #
    # ------------------------------------------------------------------ #
    brain_dir = data_dir.parent / "brain" / args.split
    if brain_dir.exists():
        for contrast in ["T1", "T1POST", "FLAIR"]:
            loader = ContrastShiftLoader(
                data_dir=brain_dir,
                challenge=args.challenge,
                contrast=contrast,
                acceleration=4,
            ).get_dataloader(batch_size=1)
            results[f"contrast_{contrast}"] = run_shift_experiment(
                f"contrast_{contrast}", loader, model, device, args.max_slices, args.output_dir
            )
    else:
        print(f"Brain data not found at {brain_dir}, skipping contrast shifts.")

    # ------------------------------------------------------------------ #
    # 5. ANATOMY SHIFT — knee model on brain (or vice versa)               #
    # ------------------------------------------------------------------ #
    anatomy_dir = data_dir.parent / "brain" / args.split
    if anatomy_dir.exists():
        loader = AnatomyShiftLoader(
            data_dir=anatomy_dir,
            challenge=args.challenge,
            acceleration=4,
        ).get_dataloader(batch_size=1)
        results["anatomy_brain"] = run_shift_experiment(
            "anatomy_brain", loader, model, device, args.max_slices, args.output_dir
        )

    # Save combined summary
    summary_path = Path(args.output_dir) / "summary.json"
    save_results(results, summary_path)
    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
