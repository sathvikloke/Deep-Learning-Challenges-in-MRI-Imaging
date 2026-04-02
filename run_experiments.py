"""
run_experiments.py

Main entry point. Runs all distribution shift experiments and saves results.

Usage:
    python run_experiments.py --data_dir /path/to/fastmri/knee --anatomy knee --split val
"""

import argparse
import json
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
from utils.io import save_results

# Non-numeric metadata keys — excluded from averaging
_SKIP_KEYS = {"shift", "fname", "slice_idx"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe pretrained MRI reconstruction models under distribution shift."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the anatomy split dir, e.g. /data/fastmri/knee")
    parser.add_argument("--challenge", type=str, default="multicoil",
                        choices=["singlecoil", "multicoil"])
    parser.add_argument("--split", type=str, default="val",
                        help="Subdirectory name for evaluation data (e.g. val)")
    parser.add_argument("--anatomy", type=str, default="knee",
                        choices=["knee", "brain"],
                        help="Anatomy the model was trained on — determines which "
                             "pretrained checkpoint to download")
    parser.add_argument("--model", type=str, default="varnet",
                        choices=["varnet"],
                        help="Only VarNet is supported for multicoil data")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a local .pt checkpoint. If omitted, the "
                             "official pretrained weights are downloaded automatically.")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--max_slices", type=int, default=200,
                        help="Maximum number of slices to evaluate per shift "
                             "(use a small number like 10 for a quick smoke-test)")
    parser.add_argument("--device", type=str, default=None,
                        help="One of: mps, cuda, cpu. Auto-detected if omitted.")
    return parser.parse_args()


def _resolve_device(requested: str | None) -> torch.device:
    """Pick the best available device, respecting an explicit request."""
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS is not available.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    # Auto-detect
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _unpack_metadata(metadata: dict) -> dict:
    """
    DataLoader collation wraps scalar values in lists (e.g. {"fname": ["a.h5"]}).
    This unwraps single-element lists so downstream code gets plain scalars/strings.
    """
    unpacked = {}
    for k, v in metadata.items():
        if isinstance(v, (list, tuple)) and len(v) == 1:
            unpacked[k] = v[0]
        elif isinstance(v, torch.Tensor) and v.numel() == 1:
            unpacked[k] = v.item()
        else:
            unpacked[k] = v
    return unpacked


def _safe_avg(all_metrics: list[dict]) -> dict:
    """
    Compute per-key mean over all slice records, skipping non-numeric keys
    and using nanmean so that individual NaN values don't poison the average.
    """
    numeric_keys = [
        k for k in all_metrics[0]
        if k not in _SKIP_KEYS and isinstance(all_metrics[0][k], (int, float))
    ]
    return {
        k: float(np.nanmean([m[k] for m in all_metrics]))
        for k in numeric_keys
    }


def run_shift_experiment(
    shift_name: str,
    dataloader,
    model: torch.nn.Module,
    device: torch.device,
    max_slices: int,
    output_dir: str,
) -> list[dict]:
    """Run inference under one shift condition, compute metrics, and save."""
    print(f"\n{'='*60}")
    print(f"  Shift: {shift_name}")
    print(f"{'='*60}")

    all_metrics = []
    n_slices_seen = 0

    for batch in tqdm(dataloader, desc=shift_name):
        if n_slices_seen >= max_slices:
            break

        kspace = batch["kspace"].to(device)      # (B, coils, H, W, 2)
        mask   = batch["mask"].to(device)        # (B, 1, 1, W, 1)
        target = batch["target"].to(device)      # (B, H, W)
        metadata = _unpack_metadata(batch.get("metadata", {}))

        with torch.no_grad():
            reconstruction = model(kspace, mask) # (B, H, W)

        # Iterate over batch dimension (always 1 here, but correct practice)
        batch_size = kspace.shape[0]
        for i in range(batch_size):
            slice_meta = {k: (v[i] if isinstance(v, (list, tuple)) else v)
                          for k, v in metadata.items()}
            metrics = compute_all_metrics(
                reconstruction=reconstruction[i],
                target=target[i],
                metadata=slice_meta,
            )
            metrics["shift"] = shift_name
            all_metrics.append(metrics)

        n_slices_seen += batch_size

    if not all_metrics:
        print(f"  WARNING: No slices processed for shift '{shift_name}'. "
              f"Check that data_dir is correct.")
        return []

    # Persist per-slice records
    out_path = Path(output_dir) / f"{shift_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # Print summary
    avg = _safe_avg(all_metrics)
    print(f"\n  Summary ({len(all_metrics)} slices):")
    for k, v in avg.items():
        print(f"    {k:30s}: {v:.4f}")

    return all_metrics


def main():
    args = parse_args()
    device = _resolve_device(args.device)
    print(f"\nUsing device: {device}")

    # ------------------------------------------------------------------ #
    # Load the frozen pretrained reconstruction model                      #
    # ------------------------------------------------------------------ #
    model = load_reconstructor(
        model_name=args.model,
        checkpoint=args.checkpoint,
        challenge=args.challenge,
        anatomy=args.anatomy,       # ← was missing in the original
        device=device,
    )
    model.eval()

    data_dir  = Path(args.data_dir)
    split_dir = data_dir / args.split
    results   = {}

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}\n"
            f"Expected structure: <data_dir>/<split>/*.h5"
        )

    # ------------------------------------------------------------------ #
    # 1. BASELINE — in-distribution (random mask, 4×)                     #
    # ------------------------------------------------------------------ #
    baseline_loader = MaskShiftGenerator(
        data_dir=split_dir,
        challenge=args.challenge,
        mask_type="random",
        center_fraction=0.08,
        acceleration=4,
    ).get_dataloader(batch_size=1)

    results["baseline"] = run_shift_experiment(
        "baseline", baseline_loader, model, device, args.max_slices, args.output_dir
    )

    # ------------------------------------------------------------------ #
    # 2. MASK SHIFT — equispaced (magic removed: not in standard fastmri) #
    # ------------------------------------------------------------------ #
    loader = MaskShiftGenerator(
        data_dir=split_dir,
        challenge=args.challenge,
        mask_type="equispaced",
        center_fraction=0.08,
        acceleration=4,
    ).get_dataloader(batch_size=1)
    results["mask_equispaced"] = run_shift_experiment(
        "mask_equispaced", loader, model, device, args.max_slices, args.output_dir
    )

    # ------------------------------------------------------------------ #
    # 3. ACCELERATION SHIFTS — same random mask, sparser sampling         #
    # ------------------------------------------------------------------ #
    for accel in [6, 8, 16]:
        # Center fraction shrinks proportionally to keep ACS region constant
        center_frac = max(0.01, round(0.08 / (accel / 4), 4))
        loader = AccelShiftGenerator(
            data_dir=split_dir,
            challenge=args.challenge,
            acceleration=accel,
            center_fraction=center_frac,
        ).get_dataloader(batch_size=1)
        results[f"accel_{accel}x"] = run_shift_experiment(
            f"accel_{accel}x", loader, model, device, args.max_slices, args.output_dir
        )

    # ------------------------------------------------------------------ #
    # 4. CONTRAST SHIFTS — brain only (requires brain data alongside knee) #
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
                f"contrast_{contrast}", loader, model, device,
                args.max_slices, args.output_dir
            )
    else:
        print(f"\n[INFO] Brain data not found at {brain_dir} — "
              f"skipping contrast shifts.\n"
              f"       Expected: /path/to/brain/{args.split}/*.h5")

    # ------------------------------------------------------------------ #
    # 5. ANATOMY SHIFT — feed the wrong anatomy entirely                  #
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
    else:
        print(f"\n[INFO] Anatomy shift skipped (same dir as contrast shifts).")

    # ------------------------------------------------------------------ #
    # Save combined summary                                                #
    # ------------------------------------------------------------------ #
    summary_path = Path(args.output_dir) / "summary.json"
    save_results(results, summary_path)
    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
