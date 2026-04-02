"""
shifts/mask_shift.py

Generates distribution shifts by swapping the k-space undersampling mask.
The model was trained on random masks — we test on equispaced masks.

Note: MagicMaskFunc is intentionally excluded because it is not shipped in the
standard fastmri pip package and would cause an ImportError at runtime.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import h5py
from pathlib import Path

import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import (
    RandomMaskFunc,
    EquispacedMaskFunc,
)


MASK_REGISTRY = {
    "random":     RandomMaskFunc,
    "equispaced": EquispacedMaskFunc,
}


def _make_seed(fpath: Path) -> tuple:
    """
    Stable, compact seed derived from the file path.
    Using the full character tuple can produce seeds with 80+ elements
    which some mask functions handle poorly — a single hashed int is safer.
    """
    return (abs(hash(str(fpath))) % (2 ** 31),)


class MaskedSliceDataset(Dataset):
    """
    Loads fastMRI slices and applies a chosen undersampling mask.
    Swapping the mask is the sole distribution shift.
    """

    def __init__(
        self,
        data_dir: Path,
        challenge: str,
        mask_func,
        use_seed: bool = True,
    ):
        self.data_dir  = Path(data_dir)
        self.challenge = challenge
        self.mask_func = mask_func
        self.use_seed  = use_seed

        self.examples: list[tuple[Path, int]] = []
        for fpath in sorted(self.data_dir.glob("*.h5")):
            with h5py.File(fpath, "r") as f:
                n_slices = f["kspace"].shape[0]
            for s in range(n_slices):
                self.examples.append((fpath, s))

        if len(self.examples) == 0:
            raise FileNotFoundError(
                f"No .h5 files found in {self.data_dir}. "
                f"Check that --data_dir points to the correct split folder."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        fpath, slice_idx = self.examples[idx]

        with h5py.File(fpath, "r") as f:
            kspace_np = f["kspace"][slice_idx]           # (coils, H, W), complex64
            target_np = (
                f["reconstruction_rss"][slice_idx]
                if "reconstruction_rss" in f
                else None
            )
            attrs = dict(f.attrs)

        # numpy complex → torch float [..., 2]
        kspace = T.to_tensor(kspace_np)                  # (coils, H, W, 2)

        # Apply undersampling mask (the distribution shift lives here)
        seed = _make_seed(fpath) if self.use_seed else None
        masked_kspace, mask, _ = T.apply_mask(kspace, self.mask_func, seed=seed)

        # Ground-truth RSS image
        if target_np is not None:
            target = torch.from_numpy(target_np.copy()).float()
        else:
            # Reconstruct from fully-sampled kspace as fallback
            target = fastmri.rss(
                fastmri.complex_abs(fastmri.ifft2c(kspace))
            )

        # Guard against degenerate slices (blank images, artefacts)
        target_max = target.max()
        if target_max < 1e-6:
            # Return a zeroed-out slice — metrics will produce NaN and be
            # excluded by nanmean in the aggregation step
            target = target
        else:
            target = target / target_max

        # Normalise kspace to zero-mean unit-std (standard fastMRI practice)
        mean = masked_kspace.mean()
        std  = masked_kspace.std() + 1e-11
        masked_kspace = (masked_kspace - mean) / std

        return {
            "kspace":   masked_kspace,   # (coils, H, W, 2)
            "mask":     mask,            # (1, 1, W, 1)
            "target":   target,          # (H, W)  normalised to [0, 1]
            "mean":     mean,
            "std":      std,
            "metadata": {
                "fname":     str(fpath),
                "slice_idx": slice_idx,
                "attrs":     attrs,
            },
        }


class MaskShiftGenerator:
    """
    Factory: builds a dataloader with the requested undersampling mask type.

    mask_type: "random"  → in-distribution (what the model was trained on)
               "equispaced" → out-of-distribution shift
    """

    def __init__(
        self,
        data_dir: Path,
        challenge: str = "multicoil",
        mask_type: str = "random",
        center_fraction: float = 0.08,
        acceleration: int = 4,
    ):
        if mask_type not in MASK_REGISTRY:
            raise ValueError(
                f"mask_type must be one of {list(MASK_REGISTRY)}, got '{mask_type}'"
            )

        mask_func = MASK_REGISTRY[mask_type](
            center_fractions=[center_fraction],
            accelerations=[acceleration],
        )

        self.dataset = MaskedSliceDataset(
            data_dir=data_dir,
            challenge=challenge,
            mask_func=mask_func,
        )

    def get_dataloader(self, batch_size: int = 1, num_workers: int = 0) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,   # pin_memory is not supported on MPS
        )
