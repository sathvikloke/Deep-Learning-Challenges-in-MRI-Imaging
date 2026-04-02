"""
shifts/mask_shift.py

Generates distribution shifts by swapping the k-space undersampling mask.
The model was trained on random masks — we test on equispaced and magic masks.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import h5py
from pathlib import Path
from typing import Optional

import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import (
    RandomMaskFunc,
    EquispacedMaskFunc,
    MagicMaskFunc,
)


MASK_REGISTRY = {
    "random":      RandomMaskFunc,
    "equispaced":  EquispacedMaskFunc,
    "magic":       MagicMaskFunc,
}


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
        self.data_dir = Path(data_dir)
        self.challenge = challenge
        self.mask_func = mask_func
        self.use_seed = use_seed

        self.examples = []  # list of (filepath, slice_idx)
        for fpath in sorted(self.data_dir.glob("*.h5")):
            with h5py.File(fpath, "r") as f:
                n_slices = f["kspace"].shape[0]
            for s in range(n_slices):
                self.examples.append((fpath, s))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        fpath, slice_idx = self.examples[idx]

        with h5py.File(fpath, "r") as f:
            kspace_np = f["kspace"][slice_idx]           # (coils, H, W)
            target_np = f["reconstruction_rss"][slice_idx] if \
                "reconstruction_rss" in f else None
            attrs = dict(f.attrs)

        # numpy complex → torch float [..., 2]
        kspace = T.to_tensor(kspace_np)                  # (coils, H, W, 2)

        # Apply shift mask
        seed = None if not self.use_seed else tuple(map(ord, str(fpath)))
        masked_kspace, mask, _ = T.apply_mask(kspace, self.mask_func, seed=seed)

        # RSS ground-truth image
        if target_np is not None:
            target = torch.from_numpy(target_np).float()
        else:
            target = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)))

        # Normalise (zero-mean unit-std, standard fastMRI practice)
        mean = masked_kspace.mean()
        std  = masked_kspace.std() + 1e-11
        masked_kspace = (masked_kspace - mean) / std
        target = target / target.max()

        return {
            "kspace":   masked_kspace,   # (coils, H, W, 2)
            "mask":     mask,            # (1, 1, W, 1)
            "target":   target,          # (H, W)
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
    Factory: creates a dataloader with the requested mask type.

    mask_type: one of "random" (trained dist.), "equispaced", "magic"
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
            raise ValueError(f"mask_type must be one of {list(MASK_REGISTRY)}")

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
            pin_memory=False,   # pin_memory not supported on MPS
        )
