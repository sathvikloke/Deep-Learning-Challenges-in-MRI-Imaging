"""
shifts/contrast_shift.py

Distribution shift by filtering fastMRI brain files to a specific contrast.
Model trained on mixed contrasts (T1/T1POST/T2/FLAIR) — we isolate one.

FastMRI brain HDF5 files store their acquisition type in f.attrs["acquisition"].
"""

import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Optional

import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc


# Maps human-readable contrast name → substring to match in attrs["acquisition"]
CONTRAST_MAP = {
    "T1":     "T1",
    "T1POST": "T1POST",
    "T2":     "T2",
    "FLAIR":  "FLAIR",
}


class ContrastFilteredDataset(Dataset):
    """
    Loads only slices whose parent file matches the requested contrast.
    Everything else (mask, normalisation) is identical to the baseline.
    """

    def __init__(
        self,
        data_dir: Path,
        challenge: str,
        contrast: str,
        mask_func,
    ):
        self.challenge = challenge
        self.mask_func = mask_func
        contrast_key = CONTRAST_MAP[contrast]

        self.examples = []
        skipped = 0
        for fpath in sorted(Path(data_dir).glob("*.h5")):
            with h5py.File(fpath, "r") as f:
                acq = f.attrs.get("acquisition", "")
                if contrast_key not in str(acq):
                    skipped += 1
                    continue
                n_slices = f["kspace"].shape[0]
            for s in range(n_slices):
                self.examples.append((fpath, s))

        print(f"[ContrastShift] contrast={contrast}: "
              f"{len(self.examples)} slices kept, {skipped} files skipped")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        fpath, slice_idx = self.examples[idx]

        with h5py.File(fpath, "r") as f:
            kspace_np = f["kspace"][slice_idx]
            target_np = f["reconstruction_rss"][slice_idx] \
                if "reconstruction_rss" in f else None
            attrs = dict(f.attrs)

        kspace = T.to_tensor(kspace_np)
        seed   = tuple(map(ord, str(fpath)))
        masked_kspace, mask, _ = T.apply_mask(kspace, self.mask_func, seed=seed)

        if target_np is not None:
            target = torch.from_numpy(target_np).float()
        else:
            target = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)))

        mean = masked_kspace.mean()
        std  = masked_kspace.std() + 1e-11
        masked_kspace = (masked_kspace - mean) / std
        target = target / target.max()

        return {
            "kspace":   masked_kspace,
            "mask":     mask,
            "target":   target,
            "mean":     mean,
            "std":      std,
            "metadata": {
                "fname":      str(fpath),
                "slice_idx":  slice_idx,
                "attrs":      attrs,
                "contrast":   attrs.get("acquisition", "unknown"),
            },
        }


class ContrastShiftLoader:
    """
    Factory for contrast-filtered dataloaders.

    contrast: one of "T1", "T1POST", "T2", "FLAIR"
    """

    def __init__(
        self,
        data_dir: Path,
        challenge: str = "multicoil",
        contrast: str = "FLAIR",
        acceleration: int = 4,
        center_fraction: float = 0.08,
    ):
        mask_func = RandomMaskFunc(
            center_fractions=[center_fraction],
            accelerations=[acceleration],
        )
        self.dataset = ContrastFilteredDataset(
            data_dir=data_dir,
            challenge=challenge,
            contrast=contrast,
            mask_func=mask_func,
        )

    def get_dataloader(self, batch_size: int = 1, num_workers: int = 0) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
