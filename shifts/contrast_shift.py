"""
shifts/contrast_shift.py

Distribution shift by filtering fastMRI brain files to a specific contrast.
Model trained on mixed contrasts (T1/T1POST/T2/FLAIR) — we isolate one.

FastMRI brain HDF5 files store their acquisition type in f.attrs["acquisition"].

Ordering note: "T1POST" must be checked before "T1" when matching, otherwise
a file with acquisition "T1POST" would incorrectly match the "T1" filter.
This is handled by CONTRAST_MAP using exact string matching via == not `in`.
"""

import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc

from shifts.mask_shift import _make_seed


# Exact acquisition strings as stored in fastMRI brain HDF5 attrs.
# We match with == (not `in`) to prevent "T1" matching "T1POST".
VALID_CONTRASTS = {"T1", "T1POST", "T2", "FLAIR"}


class ContrastFilteredDataset(Dataset):
    """
    Loads only slices whose parent file exactly matches the requested contrast.
    Everything else (mask, normalisation) is identical to the baseline.
    """

    def __init__(
        self,
        data_dir: Path,
        challenge: str,
        contrast: str,
        mask_func,
    ):
        if contrast not in VALID_CONTRASTS:
            raise ValueError(f"contrast must be one of {VALID_CONTRASTS}, got '{contrast}'")

        self.challenge = challenge
        self.mask_func = mask_func

        self.examples: list[tuple[Path, int]] = []
        skipped = 0

        for fpath in sorted(Path(data_dir).glob("*.h5")):
            with h5py.File(fpath, "r") as f:
                # Exact match — avoids "T1" matching "T1POST"
                acq = str(f.attrs.get("acquisition", "")).strip()
                if acq != contrast:
                    skipped += 1
                    continue
                n_slices = f["kspace"].shape[0]
            for s in range(n_slices):
                self.examples.append((fpath, s))

        print(
            f"[ContrastShift] contrast={contrast}: "
            f"{len(self.examples)} slices kept, {skipped} files skipped"
        )
        if len(self.examples) == 0:
            raise RuntimeError(
                f"No slices found for contrast='{contrast}' in {data_dir}.\n"
                f"Check that the brain split contains files with "
                f"attrs['acquisition'] == '{contrast}'."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        fpath, slice_idx = self.examples[idx]

        with h5py.File(fpath, "r") as f:
            kspace_np = f["kspace"][slice_idx]
            target_np = (
                f["reconstruction_rss"][slice_idx]
                if "reconstruction_rss" in f
                else None
            )
            attrs = dict(f.attrs)

        kspace = T.to_tensor(kspace_np)
        seed   = _make_seed(fpath)
        masked_kspace, mask, _ = T.apply_mask(kspace, self.mask_func, seed=seed)

        if target_np is not None:
            target = torch.from_numpy(target_np.copy()).float()
        else:
            target = fastmri.rss(
                fastmri.complex_abs(fastmri.ifft2c(kspace))
            )

        target_max = target.max()
        target = target / target_max if target_max > 1e-6 else target

        mean = masked_kspace.mean()
        std  = masked_kspace.std() + 1e-11
        masked_kspace = (masked_kspace - mean) / std

        return {
            "kspace":   masked_kspace,
            "mask":     mask,
            "target":   target,
            "mean":     mean,
            "std":      std,
            "metadata": {
                "fname":     str(fpath),
                "slice_idx": slice_idx,
                "contrast":  str(attrs.get("acquisition", "unknown")),
                "attrs":     attrs,
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
