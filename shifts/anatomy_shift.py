"""
shifts/anatomy_shift.py

The most severe shift: evaluate a knee-trained model on brain data (or vice versa).
No k-space manipulation — the shift comes purely from feeding the wrong anatomy.
"""

import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc

from shifts.mask_shift import _make_seed


class AnatomyDataset(Dataset):
    """
    Standard loader for an out-of-distribution anatomy directory.
    Plug in brain_val/ when the model was trained on knee, or vice versa.
    """

    def __init__(self, data_dir: Path, challenge: str, mask_func):
        self.mask_func = mask_func
        self.examples: list[tuple[Path, int]] = []

        for fpath in sorted(Path(data_dir).glob("*.h5")):
            with h5py.File(fpath, "r") as f:
                n_slices = f["kspace"].shape[0]
            for s in range(n_slices):
                self.examples.append((fpath, s))

        if len(self.examples) == 0:
            raise FileNotFoundError(
                f"No .h5 files found in {data_dir} for anatomy shift."
            )
        print(f"[AnatomyShift] {len(self.examples)} slices from {data_dir}")

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
                "attrs":     attrs,
            },
        }


class AnatomyShiftLoader:
    """
    Feeds a different anatomy into a frozen model.
    data_dir should point to the out-of-distribution anatomy split.
    """

    def __init__(
        self,
        data_dir: Path,
        challenge: str = "multicoil",
        acceleration: int = 4,
        center_fraction: float = 0.08,
    ):
        mask_func = RandomMaskFunc(
            center_fractions=[center_fraction],
            accelerations=[acceleration],
        )
        self.dataset = AnatomyDataset(
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
            pin_memory=False,
        )
