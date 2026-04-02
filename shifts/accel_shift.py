"""
shifts/accel_shift.py

Distribution shift via acceleration factor.
Model trained at 4x — we test at 6x, 8x, 16x.
Same random mask type, just sparser sampling.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from fastmri.data.subsample import RandomMaskFunc
from shifts.mask_shift import MaskedSliceDataset


class AccelShiftGenerator:
    """
    Keeps mask type (random) constant but raises acceleration factor.
    This isolates under-sampling severity as the sole variable.

    The center_fraction shrinks proportionally so the ACS region
    stays the same physical size as acceleration increases.
    """

    def __init__(
        self,
        data_dir: Path,
        challenge: str = "multicoil",
        acceleration: int = 8,
        center_fraction: float = 0.04,   # caller should scale with accel
    ):
        mask_func = RandomMaskFunc(
            center_fractions=[center_fraction],
            accelerations=[acceleration],
        )

        self.dataset = MaskedSliceDataset(
            data_dir=data_dir,
            challenge=challenge,
            mask_func=mask_func,
        )
        self.acceleration = acceleration

    def get_dataloader(self, batch_size: int = 1, num_workers: int = 0) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
