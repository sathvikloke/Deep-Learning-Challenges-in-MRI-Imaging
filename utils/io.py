"""
utils/io.py

Helpers for saving and loading experiment results.
"""

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict


class _NumpyEncoder(json.JSONEncoder):
    """Serialise numpy scalars and arrays to JSON."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_kspace(fpath: str, slice_idx: int):
    """Load a single k-space slice from an HDF5 file."""
    import h5py
    with h5py.File(fpath, "r") as f:
        kspace = f["kspace"][slice_idx]
        attrs  = dict(f.attrs)
        target = f["reconstruction_rss"][slice_idx] \
            if "reconstruction_rss" in f else None
    return kspace, target, attrs


def save_results(results: Dict[str, Any], path: Path):
    """Save experiment results dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    print(f"Results saved to {path}")
