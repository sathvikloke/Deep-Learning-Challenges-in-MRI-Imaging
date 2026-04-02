"""
models/reconstructor.py

Loads pretrained fastMRI reconstruction models (VarNet, U-Net).
Downloads official checkpoints if not cached locally.

These are the models whose robustness we are PROBING — we never modify them.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

# fastmri provides reference implementations of VarNet and UNet
from fastmri.models import VarNet, Unet


# Official pretrained checkpoint URLs from the fastMRI repository
CHECKPOINT_URLS = {
    "varnet_knee_mc":    "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/knee_leaderboard_state_dict.pt",
    "varnet_brain_mc":   "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/brain_leaderboard_state_dict.pt",
    "unet_knee_sc":      "https://dl.fbaipublicfiles.com/fastMRI/trained_models/unet/knee_sc_leaderboard_state_dict.pt",
}

CACHE_DIR = Path.home() / ".cache" / "fastmri_checkpoints"


def _download_checkpoint(url: str, dest: Path) -> Path:
    """Download checkpoint if not already cached."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Using cached checkpoint: {dest}")
        return dest
    print(f"  Downloading checkpoint from {url} ...")
    import urllib.request
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")
    return dest


def _build_varnet(challenge: str = "multicoil") -> VarNet:
    """Instantiate VarNet with standard fastMRI hyperparameters."""
    return VarNet(
        num_cascades=12,
        pools=4,
        chans=18,
        sens_pools=4,
        sens_chans=8,
    )


def _build_unet(challenge: str = "singlecoil") -> nn.Module:
    """Instantiate U-Net with standard fastMRI hyperparameters."""
    return Unet(
        in_chans=1,
        out_chans=1,
        chans=256,
        num_pool_layers=4,
        drop_prob=0.0,
    )


class ReconstructorWrapper(nn.Module):
    """
    Thin wrapper so both VarNet and U-Net share the same call signature:
        reconstruction = model(kspace, mask)
    """

    def __init__(self, model: nn.Module, model_type: str):
        super().__init__()
        self.model = model
        self.model_type = model_type

    def forward(self, kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.model_type == "varnet":
            # VarNet expects (batch, coils, H, W, 2) kspace and (batch, 1, 1, W, 1) mask
            return self.model(kspace, mask)
        elif self.model_type == "unet":
            # U-Net (singlecoil) expects the zero-filled image as input
            import fastmri
            from fastmri.data.transforms import center_crop_to_smallest
            # Inverse FFT → magnitude image → feed to UNet
            image = fastmri.ifft2c(kspace)           # complex image
            image = fastmri.complex_abs(image)        # magnitude
            # UNet expects (batch, 1, H, W)
            image = image.unsqueeze(1)
            return self.model(image).squeeze(1)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")


def load_reconstructor(
    model_name: str = "varnet",
    checkpoint: Optional[str] = None,
    challenge: str = "multicoil",
    anatomy: str = "knee",
    device: torch.device = torch.device("cpu"),
) -> ReconstructorWrapper:
    """
    Load a pretrained reconstruction model.

    Args:
        model_name: "varnet" or "unet"
        checkpoint:  path to a local .pt file; if None downloads the official one
        challenge:   "multicoil" or "singlecoil"
        anatomy:     "knee" or "brain" (selects which pretrained weights to use)
        device:      torch device

    Returns:
        ReconstructorWrapper in eval mode on the requested device
    """

    print(f"\nLoading {model_name} ({challenge}, {anatomy}) ...")

    # Build the architecture
    if model_name == "varnet":
        model = _build_varnet(challenge)
        ckpt_key = f"varnet_{anatomy}_mc"
    elif model_name == "unet":
        model = _build_unet(challenge)
        ckpt_key = "unet_knee_sc"
    else:
        raise ValueError(f"model_name must be 'varnet' or 'unet', got {model_name}")

    # Resolve checkpoint
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
    else:
        url       = CHECKPOINT_URLS[ckpt_key]
        ckpt_path = CACHE_DIR / f"{ckpt_key}.pt"
        _download_checkpoint(url, ckpt_path)

    # Load weights
    state_dict = torch.load(ckpt_path, map_location="cpu")
    # fastMRI checkpoints are sometimes wrapped in {"state_dict": ...}
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    print(f"  Weights loaded from {ckpt_path}")

    wrapper = ReconstructorWrapper(model, model_name)
    wrapper = wrapper.to(device)
    wrapper.eval()
    return wrapper
