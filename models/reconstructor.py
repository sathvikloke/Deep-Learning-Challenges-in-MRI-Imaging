"""
models/reconstructor.py

Loads pretrained fastMRI reconstruction models (VarNet, U-Net).
Downloads official checkpoints if not cached locally.

These are the models whose robustness we are PROBING — we never modify their weights.

Important notes:
  - VarNet is the recommended model for multicoil data. Use it unless you have
    a specific reason to use U-Net.
  - U-Net is singlecoil only. Passing multicoil data to it will raise a clear
    error rather than silently producing wrong reconstructions.
  - torch.load is called with weights_only=False because fastMRI checkpoints
    contain Python objects beyond raw tensors. This is safe for checkpoints
    downloaded from the official fastMRI URLs.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

import fastmri
from fastmri.models import VarNet, Unet


# Official pretrained checkpoint URLs.
# Verify these are still live before your first run:
#   curl -I <url>
# If dead, download manually from https://github.com/facebookresearch/fastMRI
CHECKPOINT_URLS = {
    "varnet_knee_mc":  "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/knee_leaderboard_state_dict.pt",
    "varnet_brain_mc": "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/brain_leaderboard_state_dict.pt",
    "unet_knee_sc":    "https://dl.fbaipublicfiles.com/fastMRI/trained_models/unet/knee_sc_leaderboard_state_dict.pt",
}

CACHE_DIR = Path.home() / ".cache" / "fastmri_checkpoints"


def _download_checkpoint(url: str, dest: Path) -> None:
    """Download a checkpoint to dest if it is not already cached."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Using cached checkpoint: {dest}")
        return
    print(f"  Downloading checkpoint from:\n    {url}")
    import urllib.request
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        dest.unlink(missing_ok=True)   # remove partial download
        raise RuntimeError(
            f"Failed to download checkpoint from {url}.\n"
            f"Download it manually and pass --checkpoint /path/to/file.pt\n"
            f"Original error: {e}"
        )
    print(f"  Saved to {dest}")


def _build_varnet() -> VarNet:
    """Instantiate VarNet with the hyperparameters used for the leaderboard checkpoint."""
    return VarNet(
        num_cascades=12,
        pools=4,
        chans=18,
        sens_pools=4,
        sens_chans=8,
    )


def _build_unet() -> Unet:
    """Instantiate U-Net with the hyperparameters used for the singlecoil checkpoint."""
    return Unet(
        in_chans=1,
        out_chans=1,
        chans=256,
        num_pool_layers=4,
        drop_prob=0.0,
    )


class ReconstructorWrapper(nn.Module):
    """
    Thin wrapper so VarNet and U-Net share the same call signature:

        reconstruction = model(kspace, mask)   →   (B, H, W) image

    VarNet:  multicoil, operates in k-space domain natively.
    U-Net:   singlecoil only. Raises a clear error if coils > 1.
    """

    def __init__(self, model: nn.Module, model_type: str, challenge: str):
        super().__init__()
        self.model      = model
        self.model_type = model_type
        self.challenge  = challenge

    def forward(self, kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kspace : (B, coils, H, W, 2)  — masked, normalised k-space
            mask   : (B, 1, 1, W, 1)      — binary undersampling mask

        Returns:
            reconstruction : (B, H, W)    — magnitude image in [0, ~1]
        """
        if self.model_type == "varnet":
            # VarNet handles multicoil internally via sensitivity maps
            return self.model(kspace, mask)

        elif self.model_type == "unet":
            n_coils = kspace.shape[1]
            if n_coils > 1:
                raise ValueError(
                    f"U-Net is a singlecoil model but received kspace with "
                    f"{n_coils} coils. Either use --model varnet or provide "
                    f"singlecoil data with --challenge singlecoil."
                )
            # Singlecoil: zero-filled IFFT → magnitude → U-Net
            # kspace shape: (B, 1, H, W, 2) → squeeze coil dim
            image = fastmri.ifft2c(kspace[:, 0])     # (B, H, W, 2) complex
            image = fastmri.complex_abs(image)        # (B, H, W)    magnitude
            image = image.unsqueeze(1)                # (B, 1, H, W) for UNet
            out   = self.model(image)                 # (B, 1, H, W)
            return out.squeeze(1)                     # (B, H, W)

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
    Load a pretrained MRI reconstruction model onto the specified device.

    Args:
        model_name : "varnet" or "unet"
        checkpoint : Path to a local .pt file. If None, downloads the official
                     pretrained weights automatically.
        challenge  : "multicoil" or "singlecoil"
        anatomy    : "knee" or "brain" — selects which pretrained weights to use.
                     Must match the data you intend to evaluate on for a fair
                     in-distribution baseline.
        device     : torch.device to move the model to.

    Returns:
        ReconstructorWrapper in eval mode, moved to device.
    """
    if model_name not in ("varnet", "unet"):
        raise ValueError(f"model_name must be 'varnet' or 'unet', got '{model_name}'")
    if anatomy not in ("knee", "brain"):
        raise ValueError(f"anatomy must be 'knee' or 'brain', got '{anatomy}'")
    if challenge == "multicoil" and model_name == "unet":
        raise ValueError(
            "U-Net does not support multicoil data. "
            "Use --model varnet --challenge multicoil, or "
            "--model unet --challenge singlecoil."
        )

    print(f"\nLoading {model_name} | challenge={challenge} | anatomy={anatomy}")

    # Build architecture
    if model_name == "varnet":
        model    = _build_varnet()
        ckpt_key = f"varnet_{anatomy}_mc"
    else:
        model    = _build_unet()
        ckpt_key = "unet_knee_sc"

    # Resolve checkpoint path
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        ckpt_path = CACHE_DIR / f"{ckpt_key}.pt"
        _download_checkpoint(CHECKPOINT_URLS[ckpt_key], ckpt_path)

    # Load weights.
    # weights_only=False is required because fastMRI checkpoints store Python
    # objects (e.g. OrderedDict subclasses) alongside raw tensors.
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys in state_dict "
              f"(may be harmless for VarNet sensitivity net): {missing[:3]} ...")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys ignored: "
              f"{unexpected[:3]} ...")

    print(f"  Weights loaded from {ckpt_path}")

    wrapper = ReconstructorWrapper(model, model_name, challenge)
    wrapper = wrapper.to(device)
    wrapper.eval()
    return wrapper
