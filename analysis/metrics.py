"""
analysis/metrics.py

The core measurement layer — this is where the paper's contribution lives.

Standard metrics (SSIM, PSNR) are computed alongside:
  1. Spectral fidelity  — does the reconstruction preserve high-frequency content?
  2. SSIM vs LPIPS disagreement — where do pixel and perceptual metrics diverge?
  3. Edge preservation  — are fine structures (edges) maintained?
  4. Feature suppression index — do small structures survive?

The disagreement between metrics under shift is the key finding we are hunting for.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any


# ------------------------------------------------------------------ #
# Standard pixel metrics                                               #
# ------------------------------------------------------------------ #

def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Structural Similarity Index. Shape: (H, W) or (B, H, W).
    Returns scalar float.
    """
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
    if pred.dim() == 2:
        pred   = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    elif pred.dim() == 3:
        pred   = pred.unsqueeze(1)
        target = target.unsqueeze(1)
    return ssim_fn(pred, target, data_range=data_range).item()


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(data_range ** 2 / mse)


def nmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Normalised Mean Squared Error — fastMRI's primary metric."""
    return (F.mse_loss(pred, target) / (target ** 2).mean()).item()


# ------------------------------------------------------------------ #
# Spectral fidelity                                                    #
# ------------------------------------------------------------------ #

def spectral_fidelity(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compares the 2D power spectra of reconstruction vs. ground truth.

    High-frequency power ratio < 1 means the model is suppressing fine detail.
    Low-frequency power ratio > 1 means the model is hallucinating smooth content.

    Returns:
        hf_ratio   : high-frequency power (pred) / high-frequency power (target)
        lf_ratio   : low-frequency power ratio
        spectral_mse: MSE in log-power-spectrum space
    """
    pred_np   = pred.squeeze().cpu().numpy().astype(np.float32)
    target_np = target.squeeze().cpu().numpy().astype(np.float32)

    # 2D FFT → power spectrum
    pred_ps   = np.abs(np.fft.fftshift(np.fft.fft2(pred_np)))   ** 2
    target_ps = np.abs(np.fft.fftshift(np.fft.fft2(target_np))) ** 2

    H, W = pred_ps.shape
    cy, cx = H // 2, W // 2
    radius = min(H, W) // 4   # inner 25% = low frequency

    # Masks
    y, x = np.ogrid[:H, :W]
    dist  = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    lf_mask = dist <= radius
    hf_mask = ~lf_mask

    pred_lf   = pred_ps[lf_mask].sum()
    pred_hf   = pred_ps[hf_mask].sum()
    target_lf = target_ps[lf_mask].sum() + 1e-12
    target_hf = target_ps[hf_mask].sum() + 1e-12

    # Log-power-spectrum MSE
    log_pred   = np.log1p(pred_ps)
    log_target = np.log1p(target_ps)
    spectral_mse = float(np.mean((log_pred - log_target) ** 2))

    return {
        "hf_power_ratio":  float(pred_hf / target_hf),
        "lf_power_ratio":  float(pred_lf / target_lf),
        "spectral_mse":    spectral_mse,
    }


# ------------------------------------------------------------------ #
# Edge preservation                                                    #
# ------------------------------------------------------------------ #

def edge_preservation(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Sobel-based edge preservation score.
    Returns the correlation between edge maps of pred and target.
    A score close to 1 means edges are well preserved; close to 0 means blurred.
    """
    pred_np   = pred.squeeze().cpu().numpy().astype(np.float32)
    target_np = target.squeeze().cpu().numpy().astype(np.float32)

    def sobel_magnitude(img: np.ndarray) -> np.ndarray:
        # Simple finite-difference Sobel
        gx = np.gradient(img, axis=1)
        gy = np.gradient(img, axis=0)
        return np.sqrt(gx ** 2 + gy ** 2)

    pred_edges   = sobel_magnitude(pred_np).ravel()
    target_edges = sobel_magnitude(target_np).ravel()

    # Pearson correlation
    corr = np.corrcoef(pred_edges, target_edges)[0, 1]
    return float(corr)


# ------------------------------------------------------------------ #
# Feature suppression index                                            #
# ------------------------------------------------------------------ #

def feature_suppression_index(
    pred: torch.Tensor,
    target: torch.Tensor,
    small_threshold: float = 0.1,   # features below this intensity (after norm) count as "small"
    large_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Measures whether small features are disproportionately suppressed.

    Strategy:
      - Binarise target at two thresholds (small vs. large structures)
      - Compute reconstruction fidelity within each mask
      - If small_fidelity << large_fidelity, the model is suppressing fine detail

    Returns:
        small_nmse : NMSE in small-feature regions
        large_nmse : NMSE in large-feature regions
        suppression_ratio : small_nmse / (large_nmse + eps)  — higher = more suppression
    """
    p = pred.squeeze()
    t = target.squeeze()

    small_mask = (t > small_threshold) & (t <= large_threshold)
    large_mask = t > large_threshold

    def region_nmse(mask):
        if mask.sum() < 10:   # too few pixels
            return float("nan")
        p_r = p[mask]
        t_r = t[mask]
        return ((p_r - t_r) ** 2).mean().item() / (t_r ** 2).mean().item()

    small_nmse = region_nmse(small_mask)
    large_nmse = region_nmse(large_mask)

    if np.isnan(small_nmse) or np.isnan(large_nmse) or large_nmse < 1e-12:
        ratio = float("nan")
    else:
        ratio = small_nmse / (large_nmse + 1e-12)

    return {
        "small_feature_nmse":  small_nmse,
        "large_feature_nmse":  large_nmse,
        "suppression_ratio":   ratio,
    }


# ------------------------------------------------------------------ #
# LPIPS (perceptual) — optional, requires lpips package               #
# ------------------------------------------------------------------ #

_lpips_model = None

def lpips_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Learned Perceptual Image Patch Similarity.
    Requires: pip install lpips
    Returns float (lower = more similar, like a distance).
    """
    global _lpips_model
    try:
        import lpips
        if _lpips_model is None:
            _lpips_model = lpips.LPIPS(net="alex")
            _lpips_model.eval()

        def _prep(t):
            t = t.squeeze()
            # LPIPS expects (B, 3, H, W) in [-1, 1]
            t = t.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
            t = (t * 2) - 1   # [0,1] → [-1,1]
            return t.float().cpu()

        with torch.no_grad():
            score = _lpips_model(_prep(pred), _prep(target))
        return float(score.item())
    except ImportError:
        return float("nan")   # gracefully skip if not installed


# ------------------------------------------------------------------ #
# SSIM vs LPIPS disagreement                                           #
# ------------------------------------------------------------------ #

def metric_disagreement(ssim_val: float, lpips_val: float) -> float:
    """
    Disagreement index: high SSIM but high LPIPS means the model is producing
    pixel-accurate but perceptually wrong reconstructions — a hallucination signal.

    score > 0  → model looks good by SSIM but bad perceptually (suspicious)
    score < 0  → model looks bad by SSIM but perceptually OK (blurring)
    """
    if np.isnan(lpips_val):
        return float("nan")
    # Normalise: SSIM in [0,1] where 1=perfect; LPIPS in [0,∞] where 0=perfect
    return float(ssim_val - (1.0 - min(lpips_val, 1.0)))


# ------------------------------------------------------------------ #
# Master function                                                       #
# ------------------------------------------------------------------ #

def compute_all_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    metadata: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """
    Compute the full metric suite for one batch of reconstructions.

    Args:
        reconstruction : (B, H, W) or (H, W) predicted image, values in [0, 1]
        target         : (B, H, W) or (H, W) ground truth image, values in [0, 1]
        metadata       : passthrough dict (fname, slice_idx, etc.)

    Returns:
        dict of all metric values
    """
    # Ensure CPU float for numpy-backed metrics
    rec = reconstruction.float().cpu()
    tgt = target.float().cpu()

    # Normalise to [0, 1] if needed
    rec = rec / (rec.max() + 1e-8)
    tgt = tgt / (tgt.max() + 1e-8)

    ssim_val  = ssim(rec, tgt)
    psnr_val  = psnr(rec, tgt)
    nmse_val  = nmse(rec, tgt)
    lpips_val = lpips_score(rec, tgt)
    spectral  = spectral_fidelity(rec, tgt)
    edge_val  = edge_preservation(rec, tgt)
    fsi       = feature_suppression_index(rec, tgt)
    disagree  = metric_disagreement(ssim_val, lpips_val)

    return {
        # Standard
        "ssim":                 ssim_val,
        "psnr":                 psnr_val,
        "nmse":                 nmse_val,
        "lpips":                lpips_val,
        # Spectral
        "hf_power_ratio":       spectral["hf_power_ratio"],
        "lf_power_ratio":       spectral["lf_power_ratio"],
        "spectral_mse":         spectral["spectral_mse"],
        # Edge
        "edge_preservation":    edge_val,
        # Feature suppression
        "small_feature_nmse":   fsi["small_feature_nmse"],
        "large_feature_nmse":   fsi["large_feature_nmse"],
        "suppression_ratio":    fsi["suppression_ratio"],
        # Disagreement
        "metric_disagreement":  disagree,
        # Passthrough
        "fname":                metadata.get("fname", ""),
        "slice_idx":            metadata.get("slice_idx", -1),
    }
