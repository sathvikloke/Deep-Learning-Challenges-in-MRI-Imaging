"""
analysis/metrics.py

The core measurement layer — this is where the paper's contribution lives.

Standard metrics (SSIM, PSNR, NMSE) are computed alongside:
  1. Spectral fidelity    — does the reconstruction preserve high-frequency content?
  2. Edge preservation    — are structural boundaries maintained?
  3. Feature suppression  — are small features degraded more than large ones?
  4. LPIPS               — perceptual similarity (optional; graceful fallback if absent)
  5. Metric disagreement — where do SSIM and LPIPS diverge? (hallucination signal)

All functions accept (H, W) or (B, H, W) tensors and return plain Python floats
or dicts of floats, so results are always JSON-serialisable.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Any


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _to_hw(t: torch.Tensor) -> torch.Tensor:
    """Squeeze a (1, H, W) or (H, W) tensor to (H, W). Raises on batch > 1."""
    t = t.squeeze()
    if t.dim() != 2:
        raise ValueError(
            f"Expected a single 2-D slice after squeezing, got shape {t.shape}. "
            f"Iterate over the batch dimension before calling metric functions."
        )
    return t


def _is_degenerate(t: torch.Tensor) -> bool:
    """True if the tensor is blank or contains non-finite values."""
    return bool(t.max() < 1e-6 or not torch.isfinite(t).all())


# ------------------------------------------------------------------ #
# Standard pixel metrics                                               #
# ------------------------------------------------------------------ #

def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Structural Similarity Index.
    Accepts (H, W) tensors; returns a scalar float.
    """
    from torchmetrics.functional import structural_similarity_index_measure as _ssim
    # torchmetrics expects (B, C, H, W)
    p = pred.unsqueeze(0).unsqueeze(0).float()
    t = target.unsqueeze(0).unsqueeze(0).float()
    return float(_ssim(p, t, data_range=data_range).item())


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio. Returns float (inf if pred == target exactly)."""
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def nmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Normalised Mean Squared Error — fastMRI's primary leaderboard metric.
    Returns NaN if target energy is zero (degenerate slice).
    """
    target_energy = (target.float() ** 2).mean().item()
    if target_energy < 1e-12:
        return float("nan")
    return float(F.mse_loss(pred.float(), target.float()).item() / target_energy)


# ------------------------------------------------------------------ #
# Spectral fidelity                                                    #
# ------------------------------------------------------------------ #

def spectral_fidelity(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """
    Compares the 2D power spectra of reconstruction vs. ground truth.

    hf_power_ratio < 1  → model is suppressing fine detail (blurring)
    lf_power_ratio > 1  → model is adding low-frequency energy (hallucination)
    spectral_mse        → overall log-spectrum distortion

    Both tensors must be (H, W), normalised to [0, 1].
    """
    pred_np   = _to_hw(pred).cpu().numpy().astype(np.float32)
    target_np = _to_hw(target).cpu().numpy().astype(np.float32)

    pred_ps   = np.abs(np.fft.fftshift(np.fft.fft2(pred_np)))   ** 2
    target_ps = np.abs(np.fft.fftshift(np.fft.fft2(target_np))) ** 2

    H, W = pred_ps.shape
    cy, cx = H // 2, W // 2
    radius = min(H, W) // 4          # inner quarter = low-frequency region

    y_idx, x_idx = np.ogrid[:H, :W]
    dist    = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    lf_mask = dist <= radius
    hf_mask = ~lf_mask

    eps = 1e-12
    hf_ratio = float(pred_ps[hf_mask].sum() / (target_ps[hf_mask].sum() + eps))
    lf_ratio = float(pred_ps[lf_mask].sum() / (target_ps[lf_mask].sum() + eps))

    log_pred   = np.log1p(pred_ps)
    log_target = np.log1p(target_ps)
    spectral_mse = float(np.mean((log_pred - log_target) ** 2))

    return {
        "hf_power_ratio": hf_ratio,
        "lf_power_ratio": lf_ratio,
        "spectral_mse":   spectral_mse,
    }


# ------------------------------------------------------------------ #
# Edge preservation                                                    #
# ------------------------------------------------------------------ #

def edge_preservation(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Pearson correlation between the Sobel edge maps of pred and target.

    1.0  → edges perfectly preserved
    ~0.0 → edges blurred or destroyed
    <0.0 → shouldn't happen in practice; indicates a degenerate input
    """
    pred_np   = _to_hw(pred).cpu().numpy().astype(np.float32)
    target_np = _to_hw(target).cpu().numpy().astype(np.float32)

    def _sobel(img: np.ndarray) -> np.ndarray:
        gx = np.gradient(img, axis=1)
        gy = np.gradient(img, axis=0)
        return np.sqrt(gx ** 2 + gy ** 2)

    pred_edges   = _sobel(pred_np).ravel()
    target_edges = _sobel(target_np).ravel()

    # Pearson correlation — returns NaN if either vector is constant
    if pred_edges.std() < 1e-8 or target_edges.std() < 1e-8:
        return float("nan")

    corr = float(np.corrcoef(pred_edges, target_edges)[0, 1])
    return corr


# ------------------------------------------------------------------ #
# Feature suppression index                                            #
# ------------------------------------------------------------------ #

def feature_suppression_index(
    pred: torch.Tensor,
    target: torch.Tensor,
    small_threshold: float = 0.1,
    large_threshold: float = 0.5,
) -> dict[str, float]:
    """
    Measures whether low-intensity (small/fine) features are disproportionately
    suppressed compared to high-intensity (large/dominant) structures.

    Requires target normalised to [0, 1].

    suppression_ratio = small_nmse / large_nmse
      > 1  → fine features are degraded more than large structures (bad)
      ~ 1  → uniform degradation across feature sizes
      < 1  → unusually — large structures degrade more (rare)

    Returns NaN for any region with fewer than 10 pixels (insufficient samples).
    """
    p = _to_hw(pred).float()
    t = _to_hw(target).float()

    small_mask = (t > small_threshold) & (t <= large_threshold)
    large_mask = t > large_threshold

    def _region_nmse(mask: torch.Tensor) -> float:
        n = int(mask.sum().item())
        if n < 10:
            return float("nan")
        p_r = p[mask]
        t_r = t[mask]
        energy = (t_r ** 2).mean().item()
        if energy < 1e-12:
            return float("nan")
        return float(((p_r - t_r) ** 2).mean().item() / energy)

    small_nmse_val = _region_nmse(small_mask)
    large_nmse_val = _region_nmse(large_mask)

    if np.isnan(small_nmse_val) or np.isnan(large_nmse_val) or large_nmse_val < 1e-12:
        ratio = float("nan")
    else:
        ratio = float(small_nmse_val / large_nmse_val)

    return {
        "small_feature_nmse": small_nmse_val,
        "large_feature_nmse": large_nmse_val,
        "suppression_ratio":  ratio,
    }


# ------------------------------------------------------------------ #
# LPIPS (perceptual)                                                   #
# ------------------------------------------------------------------ #

_lpips_model = None


def lpips_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Learned Perceptual Image Patch Similarity (lower = more similar).
    Requires: pip install lpips

    Returns NaN if lpips is not installed — all downstream computations
    handle NaN gracefully via np.nanmean.
    """
    global _lpips_model
    try:
        import lpips as _lpips_lib
    except ImportError:
        return float("nan")

    if _lpips_model is None:
        _lpips_model = _lpips_lib.LPIPS(net="alex")
        _lpips_model.eval()

    def _prep(t: torch.Tensor) -> torch.Tensor:
        t = _to_hw(t).float().cpu()
        # LPIPS expects (B, 3, H, W) in [-1, 1]
        t = t.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1)
        return t * 2.0 - 1.0   # [0,1] → [-1,1]

    with torch.no_grad():
        score = _lpips_model(_prep(pred), _prep(target))
    return float(score.item())


# ------------------------------------------------------------------ #
# Metric disagreement                                                  #
# ------------------------------------------------------------------ #

def metric_disagreement(ssim_val: float, lpips_val: float) -> float:
    """
    Disagreement index between SSIM (pixel) and LPIPS (perceptual).

    > 0  → high SSIM but high LPIPS: model looks pixel-accurate but
            perceptually wrong — the hallucination signal we are hunting for.
    < 0  → low SSIM but low LPIPS: model blurs globally but preserves
            perceptual structure (e.g. consistent smoothing).
    NaN  → LPIPS unavailable (lpips not installed).

    Formula: ssim - (1 - clip(lpips, 0, 1))
      Both terms are in [0,1] where 1 = perfect, so the difference is signed
      and zero when SSIM and perceptual quality agree.
    """
    if np.isnan(lpips_val):
        return float("nan")
    lpips_normalised = 1.0 - float(np.clip(lpips_val, 0.0, 1.0))
    return float(ssim_val - lpips_normalised)


# ------------------------------------------------------------------ #
# Master function                                                       #
# ------------------------------------------------------------------ #

def compute_all_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute the full metric suite for a single reconstructed slice.

    Args:
        reconstruction : (H, W) predicted image, values in [0, 1]
        target         : (H, W) ground-truth image, values in [0, 1]
        metadata       : optional dict with fname, slice_idx, etc.
                         (already unpacked from DataLoader batch by the caller)

    Returns:
        Flat dict of metric name → float, plus fname and slice_idx passthrough.
        All values are JSON-serialisable (float or str; NaN → serialised as null
        via the custom encoder in utils/io.py).
    """
    if metadata is None:
        metadata = {}

    # Move to CPU and cast to float32 for all metric computations
    rec = _to_hw(reconstruction).float().cpu()
    tgt = _to_hw(target).float().cpu()

    # Re-normalise defensively in case the caller didn't
    rec_max = rec.max()
    tgt_max = tgt.max()
    rec = rec / (rec_max + 1e-8)
    tgt = tgt / (tgt_max + 1e-8)

    # Early exit for degenerate slices — all metrics will be NaN,
    # which nanmean in the aggregation step handles cleanly.
    if _is_degenerate(tgt):
        nan = float("nan")
        return {
            "ssim": nan, "psnr": nan, "nmse": nan, "lpips": nan,
            "hf_power_ratio": nan, "lf_power_ratio": nan, "spectral_mse": nan,
            "edge_preservation": nan,
            "small_feature_nmse": nan, "large_feature_nmse": nan,
            "suppression_ratio": nan,
            "metric_disagreement": nan,
            "fname":     str(metadata.get("fname", "")),
            "slice_idx": int(metadata.get("slice_idx", -1)),
        }

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
        "ssim":                ssim_val,
        "psnr":                psnr_val,
        "nmse":                nmse_val,
        "lpips":               lpips_val,
        # Spectral
        "hf_power_ratio":      spectral["hf_power_ratio"],
        "lf_power_ratio":      spectral["lf_power_ratio"],
        "spectral_mse":        spectral["spectral_mse"],
        # Edge
        "edge_preservation":   edge_val,
        # Feature suppression
        "small_feature_nmse":  fsi["small_feature_nmse"],
        "large_feature_nmse":  fsi["large_feature_nmse"],
        "suppression_ratio":   fsi["suppression_ratio"],
        # Disagreement
        "metric_disagreement": disagree,
        # Passthrough
        "fname":     str(metadata.get("fname", "")),
        "slice_idx": int(metadata.get("slice_idx", -1))
        if metadata.get("slice_idx") is not None else -1,
    }
