# fastMRI Robustness Probe

**Where Do Robust Models Actually Fail?**

This codebase measures how deep learning MRI reconstruction models fail under
controlled distribution shifts, using metrics that go beyond SSIM/PSNR to expose
clinically meaningful degradation.

---

## Project Structure

```
fastmri_robustness/
│
├── run_experiments.py        ← Main entry point
│
├── shifts/                   ← Distribution shift generators
│   ├── mask_shift.py         Swap undersampling mask type
│   ├── accel_shift.py        Increase acceleration factor
│   ├── contrast_shift.py     Isolate a single MRI contrast
│   └── anatomy_shift.py      Feed wrong anatomy to model
│
├── models/
│   └── reconstructor.py      Load pretrained VarNet / U-Net
│
├── analysis/
│   ├── metrics.py            ← Core contribution: full metric suite
│   └── plots.py              Generate paper figures
│
└── utils/
    └── io.py                 HDF5 loading, JSON saving
```

---

## Setup

```bash
# 1. Create environment
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install fastmri from source (recommended for latest checkpoints)
pip install git+https://github.com/facebookresearch/fastMRI.git
```

---

## Data Layout

Point `--data_dir` at your fastMRI root. Expected structure:

```
/path/to/fastmri/
├── knee/
│   ├── train/   *.h5
│   └── val/     *.h5
└── brain/
    ├── train/   *.h5
    └── val/     *.h5
```

---

## Running Experiments

```bash
# Full run — knee VarNet probed across all shifts
python run_experiments.py \
    --data_dir /path/to/fastmri/knee \
    --challenge multicoil \
    --split val \
    --model varnet \
    --output_dir results/ \
    --device mps \
    --max_slices 200

# Quick smoke test (5 slices per shift)
python run_experiments.py \
    --data_dir /path/to/fastmri/knee \
    --max_slices 5 \
    --device mps
```

---

## Generating Figures

```bash
python analysis/plots.py --results_dir results/ --output_dir figures/
```

Outputs:
- `fig1_metric_panel.pdf`     — SSIM, NMSE, edge preservation, disagreement
- `fig2_silent_failure.pdf`   — Key scatter: SSIM vs. suppression ratio
- `fig3_spectral.pdf`         — HF vs LF power ratio across shifts

---

## Metrics Explained

| Metric | What it captures |
|--------|-----------------|
| SSIM / PSNR / NMSE | Standard pixel fidelity — what prior work uses |
| `hf_power_ratio` | Does the model preserve high-frequency (fine) detail? |
| `lf_power_ratio` | Is the model hallucinating smooth low-frequency content? |
| `edge_preservation` | Sobel-edge correlation — are structural boundaries intact? |
| `suppression_ratio` | Are small features degraded more than large ones? |
| `metric_disagreement` | SSIM vs. LPIPS gap — detects perceptually wrong reconstructions |

The paper's central claim lives in `suppression_ratio` and `metric_disagreement`:
models that appear robust by SSIM can simultaneously be destroying fine-grained
clinically relevant features.

---

## Shift Severity Ladder

```
Mild ──────────────────────────────────── Severe
  mask_type → accel_factor → contrast → anatomy
```

This ordering lets you make a claim about *how* failure scales, not just *that* it occurs.

---

## Pretrained Checkpoints

Downloaded automatically on first run to `~/.cache/fastmri_checkpoints/`:
- VarNet knee multicoil (Facebook Research, fastMRI leaderboard winner)
- VarNet brain multicoil
- U-Net knee singlecoil

---

## Citation

If you use this code, please cite:
```
[Your paper here — NeurIPS 2026]
```
