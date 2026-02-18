# GAIA_NN_DIST

A neural network that combines **GAIA DR3** and **VVV** parallaxes to produce improved distance estimates toward the Galactic bulge and plane — regions where GAIA alone is crowding-limited and VVV alone lacks astrometric precision.

---

## How it works

The model is a 7-layer MLP trained to predict a corrected GAIA parallax (`parallax_corr`) from a set of 37 GAIA + VVV astrometric, photometric, and correlation features. By jointly leveraging the high astrometric precision of GAIA and the near-infrared depth of VVV, the network learns to denoise parallaxes in crowded sightlines.

---

## Repository structure

```
GAIA_NN_DIST/
├── train.py          ← train the model
├── infer.py          ← run predictions on new data
├── README.md
└── GAIA_NN_DIST_DATA/
    └── VVV_GAIA_STANDARDS.csv   ← training data (not included)
```

---

## Setup

```bash
pip install torch numpy pandas scikit-learn tqdm
```

---

## Training

Open `train.py` and set the paths and hyperparameters at the top of the file under **CONFIG**:

```python
DATA_PATH  = "GAIA_NN_DIST_DATA/VVV_GAIA_STANDARDS.csv"
OUTPUT_DIR = "output/"
EPOCHS     = 2048
LEARN_RATE = 1e-3
```

The quality filters (S/N cuts, crowding cut) are also set there.

Then run:

```bash
python train.py
```

This will save the following to `OUTPUT_DIR`:

| File | Contents |
|---|---|
| `model.pt` | trained model weights |
| `scaler_x.pkl` | fitted StandardScaler for inputs |
| `scaler_y.pkl` | fitted StandardScaler for the target |
| `train_stats.csv` | per-epoch loss, R², MAE log |

Training uses early stopping (default patience = 30 epochs) and ReduceLROnPlateau scheduling. It runs on GPU automatically if one is available.

---

## Inference

```bash
python infer.py \
  --input  path/to/stars.csv \
  --model  output/ \
  --output predictions.csv
```

The output CSV is identical to the input with one extra column: **`parallax_NN`** — the NN-predicted corrected parallax in the same units as `parallax_corr`.

The input CSV must contain the same 37 feature columns used during training (listed in `X_FEATURES` in both scripts). Missing values / NaNs / Infs are filled with 0 before inference.

---

## Input features

| Group | Features |
|---|---|
| GAIA astrometry | `ra`, `dec`, `l`, `b`, `ecl_lon`, `ecl_lat`, `parallax`, `pmra`, `pmdec`, `pm`, `radial_velocity` |
| GAIA correlations | `dec_parallax_corr`, `dec_pmdec_corr`, `dec_pmra_corr`, `parallax_pmdec_corr`, `parallax_pmra_corr`, `pmra_pmdec_corr`, `ra_dec_corr`, `ra_parallax_corr`, `ra_pmdec_corr`, `ra_pmra_corr` |
| GAIA photometry | `phot_bp_rp_excess_factor_corr`, `bp_g`, `bp_rp`, `g_rp`, `grvs_mag` |
| VVV astrometry | `ra_vvv`, `dec_vvv`, `l_vvv`, `b_vvv`, `parallax_vvv`, `pmra_vvv`, `pmdec_vvv` |
| VVV photometry | `J-K`, `H-K`, `Z-K`, `Y-K` |

**Target:** `parallax_corr` (bias-corrected GAIA parallax)

---

## Training data quality filters

Applied before training (configurable in `train.py`):

| Column | Cut | Purpose |
|---|---|---|
| `parallax_corr_over_error` | `abs > 2.0` | GAIA S/N cut |
| `parallax_over_error_vvv` | `abs > 2.0` | VVV S/N cut |
| `ipd_frac_multi_peak` | `abs < 0.1` | Crowding / blending cut |
