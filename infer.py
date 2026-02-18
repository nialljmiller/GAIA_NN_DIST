"""
infer.py — GAIA_NN_DIST Inference Script
==========================================
Loads a trained model + scalers produced by train.py and predicts
corrected parallax for all rows in a CSV.

Output: same CSV with an extra column  `parallax_NN`  appended.

Usage:
  python infer.py --input  path/to/stars.csv \
                  --model  output/             \
                  --output predictions.csv
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# =============================================================================
# CONFIG — must match what was used in train.py
# =============================================================================

X_FEATURES = [
    "phot_bp_rp_excess_factor_corr",
    "ra", "dec", "l", "b", "ecl_lon", "ecl_lat",
    "parallax", "pmra", "pmdec",
    "dec_parallax_corr", "dec_pmdec_corr", "dec_pmra_corr",
    "parallax_pmdec_corr", "parallax_pmra_corr",
    "pm", "pmra_pmdec_corr", "ra_dec_corr", "radial_velocity",
    "ra_parallax_corr", "ra_pmdec_corr", "ra_pmra_corr",
    "ra_vvv", "dec_vvv", "l_vvv", "b_vvv", "parallax_vvv",
    "pmra_vvv", "pmdec_vvv",
    "bp_g", "bp_rp", "g_rp", "grvs_mag",
    "J-K", "H-K", "Z-K", "Y-K",
]

BATCH_SIZE = 65536   # rows per forward pass — increase if you have lots of VRAM

# =============================================================================
# MODEL — must be identical to train.py
# =============================================================================

class MLP(nn.Module):
    def __init__(self, n_inputs: int):
        super().__init__()
        dims = [n_inputs, 512, 256, 128, 64, 32, 16]
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(0.2),
            ]
        layers.append(nn.Linear(16, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# =============================================================================
# INFERENCE
# =============================================================================

def load_artifacts(model_dir: str, device: torch.device):
    scaler_x_path = os.path.join(model_dir, "scaler_x.pkl")
    scaler_y_path = os.path.join(model_dir, "scaler_y.pkl")
    model_path    = os.path.join(model_dir, "model.pt")

    for p in [scaler_x_path, scaler_y_path, model_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    with open(scaler_x_path, "rb") as f:
        scaler_x = pickle.load(f)
    with open(scaler_y_path, "rb") as f:
        scaler_y = pickle.load(f)

    model = MLP(len(X_FEATURES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, scaler_x, scaler_y


def predict(model, scaler_x, scaler_y, x_raw: np.ndarray, device: torch.device) -> np.ndarray:
    """Run batched inference. Returns un-scaled predictions (original parallax units)."""
    x_scaled = scaler_x.transform(x_raw).astype(np.float32)
    preds = []
    with torch.no_grad():
        for start in tqdm(range(0, len(x_scaled), BATCH_SIZE), desc="Inference"):
            batch = torch.from_numpy(x_scaled[start: start + BATCH_SIZE]).to(device)
            out   = model(batch).cpu().numpy()
            preds.append(out)
    preds_scaled = np.vstack(preds)
    return scaler_y.inverse_transform(preds_scaled).ravel()


def main():
    parser = argparse.ArgumentParser(description="GAIA_NN_DIST inference")
    parser.add_argument("--input",  required=True, help="Input CSV path")
    parser.add_argument("--model",  default="output/", help="Directory containing model.pt + scalers")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, scaler_x, scaler_y = load_artifacts(args.model, device)
    print(f"Loaded model from {args.model}")

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input, low_memory=False)
    df = df.replace([float("nan"), float("inf"), float("-inf")], 0)

    missing = [c for c in X_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    x_raw = df[X_FEATURES].values.astype(np.float32)
    print(f"Running inference on {len(df):,} rows ...")

    df["parallax_NN"] = predict(model, scaler_x, scaler_y, x_raw, device)

    df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")
    print(f"parallax_NN — mean: {df['parallax_NN'].mean():.4f}  "
          f"std: {df['parallax_NN'].std():.4f}  "
          f"min: {df['parallax_NN'].min():.4f}  "
          f"max: {df['parallax_NN'].max():.4f}")


if __name__ == "__main__":
    main()
