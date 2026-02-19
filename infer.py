"""
infer.py — GAIA_NN_DIST Inference Script
==========================================
Loads a trained model produced by train.py and predicts corrected parallax
(and its uncertainty) for all rows in a CSV.

Output columns appended to the input CSV
-----------------------------------------
  parallax_NN        — predicted parallax in original units
                       (parallax_corr if no Lindegren mode,
                        parallax_lindegren + residual_NN otherwise)
  parallax_NN_sigma  — predicted 1-σ uncertainty in the same units

Supports all three backends (mlp / gnn / gp) by reading config.json
from the model directory; no --model-type flag needed.

Usage
-----
  python infer.py --input path/to/stars.csv \
                  --model output/            \
                  --output predictions.csv
"""

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from models import build_model, build_knn_graph
from train import (
    engineer_features,
    CORR_FEATURES,
    LINDEGREN_COLUMN,
    X_FEATURES_OPTIONAL,
)

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
    _PYGEOM_AVAILABLE = True
except ImportError:
    _PYGEOM_AVAILABLE = False

BATCH_SIZE = 65536


# =============================================================================
# LOAD ARTIFACTS
# =============================================================================

def load_artifacts(model_dir: str, device: torch.device):
    config_path  = os.path.join(model_dir, "config.json")
    model_path   = os.path.join(model_dir, "model.pt")
    sx_path      = os.path.join(model_dir, "scaler_x.pkl")
    sy_path      = os.path.join(model_dir, "scaler_y.pkl")

    for p in [config_path, model_path, sx_path, sy_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    with open(config_path) as f:
        config = json.load(f)

    with open(sx_path, "rb") as f:
        scaler_x = pickle.load(f)
    with open(sy_path, "rb") as f:
        scaler_y = pickle.load(f)

    mtype  = config["model_type"]
    kwargs = {}
    if mtype == "gnn":
        kwargs = {"hidden": config["gnn_hidden"], "n_layers": config["gnn_layers"]}
    elif mtype == "gp":
        kwargs = {
            "n_inducing":  config["gp_inducing"],
            "feature_dim": config["gp_feature_dim"],
        }

    model = build_model(mtype, len(config["x_features"]), **kwargs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    if mtype == "gp":
        lik_path = os.path.join(model_dir, "likelihood.pt")
        if os.path.exists(lik_path):
            import gpytorch
            model.likelihood.load_state_dict(
                torch.load(lik_path, map_location=device)
            )
            model.likelihood.eval().to(device)

    model.eval()
    return model, scaler_x, scaler_y, config


# =============================================================================
# PREDICTION
# =============================================================================

def predict_mlp(model, scaler_x, x_raw, device):
    """Batched MLP inference → (mean_scaled, sigma_scaled) arrays."""
    x_sc = scaler_x.transform(x_raw).astype(np.float32)
    means, sigmas = [], []
    with torch.no_grad():
        for start in tqdm(range(0, len(x_sc), BATCH_SIZE), desc="Inference"):
            batch = torch.from_numpy(x_sc[start: start + BATCH_SIZE]).to(device)
            mean, logvar = model(batch)
            means.append(mean.cpu().numpy())
            sigmas.append(np.exp(0.5 * logvar.cpu().numpy()))
    return np.vstack(means), np.vstack(sigmas)


def predict_gnn(model, scaler_x, x_raw, l, b, k, device):
    """GNN inference: build a spatial graph over inference stars, run batched forward."""
    if not _PYGEOM_AVAILABLE:
        raise ImportError("torch_geometric required for GNN inference. pip install torch_geometric")

    x_sc     = torch.from_numpy(scaler_x.transform(x_raw).astype(np.float32))
    edge_idx = build_knn_graph(l, b, k=k)
    data     = Data(x=x_sc, edge_index=edge_idx, num_nodes=len(x_sc))

    loader = NeighborLoader(
        data,
        num_neighbors=[k, k // 2],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    means, sigmas = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            n   = batch.batch_size
            xb  = batch.x.to(device)
            ei  = batch.edge_index.to(device)
            mean, logvar = model(xb, ei)
            means.append(mean[:n].cpu().numpy())
            sigmas.append(np.exp(0.5 * logvar[:n].cpu().numpy()))

    return np.vstack(means), np.vstack(sigmas)


def predict_gp(model, scaler_x, x_raw, device):
    """GP inference — model.eval() forward returns (mean, logvar)."""
    x_sc = scaler_x.transform(x_raw).astype(np.float32)
    means, sigmas = [], []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(x_sc), BATCH_SIZE), desc="Inference"):
            batch = torch.from_numpy(x_sc[start: start + BATCH_SIZE]).to(device)
            mean, logvar = model(batch)
            means.append(mean.cpu().numpy())
            sigmas.append(np.exp(0.5 * logvar.cpu().numpy()))
    return np.vstack(means), np.vstack(sigmas)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GAIA_NN_DIST inference")
    parser.add_argument("--input",  required=True, help="Input CSV path")
    parser.add_argument("--model",  default="output/", help="Model directory")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, scaler_x, scaler_y, config = load_artifacts(args.model, device)
    mtype      = config["model_type"]
    x_features = config["x_features"]
    use_lind   = config.get("use_lindegren", False)
    print(f"Loaded {mtype.upper()} model from {args.model}  "
          f"({len(x_features)} features)")
    if use_lind:
        print(f"Lindegren residual mode active ({LINDEGREN_COLUMN})")

    # ------------------------------------------------------------------
    # Load & engineer input data
    # ------------------------------------------------------------------
    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input, low_memory=False)
    df = engineer_features(df)

    # Add optional columns with zero fill if absent
    for col in X_FEATURES_OPTIONAL:
        if col not in df.columns:
            df[col] = 0.0

    missing = [c for c in x_features if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    x_raw = df[x_features].values.astype(np.float32)
    print(f"Running inference on {len(df):,} rows ...")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    if mtype == "mlp":
        means_sc, sigmas_sc = predict_mlp(model, scaler_x, x_raw, device)
    elif mtype == "gnn":
        l = df["l"].values
        b = df["b"].values
        means_sc, sigmas_sc = predict_gnn(model, scaler_x, x_raw, l, b,
                                           k=config["gnn_k"], device=device)
    elif mtype == "gp":
        means_sc, sigmas_sc = predict_gp(model, scaler_x, x_raw, device)
    else:
        raise ValueError(f"Unknown model_type: {mtype}")

    # ------------------------------------------------------------------
    # Unscale: y_scaled ~ N(0,1) → original parallax units
    #   mean_orig  = scaler_y.mean_ + mean_scaled  * scaler_y.scale_
    #   sigma_orig = sigma_scaled * scaler_y.scale_   (scale, not shift)
    # ------------------------------------------------------------------
    scale = scaler_y.scale_[0]
    mean_orig  = scaler_y.inverse_transform(means_sc).ravel()
    sigma_orig = (sigmas_sc * scale).ravel()

    # If Lindegren residual mode, add back the Lindegren zero-point
    if use_lind and LINDEGREN_COLUMN in df.columns:
        mean_orig = mean_orig + df[LINDEGREN_COLUMN].values
        # uncertainty is unaffected (Lindegren ZP is treated as deterministic)

    df["parallax_NN"]       = mean_orig
    df["parallax_NN_sigma"] = sigma_orig

    df.to_csv(args.output, index=False)
    print(f"\nSaved predictions to {args.output}")
    print(f"parallax_NN       — mean:{mean_orig.mean():.4f}  "
          f"std:{mean_orig.std():.4f}  "
          f"min:{mean_orig.min():.4f}  "
          f"max:{mean_orig.max():.4f}")
    print(f"parallax_NN_sigma — mean:{sigma_orig.mean():.4f}  "
          f"std:{sigma_orig.std():.4f}  "
          f"min:{sigma_orig.min():.4f}  "
          f"max:{sigma_orig.max():.4f}")


if __name__ == "__main__":
    main()
