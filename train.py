"""
train.py — GAIA_NN_DIST Training Script
========================================
Trains a neural network to predict corrected GAIA parallax (or its residual
relative to the Lindegren+2021 zero-point) from combined GAIA + VVV features.

Improvements over baseline
--------------------------
  * Three interchangeable backends: --model-type {mlp, gnn, gp}
  * Scaler fitted on training split only (no data leakage)
  * Sky-patch split: holds out contiguous sky regions rather than random rows
  * Fisher (arctanh) transform applied to bounded correlation coefficients
  * Radial-velocity missingness indicator feature (rv_missing)
  * New GAIA quality features: ruwe, astrometric_n_good_obs_al,
    visibility_periods_used
  * Optional extinction columns (a_ks, ebv) added when present in CSV
  * Optional Lindegren residual mode: predict (parallax_corr − parallax_lindegren)
    rather than parallax_corr directly, giving the network an easier target
  * Heteroscedastic Gaussian NLL loss with optional per-star weighting by
    1/parallax_error² (if parallax_error column is present)
  * LayerNorm replaces BatchNorm + Dropout (avoids their adverse interaction)
  * GNN backend: spatially-aware via k-NN graph on (l, b); uses NeighborLoader
  * GP backend: Deep Kernel Learning with Sparse Variational GP (ELBO training)

Saves to OUTPUT_DIR
-------------------
  model.pt          trained model weights (+ GP likelihood if applicable)
  scaler_x.pkl      StandardScaler for inputs (fitted on train split only)
  scaler_y.pkl      StandardScaler for target (fitted on train split only)
  config.json       hyperparameters + feature list (read by infer.py)
  train_stats.csv   per-epoch loss / metric log

Usage
-----
  python train.py --model-type mlp
  python train.py --model-type gnn --gnn-k 16 --gnn-hidden 256 --gnn-layers 3
  python train.py --model-type gp  --gp-inducing 512 --gp-feature-dim 16
"""

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from models import build_model, build_knn_graph

try:
    import gpytorch
    from gpytorch.mlls import VariationalELBO
    _GPYTORCH_AVAILABLE = True
except ImportError:
    _GPYTORCH_AVAILABLE = False

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
    _PYGEOM_AVAILABLE = True
except ImportError:
    _PYGEOM_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIG — edit before running
# =============================================================================

DATA_PATH  = "GAIA_NN_DIST_DATA/VVV_GAIA_STANDARDS.csv"
OUTPUT_DIR = "output/"

EPOCHS             = 2048
BATCH_TRAIN        = 16384
BATCH_TEST         = 4096
LEARN_RATE         = 1e-3
LR_FACTOR          = 0.5
LR_PATIENCE        = 10
EARLY_STOP_PATIENCE = 30
TEST_SPLIT         = 0.20         # fraction of sky patches held out
PATCH_SIZE_DEG     = 5.0          # sky-patch grid cell size for spatial split

# Quality filters
FILTERS = {
    "parallax_corr_over_error": ("abs >", 2.0),
    "parallax_over_error_vvv":  ("abs >", 2.0),
    "ipd_frac_multi_peak":      ("abs <", 0.1),
}

# ---------- Feature lists ----------

# Correlation features that receive a Fisher (arctanh) transform before scaling
CORR_FEATURES = [
    "dec_parallax_corr", "dec_pmdec_corr", "dec_pmra_corr",
    "parallax_pmdec_corr", "parallax_pmra_corr",
    "pmra_pmdec_corr", "ra_dec_corr", "ra_parallax_corr",
    "ra_pmdec_corr", "ra_pmra_corr",
]

# Core features (always required)
X_FEATURES_CORE = [
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
    # New GAIA quality indicators
    "ruwe",
    "astrometric_n_good_obs_al",
    "visibility_periods_used",
    # Derived missingness indicator (always present after feature engineering)
    "rv_missing",
]

# Optional columns — included automatically if present in the CSV
X_FEATURES_OPTIONAL = ["a_ks", "ebv"]

# Target and auxiliary columns
Y_FEATURE          = "parallax_corr"
LINDEGREN_COLUMN   = "parallax_lindegren"   # subtract from target if present
WEIGHT_COLUMN      = "parallax_error"        # 1/error² sample weights if present


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature transformations in-place and return the modified df.

    1. Record rv_missing BEFORE filling NaNs.
    2. Fill NaN / ±Inf with 0.
    3. Apply Fisher (arctanh) transform to bounded correlation features.
    4. Add optional columns if absent (filled with 0).
    """
    # 1 — missingness indicator for radial_velocity (very incomplete in DR3)
    if "radial_velocity" in df.columns:
        df["rv_missing"] = df["radial_velocity"].isna().astype(np.float32)
    else:
        df["rv_missing"] = 0.0

    # 2 — fill NaN / Inf
    df = df.replace([np.nan, np.inf, -np.inf], 0)

    # 3 — Fisher transform correlation coefficients
    for col in CORR_FEATURES:
        if col in df.columns:
            vals = df[col].values.clip(-0.9999, 0.9999)
            df[col] = np.arctanh(vals)

    return df


def build_feature_list(df: pd.DataFrame) -> list[str]:
    """Return the final ordered feature list, including any optional columns present."""
    features = list(X_FEATURES_CORE)
    for col in X_FEATURES_OPTIONAL:
        if col in df.columns:
            features.append(col)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return features


# =============================================================================
# DATA LOADING & SPLITTING
# =============================================================================

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, (op, val) in FILTERS.items():
        if col not in df.columns:
            print(f"  [warn] filter column '{col}' not found, skipping.")
            continue
        if op == "abs >":
            mask &= df[col].abs() > val
        elif op == "abs <":
            mask &= df[col].abs() < val
    n_before = len(df)
    df = df[mask]
    print(f"  Quality filter: {n_before:,} → {len(df):,} rows retained.")
    return df


def sky_patch_split(
    df: pd.DataFrame,
    test_frac: float = TEST_SPLIT,
    patch_size_deg: float = PATCH_SIZE_DEG,
    seed: int = 42,
):
    """
    Divide the sky into a (l, b) grid and hold out ~test_frac of the patches.
    This ensures validation stars are in entirely different sky regions from
    training stars, giving honest generalisation metrics.

    Returns boolean arrays train_mask, val_mask.
    """
    l = df["l"].values
    b = df["b"].values
    l_bin = np.floor(l / patch_size_deg).astype(int)
    b_bin = np.floor(b / patch_size_deg).astype(int)
    patch_id = l_bin * 10000 + b_bin       # unique patch identifier

    unique_patches = np.unique(patch_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_patches)
    n_test = max(1, int(test_frac * len(unique_patches)))
    test_set = set(unique_patches[:n_test])

    val_mask   = np.array([p in test_set for p in patch_id])
    train_mask = ~val_mask
    print(f"  Sky-patch split: {train_mask.sum():,} train | {val_mask.sum():,} val "
          f"({len(unique_patches)} patches, {n_test} held out)")
    return train_mask, val_mask


def load_data(path: str):
    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    df = apply_filters(df)
    df = engineer_features(df)

    x_features = build_feature_list(df)

    if Y_FEATURE not in df.columns:
        raise ValueError(f"Missing target column: {Y_FEATURE}")

    # Lindegren residual mode
    use_lindegren = LINDEGREN_COLUMN in df.columns
    if use_lindegren:
        print(f"  Lindegren residual mode: predicting "
              f"({Y_FEATURE} − {LINDEGREN_COLUMN})")
        y_vals = (df[Y_FEATURE] - df[LINDEGREN_COLUMN]).values.astype(np.float32)
    else:
        y_vals = df[Y_FEATURE].values.astype(np.float32)

    x_vals = df[x_features].values.astype(np.float32)

    # Per-star loss weights: 1 / parallax_error² (normalised)
    if WEIGHT_COLUMN in df.columns:
        w = 1.0 / (df[WEIGHT_COLUMN].values.astype(np.float32).clip(min=1e-6) ** 2)
        w = (w / w.mean()).astype(np.float32)
        print(f"  Using {WEIGHT_COLUMN} for per-star loss weighting.")
    else:
        w = np.ones(len(df), dtype=np.float32)

    return x_vals, y_vals[:, None], w, x_features, df, use_lindegren


# =============================================================================
# DATASETS & DATALOADERS
# =============================================================================

class ParallaxDataset(Dataset):
    def __init__(self, x, y, w, scaler_x, scaler_y):
        self.x = torch.from_numpy(scaler_x.transform(x).astype(np.float32))
        self.y = torch.from_numpy(scaler_y.transform(y).astype(np.float32))
        self.w = torch.from_numpy(w.astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]


def build_mlp_gp_loaders(x_tr, y_tr, w_tr, x_va, y_va, w_va, scaler_x, scaler_y):
    train_ds = ParallaxDataset(x_tr, y_tr, w_tr, scaler_x, scaler_y)
    val_ds   = ParallaxDataset(x_va, y_va, w_va, scaler_x, scaler_y)
    train_dl = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_TEST,  shuffle=False,
                          num_workers=4, pin_memory=True)
    return train_dl, val_dl


def build_gnn_loaders(x_tr, y_tr, w_tr, x_va, y_va, w_va,
                      l_tr, b_tr, l_va, b_va,
                      scaler_x, scaler_y, k: int = 16):
    """
    Build PyG NeighborLoader objects for the GNN.
    Training and validation graphs are constructed independently to prevent
    data leakage through the neighbourhood structure.
    """
    if not _PYGEOM_AVAILABLE:
        raise ImportError("torch_geometric is required for GNN. pip install torch_geometric")

    def make_pyg_data(x_raw, y_raw, w_raw, l, b):
        x_sc = torch.from_numpy(scaler_x.transform(x_raw).astype(np.float32))
        y_sc = torch.from_numpy(scaler_y.transform(y_raw).astype(np.float32))
        w_t  = torch.from_numpy(w_raw.astype(np.float32))
        ei   = build_knn_graph(l, b, k=k)
        return Data(x=x_sc, y=y_sc, w=w_t, edge_index=ei, num_nodes=len(x_sc))

    train_data = make_pyg_data(x_tr, y_tr, w_tr, l_tr, b_tr)
    val_data   = make_pyg_data(x_va, y_va, w_va, l_va, b_va)

    # NeighborLoader samples 2-hop subgraphs for each mini-batch
    train_dl = NeighborLoader(
        train_data,
        num_neighbors=[k, k // 2],
        batch_size=min(BATCH_TRAIN, len(x_tr)),
        shuffle=True,
        num_workers=4,
    )
    val_dl = NeighborLoader(
        val_data,
        num_neighbors=[k, k // 2],
        batch_size=min(BATCH_TEST, len(x_va)),
        shuffle=False,
        num_workers=4,
    )
    return train_dl, val_dl


# =============================================================================
# LOSS
# =============================================================================

def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor,
                 target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Heteroscedastic Gaussian negative log-likelihood:
        L = 0.5 * ( logvar + (y − μ)² · exp(−logvar) )
    Optionally weighted by per-star importance weights.
    """
    precision = torch.exp(-logvar)
    nll = 0.5 * (logvar + (target - mean).pow(2) * precision)
    return (nll * weights.unsqueeze(-1)).mean()


# =============================================================================
# TRAINING LOOPS
# =============================================================================

def _run_epoch_mlp_gnn(model, loader, optimizer, device,
                        is_gnn: bool, training: bool):
    """One epoch for MLP or GNN."""
    model.train() if training else model.eval()
    total_loss = 0.0
    preds_all, targets_all = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            if is_gnn:
                # NeighborLoader batch: first .batch_size rows are target nodes
                n = batch.batch_size
                xb  = batch.x.to(device)
                ei  = batch.edge_index.to(device)
                yb  = batch.y[:n].to(device)
                wb  = batch.w[:n].to(device)
                mean, logvar = model(xb, ei)
                mean   = mean[:n]
                logvar = logvar[:n]
            else:
                xb, yb, wb = [t.to(device) for t in batch]
                mean, logvar = model(xb)

            loss = gaussian_nll(mean, logvar, yb, wb)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(yb)
            preds_all.append(mean.detach().cpu().numpy())
            targets_all.append(yb.detach().cpu().numpy())

    n_samples = sum(len(p) for p in preds_all)
    return total_loss / n_samples, np.vstack(preds_all), np.vstack(targets_all)


def _run_epoch_gp(model, loader, optimizer, mll, device, training: bool):
    """One epoch for the GP backend (ELBO loss, no logvar head)."""
    model.train()  if training else model.eval()
    model.likelihood.train() if training else model.likelihood.eval()
    total_loss = 0.0
    preds_all, targets_all = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for xb, yb, _wb in loader:
            xb = xb.to(device)
            yb = yb.to(device).squeeze()

            if training:
                output = model(xb)          # MultivariateNormal in train mode
                loss   = -mll(output, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                model.eval()
                mean_t, logvar_t = model(xb)   # eval mode → (mean, logvar)
                loss = torch.tensor(0.0)
                preds_all.append(mean_t.cpu().numpy())
                targets_all.append(yb.cpu().unsqueeze(-1).numpy())
                total_loss += loss.item() * len(yb)
                continue

            total_loss += loss.item() * len(yb)
            # Collect predictions for metrics (eval forward on same batch)
            model.eval()
            mean_t, _ = model(xb)
            model.train()
            preds_all.append(mean_t.detach().cpu().numpy())
            targets_all.append(yb.detach().cpu().unsqueeze(-1).numpy())

    n_samples = sum(len(p) for p in preds_all) or 1
    return total_loss / n_samples, np.vstack(preds_all), np.vstack(targets_all)


def train(model, model_type, train_dl, val_dl, device, scaler_y, n_train):
    optimizer = optim.Adam(
        list(model.parameters()) +
        (list(model.likelihood.parameters()) if model_type == "gp" else []),
        lr=LEARN_RATE,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, verbose=True
    )

    mll = None
    if model_type == "gp":
        if not _GPYTORCH_AVAILABLE:
            raise ImportError("gpytorch required for GP training.")
        mll = VariationalELBO(model.likelihood, model.gp, num_data=n_train)

    is_gnn = model_type == "gnn"
    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    stats         = []

    for epoch in tqdm(range(1, EPOCHS + 1), desc="Training"):
        if model_type == "gp":
            tr_loss, tr_preds, tr_targ = _run_epoch_gp(
                model, train_dl, optimizer, mll, device, training=True)
            model.eval()
            va_loss, va_preds, va_targ = _run_epoch_gp(
                model, val_dl, optimizer, mll, device, training=False)
            model.train()
        else:
            tr_loss, tr_preds, tr_targ = _run_epoch_mlp_gnn(
                model, train_dl, optimizer, device, is_gnn, training=True)
            va_loss, va_preds, va_targ = _run_epoch_mlp_gnn(
                model, val_dl, optimizer, device, is_gnn, training=False)

        # Compute metrics in original units
        va_preds_orig = scaler_y.inverse_transform(va_preds)
        va_targ_orig  = scaler_y.inverse_transform(va_targ)
        r2  = r2_score(va_targ_orig, va_preds_orig)
        mse = mean_squared_error(va_targ_orig, va_preds_orig)
        mae = mean_absolute_error(va_targ_orig, va_preds_orig)

        scheduler.step(va_loss)

        stats.append({
            "epoch":      epoch,
            "lr":         optimizer.param_groups[0]["lr"],
            "train_loss": tr_loss,
            "val_loss":   va_loss,
            "r2":         r2,
            "mse":        mse,
            "mae":        mae,
        })

        if va_loss < best_val_loss - 1e-5:
            best_val_loss = va_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if model_type == "gp":
                best_lik_state = {k: v.cpu().clone()
                                  for k, v in model.likelihood.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    if model_type == "gp":
        model.likelihood.load_state_dict(best_lik_state)

    return model, pd.DataFrame(stats)


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="GAIA_NN_DIST training")
    p.add_argument("--model-type", default="mlp",
                   choices=["mlp", "gnn", "gp"],
                   help="Model backend (default: mlp)")
    # GNN options
    p.add_argument("--gnn-k",      type=int, default=16,  help="GNN k-NN neighbours")
    p.add_argument("--gnn-hidden", type=int, default=256, help="GNN hidden dim")
    p.add_argument("--gnn-layers", type=int, default=3,   help="GNN conv layers")
    # GP options
    p.add_argument("--gp-inducing",    type=int, default=512, help="GP inducing points")
    p.add_argument("--gp-feature-dim", type=int, default=16,  help="GP feature dim")
    return p.parse_args()


def main():
    args   = parse_args()
    mtype  = args.model_type
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Backend : {mtype}")

    # ------------------------------------------------------------------
    # Load & engineer features
    # ------------------------------------------------------------------
    x, y, w, x_features, df, use_lindegren = load_data(DATA_PATH)
    print(f"Dataset : {len(x):,} rows | {len(x_features)} features → 1 target")

    # ------------------------------------------------------------------
    # Sky-patch split
    # ------------------------------------------------------------------
    l_col = df["l"].values
    b_col = df["b"].values
    train_mask, val_mask = sky_patch_split(df)

    x_tr, y_tr, w_tr = x[train_mask], y[train_mask], w[train_mask]
    x_va, y_va, w_va = x[val_mask],   y[val_mask],   w[val_mask]

    # ------------------------------------------------------------------
    # Fit scalers on TRAIN only (no leakage)
    # ------------------------------------------------------------------
    scaler_x = StandardScaler().fit(x_tr)
    scaler_y = StandardScaler().fit(y_tr)

    # ------------------------------------------------------------------
    # Build loaders
    # ------------------------------------------------------------------
    if mtype == "gnn":
        l_tr, b_tr = l_col[train_mask], b_col[train_mask]
        l_va, b_va = l_col[val_mask],   b_col[val_mask]
        train_dl, val_dl = build_gnn_loaders(
            x_tr, y_tr, w_tr, x_va, y_va, w_va,
            l_tr, b_tr, l_va, b_va,
            scaler_x, scaler_y, k=args.gnn_k,
        )
    else:
        train_dl, val_dl = build_mlp_gp_loaders(
            x_tr, y_tr, w_tr, x_va, y_va, w_va, scaler_x, scaler_y
        )

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model_kwargs = {}
    if mtype == "gnn":
        model_kwargs = {"hidden": args.gnn_hidden, "n_layers": args.gnn_layers}
    elif mtype == "gp":
        model_kwargs = {
            "n_inducing":  args.gp_inducing,
            "feature_dim": args.gp_feature_dim,
        }

    model = build_model(mtype, len(x_features), **model_kwargs).to(device)
    print(f"Params  : {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    model, stats = train(model, mtype, train_dl, val_dl, device,
                         scaler_y, n_train=len(x_tr))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
    if mtype == "gp":
        torch.save(model.likelihood.state_dict(),
                   os.path.join(OUTPUT_DIR, "likelihood.pt"))

    with open(os.path.join(OUTPUT_DIR, "scaler_x.pkl"), "wb") as f:
        pickle.dump(scaler_x, f)
    with open(os.path.join(OUTPUT_DIR, "scaler_y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)

    config = {
        "model_type":       mtype,
        "x_features":       x_features,
        "use_lindegren":    use_lindegren,
        "lindegren_column": LINDEGREN_COLUMN,
        "gnn_k":            args.gnn_k,
        "gnn_hidden":       args.gnn_hidden,
        "gnn_layers":       args.gnn_layers,
        "gp_inducing":      args.gp_inducing,
        "gp_feature_dim":   args.gp_feature_dim,
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    stats.to_csv(os.path.join(OUTPUT_DIR, "train_stats.csv"), index=False)

    best = stats.loc[stats["val_loss"].idxmin()]
    print(f"\nBest epoch {int(best.epoch):4d} | val_loss={best.val_loss:.6f} "
          f"| R²={best.r2:.4f} | MAE={best.mae:.4f}")
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
