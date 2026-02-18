"""
train.py — GAIA_NN_DIST Training Script
========================================
Trains a neural network to predict corrected GAIA parallax using combined
GAIA + VVV astrometric, photometric, and correlation features.

Saves:
  <OUTPUT_DIR>/model.pt       — model state dict
  <OUTPUT_DIR>/scaler_x.pkl   — fitted input scaler
  <OUTPUT_DIR>/scaler_y.pkl   — fitted output scaler
  <OUTPUT_DIR>/train_stats.csv — per-epoch loss/metric log

Usage:
  python train.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# =============================================================================
# CONFIG — edit these before running
# =============================================================================

DATA_PATH   = "GAIA_NN_DIST_DATA/VVV_GAIA_STANDARDS.csv"
OUTPUT_DIR  = "output/"

EPOCHS      = 2048
BATCH_TRAIN = 16384
BATCH_TEST  = 4096
LEARN_RATE  = 1e-3
LR_FACTOR   = 0.5          # ReduceLROnPlateau reduction factor
LR_PATIENCE = 10           # epochs without improvement before LR drop
EARLY_STOP_PATIENCE = 30   # epochs without improvement before stopping
TEST_SPLIT  = 0.20         # fraction of data held out for validation

# Quality filters applied to training data
FILTERS = {
    "parallax_corr_over_error": ("abs >", 2.0),   # GAIA S/N cut
    "parallax_over_error_vvv":  ("abs >", 2.0),   # VVV S/N cut
    "ipd_frac_multi_peak":      ("abs <", 0.1),   # crowding cut
}

# Input features
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

# Target
Y_FEATURE = "parallax_corr"

# =============================================================================
# MODEL
# =============================================================================

class MLP(nn.Module):
    """
    7-layer MLP: 512 → 256 → 128 → 64 → 32 → 16 → 1
    Each hidden layer: Linear → BatchNorm → ReLU → Dropout(0.2)
    Kaiming initialisation (suitable for ReLU networks).
    """
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# =============================================================================
# DATA
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
    print(f"  Quality filter: {n_before} → {len(df)} rows retained.")
    return df


def load_data(path: str):
    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    df = df.replace([np.nan, np.inf, -np.inf], 0)
    df = apply_filters(df)

    missing_x = [c for c in X_FEATURES if c not in df.columns]
    if missing_x:
        raise ValueError(f"Missing feature columns: {missing_x}")
    if Y_FEATURE not in df.columns:
        raise ValueError(f"Missing target column: {Y_FEATURE}")

    x = df[X_FEATURES].values.astype(np.float32)
    y = df[[Y_FEATURE]].values.astype(np.float32)
    return x, y


class ParallaxDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray,
                 scaler_x: StandardScaler, scaler_y: StandardScaler):
        self.x = torch.from_numpy(scaler_x.transform(x))
        self.y = torch.from_numpy(scaler_y.transform(y))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_loaders(x, y, scaler_x, scaler_y):
    dataset = ParallaxDataset(x, y, scaler_x, scaler_y)
    n_test  = round(TEST_SPLIT * len(dataset))
    n_train = len(dataset) - n_test
    train_ds, test_ds = random_split(dataset, [n_train, n_test])
    train_dl = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,
                          num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_TEST,  shuffle=False,
                          num_workers=4, pin_memory=True)
    print(f"  Train: {n_train} | Validation: {n_test}")
    return train_dl, test_dl

# =============================================================================
# TRAINING
# =============================================================================

def train(model, train_dl, test_dl, device, scaler_y):
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, verbose=True
    )

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    stats         = []

    for epoch in tqdm(range(1, EPOCHS + 1), desc="Training"):
        # ---- train ----
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_dl.dataset)

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        preds_all, targets_all = [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                yhat = model(xb)
                val_loss += criterion(yhat, yb).item() * len(xb)
                preds_all.append(yhat.cpu().numpy())
                targets_all.append(yb.cpu().numpy())
        val_loss /= len(test_dl.dataset)

        preds_np   = scaler_y.inverse_transform(np.vstack(preds_all))
        targets_np = scaler_y.inverse_transform(np.vstack(targets_all))
        r2  = r2_score(targets_np, preds_np)
        mse = mean_squared_error(targets_np, preds_np)
        mae = mean_absolute_error(targets_np, preds_np)

        scheduler.step(val_loss)

        stats.append({
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss, "val_loss": val_loss,
            "r2": r2, "mse": mse, "mae": mae,
        })

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # restore best weights
    model.load_state_dict(best_state)
    return model, pd.DataFrame(stats)

# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    x, y = load_data(DATA_PATH)
    print(f"Dataset: {len(x)} rows, {x.shape[1]} features → 1 target")

    scaler_x = StandardScaler().fit(x)
    scaler_y = StandardScaler().fit(y)

    train_dl, test_dl = build_loaders(x, y, scaler_x, scaler_y)

    model = MLP(x.shape[1]).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    model, stats = train(model, train_dl, test_dl, device, scaler_y)

    # ---- save ----
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
    with open(os.path.join(OUTPUT_DIR, "scaler_x.pkl"), "wb") as f:
        pickle.dump(scaler_x, f)
    with open(os.path.join(OUTPUT_DIR, "scaler_y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)
    stats.to_csv(os.path.join(OUTPUT_DIR, "train_stats.csv"), index=False)

    best = stats.loc[stats["val_loss"].idxmin()]
    print(f"\nBest epoch {int(best.epoch):4d} | val_loss={best.val_loss:.6f} "
          f"| R²={best.r2:.4f} | MAE={best.mae:.4f}")
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
