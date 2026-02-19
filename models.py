"""
models.py — GAIA_NN_DIST Model Architectures
==============================================
Three interchangeable model backends, all exposing the same interface:

    forward(x, edge_index=None) → (mean, logvar)   [eval mode]
    forward(x, edge_index=None) → GP distribution  [train mode, GP only]

Backends
--------
  mlp  — 7-layer MLP with LayerNorm, dual output heads (mean + log-variance)
  gnn  — GraphSAGE GNN; nodes = stars, edges = spatial k-NN on (l, b)
  gp   — Deep Kernel Learning: MLP feature extractor + Sparse Variational GP

Dependencies
------------
  mlp : torch (always available)
  gnn : pip install torch_geometric
  gp  : pip install gpytorch
"""

import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

try:
    from torch_geometric.nn import SAGEConv
    _PYGEOM_AVAILABLE = True
except ImportError:
    _PYGEOM_AVAILABLE = False

try:
    import gpytorch
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import (
        CholeskyVariationalDistribution,
        VariationalStrategy,
    )
    _GPYTORCH_AVAILABLE = True
except ImportError:
    _GPYTORCH_AVAILABLE = False


# ===========================================================================
# MLP
# ===========================================================================

class MLPModel(nn.Module):
    """
    7-layer MLP with LayerNorm (replaces BatchNorm + Dropout which interact
    poorly together).  Dual output heads: mean and log-variance, enabling
    heteroscedastic uncertainty estimation.

    Architecture: n_inputs → 512 → 256 → 128 → 64 → 32 → 16 → {mean, logvar}
    """

    def __init__(self, n_inputs: int):
        super().__init__()
        dims = [n_inputs, 512, 256, 128, 64, 32, 16]
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.ReLU(),
            ]
        self.backbone = nn.Sequential(*layers)
        self.mean_head   = nn.Linear(16, 1)
        self.logvar_head = nn.Linear(16, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, edge_index=None):
        h      = self.backbone(x)
        mean   = self.mean_head(h)
        logvar = self.logvar_head(h)
        return mean, logvar


# ===========================================================================
# GNN
# ===========================================================================

class GNNModel(nn.Module):
    """
    GraphSAGE network for spatially-aware parallax correction.

    Each star is a node; directed edges connect each star to its k nearest
    neighbours on the sky (computed from galactic l, b before training and
    re-computed at inference time).  Message passing lets each star borrow
    astrometric context from nearby stars, capturing scanning-law and
    crowding systematics that are spatially correlated.

    Architecture:
        Linear input projection → N × SAGEConv(LayerNorm + ReLU) → {mean, logvar}

    Requires: pip install torch_geometric
    """

    def __init__(self, n_inputs: int, hidden: int = 256, n_layers: int = 3):
        if not _PYGEOM_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for the GNN backend.\n"
                "  pip install torch_geometric"
            )
        super().__init__()
        self.input_proj = nn.Linear(n_inputs, hidden)
        self.convs = nn.ModuleList(
            [SAGEConv(hidden, hidden) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(n_layers)]
        )
        self.mean_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = torch.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            h = torch.relu(norm(conv(h, edge_index)))
        return self.mean_head(h), self.logvar_head(h)


# ===========================================================================
# GP — Deep Kernel Learning (DKL) with Sparse Variational Approximation
# ===========================================================================

class _DKLFeatureExtractor(nn.Module):
    """MLP that maps raw (scaled) features to a low-dim kernel space."""

    def __init__(self, n_inputs: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128),      nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64),       nn.LayerNorm(64),  nn.ReLU(),
            nn.Linear(64,  out_dim),
        )

    def forward(self, x):
        return self.net(x)


class _SVGPLayer(ApproximateGP):
    """Sparse Variational GP with Cholesky variational distribution."""

    def __init__(self, inducing_points: torch.Tensor):
        vd = CholeskyVariationalDistribution(inducing_points.size(0))
        vs = VariationalStrategy(
            self, inducing_points, vd, learn_inducing_locations=True
        )
        super().__init__(vs)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class GPModel(nn.Module):
    """
    Deep Kernel Learning: MLP feature extractor feeds into a Sparse
    Variational GP, giving principled uncertainty estimates without the
    cubic cost of exact GP inference.

    Training mode  → forward() returns a GP distribution (used with ELBO).
    Eval / infer   → forward() returns (mean, logvar) matching MLP/GNN.

    Requires: pip install gpytorch
    """

    def __init__(
        self,
        n_inputs:    int,
        n_inducing:  int = 512,
        feature_dim: int = 16,
    ):
        if not _GPYTORCH_AVAILABLE:
            raise ImportError(
                "gpytorch is required for the GP backend.\n"
                "  pip install gpytorch"
            )
        super().__init__()
        self.feature_extractor = _DKLFeatureExtractor(n_inputs, feature_dim)
        # Inducing points live in the learned feature space; initialised randomly
        inducing_points = torch.randn(n_inducing, feature_dim)
        self.gp         = _SVGPLayer(inducing_points)
        self.likelihood  = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, x: torch.Tensor, edge_index=None):
        features = self.feature_extractor(x)
        if self.training:
            # Return GP distribution so the caller can compute the ELBO
            return self.gp(features)
        else:
            with gpytorch.settings.fast_pred_var():
                pred = self.likelihood(self.gp(features))
            mean   = pred.mean.unsqueeze(-1)
            logvar = pred.variance.clamp(min=1e-8).log().unsqueeze(-1)
            return mean, logvar


# ===========================================================================
# Factory
# ===========================================================================

def build_model(model_type: str, n_inputs: int, **kwargs) -> nn.Module:
    """
    Instantiate a model by name.

    Parameters
    ----------
    model_type : "mlp" | "gnn" | "gp"
    n_inputs   : number of input features
    **kwargs   : forwarded to the model constructor (e.g. hidden, n_layers,
                 n_inducing, feature_dim, gnn_k)
    """
    if model_type == "mlp":
        return MLPModel(n_inputs)
    elif model_type == "gnn":
        gnn_kwargs = {k: v for k, v in kwargs.items()
                      if k in ("hidden", "n_layers")}
        return GNNModel(n_inputs, **gnn_kwargs)
    elif model_type == "gp":
        gp_kwargs = {k: v for k, v in kwargs.items()
                     if k in ("n_inducing", "feature_dim")}
        return GPModel(n_inputs, **gp_kwargs)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from: mlp, gnn, gp"
        )


# ===========================================================================
# Spatial graph construction (used by GNN train + infer)
# ===========================================================================

def build_knn_graph(l: np.ndarray, b: np.ndarray, k: int = 16) -> torch.Tensor:
    """
    Build a directed k-NN graph over stars using galactic (l, b) coordinates
    and the haversine metric (great-circle distance on the unit sphere).

    Parameters
    ----------
    l, b : galactic longitude / latitude in degrees, shape (N,)
    k    : number of neighbours per star

    Returns
    -------
    edge_index : torch.LongTensor of shape (2, N*k)  — (source, target) pairs
    """
    from sklearn.neighbors import NearestNeighbors

    # haversine expects (latitude, longitude) in radians
    coords_rad = np.radians(np.stack([b, l], axis=1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="haversine", algorithm="ball_tree")
    nn.fit(coords_rad)
    _, indices = nn.kneighbors(coords_rad)

    sources = np.repeat(np.arange(len(l)), k)
    targets = indices[:, 1:].ravel()          # drop self-loop (col 0)
    edge_index = torch.from_numpy(
        np.stack([sources, targets], axis=0)
    ).long()
    return edge_index
