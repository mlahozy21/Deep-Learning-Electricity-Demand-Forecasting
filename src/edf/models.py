"""Forecasting models with a common (fit / predict) interface.

Three models of increasing sophistication:
  - SeasonalNaive : climatology baseline (mean by month x weekday x time-of-day).
  - TorchMLP      : the project's neural network, rewritten cleanly.
  - GBMModel      : gradient-boosted trees (one per target series).

All operate on the 25 target series simultaneously and tolerate missing values
in the targets (the métropoles only start partway through the training period).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _check_feature_columns(feature_names_in, X):
    """Assert that ``X``'s columns match the columns seen at fit time.

    Predictions are made on a positional feature matrix, so train and test must
    share the *same feature columns in the same order*. A silent positional
    mismatch (different column order, a dropped/extra feature) would corrupt
    every prediction without error, so we check by name when names are available.
    """
    if feature_names_in is None or not isinstance(X, pd.DataFrame):
        return
    got = list(X.columns)
    if got != feature_names_in:
        missing = [c for c in feature_names_in if c not in got]
        extra = [c for c in got if c not in feature_names_in]
        raise ValueError(
            "Feature columns at predict time do not match those seen at fit "
            "time (by name and order).\n"
            f"  expected ({len(feature_names_in)}): {feature_names_in}\n"
            f"  got      ({len(got)}): {got}\n"
            f"  missing: {missing}\n  unexpected: {extra}\n"
            f"  order mismatch only: {sorted(got) == sorted(feature_names_in)}"
        )


# --------------------------------------------------------------------------- #
# Baseline: seasonal climatology
# --------------------------------------------------------------------------- #
class SeasonalNaive:
    """Predict the historical mean for each (month, weekday, half-hour) cell.

    A strong, weather-free baseline that captures the dominant daily, weekly
    and yearly seasonality of electricity demand.
    """

    def __init__(self) -> None:
        self.table_ = None
        self.global_mean_ = None
        self.columns_ = None

    @staticmethod
    def _keys(index: pd.DatetimeIndex):
        local = index.tz_convert("Europe/Paris")
        tod = local.hour * 2 + (local.minute >= 30).astype(int)  # 0..47
        return local.month.to_numpy(), local.dayofweek.to_numpy(), np.asarray(tod)

    def fit(self, y: pd.DataFrame) -> "SeasonalNaive":
        self.columns_ = list(y.columns)
        self.global_mean_ = y.mean()
        m, d, t = self._keys(y.index)
        grp = y.groupby([m, d, t]).mean()
        grp.index.set_names(["month", "dow", "tod"], inplace=True)
        self.table_ = grp
        return self

    def predict(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        m, d, t = self._keys(index)
        keys = pd.MultiIndex.from_arrays([m, d, t], names=["month", "dow", "tod"])
        pred = self.table_.reindex(keys).fillna(self.global_mean_)
        pred.index = index
        return pred[self.columns_]


# --------------------------------------------------------------------------- #
# Loss: masked sum of per-column RMSE (the challenge metric)
# --------------------------------------------------------------------------- #
def masked_sum_rmse(pred, target, mask):
    """Sum over targets of the per-column RMSE, ignoring masked-out entries.

    This is the metric the challenge optimises (and the loss used in the
    original project), generalised with a mask so series with missing values
    (the métropoles) contribute only where ground truth exists.

    A column with **zero** valid entries (e.g. a series entirely missing in a
    mini-batch) has an undefined RMSE; it is excluded from the sum rather than
    contributing a spurious 0 (which would deflate the loss / score). The
    ``counts`` clamp only guards the division for already-excluded columns.
    """
    import torch

    se = (pred - target) ** 2 * mask
    col_counts = mask.sum(dim=0)
    counts = torch.clamp(col_counts, min=1.0)
    rmse_per_col = torch.sqrt(se.sum(dim=0) / counts)
    # Drop columns with no valid points from the sum (their RMSE is undefined).
    rmse_per_col = torch.where(
        col_counts > 0, rmse_per_col, torch.zeros_like(rmse_per_col)
    )
    return rmse_per_col.sum()


# --------------------------------------------------------------------------- #
# Neural network (PyTorch) — the project's original model, rewritten
# --------------------------------------------------------------------------- #
class TorchMLP:
    """Two-hidden-layer MLP with batch-norm, trained with mini-batches.

    Faithful to the original project: the training loss is the masked
    sum-of-per-column-RMSE (= the challenge metric), and the targets are kept
    on their natural scale. Improvements: standardised inputs, mini-batches,
    early stopping on the validation metric, dropout, and device auto-select.
    """

    def __init__(self, hidden=(128, 64), dropout=0.1, lr=1e-2, max_epochs=200,
                 batch_size=512, patience=15, seed=42):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed

    def _build(self, n_in, n_out):
        import torch.nn as nn

        layers, prev = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        net = nn.Sequential(*layers)
        for p in net.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return net

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from .utils import set_seed

        set_seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.columns_ = list(y_train.columns)
        self.feature_names_in_ = (
            list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
        )

        self.x_scaler = StandardScaler().fit(np.asarray(X_train, dtype=np.float32))
        # NaN-aware target stats: the network predicts in standardised output
        # space (well-conditioned) while the loss is computed on the raw scale.
        ytr = np.asarray(y_train, dtype=np.float32)
        self.y_mean = np.nanmean(ytr, axis=0).astype(np.float32)
        self.y_std = np.nanstd(ytr, axis=0).astype(np.float32)
        self.y_std[self.y_std == 0] = 1.0

        def to_xy(X, y):
            Xt = torch.tensor(self.x_scaler.transform(np.asarray(X, dtype=np.float32)),
                              dtype=torch.float32)
            ya = np.asarray(y, dtype=np.float32)
            mask = (~np.isnan(ya)).astype(np.float32)
            yt = torch.tensor(np.nan_to_num(ya, nan=0.0), dtype=torch.float32)
            mt = torch.tensor(mask, dtype=torch.float32)
            return Xt, yt, mt

        Xt, yt, mt = to_xy(X_train, y_train)
        self.net = self._build(Xt.shape[1], yt.shape[1]).to(self.device)
        ymean_t = torch.tensor(self.y_mean, device=self.device)
        ystd_t = torch.tensor(self.y_std, device=self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loader = DataLoader(TensorDataset(Xt, yt, mt), batch_size=self.batch_size,
                            shuffle=True)

        has_val = X_val is not None and y_val is not None
        if has_val:
            Xv, yv, mv = (t.to(self.device) for t in to_xy(X_val, y_val))

        best_val, best_state, bad = float("inf"), None, 0
        for _ in range(self.max_epochs):
            self.net.train()
            for xb, yb, mb in loader:
                xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                opt.zero_grad(set_to_none=True)
                pred = self.net(xb) * ystd_t + ymean_t
                masked_sum_rmse(pred, yb, mb).backward()
                opt.step()
            if has_val:
                self.net.eval()
                with torch.no_grad():
                    v = masked_sum_rmse(self.net(Xv) * ystd_t + ymean_t, yv, mv).item()
                if v < best_val - 1e-4:
                    best_val, best_state, bad = v, {k: t.clone() for k, t in
                                                    self.net.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, X):
        import torch

        _check_feature_columns(getattr(self, "feature_names_in_", None), X)
        self.net.eval()
        Xs = torch.tensor(self.x_scaler.transform(np.asarray(X, dtype=np.float32)),
                          dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.net(Xs).cpu().numpy() * self.y_std + self.y_mean
        idx = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(out, columns=self.columns_, index=idx)


# --------------------------------------------------------------------------- #
# Gradient-boosted trees (one regressor per target)
# --------------------------------------------------------------------------- #
class GBMModel:
    """One HistGradientBoostingRegressor per target (sklearn, no extra deps)."""

    def __init__(self, max_iter=300, learning_rate=0.05, max_depth=None,
                 l2_regularization=1.0, seed=42, n_jobs=-1):
        self.kwargs = dict(max_iter=max_iter, learning_rate=learning_rate,
                           max_depth=max_depth, l2_regularization=l2_regularization,
                           early_stopping=True, validation_fraction=0.1,
                           random_state=seed)
        self.n_jobs = n_jobs
        self.models_ = {}

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from joblib import Parallel, delayed
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.columns_ = list(y_train.columns)
        self.feature_names_in_ = (
            list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
        )
        Xa = np.asarray(X_train, dtype=np.float32)

        def _fit_one(col):
            yc = np.asarray(y_train[col], dtype=np.float32)
            valid = ~np.isnan(yc)  # métropoles start late: train on available rows
            m = HistGradientBoostingRegressor(**self.kwargs)
            m.fit(Xa[valid], yc[valid])
            return col, m

        self.models_ = dict(
            Parallel(n_jobs=self.n_jobs)(delayed(_fit_one)(c) for c in self.columns_)
        )
        return self

    def predict(self, X):
        _check_feature_columns(getattr(self, "feature_names_in_", None), X)
        Xa = np.asarray(X, dtype=np.float32)
        out = {col: self.models_[col].predict(Xa) for col in self.columns_}
        idx = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(out, columns=self.columns_, index=idx)
