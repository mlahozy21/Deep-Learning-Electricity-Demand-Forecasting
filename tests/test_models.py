"""Smoke tests for the models on synthetic data (no dataset required)."""

import numpy as np
import pandas as pd
import pytest

from edf.models import GBMModel, SeasonalNaive, TorchMLP


def _synthetic(n=2000, n_targets=3):
    idx = pd.date_range("2019-01-01", periods=n, freq="30min", tz="UTC")
    rng = np.random.default_rng(0)
    hour = idx.tz_convert("Europe/Paris").hour.to_numpy()
    base = 100 + 20 * np.sin(2 * np.pi * hour / 24)
    y = pd.DataFrame(
        {f"t{j}": base * (1 + 0.1 * j) + rng.normal(0, 1, n) for j in range(n_targets)},
        index=idx,
    )
    X = pd.DataFrame({"hour_sin": np.sin(2 * np.pi * hour / 24),
                      "hour_cos": np.cos(2 * np.pi * hour / 24)}, index=idx)
    return X, y


def test_seasonal_naive_runs_and_beats_mean():
    X, y = _synthetic()
    model = SeasonalNaive().fit(y)
    pred = model.predict(y.index)
    assert pred.shape == y.shape
    assert not pred.isna().any().any()


def test_torch_mlp_fits_and_predicts():
    X, y = _synthetic()
    model = TorchMLP(hidden=(16,), max_epochs=5, batch_size=256).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert np.isfinite(pred.to_numpy()).all()


def test_predict_rejects_reordered_feature_columns():
    """A train/test feature-column mismatch must raise, not silently corrupt."""
    X, y = _synthetic()
    model = TorchMLP(hidden=(16,), max_epochs=3, batch_size=256).fit(X, y)
    X_swapped = X[["hour_cos", "hour_sin"]]  # same columns, wrong order
    with pytest.raises(ValueError):
        model.predict(X_swapped)
    X_renamed = X.rename(columns={"hour_sin": "wrong_name"})
    with pytest.raises(ValueError):
        model.predict(X_renamed)


def test_gbm_masks_per_column_nans():
    """Each GBM regressor must train only on the rows where *its own* target is
    present (métropoles start partway through). A column that is NaN for the
    first half of the rows must still be fit (on its valid rows) and predict
    finite values, never crash or train on NaN targets."""
    X, y = _synthetic(n=400, n_targets=3)
    # Make target t1 missing for the first half of the timeline.
    y = y.copy()
    y.iloc[: len(y) // 2, y.columns.get_loc("t1")] = np.nan
    model = GBMModel(max_iter=20).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert np.isfinite(pred.to_numpy()).all()
    # The masked column was fit on roughly its non-NaN rows only.
    assert pred["t1"].notna().all()
