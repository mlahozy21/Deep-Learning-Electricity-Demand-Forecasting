"""Evaluation metrics, including the challenge score (NaN-aware).

Some target series (the métropoles) only start partway through the training
period, so the ground truth contains missing values. All metrics ignore the
positions where ``y_true`` is NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_array(a):
    return a.to_numpy() if isinstance(a, (pd.DataFrame, pd.Series)) else np.asarray(a)


def per_column_rmse(y_true, y_pred) -> np.ndarray:
    """RMSE per target column, ignoring NaN positions in ``y_true``.

    A column with **zero** valid (non-NaN) points has an undefined RMSE and is
    returned as NaN, rather than a misleading 0. The challenge-score reduction
    drops these columns from the sum so a series that is entirely missing in the
    validation year neither inflates nor deflates the total.
    """
    yt = _to_array(y_true).astype(float)
    yp = _to_array(y_pred).astype(float)
    mask = ~np.isnan(yt)
    se = np.where(mask, (yt - yp) ** 2, 0.0)
    counts = mask.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse = np.sqrt(se.sum(axis=0) / counts)
    # counts == 0 -> 0/0 -> NaN (kept NaN, not zeroed). Columns with valid
    # points keep their real RMSE.
    rmse = np.where(counts > 0, rmse, np.nan)
    return rmse


def challenge_score(y_true, y_pred) -> float:
    """Codabench metric: sum of the per-column RMSEs across the series.

    Columns with no valid ground truth in the evaluation window (RMSE is NaN)
    are *excluded* from the sum. Previously such a series silently scored 0,
    which deflated the total. Raises if no column has any valid points.
    """
    rmse = per_column_rmse(y_true, y_pred)
    valid = ~np.isnan(rmse)
    if not valid.any():
        raise ValueError(
            "challenge_score: no target column has any valid (non-NaN) "
            "ground-truth points; cannot compute a score."
        )
    return float(rmse[valid].sum())


def mae(y_true, y_pred) -> float:
    yt = _to_array(y_true).astype(float)
    yp = _to_array(y_pred).astype(float)
    mask = ~np.isnan(yt)
    return float(np.abs(np.where(mask, yt - yp, 0.0)).sum() / np.maximum(mask.sum(), 1))
