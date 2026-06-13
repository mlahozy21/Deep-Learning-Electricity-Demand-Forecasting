"""Unit tests for the evaluation metrics (no dataset required)."""

import numpy as np
import pytest

from edf import metrics


def test_per_column_rmse_simple():
    y_true = np.array([[0.0, 10.0], [0.0, 10.0]])
    y_pred = np.array([[1.0, 13.0], [-1.0, 7.0]])
    rmse = metrics.per_column_rmse(y_true, y_pred)
    np.testing.assert_allclose(rmse, [1.0, 3.0])


def test_challenge_score_is_sum_of_rmse():
    y_true = np.zeros((4, 3))
    y_pred = np.ones((4, 3))
    assert metrics.challenge_score(y_true, y_pred) == 3.0  # 1 + 1 + 1


def test_metrics_ignore_nan_in_truth():
    y_true = np.array([[0.0, np.nan], [0.0, 10.0]])
    y_pred = np.array([[2.0, 5.0], [-2.0, 10.0]])
    # column 0 RMSE = 2; column 1 ignores the NaN row -> error 0
    np.testing.assert_allclose(metrics.per_column_rmse(y_true, y_pred), [2.0, 0.0])


def test_all_nan_series_is_excluded_not_scored_zero():
    # Column 1 is entirely NaN in the truth -> its RMSE is undefined (NaN),
    # and it must be *excluded* from the challenge score, not counted as 0.
    y_true = np.array([[0.0, np.nan], [4.0, np.nan]])
    y_pred = np.array([[3.0, 100.0], [0.0, -100.0]])  # col 0 RMSE = sqrt((9+16)/2)
    rmse = metrics.per_column_rmse(y_true, y_pred)
    assert np.isfinite(rmse[0])
    assert np.isnan(rmse[1])  # not 0.0
    # Score is just column 0's RMSE; the all-NaN column does not deflate it.
    np.testing.assert_allclose(
        metrics.challenge_score(y_true, y_pred), np.sqrt((9 + 16) / 2)
    )


def test_challenge_score_raises_when_no_valid_points():
    y_true = np.full((3, 2), np.nan)
    y_pred = np.zeros((3, 2))
    with pytest.raises(ValueError):
        metrics.challenge_score(y_true, y_pred)


def test_masked_sum_rmse_excludes_all_masked_column():
    torch = pytest.importorskip("torch")
    pred = torch.zeros((2, 2))
    target = torch.tensor([[1.0, 5.0], [1.0, 7.0]])
    # Column 1 fully masked out -> excluded; column 0 RMSE = 1.0.
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    from edf.models import masked_sum_rmse

    val = masked_sum_rmse(pred, target, mask).item()
    np.testing.assert_allclose(val, 1.0)
