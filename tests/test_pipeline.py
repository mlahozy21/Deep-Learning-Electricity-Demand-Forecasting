"""Tests for the temporal split — the leakage-critical function.

The split must train strictly on years before ``val_year`` and validate on
``val_year`` only, with no overlap. A bug here would leak validation-year (or
future) data into training and invalidate every reported metric.
"""

import numpy as np
import pandas as pd

from edf import pipeline


def _toy(years=(2019, 2020, 2021), per_year=24):
    idx = pd.DatetimeIndex([])
    for yr in years:
        idx = idx.append(
            pd.date_range(f"{yr}-06-01", periods=per_year, freq="h", tz="UTC")
        )
    X = pd.DataFrame({"f": np.arange(len(idx), dtype=float)}, index=idx)
    y = pd.DataFrame({"t0": np.arange(len(idx), dtype=float)}, index=idx)
    return X, y


def test_temporal_split_val_is_only_val_year():
    X, y = _toy()
    Xtr, ytr, Xva, yva = pipeline.temporal_split(X, y, val_year=2021)
    val_years = set(Xva.index.tz_convert("Europe/Paris").year.unique())
    assert val_years == {2021}
    yval_years = set(yva.index.tz_convert("Europe/Paris").year.unique())
    assert yval_years == {2021}


def test_temporal_split_train_strictly_before_val_year():
    X, y = _toy()
    Xtr, ytr, Xva, yva = pipeline.temporal_split(X, y, val_year=2021)
    train_years = set(Xtr.index.tz_convert("Europe/Paris").year.unique())
    assert all(yr < 2021 for yr in train_years)
    assert train_years == {2019, 2020}


def test_temporal_split_no_index_overlap():
    X, y = _toy()
    Xtr, ytr, Xva, yva = pipeline.temporal_split(X, y, val_year=2021)
    assert Xtr.index.intersection(Xva.index).empty
    # Train and val together reconstruct the full timeline with no loss/dup.
    assert len(Xtr) + len(Xva) == len(X)
    assert Xtr.index.union(Xva.index).equals(X.index.sort_values())


def test_temporal_split_X_and_y_aligned():
    X, y = _toy()
    Xtr, ytr, Xva, yva = pipeline.temporal_split(X, y, val_year=2020)
    assert Xtr.index.equals(ytr.index)
    assert Xva.index.equals(yva.index)
