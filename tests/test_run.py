import numpy as np
import pandas as pd
import pytest

from protstab.run import ad_table, conformal_eval, metrics


class _ConstModel:
    def __init__(self, v):
        self.v = v

    def predict(self, X):
        return np.full(len(X), self.v)


def test_conformal_coverage_exact_when_residuals_uniform():
    # cal residuals all equal to 1 -> q=1; test residuals of 0.5 covered,
    # 1.5 not
    m = _ConstModel(0.0)
    X_cal = np.zeros((10, 2))
    y_cal = np.ones(10)  # |0-1|=1 residual each
    X_te = np.zeros((4, 2))
    y_te = np.array([0.5, -0.5, 1.5, -1.5])
    q, yhat, cov = conformal_eval(m, X_cal, y_cal, X_te, y_te, alpha=0.10)
    assert q == pytest.approx(1.0)
    assert cov.tolist() == [True, True, False, False]


def test_metrics_exact():
    m = metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert m["rmse"] == 0.0 and m["mae"] == 0.0 and m["spearman"] == 1.0


def test_ad_table_bins_partition():
    dist = np.arange(12, dtype=float)
    cov = np.ones(12, dtype=bool)
    cov[-3:] = False  # far points uncovered
    rows = ad_table(dist, cov, np.ones(12), bins=4)
    assert sum(r["n"] for r in rows) == 12
    assert rows[-1]["coverage"] < rows[0]["coverage"]


def test_ad_table_empty_bin_skipped():
    dist = np.zeros(5)
    rows = ad_table(dist, np.ones(5, dtype=bool), np.ones(5), bins=4)
    assert all(r["n"] > 0 for r in rows)
