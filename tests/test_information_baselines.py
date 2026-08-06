from __future__ import annotations

import numpy as np

from scripts.eval_exact_composition import infer_dim
from scripts.eval_label_budget_baselines import apply_ridge, fit_ridge, metrics


def test_infer_dim_full_choi_features() -> None:
    assert infer_dim(32) == 2
    assert infer_dim(512) == 4


def test_ridge_recovers_simple_affine_relation() -> None:
    x = np.arange(20, dtype=float)[:, None]
    y = 0.2 + 0.03 * x[:, 0]
    pred = apply_ridge(x, fit_ridge(x, y, ridge=1e-10))
    np.testing.assert_allclose(pred, y, atol=1e-10)


def test_metrics_clips_physical_range() -> None:
    got = metrics(np.array([-1.0, 2.0]), np.array([0.0, 1.0]))
    assert got["mae_F_e"] == 0.0
