"""Spatial CV and distance-to-mine helpers."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.ml_spatial import (
    deadzone_cv_aucs,
    nearest_gold_mrds_deg,
    spatial_flag,
    summarize_aucs,
)


def test_spatial_flag_rules():
    assert spatial_flag(None) == 'unknown_no_location'
    assert spatial_flag(0.01) == 'near_known_gold'
    assert spatial_flag(0.15) == 'near_known_gold'
    assert spatial_flag(0.20) == 'far_from_known_gold'


def test_nearest_gold_distance():
    gold = np.array([[-118.0, 48.0], [-117.0, 49.0]])
    assert nearest_gold_mrds_deg(-118.0, 48.0, gold) == 0.0
    d = nearest_gold_mrds_deg(-118.1, 48.0, gold)
    assert abs(d - 0.1) < 1e-9


def test_deadzone_cv_runs_on_clustered_labels():
    """Three mine circles so holding out neighbors of one still leaves other yes-samples."""
    rng = np.random.default_rng(1)
    coords = rng.uniform(low=[-119.5, 47.8], high=[-117.5, 49.0], size=(120, 2))
    mines = np.array([[-119.2, 48.8], [-117.8, 48.0], [-118.5, 48.4]])
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(mines).query(coords)
    y = (dist < 0.25).astype(int)
    X = np.column_stack([coords, rng.normal(size=120)])
    aucs, dropped = deadzone_cv_aucs(X, y, coords, n_splits=5, deadzone_deg=0.15, n_estimators=20)
    mean, _, folds = summarize_aucs(aucs)
    assert mean is not None
    assert 0.0 <= mean <= 1.0
    assert len(folds) >= 1
    assert sum(dropped) > 0
