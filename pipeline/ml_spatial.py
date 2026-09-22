"""Spatial checks for the published gold-placer forest.

Random k-fold AUC can look good because a test grab sits next to a training
grab from the same mine circle (Airola 2018). Dead-zone CV drops those
neighbors from training. Distance-to-mine is the serve-time flag.
"""

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

TARGET_QUESTION = (
    "Does this stream-sediment grab look like NURE samples taken near known "
    "gold placers in the study area? This is a gold-placer lookalike screen, "
    "not a monazite or REE detector."
)

NEGATIVE_CLASS_NOTE = (
    "A training label of 0 means the grab was not near a mapped gold mine "
    "(and, when a DEM was present, not on the same valley floor). "
    "It does not mean the ground is barren."
)

CV_OPTIMISM_NOTE = (
    "cv_auc_mean is shuffled 5-fold stratified CV. It can be optimistic "
    "because nearby samples often share the same mine. spatial_cv_auc_mean "
    "is the same fold split after dropping training samples within 0.15° of "
    "any test sample (the label radius). That is the more honest number for "
    "new drainages."
)

FAR_FROM_MINE_DEG = 0.15


def nearest_gold_mrds_deg(lon, lat, gold_xy):
    """Euclidean degrees to the nearest stored gold MRDS point. None if no points."""
    if gold_xy is None or len(gold_xy) == 0:
        return None
    tree = cKDTree(np.asarray(gold_xy, dtype=float))
    dist, _ = tree.query([[float(lon), float(lat)]])
    return float(dist[0])


def spatial_flag(distance_deg, radius_deg=FAR_FROM_MINE_DEG):
    if distance_deg is None:
        return 'unknown_no_location'
    if distance_deg <= radius_deg:
        return 'near_known_gold'
    return 'far_from_known_gold'


def _fit_auc(X, y, tr, te, n_estimators=200):
    if len(tr) < 10 or len(te) < 5:
        return None
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return None
    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=42, class_weight='balanced'
    )
    rf.fit(X[tr], y[tr])
    proba = rf.predict_proba(X[te])
    if proba.shape[1] < 2:
        return None
    return float(roc_auc_score(y[te], proba[:, 1]))


def deadzone_cv_aucs(X, y, coords, n_splits=5, deadzone_deg=0.15, n_estimators=200):
    """Stratified k-fold, then strip training rows within deadzone_deg of test rows."""
    coords = np.asarray(coords, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    n_dropped = []
    for tr, te in skf.split(X, y):
        tree = cKDTree(coords[te])
        dist, _ = tree.query(coords[tr])
        keep = dist > deadzone_deg
        n_dropped.append(int((~keep).sum()))
        fold_auc = _fit_auc(X, y, tr[keep], te, n_estimators=n_estimators)
        if fold_auc is not None:
            aucs.append(fold_auc)
    return aucs, n_dropped


def block_ids(lon, lat, cell_deg=0.4):
    """Integer cell id from lon/lat. Same cell → likely the same cluster of mines."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    ix = np.floor(lon / cell_deg).astype(int)
    iy = np.floor(lat / cell_deg).astype(int)
    return ix * 10_000 + iy


def block_cv_aucs(X, y, lon, lat, n_splits=5, cell_deg=0.4, n_estimators=200):
    """StratifiedGroupKFold on 0.4° cells. Skips folds that have only one class."""
    groups = block_ids(lon, lat, cell_deg=cell_deg)
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    if splits < 2:
        return []
    sgkf = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=42)
    aucs = []
    try:
        iterator = list(sgkf.split(X, y, groups))
    except ValueError:
        return []
    for tr, te in iterator:
        fold_auc = _fit_auc(X, y, tr, te, n_estimators=n_estimators)
        if fold_auc is not None:
            aucs.append(fold_auc)
    return aucs


def leave_one_block_aucs(X, y, lon, lat, cell_deg=0.4, n_estimators=200):
    """Train on every cell but one; return {block_id: AUC or None}.

    This is the per-cell version of block CV. A cell with only one class, or
    too few grabs, gets None — that is not 0.50, it is 'cannot score'.
    """
    groups = block_ids(lon, lat, cell_deg=cell_deg)
    out = {}
    for gid in np.unique(groups):
        te = np.where(groups == gid)[0]
        tr = np.where(groups != gid)[0]
        out[int(gid)] = _fit_auc(X, y, tr, te, n_estimators=n_estimators)
    return out


def summarize_aucs(aucs):
    if not aucs:
        return None, None, []
    arr = np.asarray(aucs, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), std, [float(round(a, 4)) for a in arr]
