"""Score a chemistry request with the persisted Task 9 forest."""

import numpy as np
import pandas as pd

from pipeline.ml_preprocess import (
    FEATURES,
    FE_PPM_MIN_PLAUSIBLE,
    FE_WT_PCT_MAX,
    _log_impute,
    apply_nure_mdl,
)
from pipeline.ml_spatial import FAR_FROM_MINE_DEG, nearest_gold_mrds_deg, spatial_flag


class UnitError(ValueError):
    """Raised when Fe is still the wrong unit after MDL handling."""


def request_to_ppm_row(req):
    """Apply half-MDL and convert Fe to ppm. Missing → NaN (imputed later)."""
    row = {}
    for feat in FEATURES:
        converted = apply_nure_mdl(getattr(req, feat))
        row[feat] = np.nan if converted is None else float(converted)

    fe = row['Fe']
    if np.isfinite(fe):
        if req.fe_unit == 'wt_pct':
            if fe >= FE_WT_PCT_MAX:
                raise UnitError(
                    f"Fe={fe} wt% is not a plausible stream-sediment value"
                )
            row['Fe'] = fe * 10000.0
        elif 0 < fe < FE_PPM_MIN_PLAUSIBLE:
            raise UnitError(
                f"Fe={fe} ppm is below {FE_PPM_MIN_PLAUSIBLE:.0f}; "
                "this is almost certainly unconverted wt%"
            )
    return row


def predict_with_spread(estimator, metadata, req, gold_xy=None):
    """Return label, probability, tree-vote spread, and distance-to-gold flag."""
    row = request_to_ppm_row(req)
    order = list(metadata['feature_order'])
    df = pd.DataFrame([{f: row[f] for f in order}])
    log_X, _ = _log_impute(df, order, medians=metadata['log_medians'])
    X = log_X.values

    probability = float(estimator.predict_proba(X)[0, 1])
    tree_p = np.array(
        [float(est.predict_proba(X)[0, 1]) for est in estimator.estimators_]
    )
    spread = float(tree_p.std(ddof=0))
    label = int(probability >= 0.5)

    dist = None
    if req.lon is not None and req.lat is not None:
        dist = nearest_gold_mrds_deg(req.lon, req.lat, gold_xy)
    return {
        'label': label,
        'probability': probability,
        'tree_vote_spread': spread,
        'n_trees': int(len(estimator.estimators_)),
        'nearest_gold_mrds_deg': dist,
        'spatial_flag': spatial_flag(dist, metadata.get('mrds_proximity_deg', FAR_FROM_MINE_DEG)),
    }
