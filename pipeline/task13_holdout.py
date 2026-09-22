"""In-belt hold-out: train west of a lon cut, test east. Do not touch the published joblib.

This is Wave 2 in PROJECT_ROADMAP.md. It answers: does the chemistry story
survive on the other side of the Kettle? A sag is the result, not a reason
to refit models/task9_rf_placer_gold.joblib.
"""

import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from pipeline.ml_preprocess import FEATURES, _log_impute
from pipeline.ml_spatial import FAR_FROM_MINE_DEG
from pipeline.task11_field_campaign import campaign_gold_xy
from pipeline.task12_second_belt import score_frozen_transfer
from pipeline.utils import bbox, ensure_outputs, load_nure, out


def lon_cut(cfg):
    lon_min, lon_max, _lat_min, _lat_max = bbox(cfg)
    explicit = (cfg.get('holdout') or {}).get('lon_cut')
    if explicit is not None:
        return float(explicit)
    return 0.5 * (lon_min + lon_max)


def _proximity_labels(df, gold_xy, radius_deg=FAR_FROM_MINE_DEG):
    from scipy.spatial import cKDTree
    coords = np.column_stack([df['lon'].values, df['lat'].values])
    tree = cKDTree(np.asarray(gold_xy, dtype=float))
    dist, _ = tree.query(coords)
    return (dist <= radius_deg).astype(int), dist


def run(cfg):
    ensure_outputs(cfg['outputs_dir'])
    lon_min, lon_max, lat_min, lat_max = bbox(cfg)
    cut = lon_cut(cfg)
    df = load_nure(cfg).dropna(subset=['lat', 'lon'])
    df = df[
        (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
        & (df['lat'] >= lat_min) & (df['lat'] <= lat_max)
    ].copy()
    gold_xy = campaign_gold_xy(cfg)
    if gold_xy is None:
        raise RuntimeError('Need gold MRDS to label the hold-out')
    y, dist = _proximity_labels(df, gold_xy)
    df['label'] = y
    west = df['lon'] < cut
    east = ~west
    n_w, n_e = int(west.sum()), int(east.sum())
    pos_w, pos_e = int(y[west].sum()), int(y[east].sum())

    frozen, frozen_auc = None, None
    try:
        from pipeline.ml_artifacts import load_published_artifacts
        estimator, metadata = load_published_artifacts()
        scored, frozen_auc = score_frozen_transfer(df.loc[east], gold_xy, estimator, metadata)
        frozen = {
            'n_test': int(len(scored)),
            'n_positive': int(scored['label'].sum()),
            'auc': None if frozen_auc is None else round(float(frozen_auc), 4),
            'mean_p': round(float(scored['p_anomalous'].mean()), 4),
        }
    except Exception as exc:
        print(f'  Frozen score on east skipped: {exc}')

    log_X, _ = _log_impute(df, FEATURES)
    X = log_X.values
    refit_auc = None
    if n_w >= 40 and n_e >= 20 and len(np.unique(y[west])) == 2 and len(np.unique(y[east])) == 2:
        rf = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight='balanced',
        )
        rf.fit(X[west.values], y[west.values])
        proba = rf.predict_proba(X[east.values])[:, 1]
        refit_auc = float(roc_auc_score(y[east.values], proba))
        print(f'  Refit-west → test-east AUC: {refit_auc:.3f}  (not the published joblib)')
    else:
        print('  Hold-out refit skipped — one side is too small or one-class')

    summary = {
        'belt': (cfg.get('study_area') or {}).get('name', 'NE Washington'),
        'short': 'holdout_ew',
        'lon_cut': round(cut, 4),
        'n_west': n_w,
        'n_east': n_e,
        'n_positive_west': pos_w,
        'n_positive_east': pos_e,
        'refit_west_test_east_auc': None if refit_auc is None else round(refit_auc, 4),
        'frozen_forest_on_east': frozen,
        'retrained_published': False,
        'note': (
            f'Train west of {cut:.2f}°, test east. The published joblib was not rewritten. '
            'This is an in-belt hold-out, not a new model.'
        ),
    }
    path = out(cfg, 'text', 'task13_holdout_east_west.txt')
    lines = [
        'IN-BELT HOLD-OUT — TRAIN WEST, TEST EAST',
        summary['note'],
        '=' * 68,
        '',
        f'Lon cut:          {cut:.3f}',
        f'West grabs / pos: {n_w} / {pos_w}',
        f'East grabs / pos: {n_e} / {pos_e}',
        f'Refit west→east AUC: {summary["refit_west_test_east_auc"]}',
        f'Frozen forest on east AUC: {None if not frozen else frozen["auc"]}',
        '',
        'Do not replace models/task9_rf_placer_gold.joblib with this refit.',
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return summary


if __name__ == '__main__':
    import sys
    import yaml
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/ne_washington/config.yaml'
    with open(cfg_path) as f:
        run(yaml.safe_load(f))
