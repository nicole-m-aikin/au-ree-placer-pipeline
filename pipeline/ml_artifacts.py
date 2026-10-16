"""Persist / locate the published Task 9 Random Forest.

The API loads these files. Task 9 writes them after rf_final.fit so the
log-medians are the ones training actually used.

Only NE Washington may overwrite the doorbell joblib. Other belts write
task9_rf_placer_gold.{short}.joblib next to it — a different model.
"""

import json
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / 'models'
MODEL_FILENAME = 'task9_rf_placer_gold.joblib'
META_FILENAME = 'task9_rf_placer_gold.meta.json'
GOLD_MRDS_FILENAME = 'task9_rf_placer_gold.gold_mrds.json'
TRANSFER_FILENAME = 'task9_rf_placer_gold.transfer.json'
TRANSFERS_FILENAME = 'task9_rf_placer_gold.transfers.json'
IDAHO_TRANSFER_KEYS = ('id_batholith', 'Idaho Batholith')
# Only these shorts may overwrite the doorbell files.
PUBLISHED_SHORTS = frozenset({'ne_wa', 'ne_washington', ''})

HEADLINE_NOTE = (
    'Quote the spatial, validated numbers. On the validated NE WA NURE '
    '(287 grabs, CIM audit) the forest scores 0.58 0.4-degree block / 0.63 '
    'dead-zone / 0.66 shuffled. On the unvalidated extract the dead-zone CV '
    'is 0.70. The frozen forest scores about 0.5 on new belts (Montana 0.34).'
)

LEGACY_CV_NOTE = (
    'cv_auc_mean (0.891) is history: shuffled 5-fold CV on the unvalidated '
    'extract. It does not survive the CIM audit (548 of 1,045 grabs on one '
    'placeholder point; submitter alone predicts the label at AUC 0.57). '
    'Kept for backward compatibility, not as a performance claim.'
)


def validated_cv_summary(cv_csv, confound_csv=None):
    """Validated-scenario AUCs from task9_validated_cv.csv (validation_impact).

    Returns {} when the table or its 'validated' row is missing, so a run
    without the validated impact table still persists.
    """
    import os

    import pandas as pd

    if not cv_csv or not os.path.exists(cv_csv):
        return {}
    cv = pd.read_csv(cv_csv)
    row = cv.loc[cv['scenario'] == 'validated']
    if row.empty:
        return {}
    r = row.iloc[0]

    def _num(key):
        v = r.get(key)
        return None if v is None or pd.isna(v) else round(float(v), 4)

    summary = {
        'validated_block_cv_auc_mean': _num('block_auc_mean'),
        'validated_deadzone_cv_auc_mean': _num('deadzone_auc_mean'),
        'validated_shuffled_cv_auc_mean': _num('shuffled_auc_mean'),
        'validated_n_samples': int(r['n_samples']),
        'validated_n_positive': int(r['n_positive']),
        'validated_features': [f for f in str(r.get('features') or '').split(',') if f],
        'validated_source': 'ne_wa_ree/outputs/tables/task9_validated_cv.csv (scenario=validated)',
    }
    if confound_csv and os.path.exists(confound_csv):
        conf = pd.read_csv(confound_csv)
        hit = conf.loc[conf['group'] == 'submitter_only_predictor', 'shuffled_auc_mean']
        if len(hit) and pd.notna(hit.iloc[0]):
            summary['submitter_only_auc'] = round(float(hit.iloc[0]), 4)
    return summary


def published_model_path(model_dir=None):
    d = Path(model_dir) if model_dir else MODEL_DIR
    return d / MODEL_FILENAME


def published_meta_path(model_dir=None):
    d = Path(model_dir) if model_dir else MODEL_DIR
    return d / META_FILENAME


def published_gold_mrds_path(model_dir=None):
    d = Path(model_dir) if model_dir else MODEL_DIR
    return d / GOLD_MRDS_FILENAME


def study_area_short(cfg):
    """Config study_area.short, trimmed. Empty string if unset."""
    return str((cfg.get('study_area') or {}).get('short') or '').strip()


def is_published_training_belt(cfg):
    """True only for NE Washington — the doorbell training box."""
    short = study_area_short(cfg).lower()
    return short in PUBLISHED_SHORTS


def belt_model_stem(short):
    """Filename stem for a non-doorbell forest."""
    slug = str(short or 'belt').strip().replace('-', '_')
    return f'task9_rf_placer_gold.{slug}'


def belt_model_path(short, model_dir=None):
    d = Path(model_dir) if model_dir else MODEL_DIR
    return d / f'{belt_model_stem(short)}.joblib'


def belt_meta_path(short, model_dir=None):
    d = Path(model_dir) if model_dir else MODEL_DIR
    return d / f'{belt_model_stem(short)}.meta.json'


def belt_gold_mrds_path(short, model_dir=None):
    d = Path(model_dir) if model_dir else MODEL_DIR
    return d / f'{belt_model_stem(short)}.gold_mrds.json'


def persist_published_model(estimator, metadata, model_dir=None):
    """Write the doorbell joblib + metadata. NE WA only — callers must gate."""
    dest = Path(model_dir) if model_dir else MODEL_DIR
    dest.mkdir(parents=True, exist_ok=True)
    model_path = dest / MODEL_FILENAME
    meta_path = dest / META_FILENAME
    joblib.dump(estimator, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    return model_path, meta_path


def persist_belt_model(estimator, metadata, short, model_dir=None):
    """Write a belt-local forest. Never touches the doorbell filenames."""
    dest = Path(model_dir) if model_dir else MODEL_DIR
    dest.mkdir(parents=True, exist_ok=True)
    model_path = belt_model_path(short, dest)
    meta_path = belt_meta_path(short, dest)
    joblib.dump(estimator, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    return model_path, meta_path


def persist_gold_mrds(coords, model_dir=None, short=None):
    """Write gold MRDS lon/lat. Belt short → sidecar next to that belt's forest."""
    dest = Path(model_dir) if model_dir else MODEL_DIR
    dest.mkdir(parents=True, exist_ok=True)
    if short and str(short).strip().lower() not in PUBLISHED_SHORTS:
        path = belt_gold_mrds_path(short, dest)
    else:
        path = dest / GOLD_MRDS_FILENAME
    arr = np.asarray(coords, dtype=float).reshape(-1, 2)
    path.write_text(json.dumps({
        'crs': 'EPSG:4326',
        'columns': ['lon', 'lat'],
        'coordinates': [[float(x), float(y)] for x, y in arr],
    }) + '\n')
    return path


def load_gold_mrds(model_dir=None):
    """Return (N, 2) lon/lat array, or None if the sidecar is missing."""
    path = published_gold_mrds_path(model_dir)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    coords = payload.get('coordinates') or []
    if not coords:
        return None
    return np.asarray(coords, dtype=float)


def _transfer_key(summary):
    return summary.get('short') or summary.get('belt') or 'unknown'


def persist_transfer(summary, model_dir=None):
    """Upsert one belt. Idaho stays the legacy single-file sidecar."""
    dest = Path(model_dir) if model_dir else MODEL_DIR
    dest.mkdir(parents=True, exist_ok=True)
    catalog = load_transfers(dest)
    catalog[_transfer_key(summary)] = summary
    catalog_path = dest / TRANSFERS_FILENAME
    catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + '\n')
    legacy = None
    for key in IDAHO_TRANSFER_KEYS:
        if key in catalog:
            legacy = catalog[key]
            break
    if legacy is None:
        legacy = summary
    (dest / TRANSFER_FILENAME).write_text(
        json.dumps(legacy, indent=2, sort_keys=True) + '\n'
    )
    return catalog_path


def load_transfers(model_dir=None):
    """All scored belts, keyed by short name or belt title."""
    dest = Path(model_dir) if model_dir else MODEL_DIR
    path = dest / TRANSFERS_FILENAME
    if path.exists():
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return data
    legacy_path = dest / TRANSFER_FILENAME
    if legacy_path.exists():
        old = json.loads(legacy_path.read_text())
        return {_transfer_key(old): old}
    return {}


def load_transfer(model_dir=None):
    """Idaho (or the only belt) for the legacy /model-info fields."""
    catalog = load_transfers(model_dir)
    if not catalog:
        return None
    for key in IDAHO_TRANSFER_KEYS:
        if key in catalog:
            return catalog[key]
    return next(iter(catalog.values()))


def load_published_artifacts(model_dir=None):
    """Return (estimator, metadata_dict)."""
    model_path = published_model_path(model_dir)
    meta_path = published_meta_path(model_dir)
    estimator = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text())
    return estimator, metadata
