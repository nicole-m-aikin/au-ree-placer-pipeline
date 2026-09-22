"""Persist / locate the published Task 9 Random Forest.

The API loads these files. Task 9 writes them after rf_final.fit so the
log-medians are the ones training actually used.
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


def published_model_path(model_dir=None):
    d = Path(model_dir) if model_dir else MODEL_DIR
    return d / MODEL_FILENAME


def published_meta_path(model_dir=None):
    d = Path(model_dir) if model_dir else MODEL_DIR
    return d / META_FILENAME


def published_gold_mrds_path(model_dir=None):
    d = Path(model_dir) if model_dir else MODEL_DIR
    return d / GOLD_MRDS_FILENAME


def persist_published_model(estimator, metadata, model_dir=None):
    """Write joblib + metadata JSON. Returns (model_path, meta_path)."""
    dest = Path(model_dir) if model_dir else MODEL_DIR
    dest.mkdir(parents=True, exist_ok=True)
    model_path = dest / MODEL_FILENAME
    meta_path = dest / META_FILENAME
    joblib.dump(estimator, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    return model_path, meta_path


def persist_gold_mrds(coords, model_dir=None):
    """Write gold MRDS lon/lat used for training labels (serve-time distance flag)."""
    dest = Path(model_dir) if model_dir else MODEL_DIR
    dest.mkdir(parents=True, exist_ok=True)
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
