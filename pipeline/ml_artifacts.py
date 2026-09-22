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


def persist_transfer(summary, model_dir=None):
    dest = Path(model_dir) if model_dir else MODEL_DIR
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / TRANSFER_FILENAME
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return path


def load_transfer(model_dir=None):
    """Idaho (or later) transfer sidecar. None if the second belt has not run."""
    dest = Path(model_dir) if model_dir else MODEL_DIR
    path = dest / TRANSFER_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_published_artifacts(model_dir=None):
    """Return (estimator, metadata_dict)."""
    model_path = published_model_path(model_dir)
    meta_path = published_meta_path(model_dir)
    estimator = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text())
    return estimator, metadata
