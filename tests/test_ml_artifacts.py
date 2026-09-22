"""Tests for published-model persist and the shared MDL helper."""

import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.ml_artifacts import (
    load_published_artifacts,
    load_transfer,
    load_transfers,
    persist_published_model,
    persist_transfer,
    published_meta_path,
    published_model_path,
)
from pipeline.ml_preprocess import apply_nure_mdl


def test_persist_writes_joblib_and_log_medians(tmp_path):
    rf = RandomForestClassifier(n_estimators=3, random_state=0)
    rf.fit([[0.0, 1.0], [1.0, 0.0]], [0, 1])
    meta = {
        'feature_order': ['Th', 'U'],
        'log_medians': {'Th': 1.1, 'U': 0.4},
        'sklearn_version': '1.6.1',
    }
    persist_published_model(rf, meta, model_dir=tmp_path)
    assert published_model_path(tmp_path).exists()
    assert published_meta_path(tmp_path).exists()
    loaded_rf, loaded_meta = load_published_artifacts(tmp_path)
    assert loaded_meta['log_medians']['Th'] == 1.1
    assert list(loaded_rf.predict([[0.0, 1.0]])) == list(rf.predict([[0.0, 1.0]]))
    raw = json.loads(published_meta_path(tmp_path).read_text())
    assert raw['log_medians']['U'] == 0.4


def test_persist_transfer_keeps_idaho_when_california_is_added(tmp_path):
    persist_transfer({
        'belt': 'Idaho Batholith',
        'short': 'id_batholith',
        'transfer_auc': 0.50,
        'retrained': False,
    }, model_dir=tmp_path)
    persist_transfer({
        'belt': 'California Sierra placer foothills',
        'short': 'ca_sierra_placer',
        'transfer_auc': 0.51,
        'retrained': False,
    }, model_dir=tmp_path)
    catalog = load_transfers(tmp_path)
    assert catalog['id_batholith']['transfer_auc'] == 0.50
    assert catalog['ca_sierra_placer']['transfer_auc'] == 0.51
    legacy = load_transfer(tmp_path)
    assert legacy['short'] == 'id_batholith'
    assert legacy['transfer_auc'] == 0.50


def test_apply_nure_mdl_half_and_missing():
    assert apply_nure_mdl(-5.0) == 2.5
    assert apply_nure_mdl(12.0) == 12.0
    assert apply_nure_mdl(0.0) is None
    assert apply_nure_mdl(None) is None
    assert apply_nure_mdl(np.nan) is None
