"""Tests for the Task 9 FastAPI endpoints."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import sklearn
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.ml_artifacts import persist_gold_mrds, persist_published_model, persist_transfer
from pipeline.ml_preprocess import FEATURES
from pipeline.ml_spatial import NEGATIVE_CLASS_NOTE, TARGET_QUESTION

_VALID = {
    'Th': 12.0, 'Ce': 66.0, 'La': 38.0, 'P': 700.0, 'U': 3.2,
    'Au': 0.004, 'As': 1.5, 'Ti': 520.0, 'Fe': 2.7, 'Zr': 96.0, 'Y': 18.0,
    'fe_unit': 'wt_pct',
}


def _tiny_artifacts(tmp_path, sklearn_version=None):
    """Train a tiny 11-feature forest and persist it with known medians."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, len(FEATURES)))
    y = (X[:, 0] + X[:, 4] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=11, random_state=0)
    rf.fit(X, y)
    meta = {
        'model_id': 'task9_rf_placer_gold',
        'feature_order': list(FEATURES),
        'feature_units': {f: 'ppm' for f in FEATURES},
        'log_medians': {f: 1.0 for f in FEATURES},
        'training_date': '2026-09-21T00:00:00Z',
        'sklearn_version': sklearn_version or sklearn.__version__,
        'n_estimators': 11,
        'label_method': 'proximity',
        'mrds_proximity_deg': 0.15,
        'mrds_commodity_filter': 'placer_gold',
        'n_samples': 80,
        'n_positive': int(y.sum()),
        'cv_auc_folds': [0.87, 0.89, 0.90, 0.88, 0.91],
        'cv_auc_mean': 0.89,
        'cv_auc_std': 0.015,
        'feature_importances': {f: 0.0 for f in FEATURES},
        'study_area': 'test',
        'resource_tonnage_uncertainty': (
            'Resource-tonnage uncertainty (Task 4 Monte Carlo; P10/P50/P90 '
            'NdPr tonnes) is a separate quantity and is not exposed by this API.'
        ),
        'spatial_cv_auc_mean': 0.72,
        'spatial_cv_auc_std': 0.04,
        'spatial_cv_auc_folds': [0.70, 0.74],
        'spatial_block_cv_auc_mean': 0.68,
        'spatial_cv_method': 'test',
        'target_question': TARGET_QUESTION,
        'negative_class_note': NEGATIVE_CLASS_NOTE,
        'cv_optimism_note': 'random CV can be optimistic',
    }
    meta['feature_importances']['U'] = 0.233
    persist_published_model(rf, meta, model_dir=tmp_path)
    persist_gold_mrds([[-118.21, 48.14], [-117.80, 48.60]], model_dir=tmp_path)
    persist_transfer({
        'belt': 'Idaho Batholith',
        'transfer_auc': 0.61,
        'n_samples': 400,
        'n_positive': 80,
        'retrained': False,
        'note': 'Frozen NE WA forest. Not a national model.',
    }, model_dir=tmp_path)
    return tmp_path


@pytest.fixture
def client(tmp_path):
    _tiny_artifacts(tmp_path)
    from api.app import create_app
    app = create_app(model_dir=tmp_path)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mismatch_client(tmp_path):
    _tiny_artifacts(tmp_path, sklearn_version='0.0.0-not-installed')
    from api.app import create_app
    app = create_app(model_dir=tmp_path)
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_503_when_artifacts_missing(self, tmp_path):
        from api.app import create_app
        app = create_app(model_dir=tmp_path)
        with TestClient(app) as c:
            r = c.get('/health')
        assert r.status_code == 503
        assert r.json()['model_loaded'] is False

    def test_ok_when_artifacts_match(self, client):
        r = client.get('/health')
        assert r.status_code == 200
        body = r.json()
        assert body['status'] == 'ok'
        assert body['model_loaded'] is True
        assert body['metadata_loaded'] is True
        assert body['sklearn_match'] is True
        assert body['sklearn_version'] == sklearn.__version__
        assert body['sklearn_version_training'] == sklearn.__version__

    def test_503_on_sklearn_mismatch(self, mismatch_client):
        r = mismatch_client.get('/health')
        assert r.status_code == 503
        body = r.json()
        assert body['status'] == 'unavailable'
        assert body['sklearn_match'] is False
        assert body['model_loaded'] is True


class TestModelInfo:
    def test_returns_cv_and_task4_disclaimer(self, client):
        r = client.get('/model-info')
        assert r.status_code == 200
        body = r.json()
        assert body['cv_auc_mean'] == pytest.approx(0.89)
        assert body['feature_importances']['U'] == pytest.approx(0.233)
        assert body['label_method'] == 'proximity'
        assert 'Task 4' in body['resource_tonnage_uncertainty']
        assert 'P10/P50/P90' in body['resource_tonnage_uncertainty']
        assert body['feature_order'] == FEATURES
        assert 'gold' in body['target_question'].lower()
        assert 'barren' in body['negative_class_note']
        assert body['spatial_cv_auc_mean'] == pytest.approx(0.72)
        assert body['cv_auc_mean'] == pytest.approx(0.89)
        assert body['transfer_belt'] == 'Idaho Batholith'
        assert body['transfer_auc'] == pytest.approx(0.61)
        assert body['transfer_retrained'] is False
        assert 'national' in body['transfer_note']
        assert body['transfer_belts']
        assert body['transfer_belts'][0]['belt'] == 'Idaho Batholith'

    def test_503_when_sklearn_mismatch(self, mismatch_client):
        r = mismatch_client.get('/model-info')
        assert r.status_code == 503


class TestDoorbell:
    def test_root_says_what_this_is_not(self, client):
        r = client.get('/')
        assert r.status_code == 200
        body = r.json()
        assert 'doorbell' in body['name'].lower()
        assert any('national' in str(x).lower() for x in body['not'])


class TestPredict:
    def test_valid_request_returns_label_proba_and_spread(self, client):
        r = client.post('/predict', json=_VALID)
        assert r.status_code == 200
        body = r.json()
        assert body['label'] in (0, 1)
        assert 0.0 <= body['probability'] <= 1.0
        assert body['tree_vote_spread'] >= 0.0
        assert body['n_trees'] == 11
        assert 'monte_carlo' not in body
        assert 'monte_carlo' not in json.dumps(body).lower()
        assert body['spatial_flag'] == 'unknown_no_location'
        assert body['nearest_gold_mrds_deg'] is None

    def test_near_gold_location_is_flagged(self, client):
        payload = dict(_VALID)
        payload['lon'] = -118.21
        payload['lat'] = 48.14
        r = client.post('/predict', json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body['spatial_flag'] == 'near_known_gold'
        assert body['nearest_gold_mrds_deg'] == pytest.approx(0.0, abs=1e-6)

    def test_far_from_gold_location_is_flagged(self, client):
        payload = dict(_VALID)
        payload['lon'] = -119.90
        payload['lat'] = 47.60
        r = client.post('/predict', json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body['spatial_flag'] == 'far_from_known_gold'
        assert body['nearest_gold_mrds_deg'] > 0.15

    def test_null_au_uses_frozen_median_not_crash(self, client):
        payload = dict(_VALID)
        payload['Au'] = None
        r = client.post('/predict', json=payload)
        assert r.status_code == 200

    def test_fe_wt_pct_and_fe_ppm_are_not_the_same_score(self, client):
        wt = dict(_VALID)
        wt['Fe'] = 3.0
        wt['fe_unit'] = 'wt_pct'
        ppm = dict(_VALID)
        ppm['Fe'] = 30000.0
        ppm['fe_unit'] = 'ppm'
        # 3 wt% × 10,000 = 30,000 ppm — these two must match after conversion
        a = client.post('/predict', json=wt).json()
        b = client.post('/predict', json=ppm).json()
        assert a['probability'] == pytest.approx(b['probability'], abs=1e-12)

    def test_malformed_missing_feature_is_422(self, client):
        bad = dict(_VALID)
        del bad['Th']
        r = client.post('/predict', json=bad)
        assert r.status_code == 422

    def test_malformed_string_feature_is_422(self, client):
        bad = dict(_VALID)
        bad['Th'] = 'not-a-number'
        r = client.post('/predict', json=bad)
        assert r.status_code == 422

    def test_out_of_range_th_is_422(self, client):
        bad = dict(_VALID)
        bad['Th'] = 1.0e6
        r = client.post('/predict', json=bad)
        assert r.status_code == 422

    def test_artifact_negative_below_mdl_convention_is_422(self, client):
        bad = dict(_VALID)
        bad['Au'] = -99.0
        r = client.post('/predict', json=bad)
        assert r.status_code == 422

    def test_unconverted_p_wt_pct_is_422(self, client):
        bad = dict(_VALID)
        bad['P'] = 0.07
        r = client.post('/predict', json=bad)
        assert r.status_code == 422

    def test_fe_ppm_in_wt_pct_range_is_422(self, client):
        bad = dict(_VALID)
        bad['Fe'] = 2.7
        bad['fe_unit'] = 'ppm'
        r = client.post('/predict', json=bad)
        assert r.status_code == 422

    def test_fe_wt_pct_out_of_range_is_422(self, client):
        bad = dict(_VALID)
        bad['Fe'] = 50.0
        bad['fe_unit'] = 'wt_pct'
        r = client.post('/predict', json=bad)
        assert r.status_code == 422

    def test_missing_fe_unit_is_422(self, client):
        bad = dict(_VALID)
        del bad['fe_unit']
        r = client.post('/predict', json=bad)
        assert r.status_code == 422
