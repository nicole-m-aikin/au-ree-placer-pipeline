"""MLflow record shape from persisted metadata. Does not retrain."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.ml_artifacts import persist_belt_model, persist_published_model, persist_transfer
from pipeline.mlflow_tracking import (
    EXPERIMENT_NAME,
    PROMOTION_RULE,
    STAGE_EXPERIMENTAL,
    STAGE_PRODUCTION,
    iter_persisted_forests,
    region_and_stage,
    run_record_from_meta,
)
from sklearn.ensemble import RandomForestClassifier


def _doorbell_meta():
    return {
        'model_id': 'task9_rf_placer_gold',
        'is_doorbell': True,
        'study_area': 'NE Washington',
        'study_area_short': 'ne_wa',
        'feature_order': ['Th', 'U', 'Zr'],
        'feature_importances': {'U': 0.233, 'Th': 0.05, 'Zr': 0.04},
        'mrds_proximity_deg': 0.15,
        'n_samples': 1045,
        'n_positive': 290,
        'sklearn_version': '1.6.1',
        'random_state': 42,
        'n_estimators': 200,
        'class_weight': 'balanced',
        'label_method': 'proximity',
        'cv_auc_mean': 0.891,
        'spatial_cv_auc_mean': 0.699,
        'spatial_block_cv_auc_mean': 0.757,
    }


def _ca_meta():
    return {
        'model_id': 'task9_rf_placer_gold.ca_sierra_placer',
        'is_doorbell': False,
        'study_area': 'California Sierra placer foothills',
        'study_area_short': 'ca_sierra_placer',
        'feature_order': ['Th', 'Zr', 'Fe'],
        'feature_importances': {'Zr': 0.19, 'Fe': 0.18},
        'mrds_proximity_deg': 0.03,
        'n_samples': 596,
        'n_positive': 360,
        'sklearn_version': '1.6.1',
        'random_state': 42,
        'n_estimators': 200,
        'cv_auc_mean': 0.859,
        'spatial_cv_auc_mean': 0.831,
        'spatial_block_cv_auc_mean': 0.694,
    }


def _transfers():
    return {
        'Idaho Batholith': {'transfer_auc': 0.4986},
        'ca_sierra_placer': {'transfer_auc': 0.5223},
        'mt_placer': {'transfer_auc': 0.3438},
        'co_wet_mtns': {'transfer_auc': 0.5568},
        'nc_fall_zone': {'transfer_auc': 0.4957},
    }


def test_doorbell_is_production_ne_wa():
    region, stage = region_and_stage(_doorbell_meta())
    assert region == 'NE_WA'
    assert stage == STAGE_PRODUCTION


def test_california_stays_experimental_local():
    region, stage = region_and_stage(_ca_meta())
    assert region == 'CA_Sierra'
    assert stage == STAGE_EXPERIMENTAL


def test_doorbell_record_stores_spatial_and_transfer_not_just_shuffled():
    record = run_record_from_meta(_doorbell_meta(), transfers=_transfers())
    assert record['experiment'] == EXPERIMENT_NAME
    assert record['run_name'] == 'task9_rf_placer_gold'
    assert record['tags']['stage'] == 'production'
    assert record['tags']['region'] == 'NE_WA'
    assert record['tags']['task9_rf_placer_gold'] == 'true'
    assert record['tags']['promotion_rule'] == PROMOTION_RULE
    assert record['params']['feature_set'] == 'Th,U,Zr'
    assert record['params']['n_samples'] == 1045
    assert record['params']['sklearn_version'] == '1.6.1'
    assert record['params']['random_state'] == 42
    assert record['params']['label_radius_deg'] == 0.15
    assert record['metrics']['shuffled_auc'] == 0.891
    assert record['metrics']['deadzone_auc'] == 0.699
    assert record['metrics']['block_auc'] == 0.757
    assert record['metrics']['transfer_auc_id_batholith'] == 0.4986
    assert record['metrics']['transfer_auc_ca_sierra'] == 0.5223
    assert record['metrics']['importance_U'] == 0.233
    assert 'never_promote_largest_shuffled_cv' in record['tags']['promotion_rule']


def test_california_record_does_not_outpromote_doorbell():
    record = run_record_from_meta(_ca_meta(), transfers=_transfers())
    assert record['tags']['stage'] == 'experimental/local'
    assert record['tags']['region'] == 'CA_Sierra'
    assert 'task9_rf_placer_gold' not in record['tags']
    assert record['metrics']['shuffled_auc'] == 0.859
    assert record['metrics']['block_auc'] == 0.694
    assert record['metrics']['doorbell_transfer_auc'] == 0.5223
    wa = run_record_from_meta(_doorbell_meta(), transfers=_transfers())
    # CA dead-zone 0.83 looks better than WA 0.70. That is not a ship rule.
    assert record['metrics']['deadzone_auc'] > wa['metrics']['deadzone_auc']
    assert record['metrics']['block_auc'] < wa['metrics']['block_auc']
    assert record['tags']['stage'] != wa['tags']['stage']
    assert wa['tags']['stage'] == 'production'


def test_checked_in_models_keep_wa_as_production():
    from pipeline.ml_artifacts import MODEL_DIR, load_transfers, published_meta_path
    meta_path = published_meta_path(MODEL_DIR)
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    record = run_record_from_meta(meta, transfers=load_transfers(MODEL_DIR))
    assert record['tags']['stage'] == 'production'
    assert record['tags']['region'] == 'NE_WA'
    assert record['metrics']['shuffled_auc'] == meta['cv_auc_mean']
    assert 'transfer_auc_ca_sierra' in record['metrics']


def test_maybe_log_is_noop_without_tracking_uri(tmp_path, monkeypatch):
    from pipeline.mlflow_tracking import maybe_log_after_persist
    monkeypatch.delenv('MLFLOW_TRACKING_URI', raising=False)
    maybe_log_after_persist(tmp_path / 'missing.meta.json', tmp_path / 'missing.joblib')


def test_iter_persisted_forests_finds_doorbell_and_sidecar(tmp_path):
    rf = RandomForestClassifier(n_estimators=3, random_state=0)
    rf.fit([[0.0, 1.0], [1.0, 0.0]], [0, 1])
    persist_published_model(rf, _doorbell_meta(), model_dir=tmp_path)
    persist_belt_model(rf, _ca_meta(), 'ca_sierra_placer', model_dir=tmp_path)
    persist_transfer(
        {'belt': 'Idaho Batholith', 'transfer_auc': 0.5, 'retrained': False},
        model_dir=tmp_path,
    )
    pairs = list(iter_persisted_forests(tmp_path))
    stems = {meta.name for meta, _ in pairs}
    assert 'task9_rf_placer_gold.meta.json' in stems
    assert 'task9_rf_placer_gold.ca_sierra_placer.meta.json' in stems
    for meta_path, model_path in pairs:
        assert model_path.exists()
        json.loads(meta_path.read_text())
