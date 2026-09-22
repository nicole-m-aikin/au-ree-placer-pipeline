"""Tests for published-model persist and the shared MDL helper."""

import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.ml_artifacts import (
    belt_meta_path,
    belt_model_path,
    is_published_training_belt,
    load_published_artifacts,
    load_transfer,
    load_transfers,
    persist_belt_model,
    persist_published_model,
    persist_transfer,
    published_meta_path,
    published_model_path,
    study_area_short,
)
from pipeline.ml_preprocess import (
    COUSIN_FEATURES,
    apply_nure_mdl,
    feature_list_for_cfg,
    usable_feature_columns,
)


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


def test_belt_persist_does_not_touch_doorbell_files(tmp_path):
    rf = RandomForestClassifier(n_estimators=3, random_state=0)
    rf.fit([[0.0, 1.0], [1.0, 0.0]], [0, 1])
    persist_published_model(rf, {'feature_order': ['Th']}, model_dir=tmp_path)
    before = published_model_path(tmp_path).read_bytes()
    persist_belt_model(
        rf,
        {'feature_order': ['Th'], 'is_doorbell': False, 'study_area_short': 'ca_sierra_placer'},
        'ca_sierra_placer',
        model_dir=tmp_path,
    )
    assert belt_model_path('ca_sierra_placer', tmp_path).exists()
    assert belt_meta_path('ca_sierra_placer', tmp_path).exists()
    assert published_model_path(tmp_path).read_bytes() == before


def test_only_ne_wa_is_published_training_belt():
    assert is_published_training_belt({'study_area': {'short': 'ne_wa'}})
    assert is_published_training_belt({'study_area': {'short': 'ne_washington'}})
    assert not is_published_training_belt(
        {'study_area': {'short': 'ca_sierra_placer'}}
    )
    assert study_area_short({'study_area': {'short': 'ca_sierra_placer'}}) == (
        'ca_sierra_placer'
    )


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


def test_cousins_are_opt_in_and_same_list():
    assert COUSIN_FEATURES == ['Cr', 'Nb', 'Hf', 'Sc', 'W']
    base = feature_list_for_cfg({'ml': {}}, 'placer_gold')
    assert 'Cr' not in base
    with_c = feature_list_for_cfg({'ml': {'include_cousins': True}}, 'placer_gold')
    assert with_c[-5:] == COUSIN_FEATURES
    assert 'Th' in with_c


def test_usable_features_drop_empty_au_as():
    import pandas as pd
    df = pd.DataFrame({
        'Th': [12.0, 8.0],
        'Au': [np.nan, np.nan],
        'As': [np.nan, 0.0],
        'Fe': [2.1, 3.0],
    })
    kept, dropped = usable_feature_columns(df, ['Th', 'Au', 'As', 'Fe', 'Zr'])
    assert kept == ['Th', 'Fe']
    assert dropped == ['Au', 'As', 'Zr']


def test_ca_sierra_config_uses_tight_label_radius():
    import yaml
    path = os.path.join(
        os.path.dirname(__file__), '..', 'configs', 'california_sierra', 'config.yaml',
    )
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert cfg['ml']['mrds_proximity_deg'] == 0.03
    assert cfg['ml']['mrds_commodity_filter'] == 'placer_gold'
    assert cfg['task11']['p_is_transfer'] is False
    assert not is_published_training_belt(cfg)


def test_idaho_and_montana_sidecars_use_the_same_tight_recipe():
    import yaml
    root = os.path.join(os.path.dirname(__file__), '..', 'configs')
    for rel in (
        'idaho_batholith/config.yaml',
        'montana_placer/config.yaml',
        'colorado_wet_mtns/config.yaml',
    ):
        with open(os.path.join(root, rel)) as f:
            cfg = yaml.safe_load(f)
        assert cfg['ml']['mrds_proximity_deg'] == 0.03
        assert cfg['ml']['mrds_commodity_filter'] == 'placer_gold'
        assert cfg['task11']['p_is_transfer'] is False
        assert not is_published_training_belt(cfg)
