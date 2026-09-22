"""Second-belt transfer helpers — no USGS download required."""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.ml_preprocess import FEATURES, _log_impute
from pipeline.task12_second_belt import (
    _assert_not_training_belt,
    _transfer_note,
    belt_slug,
    hssr_to_nure_frame,
    parse_mrds_bbox_xml,
    score_frozen_transfer,
)


def test_hssr_maps_units_and_mdl_flags():
    raw = pd.DataFrame({
        'latitude': [45.5],
        'longitude': [-115.5],
        'th_ppm': [31],
        'ce_ppm': ['188'],
        'la_ppm': ['<5'],
        'u_dn_ppm': [7.84],
        'u_ppm': [99.0],
        'fe_pct': [2.16],
        'au_ppm': ['L0.07'],
        'as_ppm': [9],
        'ti_ppm': [4873],
        'zr_ppm': [717],
    })
    out = hssr_to_nure_frame(raw)
    assert out.loc[0, 'lat'] == 45.5
    assert out.loc[0, 'Th'] == 31
    assert out.loc[0, 'U'] == 7.84  # delayed-neutron wins over u_ppm
    assert out.loc[0, 'La'] == -5.0
    assert out.loc[0, 'Au'] == -0.07
    assert out.loc[0, 'Fe'] == 2.16
    assert 'P' not in out.columns or pd.isna(out.loc[0].get('P', np.nan))


def test_mrds_xml_keeps_gold_drops_ree():
    xml = '''<search>
      <record dep_id="1" site_name="Warren Creek" code_list=" AU " x="-115.65" y="45.25"/>
      <record dep_id="2" site_name="Monazite pit" code_list=" REE TH " x="-115.60" y="45.30"/>
      <record dep_id="3" site_name="No coords" code_list=" AU "/>
    </search>'''
    df = parse_mrds_bbox_xml(xml)
    assert list(df['dep_id']) == ['1']
    assert df.iloc[0]['target_commodity'] == 'Gold'


def test_frozen_transfer_auc_is_not_random_on_separated_chemistry():
    rng = np.random.default_rng(0)
    n = 80
    # Near-gold chemistry is shifted on U/Th the same way the forest can see.
    near = np.ones(n // 2, dtype=int)
    y = np.concatenate([near, np.zeros(n // 2, dtype=int)])
    lon = np.concatenate([np.full(n // 2, -115.5), np.full(n // 2, -114.2)])
    lat = np.concatenate([np.full(n // 2, 45.4), np.full(n // 2, 45.4)])
    gold_xy = np.array([[-115.5, 45.4]])
    raw = {f: rng.uniform(1, 3, n) for f in FEATURES}
    raw['U'] = np.where(y == 1, 20.0, 2.0)
    raw['Th'] = np.where(y == 1, 30.0, 5.0)
    raw['lon'] = lon
    raw['lat'] = lat
    df = pd.DataFrame(raw)
    log_X, med = _log_impute(df, FEATURES)
    rf = RandomForestClassifier(n_estimators=20, random_state=0)
    rf.fit(log_X.values, y)
    meta = {'feature_order': list(FEATURES), 'log_medians': med}
    scored, auc = score_frozen_transfer(df, gold_xy, rf, meta, radius_deg=0.15)
    assert auc is not None
    assert auc > 0.8
    assert int(scored['label'].sum()) == n // 2


def test_belt_slug_keeps_historic_idaho_name():
    assert belt_slug({'study_area': {'short': 'id_batholith'}}) == 'idaho'
    assert belt_slug({'study_area': {'short': 'ca_sierra_placer'}}) == 'ca_sierra_placer'
    assert belt_slug({'study_area': {'short': 'mt_placer'}}) == 'mt_placer'


def test_training_belt_is_rejected():
    try:
        _assert_not_training_belt(
            {'study_area': {'name': 'NE Washington'}},
            {'study_area': 'NE Washington'},
        )
    except RuntimeError as exc:
        assert 'training belt' in str(exc)
    else:
        raise AssertionError('expected RuntimeError')
    _assert_not_training_belt(
        {'study_area': {'name': 'California Sierra placer foothills'}},
        {'study_area': 'NE Washington'},
    )


def test_transfer_note_calls_coin_flip_when_auc_is_chance():
    note = _transfer_note('California Sierra placer foothills', 0.50, 0.48, 0.8, 0.38, 0.37)
    assert 'coin flip' in note
    assert 'weak negative' in note
    assert 'national' in note


def test_transfer_note_flags_inverted_mean_p():
    note = _transfer_note('Montana SW gold gulches', 0.34, 0.33, 0.94, 0.28, 0.38)
    assert 'higher far from gold' in note
    assert 'flat' not in note
