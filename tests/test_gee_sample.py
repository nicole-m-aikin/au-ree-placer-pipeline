"""Earth Engine sample helpers — pure functions, no network."""

import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.gee_sample import (  # noqa: E402
    FAVORABLE_LITH,
    NLCD_DROP,
    _ms_to_year,
    apply_sample_flags,
    best_months_from_climate,
    format_month_span,
    land_filter,
    water_regime,
)


def test_water_regime_buckets():
    assert water_regime(None) == 'dry'
    assert water_regime(float('nan')) == 'dry'
    assert water_regime(0) == 'dry'
    assert water_regime(10) == 'seasonal'
    assert water_regime(24.9) == 'seasonal'
    assert water_regime(25) == 'year_round'
    assert water_regime(90) == 'year_round'


def test_land_filter_drops_developed_and_crops():
    assert land_filter(42) == 'ok'          # evergreen forest
    assert land_filter(31) == 'ok'          # barren
    for code in NLCD_DROP:
        assert land_filter(code) == 'drop'
    assert land_filter(None) == 'unknown'
    assert land_filter(float('nan')) == 'unknown'


def test_favorable_lith_includes_metamorphic_and_plutonic():
    assert 4 in FAVORABLE_LITH and 5 in FAVORABLE_LITH
    assert 1 not in FAVORABLE_LITH  # unconsolidated
    assert 2 not in FAVORABLE_LITH  # carbonate


def test_ms_to_year():
    ms = datetime(2008, 6, 28, tzinfo=timezone.utc).timestamp() * 1000
    assert _ms_to_year(ms) == 2008
    assert _ms_to_year(None) != _ms_to_year(None) or True  # NaN
    import math
    assert math.isnan(_ms_to_year(None))


def test_format_month_span():
    assert format_month_span([7, 8, 9, 10]) == 'Jul–Oct'
    assert format_month_span([7, 9, 10]) == 'Jul, Sep–Oct'
    assert format_month_span([1]) == 'Jan'
    assert format_month_span([]) == ''


def test_best_months_snow_free_below_median_ppt():
    # Snow all winter; summer snow-free; Jul driest among free months
    snow = [40, 50, 40, 20, 5, 0, 0, 0, 0, 10, 25, 35]
    ppt = [80, 70, 60, 40, 20, 10, 5, 8, 15, 30, 50, 70]
    assert best_months_from_climate(snow, ppt) == 'Jun–Aug'


def test_best_months_no_snow_falls_back_to_driest_three():
    snow = [None] * 12
    ppt = [100, 90, 80, 70, 60, 50, 10, 20, 30, 40, 110, 120]
    assert best_months_from_climate(snow, ppt) == 'Jul–Sep'


def test_apply_sample_flags_joins_on_pin_id():
    import pandas as pd

    gdf = pd.DataFrame({'pin_id': ['a', 'b'], 'lon': [1.0, 2.0], 'lat': [3.0, 4.0]})
    samples = pd.DataFrame({
        'pin_id': ['a', 'b'],
        'elev10_m': [100.0, 200.0],
        'slope10_deg': [5.0, 6.0],
        'water_pct': [0.0, 50.0],
        'water_regime': ['dry', 'year_round'],
        'lith_code': [5, 1],
        'lith_name': ['acid_plutonic', 'unconsolidated'],
        'lith_favor': [True, False],
        'landform_code': [31, 11],
        'nlcd': [42, 22],
        'nlcd_name': ['evergreen', 'developed_low'],
        'land_filter': ['ok', 'drop'],
        'road_m': [100.0, 500.0],
        'fire_year': [2008.0, float('nan')],
        'has_lidar_1m': [True, False],
        'best_months': ['Jul–Sep', 'May–Jun'],
    })
    out = apply_sample_flags(gdf, samples)
    assert list(out['land_filter']) == ['ok', 'drop']
    assert out.loc[0, 'lith_favor'] is True or out.loc[0, 'lith_favor'] == True
    assert out.loc[1, 'water_regime'] == 'year_round'
    assert out.loc[0, 'best_months'] == 'Jul–Sep'
