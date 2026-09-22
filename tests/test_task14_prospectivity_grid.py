"""Task 14 extras — lith score, slope, matrix. No doorbell, no downloads."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.task14_prospectivity_grid import (
    LITH_SCORE,
    build_matrix,
    lith_score_from_type,
    sample_slope_deg,
)


def test_lith_score_is_the_same_number_on_every_belt():
    assert lith_score_from_type('MCC_metapelite') == 3.0
    assert lith_score_from_type('felsic_intrusive') == 2.0
    assert lith_score_from_type('sedimentary_cover') == 1.0
    assert lith_score_from_type('mafic_ultramafic') == 0.0
    assert lith_score_from_type('unknown_unit') == 1.0
    assert np.isnan(lith_score_from_type(None))
    assert set(LITH_SCORE) == {
        'MCC_metapelite', 'felsic_intrusive', 'sedimentary_cover', 'mafic_ultramafic',
    }


def test_sample_slope_deg_sees_a_ramp(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin

    # 5×5 geographic grid, 0.001° cells, z rises 100 m per cell east.
    z = np.tile(np.arange(5, dtype=np.float32) * 100.0, (5, 1))
    path = tmp_path / 'ramp.tif'
    transform = from_origin(-120.0, 48.0, 0.001, 0.001)
    with rasterio.open(
        path, 'w', driver='GTiff', height=5, width=5, count=1,
        dtype='float32', crs='EPSG:4326', transform=transform,
    ) as dst:
        dst.write(z, 1)
    # Center pixel (row 2, col 2) → lon -120+0.0025, lat 48-0.0025
    slope = sample_slope_deg(str(path), [-119.9975], [47.9975])
    assert np.isfinite(slope[0])
    assert slope[0] > 20  # a 100 m rise over ~80 m is steep
    edge = sample_slope_deg(str(path), [-120.0], [48.0])
    assert np.isnan(edge[0])  # 3×3 does not fit on the rim


def test_build_matrix_logs_chem_and_leaves_lith_linear():
    df = pd.DataFrame({
        'Th': [10.0, 100.0],
        'Ce': [20.0, 200.0],
        'lith_score': [3.0, np.nan],
        'slope_deg': [2.0, 8.0],
    })
    chem = ['Th', 'Ce']
    X, cols, med = build_matrix(df, ('chem', 'lith'), chem)
    assert cols == ['Th', 'Ce', 'lith_score']
    assert X.shape == (2, 3)
    assert abs(X[0, 0] - np.log10(10)) < 1e-9
    assert X[1, 2] == 3.0  # NaN lith filled with the column median (3)
    X2, _c2, _ = build_matrix(df, ('chem', 'lith'), chem, medians=med)
    np.testing.assert_allclose(X, X2)


def test_lith_slope_only_has_no_chemistry_columns():
    df = pd.DataFrame({
        'Th': [10.0, 20.0],
        'lith_score': [0.0, 3.0],
        'slope_deg': [12.0, 1.0],
    })
    X, cols, _ = build_matrix(df, ('lith', 'slope'), ['Th'])
    assert cols == ['lith_score', 'slope_deg']
    assert X[0, 0] == 0.0
    assert X[1, 1] == 1.0
