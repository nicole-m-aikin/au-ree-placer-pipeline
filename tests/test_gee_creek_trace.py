"""Pure-function tests for EE MERIT creek walk helpers (no network)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.gee_creek_trace import (  # noqa: E402
    DIR_DELTA,
    _path_to_vertices,
    _rc_to_lonlat,
    _snap,
    _step,
    ee_trace_enabled,
)


def _toy_window():
    # 5×5 grid; flow east then south into a channel with higher upa
    dir_a = np.full((5, 5), 1.0)  # east
    dir_a[:, 4] = 4.0             # south on east edge
    upa = np.ones((5, 5)) * 0.1
    upa[2, :] = 5.0
    elv = np.linspace(1200, 1000, 25).reshape(5, 5)
    return {
        'dir': dir_a, 'upa': upa, 'elv': elv,
        'west': -118.0, 'south': 35.0, 'east': -117.9, 'north': 35.1,
        'res_x': 0.02, 'res_y': 0.02,
        'nrows': 5, 'ncols': 5,
    }


def test_ee_trace_enabled_defaults_true():
    assert ee_trace_enabled({}) is True
    assert ee_trace_enabled({'county_rank': {'ee_trace': False}}) is False
    assert ee_trace_enabled({'gee': {'trace': False}}) is False


def test_dir_delta_has_eight_neighbors():
    assert set(DIR_DELTA) == {1, 2, 4, 8, 16, 32, 64, 128}


def test_snap_picks_highest_upa_in_radius():
    w = _toy_window()
    lon, lat = _rc_to_lonlat(w, 2, 2)
    r, c, d, u = _snap(w, lon, lat, snap_m=5000, min_snap_km2=1.0)
    assert r == 2
    assert u >= 5.0
    assert d >= 0


def test_step_follows_dir_and_reverse():
    w = _toy_window()
    assert _step(w, 2, 0, reverse=False) == (2, 1)  # east
    assert _step(w, 2, 1, reverse=True) == (2, 0)


def test_path_to_vertices_returns_reach():
    path = [(-118.05 + i * 0.001, 35.05) for i in range(20)]
    elev = list(range(1100, 1080, -1))
    upa = [2.0] * 20
    vert = _path_to_vertices(path, elev, upa, spacing_m=50)
    assert vert is not None
    assert vert['reach_km'] > 0
    assert len(vert['lon']) >= 3
