"""Geometry-only walks — synthetic grids, no DEM or network."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from affine import Affine
from pipeline.task11_geometry_walk import (
    _read_window,
    flat_water_mask,
    junction_mask,
    pick_pans,
    rank_walks,
    snap_to_channel,
    trace_downstream,
    trace_upstream,
    walk_specs,
)

RES = 1 / 3600.0
E, S = 1, 4  # pysheds dirmap: east, south


def _east_river(n=40):
    """Row 20 flows east; a tributary enters from the north at column 20."""
    fdir = np.full((n, n), S, dtype=np.int32)
    acc = np.ones((n, n), dtype=float)
    fdir[20, :] = E
    acc[20, :] = 1000.0 + np.arange(n)
    acc[20, 20:] += 400.0
    fdir[:20, 20] = S
    acc[:20, 20] = 10.0 + np.arange(20)
    return fdir, acc


def test_snap_prefers_nearest_big_channel_over_biggest_gully():
    acc = np.ones((41, 41))
    acc[20, 22] = 50.0          # gully next to the pin
    acc[25, :] = 5000.0         # the named river, five cells away
    acc[40, 40] = 1e6           # a bigger trunk at the edge of the window
    t = Affine(RES, 0, -121.0, 0, -RES, 39.0)
    lon = -121.0 + 20.5 * RES
    lat = 39.0 - 20.5 * RES
    r, c, d = snap_to_channel(acc, t, lon, lat, min_cells=1000, snap_m=400)
    assert r == 25
    assert d < 200


def test_snap_returns_none_when_no_channel_is_big_enough():
    acc = np.ones((11, 11))
    t = Affine(RES, 0, -121.0, 0, -RES, 39.0)
    assert snap_to_channel(acc, t, -121.0 + 5 * RES, 39.0 - 5 * RES, min_cells=10) is None


def test_trace_downstream_stops_at_trunk():
    fdir, acc = _east_river()
    path, why = trace_downstream(fdir, acc, 5, 20, RES, 39.0, 5000.0, stop_cells=900.0)
    assert why == 'reached_trunk'
    assert path[-1] == (20, 20)


def test_trace_downstream_runs_to_length_on_the_trunk():
    fdir, acc = _east_river()
    path, why = trace_downstream(fdir, acc, 20, 2, RES, 39.0, 300.0)
    assert why == 'length'
    assert all(r == 20 for r, _ in path)
    assert path[-1][1] > 2


def test_trace_upstream_follows_largest_inflow():
    fdir, acc = _east_river()
    path, _ = trace_upstream(fdir, acc, 20, 25, RES, 39.0, 400.0)
    # Stays on the trunk (acc ~1000) past the tributary mouth (acc ~30).
    assert all(r == 20 for r, _ in path)


def test_junction_mask_flags_a_real_tributary_only():
    acc = np.array([100, 101, 102, 150, 151, 152, 153], dtype=float)
    s = np.arange(acc.size) * 30.0
    sv = np.array([0.0, 90.0, 180.0, 600.0])
    hit = junction_mask(acc, s, sv, cells_per_km2=10.0, frac=0.1, min_km2=1.0, radius_m=60.0)
    assert hit.tolist() == [False, True, False, False]


def test_flat_water_mask_catches_a_reservoir_pool():
    z = np.array([200.0, 190.0, 180.0, 170.0, 160.6, 160.6, 160.6, 160.6, 160.6, 160.6])
    flat = flat_water_mask(z)
    assert not flat[:3].any()
    assert flat[6:].all()


def test_pick_pans_respects_earlier_walks():
    v = pd.DataFrame({
        'x': [0.0, 100.0, 1000.0],
        'y': [0.0, 0.0, 0.0],
        'trap_score': [6.0, 5.0, 4.0],
    })
    picked = pick_pans(v, n=2, taken_xy=[(50.0, 0.0)])
    assert picked['x'].tolist() == [1000.0]


def test_rank_walks_puts_access_ok_pans_first():
    walks = pd.DataFrame({'walk_id': ['a', 'b']})
    pans = pd.DataFrame({
        'walk_id': ['a', 'a', 'b'],
        'trap_score': [8.0, 7.5, 6.0],
        'access_ok': ['no', 'no', 'yes'],
    })
    ranked = rank_walks(walks, pans)
    assert ranked.iloc[0]['walk_id'] == 'b'
    assert ranked.iloc[0]['walk_rank'] == 1


def test_read_window_past_raster_edge_does_not_shift_header(tmp_path):
    """A seed within WINDOW_PAD_DEG of a small region's bbox edge used to get a
    transform computed for the unclipped (overflowing) window while read()
    silently clipped the pixels to the raster — same bug class the DEM mosaic
    clip (fetch_dem._clip_dem) was fixed for. snap_to_channel then measured
    distance from the wrong origin and could miss a channel that was really
    right next to the seed (see pipeline/compare_dem_sources.py's Kern River
    subset test, where this made every seed miss)."""
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / 'acc.tif'
    n, res = 20, 1 / 3600.0
    # Row index encodes latitude so a header shift away from the true pixel
    # origin is visible in the values, not just the array shape.
    data = np.repeat(np.arange(n, dtype=np.float32)[:, None], n, axis=1)
    with rasterio.open(
        path, 'w', driver='GTiff', height=n, width=n, count=1, dtype='float32',
        crs='EPSG:4326', transform=from_origin(-121.0, 39.0, res, res),
    ) as dst:
        dst.write(data, 1)

    # Request a window padded 0.15° past every edge of a 20-pixel (~0.0056°)
    # raster — exactly the WINDOW_PAD_DEG situation for a seed near a small
    # region's bbox edge.
    arr, transform, res_out = _read_window(str(path), -121.15, 38.85, -120.85, 39.15)
    assert arr.shape == (n, n)
    np.testing.assert_array_equal(arr, data)
    # The returned transform must describe these same pixels: row 0's value
    # (0.0) sits at the raster's true top, not 0.15° further north.
    assert abs(transform.f - 39.0) < 1e-9


def test_walk_specs_expand_pins_from_hobby_csv(tmp_path):
    csv = tmp_path / 'hobby.csv'
    csv.write_text(
        'report_id,name,creek_name,lon,lat\n'
        'P1,Crossing,Big River,-121.0,39.3\n'
    )
    cfg = {
        'data': {'hobby_reports_csv': str(csv)},
        'task11': {'geometry_walks': {'walks': [
            {'id': 'big', 'name': 'Big River', 'report_ids': ['P1', 'MISSING'],
             'min_snap_km2': 300, 'stop_above_km2': 100},
        ]}},
    }
    specs = walk_specs(cfg)
    assert len(specs) == 1
    assert specs[0]['walk_id'] == 'big:P1'
    assert specs[0]['min_snap_km2'] == 300.0
    assert specs[0]['stop_above_km2'] == 100.0
