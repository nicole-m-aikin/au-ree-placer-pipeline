"""Tests for NURE aerial gamma-ray fetch/clip and the Task 1 eTh overlay."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.fetch_radiometric import (
    clip_bounds_from_cfg, clip_geotiff, dest_path, reproject_clip_to_geo, DNAG_TM,
)
from pipeline.task1_coplacer import attach_radiometrics, _plot_eth_overlay
from pipeline.utils import RAD_AGREE_BOTH, RAD_AGREE_NONE


def _write_tiny_tif(path, origin_x, origin_y, px, values, nodata=None):
    import rasterio
    from rasterio.transform import from_origin

    arr = np.asarray(values, dtype=np.float32)
    profile = {
        'driver': 'GTiff',
        'height': arr.shape[0],
        'width': arr.shape[1],
        'count': 1,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': from_origin(origin_x, origin_y, px, px),
    }
    if nodata is not None:
        profile['nodata'] = nodata
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr, 1)


def _study_cfg(tmp, **data_extra):
    return {
        'study_area': {
            'name': 'Test Area',
            'short': 'test',
            'bbox': {
                'lon_min': -120.0, 'lon_max': -117.0,
                'lat_min': 47.5, 'lat_max': 49.1,
            },
            'map_padding': 0.08,
        },
        'outputs_dir': str(tmp / 'out'),
        'data': dict(data_extra),
        'sites': [],
    }


class TestClipBounds:
    def test_default_pad_matches_data_sources_ne_wa(self, tmp_path):
        cfg = _study_cfg(tmp_path)
        xmin, xmax, ymin, ymax = clip_bounds_from_cfg(cfg)
        # DATA_SOURCES.md: lon -120.2→-116.8, lat 47.3→49.3 with pad 0.2
        # (DATA_SOURCES quotes 49.2; 49.1+0.2 is the consistent pad)
        assert xmin == pytest.approx(-120.2)
        assert xmax == pytest.approx(-116.8)
        assert ymin == pytest.approx(47.3)
        assert ymax == pytest.approx(49.3)

    def test_custom_pad(self, tmp_path):
        cfg = _study_cfg(tmp_path)
        b = clip_bounds_from_cfg(cfg, pad=0.1)
        assert b[0] == pytest.approx(-120.1)
        assert b[1] == pytest.approx(-116.9)


class TestClipGeotiff:
    def test_writes_smaller_window(self, tmp_path):
        src = tmp_path / 'full.tif'
        dst = tmp_path / 'clip.tif'
        grid = np.arange(100, dtype=np.float32).reshape(10, 10)
        _write_tiny_tif(src, -120.0, 49.2, 0.2, grid)
        h, w = clip_geotiff(str(src), dst, (-119.4, -118.4, 47.8, 48.6))
        assert dst.exists()
        assert h * w < 100
        import rasterio
        with rasterio.open(dst) as ds:
            assert ds.bounds.left >= -119.4 - 0.2
            assert ds.bounds.right <= -118.4 + 0.2

    def test_empty_window_raises(self, tmp_path):
        src = tmp_path / 'full.tif'
        _write_tiny_tif(src, -120.0, 49.2, 0.2, np.ones((4, 4)))
        with pytest.raises(ValueError, match='empty'):
            clip_geotiff(str(src), tmp_path / 'x.tif', (0, 1, 0, 1))


class TestReprojectClip:
    def test_geographic_output_covers_requested_bbox(self, tmp_path):
        # Tiny DNAG-metre grid around a known NE WA point.
        from pyproj import Transformer
        to_dnag = Transformer.from_crs("EPSG:4326", DNAG_TM, always_xy=True)
        cx, cy = to_dnag.transform(-118.5, 48.5)
        # 5×5 @ 2000 m, origin NW
        grid = np.full((5, 5), 8.0, dtype=np.float32)
        grid[2, 2] = 20.0
        src = tmp_path / 'src.tif'
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.crs import CRS
        with rasterio.open(src, 'w', driver='GTiff', height=5, width=5, count=1,
                           dtype='float32', crs=CRS.from_proj4(DNAG_TM),
                           transform=from_origin(cx - 5000, cy + 5000, 2000, 2000),
                           nodata=-9999.0) as dst:
            dst.write(grid, 1)
        out = tmp_path / 'clip.tif'
        h, w = reproject_clip_to_geo(src, out, (-118.7, -118.3, 48.3, 48.7), cell_m=2000)
        assert out.exists()
        assert h > 1 and w > 1
        with rasterio.open(out) as ds:
            assert ds.crs.to_epsg() == 4326
            assert ds.bounds.left <= -118.7 + 0.05
            assert ds.bounds.right >= -118.3 - 0.05

    def test_crs_less_source_still_reprojects(self, tmp_path):
        # Esri FLT grids ship without a dataset CRS; warp must use src_crs=.
        from pyproj import Transformer
        import rasterio
        from rasterio.transform import from_origin
        to_dnag = Transformer.from_crs("EPSG:4326", DNAG_TM, always_xy=True)
        cx, cy = to_dnag.transform(-118.5, 48.5)
        grid = np.full((5, 5), 8.0, dtype=np.float32)
        grid[2, 2] = 20.0
        src = tmp_path / 'nocrs.tif'
        with rasterio.open(src, 'w', driver='GTiff', height=5, width=5, count=1,
                           dtype='float32', crs=None,
                           transform=from_origin(cx - 5000, cy + 5000, 2000, 2000),
                           nodata=-9999.0) as dst:
            dst.write(grid, 1)
        out = tmp_path / 'clip.tif'
        reproject_clip_to_geo(src, out, (-118.7, -118.3, 48.3, 48.7), cell_m=2000)
        with rasterio.open(out) as ds:
            arr = ds.read(1)
            arr = np.where(arr == ds.nodata, np.nan, arr)
            assert np.isfinite(arr).sum() > 0


class TestDestPath:
    def test_uses_configured_path(self, tmp_path):
        cfg = _study_cfg(tmp_path, radiometric_eth_tif=str(tmp_path / 'ne_wa_eTh.tif'))
        assert dest_path(cfg, 'eth') == tmp_path / 'ne_wa_eTh.tif'


class TestAttachRadiometrics:
    def _nure_sites(self):
        import geopandas as gpd
        from shapely.geometry import Point
        nure = gpd.GeoDataFrame({
            'lab_id': ['A', 'B', 'C', 'D'],
            'lon': [-119.5, -118.5, -119.5, -118.5],
            'lat': [48.5, 48.5, 47.7, 47.7],
            'Th': [8.0, 40.0, 9.0, 12.0],
            'th_anomaly': [False, True, False, False],
            'th_source': ['BACKGROUND', 'MONAZITE', 'BACKGROUND', 'BACKGROUND'],
        }, geometry=[Point(-119.5, 48.5), Point(-118.5, 48.5),
                     Point(-119.5, 47.7), Point(-118.5, 47.7)],
        crs='EPSG:4326')
        sites = gpd.GeoDataFrame({
            'name': ['Test Placer'],
            'lon': [-118.5],
            'lat': [48.5],
            'th_near': [True],
        }, geometry=[Point(-118.5, 48.5)], crs='EPSG:4326')
        return nure, sites

    def test_noop_without_tiff(self, tmp_path):
        cfg = _study_cfg(tmp_path)
        nure, sites = self._nure_sites()
        s2, n2, thresh = attach_radiometrics(cfg, sites, nure)
        assert thresh is None
        assert 'rad_eTh' not in s2.columns

    def test_samples_and_classifies(self, tmp_path):
        # 5×5 of background 8 ppm, one 50 ppm pixel whose center is (-118.6, 48.5)
        # so mean+2SD is well below 50 and the Test Placer / NURE-B point hit it.
        grid = np.full((5, 5), 8.0, dtype=np.float32)
        grid[2, 3] = 50.0
        tif = tmp_path / 'eth.tif'
        _write_tiny_tif(tif, -120.0, 49.5, 0.4, grid)
        cfg = _study_cfg(tmp_path, radiometric_eth_tif=str(tif))
        nure, sites = self._nure_sites()
        sites, nure, thresh = attach_radiometrics(cfg, sites, nure)
        assert thresh is not None
        assert thresh < 50.0
        assert sites['rad_eTh'].iloc[0] == pytest.approx(50.0)
        assert bool(sites['rad_eTh_high'].iloc[0]) is True
        assert sites['rad_th_agree'].iloc[0] == RAD_AGREE_BOTH
        b = nure.set_index('lab_id').loc['B']
        assert b['rad_th_agree'] == RAD_AGREE_BOTH
        a = nure.set_index('lab_id').loc['A']
        assert a['rad_th_agree'] == RAD_AGREE_NONE


class TestEthOverlayNoop:
    def test_plot_skips_without_tiff(self, tmp_path):
        cfg = _study_cfg(tmp_path)
        import geopandas as gpd
        from shapely.geometry import Point
        empty = gpd.GeoDataFrame(
            {'lon': [], 'lat': [], 'name': [], 'th_anomaly': []},
            geometry=[], crs='EPSG:4326')
        _plot_eth_overlay(cfg, empty, empty, eth_thresh=12.0)
        fig = tmp_path / 'out' / 'figures' / 'fig1b_airborne_eth_vs_nure_th.png'
        assert not fig.exists()
