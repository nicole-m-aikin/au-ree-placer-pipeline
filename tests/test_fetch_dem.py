"""DEM clip keeps pixels and header aligned when the pad runs past the mosaic."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.fetch_dem import _clip_dem


def _write_tile(path, west, north, n=360, res=1 / 360):
    import rasterio
    from rasterio.transform import from_origin

    # Row index encodes latitude so a header shift is visible in the values.
    data = np.repeat(np.arange(n, dtype=np.float32)[:, None], n, axis=1)
    with rasterio.open(
        path, 'w', driver='GTiff', height=n, width=n, count=1, dtype='float32',
        crs='EPSG:4326', transform=from_origin(west, north, res, res),
    ) as dst:
        dst.write(data, 1)
    return data


def test_clip_at_mosaic_top_does_not_shift_header(tmp_path):
    import rasterio

    path = tmp_path / 'tile.tif'
    _write_tile(path, west=-121.0, north=40.0)
    # bbox north sits on the tile edge; the 0.02° pad lands outside it.
    _clip_dem(str(path), -120.9, 39.5, -120.1, 40.0, pad=0.02)
    with rasterio.open(path) as src:
        assert abs(src.bounds.top - 40.0) < 1e-9
        # The pixel at 39.9° must still be the tile's row for 39.9°.
        row = int((src.transform.f - 39.9) / abs(src.transform.e))
        assert src.read(1)[row, 0] == 36.0


def test_clip_inside_mosaic_keeps_pad(tmp_path):
    import rasterio

    path = tmp_path / 'tile.tif'
    _write_tile(path, west=-121.0, north=40.0)
    _clip_dem(str(path), -120.9, 39.5, -120.1, 39.8, pad=0.02)
    with rasterio.open(path) as src:
        px = abs(src.transform.e)
        assert abs(src.bounds.top - 39.82) <= px
        assert abs(src.bounds.left - (-120.92)) <= px
        row = int(round((src.transform.f - 39.7) / px))
        assert src.read(1)[row, 0] == 108.0
