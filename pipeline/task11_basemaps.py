"""Task 11 basemaps: SGMC geology + 3DEP 1 m LiDAR clips for walkable catchments.

LiDAR is clipped to a 2 km buffer around each pour (the walk, not the
whole 0.4° cell). Newest 1 m project wins when tiles overlap. Geology is
the USGS SGMC Washington extract, clipped to the study map.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import zipfile
import warnings

import numpy as np
import pandas as pd

from pipeline.utils import map_extent, resolve_data_path

SGMC_WA_URL = 'https://mrdata.usgs.gov/geology/state/shp/WA.zip'
TNM_PRODUCTS = 'https://tnmaccess.nationalmap.gov/api/v1/products'
USER_AGENT = 'AuREE-pipeline/1.0 (research; 3DEP/SGMC clip)'
LIDAR_RANKS = 10
POUR_BUFFER_M = 2000

SGMC_LITH_MAP = {
    'Metamorphic, gneiss': 'MCC_metapelite',
    'Metamorphic, schist': 'MCC_metapelite',
    'Metamorphic, amphibolite': 'MCC_metapelite',
    'Metamorphic, sedimentary': 'MCC_metapelite',
    'Metamorphic, sedimentary clastic': 'MCC_metapelite',
    'Metamorphic and Sedimentary, undifferentiated': 'MCC_metapelite',
    'Igneous, intrusive': 'felsic_intrusive',
    'Igneous, volcanic': 'sedimentary_cover',
    'Sedimentary, clastic': 'sedimentary_cover',
    'Sedimentary, carbonate': 'sedimentary_cover',
    'Metamorphic, serpentinite': 'mafic_ultramafic',
    'Metamorphic, volcanic': 'mafic_ultramafic',
}


def _sgmc_dir(repo_root=None):
    root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, 'ne_wa_ree', 'data', 'geologic', 'WA_sgmc_extracted')


def _lidar_dir(repo_root=None):
    root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, 'ne_wa_ree', 'data', 'lidar')


def _download(url, dest):
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or '.', exist_ok=True)
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, 'wb') as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest


def ensure_sgmc_wa(repo_root=None, force=False):
    """Download and unzip the 5.4 MB USGS SGMC Washington extract if needed."""
    dest_dir = _sgmc_dir(repo_root)
    shp = os.path.join(dest_dir, 'WA_geol_poly.shp')
    if os.path.exists(shp) and not force:
        return dest_dir
    os.makedirs(dest_dir, exist_ok=True)
    zpath = os.path.join(os.path.dirname(dest_dir), 'WA.zip')
    if force or not os.path.exists(zpath):
        print(f"  Downloading SGMC Washington geology ({SGMC_WA_URL})...")
        _download(SGMC_WA_URL, zpath)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(dest_dir)
    if not os.path.exists(shp):
        raise FileNotFoundError(f'SGMC extract missing {shp}')
    print(f"  SGMC Washington geology ready at {dest_dir}")
    return dest_dir


def lith_type_from_generalize(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'sedimentary_cover'
    return SGMC_LITH_MAP.get(str(value), 'sedimentary_cover')


def load_geology(cfg, dest_dir=None):
    """SGMC polygons clipped to the map, with unit name/age and lith_type."""
    import geopandas as gpd
    from shapely.geometry import box

    dest_dir = dest_dir or ensure_sgmc_wa()
    gdf = gpd.read_file(os.path.join(dest_dir, 'WA_geol_poly.shp'))
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    else:
        gdf = gdf.to_crs('EPSG:4326')
    xmin, xmax, ymin, ymax = map_extent(cfg)
    clip = box(xmin, ymin, xmax, ymax)
    gdf = gdf[gdf.intersects(clip)].copy()
    gdf['geometry'] = gdf.geometry.intersection(clip)
    gdf = gdf[~gdf.geometry.is_empty]
    gdf['lith_type'] = gdf['GENERALIZE'].map(lith_type_from_generalize)
    units_csv = os.path.join(dest_dir, 'WA_units.csv')
    if os.path.exists(units_csv):
        units = pd.read_csv(units_csv)
        units.columns = [c.strip().lower() for c in units.columns]
        keep = [c for c in ('unit_link', 'unit_name', 'unit_age', 'rocktype1')
                if c in units.columns]
        if 'unit_link' in keep:
            units = units[keep].drop_duplicates('unit_link')
            gdf = gdf.merge(
                units, left_on='UNIT_LINK', right_on='unit_link', how='left',
            )
    rename = {
        'ORIG_LABEL': 'orig_label',
        'UNIT_LINK': 'unit_link_src',
        'GENERALIZE': 'generalize',
        'unit_name': 'unit_name',
        'unit_age': 'unit_age',
        'rocktype1': 'rocktype1',
        'lith_type': 'lith_type',
    }
    out = gdf[[c for c in rename if c in gdf.columns] + ['geometry']].rename(
        columns=rename
    )
    return out


def load_structures(cfg, dest_dir=None):
    """SGMC structure lines clipped to the map."""
    import geopandas as gpd
    from shapely.geometry import box

    dest_dir = dest_dir or ensure_sgmc_wa()
    path = os.path.join(dest_dir, 'WA_structure.shp')
    if not os.path.exists(path):
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    else:
        gdf = gdf.to_crs('EPSG:4326')
    xmin, xmax, ymin, ymax = map_extent(cfg)
    clip = box(xmin, ymin, xmax, ymax)
    gdf = gdf[gdf.intersects(clip)].copy()
    gdf['geometry'] = gdf.geometry.intersection(clip)
    gdf = gdf[~gdf.geometry.is_empty]
    keep = [c for c in ('DESCRIPT', 'SYMBOL', 'GEOM') if c in gdf.columns]
    out = gdf[keep + ['geometry']].rename(columns={
        'DESCRIPT': 'descript', 'SYMBOL': 'symbol', 'GEOM': 'struct_type',
    })
    return out


def _tile_score(title):
    t = title or ''
    if 'NorthEast_B22' in t:
        return 50
    if 'B22' in t or 'D22' in t:
        return 40
    if 'B21' in t or '2021' in t:
        return 30
    if 'D20' in t or '2019' in t:
        return 20
    return 10


def _tile_key(title):
    m = re.search(r'(\d+)\s+x(\d+y\d+)', title or '')
    return m.group(0) if m else (title or '')


def tnm_1m_tiles(west, south, east, north, geom=None):
    """Newest 3DEP 1 m GeoTIFF tiles intersecting a bbox (and optional polygon)."""
    from shapely.geometry import box

    bbox = f'{west},{south},{east},{north}'
    url = (
        f'{TNM_PRODUCTS}?bbox={bbox}'
        '&datasets=Digital%20Elevation%20Model%20(DEM)%201%20meter'
        '&prodFormats=GeoTIFF&max=50'
    )
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=90) as resp:
        items = json.load(resp).get('items') or []
    grouped = {}
    for it in items:
        bb = it.get('boundingBox') or {}
        if not bb:
            continue
        tb = box(bb['minX'], bb['minY'], bb['maxX'], bb['maxY'])
        if geom is not None and not tb.intersects(geom):
            continue
        title = it.get('title') or ''
        href = (it.get('urls') or {}).get('TIFF') or it.get('downloadURL')
        if not href:
            continue
        key = _tile_key(title)
        rec = {
            'score': _tile_score(title),
            'title': title,
            'url': href,
            'bytes': int(it.get('sizeInBytes') or 0),
        }
        grouped.setdefault(key, []).append(rec)
    # Newest project first; callers can fall back if that mosaic is empty.
    out = []
    for recs in grouped.values():
        recs.sort(key=lambda r: r['score'], reverse=True)
        out.extend(recs)
    return out


def hillshade(elev, resolution=1.0, azimuth=315.0, altitude=45.0):
    """Simple 8-bit hillshade from a DEM array. Nodata stays 0."""
    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    dy, dx = np.gradient(np.asarray(elev, dtype='float64'), resolution)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    hs = (
        np.sin(alt) * np.sin(slope)
        + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    )
    hs = np.clip(hs, 0, 1)
    out = (hs * 255).astype('uint8')
    return out


def _pour_buffer(lon, lat, meters=POUR_BUFFER_M):
    import geopandas as gpd
    from shapely.geometry import Point

    pt = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs='EPSG:4326')
    utm = pt.estimate_utm_crs()
    return pt.to_crs(utm).buffer(meters).to_crs('EPSG:4326').iloc[0]


def fetch_lidar_clips(catchments, dest_dir=None, max_rank=LIDAR_RANKS,
                      buffer_m=POUR_BUFFER_M):
    """Download 1 m tiles and clip a hillshade+DEM around each top pour."""
    import geopandas as gpd
    import rasterio
    from rasterio.merge import merge
    from rasterio.mask import mask
    from shapely.geometry import mapping

    dest_dir = dest_dir or _lidar_dir()
    os.makedirs(dest_dir, exist_ok=True)
    tile_dir = os.path.join(dest_dir, 'tiles')
    os.makedirs(tile_dir, exist_ok=True)

    if catchments is None or len(catchments) == 0:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')

    work = catchments.copy()
    if 'walk_rank' in work.columns:
        work = work[work['walk_rank'].notna() & (work['walk_rank'] <= max_rank)]
    rows = []
    for _, rec in work.sort_values('walk_rank').iterrows():
        rank = int(rec['walk_rank'])
        lon = rec.get('pour_lon', rec.geometry.centroid.x)
        lat = rec.get('pour_lat', rec.geometry.centroid.y)
        if pd.isna(lon) or pd.isna(lat):
            lon, lat = rec.geometry.centroid.x, rec.geometry.centroid.y
        clip_geom = _pour_buffer(float(lon), float(lat), buffer_m)
        if rec.geometry is not None:
            clip_geom = clip_geom.intersection(rec.geometry.buffer(0.01))
            if clip_geom.is_empty:
                clip_geom = _pour_buffer(float(lon), float(lat), buffer_m)
        dest_dem = os.path.join(dest_dir, f'lidar_r{rank:02d}_dem.tif')
        dest_hs = os.path.join(dest_dir, f'lidar_r{rank:02d}_hs.tif')
        if os.path.exists(dest_hs) and os.path.getsize(dest_hs) > 1000:
            rows.append({
                'walk_rank': rank,
                'block_id': rec.get('block_id'),
                'source': 'cached_1m_clip',
                'dem_path': dest_dem,
                'hs_path': dest_hs,
                'geometry': clip_geom,
            })
            continue
        minx, miny, maxx, maxy = clip_geom.bounds
        try:
            tiles = tnm_1m_tiles(minx, miny, maxx, maxy, geom=clip_geom)
        except Exception as e:
            warnings.warn(f'  3DEP query failed for #{rank}: {e}')
            continue
        if not tiles:
            print(f"  No 1 m LiDAR for walk #{rank}")
            continue
        elev = None
        ctf = None
        crs = None
        source_title = tiles[0]['title']
        scores = sorted({t.get('score', 0) for t in tiles}, reverse=True) or [0]
        for min_score in scores:
            subset = [t for t in tiles if t.get('score', 0) >= min_score]
            # On later passes, drop the empty newer projects.
            if min_score != scores[0]:
                subset = [t for t in tiles if t.get('score', 0) <= min_score]
            local = []
            for tile in subset:
                name = tile['url'].split('/')[-1]
                dest = os.path.join(tile_dir, name)
                if not os.path.exists(dest) or os.path.getsize(dest) < 1000:
                    print(f"  Downloading {name} ({tile['bytes']/1e6:.0f} MB) for #{rank}...")
                    try:
                        _download(tile['url'], dest)
                    except Exception as e:
                        warnings.warn(f'  tile download failed {name}: {e}')
                        continue
                local.append(dest)
            if not local:
                continue
            datasets = [rasterio.open(p) for p in local]
            try:
                mosaic, transform = merge(datasets)
                crs = datasets[0].crs
                nodata = datasets[0].nodata if datasets[0].nodata is not None else -9999
                meta = datasets[0].meta.copy()
                source_title = subset[0]['title']
            finally:
                for ds in datasets:
                    ds.close()
            meta.update(
                driver='GTiff', height=mosaic.shape[1], width=mosaic.shape[2],
                transform=transform, count=1, compress='deflate', nodata=nodata,
            )
            clip_src = (
                gpd.GeoDataFrame(geometry=[clip_geom], crs='EPSG:4326')
                .to_crs(crs).iloc[0].geometry
            )
            with rasterio.io.MemoryFile() as mem:
                with mem.open(**meta) as src:
                    src.write(mosaic[0], 1)
                    clipped, ctf = mask(
                        src, [mapping(clip_src)], crop=True, nodata=nodata,
                    )
            elev = clipped[0].astype('float32')
            for nd in (nodata, -999999.0, -9999.0):
                if nd is None:
                    continue
                elev = np.where(elev == nd, np.nan, elev)
            if np.isfinite(elev).sum() >= 500:
                break
            elev = None
        if elev is None:
            print(f"  1 m LiDAR for #{rank} is nodata at the pour — skipped")
            continue
        dest_dem = os.path.join(dest_dir, f'lidar_r{rank:02d}_dem.tif')
        dest_hs = os.path.join(dest_dir, f'lidar_r{rank:02d}_hs.tif')
        hs = hillshade(np.where(np.isnan(elev), 0, elev), resolution=1.0)
        hs = np.where(np.isnan(elev), 0, hs).astype('uint8')
        profile = {
            'driver': 'GTiff', 'height': elev.shape[0], 'width': elev.shape[1],
            'count': 1, 'crs': crs, 'transform': ctf, 'compress': 'deflate',
            'tiled': True,
        }
        dem_prof = dict(profile, dtype='float32', nodata=-9999.0)
        hs_prof = dict(profile, dtype='uint8', nodata=0)
        with rasterio.open(dest_dem, 'w', **dem_prof) as dst_ds:
            dst_ds.write(np.where(np.isnan(elev), -9999.0, elev), 1)
        with rasterio.open(dest_hs, 'w', **hs_prof) as hs_ds:
            hs_ds.write(hs, 1)
        print(f"  LiDAR #{rank}: {dest_hs}")
        rows.append({
            'walk_rank': rank,
            'block_id': rec.get('block_id'),
            'source': source_title,
            'dem_path': dest_dem,
            'hs_path': dest_hs,
            'geometry': clip_geom,
        })
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    return gpd.GeoDataFrame(rows, crs='EPSG:4326')


def append_lidar_rasters(gpkg_path, index):
    """Append hillshade rasters as GPKG tables lidar_01 .. lidar_N."""
    if index is None or len(index) == 0:
        return []
    from rasterio.shutil import copy as rio_copy

    written = []
    for _, rec in index.iterrows():
        src = rec['hs_path']
        if not src or not os.path.exists(src):
            continue
        table = f"lidar_{int(rec['walk_rank']):02d}"
        try:
            rio_copy(
                src, gpkg_path, driver='GPKG',
                raster_table=table, append_subdataset=True,
            )
            written.append(table)
        except Exception as e:
            warnings.warn(f'  could not embed {table} in GPKG ({e}); TIFF remains at {src}')
    return written


def add_basemaps_to_gpkg(cfg, gpkg_path, catchments=None, fetch_lidar=True):
    """Write geology + structures into the GPKG; optionally embed LiDAR hillshades."""
    import pyogrio

    dest_dir = ensure_sgmc_wa()
    geology = load_geology(cfg, dest_dir=dest_dir)
    structures = load_structures(cfg, dest_dir=dest_dir)
    added = []
    if len(geology):
        pyogrio.write_dataframe(
            geology, gpkg_path, layer='geology', driver='GPKG',
            geometry_type='Unknown', append=True,
            layer_options={'SPATIAL_INDEX': 'NO'},
        )
        added.append(('geology', int(len(geology))))
        print(f"    layer geology: {len(geology)} features")
    if len(structures):
        pyogrio.write_dataframe(
            structures, gpkg_path, layer='geology_structure', driver='GPKG',
            geometry_type='Unknown', append=True,
            layer_options={'SPATIAL_INDEX': 'NO'},
        )
        added.append(('geology_structure', int(len(structures))))
        print(f"    layer geology_structure: {len(structures)} features")

    index = None
    if fetch_lidar:
        index = fetch_lidar_clips(catchments)
        if index is not None and len(index):
            slim = index.drop(columns=[], errors='ignore').copy()
            # Keep portable relative paths next to the GPKG.
            slim['dem_path'] = slim['dem_path'].map(os.path.basename)
            slim['hs_path'] = slim['hs_path'].map(os.path.basename)
            pyogrio.write_dataframe(
                slim, gpkg_path, layer='lidar_index', driver='GPKG',
                geometry_type='Polygon', append=True,
                layer_options={'SPATIAL_INDEX': 'NO'},
            )
            added.append(('lidar_index', int(len(slim))))
            print(f"    layer lidar_index: {len(slim)} features")
            rasters = append_lidar_rasters(gpkg_path, index)
            for name in rasters:
                added.append((name, 1))
                print(f"    raster {name}")
    from pipeline.task11_qgis_styles import embed_qgis_styles
    embed_qgis_styles(gpkg_path)
    return added
