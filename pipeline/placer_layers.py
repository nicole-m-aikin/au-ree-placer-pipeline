"""Mined-out placer ground and the Eocene gravel of the Sierra.

Used by county_rank. Both are screens on a pin, not new creek traces.

Mined ground (score is cut in half when a pin sits inside)
  USGS hydraulic mine pits, Orlando 2016, ScienceBase 571976c2e4b071321fe22947.
  USMIN California polygons whose feature type is a placer working:
  dredge tailings, hydraulic mine, placer mine, placer tailings, diggings.
  Quarries, gravel pits, and undifferentiated tailings are not included.

Eocene gravel (raises a pin where the modern creek cuts it)
  State Geologic Map Compilation unit EOc1: Jennings' "Eocene nonmarine
  rocks, unit 1 (Northern and Central California)". At 1:750,000 that unit
  is the auriferous gravel and the Ione Formation. It is not a Lindgren
  channel centerline. Distance is in kilometres from the pin to the unit.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import requests

from pipeline.ml_artifacts import REPO_ROOT

RAW = REPO_ROOT / 'data' / 'placer_layers' / 'raw'
MINED_PATH = REPO_ROOT / 'data' / 'placer_layers' / 'placer_mined.geojson'
CHANNEL_PATH = REPO_ROOT / 'data' / 'placer_layers' / 'eocene_gravel.geojson'

HYDRAULIC_URL = (
    'https://www.sciencebase.gov/catalog/file/get/571976c2e4b071321fe22947'
)
USMIN_URL = 'https://mrdata.usgs.gov/usmin/state/usmin-CA.zip'
SGMC_QUERY = (
    'https://services.arcgis.com/v01gqwM5QqNysAAi/arcgis/rest/services/'
    'SB_5888bf4fe4b05ccb964bab9d_USGS_SGMC_feature/FeatureServer/3/query'
)
EOCENE_WHERE = "STATE='CA' AND SGMC_LABEL LIKE 'EOc1%'"

# USMIN feature types that are placer ground already worked, not hard-rock
# dumps or aggregate pits.
MINED_FTR = frozenset({
    'Tailings - Dredge',
    'Hydraulic Mine',
    'Placer Mine',
    'Tailings - Placer',
    'Diggings',
})


def _download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(tmp, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.replace(dest)
    return dest


def _unzip(zpath, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(dest_dir)


def _build_mined():
    import geopandas as gpd

    pit_zip = RAW / 'HydraulicMinePi.zip'
    usmin_zip = RAW / 'usmin-CA.zip'
    if not pit_zip.exists():
        print(f'  Downloading hydraulic mine pits → {pit_zip}', flush=True)
        _download(HYDRAULIC_URL, pit_zip)
    if not usmin_zip.exists():
        print(f'  Downloading USMIN California → {usmin_zip}', flush=True)
        _download(USMIN_URL, usmin_zip)
    _unzip(pit_zip, RAW)
    _unzip(usmin_zip, RAW)
    pit_shp = next(RAW.rglob('Hydraulic_Mine_Pits_of_California.shp'))
    pits = gpd.read_file(pit_shp).to_crs('EPSG:4326')
    pits = pits.rename(columns={'Pit_Name': 'name'})
    pits['kind'] = 'hydraulic pit'
    pits['name'] = pits['name'].fillna('hydraulic pit').astype(str)
    poly = gpd.read_file(RAW / 'CA-poly.shp').to_crs('EPSG:4326')
    poly = poly[poly['FTR_TYPE'].isin(MINED_FTR)].copy()
    poly['kind'] = poly['FTR_TYPE'].astype(str)
    poly['name'] = poly['FTR_NAME'].fillna(poly['kind']).astype(str)
    mined = gpd.GeoDataFrame(
        {
            'kind': np.concatenate([pits['kind'].to_numpy(), poly['kind'].to_numpy()]),
            'name': np.concatenate([pits['name'].to_numpy(), poly['name'].to_numpy()]),
        },
        geometry=list(pits.geometry) + list(poly.geometry),
        crs='EPSG:4326',
    )
    mined['geometry'] = mined.geometry.simplify(0.00015)
    MINED_PATH.parent.mkdir(parents=True, exist_ok=True)
    mined[['kind', 'name', 'geometry']].to_file(MINED_PATH, driver='GeoJSON')
    print(f'  Wrote {len(mined)} mined-ground polygons → {MINED_PATH}', flush=True)
    return mined


def _build_channels():
    import geopandas as gpd

    print('  Downloading SGMC Eocene nonmarine (EOc1) …', flush=True)
    resp = requests.get(SGMC_QUERY, params={
        'f': 'geojson',
        'where': EOCENE_WHERE,
        'outFields': 'SGMC_LABEL,UNIT_NAME',
        'returnGeometry': 'true',
        'outSR': '4326',
    }, timeout=180)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get('error'):
        raise RuntimeError(payload['error'])
    gdf = gpd.GeoDataFrame.from_features(payload.get('features') or [], crs='EPSG:4326')
    if gdf.empty:
        raise RuntimeError('SGMC returned no EOc1 polygons')
    gdf = gdf.rename(columns={'UNIT_NAME': 'name', 'SGMC_LABEL': 'label'})
    keep = [c for c in ('label', 'name', 'geometry') if c in gdf.columns]
    CHANNEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    gdf[keep].to_file(CHANNEL_PATH, driver='GeoJSON')
    print(f'  Wrote {len(gdf)} Eocene gravel polygons → {CHANNEL_PATH}', flush=True)
    return gdf


def ensure_placer_layers(force=False):
    """Build the cached geojsons. Returns (mined_path, channel_path)."""
    if force or not MINED_PATH.exists() or MINED_PATH.stat().st_size < 1000:
        _build_mined()
    if force or not CHANNEL_PATH.exists() or CHANNEL_PATH.stat().st_size < 1000:
        _build_channels()
    return str(MINED_PATH), str(CHANNEL_PATH)


def _load(path):
    import geopandas as gpd

    if not path.exists() or path.stat().st_size < 100:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    return gpd.read_file(path)


def pin_flags(points):
    """on_mined, mined_name, channel_km for each point. Empty layers → no flags.

    channel_km is NaN when the gravel layer is missing, and 0 when the pin
    sits on the unit. on_mined is False when the mined layer is missing so a
    failed download does not invent a penalty; the caller treats an empty
    mined file as "not checked" via mined_checked.
    """
    import geopandas as gpd

    from pipeline.geo_crs import utm_epsg

    n = len(points)
    on_mined = np.zeros(n, dtype=bool)
    names = np.array([''] * n, dtype=object)
    channel_km = np.full(n, np.nan)
    mined_checked = False
    channel_checked = False
    if n == 0:
        return _flag_frame(on_mined, names, channel_km, False, False)
    pts = gpd.GeoDataFrame(
        {'_i': np.arange(n)}, geometry=points.geometry.values,
        crs=points.crs or 'EPSG:4326',
    )
    mined = _load(MINED_PATH)
    if len(mined):
        mined_checked = True
        joined = gpd.sjoin(
            pts, mined[['name', 'geometry']], how='inner', predicate='intersects',
        )
        for i, grp in joined.groupby('_i'):
            on_mined[int(i)] = True
            nm = grp['name'].dropna().astype(str)
            names[int(i)] = str(nm.iloc[0]) if len(nm) else 'mined ground'
    channels = _load(CHANNEL_PATH)
    if len(channels):
        channel_checked = True
        lat0 = float(pts.geometry.y.mean())
        lon0 = float(pts.geometry.x.mean())
        epsg = utm_epsg(lon0, lat0)
        union = channels.to_crs(epsg).geometry.union_all()
        dist_m = pts.to_crs(epsg).geometry.distance(union)
        channel_km = np.round(dist_m.to_numpy() / 1000.0, 2)
    return _flag_frame(on_mined, names, channel_km, mined_checked, channel_checked)


def _flag_frame(on_mined, names, channel_km, mined_checked, channel_checked):
    import pandas as pd

    return pd.DataFrame({
        'on_mined': on_mined,
        'mined_name': names,
        'channel_km': channel_km,
        'mined_checked': mined_checked,
        'channel_checked': channel_checked,
    })
