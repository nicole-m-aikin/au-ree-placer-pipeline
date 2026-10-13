"""California geologic units and faults for the placer map.

USGS State Geologic Map Compilation (Horton and others, 2017), California
only. Units are Jennings 1:750,000, colored by generalized lithology.
Faults are the structure lines whose description is a fault. Folds are
left out. Both layers are simplified for the Earth Engine app.
"""

from __future__ import annotations

import sys

import requests

from pipeline.ml_artifacts import REPO_ROOT

SGMC = (
    'https://services.arcgis.com/v01gqwM5QqNysAAi/arcgis/rest/services/'
    'SB_5888bf4fe4b05ccb964bab9d_USGS_SGMC_feature/FeatureServer'
)
OUT = REPO_ROOT / 'data' / 'placer_layers'
PROJECT = 'placer-picker-511303'
ASSET_ROOT = f'projects/{PROJECT}/assets/placer'

# Fill colors are RRGGBBAA so the units stay readable over the terrain.
LITH_FILL = {
    'Sedimentary, clastic': 'C4A57499',
    'Sedimentary, undifferentiated': 'D4B48399',
    'Sedimentary, carbonate': '7EB0C499',
    'Sedimentary, chemical': '9ECCC499',
    'Unconsolidated, undifferentiated': 'E4D5A866',
    'Igneous, volcanic': 'C45C4A99',
    'Igneous, intrusive': 'C47AA899',
    'Igneous and Metamorphic, undifferentiated': '8A6A8899',
    'Igneous and Sedimentary, undifferentiated': 'B0706099',
    'Metamorphic, undifferentiated': '6E7A6299',
    'Metamorphic, gneiss': '5E6A5499',
    'Metamorphic, schist': '4E5C4899',
    'Metamorphic, serpentinite': '2F6B4F99',
    'Metamorphic, volcanic': '8A5A4A99',
    'Metamorphic, sedimentary clastic': '8A745899',
    'Metamorphic, carbonate': '6A90A099',
    'Metamorphic, amphibolite': '3E5A4899',
    'Metamorphic, other': '5A645899',
    'Metamorphic and Sedimentary, undifferentiated': 'A0886899',
    'Melange': '3E465099',
}


def _pages(layer, where, fields, box=False):
    url = f'{SGMC}/{layer}/query'
    offset = 0
    feats = []
    while True:
        params = {
            'f': 'geojson',
            'where': where,
            'outFields': fields,
            'returnGeometry': 'true',
            'outSR': '4326',
            'geometryPrecision': 4,
            'resultRecordCount': 2000,
            'resultOffset': offset,
        }
        if box:
            params.update({
                'geometry': '-124.6,32.4,-114.0,42.1',
                'geometryType': 'esriGeometryEnvelope',
                'inSR': '4326',
                'spatialRel': 'esriSpatialRelIntersects',
            })
        resp = requests.get(url, params=params, timeout=180)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get('error'):
            raise RuntimeError(payload['error'])
        batch = payload.get('features') or []
        feats.extend(batch)
        print(f'  layer {layer}: {len(feats)}', flush=True)
        if len(batch) < 2000:
            break
        offset += len(batch)
    return feats


def _frame(features):
    import geopandas as gpd

    gdf = gpd.GeoDataFrame.from_features(features, crs='EPSG:4326')
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf['geometry'] = gdf.geometry.simplify(0.002)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf


def build_units():
    print('Downloading SGMC units …', flush=True)
    feats = _pages(3, "STATE='CA'", 'SGMC_LABEL,UNIT_NAME,GENERALIZED_LITH')
    gdf = _frame(feats)
    gdf = gdf[gdf['GENERALIZED_LITH'].fillna('') != 'Water'].copy()
    gdf['unit'] = gdf['UNIT_NAME'].fillna('').astype(str)
    gdf['label'] = gdf['SGMC_LABEL'].fillna('').astype(str)
    gdf['lith'] = gdf['GENERALIZED_LITH'].fillna('').astype(str)
    gdf['fill'] = gdf['lith'].map(LITH_FILL).fillna('B0A89899')
    keep = gdf[['unit', 'label', 'lith', 'fill', 'geometry']]
    path = OUT / 'sgmc_units.geojson'
    path.parent.mkdir(parents=True, exist_ok=True)
    keep.to_file(path, driver='GeoJSON')
    print(f'  {len(keep)} units → {path}', flush=True)
    return keep


def build_faults():
    print('Downloading SGMC faults …', flush=True)
    feats = _pages(1, "STATE='CA' AND DESCRIPTION LIKE '%ault%'", 'DESCRIPTION')
    gdf = _frame(feats)
    desc = gdf['DESCRIPTION'].fillna('Fault').astype(str)
    gdf['kind'] = desc.str.split(',').str[0].str.strip()
    keep = gdf[['kind', 'geometry']]
    path = OUT / 'sgmc_faults.geojson'
    keep.to_file(path, driver='GeoJSON')
    print(f'  {len(keep)} faults → {path}', flush=True)
    return keep


def build_contacts():
    print('Downloading SGMC contacts …', flush=True)
    feats = _pages(2, '1=1', 'Symbology', box=True)
    gdf = _frame(feats)
    gdf['kind'] = gdf['Symbology'].fillna('Contact').astype(str)
    keep = gdf[['kind', 'geometry']]
    path = OUT / 'sgmc_contacts.geojson'
    keep.to_file(path, driver='GeoJSON')
    print(f'  {len(keep)} contacts → {path}', flush=True)
    return keep


def build_bedding():
    print('Downloading SGMC bedding …', flush=True)
    feats = _pages(0, "STATE='CA'", 'DESCRIPTION')
    gdf = _frame(feats)
    desc = gdf['DESCRIPTION'].fillna('Bedding').astype(str)
    gdf['kind'] = desc.str.split(' - ').str[0].str.strip()
    keep = gdf[['kind', 'geometry']]
    path = OUT / 'sgmc_bedding.geojson'
    keep.to_file(path, driver='GeoJSON')
    print(f'  {len(keep)} bedding points → {path}', flush=True)
    return keep


def _publish(ee, asset_id):
    acl = ee.data.getAssetAcl(asset_id)
    acl['all_users_can_read'] = True
    ee.data.setAssetAcl(asset_id, acl)
    print(f'  public read {asset_id}', flush=True)


def upload(units, faults):
    from pipeline.gee_export import feature_dicts_from_gdf, upload_features
    import ee

    ee.Initialize(project=PROJECT)
    units_id = f'{ASSET_ROOT}/sgmc_units'
    faults_id = f'{ASSET_ROOT}/sgmc_faults'
    print(f'Uploading {len(units)} units …', flush=True)
    upload_features(
        feature_dicts_from_gdf(units), PROJECT, units_id, 'sgmc_units', chunk=300)
    _publish(ee, units_id)
    print(f'Uploading {len(faults)} faults …', flush=True)
    upload_features(
        feature_dicts_from_gdf(faults), PROJECT, faults_id, 'sgmc_faults', chunk=800)
    _publish(ee, faults_id)


def _upload_one(ee, gdf, name, chunk):
    from pipeline.gee_export import feature_dicts_from_gdf, upload_features

    asset_id = f'{ASSET_ROOT}/{name}'
    print(f'Uploading {len(gdf)} {name} …', flush=True)
    upload_features(
        feature_dicts_from_gdf(gdf), PROJECT, asset_id, name, chunk=chunk)
    _publish(ee, asset_id)


def upload_rest():
    """Contacts, bedding, mined ground, and the Eocene gravel. Units and faults stay."""
    import geopandas as gpd
    import ee

    from pipeline.placer_layers import CHANNEL_PATH, MINED_PATH

    ee.Initialize(project=PROJECT)
    bedding = build_bedding()
    _upload_one(ee, bedding, 'sgmc_bedding', chunk=500)
    mined = gpd.read_file(MINED_PATH)
    gravel = gpd.read_file(CHANNEL_PATH)
    _upload_one(ee, mined, 'placer_mined', chunk=400)
    _upload_one(ee, gravel, 'eocene_gravel', chunk=50)
    contacts = build_contacts()
    _upload_one(ee, contacts, 'sgmc_contacts', chunk=800)


def main():
    if '--rest' in sys.argv:
        upload_rest()
        return
    units = build_units()
    faults = build_faults()
    if '--upload' in sys.argv:
        upload(units, faults)


if __name__ == '__main__':
    main()
