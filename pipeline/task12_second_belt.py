"""Score the frozen NE WA forest on one other placer belt. Do not retrain.

Literature (Airola 2018; LIT_REVIEW §8 item 7): transfer is the honest test.
This module fetches a belt's NURE + gold MRDS, applies the published
log-medians, and reports ROC-AUC. A sag is the result, not a failure.

Not a national model. Do not run this against the training box.
"""

import os
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline.ml_artifacts import (
    REPO_ROOT,
    load_published_artifacts,
    persist_transfer,
)
from pipeline.ml_preprocess import FEATURES, _log_impute
from pipeline.ml_spatial import FAR_FROM_MINE_DEG
from pipeline.utils import bbox, ensure_outputs, load_nure, out

NURE_CSV_ZIP = 'https://mrdata.usgs.gov/nure/sediment/nuresed-csv.zip'
MRDS_BBOX = 'https://mrdata.usgs.gov/mrds/search-bbox.php'
# HSSR field names → the forest's NGDB-style columns.
# U: delayed-neutron (u_dn_ppm) is the NURE uranium everyone quotes.
HSSR_TO_FEATURE = {
    'th_ppm': 'Th',
    'ce_ppm': 'Ce',
    'la_ppm': 'La',
    'p_ppm': 'P',
    'u_dn_ppm': 'U',
    'u_ppm': 'U',
    'au_ppm': 'Au',
    'as_ppm': 'As',
    'ti_ppm': 'Ti',
    'fe_pct': 'Fe',
    'zr_ppm': 'Zr',
    'y_ppm': 'Y',
}


def hssr_to_nure_frame(raw):
    """Map a USGS HSSR table onto load_nure columns (element symbols + lat/lon)."""
    work = raw.copy()
    work.columns = [str(c).strip().lower() for c in work.columns]
    out = pd.DataFrame(index=work.index)
    if 'latitude' in work.columns:
        out['lat'] = pd.to_numeric(work['latitude'], errors='coerce')
    elif 'lat' in work.columns:
        out['lat'] = pd.to_numeric(work['lat'], errors='coerce')
    if 'longitude' in work.columns:
        out['lon'] = pd.to_numeric(work['longitude'], errors='coerce')
    elif 'lon' in work.columns:
        out['lon'] = pd.to_numeric(work['lon'], errors='coerce')
    for src, dest in HSSR_TO_FEATURE.items():
        if src not in work.columns or dest in out.columns:
            continue
        out[dest] = _hssr_numeric(work[src])
    if 'rec_no' in work.columns:
        out['lab_id'] = work['rec_no'].astype(str)
    elif 'primeid' in work.columns:
        out['lab_id'] = work['primeid'].astype(str)
    return out


def _hssr_numeric(series):
    """Turn HSSR cells into numbers. '<5' / 'L5' → -5 (half-MDL convention)."""
    def _one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        s = str(v).strip()
        if s == '' or s.lower() in ('nan', 'none', 'null'):
            return np.nan
        m = re.match(r'^[<Ll]\s*([0-9]*\.?[0-9]+)$', s)
        if m:
            return -float(m.group(1))
        try:
            return float(s)
        except ValueError:
            return np.nan
    return series.map(_one)


def parse_mrds_bbox_xml(xml_text):
    """Gold sites from an MRDS search-bbox XML blob."""
    root = ET.fromstring(xml_text)
    rows = []
    for rec in root.iter('record'):
        codes = (rec.attrib.get('code_list') or '').upper()
        tokens = codes.replace(',', ' ').split()
        if 'AU' not in tokens:
            continue
        if 'REE' in tokens:
            continue
        try:
            lon = float(rec.attrib['x'])
            lat = float(rec.attrib['y'])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({
            'dep_id': rec.attrib.get('dep_id'),
            'site_name': rec.attrib.get('site_name'),
            'dev_stat': rec.attrib.get('dev_stat'),
            'code_list': rec.attrib.get('code_list'),
            'target_commodity': 'Gold',
            'lon': lon,
            'lat': lat,
        })
    return pd.DataFrame(rows)


def _http_get(url, timeout=180):
    req = Request(url, headers={'User-Agent': 'au-ree-placer-pipeline/task12'})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_mrds_gold(lon_min, lat_min, lon_max, lat_max, dest_geojson):
    url = (
        f'{MRDS_BBOX}?xmin={lon_min}&ymin={lat_min}'
        f'&xmax={lon_max}&ymax={lat_max}'
    )
    print(f'  MRDS bbox {url}')
    xml_text = _http_get(url, timeout=180).decode('utf-8', errors='replace')
    df = parse_mrds_bbox_xml(xml_text)
    if df.empty:
        raise RuntimeError('MRDS bbox returned no gold sites')
    import geopandas as gpd
    from shapely.geometry import Point
    gdf = gpd.GeoDataFrame(
        df, geometry=[Point(xy) for xy in zip(df['lon'], df['lat'])],
        crs='EPSG:4326',
    )
    os.makedirs(os.path.dirname(os.path.abspath(dest_geojson)) or '.', exist_ok=True)
    gdf.to_file(dest_geojson, driver='GeoJSON')
    print(f'  Wrote {len(gdf)} gold MRDS sites → {dest_geojson}')
    return gdf


def fetch_nure_hssr(lon_min, lat_min, lon_max, lat_max, dest_csv, cache_dir=None):
    """Download the national HSSR CSV once, clip, write NGDB-shaped CSV."""
    cache_dir = Path(cache_dir or (REPO_ROOT / 'data' / 'nure' / 'raw'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / 'nuresed-csv.zip'
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        print(f'  Downloading NURE HSSR CSV ({NURE_CSV_ZIP}) ...')
        zip_path.write_bytes(_http_get(NURE_CSV_ZIP, timeout=600))
        print(f'  Cached {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)')
    csv_name = None
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith('.csv')]
        if not names:
            raise RuntimeError(f'no CSV inside {zip_path}')
        csv_name = names[0]
        print(f'  Reading {csv_name} in chunks ...')
        chunks = []
        with zf.open(csv_name) as fh:
            for chunk in pd.read_csv(fh, chunksize=50_000, low_memory=False):
                cols = {c.lower(): c for c in chunk.columns}
                lat_c = cols.get('latitude') or cols.get('lat')
                lon_c = cols.get('longitude') or cols.get('lon')
                if not lat_c or not lon_c:
                    raise RuntimeError(f'NURE CSV missing lat/lon: {list(chunk.columns)[:12]}')
                lat = pd.to_numeric(chunk[lat_c], errors='coerce')
                lon = pd.to_numeric(chunk[lon_c], errors='coerce')
                keep = (
                    lat.between(lat_min, lat_max) & lon.between(lon_min, lon_max)
                )
                if keep.any():
                    chunks.append(chunk.loc[keep])
    if not chunks:
        raise RuntimeError(
            f'NURE clip was empty — check the bbox '
            f'({lon_min}, {lat_min})–({lon_max}, {lat_max})'
        )
    raw = pd.concat(chunks, ignore_index=True)
    mapped = hssr_to_nure_frame(raw)
    mapped = mapped.dropna(subset=['lat', 'lon'])
    os.makedirs(os.path.dirname(os.path.abspath(dest_csv)) or '.', exist_ok=True)
    mapped.to_csv(dest_csv, index=False)
    present = [f for f in FEATURES if f in mapped.columns and mapped[f].notna().any()]
    print(f'  Wrote {len(mapped)} NURE rows → {dest_csv}')
    print(f'  Features present: {present}')
    missing = [f for f in FEATURES if f not in present]
    if missing:
        print(f'  Missing (will use NE WA log-medians): {missing}')
    return mapped


def score_frozen_transfer(df, gold_xy, estimator, metadata, radius_deg=FAR_FROM_MINE_DEG):
    """Apply the published forest. Labels = gold proximity (no DEM)."""
    from scipy.spatial import cKDTree

    order = list(metadata['feature_order'])
    log_X, _ = _log_impute(df, order, medians=metadata['log_medians'])
    proba = estimator.predict_proba(log_X.values)[:, 1]
    coords = np.column_stack([df['lon'].values, df['lat'].values])
    tree = cKDTree(np.asarray(gold_xy, dtype=float))
    dist, _ = tree.query(coords)
    y = (dist <= radius_deg).astype(int)
    auc = None
    if y.min() != y.max() and y.sum() >= 5 and (len(y) - y.sum()) >= 5:
        auc = float(roc_auc_score(y, proba))
    return pd.DataFrame({
        'lon': df['lon'].values,
        'lat': df['lat'].values,
        'p_anomalous': proba,
        'label': y,
        'nearest_gold_deg': dist,
    }), auc


def belt_slug(cfg):
    """Filename stem. Idaho keeps the historic 'idaho' name."""
    short = (cfg.get('study_area', {}) or {}).get('short') or 'second_belt'
    if short in ('id_batholith', 'idaho'):
        return 'idaho'
    return str(short).replace('-', '_')


def _assert_not_training_belt(cfg, metadata):
    trained = str(metadata.get('study_area') or '').strip().lower()
    this = str((cfg.get('study_area') or {}).get('name') or '').strip().lower()
    if trained and this and trained == this:
        raise RuntimeError(
            f'{this} is the training belt. Transfer needs a different box.'
        )


def _transfer_note(belt, auc, auc_tight, frac_near, mean_pos, mean_neg):
    bits = [f'Frozen NE Washington forest scored on {belt} NURE.']
    if auc is None:
        bits.append('Transfer AUC could not be scored (one class).')
    elif auc < 0.60:
        bits.append(f'Transfer AUC is {auc:.2f} (coin flip — the forest does not travel).')
    else:
        bits.append(f'Transfer AUC is {auc:.2f}.')
    if auc_tight is not None:
        bits.append(f'Tightening the gold circle to 0.05° gives {auc_tight:.2f}.')
    if frac_near is not None and frac_near >= 0.60:
        bits.append(
            '0.15° labels most of this box because gold MRDS pins are dense '
            '— that is a weak negative class.'
        )
    if mean_pos is not None and mean_neg is not None:
        if abs(mean_pos - mean_neg) < 0.05:
            bits.append('Mean P is flat near gold and far from it.')
        elif mean_neg > mean_pos + 0.05:
            bits.append('Mean P is higher far from gold than next to it.')
    bits.append(
        'This is a transfer test, not a new model. Do not call the doorbell national.'
    )
    return ' '.join(bits)


def run(cfg):
    ensure_outputs(cfg['outputs_dir'])
    slug = belt_slug(cfg)
    belt = (cfg.get('study_area') or {}).get('name', slug)
    short = (cfg.get('study_area') or {}).get('short', slug)
    lon_min, lon_max, lat_min, lat_max = bbox(cfg)
    nure_path = cfg['data']['nure_csv']
    mrds_path = cfg['data']['mrds_geojson']
    if not os.path.isabs(nure_path):
        nure_path = str(REPO_ROOT / nure_path)
    if not os.path.isabs(mrds_path):
        mrds_path = str(REPO_ROOT / mrds_path)

    if not os.path.exists(nure_path):
        fetch_nure_hssr(lon_min, lat_min, lon_max, lat_max, nure_path)
    else:
        print(f'  Using cached NURE {nure_path}')
    if not os.path.exists(mrds_path):
        fetch_mrds_gold(lon_min, lat_min, lon_max, lat_max, mrds_path)
    else:
        print(f'  Using cached MRDS {mrds_path}')

    df = load_nure({'data': {'nure_csv': nure_path}})
    df = df.dropna(subset=['lat', 'lon'])
    df = df[
        (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
        & (df['lat'] >= lat_min) & (df['lat'] <= lat_max)
    ].copy()

    import geopandas as gpd
    mrds = gpd.read_file(mrds_path)
    gold = mrds
    if 'target_commodity' in mrds.columns:
        gold = mrds[
            mrds['target_commodity'].fillna('').str.contains('Gold')
            & ~mrds['target_commodity'].fillna('').str.contains('Rare Earth')
        ]
    gold_xy = np.column_stack([gold.geometry.x.values, gold.geometry.y.values])

    estimator, metadata = load_published_artifacts()
    _assert_not_training_belt(cfg, metadata)
    scored, auc = score_frozen_transfer(df, gold_xy, estimator, metadata)
    scored_tight, auc_tight = score_frozen_transfer(
        df, gold_xy, estimator, metadata, radius_deg=0.05,
    )
    scored.to_csv(out(cfg, 'tables', f'task12_{slug}_transfer_scores.csv'), index=False)

    n_pos = int(scored['label'].sum())
    n = int(len(scored))
    mean_pos = float(scored.loc[scored['label'] == 1, 'p_anomalous'].mean()) if n_pos else None
    mean_neg = float(scored.loc[scored['label'] == 0, 'p_anomalous'].mean()) if n_pos < n else None
    frac = None if not n else round(n_pos / n, 4)
    auc_r = None if auc is None else round(auc, 4)
    auc_t = None if auc_tight is None else round(auc_tight, 4)
    summary = {
        'belt': belt,
        'short': short,
        'model_id': metadata.get('model_id'),
        'trained_on': metadata.get('study_area'),
        'retrained': False,
        'n_samples': n,
        'n_positive': n_pos,
        'n_gold_mrds': int(len(gold_xy)),
        'fraction_near_gold': frac,
        'transfer_auc': auc_r,
        'transfer_auc_0p05deg': auc_t,
        'mean_p_near_gold': None if mean_pos is None else round(mean_pos, 4),
        'mean_p_far_from_gold': None if mean_neg is None else round(mean_neg, 4),
        'mean_p_all': None if not n else round(float(scored['p_anomalous'].mean()), 4),
        'label_method': (
            'proximity (Task 12 does not apply a DEM elevation cut)'
        ),
        'mrds_proximity_deg': FAR_FROM_MINE_DEG,
        'note': _transfer_note(belt, auc, auc_tight, frac, mean_pos, mean_neg),
    }
    persist_transfer(summary)
    path = out(cfg, 'text', f'task12_{slug}_transfer_summary.txt')
    title = f'SECOND BELT — FROZEN NE WA FOREST ON {belt.upper()}'
    lines = [
        title,
        summary['note'],
        '=' * 68,
        '',
        f"NURE grabs:     {n}",
        f"Near gold MRDS: {n_pos}  ({100.0 * n_pos / n:.1f}%)" if n else '',
        f"Gold MRDS pins: {len(gold_xy)}",
        f"Transfer AUC (0.15°): {summary['transfer_auc']}",
        f"Transfer AUC (0.05°): {summary['transfer_auc_0p05deg']}",
        f"Mean P near gold / far: {summary['mean_p_near_gold']} / {summary['mean_p_far_from_gold']}",
        f"Mean P all grabs:       {summary['mean_p_all']}",
        '',
        f"NE WA random CV:  {metadata.get('cv_auc_mean')}",
        f"NE WA spatial CV: {metadata.get('spatial_cv_auc_mean')}",
        '',
        'Do not retrain from this number. Do not call the doorbell national.',
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return summary


if __name__ == '__main__':
    import sys
    import yaml
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/idaho_batholith/config.yaml'
    with open(cfg_path) as f:
        run(yaml.safe_load(f))
