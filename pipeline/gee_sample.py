"""Sample Earth Engine catalog layers at county-rank pan pins.

Usage:
  python -m pipeline.gee_sample --config configs/kern_river/config.yaml
  # or called from pipeline.county_rank when county_rank.gee_sample is true

Returns one row per pin_id with:
  elev10_m, slope10_deg   10 m 3DEP
  water_pct, water_regime JRC Global Surface Water occurrence
  lith_code, lith_favor   CSP ERGo US lithology
  landform_code           CSP ERGo US landforms
  nlcd, land_filter       NLCD 2021 (drop = developed / cropland)
  road_m                  distance to nearest TIGER road (m, capped)
  fire_year               most recent MTBS ignition year within 2 km (or NaN)
  has_lidar_1m            True where USGS 3DEP 1 m elevation exists
  best_months             snow-free + below-median precip span (MODIS + PRISM)

Filters / flags only — they do not change evidence_score weights. county_rank
drops land_filter=='drop' from the recreational list and records the flags.
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from pipeline.ml_artifacts import REPO_ROOT
from pipeline.utils import out

DEFAULT_PROJECT = os.environ.get('EE_PROJECT', 'placer-picker-511303')
ROAD_SEARCH_M = 5000.0
FIRE_BUFFER_M = 2000.0
SAMPLE_SCALE_M = 30

# CSP ERGo US lithology (b1). Favorable for Au-placer source rock: metamorphics,
# acid/basic plutonics and volcanics, ultrabasics. Sediment/carbonate/water out.
FAVORABLE_LITH = frozenset({4, 5, 8, 9, 12, 13, 14, 15})
LITH_NAME = {
    1: 'unconsolidated', 2: 'carbonate', 3: 'non_carb_sed', 4: 'metamorphic',
    5: 'acid_plutonic', 6: 'mixed_sed', 7: 'silicic_residual', 8: 'basic_plutonic',
    9: 'basic_volcanic', 10: 'ice', 11: 'water', 12: 'acid_volcanic',
    13: 'intermediate_volcanic', 14: 'basic_volcanic_alt', 15: 'ultrabasic',
}

# NLCD 2021 landcover. Drop developed + cultivated from "go pan here".
NLCD_DROP = frozenset({21, 22, 23, 24, 81, 82})
NLCD_NAME = {
    11: 'open_water', 12: 'ice', 21: 'developed_open', 22: 'developed_low',
    23: 'developed_med', 24: 'developed_high', 31: 'barren', 41: 'deciduous',
    42: 'evergreen', 43: 'mixed_forest', 52: 'shrub', 71: 'grassland',
    81: 'pasture', 82: 'cropland', 90: 'woody_wetland', 95: 'herb_wetland',
}

MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
SNOW_FREE_MAX = 15.0  # mean NDSI snow cover %

SAMPLE_COLS = [
    'pin_id', 'elev10_m', 'slope10_deg', 'water_pct', 'water_regime',
    'lith_code', 'lith_name', 'lith_favor', 'landform_code',
    'nlcd', 'nlcd_name', 'land_filter', 'road_m', 'fire_year', 'has_lidar_1m',
    'best_months',
]


def water_regime(pct):
    if pct is None:
        return 'dry'
    try:
        v = float(pct)
    except (TypeError, ValueError):
        return 'dry'
    if not math.isfinite(v) or v <= 0:
        return 'dry'
    if v < 25:
        return 'seasonal'
    return 'year_round'


def land_filter(nlcd):
    if nlcd is None or (isinstance(nlcd, float) and math.isnan(nlcd)):
        return 'unknown'
    return 'drop' if int(nlcd) in NLCD_DROP else 'ok'


def format_month_span(months):
    """Turn [7,8,9,10] into 'Jul–Oct'. Non-contiguous → 'Jul, Sep–Oct'."""
    if not months:
        return ''
    months = sorted({int(m) for m in months if 1 <= int(m) <= 12})
    runs = []
    start = prev = months[0]
    for m in months[1:]:
        if m == prev + 1:
            prev = m
            continue
        runs.append((start, prev))
        start = prev = m
    runs.append((start, prev))
    parts = []
    for a, b in runs:
        if a == b:
            parts.append(MONTH_ABBR[a - 1])
        else:
            parts.append(f'{MONTH_ABBR[a - 1]}–{MONTH_ABBR[b - 1]}')
    return ', '.join(parts)


def best_months_from_climate(snow12, ppt12, snow_max=SNOW_FREE_MAX):
    """Snow-free months with below-median precip among those months.

    snow12 / ppt12: length-12 sequences (Jan–Dec). Missing → skipped.
    """
    snow = [None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)
            for v in snow12]
    ppt = [None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)
           for v in ppt12]
    free = [i + 1 for i, s in enumerate(snow) if s is not None and s <= snow_max]
    if not free:
        # No snow climatology — fall back to driest 3 months
        ranked = sorted(
            ((i + 1, p) for i, p in enumerate(ppt) if p is not None),
            key=lambda t: t[1],
        )
        return format_month_span([m for m, _ in ranked[:3]])
    free_ppt = [ppt[m - 1] for m in free if ppt[m - 1] is not None]
    if not free_ppt:
        return format_month_span(free)
    med = float(np.median(free_ppt))
    good = [m for m in free if ppt[m - 1] is not None and ppt[m - 1] <= med]
    return format_month_span(good or free)


def _init_ee(project):
    import ee

    ee.Initialize(project=project)
    return ee


def _climate_stack(ee, start='2018-01-01', end='2024-01-01'):
    """12 snow% bands + 12 ppt bands (Jan–Dec climatology)."""
    snow_col = (ee.ImageCollection('MODIS/061/MOD10A1')
                .filterDate(start, end).select('NDSI_Snow_Cover'))
    ppt_col = (ee.ImageCollection('OREGONSTATE/PRISM/ANm')
               .filterDate(start, end).select('ppt'))
    bands = []
    for m in range(1, 13):
        snow_m = snow_col.filter(ee.Filter.calendarRange(m, m, 'month')).mean()
        ppt_m = ppt_col.filter(ee.Filter.calendarRange(m, m, 'month')).mean()
        bands.append(snow_m.rename(f'snow_{m:02d}'))
        bands.append(ppt_m.rename(f'ppt_{m:02d}'))
    return ee.Image.cat(bands)


def _stack(ee):
    dem10 = ee.ImageCollection('USGS/3DEP/10m_collection').mosaic().select('elevation')
    nlcd = (ee.ImageCollection('USGS/NLCD_RELEASES/2021_REL/NLCD')
            .filterDate('2021-01-01', '2022-01-01').first()
            .select('landcover'))
    # Mask band (1 = 1 m lidar exists). Mosaic elevation alone returns 0 over
    # no-data voids and would falsely set has_lidar_1m everywhere.
    lidar_mask = (ee.ImageCollection('USGS/3DEP/1m').select('elevation')
                  .mosaic().mask().rename('lidar1m'))
    return ee.Image.cat([
        dem10.rename('elev10'),
        ee.Terrain.slope(dem10).rename('slope10'),
        ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').rename('water_pct'),
        ee.Image('CSP/ERGo/1_0/US/lithology').select(0).rename('lith'),
        ee.Image('CSP/ERGo/1_0/US/landforms').select(0).rename('landform'),
        nlcd.rename('nlcd'),
        lidar_mask,
        _climate_stack(ee),
    ])


def _points_fc(ee, pans):
    feats = []
    for _, r in pans.iterrows():
        feats.append(ee.Feature(
            ee.Geometry.Point([float(r['lon']), float(r['lat'])]),
            {'pin_id': str(r['pin_id'])},
        ))
    return ee.FeatureCollection(feats)


def _road_distance(ee, fc):
    """Nearest TIGER road distance (m), capped at ROAD_SEARCH_M."""
    roads = ee.FeatureCollection('TIGER/2016/Roads')
    def _one(f):
        geom = f.geometry()
        near = roads.filterBounds(geom.buffer(ROAD_SEARCH_M))
        d = near.distance(ROAD_SEARCH_M).reduceRegion(
            ee.Reducer.min(), geom, 30, maxPixels=1e6,
        ).get('distance')
        return f.set('road_m', d)
    return fc.map(_one)


def _fire_year(ee, fc):
    fires = ee.FeatureCollection('USFS/GTAC/MTBS/burned_area_boundaries/v1')
    def _one(f):
        hits = fires.filterBounds(f.geometry().buffer(FIRE_BUFFER_M))
        latest = hits.aggregate_max('Ig_Date')
        return f.set('fire_ms', latest)
    return fc.map(_one)


def _ms_to_year(ms):
    if ms is None or (isinstance(ms, float) and math.isnan(ms)):
        return np.nan
    try:
        return int(datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc).year)
    except (OverflowError, OSError, ValueError, TypeError):
        return np.nan


def sample_pins(pans, project=DEFAULT_PROJECT):
    """Sample EE layers at pan lon/lat. Returns a DataFrame indexed by pin_id order."""
    if pans is None or len(pans) == 0:
        return pd.DataFrame(columns=SAMPLE_COLS)
    ee = _init_ee(project)
    fc = _points_fc(ee, pans)
    sampled = _stack(ee).reduceRegions(fc, ee.Reducer.first(), SAMPLE_SCALE_M)
    sampled = _road_distance(ee, sampled)
    sampled = _fire_year(ee, sampled)
    info = sampled.getInfo()
    rows = []
    for feat in info.get('features') or []:
        p = feat.get('properties') or {}
        nlcd = p.get('nlcd')
        lith = p.get('lith')
        water = p.get('water_pct')
        lidar = p.get('lidar1m')
        lith_i = None if lith is None else int(lith)
        nlcd_i = None if nlcd is None else int(nlcd)
        try:
            has_lidar = bool(lidar is not None and float(lidar) > 0.5)
        except (TypeError, ValueError):
            has_lidar = False
        snow12 = [p.get(f'snow_{m:02d}') for m in range(1, 13)]
        ppt12 = [p.get(f'ppt_{m:02d}') for m in range(1, 13)]
        rows.append({
            'pin_id': p.get('pin_id'),
            'elev10_m': None if p.get('elev10') is None else round(float(p['elev10']), 1),
            'slope10_deg': None if p.get('slope10') is None else round(float(p['slope10']), 2),
            'water_pct': None if water is None else round(float(water), 1),
            'water_regime': water_regime(water),
            'lith_code': lith_i,
            'lith_name': LITH_NAME.get(lith_i or -1, ''),
            'lith_favor': bool(lith_i in FAVORABLE_LITH) if lith_i is not None else False,
            'landform_code': None if p.get('landform') is None else int(p['landform']),
            'nlcd': nlcd_i,
            'nlcd_name': NLCD_NAME.get(nlcd_i or -1, ''),
            'land_filter': land_filter(nlcd),
            'road_m': None if p.get('road_m') is None else round(float(p['road_m']), 0),
            'fire_year': _ms_to_year(p.get('fire_ms')),
            'has_lidar_1m': has_lidar,
            'best_months': best_months_from_climate(snow12, ppt12),
        })
    out_df = pd.DataFrame(rows)
    # Preserve input order
    order = {pid: i for i, pid in enumerate(pans['pin_id'].astype(str))}
    out_df['_ord'] = out_df['pin_id'].map(order)
    return out_df.sort_values('_ord').drop(columns='_ord').reset_index(drop=True)


def apply_sample_flags(gdf, samples):
    """Join sample columns onto a pin GeoDataFrame / DataFrame."""
    if samples is None or len(samples) == 0:
        for c in SAMPLE_COLS:
            if c != 'pin_id' and c not in gdf.columns:
                gdf[c] = np.nan if c not in (
                    'water_regime', 'lith_name', 'nlcd_name', 'land_filter',
                    'lith_favor', 'has_lidar_1m', 'best_months',
                ) else (False if c in ('lith_favor', 'has_lidar_1m') else '')
        return gdf
    keep = [c for c in SAMPLE_COLS if c != 'pin_id' and c in samples.columns]
    cols = ['pin_id', *keep]
    merged = gdf.drop(columns=[c for c in keep if c in gdf.columns], errors='ignore')
    return merged.merge(samples[cols], on='pin_id', how='left')


def gee_sample_enabled(cfg):
    cr = cfg.get('county_rank') or {}
    if 'gee_sample' in cr:
        return bool(cr['gee_sample'])
    return bool((cfg.get('gee') or {}).get('sample', True))


def gee_project(cfg):
    return str(
        (cfg.get('gee') or {}).get('project')
        or os.environ.get('EE_PROJECT')
        or DEFAULT_PROJECT
    )


def run(cfg, pans=None, project=None):
    """Sample pins for a region. pans=None → read county_rank_pans.csv."""
    project = project or gee_project(cfg)
    if pans is None:
        path = out(cfg, 'tables', 'county_rank_pans.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f'No {path}; run pipeline.county_rank first')
        pans = pd.read_csv(path)
    print(f'  Sampling {len(pans)} pins on Earth Engine project {project} …', flush=True)
    samples = sample_pins(pans, project=project)
    dest = out(cfg, 'tables', 'county_rank_gee_samples.csv')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    samples.to_csv(dest, index=False)
    n_drop = int((samples['land_filter'] == 'drop').sum())
    n_dry = int((samples['water_regime'] == 'dry').sum())
    n_burn = int(samples['fire_year'].notna().sum())
    n_lidar = int(samples['has_lidar_1m'].sum())
    print(
        f'  Wrote {dest}  land_drop={n_drop} dry={n_dry} '
        f'burned={n_burn} lidar1m={n_lidar}',
        flush=True,
    )
    return samples


def main():
    parser = argparse.ArgumentParser(description='Sample EE layers at county-rank pins')
    parser.add_argument('--config', required=True)
    parser.add_argument('--project', default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg, project=args.project)


if __name__ == '__main__':
    main()
