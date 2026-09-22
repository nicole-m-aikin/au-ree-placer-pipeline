"""Task 14 — sidecar test: does lithology + slope beat chemistry-only transfer?

Not the doorbell. Does not write models/task9_rf_placer_gold.joblib.

The published forest is NURE ppm only. This module stacks extras that exist
on every belt (SGMC lithology score, 30 m DEM slope) and asks whether a
Carranza-style prospectivity mix travels better than chemistry alone.

Magnetics and airborne eTh are WA-only. They are scored in-belt if the
rasters exist. They are not in the transfer forest.

Labels
------
Train (NE WA): same as Task 9 — 0.15° gold MRDS + 200 m valley floor.
Transfer: same as Task 12 — 0.15° gold proximity, no DEM filter.

Slope is a known leak of the valley-floor label. The lith+slope-only
variant is the control: if that transfers, we built a valley detector.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from pipeline.ml_artifacts import REPO_ROOT
from pipeline.ml_preprocess import FEATURES, _log_impute, usable_feature_columns
from pipeline.ml_spatial import (
    FAR_FROM_MINE_DEG,
    block_cv_aucs,
    deadzone_cv_aucs,
    summarize_aucs,
)
from pipeline.task11_field_campaign import campaign_gold_xy
from pipeline.utils import (
    bbox,
    ensure_outputs,
    load_nure,
    load_radiometric_at_points,
    out,
    resolve_data_path,
    sample_raster_at_xy,
)

# Same 0–3 map as configs/ne_washington geology_score_map. Used on every belt
# so "felsic_intrusive" in Idaho is the same number as in Washington.
LITH_SCORE = {
    'MCC_metapelite': 3.0,
    'felsic_intrusive': 2.0,
    'sedimentary_cover': 1.0,
    'mafic_ultramafic': 0.0,
}

CHEM_FEATURES = list(FEATURES)
LINEAR_EXTRAS = ('lith_score', 'slope_deg', 'mag_nT', 'rad_eTh')

VARIANTS = {
    'chem': ('chem',),
    'chem_lith': ('chem', 'lith'),
    'chem_lith_slope': ('chem', 'lith', 'slope'),
    'lith_slope': ('lith', 'slope'),
}

DEFAULT_TRANSFER = (
    'configs/idaho_batholith/config.yaml',
    'configs/california_sierra/config.yaml',
    'configs/montana_placer/config.yaml',
    'configs/colorado_wet_mtns/config.yaml',
    'configs/fall_zone_nc/config.yaml',
)


def lith_score_from_type(lith_type):
    if lith_type is None or (isinstance(lith_type, float) and np.isnan(lith_type)):
        return np.nan
    return LITH_SCORE.get(str(lith_type), 1.0)


def sample_slope_deg(dem_path, lons, lats):
    """3×3 slope (degrees) at lon/lat. Geographic DEMs use metres at raster centre."""
    import rasterio
    from rasterio.transform import rowcol

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    out_vals = np.full(lons.shape, np.nan, dtype=float)
    if not dem_path or not os.path.exists(dem_path):
        return out_vals
    with rasterio.open(dem_path) as src:
        z = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            z[z == nodata] = np.nan
        px_x = float(src.transform.a)
        px_y = abs(float(src.transform.e))
        geographic = src.crs is None or src.crs.is_geographic
        if geographic:
            cy = 0.5 * (src.bounds.bottom + src.bounds.top)
            mx = 111_320.0 * np.cos(np.radians(cy)) * px_x
            my = 110_540.0 * px_y
        else:
            mx, my = px_x, px_y
        rows, cols = rowcol(src.transform, lons, lats)
        h, w = z.shape
        for i, (r, c) in enumerate(zip(rows, cols)):
            r, c = int(r), int(c)
            if r < 1 or c < 1 or r >= h - 1 or c >= w - 1:
                continue
            win = z[r - 1:r + 2, c - 1:c + 2]
            if not np.isfinite(win).all():
                continue
            dzdx = (win[1, 2] - win[1, 0]) / (2.0 * mx)
            dzdy = (win[2, 1] - win[0, 1]) / (2.0 * my)
            out_vals[i] = float(np.degrees(np.arctan(np.hypot(dzdx, dzdy))))
    return out_vals


def attach_lithology(df, cfg):
    """SGMC lith_type → 0–3 score at each grab. NaN if no polygon."""
    import geopandas as gpd

    from pipeline.task11_basemaps import load_geology

    geo = load_geology(cfg)
    if geo is None or len(geo) == 0:
        df = df.copy()
        df['lith_type'] = pd.NA
        df['lith_score'] = np.nan
        return df
    n = len(df)
    pts = gpd.GeoDataFrame(
        {'ix': np.arange(n)},
        geometry=gpd.points_from_xy(df['lon'].values, df['lat'].values),
        crs='EPSG:4326',
    )
    joined = gpd.sjoin(
        pts, geo[['lith_type', 'geometry']], how='left', predicate='intersects',
    )
    joined = joined.drop_duplicates('ix', keep='first').set_index('ix')
    lith = joined['lith_type'].reindex(np.arange(n))
    out_df = df.copy()
    out_df['lith_type'] = lith.values
    out_df['lith_score'] = [lith_score_from_type(v) for v in lith]
    return out_df


def attach_slope(df, cfg):
    path = resolve_data_path(cfg, 'dem_tif')
    out_df = df.copy()
    out_df['slope_deg'] = sample_slope_deg(path, df['lon'].values, df['lat'].values)
    return out_df


def attach_wa_only_rasters(df, cfg):
    """Mag / eTh when the files exist. Empty → all-NaN (variant is skipped)."""
    out_df = df.copy()
    mag = resolve_data_path(cfg, 'aeromagnetic_tif')
    if not mag or not os.path.exists(mag):
        fallback = REPO_ROOT / 'ne_wa_ree' / 'data' / 'aeromagnetic' / 'wa_mag_anomaly.tif'
        mag = str(fallback) if fallback.exists() else None
    out_df['mag_nT'] = sample_raster_at_xy(mag, df['lon'].values, df['lat'].values)
    rad = load_radiometric_at_points(cfg, df['lon'].values, df['lat'].values)
    out_df['rad_eTh'] = rad['rad_eTh'].values
    return out_df


def attach_extras(df, cfg, wa_only=False):
    work = attach_lithology(df, cfg)
    work = attach_slope(work, cfg)
    if wa_only:
        work = attach_wa_only_rasters(work, cfg)
    return work


def proximity_labels(df, gold_xy, radius_deg=FAR_FROM_MINE_DEG, dem_path=None,
                     max_elev_diff_m=200):
    """Gold-proximity labels. Optional 200 m valley-floor filter (train only)."""
    from scipy.spatial import cKDTree
    import rasterio

    coords = np.column_stack([df['lon'].values, df['lat'].values])
    tree = cKDTree(np.asarray(gold_xy, dtype=float))
    dist, nearest = tree.query(coords)
    near = dist <= radius_deg
    if not dem_path or not os.path.exists(dem_path) or max_elev_diff_m is None:
        return near.astype(int), dist
    with rasterio.open(dem_path) as src:
        nodata = src.nodata
        nure_z = np.array([
            v[0] if (v[0] != nodata and np.isfinite(v[0])) else np.nan
            for v in src.sample(zip(df['lon'], df['lat']))
        ])
        gold = np.asarray(gold_xy, dtype=float)
        gold_z = np.array([
            v[0] if (v[0] != nodata and np.isfinite(v[0])) else np.nan
            for v in src.sample(gold)
        ])
    elev_ok = (np.abs(nure_z - gold_z[nearest]) <= max_elev_diff_m) | np.isnan(nure_z)
    return (near & elev_ok).astype(int), dist


def _linear_impute(df, cols, medians=None):
    used = {}
    result = pd.DataFrame(index=df.index)
    for col in cols:
        v = pd.to_numeric(df[col], errors='coerce') if col in df.columns else pd.Series(np.nan, index=df.index)
        if medians is not None and col in medians and medians[col] is not None:
            med = float(medians[col])
            if np.isnan(med):
                med = 0.0
        else:
            computed = v.median()
            med = 0.0 if pd.isna(computed) else float(computed)
        used[col] = med
        result[col] = v.fillna(med)
    return result, used


def build_matrix(df, parts, chem_cols, medians=None):
    """Stack log-chemistry and linear extras. Returns (X, columns, medians)."""
    frames = []
    used = {} if medians is None else dict(medians)
    names = []
    if 'chem' in parts:
        log_df, chem_med = _log_impute(df, chem_cols, medians=medians)
        frames.append(log_df)
        names.extend(list(log_df.columns))
        used.update({f'chem:{k}': v for k, v in chem_med.items()})
    linear = []
    if 'lith' in parts:
        linear.append('lith_score')
    if 'slope' in parts:
        linear.append('slope_deg')
    if 'mag' in parts:
        linear.append('mag_nT')
    if 'eth' in parts:
        linear.append('rad_eTh')
    if linear:
        lin_med_in = None
        if medians is not None:
            lin_med_in = {c: medians.get(f'lin:{c}') for c in linear}
        lin_df, lin_med = _linear_impute(df, linear, medians=lin_med_in)
        frames.append(lin_df)
        names.extend(linear)
        used.update({f'lin:{k}': v for k, v in lin_med.items()})
    if not frames:
        raise ValueError('No feature parts requested')
    X = pd.concat(frames, axis=1)
    return X.values, names, used


def _auc(y, proba):
    y = np.asarray(y)
    if y.min() == y.max() or y.sum() < 5 or (len(y) - y.sum()) < 5:
        return None
    return float(roc_auc_score(y, proba))


def fit_forest(X, y):
    rf = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight='balanced',
    )
    rf.fit(X, y)
    return rf


def coverage_ok(series, min_frac=0.4):
    v = pd.to_numeric(series, errors='coerce')
    return float(v.notna().mean()) >= min_frac


def clip_nure(cfg):
    lon_min, lon_max, lat_min, lat_max = bbox(cfg)
    df = load_nure(cfg).dropna(subset=['lat', 'lon'])
    return df[
        (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
        & (df['lat'] >= lat_min) & (df['lat'] <= lat_max)
    ].copy().reset_index(drop=True)


def load_yaml(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def _mean_p(proba, y):
    pos = proba[y == 1]
    neg = proba[y == 0]
    return (
        float(pos.mean()) if len(pos) else None,
        float(neg.mean()) if len(neg) else None,
    )


def train_variants(train_cfg):
    df = clip_nure(train_cfg)
    gold_xy = campaign_gold_xy(train_cfg)
    if gold_xy is None:
        raise RuntimeError('NE WA gold MRDS required to train Task 14')
    dem = resolve_data_path(train_cfg, 'dem_tif')
    y, _dist = proximity_labels(
        df, gold_xy, radius_deg=FAR_FROM_MINE_DEG,
        dem_path=dem, max_elev_diff_m=200,
    )
    df = attach_extras(df, train_cfg, wa_only=True)
    chem_cols, dropped = usable_feature_columns(df, CHEM_FEATURES)
    if dropped:
        print(f'  Chem dropped: {dropped}')
    coords = np.column_stack([df['lon'].values, df['lat'].values])

    variants = dict(VARIANTS)
    if coverage_ok(df.get('mag_nT', pd.Series(dtype=float))) and coverage_ok(
        df.get('rad_eTh', pd.Series(dtype=float))
    ):
        variants['chem_mag_eth'] = ('chem', 'mag', 'eth')
        print('  WA-only mag + eTh have coverage — adding chem_mag_eth (in-belt only)')
    else:
        print('  WA-only mag/eTh sparse or missing — not used')

    fitted = {}
    rows = []
    for name, parts in variants.items():
        X, cols, medians = build_matrix(df, parts, chem_cols)
        rf = fit_forest(X, y)
        dz, _ = deadzone_cv_aucs(X, y, coords, deadzone_deg=FAR_FROM_MINE_DEG)
        blk = block_cv_aucs(X, y, df['lon'].values, df['lat'].values)
        dz_mean, dz_std, _ = summarize_aucs(dz)
        blk_mean, blk_std, _ = summarize_aucs(blk)
        fitted[name] = {
            'estimator': rf,
            'parts': parts,
            'chem_cols': chem_cols,
            'columns': cols,
            'medians': medians,
            'transferable': 'mag' not in parts and 'eth' not in parts,
        }
        imp = dict(zip(cols, [float(x) for x in rf.feature_importances_]))
        print(
            f'  {name}: n={len(y)} pos={int(y.sum())} '
            f'dead-zone={_fmt(dz_mean)} block={_fmt(blk_mean)} '
            f'top={sorted(imp, key=imp.get, reverse=True)[:3]}'
        )
        rows.append({
            'belt': 'NE Washington (train)',
            'variant': name,
            'n': int(len(y)),
            'n_pos': int(y.sum()),
            'frac_near': float(y.mean()),
            'shuffled_proxy': None,
            'deadzone_auc': dz_mean,
            'block_auc': blk_mean,
            'deadzone_std': dz_std,
            'block_std': blk_std,
            'transfer_auc': None,
            'mean_p_near': None,
            'mean_p_far': None,
            'transferable': fitted[name]['transferable'],
            'features': ','.join(cols),
        })
    return df, y, fitted, rows


def _fmt(x):
    return 'NA' if x is None else f'{x:.3f}'


def score_variant(fitted, df, y):
    X, _cols, _ = build_matrix(
        df, fitted['parts'], fitted['chem_cols'], medians=fitted['medians'],
    )
    proba = fitted['estimator'].predict_proba(X)[:, 1]
    return proba, _auc(y, proba)


def transfer_variants(fitted, transfer_cfg, train_rows):
    belt = (transfer_cfg.get('study_area') or {}).get('name', 'belt')
    df = clip_nure(transfer_cfg)
    gold_xy = campaign_gold_xy(transfer_cfg)
    if gold_xy is None:
        raise RuntimeError(f'No gold MRDS for {belt}')
    y, _dist = proximity_labels(
        df, gold_xy, radius_deg=FAR_FROM_MINE_DEG,
        dem_path=None, max_elev_diff_m=None,
    )
    df = attach_extras(df, transfer_cfg, wa_only=False)
    rows = []
    print(f'  {belt}: n={len(df)} pos={int(y.sum())} ({100 * y.mean():.0f}% near gold)')
    for name, spec in fitted.items():
        if not spec['transferable']:
            continue
        proba, auc = score_variant(spec, df, y)
        near, far = _mean_p(proba, y)
        print(
            f'    {name}: transfer AUC={_fmt(auc)}  '
            f'mean P near/far={_fmt(near)}/{_fmt(far)}'
        )
        rows.append({
            'belt': belt,
            'variant': name,
            'n': int(len(y)),
            'n_pos': int(y.sum()),
            'frac_near': float(y.mean()),
            'shuffled_proxy': None,
            'deadzone_auc': None,
            'block_auc': None,
            'deadzone_std': None,
            'block_std': None,
            'transfer_auc': auc,
            'mean_p_near': near,
            'mean_p_far': far,
            'transferable': True,
            'features': ','.join(spec['columns']),
        })
    return rows


def write_summary(cfg, table):
    ensure_outputs(cfg['outputs_dir'])
    path = out(cfg, 'tables', 'task14_prospectivity_compare.csv')
    table.to_csv(path, index=False)
    lines = [
        'TASK 14 — extras vs chemistry-only transfer',
        'Sidecar. Doorbell unchanged.',
        '',
        'Train labels: 0.15° gold + 200 m valley floor.',
        'Transfer labels: 0.15° gold (Task 12).',
        'Transferable extras: SGMC lith_score (0–3), 30 m slope_deg.',
        'Slope leaks the valley-floor label. lith_slope is the control.',
        '',
    ]
    for belt, grp in table.groupby('belt', sort=False):
        lines.append(belt)
        for _, r in grp.iterrows():
            if pd.notna(r.get('transfer_auc')):
                metric = f"transfer={_fmt(r['transfer_auc'])}"
            else:
                metric = (
                    f"dead-zone={_fmt(r['deadzone_auc'])}  "
                    f"block={_fmt(r['block_auc'])}"
                )
            lines.append(f"  {r['variant']:<18} {metric}")
        lines.append('')
    chem_t = table[(table['variant'] == 'chem') & table['transfer_auc'].notna()]
    extra_t = table[(table['variant'] == 'chem_lith_slope') & table['transfer_auc'].notna()]
    if len(chem_t) and len(extra_t):
        lines.append('Did extras beat chemistry on transfer?')
        merged = chem_t[['belt', 'transfer_auc']].merge(
            extra_t[['belt', 'transfer_auc']], on='belt', suffixes=('_chem', '_extra'),
        )
        wins = 0
        for _, r in merged.iterrows():
            d = r['transfer_auc_extra'] - r['transfer_auc_chem']
            flag = 'YES' if d > 0.03 else ('tie' if abs(d) <= 0.03 else 'no')
            if flag == 'YES':
                wins += 1
            lines.append(
                f"  {r['belt']}: chem {r['transfer_auc_chem']:.3f} → "
                f"chem+lith+slope {r['transfer_auc_extra']:.3f}  ({flag})"
            )
        if wins == 0:
            lines.append(
                'Extras did not travel. Prospectivity layers did not rescue the doorbell.'
            )
        lines.append('')
    text_path = out(cfg, 'text', 'task14_prospectivity_summary.txt')
    Path(text_path).parent.mkdir(parents=True, exist_ok=True)
    Path(text_path).write_text('\n'.join(lines) + '\n')
    print(f'  Wrote {path}')
    print(f'  Wrote {text_path}')
    return path, text_path


def run(train_cfg, transfer_cfgs=None):
    transfer_cfgs = transfer_cfgs or []
    print('Task 14: prospectivity extras vs NURE-only (doorbell untouched)')
    _df, _y, fitted, rows = train_variants(train_cfg)
    for tcfg in transfer_cfgs:
        rows.extend(transfer_variants(fitted, tcfg, rows))
    table = pd.DataFrame(rows)
    write_summary(train_cfg, table)
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description='Task 14 extras vs chemistry transfer')
    parser.add_argument(
        '--train', default='configs/ne_washington/config.yaml',
        help='Training config (must be NE WA)',
    )
    parser.add_argument(
        '--transfer', nargs='*', default=None,
        help='Transfer configs (default: ID CA MT CO NC)',
    )
    args = parser.parse_args(argv)
    train_cfg = load_yaml(args.train)
    short = str((train_cfg.get('study_area') or {}).get('short') or '').lower()
    if short not in ('ne_wa', 'ne_washington', ''):
        raise SystemExit('Task 14 trains on NE Washington only. Do not overwrite that.')
    paths = args.transfer if args.transfer is not None else list(DEFAULT_TRANSFER)
    transfer_cfgs = [load_yaml(p) for p in paths]
    run(train_cfg, transfer_cfgs)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
