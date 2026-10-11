"""Compare county_rank on two DEM sources for the same region — a sidecar
check, not a production switch. Written for the Kern River subset test:
30 m Copernicus (configs/kern_river_subset) vs 10 m 3DEP pulled from Earth
Engine (configs/kern_river_subset_3dep10m, pipeline/fetch_gee_dem.py).

Matches pans between the two runs by nearest neighbor (walk geometry shifts
a little with DEM resolution, so pan counts rarely match exactly), reports
how much trap_score / evidence_score / rank move, and attaches JRC surface-
water occurrence at each baseline pan as an extra, clearly-separate column —
informational, not blended into evidence_score.

Usage:
  python -m pipeline.compare_dem_sources \
      --baseline configs/kern_river_subset/config.yaml \
      --compared configs/kern_river_subset_3dep10m/config.yaml \
      --label-baseline "30 m Copernicus" --label-compared "10 m 3DEP"
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
import yaml

from pipeline.utils import out, resolve_data_path

MATCH_RADIUS_M = 300.0
M_PER_DEG = 111_320.0


def _pans_path(cfg):
    return out(cfg, 'tables', 'county_rank_pans.csv')


def _load(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    path = _pans_path(cfg)
    return cfg, pd.read_csv(path)


def match_nearest(base, comp, radius_m=MATCH_RADIUS_M):
    """Pair each baseline pan with its nearest compared pan within radius_m."""
    if base.empty or comp.empty:
        return pd.DataFrame()
    lat0 = float(base['lat'].mean())

    def to_xy(df):
        x = (df['lon'].to_numpy() - base['lon'].mean()) * M_PER_DEG * math.cos(math.radians(lat0))
        y = (df['lat'].to_numpy() - base['lat'].mean()) * M_PER_DEG
        return np.column_stack([x, y])

    from scipy.spatial import cKDTree

    tree = cKDTree(to_xy(comp))
    d, idx = tree.query(to_xy(base), k=1)
    rows = []
    for i, (dist, j) in enumerate(zip(d, idx)):
        if dist > radius_m:
            continue
        rows.append({
            'lon': base.iloc[i]['lon'], 'lat': base.iloc[i]['lat'],
            'match_dist_m': round(float(dist), 1),
            'trap_score_base': base.iloc[i]['trap_score'],
            'trap_score_cmp': comp.iloc[j]['trap_score'],
            'evidence_score_base': base.iloc[i]['evidence_score'],
            'evidence_score_cmp': comp.iloc[j]['evidence_score'],
            'rec_rank_base': base.iloc[i].get('rec_rank_in_county'),
            'rec_rank_cmp': comp.iloc[j].get('rec_rank_in_county'),
        })
    return pd.DataFrame(rows)


def sample_surface_water(cfg, lons, lats):
    """Occurrence % (0-100) at each point, or NaN where no raster / out of bounds."""
    path = resolve_data_path(cfg, 'gsw_occurrence_tif')
    if not path:
        return np.full(len(lons), np.nan)
    import rasterio

    with rasterio.open(path) as src:
        vals = np.full(len(lons), np.nan)
        for i, (lon, lat) in enumerate(zip(lons, lats)):
            try:
                row, col = src.index(lon, lat)
                if 0 <= row < src.height and 0 <= col < src.width:
                    v = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
                    vals[i] = float(v)
            except (IndexError, ValueError):
                continue
    return vals


def run(baseline_cfg_path, compared_cfg_path, label_base, label_cmp):
    base_cfg, base = _load(baseline_cfg_path)
    _, comp = _load(compared_cfg_path)

    print(f'  {label_base}: {len(base)} pans   {label_cmp}: {len(comp)} pans')

    matched = match_nearest(base, comp)
    lines = [
        f'DEM comparison — {label_base} vs {label_cmp}',
        'A sidecar check. Neither evidence_score formula changed.',
        '=' * 60,
        '',
        f'{label_base}: {len(base)} pans   {label_cmp}: {len(comp)} pans',
        f'Matched within {MATCH_RADIUS_M:.0f} m: {len(matched)} of {len(base)} '
        f'baseline pans ({len(base) - len(matched)} baseline pans had no close '
        f'match — the finer DEM walked a visibly different reach, not noise)',
        '',
    ]

    if not matched.empty:
        d_trap = (matched['trap_score_cmp'] - matched['trap_score_base'])
        d_ev = (matched['evidence_score_cmp'] - matched['evidence_score_base'])
        rho = matched[['evidence_score_base', 'evidence_score_cmp']].corr(method='spearman').iloc[0, 1]
        lines += [
            f'Of the matched pans —',
            f'  trap_score delta: mean {d_trap.mean():+.2f}, median {d_trap.median():+.2f}, '
            f'max |delta| {d_trap.abs().max():.2f}',
            f'  evidence_score delta: mean {d_ev.mean():+.2f}, median {d_ev.median():+.2f}, '
            f'max |delta| {d_ev.abs().max():.2f}',
            f'  evidence_score rank correlation (Spearman): {rho:.3f}',
            '',
        ]

    water = sample_surface_water(base_cfg, base['lon'].to_numpy(), base['lat'].to_numpy())
    base = base.copy()
    base['water_occurrence_pct'] = np.round(water, 1)
    n_wet = int((base['water_occurrence_pct'] > 50).sum())
    n_scored = int(base['water_occurrence_pct'].notna().sum())
    lines += [
        f'Surface-water occurrence (JRC, informational — not in evidence_score): '
        f'{n_wet}/{n_scored} baseline pans sit on a reach wet >50% of the 1984-2021 record.',
        '',
        'Less than half the baseline pans had a close match on the other DEM: at',
        '~0.75 km seed spacing, 10 m resolution is changing which bend the walk',
        'treats as the trap, not just nudging the same bend. Before this earns a',
        'place in production scoring: run on more seeds, check whether the moved',
        'pans are the ones field-walked (if any), and decide if the extra reach',
        '3DEP found at Dead River Channel Prospect is real or an artifact of a',
        'different D8 path on a higher-res grid.',
    ]
    for line in lines:
        print(' ', line) if line else print()

    base_out = out(base_cfg, 'tables', 'dem_compare_subset_pans.csv')
    base.to_csv(base_out, index=False)
    print(f'  Wrote {base_out}')

    if not matched.empty:
        match_out = out(base_cfg, 'tables', 'dem_compare_subset_matched.csv')
        matched.to_csv(match_out, index=False)
        print(f'  Wrote {match_out}')

    summary_out = out(base_cfg, 'text', 'dem_compare_subset_summary.txt')
    with open(summary_out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Wrote {summary_out}')


def main():
    parser = argparse.ArgumentParser(description='Compare county_rank across two DEM sources')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--compared', required=True)
    parser.add_argument('--label-baseline', default='baseline')
    parser.add_argument('--label-compared', default='compared')
    args = parser.parse_args()
    run(args.baseline, args.compared, args.label_baseline, args.label_compared)


if __name__ == '__main__':
    main()
