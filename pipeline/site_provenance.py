"""Verify config site names and coordinates against the MRDS extract.

CIM (2018) exploration best practice: legacy data must be validated and its
source tracked. Each config site is matched to MRDS records by
  (a) fuzzy name match after stripping generic words (Placer, Mine, ...), and
  (b) nearest record by great-circle distance,
then classified:
  matched_name_near — name match within near_km (default 5 km)
  matched_name_far  — name match, but every matching record is > near_km away
  nearest_only      — no name match; nearest record reported for context

The table is evidence for a human reviewer. It never moves a site or assigns an
MRDS id by itself.

Usage:
  python -m pipeline.site_provenance --config configs/ne_washington/config.yaml
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.utils import out, resolve_data_path

NEAR_KM = 5.0
GENERIC_WORDS = {
    'placer', 'placers', 'mine', 'mines', 'district', 'river', 'gold',
    'the', 'of', 'and',
}
# Historical/variant spellings folded to one token before comparison.
SPELLING_ALIASES = {'meyers': 'myers', 'meyer': 'myers'}
FUZZY_MIN_RATIO = 0.85
MRDS_URL = 'https://mrdata.usgs.gov/mrds/show-mrds.php?dep_id={}'
EARTH_R_KM = 6371.0088

CLASS_NEAR = 'matched_name_near'
CLASS_FAR = 'matched_name_far'
CLASS_NEAREST = 'nearest_only'


def name_tokens(name, strip_generic=False):
    """Lower-case alphanumeric tokens with spelling aliases applied."""
    toks = re.findall(r'[a-z0-9]+', str(name or '').lower())
    toks = [SPELLING_ALIASES.get(t, t) for t in toks]
    if strip_generic:
        toks = [t for t in toks if t not in GENERIC_WORDS]
    return toks


def _token_hit(tok, candidates):
    if tok in candidates:
        return True
    if len(tok) < 4:
        return False
    return any(
        len(c) >= 4 and c[0] == tok[0]
        and SequenceMatcher(None, tok, c).ratio() >= FUZZY_MIN_RATIO
        for c in candidates
    )


def match_kind(site_name, mrds_name):
    """'exact' | 'fuzzy' | None — how the core tokens of site_name occur in mrds_name."""
    core = name_tokens(site_name, strip_generic=True)
    cand = name_tokens(mrds_name)
    if not core or not cand:
        return None
    if all(t in cand for t in core):
        return 'exact'
    if all(_token_hit(t, cand) for t in core):
        return 'fuzzy'
    # 'San Poil' vs 'Sanpoil': compare with whitespace removed.
    if ''.join(core) in ''.join(cand):
        return 'exact'
    return None


def names_match(site_name, mrds_name):
    """True if every core token of site_name appears (fuzzily) in mrds_name."""
    return match_kind(site_name, mrds_name) is not None


def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km; broadcasts over numpy arrays."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def load_mrds(path):
    """Read an MRDS GeoJSON into a flat DataFrame with lon/lat/url/commodity columns."""
    import geopandas as gpd
    gdf = gpd.read_file(path)
    df = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
    if 'geometry' in gdf and gdf.geometry.notna().all():
        df['lon'] = gdf.geometry.x.to_numpy()
        df['lat'] = gdf.geometry.y.to_numpy()
    elif {'longitude', 'latitude'} <= set(df.columns):
        df['lon'], df['lat'] = df['longitude'], df['latitude']
    df['dep_id'] = df['dep_id'].astype(str)
    if 'url' not in df.columns:
        df['url'] = df['dep_id'].map(MRDS_URL.format)
    if 'commodity' not in df.columns:
        src = next((c for c in ('code_list', 'target_commodity') if c in df.columns), None)
        df['commodity'] = df[src] if src else None
    for col in ('site_name', 'dev_stat'):
        if col not in df.columns:
            df[col] = None
    return df.reset_index(drop=True)


def _record_fields(prefix, row, dist_km):
    return {
        f'{prefix}_dep_id': row['dep_id'],
        f'{prefix}_site_name': row['site_name'],
        f'{prefix}_dev_stat': row['dev_stat'],
        f'{prefix}_commodity': row['commodity'],
        f'{prefix}_url': row['url'],
        f'{prefix}_lon': float(row['lon']),
        f'{prefix}_lat': float(row['lat']),
        f'{prefix}_dist_km': round(float(dist_km), 2),
    }


def verify_site(site, mrds, near_km=NEAR_KM, n_other=5):
    """Return one verification row (dict) for a config site dict."""
    dist = haversine_km(site['lon'], site['lat'],
                        mrds['lon'].to_numpy(), mrds['lat'].to_numpy())
    cfg_id = site.get('mrds_id')
    row = {
        'site_name': site['name'],
        'config_lon': site['lon'],
        'config_lat': site['lat'],
        'config_commodity': site.get('commodity'),
        'config_mrds_id': cfg_id,
        'config_mrds_id_in_extract': (
            bool(cfg_id) and str(cfg_id) in set(mrds['dep_id'])
        ),
        'name_core': ' '.join(name_tokens(site['name'], strip_generic=True)),
    }
    kinds = [match_kind(site['name'], n) for n in mrds['site_name']]
    idx = [i for i, k in enumerate(kinds) if k]
    row['n_name_matches'] = len(idx)

    if idx:
        # Near beats far; within each, an exact token match beats a fuzzy one.
        order = sorted(idx, key=lambda i: (dist[i] > near_km, kinds[i] != 'exact', dist[i]))
        best = order[0]
        row['match_class'] = CLASS_NEAR if dist[best] <= near_km else CLASS_FAR
        row['match_kind'] = kinds[best]
        row.update(_record_fields('match', mrds.iloc[best], dist[best]))
        row['other_name_matches'] = '; '.join(
            f"{mrds.at[i, 'site_name']} ({mrds.at[i, 'dep_id']}, {dist[i]:.1f} km)"
            for i in order[1:1 + n_other]
        )
    else:
        row['match_class'] = CLASS_NEAREST
        row['match_kind'] = None
        row['other_name_matches'] = ''

    nearest = int(np.argmin(dist))
    row.update(_record_fields('nearest', mrds.iloc[nearest], dist[nearest]))

    # Human decisions live in config so a re-run from moved coordinates keeps them.
    review = site.get('mrds_review') or {}
    row['review_decision'] = review.get('decision')
    row['review_dep_id'] = review.get('dep_id')
    row['review_note'] = review.get('note')
    row['config_location_status'] = site.get('location_status')
    return row


COLUMNS = [
    'site_name', 'match_class', 'config_lon', 'config_lat', 'config_commodity',
    'config_mrds_id', 'config_mrds_id_in_extract', 'name_core', 'n_name_matches',
    'match_kind', 'match_dep_id', 'match_site_name', 'match_dev_stat', 'match_commodity',
    'match_url', 'match_lon', 'match_lat', 'match_dist_km', 'other_name_matches',
    'nearest_dep_id', 'nearest_site_name', 'nearest_dev_stat', 'nearest_commodity',
    'nearest_url', 'nearest_lon', 'nearest_lat', 'nearest_dist_km',
    'review_decision', 'review_dep_id', 'review_note', 'config_location_status',
]


def verify_sites(sites, mrds, near_km=NEAR_KM):
    """Verification table for a list of config site dicts."""
    if mrds.empty:
        raise ValueError('MRDS extract is empty — nothing to verify against')
    rows = [verify_site(s, mrds, near_km=near_km) for s in (sites or [])]
    return pd.DataFrame(rows).reindex(columns=COLUMNS)


def run(cfg, near_km=NEAR_KM, out_path=None):
    """Write <outputs_dir>/tables/site_verification.csv and return the table."""
    mrds_path = resolve_data_path(cfg, 'mrds_geojson')
    if not mrds_path or not os.path.exists(mrds_path):
        raise FileNotFoundError(f'MRDS GeoJSON not found: {mrds_path}')
    table = verify_sites(cfg.get('sites') or [], load_mrds(mrds_path), near_km=near_km)
    out_path = out_path or out(cfg, 'tables', 'site_verification.csv')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f'Wrote: {out_path}')
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description='Verify config sites against MRDS')
    parser.add_argument('--config', default='configs/ne_washington/config.yaml')
    parser.add_argument('--near-km', type=float, default=NEAR_KM,
                        help='Max distance for matched_name_near (km)')
    parser.add_argument('--out', default=None, help='Override output CSV path')
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    table = run(cfg, near_km=args.near_km, out_path=args.out)
    if table.empty:
        print('No sites in config.')
        return 0
    show = table[['site_name', 'match_class', 'match_dep_id', 'match_site_name',
                  'match_dist_km', 'nearest_site_name', 'nearest_dist_km']]
    with pd.option_context('display.width', 200, 'display.max_colwidth', 40):
        print(show.to_string(index=False))
    print('\n' + table['match_class'].value_counts().to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
