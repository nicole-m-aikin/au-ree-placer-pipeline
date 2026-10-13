"""Per-county ranking: where to go pan, and where a claim might be stakeable.

Usage:
  python -m pipeline.county_rank --config configs/kern_river/config.yaml

Seeds
  Gold MRDS sites gridded to county_rank.seed_cell_km (one seed per cell,
  producers first, then the densest cells, capped at max_seeds), plus every
  hobby / pamphlet pin when the region has a hobby CSV. Each seed snaps to its
  creek and walks downstream (task11_geometry_walk.build_walk); pans go on the
  best trap vertices (task11_pan_traps.score_reach). Seeds that land on a reach
  already walked are merged, not re-walked.

Evidence per pan, each 0–1
  trap     trap_score / 8.5
  source   gold sites in the upstream corridor of the creek (not the nearest
           dot on the next ridge). MRDS bbox extracts have no deposit type, so
           every gold site counts; side-ridge mines fall outside the corridor.
  chem     best NURE Au/As percentile within chem_radius_km (NaN without NURE)
  Weights renormalise when chem is missing. evidence_score is 0–100.
  Eocene gravel is a bonus of up to 10 points, and only within 5 km. A creek
  in Kern or the desert is not punished for being far from the Sierra channels.
  A pin inside a hydraulic pit or dredge-tailings polygon is then halved.

Two lists, ranked inside each county
  recreational  access_ok == yes (BLM / USFS / state park, claim-free, open)
  stakeable     BLM or USFS land, no active MLRS claim in the buffer, no PAD-US
                withdrawal hint, not tribal. A claim BLM closed in the last
                year nearby adds a bonus — someone held it, then let it go.

Both are screens. Nothing here says there is gold in the creek, and nothing
here says ground is open to mineral entry: BLM master title plats, the MLRS
case record and the county recorder decide that.
"""

from __future__ import annotations

import argparse
import math
import os
import warnings

import numpy as np
import pandas as pd
import yaml

from pipeline.utils import ensure_outputs, out, resolve_data_path

TRAP_MAX = 8.5
SOURCE_SCALE_KM = 3.0
W_TRAP, W_SOURCE, W_CHEM = 0.45, 0.35, 0.20
CLOSED_CLAIM_BONUS = 10.0
CHANNEL_BONUS = 10.0
CHANNEL_NEAR_KM = 5.0
MINED_FACTOR = 0.5
UPSTREAM_SCALE_KM = 6.0
UPSTREAM_HALF_WIDTH_KM = 2.5
UPSTREAM_MAX_KM = 12.0
CHANNEL_SCALE_KM = 2.0
KM_PER_DEG = 111.32

DEFAULTS = {
    'seed_cell_km': 3.0,
    'max_seeds': 300,
    'up_km': 0.5,
    'down_km': 4.0,
    'snap_m': 600.0,
    'min_snap_km2': 1.0,
    'pans_per_walk': 2,
    'chem_radius_km': 3.0,
    'chem_elements': ['Au', 'As'],
    'closed_claim_radius_m': 300.0,
    # Sample EE catalog layers at each pin (filters/flags; no score weights yet).
    'gee_sample': True,
    # Creek walks on MERIT Hydro in Earth Engine (no local pysheds D8).
    'ee_trace': True,
}

# PAD-US designation types that usually mean the ground is withdrawn from
# mineral entry. A hint only — withdrawals live on the master title plats.
WITHDRAWN_DES = {
    'WA': 'wilderness',
    'NP': 'national park',
    'NM': 'national monument',
    'NCA': 'national conservation area',
    'WSR': 'wild & scenic river corridor',
    'RNA': 'research natural area',
    'NWR': 'national wildlife refuge',
    'MIL': 'military land',
}
CAUTION_DES = {
    'WSA': 'wilderness study area (non-impairment rules)',
    'ACEC': 'area of critical environmental concern',
    'IRA': 'inventoried roadless area',
}
FEDERAL_MINERAL_MANAGERS = frozenset({'BLM', 'USFS'})

STAKE_STATUSES = (
    'open_screen', 'claimed', 'withdrawn_hint', 'not_federal', 'mixed_manager',
    'tribal', 'unchecked',
)
FEDERAL_ACCESS_TYPES = frozenset({'blm', 'usfs'})


def rank_cfg(cfg):
    merged = dict(DEFAULTS)
    merged.update(cfg.get('county_rank') or {})
    return merged


def _short(cfg):
    return str((cfg.get('study_area') or {}).get('short') or 'study')


def to_km_xy(lon, lat, lat0):
    """Equirectangular km about lat0 — fine for nearest-neighbour inside one region."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    return np.column_stack([
        lon * KM_PER_DEG * math.cos(math.radians(lat0)),
        lat * KM_PER_DEG,
    ])


# ── Seeds ─────────────────────────────────────────────────────────────────────

def grid_seeds(mrds, cell_km, max_seeds, lat0=None):
    """One seed per cell_km grid cell of gold MRDS.

    The seed is a real site (a producer when the cell has one), not a centroid,
    so it sits where the mine was. Cells rank by producers, then site count.
    """
    if mrds is None or len(mrds) == 0:
        return pd.DataFrame(columns=[
            'seed_kind', 'name', 'creek_name', 'lon', 'lat', 'n_sites', 'n_producers',
        ])
    df = mrds.copy()
    lat0 = float(df['lat'].mean()) if lat0 is None else float(lat0)
    xy = to_km_xy(df['lon'], df['lat'], lat0)
    df['_ix'] = np.floor(xy[:, 0] / cell_km).astype(int)
    df['_iy'] = np.floor(xy[:, 1] / cell_km).astype(int)
    dev = df.get('dev_stat', pd.Series('', index=df.index)).fillna('').astype(str)
    df['_prod'] = dev.str.contains('Producer', case=False)
    rows = []
    for _, grp in df.groupby(['_ix', '_iy'], sort=False):
        prod = grp[grp['_prod']]
        pick = (prod if len(prod) else grp).iloc[0]
        rows.append({
            'seed_kind': 'mrds',
            'name': str(pick.get('site_name') or pick.get('dep_id') or 'MRDS gold'),
            'creek_name': '',
            'lon': float(pick['lon']),
            'lat': float(pick['lat']),
            'n_sites': int(len(grp)),
            'n_producers': int(grp['_prod'].sum()),
        })
    seeds = pd.DataFrame(rows).sort_values(
        ['n_producers', 'n_sites'], ascending=[False, False], kind='mergesort',
    )
    return seeds.head(int(max_seeds)).reset_index(drop=True)


def hobby_seeds(hobby):
    if hobby is None or len(hobby) == 0:
        return pd.DataFrame()
    return pd.DataFrame({
        'seed_kind': 'hobby',
        'name': hobby.get('name', hobby.get('report_id')).astype(str).values,
        'creek_name': hobby.get('creek_name', pd.Series('', index=hobby.index))
        .fillna('').astype(str).values,
        'lon': pd.to_numeric(hobby['lon'], errors='coerce').values,
        'lat': pd.to_numeric(hobby['lat'], errors='coerce').values,
        'n_sites': 0,
        'n_producers': 0,
    }).dropna(subset=['lon', 'lat'])


def seed_specs(seeds, rc, short):
    """build_walk specs. Hobby pins first so a named spot owns its reach."""
    order = seeds.assign(_k=(seeds['seed_kind'] != 'hobby').astype(int))
    order = order.sort_values('_k', kind='mergesort').drop(columns='_k')
    specs = []
    for i, s in enumerate(order.itertuples(index=False), 1):
        specs.append({
            'walk_id': f'{short}:w{i:04d}',
            'area': s.seed_kind,
            'report_id': '',
            'anchor_name': s.name,
            'creek_name': s.creek_name,
            'anchor_lon': float(s.lon),
            'anchor_lat': float(s.lat),
            'up_km': float(rc['up_km']),
            'down_km': float(rc['down_km']),
            'stop_above_km2': None,
            'snap_m': float(rc['snap_m']),
            'min_snap_km2': float(rc['min_snap_km2']),
            'seed_kind': s.seed_kind,
            'seed_n_sites': int(s.n_sites),
            'seed_n_producers': int(s.n_producers),
        })
    return specs


# ── Evidence ─────────────────────────────────────────────────────────────────

def source_component(nearest_km, scale_km=SOURCE_SCALE_KM):
    d = np.asarray(nearest_km, dtype=float)
    return np.where(np.isfinite(d), np.exp(-np.maximum(d, 0.0) / scale_km), 0.0)


def bearing_upstream_deg(line, lon, lat, lat0):
    """Degrees clockwise from north, toward the upstream end of a reach.

    Reach lines are ordered upstream-end to downstream. The bearing is from
    the vertex nearest the pin toward the vertex upstream of it.
    """
    coords = np.asarray(line.coords, dtype=float)
    if len(coords) < 2:
        return np.nan
    xy = to_km_xy(coords[:, 0], coords[:, 1], lat0)
    p = to_km_xy([lon], [lat], lat0)[0]
    i = int(np.argmin(np.hypot(xy[:, 0] - p[0], xy[:, 1] - p[1])))
    j = i - 1 if i > 0 else 1
    vec = xy[j] - xy[i] if i > 0 else xy[0] - xy[1]
    norm = float(np.hypot(vec[0], vec[1]))
    if norm < 1e-9:
        return np.nan
    return float(np.degrees(np.arctan2(vec[0], vec[1])) % 360.0)


def corridor_alongs(pan_lon, pan_lat, bearing_deg, site_lon, site_lat, lat0,
                    half_width_km=UPSTREAM_HALF_WIDTH_KM, max_km=UPSTREAM_MAX_KM):
    """Along-track km of sites upstream of one pin, inside the creek corridor."""
    if not np.isfinite(bearing_deg):
        return np.array([])
    site_lon = np.asarray(site_lon, dtype=float)
    site_lat = np.asarray(site_lat, dtype=float)
    if len(site_lon) == 0:
        return np.array([])
    dx = (site_lon - pan_lon) * KM_PER_DEG * math.cos(math.radians(lat0))
    dy = (site_lat - pan_lat) * KM_PER_DEG
    brad = math.radians(float(bearing_deg))
    ux, uy = math.sin(brad), math.cos(brad)
    along = dx * ux + dy * uy
    cross = dx * (-uy) + dy * ux
    keep = (along >= 0.0) & (along <= max_km) & (np.abs(cross) <= half_width_km)
    return along[keep]


def upstream_source_component(alongs_km, scale_km=UPSTREAM_SCALE_KM):
    """0–1. Nearest upstream site dominates; more sites saturate at 1."""
    alongs = np.asarray(alongs_km, dtype=float)
    alongs = alongs[np.isfinite(alongs)]
    if len(alongs) == 0:
        return 0.0
    return float(np.clip(np.exp(-alongs / scale_km).sum(), 0.0, 1.0))


def channel_component(dist_km, scale_km=CHANNEL_SCALE_KM):
    """1 on the Eocene gravel, decaying with distance. NaN where unknown."""
    d = np.asarray(dist_km, dtype=float)
    return np.where(np.isfinite(d), np.exp(-np.maximum(d, 0.0) / scale_km), np.nan)


def channel_bonus(dist_km, near_km=CHANNEL_NEAR_KM, points=CHANNEL_BONUS):
    """Up to `points` added when the creek is near the Eocene gravel.

    Farther than near_km the bonus is 0. The base score is unchanged, so a
    desert or Kern pin is not penalised for missing a Sierra unit.
    """
    d = np.asarray(dist_km, dtype=float)
    comp = channel_component(d)
    near = np.isfinite(d) & (d <= near_km)
    return np.where(near, points * comp, 0.0)


def nure_sample_scores(nure, elements):
    """Mean within-region percentile of the chosen elements, per NURE sample."""
    cols = [e for e in elements if e in nure.columns and nure[e].notna().sum() >= 10]
    if not cols:
        return None
    pct = pd.concat([nure[c].rank(pct=True) for c in cols], axis=1)
    return pct.mean(axis=1, skipna=True)


def chem_component(pan_lon, pan_lat, nure, elements, radius_km, lat0):
    """Best NURE pathfinder percentile within radius_km of each pan. NaN if none."""
    n = len(pan_lon)
    if nure is None or len(nure) == 0:
        return np.full(n, np.nan)
    scores = nure_sample_scores(nure, elements)
    if scores is None:
        return np.full(n, np.nan)
    from scipy.spatial import cKDTree

    ok = scores.notna().to_numpy()
    if not ok.any():
        return np.full(n, np.nan)
    tree = cKDTree(to_km_xy(nure['lon'].to_numpy()[ok], nure['lat'].to_numpy()[ok], lat0))
    vals = scores.to_numpy()[ok]
    out_ = np.full(n, np.nan)
    for i, hits in enumerate(tree.query_ball_point(to_km_xy(pan_lon, pan_lat, lat0), radius_km)):
        if hits:
            out_[i] = float(np.max(vals[hits]))
    return out_


def evidence_score(trap, source, chem):
    """0–100 weighted mean. Chem weight drops out where chem is NaN."""
    trap = np.clip(np.asarray(trap, dtype=float) / TRAP_MAX, 0.0, 1.0)
    source = np.asarray(source, dtype=float)
    chem = np.asarray(chem, dtype=float)
    has_chem = np.isfinite(chem)
    num = W_TRAP * trap + W_SOURCE * source + np.where(has_chem, W_CHEM * chem, 0.0)
    den = W_TRAP + W_SOURCE + np.where(has_chem, W_CHEM, 0.0)
    return np.round(100.0 * num / den, 1)


# ── Stakeable screen ─────────────────────────────────────────────────────────

def padus_overlap_flags(points, padus):
    """Per point: on_fee_land, federal_manager, withdrawal_hint, caution_note.

    Uses every PAD-US polygon under the point, not just the smallest one —
    a wild & scenic corridor sits on top of the national forest fee polygon.
    A Proclamation or Designation boundary with no Fee polygon underneath is
    usually a private inholding, so it is neither open nor federal here.
    """
    import geopandas as gpd

    n = len(points)
    on_fee = [False] * n
    fed = [''] * n
    withdrawn = [''] * n
    caution = [''] * n
    if padus is None or len(padus) == 0 or n == 0:
        return pd.DataFrame({'on_fee_land': on_fee, 'federal_manager': fed,
                             'withdrawal_hint': withdrawn, 'caution_note': caution})
    pts = gpd.GeoDataFrame({'_i': np.arange(n)}, geometry=points.geometry.values,
                           crs=points.crs or 'EPSG:4326')
    keep = [c for c in ('Mang_Name', 'Des_Tp', 'GAP_Sts', 'Category', 'Unit_Nm')
            if c in padus.columns]
    joined = gpd.sjoin(pts, padus[keep + ['geometry']], how='inner', predicate='intersects')
    for i, grp in joined.groupby('_i'):
        mang = grp.get('Mang_Name', pd.Series(dtype=str)).fillna('').astype(str).str.upper()
        cat = grp.get('Category', pd.Series('Fee', index=grp.index)).fillna('').astype(str)
        on_fee[i] = bool((cat == 'Fee').any())
        fed_hit = sorted(set(mang[(cat == 'Fee') & mang.isin(FEDERAL_MINERAL_MANAGERS)]))
        fed[i] = ';'.join(fed_hit)
        des = grp.get('Des_Tp', pd.Series(dtype=str)).fillna('').astype(str).str.upper()
        gap = grp.get('GAP_Sts', pd.Series(dtype=str)).fillna('').astype(str)
        unit = grp.get('Unit_Nm', pd.Series('', index=grp.index)).fillna('').astype(str)
        w = [f'{WITHDRAWN_DES[d]} ({u})' if u else WITHDRAWN_DES[d]
             for d, u in zip(des, unit) if d in WITHDRAWN_DES]
        if (gap == '1').any() and not w:
            w.append('GAP 1 protected area')
        withdrawn[i] = '; '.join(dict.fromkeys(w))
        c = [CAUTION_DES[d] for d in des if d in CAUTION_DES]
        caution[i] = '; '.join(dict.fromkeys(c))
    return pd.DataFrame({'on_fee_land': on_fee, 'federal_manager': fed,
                         'withdrawal_hint': withdrawn, 'caution_note': caution})


def stake_status(tribal_land, claim_status, withdrawn, federal, access_type='blm'):
    """First failing gate wins. mixed_manager = a BLM/USFS fee polygon overlaps a
    tighter unit run by someone else (state park, restricted area) — check it."""
    if bool(tribal_land):
        return 'tribal'
    if str(claim_status) == 'claimed':
        return 'claimed'
    if withdrawn:
        return 'withdrawn_hint'
    if not federal:
        return 'not_federal'
    if str(access_type) not in FEDERAL_ACCESS_TYPES:
        return 'mixed_manager'
    if str(claim_status) == 'unchecked':
        return 'unchecked'
    return 'open_screen'


def closed_claim_hits(points, closed, radius_m):
    """Name of a claim BLM closed in the last year within radius_m, else ''."""
    n = len(points)
    if closed is None or len(closed) == 0 or n == 0:
        return [''] * n
    import geopandas as gpd

    from pipeline.geo_crs import utm_epsg

    epsg = utm_epsg(float(points.geometry.x.mean()), float(points.geometry.y.mean()))
    buf = closed.to_crs(epsg)
    buf['geometry'] = buf.buffer(float(radius_m))
    pts = gpd.GeoDataFrame({'_i': np.arange(n)}, geometry=points.geometry.values,
                           crs=points.crs or 'EPSG:4326').to_crs(epsg)
    joined = gpd.sjoin(pts, buf[['CSE_NAME', 'geometry']] if 'CSE_NAME' in buf.columns
                       else buf[['geometry']], how='inner', predicate='intersects')
    names = [''] * n
    for i, grp in joined.groupby('_i'):
        nm = grp['CSE_NAME'].dropna().astype(str) if 'CSE_NAME' in grp.columns else []
        names[i] = str(nm.iloc[0]) if len(nm) else 'closed claim'
    return names


# ── Counties + ranks ─────────────────────────────────────────────────────────

def assign_counties(points, counties):
    import geopandas as gpd

    n = len(points)
    if counties is None or len(counties) == 0 or n == 0:
        return [''] * n, [''] * n
    pts = gpd.GeoDataFrame({'_i': np.arange(n)}, geometry=points.geometry.values,
                           crs=points.crs or 'EPSG:4326')
    joined = gpd.sjoin(pts, counties[['GEOID', 'NAME', 'geometry']], how='left',
                       predicate='within').drop_duplicates('_i').set_index('_i')
    name = joined['NAME'].reindex(range(n)).fillna('').astype(str).tolist()
    geoid = joined['GEOID'].reindex(range(n)).fillna('').astype(str).tolist()
    return name, geoid


def _text(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ''
    text = str(value).strip()
    return '' if text.lower() in ('', 'nan', 'none') else text


def rank_within_county(df, eligible, score_col, rank_col):
    df[rank_col] = pd.Series(pd.NA, index=df.index, dtype='Int64')
    pool = df[eligible & (df['county'] != '')]
    for _, grp in pool.groupby('county'):
        order = grp.sort_values([score_col, 'trap_score'], ascending=[False, False])
        df.loc[order.index, rank_col] = np.arange(1, len(order) + 1)
    return df


def score_pans(pans, nure, gold_xy, rc, lat0, upstream_alongs=None,
               channel_km=None, on_mined=None):
    """Add gold distance, source/channel/chem components, and evidence_score.

    upstream_alongs: list of along-track km arrays, one per pan. When given,
    source_c is the upstream corridor. Otherwise source_c is nearest-dot decay.
    on_mined halves the score. None means the mined layer was not checked.
    """
    from scipy.spatial import cKDTree

    lon, lat = pans['lon'].to_numpy(), pans['lat'].to_numpy()
    pts = to_km_xy(lon, lat, lat0)
    if gold_xy is not None and len(gold_xy):
        tree = cKDTree(to_km_xy(gold_xy[:, 0], gold_xy[:, 1], lat0))
        d, _ = tree.query(pts)
        n2 = np.array([len(h) for h in tree.query_ball_point(pts, 2.0)])
    else:
        d = np.full(len(pans), np.nan)
        n2 = np.zeros(len(pans), dtype=int)
    pans['nearest_gold_km'] = np.round(d, 2)
    pans['n_gold_mrds_2km'] = n2
    if upstream_alongs is None:
        pans['source_c'] = np.round(source_component(d), 3)
        pans['upstream_gold_km'] = np.nan
        pans['n_upstream_gold'] = 0
    else:
        src, up_km, n_up = [], [], []
        for alongs in upstream_alongs:
            alongs = np.asarray(alongs, dtype=float)
            src.append(upstream_source_component(alongs))
            n_up.append(int(np.isfinite(alongs).sum()) if len(alongs) else 0)
            up_km.append(float(np.min(alongs)) if len(alongs) else np.nan)
        pans['source_c'] = np.round(src, 3)
        pans['upstream_gold_km'] = np.round(up_km, 2)
        pans['n_upstream_gold'] = n_up
    chem = chem_component(lon, lat, nure, rc['chem_elements'], float(rc['chem_radius_km']), lat0)
    pans['chem_c'] = np.round(chem, 3)
    if channel_km is None:
        channel_km = np.full(len(pans), np.nan)
    pans['channel_km'] = np.round(np.asarray(channel_km, dtype=float), 2)
    channel = channel_component(pans['channel_km'])
    pans['channel_c'] = np.round(channel, 3)
    evidence = evidence_score(pans['trap_score'], pans['source_c'], chem)
    evidence = np.minimum(100.0, evidence + channel_bonus(pans['channel_km']))
    if on_mined is None:
        pans['on_mined'] = False
        pans['evidence_score'] = np.round(evidence, 1)
    else:
        mined = np.asarray(on_mined, dtype=bool)
        pans['on_mined'] = mined
        pans['evidence_score'] = np.round(
            np.where(mined, evidence * MINED_FACTOR, evidence), 1)
    return pans


def upstream_alongs_for_pans(pans, reaches, gold, lat0):
    """Along-track km of gold sites in each pin's upstream corridor.

    reaches is a list of dicts or a GeoDataFrame with walk_id and geometry.
    gold is a DataFrame with lon, lat. Missing bearings yield an empty corridor
    so that pin falls back to a zero source rather than a side-ridge dot.
    """
    if hasattr(reaches, 'iterrows'):
        by_id = {row.walk_id: row.geometry for row in reaches.itertuples(index=False)}
    else:
        by_id = {r['walk_id']: r['geometry'] for r in reaches}
    if gold is None or len(gold) == 0:
        site_lon = np.array([])
        site_lat = np.array([])
    else:
        site_lon = gold['lon'].to_numpy(dtype=float)
        site_lat = gold['lat'].to_numpy(dtype=float)
    alongs = []
    for p in pans.itertuples(index=False):
        geom = by_id.get(p.walk_id)
        if geom is None:
            alongs.append(np.array([]))
            continue
        bearing = bearing_upstream_deg(geom, float(p.lon), float(p.lat), lat0)
        alongs.append(corridor_alongs(
            float(p.lon), float(p.lat), bearing, site_lon, site_lat, lat0))
    return alongs


# ── Run ──────────────────────────────────────────────────────────────────────

def _load_mrds(cfg):
    import geopandas as gpd

    path = resolve_data_path(cfg, 'mrds_geojson')
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=['lon', 'lat'])
    g = gpd.read_file(path)
    if 'target_commodity' in g.columns:
        tc = g['target_commodity'].fillna('')
        g = g[tc.str.contains('Gold') & ~tc.str.contains('Rare Earth')]
    df = pd.DataFrame(g.drop(columns='geometry'))
    df['lon'] = g.geometry.x.values
    df['lat'] = g.geometry.y.values
    return df


def _load_nure(cfg):
    from pipeline.utils import load_nure

    path = resolve_data_path(cfg, 'nure_csv')
    if not path or not os.path.exists(path):
        return None
    df = load_nure({'data': {'nure_csv': path}})
    return df.dropna(subset=['lat', 'lon'])


def _load_hobby(cfg):
    path = resolve_data_path(cfg, 'hobby_reports_csv')
    if not path or not os.path.exists(path):
        return None
    return pd.read_csv(path)


def walk_seeds(specs, fdir_path, acc_path, dem_path, pans_per_walk):
    """Trace every seed. Returns (reaches, pan rows). Seeds on a walked reach merge."""
    from pipeline.task11_geometry_walk import build_walk, pick_pans
    from pipeline.task11_pan_traps import why_flags

    reaches, pan_rows, taken_xy = [], [], []
    walked = set()
    for k, spec in enumerate(specs, 1):
        built = build_walk(spec, fdir_path, acc_path, dem_path)
        if built is None:
            continue
        reach, vertices = built
        key = (round(reach['snap_lon'], 5), round(reach['snap_lat'], 5))
        if key in walked:
            continue
        walked.update((round(x, 5), round(y, 5)) for x, y in reach['geometry'].coords)
        reaches.append(reach)
        picked = pick_pans(vertices, int(pans_per_walk), taken_xy)
        for _, p in picked.iterrows():
            taken_xy.append((float(p['x']), float(p['y'])))
            pan_rows.append({
                'walk_id': spec['walk_id'],
                'seed_kind': spec['seed_kind'],
                'seed_name': spec['anchor_name'],
                'creek_name': spec['creek_name'],
                'lon': round(float(p['lon']), 6),
                'lat': round(float(p['lat']), 6),
                'elev_m': None if not np.isfinite(p['z']) else round(float(p['z']), 1),
                'along_km': float(p['along_km']),
                'upstream_km2': float(p['upstream_km2']),
                'trap_score': round(float(p['trap_score']), 2),
                'why': why_flags(p),
            })
        if k % 50 == 0:
            print(f'    walked {k}/{len(specs)} seeds, {len(reaches)} reaches', flush=True)
    return reaches, pan_rows


PAN_COLS = [
    'pin_id', 'region', 'county', 'county_geoid', 'lon', 'lat', 'elev_m',
    'creek_name', 'seed_kind', 'seed_name', 'walk_id', 'along_km', 'upstream_km2',
    'trap_score', 'why', 'nearest_gold_km', 'n_gold_mrds_2km',
    'upstream_gold_km', 'n_upstream_gold', 'source_c',
    'channel_km', 'channel_c', 'on_mined', 'mined_name', 'chem_c',
    'evidence_score',
    'access_type', 'access_ok', 'access_reason', 'manager_name', 'claim_status',
    'pub_access', 'tribal_land', 'tribal_name', 'rules_note', 'on_fee_land',
    'rec_score', 'rec_rank_in_county',
    'federal_manager', 'withdrawal_hint', 'caution_note', 'recent_closed_claim',
    'stake_status', 'stake_score', 'stake_rank_in_county',
    'elev10_m', 'slope10_deg', 'water_pct', 'water_regime',
    'lith_code', 'lith_name', 'lith_favor', 'landform_code',
    'nlcd', 'nlcd_name', 'land_filter', 'road_m', 'fire_year', 'has_lidar_1m',
    'best_months',
]


def _score_with_layers(pans, reaches, mrds, gold_xy, cfg, rc, lat0):
    """Upstream gold, Eocene gravel, mined-ground penalty, then evidence."""
    import geopandas as gpd

    from pipeline.placer_layers import ensure_placer_layers, pin_flags

    alongs = upstream_alongs_for_pans(pans, reaches, mrds, lat0)
    try:
        ensure_placer_layers()
    except Exception as exc:
        warnings.warn(f'  Placer layers unavailable ({exc}); scoring without them',
                      stacklevel=2)
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(pans['lon'], pans['lat']), crs='EPSG:4326')
    flags = pin_flags(pts)
    channel = flags['channel_km'].to_numpy() if flags['channel_checked'].iloc[0] else None
    mined = flags['on_mined'].to_numpy() if flags['mined_checked'].iloc[0] else None
    pans = score_pans(
        pans, _load_nure(cfg), gold_xy, rc, lat0,
        upstream_alongs=alongs, channel_km=channel, on_mined=mined,
    )
    pans['mined_name'] = flags['mined_name'].to_numpy()
    if flags['mined_checked'].iloc[0]:
        print(f'  Mined ground: {int(pans["on_mined"].sum())} pins', flush=True)
    if flags['channel_checked'].iloc[0]:
        print(
            f'  Eocene gravel within 2 km: {(pans["channel_km"] <= 2).sum()} pins',
            flush=True,
        )
    return pans


def rescore(cfg):
    """Recompute evidence on pans already walked. Does not retrace creeks."""
    import geopandas as gpd

    rc = rank_cfg(cfg)
    lat0 = float(np.mean([cfg['study_area']['bbox']['lat_min'],
                          cfg['study_area']['bbox']['lat_max']]))
    csv_path = out(cfg, 'tables', 'county_rank_pans.csv')
    walk_path = out(cfg, 'geojson', 'county_rank_walks.geojson')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    pans = pd.read_csv(csv_path)
    walks = gpd.read_file(walk_path) if os.path.exists(walk_path) else []
    mrds = _load_mrds(cfg)
    gold_xy = mrds[['lon', 'lat']].to_numpy() if len(mrds) else None
    was_rec = pans['rec_score'].notna() if 'rec_score' in pans.columns else False
    was_stake = pans['stake_score'].notna() if 'stake_score' in pans.columns else False
    pans = _score_with_layers(pans, walks, mrds, gold_xy, cfg, rc, lat0)
    pans['rec_score'] = np.where(was_rec, pans['evidence_score'], np.nan)
    closed = pans['recent_closed_claim'].fillna('').astype(str).str.len().gt(0) \
        if 'recent_closed_claim' in pans.columns else False
    bonus = np.where(closed, CLOSED_CLAIM_BONUS, 0.0)
    pans['stake_score'] = np.where(
        was_stake, np.minimum(100.0, pans['evidence_score'] + bonus), np.nan)
    gdf = gpd.GeoDataFrame(
        pans, geometry=gpd.points_from_xy(pans['lon'], pans['lat']), crs='EPSG:4326')
    gdf = rank_within_county(gdf, was_rec, 'rec_score', 'rec_rank_in_county')
    gdf = rank_within_county(gdf, was_stake, 'stake_score', 'stake_rank_in_county')
    _write_outputs(cfg, gdf, None)
    return gdf


def run(cfg):
    import geopandas as gpd

    from pipeline.fetch_region import counties_path, mlrs_closed_path, state_fips
    from pipeline.task11_geometry_walk import ensure_flowdir_tif
    from pipeline.task11_land_access import _load_gdf, annotate_points, padus_path
    from pipeline.task11_pan_traps import ensure_accumulation_tif

    ensure_outputs(cfg['outputs_dir'])
    rc = rank_cfg(cfg)
    short = _short(cfg)

    mrds = _load_mrds(cfg)
    hobby = _load_hobby(cfg)
    lat0 = float(np.mean([cfg['study_area']['bbox']['lat_min'],
                          cfg['study_area']['bbox']['lat_max']]))
    seeds = pd.concat([
        hobby_seeds(hobby),
        grid_seeds(mrds, float(rc['seed_cell_km']), int(rc['max_seeds']), lat0),
    ], ignore_index=True)
    if seeds.empty:
        print('  No MRDS gold sites or hobby pins — nothing to seed.')
        return None
    specs = seed_specs(seeds, rc, short)
    print(f'  {len(specs)} seeds ({(seeds["seed_kind"] == "hobby").sum()} hobby, '
          f'{(seeds["seed_kind"] == "mrds").sum()} MRDS cells from {len(mrds)} gold sites)',
          flush=True)

    from pipeline.gee_creek_trace import ee_trace_enabled, gee_project as _gee_proj, walk_seeds_ee
    if ee_trace_enabled(cfg):
        print('  Creek walks: Earth Engine MERIT Hydro', flush=True)
        reaches, pan_rows = walk_seeds_ee(
            specs, project=_gee_proj(cfg),
            pans_per_walk=int(rc['pans_per_walk']),
            up_km=float(rc['up_km']), down_km=float(rc['down_km']),
            snap_m=float(rc['snap_m']), min_snap_km2=float(rc['min_snap_km2']),
        )
    else:
        dem_path = resolve_data_path(cfg, 'dem_tif')
        if not dem_path or not os.path.exists(dem_path):
            raise FileNotFoundError(
                f'No DEM at {dem_path}. Run: python -m pipeline.fetch_region '
                f'--config <config> (or set county_rank.ee_trace: true)')
        print('  Creek walks: local pysheds D8', flush=True)
        acc_path = ensure_accumulation_tif(
            dem_path, out(cfg, 'geojson', 'task11_d8_accumulation.tif'))
        fdir_path = ensure_flowdir_tif(
            dem_path, out(cfg, 'geojson', 'task11_d8_flowdir.tif'))
        reaches, pan_rows = walk_seeds(
            specs, fdir_path, acc_path, dem_path, rc['pans_per_walk'])
    if not pan_rows:
        print('  No pans picked — every seed missed a channel.')
        return None
    pans = pd.DataFrame(pan_rows)
    pans.insert(0, 'region', short)
    pans.insert(0, 'pin_id', [f'{short}:{i:05d}' for i in range(1, len(pans) + 1)])

    gold_xy = mrds[['lon', 'lat']].to_numpy() if len(mrds) else None
    pans = _score_with_layers(pans, reaches, mrds, gold_xy, cfg, rc, lat0)

    gdf = gpd.GeoDataFrame(pans, geometry=gpd.points_from_xy(pans['lon'], pans['lat']),
                           crs='EPSG:4326')
    hobby_gdf = None
    if hobby is not None and len(hobby):
        hobby_gdf = gpd.GeoDataFrame(
            hobby, geometry=gpd.points_from_xy(hobby['lon'], hobby['lat']), crs='EPSG:4326')
    gdf = annotate_points(gdf, cfg, hobby_gdf=hobby_gdf)

    flags = padus_overlap_flags(gdf, _load_gdf(padus_path(cfg)))
    for col in flags.columns:
        gdf[col] = flags[col].to_numpy()
    gdf['recent_closed_claim'] = closed_claim_hits(
        gdf, _load_gdf(mlrs_closed_path(cfg)), rc['closed_claim_radius_m'])
    gdf['stake_status'] = [
        stake_status(t, c, w, f, a) for t, c, w, f, a in
        zip(gdf['tribal_land'], gdf['claim_status'], gdf['withdrawal_hint'],
            gdf['federal_manager'], gdf['access_type'])
    ]

    counties = _load_gdf(counties_path(state_fips(cfg)))
    gdf['county'], gdf['county_geoid'] = assign_counties(gdf, counties)

    from pipeline.gee_sample import (
        apply_sample_flags, gee_project, gee_sample_enabled, sample_pins,
    )
    if gee_sample_enabled(cfg):
        try:
            samples = sample_pins(gdf[['pin_id', 'lon', 'lat']], project=gee_project(cfg))
            samples.to_csv(out(cfg, 'tables', 'county_rank_gee_samples.csv'), index=False)
            gdf = apply_sample_flags(gdf, samples)
            print(
                f'  EE samples: land_drop='
                f'{(gdf["land_filter"] == "drop").sum()} '
                f'dry={(gdf["water_regime"] == "dry").sum()} '
                f'burned={gdf["fire_year"].notna().sum()} '
                f'lidar1m={gdf["has_lidar_1m"].sum()}',
                flush=True,
            )
        except Exception as exc:
            warnings.warn(f'  EE sample failed ({exc}); ranking without EE flags',
                          stacklevel=2)

    land_ok = (
        gdf['land_filter'].fillna('ok').astype(str).ne('drop')
        if 'land_filter' in gdf.columns else True
    )
    rec_ok = (
        gdf['access_ok'].astype(str).str.lower().eq('yes')
        & gdf['on_fee_land']
        & land_ok
    )
    gdf['rec_score'] = np.where(rec_ok, gdf['evidence_score'], np.nan)
    stake_ok = gdf['stake_status'].eq('open_screen')
    bonus = np.where(gdf['recent_closed_claim'].astype(bool), CLOSED_CLAIM_BONUS, 0.0)
    gdf['stake_score'] = np.where(
        stake_ok, np.minimum(100.0, gdf['evidence_score'] + bonus), np.nan)
    gdf = rank_within_county(gdf, rec_ok, 'rec_score', 'rec_rank_in_county')
    gdf = rank_within_county(gdf, stake_ok, 'stake_score', 'stake_rank_in_county')

    _write_outputs(cfg, gdf, reaches)
    return gdf


def _write_outputs(cfg, gdf, reaches):
    import geopandas as gpd

    cols = [c for c in PAN_COLS if c in gdf.columns]
    gdf[cols].to_csv(out(cfg, 'tables', 'county_rank_pans.csv'), index=False)
    gdf[cols + ['geometry']].to_file(out(cfg, 'geojson', 'county_rank_pans.geojson'),
                                     driver='GeoJSON')
    rec = gdf[gdf['rec_rank_in_county'].notna()].sort_values(['county', 'rec_rank_in_county'])
    rec[cols].to_csv(out(cfg, 'tables', 'county_rank_recreational.csv'), index=False)
    stake = gdf[gdf['stake_rank_in_county'].notna()].sort_values(
        ['county', 'stake_rank_in_county'])
    stake[cols].to_csv(out(cfg, 'tables', 'county_rank_stakeable.csv'), index=False)
    if reaches:
        walk_cols = ['walk_id', 'seed_kind', 'anchor_name', 'creek_name', 'snap_lon',
                     'snap_lat', 'reach_km', 'up_stop', 'down_stop', 'geometry']
        w = gpd.GeoDataFrame(reaches, crs='EPSG:4326')
        w[[c for c in walk_cols if c in w.columns]].to_file(
            out(cfg, 'geojson', 'county_rank_walks.geojson'), driver='GeoJSON')
    _write_summary(cfg, gdf, rec, stake)


def _write_summary(cfg, gdf, rec, stake, top=5):
    name = (cfg.get('study_area') or {}).get('name', _short(cfg))
    lines = [
        f'COUNTY RANKING — {name}',
        'A screen, not a gold claim and not a land-status opinion.',
        '=' * 68,
        '',
        f'Pans scored: {len(gdf)}   recreational-ok: {len(rec)}   '
        f'stakeable screen: {len(stake)}',
        'stake_status counts: ' + ', '.join(
            f'{k}={v}' for k, v in gdf['stake_status'].value_counts().items()),
        f'Chemistry available on {int(gdf["chem_c"].notna().sum())} of {len(gdf)} pans.',
    ]
    if 'on_mined' in gdf.columns:
        lines.append(f'Mined ground (score halved): {int(gdf["on_mined"].astype(bool).sum())}.')
    if 'channel_km' in gdf.columns:
        near = int((pd.to_numeric(gdf['channel_km'], errors='coerce') <= 2).sum())
        lines.append(f'Within 2 km of Eocene gravel: {near}.')
    if 'land_filter' in gdf.columns:
        lines.append(
            'EE flags: land_drop='
            f'{int((gdf["land_filter"] == "drop").sum())}  '
            f'dry={int((gdf["water_regime"] == "dry").sum())}  '
            f'seasonal={int((gdf["water_regime"] == "seasonal").sum())}  '
            f'year_round={int((gdf["water_regime"] == "year_round").sum())}  '
            f'burned={int(gdf["fire_year"].notna().sum())}  '
            f'lidar1m={int(gdf["has_lidar_1m"].sum())}'
        )
    lines.append('')
    for county in sorted(c for c in gdf['county'].unique() if c):
        lines.append(f'{county.upper()} COUNTY')
        r = rec[rec['county'] == county].head(top)
        s = stake[stake['county'] == county].head(top)
        n_rec = int((rec['county'] == county).sum())
        n_stake = int((stake['county'] == county).sum())
        lines.append(f'  Go pan here (top {top} of {n_rec})')
        for _, p in r.iterrows():
            lines.append(
                f"    #{int(p['rec_rank_in_county'])}  {p['lat']:.5f}, {p['lon']:.5f}  "
                f"score {p['rec_score']:.0f}  trap {p['trap_score']:.1f}  "
                f"gold {p['nearest_gold_km']:.1f} km  {p['access_type']} "
                f"({p.get('manager_name') or ''})  near {p['seed_name']}")
        if r.empty:
            lines.append('    (none — no claim-free open public land on a walked reach)')
        lines.append(f'  Possibly stakeable (top {top} of {n_stake})')
        for _, p in s.iterrows():
            closed = _text(p.get('recent_closed_claim'))
            caution = _text(p.get('caution_note'))
            extra = f'  closed: {closed}' if closed else ''
            caution_bit = f'  caution: {caution}' if caution else ''
            lines.append(
                f"    #{int(p['stake_rank_in_county'])}  {p['lat']:.5f}, {p['lon']:.5f}  "
                f"score {p['stake_score']:.0f}  {p['federal_manager']}"
                f"{caution_bit}{extra}")
        if s.empty:
            lines.append('    (none)')
        lines.append('')
    lines += [
        'evidence = 45% trap + 35% gold upstream on the creek + 20% NURE Au/As.',
        'Chem drops out with no sample in range. Eocene gravel adds up to 10',
        'points within 5 km and nothing beyond that. A pin inside a hydraulic',
        'pit or dredge field is then halved.',
        'EE flags (filters only): NLCD developed/cropland drops recreational pins;',
        'water_regime / best_months / fire_year / lith_favor / road_m are recorded,',
        'not weighted. Creek walks use MERIT Hydro in Earth Engine when ee_trace.',
        'Stakeable = BLM/USFS fee land, no active MLRS claim in the buffer, no',
        'PAD-US withdrawal hint, not tribal. Check the BLM master title plat',
        'and MLRS case records before locating anything.',
    ]
    path = out(cfg, 'text', 'county_rank_summary.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Per-county panning + stakeable ranking')
    parser.add_argument('--config', required=True)
    parser.add_argument(
        '--rescore', action='store_true',
        help='Recompute evidence on existing pans (no creek retrace)',
    )
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='Geometry is in a geographic CRS')
        rescore(cfg) if args.rescore else run(cfg)


if __name__ == '__main__':
    main()
