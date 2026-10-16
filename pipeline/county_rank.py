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

Evidence per pan, each 0–1, then scaled 0–100
  trap     trap_score / 8. A gradient break counts once (2). A tributary
           mouth is 3 and a valley flare is 3. The three slope tests are one
           physical thing.
  source   gold sites on the walked creek, within 0.8 km of the upstream
           line, not a straight corridor onto the next gulch. The seed that
           started the walk is excluded. A producer counts in full. A
           prospect, an occurrence, or an unnamed site counts a quarter.
  chem     As, Sb, and W on that same creek. A badge only. It does not move
           the rank, so a blank sample cannot outrank a weak one and a desert
           percentile cannot sit next to a foothill pin.
  Eocene gravel adds 10 points only when the pin sits on the unit. EOc1 mixes
  the auriferous gravel and the Ione Formation and cannot be split at
  1:750,000. Off the polygon the bonus is 0, so a desert creek is not punished.
  A pin inside a hydraulic pit or dredge-tailings polygon is then halved.

Two lists, ranked inside each county
  recreational  access_ok == yes (BLM / USFS / state park, claim-free, open)
  stakeable     BLM or USFS land, no active MLRS claim in the buffer, no PAD-US
                withdrawal hint, not tribal. A claim BLM closed in the last
                year is named on the card and does not change the score.

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

TRAP_MAX = 8.0
SOURCE_SCALE_KM = 3.0
W_TRAP, W_SOURCE = 0.45, 0.35
# A closed claim is a land lead. It used to add 10 and was the whole stake
# leaderboard. The name stays on the card. The score does not move.
CLOSED_CLAIM_BONUS = 0.0
REACH_HALF_KM = 0.8
SEED_EXCLUDE_KM = 0.4
PROSPECT_WEIGHT = 0.25
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
    'chem_elements': ['As', 'Sb', 'W'],
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


def upstream_source_component(alongs_km, weights=None, scale_km=UPSTREAM_SCALE_KM):
    """0–1. Nearest upstream site dominates; more sites saturate at 1.

    weights scale each site. A producer is 1. A prospect is a quarter.
    """
    alongs = np.asarray(alongs_km, dtype=float)
    if weights is None:
        w = np.ones(alongs.shape, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    ok = np.isfinite(alongs) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return 0.0
    return float(np.clip(np.sum(np.exp(-alongs[ok] / scale_km) * w[ok]), 0.0, 1.0))


def gold_site_weight(name, dev_stat, prospect_weight=PROSPECT_WEIGHT):
    """Full credit for a producer. A prospect or unnamed site is a quarter."""
    if 'Producer' in str(dev_stat or ''):
        return 1.0
    return float(prospect_weight)


def channel_component(dist_km, scale_km=CHANNEL_SCALE_KM):
    """1 on the Eocene gravel, decaying with distance. NaN where unknown."""
    d = np.asarray(dist_km, dtype=float)
    return np.where(np.isfinite(d), np.exp(-np.maximum(d, 0.0) / scale_km), np.nan)


def channel_bonus(dist_km, near_km=CHANNEL_NEAR_KM, points=CHANNEL_BONUS):
    """`points` only when the pin sits on the Eocene unit.

    A fading bonus out to 5 km let a pin 0.8 km off the polygon outrank a
    better trap. Off the polygon the bonus is 0, so a desert creek is not
    penalised. EOc1 cannot be split into auriferous gravel versus the Ione.
    """
    d = np.asarray(dist_km, dtype=float)
    on = np.isfinite(d) & (d <= 0.0)
    return np.where(on, points, 0.0)


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


def trap_on_new_scale(trap, why):
    """Old pins stored 2+2+2 for one gradient break. Count that break once.

    A why string that already says gradient_break is on the new scale and
    is left alone. Tributary and valley points are kept.
    """
    trap = np.asarray(trap, dtype=float).copy()
    if why is None:
        return trap
    why = np.asarray(why, dtype=object)
    for i, text in enumerate(why):
        if text is None or (isinstance(text, float) and not np.isfinite(text)):
            text = ''
        text = str(text)
        if 'gradient_break' in text:
            continue
        bits = [b for b in ('slope_break', 'knickpoint_foot', 'power_drop') if b in text]
        if not bits:
            continue
        score = 2.0
        if 'tributary' in text:
            score += 3.0
        if 'valley' in text:
            score += 3.0
        trap[i] = score
    return trap


def evidence_score(trap, source, chem=None):
    """0–100 from trap and upstream gold. Chemistry does not move this number.

    chem is accepted so older callers still pass it. A missing sample used to
    drop out of the denominator and outrank a real ordinary sample.
    """
    trap = np.clip(np.asarray(trap, dtype=float) / TRAP_MAX, 0.0, 1.0)
    source = np.asarray(source, dtype=float)
    den = W_TRAP + W_SOURCE
    return np.round(100.0 * (W_TRAP * trap + W_SOURCE * source) / den, 1)


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

    if 'why' in pans.columns:
        pans['trap_score'] = trap_on_new_scale(pans['trap_score'], pans['why'])
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
        for item in upstream_alongs:
            if isinstance(item, tuple):
                alongs, weights = item
            else:
                alongs, weights = item, None
            alongs = np.asarray(alongs, dtype=float)
            src.append(upstream_source_component(alongs, weights))
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


def _as_line(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == 'LineString':
        return geom
    if geom.geom_type == 'MultiLineString' and len(geom.geoms):
        return max(geom.geoms, key=lambda g: g.length)
    return None


def _reach_table(reaches):
    """walk_id → (line, seed lon, seed lat)."""
    out = {}
    if reaches is None:
        return out
    if hasattr(reaches, 'iterrows'):
        rows = reaches.itertuples(index=False)
    else:
        rows = reaches
    for row in rows:
        if hasattr(row, 'walk_id'):
            walk_id = row.walk_id
            geom = row.geometry
            seed_lon = getattr(row, 'anchor_lon', None)
            seed_lat = getattr(row, 'anchor_lat', None)
        else:
            walk_id = row['walk_id']
            geom = row['geometry']
            seed_lon = row.get('anchor_lon')
            seed_lat = row.get('anchor_lat')
        line = _as_line(geom)
        if line is not None:
            out[walk_id] = (line, seed_lon, seed_lat)
    return out


def _project_many(xy, chain, points):
    """Nearest point on a km polyline for many sites.

    Returns cross-track km, along-track km, and whether the nearest point
    falls off either end of the line. Each array has one row per site.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    a = xy[:-1]
    b = xy[1:]
    ab = b - a
    length2 = np.sum(ab * ab, axis=1)
    valid = length2 >= 1e-12
    diff = points[:, None, :] - a[None, :, :]
    t = np.sum(diff * ab[None, :, :], axis=2) / np.where(valid, length2, 1.0)[None, :]
    t_clamped = np.clip(t, 0.0, 1.0)
    proj = a[None, :, :] + t_clamped[:, :, None] * ab[None, :, :]
    delta = points[:, None, :] - proj
    cross = np.hypot(delta[:, :, 0], delta[:, :, 1])
    cross = np.where(valid[None, :], cross, np.inf)
    seg_len = np.sqrt(np.where(valid, length2, 0.0))
    along = chain[:-1][None, :] + t_clamped * seg_len[None, :]
    k = np.argmin(cross, axis=1)
    rows = np.arange(points.shape[0])
    nseg = a.shape[0]
    t_best = t[rows, k]
    off = ((k == 0) & (t_best < 0.0)) | ((k == nseg - 1) & (t_best > 1.0))
    bad = ~np.isfinite(cross[rows, k])
    return cross[rows, k], along[rows, k], off | bad


def _project_chain(xy, chain, point):
    """Nearest point on a km polyline. The third value is True off either end."""
    cross, along, off = _project_many(xy, chain, np.asarray(point, dtype=float).reshape(1, 2))
    if not np.isfinite(cross[0]):
        return None
    return float(cross[0]), float(along[0]), bool(off[0])


def reach_hits(line, pin_lon, pin_lat, site_lon, site_lat, lat0,
               half_width_km=REACH_HALF_KM, max_km=UPSTREAM_MAX_KM,
               seed_lon=None, seed_lat=None, seed_km=SEED_EXCLUDE_KM):
    """Indices and upstream km of sites on the walked creek.

    The line is ordered upstream end to downstream. A site counts when it is
    upstream of the pin, within half_width_km of the line, and not past the
    end of the trace. The seed that started the walk is left out.
    """
    site_lon = np.asarray(site_lon, dtype=float)
    site_lat = np.asarray(site_lat, dtype=float)
    empty = (np.array([], dtype=int), np.array([], dtype=float))
    line = _as_line(line)
    if line is None or len(site_lon) == 0:
        return empty
    coords = np.asarray(line.coords, dtype=float)
    if len(coords) < 2:
        return empty
    xy = to_km_xy(coords[:, 0], coords[:, 1], lat0)
    step = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    chain = np.concatenate([[0.0], np.cumsum(step)])
    pin = to_km_xy([pin_lon], [pin_lat], lat0)[0]
    pin_hit = _project_chain(xy, chain, pin)
    if pin_hit is None:
        return empty
    pin_s = pin_hit[1]
    sites = to_km_xy(site_lon, site_lat, lat0)
    cross, along, off = _project_many(xy, chain, sites)
    upstream = pin_s - along
    keep = (~off) & (upstream >= 0.0) & (upstream <= max_km) & (cross <= half_width_km)
    if seed_lon is not None and seed_lat is not None:
        try:
            if np.isfinite(float(seed_lon)) and np.isfinite(float(seed_lat)):
                seed = to_km_xy([float(seed_lon)], [float(seed_lat)], lat0)[0]
                keep &= np.hypot(sites[:, 0] - seed[0], sites[:, 1] - seed[1]) > seed_km
        except (TypeError, ValueError):
            pass
    idx = np.flatnonzero(keep)
    return idx.astype(int), upstream[idx]


def upstream_alongs_for_pans(pans, reaches, gold, lat0):
    """(along-km, weights) of gold sites on each pin's walked creek.

    reaches is a list of dicts or a GeoDataFrame with walk_id and geometry.
    A missing line yields an empty creek, so the pin does not inherit a
    side-ridge dot. The seed is excluded. Prospects weigh a quarter.
    """
    table = _reach_table(reaches)
    if gold is None or len(gold) == 0:
        site_lon = np.array([])
        site_lat = np.array([])
        weights_all = np.array([])
    else:
        site_lon = gold['lon'].to_numpy(dtype=float)
        site_lat = gold['lat'].to_numpy(dtype=float)
        names = gold['site_name'] if 'site_name' in gold.columns else pd.Series('', index=gold.index)
        devs = gold['dev_stat'] if 'dev_stat' in gold.columns else pd.Series('', index=gold.index)
        weights_all = np.array([
            gold_site_weight(n, d) for n, d in zip(names, devs)
        ], dtype=float)
    alongs = []
    for p in pans.itertuples(index=False):
        rec = table.get(p.walk_id)
        if rec is None or len(site_lon) == 0:
            alongs.append((np.array([]), np.array([])))
            continue
        line, seed_lon, seed_lat = rec
        idx, dist = reach_hits(
            line, float(p.lon), float(p.lat), site_lon, site_lat, lat0,
            seed_lon=seed_lon, seed_lat=seed_lat,
        )
        alongs.append((dist, weights_all[idx] if len(idx) else np.array([])))
    return alongs


def creek_chem(pans, reaches, nure, elements, lat0):
    """Best pathfinder percentile on the walked creek. NaN when none is on it."""
    n = len(pans)
    out_ = np.full(n, np.nan)
    if nure is None or len(nure) == 0 or n == 0:
        return out_
    scores = nure_sample_scores(nure, elements)
    if scores is None:
        return out_
    ok = scores.notna().to_numpy()
    if not ok.any():
        return out_
    slon = nure['lon'].to_numpy(dtype=float)[ok]
    slat = nure['lat'].to_numpy(dtype=float)[ok]
    sval = scores.to_numpy()[ok]
    table = _reach_table(reaches)
    for i, p in enumerate(pans.itertuples(index=False)):
        rec = table.get(getattr(p, 'walk_id', None))
        if rec is None:
            continue
        line, _seed_lon, _seed_lat = rec
        idx, _dist = reach_hits(
            line, float(p.lon), float(p.lat), slon, slat, lat0,
            seed_lon=None, seed_lat=None,
        )
        if len(idx):
            out_[i] = float(np.max(sval[idx]))
    return out_


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
    nure = _load_nure(cfg)
    pans = score_pans(
        pans, nure, gold_xy, rc, lat0,
        upstream_alongs=alongs, channel_km=channel, on_mined=mined,
    )
    # Badge only: pathfinders on this creek, not the best sample within 3 km.
    pans['chem_c'] = np.round(creek_chem(pans, reaches, nure, rc['chem_elements'], lat0), 3)
    pans['mined_name'] = flags['mined_name'].to_numpy()
    if flags['mined_checked'].iloc[0]:
        print(f'  Mined ground: {int(pans["on_mined"].sum())} pins', flush=True)
    if flags['channel_checked'].iloc[0]:
        print(
            f'  On Eocene gravel: {(pans["channel_km"] <= 0).sum()} pins',
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
    pans['stake_score'] = np.where(was_stake, pans['evidence_score'], np.nan)
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
    gdf['stake_score'] = np.where(stake_ok, gdf['evidence_score'], np.nan)
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
        on_unit = int((pd.to_numeric(gdf['channel_km'], errors='coerce') <= 0).sum())
        lines.append(f'On the Eocene unit: {on_unit}.')
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
        'evidence = trap and gold on the walked creek. Chemistry (As, Sb, W on',
        'that creek) is a badge and does not move the rank. Eocene gravel adds',
        '10 points only on the mapped unit. A pin inside a hydraulic pit or',
        'dredge field is then halved. A closed claim is named, not scored.',
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
