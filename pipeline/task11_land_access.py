"""Land-access labels for Task 11 (CA first).

Chemistry stays honest. This layer answers: what kind of land is the pin on,
can a recreational panber reasonably test it, and how do drainages rank
overall vs inside each access bucket.

Overlays (cached under data/):
  PAD-US 4.1 — public land / manager / Pub_Access
  BLM MLRS   — mining claims not closed
  AIANNH     — Census TIGER/Line reservations + off-reservation trust land
  hobby CSV  — optional access_type=club|industry on gazetteer rows

Priority at a point:
  claimed > tribal > restricted > club/industry > PAD-US manager > private

Claimed stays first so the claim name is the reason shown; tribal_land /
tribal_name are set from the AIANNH hit regardless of which label wins, so
a claim inside a reservation is still flagged tribal. Census statistical
areas (TDSA/SDTSA/OTSA/ANVSA, FUNCSTAT=S) are not tribal land; they only
add a consult-the-tribe line to rules_note. Every pin also gets rules_note
(state + manager reminders). All of it is a screen, not permission.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pipeline.ml_artifacts import REPO_ROOT
from pipeline.utils import bbox, resolve_data_path

ACCESS_TYPES = (
    'blm', 'usfs', 'state_park', 'state_other', 'nps', 'local_gov',
    'private', 'claimed', 'club', 'industry', 'restricted', 'tribal', 'unknown',
)
# Recreational hands-and-pans candidates when claim-free and not restricted.
ACCESS_OK_TYPES = frozenset({'blm', 'usfs', 'state_park'})
CLAIM_STATUS = ('free', 'claimed', 'unchecked')

PADUS_URL = (
    'https://edits.nationalmap.gov/arcgis/rest/services/'
    'PAD-US/PAD_US_gaz_combined/MapServer/0/query'
)
MLRS_URL = (
    'https://gis.blm.gov/nlsdb/rest/services/HUB/'
    'BLM_Natl_MLRS_Mining_Claims_Not_Closed/FeatureServer/0/query'
)
PADUS_FIELDS = (
    'Unit_Nm,Mang_Name,Mang_Type,Own_Name,Own_Type,Des_Tp,'
    'Pub_Access,GAP_Sts,State_Nm,Category'
)
MLRS_FIELDS = 'OBJECTID,ID,CSE_NAME,CSE_DISP,BLM_PROD,CSE_NR,LEG_CSE_NR'
TIMEOUT = 180
PAGE = 2000

# Fig / legend colors (Wong-safe-ish; private/claimed muted).
ACCESS_COLOR = {
    'blm': '#E69F00',
    'usfs': '#009E73',
    'state_park': '#56B4E9',
    'state_other': '#0072B2',
    'nps': '#CC79A7',
    'local_gov': '#D55E00',
    'private': '#999999',
    'claimed': '#000000',
    'club': '#F0E442',
    'industry': '#882255',
    'restricted': '#661100',
    'tribal': '#AA4499',
    'unknown': '#CCCCCC',
}

# Per-pin columns written by annotate_points and rolled up to the walk table.
ACCESS_COLS = (
    'access_type', 'access_ok', 'access_reason', 'manager_name',
    'claim_status', 'pub_access', 'tribal_land', 'tribal_name', 'rules_note',
)
ACCESS_DEFAULTS = {
    'access_type': 'unknown',
    'access_ok': 'no',
    'claim_status': 'unchecked',
    'tribal_land': False,
}
TRIBAL_REASON = 'tribal land — requires tribal permission'
TRIBAL_NOT_LOADED = 'tribal layer not loaded'

RULES_BY_ACCESS = {
    'tribal': 'tribal permission required',
    'usfs': (
        'USFS open land, negligible disturbance, no notice (36 CFR 228.4): '
        'pan, hand tools, crevice tool, classifier, non-motorized sluice, '
        'metal detector, battery drywasher. Notice of Intent to the district '
        'ranger before a motorized highbanker, suction dredge, trommel, or '
        'other work that might significantly disturb the surface'
    ),
    'blm': (
        'BLM open land, casual use (43 CFR 3809.5): pan, hand tools, crevice '
        'tool, classifier, non-motorized sluice, metal detector, battery '
        'drywasher. Ask the field office before a suction dredge or motorized '
        'highbanker. Mechanized earth-moving equipment is not casual use'
    ),
    'state_park': 'state park: confirm panning rules with park staff',
    'state_other': 'state land: confirm rules with the managing agency',
    'nps': 'NPS: confirm collecting rules with the park',
    'local_gov': 'local park/land: confirm rules with the managing agency',
    'restricted': 'restricted: confirm access with the manager',
    'claimed': 'claim-holder permission required',
    'club': 'club membership / permission required',
    'industry': 'owner permission required',
    'private': 'landowner permission required',
    'unknown': 'land status unchecked: confirm owner / manager',
}
RULES_BY_STATE = {
    'CA': (
        'CA: motorized suction-dredge and vacuum dredge are prohibited in '
        'rivers, streams, and lakes (Fish and Game Code 5653.1)'
    ),
    'WA': 'WA: follow current WDFW Gold and Fish rules / HPA',
}
_STATE_ABBR = frozenset({
    'AK', 'AZ', 'CA', 'CO', 'ID', 'MT', 'NC', 'NM', 'NV', 'OR', 'SC', 'UT',
    'VA', 'WA', 'WY',
})

CLASS_ORDER = {'expedition': 0, 'confirm': 1, 'watch': 2, 'skip': 3}
RANK_BY_CHEMISTRY = 'chemistry'
RANK_BY_GEOMETRY_ACCESS = 'geometry_access'


def land_access_enabled(cfg):
    return bool((cfg.get('task11') or {}).get('land_access', False))


def _short(cfg):
    return str((cfg.get('study_area') or {}).get('short') or 'study').strip() or 'study'


def padus_path(cfg):
    p = resolve_data_path(cfg, 'padus_geojson')
    if p:
        return p
    return str(REPO_ROOT / 'data' / 'padus' / f'padus_{_short(cfg)}.geojson')


def mlrs_path(cfg):
    p = resolve_data_path(cfg, 'mlrs_geojson')
    if p:
        return p
    return str(REPO_ROOT / 'data' / 'mlrs' / f'mlrs_claims_{_short(cfg)}.geojson')


def aiannh_path(cfg):
    p = resolve_data_path(cfg, 'aiannh_geojson')
    if p:
        return p
    return str(REPO_ROOT / 'data' / 'aiannh' / f'aiannh_{_short(cfg)}.geojson')


def config_state(cfg):
    """Two-letter state for rules_note: data.sgmc_state, study_area.state, or short."""
    for raw in (
        (cfg.get('data') or {}).get('sgmc_state'),
        (cfg.get('study_area') or {}).get('state'),
    ):
        if raw and str(raw).strip().upper() in _STATE_ABBR:
            return str(raw).strip().upper()
    for tok in _short(cfg).upper().split('_'):
        if tok in _STATE_ABBR:
            return tok
    return ''


def rules_note(access_type, state='', tribal_loaded=True, tribal_stat_name=''):
    """Agency reminders for a pin. Not legal advice — confirm with the agency."""
    parts = []
    base = RULES_BY_ACCESS.get(str(access_type or 'unknown'))
    if base:
        parts.append(base)
    st = RULES_BY_STATE.get(str(state or '').upper())
    if st:
        parts.append(st)
    if tribal_stat_name:
        parts.append(f'in {tribal_stat_name} (statistical area): consult the tribe')
    if not tribal_loaded:
        parts.append(TRIBAL_NOT_LOADED)
    return '; '.join(parts)


def load_tribal_layers(cfg):
    """(legal, statistical) AIANNH GeoDataFrames, or (None, None) if cache missing."""
    path = aiannh_path(cfg)
    if not path or not os.path.exists(path) or os.path.getsize(path) < 40:
        return None, None
    gdf = _load_gdf(path)
    if len(gdf) == 0:
        return gdf, gdf
    stat = pd.Series(False, index=gdf.index)
    if 'FUNCSTAT' in gdf.columns:
        stat = gdf['FUNCSTAT'].astype(str).str.upper().eq('S')
    return gdf.loc[~stat].copy(), gdf.loc[stat].copy()


def _tribal_label(row):
    for key in ('NAMELSAD', 'NAME', 'tribal_name', 'name'):
        val = row.get(key) if hasattr(row, 'get') else None
        if val is not None and pd.notna(val) and str(val).strip():
            return str(val).strip()
    return 'tribal area'


def _tribal_hit(layer, geom):
    if layer is None or len(layer) == 0 or geom is None or geom.is_empty:
        return ''
    hits = layer[layer.intersects(geom)]
    if len(hits) == 0:
        return ''
    return _tribal_label(hits.iloc[0])


def claim_buffer_m(cfg):
    try:
        return float((cfg.get('task11') or {}).get('claim_buffer_m', 50))
    except (TypeError, ValueError):
        return 50.0


def min_pour_elev_m(cfg):
    raw = (cfg.get('task11') or {}).get('min_pour_elev_m')
    if raw is None or raw == '':
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def rank_by_mode(cfg=None, rank_by=None):
    """Walk-list sort. Default chemistry (expedition → isolation → P).

    geometry_access: access_ok, then trap_score, then catchment_ok, P last.
    Only California sets task11.rank_by: geometry_access. Test belts stay
    chemistry so transfer GPKGs are not sold as walk apps.
    """
    if rank_by is None and cfg is not None:
        rank_by = (cfg.get('task11') or {}).get('rank_by')
    raw = str(rank_by or RANK_BY_CHEMISTRY).strip().lower().replace('-', '_')
    if raw in (RANK_BY_GEOMETRY_ACCESS, 'geometry', 'access'):
        return RANK_BY_GEOMETRY_ACCESS
    return RANK_BY_CHEMISTRY


def add_walk_sort_keys(df, mode=RANK_BY_CHEMISTRY):
    """Add helper columns. Lower sorts earlier. Returns the column names."""
    if mode == RANK_BY_GEOMETRY_ACCESS:
        if 'access_ok' in df.columns:
            df['_access'] = np.where(
                df['access_ok'].astype(str).str.lower() == 'yes', 0, 1,
            )
        else:
            df['_access'] = 1
        if 'trap_score' in df.columns:
            df['_trap'] = -pd.to_numeric(df['trap_score'], errors='coerce').fillna(0.0)
        else:
            df['_trap'] = 0.0
        if 'catchment_ok' in df.columns:
            catch = df['catchment_ok']
            if catch.dtype == object:
                catch = catch.fillna(False).astype(str).str.lower().isin(
                    ('true', '1', 'yes')
                )
            else:
                catch = catch.fillna(False).astype(bool)
            df['_catch'] = np.where(catch, 0, 1)
        else:
            df['_catch'] = 0
        if 'max_p' in df.columns:
            df['_p'] = -pd.to_numeric(df['max_p'], errors='coerce').fillna(-1.0)
        else:
            df['_p'] = 0.0
        return ['_access', '_trap', '_catch', '_p']

    df['_ord'] = df['campaign_class'].map(CLASS_ORDER).fillna(9)
    if 'nearest_gold_deg' in df.columns:
        df['_iso'] = -pd.to_numeric(df['nearest_gold_deg'], errors='coerce').fillna(-1.0)
    else:
        df['_iso'] = 0.0
    if 'max_p' in df.columns:
        df['_p'] = -pd.to_numeric(df['max_p'], errors='coerce').fillna(-1.0)
    else:
        df['_p'] = 0.0
    if 'n_hobby_hit' in df.columns:
        df['_hobby'] = -(
            pd.to_numeric(df['n_hobby_hit'], errors='coerce').fillna(0) > 0
        ).astype(int)
    else:
        df['_hobby'] = 0
    return ['_ord', '_iso', '_p', '_hobby']


def _cell_trap_from_pans(pans):
    """Max trap_score per block. Prefer access_ok pans when any exist."""
    if pans is None or len(pans) == 0 or 'trap_score' not in pans.columns:
        return pd.DataFrame(columns=['block_id', 'trap_score'])
    if 'block_id' not in pans.columns:
        return pd.DataFrame(columns=['block_id', 'trap_score'])
    rows = []
    for bid, grp in pans.groupby('block_id'):
        src = grp
        if 'access_ok' in grp.columns:
            ok = grp[grp['access_ok'].astype(str).str.lower() == 'yes']
            if len(ok):
                src = ok
        score = pd.to_numeric(src['trap_score'], errors='coerce').max()
        rows.append({
            'block_id': bid,
            'trap_score': None if pd.isna(score) else float(score),
        })
    return pd.DataFrame(rows)


def _query_arcgis(url, lon_min, lat_min, lon_max, lat_max, out_fields='*',
                  geometry_json=False):
    """Paginated FeatureCollection for a WGS84 envelope.

    Some BLM services reject large envelopes or the simple geometry=xmin,…
    string — set geometry_json=True to send an Esri JSON envelope instead.
    """
    features = []
    offset = 0
    while True:
        if geometry_json:
            geom = json.dumps({
                'xmin': lon_min, 'ymin': lat_min,
                'xmax': lon_max, 'ymax': lat_max,
                'spatialReference': {'wkid': 4326},
            })
        else:
            geom = f'{lon_min},{lat_min},{lon_max},{lat_max}'
        params = {
            'f': 'geojson',
            'where': '1=1',
            'geometry': geom,
            'geometryType': 'esriGeometryEnvelope',
            'inSR': '4326',
            'spatialRel': 'esriSpatialRelIntersects',
            'outFields': out_fields,
            'returnGeometry': 'true',
            'outSR': '4326',
            'resultOffset': offset,
            'resultRecordCount': PAGE,
        }
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and payload.get('error'):
            raise RuntimeError(f'ArcGIS error: {payload["error"]}')
        batch = payload.get('features') or []
        features.extend(batch)
        if len(batch) < PAGE:
            break
        offset += PAGE
        if offset > 500_000:
            break
    return {'type': 'FeatureCollection', 'features': features}


def _write_geojson(path, fc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(fc, f)
    return str(path)


def _tile_bbox(lon_min, lat_min, lon_max, lat_max, step=0.5):
    """Yield overlapping 0.5° tiles covering the study box."""
    lon = lon_min
    while lon < lon_max - 1e-12:
        lat = lat_min
        lon2 = min(lon + step, lon_max)
        while lat < lat_max - 1e-12:
            lat2 = min(lat + step, lat_max)
            yield lon, lat, lon2, lat2
            lat = lat2
        lon = lon2


def _dedupe_features(features):
    seen = set()
    out = []
    for feat in features:
        props = feat.get('properties') or {}
        key = (
            props.get('OBJECTID')
            or props.get('ID')
            or props.get('CSE_NR')
            or props.get('CSE_NAME')
            or json.dumps(feat.get('geometry'), sort_keys=True)[:120]
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(feat)
    return out


def fetch_padus(cfg, force=False):
    dest = padus_path(cfg)
    if os.path.exists(dest) and not force and os.path.getsize(dest) > 500:
        print(f'  PAD-US already at {dest}', flush=True)
        return dest
    lon_min, lon_max, lat_min, lat_max = bbox(cfg)
    print(
        f'  Fetching PAD-US for {lon_min},{lat_min}–{lon_max},{lat_max}',
        flush=True,
    )
    fc = _query_arcgis(PADUS_URL, lon_min, lat_min, lon_max, lat_max, PADUS_FIELDS)
    _write_geojson(dest, fc)
    print(f'  Wrote {len(fc["features"])} PAD-US features → {dest}', flush=True)
    return dest


def fetch_mlrs(cfg, force=False):
    dest = mlrs_path(cfg)
    unchecked_marker = Path(str(dest) + '.unchecked')
    if (
        os.path.exists(dest) and not force and os.path.getsize(dest) > 200
        and not unchecked_marker.exists()
    ):
        print(f'  MLRS claims already at {dest}', flush=True)
        return dest
    lon_min, lon_max, lat_min, lat_max = bbox(cfg)
    print(
        f'  Fetching MLRS claims for {lon_min},{lat_min}–{lon_max},{lat_max} '
        f'(0.5° tiles)',
        flush=True,
    )
    all_feats = []
    for tlon0, tlat0, tlon1, tlat1 in _tile_bbox(
        lon_min, lat_min, lon_max, lat_max, step=0.5,
    ):
        try:
            fc = _query_arcgis(
                MLRS_URL, tlon0, tlat0, tlon1, tlat1, MLRS_FIELDS,
                geometry_json=True,
            )
            all_feats.extend(fc.get('features') or [])
            print(
                f'    tile {tlon0:.2f},{tlat0:.2f}–{tlon1:.2f},{tlat1:.2f}: '
                f'{len(fc.get("features") or [])}',
                flush=True,
            )
        except Exception as exc:
            print(f'    tile failed ({exc}); continuing', flush=True)
    features = _dedupe_features(all_feats)
    _write_geojson(dest, {'type': 'FeatureCollection', 'features': features})
    if unchecked_marker.exists():
        unchecked_marker.unlink()
    print(f'  Wrote {len(features)} MLRS claims → {dest}', flush=True)
    return dest


def ensure_land_access_data(cfg, force=False):
    """Fetch caches when land_access is on. Returns (padus_path, mlrs_path|None)."""
    if not land_access_enabled(cfg):
        return None, None
    padus = fetch_padus(cfg, force=force)
    try:
        mlrs = fetch_mlrs(cfg, force=force)
    except Exception as exc:
        print(f'  MLRS fetch failed ({exc}); claim_status will be unchecked', flush=True)
        mlrs = None
        dest = mlrs_path(cfg)
        if not os.path.exists(dest) or os.path.getsize(dest) < 50:
            _write_geojson(dest, {'type': 'FeatureCollection', 'features': []})
        Path(str(dest) + '.unchecked').write_text(str(exc))
    try:
        from pipeline.fetch_land_access import fetch_aiannh
        fetch_aiannh(cfg, force=force)
    except Exception as exc:
        print(f'  AIANNH fetch failed ({exc}); tribal layer not loaded', flush=True)
    return padus, mlrs


def _load_gdf(path):
    import geopandas as gpd
    from shapely.validation import make_valid

    if not path or not os.path.exists(path):
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    if os.path.getsize(path) < 50:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    elif '4326' not in str(gdf.crs):
        gdf = gdf.to_crs('EPSG:4326')
    if len(gdf) and hasattr(gdf, 'is_valid') and (~gdf.is_valid).any():
        gdf = gdf.copy()
        gdf['geometry'] = gdf.geometry.map(
            lambda g: make_valid(g) if g is not None and not g.is_empty else g
        )
    return gdf


def padus_manager_type(props):
    """Map PAD-US manager / owner / designation → access_type (not claimed)."""
    if props is None:
        return 'unknown', '', 'UK'
    get = props.get if hasattr(props, 'get') else lambda k, d=None: (
        props[k] if k in props.index and pd.notna(props[k]) else d
    )
    mang = str(get('Mang_Name') or get('Mang_Type') or '').strip()
    own = str(get('Own_Name') or get('Own_Type') or '').strip()
    des = str(get('Des_Tp') or get('Category') or '').strip()
    unit = str(get('Unit_Nm') or '').strip()
    pub = str(get('Pub_Access') or 'UK').strip().upper() or 'UK'
    blob = f'{mang} {own} {des} {unit}'.lower()
    manager = mang or own or unit or ''

    if pub in ('RA', 'XA'):
        return 'restricted', manager, pub

    if (
        'bureau of land management' in blob
        or blob.strip() == 'blm'
        or ' blm' in f' {blob}'
        or mang.upper() == 'BLM'
    ):
        return 'blm', manager, pub
    if (
        'forest service' in blob
        or 'usfs' in blob
        or mang.upper() in ('FS', 'USFS', 'USDA FOREST SERVICE')
        or 'national forest' in blob
    ):
        return 'usfs', manager, pub
    if 'national park' in blob or mang.upper() == 'NPS' or ' nps' in f' {blob}':
        return 'nps', manager, pub
    if (
        'fish and wildlife' in blob
        or 'fws' in blob
        or mang.upper() == 'FWS'
        or 'wildlife refuge' in blob
    ):
        return 'restricted', manager, pub
    if (
        'state park' in blob
        or 'state recreation' in blob
        or 'state historic' in blob
        or ('shp' in des.lower() and 'state' in blob)
    ):
        return 'state_park', manager, pub
    if 'state' in blob or mang.upper().startswith('STATE'):
        return 'state_other', manager, pub
    if (
        'city' in blob
        or 'county' in blob
        or 'local' in blob
        or 'regional park' in blob
    ):
        return 'local_gov', manager, pub
    if 'private' in blob or 'ngo' in blob or 'non-governmental' in blob:
        return 'private', manager, pub
    if not mang and not own:
        return 'unknown', manager, pub
    return 'unknown', manager, pub


def _hobby_access_points(hobby_gdf):
    """Points tagged club/industry (optional access_type column or aliases)."""
    import geopandas as gpd

    if hobby_gdf is None or len(hobby_gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    work = hobby_gdf.copy()
    if 'access_type' not in work.columns:
        work['access_type'] = pd.NA
    # Infer from notes / name if blank
    def _infer(row):
        raw = row.get('access_type')
        if pd.notna(raw) and str(raw).strip():
            key = str(raw).strip().lower().replace(' ', '_')
            if key in ('club', 'claim_club', 'prospecting_club'):
                return 'club'
            if key in ('industry', 'mine', 'company', 'patented'):
                return 'industry'
            if key in ACCESS_TYPES:
                return key
        blob = ' '.join(
            str(row.get(c) or '') for c in ('notes', 'name', 'source_cite', 'source_type')
        ).lower()
        if 'club' in blob or 'gpma' in blob:
            return 'club'
        if 'mining company' in blob or 'patented' in blob or 'industry' in blob:
            return 'industry'
        return None

    work['_atype'] = work.apply(_infer, axis=1)
    work = work[work['_atype'].isin(['club', 'industry'])]
    if work.empty:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    return work


def classify_point(claimed, padus_hit, hobby_hit, claim_status_global='free',
                   tribal_hit=None):
    """Apply priority stack. Returns dict of access fields.

    tribal_hit: AIANNH area name (str) when the point is on reservation /
    trust land, else None/''. Sets tribal_land / tribal_name on every label.
    """
    tribal = {
        'tribal_land': bool(tribal_hit),
        'tribal_name': str(tribal_hit) if tribal_hit else '',
    }
    if claimed:
        name = ''
        if isinstance(claimed, dict):
            name = claimed.get('CSE_NAME') or claimed.get('CSE_NR') or claimed.get('ID') or ''
        reason = f'active MLRS claim{": " + str(name) if name else ""}'
        if tribal_hit:
            reason += f'; {TRIBAL_REASON} ({tribal_hit})'
        return {
            'access_type': 'claimed',
            'access_ok': 'no',
            'access_reason': reason.strip(),
            'manager_name': str(name) if name else '',
            'claim_status': 'claimed',
            'pub_access': '',
            **tribal,
        }

    if claim_status_global == 'unchecked':
        claim_status = 'unchecked'
        claim_note = 'MLRS unchecked'
    else:
        claim_status = 'free'
        claim_note = 'no MLRS claim in buffer'

    if tribal_hit:
        return {
            'access_type': 'tribal',
            'access_ok': 'no',
            'access_reason': TRIBAL_REASON,
            'manager_name': str(tribal_hit),
            'claim_status': claim_status,
            'pub_access': '',
            **tribal,
        }

    out = _classify_non_tribal(padus_hit, hobby_hit, claim_status, claim_note)
    out.update(tribal)
    return out


def _classify_non_tribal(padus_hit, hobby_hit, claim_status, claim_note):
    if hobby_hit:
        at = hobby_hit.get('_atype') or hobby_hit.get('access_type') or 'club'
        if at not in ('club', 'industry'):
            at = 'club'
        nm = hobby_hit.get('name') or hobby_hit.get('creek_name') or at
        return {
            'access_type': at,
            'access_ok': 'no',
            'access_reason': f'hobby gazetteer {at}: {nm}; {claim_note}',
            'manager_name': str(nm),
            'claim_status': claim_status,
            'pub_access': '',
        }

    if padus_hit is not None:
        atype, manager, pub = padus_manager_type(padus_hit)
        if atype == 'restricted' or pub in ('RA', 'XA'):
            return {
                'access_type': 'restricted',
                'access_ok': 'no',
                'access_reason': (
                    f'PAD-US restricted Pub_Access={pub} Mang={manager or "?"}; {claim_note}'
                ),
                'manager_name': manager,
                'claim_status': claim_status,
                'pub_access': pub,
            }
        # Open managers + claim-free (+ Pub_Access not closed) → access_ok
        ok = (
            'yes'
            if (
                atype in ACCESS_OK_TYPES
                and claim_status == 'free'
                and pub not in ('RA', 'XA')
            )
            else 'no'
        )
        reason = (
            f'PAD-US Mang={manager or "?"} type={atype} Pub_Access={pub or "UK"}; {claim_note}'
        )
        return {
            'access_type': atype,
            'access_ok': ok,
            'access_reason': reason,
            'manager_name': manager,
            'claim_status': claim_status,
            'pub_access': pub,
        }

    return {
        'access_type': 'private',
        'access_ok': 'no',
        'access_reason': f'not in PAD-US public layer; {claim_note}',
        'manager_name': '',
        'claim_status': claim_status,
        'pub_access': '',
    }


def _empty_access_cols(n):
    return {
        'access_type': ['unknown'] * n,
        'access_ok': ['no'] * n,
        'access_reason': ['land_access off or no geometry'] * n,
        'manager_name': [''] * n,
        'claim_status': ['unchecked'] * n,
        'pub_access': [''] * n,
        'tribal_land': [False] * n,
        'tribal_name': [''] * n,
        'rules_note': [''] * n,
    }


def _tribal_columns(out, cfg, legal, stat, states=None):
    """tribal_land / tribal_name / rules_note for already-labelled points."""
    loaded = legal is not None
    default_state = config_state(cfg)
    if states is None:
        states = [default_state] * len(out)
    states = [s or default_state for s in states]
    names, stat_names = [], []
    for geom in out.geometry:
        names.append(_tribal_hit(legal, geom) if loaded else '')
        stat_names.append(_tribal_hit(stat, geom) if loaded else '')
    out['tribal_land'] = [bool(nm) for nm in names]
    out['tribal_name'] = names
    out['rules_note'] = [
        rules_note(
            'tribal' if nm else at, st,
            tribal_loaded=loaded, tribal_stat_name=sn,
        )
        for at, nm, sn, st in zip(out['access_type'], names, stat_names, states)
    ]
    return out


def _padus_state(padus_hit):
    if padus_hit is None:
        return ''
    raw = padus_hit.get('State_Nm') if hasattr(padus_hit, 'get') else None
    if raw is None or (not isinstance(raw, str) and pd.isna(raw)):
        return ''
    raw = str(raw).strip().upper()
    return raw if raw in _STATE_ABBR else ''


def annotate_points(points_gdf, cfg, hobby_gdf=None):
    """Add access_* + tribal_land / tribal_name / rules_note to a Point GeoDataFrame.

    The tribal screen runs whenever the AIANNH cache exists, even with
    land_access off (labels stay unknown/no; tribal columns are still set).
    """
    if points_gdf is None or len(points_gdf) == 0:
        return points_gdf

    out = points_gdf.copy()
    n = len(out)
    legal, stat = load_tribal_layers(cfg)
    if not land_access_enabled(cfg):
        for k, v in _empty_access_cols(n).items():
            out[k] = v
        out['access_reason'] = ['land_access disabled'] * n
        return _tribal_columns(out, cfg, legal, stat)

    padus = _load_gdf(padus_path(cfg))
    mlrs = _load_gdf(mlrs_path(cfg))
    unchecked = os.path.exists(mlrs_path(cfg) + '.unchecked') and len(mlrs) == 0
    claim_global = 'unchecked' if unchecked else 'free'

    buf_m = claim_buffer_m(cfg)
    claims_buf_gdf = None
    if len(mlrs):
        from pipeline.geo_crs import utm_epsg
        lon0 = float(out.geometry.x.mean())
        lat0 = float(out.geometry.y.mean())
        epsg = utm_epsg(lon0, lat0)
        claims_utm = mlrs.to_crs(epsg)
        claims_buf_gdf = claims_utm.copy()
        claims_buf_gdf['geometry'] = claims_utm.buffer(buf_m)
        claims_buf_gdf = claims_buf_gdf.to_crs('EPSG:4326')

    hobby_pts = _hobby_access_points(hobby_gdf)
    hobby_tree = None
    if len(hobby_pts):
        from scipy.spatial import cKDTree
        hobby_coords = np.column_stack([
            hobby_pts.geometry.x.values, hobby_pts.geometry.y.values,
        ])
        hobby_tree = cKDTree(hobby_coords)

    rows = []
    states = []
    for _, row in out.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            rows.append(classify_point(False, None, None, claim_global))
            states.append('')
            continue

        claim_props = False
        if claims_buf_gdf is not None and len(claims_buf_gdf):
            hits = claims_buf_gdf[claims_buf_gdf.intersects(geom)]
            if len(hits):
                claim_props = hits.iloc[0].drop(
                    labels='geometry', errors='ignore',
                ).to_dict()

        padus_hit = None
        if len(padus):
            hits = padus[padus.contains(geom)]
            if len(hits) == 0:
                hits = padus[padus.intersects(geom)]
            if len(hits):
                areas = hits.geometry.area
                padus_hit = hits.loc[areas.idxmin()].drop(
                    labels='geometry', errors='ignore',
                )

        hobby_hit = None
        if hobby_tree is not None:
            lat = float(geom.y)
            deg = 400.0 / (111_000.0 * max(0.2, abs(np.cos(np.radians(lat)))))
            d, i = hobby_tree.query([geom.x, geom.y])
            if float(d) <= deg:
                hobby_hit = hobby_pts.iloc[int(i)].to_dict()

        tribal_hit = _tribal_hit(legal, geom) if legal is not None else ''
        rows.append(classify_point(
            claim_props, padus_hit, hobby_hit, claim_global, tribal_hit=tribal_hit,
        ))
        states.append(_padus_state(padus_hit))

    ann = pd.DataFrame(rows, index=out.index)
    for col in ann.columns:
        out[col] = ann[col]
    return _tribal_columns(out, cfg, legal, stat, states=states)


def _cell_access_from_pans(pans):
    """One access label per block_id from pan pins (prefer any access_ok=yes)."""
    cols = ['block_id', *ACCESS_COLS]
    if pans is None or len(pans) == 0 or 'block_id' not in pans.columns:
        return pd.DataFrame(columns=cols)
    if 'access_type' not in pans.columns:
        return pd.DataFrame(columns=cols)

    rows = []
    for bid, grp in pans.groupby('block_id'):
        ok = grp[grp['access_ok'] == 'yes'] if 'access_ok' in grp.columns else grp.iloc[0:0]
        pick = ok.iloc[0] if len(ok) else grp.iloc[0]
        rec = {'block_id': bid}
        for col in ACCESS_COLS:
            rec[col] = pick.get(col, ACCESS_DEFAULTS.get(col, ''))
        rows.append(rec)
    return pd.DataFrame(rows)


def apply_access_ranks(walk, cfg):
    """Set walk_rank, access_rank, rank_in_access_type on the walk table.

    walk_rank  — chemistry interesting + catchment_ok + elev gate
    access_rank — same sort among access_ok=yes
    rank_in_access_type — within each access_type bucket

    Sort is chemistry (expedition → isolation → P) unless
    task11.rank_by is geometry_access (access + trap, P last).
    """
    out = walk.copy()
    for col in (*ACCESS_COLS, 'access_rank', 'rank_in_access_type'):
        if col not in out.columns:
            if col in ('access_rank', 'rank_in_access_type'):
                out[col] = pd.Series(pd.NA, index=out.index, dtype='Int64')
            else:
                out[col] = ACCESS_DEFAULTS.get(col, '')

    elev_min = min_pour_elev_m(cfg)
    chem_ok = out['campaign_class'].isin(['expedition', 'confirm', 'watch'])
    catch_ok = out['catchment_ok'].fillna(False).astype(bool) if 'catchment_ok' in out.columns else True
    if elev_min is not None and 'pour_elev_m' in out.columns:
        elev_ok = out['pour_elev_m'].fillna(-1e9) >= elev_min
    else:
        elev_ok = True
    screen = chem_ok & catch_ok & elev_ok

    mode = rank_by_mode(cfg)
    sort_cols = add_walk_sort_keys(out, mode)

    out['walk_rank'] = pd.Series(pd.NA, index=out.index, dtype='Int64')
    ranked = out.loc[screen].sort_values(sort_cols, ascending=True)
    out.loc[ranked.index, 'walk_rank'] = np.arange(1, len(ranked) + 1)

    out['access_rank'] = pd.Series(pd.NA, index=out.index, dtype='Int64')
    access_screen = screen & (out['access_ok'] == 'yes')
    ar = out.loc[access_screen].sort_values(sort_cols, ascending=True)
    out.loc[ar.index, 'access_rank'] = np.arange(1, len(ar) + 1)

    out['rank_in_access_type'] = pd.Series(pd.NA, index=out.index, dtype='Int64')
    for atype, grp in out.loc[screen].groupby('access_type', dropna=False):
        g = grp.sort_values(sort_cols, ascending=True)
        out.loc[g.index, 'rank_in_access_type'] = np.arange(1, len(g) + 1)

    return out.drop(columns=sort_cols, errors='ignore')


def padus_open_layer(cfg):
    """PAD-US polygons filtered to open-ish public managers for the GPKG."""
    gdf = _load_gdf(padus_path(cfg))
    if gdf.empty:
        return gdf
    types = []
    managers = []
    pubs = []
    for _, row in gdf.iterrows():
        at, mang, pub = padus_manager_type(row)
        types.append(at)
        managers.append(mang)
        pubs.append(pub)
    gdf = gdf.copy()
    gdf['access_type'] = types
    gdf['manager_name'] = managers
    # GPKG/SQLite is case-insensitive — drop PAD-US Pub_Access before
    # writing our lowercase pub_access label.
    drop = [c for c in gdf.columns if c.lower() in (
        'pub_access', 'access_type', 'manager_name',
    ) and c not in ('access_type', 'manager_name')]
    gdf = gdf.drop(columns=drop, errors='ignore')
    gdf['pub_access'] = pubs
    keep = gdf['access_type'].isin(
        ['blm', 'usfs', 'state_park', 'state_other', 'nps', 'local_gov']
    )
    # Keep a lean attribute set for the GPKG
    prefer = [
        'Unit_Nm', 'Mang_Name', 'Own_Name', 'Des_Tp', 'GAP_Sts', 'State_Nm',
        'access_type', 'manager_name', 'pub_access', 'geometry',
    ]
    cols = [c for c in prefer if c in gdf.columns]
    return gdf.loc[keep, cols].copy()


def mlrs_layer(cfg):
    return _load_gdf(mlrs_path(cfg))


def annotate_campaign(cfg, pans, pour_points, walk, hobby=None):
    """Full pass: annotate pans + pours, roll up to walk, re-rank.

    Returns (pans, pour_points, walk, padus_open, mlrs_claims).
    """
    if not land_access_enabled(cfg):
        return pans, pour_points, walk, None, None

    ensure_land_access_data(cfg)

    hobby_gdf = hobby
    if hobby is not None and not hasattr(hobby, 'geometry'):
        import geopandas as gpd
        from shapely.geometry import Point
        if 'lon' in hobby.columns and 'lat' in hobby.columns:
            hobby_gdf = gpd.GeoDataFrame(
                hobby.copy(),
                geometry=[
                    Point(xy) if pd.notna(xy[0]) and pd.notna(xy[1]) else None
                    for xy in zip(hobby['lon'], hobby['lat'])
                ],
                crs='EPSG:4326',
            )

    if pans is not None and len(pans):
        pans = annotate_points(pans, cfg, hobby_gdf=hobby_gdf)
    if pour_points is not None and len(pour_points):
        pour_points = annotate_points(pour_points, cfg, hobby_gdf=hobby_gdf)

    cell = _cell_access_from_pans(pans)
    if cell is not None and len(cell) and walk is not None and len(walk):
        walk = walk.drop(
            columns=[c for c in ACCESS_COLS if c in walk.columns],
            errors='ignore',
        )
        walk = walk.merge(cell, on='block_id', how='left')
        no_pan = ~walk['block_id'].isin(cell['block_id'])
        # Fill cells with no pans from pour annotation
        if pour_points is not None and len(pour_points) and 'block_id' in pour_points.columns:
            pour_cell = pour_points.drop(columns='geometry', errors='ignore')
            pour_cell = pour_cell.drop_duplicates('block_id')
            for col in ACCESS_COLS:
                if col not in walk.columns:
                    walk[col] = pd.NA
                if col in pour_cell.columns:
                    if col in ('tribal_land', 'tribal_name', 'rules_note'):
                        miss = no_pan
                    else:
                        miss = walk[col].isna() | (walk[col].astype(str) == '')
                    if miss.any():
                        m = dict(zip(pour_cell['block_id'], pour_cell[col]))
                        walk.loc[miss, col] = walk.loc[miss, 'block_id'].map(m)
        walk['access_type'] = walk['access_type'].fillna('unknown')
        walk['access_ok'] = walk['access_ok'].fillna('no')
        walk['claim_status'] = walk['claim_status'].fillna('unchecked')
        walk['access_reason'] = walk['access_reason'].fillna('no pan/pour geometry')
        walk['tribal_land'] = walk['tribal_land'].fillna(False).astype(bool)
        walk['tribal_name'] = walk['tribal_name'].fillna('')
        walk['rules_note'] = walk['rules_note'].fillna('')
        trap = _cell_trap_from_pans(pans)
        if trap is not None and len(trap):
            walk = walk.drop(columns=['trap_score'], errors='ignore').merge(
                trap, on='block_id', how='left',
            )
        walk = apply_access_ranks(walk, cfg)
    elif walk is not None and len(walk):
        walk = apply_access_ranks(walk, cfg)

    pans = sync_walk_rank(pans, walk)
    return pans, pour_points, walk, padus_open_layer(cfg), mlrs_layer(cfg)


def sync_walk_rank(points, walk):
    """Copy the final walk_rank onto per-cell points (pans, NURE spots) by block_id."""
    if (points is None or len(points) == 0 or walk is None or len(walk) == 0
            or 'block_id' not in points.columns or 'walk_rank' not in walk.columns):
        return points
    rank = walk.drop_duplicates('block_id').set_index('block_id')['walk_rank']
    points = points.copy()
    points['walk_rank'] = points['block_id'].map(rank)
    return points
