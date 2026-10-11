"""Data verification and source register (CIM 2018 §2.1.1.1, §2.3).

Validation (pipeline.data_validation) asks whether a record is well formed.
Verification asks whether legacy results are reasonable and accurate. This
module records three checks for a belt and a register of every dataset:

  a. Re-analysis pairs: previous_lab_id resolution, then same-field_id
     resubmissions at identical coordinates. HARD = |a-b|/(a+b).
  b. Airborne eTh vs stream-sediment Th, legacy rule vs validated samples.
  c. NURE stream sediment vs WGS OFR 2026-02 mine-waste sites (≤ 5 km).
  d. data/VERIFICATION_REGISTER.csv: path, URL, size, sha256, mtime,
     validation checks, verification result, status — one row per dataset
     per belt.

CLI:
  python -m pipeline.verification --config configs/ne_washington/config.yaml [--register]

Outputs ({outputs_dir}/tables/):
  verification_reanalysis_pairs.csv, verification_reanalysis_detail.csv,
  verification_radiometric.csv, verification_wgs_nure.csv
"""

import argparse
import copy
import glob
import hashlib
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.utils import (ensure_outputs, load_nure, load_radiometric_at_points, out,
                            radiometric_tif_path, wgs_path)

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTER_PATH = REPO_ROOT / 'data' / 'VERIFICATION_REGISTER.csv'
HSSR_ZIP = REPO_ROOT / 'data' / 'nure' / 'raw' / 'nuresed-csv.zip'
NURE_WA_STATEWIDE = 'data/nure/nure_wa_sediment.csv'

MIN_PAIRS = 10
WGS_RADIUS_KM = 5.0
HASH_MAX_BYTES = 500 * 1024 ** 2
WGS_ELEMENTS = ['Th', 'La', 'Ce', 'Nd', 'Y', 'Au']

# Inline fallback when pipeline.data_validation is not importable.
PRECISION_OK = {'1 second', '5 seconds', '10 seconds', '15 seconds', '0.01 degree'}
SAMPLE_SOURCE_OK = 'stream/river'

_NON_ANALYTE = {
    'lat', 'lon', 'lat_orig', 'long_orig', 'depth', 'lab_id', 'field_id', 'job_id',
    'mineralization', 'collection_protocol', 'previous_job_id', 'previous_lab_id', 'SplWtAu',
}

PAIR_SUMMARY_COLUMNS = [
    'scope', 'pair_source', 'element', 'n_pairs_analysed', 'n_both_detected',
    'n_one_censored', 'n_one_censored_detected_above_partner_mdl', 'n_both_censored',
    'meets_min_pairs', 'median_hard', 'pct_hard_le_10', 'pct_hard_le_20',
    'median_log10_ratio', 'spearman_rho', 'spearman_p', 'pct_values_on_6step_grid',
    'status', 'note',
]


def _repo_path(p):
    if p is None:
        return None
    p = str(p)
    return p if os.path.isabs(p) else str(REPO_ROOT / p)


def _rel(p):
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def read_nure_raw(path):
    """Read an NGDB/NURE extract without any censoring, ids as strings."""
    return pd.read_csv(_repo_path(path), low_memory=False,
                       dtype={'lab_id': str, 'field_id': str, 'job_id': str,
                              'previous_lab_id': str, 'previous_job_id': str})


def element_columns(df):
    """Numeric analyte columns (ids, coordinates and sample weights excluded)."""
    return [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in _NON_ANALYTE and not c.endswith('_mdl') and not c.endswith('_censored')]


# ── Validation hook ───────────────────────────────────────────────────────────

def _inline_validate(df_raw, cfg):
    """Minimal filter: coord precision set, stream/river medium, half-MDL all negatives."""
    df = df_raw.copy()
    ledger = []
    if 'coord_prec_est' in df.columns:
        keep = df['coord_prec_est'].isin(PRECISION_OK)
        ledger.append({'check': 'coord_precision', 'n_in': len(df), 'n_out': int(keep.sum())})
        df = df[keep]
    if 'sample_source' in df.columns:
        keep = df['sample_source'] == SAMPLE_SOURCE_OK
        ledger.append({'check': 'sample_medium', 'n_in': len(df), 'n_out': int(keep.sum())})
        df = df[keep]
    for c in element_columns(df):
        v = pd.to_numeric(df[c], errors='coerce')
        neg = v < 0
        df[c] = v.where(~neg, v.abs() / 2.0)
        df[f'{c}_censored'] = neg.astype('boolean').mask(v.isna())
    return df, pd.DataFrame(ledger)


def validate_samples(df_raw, cfg):
    """Return (clean_df, ledger_df, method). Uses data_validation.validate_nure when importable."""
    try:
        from pipeline.data_validation import validate_nure
    except Exception:
        clean, ledger = _inline_validate(df_raw, cfg)
        return clean, ledger, 'inline minimal filter (precision set, stream/river, half-MDL)'
    clean, ledger = validate_nure(df_raw, cfg)
    return clean, ledger, 'pipeline.data_validation.validate_nure'


# ── a. Re-analysis pairs ──────────────────────────────────────────────────────

def hard(a, b):
    """Half absolute relative difference |a-b|/(a+b); NaN where a+b <= 0."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    s = a + b
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(s > 0, np.abs(a - b) / s, np.nan)


def pair_precision_stats(a, b, min_pairs=MIN_PAIRS):
    """HARD / bias / rank stats for detected pairs (a = original, b = re-analysis)."""
    from scipy.stats import spearmanr

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    a, b = a[ok], b[ok]
    res = {'n_both_detected': int(len(a)), 'meets_min_pairs': bool(len(a) >= min_pairs),
           'median_hard': np.nan, 'pct_hard_le_10': np.nan, 'pct_hard_le_20': np.nan,
           'median_log10_ratio': np.nan, 'spearman_rho': np.nan, 'spearman_p': np.nan}
    if len(a) < min_pairs:
        return res
    h = hard(a, b)
    res.update(median_hard=round(float(np.median(h)), 4),
               pct_hard_le_10=round(float(np.mean(h <= 0.10) * 100), 1),
               pct_hard_le_20=round(float(np.mean(h <= 0.20) * 100), 1),
               median_log10_ratio=round(float(np.median(np.log10(b / a))), 4))
    if np.ptp(a) > 0 and np.ptp(b) > 0:
        rho, p = spearmanr(a, b)
        res.update(spearman_rho=round(float(rho), 4), spearman_p=float(p))
    return res


def _norm_id(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    if not s or s == 'NAN':
        return None
    s = re.sub(r'\.0+$', '', s)
    return s.lstrip('0') or '0'


def _find_token(data, token):
    """True if token occurs in data as a whole alphanumeric token (leading zeros allowed)."""
    start = 0
    while True:
        i = data.find(token, start)
        if i < 0:
            return False
        j = i
        while j > 0 and data[j - 1:j] == b'0':
            j -= 1
        before = data[j - 1:j] if j > 0 else b''
        after = data[i + len(token):i + len(token) + 1]
        if not before.isalnum() and not after.isalnum():
            return True
        start = i + 1


def scan_hssr_for_ids(zip_path, ids):
    """Return the subset of numeric ids found as whole tokens in the national HSSR CSV zip."""
    ids = sorted({i for i in ids if i and i.isdigit()})
    if not ids or not zip_path or not os.path.exists(zip_path):
        return None
    with zipfile.ZipFile(zip_path) as zf:
        name = next((n for n in zf.namelist() if n.lower().endswith('.csv')), None)
        if name is None:
            return None
        data = zf.read(name)
    return {i for i in ids if _find_token(data, i.encode())}


def previous_lab_id_candidates(df):
    """Normalized previous_lab_id values in df (empty set if the column is absent)."""
    if 'previous_lab_id' not in df.columns:
        return set()
    return {n for n in df['previous_lab_id'].map(_norm_id) if n is not None}


def resolve_previous_lab_ids(df, pools, hssr_found=None):
    """Match previous_lab_id against lab_id / field_id in each pool.

    pools: {name: DataFrame}. hssr_found: ids already found in the national
    HSSR zip (None = not scanned). Returns (pairs_df, summary_dict); pairs_df
    has orig_lab_id / re_lab_id / pool for resolved rows only.
    """
    summary = {'n_candidates': 0, 'n_unique_ids': 0, 'resolved_by_pool': {}, 'n_resolved': 0,
               'submitters': {}, 'sample_sources': {}, 'hssr_scanned': False, 'hssr_matches': None}
    if 'previous_lab_id' not in df.columns:
        summary['note'] = 'column previous_lab_id absent'
        return pd.DataFrame(columns=['orig_lab_id', 're_lab_id', 'pool']), summary
    cand = df[df['previous_lab_id'].notna()].copy()
    cand['prev_norm'] = cand['previous_lab_id'].map(_norm_id)
    cand = cand[cand['prev_norm'].notna()]
    summary.update(n_candidates=int(len(cand)), n_unique_ids=int(cand['prev_norm'].nunique()))
    for col in ('submitter', 'sample_source'):
        if col in cand.columns:
            key = 'submitters' if col == 'submitter' else 'sample_sources'
            summary[key] = cand[col].fillna('NaN').value_counts().to_dict()
    rows = []
    resolved = set()
    for pname, pool in pools.items():
        lookup = {}
        for col in ('lab_id', 'field_id'):
            if col in pool.columns:
                for lab, v in zip(pool['lab_id'], pool[col]):
                    n = _norm_id(v)
                    if n is not None:
                        lookup.setdefault(n, lab)
        hits = 0
        for _, r in cand.iterrows():
            orig = lookup.get(r['prev_norm'])
            if orig is not None and orig != r['lab_id']:
                hits += 1
                if r['lab_id'] not in resolved:
                    resolved.add(r['lab_id'])
                    rows.append({'orig_lab_id': orig, 're_lab_id': r['lab_id'], 'pool': pname})
        summary['resolved_by_pool'][pname] = hits
    if hssr_found is not None:
        summary['hssr_scanned'] = True
        summary['hssr_matches'] = int(cand['prev_norm'].isin(hssr_found).sum())
    summary['n_resolved'] = len(resolved)
    return pd.DataFrame(rows, columns=['orig_lab_id', 're_lab_id', 'pool']), summary


def field_id_pairs(df):
    """Pairs of different lab_ids sharing field_id, lat/lon (5 dp) and depth.

    The earliest submission (date_submitted, then job_id, lab_id) is the
    original; each later lab_id pairs with it. Returns orig_lab_id, re_lab_id,
    same_job.
    """
    need = {'lab_id', 'field_id', 'lat', 'lon'}
    if not need.issubset(df.columns):
        return pd.DataFrame(columns=['orig_lab_id', 're_lab_id', 'same_job'])
    d = df[df['field_id'].notna() & df['lab_id'].notna()].copy()
    d['_lat'] = d['lat'].round(5)
    d['_lon'] = d['lon'].round(5)
    d['_depth'] = d['depth'].fillna('').astype(str) if 'depth' in d.columns else ''
    d['_date'] = pd.to_datetime(d.get('date_submitted'), errors='coerce')
    d['_job'] = d['job_id'].fillna('').astype(str) if 'job_id' in d.columns else ''
    rows = []
    for _, g in d.groupby(['field_id', '_lat', '_lon', '_depth'], sort=True):
        g = g.drop_duplicates('lab_id')
        if len(g) < 2:
            continue
        g = g.sort_values(['_date', '_job', 'lab_id'], na_position='last')
        first = g.iloc[0]
        for _, r in g.iloc[1:].iterrows():
            rows.append({'orig_lab_id': first['lab_id'], 're_lab_id': r['lab_id'],
                         'same_job': bool(first['_job'] == r['_job'] and first['_job'] != '')})
    return pd.DataFrame(rows, columns=['orig_lab_id', 're_lab_id', 'same_job'])


def pair_detail(pairs, df, elements, scope, pair_source_col=None, pair_source=None):
    """Long table: one row per pair per element where both sides were analysed."""
    idx = df.drop_duplicates('lab_id').set_index('lab_id')
    rows = []
    for _, p in pairs.iterrows():
        if p['orig_lab_id'] not in idx.index or p['re_lab_id'] not in idx.index:
            continue
        o, r = idx.loc[p['orig_lab_id']], idx.loc[p['re_lab_id']]
        src = p[pair_source_col] if pair_source_col else pair_source
        for el in elements:
            a = pd.to_numeric(o.get(el), errors='coerce')
            b = pd.to_numeric(r.get(el), errors='coerce')
            if pd.isna(a) or pd.isna(b) or a == 0 or b == 0:
                continue
            ca, cb = bool(a < 0), bool(b < 0)
            va = abs(a) / 2 if ca else a
            vb = abs(b) / 2 if cb else b
            both = not ca and not cb
            rows.append({
                'scope': scope, 'pair_source': src, 'element': el,
                'orig_lab_id': p['orig_lab_id'], 're_lab_id': p['re_lab_id'],
                'orig_job_id': o.get('job_id'), 're_job_id': r.get('job_id'),
                'field_id': o.get('field_id'), 'lat': o.get('lat'), 'lon': o.get('lon'),
                'orig_raw': a, 're_raw': b, 'orig_censored': ca, 're_censored': cb,
                'orig_value_half_mdl': va, 're_value_half_mdl': vb,
                'hard': float(hard(a, b)) if both else np.nan,
                'log10_ratio': float(np.log10(b / a)) if both else np.nan,
            })
    return pd.DataFrame(rows)


_ES_MANTISSAS = np.array([1.0, 1.5, 2.0, 3.0, 5.0, 7.0])


def on_six_step_grid(values, rtol=1e-6):
    """Boolean array: value is 1, 1.5, 2, 3, 5 or 7 × 10^k (semi-quantitative spectrography steps)."""
    v = np.asarray(values, dtype=float)
    out_ = np.zeros(v.shape, dtype=bool)
    ok = np.isfinite(v) & (v > 0)
    mant = v[ok] / 10.0 ** np.floor(np.log10(v[ok]) + 1e-9)
    out_[ok] = np.isclose(mant[:, None], _ES_MANTISSAS[None, :], rtol=rtol).any(axis=1)
    return out_


def _pooled_row(g, scope, src):
    both = g[~g['orig_censored'] & ~g['re_censored']]
    h = both['hard'].dropna().values
    row = {c: np.nan for c in PAIR_SUMMARY_COLUMNS}
    row.update(scope=scope, pair_source=src, element='ALL_ELEMENTS_POOLED',
               n_pairs_analysed=int(len(g)), n_both_detected=int(len(both)),
               n_one_censored=int((g['orig_censored'] ^ g['re_censored']).sum()),
               n_both_censored=int((g['orig_censored'] & g['re_censored']).sum()),
               meets_min_pairs=bool(len(h) >= MIN_PAIRS),
               status='pooled across elements; Spearman not computed (mixed scales)')
    if len(h):
        row.update(median_hard=round(float(np.median(h)), 4),
                   pct_hard_le_10=round(float(np.mean(h <= 0.10) * 100), 1),
                   pct_hard_le_20=round(float(np.mean(h <= 0.20) * 100), 1),
                   median_log10_ratio=round(float(both['log10_ratio'].median()), 4),
                   pct_values_on_6step_grid=round(float(np.mean(on_six_step_grid(
                       np.r_[both['orig_raw'].values, both['re_raw'].values])) * 100), 1))
    return row


def summarize_pairs(detail, min_pairs=MIN_PAIRS):
    """Per (scope, pair_source, element) precision summary from pair_detail output.

    Adds one ALL_ELEMENTS_POOLED row per (scope, pair_source).
    """
    rows = []
    if detail is None or detail.empty:
        return pd.DataFrame(columns=PAIR_SUMMARY_COLUMNS)
    for (scope, src), g in detail.groupby(['scope', 'pair_source'], sort=True):
        rows.append(_pooled_row(g, scope, src))
    for (scope, src, el), g in detail.groupby(['scope', 'pair_source', 'element'], sort=True):
        both = g[~g['orig_censored'] & ~g['re_censored']]
        one = g[g['orig_censored'] ^ g['re_censored']]
        # Detected value above the censored partner's MDL = the two analyses disagree.
        det = np.where(one['orig_censored'], one['re_raw'], one['orig_raw'])
        mdl = np.where(one['orig_censored'], one['orig_raw'].abs(), one['re_raw'].abs())
        stats = pair_precision_stats(both['orig_raw'], both['re_raw'], min_pairs)
        grid = on_six_step_grid(np.r_[both['orig_raw'].values, both['re_raw'].values])
        rows.append({
            'scope': scope, 'pair_source': src, 'element': el,
            'n_pairs_analysed': int(len(g)),
            'n_one_censored': int(len(one)),
            'n_one_censored_detected_above_partner_mdl': int(np.sum(det > mdl)),
            'n_both_censored': int((g['orig_censored'] & g['re_censored']).sum()),
            **stats,
            'pct_values_on_6step_grid': round(float(grid.mean() * 100), 1) if len(grid) else np.nan,
            'status': 'ok' if stats['meets_min_pairs'] else f'< {min_pairs} detected pairs',
            'note': '',
        })
    return pd.DataFrame(rows)[PAIR_SUMMARY_COLUMNS]


def reanalysis_verification(scopes, hssr_zip=HSSR_ZIP):
    """Run previous_lab_id resolution and field_id pairing for each scope.

    scopes: {scope_name: raw DataFrame}. All scopes are searched for
    previous_lab_id. Returns (summary_df, detail_df, resolution_by_scope).
    """
    pools = dict(scopes)
    all_ids = set().union(*(previous_lab_id_candidates(df) for df in scopes.values()))
    hssr_found = scan_hssr_for_ids(str(hssr_zip), all_ids) if hssr_zip else None
    summaries, details, resolution = [], [], {}
    for scope, df in scopes.items():
        els = element_columns(df)
        pairs, res = resolve_previous_lab_ids(df, pools, hssr_found=hssr_found)
        resolution[scope] = res
        pools_txt = '; '.join(f'{k}: {v}' for k, v in res['resolved_by_pool'].items())
        hssr_txt = (f"national HSSR zip scanned, {res['hssr_matches']} id matches"
                    if res['hssr_scanned'] else 'national HSSR zip not scanned')
        subs = '; '.join(f'{k} ({v})' for k, v in res['submitters'].items())
        media = '; '.join(f'{k} ({v})' for k, v in res['sample_sources'].items())
        note = (f"{res['n_candidates']} rows carry previous_lab_id ({res['n_unique_ids']} unique); "
                f"{res['n_resolved']} resolve to a lab_id/field_id in [{pools_txt}]; {hssr_txt}. "
                f'Submitters: {subs}. Media: {media}. Unresolved IDs look like USGS LIMS '
                'numbers whose originals are not in the NGDB extracts.')
        if len(pairs):
            det = pair_detail(pairs, pd.concat(pools.values()).drop_duplicates('lab_id'),
                              els, scope, pair_source='previous_lab_id')
            details.append(det)
            s = summarize_pairs(det)
            s['note'] = note
            summaries.append(s)
        if res['n_candidates'] - res['n_resolved'] > 0 or not len(pairs):
            summaries.append(pd.DataFrame([{
                'scope': scope, 'pair_source': 'previous_lab_id', 'element': 'ALL',
                'n_pairs_analysed': res['n_resolved'], 'n_both_detected': np.nan,
                'n_one_censored': np.nan, 'n_one_censored_detected_above_partner_mdl': np.nan,
                'n_both_censored': np.nan, 'meets_min_pairs': False,
                'status': (f"resolution: {res['n_resolved']} of {res['n_candidates']} resolved"
                           if res['n_candidates'] else 'no previous_lab_id values'),
                'note': note,
            }]))
        fp = field_id_pairs(df)
        if len(fp):
            fp['pair_source'] = np.where(fp['same_job'], 'field_id_same_job', 'field_id_cross_job')
            det = pair_detail(fp, df, els, scope, pair_source_col='pair_source')
            if len(det):
                pooled = det.copy()
                pooled['pair_source'] = 'field_id_any'
                details.append(det)
                s = summarize_pairs(pd.concat([det, pooled], ignore_index=True))
                s['note'] = (f'{len(fp)} field_id pairs ({int(fp["same_job"].sum())} same job_id); '
                             'same field_id + lat/lon (5 dp) + depth, different lab_id. '
                             'Analytical method may differ between jobs. High '
                             'pct_values_on_6step_grid = semi-quantitative spectrography steps '
                             '(1, 1.5, 2, 3, 5, 7 × 10^k): HARD is quantized, adjacent steps give 0.14–0.2.')
                summaries.append(s)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(
        columns=PAIR_SUMMARY_COLUMNS)
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return summary[PAIR_SUMMARY_COLUMNS], detail, resolution


# ── b. Airborne eTh vs stream Th ──────────────────────────────────────────────

def concordance(x, y):
    """Spearman ρ between finite pairs. Returns dict n, rho, p (NaN if n < 3)."""
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return {'n': n, 'rho': np.nan, 'p': np.nan}
    rho, p = spearmanr(x[ok], y[ok])
    return {'n': n, 'rho': float(rho), 'p': float(p)}


def _positive(v):
    v = np.asarray(v, dtype=float)
    return np.where(v > 0, v, np.nan)


def _composition(df_raw, idx):
    """'sample_source (n) | coord_prec_est (n)' for the rows of df_raw in idx."""
    sub = df_raw.loc[df_raw.index.intersection(idx)]
    parts = []
    for col in ('sample_source', 'coord_prec_est'):
        if col in sub.columns and len(sub):
            vc = sub[col].fillna('NaN').value_counts()
            parts.append(', '.join(f'{k} {v}' for k, v in vc.items()))
    return ' | '.join(parts)


def radiometric_verification(cfg, df_raw, clean, method):
    """Spearman ρ(stream Th, airborne eTh) under the legacy and validated rules."""
    cfg_abs = copy.deepcopy(cfg)
    cfg_abs['data']['nure_csv'] = _repo_path(cfg['data']['nure_csv'])
    legacy = load_nure(cfg_abs)
    rows = []

    def _row(variant, samples, censoring, frame, th, n_in, note=''):
        rad = load_radiometric_at_points(cfg, frame['lon'].values, frame['lat'].values)
        x, y = _positive(th), _positive(rad['rad_eTh'].values)
        c = concordance(x, y)
        used = frame.index[np.isfinite(x) & np.isfinite(y)]
        comp = _composition(df_raw, used)
        rows.append({'variant': variant, 'samples': samples, 'censoring_rule': censoring,
                     'n_samples_input': n_in, 'n_pairs': c['n'],
                     'spearman_rho': round(c['rho'], 4) if np.isfinite(c['rho']) else np.nan,
                     'p_value': c['p'], 'validation_method': method,
                     'pairs_composition': comp, 'note': note})

    if radiometric_tif_path(cfg, 'eth') is None:
        return pd.DataFrame([{'variant': 'none', 'n_pairs': 0, 'spearman_rho': np.nan,
                              'p_value': np.nan, 'validation_method': method,
                              'note': 'data.radiometric_eth_tif not configured or missing'}])
    _row('i_legacy_all', 'all samples in extract',
         'legacy load_nure: |v|<=10 → |v|/2, |v|>10 → NaN', legacy, legacy['Th'], len(legacy),
         'Reproduces the Task 1 / Fig 1b result of record.')
    leg_val = legacy.loc[legacy.index.intersection(clean.index)]
    _row('ia_legacy_rule_validated_samples', 'validated samples',
         'legacy load_nure rule', leg_val, leg_val['Th'], len(leg_val),
         'Isolates the sample filter from the censoring rule.')
    raw_th = pd.to_numeric(df_raw.loc[clean.index, 'Th'], errors='coerce')
    mdls = raw_th[raw_th < 0].abs().value_counts()
    cens_txt = (f"{int((raw_th < 0).sum())} of {int(raw_th.notna().sum())} validated Th results "
                f"censored (MDL {', '.join(f'{k:g} ppm x{v}' for k, v in mdls.items())})")
    _row('ii_validated_half_mdl', 'validated samples',
         'all negatives → |v|/2 (any magnitude)', clean, clean['Th'], len(clean),
         cens_txt + '; ranks are dominated by ties at half-MDL.')
    det = clean
    if 'Th_censored' in clean.columns:
        det = clean[~clean['Th_censored'].fillna(True).astype(bool)]
    _row('iii_validated_detected_only', 'validated samples, Th detected',
         'censored Th excluded', det, det['Th'], len(clean),
         'Sensitivity: detected Th only. ' + cens_txt + '.')
    return pd.DataFrame(rows)


# ── c. NURE vs WGS OFR 2026-02 ────────────────────────────────────────────────

def _parse_censored(v):
    """'<4' → (2.0, True); 5.1 → (5.1, False); blank → (NaN, False)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan, False
    s = str(v).strip()
    if s.startswith('<'):
        try:
            return float(s[1:]) / 2.0, True
        except ValueError:
            return np.nan, False
    try:
        return float(s), False
    except ValueError:
        return np.nan, False


def load_wgs_sites(xl_path, elements=WGS_ELEMENTS):
    """Per-site medians (censored '<x' → x/2) from the WGS 'Geochemical Data' sheet."""
    raw = pd.read_excel(xl_path, sheet_name='Geochemical Data', header=0)
    units = raw.iloc[0]
    df = raw.iloc[1:].reset_index(drop=True)
    df = df[df['Site_Name'].notna() & (df['Site_Name'] != 'UCC')].copy()
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    rows = []
    for site, g in df.groupby('Site_Name', sort=True):
        row = {'wgs_site': site, 'wgs_n_samples': int(len(g)),
               'wgs_lat': float(g['Latitude'].median()), 'wgs_lon': float(g['Longitude'].median()),
               'wgs_materials': '; '.join(sorted(g['Material'].dropna().astype(str).unique()))}
        for el in elements:
            if el not in g.columns:
                continue
            parsed = [_parse_censored(v) for v in g[el]]
            vals = np.array([p[0] for p in parsed], dtype=float)
            row[f'wgs_{el}_ppm'] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
            row[f'wgs_{el}_n_censored'] = int(sum(p[1] for p in parsed))
            row[f'wgs_{el}_unit'] = units.get(el)
        rows.append(row)
    return pd.DataFrame(rows)


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def wgs_nure_comparison(wgs_sites, nure, radius_km=WGS_RADIUS_KM, elements=WGS_ELEMENTS,
                        wgs_bbox=None, exclude_sites=(), nure_any=None):
    """Nearest validated NURE sample for each WGS site; element comparison within radius_km.

    nure_any: optional unvalidated extract, reported as distance context only.
    """
    rows = []
    lons, lats = nure['lon'].values, nure['lat'].values
    for _, s in wgs_sites.iterrows():
        row = s.to_dict()
        if nure_any is not None and len(nure_any):
            da = haversine_km(s['wgs_lon'], s['wgs_lat'], nure_any['lon'].values, nure_any['lat'].values)
            ia = int(np.nanargmin(da))
            row.update(nure_any_nearest_lab_id=nure_any.iloc[ia].get('lab_id'),
                       nure_any_nearest_km=round(float(da[ia]), 3),
                       n_nure_any_within_radius=int(np.sum(da <= radius_km)),
                       nure_any_within_radius_composition=_composition(
                           nure_any, nure_any.index[da <= radius_km]))
        d = haversine_km(s['wgs_lon'], s['wgs_lat'], lons, lats) if len(nure) else np.array([])
        row['in_wgs_bbox'] = (bool(wgs_bbox['lon_min'] <= s['wgs_lon'] <= wgs_bbox['lon_max']
                                   and wgs_bbox['lat_min'] <= s['wgs_lat'] <= wgs_bbox['lat_max'])
                              if wgs_bbox else np.nan)
        row['excluded_by_config'] = s['wgs_site'] in set(exclude_sites)
        if len(d) == 0:
            row['note'] = 'no validated NURE samples'
            rows.append(row)
            continue
        i = int(np.nanargmin(d))
        near = nure.iloc[i]
        row.update(nure_nearest_lab_id=near.get('lab_id'), nure_nearest_km=round(float(d[i]), 3),
                   nure_nearest_source=near.get('_extract'),
                   n_nure_within_radius=int(np.sum(d <= radius_km)))
        within = d[i] <= radius_km
        for el in elements:
            if f'wgs_{el}_ppm' not in row:
                continue
            nv = pd.to_numeric(near.get(el), errors='coerce')
            cens = near.get(f'{el}_censored')
            row[f'nure_{el}_ppm'] = float(nv) if within and pd.notna(nv) else np.nan
            row[f'nure_{el}_censored'] = (bool(cens) if within and pd.notna(cens) else pd.NA)
            w = row[f'wgs_{el}_ppm']
            row[f'log10_wgs_over_nure_{el}'] = (
                round(float(np.log10(w / nv)), 3)
                if within and pd.notna(nv) and nv > 0 and pd.notna(w) and w > 0 else np.nan)
        row['note'] = (f'nearest validated NURE within {radius_km:g} km; mine waste vs stream '
                       'sediment — disagreement expected' if within
                       else f'no validated NURE sample within {radius_km:g} km')
        rows.append(row)
    res = pd.DataFrame(rows)
    n_overlap = int((res.get('nure_nearest_km', pd.Series(dtype=float)) <= radius_km).sum())
    near_txt = ''
    if len(res) and 'nure_nearest_km' in res:
        near_txt = (f"; nearest validated sample {res['nure_nearest_km'].min():.1f}–"
                    f"{res['nure_nearest_km'].max():.1f} km")
    any_txt = ''
    if 'nure_any_nearest_km' in res:
        n_any = int((res['nure_any_nearest_km'] <= radius_km).sum())
        any_txt = (f'; {n_any} sites have an unvalidated sample (any medium / precision) within '
                   f'{radius_km:g} km (context only, not compared)')
    summary = {'wgs_site': '__summary__', 'wgs_n_samples': int(wgs_sites['wgs_n_samples'].sum())
               if len(wgs_sites) else 0,
               'n_nure_within_radius': n_overlap,
               'note': (f'{n_overlap} of {len(wgs_sites)} WGS sites have a validated NURE '
                        f'stream-sediment sample within {radius_km:g} km'
                        + ('' if n_overlap else ' — no usable overlap; comparison not possible')
                        + near_txt + any_txt + f'; validated NURE pool n={len(nure)}')}
    return pd.concat([res, pd.DataFrame([summary])], ignore_index=True)


# ── d. Source register ────────────────────────────────────────────────────────

def sha256_file(path, max_bytes=HASH_MAX_BYTES):
    """Hex sha256, or None when the file is larger than max_bytes."""
    if os.path.getsize(path) > max_bytes:
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def file_facts(paths, max_bytes=HASH_MAX_BYTES):
    """Size / sha256 / mtime for one file or a manifest of several.

    Several files: sha256 is over sorted '<name> <sha256>' lines (a manifest
    hash); files over max_bytes are left out of the manifest and listed.
    """
    paths = [p for p in (paths if isinstance(paths, (list, tuple)) else [paths]) if p]
    present = sorted(p for p in paths if os.path.isfile(p))
    facts = {'n_files': len(present), 'file_size_bytes': np.nan, 'sha256': '',
             'file_mtime_utc': '', 'hash_note': ''}
    if not present:
        facts['hash_note'] = 'file missing locally' if paths else ''
        return facts
    facts['file_size_bytes'] = int(sum(os.path.getsize(p) for p in present))
    mtimes = [os.path.getmtime(p) for p in present]
    facts['file_mtime_utc'] = datetime.fromtimestamp(max(mtimes), timezone.utc).strftime(
        '%Y-%m-%dT%H:%M:%SZ')
    skipped, lines = [], []
    for p in present:
        h = sha256_file(p, max_bytes)
        if h is None:
            skipped.append(os.path.basename(p))
        else:
            lines.append(f'{os.path.basename(p)} {h}')
    if len(present) == 1:
        facts['sha256'] = lines[0].split(' ')[1] if lines else ''
    elif lines:
        facts['sha256'] = 'manifest:' + hashlib.sha256('\n'.join(lines).encode()).hexdigest()
    notes = []
    if skipped:
        notes.append(f'sha256 skipped (> {max_bytes // 1024 ** 2} MB): {", ".join(skipped)}')
    if len(present) > 1:
        earliest = datetime.fromtimestamp(min(mtimes), timezone.utc).strftime('%Y-%m-%d')
        notes.append(f'{len(present)} files; mtime range {earliest} → {facts["file_mtime_utc"][:10]}')
    facts['hash_note'] = '; '.join(notes)
    return facts


def _read_csv_opt(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def verification_results(outputs_dir):
    """Summarize existing verification CSVs for one belt: {key: (method, result)}."""
    res = {}
    tdir = os.path.join(_repo_path(outputs_dir), 'tables')
    pairs = _read_csv_opt(os.path.join(tdir, 'verification_reanalysis_pairs.csv'))
    if pairs is not None and len(pairs):
        for scope in pairs['scope'].unique():
            s = pairs[pairs['scope'] == scope]
            prev = s[s['pair_source'] == 'previous_lab_id']
            prev_all = prev[prev['element'] == 'ALL']
            fid = s[(s['pair_source'] == 'field_id_any') & s['meets_min_pairs'].astype(bool)
                    & (s['element'] != 'ALL_ELEMENTS_POOLED')]
            prev = prev[prev['element'] != 'ALL_ELEMENTS_POOLED']
            parts = []
            if len(prev_all):
                parts.append(str(prev_all['status'].iloc[0]).replace('resolution:', 'previous_lab_id:'))
            elif len(prev):
                parts.append(f"previous_lab_id: {int(prev['n_pairs_analysed'].max())} pairs")
            if len(prev) > len(prev_all) and not prev['meets_min_pairs'].astype(bool).any():
                parts[-1] += f' (< {MIN_PAIRS} detected pairs per element; no stats)'
            if len(fid):
                th = fid[fid['element'] == 'Th']
                th_txt = (f"; Th median HARD {th['median_hard'].iloc[0]:.3f} "
                          f"(n={int(th['n_both_detected'].iloc[0])})" if len(th) else '')
                parts.append(f"field_id resubmissions: {len(fid)} elements with ≥{MIN_PAIRS} "
                             f"detected pairs, median of median HARD "
                             f"{fid['median_hard'].median():.3f}{th_txt}")
            else:
                parts.append(f'field_id resubmissions: no element with ≥{MIN_PAIRS} detected pairs')
            res[f'nure_pairs:{scope}'] = ('re-analysis pairs (HARD, bias, Spearman)', '; '.join(parts))
    rad = _read_csv_opt(os.path.join(tdir, 'verification_radiometric.csv'))
    if rad is not None and len(rad):
        txt = '; '.join(f"{r.variant}: ρ={r.spearman_rho:.3f} (n={int(r.n_pairs)}, p={r.p_value:.2g})"
                        for r in rad.itertuples() if pd.notna(r.spearman_rho))
        res['radiometric'] = ('Spearman ρ airborne eTh vs stream Th', txt)
    wgs = _read_csv_opt(os.path.join(tdir, 'verification_wgs_nure.csv'))
    if wgs is not None and len(wgs):
        summ = wgs[wgs['wgs_site'] == '__summary__']
        res['wgs'] = (f'nearest validated NURE ≤ {WGS_RADIUS_KM:g} km, Th/REE/Au comparison',
                      summ['note'].iloc[0] if len(summ) else f'{len(wgs)} rows')
    return res


def _belt_entries(belt, cfg):
    """Dataset specs for one belt: dicts with dataset, paths, source_url, checks, vkey, status."""
    data = cfg.get('data') or {}
    short = (cfg.get('study_area') or {}).get('short', belt)
    belt_root = os.path.dirname(_repo_path(cfg.get('outputs_dir', belt)))
    from pipeline.geo_crs import sgmc_state
    st = sgmc_state(cfg)
    has_provenance = False
    nure_path = _repo_path(data.get('nure_csv'))
    if nure_path and os.path.exists(nure_path):
        has_provenance = 'coord_prec_est' in pd.read_csv(nure_path, nrows=0).columns
    nure_checks = ('half-MDL for all negatives with {el}_mdl/{el}_censored; coord_prec_est ≤ 0.01°; '
                   'sample_source = stream/river; study bbox; duplicate-coordinate report; '
                   'feature censoring gate; submitter batch check (pipeline.data_validation)'
                   if has_provenance else
                   'half-MDL substitution only; HSSR clip has no coord_prec_est / sample_source / '
                   'submitter columns, so precision, medium and batch checks are not assessable')
    E = []
    E.append(dict(dataset='NURE/NGDB stream sediment extract', paths=[nure_path],
                  url=('https://mrdata.usgs.gov/ngdb/sediment/' if has_provenance
                       else 'https://mrdata.usgs.gov/nure/sediment/nuresed-csv.zip'),
                  checks=nure_checks, vkey=f'nure_pairs:{short}', ok='accepted-with-caveat',
                  notes=('Legacy utils.load_nure drops |v|>10 censored values; use validated path.'
                         if has_provenance else 'Transfer-test extract (Task 12).')))
    if belt == 'ne_washington':
        E.append(dict(dataset='NURE/NGDB stream sediment extract (WA statewide)',
                      paths=[_repo_path(NURE_WA_STATEWIDE)],
                      url='https://mrdata.usgs.gov/ngdb/sediment/', checks=nure_checks.replace(
                          'study bbox; ', ''),
                      vkey='nure_pairs:wa_statewide', ok='accepted-with-caveat',
                      notes='Search pool for re-analysis pairs and WGS nearest-sample check.'))
        syn = REPO_ROOT / 'ne_wa_ree' / 'data' / 'nure' / 'nure_wa_synthetic.csv'
        if syn.exists():
            E.append(dict(dataset='Synthetic NURE (generate_synthetic_nure.py)', paths=[str(syn)],
                          url='', checks='none', vkey=None,
                          fixed_status='rejected: synthetic placeholder',
                          notes='Generated file; not referenced by any config. Do not use.'))
    E.append(dict(dataset='MRDS mine sites', paths=[_repo_path(data.get('mrds_geojson'))],
                  url='https://mrdata.usgs.gov/mrds/',
                  checks='commodity filter (mrds_commodities); dedup by dep_id then 4-dp '
                         'coordinates (data_validation.dedup_mrds); bbox clip',
                  vkey=None, ok='accepted-with-caveat',
                  notes=('Config site coordinates checked against MRDS in '
                         'tables/site_verification.csv (site_provenance); MRDS itself not verified.'
                         if os.path.exists(os.path.join(_repo_path(cfg['outputs_dir']), 'tables',
                                                        'site_verification.csv')) else '')))
    for key, label in (('radiometric_k_tif', 'K'), ('radiometric_eth_tif', 'eTh'),
                       ('radiometric_eu_tif', 'eU')):
        if data.get(key):
            E.append(dict(dataset=f'Airborne radiometric {label} grid (NAMrad, 2 km)',
                          paths=[_repo_path(data[key])],
                          url=f"https://mrdata.usgs.gov/radiometric/data/NAMrad_{label.lstrip('e')}.zip",
                          checks='reprojected to EPSG:4326 and clipped (fetch_radiometric); nodata → NaN',
                          vkey='radiometric' if label == 'eTh' else None, ok='accepted-with-caveat',
                          notes=('Published USGS compilation (Duval et al. 2005). Not scored in '
                                 'integration or Task 9.' if label == 'eTh' else
                                 'Verified only indirectly via the eTh grid from the same compilation.')))
    E.append(dict(dataset='DEM (Copernicus GLO-30)', paths=[_repo_path(data.get('dem_tif'))],
                  url='https://copernicus-dem-30m.s3.amazonaws.com',
                  checks='tile mosaic + clip (fetch_dem)', vkey=None, ok=None, notes=''))
    sg_zip = (REPO_ROOT / 'ne_wa_ree' / 'data' / 'geologic' / 'WA.zip' if st == 'WA'
              else REPO_ROOT / 'data' / 'geologic' / f'{st}.zip')
    E.append(dict(dataset=f'SGMC state geology ({st})', paths=[str(sg_zip)],
                  url=f'https://mrdata.usgs.gov/geology/state/shp/{st}.zip',
                  checks='shapefile read; clipped to study area (task11_basemaps)',
                  vkey=None, ok=None, notes='Horton 2017 compilation.'))
    if data.get('wgs_excel'):
        E.append(dict(dataset='WGS OFR 2026-02 mine-waste data supplement',
                      paths=[_repo_path(wgs_path(cfg))],
                      url='https://www.dnr.wa.gov/publications/ger_ofr2026-02_mine_waste_characterization_part_1.zip',
                      checks="units row read; UCC reference row dropped; '<x' censored; "
                             'wgs_bbox + exclude_sites filter (task8)',
                      vkey='wgs', ok='accepted-with-caveat',
                      notes='Published agency report (van Alderwerelt & Di Fiori 2026). '
                            'Mine waste, not stream sediment.'))
    padus = _repo_path(data.get('padus_geojson')) or str(REPO_ROOT / 'data' / 'padus' / f'padus_{short}.geojson')
    E.append(dict(dataset='PAD-US protected areas', paths=[padus],
                  url='https://edits.nationalmap.gov/arcgis/rest/services/PAD-US/PAD_US_gaz_combined/MapServer/0/query',
                  checks='bbox query; access-type mapping (task11_land_access)', vkey=None, ok=None,
                  notes='Land status changes; re-query before field work.'))
    mlrs = _repo_path(data.get('mlrs_geojson')) or str(REPO_ROOT / 'data' / 'mlrs' / f'mlrs_claims_{short}.geojson')
    E.append(dict(dataset='BLM MLRS mining claims (not closed)', paths=[mlrs],
                  url='https://gis.blm.gov/nlsdb/rest/services/HUB/BLM_Natl_MLRS_Mining_Claims_Not_Closed/FeatureServer/0/query',
                  checks='bbox query (task11_land_access)', vkey=None, ok=None,
                  notes='Claims change monthly; re-query before field work.'))
    aiannh = _repo_path(data.get('aiannh_geojson')) or str(REPO_ROOT / 'data' / 'aiannh' / f'aiannh_{short}.geojson')
    if os.path.exists(aiannh):
        E.append(dict(dataset='Census TIGER AIANNH (tribal lands)', paths=[aiannh],
                      url='https://www2.census.gov/geo/tiger/', checks='clip to study box (fetch_land_access)',
                      vkey=None, ok=None, notes=''))
    if data.get('hobby_reports_csv'):
        E.append(dict(dataset='Hobby / pamphlet gold reports', paths=[_repo_path(data['hobby_reports_csv'])],
                      url='per-row source_url column (WA DNR / state pamphlets; USGS GNIS)',
                      checks='schema + location_precision / geom_confidence fields (task11_hobby_reports)',
                      vkey=None, ok=None, notes='Cited gazetteer and opt-in form rows. Not labels.'))
    lidar_dir = os.path.join(belt_root, 'data', 'lidar')
    groups = (('Lidar per-site DEMs (Task 4)', sorted(glob.glob(os.path.join(lidar_dir, '*_lidar.tif')))),
              ('Lidar reach DEM / hillshade clips (Task 11)',
               sorted(glob.glob(os.path.join(lidar_dir, 'lidar_r*_*.tif')))),
              ('USGS 3DEP 1 m lidar source tiles', sorted(glob.glob(os.path.join(lidar_dir, 'tiles', '*.tif')))))
    for label, files in groups:
        if files:
            E.append(dict(dataset=label, paths=files,
                          url=('https://lidar.wa.gov' if 'per-site' in label
                               else 'https://tnmaccess.nationalmap.gov/api/v1/products'),
                          checks='none recorded', vkey=None, ok=None, notes=''))
    mag = data.get('aeromagnetic_tif')
    if mag:
        E.append(dict(dataset='Aeromagnetic anomaly grid', paths=[_repo_path(mag)],
                      url='https://mrdata.usgs.gov/magnetic/', checks='used as-is (RTP derivative)',
                      vkey=None, ok=None, notes='Scored in integration.'))
    elif cfg.get('aeromagnetic_anomaly_centers'):
        E.append(dict(dataset='Aeromagnetic (synthetic anomaly centers)', paths=[],
                      url='', checks='none — Gaussian centers from config (Blakely et al. 1999 sketch)',
                      vkey=None, fixed_status='rejected: synthetic placeholder',
                      notes='data.aeromagnetic_tif is null. Integration excludes the magnetic '
                            'criterion unless integration.score_synthetic_magnetics is true.'))
        loose = REPO_ROOT / 'ne_wa_ree' / 'data' / 'aeromagnetic' / 'wa_mag_anomaly.tif'
        if belt == 'ne_washington' and loose.exists():
            E.append(dict(dataset='Aeromagnetic grid on disk, not configured', paths=[str(loose)],
                          url='https://mrdata.usgs.gov/magnetic/ (provenance not recorded)',
                          checks='none', vkey=None, ok=None,
                          notes='0.01° grid, lon -124.9→-116.9, lat 45.5→49.1. Source and '
                                'processing undocumented; not referenced by config; not used.'))
    else:
        E.append(dict(dataset='Aeromagnetic', paths=[], url='https://mrdata.usgs.gov/magnetic/',
                      checks='none', vkey=None, ok=None,
                      notes='Not configured for this belt; no magnetic layer.'))
    return E


def build_register(config_paths=None, max_bytes=HASH_MAX_BYTES, verification=None):
    """Build the dataset register for every belt config. Returns a DataFrame.

    verification: optional {belt: {vkey: (method, result)}}; defaults to
    reading each belt's verification_*.csv outputs.
    """
    if config_paths is None:
        config_paths = sorted(glob.glob(str(REPO_ROOT / 'configs' / '*' / 'config.yaml')))
    rows = []
    for cp in config_paths:
        with open(cp) as f:
            cfg = yaml.safe_load(f)
        belt = Path(cp).parent.name
        ver = (verification or {}).get(belt)
        if ver is None:
            ver = verification_results(cfg.get('outputs_dir', belt))
        for e in _belt_entries(belt, cfg):
            facts = file_facts(e['paths'], max_bytes)
            method, result = ver.get(e['vkey'], ('not verified', 'not verified')) if e['vkey'] \
                else ('not verified', 'not verified')
            if e.get('fixed_status'):
                status = e['fixed_status']
            elif facts['n_files'] == 0:
                status = 'not-verified'
                if e['paths'] and method == 'not verified':
                    result = 'not verified (file missing locally)'
            elif method != 'not verified' and e.get('ok'):
                status = e['ok']
            else:
                status = 'not-verified'
            notes = '; '.join(x for x in (e.get('notes', ''), facts['hash_note']) if x)
            paths = [p for p in e['paths'] if p]
            local = (_rel(paths[0]) if len(paths) == 1
                     else (_rel(os.path.dirname(paths[0])) + '/' if paths else ''))
            rows.append({
                'dataset': e['dataset'], 'belt': belt, 'local_path': local,
                'source_url': e['url'], 'n_files': facts['n_files'],
                'file_size_bytes': facts['file_size_bytes'], 'sha256': facts['sha256'],
                'file_mtime_utc': facts['file_mtime_utc'],
                'validation_checks': e['checks'], 'verification_method': method,
                'verification_result': result, 'status': status, 'notes': notes,
            })
    return pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def run(cfg, register=False, hssr_zip=HSSR_ZIP):
    """Run checks a–c for one belt and optionally rebuild the register."""
    ensure_outputs(_repo_path(cfg['outputs_dir']))
    cfg = copy.deepcopy(cfg)
    cfg['outputs_dir'] = _repo_path(cfg['outputs_dir'])
    short = cfg['study_area'].get('short', 'belt')

    raw = read_nure_raw(cfg['data']['nure_csv'])
    scopes = {short: raw}
    if short == 'ne_wa' and os.path.exists(_repo_path(NURE_WA_STATEWIDE)):
        scopes['wa_statewide'] = read_nure_raw(NURE_WA_STATEWIDE)

    print('a. Re-analysis pairs')
    pairs, detail, resolution = reanalysis_verification(scopes, hssr_zip=hssr_zip)
    pairs.to_csv(out(cfg, 'tables', 'verification_reanalysis_pairs.csv'), index=False)
    detail.to_csv(out(cfg, 'tables', 'verification_reanalysis_detail.csv'), index=False)
    for scope, res in resolution.items():
        print(f"  [{scope}] previous_lab_id: {res['n_candidates']} rows, {res['n_resolved']} resolved"
              f"; HSSR matches: {res['hssr_matches']}")
    ok = pairs[(pairs['pair_source'] == 'field_id_any') & pairs['meets_min_pairs'].astype(bool)
               & (pairs['element'] != 'ALL_ELEMENTS_POOLED')]
    for scope, g in ok.groupby('scope'):
        print(f"  [{scope}] field_id pairs: {len(g)} elements with ≥{MIN_PAIRS} detected pairs")

    print('b. Airborne eTh vs stream Th')
    clean, _, method = validate_samples(raw, cfg)
    rad = radiometric_verification(cfg, raw, clean, method)
    rad.to_csv(out(cfg, 'tables', 'verification_radiometric.csv'), index=False)
    for r in rad.itertuples():
        print(f'  {r.variant}: rho={r.spearman_rho}  n={r.n_pairs}  p={r.p_value:.3g}')

    print('c. NURE vs WGS OFR 2026-02')
    xl = _repo_path(wgs_path(cfg))
    wgs_out = out(cfg, 'tables', 'verification_wgs_nure.csv')
    if xl and os.path.exists(xl):
        pool = clean.assign(_extract=short)
        any_pool = raw
        if 'wa_statewide' in scopes:
            cfg_sw = copy.deepcopy(cfg)
            cfg_sw['study_area'] = {k: v for k, v in cfg_sw['study_area'].items() if k != 'bbox'}
            clean_sw, _, _ = validate_samples(scopes['wa_statewide'], cfg_sw)
            pool = pd.concat([pool, clean_sw.assign(_extract='wa_statewide')], ignore_index=True)
            pool = pool.drop_duplicates('lab_id')
            any_pool = pd.concat([raw, scopes['wa_statewide']], ignore_index=True).drop_duplicates('lab_id')
        wgs = wgs_nure_comparison(load_wgs_sites(xl), pool,
                                  wgs_bbox=cfg['study_area'].get('wgs_bbox'),
                                  exclude_sites=(cfg.get('wgs') or {}).get('exclude_sites', []),
                                  nure_any=any_pool.dropna(subset=['lat', 'lon']).reset_index(drop=True))
    else:
        wgs = pd.DataFrame([{'wgs_site': '__summary__', 'note': f'WGS Excel not found at {xl}'}])
    wgs.to_csv(wgs_out, index=False)
    print(f"  {wgs.loc[wgs['wgs_site'] == '__summary__', 'note'].iloc[0]}")

    if register:
        reg = build_register()
        reg.to_csv(REGISTER_PATH, index=False)
        print(f'd. Register: {len(reg)} rows → {_rel(REGISTER_PATH)}')
        print(reg['status'].value_counts().to_string())
    return pairs, detail, rad, wgs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default='configs/ne_washington/config.yaml')
    ap.add_argument('--register', action='store_true', help='rebuild data/VERIFICATION_REGISTER.csv')
    ap.add_argument('--no-hssr', action='store_true', help='skip scanning the national HSSR zip')
    args = ap.parse_args(argv)
    with open(_repo_path(args.config)) as f:
        cfg = yaml.safe_load(f)
    run(cfg, register=args.register, hssr_zip=None if args.no_hssr else HSSR_ZIP)


if __name__ == '__main__':
    main()
