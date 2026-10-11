"""QA/QC gate for field + lab assay results (CIM 2018 §2.7.4 style).

Input: data/field_results/<campaign>.csv (see field_results_template.csv)
and a CRM table (crm_values_template.csv). Protocol: docs/FIELD_PROTOCOL.md.

Checks per lab batch (batch_id):
  blanks      — fail if a blank element > mdl_multiple × MDL or > a set limit
  CRMs        — z = (reported − certified) / sd: |z|≤2 pass, 2–3 warn, >3 fail
  duplicates  — HARD = 100·|a−b|/(a+b) per pair; field dup > 30 %, lab/pulp
                dup and check assay > 20 % flag the pair. The batch fails when
                more than dup_fail_fraction of assessed pairs flag, warns if any
  insertion   — blanks / CRMs / field dups per primary vs target 1/20; warn
                below target × tolerance, fail if a type is missing entirely
  metadata    — primaries without lon/lat, lab, method or certificate_id warn

Duplicate and check-assay pairs count against the parent primary's batch.
Below-detection strings ('<0.005', or a negative number as some labs report)
are censored at the stated limit; the smallest '<x' per batch/element is
that batch's MDL, else the lab's method MDL from the mdl mapping (--mdl).
A detected blank with no MDL and no limit warns. '>x' is over-range.

Gate: nothing else in the pipeline may consume field results except through
accepted_primaries(), which returns only primary samples from batches whose
status is pass (warn only with allow_warn=True). QC rows never leave here.

CLI:
  python -m pipeline.field_qaqc --results <csv> --crm <csv> --out <dir>
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

ELEMENTS = (
    'Au_ppm', 'Th_ppm', 'Ce_ppm', 'La_ppm', 'Nd_ppm', 'Pr_ppm', 'Y_ppm',
    'Zr_ppm', 'Ti_ppm', 'Fe_pct',
)
META_COLS = (
    'sample_id', 'parent_id', 'qc_type', 'crm_id', 'site_id', 'block_id',
    'lon', 'lat', 'gps_accuracy_m', 'sample_medium', 'collected_by',
    'date_collected', 'lab', 'certificate_id', 'method', 'batch_id',
)
RESULT_COLS = META_COLS + ELEMENTS
QC_TYPES = (
    'primary', 'field_duplicate', 'blank', 'crm', 'lab_duplicate', 'check_assay',
)
DUP_TYPES = ('field_duplicate', 'lab_duplicate', 'check_assay')
# Silica-sand blanks carry trace Zr/Ti/Fe; only target elements are judged.
BLANK_ELEMENTS = ('Au_ppm', 'Th_ppm', 'Ce_ppm', 'La_ppm', 'Nd_ppm', 'Pr_ppm', 'Y_ppm')
CRM_COLS = ('crm_id', 'element', 'certified_value', 'sd')
CHECK_COLS = (
    'batch_id', 'check', 'sample_id', 'parent_id', 'qc_type', 'crm_id',
    'element', 'value', 'reference', 'metric', 'threshold', 'status', 'reason',
)
STATUS_ORDER = {'pass': 0, 'warn': 1, 'fail': 2}
_MISSING = {'', 'na', 'n/a', 'n.a.', 'nan', 'none', '-', '--', 'is', 'nss', 'ns', 'lnr'}


def parse_value(raw):
    """Assay string → (value, flag, limit). flag is '', '<' or '>'."""
    if raw is None:
        return np.nan, '', np.nan
    if isinstance(raw, (int, float, np.integer, np.floating)):
        if not np.isfinite(raw):
            return np.nan, '', np.nan
        if raw < 0:
            return abs(float(raw)), '<', abs(float(raw))
        return float(raw), '', np.nan
    s = str(raw).strip().replace(',', '')
    if s.lower() in _MISSING:
        return np.nan, '', np.nan
    flag = ''
    if s[0] in '<>':
        flag, s = s[0], s[1:].strip()
    try:
        val = float(s)
    except ValueError:
        return np.nan, '', np.nan
    if not flag and val < 0:
        flag, val = '<', abs(val)
    if flag:
        return val, flag, val
    return val, '', np.nan


def _normalize(df):
    out = df.copy()
    for col in META_COLS:
        if col not in out.columns:
            out[col] = np.nan if col in ('lon', 'lat', 'gps_accuracy_m') else ''
    for col in ('sample_id', 'parent_id', 'qc_type', 'crm_id', 'batch_id',
                'lab', 'method', 'certificate_id'):
        out[col] = out[col].fillna('').astype(str).str.strip()
    out['qc_type'] = out['qc_type'].str.lower()
    bad = sorted(set(out['qc_type']) - set(QC_TYPES))
    if bad:
        raise ValueError(f'unknown qc_type values: {bad}; expected {QC_TYPES}')
    for col in ('lon', 'lat', 'gps_accuracy_m'):
        out[col] = pd.to_numeric(out[col], errors='coerce')
    for el in ELEMENTS:
        raw = out[el] if el in out.columns else pd.Series([None] * len(out), index=out.index)
        parsed = [parse_value(v) for v in raw]
        out[el] = [p[0] for p in parsed]
        out[f'{el}_flag'] = [p[1] for p in parsed]
        out[f'{el}_dl'] = [p[2] for p in parsed]
    return out


def load_results(path_or_df):
    """Read a results CSV (or DataFrame) and parse censored assay strings."""
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df
    else:
        df = pd.read_csv(path_or_df, dtype=str, keep_default_na=False)
    return _normalize(df)


def _element_name(raw):
    s = str(raw).strip()
    if s in ELEMENTS:
        return s
    match = [e for e in ELEMENTS if e.split('_')[0].lower() == s.lower()]
    return match[0] if len(match) == 1 else s


def load_crm_values(path_or_df):
    """CRM table: crm_id, element (e.g. Au_ppm), certified_value, sd. '#' lines skipped."""
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    elif path_or_df is None or not os.path.exists(path_or_df):
        return pd.DataFrame(columns=list(CRM_COLS))
    else:
        df = pd.read_csv(path_or_df, dtype=str, comment='#', skip_blank_lines=True)
    for col in CRM_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[list(CRM_COLS)].copy()
    df['crm_id'] = df['crm_id'].fillna('').astype(str).str.strip()
    df['element'] = df['element'].map(_element_name)
    df['certified_value'] = pd.to_numeric(df['certified_value'], errors='coerce')
    df['sd'] = pd.to_numeric(df['sd'], errors='coerce')
    return df[(df['crm_id'] != '') & df['certified_value'].notna()].reset_index(drop=True)


def detection_limits(df, mdl=None):
    """{(batch_id, element): MDL} from '<x' strings, else the mdl mapping."""
    mdl = mdl or {}
    out = {}
    for bid, grp in df.groupby('batch_id'):
        for el in ELEMENTS:
            dl = pd.to_numeric(grp.get(f'{el}_dl'), errors='coerce')
            dl = dl[dl > 0] if dl is not None else pd.Series(dtype=float)
            if len(dl):
                out[(bid, el)] = float(dl.min())
            elif el in mdl and mdl[el]:
                out[(bid, el)] = float(mdl[el])
    return out


def _row(batch_id, check, status, reason, sample=None, element='', value=np.nan,
         reference=np.nan, metric=np.nan, threshold=np.nan, parent_id=''):
    sample = sample if sample is not None else {}
    return {
        'batch_id': batch_id,
        'check': check,
        'sample_id': sample.get('sample_id', ''),
        'parent_id': parent_id or sample.get('parent_id', ''),
        'qc_type': sample.get('qc_type', ''),
        'crm_id': sample.get('crm_id', ''),
        'element': element,
        'value': value,
        'reference': reference,
        'metric': metric,
        'threshold': threshold,
        'status': status,
        'reason': reason,
    }


def _frame(rows):
    return pd.DataFrame(rows, columns=list(CHECK_COLS))


def check_blanks(df, mdl=None, limits=None, elements=BLANK_ELEMENTS, mdl_multiple=5.0):
    """Blank contamination: fail if value > mdl_multiple × MDL or > limits[element]."""
    limits = limits or {}
    dls = detection_limits(df, mdl)
    rows = []
    for _, s in df[df['qc_type'] == 'blank'].iterrows():
        for el in elements:
            val = s[el]
            if not np.isfinite(val):
                continue
            bid = s['batch_id']
            if s[f'{el}_flag'] == '<':
                rows.append(_row(bid, 'blank', 'pass', 'below detection', s, el, val))
                continue
            caps = []
            dl = dls.get((bid, el))
            if dl:
                caps.append(mdl_multiple * dl)
            if limits.get(el) is not None:
                caps.append(float(limits[el]))
            if not caps:
                rows.append(_row(
                    bid, 'blank', 'warn', 'no MDL or blank limit to judge against',
                    s, el, val,
                ))
                continue
            cap = min(caps)
            metric = val / dl if dl else np.nan
            if val > cap:
                rows.append(_row(
                    bid, 'blank', 'fail', f'blank {el}={val:g} > limit {cap:g}',
                    s, el, val, metric=metric, threshold=cap,
                ))
            else:
                rows.append(_row(bid, 'blank', 'pass', 'within limit', s, el, val,
                                 metric=metric, threshold=cap))
    return _frame(rows)


def check_crms(df, crm_values, warn_z=2.0, fail_z=3.0):
    """CRM accuracy: z vs certified value and sd for every certified element."""
    crm = load_crm_values(crm_values)
    rows = []
    for _, s in df[df['qc_type'] == 'crm'].iterrows():
        bid = s['batch_id']
        certs = crm[crm['crm_id'] == s['crm_id']] if s['crm_id'] else crm.iloc[0:0]
        if certs.empty:
            why = 'crm_id blank' if not s['crm_id'] else f'crm_id {s["crm_id"]} not in CRM table'
            rows.append(_row(bid, 'crm', 'warn', why, s))
            continue
        for _, c in certs.iterrows():
            el = c['element']
            if el not in ELEMENTS:
                continue
            val, cert, sd = s[el], float(c['certified_value']), c['sd']
            if not np.isfinite(val):
                rows.append(_row(bid, 'crm', 'warn', f'{el} not reported', s, el,
                                 reference=cert))
                continue
            if not (np.isfinite(sd) and sd > 0):
                rows.append(_row(bid, 'crm', 'warn', f'{el} has no certified sd', s,
                                 el, val, cert))
                continue
            z = (val - cert) / sd
            az = abs(z)
            if az > fail_z:
                status, why = 'fail', f'|z|={az:.2f} > {fail_z:g}'
            elif az > warn_z:
                status, why = 'warn', f'|z|={az:.2f} > {warn_z:g}'
            else:
                status, why = 'pass', f'|z|={az:.2f}'
            if s[f'{el}_flag']:
                why += f' (reported {s[f"{el}_flag"]}{val:g})'
            rows.append(_row(bid, 'crm', status, why, s, el, val, cert, z, fail_z))
    return _frame(rows)


def _threshold(spec, element):
    if isinstance(spec, dict):
        return float(spec.get(element, spec.get('default', np.nan)))
    return float(spec)


def hard(a, b):
    """Half absolute relative difference, percent."""
    tot = a + b
    if not np.isfinite(tot) or tot <= 0:
        return np.nan
    return 100.0 * abs(a - b) / tot


def check_duplicates(df, field_max_hard=30.0, pulp_max_hard=20.0,
                     check_max_hard=20.0, min_mean_mdl_multiple=10.0,
                     mdl=None, elements=ELEMENTS):
    """HARD per duplicate/check-assay pair. Thresholds may be {element: pct}.

    Pairs with a censored value, or mean < min_mean_mdl_multiple × MDL, are
    'skip' (precision is not meaningful near detection).
    """
    limits = {
        'field_duplicate': field_max_hard,
        'lab_duplicate': pulp_max_hard,
        'check_assay': check_max_hard,
    }
    dls = detection_limits(df, mdl)
    by_id = df.drop_duplicates('sample_id').set_index('sample_id')
    rows = []
    for _, d in df[df['qc_type'].isin(DUP_TYPES)].iterrows():
        pid = d['parent_id']
        if not pid or pid not in by_id.index:
            rows.append(_row(d['batch_id'], d['qc_type'], 'warn',
                             f'parent_id {pid or "(blank)"} not found', d))
            continue
        p = by_id.loc[pid]
        bid = p['batch_id']
        for el in elements:
            a, b = p[el], d[el]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            lim = _threshold(limits[d['qc_type']], el)
            if p[f'{el}_flag'] or d[f'{el}_flag']:
                rows.append(_row(bid, d['qc_type'], 'skip', 'censored value', d, el,
                                 b, a, threshold=lim, parent_id=pid))
                continue
            dl = dls.get((bid, el))
            mean = (a + b) / 2.0
            if dl and mean < min_mean_mdl_multiple * dl:
                rows.append(_row(bid, d['qc_type'], 'skip',
                                 f'mean < {min_mean_mdl_multiple:g}×MDL', d, el,
                                 b, a, threshold=lim, parent_id=pid))
                continue
            h = hard(a, b)
            if not np.isfinite(h):
                continue
            status = 'fail' if h > lim else 'pass'
            rows.append(_row(bid, d['qc_type'], status, f'HARD={h:.1f}%', d, el,
                             b, a, h, lim, parent_id=pid))
    return _frame(rows)


def check_insertion_rates(df, target=1 / 20, tolerance=0.8,
                          qc_types=('field_duplicate', 'blank', 'crm'),
                          check_assay_target=0.05):
    """QC count per primary in each batch vs target; project-level check-assay rate."""
    rows = []
    for bid, grp in df.groupby('batch_id'):
        n_prim = int((grp['qc_type'] == 'primary').sum())
        if n_prim == 0:
            continue
        for qt in qc_types:
            n = int((grp['qc_type'] == qt).sum())
            rate = n / n_prim
            if n == 0:
                status, why = 'fail', f'no {qt} inserted ({n_prim} primaries)'
            elif rate < target * tolerance:
                status, why = 'warn', f'{qt} rate {n}/{n_prim}={rate:.3f} < target {target:.3f}'
            else:
                status, why = 'pass', f'{qt} rate {n}/{n_prim}={rate:.3f}'
            rows.append(_row(bid, 'insertion_rate', status, why, {'qc_type': qt},
                             value=n, reference=n_prim, metric=rate, threshold=target))
    n_prim = int((df['qc_type'] == 'primary').sum())
    if n_prim:
        n_chk = int((df['qc_type'] == 'check_assay').sum())
        rate = n_chk / n_prim
        status = 'pass' if rate >= check_assay_target else 'warn'
        rows.append(_row(
            'ALL', 'check_assay_rate', status,
            f'check assays {n_chk}/{n_prim}={rate:.3f} (target ≥ {check_assay_target:.2f})',
            {'qc_type': 'check_assay'}, value=n_chk, reference=n_prim, metric=rate,
            threshold=check_assay_target,
        ))
    return _frame(rows)


def check_metadata(df):
    """Primaries missing location / lab / method / certificate warn their batch."""
    rows = []
    prim = df[df['qc_type'] == 'primary']
    for bid, grp in prim.groupby('batch_id'):
        missing = []
        if grp['lon'].isna().any() or grp['lat'].isna().any():
            missing.append('lon/lat')
        for col in ('lab', 'method', 'certificate_id'):
            if (grp[col] == '').any():
                missing.append(col)
        if missing:
            rows.append(_row(bid, 'metadata', 'warn', 'missing ' + ', '.join(missing)))
        else:
            rows.append(_row(bid, 'metadata', 'pass', 'complete'))
    return _frame(rows)


def run_checks(df, crm_values=None, mdl=None, blank_limits=None,
               field_max_hard=30.0, pulp_max_hard=20.0, check_max_hard=20.0,
               target_rate=1 / 20):
    """All checks on a parsed results frame → one long checks table."""
    parts = [
        check_blanks(df, mdl=mdl, limits=blank_limits),
        check_crms(df, crm_values),
        check_duplicates(df, field_max_hard=field_max_hard,
                         pulp_max_hard=pulp_max_hard,
                         check_max_hard=check_max_hard, mdl=mdl),
        check_insertion_rates(df, target=target_rate),
        check_metadata(df),
    ]
    parts = [p for p in parts if len(p)]
    if not parts:
        return _frame([])
    return pd.concat(parts, ignore_index=True)


def batch_status(checks, df=None, dup_fail_fraction=0.10):
    """pass / warn / fail per batch with reasons. batch_id 'ALL' is project-level."""
    batches = set(checks['batch_id']) if len(checks) else set()
    if df is not None:
        batches |= set(df['batch_id'])
    batches.discard('ALL')
    out = []
    for bid in sorted(batches):
        grp = checks[checks['batch_id'] == bid] if len(checks) else checks
        flags = []

        plain = grp[~grp['check'].isin(DUP_TYPES)]
        for _, r in plain[plain['status'].isin(['warn', 'fail'])].iterrows():
            tag = f"{r['check']}"
            if r['sample_id']:
                tag += f" {r['sample_id']}"
            if r['element']:
                tag += f" {r['element']}"
            flags.append((r['status'], f"{tag}: {r['reason']}"))

        dups = grp[grp['check'].isin(DUP_TYPES)]
        for _, r in dups[dups['status'] == 'warn'].iterrows():
            flags.append(('warn', f"{r['check']} {r['sample_id']}: {r['reason']}"))
        assessed = dups[dups['status'].isin(['pass', 'fail'])]
        for (chk, el), g in assessed.groupby(['check', 'element']):
            n_fail = int((g['status'] == 'fail').sum())
            if n_fail == 0:
                continue
            frac = n_fail / len(g)
            why = f'{chk} {el}: {n_fail}/{len(g)} pairs over HARD limit'
            flags.append(('fail' if frac > dup_fail_fraction else 'warn', why))

        status = max((s for s, _ in flags), key=STATUS_ORDER.get, default='pass')
        reasons = [why for _, why in flags]

        n_primary = 0
        if df is not None:
            n_primary = int(((df['batch_id'] == bid) & (df['qc_type'] == 'primary')).sum())
        out.append({
            'batch_id': bid,
            'status': status,
            'n_primary': n_primary,
            'n_fail': int((grp['status'] == 'fail').sum()),
            'n_warn': int((grp['status'] == 'warn').sum()),
            'reasons': '; '.join(reasons),
        })
    return pd.DataFrame(out, columns=['batch_id', 'status', 'n_primary', 'n_fail',
                                      'n_warn', 'reasons'])


def run_qaqc(results, crm_values=None, **kwargs):
    """Parse → checks → batch table. Returns (parsed_df, checks, batches)."""
    dup_frac = kwargs.pop('dup_fail_fraction', 0.10)
    df = load_results(results)
    checks = run_checks(df, crm_values, **kwargs)
    return df, checks, batch_status(checks, df, dup_fail_fraction=dup_frac)


def accepted_primaries(df, crm_values=None, batches=None, allow_warn=False, **kwargs):
    """Primary samples from passing batches only — the single exit for field data."""
    parsed = load_results(df) if 'Au_ppm_flag' not in getattr(df, 'columns', []) else df
    if batches is None:
        dup_frac = kwargs.pop('dup_fail_fraction', 0.10)
        checks = run_checks(parsed, crm_values, **kwargs)
        batches = batch_status(checks, parsed, dup_fail_fraction=dup_frac)
    ok = {'pass', 'warn'} if allow_warn else {'pass'}
    good = set(batches.loc[batches['status'].isin(ok), 'batch_id'])
    keep = parsed[(parsed['qc_type'] == 'primary') & parsed['batch_id'].isin(good)]
    return keep.reset_index(drop=True)


def parse_mdl_arg(raw):
    """'Au_ppm=0.005,Ce=0.1' → {'Au_ppm': 0.005, 'Ce_ppm': 0.1}."""
    out = {}
    for part in str(raw or '').split(','):
        if '=' not in part:
            continue
        key, val = part.split('=', 1)
        out[_element_name(key)] = float(val)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description='Field assay QA/QC batch gate')
    parser.add_argument('--results', required=True)
    parser.add_argument('--crm', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--mdl', default='',
                        help="Lab method detection limits, e.g. 'Au_ppm=0.005,Ce_ppm=0.1'")
    parser.add_argument('--allow-warn', action='store_true',
                        help='Also accept primaries from warn batches')
    args = parser.parse_args(argv)
    df, checks, batches = run_qaqc(args.results, args.crm, mdl=parse_mdl_arg(args.mdl))
    os.makedirs(args.out, exist_ok=True)
    checks.to_csv(os.path.join(args.out, 'field_qaqc_checks.csv'), index=False)
    batches.to_csv(os.path.join(args.out, 'field_qaqc_batches.csv'), index=False)
    acc = accepted_primaries(df, batches=batches, allow_warn=args.allow_warn)
    keep = [c for c in RESULT_COLS if c in acc.columns]
    keep += [f'{el}_flag' for el in ELEMENTS if f'{el}_flag' in acc.columns]
    acc[keep].to_csv(os.path.join(args.out, 'field_accepted_primaries.csv'), index=False)
    counts = batches['status'].value_counts().to_dict()
    print(
        f'{len(batches)} batches: pass={counts.get("pass", 0)} '
        f'warn={counts.get("warn", 0)} fail={counts.get("fail", 0)}; '
        f'{len(acc)} primaries accepted → {args.out}',
        flush=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
