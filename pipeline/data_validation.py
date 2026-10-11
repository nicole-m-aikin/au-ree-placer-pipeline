"""Data validation for legacy NURE/NGDB stream sediment and MRDS (CIM 2018).

The frozen doorbell path (utils.load_nure) is not changed. This module is the
validated path: location precision, sample medium, below-detection results
kept as censored half-MDL values with their MDL, a feature gate for Task 9,
and a lab-batch check. Every step writes ledger rows.

CLI:
  python -m pipeline.data_validation --config configs/ne_washington/config.yaml

Outputs ({outputs_dir}/tables/):
  data_validation_ledger.csv
  data_validation_feature_gate.csv
  data_validation_batches.csv
  data_validation_mrds_dedup.csv
"""

import argparse
import re

import numpy as np
import pandas as pd

from pipeline.ml_preprocess import FEATURES, _log_impute, usable_feature_columns
from pipeline.utils import anomaly_threshold, ensure_outputs, out

LEDGER_COLUMNS = ['check', 'field', 'n_in', 'n_flagged', 'n_out', 'rule', 'note']

# Numeric columns in NGDB extracts that are not analytes.
_NON_ANALYTE = {
    'lat', 'lon', 'lat_orig', 'long_orig', 'depth', 'lab_id', 'field_id',
    'job_id', 'mineralization', 'collection_protocol', 'previous_job_id',
    'previous_lab_id', 'SplWtAu',
}

# Same (column, max-wt%-median) rule as utils.load_nure.
_WT_PCT_COLS = [('P', 1), ('Ca', 100), ('Fe', 100)]

DEFAULT_MAX_COORD_PREC = '0.01 degree'
DEFAULT_SAMPLE_SOURCES = ['stream/river']
DEFAULT_MAX_CENSORED_FRAC = 0.5
MIN_BATCH_N = 20
NOT_ASSESSABLE = 'not assessable'

_PREC_RE = re.compile(r'^\s*([\d.]+)\s*(second|minute|degree)s?\s*$', re.I)
_ARCSEC_PER_UNIT = {'second': 1.0, 'minute': 60.0, 'degree': 3600.0}


def coord_prec_arcsec(label):
    """'7.5 minutes' → 450.0 arc-seconds. None if the label is missing or unparseable."""
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    m = _PREC_RE.match(str(label))
    if not m:
        return None
    return float(m.group(1)) * _ARCSEC_PER_UNIT[m.group(2).lower()]


def _ledger_row(check, field, n_in, n_flagged, n_out, rule, note=''):
    return dict(zip(LEDGER_COLUMNS, [check, field, n_in, n_flagged, n_out, rule, note]))


def _counts_note(series):
    vc = series.fillna('NaN').astype(str).value_counts()
    return '; '.join(f'{k}: {v}' for k, v in vc.items())


def analyte_columns(df):
    """Numeric element / analyte columns, excluding coordinates and ids."""
    return [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in _NON_ANALYTE
            and not c.endswith('_mdl') and not c.endswith('_censored')]


def censor_half_mdl(df, cols=None):
    """Negative v → |v|/2 with {el}_mdl = |v| and {el}_censored = True, any magnitude.

    NaN stays NaN (not analysed); {el}_censored is <NA> there. P/Ca/Fe wt% → ppm
    uses the load_nure median rule; {el}_mdl is scaled with the value.
    Returns (df, list of converted columns).
    """
    df = df.copy()
    cols = analyte_columns(df) if cols is None else cols
    extra = {}
    for c in cols:
        v = pd.to_numeric(df[c], errors='coerce')
        if not v.notna().any():
            continue
        neg = v < 0
        df[c] = v.where(~neg, v.abs() / 2.0)
        extra[f'{c}_mdl'] = v.abs().where(neg)
        extra[f'{c}_censored'] = neg.astype('boolean').mask(v.isna())
    if extra:
        df = pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)
    converted = []
    for col, thresh in _WT_PCT_COLS:
        if col in df.columns and df[col].notna().any() and df[col].dropna().median() < thresh:
            df[col] = df[col] * 10000
            if f'{col}_mdl' in df.columns:
                df[f'{col}_mdl'] = df[f'{col}_mdl'] * 10000
            converted.append(col)
    return df, converted


def duplicate_coordinate_groups(df, decimals=4):
    """Groups of samples sharing lat/lon at `decimals` dp. Returns a summary dict."""
    if len(df) == 0:
        return {'n_groups': 0, 'n_samples': 0, 'largest': 0, 'size_counts': {}}
    sizes = df.groupby([df['lat'].round(decimals), df['lon'].round(decimals)]).size()
    dup = sizes[sizes > 1]
    return {
        'n_groups': int(len(dup)),
        'n_samples': int(dup.sum()),
        'largest': int(dup.max()) if len(dup) else 0,
        'size_counts': {int(k): int(v) for k, v in dup.value_counts().sort_index().items()},
    }


def _dup_row(df, stage):
    d = duplicate_coordinate_groups(df)
    sizes = ', '.join(f'size {k} x{v}' for k, v in sorted(d['size_counts'].items(), reverse=True))
    return _ledger_row(
        'duplicate_coordinates', f'lat/lon ({stage})', len(df), d['n_samples'], len(df),
        'report groups sharing lat/lon at 4 dp; not removed',
        f"{d['n_groups']} groups, largest {d['largest']}" + (f'; {sizes}' if sizes else ''),
    )


def precision_mask(df, max_coord_prec=DEFAULT_MAX_COORD_PREC):
    """Boolean keep-mask on coord_prec_est, or None if the column is absent.

    max_coord_prec: a label ('0.01 degree'), a number of arc-seconds, or a list
    of labels to keep verbatim. Unknown / NaN precision is excluded.
    """
    if 'coord_prec_est' not in df.columns:
        return None, 'column coord_prec_est absent'
    labels = df['coord_prec_est']
    if isinstance(max_coord_prec, (list, tuple, set)):
        allowed = {str(x) for x in max_coord_prec}
        return labels.astype(str).isin(allowed) & labels.notna(), f'keep {sorted(allowed)}'
    if isinstance(max_coord_prec, (int, float)):
        limit = float(max_coord_prec)
    else:
        limit = coord_prec_arcsec(max_coord_prec)
        if limit is None:
            raise ValueError(f'nure_validation.max_coord_prec not understood: {max_coord_prec!r}')
    arcsec = labels.map(coord_prec_arcsec)
    keep = arcsec.notna() & (arcsec.astype(float) <= limit)
    return keep, f'keep coord_prec_est <= {limit:g} arc-seconds ({max_coord_prec}); unknown excluded'


def feature_gate(df, features=None, max_censored_frac=DEFAULT_MAX_CENSORED_FRAC):
    """Keep/drop each Task 9 feature on censoring. Needs censor_half_mdl output.

    Drop when censored fraction > max_censored_frac, or when the median MDL of
    censored results exceeds the log mean+2SD anomaly threshold of detected
    results (the lab could not see an anomaly), or when < 5 detected values.
    """
    features = FEATURES if features is None else features
    rows = []
    for f in features:
        row = {'feature': f, 'n_samples': len(df), 'n_analysed': 0, 'analysed_frac': 0.0,
               'n_detected': 0, 'n_censored': 0, 'censored_frac': np.nan, 'median_mdl': np.nan,
               'anomaly_threshold': np.nan, 'keep': False, 'reason': ''}
        if f not in df.columns:
            row['reason'] = 'column absent'
            rows.append(row)
            continue
        v = pd.to_numeric(df[f], errors='coerce')
        analysed = v.notna()
        if f'{f}_censored' in df.columns:
            cens = df[f'{f}_censored'].fillna(False).astype(bool) & analysed
        else:
            cens = pd.Series(False, index=df.index)
        detected = v[analysed & ~cens & (v > 0)]
        n_an, n_c = int(analysed.sum()), int(cens.sum())
        thr = anomaly_threshold(detected)
        med_mdl = (float(df.loc[cens, f'{f}_mdl'].median())
                   if n_c and f'{f}_mdl' in df.columns else np.nan)
        frac = n_c / n_an if n_an else np.nan
        row.update(n_analysed=n_an,
                   analysed_frac=round(n_an / len(df), 4) if len(df) else 0.0,
                   n_detected=int(len(detected)), n_censored=n_c,
                   censored_frac=round(frac, 4) if n_an else np.nan,
                   median_mdl=med_mdl,
                   anomaly_threshold=round(thr, 4) if thr is not None else np.nan)
        reasons = []
        if n_an == 0:
            reasons.append('not analysed')
        else:
            if frac > max_censored_frac:
                reasons.append(f'censored fraction {frac:.2f} > {max_censored_frac:g}')
            if thr is None:
                reasons.append('fewer than 5 detected values; no anomaly threshold')
            elif n_c and med_mdl > thr:
                reasons.append(f'median MDL {med_mdl:g} > anomaly threshold {thr:.3g} '
                               '(log mean+2SD of detected)')
        row['keep'] = not reasons
        row['reason'] = '; '.join(reasons) if reasons else 'kept'
        rows.append(row)
    return pd.DataFrame(rows)


def batch_table(df, validated_index=None, features=None):
    """Per submitter / job_id: count, median and censored fraction for each feature."""
    features = [f for f in (FEATURES if features is None else features) if f in df.columns]
    in_val = pd.Series(
        df.index.isin(validated_index) if validated_index is not None else False,
        index=df.index)
    rows = []
    for group_col in ('submitter', 'job_id'):
        if group_col not in df.columns:
            continue
        for g, sub in df.groupby(df[group_col].fillna('NaN').astype(str)):
            n_val = int(in_val.loc[sub.index].sum())
            for f in features:
                v = pd.to_numeric(sub[f], errors='coerce')
                analysed = v.notna()
                cens = (sub[f'{f}_censored'].fillna(False).astype(bool) & analysed
                        if f'{f}_censored' in sub.columns else pd.Series(False, index=sub.index))
                det = v[analysed & ~cens]
                n_an = int(analysed.sum())
                rows.append({
                    'group_by': group_col, 'group': g, 'feature': f,
                    'n_samples': len(sub), 'n_validated': n_val,
                    'n_analysed': n_an, 'n_censored': int(cens.sum()),
                    'censored_frac': round(cens.sum() / n_an, 4) if n_an else np.nan,
                    'median_value': float(v.median()) if n_an else np.nan,
                    'median_detected': float(det.median()) if len(det) else np.nan,
                    'median_mdl': (float(sub.loc[cens, f'{f}_mdl'].median())
                                   if cens.any() and f'{f}_mdl' in sub.columns else np.nan),
                })
    return pd.DataFrame(rows)


def submitter_prediction(df, features, group_col='submitter', min_n=MIN_BATCH_N):
    """Can a forest tell the submitting lab from the log features? Balanced accuracy vs chance."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    if group_col not in df.columns:
        return {'status': f'{NOT_ASSESSABLE}: column {group_col} absent'}
    counts = df[group_col].value_counts()
    keep = counts[counts >= min_n]
    if len(keep) < 2:
        return {'status': f'skipped: {len(keep)} {group_col}(s) with >= {min_n} samples'}
    sub = df[df[group_col].isin(keep.index)]
    feats, _ = usable_feature_columns(sub, features)
    if not feats:
        return {'status': 'skipped: no usable features'}
    X, _ = _log_impute(sub, feats)
    y = sub[group_col].astype(str).values
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X.values, y, cv=skf, scoring='balanced_accuracy')
    return {
        'status': 'ok',
        'n_samples': int(len(sub)),
        'n_classes': int(len(keep)),
        'chance': round(1.0 / len(keep), 4),
        'balanced_accuracy': round(float(scores.mean()), 4),
        'balanced_accuracy_sd': round(float(scores.std(ddof=1)), 4),
        'classes': '; '.join(f'{k} ({v})' for k, v in keep.items()),
        'features': ','.join(feats),
    }


def _prediction_row(stage, res, rule):
    if res.get('status') != 'ok':
        return _ledger_row('submitter_prediction', stage, None, None, None, rule, res['status'])
    return _ledger_row(
        'submitter_prediction', stage, res['n_samples'], None, None, rule,
        f"balanced_accuracy={res['balanced_accuracy']:.3f} ± {res['balanced_accuracy_sd']:.3f}; "
        f"chance={res['chance']:.3f}; classes={res['classes']}; features={res['features']}",
    )


def validate_nure(df_raw, cfg, return_details=False):
    """Validate a raw NURE/NGDB extract (read with pandas, not load_nure).

    Returns (clean_df, ledger_df). The raw index is kept so rows can be matched
    back to the extract. With return_details=True also returns a dict with
    feature_gate, batches, features_kept, submitter_prediction and all_df
    (censored, in-bbox, before the location / medium filters).
    """
    vcfg = cfg.get('nure_validation', {}) or {}
    max_prec = vcfg.get('max_coord_prec', DEFAULT_MAX_COORD_PREC)
    sources = vcfg.get('sample_sources', DEFAULT_SAMPLE_SOURCES)
    max_cf = float(vcfg.get('max_censored_frac', DEFAULT_MAX_CENSORED_FRAC))
    ledger = []

    raw_cols = analyte_columns(df_raw)
    df, converted = censor_half_mdl(df_raw, raw_cols)
    for col in converted:
        ledger.append(_ledger_row(
            'units', col, int(df[col].notna().sum()), int(df[col].notna().sum()),
            int(df[col].notna().sum()), 'wt% → ppm (×10,000) when median below load_nure threshold',
            f'{col} and {col}_mdl scaled'))

    n0 = len(df)
    df = df.dropna(subset=['lat', 'lon'])
    sa = (cfg.get('study_area') or {}).get('bbox')
    if sa:
        in_box = (df['lon'].between(sa['lon_min'], sa['lon_max'])
                  & df['lat'].between(sa['lat_min'], sa['lat_max']))
        df = df[in_box]
        ledger.append(_ledger_row('study_area', 'lat/lon', n0, n0 - len(df), len(df),
                                  'drop missing lat/lon and samples outside study_area.bbox'))
    all_df = df.copy()
    ledger.append(_dup_row(df, 'input'))

    keep, rule = precision_mask(df, max_prec)
    if keep is None:
        ledger.append(_ledger_row('coord_precision', 'coord_prec_est', len(df), 0, len(df),
                                  f'{NOT_ASSESSABLE}: {rule}'))
    else:
        dropped = df.loc[~keep, 'coord_prec_est']
        ledger.append(_ledger_row('coord_precision', 'coord_prec_est', len(df), int((~keep).sum()),
                                  int(keep.sum()), rule,
                                  f'excluded: {_counts_note(dropped)}' if len(dropped) else ''))
        df = df[keep]

    if 'sample_source' not in df.columns:
        ledger.append(_ledger_row('sample_medium', 'sample_source', len(df), 0, len(df),
                                  f'{NOT_ASSESSABLE}: column sample_source absent'))
    else:
        keep = df['sample_source'].astype(str).isin([str(s) for s in sources]) \
            & df['sample_source'].notna()
        dropped = df.loc[~keep, 'sample_source']
        ledger.append(_ledger_row('sample_medium', 'sample_source', len(df), int((~keep).sum()),
                                  int(keep.sum()), f'keep sample_source in {list(sources)}',
                                  f'excluded: {_counts_note(dropped)}' if len(dropped) else ''))
        df = df[keep]

    ledger.append(_dup_row(df, 'validated'))

    for c in raw_cols:
        if f'{c}_censored' not in df.columns:
            continue
        raw_v = pd.to_numeric(df_raw.loc[df.index, c], errors='coerce')
        n_an = int(raw_v.notna().sum())
        if n_an == 0:
            continue
        neg = raw_v[raw_v < 0]
        top = neg.abs().value_counts().head(5)
        mdls = ', '.join(f'{k:g} x{v}' for k, v in top.items())
        legacy_nan = int((neg.abs() > 10).sum())
        ledger.append(_ledger_row(
            'censored', c, n_an, int(len(neg)), n_an,
            'negative v → |v|/2, {el}_mdl=|v|, {el}_censored=True (any magnitude)',
            (f'MDLs: {mdls}; legacy load_nure sets {legacy_nan} to NaN' if len(neg)
             else 'no censored values')))

    gate = feature_gate(df, FEATURES, max_censored_frac=max_cf)
    for _, g in gate.iterrows():
        ledger.append(_ledger_row(
            'feature_gate', g['feature'], g['n_analysed'], g['n_censored'],
            g['n_analysed'] if g['keep'] else 0,
            f'drop if censored_frac > {max_cf:g} or median MDL > log mean+2SD of detected',
            g['reason']))
    features_kept = gate.loc[gate['keep'], 'feature'].tolist()

    batches = batch_table(all_df, validated_index=df.index)
    preds = {}
    pred_rule = 'RF(200, random_state=42) stratified 5-fold balanced accuracy; submitters with >= 20 samples'
    for stage, frame, feats in (
        ('all_samples / all features', all_df, FEATURES),
        ('validated / all features', df, FEATURES),
        ('validated / gated features', df, features_kept),
    ):
        res = submitter_prediction(frame, feats)
        preds[stage] = res
        ledger.append(_prediction_row(stage, res, pred_rule))

    if 'submitter' in all_df.columns and len(batches):
        sub_b = batches[(batches['group_by'] == 'submitter') & (batches['n_samples'] >= MIN_BATCH_N)]
        for f, fb in sub_b.groupby('feature', sort=False):
            med = fb['median_detected'].dropna()
            med = med[med > 0]
            if len(med) < 2:
                continue
            ratio = float(med.max() / med.min())
            ledger.append(_ledger_row(
                'batch_median_spread', f, int(fb['n_samples'].sum()), None, None,
                'max/min median detected value across submitters with >= 20 samples; flag > 10',
                f"{'FLAG ' if ratio > 10 else ''}ratio {ratio:.3g} "
                f"(min {med.min():.3g}, max {med.max():.3g})"))
    else:
        ledger.append(_ledger_row('lab_batch', 'submitter', len(all_df), None, None,
                                  f'{NOT_ASSESSABLE}: column submitter absent'))

    ledger_df = pd.DataFrame(ledger, columns=LEDGER_COLUMNS)
    if not return_details:
        return df, ledger_df
    return df, ledger_df, {
        'feature_gate': gate,
        'features_kept': features_kept,
        'batches': batches,
        'submitter_prediction': preds,
        'all_df': all_df,
    }


def dedup_mrds(gdf):
    """Drop duplicate dep_id, then duplicate coordinates at 4 dp (keep first).

    Returns (gdf_dedup, report) where report is a list of step dicts.
    """
    report = []
    n = len(gdf)
    if 'dep_id' in gdf.columns:
        dup = gdf['dep_id'].notna() & gdf['dep_id'].duplicated(keep='first')
        gdf = gdf[~dup]
        report.append({'step': 'duplicate dep_id', 'n_in': n, 'n_dropped': int(dup.sum()),
                       'n_out': len(gdf), 'rule': 'keep first dep_id'})
    else:
        report.append({'step': 'duplicate dep_id', 'n_in': n, 'n_dropped': 0, 'n_out': n,
                       'rule': f'{NOT_ASSESSABLE}: column dep_id absent'})
    n = len(gdf)
    key = pd.DataFrame({'x': gdf.geometry.x.round(4).values, 'y': gdf.geometry.y.round(4).values},
                       index=gdf.index)
    dup = key.duplicated(keep='first')
    gdf = gdf[~dup.values]
    report.append({'step': 'duplicate coordinates (4 dp)', 'n_in': n, 'n_dropped': int(dup.sum()),
                   'n_out': len(gdf), 'rule': 'keep first lon/lat rounded to 4 dp'})
    return gdf, report


def run(cfg):
    """Validate NURE and MRDS for one belt; write the four ledger tables."""
    import geopandas as gpd

    ensure_outputs(cfg['outputs_dir'])
    raw = pd.read_csv(cfg['data']['nure_csv'])
    clean, ledger, details = validate_nure(raw, cfg, return_details=True)

    mrds = gpd.read_file(cfg['data']['mrds_geojson'])
    _, mrds_report = dedup_mrds(mrds)
    mrds_df = pd.DataFrame(mrds_report)

    paths = {
        'data_validation_ledger.csv': ledger,
        'data_validation_feature_gate.csv': details['feature_gate'],
        'data_validation_batches.csv': details['batches'],
        'data_validation_mrds_dedup.csv': mrds_df,
    }
    for name, frame in paths.items():
        p = out(cfg, 'tables', name)
        frame.to_csv(p, index=False)
        print(f'  Wrote {p}')

    print(f"  NURE: {len(raw)} raw → {len(clean)} validated samples")
    print(f"  Features kept: {details['features_kept']}")
    for stage, res in details['submitter_prediction'].items():
        if res.get('status') == 'ok':
            print(f"  Submitter prediction [{stage}]: balanced accuracy "
                  f"{res['balanced_accuracy']:.3f} vs chance {res['chance']:.3f}")
        else:
            print(f"  Submitter prediction [{stage}]: {res['status']}")
    print(f"  MRDS: {mrds_report[0]['n_in']} → {mrds_report[-1]['n_out']} after dedup")
    return clean, ledger, details


def main(argv=None):
    import yaml

    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default='configs/ne_washington/config.yaml')
    args = ap.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)


if __name__ == '__main__':
    main()
