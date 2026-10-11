"""Impact of NURE / MRDS data validation on Task 9 (method test).

Same proximity + elevation labels and the same three CV schemes (shuffled
5-fold, dead-zone, 0.4° block) as the published run. The forest stays frozen:
no joblib, no Fig 10, no task9_ml_* tables. Writes only:
  {outputs_dir}/tables/task9_validated_cv.csv
  {outputs_dir}/tables/task9_validated_feature_importance.csv
  {outputs_dir}/tables/data_validation_site_join.csv
  {outputs_dir}/tables/task9_validated_submitter_confound.csv

  python -m pipeline.task9_ml_targeting configs/ne_washington/config.yaml --validated
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from pipeline.data_validation import dedup_mrds, validate_nure
from pipeline.ml_preprocess import _log_impute, feature_list_for_cfg, usable_feature_columns
from pipeline.ml_spatial import _fit_auc, block_cv_aucs, deadzone_cv_aucs, summarize_aucs
from pipeline.task9_ml_targeting import filter_mrds_commodity, proximity_labels
from pipeline.utils import bbox, ensure_outputs, load_nure, out, resolve_data_path


def _in_bbox(df, cfg):
    lon_min, lon_max, lat_min, lat_max = bbox(cfg)
    df = df.dropna(subset=['lat', 'lon'])
    return df[(df['lon'] >= lon_min) & (df['lon'] <= lon_max)
              & (df['lat'] >= lat_min) & (df['lat'] <= lat_max)].copy()


def shuffled_cv_aucs(X, y, n_splits=5):
    """Published shuffled stratified k-fold AUCs (same RF and seed as Task 9)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        a = _fit_auc(X, y, tr, te)
        if a is not None:
            aucs.append(a)
    return aucs


def evaluate_scenario(name, df, feats, mrds_gdf, cfg):
    """Label, cross-validate and fit one scenario. Returns (row, importance_df)."""
    ml = cfg.get('ml', {}) or {}
    radius = ml.get('mrds_proximity_deg', 0.15)
    elev = ml.get('mrds_elev_diff_m', 200)
    cf = ml.get('mrds_commodity_filter', 'all')
    dem_path = resolve_data_path(cfg, 'dem_tif')

    y, label_desc = proximity_labels(df, mrds_gdf, dem_path, radius, elev, cf)
    row = {
        'scenario': name, 'n_samples': int(len(y)), 'n_positive': int(y.sum()),
        'positive_rate': round(float(y.mean()), 4) if len(y) else np.nan,
        'n_features': len(feats), 'features': ','.join(feats),
        'mrds_sites': int(len(mrds_gdf)), 'label_desc': label_desc,
    }
    if not feats or y.sum() < 2 or y.sum() > len(y) - 2:
        row['note'] = 'not scored: no features or a single label class'
        return row, pd.DataFrame(columns=['scenario', 'feature', 'importance', 'rank'])

    X = _log_impute(df, feats)[0].values
    lon, lat = df['lon'].values, df['lat'].values
    dz_aucs, dz_dropped = deadzone_cv_aucs(X, y, np.column_stack([lon, lat]), deadzone_deg=radius)
    row['deadzone_train_dropped_mean'] = round(float(np.mean(dz_dropped)), 1)
    row['train_per_fold'] = int(round(len(y) * 0.8))
    for scheme, aucs in (
        ('shuffled', shuffled_cv_aucs(X, y)),
        ('deadzone', dz_aucs),
        ('block', block_cv_aucs(X, y, lon, lat)),
    ):
        mean, sd, folds = summarize_aucs(aucs)
        row[f'{scheme}_auc_mean'] = round(mean, 4) if mean is not None else np.nan
        row[f'{scheme}_auc_sd'] = round(sd, 4) if sd is not None else np.nan
        row[f'{scheme}_n_folds'] = len(folds)
        row[f'{scheme}_folds'] = ';'.join(f'{a:.4f}' for a in folds)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    imp = (pd.DataFrame({'feature': feats, 'importance': rf.feature_importances_})
           .sort_values('importance', ascending=False).reset_index(drop=True))
    imp.insert(0, 'scenario', name)
    imp['rank'] = np.arange(1, len(imp) + 1)
    row['top_feature'] = imp.loc[0, 'feature']
    row['top_feature_importance'] = round(float(imp.loc[0, 'importance']), 4)
    return row, imp


def submitter_confound(df, feats, mrds_gdf, cfg, min_class=5):
    """Label AUC from submitter alone (out-of-fold rate) and within-submitter CV.

    If the submitter-only AUC is close to the forest's, the forest may be
    ranking labs, not drainages.
    """
    from sklearn.metrics import roc_auc_score

    if 'submitter' not in df.columns or not feats:
        return pd.DataFrame([{'group': 'all', 'note': 'no submitter column'}])
    ml = cfg.get('ml', {}) or {}
    y, _ = proximity_labels(df, mrds_gdf, resolve_data_path(cfg, 'dem_tif'),
                            ml.get('mrds_proximity_deg', 0.15), ml.get('mrds_elev_diff_m', 200),
                            ml.get('mrds_commodity_filter', 'all'))
    y = np.asarray(y)
    sub = df['submitter'].fillna('unknown').to_numpy()
    rows = []
    if len(np.unique(y)) == 2:
        oof = np.full(len(y), np.nan)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, te in skf.split(np.zeros(len(y)), y):
            rates = pd.Series(y[tr]).groupby(sub[tr]).mean()
            oof[te] = pd.Series(sub[te]).map(rates).fillna(y[tr].mean()).to_numpy()
        rows.append({'group': 'submitter_only_predictor', 'n_samples': len(y),
                     'n_positive': int(y.sum()),
                     'shuffled_auc_mean': round(float(roc_auc_score(y, oof)), 4),
                     'note': 'out-of-fold positive rate by submitter; no chemistry'})
    for name in pd.unique(sub):
        m = sub == name
        yg = y[m]
        n_min = int(min(yg.sum(), len(yg) - yg.sum()))
        row = {'group': f'within:{name}', 'n_samples': int(m.sum()), 'n_positive': int(yg.sum())}
        if n_min < min_class:
            row['note'] = f'not scored: minority class {n_min} < {min_class}'
            rows.append(row)
            continue
        g = df.loc[m]
        X = _log_impute(g, feats)[0].values
        k = min(5, n_min)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        aucs = [a for tr, te in skf.split(X, yg) if (a := _fit_auc(X, yg, tr, te)) is not None]
        blk = block_cv_aucs(X, yg, g['lon'].values, g['lat'].values)
        for scheme, a in (('shuffled', aucs), ('block', blk)):
            mean, sd, folds = summarize_aucs(a)
            row[f'{scheme}_auc_mean'] = round(mean, 4) if mean is not None else np.nan
            row[f'{scheme}_n_folds'] = len(folds)
        rows.append(row)
    return pd.DataFrame(rows)


def _matches_published(cfg, row):
    """Do recomputed published folds equal task9_ml_cv_scores / spatial_cv? None if absent."""
    cv_p = out(cfg, 'tables', 'task9_ml_cv_scores.csv')
    if not os.path.exists(cv_p) or not row.get('shuffled_folds'):
        return None
    pub = pd.read_csv(cv_p)['roc_auc'].round(4).tolist()
    mine = [float(a) for a in row['shuffled_folds'].split(';')]
    ok = len(pub) == len(mine) and np.allclose(pub, mine, atol=1e-4)
    sp_p = out(cfg, 'tables', 'task9_ml_spatial_cv.csv')
    if ok and os.path.exists(sp_p):
        sp = pd.read_csv(sp_p)
        for scheme, key in (('deadzone', 'deadzone_'), ('block', 'block_')):
            pub_s = sp.loc[sp['scheme'].str.startswith(key), 'roc_auc'].round(4).tolist()
            mine_s = [float(a) for a in str(row.get(f'{scheme}_folds') or '').split(';') if a]
            ok = ok and len(pub_s) == len(mine_s) and np.allclose(pub_s, mine_s, atol=1e-4)
    return bool(ok)


def site_join_table(cfg, before_df, after_df):
    """Nearest NURE sample within site_join_radius_deg per config site, before vs after."""
    from pipeline.task1_coplacer import nearest_nure_row, site_join_radius_deg

    radius = site_join_radius_deg(cfg)
    rows = []
    for s in cfg.get('sites') or []:
        rec = {'site': s.get('name'), 'radius_deg': radius}
        picks = {}
        for tag, frame in (('before', before_df), ('after', after_df)):
            r, d_km = nearest_nure_row(s['lon'], s['lat'], frame, radius)
            picks[tag] = r
            rec[f'{tag}_lab_id'] = r.get('lab_id') if r is not None else None
            rec[f'{tag}_Th'] = float(r['Th']) if r is not None and pd.notna(r.get('Th')) else np.nan
            rec[f'{tag}_dist_km'] = round(d_km, 2) if r is not None else np.nan
            rec[f'{tag}_sample_source'] = r.get('sample_source') if r is not None else None
            rec[f'{tag}_coord_prec_est'] = r.get('coord_prec_est') if r is not None else None
        a = picks['after']
        rec['after_Th_censored'] = (bool(a['Th_censored'])
                                    if a is not None and pd.notna(a.get('Th_censored')) else None)
        rec['after_Th_mdl'] = (float(a['Th_mdl'])
                               if a is not None and pd.notna(a.get('Th_mdl')) else np.nan)
        rec['sample_changed'] = rec['before_lab_id'] != rec['after_lab_id']
        rows.append(rec)
    return pd.DataFrame(rows)


def run_validated_impact(cfg):
    """Published vs validated Task 9 side by side; frozen outputs untouched."""
    import geopandas as gpd

    ensure_outputs(cfg['outputs_dir'])
    ml = cfg.get('ml', {}) or {}
    cf = ml.get('mrds_commodity_filter', 'all')
    feature_set = feature_list_for_cfg(cfg, cf)

    pub_df = _in_bbox(load_nure(cfg), cfg)
    raw = pd.read_csv(cfg['data']['nure_csv'])
    clean, _, details = validate_nure(raw, cfg, return_details=True)

    mrds = filter_mrds_commodity(gpd.read_file(cfg['data']['mrds_geojson']), cf)
    mrds_dedup, _ = dedup_mrds(mrds)

    pub_feats, _ = usable_feature_columns(pub_df, feature_set)
    filt_df = pub_df.loc[pub_df.index.intersection(clean.index)]
    filt_feats, _ = usable_feature_columns(filt_df, feature_set)
    gated = [f for f in feature_set if f in details['features_kept']]
    val_feats, _ = usable_feature_columns(clean, gated)

    scenarios = [
        ('published', pub_df, pub_feats, mrds),
        ('filters_only_legacy_values', filt_df, filt_feats, mrds),
        ('validated', clean, val_feats, mrds_dedup),
    ]
    vcfg = cfg.get('nure_validation', {}) or {}
    sens_sources = vcfg.get('sensitivity_sample_sources')
    if sens_sources:
        s_cfg = {**cfg, 'nure_validation': {**vcfg, 'sample_sources': list(sens_sources)}}
        s_clean, _, s_det = validate_nure(raw, s_cfg, return_details=True)
        s_gated = [f for f in feature_set if f in s_det['features_kept']]
        s_feats, _ = usable_feature_columns(s_clean, s_gated)
        tag = '+'.join(str(s).split('/')[0] for s in sens_sources)
        scenarios.append((f'sensitivity_{tag}', s_clean, s_feats, mrds_dedup))

    rows, imps = [], []
    for name, frame, feats, m in scenarios:
        print(f"  Scenario {name}: {len(frame)} samples, features {feats}")
        row, imp = evaluate_scenario(name, frame, feats, m, cfg)
        rows.append(row)
        imps.append(imp)

    rows[0]['matches_published_tables'] = _matches_published(cfg, rows[0])
    dem_path = resolve_data_path(cfg, 'dem_tif')
    y_raw, _ = proximity_labels(clean, mrds, dem_path, ml.get('mrds_proximity_deg', 0.15),
                                ml.get('mrds_elev_diff_m', 200), cf)
    y_ded, _ = proximity_labels(clean, mrds_dedup, dem_path, ml.get('mrds_proximity_deg', 0.15),
                                ml.get('mrds_elev_diff_m', 200), cf)
    rows[2]['labels_changed_by_mrds_dedup'] = int((y_raw != y_ded).sum())
    if ml.get('label_method', 'proximity') != 'proximity':
        for r in rows:
            r['note'] = (r.get('note', '') + ' config label_method is '
                         f"{ml.get('label_method')}; impact run uses proximity").strip()

    cv_df = pd.DataFrame(rows)
    imp_df = pd.concat([i for i in imps if len(i)], ignore_index=True)
    join_df = site_join_table(cfg, pub_df, clean)
    conf_df = submitter_confound(clean, val_feats, mrds_dedup, cfg)

    for name, frame in (('task9_validated_cv.csv', cv_df),
                        ('task9_validated_feature_importance.csv', imp_df),
                        ('data_validation_site_join.csv', join_df),
                        ('task9_validated_submitter_confound.csv', conf_df)):
        p = out(cfg, 'tables', name)
        frame.to_csv(p, index=False)
        print(f'  Wrote {p}')

    cols = ['scenario', 'n_samples', 'n_positive', 'n_features', 'shuffled_auc_mean',
            'deadzone_auc_mean', 'block_auc_mean', 'top_feature']
    print(cv_df[[c for c in cols if c in cv_df.columns]].to_string(index=False))
    print('  Did not touch the published joblib, Fig 10 or task9_ml_* tables.')
    return cv_df, imp_df, join_df
