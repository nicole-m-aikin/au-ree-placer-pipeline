"""
Task 9: ML anomaly targeting — Random Forest classifier on NURE geochemistry.

Geological feature engineering: log10-transformed multi-element stream sediment
geochemistry fed to a Random Forest classifier, then spatially continuous probability
surface via IDW geostatistical interpolation.

Outputs:
  {outputs_dir}/figures/fig10_ml_anomaly_probability.png
  {outputs_dir}/tables/task9_ml_feature_importance.csv
  {outputs_dir}/tables/task9_ml_cv_scores.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

from pipeline.utils import (WONG, setup_mpl, load_nure, anomaly_threshold,
                             watermark, save_fig, ensure_outputs, out, bbox)

# Geological feature engineering: multi-element pathfinder suite
FEATURES = ['Th', 'Ce', 'La', 'P', 'U', 'Au', 'As']


def _log_impute(df, cols):
    """Log10-transform features; impute NaN and non-positive values with column median."""
    result = pd.DataFrame(index=df.index)
    for col in cols:
        v = df[col].copy() if col in df.columns else pd.Series(np.nan, index=df.index)
        v = v.where(v > 0, np.nan)
        result[col] = np.log10(v)
    for col in cols:
        med = result[col].median()
        result[col] = result[col].fillna(0.0 if pd.isna(med) else med)
    return result


def run(cfg):
    setup_mpl()
    ensure_outputs(cfg['outputs_dir'])

    df = load_nure(cfg)

    lon_min, lon_max, lat_min, lat_max = bbox(cfg)
    df = df.dropna(subset=['lat', 'lon'])
    df = df[
        (df['lon'] >= lon_min) & (df['lon'] <= lon_max) &
        (df['lat'] >= lat_min) & (df['lat'] <= lat_max)
    ].copy().reset_index(drop=True)

    avail_feats = [f for f in FEATURES if f in df.columns]
    X = _log_impute(df, avail_feats).values

    # Anomaly label: Th > mean+2SD(log) OR Au > mean+2SD(log)
    th_thresh = anomaly_threshold(df['Th']) if 'Th' in df.columns else None
    au_thresh = anomaly_threshold(df['Au']) if 'Au' in df.columns else None
    label = pd.Series(False, index=df.index)
    if th_thresh is not None:
        label = label | (df['Th'] > th_thresh)
    if au_thresh is not None and 'Au' in df.columns:
        label = label | (df['Au'] > au_thresh)
    y = label.astype(int).values

    n_anom = int(y.sum())
    print(f"Task 9: {len(y)} NURE samples; {n_anom} anomalous ({100*n_anom/len(y):.1f}%); "
          f"features: {avail_feats}")

    # ── 5-fold stratified cross-validation ───────────────────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fpr_grid = np.linspace(0, 1, 200)
    tpr_folds = []
    cv_rows = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        rf.fit(X[tr_idx], y[tr_idx])
        proba = rf.predict_proba(X[te_idx])[:, 1]
        pred  = (proba >= 0.5).astype(int)

        fpr, tpr, _ = roc_curve(y[te_idx], proba)
        fold_auc = auc(fpr, tpr)
        tpr_folds.append(np.interp(fpr_grid, fpr, tpr))

        prec, rec, f1, _ = precision_recall_fscore_support(
            y[te_idx], pred, labels=[1], zero_division=0)
        cv_rows.append({
            'fold': fold,
            'roc_auc': round(fold_auc, 4),
            'precision_anomalous': round(float(prec[0]), 4),
            'recall_anomalous':    round(float(rec[0]),  4),
            'f1_anomalous':        round(float(f1[0]),   4),
            'n_test': len(te_idx),
            'n_anomalous_test': int(y[te_idx].sum()),
        })

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(out(cfg, 'tables', 'task9_ml_cv_scores.csv'), index=False)
    mean_auc = cv_df['roc_auc'].mean()
    std_auc  = cv_df['roc_auc'].std()
    print(f"  CV ROC-AUC: {mean_auc:.3f} ± {std_auc:.3f}")

    # ── Final model on all data ───────────────────────────────────────────────
    rf_final = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_final.fit(X, y)
    df['p_anomalous'] = rf_final.predict_proba(X)[:, 1]

    feat_imp = pd.DataFrame({'feature': avail_feats,
                             'importance': rf_final.feature_importances_}) \
                 .sort_values('importance', ascending=False).reset_index(drop=True)
    feat_imp.to_csv(out(cfg, 'tables', 'task9_ml_feature_importance.csv'), index=False)

    # ── IDW-interpolated probability surface ──────────────────────────────────
    lon_g, lat_g = np.meshgrid(
        np.linspace(lon_min, lon_max, 200),
        np.linspace(lat_min, lat_max, 200),
    )
    prob_grid = griddata(
        points=np.column_stack([df['lon'], df['lat']]),
        values=df['p_anomalous'].values,
        xi=(lon_g, lat_g),
        method='linear',
    )
    # Mask cells > 0.5° from nearest sample
    tree = cKDTree(np.column_stack([df['lon'], df['lat']]))
    dist, _ = tree.query(np.column_stack([lon_g.ravel(), lat_g.ravel()]))
    prob_grid[dist.reshape(lon_g.shape) > 0.5] = np.nan

    # ── Figure 10 ─────────────────────────────────────────────────────────────
    tpr_mean = np.mean(tpr_folds, axis=0)
    tpr_std  = np.std(tpr_folds, axis=0)

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        'Figure 10 — ML Anomaly Probability: Geological Feature Engineering on NURE Stream Sediment\n'
        f'Random Forest (200 trees; 5-fold stratified CV; ROC-AUC {mean_auc:.3f} ± {std_auc:.3f}); '
        'IDW-interpolated probability surface',
        fontsize=11, fontweight='bold',
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # Panel A — feature importance bar chart (Gini impurity, sorted descending)
    ax1 = fig.add_subplot(gs[0, 0])
    bar_colors = [WONG['blue'], WONG['orange'], WONG['green'], WONG['vermillion'],
                  WONG['sky'], WONG['pink'], WONG['yellow']][:len(feat_imp)]
    bars = ax1.barh(feat_imp['feature'], feat_imp['importance'],
                    color=bar_colors, edgecolor='black', linewidth=0.5)
    ax1.invert_yaxis()
    ax1.set_xlabel('Gini feature importance', fontsize=10)
    ax1.set_title('A.  Feature importance\n(Gini impurity, final model)', fontsize=9)
    ax1.tick_params(labelsize=9)
    ax1.grid(True, axis='x', alpha=0.3)
    ax1.set_xlim(0, feat_imp['importance'].max() * 1.3)
    for bar, val in zip(bars, feat_imp['importance']):
        ax1.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', fontsize=8)

    # Panel B — ROC curve from CV folds (mean ± 1SD shaded band)
    ax2 = fig.add_subplot(gs[0, 1])
    for tpr_fold in tpr_folds:
        ax2.plot(fpr_grid, tpr_fold, color='lightgray', lw=0.8, alpha=0.6, zorder=1)
    ax2.fill_between(fpr_grid,
                     np.clip(tpr_mean - tpr_std, 0, 1),
                     np.clip(tpr_mean + tpr_std, 0, 1),
                     color=WONG['sky'], alpha=0.40, label='± 1 SD', zorder=2)
    ax2.plot(fpr_grid, tpr_mean, color=WONG['blue'], lw=2.2,
             label=f'Mean ROC (AUC = {mean_auc:.3f})', zorder=3)
    ax2.plot([0, 1], [0, 1], color='gray', ls='--', lw=1, label='Random classifier', zorder=2)
    ax2.set_xlabel('False positive rate', fontsize=10)
    ax2.set_ylabel('True positive rate', fontsize=10)
    ax2.set_title('B.  ROC curve — 5-fold CV\n(mean ± 1 SD across folds)', fontsize=9)
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
    ax2.tick_params(labelsize=9)

    # Panel C — spatial IDW-interpolated probability surface
    ax3 = fig.add_subplot(gs[0, 2])
    pm = ax3.pcolormesh(lon_g, lat_g, prob_grid,
                        cmap='viridis', vmin=0, vmax=1, shading='auto')
    cbar = plt.colorbar(pm, ax=ax3, shrink=0.85)
    cbar.set_label('P(anomalous)', fontsize=9)
    cbar.ax.axhline(0.6, color='white', lw=1.5, ls='--')
    cbar.ax.axhline(0.4, color='white', lw=1.0, ls=':')

    # MRDS placer site overlay
    try:
        import geopandas as gpd
        mrds_gdf = gpd.read_file(cfg['data']['mrds_geojson'])
        ax3.scatter(mrds_gdf.geometry.x, mrds_gdf.geometry.y,
                    marker='D', c='black', s=45, zorder=6, label='MRDS placer sites')
        ax3.legend(fontsize=8, loc='lower left')
    except Exception as _e:
        print(f"  MRDS overlay skipped: {_e}")

    ax3.plot([lon_min, lon_max, lon_max, lon_min, lon_min],
             [lat_min, lat_min, lat_max, lat_max, lat_min],
             color='white', lw=1.2, ls='--', zorder=7)
    ax3.set_xlim(lon_min - 0.05, lon_max + 0.05)
    ax3.set_ylim(lat_min - 0.05, lat_max + 0.05)
    ax3.set_xlabel('Longitude', fontsize=10)
    ax3.set_ylabel('Latitude', fontsize=10)
    ax3.set_title(
        'C.  ML anomaly probability surface\n'
        '(geological feature engineering; IDW-interpolated;\n'
        ' exploratory screening only)',
        fontsize=9,
    )
    ax3.tick_params(labelsize=8)

    fig.text(
        0.5, -0.04,
        'ML anomaly probability surface (geological feature engineering on NURE stream sediment; '
        'IDW-interpolated; exploratory screening only). '
        'P(anomaly) > 0.6 = high interest; 0.4–0.6 = moderate interest. '
        'Use alongside Tasks 1, 3, 7 — not as a standalone filter.',
        ha='center', fontsize=7.5, color='#444444', style='italic',
    )

    watermark(fig, cfg)
    save_fig(fig, out(cfg, 'figures', 'fig10_ml_anomaly_probability.png'))
    print("Task 9 complete — fig10 saved.")


if __name__ == '__main__':
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/ne_washington/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
