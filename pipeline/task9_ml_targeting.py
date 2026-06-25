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
import warnings
warnings.filterwarnings('ignore')

from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

from pipeline.utils import (WONG, setup_mpl, load_nure, anomaly_threshold,
                             watermark, save_fig, ensure_outputs, out, bbox,
                             map_extent, north_arrow, scale_bar,
                             canada_border, locator_inset,
                             topo_contours, rivers_with_arrows,
                             MAP_W, MAP_H, _FIG_LM, _FIG_RM, _FIG_TM, _FIG_BM,
                             _FIG_HGAP, _FIG_CW, _FIG_CG, _ax_rect)

# Geological feature engineering: full placer heavy mineral suite.
# REE/actinide minerals: Th, Ce, La, P (monazite — (LREE)PO4), U (uraninite/thorite)
# Au pathfinder:        Au, As (arsenopyrite halo around placer Au)
# Oxide heavy minerals: Ti (rutile + ilmenite — TiO2, FeTiO3), Fe (magnetite — Fe3O4)
# Silicate heavy minerals: Zr (zircon — ZrSiO4), Y (xenotime — YPO4)
# Ti, Fe, Zr, Y co-concentrate with monazite in placer systems through identical
# hydraulic sorting mechanisms; their inclusion captures the full heavy-mineral
# assemblage rather than only the REE-bearing fraction.
FEATURES = ['Th', 'Ce', 'La', 'P', 'U', 'Au', 'As', 'Ti', 'Fe', 'Zr', 'Y']

# Per-commodity feature sets for split-model training.
# Separating by commodity prevents feature cross-contamination: e.g. U dominates
# a blended model because Colville U anomalies spatially overlap REE MRDS sites,
# inflating U importance relative to Au/As for a gold-placer model.
FEATURES_BY_COMMODITY = {
    'placer_gold': ['Th', 'Ce', 'La', 'P', 'U', 'Au', 'As', 'Ti', 'Fe', 'Zr', 'Y'],
    'ree':         ['Th', 'Ce', 'La', 'P', 'U', 'Ti', 'Zr', 'Y'],
    'cu_mo':       ['Cu', 'Mo', 'Pb', 'Zn', 'Ag', 'Au', 'As'],
    'all':         ['Th', 'Ce', 'La', 'P', 'U', 'Au', 'As', 'Ti', 'Fe', 'Zr', 'Y'],
}


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

    # Label: MRDS proximity — geochemistry-independent ground truth.
    #
    # Any label derived from the input features (e.g. Th > threshold, or top-N% of
    # anomaly index) creates a circular classifier that achieves near-perfect AUC by
    # reconstructing its own label inputs. The only non-circular label available
    # without assay data is spatial proximity to a known MRDS placer deposit: does
    # this NURE sample have the geochemical signature of a sample taken near a known
    # placer site? This is the exact question a real targeting model answers — train
    # on confirmed deposit proximity, predict on unsampled terrain.
    #
    # label = 1 if NURE sample lies within `mrds_radius_deg` of any MRDS site for
    # the target commodity. Radius default: 0.15 degrees (~15 km), roughly one
    # drainage-basin width and two average NURE sample spacings — geologically
    # meaningful at the catchment scale rather than point-proximity.
    mrds_radius       = cfg.get('ml', {}).get('mrds_proximity_deg', 0.15)
    max_elev_diff     = cfg.get('ml', {}).get('mrds_elev_diff_m', 200)
    top_pct           = cfg.get('ml', {}).get('anomaly_top_pct', 0.10)  # fallback only
    commodity_filter  = cfg.get('ml', {}).get('mrds_commodity_filter', 'all')

    # Select per-commodity feature set when splitting models; fall back to full set.
    feature_set = FEATURES_BY_COMMODITY.get(commodity_filter, FEATURES)
    avail_feats = [f for f in feature_set if f in df.columns]
    log_X = _log_impute(df, avail_feats)
    X = log_X.values
    elements_used = avail_feats

    mrds_label_used = False
    try:
        import geopandas as gpd
        mrds_gdf = gpd.read_file(cfg['data']['mrds_geojson'])

        # Filter MRDS to the target commodity before labeling so the model learns
        # the geochemical signature of that specific deposit type.
        if commodity_filter == 'placer_gold':
            mrds_gdf = mrds_gdf[
                mrds_gdf['target_commodity'].str.contains('Gold', na=False) &
                ~mrds_gdf['target_commodity'].str.contains('Rare Earth', na=False)
            ]
        elif commodity_filter == 'ree':
            mrds_gdf = mrds_gdf[
                mrds_gdf['target_commodity'].str.contains('Rare Earth', na=False)
            ]
        elif commodity_filter == 'cu_mo':
            mrds_gdf = mrds_gdf[
                mrds_gdf['code_list'].str.contains(r'\bCU\b|\bMO\b', na=False, regex=True)
            ]
        # 'all' → no filter

        if len(mrds_gdf) < 3:
            raise ValueError(
                f"Only {len(mrds_gdf)} MRDS sites for commodity_filter='{commodity_filter}'; "
                "too few for KD-tree labeling — falling back to anomaly index."
            )

        print(f"  MRDS label: {len(mrds_gdf)} sites after commodity_filter='{commodity_filter}'")
        mrds_coords  = np.column_stack([mrds_gdf.geometry.x.values,
                                        mrds_gdf.geometry.y.values])
        nure_coords  = np.column_stack([df['lon'].values, df['lat'].values])

        from scipy.spatial import cKDTree as _cKDTree
        mrds_tree = _cKDTree(mrds_coords)
        dist_to_mrds, nearest_mrds_idx = mrds_tree.query(nure_coords)
        in_radius = dist_to_mrds <= mrds_radius

        # Elevation filter: stream sediment placers are hydraulically sorted on
        # valley floors. A NURE sample taken on a steep hillside within 3 km of a
        # placer mine is in the source-rock terrain, not the deposit zone — its
        # geochemistry reflects bedrock weathering, not placer concentration.
        # Require the NURE sample and its nearest MRDS site to be within
        # `max_elev_diff` metres of each other (both on the valley floor).
        elev_filter_applied = False
        dem_path = cfg.get('data', {}).get('dem_tif')
        if dem_path:
            import os as _os
            if not _os.path.isabs(dem_path):
                dem_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)),
                                         dem_path)
            if _os.path.exists(dem_path):
                import rasterio as _rio
                with _rio.open(dem_path) as src:
                    nodata = src.nodata
                    nure_elevs = np.array([
                        v[0] if (v[0] != nodata and not np.isnan(v[0])) else np.nan
                        for v in src.sample(zip(df['lon'], df['lat']))
                    ])
                    mrds_elevs = np.array([
                        v[0] if (v[0] != nodata and not np.isnan(v[0])) else np.nan
                        for v in src.sample(zip(mrds_gdf.geometry.x,
                                                mrds_gdf.geometry.y))
                    ])
                nearest_mrds_elev = mrds_elevs[nearest_mrds_idx]
                elev_diff = np.abs(nure_elevs - nearest_mrds_elev)
                on_valley_floor = (elev_diff <= max_elev_diff) | np.isnan(elev_diff)
                y = (in_radius & on_valley_floor).astype(int)
                elev_filter_applied = True
                label_desc = (f"MRDS proximity ≤{mrds_radius}° [{commodity_filter}] + "
                              f"elevation within {max_elev_diff} m")
        if not elev_filter_applied:
            y = in_radius.astype(int)
            label_desc = (f"MRDS proximity ≤{mrds_radius}° [{commodity_filter}] "
                          f"(no DEM elevation filter)")

        mrds_label_used = True
    except Exception as _e:
        print(f"  MRDS label unavailable ({_e}); falling back to top-{top_pct*100:.0f}% index")
        z = (log_X - log_X.mean()) / log_X.std().replace(0, 1)
        anomaly_index = z.clip(lower=0).sum(axis=1)
        y = (anomaly_index >= anomaly_index.quantile(1.0 - top_pct)).astype(int).values
        label_desc = f"top {top_pct*100:.0f}% anomaly index (fallback)"

    n_anom = int(y.sum())
    print(f"Task 9: {len(y)} NURE samples; {n_anom} positive ({100*n_anom/len(y):.1f}%); "
          f"label: {label_desc}; features: {avail_feats}")

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

    # Write per-sample probability for integration scoring (Change 4a)
    df[['lat', 'lon', 'p_anomalous']].to_csv(
        out(cfg, 'tables', 'task9_ml_nure_probability.csv'), index=False)

    # ── Cu-Mo discriminator model (secondary; no CV, importance only) ────────
    feat_imp_cumo = None
    _cumo_feats   = FEATURES_BY_COMMODITY['cu_mo']
    _cumo_avail   = [f for f in _cumo_feats if f in df.columns]
    try:
        import geopandas as _gpd_cm
        _mrds_cm = _gpd_cm.read_file(cfg['data']['mrds_geojson'])
        _mrds_cm = _mrds_cm[
            _mrds_cm['code_list'].str.contains(r'\bCU\b|\bMO\b', na=False, regex=True)
        ]
        if len(_mrds_cm) >= 3:
            _mrds_coords_cm = np.column_stack([_mrds_cm.geometry.x.values,
                                               _mrds_cm.geometry.y.values])
            _nure_coords_cm = np.column_stack([df['lon'].values, df['lat'].values])
            from scipy.spatial import cKDTree as _cKD2
            _tree_cm = _cKD2(_mrds_coords_cm)
            _dist_cm, _ = _tree_cm.query(_nure_coords_cm)
            _y_cm = (_dist_cm <= mrds_radius).astype(int)
            _X_cm = _log_impute(df, _cumo_avail).values
            _rf_cm = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
            _rf_cm.fit(_X_cm, _y_cm)
            feat_imp_cumo = (pd.DataFrame({'feature': _cumo_avail,
                                           'importance': _rf_cm.feature_importances_})
                             .sort_values('importance', ascending=False)
                             .reset_index(drop=True))
            print(f"  Cu-Mo discriminator model: {len(_mrds_cm)} MRDS sites, "
                  f"{int(_y_cm.sum())} positive labels.")
        else:
            print(f"  Cu-Mo discriminator: only {len(_mrds_cm)} sites; skipped.")
    except Exception as _e_cm:
        print(f"  Cu-Mo discriminator model skipped: {_e_cm}")

    # ── IDW-interpolated probability surface ──────────────────────────────────
    # Use map_extent bounds so the colormap fills the same padded area as all other map panels
    _xmin_g, _xmax_g, _ymin_g, _ymax_g = map_extent(cfg)
    lon_g, lat_g = np.meshgrid(
        np.linspace(_xmin_g, _xmax_g, 220),
        np.linspace(_ymin_g, _ymax_g, 220),
    )
    prob_grid = griddata(
        points=np.column_stack([df['lon'], df['lat']]),
        values=df['p_anomalous'].values,
        xi=(lon_g, lat_g),
        method='nearest',   # 'nearest' avoids NaN outside convex hull in padded margins
    )
    # Apply sparse-data guard before blending so it takes precedence: cells >0.5°
    # from any sample are unreliable regardless of whether linear interpolation
    # would fill them inside the convex hull.
    tree = cKDTree(np.column_stack([df['lon'], df['lat']]))
    dist, _ = tree.query(np.column_stack([lon_g.ravel(), lat_g.ravel()]))
    sparse_mask = dist.reshape(lon_g.shape) > 0.5
    prob_grid[sparse_mask] = np.nan
    # Blend: replace nearest with smoother linear estimate inside the data hull
    prob_grid_lin = griddata(
        points=np.column_stack([df['lon'], df['lat']]),
        values=df['p_anomalous'].values,
        xi=(lon_g, lat_g),
        method='linear',
    )
    valid_lin = ~np.isnan(prob_grid_lin) & ~sparse_mask
    prob_grid[valid_lin] = prob_grid_lin[valid_lin]

    # ── Figure 10 ─────────────────────────────────────────────────────────────
    tpr_mean = np.mean(tpr_folds, axis=0)
    tpr_std  = np.std(tpr_folds, axis=0)

    # Layout: [A1_feat_imp | A2_cumo_imp | ROC_B | map_C | cbar]
    # A1 and A2 share the space previously held by a single feat_imp panel.
    _FEAT_W = 2.4   # width for each feature importance panel (A1, A2)
    _ROC_W  = 2.8   # width for ROC panel (B)

    figW = (_FIG_LM + _FEAT_W + _FIG_HGAP + _FEAT_W + _FIG_HGAP +
            _ROC_W + _FIG_HGAP + MAP_W + _FIG_CG + _FIG_CW + _FIG_RM)
    figH = _FIG_TM + MAP_H + _FIG_BM

    fig = plt.figure(figsize=(figW, figH))
    fig.suptitle(
        'Figure 10 — ML Targeting Probability: Geological Feature Engineering on NURE Stream Sediment\n'
        f'Random Forest · 5-fold CV · ROC-AUC {mean_auc:.3f} ± {std_auc:.3f} · '
        f'label: {label_desc}',
        fontsize=10, fontweight='bold',
    )

    _col1 = _FIG_LM
    _col2 = _col1 + _FEAT_W + _FIG_HGAP
    _col3 = _col2 + _FEAT_W + _FIG_HGAP   # ROC (B)
    _col4 = _col3 + _ROC_W  + _FIG_HGAP   # map (C)
    _col5 = _col4 + MAP_W   + _FIG_CG     # colorbar

    # Panel A1 — primary model feature importance
    ax1 = fig.add_axes(_ax_rect(_col1, _FIG_BM, _FEAT_W, MAP_H, figW, figH))
    bar_colors = [WONG['blue'], WONG['orange'], WONG['green'], WONG['vermillion'],
                  WONG['sky'], WONG['pink'], WONG['yellow']][:len(feat_imp)]
    bars = ax1.barh(feat_imp['feature'], feat_imp['importance'],
                    color=bar_colors, edgecolor='black', linewidth=0.5)
    ax1.invert_yaxis()
    ax1.set_xlabel('Gini feature importance', fontsize=9)
    ax1.set_title(f'A1.  {commodity_filter.replace("_"," ").title()} model\n'
                  '(Gini impurity, final model)', fontsize=9)
    ax1.tick_params(labelsize=8)
    ax1.grid(True, axis='x', alpha=0.3)
    ax1.set_xlim(0, feat_imp['importance'].max() * 1.35)
    for bar, val in zip(bars, feat_imp['importance']):
        ax1.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', fontsize=7)

    # Callout on U bar explaining why it leads
    try:
        u_imp = feat_imp.loc[feat_imp['feature'] == 'U', 'importance'].values
        u_pos = feat_imp.index[feat_imp['feature'] == 'U'].values
        if len(u_imp) > 0:
            _xlim_right = ax1.get_xlim()[1]
            _text_x = min(u_imp[0] + 0.04, _xlim_right - 0.01)
            ax1.annotate('U/Th discriminates thorite\nvs. monazite catchments\n(cf. Task 3)',
                         xy=(u_imp[0], u_pos[0]),
                         xytext=(_text_x, u_pos[0]),
                         fontsize=6, va='center', color=WONG['vermillion'], style='italic',
                         arrowprops=dict(arrowstyle='->', color=WONG['vermillion'], lw=0.8))
    except Exception:
        pass

    # Panel A2 — Cu-Mo discriminator feature importance
    ax_a2 = fig.add_axes(_ax_rect(_col2, _FIG_BM, _FEAT_W, MAP_H, figW, figH))
    if feat_imp_cumo is not None and not feat_imp_cumo.empty:
        bar_colors_cm = [WONG['vermillion'], WONG['blue'], WONG['orange'], WONG['green'],
                         WONG['sky'], WONG['pink'], WONG['yellow']][:len(feat_imp_cumo)]
        bars_cm = ax_a2.barh(feat_imp_cumo['feature'], feat_imp_cumo['importance'],
                              color=bar_colors_cm, edgecolor='black', linewidth=0.5)
        ax_a2.invert_yaxis()
        for bar, val in zip(bars_cm, feat_imp_cumo['importance']):
            ax_a2.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                       f'{val:.3f}', va='center', fontsize=7)
        ax_a2.set_xlim(0, feat_imp_cumo['importance'].max() * 1.35)
        ax_a2.grid(True, axis='x', alpha=0.3)
        ax_a2.tick_params(labelsize=8)
    else:
        ax_a2.text(0.5, 0.5, 'Cu-Mo model\nnot available\n(insufficient MRDS sites)',
                   ha='center', va='center', transform=ax_a2.transAxes,
                   fontsize=9, color='gray', style='italic')
        ax_a2.set_xlim(0, 1); ax_a2.set_ylim(0, 1)
    ax_a2.set_xlabel('Gini feature importance', fontsize=9)
    ax_a2.set_title('A2.  Cu-Mo porphyry discriminator\n'
                    '(features: Cu, Mo, Pb, Zn, Ag, Au, As)', fontsize=9)

    # Panel B — ROC curve from CV folds (mean ± 1SD shaded band)
    ax2 = fig.add_axes(_ax_rect(_col3, _FIG_BM, _ROC_W, MAP_H, figW, figH))
    for tpr_fold in tpr_folds:
        ax2.plot(fpr_grid, tpr_fold, color='lightgray', lw=0.8, alpha=0.6, zorder=1)
    ax2.fill_between(fpr_grid,
                     np.clip(tpr_mean - tpr_std, 0, 1),
                     np.clip(tpr_mean + tpr_std, 0, 1),
                     color=WONG['sky'], alpha=0.40, label='± 1 SD', zorder=2)
    ax2.plot(fpr_grid, tpr_mean, color=WONG['blue'], lw=3.0,
             label=f'Mean ROC (AUC = {mean_auc:.3f})', zorder=4)
    ax2.plot([0, 1], [0, 1], color='gray', ls='--', lw=1, label='Random classifier', zorder=2)
    ax2.set_xlabel('False positive rate', fontsize=10)
    ax2.set_ylabel('True positive rate', fontsize=10)
    ax2.set_title('B.  ROC curve — 5-fold CV\n(mean ± 1 SD across folds)', fontsize=9)
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
    ax2.tick_params(labelsize=9)

    # Panel C — spatial IDW-interpolated probability surface
    ax3  = fig.add_axes(_ax_rect(_col4, _FIG_BM, MAP_W,   MAP_H, figW, figH))
    cax3 = fig.add_axes(_ax_rect(_col5, _FIG_BM, _FIG_CW, MAP_H, figW, figH))
    xmin_m, xmax_m, ymin_m, ymax_m = map_extent(cfg)
    ax3.set_xlim(xmin_m, xmax_m)
    ax3.set_ylim(ymin_m, ymax_m)
    ax3.set_aspect('auto')
    pm = ax3.pcolormesh(lon_g, lat_g, prob_grid,
                        cmap='viridis', vmin=0, vmax=1, shading='auto')
    cbar = fig.colorbar(pm, cax=cax3)
    cbar.set_label('P(anomalous)', fontsize=9)
    cbar.ax.axhline(0.6, color='white', lw=1.5, ls='--')
    cbar.ax.axhline(0.4, color='white', lw=1.0, ls=':')
    topo_contours(ax3, cfg, alpha=0.25, color='white')
    rivers_with_arrows(ax3, cfg, color='#7ecef4', lw=1.2, alpha=0.85)

    # MRDS placer site overlay
    try:
        import geopandas as gpd
        mrds_gdf = gpd.read_file(cfg['data']['mrds_geojson'])
        ax3.scatter(mrds_gdf.geometry.x, mrds_gdf.geometry.y,
                    marker='D', facecolors='white', edgecolors='black',
                    s=60, lw=1.2, zorder=6, label='MRDS placer sites')
        ax3.legend(fontsize=8, loc='lower left', bbox_to_anchor=(0.01, 0.01))
    except Exception as _e:
        print(f"  MRDS overlay skipped: {_e}")

    # Annotation: high-P zones without MRDS = unrecorded targets
    ax3.text(0.97, 0.97,
             'High-P zones without\nMRDS sites = potential\nunrecorded targets\n(primary output)',
             transform=ax3.transAxes, ha='right', va='top', fontsize=7,
             style='italic', color='white',
             bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.55, ec='none'))

    canada_border(ax3, cfg)
    north_arrow(ax3, x=0.96, y=0.96, size=9)
    scale_bar(ax3, cfg, length_km=50, x=0.04, y=0.12)
    locator_inset(fig, ax3, cfg)

    ax3.set_xlabel('Longitude', fontsize=10)
    ax3.set_ylabel('Latitude', fontsize=10)
    ax3.set_title(
        'C.  MRDS-proximity targeting probability\n'
        f'(IDW-interpolated; exploratory screening only)',
        fontsize=9,
    )
    ax3.tick_params(labelsize=8)
    ax3.text(0.5, -0.07,
             'P(placer-like geochemistry) > 0.6 = high interest; 0.4–0.6 = moderate interest. '
             'Use alongside Tasks 1, 3, 7.',
             transform=ax3.transAxes, ha='center', va='top',
             fontsize=7, color='#444444', style='italic')

    watermark(fig, cfg)
    save_fig(fig, out(cfg, 'figures', 'fig10_ml_anomaly_probability.png'))
    print("Task 9 complete — fig10 saved.")


if __name__ == '__main__':
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/ne_washington/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
