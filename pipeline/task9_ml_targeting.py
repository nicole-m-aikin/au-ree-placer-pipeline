"""
Task 9: ML anomaly targeting — Random Forest classifier on NURE geochemistry.

Geological feature engineering: log10-transformed multi-element stream sediment
geochemistry fed to a Random Forest classifier, then spatially continuous probability
surface via IDW geostatistical interpolation.

Published labels are MRDS proximity (see configs/*/config.yaml label_method).
Catchment-of-ranked-sites labeling is implemented but not the result of record.

Outputs:
  {outputs_dir}/figures/fig10_ml_anomaly_probability.png
  {outputs_dir}/tables/task9_ml_feature_importance.csv
  {outputs_dir}/tables/task9_ml_cv_scores.csv
  models/task9_rf_placer_gold.joblib          # published estimator (proximity / placer_gold)
  models/task9_rf_placer_gold.meta.json       # feature order, log-medians, CV, sklearn pin
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.filterwarnings('ignore')

from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

from pipeline.ml_preprocess import (
    COUSIN_FEATURES,
    FEATURES,
    FEATURES_BY_COMMODITY,
    feature_list_for_cfg,
    usable_feature_columns,
)
from pipeline.ml_artifacts import (
    is_published_training_belt,
    persist_belt_model,
    persist_gold_mrds,
    persist_published_model,
    study_area_short,
)
from pipeline.ml_preprocess import FEATURES, FEATURES_BY_COMMODITY, _log_impute
from pipeline.ml_spatial import (
    CV_OPTIMISM_NOTE,
    NEGATIVE_CLASS_NOTE,
    TARGET_QUESTION,
    block_cv_aucs,
    deadzone_cv_aucs,
    summarize_aucs,
)
from pipeline.utils import (WONG, setup_mpl, load_nure, anomaly_threshold,
                             watermark, save_fig, ensure_outputs, out, bbox,
                             map_extent, north_arrow, scale_bar,
                             canada_border, locator_inset,
                             topo_contours, rivers_with_arrows,
                             MAP_W, MAP_H, _FIG_LM, _FIG_RM, _FIG_TM, _FIG_BM,
                             _FIG_HGAP, _FIG_CW, _FIG_CG, _ax_rect)


def _pour_gdf_from_sites(cfg):
    """Build pour-point GeoDataFrame from the study-area site list in config."""
    import geopandas as gpd
    sites = cfg.get('sites') or []
    if not sites:
        raise ValueError("label_method='catchment' requires cfg['sites'] pour points")
    return gpd.GeoDataFrame(
        {'name': [s.get('name') for s in sites]},
        geometry=gpd.points_from_xy(
            [s['lon'] for s in sites],
            [s['lat'] for s in sites],
        ),
        crs=cfg.get('study_area', {}).get('crs', 'EPSG:4326'),
    )


def _polygonize_catchment(grid, catch):
    """Convert a catchment raster to a polygon, windowed to the True cells only.

    Full-grid polygonize on a ~100 M-cell DEM is the dominant cost of the old
    loop. Windowing keeps the same geometry and drops that cost by 10–100×.
    """
    from rasterio.features import shapes as rio_shapes
    from rasterio.transform import Affine
    from shapely.geometry import shape
    from shapely.ops import unary_union

    arr = np.asarray(catch)
    if arr.dtype != np.uint8:
        arr = (arr > 0).astype(np.uint8)
    ys, xs = np.nonzero(arr)
    if ys.size == 0:
        return None
    r0, r1 = int(ys.min()), int(ys.max()) + 1
    c0, c1 = int(xs.min()), int(xs.max()) + 1
    window = arr[r0:r1, c0:c1]
    aff = Affine(*tuple(grid.affine)[:6])
    window_aff = aff * Affine.translation(c0, r0)
    polys = [
        shape(geom)
        for geom, val in rio_shapes(
            window, mask=window.astype(bool), transform=window_aff
        )
        if val == 1
    ]
    if not polys:
        return None
    return unary_union(polys)


def delineate_catchments(mrds_gdf, dem_path, snap_max_m=5000, snap_percentile=99.5):
    """
    Return a GeoDataFrame of upstream drainage catchment polygons for each
    unique snapped pour point in mrds_gdf. Uses pysheds 0.5 API with D8 flow
    direction.

    Intended input is the study-area site list (typically ~12 pour points),
    not the full MRDS gold inventory. Sites where snapping moves the pour
    point > snap_max_m metres or where delineation fails are skipped.
    Output CRS matches mrds_gdf.crs (expected EPSG:4326).

    snap_percentile defaults to 99.5 (not 90). On this 30 m DEM the 90th
    percentile of D8 accumulation is only ~41 cells — a hillslope rivulet —
    which produced empty NURE labels. 99.5 is ~19k cells (~17 km²).
    """
    import geopandas as gpd
    from pysheds.grid import Grid

    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)

    pit_filled = grid.fill_pits(dem)
    flooded    = grid.fill_depressions(pit_filled)
    inflated   = grid.resolve_flats(flooded)
    fdir       = grid.flowdir(inflated)
    acc        = grid.accumulation(fdir)

    snap_threshold = int(np.percentile(acc.flatten(), snap_percentile))
    snap_mask = acc > snap_threshold
    print(f"  Snap mask: accumulation > p{snap_percentile} ({snap_threshold} cells)")

    # metres per degree latitude (approximate, good enough for snapping check)
    M_PER_DEG = 111_000.0

    snapped = []
    seen = set()
    for _, site in mrds_gdf.iterrows():
        lon, lat = site.geometry.x, site.geometry.y
        try:
            x_snap, y_snap = grid.snap_to_mask(snap_mask, (lon, lat))
        except Exception as e:
            warnings.warn(f"  catchment snap failed for site at ({lon:.3f},{lat:.3f}): {e}")
            continue

        dx_m = abs(x_snap - lon) * M_PER_DEG * np.cos(np.radians(lat))
        dy_m = abs(y_snap - lat) * M_PER_DEG
        if np.hypot(dx_m, dy_m) > snap_max_m:
            warnings.warn(
                f"  catchment snap for ({lon:.3f},{lat:.3f}) moved "
                f"{np.hypot(dx_m, dy_m):.0f} m > {snap_max_m} m limit; skipping"
            )
            continue

        key = (round(float(x_snap), 5), round(float(y_snap), 5))
        if key in seen:
            continue
        seen.add(key)
        name = None
        if 'name' in site.index and pd.notna(site.get('name')):
            name = str(site['name'])
        snapped.append((float(x_snap), float(y_snap), lon, lat, name))

    print(f"  Delineating {len(snapped)} unique pour points "
          f"(from {len(mrds_gdf)} input sites)")

    rows = []
    for i, (x_snap, y_snap, lon, lat, name) in enumerate(snapped, 1):
        print(f"  catchment {i}/{len(snapped)} at ({lon:.3f},{lat:.3f})")
        try:
            catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype='coordinate')
            geom = _polygonize_catchment(grid, catch)
        except Exception as e:
            warnings.warn(f"  catchment delineation failed for ({lon:.3f},{lat:.3f}): {e}")
            continue

        if geom is None or geom.is_empty:
            warnings.warn(f"  empty catchment polygon for ({lon:.3f},{lat:.3f}); skipping")
            continue

        rows.append({
            'geometry': geom,
            'pour_lon': float(x_snap),
            'pour_lat': float(y_snap),
            'orig_lon': float(lon),
            'orig_lat': float(lat),
            'name': name,
        })

    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs=mrds_gdf.crs)

    result = gpd.GeoDataFrame(rows, crs=mrds_gdf.crs)
    if result.crs != mrds_gdf.crs:
        result = result.to_crs(mrds_gdf.crs)
    return result


def run(cfg, export_only=False):
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
    label_method      = cfg.get('ml', {}).get('label_method', 'proximity')
    mrds_radius       = cfg.get('ml', {}).get('mrds_proximity_deg', 0.15)
    max_elev_diff     = cfg.get('ml', {}).get('mrds_elev_diff_m', 200)
    top_pct           = cfg.get('ml', {}).get('anomaly_top_pct', 0.10)  # fallback only
    commodity_filter  = cfg.get('ml', {}).get('mrds_commodity_filter', 'all')

    # Select per-commodity feature set when splitting models; fall back to full set.
    # Drop columns with no positive measurements — empty Au/As would become a
    # constant log-median fill and fake a pathfinder the belt never assayed.
    include_cousins = bool((cfg.get('ml') or {}).get('include_cousins'))
    feature_set = feature_list_for_cfg(cfg, commodity_filter)
    avail_feats, features_dropped = usable_feature_columns(df, feature_set)
    if include_cousins:
        thin = []
        for f in list(avail_feats):
            if f not in COUSIN_FEATURES:
                continue
            n_pos = int((pd.to_numeric(df[f], errors='coerce') > 0).sum())
            if n_pos < 20:
                thin.append(f)
                avail_feats.remove(f)
        if thin:
            features_dropped = list(features_dropped) + thin
            print(f"  Cousins dropped (<20 positives): {thin}")
    if features_dropped:
        print(f"  Features dropped (missing or no positive values): {features_dropped}")
    if not avail_feats:
        raise ValueError('No usable Task 9 features after dropping empty columns.')
    log_X, log_medians = _log_impute(df, avail_feats)
    X = log_X.values
    elements_used = avail_feats

    mrds_label_used = False
    gold_mrds_coords = None
    nure_coords = None
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
                "too few for labeling — falling back to anomaly index."
            )

        print(f"  MRDS label: {len(mrds_gdf)} sites after commodity_filter='{commodity_filter}'")
        gold_mrds_coords = np.column_stack([mrds_gdf.geometry.x.values,
                                            mrds_gdf.geometry.y.values])
        nure_coords = np.column_stack([df['lon'].values, df['lat'].values])

        # Resolve DEM path once — used by both catchment and proximity (elevation) branches.
        dem_path = cfg.get('data', {}).get('dem_tif')
        if dem_path and not os.path.isabs(dem_path):
            dem_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), dem_path)

        if label_method == 'catchment' and dem_path and os.path.exists(dem_path):
            # Pour points are the ranked study-area sites (~12), not the full
            # MRDS gold inventory (~1,600 overlapping records).
            pour_gdf = _pour_gdf_from_sites(cfg)
            print(f"  Catchment pour points: {len(pour_gdf)} config sites "
                  f"(not {len(mrds_gdf)} MRDS records)")
            snap_max = cfg.get('ml', {}).get('catchment_snap_max_m', 5000)
            snap_pct = cfg.get('ml', {}).get('catchment_snap_percentile', 99.5)
            catchment_gdf = delineate_catchments(
                pour_gdf, dem_path, snap_max_m=snap_max, snap_percentile=snap_pct
            )
            if len(catchment_gdf) == 0:
                raise ValueError("No catchments delineated — falling back to proximity")
            nure_points = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(df['lon'], df['lat']),
                crs='EPSG:4326',
            )
            joined = gpd.sjoin(nure_points, catchment_gdf[['geometry']],
                               how='left', predicate='within')
            # Overlapping catchments duplicate rows; collapse back to one label
            # per NURE sample or X and y lengths diverge.
            is_pos = joined['index_right'].notna().groupby(joined.index).any()
            y = is_pos.reindex(nure_points.index, fill_value=False).astype(int).values
            label_desc = (f"catchment-based [{commodity_filter}], "
                          f"{len(catchment_gdf)} catchments")
            mrds_label_used = True
        else:
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
            if dem_path and os.path.exists(dem_path):
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
    if n_anom == 0 or n_anom == len(y):
        raise ValueError(
            f"ML labels are a single class ({n_anom}/{len(y)} positive) under "
            f"{label_desc}. Check pour-point snap settings or fall back to "
            f"label_method: proximity."
        )

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
    table_tag = 'task9_cousins' if include_cousins else 'task9_ml'
    cv_df.to_csv(out(cfg, 'tables', f'{table_tag}_cv_scores.csv'), index=False)
    mean_auc = cv_df['roc_auc'].mean()
    std_auc  = cv_df['roc_auc'].std()
    print(f"  CV ROC-AUC: {mean_auc:.3f} ± {std_auc:.3f}")

    # Spatial CV — neighbors of a test grab are dropped from training (dead zone
    # = label radius) and a second check holds out 0.4° map cells.
    if nure_coords is None:
        nure_coords = np.column_stack([df['lon'].values, df['lat'].values])
    dz_aucs, dz_dropped = deadzone_cv_aucs(X, y, nure_coords, deadzone_deg=mrds_radius)
    dz_mean, dz_std, dz_folds = summarize_aucs(dz_aucs)
    blk_aucs = block_cv_aucs(X, y, df['lon'].values, df['lat'].values)
    blk_mean, blk_std, blk_folds = summarize_aucs(blk_aucs)
    if dz_mean is not None:
        print(f"  Spatial dead-zone CV ROC-AUC: {dz_mean:.3f} ± {dz_std:.3f} "
              f"(dropped ~{int(np.mean(dz_dropped))} train neighbors/fold)")
    else:
        print("  Spatial dead-zone CV: skipped (a fold had one class)")
    if blk_mean is not None:
        print(f"  Spatial block CV ROC-AUC:     {blk_mean:.3f} ± {blk_std:.3f}")
    else:
        print("  Spatial block CV: skipped")
    spatial_rows = []
    for i, a in enumerate(dz_folds, 1):
        spatial_rows.append({'scheme': 'deadzone_0.15deg', 'fold': i, 'roc_auc': a})
    for i, a in enumerate(blk_folds, 1):
        spatial_rows.append({'scheme': 'block_0.4deg', 'fold': i, 'roc_auc': a})
    if spatial_rows:
        pd.DataFrame(spatial_rows).to_csv(
            out(cfg, 'tables', f'{table_tag}_spatial_cv.csv'), index=False
        )

    # ── Final model on all data ───────────────────────────────────────────────
    rf_final = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_final.fit(X, y)
    df['p_anomalous'] = rf_final.predict_proba(X)[:, 1]

    feat_imp = pd.DataFrame({'feature': avail_feats,
                             'importance': rf_final.feature_importances_}) \
                 .sort_values('importance', ascending=False).reset_index(drop=True)
    feat_imp.to_csv(out(cfg, 'tables', f'{table_tag}_feature_importance.csv'), index=False)

    # Persist proximity / placer_gold forests. Only NE WA may overwrite the
    # doorbell joblib. Cousins tests and other belts write sidecars.
    if (commodity_filter == 'placer_gold'
            and cfg.get('ml', {}).get('label_method', 'proximity') == 'proximity'):
        short = study_area_short(cfg) or 'ne_wa'
        if include_cousins:
            short = f'{short}_cousins'
        is_doorbell = is_published_training_belt(cfg) and not include_cousins
        metadata = {
            'model_id': (
                'task9_rf_placer_gold' if is_doorbell
                else f'task9_rf_placer_gold.{short}'
            ),
            'feature_order': list(avail_feats),
            'feature_units': {f: 'ppm' for f in avail_feats},
            'features_dropped': list(features_dropped),
            'log_medians': {k: float(v) for k, v in log_medians.items()},
            'training_date': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'sklearn_version': sklearn.__version__,
            'n_estimators': 200,
            'random_state': 42,
            'class_weight': 'balanced',
            'label_method': label_method,
            'label_desc': label_desc,
            'mrds_proximity_deg': mrds_radius,
            'mrds_commodity_filter': commodity_filter,
            'mrds_elev_diff_m': max_elev_diff,
            'n_samples': int(len(y)),
            'n_positive': int(n_anom),
            'cv_auc_folds': [float(v) for v in cv_df['roc_auc'].tolist()],
            'cv_auc_mean': float(mean_auc),
            'cv_auc_std': float(std_auc) if pd.notna(std_auc) else 0.0,
            'spatial_cv_method': (
                'dead-zone 5-fold: drop train samples within mrds_proximity_deg '
                'of any test sample; plus 0.4° cell StratifiedGroupKFold'
            ),
            'spatial_cv_auc_mean': dz_mean,
            'spatial_cv_auc_std': dz_std,
            'spatial_cv_auc_folds': dz_folds,
            'spatial_block_cv_auc_mean': blk_mean,
            'spatial_block_cv_auc_std': blk_std,
            'spatial_block_cv_auc_folds': blk_folds,
            'target_question': TARGET_QUESTION,
            'negative_class_note': NEGATIVE_CLASS_NOTE,
            'cv_optimism_note': CV_OPTIMISM_NOTE,
            'feature_importances': {
                row['feature']: float(row['importance']) for _, row in feat_imp.iterrows()
            },
            'study_area': cfg.get('study_area', {}).get('name', 'unknown'),
            'study_area_short': short or 'ne_wa',
            'is_doorbell': is_doorbell,
            'resource_tonnage_uncertainty': (
                'Resource-tonnage uncertainty (Task 4 Monte Carlo; P10/P50/P90 '
                'NdPr tonnes) is a separate quantity and is not exposed by this model.'
            ),
        }
        if not is_doorbell:
            metadata['doorbell_note'] = (
                'Not the public doorbell. The frozen NE Washington forest stays '
                'on /predict. This file is a belt-local forest for a walk list.'
            )
        if is_doorbell:
            model_path, meta_path = persist_published_model(rf_final, metadata)
            print(f"  Persisted published model: {model_path}")
            print(f"  Persisted metadata:        {meta_path}")
            gold_short = None
        else:
            model_path, meta_path = persist_belt_model(
                rf_final, metadata, short,
            )
            print(f"  Persisted belt-local model: {model_path}")
            print(f"  Persisted belt-local metadata: {meta_path}")
            print("  Did not touch the doorbell joblib.")
            gold_short = short
        if gold_mrds_coords is not None:
            gold_path = persist_gold_mrds(
                gold_mrds_coords, short=gold_short,
            )
            print(f"  Persisted gold MRDS coords: {gold_path} "
                  f"({len(gold_mrds_coords)} sites)")

    if include_cousins:
        df[['lat', 'lon', 'p_anomalous']].to_csv(
            out(cfg, 'tables', 'task9_cousins_nure_probability.csv'), index=False)
        print("  Cousins test — did not overwrite task9_ml_nure_probability.csv.")

    if export_only:
        print("Task 9 export-only — skipping figure and secondary models.")
        return

    # Write per-sample probability for integration scoring (Change 4a)
    if not include_cousins:
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
            _X_cm, _ = _log_impute(df, _cumo_avail)
            _X_cm = _X_cm.values
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


def unique_placer_pour_points(mrds_gdf, holdout_xy=None, cell_deg=0.02, holdout_deg=0.03):
    """Deduplicate placer-named MRDS points and drop those near ranked targets.

    holdout_xy: (N, 2) lon/lat of sites to exclude from training pour points.
    cell_deg: grid size for uniqueness (~0.02° ≈ 2 km).
    holdout_deg: exclusion radius around each ranked site (~0.03° ≈ 3 km).
    """
    import geopandas as gpd

    blob = (
        mrds_gdf.get('commodity', pd.Series('', index=mrds_gdf.index)).fillna('').astype(str)
        + ' '
        + mrds_gdf.get('site_name', pd.Series('', index=mrds_gdf.index)).fillna('').astype(str)
        + ' '
        + mrds_gdf.get('code_list', pd.Series('', index=mrds_gdf.index)).fillna('').astype(str)
    ).str.lower()
    placer = mrds_gdf.loc[blob.str.contains('placer')].copy()
    if placer.empty:
        return gpd.GeoDataFrame(geometry=[], crs=mrds_gdf.crs)

    if holdout_xy is not None and len(holdout_xy):
        hx = np.asarray(holdout_xy)[:, 0]
        hy = np.asarray(holdout_xy)[:, 1]
        keep = []
        for geom in placer.geometry:
            keep.append(np.hypot(geom.x - hx, geom.y - hy).min() > holdout_deg)
        placer = placer.loc[np.asarray(keep)]

    placer = placer.copy()
    placer['_cell'] = list(zip(
        np.round(placer.geometry.x / cell_deg).astype(int),
        np.round(placer.geometry.y / cell_deg).astype(int),
    ))
    placer = placer.drop_duplicates('_cell').drop(columns='_cell')
    return placer.reset_index(drop=True)


def run_same_basin_qa(cfg):
    """Method test: D8 catchments of deduped placer pour points, 12 sites held out.

    Does not overwrite the published Task 9 figure or CV tables.
    Writes task9_same_basin_qa.csv and prints concordance with proximity labels.
    """
    import geopandas as gpd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

    ensure_outputs(cfg['outputs_dir'])
    df = load_nure(cfg)
    lon_min, lon_max, lat_min, lat_max = bbox(cfg)
    df = df.dropna(subset=['lat', 'lon'])
    df = df[
        (df['lon'] >= lon_min) & (df['lon'] <= lon_max) &
        (df['lat'] >= lat_min) & (df['lat'] <= lat_max)
    ].copy().reset_index(drop=True)

    mrds = gpd.read_file(cfg['data']['mrds_geojson'])
    holdout = np.array([[s['lon'], s['lat']] for s in cfg.get('sites', [])])
    pour = unique_placer_pour_points(mrds, holdout_xy=holdout)
    print(f"  Same-basin pour points: {len(pour)} unique placer locations "
          f"(12 ranked sites held out)")
    if len(pour) < 5:
        raise ValueError("Too few unique placer pour points after hold-out")

    dem_path = cfg.get('data', {}).get('dem_tif')
    if dem_path and not os.path.isabs(dem_path):
        dem_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), dem_path)
    snap_max = cfg.get('ml', {}).get('catchment_snap_max_m', 5000)
    snap_pct = cfg.get('ml', {}).get('catchment_snap_percentile', 99.5)
    catch_gdf = delineate_catchments(pour, dem_path, snap_max_m=snap_max, snap_percentile=snap_pct)

    nure_pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df['lon'], df['lat']),
        crs='EPSG:4326',
    )
    joined = gpd.sjoin(nure_pts, catch_gdf[['geometry']], how='left', predicate='within')
    in_catch = joined['index_right'].notna().groupby(joined.index).any()
    in_catch = in_catch.reindex(nure_pts.index, fill_value=False)

    # Trap / near-downstream allowance: 0.02° (~2 km) of a training pour point.
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([pour.geometry.x.values, pour.geometry.y.values]))
    dist, _ = tree.query(np.column_stack([df['lon'].values, df['lat'].values]))
    near_trap = dist <= 0.02
    y = (in_catch.values | near_trap).astype(int)

    # Proximity baseline on the same NURE rows (published definition, for overlap only).
    gold = mrds[
        mrds['target_commodity'].fillna('').str.contains('Gold')
        & ~mrds['target_commodity'].fillna('').str.contains('Rare Earth')
    ]
    ptree = cKDTree(np.column_stack([gold.geometry.x.values, gold.geometry.y.values]))
    pdist, _ = ptree.query(np.column_stack([df['lon'].values, df['lat'].values]))
    y_prox = (pdist <= cfg.get('ml', {}).get('mrds_proximity_deg', 0.15)).astype(int)
    both = int(((y == 1) & (y_prox == 1)).sum())

    avail = [f for f in FEATURES if f in df.columns]
    X, _ = _log_impute(df, avail)
    X = X.values
    n_pos = int(y.sum())
    print(f"  Same-basin labels: {n_pos}/{len(y)} positive ({100*n_pos/len(y):.1f}%); "
          f"overlap with proximity positives: {both}")
    if n_pos < 10 or n_pos == len(y):
        raise ValueError(f"Same-basin labels unusable ({n_pos}/{len(y)})")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, precs, recs = [], [], []
    for tr, te in skf.split(X, y):
        rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        rf.fit(X[tr], y[tr])
        proba = rf.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[te], proba))
        p, r, _, _ = precision_recall_fscore_support(y[te], pred, labels=[1], zero_division=0)
        precs.append(float(p[0]))
        recs.append(float(r[0]))

    rf_all = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_all.fit(X, y)
    imp = sorted(zip(avail, rf_all.feature_importances_), key=lambda t: -t[1])

    qa = pd.DataFrame([{
        'n_pour_points': len(pour),
        'n_catchments': len(catch_gdf),
        'n_nure': len(y),
        'n_positive': n_pos,
        'positive_rate': round(n_pos / len(y), 4),
        'overlap_with_proximity': both,
        'cv_roc_auc_mean': round(float(np.mean(aucs)), 4),
        'cv_roc_auc_sd': round(float(np.std(aucs, ddof=1)), 4),
        'cv_precision_mean': round(float(np.mean(precs)), 4),
        'cv_recall_mean': round(float(np.mean(recs)), 4),
        'top_feature': imp[0][0],
        'top_feature_importance': round(float(imp[0][1]), 4),
        'holdout': '12 ranked config sites within 0.03 deg',
        'pour_point_rule': 'placer-named MRDS, 0.02 deg unique cells',
        'trap_buffer_deg': 0.02,
    }])
    path = out(cfg, 'tables', 'task9_same_basin_qa.csv')
    qa.to_csv(path, index=False)
    print(f"  Wrote {path}")
    print(qa.T.to_string(header=False))
    return qa


if __name__ == '__main__':
    import yaml, sys
    args = sys.argv[1:]
    same_basin = '--same-basin' in args
    export_only = '--export-only' in args
    include_cousins = '--cousins' in args
    args = [a for a in args if a not in ('--same-basin', '--export-only', '--cousins')]
    cfg_path = args[0] if args else 'configs/ne_washington/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if include_cousins:
        cfg.setdefault('ml', {})['include_cousins'] = True
    if same_basin:
        run_same_basin_qa(cfg)
    else:
        run(cfg, export_only=export_only)
