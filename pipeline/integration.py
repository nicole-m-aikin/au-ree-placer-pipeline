"""
Integration task: multi-criterion priority ranking across all tasks.

Reads all task outputs, computes combined scores, writes:
  {outputs_dir}/geojson/fig7_integrated_priority_tier.geojson
  {outputs_dir}/figures/fig7_integrated_priority_map.png
  {outputs_dir}/text/executive_summary_top3_sites.txt
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

from pipeline.utils import (WONG, setup_mpl, watermark, save_fig, ensure_outputs, out,
                             map_extent, hillshade, north_arrow, scale_bar,
                             canada_border, locator_inset,
                             topo_contours, rivers_with_arrows,
                             MAP_W, MAP_H, _FIG_LM, _FIG_RM, _FIG_TM, _FIG_BM,
                             _FIG_HGAP, _FIG_VGAP, _ax_rect)


def run(cfg):
    setup_mpl()
    ensure_outputs(cfg['outputs_dir'])

    w         = cfg['scoring']['weights']
    tiers     = cfg['scoring']['tier_thresholds']          # e.g. [8, 6, 4]
    ndpr_cur  = cfg['economics']['ndpr_price_central']
    patches   = cfg.get('map_domain_patches', [])
    cnty_lbls = cfg.get('county_labels', [])

    def tier_color(score):
        if score >= tiers[0]: return WONG['blue']
        if score >= tiers[1]: return WONG['orange']
        if score >= tiers[2]: return WONG['yellow']
        return '#CCCCCC'

    def score_th(src):
        return {'MONAZITE': 2, 'MIXED_UNCLEAR': 1}.get(src, 0)

    def score_conf(conf):
        return {'HIGH': 2, 'MEDIUM': 1, 'LOW': 0}.get(conf, 0)

    def score_ndpr(ndpr_t):
        if pd.isna(ndpr_t) or ndpr_t <= 0: return 0.0
        return min(2.0, np.log10(max(ndpr_t, 1)) / np.log10(1000) * 2)

    def score_au(row):
        if row.get('dual_anomaly_flag'): return 2.0
        if row.get('has_th_anomaly'):    return 0.5
        return 0.0

    # ── Load task outputs ─────────────────────────────────────────────────────
    task1_df = pd.read_csv(out(cfg, 'tables', 'task1_site_summary.csv'))
    task2_df = pd.read_csv(out(cfg, 'tables', 'task2_catchment_scores.csv'))
    task4_df = pd.read_csv(out(cfg, 'tables', 'task4_volume_tonnage_summary.csv'))
    task5_df = pd.read_csv(out(cfg, 'tables', 'task5_breakeven_analysis.csv'))
    task7_df = pd.read_csv(out(cfg, 'tables', 'task7_au_as_summary.csv'))
    nure_gdf = gpd.read_file(out(cfg, 'geojson', 'nure_classified_th_sources.geojson'))
    base_gdf = gpd.read_file(out(cfg, 'geojson', 'task1_multicommodity_targets.geojson'))

    # Load ML probability per NURE sample (written by task9; optional)
    task9_prob_df = None
    try:
        task9_prob_df = pd.read_csv(out(cfg, 'tables', 'task9_ml_nure_probability.csv'))
        print(f"  Loaded ML probability: {len(task9_prob_df)} NURE samples")
    except Exception as _e9:
        print(f"  ML probability CSV not available ({_e9}); score_ml = 0 for all sites")

    # ── Merge all task outputs ─────────────────────────────────────────────────
    merged = base_gdf.copy()

    # y_near and y_value_ppm are already in base_gdf (task1_multicommodity_targets.geojson).
    # Ensure the column exists with a safe default in case the GeoJSON was built by an older run.
    if 'y_near' not in merged.columns:
        # Fall back: try to load from task1 CSV
        if 'y_near' in task1_df.columns:
            t1_y = task1_df.set_index('name')[['y_near','y_value_ppm']]
            merged = merged.join(t1_y, on='name', how='left', lsuffix='', rsuffix='_t1')
        merged['y_near'] = merged.get('y_near', pd.Series(False, index=merged.index))
    merged['y_near'] = merged['y_near'].fillna(False)

    t2 = task2_df.set_index('name')[['source_lith_score','source_lith_desc']]
    merged = merged.join(t2, on='name', how='left')

    t4 = task4_df.set_index('site_name')[['tonnage_t','ndpr_ppm',
                                           'ndpr_tonnes','ndpr_t_p10','ndpr_t_p50','ndpr_t_p90','confidence']]
    merged = merged.join(t4, on='name', how='left')

    t5 = task5_df.set_index('site')[['total_cost_$M','npv_central_$M','breakeven_$/kg']]
    merged = merged.join(t5, on='name', how='left')
    merged['processing_viable'] = merged['breakeven_$/kg'].apply(
        lambda x: True if pd.notna(x) and x < ndpr_cur else (False if pd.notna(x) else None))

    t7 = task7_df.set_index('site_name')[['dual_anomaly_flag','has_th_anomaly']]
    merged = merged.join(t7, on='name', how='left')
    merged['dual_anomaly_flag'] = merged['dual_anomaly_flag'].fillna(False)
    merged['has_th_anomaly']    = merged['has_th_anomaly'].fillna(False)

    # ── Compute combined score ────────────────────────────────────────────────
    merged['score_th']   = merged['th_source'].apply(score_th)
    merged['score_mag']  = merged['mag_high'].astype(int)
    merged['score_lith'] = merged['source_lith_score'].fillna(1.0)
    merged['score_conf'] = merged['confidence'].apply(score_conf)
    merged['score_ndpr'] = merged['ndpr_tonnes'].apply(score_ndpr)
    merged['score_au']   = merged.apply(score_au, axis=1)
    # Xenotime/HREE score: 1 if Y anomaly near site, 0 otherwise.
    merged['score_y']    = merged['y_near'].astype(int)

    # ML probability score: p_anomalous of nearest NURE sample within 0.15°
    ML_RADIUS = 0.15
    if task9_prob_df is not None and len(task9_prob_df) > 0:
        from scipy.spatial import cKDTree as _cKDTree_int
        _nure_coords_p = task9_prob_df[['lon', 'lat']].values
        _nure_tree_p   = _cKDTree_int(_nure_coords_p)
        _site_coords   = np.column_stack([merged['lon'].values, merged['lat'].values])
        _dist_p, _idx_p = _nure_tree_p.query(_site_coords)
        _p_vals = task9_prob_df['p_anomalous'].values[_idx_p]
        merged['score_ml'] = np.where(_dist_p <= ML_RADIUS, _p_vals, 0.0)
    else:
        merged['score_ml'] = 0.0

    merged['combined_score'] = (
        merged['score_th']   * w.get('th_source_monazite', w.get('th_source', 1.0)) +
        merged['score_mag']  * w.get('magnetic_high',      w.get('mag_high', 1.0)) +
        merged['score_lith'] * w.get('source_lith',         1.0) +
        merged['score_conf'] * w.get('coverage',            w.get('confidence', 1.0)) +
        merged['score_ndpr'] * w.get('ndpr_volume',         w.get('ndpr_tonnes', 2.0)) +
        merged['score_au']   * w.get('au_pathfinder',       1.0) +
        merged['score_y']    * w.get('y_xenotime',          0.5) +
        merged['score_ml']   * w.get('ml_probability',      1.0)
    ).round(2)

    merged_sorted = merged.sort_values('combined_score', ascending=False).reset_index(drop=True)
    merged_sorted['rank'] = merged_sorted.index + 1
    merged_sorted.to_file(out(cfg, 'geojson', 'fig7_integrated_priority_tier.geojson'), driver='GeoJSON')

    # ── Figure 7 ──────────────────────────────────────────────────────────────
    # Layout (2 rows):
    #   Row 1: [map_A (MAP_W) | radar_B (RADAR_W)]
    #   Row 2: [stacked_bar_C (MAP_W) | scatter_D (RADAR_W)]
    _RADAR_W = 3.8
    _BAR_H   = 3.0

    figW = _FIG_LM + MAP_W + _FIG_HGAP + _RADAR_W + _FIG_RM
    figH = _FIG_TM + MAP_H + _FIG_VGAP + _BAR_H + _FIG_BM

    fig = plt.figure(figsize=(figW, figH))
    fig.suptitle(
        f'Figure 7 — Integrated Multi-criterion Priority Tier Map\n'
        f'{cfg["study_area"]["name"]} Placer Mine Tailings REE Assessment\n'
        f'Mineral Systems: Integrated priority — all components',
        fontsize=12, fontweight='bold',
    )

    _row1_bottom = _FIG_BM + _BAR_H + _FIG_VGAP
    _row2_bottom = _FIG_BM
    _col1_left   = _FIG_LM
    _col2_left   = _FIG_LM + MAP_W + _FIG_HGAP

    # Panel A: priority score map
    ax1 = fig.add_axes(_ax_rect(_col1_left, _row1_bottom, MAP_W,    MAP_H,   figW, figH))
    b = cfg['study_area']['bbox']
    xmin_a, xmax_a, ymin_a, ymax_a = map_extent(cfg)

    ax1.set_xlim(xmin_a, xmax_a)
    ax1.set_ylim(ymin_a, ymax_a)
    ax1.set_aspect('auto')

    hillshade(cfg, ax1, alpha=0.18)
    topo_contours(ax1, cfg)
    rivers_with_arrows(ax1, cfg)

    # Neutral full-extent background so no bare white/transparent zones appear
    # outside the config-defined domain patches (e.g. eastern map margin).
    ax1.fill([xmin_a, xmax_a, xmax_a, xmin_a],
             [ymin_a, ymin_a, ymax_a, ymax_a],
             color='#f5f0eb', alpha=0.35, zorder=0)

    # Compute the outermost patch boundary in each direction so we know which
    # patches should be extended to the map_extent edge (eliminating bare fringe
    # strips while keeping non-edge patches at their configured boundaries).
    if patches:
        xmin_p = min(p['lon_range'][0] for p in patches)
        xmax_p = max(p['lon_range'][1] for p in patches)
        ymin_p = min(p['lat_range'][0] for p in patches)
        ymax_p = max(p['lat_range'][1] for p in patches)
    else:
        xmin_p = xmax_p = ymin_p = ymax_p = None

    for patch in patches:
        lr_raw, lt_raw = patch['lon_range'], patch['lat_range']
        # Extend outer edges to map_extent; clamp inner edges normally.
        lr0 = xmin_a if lr_raw[0] <= xmin_p else max(lr_raw[0], xmin_a)
        lr1 = xmax_a if lr_raw[1] >= xmax_p else min(lr_raw[1], xmax_a)
        lt0 = ymin_a if lt_raw[0] <= ymin_p else max(lt_raw[0], ymin_a)
        lt1 = ymax_a if lt_raw[1] >= ymax_p else min(lt_raw[1], ymax_a)
        if lr0 >= lr1 or lt0 >= lt1:
            continue
        ax1.fill([lr0, lr1, lr1, lr0], [lt0, lt0, lt1, lt1],
                 color=patch['color'], alpha=0.4, zorder=0)

    nure_anom = nure_gdf[nure_gdf['th_anomaly'] == True]
    ax1.scatter(nure_anom['lon'], nure_anom['lat'],
                c='#fdbb84', s=12, alpha=0.5, zorder=1, label='NURE Th anomaly')

    # Y anomaly NURE samples (xenotime/HREE proxy) — load from full NURE data
    from pipeline.utils import load_nure, anomaly_threshold
    _nure_full = load_nure(cfg)
    _y_vals = pd.to_numeric(_nure_full['Y'], errors='coerce')
    _y_thresh_int = cfg['geochemistry'].get('xenotime_y_min_ppm',
                        float(_y_vals.mean() + 2 * _y_vals.std()))
    _nure_y_anom = _nure_full[_y_vals > _y_thresh_int].copy()
    if not _nure_y_anom.empty:
        ax1.scatter(_nure_y_anom['lon'], _nure_y_anom['lat'],
                    c=WONG['green'], s=20, marker='s', alpha=0.6, zorder=2,
                    label=f'NURE Y anomaly (>{_y_thresh_int:.0f} ppm; xenotime HREE proxy)')

    # ── Critical-mineral pathfinder anomalies (Au, Ag, Co, Mo) ──────────────
    # Each uses mean+2SD on log10 of positive values (same method as anomaly_threshold).
    # Au/As = placer gold + epithermal halo; Ag = silver-gold veins; Co = Co-Cu skarn;
    # Mo = porphyry copper-molybdenum (flags Cu-Mo systems misread as REE by blended model).
    _cm_layers = [
        ('Au',  '#FFD700',        'o', 9,  0.55, 'NURE Au anomaly (placer/epithermal)'),
        ('Ag',  WONG['pink'],     's', 9,  0.55, 'NURE Ag anomaly (Ag-Au vein)'),
        ('Co',  WONG['sky'],      '^', 9,  0.55, 'NURE Co anomaly (Co-Cu skarn)'),
        ('Mo',  '#9B59B6',        'v', 9,  0.55, 'NURE Mo anomaly (porphyry Cu-Mo)'),
    ]
    _nure_cm_handles = []
    for _elem, _color, _mk, _sz, _al, _lbl in _cm_layers:
        if _elem not in _nure_full.columns:
            continue
        _vals = pd.to_numeric(_nure_full[_elem], errors='coerce')
        _thresh = anomaly_threshold(_vals)
        if _thresh is None:
            continue
        _anom = _nure_full[_vals > _thresh].copy()
        if _anom.empty:
            continue
        ax1.scatter(_anom['lon'], _anom['lat'],
                    c=_color, s=_sz, marker=_mk, alpha=_al, zorder=2)
        _nure_cm_handles.append(
            Line2D([0], [0], marker=_mk, color='w', markerfacecolor=_color,
                   markersize=6, alpha=_al,
                   label=f'{_lbl} (>{_thresh:.2g} ppm, mean+2SD)')
        )

    scores = merged_sorted['combined_score'].values
    norm_s = (scores - scores.min()) / (scores.max() - scores.min() + 0.01)
    for i, (_, row) in enumerate(merged_sorted.iterrows()):
        color  = tier_color(row['combined_score'])
        marker = '*' if row['rank'] <= 3 else 'o'
        size   = 200 if row['rank'] <= 3 else 80 + 120 * norm_s[i]
        ax1.scatter(row.lon, row.lat, c=color, s=size,
                    edgecolors='black', linewidths=0.8, zorder=5, marker=marker)
        ax1.annotate(f"#{row['rank']} {row['name'].split()[0]}",
                     (row.lon, row.lat), xytext=(5, 4), textcoords='offset points',
                     fontsize=7.5, fontweight='bold', color='black')

    # Green ring overlay for sites with Y anomaly nearby (xenotime/HREE)
    for _, row in merged_sorted.iterrows():
        if row.get('y_near'):
            ax1.scatter(row.lon, row.lat, s=320, marker='o',
                        facecolors='none', edgecolors=WONG['green'],
                        linewidths=2.0, zorder=6)

    canada_border(ax1, cfg)
    north_arrow(ax1)
    scale_bar(ax1, cfg, x=0.65, y=0.05)
    locator_inset(fig, ax1, cfg)
    ax1.set_xlabel('Longitude', fontsize=11); ax1.set_ylabel('Latitude', fontsize=11)
    ax1.tick_params(labelsize=9)
    ax1.set_title('A.  Multi-criterion combined priority score\n(★ = top 3 sites; circle size scales with score)', fontsize=9)
    ax1.grid(True, alpha=0.2)

    tier_patches = [
        mpatches.Patch(facecolor=WONG['blue'],   edgecolor='black', label=f'Score ≥ {tiers[0]} (highest)'),
        mpatches.Patch(facecolor=WONG['orange'],  edgecolor='black', label=f'Score {tiers[1]}–{tiers[0]} (high)'),
        mpatches.Patch(facecolor=WONG['yellow'],  edgecolor='black', label=f'Score {tiers[2]}–{tiers[1]} (moderate)'),
        mpatches.Patch(facecolor='#CCCCCC',       edgecolor='black', label=f'Score < {tiers[2]} (low)'),
        mpatches.Patch(facecolor='#fdbb84',       edgecolor='none',  label='NURE Th anomaly'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=WONG['green'],
               markersize=7, label='NURE Y anomaly (xenotime HREE proxy)'),
        Line2D([0],[0], marker='o', color=WONG['green'], markerfacecolor='none',
               markeredgewidth=2, markersize=11, label='Mine site: Y anomaly nearby'),
    ] + _nure_cm_handles
    ax1.legend(handles=tier_patches, loc='lower left', fontsize=7.0, framealpha=0.88,
               ncol=2 if len(_nure_cm_handles) > 0 else 1)
    for cl in cnty_lbls:
        ax1.text(cl['lon'], cl['lat'], cl['label'], fontsize=7, color='gray',
                 style='italic', ha='center')

    # Panel B: radar chart
    ax2 = fig.add_axes(_ax_rect(_col2_left, _row1_bottom, _RADAR_W, MAP_H, figW, figH), polar=True)
    categories = ['Th Source\n(0-2)', 'Mag High\n(0-1)', 'Source\nLith (0-3)',
                  'Coverage\n(0-2)', 'NdPr\nVolume (0-2)', 'Au/As\nPathfinder (0-2)',
                  'Y Xenotime\nHREE (0-1)', 'ML\nProbability']
    N      = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, size=7)
    ax2.set_ylim(0, 3)
    ax2.set_yticks([1,2,3]); ax2.set_yticklabels(['1','2','3'], size=6)

    top5_colors = [WONG['blue'], WONG['orange'], WONG['green'], WONG['yellow'], '#CCCCCC']
    for idx, (_, row) in enumerate(merged_sorted.head(5).iterrows()):
        vals = [row['score_th'], row['score_mag'], row['score_lith'],
                row['score_conf'], row['score_ndpr'], row['score_au'],
                row['score_y'], row['score_ml']] + [row['score_th']]
        ax2.plot(angles, vals, color=top5_colors[idx], lw=2, ls='-')
        ax2.fill(angles, vals, color=top5_colors[idx], alpha=0.1)

    top5_names = list(merged_sorted.head(5)['name'])
    _radar_handles = [plt.Line2D([0],[0], color=c, lw=2.5, label=n)
                      for c, n in zip(top5_colors, top5_names)]
    # Place legend inside the polar bounding box — bbox_to_anchor clips on polar axes.
    ax2.legend(handles=_radar_handles, loc='lower left', fontsize=6.5,
               framealpha=0.9, ncol=1)
    ax2.set_title('B.  Criterion profile\n(top 5 sites)', fontsize=9, pad=15)

    # Panel C: ranked stacked bar
    ax3 = fig.add_axes(_ax_rect(_col1_left, _row2_bottom, MAP_W, _BAR_H, figW, figH))
    score_comps  = ['score_th','score_mag','score_lith','score_conf','score_ndpr','score_au','score_y','score_ml']
    comp_labels  = ['Th source (monazite)','Magnetic high','Source lithology',
                    'Data coverage','NdPr P50 (Monte Carlo)','Au/As pathfinder',
                    'Y xenotime (HREE co-product)', 'ML probability']
    comp_colors  = [WONG['green'], WONG['blue'], WONG['sky'], WONG['pink'],
                    WONG['orange'], WONG['vermillion'], '#4dac26', '#9B59B6']
    sorted_df = merged_sorted.sort_values('combined_score', ascending=True).tail(12)
    y_pos    = np.arange(len(sorted_df))
    bottoms  = np.zeros(len(sorted_df))
    for comp, label, color in zip(score_comps, comp_labels, comp_colors):
        vals = sorted_df[comp].values
        ax3.barh(y_pos, vals, left=bottoms, color=color, label=label, edgecolor='white', lw=0.3)
        bottoms += vals
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([f"#{r['rank']} {r['name']}" for _, r in sorted_df.iterrows()], fontsize=8)
    ax3.set_xlabel('Combined priority score', fontsize=11)
    ax3.tick_params(labelsize=9)
    ax3.set_title('C.  Score breakdown by criterion (all sites, ranked)\n(threshold ≥ 4.0)', fontsize=9)
    ax3.legend(fontsize=8, loc='lower right')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.axvline(tiers[2], color='red', ls='--', lw=1.2)
    ax3.set_xlim(0, 14)

    # Panel D: NdPr tonnes vs break-even price
    ax4 = fig.add_axes(_ax_rect(_col2_left, _row2_bottom, _RADAR_W, _BAR_H, figW, figH))
    conf_cmap = {'HIGH': WONG['green'], 'MEDIUM': WONG['orange'], 'LOW': WONG['vermillion']}
    t4_lookup = task4_df.set_index('site_name')
    # Build a rank lookup so we can label each dot with its integrated priority rank
    rank_lookup = merged_sorted.set_index('name')['rank'].to_dict()
    for _, row in task4_df.iterrows():
        sname = row['site_name']
        be = task5_df[task5_df['site'] == sname]['breakeven_$/kg'].values
        if len(be) > 0 and pd.notna(be[0]) and be[0] <= ndpr_cur:
            ax4.scatter(be[0], row['ndpr_tonnes'],
                        c=conf_cmap.get(row['confidence'], 'gray'),
                        s=100, edgecolors='black', linewidths=0.8, zorder=4)
            # Label with priority rank number — avoids name congestion
            rank_num = rank_lookup.get(sname, '')
            ax4.annotate(f'#{rank_num}', (be[0], row['ndpr_tonnes']),
                         xytext=(4, 3), textcoords='offset points',
                         fontsize=7.5, fontweight='bold')
            # P10-P90 error bars on NdPr endowment axis
            p10 = row.get('ndpr_t_p10', np.nan)
            p90 = row.get('ndpr_t_p90', np.nan)
            if pd.notna(p10) and pd.notna(p90):
                ax4.errorbar(be[0], row['ndpr_tonnes'],
                             yerr=[[row['ndpr_tonnes'] - p10], [p90 - row['ndpr_tonnes']]],
                             fmt='none', color='gray', lw=1.0, capsize=3, zorder=2)
    _vline_cur  = ax4.axvline(ndpr_cur, color=WONG['blue'], ls='--', lw=2,
                              label=f'Current NdPr price (${ndpr_cur:.0f}/kg)')
    _vline_low  = ax4.axvline(cfg['economics']['ndpr_price_low'], color='gray', ls=':', lw=1.5,
                              label=f'2024 trough (${cfg["economics"]["ndpr_price_low"]:.0f}/kg)')
    ax4.set_xlabel('Break-even NdPr price ($/kg oxide)\n'
                   '(sites left of dashed line = viable at current price)', fontsize=10)
    ax4.set_ylabel('NdPr metal P50 (estimated tonnes)', fontsize=11)
    ax4.tick_params(labelsize=9)
    ax4.set_title('D.  Break-even price vs. NdPr endowment — viable sites only\n'
                  '(color = data confidence; dashed = current $109/kg; dotted = 2024 trough $60/kg)', fontsize=9)
    ax4.grid(True, alpha=0.3)
    _d_legend_handles = ([mpatches.Patch(facecolor=c, edgecolor='black', label=l)
                          for l, c in conf_cmap.items()] +
                         [_vline_cur, _vline_low])
    ax4.legend(handles=_d_legend_handles, fontsize=7, loc='upper right')

    fig.text(0.5, 0.005,
             f'{cfg["study_area"]["name"]} REE Tailings Assessment — Au+REE Pipeline — EXPLORATION TARGET ONLY',
             ha='center', fontsize=7, color='gray', style='italic')

    save_fig(fig, out(cfg, 'figures', 'fig7_integrated_priority_map.png'))

    # ── Executive summary ─────────────────────────────────────────────────────
    top3    = merged_sorted.head(3)
    t4_idx  = task4_df.set_index('site_name')
    t5_idx  = task5_df.set_index('site')

    ec = cfg['economics']
    hdr = [
        "EXECUTIVE SUMMARY — TOP 3 PRIORITY SITES",
        f"{cfg['study_area']['name']} Placer Mine Tailings REE Assessment",
        "Intended for: field sampling decision-maker",
        "="*68,
        "",
        "BACKGROUND",
        "",
        f"This project evaluated sites in {cfg['study_area']['name']} for potential REE",
        "content (monazite-hosted NdPr). All grade estimates are proxy-based.",
        "",
        f"NdPr context: ~${ec['ndpr_price_central']:.0f}/kg current, ${ec['ndpr_price_low']:.0f}/kg 2024 trough.",
        f"Processor: {ec.get('processor_name','Energy Fuels White Mesa Mill')} ({ec.get('processor_location','Blanding, UT')})",
        "="*68,
    ]

    body = []
    for rank, (_, site) in enumerate(top3.iterrows(), 1):
        name = site['name']
        t4   = t4_idx.loc[name] if name in t4_idx.index else pd.Series()
        t5   = t5_idx.loc[name] if name in t5_idx.index else pd.Series()
        body += [
            "",
            f"SITE #{rank}: {name.upper()}",
            "="*60,
            f"  Coordinates:    {site.lat:.4f}°N, {site.lon:.4f}°W",
            f"  Combined score: {site.combined_score:.1f}",
            f"  Th source:      {site.get('th_source','N/A')}",
            f"  Mag high:       {'YES' if site.get('mag_high') else 'NO'}",
            f"  Source lith:    {site.get('source_lith_desc','N/A')}",
            f"  Est. tonnage:   {int(t4.get('tonnage_t', 0)):,} t" if len(t4) else "  Est. tonnage:   N/A",
            f"  NdPr grade:     ~{int(t4.get('ndpr_ppm', 0))} ppm (±50%)" if len(t4) else "  NdPr grade:     N/A",
            f"  NdPr metal:     {t4.get('ndpr_tonnes', 0):.0f} t" if len(t4) else "  NdPr metal:     N/A",
        ]
        if len(t5) > 0:
            body += [
                f"  Break-even:     ${t5.get('breakeven_$/kg', 0):.0f}/kg NdPr",
                f"  NPV (central):  ${t5.get('npv_central_$M', 0):.0f}M (undiscounted; ESTIMATE ONLY)",
            ]
        body += [
            "",
            "  NOTE: Exploration target only; not a resource estimate. No field",
            "  sampling conducted. Verify by auger program before any investment.",
        ]

    footer = [
        "",
        "="*68,
        "IMPORTANT DISCLAIMERS",
        "",
        "Exploration target only. Estimates have not been verified by in-situ",
        "sampling. Do not use for investment decisions, property transactions,",
        "or regulatory filings. Does not comply with NI 43-101 or JORC Code.",
        "All NdPr prices as of June 2026; may change substantially.",
        "="*68,
    ]

    with open(out(cfg, 'text', 'executive_summary_top3_sites.txt'), 'w') as f:
        f.write('\n'.join(hdr + body + footer))

    # ── Weight sensitivity (one-at-a-time ±50%) ──────────────────────────────
    _WEIGHT_KEYS  = ['th_source_monazite', 'magnetic_high', 'source_lith',
                     'coverage', 'ndpr_volume', 'au_pathfinder', 'y_xenotime', 'ml_probability']
    _SCORE_COLS   = ['score_th', 'score_mag', 'score_lith',
                     'score_conf', 'score_ndpr', 'score_au', 'score_y', 'score_ml']
    _DEFAULTS     = [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0]
    base_w        = [w.get(k, d) for k, d in zip(_WEIGHT_KEYS, _DEFAULTS)]

    # Collect rank for each site under every perturbation
    site_names   = list(merged_sorted['name'])
    all_ranks    = {n: [int(merged_sorted.loc[merged_sorted['name'] == n, 'rank'].iloc[0])]
                    for n in site_names}

    for wi in range(len(_WEIGHT_KEYS)):
        for factor in (0.5, 1.5):
            pw = list(base_w)
            pw[wi] = base_w[wi] * factor
            perturbed_score = sum(
                merged[sc].values * pw[i] for i, sc in enumerate(_SCORE_COLS)
            )
            order = np.argsort(perturbed_score)[::-1]
            for rank_pos, idx in enumerate(order, 1):
                all_ranks[merged.iloc[idx]['name']].append(rank_pos)

    sens_rows = []
    for _, row in merged_sorted.iterrows():
        ranks = all_ranks[row['name']]
        sens_rows.append({
            'site_name':   row['name'],
            'nominal_rank': int(row['rank']),
            'min_rank':    int(min(ranks)),
            'max_rank':    int(max(ranks)),
            'rank_range':  int(max(ranks) - min(ranks)),
            'stable_top3': bool(max(ranks) <= 3),
        })

    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(out(cfg, 'tables', 'integration_weight_sensitivity.csv'), index=False)

    top3_names   = list(merged_sorted.head(3)['name'])
    stable_top3  = [n for n in top3_names if sens_df.loc[sens_df['site_name'] == n, 'stable_top3'].iloc[0]]
    unstable     = [n for n in top3_names if n not in stable_top3]

    print("\nWEIGHT SENSITIVITY (one-at-a-time ±50%):")
    print(sens_df[['site_name','nominal_rank','min_rank','max_rank','rank_range']].to_string(index=False))
    if len(stable_top3) == 3:
        print(f"\n  TOP-3 RANK IS STABLE: {', '.join(top3_names)} remain in top-3 "
              f"across all weight perturbations.")
    elif stable_top3:
        print(f"\n  PARTIALLY STABLE: {', '.join(stable_top3)} remain top-3 in all "
              f"perturbations; {', '.join(unstable)} can fall outside top-3.")
    else:
        print(f"\n  NOTE: Top-3 ranking is sensitive to weight choice. "
              f"Nominal top-3: {', '.join(top3_names)}.")

    print("\nINTEGRATION COMPLETE")
    print(merged_sorted[['rank','name','combined_score','score_th','score_mag',
                          'score_lith','score_conf','score_ndpr','score_au',
                          'score_ml','ndpr_tonnes']].to_string(index=False))


if __name__ == '__main__':
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/ne_washington/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
