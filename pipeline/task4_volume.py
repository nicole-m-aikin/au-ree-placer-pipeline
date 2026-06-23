"""
Task 4: Volume estimation from lidar and historical topography.

Output:
  {outputs_dir}/tables/task4_volume_tonnage_summary.csv
  {outputs_dir}/figures/fig4_volume_estimation_example.png
  {outputs_dir}/text/task4_exploration_target_statement.txt
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings('ignore')

from pipeline.utils import WONG, setup_mpl, wgs_path, watermark, save_fig, ensure_outputs, out


def run(cfg):
    setup_mpl()
    ensure_outputs(cfg['outputs_dir'])

    geo = cfg['geochemistry']
    wgs_b_lat = (cfg['study_area'].get('wgs_bbox', cfg['study_area']['bbox'])['lat_min'],
                 cfg['study_area'].get('wgs_bbox', cfg['study_area']['bbox'])['lat_max'])
    wgs_b_lon = (cfg['study_area'].get('wgs_bbox', cfg['study_area']['bbox'])['lon_min'],
                 cfg['study_area'].get('wgs_bbox', cfg['study_area']['bbox'])['lon_max'])
    wgs_exclude = set(cfg.get('wgs', {}).get('exclude_sites', []))

    BULK_DENSITY_T_M3   = geo.get('bulk_density_t_m3', 1.5)
    TH_IN_MONAZITE_FRAC = geo.get('th_in_monazite_frac', 0.044)
    LREE_IN_MONAZITE    = geo.get('lree_in_monazite', 0.28)
    NDPR_IN_MONAZITE    = geo.get('ndpr_in_monazite', 0.09)
    STREAM_FACTOR       = geo.get('stream_to_insitu_factor', 5.0)
    REGIONAL_BG_TH      = geo.get('regional_bg_th', 8.0)

    np.random.seed(77)

    sites_gdf = gpd.read_file(out(cfg, 'geojson', 'task2_source_lithology_scored.geojson'))
    nure_df   = pd.read_csv(cfg['data']['nure_csv'])
    task1_df  = pd.read_csv(out(cfg, 'tables', 'task1_site_summary.csv'))

    th_lookup = task1_df.set_index('name')['th_value_ppm'].to_dict()
    for site in list(th_lookup.keys()):
        val = th_lookup[site]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            th_lookup[site] = REGIONAL_BG_TH

    # Build per-site dictionaries from config
    site_cfg = {s['name']: s for s in cfg['sites']}

    results = []
    for _, site in sites_gdf.iterrows():
        name = site['name']
        sc   = site_cfg.get(name, {})
        lidar_year    = sc.get('lidar_year')
        lidar_res_m   = sc.get('lidar_res_m')
        lidar_quality = sc.get('lidar_quality', 'UNKNOWN')
        topo_year       = sc.get('topo_year')
        topo_contour_ft = sc.get('topo_contour_ft')
        topo_available  = sc.get('topo_available', False)
        area_ha  = sc.get('area_ha', 5.0)
        depth_m  = sc.get('depth_m', 3.0)
        depth_unc = sc.get('depth_uncertainty_m', 1.5)

        area_m2    = area_ha * 10000
        vol_m3     = area_m2 * depth_m
        vol_unc_m3 = area_m2 * depth_unc

        tonnage = vol_m3 * BULK_DENSITY_T_M3

        th_stream = th_lookup.get(name, REGIONAL_BG_TH)
        th_insitu = th_stream * STREAM_FACTOR
        mnz_ppm  = th_insitu / TH_IN_MONAZITE_FRAC
        lree_ppm = mnz_ppm * LREE_IN_MONAZITE
        ndpr_ppm = mnz_ppm * NDPR_IN_MONAZITE

        # Monte Carlo: 2000-sample endowment chain per site
        N_MC = 2000
        depth_samp   = np.clip(np.random.normal(depth_m, depth_unc, N_MC), 0, None)
        grade_samp   = np.random.lognormal(np.log(max(ndpr_ppm, 1e-6)), 0.4, N_MC)
        tonnage_samp = area_m2 * depth_samp * BULK_DENSITY_T_M3
        endow_samp   = tonnage_samp * grade_samp / 1e6

        ndpr_t_p10 = float(np.percentile(endow_samp, 10))
        ndpr_t_p50 = float(np.percentile(endow_samp, 50))
        ndpr_t_p90 = float(np.percentile(endow_samp, 90))
        ton_p10    = float(np.percentile(tonnage_samp, 10))
        ton_p50    = float(np.percentile(tonnage_samp, 50))
        ton_p90    = float(np.percentile(tonnage_samp, 90))
        ndpr_t     = ndpr_t_p50  # P50 is the headline estimate

        if lidar_quality == 'HIGH' and topo_available and (topo_contour_ft or 99) <= 20:
            conf = 'HIGH'
        elif lidar_quality in ('HIGH', 'MEDIUM') and topo_available:
            conf = 'MEDIUM'
        else:
            conf = 'LOW'

        results.append({
            'site_name':       name,
            'lon':             site.lon,
            'lat':             site.lat,
            'area_ha':         area_ha,
            'depth_m':         depth_m,
            'vol_m3':          round(vol_m3),
            'vol_unc_m3':      round(vol_unc_m3),
            'tonnage_t':       round(tonnage),
            'tonnage_p10_t':   round(ton_p10),
            'tonnage_p50_t':   round(ton_p50),
            'tonnage_p90_t':   round(ton_p90),
            'th_stream_ppm':   round(th_stream, 1),
            'th_insitu_proxy': round(th_insitu, 1),
            'monazite_ppm':    round(mnz_ppm, 0),
            'lree_ppm':        round(lree_ppm, 0),
            'ndpr_ppm':        round(ndpr_ppm, 0),
            'ndpr_tonnes':     round(ndpr_t, 1),    # P50 — used by integration scoring
            'ndpr_t_p10':      round(ndpr_t_p10, 1),
            'ndpr_t_p50':      round(ndpr_t_p50, 1),
            'ndpr_t_p90':      round(ndpr_t_p90, 1),
            'lidar_year':      lidar_year,
            'lidar_res_m':     lidar_res_m,
            'topo_year':       topo_year,
            'topo_contour_ft': topo_contour_ft,
            'confidence':      conf,
            'source_lith_score': site.get('source_lith_score', None),
            'priority_score_t1': site.get('priority_score', None),
        })

    results_df = pd.DataFrame(results).sort_values('ndpr_tonnes', ascending=False)
    results_df.to_csv(out(cfg, 'tables', 'task4_volume_tonnage_summary.csv'), index=False)

    # ── Figure 4: illustrative volume estimation (top-volume site) ────────────
    # Use the first site from config as the illustration example (Oroville analog)
    example_site = results_df.iloc[0]

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f'Figure 4 — Volume Estimation: Lidar Surface vs. Historical Topography\n'
        f'Illustrative example — {example_site["site_name"]} '
        f'(largest tailings volume; not necessarily top-ranked by NdPr metal)',
        fontsize=12, fontweight='bold',
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35,
                           height_ratios=[1, 1, 0.9])

    x = np.linspace(0, 500, 50)
    y = np.linspace(0, 500, 50)
    X, Y = np.meshgrid(x, y)
    pre_mining = 290 + 0.02*X - 0.01*Y + 2*np.sin(X/80)*np.cos(Y/90)
    R2 = (X-250)**2 + (Y-250)**2
    current_surface = (pre_mining + 8*np.exp(-R2/30000) - 2*np.exp(-R2/70000)
                       + np.random.normal(0, 0.3, X.shape))
    diff = current_surface - pre_mining
    row_idx = 25

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.pcolormesh(X, Y, pre_mining, cmap='cividis', shading='auto')
    plt.colorbar(im1, ax=ax1, label='Elevation (m)', shrink=0.85)
    ax1.set_title(f'A.  Historical topo surface\n({example_site["topo_year"]} USGS 7.5-min quad, '
                  f'{example_site["topo_contour_ft"]}-ft contours)\nNorthings/Eastings relative to site SW corner', fontsize=8)
    ax1.set_xlabel('Easting (m)', fontsize=11); ax1.set_ylabel('Northing (m)', fontsize=11)
    ax1.tick_params(labelsize=9)
    cs1 = ax1.contour(X, Y, pre_mining, levels=8, colors='white', linewidths=0.5, alpha=0.7)
    ax1.clabel(cs1, inline=True, fontsize=6, fmt='%.0f m', colors='white')

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.pcolormesh(X, Y, current_surface, cmap='cividis', shading='auto')
    plt.colorbar(im2, ax=ax2, label='Elevation (m)', shrink=0.85)
    ax2.set_title(f'B.  Current lidar surface\n({example_site["lidar_year"]} acquisition, '
                  f'{example_site["lidar_res_m"]}m resolution)\nNorthings/Eastings relative to site SW corner', fontsize=8)
    ax2.set_xlabel('Easting (m)', fontsize=11)
    ax2.tick_params(labelsize=9)
    cs2 = ax2.contour(X, Y, current_surface, levels=8, colors='white', linewidths=0.5, alpha=0.7)
    ax2.clabel(cs2, inline=True, fontsize=6, fmt='%.0f m', colors='white')

    y_cross = y[row_idx]
    for ax_panel in [ax1, ax2]:
        ax_panel.axhline(y_cross, color='black', lw=1.0, ls='--', zorder=6)
        ax_panel.text(x[0] - 15, y_cross, "A", fontsize=8, fontweight='bold',
                      ha='right', va='center', color='black', clip_on=False)
        ax_panel.text(x[-1] + 15, y_cross, "A'", fontsize=8, fontweight='bold',
                      ha='left', va='center', color='black', clip_on=False)

    ax3 = fig.add_subplot(gs[0, 2])
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=10)
    im3 = ax3.pcolormesh(X, Y, diff, cmap='PuOr_r', shading='auto', norm=norm)
    cb3 = plt.colorbar(im3, ax=ax3, shrink=0.85)
    cb3.set_label('+ve = net elevation\ngain (m)', fontsize=8)
    ax3.set_title('C.  Lidar − historical surface\n(warm = net gain; cool = net loss)', fontsize=9)
    ax3.set_xlabel('Easting (m)', fontsize=11)
    ax3.contour(X, Y, diff, levels=[0], colors='black', linewidths=1.5)
    ax3.text(0.5, -0.15,
             'Positive values = elevation gain since mining\n'
             '(tailings + disturbance; not all recoverable)',
             transform=ax3.transAxes, fontsize=6.5, ha='center', color='#555555', style='italic')

    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.fill_between(x, pre_mining[row_idx], current_surface[row_idx],
                     where=current_surface[row_idx] > pre_mining[row_idx],
                     alpha=0.5, color=WONG['green'], label='Net elevation gain (tailings + disturbance)')
    ax4.fill_between(x, pre_mining[row_idx], current_surface[row_idx],
                     where=current_surface[row_idx] <= pre_mining[row_idx],
                     alpha=0.3, color=WONG['vermillion'], label='Net elevation loss (excavation/erosion)')
    ax4.plot(x, pre_mining[row_idx], color=WONG['blue'], ls='--', lw=2,
             label=f'Historical surface ({example_site["topo_year"]})')
    ax4.plot(x, current_surface[row_idx], color=WONG['orange'], lw=2,
             label=f'Lidar surface ({example_site["lidar_year"]})')
    ax4.set_xlabel('Distance along A–A\' profile (m)', fontsize=11)
    ax4.set_ylabel('Elevation (m)', fontsize=11)
    ax4.tick_params(labelsize=9)
    ax4.set_title('D.  E–W cross-section through tailings center', fontsize=9)
    ax4.legend(fontsize=8, loc='upper left'); ax4.grid(True, alpha=0.3)
    ax4.annotate(f'Max depth: {diff[row_idx].max():.1f} m',
                 xy=(x[diff[row_idx].argmax()], current_surface[row_idx][diff[row_idx].argmax()]),
                 xytext=(320, 297), arrowprops=dict(arrowstyle='->', color='black'), fontsize=8)

    ax5 = fig.add_subplot(gs[1, 2])
    plot_df = results_df.sort_values('ndpr_t_p50', ascending=True)
    y_pos   = np.arange(len(plot_df))
    xerr_lo = (plot_df['ndpr_t_p50'] - plot_df['ndpr_t_p10']).values
    xerr_hi = (plot_df['ndpr_t_p90'] - plot_df['ndpr_t_p50']).values
    ax5.errorbar(plot_df['ndpr_t_p50'].values, y_pos,
                 xerr=[xerr_lo, xerr_hi],
                 fmt='o', color=WONG['orange'], ecolor=WONG['blue'],
                 capsize=4, elinewidth=1.5, markersize=5)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels([' '.join(n.split()[:2]) for n in plot_df['site_name']], fontsize=7)
    ax5.set_xlabel('NdPr metal (tonnes)', fontsize=9)
    ax5.set_title('E.  NdPr endowment P10 – P50 – P90\n(Monte Carlo, 2 000 samples/site)', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='x')
    ax5.tick_params(labelsize=7)

    # ── Panel F: WGS endowment table ─────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    _xl = wgs_path(cfg)
    _f_title = 'F.  WGS OFR 2026-02: Field-Measured Mine Waste Tonnage\n(Earth MRI, 2024 field campaign)'
    wgs_end_df = None
    try:
        _end = pd.read_excel(_xl, sheet_name='Endowment Calculations', header=0)
        for _col in ['Latitude', 'Longitude', 'Tonnage (metric tons)', 'Average Conc TREE (ppm)',
                     'Endowment TREE (kg)', 'Area (m^2)', 'Est. Volume (m^3)']:
            if _col in _end.columns:
                _end[_col] = pd.to_numeric(_end[_col], errors='coerce')
        if 'Latitude' in _end.columns and 'Longitude' in _end.columns:
            _end = _end[_end['Latitude'].between(*wgs_b_lat) & _end['Longitude'].between(*wgs_b_lon)]
        _mine_col = next((c for c in ['Mine Name','Mine_Name'] if c in _end.columns), _end.columns[0])
        _end = _end[~_end[_mine_col].isin(wgs_exclude)]
        wgs_end_df = _end.reset_index(drop=True)
        print(f"WGS Endowment data loaded: {len(wgs_end_df)} rows")
    except Exception as _e:
        print(f"WGS Endowment data not available for Panel F: {_e}")

    if wgs_end_df is not None and len(wgs_end_df) > 0:
        _mine_col = next((c for c in ['Mine Name','Mine_Name'] if c in wgs_end_df.columns),
                         wgs_end_df.columns[0])
        _agg = wgs_end_df.groupby(_mine_col).agg(
            Tonnage_t=('Tonnage (metric tons)', 'sum'),
            TREE_ppm=('Average Conc TREE (ppm)', 'mean'),
            TREE_kg=('Endowment TREE (kg)', 'sum'),
            Lat=('Latitude', 'mean'),
            Lon=('Longitude', 'mean'),
        ).reset_index()

        _disp = _agg[[_mine_col, 'Lat', 'Lon', 'Tonnage_t', 'TREE_ppm', 'TREE_kg']].copy()
        _disp.columns = ['Mine Site', 'Lat (°N)', 'Lon (°E)',
                         'Tonnage (metric t)', 'TREE avg (ppm)', 'TREE endow. (kg)']
        _disp['Tonnage (metric t)'] = _disp['Tonnage (metric t)'].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else 'N/A')
        _disp['TREE avg (ppm)']     = _disp['TREE avg (ppm)'].apply(lambda v: f"{v:.1f}" if pd.notna(v) else 'N/A')
        _disp['TREE endow. (kg)']   = _disp['TREE endow. (kg)'].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else 'N/A')
        _disp['Lat (°N)'] = _disp['Lat (°N)'].apply(lambda v: f"{v:.4f}" if pd.notna(v) else 'N/A')
        _disp['Lon (°E)'] = _disp['Lon (°E)'].apply(lambda v: f"{v:.4f}" if pd.notna(v) else 'N/A')

        _tbl = ax6.table(cellText=_disp.values.tolist(), colLabels=list(_disp.columns),
                         cellLoc='center', loc='center', bbox=[0.0, 0.12, 1.0, 0.80])
        _tbl.auto_set_font_size(False); _tbl.set_fontsize(8)
        _tbl.auto_set_column_width(range(len(_disp.columns)))
        for _j in range(len(_disp.columns)):
            _tbl[(0, _j)].set_facecolor(WONG['green'])
            _tbl[(0, _j)].set_text_props(color='white', fontweight='bold')
        ax6.text(0.5, 0.04,
                 f"WGS field-measured hard-rock mine waste ({len(_agg)} sites in study area). "
                 f"Pipeline estimates use MRDS placer Au sites ({len(results_df)} sites). "
                 "No direct site overlap: WGS = hard-rock mine waste; Pipeline = placer Au operations.",
                 transform=ax6.transAxes, ha='center', va='bottom',
                 fontsize=7.5, color='#444444', style='italic')
    else:
        ax6.text(0.5, 0.5,
                 'WGS OFR 2026-02 endowment data not found.\n'
                 'Expected sheet "Endowment Calculations" in the Excel supplement.',
                 ha='center', va='center', transform=ax6.transAxes, fontsize=9, color='gray')

    ax6.set_title(_f_title, fontsize=9, fontweight='bold', loc='left', pad=8)

    watermark(fig, cfg)
    save_fig(fig, out(cfg, 'figures', 'fig4_volume_estimation.png'))

    # ── Exploration target statement ──────────────────────────────────────────
    top3 = results_df.head(3)
    total_lo = top3['ndpr_t_p10'].sum()
    total_hi = top3['ndpr_t_p90'].sum()
    total_c  = top3['ndpr_tonnes'].sum()

    expl_target = f"""
EXPLORATION TARGET STATEMENT
{cfg['study_area']['name']} Placer Mine Tailings REE Project
(THIS IS NOT A MINERAL RESOURCE ESTIMATE)
================================================================

BASIS OF ESTIMATE:
  - Volume: aerial extent from MRDS historical production records
  - Grade proxy: NURE stream sediment Th values upscaled by {STREAM_FACTOR}x dilution factor
  - Th→monazite conversion: {TH_IN_MONAZITE_FRAC*100:.1f}% Th in metamorphic monazite
  - Monazite→NdPr: {NDPR_IN_MONAZITE*100:.0f}% NdPr by mass
  - CAUTION: grade estimates carry ±50% uncertainty (log-normal, σ=0.4)
  - Monte Carlo: 2 000 samples per site; depth ~ N(μ, σ), grade ~ LogNormal

COMBINED TOP-3 EXPLORATION TARGET (Monte Carlo P10 / P50 / P90):
  NdPr metal P10: {total_lo:.0f} t
  NdPr metal P50: {total_c:.1f} t  (headline estimate)
  NdPr metal P90: {total_hi:.0f} t

IMPORTANT DISCLAIMER:
  This is an exploration target only. Do not use for investment decisions
  without completion of a formal resource estimation study.
"""
    tgt_path = out(cfg, 'text', 'task4_exploration_target_statement.txt')
    with open(tgt_path, 'w') as f:
        f.write(expl_target)

    print(f"\nTask 4 Results: {len(results_df)} sites, total NdPr P50 = {results_df['ndpr_tonnes'].sum():.0f} t")
    print(results_df[['site_name','tonnage_t','ndpr_ppm','ndpr_t_p10','ndpr_t_p50','ndpr_t_p90','confidence']].head(5).to_string(index=False))


if __name__ == '__main__':
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/ne_washington/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
