"""
Task 1: Co-placer mineral characterization — magnetite and ilmenite
Aeromagnetic anomaly cross-referenced with NURE Th anomalies.

Output:
  {outputs_dir}/geojson/task1_multicommodity_targets.geojson
  {outputs_dir}/figures/fig1_coplacer_magnetic_th_overlay.png
  {outputs_dir}/figures/fig1b_airborne_eth_vs_nure_th.png   (only if eTh TIFF exists)
  {outputs_dir}/tables/task1_site_summary.csv
  {outputs_dir}/tables/task1_radiometric_concordance.csv    (only if eTh TIFF exists)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

from pipeline.utils import (WONG, setup_mpl, watermark, save_fig, ensure_outputs, out,
                             map_extent, hillshade, north_arrow, scale_bar,
                             canada_border, locator_inset, clip_gdf_to_map,
                             load_radiometric_at_points, load_raster_window,
                             radiometric_tif_path, linear_mean_2sd,
                             classify_rad_concordance,
                             RAD_AGREE_BOTH, RAD_AGREE_STREAM, RAD_AGREE_AIR,
                             RAD_AGREE_NONE,
                             MAP_W, MAP_H, _FIG_LM, _FIG_RM, _FIG_TM, _FIG_BM,
                             _FIG_HGAP, _FIG_CW, _FIG_CG, _ax_rect)


# Old 0.25° window (~28 km) plus max-Th-in-window assigned Hunters a monazite
# grab 26 km SSE. Default is one NURE spacing; nearest sample, not the peak.
_DEFAULT_JOIN_RADIUS_DEG = 0.10
_KM_PER_DEG = 111.0


def site_join_radius_deg(cfg):
    return float(cfg.get('geochemistry', {}).get(
        'site_join_radius_deg', _DEFAULT_JOIN_RADIUS_DEG))


def nearest_nure_row(lon, lat, gdf, radius_deg):
    """Closest row in gdf within radius_deg, or (None, nan). Uses lon/lat columns."""
    if gdf is None or len(gdf) == 0:
        return None, np.nan
    dist = np.sqrt(
        (gdf['lon'].astype(float) - lon) ** 2 + (gdf['lat'].astype(float) - lat) ** 2
    )
    near = dist <= radius_deg
    if not near.any():
        return None, np.nan
    idx = dist[near].idxmin()
    return gdf.loc[idx], float(dist.loc[idx]) * _KM_PER_DEG


_AGREE_COLORS = {
    RAD_AGREE_BOTH:   WONG['blue'],
    RAD_AGREE_STREAM: WONG['orange'],
    RAD_AGREE_AIR:    WONG['vermillion'],
    RAD_AGREE_NONE:   '#BBBBBB',
}
_AGREE_LABELS = {
    RAD_AGREE_BOTH:   'Both high (concordant)',
    RAD_AGREE_STREAM: 'Stream Th only (transport / cover)',
    RAD_AGREE_AIR:    'Airborne eTh only (cover / gap)',
    RAD_AGREE_NONE:   'Neither (background)',
}


def attach_radiometrics(cfg, sites_gdf, nure_gdf):
    """Sample aerial K / eTh / eU at sites and NURE points. No-op if TIFFs absent.

    Returns (sites_gdf, nure_gdf, eth_thresh_or_None). Never invents a grid.
    """
    eth_path = radiometric_tif_path(cfg, 'eth')
    if eth_path is None:
        return sites_gdf, nure_gdf, None

    site_rad = load_radiometric_at_points(cfg, sites_gdf['lon'].values, sites_gdf['lat'].values)
    for col in site_rad.columns:
        sites_gdf[col] = site_rad[col].values

    nure_rad = load_radiometric_at_points(cfg, nure_gdf['lon'].values, nure_gdf['lat'].values)
    for col in nure_rad.columns:
        nure_gdf[col] = nure_rad[col].values

    xmin, xmax, ymin, ymax = map_extent(cfg)
    grid, _ = load_raster_window(eth_path, (xmin, xmax, ymin, ymax))
    eth_thresh = linear_mean_2sd(grid) if grid is not None else linear_mean_2sd(nure_gdf['rad_eTh'])
    if eth_thresh is None:
        print("  Aerial eTh TIFF present but no finite positives — skipping overlay")
        return sites_gdf, nure_gdf, None

    sites_gdf['rad_eTh_high'] = sites_gdf['rad_eTh'] >= eth_thresh
    sites_gdf['rad_th_agree'] = classify_rad_concordance(
        sites_gdf['th_near'], sites_gdf['rad_eTh_high'].where(sites_gdf['rad_eTh'].notna())
    )
    nure_gdf['rad_eTh_high'] = nure_gdf['rad_eTh'] >= eth_thresh
    nure_th_high = nure_gdf['th_anomaly'].astype(bool)
    nure_gdf['rad_th_agree'] = classify_rad_concordance(
        nure_th_high, nure_gdf['rad_eTh_high'].where(nure_gdf['rad_eTh'].notna())
    )
    n_ok = int(nure_gdf['rad_eTh'].notna().sum())
    print(f"  Aerial radiometrics sampled at {int(sites_gdf['rad_eTh'].notna().sum())} sites "
          f"and {n_ok} NURE points  (eTh high ≥ {eth_thresh:.1f} ppm)")
    return sites_gdf, nure_gdf, eth_thresh


def _plot_eth_overlay(cfg, nure_gdf, sites_gdf, eth_thresh):
    """Fig 1b: airborne eTh grid vs stream-sediment Th. No-op if the TIFF is gone."""
    eth_path = radiometric_tif_path(cfg, 'eth')
    if eth_path is None or eth_thresh is None:
        return

    xmin, xmax, ymin, ymax = map_extent(cfg)
    grid, extent = load_raster_window(eth_path, (xmin, xmax, ymin, ymax))
    if grid is None:
        return

    finite = grid[np.isfinite(grid) & (grid > 0)]
    vmin = float(np.percentile(finite, 2)) if finite.size else 0.0
    vmax = float(np.percentile(finite, 98)) if finite.size else 1.0

    figW = _FIG_LM + MAP_W + _FIG_CG + _FIG_CW + _FIG_HGAP + MAP_W + _FIG_RM
    figH = _FIG_TM + MAP_H + _FIG_BM
    fig = plt.figure(figsize=(figW, figH))
    fig.suptitle(
        f'Figure 1b — Airborne eTh vs NURE Stream-Sediment Th\n'
        f'Same program, different physics: aircraft γ-ray (top ~30 cm) × lab chemistry — '
        f'{cfg["study_area"]["name"]}',
        fontsize=12, fontweight='bold',
    )

    ax  = fig.add_axes(_ax_rect(_FIG_LM,                                        _FIG_BM, MAP_W,   MAP_H, figW, figH))
    cax = fig.add_axes(_ax_rect(_FIG_LM + MAP_W + _FIG_CG,                      _FIG_BM, _FIG_CW, MAP_H, figW, figH))
    ax2 = fig.add_axes(_ax_rect(_FIG_LM + MAP_W + _FIG_CG + _FIG_CW + _FIG_HGAP, _FIG_BM, MAP_W,   MAP_H, figW, figH))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('auto')
    hillshade(cfg, ax, alpha=0.15)
    im = ax.imshow(grid, extent=extent, origin='upper', cmap='cividis',
                   vmin=vmin, vmax=vmax, alpha=0.88, zorder=1, aspect='auto')
    cbar = fig.colorbar(im, cax=cax)
    cax.yaxis.set_ticks_position('left')
    cax.yaxis.set_label_position('left')
    cbar.set_label('Airborne eTh (ppm)', fontsize=8, rotation=90, labelpad=3)

    # Contour the eTh-high threshold — gray on cividis, not white
    try:
        lons = np.linspace(extent[0], extent[1], grid.shape[1])
        lats = np.linspace(extent[3], extent[2], grid.shape[0])
        ax.contour(lons, lats, grid, levels=[eth_thresh], colors='gray',
                   linewidths=1.4, linestyles='--', zorder=2)
    except Exception:
        pass

    nure_plot = clip_gdf_to_map(nure_gdf, cfg)
    bg = nure_plot[~nure_plot['th_anomaly'].astype(bool)]
    an = nure_plot[nure_plot['th_anomaly'].astype(bool)]
    if len(bg):
        ax.scatter(bg['lon'], bg['lat'], s=4, c='#444444', alpha=0.25,
                   linewidths=0, zorder=3, label='NURE (background)')
    if len(an):
        ax.scatter(an['lon'], an['lat'], s=28, marker='^', c=WONG['orange'],
                   edgecolors='black', linewidths=0.4, zorder=4,
                   label='NURE Th anomaly (stream sediment)')

    sites_plot = clip_gdf_to_map(sites_gdf, cfg)
    ax.scatter(sites_plot['lon'], sites_plot['lat'], s=55, marker='o',
               facecolors='white', edgecolors='black', linewidths=0.7, zorder=5,
               label='Mine site')
    _placed = []
    for _, row in sites_plot.iterrows():
        oy = 4
        for (plon, plat) in _placed:
            if abs(row.lon - plon) < 0.15 and abs(row.lat - plat) < 0.15:
                oy += 10
        _placed.append((row.lon, row.lat))
        ax.annotate(row['name'].split()[0], (row.lon, row.lat),
                    xytext=(4, oy), textcoords='offset points', fontsize=6.5,
                    color='black', clip_on=True,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    canada_border(ax, cfg)
    north_arrow(ax)
    scale_bar(ax, cfg)
    locator_inset(fig, ax, cfg)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.tick_params(labelsize=9)
    ax.set_title('A.  Airborne eTh + stream-sediment Th anomalies\n'
                 f'(dashed contour = eTh mean+2SD = {eth_thresh:.1f} ppm)',
                 fontsize=9)
    ax.grid(True, alpha=0.2, color='gray')
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.88)

    # Panel B — point-level concordance scatter (NURE lab Th vs airborne eTh)
    valid = nure_gdf.dropna(subset=['Th', 'rad_eTh']).copy()
    valid = valid[(valid['Th'] > 0) & (valid['rad_eTh'] > 0)]
    if len(valid):
        colors = valid['rad_th_agree'].map(_AGREE_COLORS).fillna('#BBBBBB')
        ax2.scatter(valid['rad_eTh'], valid['Th'], c=colors, s=18,
                    edgecolors='black', linewidths=0.25, alpha=0.85, zorder=3)
        from scipy.stats import spearmanr
        rho, pval = spearmanr(valid['rad_eTh'], valid['Th'])
        ax2.text(0.04, 0.96,
                 f'Spearman ρ = {rho:.2f}\n(n = {len(valid)})',
                 transform=ax2.transAxes, ha='left', va='top', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='gray', alpha=0.9))
        print(f"  NURE Th vs airborne eTh: Spearman ρ={rho:.3f}  p={pval:.3g}  n={len(valid)}")
        counts = valid['rad_th_agree'].value_counts()
        for cls, n in counts.items():
            print(f"    {cls}: {n}")

    if len(valid):
        xmax_s = max(valid['rad_eTh'].max() * 1.08, eth_thresh * 1.15)
        ymax_s = valid['Th'].max() * 1.08
        ax2.set_xlim(0, xmax_s)
        ax2.set_ylim(0, ymax_s)
    ax2.axvline(eth_thresh, color='gray', ls='--', lw=1.1, zorder=2)
    # Stream-sediment Th anomaly threshold from the classified flag, if recoverable
    th_anom_vals = nure_gdf.loc[nure_gdf['th_anomaly'].astype(bool), 'Th']
    if th_anom_vals.notna().any():
        th_cut = float(th_anom_vals.min())
        ax2.axhline(th_cut, color=WONG['orange'], ls=':', lw=1.1, zorder=2)
    legend_p = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_AGREE_COLORS[k],
               markeredgecolor='black', markersize=8, label=_AGREE_LABELS[k])
        for k in (RAD_AGREE_BOTH, RAD_AGREE_STREAM, RAD_AGREE_AIR, RAD_AGREE_NONE)
    ]
    ax2.legend(handles=legend_p, loc='lower right', fontsize=7.5, framealpha=0.88)
    ax2.set_xlabel('Airborne eTh (ppm)', fontsize=11)
    ax2.set_ylabel('NURE stream-sediment Th (ppm)', fontsize=11)
    ax2.tick_params(labelsize=9)
    ax2.set_title('B.  Concordance at NURE sample sites\n'
                  '(agreement = ground truth; disagreement = cover, transport, or gap)',
                  fontsize=9)
    ax2.grid(True, alpha=0.3)

    watermark(fig, cfg)
    save_fig(fig, out(cfg, 'figures', 'fig1b_airborne_eth_vs_nure_th.png'))


def run(cfg):
    setup_mpl()
    ensure_outputs(cfg['outputs_dir'])

    b = cfg['study_area']['bbox']
    LON_MIN, LON_MAX = b['lon_min'], b['lon_max']
    LAT_MIN, LAT_MAX = b['lat_min'], b['lat_max']
    GRID_RES = 0.02

    # Use padded map_extent so the grid fills the full axes (no empty border)
    _gxmin, _gxmax, _gymin, _gymax = map_extent(cfg)
    lons_grid = np.arange(_gxmin, _gxmax + GRID_RES, GRID_RES)
    lats_grid = np.arange(_gymin, _gymax + GRID_RES, GRID_RES)
    LON_G, LAT_G = np.meshgrid(lons_grid, lats_grid)

    np.random.seed(123)
    mag_grid = np.random.normal(0, 15, LON_G.shape)

    for ac in cfg.get('aeromagnetic_anomaly_centers', []):
        dist2 = ((LON_G - ac['lon'])**2 + (LAT_G - ac['lat'])**2) / ac['width']**2
        mag_grid += ac['amplitude'] * np.exp(-dist2)

    mag_threshold = mag_grid.mean() + 2 * mag_grid.std()
    print(f"Magnetic anomaly threshold (mean+2SD): {mag_threshold:.1f} nT")

    sites = cfg['sites']
    sites_df = pd.DataFrame(sites)
    sites_gdf = gpd.GeoDataFrame(
        sites_df,
        geometry=[Point(s['lon'], s['lat']) for s in sites],
        crs='EPSG:4326',
    )

    # Load NURE Th classified output from task3
    nure_gdf = gpd.read_file(out(cfg, 'geojson', 'nure_classified_th_sources.geojson'))
    nure_anomaly = nure_gdf[nure_gdf['th_anomaly'].astype(bool)].copy()

    # Y anomaly threshold — xenotime (YPO4) proxy for HREE potential.
    # Y and Th are spatially anti-correlated in this dataset (r≈-0.66), so Y flags
    # an entirely different set of sites that the Th-based score is blind to.
    y_vals = pd.to_numeric(nure_gdf['Y'], errors='coerce')
    y_vals = y_vals[y_vals > 0]
    y_threshold = y_vals.mean() + 2 * y_vals.std()
    nure_gdf['y_anomaly'] = pd.to_numeric(nure_gdf['Y'], errors='coerce') > y_threshold
    nure_y_anomaly = nure_gdf[nure_gdf['y_anomaly'].fillna(False)].copy()
    print(f"Y anomaly threshold (mean+2SD): {y_threshold:.1f} ppm  "
          f"({len(nure_y_anomaly)} anomalous samples)")

    def get_mag_at_point(lon, lat):
        # Grid was built from map_extent (padded) origins _gymin/_gxmin, not raw LAT_MIN/LON_MIN.
        # Using the wrong origin shifted every sample ~0.08° SW of each site's true position.
        i = int((lat - _gymin) / GRID_RES)
        j = int((lon - _gxmin) / GRID_RES)
        i = np.clip(i, 0, mag_grid.shape[0] - 1)
        j = np.clip(j, 0, mag_grid.shape[1] - 1)
        return mag_grid[i, j]

    sites_gdf['mag_anomaly_nT'] = sites_gdf.apply(
        lambda r: get_mag_at_point(r.lon, r.lat), axis=1)
    sites_gdf['mag_high'] = sites_gdf['mag_anomaly_nT'] >= mag_threshold

    radius = site_join_radius_deg(cfg)
    print(f"  Site–NURE join radius: {radius:.2f}° (~{radius * _KM_PER_DEG:.0f} km); "
          f"nearest sample, not max-Th in window")

    th_vals, th_srcs, th_near = [], [], []
    th_local, th_ids, th_dists = [], [], []
    y_vals_site, y_near = [], []
    for _, row in sites_gdf.iterrows():
        anom, d_an = nearest_nure_row(row.lon, row.lat, nure_anomaly, radius)
        local, d_loc = nearest_nure_row(row.lon, row.lat, nure_gdf, radius)
        yrow, _ = nearest_nure_row(row.lon, row.lat, nure_y_anomaly, radius)

        if anom is not None:
            th_vals.append(float(anom['Th']) if pd.notna(anom.get('Th')) else None)
            th_srcs.append(anom.get('th_source'))
            th_near.append(True)
            th_ids.append(anom.get('lab_id'))
            th_dists.append(d_an)
        else:
            th_vals.append(None)
            th_srcs.append(None)
            th_near.append(False)
            th_ids.append(local.get('lab_id') if local is not None else None)
            th_dists.append(d_loc if local is not None else None)

        if local is not None and pd.notna(local.get('Th')):
            th_local.append(float(local['Th']))
        else:
            th_local.append(None)

        if yrow is not None and pd.notna(yrow.get('Y')):
            y_vals_site.append(float(yrow['Y']))
            y_near.append(True)
        else:
            y_vals_site.append(None)
            y_near.append(False)

    sites_gdf['th_value_ppm'] = th_vals
    sites_gdf['th_source']    = th_srcs
    sites_gdf['th_near']      = th_near
    sites_gdf['th_local_ppm'] = th_local
    sites_gdf['th_assign_lab_id'] = th_ids
    sites_gdf['th_assign_dist_km'] = th_dists
    sites_gdf['y_value_ppm']  = y_vals_site
    # y_near is a PARALLEL indicator — does not roll into the 0-3 LREE priority_score
    # so that downstream integration weighting is unchanged.
    sites_gdf['y_near']       = y_near

    sites_gdf['multicommodity_target'] = sites_gdf['th_near'] & sites_gdf['mag_high']
    sites_gdf['priority_score'] = (
        sites_gdf['th_near'].astype(int) +
        sites_gdf['mag_high'].astype(int) +
        (sites_gdf['th_source'] == 'MONAZITE').astype(int)
    )

    site_legend = [
        Line2D([0],[0], marker='*', color='w', markerfacecolor=WONG['blue'],
               markeredgecolor='black', markersize=14, label='Multi-commodity target'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor=WONG['blue'],
               markeredgecolor='black', markersize=9, label='Magnetic high'),
        Line2D([0],[0], marker='^', color='w', markerfacecolor=WONG['orange'],
               markeredgecolor='black', markersize=9, label='Th anomaly (NURE; mixed/unclear)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=9, label='No anomaly'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor=WONG['green'], markeredgewidth=2, markersize=13,
               label='Y anomaly (xenotime/HREE proxy)'),
    ]

    label_offsets = {k: tuple(v) for k, v in
                     cfg.get('task1_label_offsets', {}).items()}

    # Layout: [map_A | cbar | map_B] — all axes sized to exact physical inches.
    figW = _FIG_LM + MAP_W + _FIG_CG + _FIG_CW + _FIG_HGAP + MAP_W + _FIG_RM
    figH = _FIG_TM + MAP_H + _FIG_BM

    fig = plt.figure(figsize=(figW, figH))
    fig.suptitle(
        f'Figure 1 — Co-placer Mineral Targets: Aeromagnetic Anomalies × NURE Th Anomalies\n'
        f'Mineral Systems: Source mineralogy (co-placer indicators) — {cfg["study_area"]["name"]}',
        fontsize=12, fontweight='bold',
    )

    ax  = fig.add_axes(_ax_rect(_FIG_LM,                                        _FIG_BM, MAP_W,   MAP_H, figW, figH))
    cax = fig.add_axes(_ax_rect(_FIG_LM + MAP_W + _FIG_CG,                      _FIG_BM, _FIG_CW, MAP_H, figW, figH))
    ax2 = fig.add_axes(_ax_rect(_FIG_LM + MAP_W + _FIG_CG + _FIG_CW + _FIG_HGAP, _FIG_BM, MAP_W,   MAP_H, figW, figH))

    xmin_1, xmax_1, ymin_1, ymax_1 = map_extent(cfg)
    ax.set_xlim(xmin_1, xmax_1)
    ax.set_ylim(ymin_1, ymax_1)
    ax.set_aspect('auto')
    hillshade(cfg, ax, alpha=0.15)
    im = ax.pcolormesh(LON_G, LAT_G, mag_grid, cmap='RdBu_r', shading='auto',
                       vmin=-200, vmax=400)
    cbar_a = fig.colorbar(im, cax=cax)
    cax.yaxis.set_ticks_position('left')    # labels face Panel A, not Panel B
    cax.yaxis.set_label_position('left')
    cbar_a.set_label('Total magnetic intensity anomaly (nT)', fontsize=8, rotation=90, labelpad=3)

    top_site_name = cfg.get('task1_highlight_site') or None

    for _, row in sites_gdf.iterrows():
        name = row['name']
        if top_site_name and name == top_site_name:
            marker, color, size, zorder = '*', WONG['blue'], 220, 6
        elif row['multicommodity_target']:
            marker, color, size, zorder = 'D', WONG['blue'], 120, 5
        elif row['th_near']:
            color = WONG['orange']
            marker, size, zorder = '^', 80, 4
        elif row['mag_high']:
            marker, color, size, zorder = 'D', WONG['blue'], 80, 4
        else:
            marker, color, size, zorder = 'o', 'white', 50, 3
        ax.scatter(row.lon, row.lat, c=color, s=size, marker=marker,
                   edgecolors='black', linewidths=0.5, zorder=zorder)
        if row['y_near']:
            ax.scatter(row.lon, row.lat, s=size * 2.2, marker='o',
                       facecolors='none', edgecolors=WONG['green'],
                       linewidths=1.8, zorder=zorder + 1)
        ax.annotate(row['name'].split()[0], (row.lon, row.lat),
                    xytext=(4, 4), textcoords='offset points', fontsize=6.5, color='black',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    ax.contour(LON_G, LAT_G, mag_grid, levels=[mag_threshold], colors='white',
               linewidths=2.0, linestyles='--')
    ax.set_xlabel('Longitude', fontsize=11); ax.set_ylabel('Latitude', fontsize=11)
    ax.tick_params(labelsize=9)
    ax.set_title('A.  Aeromagnetic anomaly map + mine sites\n'
                 '(dashed contour = mean+2SD threshold; ★ = multicommodity target)', fontsize=9)
    canada_border(ax, cfg)
    north_arrow(ax)
    scale_bar(ax, cfg)
    locator_inset(fig, ax, cfg)
    ax.grid(True, alpha=0.2, color='gray')
    ax.legend(handles=site_legend, loc='upper left', fontsize=8, framealpha=0.88)

    colors_map  = {0: '#CCCCCC', 1: WONG['yellow'], 2: WONG['orange'], 3: WONG['blue']}
    color_label = {0: 'No anomaly (0)', 1: 'Single anomaly (1)',
                   2: 'Dual anomaly (2)', 3: 'Triple (highest priority, 3)'}

    ax2.set_facecolor('#e8f4f8')
    ax2.set_xlim(xmin_1, xmax_1)
    ax2.set_ylim(ymin_1, ymax_1)
    ax2.set_aspect('auto')

    _placed_labels = []   # (lon, lat) of already-annotated sites for collision detection
    for _, row in sites_gdf.iterrows():
        name  = row['name']
        score = int(row['priority_score'])
        color  = WONG['blue'] if (top_site_name and name == top_site_name) else colors_map.get(score, '#CCCCCC')
        marker = '*' if (top_site_name and name == top_site_name) else 'o'
        size   = 250 if (top_site_name and name == top_site_name) else 150
        ax2.scatter(row.lon, row.lat, c=color, s=size, marker=marker,
                    edgecolors='black', linewidths=0.7, zorder=4)
        if row['y_near']:
            # Green ring = Y anomaly (xenotime/HREE). Plotted separately from the
            # 0-3 LREE score so the two signals remain visually and analytically distinct.
            ax2.scatter(row.lon, row.lat, s=size * 2.5, marker='o',
                        facecolors='none', edgecolors=WONG['green'],
                        linewidths=2.2, zorder=5)
        ox, oy = label_offsets.get(name, (5, 3))
        # Stagger labels that are within 0.15° of an already-placed label
        for (pl_lon, pl_lat) in _placed_labels:
            if abs(row.lon - pl_lon) < 0.15 and abs(row.lat - pl_lat) < 0.15:
                oy += 10
        _placed_labels.append((row.lon, row.lat))
        ax2.annotate(row['name'], (row.lon, row.lat),
                     xytext=(ox, oy), textcoords='offset points', fontsize=7,
                     clip_on=True)

    canada_border(ax2, cfg)
    north_arrow(ax2)
    scale_bar(ax2, cfg)
    locator_inset(fig, ax2, cfg)
    ax2.set_xlabel('Longitude', fontsize=11)
    ax2.set_ylabel('')
    ax2.set_yticklabels([])
    ax2.tick_params(labelsize=9)
    ax2.set_title('B.  Multi-criterion priority score\n'
                  '(Th anomaly + Mag high + Monazite classification; green ring = Y/HREE)\n'
                  'Provisional Task 1 rank only — see Fig 7 for full integrated ranking',
                  fontsize=9)
    ax2.grid(True, alpha=0.3)
    legend_p = [mpatches.Patch(facecolor=c, edgecolor='black', label=l)
                for c, l in zip(colors_map.values(), color_label.values())]
    legend_p.append(
        Line2D([0],[0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor=WONG['green'], markeredgewidth=2, markersize=13,
               label='Y anomaly (xenotime/HREE; parallel)')
    )
    ax2.legend(handles=legend_p, loc='upper left', fontsize=8, framealpha=0.88)

    watermark(fig, cfg)
    save_fig(fig, out(cfg, 'figures', 'fig1_coplacer_magnetic_th_overlay.png'))

    sites_gdf, nure_gdf, eth_thresh = attach_radiometrics(cfg, sites_gdf, nure_gdf)
    _plot_eth_overlay(cfg, nure_gdf, sites_gdf, eth_thresh)

    sites_gdf.to_file(out(cfg, 'geojson', 'task1_multicommodity_targets.geojson'), driver='GeoJSON')

    site_cols = ['name','commodity','lon','lat','mag_anomaly_nT','mag_high',
                 'th_value_ppm','th_source','th_local_ppm','th_assign_lab_id',
                 'th_assign_dist_km','multicommodity_target','priority_score',
                 'y_near','y_value_ppm']
    rad_cols = [c for c in ('rad_K','rad_eTh','rad_eU','rad_eU_eTh','rad_K_eTh',
                            'rad_eTh_high','rad_th_agree') if c in sites_gdf.columns]
    summary = sites_gdf[site_cols + rad_cols].copy()
    summary = summary.sort_values('priority_score', ascending=False)
    summary.to_csv(out(cfg, 'tables', 'task1_site_summary.csv'), index=False)

    if eth_thresh is not None and 'rad_eTh' in nure_gdf.columns:
        conc_cols = [c for c in ('lab_id','lon','lat','Th','th_anomaly','th_source',
                                 'rad_K','rad_eTh','rad_eU','rad_eU_eTh','rad_K_eTh',
                                 'rad_eTh_high','rad_th_agree') if c in nure_gdf.columns]
        nure_gdf[conc_cols].to_csv(out(cfg, 'tables', 'task1_radiometric_concordance.csv'),
                                   index=False)

    print(f"\nTask 1 Results:")
    print(f"  Total mine sites evaluated: {len(sites_gdf)}")
    print(f"  Magnetic high (>threshold): {sites_gdf['mag_high'].sum()}")
    print(f"  Near NURE Th anomaly:       {sites_gdf['th_near'].sum()}")
    print(f"  Near NURE Y anomaly (HREE): {sites_gdf['y_near'].sum()}")
    print(f"  Multi-commodity targets:    {sites_gdf['multicommodity_target'].sum()}")
    top = summary[summary['priority_score'] >= 2]
    if len(top):
        print(f"\nTop priority sites:")
        print(top[['name','mag_anomaly_nT','th_value_ppm','th_source','priority_score']].to_string(index=False))
    print(f"\nGeoJSON: {out(cfg, 'geojson', 'task1_multicommodity_targets.geojson')}")
    print(f"Table:   {out(cfg, 'tables', 'task1_site_summary.csv')}")
    if eth_thresh is not None:
        print(f"Fig 1b:  {out(cfg, 'figures', 'fig1b_airborne_eth_vs_nure_th.png')}")
        print(f"Concord: {out(cfg, 'tables', 'task1_radiometric_concordance.csv')}")


if __name__ == '__main__':
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/ne_washington/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
