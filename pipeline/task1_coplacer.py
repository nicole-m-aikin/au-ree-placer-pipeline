"""
Task 1: Co-placer mineral characterization — magnetite and ilmenite
Aeromagnetic anomaly cross-referenced with NURE Th anomalies.

Output:
  {outputs_dir}/geojson/task1_multicommodity_targets.geojson
  {outputs_dir}/figures/fig1_coplacer_magnetic_th_overlay.png
  {outputs_dir}/tables/task1_site_summary.csv
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

from pipeline.utils import (WONG, setup_mpl, watermark, save_fig, ensure_outputs, out,
                             map_extent, hillshade, north_arrow, scale_bar)


def run(cfg):
    setup_mpl()
    ensure_outputs(cfg['outputs_dir'])

    b = cfg['study_area']['bbox']
    LON_MIN, LON_MAX = b['lon_min'], b['lon_max']
    LAT_MIN, LAT_MAX = b['lat_min'], b['lat_max']
    GRID_RES = 0.02

    lons_grid = np.arange(LON_MIN, LON_MAX, GRID_RES)
    lats_grid = np.arange(LAT_MIN, LAT_MAX, GRID_RES)
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

    def get_mag_at_point(lon, lat):
        i = int((lat - LAT_MIN) / GRID_RES)
        j = int((lon - LON_MIN) / GRID_RES)
        i = np.clip(i, 0, mag_grid.shape[0] - 1)
        j = np.clip(j, 0, mag_grid.shape[1] - 1)
        return mag_grid[i, j]

    sites_gdf['mag_anomaly_nT'] = sites_gdf.apply(
        lambda r: get_mag_at_point(r.lon, r.lat), axis=1)
    sites_gdf['mag_high'] = sites_gdf['mag_anomaly_nT'] >= mag_threshold

    SEARCH_RADIUS_DEG = 0.25

    def nearest_th(site_lon, site_lat, radius=SEARCH_RADIUS_DEG):
        dist = np.sqrt((nure_anomaly['lon'] - site_lon)**2 + (nure_anomaly['lat'] - site_lat)**2)
        near = nure_anomaly[dist <= radius]
        if len(near) == 0:
            return None, None, False
        best = near.loc[near['Th'].idxmax()]
        return best['Th'], best['th_source'], True

    th_vals, th_srcs, th_near = [], [], []
    for _, row in sites_gdf.iterrows():
        tv, ts, tn = nearest_th(row.lon, row.lat)
        th_vals.append(tv)
        th_srcs.append(ts)
        th_near.append(tn)

    sites_gdf['th_value_ppm'] = th_vals
    sites_gdf['th_source']    = th_srcs
    sites_gdf['th_near']      = th_near

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
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#CCCCCC',
               markeredgecolor='black', markersize=9, label='No anomaly'),
    ]

    label_offsets = {k: tuple(v) for k, v in
                     cfg.get('task1_label_offsets', {}).items()}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f'Figure 1 — Co-placer Mineral Targets: Aeromagnetic Anomalies × NURE Th Anomalies\n'
        f'Mineral Systems: Source mineralogy (co-placer indicators) — {cfg["study_area"]["name"]}',
        fontsize=12, fontweight='bold',
    )

    ax = axes[0]
    xmin_1, xmax_1, ymin_1, ymax_1 = map_extent(cfg)
    hillshade(cfg, ax, alpha=0.15)
    im = ax.pcolormesh(LON_G, LAT_G, mag_grid, cmap='RdBu_r', shading='auto',
                       vmin=-200, vmax=400)
    plt.colorbar(im, ax=ax, label='Total magnetic intensity anomaly (nT)', shrink=0.85)

    top_site_name = (cfg.get('task1_highlight_site') or
                     sorted(sites, key=lambda s: s['lon'])[0]['name'])

    for _, row in sites_gdf.iterrows():
        name = row['name']
        if name == top_site_name:
            marker, color, size, zorder = '*', WONG['blue'], 220, 6
        elif row['multicommodity_target']:
            marker, color, size, zorder = 'D', WONG['blue'], 120, 5
        elif row['th_near']:
            color = WONG['orange']
            marker, size, zorder = '^', 80, 4
        elif row['mag_high']:
            marker, color, size, zorder = 'D', WONG['blue'], 80, 4
        else:
            marker, color, size, zorder = 'o', '#CCCCCC', 50, 3
        ax.scatter(row.lon, row.lat, c=color, s=size, marker=marker,
                   edgecolors='black', linewidths=0.5, zorder=zorder)
        ax.annotate(row['name'].split()[0], (row.lon, row.lat),
                    xytext=(4, 4), textcoords='offset points', fontsize=6.5, color='white')

    ax.contour(LON_G, LAT_G, mag_grid, levels=[mag_threshold], colors='yellow',
               linewidths=1.5, linestyles='--')
    ax.set_xlabel('Longitude', fontsize=11); ax.set_ylabel('Latitude', fontsize=11)
    ax.tick_params(labelsize=9)
    ax.set_title('A.  Aeromagnetic anomaly map + mine sites\n'
                 '(dashed contour = mean+2SD threshold; ★ = multicommodity target)', fontsize=9)
    ax.set_xlim(xmin_1, xmax_1); ax.set_ylim(ymin_1, ymax_1)
    north_arrow(ax)
    scale_bar(ax, cfg)
    ax.grid(True, alpha=0.2, color='gray')
    ax.legend(handles=site_legend, loc='lower right', fontsize=8)

    ax2 = axes[1]
    colors_map  = {0: '#CCCCCC', 1: WONG['yellow'], 2: WONG['orange'], 3: WONG['blue']}
    color_label = {0: 'No anomaly (0)', 1: 'Single anomaly (1)',
                   2: 'Dual anomaly (2)', 3: 'Triple (highest priority, 3)'}

    for _, row in sites_gdf.iterrows():
        name  = row['name']
        score = int(row['priority_score'])
        color  = WONG['blue'] if name == top_site_name else colors_map.get(score, '#CCCCCC')
        marker = '*' if name == top_site_name else 'o'
        size   = 250 if name == top_site_name else 150
        ax2.scatter(row.lon, row.lat, c=color, s=size, marker=marker,
                    edgecolors='black', linewidths=0.7, zorder=4)
        ox, oy = label_offsets.get(name, (5, 3))
        ax2.annotate(row['name'], (row.lon, row.lat),
                     xytext=(ox, oy), textcoords='offset points', fontsize=7)

    ax2.set_facecolor('#e8f4f8')
    ax2.set_xlim(LON_MIN, LON_MAX); ax2.set_ylim(LAT_MIN, LAT_MAX)
    ax2.set_xlabel('Longitude', fontsize=11); ax2.set_ylabel('Latitude', fontsize=11)
    ax2.tick_params(labelsize=9)
    ax2.set_title('B.  Multi-criterion priority score\n'
                  '(Th anomaly + Mag high + Monazite classification)\n'
                  'Provisional Task 1 rank only — see Fig 7 for full integrated ranking',
                  fontsize=9)
    ax2.grid(True, alpha=0.3)
    legend_p = [mpatches.Patch(facecolor=c, edgecolor='black', label=l)
                for c, l in zip(colors_map.values(), color_label.values())]
    ax2.legend(handles=legend_p, loc='lower right', fontsize=8)

    plt.tight_layout()
    watermark(fig, cfg)
    save_fig(fig, out(cfg, 'figures', 'fig1_coplacer_magnetic_th_overlay.png'))

    sites_gdf.to_file(out(cfg, 'geojson', 'task1_multicommodity_targets.geojson'), driver='GeoJSON')

    summary = sites_gdf[['name','commodity','lon','lat','mag_anomaly_nT','mag_high',
                          'th_value_ppm','th_source','multicommodity_target','priority_score']].copy()
    summary = summary.sort_values('priority_score', ascending=False)
    summary.to_csv(out(cfg, 'tables', 'task1_site_summary.csv'), index=False)

    print(f"\nTask 1 Results:")
    print(f"  Total mine sites evaluated: {len(sites_gdf)}")
    print(f"  Magnetic high (>threshold): {sites_gdf['mag_high'].sum()}")
    print(f"  Near NURE Th anomaly:       {sites_gdf['th_near'].sum()}")
    print(f"  Multi-commodity targets:    {sites_gdf['multicommodity_target'].sum()}")
    top = summary[summary['priority_score'] >= 2]
    if len(top):
        print(f"\nTop priority sites:")
        print(top[['name','mag_anomaly_nT','th_value_ppm','th_source','priority_score']].to_string(index=False))
    print(f"\nGeoJSON: {out(cfg, 'geojson', 'task1_multicommodity_targets.geojson')}")
    print(f"Table:   {out(cfg, 'tables', 'task1_site_summary.csv')}")


if __name__ == '__main__':
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/ne_washington/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
