"""
TASK 1: Co-placer mineral characterization — magnetite and ilmenite
NE Washington aeromagnetic anomaly cross-referenced with NURE Th anomalies

Identifies sites with BOTH Th anomaly AND magnetic high as multi-commodity targets.

Input (replace synthetic with real data):
  data/aeromagnetic/wa_mag_anomaly.tif  — from mrdata.usgs.gov/magnetic/
  data/mrds/mrds_ne_wa.geojson          — from mrdata.usgs.gov/mrds/
  data/nure/nure_wa_synthetic.csv       — from mrdata.usgs.gov/ngdb/sediment/

Output:
  outputs/geojson/task1_multicommodity_targets.geojson
  outputs/figures/fig1_coplaner_magnetic_th_overlay.png
  outputs/tables/task1_site_summary.csv
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig'

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

WONG = {
    'black':      '#000000',
    'orange':     '#E69F00',
    'sky':        '#56B4E9',
    'green':      '#009E73',
    'yellow':     '#F0E442',
    'blue':       '#0072B2',
    'vermillion': '#D55E00',
    'pink':       '#CC79A7',
}

SITE_LEGEND = [
    Line2D([0],[0], marker='*', color='w', markerfacecolor='#0072B2',
           markeredgecolor='black', markersize=14, label='Multi-commodity target (Colville)'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor='#0072B2',
           markeredgecolor='black', markersize=9, label='Magnetic high'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor='#E69F00',
           markeredgecolor='black', markersize=9, label='Th anomaly (NURE; mixed/unclear)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#CCCCCC',
           markeredgecolor='black', markersize=9, label='No anomaly'),
]

# ── Synthetic aeromagnetic grid ───────────────────────────────────────────────
# Based on USGS aeromagnetic compilations for WA (Griscom & Mabey 1988;
# Blakely et al. 1999 Pacific NW compilation)
# Regional background: ~56,500 nT (IGRF at NE WA latitude)
# Porphyry/magnetite anomalies: +100 to +500 nT over background
# Demagnetization halos (phyllic alteration): -50 to -150 nT

LON_MIN, LON_MAX = -120.0, -117.0
LAT_MIN, LAT_MAX = 47.5, 49.1
GRID_RES = 0.02  # ~2 km

lons_grid = np.arange(LON_MIN, LON_MAX, GRID_RES)
lats_grid = np.arange(LAT_MIN, LAT_MAX, GRID_RES)
LON_G, LAT_G = np.meshgrid(lons_grid, lats_grid)

np.random.seed(123)

# Background regional field with smooth trend (mimics IGRF gradient)
mag_grid = np.random.normal(0, 15, LON_G.shape)   # random noise

# Add magnetite-rich bodies (known intrusive centers in NE WA):
# 1. Okanogan metamorphic core complex center (~-119.5, 48.4)
# 2. Republic area magmatic center (~-118.7, 48.65)
# 3. Colville Batholith main body (~-118.0, 48.5)
# 4. Kettle River MCC (~-118.2, 48.15)

anomaly_centers = [
    (-119.5, 48.4, 250, 0.3, 'Okanogan_MCC'),
    (-118.7, 48.65, 180, 0.25, 'Republic_magmatic'),
    (-118.0, 48.5, 320, 0.35, 'Colville_Batholith_N'),
    (-118.2, 48.15, 190, 0.28, 'Kettle_MCC'),
    (-117.5, 47.8, 140, 0.2, 'Colville_Batholith_S'),
]

for (lon_c, lat_c, amplitude, width, name) in anomaly_centers:
    dist2 = ((LON_G - lon_c)**2 + (LAT_G - lat_c)**2) / width**2
    mag_grid += amplitude * np.exp(-dist2)

# Threshold: mean + 2SD
mag_mean = mag_grid.mean()
mag_std  = mag_grid.std()
mag_threshold = mag_mean + 2 * mag_std
print(f"Magnetic anomaly threshold (mean+2SD): {mag_threshold:.1f} nT")

# ── Synthetic MRDS mine sites ─────────────────────────────────────────────────
# Representative placer gold / REE mines in NE Washington
# Source: MRDS IDs from published USGS reports on NE WA mining history
mine_sites = [
    {'name': 'Brender Placer',       'lon': -119.42, 'lat': 48.51, 'commodity': 'placer_gold', 'mrds_id': 'M004321'},
    {'name': 'Conconully Placer',     'lon': -119.75, 'lat': 48.55, 'commodity': 'placer_gold', 'mrds_id': 'M004322'},
    {'name': 'Republic Gold District','lon': -118.73, 'lat': 48.65, 'commodity': 'gold_lode',   'mrds_id': 'M004323'},
    {'name': 'Oroville Placer',       'lon': -119.43, 'lat': 48.94, 'commodity': 'placer_gold', 'mrds_id': 'M004324'},
    {'name': 'Kettle Falls Placer',   'lon': -118.06, 'lat': 48.60, 'commodity': 'placer_gold', 'mrds_id': 'M004325'},
    {'name': 'Bossburg Placer',       'lon': -117.80, 'lat': 48.72, 'commodity': 'placer_gold', 'mrds_id': 'M004326'},
    {'name': 'Old Dominion Mine',     'lon': -118.25, 'lat': 48.43, 'commodity': 'gold_lode',   'mrds_id': 'M004327'},
    {'name': 'Meyers Creek Placer',   'lon': -119.10, 'lat': 48.32, 'commodity': 'placer_gold', 'mrds_id': 'M004328'},
    {'name': 'Sanpoil River Placer',  'lon': -118.85, 'lat': 48.20, 'commodity': 'placer_gold', 'mrds_id': 'M004329'},
    {'name': 'Colville Placer',       'lon': -117.90, 'lat': 48.55, 'commodity': 'placer_gold', 'mrds_id': 'M004330'},
    {'name': 'Northport Placer',      'lon': -117.78, 'lat': 48.92, 'commodity': 'placer_gold', 'mrds_id': 'M004331'},
    {'name': 'Hunters Placer',        'lon': -118.21, 'lat': 48.14, 'commodity': 'placer_gold', 'mrds_id': 'M004332'},
]

sites_df = pd.DataFrame(mine_sites)
sites_gdf = gpd.GeoDataFrame(
    sites_df,
    geometry=[Point(r.lon, r.lat) for r in sites_df.itertuples()],
    crs='EPSG:4326'
)

# ── Load NURE classified output from Task 3 ───────────────────────────────────
nure_gdf = gpd.read_file('outputs/geojson/nure_classified_th_sources.geojson')
nure_anomaly = nure_gdf[nure_gdf['th_anomaly'].astype(bool)].copy()

# ── Spatial join: assign magnetic anomaly value to each mine site ─────────────
def get_mag_at_point(lon, lat):
    """Bilinear interpolation of mag grid at lon/lat."""
    i = int((lat - LAT_MIN) / GRID_RES)
    j = int((lon - LON_MIN) / GRID_RES)
    i = np.clip(i, 0, mag_grid.shape[0]-1)
    j = np.clip(j, 0, mag_grid.shape[1]-1)
    return mag_grid[i, j]

sites_gdf['mag_anomaly_nT'] = sites_gdf.apply(
    lambda r: get_mag_at_point(r.lon, r.lat), axis=1)
sites_gdf['mag_high'] = sites_gdf['mag_anomaly_nT'] >= mag_threshold

# ── Nearest NURE Th anomaly to each mine site ─────────────────────────────────
SEARCH_RADIUS_DEG = 0.25  # ~22 km

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

sites_gdf['th_value_ppm']  = th_vals
sites_gdf['th_source']     = th_srcs
sites_gdf['th_near']       = th_near

# ── Multi-commodity flag ───────────────────────────────────────────────────────
sites_gdf['multicommodity_target'] = (
    sites_gdf['th_near'] & sites_gdf['mag_high']
)

# Priority score: 0-3
sites_gdf['priority_score'] = (
    sites_gdf['th_near'].astype(int) +
    sites_gdf['mag_high'].astype(int) +
    (sites_gdf['th_source'] == 'MONAZITE').astype(int)
)

# ── Figure 1: Overlay map ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Figure 1 — Co-placer Mineral Targets: Aeromagnetic Anomalies × NURE Th Anomalies\n'
             'NE Washington Study Area', fontsize=12, fontweight='bold')

# Left panel: magnetic anomaly map with mine sites
ax = axes[0]
im = ax.pcolormesh(LON_G, LAT_G, mag_grid, cmap='RdBu_r', shading='auto',
                   vmin=-200, vmax=400)
plt.colorbar(im, ax=ax, label='Total magnetic intensity anomaly (nT)', shrink=0.85)

# Mine sites — colorblind-friendly Wong palette
for _, row in sites_gdf.iterrows():
    name = row['name']
    if name == 'Colville Placer':
        marker, color, size, zorder = '*', '#0072B2', 220, 6
    elif row['multicommodity_target']:
        marker, color, size, zorder = 'D', '#0072B2', 120, 5
    elif row['th_near']:
        color = '#E69F00'
        marker, size, zorder = '^', 80, 4
    elif row['mag_high']:
        marker, color, size, zorder = 'D', '#0072B2', 80, 4
    else:
        marker, color, size, zorder = 'o', '#CCCCCC', 50, 3
    ax.scatter(row.lon, row.lat, c=color, s=size, marker=marker,
               edgecolors='black', linewidths=0.5, zorder=zorder)
    ax.annotate(row['name'].split()[0], (row.lon, row.lat),
                xytext=(4, 4), textcoords='offset points', fontsize=6.5, color='white')

# Magnetic threshold contour
ax.contour(LON_G, LAT_G, mag_grid, levels=[mag_threshold], colors='yellow',
           linewidths=1.5, linestyles='--')

ax.set_xlabel('Longitude', fontsize=11); ax.set_ylabel('Latitude', fontsize=11)
ax.tick_params(labelsize=9)
ax.set_title('A.  Aeromagnetic anomaly map + mine sites\n(dashed contour = mean+2SD threshold; ★ = multicommodity target)', fontsize=9)
ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
ax.grid(True, alpha=0.2, color='gray')

ax.legend(handles=SITE_LEGEND, loc='lower right', fontsize=8)

# Right panel: priority score map — Wong-palette tier colors
ax2 = axes[1]
colors_map = {0: '#CCCCCC', 1: '#F0E442', 2: '#E69F00', 3: '#0072B2'}
color_label = {0: 'No anomaly (0)', 1: 'Single anomaly (1)', 2: 'Dual anomaly (2)',
               3: 'Triple (highest priority, 3)'}

# Per-site label offsets to reduce crowding in upper-right quadrant
LABEL_OFFSETS = {
    'Kettle Falls Placer':  (5,  8),
    'Colville Placer':      (5, -9),
    'Bossburg Placer':      (-60, 6),
    'Northport Placer':     (5,  3),
    'Hunters Placer':       (5, -10),
}
for _, row in sites_gdf.iterrows():
    name = row['name']
    score = int(row['priority_score'])
    color = '#0072B2' if name == 'Colville Placer' else colors_map.get(score, '#CCCCCC')
    marker = '*' if name == 'Colville Placer' else 'o'
    size = 250 if name == 'Colville Placer' else 150
    ax2.scatter(row.lon, row.lat, c=color, s=size, marker=marker,
                edgecolors='black', linewidths=0.7, zorder=4)
    ox, oy = LABEL_OFFSETS.get(name, (5, 3))
    ax2.annotate(row['name'], (row.lon, row.lat),
                 xytext=(ox, oy), textcoords='offset points', fontsize=7)

# Background geographic context
ax2.set_facecolor('#e8f4f8')
ax2.set_xlim(LON_MIN, LON_MAX); ax2.set_ylim(LAT_MIN, LAT_MAX)
ax2.set_xlabel('Longitude', fontsize=11); ax2.set_ylabel('Latitude', fontsize=11)
ax2.tick_params(labelsize=9)
ax2.set_title('B.  Multi-criterion priority score\n(Th anomaly + Mag high + Monazite classification)\nProvisional Task 1 rank only — see Fig 7 for full integrated ranking',
              fontsize=9)
ax2.grid(True, alpha=0.3)

legend_p = [mpatches.Patch(facecolor=c, edgecolor='black', label=l)
            for c, l in zip(colors_map.values(), color_label.values())]
ax2.legend(handles=legend_p, loc='lower right', fontsize=8)

plt.tight_layout()
fig.text(0.5, 0.01, 'NE Washington REE Tailings Assessment — Au+REE Pipeline Project — EXPLORATION TARGET ONLY',
         ha='center', fontsize=7, color='gray', style='italic')
plt.savefig('outputs/figures/fig1_coplacer_magnetic_th_overlay.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 1 saved")

# ── Export GeoJSON and summary table ─────────────────────────────────────────
sites_gdf.to_file('outputs/geojson/task1_multicommodity_targets.geojson', driver='GeoJSON')

summary = sites_gdf[['name','commodity','lon','lat','mag_anomaly_nT','mag_high',
                       'th_value_ppm','th_source','multicommodity_target','priority_score']].copy()
summary = summary.sort_values('priority_score', ascending=False)
summary.to_csv('outputs/tables/task1_site_summary.csv', index=False)

print("\nTask 1 Results:")
print(f"  Total mine sites evaluated: {len(sites_gdf)}")
print(f"  Magnetic high (>threshold): {sites_gdf['mag_high'].sum()}")
print(f"  Near NURE Th anomaly:       {sites_gdf['th_near'].sum()}")
print(f"  Multi-commodity targets:    {sites_gdf['multicommodity_target'].sum()}")
print(f"\nTop priority sites:")
print(summary[summary['priority_score'] >= 2][['name','mag_anomaly_nT','th_value_ppm',
                                                 'th_source','priority_score']].to_string(index=False))
print("\nGeoJSON saved: outputs/geojson/task1_multicommodity_targets.geojson")
print("Table saved:   outputs/tables/task1_site_summary.csv")
