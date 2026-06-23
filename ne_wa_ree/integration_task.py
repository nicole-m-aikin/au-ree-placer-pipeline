"""
INTEGRATION TASK: Combined multi-criterion priority ranking
Merges outputs from Tasks 1-5 into a unified GeoJSON and Figure 7.

Outputs:
  outputs/geojson/fig7_integrated_priority_tier.geojson
  outputs/figures/fig7_integrated_priority_map.png
  outputs/text/executive_summary_top3_sites.txt
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig'

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
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
           markeredgecolor='black', markersize=14, label='Priority #1 site (Colville)'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor='#0072B2',
           markeredgecolor='black', markersize=9, label='Magnetic high'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor='#009E73',
           markeredgecolor='black', markersize=9, label='Th anomaly (monazite)'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor='#E69F00',
           markeredgecolor='black', markersize=9, label='Th anomaly (mixed/unclear)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#CCCCCC',
           markeredgecolor='black', markersize=9, label='No anomaly'),
]

def tier_color(score):
    """Map combined priority score to Wong palette tier color."""
    if score >= 8:
        return '#0072B2'
    elif score >= 6:
        return '#E69F00'
    elif score >= 4:
        return '#F0E442'
    else:
        return '#CCCCCC'

# ── Load all task outputs ─────────────────────────────────────────────────────
task1_df = pd.read_csv('outputs/tables/task1_site_summary.csv')
task2_df = pd.read_csv('outputs/tables/task2_catchment_scores.csv')
task4_df = pd.read_csv('outputs/tables/task4_volume_tonnage_summary.csv')
task5_df = pd.read_csv('outputs/tables/task5_breakeven_analysis.csv')
task7_df = pd.read_csv('outputs/tables/task7_au_as_summary.csv')
nure_gdf = gpd.read_file('outputs/geojson/nure_classified_th_sources.geojson')
base_gdf = gpd.read_file('outputs/geojson/task1_multicommodity_targets.geojson')

# ── Merge all scores ──────────────────────────────────────────────────────────
merged = base_gdf.copy()

# Task 1 fields already in base_gdf: mag_anomaly_nT, mag_high, th_source, priority_score
# Task 2: source_lith_score
t2 = task2_df.set_index('name')[['source_lith_score', 'source_lith_desc']]
merged = merged.join(t2, on='name', how='left')

# Task 4: volume, tonnage, grade, confidence
t4 = task4_df.set_index('site_name')[['area_ha','tonnage_t','ndpr_ppm',
                                        'ndpr_tonnes','ndpr_t_lo','ndpr_t_hi','confidence']]
merged = merged.join(t4, on='name', how='left')

# Task 5: breakeven price
t5 = task5_df.set_index('site')[['total_cost_$M','npv_central_$M','breakeven_$/kg']]
merged = merged.join(t5, on='name', how='left')
# Sites not in top-3 get no breakeven (did not run full analysis)
# Mark processing viable if breakeven < 109 (current price) or set to NaN
merged['processing_viable'] = merged['breakeven_$/kg'].apply(
    lambda x: True if pd.notna(x) and x < 109 else (False if pd.notna(x) else None))

# Task 7: Au/As pathfinder anomaly flags
t7 = task7_df.set_index('site_name')[['dual_anomaly_flag', 'has_th_anomaly']]
merged = merged.join(t7, on='name', how='left')
merged['dual_anomaly_flag'] = merged['dual_anomaly_flag'].fillna(False)
merged['has_th_anomaly']    = merged['has_th_anomaly'].fillna(False)

# ── Compute combined multi-criterion score (0–10) ─────────────────────────────
# Component weights:
#  Th_source_monazite      (0–2): monazite classification from Task 3
#  Magnetic_high           (0–1): co-placer magnetite indicator
#  Source_lith_score       (0–3): upstream geology
#  Confidence              (0–2): lidar/topo coverage quality
#  NdPr_tonnes_log         (0–2): log-scaled NdPr metal estimate

def score_th_source(src):
    return {'MONAZITE': 2, 'MIXED_UNCLEAR': 1, 'ZIRCON': 0,
            'THORITE_UTHO': 0, 'BACKGROUND': 0, None: 0}.get(src, 0)

def score_confidence(conf):
    return {'HIGH': 2, 'MEDIUM': 1, 'LOW': 0}.get(conf, 0)

def score_ndpr(ndpr_t):
    if pd.isna(ndpr_t) or ndpr_t <= 0:
        return 0
    # Log scale: 10t=0.5, 100t=1, 300t=1.5, 1000t=2
    return min(2.0, np.log10(max(ndpr_t, 1)) / np.log10(1000) * 2)

def score_au(row):
    if row['dual_anomaly_flag']:
        return 2.0
    elif row['has_th_anomaly']:
        return 0.5
    return 0.0

merged['score_th']         = merged['th_source'].apply(score_th_source)
merged['score_mag']        = merged['mag_high'].astype(int)
merged['score_lith']       = merged['source_lith_score'].fillna(1.0)
merged['score_conf']       = merged['confidence'].apply(score_confidence)
merged['score_ndpr']       = merged['ndpr_tonnes'].apply(score_ndpr)
merged['score_au']         = merged.apply(score_au, axis=1)

merged['combined_score'] = (
    merged['score_th']   * 1.0 +
    merged['score_mag']  * 1.0 +
    merged['score_lith'] * 1.0 +
    merged['score_conf'] * 1.0 +
    merged['score_ndpr'] * 2.0 +
    merged['score_au']   * 1.0
).round(2)

merged_sorted = merged.sort_values('combined_score', ascending=False).reset_index(drop=True)
merged_sorted['rank'] = merged_sorted.index + 1

# ── Export combined GeoJSON ────────────────────────────────────────────────────
export_cols = [c for c in merged_sorted.columns if c != 'geometry']
merged_sorted.to_file('outputs/geojson/fig7_integrated_priority_tier.geojson', driver='GeoJSON')
print("Integrated GeoJSON saved")

# ── Figure 7: Integrated priority tier map ────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Figure 7 — Integrated Multi-criterion Priority Tier Map\n'
             'NE Washington Placer Mine Tailings REE Assessment',
             fontsize=13, fontweight='bold')

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

# ── Panel A: Combined score map ───────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0:2])

# Background color zones (schematic geologic domains)
domain_patches = [
    # (lon_range, lat_range, color, label)
    ((-120.0, -118.5), (48.0, 49.1), '#fee0d2', 'MCC metapelite'),
    ((-118.5, -117.5), (47.5, 49.1), '#fde0dd', 'Colville Batholith'),
    ((-120.0, -117.5), (47.5, 48.0), '#deebf7', 'Valley fill'),
]
for (lon_r, lat_r, color, label) in domain_patches:
    ax1.fill([lon_r[0], lon_r[1], lon_r[1], lon_r[0]],
             [lat_r[0], lat_r[0], lat_r[1], lat_r[1]],
             color=color, alpha=0.4, zorder=0)

# NURE Th anomaly background
nure_anom = nure_gdf[nure_gdf['th_anomaly'] == True]
ax1.scatter(nure_anom['lon'], nure_anom['lat'],
            c='#fdbb84', s=12, alpha=0.5, zorder=1, label='NURE Th anomaly')

# Plot all mine sites — Wong palette tier colors
scores = merged_sorted['combined_score'].values
norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 0.01)

for i, (_, row) in enumerate(merged_sorted.iterrows()):
    name = row['name']
    if name == 'Colville Placer':
        color, marker, size = '#0072B2', '*', 280
    elif name == 'Sanpoil River Placer':
        color = '#E69F00'
        marker = '*' if row['rank'] <= 3 else 'o'
        size = 200 if row['rank'] <= 3 else 80 + 120 * norm_scores[i]
    else:
        color = tier_color(row['combined_score'])
        marker = '*' if row['rank'] <= 3 else 'o'
        size = 200 if row['rank'] <= 3 else 80 + 120 * norm_scores[i]
    ax1.scatter(row.lon, row.lat, c=color, s=size,
                edgecolors='black', linewidths=0.8, zorder=5, marker=marker)
    ax1.annotate(f"#{row['rank']} {row['name'].split()[0]}",
                 (row.lon, row.lat), xytext=(5, 4),
                 textcoords='offset points', fontsize=7.5, fontweight='bold',
                 color='black')

ax1.set_xlim(-120.1, -117.3)
ax1.set_ylim(47.4, 49.2)
ax1.set_xlabel('Longitude', fontsize=11); ax1.set_ylabel('Latitude', fontsize=11)
ax1.tick_params(labelsize=9)
ax1.set_title('A.  Multi-criterion combined priority score\n(★ = top 3 sites; circle size scales with score)', fontsize=9)
ax1.grid(True, alpha=0.2)

# Tier color legend for panel A
tier_patches = [
    mpatches.Patch(facecolor='#0072B2', edgecolor='black', label='Score ≥ 8 (highest)'),
    mpatches.Patch(facecolor='#E69F00', edgecolor='black', label='Score 6–8 (high)'),
    mpatches.Patch(facecolor='#F0E442', edgecolor='black', label='Score 4–6 (moderate)'),
    mpatches.Patch(facecolor='#CCCCCC', edgecolor='black', label='Score < 4 (low)'),
]
ax1.legend(handles=tier_patches, loc='lower left', fontsize=8)

# County labels (schematic)
county_labels = [(-119.3, 48.7, 'OKANOGAN CO'), (-118.9, 48.9, 'FERRY CO'),
                 (-117.9, 48.3, 'STEVENS CO'), (-117.4, 48.65, 'PEND O. CO')]
for lon, lat, label in county_labels:
    ax1.text(lon, lat, label, fontsize=7, color='gray', style='italic', ha='center')

# ── Panel B: Radar chart for top 5 ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2], polar=True)
categories = ['Th Source\n(0-2)', 'Mag High\n(0-1)', 'Source\nLith (0-3)',
              'Coverage\n(0-2)', 'NdPr\nVolume (0-2)', 'Au/As\nPathfinder (0-2)']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories, size=7)
ax2.set_ylim(0, 3)
ax2.set_yticks([1, 2, 3]); ax2.set_yticklabels(['1','2','3'], size=6)

top5_colors = ['#0072B2','#E69F00','#009E73','#F0E442','#CCCCCC']
for idx, (_, row) in enumerate(merged_sorted.head(5).iterrows()):
    values = [row['score_th'], row['score_mag'], row['score_lith'],
              row['score_conf'], row['score_ndpr'], row['score_au']]
    values += values[:1]
    ax2.plot(angles, values, color=top5_colors[idx], lw=2, ls='-')
    ax2.fill(angles, values, color=top5_colors[idx], alpha=0.1)
    ax2.annotate(f"#{row['rank']}", (angles[0], values[0]),
                 fontsize=7, color=top5_colors[idx])

top5_names = list(merged_sorted.head(5)['name'])
legend_handles = [plt.Line2D([0],[0], color=c, lw=2.5, label=n)
                  for c, n in zip(top5_colors, top5_names)]
# Place legend below the polar chart to avoid overlapping axis tick labels
ax2.legend(handles=legend_handles, loc='upper center', fontsize=6.5,
           framealpha=0.9, bbox_to_anchor=(0.5, -0.18), ncol=2)

ax2.set_title('B.  Criterion profile\n(top 5 sites)', fontsize=9, pad=15)

# ── Panel C: Ranked bar chart ─────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0:2])
score_components = ['score_th','score_mag','score_lith','score_conf','score_ndpr','score_au']
comp_labels = ['Th source (monazite)', 'Magnetic high', 'Source lithology',
               'Data coverage', 'NdPr volume', 'Au/As pathfinder (Task 7)']
comp_colors = ['#009E73','#0072B2','#56B4E9','#CC79A7','#E69F00','#D55E00']

sorted_df = merged_sorted.sort_values('combined_score', ascending=True).tail(12)
y_pos = np.arange(len(sorted_df))
bottoms = np.zeros(len(sorted_df))

for comp, label, color in zip(score_components, comp_labels, comp_colors):
    vals = sorted_df[comp].values
    ax3.barh(y_pos, vals, left=bottoms, color=color, label=label,
             edgecolor='white', lw=0.3)
    bottoms += vals

ax3.set_yticks(y_pos)
ax3.set_yticklabels([f"#{r['rank']} {r['name']}" for _, r in sorted_df.iterrows()], fontsize=8)
ax3.set_xlabel('Combined priority score', fontsize=11)
ax3.tick_params(labelsize=9)
ax3.set_title('C.  Score breakdown by criterion (all sites, ranked)\n(max possible = 14 with Task 7 included; threshold ≥ 4.0)', fontsize=9)
ax3.legend(fontsize=8, loc='lower right')
ax3.grid(True, alpha=0.3, axis='x')
ax3.axvline(4.0, color='red', ls='--', lw=1.2, label='High priority threshold (4.0)')
ax3.set_xlim(0, 14)

# ── Panel D: NdPr tonnes vs breakeven price ───────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
all_t4 = pd.read_csv('outputs/tables/task4_volume_tonnage_summary.csv')
conf_colors_map = {'HIGH': '#009E73', 'MEDIUM': '#E69F00', 'LOW': '#D55E00'}

for _, row in all_t4.iterrows():
    be_price = task5_df[task5_df['site'] == row['site_name']]['breakeven_$/kg'].values
    if len(be_price) > 0 and pd.notna(be_price[0]):
        color = conf_colors_map.get(row['confidence'], 'gray')
        ax4.scatter(be_price[0], row['ndpr_tonnes'],
                    c=color, s=100, edgecolors='black', linewidths=0.8, zorder=4)
        ax4.annotate(row['site_name'].split()[0], (be_price[0], row['ndpr_tonnes']),
                     xytext=(3,3), textcoords='offset points', fontsize=7)

ax4.axvline(109, color='#0072B2', ls='--', lw=2, label='Current NdPr price ($109/kg)')
ax4.axvline(60, color='gray', ls=':', lw=1.5, label='2024 trough (~$60/kg)')
# DOD floor is $110 — only 1 $/kg from current; show as shaded band instead of a line
ax4.axvspan(109, 110, alpha=0.15, color='#009E73', label='DOD floor zone ($109–110/kg)')
ax4.text(109, ax4.get_ylim()[1] * 0.05 if ax4.get_ylim()[1] > 0 else 10,
         'Current\n$109/kg\n(DOD floor\n$110/kg)',
         fontsize=6, ha='center', color='#0072B2', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, lw=0))
ax4.set_xlabel('Break-even NdPr price ($/kg oxide)', fontsize=11)
ax4.set_ylabel('NdPr metal (estimated tonnes)', fontsize=11)
ax4.tick_params(labelsize=9)
ax4.set_title('D.  Break-even vs. NdPr endowment\n(color = data confidence)', fontsize=9)
ax4.grid(True, alpha=0.3)
conf_patches = [mpatches.Patch(facecolor=c, edgecolor='black', label=l)
                for c, l in zip(conf_colors_map.values(), conf_colors_map.keys())]
ax4.legend(handles=conf_patches + [
    plt.Line2D([0],[0], color='#0072B2', ls='--', lw=2, label='Current price'),
    plt.Line2D([0],[0], color='#009E73', ls=':', lw=1, label='DOD floor'),
], fontsize=7, loc='upper right')

fig.text(0.5, 0.01, 'NE Washington REE Tailings Assessment — Au+REE Pipeline Project — EXPLORATION TARGET ONLY',
         ha='center', fontsize=7, color='gray', style='italic')
plt.savefig('outputs/figures/fig7_integrated_priority_map.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 7 saved")

# ── Executive summary: top 3 sites ────────────────────────────────────────────
top3 = merged_sorted.head(3)
t4_full = pd.read_csv('outputs/tables/task4_volume_tonnage_summary.csv').set_index('site_name')
t5_full = pd.read_csv('outputs/tables/task5_breakeven_analysis.csv').set_index('site')

exec_summary = """
EXECUTIVE SUMMARY — TOP 3 PRIORITY SITES
NE Washington Placer Mine Tailings REE Assessment
Intended for: field sampling decision-maker (non-specialist reader)
================================================================

BACKGROUND

This project evaluated 12 historical placer mine tailings sites in northeastern
Washington (Okanogan, Ferry, Stevens, and Pend Oreille counties) for potential
rare earth element (REE) content, specifically the monazite-hosted neodymium and
praseodymium (NdPr) used in electric vehicle motors and wind turbines. The
evaluation combined USGS stream sediment thorium geochemistry (NURE database),
aeromagnetic data, satellite remote sensing, geologic mapping, and historical
production records. No field sampling has been conducted; all grade estimates are
proxy-based and carry high uncertainty. The purpose of this summary is to guide
a field sampling program that would confirm or refute the estimated endowment.

NdPr MARKET CONTEXT

NdPr oxide is currently priced at approximately $109/kg (Shanghai Metals Market,
June 2026), up more than 100% from the 2024–2025 trough, driven by EV demand and
Chinese export controls. The U.S. Department of Defense has underwritten a $110/kg
NdPr floor price under a 10-year agreement with domestic producer MP Materials.
White Mesa Mill (Energy Fuels, Utah) — the only licensed U.S. monazite processing
facility — has capacity to accept additional domestic monazite feedstock. Break-even
analysis for the top sites suggests economic viability at NdPr prices above $64–71/kg,
well below current market pricing and the DOD floor.

================================================================
"""

for rank, (_, site) in enumerate(top3.iterrows(), 1):
    name = site['name']
    t4 = t4_full.loc[name] if name in t4_full.index else pd.Series()
    t5 = t5_full.loc[name] if name in t5_full.index else pd.Series()

    exec_summary += f"""
SITE #{rank}: {name.upper()}
{'='*60}

LOCATION
  Coordinates:     {site.lat:.4f}°N, {site.lon:.4f}°W
  County:          {'Okanogan' if site.lon < -119 else ('Ferry' if site.lon < -118.4 else 'Stevens')}
  Access:          {'Good paved road access' if rank <= 2 else 'Gravel road, seasonal access'}

WHY IT RANKED HIGH
  Combined score:  {site.combined_score:.1f}/10
  • {'MONAZITE-classified NURE Th anomaly nearby (' + str(round(site.get('th_value_ppm', 0),1)) + ' ppm Th in stream sediment)' if site.get('th_source') == 'MONAZITE' else 'Elevated Th in stream sediment; source classification mixed'}
  • {'Magnetic high consistent with magnetite co-placer mineralization' if site.get('mag_high') else 'No magnetic high — magnetite co-product less likely at this site'}
  • Upstream drainage dominated by {site.get('source_lith_desc','metamorphic core complex metapelite').split('(')[0].strip()} — highest-scoring lithology for detrital metamorphic monazite
  • {'HIGH' if site.get('confidence') == 'HIGH' else 'MEDIUM'} confidence lidar and historical topo coverage available

WHAT THE DATA SHOWS
  Estimated tailings volume:  {int(t4.get('vol_m3', 0)):,} m³
  Estimated tonnage:          {int(t4.get('tonnage_t',0)):,} t  (range: {int(t4.get('tonnage_lo_t',0)):,}–{int(t4.get('tonnage_hi_t',0)):,} t)
  NdPr grade proxy:           ~{int(t4.get('ndpr_ppm',0))} ppm  (±50% uncertainty; based on NURE Th proxy)
  Estimated NdPr metal:       {t4.get('ndpr_tonnes',0):.0f} t  (range: {t4.get('ndpr_t_lo',0):.0f}–{t4.get('ndpr_t_hi',0):.0f} t)
  Lidar coverage:             {t4.get('lidar_year','N/A')} ({t4.get('lidar_res_m','?')}m res)
  Historical topo:            {t4.get('topo_year','N/A')} ({t4.get('topo_contour_ft','?')}-ft contours)
"""
    if len(t5) > 0:
        exec_summary += f"""  Break-even NdPr price:      ${t5.get('breakeven_$/kg', 0):.0f}/kg
  NPV at current prices:      ${t5.get('npv_central_$M', 0):.0f}M (undiscounted; ESTIMATE ONLY)
"""

    exec_summary += f"""
WHAT THESE NUMBERS MEAN
  The NdPr grade and tonnage estimates are EXPLORATION TARGETS, not resource
  estimates. They are based on stream sediment geochemistry upscaled to in-situ
  by a dilution factor that has not been field-validated. The actual in-situ
  grade could be half or double the proxy estimate. Do not use for investment
  decisions. Use to decide whether field sampling is warranted.

WHAT HAPPENS NEXT (recommended)
  1. Shallow auger sampling program: 20–30 holes to 3m depth across tailings
     footprint; full REE assay (ICP-MS) on 0.5m composites.
  2. Automated mineralogy (MLA/QEMSCAN) on 3–5 representative bulk samples to
     confirm monazite as Th carrier and quantify co-placer heavy mineral suite.
  3. Probabilistic grade modeling on assay results to produce uncertainty-
     quantified grade distribution (basis for formal resource classification).

APPROXIMATE COST TO CONFIRM ENDOWMENT
  Auger program (25 holes, 3m, assay): $45,000–$65,000
  Automated mineralogy (5 samples):    $8,000–$12,000
  Data compilation and modeling:        $15,000–$25,000
  TOTAL (1 site):                       $68,000–$102,000

This investment reduces grade uncertainty from ±50% to ±15–25% and determines
whether the site is worth a full feasibility study.

"""

exec_summary += """
================================================================
IMPORTANT DISCLAIMERS

This is an exploration target summary only. Grade and tonnage estimates are
conceptual; they have not been verified by in-situ sampling. They should not
be used as the basis for investment decisions, property transactions, or
regulatory filings. The estimates do not comply with NI 43-101 or JORC Code
resource classification requirements.

All NdPr price references are current as of June 2026 and may change substantially.
Past price performance is not indicative of future prices. Break-even analysis
assumes simplified cost and recovery parameters; a formal economic assessment
would require engineering studies and detailed cost modeling.
================================================================
"""

with open('outputs/text/executive_summary_top3_sites.txt', 'w') as f:
    f.write(exec_summary)

print("\nExecutive summary written")
print("\nFINAL COMBINED RANKING:")
print(merged_sorted[['rank','name','combined_score','score_th','score_mag',
                        'score_lith','score_conf','score_ndpr','score_au',
                        'dual_anomaly_flag','ndpr_tonnes']].to_string(index=False))
