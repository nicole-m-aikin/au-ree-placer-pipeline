"""
TASK 7: Au and epithermal/porphyry pathfinder anomaly analysis
NE Washington NURE stream sediment data

Figure 8: Au/As pathfinder anomaly map.

Classifies samples by pathfinder association and cross-references
with Th anomaly results from Task 3.

Input:  data/nure/nure_ne_wa_sediment.csv
        outputs/geojson/nure_classified_th_sources.geojson
        outputs/geojson/task1_multicommodity_targets.geojson
Output: outputs/figures/fig8_au_as_anomaly_map.png
        outputs/tables/task7_au_as_summary.csv
        outputs/text/task7_au_anomaly_notes.txt
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

# ── directories ──────────────────────────────────────────────────────────────
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/tables',  exist_ok=True)
os.makedirs('outputs/text',    exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load data
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Loading data")
print("="*60)

df = pd.read_csv('data/nure/nure_ne_wa_sediment.csv')
print(f"  NURE sediment: {df.shape[0]} samples, {df.shape[1]} columns")

th_gdf = gpd.read_file('outputs/geojson/nure_classified_th_sources.geojson')
print(f"  Th source GeoJSON: {th_gdf.shape[0]} records")

mine_gdf = gpd.read_file('outputs/geojson/task1_multicommodity_targets.geojson')
print(f"  Mine sites GeoJSON: {mine_gdf.shape[0]} sites")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Below-detection handling
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Below-detection handling")
print("="*60)

ELEMENTS = ['Au', 'As', 'Sb', 'Cu', 'Mo', 'Pb', 'Zn']
present = [e for e in ELEMENTS if e in df.columns]
print(f"  Elements found: {present}")

for col in present:
    series = pd.to_numeric(df[col], errors='coerce')
    pos_vals = series[series > 0]
    if pos_vals.empty:
        df[col] = np.nan
        continue
    med = pos_vals.median()
    threshold = 10 * med
    # large negatives → NaN
    large_neg = series < -threshold
    # small negatives → half of absolute value
    small_neg = (series < 0) & (series >= -threshold)
    series = series.copy()
    series[large_neg] = np.nan
    series[small_neg] = series[small_neg].abs() / 2.0
    df[col] = series
    n_large = large_neg.sum()
    n_small = small_neg.sum()
    print(f"  {col}: median_pos={med:.4g}, threshold={threshold:.4g}, "
          f"large_neg→NaN={n_large}, small_neg→half={n_small}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Au units check
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Au units check")
print("="*60)

au_units_warning = False
au_warning_msg   = ""

if 'Au' in df.columns:
    au = pd.to_numeric(df['Au'], errors='coerce')
    pos_au = au[au > 0]
    pct_nonnull = au.notna().mean() * 100
    print(f"  Au min    : {au.min():.6g}")
    print(f"  Au median : {au.median():.6g}")
    print(f"  Au mean   : {au.mean():.6g}")
    print(f"  Au max    : {au.max():.6g}")
    print(f"  Au %non-null: {pct_nonnull:.1f}%")

    if pos_au.median() > 0.05:
        au_units_warning = True
        au_warning_msg = (
            "WARNING: Median Au > 0.05 ppm — values may be in ppb mislabeled "
            "as ppm. Verify units with NGDB bestvalue.csv before interpreting."
        )
        print(f"\n  *** {au_warning_msg} ***\n")
    else:
        print(f"  Au median ({pos_au.median():.4g}) <= 0.05 ppm — units appear consistent.")
else:
    print("  Au column not found in dataset.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Log-threshold anomalies
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: Log-threshold anomaly cutoffs (mean + 2σ of log10)")
print("="*60)

threshold_els = ['Au', 'As', 'Sb', 'Cu', 'Mo']
thresholds = {}

for col in threshold_els:
    if col not in df.columns:
        thresholds[col] = None
        print(f"  {col}: column absent")
        continue
    pos = pd.to_numeric(df[col], errors='coerce')
    pos = pos[pos > 0]
    if len(pos) < 5:
        thresholds[col] = None
        print(f"  {col}: insufficient positive values ({len(pos)})")
        continue
    log_vals = np.log10(pos)
    thr = 10 ** (log_vals.mean() + 2 * log_vals.std())
    thresholds[col] = thr
    print(f"  {col}: n_pos={len(pos)}, log_mean={log_vals.mean():.3f}, "
          f"log_sd={log_vals.std():.3f}, threshold={thr:.4g}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Classify each sample
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: Sample classification")
print("="*60)

def is_anomalous(col, thr):
    """Return boolean Series: True where col > thr."""
    if col not in df.columns or thr is None:
        return pd.Series(False, index=df.index)
    vals = pd.to_numeric(df[col], errors='coerce')
    return vals > thr

au_anom  = is_anomalous('Au', thresholds.get('Au'))
as_anom  = is_anomalous('As', thresholds.get('As'))
sb_anom  = is_anomalous('Sb', thresholds.get('Sb'))
cu_anom  = is_anomalous('Cu', thresholds.get('Cu'))
mo_anom  = is_anomalous('Mo', thresholds.get('Mo'))

pb_anom = pd.Series(False, index=df.index)
zn_anom = pd.Series(False, index=df.index)
if 'Pb' in df.columns and thresholds.get('Pb') is None:
    # compute for Pb / Zn (not in threshold_els, compute now)
    pass

# Compute Pb and Zn thresholds for BASE_METAL classification
for col in ['Pb', 'Zn']:
    if col in df.columns:
        pos = pd.to_numeric(df[col], errors='coerce')
        pos = pos[pos > 0]
        if len(pos) >= 5:
            log_vals = np.log10(pos)
            thr = 10 ** (log_vals.mean() + 2 * log_vals.std())
            thresholds[col] = thr
            print(f"  {col}: threshold={thr:.4g}")
            if col == 'Pb':
                pb_anom = pd.to_numeric(df['Pb'], errors='coerce') > thr
            elif col == 'Zn':
                zn_anom = pd.to_numeric(df['Zn'], errors='coerce') > thr

epithermal = au_anom & (as_anom | sb_anom)
porphyry   = cu_anom & mo_anom
au_only    = au_anom & ~epithermal & ~porphyry
base_metal = (pb_anom | zn_anom) & ~au_anom
background = ~epithermal & ~porphyry & ~au_only & ~base_metal

df['au_class'] = 'BACKGROUND'
df.loc[base_metal,  'au_class'] = 'BASE_METAL'
df.loc[au_only,     'au_class'] = 'AU_ONLY'
df.loc[porphyry,    'au_class'] = 'PORPHYRY_CU'
df.loc[epithermal,  'au_class'] = 'EPITHERMAL_AU'

counts = df['au_class'].value_counts()
print("\n  Classification counts:")
for cls, cnt in counts.items():
    print(f"    {cls}: {cnt}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Spatial join: nearest Au-anomaly sample per mine site
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6: Nearest Au-anomaly sample per mine site (within 0.25°)")
print("="*60)

# Build GeoDataFrame for NURE samples with Au anomalies
au_anom_df = df[df['au_class'].isin(['EPITHERMAL_AU', 'PORPHYRY_CU', 'AU_ONLY'])].copy()
au_anom_gdf = gpd.GeoDataFrame(
    au_anom_df,
    geometry=[Point(xy) for xy in zip(au_anom_df['lon'], au_anom_df['lat'])],
    crs='EPSG:4326'
)

MAX_DIST = 0.25

site_records = []
for _, site in mine_gdf.iterrows():
    site_lon = site.geometry.x
    site_lat = site.geometry.y
    site_name = site.get('name', 'unknown')

    dists = np.sqrt(
        (au_anom_gdf['lon'] - site_lon) ** 2 +
        (au_anom_gdf['lat'] - site_lat) ** 2
    )
    mask = dists <= MAX_DIST

    if mask.any():
        idx_min = dists[mask].idxmin()
        nearest = au_anom_gdf.loc[idx_min]
        rec = {
            'site_name':       site_name,
            'nearest_au_ppm':  round(float(nearest['Au']), 4) if pd.notna(nearest['Au']) else np.nan,
            'nearest_as_ppm':  round(float(nearest['As']), 4) if ('As' in nearest and pd.notna(nearest['As'])) else np.nan,
            'au_class':        nearest['au_class'],
            'distance_deg':    round(float(dists[mask].min()), 4),
        }
    else:
        rec = {
            'site_name':      site_name,
            'nearest_au_ppm': np.nan,
            'nearest_as_ppm': np.nan,
            'au_class':       'NONE',
            'distance_deg':   np.nan,
        }
    site_records.append(rec)
    print(f"  {site_name}: au_class={rec['au_class']}, dist={rec['distance_deg']}, "
          f"Au={rec['nearest_au_ppm']}")

summary_df = pd.DataFrame(site_records)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Cross-reference with Th anomalies
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 7: Cross-reference with Th anomalies")
print("="*60)

# Th anomaly points
th_anom_gdf = th_gdf[th_gdf['th_anomaly'] == True].copy()
print(f"  Th anomaly samples: {len(th_anom_gdf)}")

has_th = []
dual   = []

for _, site in mine_gdf.iterrows():
    site_lon  = site.geometry.x
    site_lat  = site.geometry.y
    site_name = site.get('name', 'unknown')

    # Distance to any Th anomaly
    th_dists = np.sqrt(
        (th_anom_gdf['lon'] - site_lon) ** 2 +
        (th_anom_gdf['lat'] - site_lat) ** 2
    )
    near_th = (th_dists <= MAX_DIST).any() if len(th_dists) else False

    # Au anomaly within range (already in summary_df)
    row = summary_df[summary_df['site_name'] == site_name]
    near_au = (not row.empty) and (row.iloc[0]['au_class'] != 'NONE')

    has_th.append(near_th)
    dual.append(near_th and near_au)
    print(f"  {site_name}: has_th={near_th}, near_au={near_au}, dual={near_th and near_au}")

summary_df['has_th_anomaly']   = has_th
summary_df['dual_anomaly_flag'] = dual

dual_sites = summary_df[summary_df['dual_anomaly_flag']]['site_name'].tolist()
print(f"\n  Dual Th+Au anomaly sites: {dual_sites}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Figure 7b
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 8: Generating Figure 8")
print("="*60)

fig, ax = plt.subplots(figsize=(11, 9))

# ── background samples ───────────────────────────────────────────────────────
bg = df[df['au_class'] == 'BACKGROUND']
ax.scatter(bg['lon'], bg['lat'], s=5, alpha=0.3, color='#CCCCCC',
           zorder=1, label='_nolegend_')

# ── AU_ONLY ──────────────────────────────────────────────────────────────────
ao = df[df['au_class'] == 'AU_ONLY']
if not ao.empty:
    ax.scatter(ao['lon'], ao['lat'], s=30, color=WONG['sky'],
               edgecolors='white', linewidths=0.5, zorder=3, label='_nolegend_')

# ── PORPHYRY_CU ──────────────────────────────────────────────────────────────
pc = df[df['au_class'] == 'PORPHYRY_CU']
if not pc.empty:
    ax.scatter(pc['lon'], pc['lat'], s=50, marker='s', color=WONG['blue'],
               edgecolors='black', linewidths=0.6, zorder=4, label='_nolegend_')
    for _, pr in pc.iterrows():
        cu_val = pr.get('Cu', float('nan'))
        mo_val = pr.get('Mo', float('nan'))
        label_txt = f"Cu={cu_val:.0f}\nMo={mo_val:.1f}"
        ax.annotate(label_txt, xy=(pr['lon'], pr['lat']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=5.5, color=WONG['blue'], zorder=9,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.6, lw=0))

# ── EPITHERMAL_AU sized by Au ────────────────────────────────────────────────
ep = df[df['au_class'] == 'EPITHERMAL_AU'].copy()
if not ep.empty:
    au_thr = thresholds.get('Au') or 1.0
    ep['Au_num'] = pd.to_numeric(ep['Au'], errors='coerce').fillna(au_thr)
    sizes = np.clip(50 * ep['Au_num'] / au_thr, 20, 400)
    ax.scatter(ep['lon'], ep['lat'], s=sizes, color=WONG['orange'],
               edgecolors='black', linewidths=0.4, zorder=5, label='_nolegend_')

# ── BASE_METAL ───────────────────────────────────────────────────────────────
bm = df[df['au_class'] == 'BASE_METAL']
if not bm.empty:
    ax.scatter(bm['lon'], bm['lat'], s=20, marker='^', color=WONG['pink'],
               edgecolors='white', linewidths=0.4, zorder=3, label='_nolegend_')

# ── Mine sites ───────────────────────────────────────────────────────────────
# Per-site label offsets to prevent overlap (lon-close neighbors)
LABEL_OFFSETS = {
    'Hunters Placer':    (5, -10),   # shift below to clear dual-anomaly star
    'Old Dominion Mine': (5,   5),   # keep above
}
DEFAULT_OFFSET = (4, 4)

for _, site in mine_gdf.iterrows():
    sname = site.get('name', 'unknown')
    row   = summary_df[summary_df['site_name'] == sname]
    is_dual  = bool(row['dual_anomaly_flag'].values[0]) if not row.empty else False
    near_au  = (not row.empty) and (row.iloc[0]['au_class'] != 'NONE')

    if is_dual:
        scolor = WONG['vermillion']
    elif near_au:
        scolor = WONG['green']
    else:
        scolor = '#CCCCCC'

    ax.scatter(site.geometry.x, site.geometry.y, marker='D', s=80,
               color=scolor, edgecolors='black', linewidths=0.8, zorder=7)

    if is_dual:
        ax.scatter(site.geometry.x, site.geometry.y, marker='*', s=300,
                   facecolor=scolor, edgecolors='black', linewidths=0.6, zorder=8)

    offset = LABEL_OFFSETS.get(sname, DEFAULT_OFFSET)
    ax.annotate(sname, xy=(site.geometry.x, site.geometry.y),
                xytext=offset, textcoords='offset points',
                fontsize=6, zorder=9, color='#222222')

# ── Legend — only include classes that have data ──────────────────────────────
SITE_LEGEND = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#CCCCCC',
           markersize=5, label='Background'),
]
# Only add Au-only if samples exist
if not ao.empty:
    SITE_LEGEND.append(Line2D([0],[0], marker='o', color='w',
                               markerfacecolor=WONG['sky'], markersize=7, label='Au only'))
# Only add Porphyry Cu if samples exist
if not pc.empty:
    SITE_LEGEND.append(Line2D([0],[0], marker='s', color='w',
                               markerfacecolor=WONG['blue'], markersize=7, label='Porphyry Cu'))
# Only add Epithermal Au if samples exist
if not ep.empty:
    SITE_LEGEND.append(Line2D([0],[0], marker='o', color='w',
                               markerfacecolor=WONG['orange'], markersize=9, label='Epithermal Au'))
# Only add Base metal if samples exist
if not bm.empty:
    SITE_LEGEND.append(Line2D([0],[0], marker='^', color='w',
                               markerfacecolor=WONG['pink'], markersize=7, label='Base metal'))

SITE_LEGEND += [
    Line2D([0],[0], marker='D', color='w', markerfacecolor='#CCCCCC',
           markeredgecolor='black', markersize=8, label='Mine site (no anomaly)'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor=WONG['green'],
           markeredgecolor='black', markersize=8, label='Mine site (Au only)'),
    Line2D([0],[0], marker='*', color='w', markerfacecolor=WONG['vermillion'],
           markeredgecolor='black', markersize=12, label='★ Dual Th+Au anomaly (Hunters)'),
]
ax.legend(handles=SITE_LEGEND, loc='lower left', fontsize=7,
          framealpha=0.85, title='Classification', title_fontsize=8)

ax.set_title('NE Washington — Au/As/Pathfinder Anomaly Map (Task 7)',
             fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Longitude', fontsize=10)
ax.set_ylabel('Latitude',  fontsize=10)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

# Flag the dense Au-only cluster near Pend Oreille County (~48.8°N, −117.2°W)
ax.annotate('Au-only cluster\n(Pend Oreille Co.)\nnot in MRDS inventory\n→ future work',
            xy=(-117.25, 48.8), xytext=(-118.6, 48.85),
            arrowprops=dict(arrowstyle='->', color=WONG['sky'], lw=1.0),
            fontsize=6.5, color=WONG['sky'], style='italic',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, lw=0))

fig.text(0.5, 0.01,
         'NE Washington REE Tailings Assessment — Au+REE Pipeline Project — EXPLORATION TARGET ONLY',
         ha='center', va='bottom', fontsize=6, color='gray', style='italic')

plt.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig('outputs/figures/fig8_au_as_anomaly_map.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  Saved: outputs/figures/fig8_au_as_anomaly_map.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Save outputs
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 9: Saving outputs")
print("="*60)

# CSV summary
summary_df.to_csv('outputs/tables/task7_au_as_summary.csv', index=False)
print("  Saved: outputs/tables/task7_au_as_summary.csv")

# Text notes
lines = []
lines.append("TASK 7 — Au/As/Pathfinder Anomaly Analysis Notes")
lines.append("=" * 60)
lines.append("")
if au_units_warning:
    lines.append("UNITS WARNING:")
    lines.append(f"  {au_warning_msg}")
    lines.append("")

lines.append("CLASSIFICATION COUNTS:")
for cls, cnt in counts.items():
    lines.append(f"  {cls}: {cnt}")
lines.append("")

lines.append("DUAL Th+Au ANOMALY SITES:")
if dual_sites:
    for s in dual_sites:
        lines.append(f"  - {s}")
else:
    lines.append("  None identified within 0.25° search radius.")
lines.append("")

lines.append("INTERPRETATION:")
n_ep   = int(counts.get('EPITHERMAL_AU', 0))
n_pc   = int(counts.get('PORPHYRY_CU',  0))
n_ao   = int(counts.get('AU_ONLY',       0))
n_dual = len(dual_sites)
lines.append(
    f"  The NE Washington NURE dataset yielded {n_ep} epithermal-Au-signature "
    f"samples (Au + As/Sb anomalous), {n_pc} porphyry-Cu samples (Cu + Mo), "
    f"and {n_ao} Au-only anomalies. Of the 12 mine sites, {n_dual} show "
    f"spatial coincidence with BOTH a Th-rich source anomaly (Task 3) and an "
    f"Au-pathfinder anomaly, representing the highest-priority exploration "
    f"targets for Au–REE co-product systems. Dual-anomaly sites warrant "
    f"follow-up sampling and petrographic characterisation. All results are "
    f"based on NURE stream sediment data and should be confirmed with modern "
    f"analytical methods before resource decisions are made."
)
lines.append("")
lines.append("THRESHOLDS USED (mean + 2σ of log10 positive values):")
for el, thr in thresholds.items():
    lines.append(f"  {el}: {thr:.4g}" if thr is not None else f"  {el}: N/A")

notes_path = 'outputs/text/task7_au_anomaly_notes.txt'
with open(notes_path, 'w') as f:
    f.write('\n'.join(lines))
print(f"  Saved: {notes_path}")

# ── Final console summary ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("TASK 7 COMPLETE — SUMMARY")
print("="*60)
print(f"\nAu units warning triggered: {au_units_warning}")
print("\nClassification counts:")
print(counts.to_string())
print("\nFull summary table:")
print(summary_df.to_string(index=False))
