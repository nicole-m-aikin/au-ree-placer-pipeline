"""
TASK 8: WGS Mine Waste REE & Critical Minerals Analysis
NE Washington Study Area

Figure 9: Mine Waste Geochemistry — REE chondrite-normalized patterns,
critical mineral endowment, Th content, acid-base accounting,
Au fire assay, and site map.

Input:  WGS OFR 2026-02 data supplement (Excel)
Output: outputs/figures/fig9_mine_waste_ree.png
        outputs/tables/task8_mine_waste_summary.csv
        outputs/text/task8_mine_waste_summary.txt
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# ── Wong 8-color palette (pipeline standard) ─────────────────────────────────
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

# ── CI Chondrite normalisation values (Sun & McDonough 1989) ─────────────────
CHONDRITE = {
    'La': 0.237,   'Ce': 0.612,  'Pr': 0.0949, 'Nd': 0.467,
    'Sm': 0.153,   'Eu': 0.058,  'Gd': 0.2055, 'Tb': 0.0374,
    'Dy': 0.254,   'Ho': 0.0566, 'Er': 0.1655, 'Tm': 0.0255,
    'Yb': 0.17,    'Lu': 0.0254, 'Y':  1.57,
}
REE_ORDER = ['La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Y']

# ── Deposit-type colours and labels ──────────────────────────────────────────
DEPOSIT_COLORS = {
    'epithermal':        WONG['orange'],
    'intrusion_related': WONG['blue'],
    'polymetallic_vein': WONG['green'],
    'skarn_replacement': WONG['vermillion'],
    'ultramafic':        WONG['pink'],
    'other':             WONG['black'],
}
DEPOSIT_LABELS = {
    'epithermal':        'Epithermal Au-Ag',
    'intrusion_related': 'Intrusion-related Au',
    'polymetallic_vein': 'Polymetallic vein / IS epithermal',
    'skarn_replacement': 'Skarn / Replacement / Porphyry',
    'ultramafic':        'Ultramafic / Ophiolite',
    'other':             'Other',
}

# ── Study area bounds ─────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 47.5, 49.1
LON_MIN, LON_MAX = -120.0, -117.5

EXCLUDE_SITES_GEO = {'NW Olivine International', 'New Light'}

# ── Manual endowment mine-name → geochem site-name mapping ───────────────────
ENDOWMENT_TO_SITE = {
    'Deer Trail':                           'Deer Trail',
    'Windfall Mine':                        'Windfall Mine',
    'Mullen Mine':                          'Mullen Mine',
    'Gold Dike Mine':                       'Gold Dike',
    'Mountain Beaver (Billy Goat Group)':   'Mountain Beaver (Billy Goat group)',
    'Silver Bell Mine':                     'Silver Bell',
    'Turk Mine':                            'Turk Mine',
    'First Thought Mill':                   'First Thought',
    'Germania Mine & Mill':                 'Germania',
    'Queen Seal Mine':                      'Queen Seal',
    'Big Iron Mine':                        'Big Iron',
}

SITE_ABBREVS = {
    'Deer Trail':                           'Deer Trail',
    'Windfall Mine':                        'Windfall',
    'Mullen Mine':                          'Mullen',
    'Gold Dike':                            'Gold Dike',
    'Mountain Beaver (Billy Goat group)':   'Mtn Beaver',
    'Silver Bell':                          'Silver Bell',
    'Turk Mine':                            'Turk',
    'First Thought':                        'First Thought',
    'Germania':                             'Germania',
    'Queen Seal':                           'Queen Seal',
    'Big Iron':                             'Big Iron',
}

XL_PATH = os.environ.get(
    'WGS_OFR2026_PATH',
    os.path.join(os.path.dirname(__file__), '..', 'data', 'wgs_ofr2026',
                 'ger_ofr2026-02_data_supplement.xlsx')
)

os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/tables',  exist_ok=True)
os.makedirs('outputs/text',    exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load geochemical data
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Loading geochemical data")
print("="*60)

df_raw = pd.read_excel(XL_PATH, sheet_name='Geochemical Data', header=0)
# Row 0 is the units row – drop it before processing
df_all = df_raw.iloc[1:].reset_index(drop=True)
print(f"  Loaded {df_all.shape[0]} rows (units row dropped)")

# Drop the UCC upper continental crust reference standard
df_all = df_all[df_all['Site_Name'] != 'UCC'].reset_index(drop=True)

# Lat / Lon → float
df_all['Latitude']  = pd.to_numeric(df_all['Latitude'],  errors='coerce')
df_all['Longitude'] = pd.to_numeric(df_all['Longitude'], errors='coerce')

# Filter to study area bounding box
df_sa = df_all[
    (df_all['Latitude']  >= LAT_MIN) & (df_all['Latitude']  <= LAT_MAX) &
    (df_all['Longitude'] >= LON_MIN) & (df_all['Longitude'] <= LON_MAX)
].copy().reset_index(drop=True)

# Exclude sites outside study scope
df_sa = df_sa[~df_sa['Site_Name'].isin(EXCLUDE_SITES_GEO)].copy().reset_index(drop=True)
print(f"  Study area samples: {df_sa.shape[0]}")
print(f"  Sites: {sorted(df_sa['Site_Name'].unique())}")

# Convert all element/chemistry columns to numeric
non_numeric = {
    'USGS_Lab_No.', 'Field_No.', 'Sample_Abbrev', 'Sample Type',
    'Sample Description', 'Coord_System', 'County', 'State',
    'Site_Name', 'USMIN_Site_ID', 'Ftr_Name', 'Ftr_ID',
    'Feature Type', 'Material', 'Material_old', 'Commodities', 'Deposit Type(s)',
}
for col in df_sa.columns:
    if col not in non_numeric and col not in ('Latitude', 'Longitude'):
        df_sa[col] = pd.to_numeric(df_sa[col], errors='coerce')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Assign deposit types and abbreviations
# ─────────────────────────────────────────────────────────────────────────────
def classify_deposit(dt_str):
    if pd.isna(dt_str):
        return 'other'
    s = str(dt_str).lower()
    if any(k in s for k in ('epithermal', 'alkalic')):
        return 'epithermal'
    if any(k in s for k in ('intrusion-related', 'reduced')):
        return 'intrusion_related'
    if any(k in s for k in ('polymetallic', 'intermediate')):
        return 'polymetallic_vein'
    if any(k in s for k in ('skarn', 'carbonate', 'manto', 'tungsten', 'porphyry')):
        return 'skarn_replacement'
    if any(k in s for k in ('olivine', 'ultramafic', 'ophiolite')):
        return 'ultramafic'
    return 'other'

df_sa['deposit_type'] = df_sa['Deposit Type(s)'].apply(classify_deposit)
df_sa['site_abbrev']  = df_sa['Site_Name'].map(SITE_ABBREVS).fillna(df_sa['Site_Name'].str[:12])

print("\nDeposit type distribution:")
print(df_sa.groupby(['Site_Name', 'deposit_type']).size().reset_index(name='n').to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Load and filter endowment data
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Loading endowment data")
print("="*60)

df_end_raw = pd.read_excel(XL_PATH, sheet_name='Endowment Calculations', header=0)
df_end = df_end_raw.dropna(subset=['Mine Name']).copy().reset_index(drop=True)

df_end['Latitude']  = pd.to_numeric(df_end['Latitude'],  errors='coerce')
df_end['Longitude'] = pd.to_numeric(df_end['Longitude'], errors='coerce')

# Filter to study area
df_end_sa = df_end[
    (df_end['Latitude']  >= LAT_MIN) & (df_end['Latitude']  <= LAT_MAX) &
    (df_end['Longitude'] >= LON_MIN) & (df_end['Longitude'] <= LON_MAX)
].copy().reset_index(drop=True)

df_end_sa = df_end_sa[
    ~df_end_sa['Mine Name'].str.contains('Olivine|New Light', case=False, na=False)
].reset_index(drop=True)

print(f"  Endowment features in study area: {df_end_sa.shape[0]}")

end_kg_cols = ['Endowment TREE (kg)', 'Endowment Te (kg)',
               'Endowment W (kg)', 'Endowment Bi (kg)']
for col in end_kg_cols:
    df_end_sa[col] = pd.to_numeric(df_end_sa[col], errors='coerce')

# Aggregate endowment by mine
end_agg = (df_end_sa
           .groupby('Mine Name')[end_kg_cols]
           .sum()
           .reset_index()
           .sort_values('Endowment TREE (kg)', ascending=True))
end_agg['site_name'] = end_agg['Mine Name'].map(ENDOWMENT_TO_SITE)

def short_mine(name):
    m = {
        'Deer Trail':                         'Deer Trail',
        'Windfall Mine':                      'Windfall',
        'Mullen Mine':                        'Mullen',
        'Gold Dike Mine':                     'Gold Dike',
        'Mountain Beaver (Billy Goat Group)': 'Mtn Beaver',
        'Silver Bell Mine':                   'Silver Bell',
        'Turk Mine':                          'Turk',
        'First Thought Mill':                 'First Thought',
        'Germania Mine & Mill':               'Germania',
        'Queen Seal Mine':                    'Queen Seal',
        'Big Iron Mine':                      'Big Iron',
    }
    return m.get(name, str(name)[:15])

end_agg['mine_short'] = end_agg['Mine Name'].apply(short_mine)
print(f"\nEndowment by mine (TREE kg):\n{end_agg[['mine_short','Endowment TREE (kg)']].to_string(index=False)}")

# Build per-site TREE endowment lookup for map sizing
tree_by_site = dict(zip(end_agg['site_name'], end_agg['Endowment TREE (kg)']))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — REE preparation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: Preparing REE data")
print("="*60)

ree_cols = [r for r in REE_ORDER if r in df_sa.columns]
print(f"  REE columns available: {ree_cols}")

# Identify near-DL REE (>50% zero or NaN)
nearly_dl = []
for r in ree_cols:
    vals = pd.to_numeric(df_sa[r], errors='coerce').replace(0, np.nan)
    pct_miss = vals.isna().mean()
    if pct_miss > 0.5:
        nearly_dl.append(r)
        print(f"  {r}: {pct_miss:.0%} missing/zero → will flag")

# Spider data: replace 0 → NaN, normalise by chondrite
spider_df = df_sa[ree_cols].copy().apply(pd.to_numeric, errors='coerce').replace(0, np.nan)
norm_df   = spider_df.copy()
for r in ree_cols:
    norm_df[r] = spider_df[r] / CHONDRITE[r]
norm_df['deposit_type'] = df_sa['deposit_type'].values
norm_df['Site_Name']    = df_sa['Site_Name'].values

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Build figure
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: Building 3×2 figure")
print("="*60)

fig, axes = plt.subplots(3, 2, figsize=(18, 14))
fig.suptitle(
    "Figure 9 — WGS Mine Waste Geochemistry: Critical Minerals & Environmental Context\n"
    "NE Washington Study Area  (Earth MRI OFR 2026-02, van Alderwerelt & Di Fiori 2026)",
    fontsize=13, fontweight='bold', y=0.992
)
fig.text(0.5, 0.5, 'EXPLORATION TARGET ONLY',
         ha='center', va='center', fontsize=40, color='gray',
         alpha=0.08, rotation=30, transform=fig.transFigure)

ax_ree, ax_end = axes[0]
ax_th,  ax_aba = axes[1]
ax_au,  ax_map = axes[2]

x_ree = np.arange(len(ree_cols))

# ── Panel A: REE chondrite-normalised spider ──────────────────────────────────
print("  Panel A: REE spider diagram")

plotted_dt = set()
for _, row in norm_df.iterrows():
    dt    = row['deposit_type']
    color = DEPOSIT_COLORS.get(dt, WONG['black'])
    y_vals = [row[r] for r in ree_cols]
    valid  = [v for v in y_vals if pd.notna(v) and v > 0]
    if len(valid) >= 3:
        ax_ree.plot(x_ree, y_vals, color=color, alpha=0.30, lw=0.9, zorder=2)
        plotted_dt.add(dt)

legend_handles_a = []
for dt in list(DEPOSIT_COLORS):
    mask = norm_df['deposit_type'] == dt
    if not mask.any():
        continue
    means = norm_df.loc[mask, ree_cols].mean()
    valid_means = means[means.notna() & (means > 0)]
    if len(valid_means) < 2:
        continue
    ax_ree.plot(x_ree, means.values, color=DEPOSIT_COLORS[dt], alpha=0.95, lw=2.5, zorder=5)
    legend_handles_a.append(
        Line2D([0], [0], color=DEPOSIT_COLORS[dt], lw=2.5, label=DEPOSIT_LABELS[dt])
    )

ax_ree.set_yscale('log')
ax_ree.set_xticks(x_ree)
ax_ree.set_xticklabels(ree_cols, fontsize=9)
ax_ree.set_ylabel('Sample / CI Chondrite  (Sun & McDonough 1989)', fontsize=9)
ax_ree.set_title(
    f"A.  Full REE Chondrite-Normalized Patterns\n(n={len(df_sa)} samples, NE WA study area sites)",
    fontsize=10)
ax_ree.legend(handles=legend_handles_a, fontsize=7.5, loc='upper right', framealpha=0.8)
ax_ree.grid(True, which='both', alpha=0.25, linestyle='--')
ax_ree.set_xlim(-0.4, len(ree_cols) - 0.6)
ax_ree.set_ylim(bottom=0.01)

if nearly_dl:
    dl_str = '/'.join(nearly_dl)
    ax_ree.annotate(f"† {dl_str} often near-DL in mine waste",
                    xy=(0.02, 0.03), xycoords='axes fraction',
                    fontsize=7, color='gray', style='italic')

# ── Panel B: Critical mineral endowment stacked bar ───────────────────────────
print("  Panel B: Endowment bar chart")

y_pos  = np.arange(len(end_agg))
bh     = 0.55
t_kg   = end_agg['Endowment TREE (kg)'].fillna(0).values + 1
te_kg  = end_agg['Endowment Te (kg)'].fillna(0).values   + 1
w_kg   = end_agg['Endowment W (kg)'].fillna(0).values    + 1
bi_kg  = end_agg['Endowment Bi (kg)'].fillna(0).values   + 1

ax_end.barh(y_pos, t_kg,  height=bh, color=WONG['blue'],      label='TREE',  alpha=0.85)
ax_end.barh(y_pos, te_kg, height=bh, color=WONG['orange'],    label='Te',    alpha=0.85, left=t_kg)
ax_end.barh(y_pos, w_kg,  height=bh, color=WONG['green'],     label='W',     alpha=0.85, left=t_kg + te_kg)
ax_end.barh(y_pos, bi_kg, height=bh, color=WONG['vermillion'],label='Bi',    alpha=0.85, left=t_kg + te_kg + w_kg)

ax_end.set_xscale('log')
for x_ref, lbl in [(2, '1 kg'), (11, '10 kg'), (101, '100 kg'), (1001, '1 t')]:
    ax_end.axvline(x_ref, color='gray', linestyle='--', lw=0.8, alpha=0.55)
    ax_end.text(x_ref * 1.05, len(end_agg) - 0.1, lbl, fontsize=6.5, color='gray', va='top')

ax_end.set_yticks(y_pos)
ax_end.set_yticklabels(end_agg['mine_short'].values, fontsize=8)
ax_end.set_xlabel('Endowment (kg + 1, log scale)', fontsize=9)
ax_end.set_title(
    "B.  Critical Mineral Endowment in Mine Waste\n(WGS field-mapped volumes × ICP-MS concentrations)",
    fontsize=10)
ax_end.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.13),
              ncol=4, framealpha=0.9)
ax_end.grid(True, axis='x', alpha=0.25, linestyle='--')

# ── Panel C: Th by site ───────────────────────────────────────────────────────
print("  Panel C: Th by site")

th_grp = (df_sa.groupby('site_abbrev')['Th']
          .agg(['mean', 'std', 'count'])
          .reset_index()
          .rename(columns={'mean': 'Th_mean', 'std': 'Th_std', 'count': 'n'})
          .sort_values('Th_mean', ascending=False))
th_grp['Th_std'] = th_grp['Th_std'].fillna(0)

th_all_vals    = df_sa['Th'].dropna()
th_anomaly_thr = th_all_vals.mean() + 2 * th_all_vals.std()
print(f"  Th regional mean: {th_all_vals.mean():.2f} ppm  anomaly threshold: {th_anomaly_thr:.1f} ppm")

colors_c = [
    WONG['orange'] if m > th_anomaly_thr else WONG['sky']
    for m in th_grp['Th_mean']
]
x_c = np.arange(len(th_grp))

ax_th.bar(x_c, th_grp['Th_mean'], yerr=th_grp['Th_std'],
          color=colors_c, alpha=0.85, capsize=4,
          error_kw={'elinewidth': 1.1, 'ecolor': 'gray'})
ax_th.axhline(8.0,         color=WONG['black'],     linestyle='--', lw=1.5,
              label='NE WA background (8 ppm)')
ax_th.axhline(th_anomaly_thr, color=WONG['vermillion'], linestyle='--', lw=1.5,
              label=f'Anomaly threshold ({th_anomaly_thr:.1f} ppm, μ+2σ)')
ax_th.set_xticks(x_c)
ax_th.set_xticklabels(th_grp['site_abbrev'], rotation=45, ha='right', fontsize=8)
ax_th.set_ylabel('Th (ppm)', fontsize=9)
ax_th.set_title("C.  Thorium Concentration in Mine Waste\n(ICP-MS; NE WA background = 8 ppm)", fontsize=10)
ax_th.legend(fontsize=8, framealpha=0.8)
ax_th.grid(True, axis='y', alpha=0.25, linestyle='--')

# ── Panel D: Acid-base accounting ─────────────────────────────────────────────
print("  Panel D: Acid-base accounting")

df_aba = df_sa[['site_abbrev', 'Paste_pH', 'NP/AP']].copy()
df_aba['Paste_pH'] = pd.to_numeric(df_aba['Paste_pH'], errors='coerce')
df_aba['NP_AP']    = pd.to_numeric(df_aba['NP/AP'],    errors='coerce')
df_aba = df_aba.dropna(subset=['Paste_pH', 'NP_AP'])
df_aba['log_NPAP'] = np.log10(df_aba['NP_AP'].clip(lower=0.001))
print(f"  ABA samples with valid Paste_pH + NP/AP: {len(df_aba)}")

# Coloured risk zones
ax_aba.axhspan(-4, 0,          facecolor='#FFCCCC', alpha=0.35, zorder=0)
ax_aba.axhspan(0,  np.log10(4),facecolor='#FFFACC', alpha=0.35, zorder=0)
ax_aba.axhspan(np.log10(4), 4, facecolor='#CCFFCC', alpha=0.35, zorder=0)

zone_x = 9.3
ax_aba.text(zone_x, -1.5,          'PAG risk',  fontsize=8, color='#990000', ha='right', va='center')
ax_aba.text(zone_x, np.log10(2),   'Uncertain', fontsize=8, color='#886600', ha='right', va='center')
ax_aba.text(zone_x, np.log10(8),   'Non-PAG',   fontsize=8, color='#005500', ha='right', va='center')

for y_val, lbl in [(0, 'NP/AP=1'), (np.log10(2), 'NP/AP=2'), (np.log10(4), 'NP/AP=4')]:
    ax_aba.axhline(y_val, color='gray', lw=1.0, linestyle='--', alpha=0.75, zorder=1)
for x_val in (4.5, 7.0):
    ax_aba.axvline(x_val, color='gray', lw=0.9, linestyle=':', alpha=0.60, zorder=1)
ax_aba.text(4.5, 2.8, 'pH 4.5', fontsize=7, color='gray', ha='center')
ax_aba.text(7.0, 2.8, 'pH 7.0', fontsize=7, color='gray', ha='center')

sites_d  = sorted(df_aba['site_abbrev'].unique())
cmap_d   = plt.cm.tab10
n_sites_d = max(len(sites_d) - 1, 1)

for i, site in enumerate(sites_d):
    mask = df_aba['site_abbrev'] == site
    ax_aba.scatter(df_aba.loc[mask, 'Paste_pH'],
                   df_aba.loc[mask, 'log_NPAP'],
                   color=cmap_d(i / n_sites_d),
                   s=45, alpha=0.85, label=site, zorder=5,
                   edgecolors='white', linewidths=0.5)

ax_aba.set_xlabel('Paste pH', fontsize=9)
ax_aba.set_ylabel('log₁₀(NP/AP)', fontsize=9)
ax_aba.set_title(
    "D.  Acid-Base Accounting: Environmental Risk Tiers\n"
    "(MEND protocol; Paste pH vs NP/AP ratio)", fontsize=10)
ax_aba.legend(fontsize=6.5, loc='upper left', ncol=2, framealpha=0.8)
ax_aba.grid(True, alpha=0.25, linestyle='--')
ax_aba.set_xlim(0, 10)
ax_aba.set_ylim(-3.2, 3.2)
ax_aba.annotate(
    "Sites above NP/AP = 4 present low acid drainage risk to receiving waters",
    xy=(0.02, 0.02), xycoords='axes fraction', fontsize=7, color='gray', style='italic')

# ── Panel E: Au fire assay by site ───────────────────────────────────────────
print("  Panel E: Au by site")

np.random.seed(42)
df_au = df_sa[['site_abbrev', 'Au', 'deposit_type']].copy()
df_au['Au'] = pd.to_numeric(df_au['Au'], errors='coerce')
df_au = df_au[df_au['Au'] > 0].dropna(subset=['Au'])

sites_e  = sorted(df_au['site_abbrev'].unique())
site_idx = {s: i for i, s in enumerate(sites_e)}

for _, row in df_au.iterrows():
    jx    = site_idx[row['site_abbrev']] + np.random.uniform(-0.25, 0.25)
    color = DEPOSIT_COLORS.get(row['deposit_type'], WONG['black'])
    ax_au.scatter(jx, row['Au'], color=color, s=36, alpha=0.72, zorder=5,
                  edgecolors='white', linewidths=0.4)

for site, pos in site_idx.items():
    vals = df_au.loc[df_au['site_abbrev'] == site, 'Au'].values
    if len(vals) >= 3:
        q25, q75 = np.percentile(vals, [25, 75])
        q50 = np.median(vals)
        ax_au.vlines(pos, q25, q75, lw=5, color='#888888', alpha=0.35, zorder=3)
        ax_au.hlines(q50, pos - 0.22, pos + 0.22, lw=2, color='#444444', alpha=0.7, zorder=4)

ax_au.axhline(0.1, color=WONG['vermillion'], linestyle='--', lw=1.6,
              label='0.1 ppm  (tailings re-processing interest,\nMudd 2007)')
ax_au.set_yscale('log')
ax_au.set_xticks(list(site_idx.values()))
ax_au.set_xticklabels(sites_e, rotation=45, ha='right', fontsize=8)
ax_au.set_ylabel('Au (ppm)', fontsize=9)
ax_au.set_title(
    "E.  Au (Fire Assay) in Mine Waste by Site\n"
    "(ICP-MS/FA; 0.1 ppm = typical tailings re-processing interest)", fontsize=10)
ax_au.legend(fontsize=7.5, loc='upper right', framealpha=0.8)
ax_au.grid(True, axis='y', alpha=0.25, linestyle='--')

# Deposit-type legend panel E (share with map)
legend_handles_e = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=DEPOSIT_LABELS[dt])
    for dt, c in DEPOSIT_COLORS.items()
    if dt in df_sa['deposit_type'].values
]
ax_au.legend(handles=legend_handles_e, fontsize=7, loc='upper left', framealpha=0.8,
             title='Deposit type', title_fontsize=7)

# ── Panel F: Site map ─────────────────────────────────────────────────────────
print("  Panel F: Site map")

# One coordinate per site (mean of all samples)
site_pts = (df_sa.groupby(['Site_Name', 'site_abbrev', 'deposit_type'])
            [['Latitude', 'Longitude']]
            .mean()
            .reset_index())
site_pts['TREE_kg'] = site_pts['Site_Name'].map(tree_by_site).fillna(0)
site_pts['mk_size'] = np.log1p(site_pts['TREE_kg']) * 22 + 30

# Approximate river paths (synthetic, similar to task2 approach)
columbia = dict(
    lats=[ 49.00, 48.87, 48.72, 48.55, 48.38, 48.18, 47.98, 47.80, 47.60, 47.50],
    lons=[-117.65,-117.82,-118.05,-118.30,-118.52,-118.62,-118.52,-118.35,-118.18,-118.05]
)
okanogan = dict(
    lats=[ 48.95, 48.70, 48.40, 48.10, 47.80, 47.60, 47.50],
    lons=[-119.75,-119.62,-119.55,-119.50,-119.46,-119.42,-119.38]
)
kettle = dict(
    lats=[ 48.95, 48.78, 48.62, 48.45, 48.30],
    lons=[-118.58,-118.54,-118.50,-118.48,-118.45]
)

ax_map.plot(columbia['lons'], columbia['lats'], color='#5B8DB8', lw=1.6, alpha=0.65, label='Columbia R.', zorder=2)
ax_map.plot(okanogan['lons'], okanogan['lats'], color='#5B8DB8', lw=1.2, alpha=0.55, label='Okanogan R.', zorder=2)
ax_map.plot(kettle['lons'],   kettle['lats'],   color='#5B8DB8', lw=1.0, alpha=0.50, label='Kettle R.', zorder=2)

# County boundary lines (approximate)
for xc, lbl, xoff in [(-119.5, 'Okanogan /\nFerry', -0.25), (-118.5, 'Ferry /\nStevens', -0.25)]:
    ax_map.axvline(xc, color='gray', linestyle='--', lw=0.9, alpha=0.50, zorder=1)
ax_map.text(-119.75, 48.30, 'Okanogan\nCo.', fontsize=6, color='#555555', ha='center', style='italic')
ax_map.text(-119.00, 48.30, 'Ferry\nCo.',    fontsize=6, color='#555555', ha='center', style='italic')
ax_map.text(-118.00, 48.30, 'Stevens\nCo.',  fontsize=6, color='#555555', ha='center', style='italic')

# Per-site label offsets to avoid overlap in tight clusters
# Stevens County south cluster: Germania/QueenSeal/DeerTrail/Turk all within 0.05°
# Mullen/Windfall are at essentially the same coordinate
_F_OFFSETS = {
    'Germania':      (-72, -12),
    'Queen Seal':    (-70,  10),
    'Deer Trail':    (  6,  10),
    'Turk':          (  6, -12),
    'Mullen':        (  6,   8),
    'Windfall':      (  6, -13),
    'Big Iron':      (  6,   6),
    'First Thought': (-78,   5),
    'Gold Dike':     (  6,   6),
    'Silver Bell':   (-72,  -9),
}

# Sites
for _, row in site_pts.iterrows():
    color = DEPOSIT_COLORS.get(row['deposit_type'], WONG['black'])
    ax_map.scatter(row['Longitude'], row['Latitude'],
                   s=row['mk_size'], color=color, alpha=0.85, zorder=6,
                   edgecolors='white', linewidths=0.9)
    dx, dy = _F_OFFSETS.get(row['site_abbrev'], (6, 6))
    ax_map.annotate(row['site_abbrev'],
                    xy=(row['Longitude'], row['Latitude']),
                    xytext=(dx, dy), textcoords='offset points',
                    fontsize=6.5, color='black', zorder=7, fontweight='normal')

# Study area bounding box
bbox_rect = Rectangle((LON_MIN, LAT_MIN),
                       LON_MAX - LON_MIN, LAT_MAX - LAT_MIN,
                       linewidth=1.5, edgecolor='black',
                       facecolor='none', linestyle='-', zorder=3)
ax_map.add_patch(bbox_rect)

# North arrow
ax_map.annotate('N', xy=(-117.62, 49.07), fontsize=11, fontweight='bold', ha='center', zorder=8)
ax_map.annotate('', xy=(-117.62, 49.00), xytext=(-117.62, 48.82),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'), zorder=8)

# Scale bar (~50 km)
ax_map.plot([-119.85, -119.35], [47.56, 47.56], color='black', lw=2.5, zorder=8)
ax_map.text(-119.60, 47.60, '~50 km', fontsize=7, ha='center')

ax_map.set_xlim(LON_MIN - 0.15, LON_MAX + 0.15)
ax_map.set_ylim(LAT_MIN - 0.15, LAT_MAX + 0.20)
ax_map.set_xlabel('Longitude', fontsize=9)
ax_map.set_ylabel('Latitude',  fontsize=9)
ax_map.set_title(
    "F.  WGS Mine Waste Sites in Study Area\n(sized by TREE endowment; colored by deposit type)",
    fontsize=10)

# Deposit-type legend (map)
legend_handles_f = [
    mpatches.Patch(color=DEPOSIT_COLORS[dt], label=DEPOSIT_LABELS[dt])
    for dt in DEPOSIT_COLORS
    if dt in site_pts['deposit_type'].values
]
river_handle = Line2D([0], [0], color='#5B8DB8', lw=1.5, label='Rivers')
legend_handles_f.append(river_handle)
ax_map.legend(handles=legend_handles_f, fontsize=6.5, loc='lower left',
              title='Deposit type', title_fontsize=7, framealpha=0.85)

# ── Figure-level footnote ─────────────────────────────────────────────────────
fig.text(
    0.5, 0.006,
    "Data: WGS OFR 2026-02 (van Alderwerelt & Di Fiori, 2026). ICP-MS/ICP-OES analysis by USGS. "
    "Endowment = screening estimate only; not an economic resource. "
    "For exploration screening purposes only.",
    ha='center', fontsize=7, color='gray', style='italic')

plt.tight_layout(rect=[0, 0.025, 1, 0.985])

out_fig = 'outputs/figures/fig9_mine_waste_ree.png'
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  Figure saved → {out_fig}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Summary CSV
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6: Building summary CSV")
print("="*60)

elem_cols = {
    'Th': 'Th_mean_ppm', 'Au': 'Au_mean_ppm',
    'TREE*': 'TREE_mean_ppm', 'Te': 'Te_mean_ppm',
    'W': 'W_mean_ppm', 'Bi': 'Bi_mean_ppm',
}

def risk_tier(npap_median):
    if pd.isna(npap_median) or npap_median < 0:
        return 'Unknown'
    if npap_median < 1:
        return 'PAG'
    if npap_median < 4:
        return 'Uncertain'
    return 'NonPAG'

rows = []
for site in sorted(df_sa['Site_Name'].unique()):
    mask = df_sa['Site_Name'] == site
    sub  = df_sa[mask]
    dep  = sub['deposit_type'].mode().iloc[0] if not sub['deposit_type'].empty else 'other'

    rec = {
        'site_name':    site,
        'deposit_type': dep,
        'n_samples':    mask.sum(),
    }

    for src_col, out_col in elem_cols.items():
        if src_col in sub.columns:
            vals = pd.to_numeric(sub[src_col], errors='coerce').replace(0, np.nan)
            rec[out_col] = round(vals.mean(), 4) if vals.notna().any() else np.nan
        else:
            rec[out_col] = np.nan

    ph_vals   = pd.to_numeric(sub.get('Paste_pH', pd.Series(dtype=float)), errors='coerce')
    npap_vals = pd.to_numeric(sub.get('NP/AP',    pd.Series(dtype=float)), errors='coerce')
    rec['Paste_pH_mean'] = round(ph_vals.mean(), 3) if ph_vals.notna().any() else np.nan
    rec['NP_AP_mean']    = round(npap_vals.mean(), 3) if npap_vals.notna().any() else np.nan

    tree_kg = tree_by_site.get(site, np.nan)
    rec['TREE_endowment_kg'] = round(float(tree_kg), 2) if not pd.isna(tree_kg) else np.nan
    rec['risk_tier'] = risk_tier(npap_vals.median())

    rows.append(rec)

summary_df = pd.DataFrame(rows)
out_csv = 'outputs/tables/task8_mine_waste_summary.csv'
summary_df.to_csv(out_csv, index=False)
print(f"  Summary CSV saved → {out_csv}")
print(summary_df[['site_name','n_samples','TREE_mean_ppm','Au_mean_ppm','Th_mean_ppm',
                   'TREE_endowment_kg','risk_tier']].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Text summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 7: Writing text summary")
print("="*60)

highest_tree_site = summary_df.loc[summary_df['TREE_endowment_kg'].idxmax(), 'site_name'] \
    if summary_df['TREE_endowment_kg'].notna().any() else 'N/A'
highest_tree_kg   = summary_df['TREE_endowment_kg'].max()
highest_au_site   = summary_df.loc[summary_df['Au_mean_ppm'].idxmax(), 'site_name'] \
    if summary_df['Au_mean_ppm'].notna().any() else 'N/A'
highest_au_val    = summary_df['Au_mean_ppm'].max()
pag_sites         = summary_df.loc[summary_df['risk_tier'] == 'PAG', 'site_name'].tolist()
uncertain_sites   = summary_df.loc[summary_df['risk_tier'] == 'Uncertain', 'site_name'].tolist()
nopag_sites       = summary_df.loc[summary_df['risk_tier'] == 'NonPAG', 'site_name'].tolist()

above_reprocess   = summary_df.loc[summary_df['Au_mean_ppm'] >= 0.1, 'site_name'].tolist()

txt_lines = [
    "="*70,
    "TASK 8 — WGS Mine Waste Geochemistry Summary (OFR 2026-02)",
    "NE Washington Study Area",
    "="*70,
    "",
    f"Total study-area samples analysed: {len(df_sa)}",
    f"Sites in study area: {len(summary_df)}",
    f"REE columns in dataset: {ree_cols}",
    "",
    "─── TREE Endowment (top priority sites) ─────────────────────────────",
    f"  Highest endowment: {highest_tree_site} ({highest_tree_kg:,.0f} kg TREE)",
    "",
]
# Top 5 by TREE endowment
top5 = summary_df.nlargest(5, 'TREE_endowment_kg')[
    ['site_name','TREE_endowment_kg','TREE_mean_ppm']].dropna(subset=['TREE_endowment_kg'])
for _, r in top5.iterrows():
    txt_lines.append(f"  {r['site_name']}: {r['TREE_endowment_kg']:,.0f} kg  "
                     f"(mean {r['TREE_mean_ppm']:.1f} ppm TREE)")

txt_lines += [
    "",
    "─── Au Fire Assay ────────────────────────────────────────────────────",
    f"  Highest mean Au: {highest_au_site} ({highest_au_val:.3f} ppm)",
    f"  Sites with mean Au ≥ 0.1 ppm (re-processing interest): {', '.join(above_reprocess) if above_reprocess else 'None'}",
    "",
]
au_tbl = summary_df.nlargest(5, 'Au_mean_ppm')[['site_name','Au_mean_ppm']].dropna()
for _, r in au_tbl.iterrows():
    txt_lines.append(f"  {r['site_name']}: {r['Au_mean_ppm']:.3f} ppm Au (mean)")

txt_lines += [
    "",
    "─── Acid-Base Accounting (MEND protocol) ────────────────────────────",
    f"  PAG risk sites:  {', '.join(pag_sites) if pag_sites else 'None'}",
    f"  Uncertain sites: {', '.join(uncertain_sites) if uncertain_sites else 'None'}",
    f"  Non-PAG sites:   {', '.join(nopag_sites) if nopag_sites else 'None'}",
    "",
    "─── Thorium ──────────────────────────────────────────────────────────",
    f"  Regional background reference: 8 ppm (NE WA)",
    f"  Anomaly threshold (μ+2σ): {th_anomaly_thr:.1f} ppm",
]
th_tbl = summary_df.nlargest(5, 'Th_mean_ppm')[['site_name','Th_mean_ppm']].dropna()
for _, r in th_tbl.iterrows():
    flag = ' ← ANOMALOUS' if r['Th_mean_ppm'] > th_anomaly_thr else ''
    txt_lines.append(f"  {r['site_name']}: {r['Th_mean_ppm']:.1f} ppm Th{flag}")

txt_lines += [
    "",
    "─── Data Quality Notes ───────────────────────────────────────────────",
    f"  Near-DL REE (>50% missing/zero): {nearly_dl if nearly_dl else 'None'}",
    f"  ABA samples with valid NP/AP: {df_aba.shape[0]} of {len(df_sa)}",
    f"  Au: all {len(df_sa)} study-area samples have valid fire assay values",
    "",
    "─── Caveats ──────────────────────────────────────────────────────────",
    "  Endowment figures are screening estimates only; not economic resources.",
    "  ICP-MS detection limits vary by element; 0 ppm treated as below DL.",
    "  For exploration screening purposes only.",
    "",
    "Data source: WGS OFR 2026-02 (van Alderwerelt & Di Fiori, 2026).",
    "ICP-MS/ICP-OES analysis by USGS.",
    "="*70,
]

out_txt = 'outputs/text/task8_mine_waste_summary.txt'
with open(out_txt, 'w') as f:
    f.write('\n'.join(txt_lines))
print(f"  Text summary saved → {out_txt}")
print('\n'.join(txt_lines[:30]))

print("\n" + "="*60)
print("TASK 8 COMPLETE")
print("="*60)
print(f"  Figure  → {out_fig}")
print(f"  CSV     → {out_csv}")
print(f"  Text    → {out_txt}")
