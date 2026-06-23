"""
TASK 3: Multi-element geochemical discrimination of Th sources
NE Washington NURE stream sediment analysis

Discriminates between:
  - Monazite (LREE-phosphate): Th co-varies with Ce, La, P; low U/Th
  - Thorite/U-Th oxides: Th co-varies with U; low LREE
  - Detrital zircon: Zr high, Th low

Input:  data/nure/nure_wa_synthetic.csv  (replace with real NURE download)
Output: outputs/geojson/nure_classified_th_sources.geojson
        outputs/figures/fig3_geochemical_discrimination.png
        outputs/tables/task3_summary_stats.csv
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig'

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from scipy import stats
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

# ── Load data ────────────────────────────────────────────────────────────────
DATA_PATH = 'data/nure/nure_ne_wa_sediment.csv'
# When real data arrives: replace above with 'data/nure/nure_wa_sediment.csv'
# and verify column names match NURE HSDB export format

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} NURE samples")

# NGDB stores below-detection values as negative numbers (-MDL).
# Treat as NaN — the MDL values are too variable across methods to use half-MDL.
# Exception: small negatives (|val| < 10 ppm) get half-MDL substitution; large
# negatives from high-MDL methods are set to NaN.
# Exclude coordinate and metadata columns from this substitution.
COORD_COLS = {'lat', 'lon', 'lat_orig', 'long_orig', 'depth'}
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in COORD_COLS]
for col in numeric_cols:
    neg_mask = df[col] < 0
    small_neg = neg_mask & (df[col].abs() <= 10)
    large_neg = neg_mask & (df[col].abs() >  10)
    df.loc[small_neg, col] = df.loc[small_neg, col].abs() / 2
    df.loc[large_neg, col] = np.nan

# P is reported in percent in the NGDB bestvalue table; convert to ppm.
if 'P' in df.columns and df['P'].notna().any() and df['P'].dropna().median() < 1:
    df['P'] = df['P'] * 10000   # pct -> ppm

print(f"Available elements: {[c for c in df.columns if c in ['Th','Ce','La','Nd','P','Y','U','Zr','Ti','Fe','Au','As','Cu','Mo']]}")

# ── Define anomaly threshold (mean + 2SD of log-transformed Th) ──────────────
# Use only detected (positive) values; below-detection samples default to BACKGROUND.
th_detected = df['Th'].dropna()
th_detected = th_detected[th_detected > 0]
log_th = np.log10(th_detected)
th_mean_log = log_th.mean()
th_std_log  = log_th.std()
th_threshold = 10 ** (th_mean_log + 2 * th_std_log)
print(f"\nTh anomaly threshold (mean+2SD log, n={len(th_detected)} detected): {th_threshold:.1f} ppm")
df['th_anomaly'] = df['Th'].fillna(0) >= th_threshold

# ── Compute discrimination ratios ─────────────────────────────────────────────
df['Ce_La_ratio'] = df['Ce'] / df['La'].clip(lower=0.1)
df['U_Th_ratio']  = df['U']  / df['Th'].clip(lower=0.01)
df['LREE_sum']    = df['Ce'] + df['La'] + df['Nd']
df['Th_P_corr']   = df['Th'] * df['P']   # proxy for monazite co-enrichment

# Monazite classification criteria (after Mücke & Bhaskara Rao 1996;
# Rasmussen & Muhling 2009):
#   U/Th < 0.5  (monazite is Th-dominant, not U-dominant)
#   Ce > 30 ppm co-occurring with elevated Th
#   Ce/La ratio typically 1.8-2.5 for LREE-enriched monazite
#   P correlates with Th (phosphate mineral)

def classify_th_source(row):
    th = row['Th']
    if pd.isna(th) or th < th_threshold / 2:
        return 'BACKGROUND'
    u_th = row['U_Th_ratio']
    ce   = row['Ce']
    la   = row['La']
    zr   = row['Zr']
    p    = row['P']

    # Monazite: low U/Th + at least one LREE indicator + P (use available cols)
    lree_ok = (pd.notna(ce) and ce > 50) or (pd.notna(la) and la > 20)
    p_ok    = pd.isna(p) or p > 400   # if P is NaN, don't penalize
    if pd.notna(u_th) and u_th < 0.5 and lree_ok and p_ok:
        return 'MONAZITE'
    elif pd.notna(u_th) and u_th > 1.5:
        return 'THORITE_UTHO'     # thorite or U-Th oxide
    elif pd.notna(zr) and zr > 200 and th < th_threshold * 1.5:
        return 'ZIRCON'           # zircon-dominated, low Th/Zr
    else:
        return 'MIXED_UNCLEAR'

df['th_source'] = df.apply(classify_th_source, axis=1)
# Sync th_anomaly with classification: any non-BACKGROUND sample is anomalous
df['th_anomaly'] = df['th_source'] != 'BACKGROUND'

# ── Correlation matrix ────────────────────────────────────────────────────────
elements = ['Th', 'Ce', 'La', 'Nd', 'P', 'Y', 'U', 'Zr', 'Ti', 'Fe']
log_df = np.log10(df[elements].clip(lower=0.01))
corr_matrix = log_df.corr()

# ── Figure 3: Geochemical discrimination plots ────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.suptitle('Figure 3 — Multi-element Geochemical Discrimination of Th Sources\n'
             'NE Washington NURE Stream Sediment Data', fontsize=13, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

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

source_colors = {
    'MONAZITE':      '#009E73',
    'THORITE_UTHO':  '#D55E00',
    'ZIRCON':        '#56B4E9',
    'MIXED_UNCLEAR': '#E69F00',
    'BACKGROUND':    '#CCCCCC'
}
source_labels = {
    'MONAZITE':      'Monazite (Th-LREE-P)',
    'THORITE_UTHO':  'Thorite/U-Th oxide',
    'ZIRCON':        'Zircon-dominated',
    'MIXED_UNCLEAR': 'Mixed/unclear',
    'BACKGROUND':    'Background'
}

anomaly_df = df[df['th_anomaly']].copy()

def scatter_panel(ax, x_col, y_col, xlabel, ylabel, highlight_source='MONAZITE'):
    # Only attempt log-scale if both columns have positive values in the anomaly set
    x_vals = anomaly_df[x_col].dropna()
    y_vals = anomaly_df[y_col].dropna()
    has_data = (x_vals > 0).any() and (y_vals > 0).any()
    if not has_data:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='gray')
        ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
        return
    for src, color in source_colors.items():
        mask = anomaly_df['th_source'] == src
        valid = mask & (anomaly_df[x_col] > 0) & (anomaly_df[y_col] > 0)
        if valid.sum() == 0:
            continue
        ax.scatter(anomaly_df.loc[valid, x_col], anomaly_df.loc[valid, y_col],
                   c=color, alpha=0.7, s=25, label=source_labels[src],
                   edgecolors='black', linewidths=0.3, zorder=2)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(labelsize=9)

# Panel A: Th vs Ce — with literature threshold line
ax1 = fig.add_subplot(gs[0, 0])
scatter_panel(ax1, 'Th', 'Ce', 'Th (ppm)', 'Ce (ppm)')
ax1.set_title('A.  Th vs Ce (monazite: positive correlation)', fontsize=9)
# Ce = 50 ppm threshold (Mücke & Bhaskara Rao 1996; crustal enrichment cutoff)
ax1.axhline(50, color='#0072B2', ls='--', lw=1.2, alpha=0.8)
ax1.text(ax1.get_xlim()[0] * 1.05 if ax1.get_xlim()[0] > 0 else 25,
         52, 'Ce > 50 ppm\n(Mücke & Rao 1996)', fontsize=6, color='#0072B2', va='bottom')
# Add monazite trend line if available
m_mask = anomaly_df['th_source'] == 'MONAZITE'
if m_mask.sum() > 5:
    slope, intercept, r, p, _ = stats.linregress(
        np.log10(anomaly_df.loc[m_mask,'Th']), np.log10(anomaly_df.loc[m_mask,'Ce']))
    x_line = np.logspace(np.log10(anomaly_df['Th'].min()), np.log10(anomaly_df['Th'].max()), 50)
    ax1.plot(x_line, 10**(intercept + slope*np.log10(x_line)), 'b--', lw=1.5, alpha=0.7,
             label=f'Monazite trend (r={r:.2f})')

# Panel B: Th vs La — with literature threshold line
ax2 = fig.add_subplot(gs[0, 1])
scatter_panel(ax2, 'Th', 'La', 'Th (ppm)', 'La (ppm)')
ax2.set_title('B.  Th vs La', fontsize=9)
# La = 20 ppm threshold (Clarke value; above = enriched relative to crust)
ax2.axhline(20, color='#0072B2', ls='--', lw=1.2, alpha=0.8)
ax2.text(ax2.get_xlim()[0] * 1.05 if ax2.get_xlim()[0] > 0 else 25,
         21, 'La > 20 ppm (crustal threshold)', fontsize=6, color='#0072B2', va='bottom')

# Panel C: Th vs P — with literature threshold line
ax3 = fig.add_subplot(gs[0, 2])
scatter_panel(ax3, 'Th', 'P', 'Th (ppm)', 'P (ppm)')
ax3.set_title('C.  Th vs P (monazite: phosphate co-enrichment)', fontsize=9)
# P = 400 ppm threshold (used in classify_th_source; above = phosphate-enriched)
ax3.axhline(400, color='#0072B2', ls='--', lw=1.2, alpha=0.8)
ax3.text(ax3.get_xlim()[0] * 1.05 if ax3.get_xlim()[0] > 0 else 25,
         420, 'P > 400 ppm (monazite proxy\nthreshold, this study)', fontsize=6,
         color='#0072B2', va='bottom')

# Panel D: Th vs U (thorite discrimination) — two threshold lines
ax4 = fig.add_subplot(gs[1, 0])
scatter_panel(ax4, 'Th', 'U', 'Th (ppm)', 'U (ppm)')
x_line = np.logspace(0, 3, 50)
ax4.plot(x_line, 0.5*x_line, color='#0072B2', ls='--', lw=1.5,
         label='U/Th = 0.5 (monazite/thorite boundary)', alpha=0.8)
ax4.plot(x_line, 1.5*x_line, color='#D55E00', ls=':', lw=1.5,
         label='U/Th = 1.5 (thorite zone)', alpha=0.8)
# Shade thorite zone (U/Th > 1.5)
ax4.fill_between(x_line, 1.5*x_line, x_line*200, alpha=0.05, color='#D55E00')
ax4.set_title('D.  Th vs U (thorite: high U/Th)', fontsize=9)
ax4.legend(fontsize=7, loc='upper left')

# Panel E: Correlation matrix heatmap
ax5 = fig.add_subplot(gs[1, 1])
im = ax5.imshow(corr_matrix.values, cmap='PuOr', vmin=-1, vmax=1, aspect='auto')
ax5.set_xticks(range(len(elements))); ax5.set_yticks(range(len(elements)))
ax5.set_xticklabels(elements, rotation=45, ha='right', fontsize=8)
ax5.set_yticklabels(elements, fontsize=8)
plt.colorbar(im, ax=ax5, shrink=0.8)
ax5.set_title('E.  Log-element correlation matrix\n(all NURE samples)', fontsize=9)
# Annotate key r values
for i in range(len(elements)):
    for j in range(len(elements)):
        r_val = corr_matrix.values[i, j]
        if abs(r_val) > 0.5 and i != j:
            ax5.text(j, i, f'{r_val:.2f}', ha='center', va='center',
                     fontsize=6, color='white' if abs(r_val) > 0.75 else 'black')
# Gray out P row/column — P is 66% NaN; Th-P correlation unreliable
p_idx = elements.index('P') if 'P' in elements else None
if p_idx is not None:
    ax5.add_patch(plt.Rectangle((-0.5, p_idx - 0.5), len(elements), 1,
                                 color='gray', alpha=0.30, zorder=3, clip_on=True))
    ax5.add_patch(plt.Rectangle((p_idx - 0.5, -0.5), 1, len(elements),
                                 color='gray', alpha=0.30, zorder=3, clip_on=True))
    # Replot the r-value annotations at higher zorder so they show through the gray
    for i in range(len(elements)):
        for j in [p_idx]:
            r_val = corr_matrix.values[i, j]
            if abs(r_val) > 0.5 and i != j:
                ax5.text(j, i, f'{r_val:.2f}', ha='center', va='center',
                         fontsize=6, color='#777777', zorder=5)
        for j in range(len(elements)):
            if i == p_idx and abs(corr_matrix.values[i, j]) > 0.5 and i != j:
                ax5.text(j, i, f'{corr_matrix.values[i, j]:.2f}', ha='center',
                         va='center', fontsize=6, color='#777777', zorder=5)
    ax5.text(p_idx, len(elements) + 0.2, '†', ha='center', va='bottom',
             fontsize=8, color='#555555', zorder=4)
    ax5.text(0.5, -0.22, '† P: 66% NaN — correlation unreliable',
             transform=ax5.transAxes, fontsize=6.5, color='#555555',
             ha='center', style='italic')

# Panel F: Th source classification map
ax6 = fig.add_subplot(gs[1, 2])
src_sizes    = {'BACKGROUND': 8,  'THORITE_UTHO': 35, 'MIXED_UNCLEAR': 35,
                'ZIRCON': 35, 'MONAZITE': 35}
src_markers  = {'BACKGROUND': 'o', 'THORITE_UTHO': '^', 'MIXED_UNCLEAR': 'o',
                'ZIRCON': 's',     'MONAZITE': 'D'}
src_alphas   = {'BACKGROUND': 0.4, 'THORITE_UTHO': 0.9, 'MIXED_UNCLEAR': 0.9,
                'ZIRCON': 0.9,     'MONAZITE': 0.9}
for src, color in source_colors.items():
    mask = df['th_source'] == src
    if mask.sum() == 0:
        continue
    ax6.scatter(df.loc[mask, 'lon'], df.loc[mask, 'lat'],
                c=color, s=src_sizes.get(src, 18),
                marker=src_markers.get(src, 'o'),
                alpha=src_alphas.get(src, 0.7),
                label=source_labels[src],
                edgecolors='black' if src != 'BACKGROUND' else 'none',
                linewidths=0.3)
ax6.set_xlabel('Longitude', fontsize=11); ax6.set_ylabel('Latitude', fontsize=11)
ax6.set_title('F.  Spatial distribution of Th source types', fontsize=9)
ax6.tick_params(labelsize=9)
ax6.grid(True, alpha=0.2)
# County boundary annotation (schematic)
for lon_line in [-119.5, -118.5, -117.5]:
    ax6.axvline(lon_line, color='gray', lw=0.5, ls='--', alpha=0.4)

# Panel G: U/Th ratio histogram by source
ax7 = fig.add_subplot(gs[2, 0])
for src, color in source_colors.items():
    mask = (df['th_source'] == src) & df['th_anomaly']
    if mask.sum() < 3:
        continue
    vals = np.log10(df.loc[mask, 'U_Th_ratio'].clip(lower=0.001))
    ax7.hist(vals, bins=20, color=color, alpha=0.6, label=source_labels[src], density=True)
ax7.axvline(np.log10(0.5), color='red', ls='--', lw=1.5, label='U/Th=0.5 threshold')
ax7.set_xlabel('log₁₀(U/Th)', fontsize=11); ax7.set_ylabel('Density', fontsize=11)
ax7.set_title('G.  U/Th ratio by Th source type\n(anomalous samples only)', fontsize=9)
ax7.legend(fontsize=8)
ax7.tick_params(labelsize=9)

# Panel H: LREE sum vs Th
ax8 = fig.add_subplot(gs[2, 1])
scatter_panel(ax8, 'Th', 'LREE_sum', 'Th (ppm)', 'Ce+La+Nd (ppm)')
ax8.set_title('H.  Th vs ΣLREE\n(monazite: parallel enrichment)', fontsize=9)

# Panel I: LREE chondrite-normalized pattern — NURE (thin) + WGS mine waste overlay (bold)
# CI chondrite values: Sun & McDonough 1989
ax9 = fig.add_subplot(gs[2, 2])
CHONDRITE = {'La': 0.237, 'Ce': 0.612, 'Nd': 0.467}
spider_els = ['La', 'Ce', 'Nd']
available_spider = [e for e in spider_els if e in anomaly_df.columns]

# --- Load WGS OFR 2026-02 geochemical data (graceful fallback if file absent) ---
_WGS_EXCEL = os.environ.get(
    'WGS_OFR2026_PATH',
    os.path.join(os.path.dirname(__file__), '..', 'data', 'wgs_ofr2026',
                 'ger_ofr2026-02_data_supplement.xlsx')
)
_WGS_STUDY_LAT = (47.5, 49.1)
_WGS_STUDY_LON = (-120.0, -117.5)
_WGS_EXCLUDE   = {'NW Olivine International', 'New Light'}
_WGS_DEP_COLORS = {
    'epithermal':        '#E69F00',
    'intrusion_related': '#0072B2',
    'polymetallic_vein': '#009E73',
    'other':             '#D55E00',
}
wgs_geochem_df = None
try:
    _wraw = pd.read_excel(_WGS_EXCEL, sheet_name='Geochemical Data', header=0)
    _wraw = _wraw.iloc[1:].reset_index(drop=True)   # row 1 = units → skip
    for _c in ['Latitude', 'Longitude', 'La', 'Ce', 'Nd']:
        if _c in _wraw.columns:
            _wraw[_c] = pd.to_numeric(_wraw[_c], errors='coerce')
    _wraw = _wraw[
        _wraw['Latitude'].between(*_WGS_STUDY_LAT) &
        _wraw['Longitude'].between(*_WGS_STUDY_LON)
    ]
    if 'Site_Name' in _wraw.columns:
        _wraw = _wraw[~_wraw['Site_Name'].isin(_WGS_EXCLUDE)]
    for _el in ['La', 'Ce', 'Nd']:
        if _el in _wraw.columns:
            _wraw[_el] = _wraw[_el].replace(0, np.nan)
    wgs_geochem_df = _wraw.reset_index(drop=True)
    print(f"WGS geochemical data loaded: {len(wgs_geochem_df)} rows (Panel I overlay)")
except Exception as _wgs_err:
    print(f"WGS data not available for Panel I overlay (fallback to NURE only): {_wgs_err}")

if len(available_spider) >= 2:
    top_lree = anomaly_df.nlargest(15, 'LREE_sum')
    for src, color in source_colors.items():
        mask = top_lree['th_source'] == src
        sub = top_lree[mask]
        if sub.empty:
            continue
        first = True
        for _, srow in sub.iterrows():
            vals, x_pos = [], []
            for xi, el in enumerate(available_spider):
                v = srow.get(el, np.nan)
                if pd.notna(v) and v > 0 and el in CHONDRITE:
                    vals.append(v / CHONDRITE[el])
                    x_pos.append(xi)
            if len(vals) >= 2:
                ax9.plot(x_pos, vals, color=color, alpha=0.45, lw=1.0,
                         marker='o', markersize=2.5,
                         label=source_labels[src] if first else '_nolegend_')
                first = False

    # WGS bold overlay — mean pattern per deposit type
    if wgs_geochem_df is not None and len(wgs_geochem_df) > 0:
        dep_col = next((c for c in ['Deposit Type(s)', 'Deposit Types', 'Deposit_Type']
                        if c in wgs_geochem_df.columns), None)

        def _dep_key(raw):
            if pd.isna(raw):
                return 'other'
            t = str(raw).lower()
            if 'epithermal' in t:
                return 'epithermal'
            if 'intrusion' in t:
                return 'intrusion_related'
            if 'polymetallic' in t or 'vein' in t:
                return 'polymetallic_vein'
            return 'other'

        wgs_geochem_df = wgs_geochem_df.copy()
        wgs_geochem_df['_dep_key'] = (
            wgs_geochem_df[dep_col].apply(_dep_key) if dep_col
            else 'other'
        )
        _dep_label = {
            'epithermal':        'epithermal',
            'intrusion_related': 'intrusion-related',
            'polymetallic_vein': 'polymetallic vein',
            'other':             'other/skarn/carbonate',
        }
        for _dk, _grp in wgs_geochem_df.groupby('_dep_key'):
            _color = _WGS_DEP_COLORS.get(_dk, '#999999')
            _mean_vals, _x_pos = [], []
            for xi, el in enumerate(available_spider):
                if el in _grp.columns:
                    _normed = pd.to_numeric(_grp[el], errors='coerce') / CHONDRITE[el]
                    _mv = _normed.mean()
                    if pd.notna(_mv) and _mv > 0:
                        _mean_vals.append(_mv)
                        _x_pos.append(xi)
            if len(_mean_vals) >= 2:
                _lbl = f"WGS {_dep_label.get(_dk, _dk)} (mine waste, ICP-MS)"
                ax9.plot(_x_pos, _mean_vals, color=_color, lw=2.5, alpha=0.9,
                         marker='D', markersize=5, label=_lbl, zorder=5)

    ax9.set_xticks(range(len(available_spider)))
    ax9.set_xticklabels(available_spider, fontsize=9)
    ax9.set_yscale('log')
    ax9.set_ylabel('Sample / CI chondrite\n(Sun & McDonough 1989)', fontsize=9)
    ax9.set_title(
        'I.  LREE chondrite-normalized pattern\n'
        '(top 15 NURE anomalies + WGS mine waste overlay)',
        fontsize=9
    )
    ax9.grid(True, alpha=0.3, which='both')
    ax9.tick_params(labelsize=9)
    ax9.legend(fontsize=6.0, loc='lower left', framealpha=0.85)
    ax9.text(
        0.97, 0.97,
        '† WGS bold lines = ICP-MS mine waste (OFR 2026-02)\n'
        '  NURE thin lines = stream sediment (La/Ce/Nd only)\n'
        '  Full REE patterns in Fig 9',
        transform=ax9.transAxes, ha='right', va='top', fontsize=6.5,
        color='#555555', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffbe6',
                  edgecolor='#ccccaa', alpha=0.85)
    )
else:
    ax9.text(0.5, 0.5, 'Insufficient REE columns\nfor spider diagram',
             ha='center', va='center', transform=ax9.transAxes, fontsize=9, color='gray')
    ax9.set_title('I.  LREE pattern (insufficient data)', fontsize=9)

# Legend (common, outside panels)
handles = [plt.scatter([], [], c=c, s=40, label=l) for c, l in zip(source_colors.values(), source_labels.values())]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8,
           bbox_to_anchor=(0.5, 0.01), framealpha=0.9)

fig.text(0.5, 0.01, 'NE Washington REE Tailings Assessment — Au+REE Pipeline Project — EXPLORATION TARGET ONLY',
         ha='center', fontsize=7, color='gray', style='italic')
plt.savefig('outputs/figures/fig3_geochemical_discrimination.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 3 saved")

# ── Summary statistics table ──────────────────────────────────────────────────
summary_rows = []
for src in df['th_source'].unique():
    mask = df['th_source'] == src
    sub = df[mask]
    summary_rows.append({
        'th_source':       src,
        'n_samples':       mask.sum(),
        'pct_of_anomalies': f"{100*mask.sum()/len(df[df['th_anomaly']]):.1f}%" if df['th_anomaly'].sum() > 0 else 'N/A',
        'Th_median_ppm':   round(sub['Th'].median(), 1),
        'Ce_median_ppm':   round(sub['Ce'].median(), 1),
        'La_median_ppm':   round(sub['La'].median(), 1),
        'U_Th_median':     round(sub['U_Th_ratio'].median(), 3),
        'notes':           {
            'MONAZITE':      'Th-LREE-P co-enrichment, low U/Th; monazite interpretation',
            'THORITE_UTHO':  'High U/Th; thorite or U-Th oxide host',
            'ZIRCON':        'High Zr, low Th/Zr; zircon-dominated',
            'MIXED_UNCLEAR': 'Mixed signal; requires SEM/EDS confirmation',
            'BACKGROUND':    'Below anomaly threshold'
        }.get(src, '')
    })

summary_df = pd.DataFrame(summary_rows).sort_values('n_samples', ascending=False)
summary_df.to_csv('outputs/tables/task3_summary_stats.csv', index=False)
print("\nTask 3 classification results:")
print(summary_df[['th_source','n_samples','Th_median_ppm','U_Th_median','notes']].to_string(index=False))

# Key Th-Ce correlation for monazite samples
mono_mask = df['th_source'] == 'MONAZITE'
if mono_mask.sum() > 5:
    r, p = stats.pearsonr(np.log10(df.loc[mono_mask,'Th']), np.log10(df.loc[mono_mask,'Ce']))
    print(f"\nMonazite-classified samples: Th-Ce log-correlation r={r:.3f}, p={p:.2e}")
    r2, p2 = stats.pearsonr(np.log10(df.loc[mono_mask,'Th']), np.log10(df.loc[mono_mask,'P']))
    print(f"Monazite-classified samples: Th-P  log-correlation r={r2:.3f}, p={p2:.2e}")

# ── Export classified GeoJSON ─────────────────────────────────────────────────
gdf = gpd.GeoDataFrame(
    df,
    geometry=[Point(r.lon, r.lat) for r in df.itertuples()],
    crs='EPSG:4326'
)
base_export = ['lab_id', 'lon', 'lat', 'Th', 'Ce', 'La', 'Nd', 'P', 'Y',
               'U', 'Zr', 'Ti', 'Fe', 'th_anomaly', 'U_Th_ratio', 'LREE_sum',
               'th_source', 'geometry']
export_cols = [c for c in base_export if c in gdf.columns]
gdf[export_cols].to_file('outputs/geojson/nure_classified_th_sources.geojson', driver='GeoJSON')
print("\nClassified GeoJSON saved: outputs/geojson/nure_classified_th_sources.geojson")
print("Figure 3 saved:           outputs/figures/fig3_geochemical_discrimination.png")
print("Summary stats saved:      outputs/tables/task3_summary_stats.csv")
