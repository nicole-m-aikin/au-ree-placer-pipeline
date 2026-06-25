"""
Task 8: WGS Mine Waste REE & Critical Minerals Analysis.

Output:
  {outputs_dir}/figures/fig9_mine_waste_ree.png
  {outputs_dir}/tables/task8_mine_waste_summary.csv
  {outputs_dir}/text/task8_mine_waste_summary.txt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

from pipeline.utils import (WONG, CHONDRITE_SUN89, setup_mpl, wgs_path,
                             watermark, save_fig, ensure_outputs, out,
                             map_extent, hillshade, north_arrow, scale_bar,
                             canada_border, locator_inset,
                             topo_contours, rivers_with_arrows,
                             MAP_W, MAP_H, _FIG_LM, _FIG_RM, _FIG_TM, _FIG_BM,
                             _FIG_HGAP, _FIG_VGAP, _ax_rect)

REE_ORDER = ['La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Y']

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


def _classify_deposit(dt_str):
    if pd.isna(dt_str):
        return 'other'
    s = str(dt_str).lower()
    if any(k in s for k in ('epithermal', 'alkalic')): return 'epithermal'
    if any(k in s for k in ('intrusion-related', 'reduced')): return 'intrusion_related'
    if any(k in s for k in ('polymetallic', 'intermediate')): return 'polymetallic_vein'
    if any(k in s for k in ('skarn', 'carbonate', 'manto', 'tungsten', 'porphyry')): return 'skarn_replacement'
    if any(k in s for k in ('olivine', 'ultramafic', 'ophiolite')): return 'ultramafic'
    return 'other'


def _placeholder_fig(cfg):
    """Write a styled placeholder fig9 and empty CSVs when WGS Excel is unavailable."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#f5f5f5')
    fig.patch.set_facecolor('#f5f5f5')
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.1, 0.2), 0.8, 0.6,
                                boxstyle='round,pad=0.02',
                                facecolor='white', edgecolor='#cccccc',
                                linewidth=1.5, transform=ax.transAxes, zorder=2))
    ax.text(0.5, 0.67, 'Figure 9 — WGS Mine Waste REE Analysis',
            ha='center', va='center', fontsize=13, fontweight='bold',
            transform=ax.transAxes, zorder=3)
    ax.text(0.5, 0.55, 'Data not loaded',
            ha='center', va='center', fontsize=11, color='#888888',
            transform=ax.transAxes, zorder=3)
    ax.text(0.5, 0.40,
            'To enable this figure, provide the WGS OFR 2026-02 Excel supplement:\n'
            '  Option 1:  set environment variable  WGS_OFR2026_PATH=/path/to/file.xlsx\n'
            '  Option 2:  set  data.wgs_excel  in the study-area config.yaml',
            ha='center', va='center', fontsize=9, color='#555555',
            linespacing=1.8, transform=ax.transAxes, zorder=3,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0',
                      edgecolor='none', alpha=0.8))
    watermark(fig, cfg)
    save_fig(fig, out(cfg, 'figures', 'fig9_mine_waste_ree.png'))
    pd.DataFrame().to_csv(out(cfg, 'tables', 'task8_mine_waste_summary.csv'), index=False)
    with open(out(cfg, 'text', 'task8_mine_waste_summary.txt'), 'w') as _f:
        _f.write('WGS data not available — set WGS_OFR2026_PATH or data.wgs_excel\n')


def run(cfg):
    setup_mpl()
    ensure_outputs(cfg['outputs_dir'])

    b        = cfg['study_area']['bbox']
    wgs_b    = cfg['study_area'].get('wgs_bbox', b)
    geo      = cfg['geochemistry']
    wgs_cfg  = cfg.get('wgs', {})

    LAT_MIN, LAT_MAX = wgs_b['lat_min'], wgs_b['lat_max']
    LON_MIN, LON_MAX = wgs_b['lon_min'], wgs_b['lon_max']
    EXCLUDE_SITES = set(wgs_cfg.get('exclude_sites', []))
    ENDOWMENT_TO_SITE = wgs_cfg.get('endowment_to_site', {})
    SITE_ABBREVS = wgs_cfg.get('site_abbrevs', {})
    F_OFFSETS = {k: tuple(v) for k, v in wgs_cfg.get('label_offsets', {}).items()}
    REGIONAL_BG_TH = geo.get('regional_bg_th', 8.0)

    XL_PATH = wgs_path(cfg)

    # ── Guard: WGS Excel required for all task8 analysis ─────────────────────
    import os as _os
    if not _os.path.exists(XL_PATH):
        print(f"  Task 8: WGS Excel not found at {XL_PATH}. "
              "Set WGS_OFR2026_PATH or data.wgs_excel in config.")
        _placeholder_fig(cfg)
        return

    # ── STEP 1: Load geochemical data ─────────────────────────────────────────
    print("\nSTEP 1: Loading geochemical data")
    df_raw = pd.read_excel(XL_PATH, sheet_name='Geochemical Data', header=0)
    df_all = df_raw.iloc[1:].reset_index(drop=True)
    df_all = df_all[df_all['Site_Name'] != 'UCC'].reset_index(drop=True)
    df_all['Latitude']  = pd.to_numeric(df_all['Latitude'],  errors='coerce')
    df_all['Longitude'] = pd.to_numeric(df_all['Longitude'], errors='coerce')
    df_sa = df_all[
        df_all['Latitude'].between(LAT_MIN, LAT_MAX) &
        df_all['Longitude'].between(LON_MIN, LON_MAX)
    ].copy().reset_index(drop=True)
    df_sa = df_sa[~df_sa['Site_Name'].isin(EXCLUDE_SITES)].reset_index(drop=True)
    print(f"  Study area samples: {df_sa.shape[0]}, sites: {sorted(df_sa['Site_Name'].unique())}")

    non_numeric = {
        'USGS_Lab_No.','Field_No.','Sample_Abbrev','Sample Type','Sample Description',
        'Coord_System','County','State','Site_Name','USMIN_Site_ID','Ftr_Name','Ftr_ID',
        'Feature Type','Material','Material_old','Commodities','Deposit Type(s)',
    }
    for col in df_sa.columns:
        if col not in non_numeric and col not in ('Latitude','Longitude'):
            df_sa[col] = pd.to_numeric(df_sa[col], errors='coerce')

    # ── STEP 2: Deposit types and abbreviations ───────────────────────────────
    df_sa['deposit_type'] = df_sa['Deposit Type(s)'].apply(_classify_deposit)
    df_sa['site_abbrev']  = df_sa['Site_Name'].map(SITE_ABBREVS).fillna(df_sa['Site_Name'].str[:12])

    # ── STEP 3: Endowment data ────────────────────────────────────────────────
    print("\nSTEP 3: Loading endowment data")
    df_end_raw = pd.read_excel(XL_PATH, sheet_name='Endowment Calculations', header=0)
    df_end = df_end_raw.dropna(subset=['Mine Name']).copy().reset_index(drop=True)
    df_end['Latitude']  = pd.to_numeric(df_end['Latitude'],  errors='coerce')
    df_end['Longitude'] = pd.to_numeric(df_end['Longitude'], errors='coerce')
    df_end_sa = df_end[
        df_end['Latitude'].between(LAT_MIN, LAT_MAX) &
        df_end['Longitude'].between(LON_MIN, LON_MAX)
    ].copy().reset_index(drop=True)
    df_end_sa = df_end_sa[
        ~df_end_sa['Mine Name'].str.contains('Olivine|New Light', case=False, na=False)
    ].reset_index(drop=True)

    # Core endowment columns — always present; extended critical mineral set
    END_BASE_COLS = ['Endowment TREE (kg)', 'Endowment Te (kg)',
                     'Endowment W (kg)', 'Endowment Bi (kg)']
    END_CRIT_COLS = ['Endowment Co (kg)', 'Endowment Li (kg)',
                     'Endowment Ga (kg)', 'Endowment Ge (kg)',
                     'Endowment V (kg)',  'Endowment Ni (kg)']
    end_kg_cols = END_BASE_COLS + END_CRIT_COLS
    # Only keep columns that actually exist in the sheet
    end_kg_cols = [c for c in end_kg_cols if c in df_end_raw.columns]
    for col in end_kg_cols:
        df_end_sa[col] = pd.to_numeric(df_end_sa[col], errors='coerce')

    def _short_mine(name):
        return SITE_ABBREVS.get(ENDOWMENT_TO_SITE.get(name, name), str(name)[:15])

    end_agg = (df_end_sa.groupby('Mine Name')[end_kg_cols].sum().reset_index()
               .sort_values('Endowment TREE (kg)', ascending=True))
    end_agg['site_name']  = end_agg['Mine Name'].map(ENDOWMENT_TO_SITE)
    end_agg['mine_short'] = end_agg['Mine Name'].apply(_short_mine)
    tree_by_site = dict(zip(end_agg['site_name'], end_agg['Endowment TREE (kg)']))

    # ── STEP 4: REE data ──────────────────────────────────────────────────────
    ree_cols = [r for r in REE_ORDER if r in df_sa.columns]
    print(f"  REE columns: {ree_cols}")
    nearly_dl = [r for r in ree_cols
                 if pd.to_numeric(df_sa[r], errors='coerce').replace(0, np.nan).isna().mean() > 0.5]

    spider_df = df_sa[ree_cols].copy().apply(pd.to_numeric, errors='coerce').replace(0, np.nan)
    norm_df   = spider_df.copy()
    for r in ree_cols:
        norm_df[r] = spider_df[r] / CHONDRITE_SUN89[r]
    norm_df['deposit_type'] = df_sa['deposit_type'].values
    norm_df['Site_Name']    = df_sa['Site_Name'].values

    # ── STEP 5: Build figure ──────────────────────────────────────────────────
    # Layout: 3 rows × 2 cols; right column = MAP_W wide; bottom row = MAP_H tall.
    # This guarantees Panel F is the same physical dimensions as every other map
    # panel in the pipeline (MAP_W × MAP_H = 5.5 × 4.2 inches).
    _LEFT_W = 6.5    # data-panel column width
    _TOP_H  = 3.0    # rows A/B and C/D height
    _V9_GAP = 0.85   # vertical gap between rows (larger than _FIG_VGAP=0.55 for 3-row layout)
    _col_l  = _FIG_LM
    _col_r  = _FIG_LM + _LEFT_W + _FIG_HGAP
    _row1_b = _FIG_BM                                               # bottom row (E, F)
    _row2_b = _FIG_BM + MAP_H  + _V9_GAP                           # middle row (C, D)
    _row3_b = _FIG_BM + MAP_H  + _V9_GAP + _TOP_H + _V9_GAP       # top row (A, B)
    figW    = _FIG_LM + _LEFT_W + _FIG_HGAP + MAP_W + _FIG_RM
    figH    = _FIG_TM + _TOP_H  + _V9_GAP  + _TOP_H + _V9_GAP + MAP_H + _FIG_BM

    fig = plt.figure(figsize=(figW, figH))
    fig.suptitle(
        f"Figure 9 — WGS Mine Waste Geochemistry: Critical Minerals & Environmental Context\n"
        f"{cfg['study_area']['name']} Study Area  "
        f"(Earth MRI OFR 2026-02, van Alderwerelt & Di Fiori 2026)",
        fontsize=13, fontweight='bold',
    )
    fig.text(0.5, 0.5, 'EXPLORATION TARGET ONLY', ha='center', va='center',
             fontsize=40, color='gray', alpha=0.08, rotation=30, transform=fig.transFigure)

    ax_ree = fig.add_axes(_ax_rect(_col_l, _row3_b, _LEFT_W, _TOP_H, figW, figH))
    ax_end = fig.add_axes(_ax_rect(_col_r, _row3_b, MAP_W,   _TOP_H, figW, figH))
    ax_th  = fig.add_axes(_ax_rect(_col_l, _row2_b, _LEFT_W, _TOP_H, figW, figH))
    ax_aba = fig.add_axes(_ax_rect(_col_r, _row2_b, MAP_W,   _TOP_H, figW, figH))
    ax_au  = fig.add_axes(_ax_rect(_col_l, _row1_b, _LEFT_W, MAP_H,  figW, figH))
    ax_map = fig.add_axes(_ax_rect(_col_r, _row1_b, MAP_W,   MAP_H,  figW, figH))
    x_ree = np.arange(len(ree_cols))

    # Panel A: REE spider
    plotted_dt = set()
    for _, row in norm_df.iterrows():
        dt = row['deposit_type']
        color = DEPOSIT_COLORS.get(dt, WONG['black'])
        y_vals = [row[r] for r in ree_cols]
        valid = [v for v in y_vals if pd.notna(v) and v > 0]
        if len(valid) >= 3:
            ax_ree.plot(x_ree, y_vals, color=color, alpha=0.30, lw=0.9, zorder=2)
            plotted_dt.add(dt)

    legend_handles_a = []
    for dt in list(DEPOSIT_COLORS):
        mask = norm_df['deposit_type'] == dt
        if not mask.any(): continue
        means = norm_df.loc[mask, ree_cols].mean()
        if (means.notna() & (means > 0)).sum() < 2: continue
        ax_ree.plot(x_ree, means.values, color=DEPOSIT_COLORS[dt], alpha=0.95, lw=2.5, zorder=5)
        legend_handles_a.append(Line2D([0],[0], color=DEPOSIT_COLORS[dt], lw=2.5, label=DEPOSIT_LABELS[dt]))

    ax_ree.set_yscale('log')
    ax_ree.set_xticks(x_ree)
    ax_ree.set_xticklabels(ree_cols, fontsize=9)
    ax_ree.set_ylabel('Sample / CI Chondrite  (Sun & McDonough 1989)', fontsize=9)
    ax_ree.set_title(f"A.  Full REE Chondrite-Normalized Patterns\n(n={len(df_sa)} samples, {cfg['study_area']['name']} study area)", fontsize=10)
    ax_ree.legend(handles=legend_handles_a, fontsize=7.5, loc='upper right', framealpha=0.8)
    ax_ree.grid(True, which='both', alpha=0.25, linestyle='--')
    ax_ree.set_xlim(-0.4, len(ree_cols) - 0.6)
    ax_ree.set_ylim(bottom=0.01)
    if nearly_dl:
        ax_ree.annotate(f"† {'/'.join(nearly_dl)} often near-DL in mine waste",
                        xy=(0.02, 0.03), xycoords='axes fraction', fontsize=7, color='gray', style='italic')

    # Panel B: Expanded critical mineral endowment stacked bar
    # Show TREE + top critical minerals; group by tier for interpretability.
    # Battery/critical: Co, Li, V; Semiconductor/tech: Ga, Ge; then Te, Bi, W baseline.
    _b_display = [
        ('Endowment TREE (kg)', 'TREE',  WONG['blue']),
        ('Endowment Co (kg)',   'Co',    WONG['orange']),
        ('Endowment Li (kg)',   'Li',    WONG['green']),
        ('Endowment V (kg)',    'V',     WONG['sky']),
        ('Endowment Ga (kg)',   'Ga',    WONG['vermillion']),
        ('Endowment Ge (kg)',   'Ge',    WONG['pink']),
        ('Endowment Ni (kg)',   'Ni',    '#8B7355'),
        ('Endowment Te (kg)',   'Te',    WONG['yellow']),
        ('Endowment W (kg)',    'W',     '#888888'),
        ('Endowment Bi (kg)',   'Bi',    '#AAAAAA'),
    ]
    # Only plot columns that exist in the data
    _b_display = [(c, l, col) for c, l, col in _b_display if c in end_agg.columns]

    y_pos = np.arange(len(end_agg))
    bh = 0.55
    bottoms_b = np.ones(len(end_agg))   # start at 1 for log-safe plotting
    for _col, _lbl, _color in _b_display:
        _vals = end_agg[_col].fillna(0).values + 1e-3
        ax_end.barh(y_pos, _vals, height=bh, left=bottoms_b,
                    color=_color, label=_lbl, alpha=0.85)
        bottoms_b = bottoms_b + _vals

    ax_end.set_xscale('log')
    for x_ref, lbl in [(2,'1 kg'),(11,'10 kg'),(101,'100 kg'),(1001,'1 t'),(10001,'10 t')]:
        ax_end.axvline(x_ref, color='gray', linestyle='--', lw=0.8, alpha=0.55)
        ax_end.text(x_ref*1.05, len(end_agg)-0.1, lbl, fontsize=6, color='gray', va='top')
    ax_end.set_yticks(y_pos)
    ax_end.set_yticklabels(end_agg['mine_short'].values, fontsize=8)
    ax_end.set_xlabel('Endowment (kg, log scale)', fontsize=9)
    ax_end.set_title("B.  Critical Mineral Endowment in Mine Waste\n"
                     "(WGS field-mapped volumes × ICP-MS concentrations)", fontsize=10)
    _ncol_b = min(len(_b_display), 5)
    ax_end.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.10),
                  ncol=_ncol_b, framealpha=0.9)
    ax_end.grid(True, axis='x', alpha=0.25, linestyle='--')
    ax_end.annotate(
        "Co/Li/V = battery critical minerals  ·  Ga/Ge = semiconductor/defense critical minerals\n"
        "WGS did not calculate Sc endowment (all samples <5–31 ppm; crustal background, no REE correlation)",
        xy=(0.01, -0.28), xycoords='axes fraction',
        fontsize=6.5, color='gray', style='italic', va='top'
    )

    # Panel C: Th by site
    th_grp = (df_sa.groupby('site_abbrev')['Th'].agg(['mean','std','count']).reset_index()
              .rename(columns={'mean':'Th_mean','std':'Th_std','count':'n'})
              .sort_values('Th_mean', ascending=False))
    th_grp['Th_std'] = th_grp['Th_std'].fillna(0)
    th_all_vals    = df_sa['Th'].dropna()
    th_anomaly_thr = th_all_vals.mean() + 2 * th_all_vals.std()
    colors_c = [WONG['orange'] if m > th_anomaly_thr else WONG['sky'] for m in th_grp['Th_mean']]
    x_c = np.arange(len(th_grp))
    ax_th.bar(x_c, th_grp['Th_mean'], yerr=th_grp['Th_std'], color=colors_c, alpha=0.85, capsize=4,
              error_kw={'elinewidth':1.1,'ecolor':'gray'})
    ax_th.axhline(REGIONAL_BG_TH, color=WONG['black'], linestyle='--', lw=1.5,
                  label=f'{cfg["study_area"]["name"]} background ({REGIONAL_BG_TH} ppm)')
    ax_th.axhline(th_anomaly_thr, color=WONG['vermillion'], linestyle='--', lw=1.5,
                  label=f'Anomaly threshold ({th_anomaly_thr:.1f} ppm, μ+2σ)')
    ax_th.set_xticks(x_c)
    ax_th.set_xticklabels(th_grp['site_abbrev'], rotation=45, ha='right', fontsize=8)
    ax_th.set_ylabel('Th (ppm)', fontsize=9)
    ax_th.set_title(f"C.  Thorium Concentration in Mine Waste\n(ICP-MS; background = {REGIONAL_BG_TH} ppm)", fontsize=10)
    ax_th.legend(fontsize=8, framealpha=0.8)
    ax_th.grid(True, axis='y', alpha=0.25, linestyle='--')

    # Panel D: Acid-base accounting
    df_aba = df_sa[['site_abbrev','Paste_pH','NP/AP']].copy()
    df_aba['Paste_pH'] = pd.to_numeric(df_aba['Paste_pH'], errors='coerce')
    df_aba['NP_AP']    = pd.to_numeric(df_aba['NP/AP'],    errors='coerce')
    df_aba = df_aba.dropna(subset=['Paste_pH','NP_AP'])
    df_aba['log_NPAP'] = np.log10(df_aba['NP_AP'].clip(lower=0.001))
    ax_aba.axhspan(-4, 0,           facecolor='#FFCCCC', alpha=0.35, zorder=0)
    ax_aba.axhspan(0,  np.log10(4), facecolor='#FFFACC', alpha=0.35, zorder=0)
    ax_aba.axhspan(np.log10(4), 4,  facecolor='#CCFFCC', alpha=0.35, zorder=0)
    zone_x = 9.3
    ax_aba.text(zone_x, -1.5,        'PAG risk',  fontsize=8, color='#990000', ha='right', va='center')
    ax_aba.text(zone_x, np.log10(2), 'Uncertain', fontsize=8, color='#886600', ha='right', va='center')
    ax_aba.text(zone_x, np.log10(8), 'Non-PAG',   fontsize=8, color='#005500', ha='right', va='center')
    for y_val in (0, np.log10(2), np.log10(4)):
        ax_aba.axhline(y_val, color='gray', lw=1.0, linestyle='--', alpha=0.75, zorder=1)
    for x_val in (4.5, 7.0):
        ax_aba.axvline(x_val, color='gray', lw=0.9, linestyle=':', alpha=0.60, zorder=1)
    ax_aba.text(4.5, 2.8, 'pH 4.5', fontsize=7, color='gray', ha='center')
    ax_aba.text(7.0, 2.8, 'pH 7.0', fontsize=7, color='gray', ha='center')
    sites_d = sorted(df_aba['site_abbrev'].unique())
    cmap_d  = plt.cm.tab10
    for i, site in enumerate(sites_d):
        mask = df_aba['site_abbrev'] == site
        ax_aba.scatter(df_aba.loc[mask,'Paste_pH'], df_aba.loc[mask,'log_NPAP'],
                       color=cmap_d(i/max(len(sites_d)-1,1)), s=45, alpha=0.85,
                       label=site, zorder=5, edgecolors='white', linewidths=0.5)
    ax_aba.set_xlabel('Paste pH', fontsize=9)
    ax_aba.set_ylabel('log₁₀(NP/AP)', fontsize=9)
    ax_aba.set_title("D.  Acid-Base Accounting: Environmental Risk Tiers\n(MEND protocol; Paste pH vs NP/AP ratio)", fontsize=10)
    ax_aba.legend(fontsize=6.5, loc='upper left', ncol=2, framealpha=0.8)
    ax_aba.grid(True, alpha=0.25, linestyle='--')
    ax_aba.set_xlim(0, 10); ax_aba.set_ylim(-3.2, 3.2)
    ax_aba.annotate("Sites above NP/AP = 4 present low acid drainage risk",
                    xy=(0.02, 0.02), xycoords='axes fraction', fontsize=7, color='gray', style='italic')

    # Panel E: Au fire assay
    np.random.seed(42)
    df_au = df_sa[['site_abbrev','Au','deposit_type']].copy()
    df_au['Au'] = pd.to_numeric(df_au['Au'], errors='coerce')
    df_au = df_au[df_au['Au'] > 0].dropna(subset=['Au'])
    sites_e  = sorted(df_au['site_abbrev'].unique())
    site_idx = {s: i for i, s in enumerate(sites_e)}
    for _, row in df_au.iterrows():
        jx = site_idx[row['site_abbrev']] + np.random.uniform(-0.25, 0.25)
        ax_au.scatter(jx, row['Au'], color=DEPOSIT_COLORS.get(row['deposit_type'], WONG['black']),
                      s=36, alpha=0.72, zorder=5, edgecolors='white', linewidths=0.4)
    for site, pos in site_idx.items():
        vals = df_au.loc[df_au['site_abbrev']==site,'Au'].values
        if len(vals) >= 3:
            q25, q75 = np.percentile(vals, [25,75])
            ax_au.vlines(pos, q25, q75, lw=5, color='#888888', alpha=0.35, zorder=3)
            ax_au.hlines(np.median(vals), pos-0.22, pos+0.22, lw=2, color='#444444', alpha=0.7, zorder=4)
    ax_au.axhline(0.1, color=WONG['vermillion'], linestyle='--', lw=1.6,
                  label='0.1 ppm  (tailings re-processing interest, Mudd 2007)')
    ax_au.set_yscale('log')
    ax_au.set_xticks(list(site_idx.values()))
    ax_au.set_xticklabels(sites_e, rotation=45, ha='right', fontsize=8)
    ax_au.set_ylabel('Au (ppm)', fontsize=9)
    ax_au.set_title("E.  Au (Fire Assay) in Mine Waste by Site\n(ICP-MS/FA; 0.1 ppm = typical tailings re-processing interest)", fontsize=10)
    legend_e = [Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markersize=8, label=DEPOSIT_LABELS[dt])
                for dt, c in DEPOSIT_COLORS.items() if dt in df_sa['deposit_type'].values]
    ax_au.legend(handles=legend_e, fontsize=7, loc='upper left', framealpha=0.8, title='Deposit type', title_fontsize=7)
    ax_au.grid(True, axis='y', alpha=0.25, linestyle='--')

    # Panel F: Site map
    site_pts = (df_sa.groupby(['Site_Name','site_abbrev','deposit_type'])[['Latitude','Longitude']]
                .mean().reset_index())
    site_pts['TREE_kg'] = site_pts['Site_Name'].map(tree_by_site).fillna(0)
    site_pts['mk_size'] = np.log1p(site_pts['TREE_kg']) * 22 + 30

    # Set consistent map extent and add hillshade background.
    xmin_f, xmax_f, ymin_f, ymax_f = map_extent(cfg)
    ax_map.set_xlim(xmin_f, xmax_f)
    ax_map.set_ylim(ymin_f, ymax_f)
    ax_map.set_aspect('auto')
    hillshade(cfg, ax_map, alpha=0.30, zorder=0)
    topo_contours(ax_map, cfg)
    rivers_with_arrows(ax_map, cfg)

    for cl in cfg.get('county_labels', []):
        ax_map.text(cl['lon'], cl['lat'], cl['label'], fontsize=6, color='#555555',
                    ha='center', style='italic')

    for _, row in site_pts.iterrows():
        color = DEPOSIT_COLORS.get(row['deposit_type'], WONG['black'])
        ax_map.scatter(row['Longitude'], row['Latitude'], s=row['mk_size'],
                       color=color, alpha=0.85, zorder=6, edgecolors='white', linewidths=0.9)
        dx, dy = F_OFFSETS.get(row['site_abbrev'], (6, 6))
        ax_map.annotate(row['site_abbrev'], xy=(row['Longitude'], row['Latitude']),
                        xytext=(dx, dy), textcoords='offset points',
                        fontsize=6.5, color='black', zorder=7)

    # Clamp Rectangle corners to map_extent so edges stay flush with the hillshade.
    _rect_x0 = max(LON_MIN, xmin_f)
    _rect_y0 = max(LAT_MIN, ymin_f)
    _rect_x1 = min(LON_MAX, xmax_f)
    _rect_y1 = min(LAT_MAX, ymax_f)
    ax_map.add_patch(Rectangle((_rect_x0, _rect_y0), _rect_x1 - _rect_x0, _rect_y1 - _rect_y0,
                                linewidth=1.5, edgecolor='black', facecolor='none', zorder=3))
    ax_map.set_xlabel('Longitude', fontsize=9)
    ax_map.set_ylabel('Latitude',  fontsize=9)
    ax_map.set_title("F.  WGS Mine Waste Sites in Study Area\n(sized by TREE endowment; colored by deposit type)", fontsize=10)
    legend_f = ([mpatches.Patch(color=DEPOSIT_COLORS[dt], label=DEPOSIT_LABELS[dt])
                 for dt in DEPOSIT_COLORS if dt in site_pts['deposit_type'].values] +
                [Line2D([0],[0], color='#5B8DB8', lw=1.5, label='Rivers')])
    ax_map.legend(handles=legend_f, fontsize=6.5, loc='lower left',
                  title='Deposit type', title_fontsize=7, framealpha=0.85)
    canada_border(ax_map, cfg)
    north_arrow(ax_map, x=0.96, y=0.96, size=9)
    scale_bar(ax_map, cfg, length_km=50, x=0.65, y=0.05)

    fig.text(0.5, _FIG_BM * 0.35 / figH,
             "Data: WGS OFR 2026-02 (van Alderwerelt & Di Fiori, 2026). "
             "ICP-MS/ICP-OES analysis by USGS. Endowment = screening estimate only; not an economic resource. "
             "For exploration screening purposes only.",
             ha='center', fontsize=7, color='gray', style='italic')

    locator_inset(fig, ax_map, cfg)
    save_fig(fig, out(cfg, 'figures', 'fig9_mine_waste_ree.png'))

    # ── STEP 6: Summary CSV ───────────────────────────────────────────────────
    def risk_tier(npap):
        if pd.isna(npap) or npap < 0: return 'Unknown'
        if npap < 1: return 'PAG'
        if npap < 4: return 'Uncertain'
        return 'NonPAG'

    # Build per-site endowment lookup for critical minerals
    _end_site_lookup = {}
    for _, erow in end_agg.iterrows():
        sn = erow.get('site_name')
        if sn:
            _end_site_lookup[sn] = erow

    rows = []
    for site in sorted(df_sa['Site_Name'].unique()):
        mask = df_sa['Site_Name'] == site
        sub  = df_sa[mask]
        dep  = sub['deposit_type'].mode().iloc[0] if len(sub) else 'other'
        ph_vals   = pd.to_numeric(sub.get('Paste_pH', pd.Series(dtype=float)), errors='coerce')
        npap_vals = pd.to_numeric(sub.get('NP/AP',    pd.Series(dtype=float)), errors='coerce')
        _erow = _end_site_lookup.get(site, pd.Series(dtype=float))
        def _end_val(col):
            v = _erow.get(col, np.nan) if len(_erow) else np.nan
            return round(float(v), 2) if pd.notna(v) else np.nan
        rec = {
            'site_name': site, 'deposit_type': dep, 'n_samples': mask.sum(),
            'TREE_mean_ppm': round(pd.to_numeric(sub.get('TREE*', pd.Series(dtype=float)), errors='coerce').replace(0, np.nan).mean(), 4) if 'TREE*' in sub else np.nan,
            'Au_mean_ppm':   round(pd.to_numeric(sub['Au'], errors='coerce').replace(0, np.nan).mean(), 4) if 'Au' in sub else np.nan,
            'Th_mean_ppm':   round(pd.to_numeric(sub['Th'], errors='coerce').replace(0, np.nan).mean(), 4) if 'Th' in sub else np.nan,
            'Paste_pH_mean': round(ph_vals.mean(), 3) if ph_vals.notna().any() else np.nan,
            'NP_AP_mean':    round(npap_vals.mean(), 3) if npap_vals.notna().any() else np.nan,
            'TREE_endowment_kg': round(float(tree_by_site.get(site, np.nan)), 2) if not pd.isna(tree_by_site.get(site, np.nan)) else np.nan,
            'Co_endowment_kg':   _end_val('Endowment Co (kg)'),
            'Li_endowment_kg':   _end_val('Endowment Li (kg)'),
            'Ga_endowment_kg':   _end_val('Endowment Ga (kg)'),
            'Ge_endowment_kg':   _end_val('Endowment Ge (kg)'),
            'V_endowment_kg':    _end_val('Endowment V (kg)'),
            'Ni_endowment_kg':   _end_val('Endowment Ni (kg)'),
            'risk_tier': risk_tier(npap_vals.median()),
        }
        rows.append(rec)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out(cfg, 'tables', 'task8_mine_waste_summary.csv'), index=False)

    # ── STEP 7: Text summary ──────────────────────────────────────────────────
    best_tree = summary_df.loc[summary_df['TREE_endowment_kg'].idxmax(), 'site_name'] if summary_df['TREE_endowment_kg'].notna().any() else 'N/A'
    best_au   = summary_df.loc[summary_df['Au_mean_ppm'].idxmax(), 'site_name'] if summary_df['Au_mean_ppm'].notna().any() else 'N/A'
    pag_sites = summary_df.loc[summary_df['risk_tier']=='PAG','site_name'].tolist()

    txt = [
        "="*70, "TASK 8 — WGS Mine Waste Geochemistry Summary (OFR 2026-02)",
        f"{cfg['study_area']['name']} Study Area", "="*70, "",
        f"Study area samples: {len(df_sa)}, sites: {len(summary_df)}",
        f"REE columns: {ree_cols}", "",
        f"Highest TREE endowment: {best_tree}",
        f"Highest mean Au: {best_au}",
        f"PAG acid-generating sites: {pag_sites if pag_sites else 'None'}",
        "", "Endowment figures are screening estimates only; not economic resources.",
        "Data source: WGS OFR 2026-02 (van Alderwerelt & Di Fiori, 2026).", "="*70,
    ]
    with open(out(cfg, 'text', 'task8_mine_waste_summary.txt'), 'w') as f:
        f.write('\n'.join(txt))

    print("TASK 8 COMPLETE")
    print(summary_df[['site_name','n_samples','TREE_endowment_kg','Au_mean_ppm','risk_tier']].to_string(index=False))


if __name__ == '__main__':
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/ne_washington/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
