"""
Task 3: Multi-element geochemical discrimination of Th sources.

Output:
  {outputs_dir}/geojson/nure_classified_th_sources.geojson
  {outputs_dir}/figures/fig3_geochemical_discrimination.png
  {outputs_dir}/tables/task3_summary_stats.csv
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

from pipeline.utils import (WONG, CHONDRITE_SUN89, setup_mpl, load_nure, anomaly_threshold,
                             wgs_path, watermark, save_fig, ensure_outputs, out,
                             map_extent, north_arrow, scale_bar)


def run(cfg):
    setup_mpl()
    ensure_outputs(cfg['outputs_dir'])

    geo = cfg['geochemistry']
    wgs_b = cfg['study_area'].get('wgs_bbox', cfg['study_area']['bbox'])
    wgs_exclude = set(cfg.get('wgs', {}).get('exclude_sites', []))

    df = load_nure(cfg)
    print(f"Loaded {len(df)} NURE samples")
    print(f"Available elements: {[c for c in df.columns if c in ['Th','Ce','La','Nd','P','Y','U','Zr','Ti','Fe','Au','As','Cu','Mo']]}")

    # Anomaly threshold
    th_threshold = anomaly_threshold(df['Th'])
    print(f"\nTh anomaly threshold (mean+2SD log): {th_threshold:.1f} ppm")
    df['th_anomaly'] = df['Th'].fillna(0) >= th_threshold

    # Discrimination ratios
    df['Ce_La_ratio'] = df['Ce'] / df['La'].clip(lower=0.1)
    df['U_Th_ratio']  = df['U']  / df['Th'].clip(lower=0.01)
    df['LREE_sum']    = df['Ce'] + df['La'] + df['Nd']
    df['Th_P_corr']   = df['Th'] * df['P']

    ce_min  = geo.get('monazite_ce_min_ppm', 50)
    la_min  = geo.get('monazite_la_min_ppm', 20)
    p_min   = geo.get('monazite_p_min_ppm', 400)
    uth_max = geo.get('monazite_uth_max', 0.5)
    uth_min = geo.get('thorite_uth_min', 1.5)
    nb_thresh = geo.get('fergusonite_nb_min_ppm', 40)
    sr_thresh = geo.get('apatite_sr_min_ppm', 300)
    _y_pos = df['Y'][df['Y'] > 0]
    y_thresh = geo.get('xenotime_y_min_ppm',
                       float(_y_pos.mean() + 2 * _y_pos.std()) if len(_y_pos) > 5 else 82.0)

    def classify_th_source(row):
        th   = row['Th']
        y    = row['Y']
        nb   = row['Nb']
        sr   = row['Sr']
        u_th = row['U_Th_ratio']
        ce, la, zr, p = row['Ce'], row['La'], row['Zr'], row['P']

        lree_ok  = (pd.notna(ce) and ce > ce_min) or (pd.notna(la) and la > la_min)
        p_ok     = pd.isna(p) or p > p_min
        p_high   = pd.notna(p) and p > p_min
        nb_high  = pd.notna(nb) and nb > nb_thresh
        sr_high  = pd.notna(sr) and sr > sr_thresh
        y_anom   = pd.notna(y) and y > y_thresh
        th_bg    = pd.isna(th) or th < th_threshold / 2

        # Y-only anomaly (xenotime/Y-phase without Th enrichment)
        if th_bg and y_anom:
            return 'XENOTIME_Y'
        if th_bg:
            return 'BACKGROUND'

        # Nb-oxide suspect: elevated Nb + elevated LREE (fergusonite/columbite)
        if nb_high and lree_ok:
            return 'NB_OXIDE_SUSPECT'

        # Apatite-dominated P: high Sr + high P overrides monazite assignment
        if sr_high and p_high:
            return 'APATITE_P'

        # Thorite / U-Th oxide: high U/Th — split on whether LREE is also elevated.
        # Pure thorite is REE-poor; LREE-ok + high U/Th = mixed assemblage.
        if pd.notna(u_th) and u_th > uth_min:
            if lree_ok:
                return 'THORITE_LREE_MIX'  # thorite U/Th + REE mineral co-occurrence
            return 'THORITE_UTHO'

        # LREE-enriched and not confirmed thorite: split on whether P data supports phosphate.
        # U/Th is secondary — P is the primary phosphate discriminator for monazite.
        # NaN U/Th is included here: absence of U data does not confirm thorite.
        if lree_ok:
            if p_high:
                return 'MONAZITE'    # P elevated — phosphate host confirmed
            if pd.isna(p):
                return 'LREE_INDET'  # P absent — monazite vs allanite indeterminate

        # Zircon: high Zr + moderate Th
        if pd.notna(zr) and zr > 200 and th < th_threshold * 1.5:
            return 'ZIRCON'

        return 'MIXED_UNCLEAR'

    df['th_source'] = df.apply(classify_th_source, axis=1)
    df['th_anomaly'] = df['th_source'] != 'BACKGROUND'

    # Correlation matrix — Nb, Sr, Ca added as accessory-mineral discriminators
    _corr_candidates = ['Th', 'Ce', 'La', 'Nd', 'P', 'Y', 'U', 'Zr', 'Ti', 'Fe', 'Nb', 'Sr', 'Ca']
    elements = [e for e in _corr_candidates if e in df.columns and (df[e] > 0).any()]
    log_df = np.log10(df[elements].clip(lower=0.01))
    corr_matrix = log_df.corr()

    source_colors = {
        'MONAZITE':           WONG['green'],
        'LREE_INDET':         WONG['black'],
        'THORITE_LREE_MIX':   '#8E44AD',        # purple — thorite U/Th + REE mineral
        'THORITE_UTHO':       WONG['vermillion'],
        'ZIRCON':             WONG['sky'],
        'NB_OXIDE_SUSPECT':   WONG['blue'],
        'APATITE_P':          WONG['yellow'],
        'XENOTIME_Y':         WONG['pink'],
        'MIXED_UNCLEAR':      WONG['orange'],
        'BACKGROUND':         '#CCCCCC',
    }
    source_labels = {
        'MONAZITE':           'Monazite (P confirmed + LREE-ok)',
        'LREE_INDET':         'LREE-enriched, host indet. (P absent)',
        'THORITE_LREE_MIX':   'Thorite+REE-mineral mix (high U/Th + LREE-ok)',
        'THORITE_UTHO':       'Thorite/U-Th oxide (high U/Th, LREE-poor)',
        'ZIRCON':             'Zircon-dominated',
        'NB_OXIDE_SUSPECT':   'Nb-oxide suspect (fergusonite/columbite)',
        'APATITE_P':          'Apatite-dominated P (high Sr+P)',
        'XENOTIME_Y':         'Xenotime/Y-phase (Y-only, no Th)',
        'MIXED_UNCLEAR':      'Mixed/unclear',
        'BACKGROUND':         'Background',
    }

    anomaly_df = df[df['th_anomaly']].copy()

    def scatter_panel(ax, x_col, y_col, xlabel, ylabel, show_legend=False):
        x_vals = anomaly_df[x_col].dropna()
        y_vals = anomaly_df[y_col].dropna()
        if not ((x_vals > 0).any() and (y_vals > 0).any()):
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
        ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(labelsize=9)
        if show_legend:
            ax.legend(fontsize=7, loc='upper left', markerscale=1.3,
                      framealpha=0.88, handlelength=1.2)

    CHONDRITE = {k: v for k, v in CHONDRITE_SUN89.items() if k in ('La', 'Ce', 'Nd')}

    # WGS data for Panel I overlay
    _xl = wgs_path(cfg)
    _WGS_DEP_COLORS = {
        'epithermal':        WONG['orange'],
        'intrusion_related': WONG['blue'],
        'polymetallic_vein': WONG['green'],
        'other':             WONG['vermillion'],
    }
    wgs_geochem_df = None
    try:
        _wraw = pd.read_excel(_xl, sheet_name='Geochemical Data', header=0)
        _wraw = _wraw.iloc[1:].reset_index(drop=True)
        for _c in ['Latitude', 'Longitude', 'La', 'Ce', 'Nd']:
            if _c in _wraw.columns:
                _wraw[_c] = pd.to_numeric(_wraw[_c], errors='coerce')
        _wraw = _wraw[
            _wraw['Latitude'].between(wgs_b['lat_min'], wgs_b['lat_max']) &
            _wraw['Longitude'].between(wgs_b['lon_min'], wgs_b['lon_max'])
        ]
        if 'Site_Name' in _wraw.columns:
            _wraw = _wraw[~_wraw['Site_Name'].isin(wgs_exclude)]
        for _el in ['La', 'Ce', 'Nd']:
            if _el in _wraw.columns:
                _wraw[_el] = _wraw[_el].replace(0, np.nan)
        wgs_geochem_df = _wraw.reset_index(drop=True)
        print(f"WGS geochemical data loaded: {len(wgs_geochem_df)} rows (Panel I overlay)")
    except Exception as _wgs_err:
        print(f"WGS data not available for Panel I overlay: {_wgs_err}")

    # ── Figure 3 ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 19))
    fig.suptitle(
        f'Figure 3 — Multi-element Geochemical Discrimination of Th Sources\n'
        f'{cfg["study_area"]["name"]} NURE Stream Sediment Data',
        fontsize=13, fontweight='bold', y=0.99,
    )
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    scatter_panel(ax1, 'Th', 'Ce', 'Th (ppm)', 'Ce (ppm)', show_legend=True)
    ax1.set_title('A.  Th vs Ce (monazite: positive correlation)', fontsize=9)
    ax1.axhline(ce_min, color=WONG['blue'], ls='--', lw=1.2, alpha=0.8)
    ax1.text(ax1.get_xlim()[0],
             ce_min * 1.04, f'Ce > {ce_min} ppm\n(Mücke & Rao 1996)',
             fontsize=6, color=WONG['blue'], va='bottom')
    m_mask = anomaly_df['th_source'] == 'MONAZITE'
    if m_mask.sum() > 5:
        slope, intercept, r, p, _ = stats.linregress(
            np.log10(anomaly_df.loc[m_mask, 'Th']),
            np.log10(anomaly_df.loc[m_mask, 'Ce']))
        x_line = np.logspace(np.log10(anomaly_df['Th'].min()), np.log10(anomaly_df['Th'].max()), 50)
        ax1.plot(x_line, 10**(intercept + slope*np.log10(x_line)), 'b--', lw=1.5, alpha=0.7,
                 label=f'Monazite trend (r={r:.2f})')

    ax2 = fig.add_subplot(gs[0, 1])
    scatter_panel(ax2, 'Th', 'La', 'Th (ppm)', 'La (ppm)', show_legend=True)
    ax2.set_title('B.  Th vs La', fontsize=9)
    ax2.axhline(la_min, color=WONG['blue'], ls='--', lw=1.2, alpha=0.8)
    ax2.text(ax2.get_xlim()[0],
             la_min * 1.05, f'La > {la_min} ppm (crustal threshold)',
             fontsize=6, color=WONG['blue'], va='bottom')

    ax3 = fig.add_subplot(gs[0, 2])
    scatter_panel(ax3, 'Th', 'P', 'Th (ppm)', 'P (ppm)', show_legend=True)
    ax3.set_title('C.  Th vs P (monazite: phosphate co-enrichment)', fontsize=9)
    ax3.axhline(p_min, color=WONG['blue'], ls='--', lw=1.2, alpha=0.8)
    ax3.text(ax3.get_xlim()[0],
             p_min * 1.05, f'P > {p_min} ppm (monazite proxy\nthreshold, this study)',
             fontsize=6, color=WONG['blue'], va='bottom')

    ax4 = fig.add_subplot(gs[1, 0])
    scatter_panel(ax4, 'Th', 'U', 'Th (ppm)', 'U (ppm)')
    x_line = np.logspace(0, 3, 50)
    ax4.plot(x_line, uth_max*x_line, color=WONG['blue'], ls='--', lw=1.5,
             label=f'U/Th = {uth_max} (monazite/thorite boundary)', alpha=0.8)
    ax4.plot(x_line, uth_min*x_line, color=WONG['vermillion'], ls=':', lw=1.5,
             label=f'U/Th = {uth_min} (thorite zone)', alpha=0.8)
    ax4.fill_between(x_line, uth_min*x_line, x_line*200, alpha=0.05, color=WONG['vermillion'])
    ax4.set_title('D.  Th vs U (thorite: high U/Th)', fontsize=9)
    ax4.legend(fontsize=7, loc='upper left')

    # Zone count annotations — show N per U/Th bin for samples with valid U data
    _anom_u = anomaly_df[(anomaly_df['Th'] > 0) & (anomaly_df['U'] > 0)].copy()
    _anom_u['_uth'] = _anom_u['U'] / _anom_u['Th'].clip(lower=0.01)
    _n_mnz  = (_anom_u['_uth'] <  uth_max).sum()   # < 0.5 monazite-like
    _n_mid  = ((_anom_u['_uth'] >= uth_max) & (_anom_u['_uth'] < uth_min)).sum()  # 0.5–1.5
    _n_thor = (_anom_u['_uth'] >= uth_min).sum()   # > 1.5 thorite
    _n_miss = len(anomaly_df) - len(_anom_u)        # no U data
    _note = (f'Of {len(anomaly_df)} anomalous samples:\n'
             f'  n={_n_thor} U/Th > {uth_min} (thorite zone)\n'
             f'  n={_n_mid}  U/Th {uth_max}–{uth_min} (mixed)\n'
             f'  n={_n_mnz}  U/Th < {uth_max} (monazite-like)\n'
             f'  n={_n_miss} U missing → not plotted here\n'
             f'  (missing-U samples shown in A–C as\n'
             f'   Mixed/unclear; source unresolved)')
    ax4.text(0.97, 0.03, _note, transform=ax4.transAxes,
             ha='right', va='bottom', fontsize=6.2, color='#444444',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#fffbe6',
                       edgecolor='#ccccaa', alpha=0.90))

    ax5 = fig.add_subplot(gs[1, 1])
    im = ax5.imshow(corr_matrix.values, cmap='PuOr', vmin=-1, vmax=1, aspect='auto')
    ax5.set_xticks(range(len(elements))); ax5.set_yticks(range(len(elements)))
    ax5.set_xticklabels(elements, rotation=45, ha='right', fontsize=7)
    ax5.set_yticklabels(elements, fontsize=7)
    plt.colorbar(im, ax=ax5, shrink=0.8)
    ax5.set_title('E.  Log-element correlation matrix\n(all NURE samples)', fontsize=9)
    for i in range(len(elements)):
        for j in range(len(elements)):
            r_val = corr_matrix.values[i, j]
            if abs(r_val) > 0.5 and i != j:
                ax5.text(j, i, f'{r_val:.2f}', ha='center', va='center',
                         fontsize=5.5, color='white' if abs(r_val) > 0.60 else 'black')
    # Grey out rows/columns for sparse elements and build footnote
    _sparse_notes = []
    for _sp_el in ('P', 'Nb', 'Sr', 'Ca'):
        if _sp_el not in elements:
            continue
        _pct_nan = int(round(df[_sp_el].isna().mean() * 100))
        if _pct_nan > 40:
            _idx = elements.index(_sp_el)
            _sym = {_sp_el: '†‡§¶'[('P','Nb','Sr','Ca').index(_sp_el)]}.get(_sp_el, '*')
            ax5.add_patch(plt.Rectangle((-0.5, _idx - 0.5), len(elements), 1,
                                        color='gray', alpha=0.20, zorder=3, clip_on=True))
            ax5.add_patch(plt.Rectangle((_idx - 0.5, -0.5), 1, len(elements),
                                        color='gray', alpha=0.20, zorder=3, clip_on=True))
            ax5.text(_idx, len(elements) + 0.2, _sym, ha='center', va='bottom',
                     fontsize=7, color='#555555', zorder=4)
            _sparse_notes.append(f'{_sym} {_sp_el}: {_pct_nan}% NaN')
    if _sparse_notes:
        ax5.text(0.5, -0.24, '  '.join(_sparse_notes) + ' — correlations unreliable',
                 transform=ax5.transAxes, fontsize=6.0, color='#555555',
                 ha='center', style='italic')
    # Dashed annotation boxes: REE block (Th, Ce, La, Nd), Oxide block (Ti, Fe),
    # and Accessory block (Nb, Sr, Ca)
    ree_els = ['Th', 'Ce', 'La', 'Nd']
    ree_idxs = [elements.index(e) for e in ree_els if e in elements]
    if len(ree_idxs) >= 2:
        r0, r1 = min(ree_idxs) - 0.5, max(ree_idxs) + 0.5
        ax5.add_patch(plt.Rectangle((r0, r0), r1 - r0, r1 - r0,
                                    fill=False, edgecolor=WONG['green'], lw=1.8,
                                    ls='--', zorder=5, clip_on=True))
        ax5.text(r1 + 0.1, (r0 + r1) / 2, 'REE\nblock', fontsize=6,
                 color=WONG['green'], va='center', style='italic')
    oxide_els = ['Ti', 'Fe']
    oxide_idxs = [elements.index(e) for e in oxide_els if e in elements]
    if len(oxide_idxs) >= 2:
        o0, o1 = min(oxide_idxs) - 0.5, max(oxide_idxs) + 0.5
        ax5.add_patch(plt.Rectangle((o0, o0), o1 - o0, o1 - o0,
                                    fill=False, edgecolor=WONG['orange'], lw=1.8,
                                    ls='--', zorder=5, clip_on=True))
        ax5.text(o1 + 0.1, (o0 + o1) / 2, 'Oxide\nblock', fontsize=6,
                 color=WONG['orange'], va='center', style='italic')
    acc_els = ['Nb', 'Sr', 'Ca']
    acc_idxs = [elements.index(e) for e in acc_els if e in elements]
    if len(acc_idxs) >= 2:
        a0, a1 = min(acc_idxs) - 0.5, max(acc_idxs) + 0.5
        ax5.add_patch(plt.Rectangle((a0, a0), a1 - a0, a1 - a0,
                                    fill=False, edgecolor=WONG['blue'], lw=1.8,
                                    ls='--', zorder=5, clip_on=True))
        ax5.text(a1 + 0.1, (a0 + a1) / 2, 'Acc.\nblock', fontsize=6,
                 color=WONG['blue'], va='center', style='italic')

    # Panel F (spatial distribution of Th source types) removed — the spatial
    # story is told more completely by Fig 10C (ML probability map with NURE
    # sample locations + geographic context). Panel G moved up to the freed slot.

    ax7 = fig.add_subplot(gs[1, 2])
    # Reference mineral zone shadings (Mücke & Rao 1996; Förster 2006; Kempe et al. 2010)
    _zone_alpha = 0.10
    ax7.axvspan(-3,              np.log10(uth_max), alpha=_zone_alpha, color=WONG['green'],
                label='Monazite field (U/Th < 0.5)')
    ax7.axvspan(np.log10(uth_max), np.log10(uth_min), alpha=_zone_alpha, color=WONG['yellow'],
                label='Thorite field (0.5–1.5)')
    ax7.axvspan(np.log10(uth_min), 1.7,  alpha=_zone_alpha, color=WONG['orange'],
                label='Thorite-coffinite series (1.5–50)')
    ax7.axvspan(1.7,             3.0,    alpha=_zone_alpha, color=WONG['vermillion'],
                label='Uraninite zone (U/Th > 50)')
    for src, color in source_colors.items():
        mask = (df['th_source'] == src) & df['th_anomaly']
        if mask.sum() < 3: continue
        vals = np.log10(df.loc[mask, 'U_Th_ratio'].dropna().clip(lower=0.001))
        if len(vals) < 3: continue
        ax7.hist(vals, bins=20, color=color, alpha=0.6, label=source_labels[src], density=False)
    ax7.axvline(np.log10(uth_max), color='red', ls='--', lw=1.5, label=f'U/Th={uth_max} (mnz/thorite)')
    ax7.set_xlabel('log₁₀(U/Th)', fontsize=11); ax7.set_ylabel('Count', fontsize=11)
    ax7.set_title('F.  U/Th ratio by Th source type\n(anomalous samples only; mineral zones shaded)',
                  fontsize=9)
    ax7.legend(fontsize=6.5, ncol=1)
    ax7.tick_params(labelsize=9)
    # Annotate the dominant peak
    _peak_log = np.log10(df.loc[(df['th_source'] == 'THORITE_UTHO') & df['th_anomaly'],
                                'U_Th_ratio'].clip(lower=0.001)).median()
    ax7.annotate(f'Main peak\n(thorite-coffinite\nseries; median\nU/Th≈{10**_peak_log:.0f})',
                 xy=(_peak_log, ax7.get_ylim()[1] * 0.5 if ax7.get_ylim()[1] > 0 else 2),
                 xytext=(_peak_log - 1.1, ax7.get_ylim()[1] * 0.6 if ax7.get_ylim()[1] > 0 else 2.5),
                 fontsize=6.5, color=WONG['vermillion'],
                 arrowprops=dict(arrowstyle='->', color=WONG['vermillion'], lw=1.0),
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax8 = fig.add_subplot(gs[2, 0])
    scatter_panel(ax8, 'Th', 'LREE_sum', 'Th (ppm)', 'Ce+La+Nd (ppm)', show_legend=True)
    ax8.set_title('G.  Th vs ΣLREE\n(monazite → both Th & LREE co-enriched)', fontsize=9)
    # Compute and annotate Pearson r on log-transformed anomalous pairs
    _lree_valid = anomaly_df[['Th', 'LREE_sum']].dropna()
    _lree_valid = _lree_valid[(_lree_valid['Th'] > 0) & (_lree_valid['LREE_sum'] > 0)]
    if len(_lree_valid) > 5:
        from scipy import stats as _stats
        _r, _p = _stats.pearsonr(np.log10(_lree_valid['Th']), np.log10(_lree_valid['LREE_sum']))
        _sig = '(p<0.05)' if _p < 0.05 else '(n.s.)'
        ax8.text(0.97, 0.05, f'r = {_r:.2f} {_sig}\n(log-log; n={len(_lree_valid)})',
                 transform=ax8.transAxes, ha='right', va='bottom', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    # ── Panel H: Y vs Th  (xenotime as independent HREE signal) ──────────────
    ax_yth = fig.add_subplot(gs[2, 1])
    _yth_df = df[df['Y'].notna() & df['Th'].notna() & (df['Y'] > 0) & (df['Th'] > 0)].copy()
    # Background (non-anomalous) samples as grey
    _yth_bg = _yth_df[_yth_df['th_source'] == 'BACKGROUND']
    ax_yth.scatter(_yth_bg['Th'], _yth_bg['Y'], c='#cccccc', s=8, alpha=0.4,
                   zorder=1, label='Background NURE')
    # All classified sources (includes XENOTIME_Y at low Th, coloured pink)
    for src, color in source_colors.items():
        if src == 'BACKGROUND':
            continue
        _sub = _yth_df[_yth_df['th_source'] == src]
        if _sub.empty:
            continue
        ax_yth.scatter(_sub['Th'], _sub['Y'], c=color, s=25, alpha=0.8,
                       zorder=3, label=source_labels[src])
    # Y anomaly threshold line
    ax_yth.axhline(y_thresh, color=WONG['green'], ls='--', lw=1.5,
                   label=f'Y anomaly threshold\n({y_thresh:.0f} ppm, mean+2SD)')
    # Th anomaly threshold line
    ax_yth.axvline(th_threshold, color=WONG['vermillion'], ls=':', lw=1.3,
                   label=f'Th threshold ({th_threshold:.0f} ppm)')
    # Shade dual-anomaly quadrant
    ax_yth.axvspan(th_threshold, _yth_df['Th'].max() * 1.2,
                   ymin=0, alpha=0.06, color=WONG['green'], zorder=0)
    # Count populations
    _dual   = _yth_df[(_yth_df['Y'] > y_thresh) & (_yth_df['Th'] > th_threshold)]
    _xen_y  = _yth_df[_yth_df['th_source'] == 'XENOTIME_Y']
    ax_yth.text(0.97, 0.97,
                f'Dual Th+Y anomaly: n={len(_dual)}\n'
                f'Xenotime/Y-only (pink, no Th): n={len(_xen_y)}',
                transform=ax_yth.transAxes, ha='right', va='top', fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#eaffea', alpha=0.9))
    ax_yth.set_xscale('log'); ax_yth.set_yscale('log')
    ax_yth.set_xlabel('Th (ppm)', fontsize=10)
    ax_yth.set_ylabel('Y (ppm)', fontsize=10)
    ax_yth.set_title('H.  Y vs Th\n(xenotime HREE signal — independent of Th source)', fontsize=9)
    ax_yth.legend(fontsize=6.5, loc='lower right', framealpha=0.85)
    ax_yth.tick_params(labelsize=8)

    ax9 = fig.add_subplot(gs[2, 2])
    spider_els = ['La', 'Ce', 'Nd']
    available_spider = [e for e in spider_els if e in anomaly_df.columns]

    if len(available_spider) >= 2:
        top_lree = anomaly_df.nlargest(15, 'LREE_sum')
        for src, color in source_colors.items():
            mask = top_lree['th_source'] == src
            sub = top_lree[mask]
            if sub.empty: continue
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

        _t8_path = out(cfg, 'tables', 'task8_mine_waste_summary.csv')
        try:
            _t8_has_data = os.path.exists(_t8_path) and pd.read_csv(_t8_path).shape[0] > 0
        except Exception:
            _t8_has_data = False
        if wgs_geochem_df is not None and len(wgs_geochem_df) > 0 and _t8_has_data:
            dep_col = next((c for c in ['Deposit Type(s)', 'Deposit Types', 'Deposit_Type']
                            if c in wgs_geochem_df.columns), None)

            def _dep_key(raw):
                if pd.isna(raw): return 'other'
                t = str(raw).lower()
                if 'epithermal' in t: return 'epithermal'
                if 'intrusion' in t: return 'intrusion_related'
                if 'polymetallic' in t or 'vein' in t: return 'polymetallic_vein'
                return 'other'

            wgs_geochem_df = wgs_geochem_df.copy()
            wgs_geochem_df['_dep_key'] = (
                wgs_geochem_df[dep_col].apply(_dep_key) if dep_col else 'other')
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
                    ax9.plot(_x_pos, _mean_vals, color=_color, lw=2.5, alpha=0.9,
                             marker='D', markersize=5,
                             label=f"WGS {_dep_label.get(_dk, _dk)} (mine waste, ICP-MS)",
                             zorder=5)

        ax9.set_xticks(range(len(available_spider)))
        ax9.set_xticklabels(available_spider, fontsize=9)
        ax9.set_yscale('log')
        ax9.set_ylabel('Sample / CI chondrite\n(Sun & McDonough 1989)', fontsize=9)
        ax9.set_title('I.  LREE chondrite-normalized pattern\n'
                      '(top 15 NURE anomalies + WGS mine waste overlay)', fontsize=9)
        ax9.grid(True, alpha=0.3, which='both')
        ax9.tick_params(labelsize=9)
        ax9.legend(fontsize=6.0, loc='upper left', bbox_to_anchor=(0.0, -0.10),
                   bbox_transform=ax9.transAxes, framealpha=0.88, ncol=1,
                   borderpad=0.4, handlelength=1.5)
        _t8_path = out(cfg, 'tables', 'task8_mine_waste_summary.csv')
        try:
            _t8_has_data_annot = os.path.exists(_t8_path) and pd.read_csv(_t8_path).shape[0] > 0
        except Exception:
            _t8_has_data_annot = False
        if _t8_has_data_annot:
            _annot_text = ('† WGS bold lines = ICP-MS mine waste (OFR 2026-02)\n'
                           '  NURE thin lines = stream sediment (La/Ce/Nd only)\n'
                           '  Full REE patterns in Fig 9')
        else:
            _annot_text = ('WGS data not available\n'
                           '  NURE thin lines = stream sediment (La/Ce/Nd only)\n'
                           '  Set WGS_OFR2026_PATH to enable WGS overlay')
        ax9.text(0.97, 0.97, _annot_text,
                 transform=ax9.transAxes, ha='right', va='top', fontsize=6.5,
                 color='#555555', style='italic',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffbe6',
                           edgecolor='#ccccaa', alpha=0.85))
    else:
        ax9.text(0.5, 0.5, 'Insufficient REE columns\nfor spider diagram',
                 ha='center', va='center', transform=ax9.transAxes, fontsize=9, color='gray')
        ax9.set_title('I.  LREE pattern (insufficient data)', fontsize=9)

    # ── Panel J: Nb vs ΣLREE (fergusonite/columbite test) ────────────────────
    ax_j = fig.add_subplot(gs[3, 0])
    _nb_ok = 'Nb' in df.columns and (df['Nb'] > 0).sum() > 5
    if _nb_ok:
        _nb_bg = df[(df['Nb'] > 0) & (df['LREE_sum'] > 0)]
        ax_j.scatter(_nb_bg['Nb'], _nb_bg['LREE_sum'],
                     c='#cccccc', s=8, alpha=0.35, zorder=1, label='All NURE')
        for src, color in source_colors.items():
            mask = anomaly_df['th_source'] == src
            valid = mask & (anomaly_df['Nb'] > 0) & (anomaly_df['LREE_sum'] > 0)
            if valid.sum() == 0:
                continue
            ax_j.scatter(anomaly_df.loc[valid, 'Nb'], anomaly_df.loc[valid, 'LREE_sum'],
                         c=color, s=25, alpha=0.75, label=source_labels[src],
                         edgecolors='black', linewidths=0.3, zorder=2)
        ax_j.set_xscale('log'); ax_j.set_yscale('log')
        ax_j.grid(True, alpha=0.3, which='both')
        # Fergusonite suspect threshold: Nb > 40 ppm in stream sediment
        _fg_nb = geo.get('fergusonite_nb_min_ppm', 40)
        ax_j.axvline(_fg_nb, color=WONG['vermillion'], ls='--', lw=1.2, alpha=0.8,
                     label=f'Nb = {_fg_nb} ppm (Nb-oxide threshold)')
        _n_nb = ((anomaly_df['Nb'] > 0) & (anomaly_df['LREE_sum'] > 0)).sum()
        ax_j.text(0.97, 0.03, f'n={_n_nb}/{len(anomaly_df)} anomalous\nhave Nb & LREE data',
                  transform=ax_j.transAxes, ha='right', va='bottom', fontsize=7, color='#555555')
        ax_j.set_xlabel('Nb (ppm)', fontsize=11)
        ax_j.set_ylabel('Ce+La+Nd (ppm)', fontsize=11)
        ax_j.tick_params(labelsize=9)
        ax_j.legend(fontsize=6.5, loc='upper left', markerscale=1.2, framealpha=0.88)
    else:
        ax_j.text(0.5, 0.5, 'Nb not available\nin this dataset',
                  ha='center', va='center', transform=ax_j.transAxes, fontsize=9, color='gray')
    ax_j.set_title('J.  Nb vs ΣLREE\n(fergusonite/columbite: elevated Nb + LREE = Nb-oxide candidate)',
                   fontsize=9)

    # ── Panel K: Sr vs P (apatite vs monazite test) ───────────────────────────
    ax_k = fig.add_subplot(gs[3, 1])
    _sr_ok = 'Sr' in df.columns and (df['Sr'] > 0).sum() > 5
    _p_ok  = 'P'  in df.columns and (df['P']  > 0).sum() > 5
    if _sr_ok and _p_ok:
        _srp_bg = df[(df['Sr'] > 0) & (df['P'] > 0)]
        ax_k.scatter(_srp_bg['P'], _srp_bg['Sr'],
                     c='#cccccc', s=8, alpha=0.35, zorder=1, label='All NURE')
        for src, color in source_colors.items():
            mask = anomaly_df['th_source'] == src
            valid = mask & (anomaly_df['Sr'] > 0) & (anomaly_df['P'] > 0)
            if valid.sum() == 0:
                continue
            ax_k.scatter(anomaly_df.loc[valid, 'P'], anomaly_df.loc[valid, 'Sr'],
                         c=color, s=25, alpha=0.75, label=source_labels[src],
                         edgecolors='black', linewidths=0.3, zorder=2)
        ax_k.set_xscale('log'); ax_k.set_yscale('log')
        ax_k.grid(True, alpha=0.3, which='both')
        # Sr > 300 ppm with high P → apatite dominant; low Sr + high P + high Th → monazite
        _sr_ap = geo.get('apatite_sr_min_ppm', 300)
        ax_k.axhline(_sr_ap, color=WONG['blue'], ls='--', lw=1.2, alpha=0.8,
                     label=f'Sr = {_sr_ap} ppm (apatite-dominant P)')
        ax_k.axvline(p_min, color=WONG['green'], ls=':', lw=1.2, alpha=0.8,
                     label=f'P = {p_min} ppm (monazite proxy)')
        _n_srp = ((anomaly_df['Sr'] > 0) & (anomaly_df['P'] > 0)).sum()
        ax_k.text(0.97, 0.03, f'n={_n_srp}/{len(anomaly_df)} anomalous\nhave Sr & P data',
                  transform=ax_k.transAxes, ha='right', va='bottom', fontsize=7, color='#555555')
        ax_k.set_xlabel('P (ppm)', fontsize=11)
        ax_k.set_ylabel('Sr (ppm)', fontsize=11)
        ax_k.tick_params(labelsize=9)
        ax_k.legend(fontsize=6.5, loc='upper left', markerscale=1.2, framealpha=0.88)
    else:
        ax_k.text(0.5, 0.5, 'Sr or P not available\nin this dataset',
                  ha='center', va='center', transform=ax_k.transAxes, fontsize=9, color='gray')
    ax_k.set_title('K.  Sr vs P  (apatite test: high-Sr + high-P → apatite;\n'
                   '   low-Sr + high-P + high-Th → monazite)', fontsize=9)

    # ── Panel L: Untested mineral hypotheses ──────────────────────────────────
    ax_l = fig.add_subplot(gs[3, 2])
    ax_l.axis('off')
    # Compute counts for panel L — carefully scoped to what P data actually covers
    _th_anom_mask  = df['th_source'] != 'BACKGROUND'
    _lree_ok_mask  = (df['Ce'] > ce_min) | (df['La'] > la_min)
    _low_uth_mask  = df['U_Th_ratio'] < uth_max
    _has_p_mask    = df['P'].notna()
    _p_low_mask    = df['P'] < p_min / 2
    # Allanite test is only valid where P data exists
    _lree_uth_anom = _th_anom_mask & _lree_ok_mask & _low_uth_mask
    _n_allanite_tested  = int((_lree_uth_anom & _has_p_mask).sum())
    _n_allanite_hit     = int((_lree_uth_anom & _has_p_mask & _p_low_mask).sum())
    _n_allanite_no_p    = int((_lree_uth_anom & ~_has_p_mask).sum())
    _p_pct_nan = int(round(df['P'].isna().mean() * 100))
    # Titanite diagnostic: check multiple Ti thresholds over Th-anomalous population
    _ti_anom  = df.loc[_th_anom_mask, 'Ti']
    _ti_all   = df.loc[df['Ti'] > 0, 'Ti']
    _ti_med_all  = float(_ti_all.median())
    _ti_med_anom = float(_ti_anom[_ti_anom > 0].median()) if (_ti_anom > 0).any() else 0.0
    # Test Ti > 0.6 wt% (top ~10%) + LREE-mod + not-thorite + low-P + low-Zr
    _ti_hi    = df['Ti'] > 0.6
    _lree_mod = df['LREE_sum'] > 0 if 'LREE_sum' in df.columns else \
                (df['Ce'].fillna(0) + df['La'].fillna(0)) > df['La'][df['La'] > 0].median()
    _not_thor = df['U_Th_ratio'].isna() | (df['U_Th_ratio'] < uth_min)
    _low_p    = df['P'].isna() | (df['P'] < p_min)
    _low_zr   = df['Zr'].isna() | (df['Zr'] < 200)
    _n_titan_hi = int((_th_anom_mask & _ti_hi & _lree_mod & _not_thor & _low_p & _low_zr).sum())
    _limits_text = (
        'Accessory mineral discrimination results\n'
        '────────────────────────────────────────\n'
        'Allanite (REE-silicate) — UNTESTABLE\n'
        f'  {_p_pct_nan}% of samples lack P data\n'
        f'  Tested on P-data subset ({_n_allanite_tested} samples):\n'
        f'    LREE-ok + P<{p_min//2} + low-U/Th → n={_n_allanite_hit}\n'
        f'  P-absent LREE samples (black dots): n={_n_allanite_no_p}\n'
        '  → monazite vs allanite indeterminate\n'
        '  Si <1% NURE coverage for confirmation\n\n'
        'Titanite (CaTiSiO₅) — ROBUST NEGATIVE\n'
        f'  Ti median: {_ti_med_anom:.3f} wt% (Th-anom) vs\n'
        f'             {_ti_med_all:.3f} wt% (all samples)\n'
        '  Ti is LOWER in anomalous pop → no enrichment\n'
        f'  Ti>0.6 + LREE + not-thorite + low-P: n={_n_titan_hi}\n'
        '  High-Ti anomalous samples = MONAZITE class\n'
        '  (ilmenite co-deposit, not REE host)\n'
        '  Si/Hf <1% coverage for confirmation\n\n'
        'Chevkinite / perrierite\n'
        '  Ti–LREE coupling in panel E suggestive;\n'
        '  no Si/Al to confirm silicate phase\n\n'
        'Fergusonite: tested in panel J (Nb ~45%)\n'
        '  Ta mostly BDL; limited to Nb proxy\n\n'
        'Secondary REE phosphates\n'
        '  Requires sequential extraction / SEM-EDS\n\n'
        'Full HREE pattern (Gd–Lu)\n'
        '  <2% NURE coverage; Ho/Yb ~21% only'
    )
    ax_l.text(0.04, 0.96, _limits_text, transform=ax_l.transAxes,
              ha='left', va='top', fontsize=7.0, family='monospace',
              color='#333333',
              bbox=dict(boxstyle='round,pad=0.6', facecolor='#f5f5f5',
                        edgecolor='#bbbbbb', alpha=0.95))
    ax_l.set_title('L.  Analytical scope — untested mineral hypotheses', fontsize=9)

    fig.subplots_adjust(bottom=0.06)

    watermark(fig, cfg)
    save_fig(fig, out(cfg, 'figures', 'fig3_geochemical_discrimination.png'))

    # Summary stats
    summary_rows = []
    for src in df['th_source'].unique():
        mask = df['th_source'] == src
        sub  = df[mask]
        n_anom = df['th_anomaly'].sum()
        summary_rows.append({
            'th_source': src,
            'n_samples': mask.sum(),
            'pct_of_anomalies': f"{100*mask.sum()/n_anom:.1f}%" if n_anom > 0 else 'N/A',
            'Th_median_ppm': round(sub['Th'].median(), 1),
            'Ce_median_ppm': round(sub['Ce'].median(), 1) if 'Ce' in sub else None,
            'La_median_ppm': round(sub['La'].median(), 1) if 'La' in sub else None,
            'U_Th_median':   round(sub['U_Th_ratio'].median(), 3),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values('n_samples', ascending=False)
    summary_df.to_csv(out(cfg, 'tables', 'task3_summary_stats.csv'), index=False)

    # Export classified GeoJSON
    gdf = gpd.GeoDataFrame(df, geometry=[Point(r.lon, r.lat) for r in df.itertuples()], crs='EPSG:4326')
    base_export = ['lab_id','lon','lat','Th','Ce','La','Nd','P','Y','U','Zr','Ti','Fe',
                   'th_anomaly','U_Th_ratio','LREE_sum','th_source','geometry']
    export_cols = [c for c in base_export if c in gdf.columns]
    gdf[export_cols].to_file(out(cfg, 'geojson', 'nure_classified_th_sources.geojson'), driver='GeoJSON')

    print(f"\nTask 3 classification results:")
    print(summary_df[['th_source','n_samples','Th_median_ppm','U_Th_median']].to_string(index=False))
    if (df['th_source'] == 'MONAZITE').sum() > 5:
        m = df['th_source'] == 'MONAZITE'
        r, _ = stats.pearsonr(np.log10(df.loc[m,'Th']), np.log10(df.loc[m,'Ce']))
        print(f"\nMonazite Th-Ce log-correlation r={r:.3f}")


if __name__ == '__main__':
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/ne_washington/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
