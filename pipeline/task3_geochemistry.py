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
                             wgs_path, watermark, save_fig, ensure_outputs, out)


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

    def classify_th_source(row):
        th = row['Th']
        if pd.isna(th) or th < th_threshold / 2:
            return 'BACKGROUND'
        u_th = row['U_Th_ratio']
        ce, la, zr, p = row['Ce'], row['La'], row['Zr'], row['P']
        lree_ok = (pd.notna(ce) and ce > ce_min) or (pd.notna(la) and la > la_min)
        p_ok    = pd.isna(p) or p > p_min
        if pd.notna(u_th) and u_th < uth_max and lree_ok and p_ok:
            return 'MONAZITE'
        elif pd.notna(u_th) and u_th > uth_min:
            return 'THORITE_UTHO'
        elif pd.notna(zr) and zr > 200 and th < th_threshold * 1.5:
            return 'ZIRCON'
        else:
            return 'MIXED_UNCLEAR'

    df['th_source'] = df.apply(classify_th_source, axis=1)
    df['th_anomaly'] = df['th_source'] != 'BACKGROUND'

    # Correlation matrix
    elements = ['Th', 'Ce', 'La', 'Nd', 'P', 'Y', 'U', 'Zr', 'Ti', 'Fe']
    log_df = np.log10(df[elements].clip(lower=0.01))
    corr_matrix = log_df.corr()

    source_colors = {
        'MONAZITE':      WONG['green'],
        'THORITE_UTHO':  WONG['vermillion'],
        'ZIRCON':        WONG['sky'],
        'MIXED_UNCLEAR': WONG['orange'],
        'BACKGROUND':    '#CCCCCC',
    }
    source_labels = {
        'MONAZITE':      'Monazite (Th-LREE-P)',
        'THORITE_UTHO':  'Thorite/U-Th oxide',
        'ZIRCON':        'Zircon-dominated',
        'MIXED_UNCLEAR': 'Mixed/unclear',
        'BACKGROUND':    'Background',
    }

    anomaly_df = df[df['th_anomaly']].copy()

    def scatter_panel(ax, x_col, y_col, xlabel, ylabel):
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
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f'Figure 3 — Multi-element Geochemical Discrimination of Th Sources\n'
        f'{cfg["study_area"]["name"]} NURE Stream Sediment Data',
        fontsize=13, fontweight='bold', y=0.98,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    scatter_panel(ax1, 'Th', 'Ce', 'Th (ppm)', 'Ce (ppm)')
    ax1.set_title('A.  Th vs Ce (monazite: positive correlation)', fontsize=9)
    ax1.axhline(ce_min, color=WONG['blue'], ls='--', lw=1.2, alpha=0.8)
    ax1.text(ax1.get_xlim()[0] * 1.05 if ax1.get_xlim()[0] > 0 else 25,
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
    scatter_panel(ax2, 'Th', 'La', 'Th (ppm)', 'La (ppm)')
    ax2.set_title('B.  Th vs La', fontsize=9)
    ax2.axhline(la_min, color=WONG['blue'], ls='--', lw=1.2, alpha=0.8)
    ax2.text(ax2.get_xlim()[0] * 1.05 if ax2.get_xlim()[0] > 0 else 25,
             la_min * 1.05, f'La > {la_min} ppm (crustal threshold)',
             fontsize=6, color=WONG['blue'], va='bottom')

    ax3 = fig.add_subplot(gs[0, 2])
    scatter_panel(ax3, 'Th', 'P', 'Th (ppm)', 'P (ppm)')
    ax3.set_title('C.  Th vs P (monazite: phosphate co-enrichment)', fontsize=9)
    ax3.axhline(p_min, color=WONG['blue'], ls='--', lw=1.2, alpha=0.8)
    ax3.text(ax3.get_xlim()[0] * 1.05 if ax3.get_xlim()[0] > 0 else 25,
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

    ax5 = fig.add_subplot(gs[1, 1])
    im = ax5.imshow(corr_matrix.values, cmap='PuOr', vmin=-1, vmax=1, aspect='auto')
    ax5.set_xticks(range(len(elements))); ax5.set_yticks(range(len(elements)))
    ax5.set_xticklabels(elements, rotation=45, ha='right', fontsize=8)
    ax5.set_yticklabels(elements, fontsize=8)
    plt.colorbar(im, ax=ax5, shrink=0.8)
    ax5.set_title('E.  Log-element correlation matrix\n(all NURE samples)', fontsize=9)
    for i in range(len(elements)):
        for j in range(len(elements)):
            r_val = corr_matrix.values[i, j]
            if abs(r_val) > 0.5 and i != j:
                ax5.text(j, i, f'{r_val:.2f}', ha='center', va='center',
                         fontsize=6, color='white' if abs(r_val) > 0.75 else 'black')
    p_idx = elements.index('P') if 'P' in elements else None
    if p_idx is not None:
        ax5.add_patch(plt.Rectangle((-0.5, p_idx - 0.5), len(elements), 1,
                                    color='gray', alpha=0.30, zorder=3, clip_on=True))
        ax5.add_patch(plt.Rectangle((p_idx - 0.5, -0.5), 1, len(elements),
                                    color='gray', alpha=0.30, zorder=3, clip_on=True))
        ax5.text(p_idx, len(elements) + 0.2, '†', ha='center', va='bottom',
                 fontsize=8, color='#555555', zorder=4)
        ax5.text(0.5, -0.22, '† P: 66% NaN — correlation unreliable',
                 transform=ax5.transAxes, fontsize=6.5, color='#555555',
                 ha='center', style='italic')

    ax6 = fig.add_subplot(gs[1, 2])
    src_sizes   = {'BACKGROUND': 8,  'THORITE_UTHO': 35, 'MIXED_UNCLEAR': 35, 'ZIRCON': 35, 'MONAZITE': 35}
    src_markers = {'BACKGROUND': 'o', 'THORITE_UTHO': '^', 'MIXED_UNCLEAR': 'o', 'ZIRCON': 's', 'MONAZITE': 'D'}
    src_alphas  = {'BACKGROUND': 0.4, 'THORITE_UTHO': 0.9, 'MIXED_UNCLEAR': 0.9, 'ZIRCON': 0.9, 'MONAZITE': 0.9}
    for src, color in source_colors.items():
        mask = df['th_source'] == src
        if mask.sum() == 0: continue
        ax6.scatter(df.loc[mask, 'lon'], df.loc[mask, 'lat'],
                    c=color, s=src_sizes.get(src, 18), marker=src_markers.get(src, 'o'),
                    alpha=src_alphas.get(src, 0.7), label=source_labels[src],
                    edgecolors='black' if src != 'BACKGROUND' else 'none', linewidths=0.3)
    ax6.set_xlabel('Longitude', fontsize=11); ax6.set_ylabel('Latitude', fontsize=11)
    ax6.set_title('F.  Spatial distribution of Th source types', fontsize=9)
    ax6.tick_params(labelsize=9)
    ax6.grid(True, alpha=0.2)
    for lon_line in np.arange(
            round(cfg['study_area']['bbox']['lon_min'] + 0.5),
            cfg['study_area']['bbox']['lon_max'], 1.0):
        ax6.axvline(lon_line, color='gray', lw=0.5, ls='--', alpha=0.4)

    ax7 = fig.add_subplot(gs[2, 0])
    for src, color in source_colors.items():
        mask = (df['th_source'] == src) & df['th_anomaly']
        if mask.sum() < 3: continue
        vals = np.log10(df.loc[mask, 'U_Th_ratio'].clip(lower=0.001))
        ax7.hist(vals, bins=20, color=color, alpha=0.6, label=source_labels[src], density=True)
    ax7.axvline(np.log10(uth_max), color='red', ls='--', lw=1.5, label=f'U/Th={uth_max} threshold')
    ax7.set_xlabel('log₁₀(U/Th)', fontsize=11); ax7.set_ylabel('Density', fontsize=11)
    ax7.set_title('G.  U/Th ratio by Th source type\n(anomalous samples only)', fontsize=9)
    ax7.legend(fontsize=8)
    ax7.tick_params(labelsize=9)

    ax8 = fig.add_subplot(gs[2, 1])
    scatter_panel(ax8, 'Th', 'LREE_sum', 'Th (ppm)', 'Ce+La+Nd (ppm)')
    ax8.set_title('H.  Th vs ΣLREE\n(monazite: parallel enrichment)', fontsize=9)

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

        if wgs_geochem_df is not None and len(wgs_geochem_df) > 0:
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
        ax9.legend(fontsize=6.0, loc='lower left', framealpha=0.85)
        ax9.text(0.97, 0.97,
                 '† WGS bold lines = ICP-MS mine waste (OFR 2026-02)\n'
                 '  NURE thin lines = stream sediment (La/Ce/Nd only)\n'
                 '  Full REE patterns in Fig 9',
                 transform=ax9.transAxes, ha='right', va='top', fontsize=6.5,
                 color='#555555', style='italic',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffbe6',
                           edgecolor='#ccccaa', alpha=0.85))
    else:
        ax9.text(0.5, 0.5, 'Insufficient REE columns\nfor spider diagram',
                 ha='center', va='center', transform=ax9.transAxes, fontsize=9, color='gray')
        ax9.set_title('I.  LREE pattern (insufficient data)', fontsize=9)

    handles = [plt.scatter([], [], c=c, s=40, label=l)
               for c, l in zip(source_colors.values(), source_labels.values())]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 0.01), framealpha=0.9)

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
