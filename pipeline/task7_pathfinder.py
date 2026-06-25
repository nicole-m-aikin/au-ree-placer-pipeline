"""
Task 7: Au and epithermal/porphyry pathfinder anomaly analysis.

Output:
  {outputs_dir}/figures/fig8_au_as_anomaly_map.png
  {outputs_dir}/tables/task7_au_as_summary.csv
  {outputs_dir}/text/task7_au_anomaly_notes.txt
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

import matplotlib.patheffects as pe

from pipeline.utils import (WONG, setup_mpl, load_nure, anomaly_threshold,
                             watermark, save_fig, ensure_outputs, out,
                             map_extent, hillshade, north_arrow, scale_bar,
                             canada_border, locator_inset,
                             topo_contours, rivers_with_arrows,
                             MAP_W, MAP_H, _FIG_LM, _FIG_RM, _FIG_TM, _FIG_BM,
                             _FIG_HGAP, _ax_rect)


def run(cfg):
    setup_mpl()
    ensure_outputs(cfg['outputs_dir'])

    print("\n" + "="*60)
    print("STEP 1: Loading data")
    print("="*60)

    df = load_nure(cfg)
    print(f"  NURE sediment: {df.shape[0]} samples, {df.shape[1]} columns")

    th_gdf   = gpd.read_file(out(cfg, 'geojson', 'nure_classified_th_sources.geojson'))
    mine_gdf = gpd.read_file(out(cfg, 'geojson', 'task1_multicommodity_targets.geojson'))
    print(f"  Th source GeoJSON: {th_gdf.shape[0]} records")
    print(f"  Mine sites GeoJSON: {mine_gdf.shape[0]} sites")

    print("\n" + "="*60)
    print("STEP 2: Below-detection handling")
    print("="*60)

    ELEMENTS = ['Au', 'As', 'Sb', 'Cu', 'Mo', 'Pb', 'Zn', 'Th', 'La', 'P']
    present  = [e for e in ELEMENTS if e in df.columns]
    print(f"  Elements found: {present}")

    for col in present:
        series   = pd.to_numeric(df[col], errors='coerce')
        pos_vals = series[series > 0]
        if pos_vals.empty:
            df[col] = np.nan; continue
        med = pos_vals.median()
        threshold = 10 * med
        large_neg = series < -threshold
        small_neg = (series < 0) & (series >= -threshold)
        series = series.copy()
        series[large_neg] = np.nan
        series[small_neg] = series[small_neg].abs() / 2.0
        df[col] = series

    print("\n" + "="*60)
    print("STEP 3: Au units check")
    print("="*60)

    au_units_warning = False
    au_warning_msg   = ""
    if 'Au' in df.columns:
        au = pd.to_numeric(df['Au'], errors='coerce')
        pos_au = au[au > 0]
        print(f"  Au median: {pos_au.median():.4g}, max: {au.max():.4g}")
        if pos_au.median() > 0.05:
            au_units_warning = True
            au_warning_msg = ("WARNING: Median Au > 0.05 ppm — values may be in ppb mislabeled as ppm.")
            print(f"\n  *** {au_warning_msg} ***\n")

    print("\n" + "="*60)
    print("STEP 4: Log-threshold anomaly cutoffs")
    print("="*60)

    threshold_els = ['Au', 'As', 'Sb', 'Cu', 'Mo', 'Th', 'La']
    thresholds = {}
    for col in threshold_els:
        if col not in df.columns:
            thresholds[col] = None; continue
        thr = anomaly_threshold(pd.to_numeric(df[col], errors='coerce'))
        thresholds[col] = thr
        print(f"  {col}: threshold={thr:.4g}" if thr else f"  {col}: insufficient data")

    for col in ['Pb', 'Zn']:
        if col in df.columns:
            thr = anomaly_threshold(pd.to_numeric(df[col], errors='coerce'))
            if thr:
                thresholds[col] = thr
                print(f"  {col}: threshold={thr:.4g}")

    print("\n" + "="*60)
    print("STEP 5: Sample classification")
    print("="*60)

    def is_anomalous(col, thr):
        if col not in df.columns or thr is None:
            return pd.Series(False, index=df.index)
        return pd.to_numeric(df[col], errors='coerce') > thr

    au_anom = is_anomalous('Au', thresholds.get('Au'))
    as_anom = is_anomalous('As', thresholds.get('As'))
    sb_anom = is_anomalous('Sb', thresholds.get('Sb'))
    cu_anom = is_anomalous('Cu', thresholds.get('Cu'))
    mo_anom = is_anomalous('Mo', thresholds.get('Mo'))
    pb_anom = is_anomalous('Pb', thresholds.get('Pb'))
    zn_anom = is_anomalous('Zn', thresholds.get('Zn'))
    th_anom = is_anomalous('Th', thresholds.get('Th'))
    la_anom = is_anomalous('La', thresholds.get('La'))

    # REE/monazite: joint Th+La anomaly; optional P gate from config
    ree_monazite = th_anom & la_anom
    monazite_p_min = cfg.get('geochemistry', {}).get('monazite_p_min_ppm')
    if monazite_p_min and 'P' in df.columns:
        p_series = pd.to_numeric(df['P'], errors='coerce')
        ree_monazite = ree_monazite & (p_series > monazite_p_min)

    epithermal = au_anom & (as_anom | sb_anom)
    porphyry   = cu_anom & mo_anom
    au_only    = au_anom & ~epithermal & ~porphyry
    base_metal = (pb_anom | zn_anom) & ~au_anom
    df['au_class'] = 'BACKGROUND'
    df.loc[base_metal, 'au_class']   = 'BASE_METAL'
    df.loc[au_only, 'au_class']      = 'AU_ONLY'
    df.loc[ree_monazite, 'au_class'] = 'REE_MONAZITE'   # overrides AU_ONLY
    df.loc[porphyry, 'au_class']     = 'PORPHYRY_CU'
    df.loc[epithermal, 'au_class']   = 'EPITHERMAL_AU'
    counts = df['au_class'].value_counts()
    print(f"  Classification: {counts.to_dict()}")

    print("\n" + "="*60)
    print("STEP 6: Nearest Au-anomaly sample per mine site")
    print("="*60)

    au_anom_df  = df[df['au_class'].isin(['EPITHERMAL_AU', 'PORPHYRY_CU', 'AU_ONLY', 'REE_MONAZITE'])].copy()
    au_anom_gdf = gpd.GeoDataFrame(
        au_anom_df,
        geometry=[Point(xy) for xy in zip(au_anom_df['lon'], au_anom_df['lat'])],
        crs='EPSG:4326',
    )
    MAX_DIST = 0.25

    site_records = []
    for _, site in mine_gdf.iterrows():
        site_lon, site_lat = site.geometry.x, site.geometry.y
        site_name = site.get('name', 'unknown')
        dists = np.sqrt((au_anom_gdf['lon'] - site_lon)**2 + (au_anom_gdf['lat'] - site_lat)**2)
        mask = dists <= MAX_DIST
        if mask.any():
            idx_min = dists[mask].idxmin()
            nearest = au_anom_gdf.loc[idx_min]
            rec = {
                'site_name':       site_name,
                'nearest_au_ppm':  round(float(nearest['Au']), 4) if pd.notna(nearest.get('Au')) else np.nan,
                'nearest_as_ppm':  round(float(nearest['As']), 4) if ('As' in nearest and pd.notna(nearest['As'])) else np.nan,
                'au_class':        nearest['au_class'],
                'distance_deg':    round(float(dists[mask].min()), 4),
            }
        else:
            rec = {'site_name': site_name, 'nearest_au_ppm': np.nan,
                   'nearest_as_ppm': np.nan, 'au_class': 'NONE', 'distance_deg': np.nan}
        site_records.append(rec)

    summary_df = pd.DataFrame(site_records)

    print("\n" + "="*60)
    print("STEP 7: Cross-reference with Th anomalies")
    print("="*60)

    th_anom_gdf = th_gdf[th_gdf['th_anomaly'] == True].copy()
    has_th, dual = [], []
    for _, site in mine_gdf.iterrows():
        site_lon, site_lat = site.geometry.x, site.geometry.y
        site_name = site.get('name', 'unknown')
        th_dists = np.sqrt((th_anom_gdf['lon'] - site_lon)**2 + (th_anom_gdf['lat'] - site_lat)**2)
        near_th = (th_dists <= MAX_DIST).any() if len(th_dists) else False
        row = summary_df[summary_df['site_name'] == site_name]
        near_au = (not row.empty) and (row.iloc[0]['au_class'] != 'NONE')
        has_th.append(near_th)
        dual.append(near_th and near_au)

    summary_df['has_th_anomaly']   = has_th
    summary_df['dual_anomaly_flag'] = dual
    dual_sites = summary_df[summary_df['dual_anomaly_flag']]['site_name'].tolist()
    print(f"  Dual Th+Au anomaly sites: {dual_sites}")

    print("\n" + "="*60)
    print("STEP 8: Generating Figure 8")
    print("="*60)

    _TERN_W = 3.6
    figW = _FIG_LM + MAP_W + _FIG_HGAP + _TERN_W + _FIG_RM
    figH = _FIG_TM + MAP_H + _FIG_BM
    fig = plt.figure(figsize=(figW, figH))
    ax  = fig.add_axes(_ax_rect(_FIG_LM, _FIG_BM, MAP_W, MAP_H, figW, figH))
    xmin_7, xmax_7, ymin_7, ymax_7 = map_extent(cfg)
    ax.set_xlim(xmin_7, xmax_7)
    ax.set_ylim(ymin_7, ymax_7)
    ax.set_aspect('auto')
    hillshade(cfg, ax, alpha=0.22)
    topo_contours(ax, cfg)
    rivers_with_arrows(ax, cfg)

    bg = df[df['au_class'] == 'BACKGROUND']
    ax.scatter(bg['lon'], bg['lat'], s=5, alpha=0.3, color='#CCCCCC', zorder=1, label='_nolegend_')

    ao = df[df['au_class'] == 'AU_ONLY']
    if not ao.empty:
        ax.scatter(ao['lon'], ao['lat'], s=30, color=WONG['sky'],
                   edgecolors='white', linewidths=0.5, zorder=3, label='_nolegend_')

    pc = df[df['au_class'] == 'PORPHYRY_CU']
    if not pc.empty:
        ax.scatter(pc['lon'], pc['lat'], s=50, marker='s', color=WONG['blue'],
                   edgecolors='black', linewidths=0.6, zorder=4, label='_nolegend_')
        for _, pr in pc.iterrows():
            cu_val = pr.get('Cu', float('nan'))
            mo_val = pr.get('Mo', float('nan'))
            ax.annotate(f"Cu={cu_val:.0f}\nMo={mo_val:.1f}", xy=(pr['lon'], pr['lat']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=5.5, color=WONG['blue'], zorder=9,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.6, lw=0))

    ep = df[df['au_class'] == 'EPITHERMAL_AU'].copy()
    if not ep.empty:
        au_thr = thresholds.get('Au') or 1.0
        ep['Au_num'] = pd.to_numeric(ep['Au'], errors='coerce').fillna(au_thr)
        sizes = np.clip(50 * ep['Au_num'] / au_thr, 20, 400)
        ax.scatter(ep['lon'], ep['lat'], s=sizes, color=WONG['orange'],
                   edgecolors='black', linewidths=0.4, zorder=5, label='_nolegend_')

    bm = df[df['au_class'] == 'BASE_METAL']
    if not bm.empty:
        ax.scatter(bm['lon'], bm['lat'], s=20, marker='^', color=WONG['pink'],
                   edgecolors='white', linewidths=0.4, zorder=3, label='_nolegend_')

    rm = df[df['au_class'] == 'REE_MONAZITE'].copy()
    if not rm.empty:
        th_thr = thresholds.get('Th') or 1.0
        rm['Th_num'] = pd.to_numeric(rm['Th'], errors='coerce').fillna(th_thr)
        sizes_rm = np.clip(60 * rm['Th_num'] / th_thr, 30, 500)
        ax.scatter(rm['lon'], rm['lat'], s=sizes_rm, marker='^', color=WONG['vermillion'],
                   edgecolors='black', linewidths=0.5, zorder=6, label='_nolegend_')

    LABEL_OFFSETS = {ann.get('site'): tuple(ann.get('label_offset', (4, 4)))
                     for ann in cfg.get('task7_site_label_offsets', [])}
    DEFAULT_OFFSET = (4, 4)

    for _, site in mine_gdf.iterrows():
        sname = site.get('name', 'unknown')
        row   = summary_df[summary_df['site_name'] == sname]
        is_dual = bool(row['dual_anomaly_flag'].values[0]) if not row.empty else False
        near_au = (not row.empty) and (row.iloc[0]['au_class'] != 'NONE')
        scolor = WONG['vermillion'] if is_dual else (WONG['green'] if near_au else '#CCCCCC')
        ax.scatter(site.geometry.x, site.geometry.y, marker='D', s=80,
                   color=scolor, edgecolors='black', linewidths=0.8, zorder=7)
        if is_dual:
            ax.scatter(site.geometry.x, site.geometry.y, marker='*', s=300,
                       facecolor=scolor, edgecolors='black', linewidths=0.6, zorder=8)
        ox, oy = LABEL_OFFSETS.get(sname, DEFAULT_OFFSET)
        ax.annotate(sname, xy=(site.geometry.x, site.geometry.y),
                    xytext=(ox, oy), textcoords='offset points', fontsize=6, zorder=9, color='#222222')

    # Custom map annotations from config (e.g., Au cluster note)
    for ann in cfg.get('task7_annotations', []):
        ax.annotate(
            ann['text'],
            xy=ann['xy'], xytext=ann['xytext'],
            arrowprops=dict(arrowstyle='->', color=WONG['sky'], lw=1.0),
            fontsize=ann.get('fontsize', 6.5),
            color=WONG['sky'],
            style=ann.get('style', 'italic'),
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, lw=0),
        )

    site_legend = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#CCCCCC', markersize=5, label='Background'),
    ]
    if not ao.empty:
        site_legend.append(Line2D([0],[0], marker='o', color='w', markerfacecolor=WONG['sky'],
                                   markersize=7, label='Au only'))
    if not rm.empty:
        site_legend.append(Line2D([0],[0], marker='^', color='w', markerfacecolor=WONG['orange'],
                                   markeredgecolor='black', markersize=9, label='REE/monazite (Th+La)'))
    if not pc.empty:
        site_legend.append(Line2D([0],[0], marker='s', color='w', markerfacecolor=WONG['blue'],
                                   markersize=7, label='Porphyry Cu'))
    if not ep.empty:
        site_legend.append(Line2D([0],[0], marker='o', color='w', markerfacecolor=WONG['orange'],
                                   markersize=9, label='Epithermal Au'))
    if not bm.empty:
        site_legend.append(Line2D([0],[0], marker='^', color='w', markerfacecolor=WONG['pink'],
                                   markersize=7, label='Base metal'))
    site_legend += [
        Line2D([0],[0], marker='D', color='w', markerfacecolor='#CCCCCC',
               markeredgecolor='black', markersize=8, label='Mine site (no anomaly)'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor=WONG['green'],
               markeredgecolor='black', markersize=8, label='Mine site (Au only)'),
        Line2D([0],[0], marker='*', color='w', markerfacecolor=WONG['vermillion'],
               markeredgecolor='black', markersize=12,
               label=f'★ Dual Th+Au anomaly ({", ".join(dual_sites) or "none"})'),
    ]
    ax.legend(handles=site_legend, loc='upper left', fontsize=7, framealpha=0.88,
              title='Classification', title_fontsize=8,
              bbox_to_anchor=(0.01, 0.99))

    ax.set_title(
        f'A.  Au/As/REE Pathfinder Geochemistry — {cfg["study_area"]["name"]}\n'
        'Mineral Systems: TRAP characterisation (Au/As/REE halos)',
        fontsize=10, fontweight='bold', pad=8,
    )
    canada_border(ax, cfg)
    north_arrow(ax)
    scale_bar(ax, cfg)
    locator_inset(fig, ax, cfg)
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    # ── Panel D: deposit-type ternary discrimination ──────────────────────────
    ax_tern = fig.add_axes(_ax_rect(_FIG_LM + MAP_W + _FIG_HGAP, _FIG_BM,
                                    _TERN_W, MAP_H, figW, figH))

    # NaN-safe log10: replace 0/NaN with column median before logging
    def _nan_safe_log10(df_, col):
        v = pd.to_numeric(df_[col], errors='coerce') if col in df_.columns else pd.Series(np.nan, index=df_.index)
        v = v.where(v > 0, np.nan)
        med = v.median()
        v = v.fillna(med if pd.notna(med) else 1.0)
        return np.log10(v.clip(lower=1e-9))

    _log_th = _nan_safe_log10(df, 'Th')
    _log_ce = _nan_safe_log10(df, 'Ce')
    _log_la = _nan_safe_log10(df, 'La')
    _log_au = _nan_safe_log10(df, 'Au')
    _log_as = _nan_safe_log10(df, 'As')
    _log_cu = _nan_safe_log10(df, 'Cu')
    _log_mo = _nan_safe_log10(df, 'Mo')

    ree_comp  = _log_th + _log_ce + _log_la
    au_comp   = _log_au + _log_as
    porp_comp = _log_cu + _log_mo

    def _minmax(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    a = _minmax(ree_comp)    # REE vertex
    b = _minmax(au_comp)     # Placer-Au vertex
    c = _minmax(porp_comp)   # Porphyry vertex
    tot = a + b + c + 1e-9

    # Standard ternary → Cartesian (Au=left, REE=right, Porphyry=top apex)
    tern_x = 0.5 * (2 * b + c) / tot
    tern_y = (np.sqrt(3) / 2) * c / tot

    class_color_map = {
        'BACKGROUND':   '#CCCCCC',
        'REE_MONAZITE': WONG['vermillion'],
        'AU_ONLY':      WONG['sky'],
        'PORPHYRY_CU':  WONG['blue'],
        'EPITHERMAL_AU': WONG['orange'],
        'BASE_METAL':   WONG['pink'],
    }
    class_marker_map = {
        'BACKGROUND':   'o',
        'REE_MONAZITE': '^',
        'AU_ONLY':      'o',
        'PORPHYRY_CU':  's',
        'EPITHERMAL_AU': 'o',
        'BASE_METAL':   '^',
    }

    # Draw ternary triangle outline
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3)/2, 0]
    ax_tern.plot(tri_x, tri_y, 'k-', lw=1.5, zorder=0)

    # Grid lines at 25%, 50%, 75%
    for frac in [0.25, 0.5, 0.75]:
        ax_tern.plot([frac*(0.5), frac*0 + (1-frac)*0.5],
                     [frac*(np.sqrt(3)/2), frac*0], color='gray', lw=0.4, alpha=0.4, ls='--')
        ax_tern.plot([1-frac, (1-frac)*0.5],
                     [0, (1-frac)*np.sqrt(3)/2], color='gray', lw=0.4, alpha=0.4, ls='--')
        ax_tern.plot([frac*0.5, 1-frac*(0.5)],
                     [frac*np.sqrt(3)/2, frac*np.sqrt(3)/2], color='gray', lw=0.4, alpha=0.4, ls='--')

    for cls in ['BACKGROUND', 'BASE_METAL', 'AU_ONLY', 'REE_MONAZITE', 'PORPHYRY_CU', 'EPITHERMAL_AU']:
        mask = df['au_class'] == cls
        if not mask.any():
            continue
        alpha = 0.25 if cls == 'BACKGROUND' else 0.75
        size  = 8   if cls == 'BACKGROUND' else 30
        ax_tern.scatter(tern_x[mask], tern_y[mask],
                        c=class_color_map[cls], marker=class_marker_map[cls],
                        s=size, alpha=alpha, edgecolors='none', zorder=3 if cls == 'BACKGROUND' else 5)

    # Annotate Pend Oreille porphyry cluster (high porphyry corner)
    po_mask = (df['au_class'] == 'PORPHYRY_CU') & (c / tot > 0.45)
    if po_mask.any():
        po_x = tern_x[po_mask].mean()
        po_y = tern_y[po_mask].mean()
        ax_tern.annotate('Pend Oreille\nCu-Mo cluster',
                         xy=(po_x, po_y), xytext=(po_x - 0.18, po_y + 0.05),
                         fontsize=6.5, color=WONG['blue'], style='italic',
                         arrowprops=dict(arrowstyle='->', color=WONG['blue'], lw=0.8),
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, lw=0),
                         zorder=8,
                         path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Vertex labels (bottom-left=REE, bottom-right=Placer-Au, top=Porphyry)
    ax_tern.text(-0.03, -0.04, 'REE/Monazite\n(Th+Ce+La)', ha='right', va='top',
                 fontsize=8, fontweight='bold')
    ax_tern.text(1.03,  -0.04, 'Placer-Au\n(Au+As)', ha='left', va='top',
                 fontsize=8, fontweight='bold')
    ax_tern.text(0.5,  np.sqrt(3)/2 + 0.04, 'Porphyry\n(Cu+Mo)', ha='center', va='bottom',
                 fontsize=8, fontweight='bold')

    # Legend for ternary
    tern_legend = []
    for cls, lbl in [('BACKGROUND','Background'), ('REE_MONAZITE','REE/monazite'),
                     ('AU_ONLY','Au only'), ('PORPHYRY_CU','Porphyry Cu'),
                     ('EPITHERMAL_AU','Epithermal Au'), ('BASE_METAL','Base metal')]:
        if (df['au_class'] == cls).any():
            tern_legend.append(Line2D([0],[0], marker=class_marker_map[cls], color='w',
                                      markerfacecolor=class_color_map[cls],
                                      markersize=7, label=lbl))
    ax_tern.legend(handles=tern_legend, fontsize=7, loc='upper right',
                   framealpha=0.88, title='Classification', title_fontsize=7)

    ax_tern.set_xlim(-0.08, 1.08)
    ax_tern.set_ylim(-0.10, np.sqrt(3)/2 + 0.12)
    ax_tern.set_aspect('equal')
    ax_tern.axis('off')
    ax_tern.set_title('D.  Deposit-type ternary discrimination\n'
                      '(normalized composite scores; proves deposit-type separation)',
                      fontsize=9, fontweight='bold', pad=8)
    ax_tern.text(0.5, -0.08,
                 'Each vertex = min-max normalized log-composite score.\n'
                 'REE/monazite: log(Th)+log(Ce)+log(La)  ·  '
                 'Placer Au: log(Au)+log(As)  ·  Porphyry: log(Cu)+log(Mo)',
                 ha='center', va='top', fontsize=6, color='gray', style='italic',
                 transform=ax_tern.transAxes)

    fig.suptitle(
        f'Figure 8 — Au/As/REE Pathfinder Geochemistry — {cfg["study_area"]["name"]}\n'
        'Mineral Systems: TRAP + REE characterisation (Au/As/REE halos)',
        fontsize=12, fontweight='bold',
    )

    watermark(fig, cfg)
    save_fig(fig, out(cfg, 'figures', 'fig8_au_as_anomaly_map.png'))

    print("\n" + "="*60)
    print("STEP 9: Saving outputs")
    print("="*60)

    # Add REE/monazite nearest-sample info per site
    ree_anom_df = df[df['au_class'] == 'REE_MONAZITE'].copy()
    ree_near_flags = []
    for _, site in mine_gdf.iterrows():
        if ree_anom_df.empty:
            ree_near_flags.append(False)
            continue
        site_lon, site_lat = site.geometry.x, site.geometry.y
        dists = np.sqrt((ree_anom_df['lon'] - site_lon)**2 + (ree_anom_df['lat'] - site_lat)**2)
        ree_near_flags.append(bool((dists <= MAX_DIST).any()))
    summary_df['has_ree_monazite_anomaly'] = ree_near_flags

    summary_df.to_csv(out(cfg, 'tables', 'task7_au_as_summary.csv'), index=False)
    print(f"  Saved: {out(cfg, 'tables', 'task7_au_as_summary.csv')}")

    ree_count = int((df['au_class'] == 'REE_MONAZITE').sum())
    print(f"  REE/monazite samples: {ree_count}")

    lines = [
        "TASK 7 — Au/As/REE/Pathfinder Anomaly Analysis Notes",
        "=" * 60, "",
    ]
    if au_units_warning:
        lines += ["UNITS WARNING:", f"  {au_warning_msg}", ""]
    lines.append("CLASSIFICATION COUNTS:")
    for cls, cnt in counts.items():
        lines.append(f"  {cls}: {cnt}")
    lines += ["", f"REE/monazite samples: {ree_count}",
              "", "DUAL Th+Au ANOMALY SITES:"]
    lines += [f"  - {s}" for s in dual_sites] if dual_sites else ["  None identified."]
    notes_path = out(cfg, 'text', 'task7_au_anomaly_notes.txt')
    with open(notes_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {notes_path}")

    print("\nTASK 7 COMPLETE")
    print(f"  Au units warning: {au_units_warning}")
    print(f"  Dual anomaly sites: {dual_sites}")


if __name__ == '__main__':
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/ne_washington/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
