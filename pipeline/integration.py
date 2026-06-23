"""
Integration task: multi-criterion priority ranking across all tasks.

Reads all task outputs, computes combined scores, writes:
  {outputs_dir}/geojson/fig7_integrated_priority_tier.geojson
  {outputs_dir}/figures/fig7_integrated_priority_map.png
  {outputs_dir}/text/executive_summary_top3_sites.txt
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

from pipeline.utils import WONG, setup_mpl, watermark, save_fig, ensure_outputs, out


def run(cfg):
    setup_mpl()
    ensure_outputs(cfg['outputs_dir'])

    w         = cfg['scoring']['weights']
    tiers     = cfg['scoring']['tier_thresholds']          # e.g. [8, 6, 4]
    ndpr_cur  = cfg['economics']['ndpr_price_central']
    patches   = cfg.get('map_domain_patches', [])
    cnty_lbls = cfg.get('county_labels', [])

    def tier_color(score):
        if score >= tiers[0]: return WONG['blue']
        if score >= tiers[1]: return WONG['orange']
        if score >= tiers[2]: return WONG['yellow']
        return '#CCCCCC'

    def score_th(src):
        return {'MONAZITE': 2, 'MIXED_UNCLEAR': 1}.get(src, 0)

    def score_conf(conf):
        return {'HIGH': 2, 'MEDIUM': 1, 'LOW': 0}.get(conf, 0)

    def score_ndpr(ndpr_t):
        if pd.isna(ndpr_t) or ndpr_t <= 0: return 0.0
        return min(2.0, np.log10(max(ndpr_t, 1)) / np.log10(1000) * 2)

    def score_au(row):
        if row.get('dual_anomaly_flag'): return 2.0
        if row.get('has_th_anomaly'):    return 0.5
        return 0.0

    # ── Load task outputs ─────────────────────────────────────────────────────
    task1_df = pd.read_csv(out(cfg, 'tables', 'task1_site_summary.csv'))
    task2_df = pd.read_csv(out(cfg, 'tables', 'task2_catchment_scores.csv'))
    task4_df = pd.read_csv(out(cfg, 'tables', 'task4_volume_tonnage_summary.csv'))
    task5_df = pd.read_csv(out(cfg, 'tables', 'task5_breakeven_analysis.csv'))
    task7_df = pd.read_csv(out(cfg, 'tables', 'task7_au_as_summary.csv'))
    nure_gdf = gpd.read_file(out(cfg, 'geojson', 'nure_classified_th_sources.geojson'))
    base_gdf = gpd.read_file(out(cfg, 'geojson', 'task1_multicommodity_targets.geojson'))

    # ── Merge all task outputs ─────────────────────────────────────────────────
    merged = base_gdf.copy()

    t2 = task2_df.set_index('name')[['source_lith_score','source_lith_desc']]
    merged = merged.join(t2, on='name', how='left')

    t4 = task4_df.set_index('site_name')[['tonnage_t','ndpr_ppm',
                                           'ndpr_tonnes','ndpr_t_p10','ndpr_t_p50','ndpr_t_p90','confidence']]
    merged = merged.join(t4, on='name', how='left')

    t5 = task5_df.set_index('site')[['total_cost_$M','npv_central_$M','breakeven_$/kg']]
    merged = merged.join(t5, on='name', how='left')
    merged['processing_viable'] = merged['breakeven_$/kg'].apply(
        lambda x: True if pd.notna(x) and x < ndpr_cur else (False if pd.notna(x) else None))

    t7 = task7_df.set_index('site_name')[['dual_anomaly_flag','has_th_anomaly']]
    merged = merged.join(t7, on='name', how='left')
    merged['dual_anomaly_flag'] = merged['dual_anomaly_flag'].fillna(False)
    merged['has_th_anomaly']    = merged['has_th_anomaly'].fillna(False)

    # ── Compute combined score ────────────────────────────────────────────────
    merged['score_th']   = merged['th_source'].apply(score_th)
    merged['score_mag']  = merged['mag_high'].astype(int)
    merged['score_lith'] = merged['source_lith_score'].fillna(1.0)
    merged['score_conf'] = merged['confidence'].apply(score_conf)
    merged['score_ndpr'] = merged['ndpr_tonnes'].apply(score_ndpr)
    merged['score_au']   = merged.apply(score_au, axis=1)

    merged['combined_score'] = (
        merged['score_th']   * w.get('th_source_monazite', 1.0) +
        merged['score_mag']  * w.get('magnetic_high',       1.0) +
        merged['score_lith'] * w.get('source_lith',         1.0) +
        merged['score_conf'] * w.get('coverage',            1.0) +
        merged['score_ndpr'] * w.get('ndpr_volume',         2.0) +
        merged['score_au']   * w.get('au_pathfinder',       1.0)
    ).round(2)

    merged_sorted = merged.sort_values('combined_score', ascending=False).reset_index(drop=True)
    merged_sorted['rank'] = merged_sorted.index + 1
    merged_sorted.to_file(out(cfg, 'geojson', 'fig7_integrated_priority_tier.geojson'), driver='GeoJSON')

    # ── Figure 7 ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f'Figure 7 — Integrated Multi-criterion Priority Tier Map\n'
        f'{cfg["study_area"]["name"]} Placer Mine Tailings REE Assessment',
        fontsize=13, fontweight='bold',
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # Panel A: priority score map
    ax1 = fig.add_subplot(gs[0, 0:2])
    b = cfg['study_area']['bbox']

    for patch in patches:
        lr, lt = patch['lon_range'], patch['lat_range']
        ax1.fill([lr[0], lr[1], lr[1], lr[0]], [lt[0], lt[0], lt[1], lt[1]],
                 color=patch['color'], alpha=0.4, zorder=0)

    nure_anom = nure_gdf[nure_gdf['th_anomaly'] == True]
    ax1.scatter(nure_anom['lon'], nure_anom['lat'],
                c='#fdbb84', s=12, alpha=0.5, zorder=1, label='NURE Th anomaly')

    scores = merged_sorted['combined_score'].values
    norm_s = (scores - scores.min()) / (scores.max() - scores.min() + 0.01)
    for i, (_, row) in enumerate(merged_sorted.iterrows()):
        color  = tier_color(row['combined_score'])
        marker = '*' if row['rank'] <= 3 else 'o'
        size   = 200 if row['rank'] <= 3 else 80 + 120 * norm_s[i]
        ax1.scatter(row.lon, row.lat, c=color, s=size,
                    edgecolors='black', linewidths=0.8, zorder=5, marker=marker)
        ax1.annotate(f"#{row['rank']} {row['name'].split()[0]}",
                     (row.lon, row.lat), xytext=(5, 4), textcoords='offset points',
                     fontsize=7.5, fontweight='bold', color='black')

    ax1.set_xlim(b['lon_min'] - 0.1, b['lon_max'] + 0.1)
    ax1.set_ylim(b['lat_min'] - 0.1, b['lat_max'] + 0.2)
    ax1.set_xlabel('Longitude', fontsize=11); ax1.set_ylabel('Latitude', fontsize=11)
    ax1.tick_params(labelsize=9)
    ax1.set_title('A.  Multi-criterion combined priority score\n(★ = top 3 sites; circle size scales with score)', fontsize=9)
    ax1.grid(True, alpha=0.2)

    tier_patches = [
        mpatches.Patch(facecolor=WONG['blue'],   edgecolor='black', label=f'Score ≥ {tiers[0]} (highest)'),
        mpatches.Patch(facecolor=WONG['orange'],  edgecolor='black', label=f'Score {tiers[1]}–{tiers[0]} (high)'),
        mpatches.Patch(facecolor=WONG['yellow'],  edgecolor='black', label=f'Score {tiers[2]}–{tiers[1]} (moderate)'),
        mpatches.Patch(facecolor='#CCCCCC',       edgecolor='black', label=f'Score < {tiers[2]} (low)'),
    ]
    ax1.legend(handles=tier_patches, loc='lower left', fontsize=8)
    for cl in cnty_lbls:
        ax1.text(cl['lon'], cl['lat'], cl['label'], fontsize=7, color='gray',
                 style='italic', ha='center')

    # Panel B: radar chart
    ax2 = fig.add_subplot(gs[0, 2], polar=True)
    categories = ['Th Source\n(0-2)', 'Mag High\n(0-1)', 'Source\nLith (0-3)',
                  'Coverage\n(0-2)', 'NdPr\nVolume (0-2)', 'Au/As\nPathfinder (0-2)']
    N      = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, size=7)
    ax2.set_ylim(0, 3)
    ax2.set_yticks([1,2,3]); ax2.set_yticklabels(['1','2','3'], size=6)

    top5_colors = [WONG['blue'], WONG['orange'], WONG['green'], WONG['yellow'], '#CCCCCC']
    for idx, (_, row) in enumerate(merged_sorted.head(5).iterrows()):
        vals = [row['score_th'], row['score_mag'], row['score_lith'],
                row['score_conf'], row['score_ndpr'], row['score_au']] + [row['score_th']]
        ax2.plot(angles, vals, color=top5_colors[idx], lw=2, ls='-')
        ax2.fill(angles, vals, color=top5_colors[idx], alpha=0.1)

    top5_names = list(merged_sorted.head(5)['name'])
    ax2.legend(handles=[plt.Line2D([0],[0], color=c, lw=2.5, label=n)
                        for c, n in zip(top5_colors, top5_names)],
               loc='upper center', fontsize=6.5, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax2.set_title('B.  Criterion profile\n(top 5 sites)', fontsize=9, pad=15)

    # Panel C: ranked stacked bar
    ax3 = fig.add_subplot(gs[1, 0:2])
    score_comps  = ['score_th','score_mag','score_lith','score_conf','score_ndpr','score_au']
    comp_labels  = ['Th source (monazite)','Magnetic high','Source lithology',
                    'Data coverage','NdPr volume','Au/As pathfinder (Task 7)']
    comp_colors  = [WONG['green'], WONG['blue'], WONG['sky'], WONG['pink'], WONG['orange'], WONG['vermillion']]
    sorted_df = merged_sorted.sort_values('combined_score', ascending=True).tail(12)
    y_pos    = np.arange(len(sorted_df))
    bottoms  = np.zeros(len(sorted_df))
    for comp, label, color in zip(score_comps, comp_labels, comp_colors):
        vals = sorted_df[comp].values
        ax3.barh(y_pos, vals, left=bottoms, color=color, label=label, edgecolor='white', lw=0.3)
        bottoms += vals
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([f"#{r['rank']} {r['name']}" for _, r in sorted_df.iterrows()], fontsize=8)
    ax3.set_xlabel('Combined priority score', fontsize=11)
    ax3.tick_params(labelsize=9)
    ax3.set_title('C.  Score breakdown by criterion (all sites, ranked)\n(threshold ≥ 4.0)', fontsize=9)
    ax3.legend(fontsize=8, loc='lower right')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.axvline(tiers[2], color='red', ls='--', lw=1.2)
    ax3.set_xlim(0, 14)

    # Panel D: NdPr tonnes vs break-even price
    ax4 = fig.add_subplot(gs[1, 2])
    conf_cmap = {'HIGH': WONG['green'], 'MEDIUM': WONG['orange'], 'LOW': WONG['vermillion']}
    for _, row in task4_df.iterrows():
        be = task5_df[task5_df['site'] == row['site_name']]['breakeven_$/kg'].values
        if len(be) > 0 and pd.notna(be[0]):
            ax4.scatter(be[0], row['ndpr_tonnes'],
                        c=conf_cmap.get(row['confidence'], 'gray'),
                        s=100, edgecolors='black', linewidths=0.8, zorder=4)
            ax4.annotate(row['site_name'].split()[0], (be[0], row['ndpr_tonnes']),
                         xytext=(3,3), textcoords='offset points', fontsize=7)
    ax4.axvline(ndpr_cur, color=WONG['blue'], ls='--', lw=2,
                label=f'Current NdPr price (${ndpr_cur:.0f}/kg)')
    ax4.axvline(cfg['economics']['ndpr_price_low'], color='gray', ls=':', lw=1.5,
                label=f'2024 trough (${cfg["economics"]["ndpr_price_low"]:.0f}/kg)')
    ax4.set_xlabel('Break-even NdPr price ($/kg oxide)', fontsize=11)
    ax4.set_ylabel('NdPr metal (estimated tonnes)', fontsize=11)
    ax4.tick_params(labelsize=9)
    ax4.set_title('D.  Break-even vs. NdPr endowment\n(color = data confidence)', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.legend(handles=[mpatches.Patch(facecolor=c, edgecolor='black', label=l)
                        for l, c in conf_cmap.items()], fontsize=7, loc='upper right')

    fig.text(0.5, 0.01,
             f'{cfg["study_area"]["name"]} REE Tailings Assessment — Au+REE Pipeline — EXPLORATION TARGET ONLY',
             ha='center', fontsize=7, color='gray', style='italic')

    watermark(fig, cfg)
    save_fig(fig, out(cfg, 'figures', 'fig7_integrated_priority_map.png'))

    # ── Executive summary ─────────────────────────────────────────────────────
    top3    = merged_sorted.head(3)
    t4_idx  = task4_df.set_index('site_name')
    t5_idx  = task5_df.set_index('site')

    ec = cfg['economics']
    hdr = [
        "EXECUTIVE SUMMARY — TOP 3 PRIORITY SITES",
        f"{cfg['study_area']['name']} Placer Mine Tailings REE Assessment",
        "Intended for: field sampling decision-maker",
        "="*68,
        "",
        "BACKGROUND",
        "",
        f"This project evaluated sites in {cfg['study_area']['name']} for potential REE",
        "content (monazite-hosted NdPr). All grade estimates are proxy-based.",
        "",
        f"NdPr context: ~${ec['ndpr_price_central']:.0f}/kg current, ${ec['ndpr_price_low']:.0f}/kg 2024 trough.",
        f"Processor: {ec.get('processor_name','Energy Fuels White Mesa Mill')} ({ec.get('processor_location','Blanding, UT')})",
        "="*68,
    ]

    body = []
    for rank, (_, site) in enumerate(top3.iterrows(), 1):
        name = site['name']
        t4   = t4_idx.loc[name] if name in t4_idx.index else pd.Series()
        t5   = t5_idx.loc[name] if name in t5_idx.index else pd.Series()
        body += [
            "",
            f"SITE #{rank}: {name.upper()}",
            "="*60,
            f"  Coordinates:    {site.lat:.4f}°N, {site.lon:.4f}°W",
            f"  Combined score: {site.combined_score:.1f}",
            f"  Th source:      {site.get('th_source','N/A')}",
            f"  Mag high:       {'YES' if site.get('mag_high') else 'NO'}",
            f"  Source lith:    {site.get('source_lith_desc','N/A')}",
            f"  Est. tonnage:   {int(t4.get('tonnage_t', 0)):,} t" if len(t4) else "  Est. tonnage:   N/A",
            f"  NdPr grade:     ~{int(t4.get('ndpr_ppm', 0))} ppm (±50%)" if len(t4) else "  NdPr grade:     N/A",
            f"  NdPr metal:     {t4.get('ndpr_tonnes', 0):.0f} t" if len(t4) else "  NdPr metal:     N/A",
        ]
        if len(t5) > 0:
            body += [
                f"  Break-even:     ${t5.get('breakeven_$/kg', 0):.0f}/kg NdPr",
                f"  NPV (central):  ${t5.get('npv_central_$M', 0):.0f}M (undiscounted; ESTIMATE ONLY)",
            ]
        body += [
            "",
            "  NOTE: Exploration target only; not a resource estimate. No field",
            "  sampling conducted. Verify by auger program before any investment.",
        ]

    footer = [
        "",
        "="*68,
        "IMPORTANT DISCLAIMERS",
        "",
        "Exploration target only. Estimates have not been verified by in-situ",
        "sampling. Do not use for investment decisions, property transactions,",
        "or regulatory filings. Does not comply with NI 43-101 or JORC Code.",
        "All NdPr prices as of June 2026; may change substantially.",
        "="*68,
    ]

    with open(out(cfg, 'text', 'executive_summary_top3_sites.txt'), 'w') as f:
        f.write('\n'.join(hdr + body + footer))

    # ── Weight sensitivity (one-at-a-time ±50%) ──────────────────────────────
    _WEIGHT_KEYS  = ['th_source_monazite', 'magnetic_high', 'source_lith',
                     'coverage', 'ndpr_volume', 'au_pathfinder']
    _SCORE_COLS   = ['score_th', 'score_mag', 'score_lith',
                     'score_conf', 'score_ndpr', 'score_au']
    _DEFAULTS     = [1.0, 1.0, 1.0, 1.0, 2.0, 1.0]
    base_w        = [w.get(k, d) for k, d in zip(_WEIGHT_KEYS, _DEFAULTS)]

    # Collect rank for each site under every perturbation
    site_names   = list(merged_sorted['name'])
    all_ranks    = {n: [int(merged_sorted.loc[merged_sorted['name'] == n, 'rank'].iloc[0])]
                    for n in site_names}

    for wi in range(len(_WEIGHT_KEYS)):
        for factor in (0.5, 1.5):
            pw = list(base_w)
            pw[wi] = base_w[wi] * factor
            perturbed_score = sum(
                merged[sc].values * pw[i] for i, sc in enumerate(_SCORE_COLS)
            )
            order = np.argsort(perturbed_score)[::-1]
            for rank_pos, idx in enumerate(order, 1):
                all_ranks[merged.iloc[idx]['name']].append(rank_pos)

    sens_rows = []
    for _, row in merged_sorted.iterrows():
        ranks = all_ranks[row['name']]
        sens_rows.append({
            'site_name':   row['name'],
            'nominal_rank': int(row['rank']),
            'min_rank':    int(min(ranks)),
            'max_rank':    int(max(ranks)),
            'rank_range':  int(max(ranks) - min(ranks)),
            'stable_top3': bool(max(ranks) <= 3),
        })

    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(out(cfg, 'tables', 'integration_weight_sensitivity.csv'), index=False)

    top3_names   = list(merged_sorted.head(3)['name'])
    stable_top3  = [n for n in top3_names if sens_df.loc[sens_df['site_name'] == n, 'stable_top3'].iloc[0]]
    unstable     = [n for n in top3_names if n not in stable_top3]

    print("\nWEIGHT SENSITIVITY (one-at-a-time ±50%):")
    print(sens_df[['site_name','nominal_rank','min_rank','max_rank','rank_range']].to_string(index=False))
    if len(stable_top3) == 3:
        print(f"\n  TOP-3 RANK IS STABLE: {', '.join(top3_names)} remain in top-3 "
              f"across all weight perturbations.")
    elif stable_top3:
        print(f"\n  PARTIALLY STABLE: {', '.join(stable_top3)} remain top-3 in all "
              f"perturbations; {', '.join(unstable)} can fall outside top-3.")
    else:
        print(f"\n  NOTE: Top-3 ranking is sensitive to weight choice. "
              f"Nominal top-3: {', '.join(top3_names)}.")

    print("\nINTEGRATION COMPLETE")
    print(merged_sorted[['rank','name','combined_score','score_th','score_mag',
                          'score_lith','score_conf','score_ndpr','score_au',
                          'ndpr_tonnes']].to_string(index=False))


if __name__ == '__main__':
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/ne_washington/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
