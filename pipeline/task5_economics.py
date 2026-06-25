"""
Task 5: Processing pathway break-even analysis.

Output:
  {outputs_dir}/text/task5_energy_fuels_paper_section.txt
  {outputs_dir}/tables/task5_breakeven_analysis.csv
  {outputs_dir}/figures/fig5_breakeven_sensitivity.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

from pipeline.utils import WONG, setup_mpl, watermark, save_fig, ensure_outputs, out


def run(cfg):
    setup_mpl()
    ensure_outputs(cfg['outputs_dir'])

    ec = cfg['economics']
    PROC_NAME   = ec.get('processor_name', 'White Mesa Mill')
    PROC_LOC    = ec.get('processor_location', 'Blanding, UT')
    NDPR_PRICE_CENTRAL = ec.get('ndpr_price_central', 109.0)
    NDPR_PRICE_LOW     = ec.get('ndpr_price_low', 60.0)
    NDPR_PRICE_HIGH    = ec.get('ndpr_price_high', 140.0)
    Y2O3_PRICE_CENTRAL = ec.get('y2o3_price_central', 3.50)
    Y2O3_PRICE_LOW     = ec.get('y2o3_price_low', 2.50)
    Y2O3_PRICE_HIGH    = ec.get('y2o3_price_high', 5.50)
    Y2O3_RECOVERY      = ec.get('y2o3_recovery', 0.75)
    TAILINGS_HANDLING  = ec.get('tailings_handling_cost', 5.0)
    CONCENTR_COST      = ec.get('concentration_cost', 10.0)
    LOGISTICS_CONC     = ec.get('logistics_conc_per_tonne', 150.0)
    PROCESSING_PER_KG  = ec.get('processing_cost_per_kg', 50.0)
    NDPR_RECOVERY      = ec.get('ndpr_recovery', 0.80)

    # ── Paper section ─────────────────────────────────────────────────────────
    paper_section = f"""
5.4  Domestic Processing Pathway and Supply Chain Context

{cfg['study_area']['name']} placer mine tailings — processing pathway via {PROC_NAME}
({PROC_LOC}) — the only U.S. facility licensed to convert monazite concentrate into
separated rare earth oxides (as of 2026).

NdPr market context:
  Current spot (June 2026): ~${NDPR_PRICE_CENTRAL:.0f}/kg NdPr oxide
  DOD floor (MP Materials):  $110/kg
  2024 trough:               ~${NDPR_PRICE_LOW:.0f}/kg
  Q1 2026 peak:              ~${NDPR_PRICE_HIGH:.0f}/kg

[Full market narrative: see ne_wa_ree/task5_energy_fuels_pathway.py paper_section variable]
"""
    txt_path = out(cfg, 'text', 'task5_energy_fuels_paper_section.txt')
    with open(txt_path, 'w') as f:
        f.write(paper_section)
    print("Paper section written")

    # ── Break-even analysis — ALL sites ──────────────────────────────────────
    # Run every site from task4 so the figure shows the full economic landscape.
    # Ranking is by NPV at central NdPr price; sites are NOT filtered by NdPr endowment.
    t4 = pd.read_csv(out(cfg, 'tables', 'task4_volume_tonnage_summary.csv'))
    t4 = t4.dropna(subset=['ndpr_tonnes'])
    all_sites = [
        {'name': row['site_name'], 'tonnage_t': int(row['tonnage_t']),
         'ndpr_ppm': float(row['ndpr_ppm']), 'confidence': row['confidence']}
        for _, row in t4.iterrows()
    ]

    t4_idx = t4.set_index('site_name')
    results = []
    for site in all_sites:
        name     = site['name']
        tonnage  = site['tonnage_t']
        ndpr_ppm = site['ndpr_ppm']

        ndpr_metal_t  = tonnage * ndpr_ppm / 1e6
        ndpr_oxide_t  = ndpr_metal_t * 1.166 * NDPR_RECOVERY
        ndpr_frac_mnz = 0.09
        mnz_ppm       = ndpr_ppm / ndpr_frac_mnz
        mnz_tonnes    = tonnage * mnz_ppm / 1e6
        mnz_conc_t    = mnz_tonnes * 0.90

        cost_handling  = tonnage * TAILINGS_HANDLING
        cost_concentr  = tonnage * CONCENTR_COST
        cost_logistics = mnz_conc_t * LOGISTICS_CONC
        cost_processing = ndpr_oxide_t * 1000 * PROCESSING_PER_KG
        total_cost     = cost_handling + cost_concentr + cost_logistics + cost_processing

        # Y₂O₃ co-product revenue (xenotime pathway; only for sites with y2o3_t_p50 > 0)
        _t4_row = t4_idx.loc[name] if name in t4_idx.index else pd.Series()
        _y2o3_t_p50 = float(_t4_row.get('y2o3_t_p50', 0.0)) if len(_t4_row) > 0 and pd.notna(_t4_row.get('y2o3_t_p50', 0.0)) else 0.0
        y2o3_oxide_t = _y2o3_t_p50 * Y2O3_RECOVERY  # already in tonnes
        y2o3_rev_central = y2o3_oxide_t * 1000 * Y2O3_PRICE_CENTRAL
        y2o3_rev_low     = y2o3_oxide_t * 1000 * Y2O3_PRICE_LOW
        y2o3_rev_high    = y2o3_oxide_t * 1000 * Y2O3_PRICE_HIGH

        rev_low     = ndpr_oxide_t * 1000 * NDPR_PRICE_LOW     + y2o3_rev_low
        rev_central = ndpr_oxide_t * 1000 * NDPR_PRICE_CENTRAL + y2o3_rev_central
        rev_high    = ndpr_oxide_t * 1000 * NDPR_PRICE_HIGH    + y2o3_rev_high

        npv_low     = rev_low     - total_cost
        npv_central = rev_central - total_cost
        npv_high    = rev_high    - total_cost

        breakeven_price = total_cost / (ndpr_oxide_t * 1000) if ndpr_oxide_t > 0 else 9999

        results.append({
            'site':              name,
            'tonnage_t':         tonnage,
            'ndpr_ppm':          ndpr_ppm,
            'ndpr_oxide_t':      round(ndpr_oxide_t, 0),
            'mnz_concentrate_t': round(mnz_conc_t, 0),
            'y2o3_t_p50':        round(_y2o3_t_p50, 1),
            'y2o3_co_rev_$M':    round(y2o3_rev_central/1e6, 2),
            'cost_handling_$M':  round(cost_handling/1e6, 1),
            'cost_concentr_$M':  round(cost_concentr/1e6, 1),
            'cost_logistics_$M': round(cost_logistics/1e6, 1),
            'cost_processing_$M':round(cost_processing/1e6, 1),
            'total_cost_$M':     round(total_cost/1e6, 1),
            'rev_low_$M':        round(rev_low/1e6, 1),
            'rev_central_$M':    round(rev_central/1e6, 1),
            'rev_high_$M':       round(rev_high/1e6, 1),
            'npv_low_$M':        round(npv_low/1e6, 1),
            'npv_central_$M':    round(npv_central/1e6, 1),
            'npv_high_$M':       round(npv_high/1e6, 1),
            'breakeven_$/kg':    round(breakeven_price, 1),
            'confidence':        site['confidence'],
        })

    results_df = pd.DataFrame(results)
    # Sort all sites by NPV at central price — this is the economically meaningful ranking.
    # NdPr endowment tonnage is NOT used as the sort key; sites with large volume but poor
    # grade or low data confidence can still have low NPV.
    results_df = results_df.sort_values('npv_central_$M', ascending=False).reset_index(drop=True)
    results_df.to_csv(out(cfg, 'tables', 'task5_breakeven_analysis.csv'), index=False)
    print("\nBreak-even analysis (all sites, ranked by NPV at central price):")
    print(results_df[['site','total_cost_$M','npv_central_$M','breakeven_$/kg']].to_string(index=False))

    viable = results_df['breakeven_$/kg'] <= NDPR_PRICE_CENTRAL
    print(f"\nViable at current ${NDPR_PRICE_CENTRAL:.0f}/kg: {viable.sum()} of {len(results_df)} sites")
    for _, r in results_df[viable].iterrows():
        print(f"  {r['site']}: ${r['breakeven_$/kg']:.0f}/kg → VIABLE")
    for _, r in results_df[~viable].iterrows():
        print(f"  {r['site']}: ${r['breakeven_$/kg']:.0f}/kg → NOT VIABLE at current price")

    # ── Figure 5 ──────────────────────────────────────────────────────────────
    # Color palette: viable sites get distinct WONG colors; non-viable get gray shades.
    _wong_colors = [WONG['blue'], WONG['orange'], WONG['green'], WONG['sky'],
                    WONG['vermillion'], WONG['yellow'], WONG['pink']]
    viable_sites    = results_df[viable].reset_index(drop=True)
    nonviable_sites = results_df[~viable].reset_index(drop=True)
    site_color_map = {}
    for i, name in enumerate(viable_sites['site']):
        site_color_map[name] = _wong_colors[i % len(_wong_colors)]
    for i, name in enumerate(nonviable_sites['site']):
        site_color_map[name] = f'#{180 - i*15:02x}{180 - i*15:02x}{180 - i*15:02x}'

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f'Figure 5 — Break-even Analysis: {cfg["study_area"]["name"]} Placer Monazite Tailings\n'
        f'All {len(results_df)} sites • ranked by NPV at current NdPr price (${NDPR_PRICE_CENTRAL:.0f}/kg)',
        fontsize=12, fontweight='bold',
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # ── Panel A: NPV vs price curves, ALL sites ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    prices = np.linspace(30, 200, 150)
    for _, row in results_df.iterrows():
        ndpr_oxide_t = row['ndpr_oxide_t']
        total_cost   = row['total_cost_$M'] * 1e6
        is_viable    = row['breakeven_$/kg'] <= NDPR_PRICE_CENTRAL
        npvs = [(ndpr_oxide_t * 1000 * p - total_cost) / 1e6 for p in prices]
        _y_flag = ' ★Y' if row.get('y2o3_t_p50', 0) > 0 else ''
        color = site_color_map[row['site']]
        lw    = 2.5 if is_viable else 1.2
        ls    = '-'  if is_viable else '--'
        alpha = 1.0  if is_viable else 0.55
        ax1.plot(prices, npvs, color=color, lw=lw, ls=ls, alpha=alpha,
                 label=f"{row['site']}{_y_flag} (BE: ${row['breakeven_$/kg']:.0f}/kg)")
    ax1.axvline(NDPR_PRICE_LOW,     color='#555555', ls='--', lw=1.8, alpha=0.8,
                label=f'2024 trough (${NDPR_PRICE_LOW:.0f}/kg)')
    ax1.axvline(NDPR_PRICE_CENTRAL, color='black',   ls='-',  lw=2.5,
                label=f'Current spot (${NDPR_PRICE_CENTRAL:.0f}/kg)')
    ax1.axvline(NDPR_PRICE_HIGH,    color='#555555', ls='-.',  lw=1.8, alpha=0.8,
                label=f'Q1 2026 peak (${NDPR_PRICE_HIGH:.0f}/kg)')
    ax1.axhline(0, color='black', lw=1.0)
    _npv_min = min((row['ndpr_oxide_t'] * 1000 * 30 - row['total_cost_$M'] * 1e6) / 1e6
                   for _, row in results_df.iterrows()) * 1.05
    ax1.fill_between(prices, _npv_min, 0,
                     where=np.ones(len(prices), dtype=bool), alpha=0.04, color='red')
    ax1.set_xlabel('NdPr oxide price ($/kg)', fontsize=11)
    ax1.set_ylabel('Undiscounted NPV ($M)', fontsize=11)
    ax1.tick_params(labelsize=9)
    ax1.set_title('A.  NPV vs. NdPr price — all sites\n'
                  '(solid = viable at $109/kg; dashed = not viable; ★Y = Y₂O₃ co-product site)', fontsize=9)
    ax1.legend(fontsize=7.5, loc='upper left', ncol=2)
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(30, 200)

    # ── Panel B: horizontal NPV bar at central price (ranked) ─────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    bar_df = results_df.sort_values('npv_central_$M', ascending=True)  # ascending for horizontal bar (bottom = best)
    y_pos  = np.arange(len(bar_df))
    bar_colors = [site_color_map[s] for s in bar_df['site']]
    ax2.barh(y_pos, bar_df['npv_central_$M'], color=bar_colors,
             edgecolor='white', lw=0.4, alpha=0.85)
    ax2.axvline(0, color='black', lw=1.0)
    # Annotate breakeven on each bar
    for i, (_, row) in enumerate(bar_df.iterrows()):
        be = row['breakeven_$/kg']
        npv = row['npv_central_$M']
        _y_flag = ' ★Y' if row.get('y2o3_t_p50', 0) > 0 else ''
        ax2.text(max(npv, 0) + 0.3, i,
                 f"BE: ${be:.0f}/kg{_y_flag}", va='center', fontsize=7.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(bar_df['site'].str.replace(' Placer','').str.replace(' Mine',''),
                        fontsize=8)
    ax2.set_xlabel('NPV at current NdPr price $109/kg ($M, undiscounted)', fontsize=10)
    ax2.set_title('B.  All sites ranked by NPV\n'
                  f'(at ${NDPR_PRICE_CENTRAL:.0f}/kg; BE = break-even price; ★Y = Y₂O₃ co-product)', fontsize=9)
    ax2.grid(True, alpha=0.25, axis='x')

    ax3 = fig.add_subplot(gs[1, 1])
    # Tornado: use the #1 priority target (Hunters Placer) if viable, else top-NPV viable site.
    # Hunters is #1 by multi-criterion combined score (Fig 7); Sanpoil is only #4 despite
    # having the most NdPr tonnes, because it has LOW data confidence and no Au signal.
    _priority_order = ['Hunters Placer', 'Colville Placer', 'Conconully Placer',
                       'Sanpoil River Placer']
    top_site = None
    for pname in _priority_order:
        _match = results_df[(results_df['site'] == pname) &
                            (results_df['breakeven_$/kg'] <= NDPR_PRICE_CENTRAL)]
        if not _match.empty:
            top_site = _match.iloc[0]
            break
    if top_site is None:
        top_site = results_df[viable].iloc[0] if viable.any() else results_df.iloc[0]
    base_npv     = top_site['npv_central_$M']
    ndpr_oxide_t = top_site['ndpr_oxide_t']
    total_cost_base = top_site['total_cost_$M']

    sensitivities = {
        'NdPr price\n(±30%)': [
            ndpr_oxide_t*1000*NDPR_PRICE_CENTRAL*0.7/1e6 - total_cost_base,
            ndpr_oxide_t*1000*NDPR_PRICE_CENTRAL*1.3/1e6 - total_cost_base,
        ],
        'Grade\n(±50%)': [
            ndpr_oxide_t*0.5*1000*NDPR_PRICE_CENTRAL/1e6 - total_cost_base,
            ndpr_oxide_t*1.5*1000*NDPR_PRICE_CENTRAL/1e6 - total_cost_base,
        ],
        'Tonnage\n(±30%)': [
            ndpr_oxide_t*1000*NDPR_PRICE_CENTRAL/1e6 - total_cost_base*1.3,
            ndpr_oxide_t*1000*NDPR_PRICE_CENTRAL/1e6 - total_cost_base*0.7,
        ],
        'Processing cost\n(±25%)': [
            ndpr_oxide_t*1000*NDPR_PRICE_CENTRAL/1e6 - total_cost_base + top_site['cost_processing_$M']*0.25,
            ndpr_oxide_t*1000*NDPR_PRICE_CENTRAL/1e6 - total_cost_base - top_site['cost_processing_$M']*0.25,
        ],
        f'NdPr recovery\n({NDPR_RECOVERY-0.1:.2f}–{NDPR_RECOVERY+0.1:.2f})': [
            ndpr_oxide_t*(NDPR_RECOVERY-0.1)/NDPR_RECOVERY*1000*NDPR_PRICE_CENTRAL/1e6 - total_cost_base,
            ndpr_oxide_t*(NDPR_RECOVERY+0.1)/NDPR_RECOVERY*1000*NDPR_PRICE_CENTRAL/1e6 - total_cost_base,
        ],
    }
    labels = list(sensitivities.keys())
    lows   = [v[0] for v in sensitivities.values()]
    highs  = [v[1] for v in sensitivities.values()]
    widths = [h-l for l, h in zip(lows, highs)]
    order  = sorted(range(len(widths)), key=lambda i: widths[i], reverse=True)

    for i, idx in enumerate(order):
        lo, hi = lows[idx] - base_npv, highs[idx] - base_npv
        ax3.barh(i, abs(lo), left=base_npv + min(lo, 0),
                 color=WONG['vermillion'] if lo < 0 else WONG['green'], alpha=0.7)
        ax3.barh(i, abs(hi - lo), left=base_npv + lo,
                 color=WONG['green'] if hi > 0 else WONG['vermillion'], alpha=0.7)

    ax3.set_yticks(range(len(order)))
    ax3.set_yticklabels([labels[i] for i in order], fontsize=8)
    ax3.axvline(base_npv, color=WONG['blue'], lw=2.5, label=f'Base NPV = ${base_npv:.1f}M')
    ax3.set_xlabel('Undiscounted NPV ($M)', fontsize=11)
    ax3.tick_params(labelsize=9)
    ax3.set_title(f'C.  Tornado: NPV sensitivity\n{top_site["site"]} (#1 priority target by combined score)', fontsize=9)
    tornado_legend = [
        Patch(facecolor=WONG['green'], alpha=0.7, label='Upside (positive impact)'),
        Patch(facecolor=WONG['vermillion'], alpha=0.7, label='Downside (negative impact)'),
        plt.Line2D([0],[0], color=WONG['blue'], lw=2.5, label=f'Base NPV = ${base_npv:.1f}M'),
    ]
    ax3.legend(handles=tornado_legend, fontsize=7.5, loc='lower right')
    ax3.grid(True, alpha=0.3, axis='x')

    watermark(fig, cfg)
    save_fig(fig, out(cfg, 'figures', 'fig5_breakeven_sensitivity.png'))

    print(f"\nTornado focus: {top_site['site']} (#1 priority target by multi-criterion combined score)")


if __name__ == '__main__':
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/ne_washington/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
