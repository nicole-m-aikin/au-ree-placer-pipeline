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

    # ── Break-even analysis ───────────────────────────────────────────────────
    t4 = pd.read_csv(out(cfg, 'tables', 'task4_volume_tonnage_summary.csv'))
    t4 = t4.dropna(subset=['ndpr_tonnes']).sort_values('ndpr_tonnes', ascending=False)
    top3 = t4.head(3)
    top3_sites = [
        {'name': row['site_name'], 'tonnage_t': int(row['tonnage_t']),
         'ndpr_ppm': float(row['ndpr_ppm']), 'confidence': row['confidence']}
        for _, row in top3.iterrows()
    ]

    results = []
    for site in top3_sites:
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

        rev_low     = ndpr_oxide_t * 1000 * NDPR_PRICE_LOW
        rev_central = ndpr_oxide_t * 1000 * NDPR_PRICE_CENTRAL
        rev_high    = ndpr_oxide_t * 1000 * NDPR_PRICE_HIGH

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
    results_df.to_csv(out(cfg, 'tables', 'task5_breakeven_analysis.csv'), index=False)
    print("\nBreak-even analysis:")
    print(results_df[['site','total_cost_$M','npv_central_$M','breakeven_$/kg']].to_string(index=False))

    # ── Figure 5 ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        f'Figure 5 — Break-even Analysis: {cfg["study_area"]["name"]} Placer Monazite Tailings\n'
        f'NdPr Price and Cost Sensitivity (Top 3 Economic Sites)',
        fontsize=12, fontweight='bold',
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    site_colors = [WONG['blue'], WONG['orange'], WONG['green']]

    ax1 = fig.add_subplot(gs[0, 0:2])
    prices = np.linspace(30, 200, 100)
    for idx, row in results_df.iterrows():
        ndpr_oxide_t = row['ndpr_oxide_t']
        total_cost   = row['total_cost_$M'] * 1e6
        npvs = [(ndpr_oxide_t * 1000 * p - total_cost)/1e6 for p in prices]
        ax1.plot(prices, npvs, color=site_colors[idx], lw=2.5,
                 label=f"{row['site']} (BE: ${row['breakeven_$/kg']:.0f}/kg)")
        be = row['breakeven_$/kg']
        be_color = WONG['green'] if be < 90 else (WONG['orange'] if be < NDPR_PRICE_CENTRAL else WONG['vermillion'])
        ax1.axvline(be, color=be_color, ls=':', lw=1.2, alpha=0.8)

    ax1.axvline(NDPR_PRICE_LOW, color='gray', ls='--', lw=2, alpha=0.85,
                label=f'2024 trough (${NDPR_PRICE_LOW:.0f}/kg)')
    ax1.axvline(NDPR_PRICE_CENTRAL, color=WONG['blue'], ls='-', lw=2.5,
                label=f'Current spot (${NDPR_PRICE_CENTRAL:.0f}/kg)')
    ax1.axvline(NDPR_PRICE_HIGH, color='gray', ls='-.', lw=2, alpha=0.85,
                label=f'Q1 2026 peak (${NDPR_PRICE_HIGH:.0f}/kg)')
    ax1.axhline(0, color='black', lw=0.8)
    ax1.fill_between(prices, 0, 1e10, where=(prices >= NDPR_PRICE_LOW), alpha=0.05, color=WONG['green'])
    ax1.set_xlabel('NdPr oxide price ($/kg)', fontsize=11)
    ax1.set_ylabel('Undiscounted NPV ($M)', fontsize=11)
    ax1.tick_params(labelsize=9)
    ax1.set_title('A.  Undiscounted NPV vs. NdPr oxide price\n'
                  '(current cost assumptions; EXPLORATION ESTIMATE ONLY)', fontsize=9)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(30, 200)
    ax1.set_ylim(bottom=0, top=results_df['rev_high_$M'].max() * 1.1)

    ax2 = fig.add_subplot(gs[1, 0])
    cost_components = ['cost_handling_$M','cost_concentr_$M','cost_logistics_$M','cost_processing_$M']
    cost_labels = ['Tailings handling', 'Concentration', f'Logistics to {PROC_NAME}', f'{PROC_NAME} processing']
    cost_colors = [WONG['blue'], WONG['sky'], WONG['yellow'], WONG['vermillion']]
    bottoms = np.zeros(len(results_df))
    x = np.arange(len(results_df))
    for comp, label, col in zip(cost_components, cost_labels, cost_colors):
        vals = results_df[comp].values
        ax2.bar(x, vals, bottom=bottoms, color=col, label=label, edgecolor='white', lw=0.5)
        bottoms += vals
    for idx, row in results_df.iterrows():
        ax2.bar(idx, row['rev_central_$M'], bottom=0, fill=False,
                edgecolor='black', lw=2.5, linestyle='-',
                label='Revenue (central)' if idx == 0 else '')
    ax2.set_xticks(x)
    ax2.set_xticklabels([r['site'].split()[0] + '\n' + r['site'].split()[1]
                         if len(r['site'].split()) > 1 else r['site']
                         for _, r in results_df.iterrows()], fontsize=8)
    ax2.set_ylabel('$M (undiscounted)', fontsize=11)
    ax2.tick_params(labelsize=9)
    ax2.set_title(f'B.  Cost breakdown vs. revenue\n(dashed = central revenue at ${NDPR_PRICE_CENTRAL:.0f}/kg)', fontsize=9)
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')

    ax3 = fig.add_subplot(gs[1, 1])
    top_site = results_df.iloc[0]
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
            ndpr_oxide_t*1000*NDPR_PRICE_CENTRAL/1e6 - total_cost_base*0.75 + top_site['cost_processing_$M']*0.25,
            ndpr_oxide_t*1000*NDPR_PRICE_CENTRAL/1e6 - total_cost_base*1.25 - top_site['cost_processing_$M']*0.25,
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
    ax3.axvline(base_npv, color='black', lw=1.5, label=f'Base NPV = ${base_npv:.0f}M')
    ax3.set_xlabel('Undiscounted NPV ($M)', fontsize=11)
    ax3.tick_params(labelsize=9)
    ax3.set_title(f'C.  Tornado: NPV sensitivity\n{top_site["site"]} (top site by NdPr metal)', fontsize=9)
    tornado_legend = [
        Patch(facecolor=WONG['green'], alpha=0.7, label='Upside (positive impact)'),
        Patch(facecolor=WONG['vermillion'], alpha=0.7, label='Downside (negative impact)'),
        plt.Line2D([0],[0], color='black', lw=1.5, label=f'Base NPV = ${base_npv:.0f}M'),
    ]
    ax3.legend(handles=tornado_legend, fontsize=7.5, loc='lower right')
    ax3.grid(True, alpha=0.3, axis='x')

    watermark(fig, cfg)
    save_fig(fig, out(cfg, 'figures', 'fig5_breakeven_sensitivity.png'))

    print(f"\nBreak-even prices for top 3 sites:")
    for _, row in results_df.iterrows():
        status = "VIABLE" if row['breakeven_$/kg'] < NDPR_PRICE_CENTRAL else "MARGINAL"
        print(f"  {row['site']}: ${row['breakeven_$/kg']:.0f}/kg → {status} at current prices")


if __name__ == '__main__':
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/ne_washington/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
