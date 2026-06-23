"""
TASK 5: Energy Fuels processing pathway — paper section + break-even analysis
Generates:
  outputs/text/task5_energy_fuels_paper_section.txt
  outputs/tables/task5_breakeven_analysis.csv
  outputs/figures/fig5_breakeven_sensitivity.png
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

# ── Paper section text ────────────────────────────────────────────────────────
paper_section = """
5.4  Domestic Processing Pathway and Supply Chain Context

The identification and characterization of potential monazite-bearing tailings in
northeastern Washington is directly relevant to an emerging domestic rare earth
element (REE) processing infrastructure that did not exist five years ago. For
decades, the combination of low NdPr prices, regulatory constraints associated
with the thorium content of monazite, and the dominance of Chinese processing
capacity made domestic monazite recovery economically unviable. That situation
has changed materially.

Energy Fuels Inc. operates the White Mesa Mill in Blanding, Utah — as of 2026,
the only facility in the United States licensed to convert monazite concentrate
into separated rare earth oxides (Energy Fuels 2025a). The mill's Phase 1A REE
circuit has demonstrated commercial-scale capacity to process up to 10,000 tonnes
of monazite concentrate per year, producing up to 1,000 tonnes of NdPr oxide
annually alongside uranium and thorium byproduct recovery (Energy Fuels 2025b;
Crux Investor 2026). Current feedstock is supplied by The Chemours Company from
heavy mineral sand operations in Florida and Georgia, providing approximately 500
tonnes of monazite per year (Energy Fuels 2025a). A Phase 2 expansion, with a
bankable feasibility study completed in January 2026 at an NPV of $1.9 billion
(8% discount rate), would increase capacity to approximately 60,000 tonnes of
monazite per year and produce over 6,000 tonnes of NdPr oxide alongside
commercial-scale dysprosium and terbium output (Discovery Alert 2026).

Two features of the White Mesa Mill are particularly relevant to the northeastern
Washington tailings resource characterization developed here. First, the mill's
NRC license explicitly covers the processing of thorium-bearing materials, a
regulatory distinction that eliminates the principal barrier that historically
prevented domestic monazite processing: the thorium liability that accumulates in
the processing waste stream. Energy Fuels routes thorium byproducts through its
existing uranium and alternate feed processing infrastructure, converting a
regulatory obstacle into an operational co-product (Energy Fuels 2021). This
resolves the processing pathway question that has been the primary economic
constraint on domestic monazite recovery since the 1990s. Second, the mill has
publicly indicated interest in expanding its domestic feedstock base. The current
Chemours agreement supplies approximately 5% of Phase 1A capacity, leaving
substantial throughput available for additional domestic sources (Energy Fuels 2025b).

The market environment for NdPr has strengthened considerably since the trough
of 2023-2024. As of mid-2026, NdPr oxide is priced at approximately $109/kg
(Shanghai Metals Market, FOB benchmark), representing a more than 100% increase
from early-2025 levels, driven by supply deficits from Chinese production quota
constraints coinciding with sustained EV and wind turbine demand growth (Rare
Earth Mining 2026; Critical Minerals News 2026). The U.S. Department of Defense
has further underwritten domestic REE processing economics through a 10-year
offtake agreement with MP Materials establishing a $110/kg NdPr oxide floor price
— a policy signal that domestic supply chain development has government backing
regardless of Chinese market volatility (Rare Earth Exchanges 2025).

Against this backdrop, the northeastern Washington placer mine tailings resource
characterized in this study represents a potentially viable domestic feedstock
source. The sites prioritized here contain monazite in association with historical
placer gold operations, in tailings piles that have not previously been evaluated
for REE content. Processing the monazite component of these tailings would not
require new mining permits, as the material has already been extracted and
stockpiled. The thorium content, while a regulatory requirement, is manageable
under the White Mesa Mill's existing NRC license framework. Logistics are
straightforward: truck transport from northeastern Washington to Blanding, Utah
(approximately 1,000 km) is consistent with the distances already managed in the
Chemours supply chain from the U.S. Southeast.

The characterization presented here does not constitute a feedstock supply
commitment or a formal resource estimate. It is a spatial prioritization of
tailings sites warranting further investigation to determine whether any
individual pile contains sufficient monazite grade and tonnage to be economically
viable as a White Mesa Mill feedstock source. The exploration targets identified
in Section 5.3 suggest that at least three sites merit detailed in-situ sampling
to collapse the uncertainty currently associated with the stream sediment grade
proxy. The processing pathway to convert a confirmed in-situ grade estimate into
a feedstock viability assessment is now available domestically in a way that it
was not at the time the original mining operations deposited these tailings.

References for this section:
  Crux Investor (2026): Energy Fuels White Mesa Mill produces first US terbium oxide.
    April 15, 2026. cruxinvestor.com
  Critical Minerals News (2026): Rare earth price 2026. critical-minerals-news.com
  Discovery Alert (2026): Energy Fuels White Mesa Mill Navajo Nation Agreement Analysed.
    June 10, 2026. discoveryalert.com.au
  Energy Fuels (2021): Form 8-K, Neo and Energy Fuels collaboration. SEC Archives.
  Energy Fuels (2025a): Q3 2025 Results announcement. investors.energyfuels.com
  Energy Fuels (2025b): U.S.-Based Energy Fuels poised to produce six REE oxides.
    April 17, 2025. investors.energyfuels.com
  Rare Earth Exchanges (2025): Calculating a cost curve for U.S. rare earth production.
    August 13, 2025. rareearthexchanges.com
  Rare Earth Mining (2026): Rare earth market pricing analysis. rare-earth-mining.com
"""

with open('outputs/text/task5_energy_fuels_paper_section.txt', 'w') as f:
    f.write(paper_section)
print("Paper section written")

# ── Break-even analysis ───────────────────────────────────────────────────────
# Top 3 sites from Task 4 output (read live instead of hardcoding)
t4 = pd.read_csv('outputs/tables/task4_volume_tonnage_summary.csv')
t4 = t4.dropna(subset=['ndpr_tonnes']).sort_values('ndpr_tonnes', ascending=False)
top3 = t4.head(3)
top3_sites = [
    {'name': row['site_name'], 'tonnage_t': int(row['tonnage_t']),
     'ndpr_ppm': float(row['ndpr_ppm']), 'confidence': row['confidence']}
    for _, row in top3.iterrows()
]

# Cost parameters
# Processing costs for monazite concentrate: Energy Fuels target ~$40-60/kg NdPr oxide
# Mining/tailings handling: $3-8/tonne of tailings (no drilling; surface reclaim)
# Concentration (gravity/magnetic separation to monazite concentrate): $5-15/tonne feed
# Logistics (trucking NE WA to White Mesa Mill): ~$0.10/tonne-km × 1000 km = $100/tonne conc
# Assuming 1% monazite in tailings → 1 tonne concentrate per 100 tonne feed → $10,000/tonne conc logistics
# NdPr recovery from monazite: ~80% in processing; NdPr = 9% of monazite mass

NDPR_PRICE_CENTRAL = 109.0   # $/kg NdPr oxide (SMM benchmark, June 2026)
NDPR_PRICE_LOW     = 60.0    # $/kg (2024 trough)
NDPR_PRICE_HIGH    = 140.0   # $/kg (Q1 2026 peak)

TAILINGS_HANDLING_COST   = 5.0    # $/tonne tailings
CONCENTRATION_COST        = 10.0   # $/tonne tailings feed
LOGISTICS_CONC_PER_TONNE = 150.0  # $/tonne concentrate (to White Mesa)
PROCESSING_COST_PER_KG   = 50.0   # $/kg NdPr oxide produced
NDPR_RECOVERY             = 0.80   # mass recovery fraction

results = []
for site in top3_sites:
    name     = site['name']
    tonnage  = site['tonnage_t']
    ndpr_ppm = site['ndpr_ppm']

    # NdPr metal in tailings (tonnes)
    ndpr_metal_t = tonnage * ndpr_ppm / 1e6
    # Recoverable NdPr oxide (tonnes; oxide = metal × 1.166 for NdPr)
    ndpr_oxide_t = ndpr_metal_t * 1.166 * NDPR_RECOVERY
    # Monazite content
    ndpr_frac_in_mnz = 0.09
    mnz_content_ppm  = ndpr_ppm / ndpr_frac_in_mnz
    mnz_tonnes        = tonnage * mnz_content_ppm / 1e6
    # Monazite concentrate tonnage (assuming 90% recovery to concentrate)
    mnz_conc_t        = mnz_tonnes * 0.90

    # Costs
    cost_handling  = tonnage * TAILINGS_HANDLING_COST
    cost_concentr  = tonnage * CONCENTRATION_COST
    cost_logistics = mnz_conc_t * LOGISTICS_CONC_PER_TONNE
    cost_processing= ndpr_oxide_t * 1000 * PROCESSING_COST_PER_KG  # convert t to kg
    total_cost     = cost_handling + cost_concentr + cost_logistics + cost_processing

    # Revenue scenarios
    rev_low     = ndpr_oxide_t * 1000 * NDPR_PRICE_LOW
    rev_central = ndpr_oxide_t * 1000 * NDPR_PRICE_CENTRAL
    rev_high    = ndpr_oxide_t * 1000 * NDPR_PRICE_HIGH

    # NPV (undiscounted, ignoring time value for simplicity)
    npv_low     = rev_low     - total_cost
    npv_central = rev_central - total_cost
    npv_high    = rev_high    - total_cost

    # Break-even NdPr price
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
results_df.to_csv('outputs/tables/task5_breakeven_analysis.csv', index=False)
print("\nBreak-even analysis:")
print(results_df[['site','total_cost_$M','npv_central_$M','breakeven_$/kg']].to_string(index=False))

# ── Figure 5: Sensitivity tornado + price sensitivity ────────────────────────
fig = plt.figure(figsize=(16, 9))
fig.suptitle('Figure 5 — Break-even Analysis: NE Washington Placer Monazite Tailings\n'
             'NdPr Price and Cost Sensitivity (Top 3 Economic Sites: Sanpoil, Colville, Conconully)',
             fontsize=12, fontweight='bold')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Wong-palette site colors for the 3 lines
site_colors = ['#0072B2', '#E69F00', '#009E73']

# Panel A: NPV vs NdPr price (all 3 sites)
ax1 = fig.add_subplot(gs[0, 0:2])
prices = np.linspace(30, 200, 100)
for idx, row in results_df.iterrows():
    ndpr_oxide_t = row['ndpr_oxide_t']
    total_cost   = row['total_cost_$M'] * 1e6
    npvs = [(ndpr_oxide_t * 1000 * p - total_cost)/1e6 for p in prices]
    ax1.plot(prices, npvs, color=site_colors[idx], lw=2.5,
             label=f"{row['site']} (BE: ${row['breakeven_$/kg']:.0f}/kg)")
    # Break-even line color based on viability
    be = row['breakeven_$/kg']
    be_color = '#009E73' if be < 90 else ('#E69F00' if be < 109 else '#D55E00')
    ax1.axvline(be, color=be_color, ls=':', lw=1.2, alpha=0.8)

ax1.axvline(NDPR_PRICE_LOW, color='gray', ls='--', lw=2, alpha=0.85, label=f'2024 trough (${NDPR_PRICE_LOW:.0f}/kg)')
ax1.axvline(NDPR_PRICE_CENTRAL, color='#0072B2', ls='-', lw=2.5, label=f'Current spot (${NDPR_PRICE_CENTRAL:.0f}/kg)')
ax1.axvline(NDPR_PRICE_HIGH, color='gray', ls='-.', lw=2, alpha=0.85, label=f'Q1 2026 peak (${NDPR_PRICE_HIGH:.0f}/kg)')
ax1.axhline(0, color='black', lw=0.8)
ax1.fill_between(prices, 0, 1e10, where=(prices >= NDPR_PRICE_LOW), alpha=0.05, color='#009E73')
ax1.set_xlabel('NdPr oxide price ($/kg)', fontsize=11)
ax1.set_ylabel('Undiscounted NPV ($M)', fontsize=11)
ax1.tick_params(labelsize=9)
ax1.set_title('A.  Undiscounted NPV vs. NdPr oxide price\n(current cost assumptions; EXPLORATION ESTIMATE ONLY)', fontsize=9)
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(30, 200)
ax1.set_ylim(bottom=0, top=results_df['rev_high_$M'].max() * 1.1)

# Panel B: Cost breakdown stacked bar — Wong palette
ax2 = fig.add_subplot(gs[1, 0])
cost_components = ['cost_handling_$M', 'cost_concentr_$M', 'cost_logistics_$M', 'cost_processing_$M']
cost_labels = ['Tailings handling', 'Concentration', 'Logistics to White Mesa', 'White Mesa processing']
cost_colors = ['#0072B2', '#56B4E9', '#F0E442', '#D55E00']
bottoms = np.zeros(len(results_df))
x = np.arange(len(results_df))
for comp, label, col in zip(cost_components, cost_labels, cost_colors):
    vals = results_df[comp].values
    ax2.bar(x, vals, bottom=bottoms, color=col, label=label, edgecolor='white', lw=0.5)
    bottoms += vals
# Overlay revenue bars (hollow)
for idx, row in results_df.iterrows():
    ax2.bar(idx, row['rev_central_$M'], bottom=0, fill=False,
            edgecolor='black', lw=2.5, linestyle='-', label='Revenue (central)' if idx==0 else '')
ax2.set_xticks(x)
ax2.set_xticklabels([r['site'].split()[0] + '\n' + r['site'].split()[1]
                     for _, r in results_df.iterrows()], fontsize=8)
ax2.set_ylabel('$M (undiscounted)', fontsize=11)
ax2.tick_params(labelsize=9)
ax2.set_title('B.  Cost breakdown vs. revenue\n(dashed = central revenue at $109/kg)', fontsize=9)
ax2.legend(fontsize=7, loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')

# Panel C: Tornado sensitivity for Oroville (top site)
ax3 = fig.add_subplot(gs[1, 1])
oroville = results_df.iloc[0]
base_npv = oroville['npv_central_$M']
ndpr_oxide_t = oroville['ndpr_oxide_t']
total_cost_base = oroville['total_cost_$M']

sensitivities = {
    'NdPr price\n(±30%)':            [ndpr_oxide_t*1000*109*0.7/1e6 - total_cost_base,
                                       ndpr_oxide_t*1000*109*1.3/1e6 - total_cost_base],
    'Grade\n(±50%)':                  [ndpr_oxide_t*0.5*1000*109/1e6 - total_cost_base,
                                       ndpr_oxide_t*1.5*1000*109/1e6 - total_cost_base],
    'Tonnage\n(±30%)':                [ndpr_oxide_t*1000*109/1e6 - total_cost_base*1.3,
                                       ndpr_oxide_t*1000*109/1e6 - total_cost_base*0.7],
    'Processing cost\n(±25%)':        [ndpr_oxide_t*1000*109/1e6 - total_cost_base*0.75 + oroville['cost_processing_$M']*0.25,
                                       ndpr_oxide_t*1000*109/1e6 - total_cost_base*1.25 - oroville['cost_processing_$M']*0.25],
    'NdPr recovery\n(0.70–0.90)':     [ndpr_oxide_t*(0.70/0.80)*1000*109/1e6 - total_cost_base,
                                       ndpr_oxide_t*(0.90/0.80)*1000*109/1e6 - total_cost_base],
}
labels = list(sensitivities.keys())
lows   = [v[0] for v in sensitivities.values()]
highs  = [v[1] for v in sensitivities.values()]
widths = [h-l for l, h in zip(lows, highs)]
order  = sorted(range(len(widths)), key=lambda i: widths[i], reverse=True)

y_pos = np.arange(len(labels))
for i, idx in enumerate(order):
    lo, hi = lows[idx] - base_npv, highs[idx] - base_npv
    color_l = '#D55E00' if lo < 0 else '#009E73'
    color_h = '#009E73' if hi > 0 else '#D55E00'
    ax3.barh(i, abs(lo), left=base_npv + min(lo, 0), color=color_l, alpha=0.7)
    ax3.barh(i, abs(hi - lo), left=base_npv + lo, color=color_h, alpha=0.7)

ax3.set_yticks(range(len(order)))
ax3.set_yticklabels([labels[i] for i in order], fontsize=8)
ax3.axvline(base_npv, color='black', lw=1.5, label=f'Base NPV = ${base_npv:.0f}M')
ax3.set_xlabel('Undiscounted NPV ($M)', fontsize=11)
ax3.tick_params(labelsize=9)
ax3.set_title('C.  Tornado: NPV sensitivity\nOroville Placer (largest tailings volume;\nnot in economic base case — high-volume scenario)\n(see Fig 4 for Oroville volume context)', fontsize=9)
# Add color legend for tornado bars
from matplotlib.patches import Patch
tornado_legend = [
    Patch(facecolor='#009E73', alpha=0.7, label='Upside (positive impact)'),
    Patch(facecolor='#D55E00', alpha=0.7, label='Downside (negative impact)'),
    plt.Line2D([0],[0], color='black', lw=1.5, label=f'Base NPV = ${base_npv:.0f}M'),
]
ax3.legend(handles=tornado_legend, fontsize=7.5, loc='lower right')
ax3.grid(True, alpha=0.3, axis='x')

fig.text(0.5, 0.01, 'NE Washington REE Tailings Assessment — Au+REE Pipeline Project — EXPLORATION TARGET ONLY',
         ha='center', fontsize=7, color='gray', style='italic')
plt.savefig('outputs/figures/fig5_breakeven_sensitivity.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 5 saved")
print(f"\nCurrent NdPr price context:")
print(f"  SMM benchmark (June 2026): ~$109/kg NdPr oxide")
print(f"  DOD/MP Materials floor:     $110/kg NdPr oxide")
print(f"  Q1 2026 peak:               ~$138/kg")
print(f"  2024 trough:                ~$53/kg")
print(f"\nBreak-even prices for top 3 sites:")
for _, row in results_df.iterrows():
    economic = "VIABLE" if row['breakeven_$/kg'] < NDPR_PRICE_CENTRAL else "MARGINAL"
    print(f"  {row['site']}: ${row['breakeven_$/kg']:.0f}/kg → {economic} at current prices")
