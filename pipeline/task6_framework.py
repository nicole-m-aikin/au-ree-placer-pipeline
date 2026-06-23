"""
Task 6: Decision framework — question-driven layout.
Three rows answering progressively specific questions:
  Row 1: Is there monazite here?
  Row 2: How much is there and where?
  Row 3: Is it worth pursuing?

Output:
  {outputs_dir}/figures/fig6_decision_framework.png
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
from pipeline.utils import WONG, setup_mpl, watermark, save_fig, ensure_outputs, out


def _build_default_rows(cfg):
    """Build decision framework rows, reading NdPr total dynamically from task4 output."""
    ndpr_total_str = '1,445 t NdPr'
    try:
        t4_path = out(cfg, 'tables', 'task4_volume_tonnage_summary.csv')
        if os.path.exists(t4_path):
            t4 = pd.read_csv(t4_path)
            total_p50 = t4['ndpr_t_p50'].sum() if 'ndpr_t_p50' in t4.columns else t4['ndpr_tonnes'].sum()
            ndpr_total_str = f'{total_p50:,.0f} t NdPr'
    except Exception:
        pass

    return [
        (
            'ROW 1\nIs there\nmonazite here?',
            WONG['blue'],
            '• MCC metapelite catchments\n  12/12 sites drain Okanogan or\n  Kettle MCC terrain\n  (source lithology analysis)\n\n'
            '• Th anomalies in stream sediment\n  61 Th-anomalous samples in NE WA\n  MIXED_UNCLEAR + THORITE_UTHO\n  (geochemical discrimination)\n\n'
            '• Aeromagnetic co-occurrence\n  2 sites: mag high + Th anomaly\n  Colville + Hunters Placer\n  (co-placer indicator analysis)',
            'No MONAZITE geochemical fingerprint\nconfirmed — Ce/La data only 35–49%\nnon-null in NURE dataset\n\n'
            'Cannot confirm Th-Ce-P triplet\nfrom stream sediment alone\n(→ need auger samples + MLA)',
            'PARTIAL\nevidence',
            WONG['orange'],
        ),
        (
            'ROW 2\nHow much is\nthere and where?',
            WONG['blue'],
            f'• 12 priority placer sites ranked\n  Combined score 5.25–11.04\n  #1 Hunters, #2 Colville\n  #3 Conconully (integrated ranking)\n\n'
            f'• Lidar volume estimation\n  7.0 Mt total tailings across sites\n  {ndpr_total_str} exploration target\n  (volume estimation)\n\n'
            '• Grade proxy from NURE Th\n  Regional background fill applied\n  ±50% grade uncertainty',
            'No in-situ grade data\nStream sediment proxy — ±50% uncertainty\n\n'
            'Lidar diff includes non-tailings\ndisturbance (floodplain reworking,\ninfrastructure, road grading)',
            'ESTIMATED\n±50%',
            WONG['orange'],
        ),
        (
            'ROW 3\nIs it worth\npursuing?',
            WONG['blue'],
            '• All 3 top sites viable at current prices\n  Break-even $74–101/kg vs.\n  current $109/kg NdPr (break-even analysis)\n\n'
            '• Domestic processing pathway:\n  Energy Fuels White Mesa Mill\n  licensed for Th-bearing monazite\n\n'
            '• Dual Au+REE signal at Hunters Placer\n  PORPHYRY_CU co-anomaly\n  (Au/As pathfinder map)',
            'No formal resource estimate\nNo field confirmation of grade\n\n'
            'Conconully break-even ($101/kg)\nis marginal — viable only if\nprice holds above 2024 trough\n($60/kg)',
            'VIABLE at\ncurrent\nprices',
            WONG['green'],
        ),
    ]


def run(cfg):
    setup_mpl()
    ensure_outputs(cfg['outputs_dir'])

    rows = cfg.get('decision_framework_rows', _build_default_rows(cfg))
    next_step_text = cfg.get(
        'decision_framework_next_step',
        'NEXT STEP:  Auger sampling program — Colville Placer + Hunters Placer\n'
        'Cost: $136,000 – $204,000  (20–30 holes per site at $2,000–3,400/hole)\n'
        'Outcome: grade uncertainty ±50% → ±15–25%  |  '
        'Enables NI 43-101 Inferred Resource estimate  |  Confirms monazite vs. thorite host',
    )

    def box(ax, x, y, w, h, text, facecolor, edgecolor='#333333', fontsize=9,
            text_color='white', alpha=1.0, bold=False, italic=False):
        patch = FancyBboxPatch((x - w/2, y - h/2), w, h,
                               boxstyle='round,pad=0.08',
                               facecolor=facecolor, edgecolor=edgecolor,
                               linewidth=1.2, alpha=alpha, zorder=2)
        ax.add_patch(patch)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                color=text_color, zorder=3, multialignment='center',
                fontstyle='italic' if italic else 'normal',
                fontweight='bold' if bold else 'normal',
                transform=ax.transData)

    def arr(ax, x1, y1, x2, y2, color='#555555', lw=1.4, style='-'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, linestyle=style),
                    zorder=4)

    COL_Q, COL_E, COL_G, COL_S = 3.5, 8.5, 13.5, 18.2
    EW, GW, QW, SW, BH = 4.8, 4.5, 3.4, 2.2, 2.4

    fig, ax = plt.subplots(figsize=(20, 13))
    ax.set_xlim(0, 20); ax.set_ylim(0, 13)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_title(f'{cfg["study_area"]["name"]} REE Tailings Assessment — Decision Framework',
                 fontsize=15, fontweight='bold', pad=14)

    for x_col, label, color in [(COL_Q, 'QUESTION', '#333333'),
                                  (COL_E, 'EVIDENCE FOR', WONG['blue']),
                                  (COL_G, 'GAP REMAINING', WONG['vermillion']),
                                  (COL_S, 'STATUS', '#333333')]:
        ax.text(x_col, 12.3, label, ha='center', va='center', fontsize=11,
                fontweight='bold', color=color)
    ax.axhline(12.0, xmin=0.01, xmax=0.99, color='#CCCCCC', lw=1.2)

    row_y_positions = [10.0, 6.5, 3.0]
    for (yc, (qtxt, qcol, etxt, gtxt, stxt, scol)) in zip(row_y_positions, rows):
        box(ax, COL_Q, yc, QW, BH, qtxt, qcol, fontsize=9.5, bold=True)
        box(ax, COL_E, yc, EW, BH, etxt, WONG['blue'], fontsize=7.5,
            alpha=0.10, text_color='#111111', edgecolor=WONG['blue'])
        box(ax, COL_G, yc, GW, BH, gtxt, WONG['vermillion'],
            fontsize=7.5, alpha=0.10, text_color='#111111', edgecolor=WONG['vermillion'])
        box(ax, COL_S, yc, SW, BH * 0.65, stxt, scol, fontsize=9, bold=True)
        arr(ax, COL_Q + QW/2, yc, COL_E - EW/2, yc, color=WONG['blue'])
        arr(ax, COL_E + EW/2, yc, COL_G - GW/2, yc, color='#888888')
        arr(ax, COL_G + GW/2, yc, COL_S - SW/2, yc, color='#888888')

    for y_div in [4.8, 8.3]:
        ax.axhline(y_div, xmin=0.01, xmax=0.99, color='#DDDDDD', lw=0.8, ls='--')

    ns_patch = FancyBboxPatch((0.3, 0.3), 19.4, 1.3,
                              boxstyle='round,pad=0.12',
                              facecolor=WONG['blue'], edgecolor='#002244',
                              linewidth=2.0, zorder=2)
    ax.add_patch(ns_patch)
    ax.text(10.0, 0.95, next_step_text, ha='center', va='center', fontsize=9.5,
            color='white', fontweight='bold', zorder=3, multialignment='center')

    for yc in row_y_positions:
        arr(ax, 10.0, yc - BH/2, 10.0, 1.65, color='#BBBBBB', lw=1.0, style='--')

    watermark(fig, cfg)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    save_fig(fig, out(cfg, 'figures', 'fig6_decision_framework.png'))


if __name__ == '__main__':
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/ne_washington/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
