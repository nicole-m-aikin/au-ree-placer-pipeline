"""
TASK 6: Decision framework — question-driven layout
NE Washington REE tailings assessment

Three rows answering progressively specific questions:
  Row 1: Is there monazite here?
  Row 2: How much is there and where?
  Row 3: Is it worth pursuing?

Output: outputs/figures/fig6_decision_framework.png
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

WONG = {
    'black':      '#000000',
    'orange':     '#E69F00',
    'sky':        '#56B4E9',
    'green':      '#009E73',
    'yellow':     '#F0E442',
    'blue':       '#0072B2',
    'vermillion': '#D55E00',
    'pink':       '#CC79A7',
    'gray':       '#AAAAAA',
    'lightgray':  '#DDDDDD',
}

fig, ax = plt.subplots(figsize=(20, 13))
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.axis('off')
fig.patch.set_facecolor('white')

ax.set_title('NE Washington REE Tailings Assessment — Decision Framework',
             fontsize=15, fontweight='bold', pad=14)

# ── Helper: draw a rounded box ────────────────────────────────────────────────
def box(ax, x, y, w, h, text, facecolor, edgecolor='#333333', fontsize=9,
        text_color='white', alpha=1.0, bold=False, italic=False):
    style = 'bold' if bold else ('italic' if italic else 'normal')
    patch = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle='round,pad=0.08',
                           facecolor=facecolor, edgecolor=edgecolor,
                           linewidth=1.2, alpha=alpha, zorder=2)
    ax.add_patch(patch)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, zorder=3, fontstyle=style if italic else 'normal',
            fontweight='bold' if bold else 'normal',
            wrap=True, multialignment='center',
            transform=ax.transData)

def arr(ax, x1, y1, x2, y2, color='#555555', lw=1.4, style='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                linestyle=style),
                zorder=4)

# ── Column headers ────────────────────────────────────────────────────────────
COL_Q  = 3.5    # Question column x-center
COL_E  = 8.5    # Evidence column x-center
COL_G  = 13.5   # Gap column x-center
COL_S  = 18.2   # Status column x-center

for x_col, label, color in [(COL_Q, 'QUESTION', '#333333'),
                              (COL_E, 'EVIDENCE FOR', WONG['blue']),
                              (COL_G, 'GAP REMAINING', WONG['vermillion']),
                              (COL_S, 'STATUS', '#333333')]:
    ax.text(x_col, 12.3, label, ha='center', va='center', fontsize=11,
            fontweight='bold', color=color)

ax.axhline(12.0, xmin=0.01, xmax=0.99, color='#CCCCCC', lw=1.2)

# ── ROW POSITIONS ─────────────────────────────────────────────────────────────
rows = [
    (10.0,
     'ROW 1\nIs there\nmonazite here?',
     WONG['blue'],
     '• MCC metapelite catchments\n  12/12 sites drain Okanogan or\n  Kettle MCC terrain (Task 2)\n\n• Th anomalies in stream sediment\n  61 Th-anomalous samples in NE WA\n  MIXED_UNCLEAR + THORITE_UTHO (Task 3)\n\n• Aeromagnetic co-occurrence\n  2 sites: mag high + Th anomaly\n  Colville + Hunters Placer (Task 1)',
     'No MONAZITE geochemical fingerprint\nconfirmed — Ce/La data only 35–49%\nnon-null in NURE dataset\n\nCannot confirm Th-Ce-P triplet\nfrom stream sediment alone\n(→ need auger samples + MLA)',
     'PARTIAL\nevidence',
     WONG['orange']),
    (6.5,
     'ROW 2\nHow much is\nthere and where?',
     WONG['blue'],
     '• 12 priority placer sites ranked\n  Combined score 5.25–11.04\n  #1 Hunters, #2 Colville\n  #3 Conconully (Task integration)\n\n• Lidar volume estimation\n  7.0 Mt total tailings across sites\n  1,445 t NdPr exploration target (Task 4)\n\n• Grade proxy from NURE Th\n  Regional background fill applied\n  ±50% grade uncertainty',
     'No in-situ grade data\nStream sediment proxy — ±50% uncertainty\n\nLidar diff includes non-tailings\ndisturbance (floodplain reworking,\ninfrastructure, road grading)',
     'ESTIMATED\n±50%',
     WONG['orange']),
    (3.0,
     'ROW 3\nIs it worth\npursuing?',
     WONG['blue'],
     '• All 3 top sites viable at current prices\n  Break-even $74–101/kg vs.\n  current $109/kg NdPr (Task 5)\n\n• Domestic processing pathway:\n  Energy Fuels White Mesa Mill\n  licensed for Th-bearing monazite\n\n• Dual Au+REE signal at Hunters Placer\n  PORPHYRY_CU co-anomaly (Task 7 / Fig 8)',
     'No formal resource estimate\nNo field confirmation of grade\n\nConconully break-even ($101/kg)\nis marginal — viable only if\nprice holds above 2024 trough\n($60/kg)',
     'VIABLE at\ncurrent\nprices',
     WONG['green']),
]

EW = 4.8    # Evidence box width
GW = 4.5    # Gap box width
QW = 3.4    # Question box width
SW = 2.2    # Status box width
BH = 2.4    # Box height

for (yc, qtxt, qcol, etxt, gtxt, stxt, scol) in rows:
    box(ax, COL_Q, yc, QW, BH, qtxt, qcol, fontsize=9.5, bold=True)
    box(ax, COL_E, yc, EW, BH, etxt, WONG['blue'], fontsize=7.5,
        alpha=0.10, text_color='#111111', edgecolor=WONG['blue'])
    box(ax, COL_G, yc, GW, BH, gtxt, WONG['vermillion'],
        fontsize=7.5, alpha=0.10, text_color='#111111',
        edgecolor=WONG['vermillion'])
    box(ax, COL_S, yc, SW, BH * 0.65, stxt, scol, fontsize=9, bold=True)
    arr(ax, COL_Q + QW/2, yc, COL_E - EW/2, yc, color=WONG['blue'])
    arr(ax, COL_E + EW/2, yc, COL_G - GW/2, yc, color='#888888')
    arr(ax, COL_G + GW/2, yc, COL_S - SW/2, yc, color='#888888')

# Row dividers — placed between rows with some breathing room
for y_div in [4.8, 8.3]:
    ax.axhline(y_div, xmin=0.01, xmax=0.99, color='#DDDDDD', lw=0.8, ls='--')

# ── NEXT STEP box (bottom) ────────────────────────────────────────────────────
ns_text = ('NEXT STEP:  Auger sampling program — Colville Placer + Hunters Placer\n'
           'Cost: $136,000 – $204,000  (20–30 holes per site at $2,000–3,400/hole)\n'
           'Outcome: grade uncertainty ±50% → ±15–25%  |  '
           'Enables NI 43-101 Inferred Resource estimate  |  Confirms monazite vs. thorite host')
ns_patch = FancyBboxPatch((0.3, 0.3), 19.4, 1.3,
                          boxstyle='round,pad=0.12',
                          facecolor=WONG['blue'], edgecolor='#002244',
                          linewidth=2.0, zorder=2)
ax.add_patch(ns_patch)
ax.text(10.0, 0.95, ns_text, ha='center', va='center', fontsize=9.5,
        color='white', fontweight='bold', zorder=3, multialignment='center')

# Arrows from each row down toward next step
for yc in [3.0, 6.5, 10.0]:
    arr(ax, 10.0, yc - BH/2, 10.0, 1.65, color='#BBBBBB', lw=1.0, style='--')

# ── Watermark ─────────────────────────────────────────────────────────────────
fig.text(0.5, 0.005,
         'NE Washington REE Tailings Assessment — Au+REE Pipeline Project — EXPLORATION TARGET ONLY',
         ha='center', fontsize=7, color='gray', style='italic')

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
out_path = 'outputs/figures/fig6_decision_framework.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

import os
size_kb = os.path.getsize(out_path) / 1024
print(f"Saved : {os.path.abspath(out_path)}")
print(f"Size  : {size_kb:.1f} KB  ({size_kb/1024:.2f} MB)")
print(f"DPI   : 300")
print("Done.")
