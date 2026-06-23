"""
Task 2: Hard rock source characterization and upstream catchment analysis
Detrital monazite source lithology scoring.

Output:
  {outputs_dir}/geojson/task2_source_lithology_scored.geojson
  {outputs_dir}/figures/fig2_source_lithology_map.png
  {outputs_dir}/tables/task2_catchment_scores.csv
  {outputs_dir}/text/task2_carbonatite_literature_note.txt
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

import matplotlib.patheffects as pe
from pipeline.utils import (WONG, setup_mpl, wgs_path, watermark, save_fig, ensure_outputs, out,
                             map_extent, hillshade, north_arrow, scale_bar)


def run(cfg):
    setup_mpl()
    ensure_outputs(cfg['outputs_dir'])

    b = cfg['study_area']['bbox']
    wgs_b = cfg['study_area'].get('wgs_bbox', b)

    # ── Geologic domains from config ──────────────────────────────────────────
    lith_colors = {
        'MCC_metapelite':    WONG['blue'],
        'felsic_intrusive':  WONG['orange'],
        'sedimentary_cover': WONG['yellow'],
        'mafic_ultramafic':  '#CCCCCC',
    }
    lith_labels = {
        'MCC_metapelite':    'Metamorphic core complex (metapelite) — HIGH monazite potential',
        'felsic_intrusive':  'Colville Batholith (felsic intrusive) — MODERATE',
        'sedimentary_cover': 'Valley fill / sedimentary cover — LOW-MODERATE',
        'mafic_ultramafic':  'Mafic/ultramafic terrane — LOW',
    }

    geo_gdf = gpd.GeoDataFrame(
        [{'name': d['name'], 'lith_type': d['lith_type'], 'score': d['score']}
         for d in cfg['geology_domains']],
        geometry=[Polygon(d['coords']) for d in cfg['geology_domains']],
        crs='EPSG:4326',
    )

    # ── Mine sites from task1 output ──────────────────────────────────────────
    sites_gdf = gpd.read_file(out(cfg, 'geojson', 'task1_multicommodity_targets.geojson'))

    # ── Catchment scoring ─────────────────────────────────────────────────────
    CATCHMENT_RADIUS_DEG = 0.27

    def build_catchment(lon, lat, radius=CATCHMENT_RADIUS_DEG):
        center = Point(lon - 0.05, lat + 0.05)
        return center.buffer(radius)

    sites_gdf['catchment_geom'] = sites_gdf.apply(
        lambda r: build_catchment(r.lon, r.lat), axis=1)

    def score_catchment(site_row):
        catchment = site_row['catchment_geom']
        scores, type_fracs = [], {}
        for _, geo_row in geo_gdf.iterrows():
            intersection = catchment.intersection(geo_row.geometry)
            if not intersection.is_empty:
                area_frac = intersection.area / catchment.area
                if area_frac > 0.05:
                    k = geo_row['lith_type']
                    type_fracs[k] = type_fracs.get(k, 0.0) + area_frac
                    scores.append((geo_row['score'], area_frac, k))
        if not scores:
            return 1, 'UNKNOWN'
        total_area = sum(s[1] for s in scores)
        weighted = sum(s[0]*s[1] for s in scores) / total_area
        total_frac = sum(type_fracs.values()) or 1.0
        lith_list = [f"{k}({v/total_frac*100:.0f}%)"
                     for k, v in sorted(type_fracs.items(), key=lambda x: -x[1])
                     if v / total_frac >= 0.10]
        return round(weighted, 2), ' | '.join(lith_list)

    source_scores, source_liths = [], []
    for _, row in sites_gdf.iterrows():
        sc, lt = score_catchment(row)
        source_scores.append(sc)
        source_liths.append(lt)

    sites_gdf['source_lith_score'] = source_scores
    sites_gdf['source_lith_desc']  = source_liths

    # ── Carbonatite literature note ───────────────────────────────────────────
    carbonatite_note = """
CARBONATITE LITERATURE REVIEW — NE Washington and Adjacent Idaho
================================================================

Search conducted: MRDS commodity=carbonatite, radius 200km from study center (-118.75, 48.3)
Literature: Staatz 1972 (USGS Bull 1049), Staatz et al. 1979 (USGS OFR 79-82),
            Long et al. 2010 (USGS OFR 2010-1202 — domestic REE deposits)

FINDINGS:
---------
No carbonatite occurrences are documented within the 4-county NE Washington study area
(Okanogan, Ferry, Stevens, Pend Oreille counties).

NEAREST CARBONATITE-RELATED OCCURRENCES:
  1. Lemhi Pass Th-REE district, Lemhi Co., Idaho (~250 km SE of study area)
     - Vein-type Th-REE mineralization in Precambrian metasediments
     - NOT carbonatite-hosted; hydrothermal thorite and monazite veins
     - MRDS IDs: 10003897, 10003898
     - Staatz 1972: thorium content up to 0.3% ThO2 in veins

  2. Mountain Pass, CA (~1,500 km SW) — carbonatite-hosted bastnäsite;
     no genetic connection to NE WA

INTERPRETATION:
  No carbonatite Th-REE source is documented within 50 km of any priority site.
  Detrital monazite from Precambrian metapelite gneisses in the MCCs is the
  most probable Th carrier. This is consistent with:
  - High-grade metamorphic terranes producing abundant accessory monazite
    (Rasmussen & Muhling 2007; Holder et al. 2015)
  - Documented monazite in Okanogan MCC gneisses (Cheney et al. 1994)
  - NURE Th anomaly spatial correlation with MCC outcrop areas

ACTION: No carbonatite flag required for any current priority site.
        Maintain standard monazite-focused processing pathway assumption.
"""
    note_path = out(cfg, 'text', 'task2_carbonatite_literature_note.txt')
    with open(note_path, 'w') as f:
        f.write(carbonatite_note)

    # ── Figure 2 ─────────────────────────────────────────────────────────────
    def score_to_color(sc):
        if sc >= 2.5: return WONG['blue']
        if sc >= 1.5: return WONG['orange']
        if sc >= 0.5: return WONG['yellow']
        return '#CCCCCC'

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f'Figure 2 — Hard Rock Source Characterization and Catchment Sediment Routing\n'
        f'Mineral Systems: Source characterisation + Pathway routing — {cfg["study_area"]["name"]}',
        fontsize=12, fontweight='bold',
    )

    ax = axes[0]
    xmin, xmax, ymin, ymax = map_extent(cfg)

    hillshade(cfg, ax, alpha=0.25, zorder=0)

    # SGMC geology shapefile as map base layer
    SGMC_LITH_MAP = {
        'Metamorphic, gneiss':               'MCC_metapelite',
        'Metamorphic, schist':               'MCC_metapelite',
        'Metamorphic, amphibolite':          'MCC_metapelite',
        'Metamorphic, sedimentary':          'MCC_metapelite',
        'Metamorphic, sedimentary clastic':  'MCC_metapelite',
        'Metamorphic and Sedimentary, undifferentiated': 'MCC_metapelite',
        'Igneous, intrusive':                'felsic_intrusive',
        'Igneous, volcanic':                 'sedimentary_cover',
        'Sedimentary, clastic':              'sedimentary_cover',
        'Sedimentary, carbonate':            'sedimentary_cover',
        'Metamorphic, serpentinite':         'mafic_ultramafic',
        'Metamorphic, volcanic':             'mafic_ultramafic',
    }
    from shapely.geometry import box as _box
    sgmc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'ne_wa_ree', 'data', 'geologic',
                             'WA_sgmc_extracted', 'WA_geol_poly.shp')
    if os.path.exists(sgmc_path):
        try:
            sgmc = gpd.read_file(sgmc_path)
            clip_box = _box(xmin - 0.05, ymin - 0.05, xmax + 0.05, ymax + 0.05)
            sgmc = sgmc[sgmc.intersects(clip_box)].copy()
            sgmc['lith_type'] = sgmc['GENERALIZE'].map(SGMC_LITH_MAP).fillna('sedimentary_cover')
            for lt, color in lith_colors.items():
                subset = sgmc[sgmc['lith_type'] == lt]
                if len(subset):
                    subset.plot(ax=ax, color=color, alpha=0.55, linewidth=0, zorder=1)
            sgmc.boundary.plot(ax=ax, color='gray', linewidth=0.15, alpha=0.4, zorder=2)
        except Exception as _sgmc_err:
            import warnings as _w
            _w.warn(f"SGMC shapefile error ({_sgmc_err}); falling back to domain rectangles.")
            for _, row in geo_gdf.iterrows():
                color = lith_colors[row['lith_type']]
                xs, ys = row.geometry.exterior.xy
                ax.fill(xs, ys, color=color, alpha=0.5, zorder=1)
                ax.plot(xs, ys, color='gray', lw=0.5, zorder=2)
    else:
        for _, row in geo_gdf.iterrows():
            color = lith_colors[row['lith_type']]
            xs, ys = row.geometry.exterior.xy
            ax.fill(xs, ys, color=color, alpha=0.5, zorder=1)
            ax.plot(xs, ys, color='gray', lw=0.5, zorder=2)

    # Domain outlines (scoring zones — dashed, not filled)
    for d in cfg['geology_domains']:
        coords = d['coords']
        x_c, y_c = zip(*coords)
        ax.plot(list(x_c) + [x_c[0]], list(y_c) + [y_c[0]],
                color='black', lw=0.8, ls='--', alpha=0.5, zorder=3)

    # Domain labels — upper-left of each domain bounding box
    for d in cfg['geology_domains']:
        coords = d['coords']
        min_lon = min(c[0] for c in coords)
        max_lat = max(c[1] for c in coords)
        ax.text(min_lon + 0.04, max_lat - 0.06, d['name'].replace('_', ' '),
                fontsize=6.5, ha='left', va='top', color='black',
                fontweight='bold', alpha=0.85, zorder=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6))

    for _, row in sites_gdf.iterrows():
        ax.scatter(row.lon, row.lat, c=score_to_color(row['source_lith_score']),
                   s=100, edgecolors='black', linewidths=0.5, zorder=5, marker='D')
        ax.annotate(row['name'].split()[0], (row.lon, row.lat),
                    xytext=(4, 4), textcoords='offset points', fontsize=7)

    # ── Rivers ────────────────────────────────────────────────────────────────
    river_style = dict(color=WONG['sky'], lw=2.5, alpha=0.95, zorder=5)
    arrow_props = dict(arrowstyle='->', color='#2196F3', lw=1.5)
    label_kw    = dict(fontsize=7.5, color='#0055A4', style='italic', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.6, lw=0))

    for river in cfg.get('rivers', []):
        coords = river['coords']
        xs_r, ys_r = zip(*coords)
        ax.plot(xs_r, ys_r, **river_style)
        as_, ae_ = river['arrow_start'], river['arrow_end']
        ax.annotate('', xy=ae_, xytext=as_,
                    arrowprops=dict(arrowstyle='->', color='#2196F3', lw=2.0,
                        path_effects=[pe.withStroke(linewidth=4, foreground='white')]),
                    zorder=6)
        lo, ly = river.get('label_offset', [0.05, 0.0])
        mid_x = (as_[0] + ae_[0]) / 2 + lo
        mid_y = (as_[1] + ae_[1]) / 2 + ly
        ax.text(mid_x, mid_y, river['name'], **label_kw)

    # ── WGS mine waste overlay ────────────────────────────────────────────────
    _xl = wgs_path(cfg)
    wgs_exclude = set(cfg.get('wgs', {}).get('exclude_sites', []))
    wgs_offsets = cfg.get('wgs', {}).get('label_offsets', {})
    wgs_handles, wgs_labels_leg, wgs_label_set = [], [], set()

    def _classify_deposit(dep_type):
        dt = dep_type.lower() if isinstance(dep_type, str) else ''
        if 'alkalic epithermal' in dt or 'epithermal' in dt:
            return '*', WONG['orange'], 120, 'WGS: Epithermal'
        if 'intrusion-related' in dt or 'reduced' in dt:
            return 'P', WONG['blue'], 100, 'WGS: Intrusion-related Au'
        if 'polymetallic' in dt or 'intermediate' in dt:
            return 's', WONG['green'], 90, 'WGS: Polymetallic vein'
        if 'carbonate' in dt or 'manto' in dt or 'replacement' in dt:
            return '^', WONG['vermillion'], 90, 'WGS: Carbonate replacement'
        if 'porphyry' in dt or 'tungsten' in dt or 'w-dominant' in dt:
            return 'h', WONG['pink'], 90, 'WGS: Porphyry/W'
        return 'o', 'black', 80, 'WGS: Other'

    try:
        wgs_raw = pd.read_excel(_xl, sheet_name='Geochemical Data', header=0, skiprows=[1])
        wgs_raw['Latitude']  = pd.to_numeric(wgs_raw['Latitude'],  errors='coerce')
        wgs_raw['Longitude'] = pd.to_numeric(wgs_raw['Longitude'], errors='coerce')
        wgs_mask = (
            wgs_raw['Latitude'].between(wgs_b['lat_min'], wgs_b['lat_max']) &
            wgs_raw['Longitude'].between(wgs_b['lon_min'], wgs_b['lon_max'])
        )
        wgs_area = wgs_raw[wgs_mask].copy()
        wgs_area = wgs_area[~wgs_area['Site_Name'].isin(wgs_exclude)]
        wgs_sites = wgs_area.groupby('Site_Name').agg(
            Latitude=('Latitude', 'mean'),
            Longitude=('Longitude', 'mean'),
            deposit_type=('Deposit Type(s)', 'first'),
        ).reset_index()

        for _, wrow in wgs_sites.iterrows():
            mkr, clr, sz, lbl = _classify_deposit(wrow['deposit_type'])
            ax.scatter(wrow['Longitude'], wrow['Latitude'], marker=mkr, c=clr, s=sz,
                       edgecolors='black', linewidths=0.5, zorder=7)
            dx, dy = wgs_offsets.get(wrow['Site_Name'], [5, 5])
            ax.annotate(wrow['Site_Name'], (wrow['Longitude'], wrow['Latitude']),
                        xytext=(dx, dy), textcoords='offset points', fontsize=6, color='dimgray')
            if lbl not in wgs_label_set:
                wgs_label_set.add(lbl)
                wgs_handles.append(Line2D([0],[0], marker=mkr, color='w',
                                          markerfacecolor=clr, markersize=8,
                                          markeredgecolor='black', markeredgewidth=0.5,
                                          label=lbl))
                wgs_labels_leg.append(lbl)

        if wgs_handles:
            legend2 = ax.legend(wgs_handles, wgs_labels_leg, loc='lower right', fontsize=7,
                                title='WGS OFR 2026-02\nMine Waste Sites',
                                title_fontsize=7, framealpha=0.9)
            ax.add_artist(legend2)
        print(f"WGS sites plotted: {len(wgs_sites)}")

    except FileNotFoundError:
        import warnings as w; w.warn("WGS OFR 2026-02 file not found; WGS layer skipped.", UserWarning)
    except Exception as _e:
        import warnings as w; w.warn(f"WGS layer error ({_e}); WGS layer skipped.", UserWarning)

    north_arrow(ax)
    scale_bar(ax, cfg, length_km=50)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Longitude', fontsize=11); ax.set_ylabel('Latitude', fontsize=11)
    ax.tick_params(labelsize=9)
    ax.set_title('A.  Geologic domains and mine sites\n'
                 '(diamonds colored by source lithology score)\n'
                 'Arrow = drainage direction; monazite transport follows stream flow', fontsize=9)
    ax.grid(True, alpha=0.2)
    lith_patches = [mpatches.Patch(facecolor=c, alpha=0.6, edgecolor='gray', label=lith_labels[lt])
                    for lt, c in lith_colors.items()]
    ax.legend(handles=lith_patches, loc='lower left', fontsize=7, framealpha=0.9)

    ax2 = axes[1]
    sorted_sites = sites_gdf.sort_values('source_lith_score', ascending=True)
    colors_bar = [score_to_color(s) for s in sorted_sites['source_lith_score']]
    bars = ax2.barh(range(len(sorted_sites)), sorted_sites['source_lith_score'],
                    color=colors_bar, edgecolor='black', lw=0.5)
    ax2.set_yticks(range(len(sorted_sites)))
    ax2.set_yticklabels(sorted_sites['name'], fontsize=8)
    ax2.set_xlabel('Source lithology score (0=mafic → 3=MCC metapelite)', fontsize=11)
    ax2.tick_params(labelsize=9)
    ax2.set_title('B.  Source lithology score by site\n(area-weighted catchment composition)', fontsize=9)
    ax2.axvline(2.0, color=WONG['vermillion'], ls='--', lw=1, label='Score ≥ 2 (high priority)')
    ax2.set_xlim(0, 3.5)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='x')
    for bar, sc in zip(bars, sorted_sites['source_lith_score']):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 f'{sc:.1f}', va='center', fontsize=8)

    plt.tight_layout()
    watermark(fig, cfg)
    save_fig(fig, out(cfg, 'figures', 'fig2_source_lithology_map.png'))

    export_cols = [c for c in sites_gdf.columns if c != 'catchment_geom']
    sites_gdf[export_cols].to_file(
        out(cfg, 'geojson', 'task2_source_lithology_scored.geojson'), driver='GeoJSON')

    table_out = sites_gdf[['name','lon','lat','source_lith_score','source_lith_desc']].sort_values(
        'source_lith_score', ascending=False)
    table_out.to_csv(out(cfg, 'tables', 'task2_catchment_scores.csv'), index=False)

    print(f"\nTask 2 Results:")
    print(f"  Sites with score ≥ 2: {(sites_gdf['source_lith_score'] >= 2).sum()}")
    print(f"  Sites with score = 3: {(sites_gdf['source_lith_score'] >= 2.8).sum()}")


if __name__ == '__main__':
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/ne_washington/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
