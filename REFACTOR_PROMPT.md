# Refactor Prompt — Au+REE Placer Assessment Pipeline

> Paste this into a new Claude Code session (agent mode) with the repo open.

---

## Project context

This is a geoscience exploration pipeline that ranks placer Au + REE (monazite) sites
using public geochemical, geophysical, and mine waste datasets. It was built for
**NE Washington State** and has been committed to git (commit `e93b8b8`). The goal is to
refactor it into a clean, config-driven tool that can be applied to **any US placer REE
district** — starting with three study areas: NE Washington, Idaho Batholith, and
Montana placer belt.

The repo is at: `/Users/nicoleaikin/projects/Au + REE pipeline`

---

## What was completed in the prior session

**Pipeline scripts (all working, all produce output figures):**
- `prep_nure.py` — cleans raw NGDB sediment export to `data/nure/`
- `ne_wa_ree/task1_coplacer_minerals.py` → `fig1` — aeromagnetic × Th anomaly co-occurrence
- `ne_wa_ree/task2_source_lithology.py` → `fig2` — source lithology map + WGS mine waste sites
- `ne_wa_ree/task3_geochemical_discrimination.py` → `fig3` — Th source discrimination (U/Th, Ce/La, P)
- `ne_wa_ree/task4_volume_estimation.py` → `fig4` — lidar volume + WGS tonnage table
- `ne_wa_ree/task5_energy_fuels_pathway.py` → `fig5` — NPV/break-even/sensitivity analysis
- `ne_wa_ree/task6_figure6.py` → `fig6` — decision framework
- `ne_wa_ree/task7_au_as_anomaly.py` → `fig8` — Au/As pathfinder map
- `ne_wa_ree/task8_mine_waste_ree.py` → `fig9` — WGS mine waste REE + ABA + critical minerals
- `ne_wa_ree/integration_task.py` → `fig7` — integrated multi-criterion priority map + radar

**Data in repo:**
- `data/nure/nure_ne_wa_sediment.csv` — real NURE stream sediment (NE WA subset, ~400 samples)
- `data/nure/nure_wa_sediment.csv` — full WA state NURE dataset
- `ne_wa_ree/data/mrds/mrds_ne_wa.geojson` — MRDS placer/REE mine sites
- `ne_wa_ree/data/geologic/wa_geology.geojson` — WA state geologic map (20 MB)
- `ne_wa_ree/outputs/` — all 9 figures (PNG, 300 dpi) + CSVs + GeoJSONs

**Large files excluded from git** (see `ne_wa_ree/DATA_SOURCES.md`):
- Lidar TIFs per site (~7 GB total, from lidar.wa.gov)
- 30m DEM (~80 MB, from USGS National Map)
- Aeromagnetic TIF (1.1 MB, from USGS)
- WGS OFR 2026-02 Excel supplement (from WA DNR, needed for task8/fig9)

**Technical debt / known issues:**
- Every script has the NE WA bounding box hardcoded: `lat 47.5–49.1, lon -120 to -117.5`
- The 12 placer site names/coordinates are embedded as Python dicts/lists in task1, task4, task5, integration_task — they come from MRDS but aren't queried dynamically
- Geological domain polygons (task2) are hand-drawn approximate polygons for NE WA MCCs — not loaded from data
- River coordinate arrays (task2) are hardcoded NE WA polylines
- Economics (task5) hardcode Energy Fuels White Mesa Mill (Blanding, UT) as the only processor
- Wong 8-color palette and `MPLCONFIGDIR` setup are copy-pasted into every script
- `task6_probabilistic_framing.py` is an unused stub (can be deleted or merged)
- All 9 scripts run `os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig'` redundantly

---

## Outstanding tasks (in priority order)

### Priority 1 — Config-driven architecture (REQUIRED for multi-site use)

Refactor the pipeline so all location-specific parameters live in a single config file
per study area. Nothing geographic, geologic, or economic should be hardcoded in task scripts.

Target directory structure:
```
Au + REE pipeline/
├── pipeline/                  # renamed from ne_wa_ree/ — shared task scripts
│   ├── utils.py               # NEW: shared utilities
│   ├── task1_coplacer.py      # renamed, reads from cfg
│   ├── task2_lithology.py
│   ├── task3_geochemistry.py
│   ├── task4_volume.py
│   ├── task5_economics.py
│   ├── task6_framework.py
│   ├── task7_pathfinder.py
│   ├── task8_mine_waste.py
│   ├── integration.py
│   └── run_pipeline.py        # NEW: CLI entry point
├── configs/
│   ├── ne_washington/
│   │   ├── config.yaml        # all NE WA parameters
│   │   └── geology_domains.geojson  # MCC/batholith polygons as real data
│   ├── idaho_batholith/
│   │   └── config.yaml        # stub with TODO markers
│   └── montana_placer/
│       └── config.yaml        # stub with TODO markers
├── data/                      # global (NURE WA state, shared)
├── prep_nure.py
├── requirements.txt
├── README.md
└── .gitignore
```

### Priority 2 — `utils.py` shared module

Extract duplicated code from all 9 scripts into a single `pipeline/utils.py`:

```python
# Must contain at minimum:
WONG = {...}                    # palette, currently copy-pasted 9x
setup_mpl()                     # sets MPLCONFIGDIR, rcParams
load_nure(config)               # reads CSV, applies half-MDL, converts P to ppm
anomaly_threshold(series)       # mean+2SD on log-transformed positives
chondrite_normalize(df, cols)   # divides by Sun & McDonough 1989 values
load_mrds(config)               # reads GeoJSON or queries MRDS API by bbox+commodity
watermark(fig)                  # adds "EXPLORATION TARGET ONLY" text
save_fig(fig, path, dpi=300)    # saves with bbox_inches='tight'
```

### Priority 3 — `config.yaml` schema

Define a YAML schema that captures everything location-specific. Every task script
should accept a config dict and contain zero hardcoded study-area values.

Minimum required fields:
```yaml
study_area:
  name: "NE Washington"
  short: "ne_wa"
  bbox: {lon_min: -120.0, lon_max: -117.5, lat_min: 47.5, lat_max: 49.1}
  crs: "EPSG:4326"

data:
  nure_csv: "data/nure/nure_ne_wa_sediment.csv"
  mrds_geojson: "pipeline/data/mrds/mrds_ne_wa.geojson"
  geology_domains: "configs/ne_washington/geology_domains.geojson"
  aeromagnetic_tif: "pipeline/data/aeromagnetic/wa_mag_anomaly.tif"
  dem_tif: "pipeline/data/dem/ne_wa_dem_30m.tif"
  wgs_excel: null  # set or use WGS_OFR2026_PATH env var

mrds_commodities: ["placer gold", "gold", "REE", "monazite", "magnetite"]

geochemistry:
  th_classification:
    monazite_ce_min_ppm: 50         # Mücke & Rao 1996
    monazite_la_min_ppm: 20
    monazite_p_min_ppm: 400
    monazite_uth_max: 0.5
    thorite_uth_min: 1.5

economics:
  processor_name: "Energy Fuels White Mesa Mill"
  processor_location: "Blanding, UT"
  distance_km: 1400
  trucking_cost_per_tonne_km: 0.12
  toll_milling_cost_per_tonne: 285
  ndpr_recovery: 0.80
  ndpr_price_central_usd_kg: 109
  ndpr_price_low_usd_kg: 60
  ndpr_price_high_usd_kg: 140

rivers:  # list of {name, coords: [[lon,lat],...]}
  - name: "Okanogan R."
    coords: [[-119.57, 49.1], [-119.57, 48.7], [-119.55, 48.4]]
  # ... etc

geology_domains:  # read from configs/<area>/geology_domains.geojson
  score_map:
    MCC_metapelite: 3
    felsic_intrusive: 2
    sedimentary_cover: 1
    mafic_ultramafic: 0
```

### Priority 4 — `run_pipeline.py` CLI entry point

```python
# Usage:
#   python run_pipeline.py --config configs/ne_washington/config.yaml
#   python run_pipeline.py --config configs/idaho_batholith/config.yaml --tasks 1,2,3
#   python run_pipeline.py --config configs/ne_washington/config.yaml --task 8

import argparse, yaml, importlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--tasks', default='all')  # e.g. "1,2,3" or "all"
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    task_map = {
        '1': 'pipeline.task1_coplacer',
        '2': 'pipeline.task2_lithology',
        ...
    }
    # run each task in order, passing cfg
```

### Priority 5 — Idaho Batholith and Montana placer belt config stubs

Create `configs/idaho_batholith/config.yaml` and `configs/montana_placer/config.yaml`
as stubs with correct bounding boxes, TODO markers for geology domains and economics,
and instructions for which datasets to download.

Idaho Batholith reference area:
- Bounding box: approximately lon -116 to -114, lat 44 to 46
- Key districts: Orogrande, Dixie, Warren, Florence placers
- Source lithology: Idaho Batholith (peraluminous two-mica granite), Belt Supergroup metasediments
- Nearest REE processor: MP Materials at Mountain Pass CA or Energy Fuels UT
- NURE data: should be available via same USGS NGDB query

Montana placer belt reference area:
- Bounding box: approximately lon -114 to -111, lat 45 to 47
- Key districts: Confederate Gulch, Alder Gulch, Montana Bar, Libby Creek
- Source lithology: Boulder Batholith, Tobacco Root metamorphics, Proterozoic Belt Supergroup
- Nearest REE processor: Energy Fuels UT or potential Canadian options

### Priority 6 — Code efficiency / deduplication

- Delete `task6_probabilistic_framing.py` (unused stub)
- Merge `task6_figure6.py` into the main task numbering (it currently produces fig6 but is called task6_figure6 — inconsistent)
- Consolidate the NE WA `ne_wa_ree/` directory into `pipeline/` so scripts are study-area-agnostic
- Remove the `{data` garbage entry that appears in `ne_wa_ree/` directory listing (orphan file)
- Each task script should be <400 lines; task8 is 763 lines and should be split if possible

---

## Constraints / do not change

- **Scientific methodology**: anomaly thresholding (mean+2SD log), U/Th monazite classification criteria, chondrite normalization (Sun & McDonough 1989), MEND protocol for ABA risk tiers, NPV undiscounted — all scientifically justified and should not be simplified away
- **Figure aesthetics**: Wong 8-color palette, 300 dpi, colorblind-safe, watermarked "EXPLORATION TARGET ONLY" — keep all of these
- **Exploration disclaimers**: all outputs must retain "EXPLORATION TARGET ONLY" watermarks and text — do not remove
- **Backward compatibility**: the NE WA pipeline must still produce all 9 figures identically after refactor (run it end-to-end to verify)
- **Python 3.9+** compatible (macOS system Python constraint)

---

## How to approach this

1. **Read first, plan second.** Before writing any code, read the full current `task1`–`task8`, `integration_task.py`, and `prep_nure.py` to understand exactly what's hardcoded. Read `ne_wa_ree/DATA_SOURCES.md` and `README.md`. Build a complete mental map of dependencies between tasks.

2. **Extract `utils.py` first** (safest change, no logic changes). This deduplicates WONG, MPLCONFIGDIR, chondrite values, anomaly threshold, and figure saving across all scripts.

3. **Create the `config.yaml` for NE WA** and convert ONE script (task3 is the cleanest) to config-driven as a proof of concept. Verify it still produces fig3 identically.

4. **Convert remaining scripts** one at a time, re-running and verifying output after each.

5. **Write Idaho Batholith and Montana config stubs** — these are YAML stubs with correct bbox and TODO markers for each required field that would need real data.

6. **Write `run_pipeline.py` CLI** last, once all tasks accept a cfg dict.

7. **Run the full NE WA pipeline end-to-end** via `run_pipeline.py --config configs/ne_washington/config.yaml` and verify all 9 figures regenerate.

8. **Commit** with a clear message. Push to GitHub.

---

## Success criteria

- [ ] `python run_pipeline.py --config configs/ne_washington/config.yaml` produces all 9 figures without errors
- [ ] No hardcoded study-area values (bounding box, site names, geology domains, economics) in any task script — all come from config
- [ ] `configs/idaho_batholith/config.yaml` and `configs/montana_placer/config.yaml` exist as valid stubs that tell a user exactly what data to acquire and what parameters to fill in
- [ ] `pipeline/utils.py` exists with at minimum: WONG, setup_mpl, load_nure, anomaly_threshold, chondrite_normalize, watermark, save_fig
- [ ] No script longer than 450 lines (task8 is currently 763 — acceptable if split or reorganized)
- [ ] `README.md` updated to document the config-driven usage and the three study area examples
- [ ] All hardcoded `/Users/nicoleaikin/` paths eliminated (4 were fixed to env var; verify none remain)
- [ ] Git committed and ready to push

---

## Key files to read at the start

Read these in order to understand the current codebase before making any changes:
1. `ne_wa_ree/task1_coplacer_minerals.py` (278 lines — simplest, good reference)
2. `ne_wa_ree/task3_geochemical_discrimination.py` (500 lines — cleanest logic)
3. `ne_wa_ree/integration_task.py` (442 lines — most complex dependencies)
4. `ne_wa_ree/task5_energy_fuels_pathway.py` (337 lines — all economic parameters)
5. `ne_wa_ree/task2_source_lithology.py` (409 lines — geology domains + rivers)
6. `README.md` and `ne_wa_ree/DATA_SOURCES.md`
