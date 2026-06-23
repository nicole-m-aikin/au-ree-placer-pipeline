# Au + REE Pipeline — NE Washington Placer Assessment

A multi-stage geoscience exploration pipeline evaluating placer Au and Rare Earth Element (REE/monazite) potential in NE Washington State using public geochemical, geophysical, and mine waste datasets.

## Overview

The pipeline integrates NURE stream sediment geochemistry, aeromagnetic data, lidar-derived volume estimates, MRDS mine site records, and WGS mine waste geochemistry (Earth MRI) to rank 12 placer sites by REE/NdPr potential and Au co-product signal.

**Top result:** Hunters Placer (#1, Stevens/Okanogan Co.) and Colville Placer (#2) score highest on combined criteria. All three top sites break even below the current NdPr oxide spot price (~$109/kg as of mid-2026). Recommended next step: auger sampling at Colville + Hunters (20–30 holes/site, ~$136k–$204k).

## Figures produced

| Figure | Script | Description |
|--------|--------|-------------|
| Fig 1 | `task1_coplacer_minerals.py` | Aeromagnetic × Th anomaly co-occurrence map |
| Fig 2 | `task2_source_lithology.py` | Source lithology map + WGS mine waste sites |
| Fig 3 | `task3_geochemical_discrimination.py` | Multi-element Th source discrimination |
| Fig 4 | `task4_volume_estimation.py` | Lidar volume estimation + WGS tonnage table |
| Fig 5 | `task5_energy_fuels_pathway.py` | Break-even / NPV / sensitivity analysis |
| Fig 6 | `task6_figure6.py` | Decision framework |
| Fig 7 | `integration_task.py` | Integrated multi-criterion priority map |
| Fig 8 | `task7_au_as_anomaly.py` | Au/As pathfinder anomaly map |
| Fig 9 | `task8_mine_waste_ree.py` | WGS mine waste REE + critical minerals (requires WGS file) |

## Data requirements

Large raster files (lidar, DEM, aeromagnetics) are **not included** in this repo due to size.  
See [`ne_wa_ree/DATA_SOURCES.md`](ne_wa_ree/DATA_SOURCES.md) for download instructions for each dataset.

Key public sources:
- **NURE stream sediment**: [USGS NGDB](https://mrdata.usgs.gov/ngdb/sediment/) — included in `data/nure/`
- **MRDS mine sites**: [USGS MRDS](https://mrdata.usgs.gov/mrds/) — included in `ne_wa_ree/data/mrds/`
- **Lidar DEMs**: [Washington Lidar Portal](https://lidar.wa.gov) — ~7 GB, download per site
- **30m DEM**: [USGS National Map](https://apps.nationalmap.gov/downloader/) — ~80 MB
- **WGS OFR 2026-02** (mine waste geochemistry): [WA DNR](https://www.dnr.wa.gov/publications/ger_ofr2026-02_mine_waste_characterization_part_1.zip) — place at `data/wgs_ofr2026/` or set `WGS_OFR2026_PATH` env var

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

Scripts must be run from the `ne_wa_ree/` directory in order (each task outputs files consumed by later tasks):

```bash
cd ne_wa_ree

# Prep data (run once from repo root)
cd .. && python3 prep_nure.py && cd ne_wa_ree

# Tasks 1–7 (independent of WGS file)
python3 task1_coplacer_minerals.py
python3 task2_source_lithology.py
python3 task3_geochemical_discrimination.py
python3 task4_volume_estimation.py
python3 task5_energy_fuels_pathway.py
python3 task6_figure6.py
python3 task7_au_as_anomaly.py

# Integration (depends on tasks 1–7)
python3 integration_task.py

# WGS mine waste integration (requires WGS_OFR2026_PATH or data/wgs_ofr2026/)
python3 task8_mine_waste_ree.py
```

On macOS, if matplotlib config errors occur:
```bash
export MPLCONFIGDIR=/tmp/mplconfig
```

## Key scientific findings

- **61 Th-anomalous NURE samples** in the NE WA study area; classified as MIXED/UNCLEAR or THORITE/U-Th oxide — no confirmed monazite fingerprint from stream sediment alone (Th-Ce-P co-enrichment required; limited by sparse NURE REE data)
- **WGS ICP-MS** (OFR 2026-02) provides first full REE suite for the region: First Thought Mine has highest TREE concentration (191 ppm) in tailings; Germania Mine has largest TREE endowment (~21,000 kg)
- **6 of 10 WGS sites** have mean Au ≥ 0.1 ppm (tailings reprocessing threshold); Deer Trail leads at 2.9 ppm
- **Environmental flag:** Big Iron and Silver Bell are acid-generating (NP/AP < 1); require acid management in any reprocessing scenario
- **Shankers Bend diatreme** (Loomis Quadrangle, ~50 km NW) contains confirmed carbonatite dikes — carbonatite REE source exists in the broader region

## Disclaimers

All outputs are **exploration screening estimates only** and do not constitute a mineral resource estimate under NI 43-101 or any other reporting standard. Economic figures are illustrative and subject to substantial uncertainty. The pipeline uses publicly available data; ground truthing is required before any investment decision.

## References

- Sun & McDonough 1989 — CI chondrite normalisation values
- Mücke & Bhaskara Rao 1996 — monazite discrimination criteria
- van Alderwerelt & Di Fiori 2026 — WGS OFR 2026-02 mine waste characterization
- USGS NURE HSDB — National Uranium Resource Evaluation geochemical database
- Rudnick & Gao 2003 — upper continental crust composition
