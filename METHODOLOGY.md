# Scientific Methodology

Au+REE Placer Assessment Pipeline — NE Washington (and extensible to other US placer REE districts)

---

## Mineral Systems Framework

**This section is the primary scientific organizing principle of the entire pipeline.**

The pipeline implements a mineral systems analysis following the established
**source → pathway → trap → preservation** framework (Wyborn et al. 1994; McCuaig & Hronsky 2014):

| Component | NE Washington expression | Pipeline task(s) |
|-----------|--------------------------|-----------------|
| **Source** | Carbonatite-associated REE (Shankers Bend diatreme, ~50 km NW) and Th-enriched Okanogan MCC metapelites as the primary Th/LREE source rocks. Felsic intrusives (peraluminous granitoids of the Colville Batholith) as a secondary accessory-mineral source. | Tasks 1, 3 |
| **Pathway** | Fluvial transport along the Okanogan, Sanpoil, Kettle, and Columbia River systems redistributing resistant heavy minerals (monazite, magnetite) from catchment headwaters to terrace and floodplain sinks. | Task 2 |
| **Trap** | Hydraulic energy breaks at valley bends, downstream of bedrock constrictions, and in alluvial fan-toe positions. The 12 MRDS placer districts represent known trap sites where historical gold-dredging operations concentrated heavy minerals. | Tasks 4, 5 |
| **Preservation** | Historical mining tailings and un-mined terrace gravels. Some sites have been disturbed (acid-generating tailings flagged via ABA). | Tasks 4, 8 |

Each pipeline task evaluates one or more components of this framework:

- **Task 1** — source mineralogy (monazite vs. thorite vs. background via U/Th discrimination). Site–NURE join is `geochemistry.site_join_radius_deg` (default 0.10°, nearest sample). The old 0.25° max-Th window assigned far chemistry to the ranked piles and is retired.
- **Task 2** — catchment geology / pathway (source lithology score per drainage basin)
- **Task 3** — geochemical discrimination (multi-element fingerprinting of REE source)
- **Task 4** — trap volume and economic grade proxy (lidar vs. topo; Monte Carlo endowment)
- **Task 5** — trap economic viability (break-even NdPr price; Energy Fuels White Mesa pathway)
- **Task 6** — decision framework (structured go/no-go criteria across all components)
- **Task 7** — pathfinder halos around traps (Au/As anomaly delineation as placer vectors)
- **Task 8** — preservation context (mine waste ABA risk; WGS OFR 2026-02 field data)
- **Task 9** — data-driven spatial targeting across all components (ML probability surface)
- **Task 11** — trap walk list: chemistry picks the drainage; stream-geometry votes pick the pan pin; pamphlet / opt-in pans are a catchment hit-rate overlay (not AUC)
- **Task 12** — transfer test: frozen NE WA forest on Idaho (AUC 0.50), California Sierra (0.52), and Montana SW gulches (0.34; do not retrain)
- **Task 13** — in-belt east-west hold-out on NE WA (does not rewrite the published joblib)

---

## Dataset QA/QC Procedures

Every data quality step applied in the pipeline :

**NURE stream sediment (primary dataset — task1, task3, task7, task9)**

- Values between 0 and −10 ppm: half-MDL substitution (replaced with abs(value)/2).
  Rationale: values reported as negative by the USGS NURE protocol represent
  below-detection-limit results where the MDL is encoded as the negative of the MDL.
- Values < −10 ppm (abs value > 10): set to NaN. These represent instrument artifacts
  or transcription errors in the USGS NURE database, not geochemical signals.
- P column: if median < 1 (implying values are in % rather than ppm), multiply by 10,000
  to convert to ppm. This handles the mixed-unit legacy encoding in the USGS NURE extract.
- Fe column: same conversion when median < 100 (NURE reports Fe as wt%, median ~2–5).
  P and Ca were already converted; Fe was not, which left Task 9 with a mixed-unit
  feature matrix. After conversion every published feature is ppm.
- Coordinate filtering: samples outside the study area bounding box are excluded per-run;
  no permanent removal from the master CSV.

**MRDS mine sites (task1, task2, integration)**

- Filtered by commodity list from config (`mrds_commodities`): retains placer gold, gold,
  REE, monazite, magnetite entries.
- Duplicate coordinates removed (same lat/lon to 4 decimal places).
- Bounding box clipped to study area per config bbox.

**Aeromagnetic data (task1)**

- Used as-is from USGS derivative product (RTP-reduced total field anomaly GeoTIFF).
- No additional processing applied. The source TIF is already a derivative product
  (RTP processed from flight-line data); raw flight-line data would be required for
  Werner/Euler deconvolution.
- When no TIF is supplied (null in config), synthetic anomaly centers from
  Blakely et al. 1999 (Pacific NW aeromagnetic compilation) are used as a placeholder.

**Lidar DEMs (task4)**

- Per-site TIFs from lidar.wa.gov.
- Quality tier assignment: 1 m lidar (acquisition ≥ 2015) = HIGH;
  2–5 m lidar or pre-2015 = MEDIUM; topo-derived only = LOW.
- Synthetic surface model used in fig4 when real TIFs are not available.

**WGS mine waste (OFR 2026-02) (tasks 2, 4, 8)**

- Excel import with column validation via `openpyxl` / `pandas`.
- Sites without ICP-MS REE data in the Geochemistry sheet are excluded from
  task8 REE analysis and flagged separately in the summary CSV.
- Sites explicitly listed in `wgs.exclude_sites` (config) are removed before
  any analysis (e.g., sites with known data quality issues).
- Coordinate filtering applied to Endowment sheet: sites outside wgs_bbox excluded.

---

## Geochemical Anomaly Threshold

**Method:** mean + 2 SD on log₁₀-transformed positive values (implemented in `utils.anomaly_threshold`).

**Rationale:** Geochemical data in natural rock and sediment samples follows a log-normal
distribution (Ahrens 1954; Reimann & Filzmoser 2000). Applying mean + 2 SD to raw (linear)
values would produce thresholds biased upward by a small number of extreme values and would
not correctly represent the statistical behaviour of the population. Log-transformation
normalises the distribution, making the mean + 2 SD criterion statistically meaningful:
approximately 2.5% of background samples are expected to exceed the threshold.

**Implementation detail:** Values ≤ 0 and NaN are excluded before log-transformation
(they are not geochemically meaningful and would produce undefined log values).
A minimum of 5 positive values is required to compute a threshold; if fewer are present,
`anomaly_threshold` returns None (threshold undefined for that element in that area).

---

## Th Source Mineral Discrimination

**Criteria (from `task3_geochemistry.py`):**

| U/Th ratio | Interpretation | Reference |
|------------|----------------|-----------|
| < 0.5 | Monazite dominant | Mücke & Bhaskara Rao 1996 |
| 0.5 – 1.5 | Mixed / ambiguous | — |
| > 1.5 | Thorite dominant | Mücke & Bhaskara Rao 1996 |

**Ce/La ≈ 1.5–2.5** is used as a LREE enrichment check consistent with monazite-hosted
LREE fractionation (typical Ce/La in metamorphic monazite from the Okanogan MCC;
Rasmussen & Muhling 2009).

**Why MIXED/UNCLEAR is the expected result:** Stream sediment integrates signals from
multiple source minerals across the entire catchment. A pure U/Th < 0.5 signal requires
dominance of monazite over all other U/Th-bearing phases (thorite, zircon, xenotime, apatite).
Mixed results do not indicate absence of monazite — they indicate a polymineral heavy-mineral
assemblage, which is geologically expected in NE Washington.

---

## Chondrite Normalization

**Reference values:** CI chondrite (Sun & McDonough 1989) — implemented in
`utils.CHONDRITE_SUN89`.

**Why CI chondrite (not primitive mantle):** CI chondrite normalization is the standard
for evaluating crustal REE patterns and is directly comparable to the global stream
sediment REE literature. Primitive mantle normalization would be appropriate for
evaluating mantle-derived rocks (komatiites, OIB) but is unconventional for sedimentary
and crustal REE work.

**What the normalized pattern shows:** A steeply negative slope from LREE to HREE
(La/Yb)_N >> 1 indicates strong LREE enrichment, consistent with a monazite source.
A flat pattern indicates apatite or xenotime contributions. The Eu anomaly (Eu/Eu*)
indicates feldspar involvement (positive = cumulate; negative = fractionated melt).

---

## Volume and Grade Uncertainty

**Depth uncertainty:** Treated as 1-sigma of a normal distribution, derived from the
range of MRDS production records for each site. The standard deviation is set per-site
in config (`depth_uncertainty_m`).

**Grade uncertainty (stream sediment → in-situ grade proxy):**

Stream sediment Th values represent a spatial average over the upstream catchment,
not a point measurement of the deposit grade. The conversion factor of 5× (config:
`stream_to_insitu_factor`) reflects the dilution of a concentrated placer deposit signal
across the catchment sample area (Bonham-Carter et al. 1988). A log-normal grade
distribution with σ = 0.4 in log space (corresponding to approximately ±50% at 1σ)
is assumed for the Monte Carlo, consistent with the grade variability observed in
published placer REE datasets.

**Monte Carlo implementation (task4):** 2,000 samples per site drawn from:

- Depth: Normal(depth_m, depth_unc_m), clipped at 0
- Grade: LogNormal(ln(ndpr_ppm), 0.4)
- Tonnage: area_m² × depth_sample × bulk_density
- Endowment: tonnage × grade / 10⁶

P10 / P50 / P90 percentiles reported. P50 is the headline estimate used in integration scoring.

**Why undiscounted NPV (task5):** Project life is assumed short (< 5 years for an
exploration-stage placer re-mining operation). At exploration stage, discount rate
assumptions carry greater uncertainty than the project life, so discounting adds false
precision. The break-even NdPr price is the primary economic screening criterion.

---

## ABA Risk Tiers

**Method:** Net Potential (NP) / Acid Potential (AP) ratio using MEND (2009) thresholds:

| NP/AP ratio | Interpretation |
|-------------|----------------|
| < 1 | Potentially acid-generating |
| 1 – 2 | Uncertain (field confirmation required) |
| > 2 | Non-acid-generating |

**Why this matters:** Any tailings reprocessing scenario for REE recovery requires
addressing the existing acid-generating potential of historical mine waste. Sites with
NP/AP < 1 carry significant remediation cost risk and regulatory complexity that must
be factored into any project economics.

---

## Combined Priority Scoring

**Scoring components and rationale (integration.py):**

| Criterion | Score range | Weight | Rationale |
|-----------|-------------|--------|-----------|
| Th source (monazite) | 0–2 | 1.0 | Primary mineralogical signal |
| Magnetic high | 0–1 | 1.0 | Proxy for magnetite/monazite concentration |
| Source lithology | 0–3 | 1.0 | Catchment REE source rock quality |
| Data coverage | 0–2 | 1.0 | Penalizes low-quality lidar/topo data |
| NdPr endowment (P50) | 0–2 | **2.0** | Primary economic driver — double-weighted |
| Au/As pathfinder | 0–2 | 1.0 | Placer concentration vector |
| Y / xenotime (HREE) | 0–1 | 0.5 | Down-weighted; Y₂O₃ ~$3.50/kg vs NdPr ~$109/kg |
| ML probability | 0–1 | 1.0 | Nearest NURE P(anomalous) within 0.15° |

**NdPr endowment is double-weighted** because it is the primary economic driver;
all other criteria are screening filters that confirm or reduce confidence in the
endowment estimate.

**Weight sensitivity:** One-at-a-time ±50% perturbation of each weight is performed
in `integration.py`. After the 0.10° nearest-sample join, Hunters stays #1–#2;
Bossburg and Oroville can leave the top 3. Do not say the top 3 are stable.
Results written to `integration_weight_sensitivity.csv`.

---

## ML Targeting Model and Geostatistical Interpolation

**Model:** RandomForestClassifier (scikit-learn; n_estimators=200, random_state=42,
class_weight='balanced').

**Why Random Forest:**
- Handles mixed feature scales without normalization (log10 transform is done for
  geological reasons, not algorithmic ones)
- Captures non-linear multi-element interactions that linear discriminant analysis misses
- Provides Gini feature importance without requiring additional hyperparameter tuning
- class_weight='balanced' corrects for the minority-class imbalance in the labelled set (see the positive fraction reported under the labelling section below)

**Anomaly label — MRDS proximity (geochemistry-independent ground truth):**

A sample is labelled positive if it lies within **0.15° (~16 km) of any MRDS placer
deposit site** in the study area. This is the critical design choice. Any label derived
from the input geochemical features (e.g. Th > threshold, top-N% anomaly index) produces
a circular classifier: the model learns to reconstruct its own input rather than discovering
geologically generalizable patterns, and achieves AUC approaching 1.0 regardless of model
complexity or cross-validation scheme.

Using MRDS proximity as the label makes the classification problem genuine: the model
learns which geochemical signatures in the NURE stream sediment data are characteristic
of samples collected near known placer deposits. This is the exact question a production
targeting model answers — train on confirmed deposit proximity, predict on unsampled terrain.

The 0.15° radius (configurable via `ml.mrds_proximity_deg` in config) captures ~28% of
samples as positive, providing reasonable class balance for 5-fold stratified CV. The
radius was selected to encompass the regional drainage catchment of each placer site.

If the MRDS GeoJSON is not available, the task falls back to a top-10% multi-element
anomaly index label (configurable via `ml.anomaly_top_pct`), with an explicit warning
in the output. The fallback AUC will be higher and should not be reported as evidence
of genuine predictive power.

**Geological feature engineering — full placer heavy mineral suite:**

Log₁₀ transformation of 11 elements converts log-normal geochemical distributions
to approximately normal (Stanley & Sinclair 1989). The feature set covers the complete
suite of minerals that co-concentrate through hydraulic sorting in placer environments:

| Element | Mineral(s) | Role |
|---------|-----------|------|
| Th, Ce, La, P | Monazite — (LREE)PO₄ | Primary REE target |
| U | Uraninite / thorite | REE/actinide indicator |
| Au, As | Native gold + arsenopyrite halo | Au pathfinder pair |
| Ti | Rutile + ilmenite — TiO₂, FeTiO₃ | Oxide heavy mineral |
| Fe | Magnetite — Fe₃O₄ | Oxide heavy mineral (co-placer with monazite) |
| Zr | Zircon — ZrSiO₄ | Silicate heavy mineral, very resistant to weathering |
| Y | Xenotime — YPO₄ | Phosphate heavy mineral, common monazite associate |

Ti, Fe, Zr, and Y all concentrate by the same hydraulic sorting mechanism as monazite
(specific gravity 4.6–5.2). Including the full oxide and silicate heavy mineral
suite means the model learns the entire placer assemblage fingerprint, not only the
REE-bearing fraction. With the full 11-element feature set, CV ROC-AUC = 0.891 ± 0.018 (NE Washington).

**Elevation filter (downstream/valley-floor constraint):**

Placer deposits form on valley floors where hydraulic energy decreases. A NURE stream
sediment sample taken on a hillside within 3 km of a placer mine may be in the
source-rock terrain (bedrock geochemistry) rather than the placer zone. When a DEM
covering the study area is available (`data.dem_tif` in config), the label applies an
additional constraint: the NURE sample and its nearest MRDS site must be within
`ml.mrds_elev_diff_m` metres of each other (default 200 m), restricting positives to
samples on the same valley-floor terrace as the deposit. Samples with no DEM coverage
(NaN elevation) pass this filter by default.

For NE Washington: the DEM is a Copernicus GLO-30 mosaic (8 × 1° tiles, downloaded from
AWS Open Data via `download_dem.py`), covering lon −120→−116 / lat 47→49, 14400×7200 px
at ~30m resolution, elevation range 171–2529m. After applying the elevation filter,
290 of 1045 samples (27.8%) are labelled positive, with final CV ROC-AUC = 0.891 ± 0.018.

**Validation:** 5-fold stratified cross-validation; metrics reported: ROC-AUC,
precision, recall, F1 for the anomalous class. Shuffled k-fold does not account
for spatial autocorrelation (Airola 2018). This repo now reports both numbers
on `/model-info` and in `models/task9_rf_placer_gold.meta.json`:

| Test | ROC-AUC |
|------|---------|
| Shuffled 5-fold (neighbors of the same mine can leak) | **0.891 ± 0.018** |
| Dead-zone spatial CV (drop train samples within 0.15° of a test sample) | **0.699 ± 0.055** |
| 0.4° cell blocked CV | **0.757 ± 0.139** |
| Frozen forest on Idaho Batholith NURE (Task 12; not retrained) | **0.50** |
| Frozen forest on CA Sierra foothills NURE (Task 12; not retrained) | **0.52** |
| Frozen forest on Montana SW gulches NURE (Task 12; not retrained) | **0.34** |
| In-belt hold-out: published forest on east of −118.50° (Task 13) | **0.53** |
| In-belt hold-out: refit west, test east (not the published joblib) | **0.63** |

0.891 is “can the forest separate yes/no in this belt when neighbors are allowed.”
0.70 is “new drainage in the same belt.” 0.53 / 0.63 is the same belt, other
side of the Kettle. 0.50 / 0.34 is “new belt.” Do not quote only 0.891.

**IDW interpolation:** `scipy.interpolate.griddata` with method='linear' interpolates
point predictions to a 200×200 grid over the study area bbox. Cells > 0.5° from the
nearest sample are masked as NaN (undefined). IDW provides a continuous probability
surface suitable for visual target screening.

**Limitations:** IDW is not kriging. It does not estimate prediction uncertainty.
The recommended upgrade for a production workflow is ordinary kriging of the prediction
probabilities (e.g., via `pykrige`), which would add a kriging variance surface and
provide spatially explicit confidence bounds. The current IDW surface is appropriate
for exploratory screening.

**Catchment labeling — QA diagnostic, not the published model:**

`label_method: catchment` delineates D8 upstream polygons from the study-area
site list (`cfg.sites`, typically 12 pour points) and labels a NURE sample
positive if it falls inside any of those polygons. Snap uses a precomputed
accumulation mask (default p99.5) and windowed polygonize. This is *not* the
figure of record, for two reasons:

1. **Sample size.** Twelve drainages labelled 54 of 1,045 NURE samples (5.2%).
   Five-fold CV then has ~11 positives per fold. The resulting ROC-AUC (0.959)
   is not comparable to the proximity result and is easy to over-read.
2. **Tautology.** The pour points are the same 12 sites the rest of the pipeline
   already ranks. The model is asked to recognise drainages we already picked,
   not to generalise to unlisted ground.

Those numbers **are** retained as a QA ledger in
`ne_wa_ree/outputs/tables/task9_label_qa.csv` and locked by
`tests/test_integration_and_ml.py::TestLabelQaLedger`. Use them as a
sanity check, not a resume metric:

- Catchment positives must stay a minority of proximity positives (54 vs 290).
  If they converge, the 12-site snap has ballooned into regional drainages.
- Catchment AUC should be *higher* and *noisier* than proximity. If it falls
  below 0.891, the diagnostic inverted.
- Published top feature is U; catchment top feature is P. A swap on the
  proximity side is a data or code regression.

A scientifically useful catchment *training* experiment would use a broader
set of independent placer occurrences as pour points — not the ranked target
list. Do not pass the full ~1,600-record MRDS gold inventory into
`delineate_catchments`; that is a multi-hour full-grid polygonize.

**Same-basin labeling — method test, still not the published model:**

`python -m pipeline.task9_ml_targeting --same-basin` uses independent
placer-named MRDS records as pour points, holds out the 12 ranked config
sites (±0.03°), unique-cells at 0.02°, D8 catchments plus a 0.02° trap
buffer, and does **not** overwrite Fig 10 or the published CV tables.

On the NE Washington extract that produced 82 unique pour points, 78
delineated catchments, and 254 of 1,045 NURE positives (24.3%). 250 of
those 254 also meet the published proximity definition, so same-basin is
essentially a hydrologic tightening of the circle, not a new labeled
population. Five-fold CV ROC-AUC is 0.947 ± 0.014 with Fe as the top
Gini feature. That AUC is not a resume number: the class is cleaner
than proximity and still spatially autocorrelated. Do not quote 0.947
or 0.959 as model performance. Result of record remains proximity
(0.891 ± 0.018, ~290 positives, U first). Ledger:
`ne_wa_ree/outputs/tables/task9_same_basin_qa.csv`.

The published Task 9 numbers, Fig 10, and resume claims use
`label_method: proximity`.

**Aerial gamma-ray (NURE K / eTh / eU) — how it enters the pipeline:**

NURE was two surveys. Stream sediment (already in Task 3/9) is laboratory
chemistry of grab samples. Aerial radiometrics is aircraft gamma-ray
spectrometry of the top ~30 cm: potassium (%K), equivalent thorium (eTh),
equivalent uranium (eU). Grids: Duval compilation at
https://mrdata.usgs.gov/radiometric/ (see `DATA_SOURCES.md`).

Fetch and clip with `python -m pipeline.fetch_radiometric` (Duval 2005
Esri FLT concentration grids, reprojected from DNAG TM to EPSG:4326).
The NArad_*_geog83.tif files on mrdata are RGB previews, not sampled.
Config `data.radiometric_*_tif` already points at the clipped NE WA GeoTIFFs.
`utils.load_radiometric_at_points` samples them. Uses, in order:

1. **Task 1 overlay (implemented).** `fig1b_airborne_eth_vs_nure_th.png`
   sits beside the synthetic magnetic grid — it does not replace it.
   Panel A is airborne eTh + NURE stream-sediment Th anomalies; panel B
   is the point-level concordance scatter. Agreement is a ground-truth
   check on both datasets; disagreement flags drainage transport, cover,
   or survey gaps. Site-table columns `rad_*` / `rad_th_agree` and
   `task1_radiometric_concordance.csv` are written only when the eTh
   TIFF exists. Airborne highs use linear mean+2SD of the windowed grid
   (same convention as the magnetic placeholder, not the log-space
   geochemical threshold). On the NE WA clip: Spearman ρ(stream Th,
   airborne eTh) ≈ 0.05; 76 stream-only vs 1 both_high (C165101, not a
   ranked pile). After the 0.10° nearest-sample join, Hunters is
   `neither` (local C164654, 20 ppm BACKGROUND). The plane is a check
   on the join, not a scoring layer.
2. **Ratio maps (not yet).** eU/eTh is the continuous version of the
   Task 3 U/Th monazite-vs-thorite screen. K/eTh separates felsic
   (high K) from Th-rich metapelite / monazite catchments (low K/eTh).
3. **Task 9 features, later — not now.** Adding rad_eTh, rad_K, rad_eU
   to the RF is possible but is *not* the next step: the grids are
   smooth at 1–10 km, so they leak spatial autocorrelation into CV.
   Use them as a map layer and a site-table column before they become
   model features.

Do not invent a synthetic K/Th grid the way Task 1 currently synthesizes
magnetics. If the TIFFs are absent, radiometric columns stay empty and
Fig 1b is skipped.

---

## Public doorbell and what `/predict` is not

The published forest is persisted by Task 9 (`models/task9_rf_placer_gold.joblib`
plus metadata and training log-medians). FastAPI (`api/app.py`) loads those
files. Live: https://placer-lookalike.onrender.com/docs

`/predict` returns P(gold-placer lookalike), tree-vote spread (std of the 200
tree probabilities), and — if lon/lat are sent — distance to the nearest
**training** gold MRDS pin. Tree-vote spread is not a Monte Carlo interval and
is not Task 4 NdPr P10/P50/P90. Fe requires `fe_unit`. A 0-class label means
“not near a mapped gold mine,” not “barren.”

## Task 11 walk list

The decision unit is the drainage (Yousefi & Carranza 2013), not an IDW blob.

- `nure_spots` — highest-P NURE grabs in the catchment (chemistry).
- `pan_locations` — D8-stream vertices flagged for slope break, stream-power
  drop, knickpoint foot, or tributary junction (geometry, after chemistry).
- `pour_points` — D8 outlet. Not automatically a pan pin.

Stay out of the forest: slope and stream power are not Random Forest features.
They leak space and retrace the 200 m valley-floor label rule.

**Hobby / pamphlet overlay (`hobby_reports`)** — a positive-only occurrence
layer, not a retraining set and not ROC-AUC. Seed rows are cited public
gazetteer points (WA DNR recreational gold-panning pamphlet / OFR 79-0;
USFS Nez Perce-Clearwater recreational mining waters; BLM / CA State Parks
designated panning areas; USGS GNIS for the named place). `gold_class` on
those rows is `unknown`, not a recovery. Opt-in form rows append to the
same CSV (`data/hobby_reports/OPT_IN_FORM.txt`). Do not scrape forums.

Join rules, enforced in `pipeline/task11_hobby_reports.py`:

- `point` / `bar` / `reach` / `creek` can hit a walkable catchment.
- `district` is mapped and does **not** count as a drainage hit.
- Only `location_precision=point` gets `nearest_pan_pin_m`.
- Metric: catchment hit-rate (how many walkable catchments contain an
  eligible report; how many of those are `expedition`). People post
  flakes, not blanks — there is no honest 0-class until form blanks arrive.

This is still not a substitute for walking the Task 11 pins. Gazetteer
rows in California and Idaho mostly rediscover MRDS geography. The useful
test is an eligible hit in a NE WA **expedition** cell.

## Transfer belts (Task 12)

Score the frozen NE WA forest on another placer box. Do not retrain.

Idaho Batholith: AUC 0.50 (2,465 grabs; 79% within 0.15° of a gold pin). Mean
P is ~0.38 next to gold and far from it. Tightening to 0.05° does not help.

California Sierra foothills (Feather / Yuba / American): AUC 0.52 (596 grabs;
88% near gold; 7,872 MRDS gold pins). Mean P is again ~0.38 both sides.
Tightening to 0.05° lifts AUC to 0.61 — still a weak ranker, not a walk list.
This HSSR clip has no Au or As; those two features are Washington log-medians.

Montana SW gulches (Confederate / Alder / Montana Bar / Elkhorn; not Libby):
AUC 0.34 (4,672 grabs; 94% near gold; 2,039 MRDS gold pins). Mean P is 0.28
next to gold and 0.38 far from it — inverted. P and Y are missing in this
HSSR clip and filled with Washington log-medians. Tightening to 0.05° is
0.33. Tasks 1–10 stay off: no public site depths, no WGS-style waste OFR.

Walk-list GeoPackages now exist for Idaho, California, and Montana. P is
still the frozen Washington forest. Gold distance uses each belt's own MRDS
pins. Idaho has two expedition cells (max P 0.78). California has zero —
every occupied cell is watch (max P 0.52). Montana has one expedition and
two confirm (mean P 0.29; leave-one-cell-out 0.52). The pans are DEM
geometry, not a new model.

That is the literature failure mode, not a reason to build a national model.

---

## 3D Modeling Scope and Limitations

The pipeline is intentionally 2D — all analysis is performed in geographic (lat/lon)
space or at site scale using lidar-derived elevations.

**What is explicitly out of scope:**

- **Depth-to-source from aeromagnetics:** Werner deconvolution and Euler deconvolution
  (standard methods for estimating depth to magnetic sources) require flight-line-level
  aeromagnetic data. The USGS derivative product (RTP GeoTIFF) used here does not
  provide the line spacing or datum information required for these methods.

- **3D geological modeling:** Leapfrog Geo, GOCAD/SKUA, or equivalent implicit modeling
  would be the next step after positive auger results, to build a block model of the
  deposit geometry. This requires drillhole assay data that does not yet exist for these
  sites.

- **Block model grade estimation:** The Monte Carlo in task4 is a 1D vertical
  uncertainty model (depth × area), not a 3D block model with spatial grade continuity.
  It appropriately represents the uncertainty at the exploration-target stage, where
  drill density is insufficient to build a geostatistical variogram.

---

## References

- Ahrens, L.H. (1954). The lognormal distribution of the elements. *Geochimica et Cosmochimica Acta*, 5(2), 49–73.
- Airola, A. et al. (2018). A comparison of leave-one-out and leave-pair-out cross-validation for assessing spatial prediction models. *Data Mining and Knowledge Discovery*.
- Blakely, R.J. et al. (1999). Aeromagnetic anomalies of the Pacific Northwest. USGS OFR 99-0440.
- Bonham-Carter, G.F. et al. (1988). Integration of geological datasets for gold exploration in Nova Scotia. *Photogrammetric Engineering & Remote Sensing*, 54(11), 1585–1592.
- McCuaig, T.C. & Hronsky, J.M.A. (2014). The mineral system concept: the key to exploration targeting. *SEG Special Publications*, 18, 153–175.
- MEND (2009). *MEND Manual, Volume 4: Sampling and Analysis*. Mine Environment Neutral Drainage Program.
- Mücke, A. & Bhaskara Rao, A. (1996). Opaque minerals in Sri Lankan gem-bearing eluvial and fluvial sediments. *Mineralogy and Petrology*, 58, 37–66.
- Rasmussen, B. & Muhling, J.R. (2009). Monazite begets monazite. *Contributions to Mineralogy and Petrology*, 158, 15–32.
- Reimann, C. & Filzmoser, P. (2000). Normal and lognormal data distribution in geochemistry. *The Science of the Total Environment*, 250(1–3), 267–281.
- Stanley, C.R. & Sinclair, A.J. (1989). Comparison of probability plots and the gap statistic in the selection of thresholds for exploration geochemistry data. *Journal of Geochemical Exploration*, 32(1–3), 355–357.
- Sun, S.S. & McDonough, W.F. (1989). Chemical and isotopic systematics of oceanic basalts. *Geological Society Special Publications*, 42, 313–345.
- Washington Division of Geology and Earth Resources (2025). *Recreational Gold Panning in Washington State*. https://dnr.wa.gov/sites/default/files/2025-03/ger_gold_panning.pdf
- Wyborn, L.A.I. et al. (1994). Australian Proterozoic mineral systems: essential ingredients and mappable criteria. *AusIMM Annual Conference*, 109–115.
