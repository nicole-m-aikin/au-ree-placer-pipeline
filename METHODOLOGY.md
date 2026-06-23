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

- **Task 1** — source mineralogy (monazite vs. thorite vs. background via U/Th discrimination)
- **Task 2** — catchment geology / pathway (source lithology score per drainage basin)
- **Task 3** — geochemical discrimination (multi-element fingerprinting of REE source)
- **Task 4** — trap volume and economic grade proxy (lidar vs. topo; Monte Carlo endowment)
- **Task 5** — trap economic viability (break-even NdPr price; Energy Fuels White Mesa pathway)
- **Task 6** — decision framework (structured go/no-go criteria across all components)
- **Task 7** — pathfinder halos around traps (Au/As anomaly delineation as placer vectors)
- **Task 8** — preservation context (mine waste ABA risk; WGS OFR 2026-02 field data)
- **Task 9** — data-driven spatial targeting across all components (ML probability surface)

---

## Dataset QA/QC Procedures

Every data quality step applied in the pipeline (Terra AI requirement: "QA/QC historical and
modern exploration datasets"):

**NURE stream sediment (primary dataset — task1, task3, task7, task9)**

- Values between 0 and −10 ppm: half-MDL substitution (replaced with abs(value)/2).
  Rationale: values reported as negative by the USGS NURE protocol represent
  below-detection-limit results where the MDL is encoded as the negative of the MDL.
- Values < −10 ppm (abs value > 10): set to NaN. These represent instrument artifacts
  or transcription errors in the USGS NURE database, not geochemical signals.
- P column: if median < 1 (implying values are in % rather than ppm), multiply by 10,000
  to convert to ppm. This handles the mixed-unit legacy encoding in the USGS NURE extract.
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

**NdPr endowment is double-weighted** because it is the primary economic driver;
all other criteria are screening filters that confirm or reduce confidence in the
endowment estimate.

**Weight sensitivity:** One-at-a-time ±50% perturbation of each weight is performed
in `integration.py` to verify that the top-3 ranking is stable. Results written to
`integration_weight_sensitivity.csv`.

---

## ML Targeting Model and Geostatistical Interpolation

**Model:** RandomForestClassifier (scikit-learn; n_estimators=200, random_state=42,
class_weight='balanced').

**Why Random Forest:**
- Handles mixed feature scales without normalization (log10 transform is done for
  geological reasons, not algorithmic ones)
- Captures non-linear multi-element interactions that linear discriminant analysis misses
- Provides Gini feature importance without requiring additional hyperparameter tuning
- class_weight='balanced' corrects for the ~15–20% anomalous sample fraction

**Geological feature engineering:** Log₁₀ transformation of the multi-element suite
(Th, Ce, La, P, U, Au, As) converts the log-normal geochemical distributions to
approximately normal, which is the standard preprocessing step in geochemical
exploration data science (Stanley & Sinclair 1989). The combination of multiple
pathfinder elements in a single model operationalizes the multi-element approach
already used in tasks 1, 3, and 7 as a probabilistic model.

**Validation:** 5-fold stratified cross-validation; metrics reported: ROC-AUC,
precision, recall, F1 for the anomalous class.

**IDW interpolation:** `scipy.interpolate.griddata` with method='linear' interpolates
point predictions to a 200×200 grid over the study area bbox. Cells > 0.5° from the
nearest sample are masked as NaN (undefined). IDW provides a continuous probability
surface suitable for visual target screening.

**Limitations:** IDW is not kriging. It does not estimate prediction uncertainty.
The recommended upgrade for a production workflow is ordinary kriging of the prediction
probabilities (e.g., via `pykrige`), which would add a kriging variance surface and
provide spatially explicit confidence bounds. The current IDW surface is appropriate
for exploratory screening.

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
- Blakely, R.J. et al. (1999). Aeromagnetic anomalies of the Pacific Northwest. USGS OFR 99-0440.
- Bonham-Carter, G.F. et al. (1988). Integration of geological datasets for gold exploration in Nova Scotia. *Photogrammetric Engineering & Remote Sensing*, 54(11), 1585–1592.
- McCuaig, T.C. & Hronsky, J.M.A. (2014). The mineral system concept: the key to exploration targeting. *SEG Special Publications*, 18, 153–175.
- MEND (2009). *MEND Manual, Volume 4: Sampling and Analysis*. Mine Environment Neutral Drainage Program.
- Mücke, A. & Bhaskara Rao, A. (1996). Opaque minerals in Sri Lankan gem-bearing eluvial and fluvial sediments. *Mineralogy and Petrology*, 58, 37–66.
- Rasmussen, B. & Muhling, J.R. (2009). Monazite begets monazite. *Contributions to Mineralogy and Petrology*, 158, 15–32.
- Reimann, C. & Filzmoser, P. (2000). Normal and lognormal data distribution in geochemistry. *The Science of the Total Environment*, 250(1–3), 267–281.
- Stanley, C.R. & Sinclair, A.J. (1989). Comparison of probability plots and the gap statistic in the selection of thresholds for exploration geochemistry data. *Journal of Geochemical Exploration*, 32(1–3), 355–357.
- Sun, S.S. & McDonough, W.F. (1989). Chemical and isotopic systematics of oceanic basalts. *Geological Society Special Publications*, 42, 313–345.
- Wyborn, L.A.I. et al. (1994). Australian Proterozoic mineral systems: essential ingredients and mappable criteria. *AusIMM Annual Conference*, 109–115.
