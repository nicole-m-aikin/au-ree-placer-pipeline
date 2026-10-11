# Phase report — desk-study phase (NE Washington doorbell, California walk list)

Structured to the internal-report headings of the CIM Mineral Exploration
Best Practice Guidelines (2018), §2.8. Status at 27 Sep 2026. Exploration
targets only; not a resource estimate; not NI 43-101 or JORC disclosure.

**Bottom line.** Compilation of public data is thorough. Validation of the
NE Washington legacy data removed most of it (287 of 1,045 NURE samples
survive) and, on what survives, Task 9 skill is weak (0.58 block vs 0.757
published) and sits mostly in one submitter's samples.
No legacy data has been verified in the field. The next phase is
verification sampling in the California Sierra under `docs/FIELD_PROTOCOL.md`,
not more modelling.

**Responsible geoscientist.** _To be named._ CIM §2.1 requires one
geoscientist accountable for planning, QA and reporting; this cannot be
delegated to the pipeline.

---

## 1. Property description and location

Desk-study areas, not tenure. NE Washington (Okanogan / Ferry / Stevens
Co.; 12 historical placer/lode sites in `configs/ne_washington/config.yaml`)
and the California Sierra foothills (Feather / Yuba / American; walk list in
`california_sierra/outputs/`). Transfer-test belts (Idaho, Montana,
Colorado, NC Fall Zone) are tests only. Coordinates are WGS84 (EPSG:4326);
distances in UTM. 4 of the 12 NE WA site locations are tied to MRDS:
Republic `10041093` and Northport `10204774` by name; Kettle Falls
`10180995` (Meyers Falls Placer) and Meyers Creek `10072470` (district-level
record; latitude corrected 48.32 → 48.83 N) accepted on review. Brender's
fuzzy match to Bender Custom Mill was rejected. The other 8 are rounded
config points marked `location_status: unverified` (`site_verification.csv`,
`mrds_review` in config).

## 2. Accessibility, climate, local resources, infrastructure, physiography

California pins carry PAD-US manager, BLM MLRS claim status, and (new)
Census AIANNH tribal-land flags plus a per-pin `rules_note`
(`pipeline/task11_land_access.py`). No California pin is on tribal land.
In NE Washington land access is still switched off; the in-memory check puts
Sanpoil River (16 km inside) and Old Dominion (3.5 km) on the Colville
Reservation. Meyers Creek was inside only at its old, wrong coordinate; the
corrected point is 39 km outside. Hunters sits 0.19 km
inside the boundary — within its coordinate rounding, so undetermined until
the location is verified. Tribal land requires tribal permission before any
visit. Climate and infrastructure are not yet documented.

## 3. Previous exploration activities and results

Compiled (all public domain; sources, hashes and dates in
`data/VERIFICATION_REGISTER.csv`): NURE/NGDB stream sediment, MRDS, USGS
SGMC geology, NURE airborne K/eTh/eU (Duval 2005), Copernicus GLO-30 DEM,
WA lidar, WGS OFR 2026-02 mine-waste characterization, PAD-US, BLM MLRS,
recreational panning gazetteers. Not compiled: a real aeromagnetic grid
(NE WA uses synthetic anomaly centres — now excluded from scoring), historical
placer production records, WA DNR placer reports. Site `area_ha`, `depth_m`
and bulk density (1.5 t/m³) are unsourced assumptions.

## 4. Geological setting and mineralization

Source → pathway → trap → preservation framework (`METHODOLOGY.md`, Mineral
Systems Framework). Monazite and heavy-mineral placers derived from the
Okanogan MCC metapelites and Colville Batholith (NE WA) and the ancestral
Yuba gravels (Sierra). Unchanged by this audit.

## 5. Deposit types

Fluvial Au placer with co-concentrated monazite / zircon / magnetite /
ilmenite; historical placer tailings as a REE re-mining target (NE WA).
See `LIT_REVIEW.md`.

## 6. Mineral exploration activities (this phase)

Desk only: data compilation, Tasks 1–14, frozen NE WA forest and transfer
tests, California local forest and walk list, and — this phase — the CIM
validation and verification work below. No field sampling by the project.

## 7. Drilling results

None. Auger testing is recommended only after field verification (§12).

## 8. Sample preparation, analyses, security, and QA/QC

*Legacy data.* NURE samples were collected and analysed by USGS/DOE
contractors across multiple jobs and methods. The NE WA extract mixes
media (stream 652, lake 143, surface 132, wetland 105, auger 13), analytical
methods with detection limits from 0.05 to 200 ppm, and semi-quantitative
spectrographic results. Per-job USGS method and QA documentation has not yet
been reviewed. A forest can name the submitting lab from the chemistry
(balanced accuracy 0.857 vs chance 0.143; 0.899 vs 0.333 on validated
samples). No certificates exist for
legacy data; this is a limitation, not a fixable gap.

*Project samples.* None yet. `docs/FIELD_PROTOCOL.md` defines the program:
orientation reach at a known placer; 6–10 twins of 1-second NURE sites;
−80 mesh sediment + fixed-volume pan concentrate; field duplicate, certified
barren blank and stream-sediment CRM each at ~1 in 20; ≥ 5% check assays at
a second lab; ISO/IEC 17025 lab with signed certificates; chain of custody,
tamper-evident seals, retained splits. `pipeline/field_qaqc.py` enforces
the acceptance gates; only primaries from passing batches
(`field_accepted_primaries.csv`) may enter any other task.

## 9. Data verification

CIM §2.1.1.1: validation checks for errors and mismatches; verification
checks that the data represent reasonable results. Full numbers in
`METHODOLOGY.md` → *Data validation and verification*.

**Validation performed** (`pipeline/data_validation.py`):

| Check | NE WA result |
|-------|--------------|
| Coordinate precision (≤ 0.01°) | 618 dropped; 548 shared one 7.5′ placeholder point |
| Sample medium (stream + "surface", admitted on review) | 140 lake dropped → 287 remain (Grosz 132, Miller 91, Frisken 63, other 1) |
| Censored values | kept at half-MDL with MDL (legacy loader NaN'd 603 of 992 Th) |
| Feature gate | Th, U, Au, As dropped (53–97% censored); keeps Ce, La, P, Ti, Fe, Zr, Y |
| Lab batch | lab predictable from gated chemistry (0.899 vs 0.333); submitter alone predicts the label (AUC 0.57) |
| MRDS duplicates | 101 removed (no label change) |
| Site IDs | placeholder IDs `M004321`–`M004332` replaced; 4 of 12 tied to MRDS (2 on review) |
| Au units (`prep_nure.py`) | now normalised to ppm with unit retained; g/t note fixed |

**Verification performed** (`pipeline/verification.py`):

| Method | Result |
|--------|--------|
| Re-analysis pairs (`previous_lab_id`) | 6 of 106 resolve; median HARD 0.014, no bias |
| Same-`field_id` pairs across labs | 22 pairs; median HARD 0.081 (La 0.17, Y 0.20 biased); Th/Au/U untestable |
| Airborne eTh vs stream Th | ρ 0.046 on all media (legacy); ρ 0.27 (n 120, p 0.003) on validated detected Th — modest corroboration |
| NURE vs WGS OFR 2026-02 | 3 of 13 WGS sites within 5 km of one validated sample; mine waste vs stream sediment, not comparable |
| Site location vs MRDS | 2 name-matched ≤ 5 km, 2 accepted on review, 1 rejected, 3 far name-only, 4 no match |

**Effect on results of record.**

- Task 9 (NE WA): published 0.891 shuffled / 0.699 dead-zone / 0.757 block
  → validated 0.66 / 0.63 / 0.58 (287 samples, result of record);
  stream-only sensitivity 0.55 / not scorable / 0.42 (155). Within one
  submitter: Grosz 0.78, Frisken 0.62, Miller 0.52 shuffled. The published
  forest is kept frozen for reproducibility and must be quoted with this
  caveat.
- Task 9 (California): unchanged by validation (0.867 / 0.843 / 0.649); its
  HSSR clip has no precision / medium / lab fields, so those checks are "not
  assessable". These figures follow a DEM correction (29 Sep 2026): the CA
  DEM header sat 0.02° north of its pixels, which moved the 200 m valley-floor
  label cut. Fixing it flipped 22 of 596 labels and lowered block CV from
  0.694 to 0.649 (shuffled 0.859 → 0.867, dead-zone 0.831 → 0.843).
- Integration (NE WA, magnetics excluded as synthetic): Bossburg, Oroville,
  Hunters (Hunters was #1 only on synthetic magnetics). Unchanged after the
  Kettle Falls and Meyers Creek moves (now #12 and #7). Bossburg's monazite
  call is an admitted surface sample, but its coordinate is 18.7 km from
  the MRDS Bossburg records.

**Not verified.** Site locations (8 of 12), site dimensions, bulk density,
every grade used in Task 4, and all stream chemistry against independent
samples. Register status: 4 accepted-with-caveat, 2 rejected (synthetic),
55 not verified.

## 10. Environmental studies, permitting, and social or community impact

Acid-base accounting of WGS mine-waste sites (Task 8; Big Iron, Silver Bell
and Turk flagged acid-generating). Permitting and community work not started.
Before fieldwork: confirm current rules with the land manager for every pin
(`rules_note`), obtain tribal permission for any tribal-land pin, landowner
permission on private land, and plan community contact early (CIM §2.1.4).

## 11. Interpretation of results

1. The NE WA doorbell's headline skill came from an unvalidated extract.
   After validation, Th and Au are too often censored to use, and the
   skill that remains (0.58 block) is weak and concentrated in the Grosz
   "surface" subset. The eTh–Th ρ 0.27 on detected values is the one new
   piece of supporting evidence.
2. California's local forest and walk list are not affected by the checks
   that could be run, but their extract lacks the metadata to run the same
   checks, so they are unvalidated rather than validated.
3. The NE WA priority list is a ranking of legacy inputs. It should not be
   used to choose auger sites until locations and grades are verified.

Changes in working hypotheses this phase: "0.891 is optimistic because of
spatial leakage" → "0.891 is largely a data artifact"; "Hunters is the top
REE target" → "Hunters led only on synthetic magnetics".

## 12. Conclusions and recommendations

**Objectives.** Desk-study screening objectives met; data-quality objective
(CIM validation + verification) not met for NE WA until field verification.

**Recommended program (next phase, California first):**

1. Orientation reach at walk #1 (USFS "confirm"; pans 38.5607, −120.5262
   and 38.5618, −120.5336; pour 38.6285, −120.6993) and 6–10 NURE twins with
   1-second coordinates; field protocol as written. Re-clip the 1 m lidar
   first: the existing clip is ~3.5 km north of these pans.
2. Batch through `field_qaqc.py`; publish QA/QC results in full.
3. NE WA: resolve the remaining site locations (Bossburg 18.7 km,
   Oroville 17.4 km and Sanpoil 29.6 km offsets; Brender and four others
   with no MRDS match), load a real aeromagnetic grid or keep it excluded,
   and review NURE job documentation — in particular what Grosz's
   "surface" medium was, since the validated skill rests on it. Only then
   consider NE WA twins and auger.
4. Measure bulk density and tailings footprints (lidar) before quoting
   tonnes.

**Budget and schedule.** Line items, quantities from the protocol: ~6–10
twin sites + 1–2 orientation reaches → ~20–30 primaries (sediment +
concentrate) + ~15% QC inserts + 5–10% check assays; lab (Au fire assay +
fusion ICP-MS REE/Th), CRMs and blank sand, sample shipping, one field
crew-week. Obtain lab and CRM quotes before committing a number.
Schedule: permissions and quotes → orientation → twins → QA/QC gate →
re-score. _Budget figures to be filled from quotes._

**Open decisions for the responsible geoscientist.**

- Republish the NE WA forest on validated data (7 features, API schema
  change, 0.58 block, largely a submitter signal) or keep the frozen one
  with the caveat (current choice), or retire the NE WA doorbell in favour
  of the California local product.
- Au field-duplicate threshold (placer nugget effect) — set before
  sampling. Recommendation: Au field duplicates flag but do not gate;
  blanks, CRMs and pulp duplicates gate Au at normal limits.

Resolved on review (27 Sep 2026): "surface" NURE media admitted
(stream-only kept as sensitivity); Kettle Falls = Meyers Falls Placer
`10180995` accepted; Meyers Creek moved to `10072470` (48.83 N); Brender ↔
Bender Custom Mill rejected.
