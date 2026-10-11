# Field protocol — Task 11 placer verification

Small hand-tools campaign on the California Sierra Task 11 pins (USFS
`confirm` cells first; walk #1 pans ≈ 38.5607, −120.5262 and 38.5618,
−120.5336 after the 29 Sep 2026 DEM fix). NE Washington pins are a
later test. Structure follows the CIM Mineral Exploration Best Practice
Guidelines (2018); section numbers are cited inline.

The hobby opt-in form (`data/hobby_reports/OPT_IN_FORM.txt`) is not this
protocol. Its `gold_class=blank` means "no gold seen". A QA/QC **blank** here
is certified barren material inserted to catch contamination. Only samples
that follow this protocol go into `data/field_results/`.

---

## 1. Objectives

1. **Verify legacy NURE chemistry.** Re-sample NURE sites with good
   coordinates (twins) and compare Th, Ce, La, Nd, Y, Zr, Ti, Fe.
2. **Test the walk-list premise.** Does a geometry-ranked, access-ok pan
   pin in a high-P cell give more gold colours and heavier REE-mineral
   concentrate than an orientation baseline? Blank results count.
3. Build an assay dataset that passes `pipeline/field_qaqc.py` so it can
   be used downstream at all.

## 2. Campaign order

Work in this order. Do not skip ahead.

### 2.1 Orientation survey (§2.5)

Before any blind cell, sample one **known producing placer** reach with
legal public access (e.g. an `access_ok=yes` pamphlet park pin in
`task11_park_pin_access.csv`). Use exactly the media, volumes, and prep in §3.

- 3–5 sites along the reach (bar head, inside bend, bedrock crevice).
- It shows which medium carries contrast (−80 mesh sediment vs pan
  concentrate) and what a "normal" colour count looks like.
- If the orientation reach shows no Au and no heavy-mineral contrast,
  stop. Rethink the medium before spending effort on blind cells.

### 2.2 Twin sampling of legacy NURE sites

Pick twins from the NGDB/NURE records, not from the clipped CSVs in
`data/nure/`. Those drop `coord_prec_est` and round to 4 decimals. Join
on `lab_id` to the full record.

Selection rules:

1. `coord_prec_est == '1 second'`
   (`pipeline.data_validation.precision_mask(df, '1 second')`).
2. Sample medium stream/river sediment.
3. Target elements reported and not censored (Th, Ce, La at minimum).
4. Spread across the value range: pick low, middle, and high Th/Ce (or
   Task 9 P) sites. Do not pick only anomalies.
5. The channel still exists. Check recent imagery for dams, dredge
   tailings, road fill, or channel migration. Drop a site if the channel
   has moved more than the GPS error plus ~30 m.
6. Passes the land screen: `access_ok=yes`, `tribal_land=False`, and the
   `rules_note` items confirmed (§8).

Aim for 6–10 twins. Check the NURE lab report for the sieve fraction used.
If it is not −80 mesh, collect that fraction too at twin sites.

### 2.3 Walk-list cells

Then walk the ranked pins (`access_rank` order, California). Collect 2–3
sites per pan pin (trap point, plus 50–100 m upstream and downstream).

## 3. Sample types and volumes (§2.5)

Use the same medium and the same prep at every site.

| Medium (`sample_medium`) | What | Amount |
|---|---|---|
| `stream_sediment_-80` | Active-channel fine sediment, composited from 5–10 points over ~20 m, wet-sieved through a stainless −80 mesh (−177 µm) sieve | ≥ 100 g of −80 mesh (≈ 1–2 kg wet bulk) |
| `pan_concentrate_10L` | 10 L level bucket of −6 mm classified gravel from the trap point, panned to black sand | Whole concentrate, bagged |

- Pan to a consistent end point (black sand, light minerals gone). Count
  visible gold colours before bagging and record the count.
- Use no mercury, no chemicals, no motorized equipment.
- Label bags with a pre-printed sample ID tag. Never write the site name
  on the bag.

## 4. Site record (§2.5)

One record per sample, on paper and in the app. Photos are required.

- `sample_id`, `site_id`, `block_id` (Task 11 cell), `collected_by`,
  `date_collected` (and time).
- GPS in WGS84 decimal degrees, averaged ≥ 60 s. Record `gps_accuracy_m`
  as reported by the device.
- Photos: upstream, downstream, the sample spot with the tag in frame,
  and the concentrate in the pan.
- Bar position: head / mid / tail; inside or outside bend; bedrock
  crevice; active channel vs terrace or tailings.
- Channel width, flow stage, clast size and lithology, and signs of past
  mining (tailings, ditches, hydraulic pits).
- Colour count (pan), sieve fraction, and any deviation from protocol.
- Access confirmation: who gave permission, and the date (§8).

## 5. QC insertion (§2.7.4)

Sample IDs are sequential and blind. The lab must not be able to tell QC
from primaries. Assign QC positions before the trip, one of each type at a
random position in every block of 20.

| QC (`qc_type`) | Rate | Material |
|---|---|---|
| `field_duplicate` | 1 in 20 | Second sample collected at the same spot, same method |
| `blank` | 1 in 20 | Certified barren silica sand, inserted before prep |
| `crm` | 1 in 20 | Stream-sediment CRM, e.g. OREAS stream-sediment CRMs. Choose ones with certified Au, Th, Ce, La, Nd at relevant levels. Alternate two grades if possible |
| `lab_duplicate` | lab's own | Ask the lab to report its pulp duplicates |
| `check_assay` | 5–10 % of pulps | Sent to a second accredited lab, picked across the full value range |

- Enter the certified value and 1 SD from each CRM certificate into a copy
  of `data/field_results/crm_values_template.csv`. Do not type values from
  memory. Units and method family must match the results columns.
- CRMs test the lab. Field duplicates test sampling plus prep. Blanks
  test contamination in prep and analysis.
- Report all QC results, including failures (§2.7.4 full disclosure).

## 6. Lab, methods, certificates (§2.7.2)

- Use a reputable lab accredited to ISO/IEC 17025 for the specific
  methods. Confirm the scope of accreditation, not just the logo.
- Method statement (put the codes in `method`):
  - Au: fire assay (e.g. 30–50 g, AA or ICP finish). For pan concentrates,
    assay the whole sample, or use screen-metallic fire assay if gold is
    visible.
  - REE, Th, Zr, Y: lithium borate fusion with ICP-MS. Fusion is needed
    because zircon and monazite do not dissolve fully in acid digests.
  - Ti, Fe: fusion ICP or XRF. Record which.
- Record the lab's method detection limits (MDLs). Pass them to QA/QC with
  `--mdl`.
- Keep the signed certificate (PDF) for every batch. Its number goes in
  `certificate_id`. Do not use numbers before the signed certificate
  arrives.

## 7. Custody, security, splits (§2.7, §2.7.1, §2.7.3)

- **Chain of custody.** Fill a dispatch list and custody form at each
  handover (collector → storage → courier → lab), signed and dated. Check
  the lab's receipt list against the dispatch list and log any
  discrepancy.
- **Security.** Use numbered, tamper-evident bag seals. Keep samples in a
  locked vehicle or locked storage, never left unattended at a trailhead.
  Ship by tracked courier.
- **Splitting.** Riffle-split dried −80 mesh material; do not scoop. The
  lab splits pulps with a rotary or riffle splitter (ask).
- **Retained fractions.** Keep a field split of every sediment sample.
  Have the lab return pulps and coarse rejects. Store them labelled for
  the life of the project. Pan concentrates are assayed whole; the photo
  and colour count are the retained record.

## 8. Permits, access, safety, community (§2.1.2–2.1.4)

Confirm everything before the trip. Every item is "confirm current rules
with the agency". This list is a reminder, not legal advice.

- [ ] Task 11 land screen: the pin has `access_ok=yes`, `tribal_land=False`,
      and each `rules_note` item is checked off.
- [ ] Mining claims: re-check BLM MLRS the week of the trip and look for
      posts on site. "Claim-free" in the screen is not permission.
- [ ] USFS: confirm casual use vs Notice of Intent with the district
      ranger for the planned activity (hand panning, sieving, small holes).
- [ ] BLM: confirm casual use vs Notice with the field office.
- [ ] California: confirm the current motorized suction-dredge rules and
      any streambed-alteration requirements with the state agencies. We
      use hand tools only; confirm they are allowed at the site.
- [ ] State or local parks: confirm panning rules with park staff.
- [ ] Washington (later): follow the current WDFW Gold and Fish pamphlet.
      Confirm whether an HPA is needed for the planned work.
- [ ] Tribal land requires tribal permission. The Census AIANNH screen
      marks reservation and trust land `access_type=tribal`. In NE WA the
      Colville Reservation covers the Sanpoil River. Contact the tribe
      before planning any work there. Outside reservations, ask about
      cultural sites and treaty-reserved uses. Engage early, not after
      results.
- [ ] Private land: written landowner permission.
- [ ] Cultural resources: do not disturb artifacts or historic mining
      features.
- [ ] Safety: work with a partner, leave a check-in plan, carry a
      satellite messenger. Watch snowmelt flows and cold water. Stay out of
      adits and pits. Legacy Sierra placers can hold mercury: wear gloves
      with concentrates and never heat them. Carry a first-aid kit; check
      for heat, ticks, and poison oak.
- [ ] Leave no trace: backfill holes, don't dig banks or roots, stay
      within casual-use limits.

## 9. pXRF (§2.7.2)

A pXRF is for screening only. Its results never go into
`data/field_results/` and are never reported as assays. If pXRF numbers
appear anywhere, disclose:

- instrument model and serial number, mode, calibration, count time;
- sample state (in situ, dried, sieved, in a cup);
- CRM and blank readings from the same session;
- limitations: pXRF does not measure Au at placer grades, and light-REE
  detection limits are high and matrix-dependent.

## 10. Data entry and acceptance

1. Enter one row per sample in a copy of
   `data/field_results/field_results_template.csv`. Keep assay strings as
   the lab reports them (`<0.005` is fine).
2. Run:

   ```bash
   venv/bin/python -m pipeline.field_qaqc \
     --results data/field_results/<campaign>.csv \
     --crm data/field_results/<crm_values>.csv \
     --mdl 'Au_ppm=<lab MDL>,Ce_ppm=<lab MDL>' \
     --out data/field_results/qaqc_<campaign>
   ```

3. Only `field_accepted_primaries.csv` (primaries from `pass` batches) may
   be used downstream. QC rows never leave the QA/QC step.

Acceptance criteria (defaults in `pipeline/field_qaqc.py`):

| Check | Pass | Warn | Fail |
|---|---|---|---|
| Blank (Au, Th, Ce, La, Nd, Pr, Y) | < detection, or ≤ 5× MDL and ≤ any set limit | detected, no MDL or limit to judge | > 5× MDL or > set limit |
| CRM (each certified element) | \|z\| ≤ 2 | 2 < \|z\| ≤ 3; CRM not in table; no SD | \|z\| > 3 |
| Field duplicate, HARD | ≤ 30 % | any pair over, ≤ 10 % of pairs | > 10 % of assessed pairs over |
| Lab duplicate / check assay, HARD | ≤ 20 % | as above | as above |
| Insertion (blank, CRM, field dup) | ≥ 1 per 25 primaries | < 1 per 25 | none of a type in the batch |
| Metadata (primaries) | lon/lat, lab, method, certificate present | any missing | — |
| Check assays (project) | ≥ 5 % of primaries | < 5 % (reported, does not gate a batch) | — |

HARD = 100·|a−b|/(a+b). Pairs with a censored value, or a mean below
10× MDL, are skipped. Duplicate and check-assay pairs count against the
parent primary's batch.

- **Warn batch:** investigate, document the cause, re-assay if needed.
  Primaries are held back unless someone accepts it in writing
  (`--allow-warn`) and records why.
- **Fail batch:** re-assay from pulps, or re-sample. Never edit values or
  drop QC rows to make a batch pass.
- **Au nugget effect:** Au field duplicates often exceed 30 % in placers.
  If you want a different Au threshold (`field_max_hard={'Au_ppm': …}`),
  set it and write it down **before** sampling, not after seeing results.

## 11. Records to keep

Field sheets and photos, GPS tracks, custody forms, dispatch and receipt
lists, signed certificates, CRM certificates, the results CSV,
`field_qaqc_checks.csv`, `field_qaqc_batches.csv`, and a log of every
re-assay and deviation.
