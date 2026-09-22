# Next expansions

The forest stays frozen. Earth MRI picks the **next bbox**, not a new
model. Idaho, California, and Montana already proved the doorbell does
not travel (AUC 0.50 / 0.52 / 0.34). The next work is a publication
upgrade, a tighter Washington product, or a walk someone can actually do.

**Do not:** national-model, retrain to hide a sag, put Task 4 tonnes on
`/predict`, add slope/eTh to the Random Forest, or treat an award press
release as a study area.

---

## Already shipped (do not redo)

| Piece | Status |
|-------|--------|
| NE WA Tasks 1–11 + GeoPackage | Done |
| Live doorbell | https://placer-lookalike.onrender.com/docs |
| Idaho transfer + walk list | AUC 0.50; 2 expedition |
| California Sierra transfer + walk list | AUC 0.52; **0 expedition** |
| Montana SW gulches transfer + walk list | AUC **0.34**; Libby dropped |
| Earth MRI watch | `EARTH_MRI_WATCH.md` |
| In-belt east-west hold-out | `pipeline/task13_holdout.py` — does not rewrite the joblib |

---

## Wave 0 — tidy (done)

1. `EARTH_MRI_WATCH.md` + this file committed with the rest of the wave.
2. Render redeploys from `main` so `/model-info` lists Idaho, California,
   and Montana on `transfer_belts`.
3. Montana bbox is Confederate / Alder / Montana Bar / Elkhorn
   (`-112.85 to -111.15, 45.20 to 46.80`). **Libby is out.**

---

## Wave 1 — Montana (done)

Same recipe as Idaho / California. No Tasks 1–10 (no public depths, no
WGS-style waste OFR).

| Step | Result |
|------|--------|
| Clip NURE + gold MRDS. 30 m DEM. SGMC `MT`. | 4,672 grabs; 2,039 gold pins |
| Task 12 | AUC **0.34** (0.05° = 0.33). Mean P 0.28 near gold / 0.38 far. P and Y filled with WA medians. |
| Task 11 | Walk list + GeoPackage. Local gold pins. `p_is_transfer: true`. |
| Tasks 1–10 | Off. |

---

## Wave 2 — Washington in-belt (hold-out shipped; maps wait)

- Optional hold-out east vs west of the Kettle (`holdout.lon_cut: -118.50`)
  is `python -m pipeline.task13_holdout`. Report sits next to the 0.70
  spatial CV. The published joblib is not rewritten.
- When WGS ships Orient / Nighthawk maps or OFR 2026-02 Part 2: rebuild
  the NE WA walk list / site ranks. Do not change the forest.

This tightens `#1`–`#10`. It is not “the next location.”

---

## Wave 3 — triggered by a publication (do not jump the queue)

| Pub lands | Do this |
|-----------|---------|
| IGS Mineral Hill maps or Phosphoria / Western Phosphate release | Decide: widen `idaho_batholith` or add `idaho_mineral_hill`. Task 12 first. Carbonatite ≠ gold walk unless the chemistry says so. |
| CGS tungsten-waste OFR or eastern-Sierra geochem **inside** the foothills box | Feed Task 8 / geology on the existing CA GPKG. Do not rerun 1–10. Mountain Pass / Salton Sea stay out. |
| VA / NC / SC Fall Zone placer maps (Grosz belt) | New config. Honest *other placer* (Ti–Zr–REE sand), still frozen gold forest as a doorbell test. Say that in the summary. |
| CGS / MBMG / IGS mine-waste OFR | Overlay like WGS 2026-02. Never a label for Task 9. Montana waste table → Task 8 on the existing gulch box. |

---

## Wave 4 — product, not geography

These do not need a new state:

- **Hobby / field pans** as a scored overlay, not training labels.
  California is the belt you can walk. A dozen pans on one Sierra
  watch-cell would be the first ground truth the forest never had.
- **1 m LiDAR on one CA catchment** (not all ten ranks). Geometry
  pans at 30 m are system-scale. Bar heads need lidar or a boot.
- **API**: `transfer_belts` lists Idaho, California, and Montana after
  Render picks up `main`. Do not add those gold pins to `/predict`
  distance — that flag is *training* mines on purpose.

---

## Out of scope until someone asks again

- A continental NURE forest.
- Putting topography or airborne eTh into the 200 trees.
- Montana + Idaho + California as one model.
- Full Figs 1–10 on a belt with no site depths / no waste table
  (fake Task 4 is worse than a missing figure).
- Field-walking Washington from California.

---

## Decision rule

```
new Earth MRI map or waste OFR
    → is there NURE + a gold/placer story in a ~2° box?
        yes → Task 12 (frozen) → Task 11 (walk list)
            → Tasks 1–10 only if sites are real
        no  → note it on EARTH_MRI_WATCH.md and wait
```

Default next keystroke: **wait for a Wave 3 publication**, or walk a
California watch-cell. Do not open Fall Zone / Colorado until a map
is on the shelf.
