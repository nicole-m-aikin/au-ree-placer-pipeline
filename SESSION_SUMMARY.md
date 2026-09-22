# Session wrap — 22 September 2026

Plain language. The Washington forest stays the doorbell. Nothing here
rewrites `/predict`.

## What worked

- **Montana box** shrunk (Libby out). Frozen WA forest: transfer AUC
  **0.34**, inverted mean P (0.28 near gold / 0.38 far). Walk list exists.
- **East-of-Kettle hold-out** (Task 13): published forest **0.53** on the
  east; a west-only refit is 0.63. Joblib not rewritten.
- **Local sidecars** (not `/predict`) for California (already), Idaho,
  Montana, and Colorado. Same recipe: 0.03° + 200 m valley floor.
- **Cousins test** (Cr, Nb, Hf, Sc, W on every belt): dead-zone barely
  moved. Sidecars deleted. Result kept.
- **Wave 3** boxes that already had publications:
  - Colorado Wet Mountains / Arkansas (Leadville out): transfer **0.56**,
    97% near gold. Local forest dead-zone **0.50**.
  - NC Fall Zone (Grosz Ti–Zr–REE sand): transfer **0.50**. Fairer
    negatives (27% near gold). Not a gold walk.
- **Wave 4 product:** hobby overlay already on the GPKGs; **one** Sierra
  1 m LiDAR clip (walk-rank 1, USFS, −120.78, 38.60).
- **California access + NURE hole:** PAD-US / MLRS ranks; HSSR is empty
  north of ~39° in the foothills box. South Yuba / Malakoff cannot be
  ML-ranked.

## What did not work

- **The doorbell does not travel.** WA forest on other boxes:

  | Box | AUC 0.15° |
  |-----|-----------|
  | Idaho | 0.50 |
  | California | 0.52 |
  | Montana | 0.34 |
  | Colorado | 0.56 |
  | NC Fall Zone | 0.50 |

- **Local forests only “learn” on the Sierra.** Honest 0.4° block CV:

  | Belt | Shuffled | 3 km dead-zone | 0.4° block |
  |------|----------|----------------|------------|
  | California | 0.86 | 0.83 | **0.69** |
  | Idaho | 0.73 | 0.67 | 0.63 |
  | Montana | 0.69 | 0.62 | 0.61 |
  | Colorado | 0.67 | 0.50 | 0.49 |

  Quote the block number, not 0.83, when you say California learned.
- **California’s forest does not beat Washington on other states.** It
  scores foreign mud high on both sides of the gold line (mean P ~0.7).
  That is a Sierra black-sand specialist, not a better doorbell.
- **Extra metals were noise.** Cousins ticked shuffled CV up and left
  dead-zone flat. Lithology leak.
- **More expedition cells after a local retrain is not success.** The
  trees scored their own homework.

## What stalled

- Live Render `/model-info` still may show Idaho-only. Sidecar on disk
  has the belts. Needs a rebuild / Manual Deploy. Do not add foreign
  gold pins to `/predict` distance.
- **No pub:** Mineral Hill / Phosphoria, CA tungsten-waste OFR, Montana
  waste table, WGS Part 2 / Orient.
- **Own pans** still missing. Gazetteer is not ground truth.
- **Northern Sierra chemistry** — NURE will not fill it.
- **Utah** is the mill (White Mesa), not a study area.

## What we learned (the science)

California worked because the foothills are a **placer factory**:
valley-floor mud is Zr–Fe–Ti (magnetite, ilmenite, zircon) and the
ridges are not. Idaho / Montana are gold wallpaper on one granite.
Colorado is missing Zr in HSSR and has almost no valley-floor yes-class.

The California model lighting up Colorado (flat high P) means the
**province** has the same heavies, not that the trees found the bar
(Grosz & Schruben 1993). **Topography makes the trap** (Slingerland &
Smith 1986; Yeend PP 772). Chemistry picks the creek. Geometry picks
where you stand. Do not put slope in the Random Forest.

The forest answers: *does this grab look like NURE next to known gold
placers in this box?* That is a doorbell. The field question is: *which
creek, which bar, can I walk it?*

## Locked decisions (22 Sep night)

These were open questions. They are not anymore.

| Question | Decision |
|----------|----------|
| Sort Task 11 by geometry + access, P only to break ties? | **Yes.** CA-only sort change tomorrow. Not a new model. P found the factory; elev + PAD-US + MLRS found the walk (McCuaig & Hronsky 2014; Yousefi & Carranza 2013). |
| California walk product + Washington doorbell? | **Yes.** That is the honest portfolio. Idaho / Montana / Colorado / Fall Zone stay **tests**. Do not sell four more walk apps. |
| Does lookalike ever pick a new bar? | **Unknown.** It re-finds the factory. Valley “expeditions” were FWS levees. Remaining high-P / far-from-mine cells are not `access_ok`. Another AUC will not answer it. Pans will. |
| Colorado heavies + flat CA scores = province map? | **Yes. Say it out loud.** Chemistry says “this province has the same heavies.” That is Grosz & Schruben 1993 (B2097), not a bar. |
| What would change our mind? | A dozen blanks and recoveries on one Sierra watch-cell. A Mineral Hill chemistry table that is not gold wallpaper. **Also:** if geometry-first ranking colors a cell the forest scored low, P is confirmed as a tiebreaker. If access-ok expedition ground is blank, lookalike does not find new bars. |

| Product | What it is |
|---------|------------|
| Published doorbell | Frozen WA forest + transfer table (0.50 / 0.52 / 0.34 / 0.56 / 0.50) |
| Walkable product | CA Zr–Fe–Ti sidecar + 1 m lidar + access ranks + park pins |

## Next steps

1. **CA-only Task 11 sort:** geometry + access first, P only to break ties.
   Not a new model. Do not apply that sort to the test belts.
2. **Manual Deploy** on `placer-lookalike` so `/model-info` lists the
   transfer belts. Do not change `/predict`.
3. **Walk one Sierra access-ok cell** (USFS confirm + 1 m hillshade).
   Write pans on the opt-in form. That is the first ground truth the
   forest never had.
4. **Northern Sierra chemistry** from something that is not NURE HSSR
   (CGS / NGDB / radiometrics as a *map*, not an RF feature).
5. **Wait** for Mineral Hill, CA tungsten waste, MT waste OFR. Do not
   invent those boxes.

Idaho / Montana / Colorado / Fall Zone stay transfer tests. Do not flip
them into walk products.

## Do not do

- National-model. Retrain to hide a sag. Slope or eTh in the 200 trees.
- Fill missing Zr/P/Y with Washington medians and call it a new state
  model. Fake Task 4 on a belt with no depths.
- Treat award press releases as study areas.
