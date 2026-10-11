# Next expansions

The forest stays frozen. Earth MRI picks the **next bbox**, not a new
model. Idaho, California, Montana, Colorado, and the NC Fall Zone already
proved the doorbell does not travel (0.50 / 0.52 / 0.34 / 0.56 / 0.50).

**Two products only.** Published doorbell = frozen WA forest + transfer
table. Walkable product = CA Zr–Fe–Ti sidecar + 1 m lidar + access ranks
+ park pins. Idaho / Montana / Colorado / Fall Zone stay **tests**.
Do not sell four more walk apps.

**Do not:** national-model, retrain to hide a sag, put Task 4 tonnes on
`/predict`, add slope/eTh to the Random Forest, or treat an award press
release as a study area.

---

## Already shipped (do not redo)

| Piece | Status |
|-------|--------|
| NE WA Tasks 1–11 + GeoPackage | Done |
| Live doorbell | https://placer-lookalike.onrender.com/docs |
| Idaho / California / Montana transfer + walk lists | 0.50 / 0.52 / 0.34 |
| Local sidecars (not `/predict`) | CA block **0.65** (fixed DEM); ID dead-zone 0.67; MT 0.62; CO 0.50 |
| Colorado Wet Mountains transfer + walk list | AUC **0.56**. Local 1 expedition / 5 confirm. |
| NC Fall Zone other-placer transfer + walk list | AUC **0.50**; not a gold walk |
| Cousins test (Cr Nb Hf Sc W) | Dead-zone flat. Sidecars deleted. |
| Earth MRI watch | `EARTH_MRI_WATCH.md` |
| In-belt east-west hold-out | Task 13 — joblib not rewritten |
| Hobby / pamphlet overlay | Gazetteer + opt-in; not labels |
| 1 m LiDAR on one CA catchment | Pre-DEM-fix walk-rank 1. New rank-1 pans 38.5607, −120.5262 / 38.5618, −120.5336 are ~3.5 km south — re-clip |

---

## Wave 3 — publications that were already on the shelf (done)

| Pub | What we did | What we did not |
|-----|-------------|-----------------|
| CGS OF-23-07 Wet Mountains | `configs/colorado_wet_mtns/` Task 12 + 11 + local sidecar (dead-zone 0.50). Leadville left out. | Tasks 1–10. Doorbell stays WA. |
| Grosz B2097 / OFR 92-396 Fall Zone | `configs/fall_zone_nc/` other-placer doorbell test. | Treat sand country as gold. |
| IGS Mineral Hill / Phosphoria | Still waiting. | Did not invent a bbox from a press release. |
| CGS tungsten-waste final report | Still promised, not released. | Did not rerun CA 1–10. |
| MBMG / IGS waste OFR | Still waiting. | No fake Task 8. |

---

## Wave 4 — product, not geography (done / live deploy still pending)

- Hobby overlay is on the GeoPackages. Own pans still missing — use
  `data/hobby_reports/OPT_IN_FORM.txt`.
- 1 m 3DEP clip on **one** Sierra catchment (walk #1, USFS confirm).
  Hillshade: `california_sierra/data/lidar/lidar_r01_hs.tif`. Not all ten ranks.
  Clipped before the DEM fix; walk #1 pans moved (pour now 38.6285,
  −120.6993), so re-clip before the field day.
- API `transfer_belts` is in the sidecar on `main`. Live Render may still
  show Idaho-only until Manual Deploy. Do not add ID/CA/CO gold to
  `/predict` distance.

---

## Next (see [`SESSION_SUMMARY.md`](SESSION_SUMMARY.md))

1. **CA-only Task 11 sort — done.** `rank_by: geometry_access`. Walk #2
   (USFS, P=0.995) rose above hotter private/claimed cells.
2. Manual Deploy so live `/model-info` lists the transfer belts.
3. Walk one Sierra USFS confirm (1 m hillshade already clipped).
4. Northern Sierra chemistry from something that is not NURE HSSR.

Change-our-mind: if geometry-first colors a low-P cell, P is a
tiebreaker. If access-ok expedition ground is blank, lookalike does
not find new bars.

## Still waiting

- Mineral Hill maps / Phosphoria release
- CA tungsten-waste OFR
- Montana waste table for Task 8
- WGS Orient / Nighthawk / Part 2
- A dozen real pans on a Sierra watch-cell

---

## Out of scope until someone asks again

- A continental NURE forest.
- Putting topography or airborne eTh into the 200 trees.
- One model for every state we scored.
- Four more walk apps (ID / MT / CO / Fall Zone stay tests).
- Full Figs 1–10 on a belt with no site depths.
- Field-walking Washington from California.
- Utah as a study area (White Mesa is the mill, not a belt).
