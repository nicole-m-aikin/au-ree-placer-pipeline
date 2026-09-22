# Interview notes — deploying the Au+REE targeting model

Phase-by-phase decisions, tradeoffs, and answers I would actually give.

---

## Phase 0 — What is actually in the repo

**Decision:** Do not invent a saved model or pretend Monte Carlo already sits on the classifier. The API has to persist the Task 9 Random Forest and add a *new* per-prediction uncertainty path. Task 4 Monte Carlo stays where it is: endowment P10/P50/P90, not `P(anomalous)`.

**Tradeoff:** Shipping `/predict` with a confidence interval means extra work (train-and-export + a prediction-level interval). Pretending Task 4 MC is model uncertainty would be faster and wrong. An interviewer who has read the README will catch that in one question.

### What I found

- Trained model lives only inside `pipeline/task9_ml_targeting.py` `run()`. `rf_final` is fit, used for `predict_proba`, then discarded. No `.pkl`, `.joblib`, ONNX, or metadata file anywhere. `.gitignore` does not even mention model artifacts.
- Feature schema for the published `placer_gold` model is hardcoded: `Th, Ce, La, P, U, Au, As, Ti, Fe, Zr, Y`. Log10-transformed; NaN / non-positive filled with the *batch* log-median. Those medians are never written out.
- Monte Carlo is in `pipeline/task4_volume.py` (2,000 draws: depth ~ Normal, grade ~ LogNormal σ=0.4). It produces NdPr tonnes P10/P50/P90. Task 5 NPV is a price/cost sensitivity tornado, not a second MC on the RF.
- Evidence the model existed: `task9_ml_cv_scores.csv` (mean ROC-AUC 0.891), `task9_ml_feature_importance.csv` (U first at 0.233), `task9_ml_nure_probability.csv` (per-sample point probabilities). Those are scores, not a loadable estimator.
- No FastAPI/Flask, no Dockerfile, no notebooks. CI already runs `pytest tests/` on push/PR to main.

### Missing before any API is honest

1. A versioned, loadable model plus the exact preprocessing stats used at train time.
2. A prediction-level interval. Tree vote fraction is not the Task 4 endowment MC.
3. Feature units and bounds for the request schema (NURE half-MDL, P wt%→ppm, Fe left in wt%).
4. Training date / sklearn pin — pickle is version-sensitive and nothing records when `rf_final` was fit.

### Likely questions

**Q: You say the model has ROC-AUC 0.891. Where is the model file?**

The 0.891 is 5-fold stratified CV written to `task9_ml_cv_scores.csv`. The final forest is fit on all 1,045 labelled NURE rows inside Task 9 and never serialized. The pipeline is a batch figure factory, not a service, so there was nothing to load until we decided to deploy.

**Q: Isn't the Monte Carlo already the uncertainty you would expose on `/predict`?**

No. Task 4 MC is depth × grade → NdPr tonnes at a named placer. Task 9 outputs `P(anomalous)` from chemistry. Different random variables, different units. Putting P10/P90 tonnes on a chemistry `/predict` would be mixing products.

**Q: How do you score a single sample the same way Task 9 scored the batch?**

You cannot replay `_log_impute` on one row and get the same fill values. Training medians come from the full log-feature matrix. A one-row request with a missing Au would fill 0.0 instead of the training median. So the export has to include those medians, or the API will silently drift from the CV number.

---

## Phase 1 — Design (implemented; live doorbell below)

**Decision 1 — freeze artifacts in Task 9, not in the API.** After `rf_final.fit`, Task 9 writes three things: the joblib estimator, a metadata JSON (feature order, training date, per-fold and mean CV AUC, label config, sklearn version), and the per-feature log-medians from `_log_impute`. The API loads those files. It does not recompute medians.

**Why this is a pipeline problem:** Train/serve skew is "did we score the request the same way we scored the training rows?" That is decided at persist time. If the API invented its own fill values, an interviewer asking about preprocessing consistency would correctly hear "I patched serving." The fill values belong next to the fit.

**Tradeoff:** One extra Task 9 run after the Fe fix, and a sklearn pin. We do not get to keep an unpinned `scikit-learn>=1.3` — unpickling a forest across sklearn versions is a known silent-break.

**Alternative I am not using:** ONNX export. Portable, and it would dodge the sklearn pin. It also adds a converter, a runtime, and a second artifact to explain, for a 200-tree, 11-feature forest that will live on a free-tier box. joblib + a pinned sklearn is the honest minimum.

**Decision 2 — tree-vote spread, not Monte Carlo.** `/predict` returns `probability` = `predict_proba` (mean vote over 200 trees) and `tree_vote_spread` = standard deviation of `predict_proba[:, 1]` across `rf.estimators_`. The field name and the schema description say "tree-vote spread." They do not say Monte Carlo.

**Tradeoff:** This is the disagreement among trees already in the fitted forest. It is cheap, needs no training data in the container, and is honest about what it is. It is *not* a calibrated 95% interval and it is *not* Task 4.

**Alternatives I am not using:**
- Wrapping Task 4's 2,000 depth/grade draws — wrong random variable.
- Bootstrap / real MC on the training set — needs the NURE matrix in the API, slow on free tier.
- Conformal prediction — needs a held-out calibration set we did not persist.
- Calling `predict_proba` itself "uncertainty" — that number is the point estimate.

**Decision 3 — Fe is converted in `load_nure`, and the API still refuses a unitless Fe.** NURE reports Fe as wt% (values ~2–5). `load_nure` already converts P and Ca with the median-threshold heuristic and skipped Fe. That is the inconsistency. Add Fe to `_wt_ppm` with threshold 100 (same idea as Ca). After conversion the model sees Fe in ppm, matching the other ten features.

Trees do not actually care: `log10(Fe_wt% × 10,000) = log10(Fe_wt%) + 4`. A uniform shift on one column does not change RF splits. I still convert, because a later reader (or an API client) will send ppm and get a garbage score if the forest was fit on wt%. The API requires `fe_unit: "wt_pct" | "ppm"` and converts `wt_pct` the same way. A `ppm` value below 100 is rejected as unconverted wt%.

**Decision 4 — FastAPI.** Eleven chemistry fields with real unit and MDL constraints. A silently accepted Fe in the wrong unit is a wrong target, not a 400 vs 500 style debate. FastAPI/Pydantic fail that request before it hits the forest. The lifespan hook loads the joblib + JSON once. Flask would work; we would hand-roll the validation and the startup load, which is where this project actually breaks.

### Likely questions

**Q: How do you keep preprocessing consistent between training and serving?**

I don't reimplement it in the API. Task 9 writes the feature order and the log-medians that `_log_impute` actually used. Serve-time `_log_impute` takes those medians as an argument. If a client omits Au, we fill the training Au log-median, not the median of a one-row request (which is 0). That is pipeline correctness. The API is just the consumer.

**Q: Why isn't Task 4's Monte Carlo on this endpoint?**

Task 4 draws depth and grade and reports NdPr tonnes P10/P50/P90 for a named placer. `/predict` takes stream-sediment chemistry and returns `P(anomalous)`. Those are different quantities. I put that sentence on `/model-info` so nobody can misread this the way the README mixed them. The uncertainty I do return is tree-vote spread: std of the 200 tree probabilities. Cheap, already in the fitted forest, and named for what it is.

**Q: Why pin sklearn instead of exporting ONNX?**

sklearn's own docs say do not unpickle an estimator with a different sklearn than you trained on. I write `sklearn.__version__` into the metadata, pin that exact version in `requirements.txt`, and `/health` 503s on mismatch. ONNX would remove that pin. It is the right next step if this leaves a single-version container. It is extra machinery I do not need to ship a 200-tree forest.

---

## Phase 2 — What actually shipped

Implemented as designed. No silent changes.

- `load_nure` now converts Fe wt% → ppm (median-threshold 100, same heuristic as Ca). After re-running Task 9 `--export-only`: **CV ROC-AUC still 0.891 ± 0.018**, U still 0.233. The log10(+4) invariance held on this extract.
- Frozen artifacts: `models/task9_rf_placer_gold.joblib` + `.meta.json`. Fe log-median is 4.49 (`10^4.49 ≈ 31,000 ppm ≈ 3.1 wt%`), which is the converted unit, not the raw NURE percent.
- sklearn pin is **1.6.1** — the version that trained this forest. `/health` 503s if it drifts.
- FastAPI on `api.app:app`. Local check: `/health` 200, `/model-info` carries the Task 4 disclaimer, `/predict` on the first NURE-like row returned `label=1`, `probability=0.535`, `tree_vote_spread=0.499` (200 trees). Borderline probability, wide tree disagreement — that is the point of exposing spread. Malformed `Th="hot"` and `Fe=2.7` with `fe_unit=ppm` both 422.

**How I start it:** `uvicorn api.app:app --host 127.0.0.1 --port 8000` from the repo root, venv on.

**Q: You converted Fe and the AUC did not move. Did the model change?**

The splits did not. Every Fe value shifted by the same +4 in log space, so Gini and AUC are identical. What changed is the contract: serve-time Fe is ppm, and the frozen median is 4.49 not ~0.49. If I had left Fe as wt% and a client sent 30,000 ppm, that one column would dominate the vote. The conversion is for the next person, not for this CSV's ROC.

---

## Literature (does this already exist?)

Full write-up: `LIT_REVIEW.md` (21 Sep 2026).

**Decision:** Do not claim we invented prospectivity mapping. Claim the missing public object: a frozen forest you can POST to that answers “does this grab look like ground next to known **gold** placers?”, with units and tree-vote spread. Not a monazite/REE detector.

**What already exists:** USGS Grosz (1993) used NURE to outline *heavy-mineral* placer provinces (Ti–Zr–REE). Carranza & Laborte (2015–16) made RF + known-gold-deposit labels the default (mostly lode). Government APIs serve NURE/MRDS *tables*. Alluvial-gold ML papers are still rare (Colombia 2025, boreholes, not NURE).

**What does not exist in the open:** POST chemistry → P(gold-placer lookalike) for US stream sediment, plus a field loop that books expeditions.

**Q: Is this a REE tool?**

No. The published model is labelled on MRDS gold. REE-bearing minerals are in the feature list because they sort with gold in the same trap. Tonnes of NdPr are a separate screen in Tasks 4–5. If I only wanted monazite I would have trained on the seven REE MRDS sites, which is too few and is not what we shipped.

**Q: Doesn’t USGS already do this?**

They published NURE as a heavy-mineral *province* screen in the 1990s and they serve the *tables* today. They do not serve a probability that a new grab looks like a known gold placer. Papers stop at a map and an AUC.

**Q: Then why is 0.891 not enough to send a crew?**

Airola et al. (2018): ordinary cross-validation on spatial mineral data can look perfect and fail on new ground. Yousefi & Carranza (2013): the unit you walk is a catchment, not a point. McCuaig & Hronsky (2014): chemistry without a trap is not a target. Field pans are the paper that is still missing.

---

## Robustness pass (spatial CV + honest API text)

Implemented. The forest did not change. How we *test* it did.

**Results on NE WA (same 1,045 grabs, 290 yes):**

| Test | AUC |
|---|---|
| Shuffled 5-fold (the old 0.891) | **0.891 ± 0.018** |
| Dead-zone: drop train samples within 0.15° of a test grab | **0.699 ± 0.055** |
| Hold out 0.4° map cells | **0.757 ± 0.139** |

Dead-zone drops ~768 training neighbors per fold. That is the leak. 0.70 is still better than a coin flip (0.50). It is not 0.89. I report both.

`/model-info` now states: gold-placer lookalike (not REE); a 0 is “not near a mapped gold mine,” not barren; 0.891 can be optimistic.

`/predict` accepts optional lon/lat. Same chemistry at Hunters vs the SW corner still scores **0.535** — the forest does not use location. The flag changes: Hunters `near_known_gold` (0.037° from a training mine); SW corner `far_from_known_gold` (0.48°). High score + far is the one a human has to look at.

**Q: So is 0.891 a lie?**

It is the right number for “can the forest separate yes/no when neighbors of the same mine are allowed in training.” It is the wrong number for “will this work on a drainage we did not train on.” 0.70 is that number. I keep both so nobody can accuse me of hiding the drop.

---

## Catchment walk list (Task 11)

The 0.4° cells are now a field product, not just a CV footnote.

**What shipped:** every cell on the map, one pour point per occupied cell (most-downstream high-P grab, snapped to the 30 m DEM), NURE chemistry stars, and pan pins on the D8 creek (slope break / power drop / knickpoint foot / tributary mouth). Walk order is isolation first: high P and farthest from a gold pin. `field_spots` was the old name for the stars; they are `nure_spots` now. They are not pan locations.

**NE WA numbers:** 45 cells, 38 with NURE. 10 expedition (>5 km from a gold pin), 18 confirm (≤5 km), 17 skip. At the 15 km *training* radius this whole belt is “near a mine,” so the walk list uses a 5 km field radius or it would be empty.

Leave-one-cell-out AUC (cells with ≥8 grabs): **0.56**, range 0.25–1.00, 11 cells scored. Cell #7 transfers (0.93). Cell #5 does not (0.34). That is the 0.94 vs 0.62 story on a map.

**Not a gold claim.** `#1` is 15 km from a pin with P=0.94. The star is the hot NURE grab. The circles are where geometry says stand with a pan — upstream of that star, which is the right order for a stream-sediment hit.

Outputs: `fig11_catchment_walk_list_map.png`, `task11_catchment_walk_list.csv`, `task11_nure_spots.csv`, `task11_pan_locations.csv`.

Idaho and California now have the same layers. P is the Washington forest. Gold pins and SGMC geology are local. California has no expedition cell.

---

## Public doorbell + second belt

**Live:** https://placer-lookalike.onrender.com/docs

**Decision:** Ship `/predict` as a frozen-forest doorbell. Score Idaho, then California, with that same joblib. Do not retrain. Do not national-model.

**Idaho result:** 2,465 NURE grabs, transfer AUC **0.50**. Mean P ≈ 0.38 next to gold and far from it. Tightening the circle to 5 km does not help.

**California result:** Northern Sierra foothills (Feather / Yuba / American — the belt you can walk). 596 NURE grabs, 7,872 gold MRDS pins, transfer AUC **0.52**. Mean P is 0.38 next to gold and far from it. Tightening to 5 km lifts AUC to 0.61; the doorbell still does not know California. This clip has no Au or As.

**Walk lists:** Idaho GeoPackage — 2 expedition / 4 confirm / 16 pans. California GeoPackage — 0 expedition / 0 confirm / 8 watch / 10 pans. P is Washington lookalike. Gold pins and geology are local. Open `~/projects/task11_ca_sierra_placer_field_campaign.gpkg` (no `+` in the path).

**Q: What does the API actually do?**

You POST eleven NURE concentrations and `fe_unit`. You get P(this grab looks like chemistry next to known gold placers), how much the 200 trees disagree, and — if you sent lon/lat — km to the nearest *training* gold pin. High P + far from any training mine is the interesting / untrusted case. It is not a claim and it is not Task 4 tonnes.

**Q: Why Idaho if you expect the AUC to drop?**

Because 0.891 is shuffled CV on one belt. Airola 2018: that number can look great and fail on new ground. The literature says score the next belt with the first forest *before* you refit. A sag is the result. Retraining Idaho or California would be a new model, not proof the doorbell travels. Living in California does not make the Washington forest a Sierra model.

**Q: Why Render / Docker?**

So someone can hit `/docs` without cloning 7 GB of LiDAR. The container is sklearn + the model files. The expedition GeoPackage stays on disk. Free-tier cold start is slow; that is the host waking up, not the forest thinking.
