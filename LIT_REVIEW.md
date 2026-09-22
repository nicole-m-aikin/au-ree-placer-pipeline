# Literature review — does a “score this creek, find a new placer” system already exist?

**Searched:** 21 September 2026  
**Question:** Is there already a live API that takes stream-sediment chemistry and returns a placer target? What published work would support building one?

**What this model is for:** **gold placers** (and the heavy-mineral mix that rides with them). It is not a monazite/REE-only tool. The published Task 9 forest is labelled on MRDS **gold** sites. Th, Ce, La, P, U, Ti, Fe, Zr, Y are in the feature list because those minerals hydraulically sort with gold — not because `/predict` is a REE product. REE/monazite tonnes live in the rest of the pipeline (Tasks 3–5), not in the API.

**Short answer:** The *science* is old and well supported. The *product* (a public predict endpoint that turns a NURE-style grab into a field expedition) does not exist as an open, documented service. USGS and others serve **data and maps**. Companies sell **closed targeting**. Papers publish **one-off maps** for one district. Nobody I found ships “POST 11 elements → P(lookalike gold placer) + uncertainty” on public US stream sediment.

---

## 1. What already exists (close, but not this)

### Government: data and maps, not a scorer

| What | What it actually is | Not |
|---|---|---|
| USGS NURE-HSSR | ~398,000 sediment analyses; download + WFS/WMS + bbox search | A model |
| USGS MRDS | Mine/occurrence inventory; bbox and name search APIs | Labels you can POST chemistry against |
| USGS Earth MRI | Focus-area polygons and new geochemistry for critical minerals | A predict API |
| USGS NURE atlas (OFR 98-622) | National maps of Th, U, Ti, Fe, Ce, As, etc. | Targeting |
| Geoscience Australia mineral potential WMTS | **Map tiles** of mineral-system potential (base metals, carbonatite REE) | Score-this-sample |

USGS *does* have web APIs. They return **records and layers** (`mrdata.usgs.gov` NURE/MRDS services). They do not return a probability that *your* grab is a placer lookalike.

### USGS already used NURE to find placer *provinces* — in the 1990s, without ML

This is the closest ancestor of **this** repo’s chemistry idea.

- **Grosz & Schruben, USGS Bulletin 2097 (1993).** Ti, Zr/Hf, REE, Th, U in NURE outline known heavy-mineral placers and new belts (Atlantic Coastal Plain, Idaho batholith, parts of Montana, WA/OR/WY). Ce–La, Ce–Th, Th–U hang together; airborne eTh can outline monazite provinces.
- **Grosz, USGS OFR 93-240-A (1993).** Same story with maps: NURE Ti / Hf-Zr / REE / Th / U as diagnostics for ilmenite, rutile, zircon, monazite, xenotime.
- **Grosz & others, OFR 92-396; Bern & others 2016 (Coastal Plain).** eTh as a proxy for monazite/xenotime; still needs ground truth. Radiometric highs are shallow and not unique.

So: “use NURE heavy-mineral chemistry to find placer country” is **USGS doctrine**, not a new trick. Grosz was hunting Ti–Zr–REE sands. This project uses the same *style* of chemistry (heavy minerals travel together) to ask a **gold** question: does this grab look like the ones next to known gold placers? What USGS did not do is train a Random Forest on MRDS gold proximity, freeze the fill-values, and serve it.

### Academic mineral prospectivity mapping (MPM) — the standard recipe

For ~15 years the recipe has been:

1. Known deposits = positive labels (often a buffer / “near a mine”).
2. Far-from-mine cells = negatives (this choice is messy; see warnings).
3. Evidence layers: geology, faults, magnetics, stream sediment, alteration.
4. Weights-of-evidence, then SVM, then **Random Forest**.
5. A prospectivity **map**, ROC/AUC, paper ends.

Key papers:

- **Carranza & Laborte (2015)** *Ore Geology Reviews* — RF gold prospectivity, Baguio. RF beat or matched weights-of-evidence / logistic / evidential belief. Sensitivity to which deposit/non-deposit points you pick.
- **Carranza & Laborte (2015, 2016)** Abra and Catanduanes — RF still works with **<20** known prospects and missing values. That is the usual citation for “RF is the right tool when you have few mines.”
- **Rodríguez-Galiano et al. (2015)** *Ore Geology Reviews* 71:804–818 — head-to-head: neural nets, regression trees, RF, SVM on epithermal Au (Rodalquilar). **RF won** on stability, ROC, and not exploding when you change training size.
- **Zuo & Carranza (2011)** — SVM for MPM (older sibling of the RF wave).
- **Yousefi, Carranza & Kamkar-Rouhani (2013)** *J. Geochem. Explor.* 128:88–96 — **drainage catchment basins**, not just sample points or contours, as the right spatial unit for stream sediment. Weighted catchments beat sample-catchment-basin and contour maps against known deposits.

Recent stream-sediment + RF gold work (lode / pathfinder flavor, not US placers):

- **Yuanbo Nang, Gansu (Minerals 2024, 14:500).** RF regression and classification on 1,859 stream sediments; Au linked to As, Sb, Hg; SHAP/PDP/ALE; known deposits as the classification target.
- **Tanzania craton (Phys. Chem. Earth 2024).** 166 stream sediments; RF beat SVM and ANN; As–Ni–W as gold pathfinders.
- **Northern Yangshan (2025).** 4,870 stream sediments + **30 m DEM catchments** + RF (200 trees) for weak Au–As–Sb anomalies.

### Alluvial / placer ML — thin, and not NURE

- **Cauca River, Colombia (Pure Appl. Geophys. 2025).** Explicitly says ML for **alluvial gold is still rare**. They used boreholes and lithology / DEM paleochannels, not a national stream-sediment lookalike model. Hybrid nets beat a simple interpolator.

So “ML for placers” exists. It is not “score US NURE against MRDS gold and serve it.”

### Companies — closed, not a public chemistry API

- **KoBold Metals** — internal models + their own sensors. No public predict API.
- **Kenex** — consultancy; Random Forest mineral-potential maps as a **service**, not a sandbox URL. Example: porphyry Cu–Au, Lachlan.
- **EarthScience.AI** — marketed platform (MRDS mentioned, “uncertainty,” API on expensive tiers). Closed, vendor claims, not a paper you can reproduce.
- **Geoscience Australia / BCGS / Kenex-style government maps** — you view a layer. You do not POST a sample.

---

## 2. The gap (what this repo is, relative to the literature)

| Piece | In the literature? | In this repo? |
|---|---|---|
| NURE as a placer-province screen | Yes — Grosz 1993 | Yes |
| RF + known-deposit labels | Yes — Carranza school | Yes (MRDS gold, 0.15°, valley floor) |
| Stream sediment as evidence | Yes — many papers | Yes (the *only* model input) |
| Heavy-mineral suite (Th, Ce, La, P, U, Au, As, Ti, Fe, Zr, Y) | Grosz listed the elements; few RF papers use that exact placer suite (most gold papers use As–Sb–Hg) | Yes, on purpose |
| Do **not** label from Th itself (anti-circular) | Implied by “deposits as positives”; rarely stated this bluntly | Stated and tested |
| Catchment as the right unit | Yousefi & Carranza 2013; Yangshan 2025 | Implemented as QA, **not** the published model (too few positives / tautology on 12 sites) |
| Mineral systems (source–path–trap) | Wyborn 1994; McCuaig & Hronsky 2014; GA, BCGS | The rest of the pipeline, not `/predict` |
| Spatial cross-validation | Strong warning: ordinary k-fold AUC can be fake | We used stratified k-fold. **Gap.** METHODOLOGY already names this. |
| Public predict API + frozen train medians + tree-vote spread | Not found | What we just built |
| Field-feedback loop that books expeditions | Not in papers; KoBold-like shops do it in-house | Overlay built: `hobby_reports` gazetteer + opt-in form, catchment hit-rate (not AUC). Own pans still missing. |

**Bottom line for a hiring manager:** you did not invent mineral prospectivity. You took the Carranza RF + known-gold-deposit recipe, used NURE chemistry (heavy-mineral suite + Au–As, not As–Sb-only), refused circular Th labels, and put a defensible serve path on it. The target is **gold-placer lookalike drainages**. Monazite/REE is a co-product question in the ranking/tonnage tasks, not the API’s job. The missing published object is the serve path, plus (still) spatial CV and a field loop.

---

## 3. Literature that supports *parts* of building the expedition version

### A. Why stream sediment can see a placer at all

- **Bonham-Carter et al. (1988).** Integrating geology for gold exploration; catchment dilution idea. This is why Task 4 uses a stream-to-in-situ factor, not raw ppm as grade.
- **Ahrens (1954); Reimann & Filzmoser (2000); Stanley & Sinclair (1989).** Trace elements are log-normal. Log10 before a forest is geology, not a sklearn trick.
- **Grosz 1993 / B2097.** The element list for heavy-mineral placers *is* NURE Ti, Zr, REE, Th, U. Our 11-element set is that list plus Au–As (gold pathfinder) and Fe (magnetite chemistry — different from the magnetic *map*).

### B. Why Random Forest, why deposit-proximity labels

- Carranza & Laborte 2015–2016; Rodríguez-Galiano et al. 2015; later RF-vs-SVM papers (e.g. *Minerals* 2023, 13:1073): RF is the default when you have few mines, mixed units, and you want Gini/importance without a tuning circus.
- Positives = known occurrences is the field standard. Our twist: label the **NURE row** (is this grab near a mine?) rather than a grid cell. Same idea, different object.
- **Carranza et al. (2008) and later NRR papers (2025)** on negative labels: “not near a mine” is not the same as “barren.” Random negatives inject uncertainty (Zuo & Wang, Goldschmidt 2020). We should not pretend our 0-class is proven waste ground.

### C. Why a magnetic high ≠ stream Fe

- Magnetite as a **geophysical** co-placer proxy is standard (aeromag). Grosz used **radiometrics** (eTh) for monazite, not stream Fe as magnetite assay. Supports our split: `mag_high` from the map; Fe in the forest is only chemistry.

### D. Why catchments before you send a crew

- **Yousefi & Carranza (2013)** and the Yangshan 2025 paper: the decision unit is the **drainage**, not the sample point or an IDW blob. That is exactly the “scale to expedition” step we named earlier.
- **McCuaig & Hronsky (2014); Wyborn et al. (1994).** A high chemistry score with no trap / no source is not a target. The integration score in this repo is that idea.

### E. Why you must not trust 0.891 as “will work in Idaho”

- **Airola et al. (2018)** *Data Min. Knowl. Disc.* Spatial leave-pair-out CV on orogenic gold MPM: ordinary CV can give AUC near **1.0** for a model that is barely better than chance on new ground. Neighbors of mines leak into the test fold.
- **2026 Ore Geology Reviews** (Ni sulphides, Canada): random CV vs blocked CV vs leave-one-cluster-out. Scores drop as the test gets more honest. Transfer across belts is the real test.
- Our METHODOLOGY already says block spatial CV is the production upgrade. The literature says that is not optional if you want “new placer,” only optional if you want a regional lookalike screen.

### F. Why an API is not in the papers

Academic MPM ends at a GeoTIFF and an AUC. Government MPM ends at a WMTS. Industry MPM ends at a PDF for a client or an internal stack (KoBold). A small, honest predict service on a frozen regional forest is a **software** object the literature does not treat. That is fine. Cite the science; do not claim the doorbell is a discovery method.

---

## 4. What I did *not* find

- A USGS or state-survey **predict** API for placer probability from NURE chemistry.
- A paper that trains RF on **continental** NURE vs MRDS gold and claims new US placers without a per-belt check.
- A published field trial that closed the loop: model → walk list → pans → retrain, for US placers.
- Anyone calling tree-vote spread “Monte Carlo.” (Good. We should not either.)

If a closed shop (KoBold, a major) has this internally, it is not in the open literature.

---

## 5. How to use this in an interview

**“Does this already exist?”**  
Mineral prospectivity mapping with Random Forest and deposit labels is a textbook method since Carranza ~2015. USGS used NURE in the 1990s to outline heavy-mineral placer provinces. What I did not find is a public, reproducible service that scores a US stream-sediment grab against a frozen forest and says how much the trees disagree. Government APIs serve the NURE and MRDS *tables*. They do not serve this *decision*.

**“Why RF?”**  
Rodríguez-Galiano 2015 and Carranza & Laborte: RF is stable with small, messy mineral-occurrence training sets. I did not pick it because it is fashionable.

**“Why not just map Th?”**  
Grosz already showed the multi-element heavy-mineral suite. Labeling from Th would be circular. Labels from MRDS proximity are the MPM standard and keep ground truth off the feature list.

**“Would you send a crew from 0.891?”**  
No. Airola 2018: random k-fold AUC on spatial mineral data can be a lie. I would re-score with blocked or leave-cluster-out CV, map to catchments (Yousefi & Carranza 2013), keep the mineral-system filters, and field-check a handful in-belt before I trusted another state.

---

## 6. Annotated sources (keep these)

### USGS / NURE / placers

- Smith, S.M. (1997). USGS OFR 97-492. Reformatted NURE-HSSR. https://pubs.usgs.gov/of/1997/ofr-97-0492/
- USGS NURE sediment service. ~397,625 records. https://mrdata.usgs.gov/nure/sediment/
- Grossman / Grosz, A.E. (1993). *NURE stream sediment… Ti-Zr-REE placer exploration.* USGS OFR 93-240-A. https://doi.org/10.3133/ofr93240a
- Grosz, A.E. & Schruben, P.G. (1993). *NURE geochemical and geophysical surveys: defining prospective terranes for United States placer exploration.* USGS Bull. 2097. https://doi.org/10.3133/b2097
- Grosz, A.E. et al. (1992). Heavy minerals and aeroradiometric anomalies, NC Fall Zone. USGS OFR 92-396.
- Bern, C.R. et al. (2016). REE potential, SE US Coastal Plain. https://pubs.usgs.gov/publication/70189106
- USGS Earth MRI data/services. https://mrdata.usgs.gov/earthmri/
- Earth MRI watch (states, awards, what has actually published): `EARTH_MRI_WATCH.md`
- USGS MRData API list (NURE/MRDS bbox). https://mrdata.usgs.gov/catalog/api.php

### Mineral systems / integration

- Wyborn, L.A.I. et al. (1994). Australian Proterozoic mineral systems. AusIMM.
- McCuaig, T.C. & Hronsky, J.M.A. (2014). The mineral system concept. SEG SP 18.
- Bonham-Carter, G.F. et al. (1988). Integration of geological datasets for gold exploration in Nova Scotia. *PE&RS* 54(11).
- Geoscience Australia, Exploring for the Future — mineral potential mapping. https://www.eftf.ga.gov.au/mineral-potential-mapping
- BCGS / Kenex (2024). NW British Columbia mineral potential, mineral-systems + weights of evidence.

### Random Forest / stream sediment MPM

- Carranza, E.J.M. & Laborte, A.G. (2015). RF gold prospectivity, Baguio. *Ore Geol. Rev.* 71:777–787.
- Carranza, E.J.M. & Laborte, A.G. (2015). RF, few prospects, missing values, Abra. *Comput. Geosci.* 74:60–70.
- Carranza, E.J.M. & Laborte, A.G. (2016). RF, Catanduanes. *Nat. Resour. Res.* 25:35–50.
- Rodríguez-Galiano, V. et al. (2015). ANN vs RF vs trees vs SVM. *Ore Geol. Rev.* 71:804–818.
- Zuo, R. & Carranza, E.J.M. (2011). SVM for MPM. *Comput. Geosci.* 37:1967–1975.
- Yousefi, M., Carranza, E.J.M. & Kamkar-Rouhani, A. (2013). Weighted drainage catchment basins. *J. Geochem. Explor.* 128:88–96.
- *Minerals* (2024) 14:500. Interpretable RF, Yuanbo Nang stream sediments.
- Abu / Nunoo et al. (2024). RF/SVM/ANN, Tanzania stream sediments. *Phys. Chem. Earth*.
- Yangshan / S-A fractal + RF + catchments (2025). *Acta Geologica Sinica* (English).
- *Minerals* (2023) 13:1073. RF vs SVM, limited training deposits.

### Alluvial-specific ML

- *Pure Appl. Geophys.* (2025). ML prospectivity for alluvial gold, Cauca, Colombia. https://doi.org/10.1007/s00024-025-03830-y

### Labels, uncertainty, spatial CV (limits)

- Airola, A. et al. (2018). Spatial leave-pair-out CV; ordinary AUC can be meaningless. *Data Min. Knowl. Disc.* https://doi.org/10.1007/s10618-018-00607-x
- Zuo, R. & Wang, Z. (2020). Uncertainty from random negative training samples. Goldschmidt abstract.
- *Nat. Resour. Res.* (2025). Class-label representativeness; recursive negative labeling.
- *Ore Geol. Rev.* (2026). Random vs blocked vs leave-one-cluster-out CV, Ni sulphides.

### Geochemistry practice (already in METHODOLOGY.md)

- Ahrens (1954); Reimann & Filzmoser (2000); Stanley & Sinclair (1989); Sun & McDonough (1989); Mücke & Bhaskara Rao (1996).

### Commercial / closed (not reproducible science)

- KoBold Metals — https://koboldmetals.com/ (no public API as of this search)
- Kenex mineral potential mapping — https://kenex.com.au/services/mineral-potential-mapping/
- EarthScience.AI — vendor platform; treat claims as marketing
- GA Mineral Potential WMTS — map service, not a sample scorer

---

## 7. Search notes (so this can be redone)

Queries used: RF + stream sediment + gold/placer prospectivity; NURE + ML prospectivity; mineral systems + catchments; Carranza RF; spatial CV MPM; USGS Earth MRI / NURE / MRDS APIs; KoBold / Kenex / EarthScience.AI; Grosz NURE placer; monazite eTh Coastal Plain; alluvial gold machine learning Colombia.

I did not have paywalled full text for every paper. Abstracts, USGS pubs, and open PDFs were enough to classify “exists / supports / warns / missing.” A follow-up should pull Carranza 2015 and Rodríguez-Galiano 2015 as PDFs if you cite them in a paper.

---

## 8. What to change so this is as robust as the literature allows

Do not rebuild the forest from scratch. The gold labels, the heavy-mineral features, the Fe unit fix, and the frozen medians stay. Change **how we test it, how we talk about 0.891, and what a “target” is**.

### Already solid — leave it

- Labels from MRDS **gold**, not from Th. That is the anti-circular point. Keep it.
- Valley-floor elevation cut. Keep it.
- Radiometrics **out** of the forest (they leak space into CV). Keep it.
- Serve-time medians + sklearn pin + named tree-vote spread. That is already more honest than most papers.
- Integration score still requires magnetics / trap / host. `/predict` should never become the only vote.

### Do now (this is what the papers would ding)

1. **Spatial cross-validation, and put both numbers on `/model-info`.**  
   Stratified 5-fold 0.891 can be “I recognized the neighbor of a mine.” Airola (2018) and the 2026 leave-one-cluster-out work say that number can look great and fail on new drainages.  
   **Change:** add blocked or leave-cluster-out CV (e.g. hold out a county or a cluster of MRDS sites). Report **spatial AUC** next to 0.891. Do not silently replace 0.891 — show the drop. If spatial AUC collapses, say so. That is robustness.

2. **Say what the 0-class is not.**  
   “Not near a mapped gold mine” ≠ “barren.” Zuo / later NRR papers: negatives are the weak point of all of this.  
   **Change:** one sentence on `/model-info` and in the README. Optional later: rerun with a few different far-from-mine rules and show the map does not flip.

3. **Flag “far from any training mine” on `/predict`.**  
   A high score on chemistry the forest never saw nearby is either the interesting hit or garbage.  
   **Change:** attach distance (degrees or km) to the nearest training-positive MRDS site. High P + large distance = untrusted until a human looks. Cheap, no retraining.

4. **Say gold, not REE, on the API.**  
   `/model-info` and the OpenAPI text should state the question in one line: gold-placer lookalike, not monazite.

### Do before you claim “send a crew”

5. **Walk a catchment, not an IDW blob.** Yousefi & Carranza (2013). Same-basin QA is already in the repo — use independent pour points as the **output unit**, not as a replacement label set of 12 ranked sites.

6. **Hold out a sub-area inside NE WA** (e.g. train west, test east). **Done** (Task 13). Published forest east of −118.50° is **0.53**. A west-only refit is 0.63. Joblib not rewritten.

7. **Other belts.** **Done.** Frozen NE WA forest: Idaho **0.50**, California **0.52**, Montana **0.34**, Colorado **0.56**, NC Fall Zone **0.50**. Local Sierra forest (sidecar) is a different object — quote 0.4° block **0.69**, not the 3 km 0.83. It does not beat Washington on other states. Do not retrain to paper over the sag. Do not talk about more than one state as a product. Living in California does not make the Washington forest a Sierra model.

8. **Field check a handful of high-P / no-mine drainages in NE WA.** Without pans, “robust” is still a computer talking to itself.

### Do not do in the name of robustness

- A national NURE model with one forest. The literature says transfer is the failure mode.
- Putting eTh/K/U into the forest to “use Grosz.” That is spatial leakage; Grosz used it as a **map**, not a CV feature.
- Calling tree-vote spread a 95% interval, or wrapping Task 4 tonnes onto `/predict`.
- Replacing the published gold labels with catchment-of-12. That QA is still tautological as a training set.
- Kriging the Fig 10 surface. Nice map upgrade, does not make the **classifier** more honest.

### Suggested order — status

1. Spatial CV + both AUCs in metadata / `/model-info` — **done** (0.891 vs 0.70)  
2. Gold-question sentence + negative-class sentence on `/model-info` — **done**  
3. Distance-to-nearest-gold-MRDS on `/predict` — **done**  
4. Catchment walk list (batch, not the doorbell) — **done** (Task 11; NURE stars ≠ pan pins)  
5. Idaho transfer — **done** (AUC 0.50; not retrained). Local sidecar walk list: 5 expedition / 12 confirm.
5b. California Sierra transfer — **done** (WA doorbell AUC 0.52). Local Sierra forest (sidecar; Au/As dropped; 0.03° labels; block CV 0.69; access-gated walk list). Doorbell stays WA.
5c. Montana / Colorado / Fall Zone transfer — **done** (0.34 / 0.56 / 0.50). Local sidecars on MT and CO; cousins test deleted.
6. Hold-out sub-area inside NE WA — **done** (Task 13: published forest 0.53 east of −118.50°)
8. Field pans — pamphlet / opt-in overlay shipped (`hobby_reports`; hit-rate, not AUC). Own pans still missing. Without those, “robust” is still a computer talking to itself.

Live doorbell: https://placer-lookalike.onrender.com/docs

---

## 9. If we build the expedition loop, cite these as the design backbone

1. **Grosz 1993** — which elements, and why NURE is legal input.  
2. **Carranza & Laborte** — RF + deposit labels.  
3. **Yousefi & Carranza 2013** — score catchments, not points.  
4. **McCuaig & Hronsky 2014** — do not send a crew on chemistry alone.  
5. **Airola 2018** — do not advertise k-fold 0.891 as “works in a new state.”  
6. **Field results** (does not exist yet) — the paper you would write after walking 10 drainages.

---

## 10. Placer river-geometry signatures (trap vote after chemistry)

**Searched:** 21 September 2026  
**Question:** Once a drainage already has the right chemistry, which river features — trace, long profile, flow/volume, roughness — are cheap GIS / geometry tests for the points a crew should pan?

**Short answer:** The trap literature is old and consistent. Gold drops where the water loses the power to carry it, or where the bed is rough enough to hide it, and it has to be buried before the next flood remixes it. Almost none of that belongs in the Random Forest. Almost all of the *system-scale* half can be asked of the 30 m DEM and the Task 11 stream network we already have. Bar-scale sites (bar heads, inside-bend point bars, boulder lees) need LiDAR or a boot. Bedrock-crack paystreaks are a shovel.

This is the mineral-system **trap** filter (McCuaig & Hronsky 2014) sitting on top of Task 9 chemistry. It is not a new feature column.

### 10.1 The physics, in one page

A placer is detrital gold (or any heavy) that was **moved, then left, then not remixed**.

- **Slingerland & Smith (1986)** *Ann. Rev. Earth Planet. Sci.* 14:113–147 — the review to cite. Water-laid placers form by hydraulic sorting, not by a special “gold process.”
- **Slingerland (1984)** *J. Sed. Petrol.* 54:137–150 — settling + entrainment on a rough bed. Heavies hide in the roughness. He names three scales: **bed (~0.1 m), bar (~100 m), system (~10 km)**. That scale split is the GIS honesty test.
- **Reid & Frostick (1985)** *J. Geol. Soc.* 142:739–746 — interstice trapping: small dense grains fall into cobble pores. This is why a riffle or a bar-head gravel works, and why a 30 m cell cannot see it.
- **Force (1991)** USGS OFR 91-306 — general law: settling equivalence, then entrainment equivalence, then **preservation**. Erosion in one part of the cycle must be followed by deposition without remixing. A high-energy reach that never dumps is not a mine.
- **Fletcher & Wolcott (1991)** *J. Geochem. Explor.* 41:253–274; **Fletcher & Day (1989)** *Explore* 66 — Harris Creek, south-central BC. Gold is only in transport at snowmelt flood, when cobble armor breaks. Magnetite and gold spike together, then vanish. The GIS geometry that matters is the **flood** geometry (valley floor), not the summer trickle.

Harris Creek is the closest measured analog to NE WA: cobble-gravel, nival flood, Okanagan-side interior.

### 10.2 Where gold actually sits (field facts)

**USGS / Alaska (Yeend)** — the miner’s list, written down:

- Inside of bends; deep pools below rapids; lee of large boulders (USGS FS 98-58, *Rivers of Gold*).
- **80–90%** of recovered gold is in the lowest ~1 m of gravel plus the upper ~0.5 m of bedrock cracks (Yeend, USGS B1943 Circle district; B2125 Fortymile). A GIS point on a creek is a walk target, not a grade.

**Harris Creek (Day & Fletcher 1989, 1991; Hou 2009 thesis)** — the measured list:

- Preferential storage of Au + magnetite on **bar heads**, especially at **slope breaks**.
- Fine gold can skip the head and show up at the bar tail.
- Valley morphology (confined vs open) and local channel form control the anomaly as much as distance from source. Dilution is not a smooth downstream fade: hydraulic traps make spikes.
- After a flood, gold-rich sediment can be buried by gold-poor sand. Sample soon after high water, or you miss it.

**Bar / confluence process papers:**

- **Smith & Beukes (1983)** *Econ. Geol.* 78:1342–1349 — bar-to-bank flow convergence as an alluvial-placer factory.
- **Best (1988)** and **Best & Brayshaw (1985)** — confluence shear and separation-zone bars. A tributary mouth is a first-class trap site.
- **Carling et al. (2006)** *Ore Geol. Rev.* — magnetite tracer on a gravel point bar: bar-head lag on an armored surface; “false-bottom” placers along bedding planes on the bar tail. Plan-view GIS will not see the false bottom.
- **Carling & Breakspear (2006)** *Ore Geol. Rev.* — review of placer formation in gravel-bed rivers. Explicitly sets aside (as not covered) confluences, separation-zone bars, push-bars below rapids, and bar-margin density sorting — i.e. the planform sites we *do* want to flag.
- **Levson & Giles (1990); Jacob et al. (1999)** — push-bars downstream of rapids (cited in that review).

**Paleoplacers / terraces:**

- Indian River, Yukon (Lowey and others): Holocene channels *and* low / intermediate / high terraces (White Channel gravel). The river cut down; the pay stayed on the bench.
- Yeend, ancestral Yuba (USGS PP 772) — same story in California.

**Yukon warning:** the Dawson placer-potential map rated streams from mine history and hard-rock potential. Terrain, overburden, flow, and local topography were **left out** because they were unknown on unmined creeks. Geometry is a walk list, not a reserve.

### 10.3 GIS questions that are actually cheap

These are geometry questions on a stream centerline. Drainage area *A* is the legal stand-in for discharge (Q ~ A^m). Total stream power Ω = ρgQS ≈ k·A·S. Unit power ω = Ω / width needs a width we do not have at 30 m.

| GIS question | Why gold stops | Literature | On our 30 m DEM? |
|---|---|---|---|
| Long profile + slope break | Competence falls just below a steep reach; bar heads sit on the inflection | Day & Fletcher 1989; Hack 1973 | **Yes** — sample *z* along the D8 line |
| Stream-power drop Ω ∝ A·S | Heavies drop where available power **falls**, not where it is highest | Bagnold; *Water* 11:1145 (2019) GIS Ω | **Yes** — we already compute accumulation |
| Hack SL / knickpoint | SL = (ΔH/ΔL)×L flags a steep step. The trap is at the **foot**, not on the lip | Hack 1973; Pérez-Peña SLk; SLiX toolbox 2020 | **Yes** — large knickzones only |
| Tributary junction | Area jump + shear + a slack-water bar | Best 1988; Day & Fletcher 1989 | **Yes** — stream-graph nodes |
| Valley confinement | Canyon transports; an opening dumps a bar | Gilbert et al. V-BET 2016; Harris Ck valley form | **Rough** — valley width, not channel width |
| Height above modern creek | Abandoned terraces / paleochannels keep older pay | Yeend; Indian River / White Channel | **Yes** — residual above the stream |
| Unit stream power ω = Ω/w | Same Ω in a wide reach is weaker | Biron et al. 2013; Vocal Ferencevic & Ashmore 2012 | **No** — 30 m cannot see width |
| Inside-bend / sinuosity | Helical flow dumps heavies on the point bar | Yeend FS 98-58; Chandler 2013 Yuba GIS | **Only** valley-scale bends |
| Bar head vs bar tail | Au + magnetite on the head; fines can skip to the tail | Day & Fletcher; Carling 2006 | **No** — LiDAR or a boot |
| Bedrock riffle / cracks | 80–90% of recovered gold | Yeend B1943 | **No** — lithology is a hint |
| Boulder lee / false bottom | Metres, not cells | Carling & Breakspear 2006; Force 1991 | **No** — field |
| Gold-grain shape vs distance | Flatter/rounder → farther travelled (gradient-modified distance) | Crawford 2007; Yukon grain-shape theses | **No** — needs pans, then a lab |

**Chandler (2013)** University of Denver GIS capstone, Yuba River: 40 “best” sites = 14 steep drops + 26 sharp bends, plus a road. That is the textbook Yeend list turned into clicks. It is a student map, not a validated deposit model. Cite it as “people have already tried this GIS trick,” not as proof.

**Pérez-Peña et al. / SLiX:** SL and SLk are reconnaissance tools for knickzones (also faults, lithologic contacts, landslides, dams). They make a **candidate list**. They do not know about gold.

### 10.4 What this means for *this* pipeline

Task 9 already said “this drainage looks like known gold placers.” Task 11 snaps a pour, lists NURE grabs (`nure_spots`), and votes for trap-like vertices on the D8 line (`pan_locations`: slope break, power drop, knickpoint foot, tributary mouth). Those pins are a walk list at 30 m, not bar heads. Need LiDAR or a boot before claiming a riffle.

Need LiDAR (or a mapped channel) before claiming bar heads, unit stream power, or real sinuosity.

Stay **out of the forest.** Slope, stream power, and “distance to a bend” are spatial. They leak the same neighborhood the MRDS labels already know, and they describe the 200 m valley-floor label rule. That is circular. The papers treat trap geometry as the filter after chemistry (Yousefi & Carranza 2013; McCuaig & Hronsky 2014). Same decision as the earlier topo question.

### 10.5 Sources for this section

**Process / theory**

- Slingerland, R. & Smith, N.D. (1986). Occurrence and formation of water-laid placers. *AREPS* 14:113–147. https://doi.org/10.1146/annurev.ea.14.050186.000553
- Slingerland, R. (1984). Role of hydraulic sorting in the origin of fluvial placers. *J. Sed. Petrol.* 54:137–150.
- Reid, I. & Frostick, L.E. (1985). Role of settling, entrainment and dispersive equivalence and of interstice trapping. *J. Geol. Soc.* 142:739–746.
- Force, E.R. (1991). USGS OFR 91-306. Placer concentration requirements.
- Carling, P.A. & Breakspear, R.M.G. (2006). Placer formation in gravel-bedded rivers: a review. *Ore Geol. Rev.*
- Carling, P.A. et al. (2006). Magnetite tracer across a gravel point-bar. *Ore Geol. Rev.*
- Smith, N.D. & Beukes, N.J. (1983). Bar-to-bank flow convergence. *Econ. Geol.* 78:1342–1349.
- Best, J.L. (1988). Sediment transport and bed morphology at river channel confluences. *Sedimentology.*
- Best, J.L. & Brayshaw, A.C. (1985). Flow separation — a physical model for the sedimentology of palaeochannels. *Sedimentology.*
- Garnett, R.H.T. & Bassett, N.C. (2005). Placer deposits. *Econ. Geol. 100th Anniv. Vol.* (classification / benches).

**Harris Creek / exploration geochemistry**

- Day, S.J. & Fletcher, W.K. (1989). Effects of valley and local channel morphology on gold in Harris Creek. *J. Geochem. Explor.* 32:1–16.
- Fletcher, W.K. & Day, S. (1989). Seasonal variation in transport of gold in Harris Creek. *Explore* 66.
- Fletcher, W.K. & Wolcott, J. (1991). Transport of magnetite and gold in Harris Creek. *J. Geochem. Explor.* 41:253–274.
- Hou, Z. (2009). Sediment budget of gold and magnetite, Harris Creek. UBC M.Sc.
- Day, S.J. & Fletcher, W.K. (1986). Particle size and abundance of gold, southern BC. *J. Geochem. Explor.* 26:203–214.

**USGS placer facts**

- Yeend, W.E. (1991). Gold placers of the Circle district, Alaska. USGS B1943. https://doi.org/10.3133/b1943
- Yeend, W.E. (1996). Gold placers of the Fortymile River region, Alaska. USGS B2125.
- Yeend, W.E. (1974). Gold-bearing gravel of the ancestral Yuba River. USGS PP 772.
- USGS FS 98-58 (1998). *Rivers of Gold: placer mining in Alaska.* https://pubs.usgs.gov/fs/1998/0058/report.pdf

**Long profile / stream power GIS (not placer papers — the calculators)**

- Hack, J.T. (1973). Stream-profile analysis and stream-gradient index. *J. Res. USGS* 1:421–429.
- Pérez-Peña, J.V. et al. Spatial analysis of stream power using GIS: SLk anomaly maps. *ESPL.*
- Troiani, F. et al. (2020). SLiX: GIS toolbox for SL / knickzones. *ISPRS Int. J. Geo-Inf.* 9:69.
- Goldrick, G. & Bishop, P. (2007). DS form of the long profile vs Hack SL. *ESPL.*
- *Water* (2019) 11:1145. Stream power determination in GIS (GRASS). https://doi.org/10.3390/w11061145
- Vocal Ferencevic, M. & Ashmore, P. (2012); Biron, P.M. et al. (2013). Extracting slope, width, Ω, ω from DEMs.
- Gilbert, J.T. et al. (2016). V-BET valley-bottom extraction. *ESPL.*
- Rice, S.P. & Church, M. — unit stream power / gravel-bed width (context for ω).

**Planform GIS attempts / terraces / grain shape**

- Chandler, J.D. (2013). GIS analysis of placer gold potential, Yuba River. Univ. of Denver MS GIS capstone. https://digitalcommons.du.edu/geog_ms_capstone/32
- Yukon Geological Survey (2012). Dawson placer gold potential map. (explicitly omitted topo/flow.)
- Lowey, G.W. and others — Indian River stratigraphy and terrace placers, Yukon.
- Crawford, E. (2007) and later Yukon theses — gradient-modified transport distance from gold-grain shape.

### 10.6 Search notes

Queries: Slingerland Smith water-laid placers; Day Fletcher Harris Creek bar head; Yeend placer bedrock; Hack SL knickpoint GIS; stream power GIS Bagnold; tributary confluence Best; V-BET confinement; Carling gravel-bed placer review; Force 1991 placer; Chandler Yuba GIS; Yukon Dawson placer potential; gold grain shape transport distance.

I did not have paywalled full text for Carling 2006 or Day & Fletcher 1989. Abstracts, USGS/BCGS open PDFs, and review snippets were enough to classify “do now / need LiDAR / field only.” A follow-up should pull Day & Fletcher 1989 and Slingerland 1984 as PDFs if this becomes a methods section.
