"""
TASK 6: Probabilistic grade model framing — future work section
Generates:
  outputs/text/task6_future_work_section.txt
  outputs/text/task6_gold_coplacer_paragraph.txt
"""

future_work = """
6.  Future Work

6.1  Probabilistic Grade Modeling

The spatial prioritization framework presented here identifies tailings sites
with elevated probability of monazite endowment based on multiple proxy datasets:
NURE stream sediment thorium anomalies, aeromagnetic signatures consistent with
magnetite-bearing source rocks, geologic catchment composition, and historical
production volumes. What it does not produce is a quantitative estimate of
monazite grade distribution within any individual tailings pile at the precision
required for a reprocessing investment decision.

Addressing this gap requires stochastic subsurface modeling applied to the
priority sites identified in Section 5. The appropriate methodological framework
integrates the geochemical proxy data compiled here with any available in-situ
drill hole, trench, or bulk sample data to produce full probability distributions
of monazite grade and volume, rather than the point estimates and wide uncertainty
ranges that are the maximum achievable from stream sediment proxy data alone.
Specifically, future work should apply Bayesian geostatistical approaches (Dowd
1994; Caers 2011; Pyrcz & Deutsch 2014) to encode physically informed priors
about the spatial distribution of heavy mineral concentration in fluvial tailings:
the physics of hydraulic sorting that governs monazite accumulation in a tailings
pile during active processing operations is well characterized (Strezov et al.
2017), providing a physically grounded prior that would update efficiently on
sparse in-situ observations. Such an approach would produce an uncertainty-
quantified distribution of grade at depth — the basis for a formal exploration
target estimate and, ultimately, a NI 43-101 or JORC Code Inferred Resource —
rather than the conceptual exploration target presented in Section 5.3.

The difference between a proxy-based spatial prioritization and a probabilistic
in-situ grade model is the difference between knowing where to look and knowing
whether it is worth looking there before committing to an extensive sampling
program. Emerging probabilistic subsurface modeling platforms designed specifically
for critical minerals exploration settings are beginning to provide this capability
at the site scale (see e.g., Wellmann & Caumon 2018; de la Varga et al. 2019 for
the methodological underpinning of uncertainty-quantified geological models). The
computational tools for integrating sparse geochemical observations with
physics-informed structural priors now exist; the limiting factor for the sites
identified here is in-situ sampling data, not analytical capability. Future work
should proceed in two stages: (1) targeted shallow sampling (auger drilling or
trenching) at the top three priority sites identified in Section 5.4 to generate
in-situ grade observations at a density sufficient to initialize a geostatistical
model; (2) application of a probabilistic grade modeling workflow to those
observations to produce site-scale estimates with explicit uncertainty bounds.

Cost guidance: A reconnaissance sampling program of 20-30 auger holes to 3m
depth at each of the top three sites, with full REE assay on each interval, is
estimated at $40,000–$80,000 per site including mobilization (based on published
rates for similar programs in the Pacific Northwest, Western Mineral Group 2023;
US Critical Minerals 2024). This sampling investment would reduce grade
uncertainty from the current ±50% to approximately ±15-25%, making the
exploration target estimates in Section 5.3 sufficient to support a preliminary
economic assessment.

6.2  Co-placer Heavy Mineral Characterization

The current study focuses on monazite as the target REE carrier. Future
mineralogical characterization of priority tailings sites should employ automated
mineralogy (MLA/QEMSCAN or similar) on representative bulk samples to quantify
the full heavy mineral assemblage: monazite, magnetite, ilmenite, zircon, and
any other accessory phases. The economic case for reprocessing improves
substantially if multiple heavy mineral co-products can be recovered from a single
processing pass. Magnetite and ilmenite, in particular, are recoverable by
straightforward magnetic and gravity separation with no radioactive waste
considerations, and may contribute meaningfully to project economics at the grade
levels implied by the aeromagnetic data at sites such as Colville Placer.

6.3  Hard Rock Source Verification

The detrital source characterization in Section 4.3 identified metamorphic core
complex metapelites as the dominant upstream contributor to NURE thorium anomalies
in the study area, consistent with the presence of monazite as an accessory phase
in high-grade metamorphic assemblages. Verification of this interpretation would
benefit from U-Pb geochronology and Lu-Hf isotope analysis of detrital monazite
grains separated from representative tailings samples, which would provide both
confirmation of the metamorphic source and age information useful for regional
provenance studies (Rasmussen & Muhling 2007; Holder et al. 2015). This is a
lower-priority analytical investment relative to in-situ grade sampling but would
substantially strengthen the peer-reviewed publication record for the district.

References for this section (selected):
  Caers J (2011): Modeling Uncertainty in the Earth Sciences. Wiley.
  de la Varga M, Schaaf A, Wellmann F (2019): GemPy 1.0 — open-source stochastic
    geological modeling. Geosci Model Dev 12, 1–32.
  Dowd PA (1994): Optimal estimation of recoverable reserves. Math Geol 26, 917–930.
  Holder RM, Hacker BR, Kylander-Clark ARC, Cottle JM (2015): Monazite trace-
    element and isotopic signatures of (ultra)high-pressure metamorphism.
    Chem Geol 409, 28–52.
  Pyrcz MJ, Deutsch CV (2014): Geostatistical Reservoir Modeling, 2nd ed. Oxford.
  Rasmussen B, Muhling JR (2007): Monazite begets monazite: evidence for dissolution
    of detrital monazite and reprecipitation of syntectonic monazite during low-grade
    regional metamorphism. Contrib Mineral Petrol 154, 675–689.
  Strezov L, Herbertson J, Kopf R (2017): Heavy mineral processing — a review.
    Min Eng 100, 85–96.
  Wellmann F, Caumon G (2018): 3D Structural geological models. Adv Geophys 59, 1–121.
"""

with open('outputs/text/task6_future_work_section.txt', 'w') as f:
    f.write(future_work)
print("Future work section written")

# ── Gold/co-placer paragraph ──────────────────────────────────────────────────
gold_coplacer_para = """
NOTE ON GOLD MINE TAILINGS AS A PRIORITY SUBSET FOR REE RECOVERY
(for inclusion in Introduction or Discussion section, or as supplementary context)

Historical placer gold mine tailings represent a particularly high-prior subset of
the broader tailings inventory for monazite endowment, for reasons rooted in the
physics of sediment transport. Gold (density 19.3 g/cm³) and monazite (density
5.1 g/cm³), while differing substantially in density, both concentrate in the
heavy mineral fraction of fluvial sediment through the same hydraulic mechanism:
preferential settling in zones of reduced stream competence — point bars, bedrock
irregularities, and riffles (Richards 1999; Boyle 1979). The result is that placer
gold and placer monazite co-concentrate in the same hydraulic traps, and the
tailings from historical gold placer operations therefore retain the heavy mineral
fraction, including monazite, that was not the target of the original operation
and was discarded with the light mineral gangue.

This co-placer relationship means that any operation that successfully
concentrated placer gold by density separation also, necessarily, concentrated
monazite and other heavy minerals in the same fraction. The subsequent discarding
of the heavy mineral tails as economically worthless — which was universally the
case for REEs prior to approximately 2010 — left a pre-concentrated heavy mineral
assemblage in the tailings pile that is amenable to REE recovery without the
sorting step that would be required for primary processing of unworked alluvial
material. For this reason, historical gold placer mine tailings should be assigned
a higher prior probability of monazite endowment than tailings from operations
targeting other commodities (base metals, industrial minerals) in the same
geographic region, even in the absence of direct geochemical documentation. The
northeastern Washington study area contains twelve documented historical placer
gold operations (Section 3.1); all twelve are included in the priority tier
assessment on this basis, rather than only those with documented NURE anomalies
in their immediate vicinity.

The same logic extends, with somewhat lower prior probability, to hard rock gold
mine tailings in certain geological settings. Lode gold deposits hosted in
metasedimentary terranes — orogenic gold systems of the type common in the
Okanogan Highlands — are spatially associated with the same high-grade metamorphic
rocks that produce accessory monazite as a metamorphic phase (Goldfarb et al.
2001). Milling of these ores generates tailings containing both the metamorphic
silicate gangue and its accessory phase assemblage, including monazite. The Republic
Gold District (Section 5.4.3) represents this category; its classification as a
priority REE target is supported by both the orogenic gold geological setting and
the MCC metamorphic source lithology identified in the catchment analysis.

References:
  Boyle RW (1979): The geochemistry of gold and its deposits. GSC Bull 280.
  Goldfarb RJ, Groves DI, Gardoll S (2001): Orogenic gold and geologic time.
    Ore Geol Rev 18, 1–75.
  Richards JP (1999): Placers. In: Eckstrand OR et al. (eds), Geology of Canadian
    mineral deposit types. GSC Geology of Canada 8.
"""

with open('outputs/text/task6_gold_coplacer_paragraph.txt', 'w') as f:
    f.write(gold_coplacer_para)
print("Gold co-placer paragraph written")
print("\nAll Task 6 outputs saved to outputs/text/")
