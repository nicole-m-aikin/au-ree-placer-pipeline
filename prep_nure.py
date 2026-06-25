"""
prep_nure.py
Reads the NGDB relational CSV export, joins metadata + best-value chemistry,
pivots to wide format, and writes cleaned state extracts.

Shared repo (written once, reused by any project):
  /Users/nicoleaikin/data-repos/ngdb-processed/WA_ngdb.csv

Project subsets (NE WA bounding box):
  data/nure/nure_ne_wa_sediment.csv
  data/nure/nure_wa_sediment.csv
"""

import pandas as pd
import os

# ── Paths (override with env vars; no hardcoded user paths) ───────────────────
EXTRACT_DIR  = os.environ.get('NGDB_EXTRACT_DIR',  'ngdbsed')
SHARED_REPO  = os.environ.get('NGDB_SHARED_REPO',  'ngdb-processed')
PROJECT_NURE = "data/nure"

# Override state and bounding box via env vars for other study areas
_lat_min = float(os.environ.get('NURE_LAT_MIN', 47.5))
_lat_max = float(os.environ.get('NURE_LAT_MAX', 49.1))
_lon_min = float(os.environ.get('NURE_LON_MIN', -120.0))
_lon_max = float(os.environ.get('NURE_LON_MAX', -117.0))
NE_WA_BBOX = dict(lat_min=_lat_min, lat_max=_lat_max, lon_min=_lon_min, lon_max=_lon_max)

# ── Load metadata ─────────────────────────────────────────────────────────────
print("Loading main.csv ...")
main = pd.read_csv(f"{EXTRACT_DIR}/main.csv", low_memory=False)
print(f"  {len(main):,} total samples")

wa_meta = main[main['state'] == 'WA'].copy()
wa_meta = wa_meta.rename(columns={'lat_wgs84': 'lat', 'long_wgs84': 'lon'})
wa_ids  = set(wa_meta['lab_id'])
print(f"  {len(wa_meta):,} Washington samples")

# ── Load best-value chemistry (long format), filter to WA only ────────────────
print("\nLoading bestvalue.csv (long format) ...")
bv = pd.read_csv(f"{EXTRACT_DIR}/bestvalue.csv", low_memory=False)
print(f"  {len(bv):,} total chemistry records")

bv_wa = bv[bv['lab_id'].isin(wa_ids)].copy()
print(f"  {len(bv_wa):,} WA chemistry records")

# ── Check Au units (may be ppm, ppb, or g/t) ─────────────────────────────────
au_units = bv_wa[bv_wa['species'] == 'Au']['unit'].value_counts()
print(f"\n  Au units in WA: {au_units.to_dict()}")

# ── Pivot to wide format ──────────────────────────────────────────────────────
# For elements with multiple units (rare), keep the best-value row (already
# de-duped by USGS in bestvalue.csv).  If duplicates remain after filtering,
# take the first occurrence per (lab_id, species).
print("\nPivoting chemistry to wide format ...")
bv_wa_dedup = bv_wa.drop_duplicates(subset=['lab_id', 'species'], keep='first')
wide = bv_wa_dedup.pivot(index='lab_id', columns='species', values='qvalue')
wide.columns.name = None
wide = wide.reset_index()
print(f"  Wide table: {len(wide):,} rows x {len(wide.columns)} columns")

# ── Join metadata + chemistry ─────────────────────────────────────────────────
wa = wa_meta.merge(wide, on='lab_id', how='left')
print(f"\nJoined WA table: {len(wa):,} rows x {len(wa.columns)} columns")

# ── Element groups for coverage report ───────────────────────────────────────
groups = {
    'Light REEs':               ['La','Ce','Pr','Nd','Sm','Eu'],
    'Heavy REEs + Y':           ['Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Y'],
    'Monazite pathfinders':     ['Th','U','P'],
    'Au + epithermal':          ['Au','Ag','As','Sb','Hg','Bi','Te','Se'],
    'Porphyry Cu':              ['Cu','Mo','Re'],
    'Base metals':              ['Pb','Zn','Ni','Co'],
    'Ore metals':               ['Sn','W','In','Cd','Ge','Ga'],
    'Mafic indicators':         ['Cr','V','Sc'],
    'Co-placer heavy minerals': ['Fe','Ti','Zr','Hf','Nb','Ta'],
    'Major elements':           ['Al','Ca','Mg','K','Na','Mn','Si'],
    'Alkali/alkaline earth':    ['Ba','Sr','Rb','Cs','Li','Be'],
}

print("\n── Element coverage in WA NGDB extract ──────────────────────────────")
all_found, all_missing = [], []
for group_name, elements in groups.items():
    found   = [e for e in elements if e in wa.columns]
    missing = [e for e in elements if e not in wa.columns]
    all_found.extend(found)
    all_missing.extend(missing)
    status = f"{len(found)}/{len(elements)}"
    print(f"  {group_name:28s} {status:5s}  found: {found}")
    if missing:
        print(f"  {'':28s}        missing: {missing}")

# ── NE Washington bounding box ────────────────────────────────────────────────
ne_wa = wa[
    (wa['lat'] >= NE_WA_BBOX['lat_min']) &
    (wa['lat'] <= NE_WA_BBOX['lat_max']) &
    (wa['lon'] >= NE_WA_BBOX['lon_min']) &
    (wa['lon'] <= NE_WA_BBOX['lon_max'])
].copy()
print(f"\nNE Washington records (bounding box): {len(ne_wa):,}")

# ── Save shared repo (full WA) ────────────────────────────────────────────────
os.makedirs(SHARED_REPO, exist_ok=True)
wa_out = os.path.join(SHARED_REPO, "WA_ngdb.csv")
wa.to_csv(wa_out, index=False)
print(f"\nWrote: {wa_out}")

# ── Save project subsets ──────────────────────────────────────────────────────
os.makedirs(PROJECT_NURE, exist_ok=True)
ne_wa.to_csv(f"{PROJECT_NURE}/nure_ne_wa_sediment.csv", index=False)
wa.to_csv(f"{PROJECT_NURE}/nure_wa_sediment.csv", index=False)
print(f"Wrote: {PROJECT_NURE}/nure_ne_wa_sediment.csv")
print(f"Wrote: {PROJECT_NURE}/nure_wa_sediment.csv")

# ── Final summary ─────────────────────────────────────────────────────────────
critical = {
    'Th (monazite proxy)':         'Th' in wa.columns,
    'Ce (LREE discrimination)':    'Ce' in wa.columns,
    'La (LREE discrimination)':    'La' in wa.columns,
    'P  (phosphate confirmation)': 'P'  in wa.columns,
    'Au (gold pathfinder)':        'Au' in wa.columns,
    'As (epithermal pathfinder)':  'As' in wa.columns,
    'Cu (porphyry pathfinder)':    'Cu' in wa.columns,
    'Mo (porphyry pathfinder)':    'Mo' in wa.columns,
}

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY""")
print(f"  Raw source:       {EXTRACT_DIR}")
print(f"  WA records:       {len(wa):,}")
print(f"  NE WA records:    {len(ne_wa):,}")
print(f"  Elements found:   {len(all_found)} / {len(all_found)+len(all_missing)}")
print(f"\n  Critical elements for pipeline:")
for label, present in critical.items():
    print(f"    {label:35s} {'YES ✓' if present else 'NO ✗  <-- gap'}")
print(f"""
  Files written:
    {wa_out}
    {PROJECT_NURE}/nure_ne_wa_sediment.csv
    {PROJECT_NURE}/nure_wa_sediment.csv

  Au units note: {au_units.to_dict()}
    (g/t ≈ ppb for Au; ppm values are also present)

  To add another state later:
    Change  main[main['state'] == 'WA']  →  main[main['state'] == 'ID']
    Change output to ID_ngdb.csv
    Raw data in {EXTRACT_DIR} is never touched.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
