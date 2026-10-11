"""
prep_nure.py
Reads the NGDB relational CSV export, joins metadata + best-value chemistry,
pivots to wide format, and writes cleaned state extracts.

Shared repo (written once, reused by any project):
  /Users/nicoleaikin/data-repos/ngdb-processed/WA_ngdb.csv

Project subsets (NE WA bounding box):
  data/nure/nure_ne_wa_sediment.csv
  data/nure/nure_wa_sediment.csv

Units: precious-metal species (PPB_PRONE) are normalized to ppm before the
pivot (ppb / 1000; 1 g/t = 1 ppm = 1000 ppb). Other species keep the value as
reported, because utils.load_nure converts P/Ca/Fe wt% -> ppm with a median
heuristic and a second conversion here would double it. The original unit of
every species is kept in a `{el}_unit` column.
"""

import os

import numpy as np
import pandas as pd

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

# Species commonly reported in ppb by some labs and ppm by others.
PPB_PRONE = ('Au', 'Ag', 'Pt', 'Pd', 'Rh', 'Ru', 'Ir', 'Os')

UNIT_ALIASES = {
    'ppm': 'ppm', 'ug/g': 'ppm', 'µg/g': 'ppm', 'μg/g': 'ppm', 'mg/kg': 'ppm', 'g/t': 'ppm',
    'ppb': 'ppb', 'ng/g': 'ppb', 'ug/kg': 'ppb', 'µg/kg': 'ppb', 'μg/kg': 'ppb', 'mg/t': 'ppb',
    'pct': 'pct', '%': 'pct', 'wt%': 'pct', 'wt.%': 'pct', 'wt %': 'pct', 'percent': 'pct',
}
TO_PPM = {'ppm': 1.0, 'ppb': 1e-3, 'pct': 1e4}


def canonical_unit(unit):
    """Map a raw NGDB unit string to 'ppm' | 'ppb' | 'pct', or None if unknown/missing."""
    if unit is None or (isinstance(unit, float) and np.isnan(unit)):
        return None
    return UNIT_ALIASES.get(str(unit).strip().lower())


def dedupe_bestvalue(bv):
    """One row per (lab_id, species), preferring the species' majority unit.

    Returns (deduped frame, stats dict). Remaining ties keep the first row in
    file order.
    """
    bv = bv.reset_index(drop=True).copy()
    if 'unit' not in bv.columns:
        bv['unit'] = None
    raw = bv['unit'].astype('string').str.strip().str.lower()
    unit_key = bv['unit'].map(canonical_unit).astype('string').fillna(raw)
    majority = unit_key.groupby(bv['species']).agg(
        lambda s: s.value_counts().index[0] if s.notna().any() else None)
    bv['_is_major'] = (unit_key == bv['species'].map(majority)).fillna(False).astype(bool)
    bv['_order'] = np.arange(len(bv))

    dup_mask = bv.duplicated(subset=['lab_id', 'species'], keep=False)
    groups = bv[dup_mask].groupby(['lab_id', 'species'])
    n_groups = groups.ngroups
    n_by_unit = int(groups['_is_major'].agg(lambda s: s.any() and not s.all()).sum()) if n_groups else 0

    out = (bv.sort_values(['lab_id', 'species', '_is_major', '_order'],
                          ascending=[True, True, False, True])
             .drop_duplicates(subset=['lab_id', 'species'], keep='first')
             .sort_values('_order')
             .drop(columns=['_is_major', '_order'])
             .reset_index(drop=True))
    stats = {
        'dup_groups': n_groups,
        'resolved_by_majority_unit': n_by_unit,
        'resolved_by_file_order': n_groups - n_by_unit,
        'majority_unit': majority.dropna().to_dict(),
    }
    return out, stats


def normalize_units(bv, elements=PPB_PRONE, convert_pct=False):
    """Convert qvalue to ppm for `elements` where the unit is known.

    Adds `unit_orig` (raw unit string) and `unit_ppm` (True once qvalue is ppm).
    pct is converted only when convert_pct=True — utils.load_nure already turns
    P/Ca/Fe wt% into ppm by a median heuristic, so converting here too would
    double it. Negative (below-detection) values keep their sign.
    """
    bv = bv.copy()
    if 'unit' not in bv.columns:
        bv['unit'] = None
    bv['unit_orig'] = bv['unit']
    canon = bv['unit'].map(canonical_unit)
    target = bv['species'].isin(elements)
    allowed = {'ppm', 'ppb'} | ({'pct'} if convert_pct else set())
    convert = target & canon.isin(allowed)
    factor = canon.map(TO_PPM).where(convert, 1.0).astype(float)
    bv['qvalue'] = pd.to_numeric(bv['qvalue'], errors='coerce') * factor
    bv['unit_ppm'] = convert | (canon == 'ppm')
    return bv


def pivot_bestvalue(bv, elements=PPB_PRONE, convert_pct=False):
    """Long NGDB bestvalue -> wide table with `{el}` values and `{el}_unit` columns.

    Returns (wide frame keyed on lab_id, stats dict).
    """
    dedup, stats = dedupe_bestvalue(bv)
    norm = normalize_units(dedup, elements=elements, convert_pct=convert_pct)

    unknown = norm['species'].isin(elements) & ~norm['unit_ppm'] & norm['qvalue'].notna()
    stats['unconverted_ppb_prone_rows'] = int(unknown.sum())

    values = norm.pivot(index='lab_id', columns='species', values='qvalue')
    units = norm.pivot(index='lab_id', columns='species', values='unit_orig')
    units.columns = [f'{c}_unit' for c in units.columns]
    wide = pd.concat([values, units], axis=1)
    wide.columns.name = None
    return wide.reset_index(), stats


def summarize_units(bv, species):
    """value_counts of raw units for one species (as a plain dict)."""
    if 'unit' not in bv.columns:
        return {}
    return bv.loc[bv['species'] == species, 'unit'].value_counts(dropna=False).to_dict()


def main():
    # ── Load metadata ─────────────────────────────────────────────────────────
    print("Loading main.csv ...")
    main_df = pd.read_csv(f"{EXTRACT_DIR}/main.csv", low_memory=False)
    print(f"  {len(main_df):,} total samples")

    wa_meta = main_df[main_df['state'] == 'WA'].copy()
    wa_meta = wa_meta.rename(columns={'lat_wgs84': 'lat', 'long_wgs84': 'lon'})
    wa_ids  = set(wa_meta['lab_id'])
    print(f"  {len(wa_meta):,} Washington samples")

    # ── Load best-value chemistry (long format), filter to WA only ────────────
    print("\nLoading bestvalue.csv (long format) ...")
    bv = pd.read_csv(f"{EXTRACT_DIR}/bestvalue.csv", low_memory=False)
    print(f"  {len(bv):,} total chemistry records")

    bv_wa = bv[bv['lab_id'].isin(wa_ids)].copy()
    print(f"  {len(bv_wa):,} WA chemistry records")

    au_units = summarize_units(bv_wa, 'Au')
    print(f"\n  Au units in WA (raw): {au_units}")

    # ── Pivot to wide format ──────────────────────────────────────────────────
    print("\nPivoting chemistry to wide format ...")
    wide, stats = pivot_bestvalue(bv_wa)
    print(f"  Duplicate (lab_id, species) groups: {stats['dup_groups']:,}"
          f"  (majority unit chose {stats['resolved_by_majority_unit']:,};"
          f" file order chose {stats['resolved_by_file_order']:,})")
    if stats['unconverted_ppb_prone_rows']:
        print(f"  WARNING: {stats['unconverted_ppb_prone_rows']:,} {'/'.join(PPB_PRONE)} rows"
              " have an unknown (or pct) unit and were left as reported")
    print(f"  Wide table: {len(wide):,} rows x {len(wide.columns)} columns")

    # ── Join metadata + chemistry ─────────────────────────────────────────────
    wa = wa_meta.merge(wide, on='lab_id', how='left')
    print(f"\nJoined WA table: {len(wa):,} rows x {len(wa.columns)} columns")

    # ── Element groups for coverage report ───────────────────────────────────
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

    # ── NE Washington bounding box ────────────────────────────────────────────
    ne_wa = wa[
        (wa['lat'] >= NE_WA_BBOX['lat_min']) &
        (wa['lat'] <= NE_WA_BBOX['lat_max']) &
        (wa['lon'] >= NE_WA_BBOX['lon_min']) &
        (wa['lon'] <= NE_WA_BBOX['lon_max'])
    ].copy()
    print(f"\nNE Washington records (bounding box): {len(ne_wa):,}")

    # ── Save shared repo (full WA) ────────────────────────────────────────────
    os.makedirs(SHARED_REPO, exist_ok=True)
    wa_out = os.path.join(SHARED_REPO, "WA_ngdb.csv")
    wa.to_csv(wa_out, index=False)
    print(f"\nWrote: {wa_out}")

    # ── Save project subsets ──────────────────────────────────────────────────
    os.makedirs(PROJECT_NURE, exist_ok=True)
    ne_wa.to_csv(f"{PROJECT_NURE}/nure_ne_wa_sediment.csv", index=False)
    wa.to_csv(f"{PROJECT_NURE}/nure_wa_sediment.csv", index=False)
    print(f"Wrote: {PROJECT_NURE}/nure_ne_wa_sediment.csv")
    print(f"Wrote: {PROJECT_NURE}/nure_wa_sediment.csv")

    # ── Final summary ─────────────────────────────────────────────────────────
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

  Au units (raw): {au_units}
    Normalized to ppm before the pivot (ppb / 1000; 1 g/t = 1 ppm = 1000 ppb).
    Raw unit kept in the Au_unit column.

  To add another state later:
    Change  main_df[main_df['state'] == 'WA']  →  main_df[main_df['state'] == 'ID']
    Change output to ID_ngdb.csv
    Raw data in {EXTRACT_DIR} is never touched.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == '__main__':
    main()
