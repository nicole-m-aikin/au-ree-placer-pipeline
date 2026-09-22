"""Shared Task 9 feature schema and log-imputation.

Imported by the training path and the API so serve-time fills use the same
function as training. Frozen log-medians come from the persisted metadata,
not from a one-row request.
"""

import numpy as np
import pandas as pd

# Geological feature engineering: full placer heavy mineral suite.
# REE/actinide minerals: Th, Ce, La, P (monazite — (LREE)PO4), U (uraninite/thorite)
# Au pathfinder:        Au, As (arsenopyrite halo around placer Au)
# Oxide heavy minerals: Ti (rutile + ilmenite — TiO2, FeTiO3), Fe (magnetite — Fe3O4)
# Silicate heavy minerals: Zr (zircon — ZrSiO4), Y (xenotime — YPO4)
# Aerial rad_K / rad_eTh / rad_eU are intentionally excluded: 1–10 km grids
# leak spatial autocorrelation into CV. Map layer first (Task 1 Fig 1b).
FEATURES = ['Th', 'Ce', 'La', 'P', 'U', 'Au', 'As', 'Ti', 'Fe', 'Zr', 'Y']

# Heavy-mineral cousins — local-forest test only, never the doorbell.
# Cr (chromite), Nb (columbite), Hf (zircon), Sc (mafic/heavy), W (scheelite).
COUSIN_FEATURES = ['Cr', 'Nb', 'Hf', 'Sc', 'W']

# After load_nure, every published feature is ppm. NURE stores P and Fe as wt%;
# load_nure multiplies those columns by 10,000 when the median-threshold fires.
FEATURE_UNITS_PPM = {
    'Th': 'ppm', 'Ce': 'ppm', 'La': 'ppm', 'P': 'ppm', 'U': 'ppm',
    'Au': 'ppm', 'As': 'ppm', 'Ti': 'ppm', 'Fe': 'ppm', 'Zr': 'ppm', 'Y': 'ppm',
}

FEATURES_BY_COMMODITY = {
    'placer_gold': ['Th', 'Ce', 'La', 'P', 'U', 'Au', 'As', 'Ti', 'Fe', 'Zr', 'Y'],
    'ree':         ['Th', 'Ce', 'La', 'P', 'U', 'Ti', 'Zr', 'Y'],
    'cu_mo':       ['Cu', 'Mo', 'Pb', 'Zn', 'Ag', 'Au', 'As'],
    'all':         ['Th', 'Ce', 'La', 'P', 'U', 'Au', 'As', 'Ti', 'Fe', 'Zr', 'Y'],
}

# NURE encodes below-detection as the negative of the MDL. |value| > 10 is an
# artifact, not a measurement (see load_nure / METHODOLOGY.md).
MDL_NEG_LIMIT = 10.0
P_WT_PCT_MAX = 1.0
FE_WT_PCT_MAX = 20.0
FE_PPM_MIN_PLAUSIBLE = 100.0
TRACE_PPM_MAX = 1.0e5


def feature_list_for_cfg(cfg, commodity_filter='placer_gold'):
    """Published suite, plus cousins when ml.include_cousins is on."""
    base = list(FEATURES_BY_COMMODITY.get(commodity_filter, FEATURES))
    if (cfg.get('ml') or {}).get('include_cousins'):
        base.extend(f for f in COUSIN_FEATURES if f not in base)
    return base


def usable_feature_columns(df, feature_set):
    """Keep features that exist and have at least one positive concentration.

    Empty columns (e.g. Sierra HSSR Au/As) must not become a constant
    log-median fill — that would fake a pathfinder the belt never assayed.
    Returns (kept, dropped).
    """
    kept, dropped = [], []
    for f in feature_set:
        if f not in df.columns:
            dropped.append(f)
            continue
        if not (pd.to_numeric(df[f], errors='coerce') > 0).any():
            dropped.append(f)
            continue
        kept.append(f)
    return kept, dropped


def apply_nure_mdl(value):
    """Return a non-negative concentration, or None to mean 'impute'.

    None / NaN / 0 → impute. (-10, 0) → half-MDL. < -10 is not handled here;
    the caller must reject it before this function.
    """
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(x) or x == 0.0:
        return None
    if x < 0:
        return abs(x) / 2.0
    return x


def _log_impute(df, cols, medians=None):
    """Log10-transform features; fill NaN / non-positive with log-medians.

    If ``medians`` is None, compute them from *this* frame (training).
    If provided, reuse those values (serving). Always returns
    ``(log_df, used_medians)`` so the training path can persist what it used.
    """
    result = pd.DataFrame(index=df.index)
    for col in cols:
        v = df[col].copy() if col in df.columns else pd.Series(np.nan, index=df.index)
        v = v.where(v > 0, np.nan)
        result[col] = np.log10(v)

    used = {}
    for col in cols:
        if medians is not None and col in medians and medians[col] is not None:
            med = float(medians[col])
            if np.isnan(med):
                med = 0.0
        else:
            computed = result[col].median()
            med = 0.0 if pd.isna(computed) else float(computed)
        used[col] = med
        result[col] = result[col].fillna(med)
    return result, used
