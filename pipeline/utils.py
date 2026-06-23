"""Shared utilities for the Au+REE placer assessment pipeline."""

import os
import numpy as np
import pandas as pd

WONG = {
    'black':      '#000000',
    'orange':     '#E69F00',
    'sky':        '#56B4E9',
    'green':      '#009E73',
    'yellow':     '#F0E442',
    'blue':       '#0072B2',
    'vermillion': '#D55E00',
    'pink':       '#CC79A7',
}

# CI chondrite normalisation values: Sun & McDonough 1989
CHONDRITE_SUN89 = {
    'La': 0.237,   'Ce': 0.612,   'Pr': 0.0949,  'Nd': 0.467,
    'Sm': 0.153,   'Eu': 0.058,   'Gd': 0.2055,  'Tb': 0.0374,
    'Dy': 0.254,   'Ho': 0.0566,  'Er': 0.1655,  'Tm': 0.0255,
    'Yb': 0.170,   'Lu': 0.0254,  'Y':  1.57,
}


def setup_mpl():
    """Set MPLCONFIGDIR and switch to non-interactive Agg backend."""
    os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig'
    import matplotlib
    matplotlib.use('Agg')


def load_nure(cfg):
    """Load NURE CSV, apply half-MDL below-detection substitution, convert P pct→ppm."""
    path = cfg['data']['nure_csv']
    df = pd.read_csv(path)
    COORD_COLS = {'lat', 'lon', 'lat_orig', 'long_orig', 'depth'}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in COORD_COLS]
    for col in numeric_cols:
        neg = df[col] < 0
        small = neg & (df[col].abs() <= 10)
        large = neg & (df[col].abs() > 10)
        df.loc[small, col] = df.loc[small, col].abs() / 2
        df.loc[large, col] = np.nan
    if 'P' in df.columns and df['P'].notna().any() and df['P'].dropna().median() < 1:
        df['P'] = df['P'] * 10000
    return df


def anomaly_threshold(series):
    """Return mean+2SD on log10 of positive values, or None if insufficient data."""
    pos = series.dropna()
    pos = pos[pos > 0]
    if len(pos) < 5:
        return None
    log_vals = np.log10(pos)
    return 10 ** (log_vals.mean() + 2 * log_vals.std())


def chondrite_normalize(df, cols):
    """Divide element columns by Sun & McDonough 1989 CI chondrite values.

    Columns absent from df are returned as NaN rather than raising KeyError.
    """
    result = pd.DataFrame(index=df.index)
    for col in cols:
        if col not in df.columns:
            result[col] = np.nan
        elif col in CHONDRITE_SUN89:
            result[col] = df[col] / CHONDRITE_SUN89[col]
        else:
            result[col] = df[col].copy()
    return result


def watermark(fig, cfg=None):
    """Add 'EXPLORATION TARGET ONLY' footer to figure."""
    area = (cfg or {}).get('study_area', {}).get('name', '')
    prefix = f"{area} REE Tailings Assessment — " if area else ''
    fig.text(
        0.5, 0.01,
        f"{prefix}Au+REE Pipeline Project — EXPLORATION TARGET ONLY",
        ha='center', fontsize=7, color='gray', style='italic',
    )


def save_fig(fig, path, dpi=300):
    """Save figure at 300 dpi and close it."""
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def ensure_outputs(outputs_dir):
    """Create standard output subdirectories if they don't exist."""
    for sub in ('figures', 'tables', 'geojson', 'text'):
        os.makedirs(os.path.join(outputs_dir, sub), exist_ok=True)


def out(cfg, subdir, filename):
    """Construct a path under the study-area outputs directory."""
    return os.path.join(cfg['outputs_dir'], subdir, filename)


def wgs_path(cfg):
    """Resolve WGS OFR 2026-02 Excel path from env var or config."""
    env = os.environ.get('WGS_OFR2026_PATH')
    if env:
        return env
    cfg_path = (cfg.get('data') or {}).get('wgs_excel')
    if cfg_path:
        return cfg_path
    return os.path.join('data', 'wgs_ofr2026', 'ger_ofr2026-02_data_supplement.xlsx')


def sites_gdf(cfg):
    """Build a GeoDataFrame of mine sites from config."""
    import geopandas as gpd
    from shapely.geometry import Point
    sites = cfg['sites']
    df = pd.DataFrame(sites)
    return gpd.GeoDataFrame(
        df,
        geometry=[Point(s['lon'], s['lat']) for s in sites],
        crs='EPSG:4326',
    )


def bbox(cfg):
    """Return (lon_min, lon_max, lat_min, lat_max) tuple."""
    b = cfg['study_area']['bbox']
    return b['lon_min'], b['lon_max'], b['lat_min'], b['lat_max']
