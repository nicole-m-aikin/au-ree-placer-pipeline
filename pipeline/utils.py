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


def map_extent(cfg):
    """
    Canonical bounding box for all spatial map panels in this study area.
    Returns (lon_min, lon_max, lat_min, lat_max) with consistent padding.
    Use instead of raw bbox() + manual padding in every task script.
    """
    b = cfg['study_area']['bbox']
    pad = cfg['study_area'].get('map_padding', cfg.get('map_padding', 0.08))
    return (b['lon_min'] - pad, b['lon_max'] + pad,
            b['lat_min'] - pad, b['lat_max'] + pad)


def hillshade(cfg, ax, alpha=0.25, zorder=0):
    """
    Add a DEM-derived hillshade as a greyscale background to a map axis.
    No-op if data.dem_tif is null/absent in config or file not found.
    Sun azimuth 315° (NW), altitude 45°.
    Always call this FIRST before plotting any data layers.
    """
    import os
    dem_path = (cfg.get('data') or {}).get('dem_tif')
    if not dem_path:
        return
    if not os.path.isabs(dem_path):
        dem_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), dem_path)
    if not os.path.exists(dem_path):
        return
    try:
        import rasterio
        from rasterio.windows import from_bounds as _win_from_bounds
        import numpy as _np
        xmin, xmax, ymin, ymax = map_extent(cfg)
        with rasterio.open(dem_path) as src:
            win = _win_from_bounds(xmin, ymin, xmax, ymax, src.transform)
            data = src.read(1, window=win).astype(float)
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = _np.nan
        mask = _np.isnan(data)
        if mask.any():
            filled = _np.where(mask, _np.nanmedian(data), data)
        else:
            filled = data
        dy, dx = _np.gradient(filled)
        az, alt = _np.radians(315), _np.radians(45)
        shade = (_np.sin(alt) + _np.cos(alt) * (-dx * _np.cos(az) - dy * _np.sin(az)))
        shade = _np.clip(shade, 0, 1)
        ax.imshow(shade, cmap='gray', extent=[xmin, xmax, ymin, ymax],
                  origin='upper', alpha=alpha, zorder=zorder, aspect='auto')
    except Exception:
        pass  # never crash a figure because of a missing hillshade


def north_arrow(ax, x=0.96, y=0.96, size=12):
    """Add a simple 'N ↑' north arrow to a map axis (axes fraction coords)."""
    ax.annotate('N\n↑', xy=(x, y), xycoords='axes fraction',
                ha='center', va='top', fontsize=size, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='gray',
                          alpha=0.8, linewidth=0.7))


def scale_bar(ax, cfg, length_km=50, x=0.05, y=0.05):
    """
    Add a simple scale bar to a map axis.
    length_km: bar length in km. At lat 48°N, 1 deg lon ≈ 74 km.
    """
    import numpy as _np
    lat_mid = (cfg['study_area']['bbox']['lat_min'] +
               cfg['study_area']['bbox']['lat_max']) / 2
    deg_per_km = 1.0 / (111.32 * _np.cos(_np.radians(lat_mid)))
    bar_deg = length_km * deg_per_km
    xmin, xmax, ymin, ymax = map_extent(cfg)
    x0 = xmin + (xmax - xmin) * x
    y0 = ymin + (ymax - ymin) * y
    ax.plot([x0, x0 + bar_deg], [y0, y0], color='black', lw=2.5,
            solid_capstyle='butt', zorder=10)
    ax.plot([x0, x0], [y0 - 0.01, y0 + 0.01], color='black', lw=1.5, zorder=10)
    ax.plot([x0 + bar_deg, x0 + bar_deg], [y0 - 0.01, y0 + 0.01],
            color='black', lw=1.5, zorder=10)
    ax.text(x0 + bar_deg / 2, y0 + 0.025, f'{length_km} km',
            ha='center', va='bottom', fontsize=7.5, fontweight='bold',
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))
