"""Shared utilities for the Au+REE placer assessment pipeline.

MAP LAYER CLIPPING CONTRACT — enforced for all figures in this pipeline
=======================================================================
Any layer drawn onto a map axis in data coordinates must be clipped to
map_extent(cfg) before plotting. The hillshade (imshow with extent=)
is always correctly bounded and serves as the visual ground-truth reference:
if any other layer extends outside the hillshade, the clipping contract
has been violated.

Required pattern for every map axis:
  1. xmin,xmax,ymin,ymax = map_extent(cfg)
  2. ax.set_xlim / set_ylim — BEFORE any layer is plotted
  3. ax.set_aspect('auto')
  4. hillshade(cfg, ax, ...)
  5. gdf = clip_gdf_to_map(gdf, cfg) — before every gdf.plot()
  6. Clamp all ax.fill() / ax.fill_between() coords to map_extent bounds
  7. Do NOT call set_xlim/ylim again after plotting — if needed, a layer
     reset the extent and step 2 must be moved later or the layer clipped.

Gut-check: zoom the figure and compare any polygon/patch layer edge to
the hillshade edge. They must be flush. Any overhang = clipping violation.
"""

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


# Standard geographic map panel size (inches) used across ALL figures.
# All spatial map panels must be laid out to exactly these dimensions so that
# the same geographic extent appears at a consistent physical scale.
MAP_W = 5.5   # map panel width  (inches)
MAP_H = 4.2   # map panel height (inches)

# Layout margin constants (inches) — shared by all figure scripts.
_FIG_LM   = 0.65   # left margin  (y-axis labels)
_FIG_RM   = 0.25   # right margin
_FIG_TM   = 1.00   # top margin   (suptitle)
_FIG_BM   = 0.55   # bottom margin (x-axis labels)
_FIG_HGAP = 0.35   # horizontal gap between adjacent panels
_FIG_VGAP = 0.55   # vertical gap between panel rows
_FIG_CW   = 0.20   # colorbar column width
_FIG_CG   = 0.08   # gap between map panel and its colorbar


def _ax_rect(x_in, y_in, w_in, h_in, figW, figH):
    """Convert physical-inch coordinates to matplotlib axes fraction rectangle."""
    return [x_in / figW, y_in / figH, w_in / figW, h_in / figH]


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
        # Clamp the read window to the actual DEM extent so out-of-bounds rows/cols
        # are marked as invalid rather than filled with a uniform median value.
        with rasterio.open(dem_path) as src2:
            dem_xmin = max(xmin, src2.bounds.left)
            dem_xmax = min(xmax, src2.bounds.right)
            dem_ymin = max(ymin, src2.bounds.bottom)
            dem_ymax = min(ymax, src2.bounds.top)
        # Build a per-pixel valid mask based on the clipped DEM bounds
        rows, cols = data.shape
        col_coords = _np.linspace(xmin, xmax, cols, endpoint=False)
        row_coords = _np.linspace(ymax, ymin, rows, endpoint=False)
        col_grid, row_grid = _np.meshgrid(col_coords, row_coords)
        oob_mask = (col_grid < dem_xmin) | (col_grid > dem_xmax) | \
                   (row_grid < dem_ymin) | (row_grid > dem_ymax)
        mask = mask | oob_mask
        filled = _np.where(mask, _np.nanmedian(data) if not mask.all() else 0, data)
        dy, dx = _np.gradient(filled)
        az, alt = _np.radians(315), _np.radians(45)
        shade = (_np.sin(alt) + _np.cos(alt) * (-dx * _np.cos(az) - dy * _np.sin(az)))
        shade = _np.clip(shade, 0, 1)
        # Build RGBA array — OOB pixels are fully transparent (alpha=0)
        rgba = _np.zeros((*shade.shape, 4), dtype=float)
        rgba[:, :, 0] = shade  # R
        rgba[:, :, 1] = shade  # G
        rgba[:, :, 2] = shade  # B
        rgba[:, :, 3] = _np.where(mask, 0.0, alpha)  # alpha=0 outside DEM coverage
        ax.imshow(rgba, extent=[xmin, xmax, ymin, ymax],
                  origin='upper', zorder=zorder, aspect='auto')
    except Exception:
        pass  # never crash a figure because of a missing hillshade


def north_arrow(ax, x=0.96, y=0.96, size=12):
    """Add a simple 'N ↑' north arrow to a map axis (axes fraction coords)."""
    ax.annotate('N\n↑', xy=(x, y), xycoords='axes fraction',
                ha='center', va='top', fontsize=size, fontweight='bold',
                zorder=15,   # above Canada hatch (zorder 8-9)
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='gray',
                          alpha=0.9, linewidth=0.7))


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


def topo_contours(ax, cfg, interval_m=300, lw=0.30, color='#555555', alpha=0.38, zorder=2):
    """
    Overlay topographic contour lines from the study-area DEM.

    Reads the DEM configured at data.dem_tif, downsamples it to ~400×300
    pixels (fast, sufficient for regional contours), clips to map_extent,
    and draws contours at `interval_m` metre intervals.  Silently no-ops when
    the DEM path is absent or the file cannot be read.
    """
    import rasterio as _rio
    from rasterio.windows import from_bounds as _from_bounds
    from rasterio.enums import Resampling as _RS
    dem_rel = cfg.get('data', {}).get('dem_tif')
    if not dem_rel:
        return
    dem_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), dem_rel)
    if not os.path.exists(dem_path):
        return
    try:
        xmin, xmax, ymin, ymax = map_extent(cfg)
        with _rio.open(dem_path) as src:
            win = _from_bounds(xmin, ymin, xmax, ymax, src.transform)
            data = src.read(1, window=win,
                            out_shape=(300, 400),
                            resampling=_RS.bilinear).astype(float)
            nd = src.nodata
        if nd is not None:
            data[data == nd] = np.nan
        data[data < -500] = np.nan
        if np.isnan(data).all():
            return
        lons = np.linspace(xmin, xmax, data.shape[1])
        lats = np.linspace(ymax, ymin, data.shape[0])
        vmin = np.floor(np.nanmin(data) / interval_m) * interval_m
        vmax = np.ceil(np.nanmax(data) / interval_m) * interval_m
        levels = np.arange(vmin, vmax + interval_m, interval_m)
        ax.contour(lons, lats, data, levels=levels,
                   colors=color, linewidths=lw, alpha=alpha, zorder=zorder)
    except Exception:
        pass  # never crash a figure because of a missing topo layer


def rivers_with_arrows(ax, cfg, color='#5B8DB8', lw=1.5, alpha=0.80, zorder=5,
                        arrow_every=2):
    """
    Draw named rivers from config.rivers as polylines with flow-direction arrows.

    Each river entry in cfg['rivers'] needs a 'coords' list of [lon, lat] pairs
    ordered from upstream to downstream.  Arrows are drawn on the midpoint of
    every `arrow_every`-th segment.
    """
    import matplotlib.patheffects as _pe
    _stroke = [_pe.withStroke(linewidth=2.8, foreground='white')]

    for river in cfg.get('rivers', []):
        coords = river.get('coords', [])
        if len(coords) < 2:
            continue
        xs, ys = zip(*coords)
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=zorder,
                solid_capstyle='round', path_effects=_stroke)
        # Arrow on midpoint of selected segments
        for i in range(0, len(coords) - 1, max(1, arrow_every)):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            dx, dy = x1 - x0, y1 - y0
            length = np.hypot(dx, dy)
            if length < 1e-10:
                continue
            mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
            off = 0.18  # fraction of segment length for arrow head span
            ax.annotate(
                '', xy=(mx + dx * off, my + dy * off),
                xytext=(mx - dx * off, my - dy * off),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.1,
                                mutation_scale=9),
                zorder=zorder + 1,
            )


_NE_STATES_SHP = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'data', 'ne_states', 'ne_50m_admin_1_states_provinces.shp',
)

CANADA_LAT = 49.0   # 49th parallel — US–Canada border


def canada_border(ax, cfg, zorder=9):
    """
    Overlay the 49°N US–Canada border on a map axis.

    Draws a solid dark line at 49°N, lightly hatches the Canadian area above
    it, and places a small 'CANADA' label. No-op when the map extent does not
    reach 49°N.
    """
    xmin, xmax, ymin, ymax = map_extent(cfg)
    if ymax <= CANADA_LAT + 0.02:
        return
    # Subtle wash + light hatching — intentionally low-opacity so the Canada
    # fringe doesn't dominate the map visually.
    ax.fill([xmin, xmax, xmax, xmin],
            [CANADA_LAT, CANADA_LAT, ymax, ymax],
            facecolor='white', alpha=0.15, linewidth=0, zorder=zorder - 2)
    ax.fill([xmin, xmax, xmax, xmin],
            [CANADA_LAT, CANADA_LAT, ymax, ymax],
            facecolor='none', edgecolor='#888888', linewidth=0,
            hatch='////', alpha=0.35, zorder=zorder - 1)
    # Border line
    ax.axhline(CANADA_LAT, color='#444444', lw=1.4, ls='-',
               zorder=zorder, alpha=0.80)
    # Labels — left-aligned so they don't compete with the centre of the map
    label_x = xmin + (xmax - xmin) * 0.04
    ax.text(label_x, CANADA_LAT + 0.014, 'CANADA',
            ha='left', va='bottom', fontsize=6.0, fontweight='bold',
            color='#444444', style='italic', zorder=zorder + 1,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.65))
    ax.text(label_x, CANADA_LAT - 0.014, 'WASHINGTON, USA',
            ha='left', va='top', fontsize=5.0, color='#444444',
            style='italic', zorder=zorder + 1,
            bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.55))


def locator_inset(fig, map_ax, cfg):
    """
    Add a small Washington State locator inset to a map figure.

    Draws the WA state outline (from Natural Earth 1:50m) with the study-area
    bbox shown as a red rectangle. Placed at the lower-right corner of map_ax.
    Silently no-ops when the states shapefile is absent.
    """
    import os as _os
    if not _os.path.exists(_NE_STATES_SHP):
        return
    try:
        import geopandas as _gpd
        import matplotlib.patches as _mp
        states = _gpd.read_file(_NE_STATES_SHP)
        wa = states[states['name'] == 'Washington']
        if len(wa) == 0:
            return

        # Position inset at lower-right of map_ax, raised enough to clear
        # scale bars (which sit at y≈0.05–0.12 of the axes height).
        pos = map_ax.get_position()
        iw = pos.width  * 0.18   # smaller: was 0.27
        ih = pos.height * 0.15   # smaller: was 0.24
        ix = pos.x0 + pos.width  - iw - pos.width  * 0.008
        iy = pos.y0              + pos.height * 0.17   # raised: was 0.005
        ax_i = fig.add_axes([ix, iy, iw, ih])

        ax_i.set_facecolor('#c8dff0')          # ocean/water tint
        wa.plot(ax=ax_i, color='#e2ddd5', edgecolor='#555555', linewidth=0.7)

        b = cfg['study_area']['bbox']
        ax_i.add_patch(_mp.Rectangle(
            (b['lon_min'], b['lat_min']),
            b['lon_max'] - b['lon_min'],
            b['lat_max'] - b['lat_min'],
            linewidth=1.8, edgecolor='#cc0000',
            facecolor='#cc0000', alpha=0.35, zorder=5,
        ))
        # Study-area centroid dot
        cx = (b['lon_min'] + b['lon_max']) / 2
        cy = (b['lat_min'] + b['lat_max']) / 2
        ax_i.scatter([cx], [cy], s=18, color='#cc0000', zorder=6)

        ax_i.set_xlim(wa.total_bounds[0] - 0.3, wa.total_bounds[2] + 0.3)
        ax_i.set_ylim(wa.total_bounds[1] - 0.2, wa.total_bounds[3] + 0.3)
        ax_i.set_aspect('auto')
        ax_i.set_xticks([]); ax_i.set_yticks([])
        for sp in ax_i.spines.values():
            sp.set_linewidth(0.9); sp.set_color('#555555')
        ax_i.set_title('study\narea', fontsize=5, pad=2, color='#333333')
    except Exception:
        pass  # never crash a figure because of a missing inset


def clip_gdf_to_map(gdf, cfg):
    """Clip a GeoDataFrame to the canonical map_extent for this study area.

    Always call this before gdf.plot(ax=ax, ...) on any map panel.
    Returns a copy with geometries intersected to the padded study-area bbox.
    Silently returns the original gdf if shapely is unavailable.
    """
    try:
        from shapely.geometry import box as _box
        xmin, xmax, ymin, ymax = map_extent(cfg)
        clip = _box(xmin, ymin, xmax, ymax)
        out_gdf = gdf[gdf.intersects(clip)].copy()
        out_gdf['geometry'] = out_gdf['geometry'].intersection(clip)
        return out_gdf
    except Exception:
        return gdf
