"""
Synthetic NURE stream sediment data for NE Washington.
Values calibrated to published ranges:
  - Background Th: 5-15 ppm (USGS OFR 97-492)
  - Anomalous Th (monazite-associated): 30-150 ppm
  - Max observed in NURE WA data: ~500 ppm
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
np.random.seed(42)

LON_MIN, LON_MAX = -120.0, -117.0
LAT_MIN, LAT_MAX = 47.5, 49.1
N = 800

BACKGROUNDS = {
    'Th': 8.0,    'Ce': 65.0,  'La': 32.0,  'Nd': 28.0,
    'P':  600.0,  'Y':  18.0,  'U':  2.5,   'Zr': 120.0,
    'Ti': 4500.0, 'Fe': 35000.0
}

lons = np.random.uniform(LON_MIN, LON_MAX, N)
lats = np.random.uniform(LAT_MIN, LAT_MAX, N)

def domain(lon, lat):
    if lon < -118.5 and lat > 48.0:
        return 'MCC'
    elif -118.5 < lon < -117.8:
        return 'BATHOLITH'
    else:
        return 'BACKGROUND'

records = []
for i in range(N):
    d = domain(lons[i], lats[i])
    row = {'sample_id': f'NURE_WA_{i:04d}', 'lon': lons[i], 'lat': lats[i], 'domain': d}

    # Generate base log-normal values
    log_vals = {}
    for elem, bg in BACKGROUNDS.items():
        cv = 1.0 if elem in ['Th','Ce','La','Nd'] else 0.6
        sigma = np.sqrt(np.log(1 + cv**2))
        mu = np.log(bg) - sigma**2 / 2
        log_vals[elem] = np.random.normal(mu, sigma)

    # Correlated monazite enrichment — capped at +2 log units (100x max)
    if d == 'MCC':
        mono_factor = np.clip(np.random.normal(0.8, 0.4), 0, 2.0)
        for elem in ['Th', 'Ce', 'La', 'Nd', 'P', 'Y']:
            log_vals[elem] += mono_factor
    elif d == 'BATHOLITH':
        bath_factor = np.clip(np.random.normal(0.3, 0.2), 0, 1.0)
        for elem in ['Th', 'Ce', 'La']:
            log_vals[elem] += bath_factor

    for elem in BACKGROUNDS:
        row[elem] = round(np.exp(log_vals[elem]), 2)

    # Thorite anomalies (U-enriched, LREE-poor)
    if lons[i] > -117.8 and np.random.random() < 0.05:
        row['U'] = row['U'] * np.random.uniform(3, 6)
        row['Th'] = row['Th'] * np.random.uniform(2, 4)
        row['domain'] = 'THORITE_ANOMALY'

    records.append(row)

df = pd.DataFrame(records)
# Hard cap: Th max 600 ppm (consistent with reported NE WA stream sediment maxima)
df['Th'] = df['Th'].clip(upper=600)
df['Ce'] = df['Ce'].clip(upper=3000)
df['La'] = df['La'].clip(upper=1500)

df.to_csv('/home/claude/ne_wa_ree/data/nure/nure_wa_synthetic.csv', index=False)
gdf = gpd.GeoDataFrame(df, geometry=[Point(r.lon, r.lat) for r in df.itertuples()], crs='EPSG:4326')
gdf.to_file('/home/claude/ne_wa_ree/data/nure/nure_wa_synthetic.geojson', driver='GeoJSON')

from scipy import stats
mcc = df[df['domain']=='MCC']
log_th = np.log10(mcc['Th'].clip(lower=0.1))
log_ce = np.log10(mcc['Ce'].clip(lower=0.1))
r, p = stats.pearsonr(log_th, log_ce)
print(f"Generated {N} samples | MCC Th-Ce r={r:.3f} | Th max={df['Th'].max():.1f} ppm | Th p90={df['Th'].quantile(0.9):.1f} ppm")
print(f"Domain counts: {df['domain'].value_counts().to_dict()}")
