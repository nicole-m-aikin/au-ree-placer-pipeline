# Data Download Instructions
## Run these commands locally before executing the analysis scripts

### NURE Stream Sediment Data (Task 3 - CRITICAL)
# Download from USGS NGDB:
# https://mrdata.usgs.gov/ngdb/sediment/
# Select: State = Washington, Elements = Th,Ce,La,Nd,P,Y,U,Zr,Ti,Fe
# Download as CSV, save to: data/nure/nure_wa_sediment.csv

### MRDS Mine Sites (Tasks 1, 2)
# https://mrdata.usgs.gov/mrds/
# Commodity filter: gold, placer gold, REE, monazite, magnetite, ilmenite
# Bounding box: -120.0,47.5,-117.0,49.1
# Download as GeoJSON, save to: data/mrds/mrds_ne_wa.geojson

### Aeromagnetic Data (Task 1)
# USGS Aeromagnetic Survey Compilation:
# https://mrdata.usgs.gov/magnetic/
# State-level grid for Washington available as netCDF or GeoTIFF
# Save to: data/aeromagnetic/wa_mag_anomaly.tif

### DEM for Catchment Analysis (Task 2)
# 3DEP 30m DEM via The National Map:
# https://apps.nationalmap.gov/downloader/
# AOI: -120.0,47.5,-117.0,49.1
# Save to: data/dem/ne_wa_dem_30m.tif

### Lidar DEMs (Task 4)
# Washington Lidar Portal: https://lidar.wa.gov
# Download highest-resolution available tiles for each mine site
# Save to: data/lidar/<site_name>_lidar.tif

### Historical Topos (Task 4)
# USGS TopoView: https://ngmdb.usgs.gov/topoview/
# Download 7.5-min quads for each site, 1950s-1970s vintage
# Save to: data/lidar/<site_name>_historical_topo.pdf

### USGS State Geologic Map (Task 2)
# https://ngmdb.usgs.gov/Prodesc/proddesc_9215.htm (WA state geologic map)
# Or GeMS format: https://www.usgs.gov/programs/national-cooperative-geologic-mapping-program
# Save to: data/geologic/wa_geology.geojson

### WGS OFR 2026-02 Mine Waste Supplement (Tasks 2, 3, 4, 8 — optional overlay)
# Washington Geological Survey Open-File Report 2026-02 (van Alderwerelt & Di Fiori, 2026)
# Free download: https://www.dnr.wa.gov/publications/ger_ofr2026-02_mine_waste_characterization_part_1.zip
# Extract the Excel data supplement and place at ONE of the following:
#   Option A (default): data/wgs_ofr2026/ger_ofr2026-02_data_supplement.xlsx
#   Option B (env var): export WGS_OFR2026_PATH=/path/to/ger_ofr2026-02_data_supplement.xlsx
# If the file is absent, Tasks 2/3/4/8 fall back gracefully (WGS overlay layer is skipped).
# Task 8 (fig9_mine_waste_ree.png) requires this file to produce any output.
