"""Small CRS helpers so Task 11 is not stuck in UTM 11N."""


def utm_epsg(lon, lat=None):
    """Northern-hemisphere UTM EPSG from a longitude (Idaho 11N, Sierra 10N)."""
    zone = int((float(lon) + 180.0) / 6.0) + 1
    zone = max(1, min(60, zone))
    south = lat is not None and float(lat) < 0
    return f'EPSG:{32700 + zone if south else 32600 + zone}'


def sgmc_state(cfg):
    """USGS SGMC state abbreviation for this study-area config."""
    explicit = (cfg.get('data') or {}).get('sgmc_state')
    if explicit:
        return str(explicit).upper()
    short = str((cfg.get('study_area') or {}).get('short') or '').lower()
    name = str((cfg.get('study_area') or {}).get('name') or '').lower()
    blob = f'{short} {name}'
    if 'idaho' in blob or short.startswith('id_'):
        return 'ID'
    if 'california' in blob or 'sierra' in blob or short.startswith('ca_'):
        return 'CA'
    if 'montana' in blob or short.startswith('mt_'):
        return 'MT'
    return 'WA'
