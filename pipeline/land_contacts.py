"""Local office contacts for California pan pins.

Who to call about conditions, closures, and what tools are allowed on that
ground. Numbers are the published field-office and forest phones. Claim
holders are not listed; for a private claim the BLM office that keeps the
case file is the contact.
"""

from __future__ import annotations

# County → BLM field office (California). Source: BLM California office pages.
BLM_BY_COUNTY = {
    'Alameda': ('Central Coast Field Office', '831-582-2200', 'BLM_CA_Web_CC@blm.gov'),
    'Alpine': ('Mother Lode Field Office', '916-941-3101', 'BLM_CA_Web_ML@blm.gov'),
    'Amador': ('Mother Lode Field Office', '916-941-3101', 'BLM_CA_Web_ML@blm.gov'),
    'Butte': ('Redding Field Office', '530-224-2100', 'BLM_CA_Web_RE@blm.gov'),
    'Calaveras': ('Mother Lode Field Office', '916-941-3101', 'BLM_CA_Web_ML@blm.gov'),
    'Colusa': ('Ukiah Field Office', '707-468-4000', 'BLM_CA_Web_UK@blm.gov'),
    'Contra Costa': ('Central Coast Field Office', '831-582-2200', 'BLM_CA_Web_CC@blm.gov'),
    'Del Norte': ('Arcata Field Office', '707-825-2300', 'BLM_CA_Web_AR@blm.gov'),
    'El Dorado': ('Mother Lode Field Office', '916-941-3101', 'BLM_CA_Web_ML@blm.gov'),
    'Fresno': ('Bakersfield Field Office', '661-391-6000', 'BLM_CA_Web_BK@blm.gov'),
    'Glenn': ('Redding Field Office', '530-224-2100', 'BLM_CA_Web_RE@blm.gov'),
    'Humboldt': ('Arcata Field Office', '707-825-2300', 'BLM_CA_Web_AR@blm.gov'),
    'Imperial': ('El Centro Field Office', '760-337-4400', 'BLM_CA_Web_EC@blm.gov'),
    'Inyo': ('Bishop Field Office', '760-872-5000', 'BLM_CA_Web_BI@blm.gov'),
    'Kern': ('Bakersfield Field Office', '661-391-6000', 'BLM_CA_Web_BK@blm.gov'),
    'Kings': ('Bakersfield Field Office', '661-391-6000', 'BLM_CA_Web_BK@blm.gov'),
    'Lake': ('Ukiah Field Office', '707-468-4000', 'BLM_CA_Web_UK@blm.gov'),
    'Lassen': ('Eagle Lake Field Office', '530-257-0456', 'BLM_CA_Web_EL@blm.gov'),
    'Los Angeles': ('Palm Springs-South Coast Field Office', '760-833-7100', 'BLM_CA_Web_PS@blm.gov'),
    'Madera': ('Bakersfield Field Office', '661-391-6000', 'BLM_CA_Web_BK@blm.gov'),
    'Marin': ('Central Coast Field Office', '831-582-2200', 'BLM_CA_Web_CC@blm.gov'),
    'Mariposa': ('Mother Lode Field Office', '916-941-3101', 'BLM_CA_Web_ML@blm.gov'),
    'Mendocino': ('Ukiah Field Office', '707-468-4000', 'BLM_CA_Web_UK@blm.gov'),
    'Merced': ('Central Coast Field Office', '831-582-2200', 'BLM_CA_Web_CC@blm.gov'),
    'Modoc': ('Applegate Field Office', '530-233-4666', 'BLM_CA_Web_AP@blm.gov'),
    'Mono': ('Bishop Field Office', '760-872-5000', 'BLM_CA_Web_BI@blm.gov'),
    'Monterey': ('Central Coast Field Office', '831-582-2200', 'BLM_CA_Web_CC@blm.gov'),
    'Napa': ('Ukiah Field Office', '707-468-4000', 'BLM_CA_Web_UK@blm.gov'),
    'Nevada': ('Mother Lode Field Office', '916-941-3101', 'BLM_CA_Web_ML@blm.gov'),
    'Orange': ('Palm Springs-South Coast Field Office', '760-833-7100', 'BLM_CA_Web_PS@blm.gov'),
    'Placer': ('Mother Lode Field Office', '916-941-3101', 'BLM_CA_Web_ML@blm.gov'),
    'Plumas': ('Eagle Lake Field Office', '530-257-0456', 'BLM_CA_Web_EL@blm.gov'),
    'Riverside': ('Palm Springs-South Coast Field Office', '760-833-7100', 'BLM_CA_Web_PS@blm.gov'),
    'Sacramento': ('Mother Lode Field Office', '916-941-3101', 'BLM_CA_Web_ML@blm.gov'),
    'San Benito': ('Central Coast Field Office', '831-582-2200', 'BLM_CA_Web_CC@blm.gov'),
    'San Bernardino': ('Barstow Field Office', '760-252-6000', 'BLM_CA_Web_BA@blm.gov'),
    'San Diego': ('Palm Springs-South Coast Field Office', '760-833-7100', 'BLM_CA_Web_PS@blm.gov'),
    'San Joaquin': ('Central Coast Field Office', '831-582-2200', 'BLM_CA_Web_CC@blm.gov'),
    'San Luis Obispo': ('Central Coast Field Office', '831-582-2200', 'BLM_CA_Web_CC@blm.gov'),
    'San Mateo': ('Central Coast Field Office', '831-582-2200', 'BLM_CA_Web_CC@blm.gov'),
    'Santa Barbara': ('Central Coast Field Office', '831-582-2200', 'BLM_CA_Web_CC@blm.gov'),
    'Santa Clara': ('Central Coast Field Office', '831-582-2200', 'BLM_CA_Web_CC@blm.gov'),
    'Santa Cruz': ('Central Coast Field Office', '831-582-2200', 'BLM_CA_Web_CC@blm.gov'),
    'Shasta': ('Redding Field Office', '530-224-2100', 'BLM_CA_Web_RE@blm.gov'),
    'Sierra': ('Mother Lode Field Office', '916-941-3101', 'BLM_CA_Web_ML@blm.gov'),
    'Siskiyou': ('Redding Field Office', '530-224-2100', 'BLM_CA_Web_RE@blm.gov'),
    'Solano': ('Ukiah Field Office', '707-468-4000', 'BLM_CA_Web_UK@blm.gov'),
    'Sonoma': ('Ukiah Field Office', '707-468-4000', 'BLM_CA_Web_UK@blm.gov'),
    'Stanislaus': ('Central Coast Field Office', '831-582-2200', 'BLM_CA_Web_CC@blm.gov'),
    'Tehama': ('Redding Field Office', '530-224-2100', 'BLM_CA_Web_RE@blm.gov'),
    'Trinity': ('Redding Field Office', '530-224-2100', 'BLM_CA_Web_RE@blm.gov'),
    'Tulare': ('Bakersfield Field Office', '661-391-6000', 'BLM_CA_Web_BK@blm.gov'),
    'Tuolumne': ('Mother Lode Field Office', '916-941-3101', 'BLM_CA_Web_ML@blm.gov'),
    'Ventura': ('Central Coast Field Office', '831-582-2200', 'BLM_CA_Web_CC@blm.gov'),
    'Yuba': ('Mother Lode Field Office', '916-941-3101', 'BLM_CA_Web_ML@blm.gov'),
}

# County → National Forest supervisor's office (primary California forests).
USFS_BY_COUNTY = {
    'Alpine': ('Eldorado National Forest', '530-622-5061', 'https://www.fs.usda.gov/r05/eldorado'),
    'Amador': ('Eldorado National Forest', '530-622-5061', 'https://www.fs.usda.gov/r05/eldorado'),
    'Butte': ('Plumas National Forest', '530-283-2050', 'https://www.fs.usda.gov/r05/plumas'),
    'Calaveras': ('Stanislaus National Forest', '209-532-3671', 'https://www.fs.usda.gov/r05/stanislaus'),
    'Del Norte': ('Six Rivers National Forest', '707-442-1721', 'https://www.fs.usda.gov/r05/sixrivers'),
    'El Dorado': ('Eldorado National Forest', '530-622-5061', 'https://www.fs.usda.gov/r05/eldorado'),
    'Fresno': ('Sierra National Forest', '559-297-0706', 'https://www.fs.usda.gov/r05/sierra'),
    'Humboldt': ('Six Rivers National Forest', '707-442-1721', 'https://www.fs.usda.gov/r05/sixrivers'),
    'Inyo': ('Inyo National Forest', '760-873-2400', 'https://www.fs.usda.gov/r05/inyo'),
    'Kern': ('Sequoia National Forest', '559-784-1500', 'https://www.fs.usda.gov/r05/sequoia'),
    'Lake': ('Mendocino National Forest', '530-934-3316', 'https://www.fs.usda.gov/r05/mendocino'),
    'Lassen': ('Lassen National Forest', '530-257-2151', 'https://www.fs.usda.gov/r05/lassen'),
    'Los Angeles': ('Angeles National Forest', '626-574-1613', 'https://www.fs.usda.gov/r05/angeles'),
    'Madera': ('Sierra National Forest', '559-297-0706', 'https://www.fs.usda.gov/r05/sierra'),
    'Mariposa': ('Sierra National Forest', '559-297-0706', 'https://www.fs.usda.gov/r05/sierra'),
    'Mendocino': ('Mendocino National Forest', '530-934-3316', 'https://www.fs.usda.gov/r05/mendocino'),
    'Modoc': ('Modoc National Forest', '530-233-5811', 'https://www.fs.usda.gov/r05/modoc'),
    'Mono': ('Inyo National Forest', '760-873-2400', 'https://www.fs.usda.gov/r05/inyo'),
    'Monterey': ('Los Padres National Forest', '805-968-6640', 'https://www.fs.usda.gov/r05/lospadres'),
    'Nevada': ('Tahoe National Forest', '530-265-4531', 'https://www.fs.usda.gov/r05/tahoe'),
    'Orange': ('Cleveland National Forest', '858-673-6180', 'https://www.fs.usda.gov/r05/cleveland'),
    'Placer': ('Tahoe National Forest', '530-265-4531', 'https://www.fs.usda.gov/r05/tahoe'),
    'Plumas': ('Plumas National Forest', '530-283-2050', 'https://www.fs.usda.gov/r05/plumas'),
    'Riverside': ('San Bernardino National Forest', '909-382-2600', 'https://www.fs.usda.gov/r05/sanbernardino'),
    'San Bernardino': ('San Bernardino National Forest', '909-382-2600', 'https://www.fs.usda.gov/r05/sanbernardino'),
    'San Diego': ('Cleveland National Forest', '858-673-6180', 'https://www.fs.usda.gov/r05/cleveland'),
    'San Luis Obispo': ('Los Padres National Forest', '805-968-6640', 'https://www.fs.usda.gov/r05/lospadres'),
    'Santa Barbara': ('Los Padres National Forest', '805-968-6640', 'https://www.fs.usda.gov/r05/lospadres'),
    'Shasta': ('Shasta-Trinity National Forest', '530-226-2500', 'https://www.fs.usda.gov/r05/shastatrinity'),
    'Sierra': ('Tahoe National Forest', '530-265-4531', 'https://www.fs.usda.gov/r05/tahoe'),
    'Siskiyou': ('Klamath National Forest', '530-842-6131', 'https://www.fs.usda.gov/r05/klamath'),
    'Tehama': ('Mendocino National Forest', '530-934-3316', 'https://www.fs.usda.gov/r05/mendocino'),
    'Trinity': ('Shasta-Trinity National Forest', '530-226-2500', 'https://www.fs.usda.gov/r05/shastatrinity'),
    'Tulare': ('Sequoia National Forest', '559-784-1500', 'https://www.fs.usda.gov/r05/sequoia'),
    'Tuolumne': ('Stanislaus National Forest', '209-532-3671', 'https://www.fs.usda.gov/r05/stanislaus'),
    'Ventura': ('Los Padres National Forest', '805-968-6640', 'https://www.fs.usda.gov/r05/lospadres'),
    'Yuba': ('Tahoe National Forest', '530-265-4531', 'https://www.fs.usda.gov/r05/tahoe'),
}

# Desert San Bernardino claims are usually Barstow / Ridgecrest BLM, not the forest.
DESERT_BLM = {
    'San Bernardino': ('Barstow Field Office', '760-252-6000', 'BLM_CA_Web_BA@blm.gov'),
    'Kern': ('Ridgecrest Field Office', '760-384-5400', 'BLM_CA_Web_RI@blm.gov'),
}

STATE_PARK = (
    'California State Parks',
    '800-777-0369',
    'https://www.parks.ca.gov',
)


def _county(name) -> str:
    text = '' if name is None else str(name).strip()
    if not text or text.lower() == 'nan':
        return ''
    return text.title().replace(' County', '')


def contact_for(access_type='', county='', manager_name='') -> dict:
    """Return office, phone, email_or_url, and a one-line note."""
    county = _county(county)
    atype = str(access_type or '').lower()
    manager = str(manager_name or '').upper()

    if atype in ('state_park',) or manager in ('SPR',):
        office, phone, email = STATE_PARK
        return {
            'contact_office': office,
            'contact_phone': phone,
            'contact_email': email,
            'contact_note': 'Ask this park before you pan. Many state parks do not allow it.',
        }

    if atype in ('usfs',) or manager in ('USFS',):
        row = USFS_BY_COUNTY.get(county)
        if row:
            office, phone, email = row
            return {
                'contact_office': office,
                'contact_phone': phone,
                'contact_email': email,
                'contact_note': 'Ask the district ranger about local closures and Notice of Intent needs.',
            }

    if atype in ('blm', 'claimed') or manager in ('BLM',) or atype == 'claimed':
        row = BLM_BY_COUNTY.get(county)
        if row:
            office, phone, email = row
            note = (
                'Ask about casual-use limits, fire restrictions, and claim status.'
                if atype != 'claimed'
                else 'Claim-holder permission is required. This BLM office keeps the case file.'
            )
            return {
                'contact_office': office,
                'contact_phone': phone,
                'contact_email': email,
                'contact_note': note,
            }

    if atype in ('nps',) or manager in ('NPS',):
        return {
            'contact_office': 'National Park Service',
            'contact_phone': '',
            'contact_email': 'https://www.nps.gov',
            'contact_note': 'Panning is usually not allowed in national parks. Call the park.',
        }

    return {
        'contact_office': '',
        'contact_phone': '',
        'contact_email': '',
        'contact_note': 'Confirm land status and local rules before you go.',
    }
