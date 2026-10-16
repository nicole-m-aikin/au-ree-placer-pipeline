"""Local office contacts for pan pins."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.land_contacts import contact_for


def test_blm_county_returns_field_office_phone():
    c = contact_for(access_type='blm', county='San Luis Obispo')
    assert c['contact_office'] == 'Central Coast Field Office'
    assert c['contact_phone'] == '831-582-2200'
    assert '@blm.gov' in c['contact_email']


def test_usfs_county_returns_forest_phone():
    c = contact_for(access_type='usfs', county='Plumas')
    assert 'Plumas' in c['contact_office']
    assert c['contact_phone'] == '530-283-2050'


def test_claimed_ground_points_to_blm_case_office():
    c = contact_for(access_type='claimed', county='Kern', manager_name='THREE R 2')
    assert 'Field Office' in c['contact_office']
    assert 'claim' in c['contact_note'].lower()


def test_state_park_routes_to_parks():
    c = contact_for(access_type='state_park', county='El Dorado')
    assert 'State Parks' in c['contact_office']
