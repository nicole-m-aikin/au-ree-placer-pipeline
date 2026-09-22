"""In-belt hold-out helpers — no NURE download required."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.task13_holdout import lon_cut


def test_lon_cut_defaults_to_bbox_midpoint():
    cfg = {
        'study_area': {
            'bbox': {
                'lon_min': -120.0,
                'lon_max': -117.0,
                'lat_min': 47.5,
                'lat_max': 49.1,
            }
        }
    }
    assert lon_cut(cfg) == -118.5


def test_lon_cut_honors_explicit_kettle_line():
    cfg = {
        'study_area': {
            'bbox': {
                'lon_min': -120.0,
                'lon_max': -117.0,
                'lat_min': 47.5,
                'lat_max': 49.1,
            }
        },
        'holdout': {'lon_cut': -118.5},
    }
    assert lon_cut(cfg) == -118.5
