"""Tests for prep_nure unit normalization, dedup, and pivot (synthetic long frames)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import prep_nure
from prep_nure import (
    canonical_unit, dedupe_bestvalue, normalize_units, pivot_bestvalue,
)


def _bv(rows):
    return pd.DataFrame(rows, columns=['lab_id', 'species', 'qvalue', 'unit'])


class TestCanonicalUnit:
    @pytest.mark.parametrize('raw, want', [
        ('ppm', 'ppm'), ('PPM ', 'ppm'), ('ug/g', 'ppm'), ('g/t', 'ppm'), ('mg/kg', 'ppm'),
        ('ppb', 'ppb'), ('ng/g', 'ppb'),
        ('pct', 'pct'), ('%', 'pct'), ('wt%', 'pct'),
    ])
    def test_aliases(self, raw, want):
        assert canonical_unit(raw) == want

    @pytest.mark.parametrize('raw', [None, np.nan, '', 'counts'])
    def test_unknown_is_none(self, raw):
        assert canonical_unit(raw) is None


class TestNormalizeUnits:
    def test_au_ppb_divided_by_1000(self):
        out = normalize_units(_bv([('A', 'Au', 25.0, 'ppb')]))
        assert out['qvalue'].iloc[0] == pytest.approx(0.025)
        assert out['unit_orig'].iloc[0] == 'ppb'
        assert bool(out['unit_ppm'].iloc[0])

    def test_g_per_t_is_ppm_not_ppb(self):
        out = normalize_units(_bv([('A', 'Au', 1.5, 'g/t')]))
        assert out['qvalue'].iloc[0] == pytest.approx(1.5)

    def test_below_detection_sign_kept(self):
        out = normalize_units(_bv([('A', 'Au', -5.0, 'ppb')]))
        assert out['qvalue'].iloc[0] == pytest.approx(-0.005)

    def test_non_precious_species_untouched(self):
        out = normalize_units(_bv([('A', 'P', 0.09, 'pct'), ('A', 'Th', 12.0, 'ppm')]))
        assert out['qvalue'].tolist() == pytest.approx([0.09, 12.0])

    def test_pct_only_with_flag(self):
        bv = _bv([('A', 'Au', 0.001, 'pct')])
        assert normalize_units(bv)['qvalue'].iloc[0] == pytest.approx(0.001)
        flagged = normalize_units(bv, convert_pct=True)['qvalue'].iloc[0]
        assert flagged == pytest.approx(10.0)

    def test_unknown_unit_left_as_reported(self):
        out = normalize_units(_bv([('A', 'Au', 7.0, None)]))
        assert out['qvalue'].iloc[0] == 7.0
        assert not bool(out['unit_ppm'].iloc[0])


class TestDedupe:
    def test_majority_unit_wins_over_file_order(self):
        bv = _bv([
            ('A', 'Au', 30.0, 'ppb'),   # minority unit, first in file
            ('A', 'Au', 0.03, 'ppm'),
            ('B', 'Au', 0.01, 'ppm'),
            ('C', 'Au', 0.02, 'ppm'),
        ])
        out, stats = dedupe_bestvalue(bv)
        a = out[out['lab_id'] == 'A']
        assert len(a) == 1 and a['unit'].iloc[0] == 'ppm'
        assert stats['dup_groups'] == 1
        assert stats['resolved_by_majority_unit'] == 1
        assert stats['resolved_by_file_order'] == 0
        assert stats['majority_unit']['Au'] == 'ppm'

    def test_equivalent_unit_spellings_count_together(self):
        bv = _bv([
            ('A', 'Au', 40.0, 'ppb'),
            ('A', 'Au', 0.04, 'ug/g'),
            ('B', 'Au', 0.01, 'ppm'),
            ('C', 'Au', 0.02, 'PPM'),
        ])
        out, _ = dedupe_bestvalue(bv)
        assert out.loc[out['lab_id'] == 'A', 'unit'].iloc[0] == 'ug/g'

    def test_same_unit_tie_keeps_first(self):
        bv = _bv([('A', 'Au', 0.1, 'ppm'), ('A', 'Au', 0.2, 'ppm')])
        out, stats = dedupe_bestvalue(bv)
        assert out['qvalue'].tolist() == [0.1]
        assert stats['resolved_by_file_order'] == 1

    def test_no_duplicates_no_change(self):
        bv = _bv([('A', 'Au', 0.1, 'ppm'), ('A', 'Th', 9.0, 'ppm')])
        out, stats = dedupe_bestvalue(bv)
        assert len(out) == 2 and stats['dup_groups'] == 0


class TestPivot:
    def test_wide_values_in_ppm_with_unit_columns(self):
        bv = _bv([
            ('A', 'Au', 20.0, 'ppb'),
            ('A', 'Th', 11.0, 'ppm'),
            ('B', 'Au', 0.05, 'ppm'),
            ('B', 'Th', 8.0, 'ppm'),
            ('B', 'Ag', 500.0, 'ppb'),
        ])
        wide, stats = pivot_bestvalue(bv)
        wide = wide.set_index('lab_id')
        assert wide.loc['A', 'Au'] == pytest.approx(0.02)
        assert wide.loc['B', 'Au'] == pytest.approx(0.05)
        assert wide.loc['B', 'Ag'] == pytest.approx(0.5)
        assert np.isnan(wide.loc['A', 'Ag'])
        assert wide.loc['A', 'Au_unit'] == 'ppb'
        assert wide.loc['B', 'Au_unit'] == 'ppm'
        assert wide.loc['A', 'Th_unit'] == 'ppm'
        assert stats['unconverted_ppb_prone_rows'] == 0

    def test_counts_unconverted_precious_rows(self):
        bv = _bv([('A', 'Au', 3.0, 'counts'), ('B', 'Au', 0.1, 'ppm')])
        _, stats = pivot_bestvalue(bv)
        assert stats['unconverted_ppb_prone_rows'] == 1

    def test_missing_unit_column(self):
        bv = pd.DataFrame({'lab_id': ['A'], 'species': ['Au'], 'qvalue': [0.1]})
        wide, stats = pivot_bestvalue(bv)
        assert wide['Au'].iloc[0] == pytest.approx(0.1)
        assert stats['unconverted_ppb_prone_rows'] == 1


def test_import_does_not_run_script():
    assert callable(prep_nure.main)
    assert 'Au' in prep_nure.PPB_PRONE
