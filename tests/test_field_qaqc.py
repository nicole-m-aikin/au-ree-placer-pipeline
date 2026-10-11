"""Field QA/QC gate on synthetic batches (values are made up, not certificates)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.field_qaqc import (
    ELEMENTS,
    accepted_primaries,
    check_crms,
    check_duplicates,
    check_insertion_rates,
    hard,
    load_crm_values,
    load_results,
    main,
    parse_value,
    run_qaqc,
)

CRM = pd.DataFrame({
    'crm_id': ['TEST-CRM-1', 'TEST-CRM-1'],
    'element': ['Au_ppm', 'Ce_ppm'],
    'certified_value': [0.50, 60.0],
    'sd': [0.02, 3.0],
})
REE = {'Th_ppm': 12.0, 'Ce_ppm': 80.0, 'La_ppm': 40.0, 'Nd_ppm': 35.0,
       'Pr_ppm': 9.0, 'Y_ppm': 25.0, 'Zr_ppm': 300.0, 'Ti_ppm': 4000.0, 'Fe_pct': 4.0}


def _sample(sid, qc_type='primary', batch='B1', parent='', crm_id='', au=0.2, **els):
    row = {
        'sample_id': sid, 'parent_id': parent, 'qc_type': qc_type, 'crm_id': crm_id,
        'site_id': 'S1', 'block_id': '7', 'lon': -120.78, 'lat': 38.60,
        'gps_accuracy_m': 4.0, 'sample_medium': 'stream_sediment_-80',
        'collected_by': 'NA', 'date_collected': '2026-10-01', 'lab': 'LabA',
        'certificate_id': f'CERT-{batch}', 'method': 'FA-AA; LiBO2-ICPMS',
        'batch_id': batch, 'Au_ppm': au,
    }
    row.update(REE)
    row.update(els)
    return row


def _batch(batch='B1', n=20, blank_au='<0.005', crm_au=0.50, dup_au=None,
           n_blank=1, n_crm=1, n_dup=1):
    rng = np.random.default_rng(0)
    rows = [
        _sample(f'{batch}-{i:03d}', batch=batch, au=float(rng.uniform(0.1, 0.5)))
        for i in range(n)
    ]
    for k in range(n_blank):
        rows.append(_sample(
            f'{batch}-BLK{k}', 'blank', batch, au=blank_au,
            **{el: '<0.1' for el in ('Th_ppm', 'Ce_ppm', 'La_ppm', 'Nd_ppm',
                                    'Pr_ppm', 'Y_ppm')},
        ))
    for k in range(n_crm):
        rows.append(_sample(f'{batch}-CRM{k}', 'crm', batch, crm_id='TEST-CRM-1',
                            au=crm_au, Ce_ppm=60.5))
    for k in range(n_dup):
        parent = rows[k]
        au = parent['Au_ppm'] * 1.05 if dup_au is None else dup_au
        rows.append(_sample(f'{batch}-FD{k}', 'field_duplicate', batch,
                            parent=parent['sample_id'], au=au))
    return rows


MDL = {'Au_ppm': 0.005, 'Th_ppm': 0.1, 'Ce_ppm': 0.1, 'La_ppm': 0.1,
       'Nd_ppm': 0.1, 'Pr_ppm': 0.1, 'Y_ppm': 0.1}


def _status(rows, **kw):
    kw.setdefault('mdl', MDL)
    _, checks, batches = run_qaqc(pd.DataFrame(rows), CRM, **kw)
    return batches.set_index('batch_id'), checks


def test_parse_value_censoring():
    assert parse_value('<0.005') == (0.005, '<', 0.005)
    v, flag, dl = parse_value('-0.01')
    assert flag == '<' and v == pytest.approx(0.01) and dl == pytest.approx(0.01)
    assert parse_value('>10')[:2] == (10.0, '>')
    assert np.isnan(parse_value('n.a.')[0])
    assert parse_value('1.25')[:2] == (1.25, '')
    df = load_results(pd.DataFrame([_sample('X', au='<0.005')]))
    assert df['Au_ppm'].iloc[0] == 0.005
    assert df['Au_ppm_flag'].iloc[0] == '<'
    for el in ELEMENTS:
        assert el in df.columns


def test_unknown_qc_type_raises():
    with pytest.raises(ValueError):
        load_results(pd.DataFrame([_sample('X', qc_type='std')]))


def test_clean_batch_passes_and_primaries_accepted():
    rows = _batch()
    batches, _ = _status(rows)
    assert batches.loc['B1', 'status'] == 'pass', batches.loc['B1', 'reasons']
    acc = accepted_primaries(pd.DataFrame(rows), crm_values=CRM)
    assert len(acc) == 20
    assert set(acc['qc_type']) == {'primary'}


def test_blank_contamination_fails():
    batches, checks = _status(_batch(blank_au=0.2))
    assert batches.loc['B1', 'status'] == 'fail'
    bad = checks[(checks['check'] == 'blank') & (checks['status'] == 'fail')]
    assert list(bad['element']) == ['Au_ppm']
    assert len(accepted_primaries(pd.DataFrame(_batch(blank_au=0.2)), CRM, mdl=MDL)) == 0


def test_detected_blank_without_mdl_warns():
    batches, checks = _status(_batch(blank_au=0.2), mdl={})
    assert batches.loc['B1', 'status'] == 'warn'
    assert 'no MDL' in batches.loc['B1', 'reasons']


def test_blank_configured_limit():
    rows = _batch(blank_au=0.02)
    assert _status(rows)[0].loc['B1', 'status'] == 'pass'
    batches, _ = _status(rows, blank_limits={'Au_ppm': 0.01})
    assert batches.loc['B1', 'status'] == 'fail'


def test_crm_warn_and_fail():
    warn, _ = _status(_batch(crm_au=0.50 + 2.5 * 0.02))
    assert warn.loc['B1', 'status'] == 'warn'
    fail, checks = _status(_batch(crm_au=0.50 + 4 * 0.02))
    assert fail.loc['B1', 'status'] == 'fail'
    z = checks[(checks['check'] == 'crm') & (checks['element'] == 'Au_ppm')]['metric']
    assert float(z.iloc[0]) == pytest.approx(4.0)
    # Warn batches are held back unless allow_warn
    rows = _batch(crm_au=0.55)
    assert len(accepted_primaries(pd.DataFrame(rows), CRM)) == 0
    assert len(accepted_primaries(pd.DataFrame(rows), CRM, allow_warn=True)) == 20


def test_crm_not_in_table_warns():
    df = load_results(pd.DataFrame(_batch()))
    got = check_crms(df, CRM.iloc[0:0])
    assert set(got['status']) == {'warn'}


def test_duplicate_near_detection_is_skipped():
    rows = _batch(dup_au=0.03)
    rows[0]['Au_ppm'] = 0.02  # pair mean 0.025 < 10 × MDL (0.005)
    d = check_duplicates(load_results(pd.DataFrame(rows)))
    assert d.loc[d['element'] == 'Au_ppm', 'status'].iloc[0] == 'skip'
    assert _status(rows)[0].loc['B1', 'status'] == 'pass'


def test_duplicate_precision_fail():
    rows = _batch()
    rows[-1]['Au_ppm'] = rows[0]['Au_ppm'] * 3.0
    batches, checks = _status(rows)
    assert batches.loc['B1', 'status'] == 'fail'
    au = checks[(checks['check'] == 'field_duplicate') & (checks['element'] == 'Au_ppm')]
    assert au['status'].iloc[0] == 'fail'
    assert float(au['metric'].iloc[0]) == pytest.approx(50.0)


def test_duplicate_thresholds_are_parameters():
    rows = _batch()
    rows[-1]['Au_ppm'] = rows[0]['Au_ppm'] * 1.7  # HARD ≈ 25.9 %
    assert _status(rows)[0].loc['B1', 'status'] == 'pass'
    tight, _ = _status(rows, field_max_hard={'Au_ppm': 20.0, 'default': 30.0})
    assert tight.loc['B1', 'status'] == 'fail'
    assert hard(1.0, 1.0) == 0.0
    assert hard(3.0, 1.0) == pytest.approx(50.0)


def test_check_assay_counts_against_parent_batch():
    rows = _batch()
    parent = rows[0]
    rows.append(_sample('CHK-1', 'check_assay', 'LABB-1', parent=parent['sample_id'],
                        au=parent['Au_ppm'] * 3.0))
    _, checks = _status(rows)
    chk = checks[checks['check'] == 'check_assay']
    assert set(chk['batch_id']) == {'B1'}
    assert 'fail' in set(chk['status'])


def test_low_insertion_rate_warns_and_missing_type_fails():
    batches, checks = _status(_batch(n=30))
    assert batches.loc['B1', 'status'] == 'warn'
    ins = checks[checks['check'] == 'insertion_rate']
    assert set(ins['status']) == {'warn'}
    missing, _ = _status(_batch(n=20, n_blank=0))
    assert missing.loc['B1', 'status'] == 'fail'
    assert 'no blank inserted' in missing.loc['B1', 'reasons']
    df = load_results(pd.DataFrame(_batch()))
    proj = check_insertion_rates(df)
    assert proj.loc[proj['batch_id'] == 'ALL', 'status'].iloc[0] == 'warn'


def test_mixed_batches_only_passing_primaries_leave():
    rows = _batch('B1') + _batch('B2', blank_au=0.3)
    batches, _ = _status(rows)
    assert batches.loc['B1', 'status'] == 'pass'
    assert batches.loc['B2', 'status'] == 'fail'
    acc = accepted_primaries(pd.DataFrame(rows), CRM, mdl=MDL)
    assert set(acc['batch_id']) == {'B1'}
    assert len(acc) == 20


def test_crm_template_loads_empty():
    here = os.path.dirname(__file__)
    tmpl = os.path.join(here, '..', 'data', 'field_results', 'crm_values_template.csv')
    assert load_crm_values(tmpl).empty
    res = os.path.join(here, '..', 'data', 'field_results', 'field_results_template.csv')
    assert load_results(res).empty


def test_cli_writes_tables(tmp_path):
    res = tmp_path / 'results.csv'
    crm = tmp_path / 'crm.csv'
    pd.DataFrame(_batch('B1') + _batch('B2', crm_au=0.7)).to_csv(res, index=False)
    CRM.to_csv(crm, index=False)
    out = tmp_path / 'qaqc'
    assert main([
        '--results', str(res), '--crm', str(crm), '--out', str(out),
        '--mdl', 'Au_ppm=0.005,Ce=0.1',
    ]) == 0
    batches = pd.read_csv(out / 'field_qaqc_batches.csv')
    assert dict(zip(batches['batch_id'], batches['status'])) == {'B1': 'pass', 'B2': 'fail'}
    assert (out / 'field_qaqc_checks.csv').exists()
    acc = pd.read_csv(out / 'field_accepted_primaries.csv')
    assert set(acc['batch_id']) == {'B1'}
