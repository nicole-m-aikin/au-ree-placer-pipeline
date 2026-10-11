"""Operations tests: structured logs, metrics, rate limit. No extra ML."""

import json
import logging
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from api.ops import CHEMISTRY_FIELDS, request_event
from test_api import _VALID, _tiny_artifacts


@pytest.fixture
def client(tmp_path):
    _tiny_artifacts(tmp_path)
    from api.app import create_app
    app = create_app(model_dir=tmp_path)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def limited_client(tmp_path, monkeypatch):
    monkeypatch.setenv('PLACER_RATE_LIMIT', '2')
    monkeypatch.setenv('PLACER_RATE_WINDOW_S', '60')
    _tiny_artifacts(tmp_path)
    from api.app import create_app
    app = create_app(model_dir=tmp_path)
    with TestClient(app) as c:
        yield c


class TestStructuredLog:
    def test_predict_log_has_ops_fields_and_no_chemistry(self, client, caplog):
        with caplog.at_level(logging.INFO, logger='placer.api'):
            r = client.post('/predict', json=_VALID)
        assert r.status_code == 200
        events = _json_events(caplog)
        predict = [e for e in events if e.get('endpoint') == '/predict']
        assert predict
        event = predict[-1]
        assert set(event) >= {
            'timestamp', 'model_version', 'endpoint', 'status',
            'latency_ms', 'feature_validation',
        }
        assert event['status'] == 200
        assert event['feature_validation'] == 'passed'
        assert event['model_version'] == 'task9_rf_placer_gold'
        assert CHEMISTRY_FIELDS.isdisjoint(event)
        text = caplog.text
        assert '0.004' not in text
        assert '"Th"' not in text

    def test_malformed_predict_is_validation_failed(self, client, caplog):
        bad = dict(_VALID)
        bad['Th'] = 'hot'
        with caplog.at_level(logging.INFO, logger='placer.api'):
            r = client.post('/predict', json=bad)
        assert r.status_code == 422
        events = [e for e in _json_events(caplog) if e.get('endpoint') == '/predict']
        assert events
        assert events[-1]['feature_validation'] == 'failed'
        assert CHEMISTRY_FIELDS.isdisjoint(events[-1])

    def test_health_log_does_not_claim_feature_validation(self, client, caplog):
        with caplog.at_level(logging.INFO, logger='placer.api'):
            client.get('/health')
        events = [e for e in _json_events(caplog) if e.get('endpoint') == '/health']
        assert events
        assert events[-1]['feature_validation'] == 'n/a'


class TestMetrics:
    def test_metrics_tracks_requests_and_model_version(self, client):
        client.get('/health')
        client.post('/predict', json=_VALID)
        bad = dict(_VALID)
        del bad['Th']
        client.post('/predict', json=bad)
        r = client.get('/metrics')
        assert r.status_code == 200
        body = r.json()
        assert body['model_version'] == 'task9_rf_placer_gold'
        assert body['request_count'] >= 3
        assert body['malformed_request_count'] >= 1
        assert body['malformed_request_rate'] > 0
        assert body['error_count'] == 0
        assert body['latency_ms']['n'] >= 3
        assert body['latency_ms']['p50'] is not None
        assert body['latency_ms']['p95'] is not None
        assert body['endpoints']['/predict'] >= 2
        assert body['endpoints']['/health'] >= 1
        assert '/metrics' not in body['endpoints']

    def test_health_failure_is_counted(self, tmp_path):
        from api.app import create_app
        app = create_app(model_dir=tmp_path)
        with TestClient(app) as c:
            r = c.get('/health')
            assert r.status_code == 503
            metrics = c.get('/metrics').json()
        assert metrics['health_failure_count'] >= 1
        assert metrics['error_count'] >= 1
        assert metrics['error_rate'] > 0


class TestRateLimit:
    def test_third_predict_is_429(self, limited_client):
        assert limited_client.post('/predict', json=_VALID).status_code == 200
        assert limited_client.post('/predict', json=_VALID).status_code == 200
        r = limited_client.post('/predict', json=_VALID)
        assert r.status_code == 429
        assert r.json()['detail'] == 'rate limit exceeded'
        metrics = limited_client.get('/metrics').json()
        assert metrics['rate_limited_count'] >= 1

    def test_health_is_not_rate_limited(self, limited_client):
        for _ in range(6):
            assert limited_client.get('/health').status_code == 200


def test_request_event_schema_matches_the_contract():
    event = request_event(
        model_version='task9_rf_placer_gold',
        endpoint='/predict',
        status=200,
        latency_ms=31.2,
    )
    assert event['endpoint'] == '/predict'
    assert event['status'] == 200
    assert event['latency_ms'] == 31.2
    assert event['feature_validation'] == 'passed'
    assert CHEMISTRY_FIELDS.isdisjoint(event)


def _json_events(caplog):
    events = []
    for rec in caplog.records:
        if rec.name != 'placer.api':
            continue
        try:
            events.append(json.loads(rec.getMessage()))
        except json.JSONDecodeError:
            continue
    return events
