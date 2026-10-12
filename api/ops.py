"""Runtime operations for the doorbell: logs, metrics, rate limit.

Application logs are one JSON object per request. They record whether
feature validation passed — not the chemistry a person submitted.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from threading import Lock

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

LOGGER = logging.getLogger('placer.api')

# Never put these on a log line. They are the request body.
CHEMISTRY_FIELDS = frozenset({
    'Th', 'Ce', 'La', 'P', 'U', 'Au', 'As', 'Ti', 'Fe', 'Zr', 'Y',
    'fe_unit', 'lon', 'lat',
})

RATE_EXEMPT_PATHS = frozenset({'/health', '/metrics'})
METRICS_SKIP_PATHS = frozenset({'/metrics'})

LogEvent = dict[str, object]


def configure_logging() -> None:
    """Write one bare JSON object per request to stderr.

    Uvicorn does not attach a handler to this logger. The stdlib last-resort
    handler only prints WARNING and above, and it prefixes the line, so
    ``docker logs`` would miss the JSON object the integration check reads.
    """
    LOGGER.setLevel(logging.INFO)
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(message)s'))
        LOGGER.addHandler(handler)


def client_ip(request: Request) -> str:
    forwarded = request.headers.get('x-forwarded-for')
    if forwarded:
        return forwarded.split(',')[0].strip() or 'unknown'
    if request.client is not None and request.client.host:
        return request.client.host
    return 'unknown'


def normalize_path(path: str) -> str:
    if path != '/' and path.endswith('/'):
        return path.rstrip('/')
    return path


def feature_validation_for(path: str, status: int) -> str:
    if path != '/predict':
        return 'n/a'
    if status == 200:
        return 'passed'
    if status == 422:
        return 'failed'
    return 'n/a'


def request_event(
    *,
    model_version: str,
    endpoint: str,
    status: int,
    latency_ms: float,
) -> LogEvent:
    return {
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        'model_version': model_version,
        'endpoint': endpoint,
        'status': status,
        'latency_ms': round(float(latency_ms), 1),
        'feature_validation': feature_validation_for(endpoint, status),
    }


def assert_no_chemistry(event: LogEvent) -> None:
    leaked = CHEMISTRY_FIELDS.intersection(event)
    if leaked:
        raise ValueError(f'structured log contains chemistry fields: {sorted(leaked)}')


def emit_request_log(event: LogEvent) -> None:
    assert_no_chemistry(event)
    LOGGER.info(json.dumps(event, separators=(',', ':')))


class RateLimiter:
    """Per-key sliding window. ``limit <= 0`` disables."""

    def __init__(self, limit: int, window_s: float) -> None:
        self.limit = int(limit)
        self.window_s = float(window_s)
        self._hits: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        if self.limit <= 0:
            return True, 0
        now = time.monotonic()
        cutoff = now - self.window_s
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.limit:
                retry = int(self.window_s - (now - q[0])) + 1
                return False, max(retry, 1)
            q.append(now)
            return True, 0


def rate_limit_from_env() -> RateLimiter:
    return RateLimiter(
        limit=int(os.environ.get('PLACER_RATE_LIMIT', '60')),
        window_s=float(os.environ.get('PLACER_RATE_WINDOW_S', '60')),
    )


class MetricsRegistry:
    """In-process counters for the public demo. One instance per app."""

    def __init__(self, latency_window: int = 1000) -> None:
        self._lock = Lock()
        self._request_count = 0
        self._error_count = 0
        self._malformed_count = 0
        self._health_failure_count = 0
        self._rate_limited_count = 0
        self._latencies: deque[float] = deque(maxlen=int(latency_window))
        self._by_endpoint: dict[str, int] = defaultdict(int)
        self.model_version = 'unloaded'

    def set_model_version(self, model_version: str | None) -> None:
        self.model_version = model_version or 'unloaded'

    def record(self, endpoint: str, status: int, latency_ms: float) -> None:
        if endpoint in METRICS_SKIP_PATHS:
            return
        with self._lock:
            self._request_count += 1
            self._by_endpoint[endpoint] += 1
            self._latencies.append(float(latency_ms))
            if status >= 500:
                self._error_count += 1
            if status == 422:
                self._malformed_count += 1
            if status == 429:
                self._rate_limited_count += 1
            if endpoint == '/health' and status >= 500:
                self._health_failure_count += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            n = self._request_count
            errors = self._error_count
            malformed = self._malformed_count
            lat = list(self._latencies)
            by_ep = dict(self._by_endpoint)
            health_fail = self._health_failure_count
            limited = self._rate_limited_count
        return {
            'model_version': self.model_version,
            'request_count': n,
            'error_count': errors,
            'error_rate': (errors / n) if n else 0.0,
            'malformed_request_count': malformed,
            'malformed_request_rate': (malformed / n) if n else 0.0,
            'health_failure_count': health_fail,
            'rate_limited_count': limited,
            'latency_ms': {
                'p50': _percentile(lat, 50),
                'p95': _percentile(lat, 95),
                'n': len(lat),
            },
            'endpoints': by_ep,
        }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    rank = (q / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return round(ordered[lo] * (1.0 - frac) + ordered[hi] * frac, 1)


async def ops_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    app = request.app
    metrics: MetricsRegistry = app.state.metrics
    limiter: RateLimiter = app.state.rate_limiter
    path = normalize_path(request.url.path)
    started = time.perf_counter()

    if path not in RATE_EXEMPT_PATHS:
        allowed, retry_after = limiter.allow(client_ip(request))
        if not allowed:
            latency_ms = (time.perf_counter() - started) * 1000.0
            event = request_event(
                model_version=metrics.model_version,
                endpoint=path,
                status=429,
                latency_ms=latency_ms,
            )
            emit_request_log(event)
            metrics.record(path, 429, latency_ms)
            return JSONResponse(
                status_code=429,
                content={
                    'detail': 'rate limit exceeded',
                    'retry_after_s': retry_after,
                },
                headers={'Retry-After': str(retry_after)},
            )

    response = await call_next(request)
    latency_ms = (time.perf_counter() - started) * 1000.0
    event = request_event(
        model_version=metrics.model_version,
        endpoint=path,
        status=response.status_code,
        latency_ms=latency_ms,
    )
    emit_request_log(event)
    metrics.record(path, response.status_code, latency_ms)
    return response
