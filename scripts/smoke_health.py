#!/usr/bin/env python3
"""Smoke-test GET /health. Used by CI after the container starts and after deploy."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


def check_health(base: str) -> dict:
    url = base.rstrip('/') + '/health'
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            status = resp.status
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode() if exc.fp else ''
        raise SystemExit(f'health {exc.code} at {url}: {detail}') from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f'health unreachable at {url}: {exc}') from exc
    if status != 200 or body.get('status') != 'ok':
        raise SystemExit(f'health not ok at {url}: {body}')
    if not body.get('sklearn_match'):
        raise SystemExit(f'sklearn mismatch at {url}: {body}')
    if not body.get('model_loaded'):
        raise SystemExit(f'model not loaded at {url}: {body}')
    return body


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    base = argv[0] if argv else 'http://127.0.0.1:8000'
    body = check_health(base)
    print(
        f"health ok at {base.rstrip('/')}/health "
        f"sklearn={body.get('sklearn_version')} "
        f"training={body.get('sklearn_version_training')}"
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
