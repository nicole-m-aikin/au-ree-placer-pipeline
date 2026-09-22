"""Fetch PAD-US + BLM MLRS claim polygons for a study-area bbox.

Usage:
  python -m pipeline.fetch_land_access --config configs/california_sierra/config.yaml
  python -m pipeline.fetch_land_access --config configs/california_sierra/config.yaml --force
"""

from __future__ import annotations

import argparse

import yaml

from pipeline.task11_land_access import ensure_land_access_data, land_access_enabled


def main():
    parser = argparse.ArgumentParser(
        description='Fetch PAD-US + MLRS land-access layers for a config bbox',
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if not land_access_enabled(cfg):
        # Allow one-shot fetch even if the flag is off — set it for this run.
        cfg.setdefault('task11', {})['land_access'] = True
        print('  (land_access was false in config; enabling for this fetch)', flush=True)
    padus, mlrs = ensure_land_access_data(cfg, force=args.force)
    print(f'PAD-US: {padus}')
    print(f'MLRS:   {mlrs}')


if __name__ == '__main__':
    main()
