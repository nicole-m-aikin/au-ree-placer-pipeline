"""
CLI entry point for the Au+REE Placer Assessment Pipeline.

Usage:
  python pipeline/run_pipeline.py --config configs/ne_washington/config.yaml
  python pipeline/run_pipeline.py --config configs/ne_washington/config.yaml --tasks 1 3 7
  python pipeline/run_pipeline.py --config configs/ne_washington/config.yaml --list
"""

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

import yaml


TASKS = {
    1:  ('pipeline.task1_coplacer',      'Co-placer minerals & site map'),
    2:  ('pipeline.task2_lithology',     'Source lithology & drainage'),
    3:  ('pipeline.task3_geochemistry',  'Geochemical discrimination'),
    4:  ('pipeline.task4_volume',        'Volume & tonnage estimation (Monte Carlo)'),
    5:  ('pipeline.task5_economics',     'Break-even / economics'),
    6:  ('pipeline.task6_framework',     'Decision framework'),
    7:  ('pipeline.task7_pathfinder',    'Au/As pathfinder anomalies'),
    8:  ('pipeline.task8_mine_waste',    'Mine waste REE & critical minerals'),
    9:  ('pipeline.task9_ml_targeting',  'ML anomaly targeting (Random Forest)'),
    10: ('pipeline.integration',         'Integration & priority ranking'),
}

# Prerequisites for each task: DEPENDENCIES[n] = tasks that must complete before n
DEPENDENCIES = {
    1:  [3],             # task1 reads task3 geojson output
    2:  [],
    3:  [],
    4:  [],
    5:  [4],             # task5 reads task4 CSV
    6:  [],
    7:  [3, 1],          # task7 reads task3 + task1 geojsons
    8:  [],
    9:  [1],             # task9 uses NURE data; anomaly labels reference task1 thresholds
    10: [1, 2, 4, 5, 7],
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Make outputs_dir absolute if relative
    cfg_dir = Path(config_path).parent.parent.parent  # project root
    outputs_dir = Path(cfg.get('outputs_dir', 'outputs'))
    if not outputs_dir.is_absolute():
        # Resolve relative to project root (where run_pipeline.py lives)
        pass  # keep as-is; ensure_outputs() will create relative to cwd
    return cfg


def run_task(task_num: int, cfg: dict, verbose: bool = True) -> bool:
    module_path, description = TASKS[task_num]
    if verbose:
        print(f"\n{'='*60}")
        print(f"  TASK {task_num}: {description}")
        print(f"{'='*60}")
    t0 = time.time()
    try:
        mod = importlib.import_module(module_path)
        mod.run(cfg)
        elapsed = time.time() - t0
        if verbose:
            print(f"  [OK] Task {task_num} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  [ERROR] Task {task_num} failed after {elapsed:.1f}s")
        print(f"  {type(e).__name__}: {e}")
        if verbose:
            traceback.print_exc()
        return False


def resolve_run_order(requested: list[int]) -> list[int]:
    """Topological sort respecting dependencies within requested set."""
    order = []
    visited = set()

    def visit(n):
        if n in visited:
            return
        visited.add(n)
        for dep in DEPENDENCIES.get(n, []):
            if dep in requested and dep not in visited:
                visit(dep)
        order.append(n)

    for t in sorted(requested):
        visit(t)
    return order


def main():
    parser = argparse.ArgumentParser(description='Au+REE Placer Assessment Pipeline')
    parser.add_argument('--config', required=True, help='Path to study-area config.yaml')
    parser.add_argument('--tasks', nargs='+', type=int, default=None,
                        help='Task numbers to run (default: all). E.g. --tasks 1 3 7')
    parser.add_argument('--list', action='store_true', help='List tasks and exit')
    parser.add_argument('--no-deps', action='store_true',
                        help='Skip dependency resolution (run exactly what is requested)')
    parser.add_argument('--fail-fast', action='store_true',
                        help='Stop on first task failure')
    args = parser.parse_args()

    if args.list:
        print("\nAvailable tasks:")
        for num, (_, desc) in TASKS.items():
            deps = DEPENDENCIES.get(num, [])
            dep_str = f"  [requires: {deps}]" if deps else ""
            print(f"  {num:2d}. {desc}{dep_str}")
        return 0

    cfg = load_config(args.config)

    requested = list(TASKS.keys()) if args.tasks is None else args.tasks
    invalid   = [t for t in requested if t not in TASKS]
    if invalid:
        print(f"[ERROR] Unknown task numbers: {invalid}. Valid: {list(TASKS.keys())}")
        return 1

    if args.no_deps:
        run_order = sorted(requested)
    else:
        run_order = resolve_run_order(requested)

    study_name = cfg.get('study_area', {}).get('name', 'Unknown')
    print(f"\n{'='*60}")
    print(f"  Au+REE Placer Assessment Pipeline")
    print(f"  Study area: {study_name}")
    print(f"  Config:     {args.config}")
    print(f"  Outputs:    {cfg.get('outputs_dir', 'outputs/')}")
    print(f"  Tasks:      {run_order}")
    print(f"{'='*60}")

    t_start = time.time()
    results = {}
    for task_num in run_order:
        ok = run_task(task_num, cfg)
        results[task_num] = ok
        if not ok and args.fail_fast:
            print("\n[FAIL-FAST] Stopping after first failure.")
            break

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total:.1f}s")
    n_ok  = sum(v for v in results.values())
    n_err = sum(not v for v in results.values())
    print(f"  {n_ok}/{len(results)} tasks succeeded", end='')
    if n_err:
        failed = [k for k, v in results.items() if not v]
        print(f"  |  FAILED: {failed}", end='')
    print(f"\n{'='*60}\n")

    return 0 if n_err == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
