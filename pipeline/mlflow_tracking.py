"""Log already-written Task 9 forests to MLflow. Does not retrain.

The science is finished in Task 9. This module reads the joblib + meta JSON
that persist already wrote and records them as experiment runs.

Production is the frozen NE Washington doorbell. A larger shuffled-CV
number on a sidecar is not a promotion criterion — California's 0.86
shuffled score is why the registry exists, not why it would ship.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pipeline.ml_artifacts import MODEL_DIR, load_transfers

EXPERIMENT_NAME = 'placer_lookalike'
REGISTERED_DOORBELL = 'task9_rf_placer_gold'

# Tag values the interview answer uses. MLflow registry aliases are separate.
STAGE_PRODUCTION = 'production'
STAGE_EXPERIMENTAL = 'experimental/local'

REGION_BY_SHORT = {
    'ne_wa': 'NE_WA',
    'ne_washington': 'NE_WA',
    'ca_sierra_placer': 'CA_Sierra',
    'id_batholith': 'ID_Batholith',
    'mt_placer': 'MT_Placer',
    'co_wet_mtns': 'CO_Wet_Mtns',
}

TRANSFER_METRIC_KEYS = {
    'Idaho Batholith': 'transfer_auc_id_batholith',
    'id_batholith': 'transfer_auc_id_batholith',
    'ca_sierra_placer': 'transfer_auc_ca_sierra',
    'mt_placer': 'transfer_auc_mt_placer',
    'co_wet_mtns': 'transfer_auc_co_wet_mtns',
    'nc_fall_zone': 'transfer_auc_nc_fall_zone',
}

PROMOTION_RULE = (
    'frozen_wa_doorbell_plus_spatial_and_transfer; '
    'never_promote_largest_shuffled_cv'
)

PROMOTION_NOTE = (
    'Production is the frozen NE Washington doorbell '
    '(task9_rf_placer_gold, region=NE_WA, stage=production). '
    'Sidecars stay stage=experimental/local. Do not promote a belt-local '
    'forest because shuffled CV is larger. California shuffled ~0.86 still '
    'has 0.4° block ~0.69 and does not beat Washington on interstate transfer.'
)


def is_doorbell_meta(meta: dict) -> bool:
    if meta.get('is_doorbell') is True:
        return True
    return meta.get('model_id') == REGISTERED_DOORBELL


def region_and_stage(meta: dict) -> tuple[str, str]:
    if is_doorbell_meta(meta):
        return 'NE_WA', STAGE_PRODUCTION
    short = str(meta.get('study_area_short') or '').strip()
    region = REGION_BY_SHORT.get(short, short or 'unknown')
    return region, STAGE_EXPERIMENTAL


def iter_persisted_forests(model_dir=None):
    """Yield (meta_path, model_path) for each persisted forest."""
    dest = Path(model_dir) if model_dir else MODEL_DIR
    for meta_path in sorted(dest.glob('task9_rf_placer_gold*.meta.json')):
        model_path = meta_path.with_name(
            meta_path.name.replace('.meta.json', '.joblib')
        )
        if model_path.exists():
            yield meta_path, model_path


def run_record_from_meta(meta: dict, transfers=None) -> dict:
    """Build the MLflow params / metrics / tags. No tracking server required."""
    region, stage = region_and_stage(meta)
    features = list(meta.get('feature_order') or [])
    doorbell = is_doorbell_meta(meta)
    params = {
        'region': region,
        'label_radius_deg': meta.get('mrds_proximity_deg'),
        'feature_set': ','.join(features),
        'n_features': len(features),
        'n_samples': meta.get('n_samples'),
        'n_positive': meta.get('n_positive'),
        'sklearn_version': meta.get('sklearn_version'),
        'random_state': meta.get('random_state'),
        'n_estimators': meta.get('n_estimators'),
        'class_weight': meta.get('class_weight'),
        'study_area': meta.get('study_area'),
        'label_method': meta.get('label_method'),
        'is_doorbell': str(doorbell).lower(),
    }
    metrics = {
        'shuffled_auc': meta.get('cv_auc_mean'),
        'deadzone_auc': meta.get('spatial_cv_auc_mean'),
        'block_auc': meta.get('spatial_block_cv_auc_mean'),
    }
    for feat, value in (meta.get('feature_importances') or {}).items():
        metrics[f'importance_{feat}'] = value

    catalog = transfers if transfers is not None else {}
    if doorbell:
        for key, summary in catalog.items():
            metric_name = TRANSFER_METRIC_KEYS.get(key)
            if metric_name and summary.get('transfer_auc') is not None:
                metrics[metric_name] = summary['transfer_auc']
            if summary.get('transfer_auc') is not None and metric_name is None:
                slug = str(key).lower().replace(' ', '_')
                metrics[f'transfer_auc_{slug}'] = summary['transfer_auc']
    else:
        short = str(meta.get('study_area_short') or '')
        local = catalog.get(short) or {}
        if local.get('transfer_auc') is not None:
            metrics['doorbell_transfer_auc'] = local['transfer_auc']

    tags = {
        'model_id': meta.get('model_id', REGISTERED_DOORBELL),
        'stage': stage,
        'region': region,
        'promotion_rule': PROMOTION_RULE,
    }
    if doorbell:
        tags[REGISTERED_DOORBELL] = 'true'
    return {
        'experiment': EXPERIMENT_NAME,
        'run_name': meta.get('model_id', REGISTERED_DOORBELL),
        'params': params,
        'metrics': {k: v for k, v in metrics.items() if v is not None},
        'tags': tags,
        'promotion_note': PROMOTION_NOTE,
    }


def _finite_metrics(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number != number:  # NaN
            continue
        out[key] = number
    return out


def _stringify_params(params: dict) -> dict:
    return {k: '' if v is None else str(v) for k, v in params.items()}


def _import_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            'mlflow is optional. pip install mlflow  (or requirements-dev.txt)'
        ) from exc
    return mlflow


def log_persisted_model(
    meta_path,
    model_path,
    transfers=None,
    tracking_uri=None,
    register=False,
):
    """Log one already-written forest. Loads the joblib only to store it."""
    mlflow = _import_mlflow()
    meta = json.loads(Path(meta_path).read_text())
    record = run_record_from_meta(meta, transfers=transfers)
    uri = tracking_uri or os.environ.get('MLFLOW_TRACKING_URI')
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(record['experiment'])
    with mlflow.start_run(run_name=record['run_name']) as run:
        mlflow.set_tags(record['tags'])
        mlflow.log_params(_stringify_params(record['params']))
        mlflow.log_metrics(_finite_metrics(record['metrics']))
        mlflow.log_text(record['promotion_note'] + '\n', 'promotion_note.txt')
        mlflow.log_dict(meta, 'meta.json')
        mlflow.log_artifact(str(model_path), artifact_path='joblib')
        _log_sklearn_flavor(mlflow, model_path)
        if register:
            _register(mlflow, run.info.run_id, record)
        return run.info.run_id


def _log_sklearn_flavor(mlflow, model_path) -> None:
    """Registry needs an MLmodel dir; the API still loads the joblib."""
    try:
        import joblib
        import mlflow.sklearn
    except ImportError:
        return
    estimator = joblib.load(model_path)
    mlflow.sklearn.log_model(estimator, artifact_path='model')


def _register(mlflow, run_id: str, record: dict) -> None:
    name = record['tags']['model_id']
    mv = mlflow.register_model(f'runs:/{run_id}/model', name)
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_tag(name, 'stage', record['tags']['stage'])
    client.set_registered_model_tag(name, 'region', record['tags']['region'])
    client.set_registered_model_tag(name, 'promotion_rule', PROMOTION_RULE)
    if record['tags']['stage'] == STAGE_PRODUCTION:
        try:
            client.set_registered_model_alias(name, 'production', mv.version)
        except Exception:
            # Older MLflow: alias API may be missing; tags still record stage.
            pass


def log_all_models(model_dir=None, tracking_uri=None, register=False):
    """Backfill every persisted forest under models/."""
    dest = Path(model_dir) if model_dir else MODEL_DIR
    transfers = load_transfers(dest)
    run_ids = []
    for meta_path, model_path in iter_persisted_forests(dest):
        run_ids.append(
            log_persisted_model(
                meta_path,
                model_path,
                transfers=transfers,
                tracking_uri=tracking_uri,
                register=register,
            )
        )
    return run_ids


def maybe_log_after_persist(meta_path, model_path) -> None:
    """Task 9 hook. No-op unless MLFLOW_TRACKING_URI is set."""
    if not os.environ.get('MLFLOW_TRACKING_URI'):
        return
    try:
        run_id = log_persisted_model(meta_path, model_path)
    except Exception as exc:
        print(f'  MLflow log skipped: {exc}')
        return
    print(f'  MLflow run: {run_id}')


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            'Log persisted Task 9 forests to MLflow. Does not retrain. '
            'Does not change /predict.'
        )
    )
    parser.add_argument('--model-dir', default=None)
    parser.add_argument('--tracking-uri', default=None)
    parser.add_argument(
        '--register',
        action='store_true',
        help='Register models; WA alias production, sidecars stay experimental/local',
    )
    args = parser.parse_args(argv)
    run_ids = log_all_models(
        model_dir=args.model_dir,
        tracking_uri=args.tracking_uri,
        register=args.register,
    )
    print(f'logged {len(run_ids)} MLflow run(s)')
    for run_id in run_ids:
        print(f'  {run_id}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
