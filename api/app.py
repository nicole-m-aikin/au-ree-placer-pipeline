"""FastAPI app serving the persisted Task 9 Random Forest."""

from contextlib import asynccontextmanager
from pathlib import Path

import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from api.scoring import UnitError, predict_with_spread
from pipeline.ml_artifacts import (
    load_gold_mrds,
    load_published_artifacts,
    load_transfer,
    published_meta_path,
    published_model_path,
)
from pipeline.ml_spatial import (
    CV_OPTIMISM_NOTE,
    NEGATIVE_CLASS_NOTE,
    TARGET_QUESTION,
)

RESOURCE_TONNAGE_NOTE = (
    "Resource-tonnage uncertainty (Task 4 Monte Carlo; P10/P50/P90 NdPr tonnes) "
    "is a separate quantity and is not exposed by this API."
)


def _load_state(model_dir):
    state = {
        'model': None,
        'meta': None,
        'gold_xy': None,
        'sklearn_version': sklearn.__version__,
        'sklearn_match': False,
        'load_error': None,
    }
    model_path = published_model_path(model_dir)
    meta_path = published_meta_path(model_dir)
    if not model_path.exists() or not meta_path.exists():
        state['load_error'] = (
            f"model or metadata missing: {model_path} / {meta_path}"
        )
        return state
    try:
        estimator, metadata = load_published_artifacts(model_dir)
    except Exception as exc:
        state['load_error'] = f"failed to load artifacts: {exc}"
        return state
    state['model'] = estimator
    state['meta'] = metadata
    state['gold_xy'] = load_gold_mrds(model_dir)
    trained = metadata.get('sklearn_version')
    state['sklearn_match'] = trained == sklearn.__version__
    if not state['sklearn_match']:
        state['load_error'] = (
            f"sklearn {sklearn.__version__} != training {trained}"
        )
    return state


def create_app(model_dir=None):
    """Build the app. ``model_dir`` is for tests; production uses models/."""

    resolved_dir = Path(model_dir) if model_dir else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loaded = _load_state(resolved_dir)
        app.state.model = loaded['model']
        app.state.meta = loaded['meta']
        app.state.gold_xy = loaded.get('gold_xy')
        app.state.sklearn_version = loaded['sklearn_version']
        app.state.sklearn_match = loaded['sklearn_match']
        app.state.load_error = loaded['load_error']
        yield

    app = FastAPI(
        title="Gold-placer lookalike doorbell",
        description=(
            "Scores NURE-style stream-sediment chemistry against a Random Forest "
            "trained on NE Washington gold-placer proximity. Returns P(lookalike), "
            "tree-vote spread, and (if lon/lat sent) distance to the nearest "
            "training gold mine. Not a monazite/REE detector. Not a national "
            "model. Tree-vote spread is not Monte Carlo and not Task 4 tonnes."
        ),
        lifespan=lifespan,
    )

    @app.get('/')
    def root():
        return {
            'name': 'Gold-placer lookalike doorbell',
            'trained_on': 'NE Washington NURE vs gold MRDS',
            'not': [
                'a national placer model',
                'a monazite / REE detector',
                'Task 4 tonne uncertainty',
            ],
            'health': '/health',
            'model_info': '/model-info',
            'predict': 'POST /predict',
            'docs': '/docs',
        }

    def _ready():
        model = getattr(app.state, 'model', None)
        meta = getattr(app.state, 'meta', None)
        match = bool(getattr(app.state, 'sklearn_match', False))
        return model is not None and meta is not None and match

    @app.get('/health', response_model=HealthResponse)
    def health():
        model = getattr(app.state, 'model', None)
        meta = getattr(app.state, 'meta', None)
        match = bool(getattr(app.state, 'sklearn_match', False))
        err = getattr(app.state, 'load_error', None)
        body = HealthResponse(
            status='ok' if _ready() else 'unavailable',
            model_loaded=model is not None,
            metadata_loaded=meta is not None,
            sklearn_version=getattr(app.state, 'sklearn_version', sklearn.__version__),
            sklearn_version_training=None if meta is None else meta.get('sklearn_version'),
            sklearn_match=match,
            detail=err,
        )
        if body.status != 'ok':
            return JSONResponse(status_code=503, content=body.model_dump())
        return body

    @app.get('/model-info', response_model=ModelInfoResponse)
    def model_info():
        if not _ready():
            raise HTTPException(
                status_code=503,
                detail=getattr(app.state, 'load_error', None) or 'model not loaded',
            )
        meta = app.state.meta
        transfer = load_transfer(resolved_dir)
        return ModelInfoResponse(
            model_id=meta.get('model_id', 'task9_rf_placer_gold'),
            training_date=meta['training_date'],
            sklearn_version=meta['sklearn_version'],
            feature_order=meta['feature_order'],
            feature_units=meta.get('feature_units', {}),
            feature_importances=meta.get('feature_importances', {}),
            cv_auc_mean=meta['cv_auc_mean'],
            cv_auc_folds=meta['cv_auc_folds'],
            cv_auc_std=meta.get('cv_auc_std', 0.0),
            spatial_cv_auc_mean=meta.get('spatial_cv_auc_mean'),
            spatial_cv_auc_std=meta.get('spatial_cv_auc_std'),
            spatial_cv_auc_folds=meta.get('spatial_cv_auc_folds'),
            spatial_block_cv_auc_mean=meta.get('spatial_block_cv_auc_mean'),
            spatial_cv_method=meta.get('spatial_cv_method'),
            target_question=meta.get('target_question', TARGET_QUESTION),
            negative_class_note=meta.get('negative_class_note', NEGATIVE_CLASS_NOTE),
            cv_optimism_note=meta.get('cv_optimism_note', CV_OPTIMISM_NOTE),
            label_method=meta['label_method'],
            mrds_proximity_deg=meta['mrds_proximity_deg'],
            mrds_commodity_filter=meta['mrds_commodity_filter'],
            n_samples=meta['n_samples'],
            n_positive=meta['n_positive'],
            study_area=meta.get('study_area', 'unknown'),
            resource_tonnage_uncertainty=meta.get(
                'resource_tonnage_uncertainty', RESOURCE_TONNAGE_NOTE
            ),
            transfer_belt=None if not transfer else transfer.get('belt'),
            transfer_auc=None if not transfer else transfer.get('transfer_auc'),
            transfer_n_samples=None if not transfer else transfer.get('n_samples'),
            transfer_n_positive=None if not transfer else transfer.get('n_positive'),
            transfer_retrained=None if not transfer else bool(transfer.get('retrained')),
            transfer_note=None if not transfer else transfer.get('note'),
        )

    @app.post('/predict', response_model=PredictResponse)
    def predict(req: PredictRequest):
        if not _ready():
            raise HTTPException(
                status_code=503,
                detail=getattr(app.state, 'load_error', None) or 'model not loaded',
            )
        try:
            result = predict_with_spread(
                app.state.model, app.state.meta, req,
                gold_xy=getattr(app.state, 'gold_xy', None),
            )
        except UnitError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictResponse(**result)

    return app


app = create_app()
