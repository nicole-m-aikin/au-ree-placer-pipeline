"""Pydantic request/response models for the Task 9 targeting API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from pipeline.ml_preprocess import (
    FEATURES,
    FE_PPM_MIN_PLAUSIBLE,
    FE_WT_PCT_MAX,
    MDL_NEG_LIMIT,
    P_WT_PCT_MAX,
    TRACE_PPM_MAX,
)

_PPM_DESC = (
    "ppm after NURE QA. Values in [-10, 0) are half-MDL (abs/2). "
    "null or 0 means not measured and is filled with the training log-median. "
    "Values < -10 are rejected (not a valid NURE MDL encoding)."
)
_P_DESC = (
    "Phosphorus in ppm. NURE stores P as wt%; load_nure already converted "
    "(×10,000). Send ppm, not percent. Values in (0, 1) are rejected as "
    "likely unconverted wt%."
)
_FE_DESC = (
    "Iron. NURE reports Fe as wt% (typical stream sediment 1–10). The model "
    "is trained on ppm (wt% × 10,000). You must set fe_unit; a unitless Fe "
    "is rejected. fe_unit=wt_pct values ≥ 20 are rejected. fe_unit=ppm "
    "values in (0, 100) are rejected as almost certainly unconverted wt%."
)


class PredictRequest(BaseModel):
    Th: Optional[float] = Field(..., description="Thorium, " + _PPM_DESC)
    Ce: Optional[float] = Field(..., description="Cerium, " + _PPM_DESC)
    La: Optional[float] = Field(..., description="Lanthanum, " + _PPM_DESC)
    P: Optional[float] = Field(..., description=_P_DESC)
    U: Optional[float] = Field(..., description="Uranium, " + _PPM_DESC)
    Au: Optional[float] = Field(..., description="Gold, " + _PPM_DESC)
    As: Optional[float] = Field(..., description="Arsenic, " + _PPM_DESC)
    Ti: Optional[float] = Field(..., description="Titanium, " + _PPM_DESC)
    Fe: Optional[float] = Field(..., description=_FE_DESC)
    Zr: Optional[float] = Field(..., description="Zirconium, " + _PPM_DESC)
    Y: Optional[float] = Field(..., description="Yttrium, " + _PPM_DESC)
    fe_unit: Literal['wt_pct', 'ppm'] = Field(
        ...,
        description=(
            "Unit of the Fe field. Required. 'wt_pct' is converted ×10,000 "
            "to match training. 'ppm' is used as-is if it is a plausible "
            "stream-sediment Fe (not a leftover 2–5 wt% value)."
        ),
    )
    lon: Optional[float] = Field(
        None,
        description="Optional sample longitude (WGS-84). Needed for distance to known gold.",
    )
    lat: Optional[float] = Field(
        None,
        description="Optional sample latitude (WGS-84). Needed for distance to known gold.",
    )

    @model_validator(mode='after')
    def check_nure_bounds(self):
        for name in FEATURES:
            v = getattr(self, name)
            if v is None:
                continue
            if v < -MDL_NEG_LIMIT:
                raise ValueError(
                    f"{name}={v} is below -{MDL_NEG_LIMIT:.0f}; "
                    "not a valid NURE half-MDL encoding"
                )
            if name != 'Fe' and v > TRACE_PPM_MAX:
                raise ValueError(
                    f"{name}={v} exceeds {TRACE_PPM_MAX:.0f} ppm; "
                    "not a plausible stream-sediment value"
                )
            if name == 'P' and 0 < v < P_WT_PCT_MAX:
                raise ValueError(
                    f"P={v} looks like wt%; send ppm (NURE wt% × 10,000)"
                )

        fe = self.Fe
        if (self.lon is None) ^ (self.lat is None):
            raise ValueError("lon and lat must be sent together, or both omitted")
        if self.lon is not None and not (-180.0 <= self.lon <= 180.0):
            raise ValueError("lon must be in [-180, 180]")
        if self.lat is not None and not (-90.0 <= self.lat <= 90.0):
            raise ValueError("lat must be in [-90, 90]")

        if fe is not None and fe >= 0:
            if self.fe_unit == 'wt_pct' and fe >= FE_WT_PCT_MAX:
                raise ValueError(
                    f"Fe={fe} wt% is not a plausible stream-sediment value "
                    f"(reject ≥ {FE_WT_PCT_MAX:.0f})"
                )
            if self.fe_unit == 'ppm' and 0 < fe < FE_PPM_MIN_PLAUSIBLE:
                raise ValueError(
                    f"Fe={fe} ppm is below {FE_PPM_MIN_PLAUSIBLE:.0f}; "
                    "this is almost certainly unconverted wt%. "
                    "Set fe_unit='wt_pct' or send Fe in ppm."
                )
        return self


class PredictResponse(BaseModel):
    label: int = Field(..., description="1 if probability ≥ 0.5, else 0")
    probability: float = Field(
        ...,
        description="P(gold-placer lookalike) = mean predict_proba over the 200 trees",
    )
    tree_vote_spread: float = Field(
        ...,
        description=(
            "Standard deviation of per-tree P(anomalous) across the 200 trees "
            "(rf.estimators_). This is tree-vote spread, not a Monte Carlo "
            "interval, and not Task 4 endowment uncertainty."
        ),
    )
    n_trees: int
    nearest_gold_mrds_deg: Optional[float] = Field(
        None,
        description="Degrees to the nearest gold MRDS site used in training labels.",
    )
    spatial_flag: str = Field(
        ...,
        description=(
            "near_known_gold if within the 0.15° label radius; "
            "far_from_known_gold if farther (the interesting / untrusted case); "
            "unknown_no_location if lon/lat were omitted."
        ),
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    metadata_loaded: bool
    sklearn_version: str
    sklearn_version_training: Optional[str] = None
    sklearn_match: bool
    detail: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_id: str
    training_date: str
    sklearn_version: str
    feature_order: list
    feature_units: dict
    feature_importances: dict
    cv_auc_mean: float
    cv_auc_folds: list
    cv_auc_std: float
    spatial_cv_auc_mean: Optional[float] = None
    spatial_cv_auc_std: Optional[float] = None
    spatial_cv_auc_folds: Optional[list] = None
    spatial_block_cv_auc_mean: Optional[float] = None
    spatial_cv_method: Optional[str] = None
    target_question: str
    negative_class_note: str
    cv_optimism_note: str
    label_method: str
    mrds_proximity_deg: float
    mrds_commodity_filter: str
    n_samples: int
    n_positive: int
    study_area: str
    resource_tonnage_uncertainty: str
    transfer_belt: Optional[str] = None
    transfer_auc: Optional[float] = None
    transfer_n_samples: Optional[int] = None
    transfer_n_positive: Optional[int] = None
    transfer_retrained: Optional[bool] = None
    transfer_note: Optional[str] = None
    transfer_belts: Optional[list] = None
