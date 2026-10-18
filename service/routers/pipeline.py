"""Conservative end-to-end orchestration over the existing in-repository DS modules.

The pipeline intentionally separates prediction from the fertilizer response model:
a yield predictor is not, by itself, a causal estimate of fertilizer response.
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ds_integration.isoxml_export import export_task_data
from ds_ontology.integration import get_or_create_pole
from ds_ontology.schema import build_schema
from ds_optimization.optimize import optimize_unconstrained, optimize_with_budget
from ds_optimization.validation import OptimizationInputError
from ds_preprocessing import fill_gaps
from ds_preprocessing.validation import DataValidationError
from ds_prediction.features import ALL_COLUMNS
from ds_prediction.model import DEFAULT_MODALITIES, evaluate_grouped_cv, make_baseline_model, make_model, train_model
from ds_prediction.synthetic import generate_dataset
from service.schemas import PredictionFeatures

logger = logging.getLogger("ds.pipeline")
router = APIRouter(prefix="/pipeline", tags=["Сквозной пайплайн"])


class PipelineField(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field_id: str = Field(min_length=1, max_length=128)
    plot_id: int = Field(ge=0)
    area_ha: float = Field(gt=0, allow_inf_nan=False)
    region: str | None = None
    geometry_wkt: str | None = None
    features: dict[str, float] | None = None
    response_parameters: dict[str, float] | None = None


class PipelineObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation_id: str = Field(min_length=1, max_length=128)
    field_id: str = Field(min_length=1, max_length=128)
    variable: str = Field(min_length=1, max_length=128)
    value: float | None = Field(default=None, allow_inf_nan=False)
    unit: str = Field(min_length=1, max_length=32)
    observed_at: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    quality_status: str = "accepted"


class PipelineEconomics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    price_yield_per_centner: float = Field(gt=0, allow_inf_nan=False)
    fertilizer_price_per_kg: float = Field(gt=0, allow_inf_nan=False)
    budget_amount: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    budget_currency: str = Field(default="RUB", pattern=r"^[A-Z]{3}$")
    fertilizer_quantity_kg_max: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    dose_min_kg_ha: float = Field(default=0, ge=0, allow_inf_nan=False)
    dose_max_kg_ha: float = Field(default=150, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.dose_max_kg_ha < self.dose_min_kg_ha:
            raise ValueError("dose_max_kg_ha must be >= dose_min_kg_ha")
        if self.budget_amount is not None and self.budget_currency != "RUB":
            raise ValueError("Current optimizer integration supports RUB economics only")
        return self


class GapfillPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field_ids: list[str] = Field(min_length=1)
    coords: list[list[float]]
    times: list[float] = Field(min_length=2)
    values: list[list[float]]
    mask_observed: list[list[bool]]
    # Maps observation_id -> [field-row-index, time-column-index] in the supplied matrix.
    observation_cells: dict[str, list[int]] = Field(default_factory=dict)
    drop_anomalies: bool = True


class PipelineOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    synthetic_only: bool = False
    export_requested: bool = False
    human_approved: bool = False
    allow_synthetic_geometry_export: bool = False


class PipelineRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str | None = None
    farm_id: str = Field(min_length=1, max_length=128)
    season_id: str = Field(min_length=1, max_length=128)
    crop: str = Field(min_length=1, max_length=128)
    fields: list[PipelineField] = Field(min_length=1)
    observations: list[PipelineObservation] = Field(default_factory=list)
    economics: PipelineEconomics
    response_model_evidence_status: str = "unknown"
    gapfill: GapfillPayload | None = None
    options: PipelineOptions = Field(default_factory=PipelineOptions)

    @model_validator(mode="after")
    def validate_references(self):
        ids = [item.field_id for item in self.fields]
        plots = [item.plot_id for item in self.fields]
        if len(ids) != len(set(ids)):
            raise ValueError("field_id values must be unique")
        if len(plots) != len(set(plots)):
            raise ValueError("plot_id values must be unique for ISOXML identifiers")
        unknown = sorted({item.field_id for item in self.observations} - set(ids))
        if unknown:
            raise ValueError(f"observations reference unknown fields: {unknown}")
        if len({item.observation_id for item in self.observations}) != len(self.observations):
            raise ValueError("observation_id values must be unique")
        for item in self.fields:
            for label, mapping in (("features", item.features), ("response_parameters", item.response_parameters)):
                if mapping is not None and any(not np.isfinite(value) for value in mapping.values()):
                    raise ValueError(f"{label} values must be finite for field {item.field_id}")
        if self.gapfill is not None:
            if len(self.gapfill.field_ids) != len(self.gapfill.coords):
                raise ValueError("gapfill.field_ids and gapfill.coords must have the same length")
            if len(self.gapfill.field_ids) != len(set(self.gapfill.field_ids)):
                raise ValueError("gapfill.field_ids must be unique")
            if len(self.gapfill.values) != len(self.gapfill.field_ids):
                raise ValueError("gapfill.values row count must match gapfill.field_ids")
            if len(self.gapfill.mask_observed) != len(self.gapfill.field_ids):
                raise ValueError("gapfill.mask_observed row count must match gapfill.field_ids")
            time_count = len(self.gapfill.times)
            if any(len(row) != time_count for row in self.gapfill.values):
                raise ValueError("every gapfill.values row must match the number of times")
            if any(len(row) != time_count for row in self.gapfill.mask_observed):
                raise ValueError("every gapfill.mask_observed row must match the number of times")
            if set(self.gapfill.field_ids) - set(ids):
                raise ValueError("gapfill.field_ids contain unknown field IDs")
            known_observation_ids = {obs.observation_id for obs in self.observations}
            unknown_cell_ids = set(self.gapfill.observation_cells) - known_observation_ids
            if unknown_cell_ids:
                raise ValueError(f"gapfill.observation_cells references unknown observations: {sorted(unknown_cell_ids)}")
            for observation_id, cell in self.gapfill.observation_cells.items():
                if len(cell) != 2:
                    raise ValueError(f"gapfill cell for {observation_id} must be [row, column]")
                row, column = cell
                if row < 0 or row >= len(self.gapfill.field_ids) or column < 0 or column >= time_count:
                    raise ValueError(f"gapfill cell for {observation_id} is outside matrix bounds")
        return self


def _fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _stage(name: str, status: str, *, reason: str | None = None, data: dict | None = None, warnings: list[str] | None = None, metrics: dict | None = None) -> dict:
    return {"name": name, "status": status, "reason": reason, "data": data, "warnings": warnings or [], "metrics": metrics or {}}


def _ontology_stage(req: PipelineRunRequest) -> dict:
    """Create an isolated ontology and validate field-level entities/areas; not a reasoner-backed ABox audit."""
    try:
        onto = build_schema()
        for item in req.fields:
            get_or_create_pole(
                onto,
                item.field_id,
                geometriya=item.geometry_wkt,
                ploschad=item.area_ha,
                region=item.region,
            )
        return _stage(
            "ontology",
            "ok",
            data={"field_entities_created": len(req.fields), "observation_semantics_mapped": False},
            warnings=["Ontology stage validates field entities only; observations are not yet mapped to ontology individuals and no OWL reasoner validation is claimed."],
        )
    except Exception as exc:
        logger.exception("Pipeline ontology stage failed")
        return _stage("ontology", "error", reason="ONTOLOGY_VALIDATION_FAILED", warnings=[type(exc).__name__])


def _quality_stage(req: PipelineRunRequest) -> tuple[dict, dict[str, Any] | None]:
    missing = [obs for obs in req.observations if obs.value is None or obs.quality_status in {"rejected", "unrestored"}]
    if not missing:
        return _stage("quality_gate", "ok", data={"missing_or_rejected_observation_ids": []}), None
    if req.gapfill is None:
        return _stage(
            "quality_gate", "blocked", reason="MISSING_VALUES_REQUIRE_GAPFILL_MATRIX",
            data={"missing_or_rejected_observation_ids": [obs.observation_id for obs in missing]},
        ), None
    payload = req.gapfill
    if not set(payload.field_ids).issubset({f.field_id for f in req.fields}):
        return _stage("gapfill", "error", reason="GAPFILL_REFERENCES_UNKNOWN_FIELD"), None
    try:
        result = fill_gaps(
            np.asarray(payload.coords, dtype=float),
            np.asarray(payload.times, dtype=float),
            np.asarray(payload.values, dtype=float),
            np.asarray(payload.mask_observed, dtype=bool),
            drop_anomalies=payload.drop_anomalies,
        )
        filled = np.asarray(result["filled"], dtype=float)
        unrestored = np.asarray(result["unrestored"], dtype=bool)
        unresolved = []
        for obs in missing:
            cell = payload.observation_cells.get(obs.observation_id)
            if not isinstance(cell, list) or len(cell) != 2:
                unresolved.append(obs.observation_id)
                continue
            i, j = cell
            if (
                i < 0 or j < 0 or i >= filled.shape[0] or j >= filled.shape[1]
                or payload.field_ids[i] != obs.field_id
                or unrestored[i, j] or not np.isfinite(filled[i, j])
            ):
                unresolved.append(obs.observation_id)
        data = {
            "missing_or_rejected_observation_ids": [obs.observation_id for obs in missing],
            "unresolved_observation_ids": unresolved,
            "filled_matrix_shape": list(filled.shape),
            "anomaly_count": int(np.asarray(result["anomalies"], dtype=bool).sum()),
        }
        if unresolved:
            return _stage("gapfill", "blocked", reason="SOME_VALUES_REMAIN_UNRESTORED", data=data), result
        return _stage("gapfill", "ok", data=data, warnings=["Restored values are not written back to the source observations; callers must persist cell-to-observation provenance before scientific use."]), result
    except (DataValidationError, ValueError, IndexError) as exc:
        return _stage("gapfill", "error", reason="GAPFILL_INPUT_OR_EXECUTION_ERROR", warnings=[str(exc)[:300]]), None
    except Exception as exc:
        logger.exception("Pipeline gapfill stage failed")
        return _stage("gapfill", "error", reason="GAPFILL_EXECUTION_FAILED", warnings=[type(exc).__name__]), None


def _prediction_stage(req: PipelineRunRequest) -> dict:
    if not all(field.features is not None for field in req.fields):
        return _stage("predict", "blocked", reason="ENGINEERED_FEATURES_MISSING")
    if not req.options.synthetic_only:
        return _stage(
            "predict", "blocked", reason="REAL_MODEL_PROVENANCE_NOT_ATTESTED",
            warnings=["The existing API may load or train a demo model; this pipeline requires an explicit model-provenance registry before non-synthetic predictions can drive decisions."],
        )
    try:
        # Explicitly synthetic demo run: evaluate on a fixed synthetic dataset and retain the baseline comparison.
        frame = generate_dataset()
        groups = int(frame["field_id"].nunique())
        if groups < 2:
            return _stage("predict", "blocked", reason="INSUFFICIENT_GROUPS_FOR_VALIDATION")
        candidate_rmse = evaluate_grouped_cv(frame, modalities=DEFAULT_MODALITIES, n_splits=min(5, groups), model_factory=make_model)
        baseline_rmse = evaluate_grouped_cv(frame, modalities=DEFAULT_MODALITIES, n_splits=min(5, groups), model_factory=make_baseline_model)
        if not np.isfinite(candidate_rmse) or not np.isfinite(baseline_rmse):
            return _stage("predict", "error", reason="NONFINITE_VALIDATION_METRIC")
        if candidate_rmse >= baseline_rmse:
            return _stage("predict", "blocked", reason="MODEL_DID_NOT_BEAT_BASELINE", metrics={"candidate_rmse": candidate_rmse, "baseline_rmse": baseline_rmse}, warnings=["Candidate model did not beat the linear baseline; no field predictions were emitted."])
        model = train_model(frame, modalities=DEFAULT_MODALITIES, model_factory=make_model)
        predictions = []
        for field in req.fields:
            missing = sorted(set(ALL_COLUMNS) - set(field.features or {}))
            if missing:
                return _stage("predict", "blocked", reason="FEATURE_COLUMNS_MISSING", data={"field_id": field.field_id, "missing": missing})
            features = PredictionFeatures(**field.features)
            row = {key: getattr(features, key) for key in ALL_COLUMNS}
            import pandas as pd
            predicted = float(model.predict(pd.DataFrame([row])[ALL_COLUMNS])[0])
            predictions.append({"field_id": field.field_id, "predicted_yield_centner_ha": predicted})
        return _stage("predict", "ok", data={"predictions": predictions, "model": "GradientBoostingRegressor", "training_data": "synthetic_demo"}, metrics={"candidate_rmse": candidate_rmse, "baseline_rmse": baseline_rmse}, warnings=["Synthetic-only prediction; not evidence of generalization to real farms."])
    except Exception as exc:
        logger.exception("Pipeline prediction stage failed")
        return _stage("predict", "error", reason="PREDICTION_FAILED", warnings=[type(exc).__name__])


def _optimization_stage(req: PipelineRunRequest) -> dict:
    if req.response_model_evidence_status != "validated" and not (req.options.synthetic_only and req.response_model_evidence_status == "synthetic"):
        return _stage("optimize", "blocked", reason="RESPONSE_MODEL_NOT_VALIDATED", warnings=["Fertilizer response parameters must be explicitly supplied and independently validated; prediction output is not substituted for a response model."])
    if any(field.response_parameters is None for field in req.fields):
        return _stage("optimize", "blocked", reason="RESPONSE_PARAMETERS_MISSING")
    plots = []
    for field in req.fields:
        params = field.response_parameters or {}
        if any(key not in params for key in ("baseline", "R", "s")):
            return _stage("optimize", "blocked", reason="RESPONSE_PARAMETERS_INCOMPLETE", data={"field_id": field.field_id})
        plots.append({"plot_id": field.plot_id, "baseline": float(params["baseline"]), "R": float(params["R"]), "s": float(params["s"]), "area": field.area_ha})
    economics = req.economics
    budget_options = []
    if economics.fertilizer_quantity_kg_max is not None:
        budget_options.append(economics.fertilizer_quantity_kg_max)
    if economics.budget_amount is not None:
        budget_options.append(economics.budget_amount / economics.fertilizer_price_per_kg)
    budget = min(budget_options) if budget_options else None
    try:
        import pandas as pd
        plots_df = pd.DataFrame(plots)
        if budget is None:
            doses = optimize_unconstrained(plots_df, economics.dose_min_kg_ha, economics.dose_max_kg_ha, economics.price_yield_per_centner, economics.fertilizer_price_per_kg)
        else:
            doses = optimize_with_budget(plots_df, budget, economics.dose_min_kg_ha, economics.dose_max_kg_ha, economics.price_yield_per_centner, economics.fertilizer_price_per_kg)
        if len(doses) != len(plots) or not np.isfinite(doses).all():
            return _stage("optimize", "error", reason="INVALID_OPTIMIZER_OUTPUT")
        quantity_used = float(np.sum(np.asarray(doses) * plots_df["area"].to_numpy()))
        if budget is not None and quantity_used > budget + 1e-6:
            return _stage("optimize", "error", reason="OPTIMIZER_BUDGET_NOT_RESPECTED", data={"quantity_used_kg": quantity_used, "budget_kg": budget})
        return _stage(
            "optimize", "ok",
            data={"plot_ids": [p["plot_id"] for p in plots], "field_ids": [f.field_id for f in req.fields], "doses_kg_ha": [float(x) for x in doses], "quantity_used_kg": quantity_used, "quantity_budget_kg": budget, "budget_currency_amount": economics.budget_amount},
            warnings=["Optimization is single-nutrient and uses the supplied Mitscherlich response coefficients; multi-nutrient shared-budget optimization is not implemented."],
        )
    except (OptimizationInputError, ValueError, RuntimeError) as exc:
        return _stage("optimize", "error", reason="OPTIMIZATION_FAILED", warnings=[str(exc)[:300]])
    except Exception as exc:
        logger.exception("Pipeline optimization stage failed")
        return _stage("optimize", "error", reason="OPTIMIZATION_FAILED", warnings=[type(exc).__name__])


@router.post("/runs", summary="Запустить сквозной конвейер с блокировкой небезопасных решений")
def create_pipeline_run(raw: dict[str, Any]) -> dict[str, Any]:
    run_id = "dsrun_" + uuid.uuid4().hex[:12]
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        req = PipelineRunRequest.model_validate(raw)
    except Exception as exc:
        return {
            "schema_version": "1.0", "run_id": run_id, "status": "error",
            "recommendation": {"status": "withheld", "reason": "INVALID_REQUEST", "actions": []},
            "stages": [_stage("intake", "error", reason="INVALID_REQUEST")],
            "errors": [{"code": "INVALID_REQUEST", "message": str(exc)[:1000]}],
            "lineage": {"input_fingerprint": hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest(), "created_at": created_at},
        }

    payload = req.model_dump(mode="json")
    stages = [_stage("intake", "ok", data={"field_count": len(req.fields), "observation_count": len(req.observations), "synthetic_only": req.options.synthetic_only})]
    ontology = _ontology_stage(req)
    stages.append(ontology)

    quality, gapfill_result = _quality_stage(req)
    stages.append(quality)
    if gapfill_result is not None and quality["name"] == "gapfill":
        quality["data"] = {**(quality.get("data") or {}), "filled_matrix": np.asarray(gapfill_result["filled"], dtype=float).tolist()}

    prediction = _prediction_stage(req) if ontology["status"] == "ok" and quality["status"] == "ok" else _stage("predict", "blocked", reason="UPSTREAM_GATE_FAILED")
    stages.append(prediction)
    optimization = _optimization_stage(req) if prediction["status"] == "ok" else _stage("optimize", "blocked", reason="PREDICTION_STAGE_NOT_READY")
    stages.append(optimization)

    actions = []
    if optimization["status"] == "ok":
        actions = [{"field_id": field_id, "plot_id": plot_id, "action": "fertilizer_application", "dose_kg_ha": dose} for field_id, plot_id, dose in zip(optimization["data"]["field_ids"], optimization["data"]["plot_ids"], optimization["data"]["doses_kg_ha"])]
        rec_status = "exploratory" if req.options.synthetic_only else ("approved" if req.options.human_approved else "draft")
        recommendation = {"status": rec_status, "objective": "maximize_expected_profit", "actions": actions, "economics": {"quantity_used_kg": optimization["data"]["quantity_used_kg"], "quantity_budget_kg": optimization["data"]["quantity_budget_kg"]}}
    else:
        recommendation = {"status": "withheld", "reason": optimization.get("reason") or "OPTIMIZATION_NOT_READY", "actions": []}

    if req.options.export_requested:
        if not req.options.synthetic_only or not req.options.allow_synthetic_geometry_export or not req.options.human_approved or optimization["status"] != "ok":
            stages.append(_stage("isoexport", "blocked", reason="EXPORT_REQUIRES_APPROVAL_AND_EXPLICIT_SYNTHETIC_GEOMETRY_MODE"))
        else:
            try:
                import pandas as pd
                plots_df = pd.DataFrame([{"plot_id": f.plot_id, "area": f.area_ha} for f in req.fields])
                xml = export_task_data(plots_df, np.asarray(optimization["data"]["doses_kg_ha"], dtype=float), customer_name=req.farm_id, farm_name=req.season_id, task_designator=f"{req.crop} synthetic demo")
                stages.append(_stage("isoexport", "ok", data={"format": "ISO11783-10 XML", "xml": xml, "geometry_mode": "synthetic_demo"}, warnings=["This export contains synthetic geometry and is for testing only; it is not suitable for machine control."]))
            except Exception as exc:
                logger.exception("Pipeline ISOXML export failed")
                stages.append(_stage("isoexport", "error", reason="ISOXML_EXPORT_FAILED", warnings=[type(exc).__name__]))

    statuses = [stage["status"] for stage in stages]
    overall = "error" if "error" in statuses else ("blocked" if "blocked" in statuses or recommendation["status"] == "withheld" else "ok")
    warnings = [warning for stage in stages for warning in stage.get("warnings", [])]
    if req.options.synthetic_only:
        warnings.append("Synthetic-only run: no output is evidence of agronomic or economic effectiveness on a real farm.")
    return {
        "schema_version": "1.0", "run_id": req.request_id or run_id, "status": overall,
        "recommendation": recommendation, "stages": stages, "warnings": warnings, "errors": [],
        "lineage": {"input_fingerprint": _fingerprint(payload), "created_at": created_at, "farm_id": req.farm_id, "season_id": req.season_id},
    }
