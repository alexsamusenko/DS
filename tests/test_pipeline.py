"""Regression tests for the end-to-end pipeline API and safety gates."""

import pytest

from service.routers.pipeline import PipelineRunRequest, _optimization_stage, create_pipeline_run


def base_request(**overrides):
    request = {
        "request_id": "pipeline-test",
        "farm_id": "farm-1",
        "season_id": "2026",
        "crop": "wheat",
        "fields": [{"field_id": "A", "plot_id": 1, "area_ha": 10.0}],
        "observations": [
            {
                "observation_id": "o1",
                "field_id": "A",
                "variable": "ndvi",
                "value": 0.5,
                "unit": "1",
                "observed_at": "2026-05-01T12:00:00+03:00",
                "source_id": "synthetic-test",
            }
        ],
        "economics": {
            "price_yield_per_centner": 1300,
            "fertilizer_price_per_kg": 30,
            "budget_amount": 3000,
            "budget_currency": "RUB",
            "fertilizer_quantity_kg_max": 100,
            "dose_min_kg_ha": 0,
            "dose_max_kg_ha": 20,
        },
        "options": {"synthetic_only": True},
    }
    request.update(overrides)
    return request


def test_pipeline_withholds_decision_when_response_model_is_missing():
    result = create_pipeline_run(base_request())
    assert result["status"] == "blocked"
    assert result["recommendation"]["status"] == "withheld"
    assert result["recommendation"]["actions"] == []
    assert next(s for s in result["stages"] if s["name"] == "optimize")["reason"] == "PREDICTION_STAGE_NOT_READY"


def test_pipeline_blocks_missing_observation_without_explicit_matrix():
    request = base_request(observations=[{
        "observation_id": "o1",
        "field_id": "A",
        "variable": "soil_moisture_mean",
        "value": None,
        "unit": "%",
        "observed_at": "2026-05-01T12:00:00+03:00",
        "source_id": "synthetic-test",
    }])
    result = create_pipeline_run(request)
    quality = next(s for s in result["stages"] if s["name"] == "quality_gate")
    assert quality["status"] == "blocked"
    assert quality["reason"] == "MISSING_VALUES_REQUIRE_GAPFILL_MATRIX"
    assert next(s for s in result["stages"] if s["name"] == "predict")["status"] == "blocked"


def test_pipeline_request_rejects_duplicate_field_and_plot_ids():
    request = base_request(fields=[
        {"field_id": "A", "plot_id": 1, "area_ha": 10},
        {"field_id": "A", "plot_id": 2, "area_ha": 20},
    ])
    with pytest.raises(ValueError, match="field_id values must be unique"):
        PipelineRunRequest.model_validate(request)

    request = base_request(fields=[
        {"field_id": "A", "plot_id": 1, "area_ha": 10},
        {"field_id": "B", "plot_id": 1, "area_ha": 20},
    ])
    with pytest.raises(ValueError, match="plot_id values must be unique"):
        PipelineRunRequest.model_validate(request)


def test_optimizer_uses_tighter_of_monetary_and_quantity_budgets():
    request = PipelineRunRequest.model_validate(base_request(
        response_model_evidence_status="synthetic",
        fields=[
            {"field_id": "A", "plot_id": 1, "area_ha": 10, "response_parameters": {"baseline": 40, "R": 8, "s": 60}},
            {"field_id": "B", "plot_id": 2, "area_ha": 10, "response_parameters": {"baseline": 42, "R": 3, "s": 60}},
        ],
    ))
    result = _optimization_stage(request)
    assert result["status"] == "ok"
    assert result["data"]["quantity_budget_kg"] == pytest.approx(100)
    assert result["data"]["quantity_used_kg"] <= 100 + 1e-6
    assert len(result["data"]["doses_kg_ha"]) == 2


def test_non_synthetic_optimizer_requires_validated_response_evidence():
    request = PipelineRunRequest.model_validate(base_request(
        response_model_evidence_status="synthetic",
        options={"synthetic_only": False},
        fields=[{"field_id": "A", "plot_id": 1, "area_ha": 10, "response_parameters": {"baseline": 40, "R": 8, "s": 60}}],
    ))
    result = _optimization_stage(request)
    assert result["status"] == "blocked"
    assert result["reason"] == "RESPONSE_MODEL_NOT_VALIDATED"



def test_pipeline_rejects_malformed_gapfill_matrix():
    request = base_request(
        observations=[{
            "observation_id": "o1",
            "field_id": "A",
            "variable": "soil_moisture_mean",
            "value": None,
            "unit": "%",
            "observed_at": "2026-05-01T12:00:00+03:00",
            "source_id": "synthetic-test",
        }],
        gapfill={
            "field_ids": ["A"],
            "coords": [[0.0, 0.0]],
            "times": [0.0, 1.0],
            "values": [[0.0]],
            "mask_observed": [[False, True]],
            "observation_cells": {"o1": [0, 0]},
        },
    )
    with pytest.raises(ValueError, match="every gapfill.values row"):
        PipelineRunRequest.model_validate(request)


def test_pipeline_rejects_nonfinite_response_parameters():
    request = base_request(fields=[{
        "field_id": "A",
        "plot_id": 1,
        "area_ha": 10,
        "response_parameters": {"baseline": 40, "R": float("nan"), "s": 60},
    }])
    with pytest.raises(ValueError, match="response_parameters values must be finite"):
        PipelineRunRequest.model_validate(request)



def test_pipeline_route_is_registered_in_openapi():
    from fastapi.testclient import TestClient

    from service.app import app

    paths = TestClient(app).get("/openapi.json").json()["paths"]
    assert "/pipeline/runs" in paths
    assert "post" in paths["/pipeline/runs"]



def test_unrestored_gapfill_cells_are_json_null():
    import numpy as np

    from service.routers.pipeline import _json_safe_matrix

    assert _json_safe_matrix([[1.0, np.nan], [np.inf, 2.0]]) == [[1.0, None], [None, 2.0]]
