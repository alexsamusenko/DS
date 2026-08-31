"""L5 -- оптимизация дифференцированного внесения удобрений, обёртка над ds_optimization."""

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from ds_optimization.economics import profit
from ds_optimization.optimize import optimize_unconstrained, optimize_with_budget
from ds_optimization.validation import OptimizationInputError

from ..schemas import OptimizeRequest, OptimizeResponse

router = APIRouter(prefix="/optimization", tags=["L5 -- оптимизация внесения"])


def _total_profit(plots_df: pd.DataFrame, doses: np.ndarray, price_yield: float, price_fert: float) -> float:
    per_area = profit(
        plots_df["baseline"].to_numpy(),
        plots_df["R"].to_numpy(),
        plots_df["s"].to_numpy(),
        doses,
        price_yield,
        price_fert,
    )
    return float(np.sum(per_area * plots_df["area"].to_numpy()))


@router.post("/optimize", response_model=OptimizeResponse, summary="Оптимальные дозы по участкам, сравнение с равномерным сценарием (§2.4)")
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    plots_df = pd.DataFrame([p.model_dump() for p in req.plots])

    try:
        if req.budget is not None:
            doses = optimize_with_budget(plots_df, req.budget, req.dose_min, req.dose_max, req.price_yield, req.price_fert)
        else:
            doses = optimize_unconstrained(plots_df, req.dose_min, req.dose_max, req.price_yield, req.price_fert)
    except OptimizationInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    total_profit = _total_profit(plots_df, doses, req.price_yield, req.price_fert)

    total_area = float(plots_df["area"].sum())
    dose_uniform = float(np.sum(doses * plots_df["area"].to_numpy())) / total_area if total_area else 0.0
    doses_uniform = np.full(len(plots_df), dose_uniform)
    total_profit_uniform = _total_profit(plots_df, doses_uniform, req.price_yield, req.price_fert)

    gain_percent = ((total_profit / total_profit_uniform) - 1) * 100 if total_profit_uniform else 0.0

    return OptimizeResponse(
        doses=doses.tolist(),
        total_profit=total_profit,
        total_profit_uniform=total_profit_uniform,
        profit_gain_percent=gain_percent,
    )
