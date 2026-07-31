"""L3 -- комбинированная предобработка, обёртка над ds_preprocessing.fill_gaps."""

import numpy as np
from fastapi import APIRouter, HTTPException

from ds_preprocessing import fill_gaps
from ds_preprocessing.validation import DataValidationError

from ..schemas import FillGapsRequest, FillGapsResponse

router = APIRouter(prefix="/preprocessing", tags=["L3 -- предобработка"])


@router.post("/fill-gaps", response_model=FillGapsResponse, summary="Восстановить пропуски в пространственно-временном ряде (§2.2)")
def fill_gaps_endpoint(req: FillGapsRequest) -> FillGapsResponse:
    coords = np.array(req.coords, dtype=float)
    times = np.array(req.times, dtype=float)
    values = np.array(req.values, dtype=float)
    mask = np.array(req.mask_observed, dtype=bool)

    try:
        result = fill_gaps(coords, times, values, mask, drop_anomalies=req.drop_anomalies)
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    filled = [[None if np.isnan(v) else float(v) for v in row] for row in result["filled"]]
    return FillGapsResponse(
        filled=filled,
        unrestored=result["unrestored"].tolist(),
        anomalies=result["anomalies"].tolist(),
    )
