"""L7 -- интеграция с внешними ИС: экспорт результата L5 в ISO 11783-10 Task Data
(docs/chapter2/integration_model.md, §2.6). Оптимизация -- та же, что в
/optimization/optimize; этот маршрут дополнительно сериализует результат в
формат, который понимают бортовые терминалы техники и FMIS агрохолдингов
независимо от производителя, вместо JSON только для программных интеграций.
"""

import pandas as pd
from fastapi import APIRouter, HTTPException, Response

from ds_integration.isoxml_export import export_task_data
from ds_optimization.optimize import optimize_unconstrained, optimize_with_budget
from ds_optimization.validation import OptimizationInputError

from ..schemas import IsoxmlExportRequest

router = APIRouter(prefix="/integration", tags=["L7 -- интеграция с внешними ИС"])


@router.post("/export-isoxml", summary="Оптимальные дозы по участкам в формате ISO 11783-10 Task Data (§2.6.3)")
def export_isoxml(req: IsoxmlExportRequest) -> Response:
    plots_df = pd.DataFrame([p.model_dump() for p in req.plots])

    try:
        if req.budget is not None:
            doses = optimize_with_budget(plots_df, req.budget, req.dose_min, req.dose_max, req.price_yield, req.price_fert)
        else:
            doses = optimize_unconstrained(plots_df, req.dose_min, req.dose_max, req.price_yield, req.price_fert)
    except OptimizationInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    xml_text = export_task_data(
        plots_df,
        doses,
        customer_name=req.customer_name,
        farm_name=req.farm_name,
        task_designator=req.task_designator,
    )
    return Response(content=xml_text, media_type="application/xml", headers={"Content-Disposition": 'attachment; filename="TASKDATA.xml"'})
