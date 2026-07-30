"""Контролируемый набор разнородных участков поля для валидации оптимизатора.

Дефицит азота -- источник внутрипольной неоднородности: чем он выше, тем
ниже базовый потенциал (baseline) участка без доп. внесения, но тем больше
его отклик на удобрение (R) -- именно эта неоднородность и делает
дифференцированное внесение осмысленным (§2.4.5).
"""

import numpy as np
import pandas as pd


def generate_plots(n_plots=20, seed=5):
    rng = np.random.default_rng(seed)

    nitrogen_deficit = rng.uniform(0, 1, size=n_plots)

    # Единицы: baseline/R -- ц/га, s/доза -- кг/га д.в. удобрения, area -- га.
    baseline = 40.0 - 5.0 * nitrogen_deficit + rng.normal(0, 1.0, size=n_plots)
    R = np.clip(3.0 + 6.0 * nitrogen_deficit + rng.normal(0, 0.4, size=n_plots), 0.1, None)
    s = rng.uniform(40, 80, size=n_plots)
    area = rng.uniform(0.8, 1.5, size=n_plots)

    return pd.DataFrame(
        {
            "plot_id": np.arange(n_plots),
            "baseline": baseline,
            "R": R,
            "s": s,
            "area": area,
        }
    )
