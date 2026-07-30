"""Метрики качества восстановления, §2.2.6."""

import numpy as np


def rmse(estimate, truth):
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    valid = ~np.isnan(estimate)
    return float(np.sqrt(np.mean((estimate[valid] - truth[valid]) ** 2)))


def mae(estimate, truth):
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    valid = ~np.isnan(estimate)
    return float(np.mean(np.abs(estimate[valid] - truth[valid])))
