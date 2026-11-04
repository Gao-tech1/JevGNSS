from __future__ import annotations

import numpy as np
import pandas as pd


def route_macro_rmse(frame: pd.DataFrame) -> tuple[float, dict[str, float]]:
    values = {}
    for route, group in frame.groupby("sequence_id", sort=False):
        ok = group.status.eq("ok") & group.reference_available.astype(bool)
        errors = pd.to_numeric(group.loc[ok, "horizontal_error_m"], errors="coerce").dropna().to_numpy(float)
        values[str(route)] = float(np.sqrt(np.mean(errors ** 2)))
    return float(np.mean(list(values.values()))), values


def auc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0 or not np.isfinite(scores).all():
        raise ValueError("AUC needs finite scores and both classes")
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def compare(actual: dict, expected: dict, tolerance: float) -> list[str]:
    failures = []
    for section in ("position_rmse_m", "risk_auc"):
        for name, target in expected[section].items():
            value = actual[section][name]
            if abs(value - target) > tolerance:
                failures.append(f"{section}.{name}: {value} != {target}")
    return failures

