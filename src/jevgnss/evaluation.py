from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROUTES
from .metrics import auc
from .positioning import solve

COORDS = ("x_ecef_m", "y_ecef_m", "z_ecef_m")


def _bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def _rmse(frame: pd.DataFrame) -> float:
    ok = frame.status.eq("ok") & _bool(frame.reference_available)
    errors = pd.to_numeric(frame.loc[ok, "horizontal_error_m"], errors="coerce").dropna().to_numpy(float)
    if not len(errors):
        raise ValueError("no scored successful epochs")
    return float(np.sqrt(np.mean(errors ** 2)))


def _baseline_metrics(evaluation: Path) -> tuple[dict, dict]:
    position, routes = {}, {}
    for key in ("wls", "gnc", "fde"):
        frame = pd.read_csv(evaluation / "baselines" / f"{key}_predictions.csv", keep_default_na=False)
        per_route = {route: _rmse(frame.loc[frame.sequence_id.eq(route)]) for route in ROUTES}
        position[key] = float(np.mean(list(per_route.values())))
        routes[key] = per_route
    return position, routes


def _learned_metrics(evaluation: Path, config: dict) -> tuple[dict, dict]:
    position, routes = {}, {}
    for variant in config["variants"]:
        values = []
        per_route = {}
        for route in ROUTES:
            local = []
            for seed in config["v008_seeds"]:
                path = evaluation / "v008" / "outer_evaluation" / route / "evaluation" / f"seed-{seed}" / variant / "predictions.csv"
                value = _rmse(pd.read_csv(path, keep_default_na=False))
                values.append(value)
                local.append(value)
            per_route[route] = float(np.mean(local))
        position[variant] = float(np.mean(values))
        routes[variant] = per_route
    values, per_route = [], {}
    for route in ROUTES:
        local = []
        for seed in config["tdl_seeds"]:
            path = evaluation / "tdl" / "outer_evaluation" / route / "evaluation" / f"seed-{seed}" / "TDL_shared" / "predictions.csv"
            value = _rmse(pd.read_csv(path, keep_default_na=False))
            values.append(value)
            local.append(value)
        per_route[route] = float(np.mean(local))
    position["tdl"] = float(np.mean(values))
    routes["tdl"] = per_route
    return position, routes


def _risk_metrics(assets: dict, evaluation: Path, config: dict) -> tuple[dict, int]:
    targets = assets["targets"].copy()
    targets = targets.loc[_bool(targets.target_available), ["sample_id", "sequence_id", "trusted_target"]]
    targets["positive"] = pd.to_numeric(targets.trusted_target).to_numpy(float) < 0.5
    records = []
    for route in ROUTES:
        target = targets.loc[targets.sequence_id.eq(route), ["sample_id", "positive"]]
        for seed in config["v008_seeds"]:
            for variant in config["variants"]:
                path = evaluation / "v008" / "risk_scores" / route / f"seed-{seed}" / f"{variant}.csv"
                scores = pd.read_csv(path)
                joined = target.merge(scores, on="sample_id", how="inner", validate="one_to_one", sort=False)
                records.append((variant, len(joined), auc(joined.student_risk_score, joined.positive)))
    result = {variant: float(np.mean([value for name, _, value in records if name == variant]))
              for variant in config["variants"]}
    # The same common observation set is evaluated for each seed; report its
    # identity count once, matching the registered audit rather than multiplying by seeds.
    observations = sum(count for name, count, _ in records if name == "soft") // len(config["v008_seeds"])
    return result, observations


def replay_v008(assets: dict, assets_root: Path, config: dict, progress=None) -> dict:
    observations = assets["observations"]
    eligible = observations.loc[_bool(observations.position_eligible), [
        "sample_id", "epoch_id", "sequence_id", "system", "sat_x_ecef_m", "sat_y_ecef_m", "sat_z_ecef_m",
        "corrected_pseudorange_m", "physical_variance_m2",
    ]].copy()
    epochs = assets["epochs"].set_index("epoch_id")
    physical = assets["physical"].set_index("epoch_id")
    evaluation = Path(assets_root) / "evaluation" / "v008"
    total = numerical = fallback = 0
    maximum = 0.0
    records = []
    for route in ROUTES:
        route_obs = eligible.loc[eligible.sequence_id.eq(route)]
        expected_ids = physical.index[physical.sequence_id.eq(route)].tolist()
        for seed in config["v008_seeds"]:
            for variant in config["variants"]:
                prediction_path = evaluation / "outer_evaluation" / route / "evaluation" / f"seed-{seed}" / variant / "predictions.csv"
                predictions = pd.read_csv(prediction_path, keep_default_na=False)
                if predictions.epoch_id.tolist() != expected_ids:
                    raise ValueError("candidate identities differ in frozen replay")
                scores = pd.read_csv(evaluation / "scores" / route / f"seed-{seed}" / f"{variant}.csv")
                joined = route_obs.merge(scores, on="sample_id", how="left", validate="one_to_one", sort=False)
                groups = {str(key): value for key, value in joined.groupby("epoch_id", sort=False)}
                local_numerical = local_fallback = 0
                local_max = 0.0
                for row in predictions.itertuples(index=False):
                    total += 1
                    if row.processing == "physical_fallback":
                        base = physical.loc[row.epoch_id]
                        if row.status != base.status:
                            raise ValueError(f"fallback status changed: {row.epoch_id}")
                        fallback += 1
                        local_fallback += 1
                        continue
                    group = groups[row.epoch_id]
                    if group[["negative_log_weight", "predicted_bias_m"]].isna().any().any():
                        raise ValueError(f"missing frozen inference output: {row.epoch_id}")
                    satellites = group[list(("sat_x_ecef_m", "sat_y_ecef_m", "sat_z_ecef_m"))].to_numpy(float)
                    ranges = group.corrected_pseudorange_m.to_numpy(float) - group.predicted_bias_m.to_numpy(float)
                    factors = np.exp(-group.negative_log_weight.to_numpy(float))
                    initial = epochs.loc[row.epoch_id, list(COORDS)].to_numpy(float)
                    solution = solve(satellites, ranges, group.system.to_numpy(str),
                                     group.physical_variance_m2.to_numpy(float), factors=factors,
                                     initial_ecef_m=initial)
                    if solution.status != row.status:
                        raise ValueError(f"solution status changed: {row.epoch_id}")
                    if solution.status == "ok":
                        expected = np.array([getattr(row, name) for name in COORDS], dtype=float)
                        difference = float(np.linalg.norm(solution.state[:3] - expected))
                        local_max = max(local_max, difference)
                        if difference > config["coordinate_tolerance_m"]:
                            raise ValueError(f"coordinate replay exceeded tolerance: {row.epoch_id}")
                    numerical += 1
                    local_numerical += 1
                maximum = max(maximum, local_max)
                item = dict(sequence_id=route, seed=seed, variant=variant,
                            candidate_epochs=len(predictions), numerical_replays=local_numerical,
                            physical_fallbacks=local_fallback, maximum_coordinate_difference_m=local_max)
                records.append(item)
                if progress:
                    progress(item)
    expected_total = config["candidate_epochs"] * len(config["v008_seeds"]) * len(config["variants"])
    if total != expected_total or numerical + fallback != expected_total:
        raise ValueError("full candidate coverage failed")
    return dict(status="passed", candidate_model_records=total, numerical_replays=numerical,
                physical_fallbacks=fallback, maximum_coordinate_difference_m=maximum,
                tolerance_m=config["coordinate_tolerance_m"], route_seed_checks=records)


def evaluate(assets: dict, assets_root: Path, config: dict, replay: bool = True, progress=None) -> tuple[dict, dict]:
    evaluation = Path(assets_root) / "evaluation"
    position, routes = _baseline_metrics(evaluation)
    learned, learned_routes = _learned_metrics(evaluation, config)
    position.update(learned)
    routes.update(learned_routes)
    risks, target_count = _risk_metrics(assets, evaluation, config)
    if target_count != config["risk_target_observations"]:
        raise ValueError(f"risk common-sample count changed: {target_count}")
    replay_result = replay_v008(assets, assets_root, config, progress) if replay else {
        "status": "skipped", "candidate_model_records": 0, "numerical_replays": 0,
        "physical_fallbacks": 0, "maximum_coordinate_difference_m": float("nan")}
    return dict(position_rmse_m=position, risk_auc=risks, per_route_rmse_m=routes), replay_result

