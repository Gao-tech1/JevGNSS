"""Multi-constellation broadcast-pseudorange weighted least squares.

Release: jevgnss-paper-v1.
Each constellation has its own receiver clock. Geometry includes the
first-order Earth-rotation correction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

MODULE_VERSION = "jevgnss-paper-v1"
EARTH_RATE_OVER_C = 7.2921151467e-5 / 299792458.0
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Solution:
    """One position solution, including a complete failure state."""

    status: str
    state: FloatArray
    residual_m: FloatArray
    rank: int
    condition_number: float | None
    gdop: float | None
    iterations: int
    used_observations: int


def geometry(
    satellite_ecef_m: FloatArray, receiver_ecef_m: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Range and receiver-position Jacobian, with Earth-rotation correction."""

    satellites = np.asarray(satellite_ecef_m, dtype=float)
    receiver = np.asarray(receiver_ecef_m, dtype=float)
    if satellites.ndim != 2 or satellites.shape[1] != 3:
        raise ValueError("satellite coordinates must have shape n×3")
    if receiver.shape != (3,):
        raise ValueError("receiver coordinates must have shape (3,)")
    if not np.isfinite(satellites).all() or not np.isfinite(receiver).all():
        raise ValueError("geometry coordinates must be finite")
    delta = receiver - satellites
    distance = np.linalg.norm(delta, axis=1)
    if np.any(distance < 1):
        raise ValueError("satellite-receiver range is invalid")
    rotation = EARTH_RATE_OVER_C * (
        satellites[:, 0] * receiver[1]
        - satellites[:, 1] * receiver[0]
    )
    derivative = delta / distance[:, None]
    derivative[:, 0] -= EARTH_RATE_OVER_C * satellites[:, 1]
    derivative[:, 1] += EARTH_RATE_OVER_C * satellites[:, 0]
    return distance + rotation, derivative


def solve(
    satellite_ecef_m: FloatArray,
    corrected_pseudorange_m: FloatArray,
    systems: NDArray[np.str_],
    variance_m2: FloatArray,
    *,
    factors: FloatArray | None = None,
    initial_ecef_m: FloatArray | None = None,
    max_iterations: int = 30,
) -> Solution:
    """Solve position and per-constellation clocks from physical variances and optional factors."""

    satellites = np.asarray(satellite_ecef_m, dtype=float)
    ranges = np.asarray(corrected_pseudorange_m, dtype=float)
    systems = np.asarray(systems)
    variances = np.asarray(variance_m2, dtype=float)
    count = len(ranges)
    if (
        satellites.shape != (count, 3)
        or systems.shape != (count,)
        or variances.shape != (count,)
    ):
        raise ValueError("weighted least-squares inputs have inconsistent shapes")
    if (
        not np.isfinite(satellites).all()
        or not np.isfinite(ranges).all()
        or not np.isfinite(variances).all()
        or np.any(variances <= 0)
    ):
        raise ValueError("positioning inputs must be finite and variances must be positive")
    if isinstance(max_iterations, bool) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    weights = (
        np.ones(count, dtype=float)
        if factors is None
        else np.asarray(factors, dtype=float)
    )
    if (
        weights.shape != (count,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("continuous weight factors are invalid")

    keep = weights > 0
    constellations = sorted(set(systems[keep]))
    state = np.zeros(3 + len(constellations), dtype=float)
    if initial_ecef_m is not None:
        initial = np.asarray(initial_ecef_m, dtype=float)
        if initial.shape != (3,) or not np.isfinite(initial).all():
            raise ValueError("initial position is invalid")
        state[:3] = initial
    used = int(keep.sum())
    if not constellations or used < len(state):
        return Solution(
            "insufficient_observations",
            state,
            np.full(count, np.nan),
            0,
            None,
            None,
            0,
            used,
        )

    clock_design = np.column_stack(
        [systems == constellation for constellation in constellations]
    ).astype(float)
    inverse_variance = weights / variances
    residual = np.zeros(count)
    rank = 0
    condition: float | None = None
    for iteration in range(1, max_iterations + 1):
        distance, derivative = geometry(satellites, state[:3])
        design = np.column_stack([derivative, clock_design])
        residual = ranges - distance - clock_design @ state[3:]
        whitened = design[keep] * np.sqrt(inverse_variance[keep, None])
        right_hand = residual[keep] * np.sqrt(inverse_variance[keep])
        step, _, rank, singular = np.linalg.lstsq(
            whitened, right_hand, rcond=None
        )
        condition = (
            float(singular[0] / singular[-1])
            if len(singular) and singular[-1] > 0
            else None
        )
        if rank < len(state) or condition is None or condition > 1e10:
            return Solution(
                "rank_deficient_or_ill_conditioned",
                state,
                residual,
                int(rank),
                condition,
                None,
                iteration,
                used,
            )
        state += step
        if np.linalg.norm(step) < 1e-4:
            distance, derivative = geometry(satellites, state[:3])
            design = np.column_stack([derivative, clock_design])
            residual = ranges - distance - clock_design @ state[3:]
            singular_geometry = np.linalg.svd(
                design[keep], compute_uv=False
            )
            gdop = float(math.sqrt(np.sum(1.0 / singular_geometry**2)))
            status = (
                "ok"
                if 5e6 < np.linalg.norm(state[:3]) < 8e6
                else "invalid_receiver_position"
            )
            return Solution(
                status,
                state,
                residual,
                int(rank),
                condition,
                gdop,
                iteration,
                used,
            )
    return Solution(
        "non_converged",
        state,
        residual,
        int(rank),
        condition,
        None,
        max_iterations,
        used,
    )
