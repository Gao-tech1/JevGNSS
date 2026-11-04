import numpy as np

from jevgnss.positioning import geometry, solve


def test_wls_recovers_synthetic_receiver():
    receiver = np.array([-2419231.0, 5385474.0, 2405342.0])
    rng = np.random.default_rng(42)
    satellites = rng.normal(size=(12, 3))
    satellites *= 26500000.0 / np.linalg.norm(satellites, axis=1)[:, None]
    systems = np.array(["G"] * 6 + ["C"] * 6)
    ranges = geometry(satellites, receiver)[0] + np.where(systems == "G", 80.0, 120.0)
    solution = solve(satellites, ranges, systems, np.full(12, 4.0), initial_ecef_m=receiver + 10.0)
    assert solution.status == "ok"
    assert np.linalg.norm(solution.state[:3] - receiver) < 1e-4

