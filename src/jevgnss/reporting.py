from __future__ import annotations

import json
from pathlib import Path


LABELS = {
    "wls": "Physical WLS", "gnc": "GNC-TLS", "fde": "Residual FDE",
    "tdl": "9-input TDL-GNSS", "zero": "Null prior",
    "random": "Random prior", "hard": "Hard Jev", "soft": "Soft Jev",
}


def write_results(output: Path, metrics: dict, expected: dict, replay: dict, failures: list[str]) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(metrics, numerical_replay=replay, verification_failures=failures,
                   status="passed" if not failures else "failed")
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    lines = ["# JevGNSS paper reproduction", "", f"Status: **{payload['status']}**.", "",
             "## Positioning", "", "| Method | Macro horizontal RMSE (m) |", "|---|---:|"]
    for key in LABELS:
        lines.append(f"| {LABELS[key]} | {metrics['position_rmse_m'][key]:.6f} |")
    lines += ["", "## Reference risk", "", "| Method | AUC |", "|---|---:|"]
    for key in ("zero", "random", "hard", "soft"):
        lines.append(f"| {LABELS[key]} | {metrics['risk_auc'][key]:.6f} |")
    lines += ["", "## Reading", "",
              "The Jev risk score separates observations. Soft Jev has lower positioning error than the null prior and 9-input TDL-GNSS.",
              "The 95% interval of Soft Jev minus the random prior crosses zero, so that comparison is not a stable positioning gain.", "",
              f"Numerical replay: {replay['candidate_model_records']} candidate-model records, "
              f"maximum coordinate difference {replay['maximum_coordinate_difference_m']:.3e} m."]
    if failures:
        lines += ["", "## Mismatches", ""] + [f"- {item}" for item in failures]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")

