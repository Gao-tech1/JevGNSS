#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevgnss.data import load_assets, read_json, verify_checksums
from jevgnss.evaluation import evaluate
from jevgnss.metrics import compare
from jevgnss.models import checkpoint_inventory
from jevgnss.reporting import write_results
from jevgnss.training import run_registered_training


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce the JevGNSS paper results")
    parser.add_argument("--mode", choices=("smoke", "evaluate", "train-and-evaluate"), default="evaluate")
    parser.add_argument("--assets", type=Path, default=ROOT / ".assets")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "paper-v1")
    parser.add_argument("--skip-numerical-replay", action="store_true",
                        help="Only for tests: recompute metrics without 174,276 full WLS replays")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_json(ROOT / "configs" / "paper_v1.json")
    expected = read_json(ROOT / "expected" / "metrics.json")
    checked = verify_checksums(ROOT, args.assets)
    assets = load_assets(args.assets, config)
    inventory = checkpoint_inventory(args.assets / "checkpoints")
    if args.mode == "smoke":
        print(json.dumps({"status": "passed", "checked_assets": checked,
                          "candidate_epochs": len(assets["epochs"]),
                          "observation_rows": len(assets["observations"]),
                          "checkpoints": inventory}, indent=2))
        return 0
    if args.mode == "train-and-evaluate":
        run_registered_training(args.assets, args.output / "retrained")
    def progress(item):
        print(json.dumps(item, ensure_ascii=False), flush=True)
    metrics, replay = evaluate(assets, args.assets, config,
                               replay=not args.skip_numerical_replay, progress=progress)
    failures = compare(metrics, expected, config["metric_tolerance"])
    if not args.skip_numerical_replay and replay["status"] != "passed":
        failures.append("full numerical replay did not pass")
    write_results(args.output, metrics, expected, replay, failures)
    print(json.dumps({"status": "passed" if not failures else "failed",
                      "output": str(args.output), "failures": failures}, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

