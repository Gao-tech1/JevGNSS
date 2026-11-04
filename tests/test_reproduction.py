import json
from pathlib import Path

from jevgnss.data import load_assets
from jevgnss.evaluation import evaluate
from jevgnss.metrics import compare

ROOT = Path(__file__).resolve().parents[1]


def test_fast_metric_reproduction_matches_registered_values():
    config = json.loads((ROOT / "configs" / "paper_v1.json").read_text())
    expected = json.loads((ROOT / "expected" / "metrics.json").read_text())
    assets = load_assets(ROOT / ".assets", config)
    metrics, replay = evaluate(assets, ROOT / ".assets", config, replay=False)
    assert not compare(metrics, expected, config["metric_tolerance"])
    assert replay["status"] == "skipped"

