import json
from pathlib import Path

from jevgnss.data import ROUTES, load_assets

ROOT = Path(__file__).resolve().parents[1]


def test_published_data_has_only_registered_routes():
    config = json.loads((ROOT / "configs" / "paper_v1.json").read_text())
    assets = load_assets(ROOT / ".assets", config)
    assert tuple(config["routes"]) == ROUTES
    assert len(assets["epochs"]) == 14523
    assert len(assets["observations"]) == 215421

