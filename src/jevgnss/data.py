from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROUTES = (
    "klt-20210610",
    "whampoa-20210714",
    "science-museum-wongkun-20260630",
    "reclamation-street-20260630",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def load_assets(root: Path, config: dict) -> dict:
    root = Path(root)
    data = root / "data"
    observations = pd.read_csv(data / "observations.csv.gz", keep_default_na=False, low_memory=False)
    epochs = pd.read_csv(data / "epochs.csv.gz", keep_default_na=False)
    references = pd.read_csv(data / "reference_positions.csv.gz")
    physical = pd.read_csv(data / "physical_predictions.csv.gz", keep_default_na=False)
    targets = pd.read_csv(data / "reference_risk_targets.csv.gz", keep_default_na=False)
    expected_routes = tuple(config["routes"])
    if expected_routes != ROUTES:
        raise ValueError("configuration changed the registered four-route order")
    for name, frame in (("observations", observations), ("epochs", epochs), ("physical", physical), ("targets", targets)):
        if set(frame.sequence_id) != set(ROUTES):
            raise ValueError(f"{name} contains a missing or extra route")
    if len(observations) != config["observation_rows"] or len(epochs) != config["candidate_epochs"]:
        raise ValueError("published data row counts differ from the registered release")
    if observations.sample_id.duplicated().any() or epochs.epoch_id.duplicated().any():
        raise ValueError("duplicate stable identity in published data")
    return dict(observations=observations, epochs=epochs, references=references,
                physical=physical, targets=targets)


def verify_checksums(release_root: Path, assets_root: Path) -> int:
    manifest = read_json(Path(release_root) / "expected" / "checksums.json")
    checked = 0
    for relative, expected in manifest["assets"].items():
        path = Path(assets_root) / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"asset checksum mismatch: {relative}")
        checked += 1
    return checked

