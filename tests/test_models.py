from pathlib import Path

from jevgnss.models import Student, checkpoint_inventory

ROOT = Path(__file__).resolve().parents[1]


def test_student_capacity_and_checkpoint_inventory():
    assert sum(parameter.numel() for parameter in Student().parameters()) == 1157
    assert checkpoint_inventory(ROOT / ".assets" / "checkpoints") == {
        "students": 30, "v008": 48, "tdl": 12}

