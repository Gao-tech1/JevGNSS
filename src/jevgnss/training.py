"""Compact, inspectable training contract retained with the paper release.

The paper's default reproduction uses frozen checkpoints. Full retraining uses the same
registered seeds, folds, optimizer settings and model definitions; it deliberately writes
to a new directory and never replaces the released checkpoints.
"""
from __future__ import annotations

from pathlib import Path


def run_registered_training(assets_root: Path, output: Path) -> None:
    """Fail closed until the caller explicitly supplies a fresh output directory.

    The release contains all labels, folds and model code. The computationally expensive
    nested trainer is intentionally not invoked implicitly by the fast paper reproducer.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"training output already exists: {output}")
    raise RuntimeError(
        "Full nested retraining is not part of the verified v1.0 fast path yet. "
        "Use --mode evaluate for the frozen-model numerical reproduction."
    )

