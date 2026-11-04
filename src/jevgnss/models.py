from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class Student(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 5))


class Standardize(nn.Module):
    def __init__(self, mean, scale, bounded: bool):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean, dtype=torch.float64), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(scale, dtype=torch.float64), requires_grad=False)
        self.bounded = bounded

    def forward(self, x):
        z = (x - self.mean) / self.std
        if self.bounded:
            z = torch.cat((5.0 * torch.tanh(z[..., :4] / 5.0), z[..., 4:]), dim=-1)
        return z


class TDLNetwork(nn.Module):
    def __init__(self, inputs: int, mean, scale, bounded: bool):
        super().__init__()
        self.network = nn.Module()
        self.network.seq = nn.Sequential(
            Standardize(mean, scale, bounded), nn.Linear(inputs, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2),
        )
        self.double()

    def forward(self, x):
        out = self.network.seq(x)
        amplitude = torch.sigmoid(out[..., 0]).clamp_min(0.001)
        return amplitude.square(), torch.relu(out[..., 1])


def checkpoint_inventory(root: Path) -> dict:
    root = Path(root)
    result = {
        "students": len(list((root / "students").rglob("model.pt"))),
        "v008": len(list((root / "v008").rglob("model.pt"))),
        "tdl": len(list((root / "tdl").rglob("model.pt"))),
    }
    if result != {"students": 30, "v008": 48, "tdl": 12}:
        raise ValueError(f"incomplete checkpoint package: {result}")
    # weights_only parsing catches truncated or incompatible PyTorch files without executing pickle code.
    for path in sorted(root.rglob("model.pt")):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            raise ValueError(f"invalid checkpoint: {path}")
    return result

