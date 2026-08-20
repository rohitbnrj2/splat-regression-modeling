"""Count spurious minima in every saved field — the metric that predicts whether descent works.

RMS against fast marching measures whether the learned field *is* the value function. It does not
measure whether the field is *usable*, and on this project's own evidence it does not predict it: the
one-obstacle torus field with 30 anchors has less than half the RMS of the label-free field (0.0856
against 0.1844) and reaches fewer shadow goals (14/20 against 18/20). What separates them is the
number of spurious basins — points where ``-grad T`` has nowhere to go — and this counts them.

A cell counts as a minimum when it is strictly below all eight of its neighbours. Read the count
**against the ground truth's own count on the identical grid**, never in absolute terms: the number is
a property of the grid as much as of the field. The true time-to-go has exactly one minimum, at the
source — but the sphere's lat-long chart reports 41 of them in the fast-marching field itself, purely
from the pole rings where many cells coincide. The difference is the meaningful quantity.

Reads the ``.npz`` files ``sweep`` saves, so it evaluates no model and can be run over a whole
results tree in seconds.

Run:
    python -m srms.experiments.minima --root results/figures
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import tyro


@dataclasses.dataclass
class MinimaConfig:
    """Where to look."""

    root: str = "results/figures"


def count(field: np.ndarray, shape: tuple[int, ...], mask: np.ndarray, periodic: bool) -> int:
    """Cells strictly below all eight neighbours, [scalar].

    Masked cells are set to ``+inf`` so an obstacle interior is never itself a minimum and never
    makes its neighbour one. ``periodic`` wraps the comparison, which is correct on the torus and
    wrong everywhere else — the sphere and Poincaré charts have genuine edges.
    """
    image = np.where(mask.reshape(shape), np.inf, field.reshape(shape))
    lower = np.ones(shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            if periodic:
                shifted = np.roll(np.roll(image, dy, axis=0), dx, axis=1)
            else:
                shifted = np.full(shape, np.inf)
                rows = slice(max(dy, 0), shape[0] + min(dy, 0)), slice(max(-dy, 0), shape[0] + min(-dy, 0))
                cols = slice(max(dx, 0), shape[1] + min(dx, 0)), slice(max(-dx, 0), shape[1] + min(-dx, 0))
                shifted[rows[0], cols[0]] = image[rows[1], cols[1]]
            lower &= image < shifted
    return int(lower.sum())


def main(cfg: MinimaConfig) -> None:
    """Report RMS beside the spurious-minimum count for every saved field under ``root``."""
    root = pathlib.Path(cfg.root)
    print(f"{'field':<56} {'RMS':>8} {'minima':>7} {'GT':>4} {'spurious':>9}")
    print("-" * 88)
    for path in sorted(root.rglob("*.npz")):
        data = np.load(path)
        shape = tuple(int(v) for v in data["shape"])
        if len(shape) != 2:
            continue  # SO(3)'s grid is 3-D; an 8-neighbour count is not the same question there
        mask, periodic = data["mask"], path.stem.startswith("torus")
        rms = float(np.sqrt(np.mean((data["prediction"][~mask] - data["truth"][~mask]) ** 2)))
        learned = count(data["prediction"], shape, mask, periodic)
        truth = count(data["truth"], shape, mask, periodic)
        label = f"{path.parent.name}/{path.stem}"
        print(f"{label:<56} {rms:>8.4f} {learned:>7} {truth:>4} {learned - truth:>9}")
    print("\n`spurious` is learned minus ground truth on the same grid — the count in excess of what")
    print("the reference itself reports. Positive means basins the true field does not have, which is")
    print("where a gradient descent gets trapped. Compare it with RMS: they do not agree, and the")
    print("shadow-goal success rate follows this column.")


if __name__ == "__main__":
    main(tyro.cli(MinimaConfig))
