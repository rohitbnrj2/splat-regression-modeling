"""Re-render every saved figure from its stored fields, without retraining anything.

``sweep`` writes an ``.npz`` beside each figure holding the ground truth, the learned field, the
out-of-domain mask and the grid shape — precisely so that a change to how a figure *looks* never
costs a change to what it *shows*. This walks those files and redraws them.

Use it when the styling changes: fonts, colour limits, panel titles, axis labels. A restyle that
required retraining would silently invite "well, the numbers moved a little" — here they cannot move
at all, because no model is evaluated.

The scene is rebuilt from the filename, which ``sweep`` writes as
``<environment>_obs<n>_seed<s>.npz``, plus the grid shape for the resolution. That is enough because a
scene depends only on those three things: the same manifold, obstacle count and seed give the same
obstacles, the same chart and the same render extent, whatever method produced the field.

Run:
    python -m srms.experiments.restyle                       # every figure under results/figures
    python -m srms.experiments.restyle --root results/figures/exp2
"""

from __future__ import annotations

import dataclasses
import pathlib
import re

import numpy as np
import tyro

from srms.experiments import figures
from srms.run import Config, _build_env

_NAME = re.compile(r"^(?P<environment>.+)_obs(?P<obstacles>\d+)_seed(?P<seed>\d+)$")


@dataclasses.dataclass
class RestyleConfig:
    """Where to look, and how to draw."""

    root: str = "results/figures"
    error_clip: float = 0.2  # symmetric colour limit on the signed-error panel
    dry_run: bool = False


def scene_from(path: pathlib.Path, shape: tuple[int, ...]):
    """Rebuild the environment a saved field was scored on, from its filename and grid shape."""
    match = _NAME.match(path.stem)
    if match is None:
        return None
    resolution = int(shape[0])
    cfg = Config(
        environment=match["environment"],
        dim=3 if match["environment"] == "so3" else 2,
        num_obstacles=int(match["obstacles"]),
        seed=int(match["seed"]),
        resolution=resolution,
    )
    return _build_env(cfg), cfg


def main(restyle: RestyleConfig) -> None:
    """Redraw every ``.npz`` under ``root``, reporting what was rewritten and what was skipped."""
    root = pathlib.Path(restyle.root)
    saved = sorted(root.rglob("*.npz"))
    if not saved:
        print(f"no saved fields under {root} — nothing to restyle")
        return
    for path in saved:
        data = np.load(path)
        shape = tuple(int(v) for v in data["shape"])
        scene = scene_from(path, shape)
        if scene is None:
            print(f"  skip   {path}  (filename does not name a scene)")
            continue
        env, cfg = scene
        truth, prediction, mask = data["truth"], data["prediction"], data["mask"]
        rms = float(np.sqrt(np.mean((prediction[~mask] - truth[~mask]) ** 2)))
        print(f"  redraw {path.with_suffix('.png')}  (RMS {rms:.4f})")
        if restyle.dry_run:
            continue
        figures.save(
            env,
            str(path.with_suffix(".png")),
            truth,
            prediction,
            shape,
            mask=mask if mask.any() else None,
            error_clip=restyle.error_clip,
            reference="analytic geodesic" if cfg.num_obstacles == 0 else "fast marching (GT)",
            title=f"{cfg.environment} — {cfg.num_obstacles} obstacle(s), self-supervised, "
            f"seed {cfg.seed} (RMS {rms:.4f})",
        )
    print(f"\n{len(saved)} saved field set(s) under {root}. No model was evaluated; the numbers "
          "cannot have moved.")


if __name__ == "__main__":
    main(tyro.cli(RestyleConfig))
