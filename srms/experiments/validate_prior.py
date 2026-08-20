"""Validate a planner prior *before* anything is supervised with it.

This repo's third standing rule (``CLAUDE.md``) exists because the rule was broken twice: 30 RRT*
anchors drawn from an under-converged tree sent the torus backwards (0.1844 -> 0.3393), and the
iteration count that fixed the torus was then reused on the Poincaré ball, where hyperbolic volume
growth defeats it, and sent that manifold to 1.3467. Both failures were the same missing step —
nobody measured the prior on the scene it was about to supervise.

Four checks, in increasing order of what they need:

1. **Self-consistency (no ground truth).** A cost-to-come is the time along a feasible path, and with
   ``s >= 1`` no path beats the free-space geodesic, so ``cost(node) >= base(node)`` must hold at
   every node. A violation means the edge integral is under-pricing, not that the tree is coarse.
2. **Convergence (no ground truth).** RRT* costs only ever decrease as the tree fills in, so the mean
   cost-to-come flattening *is* the convergence signal. ``build_roadmap`` already doubles the budget
   until it flattens; this reports where it stopped and whether it hit the cap.
3. **Accuracy against fast marching.** Validation only — never a training path. Reported as the ratio
   ``cost / truth`` at p50/p90/p99 and as the fraction of nodes more than 10% high, because that is
   the shape that actually caused the damage: at 350 iterations the *median* node was within 0.8%
   while 9% of nodes were >10% high, and shadow-biased selection concentrated on exactly that tail.
4. **Occlusion labelling.** Shadow-targeted selection asks whether the source-to-node ray is blocked.
   The flat-chart spelling ``start + f·displacement`` is correct only on a flat chart; the geodesic
   ``Exp_start(f·Log_start(x))`` is correct everywhere. This counts how often they disagree, per
   manifold, which is what says whether the historical torus recipe transfers.

Fast marching enters here and nowhere upstream: this module is under ``srms/experiments``, which is
evaluation, and ``environments/test_selfsupervised.py`` enforces that no module under ``srms/methods``
can reach it.

Run:
    python -m srms.experiments.validate_prior                              # every 2-D manifold
    python -m srms.experiments.validate_prior --environment torus --num-obstacles 3
"""

from __future__ import annotations

import dataclasses
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from srms.environments import sampling
from srms.run import Config, _build_env

_GRID_CHUNK = 64  # nodes per nearest-cell lookup batch; the full [nodes, cells] matrix is large


@dataclasses.dataclass
class ValidateConfig:
    """Which scene to validate the prior on. Defaults match the Experiment-2 configuration."""

    environment: str = "all"  # "all" runs every manifold with a dense 2-D ground truth
    dim: int = 2
    num_obstacles: int = 1
    seed: int = 1
    rrt_iters: int = 1500
    rrt_step: float = 0.5
    rrt_radius: float = 0.9
    resolution: int = 240  # scoring grid for the fast-marching reference
    subsample: int = 300  # `build_roadmap`'s node budget, applied *after* the tree is built
    num_anchors: int = 30  # anchor budget whose shadow coverage is reported
    shadow_pref: float = 3.0  # selection weight multiplier for occluded nodes
    converge: bool = True  # double the budget until the mean cost-to-come flattens


MANIFOLDS = ("torus", "sphere", "poincare_hyperbolic")


def nearest_grid_value(env, grid_points: jnp.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Value of a gridded field at each query point, by nearest grid cell in geodesic distance.

    Manifold-agnostic on purpose: interpolating would need a per-manifold chart rule, and the
    quantity being checked here is a *ratio* at the 1% level while nearest-cell error is bounded by
    half a cell times the local gradient — 0.013 on a 240-cell torus against costs of 2 to 5. The
    caller prints that floor alongside the result so it is never mistaken for the prior's own error.

    Args:
        env: Environment supplying ``geodesic``.
        grid_points: Dense grid, [cells, dim].
        values: Field on that grid, [cells]; non-finite cells are excluded from the lookup.
        query: Points to look the field up at, [n, dim].

    Returns:
        Field value at the nearest finite grid cell to each query point, [n].
    """
    finite = np.isfinite(values)
    points, field = grid_points[jnp.asarray(finite)], jnp.asarray(values[finite])
    out = []
    for lo in range(0, len(query), _GRID_CHUNK):
        batch = jnp.asarray(query[lo : lo + _GRID_CHUNK], jnp.float32)
        idx = jax.vmap(lambda q: jnp.argmin(env.geodesic(points, q)))(batch)
        out.append(np.asarray(field[idx]))
    return np.concatenate(out)


def occluded_geodesic(env, points: np.ndarray) -> np.ndarray:
    """The manifold-correct occlusion test, i.e. what the fixed selector uses."""
    return sampling.occluded_from_source(env, points)


def occluded_flat(env, points: np.ndarray) -> np.ndarray:
    """The flat-chart spelling ``start + f · displacement_np``, as used by the historical selector.

    Kept only so the two can be compared. It is exact on the torus and wrong elsewhere: on the sphere
    the renormalised chord reaches ``arctan(theta)`` rather than ``theta`` so the far part of the ray
    is never tested, in the Poincaré ball the tangent norm is a geodesic length that overshoots the
    chart, and on SO(3) it adds a 3-vector to a 4-vector.
    """
    start = np.asarray(env.start, float)
    disp = env.displacement_np(start, points)
    fractions = np.linspace(0.1, 0.95, 8)
    ray = start[None, None, :] + fractions[None, :, None] * disp[:, None, :]
    sdf = np.asarray(env.sdf(jnp.asarray(ray.reshape(-1, env.dim), jnp.float32))).reshape(len(points), -1)
    return sdf.min(axis=1) < 0.0


def grid_spacing(env, grid: jnp.ndarray, probes: int = 32) -> float:
    """Median nearest-neighbour distance on the scoring grid — the floor below which RMS is noise.

    Taken over random probe cells rather than the grid's first rows: those are one edge of the chart,
    which is a lat-long pole ring on the sphere (coincident points, spacing 0) and outside the ball in
    the Poincaré chart's corners (spacing enormous). Neither describes the grid.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(len(grid), size=min(probes, len(grid)), replace=False)
    spacings = []
    for i in idx:
        distance = np.asarray(env.geodesic(grid, grid[i]))
        finite = distance[np.isfinite(distance) & (distance > 1e-9)]
        if len(finite):
            spacings.append(float(finite.min()))
    return float(np.median(spacings)) if spacings else float("nan")


def quantiles(ratio: np.ndarray) -> dict:
    """p50/p90/p99/max of the cost/truth ratio, plus the fraction more than 10% high."""
    return {
        "p50": float(np.percentile(ratio, 50)),
        "p90": float(np.percentile(ratio, 90)),
        "p99": float(np.percentile(ratio, 99)),
        "max": float(ratio.max()),
        "frac_10pct": float(np.mean(ratio > 1.10)),
    }


def validate(cfg: ValidateConfig, environment: str) -> dict:
    """Build the prior on one scene and run all four checks. Returns the row printed by ``report``."""
    base_cfg = Config(
        environment=environment,
        dim=cfg.dim,
        num_obstacles=cfg.num_obstacles,
        seed=cfg.seed,
        resolution=cfg.resolution,
    )
    env = _build_env(base_cfg)

    started = time.time()
    # The whole converged tree, not `build_roadmap`'s subsample — that is the population any anchor
    # selector should be drawing from, and thinning it first is one of the defects this validates.
    nodes, costs = sampling.converged_tree(
        env, env.start, cfg.rrt_iters, cfg.rrt_step, cfg.rrt_radius, cfg.seed, cfg.converge
    )
    elapsed = time.time() - started
    nodes, costs = np.asarray(nodes), np.asarray(costs)

    start = jnp.asarray(env.start, jnp.float32)
    geodesic = np.asarray(env.geodesic(jnp.asarray(nodes, jnp.float32), start))
    # A feasible path with s >= 1 can never beat the free-space geodesic; the 1e-4 tolerance is the
    # edge quadrature's own discretisation, not slack for a real violation.
    violations = int(np.sum(costs < geodesic - 1e-4))

    clearance = np.asarray(env.sdf_np(nodes))
    free = clearance > 0.0

    grid, _ = env.grid(cfg.resolution)
    truth = np.asarray(env.ground_truth(cfg.resolution))
    at_node = nearest_grid_value(env, grid, truth, nodes)
    ok = np.isfinite(at_node) & (at_node > 1e-3) & free
    ratio = costs[ok] / at_node[ok]
    rms = float(np.sqrt(np.mean((costs[ok] - at_node[ok]) ** 2)))

    geo_shadow = occluded_geodesic(env, nodes)
    try:
        flat_shadow = occluded_flat(env, nodes)
        disagree = int(np.sum(flat_shadow != geo_shadow))
        flat_note = f"{disagree}/{len(nodes)}"
    except Exception as exc:  # noqa: BLE001 — the flat ray is a shape error on SO(3), and that is the finding
        flat_note = f"{type(exc).__name__}"

    # How many occluded nodes survive `build_roadmap`'s uniform subsample — the population a selector
    # that runs *after* subsampling actually sees.
    rng = np.random.default_rng(cfg.seed + 13)
    if len(nodes) > cfg.subsample:
        idx = np.concatenate([[0], rng.choice(len(nodes) - 1, cfg.subsample - 1, replace=False) + 1])
    else:
        idx = np.arange(len(nodes))

    return {
        "environment": environment,
        "seconds": elapsed,
        "nodes": len(nodes),
        "violations": violations,
        "rms": rms,
        "cell": grid_spacing(env, grid),
        "shadow_full": int(geo_shadow.sum()),
        "shadow_sub": int(geo_shadow[idx].sum()),
        "flat_note": flat_note,
        **quantiles(ratio),
    }


def report(rows: list[dict], cfg: ValidateConfig) -> None:
    """Print the validation table and the verdict for each scene."""
    print(f"\nRRT* prior validation — {cfg.num_obstacles} obstacle(s), seed {cfg.seed}, "
          f"floor {cfg.rrt_iters} iters, reference resolution {cfg.resolution}")
    print(f"{'manifold':<20} {'nodes':>6} {'sec':>6} {'RMS':>8} {'p50':>6} {'p90':>6} {'p99':>6} "
          f"{'max':>6} {'>10%':>6} {'base viol':>10}")
    print("-" * 92)
    for r in rows:
        print(f"{r['environment']:<20} {r['nodes']:>6} {r['seconds']:>6.1f} {r['rms']:>8.4f} "
              f"{r['p50']:>6.3f} {r['p90']:>6.3f} {r['p99']:>6.3f} {r['max']:>6.3f} "
              f"{100 * r['frac_10pct']:>5.1f}% {r['violations']:>10}")

    print(f"\nOcclusion labelling and shadow coverage (anchor budget {cfg.num_anchors}, "
          f"subsample {cfg.subsample})")
    print(f"{'manifold':<20} {'occluded/full':>14} {'occluded/subsample':>20} {'flat-vs-geodesic ray':>22}")
    print("-" * 80)
    for r in rows:
        full = f"{r['shadow_full']}/{r['nodes']}"
        sub = f"{r['shadow_sub']}/{min(cfg.subsample, r['nodes'])}"
        print(f"{r['environment']:<20} {full:>14} {sub:>20} {r['flat_note']:>22}")

    print("\nHow to read this:")
    print("  `base viol` must be 0 — a cost below the free-space geodesic is an under-priced edge,")
    print("  not a coarse tree. `>10%` is the tail that broke the torus at 350 iterations; shadow-")
    print("  targeted selection draws from exactly that tail, so it must be ~0 before anchors are")
    print("  used as equalities. `flat-vs-geodesic ray` counts nodes the two occlusion tests label")
    print("  differently: every disagreement is a node the historical selector mis-classified.")
    for r in rows:
        floor = r["cell"]
        verdict = (
            "EQUALITY ANCHORS OK" if r["frac_10pct"] < 0.01 and r["violations"] == 0 and r["p99"] < 1.05
            else "BOUNDS ONLY — the tail is too heavy to pin as equalities"
        )
        print(f"  {r['environment']:<20} {verdict}   (reference cell size ~{floor:.3f}, "
              f"so an RMS below that is not resolvable)")


def main(cfg: ValidateConfig) -> None:
    """Validate the prior on one manifold, or on every manifold with a dense 2-D reference."""
    names = MANIFOLDS if cfg.environment == "all" else (cfg.environment,)
    report([validate(cfg, name) for name in names], cfg)


if __name__ == "__main__":
    main(tyro.cli(ValidateConfig))
