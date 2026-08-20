"""Extract paths from a learned field by gradient descent, and check them for collisions.

The field is only useful if ``-∇T`` actually reaches the source without entering an obstacle, and
that is a different question from RMS: a field can be close in value everywhere and still have a
spurious local minimum that traps a path, or a shadow whose gradient points into the obstacle it is
supposed to route around.

Descent is Riemannian — the step is ``Exp_x(-dt · metric_inv · ∇T / ‖·‖)`` — so it stays on the
manifold and is the same on every chart. A path succeeds when it reaches the source ball; it fails by
collision (any sample inside an obstacle) or by stalling (gradient vanishes, i.e. a local minimum).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

_MAX_STEPS = 4000


def descend(grad_fn, value_fn, metric_fn, env, goal, step: float, source_value: float) -> dict:
    """Follow ``-∇T`` from ``goal`` until the learned field says the source is reached.

    **Only the learned field moves the path.** Direction is ``-metric_inv · ∇T`` and termination is
    ``T(x) < source_value`` — the field's own estimate of remaining travel time. Obstacle geometry is
    read solely to *record* whether the path entered an obstacle, and the terminal distance to the
    source is recorded for the same reason: so a claim of success can be checked rather than taken on
    the field's word. Neither quantity influences a step.

    The step is a projection retraction, ``wrap_point(x + dt·d)``, not ``exp_map``: ``metric_inv``
    yields an *ambient* vector while ``exp_map`` expects tangent-frame coordinates, so passing one to
    the other is a shape error on every embedded manifold (sphere, SO(3), Lorentz). The projection
    agrees with the exponential map to O(step³) — a property this repo already relies on in
    ``srm.post_step`` — and is correct on flat charts by construction.

    Args:
        grad_fn: ``x -> ∇T(x)``, ambient.
        value_fn: ``x -> T(x)``.
        metric_fn: ``x -> metric_inv(x)``.
        env: Environment supplying ``wrap_point``, ``sdf``, ``geodesic``, ``start``.
        goal: Starting point, [dim].
        step: Arc length per step.
        source_value: Descent stops when the learned ``T`` falls below this.

    Returns:
        Dict with ``path``, ``outcome`` (``reached``/``stalled``/``diverged``), ``collided``,
        ``end_distance`` (true geodesic distance from the endpoint to the source, measurement only)
        and ``length``.
    """
    start = jnp.asarray(env.start, jnp.float32)
    x = goal
    path, outcome, collided = [np.asarray(x)], "diverged", False
    for _ in range(_MAX_STEPS):
        if float(value_fn(x)) < source_value:
            outcome = "reached"
            break
        direction = -(metric_fn(x) @ grad_fn(x))
        norm = float(jnp.linalg.norm(direction))
        if not np.isfinite(norm) or norm < 1e-8:
            outcome = "stalled"  # stationary point of the field, or a non-finite gradient
            break
        x = env.wrap_point(x + jnp.asarray(step * direction / norm, jnp.float32))
        path.append(np.asarray(x))
        collided |= bool(float(jnp.asarray(env.sdf(x[None, :]))[0]) < 0.0)
    return {
        "path": np.asarray(path),
        "outcome": outcome,
        "collided": collided,
        "end_distance": float(env.geodesic(x[None, :], start)[0]),
        "length": step * (len(path) - 1),
    }


def evaluate(field_fn, grad_field_fn, env, goals: np.ndarray, step: float, source_value: float,
             arrive_tol: float) -> dict:
    """Descend from every goal and summarise.

    ``arrive_tol`` is applied *after* the fact: a run is a genuine success only if the field stopped
    **and** the endpoint really is near the source. Without that check "reached" is self-certified by
    the field under test, which is unsound in exactly this repo's measured failure mode — a learned
    correction that dips too low creates a spurious low-value basin the descent would halt in.

    Args:
        field_fn: ``x -> T(x)``.
        grad_field_fn: ``x -> ∇T(x)``, closed form where available (never autodiff through ``base``).
        env: Environment.
        goals: Goal points, [n, dim].
        step: Arc length per descent step.
        source_value: Learned-field threshold for stopping.
        arrive_tol: Geodesic distance within which the endpoint counts as the source.

    Returns:
        Dict of per-goal records plus the aggregate counts.
    """
    grad_fn, value_fn = jax.jit(grad_field_fn), jax.jit(field_fn)
    metric_fn = jax.jit(env.metric_inv)
    runs = [descend(grad_fn, value_fn, metric_fn, env, jnp.asarray(g, jnp.float32), step, source_value)
            for g in goals]
    arrived = np.array([r["outcome"] == "reached" and r["end_distance"] <= arrive_tol for r in runs])
    collided = np.array([r["collided"] for r in runs])
    return {
        "runs": runs,
        "arrived": arrived,
        "collided": collided,
        "success": int(np.sum(arrived & ~collided)),
        "collision": int(collided.sum()),
        "failed": int((~arrived).sum()),
    }


def sample_goals(env, n: int, near_fraction: float, seed: int, near_band: float = 0.35,
                 shadow_fraction: float = 0.0) -> np.ndarray:
    """``n`` goals spread over free space, with ``near_fraction`` of them biased onto obstacle
    boundaries and optionally ``shadow_fraction`` forced behind an obstacle.

    Two knobs, two different questions:

    - ``near_fraction`` — how many goals hug an obstacle surface (clearance below ``near_band``).
      The field is steepest there, so a small gradient error points *into* the obstacle rather than
      around it. Sampling uniformly would test the easy case and report a rate that means little.
    - ``shadow_fraction`` — how many are *forced* to sit behind an obstacle, where the source
      geodesic is blocked. Occlusion is the measured failure mode, so it is worth being able to
      demand it; at 0 the draw still contains whatever fraction of occluded points the scene
      naturally has, and ``plan`` reports that fraction rather than assuming it.

    The classes are drawn by index and de-duplicated, so a goal that is both occluded and
    boundary-hugging is counted once and the returned count is exactly ``n``.

    Args:
        env: Environment.
        n: Number of goals.
        near_fraction: Fraction biased onto an obstacle boundary.
        seed: RNG seed.
        near_band: Clearance below which a goal counts as boundary-hugging.
        shadow_fraction: Fraction forced behind an obstacle; 0 leaves occlusion to chance.

    Returns:
        Goal points, [n, dim].
    """
    from srms.environments import sampling

    rng = np.random.default_rng(seed)
    pool = env.sample_domain(rng, 40000)
    clearance = env.sdf_np(pool)
    # Goals already at the source would score as reached without taking a step, so they leave the
    # pool rather than being filtered after the draw — filtering after would silently return fewer
    # than `n` goals and quietly change what a success *rate* is a fraction of.
    away = np.asarray(env.geodesic(jnp.asarray(pool, jnp.float32), jnp.asarray(env.start, jnp.float32))) > 0.5
    index = np.arange(len(pool))
    free = (clearance > 0.05) & away
    shadow = index[free & sampling.occluded_from_source(env, pool)]
    near = index[free & (clearance < near_band)]
    spread = index[(clearance >= near_band) & away]

    picked: list[int] = []
    for group, count in ((shadow, round(n * shadow_fraction)), (near, round(n * near_fraction)), (spread, n)):
        available = np.setdiff1d(group, picked, assume_unique=False)
        take = min(count, len(available), n - len(picked))
        if take > 0:
            picked.extend(available[rng.choice(len(available), take, replace=False)].tolist())
    return pool[np.array(picked)]


def _neighbours(row: int, col: int, shape: tuple[int, int], periodic: bool):
    """The eight grid neighbours of a cell, wrapped or clipped according to the chart."""
    for drow in (-1, 0, 1):
        for dcol in (-1, 0, 1):
            if drow == dcol == 0:
                continue
            r, c = row + drow, col + dcol
            if periodic:
                yield r % shape[0], c % shape[1]
            elif 0 <= r < shape[0] and 0 <= c < shape[1]:
                yield r, c


def descend_on_grid(values: np.ndarray, shape: tuple[int, int], mask: np.ndarray,
                    goal: tuple[int, int], periodic: bool, max_steps: int = 100000) -> dict:
    """Steepest descent on a *gridded* field, cell to lowest neighbour, from ``goal``.

    This is how a path is normally extracted from a fast-marching solution, and it exists here as the
    **reference arm**: run it on the ground-truth field and it says how many goals are reachable by
    descent at all, on this grid, with no model involved. Without that number a learned field's
    success rate has no ceiling to be read against — a failure could be the field's or the method's.

    Discrete rather than interpolated on purpose. Interpolating would need a per-manifold chart rule,
    and the question being asked is about the field's *structure* — does following the downhill
    direction reach the source — which the cell graph answers exactly. It terminates at a local
    minimum by construction, so "stalled" here means a genuine basin of the gridded field.

    Args:
        values: Field on the grid, raveled.
        shape: Grid shape, 2-D.
        mask: True where a cell is out of the domain or inside an obstacle; never entered.
        goal: Starting cell, ``(row, col)``.
        periodic: Whether the chart wraps (the torus does; the sphere and Poincaré charts do not).
        max_steps: Backstop; descent on a finite grid terminates on its own.

    Returns:
        Dict with ``cells`` (the path, [k, 2]) and ``outcome`` (``stalled`` at a local minimum).
    """
    field = np.where(mask.reshape(shape), np.inf, values.reshape(shape))
    row, col = goal
    cells = [(row, col)]
    for _ in range(max_steps):
        best = min(_neighbours(row, col, shape, periodic), key=lambda rc: field[rc])
        if field[best] >= field[row, col]:
            return {"cells": np.array(cells), "outcome": "stalled"}
        row, col = best
        cells.append((row, col))
    return {"cells": np.array(cells), "outcome": "capped"}
