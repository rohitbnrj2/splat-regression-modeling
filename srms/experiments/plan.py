"""Plan on a learned field, against the optimal path as a reference.

**The point of the experiment is the cost asymmetry.** Fitting ``T`` is a one-off minutes-long solve
for a *fixed source*; once it exists, a path to *any* goal is a gradient descent on a closed-form
mixture — no graph, no re-solve, no collision-checking search. This measures both halves and prints
them side by side, because a planner that had to retrain per query would not be making that claim. It
therefore **loads** trained parameters (``--params-path``); ``srms.experiments.sweep`` writes a
``.pkl`` next to every figure. Passing no path retrains, and the printed timing says so.

**Every goal is planned twice.** Once by descending the learned field, and once by descending the
fast-marching field on its own grid — the optimal route. Both are drawn from the same goals, so a
failure is legible: the left panel shows where the path *should* have gone, the right where the
learned field sent it instead. Without the reference arm a success rate has no ceiling to be read
against, and a failure could belong to the field or to the grid.

Descent on the learned field is Riemannian and reads only that field: the direction is
``-metric_inv · ∇T`` and the stopping rule is ``T(x) < source_value``, the field's own estimate of
remaining travel time. Obstacle geometry and the endpoint's true distance to the source are recorded
so a claim of success can be checked, never to steer — see ``paths.descend``. Ground truth enters
only after the learned field has produced every path it is going to produce.

Run:
    python -m srms.experiments.plan --params-path results/figures/exp3/torus_labelfree/torus_obs3_seed1.pkl
    python -m srms.experiments.plan --environment torus --num-obstacles 3   # retrain, then plan
"""

from __future__ import annotations

import dataclasses
import pathlib
import pickle
import time

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

from srms.environments import sampling
from srms.experiments import paths
from srms.experiments.sweep import SweepConfig, _config
from srms.methods.backends import BACKENDS
from srms.methods.strategies import factored
from srms.run import Config, _build_env

_CONTOURS = 12
_SNAP_CHUNK = 64  # goals per nearest-cell lookup; the full [goals, cells] matrix is large


@dataclasses.dataclass
class PlanConfig(SweepConfig):
    """A sweep configuration plus the planning-specific knobs."""

    params_path: str = ""  # trained parameters from a sweep run; empty retrains the configuration
    # `SweepConfig` carries `seeds`, a *count*, because a sweep trains one field per seed. A plan
    # runs against exactly one field, so it needs the seed itself; they are not the same field and
    # reusing `seeds` here is what silently broke the retrain path.
    seed: int = 1
    num_goals: int = 60
    near_fraction: float = 0.5  # half the goals hug an obstacle, where descent is hardest
    shadow_fraction: float = 0.0  # 0 leaves occlusion at the scene's natural rate, and reports it
    descent_step: float = 0.02
    source_value: float = 0.30  # > Config.source_radius: inside it the PDE never saw a collocation point
    arrive_tol: float = 0.45  # endpoint must really be this close to the source (measurement, not steering)
    goal_seed: int = 7  # separate from `seed`, so goals do not move when the scene or model does
    out_path: str = "results/figures/plan/torus_paths.png"


def load_or_train(plan: PlanConfig) -> tuple[Config, object, float, bool]:
    """Return ``(cfg, params, seconds, trained)``, loading trained parameters when one is given.

    The saved ``cfg`` wins over the command line for everything defining the *scene and model*, since
    parameters are only meaningful against the configuration that produced them — loading a
    3-obstacle field into a 1-obstacle scene would plan through obstacles the field never saw.
    """
    if not plan.params_path:
        cfg = _config(plan, plan.seed)
        env = _build_env(cfg)
        started = time.time()
        params = factored.solve(env, cfg, BACKENDS[cfg.backend])
        return cfg, params, time.time() - started, True
    with open(plan.params_path, "rb") as handle:
        saved = pickle.load(handle)
    cfg = Config(**saved["cfg"])
    started = time.time()
    params = jax.tree_util.tree_map(jnp.asarray, saved["params"])
    return cfg, params, time.time() - started, False


def snap(env, grid: jnp.ndarray, goals: np.ndarray) -> np.ndarray:
    """Nearest grid cell to each goal, as a raveled index, [n].

    The reference descent runs on the grid, so a continuous goal has to be placed on it. At
    resolution 240 the torus cell is 0.026 across, three orders below the travel times involved.
    """
    out = []
    for lo in range(0, len(goals), _SNAP_CHUNK):
        batch = jnp.asarray(goals[lo : lo + _SNAP_CHUNK], jnp.float32)
        out.append(np.asarray(jax.vmap(lambda g: jnp.argmin(env.geodesic(grid, g)))(batch)))
    return np.concatenate(out)


def optimal_paths(env, cfg: Config, truth: np.ndarray, shape, blocked: np.ndarray,
                  cells: np.ndarray, arrive_tol: float) -> tuple[list, np.ndarray]:
    """Descend the fast-marching field from each goal cell — the route the learned field should find."""
    periodic = cfg.environment == "torus"
    grid, _ = env.grid(cfg.resolution)
    start = jnp.asarray(env.start, jnp.float32)
    runs, arrived = [], []
    for flat in cells:
        run = paths.descend_on_grid(
            truth, shape, blocked, (int(flat) // shape[1], int(flat) % shape[1]), periodic
        )
        last = run["cells"][-1]
        end = grid[int(last[0]) * shape[1] + int(last[1])]
        run["end_distance"] = float(env.geodesic(end[None, :], start)[0])
        runs.append(run)
        arrived.append(run["end_distance"] <= arrive_tol)
    return runs, np.array(arrived)


def _panel(ax, field, shape, blocked, extent, levels, title):
    """One field panel: the field, its level sets, and the obstacle outlines."""
    ax.imshow(np.where(blocked, np.nan, field), origin="lower", extent=extent, cmap="viridis")
    mesh_x = np.linspace(extent[0], extent[1], shape[1])
    mesh_y = np.linspace(extent[2], extent[3], shape[0])
    # Level sets are what make a descent readable: -grad T is perpendicular to them, so a path that
    # crosses contours at an angle is being bent by the correction, and a path circling *inside* a
    # closed contour is sitting in a spurious basin rather than descending.
    ax.contour(mesh_x, mesh_y, field, levels=levels, colors="white", linewidths=0.7, alpha=0.85)
    ax.contour(mesh_x, mesh_y, blocked.astype(float), levels=[0.5], colors="black", linewidths=1.5)
    ax.set_title(title, fontsize=12)


def _trace(ax, points, colour, seam):
    """Draw one path, split wherever it crosses the chart's periodic seam."""
    jump = np.abs(np.diff(points, axis=0)).max(axis=1) > seam
    for segment in np.split(points, np.where(jump)[0] + 1):
        if len(segment) > 1:
            ax.plot(segment[:, 0], segment[:, 1], lw=1.3, color=colour)


def draw(plan: PlanConfig, cfg: Config, env, result: dict, learned, truth, shape,
         reference: list, reference_ok: np.ndarray) -> None:
    """Two panels over the same goals: the optimal route, and the route the learned field produced."""
    grid, _ = env.grid(cfg.resolution)
    field = np.asarray(learned).reshape(shape)
    truth_img = np.asarray(truth).reshape(shape)
    blocked = (np.asarray(env.sdf(grid)) < 0.0).reshape(shape)
    extent = env.render_extent
    # Paths live in chart coordinates; the render extent tells us the chart's own units.
    to_chart = (lambda p: np.degrees(p)) if abs(extent[1]) > 10 else (lambda p: p)
    seam = 0.5 * (extent[1] - extent[0])
    # Shared levels, so contour *spacing* is comparable between the panels — that comparison is what
    # shows gradient error, which the colour map alone hides.
    levels = np.linspace(0.0, float(np.nanmax(np.where(blocked, np.nan, truth_img))), _CONTOURS + 2)[1:-1]
    cell_x = np.linspace(extent[0], extent[1], shape[1])
    cell_y = np.linspace(extent[2], extent[3], shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0))
    _panel(axes[0], truth_img, shape, blocked, extent, levels, "fast marching (GT) — optimal paths")
    _panel(axes[1], field, shape, blocked, extent, levels,
           "learned $T = \\mathrm{base}\\cdot e^{g}$ — paths it actually produces")

    for run, ok in zip(reference, reference_ok):
        points = np.column_stack([cell_x[run["cells"][:, 1]], cell_y[run["cells"][:, 0]]])
        _trace(axes[0], points, "white" if ok else "orange", seam)
        axes[0].plot(points[0, 0], points[0, 1], "o", ms=4, color="cyan",
                     markeredgecolor="black", markeredgewidth=0.5)
    for run, ok in zip(result["runs"], result["arrived"] & ~result["collided"]):
        points = to_chart(run["path"])
        _trace(axes[1], points, "white" if ok else ("red" if run["collided"] else "orange"), seam)
        axes[1].plot(points[0, 0], points[0, 1], "o", ms=4, color="cyan",
                     markeredgecolor="black", markeredgewidth=0.5)

    for ax in axes:
        ax.plot(*env.render_marker_deg(), marker="*", ms=19, color="#ffe14d",
                markeredgecolor="black", markeredgewidth=0.9)
        ax.set_xlabel(env.axis_labels[0], fontsize=11)
        ax.set_ylabel(env.axis_labels[1], fontsize=11)
    n = len(result["runs"])
    fig.suptitle(
        f"{cfg.environment}, {cfg.num_obstacles} obstacles, {n} goals — "
        f"optimal {int(reference_ok.sum())}/{n}, learned {result['success']}/{n} "
        f"({result['collision']} collisions).   white = reached, orange = did not arrive, red = collided",
        fontsize=12,
    )
    pathlib.Path(plan.out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    stem = plan.out_path.rsplit(".", 1)[0]
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png and {stem}.pdf")


def main(plan: PlanConfig) -> None:
    """Load (or fit) the field, plan to every goal twice, print outcomes and timings, draw both."""
    cfg, params, setup_seconds, trained = load_or_train(plan)
    env = _build_env(cfg)
    backend = BACKENDS[cfg.backend]
    grid, shape = env.grid(cfg.resolution)

    goals = paths.sample_goals(env, plan.num_goals, plan.near_fraction, plan.goal_seed,
                               shadow_fraction=plan.shadow_fraction)
    started = time.time()
    result = paths.evaluate(
        lambda x: factored.field(backend, params, x, env, cfg),
        # Closed-form gradient: `base` must never be autodiffed (unbounded at the cut locus).
        lambda x: factored.time_grad(backend, params, x, env, cfg),
        env, goals, plan.descent_step, plan.source_value, plan.arrive_tol,
    )
    plan_seconds = time.time() - started

    # The reference arm. Ground truth is read here and only here, after the learned field has already
    # produced every path it is going to produce.
    truth = np.asarray(env.ground_truth(cfg.resolution))
    blocked = (np.asarray(env.sdf(grid)) < 0.0).reshape(shape)
    reference, reference_ok = optimal_paths(env, cfg, truth, shape, blocked,
                                            snap(env, grid, goals), plan.arrive_tol)

    clearance = env.sdf_np(goals)
    occluded = sampling.occluded_from_source(env, goals)
    straight = np.asarray(env.geodesic(jnp.asarray(goals, jnp.float32), jnp.asarray(env.start, jnp.float32)))

    n = len(goals)
    print(f"\nPath extraction — {cfg.environment}, {cfg.num_obstacles} obstacles, {n} goals")
    print("Direction and stopping come from the learned field; obstacle geometry and the endpoint")
    print("distance are recorded only, never used to steer. `optimal` descends fast marching.\n")
    print(f"{'goal':>4} {'clearance':>10} {'in shadow':>10} {'optimal':>8} {'outcome':>9} "
          f"{'collided':>9} {'end dist':>9} {'len':>7} {'geodesic':>9}")
    for i, (run, clear, base, dark, best) in enumerate(
        zip(result["runs"], clearance, straight, occluded, reference_ok)
    ):
        print(f"{i:>4} {clear:>10.3f} {bool(dark)!s:>10} {('yes' if best else 'NO'):>8} "
              f"{run['outcome']:>9} {run['collided']!s:>9} {run['end_distance']:>9.3f} "
              f"{run['length']:>7.2f} {base:>9.3f}")

    ok = result["arrived"] & ~result["collided"]
    print(f"\n  optimal (fast marching, descended on its own grid): {int(reference_ok.sum())}/{n}")
    print(f"  learned  (arrived AND collision-free):               {result['success']}/{n} "
          f"= {100 * result['success'] / n:.0f}%")
    print(f"  collisions {result['collision']}   did not arrive {result['failed']}")
    # State the mix, always. 10/10 on goals that are all in open free space is a much weaker claim
    # than 10/10 with half of them behind an obstacle, and the two are indistinguishable without it.
    hugging = (clearance < 0.35) & ~occluded  # disjoint: a goal behind an obstacle counts as shadow
    print(f"  goal mix: {int(occluded.sum())} behind an obstacle, {int(hugging.sum())} hugging one, "
          f"{int(np.sum(~occluded & ~hugging))} in open free space")
    outcomes: dict[str, int] = {}
    for run in result["runs"]:
        outcomes[run["outcome"]] = outcomes.get(run["outcome"], 0) + 1
    print("  outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items())))
    for name, group in (("shadow", occluded), ("boundary", hugging), ("open", ~occluded & ~hugging)):
        if group.any():
            print(f"    {name:<9} learned {int(np.sum(group & ok)):>3}/{int(group.sum()):<3}   "
                  f"optimal {int(np.sum(group & reference_ok)):>3}/{int(group.sum()):<3}")

    # The asymmetry this experiment exists to show. The first number is paid once per *source*; the
    # second is what a query costs, and it includes JIT warm-up on the first goal.
    label = "fit the field (one-off, per source)" if trained else "load trained parameters"
    print(f"\n  {label:<38} {setup_seconds:>8.2f} s")
    print(f"  {'plan ' + str(n) + ' paths':<38} {plan_seconds:>8.2f} s "
          f"({1e3 * plan_seconds / n:.0f} ms per goal, JIT warm-up included)")
    draw(plan, cfg, env, result, factored.predict(backend, params, grid, env, cfg),
         truth, shape, reference, reference_ok)


if __name__ == "__main__":
    main(tyro.cli(PlanConfig))
