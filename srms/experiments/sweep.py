"""The experiment ladder: one self-supervised solve per seed, parameterised by obstacle count.

Experiments 1, 2 and 3 are **the same experiment** at ``--num-obstacles 0``, ``1`` and more. Nothing
else changes: same field, same loss, same optimiser, same budget. Only one thing genuinely differs,
and it is about measurement rather than method —

**the scoring reference.** With no obstacles the true field is the analytic geodesic distance from the
source, exact and closed form. With obstacles there is no closed form, so the reference becomes fast
marching, a discretised solver carrying its own error. That error is measured per scene by removing
the obstacles, where the exact answer *is* the geodesic, so it is a property of the grid and not of
any model. It is printed as ``ref err`` and **nothing below it is interpretable**: 0.046 on the torus
at resolution 120, 0.027 at 240. Run obstacle scenes at 240 or finer.

A second consequence: ``cost@truth`` — the objective evaluated at the exact answer, which is what
separates "the objective is wrong" from "the solver did not converge" — exists only at zero
obstacles. Do **not** substitute a fitted field for it when obstacles are present; that produced a
retracted conclusion earlier in this project (see ``investigation.md``). The column simply reports
``—`` instead.

Training never sees any of this. It reads the scene (``slowness``, ``sdf``), the analytic geodesic,
and its own collocation samples; the reference is computed only after ``solve`` returns.

Columns:

- ``nothing``          — the error of ``T = base``, i.e. of learning nothing at all. Every comparison
  across manifolds is an improvement over *this*, never an absolute RMS, because scene difficulty
  varies by an order of magnitude with curvature.
- ``init`` / ``RMS``   — error before and after training. ``init`` is a random perturbation of the
  correction ``g``, not the answer, so the pair shows a genuine recovery rather than a held fixed
  point.
- ``max`` / ``MAE``    — worst cell and mean absolute error.
- ``MAE far``          — over cells past half the maximum travel time *among scored cells*; errors
  accumulate outward from the source, so a gap opening against ``MAE`` means the fit is only locally
  good. The maximum must exclude obstacle interiors, where ``slowness_max`` inflates ``T`` well past
  anything in free space and would set the threshold from the obstacle rather than the manifold.
- ``MAE shadow``       — over cells whose straight source-to-point ray crosses an obstacle, found from
  the scene's own signed distance function with no solver. This is where the Eikonal residual pins
  the *slope* of the field but leaves its *level* free, and historically where pure-physics runs lose
  accuracy. Empty when there are no obstacles.

Run:
    python -m srms.experiments.sweep --num-obstacles 0 --seeds 5                    # Experiment 1
    python -m srms.experiments.sweep --num-obstacles 1 --seeds 5 --resolution 240   # Experiment 2
    python -m srms.experiments.sweep --num-obstacles 5 --seeds 5 --resolution 240   # Experiment 3
"""

from __future__ import annotations

import dataclasses
import pathlib
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from srms.environments import sampling
from srms.experiments import figures
from srms.methods.backends import BACKENDS
from srms.methods.strategies import eikonal, ntfields
from srms.run import STRATEGIES, Config, _build_env

COST_BATCH = 4096
_FACTORED = ("ntfields", "pntfields", "hntfields")


@dataclasses.dataclass
class SweepConfig:
    """One sweep: ``num_obstacles`` selects which experiment this is."""

    environment: str = "torus"
    method: str = "factored"
    backend: str = "srm"
    dim: int = 2
    num_obstacles: int = 0  # 0 = Exp 1, 1 = Exp 2, more = Exp 3
    seeds: int = 5
    steps: int = 1500
    num_splats: int = 512
    num_collocation: int = 1024
    init_weight: float = 1e-2
    resolution: int = 120
    densify: bool = True  # repo default; with it on, `num_splats` is unused (growth starts at init_splats)
    nonneg_weight: float = 1.0
    tau_bias: float = 4.0
    causal: bool = True
    causal_anneal: bool = True
    scale_floor: float = 0.07
    rrt_iters: int = 1500  # 350 leaves the tree under-converged: anchor error 0.264 vs 0.029 here
    num_anchors: int = 0  # sparse RRT* weak supervision; 0 disables
    anchor_mode: str = "bounds"  # "equality" = RRT* costs pinned; "bounds" = sphere-pack two-sided hinge
    anchor_weight: float = 1e-2  # historical lambda_R; a lower bound can be invalid, so it must nudge not dictate
    anchor_shadow_pref: float = 3.0  # "equality" mode: draw bias toward occluded nodes; 0 = uniform control
    bound_slack: float = 0.10  # lower bound = graph cost x (1 - slack), absorbing roadmap suboptimality
    variational: bool = False  # maximise T subject to ‖∇T‖<=s (largest subsolution) instead of (q-1)^2
    variational_lambda: float = 100.0
    trunc_sigma: float = 0.0  # >0 gives each splat compact support, exactly 0 beyond this many sigma
    l1_weight: float = 0.0  # L1 on the mixture weights; with the V>=0 clamp this is LASSO+projection
    nonneg_weights: bool = False
    max_aspect: float = 0.0
    spawn_scale: float = 0.3
    densify_every: int = 400
    obstacle_radius: tuple[float, float] = (0.5, 0.9)
    slow_width: float = 0.15
    slowness_max: float = 10.0
    figures: bool = True
    error_clip: float = 0.05
    out_dir: str = "figures/sweep"


def _config(sweep: SweepConfig, seed: int) -> Config:
    """Config for one seed; the seed places the obstacles as well as initialising the model."""
    return Config(
        environment=sweep.environment,
        method=sweep.method,
        backend=sweep.backend,
        dim=sweep.dim,
        num_obstacles=sweep.num_obstacles,
        seed=seed,
        steps=sweep.steps,
        num_splats=sweep.num_splats,
        num_collocation=sweep.num_collocation,
        init_weight=sweep.init_weight,
        resolution=sweep.resolution,
        densify=sweep.densify,
        nonneg_weight=sweep.nonneg_weight,
        tau_bias=sweep.tau_bias,
        causal=sweep.causal,
        causal_anneal=sweep.causal_anneal,
        scale_floor=sweep.scale_floor,
        nonneg_weights=sweep.nonneg_weights,
        l1_weight=sweep.l1_weight,
        trunc_sigma=sweep.trunc_sigma,
        rrt_iters=sweep.rrt_iters,
        num_anchors=sweep.num_anchors,
        anchor_mode=sweep.anchor_mode,
        anchor_weight=sweep.anchor_weight,
        anchor_shadow_pref=sweep.anchor_shadow_pref,
        bound_slack=sweep.bound_slack,
        hnt_nodes=sweep.num_anchors,
        variational=sweep.variational,
        variational_lambda=sweep.variational_lambda,
        max_aspect=sweep.max_aspect,
        spawn_scale=sweep.spawn_scale,
        densify_every=sweep.densify_every,
        obstacle_radius=sweep.obstacle_radius,
        slow_width=sweep.slow_width,
        slowness_max=sweep.slowness_max,
    )


def error_metrics(prediction: jnp.ndarray, truth: jnp.ndarray) -> dict:
    """RMS, max absolute and mean absolute error between two fields, [n] each."""
    err = jnp.abs(prediction - truth)
    return {
        "rms": float(jnp.sqrt(jnp.mean((prediction - truth) ** 2))),
        "max_abs": float(jnp.max(err)),
        "mae": float(jnp.mean(err)),
    }


def reference(env, cfg: Config, points: jnp.ndarray) -> tuple[jnp.ndarray, float]:
    """The field to score against, and an honest estimate of that reference's own error.

    Returns the analytic geodesic (error 0, exact) when the scene has no obstacles, and fast marching
    otherwise.

    The error is measured obstacle-free, where the exact answer is known — but at a **generic** source
    position, not the configured one. Measured on S^2 at resolution 240: with the source at the pole
    the marcher reports an error of **0.0000**, because the grid is geodesic-polar about that pole and
    the marcher is discretising a 1-D radial problem exactly; move the source 45 degrees off and the
    same grid reports 0.0134, at the equator 0.0166. The configured start would therefore report a
    floor of zero for a solver that has real error as soon as an obstacle breaks the radial symmetry.
    Taking the worst over several source positions is conservative and manifold-agnostic.
    """
    if cfg.num_obstacles == 0:
        return env.geodesic(points, jnp.asarray(env.start, jnp.float32)), 0.0
    free_cfg = dataclasses.replace(cfg, num_obstacles=0)
    free_env = _build_env(free_cfg)
    free_points, _ = free_env.grid(cfg.resolution)
    rng = np.random.default_rng(0)
    scores = []
    for probe in free_env.sample_domain(rng, 6):
        try:
            probed = _build_env(dataclasses.replace(free_cfg, start=tuple(float(v) for v in probe)))
        except ValueError:
            continue  # a sampled chart point can fall outside the domain (Poincaré ball corners)
        marched = jnp.asarray(probed.ground_truth(cfg.resolution))
        ok = jnp.isfinite(marched)
        if not bool(jnp.any(ok)):
            continue
        exact = probed.geodesic(free_points, jnp.asarray(probed.start, jnp.float32))
        scores.append(error_metrics(marched[ok], exact[ok])["rms"])
        if len(scores) == 3:
            break
    return jnp.asarray(env.ground_truth(cfg.resolution)), max(scores) if scores else float("nan")


def shadow_mask(env, points: jnp.ndarray) -> np.ndarray:
    """True where the geodesic from the source to the point passes through an obstacle, [n].

    Delegates to ``environments.sampling.occluded_from_source``, which is the single implementation:
    the anchor selector asks the same question, and this repo has already paid twice for the same
    geometric predicate existing in two places with two different answers.
    """
    return sampling.occluded_from_source(env, points)


def predict(cfg, backend, params, points, env) -> jnp.ndarray:
    """Evaluate the strategy's field, rejecting strategies this module cannot serve."""
    if cfg.method in _FACTORED:
        return STRATEGIES[cfg.method].predict(backend, params, points, env, cfg.tau_bias, cfg.tau_min)
    if cfg.method == "weak_supervision":
        raise NotImplementedError("sweep: `weak_supervision` needs an RRT* roadmap this module does not build.")
    if cfg.method == 'factored':
        return STRATEGIES[cfg.method].predict(backend, params, points, env, cfg)
    return STRATEGIES[cfg.method].predict(backend, params, points, env)


def _batch(env, cfg):
    """One fixed collocation batch and boundary ring, shared by every cost evaluation."""
    colloc = eikonal.sample_collocation(env, np.random.default_rng(0), COST_BATCH, cfg.source_radius)
    return colloc, eikonal.source_sphere(env, cfg.source_radius, cfg.n_sphere, 0)


def cost(strategy, backend, params, env, cfg) -> float | None:
    """The strategy's objective at the given parameters, on the shared batch."""
    colloc, (src_pts, src_vals) = _batch(env, cfg)
    if cfg.method in _FACTORED:
        q = ntfields.speed_ratio(backend, params, colloc, env.slowness(colloc), env, cfg.tau_bias, cfg.tau_min)
        return float(jnp.mean(ntfields.isotropic_loss(q)))
    if not hasattr(strategy, "objective"):
        return None
    total, _ = strategy.objective(backend, params, env, cfg, colloc, env.slowness(colloc), src_pts, src_vals)
    return float(total)


def cost_at_truth(env, cfg) -> float | None:
    """The objective at the closed-form answer — no model, no fit. None once obstacles exist.

    With obstacles there is no exact field to substitute, and substituting a *fitted* one is the
    error that produced a retracted conclusion in this project. The column is dropped instead.
    """
    if cfg.num_obstacles != 0:
        return None
    colloc, (src_pts, src_vals) = _batch(env, cfg)
    start = jnp.asarray(env.start, jnp.float32)
    grad = jax.vmap(env.grad_geodesic)(colloc)
    metric = jax.vmap(env.metric_inv)(colloc)
    grad_norm = jnp.sqrt(jnp.einsum("ni,nij,nj->n", grad, metric, grad) + 1e-12)
    slow = env.slowness(colloc)
    if cfg.method in _FACTORED:
        return float(jnp.mean(ntfields.isotropic_loss(grad_norm / slow)))
    pde = jnp.mean((grad_norm - slow) ** 2)
    bc = jnp.mean((env.geodesic(src_pts, start) - src_vals) ** 2)
    return float(bc + cfg.physics_weight * pde)


def run_seed(sweep: SweepConfig, seed: int) -> dict:
    """Train one scene self-supervised, then score it. Ground truth enters only after ``solve``."""
    cfg = _config(sweep, seed)
    env = _build_env(cfg)
    strategy, backend = STRATEGIES[cfg.method], BACKENDS[cfg.backend]

    points, shape = env.grid(cfg.resolution)
    init_params = backend.init_params(jax.random.PRNGKey(cfg.seed), env, cfg)
    init_field = predict(cfg, backend, init_params, points, env)

    params = strategy.solve(env, cfg, backend)
    field = predict(cfg, backend, params, points, env)

    truth, ref_err = reference(env, cfg, points)
    inside = np.asarray(env.sdf(points)) < 0.0  # obstacle interiors and out-of-chart cells
    keep = np.asarray(jnp.isfinite(truth)) & ~inside
    sel = jnp.asarray(keep)

    metrics = error_metrics(field[sel], truth[sel])
    # `do nothing` — the error of T = base, i.e. of learning nothing at all. Every comparison in the
    # write-up is an improvement *over this*, because scene difficulty varies by an order of
    # magnitude across curvatures, and quoting it per run is what stops a number being reported
    # without the baseline it was measured against (this log has one such orphan already).
    do_nothing = error_metrics(
        env.geodesic(points, jnp.asarray(env.start, jnp.float32))[sel], truth[sel]
    )["rms"]
    err = jnp.abs(field - truth)
    # Threshold from the *scored* cells: obstacle interiors reach T ~ 9 at slowness_max=10 while free
    # space tops out at 5.2, so taking the max over all cells sets the bar from the obstacle and
    # selects 0.4% of cells instead of the intended far half (52.6%).
    far = np.asarray(truth > 0.5 * jnp.max(truth[sel]))
    shadow = shadow_mask(env, points)
    mean_over = lambda m: float(jnp.mean(err[jnp.asarray(keep & m)])) if (keep & m).any() else float("nan")

    if sweep.figures:
        # Create the output directory here rather than trusting the caller: without it every run
        # trains to completion and then dies on the first write, and a shell pipeline that filters
        # for result lines swallows the traceback — four Experiment-3 arms were lost that way.
        pathlib.Path(sweep.out_dir).mkdir(parents=True, exist_ok=True)
        # Save the raw fields alongside the figure so a restyle never requires retraining.
        stem = f"{sweep.out_dir}/{sweep.environment}_obs{sweep.num_obstacles}_seed{seed}"
        np.savez_compressed(
            f"{stem}.npz",
            truth=np.asarray(truth), prediction=np.asarray(field), mask=inside, shape=np.asarray(shape),
        )
        # And the trained parameters, which the fields alone cannot substitute for: planning needs
        # `T` and `grad T` at arbitrary points, not at grid cells, and re-solving to get them would
        # hide the thing the learned field is *for* — the query is milliseconds, the fit is minutes.
        with open(f"{stem}.pkl", "wb") as handle:
            pickle.dump(
                {"params": jax.tree_util.tree_map(np.asarray, params),
                 "cfg": dataclasses.asdict(cfg),
                 "obstacles": env.obstacles},
                handle,
            )
        figures.save(
            env,
            f"{sweep.out_dir}/{sweep.environment}_obs{sweep.num_obstacles}_seed{seed}.png",
            truth,
            field,
            shape,
            mask=inside if inside.any() else None,
            error_clip=sweep.error_clip,
            reference="analytic geodesic" if sweep.num_obstacles == 0 else "fast marching (GT)",
            title=f"{sweep.environment} — {sweep.num_obstacles} obstacle(s), self-supervised, "
            f"seed {seed} (RMS {metrics['rms']:.4f})",
        )
    return {
        "seed": seed,
        "init": error_metrics(init_field[sel], truth[sel])["rms"],
        "do_nothing": do_nothing,
        "ref_err": ref_err,
        "mae_far": mean_over(far),
        "mae_shadow": mean_over(shadow),
        "cost_trained": cost(strategy, backend, params, env, cfg),
        "cost_truth": cost_at_truth(env, cfg),
        **metrics,
    }


def report(sweep: SweepConfig, rows: list[dict]) -> None:
    """Print the per-seed table and the verdict."""
    which = {0: "1 (no obstacles)", 1: "2 (one obstacle)"}.get(sweep.num_obstacles, f"3 ({sweep.num_obstacles} obstacles)")
    print(f"\nExperiment {which} — {sweep.environment}, {sweep.method}, resolution {sweep.resolution}")
    print(f"{'seed':>4} {'nothing':>9} {'init':>9} {'RMS':>9} {'max':>9} {'MAE':>9} {'MAE far':>9} "
          f"{'MAE shadow':>11} {'cost':>10}")
    print("-" * 84)
    for r in rows:
        shadow = "—" if np.isnan(r["mae_shadow"]) else f"{r['mae_shadow']:.4f}"
        print(f"{r['seed']:>4} {r['do_nothing']:>9.4f} {r['init']:>9.4f} {r['rms']:>9.4f} "
              f"{r['max_abs']:>9.4f} {r['mae']:>9.4f} {r['mae_far']:>9.4f} {shadow:>11} "
              f"{r['cost_trained']:>10.2e}")
    values = np.array([r["rms"] for r in rows])
    nothing = np.array([r["do_nothing"] for r in rows])
    ref_err = rows[0]["ref_err"]
    print(f"\n  mean RMS {values.mean():.4f} ± {values.std():.4f} over {len(rows)} seeds")
    # With no obstacles `T = base` *is* the answer, so `do nothing` is 0 and the ratio is undefined
    # rather than infinite — printing "0.00x improvement" there would read as a catastrophic result.
    if nothing.mean() > 1e-9:
        print(f"  do nothing (T = base) {nothing.mean():.4f} -> learned {values.mean():.4f} "
              f"= {nothing.mean() / max(values.mean(), 1e-12):.2f}x improvement")
    else:
        print("  do nothing (T = base) is exact here: with no obstacles the geodesic IS the answer, "
              "so this scores the machinery, not obstacle reasoning.")

    if ref_err == 0.0:
        print("  reference: the analytic geodesic — exact, so every digit above is real.")
        truth_costs = [(r["cost_trained"], r["cost_truth"]) for r in rows if r["cost_truth"] is not None]
        if truth_costs and sum(t < c for t, c in truth_costs):
            print("  VERDICT: the objective scores the trained field BELOW the exact answer — its minimum\n"
                  "  is not the answer, a formulation problem no optimizer can fix.")
        else:
            print("  VERDICT: the objective's minimum is the exact answer, so any residual gap is the solver.")
    else:
        print("  reference: fast marching = the ground truth.")
        shadow, overall = np.nanmean([r["mae_shadow"] for r in rows]), np.nanmean([r["mae"] for r in rows])
        print(f"  MAE in shadow / MAE overall = {shadow / overall:.2f}x "
              "(>1 means the level behind obstacles is where the error lives)")


def main(sweep: SweepConfig) -> None:
    """Run the sweep over ``sweep.seeds`` seeds and print the table."""
    report(sweep, [run_seed(sweep, seed) for seed in range(1, sweep.seeds + 1)])


if __name__ == "__main__":
    main(tyro.cli(SweepConfig))
