"""Our own field: ``T(x) = base(x) · exp(g(x))``, trained on the Eikonal residual.

Symbols (see CLAUDE.md "Notation"): ``x`` a point on the manifold; ``T(x)`` the time-to-go;
``base(x)`` the analytic geodesic distance from the source, in closed form; ``g(x)`` the splat
mixture's scalar output, i.e. the learned correction; ``s(x)`` the slowness, cost per unit length,
``>= 1``; ``q = ‖∇T‖_g / s`` the speed ratio, which is 1 exactly when the Eikonal equation holds.

**Why this factorization and not NTFields' ``T = base/tau``.** NTFields writes
``tau = sigmoid(g + bias)``, which lies in the *open* interval (0, 1), so ``T = base/tau`` is
*strictly* greater than ``base`` and the free-space answer is unreachable. Measured on the
obstacle-free torus: ``sigmoid(4) = 0.98201`` gives a uniform 1.83% overestimate at every distance
from the source, and training stalls at RMS 0.030 because reaching the answer needs ``g -> +inf``.
Here ``g = 0`` gives ``T = base`` **exactly**, so free space is representable rather than asymptotic.

**Why ``exp(g)`` and not ``1 + relu(g)``.** Both make ``T = base`` attainable, and ``relu`` would
additionally enforce ``T >= base`` (correct whenever ``s >= 1``, since travel can never beat the
free-space geodesic). But ``relu`` is flat for ``g < 0``: a correction that goes negative has zero
gradient and can never return — the same dead-gradient failure that made the unfactored field
untrainable from ``V = 0``. ``exp`` is smooth everywhere, has derivative 1 at ``g = 0``, so it is
well conditioned exactly at the free-space solution, and matches ``weak_supervision.py``'s existing
``base · exp(correction)``. The ``T >= base`` constraint is left to the physics rather than hardcoded.

``T(start) = 0`` holds by construction since ``base(start) = 0``, for any finite ``g``.

**The correction is non-negative, and that has to be enforced.** With ``s >= 1`` everywhere, travel
time can never beat the free-space geodesic, so ``T >= base`` and therefore ``g >= 0`` — verified on
the one-obstacle torus, where the true ``g = log(T/base)`` spans ``[0, 0.763]`` over 55,675 cells
with no negative value. ``exp`` does not know this: measured after training, the learned ``g`` was
negative in **44.5%** of free-space cells and **30.7%** of shadow cells, averaging **0.277 below**
truth in the shadow — the optimizer descends an unphysical direction the parameterization leaves
open, which is exactly the large under-prediction region the error maps show.

No smooth ``psi(g)`` can both attain ``psi = 1`` and stay ``>= 1``: attaining a minimum means
``psi' = 0`` there, a dead gradient at the free-space solution. NTFields' ``T = base/tau`` picks the
constraint (``tau`` in (0,1) makes ``T > base`` structurally) and gives up exact attainment — which is
why it *beats* this field with an obstacle (0.2027 vs 0.2737) while losing badly without one (0.0298
vs 0.0015). ``cfg.nonneg_weight`` takes both: ``exp`` keeps ``g = 0`` exactly attainable and
well conditioned, and a one-sided penalty ``mean(relu(-g)^2)`` supplies the constraint. The penalty
is exactly zero, with zero gradient, wherever the constraint holds, so the obstacle-free case is
untouched.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import trange

from srms.methods.strategies import training_aids
from srms.methods.strategies.eikonal import sample_collocation


def field(backend, params, x: jnp.ndarray, env, cfg=None) -> jnp.ndarray:
    """``T(x) = base(x) · exp(g(x))`` at a single point.

    Args:
        backend: Backend module exposing ``eval_raw``.
        params: Backend parameters.
        x: One point on the manifold, [dim].
        env: Environment supplying ``geodesic`` and ``start``.

    Returns:
        Scalar time-to-go.
    """
    base = env.geodesic(x, jnp.asarray(env.start, dtype=jnp.float32))
    return base * jnp.exp(backend.eval_raw(params, x[None, :], env, cfg)[0, 0])


def predict(backend, params, points: jnp.ndarray, env, cfg=None) -> jnp.ndarray:
    """Evaluate ``T`` over a batch of points, [n, dim] -> [n]."""
    return jax.vmap(lambda x: field(backend, params, x, env, cfg))(points)


def time_grad(backend, params, x: jnp.ndarray, env, cfg=None) -> jnp.ndarray:
    """``∇T`` at one point, via the product rule on ``T = base·exp(g)``.

    ``∇T = exp(g)·(∇base + base·∇g)``. Only ``g`` is differentiated, and ``g`` is a smooth mixture of
    Gaussians; ``∇base`` comes from ``env.grad_geodesic`` in closed form, never autodiff (see base.py).
    """
    start = jnp.asarray(env.start, dtype=jnp.float32)
    base = env.geodesic(x, start)
    correction = backend.eval_raw(params, x[None, :], env, cfg)[0, 0]
    grad_g = jax.grad(lambda y: backend.eval_raw(params, y[None, :], env, cfg)[0, 0])(x)
    return jnp.exp(correction) * (env.grad_geodesic(x) + base * grad_g)


def speed_ratio(backend, params, points: jnp.ndarray, slow: jnp.ndarray, env, cfg=None) -> jnp.ndarray:
    """``q = ‖∇T‖_g / s`` at each point; ``q = 1`` solves the Eikonal equation.

    Args:
        backend: Backend module.
        params: Backend parameters.
        points: Collocation points, [n, dim].
        slow: Slowness at those points, [n].
        env: Environment supplying ``metric_inv``.

    Returns:
        Speed ratio, [n].
    """
    grad = jax.vmap(lambda x: time_grad(backend, params, x, env, cfg))(points)
    metric = jax.vmap(env.metric_inv)(points)
    grad_norm = jnp.sqrt(jnp.einsum("ni,nij,nj->n", grad, metric, grad) + 1e-12)
    return grad_norm / slow


def nonneg_penalty(correction: jnp.ndarray) -> jnp.ndarray:
    """``relu(-g)^2`` per point: zero where ``g >= 0``, quadratic where the field dips below ``base``.

    Smooth (C^1) at the boundary, and exactly zero with zero gradient on the feasible side, so it
    cannot perturb a run that already satisfies the constraint.
    """
    return jnp.minimum(correction, 0.0) ** 2


def residual(q: jnp.ndarray) -> jnp.ndarray:
    """Per-point squared Eikonal residual ``(q − 1)²``, zero exactly when ``q = 1``.

    Smooth and least-squares structured at the optimum, unlike NTFields' ``|1−√q| + |1−1/√q|`` which
    is non-differentiable at ``q = 1`` — the point every run is trying to reach.
    """
    return (q - 1.0) ** 2


def rrt_anchors(env, cfg) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """RRT* cost-to-come at ``cfg.num_anchors`` nodes, as **equality** targets (lower == upper).

    Correct only when the prior is much more accurate than the field it is correcting: an RRT* cost is
    an upper bound, so pinning it biases the field upward by the planner's suboptimality. Measured on
    the 1-obstacle torus, prior RMS **0.033** against a field of **0.184** with p99 ratio 1.005 — a
    ~0.5% bias, negligible — and that arm is this project's best supervised result (**0.0735**). The
    same recipe on an under-converged tree (9% of nodes >10% high) drove the field to 0.339 instead,
    so measure the prior on the exact scene before choosing this mode.

    **Placement is the mechanism, so the selection rule is part of the recipe.** The anchors work by
    pinning the field's *level* where the pointwise residual leaves it free, which is behind an
    obstacle — hence ``cfg.anchor_shadow_pref``, which up-weights nodes whose source geodesic is
    occluded. Measured, same 30 anchors and same weight: shadow-targeted **0.0735**, uniform 0.1192.
    Setting the preference to 0 *is* that uniform control, not a different code path.

    Two selection defects this replaced, both of which made the historical number unreproducible.
    The draw was uniform — ``rng.choice`` with no preference at all — so the recipe that produced
    0.0735 simply was not in the code. And it ran over ``build_roadmap``'s 300-node uniform
    *subsample* rather than the tree: a uniform subsample preserves the occluded *fraction* (9.1% of
    the full torus tree, 9.0% of the draw) but not the *count*, 272 nodes against 27, so a 30-anchor
    budget came from a pool barely larger than itself and the thinning, not the preference, decided
    where the anchors went. Selection now runs on the full ``converged_tree``.

    Returns:
        ``(points, lower, upper)`` with ``lower == upper``, matching ``roadmap_bounds``' signature so
        the objective needs no branch.
    """
    from srms.environments import sampling

    nodes, costs = sampling.converged_tree(
        env, env.start, cfg.rrt_iters, cfg.rrt_step, cfg.rrt_radius, cfg.seed
    )
    points, values, occluded = sampling.shadow_anchors(
        env, nodes, costs, cfg.num_anchors, getattr(cfg, "anchor_shadow_pref", 0.0), cfg.seed
    )
    print(
        f"[anchors] {len(points)} equality anchors from a {len(nodes)}-node tree — "
        f"{int(occluded.sum())} in shadow (shadow_pref {getattr(cfg, 'anchor_shadow_pref', 0.0)})",
        flush=True,
    )
    values = jnp.asarray(values, jnp.float32)
    return jnp.asarray(points, jnp.float32), values, values


def roadmap_bounds(env, cfg) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sphere-packing roadmap nodes with two-sided travel-time bounds ``(points, lower, upper)``.

    H-NTFields' mechanism (Ni, Liu & Qureshi 2026, Sec. IV-A), which is what actually worked on the
    hyperbolic manifold historically — 1.0112 -> 0.5334, the best ratio of the three manifolds.

    **Bounds, not values.** A graph shortest path is *feasible*, so its cost is an upper bound on the
    true travel time and never an equality. Pinning it as an equality drags the field to the prior's
    own error, which is measured: RRT* equality anchors sent the torus 0.184 -> 0.339 and H²
    0.333 -> 1.347. A hinge pair costs nothing wherever the field already lies inside the corridor.

    **The lower bound is the one that matters here.** The label-free field *under*-predicts — H²'s
    error map is uniformly blue — and an upper bound is exactly zero in that regime. Only the lower
    bound lifts a level that is too low. It is valid to the extent the roadmap is near-optimal, which
    the packing's volume-adaptive placement buys: measured on H², graph cost sits at 1.05x truth
    (p50) and builds in **1 second**, against RRT* still unconverged after **635 s** at similar
    accuracy — uniform sampling is what exponential volume growth defeats, not the idea of a prior.

    Args:
        env: Environment.
        cfg: Config supplying ``hnt_nodes``/``hnt_pool``/``hnt_connect_radius``/``hnt_max_radius``,
            ``bound_slack`` and ``seed``.

    Returns:
        ``(points [k, dim], lower [k], upper [k])``.
    """
    from srms.environments import sampling

    nodes, _, times = sampling.build_sphere_roadmap(
        env, env.start, cfg.hnt_nodes, cfg.hnt_pool, cfg.hnt_connect_radius, cfg.seed, cfg.hnt_max_radius
    )
    finite = np.isfinite(times)
    points, graph_time = np.asarray(nodes)[finite], np.asarray(times)[finite]
    # The graph cost is a genuine upper bound. The lower bound is NOT: it relaxes the same number by
    # `bound_slack`, which is only valid where `graph/truth <= 1/(1 - slack)`. Measured on H², the
    # roadmap sits at 1.208x truth at p90 and 1.513x at p99, so at slack 0.10 roughly 10-25% of the
    # "lower" bounds sit ABOVE truth and push the field too high in exactly the detour regions where
    # the roadmap is worst. This repo has seen that failure before — the hntfields docstring records
    # 94% invalid lower bounds costing 6-8x against PDE-only. Hence `anchor_weight` defaults to the
    # historical 1e-2 rather than a hot weight: a wrong bound at low weight nudges, at high weight it
    # dictates. Raising slack widens the corridor and restores validity at the cost of looseness.
    lower = np.maximum(graph_time * (1.0 - cfg.bound_slack), 0.0)
    return (
        jnp.asarray(points, jnp.float32),
        jnp.asarray(lower, jnp.float32),
        jnp.asarray(graph_time, jnp.float32),
    )


def variational_objective(backend, params, env, cfg, colloc, slow):
    """Maximise ``T`` subject to ``‖∇T‖_g <= s`` — the value function is the largest subsolution.

    ``minimise  −mean(T) + lambda · mean(relu(q − 1)²)``,  ``q = ‖∇T‖_g / s``.

    Any ``u`` with ``‖∇u‖_g <= s`` and ``u(start) = 0`` grows at most at rate ``s`` along any path, so
    ``u(x) <= ∫ s dl`` for every path and therefore ``u(x) <= T(x)``. The true field is the **largest**
    such function, so the answer is picked out by pushing ``T`` up against a one-sided constraint
    rather than by driving a two-sided residual to zero.

    This targets the failure the two-sided residual cannot see. Under ``(q − 1)²`` a shadow whose
    level is too low is barely penalised — its gradient is correct and only the additive constant is
    wrong. Here it is penalised directly, and the detour cost propagates inward because the
    maximisation couples every point through the constraint.

    Args:
        backend: Backend module.
        params: Backend parameters.
        env: Environment.
        cfg: Config, read for ``variational_lambda``.
        colloc: Collocation points, [n, dim].
        slow: Slowness at those points, [n].

    Returns:
        ``(total, {"ascent": ..., "violation": ...})``.
    """
    q = speed_ratio(backend, params, colloc, slow, env, cfg)
    violation = jnp.mean(jnp.maximum(q - 1.0, 0.0) ** 2)
    ascent = jnp.mean(predict(backend, params, colloc, env, cfg))
    return cfg.variational_lambda * violation - ascent, {"ascent": ascent, "violation": violation}


def objective(backend, params, env, cfg, colloc, slow, src_pts=None, src_vals=None, rate=0.0, anchors=None):
    """Training objective: the Eikonal residual alone. No boundary term is needed.

    ``T(start) = 0`` is exact by construction here, so unlike ``eikonal.py`` there is nothing for a
    boundary ring to pin. Module-level so it can be evaluated at parameters the optimizer never
    produced (see ``srms/experiments/gate.py``).

    Args:
        backend: Backend module.
        params: Backend parameters.
        env: Environment.
        cfg: Config, read for ``causal``.
        colloc: Collocation points, [n, dim].
        slow: Slowness at those points, [n].
        src_pts: Unused; accepted so the signature matches ``eikonal.objective``.
        src_vals: Unused; same reason.
        rate: Causal decay rate; 0 disables the weighting.

    Returns:
        ``(total, {"eikonal": ...})``.
    """
    if getattr(cfg, "variational", False):
        total, aux = variational_objective(backend, params, env, cfg, colloc, slow)
        weight = getattr(cfg, "nonneg_weight", 0.0)
        if weight > 0.0:
            total = total + weight * jnp.mean(nonneg_penalty(backend.eval_raw(params, colloc, env, cfg).ravel()))
        return total, aux
    squared = residual(speed_ratio(backend, params, colloc, slow, env, cfg))
    if cfg.causal:
        # Order by the model's own predicted arrival time, not the free-space geodesic: behind an
        # obstacle those disagree, and the free-space key un-mutes shadow points too early.
        key = jax.lax.stop_gradient(predict(backend, params, colloc, env, cfg))
        eik = training_aids.causal_loss(env, colloc, squared, rate, order_by=key)
    else:
        eik = jnp.mean(squared)
    total, aux = eik, {"eikonal": eik}
    weight = getattr(cfg, "nonneg_weight", 0.0)
    if weight > 0.0:
        nonneg = jnp.mean(nonneg_penalty(backend.eval_raw(params, colloc, env, cfg).ravel()))
        total, aux["nonneg"] = total + weight * nonneg, nonneg
    if anchors is not None and cfg.anchor_weight > 0.0:
        points, lower, upper = anchors  # equality mode sets lower == upper == the RRT* cost
        predicted = predict(backend, params, points, env, cfg)
        # Two-sided hinge: zero inside the corridor, so a loose bound costs nothing.
        bound_loss = jnp.mean(
            jnp.maximum(predicted - upper, 0.0) ** 2 + jnp.maximum(lower - predicted, 0.0) ** 2
        )
        total, aux["bounds"] = total + cfg.anchor_weight * bound_loss, bound_loss
    l1 = getattr(cfg, "l1_weight", 0.0)
    if l1 > 0.0:
        sparsity = backend.weight_l1(params)
        total, aux["l1"] = total + l1 * sparsity, sparsity
    return total, aux


def solve(env, cfg, backend, checkpoint=None, progress_fn=None):
    """Fit ``g`` so that ``T = base · exp(g)`` satisfies the Eikonal equation.

    Args:
        env: Environment.
        cfg: Config.
        backend: Backend module (``srm`` or ``mlp``).
        checkpoint: Optional ``(params, step)`` callback.
        progress_fn: Optional ``(step, metrics)`` callback, every ``cfg.log_every`` steps.

    Returns:
        Trained backend parameters.
    """
    params = backend.init_params(jax.random.PRNGKey(cfg.seed), env, cfg)
    rng = np.random.default_rng(cfg.seed)
    anchors = None
    if getattr(cfg, "num_anchors", 0) > 0:
        # Two mechanisms, chosen explicitly. They were once selected by the same flag, and attributing
        # a result to the wrong one cost this project a day; the banner below names which ran.
        anchors = rrt_anchors(env, cfg) if cfg.anchor_mode == "equality" else roadmap_bounds(env, cfg)
        print(
            f"[factored] {len(anchors[0])} {cfg.anchor_mode} "
            f"{'anchors (RRT*)' if cfg.anchor_mode == 'equality' else 'bounds (sphere-pack)'}, "
            f"weight {cfg.anchor_weight}",
            flush=True,
        )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(cfg.lr))
    opt_state = optimizer.init(params)

    @jax.jit
    def step(p, state, colloc, slow, rate):
        (loss, aux), grads = jax.value_and_grad(objective, argnums=1, has_aux=True)(
            backend, p, env, cfg, colloc, slow, None, None, rate, anchors
        )
        updates, state = optimizer.update(grads, state, p)
        return backend.post_step(optax.apply_updates(p, updates), cfg, env), state, loss, aux

    def spawn_residual(points):
        """Where to add capacity: the squared Eikonal residual at each candidate centre."""
        return residual(speed_ratio(backend, params, points, env.slowness(points), env, cfg))

    densifier = training_aids.DensifyController(cfg)
    progress = trange(cfg.steps, desc="factored")
    for i in progress:
        colloc = sample_collocation(env, rng, cfg.num_collocation, cfg.source_radius)
        slow = env.slowness(colloc)
        rate = jnp.float32(training_aids.causal_rate(cfg, i)) if cfg.causal else jnp.float32(0.0)
        params, opt_state, loss, aux = step(params, opt_state, colloc, slow, rate)
        densifier.record(loss)
        if densifier.should_densify(i, backend.num_units(params)):
            params, opt_state, k = backend.adapt(params, opt_state, spawn_residual, env, cfg, rng)
            if k is not None:
                progress.write(f"[factored] step {i}: densify → {k} splats")
        if i % cfg.log_every == 0:
            progress.set_description(f"factored — log10(loss) = {float(jnp.log10(loss + 1e-12)):.3f}")
            if progress_fn is not None:
                progress_fn(i, {"loss": float(loss), "num_params": backend.num_params(params)})
        if checkpoint is not None and i > 0 and i % cfg.checkpoint_every == 0:
            checkpoint(params, i)
    print(f"[factored] {densifier.summary(backend.num_units(params))}", flush=True)
    return params
