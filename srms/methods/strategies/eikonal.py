"""Eikonal training strategy: plain physics-informed fit — boundary loss + PDE residual loss.

``T(θ)`` is a free (unfactored) field, evaluated through whichever backend is
passed in (``srms/methods/backends/{srm,mlp,kan}``) — this strategy only calls
``backend.init_params(key, env, cfg)`` / ``backend.eval_raw(params, X, env)``,
so the same objective and training loop work for any of them. The point-source
singularity is handled by fitting ``T ≈ eps·slowness(start)`` on a small sphere
of radius ``eps`` around the source (the local, locally-flat travel time); the
Eikonal PDE ``‖∇T‖ = 1/s`` is enforced on random collocation points excluding
that same ball. This mirrors ``_archive/preexisting/eikonal_splat.py``'s
``eikonal_loss``/``train_eikonal_splat`` (and the dynamic per-step resampling
of ``eikonal_nd_dynamic.py``) as closely as possible: no adaptive sampling,
obstacle-shadow anchors, or factored base — just BC + PDE loss, plus optional
causal weighting on the PDE term (``cfg.causal``, see ``training_aids.py``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import trange

from srms.methods.strategies import training_aids


def predict(backend, params, thetas: jnp.ndarray, env) -> jnp.ndarray:
    """Evaluate the raw field T over a batch of points, returned as [N]."""
    return backend.eval_raw(params, thetas, env)[:, 0]


def source_sphere(env, eps: float, n_pts: int, seed: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``n_pts`` points at geodesic distance ``eps`` from the source, with the known local travel time."""
    rng = np.random.default_rng(seed)
    points = env.boundary_ring_np(rng, eps, n_pts)
    value = eps * float(env.slowness_np(np.asarray(env.start)[None, :])[0])
    return jnp.asarray(points, dtype=jnp.float32), jnp.full(n_pts, value, dtype=jnp.float32)


def sample_collocation(env, rng: np.random.Generator, n: int, exclude_radius: float) -> jnp.ndarray:
    """Uniform random collocation over the domain, excluding a ball around the source."""
    pool = env.sample_domain(rng, n * 10)
    dist = np.linalg.norm(env.displacement_np(np.asarray(env.start), pool), axis=-1)
    return jnp.asarray(pool[dist > exclude_radius][:n], dtype=jnp.float32)


def pde_residual(backend, params, thetas: jnp.ndarray, slow: jnp.ndarray, env) -> jnp.ndarray:
    """Eikonal residual ``‖∇T‖_g − slowness`` at each point.

    ``env.slowness`` is a cost per unit length (>= 1, rising inside obstacles), so ``T = int s dl``
    and ``‖∇T‖ = s`` — not ``1/s``. Verified by finite-differencing the fast-marching field on the
    2-D torus: median ``‖∇T‖/s`` is 0.999 inside obstacles and 1.004 in free space.

    Args:
        backend: Backend module exposing ``eval_raw``.
        params: Backend parameters.
        thetas: Points, [n, dim].
        slow: Slowness at each point, [n].
        env: Environment supplying ``metric_inv``.

    Returns:
        Per-point residual, [n]. ``‖·‖_g`` uses ``env.metric_inv`` (identity on a flat chart, a
        tangent-plane projector on an embedded manifold).
    """

    def u_single(x):
        return backend.eval_raw(params, x[None, :], env)[0, 0]

    grad = jax.vmap(jax.grad(u_single))(thetas)
    metric = jax.vmap(env.metric_inv)(thetas)
    grad_sq = jnp.einsum("ni,nij,nj->n", grad, metric, grad)
    grad_norm = jnp.sqrt(grad_sq + 1e-12)
    return grad_norm - slow


def objective(backend, params, env, cfg, colloc, slow, src_pts, src_vals, rate=0.0):
    """Training objective: source boundary condition + Eikonal residual.

    Module-level rather than a closure inside ``solve`` so it can be evaluated at parameters the
    optimizer never produced — above all at a field fitted to the *true* answer. Comparing the two
    settles whether a bad result is an optimizer failure or an objective that prefers a wrong field,
    which no amount of tuning can fix. See ``srms/experiments/gate.py``.

    Args:
        backend: Backend module.
        params: Backend parameters.
        env: Environment.
        cfg: Config, for ``physics_weight`` and ``causal``.
        colloc: Collocation points, [n, dim].
        slow: Slowness at ``colloc``, [n].
        src_pts: Boundary-ring points, [m, dim].
        src_vals: Known travel time on the ring, [m].
        rate: Causal decay rate; 0 disables the weighting.

    Returns:
        ``(total, {"bc": ..., "pde": ...})``.
    """
    bc = jnp.mean((predict(backend, params, src_pts, env) - src_vals) ** 2)
    squared = pde_residual(backend, params, colloc, slow, env) ** 2
    pde = training_aids.causal_loss(env, colloc, squared, rate) if cfg.causal else jnp.mean(squared)
    return bc + cfg.physics_weight * pde, {"bc": bc, "pde": pde}


def assert_live_init(backend, params, env, cfg, rng) -> float:
    """Check the PDE term has a non-zero parameter gradient at initialisation.

    An unfactored field starting at ``T = 0`` sits on an exact stationary point of the residual:
    ``grad T = 0`` makes ``d‖grad T‖/dparams`` a regularised 0/0, so only the boundary ring has any
    gradient and the model learns a bump at the source and nothing else. That is not slow training,
    it is no training, and it is silent — hence a hard check rather than a comment.

    Args:
        backend: Backend module.
        params: Initial parameters.
        env: Environment.
        cfg: Config, for ``source_radius``.
        rng: Host RNG for the probe collocation batch.

    Returns:
        The measured gradient norm of the PDE term at initialisation.

    Raises:
        ValueError: If that norm is zero, naming the knob that fixes it.
    """
    probe = sample_collocation(env, rng, 256, cfg.source_radius)
    grads = jax.grad(lambda p: jnp.mean(pde_residual(backend, p, probe, env.slowness(probe), env) ** 2))(params)
    norm = float(jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads))))
    if norm == 0.0:
        raise ValueError(
            "eikonal: the PDE term has exactly zero gradient at initialisation, so training cannot "
            "start. The unfactored field is identically zero — pass --init-weight 1e-2 (srm backend) "
            "to break the degeneracy."
        )
    return norm


def solve(env, cfg, backend, checkpoint=None, progress_fn=None):
    """Fit a free field to the Eikonal PDE with a small-sphere boundary condition at the source.

    Args:
        env: Environment supplying geometry, slowness and sampling.
        cfg: Config.
        backend: Backend module (``srm`` or ``mlp``).
        checkpoint: Optional ``(params, step)`` callback for mid-training snapshots.
        progress_fn: Optional ``(step, metrics)`` callback, called every ``cfg.log_every`` steps
            with the scalar training metrics ``loss``/``bc``/``pde`` (see ``run.py``).

    Returns:
        The trained backend parameters.
    """
    params = backend.init_params(jax.random.PRNGKey(cfg.seed), env, cfg)
    rng = np.random.default_rng(cfg.seed)
    assert_live_init(backend, params, env, cfg, rng)
    src_pts, src_vals = source_sphere(env, cfg.source_radius, cfg.n_sphere, cfg.seed)

    # A stray large step can still blow the gradient up and NaN the run (same "srm backend" risk
    # ntfields.py/weak_supervision.py already clip against — see their docstrings); this strategy
    # was the one holdout without it.
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(cfg.lr))
    opt_state = optimizer.init(params)

    def loss_fn(p, colloc, slow, rate):
        return objective(backend, p, env, cfg, colloc, slow, src_pts, src_vals, rate)

    @jax.jit
    def step(p, state, colloc, slow, rate):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, colloc, slow, rate)
        updates, state = optimizer.update(grads, state, p)
        p = backend.post_step(optax.apply_updates(p, updates), cfg, env)
        return p, state, loss, aux

    def spawn_residual(pts):
        """Where to put new capacity: the squared Eikonal residual at each candidate centre."""
        return pde_residual(backend, params, pts, env.slowness(pts), env) ** 2

    # The model, not the schedule, decides how many splats it needs (see training_aids).
    densifier = training_aids.DensifyController(cfg)
    progress = trange(cfg.steps, desc="eikonal")
    for i in progress:
        colloc = sample_collocation(env, rng, cfg.num_collocation, cfg.source_radius)
        slow = env.slowness(colloc)
        rate = jnp.float32(training_aids.causal_rate(cfg, i)) if cfg.causal else jnp.float32(0.0)
        params, opt_state, loss, aux = step(params, opt_state, colloc, slow, rate)
        densifier.record(loss)
        if densifier.should_densify(i, backend.num_units(params)):
            params, opt_state, k = backend.adapt(params, opt_state, spawn_residual, env, cfg, rng)
            if k is not None:
                progress.write(f"[eikonal] step {i}: densify → {k} splats ({backend.num_params(params)} params)")
        if i % cfg.log_every == 0:
            progress.set_description(f"eikonal — log10(loss) = {float(jnp.log10(loss + 1e-12)):.3f}")
            if progress_fn is not None:
                progress_fn(
                    i,
                    {
                        "loss": float(loss),
                        "bc": float(aux["bc"]),
                        "pde": float(aux["pde"]),
                        "num_params": backend.num_params(params),
                    },
                )
        if checkpoint is not None and i > 0 and i % cfg.checkpoint_every == 0:
            checkpoint(params, i)
    print(f"[eikonal] {densifier.summary(backend.num_units(params))}", flush=True)
    return params
