"""SRM backend: a weighted mixture of wrapped Gaussians ("splats") on any Environment.

Delegates the manifold-specific pieces (log map, Jacobian correction) to the
Environment. The wrapped-Gaussian math is written out here rather than called
from ``srms.lib.manifold_splat.eval_wrapped_gaussian``, because the training
loop needs it batched over splats and points with the per-splat frame hoisted
out (``env.splat_precompute``) and the log map fused with its Jacobian
(``env.log_and_jac``). ``environments/test_manifolds.py`` checks the two
implementations agree pointwise (measured max relative difference 3e-11 across
all five manifolds), so the duplication is verified rather than assumed.

Parameters ``(V, A, B)``: ``V: [k, p]`` weights, ``A: [k, d, d]`` scale/rotation,
``B: [k, d]`` centres — same convention as ``srms/lib/splat.py``.

Every backend (this one, and eventually ``mlp``/``kan``) exposes the same three
entry points so a training strategy (``srms/methods/strategies``) never needs
to know which one it's using:

- ``init_params(key, env, cfg) -> params`` — reads whatever ``cfg`` fields it
  needs (here: ``cfg.num_splats``, ``cfg.init_scale``); other backends read
  their own fields off the same flat ``cfg``.
- ``eval_raw(params, X, env) -> [n, p]`` — the raw (unfactored) field value.
- ``post_step(params, cfg, env) -> params`` — called by every strategy's
  training loop after each optimizer step (see ``post_step`` below for why).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

SplatParams = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


def init_params(key: jax.Array, env, cfg, p: int = 1) -> SplatParams:
    """Initialise (V, A, B): centres drawn from env.sample_domain, isotropic scales, weights per cfg.

    Centres need environment-specific sampling (a box for the torus, a unit sphere for
    SphereEnvironment), not a generic uniform box, so this delegates to the same host-side sampler
    RRT*/collocation already use rather than assuming ``env.domain`` is a box.

    ``cfg.init_weight`` decides the weights. At 0 (the default) the mixture starts identically zero,
    which is what the factored strategies want: ``T = base/tau`` begins at the free-space geodesic,
    the do-nothing field. An **unfactored** field must not start there — ``T = 0`` gives ``grad T = 0``
    and ``d‖grad T‖/dparams = 0/0``, making it an exact stationary point of the Eikonal residual
    (measured: PDE ``|grad|`` exactly 0.0 at ``V=0``, 9.8e-1 at ``V ~ N(0, 1e-3)``). ``eikonal.solve``
    checks for that and refuses to train from it.

    Args:
        key: PRNG key seeding the centre sampler.
        env: Environment supplying ``sample_domain`` and ``tangent_dim``.
        cfg: Config supplying ``densify``/``init_splats``/``num_splats``/``init_scale``/``init_weight``.
        p: Output width; 1 for a scalar field.

    Returns:
        ``(V, A, B)`` with shapes ``[k, p]``, ``[k, tangent_dim, tangent_dim]``, ``[k, dim]``.
    """
    k = cfg.init_splats if getattr(cfg, "densify", False) else cfg.num_splats
    # `key` itself still seeds the centres, and the weight key is folded off it, so at init_weight=0
    # this is bit-identical to the zero-weight version every published number was produced with.
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    centres = jnp.asarray(env.sample_domain(np.random.default_rng(seed), k), dtype=jnp.float32)
    scales = jnp.repeat((cfg.init_scale * jnp.eye(env.tangent_dim))[None], k, axis=0)
    weight_scale = getattr(cfg, "init_weight", 0.0)
    weights = jnp.zeros((k, p))
    if weight_scale > 0.0:
        weights = weight_scale * jax.random.normal(jax.random.fold_in(key, 1), (k, p))
        # A signed init would put half the splats at exactly 0 on the first step under the V >= 0
        # clamp, so the structural arm would quietly start at half capacity.
        weights = jnp.abs(weights) if getattr(cfg, "nonneg_weights", False) else weights
    return weights, scales, centres


def compact_window(mahalanobis_sq: jnp.ndarray, trunc_sigma: float, taper: float = 0.25) -> jnp.ndarray:
    """Smooth window taking a splat to exactly zero beyond ``trunc_sigma`` standard deviations.

    A Gaussian never reaches zero, so under a non-negative mixture every active splat adds a small
    positive amount over the *whole* manifold. That is fine when weights may cancel and fatal when
    they may not: measured on the one-obstacle torus, the ``V >= 0`` mixture predicted a correction of
    +0.105 in far free space where the truth is +0.018, purely from tails, and an L1 prior on the
    weights barely touched it (0.2259 -> 0.2193). Giving each splat compact support removes the leak
    at its source, and makes a splat a genuinely *local* object: editing one changes the field only
    within its own radius, which is the interpretability claim this project rests on.

    At 3 sigma a Gaussian is ~1.1% of its peak, so the mass discarded is ~1.1% in 2-D. The window is
    a quintic smoothstep over the last ``taper`` fraction of the radius, so the field stays C^2 and
    the Eikonal residual sees no kink.

    Args:
        mahalanobis_sq: ``z^T z`` where ``z = A^-1 · log_map(mu, x)``, i.e. squared radius in sigmas.
        trunc_sigma: Cutoff radius in standard deviations; the window is 0 beyond it.
        taper: Fraction of the radius over which the window falls from 1 to 0.

    Returns:
        Window value in [0, 1], same shape as the input.
    """
    radius = jnp.sqrt(jnp.maximum(mahalanobis_sq, 1e-12))
    start = trunc_sigma * (1.0 - taper)
    t = jnp.clip((radius - start) / (trunc_sigma - start), 0.0, 1.0)
    return 1.0 - (t * t * t * (t * (t * 6.0 - 15.0) + 10.0))


def eval_raw(params: SplatParams, X: jnp.ndarray, env, cfg=None) -> jnp.ndarray:
    """Evaluate the raw splat mixture g(x) = Σ_j V[j]·N_w(x | B[j], A[j]) at each row of X. Returns [n, p].

    Centres are read through ``env.wrap_point`` so the density is always evaluated at a centre that
    lies *on* the manifold — a wrapped Gaussian is undefined otherwise. This makes the loss exactly
    invariant to ‖B‖, which in turn makes ``dL/dB`` exactly tangent (measured radial component
    4e-10), so the optimizer only ever moves a centre along the manifold.

    That invariance is necessary but **not sufficient**: with nothing pinning ‖B‖, AdamW's weight
    decay shrinks it, and the angular step ≈‖ΔB‖/‖B‖ then grows without bound. ``post_step`` performs
    the actual retraction on the parameters; see its docstring for the measurements.

    A mathematical no-op on the chart manifolds: ``wrap(x − wrap(mu)) ≡ wrap(x − mu)`` on the torus,
    and hyperbolic's ``_clamp_ball`` is the identity for interior points.
    """
    V, A, B = params
    B = env.wrap_point(B)
    # Per-splat work hoisted out of the per-point loop (see docstring).
    A_inv = jnp.linalg.inv(A)
    det_A = jnp.abs(jnp.linalg.det(A))
    pre = jax.vmap(env.splat_precompute)(B)
    norm_const = (2.0 * jnp.pi) ** (env.tangent_dim / 2.0)

    trunc = float(getattr(cfg, "trunc_sigma", 0.0)) if cfg is not None else 0.0

    def rho_at_x(x: jnp.ndarray) -> jnp.ndarray:
        def one(pre_j, a_inv, det):
            # One call for both: jac_factor used to recompute the distance log_map already had.
            v, jac = env.log_and_jac(pre_j, x)
            z = a_inv @ v
            density = jnp.exp(-0.5 * jnp.dot(z, z)) / (norm_const * (det + 1e-12)) * jac
            return density * compact_window(jnp.dot(z, z), trunc) if trunc > 0.0 else density

        return jax.vmap(one)(pre, A_inv, det_A)

    return jax.vmap(rho_at_x)(X) @ V


def post_step(params: SplatParams, cfg, env=None) -> SplatParams:
    """Floor each splat's covariance singular values, and retract the centres onto the manifold.

    **Retraction (``env`` given).** A wrapped Gaussian is only defined for a centre lying *on* the
    manifold, but ``B`` is an unconstrained optimizer variable. This matters only for **embedded**
    manifolds: on S² ``B`` must satisfy ‖B‖ = 1 and a plain ambient gradient step walks it off —
    measured 0.105 off the unit sphere by step 65, at which point ``_sphere_frame``'s Householder
    construction (which assumes a unit vector) is no longer orthonormal and the density goes to NaN.
    ``env.wrap_point`` is the projection retraction; it agrees with the exponential retraction
    ``Exp_B`` to O(‖step‖³) (measured 3.3e-4 at step 0.1, 4.3e-7 at 0.01), which is far below float32
    noise at this learning rate.

    Doing it here, on the *parameters*, rather than inside ``eval_raw`` matters. Normalizing at
    evaluation time makes the loss invariant to ‖B‖, so nothing opposes AdamW's weight decay: ‖B‖ then
    decays as (1−lr·wd)^step (measured 1.00 → 0.33 over 2400 steps) and, since the angular step on a
    centre is ≈‖ΔB‖/‖B‖, the effective step size *grows* as the norm shrinks — 3× by step 2400 — and
    training diverged at step 2481. Retracting the parameters pins ‖B‖ = 1 and removes that coupling.

    A no-op on the chart manifolds: ``wrap`` is idempotent on the torus and hyperbolic's
    ``_clamp_ball`` is the identity for interior points, so no established result moves.

    **Non-negative weights (``cfg.nonneg_weights``).** With ``T = base·exp(g)`` and slowness ``>= 1``,
    travel time can never beat the free-space geodesic, so ``g >= 0`` is a physical law. A splat
    mixture can satisfy it *structurally*: every wrapped Gaussian is a density, so clamping ``V >= 0``
    makes ``g = sum_j V_j rho_j >= 0`` everywhere, with no penalty term and no hyperparameter. Splats
    far from any obstacle simply hold ``V ~ 0`` and contribute nothing, which is also the sparsity the
    problem wants — the correction is confined to the obstacle and its shadow. This is a capability of
    the mixture that a general function approximator does not have: an MLP's output sign cannot be
    constrained by projecting its weights.

    **Scale floor — this is the fix that makes adaptive densification stable.** A sum of Gaussians
    cannot represent the true Eikonal kink at obstacle boundaries and the cut locus, so unconstrained
    optimization chases it by driving a splat's covariance to zero — an effectively infinite ``‖∇T‖``
    spike (up to ~1e7 was measured) which then blows up any speed-match residual. Flooring the SVD
    singular values at ``cfg.scale_floor`` makes that collapse impossible. Applied after every
    optimizer step; jittable, so it lives inside the strategies' ``step``.

    **Directional stretching (``cfg.max_aspect``).** A uniform floor on every singular value also
    forbids the *useful* kind of anisotropy: at an obstacle's shadow crease the field is smooth along
    the crease and kinked across it, so a splat wants to be long in one direction and thin in the
    other. Measured on the 2-D torus after 800 steps: with no obstacles splats stay isotropic (median
    aspect 1.09, max 1.58) and the floor never binds; with one obstacle they stretch to 8.3:1 and
    1.2% of them sit pinned at the floor. ``max_aspect`` replaces the per-axis floor with a bound on
    the *condition number* — the effective lower bound becomes ``max(scale_floor, s_max/max_aspect)``
    — so a splat may thin across a crease while ``scale_floor`` still prevents it collapsing to a
    point in every direction at once, which is the failure the floor was introduced for.

    The floor is a no-op when ``cfg.scale_floor <= 0``, and irrelevant to the fixed-count model
    (which never densifies and so never triggers the collapse), but harmless there. ``max_aspect``
    defaults to 0 (disabled), which reproduces the plain uniform floor exactly.
    """
    V, A, B = params
    if getattr(cfg, "nonneg_weights", False):
        # A wrapped Gaussian is a density, so it is >= 0 everywhere; clamping the weights therefore
        # makes the whole mixture g = sum_j V_j rho_j non-negative *by construction*, which is the
        # physical constraint T >= base rather than a penalty approximating it. Projection onto the
        # feasible set after each step, the same treatment A and B already get here.
        V = jnp.maximum(V, 0.0)
    if env is not None:
        B = env.wrap_point(B)
    floor = getattr(cfg, "scale_floor", 0.0)
    aspect = getattr(cfg, "max_aspect", 0.0)
    if floor > 0.0 or aspect > 0.0:
        U, S, Vt = jnp.linalg.svd(A)
        lower = jnp.full_like(S, floor)
        if aspect > 0.0:
            # Bound the condition number instead of every axis: a splat may go thin across a crease
            # so long as it stays within `aspect` of its own longest axis. `floor` remains the
            # absolute backstop against collapsing to a point in every direction at once.
            lower = jnp.maximum(lower, jnp.max(S, axis=-1, keepdims=True) / aspect)
        A = jnp.einsum("kij,kj,kjl->kil", U, jnp.maximum(S, lower), Vt)
    return V, A, B


def adapt(params: SplatParams, opt_state, residual_fn, env, cfg, rng: np.random.Generator):
    """3DGS-style prune + spawn, preserving Adam moments for the splats that survive.

    Prunes splats whose weight is below ``cfg.prune_thresh``, then spawns up to ``cfg.spawn_per`` new
    ones at the highest-``residual_fn`` free-space points, capped at ``cfg.max_splats``.

    The optimizer state is grown **surgically**: every leaf whose leading axis indexes splats is
    sliced to the survivors and zero-padded for the spawns. A plain ``optimizer.init()`` reset would
    re-kick every already-converged splat, which is what made earlier densification runs diverge.
    Scalar leaves (Adam's step count) pass through untouched, so this is robust to ``optax.chain``
    nesting without depending on the state's namedtuple layout.

    Returns ``(params, opt_state, num_splats)``.
    """
    V = np.asarray(params[0])
    old_k = len(V)
    keep = np.abs(V[:, 0]) > cfg.prune_thresh
    if keep.sum() < 16:  # never prune the model into nothing
        keep = np.ones(old_k, bool)
    idx = np.where(keep)[0]

    spawn_B = np.zeros((0, env.dim), np.float32)
    room = cfg.max_splats - len(idx)
    if room > 0:
        pool = env.sample_domain(rng, 3000)
        pool = pool[env.sdf_np(pool) > 0.1]  # spawn in free space, not inside obstacles
        if len(pool):
            residual = np.asarray(residual_fn(jnp.asarray(pool, jnp.float32)))
            k = int(min(cfg.spawn_per, room, len(pool)))
            spawn_B = pool[np.argsort(residual)[-k:]].astype(np.float32)
    ns = len(spawn_B)

    def regrow(leaf):
        a = np.asarray(leaf)
        if a.ndim >= 1 and a.shape[0] == old_k:
            return jnp.asarray(np.concatenate([a[idx], np.zeros((ns,) + a.shape[1:], a.dtype)], 0))
        return leaf

    new_v = np.concatenate([V[idx], np.full((ns, V.shape[1]), 1e-4, np.float32)])
    new_a = np.concatenate(
        [
            np.asarray(params[1])[idx],
            np.repeat((cfg.spawn_scale * np.eye(env.tangent_dim))[None], ns, 0).astype(np.float32),
        ]
    )
    new_b = np.concatenate([np.asarray(params[2])[idx], spawn_B])
    new_params = (jnp.asarray(new_v), jnp.asarray(new_a), jnp.asarray(new_b))
    return new_params, jax.tree_util.tree_map(regrow, opt_state), len(new_v)


def weight_l1(params: SplatParams) -> jnp.ndarray:
    """Mean |V| over the mixture — an L1 prior on how many splats are active.

    Under ``nonneg_weights`` this is LASSO with a projection: the true correction is zero over most
    of the manifold, so the prior is that most splats should be switched off entirely rather than
    contributing small positive tails everywhere.
    """
    return jnp.mean(jnp.abs(params[0]))


def num_units(params: SplatParams) -> int:
    """Number of splats in the mixture — the unit adaptive densification adds and prunes."""
    return int(np.shape(params[0])[0])


def num_params(params: SplatParams) -> int:
    """Total trainable scalars — k·(p + d² + d). Reported so SRM/MLP can be compared at matched size."""
    return int(sum(np.prod(np.shape(x)) for x in jax.tree_util.tree_leaves(params)))
