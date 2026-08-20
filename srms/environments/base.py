"""Environment interface shared by every Eikonal training strategy.

A concrete Environment (e.g. ``TorusEnvironment``) owns the manifold geometry,
the obstacle/slowness field, and ground truth, so that ``srms/methods``
(backends and strategies) never import a specific manifold's functions by
name. Methods here mirror the ``(log_map_fn, jac_factor_fn, dim)`` interface
already used by ``srms/lib/manifold_splat.py``'s ``eval_wrapped_gaussian``, extended
with the pieces the Eikonal residuals, adaptive sampling, and RRT* prior need.

This is a duck-typed ``Protocol``, not an ABC — matching the un-opinionated
style already used by ``ground_truth.py``'s ``PlanningProblem``. Concrete
environments (e.g. ``TorusEnvironment``) satisfy it structurally; there is no
need to inherit from it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np


# sdf value used where a scene has no obstacles: sigmoid(-1e3/slow_width) underflows to 0 in
# float32, so slowness collapses to exactly 1 and the obstacle-free scene is a valid configuration.
FREE_SPACE_SDF = 1e3


def union_sdf(per: list, num_points: int, xp=jnp):
    """Signed distance to an obstacle union.

    Args:
        per: Per-obstacle signed distances, each [n]. May be empty.
        num_points: Number of query points, used to shape the obstacle-free result.
        xp: Array module, ``jnp`` (default) or ``np``.

    Returns:
        Elementwise minimum over ``per``, [n]; ``FREE_SPACE_SDF`` everywhere when ``per`` is empty.
    """
    if not per:
        return xp.full((num_points,), FREE_SPACE_SDF)
    return xp.min(xp.stack(per, axis=0), axis=0)


def smooth_slowness(sdf, slowness_max: float, slow_width: float, xp=jnp):
    """Cost per unit length: ~1 in free space, rising to ``slowness_max`` inside obstacles.

    Args:
        sdf: Signed distance to the obstacle union, [n]; negative inside.
        slowness_max: Slowness deep inside an obstacle.
        slow_width: Width of the sigmoid ramp across the obstacle boundary.
        xp: Array module, ``jnp`` (default) or ``np``.

    Returns:
        Slowness at each point, [n]. The exponent is clipped so ``FREE_SPACE_SDF`` does not
        overflow NumPy's ``exp``; the clipped branch is saturated either way.
    """
    return 1.0 + (slowness_max - 1.0) / (1.0 + xp.exp(xp.clip(sdf / slow_width, -60.0, 60.0)))


def unit_geodesic_gradient(env, x: jnp.ndarray) -> jnp.ndarray:
    """``∇base`` at ``x``: the metric-unit covector pointing away from the source.

    ``base(x)`` is the geodesic distance from ``env.start``, so its gradient is the unit tangent at
    ``x`` along the geodesic, pointing away from the source — note **at x**, not at the source. The
    direction is ``-log_map_ambient(x, start)``; dividing by its own metric norm makes
    ``grad @ metric_inv @ grad == 1`` exactly, which is the identity the Eikonal residual is built on.

    This exists so that ``base`` is never differentiated. On the sphere ``base = 2·arcsin(‖x−start‖/2)``
    has an unbounded derivative as the chord approaches 2, so autodiff evaluates ``0 × inf`` at the
    antipode and returns NaN: measured ``|∇base|`` of 1.1 at distance 1.0, 1.3e3 at 3.14, NaN at
    3.1415. A uniformly sampled collocation batch reaches there routinely and training NaN'd within
    60 steps. The closed form is bounded everywhere.

    Args:
        env: Environment supplying ``log_map_ambient``, ``metric_inv`` and ``start``.
        x: One point on the manifold, [dim].

    Returns:
        Gradient covector at ``x``, [dim], with unit norm under ``metric_inv``.
    """
    direction = -env.log_map_ambient(x, jnp.asarray(env.start, dtype=x.dtype))
    metric = env.metric_inv(x)
    return direction / jnp.sqrt(jnp.einsum("i,ij,j->", direction, metric, direction) + 1e-30)


@runtime_checkable
class Environment(Protocol):
    dim: int  # point-representation dimension (= ambient dimension; equals tangent_dim for flat charts)
    tangent_dim: int  # dimension of the tangent space / splat scale matrix A (= dim for flat charts)
    domain: tuple[float, float]  # (low, high) for uniform sampling over the manifold's chart
    obstacles: tuple
    axis_labels: tuple[str, str]
    render_extent: tuple[float, float, float, float]
    title: str

    # ---- manifold geometry (feeds methods/backends/srm.py's eval_wrapped_gaussian) --------------

    def log_map(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Intrinsic tangent-frame coordinates of x at base mu, size tangent_dim (= psi^-1(x), psi = Exp_mu).

        Used only for splat evaluation (matches the A scale matrix's shape); for a flat chart this
        coincides with ``log_map_ambient``, for an embedded manifold (e.g. the sphere) it does not.
        """
        ...

    def log_map_ambient(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Ambient tangent vector (size dim) at mu pointing toward x; ‖·‖ = geodesic distance.

        Used where the tangent vector needs to stay addable to an ambient point (e.g. the RRT*-roadmap
        last-hop interpolation in weak_supervision.py). Identical to log_map for flat charts.
        """
        ...

    def exp_map(self, mu: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Exp_mu(v): the inverse of ``log_map``, taking tangent-frame coordinates back to a point.

        Not used by training — it exists so the manifold plumbing is *falsifiable*:
        ``exp_map(mu, log_map(mu, x)) == x`` and ``‖log_map(mu, x)‖ == geodesic(x, mu)`` are exact
        identities with no ground truth, no solver and no tuning behind them. See
        ``environments/test_manifolds.py``, which also checks ``jac_factor`` against autodiff and
        ``metric_inv`` against ``‖∇ geodesic‖_g == 1``.
        """
        ...

    def jac_factor(self, mu: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """|det d(log_map)/dx| Riemannian-volume correction; 1.0 for flat manifolds."""
        ...

    def wrap_point(self, x: jnp.ndarray) -> jnp.ndarray:
        """Canonicalize a point into the manifold's chart (jax). Identity for non-periodic charts."""
        ...

    def wrap_point_np(self, x: np.ndarray) -> np.ndarray:
        """NumPy counterpart of wrap_point, for host-side sampling (RRT*)."""
        ...

    def displacement_np(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """NumPy ambient tangent vector pointing from a to b, ‖·‖ = geodesic distance; broadcasts."""
        ...

    def boundary_ring_np(self, rng: np.random.Generator, eps: float, n: int) -> np.ndarray:
        """``n`` points at exact geodesic distance ``eps`` from ``self.start`` (the Eikonal BC ring)."""
        ...

    # ---- PDE ingredients -----------------------------------------------------------------------

    def metric_inv(self, x: jnp.ndarray) -> jnp.ndarray:
        """Inverse metric g^{ij}(x), as a [dim, dim] quadratic form on ambient gradients.

        Identity for a flat chart; a tangent-space projector for an embedded manifold (so that
        ``grad @ metric_inv(x) @ grad`` recovers the squared intrinsic gradient norm of a function
        that was only ever evaluated through ambient coordinates). Used by every Eikonal residual.
        """
        ...

    def geodesic(self, x: jnp.ndarray, start: jnp.ndarray) -> jnp.ndarray:
        """Analytic base distance (the known base / free-space geodesic)."""
        ...

    def slowness(self, x: jnp.ndarray) -> jnp.ndarray:
        """Smooth cost field s(x) >= 1 (jax)."""
        ...

    def sdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Signed distance to the obstacle union, negative inside (jax)."""
        ...

    def slowness_np(self, points: np.ndarray) -> np.ndarray:
        """NumPy counterpart of slowness, for RRT*'s hot loop."""
        ...

    def sdf_np(self, points: np.ndarray) -> np.ndarray:
        """NumPy counterpart of sdf, for RRT*'s hot loop."""
        ...

    def in_domain_np(self, points: np.ndarray) -> np.ndarray:
        """True where a point lies on the manifold at all (host-side). Default: everywhere.

        Distinct from ``sdf > 0``. An obstacle is *traversable at high cost* — the scene models it as
        slowness, not as a hard collision — whereas a point outside the domain has no field value at
        all. Only the truncated hyperbolic charts have such points: the Poincaré ball's grid and
        sampler cover the chart *box*, whose corners fall outside the ball.
        """
        ...

    # ---- sampling / ground truth ---------------------------------------------------------------

    def sample_domain(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Uniform host-side samples over the domain, as a NumPy array [n, dim]."""
        ...

    def grid(self, resolution: int) -> tuple[jnp.ndarray, tuple[int, int]]:
        """Dense grid over the domain, raveled to [resolution**dim, dim], plus its per-axis shape."""
        ...

    def ground_truth(self, resolution: int, start: tuple[float, ...] | None = None) -> np.ndarray:
        """Dense ground-truth field on a resolution-per-axis grid, raveled to [resolution**dim]."""
        ...

    def render_marker_deg(self) -> tuple[float, float]:
        """(x, y) position of the source in the render chart's degree units (see viz.py)."""
        ...
