"""Reference wrapped-Gaussian density on a Riemannian manifold.

Pushforward definition (Said et al. / Chevallier et al.): ``N_w(x) = N(psi^-1(x)) · |det d psi^-1/dx|``
with ``psi = Exp_mu``, so ``psi^-1 = Log_mu`` and the Jacobian factor corrects for curvature.

This is the single-splat, single-point form. ``srms/methods/backends/srm.py`` carries the batched
version the training loop actually calls; keeping this one separate gives ``test_manifolds.py`` an
independent implementation to check the backend against, and a place to state the definition once.
The manifold-specific S^2 / SE(2) primitives that used to live here are superseded by
``srms/environments`` and are archived in ``_archive/lib_preexisting/manifold_splat_full.py``.
"""

from __future__ import annotations

import jax.numpy as jnp


def eval_wrapped_gaussian(x, mu, A, log_map_fn, jac_factor_fn, dim):
    """Wrapped-Gaussian density at one point for one splat.

    Args:
        x: Query point on the manifold.
        mu: Splat centre.
        A: Tangent-space covariance factor, [dim, dim].
        log_map_fn: ``(mu, x) -> [dim]`` tangent coordinates (= psi^-1).
        jac_factor_fn: ``(mu, x) -> scalar`` volume correction |det d psi^-1/dx|.
        dim: Tangent-space dimension (= size of the log map's output).

    Returns:
        Scalar density.
    """
    v = log_map_fn(mu, x)
    z = jnp.linalg.solve(A, v)
    det_A = jnp.abs(jnp.linalg.det(A))
    normal = jnp.exp(-0.5 * jnp.dot(z, z)) / ((2.0 * jnp.pi) ** (dim / 2.0) * (det_A + 1e-12))
    return normal * jac_factor_fn(mu, x)
