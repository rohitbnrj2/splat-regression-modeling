"""Levenberg-Marquardt + slowness continuation on a fixed wrapped-Gaussian basis.

An exploratory lead, NOT a validated method -- see results/lm_optimizer_note.md before using any
number from it. Run: ``python lm_staged_experiment.py [manifold ...]``

WHY LM IS AVAILABLE HERE (it usually is not, for a PINN):
  1. the collocation set is a FIXED precomputed matrix, so the objective is deterministic
     (line search / trust region need this; per-step resampling breaks them), and
  2. the splat centres/shapes (A, B) are FROZEN, so T = Phi(x)V - Phi(x_s)V is LINEAR in V and the
     residual Jacobian is closed form.
Adam has no line search and keeps moving after it should stop; LM accepts a step only if the cost
drops, so "drift past convergence" is structurally impossible.

SELF-SUPERVISION: nothing upstream of scoring reads env.ground_truth. Training consumes only
env.{geodesic, metric_inv, slowness, slowness_np, sdf, sample_domain, boundary_ring_np} -- all on the
allow-list in srms/environments/test_selfsupervised.py. Ground truth appears ONLY after training, for
scoring and for the clearly-labelled cost-at-truth diagnostic.

Stage 1 (linear solve): fit the UNFACTORED field to the OBSTACLE-FREE time-to-go, which is the
  analytic geodesic distance -- free labels, no solver. Value rows AND gradient rows (projected by
  metric_inv so only the tangential part is fitted); the joint fit is what keeps ||grad T||_g ~ 1.
Stage 2 (LM): residual r_i = sqrt(q_i) - 1, q_i = ||grad T_i||_g / s_i  (H-NTFields L_E: smooth at
  the optimum and least-squares structured, unlike the isotropic |1-sqrt q|+|1-1/sqrt q|), plus
  boundary-ring rows sqrt(w)*(Pr V - yr). Run to convergence at each of NR slowness stages.
"""
from __future__ import annotations
import sys
import numpy as np, jax, jax.numpy as jnp
from srms.environments import ENVIRONMENTS
from srms.methods.backends import srm

jax.config.update("jax_enable_x64", True)
K, SIGMA, RES, NR, EPS, NRING, WBC, RIDGE = 1024, 0.35, 120, 10, 0.25, 64, 10.0, 1e-6
BUILD = {"torus": lambda: ENVIRONMENTS["torus"](dim=2),
         "sphere": lambda: ENVIRONMENTS["sphere"](n=2),
         "poincare_hyperbolic": lambda: ENVIRONMENTS["poincare_hyperbolic"](dim=2)}


def run(name: str) -> None:
    env = BUILD[name](); xs = jnp.asarray(env.start)
    B = jnp.asarray(env.sample_domain(np.random.default_rng(1), K))
    A = jnp.repeat((SIGMA * jnp.eye(env.tangent_dim))[None], K, 0); I = jnp.eye(K)
    phi = lambda x: srm.eval_raw((I, A, B), x[None, :], env)[0]
    geo = lambda x: env.geodesic(x[None, :], xs)[0]
    p_s = phi(xs)

    Xg, _ = env.grid(RES); Xg = jnp.asarray(Xg, jnp.float64)
    Pg = jax.vmap(phi)(Xg); bg = jax.vmap(geo)(Xg)
    inside = jnp.asarray(env.sdf(Xg)) < 0.0          # repo scoring convention: mask obstacle interiors
    fitm = (~inside) & (bg > EPS)                     # gt-free: source cone tip excluded, not a failure

    # ---- stage 1: value + (tangential) gradient fit to the obstacle-free field ------------------
    idx = np.where(np.asarray(fitm))[0]; sub = jnp.asarray(idx[:: max(1, len(idx) // 4000)])
    dPs = jax.vmap(jax.jacfwd(phi))(Xg[sub]); Ms = jax.vmap(env.metric_inv)(Xg[sub])
    dbs = jax.vmap(jax.grad(geo))(Xg[sub])
    Dv = Pg[fitm] - p_s[None]; yv = bg[fitm]
    Rg = jnp.einsum("nij,nkj->nik", Ms, dPs).reshape(-1, K)
    yg = jnp.einsum("nij,nj->ni", Ms, dbs).reshape(-1)
    V1 = jnp.linalg.solve(Dv.T @ Dv + Rg.T @ Rg + RIDGE * (len(yv) + len(yg)) * jnp.eye(K),
                          Dv.T @ yv + Rg.T @ yg)

    # ---- stage 2 pieces -------------------------------------------------------------------------
    Xr = jnp.asarray(env.boundary_ring_np(np.random.default_rng(3), EPS, NRING))
    Pr = jax.vmap(phi)(Xr) - p_s[None]
    yr = EPS * float(env.slowness_np(np.asarray(env.start, float)[None, :])[0])
    Xc = jnp.asarray(env.sample_domain(np.random.default_rng(2), 8192))
    dPc = jax.vmap(jax.jacfwd(phi))(Xc); Mc = jax.vmap(env.metric_inv)(Xc)
    bc = jax.vmap(geo)(Xc); sc = env.slowness(Xc); keep = bc > EPS
    dPc, Mc, sc = dPc[keep], Mc[keep], sc[keep]

    @jax.jit
    def resid(V, s_):
        g = jnp.einsum("nkd,k->nd", dPc, V)
        n = jnp.sqrt(jnp.clip(jnp.einsum("ni,nij,nj->n", g, Mc, g), 1e-12, None))
        return jnp.concatenate([jnp.sqrt(jnp.clip(n / s_, 1e-12, None)) - 1.0, jnp.sqrt(WBC) * (Pr @ V - yr)])

    @jax.jit
    def jac(V, s_):
        g = jnp.einsum("nkd,k->nd", dPc, V)
        n = jnp.sqrt(jnp.clip(jnp.einsum("ni,nij,nj->n", g, Mc, g), 1e-12, None))
        Mg = jnp.einsum("nij,nj->ni", Mc, g)
        Je = jnp.einsum("nkd,nd->nk", dPc, Mg) / (2.0 * n * jnp.sqrt(n * s_))[:, None]
        return jnp.concatenate([Je, jnp.sqrt(WBC) * Pr], axis=0)

    cost_j = jax.jit(lambda V, s_: jnp.sum(resid(V, s_) ** 2))
    cost = lambda V, s_: float(cost_j(V, s_))

    def lm(V, sched, iters=30):
        for s_ in sched:
            lam, c = 1e-3, cost(V, s_)
            for _ in range(iters):
                J = jac(V, s_); r = resid(V, s_)
                JtJ = J.T @ J; Jtr = J.T @ r; dg = jnp.clip(jnp.diag(JtJ), 1e-12, None)
                for _try in range(6):                       # trust region: accept only if cost drops
                    d = jnp.linalg.solve(JtJ + lam * jnp.diag(dg), -Jtr)
                    cn = cost(V + d, s_)
                    if cn < c:
                        V, c, lam = V + d, cn, max(lam / 3, 1e-12); break
                    lam *= 3
                else:
                    break
        return V

    V_flat = lm(V1, [sc])
    V_cont = lm(V1, [1.0 + l * (sc - 1.0) for l in np.linspace(1.0 / NR, 1.0, NR)])

    # ---- scoring + diagnostics (ground truth enters ONLY here) ----------------------------------
    gt = jnp.asarray(env.ground_truth(RES)); ok = (~inside) & jnp.isfinite(gt)
    rms = lambda V: float(jnp.sqrt(jnp.mean(((Pg @ V - p_s @ V)[ok] - gt[ok]) ** 2)))
    # THE decisive question: does the objective actually prefer the true field?
    V_true = jnp.linalg.solve(Dv[ok[fitm]].T @ Dv[ok[fitm]] + 1e-6 * int(ok.sum()) * jnp.eye(K),
                              Dv[ok[fitm]].T @ gt[ok]) if bool((fitm & ok).any()) else V1
    print(f"\n=== {name} (k={K}) ===")
    print(f"  do-nothing (T = geodesic)      RMS {float(jnp.sqrt(jnp.mean((bg[ok]-gt[ok])**2))):.4f}")
    print(f"  stage-1 fit                    RMS {rms(V1):.4f}   cost {cost(V1, sc):.3f}")
    print(f"  LM, no continuation            RMS {rms(V_flat):.4f}   cost {cost(V_flat, sc):.3f}")
    print(f"  LM + continuation              RMS {rms(V_cont):.4f}   cost {cost(V_cont, sc):.3f}")
    print(f"  [diagnostic] gt-fitted field   RMS {rms(V_true):.4f}   cost {cost(V_true, sc):.3f}")
    print("  -> if the gt-fitted field's cost EXCEEDS LM's, the objective prefers a wrong field and\n"
          "     no optimizer can close the gap; the missing information is not optimisation.")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["torus", "poincare_hyperbolic"]):
        run(nm)
