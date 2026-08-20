# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements **Splat Regression Models (SRM)** — a function approximator parameterized as a weighted sum of transformed probability density functions:

```
f(x) = Σ_j V[j] · ρ_{A[j], B[j]}(x)
     where ρ_{A,B}(x) = det(A)^{-1} · ρ(A^{-1}(x − B))
```

Parameters `(V, A, B)` represent weights `[k, p]`, scale/rotation matrices `[k, d, d]`, and centers `[k, d]`. The default mother function `ρ` is the standard multivariate Gaussian. The framework compares SRM against MLPs and KANs on regression and physics-informed learning tasks.

## Current focus: self-supervised Eikonal time-to-go on Riemannian manifolds

Learn the time-to-go value function `T` (paths are `-grad T`) from a **self-supervised physics loss**
— no labelled distance field — with the SRM as the representation. Obstacles are a smooth slowness
field. Fast marching exists only to score, never to train.

Read, in this order: [`results/training_strategy.md`](results/training_strategy.md) (what we run and
what has to change — **start here**), [`investigation.md`](investigation.md) (running log, newest
entry last), [`results/manifolds.md`](results/manifolds.md) (the per-manifold diff and its tests).

**Experiment ladder (2-D, 5 seeds each, report one field + a table):** Exp 1 no obstacles (a *gate*:
the answer is the analytic geodesic, so it tests the machinery, not obstacle reasoning) → Exp 2 one
obstacle, 5 random positions → Exp 3 many obstacles. Each stage must run without redoing the last.

**Package layout** — everything lives under `srms/`; the repo root holds only docs and the two paper
scripts (`make_tables.py`, `manifold_figure.py`).

- `srms/environments/` — one module per manifold (`torus`, `sphere`, `so3`, `poincare_hyperbolic`,
  `lorentz_hyperbolic`), each supplying geometry (`log_map`, `jac_factor`, `metric_inv`,
  `splat_precompute`, `log_and_jac`), the scene (`sdf`, `slowness`), sampling, and fast-marching
  ground truth. `base.py` holds the `Environment` protocol plus the shared `union_sdf` /
  `smooth_slowness`. **`test_manifolds.py`** checks five exact identities per manifold (round trip,
  isometry, Jacobian vs autodiff, `‖∇d‖_g = 1`, backend-vs-reference density) before any training;
  **`test_selfsupervised.py`** proves no training path can reach ground truth.
- `srms/methods/backends/` — `srm.py` (wrapped-Gaussian mixture, adaptive densification), `mlp.py`
  (SIREN). Same five entry points: `init_params`, `eval_raw`, `post_step`, `adapt`, `num_units`.
- `srms/methods/strategies/` — the objectives: `eikonal` (unfactored field, plain BC + PDE residual),
  `ntfields` / `pntfields` / `hntfields` (published baselines, `T = base/tau`), `weak_supervision`
  (RRT* prior the PDE refines), `training_aids` (causal weighting, `DensifyController`).
- `srms/experiments/` — evaluation, separate from training. `sweep.py` runs the whole ladder (Exp 1/2/3 differ only in `--num-obstacles`) and reports
  `cost@trained` vs `cost@ceiling`, which decides whether a bad result is the optimizer or the
  objective; `ceiling.py` fits the splat basis to a known field for the representation bound.
- `srms/run.py` (CLI, `tyro` `Config`), `srms/viz.py` (render + score).

**Status.** The whole ladder runs from one command, `./run_experiments.sh {check|validate|exp1|exp2|exp3|plan|all}`.
Exp 1 (no obstacles, scored against the analytic field) is **exact** — RMS 0.0000 on all four
manifolds. Exp 2 (one obstacle, scored against fast marching), read as improvement over `do nothing`:
sphere 0.2552 → **0.0144** label-free (17.7x); torus 0.3197 → 0.1844 label-free → **0.0856** with 30
shadow-targeted RRT* anchors (3.7x); Poincare 0.6527 → 0.3330 label-free → **0.0502** with anchors
(13.0x). Exp 3 (three obstacles, label-free only): sphere 0.6156 → **0.1672** (3.7x), torus 0.4871 →
**0.2621** (1.9x), Poincare 1.3611 → **0.8740** (1.6x). Difficulty orders by curvature — at K=+1
geodesics refocus so shadows are tiny, at K=−1 they diverge so shadows are large.

**Two results that change how to read the rest.** (1) *Hyperbolic weak supervision works, and works
best* — the historical failure was three stacked defects (off-manifold tree nodes, an unconverged
tree, and a flat-chart occlusion ray that mislabelled 82% of H² nodes), all fixed. (2) *RMS does not
predict whether the field is usable.* Every goal is planned twice, once on the learned field and once
by descending fast marching on its own grid — the optimal route, which reaches 60/60 and is the
ceiling every learned number is read against. On 60 goals the label-free torus field (RMS 0.1844, 1
spurious minimum) reaches 55/60 while the *more accurate* anchored field (RMS 0.0856, 2 spurious
minima) reaches 51/60, the whole gap in goals behind an obstacle. No collisions on any field; every
failure is a shadow goal. Report `(RMS, spurious minima)` as a pair. Details in
`results/training_strategy.md`.

**Two standing rules, both learned the hard way (see `investigation.md`):**
1. **No supervised fits, anywhere in the experiment ladder, not even as a diagnostic.** Every
   experiment is self-supervised. A fit-to-truth measures representation capacity, which no
   experiment here asks about, and one silently corrupted the gate's verdict before being removed.
2. **Fast marching is the ground truth with obstacles; score against it, full stop.** With no
   obstacles use the analytic geodesic, which is exact. Compare manifolds by improvement over `do
   nothing`, since scene difficulty differs by an order of magnitude.
3. **Validate a prior before supervising with it.** 30 RRT* anchors from an under-converged tree
   (`rrt_iters=350`) sent the torus backwards, 0.1844 → 0.3393, because 9% of nodes were off by >10%
   and shadow-biased selection concentrated on exactly that tail. At 1500 iterations the same recipe
   gives 0.0735.

**Related work (the positioning; full synthesis archived at `_archive/related_work.md`).** The line we
sit in is neural Eikonal planning: **NTFields** (ICLR 2023) introduced `T = base/tau` with the
isotropic speed loss; **P-NTFields** (RSS 2023) added viscosity and progressive speed scheduling;
**TD-NTFields** (ICLR 2025) added Bellman/normal/causality losses; **H-NTFields** (2026) added sparse
roadmap bounds and is the nearest competitor. All are Euclidean MLPs. The contrast on the other side
is imitation (MPNet), which trusts a planner's values; the measured result here is that refining a
planner prior beats imitating it. Our claim is not "physics-informed planning" — that is their turf
and our physics is light — it is a **manifold-native, interpretable, differentiable** value function
whose per-manifold change is four textbook functions with exact unit tests.

**Run examples:**
```bash
./run_experiments.sh check                                          # identities + smoke matrix, no results
./run_experiments.sh validate                                       # is the RRT* prior fit to supervise with?
./run_experiments.sh exp2                                           # one obstacle, every arm
SEEDS=1 ./run_experiments.sh exp3                                   # three obstacles, quick pass
python -m srms.run --environment sphere --method ntfields --backend srm          # a single solve
python -m srms.experiments.restyle                                  # redraw every figure, no retraining
```

**Retired** (in `_archive/`, not deleted): the `torus.py` lineage and its experiment scripts — dense
RRT* base, screened Poisson, antipodal sampling, the 2-4D dimension sweep, the planning evaluation.
Findings from that line are preserved in `investigation.md`.

## Notation — use these symbols, do not invent others

One symbol, one meaning, everywhere: code, docs, comments and replies. **Define any symbol the first
time it appears in a message, in a few words.** Do not introduce a symbol that is used once.

| symbol | means | note |
|---|---|---|
| `x` | a point on the manifold | the general case; use `theta` only for torus joint angles |
| `T(x)` | time-to-go: cost to travel from the source to `x` | the field being learned |
| `start` | the source point | fixed; `T(start) = 0` always |
| `base(x)` | analytic geodesic distance from `start` to `x` | closed form (`env.geodesic`); the free-space answer |
| `s(x)` | slowness: cost per unit length, `>= 1` | `env.slowness`; `10` inside obstacles, `1` in free space |
| `g(x)` | the splat mixture's scalar output at `x` | the learned correction; `backend.eval_raw` |
| `(V, A, B)` | the splat parameters | weights `[k,1]`, scale matrices `[k,d,d]`, centres `[k,d]` |
| `k` | number of splats | |
| `d` | manifold dimension | `tangent_dim`; `dim` is the ambient size, which differs when embedded |
| `q(x)` | speed ratio `‖∇T‖_g / s` | `q = 1` solves the Eikonal equation; this is what the loss drives to 1 |
| `‖·‖_g` | Riemannian norm, via `env.metric_inv` | identity on a flat chart, a projector when embedded |
| `tau(x)` | NTFields' normalised speed, `T = base/tau` | **their** parameterisation only; ours is `base·exp(g)` |

**The governing equation.** `T` is the time-to-go if `‖∇T‖_g = s` with `T(start) = 0`. Everything
else is a choice of how to write `T` in terms of `g`, and which residual to minimise.

## Stack

- **JAX** for all numerics and autodiff
- **optax** for optimizers (Adam/SGD)
- GPU configuration via `jax.config.update('jax_platform_name', 'gpu')`

## Running scripts

No build system or test runner; modules are executed directly (`python -m srms.…`) — see the run
examples in "Current focus" above. `uv` manages the environment (`pyproject.toml`, `uv.lock`).

## Data conventions

- Inputs `X`: `[n, d]` float32
- Outputs `Y`: `[n, p]` float32
- Splat parameters: `V: [k, p]`, `A: [k, d, d]`, `B: [k, d]`
- `train_mask=(1,1,1)` controls which of `(V, A, B)` are updated during gradient descent
