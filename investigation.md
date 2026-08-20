# Investigation Log — Path Planning on Manifolds with Splats

Running log of the effort to do **optimal path planning on manifolds using Splat
Regression Models**, trained with a **self-supervised PDE loss** (no labelled
distance field). Theory and justification live in [`theory.md`](theory.md); this
file tracks what we actually try, what we measure, and what we learn.

> **Premise.** A geodesic value function `T` (time-to-go) can be represented by a
> splat field and solved from the Eikonal PDE (via the smooth transform
> `φ = e^{−T/ε}`, `ε²Δφ = s²φ`, `T = −ε log φ`). Paths are then `−∇T`. If this
> works on the plane and on the sphere, it generalizes to manifolds.

---

## Executive summary — state of play (checkpoint)

Main file is [`torus.py`](torus.py) (2-joint arm C-space = 2-torus, obstacles as a slowness field,
FMM ground truth). Canonical scene: seed 1, 3 obstacles, 384 splats, 4000 steps. **Fair baseline
suite** ([`run_baselines.sh`](run_baselines.sh)), RMS vs FMM:

| method | supervision | RMS |
|---|---|---|
| B1 vanilla Eikonal PINN | none | 0.534 |
| B2 NTFields (`base/τ` + speed-match loss) | none | 0.331 |
| B3 P-NTFields (+ progressive + causal) | none | 0.307 |
| B4 300 RRT* samples as **anchors** (trust values) | sparse | 0.393 |
| B5 300 RRT* samples as **base + Eikonal refine** | sparse | **0.109** |
| B6 supervised FMM-fit (oracle) | full GT | 0.003 |

**What we learned (confirmed, honest):**
1. **The splat representation is not the bottleneck** — the oracle (B6) fits the true field to 0.003.
   Everything above it is a *solver/supervision* limit.
2. **Pure physics-informed (P-NTFields analogue) tops out ~0.31** — the Eikonal residual constrains
   *slope*, not *level*; behind obstacles the level is genuinely under-determined.
3. **A prior is necessary, and *how* you use it decides everything.** Same 300 RRT* samples:
   trusting them as anchors (B4, 0.393) is **worse than no supervision** (bakes in RRT*'s suboptimal,
   too-high costs); using them as a rough **base the Eikonal refines** (B5, 0.109) wins by 3.6×,
   because the physics *shaves the planner's suboptimality toward the optimum*. (Anti-MPNet.)
4. **The Eikonal genuinely refines a suboptimal/sparse RRT* prior** (`refine_experiment.py`): −31–41%
   error, and it is the *physics* doing it (physics-led `base_reg=3` beats fit-base `base_reg=30` ~2×).
5. **Graceful, not immune, under thinning** (`thinning_experiment.py`, `figures/thinning.png`): refined
   error `~nodes^-0.67` vs base `~nodes^-0.89` — the physics flattens the slope and helps *most when
   sparsest* (−41% at 81 nodes, −4% at 601), but it is a power law, not flat/log — error still grows
   as samples thin.

**Honest open question (untested):** the true high-D test is error vs sample *dispersion* as dimension
rises at fixed budget (2→3→4-D) — not raw node count. Whether the physics keeps rescuing there is the
make-or-break. A **stronger base construction** at fixed samples (e.g. graph shortest-path over RRT*
*edges* instead of the straight-hop softmin) is the untried lever if it doesn't.

**Fixed-budget dimension sweep** (`torus_nd.py`, isotropic flat torus, 300 RRT* nodes, Godunov-Eikonal
GT). As dimension rises at fixed budget: dispersion 0.19→0.47→0.59, base RMS 0.113→0.201→0.333, refined
0.115→0.169→0.322; physics gap ~0 → +0.032 → +0.011. So error grows ~linearly with dimension (curse is
real, not beaten), fields stay usable through 4-D, and the refinement helps in 3-4D but modestly and
*non-monotonically* (the collocation itself thins in 4-D, so both prior and PDE degrade). Graceful
degradation, consistent with the 2-D thinning slope — supporting evidence the SRM works across
dimensions, not a claim it defeats the curse. `torus_nd.py` is self-contained (does not touch `torus.py`).

**Stress test — does the PDE genuinely help, or is it just the RRT* prior?** (`stress_test.py`, 10 d=2
scenes, obstacles 1→10, 300 RRT* nodes, aligned params.) **Every scene improved: mean +25% RMS, +29%
max-error reduction (range +11% to +33%).** So the physics is not collapsing and is not "just RRT*" —
it robustly refines the prior across complexities. A single easy scene can show ~+4% (base already
near-perfect); the *distribution* is what shows the ~25% contribution. Corroborates the earlier
refine-experiment finding (physics helps most where the base is worst). Note on `torus_nd.py`: it must
use `torus.py`-aligned params (obstacle 0.5–0.9, ramp 0.15, gamma 0.01, 4000 steps) — earlier drift to
harder obstacles + gamma 0.02 muted both the numbers and the PDE, which looked like "PDE not helping".

**3-D base_reg / capacity sweep** (`capacity_3d.py`, fixed 2048 collocation, 600-node base RMS 0.188).
The weak 3-D PDE was the *regularizer* (`base_reg·mean(g²)`, `T=base·exp(g)`) winning the prior↔physics
tug-of-war, **not** diluted collocation (random resampling accumulates coverage) and **not** capacity:

| base_reg | splats | solved RMS | impr | max |
|---|---|---|---|---|
| 3.0 | 384 | 0.176 | +7% | 2.05 |
| 1.0 | 384 | 0.164 | +13% | 1.38 |
| **0.3** | 384 | **0.158** | **+16%** | **0.93** |
| 0.1 | 384 | 0.164 | +13% | 1.13 |
| 1.0 | 768 | 0.169 | +10% | 1.88 |
| 1.0 | 1500 | 0.171 | +9% | 1.64 |

**Lower `base_reg` restores the physics** (+7%→+16%, max 2.05→0.93); ~0.3 is the sweet spot (0.1 drifts).
**Rule: scale `base_reg` DOWN as dimension rises.** Surprise: **more Gaussians does NOT help — it mildly
hurts** (384→1500: 0.164→0.171), so representation capacity is *not* the 3-D bottleneck (384 suffices;
2-D oracle already ~0.003). Training loss converges cleanly (`log10≈−2.9`) — no optimisation failure.
**Bottom line: the PDE genuinely contributes in 2-D (+25%) and 3-D (+16% at the right base_reg); the one
real high-d knob is the regularizer balance, not collocation count or splat count.**

**Planning evaluation — the honest negative result (`planning_eval_full.py`, 2-D→4-D, 6 obstacles, naive
−∇T descent, no guard).** SR / collision-free / path-cost-vs-FMM-optimal / ms-per-query, PDE field vs raw
softmin base vs RRT*:

| d | PDE (SR/cf/opt) | base (SR/cf/opt) | RRT* opt |
|---|---|---|---|
| 2 | 100/100/0.946 | 100/100/0.940 | 1.005 |
| 3 | 100/100/0.910 | 100/100/0.908 | 0.952 |
| 4 | 98/98/0.887 | **100/100**/0.887 | 0.921 |

**The PDE never improves planning and marginally *hurts* at d=4.** Reason: RRT* cost-to-come is already a
valid, monotone, minima-free value function; smoothing it (soft-min / splat) preserves that, so the base
always plans and the PDE has nothing to fix — it only nudges RMS (which diminishes: +25%→+16%→+4% over
d=2→4) and optimality marginally. **Given a roadmap, the physics is redundant for planning.** But the
*field* (either) beats RRT*'s own cost at every dimension (0.89–0.95 vs 0.92–1.005) and plans in
~90–150 ms/query — that win is the *representation*, not the physics.

**Reframe (decision pending with user).** Do NOT frame the paper as "physics-informed planning" (that is
NTFields/H-NTFields' turf, and our physics is light; H-NTFields itself came back to a roadmap). The
defensible contributions: (1) an **interpretable, differentiable, manifold-native splat value function**
with an **anisotropic Riemannian metric** (`metric_inv → M(θ)⁻¹`) that NTFields' *Euclidean* MLP cannot
represent — the uncontested part; (2) building it **cheaply from a sparse roadmap + label-free
self-supervision** (no FMM). Physics = the self-supervision mechanism, not the headline.

**SRM vs MLP on the NTFields lineage — fair head-to-head (2026-08-09, `srms/`).** H-NTFields
(arXiv 2604.13204, the weak-supervision one) and its own PDE-only ablation, 2-D torus, 4000 steps,
2048 collocation, identical scene/loss/sampling/optimizer/lr/seed — **the only CLI difference is
`--backend`**. SRM starts at 64 splats and densifies; MLP is SIREN with Fourier features.

| Baseline | Backend | Params | RMS | max err |
|---|---|---|---|---|
| No weak supervision (`--hnt-lambda-r 0`) | SRM 397 splats | 2,779 | 0.291 | 3.15 |
| | SRM 512 (to cap) | 3,584 | 0.255 | 1.13 |
| | MLP w=40 | 3,521 | 0.253 | 0.89 |
| | MLP w=128 | 33,793 | 0.218 | 1.00 |
| **H-NTFields (weak supervision)** | SRM 397 splats | 2,779 | 0.0637 | 0.395 |
| | SRM 512 (to cap) | 3,584 | 0.0659 | 0.414 |
| | MLP w=40 | 3,521 | 0.0650 | 0.328 |
| | MLP w=128 | 33,793 | 0.0667 | 0.343 |

**Both representations land in the same place.** With weak supervision the three runs agree to within
5% (0.0637 / 0.0667 / 0.0650) — almost certainly inside seed noise, so *no ordering should be claimed
from this table without a seed sweep*. Weak supervision helps ~4x on both backends (0.29→0.064,
0.22→0.067), reproducing H-NTFields' qualitative claim and beating this log's previous best sparse-
anchor result (0.177, line ~761) by 2.8x. Pure physics reproduces the historical B2/B3 plateau
(0.291/0.218 here vs 0.331/0.307 then).

**No parameter-efficiency advantage survives, once the SRM is run to its capacity bar.** The first
table hinted at one (SRM 2,779 @ 0.0637 vs MLP 3,521 @ 0.0650), but that was schedule-limited: the SRM
had grown 64 + 7x48 = 397 splats and stopped because the *schedule* ran out, never reaching
`max_splats`. Re-run with `--densify-every 200` it saturates 512 splats by step 2600 and scores
**0.0659 at 3,584 params** — i.e. at matched size the two representations are indistinguishable
(0.0659 vs 0.0650), and the earlier 21% edge was noise. **Read the table as a tie at matched
parameters, in both supervision regimes.** The 33,793-param MLP is not a baseline that was beaten; it
is simply oversized for this problem and does no better than the 3,521-param one. Neither
representation's floor has been found — nobody swept the MLP downward either.

**Capacity helps exactly where the level is under-determined.** Going 397 -> 512 splats *hurts*
slightly under weak supervision (0.0637 -> 0.0659) but clearly helps without it, above all on the
worst case: **max|err| 3.15 -> 1.13 (-64%)**, RMS 0.291 -> 0.255. That is the signature of coverage
holes, not of missing capacity per se — extra splats fill splat-free regions where `tau ~ sigma(bias)`
leaves `T ~ base` (the free-space geodesic, which under-estimates in obstacle shadows). Weak
supervision pins that level directly, which is why extra capacity buys nothing once it is on.
Consistent with the earlier 3-D sweep where more Gaussians mildly hurt.

**The SRM's max error is a coverage effect — now with supporting evidence.** Pure-physics SRM had
max|err| 3.15 vs the MLP's 1.00; raising capacity alone cut it to 1.13, and weak supervision cuts it
to 0.395 at the *lower* capacity. Both interventions target the same thing (an unpinned level in
splat-free shadow regions), and either one largely fixes it, which is what the locality account
predicts. Still not localized on the map, so treat as well-supported rather than established.

**Fidelity note.** The weak-supervision rows required one adaptation, measured not guessed: uncapped
sphere-packing saturates at 78 nodes on an open torus whose roadmap paths run 1.45x optimal, so `T_lb`
sat *above* true travel time at 94% of nodes and training was 6-8x WORSE than PDE-only (RMS 1.49/1.89/
2.07 — void, do not cite). Capping the free-sphere radius at 0.3 rad gives ~300-460 nodes with paths
optimal to within FMM grid noise. Applied to the shared scene, so identical for both backends.

**Retired / do-not-repeat:** dense (~300-node) roadmap base scored 0.070 but is *cheating* (dense RRT* is
impossible in high-D); the screened-Poisson `φ=e^{−T/ε}` route (theory.md §5) was explored but not
adopted — the direct Eikonal with the NTFields-style factored field won. Antipodal cut-locus sampling
*hurts*; tighter `τ_min` hurts; rim-seeded splats are neutral; gentle slowness ramp hurts (spreads the
obstacle). See the dated sections below for the full trail.

---

## Goal & success criterion

- **Goal:** show splats can plan optimal paths on manifolds using a
  self-supervised loss.
- **Done when:** on each test case the recovered `T` and the extracted path match
  a trusted ground truth within tolerance, *including* at least one genuinely
  curved manifold (the **sphere**).
- **Strategy:** 2D is the *flat special case*. Nail the flat cases and the sphere,
  then the exponential-map machinery (see `theory.md` §7) extrapolates to general
  manifolds.

## ★ North star

**Plan a 2-link robot arm from a start pose to a goal pose, around a workspace
obstacle, as a geodesic/Eikonal problem on its configuration-space torus `T²`
with the arm's kinetic-energy (inertia) Riemannian metric** — solved by the splat
value function and executed with feasibility-guarded path extraction.

Why this problem: it exercises every piece on a genuinely curved, non-trivially
metrized manifold, and it's a recognizable robotics result.
- **Manifold:** the arm's C-space is the 2-torus `T² = S¹×S¹` (two joint angles,
  each wrapping) — our first non-sphere manifold, with periodic geodesics.
- **Riemannian metric (the point):** the kinetic-energy metric `g(θ) = M(θ)`
  (the arm's mass/inertia matrix) is anisotropic and configuration-dependent —
  moving the shoulder costs more than the elbow, and the cost changes with pose.
  The Eikonal becomes `gⁱʲ ∂ᵢT ∂ⱼT = s²` with `gⁱʲ = M⁻¹`. This is the concrete
  "Riemannian metric for path planning" the whole project is arguing for.
- **Obstacle → C-obstacle:** a workspace obstacle maps (via forward kinematics +
  collision check) to a region on the torus; encoded as the smooth high-slowness
  field, avoided for real by the feasibility guard.
- **Deliverable:** a dual figure — the torus C-space (C-obstacle + time-to-go +
  planned path) and the workspace (the arm sweeping start→goal around the object).

Related: the **Dubins car** (repo `dubins_*`) is the same idea one step harder —
its C-space is `SE(2)=R²×S¹` and its metric is *sub-Riemannian* (only
forward motion + bounded turn radius; sideways is forbidden, so the metric is
degenerate). The arm is the cleaner *fully*-Riemannian first target; Dubins is a
natural follow-on.

---

## Reuse policy

Reuse as much of the existing repo as possible — the splat core came from the
original SRM repo and is trusted.

| Reuse | File | Role |
|---|---|---|
| **Splat core** | `lib/splat.py` | `eval_splat` (fwd), analytic grad, `gd_splat_regression`; `diag` mode for scaling |
| Eikonal splat scaffolding | `eikonal_splat.py`, `eikonal_nd_dynamic.py` | existing Eikonal-on-splats experiments |
| Manifold splat | `lib/manifold_splat.py` | tangent-space / manifold splat evaluation |
| Sphere Eikonal | `sphere_eikonal.py` | curved-manifold starting point |
| Obstacles | `obstacle_boost.py`, `demo_obstacle_field` | obstacle speed fields + FMM reference |
| PDE / PINN wiring | `physinf_comparison.py` | `PDEProblem`, `compute_pinn_loss`, `compute_derivatives` patterns |
| Ground truth | `fast_marching_2d` in `ground_truth.py` (adapted from `eikonal_splat.py`); closed-form on sphere | validation |

**Rule:** prefer extending these over rewriting. New code only where the readout
(PDE loss / path extraction) genuinely differs.

---

## Ground truth — the most important part

Getting the reference right is what makes results trustworthy. Per case:

| Case | Ground-truth `T` | How |
|---|---|---|
| 2D free space | analytic | `T(x) = ‖x − start‖` (unit speed) — `make_plane` |
| Sphere | closed form | geodesic `T = arccos(⟨x, start⟩)`; cut locus at antipode — `make_sphere` |
| 2D + obstacle(s) | numerical | **fast marching** `|∇T| = 1/speed`, speed→0 in obstacles — `make_plane_fmm` |
| Path check | — | integrate `ẋ = −∇T/|∇T|` from sampled starts; compare length & shape to reference |

`fast_marching_2d` is validated against the analytic plane: max abs diff `0.0155`
(the expected ~1% first-order fast-marching discretisation error). Iso-time
**contour lines** are drawn on every field — they are the level sets `{x : T=c}`,
i.e. the reachable-in-time-`c` wavefronts; around obstacles they bend visibly.

Metrics to log every run: `‖T_splat − T_ref‖` (RMS + max), gradient-direction
error `∠(∇T_splat, ∇T_ref)`, path length ratio vs. optimal, collision count.

---

## Roadmap

Status: ⬜ not started · 🟡 in progress · ✅ done · ⚠️ blocked

Each milestone has two stages: **(a)** validate the ground truth + splat
representation + visualisation by a *supervised* least-squares fit, then **(b)**
replace the targets with the *self-supervised* PDE loss (the real method).

| # | Milestone | Purpose | Ground truth | Status |
|---|---|---|---|---|
| M1 | **2D free grid** (single source) | flat sanity check of representation + viz, then PDE loss + path extraction | analytic `‖x−start‖` | 🟢 (a) done · (b) self-supervised solve done |
| M0 | **Sphere geodesic** (single source, no obstacles) | prove the manifold logic (ambient embed → exp/log map + PDE) on the simplest curved case | closed-form arccos distance | 🟡 (a) done · (b) PDE pending |
| M2 | **2D single obstacle** | handle a boundary / speed field; recover the cut locus | FMM | 🟢 (a) done · (b) self-supervised solve done |
| M3 | **2D multiple obstacles + intricate shapes** | scale the obstacle machinery (SDF scenes) | FMM | 🟢 5–6 objects done · intricate shapes routed (deep pockets improving) |
| M4 | **Generalize to manifolds** | combine curved metric + obstacles | closed-form / FMM-on-manifold | ⬜ |
| M5 | **Flat torus `T²`** (periodic, no obstacles) | validate the periodic-manifold machinery: intrinsic splat (angle space + wrapped log map), wrapping geodesics, Eikonal with the flat metric | analytic flat-torus geodesic | 🟢 done — RMS `2e-5`, intrinsic `d=n` |
| M6 | ★ **2-link arm on `T²`** (inertia metric + C-obstacle) | the north star: Riemannian metric + workspace→C-space + guarded path + dual viz | FMM on the metric torus | ⬜ |

> Note: M0 (sphere) and M1 (plane) can proceed in parallel — the sphere validates
> the manifold logic while the plane validates the PDE/path plumbing in the
> easiest possible setting.

---

## Method (this is what each experiment implements)

Self-supervised loss on the splat field, per `theory.md` §5–6.

- **Field:** fit `φ(x) = Σ_j V_j · ρ_{A_j,B_j}(x)` with `lib/splat.py`.
- **PDE residual (interior):** `‖ ε²Δφ − s²(x) φ ‖²` — *linear*, smooth. (Fallback:
  direct Eikonal residual `‖|∇T|² − s²‖²` on `T = −ε log φ` if the transform
  underperforms.)
- **Boundary/goal:** `φ(goal) = 1` (i.e. `T(goal)=0`); obstacles via `s(x)→∞`
  (speed → 0) or `φ→0` on obstacle interior.
- **Manifold:** evaluate splats at `log_p(x)`; use the metric `g` in the residual.
- **Path extraction:** integrate `ẋ = −∇T/|∇T|` using the closed-form splat
  gradient.

Open method decisions to resolve empirically:
- value of `ε` (accuracy vs. smoothness trade-off);
- transformed (`φ`) vs. direct (`T`) Eikonal residual;
- number of splats `k` and whether to use dynamic allocation.

---

## Experiment log

_Newest first. One entry per meaningful run; record the change, the metric, and
the decision (keep / discard / follow-up)._

### 2026-07-28 — M1(a) plane & M0(a) sphere: representation + GT + viz validated
- **Environment:** replaced Poetry with **uv**; `pyproject.toml` now PEP-621 with
  pinned deps in `uv.lock` (`uv sync`). `tyro` drives all configs.
- **Code:** `ground_truth.py` (analytic time-to-go + indexable `PlanningProblem`
  with `__getitem__ → (start, goal, time)`); `train.py` (splat fit + 3-panel
  `[GT | pred | error(BWR)]` figure, shared error scale). GT verified first:
  `figures/gt_{plane,sphere}.png` — plane max `√2`, sphere max `π`. Correct.
- **Setup:** supervised LS fit; k=256, num_train=4096, steps=3000, Adam lr=5e-3,
  init_scale=0.3, seed=0, resolution=160.
- **Result:**
  - plane — RMS `2.3e-3`, max|err| `4.6e-2`, rel-RMS `0.79%`.
  - sphere — RMS `1.6e-3`, max|err| `4.7e-2`, rel-RMS `0.28%`.
- **Decision:** keep. Representation + GT + viz confirmed on both domains.
- **Learning:** error is `< 0.01` everywhere except the **genuinely non-smooth
  points**, exactly as `theory.md` predicts: the source cone tip (both domains)
  and the **antipodal cut locus** on the sphere (lon=±180°). A smooth splat sum
  rounds these — expected, not a bug. This is the empirical face of the kink that
  motivates the `φ = e^{−T/ε}` transform for stage (b).

### 2026-07-28 — numerical GT, contours, tooling, and M2(a) obstacle batch
- **Numerical ground truth:** added `fast_marching_2d` (first-order FMM, adapted
  from `eikonal_splat._fmm_reference_2d`) and `make_plane_fmm`, so obstacle fields
  now have a real reference. Validated vs analytic plane: max abs diff `0.0155`.
- **Contours:** all fields now render iso-time contour lines; obstacles drawn as
  hatched patches; obstacle interiors masked (NaN → grey) and excluded from
  training and metrics.
- **Tooling:** `ruff` added (dev group), `ruff format` + `ruff check` clean on
  `ground_truth.py`, `train.py`, `experiment.py`; added `[tool.pyright]` (venv)
  so the IDE resolves imports — VS Code diagnostics now empty on both files.
- **M2(a) — single obstacle, batched:** `experiment.py` generates reproducible
  scenarios from `(global_seed, index)` (random start, goal, obstacle), fits a
  splat to the FMM field, writes params+metrics to `logs/obstacle_batch.tsv`.
  Batch of 4 (seed 0, k=256, steps=1500): RMS mean `4.3e-3`, max `5.1e-3`.
- **Result:** splat reproduces the obstacle-routed field well; error `< 0.05`
  except faint residual at the obstacle rim and the cut locus behind it (as
  predicted). Worst case `scenario_seed0_1` archived.
- **Decision:** keep. M2(a) representation validated. Next is the self-supervised
  PDE loss (stage b) and path extraction via `−∇T`.

### 2026-07-28 — stage (b): self-supervised Eikonal solve (`self_supervised.py`)
- **Loss (no targets):** fit the splat to the viscosity Eikonal
  `|∇T|² − εΔT = s²` via a **relative** residual `(|∇T|² − εΔT)/s² − 1`
  (dividing by `s²` balances free-space vs. stiff obstacle collocation points).
  `∇T`, `ΔT` come from autodiff (`jax.grad` / `jnp.trace(jax.hessian)`).
- **Two design choices that made it converge:**
  1. **Factored field** `T(x) = ‖x−start‖·(1 + g(x))`, splat = deviation `g`,
     `g` initialised to **zero**. This bakes in `T(start)=0` and makes the
     free-space distance the *exact* starting point. (Fitting raw `T` from
     `g≈O(1)` init — the `det(A)⁻¹` amplifies `V` — went nowhere: RMS ~1.1.)
  2. **Relative residual.** With the absolute residual the interior
     (`s=10 → |∇T|=10`) dominated the loss (stuck at `~3.3`); dividing by `s²`
     fixed it (`~0.03`).
- **Free plane (no obstacle):** RMS `2.7e-2`, max|err| `3.6e-2`. Error is a
  smooth uniform `+0.03` bias — exactly the systematic `O(ε)` viscosity bias.
- **Single obstacle (`s=10` inside, `(0.1,0.1,0.35)`):** RMS `2.3e-2`,
  rel-RMS `4.1%` vs. the FMM reference, `k=256`, 4000 steps (~8 min CPU). The
  predicted iso-time contours **bend around the obstacle** matching FMM; error is
  `< 0.05` in free space, concentrated at the obstacle rim and the cut-locus
  shadow behind it (max `0.22`) — the genuinely non-smooth region — plus a faint
  `1/r` seam at the source from the factored form. **This meets the goal: the
  self-supervised loss solves the obstacle plane with no ground-truth targets.**
- **Riemannian framing confirmed in code:** the obstacle is encoded purely as a
  high-slowness region `s(x)`, i.e. the conformal metric `g = s²I`; nothing else
  changes. The flat-plane-with-obstacle *is* the Riemannian case (see `theory.md`
  §7 note). Curved manifolds only swap `g` and add the `exp/log` charts.
- **Open:** the `−εΔT` bias (`~ε`) sets an accuracy floor; smaller `ε` or an
  `O(ε)` bias correction would tighten it. Path extraction (`−∇T`) still to do.

### 2026-07-28 — SDF scenes, reflecting-Neumann (negative result), smooth-slowness
- **Geometry (`scenes.py`):** obstacles are now **SDF** primitives (circles + rotated
  boxes); a scene is their union (`min` of SDFs). One representation gives
  membership (`d<0`), boundary normals (`∇d`), surface samples, *and* intricate
  non-convex shapes as unions. `make_plane_scene` builds the FMM ground truth from
  the same field. Rendering is shape-agnostic (obstacle region outlined from the
  mask), so 5–6 objects and shapes like `cross`/`u_trap` all render correctly.
- **Reflecting Neumann boundary — tried, did NOT work (kept as a finding).**
  Imposed `∂T/∂n = 0` on obstacle rims (normals `∇d`), `s≡1` free space, no interior
  collocation. Result: RMS `0.42`, the field stayed **near straight-through** —
  the far side behind the obstacle was uniformly too low. Diagnosis: the "shadow"
  (extra distance from routing) is a **nonlocal** effect; a purely local boundary
  condition under gradient descent does not propagate it, and the factored base
  `‖x−start‖` biases toward the through-solution. Not a capacity/sampling issue —
  a solution-selection failure. (Would need causal/propagation-aware training.)
- **Pivot — smooth SDF slowness (works).** Encode obstacles as a *smooth* high-
  slowness field `s = 1 + (s_max−1)·σ(−d/width)` (`scenes.smooth_slowness`), the
  conformal metric `g = s²I`. The high interior cost is the **global** signal that
  forces routing (which Neumann lacked); the smooth transition (vs. a hard jump)
  keeps the field splat-representable, so rim error stays small. Relative residual
  `(|∇T|²−εΔT)/s² − 1`, collocation over the whole domain.
- **Single circle (`s_max=12`):** RMS `3.8e-2`, **max|err| `0.074`** — down from
  `0.22` with the earlier hard-jump slowness. The smooth ramp ~halved the rim
  error, the compounding-mitigation we were after. Prediction routes around the
  obstacle *and* reproduces the shadow behind it; error is the mild `O(ε)` bias.
- **5 objects (mixed circles+boxes, `k=384`):** RMS `5.6e-2`, max `0.145`. Error
  did **not** compound linearly (≈1.5× the single-object RMS, not 5×) — it is
  dominated by the slowly-accumulating `O(ε)` viscosity bias (grows with distance
  from the source), not per-obstacle rim error. Routing around all five is correct.
  **This is the key scaling evidence: the smooth-metric formulation adds objects
  without per-object error compounding.**
- **Intricate non-convex shape (`cross`, plus-sign):** global routing correct and
  far field accurate; RMS `7.0e-2` (`s_max=12, k=384`). Residual failure mode is
  the **concave reentrant pockets**: to reach a pocket the wave must travel far
  around an arm, and the near-enclosed field there is underestimated (local max
  err ~1.16 in a small pocket; RMS stays ~10% because it is spatially tiny).
- **Negative tuning result:** raising contrast/capacity (`s_max=18, k=512, 2560`
  collocation) made the pockets **worse** (max `1.16 → 1.77`), not better — a
  stronger barrier deepens the true pocket time, so the same under-shoot costs
  more, and the steeper field is harder to resolve. So the deep-concavity error is
  *not* a capacity/contrast problem; it needs **pocket-focused adaptive
  collocation** (sample where residual/curvature is high) — logged as next work.
  For planning this matters little: pockets are near-enclosed dead-ends one would
  not route into. `s_max=12` is the better default.

### 2026-07-28 — adaptive collocation, weak supervision, path extraction
- **Residual-adaptive collocation (RAR/RAD).** Every `resample_every` steps, score
  a fresh candidate pool by the **PDE residual** (self-supervised — no error oracle,
  no obstacle knowledge) and resample collocation with probability
  `∝ residual^p`, blended with a uniform floor for coverage. The residual is the
  proxy for "where the physics isn't satisfied yet." Real-world-honest: it never
  looks at the ground truth or the SDF.
- **Weak supervision (optional).** Adds a sparse anchor term
  `w·mean((T_pred(xₐ) − T̃(xₐ))²)` where `T̃` is a *cheap* coarse solver
  (low-res FMM here; a stand-in for a coarse planner or sparse measured travel
  times). Anchors are drawn biased toward **large distance from the source**,
  where the accumulating `O(ε)` viscosity bias is worst and the PDE residual
  (a gradient constraint) does not pin the absolute level. Slots in as one extra
  loss term; off by default (`weak_weight=0`).
- **Path extraction.** `extract_path` integrates `ẋ = −∇T/|∇T|` from a goal down
  to the source (closed-form splat gradient), giving the optimal source→goal
  route. `figures/path_*.png` overlays the routes on the field.
- **Single circle (adaptive, with paths):** RMS `3.7e-2`; the four extracted paths
  are correct — unobstructed goals go straight, the two behind/beside the obstacle
  **route around it**. First actual planned trajectories from the splat field.
- **5 objects + paths:** all four paths route correctly around the five obstacles
  (weaving through the gap between two of them to reach the far goal) — path
  extraction is robust on a hard scene.
- **Negative result — naive weak supervision hurt (root-caused + fixed).** First
  5-object run with `weak_weight=0.3` regressed (RMS `0.056 → 0.079`, a local blue
  blob of max `0.99` on the near side of an obstacle). Cause: weak anchors were
  placed by distance only, so some landed **near obstacle rims — exactly where the
  cheap coarse (res-60) FMM is least accurate** — and dragged the field wrong
  there. Fix: sample weak anchors only in open space (`weak_clearance` from any
  obstacle), where the cheap solver is reliable. **Validated:** with the fix weak
  supervision now *helps* — RMS `0.056 → 0.048`, max `0.99 → 0.144` (blob gone),
  and the far-field `O(ε)` drift is visibly reduced (anchors pin the level).
  Lesson: weak supervision helps the **far-field open region**; never anchor where
  the cheap signal is itself unreliable. Off by default; a useful knob when on.
- **Adaptive collocation is mild here**, not a silver bullet: it slightly helped
  the single circle and did not, on its own, resolve the intricate-pocket case.
  Its real value is directing capacity by residual without an error oracle; tuning
  (`residual_power`, `resample_uniform_frac`) matters and can destabilise if it
  starves free-space coverage.

### 2026-07-28 — feasibility-guarded path extraction (soft field, hard safety)
- **Why:** the smooth slowness makes obstacles *permeable* (finite cost), so a raw
  `−∇T` path can cut through — worse in high-D where thin C-obstacles are under-
  sampled. Decouple **guidance** (soft field) from **feasibility** (hard SDF check).
- **How:** during path integration, if a step heads into an obstacle (`d<margin`,
  `−∇T·∇d < 0`), remove the inward normal component (slide along the surface) and
  project any penetration back out along `∇d`. **Head-on fix:** when the slide
  tangent degenerates (path perpendicular to the wall), fall back to a consistent
  perpendicular tangent chosen toward the source — without it, dead-centre
  incidence tunnels straight through.
- **Demo (`figures/path_guard_demo.png`, instant, zero-splat obstacle-unaware
  field):** raw paths enter the obstacle **2/4**, guarded **0/4** — the guarded
  paths hug the rim and route around. On properly-solved fields the guard is a
  no-op (the `s_max=12` wall routed around on its own), i.e. it activates exactly
  when the field would otherwise violate feasibility. This is the guarantee the
  soft metric alone can't give.
- **Limits (honest):** the guard fixes *feasibility*, not *optimality* — if the
  field is very wrong, the rerouted path is safe but not shortest. Adequate
  `s_max` + the guard together give safe-and-near-optimal.

### 2026-07-29 — M5 flat torus `T²` (intrinsic representation, known-base + splat)
- **Two placements clarified.** (A) intrinsic: splat in angle space, evaluate at the
  log map `wrap(θ − B)`, dimension `d = n`. (B) ambient: embed `θ→(cos,sin)`,
  `d = 2n`. Both give the identical exact result; **standardised on (A)** — uniform
  across manifolds (only the log map / metric change) and dimension-efficient.
  `torus.eval_splat_torus` is the periodic (wrapped-displacement) splat eval.
- **Self-supervised mis-selection (finding).** Solving the *global* distance field
  from a smooth (chordal) base drifted to a wrong Eikonal solution on the compact
  torus — RMS got *worse* as the residual dropped (`0.39→0.46`). Same class as the
  Neumann-shadow failure: local residual minimisation doesn't fix global structure.
- **Known-base + splat (the agreed pattern).** Use the analytic flat-torus geodesic
  `‖wrap(θ−start)‖` as the base (known global structure), splat learns only *local*
  corrections. Free flat torus → `g≈0`, **RMS `2e-5`** (essentially exact); shape,
  wrapping, and antipodal cut locus all correct. The splat earns its keep on the
  *unknown* parts: obstacles (C-obstacles) and the arm's non-flat metric `M(θ)`.
- **Metric stays separate** from placement: `metric_inv(θ)` (identity now, `M(θ)⁻¹`
  for the arm) enters only the residual's quadratic form. M6 = swap that hook.

### 2026-07-30 — torus `T²` WITH obstacles: debugging the solve (bug → under-determination → weak)
Goal: show the splat learns a *local* correction (obstacle routing) on the manifold, scored
against a **periodic** fast-marching GT. The convergence chase was itself the lesson:

- **Bug (necessary fix).** `fast_marching_torus` expects a *speed* map (`slowness = 1/input`);
  we passed the *slowness* map, so the GT solved the **inverse** problem (obstacles became the
  *fastest* lanes → wave went through them). The training residual solves `|∇T| = s` (obstacles
  slow → routed around) — the opposite. Diagnostic: residual on the GT was `0.20` (inconsistent)
  vs `0.0065` when the FMM is called with `speed = 1/slow`. Fixed by passing `1/slow`.
- **Under-determination (the real obstacle).** Bug-fixed but bare Eikonal → still RMS `0.85`,
  and the splat's loss (`5e-4`) was *lower* than the GT's own residual (`6.5e-3`): it found a
  valid-but-wrong (under-scaled) Eikonal solution. `|∇T| = s` fixes the field's *slope*, not its
  absolute *level* far from the source — a global level ambiguity. Not a local minimum; the
  objective simply doesn't pin the level.
- **Weak supervision fixes it.** Uniform anchors over the whole torus (from a cheap coarse
  periodic FMM, kept clear of rims) pin the level everywhere → **RMS `0.85 → 0.35`**, bulk error
  now ±0.05, global structure/routing correct. Confirms the diagnosis: the degeneracy is a level
  ambiguity, and level-pinning is the cure. (Anchors must be *uniform*, not far-biased, since the
  ambiguity is global.)
- **What remains.** Thin high-error bands hugging obstacle **rims / cut loci** (max `3.09`, tiny
  in area but inflates RMS): where anchors are excluded, the field is non-smooth, and the coarse
  FMM is least accurate. This is the *selection/smoothness* part — the next lever is **viscosity**
  (`−εΔT`), deferred by request. Bug fix + weak supervision are the two confirmed ingredients.

### 2026-08-01 — torus level under-determination: label-free attempts (all insufficient)
The obstacle-free torus is exact (`2e-5`); WITH obstacles the self-supervised solve
has a stubborn **level** error. Root cause (now clear): the Eikonal residual
`|∇T|=s` constrains the *slope*, but the *level* is the integral of the slope, so a
tiny (loss-invisible) uniform slope bias integrates into a large level offset —
severe on the torus because `T_max≈5.6` (vs plane ~2) and the domain wraps.
Attempts to fix it **without anchors**, on the base instance (`start=(-1.5,-1.5)`):

| attempt | mechanism | RMS | failure mode |
|---|---|---|---|
| plain recipe | causal + RAD + curriculum + obstacle/cut-locus bands | **0.44** | shadow *under*-scales |
| causal annealing | relax causal weight → full residual by end | 0.46 | no change |
| DP consistency | `T(x)=stopgrad(T(y))+s·ds` toward source | diverges (6.6, 7.8) | bootstrapping runaway (self-referential target/direction) — **low loss, diverged field** |
| visibility pin | pin `T=‖wrap(x−start)‖` where the source ray is clear | 0.54 | stops runaway but shadow *over*-scales (maxT≈9) |

**Lesson:** local/geometric terms just push the level around (under↔over↔runaway)
without pinning it — the level is a *global* quantity, so only a *global* fix works:
(a) data anchors (weak supervision, `0.22–0.35`), or (b) a formulation **unique by
construction**. Introspection (`maxT` logged per resample) was decisive for seeing
the divergence in real time. Next: the **screened-Poisson** reformulation
(`ε²Δφ = s²φ`, `T=−ε log φ`) — linear, elliptic, unique given `φ(source)=1`; `ε`
trades accuracy/bias vs. `φ` dynamic range (moderate `ε≈0.5` keeps `φ≳10⁻⁵`).

### 2026-08-01 — screened-Poisson (structurally right, but T-recovery capped) + conclusion
- **Screened-Poisson `ε²Δφ = s²φ`, `T = −ε log φ`** (linear, unique). It **killed the
  level ambiguity**: the field *structure* (routing, wrapping, shape) came out
  **exactly right** — the error was a *uniform* `+ε` offset, not the structural
  shadow error of every other attempt. Clean monotone convergence (solvable, as
  expected). `φ(source)=1` made exact by multiplying the correction by the base
  distance (so `T(source)=0`).
- **But the `T`-recovery is log-sensitive and doesn't beat the recipe.** `T=−ε log φ`
  amplifies φ-errors far from the source; smaller `ε` reduces the bias but *worsens*
  the sensitivity (and the `1/φ0` residual weighting over-emphasises the far field,
  causing local blow-ups). Net: `ε=0.5 → RMS 0.795`, `ε=0.3 → 0.85` (worse). No free
  lunch in `ε`.
- **Conclusion (honest).** Best **label-free** torus result remains the
  **importance-sampled + causal recipe, RMS `0.44`** (structurally imperfect but
  lowest error). The full label-free sweep — causal, causal-annealing, DP
  consistency, visibility pin, screened-Poisson — did **not** beat it. Robustly
  reaching `≤0.2` label-free on this compact, larger-scale torus is a genuine open
  problem; the reliable `≤0.2` path is a **handful of sparse anchors** (`0.22–0.35`
  earlier), whose role we now understand precisely: **global level/branch
  selection**, which local self-supervision cannot infer. Recommendation: use a
  minimal sparse-anchor budget for the working 2-D demo and move energy to the
  high-D validation + the arm (M6), the stated goal. `theory.md`/`validation_highd.md`
  hold the transferable pieces.

### 2026-08-02 — RRT* anchors + best formulation: the honest 2-D torus verdict
- **RRT* anchor source built + validated** (`rrt_star`, `rrt_star_anchors`): pure-NumPy,
  mesh-free, dimension-scalable (the intended high-D anchor source, *not* grid FMM).
  Anchor cost-to-come matches FMM to `±0.04` — accurate. Costs ≥ straight-line base;
  detours behind obstacles captured. This is the piece that transfers to high-D.
- **But 20–40 sparse anchors do not reach `≤0.2` on the torus**, across formulations
  and instances: causal recipe + 30 anchors → `0.41`; no-causal + 40 (weak 1.0) → `0.55`;
  simple-uniform + 40 → `0.48`; *moderate* instance (small obstacles) + 40 → `0.37`.
  Historically, dense anchors did better (200 → `0.35`, 400 → `0.22`) — so the **anchor
  count is the limiter**, not their accuracy or the formulation.
- **Why the torus is fundamentally harder than the plane** (the real finding): the plane
  solved obstacles *label-free* to `0.05` because it is **non-compact** — characteristics
  flow outward from the source and never return, so the exact free-space base pins the
  level. The torus is **compact**: characteristics **wrap around the loops**, so the
  value function's level must be **globally consistent around the torus**, and the local
  Eikonal residual under-determines that global level. Sparse anchors pin it at points but
  the level drifts between them. This is a *topological* difficulty, not a tuning failure.
- **Verdict:** `≤0.2` on this compact torus needs either ~150 (still cheap/scalable) RRT*
  anchors (→ ~`0.22`) or a genuinely better global-consistency formulation (open). Best
  with a 40-anchor budget ≈ `0.37`. The high-D value proposition rests on *feasibility +
  bound* certification (`validation_highd.md`), not tight RMS.

<!-- Template for future entries:
### YYYY-MM-DD — <short title>
- **Change:** what was varied vs. previous run.
- **Setup:** case (M#), k, ε, steps, lr, seed, residual type.
- **Result:** T RMS / max err, grad-angle err, path length ratio, collisions.
- **Decision:** keep / discard / follow-up.
- **Learning:** what it tells us.
-->

---

## Open questions & risks

- **Cut-locus accuracy.** Does the `φ` transform recover the kink sharply enough,
  or does it over-smooth the medial axis? (Watch `ε`.)
- **`ε → 0` conditioning.** Small `ε` makes `φ` span many orders of magnitude
  (`e^{−T/ε}`); may need log-space fitting or normalization.
- **Splat coverage.** Do splats migrate to obstacle rims on their own, or do they
  need targeted initialization / dynamic allocation there?
- **Manifold sampling.** Correct collocation-point sampling on the sphere and
  correct `log_p` near the cut locus (antipode is singular).
- **Path integration** near kinks: `∇T` direction is ambiguous on the cut locus.
- **Deep concave pockets** (e.g. `cross` reentrant corners) are under-resolved and
  not fixed by more contrast/capacity — needs **residual-adaptive collocation**
  (densify where the residual/curvature is high). Next concrete task for M3.
- **`O(ε)` viscosity bias** accumulates with distance and is the dominant error on
  clean scenes; try `ε`-annealing or a bias-corrected readout.

---

## Learnings (append as they solidify)

- **Splats represent geodesic fields accurately** on both the plane and the sphere
  (rel-RMS < 1%). Error localises to non-smooth features (source tip, cut locus),
  confirming the theory and motivating the smooth `φ` transform.
- **Reuse caveat:** `lib.splat.gd_splat_regression` re-`jit`s its `variation`
  closure every step (a static arg), so it recompiles per iteration and is
  impractical for long CPU runs. We reuse `eval_splat` (the splat core) and drive
  it with a **jitted Adam + autodiff-MSE loop** in `train.py` instead — ~40 it/s,
  3000 steps in ~75 s. The analytic `eval_splat_grad` and the self-supervised PDE
  loss are separate next steps.
- **Sphere handled in ambient R³** (splat `d=3`, evaluated on the unit sphere)
  rather than via tangent charts — the simplest first embedding. The exp/log-map
  construction from `theory.md` §7 is deferred to stage (b)/M4.
- **Obstacles cost ~2× the free-space error** (RMS `2.3e-3 → ~4.3e-3`) under a
  supervised fit — the extra error is concentrated at the obstacle rim and the
  cut locus behind it, i.e. the new non-smooth features, consistent with the
  free-space finding. No accuracy cliff; the smooth splat handles routed fields.
- **Reproducibility convention:** scenarios derive from `(global_seed, index)` via
  `np.random.default_rng([seed, index])` for geometry and
  `PRNGKey(seed*10000+index)` for the splat. `logs/obstacle_batch.tsv` records
  every scenario's start/goal/obstacles + metrics, so any case is replayable.
- **Obstacles = a metric, not a boundary (what actually works).** Encoding an
  obstacle as a *smooth high-slowness region* (`g = s²I`) is both the correct
  Riemannian view and the robust trainable one: the high interior cost is the
  **global** signal that propagates the routing/shadow. A reflecting Neumann
  boundary is physically exact but only *local*, and under gradient descent it
  fails to establish the nonlocal shadow (field stays near straight-through). Keep
  the metric formulation; it also generalises to curved manifolds by swapping `g`.
- **Smooth beats hard for the splat.** A smooth slowness ramp (SDF sigmoid) vs. a
  hard jump halved the single-obstacle rim error (`0.22 → 0.074`) — a `C∞` splat
  cannot represent a discontinuity, so smoothing the metric is what stops
  per-object error compounding. Adding objects grew RMS sub-linearly (1→5 objects:
  `0.038 → 0.056`); the dominant error is the accumulating `O(ε)` viscosity bias,
  addressable by shrinking `ε` or an `O(ε)` bias correction.
- **SDF is the right obstacle interface** — one representation yields membership
  (`d<0`), the metric ramp, boundary normals (`∇d`), surface samples, intricate
  shapes (unions), and (future) a sensed-environment SDF. See `scenes.py`.

## Checking against NTFields, and the derivation we should have used

We compared to **NTFields** (Ni & Qureshi, ICLR 2023) and **P-NTFields** (progressive
learning) to check for a bias in our own framing. The honest result: our "the torus is
fundamentally hard because it is compact" story was a **rationalisation**. NTFields solves
**4-DOF and 6-DOF manipulator C-spaces — which are compact tori — to 90–96% success** with
FMM-quality time fields. Compactness is not the blocker; our *formulation* had gaps. Three,
all confirmed against their method:

- **Bounded factorisation (the structural fix).** NTFields writes `T = ‖q_s−q_g‖ / τ`
  with `τ ∈ (0,1]`, which **guarantees `T ≥ straight-line distance`** and sends
  `τ→0 ⇒ T→∞` in obstacles. Our `field_value = base·(1+correction)` (`torus.py:187`) leaves
  `correction` unbounded, so it *permits* `T < base` — physically impossible and exactly our
  under-scaling failure. Adopt `T = base_g/τ`, `τ = σ(splat) ∈ (0,1]`: the failure mode
  becomes unrepresentable.
- **Speed-space symmetric loss (conditioning).** We minimise `(‖∇T‖²_{g⁻¹}/s² − 1)²`
  (`torus.py:197`) — quartic in `∇T`, unbounded, and with `s²=100` inside obstacles those
  points dominate the gradient while the level-setting free-space bulk starves. NTFields
  matches the *speed* `Ŝ = 1/‖∇T‖` to the known speed `S = 1/s` with a symmetric, bounded,
  √-smoothed objective. Recommended: `L = mean(Ŝ/S + S/Ŝ − 2)` (≥0, zero iff `Ŝ=S`,
  balanced across free space and obstacles).
- **Progressive obstacles (local minima).** They anneal obstacle influence `λ: 0→1` from the
  free-space solution. Our curriculum was *spatial* (source-outward), not *obstacle-contrast*.
  Use `s_λ = 1 + λ(s−1)`, `λ: 0→1`, starting from the exact free-space field `base` already
  gives.

**The derivation to standardise on** (manifold-general, NTFields-aligned):
`T = base_g/τ` (`τ=σ(splat)∈(0,1]`) → metric dual-norm `Ŝ = 1/√(∇Tᵀ metric_inv ∇T)` →
symmetric speed-match `mean(Ŝ/S + S/Ŝ − 2)` → progressive slowness `λ:0→1` → keep RAD
residual sampling for the cut locus. The `metric_inv` hook (`torus.py:152`, identity now)
carries the anisotropy; the base becomes the metric geodesic `‖wrap(θ−start)‖_g`.

### Isotropic (NTFields) vs anisotropic (ours) — two distinct anisotropies
- **Metric anisotropy `metric_inv(θ)`.** NTFields uses one *scalar* speed `S(q)` per point →
  cost is direction-independent, wavefronts are circles. That is `metric_inv = c·I`. Our arm
  metric is the inertia matrix `M(q)` from `KE = ½q̇ᵀM(q)q̇`: dense, configuration-dependent,
  off-diagonal-coupled — shoulder motion has high inertia (costly), wrist motion low (cheap),
  and cost depends on *direction* in `q̇`. **A scalar speed cannot represent this;
  `metric_inv → M(θ)⁻¹` can.** Note: our *current* runs still set `metric_inv = I`, so they are
  isotropic too — the anisotropy is a live structural hook, not yet exercised.
- **Representation anisotropy (splat covariance).** Each splat `A[j]` is a full scale+rotation
  matrix (`eval_splat_torus`, `torus.py:141`), so every basis bump is an *ellipse* that aligns
  with field ridges/creases (obstacle rims, cut locus) — one splat where an isotropic basis
  needs many. Always on, independent of the metric. An anisotropic basis is exactly what
  represents an anisotropic metric's elliptical equal-cost sets efficiently.

### What the splat representation buys over NTFields (same problem, different representation)
We solve the *same* Eikonal motion-planning problem; the difference is the value-function
representation, and it is why their techniques alone are not the ceiling for us:
1. **Explicit, interpretable geometry** — centres, covariances, weights are readable and
   editable; NTFields is a black-box `(q_s,q_g)→τ` ResNet.
2. **Anisotropic basis functions** — splats represent ridges/creases natively via covariance;
   an MLP with scalar speed must compose many units and still carries no direction structure.
3. **Manifold-native, intrinsic placement** — centres live on the torus, evaluated through the
   flat-torus log map `wrap(θ−B)`, so **periodicity is built in and the metric is pluggable**
   (`metric_inv`); we generalise to curved/anisotropic manifolds by swapping the log-map/metric.
   NTFields embeds the C-space in ambient/Euclidean coordinates and relies on the network to
   learn the wraparound, even though the C-space is topologically a torus.
4. **Known base + sparse correction** — the analytic geodesic *is* the prior; splats model only
   the obstacle-induced deviation. NTFields' `‖Δ‖/τ` is the same idea but with a black-box
   correction.

**Why their tricks are necessary but not sufficient here.** NTFields validated the *isotropic,
identity-metric* case and scaled it dimensionally. A real arm C-space adds a genuinely
*anisotropic* metric (inertia) and nontrivial topology (wraparound, cut locus). We should
adopt their bounded factorisation, speed loss, and progressive schedule (conditioning wins that
apply regardless), but the representation that lets us encode the *metric* and *topology*
directly — anisotropic splats with an intrinsic log-map and a pluggable `metric_inv` — is the
part that is ours, and the part that matters once the problem stops being isotropic.

### Head-to-head: NTFields formulation vs our baseline (same scene, seed 1, 3 obstacles)
Implemented `solve_ntfields` (`torus.py`): `T = base/τ`, `τ = τ_min + (1−τ_min)·σ(splat+bias) ∈
[τ_min, 1]`; symmetric speed-match loss `q + 1/q − 2` (`q = ‖∇T‖_{g⁻¹}/s`); progressive obstacle
contrast `λ: 0.15→1`; **no causal/DP/visibility crutches**, only RAD sampling. Both runs 4000
steps, 384 splats, identical obstacles.

| Metric | Baseline `solve` (causal+vis+RAD) | NTFields `solve_ntfields` (RAD only) |
|---|---|---|
| Final RMS | 0.563 | **0.345** (−39%) |
| max\|err\| | 3.87 | **2.27** |
| loss log10 | −1.6 | **−2.1** |
| RMS @ 1k/2k/3k | 0.55 / 0.53 / — | 0.357 / 0.333 / 0.338 |

**The error changes character, not just magnitude.** Baseline: the whole torus is a uniform blue
**global under-level offset** (pred < GT by ≥0.2 nearly everywhere) — level under-determination
pinned low. NTFields: the global offset is gone and the near-source is **clean white** (the
`base/τ` factorisation forces `T(source)=0` and `τ≈1` in the free near-field — this is the
"start should be pure white" property we kept failing at). Residual error is now *localised* to
two tunable features: (a) an over-shoot behind an obstacle where `τ` hits its floor `τ_min=0.25`
(too much headroom → `T` up to `4·base`, `maxT≈9.9` vs GT `5.6`), and (b) under-leveling in the
far field ~180° from the source — the **cut locus**, genuinely the hardest region.

**Stability lesson (v1→v2).** The first NTFields attempt (unbounded `T=base/τ`, no interior gate)
trained beautifully to loss −2.4 during low contrast, then **blew up at `λ→1`** (RMS 26.7): with
nothing bounding `τ`, the `1/q` penalty in deep obstacle interiors (`s=10`, field can't comply,
`q→0`) drove `τ→0` and `T→∞`. Fix (faithful to NTFields' speed-floor + not-chasing-interiors):
**floor `τ` at `τ_min`** (bounds `T ≤ base/τ_min`) and **gate the loss to the exterior**
(`sigmoid(sdf/w)`, ~0 inside obstacles). Both are physically justified — the field inside a
blocked cell is irrelevant to planning.

**Verdict.** The NTFields *formulation alone* beats our crutch-laden baseline by 39% RMS and,
more importantly, converts a global level failure into two localised, addressable ones. Still
above the 0.2 target; the open levers are a tighter `τ_min` (curb the shadow over-shoot),
stronger cut-locus sampling / more anneal steps for the far field, and optionally re-adding causal
weighting *on top of* the sound formulation. This is the honest confirmation that our earlier
struggle was formulation-limited, not torus-compactness-limited.

### Lever ablation + representation test: the error is level, not representation
Two systematic sweeps on the same scene (all NTFields-style, 4000 steps). Cumulative lever ladder:

| Rung | +lever | RMS | max\|err\| | reading |
|---|---|---|---|---|
| A | v2 (`τ_min=0.25`) | 0.345 | 2.27 | base |
| B | tighter `τ_min=0.40` | 0.377 | 2.13 | **hurts** — wider mismatch with the `s=10` wall |
| C | antipodal cut-locus sampling | 0.377 | 2.39 | **no help** — error is not at the antipode |
| D | causal weighting | 0.345 | **2.04** | tames the worst point (shadow) |

Representation arms (test "Gaussians corrupted by the slowness wall") on the `ref` recipe
(causal + `τ_min=0.25` + **no** antipodal boost + sharp ramp; `ref` itself = **0.307**, the best
label-free result — dropping the default `cutlocus_boost=3.0` was worth ~11%):

| Arm | change | RMS | reading |
|---|---|---|---|
| ref | — | **0.307** | best |
| E | gentle ramp 0.30 **+** rim splats | 0.746 | much worse |
| G | gentle ramp 0.30 only | 0.672 | **the culprit** — widening spreads the obstacle |
| H | rim splats only | 0.339 | ~neutral |

**Diagnostic (`diagnose.py`, near-obstacle zoom).** The violent `‖∇T‖/s` oscillation is *inside*
the gated obstacle (unconstrained, harmless); the **exterior ratio ≈ 1** (rim-annulus std ≈ 0.20)
— the *shape* is locally correct. Yet the full error map shows the **entire far half
under-predicted** (the region behind the central obstacle cluster relative to the source). So the
splats are **not** ringing/capacity-limited near obstacles; they satisfy `|∇T|=s` locally but
accumulate a **detour deficit**: the straight-line `base` prior points *through* the obstacles and
smooth `τ` under-inflates the go-around cost, leaving the shadow **under-leveled**.

**Conclusion.** The pointwise Eikonal residual constrains *slope*, not *level*; behind obstacles the
level is genuinely under-determined and neither smoothing the obstacle (G, hurts) nor adding rim
capacity (H, neutral) touches it — because the error is not at the rim. This is exactly the mode
**sparse anchors are designed to pin** (the integration constant in the shadow), and it argues for
judging the field by **path suboptimality vs RRT\***, not field RMS — the gradient *direction* in
the shadow can be right while the level is off by a constant. Best label-free result: **0.307**.

### Sparse RRT* shadow anchors break the 0.2 target — and prove the diagnosis
Added `rrt_star_anchors_shadow`: RRT* tree → keep cleared nodes → bias selection toward nodes whose
straight source→node ray is **occluded** (in a routing shadow), weighted `1 + shadow_pref·occluded`.
Wired a weak-supervision term `weak_weight·mean((T−cost)²)` into `solve_ntfields` on top of `ref`.
30 anchors (RRT*, **not** FMM; mesh-free, dimension-scalable):

| Run | anchors | weight | RMS | max\|err\| |
|---|---|---|---|---|
| ref | none | — | 0.307 | 1.64 |
| **W1** | 30 shadow (53%) | 0.5 | **0.177** ✅ | 1.00 |
| W2 | 30 shadow | 1.0 | 0.181 ✅ | 1.10 |
| U (control) | 30 **uniform** | 1.0 | 0.229 | 1.47 |

**Two things proven at once.** (1) 30 sparse anchors take us from 0.307 to **0.177 — below the 0.2
target**, matching-ish the Euclidean plane. (2) The **uniform-anchor control U (0.229) is markedly
worse than shadow-targeted W1/W2 (~0.18)** with the *same* 30 anchors — so it is specifically
*placing the anchors in the shadow* that helps, not merely adding supervision. That is direct
confirmation of the level-under-determination diagnosis: the residual supplies the *shape*, and a
handful of anchors supply the *level* exactly where the PDE leaves it free. W1's error map shows the
far-half blue filling in; the residual under-prediction that remains is at the extreme far edges /
wrap seams (near the antipode), not the immediate obstacle shadow. Lighter weight (0.5) beat firm
(1.0) — the anchors only need to pin the constant, not dominate the loss.

**Winning recipe:** NTFields factorisation `T=base/τ` + symmetric speed loss + progressive obstacles
+ causal weighting + **no** antipodal oversampling + **30 shadow-targeted RRT* anchors** (weak_weight
0.5). Obstacles stay modelled as a slowness field throughout — the anchors fix the *level* symptom
that the slowness formulation leaves under-determined, without changing the obstacle model.

### Pushing below 0.1: the straight-line base is the real ceiling; a roadmap base breaks it
Piling capacity + more anchors on the anchor recipe **plateaued** (`PUSH_A` 0.184, `PUSH_B` 0.174) and
`max|err|` *rose* to ~1.5 — adding rim/far anchors created a red over-prediction **wedge** while the far
half still under-predicted. Diagnosis: the straight-line `base` prior points *through* obstacles, so `τ`
must do heavy, conflicting **non-local** work in the shadow, and pinning it with more (biased) anchors
just fights the PDE. This burden only worsens in high-D — so *more anchors is the dimension-fragile road*.

**Fix — replace the prior with an accurate, mesh-free, obstacle-aware base.**
`roadmap_base(θ) = soft-min_i (cost_i + slowness-weighted ‖wrap(θ−node_i)‖)` over an RRT* tree, where the
last hop is weighted by the **mean slowness along it** so hops through obstacles are penalised. Base
quality (zero splat, vs FMM):

| last hop | base RMS | base max (GT 5.53) |
|---|---|---|
| straight (corner-cutting) | 0.517 | 4.42 |
| **slowness-weighted** | **0.065** | **5.54** |

The straight hop underestimates the shadow (cuts corners) — the same failure as the straight-line base.
The slowness-weighted hop makes the base **accurate zero-shot (0.065)** and it is a smooth, differentiable
(soft-min), mesh-free field — it *scales past the grid dimensions* (RRT*, no FMM).

**Two gotchas, both resolved:** (1) pure Eikonal training on top *drifts the accurate base away*
(`maxT→10`) — the same level under-determination — so regularise the splat toward the base
(`base_reg·mean(g²)`, `T=base·exp(g)`) and it only *smooths kinks* instead of re-levelling. (2) the
**soft-min `γ` must be small** (`γ·log N` is the underestimate): `γ=0.05→0.15 RMS`, `γ=0.01→0.099 RMS`,
`max|err| 0.34` (vs 1.0–2.3 for every earlier approach). **Below the 0.1 target, no FMM, no sparse
anchors** — the RRT* roadmap is a *dense mesh-free target* (self-supervised from the known slowness), and
the splat is the compact final representation fit to it + light Eikonal.

**Full sweep result (γ=0.01, 384 splats, 4000 steps):** `base_reg=30` → **RMS 0.070, max|err| 0.27**;
`base_reg=10` → RMS 0.072, max|err| 0.35. The error map is clean across the whole torus (prediction
tracks GT behind the obstacles; no far-half blue, no wedge); the residual is a faint ~0.05 under-haze
plus tiny rim speckles (the sharp `s=10` wall, where max|err| sits). The field is smooth (planning-ready).

**Progression on the same scene:** baseline `solve` 0.563 → NTFields formulation 0.345 → `ref`
(+causal, −antipode) 0.307 → +30 shadow anchors 0.177 → **roadmap base 0.070** (8× the baseline). And
`max|err|` fell from ~3.9 to **0.27**. The decisive move was the *obstacle-aware base*, not more supervision.

**Framing / open decision.** This shifts work from "splat+PDE discovers the shadow" to "splat is the
compact representation of a solution the mesh-free roadmap scaffolds" — accurate and dimension-robust,
but leaning on the roadmap for the level (which the pure PDE genuinely under-determines). Roadmap is
computed from the *known* slowness (self-supervised, RRT* not FMM), so it is legitimate and scalable;
the tradeoff is how much of the "solving" the splat vs the roadmap does. Next: high-D validation battery
(`validation_highd.md`), then the anisotropic metric (`metric_inv → M(θ)⁻¹`) for the arm.

## Fair baseline suite (same scene/seed/budget; only the method varies)
`run_baselines.sh` — canonical scene (seed 1, 3 obstacles), 384 splats, 4000 steps, 2048 collocation,
identical sampling; `cutlocus_boost=0` for all (antipodal sampling was shown to hurt).

| # | Baseline | Supervision | RMS | max\|err\| |
|---|---|---|---|---|
| B1 | Vanilla Eikonal PINN (`base·(1+g)`, residual²) | none | 0.534 | 3.78 |
| B2 | NTFields (`base/τ` + symmetric speed-match loss) | none | 0.331 | 2.02 |
| B3 | P-NTFields (B2 + progressive anneal + causal) | none | 0.307 | 1.64 |
| B4 | 300 RRT* samples as ANCHORS (trust values) | sparse (300) | 0.393 | 2.03 |
| B5 | 300 RRT* samples as BASE + physics refine | sparse (300) | **0.109** | 0.64 |
| B6 | Supervised FMM-fit (oracle) | full GT | **0.0029** | 0.026 |

**B4 vs B5 — the headline (matched ~300-sample budget).** Same RRT* samples, two injections: as trusted
*anchors* (B4) vs a rough *base the Eikonal refines* (B5). **B5 beats B4 by 3.6×** (0.109 vs 0.393),
and — striking — **B4 is *worse than B3* (0.307, no supervision at all).** RRT* costs are upper bounds
(suboptimal → too high); pinning 300 of them bakes the planner's suboptimality into the field, while
using them as a soft prior lets the Eikonal *shave them back toward optimal*. This is the direct,
same-budget measurement of the thesis: **imitating the planner's values is worse than ignoring them;
physics-refining them is what wins** (anti-MPNet, and distinct from pure-physics P-NTFields). Note the
*old* (unfair) suite compared 30-anchor B4 (0.169) to a ~300-node dense-base B5 (0.070) — retired.

### Thinning: how error grows as samples get sparser (`thinning_experiment.py`, `figures/thinning.png`)
B5 (base + physics-led refine) over node budgets 81→601, base-alone vs PDE-refined RMS:

| nodes | base RMS | refined | physics gap |
|---|---|---|---|
| 81 | 0.451 | 0.267 | +0.185 (−41%) |
| 121 | 0.411 | 0.247 | +0.164 |
| 201 | 0.238 | 0.135 | +0.103 |
| 351 | 0.160 | 0.105 | +0.055 |
| 601 | 0.077 | 0.074 | +0.003 |

Fitted: **base `~nodes^-0.89`, refined `~nodes^-0.67`.** Two encouraging signals: (1) the Eikonal
*flattens* the sparsity-degradation slope (−0.89→−0.67, i.e. sub-linear not linear); (2) the physics
contributes **most when samples are sparsest** (−41% at 81 nodes vs −4% at 601) — ideal for high-D where
samples are always sparse. **Honest caveat:** −0.67 is sub-linear but *not* flat/logarithmic — error
still grows as a power law (7× fewer nodes → ~3.6× worse refined RMS). So *graceful degradation, not
immunity*. **This is a node-count curve in 2D; the true scaling test is error vs sample *dispersion* as
dimension rises (fixed budget, 2→3→4-D)** — that, not raw count, is what transfers across dimensions.

**The oracle B6 = 0.003 is the key control: the splat *representation* is not the bottleneck.** A splat
of this budget fits the true field to 0.003 RMS, so the entire B1→B6 gap is **solver + supervision**,
never representation capacity. Clean physics-informed progression: naive 0.53 → factorisation+speed-loss
0.33 → +progressive+causal 0.31 (this is the fair planner-free NTFields/P-NTFields analogue) → sparse
anchors 0.17 → roadmap base 0.07 → representation ceiling 0.003.

**Honest read of B5 vs the literature.** B5 uses a mesh-free RRT* base — *not* imitation (MPNet) and
*not* pure-physics (P-NTFields), but a **noisy-prior + physics-refinement** (multi-fidelity) hybrid. Its
scalability hinges on one open question, tested in `refine_experiment.py`: can the Eikonal *refine a
sparse (few-node, high-D-feasible) base* down, or does it need a dense base (impossible in high-D)? Base
RMS vs RRT* tree size on this scene: 101 nodes→0.43, 251→0.18, 501→0.12, 1001→0.09. **A *dense* base
(~300 nodes) in B5 is effectively cheating and will not scale — discard it as a headline.** The legitimate
method is *sparse* RRT* (anchors or a sparse base) + Eikonal refinement.

### Can the Eikonal refine a suboptimal/sparse RRT* solution? Yes — and it's the physics doing it
`refine_experiment.py`: sparse trees, two base-weights (`base_reg=3` lets the PDE lead; `30` mostly fits
the base). Value RMS (vs FMM) and Eikonal residual (consistency), base → refined:

| tree | base RMS | refined (`base_reg=3`) | refined (`base_reg=30`) | residual (reg 3) |
|---|---|---|---|---|
| 151 nodes | 0.331 | **0.212** (−36%) | 0.279 (−16%) | 0.057 → 0.020 |
| 251 nodes | 0.183 | **0.120** (−34%) | 0.154 (−16%) | 0.035 → 0.019 |
| 501 nodes | 0.114 | **0.079** (−31%) | 0.097 (−15%) | 0.019 → 0.017 |

**(1)** The Eikonal reduces the error in *every* case. **(2)** It is the *physics*, not base-fitting: the
physics-led weight (`base_reg=3`) refines ~2× more than the fit-base weight (`30`) — if the base carried
it, the two columns would match; they don't. **(3)** Physical consistency (residual) drops 30–65%, so the
field becomes much more Eikonal-consistent (better planning gradients) even where value-RMS moves modestly
— i.e. RMS understates the PDE's contribution. **Caveat:** refinement is *partial* — final error tracks
base density (very sparse 151→0.21; you can't refine garbage to perfection). But 251 nodes→0.120 already
beats B4's 30-anchor 0.169, all mesh-free/sparse. **High-D outlook: cautiously positive** — the physics
reliably corrects a rough planner, but there is a floor set by how sparse the planner can be; that
floor-vs-dimension is the next thing to characterise. This vindicates the *noisy-prior + physics-refine*
thesis and, crucially, shows the Eikonal — not the base — is the corrector.

---

## 2026-08-19 — Package cleanup, six defects fixed, and the obstacle-free gate

Session goal: audit the code for bugs, cut what is not needed, and define the training strategy for
the planned experiment ladder (Exp 1 no obstacles → Exp 2 one obstacle → Exp 3 many, 2-D, 5 seeds).

### Defects found and fixed

| # | Where | Defect | Status |
|---|---|---|---|
| 1 | `methods/strategies/eikonal.py` | Residual was `‖∇T‖ − 1/s`. `slowness` is cost per unit length, so `T = ∫s dl` and `‖∇T‖ = s`. `manifolds.md` §6a records this as fixed; it was not fixed in this tree. | fixed |
| 2 | all `environments/*` | `num_obstacles=0` raised `ValueError: need at least one array to stack` — **Experiment 1's exact configuration could not be constructed on any manifold.** | fixed |
| 3 | `environments/lorentz_hyperbolic.py` | No `splat_precompute` / `log_and_jac`, both required by `srm.eval_raw`, so the environment could not be trained with the SRM backend at all. | fixed |
| 4 | `environments/test_selfsupervised.py` | Referenced `ENVIRONMENTS["hyperbolic"]`, removed when hyperbolic split into Poincaré/Lorentz; `KeyError` outside the `try`, so the whole suite died. | fixed |
| 5 | `run.py` | Computed `env.ground_truth` *before* training and again after, contradicting its own docstring that ground truth exists only post-`solve`. Not a contamination (it was never passed to `solve`) but the stated invariant was false. | fixed |
| 6 | `methods/strategies/eikonal.py` | No densification wiring, while `cfg.densify` defaults True — the config claimed adaptive capacity the strategy never used. | fixed |
| 7 | `environments/test_manifolds.py` | `containment_mass` applied the 2-D closed form `1 − exp(−R²/2σ²)` to SO(3), whose tangent space is 3-D: predicted 0.9675 where the chi₃ value is 0.9233 against a measured 0.9225. The standing mass-check failure was **the formula's, not the model's**. | fixed |

Convention for #1 settled by measurement, not by reading: finite-differencing the fast-marching
field on the 2-D torus gives median `‖∇T‖/s` = **0.999** inside obstacles and **1.004** in free
space, against `‖∇T‖·s` = 78.5 inside.

Two checks added, both of which would have caught a defect above:

- **`backend == reference`** in `test_manifolds.py`. `srm.eval_raw` does not call
  `eval_wrapped_gaussian`; it re-implements the density batched, with the frame hoisted into
  `splat_precompute` and the log map fused into `log_and_jac` — three chances to disagree with the
  definition that nothing checked. They agree to ≤ 3e-11 on all five manifolds. (`srm.py`'s docstring
  claimed it *delegated* to the reference; corrected.)
- `test_selfsupervised.py` now covers all five environments and all 15 strategy × manifold pairs:
  **0 violations**.

### Experiment 1 gate: the unfactored Eikonal objective prefers a wrong field

`srms/experiments/sweep.py`. Obstacle-free torus T², `eikonal` strategy (unfactored `T = SRM(x)`),
256 splats, 1500 steps, 3 seeds. With no obstacles the answer *is* the analytic geodesic, so this is
the easiest test that exists.

| seed | trained | do nothing | ceiling | cost@trained | cost@ceiling |
|---|---|---|---|---|---|
| 1 | 2.6302 | 0.0463 | 0.0794 | 0.00996 | 0.22764 |
| 2 | 2.6212 | 0.0463 | 0.1310 | 0.01190 | 0.78728 |
| 3 | 2.6821 | 0.0463 | 0.2109 | 0.01314 | 2.19735 |

`do nothing` is `T = geodesic`, exact here, so 0.0463 is the fast-marching grid's own discretisation
error. `ceiling` is the same splat basis least-squares fitted to the truth. **The trained field is
57× worse than doing nothing**, and the objective is **20–170× lower** at the trained field than at
the fitted-to-truth field, on every seed.

What the optimizer found, measured: `|∇T|` median **1.007** (p05 0.747, p95 1.125 — the PDE is
satisfied), field range **[−1.15, 1.21]** against the truth's [0.03, 4.42], and correlation with the
truth **0.053**. It is a small-amplitude sawtooth: a genuine solution of `|∇T| = s` that is not the
viscosity solution.

**This is a formulation result, not an optimizer result.** `|∇T| = s` on a compact manifold has
infinitely many Lipschitz solutions and the pointwise squared residual cannot distinguish them.
Worse, it actively *prefers* the wrong ones: the true field is singular at the source and kinked at
the cut locus, so any smooth approximation to it carries a large residual there, while a sawtooth is
smooth almost everywhere. Capacity, learning rate and optimizer are all irrelevant to this.

This reproduces, on the obstacle-free case, what `results/lm_optimizer_note.md` found with LM and
continuation (cost 401.8 at the best-RMS checkpoint vs 59.6 at the drifted endpoint). That result was
open to a "shadow under-determination" reading; **with zero obstacles there are no shadows**, so the
cause is the objective itself.

Consequence for the experiment ladder: Exp 1 cannot be run with the unfactored field as it stands.
It also cannot be run with the factored fields (`T = base/τ`, `T = base·exp(g)`) as a *test*, because
with no obstacles those are solved by their own initialisation — `τ ≡ 1` and `g ≡ 0` give the exact
answer before a single step. The formulation has to supply what the pointwise residual does not.
Options and the recommendation are in `results/training_strategy.md`.

### Cull

Root scripts of the retired `torus.py` lineage → `_archive/torus_lineage/` (21 files, all unimported
by `srms/`). `srms/lib/{nets,cgls_reference_solver,test_identification,splat}.py` → `_archive/
lib_preexisting/` (no importers; `test_identification.py` imported a `v2.lib` package that does not
exist). `manifold_splat.py` trimmed 358 → 36 lines: only `eval_wrapped_gaussian` had a caller, the
S²/SE(2) primitives being superseded by `srms/environments`. Empty `backends/kan.py` removed.
Duplicated `sdf`-union and `slowness`-ramp code across five environments replaced by
`environments/base.py`'s `union_sdf` / `smooth_slowness`.

### Does the obstacle-free field need learning at all? No.

With no obstacles the answer is the geodesic distance from the source, which is closed form
(`env.geodesic`). Nothing needs to be learned. The only open question is whether a splat mixture can
*hold* that field without gradient descent.

It can. Centres sampled from the manifold, scales fixed isotropic, weights `V` by one ridge
least-squares solve — no optimizer, no PDE, no schedule. **Scored against the analytic field, which
is the true answer here.** 2-D torus:

| splats | params | RMS vs analytic |
|---|---|---|
| 256 | 1,792 | 0.1054 |
| 1024 | 7,168 | 0.0094 |
| 4096 | 28,672 | 0.0085 |

So the representation is not in question, and neither is "learning" in the gradient-descent sense.
What the linear solve needs is a *known target*, which is exactly what obstacles remove.

#### Correction — an earlier version of this section scored against fast marching, and was wrong

The first draft reported these numbers against `env.ground_truth` (fast marching) and claimed the fit
was "5x below the grid it is scored on". That statement is incoherent and the numbers were measuring
the wrong thing. The basis was **fitted to** the FMM field and then **scored against** the FMM field,
so a low number means it reproduced FMM, discretisation error included. You cannot score below your
own reference. Measured, 1024 splats on the obstacle-free torus:

| fitted to | RMS vs FMM | RMS vs analytic |
|---|---|---|
| FMM | 0.0094 | **0.0472** |
| analytic | 0.0473 | **0.0094** |

FMM's own distance from the analytic answer is 0.0463. The fit-to-FMM row reproduced that almost
exactly: it learned the solver's error. `manifolds.md` §2 already stated the rule ("no solver RMS
below these numbers is interpretable"); this section had violated it.

**Rule, now enforced in code.** `experiments/sweep.py` scores obstacle-free runs against
`env.geodesic` and only falls back to fast marching when obstacles make the closed form unavailable;
`experiments/sweep.py` documents why. This also bounds what the existing headline numbers can mean:
at resolution 120 the reference carries ~0.046 error, so a result of 0.24 is safely above it and a
result of 0.04 is not — the weak-supervision 0.0417 in `manifolds.md` §5 is at the reference's own
noise floor and should not be quoted as a precise value without re-scoring at higher resolution.

### What the MLP comparison did and did not show

An earlier version of this section asserted that "an MLP cannot represent the field" is not
defensible. That over-reached, because the experiment behind it does not test that claim.

What was actually run: **supervised regression onto ground-truth labels**, 4000 steps of Adam. A
SIREN given ambient `(x, y, z)` reached 0.0074 on the sphere; the same network given the repo's
`sin/cos` features reached 0.2500. So *label-supervised interpolation of a field you already have*
works for an MLP given sensible input coordinates. That is curve fitting, not modelling a manifold
and not predicting time-to-go, and it says nothing about whether an MLP can learn the field
**without labels** from the physics loss — which is the setting this project is actually in, and
which was not tested.

The one measurement here that stands on its own is the encoding defect, which is independent of the
comparison: `mlp.py`'s Fourier features place the two poles of S² **1.75e-07** apart when they are pi
apart on the manifold, so with that encoding the network cannot separate them. That reproduces
`manifolds.md`'s finding and is a property of the encoding, not of MLPs.

Note also that the torus MLP number from that run (0.0250 vs FMM) is below FMM's own 0.0463 error and
is therefore uninterpretable for the same reason as above.

This also settles what Exp 1 is for. It cannot be a test of the representation, because the
representation trivially passes. It is a test of the **loss** — the only part that would still be
needed when obstacles make the closed form unavailable. Run that way it fails (see the gate above).

### RETRACTION — "the objective prefers a wrong field" was an artifact of a supervised fit

The gate above compared the objective at the trained field against the objective at **a splat basis
least-squares fitted to the truth**, and concluded from `0.010 < 9.9` that the objective's minimum is
not the answer, i.e. a formulation failure no optimizer could fix. **That conclusion is withdrawn.**

The supervised fit was the problem. A 512-splat basis fitted to the geodesic cone still has
appreciable gradient error near the source and the cut locus, and the Eikonal residual reads that
error, so the fitted field carried a large cost that had nothing to do with the objective. Removing
the fit and substituting the **analytic** field directly into the loss — no model, no parameters,
nothing fitted — gives the correct comparison:

| seed | trained RMS | do nothing | cost@trained | cost@truth |
|---|---|---|---|---|
| 1 | 2.6890 | 0.0000 | 0.01125 | **0.00000** |
| 2 | 2.6857 | 0.0000 | 0.00943 | **0.00000** |
| 3 | 2.6835 | 0.0000 | 0.01055 | **0.00000** |

`cost@truth` is exactly zero, as it must be: with no obstacles `‖∇d‖_g = 1 = s` everywhere and the
boundary ring value is `eps·s(start) = eps`. Both terms vanish at the answer.

**Corrected diagnosis.** The objective is correct — its global minimum *is* the true field. The
solver converges to a local minimum at cost 0.011: a small-amplitude sawtooth with `|∇T|` median
1.007 and correlation 0.053 with the answer. So this is a **non-convex optimisation failure**, not an
under-determined objective, and the levers are the ones that escape bad minima — continuation on the
slowness or on a viscosity parameter, a stronger or longer-held causal curriculum, and the boundary
term's weight relative to the residual (currently 1:1, with 32 ring points against 2048 collocation
points, so the only term that pins the solution branch is outvoted ~64:1).

**Two earlier notes carried the same contamination and are deleted** (`results/lm_optimizer_note.md`,
`results/staged_strategy_note.md`). Both concluded "the objective genuinely prefers a wrong field"
from a comparison against a ground-truth-fitted splat field (`V_true` in `lm_staged_experiment.py`) —
the identical error. Their one surviving claim, unaffected by it: on a *frozen* basis (`A`, `B` fixed)
the field is linear in `V`, so the residual Jacobian is closed form and Levenberg-Marquardt applies,
which Adam's lack of a line search does not allow. `lm_staged_experiment.py` still implements it; its
reported numbers should not be reused.

**Rule, now stated in `experiments/sweep.py`: no supervised fit anywhere in this ladder, not even as a
diagnostic.** Every experiment is self-supervised; a fit-to-truth measures representation capacity,
which is not a question any of these experiments ask, and it silently corrupted the one comparison it
was introduced to support. `srms/experiments/ceiling.py` deleted.

### NTFields formulation on the obstacle-free torus: the free-space answer is not representable

Ran the NTFields loss verbatim (`isotropic_loss(q) = |1−√q| + |1−1/√q|`, `q = ‖∇T‖_g/s`,
`T = base/τ`, `τ = σ(SRM+4)`) on T² with zero obstacles, scored against the analytic geodesic.
The `init` column is the field *before training*:

| seed | init | trained | cost@init | cost@trained | cost@truth |
|---|---|---|---|---|---|
| 1 | 0.0473 | 0.0298 | 0.01854 | 0.00040 | 0.00000 |
| 2 | 0.0470 | 0.0294 | 0.01818 | 0.00041 | 0.00000 |
| 3 | 0.0464 | 0.0306 | 0.01765 | 0.00039 | 0.00000 |

Training does improve the field (0.047 → 0.030) and the loss drops 45x, so nothing is diverging. But
it stalls, and the reason is structural, not optimisation:

- `τ = σ(raw + tau_bias)` lies in the **open** interval (0, 1). At init `σ(4) = 0.98201`, so
  `T = 1.01832·base` — a uniform 1.83% overestimate.
- Measured error by distance from the source: **1.83% at every band** (0-0.5, 0.5-1.5, 1.5-2.5,
  2.5-3.5, 3.5-5.0). It is a pure scale factor, not a near-source or far-field defect.
- At the source, `T = 0` exactly (`base(start) = 0`, correct by construction), but `‖∇T‖ = 1.01832`
  where it must be 1.
- With no obstacles the true field is exactly `base`, which needs `τ ≡ 1`, which needs `raw → +∞`.
  **The formulation cannot represent the answer**; it can only approach it, and the optimizer is
  pushing `raw` up against a sigmoid's tail.

`cost@truth = 0` confirms the loss is right about the target: `q = ‖∇d‖_g/s = 1` gives
`|1−1| + |1−1| = 0`. So the loss is correct and the *field parameterisation* is what bars the answer.

**Candidate fix (not implemented — for review).** Replace the sigmoid with a form in which the
free-space answer is exactly attainable, e.g. `T = base·(1 + relu(g))`:

- `T(start) = 0` still holds by construction;
- `T ≥ base` is enforced, which is correct whenever `slowness ≥ 1` (travel time can never beat the
  free-space geodesic);
- `g ≤ 0` gives `T = base` **exactly**, so free space is representable rather than asymptotic;
- `‖∇T‖ = 1` exactly wherever the field is unobstructed.

Raising `tau_bias` (`σ(8) = 0.99966`, 0.03% error) hides the symptom without removing the barrier.

### Experiment 1 PASSES with `T = base·exp(g)` — the sigmoid was the barrier

Symbols: `x` a point on the manifold, `T(x)` the time-to-go, `base(x)` the analytic geodesic distance
from the source (closed form), `g(x)` the splat mixture's scalar output, `s(x)` the slowness.

Replacing NTFields' `T = base/tau`, `tau = sigmoid(g + 4)` with `T = base·exp(g)`, same Eikonal
residual, same optimizer, same budget, obstacle-free torus, scored against the analytic answer:

| field | init | trained | recovery |
|---|---|---|---|
| `T = base/tau` (NTFields) | 0.0473 | 0.0298 | 1.6x |
| **`T = base·exp(g)`** | 0.0836 | **0.0015** | **56x** |

0.0015 is 30x below fast marching's own error at this resolution (0.046), and the training loss falls
to ~1e-5. **The self-supervised Eikonal loss recovers the time-to-go field on the torus.** Note the
initialisation is a *random perturbation* (`init_weight = 1e-2`, so `g` starts small and random, not
zero) — at `g = 0` the field would be exactly `base` and there would be nothing to recover. So this is
a genuine recovery from a perturbed start, not a fixed point being held.

**Why `exp` and not `1 + relu(g)`**, which was the first suggestion and is worse: both make `T = base`
exactly attainable, and `relu` would additionally enforce `T >= base` (correct whenever `s >= 1`,
since travel cannot beat the free-space geodesic). But `relu` is flat for `g < 0`, so a correction
that goes negative has zero gradient and can never come back — the same dead-gradient failure that
made the unfactored field untrainable from `V = 0`. `exp` is smooth everywhere, has derivative 1 at
`g = 0` so it is well conditioned exactly at the free-space solution, and matches the `base ·
exp(correction)` already used in `weak_supervision.py`. The `T >= base` constraint is left to the
physics rather than hardcoded.

Implemented as `srms/methods/strategies/factored.py` (`--method factored`). It uses `(q − 1)²` as the
residual rather than NTFields' `|1−√q| + |1−1/√q|`, which is non-differentiable at exactly `q = 1` —
the point every run is trying to reach. `ntfields.py` is untouched and remains the paper baseline.
No boundary-ring term is needed: `T(start) = 0` holds by construction for any finite `g`.

Next: the same gate on the sphere and hyperbolic, then Exp 2 (one obstacle).

### Two cut-locus defects found by running the gate on curved manifolds

Extending Experiment 1 from the torus to the sphere, hyperbolic and SO(3) exposed two independent
bugs, both at the **cut locus** (the far side of the manifold from a reference point), and both
invisible on the flat torus.

**Defect 8 — `base` must never be differentiated.** `T = base·exp(g)` was differentiated whole, so
autodiff hit `base` itself. On S², `base = 2·arcsin(‖x−start‖/2)` has an unbounded derivative as the
chord approaches 2. Measured ambient `|∇base|`: 1.1 at distance 1.0, 48 at 3.10, 1.3e3 at 3.14, NaN
at 3.1415. Uniform collocation reaches there routinely and sphere training NaN'd within 60 steps.

Fixed by giving every environment a `grad_geodesic(x)` in closed form: `∇base` is the metric-unit
covector `-log_map_ambient(x, start)`, normalised. `time_grad` then differentiates only `g`, which is
a smooth Gaussian mixture. Verified equivalent to autodiff where autodiff is defined (max relative
difference 2.7e-07 on the torus, 2.3e-04 on the sphere under the metric — the raw vectors differ only
by a radial component that `metric_inv` annihilates) and finite where it is not.

Note the first attempt at this was wrong in a way the torus hid: `∇base` was written as the tangent
**at `start`** pointing toward `x`, when it is the tangent **at `x`** pointing away from `start`. On a
flat chart those coincide, so it passed on the torus and would have been silently wrong on every
curved manifold.

**Defect 9 — the wrapped Gaussian evaluated an antipodal point as if it sat at its own centre.**
In `sphere._theta_perp`, `perp = x − cos(theta)·mu` vanishes exactly at the antipode of the splat
centre, and `perp / max(‖perp‖, 1e-8)` then returned the **zero vector** rather than a unit one. With
`e_perp = 0` the log map is 0, so the density was evaluated at the centre and multiplied by a
`jac_factor` of pi/1e-8: measured **2.04e8** where the true value is ~1e-18. Separately, `arcsin`
reaches argument 1 there, where its derivative is infinite, so `d/dB` was NaN.

Fixed with an explicit unit tangent where the direction is genuinely undefined (any is correct — the
Gaussian factor is e^-40 regardless) and by holding the `arcsin` argument off its endpoint. Antipodal
density is now **6.85e-15** with all gradients finite.

**This defect affects every strategy on the sphere, not only this one** — including the `ntfields`
sphere result of 0.0545 recorded in `results/manifolds.md` §5, which was produced with splats
returning 1e8 densities at their antipodes. It did not NaN there because `tau = sigmoid(.)` bounds the
field, but that number should be treated as suspect and re-run.

**Why the test suite missed both:** `sample_pairs` bounds separation at 1.8 rad and
`points_near_source` at 1.8, so no existing check ever evaluated anything near pi. Added
`‖∇base‖_g = 1 (cut locus)`, which samples the far side *empirically* — sorting a large pool by
`env.geodesic` — rather than constructing an antipode analytically. That construction is not
manifold-agnostic and my first version got it wrong on two manifolds: `-start` is the **same**
rotation in SO(3), which is S³ quotiented by ±1, and lands on the **wrong sheet** of the hyperboloid
in the Lorentz model. Suite now stands at **30 exact identities, 0 failures**.

### Experiment 1, all four manifolds, one seed (5 seeds running)

Self-supervised, no obstacles, `T = base·exp(g)`, 512 splats, 1500 steps, scored against the analytic
geodesic. `init` is a randomly perturbed start (`init_weight = 1e-2`), not the answer.

| manifold | curvature | init RMS | RMS | max | MAE | MAE far | recovery |
|---|---|---|---|---|---|---|---|
| torus T² | 0 | 0.0836 | 0.0046 | 0.0164 | 0.0035 | 0.0042 | 18x |
| sphere S² | +1 | 0.0680 | **0.0004** | 0.0007 | 0.0003 | 0.0004 | **183x** |
| Poincaré H² | −1 | 0.0383 | 0.0013 | 0.0078 | 0.0008 | 0.0011 | 31x |
| SO(3) | ¼ | 0.0354 | 0.0009 | 0.0038 | 0.0006 | 0.0009 | 41x |

`MAE far` (cells past half the maximum travel time) tracks `MAE` on every manifold, so the error is
uniform rather than accumulating outward from the source. All four sit well below fast marching's own
discretisation error at these resolutions, which is why the analytic reference is the one used.

### Where curvature actually lives in this code

Recorded because it was asked and the answers are not symmetric. Only two functions carry curvature —
`metric_inv` (how gradients are measured) and `jac_factor` (the wrapped Gaussian's volume correction).
Both are the same Jacobi-field expression at that manifold's `K`: `1` at `K=0`, `(r/sin r)^(d-1)` at
`K=+1`, `(r/sinh r)^(d-1)` at `K=-1`, `((r/2)/sin(r/2))²` at `K=+1/4`.

**T² is flat because it is a quotient of the plane by translations**, which are isometries — so every
point has a neighbourhood isometric to a Euclidean disc and `K = 0` exactly. The doughnut picture
misleads: that surface has varying curvature because its embedding in R³ induces a *different* metric,
and the flat torus has no smooth isometric embedding in R³ (only in R⁴, as two orthogonal circles).
Gauss-Bonnet separates topology from curvature cleanly: `∫K dA = 2·pi·chi` with `chi = 0` for a torus,
so any metric on it averages to zero; flat attains it pointwise. In the code this is literal —
`metric_inv = I`, `jac_factor = 1`, `log_map = wrap(x - mu)`.

**SO(3)'s `K = 1/4` is convention-dependent.** For a bi-invariant metric on a compact group,
`K(X,Y) = ¼‖[X,Y]‖²` on orthonormal `X, Y`; in `so(3)` the bracket is the cross product so
`‖[X,Y]‖ = 1` and `K = ¼` on every plane. The number tracks scale: SO(3) = S³/±1 and we take distance
to be the rotation angle, twice the unit-S³ great-circle distance, and doubling distance quarters
curvature from S³'s `+1`. `so3.jac_factor = ((theta/2)/sin(theta/2))²` and the marcher's radial profile
`2·sin(r/2)` both encode exactly that `K`. Note SO(3) is atypical: `K = 0` wherever `[X,Y] = 0`, so any
group of rank >= 2 has flat 2-planes and non-constant curvature; SO(3) has rank 1 and so has neither.

### Experiment 1 closed; the ladder collapsed into one runner

Experiments 1/2/3 differ **only** in `--num-obstacles` (0 / 1 / many) — same field, loss, optimiser
and budget — so `gate.py` and a half-written `obstacles.py` were merged into a single
`srms/experiments/sweep.py`. The one thing that cannot be shared is the **scoring reference**, and it
is a branch, not a second experiment:

- 0 obstacles: the analytic geodesic. Exact, so every digit is real, and `cost@truth` (the objective
  evaluated at the true answer) is available — the column that separates a wrong objective from an
  unconverged solver.
- 1+ obstacles: fast marching, which carries its own error. That error is measured per scene by
  removing the obstacles, where the exact answer *is* known, printed as `ref err`, and the runner
  warns when a result falls below it. `cost@truth` prints `—` rather than substituting a fitted
  field, which is the error retracted earlier in this log.

Two columns carry across the whole ladder: `MAE far` (cells past half the maximum travel time) and
`MAE shadow` (cells whose straight source-to-point ray crosses an obstacle, from the scene's SDF
alone — no solver). The merged runner reproduces the torus Exp 1 number exactly (0.0021 ± 0.0015 over
5 seeds), confirming the merge is faithful.

**Experiment 1 result of record — one seed per manifold, self-supervised, scored against the analytic
field.** Multi-seed was dropped as unnecessary once every manifold passed:

| manifold | K | init RMS | RMS | max | MAE | MAE far |
|---|---|---|---|---|---|---|
| torus T² | 0 | 0.0836 | 0.0046 | 0.0164 | 0.0035 | 0.0042 |
| sphere S² | +1 | 0.0680 | 0.0004 | 0.0007 | 0.0003 | 0.0004 |
| Poincaré H² | −1 | 0.0383 | 0.0013 | 0.0078 | 0.0008 | 0.0011 |
| SO(3) | +1/4 | 0.0354 | 0.0009 | 0.0038 | 0.0006 | 0.0009 |

For reference, the 5-seed means where they were collected: torus 0.0021 ± 0.0015, sphere
0.0030 ± 0.0034 — so single-seed numbers sit inside seed noise and no ordering between manifolds
should be read from them.

### The basin of attraction is finite, and it is what Experiment 2 tests

Exp 1 asks the physics to drive the correction `g` back to **zero**, since with no obstacles
`T = base` is the answer. That is a stability result, not a discovery result. Measured on the torus by
enlarging the initial perturbation (`init_weight`):

| `init_weight` | init RMS | trained RMS | outcome |
|---|---|---|---|
| 0.01 | 0.084 | 0.0046 | recovers |
| 0.05 | 0.410 | 0.0087 | recovers, 47x |
| 0.2 | 1.892 | 1.576 | **fails — barely moves** |

So the loss has a finite basin: it undoes an error of 0.41 completely and an error of 1.89 not at all.
This is continuous with the unfactored `T = g(x)` failure, which is the same thing with no `base` at
all. **The Exp 2 question is therefore not whether the splats can represent the shadow, but whether
the correction the shadow demands lands inside that basin** — a measurable quantity, not a worry.

### Directional stretching enabled (`--max-aspect`)

The uniform per-axis scale floor forbids the useful anisotropy: at an obstacle's shadow crease the
field is smooth along the crease and kinked across it, so a splat wants to be long one way and thin
the other. Measured after 800 steps on the torus:

| scene | median aspect | p95 | max | splats pinned at the floor |
|---|---|---|---|---|
| no obstacles | 1.09 | 1.25 | 1.58 | 0.0% |
| one obstacle | 1.17 | 2.37 | 8.28 | **1.2%** |

So splats stay isotropic when there is nothing to conform to and stretch as soon as an obstacle
exists, with a fraction pinned. `post_step` now bounds the **condition number** instead of every axis:
the effective lower bound is `max(scale_floor, s_max/max_aspect)`, so a splat may thin across a crease
while `scale_floor` still prevents collapse to a point in every direction — the failure the floor was
introduced for. Defaults to 0 (disabled), reproducing the old behaviour exactly. **Not yet validated:**
whether loosening the floor reintroduces the divergence it was added to prevent. Check that on the
obstacle-free case before trusting it in Exp 2, where it would be confounded with the shadow.

### RETRACTED IN ADVANCE: "Exp 2 reproduces the historical 0.31 plateau"

The first Experiment 2 torus number (RMS 0.3040, one obstacle) was reported as reproducing this log's
historical pure-physics plateau of ~0.31 at *three* obstacles, and therefore as showing that level
under-determination starts at a single obstacle. **That comparison is confounded four ways and the
claim is withdrawn pending a matched run.**

| | historical (0.2425) | first Exp 2 run (0.3040) |
|---|---|---|
| field / loss | `ntfields`, `base/tau`, isotropic | `factored`, `base·exp(g)`, `(q−1)²` |
| capacity | **densify on, grew to 1947 splats** | **densify off, 512 fixed** |
| steps | 4000 | 3000 |
| causal weighting | ignored by `ntfields` | **on** |

The likely dominant term is densification. The correction `g` is *spatially confined* — with
`T = base·exp(g)` and the free-space answer at `g = 0`, only splats in and around the shadow need to
move at all. Densification spawns capacity at the highest-residual points, i.e. exactly there;
without it, 512 splats drawn from `sample_domain` are spread uniformly over the whole manifold and
nothing concentrates them where the work is. `SweepConfig` had silently flipped the repo's
`densify=True` default to `False`. Restored.

Two further claims from that report are also premature and withdrawn: that directional stretching
cannot help, and that the finite-basin measurement explains the failure. Both were derived at 512
*fixed* splats, and spawning splats at high-residual points is itself a basin-escape mechanism.

Matched arms now running on the same seed-1 one-obstacle scene, resolution 240, 4000 steps,
densification on: (a) `ntfields` as the historical control, (b) `factored` with causal weighting,
(c) `factored` without, (d) `factored` with causal weighting but no annealing. Nothing about level
under-determination at one obstacle should be claimed until those land.

### Defect 10 — `shadow_mask` was correct on exactly one manifold

Found by review, before it reached a table. The ray was traced as
`start + f·log_map_ambient(start, x)` then `wrap_point`, which is a flat-chart shortcut:

- **torus** — correct.
- **SO(3)** — **crash**: `log_map_ambient` returns the 3-component Lie-algebra element while `start`
  is a 4-vector quaternion. The sweep would have trained SO(3) fully and then died at scoring.
- **sphere** — silently wrong: the renormalised chord reaches angle `arctan(theta)` rather than
  `theta` at `f=1`, so the far part of the path is never tested and shadows there are missed.
- **Poincaré** — overshoots: the tangent's norm is a geodesic length exceeding the chart radius, so
  `f=1` lands past `x`, often outside the ball, producing false shadows.

Fixed with `Exp_start(f · Log_start(x))`, which is the geodesic on every manifold with correct
endpoints, and which rests on the round-trip identity `test_manifolds.py` already verifies.

**Process change: `srms/experiments/preflight.py`.** Every defect this session was found by an
expensive run failing, and each had the same shape — right on the flat torus, wrong on a curved
manifold, or right without obstacles and wrong with them. The pre-flight runs every manifold x
{0, 1} obstacles for 10 steps through the *full* scoring, shadow-mask and figure path in about two
minutes. It reports 8/8 passing now; it would have caught the SO(3) crash and the sphere ray. Run it
before launching any sweep.

### Experiment 2, matched arms: the correction must be non-negative, and that was the bug

One obstacle, torus, seed 1, resolution 240, 4000 steps, densification on, identical scene and
budget. Fast marching's own error here is 0.0273, so every number below is well clear of the floor.

| arm | field | constraint on `g` | RMS | MAE shadow / MAE |
|---|---|---|---|---|
| (a) `ntfields` | `base/tau`, `tau = sigmoid` | `T > base` structurally | 0.2027 | 6.86x |
| (b) `factored` causal ON | `base·exp(g)` | none | 0.2737 | 4.88x |
| (c) `factored` causal OFF | `base·exp(g)` | none | 0.2610 | 4.16x |
| (d) `factored` causal, no anneal | `base·exp(g)` | none | 0.2698 | 4.93x |
| **(e) `factored` + `g >= 0`** | `base·exp(g)` | penalty | **0.1934** | **3.72x** |

**Read (a) correctly.** This is *not* NTFields the paper, which is a Euclidean MLP over a workspace
encoder. It is their **field parameterisation and loss** carried by *our* splat backend on *our*
manifold machinery. So the comparison isolates the parameterisation, and nothing here says anything
about MLP versus splat.

**The finding.** With `s >= 1`, travel time can never beat the free-space geodesic, so `T >= base` and
`g = log(T/base) >= 0`. Measured against the true field: `g_true` spans `[0, 0.763]` across 55,675
cells with **no negative value**. `base·exp(g)` does not encode this, and the optimizer exploits it —
after training, `g < 0` in **44.5%** of free-space cells and **30.7%** of shadow cells, averaging
**0.277 below truth** in the shadow. That unphysical region *is* the large blue under-prediction area
in the error maps.

NTFields' sigmoid forbids it structurally (`tau` in (0,1) implies `T > base`), which is why (a) beat
(b) with an obstacle while losing badly without one (0.0298 vs 0.0015, where the answer is `T = base`
and a sigmoid cannot reach `tau = 1`). Each parameterisation held one half of what is needed.

Adding a one-sided penalty `mean(min(g, 0)^2)` — exactly zero, with zero gradient, wherever the
constraint holds, so the obstacle-free case is untouched by construction — takes `base·exp(g)` from
0.2737 to **0.1934**, and the two formulations then agree to within 5% (0.1934 vs 0.2027). That is
the predicted outcome: once both respect `g >= 0` they represent the same function class.

**No smooth `psi(g)` can do both structurally.** Attaining `psi = 1` while staying `>= 1` makes that
point a minimum, so `psi' = 0` there — a dead gradient exactly at the free-space solution, which is
the defect that made the unfactored field untrainable from `V = 0`. Hence `exp` for conditioning plus
an exterior penalty for the constraint, which is the standard treatment of an inequality constraint
rather than a workaround.

**Causal weighting is mildly harmful here** (b 0.2737 vs c 0.2610), and un-annealing it does not help
(d 0.2698). Suspected cause, untested: `training_aids.causal_loss` orders points by `env.geodesic`,
the **free-space** distance, which under-estimates arrival time behind an obstacle — so shadow points
are un-muted before the wavefront has actually reached them, inverting the curriculum exactly where
it is supposed to help.

### CORRECTION — "the sigmoid cannot represent free space" was overstated

This log claimed that NTFields' `T = base/tau`, `tau = sigmoid(g + tau_bias)`, **cannot** represent
the obstacle-free answer, on the basis that `tau = 1` needs `g -> +inf`, and pointed at a measured
1.83% uniform overestimate and a stall at RMS 0.0298. The mathematics is right and the practical
conclusion was wrong: 1.83% is `sigmoid(4) = 0.98201`, i.e. **the default `tau_bias = 4`**, not the
parameterisation. Re-run at `tau_bias = 8` (`sigmoid(8) = 0.99966`), same scene and budget:

| arm | tau_bias | Exp 1 RMS (no obstacles) | Exp 2 RMS (one obstacle) |
|---|---|---|---|
| `ntfields` | 4 | 0.0298 | **0.2027** |
| `ntfields` | 8 | **0.0006** | 0.2245 |
| `factored` (`base·exp(g)`) | — | 0.0015 | 0.1934 (with `g >= 0`) |

At `tau_bias = 8` NTFields reaches **0.0006** on the obstacle-free case — better than `base·exp(g)`.
So free space is attainable to any precision that matters; the barrier was a bad default.

**The real structural difference is a tension, not an impossibility, and it now has both ends
measured.** Raising the bias fixes free space and *costs* accuracy with an obstacle (0.2027 -> 0.2245),
because `sigmoid'(8) = 3.4e-4` — the field becomes slow to move away from `tau = 1` exactly when an
obstacle demands that it does. Low bias represents obstacles well and free space badly; high bias the
reverse. `base·exp(g)` has derivative 1 at `g = 0` regardless, so it has no such trade-off, which is a
narrower and defensible claim than the one being corrected.

### Best Experiment 2 arm so far

| arm | field | constraint | causal | RMS | MAE shadow / MAE |
|---|---|---|---|---|---|
| (a) `ntfields` bias 4 | `base/tau` | structural | off | 0.2027 | 6.86x |
| (h) `ntfields` bias 8 | `base/tau` | structural | off | 0.2245 | 7.03x |
| (b) `factored` | `base·exp(g)` | none | on | 0.2737 | 4.88x |
| (c) `factored` | `base·exp(g)` | none | off | 0.2610 | 4.16x |
| (e) `factored` | `base·exp(g)` | penalty w=1 | on | 0.1934 | 3.72x |
| (f) `factored` | `base·exp(g)` | penalty w=10 | on | 0.1985 | **2.72x** |
| **(g) `factored`** | `base·exp(g)` | penalty w=1 | **off** | **0.1865** | 3.30x |

Fast marching's own error is 0.0273, so all of these are well clear of the floor. Best is `g >= 0`
enforced with causal weighting **off** — 0.1865, a 32% improvement on the unconstrained arm and 8%
better than the NTFields formulation at its best bias. Note the two levers are close to independent:
the constraint buys ~29% and dropping causal weighting a further ~4%.

### Why the sphere is easy and hyperbolic will be hard: curvature controls shadow size

Asked why S² scored so much better than T² at one obstacle. Measured, **matched obstacle radius
(0.5–0.9), seed 1, resolution 240**, so the scenes are comparable:

| manifold | K | obstacle % of volume | shadow % | median detour | max detour | do-nothing RMS |
|---|---|---|---|---|---|---|
| sphere S² | +1 | **12.6%** | **0.7%** | 1.0000 | 1.359 | 0.148 |
| torus T² | 0 | 3.3% | 5.8% | 1.0146 | 2.145 | 0.320 |
| Poincaré H² | −1 | 26.8%* | 6.5% | 1.0599 | 3.549 | 0.653 |

*includes out-of-chart cells, which the rim marks as blocked.

**On the sphere a ~4x larger obstacle casts an ~8x smaller shadow.** That is geodesic focusing: at
K=+1 geodesics leaving the source reconverge toward the antipode, so paths bending around a cap
rejoin almost immediately and the region genuinely requiring a detour is tiny. At K=0 they stay
parallel, so the shadow is an open corridor extending far behind the obstacle and fed from only two
sides. At K=−1 they diverge exponentially, which is why H² has both the largest shadow and by far the
largest detours — the ordering sphere < torus < hyperbolic follows curvature +1, 0, −1 directly, and
`do-nothing` RMS (0.148 / 0.320 / 0.653) tracks it.

So **a low absolute RMS on the sphere is mostly the scene, not the method.** Report improvement over
`do nothing`, never the raw number alone:

| manifold | do-nothing | best RMS | improvement | FMM's own error |
|---|---|---|---|---|
| sphere | 0.1256 | 0.0071 | 17.7x | 0.0218 |
| torus | 0.3197 | 0.1865 | 1.71x | 0.0281 |

### CORRECTION — "below the floor" does not mean "better than fast marching"

Twice in this session a result under the reference's own error was described as the field being "at
least as accurate as the reference". **That is wrong and the phrasing is withdrawn.** The field is
scored *against* fast marching, so the column is **agreement with FMM**, never accuracy. Agreeing
with FMM to 0.0071 while FMM is 0.0218 from truth puts the true error in [0.015, 0.029] by the
triangle inequality — unresolvable, and dominated by the reference's error rather than the model's.
Nothing scored this way can ever be shown to beat the solver it is scored against. `sweep.py` now
prints this explicitly instead of the word "floor".

The reference-error measurement was also wrong for S². Taken at the configured start it reported
**0.0000**, because the grid is geodesic-polar about the pole and the marcher was discretising a 1-D
radial problem exactly; 45° off the pole the same grid gives 0.0134, at the equator 0.0166. It is now
measured at three *generic* source positions and the worst taken: torus 0.0281, sphere 0.0218,
Poincaré 0.0508.

### L1 on the weights was not the lever

Structural `V >= 0`, torus, one obstacle: no L1 0.2259, L1 1e-4 0.2223, L1 1e-3 0.2193. The
free-space haze is not caused by too *many* active splats but by each splat reaching too far, which
is what motivated compact support (user's suggestion — kill the Gaussian past ~1% of peak).

### Compact support works, and reveals that ~0.19 on the torus is not a representation limit

User's suggestion: kill each Gaussian past ~1% of its peak so it cannot influence the field globally.
Implemented as a quintic-smoothstep window on the Mahalanobis radius (`cfg.trunc_sigma`, C² so the
Eikonal residual sees no kink; support is an ellipsoid in whitened space, so it composes with
`max_aspect`). Verified: a σ=0.35 splat is unchanged at 2σ and **exactly 0** past 3σ.

Torus, one obstacle, seed 1, res 240, 4000 steps, structural `V >= 0`, causal off:

| arm | RMS | max | MAE free space | MAE shadow |
|---|---|---|---|---|
| clamp only (control) | 0.2259 | **0.630** | 0.1810 | **0.2319** |
| + compact 3σ | 0.1958 | 0.727 | 0.1560 | 0.3001 |
| + compact 3σ + narrow spawns (0.18) | 0.1892 | 0.954 | 0.1451 | 0.3173 |
| + compact 2σ | **0.1844** | 1.269 | **0.0935** | 0.5383 |
| soft penalty instead of clamp | 0.1865 | 1.335 | 0.1177 | 0.3889 |

**The idea works for what it targets** — free-space error nearly halves (0.181 → 0.094) as the
splats become local. **But total RMS is pinned at ~0.19 across every configuration tried**: soft
penalty 0.1865, clamp + 2σ 0.1844, clamp + 3σ + narrow spawns 0.1892. Each knob trades free-space
error against shadow error and leaves the sum unchanged.

That plateau is the informative part, and it has a mechanical explanation. A compactly-supported
splat receives gradient only from collocation points *inside its own radius*, so it cannot carry
level information into the shadow. The shadow's level is not a local quantity — it is the accumulated
cost of a detour. Making splats more local fixes the leak and degrades that transport, which is
exactly the observed trade (2σ gives the cleanest free space and the worst shadow).

**Conclusion: the residual ~0.19 is not a representation limit.** The obstacle-free ceiling is ~0.003
and free space is already at 0.094; what remains is the shadow *level*, which pointwise physics
under-determines. Clamping, truncation and sparsity all act on representation and none of them move
it. The mechanism that can move it is the training *ordering*.

### The causal curriculum was ordering by the wrong quantity

`training_aids.causal_loss` ordered collocation points by `env.geodesic` — the **free-space**
distance. Behind an obstacle that is exactly backwards: a shadow point is *near* in free-space
distance but *late* in arrival time, so it was un-muted long before the wavefront reached it,
inverting the curriculum precisely where it was supposed to help. That explains the measured result
that causal weighting made things *worse* (0.2737 with, 0.2610 without).

`causal_loss` now takes an `order_by` key and `factored` passes the model's **own predicted `T`**
(stop-gradient) — self-supervised, since it is the field being learned rather than any reference.
Arms testing it are running.

### Causal ordering by predicted arrival time: right fix, wrong size

Torus, one obstacle, seed 1, res 240, 4000 steps, clamp + compact 3σ, only the ordering varying:

| arm | RMS | max | MAE free | MAE shadow |
|---|---|---|---|---|
| ordered by predicted `T`, annealed | **0.1915** | 0.838 | 0.1520 | **0.2769** |
| ordered by predicted `T`, un-annealed | 0.1975 | 0.817 | 0.1564 | 0.2909 |
| causal off (control) | 0.1958 | **0.727** | 0.1560 | 0.3001 |

The ordering key was genuinely wrong before and fixing it reverses the sign of the effect (causal
weighting used to *hurt*: 0.2737 with, 0.2610 without). Shadow MAE now improves monotonically with
the correct order, 0.3001 → 0.2769. But total RMS moves 2%, inside seed noise. **Not the lever.**

### The ~0.19 plateau, and what it rules out

Six mechanisms, spanning representation *and* optimisation, all land in 0.184–0.226 on the same scene:

| mechanism | kind | RMS |
|---|---|---|
| soft penalty on `g < 0` | objective | 0.1865 |
| structural clamp `V >= 0` | projection | 0.2259 |
| clamp + compact support 2σ | representation | **0.1844** |
| clamp + compact 3σ + narrow spawns | representation | 0.1892 |
| clamp + compact 3σ + correct causal order | curriculum | 0.1915 |
| L1 on weights | sparsity | 0.2193 |

Against `do nothing` = 0.3197 that is a 1.7x improvement, pinned. The representational explanation is
directly falsified: free space reaches MAE 0.094 while the shadow sits at 0.28–0.54, and the
obstacle-free ceiling is ~0.003. **What remains is the shadow *level*, and the pointwise residual
does not determine it.**

**Why no local fix can.** The loss is `(‖∇T‖/s − 1)²` evaluated pointwise. It constrains the
*gradient* of `T` and says nothing about its value. Behind an obstacle the level is the accumulated
cost of a detour — a path integral, not a local quantity — so a field that is off by a constant there
has near-zero residual. Compact support makes this worse by construction, which is exactly the
measured trade (2σ: cleanest free space 0.094, worst shadow 0.538).

### The variational reformulation — the untried lever that targets this directly

The Eikonal equation *does* determine the level uniquely; the pointwise residual is simply the wrong
way to extract it. The value function has a variational characterisation:

    T = max { u : ‖∇u‖ <= s everywhere, u(start) = 0 }

Any `u` with `‖∇u‖ <= s` grows at most at rate `s` along any path, so `u(x) <= ∫ s dl` for every path
and hence `u(x) <= T(x)`. The true field is the **largest subsolution**, i.e. the answer is picked out
by *maximising* `T` subject to a one-sided constraint, not by driving a two-sided residual to zero.

That is exactly the missing mechanism. Under `(q−1)²` a shadow that is too low is barely penalised —
the gradient is right, only the constant is wrong. Under the variational form it is penalised
directly, because the objective pushes `T` up until the constraint binds, and the constraint is what
propagates the detour cost inward. Concretely:

    minimise   −mean(T)  +  λ · mean(relu(‖∇T‖_g / s − 1)²)

one-sided (only *super*-unit gradients are penalised), and non-local in effect despite being computed
pointwise, since the maximisation couples every point through the constraint. Untested; this is the
next thing to run, ahead of viscosity.

### The variational reformulation diverges as written

`minimise −mean(T) + λ·mean(relu(q−1)²)`, torus, one obstacle, clamp + compact 3σ, 4000 steps.
λ = 30: **RMS 16.05**, objective −12.4 (negative — the ascent term dominated).

The characterisation is correct but the *penalised* form is not equivalent to it. Two defects:

1. `−mean(T)` is unbounded below, so the ascent term always has somewhere to go.
2. The constraint is a **mean** of squared violations over 2048 sampled points, so a small region can
   violate badly while the average stays small. The maximal-subsolution theorem needs
   `‖∇u‖ <= s` **everywhere along every path**; a sampled mean does not deliver that. A max-norm
   penalty, an augmented-Lagrangian schedule on λ, or a hard barrier would be the honest
   implementations. Higher λ (100, 300) is running but only shifts the balance point — at the
   equilibrium `k ≈ 1 + c/(2λ)` the inflation shrinks like 1/λ without ever being bounded.

Recorded as a **negative result for this implementation**, not for the idea.

### Weak supervision: 30 sparse shadow-targeted anchors

Decision (user's): stop pushing purely label-free formulations on the torus and add sparse
supervision. The 30-point budget is not arbitrary — it is this repo's own measured result from before
the rewrite (entries W1/W2/U): 30 **shadow-targeted** RRT* anchors took the 3-obstacle torus from
0.307 to **0.177**, while 30 **uniform** anchors reached only 0.229 at the same budget, and weight
**0.5 beat 1.0**. The last point matters: RRT* costs are *upper bounds* (its paths are suboptimal), so
trusting them hard bakes that suboptimality into the field — the B4 lesson, where hard anchors scored
worse than no supervision at all.

Note this is **not** the existing `weak_supervision.py`, which uses ~300 roadmap nodes as a *dense
base* (`T = roadmap_base·exp(g)`, historical B5 = 0.109) and which this log already flags as
effectively cheating, since a dense base is exactly what high dimensions cannot supply. Anchors keep
`T = base·exp(g)` and add `anchor_weight · mean((T(x_i) − c_i)²)` at 30 points. Anchor costs come from
`sampling.build_roadmap`, computed from the known slowness field — self-supervised, never from fast
marching. Selection is biased toward nodes whose straight ray from the source is occluded
(`anchor_shadow_pref`), and a uniform-anchor control is run alongside so the placement effect is
measured rather than assumed.

### Hyperbolic: the success criterion, agreed BEFORE the run

H² is predicted to be the hardest manifold, and a correct result will still look bad in absolute
terms, so the criterion is fixed in advance. Measured scene properties at one obstacle: `do nothing`
0.653 (vs torus 0.320), median detour 1.060 (vs 1.015), max detour 3.55 (vs 2.14), ground truth's own
discretisation error 0.051 (vs 0.028). Geodesics diverge exponentially at K=−1, so shadows are both
larger and deeper — that is physics, not a defect.

- **Working:** improvement over `do nothing` comparable to the torus's ~1.7x, i.e. RMS around 0.38 or
  better, with error concentrated in the shadow.
- **Broken:** NaNs, no improvement over 0.653, or error concentrated *outside* the shadow.

An absolute RMS near the torus's 0.19 is **not** the bar and should not be read as one.

### Experiment 2 clears the target on the torus — after fixing a ported-recipe error

**RMS 0.0735 with 30 sparse anchors**, one obstacle, torus, seed 1, res 240. Against `do nothing`
0.3197 that is 4.4x, and it is below the 0.1 target.

| arm | RMS | max | MAE | MAE shadow |
|---|---|---|---|---|
| label-free best (clamp + compact 2σ) | 0.1844 | 1.269 | 0.0935 | 0.5383 |
| + 30 anchors, `rrt_iters=350` | 0.3393 | 1.074 | 0.2089 | 0.2168 |
| **+ 30 anchors, `rrt_iters=1500`** | **0.0735** | **0.382** | **0.0502** | **0.1077** |

The only difference between the last two rows is the RRT* iteration count. Shadow MAE falls 0.538 →
0.108, a 5x improvement in the region that blocked every label-free mechanism tried this session.

**The error, stated plainly: the historical recipe was ported without its precondition.** The log's
own entry (line 570) records that anchors were *validated to ±0.04 against fast marching* before
being used. That validation was skipped, and the default `rrt_iters=350` does not meet it.

**And "RRT* is inaccurate" was the wrong description** — RMS hid the real shape of the problem:

| iters | p50 | p75 | p90 | p95 | p99 | max | % off by >10% |
|---|---|---|---|---|---|---|---|
| 350 | 1.008 | 1.029 | 1.089 | 1.217 | 1.424 | 1.508 | **9.0%** |
| 1500 | 0.991 | 0.996 | 1.001 | 1.003 | 1.011 | 1.016 | 0.0% |

The median node is within 0.8% even at 350 iterations. The damage is a thin tail — and
`anchor_shadow_pref=3.0` selects *occluded* nodes, which are exactly the ones a sparse tree routes
badly, so the selection rule concentrated on that tail. That is why shadow-targeting came out *worse*
than uniform at 350 iterations, inverting the historical result.

### RRT* verified correct

Checked after the anchor failure, since a broken planner would invalidate everything downstream. All
properties hold at `rrt_iters=1500`: cost >= 0 with source cost 0; cost >= the free-space geodesic
(0/300 violations); detours behind the obstacle captured (max 3.41x); accuracy against ground truth
RMS 0.0287, p99 ratio 1.011.

An intermediate hypothesis — that a fixed 6-sample edge quadrature under-priced hops clipping an
obstacle — was tested and **falsified**: nothing changed after fixing it. The adaptive quadrature was
kept anyway, since 6 samples genuinely under-priced 3.1% of step-length edges by more than 1%, but it
is not credited with any result.

### Hyperbolic H² label-free: passes the pre-registered criterion

**RMS 0.3330** against `do nothing` 0.6527 — a **1.96x** improvement, better in ratio than the torus's
label-free 1.7x, and inside the criterion fixed before the run (~0.38 or better, error in the shadow).
The larger absolute error is the predicted physics of K=−1: max detour 3.55 vs the torus's 2.14, since
geodesics diverge exponentially. The method transfers to negative curvature.

### Experiment 2 status

| manifold | K | do nothing | label-free | + 30 anchors | best improvement |
|---|---|---|---|---|---|
| sphere S² | +1 | 0.1256 | **0.0144** | — | 8.7x |
| torus T² | 0 | 0.3197 | 0.1844 | **0.0735** | **4.4x** |
| Poincaré H² | −1 | 0.6527 | **0.3330** | not run | 1.96x |

### CORRECTION — "weak supervision does not work on hyperbolic" was wrong

That claim was made off a single failed run and is withdrawn. What was actually shown is narrower:
**RRT\* equality anchors** fail on H². The record contradicts the general statement —
`results/manifolds.md` §5 has hyperbolic weak supervision at **1.0112 → 0.5334 (2.6x)**, the *best*
improvement ratio of the three manifolds in that table.

The two are different mechanisms and were conflated:

| | historical (worked) | what was run (failed) |
|---|---|---|
| prior | sphere-packing roadmap + Dijkstra (`build_sphere_roadmap`) | RRT* tree (`build_roadmap`) |
| how it is used | two-sided bounds `T_lb <= T <= T_ub`, hinge | equality anchors `(T − c)²` |
| strategy | `hntfields` | anchor term added to `factored` |

Two reasons the sphere packing should suit negative curvature better, both structural:

1. **Node placement adapts to the geometry.** Each node claims a maximal free sphere and candidates
   inside an existing sphere are rejected, so coverage is by *volume*. Uniform-sample RRT* is exactly
   what exponential volume growth defeats — measured: on H² the tree was still unconverged at 12,000
   iterations after 635 s, with anchor RMS 0.6567 against a field of 0.3330, while the torus converged
   in 21 s to 0.0326.
2. **Bounds tolerate a suboptimal prior; equalities do not.** A graph shortest path is an upper bound,
   so a hinge never drags the field toward a wrong value — which is precisely how the equality anchors
   failed, landing the field at the anchors' own error.

**Standing lesson, now twice: check what a name refers to before porting a result under it.** "Weak
supervision" named two different mechanisms in this project, and the anchor variant's failure was
generalised to the label. Measurement of the sphere-packing roadmap's quality on H² is what decides
this, and it is running.

## Session end state — read this before touching supervision again

**Trustworthy results (label-free; no planner, no roadmap anywhere in the path):**

| scene | manifold | do nothing | label-free RMS | improvement |
|---|---|---|---|---|
| no obstacles | torus / sphere / H² / SO(3) | — | 0.0046 / 0.0004 / 0.0013 / 0.0009 | vs the analytic field |
| 1 obstacle | sphere S² | 0.1256 | **0.0144** | 8.7x |
| 1 obstacle | torus T² | 0.3197 | **0.1844** | 1.7x |
| 1 obstacle | Poincaré H² | 0.6527 | **0.3330** | 2.0x |
| 3 obstacles | torus T² | 0.3158 | **0.1448** | 2.2x |

**Supervised results, and what each one actually ran.** Two different mechanisms were used under the
same CLI flag at different points in the session, which is the error that produced the confusion
below — `--num-anchors` meant *RRT\* equality anchors* before `roadmap_bounds` was written and
*sphere-packing roadmap bounds* after it, with no rename.

| scene | mechanism actually run | nodes | weight | result | vs label-free |
|---|---|---|---|---|---|
| torus, 1 obs | RRT* equality anchors | 30 | 0.5 | **0.0735** | better (2.5x) |
| torus, 1 obs | RRT* equality anchors, under-converged tree | 30 | 0.5 | 0.3393 | worse |
| Poincaré, 1 obs | RRT* equality anchors, off-manifold nodes | 30 | 0.5 | 1.3467 | far worse |
| Poincaré, 1 obs | sphere-pack bounds (pre edge-fix) | 300 | 0.5 | 0.2496 | better, but on broken edge pricing |
| torus, 3 obs | sphere-pack bounds — **mis-configured, retracted** | 30 | 0.5 | ~~0.2012~~ | not a result |

**The 3-obstacle bounds number is retracted, not reported as a comparison.** A 30-node packing at
weight 0.5 is below the configuration's own validity threshold, so it measures the mis-configuration
rather than the mechanism. It was also initially attributed to the wrong mechanism entirely: **no
RRT\* and no equality anchor was in that run.**
It used a 30-node sphere packing with `lower = 0.9 x graph_time` at weight 0.5. The repo's own
hntfields note records that the torus packing needs **~460 nodes** at `max_radius 0.3` before its
graph paths are near-optimal; at 30 nodes the "lower" bound sits *above* truth across much of the
domain, and at weight 0.5 — fifty times the historical `lambda_R = 1e-2` — an invalid bound dictates
rather than nudges. The uniformly **red** (over-predicting) error map is that signature exactly.

**The principle, stated correctly.** An RRT\* cost is an upper bound, so consumption must match
*measured* prior quality:

- **prior error << field error** -> equality anchors are fine and are the strongest tool available.
  Torus 1-obstacle: prior RMS 0.033 against a field of 0.184, p99 ratio 1.005, so the upward bias is
  ~0.5% and negligible — hence 0.0735.
- **prior loose** -> bounds, but only with enough nodes for the corridor to be valid and at
  `lambda ~ 1e-2` so a wrong bound nudges instead of dictating.
- **prior per-node noisy** -> a soft-min base `min_i(cost_i + hop)`, where one bad node loses the min
  to a better neighbour. This is how the historical `weak_supervision`/B5 path consumed RRT\* and it
  never gave trouble.

**The recurring process error, three instances:** "weak supervision" naming two mechanisms;
`tau_bias` results ported without their precondition; `--num-anchors` silently changing meaning
mid-session. In each case a name was reused across a mechanism boundary and outcomes were attributed
across it. Check what a flag currently *does* before attributing a result to what it used to do.

## 2026-08-20 — RRT* validated, and three defects in how sparse anchors were selected

The question was whether the RRT* prior runs correctly, because the sparse anchors drawn from it are
not being selected properly. Three separate defects, each measured below, plus a performance finding
that turned out to be the reason the prior had never actually been validated on the scenes it was
supervising.

### Defect 1 — the shadow-targeted selection was not in the code at all

`factored.rrt_anchors`, the live equality-anchor path, drew its anchors with a plain uniform
`rng.choice`. The recipe that produced this project's best supervised result was *shadow-targeted*
selection, and the repo's own control measures the difference at 1.6x — 30 shadow-targeted anchors
0.0735 against 30 uniform ones 0.1192, same tree, same weight. The selector that implements it,
`sampling.rrt_star_anchors_shadow`, existed but **nothing called it**: dead code since the package
reorganisation. `reproduce_exp2.sh` had a note admitting the row does not reproduce; the cause is
this.

### Defect 2 — the only shadow selector traced a flat-chart ray, which is wrong off the torus

That dead selector labelled a node occluded by walking `start + f·displacement_np(start, x)`. This
repo has already fixed that spelling twice elsewhere (`sweep.shadow_mask`, `geodesic_samples_np`) and
recorded why: it is a chord on the sphere, an overshoot past the chart wall in the Poincaré ball, and
a 3-vector-plus-4-vector shape error on SO(3). Measured against the geodesic ray
`Exp_start(f·Log_start(x))` on the converged tree, one obstacle, seed 1 — nodes the two tests label
differently:

| manifold | disagreements | of nodes |
|---|---|---|
| torus T² | 10 | 3001 (0.3%) |
| sphere S² | 1151 | 12001 (9.6%) |
| Poincaré H² | **2461** | **3001 (82%)** |
| SO(3) | `ValueError` | the shape error, not a wrong answer |

On the torus the shortcut is very nearly right, which is exactly why it survived: the manifold the
recipe was developed on is the one manifold it works on. On H² it mislabels four nodes in five.

### Defect 3 — selection ran after the wrong thinning step

Anchors were drawn from `build_roadmap`'s 300-node subsample rather than from the tree. **The first
version of this entry claimed the subsample "thins the occluded nodes"; that is wrong and is
corrected here.** A uniform subsample preserves the occluded *fraction* — 9.1% of the full
one-obstacle torus tree, 9.0% of a 300-node draw. What it does not preserve is the *count*: 272
occluded nodes become 27. A 30-anchor budget then comes from a candidate pool barely larger than
itself, so which points get pinned is decided by the thinning rather than by the preference.

### The performance finding: XLA was recompiling once per iteration

Building the tree was far slower than the sum of its parts — 350 iterations took 43.9 s on the torus
while the per-iteration primitives sum to ~1.5 ms. The cause is that every geometric primitive routes
through `env.geodesic`/`log_map`/`exp_map`, which are JAX functions, so each distinct argument
**shape** triggers an XLA compilation. An RRT* tree grows by one node per iteration, so the
tree-wide distance query presents a brand-new shape every single iteration. Measured, one
`geodesic_distance_np` call against the tree:

| shape pattern | per call |
|---|---|
| fixed shape, repeated | **0.37 ms** |
| shape grows by one each call (what RRT* did) | **66.13 ms** (178x) |
| bucketed to 6 powers of two | 3.81 ms |

Two changes follow. Array shapes are **bucketed to powers of two** (`_bucket`), with the preallocated
node array prefilled with `start` so padded rows are valid manifold points and every decision masks
them out; and the slowness integral's sample count is fixed once from `max(step, radius)`, the
largest hop any RRT* edge can have, instead of being read off each batch. Edge costs are also
evaluated for the whole neighbour set in one batched call (`edge_times_np`) — worth 26–148x on that
step alone by microbenchmark — but that was **not** the bottleneck and on its own bought nothing.

| torus, one obstacle | before | after |
|---|---|---|
| 350 iterations | 43.9 s | **1.9 s** |
| 1500 iterations | ~13 min (extrapolated) | **2.2 s** |
| 3000 iterations | — | **4.0 s** |

This resolves a contradiction in this log: an earlier entry records the torus tree building in 21 s,
which stopped reproducing after the geodesic-correctness fixes replaced chart arithmetic with JAX
calls. The correctness fix is what destroyed the speed. Bucketing keeps the correct geometry and
recovers the speed; nothing reverts.

**Equivalence, checked rather than assumed.** Node positions depend only on the sampler draws, the
nearest-node argmin and steering, none of which changed, so old and new must agree *bitwise*.
Measured at 350 iterations: `max|Δposition| = 0.000e+00` on both torus and sphere, and costs within
2e-3 relative. The cost drift is the sample-count rule — the batch samples every edge at least as
finely as the per-edge rule did, never less, so it errs toward *over*-pricing, which is the safe
direction for a quantity that must stay an upper bound.

### The validation table — what the prior is actually worth

One obstacle, seed 1, tree converged by the doubling rule, scored against fast marching at
resolution 240 (validation only; no training path can reach it). `>10%` is the fraction of nodes more
than 10% above truth — the tail that broke the torus at 350 iterations, and the tail a shadow-biased
selector concentrates on.

| manifold | nodes | build | prior RMS | p50 | p90 | p99 | max | >10% | base violations |
|---|---|---|---|---|---|---|---|---|---|
| torus T² | 3001 | 7.5 s | 0.0296 | 0.990 | 0.998 | 1.004 | 1.053 | 0.0% | 0 |
| sphere S² | 12001 | 225.6 s | 0.0078 | 0.999 | 1.004 | 1.014 | 1.147 | 0.1% | 0 |
| Poincaré H² | 3001 | 47.1 s | 0.0282 | 0.992 | 1.002 | 1.013 | 1.024 | 0.0% | 0 |
| SO(3) | 12001 | 109.4 s | 0.0393 | 1.005 | 1.022 | 1.044 | 1.323 | 0.1% | **1** |

`base violations` counts nodes whose cost-to-come falls below the free-space geodesic, which no
feasible path can do when `s >= 1`; it is a self-consistency check needing no ground truth.

**The verdict per manifold, against this repo's own principle (equality anchors need prior error
<< field error; otherwise bounds):**

- **Torus — equality anchors.** Prior 0.0296 against a label-free field of 0.1844, a 6x margin, p99
  1.004. This is the regime the historical 0.0735 came from, now confirmed on the exact scene.
- **Poincaré H² — equality anchors, and this is new.** Prior 0.0282 against a field of 0.3330 is a
  **12x** margin, the largest of the three. **The earlier claim that RRT\* cannot converge on H² is
  withdrawn**: that entry recorded the tree still unconverged at 12,000 iterations after 635 s, which
  was the XLA recompilation cost, not the geometry. The converged H² tree now builds in 47 s and is
  the second most accurate prior in the table. The historical 1.3467 failure is fully explained by
  three stacked defects, all now fixed — off-manifold nodes (`in_domain_np`), an unconverged tree
  (the doubling rule), and flat-ray selection (82% mislabelled).
- **Sphere — no anchors.** Prior 0.0078 against a label-free field of 0.0144 is only a 1.8x margin,
  nowhere near the `<<` the principle requires, and the label-free sphere is already the strongest
  result in the ladder. The "+30 anchors" cell in `results/experiments.md` stays `—` deliberately.
- **SO(3) — bounds only, not chased.** The tree is still improving at the doubling cap and carries one
  base violation, so the validator refuses equality anchors on it. SO(3) is not on the 2-D
  experiment ladder; recorded, out of scope for this session.

### What changed in the code

- `sampling.rrt_star` — batched edge costs, bucketed shapes, preallocated node array. Same algorithm,
  bit-identical trees.
- `sampling.converged_tree` — the doubling loop split out of `build_roadmap`, so a sparse selector can
  draw from the full tree. `build_roadmap`'s signature and behaviour are unchanged, so
  `weak_supervision.py` and `run.py` are untouched.
- `sampling.occluded_from_source` — one implementation of the geodesic occlusion ray.
  `sweep.shadow_mask`, the anchor selector and `validate_prior` all delegate to it.
- `sampling.shadow_anchors` — clearance filter (nodes must clear an obstacle by 0.05; the historical
  `weak_clearance` value is unrecoverable, so this is a new choice, made because the slowness ramp is
  steepest at the boundary and the tree's quadrature is least accurate exactly there), then a
  shadow-preferential draw. `shadow_pref = 0` *is* the uniform control, not a second code path.
- `sampling.rrt_star_anchors` / `rrt_star_anchors_shadow` — **deleted**. Both were dead and one
  carried the flat ray. This project's recurring failure is a stale name ported across a mechanism
  boundary; leaving a wrong selector lying around invites a fourth instance.
- `Config.anchor_shadow_pref` (default 3.0), threaded through `SweepConfig` and `sweep._config`.
- `experiments/validate_prior.py` — new; produces the table above. Run it before supervising with a
  prior, which is this repo's third standing rule and until now had no tool behind it.
- `experiments/preflight.py` — a fourth arm covering the equality path, which had no smoke coverage.

**Considered and deliberately not built.** RRT*'s shrinking k-nearest neighbour set and rewiring that
propagates to descendants are both genuine gaps against the published algorithm. The decision rule
was fixed before the table was run — if the converged tree showed 0 base violations and p99 < 1.05,
the algorithm is adequate — and the table meets it on every 2-D manifold. Changing the algorithm
would move the reproduction target for no measured benefit.

### Experiment 2 label-free reproduces exactly, after the changes

Re-run at the documented settings (res 240, 4000 steps, 2048 collocation, `V >= 0` clamp, compact
support 2σ, no causal weighting), seed 1:

| manifold | recorded | measured now |
|---|---|---|
| torus T² | 0.1844 | **0.1844** |
| sphere S² | 0.0144 | **0.0144** |
| Poincaré H² | 0.3330 | **0.3330** |

`preflight` is 12/12 PASS both before and after every change above, with identical per-arm RMS.

### Experiment 2 anchored arms, on the fixed selector

Same recipe as the label-free arms (res 240, 4000 steps, 2048 collocation, `V >= 0`, compact support
2σ, no causal weighting), plus 30 equality anchors at weight **0.5** — not the `1e-2` default, which
belongs to the *bounds* mechanism. Seed 1, one obstacle.

| arm | shadow anchors | RMS | shadow MAE | vs label-free |
|---|---|---|---|---|
| torus — label-free | — | 0.1844 | 0.5383 | — |
| torus — 30 **shadow-targeted** anchors | 6 / 30 | **0.0856** | **0.1209** | **2.2x** |
| torus — 30 **uniform** anchors (control) | 1 / 30 | 0.1446 | 0.4118 | 1.3x |
| Poincaré H² — label-free | — | 0.3330 | 1.1307 | — |
| Poincaré H² — 30 **shadow-targeted** anchors | 8 / 30 | **0.0502** | **0.1221** | **6.6x** |

**The placement effect reproduces, and it is the whole mechanism.** Shadow-targeted 0.0856 against
uniform 0.1446 at identical tree, budget and weight — 1.7x, matching the historical 1.6x (0.0735 vs
0.1192). The anchor counts show why: the preference puts 6 of 30 anchors in shadow against the
control's 1. The remaining gap to the historical 0.0735 is expected and was not chased — tree costs
moved by up to 2e-3 under the batched quadrature and the selection RNG is a different draw, so the
pre-registered bar was the 0.07–0.12 band with a measurably worse control, and both hold.

**Hyperbolic equality anchors are a new arm, and the strongest supervised result in the project.**
0.3330 -> **0.0502**, a 6.6x improvement over label-free and 13x over `do nothing` (0.6527), with
shadow MAE falling 1.1307 -> 0.1221. The log previously recorded H² weak supervision as a failure at
1.3467. That is now fully explained: off-manifold tree nodes (fixed by `in_domain_np`), an
unconverged tree (fixed by the doubling rule, which was unaffordable until the XLA shape fix), and a
flat-chart occlusion ray that mislabelled **82%** of H² nodes. All three are fixed and the mechanism
works better on H² than anywhere else — which is what the prior-quality table predicted, H² having
the largest prior-to-field margin (0.0282 against 0.3330, 12x).

**This is *equality anchors*, not the sphere-packing *bounds* mechanism.** The two have been
conflated twice in this log; they share neither prior nor loss term. Nothing here revises the bounds
results.

### Experiment 3 — three obstacles, label-free

No planner, no roadmap, no anchors anywhere in the training path. Same recipe as the Exp 2 label-free
arms, resolution 240, seed 1.

| manifold | do nothing | label-free RMS | shadow MAE / overall MAE |
|---|---|---|---|
| torus T² | (see note) | **0.2621** | 3.52x |

**Note on 0.1448.** The session-end table records a 3-obstacle torus label-free result of 0.1448.
That number appears exactly once in this log, in a summary table, with **no run entry, no flag set
and no figure** behind it; the fair-baseline suite on the same nominal scene puts label-free methods
at 0.307–0.534, which the 0.1448 does not sit with either. The 0.2621 here is measured at a recorded
recipe that reproduces all three Exp 2 label-free numbers to four decimals. Treat 0.1448 as
unsourced until someone reproduces it, and prefer the recorded run.

To stop this recurring, `sweep` now prints a **`nothing`** column — the RMS of `T = base`, i.e. of
learning nothing — for every seed of every run, and the verdict line quotes the improvement ratio.
Every comparison in the write-up is a ratio against that baseline, so the baseline should never again
be something a reader has to go looking for.

### Path extraction: the failure mode is occlusion, not proximity — and RMS does not predict it

The 3-obstacle torus field, 10 goals spread across free space, reached **10/10** collision-free. That
result is real but it is a weak test, and the goal-mix line now printed with every run says why: **0
of those 10 goals were behind an obstacle.** Splitting goals into three classes on the same field:

| goal class | n | success | how the failures fail |
|---|---|---|---|
| spread across open free space | 10 | **10/10** | — |
| hugging an obstacle boundary (clearance < 0.35) | 20 | **20/20** | — |
| **behind an obstacle (source ray blocked)** | 20 | **10/20** | 10 × `diverged` |
| mixed, a third of each | 60 | 50/60 | every failure was a shadow goal |

**Not one collision in any arm.** Every failure is the same mode: the descent runs to the 4000-step
cap (path length exactly 80.0 = cap × step) with the endpoint still ~2.8–3.0 from the source, and the
trapped endpoints *cluster* — several goals stop at 2.960, several at 2.772. A handful of spurious
basins capture many trajectories; this is not scattered noise.

That is the predicted consequence of the objective. The pointwise residual constrains `‖∇T‖` and not
the *level* of `T`, and behind an obstacle the level is a path integral it cannot see. A field whose
shadow level is wrong by a slowly-varying amount has a near-zero residual and a gradient that
circulates rather than descends. Boundary-hugging goals succeeding 20/20 is the informative negative:
**proximity to an obstacle is not what breaks descent — occlusion is.**

### The anchors improve RMS and make planning *worse*

Testing whether the supervision that fixes the shadow level also fixes path extraction. Same scene
(torus, one obstacle, seed 1), same 20 shadow goals, only the field differing:

| field | RMS | shadow-goal success |
|---|---|---|
| label-free | 0.1844 | **18/20** |
| + 30 shadow-targeted equality anchors | **0.0856** | 14/20 |

**The more accurate field plans worse**, and the reason is visible in the fields themselves. Counting
grid cells strictly below all eight neighbours — spurious basins a descent can be trapped in, with
the fast-marching field's own count as the control on the identical grid:

| torus field | RMS | minima | GT minima | spurious | shadow-goal success |
|---|---|---|---|---|---|
| 1 obstacle, label-free | 0.1844 | 2 | 1 | **1** | **18/20** |
| 1 obstacle, + 30 anchors | **0.0856** | 3 | 1 | 2 | 14/20 |
| 1 obstacle, + 30 uniform anchors | 0.1446 | 7 | 1 | 6 | — |
| 3 obstacles, label-free | 0.2621 | 4 | 1 | 3 | 10/20 |

**Planning success tracks the spurious-minimum count; it does not track RMS, and the two rank in
opposite orders.** By RMS the anchored field is best (0.0856 < 0.1446 < 0.1844); by spurious minima
it is the label-free field (1 < 2 < 6). The mechanism is straightforward: an equality anchor pins `T`
at an isolated point, so the mixture must hit that value locally, and between anchors it overshoots —
buying value accuracy and paying in dimples. The uniform-anchor arm, whose anchors are scattered
rather than concentrated in the shadow, is the worst of the three at 6 spurious minima, which is the
same effect at its least targeted.

**What this means for the claim.** RMS against fast marching measures whether the field *is* the value
function. It does not measure whether the field is *usable*, and this repo has been reporting only the
first. A value function's job is that `-∇T` reaches the source; the honest metric pair is
(RMS, spurious minima), or better, the shadow-goal success rate directly. Recorded, not chased — the
obvious levers (a monotonicity penalty along sampled rays, or consuming the prior as a soft-min base
rather than as equalities, which is how `weak_supervision`/B5 consumed it without trouble) are
untested.

Caveat on the minima count: it is a grid diagnostic, so it inherits the grid. The sphere's lat-long
chart reports 41 minima in the *ground truth* itself, from the pole rings; the count is only
meaningful read against the ground truth's own count on the same grid, which is how the table above
is written.

### Experiment 1 is now exact, on all four manifolds

| manifold | init RMS | RMS | previously recorded |
|---|---|---|---|
| torus T² | 0.0638 | **0.0000** | 0.0046 |
| sphere S² | 0.1394 | **0.0000** | 0.0004 |
| Poincaré H² | 0.0296 | **0.0000** | 0.0013 |
| SO(3) | 0.0219 | **0.0000** | 0.0009 |

The `V >= 0` projection is what closes the last three decimal places. With no obstacles the residual
is minimised at `g = 0`, and projecting the weights onto `V >= 0` after each step drives them to
*exactly* zero rather than to a small residual value, so `T = base` is attained rather than
approached. The previously recorded values predate that clamp being part of the standard arm.

The verdict line no longer prints an improvement ratio here: with no obstacles `do nothing` is 0, so
the ratio is undefined rather than infinite, and "0.00x improvement" would have read as a
catastrophic result.

### Planning, with the optimal path as the reference arm

Rewritten after the first version was rightly called out as a weak test. Two changes:

1. **Every goal is planned twice** — once by descending the learned field, once by descending the
   fast-marching field on its own grid (8-neighbour steepest descent, no interpolation). The second
   is the optimal route. It reaches **60/60 on every scene here**, which is the number that makes the
   learned result readable: every learned failure belongs to the learned field, not to the grid.
   Drawing our paths over the ground-truth panel, as the first version did, put our failures on a
   panel that had nothing to do with them.
2. **One goal set, and its mix is always printed.** 60 goals spread over the manifold with half
   biased onto obstacle boundaries; occlusion is left at the scene's natural rate rather than forced,
   and reported. The earlier 10/10 was on a draw that contained **zero** occluded goals, because the
   sampler excluded them by construction — the rate was real and meaningless.

3-obstacle torus, 60 goals, per class:

| class | goals | optimal | learned |
|---|---|---|---|
| behind an obstacle | 12 | 12/12 | **10/12** |
| hugging a boundary | 21 | 21/21 | 21/21 |
| open free space | 27 | 27/27 | 27/27 |
| **all** | 60 | **60/60** | **58/60** |

Query cost: **0.02 s** to load the trained field, **5.11 s** to plan all 60 paths (85 ms/goal,
including JIT warm-up). That asymmetry against a minutes-long one-off fit is the claim the experiment
exists to make, and it only holds because `plan` loads parameters — the previous version retrained,
which would have hidden it.

**No collisions on any field in any arm.** Every failure is a shadow goal and every failure has the
same shape: the descent runs to the step cap with the endpoint far from the source. Boundary-hugging
goals at 21/21 is the control that makes this specific — proximity to an obstacle is not what breaks
descent, occlusion is, which is exactly where the pointwise residual leaves the level free.

The RMS-vs-usability inversion holds on this goal set too. Same 60 goals, one obstacle:

| torus field | RMS | spurious minima | goals reached | shadow goals |
|---|---|---|---|---|
| label-free | 0.1844 | **1** | **55/60** | **14/19** |
| + 30 shadow-targeted anchors | **0.0856** | 2 | 51/60 | 10/19 |
| + 30 uniform anchors | 0.1446 | 6 | — | — |

`srms/experiments/minima.py` computes that column from the saved fields, so it is reproducible
without retraining and `./run_experiments.sh plan` ends by printing it.
