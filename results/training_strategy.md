# Training strategy

Symbols, defined once (canonical table in `CLAUDE.md`):

| symbol | meaning |
|---|---|
| `x` | a point on the manifold |
| `start` | the source point; fixed |
| `T(x)` | time-to-go — cost of travelling from `start` to `x`. The field being learned |
| `base(x)` | analytic geodesic distance from `start` to `x`. Closed form, no solver |
| `s(x)` | slowness — cost per unit length, `>= 1`. `1` in free space, `10` deep inside an obstacle |
| `g(x)` | the splat mixture's scalar output at `x` — the learned correction |
| `q(x)` | speed ratio `‖∇T‖_g / s`. Equals 1 exactly when the Eikonal equation holds |
| `(V, A, B)` | splat weights, scale matrices, centres |

---

## The five choices

**1. What we are solving.** `T` is the time-to-go if it satisfies the Eikonal equation
`‖∇T‖_g = s(x)` with `T(start) = 0`. `‖·‖_g` is the Riemannian norm, taken through
`env.metric_inv` — identity on a flat chart, a tangent-plane projector on an embedded manifold. That
one function is what makes the same code correct on every manifold.

**2. How `T` is written.** `T(x) = base(x) · exp(g(x))`.

- `T(start) = 0` exactly, for any finite `g`, since `base(start) = 0`. No boundary term is needed.
- `g = 0` gives `T = base` **exactly**, so the obstacle-free answer is representable, not approached.
- `exp` is smooth with derivative 1 at `g = 0`, so the optimiser is well conditioned right at the
  free-space solution.

This is where we differ from NTFields, and it is the difference that made Experiment 1 pass. They use
`T = base/tau` with `tau = sigmoid(g + 4)`. Because `sigmoid` lives in the *open* interval (0,1),
`T > base` strictly and the free-space answer needs `g -> +inf`. Measured on the obstacle-free torus:
a flat 1.83% overestimate at every distance from the source, stalling at RMS 0.030 where
`base·exp(g)` reaches 0.0015.

`1 + relu(g)` was considered and rejected. It also makes `T = base` attainable and additionally
enforces `T >= base` (correct whenever `s >= 1`), but it is flat for `g < 0`, so a correction that
goes negative has zero gradient and can never return. `exp` has no dead region; `T >= base` is left
to the physics rather than hardcoded.

**3. What `g` is.** A mixture of `k` wrapped Gaussians:
`g(x) = sum_j V_j · N(Log_{B_j}(x); 0, A_j) · |det ∂Log_{B_j}/∂x|`. `Log` is the manifold's log map,
and the determinant is the curvature correction. The per-manifold change is four textbook functions
(`log_map`, `exp_map`, `jac_factor`, `metric_inv`), each checked against an exact identity to 1e-11
before any training runs (`environments/test_manifolds.py`).

**3b. What curvature is, and where it enters.** Two functions carry it, and nothing else does.
`metric_inv(x)` sets how gradients are measured; `jac_factor(mu, x)` corrects the volume element in
the wrapped Gaussian. Both are the constant-curvature Jacobi-field expression evaluated at that
manifold's `K`:

| manifold | `K` | `jac_factor` | `metric_inv` |
|---|---|---|---|
| torus T² | 0 | `1` | `I` |
| sphere S² | +1 | `(r/sin r)^(d−1)` | `I − x xᵀ` |
| Poincaré H² | −1 | `(r/sinh r)^(d−1)` | `((1−‖x‖²)/2)²·I` |
| SO(3) | +¼ | `((r/2)/sin(r/2))²` | `¼(I − q qᵀ)` |

Two points that are easy to get wrong:

*The torus is flat, and that is about the metric, not the shape.* T² here is `R²/(2piZ)²` — a quotient
of the Euclidean plane by translations, which are isometries, so every point has a neighbourhood
isometric to a flat disc and `K = 0` exactly. The doughnut in R³ has varying curvature (positive
outside, negative inside) because that embedding induces a *different* metric; the flat torus has no
smooth isometric embedding in R³ at all, only in R⁴ as two orthogonal circles. Gauss-Bonnet makes the
independence sharp: `∫K dA = 2·pi·chi` and the torus has `chi = 0`, so *any* metric on a torus averages
to zero curvature — the flat one attains it pointwise. Physically it is flat because T² is the
joint-angle space of a 2-joint arm whose joints wrap independently and do not interact. Curvature
enters only through the arm's mass matrix, which is exactly why `metric_inv` is kept separate from
splat placement: `metric_inv -> M(theta)^-1` curves the same torus without touching where splats sit.

*SO(3)'s `K = 1/4` is a statement about our distance convention.* For a compact Lie group with a
bi-invariant metric, `K(X, Y) = ¼‖[X, Y]‖²` on orthonormal `X, Y`. In `so(3)` the bracket is the cross
product, so `‖[X, Y]‖ = 1` always and `K = ¼` on every plane — constant. The value tracks the scale:
SO(3) is S³ quotiented by ±1, and we define distance as the **rotation angle**, twice the great-circle
distance on the unit S³. Doubling distances quarters curvature, and unit S³ has `K = +1`. Define
distance as half the rotation angle instead and the same manifold reads `K = +1`. Any bi-invariant
metric on SO(3) is a constant multiple of this one, so SO(3) always has *constant positive* curvature;
only the number is convention. SO(3) is also unusually well-behaved among Lie groups here: `K = 0`
wherever `[X, Y] = 0`, so anything of rank >= 2 (SU(3)) has genuinely flat 2-planes, while SO(3) has
rank 1 and therefore none.

**4. The loss.** `mean( (q − 1)² )`, `q = ‖∇T‖_g / s`, over collocation points resampled uniformly
every step, `∇T` by autodiff.

Close to NTFields but not identical. Theirs is `|1 − √q| + |1 − 1/√q|` (their Eq. 4), which is
symmetric in `q ↔ 1/q` but **non-differentiable at exactly `q = 1`** — the point every run is trying
to reach. `(q − 1)²` is smooth there and least-squares structured, which also keeps a Gauss-Newton or
Levenberg-Marquardt step available on a frozen basis. Their symmetry argument matters when `T` is
unbounded above; here `T = base·exp(g)` already keeps `T > 0` and ties it to `base`, so the asymmetry
is not paid for. `ntfields.py` keeps their loss verbatim as the baseline.

Optional causal weighting is available (`cfg.causal`): each point's residual is discounted by the
accumulated residual of points nearer the source, a source-outward curriculum. It is a schedule, not
part of the objective.

**5. Optimisation.** Adam, learning rate 3e-3, global gradient-norm clip 1.0. After each step: the
singular values of every `A_j` are floored at `scale_floor` (without it a splat collapses onto the
Eikonal kink and spikes `‖∇T‖` to ~1e7), and centres `B_j` are retracted onto the manifold (without
it a sphere centre leaves the sphere by step 65 and the density goes NaN). Capacity is optionally
adaptive: prune near-zero weights, spawn where the residual is largest, stop when a densify pass buys
less than `densify_min_gain` fractional residual reduction per splat added.

---

## Where we sit relative to the literature

| | NTFields | P-NTFields | H-NTFields | ours |
|---|---|---|---|---|
| field | `base/tau` | `base/tau` | `base/tau` | **`base·exp(g)`** |
| loss | `\|1−√q\| + \|1−1/√q\|` | + viscosity, progressive speed | + Bellman, normal, causal, roadmap bounds | **`(q−1)²`** |
| model | MLP | MLP | MLP | **wrapped-Gaussian mixture** |
| supervision | none | none | sparse roadmap | none |
| geometry | Euclidean | Euclidean | Euclidean | **Riemannian, via `metric_inv`** |

Two deliberate departures: the field parameterisation and the residual. Everything else — the speed
ratio, per-step collocation resampling, the fixed-source setting — matches. All three baselines are
implemented verbatim in `srms/methods/strategies/` for comparison.

---

## Status — the full ladder runs, in one command

`./run_experiments.sh {check|validate|exp1|exp2|exp3|plan|all}`. Seed 1 throughout below; the seed
places the obstacles as well as initialising the model, so a 5-seed sweep is a 5-*scene* sweep.
Numbers and figures: [`experiments.md`](experiments.md).

**Experiment 1 (no obstacles, label-free, scored against the analytic field).** Exact on every
manifold — RMS **0.0000** on the torus, sphere and Poincaré H², from `init` 0.0638 / 0.1394 / 0.0296.
The answer is `T = base`, and the `V >= 0` projection is what makes it *attained* rather than
approached: the residual is minimised at `g = 0`, and projecting the weights onto `V >= 0` drives
them to exactly zero instead of to a small residual value.

**Experiment 2 (one obstacle, scored against fast marching).** Read the improvement over `do nothing`
(`T = base`), never the absolute RMS.

| manifold | K | do nothing | label-free | + 30 anchors | best |
|---|---|---|---|---|---|
| sphere S² | +1 | 0.2552 | **0.0144** | — (prior not good enough) | **17.7x** |
| torus T² | 0 | 0.3197 | 0.1844 | **0.0856** | **3.7x** |
| Poincaré H² | −1 | 0.6527 | 0.3330 | **0.0502** | **13.0x** |

**Experiment 3 (three obstacles, label-free — no planner, no roadmap, no anchors).**

| manifold | K | do nothing | label-free | improvement |
|---|---|---|---|---|
| sphere S² | +1 | 0.6156 | **0.1672** | **3.7x** |
| torus T² | 0 | 0.4871 | **0.2621** | 1.9x |
| Poincaré H² | −1 | 1.3611 | **0.8740** | 1.6x |

---

## What the label-free objective can and cannot do

**It cannot fix the level behind an obstacle.** Six mechanisms — a `g >= 0` penalty, the `V >= 0`
projection, compact support, L1 sparsity, splat geometry, and curriculum ordering — all land the
one-obstacle torus between 0.184 and 0.226. Free space reaches MAE 0.094 while the shadow sits at
0.28–0.54, and the obstacle-free ceiling is now measured at exactly 0, so the residual is **not**
representational. The pointwise residual constrains `‖∇T‖` and says nothing about the value of `T`;
behind an obstacle the level is the accumulated cost of a detour, and a field off by a constant there
has near-zero residual.

**Compact support makes that worse, by construction.** A splat truncated at `k` sigma receives
gradient only from collocation points inside its own radius, so it cannot carry level information
into the shadow. Measured: 2σ gives simultaneously the cleanest free space (0.094) and the worst
shadow (0.538). Truncation is still worth keeping — it halves free-space error and makes each splat a
provably local object — but it trades against the shadow rather than helping it.

**Thirty sparse anchors fix it, and where they go matters more than how many.** RRT* cost-to-come at
30 nodes as equality targets, weight 0.5, added to the standard objective. The anchors are
self-supervised — computed from the known slowness field, never from fast marching.

| torus, one obstacle | RMS | anchors in shadow |
|---|---|---|
| label-free | 0.1844 | — |
| 30 **uniform** anchors | 0.1446 | 1 / 30 |
| 30 **shadow-targeted** anchors | **0.0856** | 6 / 30 |

---

## Validate the prior before supervising with it

`./run_experiments.sh validate`. An RRT* cost is an *upper* bound, so pinning it as an equality biases
the field upward by the planner's suboptimality; the mode is correct only when the prior's error is
much smaller than the field's. Measured on each scene, one obstacle, converged tree, against fast
marching (validation only — no training path can reach it):

| manifold | prior RMS | p99 ratio | >10% high | base violations | field it corrects | margin | verdict |
|---|---|---|---|---|---|---|---|
| torus T² | 0.0296 | 1.004 | 0.0% | 0 | 0.1844 | 6x | equality anchors |
| Poincaré H² | 0.0282 | 1.013 | 0.0% | 0 | 0.3330 | **12x** | equality anchors |
| sphere S² | 0.0078 | 1.014 | 0.1% | 0 | 0.0144 | 1.8x | **no anchors** |
| SO(3) | 0.0393 | 1.044 | 0.1% | **1** | — | — | bounds only, unconverged |

`base violations` counts nodes whose cost-to-come falls below the free-space geodesic, which no
feasible path can do when `s >= 1` — a self-consistency check needing no ground truth at all. `>10%`
is the tail that matters: at an under-converged 350 iterations the *median* torus node was within
0.8% while 9% of nodes were more than 10% high, and shadow-biased selection concentrates on exactly
that tail, which is how anchors once sent training backwards (0.1844 → 0.3393).

Convergence is no longer a hand-set number. The tree is rebuilt at doubling budgets until the mean
cost-to-come stops falling, which is self-supervised — RRT* costs only ever decrease as the tree
fills in. That check was previously unaffordable: every geometric primitive routes through JAX, so a
tree growing by one node per iteration presented a new array shape and paid an XLA compile every
iteration (0.37 ms per query at a repeated shape, **66 ms** at a growing one). Bucketing shapes to
powers of two took a 1500-iteration torus tree from ~13 minutes to **2.2 s**.

**This is what withdrew the claim that RRT\* cannot converge on H².** That claim rested on a tree
still unconverged after 635 s at 12,000 iterations — which was compilation cost, not geometry. The
converged H² tree now builds in 47 s and is the *second most accurate prior in the table*, which is
why hyperbolic equality anchors work at all, and work best.

---

## The field can be accurate and still not be usable

`./run_experiments.sh plan`. The value function's job is that `-∇T` reaches the source. RMS against
fast marching does not measure that, and on this evidence does not predict it.

Every goal is planned twice: once by descending the learned field, once by descending the
fast-marching field on its own grid. The second is the optimal route, and it is the reference the
first has to be read against — it reaches **60/60** on every scene here, so every learned failure
belongs to the learned field and not to the grid.

60 goals, spread over the manifold with half biased onto obstacle boundaries, on the 3-obstacle torus:

| class | goals | optimal | learned |
|---|---|---|---|
| behind an obstacle | 12 | 12/12 | **10/12** |
| hugging a boundary | 21 | 21/21 | 21/21 |
| open free space | 27 | 27/27 | 27/27 |
| **all** | 60 | **60/60** | **58/60** |

**No collisions, on any field, in any arm.** Failures are all the same mode — the descent runs to the
step cap in the shadow — and all of them are shadow goals. Boundary-hugging goals at 21/21 make this
specific rather than a general statement about hard goals: **proximity to an obstacle is not what
breaks descent; occlusion is**, which is precisely where the pointwise residual leaves the level free.

**And the supervision that fixes the level makes the paths worse.** Same 60 goals, one obstacle,
only the field differing. Spurious minima are grid cells strictly below all eight neighbours, in
excess of the fast-marching field's own count on the identical grid:

| torus field, one obstacle | RMS | spurious minima | goals reached | shadow goals |
|---|---|---|---|---|
| label-free | 0.1844 | **1** | **55/60** | **14/19** |
| + 30 shadow-targeted anchors | **0.0856** | 2 | 51/60 | 10/19 |
| + 30 uniform anchors | 0.1446 | 6 | — | — |

**The rankings are opposite.** By RMS the anchored field wins; by spurious minima and by goals
reached the label-free field does, and the entire gap sits in the shadow class. An equality anchor
pins `T` at an isolated point, so the mixture must hit that value locally and overshoots between
anchors — buying value accuracy and paying in dimples. The uniform arm, whose anchors are scattered
rather than concentrated, is worst at 6.

Untested levers, in the order they look most promising: a monotonicity penalty along sampled rays
from the source, which targets the defect directly; and consuming the prior as a soft-min base
`min_i(cost_i + hop)` rather than as equalities, which is how `weak_supervision` consumed RRT* and
never produced this artefact.

---

## Reading the numbers

- **Compare manifolds by improvement over `do nothing`**, not by absolute RMS. Every run prints its
  own `nothing` column for this reason.
- **Fast marching is the reference with obstacles, the analytic geodesic without.** The reference's
  own discretisation error is printed as `ref err`, and nothing below it is interpretable: 0.046 on
  the torus at resolution 120, 0.027 at 240. Run obstacle scenes at 240 or finer.
- **Report `(RMS, spurious minima)` as a pair**, or the shadow-goal success rate directly. RMS alone
  says whether the field *is* the value function, not whether it is *usable*.
- **Say which mechanism ran.** "Weak supervision" has named two different things in this project —
  RRT* equality anchors and sphere-packing roadmap *bounds* — and results have twice been attributed
  across that boundary. Everything above is equality anchors; nothing here revises the bounds results.
