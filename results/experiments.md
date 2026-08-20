# Experiment results — self-supervised Eikonal time-to-go on Riemannian manifolds

Method and preconditions: [`training_strategy.md`](training_strategy.md). Running log:
[`../investigation.md`](../investigation.md). Reproduce everything with
[`../run_experiments.sh`](../run_experiments.sh).

Every number below comes from one command:

```bash
./run_experiments.sh check      # geometry identities, self-supervision proof, smoke matrix
./run_experiments.sh validate   # is the RRT* prior good enough to supervise with?
./run_experiments.sh exp1       # no obstacles
./run_experiments.sh exp2       # one obstacle
./run_experiments.sh exp3       # three obstacles
./run_experiments.sh plan       # paths on the learned field
```

Common settings: `T = base·exp(g)`, splat mixture with `V >= 0` and compact support (2σ), Eikonal
residual `(‖∇T‖_g/s − 1)²`, Adam, 4000 steps, 2048 collocation points/step, adaptive densification,
no causal weighting, seed 1, scored at resolution 240 (SO(3) at 40, its grid being 3-D).

**Read every comparison as an improvement over `do nothing`** — the error of `T = base`, i.e. of
learning nothing at all. Scene difficulty varies by an order of magnitude across curvatures, so
absolute RMS is not comparable between manifolds. Every run now prints its own `nothing` column, so
the baseline is never something a reader has to go looking for.

Figures: [`figures/`](figures/), one directory per arm, each holding a labelled figure, a bare
unlabelled twin for LaTeX captions and the raw fields (`.npz`). Arms rerun since parameter saving
was added also carry a `.pkl`, which is what `plan` loads; rerunning an arm regenerates it.
Panels are ground truth, learned field, signed error (blue-white-red), source marked with a star.

---

## Experiment 1 — no obstacles

The answer is the analytic geodesic distance from the source, in closed form, so this scores against
it exactly and tests the machinery rather than obstacle reasoning. `init` is a random perturbation of
the correction `g`, so the pair is a genuine recovery, not a held fixed point.

| manifold | K | init RMS | RMS | figure |
|---|---|---|---|---|
| torus T² | 0 | 0.0638 | **0.0000** | [exp1/torus](figures/exp1/torus/) |
| sphere S² | +1 | 0.1394 | **0.0000** | [exp1/sphere](figures/exp1/sphere/) |
| Poincaré H² | −1 | 0.0296 | **0.0000** | [exp1/poincare_hyperbolic](figures/exp1/poincare_hyperbolic/) |
| SO(3) | +1/4 | 0.0219 | **0.0000** | [exp1/so3](figures/exp1/so3/) |

Exact to four decimals, against 0.0046 / 0.0004 / 0.0013 / 0.0009 previously recorded. The `V >= 0` clamp is
why: the residual is minimised at `g = 0`, and a projection onto `V >= 0` drives the weights to
exactly zero rather than to a small residual value, so `T = base` is attained rather than approached.

---

## Experiment 2 — one obstacle

Scored against fast marching, the ground truth.

| manifold | K | do nothing | label-free | improvement | + 30 anchors | improvement |
|---|---|---|---|---|---|---|
| sphere S² | +1 | 0.2552 | **0.0144** | **17.7x** | — (see below) | — |
| torus T² | 0 | 0.3197 | 0.1844 | 1.7x | **0.0856** | **3.7x** |
| Poincaré H² | −1 | 0.6527 | 0.3330 | 2.0x | **0.0502** | **13.0x** |

Figures: [torus label-free](figures/exp2/torus_labelfree/) ·
[torus + anchors](figures/exp2/torus_anchors30/) · [sphere](figures/exp2/sphere_labelfree/) ·
[H² label-free](figures/exp2/poincare_hyperbolic_labelfree/) ·
[H² + anchors](figures/exp2/poincare_anchors30/).

The anchors are 30 sparse RRT* cost-to-come values used as equality targets at weight 0.5, selected
with a 3x draw preference for nodes whose source geodesic is occluded. They are self-supervised —
computed from the known slowness field, never from fast marching.

### Anchor placement is worth 1.7x

Same 30 anchors, same tree, same weight — only the selection rule differs:

| torus, one obstacle | RMS | anchors landing in shadow |
|---|---|---|
| label-free | 0.1844 | — |
| 30 **uniform** anchors | 0.1446 | 1 / 30 |
| 30 **shadow-targeted** anchors | **0.0856** | 6 / 30 |

[uniform control](figures/exp2/torus_anchors30_uniform/). The anchors work by pinning the field's
*level* where the pointwise residual leaves it free, which is behind the obstacle — so where they go
matters more than how many there are.

### Why the sphere gets no anchors

Equality anchors are only correct when the prior is much more accurate than the field it is
correcting, since an RRT* cost is an upper bound and pinning it biases the field upward by the
planner's suboptimality. Measured by `./run_experiments.sh validate`:

| manifold | prior RMS | label-free field | margin | verdict |
|---|---|---|---|---|
| torus T² | 0.0296 | 0.1844 | 6x | equality anchors |
| Poincaré H² | 0.0282 | 0.3330 | **12x** | equality anchors |
| sphere S² | 0.0078 | 0.0144 | 1.8x | **no anchors** |

The sphere's prior is barely better than the field it would be correcting, and the label-free sphere
is already the strongest result in the ladder. The cell stays empty on purpose.

### Why difficulty orders by curvature

At matched obstacle radius, measured on the scenes themselves:

| manifold | K | obstacle % of volume | shadow % | max detour |
|---|---|---|---|---|
| sphere S² | +1 | 12.6% | **0.7%** | 1.36 |
| torus T² | 0 | 3.3% | 5.8% | 2.14 |
| Poincaré H² | −1 | 26.8% | 6.5% | 3.55 |

On the sphere a ~4x larger obstacle casts an ~8x smaller shadow: at K=+1 geodesics reconverge toward
the antipode, so paths bending around a cap rejoin almost at once. At K=0 they stay parallel and the
shadow is an open corridor. At K=−1 they diverge exponentially, giving the largest shadows and the
deepest detours.

---

## Experiment 3 — three obstacles, label-free

No planner, no roadmap, no anchors anywhere in the training path.

| manifold | K | do nothing | label-free | improvement | shadow MAE / MAE | figure |
|---|---|---|---|---|---|---|
| torus T² | 0 | 0.4871 | **0.2621** | 1.9x | 3.52x | [exp3/torus](figures/exp3/torus_labelfree/) |
| sphere S² | +1 | 0.6156 | **0.1672** | **3.7x** | 3.98x | [exp3/sphere](figures/exp3/sphere_labelfree/) |
| Poincaré H² | −1 | 1.3611 | **0.8740** | 1.6x | 3.03x | [exp3/poincare](figures/exp3/poincare_hyperbolic_labelfree/) |

The curvature ordering from Experiment 2 holds and the shadow is where the error lives on every
manifold, by a factor of 3 to 4.

---

## Planning on the learned field

The point of learning `T` is that a path to *any* goal is then a gradient descent on a closed-form
mixture — no graph, no re-solve, no collision-checking search. Measured on the 3-obstacle torus,
loading trained parameters rather than refitting:

| | |
|---|---|
| load the trained field | **0.02 s** |
| plan 60 paths | **5.11 s** (85 ms/goal, JIT warm-up included) |

**Every goal is planned twice** — once by descending the learned field, once by descending the
fast-marching field on its own grid. The second is the optimal route and the reference arm: without
it a success rate has no ceiling, and a failure could belong to the field or to the grid. The figure
shows both from the same goals, so a failure is legible — the left panel is where the path *should*
have gone, the right is where the learned field sent it.

60 goals spread over the manifold with half biased onto obstacle boundaries; occlusion is left at the
scene's natural rate and reported rather than assumed.

| 3-obstacle torus, 60 goals | optimal | learned |
|---|---|---|
| all goals | **60/60** | **58/60** (97%) |
| behind an obstacle | 12/12 | **10/12** |
| hugging a boundary | 21/21 | 21/21 |
| open free space | 27/27 | 27/27 |

[figure](figures/plan/torus_paths.png). **No collisions in any arm**, on any field. Every failure is
the same mode — the descent runs to the step cap in the shadow with the endpoint still far from the
source — and every failure is a shadow goal. Boundary-hugging goals at 21/21 is the informative
control: **proximity to an obstacle is not what breaks descent; occlusion is**, which is exactly the
region where the pointwise residual leaves the field's level free.

### RMS does not predict whether the field is usable

Same 60 goals, same scene, only the field differing. Spurious minima are grid cells strictly below
all eight neighbours, in excess of the fast-marching field's own count on the identical grid:

| torus field, one obstacle | RMS | spurious minima | goals reached | shadow goals |
|---|---|---|---|---|
| label-free | 0.1844 | **1** | **55/60** | **14/19** |
| + 30 shadow-targeted anchors | **0.0856** | 2 | 51/60 | 10/19 |
| + 30 uniform anchors | 0.1446 | 6 | — | — |

[label-free](figures/plan/torus_labelfree_paths.png) ·
[anchored](figures/plan/torus_anchors30_paths.png). **The rankings are opposite.** By RMS the
anchored field wins; by spurious minima and by goals reached the label-free field does, and the whole
gap is in the shadow class. An equality anchor pins `T` at an isolated point, so the mixture must hit
that value locally and overshoots between anchors — buying value accuracy and paying in dimples.

Report `(RMS, spurious minima)` as a pair, or the goals-reached rate directly. RMS alone measures
whether the field *is* the value function, not whether it is *usable*. Reproduce the whole table with
`./run_experiments.sh plan`, which ends by printing the spurious-minimum count for every saved field.

---

## What changed from the previous version of this file

- Experiment 1 is exact (0.0000) rather than 0.0046–0.0013.
- The torus anchored arm is 0.0856, from a rebuilt selector; the previously recorded 0.0735 came from
  a shadow-targeted selection that was **not in the code** at the time this was written.
- Hyperbolic anchors are no longer "pending" — 0.0502, the strongest supervised result here.
- The sphere's `do nothing` is 0.2552, not the 0.1256 previously recorded; the improvement is larger
  than was claimed, not smaller. The torus (0.3197) and Poincaré (0.6527) baselines reproduce exactly.
- The 3-obstacle torus label-free number is 0.2621. The previously recorded 0.1448 appears once in
  the log with no run entry, flag set or figure behind it; treat it as unsourced.
- Planning is measured across goal classes rather than as a single rate, because a single rate over
  open free-space goals is 10/10 on a field that fails half its shadow goals.
