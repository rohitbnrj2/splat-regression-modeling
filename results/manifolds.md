# One SRM formulation on three curvatures — torus, sphere, hyperbolic

**Claim under test.** The same splat model — same backend code, same strategy, same
hyperparameters — solves the self-supervised Eikonal time-to-go on T² (K = 0), S² (K = +1) and
H² (K = −1). The only per-manifold code is the `Environment`'s `log_map`, `jac_factor` and
`metric_inv`.

**What this is *not* evidence of.** Both an SRM and an MLP are universal approximators, so neither
"it fits" nor "the two tie" is a finding. The substantive claim is about the *diff*: what has to
change per manifold, and whether a mistake in it is detectable. See
[Why this is a claim about the diff](#why-this-is-a-claim-about-the-diff).

Reproduce:

```bash
python -m srms.environments.test_manifolds                  # the 18 correctness checks
python manifold_figure.py --method hntfields                # the combined 3x3 figure
python -m srms.run --environment hyperbolic --method hntfields --backend srm
```

---

## 1. The manifold plumbing is verified before any training

`srms/environments/test_manifolds.py`. Four exact identities per manifold, no ground truth, no
solver, no tuning. All are written manifold-agnostically — the orthonormal frame check 3 needs is
derived from `metric_inv` by eigendecomposition rather than hand-coded per manifold, so the test file
itself contains no per-manifold branch.

| check | identity | catches |
|---|---|---|
| round trip | `Exp_μ(Log_μ(x)) == x` | log/exp are not actually inverses |
| isometry | `‖Log_μ(x)‖ == geodesic(x, μ)` | tangent coords in chart units, not Riemannian units |
| jacobian | `jac_factor == \|det ∂Log_μ/∂x\|` (autodiff) | **wrong curvature term** — `sin` vs `sinh` |
| eikonal | `‖∇ geodesic(·, start)‖_g == 1` | wrong metric; this is the exact residual the solver minimises |

Measured max |error| over 200 sampled pairs each, float64:

| manifold | round trip | isometry | jacobian | eikonal |
|---|---|---|---|---|
| torus | 1.3e-15 | 0.0 | 0.0 | 2.2e-16 |
| sphere | 8.8e-12 | 6.7e-16 | 5.7e-12 | 2.2e-16 |
| hyperbolic | 5.1e-15 | 4.4e-16 | **1.4e-14** | 3.3e-16 |

The bolded cell is the one that mattered: it confirms H²'s `jac_factor = (r/sinh r)^(d−1)` against
autodiff directly, so the formula did not need to be taken on trust from a paper.

### The wrapped Gaussian is a density, and its shortfall has a named cause

∫ N_w dvol by Monte Carlo (400k samples) against the closed-form contained mass:

| manifold | σ = 0.4 | σ = 1.2 | predicted (σ=1.2) | what bounds the domain |
|---|---|---|---|---|
| torus | 1.0031 | 0.9828 | 0.9824 | half-period cut locus |
| sphere | 1.0051 | 0.9669 | 0.9675 | antipodal cut locus |
| hyperbolic | 0.9989 | 0.9508 | 0.9507 | **domain truncation — our choice, not curvature** |

Read this by cause, not by size. The torus and sphere lose tail mass past a **cut locus**, a genuine
limit of the single-image wrapped Gaussian on those manifolds. H² has **no cut locus at all** (Exp is
a global diffeomorphism), so the wrapped Gaussian is exact there for any σ; its shortfall is entirely
the truncation wall we chose at ‖x‖ ≤ 0.9. Hyperbolic therefore shows the *largest* measured
shortfall while having the *only* exact representation — which is why the check reports cause
alongside magnitude.

---

## 2. Ground truth, and its error floor

All three fields are scored against a dense fast marcher at resolution 120. Each marcher is
different, and each has its own discretisation error — **no solver RMS below these numbers is
interpretable.**

| manifold | marcher | RMS vs analytic | mean rel | at res 240 |
|---|---|---|---|---|
| torus | periodic, isotropic, flat chart | 4.65e-2 | 1.96% | 2.74e-2 |
| sphere | anisotropic, geodesic-polar (θ, ψ) grid | 2.13e-2 | 1.44% | 1.46e-2 |
| hyperbolic | **Euclidean marcher, conformally rescaled** | 4.04e-2 | 2.56% | 2.21e-2 |

Measured with `slowness ≡ 1`, where the exact answer is the closed-form geodesic distance. All three
converge first-order (error roughly halves as resolution doubles), so what is left is ordinary
discretisation error, not a modelling error.

**Hyperbolic ground truth needs no hyperbolic solver.** The Poincaré metric is conformal
(`g = λ(x)²δ`, `λ = 2/(1−‖x‖²)`), so `‖∇T‖_g = s` is identically `‖∇T‖_euclid = s·λ`. Running the
ordinary Cartesian marcher with slowness rescaled by λ produces exact hyperbolic travel time —
verified against `d(0,x) = 2·artanh‖x‖` to first-order convergence.

**One caveat on the sphere's floor:** with the default source at `(0,0,1)` the marcher's error is
1.7e-7, not 2.1e-2, because the source sits exactly on the lat-long grid pole and the obstacle-free
field becomes a coordinate line. The 2.13e-2 above is measured with an off-pole source and is the
honest floor for a general scene.

---

## 3. Two declared modelling choices for H²

H^d is unbounded with volume growing like e^{(d−1)r}, so neither is forced by the geometry:

1. **Truncation.** The workspace is `‖x‖ ≤ trunc_radius` (default 0.9), and its rim enters `sdf` as
   an ordinary obstacle — a wall exactly like the geodesic balls. This bounds the domain, supplies
   the render mask, and keeps T finite. At 0.9 the workspace has hyperbolic *diameter* ≈ 5.9,
   comparable to T²'s π√2 ≈ 4.4 and S²'s π ≈ 3.1, so `init_scale`, `source_radius` and `scale_floor`
   transfer across all three untouched. Note the wall sits at hyperbolic *radius* 2.94 — nearer than
   the other two manifolds' cut loci at π, which is why §1's mass table reads the way it does.
2. **Sampling is chart-uniform**, not hyperbolic-volume-uniform. Volume-uniform sampling would put
   essentially every collocation point in the thin annulus at the rim. Chart-uniform matches the
   Cartesian grid the RMS is scored on, so training and evaluation see the same measure.

---

## 4. What actually changes per manifold

`srms/methods/backends/srm.py` is **byte-identical** across all three runs. It contains no periodic
encoding — no `sin`, no `cos`, no read of `env.domain` outside a docstring. Periodicity on the torus
comes from `wrap` inside the log map, not from an input transform. The backend's entire geometric
interface is two functions:

```
env.log_map(mu, x)      # tangent coordinates, ‖·‖ = geodesic distance
env.jac_factor(mu, x)   # |det ∂Log/∂x|, the curvature correction
```
plus `env.metric_inv` on the loss side (`‖∇T‖_g`). Everything else the backend touches
(`sample_domain`, `sdf_np`, `dim`, `tangent_dim`) is scene bookkeeping, not geometry.

The per-manifold geometry is therefore the entire diff:

| manifold | geometry code | `jac_factor` | `metric_inv` |
|---|---|---|---|
| torus (K = 0) | 15 lines | `1` | `I` |
| sphere (K = +1) | 27 lines | `(θ/sin θ)^(n−1)` | `I − xxᵀ` |
| hyperbolic (K = −1) | 35 lines | `(r/sinh r)^(d−1)` | `((1−‖x‖²)/2)²·I` |

The three `jac_factor`s are the same Jacobi-field expression at the three constant curvatures —
`sin`, identity, `sinh`. That substitution *is* the port, and §1's autodiff check is what makes
getting it wrong a caught error rather than a silent one.

### Embedded manifolds need the centres retracted — but this was NOT what fixed the sphere

The splat *representation* needs no change, but the *optimisation* does, and only on embedded
manifolds. `B` (the centres) is an unconstrained optimizer variable, while a wrapped Gaussian is only
defined for a centre that lies **on** the manifold:

| manifold | constraint on `B` | enforced by |
|---|---|---|
| torus | none — `B` is a chart coordinate, `wrap(x−μ)` accepts any μ | nothing needed |
| hyperbolic | `‖B‖ < 1` | `_clamp_ball` already inside `log_map` |
| sphere | **`‖B‖ = 1`** | **nothing — this was the bug** |

On S² a gradient step in ambient R³ walks the centre straight off the sphere. Measured: centres
drifted **0.105** off the unit sphere by **step 65**, at which point `_sphere_frame`'s Householder
construction — which assumes a unit vector — stops producing an orthonormal frame and the density
diverges. That is the cause of both sphere failures (`ntfields` → NaN, `hntfields` → RMS 4.9e10).

The retraction lives in `post_step`: `B ← env.wrap_point(B)` after each optimizer step. It is the
projection retraction, which agrees with the exponential retraction `Exp_B` to O(‖step‖³) — measured
3.3e-4 at step 0.1, 4.3e-7 at 0.01, hitting the float32 floor by 0.001, i.e. indistinguishable at
this learning rate. With it, `‖B‖` stays exactly 1.0000 for the whole run.

**Two corrections to what this section originally claimed, both found by measurement:**

1. **It is not a no-op on the torus.** AdamW's `weight_decay=0.1` acts on *every* parameter including
   `B`, so wrapping `B` changes its numeric value and therefore how much decay it receives. The
   density is unchanged; the optimizer trajectory is not. Measured: torus `ntfields` moved
   0.28136 → 0.28493 (1.3%) and 2,793 → 2,765 params. (Weight-decaying splat *centres* is itself
   geometrically meaningless — it biases them toward the chart origin — but that is pre-existing
   behaviour, untouched.)
2. **It is not what fixed the sphere.** The sphere's NaN was the τ→0 runaway (§6d), not centre drift.
   Retraction is a genuine correctness point — a wrapped Gaussian is undefined for a centre off the
   manifold — but the sphere trains with `--tau_min 0.01` regardless.

A first attempt that put the projection inside `eval_raw` instead was **actively harmful** and is
recorded as a warning: it made the loss invariant to `‖B‖`, so nothing opposed weight decay, `‖B‖`
decayed 1.00 → 0.33 as `(1−lr·wd)^step`, and since the angular step is ≈`‖ΔB‖/‖B‖` the effective step
size *grew* 3×, diverging at step 2481. Scale-invariance alone is not a retraction.

---

## 5. Results

**Setup, identical across runs:** SRM backend, 3 obstacles of geodesic radius 0.5–0.9,
`slowness_max=10`, 4000 steps, 2048 collocation points/step, seed 1, scored against fast marching at
resolution 240. Splat count is chosen by the model (marginal-value rule), not set.

Read every RMS against **do nothing** = `T = base`, the free-space geodesic distance, which is also
the model's initialisation:

| manifold | K | method | splats | RMS | max | do nothing | better by |
|---|---|---|---|---|---|---|---|
| torus | 0 | no supervision | 1947 | 0.2425 | 1.174 | 0.4871 | 2.0× |
| torus | 0 | weak supervision | 1463 | **0.0417** | 0.596 | 0.4871 | **11.7×** |
| torus | 0 | progressive | — | 0.2699 | 1.247 | 0.4871 | 1.8× |
| torus | 0 | temporal difference | — | 0.2295 | 1.107 | 0.4871 | 2.1× |
| sphere | +1 | no supervision | 832 | **0.0545** | 0.515 | 0.3500 | **6.4×** |
| sphere | +1 | weak supervision | 632 | 0.2169 | 0.777 | 0.3500 | 1.6× |
| sphere | +1 | progressive | — | 0.3340 | 1.847 | 0.3500 | 1.0× |
| sphere | +1 | temporal difference | — | 0.4044 | 2.494 | 0.3500 | 0.9× |
| hyperbolic | −1 | no supervision | 1994 | 1.0112 | 4.198 | 1.3611 | 1.3× |
| hyperbolic | −1 | weak supervision | 512 | 0.5334 | 3.147 | 1.3611 | 2.6× |
| hyperbolic | −1 | progressive | — | 0.9527 | 3.938 | 1.3611 | 1.4× |
| hyperbolic | −1 | temporal difference | — | 1.1071 | 4.915 | 1.3611 | 1.2× |

`results/tables.tex` holds the two supervision arms as LaTeX, regenerated from the saved models by
`make_tables.py`, which refuses to emit unless all configs match.

### Progressive annealing and temporal difference do not transfer to the SRM

A collaborator reports both helping an **MLP** on this objective. Neither helps here, and on the
sphere both are catastrophic — 0.0545 → 0.3340 and 0.4044, the latter *worse than doing nothing*.
Plausible but untested: annealing moves the target while densification is simultaneously spawning
and pruning splats, so capacity chases a moving objective — a coupling a fixed-width MLP lacks.

### Comparing against another scene requires the same denominator

A collaborator's Lorentz-model environment uses capsule-thickened geodesic *rays* rather than balls:

| | Poincaré (ours) | Lorentz (theirs) |
|---|---|---|
| obstacle coverage | 34.4% of cells | 7.7% |
| median detour factor | 1.465 | **1.010** |
| RMS of `T = base` (learning nothing) | 1.4360 | **0.2566** |

Half of that domain's true answer *is* the free-space distance, so a low RMS there is largely scene,
not model. The comparable quantity is RMS ÷ do-nothing.

---

## 6. Three pre-existing defects found while porting

None is a manifold problem — 6a and 6b affect the torus identically, and 6c is a property of the
H-NTFields baseline's roadmap meeting a small manifold. All three were in the path of this
comparison and are recorded so the next person does not re-diagnose them.

### 6a. `eikonal.py` solved the inverted PDE

`pde_residual` returned `‖∇T‖_g − 1.0/slow`. `env.slowness` is a **cost per unit length** (≥ 1,
rising inside obstacles), so time-to-go obeys `T = ∫ slowness dl` and therefore `‖∇T‖ = slowness`.
Settled empirically rather than by reading: finite-differencing the fast-marching ground truth on the
torus gives, inside obstacles, `median ‖∇T‖/slowness = 0.998` against `‖∇T‖·slowness = 84.9`. The
module's own boundary condition (`T ≈ eps·slowness(start)`) already assumed the correct convention,
and every other strategy — `ntfields.speed_ratio`, `weak_supervision.roadmap_residual`,
`hntfields.loss_terms` — uses `‖∇T‖/slowness`. **Fixed** to `grad_norm - slow`.

### 6b. `eikonal.py` cannot train from its own default initialisation

Fixing 5a did not change the result (RMS 3.35 → 3.31), so the sign was not the whole story. Cause,
measured at initialisation on the 2-D torus:

| field | `\|grad PDE\|` at V=0 | PDE loss at V=0 | `\|grad BC\|` |
|---|---|---|---|
| unfactored (`eikonal`) | **0.000e+00** | 10.975 | 1.152 |
| factored `T = base/τ` (`ntfields` family) | 2.58e-02 | 0.495 | — |

`srm.init_params` sets `V = 0`, so the unfactored field starts at `T ≡ 0`, where `∇T = 0` and
`d‖∇T‖/dparams` collapses to 0/0 (regularised to zero). `T ≡ 0` is an **exact stationary point** of
the PDE term, so only the 32-point boundary ring has gradient — the model learns a small bump at the
source and leaves `T ≈ 0` everywhere else. Unchanged by `--no-causal` and by 64 vs 384 splats. The
factored strategies are immune because `base` is the analytic geodesic distance, nonzero with nonzero
gradient at `V = 0`.

**Not fixed** — a non-degenerate init (or the `base·(1+g)` factorization the original `torus.py`
used for this method) would fix it, but that changes the method. The planner-free leg of this study
uses `ntfields` instead, which is PDE-only, roadmap-free, and structurally identical across the three
manifolds.

### 6c. The roadmap baseline needs per-scene node density (H-NTFields only)

`build_sphere_roadmap` at the shared default of 300 nodes, measured against fast marching:

| manifold | nodes reachable | free-sphere radius (min/med/max) | graph/true travel time (med/max) | frac below truth |
|---|---|---|---|---|
| torus | 300/300 | 0.00 / **0.30** / 0.30 | 0.996 / 1.056 | 2.0% |
| sphere | 300/300 | 0.00 / **0.03** / 0.30 | 0.995 / 1.086 | 44.7% |
| hyperbolic | 266/300 | 0.00 / **0.30** / 0.30 | 1.146 / 2.796 | 7.1% |

S² has area 4π ≈ 12.6 against T²'s (2π)² ≈ 39.5, so 300 non-overlapping spheres cannot be large
there; the packing collapses to a median radius of 0.03 and H-NTFields' "start–goal perturbation"
sampling degenerates onto ~300 fixed points. This is a property of the *baseline's* roadmap
construction meeting a small manifold — the SRM and the manifold plumbing are not involved. The
confirmation run (sphere, `--hnt_nodes 75`, everything else identical) is in §5.

---

## Why this is a claim about the diff

Universal approximation makes "it fits" the null hypothesis on every manifold, and the established
torus result is a **tie** at matched parameters (SRM 0.0659 / 3,584 params vs MLP 0.0650 / 3,521).
So the accuracy column is not where the argument lives. What survives:

- **The SRM's per-manifold delta is three textbook functions with exact unit tests.** §1 is that test
  suite. A wrong curvature term fails at 1e-14, before training.
- **The MLP's per-manifold delta is an input encoding with no exact criterion.** Measured this
  session: `mlp.py`'s Fourier features read `env.domain` as a *period*, which is correct on the torus
  and meaningless on the sphere — north and south poles map to features 1.7e-7 apart despite being π
  apart, and feature-distance vs geodesic-distance correlates at only 0.417. It ran, produced
  numbers, and was wrong. **Any sphere or hyperbolic MLP comparison is invalid until this is fixed**
  — it is dormant, not resolved.
- The honest steelman: "feed the canonical embedding" *is* a uniform MLP recipe (sin-cos for T²,
  ambient R³ for S², raw ball coords for H²). So the claim is not *possible vs impossible* — it is
  **principled and testable vs ad hoc and silently fragile**. The embedding recipe also has a
  theorem-backed failure case at SO(3) (no continuous representation in ≤ 4 dimensions, Zhou et al.
  2019), where the SRM needs only the SO(3) log map.
- **Hyperbolic does not discriminate between the two** and should not be pitched as if it does. H^d
  is contractible — one global chart, so a correct MLP needs no encoding trick at all, and curvature
  enters through `metric_inv`, which is loss-side and representation-agnostic. It belongs here as a
  generality demonstration for the SRM, not as a comparative win.
