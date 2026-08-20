# Method & Implementation Details — draft

Register and conventions follow `results/paper_notes.md`: every number is measured and traceable to
`results/manifolds.md`; anything not yet measured is marked **[PENDING]** so it cannot drift into a
claim. Section numbering assumes §1 Introduction, §2 Related Work.

---

## 3. Method

### 3.1 Planning as a value function on a Riemannian manifold

A robot's configuration space is rarely a box. A revolute *n*-joint arm lives on the *n*-torus `Tⁿ`;
an orientation lives on `SO(3)`; a satellite's pointing direction on `S²`; a car's pose on the
sub-Riemannian `SE(2)`. Each carries a metric that is not the identity — for the arm, the
kinetic-energy (inertia) metric `M(q)`, under which the cheapest motion is direction-dependent.

We plan by learning a **value function** rather than a path. Let `(M, g)` be a Riemannian manifold
and `s : M → [1, ∞)` a *slowness* field — cost per unit length, ≈1 in free space and rising smoothly
toward obstacles. The minimum arrival time from a fixed source `x_s` to every other configuration is
the viscosity solution of the Eikonal equation

```
 ‖∇T(x)‖_g = s(x),        T(x_s) = 0,                                                    (1)
```

and the time-optimal trajectory from any `x` is steepest descent, `ẋ = −∇T/‖∇T‖_g`. Learning `T`
therefore yields *all* start configurations at once — the field **is** the policy — and the
representation of `T` is the object under study in this paper.

Equation (1) sets four requirements on that representation, and they are what motivate the model in
§3.2–§3.4:

| requirement | why (1) demands it |
|---|---|
| defined intrinsically on `M` | `T` is a function on the manifold; a model that consumes coordinates must first *choose* how to encode them (§3.5) |
| closed-form, smooth `∇T` | the loss is built from `∇T`, and second-order terms differentiate through it again; the readout `−∇T` is the planner |
| the metric appears explicitly | `‖·‖_g` is not a rescaling of `‖·‖₂`; getting `g` wrong minimises a different PDE |
| capacity placeable where the field is hard | the viscosity solution is non-smooth exactly along the cut locus and obstacle rims, and smooth over most of the domain |

### 3.2 Splat Regression Models

A Splat Regression Model (SRM) represents a function as a weighted sum of affinely transformed
probability densities:

```
 f(x) = Σ_j V[j] · det(A[j])⁻¹ · ρ(A[j]⁻¹ (x − B[j])),                                    (2)
```

with parameters `V ∈ R^{k×p}` (weights, sign-free), `A ∈ R^{k×d×d}` (a full, unconstrained affine
scale/rotation — so each splat is anisotropic and tilted), and `B ∈ R^{k×d}` (centres). Taking the
mother density `ρ` to be the standard Gaussian makes `A` a square root of the covariance,
`Σ_j = A[j] A[j]ᵀ`. 3D Gaussian Splatting is the special case `d = 3`, `A = R·S`,
`V = (opacity, SH coefficients)`, read out through the volume-rendering integral; the SRM keeps the
dimension, the mother density, the affine `A`, and the readout free. Solving a PDE is a new readout
on the same representation.

Both an SRM and an MLP are universal approximators, so the argument here is never *what* is
representable. It is *how the function is assembled*. An MLP's unit is a global ridge `σ(wᵀx+b)`
whose active half-space extends across the whole domain; an SRM's unit is a localised bump. Three
consequences bear on (1): a parameter update is local, so credit assignment is regional rather than
global; the basis is *learned*, so centres migrate to where the field is hard rather than sitting on
a fixed grid as in an RBF or Fourier basis; and `∇f`, `∇²f` are available in closed form and smooth
to all orders.

### 3.3 Splats on a manifold: the wrapped Gaussian

A Gaussian is not defined on a curved space — `x − B` is not a manifold operation. The construction
we use is the standard **wrapped** (pushforward) density: stand at the splat's centre `μ`, where the
tangent space `T_μM` genuinely is a vector space; put the Gaussian there; push it onto `M` along
geodesics with `ψ = Exp_μ`. With `ψ⁻¹ = Log_μ`,

```
 N_w(x; μ, A) = N(Log_μ(x); 0, A Aᵀ) · |det ∂Log_μ/∂x|,                                   (3)
```

and the SRM on `M` is `f(x) = Σ_j V[j] · N_w(x; B[j], A[j])`. Designing the model on a new manifold
therefore needs exactly the diffeomorphism, its inverse, and its Jacobian — nothing else.

Two properties make (3) the right object for a value function rather than merely a legal density.

**Gauss's lemma makes the splat a geodesic distance chart.** `Exp_μ` is a radial isometry, so
`‖Log_μ(x)‖ = d_M(μ, x)`: Euclidean distance in the tangent chart *is* geodesic distance on the
manifold. Consequently the exponent of (3) is `−½ d_A(μ,x)²` for the anisotropic norm induced by
`A`, and the splat is a smooth, anisotropic geodesic-distance chart centred at `μ`. We normalise
every `Log` implementation to this convention (Euclidean norm = geodesic distance), which is what
keeps `A` in Riemannian units on every manifold and lets one evaluator serve all of them.

**A soft-min of distance charts is the shape of an optimal value function.** An arrival-time field is
a minimum over routes: `T(x) = min_i (c_i + d(x, x_i))` when the routes are wavefronts through
intermediate points `x_i`. A log-sum-exp of Gaussian-shaped distance charts converges to exactly that
minimum as the temperature falls, so the mixture in (2) is not an arbitrary basis for `T` — it is a
smoothed form of the object `T` already is. This is also the reason the roadmap prior of §3.6 takes
a soft-min form.

**The Jacobian factor is the curvature, and it is one expression evaluated at four constants.**
`|det ∂Log_μ/∂x|` is the reciprocal of the Jacobi-field volume element of `Exp_μ`; for constant
sectional curvature `K` it reduces to a single closed form. Substituting `sin → id → sinh` is the
entire per-manifold change in the density:

| manifold | `K` | `dim` (storage) | `tangent_dim` | `jac_factor` |
|---|---|---|---|---|
| torus `Tⁿ` | 0 | `n` | `n` | `1` |
| sphere `Sⁿ` | +1 | `n+1` | `n` | `(θ/sin θ)^{n−1}` |
| hyperbolic `Hᵈ` | −1 | `d` | `d` | `(r/sinh r)^{d−1}` |
| `SO(3)` | ¼ | 4 | 3 | `((θ/2)/sin(θ/2))²` |

Geodesics converge on the sphere (`θ/sin θ > 1`, density concentrated), diverge in hyperbolic space
(`r/sinh r < 1`, density spread), and do neither on the flat torus. `SO(3)` under its bi-invariant
metric is locally isometric to a round `S³` of radius 2, which is where its `K = ¼` and its
half-angle form come from.

Note `tangent_dim` in the table. The splat's covariance `A` is always square at the manifold's
**intrinsic** dimension, never at the storage dimension: a splat on `SO(3)` is a 3×3 Gaussian in
`so(3) ≅ R³` even though the point is stored as a unit quaternion. Nothing is embedded, padded, or
lifted.

### 3.4 The port is three functions

The SRM backend (`srms/methods/backends/srm.py`) is **byte-identical** across every manifold in this
paper. Its entire geometric interface is

```
 env.log_map(μ, x)     → tangent coordinates, size tangent_dim, ‖·‖ = geodesic distance   (density)
 env.jac_factor(μ, x)  → |det ∂Log_μ/∂x|, the curvature correction                        (density)
 env.metric_inv(x)     → g^{ij}(x) as a quadratic form on ambient gradients               (loss)
```

Everything else a backend touches (`sample_domain`, `sdf`, `dim`) is scene bookkeeping, not geometry.
The backend contains no `sin`, no `cos`, and no read of the domain extent outside a docstring:
periodicity on the torus arises from the `wrap` inside its log map, not from an input transform.

Two of the three functions are density-side and one is loss-side, and that split is load-bearing for
the argument in §3.5: even a model that learned its own input encoding would still need the true
`g⁻¹` to write down (1).

Measured size of the per-manifold delta:

| manifold | geometry code | `jac_factor` | `metric_inv` |
|---|---|---|---|
| torus (`K=0`) | 15 lines | `1` | `I` |
| sphere (`K=+1`) | 27 lines | `(θ/sin θ)^{n−1}` | `I − x xᵀ` |
| hyperbolic (`K=−1`) | 35 lines | `(r/sinh r)^{d−1}` | `((1−‖x‖²)/2)² I` |

On an *embedded* manifold, `metric_inv` is a tangent-plane projector rather than a positive-definite
matrix: the field is only ever evaluated through ambient coordinates, and the intrinsic gradient of a
restriction to an embedded submanifold is the ambient gradient projected onto the tangent space,
independently of how the field is extended off the manifold. `grad ᵀ g⁻¹ grad` therefore recovers the
correct intrinsic `‖∇T‖²` with no chart and no reparameterisation.

### 3.5 Why no input encoding, stated fairly

The honest claim is **not** that networks cannot learn on manifolds. It is that a coordinate-consuming
model must choose an input encoding, that this choice has **no correctness criterion**, and that its
failure is silent — whereas the SRM's three functions are *determined* by the manifold and each one
is checkable against an exact identity before training (§3.8).

Measured distortion of feature distance against geodesic distance:

| manifold | the torus's sin/cos encoding | that manifold's canonical embedding |
|---|---|---|
| torus | 1.6× | 1.6× (the same thing) |
| sphere | **14.3×** | 1.5× (raw ambient `R³`) |
| hyperbolic | **26.3×** | 4.1× (raw ball coordinates) |

Two conclusions, both needed for the claim to survive a hostile reading:

1. *A correct fixed encoding is fine.* "Feed the canonical embedding" is a genuine uniform recipe,
   and we run it as the comparator. The failure we measured was reusing the torus's periodic encoding
   on other manifolds — and it failed silently: the north and south poles of `S²` mapped to features
   **1.7e-7** apart despite being `π` apart, feature distance correlating with geodesic distance at
   only 0.417. The code ran, produced numbers, and was wrong.
2. *`SO(3)` is a theorem, not a preference.* Zhou et al. (CVPR 2019) prove no continuous
   representation of `SO(3)` exists in four or fewer dimensions, so quaternion (4-D) and Euler (3-D)
   inputs are provably discontinuous and a ≥5-D encoding is mandatory. Learning the encoding does not
   help: the obstruction is topological, and a learned encoding is still a continuous function of its
   input. The splat needs no encoding at all — `A` is 3×3 in `so(3)`.

The distinction we draw is **verification vs. validation**. The splat's pieces are determined, so one
checks *equality* and gets pass/fail at 1e-14. An encoding is chosen, so one can only measure
distortion and pick a bar. Both are testable; only one has a right answer.

Scope, stated rather than left to be found: **hyperbolic space does not discriminate between the two
models** and is not presented as if it does. `Hᵈ` is contractible — one global chart — so a correct
MLP needs no encoding trick there, and curvature enters only through `metric_inv`, which is loss-side
and representation-agnostic. Hyperbolic is in this paper as a generality demonstration and as the site
of a negative result (§3.9), not as a comparative win.

### 3.6 The self-supervised loop

Nothing in training sees a solved field. A method may read the *scene* (`slowness`, `sdf`, obstacle
geometry), the *analytic* free-space geodesic, and its own collocation samples. The fast-marching
solution exists only to score the result afterwards, and that separation is enforced by test rather
than asserted (§3.8).

**Field parameterisation.** We write the field as a known base times a learned correction, following
NTFields:

```
 T(x) = base(x) / τ(x),     base(x) = d_M(x, x_s),     τ = τ_min + (1−τ_min)·σ(f(x) + b) ∈ (0, 1]. (4)
```

Three things come for free. `T(x_s) = 0` holds identically, because `base(x_s) = 0` — the source
singularity is built in rather than fitted. `T ≥ base` holds by construction, and since `s ≥ 1` and
`base` is the shortest free-space path length, `base` is a valid *lower bound* on the true arrival
time: (4) parameterises exactly the admissible range, and `τ → 0` supplies the blow-up near
obstacles. And the model's initialisation `f ≡ 0` (splat weights start at zero) puts `τ` near 1, i.e.
`T ≈ base`, a sensible free-space field rather than a random one.

This factorisation is also what makes the objective trainable at all with this representation, for a
reason we measured rather than assumed — see §4.4.

**Objectives.** The metric enters the loss in exactly one place: the Riemannian gradient norm
`‖∇T‖_{g⁻¹} = √(∇Tᵀ g⁻¹ ∇T)`, which is the model's predicted slowness. Writing
`q(x) = ‖∇T‖_{g⁻¹}/s(x)`, equation (1) is `q ≡ 1`, and every objective below is a penalty on `q`
that is minimised only there. Three arms are implemented, holding the field (4) fixed across all of
them so a comparison isolates the objective. The curved-manifold results in §5 use the first two; the
third is at present measured on the flat torus only (§4.7):

- **No supervision (`ntfields`)** — the PDE residual alone, in NTFields' isotropic form
  `|1 − √q| + |1 − 1/√q|`: symmetric under over/under-prediction and L1-like near the optimum, so
  high-slowness obstacle points cannot dominate the free-space bulk. No planner, no roadmap, no
  demonstrations.
- **Weak supervision (`hntfields`)** — the same residual in H-NTFields' squared one-directional form
  `(√q − 1)²`, plus a finite-scale Bellman consistency term, an obstacle-normal alignment term, and a
  hinge onto travel-time bounds `[T_lb, T_ub]` from a **sparse sphere-packing roadmap**, all under a
  causality weight `exp(−λ_C T)`. The roadmap supplies *bounds*, never target values; it is not the
  planner.
- **Roadmap-prior refinement (`weak_supervision`, ours)** — `T = base_RRT*(x) · exp(f(x))` where
  `base_RRT*` is a differentiable soft-min cost-to-come over an RRT\* tree, and the PDE residual is
  free to pull the field *below* the suboptimal prior it started from. Mesh-free end to end.

**Causal weighting.** The residual in (1) constrains the *magnitude of the gradient* and nothing
else, so it fixes slope but not level; selecting the viscosity solution needs causal information,
because the value at a point is set by integrating slowness along the optimal route back to the
source. We use the source-outward curriculum of causal PINNs: order collocation points by geodesic
distance from `x_s` and weight each residual by `exp(−rate · upstream)`, where `upstream` is the
cumulative stop-gradient residual mass of all nearer points, annealing `rate → 0` so the full
residual is enforced by the end.

### 3.7 The model chooses its own capacity

Because a splat is local, "add capacity where the residual is" has a referent. Between optimisation
steps we prune splats whose weight falls below a threshold and spawn new ones at the highest-residual
free-space samples, capped by a runaway backstop. A fixed-width network has no analogue — every
weight affects the whole domain — so the MLP backend's `adapt` is necessarily a no-op, and this is a
structural difference rather than a tuning advantage.

The stopping rule is one line of arithmetic, and it is designed so a single threshold transfers across
manifolds, seeds and scenes. At each densify pass, compare the mean training residual since the
previous pass against the residual before it, and divide the fractional improvement by the number of
splats that bought it:

```
 gain_per_splat = (L_prev − L_now) / L_prev / splats_added.                                (5)
```

Growth stops when (5) falls below `densify_min_gain` on **two consecutive** passes. Dividing by
`L_prev` makes the test independent of the residual's scale and dividing by `splats_added` makes it
independent of the spawn schedule, which a fixed "improved by less than x%" tolerance is not. The
two-consecutive requirement is not conservatism for its own sake: the measurement window sits
immediately after a densify pass, which is exactly when freshly spawned splats perturb the
optimisation and the residual transiently *rises*. Measured on `T²`, a single-reading rule stopped at
1194 splats on a **negative** gain (−6.6e-4), where the model went on to reach 2676 splats and a
better RMS (0.2485 vs 0.2527) when allowed to continue. Two other stops apply, whichever comes first:
the final `densify_freeze_frac` of training always runs at fixed structure so the model converges
against a settled basis, and `max_splats` is a backstop rather than a target. Every run reports
*which* stop fired, so it explains why it chose its size.

Two implementation points are what make this stable, both recorded in §4.3: the optimizer moments are
grown surgically rather than reset, and each splat's covariance singular values are floored.

### 3.8 Verification before measurement

The claim in §3.4–§3.5 is that the SRM's per-manifold delta is *determined*, so getting it wrong is a
caught error rather than a silent one. That is only worth asserting if it is enforced. Three test
layers run before any number in this paper is read.

**(a) Exact geometric identities — no ground truth, no solver, no tuning.** Four per manifold, each
with an exact expected value:

| check | identity | what it catches |
|---|---|---|
| round trip | `Exp_μ(Log_μ(x)) = x` | log and exp are not actually inverses |
| isometry | `‖Log_μ(x)‖ = d_M(x, μ)` | tangent coordinates in chart units rather than Riemannian units |
| jacobian | `jac_factor = |det ∂Log_μ/∂x|` (autodiff) | **the wrong curvature term** — `sin` vs `sinh` |
| eikonal | `‖∇ d_M(·, x_s)‖_g = 1` | a wrong metric; this is the exact residual the solver minimises |

**All 20 pass across five environments** (torus, sphere, `SO(3)`, Poincaré `H²`, Lorentz `H²`) in
float64: 12 of the 20 at or below 1e-12, worst case 2.3e-10 (the Lorentz chart's Jacobian), against a
tolerance of 1e-5 — i.e. exact to within accumulated float64 round-off, not merely small. The checks are written manifold-agnostically: the orthonormal frame the
jacobian check needs is derived from `metric_inv` by eigendecomposition (`E = g^{−1/2}`, null
directions dropped) rather than hand-coded per manifold, so the test file itself contains no
per-manifold branch — hand-coding one frame per manifold would undercut the very claim being tested.
They earn their keep: the suite caught a newly contributed Lorentz-hyperboloid environment failing two
identities (1.9e-3, 5.8e-4) on first contact, and caught two clipping defects of our own (§4.2).

**(b) The wrapped Gaussian is a density, and its shortfall has a *named cause*.** Monte-Carlo
`∫ N_w dvol` (400k samples) against the closed-form contained mass:

| manifold | σ = 0.4 | σ = 1.2 | predicted (σ=1.2) | what bounds the domain |
|---|---|---|---|---|
| torus | 1.0031 | 0.9828 | 0.9824 | half-period cut locus |
| sphere | 1.0051 | 0.9669 | 0.9675 | antipodal cut locus |
| hyperbolic | 0.9989 | 0.9508 | 0.9507 | **domain truncation — our choice, not curvature** |

Read by cause, not by size. Torus and sphere lose tail mass past a **cut locus**, a genuine limit of
the single-image wrapped Gaussian there. `Hᵈ` has no cut locus at all (`Exp` is a global
diffeomorphism), so (3) is exact for any σ; its shortfall is entirely the truncation wall we chose.
Hyperbolic therefore shows the largest measured shortfall while having the only exact representation
— which is why the check reports cause alongside magnitude instead of ranking manifolds by a number.

**(c) No training path can reach ground truth**, checked three ways rather than asserted: statically
(no module under `srms/methods` may so much as name `ground_truth`); by interface (every `env.<attr>`
each `solve` touches must lie inside an allow-list of self-supervised quantities — scene, closed-form
geometry, sampling); and dynamically (`env.ground_truth` is monkeypatched to raise and a short solve
is run for every strategy × manifold pair). The dynamic check is the one that catches the real failure
mode: a solver handed ground truth through a callback rather than referencing it by name.

**(d) Two denominators, without which an RMS is uninterpretable.** First, the ground truth has its own
discretisation error, and each manifold's marcher is a different marcher — no solver RMS below these
is meaningful:

| manifold | marcher | RMS vs analytic (res 120) | mean rel. | at res 240 |
|---|---|---|---|---|
| torus | periodic, isotropic, flat chart | 4.65e-2 | 1.96% | 2.74e-2 |
| sphere | anisotropic, geodesic-polar `(θ, ψ)` grid | 2.13e-2 | 1.44% | 1.46e-2 |
| hyperbolic | Euclidean marcher, conformally rescaled | 4.04e-2 | 2.56% | 2.21e-2 |

All three converge first-order, so what remains is ordinary discretisation error rather than a
modelling error. Second, and more importantly, **`T = base` is the score for learning nothing** — it
is also the model's own initialisation — so every RMS is reported against it. This is what makes
results comparable across scenes: a scene with 7.7% obstacle coverage and a median detour factor of
1.010 has `RMS(base) = 0.2566`, against 1.4360 on ours, so a low RMS there is largely scene and not
model.

### 3.9 What the local residual cannot do

One negative result belongs in the Method because it is a property of the objective, not of the
representation, and it shapes what §3.6's three arms are for.

**A purely local Eikonal residual under-determines the field, and negative curvature is where it hurts
most.** Behind an obstacle, `T = base` has gradient magnitude exactly 1, and in locally free space
`s = 1` — so the residual is **zero**. Equation (1) is pointwise: it cannot see that the geodesic
reaching that point is blocked upstream. `T = base` is a valid solution of the local PDE and a wrong
solution of the problem. Measured on `H²`, where the true field needs a 2.83× detour factor:

| geodesic distance from source | `τ` used | `τ` needed |
|---|---|---|
| 0.3–1.0 | 0.792 | 0.675 |
| 1.0–2.0 | 0.966 | 0.713 |
| 2.0–3.0 | **0.972** | **0.682** |

`τ` initialises at 0.982 and never moves in the far field. Hyperbolic suffers most because volume
grows like `e^{(d−1)r}`, so the blind shadow region is most of the manifold. This is the published
statement of the difficulty as well: H-NTFields reports that PDE-only solvers "collapse into local
minima or underestimate long-range travel times without additional structural guidance", and that
"simply increasing the number of samples cannot resolve local minima". It is why weak supervision
exists, and why the representation claim and the supervision claim have to be separated.

---

## 4. Implementation Details

### 4.1 Per-manifold geometry, in closed form

All three functions of §3.4 are closed-form; none is fitted or tabulated.

**Torus `Tⁿ`** (`dim = tangent_dim = n`, chart `[−π, π)ⁿ`). `Log_μ(x) = wrap(x − μ)`,
`Exp_μ(v) = wrap(μ + v)`, `jac_factor = 1`, `metric_inv = I`, `d(x, y) = ‖wrap(x − y)‖`. Periodicity
is a property of the log map, so the model never sees a periodic input transform. Obstacles are
geodesic balls in angle space.

**Sphere `Sⁿ`** (`dim = n+1` unit vectors, `tangent_dim = n`). `Log_μ(x) = θ·(e_⊥ᵀ E_μ)` with
`θ = d(μ,x)`, `e_⊥` the unit tangent direction, and `E_μ` an orthonormal frame of `T_μSⁿ` built as a
Householder reflection carrying a reference axis to `μ` (jit/grad-safe: static shapes, both reference
choices computed and selected with `where`, no dynamic indexing). `Exp_μ(v) = cos‖v‖ μ + sin‖v‖ v̂`
along the great circle. `jac_factor = (θ/sin θ)^{n−1}`; `metric_inv = I − x xᵀ`. Obstacles are
geodesic caps.

**Hyperbolic `Hᵈ`, two charts.** *Poincaré ball* (`dim = tangent_dim = d`, `‖x‖ < 1`), conformal
metric `g = λ(x)² δ`, `λ = 2/(1−‖x‖²)`, everything closed-form from Möbius addition `⊕`:
`d(x,y) = 2·artanh‖(−x) ⊕ y‖`, `Log_μ(x) = d(μ,x)·u/‖u‖` with `u = (−μ) ⊕ x`,
`Exp_μ(v) = μ ⊕ (tanh(‖v‖/2)·v̂)`, `jac_factor = (r/sinh r)^{d−1}`,
`metric_inv = ((1−‖x‖²)/2)² I`. *Lorentz hyperboloid* (`dim = n+1`, `⟨x,x⟩_η = −1`) is implemented
independently and verified to the same tolerances, so no result depends on the chart.

**`SO(3)`** (`dim = 4` unit quaternions, `tangent_dim = 3`). `Log_μ(x)` is the axis-angle vector of
`μ⁻¹x`, with the relative quaternion negated when its scalar part is negative; `geodesic` is
`2·arccos|⟨q₁,q₂⟩|`, where the absolute value *is* the `S³/±1` quotient. `jac_factor =
((θ/2)/sin(θ/2))²`, `metric_inv = ¼(I − q qᵀ)`. Two conveniences relative to the other manifolds:
`jac_factor` is bounded on the whole group (it rises only to `(π/2)² ≈ 2.47` at the cut locus, where
the sphere's `θ/sin θ` diverges), and `SO(3)` is compact, so unlike `Hᵈ` there is no truncation to
declare. One structural convenience is worth naming because it is the reverse of `S²`: a Lie group is
parallelizable, so left-translating a fixed basis of `so(3)` gives a globally smooth frame and an
anisotropic `A` means the same thing everywhere; on `S²` the hairy-ball theorem forbids any global
continuous frame, and our Householder construction jumps by 33.7° at `|μ₀| = 0.9`.

**Two declared modelling choices for `Hᵈ`**, neither forced by the geometry (`Hᵈ` is unbounded with
volume growing like `e^{(d−1)r}`): the workspace is truncated at `‖x‖ ≤ 0.9` with the rim entering
`sdf` as an ordinary wall (hyperbolic diameter ≈ 5.9, comparable to `T²`'s `π√2 ≈ 4.4` and `S²`'s
`π ≈ 3.1`, so scale-sensitive hyperparameters transfer untouched); and sampling is **chart-uniform**
rather than volume-uniform, because volume-uniform sampling piles essentially every collocation point
into the thin annulus at the rim, and chart-uniform matches the Cartesian grid the RMS is scored on.

**Scenes.** Identical construction on every manifold: `num_obstacles` geodesic balls/caps of
geodesic radius drawn reproducibly from a seeded RNG and rejected if within 0.3 of the source, with a
smooth slowness `s(x) = 1 + (s_max − 1)·σ(−sdf(x)/w)` — so `s ≈ 1` in free space and rises to
`s_max = 10` inside an obstacle. Obstacles are *soft*: the field is finite everywhere, and there is no
hard feasibility constraint in the loss.

### 4.2 Numerical care where the geometry is singular

Every distance on an embedded manifold is naïvely `arccos`/`arccosh` of an inner product, and both
have unbounded derivatives at their singular argument, so the usual defence is to clip. Clipping is
wrong here in a way that is easy to miss: it puts a **floor** on the distance. Both the sphere and the
Lorentz hyperboloid reported a nonzero distance from a point to itself (4.47e-2), which made every
distance below 0.045 unrepresentable — precisely where the field is anchored at the source.

The fix is the half-angle form, whose derivative is bounded at zero so no clip is needed. Same
identity (`‖x−y‖² = 4 sin²(d/2)` and its hyperbolic twin), same fix, both manifolds — and the same
move `SO(3)` already used (`arctan2` rather than `arccos`):

| manifold | before | after |
|---|---|---|
| sphere | `arccos(clip⟨x,y⟩)` | `2·arcsin(‖x−y‖/2)` |
| hyperboloid | `arccosh(clip(−⟨x,y⟩_η))` | `2·arcsinh(‖x−y‖_η/2)` |

`d(x,x)` went 4.47e-2 → 0. Floors remain only where a quantity appears as a *divisor* (`sinh d`, the
norm of a perpendicular component), never inside the distance itself. Elsewhere, branch-free JAX-safe
forms are used throughout: `where` with a substituted safe argument in the non-selected branch (so
neither branch's gradient can be NaN under the nested differentiation the PDE loss needs), Taylor
series near removable singularities, and `maximum` rather than `where` on normalisation denominators.

### 4.3 Optimisation

**Retraction.** A wrapped Gaussian is defined only for a centre lying *on* the manifold, but `B` is an
unconstrained optimizer variable. On `S²` an ambient gradient step walks it straight off: measured
0.105 off the unit sphere by step 65, at which point the Householder frame construction — which
assumes a unit vector — stops being orthonormal and the density diverges. We retract in a `post_step`
hook applied to the **parameters** after each optimizer step, `B ← wrap_point(B)`; this is the
projection retraction, which agrees with the exponential retraction `Exp_B` to `O(‖step‖³)` (measured
3.3e-4 at step 0.1, 4.3e-7 at 0.01, at the float32 floor by 0.001 — indistinguishable at this learning
rate). With it, `‖B‖` stays exactly 1.0000 for the whole run.

Where the retraction is applied matters, and the wrong choice is instructive. A first attempt
normalised inside the density evaluation instead. That makes the loss exactly invariant to `‖B‖` —
which sounds strictly better, and makes `dL/dB` exactly tangent (measured radial component 4e-10) —
but with nothing pinning the norm, AdamW's weight decay shrinks it as `(1−lr·wd)^step` (measured
1.00 → 0.33 over 2400 steps), and since a centre's angular step is `≈‖ΔB‖/‖B‖`, the effective step
size *grows* 3× and training diverged at step 2481. **Scale-invariance alone is not a retraction.**

**Scale floor — the stabiliser that makes adaptive capacity work.** A sum of Gaussians cannot
represent the true Eikonal kink at obstacle boundaries and the cut locus, so unconstrained
optimisation chases it by driving a splat's covariance toward zero — an effectively infinite `‖∇T‖`
spike (up to ~1e7 measured) which then blows up any speed-match residual. We floor each splat's SVD
singular values at `scale_floor` after every step, which makes that collapse impossible.

**Optimizer surgery on densification.** When splats are pruned and spawned, the optimizer state is
grown surgically: every leaf whose leading axis indexes splats is sliced to the survivors and
zero-padded for the spawns, with scalar leaves (Adam's step count) passing through untouched. A plain
`optimizer.init()` reset would re-kick every already-converged splat, which is what made earlier
densification runs diverge.

**Settings.** `lr = 3e-3` and gradient clipping at global norm 1.0 on every strategy. The three
ported baselines (`ntfields`, `pntfields`, `hntfields`) use **AdamW, `weight_decay = 0.1`**, following
the released NTFields implementation rather than our own preference; our own two strategies
(`eikonal`, `weak_supervision`) use plain Adam, so the weight-decay coupling described above under
*Retraction* applies to the baseline arms only. The clip is not cosmetic — the symmetric residual's `1/√q` term has a genuine
singularity as `q → 0`, and without clipping a stray step NaNs the run. Collocation is **resampled
every step** (2048 points, uniform over the manifold, excluding a ball of radius 0.25 around the
source), rather than drawn once into a fixed dataset. float32 for training; float64 only for the exact
identity checks of §3.8, which are correctness tests rather than training.

Reported parameter counts are the total trainable scalars, `k·(p + m² + n)` for `k` splats with
`tangent_dim = m` and storage `dim = n`, so SRM and MLP can be compared at matched budget.

### 4.4 Two defects worth recording, because they are in the path of the comparison

**The unfactored field cannot train from its own initialisation.** With `V = 0`, a plain (unfactored)
`T` starts at `T ≡ 0`, where `∇T = 0` and `d‖∇T‖/dparams` collapses to 0/0. `T ≡ 0` is an **exact
stationary point** of the PDE term, so only the boundary ring has any gradient at all:

| field | `‖grad PDE‖` at `V=0` | PDE loss at `V=0` |
|---|---|---|
| unfactored | **0.000e+00** | 10.975 |
| factored `T = base/τ` | 2.58e-02 | 0.495 |

Measured on `T²`; unchanged by turning causal weighting off and by 64 vs 384 splats. The factored
strategies are immune because `base` is the analytic geodesic distance, nonzero with nonzero gradient
at `V = 0`. This is why (4) is not merely a convenience, and why the planner-free arm of this study
uses the factored PDE-only objective.

**The sign convention of the residual.** `slowness` is a cost per unit length (`≥ 1`, rising inside
obstacles), so `T = ∫ s dl` and therefore `‖∇T‖ = s`. We settled this empirically rather than by
reading: finite-differencing the fast-marching ground truth on `T²` gives, inside obstacles, a median
`‖∇T‖/s` of 0.998 against `‖∇T‖·s` of 84.9.

### 4.5 Baselines, ported with their deviations declared

The three published baselines (NTFields, P-NTFields, H-NTFields) are ported from the papers *and*
read against the released implementations, with every deviation stated in the module that makes it.
The deviations shared by all three are: a **fixed source** (this repo learns `T(x)` with the goal
ranging over the whole manifold, which is what makes the field a value function; the papers learn a
two-point `T(q_s,q_g)`, so only one endpoint's loss exists here); the **scene's own** smooth slowness
in place of their clipped obstacle-distance ramp (a property of the scene, shared identically by
every method, so it does not bias the comparison); **per-step resampled** collocation; and this repo's
`srm`/`mlp` backend in place of their ResNet-plus-CNN encoder — that substitution being the point of
the comparison rather than an accident of it.

Three deviations are ours and are consequences of holding the field form (4) fixed, so they are worth
naming explicitly:

- *H-NTFields' causality weight is stop-gradient by default.* The released code does not detach it,
  which is safe there because `T` is predicted directly with a bounded quasimetric head. With
  `T = base/τ`, `τ → 0` makes `T → ∞` reachable, so an attached weight pays the optimizer to inflate
  `T` and zero out its own loss. Measured on `T²` at 800 steps: attached RMS 4.34 / max|err| 170,
  detached 0.63 / 6.4. A flag restores the literal released behaviour.
- *The Bellman term follows the released code, not the paper's prose*, since that code produced the
  published numbers: the step is `Δt·S⋆·∇T` rather than the unit-normalised `u⋆Δt`; the entire target
  is computed under `no_grad`, not only the policy direction; and the term is masked wherever `T` is
  below the step's own time cost, i.e. within one step of the source.
- *The roadmap's node density had to be adapted to an open scene.* The lower bound is only a bound if
  the roadmap's own shortest path is near-optimal. The paper packs 5,000 nodes into cluttered indoor
  scenes; on an open 3-obstacle torus the maximal free spheres are enormous, the packing saturates
  after ~78 nodes, its paths run ~1.45× optimal, and the "lower" bound sat *above* the true travel
  time at **94%** of nodes — training with it was 6–8× worse than the PDE-only ablation on both
  backends. Capping the free-sphere radius at 0.3 rad brings the packing to ~460 nodes with paths
  optimal to within measurement noise. This is an adaptation to an open scene rather than a change of
  mechanism, but it is worth remembering that the node count needed for a valid bound grows with
  dimension — the same curse the roadmap prior was meant to dodge.

### 4.6 Evaluation protocol

Trained fields are scored against a dense fast-marching solution at resolution 240, computed only
*after* training returns. Ground truth on all four manifolds comes from **one** solver, via per-cell
per-axis grid spacing plus an environment-supplied topology hook, with two consequences worth stating.
Hyperbolic needs no hyperbolic solver: the Poincaré metric is conformal, so `‖∇T‖_g = s` is
identically `‖∇T‖_euclid = s·λ` with `λ = 2/(1−‖x‖²)`, and this is dimension-independent, so it holds
in 3-D too. And `SO(3)`'s grid edges are **identifications, not boundaries** — the shell at `r = π` is
an `RP²`, since the quaternion there satisfies `[0,u] = [0,−u]`; omitting that gave 14% error growing
with radius, supplying it gives 4.4%.

We report RMS and max error against ground truth, the number of splats the model chose, and — as the
primary normalisation — the ratio against `RMS(T = base)`, the do-nothing score (§3.8d). One caveat we
record rather than exploit: with the source placed exactly at the sphere's lat-long pole the marcher's
error is 1.7e-7 rather than 2.1e-2, because the obstacle-free field becomes a coordinate line; the
floors quoted in §3.8d are measured with an off-pole source and are the honest ones for a general
scene.

### 4.7 What the current evidence does and does not exercise

Stated here so the reader does not have to infer it.

- **The metric is not yet curved.** Every result in this paper has `metric_inv ∈ {I, I − xxᵀ,
  conformal scalar}`. The interface admits the arm's anisotropic inertia metric `M(θ)⁻¹` with no
  change to splat placement, and that is where an anisotropic covariance and intrinsic placement
  should genuinely matter — but it is **interface headroom, not a demonstrated capability**.
  **[PENDING]**
- **`SO(3)` has verified geometry and ground truth but no training run.** It is the sharpest case for
  §3.5's argument and is currently argued from a theorem plus exact identities, not from a fit.
  **[PENDING]**
- **The RRT\*-prior refinement arm has no curved-manifold numbers.** The soft-min roadmap base and
  its Eikonal refinement are implemented for any `Environment`, but the recorded results for that arm
  are torus-only, from the earlier line of work. §3.6's third bullet is a method description, not a
  measured claim on `S²`/`H²`. **[PENDING]**
- **One check in the verification suite has a wrong expected value, not a wrong model.** `SO(3)`'s
  mass check compares against the *2-D* injectivity-domain formula `1 − exp(−π²/2σ²)`, which is the
  sphere's; `SO(3)` is 3-dimensional, so the correct closed form is the χ₃ mass inside the radius-π
  ball. At σ=1.2 that predicts 0.9233 against a measured 0.9225 (Δ = 8e-4, inside the Monte-Carlo
  floor), where the 2-D formula predicts 0.9675 and fails by −4.5e-2. The 20 exact identities — the
  load-bearing checks — all pass; this is the expected value, not the density. **[PENDING: fix]**
- **Sub-Riemannian is out of scope.** `SE(2)`'s log map, Jacobian and a sub-Riemannian anisotropic
  `A` (`σ_lat ≪ σ_fwd`) exist in the library, but there is no environment and no result.
- **`S³ ≠ SO(3)`.** Both are 3-dimensional, but `SO(3) = S³/±1 = RP³` is not simply connected:
  different cut locus, volume (`8π²` vs `2π²`), curvature (`¼` vs 1) and grid topology. The
  `S² → S³` scaling story and the `SO(3)` Lie-group story are separate claims and are not merged.
