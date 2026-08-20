# Paper notes — SRM for time-to-go on Riemannian manifolds

Drafting notes. Every number here is measured. Anything not measured is marked **[PENDING]** so it
can't drift into a claim.

---

## Thesis

A splat model built from wrapped Gaussians learns optimal time-to-go on a Riemannian manifold with
**no coordinate projection and no learned input encoding**. The per-manifold code is the manifold's
own log map, its Jacobian, and its metric — objects with textbook definitions and exact correctness
tests.

---

## Read every RMS against the do-nothing baseline

The single most important framing decision. `T = base` (the free-space geodesic distance) is also the
model's own initialisation, so it is the score for learning **nothing**:

| manifold | method | RMS | do nothing | better by |
|---|---|---|---|---|
| torus | weak supervision | 0.0417 | 0.4871 | **11.7×** |
| sphere | no supervision | 0.0545 | 0.3500 | **6.4×** |
| torus | no supervision | 0.2425 | 0.4871 | 2.0× |
| sphere | weak supervision | 0.2169 | 0.3500 | 1.6× |
| hyperbolic | weak supervision | 0.5334 | 1.3611 | 2.6× |
| hyperbolic | no supervision | 1.0112 | 1.3611 | 1.3× |

Without this denominator a raw RMS is uninterpretable. It is also how to compare against other
people's results: a collaborator's Lorentz-model scene has **7.7%** obstacle coverage and a **median
detour factor of 1.010** — half that domain's true answer *is* the free-space distance — so doing
nothing scores 0.2566 there against 1.4360 on ours. A low RMS on that scene means very little.

Ground truth is itself accurate to ~1–1.5% at resolution 240; no RMS below that is interpretable.

---

## Contributions

**1. A manifold-general approximator.** A splat is a Gaussian in the tangent space `T_μM` pushed onto
the manifold by `Exp_μ`. Porting needs three functions: `Log_μ(x)`, `|det ∂Log_μ/∂x|`, and `g⁻¹(x)`.
Nothing is embedded, padded or encoded; the covariance is `d×d` at the manifold's true dimension.

**2. The per-manifold change is one expression, evaluated at each curvature.**

| manifold | K | `jac_factor` |
|---|---|---|
| torus Tⁿ | 0 | `1` |
| sphere Sⁿ | +1 | `(θ/sin θ)^(n−1)` |
| hyperbolic Hⁿ | −1 | `(r/sinh r)^(d−1)` |
| SO(3) | ¼ | `((θ/2)/sin(θ/2))²` |

**3. Getting it wrong is a caught error, not a silent one.** Four exact identities per manifold —
`Exp(Log(x)) = x`, `‖Log‖ = geodesic`, `jac_factor` vs autodiff, `‖∇d‖_g = 1` — with no ground truth
and no training. **20/20 pass across five environments** (torus, sphere, SO(3), Poincaré H², Lorentz
H²) at 1e-11 to 1e-16.

They earn their keep: the suite caught a collaborator's newly added Lorentz environment failing two
identities (1.9e-3 and 5.8e-4) on first contact, and caught two clipping defects of our own (below).

**4. The model chooses its own capacity; a fixed-width network cannot.** Splats are added where the
residual is high; growth stops when a densify pass buys less than a threshold fractional residual
reduction *per splat added*, on two consecutive passes. Dividing by both the loss and the splats added
makes one threshold transfer across manifolds and scenes. An MLP has no analogue — every weight
affects the whole domain, so "add capacity here" has no referent.

**5. Ground truth on four manifolds from one solver**, via per-cell per-axis grid spacing plus an
environment-supplied topology hook. Two consequences worth stating:
- **Hyperbolic needs no hyperbolic solver.** The Poincaré metric is conformal, so `‖∇T‖_g = s` is
  identically `‖∇T‖_euclid = s·λ`, `λ = 2/(1−‖x‖²)`. Dimension-independent, so it holds in 3-D too.
- **SO(3)'s grid edges are identifications, not boundaries.** The shell at `r = π` is an **RP²**
  (the quaternion there is `[0,u] = [0,−u]`). Omitting that gave 14% error growing with radius;
  supplying it gives 4.4%.

---

## Method in plain terms

A Gaussian isn't defined on a curved space. So: stand at the splat's centre `μ`, where the tangent
space *is* flat; put the Gaussian there; wrap it onto the manifold with `Exp_μ`, which walks along
geodesics; correct the density for geodesics converging (sphere) or diverging (hyperbolic) — that
correction is `jac_factor`, and it *is* the curvature.

The field is `T(θ) = base(θ)/τ(θ)` — the known free-space distance divided by a learned slowdown
`τ ∈ (0,1]`. This bakes in the source singularity, which is why it trains where an unfactored field
does not.

---

## The MLP comparison, stated fairly

The honest claim is **not** "networks can't do manifolds". It is that the encoding is a *free choice
with no correctness criterion*, and its failure is silent.

Measured distortion of feature distance against geodesic distance:

| manifold | torus's sin/cos encoding | that manifold's canonical embedding |
|---|---|---|
| torus | 1.6× | 1.6× (same thing) |
| sphere | **14.3×** | **1.5×** (raw ambient R³) |
| hyperbolic | **26.3×** | **4.1×** (raw ball coords) |

So a *correct* fixed encoding is fine — "feed the canonical embedding" is a genuine uniform recipe.
The failure was reusing the torus's on other manifolds, and it failed silently: north and south poles
mapped to features **1.7e-7 apart** despite being π apart.

What survives a hostile reviewer:
- **SO(3) is a theorem, not a preference.** Zhou et al. (CVPR 2019): no continuous representation of
  SO(3) exists in ≤ 4 dimensions, so quaternion and Euler inputs are provably discontinuous. Learning
  the encoding does not help — the obstruction is topological, and a learned encoding is still a
  continuous function of its input.
- **A learned encoding / hashgrid addresses only half the per-manifold surface.** It can replace
  `log_map`, but the Eikonal residual still needs the true metric `g⁻¹`. Get that wrong and you are
  minimising a different equation. It also still needs *some* coordinates to hash, and it makes the
  encoding un-checkable up front (a moving target during training) rather than measurable in seconds.
- **Verification vs validation.** The splat's pieces are *determined* by the manifold — you check
  equality, pass/fail at 1e-14. An encoding is *chosen* — you can only measure distortion and pick a
  bar. Both are testable; only one has a right answer.

---

## Two defects found and fixed, by one idea applied uniformly

Both the sphere and the Lorentz hyperboloid reported a **nonzero distance from a point to itself**
(4.47e-2), because `arccos`/`arccosh` were clipped away from their singular argument. Every distance
below 0.045 was unrepresentable — precisely where the field is anchored at the source.

Fix: the half-angle form, whose derivative is bounded at zero so no clip is needed.

| manifold | before | after |
|---|---|---|
| sphere | `arccos(clip⟨x,y⟩)` | `2·arcsin(‖x−y‖/2)` |
| hyperboloid | `arccosh(clip(−⟨x,y⟩_M))` | `2·arcsinh(‖x−y‖_M/2)` |

`d(x,x)` went 4.47e-2 → 0. Same identity (`‖x−y‖² = 4sin²(d/2)` and its hyperbolic twin), same fix,
both manifolds — and the same move SO(3) already used (`arctan2` rather than `arccos`).

---

## The negative result, which is worth reporting

**A purely local Eikonal residual under-determines the field, and negative curvature is where it
hurts most.**

Behind an obstacle, `T = base` has gradient exactly 1, and in locally free space slowness is 1 — so
the residual is **zero**. The equation is pointwise: it cannot see that the geodesic reaching that
point is blocked upstream. `T = base` is a valid solution of the local PDE and a wrong solution of the
problem.

Measured on H², where the truth needs a 2.83× detour factor:

| geodesic distance from source | τ used | τ needed |
|---|---|---|
| 0.3–1.0 | 0.792 | 0.675 |
| 1.0–2.0 | 0.966 | 0.713 |
| 2.0–3.0 | **0.972** | **0.682** |

τ initialises at 0.982 and **never moves** in the far field. Hyperbolic suffers most because volume
grows like `e^r`, so the blind shadow region is *most of the manifold*.

Everything obvious has been ruled out by measurement:

| candidate | result |
|---|---|
| capacity | ceiling is **0.0354** (below the GT floor); 399 → 1024 splats *widened* the gap 6.1× → 30.8× |
| scene (sampling, wall, domain) | 1.2195 → 1.1849 → 1.1412, all within noise |
| progressive annealing | 1.0112 → 0.9527 (6%) |
| temporal difference | 1.0112 → **1.1071** (worse) |
| chart (Poincaré vs Lorentz) | geometry exact to 1e-14 either way; not the cause |

**Progressive and TD actively harm the sphere**: 0.0545 → 0.3340 and 0.4044, the latter worse than
doing nothing. A collaborator reports both helping an **MLP** on the same objective. Plausible
explanation, untested: annealing moves the target while densification is simultaneously spawning and
pruning, so capacity chases a moving objective — a coupling a fixed-width MLP does not have.

---

## Open / not claimable

- **[PENDING] Hyperbolic does not reach a usable field** (best 0.9527, 1.4× better than nothing).
  Diagnosis is solid; no fix found.
- **[PENDING] SO(3) has verified geometry and ground truth, no training results.**
- **[PENDING] No MLP arm has been run.** Requires the canonical-embedding fix first — the current
  encoding distorts 14–26× on two of three manifolds, so any comparison today is rigged.
- **Weak supervision's sphere row is a roadmap defect, not an objective effect.** `_edge_time` prices
  an edge with 6 samples across a hop up to 1.5 long while the obstacle ramp is 0.1 wide, so it steps
  over the high-slowness band; Dijkstra then preferentially selects the under-priced edges. 50% of
  sphere node bounds fall below the true travel time; 93% of the field is dragged down. Left unfixed
  deliberately — it would change only some manifolds.
- **S³ ≠ SO(3).** Both 3-D, but SO(3) = S³/±1 = RP³, not simply connected; different cut locus,
  volume (8π² vs 2π²), curvature (¼ vs 1) and grid topology. The `S² → S³` scaling story and the
  SO(3) Lie-group story are separate claims.
- **Scope**: Riemannian only. Sub-Riemannian (SE(2)) has log/exp in the codebase but no environment.

---

## Reproducing

```bash
python -m srms.environments.test_manifolds        # 20 identities across 5 environments
python -m srms.environments.test_selfsupervised   # no training path can reach ground truth
python make_tables.py > results/tables.tex        # the two LaTeX tables
```
See `README.org` for the run commands. `results/manifolds.md` holds the full measurement record.
