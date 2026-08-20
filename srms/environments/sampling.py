"""RRT* prior over an Environment: mesh-free, dimension-scalable cost-to-come anchors.

Generic over any ``Environment`` (see ``srms/environments/base.py``) — the algorithm needs only
``sample_domain`` for random configurations and ``geodesic``/``log_map``/``exp_map``/``slowness_np``
for steering and edge costs, so it carries over unchanged across manifolds (torus, sphere, ...).

The pieces, in the order a caller uses them:

- ``rrt_star`` — the tree itself, with batched edge evaluation (``edge_times_np``).
- ``converged_tree`` — ``rrt_star`` at a budget doubled until the mean cost-to-come stops falling.
  **This is what a sparse anchor selector should draw from.**
- ``build_roadmap`` — ``converged_tree`` thinned uniformly to a node budget, for a *dense* base
  (``methods/strategies/weak_supervision.py``).
- ``occluded_from_source`` / ``shadow_anchors`` — where the pointwise Eikonal residual leaves the
  field's level free, and the sparse anchors that pin it there.
- ``build_sphere_roadmap`` — H-NTFields' sphere-packing roadmap, a different prior with two-sided
  bounds rather than equality anchors (``methods/strategies/hntfields.py``).

Ported from the original ``torus.py``'s ``rrt_star``/``rrt_star_anchors_shadow``/``build_roadmap``.
"""

from __future__ import annotations

import heapq

import jax
import jax.numpy as jnp
import numpy as np

_RRT_SEED_OFFSET = 11
_ANCHOR_SEED_OFFSET = 12
_ROADMAP_SEED_OFFSET = 13

_CONVERGE_TOL = 0.005  # mean cost-to-come must fall <0.5% for the tree to count as converged
_CONVERGE_MAX_DOUBLINGS = 4  # 4 doublings off the 1500 floor already reaches 12k iterations
_RAY_SAMPLES = 24  # samples along a source-to-point geodesic when testing it for occlusion
# Anchors must clear the obstacle by this much. On the boundary itself the slowness ramp is steepest
# (`slow_width` 0.15), so the tree's edge quadrature is least accurate there and the value it reports
# is the least trustworthy thing to pin a field to.
_ANCHOR_CLEARANCE = 0.05


def _bucket(n: int) -> int:
    """Round ``n`` up to a power of two, so a growing array presents O(log n) distinct shapes.

    **This is the single thing that made RRT* practical here.** Every geometric primitive routes
    through ``env.geodesic`` / ``env.log_map`` / ``env.exp_map`` — JAX functions, so each distinct
    argument *shape* triggers an XLA compilation. An RRT* tree grows by one node per iteration, so a
    query against the whole tree presents a brand-new shape every single iteration and pays a compile
    every single time. Measured on the one-obstacle torus, one ``geodesic_distance_np`` call against
    the tree: **0.37 ms** at a shape already seen, **66.13 ms** when the shape grows by one each call —
    a 178x penalty, and with several such calls per iteration it was essentially the entire runtime
    (350 iterations took 42.5 s while the primitives themselves sum to ~1.5 ms per iteration).

    Bucketing to powers of two caps the number of compiled programs at ~log2(nodes) — 14 for a
    12,000-node tree — at the cost of evaluating a padded tail that is then masked out. Padding rows
    hold ``start``, a valid point on every manifold, so the padded evaluations are finite rather than
    NaN, and results for real rows are bitwise unchanged.
    """
    return 1 << max(0, (int(n) - 1).bit_length())


def geodesic_samples_np(env, a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """``n`` points along the geodesic from ``a`` to ``b``, [n, dim], on any manifold.

    ``a + t·displacement`` is a flat-chart shortcut and is wrong off the torus: on SO(3) it adds a
    3-component Lie-algebra element to a 4-component quaternion (a broadcast error), and on the
    sphere it traces a chord whose renormalisation reaches the wrong angle. ``Exp_a(t · Log_a(b))``
    is the geodesic on every manifold, with correct endpoints, and rests on the round-trip identity
    ``environments/test_manifolds.py`` verifies.
    """
    a_j, b_j = jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32)
    tangent = env.log_map(a_j, b_j)
    return np.asarray(jax.vmap(lambda t: env.exp_map(a_j, t * tangent))(jnp.linspace(0.0, 1.0, n)))


def geodesic_distance_np(env, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic distance from ``a`` to each row of ``b`` (host-side), on any manifold.

    ``‖displacement_np‖`` is the obvious spelling and is wrong on SO(3): its tangent vectors are
    3-component Lie-algebra elements while its points are 4-component quaternions, so a
    ``norm(..., axis=1)`` over what the caller assumes is a [N, dim] array collapses. Routing through
    ``env.geodesic``, which every environment defines against its own point representation, removes
    the assumption. Same class of defect as the ambient/tangent mismatches already fixed in the
    shadow mask and the geodesic gradient.
    """
    return np.asarray(env.geodesic(jnp.asarray(np.atleast_2d(b), jnp.float32), jnp.asarray(a, jnp.float32)))


def edge_times_np(env, sources: np.ndarray, target: np.ndarray, max_length: float | None = None) -> np.ndarray:
    """Travel time of every hop ``sources[i] -> target``, in one batched pass, [m].

    Edge cost is ``geodesic length x mean slowness along the geodesic`` — the same quantity the old
    per-edge ``edge_cost`` computed, and symmetric in its endpoints, so one call serves both RRT*'s
    choose-parent scan and its rewire scan.

    **Why this is batched rather than a loop.** Every geometric primitive here routes through
    ``env.geodesic`` / ``env.log_map`` / ``env.exp_map`` so that the edge is the real geodesic on every
    manifold — the SO(3)-correctness fix recorded in ``geodesic_samples_np``. Those are JAX functions,
    so a per-edge call costs a handful of eager dispatches, and the neighbour loops run ~200 of them
    per iteration once the tree is a few thousand nodes. Measured on the one-obstacle torus with the
    per-edge version: 350 iterations in **43.9 s**, 750 in **74.1 s**, extrapolating to ~13 min at the
    1500-iteration default and ~1 h for ``build_roadmap``'s convergence doubling — which is why that
    doubling had never actually been run to completion. Batching keeps the manifold-correct geometry
    and moves the dispatch count from O(neighbours) to O(1) per iteration.

    **The one semantic difference, and its direction.** The slowness integral needs a sample spacing
    that resolves the obstacle ramp, so the per-edge version chose ``n`` from *that edge's* length. A
    batch shares one ``n``, chosen from the **longest** edge in it, so every shorter edge is sampled
    *more* finely than before, never less. Under-pricing an edge is the failure this rule exists to
    prevent (it breaks the property that an RRT* cost is an upper bound on the optimum), so erring
    toward finer sampling is the safe direction.

    Args:
        env: Environment supplying ``geodesic``, ``log_map``, ``exp_map`` and ``slowness_np``.
        sources: Hop origins, [m, dim].
        target: Shared hop destination, [dim].
        max_length: Upper bound on any hop in this call, used to fix the sample count for the whole
            build instead of reading it off each batch. RRT* knows such a bound — no candidate is
            further than ``max(step, radius)`` from the new node — and holding the count fixed keeps
            the array shape stable, which is what ``_bucket`` exists for. ``None`` falls back to the
            longest hop actually present.

    Returns:
        Travel time of each hop, [m].
    """
    sources = np.atleast_2d(sources)
    count = len(sources)
    if not count:
        return np.zeros(0)
    # Bucket the batch size and pad with a repeat of the first source: a duplicated row is a valid
    # hop, so it costs a little arithmetic and no correctness, and the result is sliced back to `m`.
    span = _bucket(count)
    padded = np.concatenate([sources, np.repeat(sources[:1], span - count, axis=0)]) if span > count else sources
    src = jnp.asarray(padded, jnp.float32)
    tgt = jnp.asarray(target, jnp.float32)
    lengths = np.asarray(env.geodesic(src, tgt), dtype=float)
    width = float(getattr(env, "slow_width", 0.15))
    longest = float(lengths[:count].max()) if max_length is None else float(max_length)
    n = int(np.clip(np.ceil(3.0 * longest / max(width, 1e-6)), 6, 128))
    fractions = jnp.linspace(0.0, 1.0, n)
    tangent = jax.vmap(lambda a: env.log_map(a, tgt))(src)
    along = jax.vmap(lambda a, v: jax.vmap(lambda t: env.exp_map(a, t * v))(fractions))(src, tangent)
    slow = env.slowness_np(np.asarray(along).reshape(-1, env.dim)).reshape(span, n)
    return (lengths * slow.mean(axis=1))[:count]


def rrt_star(
    env, start, iters: int, step: float, radius: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """RRT* on the environment with slowness-weighted edges: returns tree nodes and cost-to-come.

    Edge cost = geodesic length x mean slowness along the edge, matching the soft-obstacle field, and
    evaluated for the whole neighbour set at once (see ``edge_times_np``). Steering, the nearest-node
    query and the neighbour query all use the manifold's own geodesic rather than a chart line, so the
    tree is the same algorithm on every manifold.

    Args:
        env: Environment supplying ``sample_domain``, ``geodesic``, ``log_map``/``exp_map``,
            ``slowness_np`` and optionally ``in_domain_np``.
        start: Source configuration, [dim]; always node 0 with cost 0.
        iters: Number of sampling iterations; at most one node is added per iteration.
        step: Maximum hop length when steering toward a sample.
        radius: Neighbourhood radius for choose-parent and rewiring.
        seed: RNG seed; the sampler draws exactly once per iteration, accepted or not.

    Returns:
        ``(nodes [n, dim], cost_to_come [n])``.
    """
    rng = np.random.default_rng(seed + _RRT_SEED_OFFSET)
    start = np.asarray(start, dtype=float)

    def in_domain(points: np.ndarray) -> np.ndarray:
        """Domain membership, defaulting to everywhere for manifolds that are not truncated.

        Without this the tree places nodes off the manifold: measured on the Poincaré ball, **116 of
        300** nodes fell outside the truncation wall, where travel time is undefined. Their costs then
        entered the anchor set and training went to RMS 1.347 against a do-nothing 0.653 — the
        supervision was worse than none. Note this is *not* the same as ``sdf > 0``: an obstacle is
        traversable at high cost by design, whereas an off-manifold point has no value at all.
        """
        checker = getattr(env, "in_domain_np", None)
        return checker(points) if checker is not None else np.ones(len(points), bool)

    # Preallocated, and prefilled with `start` rather than left as garbage: the tail past `count` is
    # handed to JAX as padding (see `_bucket`), so those rows must be points the manifold's geodesic
    # can evaluate without producing NaN. Preallocating also removes the O(iters^2) list-to-array
    # rebuild the previous version did once per iteration.
    capacity = int(iters) + 1
    nodes = np.repeat(start[None, :], capacity, axis=0)
    costs = np.zeros(capacity, dtype=float)
    count = 1
    # Every hop this build prices is either a steer (at most `step`) or a rewire candidate (at most
    # `radius`), so the slowness integral's sample count can be fixed once from that bound instead of
    # being read off each batch — one less varying shape.
    longest_hop = float(max(step, radius))

    for _ in range(iters):
        q_rand = env.sample_domain(rng, 1)[0]
        if not bool(in_domain(np.atleast_2d(q_rand))[0]):
            continue
        span = min(_bucket(count), capacity)
        tree = nodes[:span]  # rows [count, span) are padding and are masked out of every decision
        real = np.arange(span) < count
        distances = np.where(real, geodesic_distance_np(env, q_rand, tree), np.inf)
        near = int(distances.argmin())
        length = float(distances[near])
        if length < 1e-6:
            continue
        # Steer along the geodesic, not the chart line: `node + t·direction` is exact only on a flat
        # chart. It overshoots a conformal chart near its boundary and is a shape error on SO(3),
        # whose tangents are 3-vectors while its points are quaternions.
        q_new = geodesic_samples_np(env, tree[near], q_rand, 2 if length <= step else int(np.ceil(length / step)) + 1)[1]
        if not bool(in_domain(np.atleast_2d(q_new))[0]):
            continue  # off the manifold entirely: it has no travel time to report
        neighbours = np.where(real & (geodesic_distance_np(env, q_new, tree) < radius))[0]
        # The nearest node is always a parent candidate even when `step > radius` puts it outside the
        # neighbourhood, but it is *not* a rewiring candidate — rewiring is the neighbourhood's job.
        candidates = np.union1d(neighbours, [near])
        hop = edge_times_np(env, tree[candidates], q_new, longest_hop)
        nodes[count], costs[count] = q_new, float(np.min(costs[candidates] + hop))
        count += 1
        rewired = costs[count - 1] + hop
        improved = np.isin(candidates, neighbours) & (rewired < costs[candidates])
        costs[candidates[improved]] = rewired[improved]
    return nodes[:count].copy(), costs[:count].copy()


def occluded_from_source(env, points: np.ndarray) -> np.ndarray:
    """True where the geodesic from ``env.start`` to the point passes through an obstacle, [n].

    Geometric, from the scene's signed distance function alone — no solver, no ground truth, so it is
    available to a self-supervised selector.

    The ray is traced as ``Exp_start(f · Log_start(x))`` for ``f`` in [0, 1], which *is* the geodesic
    from ``start`` to ``x`` on every manifold, with the correct endpoints, by the round-trip identity
    ``environments/test_manifolds.py`` verifies. The obvious shortcut
    ``start + f·displacement_np(start, x)`` is correct only on a flat chart: on SO(3) it is a shape
    error (3-vector Lie-algebra element against a 4-vector quaternion), on the sphere the renormalised
    chord reaches angle ``arctan(theta)`` rather than ``theta`` at ``f = 1`` so the far part of the path
    is never tested, and in the Poincaré ball the tangent's norm is a geodesic length that exceeds the
    chart radius, so ``f = 1`` overshoots past ``x``. This function is the single implementation; the
    scoring path (``experiments/sweep.shadow_mask``) and the anchor selector both call it.

    Args:
        env: Environment supplying ``start``, ``log_map``, ``exp_map``, ``sdf`` and ``obstacles``.
        points: Query points, [n, dim].

    Returns:
        Boolean array, [n]; all False when the scene has no obstacles.
    """
    points = jnp.asarray(points, jnp.float32)
    if not len(env.obstacles):
        return np.zeros(len(points), bool)
    start = jnp.asarray(env.start, jnp.float32)
    tangent = jax.vmap(lambda x: env.log_map(start, x))(points)
    fractions = jnp.linspace(0.0, 1.0, _RAY_SAMPLES)
    along = jax.vmap(lambda f: jax.vmap(lambda v: env.exp_map(start, f * v))(tangent))(fractions)
    blocked = jnp.asarray(env.sdf(along.reshape(-1, env.dim))).reshape(_RAY_SAMPLES, -1) < 0.0
    return np.asarray(blocked.any(axis=0))


def converged_tree(
    env, start, iters: int, step: float, radius: float, seed: int, converge: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """The **full** RRT* tree, at a budget doubled until the mean cost-to-come stops falling.

    ``iters`` is a floor rather than the answer. This is self-supervised — RRT* costs only ever
    decrease as the tree fills in, so "they stopped decreasing" is a convergence signal needing no
    ground truth.

    The check exists because a hand-set count is not transferable, and twice was not. At the default
    350 the torus tree left 9% of nodes more than 10% above optimal, and anchors drawn from it sent
    training backwards (0.184 -> 0.339). The count that fixed the torus, 1500, was then reused on the
    Poincare ball — where hyperbolic volume grows exponentially with radius, so the same budget covers
    far less — and anchor error came out at 1.47 against a field of 0.333, sending training to 1.347
    against a do-nothing 0.653. Both failures were the same missing step, so the step is automatic now.

    Anchor selection should draw from *this*, not from ``build_roadmap``'s subsample. A uniform
    subsample preserves the *fraction* of occluded nodes — measured on the one-obstacle torus, 9.1% of
    the full tree and 9.0% of a 300-node draw — but it shrinks their absolute number tenfold, 272 to
    27. A 30-anchor budget then comes from a candidate pool barely larger than itself, so where the
    anchors land is decided by the thinning rather than by the preference.

    Returns:
        ``(nodes [n, dim], cost_to_come [n])``.
    """
    budget, previous = int(iters), None
    for _ in range(_CONVERGE_MAX_DOUBLINGS):
        nodes, costs = rrt_star(env, start, budget, step, radius, seed)
        mean_cost = float(np.mean(costs))
        if not converge or (previous is not None and previous - mean_cost <= _CONVERGE_TOL * previous):
            break
        previous, budget = mean_cost, budget * 2
    else:
        # Hitting the cap means the costs were still falling: the prior is NOT converged and using it
        # as supervision is the failure this whole mechanism exists to prevent. Say so loudly.
        print(
            f"[roadmap] WARNING: cost-to-come still decreasing at {budget // 2} iterations — the tree "
            "is not converged and its costs should not be trusted as anchors.",
            flush=True,
        )
    return nodes, costs


def build_roadmap(
    env, start, iters: int, step: float, radius: float, roadmap_nodes: int, seed: int, converge: bool = True
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``converged_tree`` subsampled to a fixed node budget — the coarse cost-to-come prior.

    Deterministic in ``seed`` so a strategy and its rendering rebuild the identical roadmap. The
    source (cost 0) is always kept. The subsample is uniform over the tree, which is what a *dense
    base* wants (``weak_supervision.py``) and what a *sparse targeted anchor set* does not — see
    ``converged_tree``.
    """
    nodes, costs = converged_tree(env, start, iters, step, radius, seed, converge)
    if len(nodes) > roadmap_nodes:
        idx = (
            np.random.default_rng(seed + _ROADMAP_SEED_OFFSET).choice(
                len(nodes) - 1, roadmap_nodes - 1, replace=False
            )
            + 1
        )
        idx = np.concatenate([[0], idx])  # keep the source node (cost 0) so T(start)=0
        nodes, costs = nodes[idx], costs[idx]
    return jnp.asarray(nodes, dtype=jnp.float32), jnp.asarray(costs, dtype=jnp.float32)


def shadow_anchors(
    env, nodes: np.ndarray, costs: np.ndarray, num_anchors: int, shadow_pref: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``num_anchors`` (point, cost-to-come) pairs from a tree, biased into obstacle shadows.

    The pointwise Eikonal residual constrains the *slope* of ``T`` and says nothing about its
    *level*: behind an obstacle the level is the accumulated cost of a detour, a path integral, so a
    field that is off by a constant there has near-zero residual. That is the measured failure mode —
    free-space MAE 0.094 against shadow MAE 0.538 on the one-obstacle torus. Sparse anchors fix it by
    pinning the free integration constant exactly where it is free, which is why *placement* beats
    *count*: same 30 anchors, same weight, shadow-targeted **0.0735** against uniform 0.1192.

    ``shadow_pref = 0`` makes the draw uniform, which is that control arm and not a separate code
    path. Occlusion is decided by ``occluded_from_source``, i.e. the manifold's own geodesic.

    Selection must run on the **full** ``converged_tree``. A uniform subsample keeps the occluded
    *fraction* but not the *count* — 272 occluded nodes in the one-obstacle torus tree against 27 in a
    300-node draw — leaving a 30-anchor budget almost no room to place anything.

    Args:
        env: Environment supplying ``sdf_np`` and the geometry ``occluded_from_source`` needs.
        nodes: Tree nodes, [n, dim].
        costs: Cost-to-come at those nodes, [n].
        num_anchors: How many to keep; capped at the number of eligible nodes.
        shadow_pref: Extra selection weight given to occluded nodes; 0 is a uniform draw.
        seed: RNG seed for the draw.

    Returns:
        ``(points [k, dim], values [k], occluded [k])`` — the last so the caller can report how many
        anchors actually landed in shadow rather than assuming the preference worked.
    """
    nodes, costs = np.asarray(nodes), np.asarray(costs)
    eligible = np.asarray(env.sdf_np(nodes)) > _ANCHOR_CLEARANCE
    nodes, costs = nodes[eligible], costs[eligible]
    if not len(nodes):
        raise ValueError("no tree node clears the obstacles; anchors cannot be selected")
    occluded = occluded_from_source(env, nodes)
    weight = 1.0 + shadow_pref * occluded
    weight = weight / weight.sum()
    k = int(min(num_anchors, len(nodes)))
    chosen = np.random.default_rng(seed + _ANCHOR_SEED_OFFSET).choice(
        len(nodes), size=k, replace=False, p=weight
    )
    return nodes[chosen], costs[chosen], occluded[chosen]


_SPHERE_ROADMAP_SEED_OFFSET = 14


def _edge_time(env, a: np.ndarray, b: np.ndarray, ksamp: int = 12) -> float:
    """Travel *time* of the hop a->b: geodesic length x mean slowness along it."""
    length = float(geodesic_distance_np(env, a, b[None, :])[0])
    return length * float(env.slowness_np(geodesic_samples_np(env, a, b, ksamp)).mean())


def _edge_is_free(env, a: np.ndarray, b: np.ndarray, ksamp: int = 12) -> bool:
    """Collision check along the geodesic: every sample must be outside the obstacles."""
    return bool(np.all(env.sdf_np(geodesic_samples_np(env, a, b, ksamp)) > 0.0))


def build_sphere_roadmap(
    env, start, num_nodes: int, pool: int, connect_radius: float, seed: int, max_radius: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """H-NTFields' sparse sphere-packing roadmap, plus travel time from ``start`` to each node.

    Implements the roadmap of H-NTFields (Ni, Liu & Qureshi 2026, arXiv 2604.13204, Sec. IV-A):

    - *Free-space volume sampling.* Draw collision-free configurations and give each a maximized
      free sphere (radius = clearance ``env.sdf``). A candidate is accepted only if it lies **outside
      the union of the spheres already accepted**, which is what makes the packing sparse and
      spread-out rather than clustered.
    - *Sparse connectivity.* Node pairs within ``connect_radius`` are joined when the straight line
      between them is collision-free.
    - *Distances.* Dijkstra from ``start`` (always node 0) gives the graph distance later used for
      the travel-time bounds.

    Edge weights are the **slowness-integrated** hop cost (``_edge_time``), not raw length. The
    paper's bounds work because its free-space speed is 1, so distance *is* time; this repo's
    ``env.slowness`` varies smoothly from 1 to ``slowness_max``, so plain lengths would bound
    nothing. Weighting by slowness restores the bound's meaning in travel-time units.

    Returns ``(nodes[N, dim], radii[N], time_from_start[N])`` for the nodes reachable from the
    source; unreachable components are dropped.
    """
    rng = np.random.default_rng(seed + _SPHERE_ROADMAP_SEED_OFFSET)
    start = np.asarray(start, dtype=float)

    cap = max_radius if max_radius > 0 else np.inf  # bound the free sphere; see the note below
    nodes = [start]
    radii = [min(max(float(env.sdf_np(start[None, :])[0]), 1e-3), cap)]
    candidates = env.sample_domain(rng, pool)
    clearance = np.minimum(env.sdf_np(candidates), cap)
    for cand, clear in zip(candidates, clearance):
        if len(nodes) >= num_nodes:
            break
        if clear <= 0.0:  # inside an obstacle
            continue
        node_arr = np.array(nodes)
        if np.any(geodesic_distance_np(env, cand, node_arr) <= np.array(radii)):
            continue  # already covered by an accepted node's free sphere
        nodes.append(cand)
        radii.append(float(clear))
    node_arr, radius_arr = np.array(nodes), np.array(radii)

    n = len(node_arr)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        near = np.where(geodesic_distance_np(env, node_arr[i], node_arr) < connect_radius)[0]
        for j in near:
            if j <= i or not _edge_is_free(env, node_arr[i], node_arr[j]):
                continue
            w = _edge_time(env, node_arr[i], node_arr[j])
            adjacency[i].append((int(j), w))
            adjacency[j].append((i, w))

    time_from_start = np.full(n, np.inf)
    time_from_start[0] = 0.0
    queue = [(0.0, 0)]
    while queue:
        d, i = heapq.heappop(queue)
        if d > time_from_start[i]:
            continue
        for j, w in adjacency[i]:
            if d + w < time_from_start[j]:
                time_from_start[j] = d + w
                heapq.heappush(queue, (d + w, j))

    reachable = np.isfinite(time_from_start)
    print(
        f"[sphere-roadmap] {int(reachable.sum())}/{n} nodes reachable "
        f"(radii {radius_arr.min():.2f}–{radius_arr.max():.2f})",
        flush=True,
    )
    return node_arr[reachable], radius_arr[reachable], time_from_start[reachable]
