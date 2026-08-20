"""Two-minute smoke matrix: every manifold x obstacle count, through the full scoring path.

Every bug this project hit in the last session was found by an expensive sweep failing, and each had
the same shape — correct on the flat torus, broken on a curved manifold, or correct without obstacles
and broken with them. A 10-step run exercises the same code as a 4000-step one everywhere that
matters: environment construction, the splat evaluator, the loss, the scoring reference, the shadow
mask, the metrics and the figure writer. Run it before launching any sweep.

Run: ``python -m srms.experiments.preflight`` (exit code 1 on any failure).
"""

from __future__ import annotations

import sys
import traceback

from srms.experiments.sweep import SweepConfig, run_seed

MANIFOLDS = (("torus", 2, 32), ("sphere", 2, 32), ("poincare_hyperbolic", 2, 32), ("so3", 3, 16))


def main() -> int:
    """Run every (manifold, obstacle count) pair briefly and report which survive."""
    print(f"{'manifold':<22} {'obstacles':>9} {'RMS':>10} {'MAE shadow':>11}  verdict")
    print("-" * 62)
    failures = 0
    for name, dim, resolution in MANIFOLDS:
        # The last arm covers the *equality* path (converged_tree -> shadow_anchors -> hinge), which
        # is a different mechanism from the bounds path above it and shares none of its code.
        for num_obstacles, anchors, mode in ((0, 0, "bounds"), (1, 0, "bounds"), (1, 64, "bounds"), (1, 30, "equality")):
            sweep = SweepConfig(
                environment=name,
                dim=dim,
                num_obstacles=num_obstacles,
                seeds=1,
                steps=10,
                num_splats=32,
                num_collocation=128,
                resolution=resolution,
                densify=False,
                num_anchors=anchors,
                anchor_mode=mode,
                figures=True,
                out_dir="figures/preflight",
            )
            try:
                row = run_seed(sweep, 1)
                shadow = "—" if row["mae_shadow"] != row["mae_shadow"] else f"{row['mae_shadow']:.4f}"
                tag = f"{num_obstacles}+{mode}" if anchors else str(num_obstacles)
                print(f"{name:<22} {tag:>9} {row['rms']:>10.4f} {shadow:>11}  PASS")
            except Exception as exc:  # noqa: BLE001 - a smoke test reports every failure mode
                failures += 1
                tag = f"{num_obstacles}+{mode}" if anchors else str(num_obstacles)
                print(f"{name:<22} {tag:>9} {'—':>10} {'—':>11}  FAIL: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=2)
    print(f"\n{failures} failure(s) across {4 * len(MANIFOLDS)} manifold x arm combinations.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
