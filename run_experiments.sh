#!/usr/bin/env zsh
# The experiment ladder, in one command per stage. Every stage trains through `srms.run`'s `Config`
# and `STRATEGIES` — `srms.experiments.sweep` is the seeds-and-scoring layer over it, not a second
# implementation — so a flag means the same thing here as it does in `python -m srms.run`.
#
#   ./run_experiments.sh check     # geometry identities, self-supervision proof, smoke matrix
#   ./run_experiments.sh validate  # is the RRT* prior good enough to supervise with? (run before exp2 anchors)
#   ./run_experiments.sh exp1      # no obstacles      — scored against the analytic geodesic
#   ./run_experiments.sh exp2      # one obstacle      — label-free x3, plus the anchored arms
#   ./run_experiments.sh exp3      # three obstacles   — label-free only, no supervision anywhere
#   ./run_experiments.sh plan      # 10 goals on the exp3 torus field, timed against the fit
#   ./run_experiments.sh all
#
# Overrides:  SEEDS=1 ./run_experiments.sh exp3     (default 5; the seed places the obstacles, so
#             5 seeds is the 5-scene sweep the write-up requires, not a repeat of one scene)
#             RES=120 ./run_experiments.sh exp2     (default 240; below 240 the fast-marching
#             reference's own error, 0.027 on the torus, swamps the differences being measured)
set -e
cd "$(dirname "$0")"

SEEDS=${SEEDS:-5}
RES=${RES:-240}
PY=${PY:-.venv/bin/python}
FIG=results/figures

# Shared across every training stage. `--nonneg-weights` clamps V >= 0 and `--trunc-sigma 2` gives
# each splat compact support: together they are the label-free arm the ladder is measured on.
COMMON=(--method factored --seeds "$SEEDS" --steps 4000 --num-collocation 2048 --densify
        --no-causal --nonneg-weights --nonneg-weight 0 --trunc-sigma 2.0 --error-clip 0.2)

# Equality anchors, and the weight is not the default. `--anchor-weight` defaults to 1e-2, which
# belongs to the *bounds* mechanism; equality anchors want 0.5. Trusting an RRT* cost harder than
# that bakes the planner's suboptimality into the field (this repo's B4 lesson).
ANCHORS=(--num-anchors 30 --anchor-mode equality --anchor-weight 0.5)

row () {  # row <label> <out-subdir> <extra args...>
  print "\n===== $1 ====="
  $PY -m srms.experiments.sweep "${COMMON[@]}" --resolution "$RES" --out-dir "$FIG/$2" "${@:3}" 2>&1 \
    | grep --line-buffered -vE "it/s\]|it\]"
}

stage_check () {
  print "\n########## geometry identities ##########"
  $PY -m srms.environments.test_manifolds
  print "\n########## no training path can see ground truth ##########"
  $PY -m srms.environments.test_selfsupervised
  print "\n########## smoke matrix (every manifold x arm, 10 steps) ##########"
  $PY -m srms.experiments.preflight
}

stage_validate () {
  print "\n########## is the prior good enough to supervise with? ##########"
  $PY -m srms.experiments.validate_prior --environment all --num-obstacles 1 --resolution "$RES"
  $PY -m srms.experiments.validate_prior --environment so3 --resolution 40
}

stage_exp1 () {  # the gate: the answer is the analytic geodesic, so this tests the machinery
  for env in torus sphere poincare_hyperbolic; do
    row "exp1 — $env, no obstacles" "exp1/$env" --environment $env --num-obstacles 0
  done
  row "exp1 — so3, no obstacles" exp1/so3 --environment so3 --dim 3 --num-obstacles 0 --resolution 40
}

stage_exp2 () {  # one obstacle: label-free everywhere, anchors only where the prior earns them
  for env in torus sphere poincare_hyperbolic; do
    row "exp2 — $env, label-free" "exp2/${env}_labelfree" --environment $env --num-obstacles 1
  done
  # Anchors on the torus and H² only. On the sphere the prior (RMS 0.0078) is barely better than the
  # label-free field it would be correcting (0.0144), which is not the `prior error << field error`
  # regime equality anchors require — see the validate stage.
  row "exp2 — torus, 30 shadow-targeted anchors" exp2/torus_anchors30 \
      --environment torus --num-obstacles 1 "${ANCHORS[@]}" --anchor-shadow-pref 3.0
  row "exp2 — torus, 30 uniform anchors (control)" exp2/torus_anchors30_uniform \
      --environment torus --num-obstacles 1 "${ANCHORS[@]}" --anchor-shadow-pref 0.0
  row "exp2 — hyperbolic, 30 shadow-targeted anchors" exp2/poincare_anchors30 \
      --environment poincare_hyperbolic --num-obstacles 1 "${ANCHORS[@]}" --anchor-shadow-pref 3.0
}

stage_exp3 () {  # three obstacles, label-free only: no planner, no roadmap, anywhere in the path
  for env in torus sphere poincare_hyperbolic; do
    row "exp3 — $env, 3 obstacles, label-free" "exp3/${env}_labelfree" --environment $env --num-obstacles 3
  done
}

stage_plan () {  # what the field is for: paths to arbitrary goals, at query cost rather than solve cost
  local exp3=$FIG/exp3/torus_labelfree/torus_obs3_seed1.pkl
  local free=$FIG/exp2/torus_labelfree/torus_obs1_seed1.pkl
  local anchored=$FIG/exp2/torus_anchors30/torus_obs1_seed1.pkl
  for needed in $exp3 $free $anchored; do
    if [[ ! -f $needed ]]; then
      print "  missing $needed — run './run_experiments.sh exp2' and './run_experiments.sh exp3' first."
      return 1
    fi
  done

  # One goal set throughout: 60 spread over the manifold with half biased onto obstacle boundaries,
  # and whatever fraction is occluded left at the scene's natural rate. Every run prints that mix and
  # the per-class breakdown, so the harder classes are still visible without separate arms.
  plan () {  # plan <label> <params> <out-name>
    print "\n===== $1 ====="
    $PY -m srms.experiments.plan --params-path "$2" --out-path "$FIG/plan/$3.png" \
        --num-goals 60 --near-fraction 0.5 2>&1 | grep --line-buffered -vE "it/s\]"
  }

  plan "3-obstacle torus, 60 goals"                    $exp3     torus_paths
  # Same 60 goals, two fields: the anchored one has less than half the RMS and more spurious minima,
  # so this is where RMS and usability come apart.
  plan "1-obstacle torus, label-free field"            $free     torus_labelfree_paths
  plan "1-obstacle torus, +30 anchors (RMS 2x better)" $anchored torus_anchors30_paths

  print "\n===== spurious minima: the metric that does predict path success ====="
  $PY -m srms.experiments.minima --root $FIG
}

case ${1:-all} in
  check)    stage_check ;;
  validate) stage_validate ;;
  exp1)     stage_exp1 ;;
  exp2)     stage_exp2 ;;
  exp3)     stage_exp3 ;;
  plan)     stage_plan ;;
  all)      stage_check; stage_validate; stage_exp1; stage_exp2; stage_exp3; stage_plan ;;
  *)        print "unknown stage: $1 (check|validate|exp1|exp2|exp3|plan|all)"; exit 2 ;;
esac

print "\nEach run writes to $FIG/<stage>/<arm>/: a labelled figure (.pdf/.png), a bare unlabelled"
print "twin for LaTeX captions, the raw fields (.npz) so figures restyle without retraining, and"
print "the trained parameters (.pkl) so planning never re-solves."
