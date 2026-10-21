#!/usr/bin/env bash
# Regenerate every benchmark CSV reported in the paper from the committed code.
# Scope matches the paper (not the larger grid in the README). Two jobs run in parallel:
#   A: R0, E1 fast models, E3            (~1.5 h)
#   B: E1 PINN curve + PINN ablations    (~3 h)
# Usage: scripts/regen_benchmarks.sh [out_dir]   (default results/benchmark; must not exist yet)
set -euo pipefail
cd "$(dirname "$0")/.."
B=${1:-results/benchmark}
TR=results/main_sweep_fast
TE=results/holdout_fast
PY=.venv/bin/python
if [ -e "$B" ]; then echo "$B exists; move it aside first" >&2; exit 1; fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then echo "commit your changes first" >&2; exit 1; fi
mkdir -p "$B/pinn" "$B/pinn_ablations"
{ echo "git_commit: $(git rev-parse HEAD)"; echo "started: $(date '+%Y-%m-%dT%H:%M:%S')"; } > "$B/PROVENANCE.txt"

FAST="analytic gp_data_only gp_hybrid mlp_hybrid physbasis physbasis_noprior physbasis_membrane_only"
(
  $PY cli.py benchmark r0 --train $TR --test $TE --out $B
  $PY cli.py benchmark e1 --train $TR --test $TE --out $B --models $FAST --sizes 16 32 64 128 256 --reps 5 --threads 4
  $PY cli.py benchmark e1 --train $TR --test $TE --out $B --models $FAST --sizes 512 --reps 2 --threads 4
  $PY cli.py benchmark e1 --train $TR --test $TE --out $B --models analytic gp_hybrid mlp_hybrid physbasis \
      --sizes 1024 --reps 1 --threads 4
  $PY cli.py benchmark e2 --train $TR --test $TE --out $B --sizes 64 256
  $PY cli.py benchmark e3 --train $TR --test $TE --out $B \
      --models analytic gp_data_only gp_hybrid mlp_hybrid physbasis --sizes 128 --reps 3 --threads 4
) > "$B/job_a.log" 2>&1 &
A=$!
(
  $PY cli.py benchmark e1 --train $TR --test $TE --out $B/pinn --models pinn --sizes 16 32 64 128 256 --reps 3 --threads 4
  $PY cli.py benchmark e1 --train $TR --test $TE --out $B/pinn_ablations --models pinn_nophys pinn_nobending \
      --sizes 64 --reps 2 --threads 4
) > "$B/job_b.log" 2>&1 &
P=$!
wait $A && ea=0 || ea=$?; echo "job A exit $ea" >> "$B/PROVENANCE.txt"
wait $P && eb=0 || eb=$?; echo "job B exit $eb" >> "$B/PROVENANCE.txt"
if [ "$ea" -ne 0 ] || [ "$eb" -ne 0 ]; then echo "a job failed; see $B/job_*.log" >&2; exit 1; fi
echo "finished: $(date '+%Y-%m-%dT%H:%M:%S')" >> "$B/PROVENANCE.txt"
$PY cli.py paper-figs --bench $B
$PY scripts/make_numbers.py --bench $B --out docs/paper/numbers.tex
