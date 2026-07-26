#!/bin/bash
#SBATCH --job-name=nhqg_p2
#SBATCH --partition=gpu-shared          # Expanse; Bridges-2: GPU-shared
#SBATCH --account=cla119                # Expanse allocation
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#
# Chained NHQG polar run for ACCESS-CI clusters.
#
# The 48 h wall-clock cap is shorter than the campaign needs, so this script
# resubmits itself with --dependency=afterok and each leg restarts from the
# newest checkpoint in $OUTDIR. Legs are idempotent: run_polar.py exits 0
# immediately if the checkpoint already reached --t-final, which is what stops
# the chain.
#
#   sbatch scripts/submit_access.slurm                 # start / continue
#   MODE=smoke sbatch scripts/submit_access.slurm      # 2000-step acceptance smoke
#
# Set ARCHIVE and OUTDIR below (or export them before sbatch).

set -euo pipefail

# ---------------------------------------------------------------- settings --
ARCHIVE=${ARCHIVE:-$HOME/projects/NHQG_runs_archive_2026-07}
OUTDIR=${OUTDIR:-output/polar_p2_continue}
T_FINAL=${T_FINAL:-400}
MODE=${MODE:-full}
MAX_LEGS=${MAX_LEGS:-8}
LEG=${LEG:-1}

# Seed state: P2 at t=158.0 (see RESTART.md in the archive).
SEED_CHECKPOINT=${SEED_CHECKPOINT:-$ARCHIVE/polar/p2_trap_sponge_512x64_t30_t158/checkpoint_03160000.npz}

# ------------------------------------------------------------- environment --
# NOTE: on Expanse neither `module` nor ~/anaconda3/bin is on PATH in a
# non-interactive shell -- a bare `python3` there is the system 3.6.8, which
# cannot even parse this codebase. Activate conda explicitly.
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda 2>/dev/null || true
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV:-nhqg}" 2>/dev/null || true

PY_VER=$(python -c 'import sys;print("%d%02d"%sys.version_info[:2])' 2>/dev/null || echo 0)
if [ "$PY_VER" -lt 308 ]; then
  echo "FATAL: python is $(python -V 2>&1) -- need >= 3.8. Check CONDA_ENV." >&2
  exit 1
fi

export JAX_ENABLE_X64=1
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

# ------------------------------------------------- pick the restart source --
# Newest checkpoint already in OUTDIR wins; otherwise start from the archive.
LATEST=""
if [ -d "$OUTDIR" ]; then
  LATEST=$(ls -1 "$OUTDIR"/checkpoint_*.npz 2>/dev/null | sort -V | tail -1 || true)
fi
RESTART=${LATEST:-$SEED_CHECKPOINT}

if [ ! -f "$RESTART" ]; then
  echo "FATAL: no restart checkpoint at $RESTART" >&2
  exit 1
fi
echo "leg $LEG/$MAX_LEGS  restart=$RESTART  t_final=$T_FINAL  mode=$MODE"

# ------------------------------------------------------------------- flags --
# RECORDED P2 configuration. Do not drop any of these: the driver's argparse
# defaults (legacy / 32_rule / jacobian / float32) match no production run, and
# a wrong restart is silent rather than an error.
COMMON=(
  --Nx 512 --Nz 64 --L-over-Lc 48
  --gamma 2.2e-3 --trap-r-star 92.45
  --sponge-rate 50 --sponge-r-start 104 --sponge-sharpness 30
  --Ra 100 --dt 5e-5
  --init convective --init-amplitude 1e-3 --seed 0
  --nu-q 1 --nu-w 1 --nu-theta 1 --hyper-order 1
  --w-bc-top neumann --thermal-closure evolve_mean
  --mean-exchange balanced_sbp2_pc --sbp-substeps 4
  --advection flux --dealias 23_rule --imex-matmul-chunk 128
  --restart-checkpoint "$RESTART"
)

step_of () { basename "$1" | tr -dc '0-9'; }
t_of () { python3 -c "print($(step_of "$1") * 5e-5)"; }

if [ "$MODE" = smoke ]; then
  # 0.1 t.u. (2000 steps) from the seed state, into a throwaway directory.
  python scripts/run_polar.py "${COMMON[@]}" \
    --t-final "$(python3 -c "print($(t_of "$RESTART") + 0.1)")" \
    --diag-interval 0.01 --snapshot-interval 0.05 --checkpoint-interval 1 \
    --output-dir "${OUTDIR}_smoke"
  echo "smoke complete; compare ${OUTDIR}_smoke/diagnostics.csv against the"
  echo "tail of \$ARCHIVE/polar/p2_trap_sponge_512x64_t30_t158/diagnostics.csv"
  exit 0
fi

# errexit off around the solver so a non-zero status is inspected, not fatal.
set +e
python scripts/run_polar.py "${COMMON[@]}" \
  --t-final "$T_FINAL" \
  --diag-interval 0.05 --snapshot-interval 1 --checkpoint-interval 2 \
  --output-dir "$OUTDIR"
STATUS=$?
set -e

# ------------------------------------------------------------ chain onward --
# run_polar.py exits 2 on non-finite diagnostics -- do NOT resubmit into a
# blowup, and do not chain past MAX_LEGS.
if [ "$STATUS" -ne 0 ]; then
  echo "leg $LEG exited $STATUS; chain stopped" >&2
  exit "$STATUS"
fi

FINAL=$(ls -1 "$OUTDIR"/checkpoint_*.npz 2>/dev/null | sort -V | tail -1 || true)
REACHED=$([ -n "$FINAL" ] && t_of "$FINAL" || echo 0)
DONE=$(python3 -c "print(1 if $REACHED >= $T_FINAL - 1e-9 else 0)")

if [ "$DONE" = 1 ]; then
  echo "target t=$T_FINAL reached; chain complete"
  exit 0
fi
if [ "$LEG" -ge "$MAX_LEGS" ]; then
  echo "MAX_LEGS=$MAX_LEGS reached at t=$REACHED; resubmit manually to continue"
  exit 0
fi

NEXT=$((LEG + 1))
echo "leg $LEG finished at t=$REACHED; submitting leg $NEXT"
ARCHIVE="$ARCHIVE" OUTDIR="$OUTDIR" T_FINAL="$T_FINAL" MAX_LEGS="$MAX_LEGS" LEG="$NEXT" \
  sbatch --dependency=afterok:"$SLURM_JOB_ID" scripts/submit_access.slurm
