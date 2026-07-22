#!/usr/bin/env bash

set -euo pipefail

ROOT="/ocean/projects/oce110003p/kaushiks/jove/NHQG"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
cd "$ROOT"

GPU="${GPU:-0}"
JOB_USER="${JOB_USER:-kaushiks}"
DT="5e-5"
NX=64
NZ=256
RA=100
EXTEND_DELTA="${EXTEND_DELTA:-20.0}"

COMMON_ARGS=(
  --thermal-closure evolve_mean
  --Nx "$NX"
  --Nz "$NZ"
  --Ra "$RA"
  --dt "$DT"
  --imex-scheme ars222
  --mean-temp-eps-sq 1.0
  --mean-exchange-discretization balanced_sbp2_pc
  --nonlinear-advection flux
  --save-every 1000
  --checkpoint-every 5000
  --snapshot-dt 0.5
)

PROBE_CKPT="${OUTPUT_ROOT}/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2_flux_fromstart_Nx64_Nz256_dt5e5_t20/checkpoint_00400000.npz"
PROBE_DIR="${OUTPUT_ROOT}/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_probe_t420_from_t20_Nx64_Nz256_dt5e5"
START_DIR="${START_DIR:-$PROBE_DIR}"
CURRENT_TARGET="${CURRENT_TARGET:-42.0}"
NEXT_TARGET="${NEXT_TARGET:-80.0}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a monitor_balancedsbp2pc_chain.log
}

checkpoint_step_for_time() {
  python - "$1" <<'PY'
import sys
t = float(sys.argv[1])
dt = 5e-5
print(f"{int(round(t / dt)):08d}")
PY
}

time_string() {
  python - "$1" <<'PY'
import sys
t = float(sys.argv[1])
s = f"{t:.2f}"
s = s.rstrip("0").rstrip(".")
print(s.replace(".", "p"))
PY
}

checkpoint_is_finite() {
  python - "$1" <<'PY'
import sys
import numpy as np

path = sys.argv[1]
with np.load(path) as data:
    ok = True
    for key in data.files:
        arr = data[key]
        if np.issubdtype(arr.dtype, np.number):
            if not np.all(np.isfinite(arr)):
                ok = False
                break
print("yes" if ok else "no")
PY
}

latest_checkpoint() {
  local outdir="$1"
  ls -1 "$outdir"/checkpoint_*.npz 2>/dev/null | sort | tail -n 1 || true
}

gpu_remaining_info() {
  local raw
  raw="$(squeue -h -u "$JOB_USER" -o "%P|%L|%t|%j|%i" 2>/dev/null || true)"
  python - "$raw" <<'PY'
import sys

raw = sys.argv[1]

def parse_walltime(value: str) -> int:
    value = value.strip()
    if not value:
        return 0
    if value.upper() in {"UNLIMITED", "INFINITE"}:
        return 10**12
    days = 0
    if "-" in value:
        day_str, value = value.split("-", 1)
        days = int(day_str)
    parts = [int(p) for p in value.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    else:
        raise ValueError(f"Unrecognized walltime format: {value!r}")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds

best_seconds = 0
best_summary = ""

for line in raw.splitlines():
    line = line.strip()
    if not line:
        continue
    partition, remaining, state, name, jobid = line.split("|", 4)
    if state != "R":
        continue
    if "gpu" not in partition.lower():
        continue
    seconds = parse_walltime(remaining)
    if seconds > best_seconds:
        best_seconds = seconds
        best_summary = f"jobid={jobid} partition={partition} remaining={remaining} name={name}"

print(best_seconds)
print(best_summary)
PY
}

gpu_time_available() {
  local info remaining
  info="$(gpu_remaining_info)"
  remaining="$(printf '%s\n' "$info" | sed -n '1p')"
  [ "${remaining:-0}" -gt 0 ]
}

log_gpu_time() {
  local info remaining summary
  info="$(gpu_remaining_info)"
  remaining="$(printf '%s\n' "$info" | sed -n '1p')"
  summary="$(printf '%s\n' "$info" | sed -n '2p')"
  log "gpu_remaining_seconds=${remaining:-0} ${summary}"
}

wait_for_output_dir_to_finish() {
  local outdir="$1"
  while pgrep -af "python scripts/continue_from_checkpoint.py .*--output-dir ${outdir}" >/dev/null; do
    sleep 60
  done
}

launch_continue() {
  local ckpt="$1"
  local outdir="$2"
  local target_t="$3"
  log "launching continuation: checkpoint=$ckpt output=$outdir t_final=$target_t"
  nohup /usr/bin/bash -lc "
    cd '$ROOT' &&
    env CUDA_VISIBLE_DEVICES=$GPU OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      JAX_ENABLE_X64=1 PYTHONPATH=. \
      python scripts/continue_from_checkpoint.py \
        --checkpoint '$ckpt' \
        --output-dir '$outdir' \
        ${COMMON_ARGS[*]} \
        --t-final '$target_t'
  " >> monitor_balancedsbp2pc_chain.log 2>&1 &
}

launch_dense_failure_probe() {
  local ckpt="$1"
  local t_start="$2"
  local t_end="$3"
  local start_tag end_tag outdir
  start_tag="$(time_string "$t_start")"
  end_tag="$(time_string "$t_end")"
  outdir="${OUTPUT_ROOT}/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_densefailure_from_t${start_tag}_Nx${NX}_Nz${NZ}_dt5e5_t${end_tag}"

  if [ -d "$outdir" ]; then
    log "dense failure probe already exists: $outdir"
    return 0
  fi

  log "launching dense failure probe: checkpoint=$ckpt output=$outdir t_final=$t_end"
  nohup /usr/bin/bash -lc "
    cd '$ROOT' &&
    env CUDA_VISIBLE_DEVICES=$GPU OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      JAX_ENABLE_X64=1 PYTHONPATH=. \
      python scripts/continue_from_checkpoint.py \
        --checkpoint '$ckpt' \
        --output-dir '$outdir' \
        --thermal-closure evolve_mean \
        --Nx '$NX' \
        --Nz '$NZ' \
        --Ra '$RA' \
        --dt '$DT' \
        --t-final '$t_end' \
        --imex-scheme ars222 \
        --mean-temp-eps-sq 1.0 \
        --mean-exchange-discretization balanced_sbp2_pc \
        --nonlinear-advection flux \
        --save-every 200 \
        --checkpoint-every 1000 \
        --snapshot-dt 0.05
  " >> monitor_balancedsbp2pc_chain.log 2>&1 &
}

run_chain() {
  local current_dir="$1"
  local current_target="$2"
  local next_target="$3"

  while true; do
    log "waiting for output to finish: $current_dir"
    wait_for_output_dir_to_finish "$current_dir"

    local latest_ckpt
    latest_ckpt="$(latest_checkpoint "$current_dir")"
    if [ -z "$latest_ckpt" ]; then
      log "no checkpoints found in $current_dir"
      return 1
    fi

    local finite
    finite="$(checkpoint_is_finite "$latest_ckpt")"
    log "latest checkpoint=$latest_ckpt finite=$finite"

    if [ "$finite" != "yes" ]; then
      log "latest checkpoint is non-finite; stopping chain"
      return 1
    fi

    local target_step final_ckpt
    target_step="$(checkpoint_step_for_time "$current_target")"
    final_ckpt="$current_dir/checkpoint_${target_step}.npz"

    if [ -f "$final_ckpt" ] && [ "$(checkpoint_is_finite "$final_ckpt")" = "yes" ]; then
      local start_tag end_tag next_dir
      log_gpu_time
      if ! gpu_time_available; then
        log "target t=$current_target reached cleanly but no GPU time remains; stopping chain"
        return 0
      fi
      start_tag="$(time_string "$current_target")"
      end_tag="$(time_string "$next_target")"
      next_dir="${OUTPUT_ROOT}/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_continue_from_t${start_tag}_Nx${NX}_Nz${NZ}_dt5e5_t${end_tag}"

      if [ ! -d "$next_dir" ]; then
        launch_continue "$final_ckpt" "$next_dir" "$next_target"
      else
        log "continuation directory already exists: $next_dir"
      fi

      current_dir="$next_dir"
      current_target="$next_target"
      next_target="$(python - "$next_target" "$EXTEND_DELTA" <<'PY'
import sys
print(float(sys.argv[1]) + float(sys.argv[2]))
PY
)"
      sleep 60
      continue
    fi

    local t_last t_dense_end
    t_last="$(python - "$latest_ckpt" <<'PY'
import os
import re
import sys
name = os.path.basename(sys.argv[1])
m = re.search(r'checkpoint_(\d+)\.npz$', name)
step = int(m.group(1))
dt = 5e-5
print(step * dt)
PY
)"
    t_dense_end="$(python - "$t_last" <<'PY'
import sys
print(float(sys.argv[1]) + 0.5)
PY
)"
    log "run stopped before target or without a finite final checkpoint; last finite time=$t_last"
    log_gpu_time
    if gpu_time_available; then
      launch_dense_failure_probe "$latest_ckpt" "$t_last" "$t_dense_end"
    else
      log "GPU time no longer available; skipping dense failure probe"
    fi
    return 0
  done
}

run_chain "$START_DIR" "$CURRENT_TARGET" "$NEXT_TARGET"
