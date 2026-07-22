#!/usr/bin/env python
"""Nz convergence study: Ra=100, Nx=256, L=48 (~10 L_c, matching Miquel).

Tests Nz = 32, 64, 128 to check whether Nu converges toward Miquel's 30.96.
Hypothesis: Nz=32 truncates the vertical cascade → artificially high energy/Nu.

Memory estimates (Nx=256, complex128):
  Nz=32:   state ~50 MB,  IMEX ~5 MB,  total ~500 MB
  Nz=64:   state ~100 MB, IMEX ~17 MB, total ~1 GB
  Nz=128:  state ~200 MB, IMEX ~54 MB, total ~2 GB
  Nz=256:  state ~400 MB, IMEX ~106 MB, total ~4 GB

Usage: python run_nz_convergence.py [Nz]   (default: 64)
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.paths import normalize_output_dir
from nhqg.solver import make_initial_state, imex_step, State, invert_psi
from nhqg.diagnostics import compute_diagnostics, energy_spectrum
from nhqg.io import save_checkpoint

# --- Parse Nz from command line ---
Nz = int(sys.argv[1]) if len(sys.argv) > 1 else 64

# --- Configuration ---
Nx = 256            # Match Miquel's horizontal resolution
L = 48.0            # ~10 L_c (Miquel domain)
dx = L / Nx
dt_max = 5e-4       # start with same dt as 512 run
dt = dt_max
cfl_target = 0.3
nu = 1.0            # Laplacian dissipation

output_dir = normalize_output_dir(f'output_nz{Nz}')
save_every = 50
wall_limit = 3200   # ~53 minutes (leave margin for interactive session)

print("=" * 70)
print(f"NHQGE Nz Convergence: Ra=100, Nx={Nx}, Nz={Nz}, L={L}")
print(f"Target: Miquel et al. Nu = 30.96 ± 1.81")
print("=" * 70)
print(f"Resolution:   Nx={Nx}, Nz={Nz}, L={L}, dx={dx:.4f}")
print(f"State arrays: 3 × ({Nz+1}, {Nx}, {Nx//2+1}) complex128")
print(f"State memory: {3 * (Nz+1) * Nx * (Nx//2+1) * 16 / 1e6:.0f} MB")
print(f"Dissipation:  Laplacian nu={nu}")
print(f"CFL target:   {cfl_target}, dt_max={dt_max}")
print(f"Output:       {output_dir}/")
print(f"Device:       {jax.devices()}")
print()


def build_grid_and_jit(dt_val):
    """Build grid with given dt and JIT the stepper."""
    cfg = NHQGConfig(
        Nx=Nx, Nz=Nz, L=L,
        Ra_tilde=100.0, sigma=1.0,
        beta=0.0, Ld=float('inf'),
        dt=dt_val, t_final=1000.0,
        nu_q=nu, hyper_order=1,
        nu_w=nu, nu_theta=nu,
        save_interval=save_every,
        output_dir=output_dir,
        float_dtype='float64',
    )
    grid = make_grid(cfg)

    @jax.jit
    def scan_body(state, _):
        return imex_step(state, grid), None

    return cfg, grid, scan_body


# --- Initial setup ---
t0_total = time.time()
cfg, grid, scan_body = build_grid_and_jit(dt)
state = make_initial_state(grid, seed=0, amplitude=1e-3)

# JIT warmup
print(f"JIT compiling (Nz={Nz}, dt={dt:.2e})...", flush=True)
t0_jit = time.time()
state_warmup, _ = jax.lax.scan(scan_body, state, None, length=1)
jax.block_until_ready(state_warmup.q_hat)
print(f"JIT compilation: {time.time()-t0_jit:.1f}s")
print()

# --- Main loop ---
print(f"{'step':>8s} {'t':>8s} {'dt':>10s} {'wall':>8s} {'ms/step':>8s} "
      f"{'CFL':>8s} {'KE_bt':>12s} {'KE_bc':>12s} "
      f"{'Nusselt':>10s} {'max_v':>10s}")
print("-" * 115)

step = 0
t_sim = 0.0
t0_run = time.time()
n_rebuilds = 0
diag_count = 0
Nu_history = []

while True:
    wall_elapsed = time.time() - t0_run
    if wall_elapsed > wall_limit:
        print(f"\nWall time limit ({wall_limit}s) reached.")
        break

    # Run a chunk
    t0_chunk = time.time()
    state, _ = jax.lax.scan(scan_body, state, None, length=save_every)
    jax.block_until_ready(state.q_hat)
    chunk_time = time.time() - t0_chunk

    step += save_every
    t_sim += save_every * dt
    wall = time.time() - t0_run
    ms_per_step = 1000 * chunk_time / save_every

    # Diagnostics
    diag = compute_diagnostics(state, grid)
    max_v = float(diag['max_speed'])
    Nu = float(diag['Nusselt'])
    cfl = max_v * dt / dx

    print(f"{step:8d} {t_sim:8.3f} {dt:10.2e} {wall:8.1f} {ms_per_step:8.2f} "
          f"{cfl:8.4f} {float(diag['KE_bt']):12.4e} {float(diag['KE_bc']):12.4e} "
          f"{Nu:10.4f} {max_v:10.4f}",
          flush=True)

    diag_count += 1
    Nu_history.append(Nu)

    # Checkpoint every 200 chunks
    if diag_count % 200 == 0 and diag_count > 0:
        save_checkpoint(state, step, cfg)

    # NaN check
    if np.isnan(max_v):
        print("NaN detected — stopping.")
        break

    # --- Adaptive CFL ---
    if max_v > 0:
        dt_cfl = cfl_target * dx / max_v
        dt_new = min(dt_cfl, dt_max)

        if dt_new < 0.8 * dt or dt_new > 1.3 * dt:
            dt_old = dt
            dt = dt_new

            print(f"  >>> CFL adapt: dt {dt_old:.2e} -> {dt:.2e} "
                  f"(rebuilding grid...)", end='', flush=True)
            t0_rebuild = time.time()
            cfg, grid, scan_body = build_grid_and_jit(dt)
            _, _ = jax.lax.scan(scan_body, state, None, length=1)
            jax.block_until_ready(state.q_hat)
            n_rebuilds += 1
            print(f" done in {time.time()-t0_rebuild:.1f}s "
                  f"(rebuild #{n_rebuilds})", flush=True)

# --- Summary ---
total_wall = time.time() - t0_run
print()
print(f"Completed: {step} steps, t={t_sim:.3f}, wall={total_wall:.1f}s, "
      f"{n_rebuilds} dt rebuilds")
print(f"Final dt={dt:.2e}")

# Save final checkpoint
os.makedirs(output_dir, exist_ok=True)
save_checkpoint(state, step, cfg)

# Summary statistics
if len(Nu_history) >= 20:
    recent = Nu_history[-20:]
    print(f"\nLast 20 diagnostics: Nu = {np.mean(recent):.2f} ± {np.std(recent):.2f}")
    print(f"Miquel target: 30.96 ± 1.81")
    print(f"Ratio (ours/Miquel): {np.mean(recent)/30.96:.2f}")

# Final spectrum
psi_hat = invert_psi(state.q_hat, grid.inv_denom)
k_bins, E_bt, E_bc, E_tot = energy_spectrum(psi_hat, grid.ksq, grid.cc_weights, L)

spec_data = np.column_stack([
    np.array(k_bins), np.array(E_bt), np.array(E_bc), np.array(E_tot)
])
np.savetxt(os.path.join(output_dir, 'spectrum_final.txt'), spec_data,
           header='k  E_bt  E_bc  E_tot', fmt='%.8e')
print(f"Saved spectrum to {output_dir}/spectrum_final.txt")
