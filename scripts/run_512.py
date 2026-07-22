#!/usr/bin/env python
"""512x32 Rubio/Miquel case with adaptive CFL time stepping.

Uses Laplacian dissipation (nu=1) matching Miquel et al. 2026 fNHQGE.
Adjusts dt to maintain CFL < 0.3 during the transient growth phase.
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
from nhqg.diagnostics import compute_diagnostics, barotropic_mode, energy_spectrum
from nhqg.io import save_snapshot, save_checkpoint

# --- Configuration ---
Nx, Nz = 512, 32
L = 20.0
dx = L / Nx
dt_max = 5e-4       # maximum dt (used when flow is calm)
dt = dt_max
cfl_target = 0.3    # target CFL number (conservative)

# Dissipation — Laplacian (matching Miquel et al. 2026, eq 3.1)
# nu=1 in the non-dimensional NHQGE: diss = exp(-nu * |k|^2 * dt)
nu = 1.0

output_dir = normalize_output_dir('output_512')
save_every = 50      # check CFL every 50 steps for responsiveness during growth
wall_limit = 3400    # stop after ~56 minutes

print("=" * 70)
print("NHQGE 512x32 Rubio/Miquel Case (beta=0, Ld=inf, Ra=100)")
print("=" * 70)
print(f"Resolution:   Nx={Nx}, Nz={Nz}, L={L}, dx={dx:.4f}")
print(f"Dissipation:  Laplacian nu={nu} (Miquel et al. 2026)")
print(f"CFL target:   {cfl_target}, dt_max={dt_max}")
print(f"Output:       {output_dir}/")
print(f"Device:       {jax.devices()}")
print(f"JAX:          {jax.__version__}")
print()


def build_grid_and_jit(dt_val):
    """Build grid with given dt and JIT the stepper."""
    cfg = NHQGConfig(
        Nx=Nx, Nz=Nz, L=L,
        Ra_tilde=100.0, sigma=1.0,
        beta=0.0, Ld=float('inf'),
        dt=dt_val, t_final=1000.0,
        nu_q=nu, hyper_order=1,   # Laplacian dissipation
        nu_w=nu, nu_theta=nu,     # 1/sigma factor applied in grid.py
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
print(f"JIT compiling (dt={dt:.2e})...", flush=True)
t0_jit = time.time()
state_warmup, _ = jax.lax.scan(scan_body, state, None, length=1)
jax.block_until_ready(state_warmup.q_hat)
print(f"JIT compilation: {time.time()-t0_jit:.1f}s")
print()

# --- Main loop with adaptive CFL ---
print(f"{'step':>8s} {'t':>8s} {'dt':>10s} {'wall':>8s} {'ms/step':>8s} "
      f"{'CFL':>8s} {'KE_bt':>12s} {'KE_bc':>12s} "
      f"{'Nusselt':>10s} {'max_v':>10s}")
print("-" * 110)

step = 0
t_sim = 0.0
t0_run = time.time()
n_rebuilds = 0
diag_count = 0

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
    cfl = max_v * dt / dx

    print(f"{step:8d} {t_sim:8.3f} {dt:10.2e} {wall:8.1f} {ms_per_step:8.2f} "
          f"{cfl:8.4f} {float(diag['KE_bt']):12.4e} {float(diag['KE_bc']):12.4e} "
          f"{float(diag['Nusselt']):10.4f} {max_v:10.4f}",
          flush=True)

    diag_count += 1

    # Save NetCDF snapshot every 20 chunks (~1000 steps at base dt)
    if diag_count % 20 == 0:
        save_snapshot(state, t_sim, step, cfg, grid)

    # Checkpoint every 100 chunks
    if diag_count % 100 == 0 and diag_count > 0:
        save_checkpoint(state, step, cfg)

    # Bail on NaN
    if np.isnan(max_v):
        print("NaN detected — stopping.")
        break

    # --- Adaptive CFL ---
    if max_v > 0:
        dt_cfl = cfl_target * dx / max_v
        dt_new = min(dt_cfl, dt_max)

        # Only rebuild if dt changes by more than 20%
        if dt_new < 0.8 * dt or dt_new > 1.3 * dt:
            dt_old = dt
            dt = dt_new

            # Rebuild grid with new dt (re-JIT takes ~5s)
            print(f"  >>> CFL adapt: dt {dt_old:.2e} -> {dt:.2e} "
                  f"(rebuilding grid...)", end='', flush=True)
            t0_rebuild = time.time()
            cfg, grid, scan_body = build_grid_and_jit(dt)
            # JIT warmup
            _, _ = jax.lax.scan(scan_body, state, None, length=1)
            jax.block_until_ready(state.q_hat)
            n_rebuilds += 1
            print(f" done in {time.time()-t0_rebuild:.1f}s "
                  f"(rebuild #{n_rebuilds})", flush=True)

# --- Summary ---
total_wall = time.time() - t0_run
print()
print(f"Completed: {step} steps, t={t_sim:.2f}, wall={total_wall:.1f}s, "
      f"{n_rebuilds} dt rebuilds")
print(f"Final dt={dt:.2e}")

# Final checkpoint
save_checkpoint(state, step, cfg)

# Energy spectrum of final state
psi_hat = invert_psi(state.q_hat, grid.inv_denom)
k_bins, E_bt, E_bc, E_tot = energy_spectrum(psi_hat, grid.ksq, grid.cc_weights, L)

os.makedirs(output_dir, exist_ok=True)
spec_data = np.column_stack([
    np.array(k_bins), np.array(E_bt), np.array(E_bc), np.array(E_tot)
])
np.savetxt(os.path.join(output_dir, 'spectrum_final.txt'), spec_data,
           header='k  E_bt  E_bc  E_tot', fmt='%.8e')
print(f"Saved final spectrum to {output_dir}/spectrum_final.txt")
print(f"Snapshots in {output_dir}/snapshot_*.nc")
