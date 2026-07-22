#!/usr/bin/env python
"""NHQGE with evolving mean temperature and moderate hyperviscosity.

Usage:
    JAX_ENABLE_X64=1 PYTHONPATH=. python scripts/run_evolve.py [Nx] [Nz] [Ra]

Defaults: Nx=256, Nz=128, Ra=100.
Target: Miquel Table 1 (ϑ_f=0°): Nu = 43.37 ± 2.54, Re_ℓ = 32.05 ± 8.24
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.paths import normalize_output_dir
from nhqg.solver import make_initial_state, run
from nhqg.diagnostics import compute_diagnostics
from nhqg.io import save_snapshot, save_checkpoint


def main():
    Nx = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    Nz = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    Ra = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    L = 20.0

    # --- Time step ---
    # CFL: dt < dx/v_max.  At Ra=100, max_v ~ 200-400.
    # dx = L/Nx = 0.078 at Nx=256.  dt=1e-4 → CFL ≈ 0.13-0.51.
    dt = 1e-4

    # --- Dissipation: aggressive hyper-4 (n_efold=5) ---
    p = 4
    k_max = float(np.pi * Nx / L)
    n_efold = 5.0
    nu_all = n_efold / (dt * k_max ** (2 * p))
    k_d = nu_all ** (-1.0 / (2 * p))

    output_dir = normalize_output_dir(f'output_Ra{int(Ra)}_Nx{Nx}_Nz{Nz}_evolve')

    cfg = NHQGConfig(
        Nx=Nx, Nz=Nz, L=L,
        Ra_tilde=Ra, sigma=1.0,
        beta=0.0, Ld=float('inf'),
        dt=dt, t_final=10.0,
        nu_q=nu_all, hyper_order=p,
        nu_w=nu_all, nu_theta=nu_all,
        thermal_closure="evolve_mean",
        mean_temp_eps_sq=1.0,
        save_interval=1000,
        output_dir=output_dir,
        float_dtype='float64',
    )

    # --- Print dissipation diagnostics ---
    rate_kmax = nu_all * k_max ** (2 * p)
    rate_kc = nu_all * 1.3048 ** (2 * p)
    rate_10 = nu_all * 10.0 ** (2 * p)
    print(f"NHQGE: Ra={Ra}, Nx={Nx}, Nz={Nz}, L={L}")
    print(
        f"  dt={dt}, t_final={cfg.t_final}, closure={cfg.thermal_closure}, "
        f"eps^2={cfg.mean_temp_eps_sq}"
    )
    print(f"  hyper_order p={p},  nu = {nu_all:.4e}")
    print(f"  k_d = {k_d:.1f}  (equiv dissipation wavenumber)")
    print(f"  k_max = {k_max:.1f},  k_c ≈ 1.3,  dx = {L/Nx:.4f}")
    print(f"  rate at k_max={k_max:.0f}: {rate_kmax:.0f}  "
          f"(per-step factor: {float(np.exp(-rate_kmax*dt)):.4f})")
    print(f"  rate at k=10:       {rate_10:.2e}  "
          f"(e-fold time: {1.0/rate_10:.1f})")
    print(f"  rate at k_c=1.3:    {rate_kc:.2e}  (negligible)")
    print(f"  CFL limit: v_max < dx/dt = {L/Nx/dt:.0f}")
    print()
    print(f"  Miquel targets (Ra=100): Nu = 43.37 ± 2.54,  Re_ℓ = 32.05 ± 8.24")
    print(f"  NOTE: Miquel uses Laplacian ν=1 (our hyper-4 is less dissipative at k_c)")
    print(f"  NOTE: Miquel uses L ≈ 48 (10 L_c), Nz=384;  we use L={L}, Nz={Nz}")
    print()
    print(f"Device: {jax.devices()}")
    print()

    # --- Grid + IC ---
    t0 = time.time()
    grid = make_grid(cfg)
    state = make_initial_state(grid, seed=0, amplitude=1e-3)
    print(f"Grid + IC setup: {time.time()-t0:.1f}s")

    # IMEX memory estimate
    n_shells = grid.imex_inv.shape[0]
    imex_mb = 2 * n_shells * (Nz+1)**2 * 8 / 1e6
    print(f"IMEX shells: {n_shells},  IMEX memory: {imex_mb:.0f} MB")
    print()

    # --- Run ---
    total_steps = int(cfg.t_final / cfg.dt)

    def callback(state, step, t):
        diag = compute_diagnostics(state, grid)
        max_v = float(diag['max_speed'])
        cfl = max_v * dt / (L / Nx)

        # Mean temperature profile extremes
        th_bar_max = float(jnp.max(jnp.abs(state.th_bar)))

        print(f"  step={step:8d}  t={t:8.3f}  "
              f"Nu={float(diag['Nusselt']):8.3f}  "
              f"KE_bt={float(diag['KE_bt']):10.4e}  "
              f"KE_bc={float(diag['KE_bc']):10.4e}  "
              f"max_v={max_v:8.2f}  CFL={cfl:.3f}  "
              f"|Θ̄'|={th_bar_max:.4f}")

        if cfl > 0.5:
            print(f"  *** CFL WARNING: {cfl:.2f} — reduce dt! ***")

        # Check for blowup
        if not jnp.isfinite(float(diag['KE_tot'])):
            print("  *** BLOWUP DETECTED — aborting ***")
            save_checkpoint(state, step, cfg)
            sys.exit(1)

        # Save snapshot
        save_snapshot(state, t, step, cfg, grid)

        # Checkpoint every 10 saves
        if step % (cfg.save_interval * 10) == 0:
            save_checkpoint(state, step, cfg)

    t0 = time.time()
    final_state, snapshots = run(grid, state, n_steps=total_steps,
                                  save_interval=cfg.save_interval,
                                  use_imex=True, callback=callback)
    elapsed = time.time() - t0
    print(f"\nCompleted {total_steps} steps in {elapsed:.1f}s "
          f"({total_steps/elapsed:.0f} steps/s)")

    # Final checkpoint
    save_checkpoint(final_state, total_steps, cfg)


if __name__ == '__main__':
    main()
