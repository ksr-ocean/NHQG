#!/usr/bin/env python
"""Case 1: Rubio et al. 2014 reproduction.

beta=0, Ld=inf, Ra=100, sigma=1, L=20.
Laplacian diffusion (p=1, nu=1) on ALL fields: q', w, theta.
Matches Miquel's setup: molecular diffusion, no vertical diffusion.
Fixed dt with conservative CFL.
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import sys
import time

import jax
import jax.numpy as jnp

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.paths import normalize_output_dir
from nhqg.solver import make_initial_state, run
from nhqg.diagnostics import compute_diagnostics
from nhqg.io import save_snapshot, save_checkpoint


def main():
    # --- Configuration ---
    Nx = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    Nz = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    L = 20.0

    # --- Dissipation: Laplacian (p=1, nu=1) on ALL fields ---
    # Matches Miquel: molecular diffusion nu=1, no vertical diffusion.
    p = 1
    nu = 1.0

    k_max = float(jnp.pi * Nx / L)
    dt = 1e-4  # conservative CFL: safe for max_v ~ 200 (dx/max_v ~ 2e-4)

    output_dir = normalize_output_dir(f'output_Ra100_Nx{Nx}_Nz{Nz}')

    cfg = NHQGConfig(
        Nx=Nx, Nz=Nz, L=L,
        Ra_tilde=100.0, sigma=1.0,
        beta=0.0, Ld=float('inf'),
        dt=dt, t_final=20.0,
        nu_q=nu, hyper_order=p,
        nu_w=nu, nu_theta=nu,
        save_interval=1000,  # save every 1000 steps = every t=0.1
        output_dir=output_dir,
        float_dtype='float64',
    )

    # Print dissipation diagnostics
    rate_kmax = nu * k_max ** (2 * p)
    rate_kc = nu * 1.3048 ** (2 * p)
    alpha_kmax = 1 + 0.293 * dt * rate_kmax
    print(f"Rubio Case 1: Nx={Nx}, Nz={Nz}, L={L}, Ra={cfg.Ra_tilde}")
    print(f"dt={dt}, t_final={cfg.t_final}, hyper_order={p} (Laplacian)")
    print(f"nu (all fields) = {nu:.4e}")
    print(f"  rate at k_max={k_max:.1f}: {rate_kmax:.1f} "
          f"(exp(-rate*dt)={float(jnp.exp(-rate_kmax*dt)):.4e}, "
          f"IMEX alpha={alpha_kmax:.2f})")
    print(f"  rate at k_c=1.3: {rate_kc:.2e}")
    print(f"  CFL limit: dt < dx/v_max = {L/Nx:.4e}/v_max")
    print(f"  q_boundary: {cfg.q_boundary}")
    print(f"Device: {jax.devices()}")
    print()

    # --- Grid + IC ---
    t0 = time.time()
    grid = make_grid(cfg)
    state = make_initial_state(grid, seed=0, amplitude=1e-3)
    print(f"Grid + IC setup: {time.time()-t0:.1f}s")

    # --- Run ---
    total_steps = int(cfg.t_final / cfg.dt)

    def callback(state, step, t):
        diag = compute_diagnostics(state, grid)
        max_v = float(diag['max_speed'])
        cfl = max_v * dt / (L / Nx)
        print(f"  step={step:8d}  t={t:8.3f}  "
              f"KE_bt={float(diag['KE_bt']):10.4e}  "
              f"KE_bc={float(diag['KE_bc']):10.4e}  "
              f"Nu={float(diag['Nusselt']):8.4f}  "
              f"max_v={max_v:8.4f}  "
              f"CFL={cfl:.3f}")

        if cfl > 0.5:
            print(f"  *** CFL WARNING: {cfl:.2f} (reduce dt!) ***")

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
