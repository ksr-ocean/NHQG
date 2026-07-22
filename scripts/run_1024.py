#!/usr/bin/env python
"""High-resolution 1024^2 run with hyper-4 on all fields.

Higher resolution pushes the dissipation wavenumber to larger k,
giving a wider inertial/advective range compared to 512^2.
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
from nhqg.io import save_checkpoint


def main():
    Nx = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    Nz = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    L = 20.0

    # --- Dissipation: hyper-4 on ALL fields ---
    p = 4
    k_max = float(jnp.pi * Nx / L)
    # At 1024, k_max ~ 161, so k_max^8 ~ 4.3e17
    # CFL: dx = L/Nx = 0.0195, dt < dx/max_v.
    # For max_v ~ 100: dt < 1.95e-4. Use dt=1e-4 (safe).
    dt = 1e-4
    n_efold = 5.0
    nu_all = n_efold / (dt * k_max ** (2 * p))

    output_dir = normalize_output_dir(f'output_1024_Nz{Nz}_hyper{p}')

    cfg = NHQGConfig(
        Nx=Nx, Nz=Nz, L=L,
        Ra_tilde=100.0, sigma=1.0,
        beta=0.0, Ld=float('inf'),
        dt=dt, t_final=10.0,
        nu_q=nu_all, hyper_order=p,
        nu_w=nu_all, nu_theta=nu_all,
        save_interval=2000,  # save every 2000 steps = every t=0.2
        output_dir=output_dir,
        float_dtype='float64',
    )

    # Dissipation diagnostics
    rate_kmax = nu_all * k_max ** (2 * p)
    rate_kc = nu_all * 1.3048 ** (2 * p)
    alpha_kmax = 1 + 0.293 * dt * rate_kmax
    print(f"=== Nx={Nx}, Nz={Nz}, L={L}, Ra=100, hyper_order={p} ===")
    print(f"dt={dt}, t_final={cfg.t_final}")
    print(f"nu = {nu_all:.4e}")
    print(f"  rate@k_max={k_max:.1f}: {rate_kmax:.0f} "
          f"(exp(-rate*dt)={float(jnp.exp(-rate_kmax*dt)):.4e}, "
          f"alpha={alpha_kmax:.2f})")
    print(f"  rate@k_c=1.3: {rate_kc:.2e} (negligible)")
    print(f"  CFL limit: max_v < {L/(Nx*dt):.0f}")
    print(f"  k_max/k_c ratio: {k_max/1.3048:.0f} "
          f"(vs 512: {jnp.pi*512/L/1.3048:.0f})")
    print(f"Device: {jax.devices()}")
    print(flush=True)

    # --- Grid + IC ---
    t0 = time.time()
    grid = make_grid(cfg)
    state = make_initial_state(grid, seed=0, amplitude=1e-3)
    print(f"Grid + IC setup: {time.time()-t0:.1f}s", flush=True)

    # --- Run ---
    total_steps = int(cfg.t_final / cfg.dt)

    def callback(state, step, t):
        diag = compute_diagnostics(state, grid)
        max_v = float(diag['max_speed'])
        cfl = max_v * dt / (L / Nx)
        flag = '  *** CFL > 0.5! ***' if cfl > 0.5 else ''
        print(f"  step={step:8d}  t={t:8.3f}  "
              f"KE_bt={float(diag['KE_bt']):10.4e}  "
              f"KE_bc={float(diag['KE_bc']):10.4e}  "
              f"Nu={float(diag['Nusselt']):8.4f}  "
              f"max_v={max_v:8.4f}  CFL={cfl:.4f}{flag}",
              flush=True)

        # Checkpoint every 5 saves
        if step % (cfg.save_interval * 5) == 0:
            save_checkpoint(state, step, cfg)

    t0 = time.time()
    final_state, snapshots = run(grid, state, n_steps=total_steps,
                                  save_interval=cfg.save_interval,
                                  use_imex=True, callback=callback)
    elapsed = time.time() - t0
    print(f"\nCompleted {total_steps} steps in {elapsed:.1f}s "
          f"({total_steps/elapsed:.0f} steps/s)")

    save_checkpoint(final_state, total_steps, cfg)


if __name__ == '__main__':
    main()
