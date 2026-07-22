#!/usr/bin/env python
"""Ra=100 production run with Chebyshev filter fix.

Miquel target: Nu = 43.37 +/- 2.54 at 256^2 x 384.
We use Nx=256, Nz=32, hyper-4 dissipation on all fields.
Saves NetCDF snapshots for visualization.
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
from nhqg.solver import make_initial_state, imex_step, State
from nhqg.diagnostics import compute_diagnostics
from nhqg.io import save_snapshot, save_checkpoint


def main():
    Nx = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    Nz = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    Ra = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0

    Lc = 2.0 * np.pi / 1.3048
    L = 10.0 * Lc  # 10 critical wavelengths

    # Dissipation: hyper-4 on all fields
    p = 4
    k_max = float(jnp.pi * Nx / L)
    dt = 5e-5  # conservative for Ra=100 (max growth rate ~8.6)
    n_efold = 5.0
    nu_all = n_efold / (dt * k_max ** (2 * p))

    output_dir = normalize_output_dir(f'output_Ra{int(Ra)}_Nx{Nx}_Nz{Nz}')

    cfg = NHQGConfig(
        Nx=Nx, Nz=Nz, L=L,
        Ra_tilde=Ra, sigma=1.0,
        beta=0.0, Ld=float('inf'),
        dt=dt, t_final=50.0,
        nu_q=nu_all, hyper_order=p,
        nu_w=nu_all, nu_theta=nu_all,
        save_interval=2000,  # save every 2000 steps = every t=0.1
        output_dir=output_dir,
        float_dtype='float64',
        thermal_closure='evolve_mean',
        mean_temp_eps_sq=1.0,
    )

    rate_kmax = nu_all * k_max ** (2 * p)
    rate_kc = nu_all * 1.3048 ** (2 * p)
    print(f"Ra={Ra}, Nx={Nx}, Nz={Nz}, L={L:.2f}, dt={dt}")
    print(f"Hyper-{p} dissipation, nu={nu_all:.4e}")
    print(f"  rate at k_max={k_max:.1f}: {rate_kmax:.1f}")
    print(f"  rate at k_c=1.3: {rate_kc:.2e}")
    print(f"Output: {output_dir}")
    print(f"Device: {jax.devices()}")
    print()

    # Grid + IC
    t0 = time.time()
    grid = make_grid(cfg)
    state = make_initial_state(grid, seed=0, amplitude=1e-2)
    print(f"Grid + IC setup: {time.time()-t0:.1f}s")

    # JIT compile
    @jax.jit
    def scan_body(state, _):
        return imex_step(state, grid), None

    t0 = time.time()
    state, _ = jax.lax.scan(scan_body, state, None, length=10)
    jax.block_until_ready(state.q_hat)
    print(f"JIT compile: {time.time()-t0:.1f}s")
    print()

    # Main loop
    total_steps = int(cfg.t_final / cfg.dt)
    save_every = cfg.save_interval
    n_blocks = total_steps // save_every

    t0 = time.time()
    for block in range(n_blocks):
        state, _ = jax.lax.scan(scan_body, state, None, length=save_every)
        jax.block_until_ready(state.q_hat)
        step = (block + 1) * save_every + 10
        t_sim = step * dt
        diag = compute_diagnostics(state, grid)
        Nu = float(diag['Nusselt'])
        KE = float(diag['KE_tot'])
        max_v = float(diag['max_speed'])
        elapsed = time.time() - t0

        print(f"  step={step:8d}  t={t_sim:8.3f}  "
              f"Nu={Nu:10.4f}  KE={KE:10.4e}  max_v={max_v:8.4f}  "
              f"[{elapsed:.0f}s]")

        # Save NetCDF snapshot
        save_snapshot(state, t_sim, step, cfg, grid, output_dir)

        # Checkpoint every 10 saves
        if step % (save_every * 10) == 0:
            save_checkpoint(state, step, cfg, output_dir)

        # Check for blowup
        if not jnp.isfinite(KE) or KE > 1e10:
            print("  *** BLOWUP ***")
            break

        # CFL check
        cfl = max_v * dt / (L / Nx)
        if cfl > 0.5:
            print(f"  *** CFL WARNING: {cfl:.2f} ***")

    elapsed = time.time() - t0
    print(f"\nCompleted {step} steps in {elapsed:.1f}s "
          f"({step/elapsed:.0f} steps/s)")

    # Final checkpoint
    save_checkpoint(state, step, cfg, output_dir)


if __name__ == '__main__':
    main()
