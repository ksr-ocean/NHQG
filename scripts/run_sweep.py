#!/usr/bin/env python
"""Parametric (beta, Ld) sweep.

Groups runs by Ld (each needs different IMEX LU factors).
Within each Ld group, runs different beta values sequentially.
(True vmap batching over beta requires batched grid/state.)
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import sys
import time
import itertools

import jax
import jax.numpy as jnp
import numpy as np

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.paths import normalize_output_dir
from nhqg.solver import make_initial_state, imex_step
from nhqg.diagnostics import compute_diagnostics
from nhqg.io import save_snapshot, save_checkpoint


def main():
    Nx = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    Nz = 32

    beta_values = [0.0, 0.5, 1.0, 2.0, 5.0]
    Ld_values = [2.0, 3.0, 5.0, 10.0, float('inf')]

    dt = 5e-4
    t_final = 50.0
    total_steps = int(t_final / dt)
    save_interval = 500

    k_max = np.pi * Nx / 20.0
    nu_q = 5.0 * dt / k_max ** 8

    print(f"Parametric sweep: Nx={Nx}, Nz={Nz}")
    print(f"beta = {beta_values}")
    print(f"Ld = {Ld_values}")
    print(f"Total: {len(beta_values) * len(Ld_values)} runs")
    print(f"Device: {jax.devices()}")
    print()

    results = {}

    for Ld in Ld_values:
        Ld_str = 'inf' if Ld == float('inf') else f'{Ld:.1f}'
        print(f"=== Ld = {Ld_str} ===")

        for beta in beta_values:
            run_name = f"beta{beta:.1f}_Ld{Ld_str}"
            output_dir = normalize_output_dir(f"output_sweep_Nx{Nx}/{run_name}")

            cfg = NHQGConfig(
                Nx=Nx, Nz=Nz, L=20.0,
                Ra_tilde=100.0, sigma=1.0,
                beta=beta, Ld=Ld,
                dt=dt, t_final=t_final,
                nu_q=nu_q, hyper_order=4,
                nu_w=1e-3, nu_theta=1e-3,
                save_interval=save_interval,
                output_dir=output_dir,
                float_dtype='float64',
            )

            grid = make_grid(cfg)
            state = make_initial_state(grid, seed=42, amplitude=1e-3)

            t0 = time.time()

            # Use lax.scan for JIT inner loop
            @jax.jit
            def scan_body(state, _):
                return imex_step(state, grid), None

            for i_outer in range(total_steps // save_interval):
                state, _ = jax.lax.scan(scan_body, state, None,
                                         length=save_interval)
                step = (i_outer + 1) * save_interval
                t = step * dt

                if i_outer % 10 == 0:
                    diag = compute_diagnostics(state, grid)
                    print(f"  [{run_name}] step={step:6d} t={t:6.2f} "
                          f"KE_bt={float(diag['KE_bt']):9.3e}")

            elapsed = time.time() - t0
            diag_final = compute_diagnostics(state, grid)

            results[(beta, Ld)] = {
                'KE_bt': float(diag_final['KE_bt']),
                'KE_bc': float(diag_final['KE_bc']),
                'Nusselt': float(diag_final['Nusselt']),
                'time': elapsed,
            }

            save_checkpoint(state, total_steps, cfg, output_dir)
            print(f"  [{run_name}] done in {elapsed:.1f}s")

    # Summary table
    print("\n=== Summary ===")
    print(f"{'beta':>6s} {'Ld':>6s} {'KE_bt':>10s} {'KE_bc':>10s} "
          f"{'Nusselt':>8s} {'time(s)':>8s}")
    for (beta, Ld), r in results.items():
        Ld_str = 'inf' if Ld == float('inf') else f'{Ld:.1f}'
        print(f"{beta:6.1f} {Ld_str:>6s} {r['KE_bt']:10.3e} "
              f"{r['KE_bc']:10.3e} {r['Nusselt']:8.4f} {r['time']:8.1f}")


if __name__ == '__main__':
    main()
