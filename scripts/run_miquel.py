#!/usr/bin/env python
"""Miquel et al. 2026 exact reproduction at ϑ_f=0° (upright convection).

Matching their baseline as closely as the current solver permits:
  - Domain: 10 L_c × 10 L_c × 1, where L_c = 2π/k_c ≈ 4.815
  - Laplacian diffusion on all fluctuation fields
  - No explicit q-boundary condition (Miquel-style)
  - Resolution from their Table 1:
      Ra ≤ 80:  128² × 256
      Ra ≥ 100: 256² × 384

Miquel targets at ϑ_f=0°:
  Ra=10:  Nu=1.27±0.01,  Re=0.75±0.11
  Ra=20:  Nu=4.02±0.13,  Re=3.55±0.79
  Ra=40:  Nu=12.28±0.60, Re=10.67±2.43
  Ra=80:  Nu=30.96±1.81, Re=24.28±7.39
  Ra=100: Nu=43.37±2.54, Re=32.05±8.24
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


# Critical wavenumber and wavelength (stress-free, upright)
Ra_c = 8.6956
k_c = 1.3048
L_c = 2 * jnp.pi / k_c  # ≈ 4.815


def main():
    Ra = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0

    # --- Resolution from Miquel Table 1 ---
    if Ra <= 80:
        Nx, Nz = 128, 256
    else:
        Nx, Nz = 256, 384

    # Domain: 10 L_c (matching Miquel exactly)
    L = float(10.0 * L_c)

    # Miquel uses Laplacian diffusion on all fluctuation fields.
    p = 1
    dx = L / Nx
    k_max = float(jnp.pi * Nx / L)

    # Time step: CFL-based on advection only.
    dt = 0.3 * dx / 100.0  # CFL = 0.3 at max_v=100

    # Optional experiment: keep Chebyshev modes n <= cutoff_n for w,theta only.
    cutoff_n = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # t_final: need long enough for barotropic saturation
    # Miquel runs until barotropic flow saturates
    t_final = 20.0

    output_dir = normalize_output_dir(f'output_miquel_Ra{int(Ra)}_Nx{Nx}_Nz{Nz}')

    cfg = NHQGConfig(
        Nx=Nx, Nz=Nz, L=L,
        Ra_tilde=Ra, sigma=1.0,
        beta=0.0, Ld=float('inf'),
        dt=dt, t_final=t_final,
        imex_scheme='rk443',
        nu_q=1.0, hyper_order=p,
        nu_w=1.0, nu_theta=1.0,
        q_boundary='none',
        vertical_cutoff_n=cutoff_n,
        save_interval=max(1, int(0.1 / dt)),  # save every ~0.1 time units
        output_dir=output_dir,
        float_dtype='float64',
        thermal_closure='evolve_mean',
        mean_temp_eps_sq=1.0,
    )

    print(f"=== Miquel reproduction: Ra={Ra}, ϑ_f=0° ===")
    print(f"Nx={Nx}, Nz={Nz}, L={L:.2f} ({L/float(L_c):.1f} L_c)")
    print(
        f"Laplacian diffusion, dt={dt:.4e}, t_final={t_final}, q_bc={cfg.q_boundary}, "
        f"closure={cfg.thermal_closure}"
    )
    print(f"IMEX scheme: {cfg.imex_scheme}")
    print(f"  rate@k_max={k_max:.1f}: {k_max**(2*p):.0f}, rate@k_c: {k_c**(2*p):.2e}")
    print(f"  vertical_cutoff_n={cutoff_n}")
    print(f"dx={dx:.4f}, CFL limit: max_v < {0.5*dx/dt:.0f}")
    print(f"State memory: {3*(Nz+1)*Nx*(Nx//2+1)*16/1e6:.0f} MB")
    print(f"Device: {jax.devices()}")
    print(flush=True)

    # Miquel targets
    miquel_targets = {
        10: (1.27, 0.75), 20: (4.02, 3.55), 40: (12.28, 10.67),
        60: (19.88, 17.19), 80: (30.96, 24.28),
        100: (43.37, 32.05), 120: (58.84, 41.16),
    }
    if int(Ra) in miquel_targets:
        nu_target, re_target = miquel_targets[int(Ra)]
        print(f"Target: Nu={nu_target}, Re_ℓ={re_target}")

    # --- Grid + IC ---
    t0 = time.time()
    grid = make_grid(cfg)
    state = make_initial_state(grid, seed=0, amplitude=1e-3)
    print(f"Grid + IC setup: {time.time()-t0:.1f}s", flush=True)

    # --- Run ---
    total_steps = int(cfg.t_final / cfg.dt)
    print(f"Total steps: {total_steps}", flush=True)

    def callback(state, step, t):
        diag = compute_diagnostics(state, grid)
        max_v = float(diag['max_speed'])
        cfl = max_v * dt / dx
        nu_val = float(diag['Nusselt'])
        flag = '  *** CFL > 0.5! ***' if cfl > 0.5 else ''
        print(f"  step={step:8d}  t={t:8.3f}  "
              f"KE_bt={float(diag['KE_bt']):10.4e}  "
              f"KE_bc={float(diag['KE_bc']):10.4e}  "
              f"Nu={nu_val:8.4f}  "
              f"max_v={max_v:8.4f}  CFL={cfl:.4f}  "
              f"|th_bar|={float(jnp.max(jnp.abs(state.th_bar))):8.2e}  "
              f"w_hi={float(diag['w_high_frac']):6.3f}  "
              f"th_hi={float(diag['th_high_frac']):6.3f}{flag}",
              flush=True)

        # Checkpoint every 20 saves
        if step % (cfg.save_interval * 20) == 0:
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
