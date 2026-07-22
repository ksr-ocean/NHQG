"""Stability test with correct growth rate analysis and small dt.

The NHQGE dispersion relation (stress-free, n=1 vertical mode):
  (s + k^2)^2 = Ra - pi^2/k^2
  s = -k^2 + sqrt(Ra - pi^2/k^2)

At Ra=100, k=1: s = -1 + sqrt(90) = 8.49 (very fast growth).
The previous tests blew up because CFL was violated at high amplitudes.
This test uses smaller dt to avoid CFL issues and see if nonlinear
saturation actually occurs.
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import sys
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.solver import State, make_initial_state, imex_step, invert_psi


def max_amplitude(state):
    q_max = float(jnp.max(jnp.abs(state.q_hat)))
    w_max = float(jnp.max(jnp.abs(state.w_hat)))
    th_max = float(jnp.max(jnp.abs(state.th_hat)))
    return q_max, w_max, th_max


def compute_cfl(state, grid):
    """Estimate CFL = u_max * dt / dx."""
    psi = invert_psi(state.q_hat, grid.inv_denom)
    # u = -dpsi/dy, v = dpsi/dx
    u_hat = -1j * grid.ky[None, :, :] * psi
    v_hat = 1j * grid.kx[None, :, :] * psi
    u_phys = jnp.fft.irfft2(u_hat, s=(grid.Nx, grid.Nx))
    v_phys = jnp.fft.irfft2(v_hat, s=(grid.Nx, grid.Nx))
    u_max = float(jnp.max(jnp.abs(u_phys)))
    v_max = float(jnp.max(jnp.abs(v_phys)))
    vel_max = max(u_max, v_max)
    dx = float(grid.L) / grid.Nx
    return vel_max * float(grid.dt) / dx


def run_test(name, cfg, n_steps, report_interval=5000):
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"  Nx={cfg.Nx}, Nz={cfg.Nz}, L={cfg.L}, Ra={cfg.Ra_tilde}")
    print(f"  dt={cfg.dt}, nu_q={cfg.nu_q}, nu_w={cfg.nu_w}, nu_theta={cfg.nu_theta}")
    print(f"  hyper_order={cfg.hyper_order}, drag={cfg.drag}")
    print(f"  thermal_closure={cfg.thermal_closure}")
    print(f"  n_steps={n_steps}, t_final={n_steps*cfg.dt:.3f}")
    print(f"{'='*70}")

    g = make_grid(cfg)

    # Correct linear growth rate
    Ra = cfg.Ra_tilde
    k_vals = np.linspace(0.35, 30, 500)
    pi2 = np.pi**2
    valid = k_vals**2 > pi2/Ra
    growth = np.full_like(k_vals, -np.inf)
    growth[valid] = -k_vals[valid]**2 + np.sqrt(Ra - pi2/k_vals[valid]**2)
    if np.any(growth > 0):
        max_growth = np.max(growth)
        k_max_growth = k_vals[np.argmax(growth)]
        unstable = k_vals[growth > 0]
        print(f"  Linear growth: max={max_growth:.3f} at k={k_max_growth:.2f}")
        print(f"  Unstable range: k={unstable[0]:.2f} to k={unstable[-1]:.2f}")
        print(f"  Time to reach O(1) from 1e-6: {np.log(1e6)/max_growth:.2f}")
    else:
        print(f"  All modes stable")

    state = make_initial_state(g, seed=42, amplitude=1e-6)

    # JIT compile
    print("  Compiling...", flush=True)
    state = imex_step(state, g)
    jax.block_until_ready(state.q_hat)
    print("  Running...", flush=True)

    for step in range(1, n_steps + 1):
        state = imex_step(state, g)

        if step % report_interval == 0 or step == 1:
            jax.block_until_ready(state.q_hat)
            q_max, w_max, th_max = max_amplitude(state)
            cfl = compute_cfl(state, g)
            t = step * cfg.dt
            print(f"  t={t:.4f}  step={step:6d}  "
                  f"|q|={q_max:.4e}  |w|={w_max:.4e}  |th|={th_max:.4e}  "
                  f"CFL={cfl:.4f}")

            if q_max > 1e10 or np.isnan(q_max):
                print("  >>> BLOWUP!")
                return False

    q_max, w_max, th_max = max_amplitude(state)
    print(f"  Final: |q|={q_max:.4e}  |w|={w_max:.4e}  |th|={th_max:.4e}")
    if q_max < 1e10 and not np.isnan(q_max):
        print("  >>> STABLE")
        return True
    else:
        print("  >>> BLOWUP")
        return False


if __name__ == "__main__":
    results = {}

    # Test A: Molecular diffusion, small dt to avoid CFL issues
    # At saturation |q|~10: CFL = 10*2e-5/0.156 = 0.0013 (safe)
    # Even at |q|~1000: CFL = 0.13 (still safe)
    cfg_a = NHQGConfig(
        Nx=128, Nz=16, L=20.0, Ra_tilde=100.0, sigma=1.0,
        dt=2e-5, float_dtype='float64',
        nu_q=1.0, hyper_order=1,
        nu_w=1.0, nu_theta=1.0,
    )
    results['A_mol_smalldt'] = run_test(
        "Molecular diffusion, dt=2e-5",
        cfg_a, 200000, 20000  # t=4.0
    )

    # Test B: Molecular + q' hyperviscosity, small dt
    k_max_b = np.pi * 128 / 20.0
    nu_q_hyp = 10 * 2e-5 / k_max_b**8
    cfg_b = NHQGConfig(
        Nx=128, Nz=16, L=20.0, Ra_tilde=100.0, sigma=1.0,
        dt=2e-5, float_dtype='float64',
        nu_q=nu_q_hyp, hyper_order=4,
        nu_w=1.0, nu_theta=1.0,
    )
    results['B_mol_hyper'] = run_test(
        f"Molecular + q' hyper (nu_q={nu_q_hyp:.2e})",
        cfg_b, 200000, 20000
    )

    # Test C: q'-only dissipation (testing enstrophy hypothesis)
    cfg_c = NHQGConfig(
        Nx=128, Nz=16, L=20.0, Ra_tilde=100.0, sigma=1.0,
        dt=2e-5, float_dtype='float64',
        nu_q=1.0, hyper_order=1,
        nu_w=0.0, nu_theta=0.0,
    )
    results['C_q_only'] = run_test(
        "q'-only Laplacian",
        cfg_c, 200000, 20000
    )

    # Test D: evolve_mean with molecular diffusion
    cfg_d = NHQGConfig(
        Nx=128, Nz=16, L=20.0, Ra_tilde=100.0, sigma=1.0,
        dt=2e-5, float_dtype='float64',
        nu_q=1.0, hyper_order=1,
        nu_w=1.0, nu_theta=1.0,
        thermal_closure='evolve_mean',
        mean_temp_eps_sq=1.0,
    )
    results['D_evolve_mean'] = run_test(
        "Molecular + evolve_mean",
        cfg_d, 200000, 20000
    )

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, stable in results.items():
        status = "STABLE" if stable else "BLOWUP"
        print(f"  {name:30s}: {status}")
