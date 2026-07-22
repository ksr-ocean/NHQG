"""Systematic stability test at Ra=100 with molecular diffusion.

Tests whether the NHQGE solver reaches nonlinear saturation with:
1. Molecular diffusion (Laplacian) on all three fields (nu_w=1, nu_theta=1)
2. Additional hyperviscosity on q' for enstrophy cascade control

The NHQGE equations (Sprague et al. 2006, Julien et al. 2006) include
molecular diffusion as part of the asymptotic equations. Without it,
linear modes grow without bound.
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
    """Return max |q'|, |w|, |theta| across all modes."""
    q_max = float(jnp.max(jnp.abs(state.q_hat)))
    w_max = float(jnp.max(jnp.abs(state.w_hat)))
    th_max = float(jnp.max(jnp.abs(state.th_hat)))
    return q_max, w_max, th_max


def compute_ke(state, grid):
    """Compute total kinetic energy = sum |k|^2 |psi|^2."""
    psi = invert_psi(state.q_hat, grid.inv_denom)
    return float(jnp.sum(grid.ksq[None, :, :] * jnp.abs(psi)**2))


def run_test(name, cfg, n_steps, report_interval=500):
    """Run a stability test and report amplitude evolution."""
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"  Nx={cfg.Nx}, Nz={cfg.Nz}, L={cfg.L}, Ra={cfg.Ra_tilde}")
    print(f"  dt={cfg.dt}, nu_q={cfg.nu_q}, nu_w={cfg.nu_w}, nu_theta={cfg.nu_theta}")
    print(f"  hyper_order={cfg.hyper_order}, drag={cfg.drag}")
    print(f"  n_steps={n_steps}, t_final={n_steps*cfg.dt:.3f}")
    print(f"{'='*70}")

    g = make_grid(cfg)
    state = make_initial_state(g, seed=42, amplitude=1e-6)

    # Print dissipation info
    ksq_np = np.array(g.ksq)
    k_max = np.sqrt(np.max(ksq_np))
    dk = 2 * np.pi / cfg.L
    print(f"  k_max={k_max:.1f}, dk={dk:.3f}")
    diss_q_max = float(jnp.max(g.diss_rate_q))
    diss_w_max = float(jnp.max(g.diss_rate_w))
    diss_th_max = float(jnp.max(g.diss_rate_th))
    print(f"  diss_rate_q_max={diss_q_max:.2e}")
    print(f"  diss_rate_w_max={diss_w_max:.2e}")
    print(f"  diss_rate_th_max={diss_th_max:.2e}")

    # Linear growth rate analysis
    Ra = cfg.Ra_tilde
    sigma = cfg.sigma
    k_vals = np.linspace(0.1, k_max, 200)
    # n=1 vertical mode: growth rate for w-theta system
    # lambda = sqrt(Ra*k^2/(k^2+pi^2) - pi^2) - nu_w*k^2  (roughly)
    # More precisely: coupled (w,theta) growth rate at each k for n=1 mode
    pi2 = np.pi**2
    growth = np.sqrt(np.maximum(Ra * k_vals**2 / (k_vals**2 + pi2), 0) * pi2 / (k_vals**2 + pi2)) - cfg.nu_w * k_vals**2
    # Actually the dispersion relation for NHQGE (stress-free, n=1) is:
    # sigma_growth^2 + (1+1/sigma)*nu*k^2*sigma_growth + nu^2*k^4/sigma - Ra*k^2/(k^2+pi^2) = 0
    # For sigma=1, nu=nu_w: sigma_g^2 + 2*nu*k^2*sigma_g + nu^2*k^4 - Ra*k^2/(k^2+pi^2) = 0
    # sigma_g = -nu*k^2 + sqrt(Ra*k^2/(k^2+pi^2))
    growth2 = -cfg.nu_w * k_vals**2 + np.sqrt(np.maximum(Ra * k_vals**2 / (k_vals**2 + pi2), 0))
    max_growth = np.max(growth2)
    k_max_growth = k_vals[np.argmax(growth2)]
    unstable = k_vals[growth2 > 0]
    if len(unstable) > 0:
        print(f"  Linear growth: max={max_growth:.3f} at k={k_max_growth:.2f}")
        print(f"  Unstable range: k={unstable[0]:.2f} to k={unstable[-1]:.2f}")
    else:
        print(f"  All modes stable (Ra below onset with dissipation)")

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
            ke = compute_ke(state, g)
            t = step * cfg.dt
            print(f"  t={t:.4f}  step={step:6d}  "
                  f"|q|={q_max:.4e}  |w|={w_max:.4e}  |th|={th_max:.4e}  "
                  f"KE={ke:.4e}")

            if q_max > 1e10 or np.isnan(q_max):
                print("  >>> BLOWUP detected!")
                return False

    jax.block_until_ready(state.q_hat)
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

    # Test 1: Molecular diffusion only (Laplacian on all fields, nu=1)
    # hyper_order=1 so nu_q also acts as Laplacian
    cfg1 = NHQGConfig(
        Nx=128, Nz=16, L=20.0, Ra_tilde=100.0, sigma=1.0,
        dt=5e-4, float_dtype='float64',
        nu_q=1.0, hyper_order=1,  # Laplacian on q'
        nu_w=1.0, nu_theta=1.0,   # Laplacian on w, theta
    )
    results['mol_only'] = run_test("Molecular diffusion (nu=1 Laplacian all fields)", cfg1, 20000, 2000)

    # Test 2: Molecular diffusion + hyperviscosity on q'
    # nu_w=1, nu_theta=1 (Laplacian), nu_q with hyper_order=4
    # Set nu_q so grid-scale damping time ~ 10 dt
    k_max_2 = np.pi * 128 / 20.0
    nu_q_hyp = 10 * 5e-4 / k_max_2**8  # e-folding at k_max ~ 10 dt
    cfg2 = NHQGConfig(
        Nx=128, Nz=16, L=20.0, Ra_tilde=100.0, sigma=1.0,
        dt=5e-4, float_dtype='float64',
        nu_q=nu_q_hyp, hyper_order=4,
        nu_w=1.0, nu_theta=1.0,
    )
    results['mol_plus_hyper'] = run_test(
        f"Molecular + q' hyperviscosity (nu_q={nu_q_hyp:.2e}, p=4)",
        cfg2, 20000, 2000
    )

    # Test 3: Higher resolution with molecular diffusion
    cfg3 = NHQGConfig(
        Nx=256, Nz=32, L=20.0, Ra_tilde=100.0, sigma=1.0,
        dt=2e-4, float_dtype='float64',
        nu_q=1.0, hyper_order=1,
        nu_w=1.0, nu_theta=1.0,
    )
    results['hires_mol'] = run_test("Hi-res molecular diffusion (256x32)", cfg3, 20000, 2000)

    # Test 4: q'-only dissipation (testing enstrophy cascade hypothesis)
    cfg4 = NHQGConfig(
        Nx=128, Nz=16, L=20.0, Ra_tilde=100.0, sigma=1.0,
        dt=5e-4, float_dtype='float64',
        nu_q=1.0, hyper_order=1,
        nu_w=0.0, nu_theta=0.0,
    )
    results['q_only'] = run_test("q'-only Laplacian (nu_q=1, nu_w=0, nu_th=0)", cfg4, 20000, 2000)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, stable in results.items():
        status = "STABLE" if stable else "BLOWUP"
        print(f"  {name:30s}: {status}")
