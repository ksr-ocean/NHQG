"""Quick stability test using jax.lax.scan for speed, CPU only."""

import os
os.environ['JAX_ENABLE_X64'] = '1'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import sys
import time
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.solver import State, make_initial_state, imex_step, invert_psi


def run_scan(cfg, n_outer, n_inner, label=""):
    """Run with jax.lax.scan inner loop for speed."""
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"  Nx={cfg.Nx} Nz={cfg.Nz} L={cfg.L} Ra={cfg.Ra_tilde}")
    print(f"  dt={cfg.dt} nu_q={cfg.nu_q}(p={cfg.hyper_order}) nu_w={cfg.nu_w} nu_th={cfg.nu_theta}")
    print(f"  closure={cfg.thermal_closure}")
    print(f"  total steps={n_outer*n_inner}, t_final={n_outer*n_inner*cfg.dt:.3f}")
    print(f"{'='*60}", flush=True)

    g = make_grid(cfg)
    state = make_initial_state(g, seed=42, amplitude=1e-6)

    @jax.jit
    def scan_body(state, _):
        return imex_step(state, g), None

    # Compile
    t0 = time.time()
    state, _ = jax.lax.scan(scan_body, state, None, length=n_inner)
    jax.block_until_ready(state.q_hat)
    print(f"  Compiled in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    for i in range(1, n_outer + 1):
        state, _ = jax.lax.scan(scan_body, state, None, length=n_inner)
        jax.block_until_ready(state.q_hat)

        step = (i + 1) * n_inner  # +1 for compile step
        t_sim = step * cfg.dt
        q_max = float(jnp.max(jnp.abs(state.q_hat)))
        w_max = float(jnp.max(jnp.abs(state.w_hat)))
        th_max = float(jnp.max(jnp.abs(state.th_hat)))
        wall = time.time() - t0

        print(f"  t={t_sim:.4f} step={step:7d} "
              f"|q|={q_max:.4e} |w|={w_max:.4e} |th|={th_max:.4e} "
              f"wall={wall:.1f}s", flush=True)

        if q_max > 1e10 or np.isnan(q_max):
            print("  >>> BLOWUP!")
            return False
    return True


if __name__ == "__main__":
    # Correct growth rate at Ra=100:
    # s(k) = -k^2 + sqrt(Ra - pi^2/k^2)
    # max = 8.57 at k~0.83, unstable range k~0.35 to k~3.1
    # Time to O(1) from 1e-6: ~1.6 time units

    # Test A: Molecular diffusion, small dt, Nx=64 for CPU speed
    run_scan(
        NHQGConfig(
            Nx=64, Nz=8, L=20.0, Ra_tilde=100.0, sigma=1.0,
            dt=5e-5, float_dtype='float64',
            nu_q=1.0, hyper_order=1,
            nu_w=1.0, nu_theta=1.0,
        ),
        n_outer=80, n_inner=500,  # 40000 steps, t=2.0
        label="A: Molecular (nu=1 all), Nx=64, dt=5e-5"
    )

    # Test B: q'-only dissipation
    run_scan(
        NHQGConfig(
            Nx=64, Nz=8, L=20.0, Ra_tilde=100.0, sigma=1.0,
            dt=5e-5, float_dtype='float64',
            nu_q=1.0, hyper_order=1,
            nu_w=0.0, nu_theta=0.0,
        ),
        n_outer=80, n_inner=500,
        label="B: q'-only (nu_q=1, nu_w=0, nu_th=0), Nx=64"
    )

    # Test C: evolve_mean with molecular
    run_scan(
        NHQGConfig(
            Nx=64, Nz=8, L=20.0, Ra_tilde=100.0, sigma=1.0,
            dt=5e-5, float_dtype='float64',
            nu_q=1.0, hyper_order=1,
            nu_w=1.0, nu_theta=1.0,
            thermal_closure='evolve_mean',
            mean_temp_eps_sq=1.0,
        ),
        n_outer=80, n_inner=500,
        label="C: Molecular + evolve_mean, Nx=64"
    )

    # Test D: Lower Ra (Ra=15, ~1.7x supercritical)
    # Growth rate: s = -k^2 + sqrt(15 - pi^2/k^2)
    # At k=1: s = -1 + sqrt(15-9.87) = -1 + 2.26 = 1.26
    # Time to O(1) from 1e-6: ~11 time units
    run_scan(
        NHQGConfig(
            Nx=64, Nz=8, L=20.0, Ra_tilde=15.0, sigma=1.0,
            dt=1e-4, float_dtype='float64',
            nu_q=1.0, hyper_order=1,
            nu_w=1.0, nu_theta=1.0,
        ),
        n_outer=200, n_inner=500,  # 100000 steps, t=10.0
        label="D: Molecular at Ra=15, Nx=64, dt=1e-4"
    )

    # Test E: Ra=100 with strong molecular diffusion (nu=5)
    # Growth rate: s = -5*k^2 + sqrt(100-pi^2/k^2)
    # At k=1: s = -5 + 9.49 = 4.49  (still unstable)
    # At k=2: s = -20 + sqrt(100-2.47) = -20 + 9.87 = -10.1 (stable!)
    # Unstable range: k~0.35 to k~1.6
    # Max growth: ~4.7 at k~0.7
    run_scan(
        NHQGConfig(
            Nx=64, Nz=8, L=20.0, Ra_tilde=100.0, sigma=1.0,
            dt=5e-5, float_dtype='float64',
            nu_q=5.0, hyper_order=1,
            nu_w=5.0, nu_theta=5.0,
        ),
        n_outer=80, n_inner=500,
        label="E: Strong diffusion (nu=5), Ra=100, Nx=64"
    )
