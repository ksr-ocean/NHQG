"""Long stability test: molecular diffusion at Ra=100, run to t=4+.
Focus on whether nonlinear saturation occurs after the linear growth phase.

Growth timeline (from previous runs):
- t=0 to t~1.6: linear growth from 1e-6 to ~0.05
- t~2.0: |q|~1.4, still exponential (Jacobians ~17% of linear growth)
- t~2.23 (predicted): |q|~10, Jacobians competitive with linear growth
- t~2.3+: nonlinear cascade should establish... does it saturate?
"""

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


cfg = NHQGConfig(
    Nx=64, Nz=8, L=20.0, Ra_tilde=100.0, sigma=1.0,
    dt=5e-5, float_dtype='float64',
    nu_q=1.0, hyper_order=1,
    nu_w=1.0, nu_theta=1.0,
)

g = make_grid(cfg)
state = make_initial_state(g, seed=42, amplitude=1e-6)

@jax.jit
def scan_body(state, _):
    return imex_step(state, g), None

n_inner = 500  # report every 500 steps = 0.025 time units
n_outer = 160  # total 80000 steps = t=4.0

# Compile
t0 = time.time()
state, _ = jax.lax.scan(scan_body, state, None, length=n_inner)
jax.block_until_ready(state.q_hat)
print(f"Compiled in {time.time()-t0:.1f}s")
print(f"Ra=100, nu=1 (Laplacian all), Nx=64, Nz=8, dt=5e-5")
print(f"Linear max growth rate: 8.57 at k~0.83")
print(f"Running to t={n_outer*n_inner*cfg.dt:.1f}...")

t0 = time.time()
for i in range(1, n_outer + 1):
    state, _ = jax.lax.scan(scan_body, state, None, length=n_inner)
    jax.block_until_ready(state.q_hat)

    step = (i + 1) * n_inner
    t_sim = step * cfg.dt
    q_max = float(jnp.max(jnp.abs(state.q_hat)))
    w_max = float(jnp.max(jnp.abs(state.w_hat)))
    th_max = float(jnp.max(jnp.abs(state.th_hat)))

    # Estimate CFL
    psi = invert_psi(state.q_hat, g.inv_denom)
    vel = float(jnp.max(jnp.abs(g.kx[None,:,:] * psi)) +
                jnp.max(jnp.abs(g.ky[None,:,:] * psi)))
    cfl = vel * float(g.dt) / (float(g.L) / g.Nx)

    wall = time.time() - t0

    print(f"t={t_sim:.4f} |q|={q_max:.4e} |w|={w_max:.4e} |th|={th_max:.4e} "
          f"CFL={cfl:.4f} wall={wall:.0f}s", flush=True)

    if q_max > 1e10 or np.isnan(q_max):
        print("BLOWUP!")
        break
