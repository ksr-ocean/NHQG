#!/usr/bin/env python
"""Quick validation: Ra=10 near onset, compare Nu with Miquel et al. (1.50 ± 0.01).

Uses Nx=128, Nz=64, L=48 (~10 L_c) to match Miquel's domain.
Should converge quickly since flow is barely supercritical (Ra/Ra_c ≈ 1.15).
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.paths import normalize_output_dir
from nhqg.solver import make_initial_state, imex_step, State
from nhqg.diagnostics import compute_diagnostics

# --- Configuration ---
Nx, Nz = 128, 64
L = 48.0          # ~10 L_c (Miquel's domain)
dt = 5e-3         # can be larger since flow is slow near onset
nu = 1.0          # Laplacian, matching Miquel

save_every = 100
n_total = 100000  # should be enough for saturation at Ra=10

print("=" * 60)
print("NHQGE Onset Validation: Ra=10, Miquel target Nu=1.50")
print("=" * 60)
print(f"Resolution: Nx={Nx}, Nz={Nz}, L={L}")
print(f"Dissipation: Laplacian nu={nu}")
print(f"dt={dt}, save_every={save_every}")
print(f"Device: {jax.devices()}")
print()

output_dir = normalize_output_dir('output_onset')

cfg = NHQGConfig(
    Nx=Nx, Nz=Nz, L=L,
    Ra_tilde=10.0, sigma=1.0,
    beta=0.0, Ld=float('inf'),
    dt=dt, t_final=1000.0,
    nu_q=nu, hyper_order=1,
    nu_w=nu, nu_theta=nu,
    save_interval=save_every,
    output_dir=output_dir,
    float_dtype='float64',
)
grid = make_grid(cfg)

@jax.jit
def scan_body(state, _):
    return imex_step(state, grid), None

# Initial state
state = make_initial_state(grid, seed=0, amplitude=1e-3)

# JIT warmup
print("JIT compiling...", flush=True)
t0 = time.time()
state_warmup, _ = jax.lax.scan(scan_body, state, None, length=1)
jax.block_until_ready(state_warmup.q_hat)
print(f"JIT: {time.time()-t0:.1f}s\n")

# Run
print(f"{'step':>8s} {'t':>8s} {'wall':>8s} {'ms/step':>8s} "
      f"{'KE_bc':>12s} {'Nusselt':>10s} {'max_v':>10s}")
print("-" * 80)

t0_run = time.time()
Nu_history = []

for chunk in range(n_total // save_every):
    t0_chunk = time.time()
    state, _ = jax.lax.scan(scan_body, state, None, length=save_every)
    jax.block_until_ready(state.q_hat)
    chunk_time = time.time() - t0_chunk

    step = (chunk + 1) * save_every
    t_sim = step * dt
    wall = time.time() - t0_run
    ms = 1000 * chunk_time / save_every

    diag = compute_diagnostics(state, grid)
    Nu = float(diag['Nusselt'])
    Nu_history.append(Nu)

    print(f"{step:8d} {t_sim:8.2f} {wall:8.1f} {ms:8.2f} "
          f"{float(diag['KE_bc']):12.4e} {Nu:10.4f} {float(diag['max_speed']):10.4f}",
          flush=True)

    # Check for NaN
    if np.isnan(Nu):
        print("NaN detected!")
        break

    # Check convergence: if last 10 Nu values vary < 1%
    if len(Nu_history) >= 20:
        recent = Nu_history[-10:]
        mean_nu = np.mean(recent)
        spread = (np.max(recent) - np.min(recent)) / max(mean_nu, 1e-10)
        if spread < 0.01 and t_sim > 10.0:
            print(f"\nConverged! Nu = {mean_nu:.4f} ± {np.std(recent):.4f}")
            print(f"Miquel target: 1.50 ± 0.01")
            print(f"Ratio: {mean_nu/1.50:.3f}")
            break

print(f"\nTotal: {time.time()-t0_run:.1f}s")
