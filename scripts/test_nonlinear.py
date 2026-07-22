"""Quick nonlinear stability test at Ra=10 (Miquel target: Nu=1.27).

Run on CPU to check if the BC-consistent IMEX solve fixes the instability.
"""
import os
os.environ['JAX_ENABLE_X64'] = '1'
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, '.')

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.solver import make_initial_state, imex_step
from nhqg.diagnostics import compute_diagnostics

# Miquel Ra=10 target: Nu = 1.27 ± 0.01
# Use Nx=64, Nz=32, L=10*Lc where Lc = 2*pi/kc ≈ 4.81
Ra = 10.0
Lc = 2 * 3.14159265 / 1.3048
L = 10.0 * Lc
Nx = 64
Nz = 32
dt = 5e-4

# Laplacian (p=1) dissipation on all fields, nu=1 (Miquel setup)
cfg = NHQGConfig(
    Nx=Nx, Nz=Nz, L=L,
    Ra_tilde=Ra, sigma=1.0,
    beta=0.0, Ld=float('inf'),
    dt=dt,
    nu_q=1.0, nu_w=1.0, nu_theta=1.0,
    hyper_order=1,
    float_dtype='float64',
    thermal_closure='fixed_conduction',
)

print(f"Ra={Ra}, Nx={Nx}, Nz={Nz}, L={L:.2f}, dt={dt}")
print(f"Laplacian diss nu=1 on all fields")
print(f"Target: Nu ≈ 1.27 (Miquel Table 1)")
print(f"Device: {jax.devices()}")
print()

grid = make_grid(cfg)
state = make_initial_state(grid, seed=0, amplitude=1e-3)

n_steps = 40000  # t_final = 20
print_every = 2000

for step in range(1, n_steps + 1):
    state = imex_step(state, grid)
    
    if step % print_every == 0:
        diag = compute_diagnostics(state, grid)
        Nu = float(diag['Nusselt'])
        KE = float(diag['KE_bt']) + float(diag['KE_bc'])
        max_v = float(diag['max_speed'])
        t = step * dt
        print(f"  step={step:6d}  t={t:7.3f}  Nu={Nu:8.4f}  KE={KE:10.4e}  max_v={max_v:8.4f}")
        
        # Check for blowup
        if not jnp.isfinite(KE) or KE > 1e6:
            print("  *** BLOWUP DETECTED ***")
            break

print("\nDone.")
