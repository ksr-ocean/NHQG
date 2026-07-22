FD-in-Z Benchmark Solver

This folder contains a deliberately separate NHQG benchmark solver that keeps
the horizontal Fourier pseudospectral discretization but replaces the vertical
Chebyshev/Galerkin machinery with a uniform-grid finite-difference
discretization.

Purpose:
- provide an independent benchmark against the production `nhqg/` solver
- remove Chebyshev/tau/Galerkin details from the vertical discretization
- keep the same upright standard-`f` case and the same dealiased horizontal
  nonlinear terms

State layout:
- `psi_hat`: interior vertical nodes only, horizontal Fourier coefficients
- `w_hat`: interior vertical nodes only, horizontal Fourier coefficients
- `th_hat`: interior vertical nodes only, horizontal Fourier coefficients
- `th_bar`: interior vertical nodes only, mean-temperature deviation

Vertical boundary conditions:
- `dpsi/dz = 0` at `z=0,1`, enforced by eliminating boundary values through a
  Neumann reconstruction; the compact branch can either use the legacy
  `projected` reconstruction from `D1_full` or a `direct` reduced solve built
  straight from the compact relation `A d = B f`
- `w = 0`, `theta = 0`, `Theta_bar' = 0` at `z=0,1`, enforced by zero boundary
  extension of the interior arrays

Time stepping:
- ARS(2,2,2) IMEX only
- horizontal dissipation is implicit, matching the production solver split
- the mean-temperature diffusion solve is a dense interior Dirichlet solve
- vertical first derivatives can use either the legacy `centered2` stencil or a
  fourth-order tridiagonal compact scheme `compact4`, or a diagonal-norm SBP
  `D1(4,2)` operator `sbp42`
- the `psi` Neumann treatment can use either the legacy `projected` path or the
  reduced-system `direct` path
- vertical second derivatives can use the legacy `centered2` Dirichlet
  Laplacian, a raw compact branch `compact4_raw`, the generic
  energy-compatible branch `from_d1_energy`, or the SBP-induced
  energy-compatible branch `sbp42_energy`

Current priority:
- treat `D1` as the important operator family, because it controls the
  fluctuation coupling through `B = D1_psi D1_dir`
- treat `D2` as a secondary robustness choice, because it only appears in the
  implicit mean-temperature diffusion solve
- the main stability comparison is therefore `compact4 + centered2` versus
  `sbp42 + centered2`
- `sbp42_energy` is kept as a diagnostic branch, not the main path

Scope:
- upright `beta=0`, `Ld=inf` benchmark case
- `fixed_conduction` and `evolve_mean` thermal closures
- `jacobian` or conservative `flux` horizontal advection

Useful scripts:
- `run_miquel_fd.py`: launch matched FD benchmark runs
- `rebuild_history_from_checkpoints.py`: rebuild `spectrum_history.npz` from checkpoints
- `audit_operators.py`: inspect spectra, nonnormality, boundary amplification,
  and shell conditioning for selected `(D1, D2, psi-Neumann)` combinations

Run:

```bash
JAX_ENABLE_X64=1 PYTHONPATH=. python fd_vertical_benchmark/run_miquel_fd.py \
  --Nx 128 --Nz 128 --dt 5e-5 --t-final 5.0 \
  --thermal-closure evolve_mean \
  --vertical-derivative sbp42 \
  --vertical-second-derivative centered2 \
  --psi-neumann-treatment direct
```

Operator audit:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=. \
python fd_vertical_benchmark/audit_operators.py --Nz 128
```

Outputs:
- `checkpoint_XXXXXXXX.npz`
- `spectra/spectrum_history.npz`

The diagnostics archive is intentionally close to the production run format so
the scalar blowup history and horizontal shell spectra can be compared
directly.
