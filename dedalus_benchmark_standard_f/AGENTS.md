## Objective

Use Dedalus as an independent benchmark for the upright standard-`f` NHQG
runaway, separate from both the JAX Chebyshev solver and the Coral Fortran
port.

## Current Status

- This folder is a scaffold only.
- Dedalus is not installed on the node where this folder was created.
- `run_nhqg_dedalus.py` has been syntax-checked but not executed.

## Files

- `run_nhqg_dedalus.py`
  Main draft benchmark script. It uses:
  - Fourier in `x,y`
  - Chebyshev in `z`
  - prefactored Dirichlet fields `w=S*wt`, `theta=S*tt`, `Theta_bar'=S*mt`
  - conditioned zero-mode constraints for the fluctuation fields

- `extract_history.py`
  Postprocess Dedalus HDF5 outputs into a compact `.npz` file for comparison
  with the JAX `spectrum_history.npz` archives.

- `FORMULATION.md`
  Detailed mapping between the JAX equations and the Dedalus benchmark.

## Immediate Tasks

1. Install or activate a Dedalus v3 environment.
2. Run a smoke case at `Nx=32`, `Nz=32`, `t_final=0.05`.
3. Confirm:
   - the conditioned zero-mode equations parse
   - the prefactored fields behave cleanly at `z=0,1`
   - the HDF5 analysis files are produced
4. If that passes, run the matched `Nx=128`, `Nz=128`, `dt=5e-5`,
   `t_final=5.0` `evolve_mean` case.
5. Convert the Dedalus output with `extract_history.py` and compare against:
   - JAX Chebyshev benchmark
   - FD vertical benchmark
   - Coral run if available

## Comparison Targets

The JAX matched benchmark to compare against is:

- `/ocean/projects/oce110003p/kaushiks/jove/NHQG/output_miquel_zero_tilt_galerkin_ars222_evolvemean_blas1_Nx128_Nz128_dt5e5_t10/spectra/spectrum_history.npz`

The FD vertical benchmark is:

- `/ocean/projects/oce110003p/kaushiks/jove/NHQG/output_miquel_zero_tilt_fd_ars222_evolvemean_blas1_Nx128_Nz128_dt5e5_t5/spectra/spectrum_history.npz`

## Important Caveat

This Dedalus scaffold is not yet evidence either for or against the runaway.
Treat it as a structured starting point until a real Dedalus run succeeds.
