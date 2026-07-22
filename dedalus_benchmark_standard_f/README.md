Dedalus Benchmark For Upright Standard-f NHQG

This folder is an isolated benchmark branch for the upright `beta=0`,
`Ld=inf` NHQG case. It is intended to answer a different question from the
Coral branch:

- Coral benchmark: are we matching Miquel's production formulation?
- Dedalus benchmark: does an independent spectral PDE framework show the same
  runaway?

Status:
- Dedalus is not installed on this node, so nothing in this folder was run here.
- The Python files were syntax-checked only.
- The benchmark script is therefore a concrete starting point, not a validated
  result.

Why this formulation:
- We want to avoid reproducing the Chebyshev/Galerkin/tau machinery from the
  JAX code.
- Dedalus v3 handles mixed Fourier/Chebyshev systems well, but our fluctuation
  fields `w` and `theta` have boundary conditions without vertical diffusion.
- To avoid building a fragile tau formulation for those fields, this scaffold
  encodes the Dirichlet conditions directly with the prefactor
  `S(z) = z (1-z)`:
  - `w = S * wt`
  - `theta = S * tt`
  - `Theta_bar' = S * mt`
- The fluctuating horizontal-mean mode is set to zero with mode-conditioned
  equations, which matches the JAX solver's `zero_mode` projection.

Folder contents:
- `run_nhqg_dedalus.py`: draft Dedalus v3 benchmark script
- `extract_history.py`: convert Dedalus HDF5 analysis output into a simple `.npz`
  archive for comparison with the JAX runs
- `FORMULATION.md`: detailed mapping from the JAX equations to the Dedalus
  representation
- `AGENTS.md`: orientation file for a Codex session started in this folder

Recommended first use:
1. Install Dedalus in a separate environment.
2. Run a tiny smoke case, for example:

```bash
python run_nhqg_dedalus.py \
  --Nx 32 --Nz 32 --dt 5e-5 --t-final 0.05 \
  --thermal-closure evolve_mean \
  --output-dir output_smoke
```

3. Inspect whether Dedalus accepts the mode-conditioned `q=0`, `psi=0`,
   `wt=0`, `tt=0` zero-mode constraints.
4. If the smoke case is clean, move to the matched benchmark:

```bash
python run_nhqg_dedalus.py \
  --Nx 128 --Nz 128 --dt 5e-5 --t-final 5.0 \
  --thermal-closure evolve_mean \
  --output-dir output_miquel_zero_tilt_dedalus_evolvemean_Nx128_Nz128_dt5e5_t5
```

Notes:
- The time stepper in the scaffold is `d3.RK222`, because that is a standard
  Dedalus IMEX option documented in the official tutorial. It is not an exact
  clone of our JAX ARS222.
- The direct comparison target is therefore qualitative first: does the run
  saturate, delay-runaway, or blow up in essentially the same way?

Official references used for the scaffold:
- Dedalus tutorial on IVPs and timesteppers:
  https://dedalus-project.readthedocs.io/en/latest/notebooks/dedalus_tutorial_3.html
- Dedalus v3 tau-method guidance:
  https://dedalus-project.readthedocs.io/en/latest/pages/tau_method.html
