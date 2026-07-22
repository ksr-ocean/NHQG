Trig Vertical Benchmark Solver

This folder contains a separate NHQG benchmark branch that keeps the horizontal
Fourier pseudospectral machinery but replaces the vertical finite differences by
a trigonometric modal representation:
- `psi` uses cosine modes so Neumann BCs are exact
- `w`, `theta`, and `Theta_b'` use sine modes so Dirichlet BCs are exact

The linear vertical operators are diagonal or basis-swap maps in modal space.
Nonlinear products are evaluated on an oversampled uniform vertical work grid
and projected back to sine/cosine coefficients.

Run:

```bash
JAX_ENABLE_X64=1 PYTHONPATH=. python trig_vertical_benchmark/run_miquel_trig.py \
  --Nx 128 --Nz 128 --dt 5e-5 --t-final 5.0 \
  --thermal-closure evolve_mean
```
