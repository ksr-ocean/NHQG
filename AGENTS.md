# Agent Instructions

**Read `CLAUDE.md` — it is the single source of truth for this repository.**

This file previously carried a full copy of the project documentation. That
copy was a frozen 2026-03-12 snapshot describing the *pre-Galerkin* solver
(dense `D_Z` collocation matrices, Neumann BC on `q'`, 33 tests, three
prognostic fields) and had become actively misleading. It was replaced with
this pointer on 2026-07-04 so the two files cannot drift apart again.

## Quick orientation (verified 2026-07-04)

- **What this is**: a JAX GPU pseudospectral solver for the nonhydrostatic
  quasi-geostrophic equations (rotating Rayleigh-Benard, Ek -> 0), targeting
  Jupiter polar vortex dynamics. Fourier horizontal x Chebyshev
  Galerkin vertical, ARS(2,2,2) IMEX.
- **Start here**: `CLAUDE.md` (framework + "Current Status (2026-07-04)"),
  then `hermitian_ghost.md` (the ghost-mode diagnosis — read before trusting
  any raw/spectral diagnostic), then `NHQG_framework_deck.pdf` (pedagogical
  walkthrough of everything).
- **Tests**: `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest tests/ -q`
  (89 tests, ~70 s). Always set `JAX_ENABLE_X64=1`; production physics
  requires float64 (the `NHQGConfig` dataclass default is float32 — scripts
  override it).
- **GPU runs**: keep BLAS threads at 1 (`OPENBLAS_NUM_THREADS=1` etc.);
  the startup stall is host-side dense linear algebra in `make_grid()`.
- **LaTeX**: build with `tectonic` (pdflatex is no longer on PATH).
- **Run outputs**: everything lives under `output/` (~310 GB — never glob it
  casually, never commit it).
- **Production configuration** is set by script flags, not config defaults:
  `mean_exchange_discretization='balanced_sbp2_pc'`, `sbp_corrector_substeps=4`,
  `nonlinear_advection='flux'`, `horizontal_dealiasing='23_rule'`,
  `float_dtype='float64'`, `thermal_closure='evolve_mean'`, Ra=100, dt=5e-5.
- **Historical notes** (`blowup.md`, `adjoint_mean_exchange.md`,
  `spectral_analysis.md`) carry banners describing what superseded them —
  trust the banners over the body text.
