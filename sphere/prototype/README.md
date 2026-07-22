# Spherical Prototype Scaffold

This directory is for the small polar-cap prototype described in
`TRIAGE.md`. It is intentionally not a full NHQGE solver yet.

## Purpose

Answer the implementation question before committing to a full rewrite:

- Can the stereographic disk geometry be implemented cleanly?
- Can the spherical area measure be audited numerically?
- Does the small-cap limit reduce to the Cartesian operators?
- What radial basis / CPU parallelization path should be used?

## Current Status

`geometry_checks.py` is the first executable check. It verifies:

- stereographic cap radius from jet latitude,
- exact spherical cap area,
- numerical quadrature of the spherical area measure,
- small-cap convergence of spherical area to flat disk area.

This is deliberately language-neutral numerics. Julia is not currently
available on this machine, so the first scaffold is Python/NumPy. The checks
can be ported directly to Julia once the Julia environment exists.

## Prototype Milestones

1. Geometry and area-measure checks.
2. Scalar functions on a Fourier-azimuth / radial grid.
3. Radial basis comparison:
   - Jacobi/Zernike-style regular disk basis,
   - Chebyshev-on-r2 fallback,
   - Dedalus DiskBasis reference if available.
4. Spherical Laplace-Beltrami application:
   `Lap_sphere psi = mu(r)^(-1) Lap_flat psi`.
5. Per-azimuthal-mode Helmholtz/PV inversion.
6. Spherical-area horizontal means and thermal-exchange audit.
7. Small-cap comparison against the Cartesian solver.

Only after those are passing should the full ARS IMEX step and
`balanced_sbp2_pc` thermal corrector be ported.

## Commands

```bash
python sphere/prototype/geometry_checks.py
```
