# sphere_nhqg/tests/test_geometry.py

## Responsibility

The geometry tests pin down the exact stereographic formulas before any radial
basis, Laplacian, Jacobian, or time-stepper code is built on top of them.

## Numerical Assumptions

The tests use unit-sphere formulas. Radial quadrature uses NumPy
Gauss-Legendre nodes only as an independent audit of the analytic area formula;
it is not the production radial quadrature rule.

## Data Layout And Normalization

The test converts JAX arrays to NumPy arrays through `np.asarray` for scalar
comparisons. This avoids depending on a specific JAX device and keeps the
checks valid on CPU-only development machines.

## Invariants And Tests

The tests deliberately compare the same quantity through multiple formulas:

- cap radius from latitude against known trigonometric values,
- cap area from projected radius against the latitude area fraction,
- cap area from formula against numerical integration of `mu(r) r`,
- Coriolis profile against the physical identity `f = 2 Omega sin(phi)`.

## Known Risks

Finite-difference checks of `df/dr` are performed in float64 NumPy space, not
through JAX autodiff. That is intentional for now: the test is meant to catch
formula regressions, not validate autodiff behavior.
