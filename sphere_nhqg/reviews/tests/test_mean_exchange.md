# sphere_nhqg/tests/test_mean_exchange.py

## Responsibility

The mean-exchange tests validate the spherical-area horizontal mean primitive
before it is threaded into the SBP thermal corrector or diagnostics.

## Numerical Assumptions

The tests assume the input is already the azimuthal mean `F_0(r, ...)` on an
`m=0` radial quadrature grid. Nonzero azimuthal modes are rejected by the
primitive because they do not directly define a horizontal mean.

## Data Layout And Normalization

Radial node axis is first. The tests include trailing dimensions to ensure the
mean operation can later consume vertical profiles or batches without changing
their trailing layout.

The normalization check requires constants to average to one. This is the
minimum invariant needed before the primitive can be trusted in heat-budget
diagnostics.

## Invariants And Tests

The tests cover:

- positive normalized spherical mean weights,
- constant field mean,
- trailing-dimension preservation,
- analytic spherical cap average of the Coriolis profile,
- rejection of nonzero-`m` bases.

## Known Risks

These tests do not yet cover dealiased nonlinear products. The future
`w theta` path must compute products on an overresolved radial/azimuthal grid
before applying these weights.
