# sphere_nhqg/tests/test_operators.py

## Responsibility

The operator tests validate the scalar coefficient-space radial operators
before they are used in PV inversion, IMEX block solves, or nonlinear
Jacobians.

## Numerical Assumptions

The tests use analytic polar-Laplacian identities:

```text
L_m r^m = 0,
L_m r^(m+2) = 4(m+1) r^m.
```

The implemented basis uses `rho = r / r_jet`, so the second identity carries a
factor `r_jet^-2`.

## Data Layout And Normalization

Matrices act on radial coefficients with radial axis first. The tests convert
back to physical radial nodes before comparing pointwise identities because
that is how the later nonlinear and diagnostic paths will audit operators.

The Helmholtz tests follow the sign convention:

```text
(Lap_sphere - Ld^-2) psi = -q.
```

The manufactured `psi` coefficients are adjusted so that `sum_n psi_n = 0`,
which satisfies the jet Dirichlet tau row.

## Invariants And Tests

The tests cover:

- flat Laplacian annihilation of the harmonic leading basis function,
- flat Laplacian normalization on the next radial power,
- spherical Laplacian pointwise equivalence to `mu^-1 L_m`,
- tau Helmholtz recovery of manufactured real solutions,
- tau Helmholtz recovery of batched complex solutions.

## Known Risks

The current tests validate collocation/projection consistency at modest `Nr`.
They do not yet measure conditioning of the Helmholtz matrix as `m` and `Nr`
grow. That conditioning audit should be added before production-resolution
precompute work begins.
