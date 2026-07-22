# sphere_nhqg/operators.py

## Responsibility

`operators.py` builds the first dense scalar radial operators on top of the
tested Jacobi/Zernike radial basis:

- flat polar radial Laplacian `L_m`,
- spherical Laplacian `mu(r)^-1 L_m`,
- tau-row Helmholtz matrix `(Lap_sphere - Ld^-2)`,
- tau right-hand side and solve helper for `H psi = -q`.

This file is intentionally scalar and per-azimuthal-mode. It does not include
vertical coupling, IMEX block elimination, nonlinear Jacobians, or thermal
exchange.

## Numerical Assumptions

For a mode `m`, the flat radial Laplacian is

```text
L_m = d^2/dr^2 + (1/r) d/dr - m^2/r^2.
```

Applied to the basis function `rho^m p(xi)`, where
`xi = 2 rho^2 - 1`, the singular-looking pole terms cancel exactly:

```text
L_m[rho^m p] =
  8 / r_jet^2 * rho^m * ((m + 1) p'(xi) + (1 + xi) p''(xi)).
```

This cancellation is the main reason the operator is evaluated from the
regular Jacobi form instead of differentiating physical nodal values with
explicit `1/r` and `1/r^2` factors.

The spherical operator is built by evaluating `L_m phi_n` at radial nodes,
multiplying by `mu(r)^-1`, and projecting back to radial coefficients. This is
the dense prototype path described in the formulation notes.

## Data Layout And Normalization

All matrices are coefficient-space matrices with shape `(Nr, Nr)` and act on
coefficient arrays whose radial axis is first:

```text
out[n, ...] = matrix[n, k] coeffs[k, ...]
```

The Helmholtz matrix follows the continuum sign convention

```text
(Lap_sphere - Ld^-2) psi = -q.
```

The tau version replaces the last matrix row with the jet Dirichlet constraint
`sum_n psi_n = 0`, and `helmholtz_rhs` zeros the last right-hand-side entry.

## Invariants And Tests

The operator tests should verify:

- `L_m rho^m = 0` for the harmonic leading basis function,
- `L_m rho^(m+2) = 4(m+1) rho^m / r_jet^2`,
- `Lap_sphere` equals `mu^-1 L_m` pointwise after transforming back to nodes,
- tau Helmholtz solve recovers manufactured `psi` satisfying the jet
  Dirichlet row,
- batched complex right-hand sides solve with radial axis first.

## Known Risks

This is a dense prototype. It is the right first implementation because
residuals and normalization are transparent, but memory will eventually matter
for large `(m, Nr, Nz)` counts. Once the tests are strong, we can decide
whether to keep dense matrices on GPU, cache factorizations, or introduce a
more structured radial transform/operator path.

The spherical Laplacian projection uses the same quadrature grid as the basis.
Because `mu(r)^-1` is polynomial of degree two in `r^2`, this particular
multiplication is benign. Later rational factors such as `mu(r)` in spherical
area means need explicit overintegration audits.
