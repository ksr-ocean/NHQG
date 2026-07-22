# sphere_nhqg/tests/test_radial.py

## Responsibility

The radial tests validate the Jacobi/Zernike basis independently of the
spherical operator layer. They are intended to catch normalization, projection,
regularity, derivative, and jet-tau mistakes before those mistakes are hidden
inside Helmholtz inversions.

## Numerical Assumptions

The tests use the production-intended Option A basis:

```text
rho^m P_n^(0,m)(2 rho^2 - 1), rho = r / r_jet.
```

They test nonzero `m` values because the pole regularity and projection
normalization are trivial for `m = 0` but easy to get wrong once the
`rho^m` factor is present.

## Data Layout And Normalization

The tests assume radial coefficients use axis 0 and nodal radial values use
axis 0 after transformation. Batched complex data are included in the roundtrip
test because the solver state will be complex after azimuthal FFTs.

The flat inner-product test compares the analytic Jacobi norm scaling against
an independent high-order Legendre quadrature in physical `r`, not against the
same Gauss-Jacobi weights used to build the basis.

## Invariants And Tests

The tests cover:

- Gauss-Jacobi orthogonality for `P_n^(0,m)`,
- exact transform roundtrip for complex batched coefficients,
- pole values for `m > 0` vanish by construction,
- radial derivatives match finite differences of basis functions,
- derivative evaluation from nodal values agrees with coefficient evaluation,
- jet Dirichlet and Neumann tau rows match analytic formulas,
- flat-disk radial inner products match physical quadrature.

## Known Risks

Finite-difference derivative checks use a small absolute step at interior
Gauss-Jacobi nodes. This is a formula regression test, not a conditioning
study. When production resolutions are chosen, we should add higher-`Nr` and
higher-`m` conditioning audits.
