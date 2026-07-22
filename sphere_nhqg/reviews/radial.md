# sphere_nhqg/radial.py

## Responsibility

`radial.py` builds the first production-intended radial basis for the
stereographic disk: a per-azimuthal-mode Jacobi/Zernike basis with exact pole
regularity. It provides quadrature nodes and weights, dense coefficient to
physical transforms, dense physical to coefficient projections, radial
derivative evaluation at quadrature nodes, jet boundary tau rows, and flat
radial inner products.

It does not yet build the spherical Laplacian or Helmholtz/PV inversion. Those
belong in the next operator layer once this basis is stable.

## Numerical Assumptions

For azimuthal mode `m >= 0`, a smooth scalar radial profile is represented as

```text
F_m(r) = (r / r_jet)^m sum_n c_n P_n^(0,m)(xi),
xi = 2 (r / r_jet)^2 - 1.
```

The leading `(r / r_jet)^m` factor enforces the disk regularity condition
`F_m(r) ~ r^m` at the pole. This is the main lesson imported from the
formulation: regularity should be part of the basis, not an after-the-fact tau
patch at `r=0`.

Gauss-Jacobi quadrature uses weight `(1 + xi)^m`. Orthogonality is for the
polynomial part `P_n^(0,m)`, not for raw physical nodal values unless the
leading `rho^m` factor is removed.

## Data Layout And Normalization

`RadialBasis` stores arrays with node index first:

```text
poly_vander[j, n]      = P_n^(0,m)(xi_j)
physical_vander[j, n]  = rho_j^m P_n^(0,m)(xi_j)
```

The transform helpers therefore use:

```text
values[j, ...] = physical_vander[j, n] coeffs[n, ...]
coeffs[n, ...] = coeff_from_physical_nodal[n, j] values[j, ...]
```

The Jacobi norm is

```text
int_-1^1 (1 + xi)^m [P_n^(0,m)]^2 dxi = 2^(m+1) / (2n + m + 1).
```

The corresponding flat-disk radial basis norm is scaled by

```text
r_jet^2 / (4 * 2^m).
```

That scaling comes from `r dr = r_jet^2 dxi / 4` and
`rho^(2m) = 2^-m (1 + xi)^m`.

## Invariants And Tests

The radial tests should keep checking:

- Gauss-Jacobi orthogonality and exact norms,
- coefficient -> physical -> coefficient roundtrip for batched complex data,
- pole values vanish for `m > 0`,
- derivative evaluation against finite differences,
- jet Dirichlet tau row is all ones,
- jet Neumann tau row matches the analytic derivative formula,
- flat inner products agree with high-order quadrature.

These are the low-level mistakes that would otherwise contaminate the
Laplacian and Helmholtz layers.

## Known Risks

This first implementation uses SciPy at setup time for roots and Jacobi
polynomial evaluation, then stores dense JAX arrays. That is acceptable for
prototype and single-GPU runs because the expensive pieces are precomputed.
If setup time or memory becomes a bottleneck, the future optimized path can
replace dense matrices with recurrences or structured transforms.

The projection from physical nodal values divides out `rho^m`. Gauss-Jacobi
nodes do not include the pole, so this is numerically safe, but high `m` and
large `Nr` can amplify roundoff near the first node. The tests should be
expanded as target production `m` and `Nr` increase.
