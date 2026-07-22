# sphere_nhqg/mean_exchange.py

## Responsibility

`mean_exchange.py` owns the spherical horizontal mean convention. The full SBP
thermal corrector will be added here later, but this first version only
provides the weighted cap-mean primitive needed by both the corrector and
diagnostics.

## Numerical Assumptions

For an azimuthally averaged field `F_0(r, ...)`, the spherical area mean is

```text
<F>_S = (2 pi / A_cap) int_0^rjet F_0(r) mu(r) r dr.
```

The function requires an `m=0` radial basis because the input is already the
azimuthal mean. Non-axisymmetric modes do not contribute directly to the
horizontal mean after the azimuthal average; nonlinear products must be
transformed/evaluated before this primitive is called.

## Data Layout And Normalization

Input values use radial node axis first:

```text
values[j, ...] = F_0(r_j, ...)
```

The returned shape is the trailing shape `...`. The mean weights are normalized
so that a constant field has mean one:

```text
sum_j weights[j] ~= 1.
```

The quadrature is Gauss-Legendre in `xi` for the `m=0` family. Because
`mu(r)` is rational in `r`, the integral is spectrally accurate but not
algebraically exact at finite `Nr`.

## Invariants And Tests

The tests check:

- mean weights sum to one at useful resolution,
- a constant field has mean one,
- batched profiles preserve trailing dimensions,
- the spherical mean of the Coriolis profile matches the analytic cap average.

These tests enforce the convention that later thermal exchange and diagnostics
must share.

## Known Risks

This primitive assumes the radial data are already on an adequate `m=0`
quadrature grid. For products such as `w theta`, the production path should
evaluate the product on an overresolved radial grid before applying these
weights. Reusing underresolved product values here would recreate the same
kind of raw/dealiased diagnostic split that caused confusion in the Cartesian
solver.
