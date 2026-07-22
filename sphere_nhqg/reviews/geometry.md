# sphere_nhqg/geometry.py

## Responsibility

`geometry.py` owns the exact stereographic geometry for the unit-sphere polar
cap. It provides the cap radius, inverse map, conformal metric factor, area
density, exact cap area, Coriolis profile, radial Coriolis derivative, and
arc-length conversion factors.

## Numerical Assumptions

The module assumes the sphere radius is one. This matches the current
formulation notes and avoids hiding metric normalization mistakes behind an
extra dimensional scale. If a dimensional radius is restored later, metric
factors need powers of that radius applied consistently:

- lengths scale like `a`,
- areas scale like `a^2`,
- Laplacians scale like `a^-2`.

The stereographic map is from the south pole to the plane tangent to the north
pole:

```text
r = tan(theta / 2)
theta = 2 atan(r)
mu(r) = 4 / (1 + r^2)^2
dA = mu(r) r dr dphi
```

For a northern cap bounded by latitude `phi_jet`, the projected cap radius is
`tan(pi/4 - phi_jet/2)`.

## Data Layout And Normalization

All functions accept scalars or arrays and return JAX arrays. This keeps the
geometry layer compatible with later `jit`-compiled operator code. No FFT or
radial-transform normalization appears here; this file only defines physical
metric factors.

The exact cap area is

```text
A = 4 pi r_jet^2 / (1 + r_jet^2)
```

and the equivalent latitude fraction is

```text
A / (4 pi) = (1 - sin(phi_jet)) / 2.
```

## Invariants And Tests

The geometry tests check:

- known cap radii for 45 and 30 degree jet latitudes,
- exact cap area from `r_jet` against the latitude formula,
- high-order radial quadrature of `mu(r) r`,
- small-cap convergence of spherical area to `4 pi r^2`,
- Coriolis values at the pole, equator, and cap edge,
- analytic `df/dr` against finite differences.

These tests should remain cheap and should be run before touching any radial
basis or solver code.

## Known Risks

The small-cap projected disk area differs from spherical area by a factor of
four because this stereographic coordinate uses `r = tan(theta/2)`. That is
expected, but it is an easy source of normalization errors when comparing to
Cartesian local coordinates. Small-cap Cartesian comparisons must account for
the local metric scale, not just reuse projected `r` as a physical distance.
