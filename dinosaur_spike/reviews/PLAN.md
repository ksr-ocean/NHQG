# dinosaur_spike/PLAN.md

## Responsibility

`PLAN.md` defines the separate Dinosaur/NeuralGCM feasibility spike. It keeps
the global spherical-harmonic/sharding path distinct from the stereographic
Jacobi/Zernike reference scaffold in `sphere_nhqg/`.

## Dinosaur / NeuralGCM API Surface

The plan intentionally targets lower-level Dinosaur infrastructure:

- spherical harmonic grids and transforms,
- differential operators,
- possible vorticity/streamfunction helper APIs,
- multi-device sharding/device mesh setup.

It does not assume the high-level trained NeuralGCM model API is stable or
appropriate for custom NHQGE/QG dynamics.

## Numerical Assumptions

The first meaningful model should be two-layer Phillips-style QG rather than
barotropic vorticity. The reason is diagnostic: two-layer QG exposes the
baroclinic/deformation-radius and `f -> 0` issues that a globalized QG model
must face, while barotropic vorticity is too forgiving.

The trusted dynamical domain is broad, not a narrow polar cap. Invalid or
less-trusted low-latitude dynamics should be controlled through a smooth
Tukey-style mask, forcing envelope, and sponge, not through a hard spectral
boundary.

The plan now includes an explicit operator-precision gate: long runs must pass
the constant-scalar advection sanity check before their dynamics are trusted.
This captures the observed `FastSphericalHarmonics + float32` failure mode,
where transform/metric roundoff becomes high-degree latitude bands.

It also now requires an exact linear-instability check before interpreting
random nonlinear movies. This records the important finding that the original
solid-body base state is neutral, and that the `sin_plus_sin3` profile is the
first regular profile with a verified positive Phillips-type eigenvalue.

The plan now treats equatorial regularization as part of the operator, not as a
postprocessing tweak. Latitude-dependent deformation profiles such as
`f_squared_floor` must be validated with the dense `m`-block spectrum tool,
because they are no longer diagonal in spherical harmonic degree.

## Data Layout, Sharding, And Normalization

The plan does not yet define concrete array layouts because those should be
inherited from Dinosaur where possible. The feasibility spike must document
Dinosaur's spectral coefficient layout, nodal layout, spherical-harmonic
normalization, and sharding annotations before any custom solver code is
trusted.

## Invariants And Tests

The planned validation ladder starts with:

- spectral/nodal roundtrip,
- Laplacian eigenvalue checks,
- constant-scalar advection by a nondivergent velocity,
- exact `m`-block linear spectra for candidate zonal base states,
- exact `m`-block spectra for regularized latitude-dependent deformation
  profiles,
- manufactured two-layer inversion,
- nonlinear PV advection sanity checks,
- Tukey-mask ringiness diagnostics,
- single-GPU and multi-GPU step benchmarks.

These are intentionally ordered so API and normalization mistakes appear before
time-stepping or physics interpretation.

## Known Risks

Dinosaur may not expose the necessary lower-level pieces cleanly enough for
custom dynamics. If using its sharding requires effectively rewriting the
infrastructure, this path loses its main advantage.

The broad Tukey mask may contaminate the target region through spectral
ringing or artificial tendency leakage. That is a central feasibility question,
not a detail to postpone.
