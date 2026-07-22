# dinosaur_spike/two_layer_model.py

## Responsibility

`two_layer_model.py` adds the first actual masked two-layer QG dynamics:

- Coriolis field in Dinosaur modal representation,
- regularized latitude-dependent deformation coefficients for linear tests,
- smooth zonal background streamfunctions and base PVs,
- nodal Tukey latitude mask,
- flux-form layer PV tendency,
- sponge damping outside the trusted region,
- optional modal hyperdiffusion,
- explicit RK4 step and Lawson integrating-factor RK4 step,
- a simple windowed enstrophy diagnostic.

## Dinosaur / NeuralGCM API Surface

The module uses lower-level Dinosaur grid methods:

- `grid.latitudes`,
- `grid.to_modal`,
- `grid.to_nodal`,
- `grid.cos_lat_grad`,
- `grid.k_cross`,
- `grid.div_cos_lat`,
- `grid.sec2_lat`,
- `grid.clip_wavenumbers`,
- `grid.integrate`.

No high-level NeuralGCM model object is used.

## Numerical Assumptions

The layer tendency uses a Dinosaur shallow-water-style flux form. It computes a
nondivergent velocity from each layer streamfunction, multiplies by total PV,
and takes a spherical flux divergence.

With nonzero background shear, the code evolves perturbation PV about the
zonal base state. For each layer it computes
`-J(psi_prime, q_prime + Q0) - J(Psi0, q_prime)`, omitting
`-J(Psi0, Q0)` because the prescribed base state is zonal and should not evolve
by itself. This is the Phillips/baroclinic-instability hook.

The default `solid_body` background uses `Psi_i^0 = -U_i sin(latitude)` and is
regular at the poles, but the exact spherical linear spectrum shows it is
neutral rather than Phillips unstable. The `sin_plus_sin3` profile keeps polar
regularity while adding curvature to the zonal shear; it is the first profile
that the linear spectrum tool finds to have positive baroclinic growth.

The model now also defines deformation-coefficient profiles. `constant` is the
original block-diagonal inversion path. `f_squared_floor` uses
`sin(latitude)^2 + sin_floor^2`, normalized at a reference latitude, so the
effective deformation coupling has an equatorial floor instead of vanishing.
`inverse_f_squared_floor` is included as a diagnostic reciprocal-f case with
the same floor. These variable-coefficient profiles are used by the exact
linear spectrum tool.

The nonlinear tendency can be multiplied by the smooth latitude mask. This is
not a conservative theorem; it is a feasibility device to test broad
high-latitude/midlatitude dynamics embedded in a full-sphere spectral grid.

Sponge damping removes PV anomalies where the mask drops below one.

Hyperdiffusion is diagonal in spherical-harmonic degree:
`dq/dt = -nu [l(l+1)]^p q`. The legacy `rk4_step` includes it explicitly in the
RHS, while `ifrk4_step` treats this diagonal linear part exactly with a Lawson
integrating factor and applies RK4 only to the remaining advective/base/sponge
tendency. With `nu=0`, `ifrk4_step` reduces to the same update as explicit RK4.

## Data Layout, Sharding, And Normalization

State arrays are Dinosaur modal arrays with shape `grid.modal_shape`.
Mask/sponge arrays are nodal arrays with shape `grid.nodal_shape` and are
projected to modal space only through `grid.to_modal`.

The explicit RK4 step is retained for comparisons. The integrating-factor RK4
step is the preferred path for retrying viscous runs because it avoids the
explicit timestep ceiling from high-order hyperdiffusion.

Nonlinear stepping currently refuses variable deformation profiles. That is
intentional: once `F_i` varies with latitude, PV inversion is no longer
block-diagonal in spherical harmonic degree, so the production stepper needs a
precomputed or iterative variable-coefficient inversion rather than silently
using the constant-`F` inverse.

## Invariants And Tests

Tests should check:

- Coriolis modal/nodal roundtrip,
- regularized deformation-coefficient formula and positivity,
- background streamfunction roundtrip,
- `sin_plus_sin3` background profile reconstruction,
- constant-profile model PV equals the original block-diagonal PV formula,
- zero background recovers the original no-shear RHS,
- zero perturbation has no spontaneous tendency under background shear,
- nonzero background shear changes a nonzonal perturbation tendency,
- mask shape and bounds,
- RHS finiteness and shape preservation,
- RK4 step finiteness,
- integrating-factor decay factors and zero-hyperdiffusion equivalence to RK4,
- mean PV remains zero after stepping.

## Known Risks

The flux-form tendency sign follows Dinosaur's velocity conventions and still
needs deeper validation against a known analytic or benchmark flow. The current
tests are smoke tests, not physics validation.

Masking the nonlinear tendency can inject spectral artifacts. The next
diagnostics should explicitly track spectrum/ringiness of the masked tendency.
