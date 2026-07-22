# dinosaur_spike/two_layer_qg_formulation.tex

## Responsibility

This TeX file is the living mathematical specification for the Dinosaur
two-layer QG feasibility spike. It records the exact equations, inversion,
masking strategy, and numerical method before solver code is written.

## Dinosaur / NeuralGCM API Surface

The document intentionally leaves coefficient layout and spherical-harmonic
normalization as Dinosaur-owned conventions. The code reviews for the first
API smoke tests must fill in the concrete names and signs once we inspect the
installed package.

Expected API areas:

- coordinate systems,
- spherical harmonic transforms,
- Laplacian or vorticity operators,
- gradient/Jacobian helpers if available,
- sharding/device mesh utilities.

## Numerical Assumptions

The model is two-layer Phillips-style QG:

```text
q1' = Lap psi1 + F1 (psi2 - psi1)
q2' = Lap psi2 + F2 (psi1 - psi2)
```

with total PVs including planetary vorticity. The first implementation uses
constant `F1,F2` so inversion is a 2x2 block per spherical harmonic degree.

The document now also records the equatorial-regularized deformation profiles.
The main candidate is the `f_squared_floor` profile,
`F_i(phi) = F_i* (sin(phi)^2 + s_min^2)/(sin(phi_ref)^2 + s_min^2)`.
This keeps the effective deformation coupling from degenerating at the
equator. The document explicitly notes that latitude-dependent `F_i` destroys
the per-degree block diagonal inversion and must be handled with a dense or
iterative variable-coefficient inversion.

The document now includes the prescribed zonal base state
`Psi_i^0 = -U_i sin(latitude)`. The perturbation equations include
`J(Psi_i^0, q_i')` and `J(psi_i', Q_i^0)`, which are the background-advection
and base-PV-gradient terms needed for a Phillips/baroclinic-instability test.

It also records the newer `sin_plus_sin3` base profile. The exact spherical
linear checks show that the original solid-body profile is neutral, while this
regular curved-shear profile has positive two-layer growth.

The evolution uses a smooth latitude envelope and sponge. This is explicitly a
feasibility model, not a conservation-proof formulation.

## Data Layout, Sharding, And Normalization

The file states the mathematical spherical-harmonic convention but defers
implementation layout to Dinosaur. That is deliberate: guessing layout before
inspecting the package would create silent sign and normalization errors.

The zero spherical harmonic mode is gauge-fixed by setting mean
streamfunctions to zero.

## Invariants And Tests

The document requires early tests for:

- spectral/nodal roundtrip,
- Laplacian eigenvalue convention,
- manufactured two-layer inversion,
- Jacobian sign convention,
- mask/taper spectral cleanliness,
- one-GPU and multi-GPU timing.

## Known Risks

Multiplying nonlinear tendencies by a mask breaks conservation identities and
can create spectral ringing. The formulation keeps this visible as a central
diagnostic rather than hiding it as an implementation detail.

The exact QG validity near the equator remains unresolved. The whole point of
the broad Tukey/sponge setup is to test whether the invalid region can be made
dynamically irrelevant without ruining the target latitudes.
