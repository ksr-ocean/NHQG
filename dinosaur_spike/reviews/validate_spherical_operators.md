# dinosaur_spike/validate_spherical_operators.py

This script checks the most basic metric identity used by the two-layer QG
prototype: a nondivergent streamfunction velocity must not advect a constant
scalar. In the implemented flux form, `_layer_flux_tendency(grid, psi, 1, None)`
should be zero up to transform precision.

The test constructs a low-pass random streamfunction, normalizes its
cos-latitude velocity scale to order one, inserts an exact modal constant using
Dinosaur's `add_constant`, and reports the nodal RMS/max residual plus where the
residual lives in spherical-harmonic degree.

This is the diagnostic that exposed the bad production setup: `fast + float32`
injects high-degree noise in the flux-divergence operator, while `fast +
float64` and `real + float64` keep the residual near roundoff. The script exits
nonzero when the residual exceeds the configured thresholds, so it can be run
before long sweeps.

The check is intentionally narrow. It does not validate baroclinic instability,
the mask, the sponge, or timestep stability; it only protects the spherical
metric/operator path from a defect that immediately appears as latitude bands.
