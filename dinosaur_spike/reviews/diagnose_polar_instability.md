# dinosaur_spike/diagnose_polar_instability.py

## Responsibility

This script inspects saved `state_step_*.npz` checkpoints from a two-layer QG run
and asks whether the growing pointwise amplitudes are localized on the polar
latitude rings.

## Numerical Meaning

The diagnostic reconstructs the same Dinosaur spherical grid and reloads modal
PV anomalies `q1,q2`. It then inverts the constant-deformation two-layer PV
relation to recover `psi1,psi2`, rebuilds the zonal base state and damping
parameters from checkpoint metadata, and evaluates the same metric-sensitive
nonlinear flux ingredients used by `two_layer_model.py`:

`vcos = k_cross(cos_lat_grad(psi))`

`metric_flux = sec^2(latitude) * vcos * q`

This is not a substitute for the RHS, but it isolates the polar-sensitive factor
that can become large if nodal nonlinear products fail to satisfy the regularity
conditions near the coordinate singularity.

## Outputs

The CSV records one row per checkpoint with:

- the latitude of the global `max |q'|`,
- the latitude of the global metric-flux maximum,
- the latitude of the RHS tendency maximum,
- total enstrophy and polar-cap enstrophy fractions.

The NPZ stores full latitude profiles for later replotting. The time-series plot
shows whether maxima migrate onto the first/last latitude rings, while the
latitude-profile plot shows where growth is concentrated at selected times.

## Interpretation

A polar numerical instability is supported if the pointwise `q`, metric-flux, or
tendency maxima sit on the first few latitude rings before the solution becomes
nonfinite, especially when the polar-cap enstrophy fraction remains small. That
pattern means a tiny physical area is controlling the maximum norm through the
spherical metric factors rather than through a broad turbulent cascade.

If the maxima remain in the intended midlatitude band until late times, the
failure is less likely to be a pure polar-ring regularity problem and more likely
to be a physical or model-level instability of the chosen base state.
