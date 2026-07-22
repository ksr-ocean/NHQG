# dinosaur_spike/run_two_layer_solution.py

## Responsibility

This script runs a longer masked two-layer QG prototype and writes artifacts that
can be inspected after the run: PV snapshot PNGs, a diagnostics CSV, and the
final modal state.

## Numerical Setup

The initial condition is random but deliberately smooth. It keeps only modal
degrees up to `--init-max-wavenumber`, transforms to nodal space, multiplies by
the latitude mask, normalizes to `--amplitude`, then returns to modal space.
This avoids spending the first visual run on grid-scale random noise.

The default timestepper is the Lawson integrating-factor RK4 step from
`two_layer_model.py`. It treats diagonal modal hyperdiffusion exactly and applies
RK4 to the remaining advective/base/sponge tendency. The legacy explicit RK4 path
is still available with `--time-stepper explicit`. The saved PV fields are the
prognostic PV anomalies, not total PV including planetary vorticity or the
prescribed background PV.

The script now records the actual initial condition at `step 0`. It still
compiles the RK4 step before timing, but that warmup result is discarded rather
than written as the first snapshot.

The background velocity flags can impose the smooth zonal base state
`Psi_i^0 = -U_i sin(latitude)`, with `U1-U2` controlled by
`--background-shear-velocity`. That is the first path toward a Phillips-style
baroclinic-instability run.

The `--background-profile` flag selects the latitude dependence of that zonal
streamfunction. `solid_body` is the original regular profile and is useful as a
neutral control. `sin_plus_sin3` is a regular curved-shear profile that the
linear spectrum script identifies as Phillips unstable.

## Mask And Sponge

The run uses the same broad southern Tukey mask as the model code. The black
contour on each snapshot marks the 0.5 mask level, so it is easy to see whether
activity remains in the intended QG validity region and how much leaks into the
sponge zone.

For full-sphere linear/eigenmode checks, the mask can be made identically one
with `--mask-plateau-north-edge-deg 90 --mask-taper-north-edge-deg 91`, and
the nonlinear-tendency mask can be disabled with
`--no-mask-nonlinear-tendency`.

## Diagnostics

The diagnostics CSV records step, model time, elapsed wall time, cumulative and
interval step time, mask-windowed enstrophy, outside-mask enstrophy, basic
min/max field amplitudes, and shell-power summaries. The main spectral
resolution checks are peak spherical-harmonic degree, mean/RMS degree, and the
power fractions in the top 20% and top 10% of retained degrees.

The companion `shell_power.csv` writes the full layer-summed modal PV power by
spherical-harmonic degree at each snapshot. That is the direct artifact to use
when deciding whether the run is resolved or whether activity is piling up near
the truncation edge.

Long runs can be guarded with `--max-walltime-hours`, `--stop-top10-fraction`,
and `--stop-q-abs-max`. These are checkpoint-time guards, so they stop after the
next diagnostic record rather than interrupting a single RK4 step. The final
state stores the requested step count, completed step count, and stop reason.

The `--save-state-every` option writes modal `state_step_*.npz` checkpoints at
diagnostic records. These are the preferred artifacts for exact derived-field
movies, because relative vorticity can be reconstructed from the modal PV
through the two-layer inversion instead of inferred from rendered PNG colors.

The default precision is `float64`. The runner refuses `--impl fast --dtype
float32` unless `--allow-float32-fast` is supplied, because that combination was
found to fail the constant-scalar advection identity and inject high-degree
latitude bands.

`--hyperdiffusion-rate` and `--hyperdiffusion-order` control the diagonal modal
damping. With the default `--time-stepper ifrk4`, that damping no longer imposes
an explicit RK4 stability ceiling.

## Data Layout

State is stored in Dinosaur real modal arrays. Snapshot fields are converted to
nodal longitude-latitude arrays only at output times. The final `.npz` stores the
modal `q1`, `q2`, latitude/longitude arrays, and the main run parameters needed
to identify the run.

The `--restart-state` flag loads a previous `final_state.npz` and continues from
its modal `q1`, `q2` arrays on a grid with matching shape.

## Known Limits

This is still an unforced prototype unless a background shear is supplied. Even
with background shear, it is not yet a calibrated growth-rate experiment. The
plotting path is intentionally simple and single-process; it is not part of the
performance-critical timestep loop.
