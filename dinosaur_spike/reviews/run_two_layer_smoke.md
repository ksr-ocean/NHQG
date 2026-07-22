# dinosaur_spike/run_two_layer_smoke.py

## Responsibility

This script runs a short masked two-layer QG RK4 smoke test on CPU or GPU 7. It
is the first end-to-end stepper exercise for the Dinosaur spike.

## Dinosaur / NeuralGCM API Surface

The script uses `spherical_harmonic.Grid.with_wavenumbers` and the lower-level
model functions in `two_layer_model.py`. It does not use high-level NeuralGCM
model APIs.

## Numerical Assumptions

The run uses small random modal PV anomalies and short explicit RK4 steps. This
is a numerical plumbing smoke, not a physically meaningful Phillips experiment.
The background velocity flags can turn on the zonal base state used for
baroclinic-instability plumbing checks, but the random short run is still not a
growth-rate validation.

The smoke exposes the same background-profile and mask controls as the longer
runner so quick checks can use either the neutral `solid_body` control or the
unstable `sin_plus_sin3` profile.

The initial condition is not mask-localized yet, so outside-mask enstrophy is a
diagnostic but not expected to vanish.

## Data Layout, Sharding, And Normalization

State arrays are Dinosaur modal arrays. The default execution target is
`--device gpu7`, which sets `CUDA_VISIBLE_DEVICES=7`. The `--dtype` flag is
applied before JAX import by setting `JAX_ENABLE_X64`; float64 is the default
because the `fast + float32` spherical operator path injects latitude-band
noise. That unsafe combination requires the explicit `--allow-float32-fast`
override.

The script defaults to the same integrating-factor RK4 step as the longer runner
and can still use the legacy explicit RK4 path with `--time-stepper explicit`.
It compiles the chosen step with a discarded warmup result before timing, so the
reported initial diagnostics are the true initial state.

The script reports modal/nodal shapes, initial/final windowed enstrophy,
outside-mask enstrophy, mean RK4 step time, and backend memory stats.

## Invariants And Tests

Expected smoke behavior:

- stepper compiles and runs,
- diagnostics remain finite,
- mean step time is measurable,
- memory stats are reported when available.

## Known Risks

This is still single-device unless a future version passes a Dinosaur SPMD
mesh. It also lacks forcing, tuned sponge parameters, spectral ringiness
diagnostics, and a physically structured initial condition.
