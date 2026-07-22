# dinosaur_spike/run_tuning_sweep.py

## Responsibility

This script runs a small guarded parameter sweep for the two-layer QG prototype.
It is meant to replace ad hoc single-case tuning when looking for a state that
is dynamically active but not dominated by truncation-edge power.

## Numerical Strategy

Each case calls `run_two_layer_solution.py` with a different background shear,
hyperdiffusion rate, timestep, amplitude, and initial maximum wavenumber. The
baseline cases focus on the region between the two earlier failures:

- too tame: shear `1.1`, hyperdiffusion `1e-10`;
- too aggressive: shear `1.2`, weak/no hyperdiffusion.

The defaults use `dt=5e-4` so the same physical interval is sampled with a
smaller explicit step.

The sweep default is now `float64`, following the spherical-operator validation
failure for `fast + float32`. Passing `--allow-float32-fast` forwards the same
override to each child run, but that should be treated as a diagnostic path
rather than a trusted production setup.

The sweep forwards the background-profile and mask controls to each child run.
This matters because the original `solid_body` profile is a neutral control,
whereas `sin_plus_sin3` is the first regular spherical profile with a positive
linear Phillips eigenvalue.

The `high_shear` profile raises the imposed layer velocity difference while
reducing `dt` to `2.5e-4`. This directly tests the simplest
baroclinic-instability lever: increasing vertical shear while leaving the
external base state fixed.

## Safety Guards

Each case uses the solution runner's guards:

- `--stop-top10-fraction`, to stop when too much modal PV power reaches the top
  10% of spherical harmonic degrees;
- `--stop-q-abs-max`, to stop amplitude blowup.

State checkpoints are saved at every diagnostic snapshot so the best case can
later be rendered as exact relative vorticity.

## Outputs

The sweep writes one subdirectory per case plus a root `summary.csv`. The summary
records the final step, final time, enstrophy growth factor, final amplitude,
peak degree, and truncation-edge power fractions.

## Known Limits

This script runs cases sequentially in separate Python processes, so JAX compile
cost is paid per case. That is acceptable for this small tuning pass and keeps
each case's logs/artifacts isolated.
