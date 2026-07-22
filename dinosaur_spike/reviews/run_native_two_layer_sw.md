# dinosaur_spike/run_native_two_layer_sw.py

## Responsibility

This script exercises Dinosaur's built-in two-layer shallow-water equations. It
does not reuse the custom two-layer QG RHS. The goal is to verify that the native
multilayer vorticity/divergence/potential path works in our environment and can
produce checkpoints for visual inspection.

## Numerical Setup

The script builds:

- a `spherical_harmonic.Grid`,
- `LayerCoordinates(2)`,
- `CoordinateSystem(grid, vertical)`,
- `ShallowWaterSpecs` with two densities,
- `ShallowWaterEquations` with two reference layer potentials.

The default `multi_layer` initial condition uses Dinosaur's
`shallow_water_states.multi_layer` helper to construct a balanced two-layer
zonal jet with vertical shear. A small baroclinic potential bump is then added to
perturb the steady state.

The multi-layer branch can also add band-limited baroclinic potential noise.
The perturbation is added to the upper layer and multiplied by
`lower_perturbation_factor` in the lower layer. A value near `-1` is an
interface-like perturbation; a value near `+1` is barotropic thickness noise.
The older `shear_bump` name remains as an alias for this same branch.

The `galewsky` option uses Dinosaur's built-in Galewsky/Scott/Polvani
barotropic-instability test-case generator with two vertical layers. This is the
preferred native-Dinosaur benchmark because it exercises their established
shallow-water initial-condition machinery as well as the two-layer equation
path.

## Time Stepper

The time stepper is Dinosaur's native semi-implicit leapfrog integrator applied
to `ShallowWaterEquations`. It uses an exponential spectral leapfrog filter and a
Robert-Asselin filter, matching the pattern used by Dinosaur's own shallow-water
trajectory helper.

The runner advances in chunks of `min(snapshot_every, remaining_steps)`, so the
requested final step is exact even when `steps` is not divisible by
`snapshot_every`. GPU runs disable JAX's default full-device memory
preallocation, which lets this spike share a busy GPU without failing before the
actual arrays are allocated.

## Outputs

Each diagnostic interval writes:

- a diagnostics CSV row,
- upper/lower vorticity PNG snapshots,
- a modal checkpoint containing vorticity, divergence, potential, mean
  potentials, and run metadata.

Diagnostics track vorticity/divergence amplitudes, total potential bounds,
barotropic/baroclinic vorticity RMS, layer potential mass proxies, mixed
state-shell metrics, upper-vorticity shell metrics, and baroclinic-vorticity
shell metrics. The vorticity-only shell metrics are important because a balanced
pressure field can otherwise dominate the mixed spectrum and hide the vortical
cascade.

## Known Limits

This is a native-Dinosaur plumbing and dynamics spike, not yet a calibrated
Jupiter polar model. The default jet and perturbation are deliberately simple.
If this path is promising, the next step is to choose a physically motivated
two-layer shallow-water base state and forcing/dissipation strategy.
