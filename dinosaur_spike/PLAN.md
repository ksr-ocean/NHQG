# Dinosaur / NeuralGCM Feasibility Spike

Date: 2026-05-22

This is a separate feasibility spike from the `sphere_nhqg` stereographic
Jacobi/Zernike reference scaffold. The goal here is to test whether
Dinosaur/NeuralGCM lower-level spherical spectral infrastructure can support a
custom QG/NHQGE-like model while inheriting its multi-device sharding approach.

The existing `sphere_nhqg` work remains useful as a correctness reference for
local spherical geometry, regularity, and operator identities. It should not be
expanded into a production solver until this global-spectral path is evaluated.

## Review Directive

Every Python module, executable script, notebook-like experiment, and
nontrivial test added under `dinosaur_spike/` must have a corresponding
pedagogical review under `dinosaur_spike/reviews/`.

The review must be updated in the same change as the code or experiment. It
should explain:

- what the file is responsible for,
- what Dinosaur/NeuralGCM API surface it relies on,
- the numerical assumptions and sign conventions,
- data layout, sharding, and transform normalization,
- invariants or tests that should catch mistakes,
- known limitations and next changes expected.

Missing review coverage is treated as an incomplete implementation.

## Operator-Precision Directive

Before trusting any long spherical run, validate the metric/operator path with
`dinosaur_spike/validate_spherical_operators.py`. The required sanity check is
that a streamfunction-induced nondivergent velocity does not advect an exact
constant scalar:

```text
_layer_flux_tendency(grid, psi, 1, None) ~= 0
```

The `FastSphericalHarmonics` plus `float32` path fails this check at useful
resolutions and injects high-degree latitude-band noise. Production/tuning runs
should therefore use `float64` by default. Any `fast + float32` run must be
treated as an explicit diagnostic/benchmark only, not a trusted solution.

## Linear-Instability Directive

Do not infer Phillips/baroclinic instability from random nonlinear movies.
Before a base state is used for long runs, compute its exact spherical
linearized spectrum with `dinosaur_spike/linear_phillips_spectrum.py`.

The original `solid_body` profile
`Psi_i^0 = -U_i sin(phi)` is a useful regular neutral control: its low-order
`m`-block spectra are neutral to roundoff even at large vertical shear. The
first regular profile that currently produces a clear Phillips-type mode is
`sin_plus_sin3`,

```text
Psi_i^0 = -U_i (sin(phi) + 0.75 sin(phi)^3).
```

For a clean validation, save the most unstable eigenvector from the spectrum
tool, restart `run_two_layer_solution.py` from that checkpoint at tiny
amplitude, and verify that `0.5 d log(enstrophy) / dt` matches the linear
growth rate before nonlinear terms matter.

Candidate global-QG operators must also be checked with an equatorial
regularization strategy. The current low-resolution tests support:

- `constant`: the original block-diagonal deformation coupling;
- `f_squared_floor`: deformation coupling proportional to
  `sin(phi)^2 + sin_floor^2`, normalized at a reference latitude;
- `inverse_f_squared_floor`: diagnostic reciprocal-f coupling using the same
  floor.

Once deformation varies with latitude, PV inversion is no longer diagonal in
spherical harmonic degree. For now, use the dense `m`-block inversion in
`linear_phillips_spectrum.py` to validate spectra. Do not run the nonlinear
stepper with variable deformation until a consistent precomputed or iterative
PV inversion is added there.

## Motivation

The dense Jacobi/Zernike radial transform path is mathematically clean but has
poor nonlinear-transform scaling compared with FFT/spherical-harmonic
infrastructure. The hard engineering problem for production is no longer just
"spherical geometry"; it is scalable global spherical transforms and
multi-GPU sharding.

Dinosaur is attractive because it already provides JAX spherical spectral
machinery built for accelerator execution. The feasibility question is whether
we can build the reduced QG/NHQGE-like dynamics on top of that lower-level
infrastructure without depending on unstable high-level NeuralGCM model
internals.

## Not A Narrow Polar-Cap Simulation

Do not frame the first successful prototype as a small polar-cap computation.
The more ambitious target is a broad spherical model whose trusted dynamical
region includes the polar and midlatitude zones while smoothly suppressing the
formally invalid equatorial region.

The mask should be a smooth Tukey-style latitude window, not a hard cap wall.
The taper can begin equatorward of about `30S` in the southern-hemisphere
configuration, with analogous choices for a northern or two-hemisphere setup.
The precise plateau and taper widths are experimental parameters, not geometry
facts.

The mask is best thought of as a validity/forcing/sponge envelope:

- full or near-full dynamics over the target high-latitude and midlatitude
  region,
- smooth transition through a broad buffer,
- strong damping or tendency suppression where the QG asymptotic is not meant
  to be trusted,
- no spectral sharp edges.

## First Dynamical Prototype: Two-Layer Phillips QG

A barotropic vorticity equation is too forgiving because nothing essential
breaks when `f -> 0`. It would test transforms and sharding, but not the
dangerous part of globalizing QG.

The better first dynamical model is a two-layer Phillips-style QG system:

```text
q1 = Lap psi1 + F1 (psi2 - psi1) + f
q2 = Lap psi2 + F2 (psi1 - psi2) + f + topography
```

with layerwise PV advection and optional damping/relaxation. This exposes the
baroclinic/deformation-radius issue that a global QG model must confront.

For the first implementation, use constant polar/midlatitude `F1, F2` so the
PV-to-streamfunction inversion is block-diagonal for each spherical harmonic
degree `l`. Per `(l, m)`:

```text
[q1 - f] = [ -L - F1    F1 ] [psi1]
[q2 - f]   [  F2    -L - F2] [psi2]
```

where `L = l(l+1)/a^2`, subject to Dinosaur's sign conventions.

Later variants can test latitude-dependent effective deformation physics, but
that must be paired with a regularized PV inversion and a linear spectrum check.

## Mask / Sponge Design

Use a smooth Tukey-style latitude envelope `chi(phi)`:

- `chi = 1` in the trusted dynamical region,
- `0 < chi < 1` across a broad taper,
- `chi = 0` or near zero in the invalid/equatorial reservoir.

Candidate uses:

- multiply imposed forcing/relaxation by `chi`,
- sponge PV anomalies or streamfunction outside the trusted region,
- optionally multiply nonlinear tendencies by `chi` only after checking
  conservation damage,
- compute primary diagnostics weighted by `chi` or by a stricter diagnostic
  window.

Avoid a discontinuous cap or hard-wall boundary unless specifically testing a
wall model. Sharp masks will ring in spherical harmonics and will obscure the
actual question.

## What The Spike Must Answer

1. Can Dinosaur's lower-level spherical-harmonic transforms be used directly
   for custom prognostic variables?
2. Can its sharding approach be reused for custom tendencies across multiple
   GPUs?
3. What is the single-GPU and multi-GPU cost of:
   - physical-to-spectral transforms,
   - spectral-to-physical transforms,
   - layerwise Jacobians,
   - per-`l` two-layer inversion?
4. Does a broad Tukey/tapered QG region stay spectrally clean?
5. Does the equatorial damping/tendency suppression prevent invalid dynamics
   from feeding back into the target region?
6. Does two-layer baroclinic dynamics behave qualitatively sensibly under the
   mask?

## Minimal Milestones

1. Install/import Dinosaur and identify the low-level modules for:
   - spherical harmonic grids,
   - transforms,
   - vorticity/streamfunction-style operators,
   - sharding/device mesh setup.

2. Write a standalone transform smoke test:
   - scalar field spectral -> nodal -> spectral roundtrip,
   - Laplacian/eigenvalue check,
   - one-device first.

3. Add two-layer inversion:
   - manufactured `psi1, psi2`,
   - compute `q1, q2`,
   - invert back per spherical harmonic mode.

4. Add nonlinear PV advection:
   - compute `J(psi_i, q_i)`,
   - verify self/antisymmetry-like sanity checks where applicable,
   - add hyperdiffusion or spectral filtering.

5. Add Tukey latitude envelope:
   - forcing mask,
   - damping/sponge mask,
   - diagnostic mask,
   - ringiness diagnostics in spectral space.

6. Run a short baroclinic/Phillips experiment:
   - imposed vertical shear or relaxation,
   - broad southern-hemisphere or two-hemisphere active region,
   - diagnostics for leakage into the damped region.

7. Repeat the smoke and step benchmarks under Dinosaur-style sharding.

## Decision Criteria

Prefer the Dinosaur/global-spectral path if:

- custom variables and tendencies can use the low-level sharded transform stack,
- multi-GPU scaling is real without a major custom sharding rewrite,
- the Tukey/sponge region remains spectrally clean,
- two-layer dynamics are robust enough to justify porting NHQGE vertical
  structure.

Fall back to a stereographic/colatitude finite-difference or finite-volume
route if:

- Dinosaur internals are too coupled to primitive-equation assumptions,
- custom sharding is effectively as hard as writing our own,
- the global mask contaminates the target region,
- the `f -> 0`/deformation issue cannot be controlled without unphysical
  damping.
