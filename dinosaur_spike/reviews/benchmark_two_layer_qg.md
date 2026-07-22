# dinosaur_spike/benchmark_two_layer_qg.py

## Responsibility

This script benchmarks the first relevant Dinosaur two-layer QG kernel on one
device. The kernel includes:

- two-layer modal PV-to-streamfunction inversion,
- velocity construction via Dinosaur's spherical operators,
- nodal PV flux products,
- modal projection of fluxes,
- flux-form PV advection tendency.

It is intended to answer whether the Dinosaur spectral stack is promising on
GPU before adding full time stepping, masks, or sharding.

## Dinosaur / NeuralGCM API Surface

The script uses lower-level Dinosaur APIs:

- `spherical_harmonic.Grid.with_wavenumbers`,
- `RealSphericalHarmonics` or `FastSphericalHarmonics`,
- `grid.cos_lat_grad`,
- `grid.k_cross`,
- `grid.to_nodal`,
- `grid.to_modal`,
- `grid.div_cos_lat`,
- `grid.sec2_lat`,
- modal masks and shapes.

No high-level NeuralGCM model APIs are used.

## Numerical Assumptions

The benchmark tendency follows Dinosaur shallow-water's flux-form pattern.
For each layer, it computes a nondivergent velocity from streamfunction and
then computes approximately:

```text
- div(u q)
```

on the sphere, using Dinosaur's `v cos(latitude)` convention and `sec2_lat`
factor. This is not yet the full validated two-layer model; it is a realistic
transform-heavy RHS kernel for timing.

## Data Layout, Sharding, And Normalization

The script accepts:

- `--device gpu7`, which sets `CUDA_VISIBLE_DEVICES=7`,
- `--impl fast`, the default production-relevant transform implementation,
- `--dtype float32`, the default benchmark precision.

It reports:

- modal and nodal shapes,
- Dinosaur basis memory in MB,
- a rough lower-bound array memory estimate,
- mean kernel time after JIT warmups,
- JAX backend memory stats when available.

The memory estimate is not a full XLA peak. It is a sanity lower bound for
live arrays plus constants. Treat actual H200 memory behavior as authoritative.

## Invariants And Tests

The script should be run after:

- `api_smoke.py` passes,
- `test_two_layer_qg.py` passes.

Those tests pin down transform normalization and inversion signs. This script
only measures performance.

## Known Risks

The benchmark is single-device unless a future option adds an SPMD mesh. It
therefore tests Dinosaur's transform implementation on GPU 7, not yet the
multi-GPU advantage that motivates this path.

The flux-form tendency lacks the Tukey mask, Coriolis/planetary PV, sponge,
and diffusion. Those should be added only after this kernel's scaling looks
viable.
