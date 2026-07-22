# dinosaur_spike/two_layer_qg.py

## Responsibility

`two_layer_qg.py` contains the first reusable numerical building blocks for
the Dinosaur spike:

- a modal two-layer PV anomaly state,
- a modal two-layer streamfunction state,
- PV-from-streamfunction evaluation,
- mode-by-mode two-layer inversion,
- mean-PV removal helper.

It deliberately does not implement nonlinear advection, masking, time
stepping, or sharding yet.

## Dinosaur / NeuralGCM API Surface

The module expects a Dinosaur `spherical_harmonic.Grid`-like object with:

- `grid.laplacian(x)`,
- `grid.laplacian_eigenvalues`,
- modal arrays using Dinosaur's real modal layout.

No high-level NeuralGCM APIs are used.

## Numerical Assumptions

The two-layer inversion follows:

```text
q1 = Lap psi1 + F1 (psi2 - psi1)
q2 = Lap psi2 + F2 (psi1 - psi2)
```

Since Dinosaur uses `Lap -> -l(l+1)` on the unit sphere, each mode solves:

```text
[q1] = [ -L - F1    F1 ] [psi1]
[q2]   [  F2    -L - F2] [psi2]
```

with `L = l(l+1)`.

The `(l,m)=(0,0)` barotropic streamfunction is a gauge mode. The inversion sets
both layer mean streamfunctions to zero.

## Data Layout, Sharding, And Normalization

The functions preserve the input modal shape and support leading batch axes
through JAX broadcasting. Arrays are in Dinosaur modal layout, normally
`(..., m_index, l_index)`.

The helper uses `[..., 0, 0]` for the mean mode, matching Dinosaur's real
spherical-harmonic layout.

## Invariants And Tests

The tests should verify:

- manufactured `psi1, psi2 -> q1, q2 -> psi1, psi2` recovery,
- mean streamfunction gauge is zero,
- mean PV removal zeros both layer mean modes,
- batched arrays preserve leading dimensions.

## Known Risks

The inversion assumes constant `F1,F2`. Latitude-dependent deformation physics
would couple modes and should not be added until the constant-coefficient
Dinosaur path has passed sharding and masking tests.

The inversion currently zeroes all `l=0` streamfunction modes. This is a
conservative first-spike gauge choice that avoids the singular barotropic mean
block. If global baroclinic mean structure becomes important, the `l=0`
baroclinic difference should be treated explicitly rather than discarded.
