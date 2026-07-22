# dinosaur_spike/api_smoke.py

## Responsibility

`api_smoke.py` checks the low-level Dinosaur spherical-harmonic conventions
needed before implementing two-layer QG. It is not a solver; it is an API and
normalization probe.

## Dinosaur / NeuralGCM API Surface

The script uses:

- `dinosaur.spherical_harmonic.Grid.with_wavenumbers`,
- `RealSphericalHarmonics` for the stable pedagogical modal layout,
- `grid.to_nodal` and `grid.to_modal`,
- `grid.laplacian`,
- `grid.integrate`,
- `grid.modal_axes`, `grid.mask`, `grid.modal_shape`, and `grid.nodal_shape`.

It deliberately starts with `RealSphericalHarmonics`, not
`FastSphericalHarmonics`, because the latter has a padded/zero-imag layout
optimized for sharding. Once the mathematical conventions are validated, a
separate sharding smoke test should use the fast implementation.

## Numerical Assumptions

Dinosaur's real modal layout is:

```text
m index: [0, +1, -1, +2, -2, ...]
l index: [0, 1, 2, ...]
```

Basis functions are normalized to unit spherical `L2` norm. Therefore a
physical constant field `1` has coefficient `sqrt(4 pi)` in the `(m,l)=(0,0)`
slot, matching Dinosaur's internal constant normalization.

The unit-sphere Laplacian convention is:

```text
Lap Y_lm = -l(l+1) Y_lm.
```

## Data Layout, Sharding, And Normalization

The script works with 2D modal arrays of shape `grid.modal_shape` and 2D nodal
arrays of shape `grid.nodal_shape`. No vertical or layer dimension is present
yet.

Device behavior:

- `--device cpu` sets `JAX_PLATFORM_NAME=cpu`;
- `--device gpu7` sets `CUDA_VISIBLE_DEVICES=7`;
- `--device default` leaves device selection untouched.

Use CPU for fast API checks. Use GPU 7 only for JIT/benchmark work where GPU
execution is relevant.

## Invariants And Tests

The smoke test verifies:

- modal -> nodal -> modal roundtrip,
- constant-field reconstruction,
- constant coefficient normalization,
- Laplacian eigenvalue sign,
- integral of the constant field equals `4 pi`.

These checks are prerequisites for the two-layer inversion tests.

## Known Risks

The tolerance is set around `float32` behavior because Dinosaur's pedagogical
real transform advertises `float32` modal dtype and may not retain full
float64 accuracy internally. This is acceptable for API discovery, but the
precision choice must be revisited before production-quality NHQGE runs.
