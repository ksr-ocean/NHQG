# dinosaur_spike/test_two_layer_qg.py

## Responsibility

This test file validates the constant-coefficient two-layer modal inversion
before nonlinear tendencies, masks, or time integration are introduced.

## Dinosaur / NeuralGCM API Surface

The tests use `dinosaur.spherical_harmonic.Grid.with_wavenumbers` with
`RealSphericalHarmonics`. They rely on `grid.modal_shape`, `grid.mask`, and
`grid.laplacian` through the implementation.

## Numerical Assumptions

The tests manufacture streamfunctions, compute PV anomalies with the same
two-layer operator, then invert back. The mean streamfunction mode is set to
zero because it is a gauge.

## Data Layout, Sharding, And Normalization

Arrays use Dinosaur's modal shape `(m_index, l_index)`. One test stacks a
leading batch dimension to ensure the inversion broadcasts across leading axes,
which is a precursor to treating layers/vertical/batches cleanly.

## Invariants And Tests

The tests cover:

- manufactured inversion recovery,
- zero mean streamfunction gauge,
- mean-PV removal,
- batch-shape preservation.

## Known Risks

These tests use the pedagogical non-sharded transform implementation and do not
exercise `FastSphericalHarmonics` or SPMD mesh behavior. A separate sharding
test is required before this can be called a Dinosaur scaling spike.
