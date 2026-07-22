# dinosaur_spike/test_masks_and_model.py

## Responsibility

This test file validates the first mask and masked two-layer QG model smokes.
It verifies numerical plumbing before longer GPU runs.

## Dinosaur / NeuralGCM API Surface

The tests use `spherical_harmonic.Grid.with_wavenumbers` with the stable
`RealSphericalHarmonics` implementation. The model code exercises Dinosaur
modal/nodal transforms, gradient/divergence helpers, and integration.

## Numerical Assumptions

The tests do not validate a physical instability or conservation law. They
verify that the mask, Coriolis field, background shear pieces, RHS, RK4 step,
and enstrophy diagnostic are finite and shape-consistent.

## Data Layout, Sharding, And Normalization

States are modal arrays shaped like `grid.modal_shape`. Masks and Coriolis
fields are checked in nodal space against latitude arrays from Dinosaur.

## Invariants And Tests

The tests cover:

- Tukey mask bounds and plateau/zero regions,
- sponge-rate formula,
- Coriolis nodal reconstruction,
- regularized deformation coefficient reconstruction and positivity,
- background zonal streamfunction reconstruction,
- constant deformation model PV agrees with the original block-diagonal PV,
- zero-shear RHS equivalence to the original planetary-PV formulation,
- no advection of an exact constant scalar by a streamfunction-induced
  nondivergent velocity in x64,
- layer flux tendency equivalence to Dinosaur's native `get_cos_lat_vector`
  vorticity-flux construction,
- exact diagonal hyperdiffusion decay factors and zero-viscosity equivalence of
  integrating-factor RK4 to explicit RK4,
- zero perturbation under background shear remains zero,
- background shear changes a nonzonal perturbation tendency,
- spectral shell metrics report truncation-edge power fractions,
- RHS shape/finiteness,
- RK4 step shape/finiteness,
- zero mean PV after stepping,
- nonnegative windowed enstrophy.

## Known Risks

These tests run the non-sharded pedagogical transform implementation. They do
not measure GPU performance or spectral ringiness. Those require short run
scripts and diagnostics.
