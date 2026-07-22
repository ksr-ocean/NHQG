# sphere_nhqg/tests/test_spectral.py

## Responsibility

The spectral tests validate the first coupled azimuth/radial transform layer.
They focus on normalization and product projection, not on spherical dynamics.

## Numerical Assumptions

The tests enforce NumPy/JAX rFFT normalization. A physical constant `c` must
have coefficient `c * Nphi` in the `m=0, n=0` slot. The `m=1, n=0` basis mode
with coefficient `Nphi / 2` represents the regular field `rho cos(phi)`, not
the pole-singular field `cos(phi)`.

Overresolved transforms are expected to preserve physical amplitudes. This is
the analog of the Cartesian solver's hard-won FFT-padding normalization lesson.

## Data Layout And Normalization

Coefficient arrays use `(m, radial_coeff, batch...)`. Physical arrays use
`(radial_node, phi, batch...)`.

Random roundtrip coefficients are made compatible with a real physical field:
the `m=0` and Nyquist modes are real, while interior modes may be complex.

## Invariants And Tests

The tests cover:

- constant DFT normalization,
- coefficient -> physical -> coefficient roundtrip on an overresolved grid,
- padded inverse transform amplitude for `cos(phi)`,
- dealiased `(rho cos(phi))^2 = rho^2 / 2 + rho^2 cos(2phi) / 2`,
- radial product `rho^2 * rho^2 = rho^4`,
- shape validation for physical-to-coefficient projection.
- radial and azimuthal derivatives of `rho cos(phi)`,
- analytic flat Jacobian for `0.5 rho^2` and `rho cos(phi)`,
- spherical Jacobian pointwise metric scaling,
- self-Jacobian and antisymmetry after coefficient projection.

## Known Risks

The derivative and Jacobian tests avoid the azimuthal Nyquist mode. Production
dealiasing should avoid relying on Nyquist derivatives under rFFT conventions.
