# sphere_nhqg/spectral.py

## Responsibility

`spectral.py` is the first coupled horizontal transform layer. It connects
nonnegative azimuthal rFFT modes with radial Jacobi/Zernike coefficients and a
common physical product grid.

It provides:

- `SpectralGrid`, a dense transform/projection cache,
- coefficient to physical transform,
- physical to coefficient projection,
- a physical-constant coefficient helper,
- a dealiased product through overresolved physical space.
- radial and azimuthal derivative evaluation,
- flat and spherical Jacobian evaluation/projection.

This is still a dense prototype, but it now contains the nonlinear geometric
core needed by the future explicit RHS.

## Numerical Assumptions

Azimuthal coefficients follow NumPy/JAX rFFT normalization:

```text
physical = irfft(coefficients)
coefficients = rfft(physical)
```

Therefore a physical constant `1` is represented by the `m=0` coefficient
`Nphi`, not by coefficient `1`. This mirrors the Cartesian solver's FFT
normalization and avoids silently mixing Fourier-series and DFT conventions.

When `Nphi_phys > Nphi`, inverse transforms explicitly zero-pad high azimuthal
modes and multiply by `Nphi_phys / Nphi` so the physical values represent the
same function, not the attenuated padded-DFT values. Forward transforms multiply
by `Nphi / Nphi_phys` to return to the base-grid DFT normalization.

Radially, all modes are evaluated on one common Legendre grid in `xi`. For each
azimuthal mode `m`, projection uses a dense weighted least-squares/Galerkin
matrix:

```text
projection_m = (V_m^T W V_m)^-1 V_m^T W.
```

This gives exact discrete roundtrips on the chosen product grid while keeping
the first implementation straightforward.

## Data Layout And Normalization

Coefficient arrays use:

```text
coeffs[m, n, ...]
```

Physical arrays use:

```text
values[j_r, j_phi, ...]
```

`SpectralGrid.Nphi` and `SpectralGrid.Nr` describe the retained coefficient
resolution. `Nphi_phys` and `Nr_phys` describe the physical product grid, which
may be overresolved for dealiasing.

## Invariants And Tests

The spectral tests check:

- constants use the expected DFT normalization,
- coefficient to physical to coefficient roundtrips survive azimuthal/radial
  overresolution,
- padded and unpadded transforms produce the same physical values for
  band-limited fields,
- the regular `m=1,n=0` mode squared produces exactly the expected `m=0`
  and `m=2` coefficients,
- radial products such as `rho^2 * rho^2` project to the expected represented
  function.
- radial and azimuthal derivatives of a regular `rho cos(phi)` mode,
- analytic flat Jacobian for a regular manufactured pair,
- `J_sphere = mu^-1 J_flat` pointwise,
- self-Jacobian and antisymmetry identities.

## Known Risks

The common radial grid is dense and simple, not production optimized. It is a
good first target because normalization is visible and tests are small. Later,
we may replace it with mode-aware quadrature or structured transforms if memory
or throughput requires it.

The radial projection is a discrete projection on the chosen product grid. For
nonlinear products, the product grid must be sufficiently overresolved. Using a
base-resolution grid here would recreate aliasing problems similar to the
early Cartesian runs.

Derivative tests avoid the azimuthal Nyquist mode. Real-grid spectral
derivatives at the Nyquist frequency are ambiguous under rFFT conventions and
should be filtered out by production dealiasing.
