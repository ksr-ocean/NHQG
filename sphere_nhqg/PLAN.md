# Spherical NHQG Implementation Plan

Date: 2026-05-22

This is the active plan for the stereographic polar-cap NHQG solver. The
Cartesian CPU/Julia route is closed for this repository: the separate Julia
implementation and its documentation live elsewhere. This code path targets
JAX on the available GPU/H200 environment, with the current Cartesian solver
kept as the validated reference.

## Review Directive

Every Python module, executable script, and nontrivial test added for the
spherical solver must have a corresponding pedagogical review under
`sphere_nhqg/reviews/`.

The review must be updated in the same change as the code. It should explain:

- what the file is responsible for,
- the numerical assumptions it encodes,
- the data layout and normalization conventions,
- the invariants or tests that should catch mistakes,
- known limitations or later changes expected.

Missing review coverage is treated as an incomplete implementation, even if
the code itself runs.

## Package Layout

Planned package structure:

```text
sphere_nhqg/
├── PLAN.md
├── __init__.py
├── config.py
├── geometry.py
├── radial.py
├── operators.py
├── spectral.py
├── mean_exchange.py
├── solver.py
├── diagnostics.py
├── io.py
├── tests/
└── reviews/
```

The existing `nhqg/` Cartesian package should not grow spherical conditionals
except where genuinely shared utility extraction is justified. Keep the new
horizontal geometry isolated until the operator tests are mature.

## Cartesian Lessons To Preserve

- Do not return to vertical collocation. The spherical solver inherits the
  Cartesian coefficient-space Chebyshev Galerkin/tau vertical machinery.
- Treat dealiasing as part of the method. Raw nonlinear products and raw
  horizontal means are not trusted diagnostics.
- Keep the `balanced_sbp2_pc` idea: the thermal exchange must be a discrete
  adjoint pair on the SBP work grid. The spherical change is the horizontal
  mean primitive, not the `z`-direction SBP algebra.
- Build operator tests before the time stepper. Geometry, transforms,
  Laplacian, Helmholtz inversion, Jacobian identities, and spherical means
  should all fail loudly before a full ARS step exists.
- Precompute implicit radial operators, but track memory shape from day one.
  Cartesian shell deduplication becomes per-azimuthal-mode dense radial
  factors/inverses.
- Diagnostics should lead with trusted spherical-area/dealiased/SBP values.
  Raw CGL or underresolved comparisons are secondary audits.

## Implementation Milestones

1. Geometry and area weights.
   Implement stereographic radius, conformal factor, Coriolis profile, exact
   cap area, radial area density, and spherical-area quadrature checks.

2. Radial Jacobi/Zernike basis.
   Implement per-azimuthal-mode nodes, weights, transforms, derivative
   matrices, pole regularity, and jet tau rows. Validate against exact
   polynomial/radial identities and, when available, Dedalus DiskBasis
   references.

3. Scalar spherical operators.
   Implement `Lap_sphere = mu^-1 Lap_flat`, `J_sphere = mu^-1 J_flat`, and
   per-mode Helmholtz/PV inversion. Start with dense matrices; optimize only
   after residual tests pass.

4. Dealiased nonlinear path.
   Implement azimuthal FFT plus overresolved radial transforms. Normalize the
   transform path explicitly and test alias-sensitive products.

5. Spherical mean and thermal exchange.
   Replace Cartesian horizontal means with spherical-area means in the
   `balanced_sbp2_pc` predictor-corrector structure. The first target is an
   SBP exchange residual at roundoff for manufactured fields.

6. ARS222 spherical stepper.
   Assemble the full state and IMEX step only after the operator and exchange
   tests are stable.

7. Diagnostics and run scripts.
   Add spherical-area Nusselt, exchange residuals, modal spectra, CFL based on
   spherical arc length, checkpointing, and a GPU launch script. If GPU 7 is
   still free during local runs, use `CUDA_VISIBLE_DEVICES=7`.

## Theoretical Gates

These points can be developed around but must be confirmed before production
claims:

- the streamfunction-PV inversion operator is exactly
  `(Lap_sphere - Ld^-2) psi = -q'`,
- the planetary-stretching term is `J_sphere(psi, f)` with no additional
  curvature terms,
- the temperature equation carries no horizontal metric factor in the vertical
  buoyancy coupling,
- `Ld` is independent of horizontal position, or the implementation explicitly
  supports `Ld(r, z)`.

## Validation Ladder

Each stage should have a small, fast test before moving on:

- exact cap area and small-cap area limit,
- radial transform roundtrip and pole regularity,
- spherical Laplacian residuals on known smooth modes,
- Helmholtz inversion residuals per azimuthal mode,
- Jacobian antisymmetry / self-Jacobian identities under spherical area
  weighting,
- spherical mean of manufactured products against high-order quadrature,
- SBP thermal exchange closure,
- small-cap comparison against the Cartesian solver.
