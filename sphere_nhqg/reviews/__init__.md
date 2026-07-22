# sphere_nhqg/__init__.py

## Responsibility

`__init__.py` defines the first public surface of the spherical package. It
currently re-exports geometry helpers, radial-basis helpers, scalar operator
helpers, spherical mean primitives, and coupled spectral-transform helpers.
Later solver symbols should be added only after their owning modules have
tests and reviews.

## Numerical Assumptions

No numerical method is implemented here. The assumptions are inherited from
the exported modules: unit-sphere stereographic geometry, per-mode
Jacobi/Zernike radial regularity, dense coefficient-space scalar operators,
spherical-area horizontal means, and NumPy/JAX rFFT normalization.

## Data Layout And Normalization

No arrays are created in this file. It only centralizes names that downstream
code may import as `sphere_nhqg.<name>`.

## Invariants And Tests

The geometry, radial, operator, mean, and spectral tests indirectly exercise
these exports through package import. If the public API grows, tests should
import through both direct module paths and the package root.

## Known Risks

Keeping too many implementation details in the package root can make later
refactors painful. New symbols should be exported here only after the owning
module has stable tests.
