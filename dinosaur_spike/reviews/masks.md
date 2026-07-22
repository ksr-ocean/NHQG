# dinosaur_spike/masks.py

## Responsibility

`masks.py` provides smooth latitude envelopes for the broad spherical QG
prototype. These masks define where the QG dynamics are trusted and where
sponge damping suppresses invalid or less-trusted low-latitude behavior.

## Dinosaur / NeuralGCM API Surface

The module does not depend directly on Dinosaur. It consumes latitude arrays,
which will normally come from `grid.latitudes`.

## Numerical Assumptions

The first mask is southern-hemisphere oriented:

- `chi = 1` southward of `plateau_north_edge_deg`,
- `chi` tapers smoothly through a cosine window,
- `chi = 0` northward of `taper_north_edge_deg`.

The default plateau edge is `30S`, with taper extending to `5N`. This is a
starting point, not a physical boundary.

## Data Layout, Sharding, And Normalization

Masks are one-dimensional latitude arrays. They are intended to broadcast over
Dinosaur nodal fields with shape `(longitude, latitude)` via `mask[None, :]`.

## Invariants And Tests

Tests should verify:

- bounds `0 <= chi <= 1`,
- exact plateau and zero regions,
- smooth monotone transition,
- sponge rate is zero where `chi=1` and positive where `chi<1`.

## Known Risks

Masking nonlinear tendencies breaks exact conservation. This file only defines
the envelope; diagnostics must determine whether the resulting spectral
ringing/leakage is acceptable.
