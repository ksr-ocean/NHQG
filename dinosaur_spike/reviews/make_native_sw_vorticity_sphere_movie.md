# dinosaur_spike/make_native_sw_vorticity_sphere_movie.py

## Responsibility

This script renders relative vorticity from Dinosaur's native shallow-water
modal checkpoints as an orthographic grayscale sphere movie. It is separate from
the QG movie renderer because the checkpoint fields are named and interpreted
differently.

## Numerical Meaning

Native shallow-water checkpoints store `vorticity`, `divergence`, and
`potential` directly. No PV inversion is performed here. The selected layer's
modal vorticity is transformed to nodal space with the same Dinosaur spherical
harmonic grid used for the run, then plotted as the sphere texture.

Layer `0` is the upper layer and layer `1` is the lower layer. The default movie
uses the upper layer because that is the field we have been inspecting in the QG
experiments.

## Orientation

Dinosaur nodal arrays are longitude-major, and the latitude axis is ordered from
south to north. The sphere renderer expects a texture whose first row is the
northern edge, so the script transposes the field and flips it vertically before
sampling the sphere.

## Normalization

The default `global` normalization uses the largest requested percentile scale
over all frames. This preserves visible amplitude changes over time better than
per-frame normalization. The `per-frame` option is still available when the goal
is to inspect morphology at each checkpoint.

## Device Use

Rendering defaults to CPU. That keeps the GPU free for solver runs and avoids
JAX device-memory contention during visualization.
