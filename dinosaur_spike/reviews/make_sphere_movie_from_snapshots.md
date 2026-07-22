# dinosaur_spike/make_sphere_movie_from_snapshots.py

## Responsibility

This script converts the saved two-panel `pv_step_*.png` snapshots from a
solution run into a grayscale spherical movie of the upper panel.

## Numerical Meaning

The current long-run snapshots are PNG images, not modal state checkpoints.
Therefore this script cannot reconstruct exact upper-layer relative vorticity
from the two-layer inversion. It extracts the upper-layer plotted field from
the snapshot image and maps its red-blue contrast to a signed grayscale scalar.

For exact vorticity movies, future runs should save modal or nodal checkpoints
at each frame so that `laplacian(psi1)` can be rendered directly.

## Rendering Method

The script crops the upper data panel, interprets it as a longitude-latitude
texture, projects it onto an orthographic sphere centered in the southern
hemisphere, and writes an MP4 with a black-white colorbar. The sphere longitude
view can rotate slowly across the sequence.

## Known Limits

The crop coordinates depend on the existing snapshot layout. They are CLI
arguments so a different figure layout can be handled without editing code. The
black-white field is a signed visual proxy from the PNG colormap, not a
conservative or analysis-grade dataset.
