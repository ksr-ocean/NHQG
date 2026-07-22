# dinosaur_spike/make_vorticity_sphere_movie.py

## Responsibility

This script renders exact upper-layer relative vorticity from modal
`state_step_*.npz` checkpoints as a grayscale orthographic sphere movie.

## Numerical Meaning

For each checkpoint, the script loads modal perturbation PV, inverts the
two-layer QG relation to recover `psi1`, and computes
`zeta1 = laplacian(psi1)`. This is the correct upper-layer relative vorticity
for the current two-layer model, unlike movies derived from saved PNG colors.

The rendered field is normalized independently per frame by a high percentile
for visualization. That is useful for morphology but not for comparing absolute
amplitudes; diagnostics CSV remains the quantitative source.

## Device Use

The default device is CPU so movie rendering does not interfere with an active
GPU run. The spherical-harmonic implementation must match the checkpoint modal
layout; GPU production runs currently use Dinosaur's `fast` layout.

## Outputs

The output is an animated GIF with a black-white signed colorbar. MP4 output is
not the default because ffmpeg/imageio-ffmpeg is not available in the current
environment.

## Known Limits

The script assumes the checkpoint grid shape matches `--wavenumbers` or the
checkpoint metadata. It also assumes the current default two-layer inversion
parameters `F1=0.7`, `F2=0.4` unless explicitly overridden.
