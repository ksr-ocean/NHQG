#!/usr/bin/env python
"""Horizontally upsample an NHQG checkpoint in Fourier (rfft2) space.

Zero-pads kx and ky of q_hat, w_hat, th_hat from Nx_old -> Nx_new while
keeping Nz and t unchanged. Implements spectral (sinc) upsampling by:

  1. irfft2 -> physical field on the old (Nx_old, Nx_old) grid
  2. fft2 -> full-FFT spectrum, fftshift to center DC
  3. halve Nyquist rows/cols and duplicate them at the positive-Nyquist
     side of an extended centered array (the standard sinc trick)
  4. zero-pad into (Nx_new, Nx_new) centered
  5. ifftshift and scale by (Nx_new/Nx_old)^2 for jax/numpy unnormalized
     forward-FFT convention
  6. keep the non-negative ky half for rfft2 form

This preserves the physical field exactly at co-located grid points.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _upsample_horiz_block(X_rfft_old: np.ndarray, Nx_new: int) -> np.ndarray:
    """Spectral upsample a (Nv, Nx_old, Nx_old//2+1) rfft2 block to
    (Nv, Nx_new, Nx_new//2+1). Preserves physical values at co-located
    grid points."""
    Nv, Nx_old, Nky_old = X_rfft_old.shape
    assert Nky_old == Nx_old // 2 + 1
    assert Nx_new >= Nx_old
    assert (Nx_new - Nx_old) % 2 == 0

    # to physical, then to full fft
    phys_old = np.fft.irfft2(X_rfft_old, s=(Nx_old, Nx_old), axes=(1, 2))
    X_full = np.fft.fft2(phys_old, axes=(1, 2))
    X_shift = np.fft.fftshift(X_full, axes=(1, 2))  # DC at (Nx_old/2, Nx_old/2)

    # extend by one row/col: the old Nyquist (shifted index 0) gets
    # halved and duplicated at both shifted index 0 and shifted index Nx_old
    ext = np.zeros((Nv, Nx_old + 1, Nx_old + 1), dtype=X_shift.dtype)
    ext[:, :Nx_old, :Nx_old] = X_shift
    ext[:, 0, :Nx_old] *= 0.5
    ext[:, Nx_old, :Nx_old] = ext[:, 0, :Nx_old]
    # careful ordering: need to operate on the full Nx_old+1 cols now
    ext[:, :, 0] *= 0.5
    ext[:, :, Nx_old] = ext[:, :, 0]

    # zero-pad centered into Nx_new x Nx_new
    pad = (Nx_new - Nx_old) // 2
    X_new_shift = np.zeros((Nv, Nx_new, Nx_new), dtype=X_shift.dtype)
    X_new_shift[:, pad:pad + Nx_old + 1, pad:pad + Nx_old + 1] = ext

    X_new_full = np.fft.ifftshift(X_new_shift, axes=(1, 2)) * (Nx_new / Nx_old) ** 2
    # truncate to rfft2 form: keep ky in [0, Nx_new//2]
    X_new_rfft = X_new_full[:, :, : Nx_new // 2 + 1]
    return X_new_rfft


def _selftest() -> None:
    rng = np.random.default_rng(0)
    for (Nx_old, Nx_new) in [(16, 32), (8, 16), (16, 64)]:
        Nv = 3
        x_old = rng.standard_normal((Nv, Nx_old, Nx_old))
        X_old = np.fft.rfft2(x_old, axes=(1, 2))
        X_new = _upsample_horiz_block(X_old, Nx_new)
        x_new = np.fft.irfft2(X_new, s=(Nx_new, Nx_new), axes=(1, 2))
        step = Nx_new // Nx_old
        err = np.max(np.abs(x_new[:, ::step, ::step] - x_old))
        assert err < 1e-10, f"mismatch Nx_old={Nx_old} Nx_new={Nx_new}: {err:.3e}"
        # also check max(|x_new|) stays in a reasonable range (no blow-up)
        print(f"[selftest] {Nx_old}->{Nx_new}: co-located max-abs err = {err:.3e}, "
              f"max|x_new|={np.max(np.abs(x_new)):.3f} (old max={np.max(np.abs(x_old)):.3f})")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=str, required=True)
    p.add_argument("--out", dest="out", type=str, required=True)
    p.add_argument("--Nx-new", type=int, required=True)
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.selftest:
        _selftest()

    data = np.load(args.inp)
    fields_3d = ["q_hat_real", "q_hat_imag", "w_hat_real", "w_hat_imag",
                 "th_hat_real", "th_hat_imag"]
    Nx_old = int(data["q_hat_real"].shape[1])
    Nx_new = args.Nx_new
    print(f"Upsampling {args.inp}")
    print(f"  Nx: {Nx_old} -> {Nx_new}  (scale factor = {(Nx_new/Nx_old)**2})")

    # Real and imag need to be combined before transforming: irfft2 of
    # just the real part is not well-defined. Reconstruct complex arrays.
    def combine(name_r: str, name_i: str) -> np.ndarray:
        return data[name_r] + 1j * data[name_i]

    def split(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return arr.real.copy(), arr.imag.copy()

    out = {}
    for base in ["q_hat", "w_hat", "th_hat"]:
        X = combine(f"{base}_real", f"{base}_imag")
        X_up = _upsample_horiz_block(X, Nx_new)
        r, i = split(X_up)
        out[f"{base}_real"] = r
        out[f"{base}_imag"] = i
        print(f"  {base}: {X.shape} -> {X_up.shape}")

    out["th_bar"] = data["th_bar"]
    out["step"] = data["step"]
    out["t"] = data["t"]
    print(f"  th_bar: {data['th_bar'].shape} (passthrough)")
    print(f"  step = {int(data['step'])}  t = {float(data['t'])}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
