#!/usr/bin/env python
"""Per-depth horizontal radial (azimuthally-averaged) kinetic energy spectra
at a single snapshot, with depth-mean overlay.

Outputs a 1x2 figure:
  left:  E_h(k) = 0.5 |k|^2 |psi_hat|^2  (horizontal-flow KE)
  right: E_w(k) = 0.5 |w_hat|^2           (vertical-flow KE)

Per depth: translucent line.
Depth mean: solid line of same color, overlaid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import netCDF4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def radial_bin_per_depth(energy_per_mode: np.ndarray,
                         k_mag: np.ndarray,
                         bins: np.ndarray) -> np.ndarray:
    """Sum per-mode energy into |k| shells, per depth.

    energy_per_mode: (Nz+1, Nx, Nky) -- already includes rfft2 hermitian doubling
    k_mag:           (Nx, Nky)
    bins:            (Nb+1,) shell edges
    returns:         (Nz+1, Nb) shell-integrated energy
    """
    Nz1 = energy_per_mode.shape[0]
    Nb = len(bins) - 1
    out = np.zeros((Nz1, Nb), dtype=np.float64)
    for b in range(Nb):
        mask = (k_mag >= bins[b]) & (k_mag < bins[b + 1])
        out[:, b] = energy_per_mode[:, mask].sum(axis=-1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True, type=str,
                   help="NetCDF snapshot path")
    p.add_argument("--output", required=True, type=str,
                   help="Output PNG path")
    p.add_argument("--title", default=None, type=str)
    args = p.parse_args()

    with netCDF4.Dataset(args.snapshot) as ds:
        Nx = int(ds.Nx)
        L = float(ds.L)
        t = float(ds.time)
        psi = np.array(ds["psi"])  # (z, y, x), float32
        w = np.array(ds["w"])
        z = np.array(ds["z"])

    Nz1, Ny, Nx_check = psi.shape
    assert Nx == Nx_check and Ny == Nx, "expected square horizontal"

    psi_hat = np.fft.rfft2(psi.astype(np.float64), axes=(1, 2))
    w_hat = np.fft.rfft2(w.astype(np.float64), axes=(1, 2))

    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)            # (Nx,)
    ky = 2.0 * np.pi * np.arange(Nx // 2 + 1) / L              # (Nky,)
    k_mag = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)       # (Nx, Nky)

    # rfft2 hermitian: columns 1..Nky-2 represent both +ky and -ky
    Nky = psi_hat.shape[-1]
    herm = np.ones(Nky)
    herm[1:-1] = 2.0

    # Per-mode energies (Parseval-normalized to physical-space energy density)
    norm = 1.0 / (Nx * Ny) ** 2
    e_h = 0.5 * (k_mag ** 2)[None, :, :] * (np.abs(psi_hat) ** 2) * norm
    e_w = 0.5 * (np.abs(w_hat) ** 2) * norm
    e_h *= herm[None, None, :]
    e_w *= herm[None, None, :]

    # Radial shell binning
    k0 = 2.0 * np.pi / L
    k_max = k_mag.max()
    bins = np.arange(0.0, k_max + k0, k0)
    k_centers = 0.5 * (bins[:-1] + bins[1:])

    Eh = radial_bin_per_depth(e_h, k_mag, bins)  # (Nz+1, Nb)
    Ew = radial_bin_per_depth(e_w, k_mag, bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    FLOOR = 1e-30
    color_h = "#1f77b4"
    color_w = "#d62728"

    def _add_slope(ax, k_arr, spec_mean, exponent, label, color):
        """Reference power-law line, anchored to the spectrum mean near a chosen k."""
        # Anchor at a k inside the inertial-ish range (use a few bins above the
        # peak as the visual anchor)
        valid = (spec_mean > FLOOR * 1e6) & (k_arr > 0)
        if not valid.any():
            return
        k_anchor_idx = np.where(valid)[0][len(np.where(valid)[0]) // 4]
        k0_a = k_arr[k_anchor_idx]
        e0_a = spec_mean[k_anchor_idx]
        k_line = k_arr[valid]
        line_y = e0_a * (k_line / k0_a) ** exponent
        ax.loglog(k_line, line_y, color=color, ls="--", lw=1.4, alpha=0.85,
                  label=label)

    # --- Horizontal KE ---
    for zi in range(Nz1):
        ax1.loglog(k_centers, Eh[zi] + FLOOR, color=color_h, alpha=0.12, lw=0.8)
    Eh_mean = Eh.mean(axis=0)
    _add_slope(ax1, k_centers, Eh_mean, -5.0 / 3.0, r"$k^{-5/3}$", "tab:green")
    _add_slope(ax1, k_centers, Eh_mean, -3.0,        r"$k^{-3}$",   "tab:purple")
    ax1.loglog(k_centers, Eh_mean + FLOOR, color="black", lw=2.6, label="depth mean")
    ax1.set_xlabel(r"$k$ (horizontal)")
    ax1.set_ylabel(r"$E_h(k) = \frac{1}{2}|k|^2|\hat{\psi}|^2$")
    ax1.set_title(r"Horizontal kinetic energy spectrum")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(loc="lower left", fontsize=9)

    # --- Vertical KE ---
    for zi in range(Nz1):
        ax2.loglog(k_centers, Ew[zi] + FLOOR, color=color_w, alpha=0.12, lw=0.8)
    Ew_mean = Ew.mean(axis=0)
    _add_slope(ax2, k_centers, Ew_mean, -5.0 / 3.0, r"$k^{-5/3}$", "tab:green")
    _add_slope(ax2, k_centers, Ew_mean, -3.0,        r"$k^{-3}$",   "tab:purple")
    ax2.loglog(k_centers, Ew_mean + FLOOR, color="black", lw=2.6, label="depth mean")
    ax2.set_xlabel(r"$k$ (horizontal)")
    ax2.set_ylabel(r"$E_w(k) = \frac{1}{2}|\hat{w}|^2$")
    ax2.set_title(r"Vertical kinetic energy spectrum")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(loc="lower left", fontsize=9)

    header = args.title or f"{Path(args.snapshot).name}"
    fig.suptitle(f"{header}    t = {t:.3f}    Nx={Nx}  Nz+1={Nz1}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.output, dpi=130)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
