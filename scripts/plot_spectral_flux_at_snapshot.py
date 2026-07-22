#!/usr/bin/env python
"""Per-depth horizontal spectral kinetic-energy flux + depth-mean overlay.

For one snapshot:
  Pi_h(k) -- rotational-advection cumulative flux from
             T_h(k) = +Re[psi_hat^* * J[psi, zeta]_hat]
  Pi_d(k) -- divergent (planetary-vortex-stretching) cumulative flux from
             T_d(k) = -Re[psi_hat^* * (d_z w)_hat]
  Pi_tot(k) = Pi_h(k) + Pi_d(k)

Sign derivation.  In NHQGE/QG the vorticity equation reads
    d_t zeta + J[psi, zeta] = +d_z w + ...    (planetary stretching f=1)
Multiplying by -psi and integrating:
    d_t (1/2 |grad psi|^2) = +psi * J[psi, zeta] - psi * d_z w + ...
yielding the per-mode contributions T_h (positive) and T_d (negative)
above.  By continuity delta_h = -d_z w, so the "divergence term" is exactly
-d_z w; the full minus sign in T_d combines (-psi)*(+d_z w) = -psi*d_z w.

Convention: Pi(K) = -sum_{k_bin <= K} T_shell(k_bin) (Frisch / Kraichnan).
Positive Pi(K) = forward cascade past K; negative = inverse cascade past K.

Plot: 1 panel, log x; per-depth Pi_h (red), Pi_d (blue), sum (green) translucent;
depth-mean of each in black, plotted last so they sit on top.

Vorticity convention: zeta = laplacian_h(psi), so zeta_hat = -|k|^2 * psi_hat.
The vertical derivative d_z w is computed via the standard Chebyshev
collocation D matrix on the snapshot's CGL grid.  In late-time snapshots
the saved physical w can carry spurious high Chebyshev modes that the
Chebyshev derivative amplifies as ~n^2; we apply a 2/3 vertical Cheb
truncation filter to w prior to differentiation to suppress that artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import netCDF4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cheb_diff_matrix(N: int, x: np.ndarray) -> np.ndarray:
    """First-derivative collocation matrix on N+1 nodes x = cos(j*pi/N) on [-1,1].

    Trefethen "Spectral Methods in MATLAB" Ch. 6.
    Returns D such that (D @ f) approximates f'(x) at each node.
    """
    if N == 0:
        return np.zeros((1, 1))
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0
    sign = np.array([(-1.0) ** i for i in range(N + 1)])

    D = np.zeros((N + 1, N + 1), dtype=np.float64)
    for j in range(N + 1):
        for k in range(N + 1):
            if j != k:
                D[j, k] = (c[j] / c[k]) * sign[j] * sign[k] / (x[j] - x[k])
    # Diagonal from negative row-sum so that D acts as zero on constants
    D[np.arange(N + 1), np.arange(N + 1)] = -D.sum(axis=1)
    return D


def shell_bin(values_per_mode: np.ndarray,
              k_mag: np.ndarray,
              bins: np.ndarray) -> np.ndarray:
    """Sum per-mode quantity over |k| shells. values_per_mode shape (Nz+1, Nx, Nky)."""
    Nz1 = values_per_mode.shape[0]
    Nb = len(bins) - 1
    out = np.zeros((Nz1, Nb), dtype=np.float64)
    for b in range(Nb):
        mask = (k_mag >= bins[b]) & (k_mag < bins[b + 1])
        out[:, b] = values_per_mode[:, mask].sum(axis=-1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--title", default=None, type=str)
    args = p.parse_args()

    with netCDF4.Dataset(args.snapshot) as ds:
        Nx = int(ds.Nx)
        L = float(ds.L)
        t = float(ds.time)
        psi = np.asarray(ds["psi"]).astype(np.float64)   # (z, y, x)
        w = np.asarray(ds["w"]).astype(np.float64)
        z = np.asarray(ds["z"]).astype(np.float64)        # CGL nodes on [0, 1]

    Nz1, Ny, Nx_check = psi.shape
    assert Nx == Nx_check and Ny == Nx, "expected square horizontal"
    Nz = Nz1 - 1

    # ---- Vertical derivative: build collocation D on the snapshot's CGL grid ----
    # The snapshot's z is on [0, 1] with z[0]=1, z[N]=0 (descending), i.e.
    # z[j] = (1 + cos(j*pi/N))/2.  Map to xi = 2z - 1 = cos(j*pi/N) (the standard
    # CGL ordering required by the cheb_diff_matrix sign formula).  Then
    # dxi/dz = +2, so d/dz = +2 * d/dxi.
    xi = 2.0 * z - 1.0
    Dxi = cheb_diff_matrix(Nz, xi)
    Dz = 2.0 * Dxi

    # ---- Vertical 2/3-mode filter on w before differentiating ----
    # In late-time snapshots the saved physical w can carry spurious high
    # Chebyshev modes (vertical aliasing not removed by the horizontal 2/3
    # rule).  Differentiating amplifies these by ~n^2.  Truncate Cheb modes
    # n > 2N/3 to suppress that artifact.
    # Build V_inv (analytic DCT-I) and V (Chebyshev evaluation) for filtering.
    n_grid = np.arange(Nz + 1)
    j_grid = np.arange(Nz + 1)
    cn = np.ones(Nz + 1); cn[0] = 2.0; cn[Nz] = 2.0
    V_inv = (2.0 / (Nz * cn[:, None] * cn[None, :])) * np.cos(
        n_grid[:, None] * np.pi * j_grid[None, :] / Nz
    )
    V = np.cos(j_grid[:, None] * np.pi * n_grid[None, :] / Nz)  # T_n(xi_j)
    n_cut = (2 * Nz) // 3
    cheb_mask = np.zeros(Nz + 1)
    cheb_mask[: n_cut + 1] = 1.0
    # nodal -> coeffs -> mask -> nodal
    w_cheb = np.einsum("nj,jyx->nyx", V_inv, w)
    w_cheb *= cheb_mask[:, None, None]
    w_filt = np.einsum("jn,nyx->jyx", V, w_cheb)
    # Note: snapshot ordering (z descending) matches V indices via xi = cos(j*pi/N)
    # so V_inv reads from snapshot order and V writes to snapshot order.
    w = w_filt

    # dw/dz at every (z, y, x); apply along axis 0
    dwdz = np.einsum("ij,jyx->iyx", Dz, w)

    # ---- Horizontal Fourier ----
    psi_hat = np.fft.rfft2(psi, axes=(1, 2))     # (Nz+1, Ny, Nky)
    w_hat = np.fft.rfft2(w, axes=(1, 2))

    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    ky = 2.0 * np.pi * np.arange(Nx // 2 + 1) / L
    k_mag = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    Nky = psi_hat.shape[-1]

    # rfft2 hermitian doubling (account for unstored negative-ky modes)
    herm = np.ones(Nky)
    herm[1:-1] = 2.0
    herm_b = herm[None, None, :]

    # Vorticity in spectral: zeta_hat = -|k|^2 psi_hat
    zeta_hat = -(k_mag[None, :, :] ** 2) * psi_hat

    # ---- Build J[psi, zeta] in physical, then back to spectral ----
    psi_x = np.fft.irfft2(1j * kx[None, :, None] * psi_hat, s=(Ny, Nx), axes=(1, 2))
    psi_y = np.fft.irfft2(1j * ky[None, None, :] * psi_hat, s=(Ny, Nx), axes=(1, 2))
    zet_x = np.fft.irfft2(1j * kx[None, :, None] * zeta_hat, s=(Ny, Nx), axes=(1, 2))
    zet_y = np.fft.irfft2(1j * ky[None, None, :] * zeta_hat, s=(Ny, Nx), axes=(1, 2))
    J_phys = psi_x * zet_y - psi_y * zet_x
    J_hat = np.fft.rfft2(J_phys, axes=(1, 2))

    # ---- d_z w in spectral (it is real, so just rfft2 it) ----
    dwdz_hat = np.fft.rfft2(dwdz, axes=(1, 2))

    # ---- Per-mode KE tendency contributions ----
    # Note FFT normalization: for Parseval-consistent values, divide by (Nx*Ny)^2
    # T_h: +Re[psi^* * J_hat]   (rotational advection)
    # T_d: -Re[psi^* * dwdz_hat] (planetary vortex stretching)
    norm = 1.0 / (Nx * Ny) ** 2
    T_h_mode = +np.real(np.conj(psi_hat) * J_hat) * herm_b * norm
    T_d_mode = -np.real(np.conj(psi_hat) * dwdz_hat) * herm_b * norm

    # ---- Shell binning ----
    k0 = 2.0 * np.pi / L
    k_max = k_mag.max()
    bins = np.arange(0.0, k_max + k0, k0)
    k_centers = 0.5 * (bins[:-1] + bins[1:])

    T_h_shell = shell_bin(T_h_mode, k_mag, bins)   # (Nz+1, Nb)
    T_d_shell = shell_bin(T_d_mode, k_mag, bins)

    # ---- Cumulative spectral flux ----
    # Pi(K) = -sum_{k_bin <= K} T_shell(k_bin)  (Frisch convention)
    Pi_h = -np.cumsum(T_h_shell, axis=1)
    Pi_d = -np.cumsum(T_d_shell, axis=1)
    Pi_tot = Pi_h + Pi_d

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10.0, 6.5))

    color_h = "#d62728"   # red
    color_d = "#1f77b4"   # blue
    color_t = "#2ca02c"   # green

    for zi in range(Nz1):
        ax.semilogx(k_centers, Pi_h[zi], color=color_h, alpha=0.06, lw=0.6)
        ax.semilogx(k_centers, Pi_d[zi], color=color_d, alpha=0.06, lw=0.6)
        ax.semilogx(k_centers, Pi_tot[zi], color=color_t, alpha=0.06, lw=0.6)

    Pi_h_mean = Pi_h.mean(axis=0)
    Pi_d_mean = Pi_d.mean(axis=0)
    Pi_tot_mean = Pi_tot.mean(axis=0)

    # Means: thick black with a thin white edge for contrast against the
    # translucent spread; plotted last so they sit on top.
    from matplotlib.patheffects import withStroke
    pe = [withStroke(linewidth=4.5, foreground="white")]
    ax.semilogx(k_centers, Pi_h_mean,   color="black", lw=2.6, ls="-",
                path_effects=pe,
                label=r"$\langle\Pi_h\rangle_z$ (red, rotational)")
    ax.semilogx(k_centers, Pi_d_mean,   color="black", lw=2.6, ls="--",
                path_effects=pe,
                label=r"$\langle\Pi_d\rangle_z$ (blue, divergent)")
    ax.semilogx(k_centers, Pi_tot_mean, color="black", lw=2.8, ls=":",
                path_effects=pe,
                label=r"$\langle\Pi_h + \Pi_d\rangle_z$ (green, total)")

    ax.axhline(0.0, color="gray", lw=0.6, alpha=0.7)
    ax.set_xlabel(r"$k$ (horizontal)")
    ax.set_ylabel(r"$\Pi(k)$ -- cumulative spectral KE flux")
    ax.set_title("Spectral KE flux: rotational vs divergent contributions")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # Auto-scale y to show the depth means clearly: 5x the largest mean
    # magnitude.  Per-depth extreme excursions can extend several times
    # outside this and will be visible (translucent) but not dominate.
    mean_scale = max(
        np.abs(Pi_h_mean).max(),
        np.abs(Pi_d_mean).max(),
        np.abs(Pi_tot_mean).max(),
    )
    if mean_scale > 0:
        ax.set_ylim(-5.0 * mean_scale, 5.0 * mean_scale)

    header = args.title or Path(args.snapshot).name
    fig.suptitle(f"{header}    t = {t:.3f}    Nx={Nx}  Nz+1={Nz1}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=130)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
