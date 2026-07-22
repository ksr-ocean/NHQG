#!/usr/bin/env python
"""Time-averaged per-depth spectral KE flux over a window of snapshots.

For each snapshot in [t_start, t_end]:
    Pi_h(k, z) = -cumsum_k Re[psi_hat^* * J[psi, zeta]_hat]_shell
    Pi_d(k, z) = -cumsum_k (-Re[psi_hat^* * (d_z w)_hat])_shell
    Pi_t(k, z) = Pi_h + Pi_d

Per-snapshot work runs in parallel (multiprocessing).  Time-average over
the window, then plot per-depth time-averaged Pi (translucent red/blue/green)
with depth-mean overlays in black, exactly as the single-snapshot plot.

Sign convention (see plot_spectral_flux_at_snapshot.py for derivation):
- T_h = +Re[psi^* J_hat]   (rotational advection)
- T_d = -Re[psi^* dwdz_hat] (planetary stretching, since vort eq has +d_z w)
- Pi(K) = -sum_{k<=K} T(k); positive = forward cascade.

Vertical Cheb 2/3-mode filter applied to w before differentiating.
"""

from __future__ import annotations

import argparse
import re
import time as _time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import netCDF4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke


def cheb_diff_matrix(N: int, x: np.ndarray) -> np.ndarray:
    if N == 0:
        return np.zeros((1, 1))
    c = np.ones(N + 1); c[0] = 2.0; c[N] = 2.0
    sign = np.array([(-1.0) ** i for i in range(N + 1)])
    D = np.zeros((N + 1, N + 1), dtype=np.float64)
    for j in range(N + 1):
        for k in range(N + 1):
            if j != k:
                D[j, k] = (c[j] / c[k]) * sign[j] * sign[k] / (x[j] - x[k])
    D[np.arange(N + 1), np.arange(N + 1)] = -D.sum(axis=1)
    return D


# ---- Per-worker shared precomputes (filled in initializer) ----
_W = {}


def _worker_init(Nx: int, L: float, Nz: int, z_arr: np.ndarray):
    """Build per-worker precomputes once: Cheb deriv, vert filter, k-grid."""
    xi = 2.0 * z_arr - 1.0
    Dz = 2.0 * cheb_diff_matrix(Nz, xi)

    # Vertical 2/3 Cheb filter on w
    n = np.arange(Nz + 1); j = np.arange(Nz + 1)
    cn = np.ones(Nz + 1); cn[0] = 2.0; cn[Nz] = 2.0
    V_inv = (2.0 / (Nz * cn[:, None] * cn[None, :])) * np.cos(
        n[:, None] * np.pi * j[None, :] / Nz
    )
    V = np.cos(j[:, None] * np.pi * n[None, :] / Nz)
    n_cut = (2 * Nz) // 3
    cmask = np.zeros(Nz + 1); cmask[: n_cut + 1] = 1.0
    Filt_w = V @ np.diag(cmask) @ V_inv  # nodal filter operator

    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    ky = 2.0 * np.pi * np.arange(Nx // 2 + 1) / L
    k_mag = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    Nky = Nx // 2 + 1
    herm = np.ones(Nky); herm[1:-1] = 2.0
    k0 = 2.0 * np.pi / L
    bins = np.arange(0.0, k_mag.max() + k0, k0)
    # Precompute shell index for each (kx,ky) mode
    # bin index = floor(k_mag / k0)
    bin_idx = np.minimum((k_mag / k0).astype(np.int64), len(bins) - 2)

    _W["Nx"] = Nx; _W["Ny"] = Nx; _W["Nky"] = Nky
    _W["L"] = L; _W["Nz"] = Nz
    _W["Dz"] = Dz; _W["Filt_w"] = Filt_w
    _W["kx"] = kx; _W["ky"] = ky; _W["k_mag"] = k_mag
    _W["herm"] = herm
    _W["bins"] = bins; _W["bin_idx"] = bin_idx
    _W["norm"] = 1.0 / (Nx * Nx) ** 2


def _process_snapshot(path: str):
    """Compute per-depth Pi_h(k), Pi_d(k) for one snapshot."""
    Nx = _W["Nx"]; Ny = _W["Ny"]; Nky = _W["Nky"]
    Dz = _W["Dz"]; Filt_w = _W["Filt_w"]
    kx = _W["kx"]; ky = _W["ky"]; k_mag = _W["k_mag"]
    herm = _W["herm"]; bins = _W["bins"]; bin_idx = _W["bin_idx"]
    norm = _W["norm"]

    with netCDF4.Dataset(path) as ds:
        psi = np.asarray(ds["psi"]).astype(np.float64)
        w = np.asarray(ds["w"]).astype(np.float64)
        t = float(ds.time)
    Nz1 = psi.shape[0]

    # vertical Cheb 2/3 filter on w then differentiate
    w_filt = np.einsum("ij,jyx->iyx", Filt_w, w)
    dwdz = np.einsum("ij,jyx->iyx", Dz, w_filt)

    psi_hat = np.fft.rfft2(psi, axes=(1, 2))
    zeta_hat = -(k_mag[None, :, :] ** 2) * psi_hat

    # J[psi, zeta] in physical
    psi_x = np.fft.irfft2(1j * kx[None, :, None] * psi_hat, s=(Ny, Nx), axes=(1, 2))
    psi_y = np.fft.irfft2(1j * ky[None, None, :] * psi_hat, s=(Ny, Nx), axes=(1, 2))
    zet_x = np.fft.irfft2(1j * kx[None, :, None] * zeta_hat, s=(Ny, Nx), axes=(1, 2))
    zet_y = np.fft.irfft2(1j * ky[None, None, :] * zeta_hat, s=(Ny, Nx), axes=(1, 2))
    J_phys = psi_x * zet_y - psi_y * zet_x
    J_hat = np.fft.rfft2(J_phys, axes=(1, 2))

    dwdz_hat = np.fft.rfft2(dwdz, axes=(1, 2))

    T_h = +np.real(np.conj(psi_hat) * J_hat) * herm[None, None, :] * norm
    T_d = -np.real(np.conj(psi_hat) * dwdz_hat) * herm[None, None, :] * norm

    # Shell-bin via np.bincount per depth (faster than masking loop)
    Nb = len(bins) - 1
    bidx_flat = bin_idx.ravel()
    T_h_shell = np.zeros((Nz1, Nb))
    T_d_shell = np.zeros((Nz1, Nb))
    for zi in range(Nz1):
        T_h_shell[zi] = np.bincount(bidx_flat, weights=T_h[zi].ravel(), minlength=Nb)[:Nb]
        T_d_shell[zi] = np.bincount(bidx_flat, weights=T_d[zi].ravel(), minlength=Nb)[:Nb]

    Pi_h = -np.cumsum(T_h_shell, axis=1)
    Pi_d = -np.cumsum(T_d_shell, axis=1)
    return t, Pi_h, Pi_d


def _list_snapshots(snapshot_dir: Path, t_start: float, t_end: float, dt_step: float):
    """Find snapshot files in [t_start, t_end].  step number = round(t/dt_step)."""
    pat = re.compile(r"snapshot_(\d{8})\.nc$")
    out = []
    for f in sorted(snapshot_dir.glob("snapshot_*.nc")):
        m = pat.search(f.name)
        if not m:
            continue
        step = int(m.group(1))
        t = step * dt_step
        if t_start - 1e-9 <= t <= t_end + 1e-9:
            out.append((t, str(f)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-dir", required=True, type=str)
    p.add_argument("--t-start", required=True, type=float)
    p.add_argument("--t-end", required=True, type=float)
    p.add_argument("--dt", type=float, default=5e-5,
                   help="Time-step (used to map snapshot step number to t)")
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    snap_dir = Path(args.snapshot_dir)
    files = _list_snapshots(snap_dir, args.t_start, args.t_end, args.dt)
    if not files:
        raise SystemExit(f"No snapshots found in {snap_dir} for t in [{args.t_start}, {args.t_end}]")
    print(f"Snapshots: {len(files)} files in t in [{files[0][0]:.2f}, {files[-1][0]:.2f}]")

    # Read first snapshot for grid metadata
    with netCDF4.Dataset(files[0][1]) as ds:
        Nx = int(ds.Nx); L = float(ds.L)
        z = np.asarray(ds["z"]).astype(np.float64)
    Nz = len(z) - 1

    paths = [f for _, f in files]

    t0 = _time.time()
    with Pool(args.workers, initializer=_worker_init, initargs=(Nx, L, Nz, z)) as pool:
        results = pool.imap_unordered(_process_snapshot, paths, chunksize=4)
        Pi_h_sum = None
        Pi_d_sum = None
        n_done = 0
        for t, Pi_h, Pi_d in results:
            if Pi_h_sum is None:
                Pi_h_sum = np.zeros_like(Pi_h)
                Pi_d_sum = np.zeros_like(Pi_d)
            Pi_h_sum += Pi_h
            Pi_d_sum += Pi_d
            n_done += 1
            if n_done % max(1, len(paths) // 20) == 0:
                elapsed = _time.time() - t0
                eta = elapsed * (len(paths) - n_done) / max(n_done, 1)
                print(f"  {n_done}/{len(paths)} done  elapsed={elapsed:.1f}s  ETA={eta:.1f}s")
    Pi_h_avg = Pi_h_sum / n_done
    Pi_d_avg = Pi_d_sum / n_done
    Pi_t_avg = Pi_h_avg + Pi_d_avg
    print(f"Time-averaged over {n_done} snapshots in {_time.time() - t0:.1f}s")

    # k bin centers
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    ky = 2.0 * np.pi * np.arange(Nx // 2 + 1) / L
    k_mag = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    k0 = 2.0 * np.pi / L
    bins = np.arange(0.0, k_mag.max() + k0, k0)
    k_centers = 0.5 * (bins[:-1] + bins[1:])

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    color_h = "#d62728"
    color_d = "#1f77b4"
    color_t = "#2ca02c"

    Nz1 = Pi_h_avg.shape[0]
    for zi in range(Nz1):
        ax.semilogx(k_centers, Pi_h_avg[zi], color=color_h, alpha=0.07, lw=0.6)
        ax.semilogx(k_centers, Pi_d_avg[zi], color=color_d, alpha=0.07, lw=0.6)
        ax.semilogx(k_centers, Pi_t_avg[zi], color=color_t, alpha=0.07, lw=0.6)

    Pi_h_dm = Pi_h_avg.mean(axis=0)
    Pi_d_dm = Pi_d_avg.mean(axis=0)
    Pi_t_dm = Pi_t_avg.mean(axis=0)

    pe = [withStroke(linewidth=4.5, foreground="white")]
    ax.semilogx(k_centers, Pi_h_dm, color="black", lw=2.6, ls="-",  path_effects=pe,
                label=r"$\langle\Pi_h\rangle_{z,t}$ (red, rotational)")
    ax.semilogx(k_centers, Pi_d_dm, color="black", lw=2.6, ls="--", path_effects=pe,
                label=r"$\langle\Pi_d\rangle_{z,t}$ (blue, divergent)")
    ax.semilogx(k_centers, Pi_t_dm, color="black", lw=2.8, ls=":",  path_effects=pe,
                label=r"$\langle\Pi_h + \Pi_d\rangle_{z,t}$ (green, total)")

    ax.axhline(0.0, color="gray", lw=0.6, alpha=0.7)
    ax.set_xlabel(r"$k$ (horizontal)")
    ax.set_ylabel(r"$\langle\Pi(k)\rangle_t$ -- time-averaged spectral KE flux")
    title_extra = (args.title or snap_dir.name) + \
        f"   t in [{args.t_start:.1f}, {args.t_end:.1f}]   " \
        f"({n_done} snaps)"
    ax.set_title("Time-averaged spectral KE flux: rotational vs divergent")
    fig.suptitle(title_extra, fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    mean_scale = max(np.abs(Pi_h_dm).max(), np.abs(Pi_d_dm).max(), np.abs(Pi_t_dm).max())
    if mean_scale > 0:
        ax.set_ylim(-5.0 * mean_scale, 5.0 * mean_scale)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=130)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
