#!/usr/bin/env python
"""Time-averaged per-depth horizontal radial KE spectra over a snapshot window.

Same 1x2 layout as plot_horiz_spectra_at_snapshot.py:
  left:  E_h(k,z) = 0.5 |k|^2 |psi_hat|^2  (rotational/horizontal KE)
  right: E_w(k,z) = 0.5 |w_hat|^2           (vertical KE)

For each snapshot in [t_start, t_end], compute Eh and Ew per depth.
Time-average across snapshots.  Plot per-depth time-averaged spectrum
translucent + depth-mean of time average in solid black, with k^{-5/3}
and k^{-3} reference slopes.

Per-snapshot work runs in parallel (multiprocessing).
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


# ---- Per-worker shared precomputes ----
_W = {}


def _worker_init(Nx: int, L: float, Nz: int):
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    ky = 2.0 * np.pi * np.arange(Nx // 2 + 1) / L
    k_mag = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    Nky = Nx // 2 + 1
    herm = np.ones(Nky); herm[1:-1] = 2.0
    k0 = 2.0 * np.pi / L
    bins = np.arange(0.0, k_mag.max() + k0, k0)
    bin_idx = np.minimum((k_mag / k0).astype(np.int64), len(bins) - 2)

    _W["Nx"] = Nx; _W["Ny"] = Nx; _W["Nky"] = Nky
    _W["L"] = L; _W["Nz"] = Nz
    _W["k_mag"] = k_mag
    _W["herm"] = herm
    _W["bins"] = bins; _W["bin_idx"] = bin_idx
    _W["norm"] = 1.0 / (Nx * Nx) ** 2


def _process_snapshot(path: str):
    Nx = _W["Nx"]; Nky = _W["Nky"]
    k_mag = _W["k_mag"]
    herm = _W["herm"]; bins = _W["bins"]; bin_idx = _W["bin_idx"]
    norm = _W["norm"]

    with netCDF4.Dataset(path) as ds:
        psi = np.asarray(ds["psi"]).astype(np.float64)
        w = np.asarray(ds["w"]).astype(np.float64)
    Nz1 = psi.shape[0]

    psi_hat = np.fft.rfft2(psi, axes=(1, 2))
    w_hat = np.fft.rfft2(w, axes=(1, 2))

    e_h = 0.5 * (k_mag ** 2)[None, :, :] * (np.abs(psi_hat) ** 2) * norm
    e_w = 0.5 * (np.abs(w_hat) ** 2) * norm
    e_h *= herm[None, None, :]
    e_w *= herm[None, None, :]

    Nb = len(bins) - 1
    bidx_flat = bin_idx.ravel()
    Eh = np.zeros((Nz1, Nb))
    Ew = np.zeros((Nz1, Nb))
    for zi in range(Nz1):
        Eh[zi] = np.bincount(bidx_flat, weights=e_h[zi].ravel(), minlength=Nb)[:Nb]
        Ew[zi] = np.bincount(bidx_flat, weights=e_w[zi].ravel(), minlength=Nb)[:Nb]
    return Eh, Ew


def _list_snapshots(snapshot_dir: Path, t_start: float, t_end: float, dt_step: float):
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
    p.add_argument("--dt", type=float, default=5e-5)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--title", default=None, type=str)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    snap_dir = Path(args.snapshot_dir)
    files = _list_snapshots(snap_dir, args.t_start, args.t_end, args.dt)
    if not files:
        raise SystemExit(f"No snapshots in {snap_dir} for t in [{args.t_start}, {args.t_end}]")
    print(f"Snapshots: {len(files)} files in t in [{files[0][0]:.2f}, {files[-1][0]:.2f}]")

    with netCDF4.Dataset(files[0][1]) as ds:
        Nx = int(ds.Nx); L = float(ds.L)
        z = np.asarray(ds["z"]).astype(np.float64)
    Nz = len(z) - 1

    paths = [f for _, f in files]

    t0 = _time.time()
    with Pool(args.workers, initializer=_worker_init, initargs=(Nx, L, Nz)) as pool:
        results = pool.imap_unordered(_process_snapshot, paths, chunksize=4)
        Eh_sum = None; Ew_sum = None
        n = 0
        for Eh, Ew in results:
            if Eh_sum is None:
                Eh_sum = np.zeros_like(Eh)
                Ew_sum = np.zeros_like(Ew)
            Eh_sum += Eh
            Ew_sum += Ew
            n += 1
            if n % max(1, len(paths) // 20) == 0:
                el = _time.time() - t0
                eta = el * (len(paths) - n) / max(n, 1)
                print(f"  {n}/{len(paths)} done elapsed={el:.1f}s ETA={eta:.1f}s")
    Eh_avg = Eh_sum / n
    Ew_avg = Ew_sum / n
    print(f"Time-averaged over {n} snapshots in {_time.time() - t0:.1f}s")

    # k bin centers
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    ky = 2.0 * np.pi * np.arange(Nx // 2 + 1) / L
    k_mag = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    k0 = 2.0 * np.pi / L
    bins = np.arange(0.0, k_mag.max() + k0, k0)
    k_centers = 0.5 * (bins[:-1] + bins[1:])

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    FLOOR = 1e-30
    color_h = "#1f77b4"
    color_w = "#d62728"
    Nz1 = Eh_avg.shape[0]

    def _add_slope(ax, k_arr, spec_mean, exponent, label, color):
        valid = (spec_mean > FLOOR * 1e6) & (k_arr > 0)
        if not valid.any():
            return
        idx = np.where(valid)[0][len(np.where(valid)[0]) // 4]
        k0_a = k_arr[idx]; e0_a = spec_mean[idx]
        kl = k_arr[valid]
        ly = e0_a * (kl / k0_a) ** exponent
        ax.loglog(kl, ly, color=color, ls="--", lw=1.4, alpha=0.85, label=label)

    # Horizontal KE
    for zi in range(Nz1):
        ax1.loglog(k_centers, Eh_avg[zi] + FLOOR, color=color_h, alpha=0.12, lw=0.8)
    Eh_dm = Eh_avg.mean(axis=0)
    _add_slope(ax1, k_centers, Eh_dm, -5.0/3.0, r"$k^{-5/3}$", "tab:green")
    _add_slope(ax1, k_centers, Eh_dm, -3.0,     r"$k^{-3}$",   "tab:purple")
    ax1.loglog(k_centers, Eh_dm + FLOOR, color="black", lw=2.6,
               label=r"$\langle E_h \rangle_{z,t}$")
    ax1.set_xlabel(r"$k$ (horizontal)")
    ax1.set_ylabel(r"$\langle E_h(k)\rangle_t = \frac{1}{2}|k|^2|\hat{\psi}|^2$")
    ax1.set_title("Time-averaged horizontal KE spectrum")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(loc="lower left", fontsize=9)

    # Vertical KE
    for zi in range(Nz1):
        ax2.loglog(k_centers, Ew_avg[zi] + FLOOR, color=color_w, alpha=0.12, lw=0.8)
    Ew_dm = Ew_avg.mean(axis=0)
    _add_slope(ax2, k_centers, Ew_dm, -5.0/3.0, r"$k^{-5/3}$", "tab:green")
    _add_slope(ax2, k_centers, Ew_dm, -3.0,     r"$k^{-3}$",   "tab:purple")
    ax2.loglog(k_centers, Ew_dm + FLOOR, color="black", lw=2.6,
               label=r"$\langle E_w \rangle_{z,t}$")
    ax2.set_xlabel(r"$k$ (horizontal)")
    ax2.set_ylabel(r"$\langle E_w(k)\rangle_t = \frac{1}{2}|\hat{w}|^2$")
    ax2.set_title("Time-averaged vertical KE spectrum")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(loc="lower left", fontsize=9)

    header = (args.title or snap_dir.name) + \
        f"   t in [{args.t_start:.1f}, {args.t_end:.1f}]   ({n} snaps)"
    fig.suptitle(header, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=130)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
