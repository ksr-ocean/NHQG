#!/usr/bin/env python
"""Analyze vortex-crystal length scales in P0 polar-cap runs."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import netCDF4 as _netCDF4
import numpy as np

from nhqg import polar_diagnostics as _polar_diagnostics


_SUMMARY_FIELDS = (
    "gamma",
    "U_late",
    "L_gamma",
    "n_vortices_med",
    "R_crystal_mean",
    "R_crystal_std",
    "spacing_mean",
    "spacing_std",
    "n_snapshots",
    "t_max",
)


def crystal_metrics(zeta_bt, L, threshold_frac=0.3, min_separation=2.0):
    """Measure periodic vortex-core radii and nearest-neighbor spacings."""
    zeta_bt = np.asarray(zeta_bt, dtype=np.float64)
    L = float(L)
    positions = _polar_diagnostics.vortex_positions(
        zeta_bt, L, threshold_frac=threshold_frac, min_separation=min_separation
    )
    positions = np.asarray(positions, dtype=np.float64)
    n_vortices = int(len(positions))

    if n_vortices == 0:
        r_crystal = float("nan")
    else:
        displacement = (positions - L / 2.0 + L / 2.0) % L - L / 2.0
        r_crystal = float(np.sqrt(np.mean(np.sum(displacement ** 2, axis=1))))

    if n_vortices < 2:
        spacing = float("nan")
    else:
        displacement = positions[:, None, :] - positions[None, :, :]
        displacement = (displacement + L / 2.0) % L - L / 2.0
        distances = np.sqrt(np.sum(displacement ** 2, axis=2))
        np.fill_diagonal(distances, np.inf)
        spacing = float(np.median(np.min(distances, axis=1)))

    return {
        "positions": positions,
        "n_vortices": n_vortices,
        "R_crystal": r_crystal,
        "spacing": spacing,
    }


def zeta_bt_from_snapshot(path):
    """Return vertically averaged barotropic vorticity, domain size, and time."""
    with _netCDF4.Dataset(path) as ds:
        q = ds["q_prime"][:]
        z = ds["z"][:]
        x = ds["x"][:]
        qbar = np.trapezoid(q, x=z, axis=0) / (z[-1] - z[0])
        zeta_bt = np.asarray(qbar, dtype=np.float64).T
        L = float(x[1] - x[0]) * len(x)
        t = float(ds.time)
    return zeta_bt, L, t


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs=2, metavar=("OUTPUT_DIR", "GAMMA"),
                        required=True)
    parser.add_argument("--avg-window", type=float, default=200.0)
    parser.add_argument("--threshold-frac", type=float, default=0.3)
    parser.add_argument("--min-separation", type=float, default=2.0)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def _read_diagnostics(path, avg_window):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no diagnostic rows in {path}")

    t_max = float(rows[-1]["t"])
    late_u = [np.sqrt(2.0 * float(row["KE_tot"])) for row in rows
              if float(row["t"]) >= t_max - avg_window]
    if not late_u:
        raise ValueError(f"no in-window diagnostic rows in {path}")
    return float(np.mean(late_u)), t_max


def _snapshot_time(path):
    with _netCDF4.Dataset(path) as ds:
        return float(ds.time)


def _nan_mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return mean, std


def _save_overlay(path, gamma, zeta_bt, L, t, metrics, r_crystal_mean):
    vmax = float(np.max(np.abs(zeta_bt)))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        zeta_bt.T,
        origin="lower",
        extent=[0, L, 0, L],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    positions = metrics["positions"]
    if len(positions):
        ax.plot(positions[:, 0], positions[:, 1], "kx", markersize=6)
    if np.isfinite(r_crystal_mean):
        ax.add_patch(Circle((L / 2.0, L / 2.0), r_crystal_mean,
                            fill=False, color="k", linestyle="-"))
    ax.add_patch(Circle((L / 2.0, L / 2.0), 0.45 * L / 2.0,
                        fill=False, color="k", linestyle="--"))
    ax.set_title(
        f"gamma={gamma:g}  t={t:.1f}  N={metrics['n_vortices']}  "
        f"R={metrics['R_crystal']:.2f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fit_and_plot_series(ax, records, y_name, std_name, label, marker, color):
    x = np.asarray([record["L_gamma"] for record in records], dtype=np.float64)
    y = np.asarray([record[y_name] for record in records], dtype=np.float64)
    yerr = np.asarray([record[std_name] for record in records], dtype=np.float64)
    valid = np.isfinite(x) & (x > 0.0) & np.isfinite(y) & (y > 0.0)
    ax.errorbar(x[valid], y[valid], yerr=yerr[valid], fmt=marker, color=color,
                capsize=3, linestyle="none", label=label)

    if np.count_nonzero(valid) < 2:
        return float("nan")
    slope, intercept = np.polyfit(np.log(x[valid]), np.log(y[valid]), 1)
    fit_x = np.geomspace(np.min(x[valid]), np.max(x[valid]), 100)
    fit_y = np.exp(intercept) * fit_x ** slope
    ax.plot(fit_x, fit_y, color=color, label=f"{label} (slope {slope:.2f})")
    return float(slope)


def _save_scaling(path, records):
    fig, ax = plt.subplots(figsize=(7, 5))
    r_slope = _fit_and_plot_series(
        ax, records, "R_crystal_mean", "R_crystal_std", "R_crystal", "o", "C0"
    )
    spacing_slope = _fit_and_plot_series(
        ax, records, "spacing_mean", "spacing_std", "spacing", "s", "C1"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("L_gamma = (U/gamma)^{1/3}")
    ax.set_ylabel("length")
    ax.set_title("P0 crystal scaling")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return r_slope, spacing_slope


def _main():
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for run_dir_text, gamma_text in args.run:
        run_dir = Path(run_dir_text)
        gamma = float(gamma_text)
        u_late, t_max = _read_diagnostics(run_dir / "diagnostics.csv", args.avg_window)
        l_gamma = float((u_late / gamma) ** (1.0 / 3.0))
        window_start = t_max - args.avg_window

        snapshots = []
        for snapshot_path in sorted(run_dir.glob("snapshot_*.nc")):
            snapshot_t = _snapshot_time(snapshot_path)
            if window_start <= snapshot_t <= t_max:
                zeta_bt, L, t = zeta_bt_from_snapshot(snapshot_path)
                snapshots.append((t, zeta_bt, L, crystal_metrics(
                    zeta_bt, L, threshold_frac=args.threshold_frac,
                    min_separation=args.min_separation
                )))
        if not snapshots:
            print(f"warning: skipping {run_dir}: no snapshots in averaging window", file=sys.stderr)
            continue

        snapshots.sort(key=lambda item: item[0])
        r_mean, r_std = _nan_mean_std([item[3]["R_crystal"] for item in snapshots])
        spacing_mean, spacing_std = _nan_mean_std([item[3]["spacing"] for item in snapshots])
        record = {
            "gamma": gamma,
            "U_late": u_late,
            "L_gamma": l_gamma,
            "n_vortices_med": float(np.median([item[3]["n_vortices"] for item in snapshots])),
            "R_crystal_mean": r_mean,
            "R_crystal_std": r_std,
            "spacing_mean": spacing_mean,
            "spacing_std": spacing_std,
            "n_snapshots": len(snapshots),
            "t_max": t_max,
        }
        records.append(record)
        t, zeta_bt, L, metrics = snapshots[-1]
        _save_overlay(out_dir / f"overlay_g{gamma:g}.png", gamma, zeta_bt, L, t,
                      metrics, r_mean)

    summary_path = out_dir / "p0_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    r_slope, spacing_slope = _save_scaling(out_dir / "p0_scaling.png", records)
    writer = csv.DictWriter(sys.stdout, fieldnames=_SUMMARY_FIELDS)
    writer.writeheader()
    writer.writerows(records)
    print(f"SLOPES R_crystal={r_slope:.4f} spacing={spacing_slope:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
