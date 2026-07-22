#!/usr/bin/env python
"""Diagnose whether a two-layer QG run is failing at the polar rings."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np


def _configure_device(device: str) -> None:
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    if device == "cpu":
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    elif device == "gpu7":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
    elif device == "default":
        return
    else:
        raise ValueError(f"unsupported device option {device!r}")


def _step_from_path(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.rsplit("_", maxsplit=1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"not a recognized checkpoint path: {path}") from exc


def _checkpoint_paths(run_dir: Path, max_frames: int | None) -> list[Path]:
    paths = sorted(run_dir.glob("state_step_*.npz"), key=_step_from_path)
    if not paths:
        raise FileNotFoundError(f"no state_step_*.npz checkpoints found in {run_dir}")
    if max_frames is None or max_frames >= len(paths):
        return paths
    indices = np.linspace(0, len(paths) - 1, max_frames).round().astype(int)
    return [paths[int(i)] for i in np.unique(indices)]


def _scalar(data, key: str, default):
    if key not in data.files:
        return default
    value = data[key]
    if value.shape == ():
        return value.item()
    return value


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _make_grid(wavenumbers: int, impl_name: str):
    from dinosaur import spherical_harmonic
    from dinosaur_spike.run_two_layer_solution import _impl_from_name

    return spherical_harmonic.Grid.with_wavenumbers(
        longitude_wavenumbers=wavenumbers,
        dealiasing="quadratic",
        spherical_harmonics_impl=_impl_from_name(impl_name),
        radius=1.0,
    )


def _params_from_checkpoint(data, args):
    from dinosaur_spike.two_layer_model import TwoLayerParams

    return TwoLayerParams(
        F1=args.F1,
        F2=args.F2,
        omega=args.omega,
        sponge_max_rate=float(_scalar(data, "sponge_max_rate", 0.0)),
        hyperdiffusion_rate=float(_scalar(data, "hyperdiffusion_rate", 0.0)),
        hyperdiffusion_order=int(_scalar(data, "hyperdiffusion_order", 2)),
        background_barotropic_velocity=float(
            _scalar(data, "background_barotropic_velocity", 0.0)
        ),
        background_shear_velocity=float(_scalar(data, "background_shear_velocity", 0.0)),
        background_profile=str(_scalar(data, "background_profile", "solid_body")),
        background_sin3_weight=float(_scalar(data, "background_sin3_weight", 0.75)),
        mask_plateau_north_edge_deg=float(
            _scalar(data, "mask_plateau_north_edge_deg", -30.0)
        ),
        mask_taper_north_edge_deg=float(
            _scalar(data, "mask_taper_north_edge_deg", 5.0)
        ),
        mask_nonlinear_tendency=_as_bool(
            _scalar(data, "mask_nonlinear_tendency", True)
        ),
    )


def _max_by_lat(field: np.ndarray) -> np.ndarray:
    return np.nanmax(np.abs(field), axis=0)


def _global_lat_peak(profile: np.ndarray, lat_deg: np.ndarray) -> tuple[float, float, int]:
    if np.all(~np.isfinite(profile)):
        return math.nan, math.nan, -1
    index = int(np.nanargmax(profile))
    return float(profile[index]), float(lat_deg[index]), index


def _flux_metric_profile(grid, psi_modal, q_modal) -> np.ndarray:
    import jax.numpy as jnp

    vcos = grid.k_cross(grid.cos_lat_grad(psi_modal, clip=True))
    vcos_nodal = grid.to_nodal(jnp.stack(vcos))
    q_nodal = grid.to_nodal(q_modal)
    flux_nodal = vcos_nodal * q_nodal[None, :, :] * grid.sec2_lat[None, None, :]
    flux_mag = jnp.sqrt(flux_nodal[0] * flux_nodal[0] + flux_nodal[1] * flux_nodal[1])
    return _max_by_lat(np.asarray(flux_mag))


def _ring_enstrophy_fractions(grid, q1_nodal: np.ndarray, q2_nodal: np.ndarray):
    weights = np.asarray(grid.spherical_harmonics.basis.w)
    density = 0.5 * (q1_nodal * q1_nodal + q2_nodal * q2_nodal)
    ring = weights * np.nansum(density, axis=0)
    total = float(np.nansum(ring))
    if not np.isfinite(total) or total <= 0.0:
        return total, math.nan, math.nan, math.nan, math.nan, math.nan

    lat_deg = np.asarray(grid.latitudes) * 180.0 / np.pi
    north80 = float(np.nansum(ring[lat_deg >= 80.0]) / total)
    south80 = float(np.nansum(ring[lat_deg <= -80.0]) / total)
    north85 = float(np.nansum(ring[lat_deg >= 85.0]) / total)
    south85 = float(np.nansum(ring[lat_deg <= -85.0]) / total)
    outer3 = float((np.nansum(ring[:3]) + np.nansum(ring[-3:])) / total)
    return total, north80, south80, north85, south85, outer3


def _diagnose_checkpoint(path: Path, grid, params):
    import jax.numpy as jnp

    from dinosaur_spike.two_layer_model import (
        background_pv_modal,
        background_streamfunction_modal,
        rhs,
    )
    from dinosaur_spike.two_layer_qg import (
        TwoLayerState,
        invert_streamfunction,
        remove_mean_pv,
    )

    data = np.load(path, allow_pickle=False)
    state = remove_mean_pv(
        TwoLayerState(q1=jnp.asarray(data["q1"]), q2=jnp.asarray(data["q2"]))
    )
    step = int(_scalar(data, "completed_steps", _step_from_path(path)))
    dt = float(_scalar(data, "dt", math.nan))

    lat_deg = np.asarray(grid.latitudes) * 180.0 / np.pi
    q1 = np.asarray(grid.to_nodal(state.q1))
    q2 = np.asarray(grid.to_nodal(state.q2))
    q_profile = np.maximum(_max_by_lat(q1), _max_by_lat(q2))
    q_max, q_lat, q_index = _global_lat_peak(q_profile, lat_deg)

    psi = invert_streamfunction(state, grid, F1=params.F1, F2=params.F2)
    psi1 = np.asarray(grid.to_nodal(psi.psi1))
    psi2 = np.asarray(grid.to_nodal(psi.psi2))
    psi_profile = np.maximum(_max_by_lat(psi1), _max_by_lat(psi2))
    psi_max, psi_lat, _ = _global_lat_peak(psi_profile, lat_deg)

    q0 = background_pv_modal(grid, params)
    psi0 = background_streamfunction_modal(grid, params)
    flux_candidates = [
        ("layer1_perturbation_velocity_total_pv", _flux_metric_profile(grid, psi.psi1, state.q1 + q0.q1)),
        ("layer2_perturbation_velocity_total_pv", _flux_metric_profile(grid, psi.psi2, state.q2 + q0.q2)),
        ("layer1_base_velocity_anomaly_pv", _flux_metric_profile(grid, psi0.psi1, state.q1)),
        ("layer2_base_velocity_anomaly_pv", _flux_metric_profile(grid, psi0.psi2, state.q2)),
    ]
    flux_kind, flux_profile = max(
        flux_candidates,
        key=lambda item: -math.inf
        if np.all(~np.isfinite(item[1]))
        else float(np.nanmax(item[1])),
    )
    flux_max, flux_lat, _ = _global_lat_peak(flux_profile, lat_deg)

    tendency = rhs(state, grid, params)
    t1 = np.asarray(grid.to_nodal(tendency.q1))
    t2 = np.asarray(grid.to_nodal(tendency.q2))
    tendency_profile = np.maximum(_max_by_lat(t1), _max_by_lat(t2))
    tendency_max, tendency_lat, _ = _global_lat_peak(tendency_profile, lat_deg)

    enstrophy = _ring_enstrophy_fractions(grid, q1, q2)
    polar_ring = q_index in {0, 1, 2, len(lat_deg) - 3, len(lat_deg) - 2, len(lat_deg) - 1}

    row = {
        "step": step,
        "time": step * dt if np.isfinite(dt) else math.nan,
        "q_abs_max": q_max,
        "q_abs_peak_lat_deg": q_lat,
        "q_abs_peak_outer3": polar_ring,
        "psi_abs_max": psi_max,
        "psi_abs_peak_lat_deg": psi_lat,
        "metric_flux_abs_max": flux_max,
        "metric_flux_peak_lat_deg": flux_lat,
        "metric_flux_kind": flux_kind,
        "tendency_abs_max": tendency_max,
        "tendency_peak_lat_deg": tendency_lat,
        "enstrophy_total": enstrophy[0],
        "enstrophy_north80_fraction": enstrophy[1],
        "enstrophy_south80_fraction": enstrophy[2],
        "enstrophy_north85_fraction": enstrophy[3],
        "enstrophy_south85_fraction": enstrophy[4],
        "enstrophy_outer3_fraction": enstrophy[5],
    }
    profiles = {
        "step": step,
        "time": row["time"],
        "lat_deg": lat_deg,
        "q_abs_max_by_lat": q_profile,
        "psi_abs_max_by_lat": psi_profile,
        "metric_flux_abs_max_by_lat": flux_profile,
        "tendency_abs_max_by_lat": tendency_profile,
    }
    return row, profiles


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_profiles(path: Path, profiles: list[dict]) -> None:
    np.savez(
        path,
        steps=np.asarray([item["step"] for item in profiles]),
        times=np.asarray([item["time"] for item in profiles]),
        lat_deg=profiles[0]["lat_deg"],
        q_abs_max_by_lat=np.asarray([item["q_abs_max_by_lat"] for item in profiles]),
        psi_abs_max_by_lat=np.asarray([item["psi_abs_max_by_lat"] for item in profiles]),
        metric_flux_abs_max_by_lat=np.asarray(
            [item["metric_flux_abs_max_by_lat"] for item in profiles]
        ),
        tendency_abs_max_by_lat=np.asarray(
            [item["tendency_abs_max_by_lat"] for item in profiles]
        ),
    )


def _plot_outputs(out_prefix: Path, rows: list[dict], profiles: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.asarray([row["time"] for row in rows])
    qmax = np.asarray([row["q_abs_max"] for row in rows])
    qlat = np.asarray([row["q_abs_peak_lat_deg"] for row in rows])
    fluxmax = np.asarray([row["metric_flux_abs_max"] for row in rows])
    fluxlat = np.asarray([row["metric_flux_peak_lat_deg"] for row in rows])
    outer3 = np.asarray([row["enstrophy_outer3_fraction"] for row in rows])
    north85 = np.asarray([row["enstrophy_north85_fraction"] for row in rows])
    south85 = np.asarray([row["enstrophy_south85_fraction"] for row in rows])

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    axes[0, 0].semilogy(times, qmax, marker="o")
    axes[0, 0].set_ylabel("max |q'|")
    axes[0, 0].set_xlabel("time")
    axes[0, 1].plot(times, qlat, marker="o")
    axes[0, 1].set_ylabel("latitude of max |q'|")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylim(-92, 92)

    axes[1, 0].semilogy(times, fluxmax, marker="o")
    axes[1, 0].set_ylabel(r"max $|\sec^2\phi\,v_\cos q|$")
    axes[1, 0].set_xlabel("time")
    axes[1, 1].plot(times, fluxlat, marker="o")
    axes[1, 1].set_ylabel("latitude of metric-flux max")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylim(-92, 92)

    axes[2, 0].semilogy(times, np.maximum(outer3, 1e-30), marker="o", label="outer 3 rings")
    axes[2, 0].semilogy(times, np.maximum(north85, 1e-30), marker="o", label="north >=85")
    axes[2, 0].semilogy(times, np.maximum(south85, 1e-30), marker="o", label="south <=-85")
    axes[2, 0].set_ylabel("enstrophy fraction")
    axes[2, 0].set_xlabel("time")
    axes[2, 0].legend()

    axes[2, 1].axis("off")
    axes[2, 1].text(
        0.0,
        0.8,
        "A polar numerical failure should move pointwise maxima\n"
        "and metric-flux maxima onto the first/last latitude rings\n"
        "before the global spectrum looks truncation dominated.",
        va="top",
    )
    fig.savefig(out_prefix.with_name(out_prefix.name + "_timeseries.png"), dpi=180)
    plt.close(fig)

    selected = profiles
    if len(selected) > 8:
        indices = np.linspace(0, len(selected) - 1, 8).round().astype(int)
        selected = [selected[int(i)] for i in np.unique(indices)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for item in selected:
        label = f"t={item['time']:.1f}"
        axes[0].semilogy(item["lat_deg"], item["q_abs_max_by_lat"], label=label)
        axes[1].semilogy(item["lat_deg"], item["metric_flux_abs_max_by_lat"], label=label)
    axes[0].set_xlabel("latitude")
    axes[0].set_ylabel("max_lon/layer |q'|")
    axes[1].set_xlabel("latitude")
    axes[1].set_ylabel(r"max metric-flux magnitude")
    for ax in axes:
        ax.set_xlim(-90, 90)
        ax.legend(fontsize=8)
    fig.savefig(out_prefix.with_name(out_prefix.name + "_lat_profiles.png"), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--impl", choices=["fast", "real"], default="fast")
    parser.add_argument("--device", choices=["cpu", "gpu7", "default"], default="cpu")
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--F1", type=float, default=0.7)
    parser.add_argument("--F2", type=float, default=0.4)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--out-prefix", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    _configure_device(args.device)

    paths = _checkpoint_paths(args.run_dir, args.max_checkpoints)
    with np.load(paths[0], allow_pickle=False) as first:
        wavenumbers = int(_scalar(first, "wavenumbers", 0))
        if wavenumbers <= 0:
            raise ValueError("checkpoint lacks valid wavenumbers metadata")
        params = _params_from_checkpoint(first, args)

    grid = _make_grid(wavenumbers, args.impl)
    rows = []
    profiles = []
    for path in paths:
        row, profile = _diagnose_checkpoint(path, grid, params)
        rows.append(row)
        profiles.append(profile)
        print(
            f"step={row['step']} t={row['time']:.3f} "
            f"max|q|={row['q_abs_max']:.6g} at {row['q_abs_peak_lat_deg']:.3f} "
            f"flux={row['metric_flux_abs_max']:.6g} at {row['metric_flux_peak_lat_deg']:.3f}"
        )

    out_prefix = args.out_prefix or args.run_dir / "polar_instability"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out_prefix.with_name(out_prefix.name + "_diagnostics.csv"), rows)
    _save_profiles(out_prefix.with_name(out_prefix.name + "_profiles.npz"), profiles)
    if not args.no_plots:
        _plot_outputs(out_prefix, rows, profiles)


if __name__ == "__main__":
    main()
