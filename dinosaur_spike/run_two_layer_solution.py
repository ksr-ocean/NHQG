#!/usr/bin/env python
"""Run a longer masked two-layer QG solution and save diagnostics/snapshots."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import time


def _configure_device(device: str, dtype: str) -> None:
    os.environ["JAX_ENABLE_X64"] = "1" if dtype == "float64" else "0"
    if device == "cpu":
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    elif device == "gpu7":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
    elif device == "default":
        return
    else:
        raise ValueError(f"unsupported device option {device!r}")


def _impl_from_name(name: str):
    from dinosaur import spherical_harmonic

    if name == "real":
        return spherical_harmonic.RealSphericalHarmonics
    if name == "fast":
        return spherical_harmonic.FastSphericalHarmonics
    raise ValueError(f"unsupported implementation {name!r}")


def _dtype_from_name(name: str):
    import jax.numpy as jnp

    if name == "float32":
        return jnp.float32
    if name == "float64":
        return jnp.float64
    raise ValueError(f"unsupported dtype {name!r}")


def _guard_operator_combo(impl: str, dtype: str, allow_float32_fast: bool) -> None:
    if impl == "fast" and dtype == "float32" and not allow_float32_fast:
        raise SystemExit(
            "refusing --impl fast --dtype float32: the spherical flux-divergence "
            "operator fails the constant-advection sanity check in this precision. "
            "Use --dtype float64, --impl real, or pass --allow-float32-fast for "
            "explicit diagnostic runs."
        )


def _make_grid(wavenumbers: int, impl_name: str):
    from dinosaur import spherical_harmonic

    return spherical_harmonic.Grid.with_wavenumbers(
        longitude_wavenumbers=wavenumbers,
        dealiasing="quadratic",
        spherical_harmonics_impl=_impl_from_name(impl_name),
        radius=1.0,
    )


def _default_output_dir(wavenumbers: int, steps: int) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("output") / "dinosaur_two_layer" / f"w{wavenumbers}_n{steps}_{stamp}"


def _initial_state(grid, params, amplitude: float, max_wavenumber: int, dtype):
    import jax
    import jax.numpy as jnp

    from dinosaur_spike.two_layer_model import latitude_mask_nodal
    from dinosaur_spike.two_layer_qg import TwoLayerState, remove_mean_pv

    key1, key2 = jax.random.split(jax.random.PRNGKey(123))
    ell = jnp.arange(grid.modal_shape[1])
    lowpass = (ell[None, :] <= max_wavenumber).astype(dtype)
    spectral_mask = jnp.asarray(grid.mask, dtype=dtype) * lowpass
    mask_nodal = latitude_mask_nodal(grid, params).astype(dtype)

    q1_modal = jax.random.normal(key1, grid.modal_shape, dtype=dtype) * spectral_mask
    q2_modal = jax.random.normal(key2, grid.modal_shape, dtype=dtype) * spectral_mask
    q1_nodal = grid.to_nodal(q1_modal) * mask_nodal
    q2_nodal = grid.to_nodal(q2_modal) * mask_nodal
    max_abs = jnp.maximum(jnp.max(jnp.abs(q1_nodal)), jnp.max(jnp.abs(q2_nodal)))
    scale = amplitude / jnp.maximum(max_abs, jnp.asarray(1e-30, dtype=dtype))
    state = TwoLayerState(
        q1=grid.clip_wavenumbers(grid.to_modal(scale * q1_nodal)),
        q2=grid.clip_wavenumbers(grid.to_modal(scale * q2_nodal)),
    )
    return remove_mean_pv(state)


def _load_state(path: Path, dtype):
    import jax.numpy as jnp
    import numpy as np

    from dinosaur_spike.two_layer_qg import TwoLayerState, remove_mean_pv

    with np.load(path) as data:
        state = TwoLayerState(
            q1=jnp.asarray(data["q1"], dtype=dtype),
            q2=jnp.asarray(data["q2"], dtype=dtype),
        )
    return remove_mean_pv(state)


def _outside_enstrophy(state, grid, mask):
    q1 = grid.to_nodal(state.q1)
    q2 = grid.to_nodal(state.q2)
    return 0.5 * grid.integrate((1.0 - mask) * (q1 * q1 + q2 * q2))


def _field_metrics(state, grid):
    import jax.numpy as jnp

    q1 = grid.to_nodal(state.q1)
    q2 = grid.to_nodal(state.q2)
    return {
        "q1_min": jnp.min(q1),
        "q1_max": jnp.max(q1),
        "q2_min": jnp.min(q2),
        "q2_max": jnp.max(q2),
        "q_abs_max": jnp.maximum(jnp.max(jnp.abs(q1)), jnp.max(jnp.abs(q2))),
    }


def _shell_power(state, grid):
    import jax.numpy as jnp

    valid = jnp.asarray(grid.mask, dtype=state.q1.dtype)
    q1_power = jnp.real(state.q1 * jnp.conj(state.q1))
    q2_power = jnp.real(state.q2 * jnp.conj(state.q2))
    return jnp.sum((q1_power + q2_power) * valid, axis=0)


def _spectral_metrics_from_shell(shell_power):
    import numpy as np

    shell = np.asarray(shell_power, dtype=np.float64)
    ell = np.arange(shell.size, dtype=np.float64)
    total = float(np.sum(shell))
    if total <= 0.0:
        return {
            "spectral_peak_l": 0.0,
            "spectral_mean_l": 0.0,
            "spectral_rms_l": 0.0,
            "spectral_top20_fraction": 0.0,
            "spectral_top10_fraction": 0.0,
        }

    nonzero_shell = shell.copy()
    nonzero_shell[0] = 0.0
    top20 = max(1, int(0.8 * shell.size))
    top10 = max(1, int(0.9 * shell.size))
    return {
        "spectral_peak_l": float(np.argmax(nonzero_shell)),
        "spectral_mean_l": float(np.sum(ell * shell) / total),
        "spectral_rms_l": float(np.sqrt(np.sum(ell * ell * shell) / total)),
        "spectral_top20_fraction": float(np.sum(shell[top20:]) / total),
        "spectral_top10_fraction": float(np.sum(shell[top10:]) / total),
    }


def _save_snapshot(state, grid, mask, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    q1 = np.asarray(grid.to_nodal(state.q1)).T
    q2 = np.asarray(grid.to_nodal(state.q2)).T
    mask_np = np.asarray(mask).T
    lon = np.rad2deg(np.asarray(grid.longitudes))
    lat = np.rad2deg(np.asarray(grid.latitudes))
    vmax = max(float(np.max(np.abs(q1))), float(np.max(np.abs(q2))), 1e-12)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for ax, field, layer in zip(axes, [q1, q2], ["layer 1", "layer 2"], strict=True):
        im = ax.imshow(
            field,
            origin="lower",
            extent=[lon[0], lon[-1], lat[0], lat[-1]],
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.contour(lon, lat, mask_np, levels=[0.5], colors="black", linewidths=0.8)
        ax.set_ylabel("latitude")
        ax.set_title(layer)
        fig.colorbar(im, ax=ax, shrink=0.86, label="PV anomaly")
    axes[-1].set_xlabel("longitude")
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _append_diagnostic(path: Path, row: dict[str, float]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _append_shell_power(path: Path, step: int, shell_power) -> None:
    import numpy as np

    exists = path.exists()
    shell = np.asarray(shell_power, dtype=np.float64)
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["step", "ell", "q_power"])
        writer.writerows((step, ell, value) for ell, value in enumerate(shell))


def _save_state_checkpoint(path: Path, state, grid, args, step: int, stop_reason: str) -> None:
    import numpy as np

    np.savez(
        path,
        q1=np.asarray(state.q1),
        q2=np.asarray(state.q2),
        latitudes=np.asarray(grid.latitudes),
        longitudes=np.asarray(grid.longitudes),
        dt=args.dt,
        requested_steps=args.steps,
        completed_steps=step,
        stop_reason=stop_reason,
        wavenumbers=args.wavenumbers,
        amplitude=args.amplitude,
        init_max_wavenumber=args.init_max_wavenumber,
        restart_state="" if args.restart_state is None else str(args.restart_state),
        background_barotropic_velocity=args.background_barotropic_velocity,
        background_shear_velocity=args.background_shear_velocity,
        background_profile=args.background_profile,
        background_sin3_weight=args.background_sin3_weight,
        hyperdiffusion_rate=args.hyperdiffusion_rate,
        hyperdiffusion_order=args.hyperdiffusion_order,
        time_stepper=args.time_stepper,
        sponge_max_rate=args.sponge_max_rate,
        mask_plateau_north_edge_deg=args.mask_plateau_north_edge_deg,
        mask_taper_north_edge_deg=args.mask_taper_north_edge_deg,
        mask_nonlinear_tendency=not args.no_mask_nonlinear_tendency,
    )


def _memory_stats_string():
    import jax

    stats = jax.devices()[0].memory_stats()
    if not stats:
        return "memory_stats=unavailable"
    keys = ["bytes_in_use", "peak_bytes_in_use", "bytes_limit"]
    return " ".join(
        f"{key}={stats[key] / 1e9:.3f}GB" for key in keys if key in stats
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu7", "default"], default="gpu7")
    parser.add_argument("--impl", choices=["real", "fast"], default="fast")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--wavenumbers", type=int, default=255)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--snapshot-every", type=int, default=200)
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument("--init-max-wavenumber", type=int, default=24)
    parser.add_argument("--sponge-max-rate", type=float, default=0.5)
    parser.add_argument("--hyperdiffusion-rate", type=float, default=0.0)
    parser.add_argument("--hyperdiffusion-order", type=int, default=2)
    parser.add_argument("--time-stepper", choices=["explicit", "ifrk4"], default="ifrk4")
    parser.add_argument("--background-barotropic-velocity", type=float, default=0.0)
    parser.add_argument("--background-shear-velocity", type=float, default=0.0)
    parser.add_argument(
        "--background-profile",
        choices=["solid_body", "sin_plus_sin3"],
        default="solid_body",
    )
    parser.add_argument("--background-sin3-weight", type=float, default=0.75)
    parser.add_argument("--mask-plateau-north-edge-deg", type=float, default=-30.0)
    parser.add_argument("--mask-taper-north-edge-deg", type=float, default=5.0)
    parser.add_argument("--no-mask-nonlinear-tendency", action="store_true")
    parser.add_argument("--restart-state", type=Path, default=None)
    parser.add_argument("--max-walltime-hours", type=float, default=0.0)
    parser.add_argument("--stop-top10-fraction", type=float, default=0.0)
    parser.add_argument("--stop-q-abs-max", type=float, default=0.0)
    parser.add_argument("--save-state-every", type=int, default=0)
    parser.add_argument("--allow-float32-fast", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    _guard_operator_combo(args.impl, args.dtype, args.allow_float32_fast)
    _configure_device(args.device, args.dtype)

    import jax
    import numpy as np

    from dinosaur_spike.two_layer_model import (
        TwoLayerParams,
        ifrk4_step,
        latitude_mask_nodal,
        rk4_step,
        windowed_enstrophy,
    )

    out_dir = args.out_dir or _default_output_dir(args.wavenumbers, args.steps)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = _make_grid(args.wavenumbers, args.impl)
    params = TwoLayerParams(
        sponge_max_rate=args.sponge_max_rate,
        hyperdiffusion_rate=args.hyperdiffusion_rate,
        hyperdiffusion_order=args.hyperdiffusion_order,
        background_barotropic_velocity=args.background_barotropic_velocity,
        background_shear_velocity=args.background_shear_velocity,
        background_profile=args.background_profile,
        background_sin3_weight=args.background_sin3_weight,
        mask_plateau_north_edge_deg=args.mask_plateau_north_edge_deg,
        mask_taper_north_edge_deg=args.mask_taper_north_edge_deg,
        mask_nonlinear_tendency=not args.no_mask_nonlinear_tendency,
    )
    dtype = _dtype_from_name(args.dtype)
    if args.restart_state is None:
        state = _initial_state(
            grid, params, args.amplitude, args.init_max_wavenumber, dtype
        )
    else:
        state = _load_state(args.restart_state, dtype)
    mask = latitude_mask_nodal(grid, params)

    if args.time_stepper == "explicit":
        step_fn = jax.jit(lambda s: rk4_step(s, grid, params, args.dt))
    elif args.time_stepper == "ifrk4":
        step_fn = jax.jit(lambda s: ifrk4_step(s, grid, params, args.dt))
    else:
        raise ValueError(f"unsupported time_stepper {args.time_stepper!r}")
    enstrophy_fn = jax.jit(lambda s: windowed_enstrophy(s, grid, params))
    outside_fn = jax.jit(lambda s: _outside_enstrophy(s, grid, mask))
    metrics_fn = jax.jit(lambda s: _field_metrics(s, grid))
    shell_power_fn = jax.jit(lambda s: _shell_power(s, grid))

    diagnostics_path = out_dir / "diagnostics.csv"
    shell_power_path = out_dir / "shell_power.csv"
    last_record = {"step": 0, "elapsed": 0.0}
    last_row = {}

    def record(step: int, elapsed: float) -> None:
        metrics = {key: float(value) for key, value in metrics_fn(state).items()}
        shell_power = shell_power_fn(state)
        spectral_metrics = _spectral_metrics_from_shell(shell_power)
        interval_steps = max(step - last_record["step"], 1)
        interval_elapsed = max(elapsed - last_record["elapsed"], 0.0)
        row = {
            "step": step,
            "time": step * args.dt,
            "elapsed_s": elapsed,
            "cumulative_step_ms": 1000.0 * elapsed / max(step, 1),
            "interval_step_ms": 1000.0 * interval_elapsed / interval_steps,
            "windowed_enstrophy": float(enstrophy_fn(state)),
            "outside_enstrophy": float(outside_fn(state)),
            **metrics,
            **spectral_metrics,
        }
        _append_diagnostic(diagnostics_path, row)
        _append_shell_power(shell_power_path, step, shell_power)
        image_path = out_dir / f"pv_step_{step:06d}.png"
        _save_snapshot(state, grid, mask, image_path, f"step {step}, t={step * args.dt:.3f}")
        if args.save_state_every > 0 and step % args.save_state_every == 0:
            checkpoint_path = out_dir / f"state_step_{step:06d}.npz"
            _save_state_checkpoint(checkpoint_path, state, grid, args, step, "checkpoint")
        print(
            f"step={step:6d} t={step * args.dt:.3f} "
            f"step_ms={row['interval_step_ms']:.3f} "
            f"ens={row['windowed_enstrophy']:.6e} "
            f"outside={row['outside_enstrophy']:.6e} "
            f"q_abs_max={row['q_abs_max']:.6e} "
            f"peak_l={row['spectral_peak_l']:.0f} "
            f"top10={row['spectral_top10_fraction']:.3e}",
            flush=True,
        )
        last_record["step"] = step
        last_record["elapsed"] = elapsed
        last_row.clear()
        last_row.update(row)

    print(f"devices={jax.devices()}")
    print(f"dtype={args.dtype}")
    print(
        "background_velocities="
        f"({args.background_barotropic_velocity}, {args.background_shear_velocity})"
    )
    print(
        "background_profile="
        f"{args.background_profile} sin3_weight={args.background_sin3_weight}"
    )
    print(
        "stepper="
        f"{args.time_stepper} hyperdiffusion_rate={args.hyperdiffusion_rate} "
        f"order={args.hyperdiffusion_order}"
    )
    print(f"modal_shape={grid.modal_shape} nodal_shape={grid.nodal_shape}")
    print(f"out_dir={out_dir}")

    warmup_state = step_fn(state)
    jax.block_until_ready(warmup_state)
    start = time.perf_counter()
    final_step = 0
    stop_reason = "completed"
    record(0, 0.0)
    for step in range(1, args.steps + 1):
        state = step_fn(state)
        if step % args.snapshot_every == 0 or step == args.steps:
            jax.block_until_ready(state)
            elapsed = time.perf_counter() - start
            record(step, elapsed)
            final_step = step
            if not np.isfinite(last_row["q_abs_max"]):
                stop_reason = "nonfinite_q_abs_max"
                break
            if args.stop_q_abs_max > 0.0 and last_row["q_abs_max"] >= args.stop_q_abs_max:
                stop_reason = "q_abs_max_limit"
                break
            if (
                args.stop_top10_fraction > 0.0
                and last_row["spectral_top10_fraction"] >= args.stop_top10_fraction
            ):
                stop_reason = "spectral_top10_limit"
                break
            if args.max_walltime_hours > 0.0 and elapsed >= 3600.0 * args.max_walltime_hours:
                stop_reason = "walltime_limit"
                break

    jax.block_until_ready(state)
    _save_state_checkpoint(
        out_dir / "final_state.npz",
        state,
        grid,
        args,
        final_step,
        stop_reason,
    )
    elapsed = time.perf_counter() - start
    print(f"stop_reason={stop_reason}")
    print(f"elapsed_s={elapsed:.3f}")
    print(_memory_stats_string())


if __name__ == "__main__":
    main()
