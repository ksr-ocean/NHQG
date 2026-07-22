#!/usr/bin/env python
"""Run Dinosaur's native two-layer shallow-water equations."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import time

import numpy as np


def _configure_device(device: str, dtype: str) -> None:
    os.environ["JAX_ENABLE_X64"] = "1" if dtype == "float64" else "0"
    if device == "cpu":
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    elif device == "gpu7":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    elif device == "default":
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
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


def _default_output_dir(wavenumbers: int, steps: int) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("output") / "dinosaur_native_sw" / f"w{wavenumbers}_n{steps}_{stamp}"


def _make_grid(wavenumbers: int, impl_name: str):
    from dinosaur import spherical_harmonic

    return spherical_harmonic.Grid.with_wavenumbers(
        longitude_wavenumbers=wavenumbers,
        dealiasing="quadratic",
        spherical_harmonics_impl=_impl_from_name(impl_name),
        radius=1.0,
    )


def _make_specs(density_ratio: float):
    from dinosaur import scales
    from dinosaur import shallow_water

    densities = np.asarray([density_ratio, 1.0]) * scales.WATER_DENSITY
    return shallow_water.ShallowWaterSpecs.from_si(densities=densities)


def _initial_state(grid, coords, physics_specs, args):
    import jax
    import jax.numpy as jnp

    from dinosaur import shallow_water
    from dinosaur import shallow_water_states

    lon, sin_lat = grid.nodal_mesh
    lat = jnp.arcsin(sin_lat)
    center = jnp.deg2rad(args.jet_center_lat_deg)
    width = jnp.deg2rad(args.jet_width_deg)
    envelope = jnp.cos(lat) * jnp.exp(-0.5 * ((lat - center) / width) ** 2)
    u = jnp.stack(
        [
            args.top_jet_velocity * envelope,
            args.bottom_jet_velocity * envelope,
        ]
    )
    state = shallow_water_states.multi_layer(u, physics_specs.densities, coords)

    lon0 = jnp.deg2rad(args.bump_lon_deg)
    lat0 = jnp.deg2rad(args.bump_lat_deg)
    bump_lon_width = jnp.deg2rad(args.bump_lon_width_deg)
    bump_lat_width = jnp.deg2rad(args.bump_lat_width_deg)
    dlon = jnp.mod(lon - lon0 + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    bump = args.bump_amplitude * jnp.cos(lat) * jnp.exp(
        -0.5 * (dlon / bump_lon_width) ** 2
        -0.5 * ((lat - lat0) / bump_lat_width) ** 2
    )
    bump_modal = grid.to_modal(bump)
    bump_modal = bump_modal.at[0, 0].set(0.0)

    if args.baroclinic_noise_amplitude > 0.0:
        key = jax.random.PRNGKey(args.seed)
        noise = jax.random.normal(key, grid.nodal_shape, dtype=lat.dtype)
        noise = noise * jnp.cos(lat) * jnp.exp(-0.5 * ((lat - center) / width) ** 2)
        noise_modal = grid.to_modal(noise)
        degree = jnp.arange(noise_modal.shape[-1])
        band = (degree >= args.noise_min_degree) & (degree <= args.noise_max_degree)
        noise_modal = noise_modal * jnp.asarray(grid.mask) * band[None, :]
        noise_modal = noise_modal.at[0, 0].set(0.0)
        noise_nodal = grid.to_nodal(noise_modal)
        noise_rms = jnp.sqrt(jnp.mean(noise_nodal * noise_nodal))
        noise_modal = noise_modal * args.baroclinic_noise_amplitude / jnp.maximum(noise_rms, 1e-30)
        bump_modal = bump_modal + noise_modal

    potential = state.potential.at[0].add(bump_modal)
    potential = potential.at[1].add(args.lower_perturbation_factor * bump_modal)
    return shallow_water.State(
        vorticity=state.vorticity,
        divergence=state.divergence,
        potential=potential,
    )


def _shell_stats(shell: np.ndarray, prefix: str) -> dict[str, float]:
    ell = np.arange(shell.size, dtype=np.float64)
    total = float(np.sum(shell))
    if total <= 0.0:
        return {
            f"{prefix}_peak_l": 0.0,
            f"{prefix}_mean_l": 0.0,
            f"{prefix}_rms_l": 0.0,
            f"{prefix}_top10_fraction": 0.0,
            f"{prefix}_band8_80_fraction": 0.0,
        }
    nonzero = shell.copy()
    nonzero[0] = 0.0
    top10 = max(1, int(0.9 * shell.size))
    band_hi = min(81, shell.size)
    return {
        f"{prefix}_peak_l": float(np.argmax(nonzero)),
        f"{prefix}_mean_l": float(np.sum(ell * shell) / total),
        f"{prefix}_rms_l": float(np.sqrt(np.sum(ell * ell * shell) / total)),
        f"{prefix}_top10_fraction": float(np.sum(shell[top10:]) / total),
        f"{prefix}_band8_80_fraction": float(np.sum(shell[8:band_hi]) / total),
    }


def _shell_metrics(state, grid) -> dict[str, float]:
    mask = np.asarray(grid.mask)
    power = (
        np.real(np.asarray(state.vorticity * np.conj(state.vorticity)))
        + np.real(np.asarray(state.divergence * np.conj(state.divergence)))
        + np.real(np.asarray(state.potential * np.conj(state.potential)))
    )
    shell = np.sum(power * mask[None, :, :], axis=(0, 1))

    vorticity = np.asarray(state.vorticity)
    upper_vort_shell = np.sum(
        np.real(vorticity[0] * np.conj(vorticity[0])) * mask,
        axis=0,
    )
    baroclinic_vort = 0.5 * (vorticity[0] - vorticity[1])
    baroclinic_vort_shell = np.sum(
        np.real(baroclinic_vort * np.conj(baroclinic_vort)) * mask,
        axis=0,
    )

    return {
        **_shell_stats(shell, "spectral"),
        **_shell_stats(upper_vort_shell, "upper_vort_spectral"),
        **_shell_stats(baroclinic_vort_shell, "baroclinic_vort_spectral"),
    }


def _diagnostics(state, grid, mean_potential) -> dict[str, float]:
    import jax.numpy as jnp

    vort = grid.to_nodal(state.vorticity)
    div = grid.to_nodal(state.divergence)
    pot = grid.to_nodal(state.potential)
    total_pot = pot + mean_potential[:, None, None]
    enstrophy = 0.5 * jnp.sum(grid.integrate(vort * vort))
    divergence_rms = jnp.sqrt(jnp.mean(div * div))
    layer_mass = grid.integrate(total_pot)
    barotropic_vort = 0.5 * (vort[0] + vort[1])
    baroclinic_vort = 0.5 * (vort[0] - vort[1])
    barotropic_vort_rms = jnp.sqrt(jnp.mean(barotropic_vort * barotropic_vort))
    baroclinic_vort_rms = jnp.sqrt(jnp.mean(baroclinic_vort * baroclinic_vort))
    out = {
        "vort_abs_max": float(jnp.max(jnp.abs(vort))),
        "div_abs_max": float(jnp.max(jnp.abs(div))),
        "div_rms": float(divergence_rms),
        "pot_abs_max": float(jnp.max(jnp.abs(pot))),
        "total_pot_min": float(jnp.min(total_pot)),
        "total_pot_max": float(jnp.max(total_pot)),
        "enstrophy": float(enstrophy),
        "barotropic_vort_rms": float(barotropic_vort_rms),
        "baroclinic_vort_rms": float(baroclinic_vort_rms),
        "baroclinic_vort_fraction": float(
            baroclinic_vort_rms / jnp.maximum(barotropic_vort_rms + baroclinic_vort_rms, 1e-30)
        ),
        "layer0_mass": float(layer_mass[0]),
        "layer1_mass": float(layer_mass[1]),
    }
    out.update(_shell_metrics(state, grid))
    return out


def _append_csv(path: Path, row: dict[str, float]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _save_snapshot(path: Path, state, grid, step: int, time_value: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vort = np.asarray(grid.to_nodal(state.vorticity))
    fields = [vort[0].T, vort[1].T]
    lon = np.rad2deg(np.asarray(grid.longitudes))
    lat = np.rad2deg(np.asarray(grid.latitudes))
    vmax = max(float(np.max(np.abs(fields[0]))), float(np.max(np.abs(fields[1]))), 1e-12)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for ax, field, title in zip(axes, fields, ["upper layer", "lower layer"], strict=True):
        im = ax.imshow(
            field,
            origin="lower",
            extent=[lon[0], lon[-1], lat[0], lat[-1]],
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_ylabel("latitude")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.86, label="relative vorticity")
    axes[-1].set_xlabel("longitude")
    fig.suptitle(f"native 2-layer SW step {step}, t={time_value:.3f}")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_checkpoint(path: Path, state, grid, mean_potential, args, step: int, stop_reason: str) -> None:
    np.savez(
        path,
        vorticity=np.asarray(state.vorticity),
        divergence=np.asarray(state.divergence),
        potential=np.asarray(state.potential),
        mean_potential=np.asarray(mean_potential),
        latitudes=np.asarray(grid.latitudes),
        longitudes=np.asarray(grid.longitudes),
        completed_steps=step,
        requested_steps=args.steps,
        dt=args.dt,
        stop_reason=stop_reason,
        wavenumbers=args.wavenumbers,
        density_ratio=args.density_ratio,
        top_jet_velocity=args.top_jet_velocity,
        bottom_jet_velocity=args.bottom_jet_velocity,
        jet_center_lat_deg=args.jet_center_lat_deg,
        jet_width_deg=args.jet_width_deg,
        bump_amplitude=args.bump_amplitude,
        lower_perturbation_factor=args.lower_perturbation_factor,
        baroclinic_noise_amplitude=args.baroclinic_noise_amplitude,
        noise_min_degree=args.noise_min_degree,
        noise_max_degree=args.noise_max_degree,
        initial_condition=args.initial_condition,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu7", "default"], default="gpu7")
    parser.add_argument("--impl", choices=["real", "fast"], default="fast")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--wavenumbers", type=int, default=63)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--snapshot-every", type=int, default=500)
    parser.add_argument(
        "--initial-condition",
        choices=["multi_layer", "shear_bump", "galewsky"],
        default="multi_layer",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--density-ratio", type=float, default=0.95)
    parser.add_argument("--mean-potential-upper", type=float, default=0.08)
    parser.add_argument("--mean-potential-lower", type=float, default=0.12)
    parser.add_argument("--top-jet-velocity", type=float, default=0.16)
    parser.add_argument("--bottom-jet-velocity", type=float, default=-0.04)
    parser.add_argument("--jet-center-lat-deg", type=float, default=-60.0)
    parser.add_argument("--jet-width-deg", type=float, default=14.0)
    parser.add_argument("--bump-amplitude", type=float, default=1e-3)
    parser.add_argument("--bump-lon-deg", type=float, default=20.0)
    parser.add_argument("--bump-lat-deg", type=float, default=-58.0)
    parser.add_argument("--bump-lon-width-deg", type=float, default=18.0)
    parser.add_argument("--bump-lat-width-deg", type=float, default=8.0)
    parser.add_argument("--lower-perturbation-factor", type=float, default=-0.5)
    parser.add_argument("--baroclinic-noise-amplitude", type=float, default=0.0)
    parser.add_argument("--noise-min-degree", type=int, default=8)
    parser.add_argument("--noise-max-degree", type=int, default=32)
    parser.add_argument("--filter-tau", type=float, default=0.010938)
    parser.add_argument("--filter-order", type=int, default=18)
    parser.add_argument("--filter-cutoff", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.snapshot_every <= 0:
        raise ValueError("--snapshot-every must be positive")
    if args.noise_min_degree < 0 or args.noise_max_degree < args.noise_min_degree:
        raise ValueError("noise degree range must satisfy 0 <= min <= max")

    _configure_device(args.device, args.dtype)

    import jax
    import jax.numpy as jnp

    from dinosaur import coordinate_systems
    from dinosaur import layer_coordinates
    from dinosaur import shallow_water
    from dinosaur import shallow_water_states
    from dinosaur import time_integration
    from dinosaur import xarray_utils

    grid = _make_grid(args.wavenumbers, args.impl)
    vertical = layer_coordinates.LayerCoordinates(2)
    coords = coordinate_systems.CoordinateSystem(grid, vertical)
    physics_specs = _make_specs(args.density_ratio)
    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    if args.initial_condition in {"multi_layer", "shear_bump"}:
        mean_potential = jnp.asarray(
            [args.mean_potential_upper, args.mean_potential_lower],
            dtype=dtype,
        )
        state0 = _initial_state(grid, coords, physics_specs, args)
    elif args.initial_condition == "galewsky":
        state_fn, aux_features = shallow_water_states.barotropic_instability_tc(
            coords, physics_specs
        )
        mean_potential = jnp.asarray(
            aux_features[xarray_utils.REF_POTENTIAL_KEY],
            dtype=dtype,
        )
        state0 = state_fn(jax.random.PRNGKey(args.seed))
    else:
        raise ValueError(f"unsupported initial_condition {args.initial_condition!r}")
    state_pair = (state0, state0)

    equation = shallow_water.ShallowWaterEquations(
        coords=coords,
        physics_specs=physics_specs,
        orography=None,
        reference_potential=mean_potential,
    )
    step_fn = time_integration.semi_implicit_leapfrog(equation, args.dt)
    filters = (
        time_integration.exponential_leapfrog_step_filter(
            grid,
            args.dt,
            tau=args.filter_tau,
            order=args.filter_order,
            cutoff=args.filter_cutoff,
        ),
        time_integration.robert_asselin_leapfrog_filter(0.05),
    )
    step_fn = time_integration.step_with_filters(step_fn, filters)
    advance_cache = {}

    def advance_for(step_count: int):
        if step_count not in advance_cache:
            advance_cache[step_count] = jax.jit(
                time_integration.repeated(step_fn, step_count)
            )
        return advance_cache[step_count]

    out_dir = args.out_dir or _default_output_dir(args.wavenumbers, args.steps)
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = out_dir / "diagnostics.csv"

    print(f"devices={jax.devices()}")
    print(f"modal_shape={grid.modal_shape} nodal_shape={grid.nodal_shape}")
    print(f"vertical_layers={vertical.layers}")
    print(f"initial_condition={args.initial_condition} seed={args.seed}")
    print(f"mean_potential={np.asarray(mean_potential)} density_ratio={args.density_ratio}")
    print(f"out_dir={out_dir}")

    start = time.perf_counter()
    stop_reason = "completed"
    completed = 0

    def record(step: int, elapsed: float) -> None:
        current = state_pair[1]
        row = {
            "step": step,
            "time": step * args.dt,
            "elapsed_s": elapsed,
            "cumulative_step_ms": 1000.0 * elapsed / max(step, 1),
            **_diagnostics(current, grid, mean_potential),
        }
        _append_csv(diagnostics_path, row)
        _save_snapshot(out_dir / f"vorticity_step_{step:06d}.png", current, grid, step, step * args.dt)
        _save_checkpoint(out_dir / f"state_step_{step:06d}.npz", current, grid, mean_potential, args, step, "checkpoint")
        print(
            f"step={step:7d} t={step * args.dt:.4f} "
            f"step_ms={row['cumulative_step_ms']:.3f} "
            f"vort_abs={row['vort_abs_max']:.6e} "
            f"div_abs={row['div_abs_max']:.6e} "
            f"ens={row['enstrophy']:.6e} "
            f"pot_min={row['total_pot_min']:.6e} "
            f"bc_frac={row['baroclinic_vort_fraction']:.3f} "
            f"peak_l={row['spectral_peak_l']:.0f} "
            f"zeta_l={row['upper_vort_spectral_mean_l']:.1f} "
            f"top10={row['spectral_top10_fraction']:.3e}",
            flush=True,
        )

    record(0, 0.0)
    while completed < args.steps:
        chunk = min(args.snapshot_every, args.steps - completed)
        state_pair = advance_for(chunk)(state_pair)
        jax.block_until_ready(state_pair[1].vorticity)
        completed += chunk
        elapsed = time.perf_counter() - start
        record(completed, elapsed)
        metrics = _diagnostics(state_pair[1], grid, mean_potential)
        if not np.isfinite(metrics["vort_abs_max"]):
            stop_reason = "nonfinite_vorticity"
            break

    _save_checkpoint(out_dir / "final_state.npz", state_pair[1], grid, mean_potential, args, completed, stop_reason)
    print(f"stop_reason={stop_reason}")
    print(f"elapsed_s={time.perf_counter() - start:.3f}")


if __name__ == "__main__":
    main()
