#!/usr/bin/env python
"""Short masked two-layer QG run for GPU/CPU smoke testing."""

from __future__ import annotations

import argparse
import os
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
            "refusing --impl fast --dtype float32: use --dtype float64, "
            "--impl real, or pass --allow-float32-fast for explicit diagnostics."
        )


def _make_grid(wavenumbers: int, impl_name: str):
    from dinosaur import spherical_harmonic

    return spherical_harmonic.Grid.with_wavenumbers(
        longitude_wavenumbers=wavenumbers,
        dealiasing="quadratic",
        spherical_harmonics_impl=_impl_from_name(impl_name),
        radius=1.0,
    )


def _initial_state(grid, amplitude: float, dtype):
    import jax
    import jax.numpy as jnp

    key1, key2 = jax.random.split(jax.random.PRNGKey(42))
    mask = jnp.asarray(grid.mask, dtype=dtype)
    q1 = amplitude * jax.random.normal(key1, grid.modal_shape, dtype=dtype) * mask
    q2 = amplitude * jax.random.normal(key2, grid.modal_shape, dtype=dtype) * mask
    q1 = q1.at[0, 0].set(0.0)
    q2 = q2.at[0, 0].set(0.0)
    from dinosaur_spike.two_layer_qg import TwoLayerState

    return TwoLayerState(q1=q1, q2=q2)


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
    parser.add_argument("--wavenumbers", type=int, default=127)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--amplitude", type=float, default=1e-3)
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
    parser.add_argument("--allow-float32-fast", action="store_true")
    args = parser.parse_args()
    _guard_operator_combo(args.impl, args.dtype, args.allow_float32_fast)
    _configure_device(args.device, args.dtype)

    import jax
    import jax.numpy as jnp

    from dinosaur_spike.two_layer_model import (
        TwoLayerParams,
        ifrk4_step,
        latitude_mask_nodal,
        rk4_step,
        windowed_enstrophy,
    )

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
    state = _initial_state(grid, args.amplitude, dtype)
    mask = latitude_mask_nodal(grid, params)

    if args.time_stepper == "explicit":
        step_fn = jax.jit(lambda s: rk4_step(s, grid, params, args.dt))
    elif args.time_stepper == "ifrk4":
        step_fn = jax.jit(lambda s: ifrk4_step(s, grid, params, args.dt))
    else:
        raise ValueError(f"unsupported time_stepper {args.time_stepper!r}")
    enstrophy_fn = jax.jit(lambda s: windowed_enstrophy(s, grid, params))
    outside_fn = jax.jit(
        lambda s: 0.5
        * grid.integrate(
            (1.0 - mask)
            * (
                grid.to_nodal(s.q1) * grid.to_nodal(s.q1)
                + grid.to_nodal(s.q2) * grid.to_nodal(s.q2)
            )
        )
    )

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
    print(f"modal_shape={grid.modal_shape} nodal_shape={grid.nodal_shape}")
    print(f"initial_enstrophy={float(enstrophy_fn(state)):.6e}")
    print(f"initial_outside_enstrophy={float(outside_fn(state)):.6e}")

    warmup_state = step_fn(state)
    jax.block_until_ready(warmup_state)
    start = time.perf_counter()
    for _ in range(args.steps):
        state = step_fn(state)
    jax.block_until_ready(state)
    elapsed = time.perf_counter() - start
    print(f"mean_step_ms={1000.0 * elapsed / max(args.steps, 1):.3f}")
    print(f"final_enstrophy={float(enstrophy_fn(state)):.6e}")
    print(f"final_outside_enstrophy={float(outside_fn(state)):.6e}")
    print(_memory_stats_string())


if __name__ == "__main__":
    main()
