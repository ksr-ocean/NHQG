#!/usr/bin/env python
"""Sanity checks for the spherical flux-divergence QG operator."""

from __future__ import annotations

import argparse
import os


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


def _make_grid(wavenumbers: int, impl_name: str):
    from dinosaur import spherical_harmonic

    return spherical_harmonic.Grid.with_wavenumbers(
        longitude_wavenumbers=wavenumbers,
        dealiasing="quadratic",
        spherical_harmonics_impl=_impl_from_name(impl_name),
        radius=1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu7", "default"], default="gpu7")
    parser.add_argument("--impl", choices=["real", "fast"], default="fast")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--wavenumbers", type=int, default=128)
    parser.add_argument("--max-streamfunction-wavenumber", type=int, default=24)
    parser.add_argument("--fail-rms-above", type=float, default=1e-8)
    parser.add_argument("--fail-max-above", type=float, default=1e-6)
    args = parser.parse_args()
    _configure_device(args.device, args.dtype)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from dinosaur import spherical_harmonic

    from dinosaur_spike.two_layer_model import _layer_flux_tendency

    dtype = _dtype_from_name(args.dtype)
    grid = _make_grid(args.wavenumbers, args.impl)
    ell = jnp.arange(grid.modal_shape[1])
    lowpass = (ell[None, :] <= args.max_streamfunction_wavenumber).astype(dtype)
    psi = (
        jax.random.normal(jax.random.PRNGKey(17), grid.modal_shape, dtype=dtype)
        * jnp.asarray(grid.mask, dtype=dtype)
        * lowpass
    )
    psi = psi.at[0, 0].set(jnp.asarray(0.0, dtype=dtype))

    vcos = grid.k_cross(grid.cos_lat_grad(psi, clip=True))
    vcos_nodal = grid.to_nodal(jnp.stack(vcos))
    velocity_scale = jnp.maximum(jnp.max(jnp.abs(vcos_nodal)), jnp.asarray(1e-30, dtype))
    psi = psi / velocity_scale

    constant = spherical_harmonic.add_constant(
        jnp.zeros(grid.modal_shape, dtype=dtype), jnp.asarray(1.0, dtype=dtype)
    )
    tendency = _layer_flux_tendency(grid, psi, constant, None)
    nodal = grid.to_nodal(tendency)
    jax.block_until_ready(nodal)

    nodal_np = np.asarray(nodal)
    modal_np = np.asarray(tendency)
    shell_power = np.sum(np.abs(modal_np) ** 2, axis=0)
    total_power = float(np.sum(shell_power))
    top10 = 0.0
    peak_l = 0
    if total_power > 0.0:
        peak_l = int(np.argmax(shell_power[1:]) + 1)
        top10 = float(np.sum(shell_power[int(0.9 * len(shell_power)) :]) / total_power)
    max_abs = float(np.max(np.abs(nodal_np)))
    rms = float(np.sqrt(np.mean(nodal_np * nodal_np)))

    print(f"devices={jax.devices()}")
    print(f"impl={args.impl} dtype={args.dtype} wavenumbers={args.wavenumbers}")
    print(f"modal_shape={grid.modal_shape} nodal_shape={grid.nodal_shape}")
    print(f"sec2_lat_max={float(jnp.max(grid.sec2_lat)):.6e}")
    print(f"constant_advection_max_abs={max_abs:.6e}")
    print(f"constant_advection_rms={rms:.6e}")
    print(f"constant_advection_peak_l={peak_l}")
    print(f"constant_advection_top10_fraction={top10:.6e}")

    if rms > args.fail_rms_above or max_abs > args.fail_max_above:
        raise SystemExit(
            "constant-advection residual is too large: "
            f"rms={rms:.3e}, max={max_abs:.3e}"
        )


if __name__ == "__main__":
    main()
