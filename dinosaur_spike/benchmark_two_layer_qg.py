#!/usr/bin/env python
"""Benchmark two-layer QG kernels on Dinosaur spherical harmonic grids."""

from __future__ import annotations

import argparse
import os
import time


def _configure_device(device: str) -> None:
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
        os.environ.setdefault("JAX_ENABLE_X64", "1")
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


def _basis_memory_bytes(grid) -> int:
    basis = grid.spherical_harmonics.basis
    return int(basis.f.nbytes + basis.p.nbytes + basis.w.nbytes)


def _array_memory_estimate_bytes(grid, dtype_bytes: int) -> int:
    modal = int(grid.modal_shape[0] * grid.modal_shape[1])
    nodal = int(grid.nodal_shape[0] * grid.nodal_shape[1])
    # A rough lower bound for the RHS kernel: q, psi, tendencies, modal fluxes,
    # nodal q/psi/velocity/flux intermediates for two layers.
    return dtype_bytes * (12 * modal + 10 * nodal)


def _random_modal_pair(grid, dtype):
    import jax
    import jax.numpy as jnp

    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    mask = jnp.asarray(grid.mask, dtype=dtype)
    q1 = jax.random.normal(key1, grid.modal_shape, dtype=dtype) * mask
    q2 = jax.random.normal(key2, grid.modal_shape, dtype=dtype) * mask
    q1 = q1.at[0, 0].set(0.0)
    q2 = q2.at[0, 0].set(0.0)
    return q1, q2


def _memory_stats_string():
    import jax

    try:
        stats = jax.devices()[0].memory_stats()
    except Exception as exc:  # pragma: no cover - backend dependent.
        return f"memory_stats=unavailable ({type(exc).__name__})"
    if not stats:
        return "memory_stats=unavailable"
    keys = ["bytes_in_use", "peak_bytes_in_use", "bytes_limit"]
    parts = []
    for key in keys:
        if key in stats:
            parts.append(f"{key}={stats[key] / 1e9:.3f}GB")
    return " ".join(parts) if parts else f"memory_stats_keys={sorted(stats)}"


def run_case(wavenumbers: int, impl: str, dtype_name: str, steps: int, warmups: int):
    import jax
    import jax.numpy as jnp

    from dinosaur_spike.two_layer_qg import (
        TwoLayerState,
        invert_streamfunction,
    )

    dtype = _dtype_from_name(dtype_name)
    grid = _make_grid(wavenumbers, impl)
    q1, q2 = _random_modal_pair(grid, dtype)
    F1 = dtype.type(0.7) if hasattr(dtype, "type") else jnp.asarray(0.7, dtype=dtype)
    F2 = dtype.type(0.4) if hasattr(dtype, "type") else jnp.asarray(0.4, dtype=dtype)

    def layer_tendency(psi_modal, q_modal):
        vcos = grid.k_cross(grid.cos_lat_grad(psi_modal, clip=True))
        vcos_nodal = grid.to_nodal(jnp.stack(vcos))
        q_nodal = grid.to_nodal(q_modal)
        flux_nodal = vcos_nodal * q_nodal * grid.sec2_lat
        flux_modal = grid.to_modal(flux_nodal)
        return -grid.div_cos_lat((flux_modal[0], flux_modal[1]), clip=True)

    def rhs(q1_in, q2_in):
        state = TwoLayerState(q1=q1_in, q2=q2_in)
        psi = invert_streamfunction(state, grid, F1=F1, F2=F2)
        t1 = layer_tendency(psi.psi1, q1_in)
        t2 = layer_tendency(psi.psi2, q2_in)
        return t1, t2

    rhs_jit = jax.jit(rhs)
    for _ in range(warmups):
        out = rhs_jit(q1, q2)
        jax.block_until_ready(out)

    start = time.perf_counter()
    for _ in range(steps):
        out = rhs_jit(q1, q2)
        jax.block_until_ready(out)
    elapsed = time.perf_counter() - start
    mean_ms = 1000.0 * elapsed / max(steps, 1)

    dtype_bytes = jnp.dtype(dtype).itemsize
    return {
        "wavenumbers": wavenumbers,
        "impl": impl,
        "dtype": dtype_name,
        "modal_shape": grid.modal_shape,
        "nodal_shape": grid.nodal_shape,
        "basis_mb": _basis_memory_bytes(grid) / 1e6,
        "array_est_mb": _array_memory_estimate_bytes(grid, dtype_bytes) / 1e6,
        "mean_ms": mean_ms,
        "memory_stats": _memory_stats_string(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu7", "default"], default="gpu7")
    parser.add_argument("--impl", choices=["real", "fast"], default="fast")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--wavenumbers", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()

    _configure_device(args.device)

    import jax

    print(f"devices={jax.devices()}")
    print("wavenumbers impl dtype modal_shape nodal_shape basis_MB array_est_MB mean_ms memory")
    for wavenumbers in args.wavenumbers:
        result = run_case(wavenumbers, args.impl, args.dtype, args.steps, args.warmups)
        print(
            f"{result['wavenumbers']} {result['impl']} {result['dtype']} "
            f"{result['modal_shape']} {result['nodal_shape']} "
            f"{result['basis_mb']:.1f} {result['array_est_mb']:.1f} "
            f"{result['mean_ms']:.3f} {result['memory_stats']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
