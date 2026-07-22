#!/usr/bin/env python
"""Smoke tests for Dinosaur spherical harmonic API conventions.

This script is intentionally small and diagnostic. It checks the low-level
transform and Laplacian conventions that the two-layer QG spike will rely on.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass


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


@dataclass(frozen=True)
class SmokeResult:
    roundtrip_error: float
    constant_coefficient: float
    constant_error: float
    laplacian_error: float
    integral_error: float
    modal_shape: tuple[int, int]
    nodal_shape: tuple[int, int]


def run_smoke(wavenumbers: int = 8) -> SmokeResult:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from dinosaur import spherical_harmonic

    grid = spherical_harmonic.Grid.with_wavenumbers(
        longitude_wavenumbers=wavenumbers,
        dealiasing="quadratic",
        spherical_harmonics_impl=spherical_harmonic.RealSphericalHarmonics,
        radius=1.0,
    )

    modal = jnp.zeros(grid.modal_shape, dtype=jnp.float64)
    modal = modal.at[0, 0].set(1.0)
    modal = modal.at[1, 2].set(0.25)
    modal = modal.at[2, 3].set(-0.1)
    modal = modal * jnp.asarray(grid.mask, dtype=modal.dtype)

    nodal = grid.to_nodal(modal)
    recovered = grid.to_modal(nodal)
    roundtrip_error = float(jnp.max(jnp.abs(recovered - modal)))

    constant = jnp.ones(grid.nodal_shape, dtype=jnp.float64)
    constant_modal = grid.to_modal(constant)
    reconstructed_constant = grid.to_nodal(constant_modal)
    constant_error = float(jnp.max(jnp.abs(reconstructed_constant - constant)))
    constant_coefficient = float(constant_modal[0, 0])

    m_axes, l_axes = grid.modal_axes
    l_mesh = jnp.asarray(np.broadcast_to(l_axes[None, :], grid.modal_shape))
    lap = grid.laplacian(modal)
    expected_lap = -l_mesh * (l_mesh + 1.0) * modal
    laplacian_error = float(jnp.max(jnp.abs(lap - expected_lap)))

    sphere_area = 4.0 * np.pi
    integral_error = float(jnp.abs(grid.integrate(constant) - sphere_area))

    # Materialize JAX work before returning so timing/callers see failures here.
    jax.block_until_ready(recovered)
    jax.block_until_ready(reconstructed_constant)
    jax.block_until_ready(lap)

    return SmokeResult(
        roundtrip_error=roundtrip_error,
        constant_coefficient=constant_coefficient,
        constant_error=constant_error,
        laplacian_error=laplacian_error,
        integral_error=integral_error,
        modal_shape=grid.modal_shape,
        nodal_shape=grid.nodal_shape,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavenumbers", type=int, default=8)
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu7", "default"],
        default="cpu",
        help="Use cpu for API checks; use gpu7 for GPU compile/run checks.",
    )
    args = parser.parse_args()
    _configure_device(args.device)

    result = run_smoke(args.wavenumbers)
    print(f"modal_shape={result.modal_shape}")
    print(f"nodal_shape={result.nodal_shape}")
    print(f"roundtrip_error={result.roundtrip_error:.3e}")
    print(f"constant_coefficient={result.constant_coefficient:.8f}")
    print(f"constant_error={result.constant_error:.3e}")
    print(f"laplacian_error={result.laplacian_error:.3e}")
    print(f"integral_error={result.integral_error:.3e}")

    tol = 5e-6
    if result.roundtrip_error > tol:
        raise SystemExit(f"roundtrip_error too large: {result.roundtrip_error}")
    if result.constant_error > tol:
        raise SystemExit(f"constant_error too large: {result.constant_error}")
    if result.laplacian_error > tol:
        raise SystemExit(f"laplacian_error too large: {result.laplacian_error}")
    if result.integral_error > tol:
        raise SystemExit(f"integral_error too large: {result.integral_error}")


if __name__ == "__main__":
    main()
