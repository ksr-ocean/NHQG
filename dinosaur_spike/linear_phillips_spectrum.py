#!/usr/bin/env python
"""Compute exact spherical linear modes for the two-layer QG base state."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


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


def _guard_operator_combo(impl: str, dtype: str, allow_float32_fast: bool) -> None:
    if impl == "fast" and dtype == "float32" and not allow_float32_fast:
        raise SystemExit(
            "refusing --impl fast --dtype float32: use --dtype float64, "
            "--impl real, or pass --allow-float32-fast for explicit diagnostics."
        )


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


def _linear_tendency(state, psi, grid, params, mask_nodal):
    from dinosaur_spike.two_layer_model import (
        _layer_flux_tendency,
        background_pv_modal,
        background_streamfunction_modal,
        sponge_rate_from_mask,
    )
    from dinosaur_spike.two_layer_qg import (
        TwoLayerState,
        remove_mean_pv,
    )

    state = remove_mean_pv(state)
    psi0 = background_streamfunction_modal(grid, params)
    q0 = background_pv_modal(grid, params)
    t1 = _layer_flux_tendency(grid, psi.psi1, q0.q1, mask_nodal)
    t1 = t1 + _layer_flux_tendency(grid, psi0.psi1, state.q1, mask_nodal)
    t2 = _layer_flux_tendency(grid, psi.psi2, q0.q2, mask_nodal)
    t2 = t2 + _layer_flux_tendency(grid, psi0.psi2, state.q2, mask_nodal)
    if params.sponge_max_rate > 0.0:
        sponge_nodal = sponge_rate_from_mask(mask_nodal, params.sponge_max_rate)
        t1 = t1 - grid.to_modal(sponge_nodal * grid.to_nodal(state.q1))
        t2 = t2 - grid.to_modal(sponge_nodal * grid.to_nodal(state.q2))
    return remove_mean_pv(
        TwoLayerState(q1=grid.clip_wavenumbers(t1), q2=grid.clip_wavenumbers(t2))
    )


def _mode_rows(grid, zonal_wavenumber: int):
    import numpy as np

    modal_m = np.asarray(grid.modal_mesh[0])[:, 0]
    return np.where(np.abs(modal_m) == zonal_wavenumber)[0]


def _mode_indices(grid, zonal_wavenumber: int):
    import numpy as np

    rows = _mode_rows(grid, zonal_wavenumber)
    local_rows, local_ells = np.where(np.asarray(grid.mask)[rows, :])
    return rows[local_rows], local_ells


def _unpack_vector(vector, grid, row_indices, ell_indices, dtype):
    import jax.numpy as jnp

    from dinosaur_spike.two_layer_qg import TwoLayerState

    n_layer = len(row_indices)
    q1 = jnp.zeros(grid.modal_shape, dtype=dtype).at[row_indices, ell_indices].set(
        jnp.asarray(vector[:n_layer], dtype=dtype)
    )
    q2 = jnp.zeros(grid.modal_shape, dtype=dtype).at[row_indices, ell_indices].set(
        jnp.asarray(vector[n_layer:], dtype=dtype)
    )
    return TwoLayerState(q1=q1, q2=q2)


def _unpack_psi_vector(vector, grid, row_indices, ell_indices, dtype):
    import jax.numpy as jnp

    from dinosaur_spike.two_layer_qg import TwoLayerPsi

    n_layer = len(row_indices)
    psi1 = jnp.zeros(grid.modal_shape, dtype=dtype).at[row_indices, ell_indices].set(
        jnp.asarray(vector[:n_layer], dtype=dtype)
    )
    psi2 = jnp.zeros(grid.modal_shape, dtype=dtype).at[row_indices, ell_indices].set(
        jnp.asarray(vector[n_layer:], dtype=dtype)
    )
    return TwoLayerPsi(psi1=psi1, psi2=psi2)


def _pack_state(state, row_indices, ell_indices):
    import jax.numpy as jnp

    return jnp.concatenate(
        [
            state.q1[row_indices, ell_indices].reshape(-1),
            state.q2[row_indices, ell_indices].reshape(-1),
        ]
    )


def _pv_matrix_for_m(grid, params, row_indices, ell_indices, dtype):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from dinosaur_spike.two_layer_model import pv_from_streamfunction_model

    n = 2 * len(row_indices)

    def apply_psi(vector):
        psi = _unpack_psi_vector(vector, grid, row_indices, ell_indices, dtype)
        pv = pv_from_streamfunction_model(psi, grid, params)
        return _pack_state(pv, row_indices, ell_indices)

    apply_jit = jax.jit(apply_psi)
    apply_jit(jnp.zeros(n, dtype=dtype)).block_until_ready()
    matrix = np.empty((n, n), dtype=np.float64)
    for col in range(n):
        basis = np.zeros(n, dtype=np.float64)
        basis[col] = 1.0
        matrix[:, col] = np.asarray(apply_jit(jnp.asarray(basis, dtype=dtype)))
    return matrix


def _matrix_for_m(grid, params, zonal_wavenumber: int, dtype, mask_nodal):
    import jax
    import jax.numpy as jnp
    import numpy as np

    row_indices, ell_indices = _mode_indices(grid, zonal_wavenumber)
    if len(row_indices) == 0:
        raise ValueError(f"zonal wavenumber {zonal_wavenumber} is not represented")
    n = 2 * len(row_indices)
    pv_matrix = _pv_matrix_for_m(grid, params, row_indices, ell_indices, dtype)
    pv_inverse = jnp.asarray(np.linalg.inv(pv_matrix), dtype=dtype)

    def apply(vector):
        state = _unpack_vector(vector, grid, row_indices, ell_indices, dtype)
        psi_vector = pv_inverse @ jnp.asarray(vector, dtype=dtype)
        psi = _unpack_psi_vector(psi_vector, grid, row_indices, ell_indices, dtype)
        tendency = _linear_tendency(state, psi, grid, params, mask_nodal)
        return _pack_state(tendency, row_indices, ell_indices)

    apply_jit = jax.jit(apply)
    apply_jit(jnp.zeros(n, dtype=dtype)).block_until_ready()
    matrix = np.empty((n, n), dtype=np.float64)
    for col in range(n):
        basis = np.zeros(n, dtype=np.float64)
        basis[col] = 1.0
        matrix[:, col] = np.asarray(apply_jit(jnp.asarray(basis, dtype=dtype)))
    return matrix, row_indices, ell_indices


def _scale_state_to_amplitude(state, grid, amplitude: float):
    import jax.numpy as jnp

    from dinosaur_spike.two_layer_qg import TwoLayerState, remove_mean_pv

    q1 = grid.to_nodal(state.q1)
    q2 = grid.to_nodal(state.q2)
    max_abs = jnp.maximum(jnp.max(jnp.abs(q1)), jnp.max(jnp.abs(q2)))
    scale = amplitude / jnp.maximum(max_abs, jnp.asarray(1e-30, dtype=q1.dtype))
    return remove_mean_pv(TwoLayerState(q1=scale * state.q1, q2=scale * state.q2))


def _save_mode(path: Path, state, grid, args, eigenvalue, zonal_wavenumber: int) -> None:
    import numpy as np

    np.savez(
        path,
        q1=np.asarray(state.q1),
        q2=np.asarray(state.q2),
        latitudes=np.asarray(grid.latitudes),
        longitudes=np.asarray(grid.longitudes),
        wavenumbers=args.wavenumbers,
        background_shear_velocity=args.background_shear_velocity,
        background_profile=args.background_profile,
        background_sin3_weight=args.background_sin3_weight,
        deformation_profile=args.deformation_profile,
        deformation_reference_lat_deg=args.deformation_reference_lat_deg,
        deformation_f_floor_sin=args.deformation_f_floor_sin,
        sponge_max_rate=args.sponge_max_rate,
        mask_plateau_north_edge_deg=args.mask_plateau_north_edge_deg,
        mask_taper_north_edge_deg=args.mask_taper_north_edge_deg,
        mask_nonlinear_tendency=args.mask_linear_tendency,
        linear_growth_rate=float(eigenvalue.real),
        linear_frequency=float(eigenvalue.imag),
        linear_zonal_wavenumber=zonal_wavenumber,
        amplitude=args.amplitude,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu7", "default"], default="gpu7")
    parser.add_argument("--impl", choices=["real", "fast"], default="fast")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--wavenumbers", type=int, default=31)
    parser.add_argument("--max-m", type=int, default=12)
    parser.add_argument("--F1", type=float, default=0.7)
    parser.add_argument("--F2", type=float, default=0.4)
    parser.add_argument(
        "--deformation-profile",
        choices=["constant", "f_squared_floor", "inverse_f_squared_floor"],
        default="constant",
    )
    parser.add_argument("--deformation-reference-lat-deg", type=float, default=-60.0)
    parser.add_argument("--deformation-f-floor-sin", type=float, default=0.2)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--background-shear-velocity", type=float, default=1.5)
    parser.add_argument(
        "--background-profile",
        choices=["solid_body", "sin_plus_sin3"],
        default="sin_plus_sin3",
    )
    parser.add_argument("--background-sin3-weight", type=float, default=0.75)
    parser.add_argument("--mask-plateau-north-edge-deg", type=float, default=90.0)
    parser.add_argument("--mask-taper-north-edge-deg", type=float, default=91.0)
    parser.add_argument("--mask-linear-tendency", action="store_true")
    parser.add_argument("--sponge-max-rate", type=float, default=0.0)
    parser.add_argument("--amplitude", type=float, default=1e-6)
    parser.add_argument("--save-mode", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--allow-float32-fast", action="store_true")
    args = parser.parse_args()
    _guard_operator_combo(args.impl, args.dtype, args.allow_float32_fast)
    _configure_device(args.device, args.dtype)

    import jax
    import numpy as np

    from dinosaur_spike.two_layer_model import TwoLayerParams

    dtype = _dtype_from_name(args.dtype)
    grid = _make_grid(args.wavenumbers, args.impl)
    params = TwoLayerParams(
        F1=args.F1,
        F2=args.F2,
        deformation_profile=args.deformation_profile,
        deformation_reference_lat_deg=args.deformation_reference_lat_deg,
        deformation_f_floor_sin=args.deformation_f_floor_sin,
        omega=args.omega,
        sponge_max_rate=args.sponge_max_rate,
        background_shear_velocity=args.background_shear_velocity,
        background_profile=args.background_profile,
        background_sin3_weight=args.background_sin3_weight,
        mask_plateau_north_edge_deg=args.mask_plateau_north_edge_deg,
        mask_taper_north_edge_deg=args.mask_taper_north_edge_deg,
        mask_nonlinear_tendency=args.mask_linear_tendency,
    )
    if params.sponge_max_rate > 0.0 and not params.mask_nonlinear_tendency:
        raise SystemExit("--sponge-max-rate requires --mask-linear-tendency")
    if params.mask_nonlinear_tendency:
        from dinosaur_spike.two_layer_model import latitude_mask_nodal

        mask_nodal = latitude_mask_nodal(grid, params)
    else:
        mask_nodal = None

    print(f"devices={jax.devices()}")
    print(
        f"profile={args.background_profile} shear={args.background_shear_velocity} "
        f"wavenumbers={args.wavenumbers} deformation={args.deformation_profile}"
    )

    rows = []
    best = None
    for m in range(1, min(args.max_m, args.wavenumbers - 1) + 1):
        matrix, mode_rows, mode_ells = _matrix_for_m(grid, params, m, dtype, mask_nodal)
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        idx = int(np.argmax(eigenvalues.real))
        value = eigenvalues[idx]
        row = {
            "m": m,
            "growth_rate": float(value.real),
            "frequency": float(value.imag),
            "dimension": matrix.shape[0],
        }
        rows.append(row)
        if best is None or value.real > best["eigenvalue"].real:
            best = {
                "m": m,
                "row_indices": mode_rows,
                "ell_indices": mode_ells,
                "eigenvalue": value,
                "eigenvector": eigenvectors[:, idx],
            }
        print(
            f"m={m:3d} growth={value.real:.6e} frequency={value.imag:.6e}",
            flush=True,
        )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    assert best is not None
    print(
        "best "
        f"m={best['m']} growth={best['eigenvalue'].real:.6e} "
        f"frequency={best['eigenvalue'].imag:.6e}"
    )

    if args.save_mode is not None:
        vector = np.real(best["eigenvector"])
        if np.linalg.norm(vector) < 1e-14:
            vector = np.imag(best["eigenvector"])
        state = _unpack_vector(
            vector, grid, best["row_indices"], best["ell_indices"], dtype
        )
        state = _scale_state_to_amplitude(state, grid, args.amplitude)
        args.save_mode.parent.mkdir(parents=True, exist_ok=True)
        _save_mode(args.save_mode, state, grid, args, best["eigenvalue"], best["m"])
        print(args.save_mode)


if __name__ == "__main__":
    main()
