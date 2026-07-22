"""Dense azimuthal-Fourier / radial-Jacobi spectral transforms.

This is the first coupled horizontal transform layer for the spherical solver.
It uses a common radial product grid and dense projection matrices. The goal is
clear normalization and operator tests before introducing optimized transforms.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from scipy.special import roots_jacobi

from sphere_nhqg.geometry import inverse_conformal_factor
from sphere_nhqg.radial import basis_values_at_r, derivative_values_at_r


@dataclass(frozen=True)
class SpectralGrid:
    """Precomputed common-grid transforms for azimuth/radial spectral fields."""

    Nphi: int
    Nr: int
    r_jet: float
    Nphi_phys: int
    Nr_phys: int
    Nm: int
    Nm_phys: int
    xi: jnp.ndarray
    r: jnp.ndarray
    radial_weights: jnp.ndarray
    radial_vander: jnp.ndarray
    radial_derivative_vander: jnp.ndarray
    radial_projection: jnp.ndarray


def _dtype_pair(dtype) -> tuple[np.dtype, jnp.dtype]:
    jax_dtype = jnp.dtype(dtype)
    return np.dtype(jax_dtype.name), jax_dtype


def _validate_grid_sizes(Nphi: int, Nr: int, Nphi_phys: int, Nr_phys: int) -> None:
    if Nphi <= 0 or Nphi % 2 != 0:
        raise ValueError("Nphi must be a positive even integer")
    if Nr <= 0:
        raise ValueError("Nr must be positive")
    if Nphi_phys < Nphi or Nphi_phys % 2 != 0:
        raise ValueError("Nphi_phys must be an even integer >= Nphi")
    if Nr_phys < Nr:
        raise ValueError("Nr_phys must be >= Nr")


def make_spectral_grid(
    Nphi: int,
    Nr: int,
    r_jet: float,
    *,
    Nphi_phys: int | None = None,
    Nr_phys: int | None = None,
    dtype=jnp.float64,
) -> SpectralGrid:
    """Build dense common-radial-grid transforms for all nonnegative modes."""
    if r_jet <= 0.0:
        raise ValueError("r_jet must be positive")

    Nphi_phys = Nphi if Nphi_phys is None else Nphi_phys
    Nr_phys = Nr if Nr_phys is None else Nr_phys
    _validate_grid_sizes(Nphi, Nr, Nphi_phys, Nr_phys)

    np_dtype, jax_dtype = _dtype_pair(dtype)
    Nm = Nphi // 2 + 1
    Nm_phys = Nphi_phys // 2 + 1

    xi_np, legendre_weights_np = roots_jacobi(Nr_phys, 0.0, 0.0)
    xi_np = xi_np.astype(np_dtype)
    legendre_weights_np = legendre_weights_np.astype(np_dtype)
    r_np = (float(r_jet) * np.sqrt((1.0 + xi_np) / 2.0)).astype(np_dtype)
    flat_radial_weights_np = ((float(r_jet) ** 2) / 4.0 * legendre_weights_np).astype(
        np_dtype
    )

    radial_vander_np = np.empty((Nm, Nr_phys, Nr), dtype=np_dtype)
    radial_derivative_vander_np = np.empty((Nm, Nr_phys, Nr), dtype=np_dtype)
    radial_projection_np = np.empty((Nm, Nr, Nr_phys), dtype=np_dtype)

    W = np.diag(flat_radial_weights_np)
    for m in range(Nm):
        V = basis_values_at_r(m=m, Nr=Nr, r_jet=r_jet, r=r_np, dtype=np_dtype)
        dV = derivative_values_at_r(m=m, Nr=Nr, r_jet=r_jet, r=r_np, dtype=np_dtype)
        gram = V.T @ W @ V
        radial_vander_np[m] = V
        radial_derivative_vander_np[m] = dV
        radial_projection_np[m] = np.linalg.solve(gram, V.T @ W)

    return SpectralGrid(
        Nphi=Nphi,
        Nr=Nr,
        r_jet=float(r_jet),
        Nphi_phys=Nphi_phys,
        Nr_phys=Nr_phys,
        Nm=Nm,
        Nm_phys=Nm_phys,
        xi=jnp.asarray(xi_np, dtype=jax_dtype),
        r=jnp.asarray(r_np, dtype=jax_dtype),
        radial_weights=jnp.asarray(flat_radial_weights_np, dtype=jax_dtype),
        radial_vander=jnp.asarray(radial_vander_np, dtype=jax_dtype),
        radial_derivative_vander=jnp.asarray(radial_derivative_vander_np, dtype=jax_dtype),
        radial_projection=jnp.asarray(radial_projection_np, dtype=jax_dtype),
    )


def _mode_values_to_physical(mode_values: jnp.ndarray, grid: SpectralGrid) -> jnp.ndarray:
    pad_modes = grid.Nm_phys - grid.Nm
    if pad_modes < 0:
        raise ValueError("physical mode count cannot be smaller than coefficient mode count")
    if pad_modes:
        pad_width = [(0, pad_modes), (0, 0)] + [(0, 0)] * (mode_values.ndim - 2)
        mode_values = jnp.pad(mode_values, pad_width)

    phi_first = jnp.fft.irfft(mode_values, n=grid.Nphi_phys, axis=0)
    phi_first = phi_first * (grid.Nphi_phys / grid.Nphi)
    return jnp.moveaxis(phi_first, 0, 1)


def coefficients_to_physical(coeffs: jnp.ndarray, grid: SpectralGrid) -> jnp.ndarray:
    """Transform coefficients ``(Nm, Nr, ...)`` to physical ``(r, phi, ...)``."""
    if coeffs.shape[0] != grid.Nm or coeffs.shape[1] != grid.Nr:
        raise ValueError("coeffs must have shape (grid.Nm, grid.Nr, ...)")

    mode_values = jnp.einsum("mjn,mn...->mj...", grid.radial_vander, coeffs)
    return _mode_values_to_physical(mode_values, grid)


def physical_to_coefficients(values: jnp.ndarray, grid: SpectralGrid) -> jnp.ndarray:
    """Project physical values ``(r, phi, ...)`` to coefficients ``(Nm, Nr, ...)``."""
    if values.shape[0] != grid.Nr_phys or values.shape[1] != grid.Nphi_phys:
        raise ValueError("values must have shape (grid.Nr_phys, grid.Nphi_phys, ...)")

    phi_first = jnp.moveaxis(values, 1, 0)
    modes = jnp.fft.rfft(phi_first, n=grid.Nphi_phys, axis=0)
    modes = modes[: grid.Nm, ...] * (grid.Nphi / grid.Nphi_phys)
    return jnp.einsum("mnj,mj...->mn...", grid.radial_projection, modes)


def constant_coefficients(
    value: float | complex,
    grid: SpectralGrid,
    *,
    dtype=jnp.complex128,
) -> jnp.ndarray:
    """Coefficient array representing a physical constant on the disk."""
    coeffs = jnp.zeros((grid.Nm, grid.Nr), dtype=dtype)
    return coeffs.at[0, 0].set(jnp.asarray(value, dtype=dtype) * grid.Nphi)


def dealiased_product(
    a_coeffs: jnp.ndarray,
    b_coeffs: jnp.ndarray,
    grid: SpectralGrid,
) -> jnp.ndarray:
    """Compute a truncated product through the grid's overresolved physical space."""
    a_phys = coefficients_to_physical(a_coeffs, grid)
    b_phys = coefficients_to_physical(b_coeffs, grid)
    return physical_to_coefficients(a_phys * b_phys, grid)


def radial_derivative_to_physical(coeffs: jnp.ndarray, grid: SpectralGrid) -> jnp.ndarray:
    """Evaluate ``partial_r`` of a coefficient field on the physical grid."""
    if coeffs.shape[0] != grid.Nm or coeffs.shape[1] != grid.Nr:
        raise ValueError("coeffs must have shape (grid.Nm, grid.Nr, ...)")

    mode_values = jnp.einsum("mjn,mn...->mj...", grid.radial_derivative_vander, coeffs)
    return _mode_values_to_physical(mode_values, grid)


def azimuthal_derivative_to_physical(coeffs: jnp.ndarray, grid: SpectralGrid) -> jnp.ndarray:
    """Evaluate ``partial_phi`` of a coefficient field on the physical grid."""
    if coeffs.shape[0] != grid.Nm or coeffs.shape[1] != grid.Nr:
        raise ValueError("coeffs must have shape (grid.Nm, grid.Nr, ...)")

    mode_shape = (grid.Nm,) + (1,) * (coeffs.ndim - 1)
    modes = jnp.arange(grid.Nm, dtype=coeffs.real.dtype).reshape(mode_shape)
    mode_values = jnp.einsum("mjn,mn...->mj...", grid.radial_vander, 1j * modes * coeffs)
    return _mode_values_to_physical(mode_values, grid)


def flat_jacobian_physical(
    a_coeffs: jnp.ndarray,
    b_coeffs: jnp.ndarray,
    grid: SpectralGrid,
) -> jnp.ndarray:
    """Evaluate J_flat(a,b) = (a_r b_phi - a_phi b_r) / r on the product grid."""
    a_r = radial_derivative_to_physical(a_coeffs, grid)
    a_phi = azimuthal_derivative_to_physical(a_coeffs, grid)
    b_r = radial_derivative_to_physical(b_coeffs, grid)
    b_phi = azimuthal_derivative_to_physical(b_coeffs, grid)
    radius_shape = (grid.Nr_phys,) + (1,) * (a_r.ndim - 1)
    radius = grid.r.reshape(radius_shape)
    return (a_r * b_phi - a_phi * b_r) / radius


def spherical_jacobian_physical(
    a_coeffs: jnp.ndarray,
    b_coeffs: jnp.ndarray,
    grid: SpectralGrid,
) -> jnp.ndarray:
    """Evaluate J_sphere(a,b) = mu(r)^-1 J_flat(a,b) on the product grid."""
    flat = flat_jacobian_physical(a_coeffs, b_coeffs, grid)
    metric_shape = (grid.Nr_phys,) + (1,) * (flat.ndim - 1)
    mu_inv = inverse_conformal_factor(grid.r).reshape(metric_shape)
    return mu_inv * flat


def flat_jacobian_coefficients(
    a_coeffs: jnp.ndarray,
    b_coeffs: jnp.ndarray,
    grid: SpectralGrid,
) -> jnp.ndarray:
    """Evaluate and project the flat polar Jacobian to retained coefficients."""
    return physical_to_coefficients(flat_jacobian_physical(a_coeffs, b_coeffs, grid), grid)


def spherical_jacobian_coefficients(
    a_coeffs: jnp.ndarray,
    b_coeffs: jnp.ndarray,
    grid: SpectralGrid,
) -> jnp.ndarray:
    """Evaluate and project the spherical Jacobian to retained coefficients."""
    return physical_to_coefficients(spherical_jacobian_physical(a_coeffs, b_coeffs, grid), grid)
