"""Diagnostics for the FD-in-z benchmark solver."""

from __future__ import annotations

import jax.numpy as jnp

from fd_vertical_benchmark.operators import Grid
from fd_vertical_benchmark.solver import (
    State,
    balanced_exchange_residual_fd,
    extend_dirichlet,
    extend_dirichlet_1d,
    q_from_psi,
    psi_full_from_state,
)


def shell_spectrum(field_full: jnp.ndarray, ksq: jnp.ndarray, z_weights: jnp.ndarray, L: float):
    """Depth-integrated horizontal shell spectrum."""
    Nk = field_full.shape[2]
    k_mag = jnp.sqrt(ksq)
    dk = 2.0 * jnp.pi / L
    k_max = jnp.sqrt(jnp.max(ksq))
    n_bins = int(float(k_max / dk)) + 1
    k_bins = jnp.arange(n_bins) * dk + dk / 2.0

    field_sq_int = jnp.einsum("j,j...->...", z_weights, jnp.abs(field_full) ** 2)
    weight = jnp.ones_like(ksq)
    if Nk > 2:
        weight = weight.at[:, 1:Nk - 1].set(2.0)

    spec = jnp.zeros(n_bins, dtype=field_full.real.dtype)
    for i in range(n_bins):
        mask = (k_mag >= i * dk) & (k_mag < (i + 1) * dk)
        spec = spec.at[i].set(jnp.sum(jnp.where(mask, field_sq_int * weight, 0.0)))
    return k_bins, spec


def compute_diagnostics(state: State, grid: Grid) -> dict:
    """Compute scalar and spectral diagnostics comparable to the main solver."""
    psi_full = psi_full_from_state(state.psi_hat, grid)
    q_full = q_from_psi(psi_full, grid.denom)
    w_full = extend_dirichlet(state.w_hat, grid.Nz)
    th_full = extend_dirichlet(state.th_hat, grid.Nz)
    th_bar_full = extend_dirichlet_1d(state.th_bar, grid.Nz)

    psi_bt = jnp.einsum("j,j...->...", grid.z_weights, psi_full)

    weight = jnp.ones_like(grid.ksq)
    if grid.Nk > 2:
        weight = weight.at[:, 1:grid.Nk - 1].set(2.0)
    norm = grid.Nx ** 4

    psi_sq_int = jnp.einsum("j,j...->...", grid.z_weights, jnp.abs(psi_full) ** 2)
    q_sq_int = jnp.einsum("j,j...->...", grid.z_weights, jnp.abs(q_full) ** 2)
    w_sq_int = jnp.einsum("j,j...->...", grid.z_weights, jnp.abs(w_full) ** 2)
    th_sq_int = jnp.einsum("j,j...->...", grid.z_weights, jnp.abs(th_full) ** 2)

    KE_bt = 0.5 * jnp.sum(grid.ksq * jnp.abs(psi_bt) ** 2 * weight) / norm
    KE_tot = 0.5 * jnp.sum(grid.ksq * psi_sq_int * weight) / norm
    KE_bc = KE_tot - KE_bt
    enstrophy = 0.5 * jnp.sum(q_sq_int * weight) / norm

    u_hat = -1j * grid.ky[None, :, :] * psi_full
    v_hat = 1j * grid.kx[None, :, :] * psi_full
    u_phys = jnp.fft.irfft2(u_hat, s=(grid.Nx, grid.Nx))
    v_phys = jnp.fft.irfft2(v_hat, s=(grid.Nx, grid.Nx))
    w_phys = jnp.fft.irfft2(w_full, s=(grid.Nx, grid.Nx))
    th_phys = jnp.fft.irfft2(th_full, s=(grid.Nx, grid.Nx))
    tw_phys = w_phys * th_phys

    max_speed = jnp.max(jnp.sqrt(u_phys ** 2 + v_phys ** 2))
    max_w = jnp.max(jnp.abs(w_phys))
    max_theta = jnp.max(jnp.abs(th_phys))
    max_tw = jnp.max(jnp.abs(tw_phys))

    wth_int = jnp.einsum("j,j...->...", grid.z_weights, jnp.real(w_full * jnp.conj(th_full)))
    vol_avg_tw = jnp.sum(wth_int * weight) / norm
    Nusselt = 1.0 + vol_avg_tw

    q_rms = jnp.sqrt(jnp.sum(q_sq_int * weight) / norm)
    w_rms = jnp.sqrt(jnp.sum(w_sq_int * weight) / norm)
    th_rms = jnp.sqrt(jnp.sum(th_sq_int * weight) / norm)

    q_z_power = jnp.sum(jnp.abs(q_full) ** 2 * weight[None, :, :], axis=(1, 2)) / norm
    w_z_power = jnp.sum(jnp.abs(w_full) ** 2 * weight[None, :, :], axis=(1, 2)) / norm
    th_z_power = jnp.sum(jnp.abs(th_full) ** 2 * weight[None, :, :], axis=(1, 2)) / norm

    k_bins, q_horiz_spec = shell_spectrum(q_full, grid.ksq, grid.z_weights, grid.L)
    _, w_horiz_spec = shell_spectrum(w_full, grid.ksq, grid.z_weights, grid.L)
    _, th_horiz_spec = shell_spectrum(th_full, grid.ksq, grid.z_weights, grid.L)

    return {
        "KE_bt": KE_bt,
        "KE_bc": KE_bc,
        "KE_tot": KE_tot,
        "enstrophy": enstrophy,
        "Nusselt": Nusselt,
        "vol_avg_tw": vol_avg_tw,
        "max_speed": max_speed,
        "max_w": max_w,
        "max_theta": max_theta,
        "max_tw": max_tw,
        "q_rms": q_rms,
        "w_rms": w_rms,
        "th_rms": th_rms,
        "th_bar_max": jnp.max(jnp.abs(th_bar_full)),
        "R_ex_sbp": balanced_exchange_residual_fd(state, grid),
        "q_z_power": q_z_power,
        "w_z_power": w_z_power,
        "th_z_power": th_z_power,
        "k_bins": k_bins,
        "q_horiz_spec": q_horiz_spec,
        "w_horiz_spec": w_horiz_spec,
        "th_horiz_spec": th_horiz_spec,
    }
