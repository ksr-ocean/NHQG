"""Diagnostics for the trigonometric vertical benchmark solver."""

from __future__ import annotations

import jax.numpy as jnp

from trig_vertical_benchmark.operators import Grid
from trig_vertical_benchmark.solver import (
    State,
    _horizontal_mean_product_dealiased,
    d1_sin_to_work,
    eval_cos,
    eval_sin,
    explicit_rhs,
    implicit_tendency,
    q_from_psi,
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


def _shell_bins(ksq: jnp.ndarray, L: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    k_mag = jnp.sqrt(ksq)
    dk = 2.0 * jnp.pi / L
    k_max = jnp.sqrt(jnp.max(ksq))
    n_bins = int(float(k_max / dk)) + 1
    k_bins = jnp.arange(n_bins) * dk + dk / 2.0
    return k_mag, dk, k_bins


def _dealiased_shell_flux_profiles(
    w_full: jnp.ndarray,
    th_full: jnp.ndarray,
    grid: Grid,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Shell-filtered dealiased mean heat-flux profiles on the work grid."""
    k_mag, dk, k_bins = _shell_bins(grid.ksq, grid.L)
    profiles = []
    for i in range(k_bins.shape[0]):
        mask = ((k_mag >= i * dk) & (k_mag < (i + 1) * dk))[None, :, :]
        w_shell = jnp.where(mask, w_full, 0.0)
        th_shell = jnp.where(mask, th_full, 0.0)
        profiles.append(
            _horizontal_mean_product_dealiased(w_shell, th_shell, grid.Nx, grid.Npad)
        )
    flux_shell_profiles = jnp.stack(profiles, axis=0)
    flux_profile = jnp.sum(flux_shell_profiles, axis=0)
    return k_bins, flux_shell_profiles, flux_profile


def compute_dealiased_thermal_budgets(
    state: State,
    grid: Grid,
    w_full: jnp.ndarray,
    th_full: jnp.ndarray,
) -> dict:
    """Mean thermal diagnostics built from the same dealiased flux path as the solver."""
    k_bins, flux_shell_profiles, flux_profile = _dealiased_shell_flux_profiles(
        w_full, th_full, grid
    )
    dth_bar_full = d1_sin_to_work(state.th_bar, grid)

    heat_flux_shell = jnp.einsum("j,ij->i", grid.z_weights, flux_shell_profiles)
    th_conduction_shell = heat_flux_shell
    w_buoyancy_shell = grid.Ra_sigma * heat_flux_shell
    th_mean_feedback_shell = -jnp.einsum(
        "j,ij,j->i", grid.z_weights, flux_shell_profiles, dth_bar_full
    )

    return {
        "k_bins": k_bins,
        "flux_profile_dealiased": flux_profile,
        "heat_flux_shell_dealiased": heat_flux_shell,
        "th_conduction_shell_tendency_dealiased": th_conduction_shell,
        "w_buoyancy_shell_tendency_dealiased": w_buoyancy_shell,
        "th_mean_feedback_shell_tendency_dealiased": th_mean_feedback_shell,
        "heat_flux_sum_dealiased": jnp.sum(heat_flux_shell),
        "th_conduction_sum_dealiased": jnp.sum(th_conduction_shell),
        "w_buoyancy_sum_dealiased": jnp.sum(w_buoyancy_shell),
        "th_mean_feedback_sum_dealiased": jnp.sum(th_mean_feedback_shell),
    }


def compute_mean_temperature_budget(
    state: State,
    grid: Grid,
    thermal_dealiased: dict | None = None,
) -> dict:
    """Global mean-temperature diagnostics and discrete exchange residual."""
    th_bar_full = eval_sin(state.th_bar, grid.S_eval)
    dth_bar_full = d1_sin_to_work(state.th_bar, grid)
    mean_grad = 1.0 - dth_bar_full

    explicit = explicit_rhs(state, grid)
    implicit = implicit_tendency(state, grid)
    explicit_full = eval_sin(explicit.th_bar, grid.S_eval)
    implicit_full = eval_sin(implicit.th_bar, grid.S_eval)

    eps_sq = float(grid.mean_temp_eps_sq)
    if grid.thermal_closure == "evolve_mean" and abs(eps_sq) > 0.0:
        prefac = jnp.asarray(1.0 / eps_sq, dtype=th_bar_full.dtype)
        mean_energy = 0.5 * prefac * jnp.sum(grid.z_weights * th_bar_full ** 2)
        mean_flux_exchange = prefac * jnp.sum(grid.z_weights * th_bar_full * explicit_full)
        mean_diffusion = prefac * jnp.sum(grid.z_weights * th_bar_full * implicit_full)
        mean_total = mean_flux_exchange + mean_diffusion
    else:
        mean_energy = jnp.asarray(0.0, dtype=th_bar_full.dtype)
        mean_flux_exchange = jnp.asarray(0.0, dtype=th_bar_full.dtype)
        mean_diffusion = jnp.asarray(0.0, dtype=th_bar_full.dtype)
        mean_total = jnp.asarray(0.0, dtype=th_bar_full.dtype)

    theta_mean_feedback_dealiased = (
        thermal_dealiased["th_mean_feedback_sum_dealiased"]
        if thermal_dealiased is not None
        and "th_mean_feedback_sum_dealiased" in thermal_dealiased
        else jnp.asarray(0.0, dtype=th_bar_full.dtype)
    )
    exchange_residual_dealiased = theta_mean_feedback_dealiased + mean_flux_exchange
    exchange_scale_dealiased = jnp.maximum(
        jnp.abs(theta_mean_feedback_dealiased) + jnp.abs(mean_flux_exchange),
        jnp.asarray(1e-300, dtype=th_bar_full.dtype),
    )
    mid_idx = int(jnp.argmin(jnp.abs(grid.z_work - 0.5)))

    return {
        "th_bar_phys_max": jnp.max(jnp.abs(th_bar_full)),
        "dth_bar_dz_max": jnp.max(jnp.abs(dth_bar_full)),
        "mean_grad_min": jnp.min(mean_grad),
        "mean_grad_max": jnp.max(mean_grad),
        "mean_grad_mid": mean_grad[mid_idx],
        "mean_energy": mean_energy,
        "mean_flux_exchange_tendency": mean_flux_exchange,
        "mean_diffusion_tendency": mean_diffusion,
        "mean_total_tendency": mean_total,
        "mean_theta_exchange_residual_dealiased": exchange_residual_dealiased,
        "mean_theta_exchange_residual_dealiased_rel": (
            exchange_residual_dealiased / exchange_scale_dealiased
        ),
    }


def compute_diagnostics(state: State, grid: Grid) -> dict:
    """Compute scalar and spectral diagnostics comparable to the main solver."""
    psi_full = eval_cos(state.psi_hat, grid.C_eval)
    q_full = q_from_psi(psi_full, grid.denom)
    w_full = eval_sin(state.w_hat, grid.S_eval)
    th_full = eval_sin(state.th_hat, grid.S_eval)
    th_bar_full = eval_sin(state.th_bar, grid.S_eval)

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
    flux_profile_dealiased = _horizontal_mean_product_dealiased(w_full, th_full, grid.Nx, grid.Npad)
    vol_avg_tw_dealiased = jnp.sum(grid.z_weights * flux_profile_dealiased)
    Nusselt = 1.0 + vol_avg_tw
    Nusselt_dealiased = 1.0 + vol_avg_tw_dealiased

    q_rms = jnp.sqrt(jnp.sum(q_sq_int * weight) / norm)
    w_rms = jnp.sqrt(jnp.sum(w_sq_int * weight) / norm)
    th_rms = jnp.sqrt(jnp.sum(th_sq_int * weight) / norm)

    q_z_power = jnp.sum(jnp.abs(q_full) ** 2 * weight[None, :, :], axis=(1, 2)) / norm
    w_z_power = jnp.sum(jnp.abs(w_full) ** 2 * weight[None, :, :], axis=(1, 2)) / norm
    th_z_power = jnp.sum(jnp.abs(th_full) ** 2 * weight[None, :, :], axis=(1, 2)) / norm

    k_bins, q_horiz_spec = shell_spectrum(q_full, grid.ksq, grid.z_weights, grid.L)
    _, w_horiz_spec = shell_spectrum(w_full, grid.ksq, grid.z_weights, grid.L)
    _, th_horiz_spec = shell_spectrum(th_full, grid.ksq, grid.z_weights, grid.L)
    thermal_dealiased = compute_dealiased_thermal_budgets(state, grid, w_full, th_full)
    mean_budget = compute_mean_temperature_budget(state, grid, thermal_dealiased)

    return {
        "KE_bt": KE_bt,
        "KE_bc": KE_bc,
        "KE_tot": KE_tot,
        "enstrophy": enstrophy,
        "Nusselt": Nusselt,
        "Nusselt_dealiased": Nusselt_dealiased,
        "vol_avg_tw": vol_avg_tw,
        "vol_avg_tw_dealiased": vol_avg_tw_dealiased,
        "heat_flux_mismatch": vol_avg_tw_dealiased - vol_avg_tw,
        "max_speed": max_speed,
        "max_w": max_w,
        "max_theta": max_theta,
        "max_tw": max_tw,
        "q_rms": q_rms,
        "w_rms": w_rms,
        "th_rms": th_rms,
        "th_bar_max": jnp.max(jnp.abs(th_bar_full)),
        "q_z_power": q_z_power,
        "w_z_power": w_z_power,
        "th_z_power": th_z_power,
        "k_bins": k_bins,
        "q_horiz_spec": q_horiz_spec,
        "w_horiz_spec": w_horiz_spec,
        "th_horiz_spec": th_horiz_spec,
        **thermal_dealiased,
        **mean_budget,
    }
