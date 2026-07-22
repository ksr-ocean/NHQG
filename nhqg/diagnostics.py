"""Diagnostic quantities: spectra, integral diagnostics, and KE shell budgets."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nhqg.grid import Grid
from nhqg.spectral import _zero_pad
from nhqg.solver import (
    State,
    _cheb_to_dirichlet,
    _to_coeffs,
    _to_coeffs_1d,
    _to_nodal_1d,
    _truncate_cheb_coeffs,
    _dirichlet_to_cheb,
    _to_nodal,
    explicit_rhs_dispatch,
    horizontal_mean_from_nodal_spectral,
    horizontal_mean_wtheta,
    implicit_tendency,
    invert_psi,
    paired_theta_feedback_from_work,
    project_dirichlet,
    sbp2_exchange_state_fields,
    sbp2_flux_profile_nodal,
    sbp2_mean_rhs_nodal,
    sbp2_theta_feedback_cheb,
    thermal_exchange_workgrid_fields,
    thermal_exchange_workgrid_coeffs,
    uses_balanced_midpoint_exchange,
    uses_balanced_sbp2_exchange,
    uses_coral_exchange_workgrid,
    uses_paired_mean_exchange,
)


def barotropic_mode(field_nodal: jnp.ndarray, cc_weights: jnp.ndarray) -> jnp.ndarray:
    """Depth-averaged (barotropic) field via CC quadrature on nodal values.

    field_nodal: (Nz+1, Nx, Nk), cc_weights: (Nz+1,)
    Returns: (Nx, Nk)
    """
    return jnp.einsum('j,j...->...', cc_weights, field_nodal)


def _horizontal_rfft_weight(ksq: jnp.ndarray) -> jnp.ndarray:
    """rfft2 Parseval weights for the half-plane horizontal spectrum."""
    Nk = ksq.shape[1]
    weight = jnp.ones_like(ksq)
    if Nk > 2:
        weight = weight.at[:, 1:Nk - 1].set(2.0)
    return weight


def _shell_bins(ksq: jnp.ndarray, L: float,
                n_bins: int | None = None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Horizontal shell metadata for shell-binned spectra."""
    k_mag = jnp.sqrt(ksq)
    dk = 2.0 * jnp.pi / L
    k_max = jnp.sqrt(jnp.max(ksq))
    if n_bins is None:
        n_bins = int(float(k_max / dk)) + 1
    k_bins = jnp.arange(n_bins) * dk + dk / 2
    return k_mag, dk, k_bins


def _shell_bin_sum(mode_quantity: jnp.ndarray, ksq: jnp.ndarray, L: float,
                   n_bins: int | None = None) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sum a 2D mode quantity into horizontal wavenumber shells."""
    k_mag, dk, k_bins = _shell_bins(ksq, L, n_bins=n_bins)
    spec = jnp.zeros_like(k_bins)
    for i in range(k_bins.shape[0]):
        mask = (k_mag >= i * dk) & (k_mag < (i + 1) * dk)
        spec = spec.at[i].set(jnp.sum(jnp.where(mask, mode_quantity, 0.0)))
    return k_bins, spec


def energy_spectrum(psi_nodal: jnp.ndarray, ksq: jnp.ndarray,
                    cc_weights: jnp.ndarray, L: float,
                    n_bins: int = None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute barotropic, baroclinic, and total kinetic energy spectra.

    psi_nodal: (Nz+1, Nx, Nk) — psi at CGL nodes.
    Returns: k_bins, E_bt(k), E_bc(k), E_tot(k)
    """
    Nk = psi_nodal.shape[2]
    psi_bt = barotropic_mode(psi_nodal, cc_weights)

    E_bt_power = 0.5 * ksq * jnp.abs(psi_bt) ** 2

    psi_sq_int = jnp.einsum('j,j...->...', cc_weights, jnp.abs(psi_nodal) ** 2)
    E_tot_power = 0.5 * ksq * psi_sq_int

    weight = _horizontal_rfft_weight(ksq)
    k_bins, E_bt = _shell_bin_sum(E_bt_power * weight, ksq, L, n_bins=n_bins)
    _, E_tot = _shell_bin_sum(E_tot_power * weight, ksq, L, n_bins=n_bins)

    E_bc = E_tot - E_bt
    return k_bins, E_bt, E_bc, E_tot


def shell_spectrum(field_nodal: jnp.ndarray, ksq: jnp.ndarray,
                   cc_weights: jnp.ndarray, L: float,
                   n_bins: int = None) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Depth-integrated shell spectrum of a horizontally spectral field."""
    field_sq_int = jnp.einsum('j,j...->...', cc_weights, jnp.abs(field_nodal) ** 2)
    weight = _horizontal_rfft_weight(ksq)
    k_bins, spec = _shell_bin_sum(field_sq_int * weight, ksq, L, n_bins=n_bins)
    return k_bins, spec


def _ke_shell_tendency_from_q_term(psi_nodal: jnp.ndarray,
                                   q_term_nodal: jnp.ndarray,
                                   grid: Grid) -> jnp.ndarray:
    """Horizontal-shell KE tendency induced by a q' tendency term.

    The shell density is computed from

      d/dt (0.5 |grad_h psi|^2) = -(|k|^2 / denom) Re[psi* q_t]

    depth-integrated with CC weights and horizontally normalized with the
    same rfft Parseval factors used in the existing KE diagnostics.
    """
    integrand = jnp.real(jnp.conj(psi_nodal) * q_term_nodal)
    depth_int = jnp.einsum('j,j...->...', grid.cc_weights, integrand)
    mode_tendency = -(grid.ksq * grid.inv_denom) * depth_int
    weighted = mode_tendency * _horizontal_rfft_weight(grid.ksq) / (grid.Nx ** 4)
    _, shell_tendency = _shell_bin_sum(weighted, grid.ksq, float(grid.L))
    return shell_tendency


def _quadratic_shell_tendency(field_nodal: jnp.ndarray,
                              term_nodal: jnp.ndarray,
                              grid: Grid) -> jnp.ndarray:
    """Shell tendency of the quadratic density 0.5*|field|^2."""
    integrand = jnp.real(jnp.conj(field_nodal) * term_nodal)
    depth_int = jnp.einsum('j,j...->...', grid.cc_weights, integrand)
    weighted = depth_int * _horizontal_rfft_weight(grid.ksq) / (grid.Nx ** 4)
    _, shell_tendency = _shell_bin_sum(weighted, grid.ksq, float(grid.L))
    return shell_tendency


def _theta_mean_feedback_cheb(state: State, grid: Grid) -> jnp.ndarray:
    """Theta explicit source from the evolving mean-temperature gradient."""
    w_cheb = _dirichlet_to_cheb(state.w_hat, grid.dirichlet_stencil)
    zeros = jnp.zeros_like(w_cheb)
    if grid.thermal_closure != "evolve_mean":
        return zeros
    if uses_balanced_sbp2_exchange(grid):
        return sbp2_theta_feedback_cheb(state, grid)
    if uses_balanced_midpoint_exchange(grid):
        # This mode applies thermal coupling in a separate midpoint substep,
        # not through explicit RHS forcing.
        return zeros
    if uses_paired_mean_exchange(grid):
        w_work, _, dth_bar_dZ_work = thermal_exchange_workgrid_fields(state, grid)
        feedback_gal = paired_theta_feedback_from_work(
            dth_bar_dZ_work[:, None, None] * w_work, grid
        )
        return _dirichlet_to_cheb(feedback_gal, grid.dirichlet_stencil)
    if uses_coral_exchange_workgrid(grid):
        product_coeffs, _ = thermal_exchange_workgrid_coeffs(state, grid)
        return project_dirichlet(-product_coeffs, grid.proj_dirichlet)

    dth_bar_dZ_coeffs = grid.G_Z @ state.th_bar
    if grid.vertical_dealiasing == "none":
        dth_bar_dZ_nodal = _to_nodal_1d(dth_bar_dZ_coeffs, grid.V)
        w_nodal = _to_nodal(w_cheb, grid.V)
        product_coeffs = _to_coeffs(
            dth_bar_dZ_nodal[:, None, None] * w_nodal, grid.V_inv
        )
    elif grid.vertical_dealiasing in {"cheb_3o2", "cheb_2x"}:
        dth_bar_dZ_nodal = _to_nodal_1d(dth_bar_dZ_coeffs, grid.V_dealias)
        w_nodal = _to_nodal(w_cheb, grid.V_dealias)
        product_hi = _to_coeffs(
            dth_bar_dZ_nodal[:, None, None] * w_nodal, grid.V_dealias_inv
        )
        product_coeffs = _truncate_cheb_coeffs(product_hi, grid.Nz)
    else:
        raise ValueError(f"Unsupported vertical_dealiasing={grid.vertical_dealiasing!r}")

    return project_dirichlet(-product_coeffs, grid.proj_dirichlet)


def _horizontal_mean_product_from_nodal_spectral(a_nodal: jnp.ndarray,
                                                 b_nodal: jnp.ndarray,
                                                 grid: Grid) -> jnp.ndarray:
    """Dealiased horizontal mean of a nodal spectral product as a function of z."""
    Nx = grid.Nx
    Npad = grid.Npad
    if Npad is None or Npad == Nx:
        a_phys = jnp.fft.irfft2(a_nodal, s=(Nx, Nx))
        b_phys = jnp.fft.irfft2(b_nodal, s=(Nx, Nx))
    else:
        pad_one = lambda field: _zero_pad(field, Nx, Npad)
        a_pad = jax.vmap(pad_one)(a_nodal)
        b_pad = jax.vmap(pad_one)(b_nodal)
        scale = (Npad / Nx) ** 2
        a_phys = scale * jnp.fft.irfft2(a_pad, s=(Npad, Npad))
        b_phys = scale * jnp.fft.irfft2(b_pad, s=(Npad, Npad))
    return jnp.mean(a_phys * b_phys, axis=(1, 2))


def _solver_mean_flux_profile_nodal(state: State, grid: Grid) -> jnp.ndarray:
    """Mean heat-flux profile using the same path as the mean equation."""
    if uses_balanced_sbp2_exchange(grid):
        return sbp2_flux_profile_nodal(state, grid)
    if uses_coral_exchange_workgrid(grid):
        _, flux_coeffs = thermal_exchange_workgrid_coeffs(state, grid)
        return _to_nodal_1d(flux_coeffs, grid.V)
    if grid.vertical_dealiasing == "none":
        return horizontal_mean_wtheta(
            state.w_hat, state.th_hat, grid.V, grid.dirichlet_stencil, grid.Nx, grid.Npad
        )
    if grid.vertical_dealiasing in {"cheb_3o2", "cheb_2x"}:
        flux_nodal_hi = horizontal_mean_wtheta(
            state.w_hat, state.th_hat, grid.V_dealias, grid.dirichlet_stencil, grid.Nx, grid.Npad
        )
        flux_coeffs_hi = _to_coeffs_1d(flux_nodal_hi, grid.V_dealias_inv)
        flux_coeffs = _truncate_cheb_coeffs(flux_coeffs_hi, grid.Nz)
        return _to_nodal_1d(flux_coeffs, grid.V)
    raise ValueError(f"Unsupported vertical_dealiasing={grid.vertical_dealiasing!r}")


def _dealiased_shell_flux_profiles(w_nodal: jnp.ndarray,
                                   th_nodal: jnp.ndarray,
                                   grid: Grid,
                                   state: State | None = None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Shell-filtered dealiased mean heat-flux profiles."""
    k_mag, dk, k_bins = _shell_bins(grid.ksq, float(grid.L))
    if uses_balanced_sbp2_exchange(grid):
        w_sbp = jnp.einsum('ij,j...->i...', grid.cgl_to_sbp, w_nodal)
        th_sbp = jnp.einsum('ij,j...->i...', grid.cgl_to_sbp, th_nodal)
        profiles = []
        for i in range(k_bins.shape[0]):
            mask = ((k_mag >= i * dk) & (k_mag < (i + 1) * dk))[None, :, :]
            w_shell = jnp.where(mask, w_sbp, 0.0)
            th_shell = jnp.where(mask, th_sbp, 0.0)
            flux_sbp = horizontal_mean_from_nodal_spectral(
                w_shell, th_shell, grid.Nx, grid.Npad
            )
            profiles.append(jnp.einsum('ij,j->i', grid.sbp_to_cgl, flux_sbp))
        flux_shell_profiles = jnp.stack(profiles, axis=0)
        flux_profile = jnp.sum(flux_shell_profiles, axis=0)
        return k_bins, flux_shell_profiles, flux_profile
    if uses_coral_exchange_workgrid(grid):
        if state is None:
            raise ValueError("state is required for Coral work-grid shell flux profiles")
        w_cheb = _dirichlet_to_cheb(state.w_hat, grid.dirichlet_stencil)
        th_cheb = _dirichlet_to_cheb(state.th_hat, grid.dirichlet_stencil)
        w_work = _to_nodal(w_cheb, grid.V_exchange)
        th_work = _to_nodal(th_cheb, grid.V_exchange)
        profiles = []
        for i in range(k_bins.shape[0]):
            mask = ((k_mag >= i * dk) & (k_mag < (i + 1) * dk))[None, :, :]
            w_shell = jnp.where(mask, w_work, 0.0)
            th_shell = jnp.where(mask, th_work, 0.0)
            flux_work = horizontal_mean_from_nodal_spectral(
                w_shell, th_shell, grid.Nx, grid.Npad
            )
            flux_coeffs_hi = _to_coeffs_1d(flux_work, grid.V_exchange_inv)
            flux_coeffs = _truncate_cheb_coeffs(flux_coeffs_hi, grid.Nz)
            profiles.append(_to_nodal_1d(flux_coeffs, grid.V))
        flux_shell_profiles = jnp.stack(profiles, axis=0)
        flux_profile = jnp.sum(flux_shell_profiles, axis=0)
        return k_bins, flux_shell_profiles, flux_profile

    profiles = []
    for i in range(k_bins.shape[0]):
        mask = ((k_mag >= i * dk) & (k_mag < (i + 1) * dk))[None, :, :]
        w_shell = jnp.where(mask, w_nodal, 0.0)
        th_shell = jnp.where(mask, th_nodal, 0.0)
        profiles.append(_horizontal_mean_product_from_nodal_spectral(w_shell, th_shell, grid))
    flux_shell_profiles = jnp.stack(profiles, axis=0)
    flux_profile = jnp.sum(flux_shell_profiles, axis=0)
    return k_bins, flux_shell_profiles, flux_profile


def compute_dealiased_thermal_shell_budgets(state: State, grid: Grid,
                                            w_nodal: jnp.ndarray,
                                            th_nodal: jnp.ndarray) -> dict:
    """Thermal shell budgets built from the same dealiased flux path as the mean equation."""
    _, flux_shell_profiles, flux_profile = _dealiased_shell_flux_profiles(
        w_nodal, th_nodal, grid, state=state
    )
    dth_bar_dz_nodal = _to_nodal_1d(grid.G_Z @ state.th_bar, grid.V)

    heat_flux_shell = jnp.einsum('j,ij->i', grid.cc_weights, flux_shell_profiles)
    th_conduction_shell = heat_flux_shell
    w_buoyancy_shell = grid.Ra_sigma * heat_flux_shell
    if uses_paired_mean_exchange(grid):
        k_mag, dk, k_bins = _shell_bins(grid.ksq, float(grid.L))
        w_cheb = _dirichlet_to_cheb(state.w_hat, grid.dirichlet_stencil)
        w_work, _, dth_bar_dZ_work = thermal_exchange_workgrid_fields(state, grid)
        th_mean_feedback_entries = []
        for i in range(k_bins.shape[0]):
            mask = ((k_mag >= i * dk) & (k_mag < (i + 1) * dk))[None, :, :]
            w_shell = jnp.where(mask, w_work, 0.0)
            feedback_gal = paired_theta_feedback_from_work(
                dth_bar_dZ_work[:, None, None] * w_shell, grid
            )
            feedback_cheb = _dirichlet_to_cheb(feedback_gal, grid.dirichlet_stencil)
            feedback_nodal = _to_nodal(feedback_cheb, grid.V)
            th_mean_feedback_entries.append(jnp.sum(_quadratic_shell_tendency(
                th_nodal, feedback_nodal, grid
            )))
        th_mean_feedback_shell = jnp.stack(th_mean_feedback_entries)
    else:
        th_mean_feedback_shell = -jnp.einsum(
            'j,ij,j->i', grid.cc_weights, flux_shell_profiles, dth_bar_dz_nodal
        )

    return {
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


def compute_sbp_internal_exchange_budget(state: State, grid: Grid) -> dict:
    """Exchange monitor built entirely on the SBP grid and SBP norm.

    For SBP-based mean-exchange modes, the fluctuation-side exchange term and
    the mean-side exchange term should cancel under the same SBP norm/operator
    pair up to the boundary contribution. This diagnostic isolates that
    internal algebra from the CGL/CC transfer and monitoring layer.
    """
    dtype = state.th_bar.dtype
    if grid.thermal_closure != "evolve_mean" or not uses_balanced_sbp2_exchange(grid):
        nan = jnp.asarray(jnp.nan, dtype=dtype)
        return {
            "th_mean_feedback_sum_sbp": nan,
            "mean_flux_exchange_tendency_sbp": nan,
            "mean_theta_exchange_boundary_sbp": nan,
            "mean_theta_exchange_residual_sbp": nan,
            "mean_theta_exchange_residual_sbp_rel": nan,
        }

    w_sbp, th_sbp, th_bar_sbp, dth_bar_dz_sbp = sbp2_exchange_state_fields(state, grid)
    flux_sbp = horizontal_mean_from_nodal_spectral(w_sbp, th_sbp, grid.Nx, grid.Npad)

    H_flux = grid.sbp_H @ flux_sbp
    d_flux = grid.sbp_D1 @ flux_sbp
    theta_mean_feedback = -(dth_bar_dz_sbp @ H_flux)
    mean_flux_exchange = -(th_bar_sbp @ (grid.sbp_H @ d_flux))
    boundary = th_bar_sbp[-1] * flux_sbp[-1] - th_bar_sbp[0] * flux_sbp[0]
    residual = theta_mean_feedback + mean_flux_exchange
    scale = jnp.maximum(
        jnp.abs(theta_mean_feedback) + jnp.abs(mean_flux_exchange),
        jnp.asarray(1e-300, dtype=dtype),
    )

    return {
        "th_mean_feedback_sum_sbp": theta_mean_feedback,
        "mean_flux_exchange_tendency_sbp": mean_flux_exchange,
        "mean_theta_exchange_boundary_sbp": boundary,
        "mean_theta_exchange_residual_sbp": residual,
        "mean_theta_exchange_residual_sbp_rel": residual / scale,
    }


def _eval_cheb_series(coeffs: jnp.ndarray, z: float) -> jnp.ndarray:
    """Evaluate a 1D Chebyshev series on [0, 1] at a single point."""
    xi = jnp.asarray(2.0 * z - 1.0, dtype=coeffs.dtype)
    theta = jnp.arccos(jnp.clip(xi, -1.0, 1.0))
    n = jnp.arange(coeffs.shape[0], dtype=coeffs.dtype)
    basis = jnp.cos(theta * n)
    return jnp.dot(coeffs, basis)


def compute_mean_temperature_budget(state: State, grid: Grid,
                                    theta_budget: dict | None = None,
                                    thermal_dealiased: dict | None = None) -> dict:
    """Global mean-temperature diagnostics and exchange residual."""
    th_bar_nodal = _to_nodal_1d(state.th_bar, grid.V)
    dth_bar_dz_coeffs = grid.G_Z @ state.th_bar
    dth_bar_dz_nodal = _to_nodal_1d(dth_bar_dz_coeffs, grid.V)
    mean_grad_nodal = 1.0 - dth_bar_dz_nodal

    if uses_balanced_sbp2_exchange(grid):
        explicit_nodal, implicit_nodal = sbp2_mean_rhs_nodal(state, grid)
    else:
        explicit = explicit_rhs_dispatch(state, grid)
        implicit = implicit_tendency(state, grid)
        explicit_nodal = _to_nodal_1d(explicit.th_bar, grid.V)
        implicit_nodal = _to_nodal_1d(implicit.th_bar, grid.V)

    eps_sq = float(grid.mean_temp_eps_sq)
    if grid.thermal_closure == "evolve_mean" and abs(eps_sq) > 0.0:
        prefac = jnp.asarray(1.0 / eps_sq, dtype=th_bar_nodal.dtype)
        mean_energy = 0.5 * prefac * jnp.sum(grid.cc_weights * th_bar_nodal ** 2)
        mean_flux_exchange = prefac * jnp.sum(grid.cc_weights * th_bar_nodal * explicit_nodal)
        mean_diffusion = prefac * jnp.sum(grid.cc_weights * th_bar_nodal * implicit_nodal)
        mean_total = mean_flux_exchange + mean_diffusion
    else:
        mean_energy = jnp.asarray(0.0, dtype=th_bar_nodal.dtype)
        mean_flux_exchange = jnp.asarray(0.0, dtype=th_bar_nodal.dtype)
        mean_diffusion = jnp.asarray(0.0, dtype=th_bar_nodal.dtype)
        mean_total = jnp.asarray(0.0, dtype=th_bar_nodal.dtype)

    theta_mean_feedback = (
        theta_budget["th_mean_feedback_sum"]
        if theta_budget is not None and "th_mean_feedback_sum" in theta_budget
        else jnp.asarray(0.0, dtype=th_bar_nodal.dtype)
    )
    theta_mean_feedback_dealiased = (
        thermal_dealiased["th_mean_feedback_sum_dealiased"]
        if thermal_dealiased is not None and "th_mean_feedback_sum_dealiased" in thermal_dealiased
        else jnp.asarray(0.0, dtype=th_bar_nodal.dtype)
    )
    exchange_residual = theta_mean_feedback + mean_flux_exchange
    exchange_residual_dealiased = theta_mean_feedback_dealiased + mean_flux_exchange
    exchange_scale = jnp.maximum(
        jnp.abs(theta_mean_feedback) + jnp.abs(mean_flux_exchange),
        jnp.asarray(1e-300, dtype=th_bar_nodal.dtype),
    )
    exchange_scale_dealiased = jnp.maximum(
        jnp.abs(theta_mean_feedback_dealiased) + jnp.abs(mean_flux_exchange),
        jnp.asarray(1e-300, dtype=th_bar_nodal.dtype),
    )
    sbp_exchange = compute_sbp_internal_exchange_budget(state, grid)

    return {
        "th_bar_phys_max": jnp.max(jnp.abs(th_bar_nodal)),
        "dth_bar_dz_max": jnp.max(jnp.abs(dth_bar_dz_nodal)),
        "mean_grad_min": jnp.min(mean_grad_nodal),
        "mean_grad_max": jnp.max(mean_grad_nodal),
        "mean_grad_mid": 1.0 - _eval_cheb_series(dth_bar_dz_coeffs, 0.5),
        "mean_energy": mean_energy,
        "mean_flux_exchange_tendency": mean_flux_exchange,
        "mean_diffusion_tendency": mean_diffusion,
        "mean_total_tendency": mean_total,
        "mean_theta_exchange_residual": exchange_residual,
        "mean_theta_exchange_residual_rel": exchange_residual / exchange_scale,
        "mean_theta_exchange_residual_dealiased": exchange_residual_dealiased,
        "mean_theta_exchange_residual_dealiased_rel": (
            exchange_residual_dealiased / exchange_scale_dealiased
        ),
        **sbp_exchange,
    }


def compute_ke_budget(state: State, grid: Grid) -> dict:
    """Compute shell-binned horizontal KE budget terms for the current state."""
    psi_hat = invert_psi(state.q_hat, grid.inv_denom)
    psi_nodal = _to_nodal(psi_hat, grid.V)

    explicit = explicit_rhs_dispatch(state, grid)
    implicit = implicit_tendency(state, grid)

    q_beta = -1j * grid.beta * grid.kx[None, :, :] * psi_hat
    q_nonlinear = explicit.q_hat - q_beta
    q_stretch = implicit.q_hat
    q_diss = -grid.diss_rate_q[None, :, :] * state.q_hat

    q_nonlinear_nodal = _to_nodal(q_nonlinear, grid.V)
    q_beta_nodal = _to_nodal(q_beta, grid.V)
    q_stretch_nodal = _to_nodal(q_stretch, grid.V)
    q_diss_nodal = _to_nodal(q_diss, grid.V)

    ke_nonlinear_shell = _ke_shell_tendency_from_q_term(psi_nodal, q_nonlinear_nodal, grid)
    ke_beta_shell = _ke_shell_tendency_from_q_term(psi_nodal, q_beta_nodal, grid)
    ke_stretch_shell = _ke_shell_tendency_from_q_term(psi_nodal, q_stretch_nodal, grid)
    ke_diss_shell = _ke_shell_tendency_from_q_term(psi_nodal, q_diss_nodal, grid)
    ke_total_shell = ke_nonlinear_shell + ke_beta_shell + ke_stretch_shell + ke_diss_shell
    _, _, k_bins = _shell_bins(grid.ksq, float(grid.L))

    return {
        'ke_k_bins': k_bins,
        'ke_horiz_spec': energy_spectrum(psi_nodal, grid.ksq, grid.cc_weights, float(grid.L))[3] / (grid.Nx ** 4),
        'ke_nonlinear_shell_tendency': ke_nonlinear_shell,
        'ke_beta_shell_tendency': ke_beta_shell,
        'ke_stretch_shell_tendency': ke_stretch_shell,
        'ke_diss_shell_tendency': ke_diss_shell,
        'ke_total_shell_tendency': ke_total_shell,
        'ke_nonlinear_flux': -jnp.cumsum(ke_nonlinear_shell),
        'ke_nonlinear_sum': jnp.sum(ke_nonlinear_shell),
        'ke_beta_sum': jnp.sum(ke_beta_shell),
        'ke_stretch_sum': jnp.sum(ke_stretch_shell),
        'ke_diss_sum': jnp.sum(ke_diss_shell),
        'ke_total_sum': jnp.sum(ke_total_shell),
    }


def compute_w_theta_budgets(state: State, grid: Grid) -> dict:
    """Compute shell-binned budgets for 0.5|w|^2 and 0.5|theta|^2."""
    explicit = explicit_rhs_dispatch(state, grid)
    implicit = implicit_tendency(state, grid)

    w_cheb = _dirichlet_to_cheb(state.w_hat, grid.dirichlet_stencil)
    th_cheb = _dirichlet_to_cheb(state.th_hat, grid.dirichlet_stencil)
    w_nodal = _to_nodal(w_cheb, grid.V)
    th_nodal = _to_nodal(th_cheb, grid.V)

    w_nonlinear_cheb = _dirichlet_to_cheb(explicit.w_hat, grid.dirichlet_stencil)

    dq_dZ = jnp.einsum('ij,j...->i...', grid.G_Z, state.q_hat)
    w_q_coupling_cheb = project_dirichlet(
        grid.inv_denom[None, :, :] * dq_dZ, grid.proj_dirichlet
    )
    w_buoyancy_cheb = project_dirichlet(
        grid.Ra_sigma * th_cheb, grid.proj_dirichlet
    )
    w_diss_cheb = _dirichlet_to_cheb(
        -grid.diss_rate_w[None, :, :] * state.w_hat, grid.dirichlet_stencil
    )

    th_explicit_total_cheb = _dirichlet_to_cheb(explicit.th_hat, grid.dirichlet_stencil)
    th_mean_feedback_cheb = _theta_mean_feedback_cheb(state, grid)
    th_nonlinear_cheb = th_explicit_total_cheb - th_mean_feedback_cheb
    th_conduction_cheb = _dirichlet_to_cheb(implicit.th_hat, grid.dirichlet_stencil)
    th_diss_cheb = _dirichlet_to_cheb(
        -grid.diss_rate_th[None, :, :] * state.th_hat, grid.dirichlet_stencil
    )

    w_nonlinear_nodal = _to_nodal(w_nonlinear_cheb, grid.V)
    w_q_coupling_nodal = _to_nodal(w_q_coupling_cheb, grid.V)
    w_buoyancy_nodal = _to_nodal(w_buoyancy_cheb, grid.V)
    w_diss_nodal = _to_nodal(w_diss_cheb, grid.V)

    th_nonlinear_nodal = _to_nodal(th_nonlinear_cheb, grid.V)
    th_mean_feedback_nodal = _to_nodal(th_mean_feedback_cheb, grid.V)
    th_conduction_nodal = _to_nodal(th_conduction_cheb, grid.V)
    th_diss_nodal = _to_nodal(th_diss_cheb, grid.V)

    w_nonlinear_shell = _quadratic_shell_tendency(w_nodal, w_nonlinear_nodal, grid)
    w_q_coupling_shell = _quadratic_shell_tendency(w_nodal, w_q_coupling_nodal, grid)
    w_buoyancy_shell = _quadratic_shell_tendency(w_nodal, w_buoyancy_nodal, grid)
    w_diss_shell = _quadratic_shell_tendency(w_nodal, w_diss_nodal, grid)
    w_total_shell = w_nonlinear_shell + w_q_coupling_shell + w_buoyancy_shell + w_diss_shell

    th_nonlinear_shell = _quadratic_shell_tendency(th_nodal, th_nonlinear_nodal, grid)
    th_mean_feedback_shell = _quadratic_shell_tendency(
        th_nodal, th_mean_feedback_nodal, grid
    )
    th_conduction_shell = _quadratic_shell_tendency(
        th_nodal, th_conduction_nodal, grid
    )
    th_diss_shell = _quadratic_shell_tendency(th_nodal, th_diss_nodal, grid)
    th_total_shell = (
        th_nonlinear_shell
        + th_mean_feedback_shell
        + th_conduction_shell
        + th_diss_shell
    )

    return {
        'w_nonlinear_shell_tendency': w_nonlinear_shell,
        'w_q_coupling_shell_tendency': w_q_coupling_shell,
        'w_buoyancy_shell_tendency': w_buoyancy_shell,
        'w_diss_shell_tendency': w_diss_shell,
        'w_total_shell_tendency': w_total_shell,
        'w_nonlinear_flux': -jnp.cumsum(w_nonlinear_shell),
        'w_nonlinear_sum': jnp.sum(w_nonlinear_shell),
        'w_q_coupling_sum': jnp.sum(w_q_coupling_shell),
        'w_buoyancy_sum': jnp.sum(w_buoyancy_shell),
        'w_diss_sum': jnp.sum(w_diss_shell),
        'w_total_sum': jnp.sum(w_total_shell),
        'th_nonlinear_shell_tendency': th_nonlinear_shell,
        'th_mean_feedback_shell_tendency': th_mean_feedback_shell,
        'th_conduction_shell_tendency': th_conduction_shell,
        'th_diss_shell_tendency': th_diss_shell,
        'th_total_shell_tendency': th_total_shell,
        'th_nonlinear_flux': -jnp.cumsum(th_nonlinear_shell),
        'th_nonlinear_sum': jnp.sum(th_nonlinear_shell),
        'th_mean_feedback_sum': jnp.sum(th_mean_feedback_shell),
        'th_conduction_sum': jnp.sum(th_conduction_shell),
        'th_diss_sum': jnp.sum(th_diss_shell),
        'th_total_sum': jnp.sum(th_total_shell),
    }


def vertical_mode_energy(field_hat: jnp.ndarray) -> jnp.ndarray:
    """Horizontal energy of each Chebyshev coefficient."""
    return jnp.sum(jnp.abs(field_hat) ** 2, axis=(1, 2))


def high_mode_fraction(spec: jnp.ndarray, frac: float = 0.25) -> jnp.ndarray:
    """Fraction of energy in the top ``frac`` of Chebyshev modes."""
    n = spec.shape[0]
    n_tail = max(1, int(n * frac))
    tail = jnp.sum(spec[-n_tail:])
    total = jnp.sum(spec)
    return jnp.where(total > 0, tail / total, 0.0)


def compute_diagnostics(state: State, grid: Grid) -> dict:
    """Compute scalar diagnostics.

    Converts coefficient-space fields to nodal values for depth integrals
    and physical-space computations.
    """
    # Convert to nodal values for all diagnostics
    psi_hat = invert_psi(state.q_hat, grid.inv_denom)
    psi_nodal = _to_nodal(psi_hat, grid.V)
    q_nodal = _to_nodal(state.q_hat, grid.V)
    w_cheb = _dirichlet_to_cheb(state.w_hat, grid.dirichlet_stencil)
    th_cheb = _dirichlet_to_cheb(state.th_hat, grid.dirichlet_stencil)
    w_nodal = _to_nodal(w_cheb, grid.V)
    th_nodal = _to_nodal(th_cheb, grid.V)

    psi_bt = barotropic_mode(psi_nodal, grid.cc_weights)

    ksq = grid.ksq
    Nx = grid.Nx

    w_rfft = _horizontal_rfft_weight(ksq)
    norm = Nx ** 4

    # Barotropic KE
    KE_bt = 0.5 * jnp.sum(ksq * jnp.abs(psi_bt) ** 2 * w_rfft) / norm

    # Total KE (depth-integrated)
    psi_sq_int = jnp.einsum('j,j...->...', grid.cc_weights,
                             jnp.abs(psi_nodal) ** 2)
    KE_tot = 0.5 * jnp.sum(ksq * psi_sq_int * w_rfft) / norm

    KE_bc = KE_tot - KE_bt

    # Horizontal velocity and fluctuation amplitudes
    u_hat = -1j * grid.ky[None, :, :] * psi_nodal
    v_hat = 1j * grid.kx[None, :, :] * psi_nodal
    u_phys = jnp.fft.irfft2(u_hat, s=(Nx, Nx))
    v_phys = jnp.fft.irfft2(v_hat, s=(Nx, Nx))
    w_phys = jnp.fft.irfft2(w_nodal, s=(Nx, Nx))
    th_phys = jnp.fft.irfft2(th_nodal, s=(Nx, Nx))
    tw_phys = w_phys * th_phys
    max_speed = jnp.max(jnp.sqrt(u_phys ** 2 + v_phys ** 2))
    max_w = jnp.max(jnp.abs(w_phys))
    max_theta = jnp.max(jnp.abs(th_phys))
    max_tw = jnp.max(jnp.abs(tw_phys))

    # Enstrophy
    q_sq_int = jnp.einsum('j,j...->...', grid.cc_weights,
                           jnp.abs(q_nodal) ** 2)
    enstrophy = 0.5 * jnp.sum(q_sq_int * w_rfft) / norm

    w_sq_int = jnp.einsum('j,j...->...', grid.cc_weights, jnp.abs(w_nodal) ** 2)
    th_sq_int = jnp.einsum('j,j...->...', grid.cc_weights, jnp.abs(th_nodal) ** 2)

    # Nusselt number / convective flux
    wth_int = jnp.einsum('j,j...->...', grid.cc_weights,
                          jnp.real(w_nodal * jnp.conj(th_nodal)))
    vol_avg_tw = jnp.sum(wth_int * w_rfft) / norm
    flux_profile_dealiased = _solver_mean_flux_profile_nodal(state, grid)
    vol_avg_tw_dealiased = jnp.sum(grid.cc_weights * flux_profile_dealiased)
    Nusselt = 1.0 + vol_avg_tw
    Nusselt_dealiased = 1.0 + vol_avg_tw_dealiased

    q_rms = jnp.sqrt(jnp.sum(q_sq_int * w_rfft) / norm)
    w_rms = jnp.sqrt(jnp.sum(w_sq_int * w_rfft) / norm)
    th_rms = jnp.sqrt(jnp.sum(th_sq_int * w_rfft) / norm)

    q_spec = vertical_mode_energy(state.q_hat)
    w_spec = vertical_mode_energy(w_cheb)
    th_spec = vertical_mode_energy(th_cheb)
    k_bins, q_horiz_spec = shell_spectrum(q_nodal, grid.ksq, grid.cc_weights, float(grid.L))
    _, w_horiz_spec = shell_spectrum(w_nodal, grid.ksq, grid.cc_weights, float(grid.L))
    _, th_horiz_spec = shell_spectrum(th_nodal, grid.ksq, grid.cc_weights, float(grid.L))
    ke_budget = compute_ke_budget(state, grid)
    w_th_budgets = compute_w_theta_budgets(state, grid)
    thermal_dealiased = compute_dealiased_thermal_shell_budgets(state, grid, w_nodal, th_nodal)
    flux_profile_dealiased = thermal_dealiased["flux_profile_dealiased"]
    vol_avg_tw_dealiased = jnp.sum(grid.cc_weights * flux_profile_dealiased)
    Nusselt_dealiased = 1.0 + vol_avg_tw_dealiased
    mean_budget = compute_mean_temperature_budget(state, grid, w_th_budgets, thermal_dealiased)

    return {
        'KE_bt': KE_bt,
        'KE_bc': KE_bc,
        'KE_tot': KE_tot,
        'max_speed': max_speed,
        'max_w': max_w,
        'max_theta': max_theta,
        'max_tw': max_tw,
        'enstrophy': enstrophy,
        'Nusselt': Nusselt,
        'Nusselt_dealiased': Nusselt_dealiased,
        'vol_avg_tw': vol_avg_tw,
        'vol_avg_tw_dealiased': vol_avg_tw_dealiased,
        'heat_flux_mismatch': vol_avg_tw_dealiased - vol_avg_tw,
        'q_rms': q_rms,
        'w_rms': w_rms,
        'th_rms': th_rms,
        'th_bar_max': jnp.max(jnp.abs(state.th_bar)),
        'q_vert_spec': q_spec,
        'w_vert_spec': w_spec,
        'th_vert_spec': th_spec,
        'k_bins': k_bins,
        'q_horiz_spec': q_horiz_spec,
        'w_horiz_spec': w_horiz_spec,
        'th_horiz_spec': th_horiz_spec,
        'q_high_frac': high_mode_fraction(q_spec),
        'w_high_frac': high_mode_fraction(w_spec),
        'th_high_frac': high_mode_fraction(th_spec),
        **ke_budget,
        **w_th_budgets,
        **thermal_dealiased,
        **mean_budget,
    }
