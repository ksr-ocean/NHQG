"""Core solver: RHS evaluation, IMEX-RK steppers, and main loop.

q' is stored in Chebyshev coefficients. Dirichlet fields (w, theta) are stored
in the Coral-style Galerkin basis ``-T_n + T_{n+2}``, while mean temperature
uses Chebyshev coefficients. Nonlinear terms are evaluated at CGL nodes via
V/V_inv transforms and projected back into the Dirichlet basis.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nhqg.grid import Grid
from nhqg.spectral import (
    _zero_pad,
    triple_conservative_flux_divergence,
    triple_conservative_flux_divergence_23,
    triple_jacobian,
    triple_jacobian_23,
)


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

class State(NamedTuple):
    """State vector: q' in Chebyshev, w/theta in Dirichlet Galerkin basis."""
    q_hat: jnp.ndarray   # (Nz+1, Nx, Nk) complex — Chebyshev coeffs of q'
    w_hat: jnp.ndarray   # (Nz-1, Nx, Nk) complex — Dirichlet Galerkin coeffs of w
    th_hat: jnp.ndarray  # (Nz-1, Nx, Nk) complex — Dirichlet Galerkin coeffs of theta
    th_bar: jnp.ndarray  # (Nz+1,) real — Chebyshev coeffs of mean temp deviation


# ---------------------------------------------------------------------------
# Helpers: coefficient <-> nodal transforms
# ---------------------------------------------------------------------------

def _to_nodal(field, V):
    """Chebyshev coefficients -> CGL nodal values. field: (Nz+1, ...)."""
    return jnp.einsum('ij,j...->i...', V, field)


def _to_coeffs(field, V_inv):
    """CGL nodal values -> Chebyshev coefficients. field: (Nz+1, ...)."""
    return jnp.einsum('ij,j...->i...', V_inv, field)


def _to_nodal_1d(field, V):
    """1D version: (Nz+1,) -> (Nz+1,)."""
    return V @ field


def _to_coeffs_1d(field, V_inv):
    """1D version: (Nz+1,) -> (Nz+1,)."""
    return V_inv @ field


def _dirichlet_to_cheb(field, stencil):
    """Dirichlet Galerkin coefficients -> Chebyshev coefficients."""
    return jnp.einsum('ij,j...->i...', stencil, field)


def _cheb_to_dirichlet(field, pinv):
    """Chebyshev coefficients -> Dirichlet Galerkin coefficients."""
    return jnp.einsum('ij,j...->i...', pinv, field)


def _truncate_cheb_coeffs(field, Nz):
    """Truncate overresolved Chebyshev coefficients back to degree Nz."""
    return field[:Nz + 1, ...]


def _apply_vertical_matrix(mat, field):
    """Apply a vertical operator/transfer matrix to a batched field."""
    return jnp.einsum('ij,j...->i...', mat, field)


# ---------------------------------------------------------------------------
# Streamfunction inversion: psi = -q' / (|k|^2 + Ld^{-2})
# ---------------------------------------------------------------------------

def invert_psi(q_hat: jnp.ndarray, inv_denom: jnp.ndarray) -> jnp.ndarray:
    """Recover psi_hat from q'_hat. Pointwise per Chebyshev coefficient."""
    return -q_hat * inv_denom[None, :, :]


# ---------------------------------------------------------------------------
# Zero mean: f_hat[:, 0, 0] = 0
# ---------------------------------------------------------------------------

def zero_mean(f_hat: jnp.ndarray) -> jnp.ndarray:
    """Zero out the k=0 mode at all Chebyshev coefficients."""
    return f_hat.at[:, 0, 0].set(0.0)


# ---------------------------------------------------------------------------
# State sanitization: Hermitian reality constraint + 2/3-rule band limit
# ---------------------------------------------------------------------------

def hermitian_project(f_hat: jnp.ndarray, Nx: int) -> jnp.ndarray:
    """Enforce the rfft2 reality constraint f(-kx, ky=0) = conj(f(kx, ky=0)).

    The ky=0 (and, for even Nx, ky-Nyquist) columns of an rfft2 field store
    both kx signs redundantly. Anti-Hermitian content there is invisible to
    irfft2 physics but is amplified at the unsaturated linear growth rate
    forever (see hermitian_ghost.md), poisoning every Parseval-style
    diagnostic. Projecting onto the Hermitian part is exact for any state
    that represents a real field.
    """
    neg = (-jnp.arange(Nx)) % Nx

    def sym(col):
        return 0.5 * (col + jnp.conj(col[:, neg]))

    f_hat = f_hat.at[:, :, 0].set(sym(f_hat[:, :, 0]))
    if Nx % 2 == 0:
        f_hat = f_hat.at[:, :, -1].set(sym(f_hat[:, :, -1]))
    return f_hat


def sanitize_state(state: State, grid: Grid) -> State:
    """Project the evolved state onto its physically meaningful subspace.

    (i) Hermitian reality constraint on the redundant rfft2 columns.
    (ii) Under 23_rule, zero the masked band: those modes receive no
    nonlinear tendency (the mask is applied to the products), so any state
    content there evolves purely linearly -- at Nx=64 part of that band is
    linearly UNSTABLE and grows frozen and unsaturated. Band-limiting the
    state each step removes that pathology; under 32_rule the padded
    products handle dealiasing and the state is legitimately full-band.
    """
    q, w, th = state.q_hat, state.w_hat, state.th_hat
    q = hermitian_project(q, grid.Nx)
    w = hermitian_project(w, grid.Nx)
    th = hermitian_project(th, grid.Nx)
    if grid.horizontal_dealiasing == "23_rule":
        m = grid.mask_23[None, :, :]
        q, w, th = q * m, w * m, th * m
    return State(q, w, th, state.th_bar)


# ---------------------------------------------------------------------------
# Tau BC projection (for explicit steppers)
# ---------------------------------------------------------------------------

def project_dirichlet(f_hat: jnp.ndarray, proj: jnp.ndarray) -> jnp.ndarray:
    """Adjust last 2 Chebyshev coefficients to satisfy Dirichlet BCs."""
    return jnp.einsum('ij,j...->i...', proj, f_hat)


def project_neumann(f_hat: jnp.ndarray, proj: jnp.ndarray) -> jnp.ndarray:
    """Adjust last 2 Chebyshev coefficients to satisfy Neumann BCs."""
    return jnp.einsum('ij,j...->i...', proj, f_hat)


def project_dirichlet_1d(f: jnp.ndarray, proj: jnp.ndarray) -> jnp.ndarray:
    """1D version for mean temperature."""
    return proj @ f


def _apply_q_boundary(q_hat: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Apply the configured q boundary treatment for explicit paths."""
    if grid.q_boundary == "neumann":
        return project_neumann(q_hat, grid.proj_neumann)
    if grid.q_boundary == "none":
        return q_hat
    raise ValueError(f"Unsupported q_boundary={grid.q_boundary!r}")


def _apply_vertical_cutoff(state: State, grid: Grid) -> State:
    """Optional high-n cutoff experiment applied to w and theta only."""
    cutoff_n = grid.vertical_cutoff_n
    if cutoff_n is None or cutoff_n >= grid.Nz:
        return state

    keep = (jnp.arange(grid.Nz + 1) <= cutoff_n).astype(state.q_hat.real.dtype)
    keep = keep[:, None, None]
    w_cheb = _dirichlet_to_cheb(state.w_hat, grid.w_stencil)
    th_cheb = _dirichlet_to_cheb(state.th_hat, grid.dirichlet_stencil)
    w_cheb = project_dirichlet(w_cheb * keep, grid.proj_w)
    th_cheb = project_dirichlet(th_cheb * keep, grid.proj_dirichlet)
    w_hat = _cheb_to_dirichlet(w_cheb, grid.w_pinv)
    th_hat = _cheb_to_dirichlet(th_cheb, grid.dirichlet_pinv)
    return State(state.q_hat, w_hat, th_hat, state.th_bar)


# ---------------------------------------------------------------------------
# Mean temperature helpers
# ---------------------------------------------------------------------------

def mean_temperature_total(th_bar: jnp.ndarray, Z: jnp.ndarray,
                           V: jnp.ndarray) -> jnp.ndarray:
    """Total mean temperature at CGL nodes: (1 - Z) + V @ th_bar."""
    return 1.0 - Z + _to_nodal_1d(th_bar, V)


def horizontal_mean_wtheta(w_hat: jnp.ndarray, th_hat: jnp.ndarray,
                           V: jnp.ndarray, w_stencil: jnp.ndarray,
                           th_stencil: jnp.ndarray,
                           Nx: int, Npad: int | None = None) -> jnp.ndarray:
    """Compute <w theta>_xy as a function of Z at CGL nodes.

    Inputs are Chebyshev coefficients; converts to nodal, then to physical
    space for the product, averages horizontally. Returns nodal values (Nz+1,).
    """
    w_nodal = _to_nodal(_dirichlet_to_cheb(w_hat, w_stencil), V)
    th_nodal = _to_nodal(_dirichlet_to_cheb(th_hat, th_stencil), V)

    if Npad is None or Npad == Nx:
        w_phys = jnp.fft.irfft2(w_nodal, s=(Nx, Nx))
        th_phys = jnp.fft.irfft2(th_nodal, s=(Nx, Nx))
    else:
        pad_one = lambda field: _zero_pad(field, Nx, Npad)
        w_pad = jax.vmap(pad_one)(w_nodal)
        th_pad = jax.vmap(pad_one)(th_nodal)
        # irfft2 on the padded grid attenuates amplitudes by (Nx/Npad)^2
        scale = (Npad / Nx) ** 2
        w_phys = scale * jnp.fft.irfft2(w_pad, s=(Npad, Npad))
        th_phys = scale * jnp.fft.irfft2(th_pad, s=(Npad, Npad))

    return jnp.mean(w_phys * th_phys, axis=(1, 2))


def horizontal_mean_from_nodal_spectral(w_nodal: jnp.ndarray, th_nodal: jnp.ndarray,
                                        Nx: int, Npad: int | None = None) -> jnp.ndarray:
    """Compute <w theta>_xy from nodal spectral fields on a common vertical grid."""
    if Npad is None or Npad == Nx:
        w_phys = jnp.fft.irfft2(w_nodal, s=(Nx, Nx))
        th_phys = jnp.fft.irfft2(th_nodal, s=(Nx, Nx))
    else:
        pad_one = lambda field: _zero_pad(field, Nx, Npad)
        w_pad = jax.vmap(pad_one)(w_nodal)
        th_pad = jax.vmap(pad_one)(th_nodal)
        scale = (Npad / Nx) ** 2
        w_phys = scale * jnp.fft.irfft2(w_pad, s=(Npad, Npad))
        th_phys = scale * jnp.fft.irfft2(th_pad, s=(Npad, Npad))

    return jnp.mean(w_phys * th_phys, axis=(1, 2))


def thermal_exchange_workgrid_fields(state: State, grid: Grid) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Coral-style work-grid reconstructions for the thermal exchange pair."""
    w_cheb = _dirichlet_to_cheb(state.w_hat, grid.w_stencil)
    th_cheb = _dirichlet_to_cheb(state.th_hat, grid.dirichlet_stencil)
    w_work = _to_nodal(w_cheb, grid.V_exchange)
    th_work = _to_nodal(th_cheb, grid.V_exchange)
    dth_bar_dZ_coeffs = grid.G_Z @ state.th_bar
    dth_bar_dZ_work = _to_nodal_1d(dth_bar_dZ_coeffs, grid.V_exchange)
    return w_work, th_work, dth_bar_dZ_work


def thermal_exchange_workgrid_coeffs(state: State, grid: Grid) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Coral-style thermal exchange terms on a shared vertical work grid.

    Returns:
      product_coeffs: coefficients of (d_z Theta_bar') * w on the work grid
      flux_coeffs: coefficients of <w theta>_xy on the same work grid
    """
    w_work, th_work, dth_bar_dZ_work = thermal_exchange_workgrid_fields(state, grid)

    product_hi = _to_coeffs(
        dth_bar_dZ_work[:, None, None] * w_work, grid.V_exchange_inv
    )
    product_coeffs = _truncate_cheb_coeffs(product_hi, grid.Nz)

    flux_work = horizontal_mean_from_nodal_spectral(
        w_work, th_work, grid.Nx, grid.Npad
    )
    flux_hi = _to_coeffs_1d(flux_work, grid.V_exchange_inv)
    flux_coeffs = _truncate_cheb_coeffs(flux_hi, grid.Nz)
    return product_coeffs, flux_coeffs


def uses_coral_exchange_workgrid(grid: Grid) -> bool:
    """Whether the mean/theta exchange pair is built on the Coral-style work grid."""
    return grid.mean_exchange_discretization in {
        "coral_workgrid",
        "coral_workgrid_weakmean",
        "coral_workgrid_paired",
    }


def uses_balanced_midpoint_exchange(grid: Grid) -> bool:
    """Whether thermal mean/fluctuation exchange uses the balanced midpoint substep."""
    return grid.mean_exchange_discretization == "balanced_midpoint"


def uses_balanced_sbp2_split_exchange(grid: Grid) -> bool:
    """Whether thermal exchange uses the original split SBP2 substep."""
    return grid.mean_exchange_discretization == "balanced_sbp2"


def uses_balanced_sbp2_pc_exchange(grid: Grid) -> bool:
    """Whether thermal exchange uses the stage-wise SBP2 predictor/corrector."""
    return grid.mean_exchange_discretization == "balanced_sbp2_pc"


def uses_balanced_sbp2_exchange(grid: Grid) -> bool:
    """Whether thermal mean/fluctuation exchange uses an SBP2-based substep."""
    return uses_balanced_sbp2_split_exchange(grid) or uses_balanced_sbp2_pc_exchange(grid)


def uses_split_thermal_exchange(grid: Grid) -> bool:
    """Whether thermal exchange is handled in a separate split substep."""
    return (
        uses_balanced_midpoint_exchange(grid)
        or uses_balanced_sbp2_split_exchange(grid)
        or uses_balanced_sbp2_pc_exchange(grid)
    )


def uses_paired_mean_exchange(grid: Grid) -> bool:
    """Whether both sides of the thermal exchange pair use the paired work-grid map."""
    return grid.mean_exchange_discretization == "coral_workgrid_paired"


def mean_flux_exchange_rhs_coeffs(flux_coeffs: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Explicit mean-temperature exchange RHS from a vertical flux profile."""
    if grid.mean_exchange_discretization == "coral_workgrid_weakmean":
        weighted_flux = grid.mean_mass @ flux_coeffs
        return grid.mean_mass_inv @ (grid.G_Z.T @ weighted_flux)
    return -grid.G_Z @ flux_coeffs


def paired_theta_feedback_from_work(product_work: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Return the paired theta mean-feedback term in the Dirichlet Galerkin basis."""
    weighted_product = grid.exchange_weights[:, None, None] * product_work
    lifted = jnp.einsum('ji,j...->i...', grid.V_exchange_dirichlet, weighted_product)
    return -jnp.einsum('ij,j...->i...', grid.theta_mass_inv, lifted)


def paired_mean_flux_exchange_rhs(flux_work: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Return the paired explicit mean-temperature exchange RHS in Chebyshev coefficients."""
    weighted_flux = grid.exchange_weights * flux_work
    lifted = grid.G_exchange.T @ weighted_flux
    return grid.mean_mass_inv @ lifted


def sbp2_exchange_state_fields(state: State, grid: Grid) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Current-state thermal fields on the uniform SBP work grid."""
    w_cheb = _dirichlet_to_cheb(state.w_hat, grid.w_stencil)
    th_cheb = _dirichlet_to_cheb(state.th_hat, grid.dirichlet_stencil)
    w_cgl = _to_nodal(w_cheb, grid.V)
    th_cgl = _to_nodal(th_cheb, grid.V)
    th_bar_cgl = _to_nodal_1d(state.th_bar, grid.V)

    w_sbp = _apply_vertical_matrix(grid.cgl_to_sbp, w_cgl)
    th_sbp = _apply_vertical_matrix(grid.cgl_to_sbp, th_cgl)
    th_bar_sbp = _apply_vertical_matrix(grid.cgl_to_sbp, th_bar_cgl)
    dth_bar_dz_sbp = grid.sbp_D1 @ th_bar_sbp
    return w_sbp, th_sbp, th_bar_sbp, dth_bar_dz_sbp


def sbp2_theta_feedback_cheb(state: State, grid: Grid) -> jnp.ndarray:
    """Current-state theta mean-feedback term reconstructed from the SBP grid."""
    w_sbp, _, _, dth_bar_dz_sbp = sbp2_exchange_state_fields(state, grid)
    product_sbp = dth_bar_dz_sbp[:, None, None] * w_sbp
    product_cgl = _apply_vertical_matrix(grid.sbp_to_cgl, product_sbp)
    product_coeffs = _to_coeffs(product_cgl, grid.V_inv)
    return project_dirichlet(-product_coeffs, grid.proj_dirichlet)


def sbp2_flux_profile_nodal(state: State, grid: Grid) -> jnp.ndarray:
    """Current-state solver-consistent mean heat-flux profile for the SBP branch."""
    w_sbp, th_sbp, _, _ = sbp2_exchange_state_fields(state, grid)
    flux_sbp = horizontal_mean_from_nodal_spectral(w_sbp, th_sbp, grid.Nx, grid.Npad)
    return _apply_vertical_matrix(grid.sbp_to_cgl, flux_sbp)


def sbp2_mean_rhs_nodal(state: State, grid: Grid) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Current-state mean exchange and diffusion tendencies in CGL nodal form."""
    w_sbp, th_sbp, th_bar_sbp, _ = sbp2_exchange_state_fields(state, grid)
    flux_sbp = horizontal_mean_from_nodal_spectral(w_sbp, th_sbp, grid.Nx, grid.Npad)
    exchange_rhs_sbp = -grid.mean_temp_eps_sq * (grid.sbp_D1 @ flux_sbp)
    diffusion_rhs_sbp = (grid.mean_temp_eps_sq / grid.sigma) * (grid.sbp_L @ th_bar_sbp)
    return (
        _apply_vertical_matrix(grid.sbp_to_cgl, exchange_rhs_sbp),
        _apply_vertical_matrix(grid.sbp_to_cgl, diffusion_rhs_sbp),
    )


def _triple_horizontal_advection(psi_nodal: jnp.ndarray,
                                 q_nodal: jnp.ndarray,
                                 w_nodal: jnp.ndarray,
                                 th_nodal: jnp.ndarray,
                                 grid: Grid):
    """Return the horizontally dealiased advection operator for q', w, theta."""
    if grid.horizontal_dealiasing == "23_rule":
        if grid.nonlinear_advection == "jacobian":
            return triple_jacobian_23(
                psi_nodal, q_nodal, w_nodal, th_nodal,
                grid.kx, grid.ky, grid.mask_23, grid.Nx
            )
        if grid.nonlinear_advection == "flux":
            return triple_conservative_flux_divergence_23(
                psi_nodal, q_nodal, w_nodal, th_nodal,
                grid.kx, grid.ky, grid.mask_23, grid.Nx
            )
        raise ValueError(f"Unsupported nonlinear_advection={grid.nonlinear_advection!r}")
    # default 3/2-rule path
    if grid.nonlinear_advection == "jacobian":
        return triple_jacobian(
            psi_nodal, q_nodal, w_nodal, th_nodal,
            grid.kx, grid.ky, grid.Nx, grid.Npad
        )
    if grid.nonlinear_advection == "flux":
        return triple_conservative_flux_divergence(
            psi_nodal, q_nodal, w_nodal, th_nodal,
            grid.kx, grid.ky, grid.Nx, grid.Npad
        )
    raise ValueError(f"Unsupported nonlinear_advection={grid.nonlinear_advection!r}")


def _explicit_rhs_vertical_dealiased(state: State, grid: Grid) -> State:
    """Explicit RHS with overresolved vertical collocation and coefficient truncation."""
    q_hat, w_hat, th_hat, th_bar = state
    psi_hat = invert_psi(q_hat, grid.inv_denom)
    w_cheb = _dirichlet_to_cheb(w_hat, grid.w_stencil)
    th_cheb = _dirichlet_to_cheb(th_hat, grid.dirichlet_stencil)

    psi_nodal = _to_nodal(psi_hat, grid.V_dealias)
    q_nodal = _to_nodal(q_hat, grid.V_dealias)
    w_nodal = _to_nodal(w_cheb, grid.V_dealias)
    th_nodal = _to_nodal(th_cheb, grid.V_dealias)

    # Polar trap: advect the total PV q' + eta (Z-independent; see explicit_rhs)
    q_adv_nodal = (q_nodal if grid.eta_hat is None
                   else q_nodal + grid.eta_hat[None, :, :])

    Aq_n, Aw_n, Ath_n = _triple_horizontal_advection(psi_nodal, q_adv_nodal, w_nodal, th_nodal, grid)

    Aq_hi = _to_coeffs(Aq_n, grid.V_dealias_inv)
    Aw_hi = _to_coeffs(Aw_n, grid.V_dealias_inv)
    Ath_hi = _to_coeffs(Ath_n, grid.V_dealias_inv)
    Aq = _truncate_cheb_coeffs(Aq_hi, grid.Nz)
    Aw = _truncate_cheb_coeffs(Aw_hi, grid.Nz)
    Ath = _truncate_cheb_coeffs(Ath_hi, grid.Nz)

    E_q = -Aq - 1j * grid.beta * grid.kx[None, :, :] * psi_hat
    E_w = _cheb_to_dirichlet(project_dirichlet(-Aw, grid.proj_w), grid.w_pinv)
    E_th_adv = _cheb_to_dirichlet(project_dirichlet(-Ath, grid.proj_dirichlet), grid.dirichlet_pinv)

    if grid.thermal_closure == "evolve_mean":
        if uses_split_thermal_exchange(grid):
            # Exchange and mean diffusion are handled in a dedicated midpoint
            # thermal substep, not as explicit/implicit RHS terms.
            E_th = E_th_adv
            E_th_bar = jnp.zeros_like(th_bar)
        elif uses_paired_mean_exchange(grid):
            w_work, th_work, dth_bar_dZ_work = thermal_exchange_workgrid_fields(state, grid)
            E_th = E_th_adv + paired_theta_feedback_from_work(
                dth_bar_dZ_work[:, None, None] * w_work, grid
            )
            flux_work = horizontal_mean_from_nodal_spectral(
                w_work, th_work, grid.Nx, grid.Npad
            )
            E_th_bar = grid.mean_temp_eps_sq * paired_mean_flux_exchange_rhs(flux_work, grid)
        elif uses_coral_exchange_workgrid(grid):
            product_coeffs, flux_coeffs = thermal_exchange_workgrid_coeffs(state, grid)
            th_rhs_cheb = project_dirichlet(-Ath - product_coeffs, grid.proj_dirichlet)
            E_th = _cheb_to_dirichlet(th_rhs_cheb, grid.dirichlet_pinv)
            E_th_bar = grid.mean_temp_eps_sq * mean_flux_exchange_rhs_coeffs(flux_coeffs, grid)
        else:
            dth_bar_dZ_coeffs = grid.G_Z @ th_bar
            dth_bar_dZ_nodal = _to_nodal_1d(dth_bar_dZ_coeffs, grid.V_dealias)
            product_nodal = dth_bar_dZ_nodal[:, None, None] * w_nodal
            product_hi = _to_coeffs(product_nodal, grid.V_dealias_inv)
            product_coeffs = _truncate_cheb_coeffs(product_hi, grid.Nz)
            th_rhs_cheb = project_dirichlet(-Ath - product_coeffs, grid.proj_dirichlet)
            E_th = _cheb_to_dirichlet(th_rhs_cheb, grid.dirichlet_pinv)
            flux_nodal = horizontal_mean_wtheta(
                w_hat, th_hat, grid.V_dealias, grid.w_stencil, grid.dirichlet_stencil,
                grid.Nx, grid.Npad
            )
            flux_hi = _to_coeffs_1d(flux_nodal, grid.V_dealias_inv)
            flux_coeffs = _truncate_cheb_coeffs(flux_hi, grid.Nz)
            E_th_bar = grid.mean_temp_eps_sq * mean_flux_exchange_rhs_coeffs(flux_coeffs, grid)
    else:
        E_th = E_th_adv
        E_th_bar = jnp.zeros_like(th_bar)

    return State(E_q, E_w, E_th, E_th_bar)


# ---------------------------------------------------------------------------
# Explicit RHS (nonlinear + beta + buoyancy, no vertical coupling)
# ---------------------------------------------------------------------------

def explicit_rhs(state: State, grid: Grid) -> State:
    """Compute the explicit part of the RHS.

    Horizontal advection is evaluated either as a Jacobian or as the
    conservative flux divergence of u=-psi_y, v=psi_x.

    E_q = -A[psi, q'] - i*beta*kx*psi
    E_w = -A[psi, w]
    E_th = -A[psi, th] + (evolve_mean: -dTheta_bar'/dZ * w)
    """
    q_hat, w_hat, th_hat, th_bar = state
    psi_hat = invert_psi(q_hat, grid.inv_denom)
    w_cheb = _dirichlet_to_cheb(w_hat, grid.w_stencil)
    th_cheb = _dirichlet_to_cheb(th_hat, grid.dirichlet_stencil)

    # Convert to CGL nodal values for Jacobian evaluation
    psi_nodal = _to_nodal(psi_hat, grid.V)
    q_nodal = _to_nodal(q_hat, grid.V)
    w_nodal = _to_nodal(w_cheb, grid.V)
    th_nodal = _to_nodal(th_cheb, grid.V)

    # Polar trap: advect the total PV q' + eta (NHGQ_polar.tex Approach A).
    # eta is Z-independent; in flux form div(u*(q+eta)) = J(psi,q) + J(psi,eta)
    # exactly since div u = 0, so the same augmentation serves both paths.
    q_adv_nodal = (q_nodal if grid.eta_hat is None
                   else q_nodal + grid.eta_hat[None, :, :])

    # Fused horizontal advection (operates on nodal values at each Z level)
    Aq_n, Aw_n, Ath_n = _triple_horizontal_advection(psi_nodal, q_adv_nodal, w_nodal, th_nodal, grid)

    # Convert advection results back to Chebyshev coefficients
    Aq = _to_coeffs(Aq_n, grid.V_inv)
    Aw = _to_coeffs(Aw_n, grid.V_inv)
    Ath = _to_coeffs(Ath_n, grid.V_inv)

    # Assemble explicit tendencies (in coefficient space)
    # Ra*theta remains implicit through the buoyancy block-elimination.
    E_q = -Aq - 1j * grid.beta * grid.kx[None, :, :] * psi_hat
    E_w = _cheb_to_dirichlet(project_dirichlet(-Aw, grid.proj_w), grid.w_pinv)
    E_th_adv = _cheb_to_dirichlet(project_dirichlet(-Ath, grid.proj_dirichlet), grid.dirichlet_pinv)

    if grid.thermal_closure == "evolve_mean":
        if uses_split_thermal_exchange(grid):
            # Exchange and mean diffusion are handled in a dedicated midpoint
            # thermal substep, not as explicit/implicit RHS terms.
            E_th = E_th_adv
            E_th_bar = jnp.zeros_like(th_bar)
        elif uses_paired_mean_exchange(grid):
            w_work, th_work, dth_bar_dZ_work = thermal_exchange_workgrid_fields(state, grid)
            E_th = E_th_adv + paired_theta_feedback_from_work(
                dth_bar_dZ_work[:, None, None] * w_work, grid
            )
            flux_work = horizontal_mean_from_nodal_spectral(
                w_work, th_work, grid.Nx, grid.Npad
            )
            E_th_bar = grid.mean_temp_eps_sq * paired_mean_flux_exchange_rhs(flux_work, grid)
        elif uses_coral_exchange_workgrid(grid):
            product_coeffs, flux_coeffs = thermal_exchange_workgrid_coeffs(state, grid)
            th_rhs_cheb = project_dirichlet(-Ath - product_coeffs, grid.proj_dirichlet)
            E_th = _cheb_to_dirichlet(th_rhs_cheb, grid.dirichlet_pinv)
            E_th_bar = grid.mean_temp_eps_sq * mean_flux_exchange_rhs_coeffs(flux_coeffs, grid)
        else:
            # d(Theta_bar')/dZ in coefficient space, then evaluate at nodes
            dth_bar_dZ_coeffs = grid.G_Z @ th_bar
            dth_bar_dZ_nodal = _to_nodal_1d(dth_bar_dZ_coeffs, grid.V)
            # Product at nodes, then back to coefficients
            product_nodal = dth_bar_dZ_nodal[:, None, None] * w_nodal
            product_coeffs = _to_coeffs(product_nodal, grid.V_inv)
            th_rhs_cheb = project_dirichlet(-Ath - product_coeffs, grid.proj_dirichlet)
            E_th = _cheb_to_dirichlet(th_rhs_cheb, grid.dirichlet_pinv)
            # Mean temperature tendency: -eps^2 * d<wθ>/dZ
            flux_nodal = horizontal_mean_wtheta(
                w_hat, th_hat, grid.V, grid.w_stencil, grid.dirichlet_stencil,
                grid.Nx, grid.Npad
            )
            flux_coeffs = _to_coeffs_1d(flux_nodal, grid.V_inv)
            E_th_bar = grid.mean_temp_eps_sq * mean_flux_exchange_rhs_coeffs(flux_coeffs, grid)
    else:
        E_th = E_th_adv
        E_th_bar = jnp.zeros_like(th_bar)

    return State(E_q, E_w, E_th, E_th_bar)


def explicit_rhs_dispatch(state: State, grid: Grid) -> State:
    """Select explicit RHS path, including experimental vertical dealiasing."""
    if grid.vertical_dealiasing == "none":
        return explicit_rhs(state, grid)
    if grid.vertical_dealiasing in {"cheb_3o2", "cheb_2x"}:
        return _explicit_rhs_vertical_dealiased(state, grid)
    raise ValueError(f"Unsupported vertical_dealiasing={grid.vertical_dealiasing!r}")


# ---------------------------------------------------------------------------
# Implicit tendency (linear vertical coupling)
# ---------------------------------------------------------------------------

def implicit_tendency(state: State, grid: Grid) -> State:
    """Compute the implicit tendency (G_Z coupling + buoyancy).

    I_q  = G_Z @ w_hat
    I_w  = c(k) * G_Z @ q_hat + Ra/sigma * th_hat
    I_th = w_hat    (conduction gradient source)
    I_th_bar = (eps^2 / sigma) * G_Z2 @ th_bar
    """
    q_hat, w_hat, th_hat, th_bar = state
    w_cheb = _dirichlet_to_cheb(w_hat, grid.w_stencil)
    th_cheb = _dirichlet_to_cheb(th_hat, grid.dirichlet_stencil)

    dw_dZ = jnp.einsum('ij,j...->i...', grid.G_Z, w_cheb)
    dq_dZ = jnp.einsum('ij,j...->i...', grid.G_Z, q_hat)
    I_w_cheb = grid.inv_denom[None, :, :] * dq_dZ + grid.Ra_sigma * th_cheb
    I_w = _cheb_to_dirichlet(project_dirichlet(I_w_cheb, grid.proj_w), grid.w_pinv)

    # w's source in the theta equation: reduced coordinates differ when the
    # bases differ, so map w-basis -> theta-basis (identity when shared)
    I_th = (w_hat if grid.map_w_to_th is None
            else _apply_vertical_matrix(grid.map_w_to_th, w_hat))
    if grid.thermal_closure == "evolve_mean":
        I_th_bar = (grid.mean_temp_eps_sq / grid.sigma) * (grid.G_Z2 @ th_bar)
    else:
        I_th_bar = jnp.zeros_like(th_bar)

    return State(dw_dZ, I_w, I_th, I_th_bar)


# ---------------------------------------------------------------------------
# IMEX implicit solve (block elimination at each k)
# ---------------------------------------------------------------------------

def _per_shell_matmul(mat_shells: jnp.ndarray, ksq_idx: jnp.ndarray,
                      field: jnp.ndarray, chunk: int = 0) -> jnp.ndarray:
    """Apply per-|k|² shell matrices to a spectral field.

    With chunk=0 the shell gather materializes the full (Nx, Nk, m, m)
    tensor -- at large Nx*Nz that transient dominates peak VRAM (the
    recorded 114 GB at 64x256; ~4.4 GB per gather at 512²x64). chunk>0
    processes `chunk` kx rows at a time, capping the transient at
    chunk*Nk*m² while keeping each matmul large enough for GPU throughput.
    """
    f_t = jnp.transpose(field, (1, 2, 0))    # (Nx, Nk, m)
    Nx = f_t.shape[0]
    if chunk and 0 < chunk < Nx:
        n_blocks = -(-Nx // chunk)
        pad = n_blocks * chunk - Nx
        f_p = jnp.pad(f_t, ((0, pad), (0, 0), (0, 0)))
        idx_p = jnp.pad(ksq_idx, ((0, pad), (0, 0)))
        f_b = f_p.reshape(n_blocks, chunk, *f_p.shape[1:])
        idx_b = idx_p.reshape(n_blocks, chunk, idx_p.shape[1])

        def block(args):
            idx, fb = args
            return jnp.einsum('abij,abj->abi', mat_shells[idx], fb)

        r_t = jax.lax.map(block, (idx_b, f_b)).reshape(-1, *f_t.shape[1:])[:Nx]
    else:
        r_t = jnp.einsum('abij,abj->abi', mat_shells[ksq_idx], f_t)
    return jnp.transpose(r_t, (2, 0, 1))


def _q_stage_solve(R_q: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Apply the q-stage inverse.

    For q_boundary='none' the operator is diagonal: plain division by
    alpha_q(k) -- no dense matrices, no shell gather. Only the Neumann tau
    solve needs the per-shell matrices.
    """
    if grid.q_boundary == "none":
        return R_q * grid.inv_alpha_q[None, :, :]
    return _per_shell_matmul(grid.q_solve, grid.ksq_idx, R_q,
                             grid.imex_matmul_chunk)


def imex_implicit_solve(R_q: jnp.ndarray, R_w: jnp.ndarray,
                        R_th: jnp.ndarray,
                        grid: Grid) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve the IMEX implicit stage with q in Chebyshev and w/theta in Galerkin."""
    gamma = grid.gamma_imex
    dt = grid.dt

    # Step 0: Modified w RHS from buoyancy block-elimination (theta-basis
    # contribution mapped into w's basis when the bases differ)
    rth_scaled = R_th * grid.inv_alpha_th[None, :, :]
    if grid.map_th_to_w is not None:
        rth_scaled = _apply_vertical_matrix(grid.map_th_to_w, rth_scaled)
    R_w_eff = R_w + gamma * dt * grid.Ra_sigma * rth_scaled

    # Step 1: solve the q stage operator (scalar for 'none', per-shell tau
    # solve for 'neumann')
    Nz = grid.Nz
    if grid.q_boundary == "neumann":
        R_q = R_q.at[Nz - 1].set(0.0).at[Nz].set(0.0)
    temp = _q_stage_solve(R_q, grid)

    # Step 2: w RHS = R_w_eff + gamma*dt*c(k)*P_gal[G_Z @ temp]
    d_temp = jnp.einsum('ij,j...->i...', grid.G_Z, temp)
    d_temp_gal = _cheb_to_dirichlet(
        project_dirichlet(grid.inv_denom[None, :, :] * d_temp, grid.proj_w),
        grid.w_pinv,
    )
    rhs_w = R_w_eff + gamma * dt * d_temp_gal

    # Step 3: Solve A' @ w = rhs_w (per-shell, Galerkin basis)
    w_new = _per_shell_matmul(grid.imex_inv, grid.ksq_idx, rhs_w,
                              grid.imex_matmul_chunk)

    # Step 4: Back-substitute q = q_solve @ (R_q + gamma*dt*G_Z@w)
    w_cheb = _dirichlet_to_cheb(w_new, grid.w_stencil)
    dw_dZ = jnp.einsum('ij,j...->i...', grid.G_Z, w_cheb)
    combined = R_q + gamma * dt * dw_dZ
    if grid.q_boundary == "neumann":
        combined = combined.at[Nz - 1].set(0.0).at[Nz].set(0.0)
    q_new = _q_stage_solve(combined, grid)

    # Step 5: Back-substitute theta = (R_th + gamma*dt*w) / alpha_th
    # (w mapped into theta's basis when the bases differ)
    w_for_th = (w_new if grid.map_w_to_th is None
                else _apply_vertical_matrix(grid.map_w_to_th, w_new))
    th_new = (R_th + gamma * dt * w_for_th) * grid.inv_alpha_th[None, :, :]

    return q_new, w_new, th_new


def imex_mean_temp_solve(R_th_bar: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Solve the 1D implicit mean-temperature equation (coefficient space)."""
    if grid.thermal_closure != "evolve_mean":
        return project_dirichlet_1d(R_th_bar, grid.proj_dirichlet)

    gamma = grid.gamma_imex
    dt = grid.dt
    alpha = gamma * dt * grid.mean_temp_eps_sq / grid.sigma
    A = jnp.eye(grid.Nz + 1, dtype=R_th_bar.dtype) - alpha * grid.G_Z2
    # Dirichlet tau: replace last 2 rows
    N = grid.Nz
    e_plus = jnp.ones(N + 1, dtype=R_th_bar.dtype)
    e_minus = jnp.array([(-1.0)**n for n in range(N + 1)], dtype=R_th_bar.dtype)
    A = A.at[N - 1, :].set(e_plus)
    A = A.at[N, :].set(e_minus)
    # Zero tau rows of RHS
    rhs = R_th_bar.at[N - 1].set(0.0).at[N].set(0.0)
    return jnp.linalg.solve(A, rhs)


def balanced_midpoint_thermal_substep(state: State, grid: Grid) -> State:
    """Apply the balanced midpoint thermal exchange+diffusion substep.

    This implements the ``discretely_balanced_mean_fluctuation_thermal_formulation``
    idea with a frozen vertical velocity field over one full dt:
      1) solve a 1D linear system for ``th_bar^{n+1}``,
      2) update ``th_hat`` from the midpoint gradient of ``th_bar``.
    """
    if grid.thermal_closure != "evolve_mean":
        return state

    q_hat, w_hat, th_hat, th_bar_n = state
    dt = grid.dt
    mu = grid.mean_temp_eps_sq
    kappa = 1.0 / grid.sigma

    # Frozen work-grid velocity and current thermal state.
    w_work, th_work, _ = thermal_exchange_workgrid_fields(state, grid)
    flux_n_work = horizontal_mean_from_nodal_spectral(
        w_work, th_work, grid.Nx, grid.Npad
    )
    w2_work = horizontal_mean_from_nodal_spectral(
        w_work, w_work, grid.Nx, grid.Npad
    )

    # D_adj(F) = M_b^{-1} G_ex^T W_ex F.
    weighted_Gt = grid.G_exchange.T * grid.exchange_weights[None, :]
    D_adj = grid.mean_mass_inv @ weighted_Gt

    # K(Theta) = D_adj( diag(<w^2>) * G_ex Theta ).
    K = D_adj @ (w2_work[:, None] * grid.G_exchange)

    I = jnp.eye(grid.Nz + 1, dtype=th_bar_n.dtype)
    alpha = 0.5 * mu * dt * kappa
    beta = 0.25 * mu * (dt ** 2)
    L = grid.G_Z2

    A = I - alpha * L + beta * K
    b = (I + alpha * L - beta * K) @ th_bar_n + (mu * dt) * (D_adj @ flux_n_work)

    # Enforce homogeneous Dirichlet BCs via tau rows in coefficient space.
    N = grid.Nz
    e_plus = jnp.ones(N + 1, dtype=th_bar_n.dtype)
    e_minus = jnp.array([(-1.0) ** n for n in range(N + 1)], dtype=th_bar_n.dtype)
    A = A.at[N - 1, :].set(e_plus)
    A = A.at[N, :].set(e_minus)
    b = b.at[N - 1].set(0.0).at[N].set(0.0)

    th_bar_new = jnp.linalg.solve(A, b)

    # Theta midpoint update:
    #   th^{n+1} = th^n - dt/2 * w* * G(Theta^{n+1} + Theta^n).
    g_sum_work = grid.G_exchange @ (th_bar_new + th_bar_n)
    product_sum_work = g_sum_work[:, None, None] * w_work
    th_hat_new = th_hat + 0.5 * dt * paired_theta_feedback_from_work(
        product_sum_work, grid
    )

    return State(q_hat, w_hat, th_hat_new, th_bar_new)


def balanced_sbp2_thermal_substep(state: State, grid: Grid, sub_dt=None) -> State:
    """Apply the simpler uniform-grid SBP2 balanced thermal substep."""
    if grid.thermal_closure != "evolve_mean":
        return state

    q_hat, w_hat, _, _ = state
    dt = grid.dt if sub_dt is None else jnp.asarray(sub_dt, dtype=grid.dt.dtype)
    mu = grid.mean_temp_eps_sq
    kappa = 1.0 / grid.sigma

    w_sbp, th_sbp_n, th_bar_sbp_n, _ = sbp2_exchange_state_fields(state, grid)
    # For 2/3-rule, compute horizontal means on the Nx grid (mean is aliasing-
    # free via Parseval regardless of grid size; the padded path only adds FFT
    # work). For 3/2-rule, keep the padded path for consistency with the Jacobian.
    npad_for_mean = grid.Nx if grid.horizontal_dealiasing == "23_rule" else grid.Npad
    flux_n = horizontal_mean_from_nodal_spectral(w_sbp, th_sbp_n, grid.Nx, npad_for_mean)
    w2_mean = horizontal_mean_from_nodal_spectral(w_sbp, w_sbp, grid.Nx, npad_for_mean)

    I = jnp.eye(grid.Nz + 1, dtype=th_bar_sbp_n.dtype)
    M = jnp.diag(w2_mean)
    D1 = grid.sbp_D1
    L = grid.sbp_L

    A = I - 0.5 * mu * kappa * dt * L - 0.25 * mu * (dt ** 2) * (D1 @ M @ D1)
    B = I + 0.5 * mu * kappa * dt * L + 0.25 * mu * (dt ** 2) * (D1 @ M @ D1)
    rhs = B @ th_bar_sbp_n - mu * dt * (D1 @ flux_n)

    A = A.at[0, :].set(0.0)
    A = A.at[-1, :].set(0.0)
    A = A.at[0, 0].set(1.0)
    A = A.at[-1, -1].set(1.0)
    rhs = rhs.at[0].set(0.0)
    rhs = rhs.at[-1].set(0.0)

    th_bar_sbp_new = jnp.linalg.solve(A, rhs)
    g_half = 0.5 * (D1 @ (th_bar_sbp_n + th_bar_sbp_new))
    th_sbp_new = th_sbp_n - dt * w_sbp * g_half[:, None, None]
    th_sbp_new = th_sbp_new.at[0, :, :].set(0.0)
    th_sbp_new = th_sbp_new.at[-1, :, :].set(0.0)
    th_bar_sbp_new = th_bar_sbp_new.at[0].set(0.0)
    th_bar_sbp_new = th_bar_sbp_new.at[-1].set(0.0)

    th_cgl_new = _apply_vertical_matrix(grid.sbp_to_cgl, th_sbp_new)
    th_bar_cgl_new = _apply_vertical_matrix(grid.sbp_to_cgl, th_bar_sbp_new)
    th_cheb_new = project_dirichlet(_to_coeffs(th_cgl_new, grid.V_inv), grid.proj_dirichlet)
    th_hat_new = _cheb_to_dirichlet(th_cheb_new, grid.dirichlet_pinv)
    th_bar_new = project_dirichlet_1d(_to_coeffs_1d(th_bar_cgl_new, grid.V_inv), grid.proj_dirichlet)

    return State(q_hat, w_hat, th_hat_new, th_bar_new)


def _balanced_sbp2_thermal_substep_with_precomputed(
    state: State, grid: Grid, sub_dt, w_sbp, B_mat, A_lu_factor, npad_for_mean, mu,
) -> State:
    """Per-substep theta advance reusing precomputed substep-invariants.

    Numerically equivalent to `balanced_sbp2_thermal_substep` with sub_dt
    substituted, but skips the redundant (w_sbp, w2_mean, A, B, LU(A))
    rebuild that those invariants represent.
    """
    q_hat, w_hat, _, _ = state
    D1 = grid.sbp_D1

    # Only the theta-related fields vary across substeps.
    th_cheb = _dirichlet_to_cheb(state.th_hat, grid.dirichlet_stencil)
    th_cgl = _to_nodal(th_cheb, grid.V)
    th_bar_cgl = _to_nodal_1d(state.th_bar, grid.V)
    th_sbp_n = _apply_vertical_matrix(grid.cgl_to_sbp, th_cgl)
    th_bar_sbp_n = _apply_vertical_matrix(grid.cgl_to_sbp, th_bar_cgl)

    flux_n = horizontal_mean_from_nodal_spectral(w_sbp, th_sbp_n, grid.Nx, npad_for_mean)
    rhs = B_mat @ th_bar_sbp_n - mu * sub_dt * (D1 @ flux_n)
    rhs = rhs.at[0].set(0.0).at[-1].set(0.0)
    th_bar_sbp_new = jax.scipy.linalg.lu_solve(A_lu_factor, rhs)
    g_half = 0.5 * (D1 @ (th_bar_sbp_n + th_bar_sbp_new))
    th_sbp_new = th_sbp_n - sub_dt * w_sbp * g_half[:, None, None]
    th_sbp_new = th_sbp_new.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
    th_bar_sbp_new = th_bar_sbp_new.at[0].set(0.0).at[-1].set(0.0)

    th_cgl_new = _apply_vertical_matrix(grid.sbp_to_cgl, th_sbp_new)
    th_bar_cgl_new = _apply_vertical_matrix(grid.sbp_to_cgl, th_bar_sbp_new)
    th_cheb_new = project_dirichlet(_to_coeffs(th_cgl_new, grid.V_inv), grid.proj_dirichlet)
    th_hat_new = _cheb_to_dirichlet(th_cheb_new, grid.dirichlet_pinv)
    th_bar_new = project_dirichlet_1d(_to_coeffs_1d(th_bar_cgl_new, grid.V_inv), grid.proj_dirichlet)
    return State(q_hat, w_hat, th_hat_new, th_bar_new)


def _apply_balanced_sbp2_corrector(state: State, grid: Grid, total_dt) -> State:
    """Apply the SBP thermal corrector with optional subcycling.

    Hoists substep-invariants (w_sbp, <w^2>, matrix A, LU factorization) out
    of the substep loop. Physically and numerically equivalent to looping
    `balanced_sbp2_thermal_substep`, but only the theta-dependent solve /
    update runs per substep.
    """
    if grid.thermal_closure != "evolve_mean":
        return state

    n_substeps = max(1, int(grid.sbp_corrector_substeps))
    if n_substeps <= 0:
        return state

    dt_total = jnp.asarray(total_dt, dtype=grid.dt.dtype)
    sub_dt = dt_total / n_substeps
    mu = grid.mean_temp_eps_sq
    kappa = 1.0 / grid.sigma

    # --- Preamble: substep-invariant precomputes (done once per call) ---
    w_sbp, _, _, _ = sbp2_exchange_state_fields(state, grid)
    npad_for_mean = grid.Nx if grid.horizontal_dealiasing == "23_rule" else grid.Npad
    w2_mean = horizontal_mean_from_nodal_spectral(w_sbp, w_sbp, grid.Nx, npad_for_mean)

    I_mat = jnp.eye(grid.Nz + 1, dtype=w2_mean.dtype)
    M_mat = jnp.diag(w2_mean)
    D1 = grid.sbp_D1
    L_mat = grid.sbp_L
    half_term = 0.5 * mu * kappa * sub_dt * L_mat + 0.25 * mu * (sub_dt ** 2) * (D1 @ M_mat @ D1)
    A_mat = I_mat - half_term
    B_mat = I_mat + half_term
    A_mat = A_mat.at[0, :].set(0.0).at[-1, :].set(0.0).at[0, 0].set(1.0).at[-1, -1].set(1.0)
    A_lu_factor = jax.scipy.linalg.lu_factor(A_mat)

    # --- Substep loop: only theta-dependent work runs per substep ---
    out = state
    for _ in range(n_substeps):
        out = _balanced_sbp2_thermal_substep_with_precomputed(
            out, grid, sub_dt, w_sbp, B_mat, A_lu_factor, npad_for_mean, mu,
        )
    return out


def _thermal_correction_tendency(base_state: State, corrected_state: State,
                                 sub_dt) -> State:
    """Effective thermal tendency induced by a stage-local corrector."""
    dt_eff = jnp.asarray(sub_dt, dtype=corrected_state.th_bar.dtype)
    return State(
        jnp.zeros_like(base_state.q_hat),
        jnp.zeros_like(base_state.w_hat),
        (corrected_state.th_hat - base_state.th_hat) / dt_eff,
        (corrected_state.th_bar - base_state.th_bar) / dt_eff,
    )


def _add_stage_contrib(base: State, explicit_terms: list[tuple[float, State]],
                       implicit_terms: list[tuple[float, State]],
                       stage_states: list[State], dt: jnp.ndarray,
                       grid: Grid) -> State:
    """Assemble an IMEX stage RHS from prior explicit and implicit stages."""
    q_rhs, w_rhs, th_rhs, th_bar_rhs = base

    for coeff, expl in explicit_terms:
        q_rhs = q_rhs + dt * coeff * expl.q_hat
        w_rhs = w_rhs + dt * coeff * expl.w_hat
        th_rhs = th_rhs + dt * coeff * expl.th_hat
        th_bar_rhs = th_bar_rhs + dt * coeff * expl.th_bar

    for coeff, impl, stage in implicit_terms:
        q_rhs = q_rhs + dt * coeff * (
            impl.q_hat - grid.diss_rate_q[None, :, :] * stage.q_hat
        )
        w_rhs = w_rhs + dt * coeff * (
            impl.w_hat - grid.diss_rate_w[None, :, :] * stage.w_hat
        )
        th_rhs = th_rhs + dt * coeff * (
            impl.th_hat - grid.diss_rate_th[None, :, :] * stage.th_hat
        )
        th_bar_rhs = th_bar_rhs + dt * coeff * impl.th_bar

    return State(q_rhs, w_rhs, th_rhs, th_bar_rhs)


def _finalize_state(state: State, grid: Grid) -> State:
    """Apply post-step zero-mean constraints and optional cutoff."""
    q_hat = zero_mean(state.q_hat)
    w_hat = zero_mean(state.w_hat)
    th_hat = zero_mean(state.th_hat)
    state = _apply_vertical_cutoff(State(q_hat, w_hat, th_hat, state.th_bar), grid)
    return State(state.q_hat, state.w_hat, state.th_hat, state.th_bar)


def _implicit_with_diss(state: State, impl: State, grid: Grid) -> State:
    """Return the full implicit derivative including horizontal dissipation."""
    return State(
        impl.q_hat - grid.diss_rate_q[None, :, :] * state.q_hat,
        impl.w_hat - grid.diss_rate_w[None, :, :] * state.w_hat,
        impl.th_hat - grid.diss_rate_th[None, :, :] * state.th_hat,
        impl.th_bar,
    )


def imex_step_ars222(state: State, grid: Grid) -> State:
    """One full ARS(2,2,2) IMEX-RK time step."""
    gamma = grid.gamma_imex
    delta = -jnp.sqrt(jnp.array(2.0, dtype=grid.dt.dtype)) / 2.0
    dt = grid.dt

    q_n, w_n, th_n, th_bar_n = state

    # ──── Stage 1 ────
    E1 = explicit_rhs_dispatch(state, grid)

    R_q1 = q_n + gamma * dt * E1.q_hat
    R_w1 = w_n + gamma * dt * E1.w_hat
    R_th1 = th_n + gamma * dt * E1.th_hat
    R_th_bar1 = th_bar_n + gamma * dt * E1.th_bar

    q1, w1, th1 = imex_implicit_solve(R_q1, R_w1, R_th1, grid)
    th_bar1 = imex_mean_temp_solve(R_th_bar1, grid)

    state1 = State(q1, w1, th1, th_bar1)

    # ──── Stage 2 ────
    E2 = explicit_rhs_dispatch(state1, grid)
    I1 = implicit_tendency(state1, grid)

    omg = dt * (1 - gamma)
    R_q2 = q_n + dt * (delta * E1.q_hat + (1 - delta) * E2.q_hat) \
         + omg * I1.q_hat \
         - omg * grid.diss_rate_q[None, :, :] * q1
    R_w2 = w_n + dt * (delta * E1.w_hat + (1 - delta) * E2.w_hat) \
         + omg * I1.w_hat \
         - omg * grid.diss_rate_w[None, :, :] * w1
    R_th2 = th_n + dt * (delta * E1.th_hat + (1 - delta) * E2.th_hat) \
          + omg * I1.th_hat \
          - omg * grid.diss_rate_th[None, :, :] * th1
    R_th_bar2 = th_bar_n + dt * (delta * E1.th_bar + (1 - delta) * E2.th_bar) \
             + omg * I1.th_bar

    q2, w2, th2 = imex_implicit_solve(R_q2, R_w2, R_th2, grid)
    th_bar2 = imex_mean_temp_solve(R_th_bar2, grid)

    return _finalize_state(State(q2, w2, th2, th_bar2), grid)


def imex_step_rk443(state: State, grid: Grid) -> State:
    """One full ARS(4,4,3) / RK443 IMEX-RK time step.

    Coefficients match PETSc's ARKIMEX ARS443 implementation, which cites
    Ascher, Ruuth, and Spiteri (1997). Stages 2-5 share gamma=1/2.
    """
    dt = grid.dt
    y0 = state

    # Explicit stages E_j are evaluated at y_n, Y1, Y2, Y3.
    E0 = explicit_rhs_dispatch(y0, grid)

    rhs1 = State(
        y0.q_hat + 0.5 * dt * E0.q_hat,
        y0.w_hat + 0.5 * dt * E0.w_hat,
        y0.th_hat + 0.5 * dt * E0.th_hat,
        y0.th_bar + 0.5 * dt * E0.th_bar,
    )
    q1, w1, th1 = imex_implicit_solve(rhs1.q_hat, rhs1.w_hat, rhs1.th_hat, grid)
    th_bar1 = imex_mean_temp_solve(rhs1.th_bar, grid)
    Y1 = State(q1, w1, th1, th_bar1)
    K1 = _implicit_with_diss(Y1, implicit_tendency(Y1, grid), grid)

    E1 = explicit_rhs_dispatch(Y1, grid)
    rhs2 = State(
        y0.q_hat + dt * ((1.0 / 6.0) * K1.q_hat + (11.0 / 18.0) * E0.q_hat + (1.0 / 18.0) * E1.q_hat),
        y0.w_hat + dt * ((1.0 / 6.0) * K1.w_hat + (11.0 / 18.0) * E0.w_hat + (1.0 / 18.0) * E1.w_hat),
        y0.th_hat + dt * ((1.0 / 6.0) * K1.th_hat + (11.0 / 18.0) * E0.th_hat + (1.0 / 18.0) * E1.th_hat),
        y0.th_bar + dt * ((1.0 / 6.0) * K1.th_bar + (11.0 / 18.0) * E0.th_bar + (1.0 / 18.0) * E1.th_bar),
    )
    q2, w2, th2 = imex_implicit_solve(rhs2.q_hat, rhs2.w_hat, rhs2.th_hat, grid)
    th_bar2 = imex_mean_temp_solve(rhs2.th_bar, grid)
    Y2 = State(q2, w2, th2, th_bar2)
    K2 = _implicit_with_diss(Y2, implicit_tendency(Y2, grid), grid)

    E2 = explicit_rhs_dispatch(Y2, grid)
    rhs3 = State(
        y0.q_hat + dt * ((-0.5) * K1.q_hat + 0.5 * K2.q_hat + (5.0 / 6.0) * E0.q_hat - (5.0 / 6.0) * E1.q_hat + 0.5 * E2.q_hat),
        y0.w_hat + dt * ((-0.5) * K1.w_hat + 0.5 * K2.w_hat + (5.0 / 6.0) * E0.w_hat - (5.0 / 6.0) * E1.w_hat + 0.5 * E2.w_hat),
        y0.th_hat + dt * ((-0.5) * K1.th_hat + 0.5 * K2.th_hat + (5.0 / 6.0) * E0.th_hat - (5.0 / 6.0) * E1.th_hat + 0.5 * E2.th_hat),
        y0.th_bar + dt * ((-0.5) * K1.th_bar + 0.5 * K2.th_bar + (5.0 / 6.0) * E0.th_bar - (5.0 / 6.0) * E1.th_bar + 0.5 * E2.th_bar),
    )
    q3, w3, th3 = imex_implicit_solve(rhs3.q_hat, rhs3.w_hat, rhs3.th_hat, grid)
    th_bar3 = imex_mean_temp_solve(rhs3.th_bar, grid)
    Y3 = State(q3, w3, th3, th_bar3)
    K3 = _implicit_with_diss(Y3, implicit_tendency(Y3, grid), grid)

    E3 = explicit_rhs_dispatch(Y3, grid)
    rhs4 = State(
        y0.q_hat + dt * (1.5 * K1.q_hat - 1.5 * K2.q_hat + 0.5 * K3.q_hat + 0.25 * E0.q_hat + 1.75 * E1.q_hat + 0.75 * E2.q_hat - 1.75 * E3.q_hat),
        y0.w_hat + dt * (1.5 * K1.w_hat - 1.5 * K2.w_hat + 0.5 * K3.w_hat + 0.25 * E0.w_hat + 1.75 * E1.w_hat + 0.75 * E2.w_hat - 1.75 * E3.w_hat),
        y0.th_hat + dt * (1.5 * K1.th_hat - 1.5 * K2.th_hat + 0.5 * K3.th_hat + 0.25 * E0.th_hat + 1.75 * E1.th_hat + 0.75 * E2.th_hat - 1.75 * E3.th_hat),
        y0.th_bar + dt * (1.5 * K1.th_bar - 1.5 * K2.th_bar + 0.5 * K3.th_bar + 0.25 * E0.th_bar + 1.75 * E1.th_bar + 0.75 * E2.th_bar - 1.75 * E3.th_bar),
    )
    q4, w4, th4 = imex_implicit_solve(rhs4.q_hat, rhs4.w_hat, rhs4.th_hat, grid)
    th_bar4 = imex_mean_temp_solve(rhs4.th_bar, grid)
    Y4 = State(q4, w4, th4, th_bar4)
    K4 = _implicit_with_diss(Y4, implicit_tendency(Y4, grid), grid)

    y_np1 = State(
        y0.q_hat + dt * (1.5 * K1.q_hat - 1.5 * K2.q_hat + 0.5 * K3.q_hat + 0.5 * K4.q_hat + 0.25 * E0.q_hat + 1.75 * E1.q_hat + 0.75 * E2.q_hat - 1.75 * E3.q_hat),
        y0.w_hat + dt * (1.5 * K1.w_hat - 1.5 * K2.w_hat + 0.5 * K3.w_hat + 0.5 * K4.w_hat + 0.25 * E0.w_hat + 1.75 * E1.w_hat + 0.75 * E2.w_hat - 1.75 * E3.w_hat),
        y0.th_hat + dt * (1.5 * K1.th_hat - 1.5 * K2.th_hat + 0.5 * K3.th_hat + 0.5 * K4.th_hat + 0.25 * E0.th_hat + 1.75 * E1.th_hat + 0.75 * E2.th_hat - 1.75 * E3.th_hat),
        y0.th_bar + dt * (1.5 * K1.th_bar - 1.5 * K2.th_bar + 0.5 * K3.th_bar + 0.5 * K4.th_bar + 0.25 * E0.th_bar + 1.75 * E1.th_bar + 0.75 * E2.th_bar - 1.75 * E3.th_bar),
    )

    return _finalize_state(y_np1, grid)


# ---------------------------------------------------------------------------
# IMEX-RK steppers
# ---------------------------------------------------------------------------

def imex_step_balanced_midpoint(state: State, grid: Grid) -> State:
    """Split step: base IMEX without mean exchange/diffusion + balanced thermal substep."""
    # Base integrator: keep q-w and conductive-source machinery, while removing
    # evolve_mean coupling and mean diffusion from the core IMEX update.
    base_grid = grid._replace(
        thermal_closure="fixed_conduction",
        mean_exchange_discretization="legacy",
    )

    if grid.imex_scheme == "ars222":
        base_state = imex_step_ars222(state, base_grid)
    elif grid.imex_scheme == "rk443":
        base_state = imex_step_rk443(state, base_grid)
    else:
        raise ValueError(f"Unsupported imex_scheme={grid.imex_scheme!r}")

    corrected = balanced_midpoint_thermal_substep(base_state, grid)
    return _finalize_state(corrected, grid)


def imex_step_balanced_sbp2(state: State, grid: Grid) -> State:
    """Split step: base IMEX without mean exchange/diffusion + SBP2 thermal substep."""
    base_grid = grid._replace(
        thermal_closure="fixed_conduction",
        mean_exchange_discretization="legacy",
    )

    if grid.imex_scheme == "ars222":
        base_state = imex_step_ars222(state, base_grid)
    elif grid.imex_scheme == "rk443":
        base_state = imex_step_rk443(state, base_grid)
    else:
        raise ValueError(f"Unsupported imex_scheme={grid.imex_scheme!r}")

    corrected = _apply_balanced_sbp2_corrector(base_state, grid, grid.dt)
    return _finalize_state(corrected, grid)


def imex_step_balanced_sbp2_pc(state: State, grid: Grid) -> State:
    """Stage-wise ARS222 predictor/corrector using the SBP2 thermal map."""
    if grid.imex_scheme != "ars222":
        raise ValueError("balanced_sbp2_pc currently supports imex_scheme='ars222' only")

    base_grid = grid._replace(
        thermal_closure="fixed_conduction",
        mean_exchange_discretization="legacy",
    )

    gamma = grid.gamma_imex
    delta = -jnp.sqrt(jnp.array(2.0, dtype=grid.dt.dtype)) / 2.0
    dt = grid.dt
    alpha = gamma * dt
    omg = dt * (1 - gamma)

    q_n, w_n, th_n, th_bar_n = state

    # ──── Stage 1 predictor on the reduced system ────
    E1 = explicit_rhs_dispatch(state, base_grid)

    R_q1 = q_n + alpha * E1.q_hat
    R_w1 = w_n + alpha * E1.w_hat
    R_th1 = th_n + alpha * E1.th_hat
    R_th_bar1 = th_bar_n + alpha * E1.th_bar

    q1p, w1p, th1p = imex_implicit_solve(R_q1, R_w1, R_th1, base_grid)
    th_bar1p = imex_mean_temp_solve(R_th_bar1, base_grid)
    predictor1 = State(q1p, w1p, th1p, th_bar1p)

    # ──── Stage 1 thermal corrector ────
    state1 = _apply_balanced_sbp2_corrector(predictor1, grid, alpha)
    C1 = _thermal_correction_tendency(predictor1, state1, alpha)

    # ──── Stage 2 predictor on the reduced system ────
    E2 = explicit_rhs_dispatch(state1, base_grid)
    I1 = implicit_tendency(state1, base_grid)

    R_q2 = q_n + dt * (delta * E1.q_hat + (1 - delta) * E2.q_hat) \
         + omg * I1.q_hat \
         - omg * grid.diss_rate_q[None, :, :] * state1.q_hat
    R_w2 = w_n + dt * (delta * E1.w_hat + (1 - delta) * E2.w_hat) \
         + omg * I1.w_hat \
         - omg * grid.diss_rate_w[None, :, :] * state1.w_hat
    R_th2 = th_n + dt * (delta * E1.th_hat + (1 - delta) * E2.th_hat) \
          + omg * I1.th_hat \
          - omg * grid.diss_rate_th[None, :, :] * state1.th_hat \
          + omg * C1.th_hat
    R_th_bar2 = th_bar_n + dt * (delta * E1.th_bar + (1 - delta) * E2.th_bar) \
             + omg * I1.th_bar \
             + omg * C1.th_bar

    q2p, w2p, th2p = imex_implicit_solve(R_q2, R_w2, R_th2, base_grid)
    th_bar2p = imex_mean_temp_solve(R_th_bar2, base_grid)
    predictor2 = State(q2p, w2p, th2p, th_bar2p)

    # ──── Stage 2 thermal corrector ────
    corrected = _apply_balanced_sbp2_corrector(predictor2, grid, alpha)
    return _finalize_state(corrected, grid)


def imex_step(state: State, grid: Grid) -> State:
    """Dispatch to the configured IMEX-RK stepper."""
    if uses_balanced_midpoint_exchange(grid):
        new = imex_step_balanced_midpoint(state, grid)
    elif uses_balanced_sbp2_pc_exchange(grid):
        new = imex_step_balanced_sbp2_pc(state, grid)
    elif uses_balanced_sbp2_split_exchange(grid):
        new = imex_step_balanced_sbp2(state, grid)
    elif grid.imex_scheme == "ars222":
        new = imex_step_ars222(state, grid)
    elif grid.imex_scheme == "rk443":
        new = imex_step_rk443(state, grid)
    else:
        raise ValueError(f"Unsupported imex_scheme={grid.imex_scheme!r}")
    return sanitize_state(new, grid)


# ---------------------------------------------------------------------------
# RK4 explicit stepper (for comparison / validation)
# ---------------------------------------------------------------------------

def full_rhs(state: State, grid: Grid) -> State:
    """Full RHS = explicit + implicit."""
    E = explicit_rhs_dispatch(state, grid)
    I = implicit_tendency(state, grid)
    return State(
        E.q_hat + I.q_hat,
        E.w_hat + I.w_hat,
        E.th_hat + I.th_hat,
        E.th_bar + I.th_bar,
    )


def _apply_bcs(state: State, grid: Grid) -> State:
    """Enforce all BCs via tau projection."""
    q = _apply_q_boundary(state.q_hat, grid)
    w = _cheb_to_dirichlet(
        project_dirichlet(_dirichlet_to_cheb(state.w_hat, grid.w_stencil), grid.proj_w),
        grid.w_pinv,
    )
    th = _cheb_to_dirichlet(
        project_dirichlet(_dirichlet_to_cheb(state.th_hat, grid.dirichlet_stencil), grid.proj_dirichlet),
        grid.dirichlet_pinv,
    )
    th_bar = project_dirichlet_1d(state.th_bar, grid.proj_dirichlet)
    return _apply_vertical_cutoff(State(q, w, th, th_bar), grid)


def rk4_step(state: State, grid: Grid) -> State:
    """One explicit RK4 step. For validation only."""
    dt = grid.dt

    def rhs_bc(s):
        r = full_rhs(s, grid)
        return r

    k1 = rhs_bc(state)
    s2 = State(state.q_hat + 0.5 * dt * k1.q_hat,
               state.w_hat + 0.5 * dt * k1.w_hat,
               state.th_hat + 0.5 * dt * k1.th_hat,
               state.th_bar + 0.5 * dt * k1.th_bar)
    s2 = _apply_bcs(s2, grid)

    k2 = rhs_bc(s2)
    s3 = State(state.q_hat + 0.5 * dt * k2.q_hat,
               state.w_hat + 0.5 * dt * k2.w_hat,
               state.th_hat + 0.5 * dt * k2.th_hat,
               state.th_bar + 0.5 * dt * k2.th_bar)
    s3 = _apply_bcs(s3, grid)

    k3 = rhs_bc(s3)
    s4 = State(state.q_hat + dt * k3.q_hat,
               state.w_hat + dt * k3.w_hat,
               state.th_hat + dt * k3.th_hat,
               state.th_bar + dt * k3.th_bar)
    s4 = _apply_bcs(s4, grid)

    k4 = rhs_bc(s4)

    q_new = state.q_hat + (dt / 6) * (k1.q_hat + 2 * k2.q_hat + 2 * k3.q_hat + k4.q_hat)
    w_new = state.w_hat + (dt / 6) * (k1.w_hat + 2 * k2.w_hat + 2 * k3.w_hat + k4.w_hat)
    th_new = state.th_hat + (dt / 6) * (k1.th_hat + 2 * k2.th_hat + 2 * k3.th_hat + k4.th_hat)
    th_bar_new = state.th_bar + (dt / 6) * (
        k1.th_bar + 2 * k2.th_bar + 2 * k3.th_bar + k4.th_bar
    )

    # Post-step dissipation (exponential, for RK4 only)
    q_new = q_new * grid.diss_q[None, :, :]
    w_new = w_new * grid.diss_w[None, :, :]
    th_new = th_new * grid.diss_th[None, :, :]

    q_new = zero_mean(q_new)
    w_new = zero_mean(w_new)
    th_new = zero_mean(th_new)

    # BC projection
    q_new = _apply_q_boundary(q_new, grid)
    w_new = _cheb_to_dirichlet(
        project_dirichlet(_dirichlet_to_cheb(w_new, grid.w_stencil), grid.proj_w),
        grid.w_pinv,
    )
    th_new = _cheb_to_dirichlet(
        project_dirichlet(_dirichlet_to_cheb(th_new, grid.dirichlet_stencil), grid.proj_dirichlet),
        grid.dirichlet_pinv,
    )
    th_bar_new = project_dirichlet_1d(th_bar_new, grid.proj_dirichlet)

    state_new = _apply_vertical_cutoff(State(q_new, w_new, th_new, th_bar_new), grid)
    return sanitize_state(state_new, grid)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def make_initial_state(grid: Grid, seed: int = 0,
                       amplitude: float = 1e-3) -> State:
    """Random small-amplitude initial perturbation in q'.

    Vertical structure: sin(pi*Z) (most unstable linear mode).
    Generated in nodal space, then transformed to Chebyshev coefficients.
    """
    Nz1 = grid.Nz + 1
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    # Random spectral coefficients at each CGL node
    q_real = jax.random.normal(k1, (Nz1, grid.Nx, grid.Nk))
    q_imag = jax.random.normal(k2, (Nz1, grid.Nx, grid.Nk))
    q_nodal = amplitude * (q_real + 1j * q_imag)

    # Apply sin(pi*Z) vertical envelope (at CGL nodes)
    envelope = jnp.sin(jnp.pi * grid.Z)
    q_nodal = q_nodal * envelope[:, None, None]

    # Transform to Chebyshev coefficients
    q_hat = _to_coeffs(q_nodal, grid.V_inv)

    # Zero mean and optional q-boundary projection
    q_hat = zero_mean(q_hat)
    q_hat = _apply_q_boundary(q_hat, grid)

    # w and theta start at zero (in Dirichlet Galerkin space)
    gal_shape = (grid.Nz_gal, grid.Nx, grid.Nk)
    w_hat = jnp.zeros(gal_shape, dtype=q_hat.dtype)
    th_hat = jnp.zeros(gal_shape, dtype=q_hat.dtype)
    th_bar = jnp.zeros(Nz1, dtype=grid.Z.dtype)

    # Random complex noise is not Hermitian in the redundant rfft2 columns --
    # without this projection the initial condition seeds the ghost mode.
    return sanitize_state(State(q_hat, w_hat, th_hat, th_bar), grid)


# ---------------------------------------------------------------------------
# Main time loop
# ---------------------------------------------------------------------------

def run(grid: Grid, state: State, n_steps: int,
        save_interval: int, use_imex: bool = True,
        callback=None) -> tuple[State, list]:
    """Run the solver for n_steps time steps."""
    stepper = imex_step if use_imex else rk4_step

    # Cover every entry path (fresh init, restart, hand-built states): the
    # steppers keep the state sanitized, but the incoming state may not be.
    state = sanitize_state(state, grid)

    @jax.jit
    def scan_body(state, _):
        return stepper(state, grid), None

    n_outer = n_steps // save_interval
    snapshots = []

    for i_outer in range(n_outer):
        state, _ = jax.lax.scan(scan_body, state, None, length=save_interval)
        step = (i_outer + 1) * save_interval
        t = step * float(grid.dt)

        if callback is not None:
            callback(state, step, t)

        snapshots.append((t, state))

    remainder = n_steps % save_interval
    if remainder > 0:
        state, _ = jax.lax.scan(scan_body, state, None, length=remainder)

    return state, snapshots
