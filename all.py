"""Frozen configuration dataclass for the NHQGE solver."""

from __future__ import annotations

import dataclasses
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class NHQGConfig:
    """Immutable configuration for an NHQGE simulation.

    All physical and numerical parameters needed by the solver.
    """

    # --- Resolution ---
    Nx: int = 256          # Horizontal grid points (square domain)
    Nz: int = 32           # Number of Chebyshev intervals (Nz+1 CGL points)

    # --- Physical parameters ---
    Ra_tilde: float = 100.0   # Reduced Rayleigh number
    sigma: float = 1.0        # Prandtl number
    beta: float = 0.0         # PV gradient
    Ld: float = float('inf')  # Deformation radius (inf = barotropic limit)
    L: float = 20.0           # Horizontal domain size (in units of Lc)
    thermal_closure: str = "fixed_conduction"  # or "evolve_mean"
    mean_temp_eps_sq: float = 1.0  # epsilon^2 prefactor in d_t Theta_bar
    q_boundary: str = "none"  # "none" (Miquel-style) or "neumann" (stress-free q')
    nonlinear_advection: str = "jacobian"  # "jacobian" or "flux"

    # --- Time stepping ---
    dt: float = 1e-3          # Time step
    t_final: float = 100.0    # End time
    imex_scheme: str = "ars222"  # "ars222" or "rk443"

    # --- Dissipation ---
    nu_q: float = 0.0         # PV hyperviscosity coefficient
    hyper_order: int = 4      # Hyperviscosity order p (nabla^{2p})
    nu_w: float = 0.0         # Laplacian diffusion on w
    nu_theta: float = 0.0     # Laplacian diffusion on theta
    drag: float = 0.0         # Large-scale linear drag on q'

    # --- Output ---
    save_interval: int = 100  # Steps between snapshots
    output_dir: str = "output"

    # --- Precision ---
    float_dtype: str = "float32"  # "float32" or "float64"
    vertical_cutoff_n: int | None = None  # Optional high-n cutoff for w,theta
    vertical_dealiasing: str = "none"  # "none", "cheb_3o2", or "cheb_2x"

    @property
    def Nk(self) -> int:
        """Number of rfft2 modes along the last horizontal axis."""
        return self.Nx // 2 + 1

    @property
    def Npad(self) -> int:
        """Padded grid size for 3/2-rule dealiasing."""
        return 3 * self.Nx // 2

    @property
    def dx(self) -> float:
        """Horizontal grid spacing."""
        return self.L / self.Nx

    @property
    def Ld_inv_sq(self) -> float:
        """1/Ld^2, safe for Ld=inf."""
        if self.Ld == float('inf'):
            return 0.0
        return 1.0 / (self.Ld ** 2)

    @property
    def jnp_dtype(self):
        return jnp.float32 if self.float_dtype == "float32" else jnp.float64

    @property
    def complex_dtype(self):
        return jnp.complex64 if self.float_dtype == "float32" else jnp.complex128

    @property
    def n_outputs(self) -> int:
        """Number of output snapshots."""
        total_steps = int(self.t_final / self.dt)
        return total_steps // self.save_interval

    def with_updates(self, **kwargs) -> NHQGConfig:
        """Return a new config with specified fields updated."""
        return dataclasses.replace(self, **kwargs)
"""Diagnostic quantities: spectra, integral diagnostics, and KE shell budgets."""

from __future__ import annotations

import jax.numpy as jnp

from nhqg.grid import Grid
from nhqg.solver import (
    State,
    _cheb_to_dirichlet,
    _to_coeffs,
    _to_nodal_1d,
    _truncate_cheb_coeffs,
    _dirichlet_to_cheb,
    _to_nodal,
    explicit_rhs_dispatch,
    implicit_tendency,
    invert_psi,
    project_dirichlet,
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
    Nusselt = 1.0 + vol_avg_tw

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
        'vol_avg_tw': vol_avg_tw,
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
    }
"""Grid infrastructure: CGL points, coefficient-space Chebyshev operators,
tau-method BCs, wavenumber grids, dissipation, and IMEX precomputed inverses.

Uses the Galerkin/tau approach: fields are stored as Chebyshev coefficients,
derivatives use the coefficient-space recurrence, and BCs are enforced via
tau rows (replacing the last two equations with boundary constraints).

This eliminates the collocation D_Z null mode that caused instability at
high Nz.  The IMEX shell infrastructure (precomputed dense inverses per
|k|^2 shell) is unchanged.

Single entry point: ``make_grid(cfg) -> Grid``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from nhqg.config import NHQGConfig


# ---------------------------------------------------------------------------
# Grid container
# ---------------------------------------------------------------------------

class Grid(NamedTuple):
    """All precomputed arrays needed by the solver, created once."""

    # Vertical grid (CGL points, for transforms and diagnostics)
    Z: jnp.ndarray            # (Nz+1,) CGL points in [0,1]
    xi: jnp.ndarray           # (Nz+1,) CGL points in [-1,1]
    cc_weights: jnp.ndarray   # (Nz+1,) Clenshaw-Curtis quadrature weights on [0,1]

    # Coefficient-space vertical operators
    G_Z: jnp.ndarray          # (Nz+1, Nz+1) d/dZ in coefficient space
    G_Z2: jnp.ndarray         # (Nz+1, Nz+1) d²/dZ² in coefficient space

    # Chebyshev transforms (coefficient <-> nodal)
    V: jnp.ndarray            # (Nz+1, Nz+1) coeffs -> nodal values
    V_inv: jnp.ndarray        # (Nz+1, Nz+1) nodal values -> coeffs
    V_dealias: jnp.ndarray    # (Nz_dealias+1, Nz+1) coeffs -> overresolved nodal values
    V_dealias_inv: jnp.ndarray  # (Nz_dealias+1, Nz_dealias+1) overresolved nodal -> coeffs
    dirichlet_stencil: jnp.ndarray  # (Nz+1, Nz-1) Galerkin Dirichlet -> Chebyshev
    dirichlet_pinv: jnp.ndarray     # (Nz-1, Nz+1) unique-Chebyshev left inverse

    # Horizontal wavenumber grid
    kx: jnp.ndarray           # (Nx, 1) wavenumber array (full axis)
    ky: jnp.ndarray           # (1, Nk) wavenumber array (rfft axis)
    ksq: jnp.ndarray          # (Nx, Nk) |k|^2
    inv_denom: jnp.ndarray    # (Nx, Nk) 1/(|k|^2 + Ld^{-2}), k=0 -> 0

    # Dissipation: exponential multipliers (for RK4 validator only)
    diss_q: jnp.ndarray       # (Nx, Nk) exp(-diss_rate_q * dt)
    diss_w: jnp.ndarray       # (Nx, Nk) exp(-diss_rate_w * dt)
    diss_th: jnp.ndarray      # (Nx, Nk) exp(-diss_rate_th * dt)

    # Dissipation: raw rates and IMEX alpha factors (for unified IMEX)
    diss_rate_q: jnp.ndarray  # (Nx, Nk) nu_q*|k|^{2p} + drag
    diss_rate_w: jnp.ndarray  # (Nx, Nk) nu_w*|k|^{2p}
    diss_rate_th: jnp.ndarray # (Nx, Nk) (nu_theta/sigma)*|k|^{2p}
    inv_alpha_q: jnp.ndarray  # (Nx, Nk) 1/(1 + gamma*dt*diss_rate_q)
    inv_alpha_th: jnp.ndarray # (Nx, Nk) 1/(1 + gamma*dt*diss_rate_th)

    # IMEX infrastructure (inverses per |k|^2 shell)
    imex_inv: jnp.ndarray     # (n_shells, Nz-1, Nz-1) precomputed A'^{-1} in Dirichlet basis
    q_solve: jnp.ndarray      # (n_shells, Nz+1, Nz+1) q-stage solve operator
    ksq_idx: jnp.ndarray      # (Nx, Nk) int32, maps (kx,ky) -> shell index

    # Tau BC projection matrices (for RK4 / post-step BC enforcement)
    proj_dirichlet: jnp.ndarray  # (Nz+1, Nz+1) projects coeffs to satisfy Dirichlet
    proj_neumann: jnp.ndarray    # (Nz+1, Nz+1) projects coeffs to satisfy Neumann

    # Scalar parameters (as 0-d arrays for JIT compatibility)
    beta: jnp.ndarray
    Ra_sigma: jnp.ndarray
    sigma: jnp.ndarray
    L: jnp.ndarray
    Ld_inv_sq: jnp.ndarray
    dt: jnp.ndarray
    gamma_imex: jnp.ndarray
    mean_temp_eps_sq: jnp.ndarray

    # Static integers (not traced by JAX)
    Nx: int
    Nk: int
    Nz: int
    Nz_gal: int
    Nz_dealias: int
    Npad: int
    thermal_closure: str
    q_boundary: str
    nonlinear_advection: str
    vertical_cutoff_n: int | None
    imex_scheme: str
    vertical_dealiasing: str


# ---------------------------------------------------------------------------
# Chebyshev coefficient-space differentiation matrix
# ---------------------------------------------------------------------------

def _cheb_coeff_diff_matrix(N: int, dtype=np.float64) -> np.ndarray:
    """Map Chebyshev coefficients to first-derivative coefficients.

    If f(x) = sum_{n=0}^N a_n T_n(x), this returns the matrix G such that
    b = G @ a gives f'(x) = sum_{n=0}^N b_n T_n(x).
    """
    G = np.zeros((N + 1, N + 1), dtype=dtype)

    for n in range(N + 1):
        a = np.zeros(N + 1, dtype=dtype)
        a[n] = 1.0
        b = np.zeros(N + 1, dtype=dtype)

        if N >= 1:
            b[N - 1] = 2.0 * N * a[N]
            for k in range(N - 2, 0, -1):
                b[k] = b[k + 2] + 2.0 * (k + 1) * a[k + 1]
            b[0] = (0.5 * b[2] if N >= 2 else 0.0) + a[1]

        G[:, n] = b

    return G


# ---------------------------------------------------------------------------
# Clenshaw-Curtis quadrature weights on [0, 1]
# ---------------------------------------------------------------------------

def _cc_weights(N: int, dtype=np.float64) -> np.ndarray:
    """Clenshaw-Curtis weights for N+1 CGL points, mapped to [0,1]."""
    theta = np.pi * np.arange(N + 1, dtype=dtype) / N
    w = np.zeros(N + 1, dtype=dtype)

    for j in range(N + 1):
        s = 0.0
        for k in range(1, N // 2 + 1):
            b = 1.0 if k == N // 2 else 2.0
            s += b * np.cos(2.0 * k * theta[j]) / (4.0 * k * k - 1.0)
        c_j = 2.0 if (j == 0 or j == N) else 1.0
        w[j] = (1.0 - s) / (N * c_j)

    return w


def _cheb_vandermonde_and_inverse(N: int, dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
    """Return DCT-I style Chebyshev coeff<->nodal transforms for N+1 CGL points."""
    j_idx = np.arange(N + 1, dtype=dtype)
    V = np.cos(np.outer(j_idx, j_idx) * (np.pi / N)).astype(dtype)
    c = np.ones(N + 1, dtype=dtype)
    c[0] = 2.0
    c[N] = 2.0
    inv_c = 1.0 / c
    V_inv = (2.0 / N) * np.outer(inv_c, inv_c) * V.T
    return V, V_inv


def _cheb_gauss_vandermonde_and_inverse(N: int, dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
    """Return Coral-style Chebyshev coeff<->nodal transforms on a Gauss grid.

    The work grid has ``N`` Gauss-Chebyshev points
    ``x_j = cos(pi * (j + 1/2) / N)``, with the corresponding DCT-II inverse
    pair used in Coral's overresolved nonlinear transform path.
    """
    j_idx = np.arange(N, dtype=dtype)
    n_idx = np.arange(N, dtype=dtype)
    theta = np.pi * (j_idx + 0.5) / N
    V = np.cos(np.outer(theta, n_idx)).astype(dtype)
    V_inv = ((2.0 / N) * V.T).astype(dtype)
    V_inv[0, :] *= 0.5
    return V, V_inv


# ---------------------------------------------------------------------------
# Tau BC projection matrices
# ---------------------------------------------------------------------------

def _build_tau_projection(tau_rows: np.ndarray, Nz: int,
                          dtype=np.float64) -> np.ndarray:
    """Build a projection matrix that adjusts coefficients a_{N-1}, a_N
    so that the tau constraints are satisfied.

    tau_rows: (2, Nz+1) — the two BC constraint row vectors.
    Returns: (Nz+1, Nz+1) projection matrix P such that
             tau_rows @ (P @ a) = 0 for any input a.
    """
    N = Nz
    # We solve for a_{N-1}, a_N from the other coefficients:
    #   tau_rows @ a = 0
    #   [tau[0, N-1]  tau[0, N]] [a_{N-1}]   = -tau[0, :N-1] @ a[:N-1]
    #   [tau[1, N-1]  tau[1, N]] [a_{N-1}]   = -tau[1, :N-1] @ a[:N-1]
    M = tau_rows[:, N-1:N+1]  # (2, 2)
    R = tau_rows[:, :N-1]     # (2, N-1)
    M_inv = np.linalg.inv(M)
    # a_{N-1:N+1} = -M_inv @ R @ a_{:N-1}
    # Build full projection: P @ a = [a_0, ..., a_{N-2}, new_{N-1}, new_N]
    P = np.eye(N + 1, dtype=dtype)
    P[N-1, :N-1] = -(M_inv @ R)[0, :]
    P[N-1, N-1] = 0.0
    P[N-1, N] = 0.0
    P[N, :N-1] = -(M_inv @ R)[1, :]
    P[N, N-1] = 0.0
    P[N, N] = 0.0
    return P


# ---------------------------------------------------------------------------
# IMEX inverse precomputation with |k|² shell dedup (tau method)
# ---------------------------------------------------------------------------

def _build_imex_inv(G_Z: np.ndarray, dirichlet_stencil: np.ndarray,
                    dirichlet_pinv: np.ndarray, ksq_flat: np.ndarray,
                    tau_neu: np.ndarray, tau_dir: np.ndarray,
                    Ld_inv_sq: float, dt: float, gamma: float, Nz: int,
                    nu_q: float, nu_w: float, nu_theta: float,
                    sigma: float, Ra_sigma: float,
                    drag: float, hyper_order: int,
                    q_boundary: str, dtype=np.float64):
    """Precompute IMEX inverse matrices for the q-w block.

    Dirichlet BCs for w are always enforced via tau rows. For q, the solve is
    either unconstrained (Miquel-style, ``q_boundary='none'``) or uses the
    historical Neumann tau rows (``q_boundary='neumann'``).
    """
    N = Nz
    I = np.eye(N + 1, dtype=dtype)

    # P_tau: zeroes the last two rows (tau rows) of the RHS
    P_tau = I.copy()
    P_tau[N - 1, N - 1] = 0.0
    P_tau[N, N] = 0.0

    tau_neu_top = tau_neu[0]  # (N+1,) Neumann at Z=1
    tau_neu_bot = tau_neu[1]  # (N+1,) Neumann at Z=0
    tau_dir_top = tau_dir[0]  # (N+1,) Dirichlet at Z=1
    tau_dir_bot = tau_dir[1]  # (N+1,) Dirichlet at Z=0

    ksq_rounded = np.round(ksq_flat, decimals=8)
    unique_ksq, inverse_idx = np.unique(ksq_rounded, return_inverse=True)
    n_shells = len(unique_ksq)

    N_gal = N - 1
    inv_matrices = np.zeros((n_shells, N_gal, N_gal), dtype=dtype)
    q_solve_matrices = np.zeros((n_shells, N + 1, N + 1), dtype=dtype)

    for s, ksq_val in enumerate(unique_ksq):
        ksq_p = ksq_val ** hyper_order
        alpha_q = 1.0 + gamma * dt * (nu_q * ksq_p + drag)
        alpha_w = 1.0 + gamma * dt * nu_w * ksq_p
        alpha_th = 1.0 + gamma * dt * (nu_theta / sigma) * ksq_p

        alpha_w_eff = alpha_w - (gamma * dt) ** 2 * Ra_sigma / alpha_th

        if q_boundary == 'neumann':
            N_q = alpha_q * I.copy()
            N_q[N - 1, :] = tau_neu_top
            N_q[N, :] = tau_neu_bot
            q_solve = np.linalg.inv(N_q) @ P_tau
        elif q_boundary == 'none':
            q_solve = (1.0 / alpha_q) * I
        else:
            raise ValueError(f"Unsupported q_boundary={q_boundary!r}")
        q_solve_matrices[s] = q_solve

        denom = ksq_val + Ld_inv_sq
        if denom == 0.0:
            A = alpha_w_eff * np.eye(N_gal, dtype=dtype)
            inv_matrices[s] = np.linalg.inv(A)
        else:
            c_k = 1.0 / denom
            B = dirichlet_pinv @ G_Z @ q_solve @ G_Z @ dirichlet_stencil
            A = alpha_w_eff * np.eye(N_gal, dtype=dtype) - (gamma * dt) ** 2 * c_k * B
            inv_matrices[s] = np.linalg.inv(A)

    return inv_matrices, q_solve_matrices, inverse_idx.astype(np.int32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def make_grid(cfg: NHQGConfig) -> Grid:
    """Build all precomputed grid arrays from a configuration."""

    Nx, Nz, L = cfg.Nx, cfg.Nz, cfg.L
    Nk = cfg.Nk
    Npad = cfg.Npad
    Ld_inv_sq = cfg.Ld_inv_sq
    dt = cfg.dt
    fdtype = np.float64 if cfg.float_dtype == "float64" else np.float32
    build_dtype = np.float64
    N = Nz

    # ── Vertical grid ──
    j_idx = np.arange(N + 1, dtype=build_dtype)
    xi = np.cos(np.pi * j_idx / N)
    Z = 0.5 * (1.0 + xi)

    cc_w = _cc_weights(N, dtype=build_dtype)

    # ── Coefficient-space derivative operators ──
    # G_xi: d/dxi in Chebyshev coefficient space
    # G_Z = 2*G_xi: d/dZ where Z = (1+xi)/2
    G_xi = _cheb_coeff_diff_matrix(N, dtype=build_dtype)
    G_Z_np = 2.0 * G_xi
    G_Z2_np = G_Z_np @ G_Z_np

    # ── Chebyshev Vandermonde and its inverse ──
    # V[j, n] = T_n(xi_j) = cos(n*pi*j/N) — coefficients to nodal values
    V_np, V_inv_np = _cheb_vandermonde_and_inverse(N, dtype=build_dtype)

    if cfg.vertical_dealiasing == "none":
        N_dealias = N
        V_dealias_np = V_np
        V_dealias_inv_np = V_inv_np
    elif cfg.vertical_dealiasing in {"cheb_3o2", "cheb_2x"}:
        if cfg.vertical_dealiasing == "cheb_3o2":
            # Coral-style work arrays use NZAA = 3*NZ/2 on a Gauss-Chebyshev grid.
            N_dealias = max(N + 1, (3 * N) // 2)
            V_hi_np, V_hi_inv_np = _cheb_gauss_vandermonde_and_inverse(
                N_dealias, dtype=build_dtype
            )
        else:
            N_dealias = 2 * N
            V_hi_np, V_hi_inv_np = _cheb_vandermonde_and_inverse(
                N_dealias, dtype=build_dtype
            )
        V_dealias_np = V_hi_np[:, :N + 1]
        V_dealias_inv_np = V_hi_inv_np
    else:
        raise ValueError(f"Unsupported vertical_dealiasing={cfg.vertical_dealiasing!r}")

    # Dirichlet Galerkin stencil used by Coral for both-Dirichlet fields:
    # basis_j = -T_j + T_{j+2}, j = 0..N-2
    dirichlet_stencil_np = np.zeros((N + 1, N - 1), dtype=build_dtype)
    for j in range(N - 1):
        dirichlet_stencil_np[j, j] = -1.0
        dirichlet_stencil_np[j + 2, j] = 1.0
    dirichlet_unique_np = dirichlet_stencil_np[:N - 1, :]
    dirichlet_unique_inv_np = np.linalg.inv(dirichlet_unique_np)
    dirichlet_pinv_np = np.zeros((N - 1, N + 1), dtype=build_dtype)
    dirichlet_pinv_np[:, :N - 1] = dirichlet_unique_inv_np

    # ── Tau BC row vectors ──
    # Dirichlet: f(xi=+1)=0 → sum_n a_n = 0;  f(xi=-1)=0 → sum (-1)^n a_n = 0
    e_plus = np.ones(N + 1, dtype=build_dtype)           # T_n(+1) = 1
    e_minus = np.array([(-1.0)**n for n in range(N + 1)], dtype=build_dtype)  # T_n(-1)
    tau_dir = np.stack([e_plus, e_minus])  # (2, N+1)

    # Neumann: f'(xi=+1)=0 and f'(xi=-1)=0
    # f'(xi) = sum b_n T_n(xi), b = G_xi @ a.  f'(+1) = e_+ @ b = e_+ @ G_xi @ a
    # Use G_Z (= 2*G_xi) since d/dZ = 0 ⟺ d/dxi = 0 (factor of 2 irrelevant for = 0)
    tau_neu = np.stack([e_plus @ G_Z_np, e_minus @ G_Z_np])  # (2, N+1)

    # ── Tau projection matrices (for RK4 / post-step BC enforcement) ──
    proj_dir_np = _build_tau_projection(tau_dir, N, dtype=build_dtype)
    proj_neu_np = _build_tau_projection(tau_neu, N, dtype=build_dtype)

    # ── Horizontal wavenumber grid ──
    kx_1d = 2.0 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    ky_1d = 2.0 * np.pi * np.arange(Nk) / L

    kx_2d = kx_1d[:, None]
    ky_2d = ky_1d[None, :]
    ksq_np = kx_2d ** 2 + ky_2d ** 2

    denom = ksq_np + Ld_inv_sq
    inv_denom = np.zeros_like(denom)
    np.divide(1.0, denom, out=inv_denom, where=denom > 0)

    # ── Dissipation rates and IMEX alpha factors ──
    p = cfg.hyper_order
    if cfg.imex_scheme == "ars222":
        gamma_imex = 1.0 - 1.0 / np.sqrt(2.0)
    elif cfg.imex_scheme == "rk443":
        gamma_imex = 0.5
    else:
        raise ValueError(f"Unsupported imex_scheme={cfg.imex_scheme!r}")

    diss_rate_q_np = cfg.nu_q * ksq_np ** p + cfg.drag
    diss_rate_w_np = cfg.nu_w * ksq_np ** p
    diss_rate_th_np = (cfg.nu_theta / cfg.sigma) * ksq_np ** p

    diss_q_np = np.exp(-diss_rate_q_np * dt)
    diss_w_np = np.exp(-diss_rate_w_np * dt)
    diss_th_np = np.exp(-diss_rate_th_np * dt)

    inv_alpha_q_np = 1.0 / (1.0 + gamma_imex * dt * diss_rate_q_np)
    inv_alpha_th_np = 1.0 / (1.0 + gamma_imex * dt * diss_rate_th_np)

    # ── IMEX inverse matrices (tau method, coefficient space) ──
    ksq_flat = ksq_np.ravel()
    inv_matrices, q_solve_matrices, ksq_idx_flat = _build_imex_inv(
        G_Z_np, dirichlet_stencil_np, dirichlet_pinv_np, ksq_flat, tau_neu, tau_dir,
        Ld_inv_sq, dt, gamma_imex, Nz,
        cfg.nu_q, cfg.nu_w, cfg.nu_theta, cfg.sigma,
        cfg.Ra_tilde / cfg.sigma, cfg.drag, cfg.hyper_order,
        cfg.q_boundary, dtype=build_dtype
    )
    ksq_idx_2d = ksq_idx_flat.reshape(Nx, Nk)

    # ── Cast to target dtype ──
    def to_jax(arr, dtype=None):
        if dtype is None:
            dtype = fdtype
        return jnp.array(arr, dtype=dtype)

    return Grid(
        Z=to_jax(Z),
        xi=to_jax(xi),
        cc_weights=to_jax(cc_w),
        G_Z=to_jax(G_Z_np),
        G_Z2=to_jax(G_Z2_np),
        V=to_jax(V_np),
        V_inv=to_jax(V_inv_np),
        V_dealias=to_jax(V_dealias_np),
        V_dealias_inv=to_jax(V_dealias_inv_np),
        dirichlet_stencil=to_jax(dirichlet_stencil_np),
        dirichlet_pinv=to_jax(dirichlet_pinv_np),
        kx=to_jax(kx_2d),
        ky=to_jax(ky_2d),
        ksq=to_jax(ksq_np),
        inv_denom=to_jax(inv_denom),
        diss_q=to_jax(diss_q_np),
        diss_w=to_jax(diss_w_np),
        diss_th=to_jax(diss_th_np),
        diss_rate_q=to_jax(diss_rate_q_np),
        diss_rate_w=to_jax(diss_rate_w_np),
        diss_rate_th=to_jax(diss_rate_th_np),
        inv_alpha_q=to_jax(inv_alpha_q_np),
        inv_alpha_th=to_jax(inv_alpha_th_np),
        imex_inv=to_jax(inv_matrices),
        q_solve=to_jax(q_solve_matrices),
        ksq_idx=jnp.array(ksq_idx_2d, dtype=jnp.int32),
        proj_dirichlet=to_jax(proj_dir_np),
        proj_neumann=to_jax(proj_neu_np),
        beta=to_jax(np.array(cfg.beta)),
        Ra_sigma=to_jax(np.array(cfg.Ra_tilde / cfg.sigma)),
        sigma=to_jax(np.array(cfg.sigma)),
        L=to_jax(np.array(L)),
        Ld_inv_sq=to_jax(np.array(Ld_inv_sq)),
        dt=to_jax(np.array(dt)),
        gamma_imex=to_jax(np.array(gamma_imex)),
        mean_temp_eps_sq=to_jax(np.array(cfg.mean_temp_eps_sq)),
        Nx=Nx,
        Nk=Nk,
        Nz=Nz,
        Nz_gal=Nz - 1,
        Nz_dealias=N_dealias,
        Npad=Npad,
        thermal_closure=cfg.thermal_closure,
        q_boundary=cfg.q_boundary,
        nonlinear_advection=cfg.nonlinear_advection,
        vertical_cutoff_n=cfg.vertical_cutoff_n,
        imex_scheme=cfg.imex_scheme,
        vertical_dealiasing=cfg.vertical_dealiasing,
    )
"""NetCDF snapshot output and checkpoint save/load.

Uses netCDF4 directly (not xarray) to avoid pandas/numexpr compatibility issues.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import netCDF4

from nhqg.config import NHQGConfig
from nhqg.grid import Grid
from nhqg.solver import (
    State, invert_psi, mean_temperature_total, _dirichlet_to_cheb, _to_nodal, _to_nodal_1d
)


def _to_physical(field_coeffs: jnp.ndarray, V: jnp.ndarray, Nx: int) -> np.ndarray:
    """Convert coefficient-space field to physical (Nz+1, Nx, Nx).

    Steps: Chebyshev coeffs -> CGL nodal values -> irfft2 -> physical.
    """
    field_nodal = _to_nodal(field_coeffs, V)
    return np.array(jnp.fft.irfft2(field_nodal, s=(Nx, Nx)))


def save_snapshot(state: State, t: float, step: int,
                  cfg: NHQGConfig, grid: Grid, output_dir: str = None):
    """Save one snapshot as a NetCDF file via netCDF4.

    Physical-space fields: q', w, theta, psi plus mean temperature profiles.
    Dimensions: (z, y, x).
    """
    if output_dir is None:
        output_dir = cfg.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    Nx, Nz = cfg.Nx, cfg.Nz
    L = cfg.L

    # Physical-space fields (coefficient space -> nodal -> physical)
    q_phys = _to_physical(state.q_hat, grid.V, Nx)
    w_cheb = _dirichlet_to_cheb(state.w_hat, grid.dirichlet_stencil)
    th_cheb = _dirichlet_to_cheb(state.th_hat, grid.dirichlet_stencil)
    w_phys = _to_physical(w_cheb, grid.V, Nx)
    th_phys = _to_physical(th_cheb, grid.V, Nx)
    psi_hat = invert_psi(state.q_hat, grid.inv_denom)
    psi_phys = _to_physical(psi_hat, grid.V, Nx)
    th_mean = np.array(_to_nodal_1d(state.th_bar, grid.V))
    th_mean_total = np.array(mean_temperature_total(state.th_bar, grid.Z, grid.V))

    # Coordinates
    x = np.linspace(0, L, Nx, endpoint=False)
    z = np.array(grid.Z)

    fname = os.path.join(output_dir, f'snapshot_{step:08d}.nc')
    with netCDF4.Dataset(fname, 'w', format='NETCDF4') as ds:
        # Dimensions
        ds.createDimension('z', Nz + 1)
        ds.createDimension('y', Nx)
        ds.createDimension('x', Nx)

        # Coordinates
        v_x = ds.createVariable('x', 'f8', ('x',))
        v_x[:] = x
        v_x.units = 'Lc'

        v_y = ds.createVariable('y', 'f8', ('y',))
        v_y[:] = x
        v_y.units = 'Lc'

        v_z = ds.createVariable('z', 'f8', ('z',))
        v_z[:] = z
        v_z.long_name = 'depth (CGL points)'

        # Fields (float32 to save space — physical fields don't need float64)
        for name, data in [('q_prime', q_phys), ('w', w_phys),
                            ('theta', th_phys), ('psi', psi_phys)]:
            v = ds.createVariable(name, 'f4', ('z', 'y', 'x'),
                                   zlib=True, complevel=1)
            v[:] = data.astype(np.float32)

        for name, data in [('theta_bar', th_mean), ('theta_mean_total', th_mean_total)]:
            v = ds.createVariable(name, 'f4', ('z',), zlib=True, complevel=1)
            v[:] = data.astype(np.float32)

        # Global attributes
        ds.time = t
        ds.step = step
        ds.Ra_tilde = cfg.Ra_tilde
        ds.sigma = cfg.sigma
        ds.beta = cfg.beta
        ds.Ld = cfg.Ld if cfg.Ld != float('inf') else -1.0
        ds.L = cfg.L
        ds.thermal_closure = cfg.thermal_closure
        ds.Nx = cfg.Nx
        ds.Nz = cfg.Nz
        ds.dt = cfg.dt

    return fname


def save_checkpoint(state: State, step: int, cfg: NHQGConfig,
                    output_dir: str = None):
    """Save spectral state for restart."""
    if output_dir is None:
        output_dir = cfg.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fname = os.path.join(output_dir, f'checkpoint_{step:08d}.npz')
    np.savez(fname,
             q_hat_real=np.array(state.q_hat.real),
             q_hat_imag=np.array(state.q_hat.imag),
             w_hat_real=np.array(state.w_hat.real),
             w_hat_imag=np.array(state.w_hat.imag),
             th_hat_real=np.array(state.th_hat.real),
             th_hat_imag=np.array(state.th_hat.imag),
             th_bar=np.array(state.th_bar),
             step=step,
             t=step * cfg.dt)
    return fname


def load_checkpoint(path: str, dtype=jnp.complex128) -> tuple[State, int, float]:
    """Load spectral state from checkpoint.

    Returns: (state, step, t)
    """
    data = np.load(path)
    q_hat = jnp.array(data['q_hat_real'] + 1j * data['q_hat_imag'], dtype=dtype)
    w_hat = jnp.array(data['w_hat_real'] + 1j * data['w_hat_imag'], dtype=dtype)
    th_hat = jnp.array(data['th_hat_real'] + 1j * data['th_hat_imag'], dtype=dtype)
    if 'th_bar' in data:
        th_bar = jnp.array(data['th_bar'], dtype=q_hat.real.dtype)
    else:
        th_bar = jnp.zeros(q_hat.shape[0], dtype=q_hat.real.dtype)
    return State(q_hat, w_hat, th_hat, th_bar), int(data['step']), float(data['t'])
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
    triple_jacobian,
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
    w_cheb = _dirichlet_to_cheb(state.w_hat, grid.dirichlet_stencil)
    th_cheb = _dirichlet_to_cheb(state.th_hat, grid.dirichlet_stencil)
    w_cheb = project_dirichlet(w_cheb * keep, grid.proj_dirichlet)
    th_cheb = project_dirichlet(th_cheb * keep, grid.proj_dirichlet)
    w_hat = _cheb_to_dirichlet(w_cheb, grid.dirichlet_pinv)
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
                           V: jnp.ndarray, stencil: jnp.ndarray,
                           Nx: int, Npad: int | None = None) -> jnp.ndarray:
    """Compute <w theta>_xy as a function of Z at CGL nodes.

    Inputs are Chebyshev coefficients; converts to nodal, then to physical
    space for the product, averages horizontally. Returns nodal values (Nz+1,).
    """
    w_nodal = _to_nodal(_dirichlet_to_cheb(w_hat, stencil), V)
    th_nodal = _to_nodal(_dirichlet_to_cheb(th_hat, stencil), V)

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


def _triple_horizontal_advection(psi_nodal: jnp.ndarray,
                                 q_nodal: jnp.ndarray,
                                 w_nodal: jnp.ndarray,
                                 th_nodal: jnp.ndarray,
                                 grid: Grid):
    """Return the horizontally dealiased advection operator for q', w, theta."""
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
    w_cheb = _dirichlet_to_cheb(w_hat, grid.dirichlet_stencil)
    th_cheb = _dirichlet_to_cheb(th_hat, grid.dirichlet_stencil)

    psi_nodal = _to_nodal(psi_hat, grid.V_dealias)
    q_nodal = _to_nodal(q_hat, grid.V_dealias)
    w_nodal = _to_nodal(w_cheb, grid.V_dealias)
    th_nodal = _to_nodal(th_cheb, grid.V_dealias)

    Aq_n, Aw_n, Ath_n = _triple_horizontal_advection(psi_nodal, q_nodal, w_nodal, th_nodal, grid)

    Aq_hi = _to_coeffs(Aq_n, grid.V_dealias_inv)
    Aw_hi = _to_coeffs(Aw_n, grid.V_dealias_inv)
    Ath_hi = _to_coeffs(Ath_n, grid.V_dealias_inv)
    Aq = _truncate_cheb_coeffs(Aq_hi, grid.Nz)
    Aw = _truncate_cheb_coeffs(Aw_hi, grid.Nz)
    Ath = _truncate_cheb_coeffs(Ath_hi, grid.Nz)

    E_q = -Aq - 1j * grid.beta * grid.kx[None, :, :] * psi_hat
    E_w = _cheb_to_dirichlet(project_dirichlet(-Aw, grid.proj_dirichlet), grid.dirichlet_pinv)

    if grid.thermal_closure == "evolve_mean":
        dth_bar_dZ_coeffs = grid.G_Z @ th_bar
        dth_bar_dZ_nodal = _to_nodal_1d(dth_bar_dZ_coeffs, grid.V_dealias)
        product_nodal = dth_bar_dZ_nodal[:, None, None] * w_nodal
        product_hi = _to_coeffs(product_nodal, grid.V_dealias_inv)
        product_coeffs = _truncate_cheb_coeffs(product_hi, grid.Nz)
        th_rhs_cheb = project_dirichlet(-Ath - product_coeffs, grid.proj_dirichlet)
        E_th = _cheb_to_dirichlet(th_rhs_cheb, grid.dirichlet_pinv)

        flux_nodal = horizontal_mean_wtheta(
            w_hat, th_hat, grid.V_dealias, grid.dirichlet_stencil, grid.Nx, grid.Npad
        )
        flux_hi = _to_coeffs_1d(flux_nodal, grid.V_dealias_inv)
        flux_coeffs = _truncate_cheb_coeffs(flux_hi, grid.Nz)
        dflux_dZ = grid.G_Z @ flux_coeffs
        E_th_bar = -grid.mean_temp_eps_sq * dflux_dZ
    else:
        E_th = _cheb_to_dirichlet(project_dirichlet(-Ath, grid.proj_dirichlet), grid.dirichlet_pinv)
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
    w_cheb = _dirichlet_to_cheb(w_hat, grid.dirichlet_stencil)
    th_cheb = _dirichlet_to_cheb(th_hat, grid.dirichlet_stencil)

    # Convert to CGL nodal values for Jacobian evaluation
    psi_nodal = _to_nodal(psi_hat, grid.V)
    q_nodal = _to_nodal(q_hat, grid.V)
    w_nodal = _to_nodal(w_cheb, grid.V)
    th_nodal = _to_nodal(th_cheb, grid.V)

    # Fused horizontal advection (operates on nodal values at each Z level)
    Aq_n, Aw_n, Ath_n = _triple_horizontal_advection(psi_nodal, q_nodal, w_nodal, th_nodal, grid)

    # Convert advection results back to Chebyshev coefficients
    Aq = _to_coeffs(Aq_n, grid.V_inv)
    Aw = _to_coeffs(Aw_n, grid.V_inv)
    Ath = _to_coeffs(Ath_n, grid.V_inv)

    # Assemble explicit tendencies (in coefficient space)
    # Ra*theta and w source are IMPLICIT (buoyancy block-elimination)
    E_q = -Aq - 1j * grid.beta * grid.kx[None, :, :] * psi_hat
    E_w = _cheb_to_dirichlet(project_dirichlet(-Aw, grid.proj_dirichlet), grid.dirichlet_pinv)

    if grid.thermal_closure == "evolve_mean":
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
            w_hat, th_hat, grid.V, grid.dirichlet_stencil, grid.Nx, grid.Npad
        )
        flux_coeffs = _to_coeffs_1d(flux_nodal, grid.V_inv)
        dflux_dZ = grid.G_Z @ flux_coeffs
        E_th_bar = -grid.mean_temp_eps_sq * dflux_dZ
    else:
        E_th = _cheb_to_dirichlet(project_dirichlet(-Ath, grid.proj_dirichlet), grid.dirichlet_pinv)
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
    w_cheb = _dirichlet_to_cheb(w_hat, grid.dirichlet_stencil)
    th_cheb = _dirichlet_to_cheb(th_hat, grid.dirichlet_stencil)

    dw_dZ = jnp.einsum('ij,j...->i...', grid.G_Z, w_cheb)
    dq_dZ = jnp.einsum('ij,j...->i...', grid.G_Z, q_hat)
    I_w_cheb = grid.inv_denom[None, :, :] * dq_dZ + grid.Ra_sigma * th_cheb
    I_w = _cheb_to_dirichlet(project_dirichlet(I_w_cheb, grid.proj_dirichlet), grid.dirichlet_pinv)

    I_th = w_hat
    if grid.thermal_closure == "evolve_mean":
        I_th_bar = (grid.mean_temp_eps_sq / grid.sigma) * (grid.G_Z2 @ th_bar)
    else:
        I_th_bar = jnp.zeros_like(th_bar)

    return State(dw_dZ, I_w, I_th, I_th_bar)


# ---------------------------------------------------------------------------
# IMEX implicit solve (block elimination at each k)
# ---------------------------------------------------------------------------

def _per_shell_matmul(mat_shells: jnp.ndarray, ksq_idx: jnp.ndarray,
                      field: jnp.ndarray) -> jnp.ndarray:
    """Apply per-|k|² shell matrices to a spectral field."""
    mats = mat_shells[ksq_idx]               # (Nx, Nk, Nz+1, Nz+1)
    f_t = jnp.transpose(field, (1, 2, 0))    # (Nx, Nk, Nz+1)
    r_t = jnp.einsum('abij,abj->abi', mats, f_t)
    return jnp.transpose(r_t, (2, 0, 1))


def imex_implicit_solve(R_q: jnp.ndarray, R_w: jnp.ndarray,
                        R_th: jnp.ndarray,
                        grid: Grid) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve the IMEX implicit stage with q in Chebyshev and w/theta in Galerkin."""
    gamma = grid.gamma_imex
    dt = grid.dt

    # Step 0: Modified w RHS from buoyancy block-elimination
    R_w_eff = R_w + gamma * dt * grid.Ra_sigma * R_th * grid.inv_alpha_th[None, :, :]

    # Step 1: solve the q stage operator (per-shell)
    Nz = grid.Nz
    if grid.q_boundary == "neumann":
        R_q = R_q.at[Nz - 1].set(0.0).at[Nz].set(0.0)
    temp = _per_shell_matmul(grid.q_solve, grid.ksq_idx, R_q)

    # Step 2: w RHS = R_w_eff + gamma*dt*c(k)*P_gal[G_Z @ temp]
    d_temp = jnp.einsum('ij,j...->i...', grid.G_Z, temp)
    d_temp_gal = _cheb_to_dirichlet(
        project_dirichlet(grid.inv_denom[None, :, :] * d_temp, grid.proj_dirichlet),
        grid.dirichlet_pinv,
    )
    rhs_w = R_w_eff + gamma * dt * d_temp_gal

    # Step 3: Solve A' @ w = rhs_w (per-shell, Galerkin basis)
    w_new = _per_shell_matmul(grid.imex_inv, grid.ksq_idx, rhs_w)

    # Step 4: Back-substitute q = q_solve @ (R_q + gamma*dt*G_Z@w)
    w_cheb = _dirichlet_to_cheb(w_new, grid.dirichlet_stencil)
    dw_dZ = jnp.einsum('ij,j...->i...', grid.G_Z, w_cheb)
    combined = R_q + gamma * dt * dw_dZ
    if grid.q_boundary == "neumann":
        combined = combined.at[Nz - 1].set(0.0).at[Nz].set(0.0)
    q_new = _per_shell_matmul(grid.q_solve, grid.ksq_idx, combined)

    # Step 5: Back-substitute theta = (R_th + gamma*dt*w) / alpha_th
    th_new = (R_th + gamma * dt * w_new) * grid.inv_alpha_th[None, :, :]

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

def imex_step(state: State, grid: Grid) -> State:
    """Dispatch to the configured IMEX-RK stepper."""
    if grid.imex_scheme == "ars222":
        return imex_step_ars222(state, grid)
    if grid.imex_scheme == "rk443":
        return imex_step_rk443(state, grid)
    raise ValueError(f"Unsupported imex_scheme={grid.imex_scheme!r}")


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
        project_dirichlet(_dirichlet_to_cheb(state.w_hat, grid.dirichlet_stencil), grid.proj_dirichlet),
        grid.dirichlet_pinv,
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
        project_dirichlet(_dirichlet_to_cheb(w_new, grid.dirichlet_stencil), grid.proj_dirichlet),
        grid.dirichlet_pinv,
    )
    th_new = _cheb_to_dirichlet(
        project_dirichlet(_dirichlet_to_cheb(th_new, grid.dirichlet_stencil), grid.proj_dirichlet),
        grid.dirichlet_pinv,
    )
    th_bar_new = project_dirichlet_1d(th_bar_new, grid.proj_dirichlet)

    state_new = _apply_vertical_cutoff(State(q_new, w_new, th_new, th_bar_new), grid)
    return State(state_new.q_hat, state_new.w_hat, state_new.th_hat, state_new.th_bar)


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

    return State(q_hat, w_hat, th_hat, th_bar)


# ---------------------------------------------------------------------------
# Main time loop
# ---------------------------------------------------------------------------

def run(grid: Grid, state: State, n_steps: int,
        save_interval: int, use_imex: bool = True,
        callback=None) -> tuple[State, list]:
    """Run the solver for n_steps time steps."""
    stepper = imex_step if use_imex else rk4_step

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
"""Horizontal spectral operations: dealiased nonlinear advection via 3/2-rule.

Fused evaluation of three Jacobians J[ψ,q'], J[ψ,w], J[ψ,θ] sharing
ψ_x and ψ_y to reduce FFT count from 15 to 11 per vertical level.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial


# ---------------------------------------------------------------------------
# Zero-padding and truncation for 3/2-rule dealiasing
# ---------------------------------------------------------------------------

def _zero_pad(f_hat: jnp.ndarray, Nx: int, Npad: int) -> jnp.ndarray:
    """Zero-pad spectral field from (Nx, Nx//2+1) to (Npad, Npad//2+1).

    Pads high-frequency modes with zeros for dealiasing.
    Uses scatter approach compatible with vmap.
    """
    Nk = Nx // 2 + 1
    Nk_pad = Npad // 2 + 1

    out = jnp.zeros((Npad, Nk_pad)) * (0.0 + 0.0j)  # ensure complex
    out = out.astype(f_hat.dtype)
    # Positive kx: indices 0..Nx//2-1
    out = out.at[:Nx // 2, :Nk].set(f_hat[:Nx // 2, :])
    # Negative kx: indices Nx//2..Nx-1 -> Npad-Nx//2..Npad-1
    out = out.at[Npad - Nx // 2:, :Nk].set(f_hat[Nx // 2:, :])

    return out


def _truncate(f_hat_pad: jnp.ndarray, Nx: int, Npad: int) -> jnp.ndarray:
    """Truncate spectral field from (Npad, Npad//2+1) to (Nx, Nx//2+1).

    Applies normalization factor (Npad/Nx)^2 to convert from Npad-grid DFT
    coefficients to Nx-grid DFT coefficients, compensating for the (Nx/Npad)^2
    attenuation introduced by irfft2 on the padded grid.
    """
    Nk = Nx // 2 + 1

    pos = f_hat_pad[:Nx // 2, :Nk]
    neg = f_hat_pad[Npad - Nx // 2:, :Nk]

    result = jnp.concatenate([pos, neg], axis=0)
    return result * (Npad / Nx) ** 2


# ---------------------------------------------------------------------------
# Single nonlinear evaluations
# ---------------------------------------------------------------------------

def _padded_wavenumbers(ky: jnp.ndarray, Npad: int, dtype) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Construct padded-grid wavenumbers from the base-grid fundamental spacing."""
    if ky.shape[1] >= 2:
        k0 = jnp.asarray(ky[0, 1], dtype=dtype)
    else:
        k0 = jnp.asarray(0.0, dtype=dtype)
    kx_pad = k0 * jnp.fft.fftfreq(Npad, d=1.0 / Npad).astype(dtype)
    ky_pad = k0 * jnp.arange(Npad // 2 + 1, dtype=dtype)
    return kx_pad[:, None], ky_pad[None, :]

def jacobian_dealiased(A_hat: jnp.ndarray, B_hat: jnp.ndarray,
                       kx: jnp.ndarray, ky: jnp.ndarray,
                       Nx: int, Npad: int) -> jnp.ndarray:
    """Compute J[A,B] = A_x*B_y - A_y*B_x via 3/2-rule dealiased pseudospectral.

    Args:
        A_hat, B_hat: (Nx, Nk) complex spectral fields at one Z level
        kx: (Nx, 1) wavenumber array
        ky: (1, Nk) wavenumber array
        Nx, Npad: grid and padded sizes

    Returns:
        J_hat: (Nx, Nk) complex spectral Jacobian
    """
    # Spectral derivatives
    Ax_hat = 1j * kx * A_hat
    Ay_hat = 1j * ky * A_hat
    Bx_hat = 1j * kx * B_hat
    By_hat = 1j * ky * B_hat

    # Pad, transform to physical
    Ax = jnp.fft.irfft2(_zero_pad(Ax_hat, Nx, Npad), s=(Npad, Npad))
    Ay = jnp.fft.irfft2(_zero_pad(Ay_hat, Nx, Npad), s=(Npad, Npad))
    Bx = jnp.fft.irfft2(_zero_pad(Bx_hat, Nx, Npad), s=(Npad, Npad))
    By = jnp.fft.irfft2(_zero_pad(By_hat, Nx, Npad), s=(Npad, Npad))

    # Physical multiply
    J_phys = Ax * By - Ay * Bx

    # Transform back and truncate
    J_hat_pad = jnp.fft.rfft2(J_phys)
    return _truncate(J_hat_pad, Nx, Npad)


def conservative_flux_divergence_dealiased(psi_hat: jnp.ndarray, f_hat: jnp.ndarray,
                                           kx: jnp.ndarray, ky: jnp.ndarray,
                                           Nx: int, Npad: int) -> jnp.ndarray:
    """Compute div(u f, v f) with u=-psi_y, v=psi_x via 3/2-rule dealiasing."""
    u_hat = -1j * ky * psi_hat
    v_hat = 1j * kx * psi_hat

    u = jnp.fft.irfft2(_zero_pad(u_hat, Nx, Npad), s=(Npad, Npad))
    v = jnp.fft.irfft2(_zero_pad(v_hat, Nx, Npad), s=(Npad, Npad))
    f = jnp.fft.irfft2(_zero_pad(f_hat, Nx, Npad), s=(Npad, Npad))

    uf_hat_pad = jnp.fft.rfft2(u * f)
    vf_hat_pad = jnp.fft.rfft2(v * f)
    kx_pad, ky_pad = _padded_wavenumbers(ky, Npad, psi_hat.real.dtype)
    div_hat_pad = 1j * kx_pad * uf_hat_pad + 1j * ky_pad * vf_hat_pad
    return _truncate(div_hat_pad, Nx, Npad)


# ---------------------------------------------------------------------------
# Fused triple nonlinear operators
# ---------------------------------------------------------------------------

def _triple_jacobian_one_level(psi_hat: jnp.ndarray,
                                q_hat: jnp.ndarray,
                                w_hat: jnp.ndarray,
                                th_hat: jnp.ndarray,
                                kx: jnp.ndarray,
                                ky: jnp.ndarray,
                                Nx: int, Npad: int):
    """Compute J[ψ,q'], J[ψ,w], J[ψ,θ] at one Z level, sharing ψ_x/ψ_y.

    Total: 2 + 3×3 = 11 FFTs instead of 3×5 = 15.

    Args:
        psi_hat, q_hat, w_hat, th_hat: (Nx, Nk) spectral fields

    Returns:
        Jq_hat, Jw_hat, Jth_hat: (Nx, Nk) spectral Jacobians
    """
    # ψ derivatives (shared): 2 FFTs
    psi_x_hat = 1j * kx * psi_hat
    psi_y_hat = 1j * ky * psi_hat
    psi_x = jnp.fft.irfft2(_zero_pad(psi_x_hat, Nx, Npad), s=(Npad, Npad))
    psi_y = jnp.fft.irfft2(_zero_pad(psi_y_hat, Nx, Npad), s=(Npad, Npad))

    def _one_jac(f_hat):
        """J[ψ,f] using precomputed ψ_x, ψ_y. 3 FFTs."""
        fx_hat = 1j * kx * f_hat
        fy_hat = 1j * ky * f_hat
        fx = jnp.fft.irfft2(_zero_pad(fx_hat, Nx, Npad), s=(Npad, Npad))
        fy = jnp.fft.irfft2(_zero_pad(fy_hat, Nx, Npad), s=(Npad, Npad))
        J_phys = psi_x * fy - psi_y * fx
        J_hat_pad = jnp.fft.rfft2(J_phys)
        return _truncate(J_hat_pad, Nx, Npad)

    Jq_hat = _one_jac(q_hat)
    Jw_hat = _one_jac(w_hat)
    Jth_hat = _one_jac(th_hat)

    return Jq_hat, Jw_hat, Jth_hat


def _triple_flux_divergence_one_level(psi_hat: jnp.ndarray,
                                      q_hat: jnp.ndarray,
                                      w_hat: jnp.ndarray,
                                      th_hat: jnp.ndarray,
                                      kx: jnp.ndarray,
                                      ky: jnp.ndarray,
                                      Nx: int, Npad: int):
    """Compute div(u f, v f) for q', w, theta at one Z level, sharing u and v."""
    u_hat = -1j * ky * psi_hat
    v_hat = 1j * kx * psi_hat
    u = jnp.fft.irfft2(_zero_pad(u_hat, Nx, Npad), s=(Npad, Npad))
    v = jnp.fft.irfft2(_zero_pad(v_hat, Nx, Npad), s=(Npad, Npad))
    kx_pad, ky_pad = _padded_wavenumbers(ky, Npad, psi_hat.real.dtype)

    def _one_div(f_hat):
        f = jnp.fft.irfft2(_zero_pad(f_hat, Nx, Npad), s=(Npad, Npad))
        uf_hat_pad = jnp.fft.rfft2(u * f)
        vf_hat_pad = jnp.fft.rfft2(v * f)
        div_hat_pad = 1j * kx_pad * uf_hat_pad + 1j * ky_pad * vf_hat_pad
        return _truncate(div_hat_pad, Nx, Npad)

    Aq_hat = _one_div(q_hat)
    Aw_hat = _one_div(w_hat)
    Ath_hat = _one_div(th_hat)

    return Aq_hat, Aw_hat, Ath_hat


# ---------------------------------------------------------------------------
# Batched over Z levels via vmap
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(6, 7))
def triple_jacobian(psi_hat: jnp.ndarray,
                    q_hat: jnp.ndarray,
                    w_hat: jnp.ndarray,
                    th_hat: jnp.ndarray,
                    kx: jnp.ndarray,
                    ky: jnp.ndarray,
                    Nx: int, Npad: int):
    """Batched fused triple-Jacobian over all Z levels.

    Args:
        psi_hat, q_hat, w_hat, th_hat: (Nz+1, Nx, Nk)
        kx: (Nx, 1), ky: (1, Nk)

    Returns:
        Jq, Jw, Jth: (Nz+1, Nx, Nk) spectral Jacobians
    """
    # vmap over axis 0 (Z levels)
    fn = partial(_triple_jacobian_one_level,
                 kx=kx, ky=ky, Nx=Nx, Npad=Npad)
    return jax.vmap(fn)(psi_hat, q_hat, w_hat, th_hat)


@partial(jax.jit, static_argnums=(6, 7))
def triple_conservative_flux_divergence(psi_hat: jnp.ndarray,
                                        q_hat: jnp.ndarray,
                                        w_hat: jnp.ndarray,
                                        th_hat: jnp.ndarray,
                                        kx: jnp.ndarray,
                                        ky: jnp.ndarray,
                                        Nx: int, Npad: int):
    """Batched dealiased conservative-form advection over all Z levels."""
    fn = partial(_triple_flux_divergence_one_level,
                 kx=kx, ky=ky, Nx=Nx, Npad=Npad)
    return jax.vmap(fn)(psi_hat, q_hat, w_hat, th_hat)

# Primary run script
#!/usr/bin/env python
"""NHQGE with evolving mean temperature and moderate hyperviscosity.

Usage:
    JAX_ENABLE_X64=1 PYTHONPATH=. python scripts/run_evolve.py [Nx] [Nz] [Ra]

Defaults: Nx=256, Nz=128, Ra=100.
Target: Miquel Table 1 (ϑ_f=0°): Nu = 43.37 ± 2.54, Re_ℓ = 32.05 ± 8.24
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.solver import make_initial_state, run
from nhqg.diagnostics import compute_diagnostics
from nhqg.io import save_snapshot, save_checkpoint


def main():
    Nx = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    Nz = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    Ra = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    L = 20.0

    # --- Time step ---
    # CFL: dt < dx/v_max.  At Ra=100, max_v ~ 200-400.
    # dx = L/Nx = 0.078 at Nx=256.  dt=1e-4 → CFL ≈ 0.13-0.51.
    dt = 1e-4

    # --- Dissipation: aggressive hyper-4 (n_efold=5) ---
    p = 4
    k_max = float(np.pi * Nx / L)
    n_efold = 5.0
    nu_all = n_efold / (dt * k_max ** (2 * p))
    k_d = nu_all ** (-1.0 / (2 * p))

    cfg = NHQGConfig(
        Nx=Nx, Nz=Nz, L=L,
        Ra_tilde=Ra, sigma=1.0,
        beta=0.0, Ld=float('inf'),
        dt=dt, t_final=10.0,
        nu_q=nu_all, hyper_order=p,
        nu_w=nu_all, nu_theta=nu_all,
        thermal_closure="evolve_mean",
        mean_temp_eps_sq=1.0,
        save_interval=1000,
        output_dir=f'output_Ra{int(Ra)}_Nx{Nx}_Nz{Nz}_evolve',
        float_dtype='float64',
    )

    # --- Print dissipation diagnostics ---
    rate_kmax = nu_all * k_max ** (2 * p)
    rate_kc = nu_all * 1.3048 ** (2 * p)
    rate_10 = nu_all * 10.0 ** (2 * p)
    print(f"NHQGE: Ra={Ra}, Nx={Nx}, Nz={Nz}, L={L}")
    print(
        f"  dt={dt}, t_final={cfg.t_final}, closure={cfg.thermal_closure}, "
        f"eps^2={cfg.mean_temp_eps_sq}"
    )
    print(f"  hyper_order p={p},  nu = {nu_all:.4e}")
    print(f"  k_d = {k_d:.1f}  (equiv dissipation wavenumber)")
    print(f"  k_max = {k_max:.1f},  k_c ≈ 1.3,  dx = {L/Nx:.4f}")
    print(f"  rate at k_max={k_max:.0f}: {rate_kmax:.0f}  "
          f"(per-step factor: {float(np.exp(-rate_kmax*dt)):.4f})")
    print(f"  rate at k=10:       {rate_10:.2e}  "
          f"(e-fold time: {1.0/rate_10:.1f})")
    print(f"  rate at k_c=1.3:    {rate_kc:.2e}  (negligible)")
    print(f"  CFL limit: v_max < dx/dt = {L/Nx/dt:.0f}")
    print()
    print(f"  Miquel targets (Ra=100): Nu = 43.37 ± 2.54,  Re_ℓ = 32.05 ± 8.24")
    print(f"  NOTE: Miquel uses Laplacian ν=1 (our hyper-4 is less dissipative at k_c)")
    print(f"  NOTE: Miquel uses L ≈ 48 (10 L_c), Nz=384;  we use L={L}, Nz={Nz}")
    print()
    print(f"Device: {jax.devices()}")
    print()

    # --- Grid + IC ---
    t0 = time.time()
    grid = make_grid(cfg)
    state = make_initial_state(grid, seed=0, amplitude=1e-3)
    print(f"Grid + IC setup: {time.time()-t0:.1f}s")

    # IMEX memory estimate
    n_shells = grid.imex_inv.shape[0]
    imex_mb = 2 * n_shells * (Nz+1)**2 * 8 / 1e6
    print(f"IMEX shells: {n_shells},  IMEX memory: {imex_mb:.0f} MB")
    print()

    # --- Run ---
    total_steps = int(cfg.t_final / cfg.dt)

    def callback(state, step, t):
        diag = compute_diagnostics(state, grid)
        max_v = float(diag['max_speed'])
        cfl = max_v * dt / (L / Nx)

        # Mean temperature profile extremes
        th_bar_max = float(jnp.max(jnp.abs(state.th_bar)))

        print(f"  step={step:8d}  t={t:8.3f}  "
              f"Nu={float(diag['Nusselt']):8.3f}  "
              f"KE_bt={float(diag['KE_bt']):10.4e}  "
              f"KE_bc={float(diag['KE_bc']):10.4e}  "
              f"max_v={max_v:8.2f}  CFL={cfl:.3f}  "
              f"|Θ̄'|={th_bar_max:.4f}")

        if cfl > 0.5:
            print(f"  *** CFL WARNING: {cfl:.2f} — reduce dt! ***")

        # Check for blowup
        if not jnp.isfinite(float(diag['KE_tot'])):
            print("  *** BLOWUP DETECTED — aborting ***")
            save_checkpoint(state, step, cfg)
            sys.exit(1)

        # Save snapshot
        save_snapshot(state, t, step, cfg, grid)

        # Checkpoint every 10 saves
        if step % (cfg.save_interval * 10) == 0:
            save_checkpoint(state, step, cfg)

    t0 = time.time()
    final_state, snapshots = run(grid, state, n_steps=total_steps,
                                  save_interval=cfg.save_interval,
                                  use_imex=True, callback=callback)
    elapsed = time.time() - t0
    print(f"\nCompleted {total_steps} steps in {elapsed:.1f}s "
          f"({total_steps/elapsed:.0f} steps/s)")

    # Final checkpoint
    save_checkpoint(final_state, total_steps, cfg)


if __name__ == '__main__':
    main()
