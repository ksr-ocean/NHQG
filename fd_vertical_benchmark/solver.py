"""Finite-difference vertical NHQG benchmark solver."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from fd_vertical_benchmark.operators import Grid
from nhqg.spectral import (
    _zero_pad,
    triple_conservative_flux_divergence,
    triple_jacobian,
)


class State(NamedTuple):
    """Interior-node state for the FD benchmark."""

    psi_hat: jnp.ndarray
    w_hat: jnp.ndarray
    th_hat: jnp.ndarray
    th_bar: jnp.ndarray


def zero_mode(field: jnp.ndarray) -> jnp.ndarray:
    """Zero the horizontal mean mode."""
    return field.at[:, 0, 0].set(0.0)


def extend_dirichlet(field: jnp.ndarray, Nz: int) -> jnp.ndarray:
    """Embed an interior Dirichlet field into the full vertical grid."""
    out = jnp.zeros((Nz + 1,) + field.shape[1:], dtype=field.dtype)
    return out.at[1:-1].set(field)


def extend_dirichlet_1d(field: jnp.ndarray, Nz: int) -> jnp.ndarray:
    """1D interior Dirichlet field to full nodes."""
    out = jnp.zeros((Nz + 1,), dtype=field.dtype)
    return out.at[1:-1].set(field)


def reconstruct_psi_full(psi_hat: jnp.ndarray, P_neu: jnp.ndarray) -> jnp.ndarray:
    """Recover the full-grid psi satisfying the Neumann BCs."""
    return jnp.einsum("ij,j...->i...", P_neu, psi_hat)


def psi_full_from_state(psi_hat: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Full-grid psi. For psi_boundary='none' psi IS the full-grid prognostic (no
    reconstruction, no vorticity BC). For 'neumann' it is reconstructed to satisfy
    dpsi/dz=0 from the interior values."""
    if grid.psi_boundary == "none":
        return zero_mode(psi_hat)
    return reconstruct_psi_full(zero_mode(psi_hat), grid.P_neu)


def q_from_psi(psi_full: jnp.ndarray, denom: jnp.ndarray) -> jnp.ndarray:
    """Recover q from psi in spectral space."""
    return -denom[None, :, :] * psi_full


def _horizontal_mean_product_dealiased(
    a_hat: jnp.ndarray, b_hat: jnp.ndarray, Nx: int, Npad: int
) -> jnp.ndarray:
    """Compute <a b>_xy on each vertical level with 3/2 padding."""
    pad_one = lambda field: _zero_pad(field, Nx, Npad)
    a_pad = jax.vmap(pad_one)(a_hat)
    b_pad = jax.vmap(pad_one)(b_hat)
    scale = (Npad / Nx) ** 2
    a_phys = scale * jnp.fft.irfft2(a_pad, s=(Npad, Npad))
    b_phys = scale * jnp.fft.irfft2(b_pad, s=(Npad, Npad))
    return jnp.mean(a_phys * b_phys, axis=(1, 2))


def _triple_horizontal_advection(
    psi_full: jnp.ndarray,
    q_full: jnp.ndarray,
    w_full: jnp.ndarray,
    th_full: jnp.ndarray,
    grid: Grid,
):
    """Return the configured horizontally dealiased advection operator."""
    if grid.nonlinear_advection == "jacobian":
        return triple_jacobian(
            psi_full, q_full, w_full, th_full, grid.kx, grid.ky, grid.Nx, grid.Npad
        )
    if grid.nonlinear_advection == "flux":
        return triple_conservative_flux_divergence(
            psi_full, q_full, w_full, th_full, grid.kx, grid.ky, grid.Nx, grid.Npad
        )
    raise ValueError(f"Unsupported nonlinear_advection={grid.nonlinear_advection!r}")


def explicit_rhs(state: State, grid: Grid) -> State:
    """Explicit nonlinear terms for the FD benchmark."""
    psi_hat = zero_mode(state.psi_hat)
    w_hat = zero_mode(state.w_hat)
    th_hat = zero_mode(state.th_hat)
    th_bar = state.th_bar

    psi_full = psi_full_from_state(psi_hat, grid)
    q_full = q_from_psi(psi_full, grid.denom)
    w_full = extend_dirichlet(w_hat, grid.Nz)
    th_full = extend_dirichlet(th_hat, grid.Nz)

    Aq_full, Aw_full, Ath_full = _triple_horizontal_advection(
        psi_full, q_full, w_full, th_full, grid
    )

    # psi tendency: full grid for 'none' (psi has boundary DOF), interior for 'neumann'.
    Aq = Aq_full if grid.psi_boundary == "none" else Aq_full[1:-1]
    E_psi = grid.inv_denom[None, :, :] * Aq
    E_w = -Aw_full[1:-1]

    if grid.thermal_closure == "evolve_mean":
        dth_bar = grid.D1_dir @ th_bar
        E_th = -Ath_full[1:-1] - dth_bar[:, None, None] * w_hat
        flux_full = _horizontal_mean_product_dealiased(w_full, th_full, grid.Nx, grid.Npad)
        E_th_bar = -grid.mean_temp_eps_sq * (grid.D1_dir @ flux_full[1:-1])
    else:
        E_th = -Ath_full[1:-1]
        E_th_bar = jnp.zeros_like(th_bar)

    return State(zero_mode(E_psi), zero_mode(E_w), zero_mode(E_th), E_th_bar)


def implicit_tendency(state: State, grid: Grid) -> State:
    """Implicit linear vertical coupling and buoyancy terms."""
    psi_hat = zero_mode(state.psi_hat)
    w_hat = zero_mode(state.w_hat)
    th_hat = zero_mode(state.th_hat)

    # q-equation stretching term D_Z w. For 'none', psi is full-grid so this is the
    # full-grid derivative of the (Dirichlet-extended) w; for 'neumann' it's interior.
    if grid.psi_boundary == "none":
        dw = jnp.einsum("ij,j...->i...", grid.D1_full, extend_dirichlet(w_hat, grid.Nz))
        psi_imp = -grid.inv_denom[None, :, :] * dw
    else:
        psi_imp = -grid.inv_denom[None, :, :] * jnp.einsum("ij,j...->i...", grid.D1_dir, w_hat)
    w_imp = -jnp.einsum("ij,j...->i...", grid.D1_psi, psi_hat) + grid.Ra_sigma * th_hat
    th_imp = w_hat

    if grid.thermal_closure == "evolve_mean":
        th_bar_imp = (grid.mean_temp_eps_sq / grid.sigma) * (grid.D2_dir @ state.th_bar)
    else:
        th_bar_imp = jnp.zeros_like(state.th_bar)

    return State(zero_mode(psi_imp), zero_mode(w_imp), zero_mode(th_imp), th_bar_imp)


def _per_shell_matmul(mat_shells: jnp.ndarray, ksq_idx: jnp.ndarray, field: jnp.ndarray) -> jnp.ndarray:
    """Apply shellwise dense matrices to a vertical spectral field."""
    mats = mat_shells[ksq_idx]
    f_t = jnp.transpose(field, (1, 2, 0))
    r_t = jnp.einsum("abij,abj->abi", mats, f_t)
    return jnp.transpose(r_t, (2, 0, 1))


def imex_implicit_solve(
    R_psi: jnp.ndarray, R_w: jnp.ndarray, R_th: jnp.ndarray, grid: Grid
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve one ARS222 implicit stage for psi, w, theta."""
    gamma = grid.gamma_imex
    dt = grid.dt

    R_psi = zero_mode(R_psi)
    R_w = zero_mode(R_w)
    R_th = zero_mode(R_th)

    R_w_eff = R_w + gamma * dt * grid.Ra_sigma * R_th * grid.inv_alpha_th[None, :, :]
    dR_psi = jnp.einsum("ij,j...->i...", grid.D1_psi, R_psi)  # -> interior dpsi/dz (both paths)
    rhs_w = R_w_eff - gamma * dt * grid.inv_alpha_q[None, :, :] * dR_psi
    w_new = _per_shell_matmul(grid.w_solve, grid.ksq_idx, rhs_w)

    # Back-substitute psi by division (no vorticity BC for 'none'; interior for 'neumann').
    if grid.psi_boundary == "none":
        dw_dz = jnp.einsum("ij,j...->i...", grid.D1_full, extend_dirichlet(w_new, grid.Nz))
    else:
        dw_dz = jnp.einsum("ij,j...->i...", grid.D1_dir, w_new)
    psi_new = (R_psi - gamma * dt * grid.inv_denom[None, :, :] * dw_dz) * grid.inv_alpha_q[None, :, :]
    th_new = (R_th + gamma * dt * w_new) * grid.inv_alpha_th[None, :, :]

    return zero_mode(psi_new), zero_mode(w_new), zero_mode(th_new)


def imex_mean_temp_solve(R_th_bar: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Solve the interior Dirichlet mean-temperature stage.

    For ``fixed_conduction`` the mean is not diffused implicitly here; the input
    is passed through unchanged (interior nodes are Dirichlet by construction).
    This passthrough is what lets the balanced_sbp predictor carry ``th_bar``
    untouched into the thermal corrector (which then does all mean evolution).
    """
    if grid.thermal_closure != "evolve_mean":
        return R_th_bar
    return grid.thbar_solve @ R_th_bar


def imex_step_ars222(state: State, grid: Grid) -> State:
    """One full ARS(2,2,2) step."""
    gamma = grid.gamma_imex
    delta = -jnp.sqrt(jnp.array(2.0, dtype=grid.dt.dtype)) / 2.0
    dt = grid.dt

    psi_n, w_n, th_n, th_bar_n = state

    E1 = explicit_rhs(state, grid)
    R_psi1 = psi_n + gamma * dt * E1.psi_hat
    R_w1 = w_n + gamma * dt * E1.w_hat
    R_th1 = th_n + gamma * dt * E1.th_hat
    R_th_bar1 = th_bar_n + gamma * dt * E1.th_bar

    psi1, w1, th1 = imex_implicit_solve(R_psi1, R_w1, R_th1, grid)
    th_bar1 = imex_mean_temp_solve(R_th_bar1, grid)
    state1 = State(psi1, w1, th1, th_bar1)

    E2 = explicit_rhs(state1, grid)
    I1 = implicit_tendency(state1, grid)
    omg = dt * (1.0 - gamma)

    R_psi2 = (
        psi_n
        + dt * (delta * E1.psi_hat + (1.0 - delta) * E2.psi_hat)
        + omg * I1.psi_hat
        - omg * grid.diss_rate_q[None, :, :] * psi1
    )
    R_w2 = (
        w_n
        + dt * (delta * E1.w_hat + (1.0 - delta) * E2.w_hat)
        + omg * I1.w_hat
        - omg * grid.diss_rate_w[None, :, :] * w1
    )
    R_th2 = (
        th_n
        + dt * (delta * E1.th_hat + (1.0 - delta) * E2.th_hat)
        + omg * I1.th_hat
        - omg * grid.diss_rate_th[None, :, :] * th1
    )
    R_th_bar2 = th_bar_n + dt * (delta * E1.th_bar + (1.0 - delta) * E2.th_bar) + omg * I1.th_bar

    psi2, w2, th2 = imex_implicit_solve(R_psi2, R_w2, R_th2, grid)
    th_bar2 = imex_mean_temp_solve(R_th_bar2, grid)
    return State(zero_mode(psi2), zero_mode(w2), zero_mode(th2), th_bar2)


# ---------------------------------------------------------------------------
# Balanced SBP mean-fluctuation thermal exchange (native uniform grid)
# ---------------------------------------------------------------------------
#
# This is the energy-balanced exchange of `adjoint_mean_exchange.md` /
# `balanced_sbp2_pc`, ported to the FD benchmark. Because the FD solver already
# lives on the uniform SBP grid, there is NO CGL<->SBP transfer layer: the
# frozen-w midpoint/CN substep acts directly on the state's interior fields.
# The full-grid SBP first derivative (D1_full) and diagonal norm (norm_weights)
# satisfy the summation-by-parts identity, so the exchange pair telescopes in
# the H norm (measured by `balanced_exchange_residual_fd`).


def balanced_sbp_thermal_substep_fd(state: State, grid: Grid, sub_dt) -> State:
    """One frozen-w midpoint/CN balanced thermal substep on the uniform grid."""
    if grid.thermal_closure != "evolve_mean":
        return state

    psi_hat, w_hat, th_hat, th_bar = state
    dt = jnp.asarray(sub_dt, dtype=grid.dt.dtype)
    mu = grid.mean_temp_eps_sq
    kappa = 1.0 / grid.sigma
    Nz = grid.Nz

    w_full = extend_dirichlet(w_hat, Nz)
    th_full = extend_dirichlet(th_hat, Nz)
    th_bar_full = extend_dirichlet_1d(th_bar, Nz)

    # Exchange means are computed on the Nx grid (no 3/2 padding): the horizontal
    # k=0 mode is alias-free regardless, and using Nx-grid Parseval makes the
    # M-term's <w^2> match the discrete energy norm exactly, so the exchange pair
    # conserves combined thermal energy to machine precision (R_ex_sbp ~ 1e-12).
    # Padding to Npad here would leave an O(dt) energy leak. Mirrors the
    # production npad_for_mean = Nx choice.
    flux = _horizontal_mean_product_dealiased(w_full, th_full, grid.Nx, grid.Nx)
    w2 = _horizontal_mean_product_dealiased(w_full, w_full, grid.Nx, grid.Nx)

    I = jnp.eye(Nz + 1, dtype=th_bar_full.dtype)
    M = jnp.diag(w2)
    D1 = grid.D1_full
    L = D1 @ D1  # full second derivative consistent with the SBP D1
    half = 0.5 * mu * kappa * dt * L + 0.25 * mu * (dt ** 2) * (D1 @ M @ D1)
    A = I - half
    B = I + half
    rhs = B @ th_bar_full - mu * dt * (D1 @ flux)

    # Strong Dirichlet endpoints for the mean solve.
    A = A.at[0, :].set(0.0).at[-1, :].set(0.0).at[0, 0].set(1.0).at[-1, -1].set(1.0)
    rhs = rhs.at[0].set(0.0).at[-1].set(0.0)
    th_bar_new_full = jnp.linalg.solve(A, rhs)

    g_half = 0.5 * (D1 @ (th_bar_full + th_bar_new_full))
    th_new_full = th_full - dt * w_full * g_half[:, None, None]
    th_new_full = th_new_full.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
    th_bar_new_full = th_bar_new_full.at[0].set(0.0).at[-1].set(0.0)

    return State(psi_hat, w_hat, th_new_full[1:-1], th_bar_new_full[1:-1])


def _apply_balanced_corrector_fd(state: State, grid: Grid, total_dt) -> State:
    """Apply the SBP thermal corrector with optional subcycling."""
    if grid.thermal_closure != "evolve_mean":
        return state
    n_sub = max(1, int(grid.sbp_corrector_substeps))
    sub_dt = jnp.asarray(total_dt, dtype=grid.dt.dtype) / n_sub
    out = state
    for _ in range(n_sub):
        out = balanced_sbp_thermal_substep_fd(out, grid, sub_dt)
    return out


def _thermal_correction_tendency_fd(predictor: State, corrected: State, alpha):
    """Effective (theta, th_bar) correction tendency from the corrector."""
    inv = 1.0 / jnp.asarray(alpha, dtype=corrected.th_bar.dtype)
    return (corrected.th_hat - predictor.th_hat) * inv, (corrected.th_bar - predictor.th_bar) * inv


def imex_step_balanced_pc_fd(state: State, grid: Grid) -> State:
    """ARS(2,2,2) predictor/corrector with the native SBP balanced exchange."""
    base_grid = grid._replace(thermal_closure="fixed_conduction")

    gamma = grid.gamma_imex
    delta = -jnp.sqrt(jnp.array(2.0, dtype=grid.dt.dtype)) / 2.0
    dt = grid.dt
    alpha = gamma * dt
    omg = dt * (1.0 - gamma)

    psi_n, w_n, th_n, th_bar_n = state

    # ── Stage 1 predictor (reduced system, no mean exchange) ──
    E1 = explicit_rhs(state, base_grid)
    R_psi1 = psi_n + alpha * E1.psi_hat
    R_w1 = w_n + alpha * E1.w_hat
    R_th1 = th_n + alpha * E1.th_hat
    R_th_bar1 = th_bar_n + alpha * E1.th_bar

    psi1p, w1p, th1p = imex_implicit_solve(R_psi1, R_w1, R_th1, base_grid)
    th_bar1p = imex_mean_temp_solve(R_th_bar1, base_grid)
    predictor1 = State(psi1p, w1p, th1p, th_bar1p)

    # ── Stage 1 thermal corrector ──
    state1 = _apply_balanced_corrector_fd(predictor1, grid, alpha)
    C1_th, C1_th_bar = _thermal_correction_tendency_fd(predictor1, state1, alpha)

    # ── Stage 2 predictor (reduced system) ──
    E2 = explicit_rhs(state1, base_grid)
    I1 = implicit_tendency(state1, base_grid)

    R_psi2 = (
        psi_n
        + dt * (delta * E1.psi_hat + (1.0 - delta) * E2.psi_hat)
        + omg * I1.psi_hat
        - omg * grid.diss_rate_q[None, :, :] * state1.psi_hat
    )
    R_w2 = (
        w_n
        + dt * (delta * E1.w_hat + (1.0 - delta) * E2.w_hat)
        + omg * I1.w_hat
        - omg * grid.diss_rate_w[None, :, :] * state1.w_hat
    )
    R_th2 = (
        th_n
        + dt * (delta * E1.th_hat + (1.0 - delta) * E2.th_hat)
        + omg * I1.th_hat
        - omg * grid.diss_rate_th[None, :, :] * state1.th_hat
        + omg * C1_th
    )
    R_th_bar2 = (
        th_bar_n
        + dt * (delta * E1.th_bar + (1.0 - delta) * E2.th_bar)
        + omg * I1.th_bar
        + omg * C1_th_bar
    )

    psi2p, w2p, th2p = imex_implicit_solve(R_psi2, R_w2, R_th2, base_grid)
    th_bar2p = imex_mean_temp_solve(R_th_bar2, base_grid)
    predictor2 = State(psi2p, w2p, th2p, th_bar2p)

    # ── Stage 2 thermal corrector ──
    corrected = _apply_balanced_corrector_fd(predictor2, grid, alpha)
    return State(
        zero_mode(corrected.psi_hat), zero_mode(corrected.w_hat),
        zero_mode(corrected.th_hat), corrected.th_bar,
    )


def imex_step(state: State, grid: Grid) -> State:
    """Dispatch one full step based on the configured mean-exchange scheme."""
    if grid.mean_exchange == "balanced_sbp":
        return imex_step_balanced_pc_fd(state, grid)
    return imex_step_ars222(state, grid)


def balanced_exchange_residual_fd(state: State, grid: Grid, dt=None) -> jnp.ndarray:
    """Discrete exchange+diffusion residual R_bal in the SBP norm (~0 if closed).

    R_bal = d/dt(||th||^2_H) / 2 + d/dt(||Th||^2_H) / (2 mu) - kappa Th_half^T H L Th_half,
    evaluated over one balanced substep. Near machine zero ⇒ the mean/fluctuation
    exchange pair conserves combined thermal energy on the SBP grid.
    """
    if grid.thermal_closure != "evolve_mean":
        return jnp.array(0.0, dtype=grid.dt.dtype)

    step_dt = grid.dt if dt is None else jnp.asarray(dt, dtype=grid.dt.dtype)
    mu = grid.mean_temp_eps_sq
    kappa = 1.0 / grid.sigma
    Nz = grid.Nz
    H = grid.norm_weights
    D1 = grid.D1_full
    L = D1 @ D1

    weight = jnp.ones_like(grid.ksq)
    if grid.Nk > 2:
        weight = weight.at[:, 1:grid.Nk - 1].set(2.0)
    norm = grid.Nx ** 4

    def field_energy(th_full):
        horiz = jnp.sum(jnp.abs(th_full) ** 2 * weight[None, :, :], axis=(1, 2)) / norm
        return jnp.sum(H * horiz)

    def mean_energy(th_bar_full):
        return jnp.sum(H * th_bar_full ** 2)

    th_n = extend_dirichlet(state.th_hat, Nz)
    thb_n = extend_dirichlet_1d(state.th_bar, Nz)
    new = balanced_sbp_thermal_substep_fd(state, grid, step_dt)
    th_1 = extend_dirichlet(new.th_hat, Nz)
    thb_1 = extend_dirichlet_1d(new.th_bar, Nz)

    dE_th = (field_energy(th_1) - field_energy(th_n)) / (2.0 * step_dt)
    dE_bar = (mean_energy(thb_1) - mean_energy(thb_n)) / (2.0 * mu * step_dt)
    thb_half = 0.5 * (thb_n + thb_1)
    diff_term = kappa * jnp.sum(H * thb_half * (L @ thb_half))
    return dE_th + dE_bar - diff_term


def make_initial_state(grid: Grid, seed: int = 0, amplitude: float = 1e-6) -> State:
    """Small random psi perturbation with a cos(pi z) envelope.

    For psi_boundary='none' psi is a full-grid (Nz+1) variable with no vorticity BC;
    for 'neumann' it lives on the Nz-1 interior nodes. w and theta are always interior
    (Dirichlet).
    """
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    if grid.psi_boundary == "none":
        n_psi, z_psi = grid.Nz + 1, grid.z_full
    else:
        n_psi, z_psi = grid.Ni, grid.z_int
    psi_real = jax.random.normal(k1, (n_psi, grid.Nx, grid.Nk))
    psi_imag = jax.random.normal(k2, (n_psi, grid.Nx, grid.Nk))
    envelope = jnp.cos(jnp.pi * z_psi)
    psi_hat = amplitude * (psi_real + 1j * psi_imag) * envelope[:, None, None]
    psi_hat = zero_mode(psi_hat)

    w_hat = jnp.zeros((grid.Ni, grid.Nx, grid.Nk), dtype=psi_hat.dtype)
    th_hat = jnp.zeros((grid.Ni, grid.Nx, grid.Nk), dtype=psi_hat.dtype)
    th_bar = jnp.zeros((grid.Ni,), dtype=grid.z_int.dtype)
    return State(psi_hat, w_hat, th_hat, th_bar)


def run(grid: Grid, state: State, n_steps: int, save_interval: int, callback=None) -> State:
    """Run the FD benchmark for ``n_steps`` with callback cadence ``save_interval``."""
    stepper = partial(imex_step, grid=grid)

    @jax.jit
    def scan_body(carry, _):
        return stepper(carry), None

    n_outer = n_steps // save_interval
    for i_outer in range(n_outer):
        state, _ = jax.lax.scan(scan_body, state, None, length=save_interval)
        step = (i_outer + 1) * save_interval
        t = step * float(grid.dt)
        if callback is not None:
            callback(state, step, t)

    remainder = n_steps % save_interval
    if remainder > 0:
        state, _ = jax.lax.scan(scan_body, state, None, length=remainder)
        if callback is not None:
            callback(state, n_steps, n_steps * float(grid.dt))

    return state
