"""Trigonometric vertical NHQG benchmark solver."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nhqg.spectral import (
    _zero_pad,
    triple_conservative_flux_divergence,
    triple_jacobian,
)
from trig_vertical_benchmark.operators import Grid


class State(NamedTuple):
    """Modal state for the trigonometric benchmark."""

    psi_hat: jnp.ndarray
    w_hat: jnp.ndarray
    th_hat: jnp.ndarray
    th_bar: jnp.ndarray


def zero_mode(field: jnp.ndarray) -> jnp.ndarray:
    """Zero the horizontal mean mode."""
    return field.at[:, 0, 0].set(0.0)


def eval_cos(coeff: jnp.ndarray, C_eval: jnp.ndarray) -> jnp.ndarray:
    """Evaluate cosine coefficients on the vertical work grid."""
    return jnp.einsum("zn,n...->z...", C_eval, coeff)


def eval_sin(coeff: jnp.ndarray, S_eval: jnp.ndarray) -> jnp.ndarray:
    """Evaluate sine coefficients on the vertical work grid."""
    return jnp.einsum("zn,n...->z...", S_eval, coeff)


def project_cos(field: jnp.ndarray, C_proj: jnp.ndarray) -> jnp.ndarray:
    """Project a work-grid field onto cosine coefficients."""
    return jnp.einsum("nz,z...->n...", C_proj, field)


def project_sin(field: jnp.ndarray, S_proj: jnp.ndarray) -> jnp.ndarray:
    """Project a work-grid field onto sine coefficients."""
    return jnp.einsum("nz,z...->n...", S_proj, field)


def q_from_psi(psi_full: jnp.ndarray, denom: jnp.ndarray) -> jnp.ndarray:
    """Recover q from psi in spectral space on the work grid."""
    return -denom[None, :, :] * psi_full


def dpsi_dz_sin(psi_hat: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Derivative of cosine-mode psi as sine coefficients."""
    return jnp.einsum("sn,n...->s...", grid.cos_to_sin, psi_hat)


def dw_dz_cos(w_hat: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Derivative of sine-mode w as cosine coefficients."""
    return jnp.einsum("cn,n...->c...", grid.sin_to_cos, w_hat)


def d1_sin_to_work(field_hat: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Derivative of a sine-mode field evaluated on the work grid."""
    return eval_cos(dw_dz_cos(field_hat, grid), grid.C_eval)


def _horizontal_mean_product_dealiased(
    a_hat: jnp.ndarray, b_hat: jnp.ndarray, Nx: int, Npad: int
) -> jnp.ndarray:
    """Compute <a b>_xy on each work-grid vertical level with 3/2 padding."""
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
    """Explicit nonlinear terms for the trigonometric benchmark."""
    psi_hat = zero_mode(state.psi_hat)
    w_hat = zero_mode(state.w_hat)
    th_hat = zero_mode(state.th_hat)
    th_bar = state.th_bar

    psi_full = eval_cos(psi_hat, grid.C_eval)
    q_full = q_from_psi(psi_full, grid.denom)
    w_full = eval_sin(w_hat, grid.S_eval)
    th_full = eval_sin(th_hat, grid.S_eval)

    Aq_full, Aw_full, Ath_full = _triple_horizontal_advection(
        psi_full, q_full, w_full, th_full, grid
    )

    E_psi = project_cos(Aq_full, grid.C_proj) * grid.inv_denom[None, :, :]
    E_w = -project_sin(Aw_full, grid.S_proj)

    if grid.thermal_closure == "evolve_mean":
        dth_bar_full = d1_sin_to_work(th_bar, grid)
        E_th = -project_sin(
            Ath_full + dth_bar_full[:, None, None] * w_full, grid.S_proj
        )

        flux_full = _horizontal_mean_product_dealiased(w_full, th_full, grid.Nx, grid.Npad)
        flux_hat = project_sin(flux_full, grid.S_proj)
        dflux_full = d1_sin_to_work(flux_hat, grid)
        E_th_bar = -grid.mean_temp_eps_sq * project_sin(dflux_full, grid.S_proj)
    else:
        E_th = -project_sin(Ath_full, grid.S_proj)
        E_th_bar = jnp.zeros_like(th_bar)

    return State(zero_mode(E_psi), zero_mode(E_w), zero_mode(E_th), E_th_bar)


def implicit_tendency(state: State, grid: Grid) -> State:
    """Implicit linear vertical coupling and buoyancy terms."""
    psi_hat = zero_mode(state.psi_hat)
    w_hat = zero_mode(state.w_hat)
    th_hat = zero_mode(state.th_hat)

    psi_imp = -grid.inv_denom[None, :, :] * dw_dz_cos(w_hat, grid)
    w_imp = -dpsi_dz_sin(psi_hat, grid) + grid.Ra_sigma * th_hat
    th_imp = w_hat

    if grid.thermal_closure == "evolve_mean":
        th_bar_imp = (grid.mean_temp_eps_sq / grid.sigma) * (grid.d2_sin_diag * state.th_bar)
    else:
        th_bar_imp = jnp.zeros_like(state.th_bar)

    return State(zero_mode(psi_imp), zero_mode(w_imp), zero_mode(th_imp), th_bar_imp)


def _per_shell_diagmul(diag_shells: jnp.ndarray, ksq_idx: jnp.ndarray, field: jnp.ndarray) -> jnp.ndarray:
    """Apply shellwise diagonal matrices to a sine-mode spectral field."""
    diags = diag_shells[ksq_idx]
    f_t = jnp.transpose(field, (1, 2, 0))
    r_t = diags * f_t
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
    dR_psi = dpsi_dz_sin(R_psi, grid)
    rhs_w = R_w_eff - gamma * dt * grid.inv_alpha_q[None, :, :] * dR_psi
    w_new = _per_shell_diagmul(grid.w_solve_diag, grid.ksq_idx, rhs_w)

    dw_dz = dw_dz_cos(w_new, grid)
    psi_new = (R_psi - gamma * dt * grid.inv_denom[None, :, :] * dw_dz) * grid.inv_alpha_q[None, :, :]
    th_new = (R_th + gamma * dt * w_new) * grid.inv_alpha_th[None, :, :]

    return zero_mode(psi_new), zero_mode(w_new), zero_mode(th_new)


def imex_mean_temp_solve(R_th_bar: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Solve the sine-mode mean-temperature stage."""
    if grid.thermal_closure != "evolve_mean":
        return jnp.zeros_like(R_th_bar)
    return grid.thbar_solve_diag * R_th_bar


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


def make_initial_state(grid: Grid, seed: int = 0, amplitude: float = 1e-6) -> State:
    """Small random cosine-mode psi perturbation."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    psi_real = jax.random.normal(k1, (grid.Nc, grid.Nx, grid.Nk))
    psi_imag = jax.random.normal(k2, (grid.Nc, grid.Nx, grid.Nk))
    mode_weight = 1.0 / (1.0 + grid.cos_modes) ** 2
    psi_hat = amplitude * (psi_real + 1j * psi_imag) * mode_weight[:, None, None]
    psi_hat = zero_mode(psi_hat)

    w_hat = jnp.zeros((grid.Ns, grid.Nx, grid.Nk), dtype=psi_hat.dtype)
    th_hat = jnp.zeros_like(w_hat)
    th_bar = jnp.zeros((grid.Ns,), dtype=grid.z_work.dtype)
    return State(psi_hat, w_hat, th_hat, th_bar)


def run(grid: Grid, state: State, n_steps: int, save_interval: int, callback=None) -> State:
    """Run the trig benchmark for ``n_steps`` with callback cadence ``save_interval``."""
    stepper = partial(imex_step_ars222, grid=grid)

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
