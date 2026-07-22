"""Trigonometric vertical operators and shell-precomputed IMEX data."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from trig_vertical_benchmark.config import TrigBenchmarkConfig


class Grid(NamedTuple):
    """Precomputed arrays for the trigonometric benchmark solver."""

    z_work: jnp.ndarray
    z_weights: jnp.ndarray

    sin_modes: jnp.ndarray
    cos_modes: jnp.ndarray
    S_eval: jnp.ndarray
    C_eval: jnp.ndarray
    S_proj: jnp.ndarray
    C_proj: jnp.ndarray

    cos_to_sin: jnp.ndarray
    sin_to_cos: jnp.ndarray
    d2_sin_diag: jnp.ndarray

    kx: jnp.ndarray
    ky: jnp.ndarray
    ksq: jnp.ndarray
    denom: jnp.ndarray
    inv_denom: jnp.ndarray

    diss_rate_q: jnp.ndarray
    diss_rate_w: jnp.ndarray
    diss_rate_th: jnp.ndarray
    inv_alpha_q: jnp.ndarray
    inv_alpha_th: jnp.ndarray

    w_solve_diag: jnp.ndarray
    ksq_idx: jnp.ndarray
    thbar_solve_diag: jnp.ndarray

    dt: jnp.ndarray
    gamma_imex: jnp.ndarray
    Ra_sigma: jnp.ndarray
    sigma: jnp.ndarray
    mean_temp_eps_sq: jnp.ndarray

    Nx: int
    Nk: int
    Nz: int
    Ns: int
    Nc: int
    Npad: int
    Nz_work: int
    L: float
    thermal_closure: str
    nonlinear_advection: str
    vertical_dealias_factor: float


def _trap_weights(n_intervals: int, dz: float, dtype=np.float64) -> np.ndarray:
    w = np.full(n_intervals + 1, dz, dtype=dtype)
    w[0] = 0.5 * dz
    w[-1] = 0.5 * dz
    return w


def _projection_matrix(eval_mat: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted least-squares projector from work-grid values to modal coefficients."""
    W = np.diag(weights.astype(eval_mat.dtype))
    gram = eval_mat.T @ W @ eval_mat
    return np.linalg.solve(gram, eval_mat.T @ W)


def _build_trig_operators(
    cfg: TrigBenchmarkConfig, dtype=np.float64
) -> tuple[np.ndarray, ...]:
    Ns = cfg.Ns
    Nc = cfg.Nc
    Nz_work = cfg.Nz_work

    z_work = np.linspace(0.0, 1.0, Nz_work + 1, dtype=dtype)
    dz_work = 1.0 / Nz_work
    z_weights = _trap_weights(Nz_work, dz_work, dtype=dtype)

    sin_modes = np.arange(1, Ns + 1, dtype=dtype)
    cos_modes = np.arange(0, Nc, dtype=dtype)

    S_eval = np.sin(np.pi * z_work[:, None] * sin_modes[None, :])
    C_eval = np.cos(np.pi * z_work[:, None] * cos_modes[None, :])
    S_proj = _projection_matrix(S_eval, z_weights)
    C_proj = _projection_matrix(C_eval, z_weights)

    sin_to_cos = np.zeros((Nc, Ns), dtype=dtype)
    cos_to_sin = np.zeros((Ns, Nc), dtype=dtype)
    mode_factor = np.pi * sin_modes
    sin_to_cos[1:, :] = np.diag(mode_factor)
    cos_to_sin[:, 1:] = -np.diag(mode_factor)
    d2_sin_diag = -(mode_factor ** 2)

    return (
        z_work,
        z_weights,
        sin_modes,
        cos_modes,
        S_eval,
        C_eval,
        S_proj,
        C_proj,
        cos_to_sin,
        sin_to_cos,
        d2_sin_diag,
    )


def _build_w_solve_diag(
    d2_sin_diag: np.ndarray,
    ksq_flat: np.ndarray,
    dt: float,
    gamma: float,
    nu_q: float,
    nu_w: float,
    nu_theta: float,
    sigma: float,
    Ra_sigma: float,
    drag: float,
    hyper_order: int,
    dtype=np.float64,
):
    """Precompute the shellwise diagonal solve for the coupled psi-w-theta block."""
    ksq_rounded = np.round(ksq_flat, decimals=8)
    unique_ksq, inverse_idx = np.unique(ksq_rounded, return_inverse=True)
    n_shells = len(unique_ksq)
    inv_diag = np.zeros((n_shells, d2_sin_diag.size), dtype=dtype)
    lap_pos = -d2_sin_diag

    for s, ksq_val in enumerate(unique_ksq):
        ksq_p = ksq_val ** hyper_order
        diss_q = nu_q * ksq_p + drag
        diss_w = nu_w * ksq_p
        diss_th = (nu_theta / sigma) * ksq_p

        alpha_q = 1.0 + gamma * dt * diss_q
        alpha_w = 1.0 + gamma * dt * diss_w
        alpha_th = 1.0 + gamma * dt * diss_th
        alpha_w_eff = alpha_w - (gamma * dt) ** 2 * Ra_sigma / alpha_th

        if ksq_val == 0.0:
            diag = np.full_like(lap_pos, alpha_w_eff)
        else:
            coeff = (gamma * dt) ** 2 * (1.0 / ksq_val) / alpha_q
            diag = alpha_w_eff + coeff * lap_pos
        inv_diag[s] = 1.0 / diag

    return inv_diag, inverse_idx.astype(np.int32)


def make_grid(cfg: TrigBenchmarkConfig) -> Grid:
    """Construct the trigonometric vertical operators and shell data."""
    if cfg.Nz < 4:
        raise ValueError("trig benchmark requires Nz >= 4")

    build_dtype = np.float64
    target_dtype = np.float64 if cfg.float_dtype == "float64" else np.float32

    (
        z_work,
        z_weights,
        sin_modes,
        cos_modes,
        S_eval,
        C_eval,
        S_proj,
        C_proj,
        cos_to_sin,
        sin_to_cos,
        d2_sin_diag,
    ) = _build_trig_operators(cfg, dtype=build_dtype)

    Nx = cfg.Nx
    Nk = cfg.Nk
    Npad = cfg.Npad
    Ns = cfg.Ns
    Nc = cfg.Nc

    kx_1d = 2.0 * np.pi * np.fft.fftfreq(Nx, d=cfg.L / Nx)
    ky_1d = 2.0 * np.pi * np.arange(Nk) / cfg.L
    kx = kx_1d[:, None]
    ky = ky_1d[None, :]
    ksq = kx ** 2 + ky ** 2

    denom = ksq.copy()
    inv_denom = np.zeros_like(denom)
    np.divide(1.0, denom, out=inv_denom, where=denom > 0.0)

    gamma_imex = 1.0 - 1.0 / np.sqrt(2.0)
    p = cfg.hyper_order
    diss_rate_q = cfg.nu_q * ksq ** p + cfg.drag
    diss_rate_w = cfg.nu_w * ksq ** p
    diss_rate_th = (cfg.nu_theta / cfg.sigma) * ksq ** p
    inv_alpha_q = 1.0 / (1.0 + gamma_imex * cfg.dt * diss_rate_q)
    inv_alpha_th = 1.0 / (1.0 + gamma_imex * cfg.dt * diss_rate_th)

    w_solve_diag, ksq_idx = _build_w_solve_diag(
        d2_sin_diag,
        ksq.ravel(),
        cfg.dt,
        gamma_imex,
        cfg.nu_q,
        cfg.nu_w,
        cfg.nu_theta,
        cfg.sigma,
        cfg.Ra_tilde / cfg.sigma,
        cfg.drag,
        cfg.hyper_order,
        dtype=build_dtype,
    )
    ksq_idx = ksq_idx.reshape(Nx, Nk)

    alpha_bar = gamma_imex * cfg.dt * cfg.mean_temp_eps_sq / cfg.sigma
    thbar_solve_diag = 1.0 / (1.0 - alpha_bar * d2_sin_diag)

    def to_jax(arr, dtype=None):
        arr_dtype = target_dtype if dtype is None else dtype
        return jnp.array(arr, dtype=arr_dtype)

    return Grid(
        z_work=to_jax(z_work),
        z_weights=to_jax(z_weights),
        sin_modes=to_jax(sin_modes),
        cos_modes=to_jax(cos_modes),
        S_eval=to_jax(S_eval),
        C_eval=to_jax(C_eval),
        S_proj=to_jax(S_proj),
        C_proj=to_jax(C_proj),
        cos_to_sin=to_jax(cos_to_sin),
        sin_to_cos=to_jax(sin_to_cos),
        d2_sin_diag=to_jax(d2_sin_diag),
        kx=to_jax(kx),
        ky=to_jax(ky),
        ksq=to_jax(ksq),
        denom=to_jax(denom),
        inv_denom=to_jax(inv_denom),
        diss_rate_q=to_jax(diss_rate_q),
        diss_rate_w=to_jax(diss_rate_w),
        diss_rate_th=to_jax(diss_rate_th),
        inv_alpha_q=to_jax(inv_alpha_q),
        inv_alpha_th=to_jax(inv_alpha_th),
        w_solve_diag=to_jax(w_solve_diag),
        ksq_idx=jnp.array(ksq_idx, dtype=jnp.int32),
        thbar_solve_diag=to_jax(thbar_solve_diag),
        dt=to_jax(np.array(cfg.dt)),
        gamma_imex=to_jax(np.array(gamma_imex)),
        Ra_sigma=to_jax(np.array(cfg.Ra_tilde / cfg.sigma)),
        sigma=to_jax(np.array(cfg.sigma)),
        mean_temp_eps_sq=to_jax(np.array(cfg.mean_temp_eps_sq)),
        Nx=cfg.Nx,
        Nk=cfg.Nk,
        Nz=cfg.Nz,
        Ns=Ns,
        Nc=Nc,
        Npad=Npad,
        Nz_work=cfg.Nz_work,
        L=cfg.L,
        thermal_closure=cfg.thermal_closure,
        nonlinear_advection=cfg.nonlinear_advection,
        vertical_dealias_factor=cfg.vertical_dealias_factor,
    )
