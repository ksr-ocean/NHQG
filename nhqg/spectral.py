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


@partial(jax.jit, static_argnums=(3,))
def sponge_product_23(f_hat: jnp.ndarray, sponge_phys: jnp.ndarray,
                      mask_23: jnp.ndarray, Nx: int) -> jnp.ndarray:
    """Dealiased product sigma*f under the 2/3 rule: product on the Nx grid,
    output-masked. f_hat: (Nz_levels, Nx, Nk) complex, one vertical batch axis.
    """
    f = jnp.fft.irfft2(f_hat, s=(Nx, Nx))
    return jnp.fft.rfft2(sponge_phys[None, :, :] * f) * mask_23[None, :, :]


@partial(jax.jit, static_argnums=(2, 3))
def sponge_product_32(f_hat: jnp.ndarray, sponge_phys_pad: jnp.ndarray,
                      Nx: int, Npad: int) -> jnp.ndarray:
    """Dealiased product sigma*f under the 3/2 rule: zero-pad f (per level via
    vmap over the leading axis, mirroring triple_jacobian's batching pattern),
    multiply by the pre-attenuated padded sigma, truncate back (which applies
    the (Npad/Nx)^2 factor).
    """
    f_pad_hat = jax.vmap(lambda level: _zero_pad(level, Nx, Npad))(f_hat)
    f_pad = jnp.fft.irfft2(f_pad_hat, s=(Npad, Npad))
    prod_pad_hat = jnp.fft.rfft2(sponge_phys_pad[None, :, :] * f_pad)
    return jax.vmap(lambda level: _truncate(level, Nx, Npad))(prod_pad_hat)


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


# ---------------------------------------------------------------------------
# 2/3-rule horizontal dealiasing (unpadded FFT + high-k mask)
# ---------------------------------------------------------------------------

def _make_23_mask(Nx: int, dtype=jnp.float64) -> jnp.ndarray:
    """Boolean keep-mask on the (Nx, Nx//2+1) rfft2 grid for 2/3-rule dealiasing.

    Retains modes with |kx_int| <= Nx//3 and ky_int <= Nx//3. For a quadratic
    nonlinearity of fields supported on this set, the aliased part of the
    product lands outside the retained set and is zeroed.
    """
    Nk = Nx // 2 + 1
    K = Nx // 3
    i = jnp.arange(Nx)
    kx_int = jnp.where(i <= Nx // 2, i, i - Nx)
    ky_int = jnp.arange(Nk)
    keep = (jnp.abs(kx_int)[:, None] <= K) & (ky_int[None, :] <= K)
    return keep.astype(dtype)


def _apply_23_mask(f_hat: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Zero out modes outside the 2/3 cutoff via a precomputed mask."""
    return f_hat * mask


def _triple_jacobian_one_level_23(psi_hat: jnp.ndarray,
                                  q_hat: jnp.ndarray,
                                  w_hat: jnp.ndarray,
                                  th_hat: jnp.ndarray,
                                  kx: jnp.ndarray,
                                  ky: jnp.ndarray,
                                  mask: jnp.ndarray,
                                  Nx: int):
    """2/3-rule J[ψ, f] for f in {q, w, θ} at one Z level. FFTs on the Nx grid."""
    psi_x_hat = 1j * kx * psi_hat
    psi_y_hat = 1j * ky * psi_hat
    psi_x = jnp.fft.irfft2(psi_x_hat, s=(Nx, Nx))
    psi_y = jnp.fft.irfft2(psi_y_hat, s=(Nx, Nx))

    def _one_jac(f_hat):
        fx_hat = 1j * kx * f_hat
        fy_hat = 1j * ky * f_hat
        fx = jnp.fft.irfft2(fx_hat, s=(Nx, Nx))
        fy = jnp.fft.irfft2(fy_hat, s=(Nx, Nx))
        J_phys = psi_x * fy - psi_y * fx
        J_hat = jnp.fft.rfft2(J_phys)
        return _apply_23_mask(J_hat, mask)

    return _one_jac(q_hat), _one_jac(w_hat), _one_jac(th_hat)


def _triple_flux_divergence_one_level_23(psi_hat: jnp.ndarray,
                                         q_hat: jnp.ndarray,
                                         w_hat: jnp.ndarray,
                                         th_hat: jnp.ndarray,
                                         kx: jnp.ndarray,
                                         ky: jnp.ndarray,
                                         mask: jnp.ndarray,
                                         Nx: int):
    """2/3-rule div(u f, v f) with u=-ψ_y, v=ψ_x at one Z level."""
    u_hat = -1j * ky * psi_hat
    v_hat = 1j * kx * psi_hat
    u = jnp.fft.irfft2(u_hat, s=(Nx, Nx))
    v = jnp.fft.irfft2(v_hat, s=(Nx, Nx))

    def _one_div(f_hat):
        f = jnp.fft.irfft2(f_hat, s=(Nx, Nx))
        uf_hat = jnp.fft.rfft2(u * f)
        vf_hat = jnp.fft.rfft2(v * f)
        div_hat = 1j * kx * uf_hat + 1j * ky * vf_hat
        return _apply_23_mask(div_hat, mask)

    return _one_div(q_hat), _one_div(w_hat), _one_div(th_hat)


@partial(jax.jit, static_argnums=(7,))
def triple_jacobian_23(psi_hat: jnp.ndarray,
                       q_hat: jnp.ndarray,
                       w_hat: jnp.ndarray,
                       th_hat: jnp.ndarray,
                       kx: jnp.ndarray,
                       ky: jnp.ndarray,
                       mask: jnp.ndarray,
                       Nx: int):
    """Batched 2/3-rule triple-Jacobian over all Z levels (FFTs on Nx grid)."""
    fn = partial(_triple_jacobian_one_level_23,
                 kx=kx, ky=ky, mask=mask, Nx=Nx)
    return jax.vmap(fn)(psi_hat, q_hat, w_hat, th_hat)


@partial(jax.jit, static_argnums=(7,))
def triple_conservative_flux_divergence_23(psi_hat: jnp.ndarray,
                                           q_hat: jnp.ndarray,
                                           w_hat: jnp.ndarray,
                                           th_hat: jnp.ndarray,
                                           kx: jnp.ndarray,
                                           ky: jnp.ndarray,
                                           mask: jnp.ndarray,
                                           Nx: int):
    """Batched 2/3-rule conservative-form advection over all Z levels."""
    fn = partial(_triple_flux_divergence_one_level_23,
                 kx=kx, ky=ky, mask=mask, Nx=Nx)
    return jax.vmap(fn)(psi_hat, q_hat, w_hat, th_hat)
