"""Tests for the SYI22-style Rayleigh sponge layer."""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import pytest

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.solver import make_initial_state, imex_step
from nhqg.spectral import sponge_product_23, sponge_product_32


BASE = dict(Nx=128, Nz=8, L=20.0, Ra_tilde=50.0, sigma=1.0,
            beta=0.0, Ld=float('inf'), dt=1e-4, float_dtype='float64')


def _mask_23(Nx):
    Nk = Nx // 2 + 1
    K = Nx // 3
    kx = np.arange(Nx)
    kx = np.where(kx <= Nx // 2, kx, kx - Nx)
    ky = np.arange(Nk)
    return ((np.abs(kx)[:, None] <= K) & (ky[None, :] <= K)).astype(float)


def _sponge_profile(cfg):
    xy = np.arange(cfg.Nx) * (cfg.L / cfg.Nx)
    X, Y = np.meshgrid(xy, xy, indexing='ij')
    r = np.sqrt((X - cfg.L / 2.0) ** 2 + (Y - cfg.L / 2.0) ** 2)
    return cfg.sponge_rate * 0.5 * (
        1.0 + np.tanh(cfg.sponge_sharpness * (r - cfg.sponge_r_start)
                          / cfg.sponge_r_start))


def _zero_pad_np(f_hat, Nx, Npad):
    Nk = Nx // 2 + 1
    out = np.zeros((Npad, Npad // 2 + 1), dtype=np.complex128)
    out[:Nx // 2, :Nk] = f_hat[:Nx // 2, :]
    out[Npad - Nx // 2:, :Nk] = f_hat[Nx // 2:, :]
    return out


def _truncate_np(f_hat_pad, Nx, Npad):
    Nk = Nx // 2 + 1
    return np.concatenate([
        f_hat_pad[:Nx // 2, :Nk],
        f_hat_pad[Npad - Nx // 2:, :Nk],
    ], axis=0) * (Npad / Nx) ** 2


def test_sponge_off_gives_none():
    g = make_grid(NHQGConfig(**BASE))
    assert g.sponge_phys is None
    assert g.sponge_phys_pad is None


def test_uniform_reference_product_23():
    cfg = NHQGConfig(**BASE, gamma=0.0, sponge_rate=3.0,
                     sponge_r_start=6.0, sponge_sharpness=8.0,
                     horizontal_dealiasing='23_rule')
    g = make_grid(cfg)
    q_hat = np.fft.rfft2(np.random.default_rng(0).standard_normal(
        (cfg.Nz + 1, cfg.Nx, cfg.Nx)))

    product = np.array(sponge_product_23(q_hat, g.sponge_phys, g.mask_23, g.Nx))
    mask_ref = _mask_23(cfg.Nx)
    # Under the 2/3 rule the stored sigma is band-limited at build time; the
    # oracle must apply the same masking or it disagrees at the out-band level.
    sigma_ref = np.fft.irfft2(
        np.fft.rfft2(_sponge_profile(cfg)) * mask_ref, s=(cfg.Nx, cfg.Nx))
    reference = np.stack([
        np.fft.rfft2(sigma_ref * np.fft.irfft2(level, s=(cfg.Nx, cfg.Nx)))
        * mask_ref
        for level in q_hat
    ])

    assert np.allclose(product, reference, atol=1e-12, rtol=1e-12)


def test_uniform_reference_product_32():
    cfg = NHQGConfig(**BASE, gamma=0.0, sponge_rate=3.0,
                     sponge_r_start=6.0, sponge_sharpness=8.0,
                     horizontal_dealiasing='32_rule')
    g = make_grid(cfg)
    q_hat = np.fft.rfft2(np.random.default_rng(0).standard_normal(
        (cfg.Nz + 1, cfg.Nx, cfg.Nx)))

    product = np.array(sponge_product_32(q_hat, g.sponge_phys_pad, g.Nx, g.Npad))
    sigma_hat = np.fft.rfft2(_sponge_profile(cfg))
    sigma_pad = np.fft.irfft2(_zero_pad_np(sigma_hat, cfg.Nx, cfg.Npad),
                               s=(cfg.Npad, cfg.Npad))
    reference = np.stack([
        _truncate_np(
            np.fft.rfft2(sigma_pad * np.fft.irfft2(
                _zero_pad_np(level, cfg.Nx, cfg.Npad), s=(cfg.Npad, cfg.Npad))),
            cfg.Nx, cfg.Npad,
        )
        for level in q_hat
    ])

    assert np.allclose(product, reference, atol=1e-12, rtol=1e-12)


def test_sponge_localization():
    cfg = NHQGConfig(**BASE, gamma=0.0, sponge_r_start=6.0,
                     sponge_sharpness=10.0, sponge_rate=5.0,
                     horizontal_dealiasing='23_rule')
    g = make_grid(cfg)
    xy = np.arange(cfg.Nx) * (cfg.L / cfg.Nx)
    X, Y = np.meshgrid(xy, xy, indexing='ij')

    def _gaussian(x0, y0):
        dx = (X - x0 + cfg.L / 2.0) % cfg.L - cfg.L / 2.0
        dy = (Y - y0 + cfg.L / 2.0) % cfg.L - cfg.L / 2.0
        q = np.exp(-(dx ** 2 + dy ** 2) / (2.0 * (cfg.L / 16.0) ** 2))
        return q, np.fft.rfft2(np.repeat(q[None, :, :], cfg.Nz + 1, axis=0))

    q_center, q_center_hat = _gaussian(cfg.L / 2.0, cfg.L / 2.0)
    center_tendency = -np.fft.irfft2(
        np.array(sponge_product_23(q_center_hat, g.sponge_phys, g.mask_23, g.Nx)),
        s=(cfg.Nx, cfg.Nx), axes=(-2, -1))
    # Band-limited sigma carries Gibbs ringing in the sponge-free interior at
    # up to ~1e-5 of sigma_max (the guard bounds out-band ENERGY at 1e-10, i.e.
    # ~1e-5 in amplitude), so locality can only be certified to ~1e-4.
    assert np.max(np.abs(center_tendency)) < 1e-4 * cfg.sponge_rate * np.max(np.abs(q_center))

    q_edge, q_edge_hat = _gaussian(cfg.L / 2.0, 0.98 * cfg.L)
    edge_tendency = -np.fft.irfft2(
        np.array(sponge_product_23(q_edge_hat, g.sponge_phys, g.mask_23, g.Nx)),
        s=(cfg.Nx, cfg.Nx), axes=(-2, -1))[0]
    expected = -np.array(g.sponge_phys) * q_edge
    region = np.abs(q_edge) > 0.1 * np.max(np.abs(q_edge))
    error = np.max(np.abs(edge_tendency[region] - expected[region]))
    scale = np.max(np.abs(expected[region]))
    assert error < 0.05 * scale


def test_dirichlet_bcs_with_sponge():
    cfg = NHQGConfig(**BASE, gamma=5e-3, trap_sharpness=6.0,
                     sponge_rate=5.0, sponge_sharpness=8.0)
    g = make_grid(cfg)
    state = make_initial_state(g, amplitude=1e-2)
    for _ in range(5):
        state = imex_step(state, g)

    V = np.array(g.V)
    w_cheb = np.einsum('ij,j...->i...', np.array(g.w_stencil), np.array(state.w_hat))
    th_cheb = np.einsum('ij,j...->i...', np.array(g.dirichlet_stencil), np.array(state.th_hat))
    w_nodal = np.einsum('ij,j...->i...', V, w_cheb)
    th_nodal = np.einsum('ij,j...->i...', V, th_cheb)
    w_bdy = max(float(np.max(np.abs(w_nodal[0]))), float(np.max(np.abs(w_nodal[-1]))))
    th_bdy = max(float(np.max(np.abs(th_nodal[0]))), float(np.max(np.abs(th_nodal[-1]))))

    assert w_bdy < 1e-12
    assert th_bdy < 1e-12


def test_sponge_guard_trips():
    cfg = NHQGConfig(**BASE, gamma=0.0, sponge_rate=5.0,
                     sponge_r_start=6.0, sponge_sharpness=500.0)
    with pytest.raises(ValueError):
        make_grid(cfg)


def test_sponge_requires_r_start():
    cfg = NHQGConfig(**BASE, gamma=0.0, sponge_rate=1.0)
    with pytest.raises(ValueError):
        make_grid(cfg)
