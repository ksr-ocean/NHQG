"""Tests for spectral operations: Jacobian accuracy, dealiasing, antisymmetry."""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import jax.numpy as jnp
import pytest

from nhqg.spectral import (
    conservative_flux_divergence_dealiased,
    jacobian_dealiased,
    triple_conservative_flux_divergence,
    triple_jacobian,
)


@pytest.fixture
def spectral_setup():
    """Common spectral grid setup."""
    Nx = 64
    Nk = Nx // 2 + 1
    Npad = 3 * Nx // 2
    L = 2 * np.pi

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    ky = 2 * np.pi * np.arange(Nk) / L
    kx_2d = jnp.array(kx[:, None])
    ky_2d = jnp.array(ky[None, :])

    x = np.linspace(0, L, Nx, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing='ij')

    return Nx, Nk, Npad, L, kx_2d, ky_2d, X, Y


class TestJacobian:
    """Dealiased Jacobian must match analytic results."""

    def test_sin_x_sin_y(self, spectral_setup):
        """J[sin(x), sin(y)] = cos(x)*cos(y)."""
        Nx, Nk, Npad, L, kx, ky, X, Y = spectral_setup

        A_hat = jnp.fft.rfft2(jnp.array(np.sin(X)))
        B_hat = jnp.fft.rfft2(jnp.array(np.sin(Y)))
        J_hat = jacobian_dealiased(A_hat, B_hat, kx, ky, Nx, Npad)
        J_phys = jnp.fft.irfft2(J_hat, s=(Nx, Nx))
        exact = np.cos(X) * np.cos(Y)
        np.testing.assert_allclose(np.array(J_phys), exact, atol=1e-12)

    def test_sin_2x_sin_3y(self, spectral_setup):
        """J[sin(2x), sin(3y)] = 6*cos(2x)*cos(3y)."""
        Nx, Nk, Npad, L, kx, ky, X, Y = spectral_setup

        A_hat = jnp.fft.rfft2(jnp.array(np.sin(2 * X)))
        B_hat = jnp.fft.rfft2(jnp.array(np.sin(3 * Y)))
        J_hat = jacobian_dealiased(A_hat, B_hat, kx, ky, Nx, Npad)
        J_phys = jnp.fft.irfft2(J_hat, s=(Nx, Nx))
        exact = 6 * np.cos(2 * X) * np.cos(3 * Y)
        np.testing.assert_allclose(np.array(J_phys), exact, atol=1e-11)

    def test_antisymmetry(self, spectral_setup):
        """J[A,B] = -J[B,A]."""
        Nx, Nk, Npad, L, kx, ky, X, Y = spectral_setup

        A_hat = jnp.fft.rfft2(jnp.array(np.sin(X) + 0.5 * np.cos(3 * Y)))
        B_hat = jnp.fft.rfft2(jnp.array(np.cos(2 * X) * np.sin(Y)))

        J_ab = jacobian_dealiased(A_hat, B_hat, kx, ky, Nx, Npad)
        J_ba = jacobian_dealiased(B_hat, A_hat, kx, ky, Nx, Npad)

        np.testing.assert_allclose(
            np.array(J_ab + J_ba), 0.0, atol=1e-12
        )

    def test_self_jacobian_zero(self, spectral_setup):
        """J[A,A] = 0."""
        Nx, Nk, Npad, L, kx, ky, X, Y = spectral_setup

        A_hat = jnp.fft.rfft2(jnp.array(np.sin(X) * np.cos(Y)))
        J_aa = jacobian_dealiased(A_hat, A_hat, kx, ky, Nx, Npad)
        J_phys = jnp.fft.irfft2(J_aa, s=(Nx, Nx))

        np.testing.assert_allclose(np.array(J_phys), 0.0, atol=1e-12)

    def test_dealiasing_matters(self, spectral_setup):
        """Products that alias should differ between padded and unpadded."""
        Nx, Nk, Npad, L, kx, ky, X, Y = spectral_setup

        # Use high-frequency modes that will alias at Nx=64
        A = np.sin(20 * X) * np.cos(20 * Y)
        B = np.cos(20 * X) * np.sin(20 * Y)
        A_hat = jnp.fft.rfft2(jnp.array(A))
        B_hat = jnp.fft.rfft2(jnp.array(B))

        # Dealiased Jacobian
        J_deal = jacobian_dealiased(A_hat, B_hat, kx, ky, Nx, Npad)

        # Direct (aliased) computation
        Ax = jnp.fft.irfft2(1j * kx * A_hat, s=(Nx, Nx))
        Ay = jnp.fft.irfft2(1j * ky * A_hat, s=(Nx, Nx))
        Bx = jnp.fft.irfft2(1j * kx * B_hat, s=(Nx, Nx))
        By = jnp.fft.irfft2(1j * ky * B_hat, s=(Nx, Nx))
        J_alias = jnp.fft.rfft2(Ax * By - Ay * Bx)

        # They should differ (dealiasing removes aliased energy)
        diff = float(jnp.max(jnp.abs(J_deal - J_alias)))
        assert diff > 1e-6, f"Aliased and dealiased agree too well: {diff}"

    def test_conservative_flux_matches_jacobian(self, spectral_setup):
        """div(uB, vB) should match J[psi, B] for incompressible u,v."""
        Nx, Nk, Npad, L, kx, ky, X, Y = spectral_setup

        psi_hat = jnp.fft.rfft2(jnp.array(np.sin(X) + 0.25 * np.cos(2 * Y)))
        B_hat = jnp.fft.rfft2(jnp.array(np.cos(2 * X - Y)))

        J_hat = jacobian_dealiased(psi_hat, B_hat, kx, ky, Nx, Npad)
        F_hat = conservative_flux_divergence_dealiased(psi_hat, B_hat, kx, ky, Nx, Npad)

        np.testing.assert_allclose(np.array(F_hat), np.array(J_hat), atol=1e-11)


class TestTripleJacobian:
    """Fused triple-Jacobian must match individual evaluations."""

    def test_consistency_with_single(self, spectral_setup):
        Nx, Nk, Npad, L, kx, ky, X, Y = spectral_setup
        Nz = 4

        A_hat = jnp.fft.rfft2(jnp.array(np.sin(X)))
        B_hat = jnp.fft.rfft2(jnp.array(np.cos(Y)))
        C_hat = jnp.fft.rfft2(jnp.array(np.sin(2 * X + Y)))

        psi = jnp.stack([A_hat] * (Nz + 1))
        q = jnp.stack([B_hat] * (Nz + 1))
        w = jnp.stack([C_hat] * (Nz + 1))
        th = jnp.stack([A_hat * 0.5] * (Nz + 1))

        Jq, Jw, Jth = triple_jacobian(psi, q, w, th, kx, ky, Nx, Npad)

        # Compare with individual evaluations
        Jq_ref = jacobian_dealiased(A_hat, B_hat, kx, ky, Nx, Npad)
        Jw_ref = jacobian_dealiased(A_hat, C_hat, kx, ky, Nx, Npad)

        np.testing.assert_allclose(np.array(Jq[0]), np.array(Jq_ref), atol=1e-12)
        np.testing.assert_allclose(np.array(Jw[0]), np.array(Jw_ref), atol=1e-12)

    def test_flux_triple_matches_jacobian_triple(self, spectral_setup):
        Nx, Nk, Npad, L, kx, ky, X, Y = spectral_setup
        Nz = 3

        psi_hat = jnp.fft.rfft2(jnp.array(np.sin(X) + 0.3 * np.cos(Y)))
        q_hat = jnp.fft.rfft2(jnp.array(np.cos(2 * X - Y)))
        w_hat = jnp.fft.rfft2(jnp.array(np.sin(X + 2 * Y)))
        th_hat = jnp.fft.rfft2(jnp.array(np.cos(3 * X) * np.sin(Y)))

        psi = jnp.stack([psi_hat] * (Nz + 1))
        q = jnp.stack([q_hat] * (Nz + 1))
        w = jnp.stack([w_hat] * (Nz + 1))
        th = jnp.stack([th_hat] * (Nz + 1))

        Jq, Jw, Jth = triple_jacobian(psi, q, w, th, kx, ky, Nx, Npad)
        Fq, Fw, Fth = triple_conservative_flux_divergence(psi, q, w, th, kx, ky, Nx, Npad)

        np.testing.assert_allclose(np.array(Fq), np.array(Jq), atol=1e-11)
        np.testing.assert_allclose(np.array(Fw), np.array(Jw), atol=1e-11)
        np.testing.assert_allclose(np.array(Fth), np.array(Jth), atol=1e-11)
