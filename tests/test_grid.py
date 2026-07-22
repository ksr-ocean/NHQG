"""Tests for grid infrastructure: coefficient-space derivatives, V/V_inv
transforms, tau BC projections, CC integration, IMEX shells."""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import jax.numpy as jnp
import pytest

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid


@pytest.fixture
def grid8():
    """Small grid for unit tests."""
    cfg = NHQGConfig(Nx=32, Nz=8, L=20.0, Ra_tilde=100.0,
                     dt=1e-3, float_dtype='float64')
    return make_grid(cfg)


@pytest.fixture
def grid32():
    """Moderate-resolution grid."""
    cfg = NHQGConfig(Nx=64, Nz=32, L=20.0, Ra_tilde=100.0,
                     dt=1e-3, float_dtype='float64')
    return make_grid(cfg)


class TestTransforms:
    """V and V_inv must be exact inverses."""

    def test_roundtrip_V_Vinv(self, grid8):
        """V @ V_inv = I."""
        V = np.array(grid8.V)
        V_inv = np.array(grid8.V_inv)
        I = V @ V_inv
        np.testing.assert_allclose(I, np.eye(grid8.Nz + 1), atol=1e-13)

    def test_roundtrip_Vinv_V(self, grid8):
        """V_inv @ V = I."""
        V = np.array(grid8.V)
        V_inv = np.array(grid8.V_inv)
        I = V_inv @ V
        np.testing.assert_allclose(I, np.eye(grid8.Nz + 1), atol=1e-13)

    def test_V_evaluates_chebyshev(self, grid8):
        """V[j, n] = T_n(xi_j)."""
        V = np.array(grid8.V)
        xi = np.array(grid8.xi)
        N = grid8.Nz
        for n in range(N + 1):
            Tn = np.cos(n * np.arccos(xi))
            np.testing.assert_allclose(V[:, n], Tn, atol=1e-14)

    def test_dirichlet_stencil_roundtrip(self, grid8):
        """Dirichlet stencil and pseudoinverse should invert on the Galerkin subspace."""
        G = np.array(grid8.dirichlet_stencil)
        G_pinv = np.array(grid8.dirichlet_pinv)
        I = G_pinv @ G
        np.testing.assert_allclose(I, np.eye(grid8.Nz_gal), atol=1e-13)

    def test_cheb_3o2_gauss_dealias_roundtrip_for_base_modes(self):
        """Coral-style Gauss work grid should exactly recover base Chebyshev modes."""
        cfg = NHQGConfig(
            Nx=32, Nz=8, L=20.0, Ra_tilde=100.0,
            dt=1e-3, float_dtype='float64',
            vertical_dealiasing='cheb_3o2',
        )
        g = make_grid(cfg)

        Z = np.array(g.Z)
        coeffs = np.array(g.V_inv) @ np.sin(np.pi * Z)
        nodal_hi = np.array(g.V_dealias) @ coeffs
        coeffs_hi = np.array(g.V_dealias_inv) @ nodal_hi

        np.testing.assert_allclose(coeffs_hi[:g.Nz + 1], coeffs, atol=1e-12)
        np.testing.assert_allclose(coeffs_hi[g.Nz + 1:], 0.0, atol=1e-12)

    def test_sbp_return_map_is_mass_adjoint(self):
        """The SBP->CGL transfer should be the weighted adjoint of CGL->SBP."""
        cfg = NHQGConfig(
            Nx=64, Nz=32, L=20.0, Ra_tilde=100.0,
            dt=1e-3, float_dtype='float64',
            sbp_transfer_mode='weighted_polar',
        )
        g = make_grid(cfg)
        T = np.array(g.cgl_to_sbp)
        S = np.array(g.sbp_to_cgl)
        H = np.array(g.sbp_H)
        M_cc = np.diag(np.array(g.cc_weights))

        np.testing.assert_allclose(M_cc @ S, T.T @ H, atol=1e-13)

    def test_sbp_transfer_preserves_weighted_identity(self):
        """The weighted-polar transfer pair should roundtrip exactly."""
        cfg = NHQGConfig(
            Nx=64, Nz=32, L=20.0, Ra_tilde=100.0,
            dt=1e-3, float_dtype='float64',
            sbp_transfer_mode='weighted_polar',
        )
        g = make_grid(cfg)
        T = np.array(g.cgl_to_sbp)
        S = np.array(g.sbp_to_cgl)

        np.testing.assert_allclose(S @ T, np.eye(g.Nz + 1), atol=1e-12)
        np.testing.assert_allclose(T @ S, np.eye(g.Nz + 1), atol=1e-12)


class TestCoefficientDerivatives:
    """G_Z must correctly differentiate Chebyshev representations of polynomials."""

    @pytest.mark.parametrize("deg", [1, 2, 3, 5, 8])
    def test_polynomial_derivative_nz8(self, grid8, deg):
        """G_Z applied to coefficients of Z^d gives coefficients of d*Z^{d-1}."""
        Z = np.array(grid8.Z)
        V_inv = np.array(grid8.V_inv)
        V = np.array(grid8.V)
        G_Z = np.array(grid8.G_Z)

        f_nodal = Z ** deg
        f_coeffs = V_inv @ f_nodal
        df_coeffs = G_Z @ f_coeffs
        df_nodal = V @ df_coeffs
        df_exact = deg * Z ** (deg - 1) if deg > 0 else np.zeros_like(Z)
        np.testing.assert_allclose(df_nodal, df_exact, atol=1e-10)

    @pytest.mark.parametrize("deg", [1, 5, 10, 20, 32])
    def test_polynomial_derivative_nz32(self, grid32, deg):
        Z = np.array(grid32.Z)
        V_inv = np.array(grid32.V_inv)
        V = np.array(grid32.V)
        G_Z = np.array(grid32.G_Z)

        f_nodal = Z ** deg
        f_coeffs = V_inv @ f_nodal
        df_coeffs = G_Z @ f_coeffs
        df_nodal = V @ df_coeffs
        df_exact = deg * Z ** (deg - 1)
        np.testing.assert_allclose(df_nodal, df_exact, atol=1e-8)

    @pytest.mark.parametrize("deg", [2, 3, 5, 8])
    def test_polynomial_second_derivative_nz8(self, grid8, deg):
        Z = np.array(grid8.Z)
        V_inv = np.array(grid8.V_inv)
        V = np.array(grid8.V)
        G_Z2 = np.array(grid8.G_Z2)

        f_nodal = Z ** deg
        f_coeffs = V_inv @ f_nodal
        d2f_coeffs = G_Z2 @ f_coeffs
        d2f_nodal = V @ d2f_coeffs
        d2f_exact = deg * (deg - 1) * Z ** (deg - 2)
        np.testing.assert_allclose(d2f_nodal, d2f_exact, atol=1e-9)

    @pytest.mark.parametrize("deg", [2, 5, 10, 20, 32])
    def test_polynomial_second_derivative_nz32(self, grid32, deg):
        Z = np.array(grid32.Z)
        V_inv = np.array(grid32.V_inv)
        V = np.array(grid32.V)
        G_Z2 = np.array(grid32.G_Z2)

        f_nodal = Z ** deg
        f_coeffs = V_inv @ f_nodal
        d2f_coeffs = G_Z2 @ f_coeffs
        d2f_nodal = V @ d2f_coeffs
        d2f_exact = deg * (deg - 1) * Z ** (deg - 2)
        np.testing.assert_allclose(d2f_nodal, d2f_exact, atol=1e-6)

    def test_sin_second_derivative(self, grid32):
        Z = np.array(grid32.Z)
        V_inv = np.array(grid32.V_inv)
        V = np.array(grid32.V)
        G_Z2 = np.array(grid32.G_Z2)

        f_nodal = np.sin(np.pi * Z)
        f_coeffs = V_inv @ f_nodal
        d2f_coeffs = G_Z2 @ f_coeffs
        d2f_nodal = V @ d2f_coeffs
        d2f_exact = -(np.pi ** 2) * np.sin(np.pi * Z)
        np.testing.assert_allclose(d2f_nodal, d2f_exact, atol=1e-8)


class TestTauProjection:
    """Tau projections must enforce BCs when evaluated at CGL nodes."""

    def test_dirichlet_satisfied(self, grid8):
        """After Dirichlet projection, f(Z=0) = f(Z=1) = 0."""
        V = np.array(grid8.V)
        proj = np.array(grid8.proj_dirichlet)

        rng = np.random.default_rng(42)
        a = rng.standard_normal(grid8.Nz + 1)
        a_proj = proj @ a
        f_nodal = V @ a_proj

        np.testing.assert_allclose(f_nodal[0], 0.0, atol=1e-13)   # Z=1
        np.testing.assert_allclose(f_nodal[-1], 0.0, atol=1e-13)  # Z=0

    def test_neumann_satisfied(self, grid8):
        """After Neumann projection, df/dZ = 0 at Z=0 and Z=1."""
        V = np.array(grid8.V)
        G_Z = np.array(grid8.G_Z)
        proj = np.array(grid8.proj_neumann)

        rng = np.random.default_rng(42)
        a = rng.standard_normal(grid8.Nz + 1)
        a_proj = proj @ a
        df_coeffs = G_Z @ a_proj
        df_nodal = V @ df_coeffs

        np.testing.assert_allclose(df_nodal[0], 0.0, atol=1e-12)   # Z=1
        np.testing.assert_allclose(df_nodal[-1], 0.0, atol=1e-12)  # Z=0

    def test_dirichlet_idempotent(self, grid8):
        """Projecting twice = projecting once."""
        proj = np.array(grid8.proj_dirichlet)
        rng = np.random.default_rng(123)
        a = rng.standard_normal(grid8.Nz + 1)
        a1 = proj @ a
        a2 = proj @ a1
        np.testing.assert_allclose(a2, a1, atol=1e-14)

    def test_neumann_idempotent(self, grid8):
        proj = np.array(grid8.proj_neumann)
        rng = np.random.default_rng(123)
        a = rng.standard_normal(grid8.Nz + 1)
        a1 = proj @ a
        a2 = proj @ a1
        np.testing.assert_allclose(a2, a1, atol=1e-14)


class TestClenshawCurtis:
    """CC quadrature weights must integrate polynomials exactly."""

    def test_integral_of_one(self, grid8):
        w = np.array(grid8.cc_weights)
        np.testing.assert_allclose(np.sum(w), 1.0, atol=1e-14)

    def test_integral_of_z(self, grid8):
        w = np.array(grid8.cc_weights)
        Z = np.array(grid8.Z)
        np.testing.assert_allclose(np.sum(w * Z), 0.5, atol=1e-14)

    def test_integral_of_z_squared(self, grid8):
        w = np.array(grid8.cc_weights)
        Z = np.array(grid8.Z)
        np.testing.assert_allclose(np.sum(w * Z ** 2), 1.0 / 3.0, atol=1e-14)

    def test_integral_of_sin_pi_z(self, grid32):
        w = np.array(grid32.cc_weights)
        Z = np.array(grid32.Z)
        np.testing.assert_allclose(
            np.sum(w * np.sin(np.pi * Z)), 2.0 / np.pi, atol=1e-14
        )


class TestIMEXShells:
    """IMEX |k|^2 shell deduplication."""

    def test_shell_count_less_than_total(self, grid8):
        n_shells = grid8.imex_inv.shape[0]
        total = grid8.Nx * grid8.Nk
        assert n_shells < total

    def test_shell_idx_in_range(self, grid8):
        idx = np.array(grid8.ksq_idx)
        n_shells = grid8.imex_inv.shape[0]
        assert np.all(idx >= 0)
        assert np.all(idx < n_shells)
