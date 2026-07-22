"""Tests for the mixed w boundary conditions (M2 core):
Dirichlet bottom (w(0)=0), Neumann top (dw/dZ(1)=0), theta Dirichlet both ends.

Covers: the Shen stencil algebra, exact left inverse, both-Dirichlet
regression (shared arrays, scalar IMEX path), BC enforcement after stepping
through the full IMEX machinery (including the K-matrix buoyancy coupling),
that the top is genuinely open (w != 0 there), and IMEX-vs-RK4 agreement
under the mixed basis.
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import jax.numpy as jnp
import pytest

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid, _mixed_bd_tn_stencil
from nhqg.solver import (
    State, make_initial_state, imex_step, rk4_step, _dirichlet_to_cheb,
)


def _cfg(**kw):
    base = dict(Nx=32, Nz=16, L=20.0, Ra_tilde=100.0, sigma=1.0,
                beta=0.0, Ld=float('inf'), dt=1e-4, float_dtype='float64',
                thermal_closure='fixed_conduction')
    base.update(kw)
    return NHQGConfig(**base)


def _bc_values(w_hat, grid):
    """(w at bottom, dw/dZ at top, w at top) from w-Galerkin coefficients."""
    N = grid.Nz
    w_cheb = np.array(_dirichlet_to_cheb(w_hat, grid.w_stencil))
    n = np.arange(N + 1)
    e_bot = (-1.0) ** n                    # T_n(-1), Z=0
    e_top = np.ones(N + 1)                 # T_n(+1), Z=1
    dtop = np.array(e_top @ np.array(grid.G_Z))   # d/dZ row at Z=1
    w_bot = np.einsum('i,i...->...', e_bot, w_cheb)
    w_top = np.einsum('i,i...->...', e_top, w_cheb)
    dw_top = np.einsum('i,i...->...', dtop, w_cheb)
    return w_bot, dw_top, w_top


class TestStencilAlgebra:

    @pytest.mark.parametrize("N", [8, 16, 64])
    def test_basis_satisfies_bcs(self, N):
        S = _mixed_bd_tn_stencil(N)
        n = np.arange(N + 1)
        rng = np.random.default_rng(3)
        f = S @ rng.standard_normal(N - 1)
        assert abs(np.sum(f * (-1.0) ** n)) < 1e-12          # f(-1) = 0
        assert abs(np.sum(f * n.astype(float) ** 2)) < 1e-10  # f'(+1) = 0

    @pytest.mark.parametrize("N", [8, 16, 64])
    def test_exact_left_inverse(self, N):
        g = make_grid(_cfg(Nz=N, w_bc_top='neumann'))
        P = np.array(g.w_pinv) @ np.array(g.w_stencil)
        assert np.max(np.abs(P - np.eye(N - 1))) < 1e-11

    def test_both_dirichlet_shares_arrays(self):
        g = make_grid(_cfg())
        assert np.array_equal(np.array(g.w_stencil), np.array(g.dirichlet_stencil))
        assert np.array_equal(np.array(g.w_pinv), np.array(g.dirichlet_pinv))
        assert np.array_equal(np.array(g.proj_w), np.array(g.proj_dirichlet))
        assert g.map_w_to_th is None and g.map_th_to_w is None

    def test_evolve_mean_now_permitted(self):
        """M2b lifted the guard: neumann top + evolve_mean builds a grid.

        (Was test_evolve_mean_rejected_for_now; behavior gates live in
        tests/test_mixed_bc_plumbing.py.)
        """
        g = make_grid(_cfg(w_bc_top='neumann', thermal_closure='evolve_mean'))
        assert g.w_bc_top == 'neumann'
        assert g.map_w_to_th is not None and g.map_th_to_w is not None


class TestMixedBCDynamics:

    def _stepped(self, n_steps=10, **kw):
        g = make_grid(_cfg(w_bc_top='neumann', **kw))
        state = make_initial_state(g, seed=2, amplitude=1e-3)
        for _ in range(n_steps):
            state = imex_step(state, g)
        return state, g

    def test_bcs_enforced_after_steps(self):
        state, g = self._stepped(10)
        w_bot, dw_top, w_top = _bc_values(state.w_hat, g)
        scale = float(jnp.max(jnp.abs(state.w_hat))) + 1e-30
        assert np.max(np.abs(w_bot)) < 1e-11 * max(scale, 1.0)
        assert np.max(np.abs(dw_top)) < 1e-9 * max(scale, 1.0)
        # theta stays Dirichlet at both ends
        th_cheb = np.array(_dirichlet_to_cheb(state.th_hat, g.dirichlet_stencil))
        n = np.arange(g.Nz + 1)
        for row in (np.ones(g.Nz + 1), (-1.0) ** n):
            assert np.max(np.abs(np.einsum('i,i...->...', row, th_cheb))) < 1e-11

    def test_top_is_genuinely_open(self):
        """With the rigid lid removed, w at the surface must be alive --
        comparable to the interior maximum, not roundoff."""
        state, g = self._stepped(50)
        _, _, w_top = _bc_values(state.w_hat, g)
        scale = float(jnp.max(jnp.abs(state.w_hat)))
        assert scale > 0
        assert np.max(np.abs(w_top)) > 1e-3 * scale

    def test_rigid_lid_keeps_w_zero_at_top(self):
        """Control: the both-Dirichlet configuration pins w(top) to zero."""
        g = make_grid(_cfg())
        state = make_initial_state(g, seed=2, amplitude=1e-3)
        for _ in range(50):
            state = imex_step(state, g)
        _, _, w_top = _bc_values(state.w_hat, g)
        scale = float(jnp.max(jnp.abs(state.w_hat)))
        assert np.max(np.abs(w_top)) < 1e-10 * max(scale, 1.0)

    def test_imex_vs_rk4_mixed(self):
        g = make_grid(_cfg(w_bc_top='neumann', Ra_tilde=50.0, Nz=8))
        state0 = make_initial_state(g, seed=7, amplitude=1e-3)
        a, b = state0, state0
        for _ in range(20):
            a = imex_step(a, g)
            b = rk4_step(b, g)
        max_q = float(jnp.max(jnp.abs(b.q_hat)))
        rel = float(jnp.max(jnp.abs(a.q_hat - b.q_hat))) / max_q
        assert rel < 1e-3, f"IMEX/RK4 mixed-BC relative error {rel:.2e}"

    def test_finite_through_steps(self):
        state, _ = self._stepped(30)
        for f in (state.q_hat, state.w_hat, state.th_hat, state.th_bar):
            assert bool(jnp.all(jnp.isfinite(f)))
