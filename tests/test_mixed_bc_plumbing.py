"""M2b gates: per-field w lifts in exchange/io/diagnostics + restriction lift.

Lead-written acceptance gates for the M2b plumbing task (see
polar_512_todo.md). READ-ONLY for the executor: passing a gate by editing
this file (or tests/data/m2b_dirichlet_evolvemean_ref.npz) is a hard
failure.

State before M2b: gates 1-3 are RED (make_grid raises NotImplementedError
for w_bc_top='neumann' with evolve_mean or vertical_cutoff_n); gate 4
(bitwise both-Dirichlet regression) is GREEN and must stay GREEN.
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'

from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pytest

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.solver import make_initial_state, imex_step

DATA = Path(__file__).parent / "data"


def _cfg(**kw):
    base = dict(Nx=32, Nz=16, L=20.0, Ra_tilde=100.0, sigma=1.0, beta=0.0,
                Ld=float('inf'), dt=1e-3, float_dtype='float64',
                nu_q=1.0, nu_w=1.0, nu_theta=1.0, hyper_order=1,
                nonlinear_advection='flux', horizontal_dealiasing='23_rule')
    base.update(kw)
    return NHQGConfig(**base)


def _w_bc_errors(state, grid):
    """(max |w(bottom)|, max |dw/dZ(top)|) in nodal space, all wavenumbers."""
    w_cheb = jnp.einsum('nm,m...->n...', grid.w_stencil, state.w_hat)
    w_bot = jnp.einsum('n,n...->...', grid.V[-1], w_cheb)      # CGL row -1 = Z=0
    dw_cheb = jnp.einsum('nm,m...->n...', grid.G_Z, w_cheb)
    dw_top = jnp.einsum('n,n...->...', grid.V[0], dw_cheb)     # CGL row 0 = Z=1
    return float(jnp.max(jnp.abs(w_bot))), float(jnp.max(jnp.abs(dw_top)))


class TestM2bGates:

    def test_gate1_neumann_evolve_mean_runs(self):
        """Open-top + evolve_mean constructs, steps, stays finite, BCs exact."""
        cfg = _cfg(w_bc_top='neumann', thermal_closure='evolve_mean',
                   mean_exchange_discretization='balanced_sbp2_pc',
                   sbp_corrector_substeps=2)
        g = make_grid(cfg)
        s = make_initial_state(g, seed=0, amplitude=1e-3)
        for _ in range(20):
            s = imex_step(s, g)
        for f in (s.q_hat, s.w_hat, s.th_hat, s.th_bar):
            assert bool(jnp.all(jnp.isfinite(f)))
        w_bot, dw_top = _w_bc_errors(s, g)
        assert w_bot < 1e-12
        assert dw_top < 1e-10
        # Theta_bar Dirichlet values preserved
        th_bar_nodal = np.array(g.V) @ np.array(s.th_bar)
        assert abs(th_bar_nodal[0]) < 1e-12 and abs(th_bar_nodal[-1]) < 1e-12

    def test_gate2_neumann_exchange_residual_structural(self):
        """The SBP exchange-residual identity survives the open top.

        The boundary term of the SBP audit is killed by theta's Dirichlet
        rows, which M2 did not touch -- it must still vanish with the
        mixed-basis w lift.
        """
        from nhqg.diagnostics import compute_diagnostics
        cfg = _cfg(w_bc_top='neumann', thermal_closure='evolve_mean',
                   mean_exchange_discretization='balanced_sbp2_pc',
                   sbp_corrector_substeps=2)
        g = make_grid(cfg)
        s = make_initial_state(g, seed=0, amplitude=1e-3)
        for _ in range(10):
            s = imex_step(s, g)
        diag = compute_diagnostics(s, g)
        assert abs(float(diag['mean_theta_exchange_residual_sbp'])) < 1e-8

    def test_gate3_neumann_vertical_cutoff(self):
        """vertical_cutoff_n reprojection preserves the mixed BCs exactly."""
        cfg = _cfg(w_bc_top='neumann', thermal_closure='fixed_conduction',
                   vertical_cutoff_n=12)
        g = make_grid(cfg)
        s = make_initial_state(g, seed=0, amplitude=1e-3)
        for _ in range(10):
            s = imex_step(s, g)
        for f in (s.q_hat, s.w_hat, s.th_hat):
            assert bool(jnp.all(jnp.isfinite(f)))
        w_bot, dw_top = _w_bc_errors(s, g)
        assert w_bot < 1e-12
        assert dw_top < 1e-10

    def test_gate4_dirichlet_evolve_mean_bitwise_regression(self):
        """Both-Dirichlet evolve_mean trajectory is bitwise-unchanged by M2b.

        Reference generated pre-M2b (2026-07-22) on this machine, CPU
        float64, 20 steps. Bitwise: the dirichlet w_stencil/proj_w/w_pinv
        ARE the theta arrays, so the plumbing must be an exact no-op here.
        """
        cfg = _cfg(thermal_closure='evolve_mean',
                   mean_exchange_discretization='balanced_sbp2_pc',
                   sbp_corrector_substeps=2)
        g = make_grid(cfg)
        s = make_initial_state(g, seed=0, amplitude=1e-3)
        for _ in range(20):
            s = imex_step(s, g)
        ref = np.load(DATA / "m2b_dirichlet_evolvemean_ref.npz")
        assert np.array_equal(np.array(s.q_hat), ref['q_hat'])
        assert np.array_equal(np.array(s.w_hat), ref['w_hat'])
        assert np.array_equal(np.array(s.th_hat), ref['th_hat'])
        assert np.array_equal(np.array(s.th_bar), ref['th_bar'])
