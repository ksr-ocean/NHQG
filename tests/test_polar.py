"""Tests for the polar-cap trap (SYI22 gamma-effect, NHGQ_polar.tex).

The trap enters the solver in exactly one place: the advected PV argument
becomes q' + eta(x,y). These tests pin down (i) the eta field itself,
(ii) the added tendency against the independently validated single-level
Jacobian, (iii) flux/jacobian path agreement, (iv) the resolution guard
and beta/gamma exclusivity, (v) that gamma=0 leaves everything untouched.
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import jax.numpy as jnp
import pytest

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.solver import (
    State, make_initial_state, imex_step, explicit_rhs, invert_psi,
    _to_nodal, _to_coeffs,
)
from nhqg.spectral import jacobian_dealiased


def _cfg(**kw):
    # Trap-geometry feasibility: the tanh transition must be wide enough for
    # the 2/3 band (w >~ 4 dx) AND decay before the periodic boundary
    # (L/2 - r_star >~ 7 w). At L=20 with the default r_star = 4.5 that
    # needs Nx >= 128; A_d = 6 sits comfortably inside both limits.
    # (Production 512^2, L=48Lc, A_d=20 passes with orders of margin.)
    base = dict(Nx=128, Nz=8, L=20.0, Ra_tilde=50.0, sigma=1.0,
                beta=0.0, Ld=float('inf'), dt=1e-4, float_dtype='float64',
                trap_sharpness=6.0)
    base.update(kw)
    return NHQGConfig(**base)


class TestTrapEta:

    def test_eta_absent_when_gamma_zero(self):
        g = make_grid(_cfg())
        assert g.eta_hat is None

    def test_eta_matches_analytic(self):
        cfg = _cfg(gamma=2e-3)
        g = make_grid(cfg)
        eta_phys = np.fft.irfft2(np.array(g.eta_hat), s=(cfg.Nx, cfg.Nx))

        xy = np.arange(cfg.Nx) * (cfg.L / cfg.Nx)
        X, Y = np.meshgrid(xy, xy, indexing='ij')
        r = np.sqrt((X - cfg.L / 2) ** 2 + (Y - cfg.L / 2) ** 2)
        r_star = 0.45 * cfg.L / 2
        sig = 0.5 * (1.0 - np.tanh(cfg.trap_sharpness * (r - r_star) / r_star))
        expected = -0.5 * cfg.gamma * r ** 2 * sig

        assert np.max(np.abs(eta_phys - expected)) < 1e-14 * np.max(np.abs(expected))

    def test_trap_r_star_override(self):
        g1 = make_grid(_cfg(gamma=2e-3))
        # smaller trap needs a gentler tanh to stay grid-resolved
        g2 = make_grid(_cfg(gamma=2e-3, trap_r_star=3.5, trap_sharpness=5.0))
        assert float(jnp.max(jnp.abs(g1.eta_hat - g2.eta_hat))) > 0

    def test_beta_gamma_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            make_grid(_cfg(gamma=1e-3, beta=0.1))

    def test_under_resolved_trap_rejected(self):
        with pytest.raises(ValueError, match="under-resolved"):
            make_grid(_cfg(Nx=16, gamma=1e-3, trap_sharpness=50.0))

    def test_eta_masked_under_23_rule(self):
        g = make_grid(_cfg(gamma=2e-3, horizontal_dealiasing='23_rule'))
        out_of_band = np.abs(np.array(g.eta_hat)) * (1.0 - np.array(g.mask_23))
        assert float(out_of_band.max()) == 0.0


class TestTrapTendency:

    def _developed_state(self, g):
        state = make_initial_state(g, seed=5, amplitude=1e-3)
        for _ in range(3):
            state = imex_step(state, g)
        return state

    def test_added_tendency_is_jacobian_psi_eta(self):
        """RHS(gamma) - RHS(0) must equal -J(psi, eta), level by level,
        against the independently validated single-level Jacobian."""
        g0 = make_grid(_cfg())
        gg = make_grid(_cfg(gamma=2e-3))
        state = self._developed_state(g0)

        E0 = explicit_rhs(state, g0)
        Eg = explicit_rhs(state, gg)
        dq = Eg.q_hat - E0.q_hat

        psi_nodal = _to_nodal(invert_psi(state.q_hat, gg.inv_denom), gg.V)
        J_levels = jnp.stack([
            jacobian_dealiased(psi_nodal[j], gg.eta_hat, gg.kx, gg.ky,
                               gg.Nx, gg.Npad)
            for j in range(gg.Nz + 1)
        ])
        J_ref = _to_coeffs(J_levels, gg.V_inv)

        scale = float(jnp.max(jnp.abs(J_ref)))
        assert scale > 0
        assert float(jnp.max(jnp.abs(dq + J_ref))) < 1e-12 * scale

        # eta touches ONLY the q advection: w/theta/mean tendencies unchanged
        assert float(jnp.max(jnp.abs(Eg.w_hat - E0.w_hat))) == 0.0
        assert float(jnp.max(jnp.abs(Eg.th_hat - E0.th_hat))) == 0.0
        assert float(jnp.max(jnp.abs(Eg.th_bar - E0.th_bar))) == 0.0

    def test_flux_and_jacobian_paths_agree_on_eta(self):
        """div(u eta) == J(psi, eta) discretely: div u = 0 holds spectrally
        and the 3/2 pad makes quadratic products exact -- for Nyquist-free
        fields. (Npad = 3Nx/2 exactly leaves Nyquist self-aliasing, the known
        32_rule fine print, and it lands differently in the two forms, so the
        equivalence is stated on the Nyquist-free subspace.)"""

        def _zero_nyquist(f_hat):
            f = np.array(f_hat)
            f[..., f.shape[-2] // 2, :] = 0.0   # kx-Nyquist row
            f[..., :, -1] = 0.0                 # ky-Nyquist column
            return jnp.array(f)

        diffs = {}
        for adv in ("jacobian", "flux"):
            g0 = make_grid(_cfg(nonlinear_advection=adv))
            gg = make_grid(_cfg(nonlinear_advection=adv, gamma=2e-3))
            gg = gg._replace(eta_hat=_zero_nyquist(gg.eta_hat))
            s = self._developed_state(g0)
            state = State(_zero_nyquist(s.q_hat), _zero_nyquist(s.w_hat),
                          _zero_nyquist(s.th_hat), s.th_bar)
            diffs[adv] = (explicit_rhs(state, gg).q_hat
                          - explicit_rhs(state, g0).q_hat)
        scale_eta = float(jnp.max(jnp.abs(diffs["jacobian"])))
        assert scale_eta > 0
        assert float(jnp.max(jnp.abs(diffs["jacobian"] - diffs["flux"]))) \
            < 1e-12 * scale_eta

    def test_trap_run_finite_production_flavor(self):
        cfg = _cfg(gamma=2e-3, horizontal_dealiasing='23_rule',
                   nonlinear_advection='flux', thermal_closure='evolve_mean')
        g = make_grid(cfg)
        state = make_initial_state(g, seed=5, amplitude=1e-3)
        for _ in range(5):
            state = imex_step(state, g)
        for f in (state.q_hat, state.w_hat, state.th_hat, state.th_bar):
            assert bool(jnp.all(jnp.isfinite(f)))
        # state stays band-limited with the trap active
        out = np.abs(np.array(state.q_hat)) * (1.0 - np.array(g.mask_23))[None]
        assert float(out.max()) == 0.0
