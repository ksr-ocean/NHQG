"""Linear-onset EVP validation + the mixed-BC growth-rate gate.

Three layers:
1. The EVP itself is validated against the closed-form both-Dirichlet
   dispersion sigma(k) = -nu k^2 + sqrt(Ra - n^2 pi^2 / k^2) (n=1) and the
   known onset Ra_c(k_c) = 8.6956 at k_c = 1.3048.
2. The mixed-BC (open-top) spectrum is finite, real on the convective
   branch, and measurably different from rigid-lid.
3. GATE: the time stepper's measured growth rate matches the EVP of the
   same spatial discretization to <1% at three wavenumbers under mixed BCs
   (initialized on the dominant eigenvector so there is no transient).
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import jax.numpy as jnp
import pytest

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.linear_onset import linear_operator, max_growth_rate, critical_rayleigh
from nhqg.solver import State, imex_step


def _cfg(**kw):
    base = dict(Nx=32, Nz=16, L=20.0, Ra_tilde=100.0, sigma=1.0,
                beta=0.0, Ld=float('inf'), dt=1e-3, float_dtype='float64',
                nu_q=1.0, nu_w=1.0, nu_theta=1.0, hyper_order=1,
                thermal_closure='fixed_conduction')
    base.update(kw)
    return NHQGConfig(**base)


class TestEVPAgainstAnalytics:

    def test_both_dirichlet_dispersion(self):
        cfg = _cfg(Nz=32)
        g = make_grid(cfg)
        for k in (0.8, 1.3048, 2.5):
            sig_evp = max_growth_rate(cfg, g, k)
            sig_ref = -1.0 * k ** 2 + np.sqrt(100.0 - np.pi ** 2 / k ** 2)
            assert abs(sig_evp - sig_ref) < 1e-8 * abs(sig_ref)

    def test_both_dirichlet_critical_rayleigh(self):
        cfg = _cfg(Nz=32)
        g = make_grid(cfg)
        ra_c = critical_rayleigh(cfg, g, 1.3048)
        assert abs(ra_c - 8.6956) < 1e-2

    def test_mixed_spectrum_differs_from_rigid_lid(self):
        cfg_d = _cfg(Nz=32)
        cfg_n = _cfg(Nz=32, w_bc_top='neumann')
        g_d, g_n = make_grid(cfg_d), make_grid(cfg_n)
        k = 1.3048
        s_d = max_growth_rate(cfg_d, g_d, k)
        s_n = max_growth_rate(cfg_n, g_n, k)
        assert np.isfinite(s_n)
        assert abs(s_n - s_d) > 1e-3   # the open top genuinely changes onset

    def test_mixed_nz_convergence(self):
        """The mixed-BC growth rate is Nz-converged (spectral accuracy)."""
        k = 1.3048
        rates = []
        for Nz in (16, 32, 64):
            cfg = _cfg(Nz=Nz, w_bc_top='neumann')
            rates.append(max_growth_rate(cfg, make_grid(cfg), k))
        assert abs(rates[1] - rates[2]) < 1e-9 * abs(rates[2])
        assert abs(rates[0] - rates[2]) < 1e-6 * abs(rates[2])


class TestStepperMatchesEVP:

    @pytest.mark.parametrize("m", [3, 4, 6])
    def test_mixed_bc_growth_gate(self, m):
        """Measured IMEX growth at k = m*2pi/L within 1% of the EVP."""
        cfg = _cfg(w_bc_top='neumann')
        g = make_grid(cfg)
        k = m * 2.0 * np.pi / cfg.L

        A = linear_operator(cfg, g, k)
        lam, vecs = np.linalg.eig(A)
        i = int(np.argmax(lam.real))
        sig_evp = float(lam[i].real)
        assert abs(lam[i].imag) < 1e-10 * abs(sig_evp)   # direct (non-osc.) branch
        v = np.real(vecs[:, i])

        N = g.Nz
        nq, ng = N + 1, N - 1
        amp = 1e-8
        q = np.zeros((nq, g.Nx, g.Nk), dtype=complex)
        w = np.zeros((ng, g.Nx, g.Nk), dtype=complex)
        th = np.zeros((ng, g.Nx, g.Nk), dtype=complex)
        q[:, 0, m] = amp * v[:nq]
        w[:, 0, m] = amp * v[nq:nq + ng]
        th[:, 0, m] = amp * v[nq + ng:]
        state = State(jnp.asarray(q), jnp.asarray(w), jnp.asarray(th),
                      jnp.zeros(nq))

        def amplitude(s):
            return float(jnp.linalg.norm(s.q_hat[:, 0, m]))

        n1, n2 = 50, 150
        for _ in range(n1):
            state = imex_step(state, g)
        a1 = amplitude(state)
        for _ in range(n2 - n1):
            state = imex_step(state, g)
        a2 = amplitude(state)

        sig_meas = np.log(a2 / a1) / ((n2 - n1) * cfg.dt)
        assert abs(sig_meas - sig_evp) < 0.01 * abs(sig_evp), \
            f"k={k:.3f}: measured {sig_meas:.6f} vs EVP {sig_evp:.6f}"
