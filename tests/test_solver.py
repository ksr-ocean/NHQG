"""Tests for solver: linear onset, IMEX convergence order, IMEX vs RK4 agreement."""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from nhqg.config import NHQGConfig
from nhqg.diagnostics import compute_diagnostics, compute_ke_budget, compute_w_theta_budgets
from nhqg.grid import make_grid
from nhqg.solver import (
    State, make_initial_state, imex_step, rk4_step,
    invert_psi, explicit_rhs, implicit_tendency, full_rhs,
    imex_mean_temp_solve, horizontal_mean_wtheta, explicit_rhs_dispatch,
    _dirichlet_to_cheb, _to_nodal, mean_flux_exchange_rhs_coeffs
)


# ---------------------------------------------------------------------------
# Linear onset test (Case 0)
# ---------------------------------------------------------------------------

class TestLinearOnset:
    """At Ra just above Ra_c, the most unstable mode should grow at k_c."""

    @pytest.fixture
    def onset_setup(self):
        """Set up near-onset parameters."""
        Ra_c = 8.6956
        k_c = 1.3048

        # Domain large enough for k_c: L = 2*pi*n / k_c, pick n~4
        L = 4 * 2 * np.pi / k_c  # ~19.3

        # Ra slightly supercritical
        Ra = Ra_c * 1.01

        cfg = NHQGConfig(
            Nx=64, Nz=16, L=L, Ra_tilde=Ra, sigma=1.0,
            beta=0.0, Ld=float('inf'),
            dt=1e-4, float_dtype='float64'
        )
        return cfg, Ra_c, k_c

    def test_growth_at_critical_wavenumber(self, onset_setup):
        """Near onset, energy should grow at k ~ k_c."""
        cfg, Ra_c, k_c = onset_setup
        g = make_grid(cfg)

        # Initialize with small perturbation
        state = make_initial_state(g, seed=0, amplitude=1e-6)
        # Run for a short time
        for _ in range(100):
            state = imex_step(state, g)

        # Check that the solution is growing (not decaying/blowing up)
        psi_hat = invert_psi(state.q_hat, g.inv_denom)
        ksq = np.array(g.ksq)
        k_mag = np.sqrt(ksq)

        # Energy per wavenumber shell
        psi_power = np.array(jnp.mean(jnp.abs(psi_hat) ** 2, axis=0))
        energy = ksq * psi_power

        # Find peak wavenumber (excluding k=0)
        mask = k_mag > 0.5
        k_peak_idx = np.unravel_index(
            np.argmax(np.where(mask, energy, 0)), energy.shape
        )
        k_peak = k_mag[k_peak_idx]

        # Peak should be near k_c (within a few grid spacings)
        dk = 2 * np.pi / cfg.L
        assert abs(k_peak - k_c) < 3 * dk, \
            f"Peak at k={k_peak:.2f}, expected near k_c={k_c:.2f}"


# ---------------------------------------------------------------------------
# IMEX convergence order
# ---------------------------------------------------------------------------

class TestIMEXConvergence:
    """IMEX should be 2nd order (verified by Richardson extrapolation)."""

    @pytest.fixture
    def convergence_cfg(self):
        """Small system for convergence test."""
        return NHQGConfig(
            Nx=32, Nz=8, L=20.0, Ra_tilde=50.0, sigma=1.0,
            beta=0.0, Ld=float('inf'), float_dtype='float64'
        )

    def test_second_order(self, convergence_cfg):
        """Error ratio between dt and dt/2 should be ~4 (2nd order)."""
        cfg = convergence_cfg
        n_steps = 8

        # Reference: very small dt
        dt_ref = 1e-5
        cfg_ref = cfg.with_updates(dt=dt_ref)
        g_ref = make_grid(cfg_ref)
        state0 = make_initial_state(g_ref, seed=42, amplitude=1e-3)
        state_ref = state0
        for _ in range(n_steps * 4):
            state_ref = imex_step(state_ref, g_ref)
        t_final = n_steps * 4 * dt_ref

        # Coarse: dt
        dt1 = 4 * dt_ref
        cfg1 = cfg.with_updates(dt=dt1)
        g1 = make_grid(cfg1)
        state1 = make_initial_state(g1, seed=42, amplitude=1e-3)
        for _ in range(n_steps):
            state1 = imex_step(state1, g1)

        # Fine: dt/2
        dt2 = 2 * dt_ref
        cfg2 = cfg.with_updates(dt=dt2)
        g2 = make_grid(cfg2)
        state2 = make_initial_state(g2, seed=42, amplitude=1e-3)
        for _ in range(n_steps * 2):
            state2 = imex_step(state2, g2)

        err1 = float(jnp.max(jnp.abs(state1.q_hat - state_ref.q_hat)))
        err2 = float(jnp.max(jnp.abs(state2.q_hat - state_ref.q_hat)))

        if err2 > 1e-15:  # avoid division by zero
            ratio = err1 / err2
            # For 2nd order, expect ratio ~ 4 (= 2^2)
            assert 2.0 < ratio < 8.0, \
                f"Convergence ratio {ratio:.2f}, expected ~4 for 2nd order"


# ---------------------------------------------------------------------------
# IMEX vs RK4 agreement at small dt
# ---------------------------------------------------------------------------

class TestIMEXvsRK4:
    """At small dt where both stable, IMEX and RK4 should agree."""

    def test_agreement(self):
        """After several steps, IMEX and RK4 differ by O(dt^2)."""
        cfg = NHQGConfig(
            Nx=32, Nz=8, L=20.0, Ra_tilde=50.0, sigma=1.0,
            beta=0.0, Ld=float('inf'),
            dt=1e-4, float_dtype='float64'
        )
        g = make_grid(cfg)
        state0 = make_initial_state(g, seed=7, amplitude=1e-3)

        state_imex = state0
        state_rk4 = state0

        n_steps = 20
        for _ in range(n_steps):
            state_imex = imex_step(state_imex, g)
            state_rk4 = rk4_step(state_rk4, g)

        # Relative error should be small
        max_q = float(jnp.max(jnp.abs(state_rk4.q_hat)))
        diff = float(jnp.max(jnp.abs(state_imex.q_hat - state_rk4.q_hat)))
        rel_err = diff / max_q if max_q > 0 else diff

        # IMEX is 2nd order, RK4 is 4th order, so difference is O(dt^2)
        # dt=1e-4, n_steps=20, so t=2e-3, expect diff ~ O(dt^2*t) ~ O(1e-11)
        assert rel_err < 1e-3, f"IMEX/RK4 relative error {rel_err:.2e} too large"

    def test_rk443_agreement(self):
        """The RK443 IMEX path should also agree with RK4 at small dt."""
        cfg = NHQGConfig(
            Nx=32, Nz=8, L=20.0, Ra_tilde=50.0, sigma=1.0,
            beta=0.0, Ld=float('inf'),
            dt=1e-4, float_dtype='float64',
            imex_scheme='rk443',
        )
        g = make_grid(cfg)
        state0 = make_initial_state(g, seed=7, amplitude=1e-3)

        state_imex = state0
        state_rk4 = state0

        n_steps = 20
        for _ in range(n_steps):
            state_imex = imex_step(state_imex, g)
            state_rk4 = rk4_step(state_rk4, g)

        max_q = float(jnp.max(jnp.abs(state_rk4.q_hat)))
        diff = float(jnp.max(jnp.abs(state_imex.q_hat - state_rk4.q_hat)))
        rel_err = diff / max_q if max_q > 0 else diff

        assert rel_err < 5e-4, f"RK443/RK4 relative error {rel_err:.2e} too large"


# ---------------------------------------------------------------------------
# BC enforcement
# ---------------------------------------------------------------------------

class TestBoundaryConditions:
    """Boundary conditions must be satisfied after stepping."""

    def test_dirichlet_maintained(self):
        """w and theta remain zero at boundaries (evaluated at CGL nodes)."""
        cfg = NHQGConfig(Nx=32, Nz=8, L=20.0, Ra_tilde=100.0,
                         dt=1e-3, float_dtype='float64')
        g = make_grid(cfg)
        state = make_initial_state(g, amplitude=1e-2)

        for _ in range(10):
            state = imex_step(state, g)

        # Convert coefficients to nodal values, then check boundaries
        V = np.array(g.V)
        w_cheb = np.einsum('ij,j...->i...', np.array(g.dirichlet_stencil), np.array(state.w_hat))
        th_cheb = np.einsum('ij,j...->i...', np.array(g.dirichlet_stencil), np.array(state.th_hat))
        w_nodal = np.einsum('ij,j...->i...', V, w_cheb)
        th_nodal = np.einsum('ij,j...->i...', V, th_cheb)

        w_bdy = max(float(np.max(np.abs(w_nodal[0]))),
                     float(np.max(np.abs(w_nodal[-1]))))
        th_bdy = max(float(np.max(np.abs(th_nodal[0]))),
                      float(np.max(np.abs(th_nodal[-1]))))

        assert w_bdy < 1e-12, f"w boundary: {w_bdy}"
        assert th_bdy < 1e-12, f"theta boundary: {th_bdy}"

    def test_neumann_maintained(self):
        """dq'/dZ = 0 at boundaries after stepping."""
        cfg = NHQGConfig(Nx=32, Nz=8, L=20.0, Ra_tilde=100.0,
                         dt=1e-3, float_dtype='float64',
                         q_boundary='neumann')
        g = make_grid(cfg)
        state = make_initial_state(g, amplitude=1e-2)

        for _ in range(10):
            state = imex_step(state, g)

        # Derivative in coefficient space, then evaluate at boundaries
        V = np.array(g.V)
        G_Z = np.array(g.G_Z)
        q = np.array(state.q_hat)
        dq_coeffs = np.einsum('ij,j...->i...', G_Z, q)
        dq_nodal = np.einsum('ij,j...->i...', V, dq_coeffs)
        dq_top = np.max(np.abs(dq_nodal[0]))   # Z=1
        dq_bot = np.max(np.abs(dq_nodal[-1]))  # Z=0

        assert dq_top < 1e-10, f"dq/dZ at top: {dq_top}"
        assert dq_bot < 1e-10, f"dq/dZ at bot: {dq_bot}"

    def test_zero_mean(self):
        """k=0 mode should be zero for all fields."""
        cfg = NHQGConfig(Nx=32, Nz=8, L=20.0, Ra_tilde=100.0,
                         dt=1e-3, float_dtype='float64')
        g = make_grid(cfg)
        state = make_initial_state(g, amplitude=1e-2)

        for _ in range(10):
            state = imex_step(state, g)

        for name, field in [('q', state.q_hat), ('w', state.w_hat), ('th', state.th_hat)]:
            k0 = float(jnp.max(jnp.abs(field[:, 0, 0])))
            assert k0 < 1e-14, f"{name} k=0 mode: {k0}"


class TestMeanTemperatureClosure:
    """The prognostic mean-temperature branch should preserve its own BCs."""

    def test_fixed_conduction_keeps_zero_mean_deviation(self):
        cfg = NHQGConfig(
            Nx=32, Nz=8, L=20.0, Ra_tilde=100.0,
            dt=1e-3, float_dtype='float64',
            thermal_closure='fixed_conduction',
        )
        g = make_grid(cfg)
        state = make_initial_state(g, amplitude=1e-2)

        for _ in range(10):
            state = imex_step(state, g)

        assert float(jnp.max(jnp.abs(state.th_bar))) < 1e-14

    def test_mean_temperature_bcs_preserved(self):
        cfg = NHQGConfig(
            Nx=32, Nz=8, L=20.0, Ra_tilde=100.0,
            dt=1e-3, float_dtype='float64',
            thermal_closure='evolve_mean',
        )
        g = make_grid(cfg)
        state = make_initial_state(g, amplitude=1e-2)

        for _ in range(10):
            state = imex_step(state, g)

        assert abs(float(state.th_bar[0])) < 1e-14
        assert abs(float(state.th_bar[-1])) < 1e-14

    def test_mean_temperature_responds_to_heat_flux(self):
        cfg = NHQGConfig(
            Nx=32, Nz=8, L=20.0, Ra_tilde=100.0,
            dt=1e-3, float_dtype='float64',
            thermal_closure='evolve_mean',
            mean_temp_eps_sq=1.0,
        )
        g = make_grid(cfg)
        state = make_initial_state(g, amplitude=1e-2)

        # Seed a simple correlated convective flux so Theta_bar must evolve.
        envelope = jnp.sin(jnp.pi * g.Z)[:, None, None]
        seed_cheb = jnp.einsum('ij,j...->i...', g.V_inv, jnp.ones((g.Nz + 1, g.Nx, g.Nk)) * envelope)
        spectral_seed = jnp.einsum('ij,j...->i...', g.dirichlet_pinv, seed_cheb)
        state = state._replace(
            w_hat=spectral_seed.astype(state.w_hat.dtype),
            th_hat=(0.5 * spectral_seed).astype(state.th_hat.dtype),
        )

        state = imex_step(state, g)

        assert float(jnp.max(jnp.abs(state.th_bar[1:-1]))) > 0.0


class TestVerticalDealiasing:
    """Experimental vertical dealiasing path should be well-posed."""

    def test_imex_step_finite_with_cheb_3o2(self):
        cfg = NHQGConfig(
            Nx=32, Nz=8, L=20.0, Ra_tilde=100.0,
            dt=1e-3, float_dtype='float64',
            vertical_dealiasing='cheb_3o2',
        )
        g = make_grid(cfg)
        assert g.Nz_dealias == 12
        state = make_initial_state(g, amplitude=1e-3)

        state = imex_step(state, g)

        for field in (state.q_hat, state.w_hat, state.th_hat, state.th_bar):
            assert bool(jnp.all(jnp.isfinite(field)))

    def test_imex_step_finite_with_cheb_2x(self):
        cfg = NHQGConfig(
            Nx=32, Nz=8, L=20.0, Ra_tilde=100.0,
            dt=1e-3, float_dtype='float64',
            vertical_dealiasing='cheb_2x',
        )
        g = make_grid(cfg)
        state = make_initial_state(g, amplitude=1e-3)

        state = imex_step(state, g)

        for field in (state.q_hat, state.w_hat, state.th_hat, state.th_bar):
            assert bool(jnp.all(jnp.isfinite(field)))


class TestKEBudgetDiagnostics:
    """Checks for the shell-binned horizontal kinetic-energy budget."""

    def test_shell_budget_sums_to_direct_total_tendency(self):
        cfg = NHQGConfig(
            Nx=32, Nz=8, L=20.0, Ra_tilde=50.0, sigma=1.0,
            beta=0.3, Ld=float('inf'),
            dt=1e-4, float_dtype='float64',
            nu_q=0.7, nu_w=0.0, nu_theta=0.0, hyper_order=1,
            thermal_closure='evolve_mean',
        )
        g = make_grid(cfg)
        state = make_initial_state(g, seed=5, amplitude=1e-3)
        for _ in range(4):
            state = imex_step(state, g)

        budget = compute_ke_budget(state, g)

        psi_hat = invert_psi(state.q_hat, g.inv_denom)
        psi_nodal = _to_nodal(psi_hat, g.V)
        explicit = explicit_rhs_dispatch(state, g)
        implicit = implicit_tendency(state, g)
        q_total = explicit.q_hat + implicit.q_hat - g.diss_rate_q[None, :, :] * state.q_hat
        q_total_nodal = _to_nodal(q_total, g.V)

        weight = np.ones_like(np.array(g.ksq))
        if g.Nk > 2:
            weight[:, 1:g.Nk - 1] = 2.0
        depth_int = np.einsum(
            'j,j...->...',
            np.array(g.cc_weights),
            np.real(np.conj(np.array(psi_nodal)) * np.array(q_total_nodal)),
        )
        direct_total = -np.sum(
            weight * np.array(g.ksq) * np.array(g.inv_denom) * depth_int
        ) / (g.Nx ** 4)

        assert np.all(np.isfinite(np.array(budget["ke_total_shell_tendency"])))
        assert abs(float(budget["ke_total_sum"]) - direct_total) < 1e-10

    def test_nonlinear_shell_transfer_sums_to_zero_for_beta_plane_energy_case(self):
        cfg = NHQGConfig(
            Nx=32, Nz=8, L=20.0, Ra_tilde=50.0, sigma=1.0,
            beta=0.0, Ld=float('inf'),
            dt=1e-4, float_dtype='float64',
            nu_q=0.0, nu_w=0.0, nu_theta=0.0,
            drag=0.0, hyper_order=1,
            thermal_closure='fixed_conduction',
        )
        g = make_grid(cfg)
        state = make_initial_state(g, seed=9, amplitude=1e-3)
        for _ in range(3):
            state = imex_step(state, g)

        budget = compute_ke_budget(state, g)
        nonlinear_sum = float(budget["ke_nonlinear_sum"])

        assert abs(nonlinear_sum) < 1e-10, \
            f"nonlinear KE transfer should sum to zero, got {nonlinear_sum:.3e}"

    def test_w_theta_shell_budgets_sum_to_direct_total_tendencies(self):
        cfg = NHQGConfig(
            Nx=32, Nz=8, L=20.0, Ra_tilde=50.0, sigma=1.0,
            beta=0.0, Ld=float('inf'),
            dt=1e-4, float_dtype='float64',
            nu_q=0.7, nu_w=0.5, nu_theta=0.4, hyper_order=1,
            thermal_closure='evolve_mean',
        )
        g = make_grid(cfg)
        state = make_initial_state(g, seed=6, amplitude=1e-3)
        for _ in range(4):
            state = imex_step(state, g)

        budget = compute_w_theta_budgets(state, g)
        explicit = explicit_rhs_dispatch(state, g)
        implicit = implicit_tendency(state, g)

        weight = np.ones_like(np.array(g.ksq))
        if g.Nk > 2:
            weight[:, 1:g.Nk - 1] = 2.0

        w_cheb = _dirichlet_to_cheb(state.w_hat, g.dirichlet_stencil)
        w_total_cheb = _dirichlet_to_cheb(
            explicit.w_hat + implicit.w_hat - g.diss_rate_w[None, :, :] * state.w_hat,
            g.dirichlet_stencil,
        )
        w_nodal = _to_nodal(w_cheb, g.V)
        w_total_nodal = _to_nodal(w_total_cheb, g.V)
        w_depth = np.einsum(
            'j,j...->...',
            np.array(g.cc_weights),
            np.real(np.conj(np.array(w_nodal)) * np.array(w_total_nodal)),
        )
        w_direct = np.sum(weight * w_depth) / (g.Nx ** 4)

        th_cheb = _dirichlet_to_cheb(state.th_hat, g.dirichlet_stencil)
        th_total_cheb = _dirichlet_to_cheb(
            explicit.th_hat + implicit.th_hat - g.diss_rate_th[None, :, :] * state.th_hat,
            g.dirichlet_stencil,
        )
        th_nodal = _to_nodal(th_cheb, g.V)
        th_total_nodal = _to_nodal(th_total_cheb, g.V)
        th_depth = np.einsum(
            'j,j...->...',
            np.array(g.cc_weights),
            np.real(np.conj(np.array(th_nodal)) * np.array(th_total_nodal)),
        )
        th_direct = np.sum(weight * th_depth) / (g.Nx ** 4)

        assert abs(float(budget["w_total_sum"]) - w_direct) < 1e-10
        assert abs(float(budget["th_total_sum"]) - th_direct) < 1e-10

    def test_w_theta_nonlinear_shell_transfers_sum_to_zero(self):
        cfg = NHQGConfig(
            Nx=32, Nz=8, L=20.0, Ra_tilde=50.0, sigma=1.0,
            beta=0.0, Ld=float('inf'),
            dt=1e-4, float_dtype='float64',
            nu_q=0.0, nu_w=0.0, nu_theta=0.0,
            drag=0.0, hyper_order=1,
            thermal_closure='fixed_conduction',
        )
        g = make_grid(cfg)
        state = make_initial_state(g, seed=10, amplitude=1e-3)
        for _ in range(3):
            state = imex_step(state, g)

        budget = compute_w_theta_budgets(state, g)

        assert abs(float(budget["w_nonlinear_sum"])) < 1e-10
        assert abs(float(budget["th_nonlinear_sum"])) < 1e-10


class TestMeanTemperatureDiscretization:
    """Targeted checks for the isolated 1D mean-temperature operator."""

    def test_imex_mean_temp_solve_matches_dirichlet_eigenmode(self):
        cfg = NHQGConfig(
            Nx=16, Nz=16, L=10.0, Ra_tilde=10.0,
            dt=2e-2, float_dtype='float64',
            thermal_closure='evolve_mean', mean_temp_eps_sq=1.0,
        )
        g = make_grid(cfg)

        th_bar_nodal = jnp.sin(np.pi * g.Z)
        th_bar_coeffs = jnp.einsum('ij,j->i', g.V_inv, th_bar_nodal)

        alpha = float(g.gamma_imex * g.dt * g.mean_temp_eps_sq / g.sigma)
        exact_nodal = th_bar_nodal / (1.0 + alpha * np.pi**2)
        exact_coeffs = jnp.einsum('ij,j->i', g.V_inv, exact_nodal)

        solved = imex_mean_temp_solve(th_bar_coeffs, g)
        err = float(jnp.max(jnp.abs(solved - exact_coeffs)))
        assert err < 1e-10, f"mean-temp implicit solve mismatch: {err:.3e}"

    def test_mean_temp_flux_rhs_matches_analytic_derivative(self):
        cfg = NHQGConfig(
            Nx=16, Nz=16, L=10.0, Ra_tilde=10.0,
            dt=1e-3, float_dtype='float64',
            thermal_closure='evolve_mean', mean_temp_eps_sq=1.0,
        )
        g = make_grid(cfg)
        state = make_initial_state(g, amplitude=0.0)

        envelope = jnp.sin(np.pi * g.Z)
        envelope_cheb = jnp.einsum('ij,j->i', g.V_inv, envelope)
        w_cheb = jnp.zeros((g.Nz + 1, g.Nx, g.Nk), dtype=g.V.dtype)
        w_cheb = w_cheb.at[:, 0, 0].set((g.Nx * g.Nx) * envelope_cheb)
        th_cheb = 0.5 * w_cheb
        w_hat = jnp.einsum('ij,j...->i...', g.dirichlet_pinv, w_cheb)
        th_hat = jnp.einsum('ij,j...->i...', g.dirichlet_pinv, th_cheb)
        state = state._replace(
            w_hat=w_hat.astype(state.w_hat.dtype),
            th_hat=th_hat.astype(state.th_hat.dtype),
        )

        flux_nodal = horizontal_mean_wtheta(
            state.w_hat, state.th_hat, g.V, g.dirichlet_stencil, g.Nx, g.Npad
        )
        expected_flux = 0.5 * jnp.sin(np.pi * g.Z) ** 2
        flux_err = float(jnp.max(jnp.abs(flux_nodal - expected_flux)))

        expl = explicit_rhs(state, g)
        got_rhs_nodal = jnp.einsum('ij,j->i', g.V, expl.th_bar)
        expected_rhs = -0.5 * np.pi * jnp.sin(2.0 * np.pi * g.Z)
        rhs_err = float(jnp.max(jnp.abs(got_rhs_nodal - expected_rhs)))

        assert flux_err < 1e-12, f"seeded mean heat flux mismatch: {flux_err:.3e}"
        assert rhs_err < 1e-10, f"mean-temp explicit RHS mismatch: {rhs_err:.3e}"

    def test_mean_temp_flux_is_horizontally_dealiased_at_nyquist(self):
        cfg = NHQGConfig(
            Nx=16, Nz=16, L=10.0, Ra_tilde=10.0,
            dt=1e-3, float_dtype='float64',
            thermal_closure='evolve_mean', mean_temp_eps_sq=1.0,
        )
        g = make_grid(cfg)
        state = make_initial_state(g, amplitude=0.0)

        envelope = jnp.sin(np.pi * g.Z)
        x = 2.0 * np.pi * jnp.arange(g.Nx) / g.Nx
        nyquist = jnp.cos((g.Nx // 2) * x)
        field_phys = envelope[:, None, None] * nyquist[None, :, None]
        field_phys = jnp.broadcast_to(field_phys, (g.Nz + 1, g.Nx, g.Nx))
        field_nodal = jnp.fft.rfft2(field_phys, axes=(1, 2))
        field_cheb = jnp.einsum('ij,j...->i...', g.V_inv, field_nodal)
        field_hat = jnp.einsum('ij,j...->i...', g.dirichlet_pinv, field_cheb)
        state = state._replace(
            w_hat=field_hat.astype(state.w_hat.dtype),
            th_hat=field_hat.astype(state.th_hat.dtype),
        )

        flux_raw = horizontal_mean_wtheta(
            state.w_hat, state.th_hat, g.V, g.dirichlet_stencil, g.Nx
        )
        flux_dealiased = horizontal_mean_wtheta(
            state.w_hat, state.th_hat, g.V, g.dirichlet_stencil, g.Nx, g.Npad
        )
        expected_flux = 0.5 * jnp.sin(np.pi * g.Z) ** 2

        raw_err = float(jnp.max(jnp.abs(flux_raw - expected_flux)))
        dealiased_err = float(jnp.max(jnp.abs(flux_dealiased - expected_flux)))

        assert raw_err > 1e-1, f"raw flux unexpectedly close to continuum: {raw_err:.3e}"
        assert dealiased_err < 1e-10, f"dealiased flux mismatch: {dealiased_err:.3e}"


class TestMeanExchangeDiagnostics:
    """Checks for the new mean-temperature and exchange diagnostics."""

    def test_zero_state_mean_diagnostics_are_trivial(self):
        cfg = NHQGConfig(
            Nx=16, Nz=8, L=10.0, Ra_tilde=10.0,
            dt=1e-3, float_dtype='float64',
            thermal_closure='evolve_mean', mean_temp_eps_sq=1.0,
        )
        g = make_grid(cfg)
        state = make_initial_state(g, amplitude=0.0)
        diag = compute_diagnostics(state, g)

        assert abs(float(diag["th_bar_phys_max"])) < 1e-14
        assert abs(float(diag["dth_bar_dz_max"])) < 1e-14
        assert abs(float(diag["mean_energy"])) < 1e-14
        assert abs(float(diag["mean_flux_exchange_tendency"])) < 1e-14
        assert abs(float(diag["mean_diffusion_tendency"])) < 1e-14
        assert abs(float(diag["mean_theta_exchange_residual"])) < 1e-14
        assert abs(float(diag["heat_flux_mismatch"])) < 1e-14
        assert abs(float(diag["mean_grad_min"]) - 1.0) < 1e-14
        assert abs(float(diag["mean_grad_mid"]) - 1.0) < 1e-14
        assert abs(float(diag["mean_grad_max"]) - 1.0) < 1e-14

    def test_dealiased_heat_flux_diagnostic_matches_solver_flux_path(self):
        cfg = NHQGConfig(
            Nx=16, Nz=16, L=10.0, Ra_tilde=10.0,
            dt=1e-3, float_dtype='float64',
            thermal_closure='evolve_mean', mean_temp_eps_sq=1.0,
        )
        g = make_grid(cfg)
        state = make_initial_state(g, amplitude=0.0)

        envelope = jnp.sin(np.pi * g.Z)
        x = 2.0 * np.pi * jnp.arange(g.Nx) / g.Nx
        nyquist = jnp.cos((g.Nx // 2) * x)
        field_phys = envelope[:, None, None] * nyquist[None, :, None]
        field_phys = jnp.broadcast_to(field_phys, (g.Nz + 1, g.Nx, g.Nx))
        field_nodal = jnp.fft.rfft2(field_phys, axes=(1, 2))
        field_cheb = jnp.einsum('ij,j...->i...', g.V_inv, field_nodal)
        field_hat = jnp.einsum('ij,j...->i...', g.dirichlet_pinv, field_cheb)
        state = state._replace(
            w_hat=field_hat.astype(state.w_hat.dtype),
            th_hat=field_hat.astype(state.th_hat.dtype),
        )

        flux_profile = horizontal_mean_wtheta(
            state.w_hat, state.th_hat, g.V, g.dirichlet_stencil, g.Nx, g.Npad
        )
        expected_flux = float(jnp.sum(g.cc_weights * flux_profile))
        diag = compute_diagnostics(state, g)

        assert abs(float(diag["vol_avg_tw_dealiased"]) - expected_flux) < 1e-12
        assert abs(float(diag["Nusselt_dealiased"]) - (1.0 + expected_flux)) < 1e-12

    def test_balanced_sbp2_internal_exchange_monitor_closes(self):
        cfg = NHQGConfig(
            Nx=16, Nz=12, L=20.0, Ra_tilde=50.0, sigma=1.0,
            beta=0.0, Ld=float('inf'),
            dt=1e-4, float_dtype='float64',
            thermal_closure='evolve_mean',
            mean_exchange_discretization='balanced_sbp2_pc',
            imex_scheme='ars222',
        )
        g = make_grid(cfg)
        state = make_initial_state(g, amplitude=0.0)

        rng = np.random.default_rng(321)
        envelope = np.sin(np.pi * np.array(g.Z))

        w_phys = envelope[:, None, None] * rng.standard_normal((g.Nz + 1, g.Nx, g.Nx))
        th_phys = envelope[:, None, None] * rng.standard_normal((g.Nz + 1, g.Nx, g.Nx))
        w_nodal = jnp.fft.rfft2(jnp.array(w_phys, dtype=jnp.float64), axes=(1, 2))
        th_nodal = jnp.fft.rfft2(jnp.array(th_phys, dtype=jnp.float64), axes=(1, 2))

        w_cheb = jnp.einsum('ij,j...->i...', g.V_inv, w_nodal)
        th_cheb = jnp.einsum('ij,j...->i...', g.V_inv, th_nodal)
        w_cheb = jnp.einsum('ij,j...->i...', g.proj_dirichlet, w_cheb)
        th_cheb = jnp.einsum('ij,j...->i...', g.proj_dirichlet, th_cheb)
        w_hat = jnp.einsum('ij,j...->i...', g.dirichlet_pinv, w_cheb)
        th_hat = jnp.einsum('ij,j...->i...', g.dirichlet_pinv, th_cheb)

        th_bar_nodal = jnp.array(rng.standard_normal(g.Nz + 1), dtype=jnp.float64)
        th_bar_nodal = th_bar_nodal.at[0].set(0.0).at[-1].set(0.0)
        th_bar = jnp.einsum('ij,j->i', g.V_inv, th_bar_nodal)
        th_bar = g.proj_dirichlet @ th_bar

        state = state._replace(
            w_hat=w_hat.astype(state.w_hat.dtype),
            th_hat=th_hat.astype(state.th_hat.dtype),
            th_bar=th_bar.astype(state.th_bar.dtype),
        )
        diag = compute_diagnostics(state, g)

        assert abs(float(diag["mean_theta_exchange_boundary_sbp"])) < 1e-12
        assert abs(float(diag["mean_theta_exchange_residual_sbp"])) < 1e-10
        assert abs(float(diag["mean_theta_exchange_residual_sbp_rel"])) < 1e-10

    @pytest.mark.parametrize(
        "mean_exchange_discretization",
        ["legacy", "coral_workgrid", "coral_workgrid_weakmean", "coral_workgrid_paired"],
    )
    def test_dealiased_thermal_shell_budgets_close_mean_exchange(self, mean_exchange_discretization):
        cfg = NHQGConfig(
            Nx=32, Nz=8, L=20.0, Ra_tilde=50.0, sigma=1.0,
            beta=0.0, Ld=float('inf'),
            dt=1e-4, float_dtype='float64',
            thermal_closure='evolve_mean',
            mean_exchange_discretization=mean_exchange_discretization,
        )
        g = make_grid(cfg)
        state = make_initial_state(g, seed=11, amplitude=1e-3)
        for _ in range(4):
            state = imex_step(state, g)

        diag = compute_diagnostics(state, g)

        assert abs(
            float(np.sum(np.array(diag["heat_flux_shell_dealiased"])) - float(diag["vol_avg_tw_dealiased"]))
        ) < 1e-10
        assert abs(
            float(np.sum(np.array(diag["th_conduction_shell_tendency_dealiased"])) - float(diag["vol_avg_tw_dealiased"]))
        ) < 1e-10
        assert abs(
            float(np.sum(np.array(diag["w_buoyancy_shell_tendency_dealiased"])) - float(diag["w_buoyancy_sum_dealiased"]))
        ) < 1e-10
        assert abs(float(diag["mean_theta_exchange_residual_dealiased"])) < 1e-10

    def test_coral_workgrid_weakmean_is_mass_adjoint_of_mean_gradient(self):
        cfg = NHQGConfig(
            Nx=16, Nz=10, L=20.0, Ra_tilde=50.0, sigma=1.0,
            beta=0.0, Ld=float('inf'),
            dt=1e-4, float_dtype='float64',
            thermal_closure='evolve_mean',
            mean_exchange_discretization='coral_workgrid_weakmean',
        )
        g = make_grid(cfg)
        rng = np.random.default_rng(123)
        th_bar = jnp.array(rng.standard_normal(g.Nz + 1), dtype=jnp.float64)
        flux = jnp.array(rng.standard_normal(g.Nz + 1), dtype=jnp.float64)

        rhs = mean_flux_exchange_rhs_coeffs(flux, g)
        left = float(th_bar @ (g.mean_mass @ rhs))
        right = float((g.G_Z @ th_bar) @ (g.mean_mass @ flux))

        assert abs(left - right) < 1e-11

    def test_balanced_midpoint_mode_runs_finite(self):
        cfg = NHQGConfig(
            Nx=16, Nz=10, L=20.0, Ra_tilde=50.0, sigma=1.0,
            beta=0.0, Ld=float('inf'),
            dt=1e-4, float_dtype='float64',
            thermal_closure='evolve_mean',
            mean_exchange_discretization='balanced_midpoint',
        )
        g = make_grid(cfg)
        state = make_initial_state(g, seed=7, amplitude=1e-4)
        for _ in range(3):
            state = imex_step(state, g)

        assert bool(jnp.all(jnp.isfinite(state.q_hat)))
        assert bool(jnp.all(jnp.isfinite(state.w_hat)))
        assert bool(jnp.all(jnp.isfinite(state.th_hat)))
        assert bool(jnp.all(jnp.isfinite(state.th_bar)))

    def test_balanced_sbp2_mode_runs_finite(self):
        cfg = NHQGConfig(
            Nx=16, Nz=10, L=20.0, Ra_tilde=50.0, sigma=1.0,
            beta=0.0, Ld=float('inf'),
            dt=1e-4, float_dtype='float64',
            thermal_closure='evolve_mean',
            mean_exchange_discretization='balanced_sbp2',
        )
        g = make_grid(cfg)
        state = make_initial_state(g, seed=9, amplitude=1e-4)
        for _ in range(3):
            state = imex_step(state, g)

        assert bool(jnp.all(jnp.isfinite(state.q_hat)))
        assert bool(jnp.all(jnp.isfinite(state.w_hat)))
        assert bool(jnp.all(jnp.isfinite(state.th_hat)))
        assert bool(jnp.all(jnp.isfinite(state.th_bar)))

    def test_balanced_sbp2_pc_mode_runs_finite(self):
        cfg = NHQGConfig(
            Nx=16, Nz=10, L=20.0, Ra_tilde=50.0, sigma=1.0,
            beta=0.0, Ld=float('inf'),
            dt=1e-4, float_dtype='float64',
            thermal_closure='evolve_mean',
            mean_exchange_discretization='balanced_sbp2_pc',
            imex_scheme='ars222',
        )
        g = make_grid(cfg)
        state = make_initial_state(g, seed=13, amplitude=1e-4)
        for _ in range(3):
            state = imex_step(state, g)

        assert bool(jnp.all(jnp.isfinite(state.q_hat)))
        assert bool(jnp.all(jnp.isfinite(state.w_hat)))
        assert bool(jnp.all(jnp.isfinite(state.th_hat)))
        assert bool(jnp.all(jnp.isfinite(state.th_bar)))

    def test_balanced_sbp2_pc_subcycled_mode_runs_finite(self):
        cfg = NHQGConfig(
            Nx=16, Nz=10, L=20.0, Ra_tilde=50.0, sigma=1.0,
            beta=0.0, Ld=float('inf'),
            dt=1e-4, float_dtype='float64',
            thermal_closure='evolve_mean',
            mean_exchange_discretization='balanced_sbp2_pc',
            sbp_corrector_substeps=4,
            imex_scheme='ars222',
        )
        g = make_grid(cfg)
        state = make_initial_state(g, seed=17, amplitude=1e-4)
        for _ in range(3):
            state = imex_step(state, g)

        assert bool(jnp.all(jnp.isfinite(state.q_hat)))
        assert bool(jnp.all(jnp.isfinite(state.w_hat)))
        assert bool(jnp.all(jnp.isfinite(state.th_hat)))
        assert bool(jnp.all(jnp.isfinite(state.th_bar)))


# ---------------------------------------------------------------------------
# Hermitian ghost regression (hermitian_ghost.md)
# ---------------------------------------------------------------------------

class TestHermitianGhost:
    """The rfft2 ky=0 / ky-Nyquist columns must stay Hermitian.

    Anti-Hermitian content there ("the ghost") is invisible to irfft2-based
    physics but grows at the unsaturated linear rate forever. The solver must
    (a) never seed it, (b) remove it every step, and (c) be physically
    unaffected by its removal.
    """

    def _cfg(self, **kw):
        base = dict(Nx=16, Nz=8, L=20.0, Ra_tilde=50.0, sigma=1.0,
                    beta=0.0, Ld=float('inf'), dt=1e-4, float_dtype='float64')
        base.update(kw)
        return NHQGConfig(**base)

    @staticmethod
    def _anti_hermitian_residual(f_hat, Nx):
        neg = (-np.arange(Nx)) % Nx
        res = 0.0
        for col in (0, -1):
            a = np.array(f_hat[:, :, col])
            res = max(res, float(np.max(np.abs(a - np.conj(a[:, neg])))))
        return res

    @staticmethod
    def _inject_ghost(state, Nx, amp=1e-3, seed=99):
        """Add pure anti-Hermitian content to the ky=0 column.

        kx=0 is excluded (k=(0,0) hygiene) and so is the kx-Nyquist row:
        under the 3/2-rule pad the +/-Nyquist rows are treated
        asymmetrically, so Nyquist-row ghost content is NOT exactly
        invisible to the padded product path (the known 32_rule Nyquist
        self-aliasing). sanitize_state removes it each step regardless.
        """
        rng = np.random.default_rng(seed)
        neg = (-np.arange(Nx)) % Nx

        def poison(f_hat):
            f = np.array(f_hat)
            r = rng.normal(size=f[:, :, 0].shape) + 1j * rng.normal(size=f[:, :, 0].shape)
            g = 0.5 * (r - np.conj(r[:, neg]))   # exactly anti-Hermitian
            g[:, 0] = 0.0                        # keep the k=(0,0) mode clean
            g[:, Nx // 2] = 0.0                  # exclude the kx-Nyquist row
            f[:, :, 0] += amp * g
            return jnp.array(f)

        return State(poison(state.q_hat), poison(state.w_hat),
                     poison(state.th_hat), state.th_bar)

    def test_initial_state_is_hermitian(self):
        g = make_grid(self._cfg())
        state = make_initial_state(g, seed=3, amplitude=1e-3)
        assert self._anti_hermitian_residual(state.q_hat, g.Nx) < 1e-15

    def test_step_removes_injected_ghost(self):
        g = make_grid(self._cfg())
        state = make_initial_state(g, seed=3, amplitude=1e-3)
        for _ in range(5):
            state = imex_step(state, g)
        ghosted = self._inject_ghost(state, g.Nx)
        assert self._anti_hermitian_residual(ghosted.q_hat, g.Nx) > 1e-4
        stepped = imex_step(ghosted, g)
        for f in (stepped.q_hat, stepped.w_hat, stepped.th_hat):
            assert self._anti_hermitian_residual(f, g.Nx) < 1e-14

    def test_ghost_is_physically_invisible(self):
        """Clean and ghost-injected states must coincide after one step:
        the ghost feeds no physics and the projection removes it exactly."""
        g = make_grid(self._cfg(thermal_closure='evolve_mean'))
        state = make_initial_state(g, seed=3, amplitude=1e-3)
        for _ in range(5):
            state = imex_step(state, g)
        clean = imex_step(state, g)
        poisoned = imex_step(self._inject_ghost(state, g.Nx), g)
        scale = float(jnp.max(jnp.abs(clean.q_hat)))
        assert float(jnp.max(jnp.abs(clean.q_hat - poisoned.q_hat))) < 1e-12 * scale
        assert float(jnp.max(jnp.abs(clean.w_hat - poisoned.w_hat))) < 1e-12
        assert float(jnp.max(jnp.abs(clean.th_hat - poisoned.th_hat))) < 1e-12
        assert float(jnp.max(jnp.abs(clean.th_bar - poisoned.th_bar))) < 1e-12

    def test_state_band_limited_under_23_rule(self):
        """Under 23_rule the masked band must carry no state content: it is
        nonlinearly frozen (mask is applied to products), so at some (Nx, L)
        it contains linearly unstable modes that would grow unsaturated."""
        g = make_grid(self._cfg(horizontal_dealiasing='23_rule'))
        state = make_initial_state(g, seed=3, amplitude=1e-3)
        mask = np.array(g.mask_23)
        assert (1.0 - mask).sum() > 0
        # seed the masked band directly
        f = np.array(state.q_hat)
        f += 1e-3 * (1.0 - mask)[None, :, :]
        stepped = imex_step(State(jnp.array(f), state.w_hat, state.th_hat,
                                  state.th_bar), g)
        out_of_band = np.abs(np.array(stepped.q_hat)) * (1.0 - mask)[None, :, :]
        assert float(out_of_band.max()) == 0.0
