"""Tests for the trig-vertical benchmark operators."""

import os
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np

from trig_vertical_benchmark.config import TrigBenchmarkConfig
from trig_vertical_benchmark.operators import make_grid
from trig_vertical_benchmark.solver import (
    eval_cos,
    eval_sin,
    dpsi_dz_sin,
    d1_sin_to_work,
    make_initial_state,
    imex_step_ars222,
    project_cos,
    project_sin,
)


def test_trig_projection_roundtrip():
    cfg = TrigBenchmarkConfig(Nx=16, Nz=16, dt=1e-3, float_dtype="float64")
    g = make_grid(cfg)
    rng = np.random.default_rng(0)

    psi_hat = rng.standard_normal((g.Nc, 3, 2))
    w_hat = rng.standard_normal((g.Ns, 3, 2))

    psi_full = np.array(eval_cos(psi_hat, g.C_eval))
    w_full = np.array(eval_sin(w_hat, g.S_eval))

    psi_back = np.array(project_cos(psi_full, g.C_proj))
    w_back = np.array(project_sin(w_full, g.S_proj))

    np.testing.assert_allclose(psi_back, psi_hat, atol=1e-10)
    np.testing.assert_allclose(w_back, w_hat, atol=1e-10)


def test_trig_derivative_maps_match_analytic_modes():
    cfg = TrigBenchmarkConfig(Nx=16, Nz=16, dt=1e-3, float_dtype="float64")
    g = make_grid(cfg)
    z = np.array(g.z_work)

    psi_hat = np.zeros((g.Nc,), dtype=np.float64)
    psi_hat[3] = 1.0
    dpsi_num = np.array(eval_sin(dpsi_dz_sin(psi_hat, g), g.S_eval))
    dpsi_exact = -3.0 * np.pi * np.sin(3.0 * np.pi * z)
    np.testing.assert_allclose(dpsi_num, dpsi_exact, atol=1e-10)

    w_hat = np.zeros((g.Ns,), dtype=np.float64)
    w_hat[4] = 1.0
    dw_num = np.array(d1_sin_to_work(w_hat, g))
    dw_exact = 5.0 * np.pi * np.cos(5.0 * np.pi * z)
    np.testing.assert_allclose(dw_num, dw_exact, atol=1e-10)


def test_trig_laplacian_is_exact_on_sine_modes():
    cfg = TrigBenchmarkConfig(Nx=16, Nz=16, dt=1e-3, float_dtype="float64")
    g = make_grid(cfg)
    for n in [1, 4, 9]:
        th_bar = np.zeros((g.Ns,), dtype=np.float64)
        th_bar[n - 1] = 1.0
        d2_num = np.array(g.d2_sin_diag * th_bar)
        d2_exact = np.zeros_like(th_bar)
        d2_exact[n - 1] = -(n * np.pi) ** 2
        np.testing.assert_allclose(d2_num, d2_exact, atol=1e-12)


def test_trig_vertical_smoke_step_stays_finite():
    cfg = TrigBenchmarkConfig(
        Nx=16,
        Nz=16,
        dt=1e-4,
        thermal_closure="evolve_mean",
        float_dtype="float64",
    )
    g = make_grid(cfg)
    state = make_initial_state(g, seed=0, amplitude=1e-6)

    for _ in range(5):
        state = imex_step_ars222(state, g)

    for field in state:
        assert np.all(np.isfinite(np.array(field)))
