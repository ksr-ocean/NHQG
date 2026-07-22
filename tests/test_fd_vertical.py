"""Tests for the FD-in-z benchmark operators."""

import os
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np

from fd_vertical_benchmark.config import FDBenchmarkConfig
from fd_vertical_benchmark.operators import make_grid
from fd_vertical_benchmark.solver import imex_step_ars222, make_initial_state


def _first_derivative_error(scheme: str, Nz: int) -> float:
    cfg = FDBenchmarkConfig(
        Nx=16,
        Nz=Nz,
        dt=1e-3,
        vertical_derivative=scheme,
        float_dtype="float64",
    )
    g = make_grid(cfg)
    z = np.array(g.z_full)
    D1 = np.array(g.D1_full)

    f = np.sin(2.0 * np.pi * z) + 0.3 * np.cos(3.0 * np.pi * z)
    df_exact = 2.0 * np.pi * np.cos(2.0 * np.pi * z) - 0.9 * np.pi * np.sin(3.0 * np.pi * z)
    df_num = D1 @ f
    return float(np.sqrt(np.mean((df_num - df_exact) ** 2)))


def _second_derivative_error(d1_scheme: str, d2_scheme: str, Nz: int) -> float:
    cfg = FDBenchmarkConfig(
        Nx=16,
        Nz=Nz,
        dt=1e-3,
        vertical_derivative=d1_scheme,
        vertical_second_derivative=d2_scheme,
        float_dtype="float64",
    )
    g = make_grid(cfg)
    z = np.array(g.z_int)
    D2 = np.array(g.D2_dir)

    f = np.sin(np.pi * z)
    d2f_exact = -(np.pi ** 2) * np.sin(np.pi * z)
    d2f_num = D2 @ f
    return float(np.sqrt(np.mean((d2f_num - d2f_exact) ** 2)))


def test_compact4_converges_faster_than_centered2():
    err_c2_32 = _first_derivative_error("centered2", 32)
    err_c2_64 = _first_derivative_error("centered2", 64)
    err_c4_32 = _first_derivative_error("compact4", 32)
    err_c4_64 = _first_derivative_error("compact4", 64)

    ratio_c2 = err_c2_32 / err_c2_64
    ratio_c4 = err_c4_32 / err_c4_64

    assert ratio_c2 > 3.0, f"centered2 ratio too small: {ratio_c2:.3f}"
    assert ratio_c4 > 6.0, f"compact4 ratio too small: {ratio_c4:.3f}"
    assert err_c4_64 < err_c2_64, (
        f"compact4 should be more accurate than centered2 at Nz=64: "
        f"{err_c4_64:.3e} vs {err_c2_64:.3e}"
    )


def test_sbp42_satisfies_discrete_sbp_identity():
    cfg = FDBenchmarkConfig(
        Nx=16,
        Nz=32,
        dt=1e-3,
        vertical_derivative="sbp42",
        float_dtype="float64",
    )
    g = make_grid(cfg)
    H = np.diag(np.array(g.norm_weights))
    D1 = np.array(g.D1_full)
    B = np.zeros_like(D1)
    B[0, 0] = -1.0
    B[-1, -1] = 1.0
    np.testing.assert_allclose(H @ D1 + D1.T @ H, B, atol=1e-12)


def test_neumann_reconstruction_matches_selected_derivative():
    rng = np.random.default_rng(1234)

    for scheme in ["centered2", "compact4"]:
        for treatment in ["projected", "direct"]:
            cfg = FDBenchmarkConfig(
                Nx=16,
                Nz=16,
                dt=1e-3,
                vertical_derivative=scheme,
                psi_neumann_treatment=treatment,
                float_dtype="float64",
            )
            g = make_grid(cfg)
            psi_int = rng.standard_normal(g.Ni)
            psi_full = np.array(g.P_neu) @ psi_int
            dpsi = np.array(g.D1_full) @ psi_full

            np.testing.assert_allclose(psi_full[1:-1], psi_int, atol=1e-14)
            np.testing.assert_allclose(dpsi[0], 0.0, atol=1e-12)
            np.testing.assert_allclose(dpsi[-1], 0.0, atol=1e-12)


def test_compact4_direct_neumann_matches_projected():
    cfg_projected = FDBenchmarkConfig(
        Nx=16,
        Nz=32,
        dt=1e-3,
        vertical_derivative="compact4",
        psi_neumann_treatment="projected",
        float_dtype="float64",
    )
    cfg_direct = FDBenchmarkConfig(
        Nx=16,
        Nz=32,
        dt=1e-3,
        vertical_derivative="compact4",
        psi_neumann_treatment="direct",
        float_dtype="float64",
    )
    g_projected = make_grid(cfg_projected)
    g_direct = make_grid(cfg_direct)

    np.testing.assert_allclose(np.array(g_direct.P_neu), np.array(g_projected.P_neu), atol=1e-12)
    np.testing.assert_allclose(np.array(g_direct.D1_psi), np.array(g_projected.D1_psi), atol=1e-12)


def test_compact4_raw_second_derivative_beats_centered2():
    err_c2 = _second_derivative_error("centered2", "centered2", 64)
    err_c4_raw = _second_derivative_error("compact4", "compact4_raw", 64)
    assert err_c4_raw < err_c2, (
        f"compact4_raw should beat centered2 on sin(pi z): "
        f"{err_c4_raw:.3e} vs {err_c2:.3e}"
    )


def test_compact4_branches_are_H_dissipative():
    for d1_scheme, d2_scheme in [
        ("compact4", "compact4_raw"),
        ("compact4", "from_d1_energy"),
        ("sbp42", "sbp42_energy"),
    ]:
        cfg = FDBenchmarkConfig(
            Nx=16,
            Nz=32,
            dt=1e-3,
            vertical_derivative=d1_scheme,
            vertical_second_derivative=d2_scheme,
            float_dtype="float64",
        )
        g = make_grid(cfg)
        H_int = np.diag(np.array(g.norm_weights)[1:-1])
        D2 = np.array(g.D2_dir)
        sym = 0.5 * (H_int @ D2 + D2.T @ H_int)
        max_eig = np.max(np.linalg.eigvalsh(sym))
        assert max_eig < 1e-10, f"{d2_scheme}: max eig of sym(H D2) = {max_eig:.3e}"


def test_fd_smoke_step_stays_finite():
    for d1_scheme, d2_scheme, treatment in [
        ("compact4", "centered2", "projected"),
        ("compact4", "compact4_raw", "projected"),
        ("compact4", "compact4_raw", "direct"),
        ("compact4", "from_d1_energy", "projected"),
        ("sbp42", "centered2", "direct"),
        ("sbp42", "sbp42_energy", "direct"),
    ]:
        cfg = FDBenchmarkConfig(
            Nx=16,
            Nz=16,
            dt=1e-4,
            thermal_closure="evolve_mean",
            vertical_derivative=d1_scheme,
            vertical_second_derivative=d2_scheme,
            psi_neumann_treatment=treatment,
            float_dtype="float64",
        )
        g = make_grid(cfg)
        state = make_initial_state(g, seed=0, amplitude=1e-6)

        for _ in range(5):
            state = imex_step_ars222(state, g)

        for field in state:
            assert np.all(np.isfinite(np.array(field)))
