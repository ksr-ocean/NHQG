"""Tests for masks and the first masked two-layer QG model."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from dinosaur import spherical_harmonic

from dinosaur_spike.masks import southern_tukey_mask, sponge_rate_from_mask
from dinosaur_spike.run_two_layer_solution import _spectral_metrics_from_shell
from dinosaur_spike.two_layer_model import (
    TwoLayerParams,
    _hyperdiffusion_factor,
    _layer_flux_tendency,
    background_pv_modal,
    background_profile_nodal,
    background_streamfunction_modal,
    coriolis_modal,
    deformation_coefficients_nodal,
    ifrk4_step,
    latitude_mask_nodal,
    pv_from_streamfunction_model,
    rhs,
    rk4_step,
    windowed_enstrophy,
)
from dinosaur_spike.two_layer_qg import (
    TwoLayerPsi,
    TwoLayerState,
    invert_streamfunction,
    pv_from_streamfunction,
    remove_mean_pv,
)


def _grid():
    return spherical_harmonic.Grid.with_wavenumbers(
        longitude_wavenumbers=8,
        dealiasing="quadratic",
        spherical_harmonics_impl=spherical_harmonic.RealSphericalHarmonics,
        radius=1.0,
    )


def _state(grid):
    rng = np.random.default_rng(11)
    q1 = rng.normal(size=grid.modal_shape).astype(np.float64) * grid.mask
    q2 = rng.normal(size=grid.modal_shape).astype(np.float64) * grid.mask
    q1[0, 0] = 0.0
    q2[0, 0] = 0.0
    return TwoLayerState(q1=jnp.asarray(q1), q2=jnp.asarray(q2))


def test_southern_tukey_mask_bounds_and_regions():
    lat = jnp.deg2rad(jnp.linspace(-90.0, 30.0, 121))
    mask = southern_tukey_mask(lat, plateau_north_edge_deg=-30.0, taper_north_edge_deg=5.0)

    assert np.all(np.asarray(mask) >= 0.0)
    assert np.all(np.asarray(mask) <= 1.0)
    assert np.allclose(np.asarray(mask)[np.asarray(jnp.rad2deg(lat)) <= -30.0], 1.0)
    assert np.allclose(np.asarray(mask)[np.asarray(jnp.rad2deg(lat)) >= 5.0], 0.0)


def test_sponge_rate_from_mask():
    mask = jnp.asarray([1.0, 0.5, 0.0])
    sponge = sponge_rate_from_mask(mask, max_rate=2.0, power=2.0)

    assert np.allclose(np.asarray(sponge), [0.0, 0.5, 2.0])


def test_spectral_metrics_from_shell_tracks_resolved_edge_power():
    shell = np.zeros(20)
    shell[3] = 4.0
    shell[18] = 1.0

    metrics = _spectral_metrics_from_shell(shell)

    assert metrics["spectral_peak_l"] == 3.0
    assert metrics["spectral_mean_l"] > 3.0
    assert metrics["spectral_rms_l"] > metrics["spectral_mean_l"]
    assert np.isclose(metrics["spectral_top20_fraction"], 0.2)
    assert np.isclose(metrics["spectral_top10_fraction"], 0.2)


def test_coriolis_modal_roundtrip_and_mask_shape():
    grid = _grid()
    params = TwoLayerParams()
    f = coriolis_modal(grid, omega=1.7)
    f_nodal = grid.to_nodal(f)
    expected = 2.0 * 1.7 * np.sin(grid.latitudes)[None, :]
    mask = latitude_mask_nodal(grid, params)

    assert f.shape == grid.modal_shape
    assert mask.shape == grid.nodal_shape
    assert np.allclose(np.asarray(f_nodal), expected, atol=1e-11)
    assert np.all(np.asarray(mask) >= 0.0)
    assert np.all(np.asarray(mask) <= 1.0)


def test_background_streamfunction_matches_zonal_shear_profile():
    grid = _grid()
    params = TwoLayerParams(
        background_barotropic_velocity=0.1,
        background_shear_velocity=0.6,
    )

    psi0 = background_streamfunction_modal(grid, params)
    psi1 = grid.to_nodal(psi0.psi1)
    psi2 = grid.to_nodal(psi0.psi2)
    u1 = params.background_barotropic_velocity + 0.5 * params.background_shear_velocity
    u2 = params.background_barotropic_velocity - 0.5 * params.background_shear_velocity
    expected1 = -u1 * np.sin(grid.latitudes)[None, :]
    expected2 = -u2 * np.sin(grid.latitudes)[None, :]

    assert np.allclose(np.asarray(psi1), expected1, atol=1e-11)
    assert np.allclose(np.asarray(psi2), expected2, atol=1e-11)


def test_sin_plus_sin3_background_profile_matches_definition():
    grid = _grid()
    params = TwoLayerParams(
        background_profile="sin_plus_sin3",
        background_sin3_weight=0.25,
        background_shear_velocity=0.6,
    )

    profile = background_profile_nodal(grid, params)
    expected_profile = np.sin(grid.latitudes) + 0.25 * np.sin(grid.latitudes) ** 3
    psi0 = background_streamfunction_modal(grid, params)
    psi1 = grid.to_nodal(psi0.psi1)

    assert np.allclose(np.asarray(profile), expected_profile[None, :], atol=1e-11)
    assert np.allclose(np.asarray(psi1), -0.3 * expected_profile[None, :], atol=1e-11)


def test_regularized_deformation_coefficients_have_equatorial_floor():
    grid = _grid()
    params = TwoLayerParams(
        deformation_profile="f_squared_floor",
        deformation_reference_lat_deg=-60.0,
        deformation_f_floor_sin=0.25,
    )

    F1_nodal, F2_nodal = deformation_coefficients_nodal(grid, params)
    lat = np.asarray(grid.latitudes)
    factor = (np.sin(lat) ** 2 + 0.25**2) / (np.sin(np.deg2rad(-60.0)) ** 2 + 0.25**2)

    assert np.allclose(np.asarray(F1_nodal), params.F1 * factor[None, :], atol=1e-11)
    assert np.allclose(np.asarray(F2_nodal), params.F2 * factor[None, :], atol=1e-11)
    assert np.min(np.asarray(F1_nodal)) > 0.0


def test_constant_model_pv_matches_block_diagonal_pv():
    grid = _grid()
    rng = np.random.default_rng(21)
    psi = TwoLayerPsi(
        psi1=jnp.asarray(rng.normal(size=grid.modal_shape) * grid.mask),
        psi2=jnp.asarray(rng.normal(size=grid.modal_shape) * grid.mask),
    )
    params = TwoLayerParams()

    expected = pv_from_streamfunction(psi, grid, F1=params.F1, F2=params.F2)
    actual = pv_from_streamfunction_model(psi, grid, params)

    assert np.allclose(np.asarray(actual.q1), np.asarray(expected.q1), atol=1e-11)
    assert np.allclose(np.asarray(actual.q2), np.asarray(expected.q2), atol=1e-11)


def test_zero_background_pv_is_coriolis_pv():
    grid = _grid()
    params = TwoLayerParams()

    q0 = background_pv_modal(grid, params)
    f = coriolis_modal(grid, params.omega)

    assert np.allclose(np.asarray(q0.q1), np.asarray(f), atol=1e-11)
    assert np.allclose(np.asarray(q0.q2), np.asarray(f), atol=1e-11)


def test_zero_background_reproduces_original_rhs_form():
    grid = _grid()
    params = TwoLayerParams(sponge_max_rate=0.2)
    state = remove_mean_pv(_state(grid))
    mask = latitude_mask_nodal(grid, params)
    psi = invert_streamfunction(state, grid, F1=params.F1, F2=params.F2)
    f = coriolis_modal(grid, params.omega)
    expected1 = _layer_flux_tendency(grid, psi.psi1, state.q1 + f, mask)
    expected2 = _layer_flux_tendency(grid, psi.psi2, state.q2 + f, mask)
    sponge_nodal = sponge_rate_from_mask(mask, params.sponge_max_rate)
    expected1 = expected1 - grid.to_modal(sponge_nodal * grid.to_nodal(state.q1))
    expected2 = expected2 - grid.to_modal(sponge_nodal * grid.to_nodal(state.q2))
    expected = remove_mean_pv(
        TwoLayerState(
            q1=grid.clip_wavenumbers(expected1),
            q2=grid.clip_wavenumbers(expected2),
        )
    )

    actual = rhs(state, grid, params)

    assert np.allclose(np.asarray(actual.q1), np.asarray(expected.q1), atol=1e-11)
    assert np.allclose(np.asarray(actual.q2), np.asarray(expected.q2), atol=1e-11)


def test_layer_flux_tendency_does_not_advect_constant_scalar():
    import jax

    if not jax.config.read("jax_enable_x64"):
        pytest.skip("constant-advection roundoff check requires x64")
    grid = _grid()
    lon = jnp.asarray(grid.longitudes)[:, None]
    lat = jnp.asarray(grid.latitudes)[None, :]
    psi = grid.to_modal(jnp.cos(lon) * jnp.cos(lat))
    constant = spherical_harmonic.add_constant(jnp.zeros(grid.modal_shape), 1.0)

    tendency = _layer_flux_tendency(grid, psi, constant, None)

    assert float(jnp.max(jnp.abs(grid.to_nodal(tendency)))) < 1e-10


def test_background_shear_has_no_spontaneous_tendency_from_zero_perturbation():
    grid = _grid()
    params = TwoLayerParams(
        background_shear_velocity=0.6,
        sponge_max_rate=0.3,
        hyperdiffusion_rate=1e-5,
    )
    zero = jnp.zeros(grid.modal_shape)
    state = TwoLayerState(q1=zero, q2=zero)

    tendency = rhs(state, grid, params)

    assert np.allclose(np.asarray(tendency.q1), 0.0, atol=1e-12)
    assert np.allclose(np.asarray(tendency.q2), 0.0, atol=1e-12)


def test_background_shear_changes_non_zonal_perturbation_tendency():
    grid = _grid()
    state = _state(grid)
    base_params = TwoLayerParams(sponge_max_rate=0.0)
    shear_params = TwoLayerParams(sponge_max_rate=0.0, background_shear_velocity=0.7)

    base = rhs(state, grid, base_params)
    sheared = rhs(state, grid, shear_params)
    delta = np.linalg.norm(np.asarray(sheared.q1 - base.q1))
    delta += np.linalg.norm(np.asarray(sheared.q2 - base.q2))

    assert delta > 1e-8


def test_layer_flux_tendency_matches_dinosaur_vorticity_flux_form():
    grid = _grid()
    state = _state(grid)
    psi = state.q1
    zeta = grid.laplacian(psi)
    q_total = zeta + coriolis_modal(grid)

    actual = _layer_flux_tendency(grid, psi, q_total, mask_nodal=None)

    ucos_vcos = jnp.stack(
        spherical_harmonic.get_cos_lat_vector(
            zeta,
            jnp.zeros_like(zeta),
            grid,
            clip=True,
        )
    )
    nodal_flux = grid.to_nodal(ucos_vcos) * grid.to_nodal(q_total) * grid.sec2_lat
    flux_modal = grid.to_modal(nodal_flux)
    expected = -grid.div_cos_lat((flux_modal[0], flux_modal[1]), clip=True)

    assert np.allclose(np.asarray(actual), np.asarray(expected), atol=1e-11)


def test_hyperdiffusion_factor_matches_diagonal_laplacian_decay():
    grid = _grid()
    params = TwoLayerParams(hyperdiffusion_rate=0.2, hyperdiffusion_order=2)
    dt = 0.125

    factor = _hyperdiffusion_factor(grid, jnp.float64, params, dt)
    ell = np.arange(grid.modal_shape[1], dtype=np.float64)
    expected = np.exp(-params.hyperdiffusion_rate * (ell * (ell + 1.0)) ** 2 * dt)

    assert np.allclose(np.asarray(factor), expected, atol=1e-14)


def test_ifrk4_matches_explicit_rk4_without_hyperdiffusion():
    grid = _grid()
    params = TwoLayerParams(
        sponge_max_rate=0.2,
        hyperdiffusion_rate=0.0,
        background_shear_velocity=0.3,
    )
    state = _state(grid)

    explicit = rk4_step(state, grid, params, dt=1e-3)
    integrating_factor = ifrk4_step(state, grid, params, dt=1e-3)

    assert np.allclose(np.asarray(integrating_factor.q1), np.asarray(explicit.q1), atol=1e-12)
    assert np.allclose(np.asarray(integrating_factor.q2), np.asarray(explicit.q2), atol=1e-12)


def test_rhs_and_rk4_step_are_finite_and_preserve_shapes():
    grid = _grid()
    params = TwoLayerParams(
        sponge_max_rate=0.2,
        hyperdiffusion_rate=1e-5,
        background_shear_velocity=0.3,
    )
    state = _state(grid)

    tendency = rhs(state, grid, params)
    stepped = rk4_step(state, grid, params, dt=1e-3)
    ens = windowed_enstrophy(stepped, grid, params)

    assert tendency.q1.shape == grid.modal_shape
    assert tendency.q2.shape == grid.modal_shape
    assert stepped.q1.shape == grid.modal_shape
    assert stepped.q2.shape == grid.modal_shape
    assert np.all(np.isfinite(np.asarray(tendency.q1)))
    assert np.all(np.isfinite(np.asarray(tendency.q2)))
    assert np.all(np.isfinite(np.asarray(stepped.q1)))
    assert np.all(np.isfinite(np.asarray(stepped.q2)))
    assert float(stepped.q1[0, 0]) == 0.0
    assert float(stepped.q2[0, 0]) == 0.0
    assert float(ens) >= 0.0
