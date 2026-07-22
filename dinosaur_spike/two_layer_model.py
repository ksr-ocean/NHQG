"""Masked two-layer QG RHS and explicit stepping on Dinosaur grids."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from dinosaur_spike.masks import southern_tukey_mask, sponge_rate_from_mask
from dinosaur_spike.two_layer_qg import (
    TwoLayerPsi,
    TwoLayerState,
    invert_streamfunction,
    pv_from_streamfunction,
    remove_mean_pv,
)


@dataclass(frozen=True)
class TwoLayerParams:
    """Parameters for the first masked two-layer QG prototype."""

    F1: float = 0.7
    F2: float = 0.4
    deformation_profile: str = "constant"
    deformation_reference_lat_deg: float = -60.0
    deformation_f_floor_sin: float = 0.2
    omega: float = 1.0
    sponge_max_rate: float = 1.0
    hyperdiffusion_rate: float = 0.0
    hyperdiffusion_order: int = 2
    background_barotropic_velocity: float = 0.0
    background_shear_velocity: float = 0.0
    background_profile: str = "solid_body"
    background_sin3_weight: float = 0.75
    mask_plateau_north_edge_deg: float = -30.0
    mask_taper_north_edge_deg: float = 5.0
    mask_nonlinear_tendency: bool = True


def coriolis_modal(grid, omega: float = 1.0) -> jax.Array:
    """Return modal Coriolis parameter f = 2 Omega sin(latitude)."""
    lat = jnp.asarray(grid.latitudes)
    nodal = 2.0 * omega * jnp.sin(lat)[None, :] * jnp.ones(
        (grid.nodal_shape[0], 1), dtype=lat.dtype
    )
    return grid.to_modal(nodal)


def deformation_coefficients_nodal(
    grid, params: TwoLayerParams
) -> tuple[jax.Array, jax.Array]:
    """Return nodal deformation coefficients with optional f regularization."""
    dtype = jnp.asarray(grid.latitudes).dtype
    if params.deformation_profile == "constant":
        ones = jnp.ones(grid.nodal_shape, dtype=dtype)
        return params.F1 * ones, params.F2 * ones

    sin_lat = jnp.sin(jnp.asarray(grid.latitudes))
    floor = jnp.asarray(params.deformation_f_floor_sin, dtype=dtype)
    if params.deformation_f_floor_sin <= 0.0:
        raise ValueError("deformation_f_floor_sin must be positive")
    sin2 = sin_lat * sin_lat + floor * floor
    ref_sin = jnp.sin(jnp.deg2rad(jnp.asarray(params.deformation_reference_lat_deg)))
    ref = ref_sin * ref_sin + floor * floor
    if params.deformation_profile == "f_squared_floor":
        factor_1d = sin2 / ref
    elif params.deformation_profile == "inverse_f_squared_floor":
        factor_1d = ref / sin2
    else:
        raise ValueError(f"unsupported deformation_profile {params.deformation_profile!r}")
    factor = jnp.ones((grid.nodal_shape[0], 1), dtype=dtype) * factor_1d[None, :]
    return params.F1 * factor, params.F2 * factor


def pv_from_streamfunction_model(
    psi: TwoLayerPsi,
    grid,
    params: TwoLayerParams,
) -> TwoLayerState:
    """Compute model PV from streamfunction, including regularized deformation."""
    if params.deformation_profile == "constant":
        return pv_from_streamfunction(psi, grid, F1=params.F1, F2=params.F2)

    lap1 = grid.laplacian(psi.psi1)
    lap2 = grid.laplacian(psi.psi2)
    F1_nodal, F2_nodal = deformation_coefficients_nodal(grid, params)
    psi_diff = grid.to_nodal(psi.psi2 - psi.psi1)
    q1 = lap1 + grid.to_modal(F1_nodal * psi_diff)
    q2 = lap2 - grid.to_modal(F2_nodal * psi_diff)
    return TwoLayerState(q1=q1, q2=q2)


def background_profile_nodal(grid, params: TwoLayerParams) -> jax.Array:
    """Return the scalar latitude profile used by the zonal base streamfunction."""
    lat = jnp.asarray(grid.latitudes)
    sin_lat = jnp.sin(lat)
    if params.background_profile == "solid_body":
        profile = sin_lat
    elif params.background_profile == "sin_plus_sin3":
        profile = sin_lat + params.background_sin3_weight * sin_lat**3
    else:
        raise ValueError(f"unsupported background_profile {params.background_profile!r}")
    return jnp.ones((grid.nodal_shape[0], 1), dtype=profile.dtype) * profile[None, :]


def background_streamfunction_modal(grid, params: TwoLayerParams) -> TwoLayerPsi:
    """Return zonal base streamfunctions with layer shear.

    The default base state is ``Psi_i^0 = -U_i sin(latitude)``, giving
    eastward velocity ``u_i = U_i cos(latitude)`` on the unit sphere. The
    ``sin_plus_sin3`` profile is also regular at the poles, but it is not a
    solid-body rotation and can support Phillips-type unstable modes.
    """
    profile = background_profile_nodal(grid, params)
    u1 = params.background_barotropic_velocity + 0.5 * params.background_shear_velocity
    u2 = params.background_barotropic_velocity - 0.5 * params.background_shear_velocity
    psi1 = grid.to_modal(-u1 * profile)
    psi2 = grid.to_modal(-u2 * profile)
    return TwoLayerPsi(psi1=psi1, psi2=psi2)


def background_pv_modal(grid, params: TwoLayerParams) -> TwoLayerState:
    """Return layer base PV including deformation coupling and Coriolis PV."""
    psi0 = background_streamfunction_modal(grid, params)
    q0 = pv_from_streamfunction_model(psi0, grid, params)
    f = coriolis_modal(grid, params.omega)
    return TwoLayerState(q1=q0.q1 + f, q2=q0.q2 + f)


def latitude_mask_nodal(grid, params: TwoLayerParams) -> jax.Array:
    """Return nodal latitude mask with shape `(longitude, latitude)`."""
    lat = jnp.asarray(grid.latitudes)
    mask_1d = southern_tukey_mask(
        lat,
        plateau_north_edge_deg=params.mask_plateau_north_edge_deg,
        taper_north_edge_deg=params.mask_taper_north_edge_deg,
    )
    return jnp.ones((grid.nodal_shape[0], 1), dtype=mask_1d.dtype) * mask_1d[None, :]


def _layer_flux_tendency(grid, psi_modal, q_total_modal, mask_nodal):
    vcos = grid.k_cross(grid.cos_lat_grad(psi_modal, clip=True))
    vcos_nodal = grid.to_nodal(jnp.stack(vcos))
    q_nodal = grid.to_nodal(q_total_modal)
    flux_nodal = vcos_nodal * q_nodal * grid.sec2_lat
    flux_modal = grid.to_modal(flux_nodal)
    tendency = -grid.div_cos_lat((flux_modal[0], flux_modal[1]), clip=True)

    if mask_nodal is not None:
        tendency = grid.to_modal(grid.to_nodal(tendency) * mask_nodal)
    return tendency


def _hyperdiffusion(grid, q_modal, params: TwoLayerParams):
    if params.hyperdiffusion_rate == 0.0:
        return jnp.zeros_like(q_modal)
    L = -jnp.asarray(grid.laplacian_eigenvalues, dtype=q_modal.dtype)
    scale = L ** params.hyperdiffusion_order
    return -params.hyperdiffusion_rate * scale * q_modal


def _hyperdiffusion_factor(grid, dtype, params: TwoLayerParams, dt: float):
    if params.hyperdiffusion_rate == 0.0:
        return jnp.asarray(1.0, dtype=dtype)
    L = -jnp.asarray(grid.laplacian_eigenvalues, dtype=dtype)
    scale = L ** params.hyperdiffusion_order
    return jnp.exp(-params.hyperdiffusion_rate * scale * dt)


def _multiply_state(state: TwoLayerState, factor) -> TwoLayerState:
    return TwoLayerState(q1=factor * state.q1, q2=factor * state.q2)


def _add_scaled_state(state: TwoLayerState, scale: float, tendency: TwoLayerState) -> TwoLayerState:
    return TwoLayerState(q1=state.q1 + scale * tendency.q1, q2=state.q2 + scale * tendency.q2)


def _advective_rhs(state: TwoLayerState, grid, params: TwoLayerParams) -> TwoLayerState:
    """Compute all explicit tendencies except diagonal modal hyperdiffusion."""
    if params.deformation_profile != "constant":
        raise NotImplementedError(
            "nonlinear stepping with latitude-dependent deformation needs a "
            "precomputed/iterative PV inversion; use linear_phillips_spectrum.py "
            "for exact low-resolution variable-coefficient spectra."
        )
    state = remove_mean_pv(state)
    psi = invert_streamfunction(state, grid, F1=params.F1, F2=params.F2)
    psi0 = background_streamfunction_modal(grid, params)
    q0 = background_pv_modal(grid, params)
    mask = latitude_mask_nodal(grid, params)
    mask_for_nonlinear = mask if params.mask_nonlinear_tendency else None

    t1 = _layer_flux_tendency(grid, psi.psi1, state.q1 + q0.q1, mask_for_nonlinear)
    t2 = _layer_flux_tendency(grid, psi.psi2, state.q2 + q0.q2, mask_for_nonlinear)
    t1 = t1 + _layer_flux_tendency(grid, psi0.psi1, state.q1, mask_for_nonlinear)
    t2 = t2 + _layer_flux_tendency(grid, psi0.psi2, state.q2, mask_for_nonlinear)

    sponge_nodal = sponge_rate_from_mask(mask, params.sponge_max_rate)
    t1 = t1 - grid.to_modal(sponge_nodal * grid.to_nodal(state.q1))
    t2 = t2 - grid.to_modal(sponge_nodal * grid.to_nodal(state.q2))

    return remove_mean_pv(
        TwoLayerState(q1=grid.clip_wavenumbers(t1), q2=grid.clip_wavenumbers(t2))
    )


def rhs(state: TwoLayerState, grid, params: TwoLayerParams) -> TwoLayerState:
    """Compute masked two-layer QG PV-anomaly tendency with explicit diffusion."""
    tendency = _advective_rhs(state, grid, params)
    t1 = tendency.q1
    t2 = tendency.q2
    t1 = t1 + _hyperdiffusion(grid, state.q1, params)
    t2 = t2 + _hyperdiffusion(grid, state.q2, params)
    return remove_mean_pv(
        TwoLayerState(q1=grid.clip_wavenumbers(t1), q2=grid.clip_wavenumbers(t2))
    )


def rk4_step(state: TwoLayerState, grid, params: TwoLayerParams, dt: float) -> TwoLayerState:
    """One explicit RK4 step for the masked two-layer QG prototype."""
    k1 = rhs(state, grid, params)
    k2 = rhs(
        TwoLayerState(q1=state.q1 + 0.5 * dt * k1.q1, q2=state.q2 + 0.5 * dt * k1.q2),
        grid,
        params,
    )
    k3 = rhs(
        TwoLayerState(q1=state.q1 + 0.5 * dt * k2.q1, q2=state.q2 + 0.5 * dt * k2.q2),
        grid,
        params,
    )
    k4 = rhs(
        TwoLayerState(q1=state.q1 + dt * k3.q1, q2=state.q2 + dt * k3.q2),
        grid,
        params,
    )
    out = TwoLayerState(
        q1=state.q1 + (dt / 6.0) * (k1.q1 + 2.0 * k2.q1 + 2.0 * k3.q1 + k4.q1),
        q2=state.q2 + (dt / 6.0) * (k1.q2 + 2.0 * k2.q2 + 2.0 * k3.q2 + k4.q2),
    )
    return remove_mean_pv(out)


def ifrk4_step(state: TwoLayerState, grid, params: TwoLayerParams, dt: float) -> TwoLayerState:
    """One Lawson integrating-factor RK4 step for diagonal hyperdiffusion."""
    state = remove_mean_pv(state)
    factor = _hyperdiffusion_factor(grid, state.q1.dtype, params, dt)
    half_factor = _hyperdiffusion_factor(grid, state.q1.dtype, params, 0.5 * dt)

    def decay(s: TwoLayerState) -> TwoLayerState:
        return _multiply_state(s, factor)

    def half_decay(s: TwoLayerState) -> TwoLayerState:
        return _multiply_state(s, half_factor)

    k1 = _advective_rhs(state, grid, params)
    y2 = half_decay(_add_scaled_state(state, 0.5 * dt, k1))
    k2 = _advective_rhs(y2, grid, params)
    y3 = _add_scaled_state(half_decay(state), 0.5 * dt, k2)
    k3 = _advective_rhs(y3, grid, params)
    y4 = _add_scaled_state(decay(state), dt, half_decay(k3))
    k4 = _advective_rhs(y4, grid, params)
    out = TwoLayerState(
        q1=decay(state).q1
        + (dt / 6.0)
        * (decay(k1).q1 + 2.0 * half_decay(k2).q1 + 2.0 * half_decay(k3).q1 + k4.q1),
        q2=decay(state).q2
        + (dt / 6.0)
        * (decay(k1).q2 + 2.0 * half_decay(k2).q2 + 2.0 * half_decay(k3).q2 + k4.q2),
    )
    return remove_mean_pv(out)


def windowed_enstrophy(state: TwoLayerState, grid, params: TwoLayerParams) -> jax.Array:
    """Simple mask-weighted layer-summed enstrophy diagnostic."""
    mask = latitude_mask_nodal(grid, params)
    q1 = grid.to_nodal(state.q1)
    q2 = grid.to_nodal(state.q2)
    return 0.5 * grid.integrate(mask * (q1 * q1 + q2 * q2))
