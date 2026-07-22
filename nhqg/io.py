"""NetCDF snapshot output and checkpoint save/load.

Uses netCDF4 directly (not xarray) to avoid pandas/numexpr compatibility issues.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import netCDF4

from nhqg.config import NHQGConfig
from nhqg.grid import Grid
from nhqg.solver import (
    State, invert_psi, mean_temperature_total, _dirichlet_to_cheb, _to_nodal, _to_nodal_1d
)


def _to_physical(field_coeffs: jnp.ndarray, V: jnp.ndarray, Nx: int) -> np.ndarray:
    """Convert coefficient-space field to physical (Nz+1, Nx, Nx).

    Steps: Chebyshev coeffs -> CGL nodal values -> irfft2 -> physical.
    """
    field_nodal = _to_nodal(field_coeffs, V)
    return np.array(jnp.fft.irfft2(field_nodal, s=(Nx, Nx)))


def save_snapshot(state: State, t: float, step: int,
                  cfg: NHQGConfig, grid: Grid, output_dir: str = None):
    """Save one snapshot as a NetCDF file via netCDF4.

    Physical-space fields: q', w, theta, psi plus mean temperature profiles.
    Dimensions: (z, y, x).
    """
    if output_dir is None:
        output_dir = cfg.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    Nx, Nz = cfg.Nx, cfg.Nz
    L = cfg.L

    # Physical-space fields (coefficient space -> nodal -> physical)
    q_phys = _to_physical(state.q_hat, grid.V, Nx)
    w_cheb = _dirichlet_to_cheb(state.w_hat, grid.dirichlet_stencil)
    th_cheb = _dirichlet_to_cheb(state.th_hat, grid.dirichlet_stencil)
    w_phys = _to_physical(w_cheb, grid.V, Nx)
    th_phys = _to_physical(th_cheb, grid.V, Nx)
    psi_hat = invert_psi(state.q_hat, grid.inv_denom)
    psi_phys = _to_physical(psi_hat, grid.V, Nx)
    th_mean = np.array(_to_nodal_1d(state.th_bar, grid.V))
    th_mean_total = np.array(mean_temperature_total(state.th_bar, grid.Z, grid.V))

    # Coordinates
    x = np.linspace(0, L, Nx, endpoint=False)
    z = np.array(grid.Z)

    fname = os.path.join(output_dir, f'snapshot_{step:08d}.nc')
    with netCDF4.Dataset(fname, 'w', format='NETCDF4') as ds:
        # Dimensions
        ds.createDimension('z', Nz + 1)
        ds.createDimension('y', Nx)
        ds.createDimension('x', Nx)

        # Coordinates
        v_x = ds.createVariable('x', 'f8', ('x',))
        v_x[:] = x
        v_x.units = 'Lc'

        v_y = ds.createVariable('y', 'f8', ('y',))
        v_y[:] = x
        v_y.units = 'Lc'

        v_z = ds.createVariable('z', 'f8', ('z',))
        v_z[:] = z
        v_z.long_name = 'depth (CGL points)'

        # Fields (float32 to save space — physical fields don't need float64)
        for name, data in [('q_prime', q_phys), ('w', w_phys),
                            ('theta', th_phys), ('psi', psi_phys)]:
            v = ds.createVariable(name, 'f4', ('z', 'y', 'x'),
                                   zlib=True, complevel=1)
            v[:] = data.astype(np.float32)

        for name, data in [('theta_bar', th_mean), ('theta_mean_total', th_mean_total)]:
            v = ds.createVariable(name, 'f4', ('z',), zlib=True, complevel=1)
            v[:] = data.astype(np.float32)

        # Global attributes
        ds.time = t
        ds.step = step
        ds.Ra_tilde = cfg.Ra_tilde
        ds.sigma = cfg.sigma
        ds.beta = cfg.beta
        ds.Ld = cfg.Ld if cfg.Ld != float('inf') else -1.0
        ds.L = cfg.L
        ds.thermal_closure = cfg.thermal_closure
        ds.Nx = cfg.Nx
        ds.Nz = cfg.Nz
        ds.dt = cfg.dt

    return fname


def save_checkpoint(state: State, step: int, cfg: NHQGConfig,
                    output_dir: str = None):
    """Save spectral state for restart."""
    if output_dir is None:
        output_dir = cfg.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fname = os.path.join(output_dir, f'checkpoint_{step:08d}.npz')
    np.savez(fname,
             q_hat_real=np.array(state.q_hat.real),
             q_hat_imag=np.array(state.q_hat.imag),
             w_hat_real=np.array(state.w_hat.real),
             w_hat_imag=np.array(state.w_hat.imag),
             th_hat_real=np.array(state.th_hat.real),
             th_hat_imag=np.array(state.th_hat.imag),
             th_bar=np.array(state.th_bar),
             step=step,
             t=step * cfg.dt)
    return fname


def load_checkpoint(path: str, dtype=jnp.complex128) -> tuple[State, int, float]:
    """Load spectral state from checkpoint.

    Returns: (state, step, t)
    """
    data = np.load(path)
    q_hat = jnp.array(data['q_hat_real'] + 1j * data['q_hat_imag'], dtype=dtype)
    w_hat = jnp.array(data['w_hat_real'] + 1j * data['w_hat_imag'], dtype=dtype)
    th_hat = jnp.array(data['th_hat_real'] + 1j * data['th_hat_imag'], dtype=dtype)
    if 'th_bar' in data:
        th_bar = jnp.array(data['th_bar'], dtype=q_hat.real.dtype)
    else:
        th_bar = jnp.zeros(q_hat.shape[0], dtype=q_hat.real.dtype)
    return State(q_hat, w_hat, th_hat, th_bar), int(data['step']), float(data['t'])
