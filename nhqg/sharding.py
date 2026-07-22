"""M3: single-host multi-GPU sharding of the solver state (GSPMD).

Strategy: shard ONE axis of the horizontal-spectral state arrays across a
1-D device mesh and let XLA's GSPMD propagate the layout through the jitted
step. Two candidate axes (state layout (Nz*, Nx, Nk)):

- 'kx' (axis 1): vertical contractions (G_Z, stencils, the per-shell IMEX
  matmul) are batched over kx rows -> zero communication; the horizontal
  FFT contracts kx -> all-to-all inside irfft2/rfft2. Nx divides evenly.
- 'z' (axis 0): horizontal FFTs are batched over levels -> the whole
  nonlinear pipeline is communication-free; vertical contractions contract
  the sharded axis -> all-gathers. Note Nz+1 / Nz-1 are odd, so the split
  is uneven (GSPMD pads).

Which wins is an empirical question (plan Part II); measure both on GPUs
6/7 at 256^2 x 64 before wiring the production default.

The mean profile th_bar is tiny and kept replicated. Grid operator arrays
are left uncommitted and are auto-replicated by jit.
"""

from __future__ import annotations

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

SHARD_AXES = ("none", "z", "kx")


def make_mesh(n_devices: int | None = None) -> Mesh:
    """1-D mesh ('dev',) over the first n visible devices (default: all)."""
    devices = jax.devices()
    if n_devices is not None:
        devices = devices[:n_devices]
    return Mesh(np.array(devices), ("dev",))


def state_sharding_specs(shard_axis: str):
    """(spec_3d, spec_1d) PartitionSpecs for state fields."""
    if shard_axis == "z":
        return P("dev", None, None), P()
    if shard_axis == "kx":
        return P(None, "dev", None), P()
    raise ValueError(f"shard_axis must be one of {SHARD_AXES}, got {shard_axis!r}")


def shard_state(state, mesh: Mesh, shard_axis: str):
    """device_put the State pytree with the chosen layout (no-op for 'none')."""
    if shard_axis == "none":
        return state
    spec3, spec1 = state_sharding_specs(shard_axis)
    s3 = NamedSharding(mesh, spec3)
    s1 = NamedSharding(mesh, spec1)
    return type(state)(
        jax.device_put(state.q_hat, s3),
        jax.device_put(state.w_hat, s3),
        jax.device_put(state.th_hat, s3),
        jax.device_put(state.th_bar, s1),
    )
