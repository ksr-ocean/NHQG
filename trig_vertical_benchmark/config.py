"""Configuration for the trigonometric-in-z benchmark solver."""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class TrigBenchmarkConfig:
    """Immutable configuration for the trigonometric vertical benchmark solver."""

    Nx: int = 128
    Nz: int = 128  # number of nonzero sine modes; psi uses Nz+1 cosine modes

    Ra_tilde: float = 100.0
    sigma: float = 1.0
    L: float = 20.0
    thermal_closure: str = "evolve_mean"  # or "fixed_conduction"
    mean_temp_eps_sq: float = 1.0
    nonlinear_advection: str = "jacobian"  # or "flux"
    vertical_dealias_factor: float = 1.5

    dt: float = 5e-5
    t_final: float = 5.0

    nu_q: float = 1.0
    nu_w: float = 1.0
    nu_theta: float = 1.0
    hyper_order: int = 1
    drag: float = 0.0

    save_interval: int = 5000
    output_dir: str = "output_trig_vertical_benchmark"
    float_dtype: str = "float64"

    @property
    def Nk(self) -> int:
        return self.Nx // 2 + 1

    @property
    def Npad(self) -> int:
        return 3 * self.Nx // 2

    @property
    def Ns(self) -> int:
        return self.Nz

    @property
    def Nc(self) -> int:
        return self.Nz + 1

    @property
    def Nz_work(self) -> int:
        return max(self.Nz, int(math.ceil(self.vertical_dealias_factor * self.Nz)))

    @property
    def jnp_dtype(self):
        return jnp.float32 if self.float_dtype == "float32" else jnp.float64

    @property
    def complex_dtype(self):
        return jnp.complex64 if self.float_dtype == "float32" else jnp.complex128

