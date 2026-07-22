"""Configuration for the FD-in-z benchmark solver."""

from __future__ import annotations

import dataclasses
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class FDBenchmarkConfig:
    """Immutable configuration for the finite-difference benchmark solver."""

    Nx: int = 128
    Nz: int = 128  # number of uniform intervals, so there are Nz+1 full nodes

    Ra_tilde: float = 100.0
    sigma: float = 1.0
    L: float = 20.0
    thermal_closure: str = "evolve_mean"  # or "fixed_conduction"
    mean_temp_eps_sq: float = 1.0
    nonlinear_advection: str = "jacobian"  # or "flux"
    vertical_derivative: str = "compact4"  # or "centered2" or "sbp42"
    vertical_second_derivative: str = "centered2"  # or "compact4_raw" or "from_d1_energy" or "sbp42_energy"
    psi_neumann_treatment: str = "projected"  # or "direct"
    psi_boundary: str = "neumann"  # "neumann" (reconstruct dpsi/dz=0) or "none" (production-style: full-grid psi, no vorticity BC)
    mean_exchange: str = "plain"  # "plain" (explicit flux divergence) or "balanced_sbp" (energy-balanced predictor/corrector)
    sbp_corrector_substeps: int = 1  # SBP thermal corrector substeps per ARS stage (balanced_sbp only)
    vertical_grid: str = "uniform"  # "uniform" or "tanh" (boundary-clustered mapped-SBP grid)
    stretch_beta: float = 4.0  # tanh clustering strength (larger = finer near both walls); ignored when uniform

    dt: float = 5e-5
    t_final: float = 5.0

    nu_q: float = 1.0
    nu_w: float = 1.0
    nu_theta: float = 1.0
    hyper_order: int = 1
    drag: float = 0.0

    save_interval: int = 5000
    output_dir: str = "output_fd_vertical_benchmark"
    float_dtype: str = "float64"

    @property
    def Nk(self) -> int:
        return self.Nx // 2 + 1

    @property
    def Npad(self) -> int:
        return 3 * self.Nx // 2

    @property
    def dz(self) -> float:
        return 1.0 / self.Nz

    @property
    def interior_size(self) -> int:
        return self.Nz - 1

    @property
    def jnp_dtype(self):
        return jnp.float32 if self.float_dtype == "float32" else jnp.float64

    @property
    def complex_dtype(self):
        return jnp.complex64 if self.float_dtype == "float32" else jnp.complex128
