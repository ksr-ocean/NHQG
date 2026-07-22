"""Frozen configuration dataclass for the NHQGE solver."""

from __future__ import annotations

import dataclasses
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class NHQGConfig:
    """Immutable configuration for an NHQGE simulation.

    All physical and numerical parameters needed by the solver.
    """

    # --- Resolution ---
    Nx: int = 256          # Horizontal grid points (square domain)
    Nz: int = 32           # Number of Chebyshev intervals (Nz+1 CGL points)

    # --- Physical parameters ---
    Ra_tilde: float = 100.0   # Reduced Rayleigh number
    sigma: float = 1.0        # Prandtl number
    beta: float = 0.0         # PV gradient
    Ld: float = float('inf')  # Deformation radius (inf = barotropic limit)
    L: float = 20.0           # Horizontal domain size (in units of Lc)
    thermal_closure: str = "fixed_conduction"  # or "evolve_mean"
    mean_temp_eps_sq: float = 1.0  # epsilon^2 prefactor in d_t Theta_bar
    q_boundary: str = "none"  # "none" (Miquel-style) or "neumann" (stress-free q')
    nonlinear_advection: str = "jacobian"  # "jacobian" or "flux"

    # --- Time stepping ---
    dt: float = 1e-3          # Time step
    t_final: float = 100.0    # End time
    imex_scheme: str = "ars222"  # "ars222" or "rk443"

    # --- Dissipation ---
    nu_q: float = 0.0         # PV hyperviscosity coefficient
    hyper_order: int = 4      # Hyperviscosity order p (nabla^{2p})
    nu_w: float = 0.0         # Laplacian diffusion on w
    nu_theta: float = 0.0     # Laplacian diffusion on theta
    drag: float = 0.0         # Large-scale linear drag on q'

    # --- Output ---
    save_interval: int = 100  # Steps between snapshots
    output_dir: str = "output"

    # --- Precision ---
    float_dtype: str = "float32"  # "float32" or "float64"
    vertical_cutoff_n: int | None = None  # Optional high-n cutoff for w,theta
    vertical_dealiasing: str = "none"  # "none", "cheb_3o2", or "cheb_2x"
    horizontal_dealiasing: str = "32_rule"  # "32_rule" (pad+FFT+truncate) or "23_rule" (FFT on Nx grid, 2/3 mask)
    mean_exchange_discretization: str = "legacy"  # "legacy", "coral_workgrid", "coral_workgrid_weakmean", "coral_workgrid_paired", "balanced_midpoint", "balanced_sbp2", or "balanced_sbp2_pc"
    sbp_transfer_mode: str = "interp"  # "interp", "mass_adjoint", or "weighted_polar"
    sbp_corrector_substeps: int = 1  # Number of repeated SBP thermal corrector substeps per inserted stage

    @property
    def Nk(self) -> int:
        """Number of rfft2 modes along the last horizontal axis."""
        return self.Nx // 2 + 1

    @property
    def Npad(self) -> int:
        """Padded grid size for 3/2-rule dealiasing."""
        return 3 * self.Nx // 2

    @property
    def dx(self) -> float:
        """Horizontal grid spacing."""
        return self.L / self.Nx

    @property
    def Ld_inv_sq(self) -> float:
        """1/Ld^2, safe for Ld=inf."""
        if self.Ld == float('inf'):
            return 0.0
        return 1.0 / (self.Ld ** 2)

    @property
    def jnp_dtype(self):
        return jnp.float32 if self.float_dtype == "float32" else jnp.float64

    @property
    def complex_dtype(self):
        return jnp.complex64 if self.float_dtype == "float32" else jnp.complex128

    @property
    def n_outputs(self) -> int:
        """Number of output snapshots."""
        total_steps = int(self.t_final / self.dt)
        return total_steps // self.save_interval

    def with_updates(self, **kwargs) -> NHQGConfig:
        """Return a new config with specified fields updated."""
        return dataclasses.replace(self, **kwargs)
