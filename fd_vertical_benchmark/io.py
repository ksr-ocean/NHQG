"""Checkpoint helpers for the FD benchmark solver."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from fd_vertical_benchmark.config import FDBenchmarkConfig
from fd_vertical_benchmark.solver import State


def save_checkpoint(state: State, step: int, cfg: FDBenchmarkConfig, output_dir: str | None = None) -> str:
    """Save an interior-node benchmark checkpoint."""
    out_dir = Path(output_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"checkpoint_{step:08d}.npz"
    np.savez(
        path,
        psi_hat_real=np.array(state.psi_hat.real),
        psi_hat_imag=np.array(state.psi_hat.imag),
        w_hat_real=np.array(state.w_hat.real),
        w_hat_imag=np.array(state.w_hat.imag),
        th_hat_real=np.array(state.th_hat.real),
        th_hat_imag=np.array(state.th_hat.imag),
        th_bar=np.array(state.th_bar),
        step=step,
        t=step * cfg.dt,
    )
    return str(path)


def load_checkpoint(path: str, dtype=jnp.complex128) -> tuple[State, int, float]:
    """Load an interior-node benchmark checkpoint."""
    data = np.load(path)
    psi_hat = jnp.array(data["psi_hat_real"] + 1j * data["psi_hat_imag"], dtype=dtype)
    w_hat = jnp.array(data["w_hat_real"] + 1j * data["w_hat_imag"], dtype=dtype)
    th_hat = jnp.array(data["th_hat_real"] + 1j * data["th_hat_imag"], dtype=dtype)
    th_bar = jnp.array(data["th_bar"], dtype=psi_hat.real.dtype)
    return State(psi_hat, w_hat, th_hat, th_bar), int(data["step"]), float(data["t"])
