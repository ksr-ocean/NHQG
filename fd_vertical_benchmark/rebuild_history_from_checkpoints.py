#!/usr/bin/env python
"""Rebuild a spectrum_history archive from saved FD benchmark checkpoints."""

from __future__ import annotations

import os
os.environ["JAX_ENABLE_X64"] = "1"

import argparse
import math
from pathlib import Path

import numpy as np

from fd_vertical_benchmark.config import FDBenchmarkConfig
from fd_vertical_benchmark.diagnostics import compute_diagnostics
from fd_vertical_benchmark.io import load_checkpoint
from fd_vertical_benchmark.operators import make_grid


K_C = 1.3048
L_C = 2.0 * math.pi / K_C


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--Nx", type=int, required=True)
    parser.add_argument("--Nz", type=int, required=True)
    parser.add_argument("--Ra", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument(
        "--thermal-closure",
        choices=["fixed_conduction", "evolve_mean"],
        default="evolve_mean",
    )
    parser.add_argument(
        "--nonlinear-advection",
        choices=["jacobian", "flux"],
        default="jacobian",
    )
    parser.add_argument(
        "--vertical-derivative",
        choices=["centered2", "compact4", "sbp42"],
        default="compact4",
    )
    parser.add_argument(
        "--vertical-second-derivative",
        choices=["centered2", "compact4_raw", "from_d1_energy", "sbp42_energy"],
        default="centered2",
    )
    parser.add_argument(
        "--psi-neumann-treatment",
        choices=["projected", "direct"],
        default="projected",
    )
    parser.add_argument("--mean-temp-eps-sq", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = _parse_args()
    out_dir = Path(args.output_dir)
    ckpts = sorted(out_dir.glob("checkpoint_*.npz"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {out_dir}")

    cfg = FDBenchmarkConfig(
        Nx=args.Nx,
        Nz=args.Nz,
        L=10.0 * L_C,
        Ra_tilde=args.Ra,
        dt=args.dt,
        t_final=0.0,
        thermal_closure=args.thermal_closure,
        mean_temp_eps_sq=args.mean_temp_eps_sq,
        nonlinear_advection=args.nonlinear_advection,
        vertical_derivative=args.vertical_derivative,
        vertical_second_derivative=args.vertical_second_derivative,
        psi_neumann_treatment=args.psi_neumann_treatment,
        output_dir=str(out_dir),
        float_dtype="float64",
    )
    grid = make_grid(cfg)

    history: dict[str, list] = {
        "steps": [],
        "times": [],
        "Nusselt": [],
        "vol_avg_tw": [],
        "max_speed": [],
        "max_w": [],
        "max_theta": [],
        "max_tw": [],
        "q_rms": [],
        "w_rms": [],
        "th_rms": [],
        "th_bar_max": [],
        "KE_bt": [],
        "KE_bc": [],
        "KE_tot": [],
        "enstrophy": [],
        "q_horiz_spec": [],
        "w_horiz_spec": [],
        "th_horiz_spec": [],
        "q_z_power": [],
        "w_z_power": [],
        "th_z_power": [],
    }

    for ckpt in ckpts:
        state, step, t = load_checkpoint(str(ckpt))
        diag = compute_diagnostics(state, grid)
        history["steps"].append(step)
        history["times"].append(t)
        for key in [
            "Nusselt", "vol_avg_tw", "max_speed", "max_w", "max_theta", "max_tw",
            "q_rms", "w_rms", "th_rms", "th_bar_max", "KE_bt", "KE_bc", "KE_tot", "enstrophy",
        ]:
            history[key].append(float(diag[key]))
        for key in ["q_horiz_spec", "w_horiz_spec", "th_horiz_spec", "q_z_power", "w_z_power", "th_z_power"]:
            history[key].append(np.array(diag[key]))
        print(f"{ckpt.name}: t={t:.3f} Nu={float(diag['Nusselt']):.6e}", flush=True)

    spectra_path = out_dir / "spectra" / "spectrum_history.npz"
    spectra_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        spectra_path,
        steps=np.array(history["steps"], dtype=np.int32),
        times=np.array(history["times"], dtype=np.float64),
        Nusselt=np.array(history["Nusselt"], dtype=np.float64),
        vol_avg_tw=np.array(history["vol_avg_tw"], dtype=np.float64),
        max_speed=np.array(history["max_speed"], dtype=np.float64),
        max_w=np.array(history["max_w"], dtype=np.float64),
        max_theta=np.array(history["max_theta"], dtype=np.float64),
        max_tw=np.array(history["max_tw"], dtype=np.float64),
        q_rms=np.array(history["q_rms"], dtype=np.float64),
        w_rms=np.array(history["w_rms"], dtype=np.float64),
        th_rms=np.array(history["th_rms"], dtype=np.float64),
        th_bar_max=np.array(history["th_bar_max"], dtype=np.float64),
        KE_bt=np.array(history["KE_bt"], dtype=np.float64),
        KE_bc=np.array(history["KE_bc"], dtype=np.float64),
        KE_tot=np.array(history["KE_tot"], dtype=np.float64),
        enstrophy=np.array(history["enstrophy"], dtype=np.float64),
        q_horiz_spec=np.array(history["q_horiz_spec"]),
        w_horiz_spec=np.array(history["w_horiz_spec"]),
        th_horiz_spec=np.array(history["th_horiz_spec"]),
        q_z_power=np.array(history["q_z_power"]),
        w_z_power=np.array(history["w_z_power"]),
        th_z_power=np.array(history["th_z_power"]),
        k_bins=np.array(diag["k_bins"]),
        z_full=np.array(grid.z_full),
    )
    print(f"saved {spectra_path}", flush=True)


if __name__ == "__main__":
    main()
