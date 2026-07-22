#!/usr/bin/env python
"""Run the upright standard-f FD-in-z benchmark case."""

from __future__ import annotations

import os
os.environ["JAX_ENABLE_X64"] = "1"

import argparse
import math
import time
from pathlib import Path

import jax
import numpy as np

from fd_vertical_benchmark.config import FDBenchmarkConfig
from fd_vertical_benchmark.diagnostics import compute_diagnostics
from fd_vertical_benchmark.io import save_checkpoint
from fd_vertical_benchmark.operators import make_grid
from fd_vertical_benchmark.solver import make_initial_state, run


K_C = 1.3048
L_C = 2.0 * math.pi / K_C


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Nz", type=int, default=128)
    parser.add_argument("--Ra", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--t-final", type=float, default=5.0)
    parser.add_argument("--amplitude", type=float, default=1e-6)
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
    parser.add_argument(
        "--psi-boundary",
        choices=["neumann", "none"],
        default="neumann",
        help="Vorticity/streamfunction BC: 'neumann' (reconstruct dpsi/dz=0) or 'none' "
             "(production-style: full-grid psi, no vorticity BC).",
    )
    parser.add_argument(
        "--mean-exchange",
        choices=["plain", "balanced_sbp"],
        default="plain",
        help="Mean/fluctuation thermal exchange: plain explicit, or energy-balanced SBP predictor/corrector.",
    )
    parser.add_argument("--sbp-corrector-substeps", type=int, default=1)
    parser.add_argument("--vertical-grid", choices=["uniform", "tanh"], default="uniform",
                        help="Vertical node distribution: uniform, or tanh (boundary-clustered mapped SBP).")
    parser.add_argument("--stretch-beta", type=float, default=4.0,
                        help="tanh clustering strength (larger = finer near both walls).")
    parser.add_argument("--mean-temp-eps-sq", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()

    nonlinear_suffix = "" if args.nonlinear_advection == "jacobian" else f"_{args.nonlinear_advection}form"
    vertical_suffix = f"_{args.vertical_derivative}"
    d2_suffix = "" if args.vertical_second_derivative == "centered2" else f"_d2{args.vertical_second_derivative}"
    psi_suffix = "" if args.psi_neumann_treatment == "projected" else f"_neu{args.psi_neumann_treatment}"
    psi_suffix += "" if args.psi_boundary == "neumann" else "_qnone"
    exch_suffix = "" if args.mean_exchange == "plain" else f"_balancedsbp_sub{args.sbp_corrector_substeps}"
    grid_suffix = "" if args.vertical_grid == "uniform" else f"_{args.vertical_grid}b{str(args.stretch_beta).replace('.', '')}"
    output_dir = args.output_dir or (
        f"output_miquel_zero_tilt_fd_ars222_{args.thermal_closure}{nonlinear_suffix}{vertical_suffix}{d2_suffix}{psi_suffix}{exch_suffix}{grid_suffix}"
        f"_Nx{args.Nx}_Nz{args.Nz}_dt{str(args.dt).replace('.', '')}_t{str(args.t_final).replace('.', '')}"
    )

    cfg = FDBenchmarkConfig(
        Nx=args.Nx,
        Nz=args.Nz,
        L=10.0 * L_C,
        Ra_tilde=args.Ra,
        dt=args.dt,
        t_final=args.t_final,
        thermal_closure=args.thermal_closure,
        mean_temp_eps_sq=args.mean_temp_eps_sq,
        nonlinear_advection=args.nonlinear_advection,
        vertical_derivative=args.vertical_derivative,
        vertical_second_derivative=args.vertical_second_derivative,
        psi_neumann_treatment=args.psi_neumann_treatment,
        psi_boundary=args.psi_boundary,
        mean_exchange=args.mean_exchange,
        sbp_corrector_substeps=args.sbp_corrector_substeps,
        vertical_grid=args.vertical_grid,
        stretch_beta=args.stretch_beta,
        save_interval=args.save_every,
        output_dir=output_dir,
        float_dtype="float64",
    )

    print("=== FD Vertical Benchmark ===")
    print(f"Nx={cfg.Nx} Nz={cfg.Nz} L={cfg.L:.4f} ({cfg.L / L_C:.1f} Lc)")
    print(f"Ra={cfg.Ra_tilde} dt={cfg.dt} t_final={cfg.t_final}")
    print(f"thermal_closure={cfg.thermal_closure} mean_temp_eps_sq={cfg.mean_temp_eps_sq}")
    print(f"nonlinear_advection={cfg.nonlinear_advection}")
    print(f"vertical_derivative={cfg.vertical_derivative}")
    print(f"vertical_second_derivative={cfg.vertical_second_derivative}")
    print(f"psi_neumann_treatment={cfg.psi_neumann_treatment} psi_boundary={cfg.psi_boundary}")
    print(f"mean_exchange={cfg.mean_exchange} sbp_corrector_substeps={cfg.sbp_corrector_substeps}")
    print(f"vertical_grid={cfg.vertical_grid} stretch_beta={cfg.stretch_beta}")
    print(f"output_dir={cfg.output_dir}")
    print(f"devices={jax.devices()}")
    print(flush=True)

    t0 = time.time()
    grid = make_grid(cfg)
    print(f"make_grid = {time.time() - t0:.2f}s", flush=True)

    t1 = time.time()
    state = make_initial_state(grid, seed=args.seed, amplitude=args.amplitude)
    print(f"make_initial_state = {time.time() - t1:.2f}s", flush=True)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_dir, "spectra").mkdir(parents=True, exist_ok=True)

    total_steps = int(round(cfg.t_final / cfg.dt))
    history: dict[str, list] = {
        "steps": [],
        "times": [],
        "Nusselt": [],
        "vol_avg_tw": [],
        "max_speed": [],
        "max_w": [],
        "max_theta": [],
        "max_tw": [],
        "R_ex_sbp": [],
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
    aux: dict[str, np.ndarray] = {}

    def save_history():
        spectra_path = Path(cfg.output_dir) / "spectra" / "spectrum_history.npz"
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
            R_ex_sbp=np.array(history["R_ex_sbp"], dtype=np.float64),
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
            k_bins=aux.get("k_bins", np.array([])),
            z_full=aux.get("z_full", np.array([])),
        )
        return spectra_path

    def callback(state_now, step, t):
        diag = compute_diagnostics(state_now, grid)

        history["steps"].append(step)
        history["times"].append(t)
        for key in [
            "Nusselt", "vol_avg_tw", "max_speed", "max_w", "max_theta", "max_tw", "R_ex_sbp",
            "q_rms", "w_rms", "th_rms", "th_bar_max", "KE_bt", "KE_bc", "KE_tot", "enstrophy",
        ]:
            history[key].append(float(diag[key]))
        for key in ["q_horiz_spec", "w_horiz_spec", "th_horiz_spec", "q_z_power", "w_z_power", "th_z_power"]:
            history[key].append(np.array(diag[key]))

        aux["k_bins"] = np.array(diag["k_bins"])
        aux["z_full"] = np.array(grid.z_full)

        if step % args.checkpoint_every == 0 or step == total_steps:
            save_checkpoint(state_now, step, cfg)

        spectra_path = save_history()

        print(
            f"step={step:7d} t={t:7.3f} "
            f"Nu={float(diag['Nusselt']):.6e} "
            f"max_u={float(diag['max_speed']):.3e} "
            f"max_w={float(diag['max_w']):.3e} "
            f"max_th={float(diag['max_theta']):.3e} "
            f"tw_avg={float(diag['vol_avg_tw']):.6e} "
            f"th_bar_max={float(diag['th_bar_max']):.3e} "
            f"R_ex_sbp={float(diag['R_ex_sbp']):.3e} "
            f"history={spectra_path.name}",
            flush=True,
        )

    t_run = time.time()
    state = run(grid, state, total_steps, cfg.save_interval, callback=callback)
    jax.block_until_ready(state.psi_hat)
    print(f"run_time = {time.time() - t_run:.2f}s", flush=True)

    spectra_path = save_history()
    print(f"saved {spectra_path}", flush=True)


if __name__ == "__main__":
    main()
