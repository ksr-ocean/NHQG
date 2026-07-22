#!/usr/bin/env python
"""Run the upright standard-f trigonometric vertical benchmark case."""

from __future__ import annotations

import os
os.environ["JAX_ENABLE_X64"] = "1"

import argparse
import math
import time
from pathlib import Path

import jax
import numpy as np

from trig_vertical_benchmark.config import TrigBenchmarkConfig
from trig_vertical_benchmark.diagnostics import compute_diagnostics
from trig_vertical_benchmark.io import save_checkpoint
from trig_vertical_benchmark.operators import make_grid
from trig_vertical_benchmark.solver import make_initial_state, run


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
    parser.add_argument("--vertical-dealias-factor", type=float, default=1.5)
    parser.add_argument("--mean-temp-eps-sq", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()

    nonlinear_suffix = "" if args.nonlinear_advection == "jacobian" else f"_{args.nonlinear_advection}form"
    dealias_suffix = "" if args.vertical_dealias_factor == 1.5 else f"_zf{args.vertical_dealias_factor:g}"
    output_dir = args.output_dir or (
        f"output_miquel_zero_tilt_trig_ars222_{args.thermal_closure}{nonlinear_suffix}{dealias_suffix}"
        f"_Nx{args.Nx}_Nz{args.Nz}_dt{str(args.dt)}_t{str(args.t_final)}"
    )

    cfg = TrigBenchmarkConfig(
        Nx=args.Nx,
        Nz=args.Nz,
        L=10.0 * L_C,
        Ra_tilde=args.Ra,
        dt=args.dt,
        t_final=args.t_final,
        thermal_closure=args.thermal_closure,
        mean_temp_eps_sq=args.mean_temp_eps_sq,
        nonlinear_advection=args.nonlinear_advection,
        vertical_dealias_factor=args.vertical_dealias_factor,
        save_interval=args.save_every,
        output_dir=output_dir,
        float_dtype="float64",
    )

    print("=== Trig Vertical Benchmark ===")
    print(f"Nx={cfg.Nx} Nz={cfg.Nz} L={cfg.L:.4f} ({cfg.L / L_C:.1f} Lc)")
    print(f"Ra={cfg.Ra_tilde} dt={cfg.dt} t_final={cfg.t_final}")
    print(f"thermal_closure={cfg.thermal_closure} mean_temp_eps_sq={cfg.mean_temp_eps_sq}")
    print(f"nonlinear_advection={cfg.nonlinear_advection}")
    print(f"vertical_dealias_factor={cfg.vertical_dealias_factor}")
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
        "Nusselt_dealiased": [],
        "vol_avg_tw": [],
        "vol_avg_tw_dealiased": [],
        "heat_flux_mismatch": [],
        "max_speed": [],
        "max_w": [],
        "max_theta": [],
        "max_tw": [],
        "q_rms": [],
        "w_rms": [],
        "th_rms": [],
        "th_bar_max": [],
        "th_bar_phys_max": [],
        "dth_bar_dz_max": [],
        "mean_grad_min": [],
        "mean_grad_mid": [],
        "mean_grad_max": [],
        "mean_energy": [],
        "mean_flux_exchange_tendency": [],
        "mean_diffusion_tendency": [],
        "mean_total_tendency": [],
        "th_mean_feedback_sum_dealiased": [],
        "mean_theta_exchange_residual_dealiased": [],
        "mean_theta_exchange_residual_dealiased_rel": [],
        "KE_bt": [],
        "KE_bc": [],
        "KE_tot": [],
        "enstrophy": [],
        "q_horiz_spec": [],
        "w_horiz_spec": [],
        "th_horiz_spec": [],
        "heat_flux_shell_dealiased": [],
        "th_conduction_shell_tendency_dealiased": [],
        "w_buoyancy_shell_tendency_dealiased": [],
        "th_mean_feedback_shell_tendency_dealiased": [],
        "q_z_power": [],
        "w_z_power": [],
        "th_z_power": [],
    }
    aux: dict[str, np.ndarray] = {}

    def save_history():
        spectra_path = Path(cfg.output_dir) / "spectra" / "spectrum_history.npz"
        np.savez(
            spectra_path,
            t=np.array(history["times"], dtype=np.float64),
            step=np.array(history["steps"], dtype=np.int32),
            steps=np.array(history["steps"], dtype=np.int32),
            times=np.array(history["times"], dtype=np.float64),
            Nusselt=np.array(history["Nusselt"], dtype=np.float64),
            Nusselt_dealiased=np.array(history["Nusselt_dealiased"], dtype=np.float64),
            Nusselt_raw=np.array(history["Nusselt"], dtype=np.float64),
            vol_avg_tw=np.array(history["vol_avg_tw"], dtype=np.float64),
            vol_avg_tw_dealiased=np.array(history["vol_avg_tw_dealiased"], dtype=np.float64),
            vol_avg_tw_raw=np.array(history["vol_avg_tw"], dtype=np.float64),
            heat_flux_mismatch=np.array(history["heat_flux_mismatch"], dtype=np.float64),
            max_speed=np.array(history["max_speed"], dtype=np.float64),
            max_w=np.array(history["max_w"], dtype=np.float64),
            max_theta=np.array(history["max_theta"], dtype=np.float64),
            max_tw=np.array(history["max_tw"], dtype=np.float64),
            q_rms=np.array(history["q_rms"], dtype=np.float64),
            w_rms=np.array(history["w_rms"], dtype=np.float64),
            th_rms=np.array(history["th_rms"], dtype=np.float64),
            th_bar_max=np.array(history["th_bar_max"], dtype=np.float64),
            th_bar_phys_max=np.array(history["th_bar_phys_max"], dtype=np.float64),
            dth_bar_dz_max=np.array(history["dth_bar_dz_max"], dtype=np.float64),
            mean_grad_min=np.array(history["mean_grad_min"], dtype=np.float64),
            mean_grad_mid=np.array(history["mean_grad_mid"], dtype=np.float64),
            mean_grad_max=np.array(history["mean_grad_max"], dtype=np.float64),
            mean_energy=np.array(history["mean_energy"], dtype=np.float64),
            mean_flux_exchange_tendency=np.array(history["mean_flux_exchange_tendency"], dtype=np.float64),
            mean_diffusion_tendency=np.array(history["mean_diffusion_tendency"], dtype=np.float64),
            mean_total_tendency=np.array(history["mean_total_tendency"], dtype=np.float64),
            th_mean_feedback_sum_dealiased=np.array(history["th_mean_feedback_sum_dealiased"], dtype=np.float64),
            mean_theta_exchange_residual_dealiased=np.array(history["mean_theta_exchange_residual_dealiased"], dtype=np.float64),
            mean_theta_exchange_residual_dealiased_rel=np.array(history["mean_theta_exchange_residual_dealiased_rel"], dtype=np.float64),
            KE_bt=np.array(history["KE_bt"], dtype=np.float64),
            KE_bc=np.array(history["KE_bc"], dtype=np.float64),
            KE_tot=np.array(history["KE_tot"], dtype=np.float64),
            enstrophy=np.array(history["enstrophy"], dtype=np.float64),
            q_horiz_spec=np.array(history["q_horiz_spec"]),
            w_horiz_spec=np.array(history["w_horiz_spec"]),
            th_horiz_spec=np.array(history["th_horiz_spec"]),
            heat_flux_shell_dealiased=np.array(history["heat_flux_shell_dealiased"]),
            th_conduction_shell_tendency_dealiased=np.array(history["th_conduction_shell_tendency_dealiased"]),
            w_buoyancy_shell_tendency_dealiased=np.array(history["w_buoyancy_shell_tendency_dealiased"]),
            th_mean_feedback_shell_tendency_dealiased=np.array(history["th_mean_feedback_shell_tendency_dealiased"]),
            q_z_power=np.array(history["q_z_power"]),
            w_z_power=np.array(history["w_z_power"]),
            th_z_power=np.array(history["th_z_power"]),
            k_bins=aux.get("k_bins", np.array([])),
            z_work=aux.get("z_work", np.array([])),
        )
        return spectra_path

    def callback(state_now, step, t):
        diag = compute_diagnostics(state_now, grid)

        history["steps"].append(step)
        history["times"].append(t)
        for key in [
            "Nusselt", "Nusselt_dealiased", "vol_avg_tw", "vol_avg_tw_dealiased",
            "heat_flux_mismatch", "max_speed", "max_w", "max_theta", "max_tw",
            "q_rms", "w_rms", "th_rms", "th_bar_max", "th_bar_phys_max",
            "dth_bar_dz_max", "mean_grad_min", "mean_grad_mid", "mean_grad_max",
            "mean_energy", "mean_flux_exchange_tendency", "mean_diffusion_tendency",
            "mean_total_tendency", "th_mean_feedback_sum_dealiased",
            "mean_theta_exchange_residual_dealiased", "mean_theta_exchange_residual_dealiased_rel",
            "KE_bt", "KE_bc", "KE_tot", "enstrophy",
        ]:
            history[key].append(float(diag[key]))
        for key in [
            "q_horiz_spec", "w_horiz_spec", "th_horiz_spec",
            "heat_flux_shell_dealiased", "th_conduction_shell_tendency_dealiased",
            "w_buoyancy_shell_tendency_dealiased", "th_mean_feedback_shell_tendency_dealiased",
            "q_z_power", "w_z_power", "th_z_power",
        ]:
            history[key].append(np.array(diag[key]))

        aux["k_bins"] = np.array(diag["k_bins"])
        aux["z_work"] = np.array(grid.z_work)

        if step % args.checkpoint_every == 0 or step == total_steps:
            save_checkpoint(state_now, step, cfg)

        spectra_path = save_history()
        print(
            f"step={step:7d} t={t:7.3f} "
            f"Nu_d={float(diag['Nusselt_dealiased']):8.3f} "
            f"Nu_raw={float(diag['Nusselt']):.3e} "
            f"max_v={float(diag['max_speed']):8.3f} "
            f"max_w={float(diag['max_w']):8.3f} "
            f"max_th={float(diag['max_theta']):8.3f} "
            f"tw_d={float(diag['vol_avg_tw_dealiased']): .3e} "
            f"tw_raw={float(diag['vol_avg_tw']): .3e} "
            f"R_ex_d={float(diag['mean_theta_exchange_residual_dealiased']): .3e} "
            f"|th_bar|={float(diag['th_bar_phys_max']):.3e} "
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
