#!/usr/bin/env python
"""Continue a trigonometric-vertical benchmark checkpoint to a later final time."""

from __future__ import annotations

import os
os.environ["JAX_ENABLE_X64"] = "1"

import argparse
import time
from pathlib import Path

import jax
import numpy as np

from trig_vertical_benchmark.config import TrigBenchmarkConfig
from trig_vertical_benchmark.diagnostics import compute_diagnostics
from trig_vertical_benchmark.io import load_checkpoint, save_checkpoint
from trig_vertical_benchmark.operators import make_grid
from trig_vertical_benchmark.solver import imex_step_ars222


K_C = 1.3048
L_C = 2.0 * np.pi / K_C


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Nz", type=int, default=128)
    parser.add_argument("--Ra", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--t-final", type=float, required=True)
    parser.add_argument(
        "--thermal-closure",
        choices=["fixed_conduction", "evolve_mean"],
        required=True,
    )
    parser.add_argument(
        "--nonlinear-advection",
        choices=["jacobian", "flux"],
        default="jacobian",
    )
    parser.add_argument("--vertical-dealias-factor", type=float, default=1.5)
    parser.add_argument("--mean-temp-eps-sq", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    return parser.parse_args()


def _save_history(history: dict[str, list], aux: dict[str, np.ndarray], out_dir: Path) -> Path:
    spectra_path = out_dir / "spectra" / "spectrum_history.npz"
    spectra_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        spectra_path,
        t=np.array(history["t"], dtype=np.float64),
        step=np.array(history["step"], dtype=np.int64),
        steps=np.array(history["step"], dtype=np.int64),
        times=np.array(history["t"], dtype=np.float64),
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


def main():
    args = _parse_args()

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
        output_dir=args.output_dir,
        float_dtype="float64",
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spectra").mkdir(parents=True, exist_ok=True)

    print("=== Continue Trig Checkpoint ===")
    print(f"checkpoint={args.checkpoint}")
    print(f"output_dir={cfg.output_dir}")
    print(f"Nx={cfg.Nx} Nz={cfg.Nz} dt={cfg.dt} t_final={cfg.t_final}")
    print(f"thermal_closure={cfg.thermal_closure}")
    print(f"nonlinear_advection={cfg.nonlinear_advection}")
    print(f"vertical_dealias_factor={cfg.vertical_dealias_factor}")
    print(f"devices={jax.devices()}", flush=True)

    t0 = time.time()
    grid = make_grid(cfg)
    state, checkpoint_step, start_t = load_checkpoint(args.checkpoint)
    if cfg.t_final <= start_t:
        raise ValueError("t_final must be larger than the checkpoint time")
    start_step = int(round(start_t / cfg.dt))
    if abs(start_step * cfg.dt - start_t) > max(1e-12, 1e-9 * max(1.0, abs(start_t))):
        raise ValueError(
            f"checkpoint time {start_t} is not compatible with requested dt={cfg.dt}"
        )

    step_fn = jax.jit(lambda s: imex_step_ars222(s, grid))
    state = step_fn(state)
    jax.block_until_ready(state.psi_hat)
    print(f"compile+warmup={time.time() - t0:.2f}s", flush=True)

    history: dict[str, list] = {
        "t": [],
        "step": [],
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

    def record(state_now, step: int):
        t_sim = step * cfg.dt
        diag = compute_diagnostics(state_now, grid)
        history["t"].append(t_sim)
        history["step"].append(step)
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
        print(
            f"step={step:7d} t={t_sim:7.3f} "
            f"Nu_d={float(diag['Nusselt_dealiased']):8.3f} "
            f"Nu_raw={float(diag['Nusselt']):.3e} "
            f"max_v={float(diag['max_speed']):8.3f} "
            f"max_w={float(diag['max_w']):8.3f} "
            f"max_th={float(diag['max_theta']):8.3f} "
            f"tw_d={float(diag['vol_avg_tw_dealiased']): .3e} "
            f"tw_raw={float(diag['vol_avg_tw']): .3e} "
            f"R_ex_d={float(diag['mean_theta_exchange_residual_dealiased']): .3e} "
            f"|th_bar|={float(diag['th_bar_phys_max']):.3e}",
            flush=True,
        )

    print(
        f"checkpoint_step_old={checkpoint_step} checkpoint_time={start_t:.6f} "
        f"restart_step_new={start_step}",
        flush=True,
    )
    record(state, start_step)
    total_steps = int(round(cfg.t_final / cfg.dt))
    wall0 = time.time()
    for step in range(start_step + 1, total_steps + 1):
        state = step_fn(state)
        if step % cfg.save_interval == 0 or step == total_steps:
            jax.block_until_ready(state.psi_hat)
            record(state, step)
            print(f"wall={time.time() - wall0:7.1f}s", flush=True)
        if step % args.checkpoint_every == 0 or step == total_steps:
            fname = save_checkpoint(state, step, cfg, args.output_dir)
            print(f"saved {fname}", flush=True)

    spectra_path = _save_history(history, aux, out_dir)
    print(f"saved {spectra_path}", flush=True)


if __name__ == "__main__":
    main()
