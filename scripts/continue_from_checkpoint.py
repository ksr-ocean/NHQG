#!/usr/bin/env python
"""Continue a saved NHQG checkpoint to a later final time."""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.io import load_checkpoint, save_checkpoint, save_snapshot
from nhqg.paths import normalize_output_dir, resolve_existing_output_path
from nhqg.solver import imex_step
from nhqg.diagnostics import compute_diagnostics


K_C = 1.3048
L_C = 2.0 * math.pi / K_C


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--thermal-closure", type=str, choices=["fixed_conduction", "evolve_mean"], required=True)
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Nz", type=int, default=128)
    parser.add_argument("--Ra", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--t-final", type=float, required=True)
    parser.add_argument("--imex-scheme", type=str, default="ars222")
    parser.add_argument("--mean-temp-eps-sq", type=float, default=1.0)
    parser.add_argument(
        "--mean-exchange-discretization",
        type=str,
        choices=[
            "legacy",
            "coral_workgrid",
            "coral_workgrid_weakmean",
            "coral_workgrid_paired",
            "balanced_midpoint",
            "balanced_sbp2",
            "balanced_sbp2_pc",
        ],
        default="legacy",
    )
    parser.add_argument(
        "--sbp-transfer-mode",
        type=str,
        choices=["interp", "mass_adjoint", "weighted_polar"],
        default="interp",
    )
    parser.add_argument("--sbp-corrector-substeps", type=int, default=1)
    parser.add_argument(
        "--nonlinear-advection",
        type=str,
        choices=["jacobian", "flux"],
        default="jacobian",
    )
    parser.add_argument(
        "--horizontal-dealiasing",
        type=str,
        choices=["32_rule", "23_rule"],
        default="32_rule",
        help="Horizontal dealiasing scheme: 32_rule (pad+FFT+truncate) or 23_rule (FFT on Nx grid, mask top 1/3).",
    )
    parser.add_argument(
        "--vertical-dealiasing", type=str, default="none",
        choices=["none", "cheb_2x", "cheb_3o2"],
        help="Vertical Chebyshev dealiasing: none (default), cheb_2x (2x over-resolved CGL), cheb_3o2.",
    )
    parser.add_argument(
        "--hyper-order", type=int, default=1,
        help="Horizontal dissipation order p: nu*|k|^(2p). p=1=Laplacian, p=3=6th-order hyperviscosity.",
    )
    parser.add_argument(
        "--nu", type=float, default=1.0,
        help="Horizontal dissipation coefficient applied to all of nu_q, nu_w, nu_theta.",
    )
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--snapshot-times", type=float, nargs="+", default=[6.0, 7.0, 8.0, 9.0, 10.0])
    parser.add_argument(
        "--snapshot-dt",
        type=float,
        default=None,
        help="If set, also save snapshots at t = snapshot_dt, 2*snapshot_dt, ... up to t_final.",
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> NHQGConfig:
    return NHQGConfig(
        Nx=args.Nx,
        Nz=args.Nz,
        L=10.0 * L_C,
        Ra_tilde=args.Ra,
        sigma=1.0,
        beta=0.0,
        Ld=float("inf"),
        dt=args.dt,
        t_final=args.t_final,
        imex_scheme=args.imex_scheme,
        q_boundary="none",
        nu_q=args.nu,
        nu_w=args.nu,
        nu_theta=args.nu,
        hyper_order=args.hyper_order,
        vertical_dealiasing=args.vertical_dealiasing,
        thermal_closure=args.thermal_closure,
        mean_temp_eps_sq=args.mean_temp_eps_sq,
        mean_exchange_discretization=args.mean_exchange_discretization,
        sbp_transfer_mode=args.sbp_transfer_mode,
        sbp_corrector_substeps=args.sbp_corrector_substeps,
        nonlinear_advection=args.nonlinear_advection,
        horizontal_dealiasing=args.horizontal_dealiasing,
        save_interval=args.save_every,
        output_dir=args.output_dir,
        float_dtype="float64",
    )


def _record(diag_history, state, step, cfg, grid):
    t_sim = step * cfg.dt
    diag = compute_diagnostics(state, grid)
    th_bar_max = float(jnp.max(jnp.abs(state.th_bar)))
    print(
        f"step={step:7d} t={t_sim:6.2f} "
        f"Nu_d={float(diag['Nusselt_dealiased']):10.4e} "
        f"Nu_raw={float(diag['Nusselt']):10.4e} "
        f"max_v={float(diag['max_speed']):10.4e} "
        f"max_w={float(diag['max_w']):10.4e} "
        f"max_th={float(diag['max_theta']):10.4e} "
        f"tw_d={float(diag['vol_avg_tw_dealiased']):10.4e} "
        f"tw_raw={float(diag['vol_avg_tw']):10.4e} "
        f"R_ex_d={float(diag['mean_theta_exchange_residual_dealiased']):10.4e} "
        f"R_ex_sbp={float(diag['mean_theta_exchange_residual_sbp']):10.4e} "
        f"tw_max={float(diag['max_tw']):10.4e} "
        f"KE_bt={float(diag['KE_bt']):10.4e} "
        f"KE_bc={float(diag['KE_bc']):10.4e} "
        f"|th_bar|={th_bar_max:10.4e}",
        flush=True,
    )
    diag_history["t"].append(t_sim)
    diag_history["step"].append(step)
    diag_history["Nusselt"].append(float(diag["Nusselt"]))
    diag_history["Nusselt_dealiased"].append(float(diag["Nusselt_dealiased"]))
    diag_history["max_speed"].append(float(diag["max_speed"]))
    diag_history["max_w"].append(float(diag["max_w"]))
    diag_history["max_theta"].append(float(diag["max_theta"]))
    diag_history["max_tw"].append(float(diag["max_tw"]))
    diag_history["vol_avg_tw"].append(float(diag["vol_avg_tw"]))
    diag_history["vol_avg_tw_dealiased"].append(float(diag["vol_avg_tw_dealiased"]))
    diag_history["heat_flux_mismatch"].append(float(diag["heat_flux_mismatch"]))
    diag_history["q_rms"].append(float(diag["q_rms"]))
    diag_history["w_rms"].append(float(diag["w_rms"]))
    diag_history["th_rms"].append(float(diag["th_rms"]))
    diag_history["KE_bt"].append(float(diag["KE_bt"]))
    diag_history["KE_bc"].append(float(diag["KE_bc"]))
    diag_history["th_bar_max"].append(th_bar_max)
    diag_history["th_bar_phys_max"].append(float(diag["th_bar_phys_max"]))
    diag_history["dth_bar_dz_max"].append(float(diag["dth_bar_dz_max"]))
    diag_history["mean_grad_min"].append(float(diag["mean_grad_min"]))
    diag_history["mean_grad_max"].append(float(diag["mean_grad_max"]))
    diag_history["mean_grad_mid"].append(float(diag["mean_grad_mid"]))
    diag_history["mean_energy"].append(float(diag["mean_energy"]))
    diag_history["mean_flux_exchange_tendency"].append(float(diag["mean_flux_exchange_tendency"]))
    diag_history["mean_diffusion_tendency"].append(float(diag["mean_diffusion_tendency"]))
    diag_history["mean_total_tendency"].append(float(diag["mean_total_tendency"]))
    diag_history["th_mean_feedback_sum"].append(float(diag["th_mean_feedback_sum"]))
    diag_history["th_mean_feedback_sum_dealiased"].append(float(diag["th_mean_feedback_sum_dealiased"]))
    diag_history["th_mean_feedback_sum_sbp"].append(float(diag["th_mean_feedback_sum_sbp"]))
    diag_history["mean_flux_exchange_tendency_sbp"].append(float(diag["mean_flux_exchange_tendency_sbp"]))
    diag_history["mean_theta_exchange_boundary_sbp"].append(float(diag["mean_theta_exchange_boundary_sbp"]))
    diag_history["mean_theta_exchange_residual_sbp"].append(float(diag["mean_theta_exchange_residual_sbp"]))
    diag_history["mean_theta_exchange_residual_sbp_rel"].append(float(diag["mean_theta_exchange_residual_sbp_rel"]))
    diag_history["mean_theta_exchange_residual"].append(float(diag["mean_theta_exchange_residual"]))
    diag_history["mean_theta_exchange_residual_rel"].append(float(diag["mean_theta_exchange_residual_rel"]))
    diag_history["mean_theta_exchange_residual_dealiased"].append(float(diag["mean_theta_exchange_residual_dealiased"]))
    diag_history["mean_theta_exchange_residual_dealiased_rel"].append(float(diag["mean_theta_exchange_residual_dealiased_rel"]))
    diag_history["q_vert_spec"].append(np.array(diag["q_vert_spec"]))
    diag_history["w_vert_spec"].append(np.array(diag["w_vert_spec"]))
    diag_history["th_vert_spec"].append(np.array(diag["th_vert_spec"]))
    diag_history["q_horiz_spec"].append(np.array(diag["q_horiz_spec"]))
    diag_history["w_horiz_spec"].append(np.array(diag["w_horiz_spec"]))
    diag_history["th_horiz_spec"].append(np.array(diag["th_horiz_spec"]))
    diag_history["ke_horiz_spec"].append(np.array(diag["ke_horiz_spec"]))
    diag_history["ke_nonlinear_shell_tendency"].append(
        np.array(diag["ke_nonlinear_shell_tendency"])
    )
    diag_history["ke_beta_shell_tendency"].append(
        np.array(diag["ke_beta_shell_tendency"])
    )
    diag_history["ke_stretch_shell_tendency"].append(
        np.array(diag["ke_stretch_shell_tendency"])
    )
    diag_history["ke_diss_shell_tendency"].append(
        np.array(diag["ke_diss_shell_tendency"])
    )
    diag_history["ke_total_shell_tendency"].append(
        np.array(diag["ke_total_shell_tendency"])
    )
    diag_history["ke_nonlinear_flux"].append(np.array(diag["ke_nonlinear_flux"]))
    diag_history["w_nonlinear_shell_tendency"].append(
        np.array(diag["w_nonlinear_shell_tendency"])
    )
    diag_history["w_q_coupling_shell_tendency"].append(
        np.array(diag["w_q_coupling_shell_tendency"])
    )
    diag_history["w_buoyancy_shell_tendency"].append(
        np.array(diag["w_buoyancy_shell_tendency"])
    )
    diag_history["w_diss_shell_tendency"].append(
        np.array(diag["w_diss_shell_tendency"])
    )
    diag_history["w_total_shell_tendency"].append(
        np.array(diag["w_total_shell_tendency"])
    )
    diag_history["w_nonlinear_flux"].append(np.array(diag["w_nonlinear_flux"]))
    diag_history["th_nonlinear_shell_tendency"].append(
        np.array(diag["th_nonlinear_shell_tendency"])
    )
    diag_history["th_mean_feedback_shell_tendency"].append(
        np.array(diag["th_mean_feedback_shell_tendency"])
    )
    diag_history["th_mean_feedback_shell_tendency_dealiased"].append(
        np.array(diag["th_mean_feedback_shell_tendency_dealiased"])
    )
    diag_history["th_conduction_shell_tendency"].append(
        np.array(diag["th_conduction_shell_tendency"])
    )
    diag_history["th_conduction_shell_tendency_dealiased"].append(
        np.array(diag["th_conduction_shell_tendency_dealiased"])
    )
    diag_history["th_diss_shell_tendency"].append(
        np.array(diag["th_diss_shell_tendency"])
    )
    diag_history["th_total_shell_tendency"].append(
        np.array(diag["th_total_shell_tendency"])
    )
    diag_history["th_nonlinear_flux"].append(np.array(diag["th_nonlinear_flux"]))
    diag_history["w_buoyancy_shell_tendency_dealiased"].append(
        np.array(diag["w_buoyancy_shell_tendency_dealiased"])
    )
    diag_history["heat_flux_shell_dealiased"].append(
        np.array(diag["heat_flux_shell_dealiased"])
    )
    if diag_history["k_bins"] is None:
        diag_history["k_bins"] = np.array(diag["ke_k_bins"])


def main():
    args = _parse_args()
    args.checkpoint = str(resolve_existing_output_path(args.checkpoint))
    args.output_dir = normalize_output_dir(args.output_dir)
    cfg = _build_config(args)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=== Continue From Checkpoint ===", flush=True)
    print(f"checkpoint={args.checkpoint}", flush=True)
    print(f"output_dir={args.output_dir}", flush=True)
    print(f"thermal_closure={cfg.thermal_closure}", flush=True)
    print(f"mean_exchange_discretization={cfg.mean_exchange_discretization}", flush=True)
    print(f"sbp_transfer_mode={cfg.sbp_transfer_mode}", flush=True)
    print(f"sbp_corrector_substeps={cfg.sbp_corrector_substeps}", flush=True)
    print(f"nonlinear_advection={cfg.nonlinear_advection}", flush=True)
    print(f"horizontal_dealiasing={cfg.horizontal_dealiasing}", flush=True)
    print(f"vertical_dealiasing={cfg.vertical_dealiasing}", flush=True)
    print(f"hyper_order={cfg.hyper_order} nu={cfg.nu_q}", flush=True)
    print(f"dt={cfg.dt} t_final={cfg.t_final} imex_scheme={cfg.imex_scheme}", flush=True)
    print(f"device={jax.devices()}", flush=True)

    t0 = time.time()
    grid = make_grid(cfg)
    state, start_step, start_t = load_checkpoint(args.checkpoint, dtype=jnp.complex128)
    if abs(start_t - start_step * cfg.dt) > 1e-12:
        raise ValueError("Checkpoint time does not match step * dt")
    if cfg.t_final <= start_t:
        raise ValueError("t_final must be larger than checkpoint time")

    step_fn = jax.jit(lambda s: imex_step(s, grid), donate_argnums=(0,))
    state = step_fn(state)
    jax.block_until_ready(state.q_hat)
    print(f"compile+warmup={time.time() - t0:.1f}s", flush=True)

    snapshot_times = list(args.snapshot_times)
    if args.snapshot_dt is not None:
        n_snaps = int(math.floor(cfg.t_final / args.snapshot_dt + 1e-12))
        snapshot_times.extend([args.snapshot_dt * i for i in range(1, n_snaps + 1)])
    snapshot_steps = sorted({
        int(round(t / cfg.dt)) for t in snapshot_times if t > start_t and t <= cfg.t_final + 1e-12
    })
    snapshot_step_set = set(snapshot_steps)
    total_steps = int(round(cfg.t_final / cfg.dt))

    diag_history = {
        "t": [],
        "step": [],
        "Nusselt": [],
        "Nusselt_dealiased": [],
        "max_speed": [],
        "max_w": [],
        "max_theta": [],
        "max_tw": [],
        "vol_avg_tw": [],
        "vol_avg_tw_dealiased": [],
        "heat_flux_mismatch": [],
        "q_rms": [],
        "w_rms": [],
        "th_rms": [],
        "KE_bt": [],
        "KE_bc": [],
        "th_bar_max": [],
        "th_bar_phys_max": [],
        "dth_bar_dz_max": [],
        "mean_grad_min": [],
        "mean_grad_max": [],
        "mean_grad_mid": [],
        "mean_energy": [],
        "mean_flux_exchange_tendency": [],
        "mean_diffusion_tendency": [],
        "mean_total_tendency": [],
        "th_mean_feedback_sum": [],
        "th_mean_feedback_sum_dealiased": [],
        "th_mean_feedback_sum_sbp": [],
        "mean_flux_exchange_tendency_sbp": [],
        "mean_theta_exchange_boundary_sbp": [],
        "mean_theta_exchange_residual_sbp": [],
        "mean_theta_exchange_residual_sbp_rel": [],
        "mean_theta_exchange_residual": [],
        "mean_theta_exchange_residual_rel": [],
        "mean_theta_exchange_residual_dealiased": [],
        "mean_theta_exchange_residual_dealiased_rel": [],
        "q_vert_spec": [],
        "w_vert_spec": [],
        "th_vert_spec": [],
        "q_horiz_spec": [],
        "w_horiz_spec": [],
        "th_horiz_spec": [],
        "ke_horiz_spec": [],
        "ke_nonlinear_shell_tendency": [],
        "ke_beta_shell_tendency": [],
        "ke_stretch_shell_tendency": [],
        "ke_diss_shell_tendency": [],
        "ke_total_shell_tendency": [],
        "ke_nonlinear_flux": [],
        "w_nonlinear_shell_tendency": [],
        "w_q_coupling_shell_tendency": [],
        "w_buoyancy_shell_tendency": [],
        "w_diss_shell_tendency": [],
        "w_total_shell_tendency": [],
        "w_nonlinear_flux": [],
        "th_nonlinear_shell_tendency": [],
        "th_mean_feedback_shell_tendency": [],
        "th_mean_feedback_shell_tendency_dealiased": [],
        "th_conduction_shell_tendency": [],
        "th_conduction_shell_tendency_dealiased": [],
        "th_diss_shell_tendency": [],
        "th_total_shell_tendency": [],
        "th_nonlinear_flux": [],
        "w_buoyancy_shell_tendency_dealiased": [],
        "heat_flux_shell_dealiased": [],
        "k_bins": None,
    }

    _record(diag_history, state, start_step, cfg, grid)

    wall0 = time.time()
    for step in range(start_step + 1, total_steps + 1):
        state = step_fn(state)

        if step % cfg.save_interval == 0 or step in snapshot_step_set or step == total_steps:
            jax.block_until_ready(state.q_hat)
            _record(diag_history, state, step, cfg, grid)
            print(f"wall={time.time() - wall0:7.1f}s", flush=True)

        if step in snapshot_step_set:
            fname = save_snapshot(state, step * cfg.dt, step, cfg, grid, args.output_dir)
            print(f"saved {fname}", flush=True)

        if step % args.checkpoint_every == 0 or step == total_steps:
            fname = save_checkpoint(state, step, cfg, args.output_dir)
            print(f"saved {fname}", flush=True)

    np.savez(
        output_path / "diagnostics_history.npz",
        t=np.array(diag_history["t"], dtype=np.float64),
        step=np.array(diag_history["step"], dtype=np.int64),
        Nusselt=np.array(diag_history["Nusselt"], dtype=np.float64),
        Nusselt_dealiased=np.array(diag_history["Nusselt_dealiased"], dtype=np.float64),
        max_speed=np.array(diag_history["max_speed"], dtype=np.float64),
        max_w=np.array(diag_history["max_w"], dtype=np.float64),
        max_theta=np.array(diag_history["max_theta"], dtype=np.float64),
        max_tw=np.array(diag_history["max_tw"], dtype=np.float64),
        vol_avg_tw=np.array(diag_history["vol_avg_tw"], dtype=np.float64),
        vol_avg_tw_dealiased=np.array(diag_history["vol_avg_tw_dealiased"], dtype=np.float64),
        heat_flux_mismatch=np.array(diag_history["heat_flux_mismatch"], dtype=np.float64),
        q_rms=np.array(diag_history["q_rms"], dtype=np.float64),
        w_rms=np.array(diag_history["w_rms"], dtype=np.float64),
        th_rms=np.array(diag_history["th_rms"], dtype=np.float64),
        KE_bt=np.array(diag_history["KE_bt"], dtype=np.float64),
        KE_bc=np.array(diag_history["KE_bc"], dtype=np.float64),
        th_bar_max=np.array(diag_history["th_bar_max"], dtype=np.float64),
        th_bar_phys_max=np.array(diag_history["th_bar_phys_max"], dtype=np.float64),
        dth_bar_dz_max=np.array(diag_history["dth_bar_dz_max"], dtype=np.float64),
        mean_grad_min=np.array(diag_history["mean_grad_min"], dtype=np.float64),
        mean_grad_max=np.array(diag_history["mean_grad_max"], dtype=np.float64),
        mean_grad_mid=np.array(diag_history["mean_grad_mid"], dtype=np.float64),
        mean_energy=np.array(diag_history["mean_energy"], dtype=np.float64),
        mean_flux_exchange_tendency=np.array(diag_history["mean_flux_exchange_tendency"], dtype=np.float64),
        mean_diffusion_tendency=np.array(diag_history["mean_diffusion_tendency"], dtype=np.float64),
        mean_total_tendency=np.array(diag_history["mean_total_tendency"], dtype=np.float64),
        th_mean_feedback_sum=np.array(diag_history["th_mean_feedback_sum"], dtype=np.float64),
        th_mean_feedback_sum_dealiased=np.array(diag_history["th_mean_feedback_sum_dealiased"], dtype=np.float64),
        th_mean_feedback_sum_sbp=np.array(diag_history["th_mean_feedback_sum_sbp"], dtype=np.float64),
        mean_flux_exchange_tendency_sbp=np.array(diag_history["mean_flux_exchange_tendency_sbp"], dtype=np.float64),
        mean_theta_exchange_boundary_sbp=np.array(diag_history["mean_theta_exchange_boundary_sbp"], dtype=np.float64),
        mean_theta_exchange_residual_sbp=np.array(diag_history["mean_theta_exchange_residual_sbp"], dtype=np.float64),
        mean_theta_exchange_residual_sbp_rel=np.array(diag_history["mean_theta_exchange_residual_sbp_rel"], dtype=np.float64),
        mean_theta_exchange_residual=np.array(diag_history["mean_theta_exchange_residual"], dtype=np.float64),
        mean_theta_exchange_residual_rel=np.array(diag_history["mean_theta_exchange_residual_rel"], dtype=np.float64),
        mean_theta_exchange_residual_dealiased=np.array(diag_history["mean_theta_exchange_residual_dealiased"], dtype=np.float64),
        mean_theta_exchange_residual_dealiased_rel=np.array(diag_history["mean_theta_exchange_residual_dealiased_rel"], dtype=np.float64),
        Nusselt_raw=np.array(diag_history["Nusselt"], dtype=np.float64),
        vol_avg_tw_raw=np.array(diag_history["vol_avg_tw"], dtype=np.float64),
        q_vert_spec=np.stack(diag_history["q_vert_spec"]).astype(np.float64),
        w_vert_spec=np.stack(diag_history["w_vert_spec"]).astype(np.float64),
        th_vert_spec=np.stack(diag_history["th_vert_spec"]).astype(np.float64),
        q_horiz_spec=np.stack(diag_history["q_horiz_spec"]).astype(np.float64),
        w_horiz_spec=np.stack(diag_history["w_horiz_spec"]).astype(np.float64),
        th_horiz_spec=np.stack(diag_history["th_horiz_spec"]).astype(np.float64),
        ke_horiz_spec=np.stack(diag_history["ke_horiz_spec"]).astype(np.float64),
        ke_nonlinear_shell_tendency=np.stack(
            diag_history["ke_nonlinear_shell_tendency"]
        ).astype(np.float64),
        ke_beta_shell_tendency=np.stack(
            diag_history["ke_beta_shell_tendency"]
        ).astype(np.float64),
        ke_stretch_shell_tendency=np.stack(
            diag_history["ke_stretch_shell_tendency"]
        ).astype(np.float64),
        ke_diss_shell_tendency=np.stack(
            diag_history["ke_diss_shell_tendency"]
        ).astype(np.float64),
        ke_total_shell_tendency=np.stack(
            diag_history["ke_total_shell_tendency"]
        ).astype(np.float64),
        ke_nonlinear_flux=np.stack(diag_history["ke_nonlinear_flux"]).astype(np.float64),
        w_nonlinear_shell_tendency=np.stack(
            diag_history["w_nonlinear_shell_tendency"]
        ).astype(np.float64),
        w_q_coupling_shell_tendency=np.stack(
            diag_history["w_q_coupling_shell_tendency"]
        ).astype(np.float64),
        w_buoyancy_shell_tendency=np.stack(
            diag_history["w_buoyancy_shell_tendency"]
        ).astype(np.float64),
        w_diss_shell_tendency=np.stack(
            diag_history["w_diss_shell_tendency"]
        ).astype(np.float64),
        w_total_shell_tendency=np.stack(
            diag_history["w_total_shell_tendency"]
        ).astype(np.float64),
        w_nonlinear_flux=np.stack(diag_history["w_nonlinear_flux"]).astype(np.float64),
        th_nonlinear_shell_tendency=np.stack(
            diag_history["th_nonlinear_shell_tendency"]
        ).astype(np.float64),
        th_mean_feedback_shell_tendency=np.stack(
            diag_history["th_mean_feedback_shell_tendency"]
        ).astype(np.float64),
        th_mean_feedback_shell_tendency_dealiased=np.stack(
            diag_history["th_mean_feedback_shell_tendency_dealiased"]
        ).astype(np.float64),
        th_conduction_shell_tendency=np.stack(
            diag_history["th_conduction_shell_tendency"]
        ).astype(np.float64),
        th_conduction_shell_tendency_dealiased=np.stack(
            diag_history["th_conduction_shell_tendency_dealiased"]
        ).astype(np.float64),
        th_diss_shell_tendency=np.stack(
            diag_history["th_diss_shell_tendency"]
        ).astype(np.float64),
        th_total_shell_tendency=np.stack(
            diag_history["th_total_shell_tendency"]
        ).astype(np.float64),
        th_nonlinear_flux=np.stack(diag_history["th_nonlinear_flux"]).astype(np.float64),
        w_buoyancy_shell_tendency_dealiased=np.stack(
            diag_history["w_buoyancy_shell_tendency_dealiased"]
        ).astype(np.float64),
        heat_flux_shell_dealiased=np.stack(
            diag_history["heat_flux_shell_dealiased"]
        ).astype(np.float64),
        k_bins=np.array(diag_history["k_bins"], dtype=np.float64),
    )
    print(f"saved {output_path / 'diagnostics_history.npz'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
