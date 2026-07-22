#!/usr/bin/env python
"""Rebuild a rich spectrum_history archive from NHQG checkpoint files."""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import math
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from nhqg.config import NHQGConfig
from nhqg.diagnostics import compute_diagnostics
from nhqg.grid import make_grid
from nhqg.io import load_checkpoint
from nhqg.paths import normalize_output_dir, resolve_existing_output_path


K_C = 1.3048
L_C = 2.0 * math.pi / K_C


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Nz", type=int, default=128)
    parser.add_argument("--Ra", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--t-final", type=float, default=8.0)
    parser.add_argument("--imex-scheme", type=str, default="ars222")
    parser.add_argument(
        "--thermal-closure",
        choices=["fixed_conduction", "evolve_mean"],
        default="evolve_mean",
    )
    parser.add_argument("--mean-temp-eps-sq", type=float, default=1.0)
    parser.add_argument(
        "--nonlinear-advection",
        choices=["jacobian", "flux"],
        default="jacobian",
    )
    parser.add_argument(
        "--vertical-dealiasing",
        choices=["none", "cheb_3o2", "cheb_2x"],
        default="none",
    )
    parser.add_argument(
        "--mean-exchange-discretization",
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
        "--q-boundary",
        choices=["none", "neumann"],
        default="none",
    )
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--Ld", type=float, default=float("inf"))
    parser.add_argument("--nu-q", type=float, default=1.0)
    parser.add_argument("--nu-w", type=float, default=1.0)
    parser.add_argument("--nu-theta", type=float, default=1.0)
    parser.add_argument("--hyper-order", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> NHQGConfig:
    return NHQGConfig(
        Nx=args.Nx,
        Nz=args.Nz,
        L=10.0 * L_C,
        Ra_tilde=args.Ra,
        sigma=1.0,
        beta=args.beta,
        Ld=args.Ld,
        dt=args.dt,
        t_final=args.t_final,
        imex_scheme=args.imex_scheme,
        q_boundary=args.q_boundary,
        nu_q=args.nu_q,
        nu_w=args.nu_w,
        nu_theta=args.nu_theta,
        hyper_order=args.hyper_order,
        thermal_closure=args.thermal_closure,
        mean_temp_eps_sq=args.mean_temp_eps_sq,
        nonlinear_advection=args.nonlinear_advection,
        vertical_dealiasing=args.vertical_dealiasing,
        mean_exchange_discretization=args.mean_exchange_discretization,
        output_dir=args.output_dir,
        float_dtype="float64",
    )


def main():
    args = _parse_args()
    args.output_dir = normalize_output_dir(args.output_dir)
    out_dir = resolve_existing_output_path(args.output_dir)
    ckpts = sorted(out_dir.glob("checkpoint_*.npz"))
    if args.stride > 1:
        ckpts = ckpts[::args.stride]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {out_dir}")

    cfg = _build_config(args)
    grid = make_grid(cfg)

    history: dict[str, list] = {
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
        "KE_bt": [],
        "KE_bc": [],
        "KE_tot": [],
        "enstrophy": [],
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
    }

    for ckpt in ckpts:
        state, step, t = load_checkpoint(str(ckpt), dtype=jnp.complex128)
        diag = compute_diagnostics(state, grid)
        history["t"].append(t)
        history["step"].append(step)
        for key in [
            "Nusselt", "Nusselt_dealiased", "max_speed", "max_w", "max_theta",
            "max_tw", "vol_avg_tw", "vol_avg_tw_dealiased", "heat_flux_mismatch",
            "q_rms", "w_rms", "th_rms", "th_bar_max", "th_bar_phys_max",
            "dth_bar_dz_max", "mean_grad_min", "mean_grad_max", "mean_grad_mid",
            "mean_energy", "mean_flux_exchange_tendency", "mean_diffusion_tendency",
            "mean_total_tendency", "th_mean_feedback_sum", "th_mean_feedback_sum_dealiased",
            "th_mean_feedback_sum_sbp", "mean_flux_exchange_tendency_sbp",
            "mean_theta_exchange_boundary_sbp", "mean_theta_exchange_residual_sbp",
            "mean_theta_exchange_residual_sbp_rel",
            "mean_theta_exchange_residual", "mean_theta_exchange_residual_rel",
            "mean_theta_exchange_residual_dealiased", "mean_theta_exchange_residual_dealiased_rel",
            "KE_bt", "KE_bc", "KE_tot", "enstrophy",
        ]:
            history[key].append(float(diag[key]))
        for key in [
            "q_vert_spec", "w_vert_spec", "th_vert_spec",
            "q_horiz_spec", "w_horiz_spec", "th_horiz_spec",
            "ke_horiz_spec",
            "ke_nonlinear_shell_tendency", "ke_beta_shell_tendency",
            "ke_stretch_shell_tendency", "ke_diss_shell_tendency",
            "ke_total_shell_tendency", "ke_nonlinear_flux",
            "w_nonlinear_shell_tendency", "w_q_coupling_shell_tendency",
            "w_buoyancy_shell_tendency", "w_diss_shell_tendency",
            "w_total_shell_tendency", "w_nonlinear_flux",
            "th_nonlinear_shell_tendency", "th_mean_feedback_shell_tendency",
            "th_mean_feedback_shell_tendency_dealiased",
            "th_conduction_shell_tendency", "th_conduction_shell_tendency_dealiased",
            "th_diss_shell_tendency", "th_total_shell_tendency", "th_nonlinear_flux",
            "w_buoyancy_shell_tendency_dealiased", "heat_flux_shell_dealiased",
        ]:
            history[key].append(np.array(diag[key]))
        print(
            f"{ckpt.name}: t={t:.3f} Nu={float(diag['Nusselt']):.6e} "
            f"KE_sum={float(diag['ke_total_sum']):.6e}",
            flush=True,
        )

    spectra_path = out_dir / "spectra" / "spectrum_history.npz"
    spectra_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        spectra_path,
        t=np.array(history["t"], dtype=np.float64),
        step=np.array(history["step"], dtype=np.int64),
        Nusselt=np.array(history["Nusselt"], dtype=np.float64),
        Nusselt_dealiased=np.array(history["Nusselt_dealiased"], dtype=np.float64),
        max_speed=np.array(history["max_speed"], dtype=np.float64),
        max_w=np.array(history["max_w"], dtype=np.float64),
        max_theta=np.array(history["max_theta"], dtype=np.float64),
        max_tw=np.array(history["max_tw"], dtype=np.float64),
        vol_avg_tw=np.array(history["vol_avg_tw"], dtype=np.float64),
        vol_avg_tw_dealiased=np.array(history["vol_avg_tw_dealiased"], dtype=np.float64),
        heat_flux_mismatch=np.array(history["heat_flux_mismatch"], dtype=np.float64),
        q_rms=np.array(history["q_rms"], dtype=np.float64),
        w_rms=np.array(history["w_rms"], dtype=np.float64),
        th_rms=np.array(history["th_rms"], dtype=np.float64),
        th_bar_max=np.array(history["th_bar_max"], dtype=np.float64),
        th_bar_phys_max=np.array(history["th_bar_phys_max"], dtype=np.float64),
        dth_bar_dz_max=np.array(history["dth_bar_dz_max"], dtype=np.float64),
        mean_grad_min=np.array(history["mean_grad_min"], dtype=np.float64),
        mean_grad_max=np.array(history["mean_grad_max"], dtype=np.float64),
        mean_grad_mid=np.array(history["mean_grad_mid"], dtype=np.float64),
        mean_energy=np.array(history["mean_energy"], dtype=np.float64),
        mean_flux_exchange_tendency=np.array(history["mean_flux_exchange_tendency"], dtype=np.float64),
        mean_diffusion_tendency=np.array(history["mean_diffusion_tendency"], dtype=np.float64),
        mean_total_tendency=np.array(history["mean_total_tendency"], dtype=np.float64),
        th_mean_feedback_sum=np.array(history["th_mean_feedback_sum"], dtype=np.float64),
        th_mean_feedback_sum_dealiased=np.array(history["th_mean_feedback_sum_dealiased"], dtype=np.float64),
        th_mean_feedback_sum_sbp=np.array(history["th_mean_feedback_sum_sbp"], dtype=np.float64),
        mean_flux_exchange_tendency_sbp=np.array(history["mean_flux_exchange_tendency_sbp"], dtype=np.float64),
        mean_theta_exchange_boundary_sbp=np.array(history["mean_theta_exchange_boundary_sbp"], dtype=np.float64),
        mean_theta_exchange_residual_sbp=np.array(history["mean_theta_exchange_residual_sbp"], dtype=np.float64),
        mean_theta_exchange_residual_sbp_rel=np.array(history["mean_theta_exchange_residual_sbp_rel"], dtype=np.float64),
        mean_theta_exchange_residual=np.array(history["mean_theta_exchange_residual"], dtype=np.float64),
        mean_theta_exchange_residual_rel=np.array(history["mean_theta_exchange_residual_rel"], dtype=np.float64),
        mean_theta_exchange_residual_dealiased=np.array(history["mean_theta_exchange_residual_dealiased"], dtype=np.float64),
        mean_theta_exchange_residual_dealiased_rel=np.array(history["mean_theta_exchange_residual_dealiased_rel"], dtype=np.float64),
        Nusselt_raw=np.array(history["Nusselt"], dtype=np.float64),
        vol_avg_tw_raw=np.array(history["vol_avg_tw"], dtype=np.float64),
        KE_bt=np.array(history["KE_bt"], dtype=np.float64),
        KE_bc=np.array(history["KE_bc"], dtype=np.float64),
        KE_tot=np.array(history["KE_tot"], dtype=np.float64),
        enstrophy=np.array(history["enstrophy"], dtype=np.float64),
        q_vert_spec=np.stack(history["q_vert_spec"]).astype(np.float64),
        w_vert_spec=np.stack(history["w_vert_spec"]).astype(np.float64),
        th_vert_spec=np.stack(history["th_vert_spec"]).astype(np.float64),
        q_horiz_spec=np.stack(history["q_horiz_spec"]).astype(np.float64),
        w_horiz_spec=np.stack(history["w_horiz_spec"]).astype(np.float64),
        th_horiz_spec=np.stack(history["th_horiz_spec"]).astype(np.float64),
        ke_horiz_spec=np.stack(history["ke_horiz_spec"]).astype(np.float64),
        ke_nonlinear_shell_tendency=np.stack(
            history["ke_nonlinear_shell_tendency"]
        ).astype(np.float64),
        ke_beta_shell_tendency=np.stack(
            history["ke_beta_shell_tendency"]
        ).astype(np.float64),
        ke_stretch_shell_tendency=np.stack(
            history["ke_stretch_shell_tendency"]
        ).astype(np.float64),
        ke_diss_shell_tendency=np.stack(
            history["ke_diss_shell_tendency"]
        ).astype(np.float64),
        ke_total_shell_tendency=np.stack(
            history["ke_total_shell_tendency"]
        ).astype(np.float64),
        ke_nonlinear_flux=np.stack(history["ke_nonlinear_flux"]).astype(np.float64),
        w_nonlinear_shell_tendency=np.stack(
            history["w_nonlinear_shell_tendency"]
        ).astype(np.float64),
        w_q_coupling_shell_tendency=np.stack(
            history["w_q_coupling_shell_tendency"]
        ).astype(np.float64),
        w_buoyancy_shell_tendency=np.stack(
            history["w_buoyancy_shell_tendency"]
        ).astype(np.float64),
        w_diss_shell_tendency=np.stack(
            history["w_diss_shell_tendency"]
        ).astype(np.float64),
        w_total_shell_tendency=np.stack(
            history["w_total_shell_tendency"]
        ).astype(np.float64),
        w_nonlinear_flux=np.stack(history["w_nonlinear_flux"]).astype(np.float64),
        th_nonlinear_shell_tendency=np.stack(
            history["th_nonlinear_shell_tendency"]
        ).astype(np.float64),
        th_mean_feedback_shell_tendency=np.stack(
            history["th_mean_feedback_shell_tendency"]
        ).astype(np.float64),
        th_mean_feedback_shell_tendency_dealiased=np.stack(
            history["th_mean_feedback_shell_tendency_dealiased"]
        ).astype(np.float64),
        th_conduction_shell_tendency=np.stack(
            history["th_conduction_shell_tendency"]
        ).astype(np.float64),
        th_conduction_shell_tendency_dealiased=np.stack(
            history["th_conduction_shell_tendency_dealiased"]
        ).astype(np.float64),
        th_diss_shell_tendency=np.stack(
            history["th_diss_shell_tendency"]
        ).astype(np.float64),
        th_total_shell_tendency=np.stack(
            history["th_total_shell_tendency"]
        ).astype(np.float64),
        th_nonlinear_flux=np.stack(history["th_nonlinear_flux"]).astype(np.float64),
        w_buoyancy_shell_tendency_dealiased=np.stack(
            history["w_buoyancy_shell_tendency_dealiased"]
        ).astype(np.float64),
        heat_flux_shell_dealiased=np.stack(
            history["heat_flux_shell_dealiased"]
        ).astype(np.float64),
        vert_mode=np.arange(len(history["q_vert_spec"][0]), dtype=np.int64),
        k_bins=np.array(diag["k_bins"], dtype=np.float64),
    )
    print(f"saved {spectra_path}", flush=True)


if __name__ == "__main__":
    main()
