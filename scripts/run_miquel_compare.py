#!/usr/bin/env python
"""Run a zero-tilt Miquel comparison and save field panels.

This script targets the upright case shared by our beta-plane model and
Miquel et al. 2026: beta=0, gamma=0, Ld=inf, Laplacian dissipation on all
fluctuation fields, domain size L=10*Lc.

Outputs:
- NetCDF field snapshots via nhqg.io.save_snapshot
- Checkpoints (.npz)
- PNG panel images with rows [w, theta, zeta] and columns [top, mid, bottom]

Here zeta is q' because beta=0 and Ld=inf.
"""

import os
os.environ["JAX_ENABLE_X64"] = "1"

import argparse
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.paths import normalize_output_dir
from nhqg.solver import make_initial_state, imex_step, _dirichlet_to_cheb
from nhqg.io import save_checkpoint, save_snapshot, _to_physical
from nhqg.diagnostics import compute_diagnostics


K_C = 1.3048
L_C = 2.0 * math.pi / K_C


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Nz", type=int, default=128)
    parser.add_argument("--Ra", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--t-final", type=float, default=5.0)
    parser.add_argument("--amplitude", type=float, default=1e-6)
    parser.add_argument("--imex-scheme", type=str, default="rk443")
    parser.add_argument(
        "--thermal-closure",
        type=str,
        choices=["fixed_conduction", "evolve_mean"],
        default="evolve_mean",
    )
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
        "--nonlinear-advection",
        type=str,
        choices=["jacobian", "flux"],
        default="jacobian",
    )
    parser.add_argument(
        "--sbp-transfer-mode",
        type=str,
        choices=["interp", "mass_adjoint", "weighted_polar"],
        default="interp",
    )
    parser.add_argument("--sbp-corrector-substeps", type=int, default=1)
    parser.add_argument("--vertical-dealiasing", type=str, default="none")
    parser.add_argument(
        "--horizontal-dealiasing",
        type=str,
        choices=["32_rule", "23_rule"],
        default="32_rule",
        help="Horizontal dealiasing scheme.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--snapshot-times", type=float, nargs="+", default=[2.0, 3.0, 4.0, 5.0])
    parser.add_argument(
        "--snapshot-dt",
        type=float,
        default=None,
        help="If set, also save snapshots at t = snapshot_dt, 2*snapshot_dt, ... up to t_final.",
    )
    parser.add_argument("--depths", type=float, nargs=3, default=[0.95, 0.50, 0.05])
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument(
        "--skip-inline-panels",
        action="store_true",
        help="Save only NetCDF snapshots during the run; render fixed-range panels in postprocessing.",
    )
    return parser.parse_args()


def _depth_indices(z, targets):
    return [int(np.argmin(np.abs(z - target))) for target in targets]


def _symmetric_limits(arr):
    finite = np.abs(arr[np.isfinite(arr)])
    if finite.size == 0:
        return 1e-12
    vmax = float(np.max(finite))
    return max(vmax, 1e-12)


def _diverging_rgb(field2d, vmax):
    scaled = np.clip(field2d / vmax, -1.0, 1.0)
    abs_scaled = np.abs(scaled)

    rgb = np.empty(field2d.shape + (3,), dtype=np.uint8)
    neg = scaled < 0

    rgb[..., 0] = 255
    rgb[..., 1] = (255 * (1.0 - abs_scaled)).astype(np.uint8)
    rgb[..., 2] = (255 * (1.0 - abs_scaled)).astype(np.uint8)

    rgb[neg, 0] = (255 * (1.0 - abs_scaled[neg])).astype(np.uint8)
    rgb[neg, 1] = (255 * (1.0 - abs_scaled[neg])).astype(np.uint8)
    rgb[neg, 2] = 255

    return rgb


def _resize_tile(rgb, tile_size):
    return Image.fromarray(rgb, mode="RGB").resize(
        (tile_size, tile_size), resample=Image.Resampling.BICUBIC
    )


def save_panel_png(state, grid, out_path, t, depth_targets):
    """Save a 3x3 panel: rows [w, theta, zeta], cols [top, mid, bottom]."""
    z = np.array(grid.Z)
    depth_idx = _depth_indices(z, depth_targets)
    depth_labels = [f"z={z[idx]:.3f}" for idx in depth_idx]

    q_phys = _to_physical(state.q_hat, grid.V, grid.Nx)
    w_phys = _to_physical(_dirichlet_to_cheb(state.w_hat, grid.dirichlet_stencil), grid.V, grid.Nx)
    th_phys = _to_physical(_dirichlet_to_cheb(state.th_hat, grid.dirichlet_stencil), grid.V, grid.Nx)

    fields = [("w", w_phys), ("theta", th_phys), ("zeta", q_phys)]

    tile_size = 240
    left_pad = 110
    top_pad = 55
    gap = 8
    width = left_pad + 3 * tile_size + 2 * gap + 20
    height = top_pad + 3 * tile_size + 2 * gap + 20

    canvas = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((12, 12), f"t={t:.3f}", fill=(0, 0, 0), font=font)
    draw.text((12, 28), "rows: w, theta, zeta", fill=(0, 0, 0), font=font)

    for col, label in enumerate(depth_labels):
        x = left_pad + col * (tile_size + gap) + 80
        draw.text((x, 12), label, fill=(0, 0, 0), font=font)

    for row, (name, field3d) in enumerate(fields):
        y0 = top_pad + row * (tile_size + gap)
        draw.text((12, y0 + tile_size // 2), name, fill=(0, 0, 0), font=font)
        vmax = _symmetric_limits(field3d[depth_idx])
        draw.text((12, y0 + tile_size // 2 + 14), f"|max|={vmax:.2e}", fill=(80, 80, 80), font=font)
        for col, idx in enumerate(depth_idx):
            tile = _resize_tile(_diverging_rgb(field3d[idx], vmax), tile_size)
            x0 = left_pad + col * (tile_size + gap)
            canvas.paste(tile, (x0, y0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _log_compress(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.maximum(arr, 1e-300)
    return np.log10(arr)


def _heatmap_rgb(data):
    finite = np.isfinite(data)
    if not np.any(finite):
        data = np.zeros_like(data)
        finite = np.ones_like(data, dtype=bool)
    vmin = float(np.min(data[finite]))
    vmax = float(np.max(data[finite]))
    if vmax <= vmin:
        vmax = vmin + 1.0
    scaled = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = np.empty(data.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = (255 * scaled).astype(np.uint8)
    rgb[..., 1] = (255 * np.sqrt(scaled)).astype(np.uint8)
    rgb[..., 2] = (255 * (1.0 - scaled)).astype(np.uint8)
    return rgb, vmin, vmax


def save_spectrum_history_png(data, x_values, t_values, out_path, title, xlabel):
    data = np.asarray(data, dtype=np.float64)
    x_values = np.asarray(x_values, dtype=np.float64)
    t_values = np.asarray(t_values, dtype=np.float64)
    log_data = _log_compress(data)
    rgb, vmin, vmax = _heatmap_rgb(log_data)
    tile = Image.fromarray(rgb, mode="RGB").resize((900, 520), resample=Image.Resampling.BICUBIC)

    left_pad = 90
    right_pad = 20
    top_pad = 55
    bottom_pad = 70
    width = left_pad + tile.size[0] + right_pad
    height = top_pad + tile.size[1] + bottom_pad
    canvas = Image.new("RGB", (width, height), color=(250, 250, 250))
    canvas.paste(tile, (left_pad, top_pad))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((12, 12), title, fill=(0, 0, 0), font=font)
    draw.text((12, 28), f"log10 scale: [{vmin:.1f}, {vmax:.1f}]", fill=(80, 80, 80), font=font)
    draw.text((width // 2 - 40, height - 24), xlabel, fill=(0, 0, 0), font=font)
    draw.text((12, top_pad + tile.size[1] // 2), "time", fill=(0, 0, 0), font=font)

    for frac in [0.0, 0.5, 1.0]:
        x = left_pad + int(frac * (tile.size[0] - 1))
        idx = min(len(x_values) - 1, max(0, int(round(frac * (len(x_values) - 1)))))
        draw.text((x - 12, height - 44), f"{x_values[idx]:.2f}", fill=(0, 0, 0), font=font)
    for frac in [0.0, 0.5, 1.0]:
        y = top_pad + int(frac * (tile.size[1] - 1))
        idx = min(len(t_values) - 1, max(0, int(round(frac * (len(t_values) - 1)))))
        draw.text((12, y - 6), f"{t_values[idx]:.2f}", fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    args = _parse_args()

    nonlinear_suffix = "" if args.nonlinear_advection == "jacobian" else f"_{args.nonlinear_advection}form"
    output_dir = normalize_output_dir(args.output_dir or (
        f"output_miquel_zero_tilt_{args.thermal_closure}{nonlinear_suffix}_Nx{args.Nx}_Nz{args.Nz}"
    ))
    cfg = NHQGConfig(
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
        vertical_dealiasing=args.vertical_dealiasing,
        nu_q=1.0,
        nu_w=1.0,
        nu_theta=1.0,
        hyper_order=1,
        q_boundary="none",
        save_interval=args.save_every,
        output_dir=output_dir,
        float_dtype="float64",
        thermal_closure=args.thermal_closure,
        mean_temp_eps_sq=args.mean_temp_eps_sq,
        mean_exchange_discretization=args.mean_exchange_discretization,
        sbp_transfer_mode=args.sbp_transfer_mode,
        sbp_corrector_substeps=args.sbp_corrector_substeps,
        nonlinear_advection=args.nonlinear_advection,
        horizontal_dealiasing=args.horizontal_dealiasing,
    )

    snapshot_times = list(args.snapshot_times)
    if args.snapshot_dt is not None:
        n_snaps = int(math.floor(cfg.t_final / args.snapshot_dt + 1e-12))
        snapshot_times.extend([args.snapshot_dt * i for i in range(1, n_snaps + 1)])
    snapshot_steps = sorted({
        max(1, int(round(t / cfg.dt)))
        for t in snapshot_times
        if 0.0 < t <= cfg.t_final + 1e-12
    })
    snapshot_step_set = set(snapshot_steps)

    print("=== Zero-Tilt Miquel Comparison ===")
    print(f"Nx={cfg.Nx} Nz={cfg.Nz} L={cfg.L:.4f} ({cfg.L / L_C:.1f} Lc)")
    print(f"Ra={cfg.Ra_tilde} dt={cfg.dt} t_final={cfg.t_final}")
    print(f"imex_scheme={cfg.imex_scheme}")
    print(
        f"thermal_closure={cfg.thermal_closure} "
        f"mean_temp_eps_sq={cfg.mean_temp_eps_sq}"
    )
    print(f"mean_exchange_discretization={cfg.mean_exchange_discretization}")
    print(f"sbp_transfer_mode={cfg.sbp_transfer_mode}")
    print(f"sbp_corrector_substeps={cfg.sbp_corrector_substeps}")
    print(f"nonlinear_advection={cfg.nonlinear_advection}")
    print(f"horizontal_dealiasing={cfg.horizontal_dealiasing}")
    print(f"vertical_dealiasing={cfg.vertical_dealiasing}")
    print(f"beta={cfg.beta} Ld=inf q_bc={cfg.q_boundary}")
    print(f"Output: {output_dir}")
    print(f"Device: {jax.devices()}")
    print(f"Snapshots at t={snapshot_times}")
    print(f"skip_inline_panels={args.skip_inline_panels}")
    print(flush=True)

    t0 = time.time()
    grid = make_grid(cfg)
    state = make_initial_state(grid, seed=0, amplitude=args.amplitude)
    step_fn = jax.jit(lambda s: imex_step(s, grid), donate_argnums=(0,))
    state = step_fn(state)
    jax.block_until_ready(state.q_hat)
    print(f"Compile+warmup: {time.time() - t0:.1f}s", flush=True)

    output_path = Path(output_dir)
    panel_dir = output_path / "panels"
    spectrum_dir = output_path / "spectra"
    total_steps = int(round(cfg.t_final / cfg.dt))
    start = time.time()
    diag_history = {
        "t": [],
        "step": [],
        "Nu": [],
        "Nu_dealiased": [],
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
    k_bins = None

    for step in range(1, total_steps + 1):
        state = step_fn(state)

        if step % cfg.save_interval == 0 or step in snapshot_step_set or step == total_steps:
            jax.block_until_ready(state.q_hat)
            t_sim = step * cfg.dt
            diag = compute_diagnostics(state, grid)
            th_bar_max = float(jnp.max(jnp.abs(state.th_bar)))
            print(
                f"step={step:7d} t={t_sim:5.2f} "
                f"Nu_d={float(diag['Nusselt_dealiased']):8.3f} "
                f"Nu_raw={float(diag['Nusselt']):8.3f} "
                f"max_v={float(diag['max_speed']):8.3f} "
                f"max_w={float(diag['max_w']):8.3f} "
                f"max_th={float(diag['max_theta']):8.3f} "
                f"tw_d={float(diag['vol_avg_tw_dealiased']):10.3e} "
                f"tw_raw={float(diag['vol_avg_tw']):10.3e} "
                f"R_ex_d={float(diag['mean_theta_exchange_residual_dealiased']):10.3e} "
                f"R_ex_sbp={float(diag['mean_theta_exchange_residual_sbp']):10.3e} "
                f"tw_max={float(diag['max_tw']):10.3e} "
                f"KE_bt={float(diag['KE_bt']):10.3e} "
                f"KE_bc={float(diag['KE_bc']):10.3e} "
                f"|th_bar|={th_bar_max:8.3e} "
                f"wall={time.time() - start:6.1f}s",
                flush=True,
            )
            diag_history["t"].append(t_sim)
            diag_history["step"].append(step)
            diag_history["Nu"].append(float(diag["Nusselt"]))
            diag_history["Nu_dealiased"].append(float(diag["Nusselt_dealiased"]))
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
            if k_bins is None:
                k_bins = np.array(diag["k_bins"])

        if step in snapshot_step_set:
            t_sim = step * cfg.dt
            nc_path = save_snapshot(state, t_sim, step, cfg, grid, output_dir)
            print(f"  saved {nc_path}", flush=True)
            if not args.skip_inline_panels:
                png_path = panel_dir / f"panel_{step:08d}.png"
                save_panel_png(state, grid, png_path, t_sim, args.depths)
                print(f"  saved {png_path}", flush=True)

        if step % args.checkpoint_every == 0 or step == total_steps:
            ckpt_path = save_checkpoint(state, step, cfg, output_dir)
            print(f"  checkpoint {ckpt_path}", flush=True)

    if diag_history["t"]:
        spec_npz = spectrum_dir / "spectrum_history.npz"
        spectrum_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            spec_npz,
            t=np.array(diag_history["t"], dtype=np.float64),
            step=np.array(diag_history["step"], dtype=np.int64),
            Nusselt=np.array(diag_history["Nu"], dtype=np.float64),
            Nusselt_dealiased=np.array(diag_history["Nu_dealiased"], dtype=np.float64),
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
            Nusselt_raw=np.array(diag_history["Nu"], dtype=np.float64),
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
            vert_mode=np.arange(len(diag_history["q_vert_spec"][0]), dtype=np.int64),
            k_bins=k_bins.astype(np.float64),
        )
        save_spectrum_history_png(
            np.stack(diag_history["w_vert_spec"]),
            np.arange(len(diag_history["w_vert_spec"][0])),
            np.array(diag_history["t"]),
            spectrum_dir / "w_vertical_spectrum_vs_time.png",
            "w Vertical Spectrum vs Time",
            "Chebyshev mode n",
        )
        save_spectrum_history_png(
            np.stack(diag_history["w_horiz_spec"]),
            k_bins,
            np.array(diag_history["t"]),
            spectrum_dir / "w_horizontal_spectrum_vs_time.png",
            "w Horizontal Spectrum vs Time",
            "horizontal wavenumber k",
        )
        print(f"  saved {spec_npz}", flush=True)
        print(f"  saved {spectrum_dir / 'w_vertical_spectrum_vs_time.png'}", flush=True)
        print(f"  saved {spectrum_dir / 'w_horizontal_spectrum_vs_time.png'}", flush=True)

    print(f"Completed in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
