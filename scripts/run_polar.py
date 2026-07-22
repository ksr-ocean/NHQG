#!/usr/bin/env python
"""Run a polar-cap NHQG case with snapshots, checkpoints, and CSV diagnostics."""

import argparse
import csv
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.solver import State, make_initial_state, imex_step
from nhqg.io import save_checkpoint, save_snapshot, load_checkpoint
from nhqg.diagnostics import compute_diagnostics


K_C = 1.3048


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Nx", required=True, type=int)
    parser.add_argument("--Nz", required=True, type=int)
    parser.add_argument("--L-over-Lc", type=float, default=48.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--trap-r-star", type=float, default=None)
    parser.add_argument("--trap-sharpness", type=float, default=20.0)
    parser.add_argument("--Ra", type=float, default=100.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--Ld", type=float, default=float("inf"))
    parser.add_argument("--dt", required=True, type=float)
    parser.add_argument("--t-final", required=True, type=float)
    parser.add_argument(
        "--init", choices=["convective", "barotropic-vorticity"], default="convective"
    )
    parser.add_argument("--init-amplitude", type=float, default=1e-3)
    parser.add_argument("--init-kpeak", type=float, default=K_C)
    parser.add_argument("--init-kwidth", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nu-q", type=float, default=1.0)
    parser.add_argument("--nu-w", type=float, default=1.0)
    parser.add_argument("--nu-theta", type=float, default=1.0)
    parser.add_argument("--hyper-order", type=int, default=1)
    parser.add_argument("--drag", type=float, default=0.0)
    parser.add_argument(
        "--thermal-closure",
        choices=["fixed_conduction", "evolve_mean"],
        default="fixed_conduction",
    )
    parser.add_argument("--mean-exchange", type=str, default="balanced_sbp2_pc")
    parser.add_argument("--sbp-substeps", type=int, default=4)
    parser.add_argument("--w-bc-top", choices=["dirichlet", "neumann"], default="dirichlet")
    parser.add_argument("--imex-scheme", choices=["ars222", "rk443"], default="ars222")
    parser.add_argument("--advection", choices=["flux", "jacobian"], default="flux")
    parser.add_argument("--dealias", choices=["23_rule", "32_rule"], default="23_rule")
    parser.add_argument("--imex-matmul-chunk", type=int, default=0)
    parser.add_argument("--shard-axis", choices=["none", "z", "kx"], default="none")
    parser.add_argument("--shard-devices", type=int, default=None,
                        help="number of devices for --shard-axis (default: all visible)")
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--snapshot-interval", type=float, default=0.25)
    parser.add_argument("--checkpoint-interval", type=float, default=1.0)
    parser.add_argument("--diag-interval", type=float, default=0.05)
    parser.add_argument("--restart-checkpoint", type=str, default=None)
    return parser.parse_args()


def _make_barotropic_vorticity_state(args, grid, L):
    """Construct the specified annular, barotropic vorticity initial state."""
    rng = np.random.default_rng(args.seed)
    zeta = rng.standard_normal((args.Nx, args.Nx))
    zhat = np.fft.rfft2(zeta)

    kx = 2.0 * np.pi * np.fft.fftfreq(args.Nx, d=L / args.Nx)[:, None]
    ky = 2.0 * np.pi * np.arange(args.Nx // 2 + 1)[None, :] / L
    kmag = np.sqrt(kx ** 2 + ky ** 2)
    annulus = np.exp(-0.5 * ((kmag - args.init_kpeak) / args.init_kwidth) ** 2)
    annulus[0, 0] = 0.0

    kx_int = np.rint(np.fft.fftfreq(args.Nx, d=1.0 / args.Nx)).astype(int)[:, None]
    ky_int = np.arange(args.Nx // 2 + 1, dtype=int)[None, :]
    mask = (np.abs(kx_int) <= args.Nx // 3) & (ky_int <= args.Nx // 3)
    zeta = np.fft.irfft2(zhat * annulus * mask, s=(args.Nx, args.Nx))
    zeta *= args.init_amplitude / zeta.std()
    zhat_final = np.fft.rfft2(zeta)

    q_hat = np.zeros((args.Nz + 1, args.Nx, args.Nx // 2 + 1), dtype=np.complex128)
    q_hat[0] = zhat_final
    gal_shape = (args.Nz - 1, args.Nx, args.Nx // 2 + 1)
    zeros_gal = np.zeros(gal_shape, dtype=np.complex128)
    return State(
        jnp.asarray(q_hat),
        jnp.asarray(zeros_gal),
        jnp.asarray(zeros_gal),
        jnp.asarray(np.zeros(args.Nz + 1, dtype=np.float64)),
    )


def _diverging_rgb(field, vmax):
    """Map a scalar image to a linear blue-white-red RGB image."""
    scaled = np.clip(field / vmax, -1.0, 1.0)
    magnitude = np.abs(scaled)
    rgb = np.empty(field.shape + (3,), dtype=np.uint8)
    negative = scaled < 0.0
    rgb[..., 0] = 255
    rgb[..., 1] = (255.0 * (1.0 - magnitude)).astype(np.uint8)
    rgb[..., 2] = (255.0 * (1.0 - magnitude)).astype(np.uint8)
    rgb[negative, 0] = (255.0 * (1.0 - magnitude[negative])).astype(np.uint8)
    rgb[negative, 1] = (255.0 * (1.0 - magnitude[negative])).astype(np.uint8)
    rgb[negative, 2] = 255
    return rgb


def _save_zeta_bt_png(state, grid, nx, path, t):
    q_nodal = np.einsum("ij,j...->i...", np.array(grid.V), np.array(state.q_hat))
    zeta_bt_hat = np.einsum("i,i...->...", np.array(grid.cc_weights), q_nodal)
    zeta_bt = np.fft.irfft2(zeta_bt_hat, s=(nx, nx))
    vmax = max(float(np.max(np.abs(zeta_bt))), 1e-12)
    image = Image.fromarray(_diverging_rgb(zeta_bt, vmax), mode="RGB").resize(
        (720, 720), resample=Image.Resampling.BICUBIC
    )
    ImageDraw.Draw(image).text(
        (8, 8), f"zeta_bt t={t:.3f}", fill=(0, 0, 0), font=ImageFont.load_default()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


_FIXED_CONDUCTION_SBP_DIAGNOSTICS = frozenset({
    "th_mean_feedback_sum_sbp",
    "mean_flux_exchange_tendency_sbp",
    "mean_theta_exchange_boundary_sbp",
    "mean_theta_exchange_residual_sbp",
    "mean_theta_exchange_residual_sbp_rel",
})


def _scalar_diagnostics(diag, thermal_closure):
    """Select the scalar diagnostics intended for the CSV time series.

    The current solver additionally exposes spectra and shell arrays through
    ``compute_diagnostics``.  Its SBP exchange audit is undefined for the
    fixed-conduction closure, so omit those non-applicable values as well.
    """
    ignored = (
        _FIXED_CONDUCTION_SBP_DIAGNOSTICS
        if thermal_closure == "fixed_conduction"
        else frozenset()
    )
    return {
        key: float(np.asarray(value))
        for key, value in diag.items()
        if key not in ignored and np.asarray(value).ndim == 0
    }


def _diagnostics_are_finite(diag):
    return all(np.isfinite(value) for value in diag.values())


def _write_diagnostics_row(path, step, t, diag):
    keys = sorted(diag)
    new_file = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if new_file:
            writer.writerow(["step", "t", *keys])
        writer.writerow([step, t, *(diag[key] for key in keys)])


def main():
    args = _parse_args()
    L = args.L_over_Lc * 2.0 * math.pi / K_C
    cfg = NHQGConfig(
        Nx=args.Nx,
        Nz=args.Nz,
        L=L,
        Ra_tilde=args.Ra,
        sigma=args.sigma,
        beta=0.0,
        gamma=args.gamma,
        trap_r_star=args.trap_r_star,
        trap_sharpness=args.trap_sharpness,
        Ld=args.Ld,
        dt=args.dt,
        t_final=args.t_final,
        float_dtype="float64",
        thermal_closure=args.thermal_closure,
        w_bc_top=args.w_bc_top,
        mean_exchange_discretization=args.mean_exchange,
        sbp_corrector_substeps=args.sbp_substeps,
        nonlinear_advection=args.advection,
        horizontal_dealiasing=args.dealias,
        nu_q=args.nu_q,
        nu_w=args.nu_w,
        nu_theta=args.nu_theta,
        hyper_order=args.hyper_order,
        drag=args.drag,
        imex_matmul_chunk=args.imex_matmul_chunk,
        q_boundary="none",
        imex_scheme=args.imex_scheme,
        output_dir=args.output_dir,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = make_grid(cfg)

    if args.restart_checkpoint is not None:
        state, step0, t0 = load_checkpoint(args.restart_checkpoint)
    else:
        step0, t0 = 0, 0.0
        if args.init == "convective":
            state = make_initial_state(grid, seed=args.seed, amplitude=args.init_amplitude)
        else:
            state = _make_barotropic_vorticity_state(args, grid, L)

    if args.shard_axis != "none":
        from nhqg.sharding import make_mesh, shard_state
        mesh = make_mesh(args.shard_devices)
        print(f"sharding state axis={args.shard_axis} over {len(mesh.devices.ravel())} devices")
        state = shard_state(state, mesh, args.shard_axis)

    n_total = int(round(args.t_final / args.dt))
    if step0 >= n_total:
        print(f"checkpoint already at step={step0}, target step={n_total}; nothing to do")
        return 0

    diag_steps = max(1, int(round(args.diag_interval / args.dt)))
    snapshot_steps = int(round(args.snapshot_interval / args.dt))
    checkpoint_steps = int(round(args.checkpoint_interval / args.dt))
    diagnostics_path = output_dir / "diagnostics.csv"
    start_wall = time.time()

    @jax.jit(static_argnames=("n_steps",))
    def step_chunk(current_state, n_steps):
        def scan_body(carry, _):
            return imex_step(carry, grid), None

        return jax.lax.scan(scan_body, current_state, xs=None, length=n_steps)[0]

    step = step0
    last_diag = None
    while step < n_total:
        n_steps = min(diag_steps, n_total - step)
        state = step_chunk(state, n_steps)
        jax.block_until_ready(state.q_hat)
        step += n_steps
        t = t0 + (step - step0) * args.dt

        diag = _scalar_diagnostics(compute_diagnostics(state, grid), args.thermal_closure)
        _write_diagnostics_row(diagnostics_path, step, t, diag)
        last_diag = diag
        if not _diagnostics_are_finite(diag):
            save_checkpoint(state, step, cfg, str(output_dir))
            print(f"NON-FINITE DIAGNOSTICS at t={t}")
            return 2

        if snapshot_steps > 0 and step % snapshot_steps == 0:
            save_snapshot(state, t, step, cfg, grid, str(output_dir))
            _save_zeta_bt_png(state, grid, args.Nx, output_dir / "png" / f"zeta_bt_{step:08d}.png", t)

        if checkpoint_steps > 0 and step % checkpoint_steps == 0:
            save_checkpoint(state, step, cfg, str(output_dir))

    save_checkpoint(state, step, cfg, str(output_dir))
    summary_key = "KE_tot" if last_diag is not None and "KE_tot" in last_diag else sorted(last_diag)[0]
    print(f"final t={t0 + (step - step0) * args.dt:.6f} {summary_key}={float(np.asarray(last_diag[summary_key])):.6e} wall={time.time() - start_wall:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
