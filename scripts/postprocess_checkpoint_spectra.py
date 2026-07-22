#!/usr/bin/env python
"""Build side-by-side spectrum histories from saved checkpoints.

This reconstructs vertical Chebyshev-mode spectra and horizontal shell
spectra for q, w, and theta from existing checkpoint files. It is intended
for comparing the matched fixed_conduction and evolve_mean runs without
rerunning the solver.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import math
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.io import load_checkpoint
from nhqg.paths import normalize_output_dir, resolve_existing_output_path
from nhqg.solver import _dirichlet_to_cheb, _to_nodal


K_C = 1.3048
L_C = 2.0 * math.pi / K_C


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixed-dir",
        type=str,
        default="output/output_miquel_zero_tilt_galerkin_ars222_fixedconduction_blas1_Nx128_Nz128_dt5e5_t5",
    )
    parser.add_argument(
        "--evolve-dir",
        type=str,
        default="output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_blas1_Nx128_Nz128_dt5e5_t5",
    )
    parser.add_argument("--output-dir", type=str, default="postprocess_checkpoint_spectra_matched_closure")
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Nz", type=int, default=128)
    parser.add_argument("--Ra", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--t-final", type=float, default=5.0)
    parser.add_argument("--imex-scheme", type=str, default="ars222")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth checkpoint")
    parser.add_argument("--t-min", type=float, default=None, help="Optional lower time cutoff")
    parser.add_argument("--t-max", type=float, default=None, help="Optional upper time cutoff")
    return parser.parse_args()


def _build_config(args: argparse.Namespace, closure: str, output_dir: str) -> NHQGConfig:
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
        nu_q=1.0,
        nu_w=1.0,
        nu_theta=1.0,
        hyper_order=1,
        thermal_closure=closure,
        mean_temp_eps_sq=1.0,
        float_dtype="float64",
        output_dir=output_dir,
    )


def _log_compress(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.maximum(arr, 1e-300)
    return np.log10(arr)


def _heatmap_rgb(data, vmin, vmax):
    scaled = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = np.empty(data.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = (255 * scaled).astype(np.uint8)
    rgb[..., 1] = (255 * np.sqrt(scaled)).astype(np.uint8)
    rgb[..., 2] = (255 * (1.0 - scaled)).astype(np.uint8)
    return rgb


def _collect_checkpoint_paths(run_dir: Path, stride: int, t_min: float | None, t_max: float | None):
    paths = sorted(run_dir.glob("checkpoint_*.npz"))
    if stride > 1:
        paths = paths[::stride]
    if t_min is None and t_max is None:
        return paths

    selected = []
    for path in paths:
        data = np.load(path)
        t = float(data["t"])
        if t_min is not None and t < t_min - 1e-12:
            continue
        if t_max is not None and t > t_max + 1e-12:
            continue
        selected.append(path)
    return selected


def _compute_checkpoint_spectra(path: Path, grid):
    state, step, t = load_checkpoint(str(path), dtype=jnp.complex128)
    w_cheb = _dirichlet_to_cheb(state.w_hat, grid.dirichlet_stencil)
    th_cheb = _dirichlet_to_cheb(state.th_hat, grid.dirichlet_stencil)
    q_nodal = _to_nodal(state.q_hat, grid.V)
    w_nodal = _to_nodal(w_cheb, grid.V)
    th_nodal = _to_nodal(th_cheb, grid.V)

    k_bins, q_horiz = depth_averaged_shell_spectrum(q_nodal, grid.ksq, grid.cc_weights, float(grid.L))
    _, w_horiz = depth_averaged_shell_spectrum(w_nodal, grid.ksq, grid.cc_weights, float(grid.L))
    _, th_horiz = depth_averaged_shell_spectrum(th_nodal, grid.ksq, grid.cc_weights, float(grid.L))

    return {
        "t": float(t),
        "step": int(step),
        "q_vert": np.array(horizontal_avg_vertical_spectrum(state.q_hat), dtype=np.float64),
        "w_vert": np.array(horizontal_avg_vertical_spectrum(w_cheb), dtype=np.float64),
        "th_vert": np.array(horizontal_avg_vertical_spectrum(th_cheb), dtype=np.float64),
        "q_horiz": np.array(q_horiz, dtype=np.float64),
        "w_horiz": np.array(w_horiz, dtype=np.float64),
        "th_horiz": np.array(th_horiz, dtype=np.float64),
        "k_bins": np.array(k_bins, dtype=np.float64),
    }


def horizontal_avg_vertical_spectrum(field_hat: jnp.ndarray):
    """Pure vertical spectrum: horizontal average of modal power at each n."""
    nk = field_hat.shape[2]
    weight = jnp.ones((field_hat.shape[1], nk), dtype=field_hat.real.dtype)
    weight = weight.at[:, 1:nk - 1].set(2.0)
    norm = jnp.sum(weight)
    return jnp.sum(jnp.abs(field_hat) ** 2 * weight[None, :, :], axis=(1, 2)) / norm


def depth_averaged_shell_spectrum(field_nodal: jnp.ndarray, ksq: jnp.ndarray,
                                  cc_weights: jnp.ndarray, L: float):
    """Shell spectrum at each depth, then average vertically with CC weights.

    This makes the intended diagnostic explicit. Because shell binning is linear,
    it is algebraically equivalent to vertical integration before shell summation.
    """
    nk = field_nodal.shape[2]
    k_mag = jnp.sqrt(ksq)
    dk = 2.0 * jnp.pi / L
    k_max = jnp.sqrt(jnp.max(ksq))
    n_bins = int(float(k_max / dk)) + 1
    k_bins = jnp.arange(n_bins) * dk + dk / 2

    weight = jnp.ones_like(ksq)
    weight = weight.at[:, 1:nk - 1].set(2.0)
    power = jnp.abs(field_nodal) ** 2 * weight[None, :, :]

    depth_shells = []
    for i in range(n_bins):
        mask = ((k_mag >= i * dk) & (k_mag < (i + 1) * dk))[None, :, :]
        depth_shells.append(jnp.sum(jnp.where(mask, power, 0.0), axis=(1, 2)))
    depth_shells = jnp.stack(depth_shells, axis=1)
    spec = jnp.einsum("z,zs->s", cc_weights, depth_shells)
    return k_bins, spec


def _build_history(run_dir: Path, grid, stride: int, t_min: float | None, t_max: float | None):
    paths = _collect_checkpoint_paths(run_dir, stride, t_min, t_max)
    if not paths:
        raise ValueError(f"No checkpoints found in {run_dir}")

    rows = [_compute_checkpoint_spectra(path, grid) for path in paths]
    return {
        "t": np.array([row["t"] for row in rows], dtype=np.float64),
        "step": np.array([row["step"] for row in rows], dtype=np.int64),
        "q_vert": np.stack([row["q_vert"] for row in rows]).astype(np.float64),
        "w_vert": np.stack([row["w_vert"] for row in rows]).astype(np.float64),
        "th_vert": np.stack([row["th_vert"] for row in rows]).astype(np.float64),
        "q_horiz": np.stack([row["q_horiz"] for row in rows]).astype(np.float64),
        "w_horiz": np.stack([row["w_horiz"] for row in rows]).astype(np.float64),
        "th_horiz": np.stack([row["th_horiz"] for row in rows]).astype(np.float64),
        "k_bins": rows[0]["k_bins"],
        "vert_mode": np.arange(rows[0]["q_vert"].shape[0], dtype=np.int64),
    }


def _draw_ticks(draw, font, left, top, tile_w, tile_h, x_values, t_values):
    for frac in [0.0, 0.5, 1.0]:
        x = left + int(frac * (tile_w - 1))
        idx = min(len(x_values) - 1, max(0, int(round(frac * (len(x_values) - 1)))))
        draw.text((x - 14, top + tile_h + 8), f"{x_values[idx]:.2f}", fill=(0, 0, 0), font=font)
    for frac in [0.0, 0.5, 1.0]:
        y = top + int(frac * (tile_h - 1))
        idx = min(len(t_values) - 1, max(0, int(round(frac * (len(t_values) - 1)))))
        draw.text((12, y - 6), f"{t_values[idx]:.2f}", fill=(0, 0, 0), font=font)


def save_comparison_png(
    fixed_data,
    evolve_data,
    x_values,
    t_values,
    out_path: Path,
    title: str,
    xlabel: str,
):
    fixed_log = _log_compress(fixed_data)
    evolve_log = _log_compress(evolve_data)
    combined = np.concatenate([fixed_log.ravel(), evolve_log.ravel()])
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if vmax <= vmin:
            vmax = vmin + 1.0

    tile_w = 780
    tile_h = 520
    gap = 36
    left_pad = 90
    right_pad = 20
    top_pad = 60
    bottom_pad = 70
    width = left_pad + tile_w + gap + tile_w + right_pad
    height = top_pad + tile_h + bottom_pad
    canvas = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    fixed_tile = Image.fromarray(_heatmap_rgb(fixed_log, vmin, vmax), mode="RGB").resize(
        (tile_w, tile_h), resample=Image.Resampling.BICUBIC
    )
    evolve_tile = Image.fromarray(_heatmap_rgb(evolve_log, vmin, vmax), mode="RGB").resize(
        (tile_w, tile_h), resample=Image.Resampling.BICUBIC
    )

    x0 = left_pad
    x1 = left_pad + tile_w + gap
    y0 = top_pad
    canvas.paste(fixed_tile, (x0, y0))
    canvas.paste(evolve_tile, (x1, y0))

    draw.text((12, 12), title, fill=(0, 0, 0), font=font)
    draw.text((12, 28), f"shared log10 scale: [{vmin:.1f}, {vmax:.1f}]", fill=(80, 80, 80), font=font)
    draw.text((x0 + tile_w // 2 - 50, 40), "fixed_conduction", fill=(0, 0, 0), font=font)
    draw.text((x1 + tile_w // 2 - 38, 40), "evolve_mean", fill=(0, 0, 0), font=font)
    draw.text((width // 2 - 45, height - 24), xlabel, fill=(0, 0, 0), font=font)
    draw.text((12, y0 + tile_h // 2), "time", fill=(0, 0, 0), font=font)

    _draw_ticks(draw, font, x0, y0, tile_w, tile_h, x_values, t_values)
    _draw_ticks(draw, font, x1, y0, tile_w, tile_h, x_values, t_values)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _save_npz(out_path: Path, fixed_hist, evolve_hist):
    np.savez(
        out_path,
        t=fixed_hist["t"],
        step=fixed_hist["step"],
        vert_mode=fixed_hist["vert_mode"],
        k_bins=fixed_hist["k_bins"],
        fixed_q_vert=fixed_hist["q_vert"],
        fixed_w_vert=fixed_hist["w_vert"],
        fixed_th_vert=fixed_hist["th_vert"],
        fixed_q_horiz=fixed_hist["q_horiz"],
        fixed_w_horiz=fixed_hist["w_horiz"],
        fixed_th_horiz=fixed_hist["th_horiz"],
        evolve_q_vert=evolve_hist["q_vert"],
        evolve_w_vert=evolve_hist["w_vert"],
        evolve_th_vert=evolve_hist["th_vert"],
        evolve_q_horiz=evolve_hist["q_horiz"],
        evolve_w_horiz=evolve_hist["w_horiz"],
        evolve_th_horiz=evolve_hist["th_horiz"],
    )


def main():
    args = _parse_args()

    fixed_dir = resolve_existing_output_path(args.fixed_dir)
    evolve_dir = resolve_existing_output_path(args.evolve_dir)
    output_dir = Path(normalize_output_dir(args.output_dir))

    print("=== Checkpoint Spectrum Postprocess ===", flush=True)
    print(f"fixed_dir={fixed_dir}", flush=True)
    print(f"evolve_dir={evolve_dir}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"stride={args.stride} t_min={args.t_min} t_max={args.t_max}", flush=True)
    print("Using checkpoint cadence as the analysis interval.", flush=True)

    fixed_cfg = _build_config(args, "fixed_conduction", str(fixed_dir))
    evolve_cfg = _build_config(args, "evolve_mean", str(evolve_dir))

    print("Building grids...", flush=True)
    fixed_grid = make_grid(fixed_cfg)
    evolve_grid = make_grid(evolve_cfg)

    print("Reading fixed_conduction checkpoints...", flush=True)
    fixed_hist = _build_history(fixed_dir, fixed_grid, args.stride, args.t_min, args.t_max)
    print("Reading evolve_mean checkpoints...", flush=True)
    evolve_hist = _build_history(evolve_dir, evolve_grid, args.stride, args.t_min, args.t_max)

    if not np.allclose(fixed_hist["t"], evolve_hist["t"]):
        raise ValueError("Time grids do not match between fixed and evolve histories")
    if not np.array_equal(fixed_hist["step"], evolve_hist["step"]):
        raise ValueError("Checkpoint steps do not match between fixed and evolve histories")

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "matched_checkpoint_spectra.npz"
    _save_npz(npz_path, fixed_hist, evolve_hist)
    print(f"saved {npz_path}", flush=True)

    tasks = [
        (
            "q",
            "vertical_avgxy",
            fixed_hist["q_vert"],
            evolve_hist["q_vert"],
            fixed_hist["vert_mode"],
            "Chebyshev mode n",
            "q Pure Vertical Spectrum vs Time (avg over horizontal modes)",
        ),
        (
            "w",
            "vertical_avgxy",
            fixed_hist["w_vert"],
            evolve_hist["w_vert"],
            fixed_hist["vert_mode"],
            "Chebyshev mode n",
            "w Pure Vertical Spectrum vs Time (avg over horizontal modes)",
        ),
        (
            "theta",
            "vertical_avgxy",
            fixed_hist["th_vert"],
            evolve_hist["th_vert"],
            fixed_hist["vert_mode"],
            "Chebyshev mode n",
            "theta Pure Vertical Spectrum vs Time (avg over horizontal modes)",
        ),
        (
            "q",
            "horizontal_depthavg",
            fixed_hist["q_horiz"],
            evolve_hist["q_horiz"],
            fixed_hist["k_bins"],
            "horizontal wavenumber k",
            "q Horizontal Shell Spectrum vs Time (shell at each depth, then avg in z)",
        ),
        (
            "w",
            "horizontal_depthavg",
            fixed_hist["w_horiz"],
            evolve_hist["w_horiz"],
            fixed_hist["k_bins"],
            "horizontal wavenumber k",
            "w Horizontal Shell Spectrum vs Time (shell at each depth, then avg in z)",
        ),
        (
            "theta",
            "horizontal_depthavg",
            fixed_hist["th_horiz"],
            evolve_hist["th_horiz"],
            fixed_hist["k_bins"],
            "horizontal wavenumber k",
            "theta Horizontal Shell Spectrum vs Time (shell at each depth, then avg in z)",
        ),
    ]

    for field_name, spectrum_kind, fixed_data, evolve_data, x_values, xlabel, title in tasks:
        out_path = output_dir / f"{field_name}_{spectrum_kind}_comparison.png"
        save_comparison_png(
            fixed_data,
            evolve_data,
            x_values,
            fixed_hist["t"],
            out_path,
            title,
            xlabel,
        )
        print(f"saved {out_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
