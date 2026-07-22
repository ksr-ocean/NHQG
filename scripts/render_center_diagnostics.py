#!/usr/bin/env python
"""Render a 2x3 center-plane diagnostic figure from a checkpoint/snapshot pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.io import load_checkpoint
from nhqg.solver import _dirichlet_to_cheb, invert_psi


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tile-size", type=int, default=280)
    return parser.parse_args()


def _jet_rgb(field2d: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    vmin, vmax = limits
    denom = max(vmax - vmin, 1e-12)
    scaled = np.clip((field2d - vmin) / denom, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * scaled - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * scaled - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * scaled - 1.0), 0.0, 1.0)
    return np.stack(
        [(255 * r).astype(np.uint8), (255 * g).astype(np.uint8), (255 * b).astype(np.uint8)],
        axis=-1,
    )


def _bwr_rgb(field2d: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    vmin, vmax = limits
    vmax_eff = max(abs(vmin), abs(vmax), 1e-12)
    scaled = np.clip(field2d / vmax_eff, -1.0, 1.0)
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


def _map_rgb(field2d: np.ndarray, cmap: str, limits: tuple[float, float]) -> np.ndarray:
    if cmap == "bwr":
        return _bwr_rgb(field2d, limits)
    if cmap == "jet":
        return _jet_rgb(field2d, limits)
    raise ValueError(f"Unsupported colormap {cmap!r}")


def _colorbar_rgb(cmap: str, limits: tuple[float, float], height: int, width: int) -> np.ndarray:
    ramp = np.linspace(limits[1], limits[0], height, dtype=np.float64)[:, None]
    ramp = np.repeat(ramp, width, axis=1)
    return _map_rgb(ramp, cmap, limits)


def _resize_tile(rgb: np.ndarray, size: int) -> Image.Image:
    return Image.fromarray(rgb, mode="RGB").resize((size, size), resample=Image.Resampling.BICUBIC)


def _symmetric_limit(field: np.ndarray, pct: float = 99.8) -> tuple[float, float]:
    vmax = float(np.percentile(np.abs(field), pct))
    vmax = max(vmax, 1e-12)
    return (-vmax, vmax)


def _positive_limit(field: np.ndarray, pct: float = 99.8) -> tuple[float, float]:
    vmax = float(np.percentile(field, pct))
    vmax = max(vmax, 1e-12)
    return (0.0, vmax)


def _theta_limit(field: np.ndarray, pct: float = 99.8) -> tuple[float, float]:
    lo = float(np.percentile(field, 0.2))
    hi = float(np.percentile(field, pct))
    vmax = max(abs(lo), abs(hi), 1e-12)
    return (-vmax, vmax)


def _build_cfg(snapshot_path: Path) -> NHQGConfig:
    with netCDF4.Dataset(snapshot_path) as ds:
        return NHQGConfig(
            Nx=int(ds.Nx),
            Nz=int(ds.Nz),
            L=float(ds.L),
            dt=float(ds.dt),
            Ra_tilde=float(ds.Ra_tilde),
            sigma=float(ds.sigma),
            beta=float(ds.beta),
            Ld=(float("inf") if float(ds.Ld) < 0 else float(ds.Ld)),
            thermal_closure=str(ds.thermal_closure),
            output_dir=str(snapshot_path.parent),
        )


def _to_physical(field_coeffs: np.ndarray, V: np.ndarray, Nx: int) -> np.ndarray:
    nodal = np.einsum("ij,j...->i...", V, field_coeffs)
    return np.fft.irfft2(nodal, s=(Nx, Nx))


def _center_fields(checkpoint_path: Path, snapshot_path: Path):
    cfg = _build_cfg(snapshot_path)
    grid = make_grid(cfg)
    state, step, t = load_checkpoint(str(checkpoint_path))

    V = np.asarray(grid.V)
    G_Z = np.asarray(grid.G_Z)
    kx = np.asarray(grid.kx)
    ky = np.asarray(grid.ky)
    z = np.asarray(grid.Z)

    q_hat = np.asarray(state.q_hat)
    w_cheb = np.asarray(_dirichlet_to_cheb(state.w_hat, grid.dirichlet_stencil))
    th_cheb = np.asarray(_dirichlet_to_cheb(state.th_hat, grid.dirichlet_stencil))
    psi_hat = np.asarray(invert_psi(state.q_hat, grid.inv_denom))

    q_phys = _to_physical(q_hat, V, cfg.Nx)
    w_phys = _to_physical(w_cheb, V, cfg.Nx)
    th_phys = _to_physical(th_cheb, V, cfg.Nx)
    dwdz_phys = _to_physical(np.einsum("ij,j...->i...", G_Z, w_cheb), V, cfg.Nx)
    dpsidz_phys = _to_physical(np.einsum("ij,j...->i...", G_Z, psi_hat), V, cfg.Nx)

    iz = int(np.argmin(np.abs(z - 0.5)))
    theta_mid = np.asarray(th_phys[iz], dtype=np.float64)
    theta_hat_xy = np.fft.rfft2(theta_mid)
    dthdx = np.fft.irfft2(1j * kx * theta_hat_xy, s=(cfg.Nx, cfg.Nx))
    dthdy = np.fft.irfft2(1j * ky * theta_hat_xy, s=(cfg.Nx, cfg.Nx))
    grad_theta = np.sqrt(dthdx**2 + dthdy**2)

    fields = {
        "vorticity": np.asarray(q_phys[iz], dtype=np.float64),
        "w": np.asarray(w_phys[iz], dtype=np.float64),
        "divergence -dw/dz": np.asarray(-dwdz_phys[iz], dtype=np.float64),
        "buoyancy dpsi/dz": np.asarray(dpsidz_phys[iz], dtype=np.float64),
        "theta": theta_mid,
        "|grad_h theta|": np.asarray(grad_theta, dtype=np.float64),
    }
    limits = {
        "vorticity": _symmetric_limit(fields["vorticity"]),
        "w": _symmetric_limit(fields["w"]),
        "divergence -dw/dz": _symmetric_limit(fields["divergence -dw/dz"]),
        "buoyancy dpsi/dz": _symmetric_limit(fields["buoyancy dpsi/dz"]),
        "theta": _theta_limit(fields["theta"]),
        "|grad_h theta|": _positive_limit(fields["|grad_h theta|"]),
    }
    cmaps = {
        "vorticity": "bwr",
        "w": "bwr",
        "divergence -dw/dz": "bwr",
        "buoyancy dpsi/dz": "bwr",
        "theta": "jet",
        "|grad_h theta|": "jet",
    }
    return cfg, step, t, float(z[iz]), fields, limits, cmaps


def _render(output_path: Path, cfg: NHQGConfig, step: int, t: float, z_mid: float,
            fields: dict[str, np.ndarray], limits: dict[str, tuple[float, float]],
            cmaps: dict[str, str], tile_size: int) -> None:
    names = list(fields.keys())
    cols = 3
    rows = 2
    left_pad = 28
    top_pad = 74
    gap_x = 18
    gap_y = 30
    bar_gap = 8
    bar_width = 20
    right_pad = 74
    width = left_pad + cols * tile_size + (cols - 1) * gap_x + cols * (bar_gap + bar_width) + right_pad
    height = top_pad + rows * tile_size + (rows - 1) * gap_y + 36

    canvas = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((left_pad, 18), f"Center-plane diagnostics at t={t:.2f}, z={z_mid:.3f}, step={step}", fill=(0, 0, 0), font=font)
    draw.text((left_pad, 36), f"Nx={cfg.Nx}, Nz={cfg.Nz}, dt={cfg.dt:.1e}", fill=(60, 60, 60), font=font)

    for idx, name in enumerate(names):
        row = idx // cols
        col = idx % cols
        x0 = left_pad + col * (tile_size + gap_x + bar_gap + bar_width)
        y0 = top_pad + row * (tile_size + gap_y)
        vmin, vmax = limits[name]
        rgb = _map_rgb(fields[name], cmaps[name], (vmin, vmax))
        canvas.paste(_resize_tile(rgb, tile_size), (x0, y0))
        draw.text((x0, y0 - 18), name, fill=(0, 0, 0), font=font)
        draw.text((x0, y0 + tile_size + 4), f"[{vmin:.2e}, {vmax:.2e}]", fill=(80, 80, 80), font=font)

        bar_rgb = _colorbar_rgb(cmaps[name], (vmin, vmax), tile_size, bar_width)
        bar_x = x0 + tile_size + bar_gap
        canvas.paste(Image.fromarray(bar_rgb, mode="RGB"), (bar_x, y0))
        draw.text((bar_x + bar_width + 4, y0), f"{vmax:.1e}", fill=(0, 0, 0), font=font)
        draw.text((bar_x + bar_width + 4, y0 + tile_size - 10), f"{vmin:.1e}", fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = _parse_args()
    checkpoint = Path(args.checkpoint)
    snapshot = Path(args.snapshot)
    output = Path(args.output)
    cfg, step, t, z_mid, fields, limits, cmaps = _center_fields(checkpoint, snapshot)
    _render(output, cfg, step, t, z_mid, fields, limits, cmaps, args.tile_size)
    print(f"saved {output}")
    for name in fields:
        print(name, limits[name])


if __name__ == "__main__":
    main()
