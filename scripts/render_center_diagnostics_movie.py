#!/usr/bin/env python
"""Render a center-plane diagnostics movie from saved snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--ray", action="store_true")
    parser.add_argument("--tile-size", type=int, default=260)
    parser.add_argument("--percentile", type=float, default=99.8)
    return parser.parse_args()


def _snapshot_files(input_dir: Path, stride: int) -> list[Path]:
    files = sorted(input_dir.glob("snapshot_*.nc"))
    if stride > 1:
        files = files[::stride]
    if not files:
        raise FileNotFoundError(f"No snapshots found in {input_dir}")
    return files


def _build_cfg(first_snapshot: Path) -> NHQGConfig:
    with netCDF4.Dataset(first_snapshot) as ds:
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
            output_dir=str(first_snapshot.parent),
        )


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


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


def _load_fields(path: Path, D_nodal: np.ndarray, kx: np.ndarray, ky: np.ndarray, iz: int):
    with netCDF4.Dataset(path) as ds:
        t = float(ds.time)
        step = int(ds.step)
        q = np.asarray(ds.variables["q_prime"][:], dtype=np.float64)
        w = np.asarray(ds.variables["w"][:], dtype=np.float64)
        theta = np.asarray(ds.variables["theta"][:], dtype=np.float64)
        psi = np.asarray(ds.variables["psi"][:], dtype=np.float64)

    dwdz = np.einsum("ij,j...->i...", D_nodal, w)
    dpsidz = np.einsum("ij,j...->i...", D_nodal, psi)

    theta_mid = theta[iz]
    theta_hat = np.fft.rfft2(theta_mid)
    dthdx = np.fft.irfft2(1j * kx * theta_hat, s=theta_mid.shape)
    dthdy = np.fft.irfft2(1j * ky * theta_hat, s=theta_mid.shape)
    grad_theta = np.sqrt(dthdx**2 + dthdy**2)

    fields = {
        "ζ": q[iz],
        "w": w[iz],
        "−∂z w": -dwdz[iz],
        "∂z ψ": dpsidz[iz],
        "θ": theta_mid,
        "|∇h θ|": grad_theta,
    }
    return t, step, fields


def _compute_limits(files: list[Path], D_nodal: np.ndarray, kx: np.ndarray, ky: np.ndarray, iz: int,
                    percentile: float) -> dict[str, tuple[float, float]]:
    names = ["ζ", "w", "−∂z w", "∂z ψ", "θ", "|∇h θ|"]
    values = {name: [] for name in names}

    for i, path in enumerate(files, start=1):
        _, _, fields = _load_fields(path, D_nodal, kx, ky, iz)
        for name, arr in fields.items():
            if name == "|∇h θ|":
                values[name].append(float(np.percentile(arr, percentile)))
            elif name == "θ":
                lo = float(np.percentile(arr, 100.0 - percentile))
                hi = float(np.percentile(arr, percentile))
                values[name].append(max(abs(lo), abs(hi)))
            else:
                values[name].append(float(np.percentile(np.abs(arr), percentile)))
        if i % 100 == 0 or i == len(files):
            print(f"limit pass {i}/{len(files)}", flush=True)

    limits = {}
    for name in names:
        vmax = max(values[name]) if values[name] else 1.0
        vmax = max(vmax, 1e-12)
        if name == "|∇h θ|":
            limits[name] = (0.0, vmax)
        else:
            limits[name] = (-vmax, vmax)
    return limits


def _render_frame(path: Path, output_dir: Path, D_nodal: np.ndarray, kx: np.ndarray, ky: np.ndarray,
                  iz: int, z_mid: float, limits: dict[str, tuple[float, float]], tile_size: int):
    t, step, fields = _load_fields(path, D_nodal, kx, ky, iz)
    names = ["ζ", "w", "−∂z w", "∂z ψ", "θ", "|∇h θ|"]
    cmaps = {name: ("jet" if name in {"θ", "|∇h θ|"} else "bwr") for name in names}

    cols = 3
    rows = 2
    left_pad = 18
    top_pad = 28
    gap_x = 16
    gap_y = 22
    bar_gap = 6
    bar_width = 18
    right_pad = 64
    bottom_pad = 14
    width = left_pad + cols * tile_size + (cols - 1) * gap_x + cols * (bar_gap + bar_width) + right_pad
    height = top_pad + rows * tile_size + (rows - 1) * gap_y + bottom_pad

    canvas = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(18)
    small_font = _load_font(15)

    draw.text((left_pad, 4), f"t={t:.2f}    z={z_mid:.3f}", fill=(0, 0, 0), font=small_font)

    for idx, name in enumerate(names):
        row = idx // cols
        col = idx % cols
        x0 = left_pad + col * (tile_size + gap_x + bar_gap + bar_width)
        y0 = top_pad + row * (tile_size + gap_y)
        rgb = _map_rgb(fields[name], cmaps[name], limits[name])
        canvas.paste(_resize_tile(rgb, tile_size), (x0, y0))
        draw.text((x0, y0 - 22), name, fill=(0, 0, 0), font=font)

        bar_rgb = _colorbar_rgb(cmaps[name], limits[name], tile_size, bar_width)
        bar_x = x0 + tile_size + bar_gap
        canvas.paste(Image.fromarray(bar_rgb, mode="RGB"), (bar_x, y0))
        vmin, vmax = limits[name]
        draw.text((bar_x + bar_width + 4, y0), f"{vmax:.1e}", fill=(0, 0, 0), font=small_font)
        draw.text((bar_x + bar_width + 4, y0 + tile_size - 14), f"{vmin:.1e}", fill=(0, 0, 0), font=small_font)

    output_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(output_dir / f"center_{step:08d}.png")


def _render_serial(files: list[Path], output_dir: Path, D_nodal: np.ndarray, kx: np.ndarray, ky: np.ndarray,
                   iz: int, z_mid: float, limits: dict[str, tuple[float, float]], tile_size: int):
    for i, path in enumerate(files, start=1):
        _render_frame(path, output_dir, D_nodal, kx, ky, iz, z_mid, limits, tile_size)
        if i % 50 == 0 or i == len(files):
            print(f"rendered {i}/{len(files)} frames", flush=True)


def _render_with_ray(files: list[Path], output_dir: Path, D_nodal: np.ndarray, kx: np.ndarray, ky: np.ndarray,
                     iz: int, z_mid: float, limits: dict[str, tuple[float, float]], tile_size: int,
                     workers: int):
    import ray

    ray.init(num_cpus=workers, include_dashboard=False, ignore_reinit_error=True, log_to_driver=True)

    D_ref = ray.put(D_nodal)
    kx_ref = ray.put(kx)
    ky_ref = ray.put(ky)
    limits_ref = ray.put(limits)
    out_ref = ray.put(str(output_dir))

    @ray.remote
    def _remote(path_str: str, out_str: str, D_local, kx_local, ky_local, iz_local, z_local, limits_local, size_local):
        _render_frame(Path(path_str), Path(out_str), D_local, kx_local, ky_local, iz_local, z_local, limits_local, size_local)
        return path_str

    futures = [
        _remote.remote(str(path), out_ref, D_ref, kx_ref, ky_ref, iz, z_mid, limits_ref, tile_size)
        for path in files
    ]
    done_count = 0
    total = len(futures)
    while futures:
        done, futures = ray.wait(futures, num_returns=min(16, len(futures)))
        done_count += len(done)
        print(f"rendered {done_count}/{total} frames", flush=True)
    ray.shutdown()


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "center_panels_fixed"
    files = _snapshot_files(input_dir, args.stride)
    cfg = _build_cfg(files[0])
    grid = make_grid(cfg)
    z = np.asarray(grid.Z)
    iz = int(np.argmin(np.abs(z - 0.5)))
    z_mid = float(z[iz])
    V = np.asarray(grid.V)
    V_inv = np.asarray(grid.V_inv)
    G_Z = np.asarray(grid.G_Z)
    D_nodal = V @ G_Z @ V_inv
    kx = np.asarray(grid.kx)
    ky = np.asarray(grid.ky)

    limits = _compute_limits(files, D_nodal, kx, ky, iz, args.percentile)
    print("fixed ranges:", limits, flush=True)

    if args.ray:
        _render_with_ray(files, output_dir, D_nodal, kx, ky, iz, z_mid, limits, args.tile_size, args.workers)
    else:
        _render_serial(files, output_dir, D_nodal, kx, ky, iz, z_mid, limits, args.tile_size)

    print(f"saved center-plane panels to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
