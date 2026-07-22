#!/usr/bin/env python
"""Render fixed-range 3x3 snapshot panels from saved NetCDF snapshots."""

import argparse
from pathlib import Path

import netCDF4
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Run output directory containing snapshot_*.nc")
    parser.add_argument("--output-dir", default=None, help="Directory for rendered frames")
    parser.add_argument("--depths", type=float, nargs=3, default=[0.05, 0.50, 0.95])
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth snapshot")
    parser.add_argument(
        "--theta-limits",
        type=float,
        nargs=2,
        default=None,
        help="Optional fixed [vmin vmax] for theta; otherwise infer globally.",
    )
    parser.add_argument(
        "--symmetric-limits",
        action="store_true",
        help="Force theta to use symmetric limits too. w and zeta are always symmetric.",
    )
    parser.add_argument(
        "--ray",
        action="store_true",
        help="Render frames in parallel with Ray.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of Ray worker CPUs to request when --ray is enabled.",
    )
    return parser.parse_args()


def _snapshot_files(input_dir: Path, stride: int) -> list[Path]:
    files = sorted(input_dir.glob("snapshot_*.nc"))
    if stride > 1:
        files = files[::stride]
    if not files:
        raise FileNotFoundError(f"No snapshots found in {input_dir}")
    return files


def _depth_indices(z: np.ndarray, depths: list[float]) -> list[int]:
    return [int(np.argmin(np.abs(z - depth))) for depth in depths]


def _load_slices(path: Path, depth_idx: list[int]) -> tuple[float, dict[str, np.ndarray]]:
    with netCDF4.Dataset(path) as ds:
        t = float(ds.time)
        fields = {
            "w": np.asarray(ds.variables["w"][depth_idx, :, :], dtype=np.float64),
            "theta": np.asarray(ds.variables["theta"][depth_idx, :, :], dtype=np.float64),
            "zeta": np.asarray(ds.variables["q_prime"][depth_idx, :, :], dtype=np.float64),
        }
    return t, fields


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


def _compute_limits(files: list[Path], depth_idx: list[int],
                    theta_limits: tuple[float, float] | None,
                    symmetric_theta: bool) -> dict[str, tuple[float, float]]:
    mins = {"theta": np.inf}
    maxs = {"theta": -np.inf, "w": 0.0, "zeta": 0.0}

    for path in files:
        _, fields = _load_slices(path, depth_idx)
        for name in ("w", "zeta"):
            maxs[name] = max(maxs[name], float(np.max(np.abs(fields[name]))))
        mins["theta"] = min(mins["theta"], float(np.min(fields["theta"])))
        maxs["theta"] = max(maxs["theta"], float(np.max(fields["theta"])))

    limits = {
        "w": (-max(maxs["w"], 1e-12), max(maxs["w"], 1e-12)),
        "zeta": (-max(maxs["zeta"], 1e-12), max(maxs["zeta"], 1e-12)),
    }
    if theta_limits is not None:
        limits["theta"] = (float(theta_limits[0]), float(theta_limits[1]))
    elif symmetric_theta:
        vmax = max(abs(mins["theta"]), abs(maxs["theta"]), 1e-12)
        limits["theta"] = (-vmax, vmax)
    else:
        vmin = mins["theta"] if np.isfinite(mins["theta"]) else -1e-12
        vmax = maxs["theta"] if np.isfinite(maxs["theta"]) else 1e-12
        if vmax <= vmin:
            vmax = vmin + 1e-12
        limits["theta"] = (vmin, vmax)
    return limits


def _render_frame(path: Path, out_dir: Path, z: np.ndarray,
                  depth_idx: list[int], limits: dict[str, tuple[float, float]]):
    t, fields = _load_slices(path, depth_idx)
    row_specs = [("w", "bwr"), ("theta", "jet"), ("zeta", "bwr")]
    tile_size = 240
    left_pad = 120
    top_pad = 55
    gap = 8
    bar_gap = 10
    bar_width = 26
    width = left_pad + 3 * tile_size + 2 * gap + bar_gap + bar_width + 70
    height = top_pad + 3 * tile_size + 2 * gap + 20
    canvas = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((12, 12), f"t={t:.3f}", fill=(0, 0, 0), font=font)
    for col, idx in enumerate(depth_idx):
        x = left_pad + col * (tile_size + gap) + 82
        draw.text((x, 12), f"z={z[idx]:.3f}", fill=(0, 0, 0), font=font)

    for row, (name, cmap) in enumerate(row_specs):
        y0 = top_pad + row * (tile_size + gap)
        vmin, vmax = limits[name]
        draw.text((12, y0 + tile_size // 2 - 10), name, fill=(0, 0, 0), font=font)
        draw.text((12, y0 + tile_size // 2 + 6), f"[{vmin:.2e}, {vmax:.2e}]", fill=(80, 80, 80), font=font)
        for col in range(3):
            rgb = _map_rgb(fields[name][col], cmap, (vmin, vmax))
            tile = _resize_tile(rgb, tile_size)
            x0 = left_pad + col * (tile_size + gap)
            canvas.paste(tile, (x0, y0))
        bar_rgb = _colorbar_rgb(cmap, (vmin, vmax), height=tile_size, width=bar_width)
        bar_x = left_pad + 3 * tile_size + 2 * gap + bar_gap
        canvas.paste(Image.fromarray(bar_rgb, mode="RGB"), (bar_x, y0))
        draw.text((bar_x + bar_width + 4, y0), f"{vmax:.1e}", fill=(0, 0, 0), font=font)
        draw.text((bar_x + bar_width + 4, y0 + tile_size - 10), f"{vmin:.1e}", fill=(0, 0, 0), font=font)

    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(out_dir / f"{path.stem}.png")


def _render_serial(files: list[Path], out_dir: Path, z: np.ndarray,
                   depth_idx: list[int], limits: dict[str, tuple[float, float]]):
    for path in files:
        _render_frame(path, out_dir, z, depth_idx, limits)


def _render_with_ray(files: list[Path], out_dir: Path, z: np.ndarray,
                     depth_idx: list[int], limits: dict[str, tuple[float, float]], workers: int):
    import ray

    ray.init(
        num_cpus=workers,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=True,
    )

    z_ref = ray.put(z)
    depth_idx_ref = ray.put(depth_idx)
    limits_ref = ray.put(limits)
    out_dir_ref = ray.put(str(out_dir))

    @ray.remote
    def _render_remote(path_str: str, out_dir_str: str, z_arr, depth_idx_local, limits_local):
        _render_frame(Path(path_str), Path(out_dir_str), z_arr, depth_idx_local, limits_local)
        return path_str

    futures = [
        _render_remote.remote(str(path), out_dir_ref, z_ref, depth_idx_ref, limits_ref)
        for path in files
    ]
    completed = 0
    total = len(futures)
    while futures:
        done, futures = ray.wait(futures, num_returns=min(16, len(futures)))
        completed += len(done)
        print(f"rendered {completed}/{total} frames", flush=True)
    ray.shutdown()


def main():
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "panels_fixed"

    files = _snapshot_files(input_dir, args.stride)
    with netCDF4.Dataset(files[0]) as ds:
        z = np.asarray(ds.variables["z"][:], dtype=np.float64)
    depth_idx = _depth_indices(z, args.depths)
    limits = _compute_limits(
        files,
        depth_idx,
        tuple(args.theta_limits) if args.theta_limits is not None else None,
        args.symmetric_limits,
    )

    print(
        "fixed ranges:",
        f"w={limits['w']}",
        f"theta={limits['theta']}",
        f"zeta={limits['zeta']}",
        flush=True,
    )

    if args.ray:
        _render_with_ray(files, output_dir, z, depth_idx, limits, args.workers)
    else:
        _render_serial(files, output_dir, z, depth_idx, limits)

    print(f"saved fixed-range panels to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
