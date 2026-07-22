#!/usr/bin/env python
"""Render saved upper-layer snapshot PNGs as a rotating grayscale sphere movie."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
from PIL import Image


_STEP_RE = re.compile(r"pv_step_(\d+)\.png$")


def _step_from_path(path: Path) -> int:
    match = _STEP_RE.search(path.name)
    if match is None:
        raise ValueError(f"not a recognized snapshot path: {path}")
    return int(match.group(1))


def _snapshot_paths(run_dir: Path, max_frames: int | None) -> list[Path]:
    paths = sorted(run_dir.glob("pv_step_*.png"), key=_step_from_path)
    if max_frames is not None and len(paths) > max_frames:
        indices = np.linspace(0, len(paths) - 1, max_frames).round().astype(int)
        paths = [paths[i] for i in indices]
    if not paths:
        raise FileNotFoundError(f"no pv_step_*.png snapshots found in {run_dir}")
    return paths


def _signed_field_from_upper_panel(path: Path, crop: tuple[int, int, int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    panel = np.asarray(image.crop(crop), dtype=np.float32) / 255.0
    # The saved snapshots use RdBu_r. Project red-blue contrast to a signed
    # scalar so that the sphere movie can be rendered with a black-white map.
    signed = panel[..., 0] - panel[..., 2]
    signed = signed - np.nanmean(signed)
    scale = np.nanpercentile(np.abs(signed), 99.5)
    return np.clip(signed / max(scale, 1e-6), -1.0, 1.0)


def _sphere_lookup(
    size: int,
    texture_shape: tuple[int, int],
    center_lat_deg: float,
    center_lon_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tex_h, tex_w = texture_shape
    axis = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(axis, axis)
    rr2 = xx * xx + yy * yy
    visible = rr2 <= 1.0
    zz = np.sqrt(np.maximum(1.0 - rr2, 0.0))

    lat0 = np.deg2rad(center_lat_deg)
    lon0 = np.deg2rad(center_lon_deg)
    center = np.array([np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)])
    east = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
    north = np.array([-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)])

    x3 = xx * east[0] + yy * north[0] + zz * center[0]
    y3 = xx * east[1] + yy * north[1] + zz * center[1]
    z3 = xx * east[2] + yy * north[2] + zz * center[2]
    lat = np.arcsin(np.clip(z3, -1.0, 1.0))
    lon = np.mod(np.arctan2(y3, x3), 2.0 * np.pi)

    col = np.mod(lon / (2.0 * np.pi) * tex_w, tex_w).astype(int)
    row = ((np.pi / 2.0 - lat) / np.pi * (tex_h - 1)).clip(0, tex_h - 1).astype(int)
    return row, col, visible, zz


def _render_frame(
    field: np.ndarray,
    step: int,
    size: int,
    center_lat_deg: float,
    center_lon_deg: float,
    label: str,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    row, col, visible, shade = _sphere_lookup(
        size=size,
        texture_shape=field.shape,
        center_lat_deg=center_lat_deg,
        center_lon_deg=center_lon_deg,
    )
    sampled = field[row, col]
    gray = 0.5 + 0.5 * sampled
    gray = np.where(visible, gray * (0.72 + 0.28 * shade), 1.0)

    rgb = np.repeat(gray[..., None], 3, axis=-1)

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=120)
    ax.imshow(rgb, origin="upper", extent=(-1, 1, -1, 1))
    theta = np.linspace(0.0, 2.0 * np.pi, 720)
    ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{label}\nstep {step}", fontsize=12)

    sm = plt.cm.ScalarMappable(cmap="gray", norm=Normalize(vmin=-1.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("signed grayscale", rotation=270, labelpad=14)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    frame = frame.reshape(height, width, 3)
    plt.close(fig)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--size", type=int, default=760)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--center-lat-deg", type=float, default=-50.0)
    parser.add_argument("--start-lon-deg", type=float, default=210.0)
    parser.add_argument("--end-lon-deg", type=float, default=330.0)
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        default=(88, 70, 1220, 456),
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
    )
    parser.add_argument(
        "--label",
        default="upper-layer signed PV anomaly, spherical view",
    )
    args = parser.parse_args()

    paths = _snapshot_paths(args.run_dir, args.max_frames)
    output = args.output or args.run_dir / "upper_layer_sphere_bw.gif"
    lons = np.linspace(args.start_lon_deg, args.end_lon_deg, len(paths))

    frames = []
    for path, lon in zip(paths, lons, strict=True):
        field = _signed_field_from_upper_panel(path, tuple(args.crop))
        frame = _render_frame(
            field=field,
            step=_step_from_path(path),
            size=args.size,
            center_lat_deg=args.center_lat_deg,
            center_lon_deg=lon,
            label=args.label,
        )
        frames.append(Image.fromarray(frame))

    if output.suffix.lower() == ".gif":
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / args.fps),
            loop=0,
            optimize=True,
        )
    else:
        import imageio.v2 as imageio

        with imageio.get_writer(output, fps=args.fps, codec="libx264", quality=8) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame))
    print(output)
    print(f"frames={len(paths)}")


if __name__ == "__main__":
    main()
