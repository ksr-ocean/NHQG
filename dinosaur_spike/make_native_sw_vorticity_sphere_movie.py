#!/usr/bin/env python
"""Render native Dinosaur shallow-water vorticity checkpoints as a sphere movie."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image

from dinosaur_spike.make_sphere_movie_from_snapshots import _render_frame


_STEP_RE = re.compile(r"state_step_(\d+)\.npz$")


def _configure_device(device: str) -> None:
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    if device == "cpu":
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    elif device == "default":
        return
    else:
        raise ValueError(f"unsupported device option {device!r}")


def _step_from_path(path: Path) -> int:
    match = _STEP_RE.search(path.name)
    if match is None:
        raise ValueError(f"not a recognized checkpoint path: {path}")
    return int(match.group(1))


def _checkpoint_paths(run_dir: Path, max_frames: int | None) -> list[Path]:
    paths = sorted(run_dir.glob("state_step_*.npz"), key=_step_from_path)
    if max_frames is not None and len(paths) > max_frames:
        indices = np.linspace(0, len(paths) - 1, max_frames).round().astype(int)
        paths = [paths[i] for i in indices]
    if not paths:
        raise FileNotFoundError(f"no state_step_*.npz checkpoints found in {run_dir}")
    return paths


def _wavenumbers_from_checkpoint(path: Path, fallback: int | None) -> int:
    with np.load(path) as data:
        if "wavenumbers" in data:
            return int(data["wavenumbers"])
    if fallback is None:
        raise ValueError("checkpoint lacks wavenumbers metadata; pass --wavenumbers")
    return fallback


def _make_grid(wavenumbers: int, impl_name: str):
    from dinosaur import spherical_harmonic

    if impl_name == "real":
        impl = spherical_harmonic.RealSphericalHarmonics
    elif impl_name == "fast":
        impl = spherical_harmonic.FastSphericalHarmonics
    else:
        raise ValueError(f"unsupported implementation {impl_name!r}")
    return spherical_harmonic.Grid.with_wavenumbers(
        longitude_wavenumbers=wavenumbers,
        dealiasing="quadratic",
        spherical_harmonics_impl=impl,
        radius=1.0,
    )


def _vorticity_texture(path: Path, grid, layer: int) -> np.ndarray:
    import jax.numpy as jnp

    with np.load(path) as data:
        vorticity = jnp.asarray(data["vorticity"])
    nodal = np.asarray(grid.to_nodal(vorticity[layer]))
    # Dinosaur nodal fields are longitude-major with latitude ordered
    # south-to-north. The sphere texture convention here is row 0 = north.
    return np.flipud(nodal.T)


def _scale_for_paths(
    paths: list[Path],
    grid,
    layer: int,
    percentile: float,
    per_frame: bool,
) -> float:
    if per_frame:
        return 1.0
    scale = 0.0
    for path in paths:
        field = _vorticity_texture(path, grid, layer)
        field = field - np.nanmean(field)
        scale = max(scale, float(np.nanpercentile(np.abs(field), percentile)))
    return max(scale, 1e-30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "default"], default="cpu")
    parser.add_argument("--impl", choices=["real", "fast"], default="fast")
    parser.add_argument("--wavenumbers", type=int, default=None)
    parser.add_argument("--layer", type=int, choices=[0, 1], default=0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--size", type=int, default=760)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--center-lat-deg", type=float, default=-52.0)
    parser.add_argument("--start-lon-deg", type=float, default=200.0)
    parser.add_argument("--end-lon-deg", type=float, default=340.0)
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument(
        "--normalization",
        choices=["global", "per-frame"],
        default="global",
    )
    args = parser.parse_args()
    _configure_device(args.device)

    paths = _checkpoint_paths(args.run_dir, args.max_frames)
    wavenumbers = _wavenumbers_from_checkpoint(paths[0], args.wavenumbers)
    grid = _make_grid(wavenumbers, args.impl)
    layer_name = "upper" if args.layer == 0 else "lower"
    output = args.output or args.run_dir / f"{layer_name}_layer_vorticity_sphere_bw.gif"
    lons = np.linspace(args.start_lon_deg, args.end_lon_deg, len(paths))
    global_scale = _scale_for_paths(
        paths,
        grid,
        args.layer,
        args.percentile,
        per_frame=args.normalization == "per-frame",
    )

    frames = []
    for path, lon in zip(paths, lons, strict=True):
        field = _vorticity_texture(path, grid, args.layer)
        field = field - np.nanmean(field)
        scale = global_scale
        if args.normalization == "per-frame":
            scale = max(float(np.nanpercentile(np.abs(field), args.percentile)), 1e-30)
        field = np.clip(field / scale, -1.0, 1.0)
        frame = _render_frame(
            field=field,
            step=_step_from_path(path),
            size=args.size,
            center_lat_deg=args.center_lat_deg,
            center_lon_deg=lon,
            label=f"native SW {layer_name}-layer relative vorticity",
        )
        frames.append(Image.fromarray(frame))

    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / args.fps),
        loop=0,
        optimize=True,
    )
    print(output)
    print(f"frames={len(paths)}")
    print(f"normalization={args.normalization}")
    print(f"scale={global_scale:.6e}")


if __name__ == "__main__":
    main()
