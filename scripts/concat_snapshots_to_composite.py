#!/usr/bin/env python
"""Concatenate per-time snapshot_*.nc files into a single composite NetCDF
with an unlimited time dimension.

For each input snapshot:
  dims  : (z, y, x)
  vars  : q_prime, w, theta, psi  -- (z, y, x), float32
          theta_bar, theta_mean_total -- (z,), float32
  attrs : time (float), step (int), Ra_tilde, sigma, beta, Ld, L,
          thermal_closure, Nx

Output composite:
  dims  : (time, z, y, x)  with time UNLIMITED
  vars  : time -- (time,), float64; same shape per-snapshot vars promoted to
          (time, z, y, x) for 3D fields and (time, z) for 1D mean profiles.
  copies global attrs from the first snapshot.

Designed to be run on a single resolution; spawn two processes for parallel.
"""

from __future__ import annotations

import argparse
import re
import time as _time
from pathlib import Path

import netCDF4
import numpy as np


def list_snapshots(snap_dir: Path):
    pat = re.compile(r"snapshot_(\d{8})\.nc$")
    out = []
    for f in sorted(snap_dir.glob("snapshot_*.nc")):
        m = pat.search(f.name)
        if not m:
            continue
        out.append((int(m.group(1)), str(f)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-dir", required=True, type=str)
    p.add_argument("--output", required=True, type=str,
                   help="Path to composite .nc file")
    p.add_argument("--complevel", type=int, default=1,
                   help="zlib compression level (0=no compression)")
    args = p.parse_args()

    snap_dir = Path(args.snapshot_dir)
    files = list_snapshots(snap_dir)
    if not files:
        raise SystemExit(f"No snapshots in {snap_dir}")
    print(f"[{snap_dir.name}] {len(files)} snapshots: "
          f"step {files[0][0]} -> {files[-1][0]}")

    # Probe first file for dims, coords, attrs
    with netCDF4.Dataset(files[0][1]) as ds0:
        Nx = int(ds0.Nx)
        Nz1 = ds0.dimensions["z"].size
        x_coord = np.array(ds0["x"][:])
        y_coord = np.array(ds0["y"][:])
        z_coord = np.array(ds0["z"][:])
        ga = {a: getattr(ds0, a) for a in ds0.ncattrs() if a not in ("time", "step")}
        var_names_3d = [n for n in ("q_prime", "w", "theta", "psi") if n in ds0.variables]
        var_names_1d = [n for n in ("theta_bar", "theta_mean_total") if n in ds0.variables]

    # Build output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    t_start = _time.time()
    with netCDF4.Dataset(out_path, "w", format="NETCDF4") as ds:
        # Dimensions
        ds.createDimension("time", None)   # unlimited
        ds.createDimension("z", Nz1)
        ds.createDimension("y", Nx)
        ds.createDimension("x", Nx)

        # Coordinate variables
        v_t = ds.createVariable("time", "f8", ("time",))
        v_t.units = "Lc / Uc (simulation time units)"
        v_t.long_name = "physical time"

        v_step = ds.createVariable("step", "i8", ("time",))
        v_step.long_name = "integer time step number"

        v_x = ds.createVariable("x", "f8", ("x",)); v_x[:] = x_coord; v_x.units = "Lc"
        v_y = ds.createVariable("y", "f8", ("y",)); v_y[:] = y_coord; v_y.units = "Lc"
        v_z = ds.createVariable("z", "f8", ("z",)); v_z[:] = z_coord
        v_z.long_name = "depth (CGL points)"

        # 3D field variables
        v_3d = {}
        for name in var_names_3d:
            v = ds.createVariable(
                name, "f4", ("time", "z", "y", "x"),
                zlib=(args.complevel > 0), complevel=args.complevel,
                chunksizes=(1, Nz1, Nx, Nx),
            )
            v_3d[name] = v

        # 1D (z-only) variables become (time, z)
        v_1d = {}
        for name in var_names_1d:
            v = ds.createVariable(
                name, "f4", ("time", "z"),
                zlib=(args.complevel > 0), complevel=args.complevel,
            )
            v_1d[name] = v

        # Global attrs (drop per-snapshot ones)
        for a, val in ga.items():
            try:
                ds.setncattr(a, val)
            except Exception:
                pass
        ds.source_dir = str(snap_dir)
        ds.n_snapshots = len(files)

        # Loop snapshots, append along time
        for i, (step, fpath) in enumerate(files):
            with netCDF4.Dataset(fpath) as src:
                v_t[i] = float(src.time)
                v_step[i] = int(src.step)
                for name, v in v_3d.items():
                    v[i, :, :, :] = np.asarray(src[name][:], dtype=np.float32)
                for name, v in v_1d.items():
                    v[i, :] = np.asarray(src[name][:], dtype=np.float32)

            if (i + 1) % max(1, len(files) // 20) == 0 or i + 1 == len(files):
                elapsed = _time.time() - t_start
                eta = elapsed * (len(files) - (i + 1)) / max(i + 1, 1)
                print(f"  [{snap_dir.name}] {i+1}/{len(files)}  "
                      f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s")

    elapsed = _time.time() - t_start
    out_size_gb = out_path.stat().st_size / (1024 ** 3)
    print(f"[{snap_dir.name}] DONE in {elapsed:.1f}s  -> {out_path}  "
          f"({out_size_gb:.2f} GB, {len(files)} time records)")


if __name__ == "__main__":
    main()
