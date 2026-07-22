#!/usr/bin/env python
"""Extract Dedalus analysis HDF5 tasks into a compact comparison archive."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeseries-dir", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _collect_series(files: list[Path]) -> dict[str, list[np.ndarray]]:
    payload: dict[str, list[np.ndarray]] = {}
    for path in files:
        with h5py.File(path, "r") as handle:
            scales = handle["scales"]
            sim_time = np.array(scales["sim_time"])
            write_number = np.array(scales["write_number"])
            payload.setdefault("sim_time", []).append(sim_time)
            payload.setdefault("write_number", []).append(write_number)

            tasks = handle["tasks"]
            for key in tasks.keys():
                payload.setdefault(key, []).append(np.array(tasks[key]))
    return payload


def main():
    args = parse_args()
    ts_dir = Path(args.timeseries_dir)
    files = sorted(ts_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {ts_dir}")

    raw = _collect_series(files)
    merged: dict[str, np.ndarray] = {}
    for key, chunks in raw.items():
        axis = 0
        merged[key] = np.concatenate(chunks, axis=axis)

    out_path = Path(args.output) if args.output else ts_dir / "history_compact.npz"
    np.savez(out_path, **merged)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
