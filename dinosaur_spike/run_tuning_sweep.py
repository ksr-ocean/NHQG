#!/usr/bin/env python
"""Run a guarded two-layer QG parameter sweep using the solution runner."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time


@dataclass(frozen=True)
class Case:
    name: str
    shear: float
    hyperdiffusion: float
    dt: float
    amplitude: float
    init_max_wavenumber: int


DEFAULT_CASES = (
    Case("s12_h5e11_dt5e4", shear=1.2, hyperdiffusion=5e-11, dt=5e-4, amplitude=0.03, init_max_wavenumber=32),
    Case("s12_h75e12_dt5e4", shear=1.2, hyperdiffusion=7.5e-11, dt=5e-4, amplitude=0.03, init_max_wavenumber=32),
    Case("s13_h75e12_dt5e4", shear=1.3, hyperdiffusion=7.5e-11, dt=5e-4, amplitude=0.025, init_max_wavenumber=32),
    Case("s13_h1e10_dt5e4", shear=1.3, hyperdiffusion=1e-10, dt=5e-4, amplitude=0.025, init_max_wavenumber=32),
)

HIGH_SHEAR_CASES = (
    Case("s15_h5e11_dt25e5", shear=1.5, hyperdiffusion=5e-11, dt=2.5e-4, amplitude=0.025, init_max_wavenumber=32),
    Case("s18_h75e12_dt25e5", shear=1.8, hyperdiffusion=7.5e-11, dt=2.5e-4, amplitude=0.02, init_max_wavenumber=32),
    Case("s20_h1e10_dt25e5", shear=2.0, hyperdiffusion=1e-10, dt=2.5e-4, amplitude=0.02, init_max_wavenumber=32),
    Case("s24_h15e10_dt25e5", shear=2.4, hyperdiffusion=1.5e-10, dt=2.5e-4, amplitude=0.015, init_max_wavenumber=32),
)

PROFILES = {
    "baseline": DEFAULT_CASES,
    "high_shear": HIGH_SHEAR_CASES,
}


def _latest_and_initial_rows(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no diagnostics rows in {path}")
    return rows[-1], rows[0]


def _append_summary(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _case_command(args, case: Case, out_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "dinosaur_spike/run_two_layer_solution.py",
        "--device",
        args.device,
        "--impl",
        args.impl,
        "--dtype",
        args.dtype,
        "--wavenumbers",
        str(args.wavenumbers),
        "--steps",
        str(args.steps),
        "--dt",
        str(case.dt),
        "--snapshot-every",
        str(args.snapshot_every),
        "--save-state-every",
        str(args.snapshot_every),
        "--amplitude",
        str(case.amplitude),
        "--init-max-wavenumber",
        str(case.init_max_wavenumber),
        "--sponge-max-rate",
        str(args.sponge_max_rate),
        "--background-shear-velocity",
        str(case.shear),
        "--background-profile",
        args.background_profile,
        "--background-sin3-weight",
        str(args.background_sin3_weight),
        "--mask-plateau-north-edge-deg",
        str(args.mask_plateau_north_edge_deg),
        "--mask-taper-north-edge-deg",
        str(args.mask_taper_north_edge_deg),
        "--hyperdiffusion-rate",
        str(case.hyperdiffusion),
        "--stop-top10-fraction",
        str(args.stop_top10_fraction),
        "--stop-q-abs-max",
        str(args.stop_q_abs_max),
        "--out-dir",
        str(out_dir),
    ]
    if args.allow_float32_fast:
        command.append("--allow-float32-fast")
    if args.no_mask_nonlinear_tendency:
        command.append("--no-mask-nonlinear-tendency")
    return command


def _summarize_case(summary_path: Path, case: Case, out_dir: Path, returncode: int, elapsed: float) -> None:
    diagnostics = out_dir / "diagnostics.csv"
    if diagnostics.exists():
        last, first = _latest_and_initial_rows(diagnostics)
        initial_ens = float(first["windowed_enstrophy"])
        final_ens = float(last["windowed_enstrophy"])
        growth = final_ens / initial_ens if initial_ens > 0.0 else 0.0
        row = {
            "case": case.name,
            "returncode": returncode,
            "elapsed_s": elapsed,
            "out_dir": out_dir,
            "shear": case.shear,
            "hyperdiffusion": case.hyperdiffusion,
            "dt": case.dt,
            "amplitude": case.amplitude,
            "init_max_wavenumber": case.init_max_wavenumber,
            "final_step": last["step"],
            "final_time": last["time"],
            "enstrophy_growth": growth,
            "final_enstrophy": final_ens,
            "final_q_abs_max": last["q_abs_max"],
            "final_peak_l": last["spectral_peak_l"],
            "final_top10": last["spectral_top10_fraction"],
            "final_top20": last["spectral_top20_fraction"],
        }
    else:
        row = {
            "case": case.name,
            "returncode": returncode,
            "elapsed_s": elapsed,
            "out_dir": out_dir,
            "shear": case.shear,
            "hyperdiffusion": case.hyperdiffusion,
            "dt": case.dt,
            "amplitude": case.amplitude,
            "init_max_wavenumber": case.init_max_wavenumber,
            "final_step": "",
            "final_time": "",
            "enstrophy_growth": "",
            "final_enstrophy": "",
            "final_q_abs_max": "",
            "final_peak_l": "",
            "final_top10": "",
            "final_top20": "",
        }
    _append_summary(summary_path, row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu7", "default"], default="gpu7")
    parser.add_argument("--impl", choices=["real", "fast"], default="fast")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--wavenumbers", type=int, default=425)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="baseline")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--snapshot-every", type=int, default=2000)
    parser.add_argument("--sponge-max-rate", type=float, default=0.5)
    parser.add_argument(
        "--background-profile",
        choices=["solid_body", "sin_plus_sin3"],
        default="solid_body",
    )
    parser.add_argument("--background-sin3-weight", type=float, default=0.75)
    parser.add_argument("--mask-plateau-north-edge-deg", type=float, default=-30.0)
    parser.add_argument("--mask-taper-north-edge-deg", type=float, default=5.0)
    parser.add_argument("--no-mask-nonlinear-tendency", action="store_true")
    parser.add_argument("--stop-top10-fraction", type=float, default=0.45)
    parser.add_argument("--stop-q-abs-max", type=float, default=2.0)
    parser.add_argument("--allow-float32-fast", action="store_true")
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root or Path("output") / "dinosaur_two_layer" / f"tuning_sweep_w{args.wavenumbers}_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.csv"

    for case in PROFILES[args.profile]:
        case_dir = out_root / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        command = _case_command(args, case, case_dir)
        log_path = case_dir / "run.log"
        print(f"case={case.name} out_dir={case_dir}", flush=True)
        print(" ".join(command), flush=True)
        start = time.perf_counter()
        with log_path.open("w") as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        elapsed = time.perf_counter() - start
        _summarize_case(summary_path, case, case_dir, proc.returncode, elapsed)
        print(f"case={case.name} returncode={proc.returncode} elapsed_s={elapsed:.1f}", flush=True)

    print(summary_path)


if __name__ == "__main__":
    main()
