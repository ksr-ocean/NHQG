#!/usr/bin/env python
"""Summarize saved NHQG output directories.

The older checkpoint format stores state only, not the full run config. This
script inventories diagnostics archives and uses conservative filename
inference for configuration details such as horizontal dealiasing.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


METRIC_ALIASES = {
    "nu_d": ("Nusselt_dealiased", "Nu_dealiased"),
    "nu_raw": ("Nusselt", "Nu"),
    "max_w": ("max_w",),
    "max_theta": ("max_theta",),
    "r_ex_d": ("mean_theta_exchange_residual_dealiased",),
    "r_ex_sbp": ("mean_theta_exchange_residual_sbp",),
    "ke_bt": ("KE_bt",),
    "ke_bc": ("KE_bc",),
}

FINITE_METRICS = ("nu_d", "max_w", "max_theta")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output"),
        help="Output root containing run directories.",
    )
    parser.add_argument(
        "--format",
        choices=["table", "markdown", "csv"],
        default="table",
        help="Output format.",
    )
    parser.add_argument(
        "--sort",
        choices=["name", "t_last", "t_finite", "resolution"],
        default="name",
        help="Sort order.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum rows to print.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Also include output directories without diagnostics_history.npz.",
    )
    return parser.parse_args()


def _infer_dealiasing(name: str) -> str:
    lower = name.lower()
    if "23rule" in lower or "23_rule" in lower:
        return "23_rule"
    if "32rule" in lower or "32_rule" in lower:
        return "32_rule"
    return "32_rule?"


def _infer_branch(name: str) -> str:
    lower = name.lower()
    if "balancedsbp2pc" in lower:
        return "balanced_sbp2_pc"
    if "balancedsbp2" in lower:
        return "balanced_sbp2"
    if "balancedmidpoint" in lower:
        return "balanced_midpoint"
    if "coralworkgrid" in lower or "coral_workgrid" in lower:
        return "coral_workgrid"
    if "evolvemean" in lower:
        return "evolve_mean"
    return "unknown"


def _infer_substeps(name: str) -> str:
    lower = name.lower()
    match = re.search(r"subcycle(\d+)", lower)
    if match:
        return match.group(1)
    match = re.search(r"sub(\d+)", lower)
    if match:
        return match.group(1)
    return ""


def _infer_resolution(name: str) -> tuple[str, str]:
    nx = ""
    nz = ""
    match = re.search(r"Nx(\d+)", name)
    if match:
        nx = match.group(1)
    match = re.search(r"Nz(\d+)", name)
    if match:
        nz = match.group(1)
    return nx, nz


def _load_1d(npz: Any, aliases: tuple[str, ...]) -> np.ndarray | None:
    for key in aliases:
        if key in npz.files:
            arr = np.asarray(npz[key])
            if arr.ndim == 0:
                return arr.reshape(1)
            if arr.ndim == 1:
                return arr
    return None


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(out):
        return out
    return None


def _fmt(value: Any, *, sci: bool = False) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "nan"
    if sci:
        return f"{v:.3e}"
    if abs(v) >= 1e5 or (0 < abs(v) < 1e-3):
        return f"{v:.3e}"
    return f"{v:.4g}"


def _summarize_archive(run_dir: Path, include_empty: bool = False) -> dict[str, Any] | None:
    diag_path = run_dir / "diagnostics_history.npz"
    if not diag_path.exists():
        if not include_empty:
            return None
        nx, nz = _infer_resolution(run_dir.name)
        return {
            "run": run_dir.name,
            "path": str(run_dir),
            "dealiasing": _infer_dealiasing(run_dir.name),
            "branch": _infer_branch(run_dir.name),
            "substeps": _infer_substeps(run_dir.name),
            "Nx": nx,
            "Nz": nz,
            "records": 0,
            "t_first": None,
            "t_last": None,
            "t_finite": None,
            "status": "no_diagnostics",
            "checkpoint_count": len(list(run_dir.glob("checkpoint_*.npz"))),
        }

    with np.load(diag_path, allow_pickle=True) as npz:
        t = _load_1d(npz, ("t",))
        step = _load_1d(npz, ("step",))
        records = int(len(t)) if t is not None else 0

        metrics: dict[str, np.ndarray | None] = {
            name: _load_1d(npz, aliases)
            for name, aliases in METRIC_ALIASES.items()
        }

        finite_mask = None
        if records:
            finite_mask = np.ones(records, dtype=bool)
            for name in FINITE_METRICS:
                arr = metrics.get(name)
                if arr is not None and len(arr) == records:
                    finite_mask &= np.isfinite(arr.astype(float, copy=False))

        last_finite_idx = None
        first_nonfinite_idx = None
        status = "unknown"
        if finite_mask is not None and len(finite_mask):
            finite_idx = np.flatnonzero(finite_mask)
            if len(finite_idx):
                last_finite_idx = int(finite_idx[-1])
            nonfinite_idx = np.flatnonzero(~finite_mask)
            if len(nonfinite_idx):
                first_nonfinite_idx = int(nonfinite_idx[0])
            status = "finite_last" if finite_mask[-1] else "nonfinite_tail"

        last_idx = records - 1 if records else None
        metric_values: dict[str, float | None] = {}
        for name, arr in metrics.items():
            metric_values[name] = None
            if arr is not None and last_idx is not None and len(arr) > last_idx:
                metric_values[name] = _finite_float(arr[last_idx])

        nx, nz = _infer_resolution(run_dir.name)
        summary: dict[str, Any] = {
            "run": run_dir.name,
            "path": str(run_dir),
            "dealiasing": _infer_dealiasing(run_dir.name),
            "branch": _infer_branch(run_dir.name),
            "substeps": _infer_substeps(run_dir.name),
            "Nx": nx,
            "Nz": nz,
            "records": records,
            "t_first": _finite_float(t[0]) if t is not None and len(t) else None,
            "t_last": _finite_float(t[-1]) if t is not None and len(t) else None,
            "t_finite": (
                _finite_float(t[last_finite_idx])
                if t is not None and last_finite_idx is not None and len(t) > last_finite_idx
                else None
            ),
            "first_nonfinite_t": (
                _finite_float(t[first_nonfinite_idx])
                if t is not None and first_nonfinite_idx is not None and len(t) > first_nonfinite_idx
                else None
            ),
            "last_step": _finite_float(step[-1]) if step is not None and len(step) else None,
            "status": status,
            "checkpoint_count": len(list(run_dir.glob("checkpoint_*.npz"))),
        }
        summary.update(metric_values)
        return summary


def _sort_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    if key == "t_last":
        return sorted(rows, key=lambda r: (r.get("t_last") is None, -(r.get("t_last") or -1), r["run"]))
    if key == "t_finite":
        return sorted(rows, key=lambda r: (r.get("t_finite") is None, -(r.get("t_finite") or -1), r["run"]))
    if key == "resolution":
        def res_key(row: dict[str, Any]) -> tuple[int, int, str]:
            nx = int(row["Nx"]) if str(row.get("Nx", "")).isdigit() else -1
            nz = int(row["Nz"]) if str(row.get("Nz", "")).isdigit() else -1
            return (nx, nz, row["run"])
        return sorted(rows, key=res_key)
    return sorted(rows, key=lambda r: r["run"])


def _print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        ("run", "run", 64, False),
        ("Nx", "Nx", 4, False),
        ("Nz", "Nz", 4, False),
        ("dealiasing", "deal", 8, False),
        ("branch", "branch", 18, False),
        ("substeps", "sub", 3, False),
        ("t_last", "t_last", 7, False),
        ("t_finite", "t_fin", 7, False),
        ("status", "status", 14, False),
        ("nu_d", "Nu_d", 10, False),
        ("r_ex_sbp", "R_sbp", 10, True),
        ("r_ex_d", "R_d", 10, True),
        ("max_w", "max_w", 10, False),
        ("max_theta", "max_th", 10, False),
    ]
    header = " ".join(label.ljust(width) for _, label, width, _ in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        parts = []
        for key, _, width, sci in columns:
            value = _fmt(row.get(key), sci=sci)
            if key == "run" and len(value) > width:
                value = value[: width - 1] + "~"
            parts.append(value.ljust(width))
        print(" ".join(parts))


def _print_markdown(rows: list[dict[str, Any]]) -> None:
    columns = [
        "run", "Nx", "Nz", "dealiasing", "branch", "substeps", "t_last",
        "t_finite", "status", "nu_d", "r_ex_sbp", "r_ex_d", "max_w", "max_theta",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        vals = [_fmt(row.get(c), sci=c in {"r_ex_sbp", "r_ex_d"}) for c in columns]
        print("| " + " | ".join(vals) + " |")


def _print_csv(rows: list[dict[str, Any]]) -> None:
    columns = [
        "run", "path", "Nx", "Nz", "dealiasing", "branch", "substeps",
        "records", "t_first", "t_last", "t_finite", "first_nonfinite_t",
        "last_step", "status", "checkpoint_count", "nu_d", "nu_raw",
        "max_w", "max_theta", "r_ex_sbp", "r_ex_d", "ke_bt", "ke_bc",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def main() -> None:
    args = _parse_args()
    if not args.root.exists():
        raise SystemExit(f"missing output root: {args.root}")

    rows = []
    for run_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        summary = _summarize_archive(run_dir, include_empty=args.include_empty)
        if summary is not None:
            rows.append(summary)

    rows = _sort_rows(rows, args.sort)
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.format == "csv":
        _print_csv(rows)
    elif args.format == "markdown":
        _print_markdown(rows)
    else:
        _print_table(rows)


if __name__ == "__main__":
    main()
