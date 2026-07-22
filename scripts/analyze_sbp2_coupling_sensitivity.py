#!/usr/bin/env python
"""Compare SBP2 stage-placement and subcycling variants on saved checkpoints."""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import dataclasses
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from nhqg.config import NHQGConfig
from nhqg.diagnostics import compute_diagnostics
from nhqg.grid import make_grid
from nhqg.io import load_checkpoint
from nhqg.paths import normalize_output_dir, resolve_existing_output_path
from nhqg.solver import (
    State,
    _finalize_state,
    _thermal_correction_tendency,
    balanced_sbp2_thermal_substep,
    explicit_rhs_dispatch,
    imex_implicit_solve,
    imex_mean_temp_solve,
    imex_step,
    imex_step_ars222,
    implicit_tendency,
)


K_C = 1.3048
L_C = 2.0 * math.pi / K_C


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        default=(
            "output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_"
            "balancedsbp2pc_flux_continue_from_t42_Nx64_Nz256_dt5e5_t80"
        ),
    )
    parser.add_argument("--Nx", type=int, default=64)
    parser.add_argument("--Nz", type=int, default=256)
    parser.add_argument("--Ra", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--t-final", type=float, default=80.0)
    parser.add_argument("--imex-scheme", type=str, default="ars222")
    parser.add_argument(
        "--thermal-closure",
        choices=["fixed_conduction", "evolve_mean"],
        default="evolve_mean",
    )
    parser.add_argument("--mean-temp-eps-sq", type=float, default=1.0)
    parser.add_argument(
        "--nonlinear-advection",
        choices=["jacobian", "flux"],
        default="flux",
    )
    parser.add_argument(
        "--vertical-dealiasing",
        choices=["none", "cheb_3o2", "cheb_2x"],
        default="none",
    )
    parser.add_argument(
        "--mean-exchange-discretization",
        choices=[
            "legacy",
            "coral_workgrid",
            "coral_workgrid_weakmean",
            "coral_workgrid_paired",
            "balanced_midpoint",
            "balanced_sbp2",
            "balanced_sbp2_pc",
        ],
        default="balanced_sbp2_pc",
    )
    parser.add_argument(
        "--sbp-transfer-mode",
        choices=["interp", "mass_adjoint", "weighted_polar"],
        default="interp",
    )
    parser.add_argument("--q-boundary", choices=["none", "neumann"], default="none")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--Ld", type=float, default=float("inf"))
    parser.add_argument("--nu-q", type=float, default=1.0)
    parser.add_argument("--nu-w", type=float, default=1.0)
    parser.add_argument("--nu-theta", type=float, default=1.0)
    parser.add_argument("--hyper-order", type=int, default=1)
    parser.add_argument(
        "--sample-times",
        type=float,
        nargs="+",
        default=[42.25, 45.0, 48.25],
    )
    parser.add_argument(
        "--subcycles",
        type=int,
        nargs="+",
        default=[1, 2, 4],
    )
    parser.add_argument("--report-dir", type=str, default=None)
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> NHQGConfig:
    return NHQGConfig(
        Nx=args.Nx,
        Nz=args.Nz,
        L=10.0 * L_C,
        Ra_tilde=args.Ra,
        sigma=1.0,
        beta=args.beta,
        Ld=args.Ld,
        dt=args.dt,
        t_final=args.t_final,
        imex_scheme=args.imex_scheme,
        q_boundary=args.q_boundary,
        nu_q=args.nu_q,
        nu_w=args.nu_w,
        nu_theta=args.nu_theta,
        hyper_order=args.hyper_order,
        thermal_closure=args.thermal_closure,
        mean_temp_eps_sq=args.mean_temp_eps_sq,
        nonlinear_advection=args.nonlinear_advection,
        vertical_dealiasing=args.vertical_dealiasing,
        mean_exchange_discretization=args.mean_exchange_discretization,
        sbp_transfer_mode=args.sbp_transfer_mode,
        output_dir=args.run_dir,
        float_dtype="float64",
    )


def _parse_step(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _select_checkpoints(run_dir: Path, dt: float, sample_times: list[float]) -> list[Path]:
    paths = sorted(run_dir.glob("checkpoint_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    steps = np.array([_parse_step(path) for path in paths], dtype=np.int64)
    chosen: list[Path] = []
    seen: set[int] = set()
    for time_target in sample_times:
        step_target = int(round(time_target / dt))
        idx = int(np.argmin(np.abs(steps - step_target)))
        step = int(steps[idx])
        if step not in seen:
            chosen.append(paths[idx])
            seen.add(step)
    return chosen


def _diag_fields(state: State, grid) -> dict[str, float]:
    diag = compute_diagnostics(state, grid)
    return {
        "R_ex_d": float(diag["mean_theta_exchange_residual_dealiased"]),
        "R_ex_d_rel": float(diag["mean_theta_exchange_residual_dealiased_rel"]),
        "R_ex_sbp": float(diag["mean_theta_exchange_residual_sbp"]),
        "R_ex_sbp_rel": float(diag["mean_theta_exchange_residual_sbp_rel"]),
        "Nu_d": float(diag["Nusselt_dealiased"]),
        "max_w": float(diag["max_w"]),
        "max_theta": float(diag["max_theta"]),
    }


def _apply_subcycled_corrector(state: State, grid, total_dt, n_substeps: int) -> State:
    if n_substeps < 1:
        raise ValueError("n_substeps must be >= 1")
    sub_dt = total_dt / n_substeps
    out = state
    for _ in range(n_substeps):
        out = balanced_sbp2_thermal_substep(out, grid, sub_dt=sub_dt)
    return out


def _variant_stagewise(state: State, grid, n_substeps: int,
                       apply_stage1: bool = True,
                       apply_stage2: bool = True) -> State:
    base_grid = grid._replace(
        thermal_closure="fixed_conduction",
        mean_exchange_discretization="legacy",
    )

    gamma = grid.gamma_imex
    delta = -jnp.sqrt(jnp.array(2.0, dtype=grid.dt.dtype)) / 2.0
    dt = grid.dt
    alpha = gamma * dt
    omg = dt * (1 - gamma)

    q_n, w_n, th_n, th_bar_n = state

    E1 = explicit_rhs_dispatch(state, base_grid)
    R_q1 = q_n + alpha * E1.q_hat
    R_w1 = w_n + alpha * E1.w_hat
    R_th1 = th_n + alpha * E1.th_hat
    R_th_bar1 = th_bar_n + alpha * E1.th_bar
    q1p, w1p, th1p = imex_implicit_solve(R_q1, R_w1, R_th1, base_grid)
    th_bar1p = imex_mean_temp_solve(R_th_bar1, base_grid)
    predictor1 = State(q1p, w1p, th1p, th_bar1p)

    if apply_stage1:
        state1 = _apply_subcycled_corrector(predictor1, grid, alpha, n_substeps)
        C1 = _thermal_correction_tendency(predictor1, state1, alpha)
    else:
        state1 = predictor1
        C1 = State(
            jnp.zeros_like(state.q_hat),
            jnp.zeros_like(state.w_hat),
            jnp.zeros_like(state.th_hat),
            jnp.zeros_like(state.th_bar),
        )

    E2 = explicit_rhs_dispatch(state1, base_grid)
    I1 = implicit_tendency(state1, base_grid)
    R_q2 = q_n + dt * (delta * E1.q_hat + (1 - delta) * E2.q_hat) \
         + omg * I1.q_hat \
         - omg * grid.diss_rate_q[None, :, :] * state1.q_hat
    R_w2 = w_n + dt * (delta * E1.w_hat + (1 - delta) * E2.w_hat) \
         + omg * I1.w_hat \
         - omg * grid.diss_rate_w[None, :, :] * state1.w_hat
    R_th2 = th_n + dt * (delta * E1.th_hat + (1 - delta) * E2.th_hat) \
          + omg * I1.th_hat \
          - omg * grid.diss_rate_th[None, :, :] * state1.th_hat \
          + omg * C1.th_hat
    R_th_bar2 = th_bar_n + dt * (delta * E1.th_bar + (1 - delta) * E2.th_bar) \
             + omg * I1.th_bar \
             + omg * C1.th_bar
    q2p, w2p, th2p = imex_implicit_solve(R_q2, R_w2, R_th2, base_grid)
    th_bar2p = imex_mean_temp_solve(R_th_bar2, base_grid)
    predictor2 = State(q2p, w2p, th2p, th_bar2p)

    if apply_stage2:
        out = _apply_subcycled_corrector(predictor2, grid, alpha, n_substeps)
    else:
        out = predictor2
    return _finalize_state(out, grid)


def _variant_split_end(state: State, grid, n_substeps: int) -> State:
    base_grid = grid._replace(
        thermal_closure="fixed_conduction",
        mean_exchange_discretization="legacy",
    )
    base_state = imex_step_ars222(state, base_grid)
    out = _apply_subcycled_corrector(base_state, grid, grid.dt, n_substeps)
    return _finalize_state(out, grid)


def _write_report(path: Path, rows: list[dict[str, object]], subcycles: list[int]):
    lines: list[str] = []
    lines.append("# SBP2 Coupling Sensitivity Report")
    lines.append("")
    lines.append("## One-Step Variant Comparison")
    lines.append("")
    lines.append(
        "| checkpoint | t | variant | `R_ex_d` | `R_ex_d_rel` | `R_ex_sbp` | "
        "`Nu_d` | `max_w` | `max_theta` |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        for label, metrics in row["variants"].items():
            lines.append(
                f"| `{row['checkpoint']}` | {row['t']:.2f} | `{label}` | "
                f"{metrics['R_ex_d']:.6e} | {metrics['R_ex_d_rel']:.6e} | "
                f"{metrics['R_ex_sbp']:.6e} | {metrics['Nu_d']:.6e} | "
                f"{metrics['max_w']:.6e} | {metrics['max_theta']:.6e} |"
            )
    lines.append("")
    lines.append("## Reading Guide")
    lines.append("")
    lines.append(
        "- Compare `stagewise_n1`, `stagewise_n2`, and `stagewise_n4`. Strong changes there point to sensitivity in the thermal-corrector time discretization."
    )
    lines.append(
        "- Compare `stage1_only_n1` and `stage2_only_n1` against `stagewise_n1`. If one placement is much better, the ARS stage coupling is implicated."
    )
    lines.append(
        "- Compare `split_end_n1` against `stagewise_n1`. If the split end-of-step map behaves very differently, then stage consistency matters."
    )
    lines.append(
        "- Compare `halfdt2_stagewise` against `stagewise_n1`. Large differences there point to genuine global-timestep sensitivity."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = _parse_args()
    if sorted(set(args.subcycles)) != sorted(args.subcycles):
        args.subcycles = sorted(set(args.subcycles))

    run_dir = resolve_existing_output_path(args.run_dir)
    args.run_dir = str(run_dir)
    report_dir = (
        Path(normalize_output_dir(args.report_dir))
        if args.report_dir
        else (run_dir / "spectra" / "coupling_sensitivity")
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_config(args)
    grid = make_grid(cfg)
    half_cfg = dataclasses.replace(cfg, dt=cfg.dt / 2.0)
    half_grid = make_grid(half_cfg)
    selected = _select_checkpoints(run_dir, args.dt, args.sample_times)

    rows: list[dict[str, object]] = []
    npz_fields: dict[str, list[float]] = {"t": [], "step": []}

    for ckpt in selected:
        state, step, t = load_checkpoint(str(ckpt), dtype=jnp.complex128)
        variants: dict[str, dict[str, float]] = {}
        for n in args.subcycles:
            variants[f"stagewise_n{n}"] = _diag_fields(_variant_stagewise(state, grid, n, True, True), grid)
        variants["stage1_only_n1"] = _diag_fields(_variant_stagewise(state, grid, 1, True, False), grid)
        variants["stage2_only_n1"] = _diag_fields(_variant_stagewise(state, grid, 1, False, True), grid)
        variants["split_end_n1"] = _diag_fields(_variant_split_end(state, grid, 1), grid)
        half_state = imex_step(state, half_grid)
        half_state = imex_step(half_state, half_grid)
        variants["halfdt2_stagewise"] = _diag_fields(half_state, half_grid)

        rows.append({
            "checkpoint": ckpt.name,
            "step": int(step),
            "t": float(t),
            "variants": variants,
        })

        npz_fields["t"].append(float(t))
        npz_fields["step"].append(float(step))
        for label, metrics in variants.items():
            for key, value in metrics.items():
                npz_fields.setdefault(f"{label}_{key}", []).append(float(value))

        print(
            f"{ckpt.name}: t={t:.2f} "
            + " ".join(
                f"{label}(R_ex_d={metrics['R_ex_d']:.3e}, Nu_d={metrics['Nu_d']:.3e})"
                for label, metrics in variants.items()
            ),
            flush=True,
        )

    npz_path = report_dir / "coupling_sensitivity_metrics.npz"
    np.savez(npz_path, **{k: np.array(v, dtype=np.float64) for k, v in npz_fields.items()})
    md_path = report_dir / "coupling_sensitivity_report.md"
    _write_report(md_path, rows, args.subcycles)
    print(f"saved {npz_path}")
    print(f"saved {md_path}")


if __name__ == "__main__":
    main()
