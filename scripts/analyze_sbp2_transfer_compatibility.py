#!/usr/bin/env python
"""Measure SBP/CGL transfer defects and stage-local exchange drift."""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
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
    _cheb_to_dirichlet,
    _dirichlet_to_cheb,
    _finalize_state,
    _to_coeffs,
    _to_coeffs_1d,
    _to_nodal,
    _to_nodal_1d,
    balanced_sbp2_thermal_substep,
    explicit_rhs_dispatch,
    imex_implicit_solve,
    imex_mean_temp_solve,
    implicit_tendency,
    project_dirichlet,
    project_dirichlet_1d,
    _thermal_correction_tendency,
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
        help="Simulation times whose nearest checkpoints will be analyzed.",
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


def _norm_ratio(defect: np.ndarray, *refs: np.ndarray, ord_value=2) -> tuple[float, float]:
    num = float(np.linalg.norm(defect, ord=ord_value))
    denom = max((float(np.linalg.norm(ref, ord=ord_value)) for ref in refs), default=0.0)
    denom = max(denom, 1e-300)
    return num, num / denom


def _operator_metrics(grid) -> dict[str, float]:
    T = np.array(grid.cgl_to_sbp, dtype=np.float64)
    S = np.array(grid.sbp_to_cgl, dtype=np.float64)
    H = np.array(grid.sbp_H, dtype=np.float64)
    D1 = np.array(grid.sbp_D1, dtype=np.float64)
    V = np.array(grid.V, dtype=np.float64)
    V_inv = np.array(grid.V_inv, dtype=np.float64)
    G_coeff = np.array(grid.G_Z, dtype=np.float64)
    D_cgl = V @ G_coeff @ V_inv
    M_cc = np.diag(np.array(grid.cc_weights, dtype=np.float64))

    I_cgl = np.eye(T.shape[1], dtype=np.float64)
    I_sbp = np.eye(T.shape[0], dtype=np.float64)

    defects = {
        "ST_minus_I": S @ T - I_cgl,
        "TS_minus_I": T @ S - I_sbp,
        "mass_compat": M_cc @ S - T.T @ H,
        "deriv_S": D_cgl @ S - S @ D1,
        "deriv_T": D1 @ T - T @ D_cgl,
    }
    refs = {
        "ST_minus_I": (I_cgl,),
        "TS_minus_I": (I_sbp,),
        "mass_compat": (M_cc @ S, T.T @ H),
        "deriv_S": (D_cgl @ S, S @ D1),
        "deriv_T": (D1 @ T, T @ D_cgl),
    }

    out: dict[str, float] = {}
    for name, defect in defects.items():
        n2, r2 = _norm_ratio(defect, *refs[name], ord_value=2)
        ni, ri = _norm_ratio(defect, *refs[name], ord_value=np.inf)
        out[f"{name}_norm2"] = n2
        out[f"{name}_rel_norm2"] = r2
        out[f"{name}_norminf"] = ni
        out[f"{name}_rel_norminf"] = ri
    return out


def _weighted_rel_1d(diff: np.ndarray, ref: np.ndarray, weights: np.ndarray) -> float:
    num = float(np.sqrt(np.sum(weights * np.abs(diff) ** 2)))
    den = float(np.sqrt(np.sum(weights * np.abs(ref) ** 2)))
    return num / max(den, 1e-300)


def _weighted_rel_field(diff: np.ndarray, ref: np.ndarray, weights: np.ndarray) -> float:
    wz = weights[:, None, None]
    num = float(np.sqrt(np.sum(wz * np.abs(diff) ** 2)))
    den = float(np.sqrt(np.sum(wz * np.abs(ref) ** 2)))
    return num / max(den, 1e-300)


def _thermal_roundtrip_state(state: State, grid) -> tuple[State, dict[str, float]]:
    weights = np.array(grid.cc_weights, dtype=np.float64)
    T = grid.cgl_to_sbp
    S = grid.sbp_to_cgl

    w_cheb = _dirichlet_to_cheb(state.w_hat, grid.dirichlet_stencil)
    th_cheb = _dirichlet_to_cheb(state.th_hat, grid.dirichlet_stencil)
    w_cgl = _to_nodal(w_cheb, grid.V)
    th_cgl = _to_nodal(th_cheb, grid.V)
    th_bar_cgl = _to_nodal_1d(state.th_bar, grid.V)

    w_rt = jnp.einsum("ij,j...->i...", S, jnp.einsum("ij,j...->i...", T, w_cgl))
    th_rt = jnp.einsum("ij,j...->i...", S, jnp.einsum("ij,j...->i...", T, th_cgl))
    th_bar_rt = jnp.einsum("ij,j->i", S, jnp.einsum("ij,j->i", T, th_bar_cgl))

    w_cheb_rt = project_dirichlet(_to_coeffs(w_rt, grid.V_inv), grid.proj_dirichlet)
    th_cheb_rt = project_dirichlet(_to_coeffs(th_rt, grid.V_inv), grid.proj_dirichlet)
    th_bar_rt_coeff = project_dirichlet_1d(_to_coeffs_1d(th_bar_rt, grid.V_inv), grid.proj_dirichlet)

    rt_state = State(
        state.q_hat,
        _cheb_to_dirichlet(w_cheb_rt, grid.dirichlet_pinv),
        _cheb_to_dirichlet(th_cheb_rt, grid.dirichlet_pinv),
        th_bar_rt_coeff,
    )
    rt_state = _finalize_state(rt_state, grid)

    metrics = {
        "w_roundtrip_rel": _weighted_rel_field(np.array(w_rt - w_cgl), np.array(w_cgl), weights),
        "theta_roundtrip_rel": _weighted_rel_field(np.array(th_rt - th_cgl), np.array(th_cgl), weights),
        "Theta_roundtrip_rel": _weighted_rel_1d(np.array(th_bar_rt - th_bar_cgl), np.array(th_bar_cgl), weights),
        "w_roundtrip_max": float(np.max(np.abs(np.array(w_rt - w_cgl)))),
        "theta_roundtrip_max": float(np.max(np.abs(np.array(th_rt - th_cgl)))),
        "Theta_roundtrip_max": float(np.max(np.abs(np.array(th_bar_rt - th_bar_cgl)))),
    }
    return rt_state, metrics


def _diag_snapshot(label: str, state: State, grid) -> dict[str, float]:
    diag = compute_diagnostics(state, grid)
    return {
        "label": label,
        "R_ex_d": float(diag["mean_theta_exchange_residual_dealiased"]),
        "R_ex_d_rel": float(diag["mean_theta_exchange_residual_dealiased_rel"]),
        "R_ex_sbp": float(diag["mean_theta_exchange_residual_sbp"]),
        "R_ex_sbp_rel": float(diag["mean_theta_exchange_residual_sbp_rel"]),
        "B_sbp": float(diag["mean_theta_exchange_boundary_sbp"]),
        "Nu_d": float(diag["Nusselt_dealiased"]),
        "max_w": float(diag["max_w"]),
        "max_theta": float(diag["max_theta"]),
    }


def _stage_probe(state: State, grid) -> list[dict[str, float]]:
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
    corrected1 = balanced_sbp2_thermal_substep(predictor1, grid, sub_dt=alpha)
    C1 = _thermal_correction_tendency(predictor1, corrected1, alpha)

    E2 = explicit_rhs_dispatch(corrected1, base_grid)
    I1 = implicit_tendency(corrected1, base_grid)
    R_q2 = q_n + dt * (delta * E1.q_hat + (1 - delta) * E2.q_hat) \
         + omg * I1.q_hat \
         - omg * grid.diss_rate_q[None, :, :] * corrected1.q_hat
    R_w2 = w_n + dt * (delta * E1.w_hat + (1 - delta) * E2.w_hat) \
         + omg * I1.w_hat \
         - omg * grid.diss_rate_w[None, :, :] * corrected1.w_hat
    R_th2 = th_n + dt * (delta * E1.th_hat + (1 - delta) * E2.th_hat) \
          + omg * I1.th_hat \
          - omg * grid.diss_rate_th[None, :, :] * corrected1.th_hat \
          + omg * C1.th_hat
    R_th_bar2 = th_bar_n + dt * (delta * E1.th_bar + (1 - delta) * E2.th_bar) \
             + omg * I1.th_bar \
             + omg * C1.th_bar
    q2p, w2p, th2p = imex_implicit_solve(R_q2, R_w2, R_th2, base_grid)
    th_bar2p = imex_mean_temp_solve(R_th_bar2, base_grid)
    predictor2 = State(q2p, w2p, th2p, th_bar2p)
    corrected2 = balanced_sbp2_thermal_substep(predictor2, grid, sub_dt=alpha)

    return [
        _diag_snapshot("saved_state", state, grid),
        _diag_snapshot("stage1_predictor", predictor1, grid),
        _diag_snapshot("stage1_corrected", corrected1, grid),
        _diag_snapshot("stage2_predictor", predictor2, grid),
        _diag_snapshot("stage2_corrected", corrected2, grid),
    ]


def _write_report(path: Path, run_dir: Path, selected: list[Path],
                  op_metrics: dict[str, float], rows: list[dict[str, object]]):
    lines: list[str] = []
    lines.append("# SBP2 Transfer Compatibility Report")
    lines.append("")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append(f"- checkpoints: {', '.join(f'`{path.name}`' for path in selected)}")
    lines.append("")
    lines.append("## Static Operator Defects")
    lines.append("")
    lines.append("| defect | norm2 | rel_norm2 | norminf | rel_norminf |")
    lines.append("|---|---:|---:|---:|---:|")
    for prefix in ["ST_minus_I", "TS_minus_I", "mass_compat", "deriv_S", "deriv_T"]:
        lines.append(
            f"| `{prefix}` | "
            f"{op_metrics[prefix + '_norm2']:.6e} | "
            f"{op_metrics[prefix + '_rel_norm2']:.6e} | "
            f"{op_metrics[prefix + '_norminf']:.6e} | "
            f"{op_metrics[prefix + '_rel_norminf']:.6e} |"
        )
    lines.append("")
    lines.append("## Checkpoint Roundtrip Test")
    lines.append("")
    lines.append(
        "| checkpoint | t | `R_ex_d` | `R_ex_sbp` | `R_ex_d` after roundtrip | "
        "`R_ex_sbp` after roundtrip | `w` rel | `theta` rel | `Theta` rel |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| `{row['checkpoint']}` | {row['t']:.2f} | "
            f"{row['saved_R_ex_d']:.6e} | {row['saved_R_ex_sbp']:.6e} | "
            f"{row['roundtrip_R_ex_d']:.6e} | {row['roundtrip_R_ex_sbp']:.6e} | "
            f"{row['w_roundtrip_rel']:.6e} | {row['theta_roundtrip_rel']:.6e} | "
            f"{row['Theta_roundtrip_rel']:.6e} |"
        )
    lines.append("")
    lines.append("## Stage-Local Exchange Audit")
    lines.append("")
    lines.append(
        "| checkpoint | stage state | `R_ex_d` | `R_ex_d_rel` | `R_ex_sbp` | "
        "`R_ex_sbp_rel` | `B_sbp` |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        for stage in row["stage_rows"]:
            lines.append(
                f"| `{row['checkpoint']}` | `{stage['label']}` | "
                f"{stage['R_ex_d']:.6e} | {stage['R_ex_d_rel']:.6e} | "
                f"{stage['R_ex_sbp']:.6e} | {stage['R_ex_sbp_rel']:.6e} | "
                f"{stage['B_sbp']:.6e} |"
            )
    lines.append("")
    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append(
        "- If the pure roundtrip already changes `R_ex_d` materially while `R_ex_sbp` "
        "stays at roundoff, the dominant defect is in the representation-transfer layer."
    )
    lines.append(
        "- If the stage-corrected states already show large `R_ex_d` immediately after "
        "the SBP corrector returns to CGL, that again points to transfer inconsistency."
    )
    lines.append(
        "- If the stage-corrected states remain benign and the large residual appears only "
        "after subsequent stage assembly, the remaining issue is in the full-step coupling."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = _parse_args()
    run_dir = resolve_existing_output_path(args.run_dir)
    args.run_dir = str(run_dir)
    report_dir = (
        Path(normalize_output_dir(args.report_dir))
        if args.report_dir
        else (run_dir / "spectra" / "transfer_compatibility")
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_config(args)
    grid = make_grid(cfg)
    selected = _select_checkpoints(run_dir, args.dt, args.sample_times)
    op_metrics = _operator_metrics(grid)

    rows: list[dict[str, object]] = []
    scalar_columns: dict[str, list[float]] = {
        "t": [],
        "step": [],
        "saved_R_ex_d": [],
        "saved_R_ex_sbp": [],
        "roundtrip_R_ex_d": [],
        "roundtrip_R_ex_sbp": [],
        "w_roundtrip_rel": [],
        "theta_roundtrip_rel": [],
        "Theta_roundtrip_rel": [],
        "w_roundtrip_max": [],
        "theta_roundtrip_max": [],
        "Theta_roundtrip_max": [],
        "stage1_predictor_R_ex_d": [],
        "stage1_corrected_R_ex_d": [],
        "stage2_predictor_R_ex_d": [],
        "stage2_corrected_R_ex_d": [],
        "stage1_predictor_R_ex_sbp": [],
        "stage1_corrected_R_ex_sbp": [],
        "stage2_predictor_R_ex_sbp": [],
        "stage2_corrected_R_ex_sbp": [],
    }

    for ckpt in selected:
        state, step, t = load_checkpoint(str(ckpt), dtype=jnp.complex128)
        saved_diag = _diag_snapshot("saved_state", state, grid)
        rt_state, rt_metrics = _thermal_roundtrip_state(state, grid)
        rt_diag = _diag_snapshot("roundtrip_state", rt_state, grid)
        stage_rows = _stage_probe(state, grid)

        row: dict[str, object] = {
            "checkpoint": ckpt.name,
            "step": int(step),
            "t": float(t),
            "saved_R_ex_d": saved_diag["R_ex_d"],
            "saved_R_ex_sbp": saved_diag["R_ex_sbp"],
            "roundtrip_R_ex_d": rt_diag["R_ex_d"],
            "roundtrip_R_ex_sbp": rt_diag["R_ex_sbp"],
            "stage_rows": stage_rows,
            **rt_metrics,
        }
        rows.append(row)

        scalar_columns["t"].append(float(t))
        scalar_columns["step"].append(float(step))
        scalar_columns["saved_R_ex_d"].append(float(saved_diag["R_ex_d"]))
        scalar_columns["saved_R_ex_sbp"].append(float(saved_diag["R_ex_sbp"]))
        scalar_columns["roundtrip_R_ex_d"].append(float(rt_diag["R_ex_d"]))
        scalar_columns["roundtrip_R_ex_sbp"].append(float(rt_diag["R_ex_sbp"]))
        for key in [
            "w_roundtrip_rel",
            "theta_roundtrip_rel",
            "Theta_roundtrip_rel",
            "w_roundtrip_max",
            "theta_roundtrip_max",
            "Theta_roundtrip_max",
        ]:
            scalar_columns[key].append(float(rt_metrics[key]))
        label_map = {stage["label"]: stage for stage in stage_rows}
        for key, label in [
            ("stage1_predictor_R_ex_d", "stage1_predictor"),
            ("stage1_corrected_R_ex_d", "stage1_corrected"),
            ("stage2_predictor_R_ex_d", "stage2_predictor"),
            ("stage2_corrected_R_ex_d", "stage2_corrected"),
        ]:
            scalar_columns[key].append(float(label_map[label]["R_ex_d"]))
        for key, label in [
            ("stage1_predictor_R_ex_sbp", "stage1_predictor"),
            ("stage1_corrected_R_ex_sbp", "stage1_corrected"),
            ("stage2_predictor_R_ex_sbp", "stage2_predictor"),
            ("stage2_corrected_R_ex_sbp", "stage2_corrected"),
        ]:
            scalar_columns[key].append(float(label_map[label]["R_ex_sbp"]))

        print(
            f"{ckpt.name}: t={t:.2f} "
            f"saved(R_ex_d={saved_diag['R_ex_d']:.3e}, R_ex_sbp={saved_diag['R_ex_sbp']:.3e}) "
            f"roundtrip(R_ex_d={rt_diag['R_ex_d']:.3e}, R_ex_sbp={rt_diag['R_ex_sbp']:.3e}) "
            f"stage1_corr(R_ex_d={label_map['stage1_corrected']['R_ex_d']:.3e}, "
            f"R_ex_sbp={label_map['stage1_corrected']['R_ex_sbp']:.3e}) "
            f"stage2_corr(R_ex_d={label_map['stage2_corrected']['R_ex_d']:.3e}, "
            f"R_ex_sbp={label_map['stage2_corrected']['R_ex_sbp']:.3e})",
            flush=True,
        )

    npz_path = report_dir / "transfer_compatibility_metrics.npz"
    save_kwargs = {key: np.array(value, dtype=np.float64) for key, value in scalar_columns.items()}
    save_kwargs.update({key: np.array(value, dtype=np.float64) for key, value in op_metrics.items()})
    np.savez(npz_path, **save_kwargs)

    md_path = report_dir / "transfer_compatibility_report.md"
    _write_report(md_path, run_dir, selected, op_metrics, rows)

    print(f"saved {npz_path}")
    print(f"saved {md_path}")


if __name__ == "__main__":
    main()
