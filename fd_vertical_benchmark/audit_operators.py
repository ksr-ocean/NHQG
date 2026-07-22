#!/usr/bin/env python
"""Audit FD vertical operators for spectrum, conditioning, and BC response."""

from __future__ import annotations

import argparse

import numpy as np

from fd_vertical_benchmark.config import FDBenchmarkConfig
from fd_vertical_benchmark.operators import make_grid


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nz", type=int, default=128)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "centered2:centered2:projected",
            "compact4:centered2:projected",
            "compact4:compact4_raw:projected",
            "sbp42:centered2:direct",
            "sbp42:sbp42_energy:direct",
        ],
    )
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--Nx", type=int, default=16)
    return parser.parse_args()


def _first_derivative_probe(D1_full: np.ndarray, z_full: np.ndarray) -> dict[str, float]:
    cos_pi = np.cos(np.pi * z_full)
    dcos_pi = -np.pi * np.sin(np.pi * z_full)
    sin_pi = np.sin(np.pi * z_full)
    dsin_pi = np.pi * np.cos(np.pi * z_full)

    cos_num = D1_full @ cos_pi
    sin_num = D1_full @ sin_pi
    return {
        "cos_pi_max_err": float(np.max(np.abs(cos_num - dcos_pi))),
        "sin_pi_max_err": float(np.max(np.abs(sin_num - dsin_pi))),
        "cos_pi_top_num": float(cos_num[0]),
        "cos_pi_bot_num": float(cos_num[-1]),
        "sin_pi_top_num": float(sin_num[0]),
        "sin_pi_bot_num": float(sin_num[-1]),
    }


def _second_derivative_probe(D2_dir: np.ndarray, z_int: np.ndarray) -> dict[str, float]:
    f = np.sin(np.pi * z_int)
    d2f_exact = -(np.pi ** 2) * np.sin(np.pi * z_int)
    d2f_num = D2_dir @ f
    return {
        "sin_pi_max_err": float(np.max(np.abs(d2f_num - d2f_exact))),
        "sin_pi_rms_err": float(np.sqrt(np.mean((d2f_num - d2f_exact) ** 2))),
    }


def _stage_condition_metrics(B: np.ndarray, dt: float, gamma: float, Ra_sigma: float) -> list[tuple[float, float, float]]:
    ksq_values = [0.5, 0.718**2, 0.979**2, 1.0, 2.0]
    metrics: list[tuple[float, float, float]] = []
    for ksq in ksq_values:
        alpha_q = 1.0 + gamma * dt * ksq
        alpha_w = 1.0 + gamma * dt * ksq
        alpha_th = 1.0 + gamma * dt * ksq
        alpha_w_eff = alpha_w - (gamma * dt) ** 2 * Ra_sigma / alpha_th
        A = alpha_w_eff * np.eye(B.shape[0]) - (gamma * dt) ** 2 * (1.0 / ksq) / alpha_q * B
        svals = np.linalg.svd(A, compute_uv=False)
        metrics.append((ksq, float(svals[-1]), float(svals[0] / svals[-1])))
    return metrics


def _energy_symmetric_part(D2_dir: np.ndarray, norm_weights: np.ndarray) -> np.ndarray:
    H_int = np.diag(norm_weights[1:-1])
    return 0.5 * (H_int @ D2_dir + D2_dir.T @ H_int)


def main():
    args = _parse_args()

    for case in args.cases:
        parts = case.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"Invalid case {case!r}; expected "
                f"'vertical_derivative:vertical_second_derivative[:psi_neumann_treatment]'"
            )
        if len(parts) == 2:
            vertical_derivative, vertical_second_derivative = parts
            psi_neumann_treatment = "projected"
        else:
            vertical_derivative, vertical_second_derivative, psi_neumann_treatment = parts
        cfg = FDBenchmarkConfig(
            Nx=args.Nx,
            Nz=args.Nz,
            dt=args.dt,
            vertical_derivative=vertical_derivative,
            vertical_second_derivative=vertical_second_derivative,
            psi_neumann_treatment=psi_neumann_treatment,
            float_dtype="float64",
        )
        g = make_grid(cfg)

        D1_full = np.array(g.D1_full)
        D1_dir = np.array(g.D1_dir)
        D1_psi = np.array(g.D1_psi)
        D2_dir = np.array(g.D2_dir)
        P_neu = np.array(g.P_neu)
        B = D1_psi @ D1_dir

        eig_B = np.linalg.eigvals(B)
        eig_D2 = np.linalg.eigvals(D2_dir)
        _, V = np.linalg.eig(B)
        svals_B = np.linalg.svd(B, compute_uv=False)
        sym_B = 0.5 * (B + B.T)
        eig_sym_B = np.linalg.eigvalsh(sym_B)
        comm = B.T @ B - B @ B.T
        sym_H_D2 = _energy_symmetric_part(D2_dir, np.array(g.norm_weights))
        eig_sym_H_D2 = np.linalg.eigvalsh(sym_H_D2)

        print(
            f"case={vertical_derivative}:{vertical_second_derivative}:{psi_neumann_treatment}"
        )
        print(f"  Nz={args.Nz}")
        print(
            f"  eig(B): max_real={np.max(np.real(eig_B)):.6e} "
            f"min_real={np.min(np.real(eig_B)):.6e} "
            f"max_abs_imag={np.max(np.abs(np.imag(eig_B))):.6e}"
        )
        print(
            f"  eig(D2_dir): max_real={np.max(np.real(eig_D2)):.6e} "
            f"min_real={np.min(np.real(eig_D2)):.6e}"
        )
        print(
            f"  sym(H_int D2_dir): max_eig={np.max(eig_sym_H_D2):.6e} "
            f"min_eig={np.min(eig_sym_H_D2):.6e}"
        )
        print(
            f"  cond(D1_dir)={np.linalg.cond(D1_dir):.6e} "
            f"cond(D1_psi)={np.linalg.cond(D1_psi):.6e} "
            f"cond(B)={svals_B[0] / svals_B[-1]:.6e}"
        )
        print(
            f"  ||D1_full||inf={np.linalg.norm(D1_full, ord=np.inf):.6e} "
            f"||P_neu||inf={np.linalg.norm(P_neu, ord=np.inf):.6e}"
        )
        print(
            f"  boundary_row_l1(top/bot)=({np.sum(np.abs(P_neu[0,:])):.6e}, "
            f"{np.sum(np.abs(P_neu[-1,:])):.6e})"
        )
        print(
            f"  departure_normality_2={np.linalg.norm(comm, 2):.6e} "
            f"cond_eigvec={np.linalg.cond(V):.6e}"
        )
        print(
            f"  sym(B): max_eig={np.max(eig_sym_B):.6e} "
            f"min_eig={np.min(eig_sym_B):.6e}"
        )

        probe = _first_derivative_probe(D1_full, np.array(g.z_full))
        probe2 = _second_derivative_probe(D2_dir, np.array(g.z_int))
        print(
            f"  D1 probe cos(pi z): max_err={probe['cos_pi_max_err']:.6e} "
            f"top={probe['cos_pi_top_num']:.6e} bot={probe['cos_pi_bot_num']:.6e}"
        )
        print(
            f"  D1 probe sin(pi z): max_err={probe['sin_pi_max_err']:.6e} "
            f"top={probe['sin_pi_top_num']:.6e} bot={probe['sin_pi_bot_num']:.6e}"
        )
        print(
            f"  D2 probe sin(pi z): max_err={probe2['sin_pi_max_err']:.6e} "
            f"rms_err={probe2['sin_pi_rms_err']:.6e}"
        )

        for ksq, min_sv, cond_A in _stage_condition_metrics(
            B, float(g.dt), float(g.gamma_imex), float(g.Ra_sigma)
        ):
            print(f"  A(ksq={ksq:.6f}): min_sv={min_sv:.6e} cond={cond_A:.6e}")
        print()


if __name__ == "__main__":
    main()
