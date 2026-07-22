#!/usr/bin/env python
"""CPU-only sweep over compact first-derivative boundary closures.

This script explores a one-parameter family of lower-wall compact closures
for the first derivative. The interior stencil stays fixed:

    (1/4) f'_{i-1} + f'_i + (1/4) f'_{i+1} = (3/4) (f_{i+1} - f_{i-1}) / dz

At the wall we use

    f'_0 + alpha f'_1 = (1/dz) sum_{j=0}^4 c_j(alpha) f_j

with the coefficients chosen to be exact on monomials 1, x, ..., x^4.
The upper wall is mirrored. For each alpha the script reports the operator
metrics most relevant to the NHQG instability: Neumann reconstruction
amplification, nonnormality of B = D1_psi D1_dir, and smooth-function probe
errors.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from fd_vertical_benchmark.operators import _direct_neumann_reduction_from_factors


@dataclass
class CaseMetrics:
    alpha: float
    p_neu_inf: float
    boundary_l1: float
    sym_b_max: float
    departure_normality_2: float
    cond_eigvec: float
    d1_cos_err: float
    d1_sin_err: float
    max_real_eig_b: float
    min_real_eig_b: float
    max_abs_imag_b: float


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nz", type=int, default=128)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=6.0)
    parser.add_argument("--num-alpha", type=int, default=121)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def _boundary_rhs_coeffs(alpha: float) -> np.ndarray:
    """Return c_j(alpha) for the wall closure exact on monomials degree <= 4."""
    nodes = np.arange(5, dtype=np.float64)
    vand = np.vstack([nodes ** k for k in range(5)])
    rhs = np.array(
        [
            0.0,          # d/dx 1
            1.0 + alpha,  # d/dx x at x=0 plus alpha * d/dx x at x=1
            2.0 * alpha,
            3.0 * alpha,
            4.0 * alpha,
        ],
        dtype=np.float64,
    )
    return np.linalg.solve(vand, rhs)


def _first_derivative_factors_compact_family(
    n_points: int, dz: float, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    if n_points < 6:
        raise ValueError("boundary search requires at least 6 grid points")

    A = np.zeros((n_points, n_points), dtype=np.float64)
    B = np.zeros((n_points, n_points), dtype=np.float64)
    inv_dz = 1.0 / dz

    coeff = _boundary_rhs_coeffs(alpha)
    A[0, 0] = 1.0
    A[0, 1] = alpha
    B[0, 0:5] = coeff * inv_dz

    for i in range(1, n_points - 1):
        A[i, i - 1] = 0.25
        A[i, i] = 1.0
        A[i, i + 1] = 0.25
        B[i, i - 1] = (-3.0 / 4.0) * inv_dz
        B[i, i + 1] = (3.0 / 4.0) * inv_dz

    A[-1, -2] = alpha
    A[-1, -1] = 1.0
    B[-1, -5:] = -coeff[::-1] * inv_dz
    return A, B


def _first_derivative_probe(D1_full: np.ndarray, z_full: np.ndarray) -> tuple[float, float]:
    cos_pi = np.cos(np.pi * z_full)
    sin_pi = np.sin(np.pi * z_full)
    dcos_pi = -np.pi * np.sin(np.pi * z_full)
    dsin_pi = np.pi * np.cos(np.pi * z_full)
    cos_err = float(np.max(np.abs(D1_full @ cos_pi - dcos_pi)))
    sin_err = float(np.max(np.abs(D1_full @ sin_pi - dsin_pi)))
    return cos_err, sin_err


def _score(metrics: CaseMetrics) -> tuple[float, float, float, float]:
    return (
        metrics.sym_b_max,
        metrics.departure_normality_2,
        metrics.p_neu_inf,
        metrics.d1_sin_err,
    )


def _evaluate_alpha(Nz: int, alpha: float) -> CaseMetrics | None:
    dz = 1.0 / Nz
    n_points = Nz + 1
    try:
        A1, B1 = _first_derivative_factors_compact_family(n_points, dz, alpha)
        D1_full = np.linalg.solve(A1, B1)
        P_neu, D1_psi = _direct_neumann_reduction_from_factors(A1, B1, dtype=np.float64)
    except np.linalg.LinAlgError:
        return None

    D1_dir = D1_full[1:-1, 1:-1]
    B = D1_psi @ D1_dir
    eig_b = np.linalg.eigvals(B)
    _, V = np.linalg.eig(B)
    sym_b = 0.5 * (B + B.T)
    comm = B.T @ B - B @ B.T
    z_full = np.linspace(0.0, 1.0, n_points)
    d1_cos_err, d1_sin_err = _first_derivative_probe(D1_full, z_full)

    return CaseMetrics(
        alpha=alpha,
        p_neu_inf=float(np.linalg.norm(P_neu, ord=np.inf)),
        boundary_l1=float(np.sum(np.abs(P_neu[0, :]))),
        sym_b_max=float(np.max(np.linalg.eigvalsh(sym_b))),
        departure_normality_2=float(np.linalg.norm(comm, 2)),
        cond_eigvec=float(np.linalg.cond(V)),
        d1_cos_err=d1_cos_err,
        d1_sin_err=d1_sin_err,
        max_real_eig_b=float(np.max(np.real(eig_b))),
        min_real_eig_b=float(np.min(np.real(eig_b))),
        max_abs_imag_b=float(np.max(np.abs(np.imag(eig_b)))),
    )


def main():
    args = _parse_args()
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.num_alpha)

    results: list[CaseMetrics] = []
    for alpha in alphas:
        case = _evaluate_alpha(args.Nz, float(alpha))
        if case is not None and np.isfinite(_score(case)).all():
            results.append(case)

    if not results:
        raise RuntimeError("No valid alpha values found")

    results.sort(key=_score)

    print(f"compact boundary search Nz={args.Nz}")
    print(f"searched alpha in [{args.alpha_min}, {args.alpha_max}] with {args.num_alpha} samples")
    print("ranking by (sym(B)_max, departure_normality_2, ||P_neu||_inf, D1 sin error)")
    print()
    for rank, case in enumerate(results[: args.top_k], start=1):
        print(
            f"#{rank:02d} alpha={case.alpha:.6f} "
            f"symBmax={case.sym_b_max:.6e} "
            f"dep={case.departure_normality_2:.6e} "
            f"Pinf={case.p_neu_inf:.6e} "
            f"rowL1={case.boundary_l1:.6e} "
            f"condV={case.cond_eigvec:.6e} "
            f"D1sin={case.d1_sin_err:.6e} "
            f"eigmax={case.max_real_eig_b:.6e} "
            f"eigmin={case.min_real_eig_b:.6e} "
            f"eigimag={case.max_abs_imag_b:.6e}"
        )


if __name__ == "__main__":
    main()
