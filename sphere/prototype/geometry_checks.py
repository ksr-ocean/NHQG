#!/usr/bin/env python
"""Basic stereographic polar-cap geometry checks.

These checks are intentionally small and dependency-light. They establish the
geometry and weighted area measure that any Julia, Dedalus, or JAX spherical
prototype must reproduce.
"""

from __future__ import annotations

import math

import numpy as np


def r_jet_from_latitude(phi_jet: float) -> float:
    """Projected cap radius for jet latitude `phi_jet` in radians."""
    return math.tan(math.pi / 4.0 - 0.5 * phi_jet)


def mu(r: np.ndarray) -> np.ndarray:
    """Stereographic conformal factor for the unit sphere."""
    return 4.0 / (1.0 + r * r) ** 2


def exact_cap_area(r_jet: float) -> float:
    """Exact spherical area of the projected cap."""
    return 4.0 * math.pi * r_jet * r_jet / (1.0 + r_jet * r_jet)


def quadrature_cap_area(r_jet: float, n: int = 2048) -> float:
    """High-order Gauss-Legendre quadrature of int mu(r) r dr dphi."""
    x, w = np.polynomial.legendre.leggauss(n)
    r = 0.5 * r_jet * (x + 1.0)
    wr = 0.5 * r_jet * w
    return float(2.0 * math.pi * np.sum(wr * mu(r) * r))


def flat_disk_area(r_jet: float) -> float:
    """Flat projected disk area."""
    return math.pi * r_jet * r_jet


def run_checks() -> None:
    latitudes_deg = [75.0, 60.0, 45.0, 30.0]
    print("phi_jet_deg r_jet        area_exact   area_quad    rel_err")
    for lat_deg in latitudes_deg:
        r_jet = r_jet_from_latitude(math.radians(lat_deg))
        exact = exact_cap_area(r_jet)
        quad = quadrature_cap_area(r_jet)
        rel_err = abs(quad - exact) / exact
        print(f"{lat_deg:11.1f} {r_jet:12.8f} {exact:12.8f} {quad:12.8f} {rel_err:9.2e}")
        if rel_err > 1e-12:
            raise AssertionError(f"area quadrature failed for latitude {lat_deg}")

    print()
    print("small_cap_r area_sphere/area_flat expected_limit")
    for r_jet in [1e-1, 1e-2, 1e-3, 1e-4]:
        ratio = exact_cap_area(r_jet) / flat_disk_area(r_jet)
        print(f"{r_jet:11.1e} {ratio:22.15f} 4")
        if abs(ratio - 4.0) > 5.0 * r_jet * r_jet:
            raise AssertionError("small-cap area ratio did not approach 4")

    print()
    print("geometry checks passed")


if __name__ == "__main__":
    run_checks()
