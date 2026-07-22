#!/usr/bin/env python
"""Draft Dedalus benchmark for the upright standard-f NHQG case.

This script is intentionally isolated from the main JAX solver. It is a
concrete starting point for an independent benchmark in Dedalus v3, but it was
not executed here because Dedalus is not installed on this node.

Key design choices:
- Fourier x/y, Chebyshev z
- fluctuation zero mode constrained to zero
- Dirichlet boundary conditions built in through S(z)=z(1-z) prefactors for
  w, theta, and Theta_bar'
- psi solved diagnostically from q through a horizontal Poisson constraint
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)

K_C = 1.3048
L_C = 2.0 * math.pi / K_C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--Nz", type=int, default=128)
    parser.add_argument("--L", type=float, default=10.0 * L_C)
    parser.add_argument("--Ra", type=float, default=100.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--t-final", type=float, default=5.0)
    parser.add_argument("--nu-q", type=float, default=1.0)
    parser.add_argument("--nu-w", type=float, default=1.0)
    parser.add_argument("--nu-theta", type=float, default=1.0)
    parser.add_argument("--drag", type=float, default=0.0)
    parser.add_argument("--hyper-order", type=int, default=1)
    parser.add_argument("--mean-temp-eps-sq", type=float, default=1.0)
    parser.add_argument(
        "--thermal-closure",
        choices=["fixed_conduction", "evolve_mean"],
        default="evolve_mean",
    )
    parser.add_argument("--amplitude", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snapshot-dt", type=float, default=0.25)
    parser.add_argument("--timeseries-dt", type=float, default=0.05)
    parser.add_argument("--log-cadence", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_miquel_zero_tilt_dedalus_evolvemean_Nx128_Nz128_dt5e5_t5",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    try:
        import dedalus.public as d3
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Dedalus is not installed in this environment. "
            "Create/activate a Dedalus v3 environment first, then rerun."
        ) from exc

    coords = d3.CartesianCoordinates("x", "y", "z")
    dist = d3.Distributor(coords, dtype=np.float64)

    xbasis = d3.RealFourier(coords["x"], size=args.Nx, bounds=(0, args.L), dealias=3 / 2)
    ybasis = d3.RealFourier(coords["y"], size=args.Nx, bounds=(0, args.L), dealias=3 / 2)
    zbasis = d3.Chebyshev(coords["z"], size=args.Nz + 1, bounds=(0, 1), dealias=3 / 2)

    q = dist.Field(name="q", bases=(xbasis, ybasis, zbasis))
    psi = dist.Field(name="psi", bases=(xbasis, ybasis, zbasis))
    wt = dist.Field(name="wt", bases=(xbasis, ybasis, zbasis))
    tt = dist.Field(name="tt", bases=(xbasis, ybasis, zbasis))

    variables = [q, psi, wt, tt]
    mt = None
    if args.thermal_closure == "evolve_mean":
        mt = dist.Field(name="mt", bases=zbasis)
        variables.append(mt)

    z = dist.local_grid(zbasis)
    S = dist.Field(name="S", bases=zbasis)
    S["g"] = z * (1.0 - z)

    dx = lambda A: d3.Differentiate(A, coords["x"])
    dy = lambda A: d3.Differentiate(A, coords["y"])
    dz = lambda A: d3.Differentiate(A, coords["z"])
    lap_h = lambda A: dx(dx(A)) + dy(dy(A))

    u = -dy(psi)
    v = dx(psi)
    w = S * wt
    theta = S * tt
    jac_q = u * dx(q) + v * dy(q)
    jac_w = u * dx(w) + v * dy(w)
    jac_theta = u * dx(theta) + v * dy(theta)

    nu_q = args.nu_q
    nu_w = args.nu_w
    kappa_h = args.nu_theta / args.sigma
    Ra_sigma = args.Ra / args.sigma
    mean_kappa = args.mean_temp_eps_sq / args.sigma

    namespace = locals()
    problem = d3.IVP(variables, namespace=namespace)

    # Nonzero horizontal modes only.
    nz_cond = "(nx != 0) or (ny != 0)"
    z0_cond = "(nx == 0) and (ny == 0)"

    problem.add_equation(
        "dt(q) - dz(w) - nu_q*(dx(dx(q)) + dy(dy(q))) = -jac_q",
        condition=nz_cond,
    )
    problem.add_equation(
        "dx(dx(psi)) + dy(dy(psi)) + q = 0",
        condition=nz_cond,
    )
    problem.add_equation(
        "dt(w) + dz(psi) - Ra_sigma*theta - nu_w*(dx(dx(w)) + dy(dy(w))) = -jac_w",
        condition=nz_cond,
    )

    if args.thermal_closure == "evolve_mean":
        thbar = S * mt
        dthbar = dz(thbar)
        jac_theta_rhs = jac_theta + dthbar * w
        wtheta_bar = d3.Average(d3.Average(w * theta, coords["x"]), coords["y"])
        problem.add_equation(
            "dt(theta) - w - kappa_h*(dx(dx(theta)) + dy(dy(theta))) = -jac_theta_rhs",
            condition=nz_cond,
        )
        problem.add_equation(
            "dt(thbar) - mean_kappa*dz(dz(thbar)) = -mean_temp_eps_sq*dz(wtheta_bar)"
        )
    else:
        thbar = None
        wtheta_bar = d3.Average(d3.Average(w * theta, coords["x"]), coords["y"])
        problem.add_equation(
            "dt(theta) - w - kappa_h*(dx(dx(theta)) + dy(dy(theta))) = -jac_theta",
            condition=nz_cond,
        )

    # Fluctuation zero mode matches the JAX zero_mode projection.
    problem.add_equation("q = 0", condition=z0_cond)
    problem.add_equation("psi = 0", condition=z0_cond)
    problem.add_equation("wt = 0", condition=z0_cond)
    problem.add_equation("tt = 0", condition=z0_cond)

    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = args.t_final

    xg, yg, zg = dist.local_grids(xbasis, ybasis, zbasis)
    rng = np.random.default_rng(args.seed)
    q["g"] = args.amplitude * rng.standard_normal(q["g"].shape) * np.sin(np.pi * zg)
    psi["g"] = 0.0
    wt["g"] = 0.0
    tt["g"] = 0.0
    if mt is not None:
        mt["g"] = 0.0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots = solver.evaluator.add_file_handler(str(output_dir / "snapshots"), sim_dt=args.snapshot_dt, max_writes=50)
    snapshots.add_task(q, name="q")
    snapshots.add_task(psi, name="psi")
    snapshots.add_task(w, name="w")
    snapshots.add_task(theta, name="theta")
    if thbar is not None:
        snapshots.add_task(thbar, name="theta_bar")

    vol_avg_tw = d3.Integrate(wtheta_bar, coords["z"])
    nusselt = 1 + vol_avg_tw

    timeseries = solver.evaluator.add_file_handler(str(output_dir / "timeseries"), sim_dt=args.timeseries_dt, max_writes=200)
    timeseries.add_task(wtheta_bar, name="wtheta_bar")
    timeseries.add_task(vol_avg_tw, name="vol_avg_tw")
    timeseries.add_task(nusselt, name="Nusselt")
    if thbar is not None:
        timeseries.add_task(thbar, name="theta_bar")

    logger.info("Starting Dedalus NHQG benchmark")
    logger.info(
        "Nx=%d Nz=%d L=%.4f Ra=%.3f dt=%.3e t_final=%.3f closure=%s",
        args.Nx,
        args.Nz,
        args.L,
        args.Ra,
        args.dt,
        args.t_final,
        args.thermal_closure,
    )

    while solver.proceed:
        solver.step(args.dt)
        if solver.iteration % args.log_cadence == 0:
            logger.info(
                "iter=%d sim_time=%.6f dt=%.3e",
                solver.iteration,
                solver.sim_time,
                args.dt,
            )

    logger.info("Completed at sim_time=%.6f, iteration=%d", solver.sim_time, solver.iteration)


if __name__ == "__main__":
    main()
