#!/usr/bin/env python
"""Timing and memory profiling at various resolutions.

Micro-benchmarks: FFT, Jacobian, D_Z matmul, IMEX solve.
Macro: full step time at 256², 512², 1024² × Nz={32, 64}.
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import time

import jax
import jax.numpy as jnp
import numpy as np

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.solver import make_initial_state, imex_step


def benchmark_step(Nx, Nz, n_warmup=3, n_bench=10):
    """Benchmark a single IMEX step at given resolution."""
    k_max = np.pi * Nx / 20.0
    dt = 5e-4
    nu_q = 5.0 * dt / k_max ** 8

    cfg = NHQGConfig(
        Nx=Nx, Nz=Nz, L=20.0,
        Ra_tilde=100.0, sigma=1.0,
        dt=dt, nu_q=nu_q, hyper_order=4,
        nu_w=1e-3, nu_theta=1e-3,
        float_dtype='float64',
    )

    grid = make_grid(cfg)
    state = make_initial_state(grid, seed=0, amplitude=1e-3)

    # JIT compile
    step_jit = jax.jit(lambda s: imex_step(s, grid))
    state = step_jit(state)
    jax.block_until_ready(state.q_hat)

    # Warmup
    for _ in range(n_warmup):
        state = step_jit(state)
        jax.block_until_ready(state.q_hat)

    # Benchmark
    times = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        state = step_jit(state)
        jax.block_until_ready(state.q_hat)
        times.append(time.perf_counter() - t0)

    times = np.array(times)

    # Memory estimate: state = 3 complex arrays of (Nz+1, Nx, Nx//2+1)
    bytes_per = 16  # complex128
    state_mb = 3 * (Nz + 1) * Nx * (Nx // 2 + 1) * bytes_per / 1e6
    n_shells = grid.imex_inv.shape[0]
    imex_mb = n_shells * (Nz + 1) ** 2 * 8 / 1e6  # float64

    return {
        'Nx': Nx, 'Nz': Nz,
        'mean_ms': 1000 * times.mean(),
        'std_ms': 1000 * times.std(),
        'min_ms': 1000 * times.min(),
        'steps_per_sec': 1.0 / times.mean(),
        'state_MB': state_mb,
        'imex_MB': imex_mb,
        'n_shells': n_shells,
    }


def main():
    print(f"Device: {jax.devices()}")
    print()

    configs = [
        (64, 8),
        (128, 16),
        (256, 32),
    ]

    # Add larger if on GPU
    if jax.devices()[0].platform == 'gpu':
        configs.extend([
            (512, 32),
            (512, 64),
            (1024, 64),
        ])

    print(f"{'Nx':>6s} {'Nz':>4s} {'mean(ms)':>10s} {'std(ms)':>10s} "
          f"{'steps/s':>10s} {'state(MB)':>10s} {'imex(MB)':>10s} {'shells':>8s}")
    print("-" * 80)

    for Nx, Nz in configs:
        try:
            r = benchmark_step(Nx, Nz)
            print(f"{r['Nx']:6d} {r['Nz']:4d} {r['mean_ms']:10.2f} "
                  f"{r['std_ms']:10.2f} {r['steps_per_sec']:10.1f} "
                  f"{r['state_MB']:10.1f} {r['imex_MB']:10.1f} "
                  f"{r['n_shells']:8d}")
        except Exception as e:
            print(f"{Nx:6d} {Nz:4d} FAILED: {e}")


if __name__ == '__main__':
    main()
