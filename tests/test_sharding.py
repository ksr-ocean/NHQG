"""M3 gates: state sharding wiring + sharded-compute equivalence.

Findings encoded here (2026-07-22):

- The state layout (Nz*, Nx, Nk) has exactly ONE shardable axis on 2
  devices: kx (axis 1). Nz+1/Nz-1 are odd (Nz even) and Nk = Nx/2+1 is
  odd, and JAX 0.10 NamedSharding rejects uneven splits (IndivisibleError)
  — so 'z' sharding is unavailable without a padded layout.
- The XLA *CPU* backend cannot execute a partitioned FFT whose contracted
  axis is sharded (fft_thunk RET_CHECK on non-major layouts), so the
  full-step 1-vs-2-device equivalence gate can only run on real GPUs; it
  auto-skips elsewhere. CPU still certifies the wiring and the sharded
  IMEX implicit solve (the per-shell gather/matmul core), which is where
  layout bugs would bite.

Subprocess tests: XLA_FLAGS=--xla_force_host_platform_device_count=2 must
be set before jax initializes, which pytest's shared process cannot do.
"""

import os
import subprocess
import sys

import pytest

_PROLOGUE = r"""
import jax
import numpy as np
assert len(jax.devices()) >= 2, f"need 2 devices, got {len(jax.devices())}"

from nhqg.config import NHQGConfig
from nhqg.grid import make_grid
from nhqg.solver import make_initial_state, imex_step, imex_implicit_solve
from nhqg.sharding import make_mesh, shard_state

cfg = NHQGConfig(Nx=32, Nz=16, L=20.0, Ra_tilde=100.0, sigma=1.0, beta=0.0,
                 Ld=float('inf'), dt=1e-3, float_dtype='float64',
                 nu_q=1.0, nu_w=1.0, nu_theta=1.0, hyper_order=1,
                 w_bc_top='neumann', thermal_closure='evolve_mean',
                 mean_exchange_discretization='balanced_sbp2_pc',
                 sbp_corrector_substeps=2,
                 nonlinear_advection='flux', horizontal_dealiasing='23_rule')
g = make_grid(cfg)
s_ref = make_initial_state(g, seed=0, amplitude=1e-3)
mesh = make_mesh(2)

def rel(x, y):
    x, y = np.asarray(x), np.asarray(y)
    n = float(np.max(np.abs(x))) or 1.0
    return float(np.max(np.abs(x - y))) / n
"""

_WIRING = _PROLOGUE + r"""
s = shard_state(s_ref, mesh, "kx")
for name in ("q_hat", "w_hat", "th_hat"):
    arr = getattr(s, name)
    shards = arr.addressable_shards
    assert len(shards) == 2
    full = arr.shape
    assert shards[0].data.shape == (full[0], full[1] // 2, full[2]), \
        (name, shards[0].data.shape, full)
assert s.th_bar.sharding.is_fully_replicated
assert not s.q_hat.sharding.is_fully_replicated
print("WIRING-OK")
"""

_Z_REJECTED = _PROLOGUE + r"""
try:
    shard_state(s_ref, mesh, "z")
    print("Z-UNEXPECTEDLY-OK")   # JAX gained uneven sharding: revisit M3 notes
except Exception as e:
    assert "Indivisible" in type(e).__name__ or "divide" in str(e), e
    print("Z-REJECTED-AS-EXPECTED")
"""

_IMPLICIT = _PROLOGUE + r"""
# One explicit half-step to get nontrivial RHS fields, then compare the
# jitted implicit solve on sharded vs unsharded inputs.
from nhqg.sharding import state_sharding_specs
from jax.sharding import NamedSharding

@jax.jit
def solve(rq, rw, rth):
    return imex_implicit_solve(rq, rw, rth, g)

rng = np.random.default_rng(3)
def rand_like(shape):
    return jnp_arr(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
import jax.numpy as jnp
def jnp_arr(a):
    return jnp.asarray(a)

rq = rand_like(s_ref.q_hat.shape)
rw = rand_like(s_ref.w_hat.shape)
rth = rand_like(s_ref.th_hat.shape)

ref = solve(rq, rw, rth)

spec3, _ = state_sharding_specs("kx")
sh = NamedSharding(mesh, spec3)
out = solve(jax.device_put(rq, sh), jax.device_put(rw, sh), jax.device_put(rth, sh))

errs = [rel(a, b) for a, b in zip(ref, out)]
assert max(errs) < 1e-13, errs
print("IMPLICIT-MATCH", errs)
"""

_FULL_STEP = _PROLOGUE + r"""
@jax.jit
def step(s):
    return imex_step(s, g)

def advance(s, n):
    for _ in range(n):
        s = step(s)
    return s

a = advance(s_ref, 50)
b = advance(shard_state(s_ref, mesh, "kx"), 50)
errs = {f: rel(getattr(a, f), getattr(b, f))
        for f in ('q_hat', 'w_hat', 'th_hat', 'th_bar')}
assert max(errs.values()) < 1e-12, errs
print("FULL-STEP-MATCH", max(errs.values()))
"""


def _run_subprocess(body, gpu=False):
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not gpu:
        env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
        env["JAX_PLATFORMS"] = "cpu"
    return subprocess.run([sys.executable, "-c", body],
                          env=env, capture_output=True, text=True, timeout=900)


def _assert_ok(result, token):
    assert result.returncode == 0, (
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr[-2000:]}")
    assert token in result.stdout, result.stdout


class TestShardingCPU:

    def test_kx_wiring_and_local_shards(self):
        _assert_ok(_run_subprocess(_WIRING), "WIRING-OK")

    def test_z_axis_rejected_uneven(self):
        _assert_ok(_run_subprocess(_Z_REJECTED), "Z-REJECTED-AS-EXPECTED")

    def test_implicit_solve_sharded_matches(self):
        _assert_ok(_run_subprocess(_IMPLICIT), "IMPLICIT-MATCH")


class TestShardingGPU:
    """The real M3 correctness gate; needs >= 2 visible GPUs (run with
    CUDA_VISIBLE_DEVICES=6,7 per the machine's GPU policy)."""

    def test_full_step_two_gpu_match(self):
        import jax
        if jax.default_backend() != "gpu" or len(jax.devices()) < 2:
            pytest.skip("needs >=2 GPUs; CPU XLA fft_thunk cannot partition "
                        "an FFT over the sharded kx axis")
        _assert_ok(_run_subprocess(_FULL_STEP, gpu=True), "FULL-STEP-MATCH")
