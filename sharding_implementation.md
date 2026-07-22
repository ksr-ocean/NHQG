# M3 sharding — implementation notes (2026-07-22)

How the 2-GPU sharding of the NHQG solver works, from an implementation
perspective. Written as a teaching record; the code is `nhqg/sharding.py`,
the gates are `tests/test_sharding.py`, status lives in `polar_512_todo.md`
(M3 section), and the original feasibility analysis is `polar_512_plan.md`
Part II.

## The mental model: one program, XLA splits the data

JAX sharding is SPMD ("single program, multiple data"): you never write
per-device code and there is no explicit MPI. You write the exact same
`imex_step` the solver already has, but you tell JAX *how the input arrays
are laid out across devices*. The XLA compiler (its GSPMD pass) then
propagates that layout through every operation in the traced program and,
wherever an operation needs data a device doesn't hold, it inserts the
communication collective itself (all-gather, all-to-all, reduce-scatter —
over NVLink via NCCL on the H200s). The numerics code stays untouched —
that's why M3's wiring is ~60 lines.

The classical analogue is a pencil-decomposition spectral code
(P3DFFT-style), with the compiler writing the transposes: an MPI pencil
code shards one axis, FFTs the local axes, does a big `MPI_Alltoall`
transpose, FFTs the remaining axis. GSPMD's inserted collectives around
the FFT are exactly that transpose step, derived automatically.

## The three objects

```python
mesh = Mesh(np.array(jax.devices()[:2]), ("dev",))  # hardware: 1-D grid of 2 GPUs
spec = PartitionSpec(None, "dev", None)             # array axis -> mesh axis map
sharding = NamedSharding(mesh, spec)                # the pairing
q_hat = jax.device_put(state.q_hat, sharding)       # physically scatter
```

`PartitionSpec(None, "dev", None)` read against `q_hat`'s shape
`(Nz+1, Nx, Nk) = (65, 512, 257)` says: axis 0 replicated, axis 1 (kx)
split across the mesh, axis 2 replicated. After `device_put`, GPU0 holds a
`(65, 256, 257)` block (kx rows 0–255) and GPU1 holds rows 256–511.
`th_bar` is 65 numbers — `P()` keeps it fully replicated. That is the
entirety of `nhqg/sharding.py`: build mesh, build specs, `device_put` the
four state fields. The driver flags are `--shard-axis {none,kx}` and
`--shard-devices N` in `scripts/run_polar.py`.

## How the layout flows through the step

The input sharding is the *seed*; GSPMD classifies every op by how it
touches the sharded axis.

**Batched over kx → free.** Most of the step never mixes kx rows:

- All vertical operations — `einsum('ij,j...->i...', G_Z, field)`, the
  stencil lifts, `V`/`V_inv` transforms — contract axis 0 while axis 1
  comes along for the ride. Each GPU applies the (replicated, tiny)
  operator matrix to its own half of the kx rows. Zero communication.
- The per-shell IMEX solve: `mat_shells[ksq_idx]` gathers a
  `(Nz-1, Nz-1)` inverse per `(kx, ky)` point and matmuls it down the
  vertical. Indexed pointwise in kx → each device gathers and multiplies
  only for its rows. This is the flops-heavy core and it parallelizes
  perfectly — certified numerically to 1e-13 by the CPU gate.
- Pointwise algebra (alpha factors, dissipation multipliers, psi
  inversion): trivially local.

**Contracts kx → communication.** `irfft2`/`rfft2` operate over axes
(1, 2) — the FFT over axis 1 needs *every* kx row to produce *any* x
column. GSPMD must materialize that: either all-gather the halves so each
GPU FFTs a full field, or the pencil-style all-to-all (transpose so each
GPU owns complete kx lines for a subset of a batched axis, FFT locally,
transpose back). Which strategy it picks is its cost-model call — this is
what the M3 benchmark measures, and if it chooses badly,
`with_sharding_constraint` exists to override it. Every nonlinear product
costs one round trip through this: spectral → physical, multiply,
physical → spectral. At `(65, 512, 257)` float64 a field is ~270 MB
complex, so plan-level estimates put this at a few GB/step of NVLink
traffic — cheap relative to H200 NVLink (~900 GB/s), which is why an
overall win is plausible; it is also the entire risk budget of the
hoped-for ~1.9x.

**After the seed, everything is inference.** The output of `rfft2` comes
back kx-distributed (or GSPMD re-shards it to match downstream use), the
`lax.scan` carry keeps the layout step to step, so the state stays
resident-and-split for the whole run. Nothing "returns" to one GPU until
asked: `jax.device_get(state)` (in `save_checkpoint`) triggers a gather to
host — which is why the checkpoint format is completely unaware of
sharding and restarts are automatically cross-compatible between 1- and
2-GPU runs.

## The tuning knob deliberately not used yet

`with_sharding_constraint(x, sharding)` pins an intermediate to a layout,
overruling propagation. The classic use here would be a **two-phase
layout**: kx-sharded during vertical/IMEX work, explicitly re-shard to a
batch-axis (z) sharding for the FFT/product phase so FFTs are fully
local, then re-shard back — communication becomes two clean all-to-alls
per product instead of whatever GSPMD improvises. The wiring is
constraint-free on purpose: measure pure propagation first, add
constraints only where the profile shows pathological resharding.
(The z axis cannot be a *state* sharding — see findings — but
intermediates in the physical phase have even batch dims, so a two-phase
constraint remains available.)

## Findings (2026-07-22)

1. **kx is the only shardable state axis on 2 devices.** The state layout
   is `(Nz±1, Nx, Nk)`: the vertical dims are odd (Nz even → 65/63 rows)
   and `Nk = Nx/2+1` is odd. JAX 0.10 `NamedSharding` refuses uneven
   splits (`IndivisibleError`). So the plan's "measure axis-0 vs axis-1"
   question collapsed structurally: z-sharding would need a padded layout;
   the benchmark is kx vs single-GPU.
2. **The full-step smoke cannot run on CPU fake devices.** XLA's CPU FFT
   kernel demands a dense major-ordered local buffer; the partitioned
   program hands it a transposed view → internal `RET_CHECK`
   (`fft_thunk.cc: LayoutUtil::IsMonotonicWithDim0Major`). The GPU backend
   does not share the limitation (collectives produce dense buffers for
   cuFFT). Hence the honest gate split: wiring + sharded implicit solve
   certified on CPU (`--xla_force_host_platform_device_count=2`,
   subprocess-per-test because the flag must precede backend init);
   full-step 1-vs-2-GPU equivalence is a GPU-only gate
   (`tests/test_sharding.py::TestShardingGPU`, auto-skips off-GPU, run
   with `CUDA_VISIBLE_DEVICES=6,7`).

## What can still go wrong (why the GPU gate matters)

The failure mode of this approach is silent inefficiency, not wrong
answers — GSPMD is conservative about correctness but free to insert
absurd amounts of resharding if propagation gets confused (the 22k-entry
`ksq_idx` gather is the kind of op that can trip it). Gate order:

1. 1-vs-2-GPU trajectory match (<1e-12, catches genuine numerical
   divergence, e.g. reduction-order changes in collectives);
2. steps/s at 512²×64 with balanced utilization on GPUs 6/7
   (target ≥1.6x);
3. if correctness passes but throughput is ~1.2x, profile to find the
   reshard and pin it with `with_sharding_constraint` at that boundary.

Until gate 1 passes on real GPUs, sharding is wired-but-unproven.
