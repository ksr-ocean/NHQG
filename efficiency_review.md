# NHQG Solver Efficiency Review

Single-GPU audit of the per-step hot path for the `balanced_sbp2_pc` /
`subcycle4` / `flux` / `evolve_mean` branch on an H200 (141 TFLOPS FP64,
4.8 TB/s HBM3e). Baseline: **Nx=256, Nz=256, dt=5e-5 → 150 ms/step,
114 GB peak VRAM, 120 s JAX compile, 6.22 GB captured constants**.

The goal was to identify JAX- and GPU-specific wins without changing the
physics branch.

> **Status 2026-07-04.** Adopted in production: **#1** (SBP corrector
> hoisting) and **#3** (donate_argnums) — the "stack13" run-name tag; **#6**
> (2/3-rule) was adopted for all post-2026-04-18 runs (the "23rule" tag).
> **#2** was probed as sub2 ("stack123_sub2" runs) but the long chains stayed
> at sub4. Caveats discovered later (2026-07-03 review): the 2/3-rule
> implementation masks product outputs only — the state is never truncated
> (see `CLAUDE.md` Current Status item 4 and `hermitian_ghost.md`); and the
> "skipped idea" row about captured constants is now qualified — the unused
> scalar `inv_alpha_q` makes the dense per-shell `q_solve` storage a free
> deletion (see the VRAM memory note).

---

## Structural facts (verified)

- The production run scripts (`scripts/continue_from_checkpoint.py:261`,
  `scripts/run_miquel_compare.py`) use a **Python for-loop** around
  `step_fn = jax.jit(lambda s: imex_step(s, grid))`. They do **not** use
  the `lax.scan`-based `run()` wrapper in `nhqg/solver.py:1165` (that path
  is dead code for the current production flow).
- The ARS(2,2,2) step does **two** explicit-RHS + implicit-solve + SBP-
  corrector blocks per step. With `sbp_corrector_substeps=4` that is
  **8 SBP substeps/step**.
- **FFT budget per step on padded (257, 384, 384) batched grid:**
  - `triple_conservative_flux_divergence` (flux-form nonlinear advection):
    11 batched FFTs per call × 2 stages = **22**
  - SBP corrector: ~4 padded FFTs per substep × 8 substeps = **32**
  - **Total ≈ 54 padded batched FFTs/step.**
  - **The SBP corrector is doing more FFT work than the nonlinear
    advection** — an order-1 observation that reframes optimization
    priorities.
- The 6.22 GB captured-constants warning is the IMEX shell-inverse
  matrices `grid.imex_inv` + `grid.q_solve` baked into the jit cache.
  One-time compile cost; not a per-step penalty.

---

## Prioritized wins

### #1 — Hoist invariants out of the SBP corrector substep loop

**Files / symbols**:
- `nhqg/solver.py:725-734` (`_apply_balanced_sbp2_corrector`)
- `nhqg/solver.py:678-722` (`balanced_sbp2_thermal_substep`)

Inside `_apply_balanced_sbp2_corrector`, the Python-for loop over
`n_substeps=4` re-runs the full `balanced_sbp2_thermal_substep`. Three
of its ingredients are **invariant across substeps** within a given
ARS stage:

- `w_sbp` — derived from `state.w_hat`, which the corrector never
  modifies (it only updates `state.th_hat` and `state.th_bar`).
- `w2_mean = <w²>_xy` — a pure function of `w_sbp`.
- `A = I − 0.5·μκ·dt·L − 0.25·μ·dt²·(D1 M D1)` — a pure function of
  `w2_mean`, dt, and static grid matrices.

Across 4 substeps × 2 ARS stages, this is **3 redundant w-related
padded FFTs per substep × 3 wasted substeps × 2 stages = 18 redundant
padded FFTs per step**, plus 6 redundant rebuilds and BC-fixups of the
(Nz+1, Nz+1) matrix `A` and its `jnp.linalg.solve`.

**Refactor sketch:** split `balanced_sbp2_thermal_substep` into two
pieces:

1. A "preamble" called **once per stage** that returns `w_sbp`,
   a factorization of `A` (via `jax.scipy.linalg.lu_factor`), and
   any other substep-invariant quantities.
2. A "substep" kernel that is called `n_substeps` times and only
   recomputes `flux_n = <w_sbp · th_sbp_n>_xy` and then does
   `lu_solve(A_lu, rhs)` + the θ update.

Given that the corrector is currently ~32 FFTs of a ~54-FFT step, and
we eliminate ~18 of them, the expected FFT reduction is roughly one
third.

- **Estimated speedup: ~20–30%** (bandwidth-bound pseudospectral cost
  scales with FFT count; H200 is bandwidth-rich but the padded-grid
  scatter-store + irfft chain is the dominant wall-time contributor)
- **Risk: low** — the math is bit-identical to the current code;
  regression-testable via the existing solver test suite and a short
  64×256 continuation probe.
- **Effort: ~1 hour** of careful refactoring.

---

### #2 — Reduce `sbp_corrector_substeps` from 4 → 2

**File**: `NHQGConfig` field + whatever scripts pass
`--sbp-corrector-substeps`.

Per `CLAUDE.md`, subcycle4 is the "current best" recipe that got the
64×256 run clean through t=80. The substep count is a numerical
stability knob, not an algorithmically required factor. Halving it
halves the corrector cost almost exactly.

- **Estimated speedup: ~15%** *independent of #1*. Combined with #1,
  the savings compound: the non-hoistable per-substep work shrinks by
  half.
- **Risk: medium** — needs validation that late-time stability is
  preserved at the new substep count. A 64×256 probe from a developed
  state (e.g. the t=80 checkpoint) for a few time units should settle
  it quickly.
- **Effort**: trivial (config change) + validation runs.

---

### #3 — `donate_argnums=(0,)` on the step jit

**Files**:
- `scripts/continue_from_checkpoint.py:261`
- `scripts/run_miquel_compare.py` (corresponding `step_fn = jax.jit(...)`
  line)

```python
step_fn = jax.jit(lambda s: imex_step(s, grid), donate_argnums=(0,))
```

The `State` NamedTuple at Nx=256, Nz=256 is ~410 MB across its four
spectral arrays. Without donation, XLA allocates fresh output buffers
each step; with donation it can alias input and output, reducing
allocator churn and unblocking some kernel-fusion opportunities that
XLA currently can't prove safe.

On H200's 4.8 TB/s bandwidth the pure bandwidth saving is small, but
in practice this lands as a 1–3% gain and is a one-line change.

- **Estimated speedup: 1–3%**
- **Risk: none** — State is never re-read after the call.
- **Effort: 1 line per script.**

---

### #4 — Switch the production run loop to `lax.scan`

**Files**: `scripts/continue_from_checkpoint.py:343-353`,
`scripts/run_miquel_compare.py` equivalent loop. The template already
exists in `nhqg/solver.py:1165 run()`.

`lax.scan(body, state, None, length=save_interval)` with the existing
diagnostics callback invoked per save-bucket would eliminate the
~100 µs/step Python dispatch overhead.

- **Estimated speedup: <1% at Nx=256** (wholly negligible given
  150 ms/step). Worth **5–15% at Nx ≤ 64** where dispatch latency
  becomes a noticeable fraction of step time.
- **Risk: low** — the outer diag cadence is preserved; only the inner
  loop becomes a compiled scan.
- **Effort: ~30 min.** Not a standalone win at this resolution; bundle
  opportunistically when other code in `continue_from_checkpoint.py` is
  being touched.

---

### #5 — Cosmetic cleanup of `_zero_pad` complex allocation

**File**: `nhqg/spectral.py:21-29`

Current:
```python
out = jnp.zeros((Npad, Nk_pad)) * (0.0 + 0.0j)  # ensure complex
out = out.astype(f_hat.dtype)
```

Preferred:
```python
out = jnp.zeros((Npad, Nk_pad), dtype=f_hat.dtype)
```

XLA almost certainly fuses the three-op chain into the allocator call
already, so there is probably **no measurable speedup**. It is listed
only because it is misleading code that suggests allocation cost where
there is none.

- **Estimated speedup: ~0% (code hygiene).**
- **Effort: trivial.**

---

## #6 — 2/3-rule dealiasing instead of 3/2-rule (horizontal)

**Short answer**: potentially large win, but *only if* the current runs
are already decaying in the top 1/3 of horizontal modes. Must verify
empirically before committing.

### The math

3/2-rule and 2/3-rule are two implementations of the same dealiased
quadratic-nonlinearity math. They produce **bit-identical results on
the retained 2/3 of modes**. The difference is *where the FFT cost
lives*:

- **3/2-rule at Nx=256** (current): state has 256 modes; FFT on padded
  384² grid; truncate back to 256. Every stored mode is usable.
- **2/3-rule at Nx=256** (alternative): state has 256 stored modes; FFT
  on 256² grid; top 1/3 of modes are zeroed after each nonlinear eval.
  Only ~170 of the 256 stored modes are physically meaningful — the
  rest are a dissipation sink.

The FFT-cost ratio between the two options at fixed Nx is
`(384 / 256)² × log(384)/log(256) ≈ 2.4×`. Since FFTs are roughly
half the step wall time in the current code (see #1 analysis), the
**nominal** upper-bound gain is ~25-35% overall.

### The catch

Fair apples-to-apples comparisons of the two schemes *at the same
number of usable modes*:

| scheme | Nx (stored) | usable modes | FFT grid size |
|---|---|---|---|
| 3/2-rule | 128 | 128 | 192² |
| 2/3-rule | 192 | 128 | 192² |
| 3/2-rule | 256 | 256 | 384² |
| 2/3-rule | 384 | 256 | 384² |

Same FFT cost in each matched pair. So "switch to 2/3-rule" is only a
win if you are willing to **drop usable modes at fixed stored Nx**.
Concretely: keep `Nx=256`, accept the effective horizontal resolution
of an `Nx=170` 3/2-rule run. That is a genuine **physics change**, not
a free algorithmic swap.

### When it's free

If the top 1/3 of horizontal modes already carries negligible energy
(say, <1% of the per-field horizontal spectrum throughout a run), the
physics change is unobservable and the 2.4× FFT reduction is banked
for free.

**Verification**: you already archive `w_horiz_spec`, `q_horiz_spec`,
`th_horiz_spec` in `spectra/spectrum_history.npz` (see CLAUDE.md
Restart Notes 2026-04-03). Plot the fraction of total spectral energy
above `|k| = Nx/3` as a function of time, for both the `128x256` and
`64x256` long runs. If that fraction stays below ~1% through the
developed regime, 2/3-rule is a safe swap.

CLAUDE.md already hints at this: "mid and high horizontal shells
remain weak" from the 2026-04-03 spectral diagnostics note. That is
consistent with 2/3-rule being nearly lossless, but needs explicit
verification on the actual spectrum-history files before committing.

### Vertical: no change warranted

Chebyshev vertical is not Fourier; there is no analogous "2/3 rule".
The standard practice in this literature (Miquel et al. 2026, Coral)
is **no explicit vertical dealiasing**, and your own `cheb_2x`
over-resolved experiment was a null result (CLAUDE.md Restart Notes
2026-03-22: "Negative result: vertical dealiasing did not fix the
blowup"). Leave the vertical path alone.

### Implementation sketch

The current dealiasing code is cleanly isolated:

- `nhqg/spectral.py:21-50` (`_zero_pad`, `_truncate`)
- `nhqg/spectral.py:67-115` (`jacobian_dealiased`,
  `conservative_flux_divergence_dealiased`, fused triple variants)
- `nhqg/solver.py:154-196` (`horizontal_mean_wtheta`,
  `horizontal_mean_from_nodal_spectral`)

Add a new `horizontal_dealiasing` config flag (`"32_rule"` | `"23_rule"`)
defaulting to `"32_rule"`. In the `23_rule` path:

- Skip `_zero_pad`: FFT directly on the Nx grid.
- Do the physical multiply on the Nx grid.
- Replace `_truncate` with a 2/3-mode mask that zeros coefficients with
  `|kx| > Nx/3` or `ky > Nx/3`.
- Drop the `(Npad/Nx)²` amplitude compensation factor.

This keeps both paths available for A/B comparison and leaves the 3/2
regression tests untouched.

### Summary for #6

- **Estimated speedup**: ~25-35% on wall time if verification clears
  (i.e. top-1/3 modes are empty). **0% otherwise** — and in fact a
  physics regression, since you'd be silently dropping resolution.
- **Risk**: *medium-high without verification*. **Low** if you do the
  spectrum-fraction plot first.
- **Effort**: ~2-3 hours including keeping both paths behind a flag
  and adding a regression test that matches 3/2 and 2/3 results on a
  band-limited test case.
- **Prerequisite**: 1 hour of offline analysis on the existing
  `spectrum_history.npz` archives.

---

## Ideas deliberately skipped

| Idea | Reason |
|---|---|
| Mixed precision (FP32 Jacobians, FP64 implicit solve) | 2–3% upside, medium risk of late-time drift over 10k+ step integrations. Requires extensive validation before trusting the existing diagnostics chain. Revisit only if a wall is hit. |
| `XLA_FLAGS="--xla_gpu_triton_gemm_any=true"` and friends | Pseudospectral code is cuFFT-dominated, not GEMM-dominated. Marginal at best. |
| Move `grid.imex_inv` / `grid.q_solve` out of captured-constants | One-time compile cost; doesn't affect per-step throughput. Only relevant if compile time itself becomes painful for rapid experimentation. |
| Multi-GPU sharding (`pjit` / `shard_map` across 8 H200s) | Genuine ~4–6× upside at Nx=256, **but** multi-day refactor. The shell-deduplicated IMEX matrices need sharding logic, and the 3/2-rule Jacobian needs cross-device communication. Only justified if 256×256×256 becomes a serious production target with t≥80 runs on the roadmap. |
| FFT plan caching hand-tuning | JAX already caches cuFFT plans per-shape. Nothing to do. |

---

## Expected combined gain

Implementing #1 + #2 + #3:
- **30–40% faster** end to end.
- 150 ms/step → **~95–105 ms/step**.
- t=10 → t=20 walk: ~8 h → **~5 h**.
- Aspirational t=80 at 256×256: ~1 week → **~4–5 days**.

If #6 (2/3-rule) also clears the spectrum-fraction verification, stack
another 25–35% on top:
- 150 ms/step → **~65–75 ms/step**.
- Aspirational t=80 at 256×256: **~3 days.**

Beyond that, multi-GPU sharding is the only realistic lever left on
single-node throughput.

---

## Suggested implementation order

1. **#3** (donation) — low risk, low effort, immediate payback.
2. **#1** (SBP corrector hoisting) — the biggest structural win, still
   low risk because it is a pure algebraic refactor.
3. **#2** (subcycles 4→2) — run as a physics-validation experiment
   once #1 is in (so you are validating the numerical choice, not a
   concurrent code change).
4. **#4, #5** — opportunistic cleanup; no urgency.
