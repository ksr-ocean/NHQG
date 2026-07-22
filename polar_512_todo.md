# TODO — polar_512 campaign (live tracker)

Contract: `polar_512_plan.md` (frozen). This file is the live status board —
update it as tasks land. Statuses: `[ ]` open, `[~]` in progress, `[x]` done,
`[!]` blocked. "codex-able" = delegable per the §5 spec contract with CPU-only
validation; everything else is lead work. GPU gates: lead only, GPUs 6/7.

## M0 — Hygiene prerequisites

- [~] **P0.3 git**: `.gitignore` (output/ ~310 GB, derived_checkpoints/, logs,
      PDFs, aux), initial commit of code+docs. Do FIRST so refactors are tracked.
- [ ] **P0.1 ghost fix**: Hermitian-symmetrize ky=0 + ky-Nyquist rfft2 columns
      each step; Hermitian-project initial noise and checkpoint loads;
      regression test (anti-Hermitian seed must die, Hermitian state unchanged
      to roundoff). See `hermitian_ghost.md`.
- [ ] **P0.2 init/restart mask**: under `23_rule`, mask state to the retained
      band at init, checkpoint load, and finalize (currently output-only).
      Test: restart of a masked state is bit-compatible.
- [ ] **P0.4a q_solve lever**: `q_boundary='none'` → use scalar `inv_alpha_q`,
      drop dense per-shell identity q_solve matrices. (codex-able; CPU tests)
- [ ] **P0.4b gather lever**: stop materializing `mat_shells[ksq_idx]`
      (4.4 GB at 512²×64) — chunked `lax.map`/scan over kx or segment-sum.
      CPU correctness test codex-able; VRAM gate (peak < 10 GB at 256²×64) lead.
- [ ] **M0 gate**: full test suite green; 64×256 short-run trajectory match
      vs pre-M0 code (ghost projection ON changes only ghost content).

## M1 — Trap (γ-effect) [independent of M2]

- [ ] Config params: `gamma`, `trap_r_star`, `trap_sharpness`, sponge trio
      (default off); assert gamma/beta mutually exclusive. (codex-able)
- [ ] `grid.py`: η̂ precompute (unpadded grid, 2/3-masked), smoothness assert
      (out-of-band energy < 1e-12).
- [ ] `solver.py`: augmented advection `q̂+η̂` in BOTH `jacobian` and `flux`
      paths; β spectral term untouched for the beta path.
- [ ] Tests: η↔β equivalence to roundoff; conservation/antisymmetry with η;
      γ=0 invisibility. (codex-able against lead-written spec)
- [ ] **M1 gate (P0 validation)**: quasi-barotropic crystal forms; radius
      tracks L_γ=(U/γ)^{1/3} across ≥3 γ values. [GPU, lead]

## M2 — Mixed BCs: Dirichlet bottom / Neumann-w top [independent of M1]

- [ ] `grid.py`: per-field stencils — w: Shen 3-term (φ(0)=0, φ′(1)=0) with
      exact triangular left inverse (NO Moore–Penrose — see 2026-03 lesson);
      θ keeps Dirichlet stencil. Both-Dirichlet special case must reproduce
      current operators bitwise.
- [ ] `_build_imex_inv`: K-matrix buoyancy elimination
      (A′ = α_w I − (γdt)²c(k)B − (γdt)²(Ra/σ)/α_θ(k)·K, K precomputed).
- [ ] `solver.py`/`io.py`/`diagnostics.py`: replace shared dirichlet_stencil/
      pinv with per-field operators everywhere (exchange path lifts w through
      w_stencil). (bulk codex-able after lead does grid.py + IMEX)
- [ ] 1-D generalized EVP onset solver in Z per k (lead; ~30 lines, CPU);
      tabulate mixed-BC Ra_c(k), k_c.
- [ ] Tests: BC exactness 1e-12 (w′(top), w(bottom)); bitwise both-Dirichlet
      regression; IMEX vs RK4; **onset growth rates within 1% of EVP at 3 k's**.
- [ ] Route-B cross-check: same mixed-BC case in `fd_vertical_benchmark`
      (sbp42) at 128×128; compare onset + short nonlinear stats. [GPU, lead]

## M3 — 2-GPU sharding (after M0; re-bench after M2 lands)

- [ ] Mesh + NamedSharding wiring behind a config flag (`shard_axis: none|z|kx`);
      `with_sharding_constraint` at spectral/physical phase boundaries.
- [ ] Measure axis-0 vs axis-1 (vs two-phase if needed) at 256²×64. [GPU 6/7]
- [ ] **Gates**: 1-vs-2 GPU trajectory match (1e-12 state norms, 200 steps,
      128²×64); ≥1.6× steps/s at 512²×64; balanced util on 6/7 only.
- [ ] Checkpoint path: verify `jax.device_get` on sharded state → io.py npz
      unchanged; restart cross-compatible between 1- and 2-GPU.

## M4 — 512²×64 shakedown

- [ ] dt stability scan at L=48 Lc (start 5e-5; CFL vs dx=0.45 units).
- [ ] Diagnostics cadence + non-finite abort in driver (May-19 drivers don't
      stop on non-finite — fix in the polar driver).
- [ ] Polar diagnostics: azimuthal E(m), vortex tracker, trap-restricted
      spectra (`NHGQ_polar.tex` §7).
- [ ] Vertical-tail monitor (q Chebyshev tail fraction) in run diagnostics.
- [ ] **Gate**: stable 10-unit pilot, ghost-free, VRAM within budget.

## M5 — Physics campaign (per NHGQ_polar.tex §8)

- [ ] γ-calibration pilot: γ=0 spin-up at L=48 Lc, measure barotropic U_rms,
      set γ for L_γ ≈ 5–7 Lc.
- [ ] P1: γ=0 convective control (trap must not disturb condensate).
- [ ] P2: γ>0, Ld=∞ — convectively forced crystal (first new result).
- [ ] P3: finite Ld.
- [ ] P4: (γ, Ld) sweep — one run per GPU (no sharding), 6/7.

## Done

- [x] 2026-07-22 Plan written + decisions D1–D3 recorded (`polar_512_plan.md`).
- [x] 2026-07-22 Radial-budget notebook delivered
      (`analysis/spectral_budget/radial_spectral_budget.ipynb`).
