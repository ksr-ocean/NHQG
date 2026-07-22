# TODO — polar_512 campaign (live tracker)

Contract: `polar_512_plan.md` (frozen). This file is the live status board —
update it as tasks land. Statuses: `[ ]` open, `[~]` in progress, `[x]` done,
`[!]` blocked. "codex-able" = delegable per the §5 spec contract with CPU-only
validation; everything else is lead work. GPU gates: lead only, GPUs 6/7.

## M0 — Hygiene prerequisites

- [x] **P0.3 git** (2026-07-22): `.gitignore`, baseline commit `59d8aa3` on
      `main` (259 files); M0 work on branch `m0-hygiene`.
- [x] **P0.1 ghost fix** (2026-07-22, `488fd78`): `sanitize_state()` —
      Hermitian projection of ky=0 + ky-Nyquist columns at every step, at
      `run()` entry, at init (noise was non-Hermitian = the ghost seed), and
      numpy-side in `load_checkpoint`. 4 regression tests. NOTE: ghost content
      in the kx-Nyquist row is NOT invisible under the 3/2 pad (asymmetric
      ±Nyquist treatment) — documented in the test.
- [x] **P0.2 state band-limit** (2026-07-22, `488fd78`): under `23_rule` the
      masked band is zeroed **every step** (stronger than init/restart-only),
      killing the frozen-band pathology class entirely.
- [x] **P0.4a q_solve lever** (2026-07-22, `471a404`): `q_boundary='none'` →
      scalar `inv_alpha_q`; dense per-shell identities no longer built/stored/
      gathered (`grid.q_solve is None`).
- [~] **P0.4b gather lever** (2026-07-22, `471a404`): `imex_matmul_chunk`
      config — chunked `lax.map` over kx rows caps the gather transient;
      CPU-verified identical (both q_boundary modes, non-dividing chunk).
      **Pending: GPU VRAM gate** (peak < 10 GB at 256²×64, GPUs 6/7) — run
      with the M3 benchmarking session; also pick the production chunk size.
- [x] **M0 gate** (2026-07-22): 97 tests green + sanitized-start equivalence:
      pre-M0 vs post-M0 trajectories **bitwise identical** (60 steps,
      production-flavor 32²×32 balanced_sbp2_pc+sub4+flux+23_rule).
      FINDING: from an *unsanitized* start, pre-M0 in-band q differs at
      2×10⁻⁷ relative after only 60 steps — out-of-band state content
      genuinely contaminated in-band physics through the products; historical
      23_rule runs carried this at their noise level.

## M1 — Trap (γ-effect) [independent of M2]

- [x] Config params (2026-07-22): `gamma`, `trap_r_star` (None → 0.45·L/2),
      `trap_sharpness` (A_d=20); gamma/beta exclusivity enforced in make_grid.
      Sponge NOT implemented (deferred — SYI22 didn't need one; add only if
      P0/P2 show boundary wave reflection).
- [x] `grid.py` (2026-07-22): η̂ precompute + **resolution guard** (raises if
      >1e-10 of η's spectral energy is outside the 2/3 band; message reports
      tanh width vs dx). Trap-geometry feasibility learned the hard way: need
      transition width ≳ 4 dx AND edge margin (L/2−r*) ≳ 7 widths — at
      production (512², L=48Lc, A_d=20) both hold with orders of margin.
- [x] `solver.py` (2026-07-22): advected PV is `q_nodal + η̂` in both RHS
      variants (covers jacobian AND flux paths — one augmentation point);
      β spectral term untouched.
- [x] Tests (9, `tests/test_polar.py`): η matches analytic formula; added
      tendency == −J(ψ,η) vs the independently validated single-level
      Jacobian to 1e-12 (stronger than the η↔β idea, which is unrepresentable
      periodically); w/θ/Θ̄ tendencies bitwise untouched; flux ≡ jacobian on
      the η term to 1e-12 on the Nyquist-free subspace (Nyquist self-aliasing
      lands differently in the two forms — 32_rule fine print, documented);
      guard + exclusivity raise; production-flavor 23_rule trap run finite &
      band-limited. Suite: 106 passed.
- [ ] **M1 gate (P0 validation)**: quasi-barotropic crystal forms; radius
      tracks L_γ=(U/γ)^{1/3} across ≥3 γ values. [GPU 6/7, lead] — needs a
      polar driver script (γ-calibration + random-vorticity init variant).

## M2 — Mixed BCs: Dirichlet bottom / Neumann-w top [independent of M1]

- [x] **M2 core** (2026-07-22, `88cf8f9`, lead): `w_bc_top` config; Shen mixed
      stencil (b_n = −(n²+(n+1)²)/((n+1)²+(n+2)²), a_n = 1+b_n) with exact
      triangular left inverse; K-matrix buoyancy elimination (shell dedup
      survives); all solver physics sites (both RHS variants, implicit
      tendency, IMEX steps 0/2/4/5 with w↔θ basis maps, RK4/_apply_bcs) on
      per-field operators. Both-Dirichlet path shares the same arrays and
      expressions (bitwise). 13 tests incl. open-top vs rigid-lid control and
      mixed IMEX-vs-RK4.
- [ ] **M2b plumbing** (codex-able, lead-written gates): exchange paths /
      `io.py` / `diagnostics.py` w lifts → `w_stencil`; then LIFT the
      restriction (currently `w_bc_top='neumann'` requires
      `fixed_conduction` and no `vertical_cutoff_n` — enforced in make_grid).
- [ ] 1-D generalized EVP onset solver in Z per k (lead; CPU); tabulate
      mixed-BC Ra_c(k), k_c; gate solver growth rates within 1% at 3 k's.
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
- [x] Polar diagnostics module (2026-07-22, `0006e75`, Sonnet subagent vs
      lead-written gates): `nhqg/polar_diagnostics.py` — azimuthal E(m),
      vortex tracker (periodic NMS + subpixel), radial profiles, trap mask.
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
