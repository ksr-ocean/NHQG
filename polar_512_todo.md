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
