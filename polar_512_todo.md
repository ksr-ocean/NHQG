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
- [~] **M1 gate (P0 validation)**: quasi-barotropic crystal forms; radius
      tracks L_γ=(U/γ)^{1/3} across ≥3 γ values. [GPU 6/7, lead]
      Driver ready: `scripts/run_polar.py --init barotropic-vorticity --Ra 0`
      (2026-07-22, codex, `f437827`); analysis via `nhqg/polar_diagnostics.py`.
      **Runs LAUNCHED 2026-07-22**: 512²×8, L=48Lc, Ra=0, ζ_rms=1 (seed 0,
      k_peak=1.3048 ⇒ U≈0.77, KE₀=0.309 ✓), ν=1e-4 hyper_order=4, dt=0.01,
      t_final=1000; γ=0.005 on GPU6, γ=0.02→γ=0.08 chained on GPU7;
      `output/polar_p0_bt_g{γ}_Nx512_Nz8_L48`. CPU smoke of the exact config
      passed first (trap guard OK at production geometry). L_γ ≈ 5.4/3.4/2.1
      vs trap r* = 52.
      **FINDING (2026-07-22): γ=0.08 + ars222 blew up at t=48** (KE grew
      exponentially from t≈26 while enstrophy decayed; Ra=0 so the continuum
      system cannot grow KE — numerical). Diagnosis: trap-edge topographic
      Rossby waves (ω ~ |∇η|_edge·kθ/k², |∇η|_edge ∝ γ·A_d·r*) sit on the
      imaginary axis where ARS222's explicit stage is weakly unstable
      (~(ωdt)⁴ amplification/step) → γ² sensitivity: γ=0.02 clean to t=1000,
      γ=0.08 e-folds in ~30 t.u. Archived in
      `output/polar_p0_bt_g0.08_*_ars222_blowup`. FIX: rerun γ=0.08 under
      **rk443** (real imaginary-axis stability interval; SYI22's RK4
      likewise immune) + `mean_exchange=legacy` (bitwise-identical q to
      balanced_sbp2_pc for Ra=0 barotropic — verified 0.0; sbp_pc dispatcher
      is ars222-only). Driver `--imex-scheme` added (`f234db7`). Retry
      queued detached behind the γ=0.005 continuation on GPU7.
      CAUTION for M4/M5: the convective campaign at γ~O(0.1)·(calibrated)
      may face the same edge-wave constraint under ars222 — either rk443
      for trapped runs or dt set by (ω_edge·dt) ≲ 0.1, not advective CFL.
      **rk443 stability VALIDATED** (2026-07-22): γ=0.08 sharp trap,
      monotone KE decay through t=159 where ars222 blew at t=48; archived
      `*_rk443_decay_t159`. rk443 costs ~5× ars222 per t.u. at 512²×8.
      **FINDING 2 (2026-07-22): decaying runs DON'T crystallize** — they
      zonate (ring jets + lone polar cyclone + trap-edge ring; overlays in
      `analysis/polar_p0/`). SYI22 barotropic crystals need sustained
      stochastic storm injection (NHGQ_polar.tex §Forcing anticipated
      this). Driver now has --inject-* (Gaussian cyclones, k=0 absorbed).
      Injection pilot (γ=0.02, A=6, r=2.4, 1/t.u., drag 0.02, t=100):
      equilibrium KE≈4.9 (U≈3.1), discrete persistent storms ✓.
      **FORCED campaign P0F running** (GPU7, detached): γ=0.005 → γ=0.02,
      t=600 each, ~1.8 h each; γ=0.08 leg deferred to a scheme/dt decision
      (rk443 ~10 h vs ars222@dt/4 ~7 h) after the first two report.
      Decaying-run data (rings/zonation) retained — publishable contrast
      case for the regime diagram. **Analysis tooling ready** (2026-07-22, codex #2,
      verified): `scripts/analyze_polar_p0.py` — crystal_metrics (R_crystal,
      NN spacing via periodic min-image), per-run overlays, log-log slope
      fits vs L_γ; gate = synthetic 19-vortex lattice with exact R/spacing
      (`tests/test_analyze_polar_p0.py`). Live mid-run smoke at t=95 sane
      (γ=0.02: U=0.705, 356 vortices, not yet organized).

## M2 — Mixed BCs: Dirichlet bottom / Neumann-w top [independent of M1]

- [x] **M2 core** (2026-07-22, `88cf8f9`, lead): `w_bc_top` config; Shen mixed
      stencil (b_n = −(n²+(n+1)²)/((n+1)²+(n+2)²), a_n = 1+b_n) with exact
      triangular left inverse; K-matrix buoyancy elimination (shell dedup
      survives); all solver physics sites (both RHS variants, implicit
      tendency, IMEX steps 0/2/4/5 with w↔θ basis maps, RK4/_apply_bcs) on
      per-field operators. Both-Dirichlet path shares the same arrays and
      expressions (bitwise). 13 tests incl. open-top vs rigid-lid control and
      mixed IMEX-vs-RK4.
- [x] **M2b plumbing** (2026-07-22, codex, verified §6): all w lifts in
      exchange paths / `io.py` / `diagnostics.py` → `w_stencil`/`w_pinv`/
      `proj_w`; `horizontal_mean_wtheta` takes split w/θ stencils (all call
      sites updated); both make_grid guards removed. Lead gates
      (`tests/test_mixed_bc_plumbing.py`, RED 3 / GREEN 1 pre-dispatch) all
      GREEN: neumann+evolve_mean exact BCs, SBP exchange residual still
      structural-zero under open top, neumann+vertical_cutoff BC-exact,
      both-Dirichlet evolve_mean **bitwise** vs stored ref. Lead follow-ups:
      w-budget source projections in `compute_w_theta_budgets` → `proj_w`
      (no-op for dirichlet); obsolete guard test flipped to permit-check;
      4 test call sites updated to the new signature. Codex correctly
      STOPPED on the test conflict instead of editing gates. Suite: 141.
- [x] EVP onset gate (2026-07-22, `f437827`, lead): `nhqg/linear_onset.py`
      assembles the exact linear operator of the discretized solver per k;
      validated vs closed-form both-Dirichlet dispersion (1e-8) and
      Ra_c=8.6956; **mixed-BC stepper growth within 1% of EVP at 3 k's** ✓.
- [ ] Route-B cross-check: same mixed-BC case in `fd_vertical_benchmark`
      (sbp42) at 128×128; compare onset + short nonlinear stats. [GPU, lead]

## M3 — 2-GPU sharding (after M0; re-bench after M2 lands)

- [x] Wiring (2026-07-22, lead): `nhqg/sharding.py` (mesh, per-field
      NamedShardings, `shard_state`); driver flags `--shard-axis`/
      `--shard-devices` (flag lives at driver level, not NHQGConfig —
      execution concern, checkpoints stay layout-agnostic). No explicit
      `with_sharding_constraint` yet — pure GSPMD propagation first;
      constraints only if the GPU profile shows resharding.
      **FINDINGS**: (i) kx (axis 1) is the ONLY shardable state axis on 2
      devices — Nz±1 and Nk are odd and JAX 0.10 rejects uneven splits
      (IndivisibleError); 'z' needs a padded layout if ever wanted.
      (ii) XLA's CPU backend cannot run a partitioned FFT over the sharded
      axis (fft_thunk layout RET_CHECK) → full-step equivalence is GPU-only.
- [x] CPU gates (2026-07-22, `tests/test_sharding.py`, 3 passed): kx wiring
      + local shard shapes; z-rejection documented; sharded IMEX implicit
      solve (per-shell gather/matmul core) matches unsharded to 1e-13.
- [x] **GPU correctness gate PASSED** (2026-07-22): full-step 1-vs-2 GPU
      trajectory match <1e-12 (50 steps, production open-top+evolve_mean
      config) on GPUs 6+7.
- [!] **Throughput gate FAILED under pure propagation** (2026-07-22):
      512²×64 production config, H200 ×2: 1 GPU 41.9 ms/step, 2 GPU
      kx-sharded 56.6 ms/step → **0.74×** (0.79× with
      imex_matmul_chunk=0 — chunked lax.map is only a minor factor).
      GSPMD's FFT-phase collectives/resharding eat the win. NEXT: profile
      one sharded step, then `with_sharding_constraint` two-phase layout
      (kx-sharded vertical/IMEX phase ↔ batch-sharded FFT phase).
      BONUS FINDING: clean 1-GPU step is 42 ms (~14 min per t.u.) — the M4
      pilot's 89 ms/step means compute_diagnostics at diag-interval 0.01
      was ~50% of wall; extension runs use diag-interval 0.05.
- [ ] Checkpoint path: verify `jax.device_get` on sharded state → io.py npz
      unchanged; restart cross-compatible between 1- and 2-GPU.

## M4 — 512²×64 shakedown

- [~] **Open-top full-physics pilot LAUNCHED** (2026-07-22, GPU6, lead):
      512²×64, L=48Lc, Ra=100, γ=0, `w_bc_top=neumann` + `evolve_mean` +
      balanced_sbp2_pc+sub4 + flux + 23_rule, ν=1 Laplacian, dt=5e-5,
      t_final=10 (M4 pilot gate; doubles as the M5 γ-calibration spin-up —
      measure barotropic U_rms at saturation). `--w-bc-top` added to driver
      (`f184c18`; CPU smoke of neumann+evolve_mean via driver passed).
      `output/polar_m4_pilot_opentop_evolvemean_Nx512_Nz64_L48_Ra100_dt5e5`.
      P0 rearranged to free GPU6: γ=0.005 killed at t≈435, CSV truncated to
      t=400, continuation from `checkpoint_00040000.npz` queued on GPU7
      behind g0.02→g0.08 via a pgrep watcher.
- [x] **M4 pilot gate PASSED** (2026-07-22): open-top 512²×64 pilot ran
      t=0→10 clean (linear growth t≲2.5 at ~theory rate → saturation
      t≈2.7–3 → turbulent; Nu_d≈12, max_w≈220, KE=378 and still
      condensating). NOTE: open-top Nu_d≈12 at Ra=100 vs ~18–20
      both-Dirichlet — first quantitative open-top heat-transport number,
      analyze when extension lands. EXTENSION superseded (2026-07-22, user
      direction): pilot now runs **to t=100** on GPU6, then a handover
      script auto-calibrates γ from the final-40-unit U_bt (target
      L_γ = 6 Lc; plateau U_bt ≈ 15–16 at t≈10–15 ⇒ γ ≈ 6e-4 expected) and
      launches **P2** — the first convective trap run — from the t=100
      checkpoint to t=250:
      `output/polar_p2_opentop_trap_gcalib_Nx512_Nz64_L48` (handover
      `p2_handover.py` in the session scratchpad; calibration line logged
      in the P2 log). Edge-wave check: γ~6e-4 ⇒ ω_edge ≈ 0.2 → ars222 safe.
      ~23 h (pilot) + ~43 h (P2). γ=0 movie:
      `analysis/m4_pilot_opentop_3x3.mp4` (regenerate after t=100).
- [ ] dt stability scan at L=48 Lc (start 5e-5; CFL vs dx=0.45 units).
- [x] Diagnostics cadence + non-finite abort: built into `scripts/run_polar.py`
      (2026-07-22; aborts with exit 2 + emergency checkpoint; structural-NaN
      SBP audit fields excluded by name so real blowups still trigger).
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
