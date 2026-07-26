# TODO — polar_512 campaign (live tracker)

Contract: `polar_512_plan.md` (frozen). This file is the live status board —
update it as tasks land. Statuses: `[ ]` open, `[~]` in progress, `[x]` done,
`[!]` blocked. "codex-able" = delegable per the §5 spec contract with CPU-only
validation; everything else is lead work. GPU gates: lead only, GPUs 6/7.

## M-1 — Migration off the original host (2026-07-26)

GPU access ended (shared arrangement with R. Barkan's group). Files were not
at risk; the project was staged for an ACCESS-CI restart.

- [x] All 68 run directories staged into one documented archive,
      `../NHQG_runs_archive_2026-07` (77 entries, 9,776 files, 425 GB), built
      by `scripts/build_data_archive.py` with hardlinks — zero extra disk,
      0.5 s. Carries `README.md` (a section per run, keyed to the repo docs),
      `MANIFEST.tsv` (archive path ↔ original `output/` name, needed because
      every older write-up cites the old names) and `RESTART.md` (the flag
      string per run, labelled RECORDED vs RECONSTRUCTED).
- [x] Repo made self-sufficient for a cold restart: `README.md`, `DATA.md`,
      `HANDOFF.md`, `requirements.txt` with the versions the suite actually
      passes against (Python 3.13.9 / JAX 0.10.0 / numpy 2.3.5 — CLAUDE.md's
      3.12.2 / 0.9.1 / 2.0.2 were stale).
- [x] `run_polar.py` now writes `run_config.json` (argv + resolved config +
      host + JAX version) per output dir. Closes the "checkpoints carry no
      config, defaults match no production run, wrong restart is silent" hole.
- [x] `scripts/submit_access.slurm` — self-chaining SLURM job (48 h cap →
      `--dependency=afterok` legs, restart from newest checkpoint, stops on
      the non-finite exit code). `scripts/transfer_archive.sh` — resumable
      rsync + verify mode.
- [x] Repo hygiene: untracked `all.py`, `nhqg_contents.txt`, `restart*`,
      `dinosaur_spike/`; gitignored frame stacks and movies. 452 objects,
      22 MB — GitHub-safe. `main` fast-forwarded to `m0-hygiene`.
- [ ] **User action: create the private GitHub repo, then push.** Remote is
      preconfigured as `git@github.com:ksr-ocean/NHQG.git`; the key
      `~/.ssh/id_ed25519_hpc` already authenticates as `ksr-ocean`.
- [ ] **User action: rsync the archive** (needs interactive PSC/SDSC auth).
      NOTE: only SDSC Expanse (V100-32GB) was configured in `~/.ssh/config`;
      Bridges-2 (H100-80GB) has never been connected from here. Confirm which
      allocation is live — see HANDOFF.md §2 for the throughput comparison.

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
      **FORCED campaign P0F**: completed t=600 both γ — result: lone polar
      cyclone + trap-edge ring, NOT crystals (γ=0.005 briefly held 2 polar
      cyclones). Iteration pilots: 5× rate → CFL blowup at t=67 (KE eq.
      ~10× ⇒ dt must follow U) AND soup (ambient U≈8 shreds ζ=6 storms);
      bracketing pilots (2.5×rate/drag.02, base/drag.005, dt=5e-3) clean to
      t=300 but same lone-cyclone morphology.
      **ROOT CAUSE FOUND (2026-07-22, lit check)**: SYI22 crystals come
      from DECAYING evolution of a **monoscale cap-confined** init with
      scale hierarchy L_i « L_γ ≈ r*/5 (CIL24, arXiv:2403.00870: λ_i
      210–700 km vs L_γ~10⁴ km vs cap 5×10⁴ km; ±5% ring spectrum,
      geostrophically balanced, sponge outside cap). ALL our attempts had
      L_i ~ L_γ — no room for the inverse cascade before γ-arrest → the
      jets/lone-cyclone corner. Also per CIL24: crystals need small dh/h —
      the QG L_d=∞ limit (OUR model) is the crystal-friendly regime; and
      "no simulation has produced steady packed vortices under
      forced-dissipative convective conditions" (open problem = our P2).
      **P0c sweep RUNNING** (GPU7, detached, ~7–9 h): 1024²×8, L=24Lc,
      λ_i=0.5 (k_i=12.57±5%), confined to r*=26, ζ_rms=8.8 (U≈0.7),
      ν₄=5e-10 (rescaled for k_max=18.5!), dt=5e-3, t=400, decaying,
      γ ∈ {1.25e-3, 5e-3, 2e-2} → L_γ ∈ {8.2, 5.2, 3.3}, r*/L_γ ∈
      {3.2, 5, 8}. ars222 safe (ω_edge·dt ≈ 0.026 at worst γ).
      Driver: `--init-confine-radius` (`38366a5`).
      Forced-run data retained as regime-diagram contrast cases. **Analysis tooling ready** (2026-07-22, codex #2,
      verified): `scripts/analyze_polar_p0.py` — crystal_metrics (R_crystal,
      NN spacing via periodic min-image), per-run overlays, log-log slope
      fits vs L_γ; gate = synthetic 19-vortex lattice with exact R/spacing
      (`tests/test_analyze_polar_p0.py`). Live mid-run smoke at t=95 sane
      (γ=0.02: U=0.705, 356 vortices, not yet organized).
      **P0c COMPLETE (2026-07-24): all 3 legs clean to t=400.** Decay left
      U_late≈0.093 ⇒ realized L_γ = {4.2, 2.7, 1.7}, r*/L_γ = {6, 10, 16}.
      Morphology transition in the CIL24 direction: γ=1.25e-3 = mixed-sign
      vortex gas (~40–60 vortices, still coarsening t=200→400);
      γ=5e-3 = ~11 vortices, central cyclone + edge ring; γ=2e-2 = zonated
      bands + central pair. NO lattice yet. Analyzer slopes (R=0.22,
      spacing=−0.29, `analysis/polar_p0c/`) NOT meaningful as the M1 gate:
      high-γ leg zonal (detector counts filaments), low-γ leg unconverged —
      γ-drift segregation estimate ~1200 t.u. vs 400 run. M1 gate ⇒ extend
      low-γ leg to t≈1500 **with the new sponge** (GPU7 currently user-occupied;
      parked).

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
      **PILOT DIED at t=97.25 (2026-07-24, NON-FINITE abort; handover
      correctly did not fire).** Post-mortem: NOT physics — with no trap/drag
      the condensate grew unchecked (U_bt 21→120 over t=40–97, no plateau);
      max_speed 664→3320→NaN in 0.15 t.u. = ARS222 weak imaginary-axis
      amplification on ADVECTIVE frequencies once max_speed·k_max·dt ≈ 0.22
      (same family as the trap-edge instability; physical growth ≤8.6/t.u.
      can't do this). Last good checkpoint t=96. Late-pilot Nu_d
      fluctuated 36–57 (open-top Ra=100 reaches the Miquel both-Dirichlet
      range once the condensate matures). Operational rule recorded: keep
      max_speed·k_max·dt ≲ 0.1 for ars222 (k_max = √2·(Nx/3)·k0 = 6.54 here).
      KEY REFRAME (user, 2026-07-24): the t=10–30 U_bt plateau ≈ 15–16 IS
      the pre-condensate quasi-equilibrium — calibrate there and start the
      trap run from t=30 (`checkpoint_00600000.npz`), not t=96/100.
- [ ] dt stability scan at L=48 Lc (start 5e-5; CFL vs dx=0.45 units).
- [x] Diagnostics cadence + non-finite abort: built into `scripts/run_polar.py`
      (2026-07-22; aborts with exit 2 + emergency checkpoint; structural-NaN
      SBP audit fields excluded by name so real blowups still trigger).
- [x] Polar diagnostics module (2026-07-22, `0006e75`, Sonnet subagent vs
      lead-written gates): `nhqg/polar_diagnostics.py` — azimuthal E(m),
      vortex tracker (periodic NMS + subpixel), radial profiles, trap mask.
- [ ] Vertical-tail monitor (q Chebyshev tail fraction) in run diagnostics.
      **PRIORITY RAISED 2026-07-25** — the tail is now the run-ending term
      (see the M5/P2 blocker entry); a per-step `psi_vert_tail_frac` +
      `max_speed_lowmode` pair would have flagged it 30 t.u. earlier.
- [ ] **Gate**: stable 10-unit pilot, ghost-free, VRAM within budget.

## M5 — Physics campaign (per NHGQ_polar.tex §8)

- [ ] γ-calibration pilot: γ=0 spin-up at L=48 Lc, measure barotropic U_rms,
      set γ for L_γ ≈ 5–7 Lc.
- [ ] P1: γ=0 convective control (trap must not disturb condensate).
- [~] **P2 AMENDED DESIGN (2026-07-24, user-approved) — in flight**:
      restart from pilot t=30 (plateau state, small emergent vortices),
      γ=2.2e-3 from plateau U_bt=15.5 targeting **L_γ = 4 Lc** (down from
      plan's 6 Lc: P0c showed cap capacity r*/L_γ governs the end state),
      **aggressive trap r\* = 0.8·(L/2) = 92.5 = 19.2 Lc** (vs default
      0.45 ⇒ 10.8 Lc) enabled by a **new SYI22-style Rayleigh sponge**
      (σ_max=50, r_s=104, A_s=30, damping q/w/θ — quiescent absorber
      annulus, no periodic-image contamination). Sponge-geometry lesson:
      the ramp must SATURATE before the periodic face or the seam
      derivative-kink trips the 2/3-band guard (first cut r_s=1.18·r*,
      A_s=20 left σ(face)=46 still rising → 2.6e-8 out-band; final r_s=104,
      A_s=30 → 9.2e-12, σ(r*)=0.06, σ(face)=49.94). r*/L_γ = 4.8 ≈
      SYI22 crystal point with hierarchy λ_f/L_γ = 0.25. dt=5e-5 safe again
      from t=30 (advective ω·dt=0.049; edge-wave ~4e-3). ~2.2 days GPU6,
      t=30→250. Caveat recorded: Θ̄ is a domain mean — with ~half the area
      sponged, Nu becomes a cap diagnostic with an area factor; shell
      budgets don't include the sponge sink (appears as residual).
      Sponge implementation: config/grid/spectral/solver + run_polar flags
      + tests/test_sponge.py (codex, CPU-gated; lead runs GPU smoke).
      **MID-RUN VERDICT at t=71 (2026-07-25, 12.8 h in, ~19 min/t.u.):
      the trap+sponge design WORKS and the resolved flow is statistically
      steady.** Versus the γ=0 pilot control at matched t=70:
      KE_bt 92 vs 2305 (**25×**), max_speed 219 vs 356. Over t=31→70 the
      barotropic sector is PINNED: U_bt 15.1–18.4 (design 15.5),
      L_γ = 19.0–20.3 = 3.95–4.21 Lc (no drift), barotropic spectral peak
      fluctuating 3.7–6.9 Lc about L_γ with no trend (arrest, not
      condensation), cap/annulus KE contrast steady ≈ 9–10 (sponge holds),
      q_rms 18.1±0.3, enstrophy 165±5, Nu_d ≈ 10, w_rms(cap) 30.5→32.8.
      NOT organizing yet: cap zeta_bt skewness +0.05±0.05, no trend — no
      cyclone/anticyclone segregation, no lattice. Morphology (png/zeta_bt):
      fine filaments at t=32 → coarsened blobs/spiral bands at t=70 inside
      a clean quiescent-corner disk.
      **FINAL RECORD — P2 reached t=158.2 (2026-07-26), then SIGKILLed when
      the host was withdrawn. Not a blowup: max_speed 260, KE_bt 101 at the
      kill; last checkpoint `checkpoint_03160000.npz` (t=158.0) is clean and
      restartable.** 128 t.u. of trap time ≈ 115 eddy turnovers. Metrics over
      t=31→158 (every 6 t.u., cap-restricted r<r*):
      (i) **arrest is durable, not transient** — U_bt 16.9→18.6 (range
      15.7–20.1, +10% over 127 t.u.), L_γ 4.00→4.23 Lc, no drift;
      (ii) **energy-containing scale saturates at ≈2 L_γ** — barotropic
      L_peak 4.0 Lc at t=31 rising to 6.9–9.6 Lc by t≈90 and then fluctuating
      there, i.e. it does NOT run on to the cap scale (24 Lc);
      (iii) **no segregation** — cap skewness fluctuates ±0.1 about ≈+0.04
      with no trend through t=158;
      (iv) sponge holds — cap/annulus KE contrast 10.1→7.2–9.4.
      Interpretation: trap+sponge deliver a statistically steady, cap-confined,
      arrested turbulent state — the design worked — but 115 turnovers is not
      enough for crystal formation, consistent with P0c's O(10³) t.u.
      segregation estimate. **The campaign's open question is unchanged and now
      purely a matter of integration length**, which is why the vertical sink
      (blocker below) and throughput are the critical path.
- [ ] **BLOCKER (found 2026-07-25 in P2): the vertical Chebyshev tail is
      what will end every long polar run, not physics.** ψ KE fraction in
      vertical modes n>48 (of 64) went **0.296 → 0.530 over t=31→70**, at
      LOW horizontal k (q_rms/enstrophy flat, so it is a ψ-side pile-up).
      Decisive test on the t=70 snapshot: max_speed(full)=218.9 but
      max_speed(n≤32)=142.1 — **identical to t=31 (142.5)**. So ALL of the
      apparent growth (KE_bc ×2.3, max_speed +35%, KE_tot e-fold 48 t.u.)
      is the vertical grid-scale tail; the resolved flow has not changed in
      40 t.u. This is CLAUDE.md §3b CAVEAT 2 (undamped vertical enstrophy
      cascade — no vertical diffusion on q, per Miquel) grown from 27% to
      53%. Consequence: max_speed → 300 (ω·dt = 0.098, the ars222 limit) at
      **t ≈ 110–130**, death (ω·dt ≈ 0.22) at t ≈ 150–220 → P2 will not
      reach t=250. Options: (i) high-n Chebyshev filter / vertical
      hyperdiffusion on q,w,θ (`vertical_cutoff_n` exists in config but is
      NOT wired to run_polar and covers only w,θ); (ii) **Nz=64→32 is
      ~free** — n≤32 carries all resolved physics — and doubles throughput;
      (iii) harvest P2 to death. Wall-clock context: 19 min/t.u. at 512²
      means SYI22-scale segregation (P0c suggested ~1200 t.u.) is weeks —
      the vertical sink + throughput are now the campaign's critical path.
- [ ] P2: γ>0, Ld=∞ — convectively forced crystal (first new result).
- [ ] P3: finite Ld.
- [ ] P4: (γ, Ld) sweep — one run per GPU (no sharding), 6/7.
- [ ] DEFERRED (user, 2026-07-24): 1024²×64 convective run (L=96 Lc,
      r*/L_γ ~ 9–10). Fits ONE GPU memory-wise (~40–80 GB est. peak);
      cost ~2 h/t.u. single-GPU (dt≈2.5e-5 for ars222 margin at k_max
      ×2) → weeks. Revisit only if 512² P2 shows organization, ideally
      after the M3 throughput fix. Barotropic 1024² decay legs are tests,
      not campaign runs (user).

## Done

- [x] 2026-07-22 Plan written + decisions D1–D3 recorded (`polar_512_plan.md`).
- [x] 2026-07-22 Radial-budget notebook delivered
      (`analysis/spectral_budget/radial_spectral_budget.ipynb`).
