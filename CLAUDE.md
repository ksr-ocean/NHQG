## Project Overview

This repository contains a **Nonhydrostatic Quasi-Geostrophic Equation (NHQGE) solver** — a GPU-pseudospectral numerical method in JAX for rapidly rotating Rayleigh-Benard convection in the asymptotic limit Ek -> 0. The ultimate goal is modeling **Jupiter's polar vortex dynamics** (cyclonic vortex crystals observed by Juno).

## Documents

- **NHGQ.tex**: Core formulation — the modified NHQGE system with beta-effect and finite deformation radius Ld, using Chebyshev-Gauss-Lobatto (CGL) collocation in the vertical and Fourier pseudospectral in the horizontal. Includes full ARS(2,2,2) IMEX scheme with unified dissipation, DCT-I analytic inverse, mean temperature evolution.

- **NHGQ_polar.tex**: Polar cap extension via the "trap method" (Siegelman, Young & Ingersoll 2022).

- **Miquel_NHGQtilted2026_arxiv.pdf**: Reference paper by Miquel et al. 2026. Key equations: (3.1a-d) for the NHQGE system including mean temperature evolution.

- **hermitian_ghost.md**: THE diagnosis of the raw-diagnostics "blowup"
  (anti-Hermitian ky=0 ghost mode, 2026-07-04). Read this before trusting any
  raw/spectral diagnostic or the dominant-shell narrative.

- **NHQG_framework_deck.tex / .pdf**: Pedagogical slide deck (41 slides,
  2026-07-03): full numerical framework + the ghost-mode diagnosis with
  checkpoint evidence. Build with `tectonic NHQG_framework_deck.tex`.

- **blowup.md**: HISTORICAL. Record of the late-time high-`Nz` failure
  investigation. The failure is resolved (clean to `t=120`); see the
  resolution banner at its top.

- **spectral_analysis.md**: Guide to the shell-budget archives. WARNING: its
  dominant-shell (`k = 0.9786`) conclusions are ghost-contaminated — see the
  banner at its top and `hermitian_ghost.md`.

- **adjoint_mean_exchange.md**: HISTORICAL. Mean-temperature /
  fluctuation-exchange branch record; ends with the validated
  `balanced_sbp2_pc` baseline. See its final-status section.

- **CODE-REVIEW.md** (2026-04-19): pedagogical code walkthrough.
  **efficiency_review.md**: per-step cost audit (items #1/#3 since
  implemented; `23_rule` since adopted).

- **sharding_implementation.md** (2026-07-22): teaching notes on the M3
  2-GPU sharding — GSPMD mental model, how the layout flows through the
  step, why kx is the only shardable axis, CPU-vs-GPU gate split.

- **fd_vertical_benchmark/**: separate FD-vertical solver package (SBP42 /
  compact4, uniform or tanh grid) targeting the future mixed-BC goal
  (Neumann-w top / Dirichlet bottom via SBP-SAT). Own README.
  **dinosaur_spike/**: unrelated May-2026 feasibility spike (NeuralGCM /
  Dinosaur spherical hosting for a polar model).

- Run outputs now live under the repo-level `output/` directory. Treat run
  names in these notes as `output/<run_name>`.

## Solver Code Structure

```
nhqg/
├── __init__.py          # Public API re-exports
├── config.py            # Frozen dataclass NHQGConfig (all parameters)
├── grid.py              # CGL points, coefficient-space Chebyshev operators (G_Z, G_Z2),
│                        #   Chebyshev Vandermonde transforms (V, V_inv), CC weights,
│                        #   tau BC projection matrices, wavenumber grids,
│                        #   dissipation rates/alpha factors,
│                        #   IMEX inverse matrices with |k|^2 shell dedup and unified dissipation
├── spectral.py          # 3/2-rule dealiased Jacobian, fused triple-Jacobian
├── solver.py            # ARS(2,2,2) IMEX-RK stepper with unified dissipation,
│                        #   RK4 validator (exponential dissipation), RHS, BCs, main loop,
│                        #   mean temperature evolution (fixed_conduction or evolve_mean),
│                        #   coefficient <-> nodal transforms via V/V_inv
├── diagnostics.py       # Barotropic mode, energy spectra, KE/enstrophy/Nusselt
└── io.py                # NetCDF snapshots (netCDF4), checkpoint save/load (.npz)

tests/
├── test_grid.py         # V/V_inv roundtrip, G_Z polynomial accuracy, G_Z2 accuracy,
│                        #   tau projection (Dirichlet/Neumann), CC integration, IMEX shells
├── test_spectral.py     # Jacobian analytic checks, antisymmetry, dealiasing
├── test_solver.py       # Linear onset, IMEX convergence order, IMEX vs RK4, BCs,
│                        #   mean temperature closure, shell-budget closure,
│                        #   exchange-branch smoke tests
├── test_fd_vertical.py  # FD-vertical operator identities (sbp42/compact4)
├── test_trig_vertical.py# Trig-basis vertical experiments
└── test_vmap.py         # Parametric consistency over (beta, Ld) (sequential, no actual vmap)

scripts/
├── run_rubio.py         # Case 1: Rubio et al. 2014 (beta=0, Ld=inf, Ra=100)
├── run_sweep.py         # Parametric (beta, Ld) sweep
├── benchmark.py         # Timing + memory profiling
└── submit.slurm         # Bridges-2 GPU job script
```

## Running

```bash
# Run all tests (89 tests as of 2026-07-03, all passing; ~70 s on CPU)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest tests/ -q

# Rubio reproduction (default: Nx=256, Nz=128, Laplacian nu=1)
JAX_ENABLE_X64=1 PYTHONPATH=. python scripts/run_rubio.py [Nx] [Nz]

# Benchmark
JAX_ENABLE_X64=1 PYTHONPATH=. python scripts/benchmark.py

# Build LaTeX docs (pdflatex/TeX Live is no longer on PATH; use tectonic)
tectonic NHGQ.tex
tectonic NHGQ_polar.tex
tectonic NHQG_framework_deck.tex
```

## Environment

- Python 3.12.2, JAX 0.9.1 (CUDA 12), numpy 2.0.2, xarray 2026.2.0
- **JAX requires numpy >= 2.0** — `jnp.zeros` inside JIT fails with numpy 1.x
- Set `JAX_ENABLE_X64=1` for float64 support (essential for validation tests)
- CUDA JAX installed: `pip install --upgrade "jax[cuda12]"` (done)
- Bridges-2: `GPU-shared` partition, V100-SXM2-32GB and H100-80GB available, CUDA 12.6

## Array Layout

- **State representation**: horizontal-spectral (rfft2) fields with vertical coefficients:
  - `q_hat`: full Chebyshev coefficients, shape `(Nz+1, Nx, Nx//2+1)` complex
  - `w_hat`, `th_hat`: **Dirichlet-Galerkin coefficients** (Coral stencil `-T_n + T_{n+2}`), shape `(Nz-1, Nx, Nx//2+1)` complex — BCs exact by construction, no tau rows
  - `th_bar`: Chebyshev coefficients, shape `(Nz+1,)` real
- **rfft2 reality constraint**: the ky=0 (and ky-Nyquist) columns must satisfy `f_hat(-kx) = conj(f_hat(kx))`. Nothing currently enforces this on evolved state — see `hermitian_ghost.md` (the ghost-mode finding).
- **Physical space**: `(Nz+1, Npad, Npad)` float — product evaluation (Npad = 3Nx/2 under `32_rule`; Nx under `23_rule`)
- **Coefficient-space vertical ops**: `jnp.einsum('ij,j...->i...', G_Z, field)` contracts over axis 0
- **Coefficient <-> nodal transforms**: `V @ coeffs = nodal`, `V_inv @ nodal = coeffs`
- **Dissipation arrays**: `(Nx, Nk)` — broadcast with `[None, :, :]` for (Nz+1, Nx, Nk) fields

## Key Physics & Notation

- Four prognostic fields: perturbation PV (q'), vertical velocity (w), temperature perturbation (theta), mean temperature deviation (Theta_bar')
- Streamfunction psi recovered from q' via spectral inversion: psi_hat = -q_hat'/(|k|^2 + Ld^{-2}), pointwise per Chebyshev coefficient
- Vertical: Galerkin/tau method in Chebyshev coefficient space; derivatives via coefficient-space recurrence (G_Z); transforms between coefficient and nodal space via Vandermonde V and analytic DCT-I inverse V_inv
- Horizontal: doubly-periodic Fourier with dealiased products — `horizontal_dealiasing='32_rule'` (padded, code default) or `'23_rule'` (unpadded + output mask, **production since 2026-04**; caveats in the 2026-07-04 status section)
- BCs: w, theta Dirichlet (exact via the Dirichlet-Galerkin basis); q' has NO boundary condition by default (`q_boundary='none'`, Miquel-style); Theta_bar Dirichlet via tau rows
- **q' BC rationale**: q' is obtained from psi via a pure horizontal inversion (q_hat = -(|k|^2 + Ld^{-2}) * psi_hat). The physical constraint is w=0 at boundaries, which forces d(psi)/dZ = 0 there. But that constrains the w equation, not q directly. Imposing a separate Neumann BC on q was an error — q needs no BC of its own. Optional `q_boundary='neumann'` retained for comparison.
- Key parameters: Ra_tilde (reduced Rayleigh number), sigma (Prandtl), beta (PV gradient), Ld (deformation radius)
- Onset: Ra_c ~ 8.6956, k_c ~ 1.3048 (stress-free boundaries)
- Mean temperature: two modes — `fixed_conduction` (Theta_bar = 1-Z) or `evolve_mean` (prognostic via Miquel eq. 3.1d)

## Current Status (2026-07-04)

This section supersedes the two 2026-04-11 sections below (kept as historical
record). Full detail: `hermitian_ghost.md` and `NHQG_framework_deck.pdf`.

### 1. Late-time stability: RESOLVED

The `balanced_sbp2_pc` + `sbp_corrector_substeps=4` + `flux` + `23_rule`
branch is stable through the full tested window:

- `Nx=64, Nz=256`: clean to `t = 80` (the 2026-04-11 baseline below)
- `Nx=128, Nz=256`: clean to **`t = 120`**
  (`output/output_combined_Nx128_Nz256_t40_to_t70_sub4_23rule_stack13_snap01`,
  upsampled from the 64^2 state at t=40; `Nu_d ~ 19-22`, `R_ex_sbp` at roundoff)
- `Nx=256, Nz=256`: clean to `t = 63`
  (`output/output_combined_Nx256_Nz256_t41_to_t58_sub4_23rule_stack13_snap01`)

This is the current best run class. The old `t~11` and `t~41-50` failures are
history.

### 2. The raw-diagnostics "blowup": DIAGNOSED (2026-07-03)

The raw `Nu`/`KE_bc`/shell-budget explosions in otherwise healthy runs
(`Nu_raw = 6.6e27` at t=120 while `Nu_d = 21.9` and `max_w = 412`) are an
**anti-Hermitian ghost mode in the rfft2 ky=0 column**: invisible to
`irfft2`-based physics, counted by Parseval-style diagnostics, fed nothing by
the (exactly Hermitian) nonlinear tendencies, and therefore growing at the
unsaturated linear rate forever (measured 8.4/unit early vs theory 8.6; ~0.4
late on the saturated mean). Verified directly on checkpoints (ghost
amplitude 1.07e19 at t=120; Parseval/projected energy ratio ~1e26).

Consequences:

- The **`k = 0.9786` dominant-shell narrative is ghost-contaminated** (that is
  the ghost's home shell; the shell budgets are spectral inner products).
  Re-derive from Hermitian-projected states before reading physics from it.
- Trust only physical-product diagnostics: `Nusselt_dealiased`,
  `vol_avg_tw_dealiased`, dealiased shell budgets/residuals, `max_*`.
- Fix (cheap, NOT yet implemented): symmetrize the ky=0 column each step +
  Hermitian initial noise + regression test. See `hermitian_ghost.md`.
- `fd_vertical_benchmark/` has the same bug independently.

### 3. Headline open problem: the Nusselt gap

Stable runs at `Ra=100` give time-mean `Nu_d ~ 18-20` vs Miquel's
`43.37 +/- 2.54` (our value matches their `Ra=60` entry, 19.88). Stability is
solved; accuracy is not. Suspects, in order:

1. the piecewise-linear CGL<->SBP `interp` transfer smooths the thermal
   boundary layer (11 CGL points inside the first uniform cell at Nz=256,
   applied 8x/step) — and Nu IS the boundary-layer gradient;
2. `Nx=64` heritage: under `23_rule` at Nx=64 the masked band
   `k in (2.74, 3.16)` is linearly unstable but nonlinearly frozen (mask is
   output-only; the state is never truncated) and can only saturate by
   flattening the mean — the 128^2 chain inherited that mean;
3. effective horizontal resolution under `23_rule` (128^2 stored -> ~85^2
   usable) vs Miquel's fully-usable 256^2 at Ra=100.

### 3b. Ghost-clean radial spectral budgets (2026-07-04)

`scripts/spectral_budget_radial.py` computes angle-averaged (radial-shell)
energy and enstrophy budgets of the vorticity equation from checkpoints, with
the ghost Hermitian-projected out and the 2/3-rule product path matching the
solver. Results for the `128x256` chain, t=80-120 (161 snapshots), in
`analysis/spectral_budget/window_t80_t120/`:

- textbook dual cascade: inverse energy cascade `Pi_E = -1309` through
  k~0.2 into the domain-scale condensate (balanced there by Laplacian
  dissipation, no drag); forward enstrophy cascade `Pi_Z = +771` at k~2.5;
  ~8% of enstrophy injection exits through the 2/3 mask (truncation sink).
- budgets close: energy stretch (closure-inferred) +1224 vs diss -1317;
  enstrophy stretch +2543 +/- 497 vs diss -2461.
- CAVEAT 1: the raw sampled KE<->w conversion term is aliased by low-k
  (k < 0.35) oscillatory modes at the 0.25 checkpoint cadence (swings ~1e5);
  use the closure-inferred estimator (dE/dt fit - adv - diss). Raw and
  inferred agree for k > 0.5.
- CAVEAT 2 (new finding): q is vertically under-resolved — up to 27% of its
  vertical Chebyshev energy in the top 16 modes (psi at low k: 68%), from the
  undamped vertical enstrophy cascade (no vertical diffusion on q). w is
  smooth (1e-3). Inflates baroclinic E(k) at low k; barotropic sector immune.
- Window-convergence audit (2026-07-06, `analysis/spectral_budget/compare_*.png`,
  windows also run for the 64^2 from-start (3/2-rule path) and 256^2 chains):
  (i) NOT developed by t=20 — at t=20 the flow carries <10% of its mature KE;
  everything (spectra, fluxes, condensate) still grows monotonically through
  t~80. (ii) On the 128^2 chain, horizontal E(k)/Th(k) are near-stationary
  within t=80-120 (worst-shell x1.5-2.4 between 80-100 and 100-120), but the
  vertical KE tail still fills x35 between those halves — no fully
  vertically-converged window exists; use t=80-120 with that caveat.
  (iii) The young 256^2 chain (t=43-63) is the cleanest: all three spectral
  criteria converged between halves, vertical spectrum smooth to 2e-4 (no
  rough tail yet), ghost only ~3e7; but fluxes still intensify ~35% between
  halves. (iv) Cross-resolution at matched epochs: 128^2 vs 256^2 collapse in
  E(k) (x1.26 worst) and Th(k) (x1.17); 64^2 shows inflated Th(k) at k>0.8 and
  ~50% stronger forward enstrophy flux (its masked band k in (2.74,3.16) is
  the suspect).

### 4. Other 2026-07-03 review findings worth knowing

- `mean_theta_exchange_residual_sbp = 0` is a **structural identity**
  (boundary term killed by the Dirichlet rows) — it audits SBP operator
  algebra, not the integrated step. Do not read it as a step-consistency audit.
- 2/3-rule fine print: `K = Nx//3` with `<=` is incompletely dealiased when
  `3 | Nx` (Nx=96 leaves an O(1) alias on the retained shell); 64/128/256 are
  safe by accident. The state should be masked at init/restart/finalize.
- Under `balanced_sbp2_pc` the archived `th_nonlinear`/`th_total` shell
  decomposition is mislabeled (subtracts a mean-feedback term the RHS never
  contained).
- `q/w/th_horiz_spec` lack the `Nx^4` normalization that `ke_horiz_spec` has —
  cross-resolution comparisons of the raw spectra are off by `(Nx1/Nx2)^4`.
- Checkpoints store state only (no config): restart physics depends entirely
  on CLI flags, whose silent defaults are `legacy`/`32_rule`/`jacobian`.
  Neither May-19 driver stops on non-finite diagnostics, and diagnostics
  archives are written only at run end.
- The repo has **zero git commits** — highest-priority hygiene item
  (`.gitignore` for `output/` (~310 GB), logs, PDFs; then commit).

## Historical: Investigation Status (2026-04-11) — superseded above

The old interpretation "raw `Nu` blows up, therefore the resolved state is
obviously blowing up" is no longer accepted.

In the rebuilt dealiased archives:

- `128x128`, `t=8.0`: raw `Nu = 1.68e17`, dealiased `Nu_d = 3.89e1`,
  dealiased exchange residual relative error `= 8.5e-6`
- `64x256`, `t=8.0`: raw `Nu = 5.12e18`, dealiased `Nu_d = 3.70e1`,
  dealiased exchange residual relative error `= -5.9e-6`

Operational consequence:

- for `thermal_closure='evolve_mean'`, use `Nusselt_dealiased`,
  `vol_avg_tw_dealiased`, and the dealiased exchange residuals as the primary
  thermal diagnostics
- keep raw `Nusselt` and raw `vol_avg_tw` only as aliased comparison metrics

What remains unresolved is a later-time state failure at high vertical
resolution.

Best current `64x256`, `dt=5e-5` crash-window comparison:

- `legacy`: first non-finite at `t = 11.22`
- `coral_workgrid_flux`: first non-finite at `t = 11.44`
- `balanced_midpoint`: unstable earlier, with a full-start failure window
  around `t = 10.4 - 10.6`
- `balanced_sbp2`: finite through `t = 12.0` in the same restart window

The shell budgets now point away from a broadband cascade explanation:

- nonlinear shell transfer is conservative to numerical accuracy,
- nonlinear flux remains modest,
- the dominant positive KE source is low-shell stretching near
  `k \approx 0.9786`,
- mid and high horizontal shells remain weak.

So the active problem statement is:

- the unresolved issue is a late-time high-`Nz` state failure tied most
  plausibly to the discrete mean-temperature / fluctuation thermal coupling,
- not the old collocation null mode,
- and not the old aliased thermal observables themselves.

The current methodological next step is the simpler
`sbp2-balanced-exchange` branch:

- keep the horizontal Fourier pseudospectral machinery unchanged,
- keep the existing `3/2` horizontal dealiasing,
- replace only the vertical thermal-exchange substep.

Current status of that branch:

- implemented,
- uses stable nodal transfer between the CGL solver grid and a uniform SBP2
  work grid,
- currently the most robust mean-exchange variant tested in the `64x256`
  late-time window.

Active references:

- `blowup.md`
- `spectral_analysis.md`
- `adjoint_mean_exchange.md`

## Historical: Best Run Class (2026-04-11) — superseded by the 2026-07-04 status

The operational baseline has moved beyond the old `balanced_sbp2` split branch.

Current best branch:

- `mean_exchange_discretization = balanced_sbp2_pc`
- `sbp_corrector_substeps = 4`
- `sbp_transfer_mode = interp`
- `nonlinear_advection = flux`
- `Nx = 64`, `Nz = 256`, `dt = 5e-5`

Best completed clean continuation:

- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_subcycle4_probe_t80_from_t20_Nx64_Nz256_dt5e5`
- finite through `t = 80.0`
- final diagnostics:
  - `Nusselt_dealiased = 19.517732154543815`
  - `mean_theta_exchange_residual_dealiased = -0.027970012641981512`
  - `mean_theta_exchange_residual_sbp = 0.0`
  - `max_w = 262.4614942126351`
  - `max_theta = 12.613831394874051`

Operational interpretation:

- the old `t \approx 11` and delayed `t \approx 41-50` failures are no longer
  the best measure of the current branch family,
- the trusted exchange audit is now the SBP-side residual,
- and the current remaining issue is long-time confirmation plus representation /
  coupling consistency, not a clear immediate blow-up in the best branch.

Matching from-start snapshot run:

- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_subcycle4_fromstart_Nx64_Nz256_dt5e5_t80_snap01`
- this is the movie/visualization archive and is being continued to `t = 80`.


## GPU Restart Notes

When restarting GPU experiments, keep BLAS thread counts at `1`. The long
startup stalls on this node come from host-side dense linear algebra during
IMEX-shell precomputation, not from XLA compilation alone.

Recommended environment:

```bash
export JAX_ENABLE_X64=1
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Implementation Notes

### Galerkin/tau formulation (replaces collocation)
All fields are stored as **Chebyshev coefficients**. Vertical derivatives use the **coefficient-space recurrence** G_Z (not dense collocation matrices D_Z). This eliminates the D_Z interior null mode that caused instability at high Nz with the old collocation approach. BCs are enforced via **tau rows** (replacing the last two equations in each IMEX system with boundary constraints). No post-step Chebyshev spectral filter is needed.

Key operators:
- `G_Z = 2 * G_xi`: first derivative in coefficient space (G_xi from Chebyshev recurrence)
- `G_Z2 = G_Z @ G_Z`: second derivative in coefficient space
- `V[j,n] = T_n(xi_j)`: Chebyshev Vandermonde (coefficients -> nodal values)
- `V_inv`: analytic DCT-I inverse (nodal -> coefficients), `V_inv[n,j] = (2/(N*c_n*c_j))*cos(n*pi*j/N)`
- `proj_dirichlet`, `proj_neumann`: tau projection matrices adjusting last 2 coefficients to satisfy BCs

### q' boundary condition: `q_boundary` config parameter
- `q_boundary='none'` (default, Miquel-style): mathematically the q' solve is simple scalar division `q_hat / alpha_q`. No tau rows needed for q — it is a pure horizontal variable with no boundary constraint. (Implementation caveat 2026-07-03: the code currently still stores dense per-shell identity-scaled `q_solve` matrices and applies them via the expensive gather-matmul path; the scalar `inv_alpha_q` is precomputed but unused. Cleanup pending — see VRAM notes.)
- `q_boundary='neumann'` (legacy): Neumann BCs dq'/dZ=0 at Z=0,1 enforced via tau rows in the q solve matrix.

### Normalization correction (important!)
The 3/2-rule Jacobian truncation factor is `(Npad/Nx)^2 = (3/2)^2 = 9/4`. When zero-padding spectral coefficients from Nx to Npad and doing `irfft2`, physical values are attenuated by `(Nx/Npad)^2`. The product of two such fields is attenuated by `(Nx/Npad)^4`. After `rfft2` on the Npad grid and conversion to Nx-grid DFT normalization (another `(Nx/Npad)^2`), the net correction is `(Npad/Nx)^2 = 9/4`.

### Clenshaw-Curtis weights
The code uses the standard CC formula with the b_k boundary correction.
(Correction 2026-07-03: exactness was measured to end at degree ~N+1, not the
"2N" previously claimed here — CC is never 2N-exact. Harmless in practice.)

### Per-field dissipation (important!)
All fields use the same `hyper_order` p for dissipation, with per-field coefficients:
- **q'**: `nu_q*|k|^{2p} + drag` (enstrophy cascade + optional large-scale drag)
- **w**: `nu_w*|k|^{2p}` (forward cascade control)
- **theta**: `(nu_theta/sigma)*|k|^{2p}` (scalar cascade control)

The `hyper_order` config parameter (p) applies to ALL fields. With p=1 all fields get Laplacian (molecular diffusion). With p=4 all fields get hyperviscosity. The w-theta buoyancy coupling creates a positive feedback at all scales (small-scale theta -> w via Ra*theta, small-scale w -> theta via convective stirring), so both must be controlled simultaneously.

### Unified IMEX dissipation
All horizontal dissipation is folded into the IMEX implicit system via scalar alpha factors:
- `alpha_q(k) = 1 + gamma*dt*(nu_q*|k|^{2p} + drag)`
- `alpha_w(k) = 1 + gamma*dt*nu_w*|k|^{2p}`
- `alpha_theta(k) = 1 + gamma*dt*(nu_theta/sigma)*|k|^{2p}`

The modified IMEX matrix (in coefficient space) is `A'(k) = alpha_w_eff*I - (gamma*dt)^2*c(k)*B` where `B = G_Z @ q_solve @ G_Z` and `alpha_w_eff = alpha_w - (gamma*dt)^2*Ra_sigma/alpha_theta`. Tau Dirichlet rows replace the last 2 rows of A'. This preserves 2nd-order accuracy.

### Implicit buoyancy coupling
Ra*theta (w equation) and w source (theta equation) are in the implicit system. Block elimination gives `alpha_w_eff = alpha_w - (gamma*dt)^2*Ra_sigma/alpha_theta` (scalar per shell, same matrix structure). `imex_implicit_solve` takes and returns (q, w, theta) together.

### IMEX solver
Uses precomputed (A')^{-1} (matrix inverses) per |k|^2 shell rather than LU factorizations. This is simpler for JAX since (A')^{-1} * rhs is a matmul. The ARS(2,2,2) scheme has gamma = 1 - 1/sqrt(2) ~ 0.2929, delta = -sqrt(2)/2 ~ -0.7071.

### |k|^2 shell deduplication
The IMEX matrix A'(k) depends only on |k|^2, not (kx, ky) individually (since alpha_q, alpha_w, c(k) are all functions of |k|^2). Shell counts (measured): 5924 at Nx=256, 82817 at Nx=1024 — the "~1500 shells / ~12 MB" figures previously quoted here were wrong (see the benchmark table, which was always correct). Two known costs (2026-07-03 review, matches the VRAM memory note): (i) for `q_boundary='none'` the q-solve is stored as dense per-shell scaled identities while the scalar `inv_alpha_q` sits precomputed and unused; (ii) the runtime gather `mat_shells[ksq_idx]` materializes the full `(Nx, Nk, Nz+1, Nz+1)` tensor (~1.1 GB per gather at 64x256 float64) — the root of the recorded 114 GB peak. Shell dedup saves storage, not working set.

### Optional vertical cutoff (`vertical_cutoff_n`)
Experimental: zero Chebyshev coefficients n > cutoff_n for w and theta after each step. Re-apply Dirichlet projection after zeroing. Set `vertical_cutoff_n=None` (default) to disable.

### Precision control (`float_dtype`)
Config parameter `float_dtype` controls precision. WARNING: the dataclass
default is `"float32"`, but production runs REQUIRE `"float64"` (see the
float64 memory note) and every script overrides it. A bare `NHQGConfig()`
silently gets float32 operators. More generally, bare defaults match the
documented production configuration on almost no axis (`legacy`, `32_rule`,
`jacobian`, `hyper_order=4` with `nu=0`, `L=20`) — everything load-bearing
lives in script-level overrides.

### Stability and resolution requirements
The NHQGE dispersion relation (stress-free, n=1 vertical mode) gives growth rate `s(k) = -nu*k^2 + sqrt(Ra - pi^2/k^2)`. At Ra=100, max growth rate is ~8.6 at k~0.83, with unstable range k~0.35 to k~3.1.

Key findings:
- **q'-only dissipation is NOT sufficient**: the w-theta buoyancy coupling has independent growth at every wavenumber. Without dissipation on w,theta, ALL modes are unstable.
- **Laplacian diffusion (nu=1, p=1) on all fields**: Matches Miquel's formulation. Resolution-demanding (forward enstrophy cascade requires k_max ~ O(Ra)), but physically correct.
- **Recommended production config**: hyper_order=1, nu=1 for all fields (Laplacian molecular diffusion matching Miquel). No vertical diffusion on fluctuation fields.

## Verification Results (89 tests as of 2026-07-03, all passing)

Current per-file counts: `test_grid.py` 37, `test_solver.py` 31,
`test_spectral.py` 8, `test_fd_vertical.py` 7, `test_trig_vertical.py` 4,
`test_vmap.py` 2. The detailed descriptions below are from the 50-test era and
remain valid but incomplete. Known coverage gaps (2026-07-03 review): the
production configuration (`23_rule` + `flux` + `interp` + `balanced_sbp2_pc`)
is essentially only smoke-tested — no `23_rule` test exists at all, no
solver-level `flux` test, no `interp`-transfer test, no io/checkpoint
round-trip test, no float32-path test, no Hermitian-symmetry test, and
`test_vmap.py` does not actually vmap.

### Grid tests (34 tests) — `test_grid.py`
- **V/V_inv roundtrip**: V @ V_inv = I and V_inv @ V = I to ~1e-13
- **V evaluates Chebyshev**: V[j,n] = T_n(xi_j) to ~1e-14
- **G_Z polynomial accuracy**: First derivative of Z^d exact to ~1e-10 at Nz=8, ~1e-8 at Nz=32
- **G_Z2 polynomial accuracy**: Second derivative of Z^d exact to ~1e-9 at Nz=8, ~1e-6 at Nz=32
- **G_Z2 sin test**: Second derivative of sin(pi*Z) accurate to ~1e-8 at Nz=32
- **CC quadrature**: integrals of 1, Z, Z^2, sin(pi*Z) all exact to 1e-14
- **Tau Dirichlet projection**: f(Z=0) = f(Z=1) = 0 to ~1e-13 after projection; idempotent
- **Tau Neumann projection**: df/dZ|_{Z=0,1} = 0 to ~1e-12 after projection; idempotent
- **IMEX shells**: Shell count < total wavenumber pairs; all indices valid

### Spectral tests (6 tests) — `test_spectral.py`
- **J[sin(x), sin(y)] = cos(x)cos(y)**: error ~1e-12
- **J[sin(2x), sin(3y)] = 6cos(2x)cos(3y)**: error ~1e-11
- **Antisymmetry**: J[A,B] + J[B,A] = 0 to ~1e-12
- **Self-Jacobian**: J[A,A] = 0 to ~1e-12
- **Dealiasing**: padded vs unpadded results differ for aliasing-prone products
- **Triple-Jacobian**: fused evaluation matches individual calls to ~1e-12

### Solver tests (9 tests) — `test_solver.py`

#### Linear onset test (Ra_c verification)
Set Ra = 1.01 * Ra_c, beta=0, Ld=inf, Nx=64, Nz=16. Domain size L = 4*2*pi/k_c ~ 19.3. Initialize with small-amplitude random noise with sin(pi*Z) vertical envelope. Run 100 IMEX steps at dt=1e-4. Peak energy wavenumber within 3*dk of k_c.

#### IMEX convergence order
Richardson extrapolation: error ratio ~4 for 2nd-order scheme.

#### IMEX vs RK4 agreement
At dt=1e-4, 20 steps. Relative difference in q' < 1e-3.

#### Boundary condition enforcement (3 tests)
After 10 IMEX steps: w and theta boundary values < 1e-12, dq'/dZ at boundaries < 1e-10 (when q_boundary='neumann'), k=0 mode < 1e-14 for all fields.

#### Mean temperature closure (3 tests)
- `fixed_conduction`: Theta_bar stays zero
- `evolve_mean`: Theta_bar boundary values preserved to 1e-14
- `evolve_mean` with correlated flux: Theta_bar interior evolves

### Vmap tests (2 tests) — `test_vmap.py`
- Different beta values produce distinct solutions; all remain finite
- Different Ld values produce distinct solutions

### GPU Benchmark (V100-SXM2-32GB, float64, JAX 0.9.1 + CUDA 12.6)
| Nx   | Nz | mean(ms) | steps/s | state(MB) | IMEX(MB) | shells |
|------|----|----------|---------|-----------|----------|--------|
| 64   | 8  | 0.76     | 1308    | 0.9       | 0.3      | 457    |
| 128  | 16 | 2.88     | 347     | 6.8       | 3.7      | 1621   |
| 256  | 32 | 24.2     | 41      | 52.3      | 51.6     | 5924   |
| 512  | 32 | 98.5     | 10      | 208.4     | 191.9    | 22027  |
| 512  | 64 | 214.8    | 4.7     | 410.5     | 744.5    | 22027  |
| 1024 | 64 | 967.2    | 1.0     | 1639.0    | 2799.2   | 82817  |

All timings in float64. The V100 has 1/32 float64 throughput — switching to float32 (production) should give ~10-30x speedup.

## Resolved: D_Z Null-Mode Bug (2026-03-18)

This section documents a previously resolved numerical defect in the old collocation formulation. (The separate late-time high-`Nz` failure that was once tracked in `blowup.md` has since also been resolved — see the 2026-07-04 status section. Note the motif shared by the D_Z null mode, the untruncated 2/3-rule band, and the anti-Hermitian ghost of `hermitian_ghost.md`: state content invisible to the physics but amplified by linear dynamics.)

### Root cause: Chebyshev D_Z interior null mode

The original collocation first-derivative matrix D_Z had a **rank-deficient interior block**: D_Z[1:N, 1:N] has rank N-2. The null vector (even/odd CGL point alternation) created a **spurious unstable eigenmode** with growth rate = sqrt(Ra), present at ALL horizontal wavenumbers. This is the Chebyshev analog of Fourier Nyquist aliasing.

### Fix: Galerkin/tau method (replaces spectral filter)

The initial fix was a Chebyshev spectral filter projecting out the T_N coefficient. The current, definitive fix is the **Galerkin/tau formulation**: fields are stored as Chebyshev coefficients, derivatives use the coefficient-space recurrence, and BCs are enforced via tau rows. The D_Z null mode simply doesn't exist in this representation — the coefficient-space derivative operator G_Z is exact for all Chebyshev polynomial degrees.

### q' boundary condition correction

The original code imposed Neumann BCs (dq'/dZ = 0) on q', motivated by stress-free conditions. This was an error: q' = -(|k|^2 + Ld^{-2}) * psi is a **pure horizontal inversion** of psi and requires no boundary condition. The physical constraint is w=0 at boundaries (which forces d(psi)/dZ = 0 via the implicit coupling), but this constrains w, not q. With `q_boundary='none'` (now default), the q solve reduces to simple division by alpha_q.

### Implicit buoyancy coupling (retained)
Ra*theta and w source terms are in the implicit system. Block elimination gives `alpha_w_eff = alpha_w - (gamma*dt)^2*Ra_sigma/alpha_theta`.

### Miquel comparison targets (Table 1, theta_f=0)
| Ra_tilde | Nx*Ny*Nz | Nu +/- sigma | Re_l +/- sigma |
|----------|----------|--------------|----------------|
| 10 | 128^2 x 256 | 1.27 +/- 0.01 | 0.75 +/- 0.11 |
| 20 | 128^2 x 256 | 4.02 +/- 0.13 | 3.55 +/- 0.79 |
| 40 | 128^2 x 256 | 12.28 +/- 0.60 | 10.67 +/- 2.43 |
| 80 | 128^2 x 256 | 30.96 +/- 1.81 | 24.28 +/- 7.39 |
| 100 | 256^2 x 384 | 43.37 +/- 2.54 | 32.05 +/- 8.24 |

Miquel uses: Coral spectral code, IMEX RK443 (3rd order, 4 stage), Chebyshev vertical, Fourier horizontal, domain 10L_c x 10L_c x 1, Laplacian dissipation nu=1 on all fields, NO vertical diffusion on fluctuation fields.

## LaTeX Conventions

- Custom commands defined at the top of each file (e.g., `\Rav`, `\Ek`, `\Ld`, `\Jac`, `\hhat`, `\Dvec`, `\pderiv`)
- Boxed equations (`\boxed{}`) mark the key results/definitions
- Both files share the same preamble structure and command definitions

## Restart Notes (2026-03-22)

### New utility script
- Added `scripts/run_miquel_compare.py` for upright (`beta=0`, `Ld=inf`) comparison runs.
- Purpose: run long Miquel-style zero-tilt cases and save both NetCDF snapshots and PNG panel grids.
- PNG layout: rows `[w, theta, zeta]`, columns `[top, mid, bottom]`, where `zeta = q'` in the upright infinite-`Ld` case.
- Uses Pillow instead of matplotlib because the local matplotlib build is broken against NumPy 2.
- Script now stops cleanly when diagnostics become non-finite.

### Important investigation update
- Matching Miquel's horizontal domain size `L = 10 L_c` did **not** fix the blowup.
- H100 run: `Nx=128`, `Nz=128`, `Ra=100`, `dt=1e-4`, Laplacian `nu=1` on all fluctuation fields, `q_boundary='none'`, `fixed_conduction`, `L=10 L_c`.
- Behavior: linear growth until about `t\approx 2.7`, then runaway nonlinear growth and NaN by `t\approx 3.8`.
- Conclusion: the old `L=20` vs Miquel `10 L_c` mismatch was not the root cause.

### Practical note on full Miquel resolution
- A quick H100 benchmark at `Nx=256`, `Nz=384` showed the JAX compile path is extremely heavy: warning about ~14 GB captured constants and GPU memory climbed to ~61 GB before first-step timing returned.
- Full `256^2 x 384` interactive runs are therefore expensive enough that staged/batch execution is the safer path.

### Coral alignment update
- Coral source is available locally at `../coral-1.1.12`.
- `src/cheby_tools/chebyshev_galerkin_2.f90` confirms that Coral's both-Dirichlet Chebyshev Galerkin basis is the simple stencil `-T_n + T_{n+2}` (`bc_type=20`).
- The NHQG code now mirrors that choice: `w` and `theta` are stored in Dirichlet Galerkin coordinates, while `q'` remains in full Chebyshev coefficients.
- `grid.py` now builds `dirichlet_stencil` / `dirichlet_pinv`, and the IMEX `w` solve is assembled in the reduced `(Nz-1)` Galerkin space instead of tau-enforcing `w,theta` in full Chebyshev space after the fact.
- Diagnostics / snapshot output now lift `w,theta` back through the stencil before nodal or physical-space evaluation.

### RK443 correction
- Coral `src/timesteppers/IMEX_schemes.f90` showed the earlier RK443 implementation was finalized incorrectly: ARS443 is **not** stiffly accurate and requires a weighted final combination of stage derivatives.
- `solver.py` was corrected to use Coral's ARS443 weights.
- `grid.py` also now builds the precomputed IMEX matrices with the scheme-consistent diagonal coefficient `gamma` (`ars222`: `1 - 1/sqrt(2)`, `rk443`: `1/2`) instead of hard-coding the ARS222 value.

### Current status after the Galerkin refactor
- `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest tests/test_grid.py -q` → `34 passed`
- `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest tests/test_solver.py -q` → `10 passed`
- This verifies that the new Dirichlet basis path is internally consistent on the unit/regression suite before re-running the blowup comparison cases.

### Important correction inside the Galerkin refactor
- The first Dirichlet-basis attempt used a Moore-Penrose pseudoinverse to map full Chebyshev coefficients back to the reduced basis.
- That was wrong for the Coral-style basis. The stable choice is the exact "unique Chebyshev coefficient" left inverse built from the first `Nz-1` rows of the stencil matrix.
- Symptom of the wrong map: catastrophic growth on the H100 within ~20 steps even at `dt=5e-5`.
- After replacing it with the exact left inverse, the same `Nx=128`, `Nz=128`, `L=10 L_c`, `Ra=100`, `ARS222`, `dt=5e-5` probe returned to a clean linear regime instead of immediate runaway.

### Experimental vertical dealiasing
- Added optional `vertical_dealiasing` to `NHQGConfig`; current options are `"none"` and experimental `"cheb_2x"`.
- `grid.py` now builds an overresolved Chebyshev transform pair `V_dealias` / `V_dealias_inv` when `vertical_dealiasing='cheb_2x'`.
- `solver.py` now has an alternate explicit RHS path that:
  - evaluates `psi, q, w, theta` on a `2*Nz` CGL grid,
  - computes the horizontal Jacobians at that overresolved vertical grid,
  - recovers Chebyshev coefficients on the overresolved grid,
  - truncates back to degree `Nz`,
  - then projects `w,theta` back into the Dirichlet Galerkin basis.
- `scripts/run_miquel_compare.py` now accepts `--vertical-dealiasing`.
- Added a basic solver regression for the `cheb_2x` path; after this addition:
  - `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest tests/test_solver.py -q` → `11 passed`

### Negative result: vertical dealiasing did not fix the blowup
- H100 comparison at `Nx=128`, `Nz=128`, `L=10 L_c`, `Ra=100`, `ARS222`, `dt=5e-5`, `vertical_dealiasing='cheb_2x'` tracked the non-dealiased case essentially identically through the linear phase and the later runaway.
- Final run completed to `t=5.0` in `output/output_miquel_zero_tilt_galerkin_ars222_vdeal2x_Nx128_Nz128_dt5e5_t5`.
- Representative diagnostics:
  - `t=3.0`: `Nu=1.292`, `max_v=2.840`
  - `t=3.5`: `Nu=1352.988`, `max_v=196.573`
  - `t=5.0`: `Nu=3.50e12`
- Conclusion: experimental vertical Chebyshev dealiasing is not the missing stabilization mechanism here.

### Current best status of the `128x128`, `dt=5e-5` control
- The corrected Coral-style Galerkin run without vertical dealiasing completed to `t=5.0` in `output/output_miquel_zero_tilt_galerkin_ars222_Nx128_Nz128_dt5e5_t5`.
- It still ran away, but later than the original `dt=1e-4` case:
  - `t=3.0`: `Nu=1.292`, `max_v=2.840`
  - `t=3.5`: `Nu=1352.988`, `max_v=196.573`
  - `t=5.0`: `Nu=3.50e12`
- So the Dirichlet Galerkin alignment and smaller timestep were necessary cleanup steps, but they did not recover Miquel-style saturation.

## Restart Notes (2026-04-03)

### New spectral-budget diagnostics
- `nhqg/diagnostics.py` now records shell-binned budgets not just for KE, but
  also for `w` variance and `theta` variance.
- The matched comparison archive now stores:
  - `q_vert_spec`, `w_vert_spec`, `th_vert_spec`
  - `q_horiz_spec`, `w_horiz_spec`, `th_horiz_spec`
  - `ke_*` shell tendencies and cumulative flux
  - `w_*` shell tendencies and cumulative flux
  - `th_*` shell tendencies and cumulative flux
- Postprocessing utilities:
  - `scripts/plot_ke_budget_history.py`
  - `scripts/analyze_dominant_shell.py`
  - `scripts/rebuild_spectrum_history_from_checkpoints.py`

### Current best diagnostic archive
- Primary reference run:
  `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_kebudget_blas1_Nx128_Nz128_dt5e5_t8`
- Key outputs:
  - `spectra/spectrum_history.npz`
  - `spectra/ke_budget_plots/`
  - `spectra/dominant_shell/`
- The dominant shell selected from `ke_stretch_shell_tendency` at `t=8.0` is
  `k = 0.9786`.
- [2026-07-04: this dominant-shell result is ghost-contaminated —
  `k ~ 0.98` is the anti-Hermitian ghost's home shell and the shell budgets
  are spectral inner products. See `hermitian_ghost.md`.]

### Current interpretation from the new diagnostics
- The runaway is not being driven by a broadband nonlinear cascade.
- The nonlinear transfer in KE, `w`, and `theta` remains conservative to
  numerical accuracy and modest in amplitude.
- The dominant low-shell source chain is now visible:
  `theta conduction -> buoyancy into w -> q-w coupling into KE`
- Mean-temperature feedback is stabilizing but too weak to close that loop.

### New high-vertical-resolution comparison
- Fresh matched comparison run:
  `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_kebudget_blas1_Nx64_Nz256_dt5e5_t8`
- Same physical setup, but `Nx=64`, `Nz=256`.
- This run shows the same qualitative slow divergence, but stronger:
  - `t=4.5`: `Nu = 7.54e3` vs `6.66e2` in the `128x128` reference
  - `t=5.0`: `Nu = 8.55e5` vs `5.63e4`
  - `t=8.0`: `Nu = 5.12e18` vs `1.68e17`
- The dominant shell is still `k = 0.9786`, so higher vertical resolution did
  not move the instability to a different horizontal scale. It amplified the
  same low-shell chain.

## Restart Notes (2026-04-05)

### Mean-temperature / thermal-exchange audit
- Added new diagnostics in `nhqg/diagnostics.py` and archive writers:
  - `Nusselt_dealiased`, `vol_avg_tw_dealiased`, `heat_flux_mismatch`
  - `th_bar_phys_max`, `dth_bar_dz_max`
  - `mean_grad_min`, `mean_grad_max`, `mean_grad_mid`
  - `mean_energy`
  - `mean_flux_exchange_tendency`
  - `mean_diffusion_tendency`
  - `mean_total_tendency`
  - `th_mean_feedback_sum`
  - `mean_theta_exchange_residual`
  - `mean_theta_exchange_residual_rel`
- New postprocessing:
  - `scripts/plot_mean_exchange_history.py`
  - `scripts/analyze_dominant_shell.py` now includes the mean-exchange table and shell amplitudes
- Tests after the diagnostic additions:
  - `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest tests -q`
  - result: `77 passed`

### Current strongest result
- The old raw heat-flux diagnostic and the solver's dealiased mean-flux path separate catastrophically after about `t≈4.5`.
- In the `128x128` reference archive
  `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_kebudget_blas1_Nx128_Nz128_dt5e5_t8/spectra/spectrum_history.npz`:
  - `t=4.0`: `Nu_raw / Nu_dealiased ≈ 1.08`, exchange residual relative `≈ -9.5e-2`
  - `t=4.5`: `Nu_raw / Nu_dealiased ≈ 1.30e1`, exchange residual relative `≈ -9.43e-1`
  - `t=5.0`: `Nu_raw / Nu_dealiased ≈ 1.22e3`, exchange residual relative `≈ -9.99e-1`
  - `t=8.0`: `Nu_raw ≈ 1.68e17`, `Nu_dealiased ≈ 3.89e1`, `mean_flux_exchange_tendency ≈ 7.81`, `th_mean_feedback_sum ≈ -8.88e16`
- In the `64x256` comparison archive
  `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_kebudget_blas1_Nx64_Nz256_dt5e5_t8/spectra/spectrum_history.npz`:
  - `t=4.0`: `Nu_raw / Nu_dealiased ≈ 2.21`, exchange residual relative `≈ -6.00e-1`
  - `t=4.5`: `Nu_raw / Nu_dealiased ≈ 1.88e2`, exchange residual relative `≈ -9.96e-1`
  - `t=5.0`: `Nu_raw / Nu_dealiased ≈ 2.25e4`, exchange residual relative `≈ -1.00`
  - `t=8.0`: `Nu_raw ≈ 5.12e18`, `Nu_dealiased ≈ 3.70e1`, `mean_flux_exchange_tendency ≈ 7.05`, `th_mean_feedback_sum ≈ -2.35e18`

### Interpretation shift
- The low-shell chain at `k = 0.9786` is still real and still dominant.
  [2026-07-04: no longer accepted — this shell is the ghost's home shell;
  see `hermitian_ghost.md`.]
- But the new result says the runaway seen in the old raw `Nu` and raw fluctuation thermal budget is not tracking the same object as the mean equation's dealiased heat-flux pathway.
- The dominant-shell summaries now show:
  - the shellwise `th_mean_feedback` term continues to explode
  - the mean-reservoir flux exchange term stays `O(10)`
  - `mean_theta_exchange_residual_rel -> -1`
- Current best hypothesis:
  - the main remaining issue is a discrete thermal-exchange mismatch between the fluctuation-side mean-feedback path and the mean equation's dealiased flux path, not generic horizontal cascade physics and not a simple `Nz` deficiency.

### Immediate next work
- Treat the raw `Nusselt` and raw fluctuation-side mean-feedback diagnostics as suspect after `t≈4.5`.
- Prioritize:
  - a corrected fluctuation-side thermal-exchange diagnostic based on the same dealiased `⟨wθ⟩_{xy}` path used by the mean equation
  - a diagnostic-mean solve experiment
  - direct shell-`k≈0.9786` coefficient-level audits only after that consistency fix

### New methodological note: adjoint mean-exchange branch
- The current leading structural fix is documented in
  `adjoint_mean_exchange.md`.
- Main idea:
  - the continuum exchange pair is correct,
  - but the current semi-discrete JAX implementation does not build
    `-w d_Z Theta_bar'` and `-d_Z⟨wθ⟩_{xy}` as an exact discrete adjoint pair,
  - so exact mean/variance closure is not guaranteed.
- The proposed branch is to rebuild the exchange terms from one common
  dealiased work-grid bilinear form and the negative adjoint of that form under
  chosen discrete inner products.
- The recent `+w` explicit/implicit split experiment was a negative result and
  does not invalidate this idea, because `+w` is not the actual exchange pair.

### Follow-up result: dealiased thermal shell budgets also stay bounded
- Added shell-consistent dealiased thermal arrays:
  - `heat_flux_shell_dealiased`
  - `th_conduction_shell_tendency_dealiased`
  - `th_mean_feedback_shell_tendency_dealiased`
  - `w_buoyancy_shell_tendency_dealiased`
  - `mean_theta_exchange_residual_dealiased`
  - `mean_theta_exchange_residual_dealiased_rel`
- These are built from shell-filtered dealiased `⟨wθ⟩_{xy}(z)` profiles, so they use the same product path as the mean equation.
- After this addition:
  - `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest tests -q`
  - result: `78 passed`

At the dominant shell `k = 0.9786`, the contrast is now extremely sharp.

- `128x128` at `t=8.0`
  - raw `th_conduction_shell ≈ 1.56e17`
  - dealiased `th_conduction_shell ≈ 2.45`
  - raw `th_mean_feedback_shell ≈ -8.28e16`
  - dealiased `th_mean_feedback_shell ≈ -1.09`
  - raw `w_buoyancy_shell ≈ 1.56e19`
  - dealiased `w_buoyancy_shell ≈ 2.45e2`
  - raw exchange residual relative `≈ -1`
  - dealiased exchange residual relative `≈ 8.5e-6`

- `64x256` at `t=8.0`
  - raw `th_conduction_shell ≈ 4.88e18`
  - dealiased `th_conduction_shell ≈ 3.02`
  - raw `th_mean_feedback_shell ≈ -2.24e18`
  - dealiased `th_mean_feedback_shell ≈ -1.11`
  - raw `w_buoyancy_shell ≈ 4.88e20`
  - dealiased `w_buoyancy_shell ≈ 3.02e2`
  - raw exchange residual relative `≈ -1`
  - dealiased exchange residual relative `≈ -5.9e-6`

Current best interpretation:
- the apparent late thermal runaway is very likely a raw / non-dealiased diagnostic artifact
- both the global mean-flux path and the shell-consistent dealiased thermal source terms remain bounded
- if there is still a real late-time issue, it is no longer the dramatic `Nu`/thermal blowup we had been reading from the old diagnostics
