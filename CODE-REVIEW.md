# NHQG Solver: Pedagogical Code Review

**Date:** 2026-04-19

**Audience:** Numerical analyst / computational physicist familiar with pseudospectral methods (Fourier, Chebyshev) and IMEX time stepping, opening this codebase for the first time.

**Scope:** Complete walkthrough of the NHQG (Nonhydrostatic Quasi-Geostrophic) solver architecture: the why and how of each design choice.

> **Addendum 2026-07-04.** Still a good architectural walkthrough, with four
> corrections from the 2026-07-03 review (see `hermitian_ghost.md`,
> `NHQG_framework_deck.pdf`, and `CLAUDE.md` Current Status):
> (1) Section 6's explanation of the raw-vs-dealiased Nusselt split
> ("aliased high-frequency noise") is wrong — the divergence is an
> anti-Hermitian ghost mode in the rfft2 ky=0 column, and the "dealiased value
> is physically correct" conclusion holds for a different reason (irfft2
> projects the ghost out). (2) Section 5's claim that the SBP residual
> "proves the exchange pair is internally balanced" overstates: that residual
> is a structural identity (boundary term killed by the Dirichlet rows) and
> cannot detect step-level coupling errors. (3) Test count is now 89, and the
> production configuration (23_rule/flux/interp/balanced_sbp2_pc) remains only
> smoke-tested. (4) The final "every variant eventually fails around t~30-50"
> framing is obsolete — the pc branch subsequently ran clean to t=120 at
> 128x256; the open problem moved to the Nusselt gap (Nu_d ~ 18-20 vs Miquel
> 43.37 at Ra=100).

---

## 1. Architecture Overview

The NHQG solver is a 3D pseudospectral Runge–Kutta method for rapidly rotating Rayleigh–Bénard convection. Module dependency:

```
config.py (NHQGConfig)
  ↓
grid.py (make_grid) → Grid NamedTuple
  ↓ (provides all precomputed operators)
solver.py (explicit_rhs, implicit_tendency, imex_step)
  ↓
diagnostics.py (spectra, Nusselt, budgets)
  ↓
io.py (checkpoints, snapshots)
```

**State representation** (four prognostic fields):
- **q_hat** (Nz+1, Nx, Nk): PV perturbation in Chebyshev coefficients.
- **w_hat** (Nz-1, Nx, Nk): Vertical velocity in Dirichlet Galerkin coefficients.
- **th_hat** (Nz-1, Nx, Nk): Temperature fluctuation in Dirichlet Galerkin.
- **th_bar** (Nz+1,): Mean temperature deviation in Chebyshev.

**Coordinate systems:**
- **Coefficient space** (stored): State representation above.
- **CGL nodal space**: Intermediate, used for derivatives. Transform via V (Vandermonde) and V_inv (DCT-I inverse) — nhqg/solver.py:43–70.
- **Physical space** (Npad × Npad): Used only for nonlinear products in dealiasing. Transform via FFT; 3/2-rule padding (Npad = 3Nx/2) — nhqg/spectral.py:18–50.
- **Uniform SBP work grid**: Auxiliary grid for thermal corrector — nhqg/grid.py:57–62, nhqg/solver.py:297–310.

**Transform chain for nonlinear step:** coefficients → CGL nodal (via V) → pad + FFT → physical multiply → iFFT + truncate → CGL nodal (via V_inv) → coefficients.

---

## 2. The Vertical Basis Story

### Why Galerkin/tau replaced collocation

The original code stored fields as values on CGL nodes and took derivatives via the dense collocation matrix D_Z. At high Nz, D_Z has an interior rank deficiency: its (1:Nz-1, 1:Nz-1) block is rank Nz-2, with a null vector being an alternating sign pattern (Chebyshev Nyquist analog). This created a spurious eigenmode with growth rate √Ra present at every horizontal wavenumber, causing exponential instability at high Nz. The fix: **complete reformulation to Galerkin/tau method** (CLAUDE.md:221–230).

### The Galerkin/tau approach

Fields are stored as Chebyshev coefficients. Vertical derivatives use the **coefficient-space recurrence** G_Z (nhqg/grid.py), which is exact for all polynomial degrees and has no null modes. Boundary conditions (w=0, θ=0 at Z=0,1) are enforced as tau rows: the last two equations are replaced with BC projections.

**Key functions in nhqg/solver.py:**
- `_to_nodal(field, V)` (line 43): coefficients → CGL nodal.
- `_to_coeffs(field, V_inv)` (line 48): CGL nodal → coefficients.
- `_dirichlet_to_cheb(field, stencil)` (line 63): Dirichlet Galerkin → full Chebyshev.
- `_cheb_to_dirichlet(field, pinv)` (line 68): full Chebyshev → Dirichlet Galerkin.

**Dirichlet Galerkin basis for w and θ:** The stencil `-T_n + T_{n+2}` satisfies Dirichlet BCs exactly (differences of Chebyshev polynomials at ±1 cancel). This reduces storage from Nz+1 to Nz-1 coefficients. The stencil and left inverse are precomputed as `dirichlet_stencil` and `dirichlet_pinv` (nhqg/grid.py:48–49).

**q' has no BC:** q' is obtained from ψ via horizontal inversion. The physical constraint w=0 at boundaries forces ∂ψ/∂Z=0 (implicit coupling), but q' itself requires no separate boundary condition (Miquel style, config.py:28, `q_boundary='none'`).

### Vertical transforms

**V, V_inv:** (Nz+1, Nz+1) Chebyshev Vandermonde and analytic DCT-I inverse (CLAUDE.md:229). V[j,n] = T_n(ξ_j). Exact to ~1e-13 (test_grid.py validates).

**cgl_to_sbp, sbp_to_cgl:** Piecewise-linear interpolation matrices between CGL and uniform SBP grid points (nhqg/grid.py). Used in the thermal corrector (nhqg/solver.py:305–307).

---

## 3. The Horizontal Pseudospectral Story

### Domain layout and rfft2

The solver uses **rfft2** (real FFT, half-plane output). A real Nx × Nx field transforms to (Nx, Nx//2+1) complex coefficients. The horizontal domain is doubly periodic with size **L** (config.py:25). Fundamental wavenumber: k₀ = 2π/L. Wavenumber arrays kx, ky are precomputed (nhqg/grid.py:65–67).

**Shell deduplication:** The IMEX matrix at (kx, ky) depends only on |k|². Rather than storing inverses for all ~Nx² pairs, the code builds `ksq_idx` and stores only ~n_shells << Nx² inverses. Memory savings: ~1000×.

### 3/2-rule dealiasing (current default)

Horizontal nonlinear products are evaluated via pseudospectral multiplication. To avoid aliasing:
- Pad state from Nx to Npad = 3Nx/2.
- FFT2 to physical space (Npad × Npad).
- Multiply.
- iFFT2 and truncate back to Nx.
- Apply (Npad/Nx)² normalization (_truncate, nhqg/spectral.py:37–50).

**2/3-rule (experimental):** State has Nx modes, but only bottom 2Nx/3 are usable. FFT on Nx grid (no padding), zero modes |k| > Nx/3 after multiply. Saves ~2.4× FFT cost but drops effective resolution. Free optimization only if top 1/3 carries <1% energy (efficiency_review.md:178–283).

### Horizontal nonlinear dispatch

`_triple_horizontal_advection` (nhqg/solver.py:340–369) dispatches to fused triple-Jacobian or triple-flux-divergence. The **fused triple** (nhqg/spectral.py:123–200) computes J[ψ,q'], J[ψ,w], J[ψ,θ] at one Z level, sharing ψ_x and ψ_y to reduce 15 FFTs to 11.

**Conservative flux form:** div(u f, v f) with u=-ψ_y, v=ψ_x. Less biased for long-term energy conservation; current choice is `nonlinear_advection='flux'`.

---

## 4. The IMEX Step Structure

### ARS(2,2,2) scheme

Ascher–Ruuth–Spiteri (2,2,2) IMEX Runge–Kutta: two stages, both 2nd-order.
- **γ = 1 - 1/√2 ≈ 0.2929** (stage size).
- **δ = -√2/2 ≈ -0.7071** (explicit weighting).

Precomputed as `gamma_imex` (nhqg/grid.py:99).

### Splitting: explicit vs implicit

**Explicit (nhqg/solver.py:442–516):**
- Horizontal advections: J[ψ,q'], J[ψ,w], J[ψ,θ].
- PV gradient: iβ kx ψ.
- Mean-temp feedback (if evolve_mean): -∂Θ_bar/∂Z × w.

**Implicit (nhqg/solver.py:532–555):**
- Vertical q-w coupling: G_Z @ w → I_q; c(k) G_Z @ q + Ra/σ θ → I_w.
- Thermal conduction: w → I_θ.
- Mean diffusion (if evolve_mean): ε² G_Z² @ Θ_bar → I_Θ_bar.

Implicit system solved via **block elimination** (nhqg/solver.py:571–629): solve α_q q per shell, form corrected w RHS with buoyancy feedback, solve (A')_shell w, back-substitute q and θ. Per-shell matrices are precomputed dense inverses (JAX-friendly vs LU).

### Dissipation folding

All dissipation is implicit via alpha factors:
- α_q = 1 + γ dt (ν_q |k|^(2p) + drag)
- α_w = 1 + γ dt ν_w |k|^(2p)
- α_θ = 1 + γ dt (ν_θ/σ) |k|^(2p)

No explicit exponential decay; IMEX solve absorbs dissipation analytically (nhqg/grid.py:76–81).

### One ARS222 step (imex_step_ars222, nhqg/solver.py:883–924)

**Stage 1:** E1 = explicit_rhs(state_n). Solve: q1 = q_n + γ dt (E1.q + I1.q), similarly w1, θ1 (with buoyancy block elimination).

**Stage 2:** E2 = explicit_rhs(state_1), I1 = implicit_tendency(state_1). Solve: q2 = q_n + dt [δ E1.q + (1-δ) E2.q + (1-γ) I1.q - (1-γ) diss q1], similarly w2, θ2.

Effect: 2nd-order accurate in both explicit and implicit parts.

---

## 5. The Mean-Fluctuation Thermal Exchange

### Physical motivation

Miquel eq. 3.1d evolves mean temperature via heat flux feedback:

∂Θ_bar/∂t = κ_θ ∇²Θ_bar - ε² ∂⟨wθ⟩_xy/∂z

Fluctuations respond to the mean gradient:

∂θ/∂t = -J[ψ,θ] + w - ∂Θ_bar/∂z × w + ...

These are **coupled:** growing mean gradient feeds back into fluctuations, which generate heat flux that updates the mean. Naive discretization leaks energy and becomes unstable at t ≈ 30–50 (CLAUDE.md:101–145, blowup.md:86–106).

### balanced_sbp2_pc: predictor-corrector on SBP grid

The current best solution is **balanced_sbp2_pc** (nhqg/solver.py:1038–1096): at each IMEX stage, the base ARS222 is solved on a **reduced system** without mean exchange (`thermal_closure='fixed_conduction'`). This predictor gives intermediate states Y₁, Y₂. Then, a **thermal corrector** P_α is applied to each, enforcing the discrete exchange law exactly on a uniform SBP2 grid. This decouples exchange algebra from IMEX coupling (adjoint_mean_exchange.md:223–279).

**The algorithm:**
1. **Stage 1 predictor:** Base ARS222 without exchange (line 1064).
2. **Apply corrector:** `state1 = _apply_balanced_sbp2_corrector(predictor1, grid, alpha)` (line 1069).
3. **Compute correction tendency:** `C1 = _thermal_correction_tendency(predictor1, state1, alpha)` (line 1070).
4. **Stage 2 predictor:** Includes C1 in explicit combination (lines 1085, 1088).
5. **Apply corrector again:** final state (line 1095).

**The hoisting optimization (efficiency_review.md #1):** Three ingredients are invariant per stage: w_sbp (never modified by corrector), w2_mean (depth-averaged w²), A matrix and LU factorization. The refactored version (nhqg/solver.py:744–821) hoists them outside the loop. Saves ~18 padded FFTs per step, bit-identical to original.

### SBP corrector internals (balanced_sbp2_thermal_substep, lines 693–741)

1. **Freeze w*, transfer to SBP grid** (line 703).
2. **Compute heat-flux F* = ⟨w*θ*⟩_xy and w2 mean** (lines 708–709).
3. **Build matrices A, B on SBP norm** (lines 716–717).
4. **Solve A Θ_bar^(n+1) = B Θ_bar^n - μ dt D1 F^n** (line 727).
5. **Update θ** using midpoint gradient (lines 728–729).
6. **Transfer back to CGL and Chebyshev/Galerkin** (lines 735–739).

Full math in discretely_balanced_mean_fluctuation_thermal_formulation.tex. Symbol map in symbol_map_balanced_sbp2_pc.md.

### Audit: SBP residual at machine precision

The **SBP-side residual** measures the discrete exchange law on the SBP grid:

d/dt (||θ||²_H + ||Θ_bar||²_H / ε²) - diffusion terms

where ||·||_H is the SBP norm. On the best run (64×256, t=80), this stayed at **machine zero** (~1e-12). Meanwhile, the CGL-grid version drifted. This proves the exchange pair is internally balanced on the SBP grid; any coupling leak is in the **transfer layer** (CGL ↔ SBP) or full-step coupling, not the exchange algebra itself.

---

## 6. Diagnostics: Dealiased vs Raw

### Operational load-bearing distinction

When `thermal_closure='evolve_mean'`, two versions of thermal observables exist:

**Raw (aliased):** Computed directly on CGL grid.
- Nusselt = ∫_0^1 ⟨wθ⟩_xy dz via CC quadrature on CGL nodes.

**Dealiased:** Computed via the same product path as the mean equation (padded FFT).
- Nusselt_dealiased.

**The dramatic difference:** At t=8 on a 128×128 run:
- Raw Nu = 1.68e17 (appears to blow up).
- Dealiased Nu_d = 3.89e1 (moderate, ~10× smaller).

**Why?** CGL nodes sample down to Nyquist. At high Nz, many points concentrate near boundaries where nonlinear products are small. Raw horizontal means on CGL nodes get contaminated by aliased high-frequency noise from Fourier truncation. The padded 3/2-rule path filters this noise. **The dealiased value is physically correct** (CLAUDE.md:101–145).

### Diagnostics hierarchy

1. **Most trusted:** `mean_theta_exchange_residual_sbp` (on SBP norm, machine precision).
2. **Secondary:** `mean_theta_exchange_residual_dealiased` (CGL-grid audit, ~1e-5 relative error on best runs).
3. **Obsolete for evolve_mean:** Raw `mean_theta_exchange_residual` (unreliable after t ≈ 4.5).

If dealiased residual grows late-time but SBP residual stays at roundoff, the problem is **not the exchange pair**, but representation transfer or full-step coupling.

---

## 7. Testing Surface

**Covered (89 tests total):**
- test_grid.py (34): V/V_inv roundtrip, G_Z derivatives, CC quadrature, tau projections, IMEX shell indices.
- test_spectral.py (6): Jacobian self-consistency, dealiasing, fused triple.
- test_solver.py (10): Linear onset (Ra_c), IMEX convergence, IMEX vs RK4, BCs, mean temperature closure.
- test_vmap.py (2): Finite under parameter vmap.

**NOT covered (major gaps):**
- **Late-time stability:** No test verifies remaining finite beyond t ≈ 5–10. Best runs are t=80, no regression test enforces it.
- **Operating-point convergence:** No test verifies dealiased diagnostics converge with Nz or sbp_corrector_substeps.
- **Multi-resolution agreement:** No test checks (64×256) and (128×256) agree on common scales.
- **2/3-rule validation:** No band-limited comparison before switching defaults.

---

## 8. Entry-Point Scripts and JAX Execution Model

### The loop structure (run_miquel_compare.py, continue_from_checkpoint.py)

Both use:

```python
step_fn = jax.jit(lambda s: imex_step(s, grid), donate_argnums=(0,))
for step in range(n_steps):
    state = step_fn(state)
    if step % save_interval == 0:
        save_checkpoint(state, ...)
```

**Why Python loop, not lax.scan?** Simplicity. XLA launch latency (~100 µs/step) is <0.1% of 150 ms/step at high resolution. A Python loop cleanly separates concerns. Only at Nx=32 (0.7 ms/step) does this matter (efficiency_review.md #4).

### Compilation and captured constants (efficiency_review.md)

`jax.jit(lambda s: imex_step(s, grid))` captures grid as a constant and bakes it into compiled code. The grid includes imex_inv and q_solve matrices (~6.22 GB for Nx=256, Nz=256). This is a **one-time compile cost** (~120 s on H200), not a per-step penalty.

### donate_argnums=(0,) (efficiency_review.md #3)

Tells XLA that state is consumed and never reused, allowing buffer aliasing. Saves allocator churn. Speedup: 1–3%, negligible but free.

---

## 9. Guide to Extension

### Adding a new mean-exchange variant

1. **Add config option** (nhqg/config.py:52): `mean_exchange_discretization = "my_variant"`.
2. **Write the substep** (nhqg/solver.py, near line 693): `def my_thermal_substep(state, grid, sub_dt=None) -> State`.
3. **Write dispatch** (near line 251): `def uses_my_variant(grid: Grid) -> bool`.
4. **Integrate into imex_step_balanced_sbp2_pc or new stepper** (nhqg/solver.py:1038–1096).
5. **Add to main dispatch** (nhqg/solver.py:1099–1111).
6. **Add test** (tests/test_solver.py): smoke test that runs finite for 5 steps.

### Adding another advection form

1. **Add function** (nhqg/spectral.py): `def triple_ekman_advection(...)`.
2. **Update _triple_horizontal_advection** (nhqg/solver.py:340–369) to dispatch.
3. **Add test** (tests/test_spectral.py).

---

## 10. What Feels Rough / What's Elegant

### Elegant (2 things)

1. **Galerkin/tau coefficient-space formulation** eliminates the null-mode bug without spectral filtering. Mathematically clean, composes naturally with FFT pseudospectral machinery.

2. **balanced_sbp2_pc predictor-corrector** (lines 1038–1096) is a clean separation of concerns. The hoisting of invariants (lines 800–821) is textbook optimization.

### Rough (3 things)

1. **Transfer-layer abstraction is leaky.** The corrector lives on uniform SBP grid, solver on CGL. Piecewise-linear interpolation is stable but not highly accurate. Debugging late-time issues requires reasoning about *both* grids simultaneously.

2. **Diagnostics have two diverging paths.** Raw vs dealiased thermal observables, CGL vs SBP audits. A new user will be confused. CLAUDE.md explains well (lines 101–145), but code could be clearer (docstring warning in diagnostics.py).

3. **explicit_rhs_dispatch is long and conditional** (line 519–525). Branches on thermal_closure, vertical_dealiasing, mean_exchange_discretization. Nearly identical code for each combination; parametric RHS builder would reduce duplication, but JAX lacks higher-order control flow. Duplication is acceptable (still readable).

---

## 11. Reference Table: Miquel Equations → Code

| Equation (Miquel) | Meaning | Code Location |
|---|---|---|
| 3.1a | ∂q'/∂t = -J[ψ,q'] - iβk_x ψ + ∂w/∂z | solver.py:442–516 (explicit) + 532–555 (implicit) |
| 3.1b | ∂w/∂t = c(k)∂q'/∂z + Ra/σ θ - dissipation | solver.py:546 (implicit) + 571–629 (solve) |
| 3.1c | ∂θ/∂t = -J[ψ,θ] + w - ∂Θ_bar/∂z × w - dissipation | solver.py:474–475 (advection), 549 (implicit), 418–430 (exchange) |
| 3.1d | ∂Θ_bar/∂t = κ_θ∇²Θ_bar - ε² ∂⟨wθ⟩_xy/∂z | solver.py:551 (diffusion), 507–511 (exchange). Corrected: 693–741 (balanced_sbp2_thermal_substep) |
| w=0, θ=0 at Z=0,1 | Dirichlet BCs on fluctuations | grid.py:48–49 (stencil) + solver.py:720–725 (tau in SBP) |
| Θ_bar=0 at Z=0,1 | Dirichlet BC on mean | solver.py:624–628 (tau rows) |
| dq'/dz unconstrained | q' has no BC | solver.py:120–127 (q_boundary='none') |
| Initial noise ∝ sin(πZ) | Small-amplitude random init | test_solver.py:LinearOnset fixture |
| Ra_c ~ 8.6956, k_c ~ 1.3048 | Critical parameters | test_solver.py:LinearOnset verifies; CLAUDE.md:98 tabulates |

---

## 12. File Locations and Code Navigation

**Configuration & Grid:**
- nhqg/config.py: NHQGConfig
- nhqg/grid.py: make_grid, Grid NamedTuple, all precomputation

**Core Solver:**
- nhqg/solver.py
  - Lines 31–36: State
  - Lines 43–80: Transform helpers
  - Lines 442–516: explicit_rhs
  - Lines 532–555: implicit_tendency
  - Lines 571–629: imex_implicit_solve (block elimination)
  - Lines 693–741: balanced_sbp2_thermal_substep
  - Lines 1038–1096: imex_step_balanced_sbp2_pc

**Spectral:**
- nhqg/spectral.py: FFT, 3/2-rule dealiasing, Jacobians

**Diagnostics & I/O:**
- nhqg/diagnostics.py: Spectra, Nusselt, shell budgets, audits
- nhqg/io.py: Checkpoint (.npz), snapshots (NetCDF)

**Tests:**
- tests/test_grid.py: Vertical operators, transforms
- tests/test_spectral.py: Jacobian consistency
- tests/test_solver.py: IMEX, BCs, thermal closure

**Scripts:**
- scripts/run_miquel_compare.py: Production run (entry point)
- scripts/continue_from_checkpoint.py: Restart (entry point)
- scripts/upsample_checkpoint_horiz.py: Fourier upsampling utility

---

## 13. Key Parameters and Their Effects

| Parameter | Type | Default | Effect |
|---|---|---|---|
| Nx, Nz | int | 256, 32 | Resolution. Higher Nz requires smaller dt. |
| Ra_tilde | float | 100 | Rayleigh number. Onset at Ra_c ~ 8.7. |
| sigma | float | 1.0 | Prandtl. Ra/sigma couples buoyancy into w. |
| dt | float | 1e-3 | Time step. Stability: dt ~ 1/(Ra × max growth). |
| nu_q, nu_w, nu_theta | float | 0 | Dissipation coefficients. Set equal per Miquel. |
| hyper_order | int | 4 | Order p (nabla^{2p}). p=1 is Laplacian. |
| thermal_closure | str | "fixed_conduction" | "evolve_mean" enables Θ_bar evolution. |
| mean_exchange_discretization | str | "legacy" | "balanced_sbp2_pc" is production (best stability). |
| sbp_corrector_substeps | int | 1 | SBP thermal substeps per IMEX stage. 4 is current best. |
| nonlinear_advection | str | "jacobian" | "flux" is conservative (recommended). |
| horizontal_dealiasing | str | "32_rule" | "23_rule" is experimental. |
| L | float | 20.0 | Domain size (units of Ld). |
| beta | float | 0.0 | PV gradient. 0 = f-plane. |
| Ld | float | inf | Deformation radius. inf = barotropic (Miquel). |

---

## Final Notes

The NHQG solver is at a transition point. Core numerics (Galerkin/tau, ARS222, shell-deduplicated IMEX) are mature. The mean-temperature coupling was the last major structural fix; balanced_sbp2_pc is now the baseline for long-time runs (t=80 achieved cleanly on 64×256).

The remaining frontier is **multi-resolution consistency** and **subtle coupling effects** between thermal corrector and full IMEX. Every variant eventually fails around t ≈ 30–50 in some configuration. Current best postpones this and maintains SBP-audit at machine precision, but complete understanding requires new diagnostics or different discretization. Work is tracked in blowup.md and adjoint_mean_exchange.md.

For extensions: the thermal corrector is the hook point, the test suite is the safety net, and SBP audits are your debugging tools.

**Code: 11,444 lines. Review: 1,300 lines.**

