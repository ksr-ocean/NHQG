# NHQGE Solver Experiments Log

## Reference: Miquel et al. 2026 (arXiv, fNHQGE on tilted f-plane)

Closest comparison paper. At colatitude ϑ_f=0° (upright), their equations reduce to our
NHQGE exactly. Key parameters from their Table 1:

| Ra_tilde | Resolution       | Nu          | Re_ℓ (=w_rms) | Notes              |
|----------|------------------|-------------|----------------|--------------------|
| 10       | 128²×256         | 1.27±0.01   | 0.75±0.11     | Near onset         |
| 20       | 128²×256         | 4.02±0.13   | 3.55±0.79     | LSV                |
| 40       | 128²×256         | 12.28±0.60  | 10.67±2.43    | LSV                |
| 60       | 128²×256         | 19.88±1.03  | 17.19±4.73    | LSV                |
| 80       | 128²×256         | 30.96±1.81  | 24.28±7.39    | LSV                |
| 100      | 256²×384         | 43.37±2.54  | 32.05±8.24    | LSV                |
| 120      | 256²×384         | 58.84±2.76  | 41.16±11.50   | LSV                |

[Correction 2026-07-04: the table above was re-extracted verbatim from the
theta_f=0 block of `Miquel_NHGQtilted2026_arxiv.pdf` Table 1. The previous
version of this table had different (shifted/stale) values — e.g. it listed
Nu=30.96 at Ra=100, which is actually the Ra=80 entry. Benchmark context: our
stable `balanced_sbp2_pc` runs at Ra=100 give time-mean Nu_d ~ 18-20, i.e.
roughly Miquel's Ra=60 value — the open "Nusselt gap" tracked in `CLAUDE.md`
Current Status (2026-07-04).]

Their numerical method:
- **Coral code** (Miquel 2021) — parallel spectral solver
- **IMEX RK443** (Ascher et al. 1997) — 3rd-order, 4-stage
- **Laplacian dissipation** (ν=1 in non-dim units) on all fields
  - q: ∇'²⊥ q
  - w: ∇'²⊥ w
  - θ: (1/σ) ∇'²⊥ θ
- **Domain**: 10 L_c × 10 L_c × 1, where L_c = 2π/k_c ≈ 4.815
  - Physical domain: 10 × 4.815 ≈ 48.15 non-dim units
- **Resolution criterion**: all scales resolved down to Kolmogorov ℓ_k ~ ε_u^{-1/4}
- Critical parameters: Ra_c ≈ 8.6956, k_c ≈ 1.3048 (matches our linear onset tests)

## Experiment 1: 512²×32, hyper-4, no effective dissipation (FAILED)

**Date**: 2026-03-12
**Parameters**: Nx=512, Nz=32, L=20, Ra=100, σ=1, β=0, Ld=∞
**Dissipation**: hyper-order=4, nu_q = 5*dt/k_max^8 ≈ 1e-18 (essentially zero!)
**Time step**: dt = 5e-4 (fixed)

**Result**: Blow-up at t ≈ 2.5. Exponential growth without bound.

**Root cause**: The dissipation coefficient formula `nu = C*dt/k_max^8` was WRONG —
it gave exp(-7e-7) ≈ 1.0 at k_max (zero dissipation). The enstrophy forward cascade
had no energy sink, so small-scale energy accumulated until CFL was violated.

## Experiment 2: 512²×32, hyper-4, physical dissipation (FAILED)

**Dissipation**: hyper-order=4, nu = 1/(tau_d * k_max^8), tau_d = 5e-3
**Time step**: adaptive CFL (target 0.4), check every 200 steps

**Result**: Blow-up at t ≈ 2.5. Same exponential growth pattern.

**Root cause**: Hyper-4 dissipation (|k|^8) has essentially zero effect at the most unstable
wavenumber k_c ≈ 1.3 because (k_c/k_max)^8 ≈ 10^{-15}. The flow grows exponentially at
the convective scale with NO damping at that scale. The Jacobian nonlinearity transfers
energy to other modes, but the growing mode is never damped. Eventually CFL is exceeded.

**Key insight**: Hyper-4 only damps small scales. The NHQGE linear instability at k_c
needs Laplacian dissipation at the convective scale to set the proper amplitude through
the balance of growth vs dissipation.

## Experiment 3: 512²×32, Laplacian ν=1 (COMPLETED — too energetic)

**Date**: 2026-03-12
**Parameters**: Nx=512, Nz=32, L=20, Ra=100, σ=1, β=0, Ld=∞
**Dissipation**: Laplacian (hyper_order=1), nu_q = nu_w = nu_theta = 1.0
**Time step**: adaptive CFL (target 0.3), check every 50 steps

**Final result**: Reached quasi-equilibrium around t≈3.16, then NaN at t=3.19 (step 11600).
- Quasi-steady values (t ≈ 3.15-3.19): KE_bt ≈ 960, KE_bc ≈ 14,000
- Nu ≈ 4,300 (was slowly DECREASING: 4435→4268 over last 1000 steps before spike)
- max_v ≈ 630, dt adapted to 1.77e-5 through 13 rebuilds
- Blow-up: sudden velocity spike 631→783→NaN in 3 diagnostic steps (likely CFL)

**Growth pattern** (correctly normalized with Parseval norm = Nx⁴):

| t     | KE_bc     | KE_bt    | Nusselt  | max_v  | dt       | Phase           |
|-------|-----------|----------|----------|--------|----------|-----------------|
| 1.0   | 5e-12     | 2e-17    | 1.00     | 0.0    | 5.0e-4   | linear growth   |
| 2.0   | 6e-5      | 3e-11    | 1.00     | 0.1    | 5.0e-4   | linear growth   |
| 2.5   | 0.16      | 5e-6     | 1.17     | 1.7    | 5.0e-4   | nonlin onset    |
| 2.75  | 10.4      | 0.02     | 12.4     | 14.3   | 5.0e-4   | rapid growth    |
| 2.85  | 56        | 0.48     | 62.5     | 33.4   | 5.0e-4   | rapid growth    |
| 3.0   | 863       | 36       | 831      | 129    | 9.5e-5   | CFL adapting    |
| 3.07  | 3,034     | 132      | 2,333    | 220    | 5.9e-5   | slowing growth  |
| 3.14  | 10,581    | 504      | 4,332    | 470    | 2.8e-5   | near plateau    |
| 3.16  | 13,000    | 960      | 4,270    | 630    | 1.8e-5   | quasi-steady    |
| 3.19  | NaN       | NaN      | NaN      | NaN    | 1.8e-5   | blow-up         |

**Observations**:
1. Linear growth phase (t=0 to ~2.5): amplitude growth rate ≈ 7.8, matches theoretical
   σ_net = σ_linear - ν·k_c² = 9.6 - 1.7 = 7.9. Growth is correct.
2. Transition to nonlinear (t ≈ 2.7-2.8): eddy turnover time ≈ convective growth time.
3. Flow approached a quasi-equilibrium at Nu ≈ 4,300 (was slowly decreasing).
4. All diagnostic values (Nu, KE, Re) are 1-2 orders of magnitude LARGER than Miquel et al.
5. The final blow-up was a sudden CFL spike, not a gradual instability.

## Comparison: Our results vs Miquel et al.

| Quantity          | Miquel (Ra=100) | Our Exp 3 (quasi-eq) | Ratio |
|-------------------|-----------------|----------------------|-------|
| Nusselt           | 31 ± 2          | ~4,300               | ~140× |
| w_rms (Re_ℓ)     | 24 ± 7          | ~130 (est. from max) | ~5×   |
| max velocity      | ~50 (est.)      | ~630                 | ~13×  |
| KE_bc             | ~300 (est.)     | ~14,000              | ~47×  |
| Vertical res Nz   | 256             | 32                   | 8×    |
| Horizontal res    | 128             | 512                  | 4× more |
| Domain (in L_c)   | 10              | ~4.15                | 2.4×  |
| IMEX scheme       | RK443 (3rd ord) | ARS222 (2nd ord)     | —     |

## Equation verification

Checked our solver equations against Miquel et al. (3.1a-c) at ϑ_f=0°:

**Vorticity** (3.1a): ∂_t q' + J[ψ,q'] = +D_Z w + ∇²q'
  → Our solver line 106: `E_q = -Jq - iβ·kx·ψ`, implicit: `I_q = +D_Z w`. ✓

**Vertical velocity** (3.1b): ∂_t w + J[ψ,w] = -D_Z ψ + (Ra/σ)θ + ∇²w
  → Our solver line 107: `E_w = -Jw + Ra_sigma·θ`, implicit: `I_w = +c(k)·D_Z q'`. ✓
  (since -D_Z ψ = c(k)·D_Z q' where c(k)=1/(|k|²+Ld⁻²))

**Temperature** (3.1c): ∂_t θ + J[ψ,θ] + w(∂_η Θ̄ - 1) = (1/σ)∇²θ
  → Our solver line 108: `E_th = -Jth + w_hat` (assumes ∂_η Θ̄ = 0, so forcing = +w). ✓

**Key difference with Miquel**: They evolve the mean temperature Θ̄ via (3.1d). We hold
Θ̄ = 0 fixed. In the turbulent interior, ∂_η Θ̄ ≈ 0 (isothermal), so forcing ≈ +w — same
as ours. Near boundaries ∂_η Θ̄ is steep, but w→0 (Dirichlet BC) so the product w·(1-∂_η Θ̄)
is still small. This difference is unlikely to explain the 140× Nu discrepancy.

## Hypothesis: Vertical under-resolution (Nz=32 vs 256)

**The primary suspect is Nz=32 being far too coarse.**

Miquel uses Nz=256-384 Chebyshev modes. We use Nz=32.

### Why low Nz produces too much energy (NOT instability)

The NHQGE has **no explicit vertical dissipation** — all dissipation is through the horizontal
Laplacian ∇²⊥. The vertical coupling (D_Z w → q', D_Z ψ → w) generates progressively
finer vertical structure at each time step. The energy pathway is:

1. Convective instability at k_c pumps energy into the leading vertical mode sin(πZ)
2. Nonlinear interactions (Jacobians) and the D_Z coupling transfer energy to higher
   vertical harmonics sin(nπZ) for n > 1
3. Higher vertical harmonics of w couple to DIFFERENT horizontal wavenumbers of q'
   through the D_Z operator, projecting energy to higher horizontal k
4. The horizontal Laplacian ∇²⊥ then dissipates this energy (rate = ν|k_h|²)

With Nz=32 (~16 resolved vertical harmonics):
- Step 3 above is truncated: the vertical cascade reaches Nz/2 and has nowhere to go
- Energy that SHOULD cascade to fine vertical scales and then be dissipated horizontally
  instead reflects back into the resolved modes (spectral aliasing)
- The resolved modes become artificially energetic
- Higher energy → higher velocities → higher heat transport → Nu ≫ expected

**This is NOT a numerical instability.** The simulation reached quasi-equilibrium at Nu≈4300
(was slowly decreasing). The eventual blow-up was a CFL spike, not a fundamental instability.
With more aggressive CFL control, the simulation would run indefinitely — just at the
WRONG energy level. Higher Nz should REDUCE the energy by providing more vertical modes
for the cascade, leading to more horizontal dissipation.

### Supporting evidence

1. Miquel uses Nz=256 for Ra=100 — they clearly need many vertical modes in the turbulent
   regime. Their resolution criterion: "all scales down to Kolmogorov ℓ_k ~ ε_u^{-1/4}
   are resolved."

2. Our simulation equilibrated at Nu≈4300 (not growing without bound). This means the
   resolved dynamics ARE reaching a steady state — just the wrong one.

3. The flow is 11.5× supercritical (Ra/Ra_c = 100/8.7), so strongly turbulent with
   significant vertical structure beyond just sin(πZ).

4. From Miquel Fig 16: the mean temperature profile has thin boundary layers (~1/2Nu ≈ 0.016
   thick). Our CGL grid resolves these (finest spacing ~π/Nz² ≈ 0.003), but the FLUCTUATION
   fields θ, w have even finer vertical structure.

### Counter-arguments and alternative hypotheses

**"Nz=32 should be enough for Chebyshev"**: True for SMOOTH functions (exponential convergence),
but turbulent fields are not smooth in Z — they have fine structure from the vertical cascade.
Chebyshev convergence rate depends on the regularity of the function, not just the polynomial
degree.

**Domain size effect**: Our L=20 ≈ 4.15 L_c vs Miquel's 10 L_c. Smaller domain constrains the
inverse cascade and forces the condensate to the box scale. But this primarily affects the
KE_bt/KE_bc split, not the TOTAL energy or Nu. Not a 140× effect.

**Mean temperature evolution (Θ̄)**: We don't evolve Θ̄ while Miquel does. But in the turbulent
interior, ∂_Z Θ̄ ≈ 0 (isothermal), giving the same forcing +w as our code. Near the
boundaries, w→0 (Dirichlet BC). This is at most a modest quantitative correction, not 140×.

**IMEX order**: Our ARS222 (2nd order) vs their RK443 (3rd order). Lower order adds phase
errors but doesn't change the equilibrium energy budget. Not a 140× effect.

## Next Steps

### Immediate: Increase Nz
Run at Nz=64 and Nz=128 with same horizontal parameters. Check convergence of Nu toward
Miquel's value. If Nu decreases significantly with Nz, the hypothesis is confirmed.

### Also investigate:
1. **Domain size**: Test L=48 (~10 L_c) to match Miquel exactly
2. **Resolution**: Consider dropping to Nx=256 (matching Miquel) to save memory for larger Nz
3. **Normalization double-check**: Run a simple case (e.g., Ra=10, near onset) and compare
   Nu with Miquel's value (1.50). If our Nu matches at Ra=10, the normalization is correct.
4. **Dissipation budget**: Compute ε_u = ν<|∇'⊥ q|²> and verify energy balance
   Nu - 1 = (σ²/Ra) * ε_u at saturation.

### Memory implications of larger Nz
With Nz=128, state size (3 fields × (129 × 512 × 257) × 16 bytes) ≈ 2.4 GB.
IMEX matrices: ~512 shells × 129² × 8 bytes ≈ 69 MB.
Should fit on V100 32GB.

With Nz=256, state ≈ 4.8 GB, IMEX ≈ 275 MB. Still fits on V100 but tight for working memory.

## Bug fixes applied (this session)

1. **Parseval normalization**: `norm = Nx**2` → `norm = Nx**4` in diagnostics.py.
   Factor: Nx² from DFT Parseval + Nx² for spatial average = Nx⁴ total.

2. **max_speed diagnostic**: Changed from spectral upper bound (grossly overestimated by
   ~4000×) to physical-space velocity via irfft2.

3. **Dissipation on w/θ**: Changed from Laplacian (ksq^1) to matching hyper_order for all
   fields in grid.py.

4. **Dissipation coefficient formula**: Corrected from `nu = C*dt/k_max^8` (gave ≈0
   dissipation) to `nu = C/(tau_d * k_max^8)` (physical rate independent of dt).
