# High-Nz Late-Time Failure: Current State

Date: 2026-04-11

## RESOLVED (banner added 2026-07-04)

This note is now a HISTORICAL record. Everything below the banner reflects the
2026-04-11 state of knowledge. What happened since:

- The late-time failure is resolved operationally: the
  `balanced_sbp2_pc` + `sbp_corrector_substeps=4` + `flux` branch ran clean to
  `t = 80` at `64x256`, then (2026-04-19/20, 2/3-rule) to **`t = 120`** at
  `128x256` and `t = 63` at `256x256`. See `CLAUDE.md` "Current Status
  (2026-07-04)".
- The raw-diagnostics "blowup" that opens this note (raw `Nu = 1.68e17` etc.)
  has a complete mechanism now: an **anti-Hermitian ghost mode in the rfft2
  ky=0 column**, invisible to the physics, growing at the unsaturated linear
  rate. It is not classical aliasing. See `hermitian_ghost.md`.
- The `k ~ 0.9786` dominant-shell reading cited below is ghost-contaminated
  (that is the ghost's home shell); do not carry it forward without
  re-deriving from Hermitian-projected states.
- The open problem has moved from stability to accuracy: time-mean
  `Nu_d ~ 18-20` at `Ra=100` vs Miquel's `43.37 +/- 2.54` (suspects: the
  piecewise-linear CGL<->SBP transfer smoothing the thermal boundary layer;
  the untruncated 2/3-rule band at Nx=64; effective 2/3-rule resolution).

Historical record follows.

---

This note is the short operational record for the unresolved late-time failure.
Older March-era dead ends have been intentionally trimmed so the next restart
does not begin from stale hypotheses.
All preserved run archives now live under the repo-level `output/` directory.

## 2026-04-11 Update

This note now has one clear operational baseline.

Current best branch:

- `mean_exchange_discretization = balanced_sbp2_pc`
- `sbp_corrector_substeps = 4`
- `sbp_transfer_mode = interp`
- `nonlinear_advection = flux`
- `Nx = 64`, `Nz = 256`, `dt = 5e-5`

Best completed clean continuation:

- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_subcycle4_probe_t80_from_t20_Nx64_Nz256_dt5e5`
- finite through `t = 80.0`
- final trusted diagnostics:
  - `Nu_d = 19.5177`
  - `R_ex_d = -2.7970e-02`
  - `R_ex_sbp = 0`
  - `max_w = 262.46`
  - `max_theta = 12.61`

Interpretation change:

- the old late-time failure is no longer the operational baseline,
- the current best branch does not reproduce that failure in the clean
  `t = 20 -> 80` continuation,
- the remaining concern is now representation / coupling consistency and
  long-time confirmation, not an obvious immediate numerical blow-up.

Movie/snapshot replay status:

- the from-start snapshot run
  `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_subcycle4_fromstart_Nx64_Nz256_dt5e5_t80_snap01`
  is the matching visualization/archive run,
- it has been resumed and is currently extending the saved `0 -> 38.8` snapshot
  sequence toward `t = 80`.

## What Is No Longer The Active Problem

- The old Chebyshev collocation null-mode issue is solved by the current
  Galerkin/tau coefficient-space formulation.
- `q'` should not carry an extra Neumann boundary condition in the production
  Miquel-style runs; `q_boundary='none'` is the correct default.
- Raw `Nusselt` and raw `vol_avg_tw` are not trustworthy late-time observables
  for `thermal_closure='evolve_mean'`.

## What The Current Diagnostics Say

The strongest change in interpretation is that the old thermal "blow-up" was
mostly a diagnostic problem.

In the shell-consistent dealiased archives:

- `128x128` at `t=8.0`
  - raw `Nu = 1.68e17`
  - dealiased `Nu_d = 3.89e1`
  - dealiased mean/fluctuation exchange residual relative error
    `= 8.5e-6`
- `64x256` at `t=8.0`
  - raw `Nu = 5.12e18`
  - dealiased `Nu_d = 3.70e1`
  - dealiased mean/fluctuation exchange residual relative error
    `= -5.9e-6`

So the primary thermal observables are now:

- `Nusselt_dealiased`
- `vol_avg_tw_dealiased`
- `mean_theta_exchange_residual_dealiased`
- `mean_theta_exchange_residual_dealiased_rel`

The raw thermal observables should be kept only as aliased comparison
diagnostics.

## What Is Still Unresolved

Even with the dealiased thermal budgets behaving sensibly, the high-vertical
state still develops a late-time non-finite failure.

The best crash-window comparison so far at `Nx=64`, `Nz=256`, `dt=5e-5` is:

| branch | construction | first non-finite |
|---|---|---:|
| `legacy` | baseline Chebyshev exchange path | `t = 11.22` |
| `coral_workgrid_flux` | shared Coral-style vertical work grid + flux-form advection | `t = 11.44` |
| `balanced_midpoint` | split frozen-`w` midpoint/CN Chebyshev prototype | unstable around `t = 10.4 - 10.6` from start |
| `balanced_sbp2` | split frozen-`w` midpoint/CN substep on a uniform SBP2 vertical grid | finite through `t = 12.00` in the same restart window; from-start run finite to `t = 20.0`, delayed blow-up at `t = 41.05` |

So the problem is no longer well described as "Nu blows up." The unresolved
problem is:

- a late-time state failure remains,
- it is sensitive to the discrete mean/fluctuation thermal coupling,
- and the best current evidence points there more strongly than to horizontal
  cascade physics.

More precisely, the current best branch now shows:

- the old `t \approx 11` failure is not fundamental,
- the solver can remain well behaved through `t = 20`,
- but a delayed pathological-growth regime appears later, with onset roughly in
  `t \approx 30 - 36`,
- and the first non-finite diagnostics in the preserved `balanced_sbp2`
  continuation occur at `t = 41.05`.

## Spectral Reading To Carry Forward

The shell-budget archives in [`spectral_analysis.md`](./spectral_analysis.md)
show:

- nonlinear shell transfer is conservative to numerical accuracy,
- nonlinear flux stays modest compared with the runaway low-shell source terms,
- the dominant positive KE production is low-shell stretching near
  `k \approx 0.9786`,
- mid and high horizontal shells stay weak.

That is not the signature of a broadband small-scale cascade failure.

## Current Working Interpretation

The active hypothesis is now narrower:

- the main remaining issue is likely in the discrete construction of the
  mean-temperature / fluctuation thermal exchange channel,
- not in the old collocation bug,
- not in the raw thermal diagnostics themselves,
- and not in a generic failure of horizontal dealiasing.

The conservative reading is that `balanced_sbp2` weakens the instability
substantially but does not remove it. So the same underlying structural defect
may still be present, only with a smaller effective late-time growth rate.

The relevant methodological branch record is in
[`adjoint_mean_exchange.md`](./adjoint_mean_exchange.md).

## Active Next Step

Current working branch:

- `sbp2-balanced-exchange`

Current result on that branch:

- keep the horizontal Fourier pseudospectral machinery unchanged,
- keep the usual `3/2` horizontal dealiasing unchanged,
- replace only the vertical thermal-exchange substep by a simpler uniform-grid
  SBP2 construction with strong endpoint Dirichlet conditions,
- use stable nodal transfer operators between the CGL solver grid and the
  uniform SBP grid,
- restart probes remain finite well beyond the old `legacy` and
  `coral_workgrid_flux` failure times,
- the full from-start `64x256` run reaches `t = 20.0` cleanly,
- but the preserved continuation from `t = 20` develops delayed violent growth
  and first non-finite diagnostics at `t = 41.05`.

So `balanced_sbp2` is the current baseline because it is the first branch that
materially postpones failure, but it is not yet a correct long-time fix.

## Immediate Improvement Targets

Based on the current baseline, the most defensible next improvements are:

- make the `balanced_sbp2` thermal exchange treatment stage-consistent with the
  IMEX update rather than a post-step split correction,
- replace the current CGL `<->` SBP transfer pair by a better matched
  projection/interpolation pair so the exchanged mean/fluctuation operators are
  less likely to leak energy structurally,
- compare the delayed `t \approx 30 - 41` onset signatures directly against the
  old `t \approx 11` onset window to test whether this is the same mechanism
  with a smaller growth rate.

## Reference Outputs

Primary diagnostic archives:

- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_kebudget_blas1_Nx128_Nz128_dt5e5_t8`
- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_kebudget_blas1_Nx64_Nz256_dt5e5_t8`

Crash-window branch outputs:

- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_coralworkgrid_flux_dense_t114_from_t10_Nx64_Nz256_dt5e5`
- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_coralworkgrid_flux_probe_t1146_from_t114_Nx64_Nz256_dt5e5`
- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedmidpoint_flux_fromstart_Nx64_Nz256_dt5e5_t11`
- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2_flux_probe_t104_from_t10_Nx64_Nz256_dt5e5`
- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2_flux_probe_t106_from_t104_Nx64_Nz256_dt5e5`
- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2_flux_probe_t114_from_t106_Nx64_Nz256_dt5e5`
- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2_flux_probe_t120_from_t114_Nx64_Nz256_dt5e5`
- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2_flux_fromstart_Nx64_Nz256_dt5e5_t20`
- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2_flux_continue_from_t20_Nx64_Nz256_dt5e5_t110`

## Restart Notes

For GPU restarts, keep BLAS thread counts at `1` so `make_grid()` does not
stall inside dense host linear algebra during IMEX-shell precomputation.

Recommended environment:

```bash
export JAX_ENABLE_X64=1
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Canonical mathematical definitions of the stored spectrum arrays and plots are
in:

- `spectral_diagnostics_reference.tex`
- `spectral_diagnostics_reference.pdf`
