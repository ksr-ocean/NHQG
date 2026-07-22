# Spectral Analysis Notes

Date: 2026-04-07

## WARNING: dominant-shell conclusions are ghost-contaminated (2026-07-04)

The 2026-07-03 review established that the raw/spectral diagnostics used
throughout this note — `ke_*_shell_tendency`, `*_horiz_spec`, raw `Nusselt`,
`KE_bc`, `*_rms` — are Parseval-style sums that count an **anti-Hermitian
ghost mode** in the rfft2 `ky=0` column of the state (see
`hermitian_ghost.md`). The ghost lives at `ky=0`, `|kx| ~ 0.9 - 1.0`, i.e.
exactly the "dominant shell `k = 0.9786`" this note identifies. Specifically:

- Finding 2 ("dominant activity stays in a low horizontal shell") and
  Finding 3 ("KE growth dominated by the linear-source chain / stretching")
  must be re-derived from Hermitian-projected checkpoints before being read
  as physics — as recorded, they primarily measure the ghost once it
  dominates (after `t ~ 4` in the t8-era archives).
- Finding 1 (nonlinear shell transfer conservative) and Finding 4 (dealiased
  thermal closure sensible) survive: the dealiased arrays pass through
  `irfft2` and are ghost-blind.
- The "What To Trust" section below was operationally correct for the wrong
  reason: raw observables are untrustworthy not because of product aliasing
  but because of the ghost.

Also note (2026-07-03 review): `q/w/th_horiz_spec` lack the `Nx^4`
normalization that `ke_horiz_spec` carries — cross-resolution spectrum
comparisons are off by `(Nx1/Nx2)^4`.

Historical note follows.

---

This note is the compact guide to the stored shellwise diagnostics and the
current reading of those diagnostics.
Preserved run archives referenced here now live under the repo-level
`output/` directory.

## Canonical References

Mathematical definitions of the archived arrays and plots are in:

- `spectral_diagnostics_reference.tex`
- `spectral_diagnostics_reference.pdf`

Primary comparison archives:

- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_kebudget_blas1_Nx128_Nz128_dt5e5_t8/spectra/spectrum_history.npz`
- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_kebudget_blas1_Nx64_Nz256_dt5e5_t8/spectra/spectrum_history.npz`

Dominant-shell summaries:

- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_kebudget_blas1_Nx128_Nz128_dt5e5_t8/spectra/dominant_shell/dominant_shell_summary.md`
- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_kebudget_blas1_Nx64_Nz256_dt5e5_t8/spectra/dominant_shell/dominant_shell_summary.md`

## What To Trust

For `thermal_closure='evolve_mean'`, the primary thermal observables are:

- `Nusselt_dealiased`
- `vol_avg_tw_dealiased`
- `mean_theta_exchange_residual_dealiased`
- `mean_theta_exchange_residual_dealiased_rel`

Use raw `Nusselt` and raw `vol_avg_tw` only as aliased comparison diagnostics.

Representative late-time values:

- `128x128`, `t=8.0`: raw `Nu = 1.68e17`, dealiased `Nu_d = 3.89e1`
- `64x256`, `t=8.0`: raw `Nu = 5.12e18`, dealiased `Nu_d = 3.70e1`

The dealiased exchange residuals remain near machine closure at the same times:

- `128x128`: `8.5e-6`
- `64x256`: `-5.9e-6`

## Main Findings

### 1. The nonlinear cascade is not the dominant source.

- Shellwise nonlinear KE transfer is conservative to numerical accuracy.
- Nonlinear flux stays modest, typically `O(10^2 - 10^3)`, even when the old
  raw thermal diagnostics become absurd.

### 2. The dominant activity stays in a low horizontal shell.

- The main shell is consistently near `k \approx 0.9786`.
- That shell carries the strongest stretching injection and the strongest
  explicit dissipation.
- Mid and high shells remain weak, so the failure is not a broadband
  high-wavenumber cascade.

### 3. KE growth is dominated by the linear-source chain, not redistribution.

The shell budgets show the same low-shell pathway:

- conduction injects `theta` variance,
- buoyancy transfers that into `w` variance,
- `q-w` stretching transfers that into baroclinic KE.

In the archived runs, the dominant positive KE source is the stretching term,
not nonlinear transfer.

### 4. The dealiased thermal closure behaves sensibly.

The shell-consistent dealiased thermal budgets close almost exactly in both
reference archives. That is why the older interpretation "the thermal sector is
obviously blowing up" is no longer the working reading.

## Stored Diagnostics To Compare Across Branches

When a new branch is rerun, compare the same archive families:

- horizontal shell spectra:
  - `ke_horiz_spec`
  - `q_horiz_spec`
  - `w_horiz_spec`
  - `th_horiz_spec`
- KE shell budgets:
  - `ke_nonlinear_shell_tendency`
  - `ke_nonlinear_flux`
  - `ke_stretch_shell_tendency`
  - `ke_diss_shell_tendency`
  - `ke_total_shell_tendency`
- `w` and `theta` shell budgets:
  - `w_q_coupling_shell_tendency`
  - `w_buoyancy_shell_tendency`
  - `th_mean_feedback_shell_tendency`
  - `th_conduction_shell_tendency`
- dealiased thermal diagnostics:
  - `Nusselt_dealiased`
  - `vol_avg_tw_dealiased`
  - `heat_flux_shell_dealiased`
  - `mean_theta_exchange_residual_dealiased`
  - `mean_theta_exchange_residual_dealiased_rel`

## Practical Comparison Protocol

For a new mean-exchange branch, the minimum useful comparison set is:

1. first non-finite time,
2. `Nu_d(t)`,
3. `mean_theta_exchange_residual_dealiased_rel(t)`,
4. dominant-shell history near `k \approx 0.9786`,
5. fixed-range `w/theta/zeta` snapshot frames.

Useful postprocessing tools:

- `scripts/plot_ke_budget_history.py`
- `scripts/plot_mean_exchange_history.py`
- `scripts/analyze_dominant_shell.py`
- `scripts/render_snapshot_panels.py`

## Current Implication For The Next Branch

The next branch should be judged mainly by:

- whether it delays or removes the late-time non-finite state,
- whether the dealiased thermal residual stays small,
- and whether the dominant low-shell stretching chain changes
  qualitatively.

If those quantities barely move, the next fix is probably not in the thermal
exchange discretization itself.
