# Mean-Exchange Branch Notes

Date: 2026-04-11

This note records the current mean-temperature / fluctuation-exchange branches
and the next simpler branch to test.

## Current Reading

The old raw thermal observables are no longer the main guide. The unresolved
problem is now:

- the dealiased thermal diagnostics can remain moderate and nearly closed,
- but the high-vertical state still reaches a late-time non-finite failure,
- and the failure time is sensitive to how the mean/fluctuation exchange pair
  is discretized.

So this note is about branch methodology, not about proving a thermal blow-up.

## Branches Already Tested

The most useful comparison window so far is `Nx=64`, `Nz=256`, `dt=5e-5`.

| branch | main idea | outcome |
|---|---|---|
| `legacy` | baseline Chebyshev exchange construction | first non-finite at `t=11.22` |
| `coral_workgrid` | shared Coral-style vertical work-grid products | delayed the failure materially |
| `coral_workgrid_flux` | same shared work grid plus flux-form advection | first non-finite at `t=11.44` |
| `balanced_midpoint` | split frozen-`w` midpoint/CN thermal substep in the Chebyshev work-grid setting | unstable earlier, around `t=10.4-10.6` from start |
| `balanced_sbp2` | split frozen-`w` midpoint/CN thermal substep on a uniform SBP2 vertical grid | finite through `t=12.0` in the same `64x256`, `dt=5e-5`, restart-from-`t=10` window |

Two stronger adjoint-flavored variants were also tried and are now retired as
active leads:

- `coral_workgrid_weakmean`
- `coral_workgrid_paired`

Both destabilized the production window earlier than the simpler
`coral_workgrid_flux` branch.

## What We Learned

- A common product-build path for the exchange terms helps.
- A more abstract "adjoint" construction is not automatically more robust.
- The current split `balanced_midpoint` prototype was too complicated for a
  first clean test; it introduced enough moving parts that its failure is hard
  to interpret.

That is why the next branch should be simpler, not more general.

## Next Branch: `sbp2-balanced-exchange`

The next branch keeps the horizontal discretization unchanged:

- same Fourier pseudospectral representation,
- same `3/2` horizontal dealiasing,
- same horizontal `psi` inversion,
- same horizontal nonlinear evaluation.

Only the vertical thermal-exchange substep changes.

### Vertical operators

Use a uniform grid

\[
z_j = \frac{j}{N}, \qquad j=0,\dots,N,
\]

with second-order SBP operators:

- diagonal norm `H`,
- first derivative `D1`,
- simple second-order Laplacian `L` for mean diffusion.

Boundary conditions are enforced strongly:

- `Theta[0] = Theta[N] = 0`,
- `theta[0,:,:] = theta[N,:,:] = 0`.

### Frozen-`w` balanced substep

At a given stage, use the same dealiased horizontal averaging path as the mean
equation to build

\[
F_n(z) = \langle w^\star \theta_n \rangle_{xy},
\qquad
m^\star(z) = \langle (w^\star)^2 \rangle_{xy}.
\]

Let

\[
M = \operatorname{diag}(m^\star).
\]

Then solve for the updated mean profile with the midpoint/CN form

\[
A \Theta_{n+1} = \text{rhs},
\]

where

\[
A
=
I
- \frac12 \mu \kappa_\theta \Delta t\, L
- \frac14 \mu \Delta t^2\, D1\, M\, D1,
\]

\[
B
=
I
+ \frac12 \mu \kappa_\theta \Delta t\, L
+ \frac14 \mu \Delta t^2\, D1\, M\, D1,
\]

\[
\text{rhs}
=
B \Theta_n - \mu \Delta t\, D1 F_n.
\]

After enforcing the endpoint Dirichlet rows in `A`, define the midpoint
gradient

\[
g_{n+1/2} = \frac12 D1(\Theta_n + \Theta_{n+1}),
\]

and update the fluctuation field by

\[
\theta_{n+1}
=
\theta_n - \Delta t\, w^\star g_{n+1/2}.
\]

### Required implementation details

- Reconstruct `w` and `theta` on the usual dealiased horizontal physical grid.
- Interpolate only in the vertical direction between the Chebyshev state and
  the uniform SBP grid.
- Project the updated `Theta` and `theta` back into the existing Chebyshev /
  Galerkin solver representation after the substep.
- Keep the rest of the IMEX step unchanged.

Implementation note:

- A direct global inverse from Chebyshev coefficients to same-size uniform-grid
  values is numerically unusable by `Nz=256` because the corresponding
  Chebyshev-on-uniform Vandermonde matrix is extremely ill-conditioned.
- The implemented branch therefore uses stable nodal transfer matrices:
  CGL nodal values -> uniform SBP grid -> SBP2 substep -> CGL nodal values ->
  existing Chebyshev/Galerkin state.
- The current implementation uses piecewise-linear vertical transfer for that
  conversion layer.

## Residual To Monitor

For the balanced substep, monitor the discrete exchange-plus-diffusion residual

\[
R_{\mathrm{bal}}
=
\frac{\|\theta_{n+1}\|_H^2 - \|\theta_n\|_H^2}{2 \Delta t}
+ \frac{\|\Theta_{n+1}\|_H^2 - \|\Theta_n\|_H^2}{2 \mu \Delta t}
- \kappa_\theta \Theta_{1/2}^T H (L \Theta_{1/2}),
\]

with

\[
\Theta_{1/2} = \frac12(\Theta_n + \Theta_{n+1}).
\]

This is the branch-specific diagnostic that should be near solver tolerance if
the substep is assembled correctly.

## Success Criteria

The new branch is only interesting if it improves at least one of these:

1. later first non-finite time than `legacy`,
2. better robustness than `balanced_midpoint`,
3. small dealiased exchange residuals during the same crash window,
4. a qualitative change in the dominant-shell low-`k` behavior.

## Current Result

The implemented `balanced_sbp2` branch has now been tested on GPU 0 in the
same `Nx=64`, `Nz=256`, `dt=5e-5`, `nonlinear_advection='flux'` restart window.
All preserved run archives referenced below now live under the repo-level
`output/` directory.

Run sequence:

- `t=10.00 -> 10.40`
  - output:
    `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2_flux_probe_t104_from_t10_Nx64_Nz256_dt5e5`
- `t=10.40 -> 10.60`
  - output:
    `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2_flux_probe_t106_from_t104_Nx64_Nz256_dt5e5`
- `t=10.60 -> 11.40`
  - output:
    `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2_flux_probe_t114_from_t106_Nx64_Nz256_dt5e5`
- `t=11.40 -> 12.00`
  - output:
    `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2_flux_probe_t120_from_t114_Nx64_Nz256_dt5e5`

Outcome:

- the branch stays finite through `t=12.0`,
- it therefore cleanly outperforms `legacy` (`t=11.22`) and
  `coral_workgrid_flux` (`t=11.44`) in the same comparison window,
- the dealiased thermal diagnostics remain moderate throughout that window,
  e.g. at `t=12.0`:
  - `Nu_d = 2.9584e+01`

## SBP Internal Exchange Audit

The later stage-wise branch `balanced_sbp2_pc` still fails eventually, but a
new internal audit now separates the SBP exchange algebra from the old
CGL/Clenshaw-Curtis monitoring layer.

New diagnostic fields:

- `th_mean_feedback_sum_sbp`
- `mean_flux_exchange_tendency_sbp`
- `mean_theta_exchange_boundary_sbp`
- `mean_theta_exchange_residual_sbp`
- `mean_theta_exchange_residual_sbp_rel`

These are built entirely on the uniform SBP grid using the same SBP norm `H`
and first-derivative operator `D1` used by the thermal corrector itself.

### Key result

For the `balanced_sbp2_pc` continuation

- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_continue_from_t42_Nx64_Nz256_dt5e5_t80`

the rebuilt spectrum history now shows:

- at `t=42.25`:
  - `R_ex_d = -3.534665e+01`
  - `R_ex_sbp = 1.136868e-13`
- at `t=45.00`:
  - `R_ex_d = -2.709585e+01`
  - `R_ex_sbp = 2.842171e-14`
- at `t=48.25` (last finite checkpoint):
  - `R_ex_d = 1.376013e+03`
  - `R_ex_sbp = 1.818989e-12`
- the SBP boundary term stays at roundoff:
  - `|B_sbp| <= 1e-18` in these samples

Up to the last finite checkpoint:

- `max |R_ex_sbp| = 1.818989e-12`
- `max |R_ex_sbp_rel| = 3.090513e-16`
- `max |R_ex_d| = 1.679638e+03`
- `max |R_ex_d_rel| = 2.279119e-01`

Interpretation:

- the SBP thermal-exchange pair is internally closed to machine precision,
- the large late-time exchange residual seen in the old dealiased diagnostic is
  therefore not coming from the SBP substep algebra itself,
- the remaining leak must be in the representation-transfer / monitoring layer
  or in how the full solver couples that corrected thermal state back to the
  rest of the Chebyshev/CGL system.

This materially narrows the next debugging target. The right next questions are
now about CGL <-> SBP transfer consistency and full-step coupling, not about
the frozen-`w` SBP exchange cancellation on its own terms.

## SBP Corrector Subcycling Test

The next test was to keep the baseline interpolation transfer
(`sbp_transfer_mode='interp'`) but subcycle the inserted SBP corrector inside
each `balanced_sbp2_pc` stage:

- `sbp_corrector_substeps = 4`
- restart from the clean `t=42.0` checkpoint:
  `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_probe_t420_from_t20_Nx64_Nz256_dt5e5/checkpoint_00840000.npz`
- output:
  `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_subcycle4_probe_t50_from_t42_Nx64_Nz256_dt5e5`

Outcome:

- this did **not** produce a clean improvement in the real restart test,
- the branch remained finite through `t=42.60` in the short probe, but it was
  already strongly pathological by `t=42.10-42.60`,
- representative values:
  - `t=42.00`: `Nu_d = 6.2404e+01`, `R_ex_d = 5.1404e+00`
  - `t=42.10`: `Nu_d = 1.5512e+03`, `R_ex_d = -1.1127e+03`
  - `t=42.30`: `Nu_d = 1.6512e+03`, `R_ex_d = -3.3191e+03`
  - `t=42.60`: `Nu_d = 2.0810e+03`, `R_ex_d = -5.9320e+03`
- throughout the same window, `R_ex_sbp` stayed at roundoff (`~1e-12` or
  smaller)

Interpretation:

- solver-level SBP corrector subcycling by itself does not cure the late-time
  instability,
- the local coupling-sensitivity analysis was directionally useful, but the
  full restart test shows that subcycling alone is not enough to recover a
  robust long-time branch,
- the remaining lead is still the mismatch between the internally balanced SBP
  thermal correction and the full Chebyshev/CGL stage coupling.

## Earlier-Restart Subcycled Run

The subsequent test was to keep the same branch

- `mean_exchange_discretization = balanced_sbp2_pc`
- `sbp_transfer_mode = interp`
- `sbp_corrector_substeps = 4`

but restart \emph{earlier}, from the `t=40.0` checkpoint:

- restart:
  `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_probe_t420_from_t20_Nx64_Nz256_dt5e5/checkpoint_00800000.npz`
- output:
  `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_subcycle4_probe_t50_from_t40_Nx64_Nz256_dt5e5`

This run materially changed the interpretation.

Outcome:

- the run stayed finite all the way through `t=50.0`,
- there was no non-finite event in the whole `t=40 -> 50` segment,
- the internal SBP audit stayed at roundoff throughout:
  - `max |R_ex_sbp| = 9.094947e-13`
- representative values:
  - `t=42.00`: `Nu_d = 7.138127e+01`, `R_ex_d = -1.325659e+00`
  - `t=45.00`: `Nu_d = 1.527344e+01`, `R_ex_d = 7.339471e-02`
  - `t=48.00`: `Nu_d = 2.509409e+01`, `R_ex_d = -1.963622e+00`
  - `t=49.50`: `Nu_d = 2.648532e+01`, `R_ex_d = -1.183184e+00`
  - `t=50.00`: `Nu_d = 8.116067e+01`, `R_ex_d = -2.778641e+02`

The most important qualitative point is that this run \emph{did not} reproduce
the immediate catastrophe seen in the direct `t=42` subcycled restart. In the
earlier-restart run, the interval `t=42 -> 49.8` remained comparatively calm,
with `Nu_d` mostly in the `15 - 31` range and `R_ex_d` usually `O(10^{-1})`
to `O(1)`. A genuine late uptick did appear only near the end, around
`t=49.9 - 50.0`.

Interpretation:

- the direct `t=42` restart is no longer a trustworthy measure of this branch,
- the state at `t=42` on the old continuation was already contaminated enough
  to bias a branch-switch restart,
- restart timing is now part of the numerical experiment, not an incidental
  detail,
- the best current branch is therefore:
  - `balanced_sbp2_pc`
  - with `sbp_corrector_substeps = 4`
  - assessed from earlier restarts, not from a late handoff alone.

## Clean `t=20 -> 80` Continuation

The decisive next test was to stop relying on late branch-switch restarts and
run the current best subcycled stage-wise branch through the full developed
window starting from the clean `t=20.0` checkpoint.

Branch settings:

- `mean_exchange_discretization = balanced_sbp2_pc`
- `sbp_corrector_substeps = 4`
- `sbp_transfer_mode = interp`
- `nonlinear_advection = flux`
- `Nx = 64`, `Nz = 256`, `dt = 5e-5`

Run:

- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_subcycle4_probe_t80_from_t20_Nx64_Nz256_dt5e5`

Outcome:

- the run stayed finite through `t=80.0`,
- it crossed the earlier `t \approx 50` danger window without reproducing the
  old late catastrophic event,
- the internal SBP audit stayed at roundoff through the full continuation,
- final values at `t=80.0`:
  - `Nu_d = 1.951773e+01`
  - `R_ex_d = -2.797001e-02`
  - `R_ex_sbp = 0`
  - `max_w = 2.624615e+02`
  - `max_theta = 1.261383e+01`

Interpretation:

- this is now the best completed run in the branch family,
- the operational baseline should no longer be the older `balanced_sbp2`
  delayed-failure continuation,
- and the current evidence favors the view that the remaining issue is a
  narrower representation/coupling question rather than failure of the SBP
  thermal exchange algebra itself.

## From-Start Snapshot Replay

To support movie generation and full-interval diagnostics, the matching
from-start snapshot-producing run is

- `output/output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_subcycle4_fromstart_Nx64_Nz256_dt5e5_t80_snap01`

This run originally stopped at `t=38.75` and has now been resumed from

- `checkpoint_00775000.npz`

with the same branch settings and `snapshot_dt = 0.1` so the `0 -> 80`
visualization archive remains consistent.

## Final Status (2026-07-04) — branch validated; note now historical

The `balanced_sbp2_pc` + `sbp_corrector_substeps=4` baseline described above
was subsequently extended (2026-04-19/20, with 2/3-rule dealiasing) to a clean
**`t = 120`** at `128x256` and `t = 63` at `256x256`. The late-time stability
question this note tracked is closed. See `CLAUDE.md` "Current Status
(2026-07-04)".

Three later findings qualify the record above (2026-07-03 review):

1. `R_ex_sbp = 0` is a **structural identity**, not a step audit: the SBP
   residual reduces to a boundary term that the Dirichlet rows kill
   identically, regardless of the exchange coefficients actually applied. It
   certifies the operator algebra, not the integrated step. The
   "SBP Internal Exchange Audit" section above overstates what it can detect.
2. The large late-time `R_ex_d` values quoted above (and the raw
   `th_mean_feedback_sum` explosions) trace to the **anti-Hermitian ghost
   mode** (`hermitian_ghost.md`), not to a physical exchange leak. The
   "contaminated state" language around the t=42 restart discussion is partly
   literal: the state carried a large ghost, invisible to the physics.
3. The `interp` transfer pair this branch relies on is also an uncontrolled
   near-wall smoother (11 CGL points inside the first uniform cell at Nz=256,
   applied 8x per step). Some of the branch's robustness may be that implicit
   filtering, and it is the leading suspect for the open Nusselt gap
   (`Nu_d ~ 18-20` at Ra=100 vs Miquel 43.37 +/- 2.54).

## Reference Files

- [hermitian_ghost.md](./hermitian_ghost.md)
- [mean_eq.tex](./mean_eq.tex)
- [discretely_balanced_mean_fluctuation_thermal_formulation.tex](./discretely_balanced_mean_fluctuation_thermal_formulation.tex)
- [blowup.md](./blowup.md)
- [spectral_analysis.md](./spectral_analysis.md)
- [CLAUDE.md](./CLAUDE.md)
