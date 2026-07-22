# The Anti-Hermitian Ghost Mode: Diagnosis of the Raw-Diagnostics "Blowup"

Date: 2026-07-04 (review session of 2026-07-03)

This note is the canonical record of the mechanism behind the raw
(`Nusselt`, `vol_avg_tw`, `KE_bc`, shell-budget) diagnostic explosions seen in
otherwise healthy runs. It supersedes the "aliased diagnostic artifact"
interpretation in `blowup.md`, `spectral_analysis.md`, and the 2026-04-05/11
sections of `CLAUDE.md`.

A full pedagogical walkthrough (with the framework context) is in
`NHQG_framework_deck.pdf`, Part B.

## Summary

The `ky=0` column of the evolved `rfft2`-layout state accumulates a purely
**anti-Hermitian** component ("ghost"):

```
f_hat(kx, ky=0)  =  -conj(f_hat(-kx, ky=0))        (measured to all digits)
```

- `irfft2` reconstructs physical fields from the Hermitian projection only,
  so the ghost is **invisible to the physics** and to every physical-product
  ("dealiased") diagnostic.
- Parseval-style spectral diagnostics count `|f_hat|^2` directly, so they
  **see the ghost in full**.
- Nonlinear tendencies are `rfft2` of real pointwise products, hence exactly
  Hermitian: they feed the ghost nothing. Every linear operation (psi
  inversion, alpha factors, per-shell implicit solve, stretching, buoyancy,
  conduction) is diagonal in `(kx, ky)` and Hermitian-blind: it evolves the
  ghost with the same linearized dynamics as a physical mode.
- Therefore the ghost integrates the **linearized NHQG system with no
  nonlinear saturation**: pure exponential growth from a roundoff seed,
  forever.

## Evidence (verified directly on checkpoints, 2026-07-03)

Impossibility bound. For coefficients representing any real field,
`|<w theta>_xy| <= max|w| * max|theta|`. At `t=120` in the
`Nx=128, Nz=256` chain: `max|w| * max|theta| ~ 7.5e3`, but recorded
`tw_raw = 6.6e27`. The raw diagnostic and `max_w` come from the same arrays,
so the state cannot represent a real field.

Direct measurement:

| checkpoint | field | max anti-Hermitian (ky=0) | Parseval E / projected E |
|---|---|---:|---:|
| `output_combined_Nx128_Nz256_.../checkpoint_02400000.npz` (t=120) | q' | 1.07e19 | 5.9e26 |
| same | w | 2.85e18 | 6.8e24 |
| same | theta | 1.43e17 | 1.1e26 |
| `..._kebudget_blas1_Nx64_Nz256_dt5e5_t8/checkpoint_00160000.npz` (t=8) | q' | 2.26e13 (vs Hermitian 8.8e3) | 3.4e16 |

The `ky`-Nyquist column is exactly zero (dissipation keeps it clean).
Projected (physical) energies are moderate everywhere — the resolved solution
underneath is healthy, exactly as `Nusselt_dealiased` reported.

Growth-rate closure:

- Early (t=0..8, conduction-era gradients): amplitude `1e-16 -> 2.26e13`
  over 8 units gives rate ~8.4/unit vs the theoretical max linear rate
  ~8.6 at k~0.83. Match.
- Late (saturated mean state, t=40..120): ghost energy `~1e3 -> 9.1e28`
  gives amplitude rate ~0.37/unit — the residual growth rate of the
  marginal mode on the turbulently flattened mean gradient. This is why the
  stable runs' raw diagnostics grow gently for 80 units instead of exploding
  in 8.
- Location: the ghost concentrates at `ky=0`, `|kx| ~ 0.9 - 1.0` — the
  fastest-growing linear modes that fit the box.

## Consequences for past interpretations

- The historical raw-`Nu` "blowup" (1.68e17 at t=8, 6.6e27 at t=120) was the
  ghost, not physics and **not classical aliasing** (aliasing redistributes a
  real field's energy and can never break the Parseval bound; genuine
  Nyquist-band aliasing effects here are small and bounded).
- `th_mean_feedback_sum ~ -1e16..-1e18` explosions: ghost.
- **The "dominant shell k = 0.9786" result is ghost-contaminated**:
  `ke_stretch_shell_tendency` and the other shell budgets are spectral inner
  products, and k~0.98 is the ghost's home shell (`ky=0`, `kx ~ 7.5*k0`;
  bin-center label). The April "low-shell stretching injection" narrative in
  `spectral_analysis.md` must be re-derived from Hermitian-projected states
  before any physics is read from it.
- Ghost-visible archive arrays: raw `Nusselt`, `vol_avg_tw`, `KE_bc`/`KE_tot`,
  `ke_*_shell_tendency`, `ke_nonlinear_flux`, `q/w/th_horiz_spec`, `*_rms`,
  `vertical_mode_energy`, `th_mean_feedback_sum`, `heat_flux_mismatch`.
- Ghost-blind (trustworthy) arrays: `Nusselt_dealiased`,
  `vol_avg_tw_dealiased`, all dealiased shell budgets and exchange residuals,
  `max_w/max_theta/max_speed`, `th_bar` profile diagnostics,
  `mean_theta_exchange_residual_sbp`.
- The operational doctrine "trust the dealiased diagnostics" was correct —
  they are ghost-blind because they pass through `irfft2`.
- `fd_vertical_benchmark/` has the identical disease independently (raw Nu
  ~1e29-1e31 at t=10 while `max|w|*max|theta| ~ 5.7e4`; its spectrally
  computed `R_ex_sbp` reaches -3.4e29 and is currently meaningless late in a
  run). No Hermitian enforcement exists in that package either.

## Seeds

1. FFT roundoff: `rfft2` of a real array is Hermitian only to machine
   precision, so every step deposits O(1e-16) anti-Hermitian dust.
2. `make_initial_state` (`nhqg/solver.py`, ~line 1224) draws independent
   complex noise on **all** modes — it violates Hermitian symmetry at t=0 at
   the full noise amplitude. (Fourier upsampling of a checkpoint resets the
   ghost, which is why the Nx=128 chain restarted at t=40 with
   `Nu_raw = Nu_d` exactly.)

## The fix (cheap, not yet implemented as of 2026-07-04)

1. Symmetrize the self-conjugate columns after each step (or every ~100
   steps; per-step ghost growth is `exp(8.6 * 5e-5) ~ 1.0004`):

   ```python
   neg_ix = (-jnp.arange(Nx)) % Nx
   f = f.at[:, :, 0].set(0.5 * (f[:, :, 0] + jnp.conj(f[:, neg_ix, 0])))
   ```

   (and the ky-Nyquist column if it ever becomes nonzero). Cost: one
   `(Nz, Nx)` slice per field.
2. Make `make_initial_state` Hermitian (draw real noise, `rfft2` it).
3. Regression test: step N times, assert the anti-Hermitian norm of every
   field stays at roundoff.
4. Re-run `scripts/analyze_dominant_shell.py` and the KE shell budgets on
   Hermitian-projected checkpoints; re-read the April spectral narrative.
5. Port the projection into `fd_vertical_benchmark/`.

## Bonus once fixed

With a Hermitian-projected and (under `23_rule`) band-limited state, the raw
and dealiased diagnostics should agree to near machine precision (the
horizontal mean of a band-limited product on the Nx grid is exactly
alias-free). Raw-vs-dealiased agreement then becomes a **free continuous
integrity audit**: any re-divergence flags a new leak immediately instead of
1e27 later.

## The recurring motif

The D_Z null mode (March), the untruncated 2/3-rule band (see `CLAUDE.md`
2026-07-04 status), and the anti-Hermitian ghost are the same disease: state
components outside the representation the physics actually sees, amplified by
linear dynamics. The cure is always the same — make the invariant explicit
and project onto it.
