# Plan: Polar vortex crystals (Siegelman–Young–Ingersoll 2022) at 64×512×512, Neumann-w surface, 2-GPU sharding

**Date:** 2026-07-22. **Target paper:** Siegelman, Young & Ingersoll (2022),
*Polar vortex crystals: Emergence and structure*, PNAS 119(17) e2120486119 —
the barotropic polar-cap "trap" model, which `NHGQ_polar.tex` already
generalizes to the full 3-D NHQGE. **Target grid:** Nz=64, Nx=Ny=512.
**New BC:** Neumann w at the top surface (free/open surface), Dirichlet w at
the bottom — the bottom becomes the only true boundary layer.

This document has three parts: (I) the polar-cap + mixed-BC implementation
plan, (II) the 2-GPU sharding plan, (III) work order with validation gates.

---

## Part 0 — Prerequisites (before any new production run)

These are cheap and non-negotiable for a run class we intend to publish from:

- **P0.1 Hermitian ghost fix in the solver.** Symmetrize the rfft2 ky=0 (and
  ky-Nyquist) columns each step, Hermitian-project initial noise, add the
  regression test. One line + test; see `hermitian_ghost.md`. A new multi-day
  polar run must not carry a ghost — at 512² the ghost's home-shell
  contamination would again poison every Parseval diagnostic.
- **P0.2 State masking at init/restart under `23_rule`.** The state should be
  masked to the retained band at init/restart/finalize (currently output-only).
  At Nx=512 the masked band sits far above the unstable band (unlike Nx=64),
  but restart hygiene should not depend on that accident.
- **P0.3 Git hygiene.** `git init`, `.gitignore` for `output/` (~310 GB),
  logs, PDFs; first commit before the refactors below. The repo still has zero
  commits and Part I touches core solver files.
- **P0.4 VRAM levers** (see memory note; both needed at 512²):
  (a) for `q_boundary='none'`, use the precomputed scalar `inv_alpha_q`
  instead of dense per-shell identity-scaled `q_solve` matrices;
  (b) stop materializing the full `mat_shells[ksq_idx]` gather —
  at 512²×64 that tensor is 512×257×65×65×8 B ≈ **4.4 GB per gather**
  (float64). Replace with a `jax.lax.map`/scan over chunks of the kx axis or a
  segment-sum formulation. This also interacts with the sharding choice (Part II).

---

## Part I — Polar cap with a free-surface top at 64×512×512

### I.1 The trap (γ-effect): follow `NHGQ_polar.tex` Approach A

The formulation document is complete and correct; implementation is small:

1. **Config** (`nhqg/config.py`): add `gamma: float = 0.0`,
   `trap_r_star: float | None = None`, `trap_sharpness: float = 20.0`
   (SYI22's |A_d|), and optional sponge params
   (`sponge_lambda_max=0.0`, `sponge_r=None`, `sponge_sharpness=10.0`).
   `beta` stays; `gamma != 0` and `beta != 0` are mutually exclusive (assert).
2. **Grid** (`nhqg/grid.py`): precompute
   `eta_hat = rfft2(-0.5*gamma*r²*sigma_trap(r))` on the unpadded grid,
   masked to the 2/3 band, stored on the `Grid`. Z-independent, (Nx, Nk)
   complex — negligible memory. Assert at build time that the energy of η
   outside the retained band is < 1e-12 of its total (trap smoothness check:
   tanh decay scale r*/A_d must be ≳ 3 grid cells).
3. **RHS** (`nhqg/solver.py`, lines ~393/473): replace the spectral β term
   with the augmented Jacobian: pass `q_hat + eta_hat[None, :, :]` as the
   advected PV argument in the fused triple-advection (both `jacobian` and
   `flux` paths — in flux form `div(u(q+η)) = J(ψ,q) + J(ψ,η)` since
   `div u = 0`, so the same one-line augmentation is exact). The β spectral
   term remains for the `beta` path; it is simply zero here.
4. **Sponge:** default off (SYI22 did not need one). If enabled, apply
   `-λ(x,y)·f` for q,w,θ in physical space inside the existing dealiased
   product pipeline (λ is smooth/broad-band; one extra product per field).
5. **Diagnostics** (later milestone): azimuthal decomposition E(m), vortex
   tracking, trap-restricted spectra — per `NHGQ_polar.tex` §7.

Tests: (i) with η ≡ βy restricted to a linear ramp in a periodic-safe test
(or equivalently comparing `J(ψ,η)` against `iβk_x ψ̂` for a resolved η),
the two paths agree to roundoff; (ii) Jacobian conservation/antisymmetry
unchanged with the augmented argument; (iii) η is invisible when γ=0.

### I.2 Mixed vertical BCs: Dirichlet bottom, Neumann-w top

**Physics decision D1 (needs your sign-off; recommendation below).** The user
directive fixes w: `w(0)=0` (bottom wall), `∂_Z w(1)=0` (free surface). Open:
θ′ and Θ̄ at the top:

- *Recommended first configuration:* keep **θ′(1)=0 and Θ̄ Dirichlet at the
  top** (thermally conducting free surface). Heat still exits the top, so a
  statistically steady state exists and the Nusselt number stays defined;
  only the mechanical wall is removed, and the bottom is the lone
  viscous/thermal boundary layer in w.
- *Later experiment:* insulating top `∂_Zθ′(1)=0` — but then Θ̄ needs a
  fixed-flux top BC and the layer heats secularly unless a radiative-cooling
  term is added. Defer; it changes the thermal closure story.

**Discretization route (decision D2; recommendation: Route A).**

- **Route A — mixed Chebyshev–Galerkin stencil (recommended).** Generalize the
  Coral-style basis: instead of the shared both-Dirichlet stencil
  `φ_n = -T_n + T_{n+2}`, give **each field its own stencil**. For w use the
  Shen-type 3-term basis `φ_n = T_n + a_n T_{n+1} + b_n T_{n+2}` with
  `(a_n, b_n)` solving the two BC rows `φ(Z=0)=0`, `φ'(Z=1)=0` (closed-form
  from `T_n(±1)=(±1)^n`, `T_n'(±1)=(±1)^{n+1}n²`). θ keeps the Dirichlet
  stencil (per D1). Everything in the existing machinery generalizes:
  - `grid.py`: build `w_stencil`/`w_pinv` and `th_stencil`/`th_pinv`
    separately (the "unique left inverse from the first Nz−1 rows" trick
    carries over — the leading T_n coefficient is 1, so the top block stays
    triangular/invertible; this was the lesson of the 2026-03 pinv bug, do
    not use a Moore–Penrose pinv).
  - `_build_imex_inv`: assemble `B = w_pinv @ G_Z @ q_solve @ G_Z @ w_stencil`
    in the w-basis. The buoyancy block elimination is no longer a scalar
    `alpha_w_eff`: with w and θ in different bases, eliminating θ gives
    `A'(k) = α_w I − (γ_imex dt)² c(k) B − (γ_imex dt)² (Ra/σ)/α_θ(k) · K`,
    where `K = (w_pinv @ th_stencil)(th_pinv @ w_stencil)` is a fixed
    (Nz−1)² matrix precomputed once. Shell dedup is preserved (α_θ depends
    only on |k|²), matrix sizes and memory are unchanged, and setting both
    stencils to Dirichlet must reproduce today's matrices **bitwise** — that
    is the structural regression test.
  - `solver.py`/`io.py`/`diagnostics.py`: replace every
    `dirichlet_stencil`/`dirichlet_pinv` use with the per-field operators
    (lift w through `w_stencil` in the exchange path, snapshots, spectra).
  - Mean-temperature machinery: **unchanged** under D1 — Θ̄ keeps Dirichlet
    rows at both ends, so the `balanced_sbp2_pc` work-grid solve and the
    `R_ex_sbp ≡ 0` structural identity survive as-is. w enters the exchange
    as field values, not through its BCs.
- **Route B — SBP-SAT FD vertical** (`fd_vertical_benchmark/`, `sbp42`): SAT
  penalties make mixed BCs trivial and this was the original motivation for
  the package (asymmetric-BC goal). But it is a benchmark solver, never run
  end-to-end long, with its own ghost bug, and adopting it abandons the
  validated Chebyshev production stack. Keep it as a **cross-check**: run the
  mixed-BC case at 128×128 in both solvers and compare onset/growth/late-time
  statistics.

**Linear-onset gate for mixed BCs.** The stress-free `sin(πZ)` analytics no
longer apply — Ra_c and k_c shift with one rigid boundary. Build a 1-D
generalized eigenvalue solver in Z per k (30 lines with the existing `G_Z`,
`V`, stencil operators, CPU) and use it to (a) tabulate the new Ra_c(k), and
(b) gate the solver: measured growth rates at 3 values of k within 1% of the
EVP. This replaces `test_solver.py`'s onset test for the mixed-BC config.

### I.3 Domain-size / resolution design at 512² (decision D3)

Hard constraint: under `23_rule`, usable `k_max = (Nx/3)·(2π/L)`, and the
convective band extends to k ≈ 3.16 (upright value; recompute with mixed BCs
via the EVP above). Consequences at Nx=512:

| L (units of Lc≈4.815) | L (code units) | k_max | covers band ≤3.16? | r*=0.45·L/2 (Lc) |
|---|---|---|---|---|
| 100 | 481.5 | 2.23 | **NO** — injection under-resolved | 22.5 |
| 70  | 337   | 3.19 | marginal | 15.8 |
| **48** | **231** | **4.65** | **yes, with margin** | **10.8** |
| 32  | 154   | 6.98 | yes (small trap) | 7.2 |

- **Recommendation: L = 48 Lc** (≈ 4.8× the current production domain area),
  r* ≈ 52 code units ≈ 10.8 Lc. Then choose γ so the emergent crystal radius
  `L_γ = (U/γ)^{1/3} ≈ 5–7 Lc` sits comfortably inside the trap. U is
  emergent — calibrate γ with a short pilot: measure barotropic U_rms in a
  γ=0 spin-up at this domain size, then set γ for the target L_γ, then sweep.
- Nz=64 caveats, stated up front: (a) at Ra=100 Miquel used Nz≥256 for Nu
  accuracy — bottom-BL Nu from these runs will be under-resolved (we already
  carry a Nu gap; polar runs are for vortex phenomenology, not Nu); (b) the
  q vertical-tail under-resolution seen at Nz=256 will be present; monitor the
  vertical Chebyshev tail diagnostic and consider a mild `vertical_cutoff_n`
  only if it destabilizes.

### I.4 Physics campaign (after gates pass)

Follow the validation ladder of `NHGQ_polar.tex` §8, in order:
**P0** quasi-barotropic trap validation (small Nz, random-vorticity init,
crystal radius vs L_γ scaling across ≥3 γ values — the direct SYI22
reproduction); **P1** γ=0 convective control (trap must not disturb the
inverse cascade / condensate); **P2** γ>0, Ld=∞ (first new result:
convectively forced crystal); **P3** finite Ld; **P4** (γ, Ld) sweep — note
P4 is where "one run per GPU" beats sharding (Part II).

---

## Part II — Sharding across 2 GPUs: yes, we can

**Answer: yes** — single host, single process, `jax.sharding.Mesh` over GPUs
6 and 7 (per the GPU policy; they are H200-141GB, NVLink-connected), with
`NamedSharding` + jit (GSPMD). No multi-host machinery needed. JAX 0.9.1
supports all of this.

**Why shard at all:** 512²×64 float64 *fits* on one H200 (state ~0.4 GB, IMEX
tables ~0.75 GB, physical work arrays ~0.14 GB/field; the only large item is
the 4.4 GB gather, which P0.4b removes). The motivation is **throughput**:
V100 measured 215 ms/step at 512×64; H200 float64 should give ~40–60 ms/step
on one GPU, i.e. ~28 h per 100 time units at dt=5e-5. Crystal emergence needs
multi-hundred-unit integrations — a ~1.9× speedup halves multi-day runs.

**Decomposition analysis** for state layout `(Nz+1, Nx, Nk)`:

- *Shard axis 0 (vertical levels):* horizontal FFTs are batched over Z →
  the entire nonlinear/product pipeline (dominant flops) runs with **zero
  communication**. Vertical contractions (`einsum('ij,j...')` for G_Z, V,
  V_inv, stencils; the per-shell IMEX matmul) contract the sharded axis →
  GSPMD inserts all-gathers, ~70–140 MB each, O(20)/step → ~2–3 GB/step.
- *Shard axis 1 (kx rows):* vertical ops and the per-shell IMEX solve are
  batched (comm-free, and the IMEX gather working set halves per GPU); the
  FFT along the sharded axis triggers all-gathers instead, similar volume.
- At 900 GB/s NVLink, either choice costs ~3–5 ms/step of communication
  against ~50 ms of compute → **expected scaling ~1.8–1.9×**. With only two
  devices there is no deep pencil hierarchy to design; the classic two-phase
  ("transpose") layout is available via `with_sharding_constraint` if the
  single-axis variants underperform.

**Implementation steps:**

1. `CUDA_VISIBLE_DEVICES=6,7`, `mesh = jax.make_mesh((2,), ('shard',))`;
   check idle first with `nvidia-smi` (policy).
2. Shard the initial state with `jax.device_put(state, NamedSharding(mesh, P(None, 'shard', None)))`
   (axis-1 variant) and pass `in_shardings`/`out_shardings` to the jitted
   step. Precomputed tables: replicate everything except `imex_inv` gather
   inputs; `ksq_idx` shards with the state.
3. Add `jax.lax.with_sharding_constraint` at two points — after the forward
   FFTs (spectral phase) and after the inverse FFTs (physical phase) — so
   GSPMD cannot invent pathological reshardings inside the step.
4. **Measure all three variants** (axis-0, axis-1, two-phase transpose) at
   256²×64 first, pick by steps/s, then confirm at 512²×64.
5. Checkpointing: `jax.device_get` on sharded arrays assembles to host
   transparently — `io.py` needs no format change.

**Validation gates:** (a) 2-GPU trajectory identical to 1-GPU within float64
roundoff over 200 steps at 128²×64 (same seed; FFT reduction order can give
~1e-14 relative drift — gate at 1e-12 on state norms, not bitwise);
(b) steps/s ≥ 1.6× single-GPU at 512²×64; (c) `nvidia-smi` shows balanced
utilization/memory on GPUs 6 and 7 only.

**Alternative for sweeps:** for the P4 (γ, Ld) sweep, two independent runs on
one GPU each is perfect 2.0× scaling with zero code — sharding is for the
long single flagship runs (P2/P3), trivial parallelism is for the sweep.

---

## Part III — Work order, effort, and gates

| # | Milestone | Depends on | Gate (lead runs gates; GPU gates on 6/7 only) |
|---|---|---|---|
| M0 | P0.1–P0.4 hygiene (ghost fix, init-mask, git, VRAM levers) | — | ghost regression test; 64×256 restart bit-compat; peak VRAM at 256²×64 < 10 GB |
| M1 | Trap (γ, r*, A_d, η̂ precompute, augmented Jacobian) | M0 | η/β equivalence to roundoff; conservation tests; **P0 crystal + L_γ scaling** |
| M2 | Mixed BCs (per-field stencils, K-matrix IMEX, exchange/io) | M0 | both-Dirichlet bitwise reproduction; BC exactness 1e-12; **mixed-BC EVP onset within 1%**; IMEX-vs-RK4; Route-B cross-check at 128×128 |
| M3 | 2-GPU sharding | M0 (+M2 for the real step) | 1-vs-2 GPU trajectory match; ≥1.6× steps/s at 512²×64 |
| M4 | 512²×64 shakedown (dt scan, diagnostics cadence, polar diagnostics) | M1–M3 | stable 10-unit pilot; ghost-free by construction; vertical-tail monitor |
| M5 | Physics campaign P1→P4 | M4 | per `NHGQ_polar.tex` §8 |

Notes on execution per the delegation policy: M1/M2 mechanical layers
(stencil builders, per-field plumbing, tests) are spec-able to codex with
CPU-only validation commands; the IMEX K-matrix derivation, the EVP gate, all
GPU submissions, and this plan's judgment calls stay with the lead. M1 and M2
are independent and can proceed in parallel once M0 lands.

**Decision log (2026-07-22, user sign-off):**
- **D3 — DECIDED: L = 48 Lc** (user: "agree with the L=48 Lc choice").
  k_max = 4.65 covers the injection band with margin; r* ≈ 52 code units
  ≈ 10.8 Lc; γ calibrated via pilot U_rms for L_γ ≈ 5–7 Lc.
- **D1 — proceeding with recommendation** (conducting free surface: θ′ and Θ̄
  Dirichlet at top, only w goes Neumann-top); insulating top deferred as a
  later experiment. Flag before first long run in case of objection.
- **D2 — proceeding with recommendation** (Route A: per-field mixed
  Chebyshev–Galerkin stencils in the production solver; Route B SBP-SAT FD
  kept as a 128×128 cross-check only).

**Implementation started 2026-07-22.** Live task tracking:
`polar_512_todo.md` (same directory). Keep that file current as milestones
land; this plan document stays frozen as the contract.
