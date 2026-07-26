# NHQG — Nonhydrostatic Quasi-Geostrophic solver

A GPU-pseudospectral (JAX) solver for the **nonhydrostatic quasi-geostrophic
equations** — rapidly rotating Rayleigh–Bénard convection in the asymptotic
limit `Ek → 0`. The scientific target is **Jupiter's polar vortex dynamics**:
the cyclonic vortex crystals observed by Juno.

Horizontal: doubly-periodic Fourier pseudospectral with dealiased products.
Vertical: Chebyshev — Galerkin (Coral `-T_n + T_{n+2}` stencil) for `w, θ`,
full coefficients for `q'`. Time: ARS(2,2,2) IMEX with unified implicit
dissipation and implicit buoyancy coupling.

> **Status (2026-07-26): the project has been migrated off its original GPU
> host.** The code lives here; the run data lives in a separate archive. Read
> [`HANDOFF.md`](HANDOFF.md) before doing anything else, and
> [`DATA.md`](DATA.md) to find the runs.

## Two research threads

**1. Chebyshev production — reproducing Miquel et al. (2026).**
Stable to `t = 120` at `128² × 256` and `t = 63` at `256² × 256`. Textbook dual
cascade recovered from ghost-clean radial budgets (inverse energy flux
`Π_E = −1309`, forward enstrophy flux `Π_Z = +771`).
*Open problem:* the **Nusselt gap** — we get `Nu_d ≈ 18–20` at `Ra = 100`
against Miquel's `43.37 ± 2.54`.

**2. Polar-cap campaign — SYI22 trap (current work).**
The trap method of Siegelman, Young & Ingersoll (2022) confines turbulence to a
polar cap via a radial background-PV gradient, now with an SYI22-style Rayleigh
sponge outside the trap. **Result: the trap arrests the barotropic condensate.**
Measured inside the trap radius, where the sponge is inactive, over `t = 30 → 70`
from a common initial state:

| | γ = 0 control | trap + sponge |
| --- | --- | --- |
| cap velocity `U` | 17.0 → **70.6** | 16.9 → **18.3** |
| eddy scale `U/ζ` | 2.35 → **7.01** | 2.09 → **2.37** |

`U` pins at the design point (`γ = 2.2e-3` was calibrated for `L_γ = 4 L_c`;
the cap settles at `L_γ = 3.95–4.21 L_c`). The run reached `t = 158.2`.
Vortex-crystal segregation had **not** appeared by then — cap vorticity skewness
stayed at `+0.05 ± 0.05`.

Plan of record: [`polar_512_plan.md`](polar_512_plan.md) (frozen contract).
Live tracker: [`polar_512_todo.md`](polar_512_todo.md).

## Quick start

```bash
export JAX_ENABLE_X64=1 PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # see note

# tests (CPU, ~2 min)
JAX_PLATFORMS=cpu python -m pytest tests/ -q

# a small polar run
python scripts/run_polar.py --Nx 256 --Nz 32 --L-over-Lc 24 --gamma 2.2e-3 \
    --Ra 100 --dt 5e-5 --t-final 1 --w-bc-top neumann \
    --thermal-closure evolve_mean --mean-exchange balanced_sbp2_pc \
    --sbp-substeps 4 --advection flux --dealias 23_rule \
    --output-dir output/demo
```

`JAX_ENABLE_X64=1` is **required** — production runs are float64 and the
dataclass default is `float32`. Pinning the BLAS thread counts to 1 avoids long
host-side stalls during IMEX-shell precomputation.

## Repository map

| path | what |
| --- | --- |
| `nhqg/` | the solver — `config`, `grid`, `spectral`, `solver`, `diagnostics`, `io`, `polar_diagnostics`, `sharding` |
| `scripts/` | drivers (`run_polar.py`, `run_rubio.py`, `run_sweep.py`), analysis and rendering tools, the archive builder |
| `tests/` | 151 tests, CPU-runnable |
| `analysis/` | committed analysis products (spectra, spectral budgets, polar summaries) |
| `fd_vertical_benchmark/` | separate FD-in-Z solver (Route B toward mixed BCs) |
| `trig_vertical_benchmark/`, `dedalus_benchmark_standard_f/` | independent validation branches |
| `sphere/`, `sphere_nhqg/` | spherical-geometry prototypes (exploratory) |
| `*.tex` | the formulation documents; build with `tectonic` |

## Documents worth reading first

- **`CLAUDE.md`** — the long-form project record: formulation choices, every
  investigation, status sections by date. Start here for context.
- **`HANDOFF.md`** — environment, data staging, and how to resume the campaign.
- **`DATA.md`** — what is in the run archive and how it maps to this repo.
- **`hermitian_ghost.md`** — the anti-Hermitian `ky=0` ghost mode. **Read before
  trusting any raw spectral diagnostic.** The fix is diagnosed but *not yet
  implemented*.
- **`NHQG_framework_deck.tex`** — 41-slide pedagogical walkthrough.
- `blowup.md`, `spectral_analysis.md`, `adjoint_mean_exchange.md` — historical
  investigation records, each with a status banner at the top.

## Known open items

1. **Nusselt gap** — `Nu_d ≈ 18–20` vs Miquel's `43.37`. Leading suspects: the
   piecewise-linear CGL↔SBP transfer smoothing the thermal boundary layer;
   `Nx=64` heritage under the 2/3 rule; effective horizontal resolution.
2. **Vertical Chebyshev tail** — `q` has no vertical dissipation, so an undamped
   vertical enstrophy cascade piles energy at the top modes. In the P2 run the
   ψ-KE fraction above vertical mode 48 (of 64) went `0.30 → 0.53` over
   `t = 31 → 70`. The *resolved* flow is unaffected — `max_speed` restricted to
   modes `n ≤ 32` was 142.5 at `t=31` and 142.1 at `t=70` — but the tail drives
   the reported `max_speed` up and will eventually break ARS222. This is the
   binding constraint on long integrations.
3. **Hermitian ghost fix** — symmetrize the `ky=0` column each step, use
   Hermitian initial noise, add a regression test. Cheap; not yet done.
4. **ARS222 advective limit** — keep `max_speed · k_max · dt ≲ 0.1`. The γ=0
   pilot died at `≈ 0.22`.

## Reference

Miquel et al. 2026 (arXiv) is the comparison paper; the built PDF ships with the
data archive (`repo_artifacts/`) rather than git. Siegelman, Young & Ingersoll
(2022) is the source of the trap method — see `NHGQ_polar.tex`.
