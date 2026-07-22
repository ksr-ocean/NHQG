# Spherical NHQGE — geometry & discretization options

Working notes for a Jupiter-scale extension of the current Cartesian
NHQGE solver. Aim: solve the rapidly-rotating non-hydrostatic QG system on
a real spherical domain bounded poleward of a midlatitude jet, on multiple
GPUs.

## Why the existing trap-method polar cap isn't enough

`NHGQ_polar.tex` documents the Siegelman–Young–Ingersoll trap-method polar
cap. It projects the cap to a tangent plane and effectively uses a Cartesian
metric with a varying Coriolis parameter. For the science target here
(collaboration with Lia Siegelman and Nick Pizzo) the metric of the sphere
itself is part of the dynamics, so the trap method is no longer accurate at
the scales of interest. We need exact curvature in the operators, not a
linearized approximation.

## Domain

- Spherical zone: jet latitude `phi_jet` to pole, on one or both hemispheres.
- The jet at `phi_jet` is treated, at least initially, as a no-normal-flow
  latitude wall (`psi = const`, `w = 0`). Sponge variant kept as a future
  refinement.
- Vertical: same Chebyshev/CGL structure as the existing code; nothing about
  the rapid-rotation asymptotic forces a different vertical treatment.

## Three honest discretization options

### Option A — Polar disk via exact stereographic projection (currently leading)

Project the cap to a disk via the stereographic map. The map is conformal,
so the induced metric is a single position-dependent factor

```
lambda(r) = 4 / (1 + r^2)^2
```

multiplying the flat-plane Laplacian. The Laplace–Beltrami and Jacobian
become

```
Lap_sphere       = lambda^{-1} Lap_flat
J_sphere[a, b]   = lambda^{-1} J_flat[a, b]
```

and the prognostic equations pick up `lambda` factors that are evaluated
pseudospectrally. Coriolis `f(r)` and `beta(r)` are exact functions of
projected radius. The pole maps to the origin (no coordinate singularity in
the projected coordinates), and the jet maps to a circle of known radius
`r_jet`.

**Discretization fit.**
- Azimuth: Fourier (periodic, trivial FFT, natural sharding axis).
- Radial: Jacobi/Chebyshev with one-sided regularity at `r = 0` and Dirichlet
  / Neumann tau enforcement at `r = r_jet`. Disk regularity per azimuthal
  mode `m` is the standard Jacobi/Zernike construction (Dedalus' DiskBasis
  is the reference implementation to validate against).
- Vertical: Chebyshev (unchanged from the current code).

**Why this is structurally minimal.**
The existing solver's pattern — per horizontal shell IMEX solve, dealiased
Jacobian, Chebyshev tau in `z` — ports almost directly. We replace one
horizontal Fourier direction with Fourier-in-azimuth and the other with
Jacobi-radial. Vertical machinery is untouched. The conformal factor
`lambda` is just a position-dependent multiplier in physical space; it
does not couple modes through the metric.

**Honest costs.**
- Need a working Jacobi radial transform (validated against Dedalus or a
  spectral textbook reference).
- Per-mode dealiasing in the radial direction is its own design choice
  (analog of the 2/3 vs 3/2 rule). Mode-aware quadrature is the safer
  default.
- The `psi`–`q` Helmholtz inversion now has the form
  `(lambda^{-1} Lap_flat + Ld^{-2}(r)) psi = -q`. The conformal factor
  inside the inverted operator means it no longer diagonalises shell-by-shell
  in `(m, radial-mode)` — there is azimuthal-mode-by-azimuthal-mode density
  in the radial direction, which is fine, but the IMEX precompute is now
  a per-`m` dense radial solve rather than a scalar division. Cost is
  O(Nr^3) per `m` once, then O(Nr^2) per step per `m`. Manageable.

### Option B — Spherical harmonics on the hemispheric zone

Use a real SHT (e.g. `s2fft` or vendored from dinosaur) over the zone and
enforce the jet BC via tau rows. The natural global view, but spherical
harmonics are eigenfunctions of the *full*-sphere Laplacian, not of the
operator with Dirichlet BC at `phi_jet`. The BC enforcement is therefore
spectrally awkward and produces pollution near the boundary. SH is the
right tool for a full hemisphere with regularity at the equator; it is
the wrong tool for a zone that stops at an interior latitude.

Probably rules itself out for this problem. Worth keeping in mind only as
a fallback if the no-normal-flow assumption later relaxes to a global
treatment with an equatorial sponge.

### Option C — Lat–lon grid with custom latitude basis

`(phi, lambda)` grid: Fourier in `lambda` (periodic), Chebyshev or Legendre
in `phi` over `[phi_jet, pi/2]`. Pole regularity enforced through the basis
choice (e.g. Chebyshev in `cos(theta)` with a regular endpoint at the pole).
Metric factors (`1/cos phi`, etc.) appear explicitly in every operator —
no single-factor simplification like in A.

Conceptually clean and physically transparent (everything lives in real
geographic coordinates). Downsides:
- More metric bookkeeping than A.
- Pole regularity requires care per zonal mode.
- The momentum-style variables `(u, v)` carry explicit metric coupling,
  rather than collapsing into a streamfunction-vorticity-style scalar
  formulation as cleanly.

A sensible second choice if option A turns out to have an unforeseen
showstopper (e.g. radial transform cost or stability issues).

## Current ranking

A > C >> B. A inherits the most from the existing JAX code, has clean BC
enforcement at the jet, no pole pathology, and exact curvature through one
explicit conformal factor. C is more "geographically intuitive" but more
bookkeeping. B fights the geometry of an interior latitude wall.

## Multi-GPU sharding

Independent of A/B/C, the natural sharding axis is the azimuthal /
longitudinal mode `m`. Per step:

1. FFT in azimuth: shards across `m`, no communication after.
2. Per-`m` radial solve (A) or per-`m` latitude solve (C): embarrassingly
   parallel across `m`.
3. Nonlinear Jacobian: needs a gather / all-to-all to evaluate products in
   physical space, then a re-shard back to spectral. Same pattern dinosaur
   uses, and the same pattern Cartesian 2D-Fourier solvers use when sharded.

Vertical is small (`Nz ~ 256`) and dense; not a viable sharding axis.

### Multiple poles

Two-pole runs are trivially independent shard groups (one GPU pool per
pole, no cross-pole communication). For initial work it is simpler to do
one pole and only later glue a mirrored second pole.

## What to borrow vs. what to build

- **Option A path: mostly from scratch.** No JAX library currently
  provides a sharded disk basis with conformal-factor metric handling.
  Worth borrowing test cases and validation reference from Dedalus 3's
  DiskBasis even though Dedalus itself is not JAX.
- **Option B path: borrow a sharded SHT.** `s2fft` (Cobb et al.) is pure
  JAX and the lighter dependency than dinosaur. Dinosaur's primitive-equations
  layer is not relevant.
- **Option C path: nothing to borrow.** A custom Chebyshev/Legendre
  latitude transform is a small, self-contained component.

## Equation-side questions to settle before discretization

These are physics / collaborator questions, not numerics:

1. **Asymptotic on the disk.** The current NHQGE asymptotic produces a
   `psi`–`q` inversion via `(|k|^2 + Ld^{-2}) psi = -q`. On the disk,
   does the derivation cleanly reduce to
   `(lambda^{-1} Lap_flat + Ld^{-2}(r)) psi = -q`,
   or are there extra correction terms that survive at the order kept by
   the rapid-rotation asymptotic on a curved surface? Confirm with Lia
   and Nick.
2. **Stratification on the cap.** `Ld(r)` may vary across the cap.
   The IMEX precompute already handles per-shell variation; per-`m`
   variation is the analog and is fine.
3. **Mean-temperature exchange.** The current `balanced_sbp2_pc`
   pathway operates on horizontal-mean profiles `(z)`. On the disk, the
   horizontal mean over the projected disk is *not* the same as the
   horizontal mean on the sphere; the area element carries `lambda`.
   The discrete mean-flux exchange has to be defined w.r.t. the spherical
   area measure, not the flat-disk measure, or the global heat budget
   will not close. This needs to be wired in carefully.
4. **Jet boundary detail.** Hard wall vs sponge — start with hard wall;
   keep sponge as a configurable option for later.
5. **Vertical asymptotic vs vertical discretization.** No change expected
   from the Cartesian code — the rapid-rotation asymptotic is geometric
   in the horizontal, and the vertical Chebyshev / Galerkin / tau
   structure carries over unchanged.

## Open questions for the planning step (not now)

- Choice of disk radial basis: Jacobi `(0, m)`, Zernike, or
  Chebyshev-on-`r^2` mapped basis. Each has a different sparsity pattern
  for the radial Laplacian and a different dealiasing rule.
- Time integrator: keep ARS(2,2,2) or move to ARS(4,4,3) given the
  larger per-step cost of the radial dense solves.
- Precision: float64 for development, float32 for production once the
  disk machinery is verified — same posture as the Cartesian code.

## What we are explicitly choosing not to do

- Cubed sphere or Yin-Yang. Natural sharding but writing QG on a cubed
  sphere with exact metric is original work in itself and not justified
  by the science target.
- Hybrid sigma / terrain-following coordinates (the dinosaur path).
  Wrong tool: we are not doing primitive equations and we have no
  topography.
- Equatorial band. Excluded by the rapid-rotation asymptotic; that is
  precisely what makes the polar zone the right domain.
