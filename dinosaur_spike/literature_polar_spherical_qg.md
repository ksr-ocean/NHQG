# Literature Notes: Polar Spherical QG And Spectral Pole Issues

Last updated: 2026-05-28

## Executive Summary

The literature does not suggest damping the polar rings if polar dynamics are the
target. It suggests using globally regular spherical formulations, especially
vorticity/divergence or streamfunction/PV forms whose apparently singular metric
terms are grouped into bounded expressions.

The most relevant correction to our current two-layer prototype is that the
classic global shallow-water QG theory on the sphere does not use a constant
deformation term. Its invertibility relation is spheroidal:

`q = laplacian(psi) - epsilon * mu^2 * psi / a^2`, with `mu = sin(latitude)`.

That points toward variable-coefficient/spheroidal PV inversion as the right
route, not a polar sponge.

## Numerical Pole Problem

Swarztrauber (1981) is the key warning. Smooth vector fields in Cartesian space
can look discontinuous in spherical components at the poles. Individual
spherical-coordinate terms can be unbounded even though the full physical
expression is bounded. The numerical cure is to group terms into bounded
expressions and use vector/spherical harmonic machinery rather than evaluating
unbounded pieces separately.

Source:
https://epubs.siam.org/doi/10.1137/0718015

Swarztrauber (1996) compares multiple spectral-transform formulations of the
shallow-water equations on the sphere using a tilted steady zonal flow passing
over the pole. The important result is that algebraically equivalent
formulations can have different numerical behavior; stability and accuracy
depend on how the formulation handles the pole problem and nonlinear aliasing.

Source:
https://journals.ametsoc.org/view/journals/mwre/124/4/1520-0493_1996_124_0730_stmfst_2_0_co_2.xml

Swarztrauber (2004) uses vector spherical harmonics for nonlinear shallow-water
flow and emphasizes bounded forms. This is close in spirit to Dinosaur's native
vorticity/divergence machinery.

Source:
https://journals.ametsoc.org/view/journals/mwre/132/12/mwr2829.1.xml

SPHEREPACK 3.0 is the canonical software expression of these ideas: common
spherical differential operators such as vorticity, divergence, gradient, and
Laplacian are built around scalar/vector spherical harmonic transforms.

Source:
https://journals.ametsoc.org/abstract/journals/mwre/127/8/1520-0493_1999_127_1872_samdf_2.0.co_2.xml

## Global QG On The Sphere

Schubert, Taft, and Silvers (2009) reintroduce shallow-water QG theory on the
full sphere. Their closed system is

`q_t + a^{-2} J(psi, q) + 2 Omega a^{-2} psi_lambda = 0`,

with invertibility

`q = laplacian(psi) - epsilon * mu^2 * psi / a^2`.

They emphasize that the invertibility principle is linear but spheroidal, and
can be handled with spheroidal harmonics. They also derive energy and potential
enstrophy principles. This is probably the strongest clue that our constant-F
global two-layer test is not the right spherical QG analogue.

Source:
https://tropical.colostate.edu/Publications/papers/Schubert_etal_2009a.pdf

Scott (2026) compares nonlinear global QG and shallow-water evolution for
unstable jets, including broad latitudinal motion and order-one Rossby number
cases. The result is encouraging for our ambition: global QG can accurately
represent shallow-water vorticity evolution in strongly nonlinear full-sphere
flows, including midlatitude and equatorial jet cases, despite formal
assumption violations.

Source:
https://research-portal.st-andrews.ac.uk/en/publications/nonlinear-evolution-of-the-global-quasi-geostrophic-system/

## Two-Layer Baroclinic Instability On The Sphere

Moura and Stone (1976) studied spherical geometry effects in a two-layer QG
baroclinic-instability problem. The main lesson is that spherical geometry can
strongly control where unstable waves and eddy fluxes localize.

Source:
https://ntrs.nasa.gov/citations/19760046388

Baines and Frederiksen (1978) studied two-layer spherical baroclinic instability
for QG and modified-geostrophic models. They found that velocity profile choice
matters. In particular, rigid rotation and jet profiles can have very different
unstable-mode structure, and representative midlatitude jet profiles do not
necessarily produce large equatorial flows.

Source:
https://research.monash.edu/en/publications/baroclinic-instability-on-a-sphere-in-twolayer-models/

Paldor, Shamir, and Garfinkel (2020) compare nondivergent, QG, and shallow-water
linear stability for strong polar and equatorial jets. This is relevant because
it treats polar/equatorial jets as active dynamics, not as regions to remove.
It also cautions that QG and nondivergent approximations can bias growth rates
for strong jets.

Source:
https://www.tandfonline.com/doi/full/10.1080/03091929.2020.1724996

## Implications For The Spike

1. Do not polar-sponge the problem away. The correct target is a full-sphere
   formulation with active polar dynamics.

2. Keep using Dinosaur's spherical harmonic machinery, but migrate the model
   closer to native vorticity/divergence/vector-harmonic idioms where possible.
   Our new equivalence test shows the current flux operator matches Dinosaur's
   vorticity-flux construction, so the basic metric form is not obviously wrong.

3. The constant deformation coupling is the main suspect. Global shallow-water
   QG uses a `mu^2` deformation operator and spheroidal inversion. A two-layer
   extension should be derived from the spherical/global QG balance relation,
   not pasted from beta-plane Phillips QG with constant `F`.

4. The next minimal model should be one-layer global QG with spheroidal or exact
   variable-coefficient inversion. Validate it against Rossby-Haurwitz modes,
   an over-pole tilted steady-flow test, and a Scott/Galewsky-style unstable jet
   before returning to two layers.

5. For two layers, derive the spherical analogue explicitly in the formulation
   document before coding. The deformation coupling should probably inherit
   latitude dependence from the global balance relation.

