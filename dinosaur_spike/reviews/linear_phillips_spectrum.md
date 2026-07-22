# dinosaur_spike/linear_phillips_spectrum.py

This script builds the exact linearized spherical two-layer QG operator about a
zonal base state and computes its unstable modes one longitudinal wavenumber at
a time.

The linearized perturbation equations are
`dq_i'/dt = -J(psi_i', Q_i^0) - J(Psi_i^0, q_i')`, using the same Dinosaur
flux-divergence Jacobian as the nonlinear model. By default the script omits
the nonlinear `J(psi_i', q_i')`, sponge, and mask so the computed eigenvalues
describe the base state itself. It can also include the same smooth latitude
tendency mask and sponge for operator-level spike tests.

Because the base state is zonal, each absolute zonal wavenumber `m` is an
independent block. The script packs only valid modal entries from the `+m` and
`-m` real-spherical-harmonic rows for both layers, applies the linear RHS to
basis vectors, forms a dense block matrix, and computes its eigenvalues with
NumPy. This is only intended for low truncations such as `w=31` or `w=63`.

For latitude-dependent deformation profiles, the script builds the exact dense
PV-from-streamfunction matrix for the `m` block and inverts that matrix before
applying the linear tendency. This lets the spike test `f`-regularized
deformation operators without pretending they are still block-diagonal.

The optional `--save-mode` path writes the real part of the most unstable
eigenvector as a modal `q1,q2` checkpoint scaled to `--amplitude`. That file can
be used as `--restart-state` in `run_two_layer_solution.py` to verify that the
full nonlinear code grows initially at the predicted linear rate.

The important diagnostic lesson is that the original `solid_body` profile is
linearly neutral in this exact spherical check, while the regular
`sin_plus_sin3` profile produces positive Phillips-type baroclinic growth.

Recent regularization checks compare `constant`, `f_squared_floor`, and
`inverse_f_squared_floor` deformation profiles, with and without the southern
mask/sponge. The `f_squared_floor` plus mask/sponge case is the current
candidate for the equator-regularized spike path.
