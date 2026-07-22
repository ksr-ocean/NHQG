Formulation Notes

Goal:
- Independent spectral benchmark for the upright standard-`f` NHQG case
- Match the JAX benchmark as closely as practical without reusing the same
  Chebyshev/Galerkin implementation choices

Target equations in the JAX benchmark split:

`q_t = -J(psi,q) + dz(w) + nu_q * lap_h(q)`

`w_t = -J(psi,w) - dz(psi) + (Ra/sigma) theta + nu_w * lap_h(w)`

`theta_t = -J(psi,theta) + w - dz(Theta_bar') * w + (nu_theta/sigma) * lap_h(theta)`

`eps^-2 Theta_bar'_t = -dz(<w theta>_xy) + sigma^-1 dz(dz(Theta_bar'))`

with

`q = -lap_h(psi)`

where `lap_h = dxx + dyy`, `u = -psi_y`, `v = psi_x`, and
`J(psi,f) = u f_x + v f_y`.

Dedalus representation used in the scaffold:

Variables:
- `q(x,y,z,t)`
- `psi(x,y,z,t)`
- `wt(x,y,z,t)` with `w = S(z) * wt`
- `tt(x,y,z,t)` with `theta = S(z) * tt`
- `mt(z,t)` with `Theta_bar' = S(z) * mt`

Boundary encoding:
- `S(z) = z (1-z)`
- This makes `w = 0`, `theta = 0`, and `Theta_bar' = 0` automatically at
  `z=0,1`, without requiring a separate tau formulation for fields that have
  no vertical diffusion in their fluctuation equations.

Zero horizontal mode:
- The JAX code explicitly zeros the `(kx,ky)=(0,0)` mode of the fluctuation
  fields after every step.
- The Dedalus scaffold mimics that by only evolving the nonzero horizontal
  modes and imposing:
  - `q = 0`
  - `psi = 0`
  - `wt = 0`
  - `tt = 0`
  for the zero horizontal mode.

Why not a tau-heavy formulation:
- `w` and `theta` carry Dirichlet boundary conditions but no vertical
  fluctuation diffusion.
- That makes a traditional Chebyshev-tau construction awkward and easy to get
  wrong.
- The prefactor method gives a simpler independent benchmark and is easier to
  inspect term by term.

Open issues to check when Dedalus is available:
1. Whether the mode-conditioned equations using `condition="(nx == 0) and (ny == 0)"` behave as intended in the mixed Fourier/Chebyshev domain.
2. Whether the `Average` / `Integrate` operators in the mean-temperature
   equation and analysis tasks parse exactly as written.
3. Whether the zero-mode constraints should be promoted to all fluctuation
   quantities including any derived fields saved to disk.
4. Whether `RK222` is the best Dedalus IMEX choice for this benchmark, or
   whether a multistep IMEX scheme would be closer to the JAX run protocol.
