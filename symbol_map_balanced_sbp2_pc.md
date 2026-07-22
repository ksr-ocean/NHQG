# Symbol Map For `balanced_sbp2_pc`

This note is a compact dictionary from the mathematical symbols in
`discretely_balanced_mean_fluctuation_thermal_formulation.tex` to the current
`balanced_sbp2_pc` implementation in `nhqg/solver.py` and `nhqg/grid.py`.

It is intentionally narrow:
- it describes the current `balanced_sbp2_pc` branch only
- it ignores the older `balanced_midpoint` and split `balanced_sbp2` branches
- it is meant to be read side by side with the predictor/corrector section of
  `discretely_balanced_mean_fluctuation_thermal_formulation.tex`

## Minimal call chain

The active timestep path is:

`imex_step`
-> `imex_step_balanced_sbp2_pc`
-> `balanced_sbp2_thermal_substep`

Code locations:
- `nhqg/solver.py`: `imex_step` dispatch
- `nhqg/solver.py`: `imex_step_balanced_sbp2_pc`
- `nhqg/solver.py`: `balanced_sbp2_thermal_substep`

## Full-system symbols

| Math symbol | Meaning in the note | Code object |
| --- | --- | --- |
| `y=(q',w,\theta,\Theta)` | full prognostic state | `State(q_hat, w_hat, th_hat, th_bar)` |
| `y^n` | state at start of step | `state` on entry to `imex_step_balanced_sbp2_pc` |
| `q'` | PV perturbation | `state.q_hat` |
| `w` | vertical velocity | `state.w_hat` |
| `\theta` | fluctuation temperature | `state.th_hat` |
| `\Theta` | mean temperature deviation | `state.th_bar` |
| `\Delta t` | full timestep | `grid.dt` |
| `\gamma` | ARS222 diagonal coefficient | `grid.gamma_imex` |
| `\delta` | ARS222 explicit coefficient | `-jnp.sqrt(2.0)/2.0` in `imex_step_balanced_sbp2_pc` |
| `\alpha=\gamma\Delta t` | local stage-corrector size | `alpha = gamma * dt` |

## Operator split symbols

| Math symbol | Meaning in the note | Code object |
| --- | --- | --- |
| `\mathcal E(y)` | explicit horizontal terms | `explicit_rhs_dispatch(state, grid)` |
| `\mathcal I(y)` | stiff linear implicit terms | `implicit_tendency(state, grid)` plus `imex_implicit_solve` / `imex_mean_temp_solve` |
| `\mathcal C(y)` | mean-fluctuation thermal correction | represented by `balanced_sbp2_thermal_substep`, not inserted as a raw RHS |
| `\mathcal E_{\mathrm{red}}` | predictor explicit operator with exchange removed | `explicit_rhs_dispatch(state, base_grid)` |
| `\mathcal I_{\mathrm{red}}` | predictor implicit operator with exchange removed | `implicit_tendency(state, base_grid)` |
| `\mathcal P_\alpha` | stage-local thermal corrector map | `balanced_sbp2_thermal_substep(..., sub_dt=alpha)` |

Here `base_grid` means:

```python
base_grid = grid._replace(
    thermal_closure="fixed_conduction",
    mean_exchange_discretization="legacy",
)
```

This is the concrete implementation of the reduced predictor system.

## Stage symbols

| Math symbol | Meaning in the note | Code object |
| --- | --- | --- |
| `\widetilde Y_1` | stage-1 reduced predictor | `predictor1` |
| `Y_1` | stage-1 corrected state | `state1` |
| `\widetilde Y_2` | stage-2 reduced predictor | `predictor2` |
| `Y_2=y^{n+1}` | final corrected state | `corrected`, then `_finalize_state(corrected, grid)` |
| `\mathcal C_1^{\mathrm{eff}}` | effective stage-1 thermal correction tendency | `C1 = _thermal_correction_tendency(predictor1, state1, alpha)` |

## Stage 1 map

The note writes

```tex
\widetilde Y_1
=
y^n + \gamma \Delta t\,\mathcal E_{\mathrm{red}}(y^n)
+ \gamma \Delta t\,\mathcal I_{\mathrm{red}}(\widetilde Y_1).
```

This is implemented as:

- `E1 = explicit_rhs_dispatch(state, base_grid)`
- `R_q1`, `R_w1`, `R_th1`, `R_th_bar1`
- `q1p, w1p, th1p = imex_implicit_solve(...)`
- `th_bar1p = imex_mean_temp_solve(...)`
- `predictor1 = State(q1p, w1p, th1p, th_bar1p)`

The note then writes

```tex
Y_1 = \mathcal P_{\alpha_1}(\widetilde Y_1).
```

This is:

- `state1 = balanced_sbp2_thermal_substep(predictor1, grid, sub_dt=alpha)`

The effective correction

```tex
\mathcal C_1^{\mathrm{eff}}
=
\left(0,0,\frac{\theta_1-\widetilde\theta_1}{\alpha_1},
\frac{\Theta_1-\widetilde\Theta_1}{\alpha_1}\right)
```

is:

- `_thermal_correction_tendency(predictor1, state1, alpha)`

## Stage 2 map

The note writes

```tex
\widetilde Y_2
= y^n
+ \Delta t[\delta \mathcal E_{\mathrm{red}}(y^n)
+ (1-\delta)\mathcal E_{\mathrm{red}}(Y_1)]
+ (1-\gamma)\Delta t\,\mathcal I_{\mathrm{red}}(Y_1)
+ (1-\gamma)\Delta t\,\mathcal C_1^{\mathrm{eff}}
+ \gamma \Delta t\,\mathcal I_{\mathrm{red}}(\widetilde Y_2).
```

This is implemented as:

- `E2 = explicit_rhs_dispatch(state1, base_grid)`
- `I1 = implicit_tendency(state1, base_grid)`
- `R_q2`, `R_w2`, `R_th2`, `R_th_bar2`
- `q2p, w2p, th2p = imex_implicit_solve(...)`
- `th_bar2p = imex_mean_temp_solve(...)`
- `predictor2 = State(q2p, w2p, th2p, th_bar2p)`

The explicit insertion of the stage-1 thermal correction is:

- `+ omg * C1.th_hat`
- `+ omg * C1.th_bar`

where `omg = dt * (1 - gamma)`.

The final corrector

```tex
y^{n+1} = \mathcal P_{\alpha_2}(\widetilde Y_2)
```

is:

- `corrected = balanced_sbp2_thermal_substep(predictor2, grid, sub_dt=alpha)`

## Thermal corrector symbols

The balanced SBP2 thermal substep is the code realization of the note's
stage-local map `\mathcal P_\alpha`.

| Math symbol | Meaning in the note | Code object |
| --- | --- | --- |
| `w^\star` | frozen stage vertical velocity on the thermal substep | `w_sbp` |
| `\theta^\star` | predictor fluctuation temperature on SBP grid | `th_sbp_n` |
| `\Theta^\star` | predictor mean temperature on SBP grid | `th_bar_sbp_n` |
| `F^\star=\langle w^\star\theta^\star\rangle_{xy}` | stage heat-flux profile | `flux_n` |
| `M^\star=\mathrm{diag}(\langle (w^\star)^2\rangle_{xy})` | diagonal multiplier from stage velocity | `M = jnp.diag(w2_mean)` |
| `D_1` | SBP first-derivative operator | `grid.sbp_D1` |
| `L` | SBP Laplacian used for mean diffusion | `grid.sbp_L` |
| `\mu` | mean temperature prefactor | `grid.mean_temp_eps_sq` |
| `\kappa_\Theta` | mean thermal diffusivity factor | `1.0 / grid.sigma` |
| `\Theta^+` | corrected mean temperature on SBP grid | `th_bar_sbp_new` |
| `\theta^+` | corrected fluctuation temperature on SBP grid | `th_sbp_new` |

The code block is:

- construct SBP-grid predictor data with `sbp2_exchange_state_fields`
- form `flux_n` and `w2_mean`
- build `A`, `B`, and `rhs`
- solve for `th_bar_sbp_new`
- form `g_half = 0.5 * D1 @ (Theta^\star + Theta^+)`
- update `th_sbp_new = theta^\star - alpha * w^\star * g_half`

## Representation symbols

| Symbol / phrase | Meaning | Code object |
| --- | --- | --- |
| CGL grid | solver's native Chebyshev-Gauss-Lobatto nodal grid | `grid.Z`, `grid.V`, `grid.V_inv` |
| uniform SBP grid | auxiliary vertical work grid for the thermal corrector | `grid.Z_sbp` |
| CGL to SBP transfer | nodal interpolation from solver grid to SBP work grid | `grid.cgl_to_sbp` |
| SBP to CGL transfer | nodal interpolation back to solver grid | `grid.sbp_to_cgl` |

These are built in `nhqg/grid.py`:

- `sbp_H`, `sbp_D1`, `sbp_L` from `_sbp2_operators`
- `cgl_to_sbp` and `sbp_to_cgl` from `_piecewise_linear_interp_matrix`

## Horizontal operators that remain unchanged

The `balanced_sbp2_pc` branch does not change the horizontal pseudospectral
machinery. The following are inherited from the base solver:

- streamfunction inversion: `invert_psi`
- horizontal nonlinear advection: `_triple_horizontal_advection`
- 3/2 dealiased physical-space averages: `horizontal_mean_from_nodal_spectral`
- shell-deduplicated IMEX `q`-`w` block solve: `_build_imex_inv` and `imex_implicit_solve`

## What to ignore while reading this branch

If the goal is to understand the current experiment, ignore these older paths:

- `balanced_midpoint`
- split `balanced_sbp2`
- Coral work-grid exchange variants

The active implementation is only:

- `mean_exchange_discretization="balanced_sbp2_pc"`

## Recommended side-by-side reading order

1. `discretely_balanced_mean_fluctuation_thermal_formulation.tex`
   Read the section `Full-system predictor--corrector formulation`.
2. `nhqg/solver.py`
   Read `imex_step_balanced_sbp2_pc`.
3. `discretely_balanced_mean_fluctuation_thermal_formulation.tex`
   Read the subsection `What the thermal corrector actually solves`.
4. `nhqg/solver.py`
   Read `balanced_sbp2_thermal_substep`.
5. `nhqg/grid.py`
   Read `_sbp2_operators` and the SBP transfer construction in `make_grid`.

## Minimal verification hook

The lightweight regression test for this path is:

- `tests/test_solver.py::TestSolver::test_balanced_sbp2_pc_mode_runs_finite`

This is only a smoke test. It verifies that the branch runs a few small steps
without producing non-finite values. It is not a structural proof.
