#!/usr/bin/env python
"""Stage every NHQG run directory into one documented archive tree.

The archive is built with **hardlinks** (`os.link`), so it occupies no extra
disk and is created in seconds -- `output/` and the archive point at the same
inodes.  Deleting either side is safe; the data survives until both are gone.

Layout::

    <archive>/
      README.md          generated -- one section per run, keyed to the repo docs
      MANIFEST.tsv       generated -- one row per run (files, bytes, t-range)
      polar/             the SYI22 polar-vortex campaign (current work)
      chebyshev/         Miquel reproduction, Nusselt gap, dual-cascade budgets
      fd_vertical/       Route-B finite-difference vertical benchmark
      exploratory/       spikes, smokes and one-offs kept for provenance
      repo_artifacts/    gitignored repo outputs (PDFs, movies, frame stacks)

Usage::

    python scripts/build_data_archive.py --dest ../NHQG_runs_archive_2026-07
    python scripts/build_data_archive.py --dest ... --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Run classification.
#
# (source directory under output/, archive-relative destination, description)
#
# The description is what lands in README.md, so it says what the run IS and
# which repo document depends on it -- not just what the flags were.
# ---------------------------------------------------------------------------

RUNS: list[tuple[str, str, str]] = [
    # ---------------------------------------------------------------- polar --
    (
        "polar_p2_opentop_trap_sponge_Nx512_Nz64_L48",
        "polar/p2_trap_sponge_512x64_t30_t158",
        "**The P2 result run.** Convectively forced polar cap with the SYI22 "
        "trap (gamma=2.2e-3, r*=92.45=19.2 Lc) plus the Rayleigh sponge "
        "(sigma_max=50, r_s=104, A_s=30). Restarted from the gamma=0 pilot at "
        "t=30 and integrated to t=158.2 before the host was withdrawn; the run "
        "was healthy at the kill (max_speed 260, KE_bt 101). Open top "
        "(w_bc_top=neumann), evolve_mean, balanced_sbp2_pc + 4 substeps, flux "
        "advection, 23_rule, dt=5e-5. This is the run that shows the trap "
        "arrests the barotropic condensate; see polar_512_todo.md (M5/P2).",
    ),
    (
        "polar_m4_pilot_opentop_evolvemean_Nx512_Nz64_L48_Ra100_dt5e5",
        "polar/pilot_gamma0_512x64_t0_t97",
        "**The gamma=0 control** for P2, and the source of P2's t=30 restart "
        "state (checkpoint_00600000.npz). Same physics with no trap and no "
        "sponge. Ran t=0 to t=97.25, where it died NON-FINITE: with nothing to "
        "arrest it the condensate grew unchecked until ARS222 lost the "
        "advective imaginary axis (max_speed*k_max*dt ~ 0.22). That death is "
        "the origin of the operational rule max_speed*k_max*dt <= 0.1. "
        "Documented in polar_512_todo.md (M4).",
    ),
    (
        "polar_p0c_g1.25e-3_Nx1024_L24_ki12.6",
        "polar/p0c_sweep_1024_barotropic/gamma_1.25e-3",
        "P0c cap-capacity sweep, low-gamma leg (1024^2 barotropic, L=24 Lc, "
        "monoscale cap-confined init at k_i=12.6). Realized L_gamma=4.2, "
        "r*/L_gamma=6 -> vortex-gas end state at t=400.",
    ),
    (
        "polar_p0c_g5e-3_Nx1024_L24_ki12.6",
        "polar/p0c_sweep_1024_barotropic/gamma_5e-3",
        "P0c mid-gamma leg. L_gamma=2.7, r*/L_gamma=10 -> ring plus central "
        "cyclone.",
    ),
    (
        "polar_p0c_g2e-2_Nx1024_L24_ki12.6",
        "polar/p0c_sweep_1024_barotropic/gamma_2e-2",
        "P0c high-gamma leg. L_gamma=1.7, r*/L_gamma=16 -> zonation. Together "
        "the three legs establish that r*/L_gamma governs the end-state "
        "morphology, which set P2's design point (r*/L_gamma=4.8).",
    ),
    (
        "polar_p0_bt_g0.005_Nx512_Nz8_L48",
        "polar/p0_trap_512_barotropic/gamma_0.005",
        "P0 barotropic trap validation, decaying turbulence in the trap.",
    ),
    (
        "polar_p0_bt_g0.02_Nx512_Nz8_L48",
        "polar/p0_trap_512_barotropic/gamma_0.02",
        "P0 barotropic trap validation, stronger trap.",
    ),
    (
        "polar_p0_bt_g0.08_Nx512_Nz8_L48_ars222_blowup",
        "polar/p0_trap_512_barotropic/gamma_0.08_ars222_blowup",
        "**Negative result, kept deliberately.** At gamma=0.08 the trap edge "
        "excites topographic waves that ARS222 cannot hold -- the origin of the "
        "ars222 trap-edge instability note. Compare with the rk443 leg below.",
    ),
    (
        "polar_p0_bt_g0.08_Nx512_Nz8_L48_rk443_decay_t159",
        "polar/p0_trap_512_barotropic/gamma_0.08_rk443_t159",
        "The same gamma=0.08 case under RK443, which is stable to t=159. "
        "Establishes that sharp traps need rk443 (or a capped omega_edge*dt).",
    ),
    (
        "polar_p0f_pilot_g0.02_inj1_A6_drag02",
        "polar/p0f_forced_barotropic/pilot_g0.02_inj1_drag0.2",
        "P0F forced-barotropic pilot (stochastic injection + linear drag).",
    ),
    (
        "polar_p0f_g0.02_Nx512_inj1_A6_drag02",
        "polar/p0f_forced_barotropic/g0.02_inj1_drag0.2",
        "P0F forced barotropic, gamma=0.02.",
    ),
    (
        "polar_p0f_g0.005_Nx512_inj1_A6_drag02",
        "polar/p0f_forced_barotropic/g0.005_inj1_drag0.2",
        "P0F forced barotropic, gamma=0.005.",
    ),
    (
        "polar_p0f2_g0.02_inj02_A6_drag01",
        "polar/p0f_forced_barotropic/g0.02_inj0.2_drag0.1",
        "P0F injection/drag variation.",
    ),
    (
        "polar_p0f3_g0.02_inj04_A6_drag02",
        "polar/p0f_forced_barotropic/g0.02_inj0.4_drag0.2",
        "P0F injection/drag variation.",
    ),
    (
        "polar_p0f4_g0.02_inj1_A6_drag005",
        "polar/p0f_forced_barotropic/g0.02_inj1_drag0.05",
        "P0F injection/drag variation. The P0F family is a post-mortem branch: "
        "forced barotropic runs did not produce crystals, which is why the "
        "campaign moved to the SYI22-faithful decaying/confined P0c setup.",
    ),
    # ------------------------------------------------------------ chebyshev --
    (
        "output_combined_Nx128_Nz256_t40_to_t70_sub4_23rule_stack13_snap01",
        "chebyshev/production_128x256_t40_t120",
        "**Primary production chain.** 128^2 x 256, clean to t=120, upsampled "
        "from the 64^2 state at t=40. Nu_d ~ 19-22, exchange residual at "
        "roundoff. This is the source for the ghost-clean radial spectral "
        "budgets in analysis/spectral_budget/window_t80_t120/ and the "
        "dual-cascade result in CLAUDE.md section 3b.",
    ),
    (
        "output_combined_Nx256_Nz256_t41_to_t58_sub4_23rule_stack13_snap01",
        "chebyshev/production_256x256_t41_t63",
        "**Highest-resolution production chain.** 256^2 x 256, clean to t=63. "
        "The cleanest window by all three spectral convergence criteria "
        "(analysis/spectral_budget/n256_*). Vertical spectrum still smooth, "
        "ghost only ~3e7.",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_subcycle4_fromstart_Nx64_Nz256_dt5e5_t80_snap01",
        "chebyshev/baseline_64x256_t0_t80",
        "The from-start 64^2 x 256 baseline to t=80 -- the movie/visualisation "
        "archive and the parent of the 128^2 chain.",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_subcycle4_continue_from_t80_Nx64_Nz256_dt5e5_t120_snap01",
        "chebyshev/baseline_64x256_t80_t120",
        "Continuation of the 64^2 baseline to t=120.",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_subcycle4_probe_t80_from_t20_Nx64_Nz256_dt5e5",
        "chebyshev/baseline_64x256_probe_t20_t80",
        "**The documented balanced_sbp2_pc baseline** quoted in CLAUDE.md "
        "(Nusselt_dealiased = 19.5177, max_w = 262.46 at t=80).",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_evolvemean_kebudget_blas1_Nx128_Nz128_dt5e5_t8",
        "chebyshev/spectral_budget_source_128x128_t8",
        "The 128^2 x 128 shell-budget archive behind the original "
        "dominant-shell analysis. NOTE: its k=0.9786 conclusion is "
        "ghost-contaminated -- see hermitian_ghost.md.",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_evolvemean_kebudget_blas1_Nx64_Nz256_dt5e5_t8",
        "chebyshev/spectral_budget_source_64x256_t8",
        "The matched 64^2 x 256 shell-budget comparison archive. Same "
        "ghost caveat.",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2_flux_fromstart_Nx64_Nz256_dt5e5_t20",
        "chebyshev/branch_comparison/balanced_sbp2_fromstart_t20",
        "Mean-exchange branch comparison: balanced_sbp2 (pre-corrector).",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2_flux_continue_from_t20_Nx64_Nz256_dt5e5_t110",
        "chebyshev/branch_comparison/balanced_sbp2_t20_t110",
        "balanced_sbp2 continuation to t=110.",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_continue_from_t42_Nx64_Nz256_dt5e5_t80",
        "chebyshev/branch_comparison/balanced_sbp2_pc_t42_t80",
        "balanced_sbp2_pc continuation leg.",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_probe_t420_from_t20_Nx64_Nz256_dt5e5",
        "chebyshev/branch_comparison/balanced_sbp2_pc_long_probe_from_t20",
        "Long balanced_sbp2_pc probe from t=20.",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedmidpoint_flux_fromstart_Nx64_Nz256_dt5e5_t11",
        "chebyshev/branch_comparison/balanced_midpoint_fromstart_t11",
        "balanced_midpoint branch -- the least stable of the family "
        "(full-start failure window t=10.4-10.6). Kept as the negative "
        "control for adjoint_mean_exchange.md.",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_evolvemean_coralworkgrid_flux_dense_t114_from_t10_Nx64_Nz256_dt5e5",
        "chebyshev/branch_comparison/coral_workgrid_flux_t10_t114",
        "coral_workgrid_flux branch (first non-finite at t=11.44 in the old "
        "crash-window comparison).",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_evolvemean_coralworkgrid_flux_probe_t1146_from_t114_Nx64_Nz256_dt5e5",
        "chebyshev/branch_comparison/coral_workgrid_flux_probe_t114",
        "coral_workgrid_flux short probe.",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2_flux_probe_t104_from_t10_Nx64_Nz256_dt5e5",
        "chebyshev/branch_comparison/probes/balanced_sbp2_t10_t104",
        "Crash-window probe (balanced_sbp2).",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2_flux_probe_t106_from_t104_Nx64_Nz256_dt5e5",
        "chebyshev/branch_comparison/probes/balanced_sbp2_t104_t106",
        "Crash-window probe (balanced_sbp2).",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2_flux_probe_t114_from_t106_Nx64_Nz256_dt5e5",
        "chebyshev/branch_comparison/probes/balanced_sbp2_t106_t114",
        "Crash-window probe (balanced_sbp2).",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2_flux_probe_t120_from_t114_Nx64_Nz256_dt5e5",
        "chebyshev/branch_comparison/probes/balanced_sbp2_t114_t120",
        "Crash-window probe (balanced_sbp2).",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_probe_t120_from_t10_Nx64_Nz256_dt5e5",
        "chebyshev/branch_comparison/probes/balanced_sbp2_pc_t10_t120",
        "Crash-window probe (balanced_sbp2_pc).",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_subcycle4_probe_t50_from_t40_Nx64_Nz256_dt5e5",
        "chebyshev/branch_comparison/probes/balanced_sbp2_pc_sub4_t40_t50",
        "Subcycle-4 probe.",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_evolvemean_balancedsbp2pc_flux_subcycle4_probe_t50_from_t42_Nx64_Nz256_dt5e5",
        "chebyshev/branch_comparison/probes/balanced_sbp2_pc_sub4_t42_t50",
        "Subcycle-4 probe.",
    ),
    (
        "output_miquel_zero_tilt_evolvemean_flux_balancedsbp2pc_subcycle4_fromstart_Nx128_Nz256_dt5e5_t10",
        "chebyshev/resolution_ladder/128x256_fromstart_t10",
        "128^2 x 256 from-start leg.",
    ),
    (
        "output_run_Nx128_Nz128_t40_to_t50_sub4_23rule_stack13_snap01_GPU5",
        "chebyshev/resolution_ladder/128x128_t40_t50",
        "128^2 x 128 comparison leg.",
    ),
    (
        "output_continue_from_upsampled_Nx256_Nz256_from128_t10_to_t11",
        "chebyshev/resolution_ladder/256x256_upsampled_t10_t11",
        "First 256^2 leg, upsampled from 128^2.",
    ),
    (
        "output_continue_Nx256_Nz256_from_t11_to_t12_23rule",
        "chebyshev/resolution_ladder/256x256_t11_t12",
        "256^2 continuation leg.",
    ),
    (
        "output_continue_Nx256_Nz256_t12_t13_stack123_sub2_23rule",
        "chebyshev/resolution_ladder/256x256_t12_t13",
        "256^2 continuation leg.",
    ),
    (
        "output_smoke_Nx256_from_64_t40_to_t41_sub4_23rule_stack13",
        "chebyshev/resolution_ladder/256x256_smoke_t40_t41",
        "256^2 smoke leg from the 64^2 state.",
    ),
    (
        "output_smoketest_Nx256_Nz256_balancedsbp2pc_subcycle4_t1",
        "chebyshev/resolution_ladder/256x256_smoketest_t1",
        "256^2 smoke test.",
    ),
    (
        "upsampled_checkpoints",
        "chebyshev/resolution_ladder/upsampled_checkpoints",
        "Upsampled restart states used to seed the higher-resolution chains.",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_Nx128_Nz128_dt5e5_t5",
        "chebyshev/early_galerkin/128x128_t5_control",
        "Early Coral-style Galerkin control (fixed_conduction era), ran away by "
        "t=5. Historical: predates evolve_mean and the mean-exchange work.",
    ),
    (
        "output_miquel_zero_tilt_galerkin_ars222_vdeal2x_Nx128_Nz128_dt5e5_t5",
        "chebyshev/early_galerkin/128x128_t5_vertical_dealias_2x",
        "**Negative result.** The matched cheb_2x vertical-dealiasing run "
        "tracked the control essentially identically -- vertical dealiasing was "
        "not the missing stabilisation. Recorded in CLAUDE.md.",
    ),
    (
        "output_test_Nx128_Nz256_t40_to_t45_sub4_23rule_cheb2x_GPU7",
        "chebyshev/dealiasing_tests/128x256_cheb2x_t40_t45",
        "cheb_2x vertical dealiasing at production resolution.",
    ),
    (
        "output_test_Nx128_Nz256_t40_t45_sub4_23rule_hyper3_nu4.6e3_GPU5",
        "chebyshev/dealiasing_tests/128x256_hyper3_nu4.6e3",
        "Hyperviscosity (p=3) sensitivity test.",
    ),
    (
        "output_test_Nx128_Nz256_t40_t41_sub4_23rule_hyper3_nu1e3_GPU5",
        "chebyshev/dealiasing_tests/128x256_hyper3_nu1e3",
        "Hyperviscosity (p=3) sensitivity test, immediate failure.",
    ),
    (
        "postprocess_checkpoint_spectra_matched_closure",
        "chebyshev/postprocess_matched_closure",
        "Postprocessing scratch for the matched-closure spectra.",
    ),
    # ----------------------------------------------------------- fd_vertical --
    (
        "fd_sbp42_fixedcond_Nx64_Nz256_t5",
        "fd_vertical/sbp42_fixedcond_t5",
        "FD-vertical (SBP42) with fixed_conduction.",
    ),
    (
        "fd_sbp42_evolvemean_Nx64_Nz256_t5",
        "fd_vertical/sbp42_evolvemean_t5",
        "FD-vertical (SBP42) with evolve_mean.",
    ),
    (
        "fd_compact4_fixedcond_Nx64_Nz256_t5",
        "fd_vertical/compact4_fixedcond_t5",
        "FD-vertical (compact4) with fixed_conduction. The compact-Pade path is "
        "where the FD instability lives; sbp42's operator is clean.",
    ),
    (
        "fd_compact4_evolvemean_Nx64_Nz256_t5",
        "fd_vertical/compact4_evolvemean_t5",
        "FD-vertical (compact4) with evolve_mean.",
    ),
    (
        "fd_sbp42_balanced_sub1_Nx64_Nz256_t10",
        "fd_vertical/sbp42_balanced_sub1_t10",
        "FD-vertical balanced exchange, 1 substep.",
    ),
    (
        "fd_sbp42_balanced_sub4_Nx64_Nz256_t10",
        "fd_vertical/sbp42_balanced_sub4_t10",
        "FD-vertical balanced exchange, 4 substeps.",
    ),
    (
        "fd_sbp42_balanced_sub1_fix_Nx64_Nz256_t10",
        "fd_vertical/sbp42_balanced_sub1_fix_t10",
        "FD-vertical balanced exchange, 1 substep, corrected.",
    ),
    (
        "fd_sbp42_balanced_sub4_fix_Nx64_Nz256_t10",
        "fd_vertical/sbp42_balanced_sub4_fix_t10",
        "FD-vertical balanced exchange, 4 substeps, corrected.",
    ),
    (
        "fd_sbp42_balanced_sub4_tanh_b4_Nx64_Nz256_t10",
        "fd_vertical/sbp42_balanced_sub4_tanh_b4_t10",
        "FD-vertical on a tanh-stretched grid (b=4).",
    ),
    (
        "fd_sbp42_balanced_sub4_tanh_b6_Nx64_Nz256_t10",
        "fd_vertical/sbp42_balanced_sub4_tanh_b6_t10",
        "FD-vertical on a tanh-stretched grid (b=6).",
    ),
    (
        "fd_sbp42_qnone_uniform_sub4_Nx64_Nz256_t10",
        "fd_vertical/sbp42_qnone_uniform_sub4_t10",
        "FD-vertical, q_boundary=none, uniform grid.",
    ),
    (
        "fd_sbp42_qnone_tanh_b4_sub4_Nx64_Nz256_t10",
        "fd_vertical/sbp42_qnone_tanh_b4_sub4_t10",
        "FD-vertical, q_boundary=none, tanh grid.",
    ),
    (
        "fd_smoke_sbp42_fixedcond_Nx64_Nz256",
        "fd_vertical/smokes/sbp42_fixedcond",
        "FD-vertical smoke test.",
    ),
    ("fd_smoke_balanced", "fd_vertical/smokes/balanced", "FD-vertical smoke test."),
    ("fd_smoke_qnone", "fd_vertical/smokes/qnone", "FD-vertical smoke test."),
    ("fd_smoke_tanh", "fd_vertical/smokes/tanh", "FD-vertical smoke test."),
    # ---------------------------------------------------------- exploratory --
    (
        "dinosaur_two_layer",
        "exploratory/dinosaur_spike/two_layer",
        "May-2026 NeuralGCM/Dinosaur feasibility spike -- unrelated to the "
        "NHQG solver, kept for provenance only.",
    ),
    (
        "dinosaur_native_sw",
        "exploratory/dinosaur_spike/native_shallow_water",
        "Dinosaur native shallow-water spike.",
    ),
]

# Gitignored repo artifacts that must travel with the data (not with git).
ARTIFACTS: list[tuple[str, str, str]] = [
    ("derived_checkpoints", "repo_artifacts/derived_checkpoints",
     "Derived/upsampled checkpoints referenced by the resolution ladder."),
    ("analysis/frames_pilot_gamma0_full", "repo_artifacts/frames_pilot_gamma0_full",
     "3x3 panel frames for the gamma=0 pilot, t=0.5-97 (shared colour limits)."),
    ("analysis/frames_p2_trap_sponge", "repo_artifacts/frames_p2_trap_sponge",
     "3x3 panel frames for P2, t=31-75 (shared colour limits)."),
]

ARTIFACT_FILES: list[tuple[str, str, str]] = [
    ("Miquel_NHGQtilted2026_arxiv.pdf", "repo_artifacts/Miquel_NHGQtilted2026_arxiv.pdf",
     "Reference paper (Miquel et al. 2026); gitignored because it is a PDF."),
    ("NHGQ.pdf", "repo_artifacts/NHGQ.pdf", "Built core formulation document."),
    ("NHQG_framework_deck.pdf", "repo_artifacts/NHQG_framework_deck.pdf",
     "Built pedagogical slide deck (41 slides)."),
    ("analysis/m4_pilot_opentop_3x3.mp4", "repo_artifacts/m4_pilot_opentop_3x3.mp4",
     "Early gamma=0 pilot movie (t=0-10)."),
    ("analysis/movie_pilot_gamma0_t0.5_t97_3x3.mp4",
     "repo_artifacts/movie_pilot_gamma0_t0.5_t97_3x3.mp4",
     "Full gamma=0 pilot movie, t=0.5-97."),
    ("analysis/movie_p2_trap_sponge_t31_t75_3x3.mp4",
     "repo_artifacts/movie_p2_trap_sponge_t31_t75_3x3.mp4",
     "P2 movie, t=31-75 (shared colour limits with the pilot movie)."),
    ("spectral_diagnostics_reference.pdf",
     "repo_artifacts/spectral_diagnostics_reference.pdf",
     "Built spectral-diagnostics reference."),
    ("discretely_balanced_mean_fluctuation_thermal_formulation.pdf",
     "repo_artifacts/discretely_balanced_mean_fluctuation_thermal_formulation.pdf",
     "Built discrete mean/fluctuation thermal-balance note."),
    ("fd_compact_operators.pdf", "repo_artifacts/fd_compact_operators.pdf",
     "Built FD compact-operator note."),
]

# Driver stdout logs. These matter more than their size suggests: a checkpoint
# stores no configuration, so for every run whose RESTART.md entry is
# RECONSTRUCTED the log header is the only surviving record of the checkpoint,
# output directory and flag string it was launched with. LaTeX build logs (any
# `foo.log` next to a `foo.tex`) are excluded as noise.
ARTIFACT_LOG_GLOBS: list[tuple[str, str]] = [
    ("*.log", "Chebyshev-era runs, launched from the repo root."),
    ("output/*.log", "Polar-campaign runs, launched into output/."),
]


# ---------------------------------------------------------------------------


def link_tree(src: Path, dst: Path, dry_run: bool) -> tuple[int, int]:
    """Hardlink every file under `src` into `dst`. Returns (n_files, n_bytes)."""
    n_files = 0
    n_bytes = 0
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_dir = dst / rel
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            s = Path(root) / name
            if s.is_symlink() or not s.is_file():
                continue
            n_files += 1
            n_bytes += s.stat().st_size
            if dry_run:
                continue
            d = target_dir / name
            if d.exists():
                continue
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)  # cross-device fallback
    return n_files, n_bytes


def time_range(src: Path) -> str:
    """Best-effort simulation time span for a run directory."""
    csv_path = src / "diagnostics.csv"
    if csv_path.is_file():
        try:
            with csv_path.open() as fh:
                rows = list(csv.DictReader(fh))
            if rows and "t" in rows[0]:
                return f"{float(rows[0]['t']):.2f}-{float(rows[-1]['t']):.2f}"
        except Exception:
            pass
    matches = (re.search(r"checkpoint_(\d+)\.npz", p.name)
               for p in src.glob("checkpoint_*.npz"))
    steps = sorted(int(m.group(1)) for m in matches if m)
    if steps:
        return f"step {steps[0]}-{steps[-1]}"
    return "-"


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} TB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="Archive root to create")
    ap.add_argument("--source", default=str(REPO / "output"), help="Run output root")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dest = Path(args.dest).resolve()
    source = Path(args.source).resolve()
    if not source.is_dir():
        print(f"error: source {source} does not exist", file=sys.stderr)
        return 1

    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    total_files = total_bytes = 0

    for src_name, rel_dst, desc in RUNS:
        src = source / src_name
        if not src.is_dir():
            missing.append(src_name)
            continue
        nf, nb = link_tree(src, dest / rel_dst, args.dry_run)
        rows.append((rel_dst, src_name, nf, nb, time_range(src), desc))
        total_files += nf
        total_bytes += nb
        print(f"  {rel_dst:<62s} {nf:6d} files  {human(nb):>9s}", flush=True)

    for src_name, rel_dst, desc in ARTIFACTS:
        src = REPO / src_name
        if not src.is_dir():
            missing.append(src_name)
            continue
        nf, nb = link_tree(src, dest / rel_dst, args.dry_run)
        rows.append((rel_dst, src_name, nf, nb, "-", desc))
        total_files += nf
        total_bytes += nb
        print(f"  {rel_dst:<62s} {nf:6d} files  {human(nb):>9s}", flush=True)

    for src_name, rel_dst, desc in ARTIFACT_FILES:
        src = REPO / src_name
        if not src.is_file():
            missing.append(src_name)
            continue
        d = dest / rel_dst
        nb = src.stat().st_size
        if not args.dry_run:
            d.parent.mkdir(parents=True, exist_ok=True)
            if not d.exists():
                try:
                    os.link(src, d)
                except OSError:
                    shutil.copy2(src, d)
        rows.append((rel_dst, src_name, 1, nb, "-", desc))
        total_files += 1
        total_bytes += nb

    for pattern, desc in ARTIFACT_LOG_GLOBS:
        logs = sorted(p for p in REPO.glob(pattern)
                      if p.is_file() and not p.with_suffix(".tex").exists())
        nf = nb = 0
        for src in logs:
            d = dest / "repo_artifacts/run_logs" / src.name
            nb += src.stat().st_size
            nf += 1
            if not args.dry_run:
                d.parent.mkdir(parents=True, exist_ok=True)
                if not d.exists():
                    try:
                        os.link(src, d)
                    except OSError:
                        shutil.copy2(src, d)
        if nf:
            rows.append(("repo_artifacts/run_logs", pattern, nf, nb, "-", desc))
            total_files += nf
            total_bytes += nb
            print(f"  {'repo_artifacts/run_logs (' + pattern + ')':<62s} "
                  f"{nf:6d} files  {human(nb):>9s}", flush=True)

    unclassified = sorted(
        p.name for p in source.iterdir()
        if p.is_dir() and p.name not in {r[0] for r in RUNS}
    )

    print(f"\ntotal: {total_files} files, {human(total_bytes)}")
    if missing:
        print(f"missing sources ({len(missing)}): {', '.join(missing)}")
    if unclassified:
        print(f"UNCLASSIFIED, not archived ({len(unclassified)}):")
        for name in unclassified:
            print(f"   {name}")

    if args.dry_run:
        return 0

    with (dest / "MANIFEST.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["archive_path", "source_dir", "n_files", "bytes", "t_range"])
        for rel_dst, src_name, nf, nb, tr, _desc in rows:
            w.writerow([rel_dst, src_name, nf, nb, tr])

    write_readme(dest, rows, total_files, total_bytes, unclassified)
    print(f"\nwrote {dest/'MANIFEST.tsv'} and {dest/'README.md'}")
    return 0


def write_readme(dest: Path, rows, total_files: int, total_bytes: int,
                 unclassified: list[str]) -> None:
    by_section: dict[str, list] = {}
    for row in rows:
        by_section.setdefault(row[0].split("/")[0], []).append(row)

    titles = {
        "polar": ("Polar-cap campaign (current work)",
                  "Jupiter polar vortex crystals via the Siegelman-Young-Ingersoll "
                  "(2022) trap method. This is the live research thread; "
                  "`polar_512_plan.md` is the frozen contract and "
                  "`polar_512_todo.md` the live tracker."),
        "chebyshev": ("Chebyshev production thread (Miquel reproduction)",
                      "Reproducing Miquel et al. (2026) rotating convection, the "
                      "Nusselt gap, the anti-Hermitian ghost mode, and the "
                      "ghost-clean dual-cascade budgets. See `CLAUDE.md`, "
                      "`hermitian_ghost.md`, `blowup.md`, `spectral_analysis.md`, "
                      "`adjoint_mean_exchange.md`."),
        "fd_vertical": ("Finite-difference vertical benchmark (Route B)",
                        "Separate solver package targeting the eventual mixed-BC "
                        "goal (Neumann-w top / Dirichlet bottom via SBP-SAT). See "
                        "`fd_vertical_benchmark/README.md`."),
        "exploratory": ("Exploratory spikes",
                        "Kept for provenance; not part of any active result."),
        "repo_artifacts": ("Repository artifacts (gitignored)",
                           "Built PDFs, movies and frame stacks that are excluded "
                           "from git by `.gitignore` but belong with the project."),
    }

    lines: list[str] = []
    lines.append("# NHQG run archive")
    lines.append("")
    lines.append(
        "Every run directory produced by the NHQG project, staged for migration "
        "off the original GPU host (2026-07-26). This archive is the **data** "
        "half of the project; the **code** half is the git repository "
        "(`nhqg/`, `scripts/`, `tests/`, and the `.md`/`.tex` documents), which "
        "is tracked separately on GitHub."
    )
    lines.append("")
    lines.append(f"- **{len(rows)} entries, {total_files:,} files, {human(total_bytes)}**")
    lines.append("- Built by `scripts/build_data_archive.py` from the repo's `output/`.")
    lines.append(
        "- Directory names here are descriptive; `MANIFEST.tsv` maps every "
        "archive path back to its original `output/<name>` directory, which is "
        "what the older `.md` write-ups cite."
    )
    lines.append("")
    lines.append("## How the data relates to the repository")
    lines.append("")
    lines.append(
        "Checkpoints (`checkpoint_*.npz`) hold **state only -- no configuration**. "
        "Restart physics is determined entirely by the command-line flags passed "
        "to the driver, whose silent defaults (`legacy`, `32_rule`, `jacobian`, "
        "`float32`) do *not* match any production run. The exact flag string for "
        "each run is recorded in `RESTART.md` at the archive root; treat it as "
        "part of the data, not as documentation."
    )
    lines.append("")
    lines.append("| file pattern | what it is |")
    lines.append("| --- | --- |")
    lines.append("| `checkpoint_<step>.npz` | full prognostic state; restart with `--restart-checkpoint` |")
    lines.append("| `snapshot_<step>.nc` | nodal fields (`q_prime`, `w`, `theta`, `psi`) for analysis |")
    lines.append("| `diagnostics.csv` | per-diagnostic-interval scalars (polar-era runs) |")
    lines.append("| `diagnostics_history.npz`, `spectra/` | shell budgets and spectra (Chebyshev-era runs) |")
    lines.append("| `png/` | in-run preview frames |")
    lines.append("")
    lines.append("Simulation time is `step * dt`, with `dt = 5e-5` for every run here.")
    lines.append("")

    for key in ("polar", "chebyshev", "fd_vertical", "exploratory", "repo_artifacts"):
        if key not in by_section:
            continue
        title, blurb = titles[key]
        lines.append(f"## {title}")
        lines.append("")
        lines.append(blurb)
        lines.append("")
        section_bytes = sum(r[3] for r in by_section[key])
        lines.append(f"*{len(by_section[key])} entries, {human(section_bytes)}*")
        lines.append("")
        for rel_dst, src_name, nf, nb, tr, desc in by_section[key]:
            lines.append(f"### `{rel_dst}`")
            lines.append("")
            lines.append(desc)
            lines.append("")
            lines.append(f"- source: `output/{src_name}`")
            lines.append(f"- {nf:,} files, {human(nb)}" + (f", t = {tr}" if tr != "-" else ""))
            lines.append("")

    if unclassified:
        lines.append("## Not archived")
        lines.append("")
        lines.append(
            "These directories existed under `output/` but were not classified "
            "by `scripts/build_data_archive.py`; add them to `RUNS` there if they "
            "matter."
        )
        lines.append("")
        for name in unclassified:
            lines.append(f"- `{name}`")
        lines.append("")

    (dest / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
