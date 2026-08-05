# Run data — where it is and what it means

This repository holds the **code and the written record**. It holds no run data:
`output/` is gitignored and always was, because it reached 425 GB.

All of it — every run directory ever produced by this project — is staged in a
single archive built by [`scripts/build_data_archive.py`](scripts/build_data_archive.py):

```
NHQG_runs_archive_2026-07/
├── README.md          one section per run: what it is, which document depends on it
├── MANIFEST.tsv       archive path → original output/ name, file count, bytes, t-range
├── RESTART.md         the exact flag string to resume each run
├── polar/             the SYI22 polar-cap campaign          15 entries, 115 GB
├── chebyshev/         Miquel reproduction / Nusselt / cascade  ~40 entries, 300 GB
├── fd_vertical/       Route-B finite-difference benchmark    16 entries, 4.5 GB
├── exploratory/       spikes kept for provenance             2 entries, 0.8 GB
└── repo_artifacts/    gitignored repo outputs (PDFs, movies, frame stacks,
                       and run_logs/ — the driver stdout logs, see below)
```

**82 entries, 9,836 files, 425.0 GB** (456,294,434,499 bytes).

The archive is built with hardlinks, so on the original host it occupied no
extra disk — it and `output/` pointed at the same inodes.

## Where it lives

**SDSC Expanse**, account `cla119`:

```
~/projects/NHQG_runs_archive_2026-07/
   (= /expanse/lustre/projects/cla119/kaushiks/NHQG_runs_archive_2026-07/)
```

`~/projects/NHQG` on Expanse is a **stale April checkout**, not this
repository. Its `output/` holds 35 of the 68 runs and was hardlinked into the
archive to seed the transfer, so it shares inodes with the archive and can be
deleted once the archive verifies — see `HANDOFF.md` §2. Clone this repository
fresh; do not work in that directory.

## Reading it

Start with the archive's own `README.md`: it has a section per run saying what
the run *is* and which repository document depends on it, not just what the
flags were. Three cross-references matter most:

| archive path | what depends on it |
| --- | --- |
| `polar/p2_trap_sponge_512x64_t30_t158` | the trap-arrest result; `polar_512_todo.md` (M5/P2) |
| `polar/pilot_gamma0_512x64_t0_t97` | the γ=0 control, and P2's `t=30` restart state |
| `chebyshev/production_128x256_t40_t120` | `CLAUDE.md` §3b dual-cascade budgets; `analysis/spectral_budget/window_t80_t120/` |

The older `.md` write-ups (`CLAUDE.md`, `blowup.md`, `spectral_analysis.md`,
`adjoint_mean_exchange.md`) cite runs by their **original** `output/<name>`
directory. `MANIFEST.tsv` is the lookup table from those names to the archive's
descriptive paths.

## Two things that will bite you

**1. Checkpoints carry no configuration.** `checkpoint_*.npz` is state only. The
driver's argparse defaults (`legacy`, `32_rule`, `jacobian`, `float32`) match no
production run in the archive, and restarting with the wrong flags gives
plausible output rather than an error. `RESTART.md` records the flag string for
each run family, labelled `RECORDED` (captured from the live process or a launch
script) or `RECONSTRUCTED` (assembled from the log header and `CLAUDE.md` —
check it against the run's own log before quoting a result).

Those logs are in `repo_artifacts/run_logs/` (54 files, 1.7 MB) — the driver
stdout of each run, named after the run. For a `RECONSTRUCTED` entry the log
header is the only surviving record of what the run was actually launched with.

Runs started after 2026-07-26 write `run_config.json` into their output
directory — full `argv`, resolved `NHQGConfig`, host, JAX version. Every run in
this archive predates that change, so none of them have it; every future run
will.

**2. Raw spectral diagnostics are contaminated.** Every Chebyshev-era archive
carries the anti-Hermitian `ky=0` ghost mode. `Nu_raw`, `KE_bc` and the
shell-budget inner products are affected — the notorious `Nu_raw = 6.6e27` at
`t=120` in a run whose `Nu_d` was 21.9. Trust only `Nusselt_dealiased`,
`vol_avg_tw_dealiased`, dealiased shell budgets, and `max_*`. Read
`hermitian_ghost.md` first. The `k = 0.9786` "dominant shell" of the 2026-04
analysis is the ghost's home shell, not physics.

## Rebuilding the archive

The classification lives in the `RUNS` table in
`scripts/build_data_archive.py` — source directory, destination path, and the
prose that becomes the README section. To add a run, add a row:

```bash
python scripts/build_data_archive.py --dest ../NHQG_runs_archive_2026-07 --dry-run
python scripts/build_data_archive.py --dest ../NHQG_runs_archive_2026-07
```

The dry run reports anything under `output/` that the table does not classify,
so nothing gets silently dropped.
