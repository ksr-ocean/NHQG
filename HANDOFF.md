# Handoff — migrating to an ACCESS-CI cluster

Written 2026-07-26, when the project lost access to its original GPU host
(8× H200, shared arrangement). Nothing was deleted there; the code is in this
repository and the run data is in the archive described by [`DATA.md`](DATA.md).

This document is the checklist for standing the campaign back up elsewhere.

---

## 1. What has to move

| | size | how |
| --- | --- | --- |
| this repository | 22 MB tracked | `git clone` (private GitHub repo) |
| repo working tree, gitignored parts | ~1 GB | included in the archive as `repo_artifacts/` |
| run archive | 425 GB | `rsync` (§3) |

The archive is the long pole. Everything else is minutes.

## 2. Target cluster — SDSC Expanse

**The migration went to Expanse** (account `cla119`, user `kaushiks`), not
Bridges-2. Layout on the destination:

```
~/projects  ->  /expanse/lustre/projects/cla119/kaushiks     (symlink)
  NHQG_runs_archive_2026-07/    the archive — 425 GB, this is the data
  NHQG/                         STALE April copy, 374 GB — see below
```

Expanse GPU nodes are **V100-32GB**. For reference against the old host:

| | H200 (old host) | V100-32GB (Expanse) | H100-80GB (Bridges-2) |
| --- | --- | --- | --- |
| FP64 peak | 34 TFLOP/s | 7.8 TFLOP/s | 34 TFLOP/s |
| memory | 141 GB, 4.8 TB/s | 32 GB, 0.9 TB/s | 80 GB, 3.35 TB/s |
| measured / estimated per t.u. | 19 min | ~75 min (est.) | ~19–27 min (est.) |
| fits `512²×64`? | yes | yes | yes |

At ~4× the old step cost, `512² × 64` on a V100 is roughly 75 min/t.u. — the
`t = 158 → 400` continuation would be ~300 h of GPU time across chained jobs.
Two consequences worth acting on before committing to that:

- **`Nz 64 → 32` is close to free** (vertical modes `n ≤ 32` carry all the
  resolved physics — see §5) and halves the cost.
- If a Bridges-2 allocation is available, it is ~4× faster for this workload.
  Nothing in the harness is Expanse-specific except the partition line in
  `scripts/submit_access.slurm`.

### The stale copy

`~/projects/NHQG` is a 374 GB April checkout with its own `output/` holding 35
of the 68 run directories (327 GB) — essentially the whole Chebyshev thread. It
was used to **seed** the archive: those runs were hardlinked into the archive
layout on Expanse before rsync, which cut the transfer from 425 GB to 129 GB.

Because they are hardlinks, `~/projects/NHQG/output/` and the archive share
inodes. **Once the archive is verified, the stale copy can be deleted without
losing any data** — the archive keeps its own links. Verify first (§3), then:

```bash
# after verification only
rm -rf ~/projects/NHQG        # frees ~47 GB (the non-shared remainder)
git clone git@github.com:ksr-ocean/NHQG.git ~/projects/NHQG
```

**Memory is not the constraint.** The 109.7 GB that `nvidia-smi` reported for
the P2 run was JAX's preallocation pool (0.75 × 143.8 GB), not the working set.
Bottom-up estimate of the real peak at `512² × 64`, float64:

**Memory is not the constraint.** The 109.7 GB that `nvidia-smi` reported for
the P2 run was JAX's preallocation pool (0.75 × 143.8 GB), not the working set.
Bottom-up estimate of the real peak at `512² × 64`, float64:

| | |
| --- | --- |
| state (confirmed — this is the checkpoint file size) | 402 MB |
| IMEX shell inverses (22 027 shells) | ~745 MB |
| ARS222 stage copies | ~2.0–2.5 GB |
| chunked IMEX gather at `--imex-matmul-chunk 128` | ~1.05 GB |
| 23_rule physical work arrays | ~1.5–2.5 GB |
| 3/2-padded diagnostic arrays (diagnostic steps only) | ~1–2 GB |
| **estimated peak** | **~8–14 GB** |

**Measure it on the first job** rather than trusting the estimate:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python - <<'PY'
import jax
# ... run ~30 steps ...
print(jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**30, "GiB")
PY
```

**Wall time is the real constraint.** ACCESS GPU partitions cap jobs at 48 h.
At the old host's 19 min/t.u., `t = 158 → 400` is ~77 h, so the campaign needs
**chained jobs** — see §5.

## 3. Transferring the archive

Done on 2026-07-26. Reproduce or resume with:

```bash
scripts/transfer_archive.sh kaushiks@login.expanse.sdsc.edu:~/projects/
```

`--partial --append-verify` makes it resumable — re-run the identical command
after any interruption and it picks up mid-file. Expanse needs interactive auth
(Duo), so either open a session first (the `~/.ssh/config` entry keeps a
`ControlPersist 72h` master socket, which non-interactive rsync then reuses) or
run it in a login shell.

The archive must land on Lustre, not `$HOME` — `~/projects` is a symlink to
`/expanse/lustre/projects/cla119/kaushiks`, so the path above is already
correct.

### Seeding from an existing copy

If the destination already holds some of the runs under their **original**
`output/<name>` directory names, hardlink them into the archive layout first —
rsync then skips them entirely. That is what cut this transfer from 425 GB to
129 GB. `scripts/build_data_archive.py` can do this on the destination
(`--source <existing>/output --dest <archive>`), but it needs a real Python on
`PATH` — see §4, non-interactive shells on Expanse get 3.6.8. The equivalent in
plain shell, which always works:

```bash
# for each (source_dir, archive_path) row in MANIFEST.tsv:
mkdir -p "$(dirname "$DST/$archive_path")"
cp -al "$SRC/$source_dir" "$DST/$archive_path"
```

`cp -al` is a recursive hardlink copy: instant, zero extra disk.

### Verification

```bash
VERIFY_ONLY=1 scripts/transfer_archive.sh kaushiks@login.expanse.sdsc.edu:~/projects/
```

Expect **9,779 files** and **456,291,787,769 bytes** (425 GiB). `MANIFEST.tsv`
carries per-run file counts and byte totals for a finer check. Only after this
passes should the stale copy be removed (§2).

## 4. Environment

The old host ran, and the tests pass against:

```
python  3.13.9
jax     0.10.0   (CUDA 12)
numpy   2.3.5
netCDF4 1.7.4
xarray  2025.10.1
pillow  12.0.0
matplotlib 3.10.6
```

(Note these supersede the versions quoted in `CLAUDE.md`, which are stale.)

On Expanse the account already has **`~/anaconda3/bin/python3` (3.12.2)**, which
is fine for this codebase. The catch is that `~/anaconda3/bin` is **not on
`PATH` in non-interactive shells** — neither `ssh expanse 'python3 ...'` nor
`bash -lc python3` finds it; both resolve to the system `/usr/bin/python3`,
which is **3.6.8** and too old (the codebase uses
`from __future__ import annotations`, f-strings and modern typing). Batch
scripts must therefore activate conda explicitly or use the absolute path:

```bash
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate nhqg            # or: $HOME/anaconda3/bin/python3 ...
```

To build a dedicated environment:

```bash
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda create -n nhqg python=3.13
conda activate nhqg
pip install -r requirements.txt
pip install --upgrade "jax[cuda12]"
```

`module` is also unavailable in non-interactive shells; `source
/etc/profile.d/modules.sh` first if a job needs it (e.g. for CUDA).

Always export, in every job script:

```bash
export JAX_ENABLE_X64=1          # REQUIRED — float32 silently degrades everything
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The BLAS pinning is not cosmetic: the long startup stalls on the old host came
from host-side dense linear algebra during IMEX-shell precomputation, not from
XLA compilation.

**Acceptance gates, in order:**

1. `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest tests/ -q`
   → expect `151 passed, 1 skipped` (~2 min).
2. A short GPU smoke — `scripts/submit_access.slurm` in `smoke` mode — which
   runs ~2000 steps from the P2 checkpoint and must stay finite.
3. Compare the smoke's first `diagnostics.csv` rows against the tail of
   `polar/p2_trap_sponge_512x64_t30_t158/diagnostics.csv`. `KE_bt`, `q_rms` and
   `enstrophy` should match to a few percent. If they do not, the flag string is
   wrong — see `RESTART.md` in the archive.

## 5. Resuming the campaign

The live question is P2: **does the trapped cap organise into a vortex crystal?**
At `t = 158` it had not — cap vorticity skewness `+0.05 ± 0.05` — but that is
only ~115 eddy turnovers, and the P0c sweep suggests segregation is an
O(10³) t.u. process. So the campaign needs long integration, which makes two
things load-bearing:

1. **A vertical dissipation sink.** With no vertical diffusion on `q`, the
   undamped vertical enstrophy cascade piles ψ-KE into the top Chebyshev modes
   (0.30 → 0.53 above mode 48 over `t = 31 → 70`). It leaves the barotropic
   physics alone — `max_speed` restricted to `n ≤ 32` was flat at 142 — but it
   inflates `max_speed`, and ARS222 breaks when `max_speed · k_max · dt ≳ 0.2`.
   Options: a high-n Chebyshev filter, vertical hyperdiffusion, or `Nz 64 → 32`
   (nearly free — `n ≤ 32` carries all the resolved physics — and it doubles
   throughput).
2. **Chained jobs.** `scripts/submit_access.slurm` submits itself with
   `--dependency=afterok` and restarts from the newest checkpoint in the output
   directory, so a 48 h cap becomes a non-issue. Each leg writes its own
   `run_config.json`, so the chain is auditable after the fact.

Everything else outstanding is in [`polar_512_todo.md`](polar_512_todo.md).

## 6. GitHub

The repository is private, at `github.com/ksr-ocean/<repo>`. The old host
authenticated with `~/.ssh/id_ed25519_hpc`; that key is registered to the
`ksr-ocean` account. Clone with:

```bash
git clone git@github.com:ksr-ocean/<repo>.git NHQG
```

`output/`, `*.log`, `*.pdf`, `*.mp4` and `analysis/frames_*/` are gitignored by
design — they live in the archive, not in git.
