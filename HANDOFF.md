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

## 2. Target cluster

The original plan named **PSC Bridges-2** (H100-80GB). Note that the only
cluster configured in `~/.ssh/config` on the old host was **SDSC Expanse**
(`login.expanse.sdsc.edu`, V100-32GB) — confirm which allocation is actually
live before committing.

Either works for `512² × 64`:

| | H100-80GB (Bridges-2) | V100-32GB (Expanse) |
| --- | --- | --- |
| FP64 peak | 34 TFLOP/s | 7.8 TFLOP/s |
| memory | 80 GB HBM3, 3.35 TB/s | 32 GB HBM2, 0.9 TB/s |
| estimated step cost vs the old H200 | 1.0–1.4× | ~4× |
| fits `512²×64`? | comfortably | yes |

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

PSC has a dedicated data-transfer node; use it rather than the login node.
Both clusters need interactive auth (password + Duo), so run this yourself:

```bash
# Bridges-2
rsync -avhP --partial --append-verify \
      /home/kucla/kaushiks/mixed/NHQG_runs_archive_2026-07/ \
      <user>@data.bridges2.psc.edu:/ocean/projects/<grant>/<user>/NHQG_runs_archive_2026-07/

# Expanse
rsync -avhP --partial --append-verify \
      /home/kucla/kaushiks/mixed/NHQG_runs_archive_2026-07/ \
      kaushiks@login.expanse.sdsc.edu:/expanse/lustre/projects/<grant>/<user>/NHQG_runs_archive_2026-07/
```

`--partial --append-verify` makes it resumable: re-run the same command after
any interruption and it picks up mid-file. Check the destination quota first —
425 GB is more than a default `$HOME`, so it must land on Ocean / Lustre, not
`$HOME`.

Verify after transfer:

```bash
# on the destination
find NHQG_runs_archive_2026-07 -type f | wc -l    # expect 9776
du -sb NHQG_runs_archive_2026-07                  # expect ~456 GB apparent (425 GiB)
```

`MANIFEST.tsv` carries per-run file counts and byte totals for a finer check.

`scripts/transfer_archive.sh` wraps the above with the counts built in.

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

```bash
conda create -n nhqg python=3.13
conda activate nhqg
pip install -r requirements.txt
pip install --upgrade "jax[cuda12]"
```

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
