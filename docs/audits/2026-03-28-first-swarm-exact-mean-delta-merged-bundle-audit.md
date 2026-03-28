# First Swarm Exact Mean-Delta Merged Bundle Audit

Date: 2026-03-28

This audit follows:

- `docs/audits/2026-03-27-tailrun-admitted-home-tailnet-run-audit.md`
- `docs/audits/2026-03-28-homegolf-tailrun-daily-loop-operator-fix-audit.md`

## What Changed

The admitted-device Tailnet mixed-hardware run no longer stops at contribution
receipts, replay truth, and a summary-only merge disposition.

The coordinator now retains:

- the local M5 contributor adapter payload
- the remote RTX 4080 contributor adapter payload
- one exact mean-delta merged adapter artifact
- one inferable merged portable bundle
- one machine-readable merged-artifact report

The operator path also stopped depending on remote `/tmp` for:

- staged repo extraction
- cargo target cache
- rustc temporary files
- report copy-back

## Code And Operator Changes

Code path:

- `crates/psionic-train/src/swarm_first_live_runtime.rs`

Operator surfaces:

- `scripts/run-first-swarm-tailnet-admitted-live.sh`
- `scripts/check-first-swarm-trusted-lan-real-run.sh`

The runtime-side merge is exact for the current first-swarm lane. It uses rank
stacking so the merged artifact represents the mean delta of both admitted LoRA
contributors without truncating the merged surface back down to rank `2`.

## Retained Run

Exact command:

```bash
scripts/run-first-swarm-tailnet-admitted-live.sh \
  --run-id tailrun-home-admitted-20260328k \
  --bundle-dir fixtures/swarm/runs/tailrun-home-admitted-20260328k
```

Retained artifacts:

- bundle:
  `fixtures/swarm/runs/tailrun-home-admitted-20260328k/first_swarm_real_run_bundle.json`
- summary:
  `fixtures/swarm/runs/tailrun-home-admitted-20260328k/tailrun_admitted_home_run_summary.json`
- coordinator report:
  `fixtures/swarm/runs/tailrun-home-admitted-20260328k/coordinator_runtime_report.json`
- contributor report:
  `fixtures/swarm/runs/tailrun-home-admitted-20260328k/contributor_runtime_report.json`
- merged report:
  `fixtures/swarm/runs/tailrun-home-admitted-20260328k/retained_artifacts/merged_artifact_report.json`
- local contributor adapter:
  `fixtures/swarm/runs/tailrun-home-admitted-20260328k/retained_artifacts/swarm-mac-a_adapter.safetensors`
- remote contributor adapter:
  `fixtures/swarm/runs/tailrun-home-admitted-20260328k/retained_artifacts/swarm-linux-4080-a_adapter.safetensors`
- merged adapter:
  `fixtures/swarm/runs/tailrun-home-admitted-20260328k/retained_artifacts/merged_mean_delta_adapter.safetensors`
- merged portable bundle:
  `fixtures/swarm/runs/tailrun-home-admitted-20260328k/retained_artifacts/merged_portable_bundle.safetensors`

## Verified Outcome

Checker:

```bash
scripts/check-first-swarm-trusted-lan-real-run.sh \
  --bundle fixtures/swarm/runs/tailrun-home-admitted-20260328k/first_swarm_real_run_bundle.json
```

Verified result:

- verdict: `verified`
- run id: `tailrun-home-admitted-20260328k`
- result classification: `bounded_success`
- accepted contributions: `2`
- replay-checked contributions: `2`
- merge disposition: `merged`
- publish disposition: `refused`
- promotion disposition: `held`
- merge strategy: `exact_mean_delta_rank_stacking`
- merged LoRA rank: `4`

## Merged Artifact Facts

The retained merged-artifact report records:

- merged LoRA rank: `4`
- merged LoRA alpha: `4.0`
- merged portable bundle state-dict digest:
  `b452249b53bb2b9e0625f78294bbbd10825d88706e1bc42b8a6a0eacc438c62c`
- canonical profile mean loss:
  `1.788139627478813e-07`
- canonical profile bits per token:
  `2.579740172980722e-07`
- deterministic probe top token id: `2`

Retained artifact sizes:

- local contributor adapter: `1472` bytes
- remote contributor adapter: `1480` bytes
- merged adapter: `296` bytes
- merged portable bundle: `5408` bytes

## Improvement Over The Previous Retained Tailnet Run

Compared with `tailrun-home-admitted-20260327e`, this closes three real gaps:

1. the retained admitted-device run now emits one inferable mixed-device
   artifact instead of stopping at summary-only merge truth
2. the remote operator path no longer depends on `/tmp` for repo staging, cargo
   target output, or rustc temp files
3. the retained checker now validates the merged-artifact surface directly

## Honest Boundary

This audit does claim:

- one real admitted-device mixed-hardware Tailnet run completed on clean
  `psionic/main`
- the run retained both contributor payloads and one exact mean-delta merged
  artifact family
- the merged portable bundle is machine-readable and checker-verified

This audit does not claim:

- exact HOMEGOLF score closure
- exact public Parameter Golf score closure
- one promoted served runtime bundle
- public-leaderboard-equivalent hardware
