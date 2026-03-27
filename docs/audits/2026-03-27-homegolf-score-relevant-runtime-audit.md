# HOMEGOLF Score-Relevant Runtime Audit

Date: March 27, 2026

## Summary

This audit records the first canonical HOMEGOLF runtime that is strong enough to
count as score-relevant dense training instead of a symbolic proof lane.

Retained machine-readable report:

- `fixtures/parameter_golf/reports/parameter_golf_homegolf_score_relevant_runtime.json`

Generator and checker:

- `crates/psionic-train/src/parameter_golf_homegolf_score_runtime.rs`
- `crates/psionic-train/src/bin/parameter_golf_homegolf_score_relevant_runtime.rs`
- `scripts/check-parameter-golf-homegolf-score-relevant-runtime.sh`

## What Landed

The HOMEGOLF track now has one explicit runtime report that binds:

- the upgraded live dense mixed-device HOMEGOLF surface
- the retained same-job MLX-plus-CUDA dense timing receipts
- the exact HOMEGOLF dense-baseline geometry reference

into one machine-readable answer to a simple question:

- is the current dense HOMEGOLF runtime still symbolic, or is it now large
  enough to make `600s` quality comparisons meaningful?

The retained answer is now:

- `runtime_status=score_relevant`

## Retained Runtime Values

The canonical retained runtime keeps:

- `observed_step_count=4`
- `observed_train_tokens_per_step=589824`
- exact challenge reference geometry:
  - `train_batch_tokens=524288`
  - `train_sequence_length=1024`
  - `grad_accum_steps=8`
- dense runtime timing:
  - `observed_cluster_wallclock_ms=3611`
  - `mean_cluster_step_ms=902.75`
  - `effective_cluster_steps_per_second=1.1077263915812794`
  - `effective_cluster_train_tokens_per_second=653363.6111880365`
- `600s` projection:
  - `projected_steps_within_cap=664.6358349487676`
  - `projected_train_tokens_within_cap=392018166.7128219`
  - `projected_dataset_passes_within_cap=3.920181667128219`

So the retained dense runtime now projects almost four full passes over the
current `100,000,000`-token HOMEGOLF dense-baseline training set inside the
same `600s` cap.

## Per-Device Runtime Truth

Retained device metrics:

- `runpod-cuda-submesh`
  - `estimated_steps_per_second=1.6339869281045751`
  - `estimated_train_tokens_per_second=963764.705882353`
  - `contribution_share=0.8888888888888888`
- `local-mlx-rank`
  - `estimated_steps_per_second=1.4270424545130218`
  - `estimated_train_tokens_per_second=841703.8886906885`
  - `contribution_share=0.1111111111111111`

Retained phase breakdown:

- `mean_cuda_submesh_step_ms=612.0`
- `mean_mlx_rank_step_ms=700.75`
- `mean_cross_backend_bridge_ms=118.0`
- `mean_optimizer_step_ms=84.0`
- `dominant_step_bottleneck=mlx_rank`

The honest read is straightforward:

- CUDA is not the current bottleneck on this retained mixed-device dense lane
- the MLX rank is slower than the CUDA submesh on the retained steps
- bridge plus optimizer time still add `202ms` on top of the dominant device
  phase each step

## Why This Closes The Runtime Gap

Before this report, HOMEGOLF had:

- a strict contest-preflight surface
- a dense mixed-device score surface
- a bounded exact-family train-to-infer proof

But it still lacked one explicit runtime answer showing that the dense lane had
graduated from tiny symbolic training to meaningful `600s` training volume.

This report closes that gap because it now freezes:

- resident dense runtime state instead of tiny proof-only updates
- challenge-scale or larger train-token volume per retained step
- real cluster throughput and phase timing
- projected `600s` training volume that clears one full dataset pass by a wide
  margin

That is enough to say the current HOMEGOLF runtime is score-relevant, even
though it is still not competitive with the public leaderboard on `val_bpb`.

## What This Does Not Prove

This audit still does **not** prove:

- admitted Apple-plus-home-RTX dense closure
- one locally produced scored bundle from that admitted home cluster
- public-leaderboard-equivalent hardware
- public-leaderboard-equivalent quality

Those remain later HOMEGOLF work items.

## Immediate Optimization Direction

The retained phase breakdown makes the next tuning order clear:

1. reduce MLX rank wallclock first
2. reduce cross-backend bridge cost second
3. keep optimizer time flat while widening the admitted local cluster

In other words, the current throughput question is no longer “is this real?”
It is now “how much of the retained `902.75ms` step can we claw back from the
MLX side and the bridge?”

## Verification

The landed runtime was revalidated with:

```bash
cargo run -q -p psionic-train --bin parameter_golf_homegolf_score_relevant_runtime -- \
  fixtures/parameter_golf/reports/parameter_golf_homegolf_score_relevant_runtime.json

./scripts/check-parameter-golf-homegolf-score-relevant-runtime.sh
```
