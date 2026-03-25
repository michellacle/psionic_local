# First Multi-Provider Dense CUDA Run Audit

Date: March 25, 2026

## Summary

This audit records the first bounded multi-provider dense CUDA proof run under
the shared cross-provider training contracts.

Run id: `psion-xprovider-pretrain-multi-provider-cuda-20260325`

Retained machine-legible bundle:

- `fixtures/training/first_multi_provider_dense_cuda_run_v1.json`

## What The Run Did

The run executed as one shared pretraining program across Google and RunPod in
four bounded phases:

1. bootstrapped on the existing RunPod `8xH100` dense CUDA mesh
2. paused at a durable checkpoint barrier and grew to a `10`-rank mixed Google
   plus RunPod CUDA mesh
3. repaired a RunPod provider-loss event on rank `2` by replacing that rank
   with a Google spare under the admitted replace-rank recovery contract
4. completed a bounded steady-state mixed-provider CUDA phase after the repair

The retained proof surface binds directly to:

- `fixtures/training/cross_provider_program_run_graph_v1.json`
- `fixtures/training/dense_rank_recovery_contract_v1.json`
- `fixtures/training/dense_topology_revision_contract_v1.json`
- `fixtures/training/provider_neutral_training_execution_evidence_bundle_v1.json`

## What This Proved

The repo can now truthfully cite one bounded multi-provider dense CUDA program
that:

- used one shared run id instead of provider-local side jobs
- widened from single-provider dense CUDA into a mixed Google plus RunPod CUDA
  mesh through an explicit checkpoint-barrier grow-world revision
- survived one provider-loss event through an admitted replace-rank recovery
  path
- retained one machine-legible proof bundle plus the after-action audit

## What This Did Not Prove

This run still does not prove:

- mixed-backend dense training
- public or adversarial swarm compute
- generic live elastic world-size changes without checkpoint barriers
- production readiness across arbitrary providers or arbitrary dense cluster
  sizes

## Main Constraint

The grow and shrink topology operations are still checkpoint-barrier
elasticity, not hot elastic data-feed revisions. That remains the correct
claim boundary for the current codebase.
