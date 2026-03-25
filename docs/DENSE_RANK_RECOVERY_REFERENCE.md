# Dense-Rank Recovery Reference

> Status: canonical `#533` dense-rank recovery record, updated 2026-03-25
> after landing the runnable harness in
> `scripts/check-dense-rank-recovery-contract.sh`.

This document records the first provider-neutral dense-rank recovery contract
for the cross-provider pretraining system.

## What Landed

The issue added `crates/psionic-train/src/dense_rank_recovery_contract.rs`
with:

- explicit recovery scenarios for preemption, same-provider node loss,
  cross-provider provider loss, and one refused shrink-world case
- direct bindings from each recovery scenario to the retained distributed
  checkpoint restore plan
- direct bindings from each recovery scenario to the remote artifact backend
  placement and restore policy for checkpoint artifacts
- direct bindings from admitted replace-rank recoveries to the
  topology-revisable distributed data-feed report
- explicit operator and finalizer actions for recovered and refused paths

## Canonical Runner

Run the harness from the repo root:

```bash
scripts/check-dense-rank-recovery-contract.sh
```

## Pass Criteria

The dense-rank recovery layer is green only if all of the following are true:

- admitted recoveries point at real checkpoint shard assignments
- restore authority and mirror policy stay aligned with the provider-neutral
  remote artifact backend contract
- replace-rank recoveries cite replay-ordering continuity from the
  topology-revisable distributed data-feed report
- unsupported shrink-world recovery stays refused with a typed receipt

## Current Scope

The current contract admits:

- in-place checkpoint resume after preemption
- same-provider replace-rank recovery after node loss
- cross-provider replace-rank recovery after provider loss

The current contract still refuses:

- shrink-world recovery
- public-internet swarm repair
- mixed-backend dense recovery

## Operator And Finalizer Behavior

The contract keeps these actions explicit:

- operator rehydrates an in-place preempted rank from the last durable
  checkpoint
- operator admits a replacement node at the same dense rank for admitted
  replace-rank recoveries
- finalizer publishes a recovery receipt, and for replace-rank recoveries also
  publishes the revised topology binding
- refused recovery publishes a refusal receipt and holds the run instead of
  pretending recovery happened
