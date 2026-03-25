# Hybrid Pretraining Planner Reference

> Status: canonical `XTRAIN-6` / `#522` record, updated 2026-03-25 after
> landing the first hybrid dense-rank plus contributor planner in
> `crates/psionic-train/src/hybrid_pretraining_planner.rs`.

This document records the first shared planner that can emit dense-rank work
and validated contributor work under one pretraining-program identity.

## Canonical Runner

Run the checker from the repo root:

```bash
scripts/check-hybrid-pretraining-plan.sh
```

## What Landed

`psionic-train` now owns one typed hybrid plan surface with explicit work-class
boundaries:

- dense full-model ranks
- validated contributor windows
- validators
- eval workers
- checkpoint writers

The landed surface includes:

- `HybridPretrainingDenseRankAssignment`
- `HybridPretrainingContributorWindowAssignment`
- `HybridPretrainingValidatorAssignment`
- `HybridPretrainingEvalAssignment`
- `HybridPretrainingCheckpointWriterAssignment`
- `HybridPretrainingLineageBinding`
- `HybridPretrainingPlan`
- the binary `hybrid_pretraining_plan`
- the checker `scripts/check-hybrid-pretraining-plan.sh`
- the fixture `fixtures/training/hybrid_pretraining_plan_v1.json`

## Current Honest Boundary

This issue closes planning, not runtime execution.

What it proves now:

- one provider-neutral program manifest can emit dense-rank work and validated
  contributor windows in the same plan
- the whole plan stays on one dataset family and one checkpoint family
- every planned artifact slot carries one explicit execution class and one
  lineage slot for later evidence binding

What it still does not prove:

- provider launch closure
- mixed-backend dense execution
- final evidence bundle closure
- recovery or elasticity closure

The planner keeps those boundaries explicit instead of flattening contributor
windows into dense ranks or inventing provider-specific planning APIs.
