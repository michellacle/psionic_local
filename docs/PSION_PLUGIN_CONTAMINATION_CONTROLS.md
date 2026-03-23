# PSION Plugin Contamination Controls

> Status: canonical `PSION_PLUGIN-6` contamination-control contract for the first
> plugin-conditioned convergence tranche, written 2026-03-22 after landing the
> first repo-owned parent-lineage and exclusion bundle above the committed
> dataset and derivation artifacts, and extended the same day after landing the
> first mixed host-native plus guest-artifact contamination bundle.

This document freezes the first plugin-aware contamination-control surface for
the Psion plugin lane.

The contract is deliberately narrow.

It does not introduce a second isolation system next to `PSION`.

It projects the existing plugin-conditioned dataset and derivation truth into
one smaller artifact that later SFT and benchmark stages can cite directly.

## Canonical Artifacts

- `docs/PSION_PLUGIN_CONTAMINATION_CONTROLS.md` is the canonical
  human-readable contract.
- `crates/psionic-data/src/psion_plugin_contamination_controls.rs` owns the
  typed bundle, parent-lineage rows, and exclusion manifest validation.
- `crates/psionic-data/examples/psion_plugin_contamination_controls.rs`
  writes the host-native reference contamination bundle.
- `crates/psionic-data/examples/psion_plugin_mixed_contamination_controls.rs`
  writes the mixed contamination bundle.
- `fixtures/psion/plugins/datasets/psion_plugin_contamination_controls_v1/` carries
  the first committed host-native machine-readable artifact.
- `fixtures/psion/plugins/datasets/psion_plugin_mixed_contamination_controls_v1/`
  carries the first committed mixed machine-readable artifact.

The stable bundle schema version is
`psionic.psion.plugin_contamination_bundle.v1`.

## Why This Exists

The plugin-conditioned dataset bundle already freezes train and held-out split
identity.

That is not enough by itself for contamination review.

Later benchmark packages need a smaller surface that answers:

- which parent training record produced one SFT or held-out eval item
- which plugin-trace source case produced it
- which plugin runtime receipts are attached to it
- which source cases and receipt refs must be excluded from the opposite split

This document freezes that answer in one machine-readable place.

## Parent Lineage

Every row in the bundle now carries:

- one stable `item_ref`
- one `training_record_id` and `training_record_digest`
- one `split_kind`
- one `workflow_case_id`
- one controller-surface label
- one source trace identity block
- zero or more plugin runtime receipt refs and digests

The first bundle covers two item classes only:

- `sft_train_record`
- `held_out_eval_record`

That keeps the initial contamination posture honest.

The repo is not yet claiming package-specific benchmark breadth here.

## Exclusion Manifest

The bundle also emits one exclusion manifest with:

- held-out plugin-trace source cases excluded from training
- train plugin-trace source cases excluded from held-out benchmark packaging
- held-out plugin receipt refs excluded from training
- train plugin receipt refs excluded from held-out benchmark packaging

This is the plugin-aware mirror of the existing `Psion` exclusion posture.

It keeps plugin-trace and plugin-receipt ancestry explicit instead of relying
on manual reviewer memory.

## Trace-Disjoint Policy

The v1 policy is intentionally conservative:

- train and held-out rows must stay disjoint by `workflow_case_id`
- train and held-out rows must stay disjoint by `source_case_id`
- train and held-out rows must stay disjoint by plugin runtime `receipt_ref`

That is enough for the first bounded plugin-conditioned SFT and held-out eval
lane.

Later benchmark packages may add richer family-specific rules, but they may not
weaken this base disjointness contract.

## Machine-Queryable Review

Contamination review is not just prose.

The bundle carries:

- explicit parent-lineage rows
- split-scoped disjointness groups
- one typed exclusion manifest
- one stable digest across the whole bundle

That gives later training and benchmark builders one direct machine-readable
surface for contamination review and failure checks.

## Mixed Dataset Follow-On

`PSION_PLUGIN-25` adds one second contamination artifact for the mixed dataset.

That mixed contamination bundle keeps the same parent-lineage and exclusion
schema, but it now includes one guest-artifact lineage row in the `train`
split.

The mixed bundle remains explicit that:

- the guest-artifact lineage is train-only in mixed v1
- held-out lineage is still host-native only
- benchmark exclusion now also carries the guest-artifact receipt and source
  ancestry from the mixed train split

This keeps the first mixed training lane honest without pretending that the repo
already has a broader held-out guest-artifact benchmark family.
