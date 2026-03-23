# PSION Plugin-Conditioned Dataset Bundle

> Status: canonical `PSION_PLUGIN-5` dataset-bundle contract for the first
> host-native plugin-conditioned convergence tranche, written 2026-03-22 after
> landing the first repo-owned dataset builder on top of the committed
> derivation bundle, and extended the same day after landing the first mixed
> host-native plus guest-artifact dataset bundle.

This document freezes the first dataset artifact bundle built from canonical
plugin-training records.

It is intentionally small and explicit.

The point is not volume.

The point is to emit one stable dataset identity, one split contract, and one
machine-checkable held-out boundary that later training and benchmark work can
reuse directly.

## Canonical Artifacts

- `docs/PSION_PLUGIN_CONDITIONED_DATASET.md` is the canonical human-readable
  contract.
- `crates/psionic-data/src/psion_plugin_conditioned_dataset.rs` owns the
  typed dataset bundle and split validation.
- `crates/psionic-data/examples/psion_plugin_conditioned_dataset.rs` writes
  the host-native reference dataset bundle.
- `crates/psionic-data/examples/psion_plugin_mixed_conditioned_dataset.rs`
  writes the mixed host-native plus guest-artifact dataset bundle.
- `fixtures/psion/plugins/datasets/psion_plugin_conditioned_dataset_v1/` carries
  the first committed host-native dataset artifact.
- `fixtures/psion/plugins/datasets/psion_plugin_mixed_conditioned_dataset_v1/`
  carries the first committed mixed dataset artifact.

The stable bundle schema version is
`psionic.psion.plugin_conditioned_dataset_bundle.v1`.

## Host-Native Reference Split Policy

The first dataset bundle uses one intentionally conservative isolation rule:

- training and held-out records must be disjoint by `workflow_case_id`

That means the same normalized workflow case cannot appear in both splits,
even if it was observed through different controller surfaces.

The current v1 build keeps:

- one `train` split
- one `held_out` split

This is enough for the first bounded SFT and evaluation lane.

## Host-Native Reference Exclusion Boundary

The current committed derivation bundle is now broader than this dataset.

It includes one bounded guest-artifact training record.

This dataset still excludes that workflow on purpose.

Why:

- `PSION_PLUGIN-24` extends the derivation pipeline to a shared schema
- `PSION_PLUGIN-25` is the later issue that builds the first mixed dataset

So v1 dataset truth remains:

- host-native only
- workflow-case-disjoint
- no mixed guest-artifact training split yet

## Mixed Dataset V1

`PSION_PLUGIN-25` now adds one second dataset artifact instead of mutating the
host-native reference lane in place.

The mixed dataset keeps the same bundle schema and the same
`workflow_case_disjoint.v1` held-out rule, but changes the train split
membership:

- `train` now contains the host-native `web_content_success` records plus the
  bounded guest-artifact `guest_artifact_success` record
- `held_out` still contains the host-native `fetch_refusal` records

This means the first mixed dataset is honest about two things at once:

- the mixed train split really contains one admitted guest-artifact record
- the held-out split is still host-native only

That is deliberate.

It gives the mixed learned lane one real guest-artifact training example while
keeping the first held-out benchmark lineage surface conservative.

## Class Balance Reporting

The mixed dataset does not hide class coverage behind one rebalance number.

It reports class balance directly through the existing split stats:

- `split.stats.plugin_class_counts`
- `split.stats.controller_surface_counts`
- `split.stats.route_label_counts`
- `split.stats.outcome_label_counts`

For mixed v1 this means the train split machine-readably shows one
`guest_artifact_digest_bound` record while held-out shows none.

## Preserved Labels

The dataset bundle keeps the following labels visible per split:

- controller-surface counts
- plugin-class counts
- route-label counts
- outcome-label counts

That keeps later model and benchmark work from treating the derived records as
anonymous prompt/response examples.

## Stable Identity

The bundle now carries:

- one stable dataset ref
- one immutable version
- one stable `dataset_ref@version` identity
- one source derivation bundle ref and digest

That gives later training stages one direct citation surface instead of
pointing at a loose folder of JSON records.
