# Psion Tokenized Corpus

> Status: canonical `PSION-9` / `#365` tokenized-corpus contract, written
> 2026-03-22 after landing the tokenizer artifact bundle.

This document freezes the first replay-safe tokenized corpus manifest for the
`Psion` lane.

It turns tokenized corpora into first-class Psionic artifacts instead of
one-off loader outputs.

## Canonical Artifacts

- `crates/psionic-data/src/psion_tokenized_corpus.rs` owns the typed tokenized
  corpus contract.
- `fixtures/psion/tokenized/psion_tokenized_corpus_manifest_v1.json` is the
  canonical machine-readable manifest example.

The stable schema version is `psion.tokenized_corpus_manifest.v1`.

## What The Manifest Carries

The tokenized corpus manifest now freezes:

- stable dataset id and version
- tokenizer-bundle lineage and tokenizer digest
- raw-source schema version and preprocessing version
- packing-policy id and version
- replay-safe dataset identity
- source-family bindings
- split-level aggregates
- tokenized shard identities with raw-source lineage

That gives later replay, checkpoint, held-out reporting, and ablation work one
explicit tokenized-stage contract.

## Split And Surface Discipline

Tokenized shards keep the split kind explicit.

The contract validates shard lineage against the held-out isolation surface:

- `train` and `validation` shards must stay valid for `model_training`
- `held_out` shards must stay valid for `benchmark_package`

This keeps tokenizer-only and evaluation-only sources from drifting silently into
the wrong tokenized split.

## Replay Contract

The replay contract carries:

- iteration mode
- shard ordering
- deterministic shuffle seed
- stable dataset identity

That identity is part of the manifest itself because checkpoint recovery needs a
stable dataset name, not only a folder path or loader flag.

## Mechanical Enforcement

`psionic-data` now validates that:

- shard lineage matches raw-source ids, families, and normalized digests
- shard tokenizer digests match the tokenizer bundle
- packing-policy version is preserved on every shard
- split aggregates match the declared shards
- source-family bindings cover exactly the family/source usage represented by
  the shards

That gives the `Psion` lane one explicit tokenized-stage substrate before SFT
and pretraining manifests expand on top of it.
