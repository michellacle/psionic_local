# Psion Corpus Admission

> Status: implemented_early as of 2026-03-22 via `PSION-2` / issue `#358`.

This document freezes the first repo-owned admission contract for `Psion`
training sources.

It defines:

- the source classes the lane is allowed to review
- the rights postures and review outcomes that must stay explicit
- the required metadata fields for any reviewed source entry
- the exclusion and removal triggers that may not live only in reviewer memory
- the minimum workflow for admitting or rejecting a source family

This issue does **not** implement raw ingestion, tokenizer training, or model
training. It freezes the policy surface those later stages must obey.

## Canonical Artifacts

- machine-readable types: `crates/psionic-data/src/psion_corpus_admission.rs`
- canonical policy fixture:
  `fixtures/psion/corpus_admission/psion_corpus_admission_policy_v1.json`
- example reviewed-source manifest:
  `fixtures/psion/corpus_admission/psion_source_admission_manifest_v1.json`

The stable schema version is `psion.corpus_admission.v1`.

The stable policy id and version are:

- `policy_id = psion.corpus_admission`
- `policy_version = v1`

## Supported Source Classes

The first policy admits review only for the following source families:

- `textbook`
- `specification`
- `manual`
- `paper`
- `technical_documentation`
- `code`
- `postmortem`
- `expert_discussion`
- `reference_implementation`

This is intentionally narrower than "anything technical on the internet."

## Rights Posture

Every reviewed source must carry exactly one explicit rights posture:

- `internal_training_only`
- `internal_training_and_internal_serving`
- `redistribution_derived_artifacts_allowed`
- `evaluation_only`
- `tokenizer_only`
- `rejected`

This keeps training, serving, redistribution-derived artifacts, evaluation, and
tokenizer exposure separate instead of collapsing them into one yes/no flag.

## Required Metadata

Any reviewed source entry must surface the following metadata fields:

- `source_id`
- `source_family_id`
- `source_kind`
- `title`
- `author_or_origin`
- `publication_year_or_revision`
- `language`
- `canonical_reference`
- `content_digest`
- `rights_posture`
- `normative_role`
- `boundary_kind`
- `review_decision`
- `reviewer`
- `decision_summary`

The `boundary_kind` field is required now because later held-out isolation and
source-removal review depend on stable chapter, page, file, or record anchors.

The `content_digest` field is required now because later removal and re-ingest
review must bind to actual reviewed source payloads, not only titles.

## Review Outcomes

Every reviewed source must end in one explicit decision:

- `admitted`
- `admitted_with_restrictions`
- `rejected`

`evaluation_only` and `tokenizer_only` sources must use
`admitted_with_restrictions` so the policy surface does not pretend they are
general training sources.

`rejected` sources must keep `rights_posture = rejected` explicit.

## Exclusion Triggers

The first policy requires exclusion when one or more of the following is true:

- rights posture is missing or unresolved
- provenance cannot be checked
- stable digest or stable reference is missing
- stable section, page, file, or record boundaries are missing
- the source would contaminate held-out or benchmark material
- the source is a near-duplicate with no canonical-selection decision
- the source is dominated by social noise, SEO repetition, or low-signal text
- the source is generated without bounded parent lineage
- the source introduces sensitive or personal data outside the lane boundary
- the source is out of scope for `Psion`
- the required review path did not complete

## Removal Triggers

Previously reviewed sources must be reconsidered or removed when one or more of
the following becomes true:

- rights changed or were revoked
- provenance was invalidated
- content digest no longer matches the reviewed source
- benchmark or held-out contamination is discovered later
- the original review decision or classification was wrong
- the source was retracted or replaced canonically
- boundary metadata proved too weak for later isolation work

## Minimum Workflow

The minimum review workflow is:

1. intake metadata review
2. provenance review
3. rights review
4. quality review
5. contamination precheck
6. final decision

Later issues may add more workflow detail, but later work may not skip these
steps and still describe the source as admitted under `Psion`.

## Mechanical Enforcement

The first policy is not doc-only.

`psionic-data` now exposes:

- `PsionCorpusAdmissionPolicy`
- `PsionCorpusAdmissionManifest`
- `PsionSourceAdmissionRecord`
- `PsionCorpusAdmissionError`

`PsionCorpusAdmissionManifest::validate_against_policy(...)` rejects:

- missing required metadata on reviewed source entries
- duplicate `source_id` rows
- unsupported source classes or rights postures
- admitted records that try to use `rejected` posture
- `tokenizer_only` or `evaluation_only` rows that fail to stay marked
  `admitted_with_restrictions`
- rejected rows that fail to keep `rights_posture = rejected`

That gives the repo one explicit admission boundary before raw-source ingestion
expands in later `PSION-*` work.
