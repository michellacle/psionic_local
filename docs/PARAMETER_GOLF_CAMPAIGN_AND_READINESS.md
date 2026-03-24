# Parameter Golf Campaign And Readiness Surfaces

> Status: canonical typed campaign and final-readiness surface for the later
> `PGOLF_GOOGLE-10` / `#463` and `PGOLF_GOOGLE-11` / `#464` closeout steps,
> updated 2026-03-24 after landing
> `crates/psionic-train/src/parameter_golf_campaign.rs`.

This document records the repo-owned machine-readable surfaces that now sit
between raw Parameter Golf evidence bundles and the later “should we actually
submit this?” decision.

## What Landed

`psionic-train` now exposes:

- `ParameterGolfRecordCandidateFrozenConfig`
- `ParameterGolfRecordCandidateCampaignEvidence`
- `ParameterGolfRecordCandidateCampaignReport`
- `build_parameter_golf_record_candidate_campaign_report(...)`
- `ParameterGolfFinalReadinessAuditReport`
- `build_parameter_golf_final_readiness_audit_report(...)`

`psionic-train` now also ships:

- `crates/psionic-train/examples/parameter_golf_record_candidate_campaign_report.rs`
- `crates/psionic-train/examples/parameter_golf_final_readiness_audit.rs`

These surfaces are intentionally narrower than a green contest claim. They do
not replace the real hardware evidence. They keep the later campaign and final
readiness steps machine-readable once that evidence exists.

## Record-Candidate Campaign

The record-candidate campaign report freezes one exact candidate family:

- `submission_id`
- `benchmark_ref`
- `track_id`
- tokenizer reference
- accounting-posture reference
- run-recipe reference

It then binds repeated evidence bundles to that frozen candidate and checks:

- every run evidence report targets the same submission and track
- every promotion receipt targets the same submission, track, and benchmark
- every promotion receipt is tied to the same run id as its run-evidence report
- every carried evidence bundle binds one measured `8xH100` distributed receipt

The resulting campaign report makes the remaining gap explicit:

- `ready_for_readiness_audit`
- or `blocked` with ordered blocked reasons

Build one campaign report directly from the frozen candidate config plus paired
run-evidence and promotion-receipt JSONs:

```bash
cargo run -p psionic-train --example parameter_golf_record_candidate_campaign_report -- \
  campaign.parameter_golf.record_candidate.v1 \
  /tmp/frozen_candidate.json \
  /tmp/parameter_golf_record_candidate_campaign.json \
  /tmp/run_evidence_1.json /tmp/promotion_receipt_1.json \
  /tmp/run_evidence_2.json /tmp/promotion_receipt_2.json
```

## Final Readiness Audit

The final readiness audit composes:

- one frozen record-candidate campaign report
- one final PR-bundle report
- one local challenge-clone dry-run report

It keeps five gates explicit:

- `measured_8xh100_execution`
- `repeated_record_candidate_campaign`
- `promotion_posture`
- `final_pr_bundle_present`
- `local_clone_dry_run`

The audit then emits:

- `ready_to_submit`
- or `blocked` with ordered blocked reasons

Build the final readiness audit directly from the campaign report plus the
existing PR-bundle and local-clone dry-run reports:

```bash
cargo run -p psionic-train --example parameter_golf_final_readiness_audit -- \
  /tmp/parameter_golf_record_candidate_campaign.json \
  fixtures/parameter_golf/reports/parameter_golf_final_pr_bundle.json \
  fixtures/parameter_golf/reports/parameter_golf_local_clone_dry_run.json \
  /tmp/parameter_golf_final_readiness_audit.json
```

This is the intended boundary. Psionic now has one typed final decision
surface, but the surface only turns green when the upstream evidence is
actually present.

## Honest Boundary

This landed scaffolding does not close `#463` or `#464` by itself.

What is now explicit:

- how one future candidate family is frozen
- how repeated evidence bundles are tied to that candidate
- how the later final readiness decision will be computed

What is still separate work:

- real repeated `8xH100` evidence for the frozen candidate
- a promotable final promotion receipt
- the final exported-folder dry run backed by that real campaign
