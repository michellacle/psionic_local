# OpenAgentsGemini Query-Backed Google Single-Node Cost Receipt Audit

> Status: follow-up Google single-node audit written on 2026-03-23 after the
> repo closed the machine-queryable per-run cost receipt gap for the current
> accelerated `Psion` lanes on `openagentsgemini`.

## Scope

This audit covers the March 23 follow-up tranche that turned the retained
Google single-node cost receipt from a partial or failed lookup into a real
machine-queryable bounded evidence surface.

It claims:

- the Google host finalizer now retains a real run-cost receipt for the current
  accelerated single-node lanes
- the first failed pricing lookups were traced to a real service-account IAM
  gap instead of hand-waved away
- the Google training identity and local preflight were tightened so future
  launches reject that gap before a paid run starts
- the generic accelerated lane and the host-native accelerated
  plugin-conditioned lane now both have priced follow-up runs with retained
  BigQuery profile binding plus runtime-priced estimates

It does not claim:

- invoice-grade billing-export realized cost truth
- mixed guest-artifact accelerated cost truth
- cluster-scale cost truth
- broad cost optimality across machine families

## Why This Follow-Up Was Necessary

The March 23 accelerated audits proved real `cuda` execution, but they still
left one important operator gap:

- the retained Google runs had runtime windows
- the project had a BigQuery price-profile table
- but the host itself could not always bind the run to a query-backed price row

That meant the repo had honest accelerator truth, but only partial cost truth.

For a bounded single-node lane this is the right next thing to close before
spending more time on broader scale-up work.

## Typed Outcome

- tranche result: `bounded_success`
- root cause class: `training_identity_iam_gap`
- fixed surface:
  `psion.google_run_cost_receipt.v1`
- current retained proof class:
  `machine_queryable_catalog_price_profile_times_observed_runtime`

## Run History

### 1. First retained host-native cost-receipt attempt exposed an incomplete query path

- run id: `psion-g2-l4-plugin-host-native-accelerated-20260323t085045z`
- training result: `bounded_success`
- accelerator result: passed
- cost receipt result: `query_failed`
- retained price profile row: `null`

This run proved the receipt object existed and was retained, but the lookup
path was not yet strong enough to preserve a useful error or a bound price row.

### 2. Second retained host-native attempt exposed the real IAM gap

- run id: `psion-g2-l4-plugin-host-native-accelerated-20260323t090722z`
- training result: `bounded_success`
- accelerator result: passed
- cost receipt result: `query_failed`
- retained query error:
  `bq query failed for g2_l4_single_node_plugin_host_native_accelerated`

Manual in-VM reproduction then showed the actual root cause:

- the training service account did not have `bigquery.jobs.create`
- the host therefore could not run the FinOps price-profile query even though
  the table existed and the operator could query it locally

This was the critical proof that the remaining failure was not in Google GPU
allocation, not in the accelerated trainer, and not in the BigQuery table
contents. It was an identity-contract bug.

## Hardening Landed In The Repo

The repo-side contract is now tighter in three places:

### 1. Training identity profile now declares the FinOps roles explicitly

`fixtures/psion/google/psion_google_training_identity_profile_v1.json` now
requires:

- `roles/bigquery.dataViewer`
- `roles/bigquery.jobUser`
- `roles/logging.logWriter`
- `roles/monitoring.metricWriter`

### 2. Service-account bootstrap now applies the declared identity contract

`scripts/psion-google-ensure-training-service-account.sh` now reads the
required project and bucket roles from the committed identity profile instead
of partially hardcoding them.

### 3. Operator preflight now rejects missing host IAM before launch

`scripts/psion-google-operator-preflight.sh` now verifies that the training
service account actually has:

- the declared project roles from the identity profile
- the declared bucket roles from the identity profile

That means the operator lane now fails fast on the exact gap that caused the
earlier partial cost receipts.

## Validation Performed

The follow-up tranche used the repo-owned path all the way through:

- `bash scripts/psion-google-ensure-training-service-account.sh`
- `bash scripts/psion-google-operator-preflight.sh --profile g2_l4_single_node_plugin_host_native_accelerated --zone us-central1-a`
- in-VM query smoke test on the live host after the IAM repair
- one new host-native accelerated Google run
- one new generic accelerated Google run
- explicit teardown after final manifests landed

The direct in-VM query smoke test succeeded after the role repair and returned
the expected `g2_l4_single_node_plugin_host_native_accelerated` price-profile
row from:

- `openagentsgemini.psion_training_finops.single_node_price_profiles_v1`

## Successful Follow-Up Proofs

### Host-native accelerated plugin-conditioned follow-up

- run id: `psion-g2-l4-plugin-host-native-accelerated-20260323t092812z`
- result classification: `bounded_success`
- final manifest:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-plugin-host-native-accelerated-20260323t092812z/final/psion_google_run_final_manifest.json`
- cost query status: `price_profile_found`
- runtime-priced estimate: `0.1867276189313889 USD`
- launch-to-teardown runtime: `763 s`
- bootstrap runtime: `522 s`
- training runtime: `165 s`
- checkpoint runtime: `25 s`
- teardown runtime: `3 s`
- within declared ceiling: `true`
- stage backend: `cuda`
- post-warmup non-zero GPU utilization samples: `25 / 161`
- post-warmup non-zero GPU memory samples: `161 / 161`
- max GPU memory used: `1286 MiB`
- mean throughput: `20584 tokens/sec`

### Generic accelerated single-node follow-up

- run id: `psion-g2-l4-accelerated-20260323t094345z`
- result classification: `bounded_success`
- final manifest:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-accelerated-20260323t094345z/final/psion_google_run_final_manifest.json`
- cost query status: `price_profile_found`
- runtime-priced estimate: `0.15001838847305557 USD`
- launch-to-teardown runtime: `613 s`
- bootstrap runtime: `535 s`
- training runtime: `4 s`
- checkpoint runtime: `21 s`
- teardown runtime: `4 s`
- within declared ceiling: `true`
- stage backend: `cuda`
- post-warmup non-zero GPU utilization samples: `3 / 4`
- post-warmup non-zero GPU memory samples: `4 / 4`
- max GPU memory used: `202 MiB`
- mean throughput: `94929 tokens/sec`

Both VMs were deleted after the final manifests landed.

## What Is Now True

The correct current claim is:

- the Google single-node accelerated lanes now retain a machine-queryable
  bounded cost receipt that multiplies a committed BigQuery catalog price row
  by the observed runtime windows of the run

That is enough to support:

- bounded single-node operator truth
- bounded per-run postmortem cost comparison
- explicit declared-ceiling checks in the retained evidence bundle

## What Is Still Not True

This follow-up still does **not** justify stronger cost claims than the system
actually earned.

It does **not** mean:

- the repo has invoice-grade realized-cost truth from billing export
- the repo has cluster-scale cost accounting
- the mixed guest-artifact lane has accelerated cost truth
- a later A100 or multi-host lane has already inherited this same receipt path

## Bottom Line

The March 23 cost follow-up closed the real remaining single-node cost-truth
gap.

The repo now has, for the current real accelerated Google lanes:

- real `cuda` backend proof
- real GPU utilization and residency proof
- real throughput proof
- real checkpoint and evidence retention
- real machine-queryable bounded per-run cost receipts

That is the right stopping point for this tranche.

The next honest frontier is no longer "can we price a real accelerated
single-node run?" It is:

- whether to accelerate the mixed guest-artifact lane
- how to grow the proof-sized plugin-conditioned corpus materially
- whether any later larger lane preserves the same cost-receipt honesty
