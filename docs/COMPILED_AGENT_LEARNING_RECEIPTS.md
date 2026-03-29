# Compiled Agent Learning Receipts

> Status: canonical `AGENT-PLATFORM` receipt-to-replay substrate, updated 2026-03-28.

## Why This Exists

The compiled-agent vertical slice is not useful for XTRAIN unless real runtime
outputs can be normalized into Psionic-owned learning receipts and replay
bundles.

This layer is the first narrow handoff:

- raw compiled-agent run receipts from the app slice
- normalized learning receipts with expected behavior attached
- replay samples focused on the first two bounded training surfaces:
  route and grounded answer

## Canonical Fixtures

Source receipts captured from the existing `openagents` harness:

- `fixtures/compiled_agent/source/openagents_provider_ready_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_wallet_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_unsupported_receipt_v1.json`
- `fixtures/compiled_agent/source/openagents_negated_wallet_receipt_v1.json`

Normalized Psionic-owned artifacts:

- `fixtures/compiled_agent/compiled_agent_learning_receipts_v1.json`
- `fixtures/compiled_agent/compiled_agent_replay_bundle_v1.json`

## What The Ledger Keeps Explicit

- source receipt lineage and digest
- expected route and expected public response
- observed route, tool calls, tool results, and public response
- per-module correctness flags
- failure classes
- phase confidence map
- operator note for why the row matters

## What The Replay Bundle Keeps Explicit

- route replay samples
- grounded-answer replay samples
- behavioral-clone rows for already-correct behavior
- failure-correction rows for retained mistakes

The first correction row is deliberate:

- `openagents_negated_wallet_receipt_v1.json`

That raw source receipt exposed `wallet_status` for a negated unsupported
request. The replay bundle preserves that drift as a correction target instead
of hiding it.

## Entry Point

Regenerate both canonical fixtures from the repo root:

```bash
cargo run -q -p psionic-train --bin compiled_agent_receipt_to_replay
```

## Current Truth

- 4 retained source receipts
- 1 correction receipt
- 8 replay samples total
- first training surfaces are route and grounded answer only

This is enough to make the first bounded XTRAIN loop real without pretending
the whole serving stack or Tassadar lane must already be finished.
