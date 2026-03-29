# Compiled Agent Route Model

> Status: retained replay-trained compiled-agent route candidate, updated 2026-03-29.

## What This Is

This is the first compiled-agent module candidate that is backed by a trained
artifact instead of a hand-authored rule delta.

The current scope stays deliberately narrow:

- module family: `route`
- task family: `provider_status` vs `wallet_status` vs `unsupported`
- training source: replay-backed route samples from
  `fixtures/compiled_agent/compiled_agent_replay_bundle_v1.json`

## Model Shape

- family: `multinomial_naive_bayes`
- feature profile: `unigram_plus_bigram`
- artifact fixture: `fixtures/compiled_agent/compiled_agent_route_model_v1.json`
- training rows: `19`
- held-out rows: `13`

The model is trained from retained replay samples, not from hidden heuristics.
It converts the replay bundle into token and bigram counts, estimates per-route
feature likelihoods, and emits a stable JSON artifact that XTRAIN can validate
and promote.

## Current Retained Metrics

- training accuracy: `1.0`
- held-out accuracy: `0.61538464`
- artifact digest:
  `0ef312a77e31e683ddd40225acc69be01ed74c428d8049bad8c9c8550c568d1f`

## Why It Exists

The previous route candidate fixed the negated wallet failure with a bounded
keyword patch. That was useful, but it was not a real trained model.

The route model closes that gap:

- new receipts, runtime rows, and replay samples can change the trained artifact
- the validator still gates promotion with independent module evals and replay
  matches
- the candidate can improve over time without pretending the whole agent has
  become self-modifying

## Current Validator State

- independent route eval: `4/4`
- replay matches: `19/19`
- held-out matches: `8/13`
- decision in the retained XTRAIN cycle: `hold`

The route candidate fixes the retained negated-wallet false positive and clears
the bounded route eval, but it still ties the baseline on the widened held-out
split because one comparison receipt remains ambiguous:

- `receipt.compiled_agent.learning.openagents_wallet_provider_compare_heldout_receipt_v1`

That is why the promoted artifact contract still keeps route authority on the
baseline revision while the learned route model stays in retained candidate
state.

## Entry Point

Regenerate the trained artifact and the XTRAIN outputs:

```bash
cargo run -q -p psionic-train --bin compiled_agent_xtrain_loop
```

## Honest Boundary

This artifact is retained and validator-scored, but it is not silently
promoted just because it is learned.

It is still a narrow route model over a bounded task family. The right next
step is continued evidence growth and repeated bounded retraining until the
held-out ambiguity is resolved cleanly enough for promotion.
