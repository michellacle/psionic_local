# Compiled Agent Grounded-Answer Model

> Status: promoted replay-trained grounded-answer model, updated 2026-03-29.

## What This Is

This is the first retained learned artifact for the compiled-agent
`grounded_answer` module.

The scope stays deliberately narrow:

- strict fact inputs only
- provider readiness answers
- wallet balance and recent earnings answers
- unsupported refusal
- fallback on missing facts
- fallback on conflicting facts

## Model Shape

- family: `multinomial_naive_bayes`
- feature profile: `route_plus_fact_signature`
- artifact fixture:
  `fixtures/compiled_agent/compiled_agent_grounded_answer_model_v1.json`

The model does not read the raw user prompt. It learns over the bounded route
and the supplied tool facts only.

## Current Retained Metrics

- training rows: `19`
- held-out rows: `13`
- training accuracy: `1.0`
- held-out accuracy: `0.84615386`
- artifact digest:
  `1bcac89576e47ae4a1174a00077db3a389213ed2998bbafb5b76b199ea92839f`

## Why This Matters

Before this change, grounded answer was still effectively a bounded rule
revision.

Now the learning loop retains a real grounded-answer artifact that:

- learns the supported answer programs from replay-backed grounded samples
- keeps unsupported refusal inside the retained module family
- falls back when provider or wallet facts are missing
- falls back when wallet facts conflict instead of picking one arbitrarily

## Independent Validator Surface

The independent module eval now includes:

- standard provider and wallet grounding rows
- unsupported refusal
- insufficient provider facts fallback
- conflicting wallet facts fallback

The current grounded candidate report passes all `6/6` grounded-answer eval
rows.

## Current Validator State

- independent grounded-answer eval: `6/6`
- replay matches: `19/19`
- held-out matches: `11/13`
- decision in the retained XTRAIN cycle: `promote`

The grounded-answer model now clears the missing-facts and conflicting-facts
fallback rows while improving both replay and held-out grounded fidelity over
the bounded rule baseline.

## Entry Point

Regenerate the learned grounded-answer artifact and XTRAIN outputs:

```bash
cargo run -q -p psionic-train --bin compiled_agent_xtrain_loop
```

## Honest Boundary

This is a real learned module, but it is still a narrow bounded surface. It
does not claim broad natural-language synthesis, hidden prompt intelligence, or
general chat-model behavior.
