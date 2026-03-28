# Parameter Golf XTRAIN Track

This document freezes the bounded `XTRAIN -> PGOLF` score lane that now emits
the shared `v2` remote-training visualization family.

## Stable Track Identity

- track id: `parameter_golf.promoted_general_xtrain.quick_eval_window1.v1`
- track family: `xtrain`
- execution class: `bounded_train_to_infer`

## What This Track Measures

The bounded XTRAIN track measures one local-reference train-to-infer lane over
the promoted PGOLF family:

- training config: `ParameterGolfReferenceTrainingConfig::xtrain_promoted_general_small_decoder_baseline()`
- family contract: `docs/PARAMETER_GOLF_PROMOTED_FAMILY_CONTRACT.md`
- retained score metric: `parameter_golf.validation_bits_per_byte`
- score direction: lower is better
- decode posture: one-token bounded attention window

The retained score is the repo-owned quick-eval BPB over the local-reference
cycle fixture. It is not a public PGOLF leaderboard score and it is not a
HOMEGOLF score-closeout.

## Comparability

- comparability class: `same_track_comparable`
- public equivalence class: `not_public_equivalent`

This means:

- compare XTRAIN runs to other XTRAIN runs on this same track
- do not compare this score directly to HOMEGOLF or public leaderboard rows as
  if they were the same law
- keep promotion held until the repo has retained evidence that XTRAIN is score
  relevant to the downstream competitive loop

## Proof Posture

- proof posture: `bounded_train_to_infer`

The lane is strong enough to prove:

- the promoted PGOLF-shaped bundle is trained
- the emitted bundle loads for inference
- direct runtime and served runtime keep exact token parity on the retained eval
- the lane has one closed-out local-reference BPB

The lane does not prove:

- public-score equivalence
- HOMEGOLF score relevance
- challenge-track legality
- exact-cycle task correctness

## Current Honest Promotion Rule

Bounded XTRAIN stays visible in the shared dashboard grammar, but promotion
remains held until retained evidence closes the score-path question tracked in
`#639`.
