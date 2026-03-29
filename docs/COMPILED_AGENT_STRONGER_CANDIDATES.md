# Compiled Agent Stronger Candidates

> Status: stronger bounded candidate families evaluated and rejected on tie, updated 2026-03-29.

## Why This Exists

After the widened corpus landed, the next honest question was whether a
stronger bounded family could beat the incumbent learned candidates on the same
compiled-agent contracts.

This document records that comparison without reopening the architecture or the
task family.

## Candidate Families Compared

Incumbent route candidate:

- artifact: `compiled_agent.route.multinomial_nb_v1`
- family: `multinomial_naive_bayes`
- fixture: `fixtures/compiled_agent/compiled_agent_route_model_v1.json`

Stronger route candidate:

- artifact: `compiled_agent.route.tfidf_centroid_v1`
- family: `tfidf_centroid`
- fixture: `fixtures/compiled_agent/compiled_agent_route_tfidf_centroid_model_v1.json`

Incumbent grounded-answer candidate:

- artifact: `compiled_agent.grounded_answer.multinomial_nb_v1`
- family: `multinomial_naive_bayes`
- fixture: `fixtures/compiled_agent/compiled_agent_grounded_answer_model_v1.json`

Stronger grounded-answer candidate:

- artifact: `compiled_agent.grounded_answer.tfidf_centroid_v1`
- family: `tfidf_centroid`
- fixture:
  `fixtures/compiled_agent/compiled_agent_grounded_answer_tfidf_centroid_model_v1.json`

## Retained Comparison Surface

The stronger family was judged on the same bounded surfaces as the incumbent:

- independent module eval
- replay-backed training bundle
- held-out receipt split
- the normal bounded promotion rule

The retained machine-readable comparison is:

- `fixtures/compiled_agent/compiled_agent_stronger_candidate_family_report_v1.json`

## Current Retained Result

Route:

- incumbent eval: `4/4`
- stronger eval: `4/4`
- incumbent replay matches: `19/19`
- stronger replay matches: `19/19`
- incumbent held-out matches: `8/13`
- stronger held-out matches: `8/13`
- decision: `keep_incumbent`

Grounded answer:

- incumbent eval: `6/6`
- stronger eval: `6/6`
- incumbent replay matches: `19/19`
- stronger replay matches: `19/19`
- incumbent held-out matches: `11/13`
- stronger held-out matches: `11/13`
- decision: `keep_incumbent`

## Why The Stronger Family Was Rejected

The TF-IDF centroid family did not regress, but it also did not beat the
incumbent artifacts on any retained bounded surface.

That matters because this loop is supposed to improve the product through
evidence, not churn authority because a different model family sounds more
interesting.

The retained rule is therefore:

- promote a stronger family only if it improves at least one retained surface
  with no eval, replay, or held-out regressions
- keep the incumbent on ties

## Entry Point

Regenerate both the incumbent and stronger-family artifacts plus the retained
comparison report:

```bash
cargo run -q -p psionic-train --bin compiled_agent_xtrain_loop
```

## Honest Boundary

This does not claim that TF-IDF centroid is the last route or grounded-answer
family worth testing.

It does claim that the bounded loop can now evaluate stronger families without
changing the contracts, and can reject them honestly when the wider evidence
does not justify replacement.
