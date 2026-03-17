# Tassadar Economic-Kernel Workload Audit

Date: March 17, 2026

## Result

If Tassadar is going to matter to OpenAgents economically, the center of
gravity now needs to move away from Sudoku and toward deterministic programs
that sit directly under the compute market and economic kernel.

The valuable next workloads are not generic chat tasks and not more puzzle
families for their own sake. They are market-clearing, routing,
policy-checking, delivery-proof verification, and dispute-replay programs whose
outputs can feed kernel and Nexus authority as typed candidate facts.

This audit is directional only. It does not widen the current machine-readable
acceptance truth by itself.

## Boundary

The architecture split still matters:

- Psionic owns execution truth, runtime truth, artifact truth, proof posture,
  topology truth, refusal truth, and replay truth.
- Kernel and Nexus own accepted outcomes, contracts, liability, settlement, and
  final economic authority.

So the right Tassadar target is not "be the settlement engine inside Psionic."
The right target is:

`kernel-adjacent executors`

These are deterministic programs whose traces produce candidate receipts,
eligibility results, cost calculations, route decisions, replay evidence, and
challenge facts that the authority layer can later accept or reject.

## Why Sudoku Is No Longer Enough

Sudoku was the right early benchmark because it stresses:

- long traces
- branching
- backtracking
- exactness under pressure

That was useful for proving the executor substrate exists.

It is no longer the right center of gravity for OpenAgents value because it
does not map closely enough to:

- compute products
- capacity lots and structured capacity instruments
- delivery proofs
- validator policy
- challenge windows
- topology-aware route selection
- economic receipts

A green Sudoku lane proves "the executor can run hard traces." It does not yet
prove "the executor improves cost per correct economic episode."

## Most Valuable Next Workload Families

### 1. Capacity Allocation And Market Clearing

The first serious target should be compute-market allocation, not another
puzzle family.

- Canonical shape: min-cost assignment, min-cost flow, or constrained matching
  over demand jobs, provider capacity lots, topology classes, time windows,
  capability envelopes, and price vectors.
- Why it matters: this turns Hungarian-style exactness into something directly
  legible for compute procurement and saleable execution products.
- Suggested first artifact family: `capacity_lot_allocation_v0`.
- Why Tassadar fits: this workload is structured, stateful, auditable, and has
  clear final outputs plus clear refusal cases.

This is the cleanest bridge from "exact matching demo" to "economically useful
compute-market program."

### 2. Delivery-Proof And Receipt Verification

The next high-value kernel-adjacent executor is verifier logic.

- Canonical shape: verify digests, manifest linkage, staged artifact receipts,
  topology claims, runtime bindings, and proof-bundle completeness.
- Why it matters: this turns Tassadar from a solver demo into an evidence
  checker for the part of OpenAgents that already sells execution truth.
- Suggested first artifact family: `delivery_proof_verifier_v0`.
- Why Tassadar fits: verifier programs are deterministic, loop-heavy,
  branch-heavy, and produce typed pass/fail outputs with explicit reason codes.

This is especially attractive because it directly compounds Psionic's existing
strength in artifacts, proofs, manifests, and replay.

### 3. Policy And Eligibility Checking

OpenAgents will need deterministic evaluators for whether a run satisfies the
declared policy envelope around that run.

- Canonical shape: a bounded policy program over environment compatibility,
  proof posture, freshness, benchmark eligibility, route class, budget class,
  and refusal reasons.
- Why it matters: exact executor traces become productively useful only when
  they can feed accept/refuse decisions without hidden ad hoc app logic.
- Suggested first artifact family: `execution_policy_checker_v0`.
- Why Tassadar fits: policy interpreters are branch-heavy, stateful,
  auditable, and naturally produce machine-checkable traces.

This gives the economic kernel a better upstream substrate without moving final
authority into Psionic.

### 4. Challenge Replay And Dispute Verification

A real market needs reproducible dispute handling, not only first-pass
execution.

- Canonical shape: replay a contested run bundle, recompute digests, outputs,
  and validator checks, then emit exact divergence points.
- Why it matters: this is where executor traces become economically meaningful
  instead of only benchmark-friendly.
- Suggested first artifact family: `challenge_replay_verifier_v0`.
- Why Tassadar fits: replay is exactly the kind of append-only state machine
  Tassadar already models well.

This is also one of the best ways to make decentralized clusters trustworthy:
not by hiding disagreement, but by making challenge replay exact and cheap.

### 5. Topology-Aware Route Feasibility

Psionic already owns topology truth, so one valuable executor family is route
feasibility and placement verification.

- Canonical shape: choose or verify a placement under bandwidth, locality,
  hardware capability, queue, and latency constraints.
- Why it matters: this is the bridge between decentralized clusters and
  economically saleable execution products.
- Suggested first artifact family: `topology_route_feasibility_v0`.
- Why Tassadar fits: these are structured search and feasibility checks with
  clear refusal cases and measurable cost per correct placement.

This is more valuable than another pure puzzle because it feeds directly into
provider capability and product identity.

### 6. Risk-Envelope And Reserve Update Programs

The economic kernel eventually cares about bounded, auditable risk transitions.

- Canonical shape: deterministic updates of exposure counters, reserve buckets,
  collateral requirements, coverage throttles, or bounded liability state from
  accepted receipts.
- Why it matters: this links execution truth to treasury and risk operations
  without making Psionic the settlement authority.
- Suggested first artifact family: `risk_envelope_update_v0`.
- Why Tassadar fits: this is low-entropy, state-machine-style logic where
  exactness matters more than language fluency.

This family should come after the earlier receipt and policy executors, not
before them.

## Recommended Priority Order

1. `capacity_lot_allocation_v0`
2. `delivery_proof_verifier_v0`
3. `execution_policy_checker_v0`
4. `challenge_replay_verifier_v0`
5. `topology_route_feasibility_v0`
6. `risk_envelope_update_v0`

The rationale is straightforward:

- the first four directly improve compute-market closure and validator posture
- the fifth connects decentralized cluster reality to saleable execution truth
- the sixth becomes much more valuable once the earlier receipt and policy
  layers are already explicit

## How To Keep This Honest

The same claim hygiene rules that made Tassadar credible so far should stay in
force here.

- Compiled and proof-backed lanes should lead first for every
  economic-kernel-adjacent workload.
- Learned lanes should come later as proposer, reranker, or bounded fast-path
  candidates, not as the first authority-bearing surface.
- Every workload family should ship with:
  - source program
  - canonical Wasm artifact
  - validated program artifact
  - trace ABI binding
  - benchmark corpus
  - exactness report
  - compatibility report
  - claim-boundary report
  - typed refusal examples
- Served executor outputs should be candidate execution facts, not accepted
  market truth.
- Kernel and Nexus should remain the place that accepts, rejects, settles, or
  escalates those facts.

## Concrete Next Benchmark Thesis

The best single next benchmark is not "harder Sudoku."

It is one article-class compute-market episode that looks like this:

1. input demand bundle plus provider-capability snapshot
2. compiled allocation program emits a placement or refusal trace
3. delivery-proof verifier checks the resulting artifact bundle
4. policy checker evaluates the declared eligibility and freshness rules
5. challenge replay reproduces the episode exactly from the committed artifacts

If Psionic can run that sequence internally, publish the traces, and hold exact
parity against the CPU reference lane, then Tassadar starts to matter as an
economic-kernel substrate rather than only as an executor research lane.

## Final Judgment

The path beyond Sudoku is not "more abstract reasoning."

It is:

`exact, bounded, economically legible programs`

that sit directly between Psionic execution truth and kernel economic truth.

The winning workloads are the ones that reduce cost per correct market episode,
increase auditability, and turn decentralized execution into comparable,
policy-bounded, challengeable receipts.
