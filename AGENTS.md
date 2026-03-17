# Psionic Agent Contract

## Scope

- This repository is the standalone `psionic` workspace extracted from
  `openagents`.
- Primary scope is `crates/psionic-*`, `docs/`, `fixtures/`, and the small set
  of repo-local `scripts/`.
- Psionic owns the machine-facing execution substrate: tensor and compiler
  contracts, runtime and backend truth, serving interfaces, cluster and sandbox
  execution, adapters, data, eval, research, and the early training substrate.
- Keep app UX, wallet or payout flows, market orchestration, and kernel or
  settlement authority out of this repo unless the user explicitly asks for
  cross-repo work.

## Canonical Docs

- `README.md` is the Psionic entrypoint and map.
- `docs/ARCHITECTURE.md` is the canonical Psionic-wide system spec.
- `docs/FRAMEWORK_CORE_ACCEPTANCE_MATRIX.md` is the canonical framework-core
  completion bar.
- `docs/INFERENCE_ENGINE.md` is the canonical inference-engine completion doc.
- `docs/TRAIN_SYSTEM.md` is the canonical training subsystem spec.
- Domain-specific docs in `docs/` define the contract for their subsystem.
- `docs/audits/` explains rationale and follow-on direction, but audits are not
  the canonical current-state spec.

## Workspace Map

- Core execution path lives in crates such as `psionic-core`, `psionic-array`,
  `psionic-ir`, `psionic-compiler`, `psionic-runtime`, and the backend crates.
- Distributed and execution plumbing lives in crates such as
  `psionic-cluster`, `psionic-collectives`, `psionic-distributed`,
  `psionic-datastream`, `psionic-sandbox`, and `psionic-net`.
- Serving and provider-facing compute surfaces live in crates such as
  `psionic-serve`, `psionic-provider`, `psionic-router`, and
  `psionic-catalog`.
- Data, model, training, eval, environment, and research lanes live in crates
  such as `psionic-data`, `psionic-models`, `psionic-train`,
  `psionic-environments`, `psionic-eval`, `psionic-research`,
  `psionic-adapters`, and `psionic-apple-fm`.
- `fixtures/` contains committed evidence, run bundles, and compatibility
  artifacts. Treat them as versioned substrate truth, not disposable samples.

## Execution Rules

- Read the relevant canonical doc before editing a subsystem.
- Keep machine-legible truth explicit: manifests, receipts, proofs, capability
  reports, refusal reasons, runtime identity, artifact identity, and lineage
  should remain accurate and deterministic.
- Preserve replay-safe and deterministic behavior. Do not hide fallbacks,
  bounded support, or refusal posture behind optimistic defaults.
- Prefer extending existing crate boundaries over adding cross-cutting
  shortcuts or hidden control planes.
- If behavior or architecture changes, update the relevant doc in `docs/` in
  the same change.
- Use the repo status vocabulary consistently:
  `implemented`, `implemented_early`, `partial`,
  `partial_outside_psionic`, and `planned`.

## Extraction Notes

- Some Psionic docs still mention historical `openagents` files or scripts that
  are not present in this standalone repo.
- Treat those references as historical context, not live dependencies for work
  in this repository.
- Do not pull code or docs back from `openagents` by default. Restore or copy
  material only when the user explicitly asks for it.

## Validation

- Prefer targeted validation first: `cargo test -p <crate>` or the specific
  test, example, or fixture path that covers the touched subsystem.
- Use `cargo test --workspace` when the change is broad enough and the local
  environment makes that realistic.
- If you change fixtures, reports, or schema-like contracts, verify the code
  that produces and consumes them.
