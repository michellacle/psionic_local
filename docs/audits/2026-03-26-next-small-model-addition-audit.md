# 2026-03-26 Next Small Model Addition Audit

## Intent

This audit answers one concrete repo-planning question:

> after reading `~/code/alpha/psionic/smallmodels.md` and checking the current
> Psionic tree, what is the best logical next small model type to add as an
> explicit named row?

The answer should optimize for current Psionic reality.

It should not optimize for abstract model taste.

It should close the largest remaining validation gap with the least new
substrate work.

## Decision

The next model type to add should be a **SmolLM2 Llama-family instruct row**,
anchored on **`SmolLM2-360M-Instruct`**.

That is the right next step because:

- it widens Psionic from the already-landed Qwen pilot into the still-unpiloted
  Llama branch
- it stays inside the model families Psionic already executes today
- it fits the small-model health-check use case from `smallmodels.md`
- it fits the medium-small model band that Psion's own later training plan says
  matters more than toy-only checkpoints
- it stays aligned with the repo's anti-code-dominance training posture better
  than a coder-specialized checkpoint

If the repo wants a two-step ladder instead of one anchor, the correct pairing
is:

1. `SmolLM2-135M` for the absolute-minimum smoke path
2. `SmolLM2-360M-Instruct` for the first real named supported row

The primary recommendation is still `SmolLM2-360M-Instruct`.

## Sources Reviewed

Local planning input:

- `~/code/alpha/psionic/smallmodels.md`

Canonical Psionic docs:

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/INFERENCE_ENGINE.md`
- `docs/LLAMA_VLLM_SGLANG_INFERENCE_SPEC.md`
- `docs/MLX_MODEL_CATALOG.md`
- `docs/MLX_TEXT_SERVE.md`
- `docs/NON_GPT_OSS_QWEN_PILOT.md`
- `docs/PSION_COMPACT_DECODER.md`
- `docs/PSION_PRETRAIN_STAGE.md`
- `docs/PSION_SAMPLING_POLICY.md`
- `docs/PSION_DECENTRALIZED_CONTRIBUTION.md`
- `docs/audits/2026-03-22-psion-training-system-full-state-audit.md`

Relevant code paths:

- `crates/psionic-models/src/lib.rs`
- `crates/psionic-models/src/harmony.rs`
- `crates/psionic-serve/src/gguf.rs`
- `crates/psionic-serve/src/openai_http.rs`
- `crates/psionic-mlx-catalog/src/lib.rs`
- `crates/psionic-mlx-lm/src/lib.rs`
- `crates/psionic-mlx-serve/src/lib.rs`

External model references reviewed after the repo pass:

- `https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct`
- `https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/main/config.json`
- `https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF`
- `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF`
- `https://huggingface.co/EleutherAI/pythia-160m`

## Current Repo State

The current repo is already past the question "can Psionic execute anything
except GPT-OSS."

That part is closed.

The important current facts are these:

### 1. The generic GGUF runtime already executes three non-GPT-OSS decoder families

`psionic-models` classifies `llama`, `qwen2`, `mistral`, and `gpt-oss` GGUF
artifacts into concrete decoder families.

`psionic-serve/src/gguf.rs` then routes:

- `GgufDecoderFamily::GptOss` into the GPT-OSS-specific service
- `GgufDecoderFamily::Llama`
- `GgufDecoderFamily::Qwen`
- `GgufDecoderFamily::Mistral`

into the shared dense CPU GGUF path.

This means the next model type should not require a new low-level decoder
family just to become runnable.

### 2. Qwen is already the first explicit non-GPT-OSS pilot

`docs/NON_GPT_OSS_QWEN_PILOT.md` and
`scripts/release/check-psionic-qwen-pilot.sh` already freeze Qwen as the first
named non-GPT-OSS end-to-end pilot.

That matters because another Qwen row does not widen family coverage very much.

It would mostly deepen a lane Psionic already proved.

### 3. The generic server and MLX-facing package surfaces are already real

`crates/psionic-serve/src/openai_http.rs` ships the generic CPU OpenAI server.

It already fronts decoder models through:

- `/v1/chat/completions`
- `/v1/responses`

`crates/psionic-mlx-catalog` and `crates/psionic-mlx-serve` already wrap that
same serving lane with MLX-style model reference resolution.

So the best next model row is one that uses this existing path directly instead
of demanding a new serving substrate.

### 4. Dense CPU adapter hosting already works for Llama, Qwen, and Mistral

`crates/psionic-serve/src/gguf.rs` already admits LM-head LoRA hosting on dense
CPU GGUF families and explicitly allows:

- Llama
- Qwen
- Mistral

That makes a dense small model more useful than a family that only helps
metadata or benchmark stories.

### 5. The Psion learned lane still wants small serious models, not only toy smoke models

`docs/audits/2026-03-22-psion-training-system-full-state-audit.md` keeps the
planned scale bands explicit:

- pilot models around `50M` to `150M`
- later serious internal models around `300M` to `1B`

That means the best next external small model row is not just "the tiniest file
possible."

It should also sit near the first serious small-model band.

### 6. The Psion training policy explicitly resists code-assistant drift

`docs/PSION_SAMPLING_POLICY.md` caps code-token dominance and rejects coding
gains that hide reasoning regressions.

That makes a coder-specialized checkpoint a weak choice for the first next row.

The repo does track coding fluency, but it does not want the learned lane to
drift into a generic coding assistant.

## Selection Criteria

The right next model type should satisfy six constraints at once:

1. It should use a family Psionic already executes.
2. It should widen validation coverage beyond the existing Qwen pilot.
3. It should fit the 8GB-laptop health-check use case from `smallmodels.md`.
4. It should have low artifact friction and a clean license story.
5. It should help later adapter-window or small-train experiments.
6. It should align with Psion's reasoning-heavy and spec-heavy posture.

That rules out several otherwise attractive candidates.

## Candidate Assessment

### `Qwen2.5-0.5B-Instruct`

This is the cleanest **artifact** candidate.

The reasons are real:

- official GGUF exists
- Apache-2.0 is clean
- the Qwen family is already deeply wired into Psionic's prompt and tokenizer
  handling
- the MLX catalog docs already use small Qwen examples

But it is not the best **next row**.

The repo already spent its first explicit non-GPT-OSS pilot on Qwen. Adding a
second Qwen row next mostly deepens one already-proved branch instead of
widening substrate coverage.

`Qwen2.5-0.5B-Instruct` is the right fallback if artifact friction is the only
priority.

It is not the best next move if the goal is repo sequencing.

### `Qwen2.5-Coder-0.5B-Instruct`

This is a useful later variant.

It is not the best next default row.

The conflict is straightforward:

- it still duplicates the already-proved Qwen family
- it pulls the next named row toward code-first behavior
- the Psion lane explicitly tries to avoid code-dominant drift

That does not make it a bad model.

It makes it the wrong next model for this repo's current sequencing.

### `Pythia`

Pythia is the strongest **research transparency** candidate.

It has real advantages:

- explicit checkpoint ladder
- scaling-suite value
- training-dynamics value
- Apache-2.0

But it is still the wrong next addition.

The reason is structural, not aesthetic.

Pythia is a `GPTNeoXForCausalLM` family model, and Psionic does not currently
ship a NeoX-family GGUF decoder path, prompt path, or named server row.

Adding Pythia next would force new family work before it delivered the first
named row.

That is too much new substrate for the current gap.

Pythia belongs later, after the repo chooses to widen beyond the current
Llama/Qwen/Mistral/GPT-OSS decoder set.

### `SmolLM2`

SmolLM2 is the best fit because it closes the right gap.

The critical fact is not just that SmolLM2 is small.

The critical fact is that `SmolLM2-360M-Instruct` is a **Llama-family**
checkpoint.

The official config declares:

- `architectures = ["LlamaForCausalLM"]`
- `model_type = "llama"`

The official GGUF row also exists and is marked:

- architecture: `llama`
- license: `apache-2.0`

That means SmolLM2 lands directly on a decoder family Psionic already executes,
while also widening coverage beyond the Qwen pilot.

This is the exact combination the repo needs.

## Why `SmolLM2-360M-Instruct` Beats The Other SmolLM2 Sizes

The size ladder matters.

The three meaningful options are:

- `135M`
- `360M`
- `1.7B`

### Why not `135M` as the primary row

`135M` is excellent for smoke tests.

It is too weak as the first named support row.

The problems are practical:

- it is closer to a loader and request-plumbing sentinel than to a meaningful
  assistant-quality checkpoint
- it undershoots the repo's own later serious small-model band
- it is more likely to create false negatives in route, refusal, or structured
  serving probes because the model is simply too small

`135M` should exist as a companion smoke artifact, not as the main named row.

### Why not `1.7B` first

`1.7B` is still small enough for a lot of local work.

It is larger than the repo needs for the next step.

It weakens the main reasons to add a next small model now:

- fastest local validation
- lowest-friction laptop bring-up
- easy CI or developer-machine reproduction

The next row should stay clearly inside the "small and cheap to validate"
envelope.

### Why `360M` is the correct center

`360M` is the correct middle point.

It gives Psionic:

- a still-small on-device model
- a real instruct-tuned checkpoint rather than a bare base model
- a model large enough to matter for structured-serving and route probes
- a size that sits much closer to the repo's later `300M` to `1B` serious band

That is the right tradeoff.

## Why This Beats A Second Qwen Row

The repo already has a Qwen pilot because Qwen was the cleanest first
non-GPT-OSS family.

That was the correct first decision.

The next decision should optimize for **coverage expansion**, not for repeating
the same family with a different checkpoint.

`SmolLM2-360M-Instruct` does three things a second Qwen row does not:

1. it validates the Llama branch with a real named row
2. it uses an official small instruct checkpoint instead of a synthetic tiny
   fixture alone
3. it gives the repo one compact open Llama-family anchor for later small-model
   experiments

That is a better use of the next slot.

## Recommended Repo Framing

The repo should add **one named SmolLM2 Llama-family compatibility row**.

That row should be described as:

- first explicit small Llama-family pilot
- official GGUF-backed
- on-device health-check capable
- not a throughput claim
- not a full multi-backend claim
- not a broad model-family completion claim

This should look structurally similar to the Qwen pilot record, but it should
be honest about what is new:

- Qwen proved the generic non-GPT-OSS path
- SmolLM2 should prove the Llama branch with a real small instruct model

## Concrete Follow-On Sequence

The clean sequence is:

1. Add a `SmolLM2-360M-Instruct` pilot record and checker.
2. Keep the checker on the generic CPU server path first.
3. Optionally add `SmolLM2-135M` later as the ultra-fast smoke companion.
4. Defer `Qwen2.5-0.5B-Instruct` to an artifact-depth or throughput-comparison
   pass, not the next family-expansion slot.
5. Defer `Pythia` until the repo explicitly chooses to widen decoder-family
   support beyond the current GGUF set.

## Final Recommendation

The best logical next model type to add is:

- **`SmolLM2-360M-Instruct` as a named small Llama-family row**

The supporting rationale is direct:

- Psionic already executes the Llama family.
- Psionic already proved Qwen.
- Psionic has not yet named a real small Llama-family pilot.
- SmolLM2 is open, small, instruct-tuned, and officially available in GGUF.
- `360M` is large enough to matter and still small enough to stay cheap.

If the repo wants one sentence to guide the next implementation tranche, it
should be this:

> Add SmolLM2-360M-Instruct next, because it widens the generic server from the
> already-proved Qwen branch into a real small Llama-family row without
> requiring a new decoder substrate.
