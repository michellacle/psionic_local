# 2026-03-26 Psion Trained Model Inference Gap Audit

## Why This Audit Exists

The immediate question after getting the dummy Psion training run green is
simple:

- can Psionic now inference the model it just trained

That question needs an exact answer, because there are three different states
that can all be described sloppily as "inference works":

- the checkpoint can be restored inside training code
- the restored model can score token windows for eval or resumed training
- an operator can load the trained artifact, encode a text prompt, generate
  output tokens, decode them back to text, and serve the result through a
  stable interface

As of this audit, Psionic has the first two for the bounded Psion reference
pilot. It does not yet have the third.

## Bottom Line

The honest current answer is:

- yes, the reference pilot checkpoint can be restored and used internally
- no, Psionic does not yet have a real operator-facing inference path for the
  models its current Psion training lane produces

What works today is restore, validation, and resume:

- `crates/psionic-train/src/psion_reference_pilot.rs` can restore the saved
  checkpoint with `restore_psion_reference_pilot_checkpoint(...)`
- that restored model is used for post-train benchmark scoring inside the
  evidence bundle path
- the shipped resume probe can restore the checkpoint and apply one resumed
  optimizer step

What does not exist yet is the actual trained-model inference closure:

- no public load-and-generate API for the trained Psion checkpoint
- no prompt-to-token encoding path bound to the training artifacts
- no token-to-text decode path bound to the training artifacts
- no generation loop for the trained Psion checkpoint
- no train output bundle that contains everything a serving runtime needs
- no `psionic-serve` loader for the current Psion training outputs

So the repo can currently prove "we trained weights and can restore them", but
it cannot yet honestly claim "we can inference the models we train."

## What I Re-checked For This Audit

I re-ran the current bounded training and restore surfaces:

- `cargo check -q -p psionic-backend-cuda`
- `cargo check -q -p psionic-train`
- `cargo test -q -p psionic-train reference_pilot_runs_stage_and_observability_end_to_end -- --nocapture`
- `cargo test -q -p psionic-train reference_pilot_resume_probe_restores_from_last_stable_checkpoint -- --nocapture`
- `cargo run -q -p psionic-train --example psion_reference_pilot -- <tempdir>`
- `cargo run -q -p psionic-train --example psion_reference_pilot_resume_probe -- <run_dir> /tmp/psion_reference_pilot_resume_probe_20260326`

Results:

- the dummy training run succeeded and wrote a checkpoint plus receipts
- the resume probe succeeded against that checkpoint
- the resume probe reported:
  - `run_id = psion-reference-pilot-run`
  - `checkpoint_ref = psion-reference-pilot-step-16`
  - `recovery_mode = resume_from_last_stable_checkpoint`
  - `resumed_completed_steps = 1`
  - `has_restore_source = true`

This proves the checkpoint is live enough for internal restore and continued
training. It does not prove promptable text generation.

## Current Capability Boundary

### 1. What The Current Psion Pilot Actually Is

The current trained object is not a full served decoder runtime. It is a
bounded reference model implemented inside
`crates/psionic-train/src/psion_reference_pilot.rs`.

The trained model struct is:

- `PsionCompactDecoderReferencePilotModel`

That struct currently contains only:

- token embeddings
- position embeddings
- LM-head bias

Its next-token path is a private helper:

- `fn next_token_logits(&self, context_token_ids: &[u32]) -> Vec<f32>`

The function computes logits by:

- summing token and position embeddings across the bounded context window
- averaging that hidden vector
- projecting back into vocabulary space with tied token embeddings plus bias

That is enough to train and score next-token loss on the bounded pilot corpus.
It is not yet a full public inference family with a stable runtime surface.

### 2. What Has Been Proved

Psionic has proved these concrete statements:

- the reference pilot can execute optimizer steps and mutate weights
- the run emits a `.safetensors` checkpoint plus stage and observability
  receipts
- the saved checkpoint can be restored into the same internal pilot model type
- the restored checkpoint reproduces the final validation-loss state
- the saved optimizer state can resume training with explicit restore lineage

The repo proves those statements in:

- `crates/psionic-train/src/psion_reference_pilot.rs`
- `crates/psionic-train/examples/psion_reference_pilot.rs`
- `crates/psionic-train/examples/psion_reference_pilot_resume_probe.rs`

### 3. What Has Not Been Proved

The repo has not yet proved these statements for the current Psion training
lane:

- the trained checkpoint can be loaded from a stable public inference API
- the training outputs include everything required to encode prompts
- the training outputs include everything required to decode generated tokens
- the trained checkpoint can be sampled autoregressively from text prompts
- the trained checkpoint can be loaded into `psionic-serve`
- train-time restored logits match serve-time logits for the same prompt path

## What The Current Training Run Emits

The successful dummy run wrote these files:

- `psion_reference_pilot_checkpoint.safetensors`
- `psion_reference_pilot_checkpoint_manifest.json`
- `psion_reference_pilot_observability_receipt.json`
- `psion_reference_pilot_optimizer_state.json`
- `psion_reference_pilot_stage_config.json`
- `psion_reference_pilot_stage_receipt.json`
- `psion_reference_pilot_summary.json`

That is a training bundle, not an inference bundle.

Notably absent:

- no `descriptor.json`
- no tokenizer model file
- no tokenizer vocabulary export
- no template file
- no generation config
- no serving manifest
- no prompt encode/decode metadata beyond digests

The current summary file only reports run bookkeeping and losses. It does not
promote the run into an inferable artifact set.

## The Structural Gaps

### 1. The Restore Path Is Internal-Only

The checkpoint restore hook is:

- `pub(crate) fn restore_psion_reference_pilot_checkpoint(...)`

The model type is:

- `pub(crate) struct PsionCompactDecoderReferencePilotModel`

And the logits function is private:

- `fn next_token_logits(...)`

That means even though restore exists, it is not a public crate API for
external inference callers. Today the only first-class caller is
`psionic-train` itself.

### 2. There Is No Prompt Encoding Or Text Decode Surface Bound To The Trained Artifact

The reference pilot descriptor binds a tokenizer digest and special-token
digests through `PsionCompactDecoderTokenizerBinding`, but the current training
artifacts do not emit a tokenizer asset that can actually be loaded at runtime.

More importantly, the current tokenizer bundle in
`crates/psionic-data/src/psion_tokenizer_training.rs` is digest-and-inventory
oriented. The committed fixture
`fixtures/psion/tokenizer/psion_tokenizer_artifact_bundle_v1.json` contains:

- tokenizer identity
- digests
- artifact format label
- admitted and excluded source rows

It does not contain a real tokenizer model payload.

So the current train lane can say "this checkpoint is bound to tokenizer digest
X", but it cannot yet say "load tokenizer X and encode this prompt exactly as
training did."

### 3. The Runtime Tokenizer That Exists Today Is For Other Artifact Families

`crates/psionic-models/src/runtime_tokenizer.rs` implements runtime tokenizers
from GGUF metadata.

That is useful existing substrate, but it is not the same as a Psion-native
train artifact path. The current Psion reference pilot does not emit GGUF
tokenizer metadata, and its tokenizer bundle is not a loadable runtime object.

So there is existing tokenizer runtime logic in the repo, but the current
Psion-trained artifact does not plug into it.

### 4. The Current Train Artifact Does Not Match A Real Serve Bundle

The current pilot writes:

- one checkpoint manifest
- one optimizer-state artifact
- one checkpoint `.safetensors`

But it does not write the descriptor file named by the Psion compact-decoder
export contract, and it does not write a tokenizer asset.

There is also a stronger issue: the declared model-family contract and the
actual saved checkpoint are not yet the same thing.

`PsionCompactDecoderDescriptor` in
`crates/psionic-models/src/psion_compact_decoder.rs` declares a full decoder
tensor layout with per-layer attention and feed-forward tensors.

By contrast, `export_checkpoint(...)` in
`crates/psionic-train/src/psion_reference_pilot.rs` currently exports only:

- `decoder.embed_tokens.weight`
- `decoder.embed_positions.weight`
- `lm_head.bias`

So the current reference pilot checkpoint is a bounded pilot checkpoint, not a
full checkpoint for the declared compact-decoder family. That alone blocks a
truthful "train it here, serve it there" story.

### 5. There Is No Generation Loop For The Trained Psion Pilot

The pilot has loss and scoring code. It does not have a first-class inference
loop that:

- encodes a prompt
- runs iterative next-token prediction
- applies greedy or stochastic sampling
- stops on EOS or output budget
- decodes the generated tokens

There is no shipped example equivalent to:

- `cargo run -p psionic-train --example psion_reference_pilot_generate -- <bundle> "prompt"`

Without that, the system remains a train-and-score lane, not a trained-model
inference lane.

### 6. The Existing Serve Stack Does Not Load The Current Psion Training Outputs

`crates/psionic-serve/src/lib.rs` already has real generation interfaces:

- `generate(...)`
- `generate_stream(...)`

And the repo already has active generation backends in:

- `crates/psionic-serve/src/gpt_oss.rs`
- `crates/psionic-serve/src/gguf.rs`

So this is not a repo with zero inference substrate.

The actual gap is narrower and more important:

- the models we currently train in the Psion pilot lane are not emitted in a
  form the current serve stack knows how to load

### 7. The Portability Layer Exists, But The Psion Pilot Does Not Use It

`crates/psionic-train/src/model_io.rs` already defines:

- `PortableTokenizerBinding`
- `PortableModelBundle`
- `PortableModelBundleManifest`

That is the right kind of substrate for train-to-infer promotion.

But today it is still insufficient for the Psion pilot path because:

- the pilot does not emit a portable model bundle at train completion
- the portable tokenizer binding is still a binding contract, not necessarily a
  real tokenizer payload
- there is no serve-side loader wired to "load the bundle produced by the
  Psion pilot and start generating"

## Existing Substrate Worth Reusing

This should not be rebuilt from scratch. The repo already has pieces worth
reusing:

- `PsionCompactDecoderDescriptor` for family identity and explicit export
  naming
- `PortableModelBundle` and portable state-dict surfaces for train/export
  promotion
- `psionic-serve` generation request, response, and session management
  surfaces
- runtime tokenizer patterns in `psionic-models`
- existing generation backends that already solve request lifecycle, sampling,
  and serving mechanics for other model families

The missing work is mainly the train-to-infer closure for Psion-owned trained
artifacts.

## Two Honest Targets

There are two different projects that could both be described as "make Psion
models inferable."

### Target A: Smoke-Test Inference For The Current Toy Pilot

This is the short path.

Goal:

- prove that the current bounded reference pilot checkpoint can answer prompts
  through a minimal CLI

Required work:

- make restore public
- add a real tokenizer asset and loader for the current reference corpus
- add greedy sampling over `next_token_logits`
- add a small prompt CLI example

This would be enough to say:

- "the toy pilot model we just trained can be prompted"

It would not be enough to say:

- "Psionic has closed the general train-to-serve path for Psion models"

### Target B: Real Train-To-Infer Closure For Psion Models

This is the correct long path if the goal is to train models in Psionic and
then actually serve them honestly.

Goal:

- every promoted trained Psion artifact can be loaded by a stable inference
  runtime and served through the existing generation surface

That requires the broader roadmap below.

## Roadmap: Everything Needed To Make Trained Psion Models Inferable

### PINF-1: Freeze One Inferable Psion Artifact Bundle Contract

Define one promoted bundle produced by training that contains:

- `descriptor.json`
- `model.safetensors`
- tokenizer asset payload
- special-token facts
- template payload or canonical template reference
- training lineage and checkpoint provenance
- default generation config

Acceptance:

- one checker validates that the bundle is complete and self-consistent

### PINF-2: Make The Trained Artifact Match The Declared Model Family

Choose one honest route:

- either make training produce a full `PsionCompactDecoder` checkpoint matching
  the descriptor tensor layout
- or define a distinct `PsionReferencePilotDescriptor` family whose declared
  layout exactly matches the current toy pilot checkpoint

Acceptance:

- exported checkpoint tensor keys exactly match the declared descriptor layout

### PINF-3: Emit Real Tokenizer Assets Instead Of Digest-Only Records

Training-side tokenizer work must produce a loadable runtime artifact:

- SentencePiece model file, or
- explicit vocabulary plus merge/state payload, plus
- exact BOS, EOS, PAD, and unknown-token rules

Acceptance:

- the tokenizer emitted by training can encode and decode text without reaching
  back into fixture-only assumptions

### PINF-4: Add A Psion Runtime Tokenizer Loader

Implement a Psion-native runtime tokenizer path that can load the emitted
tokenizer asset directly from the promoted Psion bundle.

Acceptance:

- one runtime tokenizer can round-trip prompt text for the trained Psion family
- encode/decode tests cover special-token behavior and unknown-token handling

### PINF-5: Add A Public Load API For Trained Psion Artifacts

Expose a stable API that external crates can call to load the promoted bundle
and obtain an inferable runtime object.

Acceptance:

- loading does not require reaching into `pub(crate)` training internals

### PINF-6: Add An Actual Generation Loop

Implement the first Psion-owned generation loop with:

- prompt prefill
- iterative next-token decode
- EOS handling
- output budget handling
- deterministic seeded greedy or stochastic sampling

Acceptance:

- a single command can load a promoted bundle and print generated text

### PINF-7: Add Train-To-Infer Conformance Tests

The same promoted checkpoint must produce the same bounded logits and next-token
choices across:

- training-side restore path
- public inference API
- serve-side runtime

Acceptance:

- one fixed prompt suite compares logits or chosen next tokens across all three
  surfaces

### PINF-8: Emit Descriptor, Tokenizer, And Serve Metadata At Train Completion

The current training run must promote itself into an inferable directory rather
than leaving the operator to assemble one manually.

Acceptance:

- `run_psion_reference_pilot(...)` or its promoted successor writes a complete
  inferable bundle folder

### PINF-9: Bridge The Promoted Bundle Into `psionic-models`

`psionic-models` should own the stable runtime model representation for the
trained Psion family rather than keeping the executable model trapped inside
training code.

Acceptance:

- `psionic-models` can load the promoted Psion bundle and expose a runtime
  inference boundary

### PINF-10: Wire Psion Trained Bundles Into `psionic-serve`

Add a loader and service path that accepts the promoted Psion bundle and serves
it through the existing generation surface.

Acceptance:

- one `GenerationRequest` can target a Psion-trained model loaded from a bundle

### PINF-11: Add Minimal Operator Tooling

Ship the smallest practical operator entrypoints:

- local CLI prompt tool
- model warm/load tool
- served smoke-test command

Acceptance:

- an engineer can train, promote, load, prompt, and unload a Psion model
  without touching internal code

### PINF-12: Add Promotion Rules And Refusal Semantics

Not every checkpoint should become serveable automatically. The promotion layer
needs explicit gates for:

- artifact completeness
- tokenizer completeness
- descriptor-layout match
- benchmark floor
- safety or refusal calibration where applicable

Acceptance:

- train outputs move through a typed promotion step into inferable status

### PINF-13: Add Backend And Performance Validation

After the first CPU lane is green, validate actual inference backends for the
Psion family across the intended devices.

Acceptance:

- one backend matrix names what is real, what is benchmarked, and what still
  refuses claimability

## Recommended Sequencing

The fastest honest sequence is:

1. `PINF-1`: freeze the inferable bundle contract
2. `PINF-2`: resolve the model-family mismatch
3. `PINF-3` and `PINF-4`: make tokenizer assets real and loadable
4. `PINF-5` and `PINF-6`: public load plus generate
5. `PINF-7` and `PINF-8`: conformance plus train-time promotion
6. `PINF-9` and `PINF-10`: models and serve integration
7. `PINF-11` to `PINF-13`: operator and backend closure

If the immediate goal is just a smoke-test prompt path, then do:

1. `PINF-2`
2. `PINF-3`
3. `PINF-4`
4. `PINF-5`
5. `PINF-6`

But if the actual goal is "the models we train should be the models we can
serve," then the full sequence above is the right target.

## Final Verdict

Psionic is no longer blocked on training a bounded Psion reference model.
That part is real.

But the current system still stops at:

- train
- checkpoint
- restore
- eval
- resume

It does not yet close:

- prompt
- encode
- generate
- decode
- serve

So the correct statement today is:

- Psionic can restore and reuse the bounded Psion checkpoint internally
- Psionic cannot yet honestly inference the models it currently trains
- the gap is now concrete and implementable, not ambiguous
