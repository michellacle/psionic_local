# Promoted PGOLF End-to-End Inference Proof

This audit closes `PINF-10` for the first promoted PGOLF-shaped Psion small-decoder family.

Scope:
- train one bounded repo-owned promoted bundle
- validate the promoted-bundle promotion receipt
- load the same bundle through the public runtime API
- run one local prompt-generation proof
- run one `psionic-serve` smoke test against the same bundle
- publish the honest backend matrix for this machine

Proof baseline:
- repo commit used for the proof commands below: `545780f0`
- bundle proof root: `/tmp/psionic_pgolf_inference_proof_604.X90nOo`
- operator entrypoint: [parameter_golf_promoted_operator.rs](/Users/christopherdavid/work/psionic/crates/psionic-serve/examples/parameter_golf_promoted_operator.rs)
- promotion receipt implementation: [parameter_golf_promoted_promotion.rs](/Users/christopherdavid/work/psionic/crates/psionic-train/src/parameter_golf_promoted_promotion.rs)
- public runtime bundle loader: [parameter_golf_promoted_bundle.rs](/Users/christopherdavid/work/psionic/crates/psionic-models/src/parameter_golf_promoted_bundle.rs)
- served CPU runtime for the promoted family: [lib.rs](/Users/christopherdavid/work/psionic/crates/psionic-serve/src/lib.rs#L5584)

## Backend Matrix

| Backend | Train promoted bundle | Runtime load | Local generation | Serve smoke | Benchmarked | Claim status |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | yes | yes | yes | yes | no | validated and claimed |
| CUDA | no proof run for this issue | not executed here | not executed here | not executed here | no | unclaimed |
| Metal / MLX | no proof run for this issue | not executed here | not executed here | not executed here | no | unclaimed |

Notes:
- the promoted served runtime landed for CPU only in this issue set; the checked proof path uses [CpuPromotedParameterGolfTextGenerationService](/Users/christopherdavid/work/psionic/crates/psionic-serve/src/lib.rs#L5584)
- this repo does have broader CUDA and Metal work, but none of that was used as proof for the promoted PGOLF trained-model inference claim here
- no throughput or latency benchmark claim is made in this audit

## Exact Commands

The following commands were executed from `/Users/christopherdavid/work/psionic`:

```bash
proof_root=$(mktemp -d /tmp/psionic_pgolf_inference_proof_604.XXXXXX)
inspect_json="$proof_root/inspect.json"
receipt_json="$proof_root/receipt.json"
prompt_json="$proof_root/prompt.json"
warm_json="$proof_root/warm.json"

cargo run -q -p psionic-train --example parameter_golf_promoted_reference_run -- "$proof_root"
cargo run -q -p psionic-serve --example parameter_golf_promoted_operator -- inspect "$proof_root" > "$inspect_json"
cargo run -q -p psionic-serve --example parameter_golf_promoted_operator -- validate "$proof_root" --assume general > "$receipt_json"
cargo run -q -p psionic-serve --example parameter_golf_promoted_operator -- prompt "$proof_root" --assume general --prompt abcd --max-new-tokens 4 > "$prompt_json"
cargo run -q -p psionic-serve --example parameter_golf_promoted_operator -- warm "$proof_root" --assume general --prompt abcd --max-new-tokens 2 > "$warm_json"
```

The concrete proof directory produced by that run was:

```text
/tmp/psionic_pgolf_inference_proof_604.X90nOo
```

The concrete JSON outputs captured by that run were:

```text
/tmp/psionic_pgolf_inference_proof_604.X90nOo/inspect.json
/tmp/psionic_pgolf_inference_proof_604.X90nOo/receipt.json
/tmp/psionic_pgolf_inference_proof_604.X90nOo/prompt.json
/tmp/psionic_pgolf_inference_proof_604.X90nOo/warm.json
```

## Observed Result

### 1. Promoted bundle emitted

The training example completed successfully and wrote a promoted general-profile bundle:

```text
profile=psion_small_decoder_pgolf_core_v0
checkpoint=parameter-golf-promoted-general-proof-run:step-00002
output=/tmp/psionic_pgolf_inference_proof_604.X90nOo
```

Core emitted bundle identity from `inspect.json`:

- `bundle_id`: `psion_small_decoder_pgolf_core_v0:parameter-golf-promoted-general-proof-run:step-00002`
- `model_id`: `parameter-golf-sp1024-9x512`
- `model_revision`: `public-2026-03-18`
- `profile_kind`: `general_psion_small_decoder`
- `manifest_valid`: `true`

Selected artifact observations from `inspect.json`:

- `descriptor.json`: `19,534` bytes, SHA matches manifest
- `model.safetensors`: `68,248,296` bytes, SHA matches manifest
- `tokenizer.json`: `95,769` bytes, SHA matches manifest
- `checkpoint_manifest.json`: `4,461,076` bytes, SHA matches manifest

### 2. Promotion receipt is green

Observed from `/tmp/psionic_pgolf_inference_proof_604.X90nOo/receipt.json`:

- `disposition`: `promoted`
- `receipt_digest`: `4cd15fbee5f4e01caca031cea43cde97411a1a257d9b7d0b06f96f74309de773`
- `failed_gate_kinds`: `[]`

Passed gates:

- `bundle_integrity`
- `runtime_loadability`
- `profile_assumption`
- `tokenizer_inference_readiness`
- `metadata_closure`
- `profile_specific_rules`
- `local_inference_smoke`

This is the current machine-readable admission boundary between “trained artifact exists” and “bundle is honestly inferable.”

### 3. Local prompt generation works

Observed from `/tmp/psionic_pgolf_inference_proof_604.X90nOo/prompt.json`:

- prompt: `abcd`
- mode: `greedy`
- prompt tokens: `4`
- output tokens: `4`
- termination: `MaxNewTokens`
- output text: `<reserved_0952><reserved_1005><reserved_0951><reserved_0900>`

This proves:
- the promoted bundle loads through the public runtime loader
- the emitted tokenizer/generation config can drive inference
- the restored trained model can emit deterministic local output on CPU

### 4. Serve-path smoke test works

Observed from `/tmp/psionic_pgolf_inference_proof_604.X90nOo/warm.json`:

- `model_id`: `parameter-golf-sp1024-9x512`
- `session_id`: `sess-00000001`
- prompt: `abcd`
- output tokens: `2`
- cache tokens: `6`
- termination: `MaxOutputTokens`
- output text: `<reserved_0952><reserved_1005>`

This proves:
- the same promoted bundle can be admitted by `psionic-serve`
- one session can be created against the promoted served descriptor
- one tiny warm/smoke generation succeeds through the serve path

## Claim Boundary

What this audit now honestly claims:

- Psionic can train one bounded repo-owned promoted PGOLF-shaped model on this machine
- Psionic can emit a promoted runtime bundle for that model
- Psionic can validate the bundle through a typed promotion receipt
- Psionic can load that bundle through the public runtime API
- Psionic can run local CPU prompt generation against that trained bundle
- Psionic can run one served CPU smoke test against that same bundle

What this audit explicitly does **not** claim:

- no CUDA inference proof for the promoted PGOLF family
- no Metal or MLX inference proof for the promoted PGOLF family
- no throughput or latency benchmark claim
- no production-quality language quality claim; this is a bounded proof model, not a product model
- no strict PGOLF challenge-overlay proof; the proof here is the general Psion small-decoder profile only

## Operator Guidance

Current honest operator workflow:

1. emit a promoted bundle with `parameter_golf_promoted_reference_run`
2. inspect it with `parameter_golf_promoted_operator -- inspect`
3. require a green promotion receipt with `parameter_golf_promoted_operator -- validate`
4. run local inference with `parameter_golf_promoted_operator -- prompt`
5. run serve smoke with `parameter_golf_promoted_operator -- warm`

If `validate` refuses, the bundle is not currently claimable as an inferable Psion model bundle, even if raw checkpoint files exist.
