# OpenAgentsGemini First Google Accelerator-Backed Host-Native Plugin-Conditioned Run Audit

> Status: follow-up `PSION_ACCEL-8` audit written 2026-03-23 after the first
> truthful accelerator-backed Google host-native plugin-conditioned `Psion`
> run completed on `openagentsgemini` and the instance was deleted.

## Scope

This audit covers one bounded real Google single-node execution of the
accelerated host-native plugin-conditioned `Psion` lane.

It claims:

- one real Google-hosted host-native plugin-conditioned run now executes on
  `cuda`
- the run preserved launch truth, input-package truth, stage truth,
  accelerator-validation truth, plugin-evaluation truth, archive truth, and
  teardown truth
- the run preserved the same proved host-native authoring boundary and served
  posture instead of widening into mixed, guest-artifact, publication, or
  universality claims

It does not claim:

- mixed guest-artifact plugin-conditioned accelerator proof
- trusted-cluster readiness
- broad cost truth for this run via billing-export query data
- broad production-scale plugin-conditioned pretraining

## Typed Outcome

- result classification: `bounded_success`
- run id: `psion-g2-l4-plugin-host-native-accelerated-20260323t082304z`
- lane id: `psion_plugin_host_native_accelerated`
- repo revision: `2a5e7f35bc37aa508f004dfbeb48313c0c5fc1c2`

## Comparison To The Prior CPU-Bound Host-Native Audit

This run is the direct accelerator-backed follow-up to:

- `docs/audits/2026-03-22-openagentsgemini-first-google-host-native-plugin-conditioned-run-audit.md`

What stayed the same:

- same project, same bounded single-node Google operator lane
- same host-native proved authoring class:
  `host_native_capability_free_local_deterministic`
- same stable dataset identity:
  `dataset://openagents/psion/plugin_conditioned_host_native_reference@2026.03.22.v1`
- same bounded benchmark families and same explicit no-widening capability
  posture

What changed materially:

- the prior Google run used the metadata-oriented reference lane and retained
  `0%` GPU utilization with `0 MiB` GPU memory residency
- this run used the committed accelerated profile
  `g2_l4_single_node_plugin_host_native_accelerated`
- the launch manifest bound the accelerated training command explicitly:
  `"$CARGO_TARGET_DIR/debug/examples/psion_google_plugin_host_native_accelerated_run" "$PSION_OUTPUT_DIR"`
- the stage receipt recorded `delivered_execution.runtime_backend = cuda`
- the accelerator validation receipt recorded non-zero post-warmup GPU
  utilization, non-zero post-warmup GPU memory residency, and non-zero
  throughput
- the evaluation receipt now states that the learned rows are derived from the
  bounded CUDA-trained host-native lane rather than the metadata-only reference
  artifact

This is the first truthful Google plugin-conditioned audit that proves real
accelerator-backed host-native plugin-conditioned training rather than only a
GPU-hosted CPU reference lane.

## Topology

- project: `openagentsgemini`
- zone: `us-central1-a`
- machine type: `g2-standard-8`
- accelerator: `nvidia-l4`
- accelerator count: `1`
- expected execution backend: `cuda`
- observed GPU name: `NVIDIA L4`
- driver version: `570.211.01`
- boot disk: `pd-balanced`, `200 GB`
- external IP: `false`

## Timeline

UTC timestamps from the retained run timeline:

- launch created: `2026-03-23T08:23:14Z`
- bootstrap started: `2026-03-23T08:24:08Z`
- bootstrap finished: `2026-03-23T08:32:58Z`
- training started: `2026-03-23T08:32:58Z`
- training finished: `2026-03-23T08:35:43Z`
- checkpoint completed: `2026-03-23T08:36:08Z`
- teardown started: `2026-03-23T08:36:08Z`
- teardown finished: `2026-03-23T08:36:12Z`
- final manifest written: `2026-03-23T08:37:22Z`

The instance was deleted after finalization via:

- `bash scripts/psion-google-delete-single-node.sh --run-id psion-g2-l4-plugin-host-native-accelerated-20260323t082304z --force`

## Input And Launch Truth

The run used the committed accelerated plugin host-native descriptor:

- descriptor URI:
  `gs://openagentsgemini-psion-train-us-central1/manifests/psion_google_plugin_host_native_accelerated_input_package_v1.json`
- archive URI:
  `gs://openagentsgemini-psion-train-us-central1/runs/staging/input_packages/psion-google-plugin-host-native-accelerated-inputs-2a5e7f35bc37-20260323t082124z.tar.gz`
- archive SHA-256:
  `dded2dc46456468be64adae892f1223ff96cd68ec3de011483db22592d950133`
- manifest SHA-256:
  `564e4ad6d9f1a6e0f3199e3dd89e3f9a6d6059f35d98212c1d66b36a02273062`

The launch manifest also froze:

- profile id: `g2_l4_single_node_plugin_host_native_accelerated`
- trainer lane id: `psion_plugin_host_native_accelerated`
- expected execution backend: `cuda`
- declared run cost ceiling: `25 USD`
- image family: `common-cu128-ubuntu-2204-nvidia-570`
- image name: `common-cu128-ubuntu-2204-nvidia-570-v20260320`

The archive posture stayed explicit:

- archive command:
  `bash "$PSION_REPO_DIR/scripts/psion-google-archive-plugin-conditioned-run.sh" --stem psion_plugin_host_native_accelerated --manifest-out "$PSION_SCRATCH_DIR/psion_google_checkpoint_archive_manifest.json" "$PSION_OUTPUT_DIR"`
- restore command: `null`

That is still the correct boundary for this lane. It retains logical
checkpoint evidence and archived output receipts, not a separate dense
cold-restore proof.

## Accelerated Host-Native Lane Facts

Retained run summary:

- run id: `run-psion-plugin-host-native-accelerated`
- dataset ref:
  `dataset://openagents/psion/plugin_conditioned_host_native_reference`
- stable dataset identity:
  `dataset://openagents/psion/plugin_conditioned_host_native_reference@2026.03.22.v1`
- training example count: `3`
- optimizer steps completed: `8`
- learned plugin ids:
  - `plugin.feed.rss_atom_parse`
  - `plugin.html.extract_readable`
  - `plugin.text.url_extract`
- benchmark family count: `5`
- model artifact digest:
  `c935f241b11bc8be321e209e2dfec5d1c3986bf5cd50a291c5c6f7da6590ec38`
- evaluation receipt digest:
  `27e310fcc909b6bd3d2bbfb995ef0cf72878fd6e0e5b3d3709d97d5ef3d84b80`
- run bundle digest:
  `c8a1b9b801f768e05916c68c580201181636cd8730f080949d3b8b24a2a41f15`

The stage receipt makes the accelerated cutover explicit:

- `delivered_execution.runtime_backend = cuda`
- selected device class: `discrete_accelerator`
- selected device memory class: `dedicated_device`
- total device memory bytes: `24152899584`
- `accelerator_execution.accelerator_backed = true`
- mean step latency: `1500 ms`

The stage bundle still binds only the proved host-native traces and receipts:

- one bounded long-context general SFT trace
- three proved host-native tool-call traces
- held-out workflow case id:
  `starter_plugin.fetch_refusal.v1`

## Benchmark And Capability Boundary Truth

The evaluation receipt kept the same boundary as the earlier host-native audit:

- baseline label: `non_plugin_conditioned_baseline_v1`
- limited to proved authoring class: `true`
- proved authoring class label:
  `host_native_capability_free_local_deterministic`

Benchmark deltas:

- `discovery_selection`: baseline `6666`, trained `10000`, delta `3334`
- `argument_construction`: baseline `0`, trained `10000`, delta `10000`
- `sequencing_multi_call`: baseline `0`, trained `0`, delta `0`
- `refusal_request_structure`: baseline `6000`, trained `10000`, delta `4000`
- `result_interpretation`: baseline `0`, trained `10000`, delta `10000`

The important difference is provenance, not widened scope. Each benchmark
detail now records that the learned rows come from the bounded CUDA-trained
host-native lane rather than the earlier metadata-only reference artifact.

The run also remains inside the already-published host-native capability and
served-posture boundary frozen in:

- `docs/PSION_PLUGIN_HOST_NATIVE_CAPABILITY_MATRIX_V1.md`

That means this audit still does **not** widen:

- `host_native_networked_read_only`
- secret-backed or stateful plugin support
- guest-artifact plugin support
- plugin publication or marketplace claims
- public plugin universality
- arbitrary software-capability claims

## Accelerator Truth

This is the new proof surface that the prior host-native audit did not have.

Accelerator validation receipt:

- expected backend: `cuda`
- observed stage backend: `cuda`
- observed observability backend: `cuda`
- accelerator-backed pass: `true`
- optimizer steps completed: `8`
- mean step latency: `1500 ms`
- mean tokens per second: `20584`
- peak tokens per second: `20648`
- throughput wall clock: `12000 ms`

Retained GPU evidence:

- total GPU samples: `161`
- post-warmup samples considered: `160`
- non-zero post-warmup utilization samples: `22`
- non-zero post-warmup memory samples: `160`
- average GPU utilization after warmup: `2.47%`
- max GPU utilization after warmup: `100%`
- max GPU memory used: `1286 MiB`
- observed total GPU memory: `23034 MiB`

The correct reading is narrow but real:

- yes, this run used the accelerator materially enough to satisfy the committed
  backend and residency gates
- no, this is not yet a large or efficient production-scale plugin-conditioned
  training lane

## Archive Truth

This run retained checkpoint-linked archive truth for the accelerated
host-native lane:

- archive manifest URI:
  `gs://openagentsgemini-psion-train-us-central1/checkpoints/plugin_conditioned/psion-plugin-host-native-accelerated/run-psion-plugin-host-native-accelerated/checkpoint-psion-plugin-host-native-accelerated-1/archive/psion_google_plugin_conditioned_archive_manifest.json`
- archive manifest SHA-256:
  `f004b67fc77c51630df62328d504941e5e87ed20c3c88a5fe3c82ebb53e7dbb9`
- latest checkpoint ref:
  `checkpoint://psion/plugin_host_native_accelerated/1`
- latest checkpoint step: `1`

Archived objects include:

- accelerated run bundle
- accelerated stage bundle
- accelerated stage receipt
- accelerated observability receipt
- accelerated model artifact
- accelerated evaluation receipt
- logical checkpoint evidence manifest
- accelerated run summary

This is enough to preserve stage-lineage, accelerator-validation, and
checkpoint-lineage truth for the bounded accelerated host-native lane.

## Cost Truth

Cost truth for this specific run remains partial.

What is real:

- the launch profile ceiling was explicit at `25 USD`
- quota, budget topic, and price-profile preflight were all retained in the
  launch manifest

What is still missing:

- query-backed realized billing-export cost for the run itself is still not the
  canonical retained proof surface

So this audit does not claim billing-query-backed realized cost yet.

## Evidence Locations

Primary retained objects:

- final manifest:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-plugin-host-native-accelerated-20260323t082304z/final/psion_google_run_final_manifest.json`
- manifest of manifests:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-plugin-host-native-accelerated-20260323t082304z/final/psion_google_run_manifest_of_manifests.json`
- launch manifest:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-plugin-host-native-accelerated-20260323t082304z/launch/psion_google_single_node_launch_manifest.json`
- accelerator validation receipt:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-plugin-host-native-accelerated-20260323t082304z/receipts/psion_google_accelerator_validation_receipt.json`
- GPU summary:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-plugin-host-native-accelerated-20260323t082304z/host/psion_google_gpu_summary.json`
- run summary:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-plugin-host-native-accelerated-20260323t082304z/receipts/psion_plugin_host_native_accelerated_run_summary.json`
- evaluation receipt:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-plugin-host-native-accelerated-20260323t082304z/receipts/psion_plugin_host_native_accelerated_evaluation_receipt.json`
- stage receipt:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-plugin-host-native-accelerated-20260323t082304z/receipts/psion_plugin_host_native_accelerated_stage_receipt.json`
- archive manifest:
  `gs://openagentsgemini-psion-train-us-central1/checkpoints/plugin_conditioned/psion-plugin-host-native-accelerated/run-psion-plugin-host-native-accelerated/checkpoint-psion-plugin-host-native-accelerated-1/archive/psion_google_plugin_conditioned_archive_manifest.json`

## Conclusion

`PSION_ACCEL-8` is now closed in the narrow form it was supposed to prove.

The repo now has:

- one real Google-hosted accelerator-backed host-native plugin-conditioned run
- retained launch, host, input, stage, accelerator, eval, archive, and
  teardown evidence
- explicit proof that the host-native plugin-conditioned lane no longer has to
  hide behind a GPU-hosted CPU reference bundle

The repo still does not have:

- accelerator-backed mixed guest-artifact proof
- query-backed realized cost truth for the plugin-conditioned Google lane
- cluster-scale justification

That is the correct current boundary.
