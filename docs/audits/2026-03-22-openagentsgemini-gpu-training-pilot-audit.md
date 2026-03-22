# 2026-03-22 OpenAgentsGemini GPU Training Pilot Audit

This audit checks whether the current Google Cloud project can honestly launch
one bounded Psion GPU training pilot and retain enough infra, artifact, and
cost evidence to make the run useful.

I did not allocate a GPU VM in this audit. This is a control-plane and quota
readiness pass over the current `gcloud`-visible project state.

## Scope

- inspected local `gcloud` auth, active config, and project identity
- checked billing attachment and enabled services
- checked regional quota and zone accelerator catalogs in `us-central1`
- inventoried existing Compute Engine, network, storage, logging, and BigQuery
  surfaces
- checked project IAM for the active user
- did not verify the exact promotional-credit balance because the queried CLI
  surfaces did not expose it directly
- did not verify live GPU stock at allocation time because no launch was
  attempted

## Project Facts

- active account: `chris@openagents.com`
- active project: `openagentsgemini`
- active default region and zone: `us-central1` and `us-central1-a`
- active user project role: `roles/owner`
- billing is enabled on
  `billingAccounts/01D15C-64524A-1062EA`
- enabled services already include:
  `compute.googleapis.com`, `storage.googleapis.com`,
  `logging.googleapis.com`, `monitoring.googleapis.com`,
  `artifactregistry.googleapis.com`, `cloudbuild.googleapis.com`,
  `container.googleapis.com`, and `aiplatform.googleapis.com`

## Training-Relevant Capacity

Regional quota in `us-central1`:

- `CPUS`: limit `200`, usage `14`
- `INSTANCES`: limit `720`, usage `7`
- `SSD_TOTAL_GB`: limit `20480`, usage `3612`
- `NVIDIA_T4_GPUS`: limit `4`, usage `0`
- `PREEMPTIBLE_NVIDIA_T4_GPUS`: limit `4`, usage `0`
- `NVIDIA_L4_GPUS`: limit `8`, usage `0`
- `PREEMPTIBLE_NVIDIA_L4_GPUS`: limit `8`, usage `0`
- `NVIDIA_A100_GPUS`: limit `1`, usage `0`
- `PREEMPTIBLE_NVIDIA_A100_GPUS`: limit `16`, usage `0`
- `A2_CPUS`: limit `12`, usage `0`
- `NVIDIA_A100_80GB_GPUS`: limit `0`, usage `0`
- `PREEMPTIBLE_NVIDIA_A100_80GB_GPUS`: limit `0`, usage `0`

Zone accelerator catalog across `us-central1-a`, `us-central1-b`,
`us-central1-c`, and `us-central1-f`:

- `nvidia-tesla-t4` is visible in `a`, `b`, `c`, and `f`
- `nvidia-l4` is visible in `a`, `b`, and `c`
- `nvidia-tesla-a100` is visible in `a`, `b`, `c`, and `f`
- `nvidia-a100-80gb` is visible in `a` and `c`, but the project quota for
  `NVIDIA_A100_80GB_GPUS` is `0`
- higher-end catalog entries such as `nvidia-h100-80gb`, `nvidia-h200-141gb`,
  `nvidia-gb200`, and `nvidia-b200` are visible in some zones, but this audit
  did not find matching non-zero project quota for them, so they should be
  treated as unavailable for launch planning

Machine families seen in the same region:

- `g2-standard-4` and `g2-standard-8` in `a`, `b`, and `c`
- `a2-highgpu-1g` in `a`, `b`, `c`, and `f`

Practical reading:

- one standard A100 pilot is available on paper today
- at least one L4-backed G2 pilot is available on paper today
- T4 remains available for a cheaper smoke pass
- A100 80GB is not available today because quota is zero even though the zone
  catalog exposes the accelerator type

## Existing Infra

Current Compute Engine footprint:

- five running non-GPU VMs:
  `nexus-mainnet-1`, `nexus-staging-1`, `oa-bitcoind`, `oa-lnd`, and
  `symphony-mainnet-1`
- all current project hosts are in `us-central1-a`
- current instances are operational infrastructure, not training hosts

Current disk footprint:

- persistent disks already consume `3612 GB` of the regional SSD quota
- existing disks are attached to the current operational hosts, not a training
  lane

Current networks:

- `default`
- `oa-lightning`

Current subnets:

- `oa-lightning-us-central1` on `10.42.0.0/24`
- the standard auto-mode `default` network subnets across many regions

Current firewall posture:

- `default` still has the broad default SSH rule from `0.0.0.0/0`
- `oa-lightning` already has IAP SSH rules, but only for existing target tags
  such as `nexus-host`
- a new training VM on `oa-lightning` would need its own target tag and SSH
  rule, or the launch should use the `default` network instead

Current storage and recordkeeping:

- the only bucket listed is `gs://openagentsgemini_cloudbuild`
- there is no dedicated training artifact bucket
- there are no BigQuery datasets listed, so there is no obvious billing-export
  dataset or training-metrics dataset already in place
- logging only has the default `_Required` and `_Default` sinks

Current service-account posture:

- the project has the default Compute Engine service account plus a small set of
  app-specific service accounts
- there is no dedicated Psion training service account yet

Current governance hygiene:

- the project has no resource-manager tag bindings
- the `gcloud` config warning about a missing `environment` tag is accurate
- this is not a launch blocker, but it is a small hygiene gap

## Status Call

The GCP project is close, but not fully pilot-ready.

What is green:

- billing is enabled
- the active user has owner rights
- Compute Engine, storage, logging, monitoring, and Vertex APIs are already on
- `us-central1` quota is sufficient for a bounded single-GPU pilot
- the zone catalog shows realistic T4, L4, and A100 launch targets

What is still only `partial`:

- no dedicated training artifact bucket exists
- no dedicated training service account exists
- no billing-export dataset or budget surface was found from the queried CLI
  views
- no explicit training-host network and SSH posture exists yet
- no runbook exists yet for preserving the full infra evidence bundle around a
  GPU launch

Blunt conclusion:

- yes, this project can launch a bounded single-GPU Psion pilot
- no, it is not yet cleanly prepared to do that and preserve all relevant run
  evidence without a short setup pass first

## Recommended First GPU Pilot

Use a single L4-backed G2 launch first unless the run truly needs A100 memory
or throughput.

Why:

- L4 quota is wider than standard A100 quota
- G2 shapes are visible in multiple `us-central1` zones
- burning the only standard A100 slot is unnecessary for a first truthful pilot
- A100 80GB is not actually launchable today under the current quota

Recommended order:

1. first truthful pilot: one L4-backed G2 instance in `us-central1-a`,
   `us-central1-b`, or `us-central1-c`
2. second step if memory pressure requires it: one `a2-highgpu-1g`
3. leave Spot or preemptible usage for later once checkpoint persistence is
   proven, even though spot-like quota exists

## Minimum Setup Before Launch

- create a dedicated regional bucket for training artifacts, checkpoints, and
  receipts instead of reusing the Cloud Build bucket
- create a dedicated service account for the pilot instead of reusing the
  existing app runtime accounts
- choose the network posture explicitly:
  `default` for the fastest bring-up, or `oa-lightning` plus a new IAP SSH tag
  and rule for a cleaner private-network story
- create either a Cloud Billing budget or a billing export dataset so credits
  and cost drift are not invisible during the run
- define one launch wrapper that snapshots quota, instance metadata, machine
  details, and training outputs into the bucket

## Evidence Bundle The Pilot Should Save

At minimum, the first run should upload one bucket path containing:

- the instance description JSON
- the launch command or startup script
- `nvidia-smi`, disk, memory, and CPU snapshots
- the git revision and exact cargo or binary command used
- the Psion stage config, observability receipt, replay receipt, and checkpoint
  lineage
- stdout and stderr logs
- checkpoint manifests and checkpoint artifacts
- a final upload manifest listing every saved object

Without that bundle, the project may still run a GPU job, but it will not meet
the standard of a truthful training pilot with preserved evidence.

## Commands Used

The audit relied on these command families:

```bash
gcloud auth list
gcloud config list
gcloud projects list
gcloud beta billing projects describe openagentsgemini
gcloud beta billing accounts list
gcloud services list --enabled --project openagentsgemini
gcloud compute project-info describe --project openagentsgemini
gcloud compute regions describe us-central1 --project openagentsgemini
gcloud compute accelerator-types list --project openagentsgemini
gcloud compute machine-types list --project openagentsgemini
gcloud compute instances list --project openagentsgemini
gcloud compute disks list --project openagentsgemini
gcloud compute networks list --project openagentsgemini
gcloud compute networks subnets list --project openagentsgemini
gcloud compute firewall-rules list --project openagentsgemini
gcloud iam service-accounts list --project openagentsgemini
gcloud projects get-iam-policy openagentsgemini
gcloud storage buckets list --project openagentsgemini
bq ls --project_id=openagentsgemini
gcloud logging sinks list --project openagentsgemini
gcloud resource-manager tags bindings list --parent=//cloudresourcemanager.googleapis.com/projects/157437760789
```
