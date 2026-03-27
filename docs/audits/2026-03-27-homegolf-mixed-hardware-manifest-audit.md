# HOMEGOLF Mixed Hardware Manifest Audit

`HOMEGOLF-6` closes the H100-admission gap at the contract level without pretending that a retained H100-backed run already happened.

## What Landed

- retained manifest report:
  `fixtures/parameter_golf/reports/parameter_golf_homegolf_mixed_hardware_manifest.json`
- generator:
  `crates/psionic-train/src/parameter_golf_homegolf_manifest.rs`
- entrypoint:
  `crates/psionic-train/src/bin/parameter_golf_homegolf_mixed_hardware_manifest.rs`
- checker:
  `scripts/check-parameter-golf-homegolf-mixed-hardware-manifest.sh`
- track doc:
  `docs/HOMEGOLF_TRACK.md`

## Retained Example

The retained mixed manifest now includes four declared members:

- `local-m5-primary`
  - class: `local_apple_silicon_metal`
  - presence: `present_measured`
  - backend: `mlx`
- `home-rtx4080-node`
  - class: `home_consumer_cuda_node`
  - presence: `present_measured`
  - backend: `cuda`
- `macbook-m2-peer`
  - class: `secondary_apple_silicon_peer`
  - presence: `optional_offline`
  - backend: `mlx`
- `future-h100-slot-01`
  - class: `optional_h100_node`
  - presence: `optional_future`
  - backend: `cuda`

## What This Proves

- HOMEGOLF already admits optional H100 nodes in the same track.
- Adding an H100 does not change the `600s` wallclock rule.
- Adding an H100 does not change the `16,000,000`-byte artifact accounting rule.
- Adding an H100 does not change the comparison law:
  - `public-baseline comparable`
  - `not public-leaderboard equivalent`

## Honest Boundary

This is a retained manifest surface, not a retained H100 execution proof.

The current truth is:

- the H100 slot is admitted
- the H100 slot is declared in one committed mixed-manifest example
- the score semantics stay frozen
- no retained H100-backed HOMEGOLF run is claimed yet

## Verification

The retained surface was rechecked with:

```bash
cargo run -q -p psionic-train --bin parameter_golf_homegolf_mixed_hardware_manifest -- \
  fixtures/parameter_golf/reports/parameter_golf_homegolf_mixed_hardware_manifest.json

cargo test -q -p psionic-train mixed_hardware_manifest -- --nocapture

./scripts/check-parameter-golf-homegolf-mixed-hardware-manifest.sh
./scripts/check-parameter-golf-homegolf-track-contract.sh
```
