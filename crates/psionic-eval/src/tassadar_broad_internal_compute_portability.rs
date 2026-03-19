use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use thiserror::Error;

use psionic_models::{
    TASSADAR_INTERNAL_COMPUTE_PROFILE_LADDER_REPORT_REF,
    TassadarInternalComputePortabilityPosture, TassadarInternalComputeProfileStatus,
    tassadar_internal_compute_profile_ladder_publication,
};
use psionic_runtime::{
    TASSADAR_BROAD_INTERNAL_COMPUTE_PORTABILITY_REPORT_REF,
    TassadarBroadInternalComputePortabilityReport, TassadarBroadInternalComputePortabilityRow,
    TassadarBroadInternalComputePortabilityRowStatus,
    TassadarBroadInternalComputeSuppressionReason,
};

use crate::{
    TASSADAR_ARTICLE_CPU_REPRODUCIBILITY_REPORT_REF,
    TassadarArticleCpuReproducibilityReportError, build_tassadar_article_cpu_reproducibility_report,
};

#[derive(Debug, Error)]
pub enum TassadarBroadInternalComputePortabilityReportError {
    #[error(transparent)]
    ArticleCpuReproducibility(#[from] TassadarArticleCpuReproducibilityReportError),
    #[error("failed to create `{path}`: {error}")]
    CreateDir { path: String, error: std::io::Error },
    #[error("failed to write `{path}`: {error}")]
    Write { path: String, error: std::io::Error },
    #[error("failed to read `{path}`: {error}")]
    Read { path: String, error: std::io::Error },
    #[error("failed to decode `{path}`: {error}")]
    Decode {
        path: String,
        error: serde_json::Error,
    },
}

pub fn build_tassadar_broad_internal_compute_portability_report(
) -> Result<
    TassadarBroadInternalComputePortabilityReport,
    TassadarBroadInternalComputePortabilityReportError,
> {
    let article_report = build_tassadar_article_cpu_reproducibility_report()?;
    let ladder = tassadar_internal_compute_profile_ladder_publication();
    let machine_class_ids = article_report
        .supported_machine_class_ids
        .iter()
        .chain(article_report.unsupported_machine_class_ids.iter())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let toolchain_family = format!(
        "{}:{}:{}",
        article_report.rust_toolchain_identity.compiler_family,
        article_report.rust_toolchain_identity.compiler_version,
        article_report.rust_toolchain_identity.target,
    );
    let backend_family = String::from("cpu_reference");
    let current_host_machine_class_id = article_report.matrix.current_host_machine_class_id.clone();

    let rows = ladder
        .profiles
        .iter()
        .flat_map(|profile| {
            machine_class_ids
                .iter()
                .cloned()
                .map(|machine_class_id| {
                    portability_row_for_profile(
                        profile,
                        &machine_class_id,
                        current_host_machine_class_id.as_str(),
                        backend_family.as_str(),
                        toolchain_family.as_str(),
                        article_report.matrix.current_host_measured_green,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    Ok(TassadarBroadInternalComputePortabilityReport::new(
        current_host_machine_class_id,
        vec![
            String::from(TASSADAR_ARTICLE_CPU_REPRODUCIBILITY_REPORT_REF),
            String::from(TASSADAR_INTERNAL_COMPUTE_PROFILE_LADDER_REPORT_REF),
        ],
        rows,
    ))
}

fn portability_row_for_profile(
    profile: &psionic_models::TassadarInternalComputeProfileSpec,
    machine_class_id: &str,
    current_host_machine_class_id: &str,
    backend_family: &str,
    toolchain_family: &str,
    current_host_measured_green: bool,
) -> TassadarBroadInternalComputePortabilityRow {
    let supported_machine_class =
        profile.supported_machine_class_ids.iter().any(|id| id == machine_class_id);
    let evidence_complete = required_evidence_refs_resolved(&profile.required_evidence_refs);
    let refusal_suite_complete = !profile.refusal_classes.is_empty();
    let fully_portable = profile.status == TassadarInternalComputeProfileStatus::Implemented
        && profile.portability_posture == TassadarInternalComputePortabilityPosture::DeclaredCpuMatrix
        && evidence_complete
        && refusal_suite_complete
        && supported_machine_class;

    let (row_status, publication_allowed, suppression_reason, note) = if fully_portable {
        if machine_class_id == current_host_machine_class_id && current_host_measured_green {
            (
                TassadarBroadInternalComputePortabilityRowStatus::PublishedMeasuredCurrentHost,
                true,
                None,
                format!(
                    "profile `{}` is measured green on the current host and stays inside the declared CPU portability envelope",
                    profile.profile_id
                ),
            )
        } else {
            (
                TassadarBroadInternalComputePortabilityRowStatus::PublishedDeclaredClass,
                true,
                None,
                format!(
                    "profile `{}` is admitted on declared machine class `{}` under the same explicit CPU portability envelope",
                    profile.profile_id, machine_class_id
                ),
            )
        }
    } else if !supported_machine_class {
        (
            TassadarBroadInternalComputePortabilityRowStatus::SuppressedDriftedOutsideEnvelope,
            false,
            Some(TassadarBroadInternalComputeSuppressionReason::OutsideDeclaredEnvelope),
            format!(
                "profile `{}` is suppressed on `{}` because that machine class drifts outside the declared envelope",
                profile.profile_id, machine_class_id
            ),
        )
    } else if profile.status != TassadarInternalComputeProfileStatus::Implemented {
        (
            TassadarBroadInternalComputePortabilityRowStatus::SuppressedPendingPortabilityEvidence,
            false,
            Some(TassadarBroadInternalComputeSuppressionReason::ProfileNotImplemented),
            format!(
                "profile `{}` remains suppressed because the named profile is still planned rather than implemented",
                profile.profile_id
            ),
        )
    } else {
        (
            TassadarBroadInternalComputePortabilityRowStatus::SuppressedPendingPortabilityEvidence,
            false,
            Some(TassadarBroadInternalComputeSuppressionReason::PortabilityEvidenceIncomplete),
            format!(
                "profile `{}` remains suppressed on `{}` because portability posture or required evidence is still incomplete",
                profile.profile_id, machine_class_id
            ),
        )
    };

    TassadarBroadInternalComputePortabilityRow {
        profile_id: profile.profile_id.clone(),
        backend_family: String::from(backend_family),
        toolchain_family: String::from(toolchain_family),
        machine_class_id: String::from(machine_class_id),
        row_status,
        publication_allowed,
        evidence_complete,
        refusal_suite_complete,
        suppression_reason,
        note,
    }
}

fn required_evidence_refs_resolved(required_evidence_refs: &[String]) -> bool {
    let repo_root = repo_root();
    required_evidence_refs.iter().all(|reference| {
        !reference.trim().is_empty()
            && !reference.starts_with("issue://")
            && repo_root.join(reference).exists()
    })
}

pub fn tassadar_broad_internal_compute_portability_report_path() -> PathBuf {
    repo_root().join(TASSADAR_BROAD_INTERNAL_COMPUTE_PORTABILITY_REPORT_REF)
}

pub fn write_tassadar_broad_internal_compute_portability_report(
    output_path: impl AsRef<Path>,
) -> Result<
    TassadarBroadInternalComputePortabilityReport,
    TassadarBroadInternalComputePortabilityReportError,
> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            TassadarBroadInternalComputePortabilityReportError::CreateDir {
                path: parent.display().to_string(),
                error,
            }
        })?;
    }
    let report = build_tassadar_broad_internal_compute_portability_report()?;
    let bytes =
        serde_json::to_vec_pretty(&report).expect("broad internal compute portability serializes");
    fs::write(output_path, bytes).map_err(|error| {
        TassadarBroadInternalComputePortabilityReportError::Write {
            path: output_path.display().to_string(),
            error,
        }
    })?;
    Ok(report)
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("psionic-eval should live under <repo>/crates/psionic-eval")
        .to_path_buf()
}

#[cfg(test)]
pub fn load_tassadar_broad_internal_compute_portability_report(
    path: impl AsRef<Path>,
) -> Result<
    TassadarBroadInternalComputePortabilityReport,
    TassadarBroadInternalComputePortabilityReportError,
> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|error| {
        TassadarBroadInternalComputePortabilityReportError::Read {
            path: path.display().to_string(),
            error,
        }
    })?;
    serde_json::from_slice(&bytes).map_err(|error| {
        TassadarBroadInternalComputePortabilityReportError::Decode {
            path: path.display().to_string(),
            error,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{
        build_tassadar_broad_internal_compute_portability_report,
        load_tassadar_broad_internal_compute_portability_report,
        tassadar_broad_internal_compute_portability_report_path,
    };
    use psionic_runtime::TassadarBroadInternalComputePortabilityRowStatus;

    #[test]
    fn broad_internal_compute_portability_report_publishes_article_and_suppresses_broader_profiles() {
        let report = build_tassadar_broad_internal_compute_portability_report().expect("report");
        assert!(report
            .publication_allowed_profile_ids
            .contains(&String::from(
                "tassadar.internal_compute.article_closeout.v1"
            )));
        assert!(report
            .suppressed_profile_ids
            .contains(&String::from(
                "tassadar.internal_compute.generalized_abi.v1"
            )));
        assert!(report.rows.iter().any(|row| {
            row.profile_id == "tassadar.internal_compute.article_closeout.v1"
                && row.row_status
                    == TassadarBroadInternalComputePortabilityRowStatus::PublishedMeasuredCurrentHost
        }));
    }

    #[test]
    fn broad_internal_compute_portability_report_matches_committed_truth()
    -> Result<(), Box<dyn std::error::Error>> {
        let expected = build_tassadar_broad_internal_compute_portability_report()?;
        let committed = load_tassadar_broad_internal_compute_portability_report(
            tassadar_broad_internal_compute_portability_report_path(),
        )?;
        assert_eq!(committed, expected);
        Ok(())
    }
}
