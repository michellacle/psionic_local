use serde::{Deserialize, Serialize};
use thiserror::Error;

use psionic_eval::{
    TASSADAR_BROAD_INTERNAL_COMPUTE_ACCEPTANCE_GATE_REPORT_REF,
    TassadarBroadInternalComputeAcceptanceStatus,
    build_tassadar_broad_internal_compute_acceptance_gate_report,
};
use psionic_runtime::TASSADAR_BROAD_INTERNAL_COMPUTE_PORTABILITY_REPORT_REF;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarBroadInternalComputePublicationDecisionStatus {
    Published,
    Suppressed,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarBroadInternalComputePublicationDecision {
    pub profile_id: String,
    pub portability_report_ref: String,
    pub acceptance_gate_report_ref: String,
    pub status: TassadarBroadInternalComputePublicationDecisionStatus,
    pub detail: String,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TassadarBroadInternalComputePublicationDecisionError {
    #[error("unknown broad internal-compute profile `{profile_id}`")]
    UnknownProfile { profile_id: String },
    #[error("profile `{profile_id}` is suppressed: {detail}")]
    Suppressed { profile_id: String, detail: String },
    #[error("profile `{profile_id}` failed the acceptance gate: {detail}")]
    Failed { profile_id: String, detail: String },
}

pub fn tassadar_broad_internal_compute_publication_decision(
    profile_id: &str,
) -> Result<
    TassadarBroadInternalComputePublicationDecision,
    TassadarBroadInternalComputePublicationDecisionError,
> {
    let report = build_tassadar_broad_internal_compute_acceptance_gate_report()
        .expect("broad internal-compute acceptance gate should build");
    let Some(row) = report.profile_rows.iter().find(|row| row.profile_id == profile_id) else {
        return Err(TassadarBroadInternalComputePublicationDecisionError::UnknownProfile {
            profile_id: String::from(profile_id),
        });
    };
    let status = match row.gate_status {
        TassadarBroadInternalComputeAcceptanceStatus::Green => {
            TassadarBroadInternalComputePublicationDecisionStatus::Published
        }
        TassadarBroadInternalComputeAcceptanceStatus::Suppressed => {
            TassadarBroadInternalComputePublicationDecisionStatus::Suppressed
        }
        TassadarBroadInternalComputeAcceptanceStatus::Failed => {
            TassadarBroadInternalComputePublicationDecisionStatus::Failed
        }
    };
    Ok(TassadarBroadInternalComputePublicationDecision {
        profile_id: String::from(profile_id),
        portability_report_ref: String::from(
            TASSADAR_BROAD_INTERNAL_COMPUTE_PORTABILITY_REPORT_REF,
        ),
        acceptance_gate_report_ref: String::from(
            TASSADAR_BROAD_INTERNAL_COMPUTE_ACCEPTANCE_GATE_REPORT_REF,
        ),
        status,
        detail: row.detail.clone(),
    })
}

pub fn require_tassadar_broad_internal_compute_profile_publication(
    profile_id: &str,
) -> Result<
    TassadarBroadInternalComputePublicationDecision,
    TassadarBroadInternalComputePublicationDecisionError,
> {
    let decision = tassadar_broad_internal_compute_publication_decision(profile_id)?;
    match decision.status {
        TassadarBroadInternalComputePublicationDecisionStatus::Published => Ok(decision),
        TassadarBroadInternalComputePublicationDecisionStatus::Suppressed => {
            Err(TassadarBroadInternalComputePublicationDecisionError::Suppressed {
                profile_id: decision.profile_id.clone(),
                detail: decision.detail.clone(),
            })
        }
        TassadarBroadInternalComputePublicationDecisionStatus::Failed => {
            Err(TassadarBroadInternalComputePublicationDecisionError::Failed {
                profile_id: decision.profile_id.clone(),
                detail: decision.detail.clone(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        TassadarBroadInternalComputePublicationDecisionError,
        TassadarBroadInternalComputePublicationDecisionStatus,
        require_tassadar_broad_internal_compute_profile_publication,
        tassadar_broad_internal_compute_publication_decision,
    };

    #[test]
    fn broad_internal_compute_publication_decision_publishes_article_profile_only() {
        let article = tassadar_broad_internal_compute_publication_decision(
            "tassadar.internal_compute.article_closeout.v1",
        )
        .expect("article profile");
        assert_eq!(
            article.status,
            TassadarBroadInternalComputePublicationDecisionStatus::Published
        );

        let generalized = tassadar_broad_internal_compute_publication_decision(
            "tassadar.internal_compute.generalized_abi.v1",
        )
        .expect("generalized abi profile");
        assert_eq!(
            generalized.status,
            TassadarBroadInternalComputePublicationDecisionStatus::Suppressed
        );
    }

    #[test]
    fn broad_internal_compute_publication_gate_refuses_failed_profiles() {
        let error = require_tassadar_broad_internal_compute_profile_publication(
            "tassadar.internal_compute.runtime_support_subset.v1",
        )
        .expect_err("runtime support subset should still fail");
        assert!(matches!(
            error,
            TassadarBroadInternalComputePublicationDecisionError::Failed { .. }
        ));
    }
}
