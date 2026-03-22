use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use psionic_runtime::{
    StarterPluginAuthoringClass, StarterPluginCapabilityClass, StarterPluginInvocationStatus,
    StarterPluginOriginClass, starter_plugin_registration_by_tool_name,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    TassionAdmittedPluginRecord, TassionControllerContext, TassionControllerSurface,
    TassionPluginClass, TassionPluginInvocationRecord, TassionPluginInvocationStatus,
    TassionPluginOutcomeLabel, TassionPluginRouteLabel, TassionPluginTrainingRecord,
    TassadarMultiPluginTraceCorpusBundle, TassadarMultiPluginTraceCorpusError,
    TassadarMultiPluginTraceRecord, tassadar_multi_plugin_trace_corpus_bundle_path,
};

/// Stable schema version for the canonical plugin-trace derivation bundle.
pub const TASSION_PLUGIN_TRAINING_DERIVATION_BUNDLE_SCHEMA_VERSION: &str =
    "psionic.tassion.plugin_training_derivation_bundle.v1";
/// Canonical committed output ref for the derivation bundle.
pub const TASSION_PLUGIN_TRAINING_DERIVATION_BUNDLE_REF: &str =
    "fixtures/tassion/datasets/tassion_plugin_training_derivation_v1/tassion_plugin_training_derivation_bundle.json";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassionControllerSurfaceCountRow {
    /// Controller surface represented in the derived records.
    pub controller_surface: TassionControllerSurface,
    /// Number of source trace records normalized for the surface.
    pub source_record_count: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TassionPluginTrainingDerivationBundle {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable derivation bundle identifier.
    pub bundle_id: String,
    /// Stable source corpus ref.
    pub source_corpus_ref: String,
    /// Stable source corpus digest.
    pub source_corpus_digest: String,
    /// Per-surface source-record counts.
    pub controller_surface_counts: Vec<TassionControllerSurfaceCountRow>,
    /// Derived canonical plugin-training records.
    pub records: Vec<TassionPluginTrainingRecord>,
    /// Plain-language claim boundary for the derivation bundle.
    pub claim_boundary: String,
    /// Short machine-readable summary.
    pub summary: String,
    /// Stable digest over the bundle.
    pub bundle_digest: String,
}

impl TassionPluginTrainingDerivationBundle {
    /// Writes the derivation bundle to one JSON file.
    pub fn write_to_path(
        &self,
        output_path: impl AsRef<Path>,
    ) -> Result<(), TassionPluginTrainingDerivationError> {
        let output_path = output_path.as_ref();
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).map_err(|error| {
                TassionPluginTrainingDerivationError::CreateDir {
                    path: parent.display().to_string(),
                    error,
                }
            })?;
        }
        let json = serde_json::to_string_pretty(self)?;
        fs::write(output_path, format!("{json}\n")).map_err(|error| {
            TassionPluginTrainingDerivationError::Write {
                path: output_path.display().to_string(),
                error,
            }
        })
    }
}

#[derive(Debug, Error)]
pub enum TassionPluginTrainingDerivationError {
    #[error("failed to create `{path}`: {error}")]
    CreateDir { path: String, error: std::io::Error },
    #[error("failed to read `{path}`: {error}")]
    Read { path: String, error: std::io::Error },
    #[error("failed to write `{path}`: {error}")]
    Write { path: String, error: std::io::Error },
    #[error("unknown controller lane `{lane_id}` in trace record `{record_id}`")]
    UnknownControllerLane { lane_id: String, record_id: String },
    #[error("no starter-plugin registration exists for tool `{tool_name}`")]
    UnknownToolRegistration { tool_name: String },
    #[error(
        "trace corpus drift for tool `{tool_name}` field `{field}` expected `{expected}` but found `{actual}`"
    )]
    TraceSchemaDrift {
        tool_name: String,
        field: String,
        expected: String,
        actual: String,
    },
    #[error(transparent)]
    TraceCorpus(#[from] TassadarMultiPluginTraceCorpusError),
    #[error(transparent)]
    TrainingRecord(#[from] crate::TassionPluginTrainingRecordError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[must_use]
pub fn tassion_plugin_training_derivation_bundle_path() -> PathBuf {
    repo_root().join(TASSION_PLUGIN_TRAINING_DERIVATION_BUNDLE_REF)
}

pub fn build_tassion_plugin_training_derivation_bundle(
) -> Result<TassionPluginTrainingDerivationBundle, TassionPluginTrainingDerivationError> {
    let corpus = load_committed_tassadar_multi_plugin_trace_corpus_bundle()?;
    build_tassion_plugin_training_derivation_bundle_from_corpus(&corpus)
}

pub fn write_tassion_plugin_training_derivation_bundle(
    output_path: impl AsRef<Path>,
) -> Result<TassionPluginTrainingDerivationBundle, TassionPluginTrainingDerivationError> {
    let bundle = build_tassion_plugin_training_derivation_bundle()?;
    bundle.write_to_path(output_path)?;
    Ok(bundle)
}

pub fn build_tassion_plugin_training_derivation_bundle_from_corpus(
    corpus: &TassadarMultiPluginTraceCorpusBundle,
) -> Result<TassionPluginTrainingDerivationBundle, TassionPluginTrainingDerivationError> {
    let mut records = corpus
        .trace_records
        .iter()
        .map(derive_training_record)
        .collect::<Result<Vec<_>, _>>()?;
    records.sort_by(|left, right| left.record_id.cmp(&right.record_id));

    let mut count_by_surface = BTreeMap::new();
    for record in &records {
        *count_by_surface
            .entry(record.controller_context.controller_surface)
            .or_insert(0_u32) += 1;
    }
    let controller_surface_counts = count_by_surface
        .into_iter()
        .map(|(controller_surface, source_record_count)| TassionControllerSurfaceCountRow {
            controller_surface,
            source_record_count,
        })
        .collect::<Vec<_>>();

    let mut bundle = TassionPluginTrainingDerivationBundle {
        schema_version: String::from(TASSION_PLUGIN_TRAINING_DERIVATION_BUNDLE_SCHEMA_VERSION),
        bundle_id: String::from("tassion.plugin_training_derivation.bundle.v1"),
        source_corpus_ref: String::from(crate::TASSADAR_MULTI_PLUGIN_TRACE_CORPUS_BUNDLE_REF),
        source_corpus_digest: corpus.bundle_digest.clone(),
        controller_surface_counts,
        records,
        claim_boundary: String::from(
            "this derivation bundle normalizes deterministic workflow, router-owned plugin-loop, and local Apple FM plugin-session traces from the committed Tassadar multi-plugin trace corpus into one canonical plugin-training record schema. It preserves plugin receipt identity, plugin class, controller provenance, and route-or-outcome labels without inventing a second plugin API or implying trained-lane closure by itself.",
        ),
        summary: String::new(),
        bundle_digest: String::new(),
    };
    bundle.summary = format!(
        "plugin-training derivation bundle normalizes source_records={} into derived_records={} across controller_surfaces={}.",
        corpus.trace_records.len(),
        bundle.records.len(),
        bundle.controller_surface_counts.len(),
    );
    bundle.bundle_digest = stable_digest(
        b"psionic_tassion_plugin_training_derivation_bundle|",
        &bundle,
    );
    Ok(bundle)
}

fn derive_training_record(
    trace_record: &TassadarMultiPluginTraceRecord,
) -> Result<TassionPluginTrainingRecord, TassionPluginTrainingDerivationError> {
    let controller_surface = map_controller_surface(
        trace_record.lane_id.as_str(),
        trace_record.record_id.as_str(),
    )?;
    let admitted_plugins = trace_record
        .projected_tool_schema_rows
        .iter()
        .map(|row| {
            let registration = starter_plugin_registration_by_tool_name(row.tool_name.as_str())
                .ok_or_else(|| TassionPluginTrainingDerivationError::UnknownToolRegistration {
                    tool_name: row.tool_name.clone(),
                })?;
            check_match(
                row.tool_name.as_str(),
                "plugin_id",
                registration.plugin_id,
                row.plugin_id.as_str(),
            )?;
            check_match(
                row.tool_name.as_str(),
                "result_schema_id",
                registration.success_output_schema_id,
                row.result_schema_id.as_str(),
            )?;
            check_match(
                row.tool_name.as_str(),
                "replay_class_id",
                registration.replay_class_id,
                row.replay_class_id.as_str(),
            )?;
            let expected_refusal_schema_ids = registration
                .refusal_schema_ids
                .iter()
                .map(|schema_id| String::from(*schema_id))
                .collect::<Vec<_>>();
            if row.refusal_schema_ids != expected_refusal_schema_ids {
                return Err(TassionPluginTrainingDerivationError::TraceSchemaDrift {
                    tool_name: row.tool_name.clone(),
                    field: String::from("refusal_schema_ids"),
                    expected: format!("{expected_refusal_schema_ids:?}"),
                    actual: format!("{:?}", row.refusal_schema_ids),
                });
            }
            Ok(TassionAdmittedPluginRecord {
                plugin_id: row.plugin_id.clone(),
                tool_name: row.tool_name.clone(),
                plugin_class: map_plugin_class(registration.authoring_class),
                capability_class: capability_class_label(registration.capability_class),
                origin_class: origin_class_label(registration.origin_class),
                input_schema_id: String::from(registration.input_schema_id),
                result_schema_id: row.result_schema_id.clone(),
                refusal_schema_ids: row.refusal_schema_ids.clone(),
                replay_class_id: row.replay_class_id.clone(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let plugin_invocations = trace_record
        .step_rows
        .iter()
        .map(|step| {
            let (status, result_payload, refusal_schema_id) = match step.projected_result.status {
                StarterPluginInvocationStatus::Success => (
                    TassionPluginInvocationStatus::Success,
                    Some(step.projected_result.structured_payload.clone()),
                    None,
                ),
                StarterPluginInvocationStatus::Refusal => (
                    TassionPluginInvocationStatus::TypedRefusal,
                    None,
                    Some(step.projected_result.output_or_refusal_schema_id.clone()),
                ),
            };
            Ok(TassionPluginInvocationRecord {
                invocation_id: format!("{}.invoke.{}", trace_record.record_id, step.step_index),
                decision_ref: step.decision_ref.clone(),
                plugin_id: step.plugin_id.clone(),
                tool_name: step.tool_name.clone(),
                arguments: step.arguments.clone(),
                receipt_ref: step.projected_result.plugin_receipt.receipt_id.clone(),
                receipt_digest: step.projected_result.plugin_receipt.receipt_digest.clone(),
                status,
                result_payload,
                refusal_schema_id,
                detail: step.detail.clone(),
            })
        })
        .collect::<Result<Vec<_>, TassionPluginTrainingDerivationError>>()?;

    let route_label = if !plugin_invocations.is_empty() {
        TassionPluginRouteLabel::DelegateToAdmittedPlugin
    } else if trace_record.typed_refusal_preserved {
        TassionPluginRouteLabel::RefuseUnsupportedPluginOrCapability
    } else {
        TassionPluginRouteLabel::AnswerInLanguage
    };
    let outcome_label = if plugin_invocations
        .iter()
        .any(|invocation| invocation.status == TassionPluginInvocationStatus::TypedRefusal)
        || trace_record.typed_refusal_preserved
    {
        TassionPluginOutcomeLabel::TypedRuntimeRefusal
    } else {
        TassionPluginOutcomeLabel::CompletedSuccess
    };
    let final_response_text = if trace_record.final_message_text.trim().is_empty() {
        None
    } else {
        Some(trace_record.final_message_text.clone())
    };
    TassionPluginTrainingRecord::new(
        format!("tassion_training.{}", trace_record.record_id),
        trace_record.directive_text.clone(),
        admitted_plugins,
        TassionControllerContext {
            controller_surface,
            source_bundle_ref: trace_record.source_bundle_ref.clone(),
            source_bundle_id: trace_record.source_bundle_id.clone(),
            source_bundle_digest: trace_record.source_bundle_digest.clone(),
            source_case_id: trace_record.source_case_id.clone(),
            workflow_case_id: Some(trace_record.workflow_case_id.clone()),
            detail: trace_record.detail.clone(),
        },
        plugin_invocations,
        route_label,
        outcome_label,
        final_response_text,
        trace_record.detail.clone(),
    )
    .map_err(TassionPluginTrainingDerivationError::from)
}

fn map_controller_surface(
    lane_id: &str,
    record_id: &str,
) -> Result<TassionControllerSurface, TassionPluginTrainingDerivationError> {
    match lane_id {
        "deterministic_workflow" => Ok(TassionControllerSurface::DeterministicWorkflow),
        "router_responses" => Ok(TassionControllerSurface::RouterResponses),
        "apple_fm_session" => Ok(TassionControllerSurface::AppleFmSession),
        _ => Err(TassionPluginTrainingDerivationError::UnknownControllerLane {
            lane_id: String::from(lane_id),
            record_id: String::from(record_id),
        }),
    }
}

fn map_plugin_class(authoring_class: StarterPluginAuthoringClass) -> TassionPluginClass {
    match authoring_class {
        StarterPluginAuthoringClass::CapabilityFreeLocalDeterministic => {
            TassionPluginClass::HostNativeCapabilityFreeLocalDeterministic
        }
        StarterPluginAuthoringClass::NetworkedReadOnly => {
            TassionPluginClass::HostNativeNetworkedReadOnly
        }
    }
}

fn capability_class_label(capability_class: StarterPluginCapabilityClass) -> String {
    match capability_class {
        StarterPluginCapabilityClass::LocalDeterministic => String::from("local_deterministic"),
        StarterPluginCapabilityClass::ReadOnlyNetwork => String::from("read_only_network"),
    }
}

fn origin_class_label(origin_class: StarterPluginOriginClass) -> String {
    match origin_class {
        StarterPluginOriginClass::OperatorBuiltin => String::from("operator_builtin"),
        StarterPluginOriginClass::UserAdded => String::from("user_added"),
    }
}

fn check_match(
    tool_name: &str,
    field: &str,
    expected: &str,
    actual: &str,
) -> Result<(), TassionPluginTrainingDerivationError> {
    if expected != actual {
        return Err(TassionPluginTrainingDerivationError::TraceSchemaDrift {
            tool_name: String::from(tool_name),
            field: String::from(field),
            expected: String::from(expected),
            actual: String::from(actual),
        });
    }
    Ok(())
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .expect("repo root should resolve from psionic-data crate dir")
}

fn load_committed_tassadar_multi_plugin_trace_corpus_bundle(
) -> Result<TassadarMultiPluginTraceCorpusBundle, TassionPluginTrainingDerivationError> {
    let path = tassadar_multi_plugin_trace_corpus_bundle_path();
    let bytes = fs::read(&path).map_err(|error| TassionPluginTrainingDerivationError::Read {
        path: path.display().to_string(),
        error,
    })?;
    serde_json::from_slice(&bytes).map_err(TassionPluginTrainingDerivationError::from)
}

fn stable_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let encoded = serde_json::to_vec(value).expect("derivation bundle should serialize");
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(encoded);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::{
        TASSION_PLUGIN_TRAINING_DERIVATION_BUNDLE_SCHEMA_VERSION,
        build_tassion_plugin_training_derivation_bundle,
        build_tassion_plugin_training_derivation_bundle_from_corpus,
        load_committed_tassadar_multi_plugin_trace_corpus_bundle,
    };

    #[test]
    fn derivation_bundle_builds_from_committed_corpus() -> Result<(), Box<dyn std::error::Error>> {
        let bundle = build_tassion_plugin_training_derivation_bundle()?;
        assert_eq!(
            bundle.schema_version,
            TASSION_PLUGIN_TRAINING_DERIVATION_BUNDLE_SCHEMA_VERSION
        );
        assert_eq!(bundle.records.len(), 6);
        assert_eq!(bundle.controller_surface_counts.len(), 3);
        assert!(
            bundle
                .records
                .iter()
                .all(|record| !record.record_digest.is_empty())
        );
        Ok(())
    }

    #[test]
    fn derivation_rejects_schema_drift() -> Result<(), Box<dyn std::error::Error>> {
        let mut corpus = load_committed_tassadar_multi_plugin_trace_corpus_bundle()?;
        corpus.trace_records[0].projected_tool_schema_rows[0].result_schema_id =
            String::from("drifted.result.schema.v1");
        let error = build_tassion_plugin_training_derivation_bundle_from_corpus(&corpus)
            .expect_err("schema drift should fail closed");
        assert!(error.to_string().contains("trace corpus drift for tool"));
        Ok(())
    }
}
