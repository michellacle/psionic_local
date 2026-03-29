use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use psionic_eval::{
    CompiledAgentModuleKind, CompiledAgentPublicOutcomeKind, CompiledAgentRoute,
    CompiledAgentRuntimeState, CompiledAgentToolCall, CompiledAgentToolResult,
    compiled_agent_baseline_revision_set,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::repo_relative_path;

pub const COMPILED_AGENT_SOURCE_FIXTURE_DIR: &str = "fixtures/compiled_agent/source";
pub const COMPILED_AGENT_LEARNING_RECEIPT_LEDGER_FIXTURE_PATH: &str =
    "fixtures/compiled_agent/compiled_agent_learning_receipts_v1.json";
pub const COMPILED_AGENT_REPLAY_BUNDLE_FIXTURE_PATH: &str =
    "fixtures/compiled_agent/compiled_agent_replay_bundle_v1.json";

const LEARNING_RECEIPT_SCHEMA_VERSION: &str = "psionic.compiled_agent.learning_receipts.v1";
const REPLAY_BUNDLE_SCHEMA_VERSION: &str = "psionic.compiled_agent.replay_bundle.v1";

#[derive(Debug, Error)]
pub enum CompiledAgentReceiptError {
    #[error("failed to create `{path}`: {error}")]
    CreateDir { path: String, error: std::io::Error },
    #[error("failed to read `{path}`: {error}")]
    Read { path: String, error: std::io::Error },
    #[error("failed to write `{path}`: {error}")]
    Write { path: String, error: std::io::Error },
    #[error("source fixture `{fixture}` has no canonical supervision label")]
    MissingLabel { fixture: String },
    #[error("source receipt `{fixture}` is missing phase `{phase}`")]
    MissingPhase { fixture: String, phase: String },
    #[error("fixture `{path}` drifted from the canonical generator output")]
    FixtureDrift { path: String },
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledAgentSourcePublicResponse {
    pub kind: CompiledAgentPublicOutcomeKind,
    pub response: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledAgentSourceManifest {
    pub module_name: String,
    pub signature_name: String,
    pub implementation_family: String,
    pub implementation_label: String,
    pub version: String,
    pub promotion_state: String,
    pub confidence_floor: f32,
}

impl CompiledAgentSourceManifest {
    #[must_use]
    pub fn manifest_id(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.module_name, self.implementation_family, self.implementation_label, self.version
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledAgentSourcePhaseTraceEntry {
    pub phase: String,
    pub manifest: CompiledAgentSourceManifest,
    pub authority: String,
    pub candidate_label: Option<String>,
    pub input: Value,
    pub output: Value,
    pub confidence: f32,
    pub trace: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledAgentSourceInternalTrace {
    pub primary_phases: Vec<CompiledAgentSourcePhaseTraceEntry>,
    pub shadow_phases: Vec<CompiledAgentSourcePhaseTraceEntry>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledAgentSourceLineage {
    pub user_request: String,
    pub route: CompiledAgentRoute,
    pub tool_calls: Vec<CompiledAgentToolCall>,
    pub tool_results: Vec<CompiledAgentToolResult>,
    pub public_response: CompiledAgentSourcePublicResponse,
    pub authority_manifest_ids: Vec<String>,
    pub shadow_manifest_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledAgentSourceRun {
    pub public_response: CompiledAgentSourcePublicResponse,
    pub internal_trace: CompiledAgentSourceInternalTrace,
    pub lineage: CompiledAgentSourceLineage,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledAgentSourceReceipt {
    pub schema_version: u32,
    pub captured_at_epoch_ms: u64,
    pub state: CompiledAgentRuntimeState,
    pub run: CompiledAgentSourceRun,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompiledAgentLearningPublicResponse {
    pub kind: CompiledAgentPublicOutcomeKind,
    pub response: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompiledAgentLearningAssessment {
    pub route_correct: bool,
    pub tool_policy_correct: bool,
    pub tool_arguments_correct: bool,
    pub grounded_answer_correct: bool,
    pub verify_correct: bool,
    pub overall_success: bool,
    pub failure_classes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledAgentLearningReceipt {
    pub receipt_id: String,
    pub source_fixture_ref: String,
    pub source_receipt_digest: String,
    pub captured_at_epoch_ms: u64,
    pub user_request: String,
    pub runtime_state: CompiledAgentRuntimeState,
    pub observed_route: CompiledAgentRoute,
    pub expected_route: CompiledAgentRoute,
    pub observed_tool_names: Vec<String>,
    pub expected_tool_names: Vec<String>,
    pub observed_tool_calls: Vec<CompiledAgentToolCall>,
    pub observed_tool_results: Vec<CompiledAgentToolResult>,
    pub observed_public_response: CompiledAgentLearningPublicResponse,
    pub expected_public_response: CompiledAgentLearningPublicResponse,
    pub authority_manifest_ids: Vec<String>,
    pub shadow_manifest_ids: Vec<String>,
    pub primary_phase_confidences: BTreeMap<String, f32>,
    pub assessment: CompiledAgentLearningAssessment,
    pub tags: Vec<String>,
    pub operator_note: String,
    pub receipt_digest: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledAgentLearningReceiptLedger {
    pub schema_version: String,
    pub ledger_id: String,
    pub row_id: String,
    pub baseline_revision_id: String,
    pub source_fixture_refs: Vec<String>,
    pub correction_receipt_ids: Vec<String>,
    pub module_failure_counts: BTreeMap<String, u32>,
    pub receipts: Vec<CompiledAgentLearningReceipt>,
    pub summary: String,
    pub ledger_digest: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompiledAgentReplayCorrectionKind {
    BehavioralClone,
    FailureCorrection,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledAgentReplaySample {
    pub sample_id: String,
    pub module: CompiledAgentModuleKind,
    pub source_receipt_id: String,
    pub correction_kind: CompiledAgentReplayCorrectionKind,
    pub tags: Vec<String>,
    pub failure_classes: Vec<String>,
    pub input: Value,
    pub expected_output: Value,
    pub observed_output: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledAgentReplayBundle {
    pub schema_version: String,
    pub bundle_id: String,
    pub row_id: String,
    pub baseline_revision_id: String,
    pub source_ledger_digest: String,
    pub module_sample_counts: BTreeMap<String, u32>,
    pub correction_sample_count: u32,
    pub samples: Vec<CompiledAgentReplaySample>,
    pub summary: String,
    pub bundle_digest: String,
}

struct CanonicalSupervisionLabel {
    fixture_name: &'static str,
    expected_route: CompiledAgentRoute,
    tags: &'static [&'static str],
    operator_note: &'static str,
}

const CANONICAL_SUPERVISION_LABELS: &[CanonicalSupervisionLabel] = &[
    CanonicalSupervisionLabel {
        fixture_name: "openagents_provider_ready_receipt_v1.json",
        expected_route: CompiledAgentRoute::ProviderStatus,
        tags: &["supported", "provider"],
        operator_note: "Provider-ready source receipt is kept as a clean route and grounding success row.",
    },
    CanonicalSupervisionLabel {
        fixture_name: "openagents_wallet_receipt_v1.json",
        expected_route: CompiledAgentRoute::WalletStatus,
        tags: &["supported", "wallet", "recent_earnings_target"],
        operator_note: "Wallet source receipt preserves the narrow answer plus recent earnings so grounded-answer improvements can be measured explicitly.",
    },
    CanonicalSupervisionLabel {
        fixture_name: "openagents_unsupported_receipt_v1.json",
        expected_route: CompiledAgentRoute::Unsupported,
        tags: &["unsupported"],
        operator_note: "Unsupported source receipt stays as the clean refusal baseline.",
    },
    CanonicalSupervisionLabel {
        fixture_name: "openagents_negated_wallet_receipt_v1.json",
        expected_route: CompiledAgentRoute::Unsupported,
        tags: &["unsupported", "negated", "correction_required"],
        operator_note: "Negated wallet mention is retained as the first correction seed: the source receipt exposed a wallet route, but the expected behavior is a clean unsupported refusal.",
    },
];

#[must_use]
pub fn compiled_agent_source_fixture_dir() -> PathBuf {
    repo_relative_path(COMPILED_AGENT_SOURCE_FIXTURE_DIR)
}

#[must_use]
pub fn compiled_agent_learning_receipt_ledger_fixture_path() -> PathBuf {
    repo_relative_path(COMPILED_AGENT_LEARNING_RECEIPT_LEDGER_FIXTURE_PATH)
}

#[must_use]
pub fn compiled_agent_replay_bundle_fixture_path() -> PathBuf {
    repo_relative_path(COMPILED_AGENT_REPLAY_BUNDLE_FIXTURE_PATH)
}

pub fn load_compiled_agent_source_receipt(
    path: impl AsRef<Path>,
) -> Result<CompiledAgentSourceReceipt, CompiledAgentReceiptError> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|error| CompiledAgentReceiptError::Read {
        path: path.display().to_string(),
        error,
    })?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub fn canonical_compiled_agent_learning_receipt_ledger(
) -> Result<CompiledAgentLearningReceiptLedger, CompiledAgentReceiptError> {
    let baseline = compiled_agent_baseline_revision_set();
    let mut receipts = Vec::new();
    for label in CANONICAL_SUPERVISION_LABELS {
        let source_fixture_ref = format!("{COMPILED_AGENT_SOURCE_FIXTURE_DIR}/{}", label.fixture_name);
        let source_receipt = load_compiled_agent_source_receipt(repo_relative_path(&source_fixture_ref))?;
        receipts.push(build_learning_receipt(
            &source_fixture_ref,
            &source_receipt,
            label,
            &baseline.unsupported_template,
        )?);
    }
    Ok(build_learning_receipt_ledger(receipts, &baseline.revision_id))
}

pub fn canonical_compiled_agent_replay_bundle(
) -> Result<CompiledAgentReplayBundle, CompiledAgentReceiptError> {
    let ledger = canonical_compiled_agent_learning_receipt_ledger()?;
    Ok(build_compiled_agent_replay_bundle(&ledger))
}

pub fn write_compiled_agent_learning_receipt_ledger(
    output_path: impl AsRef<Path>,
) -> Result<CompiledAgentLearningReceiptLedger, CompiledAgentReceiptError> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| CompiledAgentReceiptError::CreateDir {
            path: parent.display().to_string(),
            error,
        })?;
    }
    let ledger = canonical_compiled_agent_learning_receipt_ledger()?;
    let json = serde_json::to_string_pretty(&ledger)?;
    fs::write(output_path, format!("{json}\n")).map_err(|error| {
        CompiledAgentReceiptError::Write {
            path: output_path.display().to_string(),
            error,
        }
    })?;
    Ok(ledger)
}

pub fn write_compiled_agent_replay_bundle(
    output_path: impl AsRef<Path>,
) -> Result<CompiledAgentReplayBundle, CompiledAgentReceiptError> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| CompiledAgentReceiptError::CreateDir {
            path: parent.display().to_string(),
            error,
        })?;
    }
    let bundle = canonical_compiled_agent_replay_bundle()?;
    let json = serde_json::to_string_pretty(&bundle)?;
    fs::write(output_path, format!("{json}\n")).map_err(|error| {
        CompiledAgentReceiptError::Write {
            path: output_path.display().to_string(),
            error,
        }
    })?;
    Ok(bundle)
}

pub fn verify_compiled_agent_learning_receipt_fixtures() -> Result<(), CompiledAgentReceiptError> {
    let expected_ledger = canonical_compiled_agent_learning_receipt_ledger()?;
    let expected_bundle = build_compiled_agent_replay_bundle(&expected_ledger);

    let committed_ledger_bytes = fs::read(compiled_agent_learning_receipt_ledger_fixture_path())
        .map_err(|error| CompiledAgentReceiptError::Read {
            path: compiled_agent_learning_receipt_ledger_fixture_path()
                .display()
                .to_string(),
            error,
        })?;
    let committed_ledger: CompiledAgentLearningReceiptLedger =
        serde_json::from_slice(&committed_ledger_bytes)?;
    if committed_ledger != expected_ledger {
        return Err(CompiledAgentReceiptError::FixtureDrift {
            path: compiled_agent_learning_receipt_ledger_fixture_path()
                .display()
                .to_string(),
        });
    }

    let committed_bundle_bytes = fs::read(compiled_agent_replay_bundle_fixture_path()).map_err(
        |error| CompiledAgentReceiptError::Read {
            path: compiled_agent_replay_bundle_fixture_path()
                .display()
                .to_string(),
            error,
        },
    )?;
    let committed_bundle: CompiledAgentReplayBundle =
        serde_json::from_slice(&committed_bundle_bytes)?;
    if committed_bundle != expected_bundle {
        return Err(CompiledAgentReceiptError::FixtureDrift {
            path: compiled_agent_replay_bundle_fixture_path()
                .display()
                .to_string(),
        });
    }

    Ok(())
}

fn build_learning_receipt(
    source_fixture_ref: &str,
    source_receipt: &CompiledAgentSourceReceipt,
    label: &CanonicalSupervisionLabel,
    unsupported_template: &str,
) -> Result<CompiledAgentLearningReceipt, CompiledAgentReceiptError> {
    let observed_route = source_receipt.run.lineage.route;
    let expected_route = label.expected_route;
    let expected_response = expected_public_response(source_fixture_ref, label, unsupported_template);
    let observed_response = CompiledAgentLearningPublicResponse {
        kind: source_receipt.run.public_response.kind,
        response: source_receipt.run.public_response.response.clone(),
    };
    let observed_tool_names = source_receipt
        .run
        .lineage
        .tool_calls
        .iter()
        .map(|call| call.tool_name.clone())
        .collect::<Vec<_>>();
    let expected_tool_names = expected_tool_names(expected_route);
    let route_correct = observed_route == expected_route;
    let tool_policy_correct = observed_tool_names == expected_tool_names;
    let tool_arguments_correct = tool_arguments_match(
        source_receipt.run.lineage.tool_calls.as_slice(),
        expected_tool_names.as_slice(),
    );
    let grounded_answer_correct = observed_response == expected_response;
    let verify_correct = verify_matches_expected(
        expected_route,
        observed_tool_names.as_slice(),
        observed_response.kind,
        &expected_response,
        &observed_response,
    );
    let mut failure_classes = Vec::new();
    if !route_correct {
        failure_classes.push(route_failure_class(&source_receipt.run.lineage.user_request, observed_route, expected_route));
    }
    if !tool_policy_correct {
        failure_classes.push(String::from("unexpected_tool_exposure"));
    }
    if !tool_arguments_correct {
        failure_classes.push(String::from("tool_argument_mismatch"));
    }
    if !grounded_answer_correct {
        failure_classes.push(String::from("grounded_answer_mismatch"));
    }
    if !verify_correct {
        failure_classes.push(String::from("unsafe_final_outcome"));
    }
    failure_classes.sort();
    failure_classes.dedup();
    let primary_phase_confidences = source_receipt
        .run
        .internal_trace
        .primary_phases
        .iter()
        .map(|phase| (phase.phase.clone(), phase.confidence))
        .collect::<BTreeMap<_, _>>();
    let receipt_id = format!(
        "receipt.compiled_agent.learning.{}",
        fixture_slug(source_fixture_ref)
    );
    let source_receipt_digest = stable_digest(b"compiled_agent_source_receipt|", source_receipt);
    let mut receipt = CompiledAgentLearningReceipt {
        receipt_id,
        source_fixture_ref: source_fixture_ref.to_string(),
        source_receipt_digest,
        captured_at_epoch_ms: source_receipt.captured_at_epoch_ms,
        user_request: source_receipt.run.lineage.user_request.clone(),
        runtime_state: source_receipt.state.clone(),
        observed_route,
        expected_route,
        observed_tool_names,
        expected_tool_names,
        observed_tool_calls: source_receipt.run.lineage.tool_calls.clone(),
        observed_tool_results: source_receipt.run.lineage.tool_results.clone(),
        observed_public_response: observed_response,
        expected_public_response: expected_response,
        authority_manifest_ids: source_receipt.run.lineage.authority_manifest_ids.clone(),
        shadow_manifest_ids: source_receipt.run.lineage.shadow_manifest_ids.clone(),
        primary_phase_confidences,
        assessment: CompiledAgentLearningAssessment {
            route_correct,
            tool_policy_correct,
            tool_arguments_correct,
            grounded_answer_correct,
            verify_correct,
            overall_success: route_correct
                && tool_policy_correct
                && tool_arguments_correct
                && grounded_answer_correct
                && verify_correct,
            failure_classes,
        },
        tags: label.tags.iter().map(|tag| (*tag).to_string()).collect(),
        operator_note: label.operator_note.to_string(),
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = stable_digest(b"compiled_agent_learning_receipt|", &receipt);
    Ok(receipt)
}

fn build_learning_receipt_ledger(
    receipts: Vec<CompiledAgentLearningReceipt>,
    baseline_revision_id: &str,
) -> CompiledAgentLearningReceiptLedger {
    let mut module_failure_counts = BTreeMap::new();
    for receipt in &receipts {
        if !receipt.assessment.route_correct {
            *module_failure_counts
                .entry(String::from("route"))
                .or_insert(0) += 1;
        }
        if !receipt.assessment.tool_policy_correct {
            *module_failure_counts
                .entry(String::from("tool_policy"))
                .or_insert(0) += 1;
        }
        if !receipt.assessment.tool_arguments_correct {
            *module_failure_counts
                .entry(String::from("tool_arguments"))
                .or_insert(0) += 1;
        }
        if !receipt.assessment.grounded_answer_correct {
            *module_failure_counts
                .entry(String::from("grounded_answer"))
                .or_insert(0) += 1;
        }
        if !receipt.assessment.verify_correct {
            *module_failure_counts
                .entry(String::from("verify"))
                .or_insert(0) += 1;
        }
    }

    let correction_receipt_ids = receipts
        .iter()
        .filter(|receipt| !receipt.assessment.overall_success)
        .map(|receipt| receipt.receipt_id.clone())
        .collect::<Vec<_>>();
    let source_fixture_refs = receipts
        .iter()
        .map(|receipt| receipt.source_fixture_ref.clone())
        .collect::<Vec<_>>();

    let mut ledger = CompiledAgentLearningReceiptLedger {
        schema_version: String::from(LEARNING_RECEIPT_SCHEMA_VERSION),
        ledger_id: String::from("compiled_agent.learning_receipt_ledger.v1"),
        row_id: String::from("compiled_agent.qwen35_9b_q4km.archlinux.consumer_gpu.v1"),
        baseline_revision_id: baseline_revision_id.to_string(),
        source_fixture_refs,
        correction_receipt_ids,
        module_failure_counts,
        receipts,
        summary: String::new(),
        ledger_digest: String::new(),
    };
    let success_count = ledger
        .receipts
        .iter()
        .filter(|receipt| receipt.assessment.overall_success)
        .count();
    ledger.summary = format!(
        "Compiled-agent learning ledger retains {} source receipts with {} fully-correct rows and {} correction rows.",
        ledger.receipts.len(),
        success_count,
        ledger.receipts.len().saturating_sub(success_count),
    );
    ledger.ledger_digest = stable_digest(b"compiled_agent_learning_ledger|", &ledger);
    ledger
}

fn build_compiled_agent_replay_bundle(
    ledger: &CompiledAgentLearningReceiptLedger,
) -> CompiledAgentReplayBundle {
    let mut samples = Vec::new();
    for receipt in &ledger.receipts {
        samples.push(route_replay_sample(receipt));
        samples.push(grounded_answer_replay_sample(receipt));
    }
    let mut module_sample_counts = BTreeMap::new();
    for sample in &samples {
        let key = match sample.module {
            CompiledAgentModuleKind::Route => "route",
            CompiledAgentModuleKind::GroundedAnswer => "grounded_answer",
            CompiledAgentModuleKind::ToolPolicy => "tool_policy",
            CompiledAgentModuleKind::ToolArguments => "tool_arguments",
            CompiledAgentModuleKind::Verify => "verify",
        };
        *module_sample_counts.entry(String::from(key)).or_insert(0) += 1;
    }
    let correction_sample_count = samples
        .iter()
        .filter(|sample| sample.correction_kind == CompiledAgentReplayCorrectionKind::FailureCorrection)
        .count() as u32;
    let mut bundle = CompiledAgentReplayBundle {
        schema_version: String::from(REPLAY_BUNDLE_SCHEMA_VERSION),
        bundle_id: String::from("compiled_agent.replay_bundle.v1"),
        row_id: ledger.row_id.clone(),
        baseline_revision_id: ledger.baseline_revision_id.clone(),
        source_ledger_digest: ledger.ledger_digest.clone(),
        module_sample_counts,
        correction_sample_count,
        samples,
        summary: String::new(),
        bundle_digest: String::new(),
    };
    bundle.summary = format!(
        "Compiled-agent replay bundle retains {} route samples and {} grounded-answer samples with {} correction samples.",
        bundle.module_sample_counts.get("route").copied().unwrap_or(0),
        bundle
            .module_sample_counts
            .get("grounded_answer")
            .copied()
            .unwrap_or(0),
        bundle.correction_sample_count,
    );
    bundle.bundle_digest = stable_digest(b"compiled_agent_replay_bundle|", &bundle);
    bundle
}

fn route_replay_sample(receipt: &CompiledAgentLearningReceipt) -> CompiledAgentReplaySample {
    let correction_kind = if receipt.assessment.route_correct {
        CompiledAgentReplayCorrectionKind::BehavioralClone
    } else {
        CompiledAgentReplayCorrectionKind::FailureCorrection
    };
    CompiledAgentReplaySample {
        sample_id: format!("sample.route.{}", receipt.receipt_id),
        module: CompiledAgentModuleKind::Route,
        source_receipt_id: receipt.receipt_id.clone(),
        correction_kind,
        tags: receipt.tags.clone(),
        failure_classes: if receipt.assessment.route_correct {
            Vec::new()
        } else {
            receipt.assessment.failure_classes.clone()
        },
        input: json!({
            "user_request": receipt.user_request,
        }),
        expected_output: json!({
            "route": receipt.expected_route,
        }),
        observed_output: json!({
            "route": receipt.observed_route,
        }),
    }
}

fn grounded_answer_replay_sample(receipt: &CompiledAgentLearningReceipt) -> CompiledAgentReplaySample {
    let correction_kind = if receipt.assessment.grounded_answer_correct {
        CompiledAgentReplayCorrectionKind::BehavioralClone
    } else {
        CompiledAgentReplayCorrectionKind::FailureCorrection
    };
    CompiledAgentReplaySample {
        sample_id: format!("sample.grounded_answer.{}", receipt.receipt_id),
        module: CompiledAgentModuleKind::GroundedAnswer,
        source_receipt_id: receipt.receipt_id.clone(),
        correction_kind,
        tags: receipt.tags.clone(),
        failure_classes: if receipt.assessment.grounded_answer_correct {
            Vec::new()
        } else {
            receipt.assessment.failure_classes.clone()
        },
        input: json!({
            "user_request": receipt.user_request,
            "route": receipt.expected_route,
            "tool_results": expected_tool_results(receipt),
        }),
        expected_output: json!({
            "kind": receipt.expected_public_response.kind,
            "response": receipt.expected_public_response.response,
        }),
        observed_output: json!({
            "kind": receipt.observed_public_response.kind,
            "response": receipt.observed_public_response.response,
        }),
    }
}

fn expected_public_response(
    source_fixture_ref: &str,
    label: &CanonicalSupervisionLabel,
    unsupported_template: &str,
) -> CompiledAgentLearningPublicResponse {
    match label.expected_route {
        CompiledAgentRoute::ProviderStatus => CompiledAgentLearningPublicResponse {
            kind: CompiledAgentPublicOutcomeKind::GroundedAnswer,
            response: String::from("Provider is ready to go online."),
        },
        CompiledAgentRoute::WalletStatus => CompiledAgentLearningPublicResponse {
            kind: CompiledAgentPublicOutcomeKind::GroundedAnswer,
            response: String::from("Wallet balance is 1200 sats, with 240 sats of recent earnings."),
        },
        CompiledAgentRoute::Unsupported => {
            let _ = source_fixture_ref;
            CompiledAgentLearningPublicResponse {
                kind: CompiledAgentPublicOutcomeKind::UnsupportedRefusal,
                response: unsupported_template.to_string(),
            }
        }
    }
}

fn expected_tool_names(route: CompiledAgentRoute) -> Vec<String> {
    match route {
        CompiledAgentRoute::ProviderStatus => vec![String::from("provider_status")],
        CompiledAgentRoute::WalletStatus => vec![String::from("wallet_status")],
        CompiledAgentRoute::Unsupported => Vec::new(),
    }
}

fn expected_tool_results(receipt: &CompiledAgentLearningReceipt) -> Vec<CompiledAgentToolResult> {
    match receipt.expected_route {
        CompiledAgentRoute::ProviderStatus => vec![CompiledAgentToolResult {
            tool_name: String::from("provider_status"),
            payload: json!({
                "ready": receipt.runtime_state.provider_ready,
                "blockers": receipt.runtime_state.provider_blockers,
            }),
        }],
        CompiledAgentRoute::WalletStatus => vec![CompiledAgentToolResult {
            tool_name: String::from("wallet_status"),
            payload: json!({
                "balance_sats": receipt.runtime_state.wallet_balance_sats,
                "recent_earnings_sats": receipt.runtime_state.recent_earnings_sats,
            }),
        }],
        CompiledAgentRoute::Unsupported => Vec::new(),
    }
}

fn tool_arguments_match(observed: &[CompiledAgentToolCall], expected_tool_names: &[String]) -> bool {
    let expected_calls = expected_tool_names
        .iter()
        .map(|tool_name| CompiledAgentToolCall {
            tool_name: tool_name.clone(),
            arguments: json!({}),
        })
        .collect::<Vec<_>>();
    observed == expected_calls
}

fn verify_matches_expected(
    expected_route: CompiledAgentRoute,
    observed_tool_names: &[String],
    observed_kind: CompiledAgentPublicOutcomeKind,
    expected_response: &CompiledAgentLearningPublicResponse,
    observed_response: &CompiledAgentLearningPublicResponse,
) -> bool {
    if observed_response != expected_response || observed_kind != expected_response.kind {
        return false;
    }
    match expected_route {
        CompiledAgentRoute::Unsupported => observed_tool_names.is_empty(),
        CompiledAgentRoute::ProviderStatus => observed_tool_names == [String::from("provider_status")],
        CompiledAgentRoute::WalletStatus => observed_tool_names == [String::from("wallet_status")],
    }
}

fn route_failure_class(
    user_request: &str,
    observed_route: CompiledAgentRoute,
    expected_route: CompiledAgentRoute,
) -> String {
    let lowered = user_request.to_ascii_lowercase();
    if expected_route == CompiledAgentRoute::Unsupported
        && observed_route == CompiledAgentRoute::WalletStatus
        && lowered.contains("do not")
        && lowered.contains("wallet")
    {
        return String::from("negated_route_false_positive");
    }
    String::from("route_mismatch")
}

fn fixture_slug(source_fixture_ref: &str) -> String {
    Path::new(source_fixture_ref)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("receipt")
        .to_string()
}

fn stable_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(serde_json::to_vec(value).unwrap_or_default());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_compiled_agent_learning_receipt_ledger, canonical_compiled_agent_replay_bundle,
        compiled_agent_learning_receipt_ledger_fixture_path,
        compiled_agent_replay_bundle_fixture_path, verify_compiled_agent_learning_receipt_fixtures,
    };

    #[test]
    fn compiled_agent_learning_ledger_retains_one_correction_row()
    -> Result<(), Box<dyn std::error::Error>> {
        let ledger = canonical_compiled_agent_learning_receipt_ledger()?;
        assert_eq!(ledger.receipts.len(), 4);
        assert_eq!(ledger.correction_receipt_ids.len(), 1);
        assert!(ledger
            .correction_receipt_ids
            .iter()
            .any(|receipt_id| receipt_id.contains("negated_wallet")));
        Ok(())
    }

    #[test]
    fn compiled_agent_replay_bundle_targets_route_and_grounded_answer_first()
    -> Result<(), Box<dyn std::error::Error>> {
        let bundle = canonical_compiled_agent_replay_bundle()?;
        assert_eq!(bundle.module_sample_counts.get("route"), Some(&4));
        assert_eq!(bundle.module_sample_counts.get("grounded_answer"), Some(&4));
        assert_eq!(bundle.correction_sample_count, 2);
        Ok(())
    }

    #[test]
    fn compiled_agent_learning_fixtures_match_committed_truth()
    -> Result<(), Box<dyn std::error::Error>> {
        assert!(compiled_agent_learning_receipt_ledger_fixture_path().exists());
        assert!(compiled_agent_replay_bundle_fixture_path().exists());
        verify_compiled_agent_learning_receipt_fixtures()?;
        Ok(())
    }
}
