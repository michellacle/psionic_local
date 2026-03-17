use std::collections::{BTreeMap, VecDeque};

use psionic_models::{
    TassadarExecutorContractError, TassadarExecutorFixture, TassadarExecutorModelDescriptor,
    TassadarTraceTokenizer,
};
use psionic_runtime::{
    TASSADAR_ARTICLE_CLASS_BENCHMARK_ENVIRONMENT_REF, TASSADAR_ARTICLE_CLASS_BENCHMARK_REF,
    TASSADAR_ARTICLE_CLASS_BENCHMARK_REPORT_REF, TassadarExecution, TassadarExecutionEvidenceBundle,
    TassadarExecutionRefusal, TassadarExecutorDecodeMode, TassadarExecutorExecutionReport,
    TassadarExecutorSelectionDiagnostic, TassadarInstruction, TassadarProgramArtifact,
    TassadarRuntimeCapabilityReport, TassadarTraceEvent, TassadarTraceStep, TassadarValidationCase,
    build_tassadar_execution_evidence_bundle, execute_tassadar_executor_request,
    tassadar_article_class_corpus, tassadar_trace_abi_for_profile_id, tassadar_wasm_profile_for_id,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Dedicated served product identifier for the Tassadar executor lane.
pub const EXECUTOR_TRACE_PRODUCT_ID: &str = "psionic.executor_trace";

/// Dedicated planner-owned routing product for exact executor delegation.
pub const PLANNER_EXECUTOR_ROUTE_PRODUCT_ID: &str = "psionic.planner_executor_route";
/// Dedicated served product identifier for article-class Tassadar sessions.
pub const ARTICLE_EXECUTOR_SESSION_PRODUCT_ID: &str = "psionic.article_executor_session";
/// Dedicated planner-owned hybrid workflow product for article-class compute spans.
pub const ARTICLE_HYBRID_WORKFLOW_PRODUCT_ID: &str = "psionic.article_hybrid_workflow";
/// Canonical acceptance artifact for the article-session serving surface.
pub const TASSADAR_ARTICLE_EXECUTOR_SESSION_ARTIFACT_REF: &str =
    "fixtures/tassadar/reports/tassadar_article_executor_session_artifact.json";
/// Canonical acceptance artifact for the article hybrid-workflow surface.
pub const TASSADAR_ARTICLE_HYBRID_WORKFLOW_ARTIFACT_REF: &str =
    "fixtures/tassadar/reports/tassadar_article_hybrid_workflow_artifact.json";

const ARTICLE_EXECUTOR_READABLE_LOG_MAX_LINES: usize = 96;
const ARTICLE_EXECUTOR_TOKEN_TRACE_MAX_TOKENS: usize = 256;
const ARTICLE_EXECUTOR_TOKEN_TRACE_CHUNK_SIZE: usize = 32;

/// Explicit request contract for the served Tassadar executor lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarExecutorRequest {
    /// Stable request identifier.
    pub request_id: String,
    /// Product identifier. Must be `psionic.executor_trace`.
    pub product_id: String,
    /// Optional explicit executor model id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_model_id: Option<String>,
    /// Digest-bound program artifact submitted to the executor.
    pub program_artifact: TassadarProgramArtifact,
    /// Requested decode mode for the execution.
    pub requested_decode_mode: TassadarExecutorDecodeMode,
    /// Explicit environment refs carried into runtime-manifest lineage.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub environment_refs: Vec<String>,
}

impl TassadarExecutorRequest {
    /// Creates a request for the explicit executor-trace product family.
    #[must_use]
    pub fn new(
        request_id: impl Into<String>,
        program_artifact: TassadarProgramArtifact,
        requested_decode_mode: TassadarExecutorDecodeMode,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            product_id: String::from(EXECUTOR_TRACE_PRODUCT_ID),
            requested_model_id: None,
            program_artifact,
            requested_decode_mode,
            environment_refs: Vec::new(),
        }
    }

    /// Pins execution to one explicit executor model.
    #[must_use]
    pub fn with_requested_model_id(mut self, requested_model_id: impl Into<String>) -> Self {
        self.requested_model_id = Some(requested_model_id.into());
        self
    }

    /// Carries environment refs into the served evidence bundle.
    #[must_use]
    pub fn with_environment_refs(mut self, environment_refs: Vec<String>) -> Self {
        let mut environment_refs = environment_refs;
        environment_refs.sort();
        environment_refs.dedup();
        self.environment_refs = environment_refs;
        self
    }

    /// Returns a stable digest for the request surface.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let encoded = serde_json::to_vec(self).expect("Tassadar executor request should serialize");
        let mut hasher = Sha256::new();
        hasher.update(b"tassadar_executor_request|");
        hasher.update(encoded);
        hex::encode(hasher.finalize())
    }
}

/// Explicit refusal response for unsupported ABI, profile, decode, or model pairings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarExecutorRefusalResponse {
    /// Stable request identifier.
    pub request_id: String,
    /// Product identifier.
    pub product_id: String,
    /// Served executor model descriptor that evaluated the request.
    pub model_descriptor: TassadarExecutorModelDescriptor,
    /// Runtime capability report visible to the caller.
    pub runtime_capability: TassadarRuntimeCapabilityReport,
    /// Contract error when model/program pairing failed before selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract_error: Option<TassadarExecutorContractError>,
    /// Runtime selection diagnostic when decode selection reached refusal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<TassadarExecutorSelectionDiagnostic>,
    /// Human-readable refusal detail.
    pub detail: String,
}

/// Completed served executor response carrying the exact runtime truth.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarExecutorResponse {
    /// Stable request identifier.
    pub request_id: String,
    /// Product identifier.
    pub product_id: String,
    /// Served executor model descriptor.
    pub model_descriptor: TassadarExecutorModelDescriptor,
    /// Runtime capability report visible to the caller.
    pub runtime_capability: TassadarRuntimeCapabilityReport,
    /// Direct/fallback selection plus realized execution.
    pub execution_report: TassadarExecutorExecutionReport,
    /// Runtime-manifest, trace, proof, and proof-bundle evidence.
    pub evidence_bundle: TassadarExecutionEvidenceBundle,
    /// Ordered environment refs carried into execution lineage.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub environment_refs: Vec<String>,
}

impl TassadarExecutorResponse {
    /// Returns the final scalar outputs emitted by the execution.
    #[must_use]
    pub fn final_outputs(&self) -> &[i32] {
        &self.execution_report.execution.outputs
    }

    /// Returns the emitted trace artifact.
    #[must_use]
    pub fn trace_artifact(&self) -> &psionic_runtime::TassadarTraceArtifact {
        &self.evidence_bundle.trace_artifact
    }

    /// Returns the proof-bearing trace artifact.
    #[must_use]
    pub fn trace_proof(&self) -> &psionic_runtime::TassadarTraceProofArtifact {
        &self.evidence_bundle.trace_proof
    }
}

/// Served outcome for one executor request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TassadarExecutorOutcome {
    /// The executor accepted and completed the request.
    Completed { response: TassadarExecutorResponse },
    /// The executor refused the request explicitly.
    Refused {
        refusal: TassadarExecutorRefusalResponse,
    },
}

/// Step event emitted by the explicit executor trace stream.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarExecutorTraceStepEvent {
    /// Ordinal step index.
    pub step_index: u64,
    /// Full append-only trace step.
    pub step: TassadarTraceStep,
}

/// Output event emitted by the explicit executor trace stream.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarExecutorOutputEvent {
    /// Output ordinal in the final ordered output vector.
    pub ordinal: usize,
    /// Output scalar value.
    pub value: i32,
}

/// Terminal event emitted by the explicit executor trace stream.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarExecutorTerminalEvent {
    /// Final served outcome.
    pub outcome: TassadarExecutorOutcome,
}

/// Typed event emitted by the pull-driven executor trace stream.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TassadarExecutorStreamEvent {
    /// Runtime capability report surfaced before execution.
    Capability {
        /// Runtime capability report visible to the caller.
        runtime_capability: TassadarRuntimeCapabilityReport,
    },
    /// Decode-selection diagnostic surfaced before the first trace step.
    Selection {
        /// Direct/fallback/refused selection diagnostic.
        selection: TassadarExecutorSelectionDiagnostic,
    },
    /// One append-only trace step.
    TraceStep {
        /// Step event payload.
        trace_step: TassadarExecutorTraceStepEvent,
    },
    /// One emitted output value.
    Output {
        /// Output event payload.
        output: TassadarExecutorOutputEvent,
    },
    /// Terminal completion or refusal.
    Terminal {
        /// Terminal payload.
        terminal: TassadarExecutorTerminalEvent,
    },
}

/// Typed request-validation or model-resolution error for the served executor lane.
#[derive(Clone, Debug, Error, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TassadarExecutorServiceError {
    /// The request targeted a different served product family.
    #[error("unsupported Tassadar served product `{product_id}`")]
    UnsupportedProduct {
        /// Product identifier supplied by the caller.
        product_id: String,
    },
    /// The request named an executor model that is not registered.
    #[error("unknown Tassadar executor model `{model_id}`")]
    UnknownModel {
        /// Requested model identifier.
        model_id: String,
    },
    /// The request reached execution even though runtime selection refused it.
    #[error(transparent)]
    ExecutionRefusal(#[from] TassadarExecutionRefusal),
}

/// Pre-execution contract and selection report for one executor request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarExecutorPreflightReport {
    /// Stable request identifier.
    pub request_id: String,
    /// Product identifier.
    pub product_id: String,
    /// Served executor model descriptor that evaluated the request.
    pub model_descriptor: TassadarExecutorModelDescriptor,
    /// Runtime capability report visible to the caller.
    pub runtime_capability: TassadarRuntimeCapabilityReport,
    /// Contract error when the request failed before decode selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract_error: Option<TassadarExecutorContractError>,
    /// Decode selection diagnostic when contract validation succeeded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<TassadarExecutorSelectionDiagnostic>,
}

/// Pull-driven local stream for explicit executor-trace products.
#[derive(Clone, Debug, Default)]
pub struct LocalTassadarExecutorStream {
    events: VecDeque<TassadarExecutorStreamEvent>,
}

impl LocalTassadarExecutorStream {
    fn from_events(events: Vec<TassadarExecutorStreamEvent>) -> Self {
        Self {
            events: VecDeque::from(events),
        }
    }

    /// Returns the next typed stream event.
    pub fn next_event(&mut self) -> Option<TassadarExecutorStreamEvent> {
        self.events.pop_front()
    }
}

/// Local reference implementation of the explicit Tassadar served product.
#[derive(Clone, Debug)]
pub struct LocalTassadarExecutorService {
    fixtures: BTreeMap<String, TassadarExecutorFixture>,
    default_model_id: String,
}

impl Default for LocalTassadarExecutorService {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalTassadarExecutorService {
    /// Creates the default in-process Tassadar executor service.
    #[must_use]
    pub fn new() -> Self {
        let fixture = TassadarExecutorFixture::new();
        let model_id = fixture.descriptor().model.model_id.clone();
        let mut fixtures = BTreeMap::new();
        fixtures.insert(model_id.clone(), fixture);
        Self {
            fixtures,
            default_model_id: model_id,
        }
    }

    /// Registers one additional executor fixture.
    #[must_use]
    pub fn with_fixture(mut self, fixture: TassadarExecutorFixture) -> Self {
        let model_id = fixture.descriptor().model.model_id.clone();
        self.fixtures.insert(model_id.clone(), fixture);
        if self.fixtures.len() == 1 {
            self.default_model_id = model_id;
        }
        self
    }

    /// Executes one request through the explicit executor-trace surface.
    pub fn execute(
        &self,
        request: &TassadarExecutorRequest,
    ) -> Result<TassadarExecutorOutcome, TassadarExecutorServiceError> {
        self.validate_product(request)?;
        let fixture = self.resolve_fixture(request)?;
        Ok(self.execute_with_fixture(fixture, request))
    }

    /// Starts a pull-driven explicit executor-trace stream.
    pub fn execute_stream(
        &self,
        request: &TassadarExecutorRequest,
    ) -> Result<LocalTassadarExecutorStream, TassadarExecutorServiceError> {
        self.validate_product(request)?;
        let fixture = self.resolve_fixture(request)?;
        let outcome = self.execute_with_fixture(fixture, request);
        Ok(LocalTassadarExecutorStream::from_events(
            stream_events_for_outcome(fixture, outcome),
        ))
    }

    /// Resolves the explicit model and selection truth without executing the program.
    pub fn preflight(
        &self,
        request: &TassadarExecutorRequest,
    ) -> Result<TassadarExecutorPreflightReport, TassadarExecutorServiceError> {
        self.validate_product(request)?;
        let fixture = self.resolve_fixture(request)?;
        let descriptor = fixture.descriptor().clone();
        let runtime_capability = fixture.runtime_capability_report();
        let contract_error = descriptor
            .validate_program_artifact(&request.program_artifact, request.requested_decode_mode)
            .err();
        let selection = if contract_error.is_none() {
            Some(fixture.runtime_selection_diagnostic(
                &request.program_artifact.validated_program,
                request.requested_decode_mode,
            ))
        } else {
            None
        };
        Ok(TassadarExecutorPreflightReport {
            request_id: request.request_id.clone(),
            product_id: request.product_id.clone(),
            model_descriptor: descriptor,
            runtime_capability,
            contract_error,
            selection,
        })
    }

    fn validate_product(
        &self,
        request: &TassadarExecutorRequest,
    ) -> Result<(), TassadarExecutorServiceError> {
        if request.product_id == EXECUTOR_TRACE_PRODUCT_ID {
            Ok(())
        } else {
            Err(TassadarExecutorServiceError::UnsupportedProduct {
                product_id: request.product_id.clone(),
            })
        }
    }

    fn resolve_fixture(
        &self,
        request: &TassadarExecutorRequest,
    ) -> Result<&TassadarExecutorFixture, TassadarExecutorServiceError> {
        let requested_model_id = request
            .requested_model_id
            .as_deref()
            .unwrap_or(self.default_model_id.as_str());
        self.fixtures.get(requested_model_id).ok_or_else(|| {
            TassadarExecutorServiceError::UnknownModel {
                model_id: requested_model_id.to_string(),
            }
        })
    }

    fn execute_with_fixture(
        &self,
        fixture: &TassadarExecutorFixture,
        request: &TassadarExecutorRequest,
    ) -> TassadarExecutorOutcome {
        let descriptor = fixture.descriptor().clone();
        let runtime_capability = fixture.runtime_capability_report();
        match descriptor
            .validate_program_artifact(&request.program_artifact, request.requested_decode_mode)
        {
            Ok(()) => {}
            Err(contract_error) => {
                return TassadarExecutorOutcome::Refused {
                    refusal: TassadarExecutorRefusalResponse {
                        request_id: request.request_id.clone(),
                        product_id: request.product_id.clone(),
                        model_descriptor: descriptor,
                        runtime_capability,
                        detail: contract_error.to_string(),
                        contract_error: Some(contract_error),
                        selection: None,
                    },
                };
            }
        }

        let selection = fixture.runtime_selection_diagnostic(
            &request.program_artifact.validated_program,
            request.requested_decode_mode,
        );
        if selection.effective_decode_mode.is_none() {
            return TassadarExecutorOutcome::Refused {
                refusal: TassadarExecutorRefusalResponse {
                    request_id: request.request_id.clone(),
                    product_id: request.product_id.clone(),
                    model_descriptor: descriptor,
                    runtime_capability,
                    detail: selection.detail.clone(),
                    contract_error: None,
                    selection: Some(selection),
                },
            };
        }

        match execute_tassadar_executor_request(
            &request.program_artifact.validated_program,
            request.requested_decode_mode,
            request.program_artifact.trace_abi_version,
            Some(descriptor.compatibility.supported_decode_modes.as_slice()),
        ) {
            Ok(execution_report) => {
                let evidence_bundle = build_tassadar_execution_evidence_bundle(
                    request.request_id.clone(),
                    request.stable_digest(),
                    request.product_id.clone(),
                    descriptor.model.model_id.clone(),
                    descriptor.stable_digest(),
                    request.environment_refs.clone(),
                    &request.program_artifact,
                    execution_report
                        .selection
                        .effective_decode_mode
                        .expect("completed execution should surface an effective decode mode"),
                    &execution_report.execution,
                );
                TassadarExecutorOutcome::Completed {
                    response: TassadarExecutorResponse {
                        request_id: request.request_id.clone(),
                        product_id: request.product_id.clone(),
                        model_descriptor: descriptor,
                        runtime_capability,
                        execution_report,
                        evidence_bundle,
                        environment_refs: request.environment_refs.clone(),
                    },
                }
            }
            Err(selection) => TassadarExecutorOutcome::Refused {
                refusal: TassadarExecutorRefusalResponse {
                    request_id: request.request_id.clone(),
                    product_id: request.product_id.clone(),
                    model_descriptor: descriptor,
                    runtime_capability,
                    detail: selection.detail.clone(),
                    contract_error: None,
                    selection: Some(selection),
                },
            },
        }
    }
}

/// Canonical benchmark/workload identity for one article-class Tassadar case.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarArticleBenchmarkIdentity {
    /// Stable benchmark ref for the article-class suite.
    pub benchmark_ref: String,
    /// Stable benchmark environment ref for the article-class suite.
    pub benchmark_environment_ref: String,
    /// Canonical committed benchmark report anchor.
    pub benchmark_report_ref: String,
    /// Stable workload family label.
    pub workload_family: String,
    /// Stable case identifier inside the article-class suite.
    pub case_id: String,
    /// Short case summary.
    pub case_summary: String,
    /// Stable program identifier for the selected workload.
    pub program_id: String,
    /// Stable Wasm profile identifier for the selected workload.
    pub wasm_profile_id: String,
    /// Digest of the validated program selected for the session.
    pub validated_program_digest: String,
}

impl TassadarArticleBenchmarkIdentity {
    fn for_case(case: &TassadarValidationCase, artifact: &TassadarProgramArtifact) -> Self {
        Self {
            benchmark_ref: String::from(TASSADAR_ARTICLE_CLASS_BENCHMARK_REF),
            benchmark_environment_ref: String::from(
                TASSADAR_ARTICLE_CLASS_BENCHMARK_ENVIRONMENT_REF,
            ),
            benchmark_report_ref: String::from(TASSADAR_ARTICLE_CLASS_BENCHMARK_REPORT_REF),
            workload_family: String::from(article_workload_family(case.case_id.as_str())),
            case_id: case.case_id.clone(),
            case_summary: case.summary.clone(),
            program_id: case.program.program_id.clone(),
            wasm_profile_id: case.program.profile_id.clone(),
            validated_program_digest: artifact.validated_program_digest.clone(),
        }
    }
}

/// Proof and trace identity that survives the article-session serving boundary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarArticleProofIdentity {
    /// Underlying exact executor product that realized the compute span.
    pub executor_product_id: String,
    /// Stable trace artifact identifier.
    pub trace_artifact_id: String,
    /// Stable trace artifact digest.
    pub trace_artifact_digest: String,
    /// Stable trace digest.
    pub trace_digest: String,
    /// Stable trace-proof artifact identifier.
    pub trace_proof_id: String,
    /// Stable trace-proof digest.
    pub trace_proof_digest: String,
    /// Stable validated-program artifact digest.
    pub program_artifact_digest: String,
    /// Stable runtime-manifest identity digest.
    pub runtime_manifest_identity_digest: String,
    /// Stable runtime-manifest digest.
    pub runtime_manifest_digest: String,
    /// Stable proof-bundle request digest.
    pub proof_bundle_request_digest: String,
    /// Stable proof-bundle model id when one exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proof_bundle_model_id: Option<String>,
}

impl TassadarArticleProofIdentity {
    fn from_response(response: &TassadarExecutorResponse) -> Self {
        Self {
            executor_product_id: response.evidence_bundle.proof_bundle.product_id.clone(),
            trace_artifact_id: response.evidence_bundle.trace_artifact.artifact_id.clone(),
            trace_artifact_digest: response.evidence_bundle.trace_artifact.artifact_digest.clone(),
            trace_digest: response.evidence_bundle.trace_artifact.trace_digest.clone(),
            trace_proof_id: response.evidence_bundle.trace_proof.proof_artifact_id.clone(),
            trace_proof_digest: response.evidence_bundle.trace_proof.proof_digest.clone(),
            program_artifact_digest: response
                .evidence_bundle
                .trace_proof
                .program_artifact_digest
                .clone(),
            runtime_manifest_identity_digest: response
                .evidence_bundle
                .runtime_manifest
                .identity_digest
                .clone(),
            runtime_manifest_digest: response
                .evidence_bundle
                .runtime_manifest
                .manifest_digest
                .clone(),
            proof_bundle_request_digest: response.evidence_bundle.proof_bundle.request_digest.clone(),
            proof_bundle_model_id: response.evidence_bundle.proof_bundle.model_id.clone(),
        }
    }
}

/// Derived readable-log view over one article-class executor session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarArticleReadableLogExcerpt {
    /// Maximum retained line count for the derived view.
    pub max_lines: usize,
    /// Total line count before truncation.
    pub total_line_count: usize,
    /// Whether the derived view was truncated.
    pub truncated: bool,
    /// Derived readable-log lines.
    pub lines: Vec<String>,
    /// Plain-language posture describing how the view relates to machine truth.
    pub derivation_posture: String,
}

/// Derived symbolic token-trace view over one article-class executor session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarArticleTokenTraceExcerpt {
    /// Maximum retained token count for the derived view.
    pub max_tokens: usize,
    /// Total token count before truncation.
    pub total_token_count: usize,
    /// Prompt/target boundary inside the symbolic token stream.
    pub prompt_token_count: usize,
    /// Whether the derived view was truncated.
    pub truncated: bool,
    /// Stable tokenizer digest for the derived symbolic view.
    pub tokenizer_digest: String,
    /// Derived symbolic token strings.
    pub tokens: Vec<String>,
    /// Plain-language posture describing how the view relates to machine truth.
    pub derivation_posture: String,
}

/// Specialized request for one canonical article-class Tassadar session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarArticleExecutorSessionRequest {
    /// Stable request identifier.
    pub request_id: String,
    /// Product identifier. Must be `psionic.article_executor_session`.
    pub product_id: String,
    /// Stable article-class case identifier.
    pub article_case_id: String,
    /// Optional explicit executor model id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_model_id: Option<String>,
    /// Requested decode mode for the exact executor path.
    pub requested_decode_mode: TassadarExecutorDecodeMode,
    /// Ordered environment refs carried into lineage in addition to the canonical benchmark ref.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub environment_refs: Vec<String>,
}

impl TassadarArticleExecutorSessionRequest {
    /// Creates a new article-class executor session request.
    #[must_use]
    pub fn new(
        request_id: impl Into<String>,
        article_case_id: impl Into<String>,
        requested_decode_mode: TassadarExecutorDecodeMode,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            product_id: String::from(ARTICLE_EXECUTOR_SESSION_PRODUCT_ID),
            article_case_id: article_case_id.into(),
            requested_model_id: None,
            requested_decode_mode,
            environment_refs: Vec::new(),
        }
    }

    /// Pins execution to one explicit executor model.
    #[must_use]
    pub fn with_requested_model_id(mut self, requested_model_id: impl Into<String>) -> Self {
        self.requested_model_id = Some(requested_model_id.into());
        self
    }

    /// Carries extra environment refs into the served evidence bundle.
    #[must_use]
    pub fn with_environment_refs(mut self, mut environment_refs: Vec<String>) -> Self {
        environment_refs.sort();
        environment_refs.dedup();
        self.environment_refs = environment_refs;
        self
    }

    /// Returns a stable digest for the article-session request.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        stable_digest(b"tassadar_article_executor_session_request|", self)
    }
}

/// Typed refusal response for article-session requests.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarArticleExecutorSessionRefusalResponse {
    /// Stable request identifier.
    pub request_id: String,
    /// Product identifier.
    pub product_id: String,
    /// Canonical benchmark/workload identity when the case was resolved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub benchmark_identity: Option<TassadarArticleBenchmarkIdentity>,
    /// Served executor model descriptor when one was resolved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_descriptor: Option<TassadarExecutorModelDescriptor>,
    /// Runtime capability report visible to the caller.
    pub runtime_capability: TassadarRuntimeCapabilityReport,
    /// Contract error when model/program pairing failed before selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract_error: Option<TassadarExecutorContractError>,
    /// Runtime selection diagnostic when one was available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<TassadarExecutorSelectionDiagnostic>,
    /// Human-readable refusal detail.
    pub detail: String,
}

/// Completed served article session carrying benchmark, proof, and derived views.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarArticleExecutorSessionResponse {
    /// Stable request identifier.
    pub request_id: String,
    /// Product identifier.
    pub product_id: String,
    /// Canonical benchmark/workload identity.
    pub benchmark_identity: TassadarArticleBenchmarkIdentity,
    /// Proof identity preserved across the serving boundary.
    pub proof_identity: TassadarArticleProofIdentity,
    /// Underlying exact executor response.
    pub executor_response: TassadarExecutorResponse,
    /// Derived readable-log view over the canonical trace.
    pub readable_log: TassadarArticleReadableLogExcerpt,
    /// Derived symbolic token-trace view over the canonical trace.
    pub token_trace: TassadarArticleTokenTraceExcerpt,
}

impl TassadarArticleExecutorSessionResponse {
    /// Returns the final scalar outputs emitted by the execution.
    #[must_use]
    pub fn final_outputs(&self) -> &[i32] {
        self.executor_response.final_outputs()
    }
}

/// Served outcome for one article-session request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TassadarArticleExecutorSessionOutcome {
    /// The article session completed through the Psionic-owned executor lane.
    Completed {
        /// Completed article-session response.
        response: TassadarArticleExecutorSessionResponse,
    },
    /// The article session refused the request explicitly.
    Refused {
        /// Typed refusal response.
        refusal: TassadarArticleExecutorSessionRefusalResponse,
    },
}

/// Benchmark identity event emitted by the article-session stream.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarArticleBenchmarkIdentityEvent {
    /// Canonical article benchmark/workload identity.
    pub benchmark_identity: TassadarArticleBenchmarkIdentity,
}

/// Proof identity event emitted by the article-session stream.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarArticleProofIdentityEvent {
    /// Preserved proof identity for the completed exact executor span.
    pub proof_identity: TassadarArticleProofIdentity,
}

/// Readable-log event emitted by the article-session stream.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarArticleReadableLogLineEvent {
    /// Zero-based readable-log line index within the derived excerpt.
    pub line_index: usize,
    /// One derived readable-log line.
    pub line: String,
}

/// Token-trace event emitted by the article-session stream.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarArticleTokenTraceChunkEvent {
    /// Zero-based chunk index.
    pub chunk_index: usize,
    /// Prompt/target boundary in the underlying symbolic token stream.
    pub prompt_token_count: usize,
    /// Total token count before truncation.
    pub total_token_count: usize,
    /// Whether the underlying symbolic view was truncated.
    pub truncated: bool,
    /// Ordered symbolic tokens in this chunk.
    pub tokens: Vec<String>,
}

/// Terminal event emitted by the article-session stream.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarArticleExecutorSessionTerminalEvent {
    /// Final served outcome.
    pub outcome: TassadarArticleExecutorSessionOutcome,
}

/// Typed event emitted by the pull-driven article-session stream.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TassadarArticleExecutorSessionStreamEvent {
    /// Canonical benchmark/workload identity for the session.
    BenchmarkIdentity {
        /// Benchmark identity payload.
        benchmark_identity: TassadarArticleBenchmarkIdentityEvent,
    },
    /// Runtime capability surfaced before execution.
    Capability {
        /// Runtime capability report visible to the caller.
        runtime_capability: TassadarRuntimeCapabilityReport,
    },
    /// Decode-selection diagnostic surfaced before derived trace views.
    Selection {
        /// Direct/fallback/refused selection diagnostic.
        selection: TassadarExecutorSelectionDiagnostic,
    },
    /// Proof identity surfaced for completed sessions.
    ProofIdentity {
        /// Proof identity payload.
        proof_identity: TassadarArticleProofIdentityEvent,
    },
    /// One derived readable-log line.
    ReadableLogLine {
        /// Readable-log line payload.
        readable_log_line: TassadarArticleReadableLogLineEvent,
    },
    /// One symbolic token-trace chunk.
    TokenTraceChunk {
        /// Token-trace chunk payload.
        token_trace_chunk: TassadarArticleTokenTraceChunkEvent,
    },
    /// One emitted output value.
    Output {
        /// Output event payload.
        output: TassadarExecutorOutputEvent,
    },
    /// Terminal completion or refusal.
    Terminal {
        /// Terminal payload.
        terminal: TassadarArticleExecutorSessionTerminalEvent,
    },
}

/// Product-validation error for the article-session serving surface.
#[derive(Clone, Debug, Error, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TassadarArticleExecutorSessionServiceError {
    /// The request targeted a different served product family.
    #[error("unsupported Tassadar article-session product `{product_id}`")]
    UnsupportedProduct {
        /// Product identifier supplied by the caller.
        product_id: String,
    },
}

/// Pull-driven local stream for explicit article-session products.
#[derive(Clone, Debug, Default)]
pub struct LocalTassadarArticleExecutorSessionStream {
    events: VecDeque<TassadarArticleExecutorSessionStreamEvent>,
}

impl LocalTassadarArticleExecutorSessionStream {
    fn from_events(events: Vec<TassadarArticleExecutorSessionStreamEvent>) -> Self {
        Self {
            events: VecDeque::from(events),
        }
    }

    /// Returns the next typed stream event.
    pub fn next_event(&mut self) -> Option<TassadarArticleExecutorSessionStreamEvent> {
        self.events.pop_front()
    }
}

/// Local reference implementation of the article-workload session surface.
#[derive(Clone, Debug)]
pub struct LocalTassadarArticleExecutorSessionService {
    executor_service: LocalTassadarExecutorService,
}

impl Default for LocalTassadarArticleExecutorSessionService {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalTassadarArticleExecutorSessionService {
    /// Creates the default in-process article-session service.
    #[must_use]
    pub fn new() -> Self {
        Self {
            executor_service: LocalTassadarExecutorService::new()
                .with_fixture(TassadarExecutorFixture::article_i32_compute_v1()),
        }
    }

    /// Replaces the backing executor service.
    #[must_use]
    pub fn with_executor_service(mut self, executor_service: LocalTassadarExecutorService) -> Self {
        self.executor_service = executor_service;
        self
    }

    /// Executes one article-class session through the specialized serving surface.
    pub fn execute(
        &self,
        request: &TassadarArticleExecutorSessionRequest,
    ) -> Result<TassadarArticleExecutorSessionOutcome, TassadarArticleExecutorSessionServiceError>
    {
        self.validate_product(request)?;
        let runtime_capability = TassadarRuntimeCapabilityReport::current();
        let Some(case) = article_case_by_id(request.article_case_id.as_str()) else {
            return Ok(TassadarArticleExecutorSessionOutcome::Refused {
                refusal: TassadarArticleExecutorSessionRefusalResponse {
                    request_id: request.request_id.clone(),
                    product_id: request.product_id.clone(),
                    benchmark_identity: None,
                    model_descriptor: None,
                    runtime_capability,
                    contract_error: None,
                    selection: None,
                    detail: format!(
                        "article workload `{}` is not present in the canonical Tassadar article corpus",
                        request.article_case_id
                    ),
                },
            });
        };

        let executor_request = self.executor_request_for(request, &case);
        let benchmark_identity =
            TassadarArticleBenchmarkIdentity::for_case(&case, &executor_request.program_artifact);

        match self.executor_service.execute(&executor_request) {
            Ok(TassadarExecutorOutcome::Completed { response }) => {
                let proof_identity = TassadarArticleProofIdentity::from_response(&response);
                let readable_log = build_article_readable_log_excerpt(&case, &response.execution_report.execution);
                let token_trace = build_article_token_trace_excerpt(
                    &case,
                    &response.execution_report.execution,
                );
                Ok(TassadarArticleExecutorSessionOutcome::Completed {
                    response: TassadarArticleExecutorSessionResponse {
                        request_id: request.request_id.clone(),
                        product_id: request.product_id.clone(),
                        benchmark_identity,
                        proof_identity,
                        executor_response: response,
                        readable_log,
                        token_trace,
                    },
                })
            }
            Ok(TassadarExecutorOutcome::Refused { refusal }) => {
                Ok(TassadarArticleExecutorSessionOutcome::Refused {
                    refusal: TassadarArticleExecutorSessionRefusalResponse {
                        request_id: request.request_id.clone(),
                        product_id: request.product_id.clone(),
                        benchmark_identity: Some(benchmark_identity),
                        model_descriptor: Some(refusal.model_descriptor),
                        runtime_capability: refusal.runtime_capability,
                        contract_error: refusal.contract_error,
                        selection: refusal.selection,
                        detail: refusal.detail,
                    },
                })
            }
            Err(TassadarExecutorServiceError::UnknownModel { model_id }) => {
                Ok(TassadarArticleExecutorSessionOutcome::Refused {
                    refusal: TassadarArticleExecutorSessionRefusalResponse {
                        request_id: request.request_id.clone(),
                        product_id: request.product_id.clone(),
                        benchmark_identity: Some(benchmark_identity),
                        model_descriptor: None,
                        runtime_capability,
                        contract_error: None,
                        selection: None,
                        detail: format!("unknown Tassadar executor model `{model_id}`"),
                    },
                })
            }
            Err(TassadarExecutorServiceError::UnsupportedProduct { product_id }) => Err(
                TassadarArticleExecutorSessionServiceError::UnsupportedProduct { product_id },
            ),
            Err(TassadarExecutorServiceError::ExecutionRefusal(refusal)) => {
                Ok(TassadarArticleExecutorSessionOutcome::Refused {
                    refusal: TassadarArticleExecutorSessionRefusalResponse {
                        request_id: request.request_id.clone(),
                        product_id: request.product_id.clone(),
                        benchmark_identity: Some(benchmark_identity),
                        model_descriptor: None,
                        runtime_capability,
                        contract_error: None,
                        selection: None,
                        detail: refusal.to_string(),
                    },
                })
            }
        }
    }

    /// Starts a pull-driven article-session stream.
    pub fn execute_stream(
        &self,
        request: &TassadarArticleExecutorSessionRequest,
    ) -> Result<LocalTassadarArticleExecutorSessionStream, TassadarArticleExecutorSessionServiceError>
    {
        let outcome = self.execute(request)?;
        Ok(LocalTassadarArticleExecutorSessionStream::from_events(
            article_stream_events_for_outcome(&outcome),
        ))
    }

    fn validate_product(
        &self,
        request: &TassadarArticleExecutorSessionRequest,
    ) -> Result<(), TassadarArticleExecutorSessionServiceError> {
        if request.product_id == ARTICLE_EXECUTOR_SESSION_PRODUCT_ID {
            Ok(())
        } else {
            Err(TassadarArticleExecutorSessionServiceError::UnsupportedProduct {
                product_id: request.product_id.clone(),
            })
        }
    }

    fn executor_request_for(
        &self,
        request: &TassadarArticleExecutorSessionRequest,
        case: &TassadarValidationCase,
    ) -> TassadarExecutorRequest {
        let profile = tassadar_wasm_profile_for_id(case.program.profile_id.as_str())
            .expect("canonical article case should use a supported profile");
        let trace_abi = tassadar_trace_abi_for_profile_id(case.program.profile_id.as_str())
            .expect("canonical article case should use a supported trace ABI");
        let artifact = TassadarProgramArtifact::fixture_reference(
            format!("artifact://tassadar/article_session/{}", case.case_id),
            &profile,
            &trace_abi,
            case.program.clone(),
        )
        .expect("canonical article case should assemble into a program artifact");
        TassadarExecutorRequest::new(
            request.request_id.clone(),
            artifact,
            request.requested_decode_mode,
        )
        .with_requested_model_id(
            request
                .requested_model_id
                .clone()
                .unwrap_or_else(|| String::from(TassadarExecutorFixture::ARTICLE_I32_COMPUTE_MODEL_ID)),
        )
        .with_environment_refs(merge_article_environment_refs(
            request.environment_refs.as_slice(),
        ))
    }
}

/// Planner-visible fallback behavior when executor routing is not taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarPlannerFallbackPolicy {
    /// Refuse the planner request rather than silently degrading it.
    Refuse,
    /// Return a typed planner fallback summary while preserving executor truth.
    PlannerSummary,
}

/// Planner-visible budget for one exact-computation subproblem.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPlannerRoutingBudget {
    /// Maximum accepted validated-program length.
    pub max_program_len: usize,
    /// Maximum accepted trace-step budget inferred from the targeted profile.
    pub max_trace_steps: usize,
    /// Maximum accepted environment refs carried into lineage.
    pub max_environment_refs: usize,
}

impl TassadarPlannerRoutingBudget {
    /// Creates one explicit routing budget.
    #[must_use]
    pub fn new(
        max_program_len: usize,
        max_trace_steps: usize,
        max_environment_refs: usize,
    ) -> Self {
        Self {
            max_program_len,
            max_trace_steps,
            max_environment_refs,
        }
    }
}

impl Default for TassadarPlannerRoutingBudget {
    fn default() -> Self {
        Self::new(128, 512, 8)
    }
}

/// Planner-visible policy for exact executor delegation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPlannerRoutingPolicy {
    /// Whether exact executor delegation is enabled at all.
    pub allow_executor_delegation: bool,
    /// Whether planner routing may accept runtime decode fallback.
    pub allow_runtime_decode_fallback: bool,
    /// Whether routing requires the requested decode path to remain direct.
    pub require_direct_decode: bool,
    /// Whether proof-bearing executor evidence must remain present on success.
    pub require_proof_bundle: bool,
    /// Typed behavior when executor routing is skipped or refused.
    pub fallback_policy: TassadarPlannerFallbackPolicy,
}

impl TassadarPlannerRoutingPolicy {
    /// Returns the canonical truthful planner routing policy.
    #[must_use]
    pub fn exact_executor_default() -> Self {
        Self {
            allow_executor_delegation: true,
            allow_runtime_decode_fallback: true,
            require_direct_decode: false,
            require_proof_bundle: true,
            fallback_policy: TassadarPlannerFallbackPolicy::Refuse,
        }
    }

    /// Replaces the fallback behavior.
    #[must_use]
    pub fn with_fallback_policy(mut self, fallback_policy: TassadarPlannerFallbackPolicy) -> Self {
        self.fallback_policy = fallback_policy;
        self
    }
}

impl Default for TassadarPlannerRoutingPolicy {
    fn default() -> Self {
        Self::exact_executor_default()
    }
}

/// Planner-owned exact-computation subproblem routed into Tassadar.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPlannerExecutorSubproblem {
    /// Stable subproblem identifier.
    pub subproblem_id: String,
    /// Human-readable planner objective for the exact executor call.
    pub objective: String,
    /// Optional explicit executor model id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_model_id: Option<String>,
    /// Digest-bound program artifact submitted to the executor.
    pub program_artifact: TassadarProgramArtifact,
    /// Requested decode mode for the exact executor path.
    pub requested_decode_mode: TassadarExecutorDecodeMode,
    /// Ordered environment refs carried into executor lineage.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub environment_refs: Vec<String>,
}

impl TassadarPlannerExecutorSubproblem {
    /// Creates one exact-computation subproblem.
    #[must_use]
    pub fn new(
        subproblem_id: impl Into<String>,
        objective: impl Into<String>,
        program_artifact: TassadarProgramArtifact,
        requested_decode_mode: TassadarExecutorDecodeMode,
    ) -> Self {
        Self {
            subproblem_id: subproblem_id.into(),
            objective: objective.into(),
            requested_model_id: None,
            program_artifact,
            requested_decode_mode,
            environment_refs: Vec::new(),
        }
    }

    /// Pins execution to one explicit executor model.
    #[must_use]
    pub fn with_requested_model_id(mut self, requested_model_id: impl Into<String>) -> Self {
        self.requested_model_id = Some(requested_model_id.into());
        self
    }

    /// Carries environment refs into the executor lineage.
    #[must_use]
    pub fn with_environment_refs(mut self, mut environment_refs: Vec<String>) -> Self {
        environment_refs.sort();
        environment_refs.dedup();
        self.environment_refs = environment_refs;
        self
    }
}

/// Stable planner-owned request for exact executor delegation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPlannerRoutingRequest {
    /// Stable planner request identifier.
    pub request_id: String,
    /// Product identifier. Must be `psionic.planner_executor_route`.
    pub product_id: String,
    /// Stable planner session identifier.
    pub planner_session_id: String,
    /// Planner model identifier making the exact-computation request.
    pub planner_model_id: String,
    /// Exact-computation subproblem to delegate.
    pub subproblem: TassadarPlannerExecutorSubproblem,
    /// Planner routing policy.
    pub routing_policy: TassadarPlannerRoutingPolicy,
    /// Planner routing budget.
    pub routing_budget: TassadarPlannerRoutingBudget,
}

impl TassadarPlannerRoutingRequest {
    /// Creates one planner-owned exact routing request.
    #[must_use]
    pub fn new(
        request_id: impl Into<String>,
        planner_session_id: impl Into<String>,
        planner_model_id: impl Into<String>,
        subproblem: TassadarPlannerExecutorSubproblem,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            product_id: String::from(PLANNER_EXECUTOR_ROUTE_PRODUCT_ID),
            planner_session_id: planner_session_id.into(),
            planner_model_id: planner_model_id.into(),
            subproblem,
            routing_policy: TassadarPlannerRoutingPolicy::default(),
            routing_budget: TassadarPlannerRoutingBudget::default(),
        }
    }

    /// Replaces the routing policy.
    #[must_use]
    pub fn with_routing_policy(mut self, routing_policy: TassadarPlannerRoutingPolicy) -> Self {
        self.routing_policy = routing_policy;
        self
    }

    /// Replaces the routing budget.
    #[must_use]
    pub fn with_routing_budget(mut self, routing_budget: TassadarPlannerRoutingBudget) -> Self {
        self.routing_budget = routing_budget;
        self
    }

    /// Returns a stable digest over the planner routing request.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        stable_digest(b"tassadar_planner_routing_request|", self)
    }
}

/// Planner-visible route state after policy and executor resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarPlannerRouteState {
    /// The request delegated successfully into Tassadar.
    Delegated,
    /// The planner received an explicit typed fallback.
    PlannerFallback,
    /// The routing contract refused the request.
    Refused,
}

/// Planner-visible reason for a route decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarPlannerRouteReason {
    /// Planner policy disabled exact executor delegation.
    PlannerPolicyDisabled,
    /// Validated-program length exceeded planner budget.
    ProgramLengthBudgetExceeded,
    /// Profile-backed trace-step budget exceeded planner budget.
    TraceStepBudgetExceeded,
    /// Environment refs exceeded planner budget.
    EnvironmentRefBudgetExceeded,
    /// Planner policy disallowed runtime decode fallback.
    ExecutorDecodeFallbackDisallowed,
    /// Planner policy required a direct decode path.
    ExecutorDirectPathRequired,
    /// Executor model/program contract was invalid.
    ExecutorContractRejected,
    /// Executor selection refused the request before execution.
    ExecutorSelectionRefused,
    /// Executor service rejected the request before delegation.
    ExecutorServiceRejected,
}

/// Replay-stable planner routing decision with executor truth attached.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPlannerRoutingDecision {
    /// Stable digest for the planner request.
    pub planner_request_digest: String,
    /// Stable digest for the routing decision itself.
    pub routing_digest: String,
    /// Stable planner request identifier.
    pub planner_request_id: String,
    /// Stable planner session identifier.
    pub planner_session_id: String,
    /// Planner model identifier that requested the route.
    pub planner_model_id: String,
    /// Planner-owned product identifier.
    pub planner_product_id: String,
    /// Executor-trace product delegated to by the router.
    pub executor_product_id: String,
    /// Stable subproblem identifier.
    pub subproblem_id: String,
    /// Requested decode mode.
    pub requested_decode_mode: TassadarExecutorDecodeMode,
    /// Effective decode mode after runtime selection, when execution remained viable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_decode_mode: Option<TassadarExecutorDecodeMode>,
    /// Stable digest for the executor request when one exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executor_request_digest: Option<String>,
    /// Route state exposed back to the planner.
    pub route_state: TassadarPlannerRouteState,
    /// Route reason when routing did not delegate directly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_reason: Option<TassadarPlannerRouteReason>,
    /// Budget policy consulted during route selection.
    pub routing_budget: TassadarPlannerRoutingBudget,
    /// Fallback and decode policy consulted during route selection.
    pub routing_policy: TassadarPlannerRoutingPolicy,
    /// Runtime capability report preserved across the planner boundary.
    pub runtime_capability: TassadarRuntimeCapabilityReport,
    /// Executor selection diagnostic preserved across the planner boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<TassadarExecutorSelectionDiagnostic>,
    /// Contract error preserved when model/program pairing failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract_error: Option<TassadarExecutorContractError>,
    /// Human-readable route summary safe for logs or UI.
    pub detail: String,
}

impl TassadarPlannerRoutingDecision {
    fn new(
        request: &TassadarPlannerRoutingRequest,
        executor_request_digest: Option<String>,
        runtime_capability: TassadarRuntimeCapabilityReport,
        selection: Option<TassadarExecutorSelectionDiagnostic>,
        contract_error: Option<TassadarExecutorContractError>,
        route_state: TassadarPlannerRouteState,
        route_reason: Option<TassadarPlannerRouteReason>,
        detail: String,
    ) -> Self {
        #[derive(Serialize)]
        struct RoutingDigestInput<'a> {
            planner_request_digest: &'a str,
            planner_request_id: &'a str,
            planner_session_id: &'a str,
            planner_model_id: &'a str,
            planner_product_id: &'a str,
            subproblem_id: &'a str,
            requested_decode_mode: TassadarExecutorDecodeMode,
            effective_decode_mode: Option<TassadarExecutorDecodeMode>,
            executor_request_digest: Option<&'a str>,
            route_state: TassadarPlannerRouteState,
            route_reason: Option<TassadarPlannerRouteReason>,
            routing_budget: &'a TassadarPlannerRoutingBudget,
            routing_policy: &'a TassadarPlannerRoutingPolicy,
            runtime_capability: &'a TassadarRuntimeCapabilityReport,
            selection: &'a Option<TassadarExecutorSelectionDiagnostic>,
            contract_error: &'a Option<TassadarExecutorContractError>,
            detail: &'a str,
        }

        let planner_request_digest = request.stable_digest();
        let effective_decode_mode = selection
            .as_ref()
            .and_then(|value| value.effective_decode_mode);
        let routing_digest = stable_digest(
            b"tassadar_planner_routing_decision|",
            &RoutingDigestInput {
                planner_request_digest: planner_request_digest.as_str(),
                planner_request_id: request.request_id.as_str(),
                planner_session_id: request.planner_session_id.as_str(),
                planner_model_id: request.planner_model_id.as_str(),
                planner_product_id: request.product_id.as_str(),
                subproblem_id: request.subproblem.subproblem_id.as_str(),
                requested_decode_mode: request.subproblem.requested_decode_mode,
                effective_decode_mode,
                executor_request_digest: executor_request_digest.as_deref(),
                route_state,
                route_reason,
                routing_budget: &request.routing_budget,
                routing_policy: &request.routing_policy,
                runtime_capability: &runtime_capability,
                selection: &selection,
                contract_error: &contract_error,
                detail: detail.as_str(),
            },
        );
        Self {
            planner_request_digest,
            routing_digest,
            planner_request_id: request.request_id.clone(),
            planner_session_id: request.planner_session_id.clone(),
            planner_model_id: request.planner_model_id.clone(),
            planner_product_id: request.product_id.clone(),
            executor_product_id: String::from(EXECUTOR_TRACE_PRODUCT_ID),
            subproblem_id: request.subproblem.subproblem_id.clone(),
            requested_decode_mode: request.subproblem.requested_decode_mode,
            effective_decode_mode,
            executor_request_digest,
            route_state,
            route_reason,
            routing_budget: request.routing_budget.clone(),
            routing_policy: request.routing_policy.clone(),
            runtime_capability,
            selection,
            contract_error,
            detail,
        }
    }
}

/// Planner-visible successful exact delegation result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPlannerCompletedResponse {
    /// Replay-stable routing decision.
    pub routing_decision: TassadarPlannerRoutingDecision,
    /// Completed exact executor response with proof-bearing evidence.
    pub executor_response: TassadarExecutorResponse,
}

/// Planner-visible typed fallback preserving executor truth.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPlannerFallbackResponse {
    /// Replay-stable routing decision.
    pub routing_decision: TassadarPlannerRoutingDecision,
    /// Human-readable planner fallback summary.
    pub fallback_summary: String,
    /// Executor refusal preserved when one existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executor_refusal: Option<TassadarExecutorRefusalResponse>,
}

/// Planner-visible typed refusal preserving executor truth.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPlannerRefusalResponse {
    /// Replay-stable routing decision.
    pub routing_decision: TassadarPlannerRoutingDecision,
    /// Human-readable refusal detail.
    pub detail: String,
    /// Executor refusal preserved when one existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executor_refusal: Option<TassadarExecutorRefusalResponse>,
}

/// Planner-visible outcome for one exact executor routing request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TassadarPlannerRoutingOutcome {
    /// The planner delegated successfully into Tassadar.
    Completed {
        /// Completed exact routing response.
        response: TassadarPlannerCompletedResponse,
    },
    /// The planner received an explicit typed fallback instead of exact execution.
    Fallback {
        /// Typed fallback response.
        fallback: TassadarPlannerFallbackResponse,
    },
    /// The routing contract refused the request.
    Refused {
        /// Typed refusal response.
        refusal: TassadarPlannerRefusalResponse,
    },
}

/// Planner-owned request validation errors for hybrid exact routing.
#[derive(Clone, Debug, Error, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TassadarPlannerRouterError {
    /// The request targeted a different planner-owned product family.
    #[error("unsupported Tassadar planner routing product `{product_id}`")]
    UnsupportedProduct {
        /// Product identifier supplied by the caller.
        product_id: String,
    },
}

/// Local planner router that delegates exact subproblems into Tassadar.
#[derive(Clone, Debug, Default)]
pub struct LocalTassadarPlannerRouter {
    executor_service: LocalTassadarExecutorService,
}

impl LocalTassadarPlannerRouter {
    /// Creates the default local planner router.
    #[must_use]
    pub fn new() -> Self {
        Self {
            executor_service: LocalTassadarExecutorService::new(),
        }
    }

    /// Replaces the backing local executor service.
    #[must_use]
    pub fn with_executor_service(mut self, executor_service: LocalTassadarExecutorService) -> Self {
        self.executor_service = executor_service;
        self
    }

    /// Routes one planner-owned exact subproblem into Tassadar when policy allows.
    pub fn route(
        &self,
        request: &TassadarPlannerRoutingRequest,
    ) -> Result<TassadarPlannerRoutingOutcome, TassadarPlannerRouterError> {
        self.validate_product(request)?;
        let executor_request = self.executor_request_for(request);
        let executor_request_digest = executor_request.stable_digest();
        let preflight = match self.executor_service.preflight(&executor_request) {
            Ok(preflight) => preflight,
            Err(TassadarExecutorServiceError::UnknownModel { model_id }) => {
                let capability = TassadarRuntimeCapabilityReport::current();
                let detail =
                    format!("planner requested unknown Tassadar executor model `{model_id}`");
                return Ok(self.policy_terminal_outcome(
                    request,
                    Some(executor_request_digest),
                    capability,
                    None,
                    None,
                    TassadarPlannerRouteReason::ExecutorServiceRejected,
                    detail,
                    None,
                ));
            }
            Err(TassadarExecutorServiceError::UnsupportedProduct { product_id }) => {
                let capability = TassadarRuntimeCapabilityReport::current();
                let detail = format!(
                    "planner delegated to unsupported Tassadar executor product `{product_id}`"
                );
                return Ok(self.policy_terminal_outcome(
                    request,
                    Some(executor_request_digest),
                    capability,
                    None,
                    None,
                    TassadarPlannerRouteReason::ExecutorServiceRejected,
                    detail,
                    None,
                ));
            }
            Err(TassadarExecutorServiceError::ExecutionRefusal(refusal)) => {
                let capability = TassadarRuntimeCapabilityReport::current();
                let detail = format!(
                    "executor service rejected planner request before delegation: {refusal}"
                );
                return Ok(self.policy_terminal_outcome(
                    request,
                    Some(executor_request_digest),
                    capability,
                    None,
                    None,
                    TassadarPlannerRouteReason::ExecutorServiceRejected,
                    detail,
                    None,
                ));
            }
        };

        if !request.routing_policy.allow_executor_delegation {
            return Ok(self.policy_terminal_outcome(
                request,
                Some(executor_request_digest),
                preflight.runtime_capability,
                preflight.selection,
                preflight.contract_error,
                TassadarPlannerRouteReason::PlannerPolicyDisabled,
                String::from("planner policy disabled exact Tassadar delegation"),
                None,
            ));
        }

        let program_len = request
            .subproblem
            .program_artifact
            .validated_program
            .instructions
            .len();
        if program_len > request.routing_budget.max_program_len {
            return Ok(self.policy_terminal_outcome(
                request,
                Some(executor_request_digest),
                preflight.runtime_capability,
                preflight.selection,
                preflight.contract_error,
                TassadarPlannerRouteReason::ProgramLengthBudgetExceeded,
                format!(
                    "validated program uses {} instructions which exceeds planner budget {}",
                    program_len, request.routing_budget.max_program_len
                ),
                None,
            ));
        }

        let conservative_trace_steps =
            route_trace_step_budget(&request.subproblem.program_artifact);
        if conservative_trace_steps > request.routing_budget.max_trace_steps {
            return Ok(self.policy_terminal_outcome(
                request,
                Some(executor_request_digest),
                preflight.runtime_capability,
                preflight.selection,
                preflight.contract_error,
                TassadarPlannerRouteReason::TraceStepBudgetExceeded,
                format!(
                    "profile-backed trace-step budget {} exceeds planner budget {}",
                    conservative_trace_steps, request.routing_budget.max_trace_steps
                ),
                None,
            ));
        }

        if request.subproblem.environment_refs.len() > request.routing_budget.max_environment_refs {
            return Ok(self.policy_terminal_outcome(
                request,
                Some(executor_request_digest),
                preflight.runtime_capability,
                preflight.selection,
                preflight.contract_error,
                TassadarPlannerRouteReason::EnvironmentRefBudgetExceeded,
                format!(
                    "environment ref count {} exceeds planner budget {}",
                    request.subproblem.environment_refs.len(),
                    request.routing_budget.max_environment_refs
                ),
                None,
            ));
        }

        if let Some(contract_error) = preflight.contract_error {
            return Ok(self.policy_terminal_outcome(
                request,
                Some(executor_request_digest),
                preflight.runtime_capability,
                preflight.selection,
                Some(contract_error.clone()),
                TassadarPlannerRouteReason::ExecutorContractRejected,
                contract_error.to_string(),
                None,
            ));
        }

        let selection = preflight
            .selection
            .expect("preflight should include selection when contract validation succeeds");
        if selection.effective_decode_mode.is_none() {
            return Ok(self.policy_terminal_outcome(
                request,
                Some(executor_request_digest),
                preflight.runtime_capability,
                Some(selection.clone()),
                None,
                TassadarPlannerRouteReason::ExecutorSelectionRefused,
                selection.detail.clone(),
                None,
            ));
        }
        if !request.routing_policy.allow_runtime_decode_fallback && selection.is_fallback() {
            return Ok(self.policy_terminal_outcome(
                request,
                Some(executor_request_digest),
                preflight.runtime_capability,
                Some(selection.clone()),
                None,
                TassadarPlannerRouteReason::ExecutorDecodeFallbackDisallowed,
                format!(
                    "planner policy disallowed runtime decode fallback: {}",
                    selection.detail
                ),
                None,
            ));
        }
        if request.routing_policy.require_direct_decode && selection.is_fallback() {
            return Ok(self.policy_terminal_outcome(
                request,
                Some(executor_request_digest),
                preflight.runtime_capability,
                Some(selection.clone()),
                None,
                TassadarPlannerRouteReason::ExecutorDirectPathRequired,
                format!(
                    "planner policy required a direct decode path: {}",
                    selection.detail
                ),
                None,
            ));
        }

        match self.executor_service.execute(&executor_request) {
            Ok(TassadarExecutorOutcome::Completed { response }) => {
                if request.routing_policy.require_proof_bundle
                    && response.evidence_bundle.proof_bundle.product_id != EXECUTOR_TRACE_PRODUCT_ID
                {
                    let capability = response.runtime_capability.clone();
                    let selection = Some(response.execution_report.selection.clone());
                    return Ok(self.policy_terminal_outcome(
                        request,
                        Some(executor_request_digest),
                        capability,
                        selection,
                        None,
                        TassadarPlannerRouteReason::ExecutorServiceRejected,
                        String::from(
                            "completed executor response was missing the required proof-bearing product identity",
                        ),
                        None,
                    ));
                }
                let detail = format!(
                    "planner delegated exact subproblem `{}` into Tassadar via `{}`",
                    request.subproblem.subproblem_id, EXECUTOR_TRACE_PRODUCT_ID
                );
                let routing_decision = TassadarPlannerRoutingDecision::new(
                    request,
                    Some(executor_request_digest),
                    response.runtime_capability.clone(),
                    Some(response.execution_report.selection.clone()),
                    None,
                    TassadarPlannerRouteState::Delegated,
                    None,
                    detail,
                );
                Ok(TassadarPlannerRoutingOutcome::Completed {
                    response: TassadarPlannerCompletedResponse {
                        routing_decision,
                        executor_response: response,
                    },
                })
            }
            Ok(TassadarExecutorOutcome::Refused { refusal }) => Ok(self.policy_terminal_outcome(
                request,
                Some(executor_request_digest),
                refusal.runtime_capability.clone(),
                refusal.selection.clone(),
                refusal.contract_error.clone(),
                TassadarPlannerRouteReason::ExecutorSelectionRefused,
                refusal.detail.clone(),
                Some(refusal),
            )),
            Err(TassadarExecutorServiceError::UnknownModel { model_id }) => {
                let capability = TassadarRuntimeCapabilityReport::current();
                let detail =
                    format!("planner requested unknown Tassadar executor model `{model_id}`");
                Ok(self.policy_terminal_outcome(
                    request,
                    Some(executor_request_digest),
                    capability,
                    None,
                    None,
                    TassadarPlannerRouteReason::ExecutorServiceRejected,
                    detail,
                    None,
                ))
            }
            Err(TassadarExecutorServiceError::UnsupportedProduct { product_id }) => {
                let capability = TassadarRuntimeCapabilityReport::current();
                let detail = format!(
                    "planner delegated to unsupported Tassadar executor product `{product_id}`"
                );
                Ok(self.policy_terminal_outcome(
                    request,
                    Some(executor_request_digest),
                    capability,
                    None,
                    None,
                    TassadarPlannerRouteReason::ExecutorServiceRejected,
                    detail,
                    None,
                ))
            }
            Err(TassadarExecutorServiceError::ExecutionRefusal(refusal)) => {
                let capability = TassadarRuntimeCapabilityReport::current();
                let detail = format!(
                    "executor service rejected planner request before delegation: {refusal}"
                );
                Ok(self.policy_terminal_outcome(
                    request,
                    Some(executor_request_digest),
                    capability,
                    None,
                    None,
                    TassadarPlannerRouteReason::ExecutorServiceRejected,
                    detail,
                    None,
                ))
            }
        }
    }

    fn validate_product(
        &self,
        request: &TassadarPlannerRoutingRequest,
    ) -> Result<(), TassadarPlannerRouterError> {
        if request.product_id == PLANNER_EXECUTOR_ROUTE_PRODUCT_ID {
            Ok(())
        } else {
            Err(TassadarPlannerRouterError::UnsupportedProduct {
                product_id: request.product_id.clone(),
            })
        }
    }

    fn executor_request_for(
        &self,
        request: &TassadarPlannerRoutingRequest,
    ) -> TassadarExecutorRequest {
        let mut executor_request = TassadarExecutorRequest::new(
            format!(
                "{}::{}",
                request.request_id, request.subproblem.subproblem_id
            ),
            request.subproblem.program_artifact.clone(),
            request.subproblem.requested_decode_mode,
        )
        .with_environment_refs(request.subproblem.environment_refs.clone());
        if let Some(requested_model_id) = request.subproblem.requested_model_id.as_deref() {
            executor_request = executor_request.with_requested_model_id(requested_model_id);
        }
        executor_request
    }

    fn policy_terminal_outcome(
        &self,
        request: &TassadarPlannerRoutingRequest,
        executor_request_digest: Option<String>,
        runtime_capability: TassadarRuntimeCapabilityReport,
        selection: Option<TassadarExecutorSelectionDiagnostic>,
        contract_error: Option<TassadarExecutorContractError>,
        route_reason: TassadarPlannerRouteReason,
        detail: String,
        executor_refusal: Option<TassadarExecutorRefusalResponse>,
    ) -> TassadarPlannerRoutingOutcome {
        let route_state = match request.routing_policy.fallback_policy {
            TassadarPlannerFallbackPolicy::Refuse => TassadarPlannerRouteState::Refused,
            TassadarPlannerFallbackPolicy::PlannerSummary => {
                TassadarPlannerRouteState::PlannerFallback
            }
        };
        let routing_decision = TassadarPlannerRoutingDecision::new(
            request,
            executor_request_digest,
            runtime_capability,
            selection,
            contract_error,
            route_state,
            Some(route_reason),
            detail.clone(),
        );
        match request.routing_policy.fallback_policy {
            TassadarPlannerFallbackPolicy::Refuse => TassadarPlannerRoutingOutcome::Refused {
                refusal: TassadarPlannerRefusalResponse {
                    routing_decision,
                    detail,
                    executor_refusal,
                },
            },
            TassadarPlannerFallbackPolicy::PlannerSummary => {
                TassadarPlannerRoutingOutcome::Fallback {
                    fallback: TassadarPlannerFallbackResponse {
                        routing_decision,
                        fallback_summary: detail,
                        executor_refusal,
                    },
                }
            }
        }
    }
}

/// Specialized planner-owned request for one article-class exact workflow.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarArticleHybridWorkflowRequest {
    /// Stable workflow request identifier.
    pub request_id: String,
    /// Product identifier. Must be `psionic.article_hybrid_workflow`.
    pub product_id: String,
    /// Stable planner session identifier.
    pub planner_session_id: String,
    /// Planner model identifier initiating the workflow.
    pub planner_model_id: String,
    /// Stable workflow step identifier.
    pub workflow_step_id: String,
    /// Human-readable planner objective for the exact compute span.
    pub objective: String,
    /// Stable canonical article-case identifier.
    pub article_case_id: String,
    /// Optional explicit executor model id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_model_id: Option<String>,
    /// Requested decode mode for the exact executor path.
    pub requested_decode_mode: TassadarExecutorDecodeMode,
    /// Planner routing policy for the exact compute span.
    pub routing_policy: TassadarPlannerRoutingPolicy,
    /// Planner routing budget for the exact compute span.
    pub routing_budget: TassadarPlannerRoutingBudget,
    /// Ordered environment refs carried into executor lineage in addition to the canonical benchmark ref.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub environment_refs: Vec<String>,
}

impl TassadarArticleHybridWorkflowRequest {
    /// Creates one article-class hybrid workflow request.
    #[must_use]
    pub fn new(
        request_id: impl Into<String>,
        planner_session_id: impl Into<String>,
        planner_model_id: impl Into<String>,
        workflow_step_id: impl Into<String>,
        objective: impl Into<String>,
        article_case_id: impl Into<String>,
        requested_decode_mode: TassadarExecutorDecodeMode,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            product_id: String::from(ARTICLE_HYBRID_WORKFLOW_PRODUCT_ID),
            planner_session_id: planner_session_id.into(),
            planner_model_id: planner_model_id.into(),
            workflow_step_id: workflow_step_id.into(),
            objective: objective.into(),
            article_case_id: article_case_id.into(),
            requested_model_id: None,
            requested_decode_mode,
            routing_policy: TassadarPlannerRoutingPolicy::default(),
            routing_budget: TassadarPlannerRoutingBudget::new(4_096, 131_072, 8),
            environment_refs: Vec::new(),
        }
    }

    /// Pins execution to one explicit executor model.
    #[must_use]
    pub fn with_requested_model_id(mut self, requested_model_id: impl Into<String>) -> Self {
        self.requested_model_id = Some(requested_model_id.into());
        self
    }

    /// Replaces the routing policy.
    #[must_use]
    pub fn with_routing_policy(mut self, routing_policy: TassadarPlannerRoutingPolicy) -> Self {
        self.routing_policy = routing_policy;
        self
    }

    /// Replaces the routing budget.
    #[must_use]
    pub fn with_routing_budget(mut self, routing_budget: TassadarPlannerRoutingBudget) -> Self {
        self.routing_budget = routing_budget;
        self
    }

    /// Carries extra environment refs into the executor lineage.
    #[must_use]
    pub fn with_environment_refs(mut self, mut environment_refs: Vec<String>) -> Self {
        environment_refs.sort();
        environment_refs.dedup();
        self.environment_refs = environment_refs;
        self
    }

    /// Returns a stable digest over the hybrid workflow request.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        stable_digest(b"tassadar_article_hybrid_workflow_request|", self)
    }
}

/// Completed article-class hybrid workflow response.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarArticleHybridWorkflowCompletedResponse {
    /// Stable workflow request identifier.
    pub request_id: String,
    /// Product identifier.
    pub product_id: String,
    /// Canonical benchmark/workload identity.
    pub benchmark_identity: TassadarArticleBenchmarkIdentity,
    /// Proof identity preserved across the workflow boundary.
    pub proof_identity: TassadarArticleProofIdentity,
    /// Underlying planner-owned completed response.
    pub planner_response: TassadarPlannerCompletedResponse,
}

/// Fallback article-class hybrid workflow response.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarArticleHybridWorkflowFallbackResponse {
    /// Stable workflow request identifier.
    pub request_id: String,
    /// Product identifier.
    pub product_id: String,
    /// Canonical benchmark/workload identity when the case was resolved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub benchmark_identity: Option<TassadarArticleBenchmarkIdentity>,
    /// Underlying planner fallback response.
    pub planner_fallback: TassadarPlannerFallbackResponse,
}

/// Refusal article-class hybrid workflow response.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarArticleHybridWorkflowRefusalResponse {
    /// Stable workflow request identifier.
    pub request_id: String,
    /// Product identifier.
    pub product_id: String,
    /// Canonical benchmark/workload identity when the case was resolved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub benchmark_identity: Option<TassadarArticleBenchmarkIdentity>,
    /// Underlying planner refusal response when routing reached the planner layer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner_refusal: Option<TassadarPlannerRefusalResponse>,
    /// Human-readable refusal detail.
    pub detail: String,
}

/// Outcome for one article-class hybrid workflow request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TassadarArticleHybridWorkflowOutcome {
    /// The workflow delegated successfully into the exact executor lane.
    Completed {
        /// Completed workflow response.
        response: TassadarArticleHybridWorkflowCompletedResponse,
    },
    /// The workflow returned a typed planner fallback.
    Fallback {
        /// Typed workflow fallback.
        fallback: TassadarArticleHybridWorkflowFallbackResponse,
    },
    /// The workflow refused the request explicitly.
    Refused {
        /// Typed workflow refusal.
        refusal: TassadarArticleHybridWorkflowRefusalResponse,
    },
}

/// Product-validation error for the article hybrid-workflow surface.
#[derive(Clone, Debug, Error, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TassadarArticleHybridWorkflowServiceError {
    /// The request targeted a different product family.
    #[error("unsupported Tassadar article hybrid-workflow product `{product_id}`")]
    UnsupportedProduct {
        /// Product identifier supplied by the caller.
        product_id: String,
    },
}

/// Local article-class hybrid workflow service above the generic planner route.
#[derive(Clone, Debug)]
pub struct LocalTassadarArticleHybridWorkflowService {
    planner_router: LocalTassadarPlannerRouter,
}

impl Default for LocalTassadarArticleHybridWorkflowService {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalTassadarArticleHybridWorkflowService {
    /// Creates the default local article hybrid-workflow service.
    #[must_use]
    pub fn new() -> Self {
        let executor_service = LocalTassadarExecutorService::new()
            .with_fixture(TassadarExecutorFixture::article_i32_compute_v1());
        Self {
            planner_router: LocalTassadarPlannerRouter::new()
                .with_executor_service(executor_service),
        }
    }

    /// Replaces the backing planner router.
    #[must_use]
    pub fn with_planner_router(mut self, planner_router: LocalTassadarPlannerRouter) -> Self {
        self.planner_router = planner_router;
        self
    }

    /// Executes one article-class hybrid workflow.
    pub fn execute(
        &self,
        request: &TassadarArticleHybridWorkflowRequest,
    ) -> Result<TassadarArticleHybridWorkflowOutcome, TassadarArticleHybridWorkflowServiceError>
    {
        self.validate_product(request)?;
        let Some(case) = article_case_by_id(request.article_case_id.as_str()) else {
            return Ok(TassadarArticleHybridWorkflowOutcome::Refused {
                refusal: TassadarArticleHybridWorkflowRefusalResponse {
                    request_id: request.request_id.clone(),
                    product_id: request.product_id.clone(),
                    benchmark_identity: None,
                    planner_refusal: None,
                    detail: format!(
                        "article workflow case `{}` is not present in the canonical Tassadar article corpus",
                        request.article_case_id
                    ),
                },
            });
        };

        let planner_request = self.planner_request_for(request, &case);
        let benchmark_identity =
            TassadarArticleBenchmarkIdentity::for_case(&case, &planner_request.subproblem.program_artifact);
        match self.planner_router.route(&planner_request) {
            Ok(TassadarPlannerRoutingOutcome::Completed { response }) => {
                let proof_identity =
                    TassadarArticleProofIdentity::from_response(&response.executor_response);
                Ok(TassadarArticleHybridWorkflowOutcome::Completed {
                    response: TassadarArticleHybridWorkflowCompletedResponse {
                        request_id: request.request_id.clone(),
                        product_id: request.product_id.clone(),
                        benchmark_identity,
                        proof_identity,
                        planner_response: response,
                    },
                })
            }
            Ok(TassadarPlannerRoutingOutcome::Fallback { fallback }) => {
                Ok(TassadarArticleHybridWorkflowOutcome::Fallback {
                    fallback: TassadarArticleHybridWorkflowFallbackResponse {
                        request_id: request.request_id.clone(),
                        product_id: request.product_id.clone(),
                        benchmark_identity: Some(benchmark_identity),
                        planner_fallback: fallback,
                    },
                })
            }
            Ok(TassadarPlannerRoutingOutcome::Refused { refusal }) => {
                Ok(TassadarArticleHybridWorkflowOutcome::Refused {
                    refusal: TassadarArticleHybridWorkflowRefusalResponse {
                        request_id: request.request_id.clone(),
                        product_id: request.product_id.clone(),
                        benchmark_identity: Some(benchmark_identity),
                        detail: refusal.detail.clone(),
                        planner_refusal: Some(refusal),
                    },
                })
            }
            Err(TassadarPlannerRouterError::UnsupportedProduct { product_id }) => Err(
                TassadarArticleHybridWorkflowServiceError::UnsupportedProduct { product_id },
            ),
        }
    }

    fn validate_product(
        &self,
        request: &TassadarArticleHybridWorkflowRequest,
    ) -> Result<(), TassadarArticleHybridWorkflowServiceError> {
        if request.product_id == ARTICLE_HYBRID_WORKFLOW_PRODUCT_ID {
            Ok(())
        } else {
            Err(TassadarArticleHybridWorkflowServiceError::UnsupportedProduct {
                product_id: request.product_id.clone(),
            })
        }
    }

    fn planner_request_for(
        &self,
        request: &TassadarArticleHybridWorkflowRequest,
        case: &TassadarValidationCase,
    ) -> TassadarPlannerRoutingRequest {
        let profile = tassadar_wasm_profile_for_id(case.program.profile_id.as_str())
            .expect("canonical article case should use a supported profile");
        let trace_abi = tassadar_trace_abi_for_profile_id(case.program.profile_id.as_str())
            .expect("canonical article case should use a supported trace ABI");
        let artifact = TassadarProgramArtifact::fixture_reference(
            format!("artifact://tassadar/article_workflow/{}", case.case_id),
            &profile,
            &trace_abi,
            case.program.clone(),
        )
        .expect("canonical article case should assemble into a program artifact");
        let mut subproblem = TassadarPlannerExecutorSubproblem::new(
            request.workflow_step_id.clone(),
            request.objective.clone(),
            artifact,
            request.requested_decode_mode,
        )
        .with_environment_refs(merge_article_environment_refs(
            request.environment_refs.as_slice(),
        ));
        if let Some(requested_model_id) = request.requested_model_id.as_deref() {
            subproblem = subproblem.with_requested_model_id(requested_model_id);
        } else {
            subproblem = subproblem
                .with_requested_model_id(TassadarExecutorFixture::ARTICLE_I32_COMPUTE_MODEL_ID);
        }
        TassadarPlannerRoutingRequest::new(
            request.request_id.clone(),
            request.planner_session_id.clone(),
            request.planner_model_id.clone(),
            subproblem,
        )
        .with_routing_policy(request.routing_policy.clone())
        .with_routing_budget(request.routing_budget.clone())
    }
}

fn route_trace_step_budget(program_artifact: &TassadarProgramArtifact) -> usize {
    tassadar_wasm_profile_for_id(program_artifact.wasm_profile_id.as_str()).map_or_else(
        || program_artifact.validated_program.instructions.len(),
        |profile| profile.max_steps,
    )
}

fn stable_digest<T>(prefix: &[u8], value: &T) -> String
where
    T: Serialize,
{
    let encoded =
        serde_json::to_vec(value).expect("Tassadar planner routing value should serialize");
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(encoded);
    hex::encode(hasher.finalize())
}

fn article_case_by_id(case_id: &str) -> Option<TassadarValidationCase> {
    tassadar_article_class_corpus()
        .into_iter()
        .find(|case| case.case_id == case_id)
}

fn article_workload_family(case_id: &str) -> &'static str {
    match case_id {
        "micro_wasm_kernel" => "MicroWasmKernel",
        "branch_heavy_kernel" => "BranchHeavyKernel",
        "memory_heavy_kernel" => "MemoryHeavyKernel",
        "long_loop_kernel" => "LongLoopKernel",
        value if value.starts_with("sudoku_") => "SudokuClass",
        value if value.starts_with("hungarian_") => "HungarianMatching",
        _ => "ArticleWorkload",
    }
}

fn merge_article_environment_refs(environment_refs: &[String]) -> Vec<String> {
    let mut merged = environment_refs.to_vec();
    merged.push(String::from(TASSADAR_ARTICLE_CLASS_BENCHMARK_ENVIRONMENT_REF));
    merged.sort();
    merged.dedup();
    merged
}

fn build_article_readable_log_excerpt(
    case: &TassadarValidationCase,
    execution: &TassadarExecution,
) -> TassadarArticleReadableLogExcerpt {
    let mut lines = Vec::with_capacity(execution.steps.len().saturating_add(8));
    lines.push(format!(
        "benchmark_ref={}",
        TASSADAR_ARTICLE_CLASS_BENCHMARK_REF
    ));
    lines.push(format!("workload_family={}", article_workload_family(case.case_id.as_str())));
    lines.push(format!("case_id={}", case.case_id));
    lines.push(format!("program_id={}", execution.program_id));
    lines.push(format!("runner_id={}", execution.runner_id));
    for step in &execution.steps {
        lines.push(format!(
            "step={} pc={}->{} instr={} event={}",
            step.step_index,
            step.pc,
            step.next_pc,
            format_instruction(&step.instruction),
            format_event(&step.event)
        ));
    }
    lines.push(format!("halt={:?}", execution.halt_reason));
    lines.push(format!("outputs={:?}", execution.outputs));

    let total_line_count = lines.len();
    let truncated = total_line_count > ARTICLE_EXECUTOR_READABLE_LOG_MAX_LINES;
    let lines = if truncated {
        let retained = ARTICLE_EXECUTOR_READABLE_LOG_MAX_LINES.saturating_sub(1);
        let mut excerpt = lines.into_iter().take(retained).collect::<Vec<_>>();
        excerpt.push(format!(
            "... truncated {} additional lines; canonical machine truth remains bound to trace_digest={}",
            total_line_count.saturating_sub(retained),
            execution.trace_digest()
        ));
        excerpt
    } else {
        lines
    };

    TassadarArticleReadableLogExcerpt {
        max_lines: ARTICLE_EXECUTOR_READABLE_LOG_MAX_LINES,
        total_line_count,
        truncated,
        lines,
        derivation_posture: String::from(
            "readable log lines are derived, non-authoritative views over the canonical append-only trace and may be truncated without changing machine truth",
        ),
    }
}

fn build_article_token_trace_excerpt(
    case: &TassadarValidationCase,
    execution: &TassadarExecution,
) -> TassadarArticleTokenTraceExcerpt {
    let tokenizer = TassadarTraceTokenizer::new();
    let tokenized = tokenizer.tokenize_program_and_execution(&case.program, execution);
    let decoded = tokenizer.decode_symbolic(&tokenized);
    let prompt_token_count = decoded.prompt_token_count;
    let total_token_count = decoded.tokens.len();
    let truncated = total_token_count > ARTICLE_EXECUTOR_TOKEN_TRACE_MAX_TOKENS;
    let tokens = if truncated {
        decoded
            .tokens
            .into_iter()
            .take(ARTICLE_EXECUTOR_TOKEN_TRACE_MAX_TOKENS)
            .collect()
    } else {
        decoded.tokens
    };
    TassadarArticleTokenTraceExcerpt {
        max_tokens: ARTICLE_EXECUTOR_TOKEN_TRACE_MAX_TOKENS,
        total_token_count,
        prompt_token_count,
        truncated,
        tokenizer_digest: tokenizer.stable_digest(),
        tokens,
        derivation_posture: String::from(
            "symbolic token traces are deterministic derivations of the canonical append-only trace artifact and may be truncated for serving without changing proof or benchmark identity",
        ),
    }
}

fn article_stream_events_for_outcome(
    outcome: &TassadarArticleExecutorSessionOutcome,
) -> Vec<TassadarArticleExecutorSessionStreamEvent> {
    let runtime_capability = match outcome {
        TassadarArticleExecutorSessionOutcome::Completed { response } => {
            response.executor_response.runtime_capability.clone()
        }
        TassadarArticleExecutorSessionOutcome::Refused { refusal } => {
            refusal.runtime_capability.clone()
        }
    };
    let mut events = Vec::new();
    match outcome {
        TassadarArticleExecutorSessionOutcome::Completed { response } => {
            events.push(TassadarArticleExecutorSessionStreamEvent::BenchmarkIdentity {
                benchmark_identity: TassadarArticleBenchmarkIdentityEvent {
                    benchmark_identity: response.benchmark_identity.clone(),
                },
            });
            events.push(TassadarArticleExecutorSessionStreamEvent::Capability { runtime_capability });
            events.push(TassadarArticleExecutorSessionStreamEvent::Selection {
                selection: response.executor_response.execution_report.selection.clone(),
            });
            events.push(TassadarArticleExecutorSessionStreamEvent::ProofIdentity {
                proof_identity: TassadarArticleProofIdentityEvent {
                    proof_identity: response.proof_identity.clone(),
                },
            });
            for (line_index, line) in response.readable_log.lines.iter().enumerate() {
                events.push(TassadarArticleExecutorSessionStreamEvent::ReadableLogLine {
                    readable_log_line: TassadarArticleReadableLogLineEvent {
                        line_index,
                        line: line.clone(),
                    },
                });
            }
            for (chunk_index, chunk) in response
                .token_trace
                .tokens
                .chunks(ARTICLE_EXECUTOR_TOKEN_TRACE_CHUNK_SIZE)
                .enumerate()
            {
                events.push(TassadarArticleExecutorSessionStreamEvent::TokenTraceChunk {
                    token_trace_chunk: TassadarArticleTokenTraceChunkEvent {
                        chunk_index,
                        prompt_token_count: response.token_trace.prompt_token_count,
                        total_token_count: response.token_trace.total_token_count,
                        truncated: response.token_trace.truncated,
                        tokens: chunk.to_vec(),
                    },
                });
            }
            for (ordinal, value) in response.final_outputs().iter().enumerate() {
                events.push(TassadarArticleExecutorSessionStreamEvent::Output {
                    output: TassadarExecutorOutputEvent {
                        ordinal,
                        value: *value,
                    },
                });
            }
        }
        TassadarArticleExecutorSessionOutcome::Refused { refusal } => {
            if let Some(benchmark_identity) = refusal.benchmark_identity.clone() {
                events.push(TassadarArticleExecutorSessionStreamEvent::BenchmarkIdentity {
                    benchmark_identity: TassadarArticleBenchmarkIdentityEvent { benchmark_identity },
                });
            }
            events.push(TassadarArticleExecutorSessionStreamEvent::Capability { runtime_capability });
            if let Some(selection) = refusal.selection.clone() {
                events.push(TassadarArticleExecutorSessionStreamEvent::Selection { selection });
            }
        }
    }
    events.push(TassadarArticleExecutorSessionStreamEvent::Terminal {
        terminal: TassadarArticleExecutorSessionTerminalEvent {
            outcome: outcome.clone(),
        },
    });
    events
}

fn format_instruction(instruction: &TassadarInstruction) -> String {
    match instruction {
        TassadarInstruction::I32Const { value } => format!("i32.const {value}"),
        TassadarInstruction::LocalGet { local } => format!("local.get {local}"),
        TassadarInstruction::LocalSet { local } => format!("local.set {local}"),
        TassadarInstruction::I32Add => String::from("i32.add"),
        TassadarInstruction::I32Sub => String::from("i32.sub"),
        TassadarInstruction::I32Mul => String::from("i32.mul"),
        TassadarInstruction::I32Lt => String::from("i32.lt"),
        TassadarInstruction::I32Load { slot } => format!("i32.load {slot}"),
        TassadarInstruction::I32Store { slot } => format!("i32.store {slot}"),
        TassadarInstruction::BrIf { target_pc } => format!("br_if {target_pc}"),
        TassadarInstruction::Output => String::from("output"),
        TassadarInstruction::Return => String::from("return"),
    }
}

fn format_event(event: &TassadarTraceEvent) -> String {
    match event {
        TassadarTraceEvent::ConstPush { value } => format!("const_push value={value}"),
        TassadarTraceEvent::LocalGet { local, value } => {
            format!("local_get local={local} value={value}")
        }
        TassadarTraceEvent::LocalSet { local, value } => {
            format!("local_set local={local} value={value}")
        }
        TassadarTraceEvent::BinaryOp {
            op,
            left,
            right,
            result,
        } => format!("binary_{op:?} left={left} right={right} result={result}"),
        TassadarTraceEvent::Load { slot, value } => format!("load slot={slot} value={value}"),
        TassadarTraceEvent::Store { slot, value } => format!("store slot={slot} value={value}"),
        TassadarTraceEvent::Branch {
            condition,
            taken,
            target_pc,
        } => format!(
            "branch condition={condition} taken={taken} target_pc={target_pc}"
        ),
        TassadarTraceEvent::Output { value } => format!("output value={value}"),
        TassadarTraceEvent::Return => String::from("return"),
    }
}

fn stream_events_for_outcome(
    fixture: &TassadarExecutorFixture,
    outcome: TassadarExecutorOutcome,
) -> Vec<TassadarExecutorStreamEvent> {
    let mut events = vec![TassadarExecutorStreamEvent::Capability {
        runtime_capability: fixture.runtime_capability_report(),
    }];
    match &outcome {
        TassadarExecutorOutcome::Completed { response } => {
            events.push(TassadarExecutorStreamEvent::Selection {
                selection: response.execution_report.selection.clone(),
            });
            let mut output_ordinal = 0usize;
            for step in &response.execution_report.execution.steps {
                events.push(TassadarExecutorStreamEvent::TraceStep {
                    trace_step: TassadarExecutorTraceStepEvent {
                        step_index: step.step_index as u64,
                        step: step.clone(),
                    },
                });
                if let TassadarTraceEvent::Output { value } = step.event {
                    events.push(TassadarExecutorStreamEvent::Output {
                        output: TassadarExecutorOutputEvent {
                            ordinal: output_ordinal,
                            value,
                        },
                    });
                    output_ordinal += 1;
                }
            }
        }
        TassadarExecutorOutcome::Refused { refusal } => {
            if let Some(selection) = &refusal.selection {
                events.push(TassadarExecutorStreamEvent::Selection {
                    selection: selection.clone(),
                });
            }
        }
    }
    events.push(TassadarExecutorStreamEvent::Terminal {
        terminal: TassadarExecutorTerminalEvent { outcome },
    });
    events
}

#[cfg(test)]
mod tests {
    use super::{
        ARTICLE_EXECUTOR_SESSION_PRODUCT_ID, ARTICLE_HYBRID_WORKFLOW_PRODUCT_ID,
        EXECUTOR_TRACE_PRODUCT_ID, LocalTassadarArticleExecutorSessionService,
        LocalTassadarArticleHybridWorkflowService, LocalTassadarExecutorService,
        LocalTassadarPlannerRouter, PLANNER_EXECUTOR_ROUTE_PRODUCT_ID,
        TassadarArticleHybridWorkflowOutcome, TassadarArticleHybridWorkflowRequest,
        TassadarArticleHybridWorkflowServiceError,
        TassadarArticleExecutorSessionOutcome, TassadarArticleExecutorSessionRequest,
        TassadarArticleExecutorSessionServiceError, TassadarArticleExecutorSessionStreamEvent,
        TassadarExecutorOutcome, TassadarExecutorRequest, TassadarExecutorServiceError,
        TassadarExecutorStreamEvent,
        TassadarPlannerExecutorSubproblem, TassadarPlannerFallbackPolicy,
        TassadarPlannerRouteReason, TassadarPlannerRouterError, TassadarPlannerRoutingBudget,
        TassadarPlannerRoutingOutcome, TassadarPlannerRoutingPolicy, TassadarPlannerRoutingRequest,
    };
    use psionic_models::TassadarExecutorFixture;
    use psionic_runtime::{
        TassadarExecutorDecodeMode, TassadarInstruction, TassadarProgram, TassadarProgramArtifact,
        TassadarTraceAbi, TassadarWasmProfile, tassadar_article_class_corpus,
        tassadar_validation_corpus,
    };

    fn request_for_case(case_id: &str) -> TassadarExecutorRequest {
        let profile = TassadarWasmProfile::core_i32_v1();
        let trace_abi = TassadarTraceAbi::core_i32_v1();
        let case = tassadar_validation_corpus()
            .into_iter()
            .find(|case| case.case_id == case_id)
            .expect("requested validation case should exist");
        let artifact = TassadarProgramArtifact::fixture_reference(
            format!("artifact://tassadar/{case_id}"),
            &profile,
            &trace_abi,
            case.program,
        )
        .expect("fixture artifact should build");
        TassadarExecutorRequest::new(
            format!("request-{case_id}"),
            artifact,
            TassadarExecutorDecodeMode::HullCache,
        )
        .with_environment_refs(vec![String::from("env.openagents.tassadar.benchmark")])
    }

    fn article_request_for_case(
        case_id: &str,
        requested_decode_mode: TassadarExecutorDecodeMode,
    ) -> TassadarArticleExecutorSessionRequest {
        TassadarArticleExecutorSessionRequest::new(
            format!("article-request-{case_id}-{requested_decode_mode:?}"),
            case_id,
            requested_decode_mode,
        )
    }

    fn article_workflow_request_for_case(
        case_id: &str,
        requested_decode_mode: TassadarExecutorDecodeMode,
    ) -> TassadarArticleHybridWorkflowRequest {
        TassadarArticleHybridWorkflowRequest::new(
            format!("article-workflow-request-{case_id}-{requested_decode_mode:?}"),
            "planner-session-article-alpha",
            "planner-article-fixture-v0",
            format!("workflow-step-{case_id}"),
            "delegate exact article workload into Tassadar",
            case_id,
            requested_decode_mode,
        )
    }

    #[test]
    fn article_session_executes_completed_request_with_benchmark_and_proof_identity() {
        let service = LocalTassadarArticleExecutorSessionService::new();
        let request = article_request_for_case(
            "memory_heavy_kernel",
            TassadarExecutorDecodeMode::HullCache,
        );
        let expected_outputs = tassadar_article_class_corpus()
            .into_iter()
            .find(|case| case.case_id == "memory_heavy_kernel")
            .expect("article case should exist")
            .expected_outputs;

        let outcome = service.execute(&request).expect("request should be typed");
        match outcome {
            TassadarArticleExecutorSessionOutcome::Completed { response } => {
                assert_eq!(response.product_id, ARTICLE_EXECUTOR_SESSION_PRODUCT_ID);
                assert_eq!(response.benchmark_identity.case_id, "memory_heavy_kernel");
                assert_eq!(
                    response.benchmark_identity.workload_family,
                    "MemoryHeavyKernel"
                );
                assert_eq!(
                    response.proof_identity.executor_product_id,
                    EXECUTOR_TRACE_PRODUCT_ID
                );
                assert_eq!(response.final_outputs(), expected_outputs.as_slice());
                assert_eq!(
                    response
                        .executor_response
                        .execution_report
                        .selection
                        .effective_decode_mode,
                    Some(TassadarExecutorDecodeMode::HullCache)
                );
                assert!(!response.readable_log.lines.is_empty());
                assert!(!response.token_trace.tokens.is_empty());
            }
            other => panic!("expected completed article session, got {other:?}"),
        }
    }

    #[test]
    fn article_session_stream_surfaces_benchmark_proof_log_trace_and_terminal() {
        let service = LocalTassadarArticleExecutorSessionService::new();
        let request = article_request_for_case(
            "memory_heavy_kernel",
            TassadarExecutorDecodeMode::HullCache,
        );
        let mut stream = service
            .execute_stream(&request)
            .expect("stream should be created");

        let mut saw_benchmark = false;
        let mut saw_proof = false;
        let mut saw_log = false;
        let mut saw_token_trace = false;
        let mut saw_terminal = false;
        while let Some(event) = stream.next_event() {
            match event {
                TassadarArticleExecutorSessionStreamEvent::BenchmarkIdentity { .. } => {
                    saw_benchmark = true
                }
                TassadarArticleExecutorSessionStreamEvent::ProofIdentity { .. } => saw_proof = true,
                TassadarArticleExecutorSessionStreamEvent::ReadableLogLine { .. } => saw_log = true,
                TassadarArticleExecutorSessionStreamEvent::TokenTraceChunk { .. } => {
                    saw_token_trace = true
                }
                TassadarArticleExecutorSessionStreamEvent::Terminal { .. } => {
                    saw_terminal = true;
                    break;
                }
                TassadarArticleExecutorSessionStreamEvent::Capability { .. }
                | TassadarArticleExecutorSessionStreamEvent::Selection { .. }
                | TassadarArticleExecutorSessionStreamEvent::Output { .. } => {}
            }
        }

        assert!(saw_benchmark);
        assert!(saw_proof);
        assert!(saw_log);
        assert!(saw_token_trace);
        assert!(saw_terminal);
    }

    #[test]
    fn article_session_preserves_fallback_truth_for_unsupported_fast_path_workloads() {
        let service = LocalTassadarArticleExecutorSessionService::new();
        let request = article_request_for_case(
            "branch_heavy_kernel",
            TassadarExecutorDecodeMode::SparseTopK,
        );

        let outcome = service.execute(&request).expect("request should be typed");
        match outcome {
            TassadarArticleExecutorSessionOutcome::Completed { response } => {
                let selection = &response.executor_response.execution_report.selection;
                assert!(selection.is_fallback());
                assert_eq!(
                    selection.effective_decode_mode,
                    Some(TassadarExecutorDecodeMode::ReferenceLinear)
                );
            }
            other => panic!("expected completed fallback article session, got {other:?}"),
        }
    }

    #[test]
    fn article_session_refuses_non_article_workloads_explicitly() {
        let service = LocalTassadarArticleExecutorSessionService::new();
        let request =
            article_request_for_case("locals_add", TassadarExecutorDecodeMode::ReferenceLinear);

        let outcome = service.execute(&request).expect("request should be typed");
        match outcome {
            TassadarArticleExecutorSessionOutcome::Refused { refusal } => {
                assert!(refusal.benchmark_identity.is_none());
                assert!(refusal.detail.contains("canonical Tassadar article corpus"));
            }
            other => panic!("expected refusal, got {other:?}"),
        }
    }

    #[test]
    fn article_session_rejects_non_article_product_id() {
        let service = LocalTassadarArticleExecutorSessionService::new();
        let mut request =
            article_request_for_case("memory_heavy_kernel", TassadarExecutorDecodeMode::HullCache);
        request.product_id = String::from("psionic.text_generation");

        let error = service
            .execute(&request)
            .expect_err("wrong article-session product should fail before execution");
        assert_eq!(
            error,
            TassadarArticleExecutorSessionServiceError::UnsupportedProduct {
                product_id: String::from("psionic.text_generation"),
            }
        );
    }

    #[test]
    fn article_hybrid_workflow_delegates_completed_request_with_routing_and_proof_identity() {
        let service = LocalTassadarArticleHybridWorkflowService::new();
        let request = article_workflow_request_for_case(
            "memory_heavy_kernel",
            TassadarExecutorDecodeMode::HullCache,
        );

        let outcome = service.execute(&request).expect("request should be typed");
        match outcome {
            TassadarArticleHybridWorkflowOutcome::Completed { response } => {
                assert_eq!(response.product_id, ARTICLE_HYBRID_WORKFLOW_PRODUCT_ID);
                assert_eq!(response.benchmark_identity.case_id, "memory_heavy_kernel");
                assert_eq!(
                    response.proof_identity.executor_product_id,
                    EXECUTOR_TRACE_PRODUCT_ID
                );
                assert_eq!(
                    response
                        .planner_response
                        .routing_decision
                        .planner_product_id,
                    PLANNER_EXECUTOR_ROUTE_PRODUCT_ID
                );
                assert_eq!(
                    response
                        .planner_response
                        .routing_decision
                        .route_state,
                    super::TassadarPlannerRouteState::Delegated
                );
            }
            other => panic!("expected completed hybrid workflow, got {other:?}"),
        }
    }

    #[test]
    fn article_hybrid_workflow_preserves_typed_planner_fallback() {
        let service = LocalTassadarArticleHybridWorkflowService::new();
        let request = article_workflow_request_for_case(
            "branch_heavy_kernel",
            TassadarExecutorDecodeMode::SparseTopK,
        )
        .with_routing_policy(
            TassadarPlannerRoutingPolicy::exact_executor_default()
                .with_fallback_policy(TassadarPlannerFallbackPolicy::PlannerSummary),
        );
        let request = TassadarArticleHybridWorkflowRequest {
            routing_policy: TassadarPlannerRoutingPolicy {
                allow_runtime_decode_fallback: false,
                ..request.routing_policy
            },
            ..request
        };

        let outcome = service.execute(&request).expect("request should be typed");
        match outcome {
            TassadarArticleHybridWorkflowOutcome::Fallback { fallback } => {
                assert_eq!(
                    fallback
                        .benchmark_identity
                        .as_ref()
                        .map(|identity| identity.case_id.as_str()),
                    Some("branch_heavy_kernel")
                );
                assert_eq!(
                    fallback.planner_fallback.routing_decision.route_reason,
                    Some(TassadarPlannerRouteReason::ExecutorDecodeFallbackDisallowed)
                );
            }
            other => panic!("expected workflow fallback, got {other:?}"),
        }
    }

    #[test]
    fn article_hybrid_workflow_preserves_typed_planner_refusal() {
        let service = LocalTassadarArticleHybridWorkflowService::new();
        let request = article_workflow_request_for_case(
            "memory_heavy_kernel",
            TassadarExecutorDecodeMode::HullCache,
        )
        .with_routing_budget(TassadarPlannerRoutingBudget::new(1, 32, 8));

        let outcome = service.execute(&request).expect("request should be typed");
        match outcome {
            TassadarArticleHybridWorkflowOutcome::Refused { refusal } => {
                assert_eq!(
                    refusal
                        .benchmark_identity
                        .as_ref()
                        .map(|identity| identity.case_id.as_str()),
                    Some("memory_heavy_kernel")
                );
                assert_eq!(
                    refusal
                        .planner_refusal
                        .as_ref()
                        .and_then(|planner_refusal| planner_refusal.routing_decision.route_reason),
                    Some(TassadarPlannerRouteReason::ProgramLengthBudgetExceeded)
                );
            }
            other => panic!("expected workflow refusal, got {other:?}"),
        }
    }

    #[test]
    fn article_hybrid_workflow_rejects_non_hybrid_product_id() {
        let service = LocalTassadarArticleHybridWorkflowService::new();
        let mut request = article_workflow_request_for_case(
            "memory_heavy_kernel",
            TassadarExecutorDecodeMode::HullCache,
        );
        request.product_id = String::from("psionic.text_generation");

        let error = service
            .execute(&request)
            .expect_err("wrong hybrid-workflow product should fail before execution");
        assert_eq!(
            error,
            TassadarArticleHybridWorkflowServiceError::UnsupportedProduct {
                product_id: String::from("psionic.text_generation"),
            }
        );
    }

    #[test]
    fn executor_service_executes_completed_request_with_explicit_product_semantics() {
        let service = LocalTassadarExecutorService::new();
        let request = request_for_case("locals_add");
        let outcome = service.execute(&request).expect("request should execute");

        match outcome {
            TassadarExecutorOutcome::Completed { response } => {
                assert_eq!(response.product_id, EXECUTOR_TRACE_PRODUCT_ID);
                assert_eq!(response.final_outputs(), &[12]);
                assert_eq!(
                    response.execution_report.selection.effective_decode_mode,
                    Some(TassadarExecutorDecodeMode::HullCache)
                );
                assert_eq!(
                    response.evidence_bundle.proof_bundle.product_id,
                    EXECUTOR_TRACE_PRODUCT_ID
                );
                assert_eq!(
                    response.trace_artifact().program_id,
                    "tassadar.locals_add.v1"
                );
            }
            TassadarExecutorOutcome::Refused { refusal } => {
                panic!("request should not be refused: {}", refusal.detail);
            }
        }
    }

    #[test]
    fn executor_service_returns_explicit_refusal_for_contract_mismatch() {
        let service = LocalTassadarExecutorService::new();
        let mut request = request_for_case("locals_add");
        request.program_artifact.trace_abi_version += 1;

        let outcome = service.execute(&request).expect("request should be typed");
        match outcome {
            TassadarExecutorOutcome::Completed { .. } => {
                panic!("mismatched ABI should not complete");
            }
            TassadarExecutorOutcome::Refused { refusal } => {
                assert!(refusal.contract_error.is_some());
                assert!(refusal.selection.is_none());
            }
        }
    }

    #[test]
    fn executor_stream_surfaces_capability_selection_trace_and_terminal() {
        let service = LocalTassadarExecutorService::new();
        let request = request_for_case("memory_roundtrip");
        let mut stream = service
            .execute_stream(&request)
            .expect("stream should be created");

        let first = stream.next_event().expect("capability event");
        assert!(matches!(
            first,
            TassadarExecutorStreamEvent::Capability { .. }
        ));
        let second = stream.next_event().expect("selection event");
        assert!(matches!(
            second,
            TassadarExecutorStreamEvent::Selection { .. }
        ));

        let mut saw_trace = false;
        let mut saw_terminal = false;
        while let Some(event) = stream.next_event() {
            match event {
                TassadarExecutorStreamEvent::TraceStep { .. } => saw_trace = true,
                TassadarExecutorStreamEvent::Terminal { .. } => {
                    saw_terminal = true;
                    break;
                }
                TassadarExecutorStreamEvent::Output { .. }
                | TassadarExecutorStreamEvent::Capability { .. }
                | TassadarExecutorStreamEvent::Selection { .. } => {}
            }
        }

        assert!(saw_trace);
        assert!(saw_terminal);
    }

    #[test]
    fn executor_service_rejects_non_executor_product_id() {
        let service = LocalTassadarExecutorService::new();
        let mut request = request_for_case("branch_guard");
        request.product_id = String::from("psionic.text_generation");

        let error = service
            .execute(&request)
            .expect_err("wrong product should fail before execution");
        assert_eq!(
            error,
            TassadarExecutorServiceError::UnsupportedProduct {
                product_id: String::from("psionic.text_generation"),
            }
        );
    }

    fn planner_request_for_case(case_id: &str) -> TassadarPlannerRoutingRequest {
        TassadarPlannerRoutingRequest::new(
            format!("planner-request-{case_id}"),
            "session-alpha",
            "planner-fixture-v0",
            TassadarPlannerExecutorSubproblem::new(
                format!("subproblem-{case_id}"),
                "exact arithmetic subproblem",
                request_for_case(case_id).program_artifact,
                TassadarExecutorDecodeMode::HullCache,
            )
            .with_environment_refs(vec![String::from("env.openagents.tassadar.benchmark")]),
        )
    }

    fn backward_branch_sparse_artifact() -> TassadarProgramArtifact {
        let profile = TassadarWasmProfile::core_i32_v2();
        let trace_abi = TassadarTraceAbi::core_i32_v2();
        let program = TassadarProgram::new(
            "tassadar.backward_branch_sparse.v1",
            &profile,
            1,
            0,
            vec![
                TassadarInstruction::I32Const { value: 1 },
                TassadarInstruction::LocalSet { local: 0 },
                TassadarInstruction::LocalGet { local: 0 },
                TassadarInstruction::BrIf { target_pc: 2 },
                TassadarInstruction::I32Const { value: 9 },
                TassadarInstruction::Output,
                TassadarInstruction::Return,
            ],
        );
        TassadarProgramArtifact::fixture_reference(
            "artifact://tassadar/backward_branch_sparse",
            &profile,
            &trace_abi,
            program,
        )
        .expect("backward branch fixture should build")
    }

    #[test]
    fn planner_router_delegates_completed_exact_request_with_full_executor_truth() {
        let router = LocalTassadarPlannerRouter::new();
        let request = planner_request_for_case("locals_add");

        let outcome = router
            .route(&request)
            .expect("planner route should succeed");
        match outcome {
            TassadarPlannerRoutingOutcome::Completed { response } => {
                assert_eq!(
                    response.routing_decision.planner_product_id,
                    PLANNER_EXECUTOR_ROUTE_PRODUCT_ID
                );
                assert_eq!(
                    response.routing_decision.executor_product_id,
                    EXECUTOR_TRACE_PRODUCT_ID
                );
                assert_eq!(response.executor_response.final_outputs(), &[12]);
                assert_eq!(
                    response
                        .executor_response
                        .evidence_bundle
                        .proof_bundle
                        .product_id,
                    EXECUTOR_TRACE_PRODUCT_ID
                );
                assert_eq!(
                    response.routing_decision.effective_decode_mode,
                    Some(TassadarExecutorDecodeMode::HullCache)
                );
                assert!(response.routing_decision.selection.is_some());
                assert!(!response.routing_decision.routing_digest.is_empty());
            }
            other => panic!("expected delegated completion, got {other:?}"),
        }
    }

    #[test]
    fn planner_router_can_return_typed_fallback_when_policy_disallows_runtime_decode_fallback() {
        let router = LocalTassadarPlannerRouter::new().with_executor_service(
            LocalTassadarExecutorService::new()
                .with_fixture(TassadarExecutorFixture::core_i32_v2()),
        );
        let request = TassadarPlannerRoutingRequest::new(
            "planner-request-sparse-fallback",
            "session-beta",
            "planner-fixture-v0",
            TassadarPlannerExecutorSubproblem::new(
                "subproblem-sparse-fallback",
                "exact sparse top-k subproblem",
                backward_branch_sparse_artifact(),
                TassadarExecutorDecodeMode::SparseTopK,
            ),
        )
        .with_routing_budget(TassadarPlannerRoutingBudget::new(128, 512, 8))
        .with_routing_policy(
            TassadarPlannerRoutingPolicy::exact_executor_default()
                .with_fallback_policy(TassadarPlannerFallbackPolicy::PlannerSummary),
        );
        let request = TassadarPlannerRoutingRequest {
            subproblem: request
                .subproblem
                .with_requested_model_id(TassadarExecutorFixture::CORE_I32_V2_MODEL_ID),
            routing_policy: TassadarPlannerRoutingPolicy {
                allow_runtime_decode_fallback: false,
                ..request.routing_policy
            },
            ..request
        };

        let outcome = router
            .route(&request)
            .expect("planner route should be typed");
        match outcome {
            TassadarPlannerRoutingOutcome::Fallback { fallback } => {
                assert_eq!(
                    fallback.routing_decision.route_reason,
                    Some(TassadarPlannerRouteReason::ExecutorDecodeFallbackDisallowed)
                );
                assert!(
                    fallback
                        .routing_decision
                        .selection
                        .as_ref()
                        .is_some_and(|selection| selection.is_fallback())
                );
                assert!(fallback.fallback_summary.contains("disallowed"));
            }
            other => panic!("expected typed fallback, got {other:?}"),
        }
    }

    #[test]
    fn planner_router_refuses_when_program_exceeds_budget() {
        let router = LocalTassadarPlannerRouter::new();
        let request = planner_request_for_case("memory_roundtrip")
            .with_routing_budget(TassadarPlannerRoutingBudget::new(4, 512, 8));

        let outcome = router
            .route(&request)
            .expect("planner route should be typed");
        match outcome {
            TassadarPlannerRoutingOutcome::Refused { refusal } => {
                assert_eq!(
                    refusal.routing_decision.route_reason,
                    Some(TassadarPlannerRouteReason::ProgramLengthBudgetExceeded)
                );
                assert!(refusal.detail.contains("exceeds planner budget"));
            }
            other => panic!("expected budget refusal, got {other:?}"),
        }
    }

    #[test]
    fn planner_router_rejects_non_planner_product_id() {
        let router = LocalTassadarPlannerRouter::new();
        let mut request = planner_request_for_case("locals_add");
        request.product_id = String::from("psionic.text_generation");

        let error = router
            .route(&request)
            .expect_err("wrong planner product should fail before routing");
        assert_eq!(
            error,
            TassadarPlannerRouterError::UnsupportedProduct {
                product_id: String::from("psionic.text_generation"),
            }
        );
    }
}
