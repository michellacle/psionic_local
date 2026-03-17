//! Bounded MLX-style text serving package above Psionic-native catalog and
//! OpenAI-compatible serving.

use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
};

use axum::Router;
use psionic_mlx_catalog::{
    MlxCatalogError, MlxCatalogResolutionReport, MlxCatalogRoots, MlxCatalogWorkspace,
    MlxRemoteMetadataPolicy,
};
use psionic_models::{
    DecoderModelDescriptor, GgufDecoderFamily, reasoning_parser_for_decoder_family,
};
use psionic_router::{ResponseStateCapability, ResponseStateRetentionPolicy, ResponseStateStore};
use psionic_runtime::{StructuredOutputCapability, local_structured_output_capabilities};
use psionic_serve::{OpenAiCompatConfig, OpenAiCompatServer, OpenAiCompatServerError};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::net::TcpListener;

/// Human-readable crate ownership summary.
pub const CRATE_ROLE: &str =
    "bounded MLX-style text serving package above Psionic-native catalog and OpenAI-compatible server surfaces";

/// Response-state storage owned by the MLX text serving package.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MlxServeResponseStateStorage {
    /// In-memory prompt-replay state.
    InMemory,
    /// JSON-file-backed prompt-replay state.
    JsonFile {
        /// Stable file path used for persistence.
        path: PathBuf,
    },
}

impl Default for MlxServeResponseStateStorage {
    fn default() -> Self {
        Self::InMemory
    }
}

/// Response-state config for the MLX text serving package.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxServeResponseStateConfig {
    /// Backing storage kind.
    pub storage: MlxServeResponseStateStorage,
    /// Bounded retention policy.
    pub retention: ResponseStateRetentionPolicy,
}

impl Default for MlxServeResponseStateConfig {
    fn default() -> Self {
        Self {
            storage: MlxServeResponseStateStorage::InMemory,
            retention: ResponseStateRetentionPolicy::default(),
        }
    }
}

impl MlxServeResponseStateConfig {
    fn build_store(&self) -> Result<ResponseStateStore, MlxServeError> {
        match &self.storage {
            MlxServeResponseStateStorage::InMemory => {
                Ok(ResponseStateStore::in_memory(self.retention))
            }
            MlxServeResponseStateStorage::JsonFile { path } => {
                Ok(ResponseStateStore::file_backed(path, self.retention)?)
            }
        }
    }
}

/// Package config for one MLX text-serving server.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxTextServeConfig {
    /// MLX-style model references to resolve and load.
    pub references: Vec<String>,
    /// Local catalog/cache roots.
    pub roots: MlxCatalogRoots,
    /// Explicit metadata trust policy.
    pub metadata_policy: MlxRemoteMetadataPolicy,
    /// Host interface for the HTTP server.
    pub host: String,
    /// TCP port for the HTTP server.
    pub port: u16,
    /// Reasoning budget forwarded to the generic Psionic server.
    pub reasoning_budget: u8,
    /// Response-state backing store and retention posture.
    pub response_state: MlxServeResponseStateConfig,
}

impl MlxTextServeConfig {
    /// Creates one package config for a single model reference.
    #[must_use]
    pub fn new(reference: impl Into<String>) -> Self {
        Self {
            references: vec![reference.into()],
            roots: MlxCatalogRoots::default(),
            metadata_policy: MlxRemoteMetadataPolicy::default(),
            host: String::from("127.0.0.1"),
            port: 8080,
            reasoning_budget: 0,
            response_state: MlxServeResponseStateConfig::default(),
        }
    }

    /// Adds one additional model reference.
    pub fn add_reference(&mut self, reference: impl Into<String>) {
        self.references.push(reference.into());
    }

    /// Returns the bound socket address for the server.
    pub fn socket_addr(&self) -> Result<SocketAddr, MlxServeError> {
        let mut server = OpenAiCompatConfig::new(PathBuf::from("/tmp/placeholder.gguf"));
        server.host = self.host.clone();
        server.port = self.port;
        server.socket_addr().map_err(MlxServeError::Server)
    }
}

/// Lifecycle truth for one MLX text-serving package instance.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxServeLifecycleReport {
    /// Current model load status.
    pub load_status: String,
    /// Current warm-control posture.
    pub warm_control: String,
    /// Current unload-control posture.
    pub unload_control: String,
    /// Current memory-pressure reporting posture.
    pub memory_pressure_reporting: String,
}

/// Reusable feature report for one served MLX text model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxTextServeFeatureReport {
    /// Stable endpoint list for the model.
    pub supported_endpoints: Vec<String>,
    /// Whether non-streaming completions are available.
    pub non_streaming: bool,
    /// Whether streaming completions are available.
    pub streaming: bool,
    /// Whether logprobs are surfaced.
    pub logprobs: bool,
    /// Whether explicit stop-sequence handling is surfaced.
    pub stop_sequences: bool,
    /// Whether shared-prefix cache reuse is surfaced.
    pub prefix_cache_reuse: bool,
    /// Tool-calling modes currently surfaced.
    pub tool_calling_modes: Vec<String>,
    /// Structured-output capabilities currently surfaced.
    pub structured_output: Vec<StructuredOutputCapability>,
    /// Family-specific reasoning parser when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_parser: Option<String>,
}

/// One resolved model owned by the MLX text-serving package.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxServedTextModelReport {
    /// Original caller-facing model reference.
    pub reference: String,
    /// Resolved direct GGUF path used by the shared server.
    pub resolved_model_path: PathBuf,
    /// Stable catalog-resolution report.
    pub resolution: MlxCatalogResolutionReport,
    /// Loaded text descriptor.
    pub descriptor: DecoderModelDescriptor,
    /// Feature/capability summary for the served model.
    pub features: MlxTextServeFeatureReport,
}

/// Machine-readable bootstrap report for one package-owned server plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxTextServeBootstrapReport {
    /// Stable schema version.
    pub schema_version: u32,
    /// Human-readable crate role.
    pub crate_role: String,
    /// HTTP host.
    pub host: String,
    /// HTTP port.
    pub port: u16,
    /// Runtime backend label.
    pub backend: String,
    /// Runtime execution mode.
    pub execution_mode: String,
    /// Runtime execution engine.
    pub execution_engine: String,
    /// Package-owned lifecycle truth.
    pub lifecycle: MlxServeLifecycleReport,
    /// Response-state capability for `/v1/responses`.
    pub response_state: ResponseStateCapability,
    /// Resolved model inventory.
    pub models: Vec<MlxServedTextModelReport>,
}

/// Planned MLX text-serving package instance.
pub struct MlxTextServePackage {
    report: MlxTextServeBootstrapReport,
    server: OpenAiCompatServer,
}

impl MlxTextServePackage {
    /// Returns the bootstrap report.
    #[must_use]
    pub fn report(&self) -> &MlxTextServeBootstrapReport {
        &self.report
    }

    /// Returns the shared OpenAI-compatible router.
    pub fn router(&self) -> Router {
        self.server.router()
    }

    /// Serves the package-owned router on one bound listener.
    pub async fn serve(&self, listener: TcpListener) -> Result<(), MlxServeError> {
        self.server.serve(listener).await.map_err(MlxServeError::Server)
    }

    /// Saves the bootstrap report to one JSON path.
    pub fn save_report_path(&self, path: impl AsRef<Path>) -> Result<(), MlxServeError> {
        let json = serde_json::to_string_pretty(self.report())?;
        std::fs::write(path, format!("{json}\n"))?;
        Ok(())
    }
}

/// Error returned by the MLX text serving package.
#[derive(Debug, Error)]
pub enum MlxServeError {
    /// The package requires at least one model reference.
    #[error("mlx text serve requires at least one `reference`")]
    MissingReferences,
    /// Local model resolution failed.
    #[error(transparent)]
    Catalog(#[from] MlxCatalogError),
    /// Response-state persistence failed.
    #[error(transparent)]
    ResponseState(#[from] psionic_router::ResponseStateError),
    /// Shared server setup failed.
    #[error(transparent)]
    Server(#[from] OpenAiCompatServerError),
    /// Serializing or decoding one JSON artifact failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Writing one package report failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Bounded MLX text-serving workspace.
#[derive(Clone, Debug)]
pub struct MlxTextServeWorkspace {
    catalog: MlxCatalogWorkspace,
}

impl Default for MlxTextServeWorkspace {
    fn default() -> Self {
        Self::new(MlxCatalogRoots::default())
    }
}

impl MlxTextServeWorkspace {
    /// Creates one MLX text-serving workspace over the provided roots.
    #[must_use]
    pub fn new(roots: MlxCatalogRoots) -> Self {
        Self {
            catalog: MlxCatalogWorkspace::new(roots),
        }
    }

    /// Returns the underlying catalog workspace.
    #[must_use]
    pub fn catalog(&self) -> &MlxCatalogWorkspace {
        &self.catalog
    }

    /// Plans one package-owned OpenAI-compatible server and returns the
    /// bootstrap report plus the shared server instance.
    pub fn plan_server(
        &self,
        config: &MlxTextServeConfig,
    ) -> Result<MlxTextServePackage, MlxServeError> {
        if config.references.is_empty() {
            return Err(MlxServeError::MissingReferences);
        }

        let mut server_config = OpenAiCompatConfig::new(PathBuf::from("/tmp/placeholder.gguf"));
        server_config.model_paths.clear();
        server_config.host = config.host.clone();
        server_config.port = config.port;
        server_config.reasoning_budget = config.reasoning_budget;

        let mut models = Vec::new();
        for reference in &config.references {
            let resolution = self.catalog.resolve(reference, &config.metadata_policy)?;
            let path = resolution.direct_gguf_path.clone().ok_or_else(|| {
                MlxCatalogError::TextRuntimeUnavailable {
                    reference: reference.clone(),
                    reason: String::from(
                        "resolved MLX serving source does not expose one direct GGUF text runtime",
                    ),
                }
            })?;
            let runtime = self.catalog.open_text_runtime(reference, &config.metadata_policy)?;
            let load_report = runtime.load_report();
            server_config.add_model_path(path.clone());
            let descriptor = load_report.descriptor.clone();
            models.push(MlxServedTextModelReport {
                reference: reference.clone(),
                resolved_model_path: path,
                resolution,
                descriptor,
                features: feature_report_for_descriptor(&load_report.descriptor),
            });
        }

        let response_state = config.response_state.build_store()?;
        let response_state_capability = response_state.capability();
        let server =
            OpenAiCompatServer::from_config_with_response_state_store(&server_config, response_state)?;
        Ok(MlxTextServePackage {
            report: MlxTextServeBootstrapReport {
                schema_version: 1,
                crate_role: String::from(CRATE_ROLE),
                host: config.host.clone(),
                port: config.port,
                backend: String::from(server.backend_label()),
                execution_mode: String::from(server.execution_mode_label()),
                execution_engine: String::from(server.execution_engine_label()),
                lifecycle: MlxServeLifecycleReport {
                    load_status: String::from("loaded"),
                    warm_control: String::from("not_implemented"),
                    unload_control: String::from("not_implemented"),
                    memory_pressure_reporting: String::from("not_implemented"),
                },
                response_state: response_state_capability,
                models,
            },
            server,
        })
    }
}

fn feature_report_for_descriptor(descriptor: &DecoderModelDescriptor) -> MlxTextServeFeatureReport {
    let reasoning_parser = family_enum_from_label(descriptor.model.family.as_str())
        .and_then(reasoning_parser_for_decoder_family)
        .map(|parser| String::from(parser.label()));
    MlxTextServeFeatureReport {
        supported_endpoints: vec![
            String::from("/v1/chat/completions"),
            String::from("/v1/responses"),
        ],
        non_streaming: true,
        streaming: true,
        logprobs: true,
        stop_sequences: true,
        prefix_cache_reuse: true,
        tool_calling_modes: vec![
            String::from("none"),
            String::from("auto"),
            String::from("required"),
            String::from("named"),
        ],
        structured_output: local_structured_output_capabilities(),
        reasoning_parser,
    }
}

fn family_enum_from_label(label: &str) -> Option<GgufDecoderFamily> {
    match label {
        "llama" => Some(GgufDecoderFamily::Llama),
        "qwen" => Some(GgufDecoderFamily::Qwen),
        "mistral" => Some(GgufDecoderFamily::Mistral),
        "gpt_oss" => Some(GgufDecoderFamily::GptOss),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MlxServeResponseStateStorage, MlxTextServeConfig, MlxTextServeWorkspace,
    };
    use psionic_models::{GgufMetadataValue, GgufTensorType};
    use psionic_runtime::StructuredOutputKind;
    use std::{fs, path::Path};

    #[test]
    fn plan_server_resolves_direct_text_model_and_reports_capabilities()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let model_path = temp.path().join("tiny-qwen.gguf");
        write_test_gguf(
            &model_path,
            qwen2_metadata("Tiny Qwen", 32).as_slice(),
            dense_decoder_tensors().as_slice(),
        )?;

        let workspace = MlxTextServeWorkspace::new(Default::default());
        let config = MlxTextServeConfig::new(model_path.to_string_lossy().to_string());
        let package = workspace.plan_server(&config)?;
        let report = package.report();

        assert_eq!(report.backend, "cpu");
        assert_eq!(report.execution_mode, "native");
        assert_eq!(report.execution_engine, "psionic");
        assert_eq!(report.lifecycle.load_status, "loaded");
        assert_eq!(report.models.len(), 1);
        assert_eq!(report.models[0].reference, model_path.to_string_lossy());
        assert_eq!(
            report.models[0].features.supported_endpoints,
            vec![
                String::from("/v1/chat/completions"),
                String::from("/v1/responses")
            ]
        );
        assert!(report.models[0].features.streaming);
        assert!(report.models[0].features.prefix_cache_reuse);
        assert_eq!(
            report
                .models[0]
                .features
                .structured_output
                .iter()
                .map(|capability| capability.kind)
                .collect::<Vec<_>>(),
            vec![
                StructuredOutputKind::Choice,
                StructuredOutputKind::Regex,
                StructuredOutputKind::Grammar,
                StructuredOutputKind::JsonSchema,
                StructuredOutputKind::JsonObject,
                StructuredOutputKind::TaggedStructure,
            ]
        );
        Ok(())
    }

    #[test]
    fn plan_server_supports_json_file_response_state_store()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let model_path = temp.path().join("tiny-qwen.gguf");
        write_test_gguf(
            &model_path,
            qwen2_metadata("Tiny Qwen", 32).as_slice(),
            dense_decoder_tensors().as_slice(),
        )?;
        let response_state_path = temp.path().join("responses.json");
        let workspace = MlxTextServeWorkspace::new(Default::default());
        let mut config = MlxTextServeConfig::new(model_path.to_string_lossy().to_string());
        config.response_state.storage = MlxServeResponseStateStorage::JsonFile {
            path: response_state_path.clone(),
        };
        let package = workspace.plan_server(&config)?;

        assert_eq!(package.report().response_state.storage, "json_file");
        assert_eq!(
            package.report().response_state.retention_scope,
            "best_effort_local_durable"
        );
        Ok(())
    }

    fn qwen2_metadata(name: &str, context_length: u32) -> Vec<(String, GgufMetadataValue)> {
        vec![
            (
                String::from("general.architecture"),
                GgufMetadataValue::String(String::from("qwen2")),
            ),
            (
                String::from("general.name"),
                GgufMetadataValue::String(name.to_string()),
            ),
            (
                String::from("qwen2.context_length"),
                GgufMetadataValue::U32(context_length),
            ),
            (
                String::from("qwen2.embedding_length"),
                GgufMetadataValue::U32(4),
            ),
            (
                String::from("qwen2.feed_forward_length"),
                GgufMetadataValue::U32(8),
            ),
            (String::from("qwen2.block_count"), GgufMetadataValue::U32(1)),
            (
                String::from("qwen2.attention.head_count"),
                GgufMetadataValue::U32(2),
            ),
            (
                String::from("qwen2.attention.head_count_kv"),
                GgufMetadataValue::U32(1),
            ),
            (
                String::from("qwen2.attention.layer_norm_rms_epsilon"),
                GgufMetadataValue::F32(1e-6),
            ),
            (
                String::from("qwen2.rope.freq_base"),
                GgufMetadataValue::F32(1_000_000.0),
            ),
            (
                String::from("qwen2.attention.sliding_window"),
                GgufMetadataValue::U32(32),
            ),
            (
                String::from("tokenizer.ggml.model"),
                GgufMetadataValue::String(String::from("gpt2")),
            ),
            (
                String::from("tokenizer.ggml.pre"),
                GgufMetadataValue::String(String::from("qwen2")),
            ),
            (
                String::from("tokenizer.ggml.tokens"),
                GgufMetadataValue::Array(vec![
                    GgufMetadataValue::String(String::from("<|bos|>")),
                    GgufMetadataValue::String(String::from("<|eos|>")),
                    GgufMetadataValue::String(String::from("hello")),
                    GgufMetadataValue::String(String::from("world")),
                ]),
            ),
            (
                String::from("tokenizer.ggml.merges"),
                GgufMetadataValue::Array(vec![GgufMetadataValue::String(String::from(
                    "hello world",
                ))]),
            ),
            (
                String::from("tokenizer.ggml.bos_token_id"),
                GgufMetadataValue::U32(0),
            ),
            (
                String::from("tokenizer.ggml.eos_token_id"),
                GgufMetadataValue::U32(1),
            ),
            (
                String::from("tokenizer.ggml.unknown_token_id"),
                GgufMetadataValue::U32(0),
            ),
            (
                String::from("tokenizer.ggml.add_bos_token"),
                GgufMetadataValue::Bool(false),
            ),
            (
                String::from("tokenizer.ggml.add_eos_token"),
                GgufMetadataValue::Bool(false),
            ),
        ]
    }

    fn dense_decoder_tensors() -> Vec<TestGgufTensor> {
        vec![
            dense_tensor(
                "token_embd.weight",
                vec![4, 4],
                vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            dense_tensor("output_norm.weight", vec![4], vec![1.0, 1.0, 1.0, 1.0]),
            dense_tensor(
                "output.weight",
                vec![4, 4],
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ),
            dense_tensor("blk.0.attn_norm.weight", vec![4], vec![1.0, 1.0, 1.0, 1.0]),
            dense_tensor("blk.0.attn_q.weight", vec![4, 4], vec![0.0; 16]),
            dense_tensor("blk.0.attn_k.weight", vec![2, 4], vec![0.0; 8]),
            dense_tensor("blk.0.attn_v.weight", vec![2, 4], vec![0.0; 8]),
            dense_tensor("blk.0.attn_output.weight", vec![4, 4], vec![0.0; 16]),
            dense_tensor("blk.0.ffn_gate.weight", vec![8, 4], vec![0.0; 32]),
            dense_tensor("blk.0.ffn_down.weight", vec![4, 8], vec![0.0; 32]),
            dense_tensor("blk.0.ffn_up.weight", vec![8, 4], vec![0.0; 32]),
            dense_tensor("blk.0.ffn_norm.weight", vec![4], vec![1.0, 1.0, 1.0, 1.0]),
            dense_tensor("blk.0.attn_q.bias", vec![4], vec![0.0; 4]),
            dense_tensor("blk.0.attn_k.bias", vec![2], vec![0.0; 2]),
            dense_tensor("blk.0.attn_v.bias", vec![2], vec![0.0; 2]),
        ]
    }

    #[derive(Clone, Debug)]
    struct TestGgufTensor {
        name: String,
        shape: Vec<usize>,
        tensor_type: GgufTensorType,
        bytes: Vec<u8>,
    }

    impl TestGgufTensor {
        fn new(
            name: impl Into<String>,
            shape: Vec<usize>,
            tensor_type: GgufTensorType,
            bytes: Vec<u8>,
        ) -> Self {
            Self {
                name: name.into(),
                shape,
                tensor_type,
                bytes,
            }
        }
    }

    fn dense_tensor(name: &str, shape: Vec<usize>, values: Vec<f32>) -> TestGgufTensor {
        TestGgufTensor::new(name, shape, GgufTensorType::F32, encode_f32_bytes(values.as_slice()))
    }

    fn encode_f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
        for value in values {
            bytes.extend(value.to_le_bytes());
        }
        bytes
    }

    fn write_test_gguf(
        path: &Path,
        metadata: &[(String, GgufMetadataValue)],
        tensors: &[TestGgufTensor],
    ) -> Result<(), Box<dyn std::error::Error>> {
        fs::write(path, build_test_gguf(metadata, tensors)?)?;
        Ok(())
    }

    fn build_test_gguf(
        metadata: &[(String, GgufMetadataValue)],
        tensors: &[TestGgufTensor],
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, u64::try_from(tensors.len())?);
        push_u64(&mut bytes, u64::try_from(metadata.len())?);

        for (key, value) in metadata {
            push_gguf_string(&mut bytes, key)?;
            push_u32(&mut bytes, gguf_metadata_value_type(value));
            push_gguf_value(&mut bytes, value)?;
        }

        let alignment = 32usize;
        let mut next_offset = 0usize;
        let mut tensor_offsets = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            tensor_offsets.push(next_offset);
            next_offset = align_usize(next_offset + tensor.bytes.len(), alignment);
        }

        for (tensor, offset) in tensors.iter().zip(&tensor_offsets) {
            push_gguf_string(&mut bytes, tensor.name.as_str())?;
            push_u32(&mut bytes, u32::try_from(tensor.shape.len())?);
            for dimension in tensor.shape.iter().rev() {
                push_u64(&mut bytes, u64::try_from(*dimension)?);
            }
            push_u32(&mut bytes, gguf_tensor_type_code(tensor.tensor_type));
            push_u64(&mut bytes, u64::try_from(*offset)?);
        }

        let tensor_data_offset = align_usize(bytes.len(), alignment);
        bytes.resize(tensor_data_offset, 0);

        for (tensor, offset) in tensors.iter().zip(&tensor_offsets) {
            let start = tensor_data_offset + offset;
            if bytes.len() < start {
                bytes.resize(start, 0);
            }
            bytes.extend_from_slice(tensor.bytes.as_slice());
            bytes.resize(align_usize(bytes.len(), alignment), 0);
        }

        Ok(bytes)
    }

    fn align_usize(value: usize, alignment: usize) -> usize {
        let remainder = value % alignment;
        if remainder == 0 {
            value
        } else {
            value + alignment - remainder
        }
    }

    fn gguf_metadata_value_type(value: &GgufMetadataValue) -> u32 {
        match value {
            GgufMetadataValue::U8(_) => 0,
            GgufMetadataValue::I8(_) => 1,
            GgufMetadataValue::U16(_) => 2,
            GgufMetadataValue::I16(_) => 3,
            GgufMetadataValue::U32(_) => 4,
            GgufMetadataValue::I32(_) => 5,
            GgufMetadataValue::F32(_) => 6,
            GgufMetadataValue::Bool(_) => 7,
            GgufMetadataValue::String(_) => 8,
            GgufMetadataValue::Array(_) => 9,
            GgufMetadataValue::U64(_) => 10,
            GgufMetadataValue::I64(_) => 11,
            GgufMetadataValue::F64(_) => 12,
        }
    }

    fn gguf_tensor_type_code(tensor_type: GgufTensorType) -> u32 {
        match tensor_type {
            GgufTensorType::F32 => 0,
            other => panic!("unsupported synthetic gguf tensor type: {other:?}"),
        }
    }

    fn push_gguf_string(
        bytes: &mut Vec<u8>,
        value: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        push_u64(bytes, u64::try_from(value.len())?);
        bytes.extend_from_slice(value.as_bytes());
        Ok(())
    }

    fn push_gguf_value(
        bytes: &mut Vec<u8>,
        value: &GgufMetadataValue,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match value {
            GgufMetadataValue::U8(value) => bytes.push(*value),
            GgufMetadataValue::I8(value) => bytes.push(value.to_le_bytes()[0]),
            GgufMetadataValue::U16(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::I16(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::U32(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::I32(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::U64(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::I64(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::F32(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::F64(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::Bool(value) => bytes.push(u8::from(*value)),
            GgufMetadataValue::String(value) => push_gguf_string(bytes, value)?,
            GgufMetadataValue::Array(values) => {
                let value_type = values.first().map_or(4, gguf_metadata_value_type);
                push_u32(bytes, value_type);
                push_u64(bytes, u64::try_from(values.len())?);
                for value in values {
                    push_gguf_value(bytes, value)?;
                }
            }
        }
        Ok(())
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend(value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend(value.to_le_bytes());
    }
}
