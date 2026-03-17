//! Bounded MLX-style model catalog workflows above Psionic-native substrate.

use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use psionic_catalog::{
    BlobReadPreference, CatalogError, LocalBlobOpenOptions, OllamaCatalogDiscovery,
    OllamaLayerKind, OllamaManifest, OllamaModelCatalog, OllamaRegistryClient,
    OllamaRegistryPullOptions, OllamaRegistryPullReport, RegistryPullError,
};
use psionic_mlx_lm::{MlxLmError, MlxLmLoadReport, MlxLmTextRuntime};
use psionic_models::{GgufDecoderAdapterLoader, ModelLoadError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Human-readable crate ownership summary.
pub const CRATE_ROLE: &str =
    "bounded MLX-style model catalog and Hugging Face cache workflows above Psionic-native substrate";

/// Default local Ollama models root.
#[must_use]
pub fn default_ollama_models_root() -> PathBuf {
    env::var_os("OLLAMA_MODELS")
        .map(PathBuf::from)
        .or_else(home_dir)
        .map(|path| {
            if path.ends_with("models") {
                path
            } else {
                path.join(".ollama/models")
            }
        })
        .unwrap_or_else(|| PathBuf::from(".ollama/models"))
}

/// Default local Hugging Face hub cache root.
#[must_use]
pub fn default_hugging_face_hub_root() -> PathBuf {
    if let Some(root) = env::var_os("HF_HOME").map(PathBuf::from) {
        return root.join("hub");
    }
    home_dir()
        .map(|path| path.join(".cache/huggingface/hub"))
        .unwrap_or_else(|| PathBuf::from(".cache/huggingface/hub"))
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME").map(PathBuf::from)
}

/// Roots used by the MLX catalog layer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxCatalogRoots {
    /// Local Ollama models root.
    pub ollama_models_root: PathBuf,
    /// Local Hugging Face hub cache root.
    pub hugging_face_hub_root: PathBuf,
}

impl Default for MlxCatalogRoots {
    fn default() -> Self {
        Self {
            ollama_models_root: default_ollama_models_root(),
            hugging_face_hub_root: default_hugging_face_hub_root(),
        }
    }
}

/// Trust mode for remote-derived metadata stored in local catalogs or caches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxMetadataTrustMode {
    /// Refuse to trust the metadata and report it explicitly.
    Refuse,
    /// Allow digest-bound metadata already materialized in one local cache.
    AllowDigestBoundLocalCache,
}

/// Explicit trust policy for remote processor and template metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxRemoteMetadataPolicy {
    /// Policy for processor or preprocessor metadata.
    pub processor_metadata: MlxMetadataTrustMode,
    /// Policy for chat-template or prompt-shaping metadata.
    pub template_metadata: MlxMetadataTrustMode,
}

impl Default for MlxRemoteMetadataPolicy {
    fn default() -> Self {
        Self {
            processor_metadata: MlxMetadataTrustMode::Refuse,
            template_metadata: MlxMetadataTrustMode::Refuse,
        }
    }
}

/// One observed metadata file or layer that influenced the trust decision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxRemoteMetadataObservation {
    /// Stable local path for the observed metadata payload.
    pub path: PathBuf,
    /// Stable digest or layer identity for the payload.
    pub digest: String,
}

/// Current disposition for one metadata class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxRemoteMetadataDisposition {
    /// No metadata of this class was discovered.
    Absent,
    /// Metadata was discovered and refused by current policy.
    Refused,
    /// Metadata was discovered and allowed because it is digest-bound in local cache.
    AllowedDigestBoundLocalCache,
}

/// Stable assessment for one metadata class.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxRemoteMetadataAssessment {
    /// Final disposition under current policy.
    pub disposition: MlxRemoteMetadataDisposition,
    /// Observed metadata payloads in stable path order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub observations: Vec<MlxRemoteMetadataObservation>,
    /// Explicit refusal reason when the metadata was not trusted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal_reason: Option<String>,
}

impl MlxRemoteMetadataAssessment {
    fn absent() -> Self {
        Self {
            disposition: MlxRemoteMetadataDisposition::Absent,
            observations: Vec::new(),
            refusal_reason: None,
        }
    }
}

/// Explicit processor/template metadata trust report.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxRemoteMetadataReport {
    /// Processor/preprocessor metadata assessment.
    pub processor_metadata: MlxRemoteMetadataAssessment,
    /// Template/prompt-shaping metadata assessment.
    pub template_metadata: MlxRemoteMetadataAssessment,
}

impl MlxRemoteMetadataReport {
    fn none() -> Self {
        Self {
            processor_metadata: MlxRemoteMetadataAssessment::absent(),
            template_metadata: MlxRemoteMetadataAssessment::absent(),
        }
    }

    /// Returns whether either metadata class was refused.
    #[must_use]
    pub fn has_refusal(&self) -> bool {
        self.processor_metadata.disposition == MlxRemoteMetadataDisposition::Refused
            || self.template_metadata.disposition == MlxRemoteMetadataDisposition::Refused
    }
}

/// Explicit loader kind the catalog registry resolves for one architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxArchitectureLoaderKind {
    /// GGUF decoder-family adapter and text runtime.
    GgufDecoder,
}

/// One architecture-specific loader registration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxArchitectureRegistration {
    /// Canonical architecture label.
    pub canonical_architecture: String,
    /// Higher-level model family used by current package layers.
    pub model_family: String,
    /// Loader kind used for the package workflow.
    pub loader_kind: MlxArchitectureLoaderKind,
    /// Whether `psionic-mlx-lm` can open the artifact directly today.
    pub direct_text_runtime_supported: bool,
    /// Additional aliases accepted by the registry.
    pub aliases: Vec<String>,
}

/// Bounded architecture registry for current MLX-class package workflows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxArchitectureRegistry {
    registrations: BTreeMap<String, MlxArchitectureRegistration>,
}

impl Default for MlxArchitectureRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

impl MlxArchitectureRegistry {
    /// Returns the builtin architecture registry.
    #[must_use]
    pub fn builtin() -> Self {
        let mut registry = Self {
            registrations: BTreeMap::new(),
        };
        registry.register(MlxArchitectureRegistration {
            canonical_architecture: String::from("llama"),
            model_family: String::from("llama"),
            loader_kind: MlxArchitectureLoaderKind::GgufDecoder,
            direct_text_runtime_supported: true,
            aliases: vec![
                String::from("llama"),
                String::from("llamaforcausallm"),
                String::from("mistral"),
                String::from("mistralforcausallm"),
            ],
        });
        registry.register(MlxArchitectureRegistration {
            canonical_architecture: String::from("qwen2"),
            model_family: String::from("qwen"),
            loader_kind: MlxArchitectureLoaderKind::GgufDecoder,
            direct_text_runtime_supported: true,
            aliases: vec![
                String::from("qwen"),
                String::from("qwen2"),
                String::from("qwen2forcausallm"),
            ],
        });
        registry.register(MlxArchitectureRegistration {
            canonical_architecture: String::from("gpt_oss"),
            model_family: String::from("gpt_oss"),
            loader_kind: MlxArchitectureLoaderKind::GgufDecoder,
            direct_text_runtime_supported: true,
            aliases: vec![
                String::from("gpt_oss"),
                String::from("gptoss"),
                String::from("gptossforcausallm"),
            ],
        });
        registry
    }

    /// Registers one architecture entry and all of its aliases.
    pub fn register(&mut self, registration: MlxArchitectureRegistration) {
        let canonical = normalize_architecture_key(registration.canonical_architecture.as_str());
        self.registrations
            .insert(canonical, registration.clone());
        for alias in &registration.aliases {
            self.registrations
                .insert(normalize_architecture_key(alias.as_str()), registration.clone());
        }
    }

    /// Resolves one architecture label to the current registration.
    #[must_use]
    pub fn resolve(&self, architecture: &str) -> Option<&MlxArchitectureRegistration> {
        self.registrations
            .get(&normalize_architecture_key(architecture))
    }
}

/// Parsed caller-facing model reference.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxParsedReference {
    /// Original caller-supplied reference.
    pub raw: String,
    /// Inferred reference kind.
    pub kind: MlxReferenceKind,
    /// Normalized reference value.
    pub value: String,
    /// Optional requested revision for Hugging Face references.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

/// Supported reference kind for the current MLX catalog layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxReferenceKind {
    /// Direct local GGUF path.
    LocalPath,
    /// Local or defaultable Ollama model reference.
    OllamaModel,
    /// Local Hugging Face hub cache reference.
    HuggingFaceRepo,
}

/// Source kind selected during resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxResolutionSourceKind {
    /// Direct local GGUF file.
    LocalGgufPath,
    /// Local Ollama manifest and model blob.
    LocalOllamaManifest,
    /// Local Hugging Face cache snapshot.
    HuggingFaceSnapshot,
}

/// Conversion readiness for one non-direct GGUF entrypoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxConversionReadiness {
    /// Required metadata and architecture registration are available.
    Ready,
    /// Required metadata exists but current trust policy refuses it.
    RefusedRemoteMetadata,
    /// No convertible weights were discovered.
    MissingWeights,
    /// The architecture is outside the current builtin registry.
    UnsupportedArchitecture,
}

/// Conversion or load entrypoint for one resolved source.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MlxConversionEntryPoint {
    /// The source can be handed directly to `psionic-mlx-lm`.
    DirectGguf {
        /// Direct GGUF path used by the text package.
        path: PathBuf,
    },
    /// One local Hugging Face snapshot can be converted or is explicitly refused.
    HuggingFaceSnapshot {
        /// Snapshot path under the local Hugging Face hub cache.
        snapshot_path: PathBuf,
        /// Discovered safetensors weight files.
        safetensors_files: Vec<PathBuf>,
        /// Parsed architecture label when present.
        #[serde(skip_serializing_if = "Option::is_none")]
        architecture: Option<String>,
        /// Current readiness under local files and metadata policy.
        readiness: MlxConversionReadiness,
        /// Metadata trust/refusal report applied to the snapshot.
        remote_metadata: MlxRemoteMetadataReport,
    },
    /// No honest entrypoint exists for the resolved source.
    Unavailable {
        /// High-signal refusal or missing-surface reason.
        reason: String,
    },
}

/// Stable summary for a resolved local Ollama manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxOllamaManifestSummary {
    /// Canonical model name.
    pub canonical_name: String,
    /// Short display name.
    pub short_name: String,
    /// Local manifest path.
    pub manifest_path: PathBuf,
    /// Stable manifest digest.
    pub manifest_sha256: String,
    /// Primary model blob path when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_model_blob: Option<PathBuf>,
}

/// One discovered local Hugging Face cache snapshot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HuggingFaceCacheSnapshot {
    /// Canonical repo id, such as `mlx-community/Qwen2-0.5B-Instruct-4bit`.
    pub repo_id: String,
    /// Snapshot revision directory name.
    pub revision: String,
    /// Snapshot path under the local hub cache.
    pub snapshot_path: PathBuf,
    /// Snapshot mtime when it could be determined.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_unix_seconds: Option<u64>,
    /// Direct GGUF candidates in stable path order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gguf_files: Vec<PathBuf>,
    /// Safetensors weight files in stable path order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub safetensors_files: Vec<PathBuf>,
    /// Parsed config path when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_path: Option<PathBuf>,
    /// Parsed architecture label when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    /// Explicit processor/template metadata trust report.
    pub remote_metadata: MlxRemoteMetadataReport,
}

/// Resolution report for one caller-facing model reference.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxCatalogResolutionReport {
    /// Original caller-facing reference.
    pub raw_reference: String,
    /// Parsed normalized reference.
    pub parsed_reference: MlxParsedReference,
    /// Selected source kind.
    pub source_kind: MlxResolutionSourceKind,
    /// Direct GGUF path when one exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direct_gguf_path: Option<PathBuf>,
    /// Parsed architecture label when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    /// Architecture registry match when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture_registration: Option<MlxArchitectureRegistration>,
    /// Processor/template metadata trust or refusal report.
    pub remote_metadata: MlxRemoteMetadataReport,
    /// Chosen conversion or direct-load entrypoint.
    pub conversion_entrypoint: MlxConversionEntryPoint,
    /// Additional bounded notes for the caller.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
    /// Ollama manifest summary when the source came through the local catalog.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ollama_manifest: Option<MlxOllamaManifestSummary>,
    /// Hugging Face snapshot summary when the source came through local cache discovery.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hugging_face_snapshot: Option<HuggingFaceCacheSnapshot>,
}

/// Failure returned by the MLX catalog package.
#[derive(Debug, Error)]
pub enum MlxCatalogError {
    /// Local Ollama catalog work failed.
    #[error(transparent)]
    Catalog(#[from] CatalogError),
    /// Local registry pull work failed.
    #[error(transparent)]
    RegistryPull(#[from] RegistryPullError),
    /// GGUF adapter loading failed.
    #[error(transparent)]
    Model(#[from] ModelLoadError),
    /// Opening a text runtime failed.
    #[error(transparent)]
    TextRuntime(#[from] MlxLmError),
    /// Reading or walking local cache files failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Decoding one cache JSON file failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// The caller supplied an invalid Hugging Face reference.
    #[error("invalid Hugging Face reference `{reference}`: {message}")]
    InvalidHuggingFaceReference {
        /// Original reference.
        reference: String,
        /// Validation failure summary.
        message: String,
    },
    /// The requested Hugging Face repo does not exist in the local cache root.
    #[error("hugging face repo `{repo_id}` does not exist under `{root}`")]
    MissingHuggingFaceRepo {
        /// Canonical repo id.
        repo_id: String,
        /// Local cache root used for lookup.
        root: PathBuf,
    },
    /// No snapshots were found for the requested repo.
    #[error("hugging face repo `{repo_id}` has no snapshots under `{root}`")]
    MissingHuggingFaceSnapshots {
        /// Canonical repo id.
        repo_id: String,
        /// Repo cache directory.
        root: PathBuf,
    },
    /// The resolved source cannot be opened directly as a text runtime.
    #[error("text runtime unavailable for `{reference}`: {reason}")]
    TextRuntimeUnavailable {
        /// Original reference.
        reference: String,
        /// High-signal refusal or missing-surface reason.
        reason: String,
    },
}

/// Bounded MLX-style model catalog workspace.
#[derive(Clone, Debug)]
pub struct MlxCatalogWorkspace {
    roots: MlxCatalogRoots,
    architecture_registry: MlxArchitectureRegistry,
}

impl Default for MlxCatalogWorkspace {
    fn default() -> Self {
        Self::new(MlxCatalogRoots::default())
    }
}

impl MlxCatalogWorkspace {
    /// Creates one catalog workspace over the provided roots.
    #[must_use]
    pub fn new(roots: MlxCatalogRoots) -> Self {
        Self {
            roots,
            architecture_registry: MlxArchitectureRegistry::builtin(),
        }
    }

    /// Returns the current roots.
    #[must_use]
    pub fn roots(&self) -> &MlxCatalogRoots {
        &self.roots
    }

    /// Returns the builtin architecture registry.
    #[must_use]
    pub fn architecture_registry(&self) -> &MlxArchitectureRegistry {
        &self.architecture_registry
    }

    /// Discovers local Ollama manifests.
    pub fn discover_ollama_models(&self) -> Result<OllamaCatalogDiscovery, MlxCatalogError> {
        Ok(OllamaModelCatalog::new(&self.roots.ollama_models_root).discover_models()?)
    }

    /// Discovers local Hugging Face cache snapshots for one repo id.
    pub fn discover_hugging_face_snapshots(
        &self,
        repo_id: &str,
        policy: &MlxRemoteMetadataPolicy,
    ) -> Result<Vec<HuggingFaceCacheSnapshot>, MlxCatalogError> {
        let repo_root = hugging_face_repo_root(&self.roots.hugging_face_hub_root, repo_id);
        if !repo_root.exists() {
            return Err(MlxCatalogError::MissingHuggingFaceRepo {
                repo_id: repo_id.to_string(),
                root: self.roots.hugging_face_hub_root.clone(),
            });
        }
        let snapshots_root = repo_root.join("snapshots");
        if !snapshots_root.exists() {
            return Err(MlxCatalogError::MissingHuggingFaceSnapshots {
                repo_id: repo_id.to_string(),
                root: repo_root,
            });
        }

        let mut snapshots = Vec::new();
        for entry in fs::read_dir(&snapshots_root)? {
            let entry = entry?;
            let snapshot_path = entry.path();
            if !snapshot_path.is_dir() {
                continue;
            }
            snapshots.push(inspect_hugging_face_snapshot(
                repo_id,
                &snapshot_path,
                policy,
            )?);
        }
        snapshots.sort_by(|left, right| {
            right
                .modified_unix_seconds
                .cmp(&left.modified_unix_seconds)
                .then_with(|| left.revision.cmp(&right.revision))
        });
        if snapshots.is_empty() {
            return Err(MlxCatalogError::MissingHuggingFaceSnapshots {
                repo_id: repo_id.to_string(),
                root: repo_root,
            });
        }
        Ok(snapshots)
    }

    /// Resolves one caller-facing model reference.
    pub fn resolve(
        &self,
        reference: &str,
        policy: &MlxRemoteMetadataPolicy,
    ) -> Result<MlxCatalogResolutionReport, MlxCatalogError> {
        let parsed = parse_reference(reference)?;
        match parsed.kind {
            MlxReferenceKind::LocalPath => self.resolve_local_path(parsed),
            MlxReferenceKind::OllamaModel => self.resolve_ollama(parsed, policy),
            MlxReferenceKind::HuggingFaceRepo => self.resolve_hugging_face(parsed, policy),
        }
    }

    /// Opens a direct text runtime for the resolved source when available.
    pub fn open_text_runtime(
        &self,
        reference: &str,
        policy: &MlxRemoteMetadataPolicy,
    ) -> Result<MlxLmTextRuntime, MlxCatalogError> {
        let resolution = self.resolve(reference, policy)?;
        let Some(path) = resolution.direct_gguf_path else {
            return Err(MlxCatalogError::TextRuntimeUnavailable {
                reference: reference.to_string(),
                reason: match resolution.conversion_entrypoint {
                    MlxConversionEntryPoint::Unavailable { reason } => reason,
                    MlxConversionEntryPoint::HuggingFaceSnapshot { readiness, .. } => {
                        format!("resolved source still requires conversion: readiness={readiness:?}")
                    }
                    MlxConversionEntryPoint::DirectGguf { .. } => {
                        String::from("resolved source omitted a direct GGUF path")
                    }
                },
            });
        };
        if resolution.architecture_registration.is_none() {
            return Err(MlxCatalogError::TextRuntimeUnavailable {
                reference: reference.to_string(),
                reason: String::from("resolved source has no registered MLX architecture loader"),
            });
        }
        Ok(MlxLmTextRuntime::from_gguf_path(path)?)
    }

    /// Resolves one model reference and returns the direct text-package load report.
    pub fn load_text_report(
        &self,
        reference: &str,
        policy: &MlxRemoteMetadataPolicy,
    ) -> Result<MlxLmLoadReport, MlxCatalogError> {
        Ok(self.open_text_runtime(reference, policy)?.load_report())
    }

    /// Pulls one Ollama model reference into the configured local models root.
    pub fn pull_ollama_model(
        &self,
        reference: &str,
        options: OllamaRegistryPullOptions,
    ) -> Result<OllamaRegistryPullReport, MlxCatalogError> {
        Ok(OllamaRegistryClient::new(options)
            .pull_model(&self.roots.ollama_models_root, reference)?)
    }

    fn resolve_local_path(
        &self,
        parsed: MlxParsedReference,
    ) -> Result<MlxCatalogResolutionReport, MlxCatalogError> {
        let path = PathBuf::from(&parsed.value);
        let adapter = GgufDecoderAdapterLoader.load_path(&path)?;
        let architecture = adapter.family_metadata().architecture.clone();
        let registration = self
            .architecture_registry
            .resolve(architecture.as_str())
            .cloned();
        let notes = registration.as_ref().map_or_else(
            || vec![format!(
                "no MLX package loader is registered for architecture `{architecture}`"
            )],
            |_| Vec::new(),
        );
        Ok(MlxCatalogResolutionReport {
            raw_reference: parsed.raw.clone(),
            parsed_reference: parsed,
            source_kind: MlxResolutionSourceKind::LocalGgufPath,
            direct_gguf_path: Some(path.clone()),
            architecture: Some(architecture),
            architecture_registration: registration,
            remote_metadata: MlxRemoteMetadataReport::none(),
            conversion_entrypoint: MlxConversionEntryPoint::DirectGguf { path },
            notes,
            ollama_manifest: None,
            hugging_face_snapshot: None,
        })
    }

    fn resolve_ollama(
        &self,
        parsed: MlxParsedReference,
        policy: &MlxRemoteMetadataPolicy,
    ) -> Result<MlxCatalogResolutionReport, MlxCatalogError> {
        let catalog = OllamaModelCatalog::new(&self.roots.ollama_models_root);
        let manifest = catalog.resolve_model(parsed.value.as_str())?;
        let adapter = GgufDecoderAdapterLoader.load_ollama_manifest(
            &manifest,
            LocalBlobOpenOptions::default().with_read_preference(BlobReadPreference::PreferBuffered),
        )?;
        let architecture = adapter.family_metadata().architecture.clone();
        let registration = self
            .architecture_registry
            .resolve(architecture.as_str())
            .cloned();
        let remote_metadata = inspect_ollama_metadata(&manifest, policy);
        let mut notes = Vec::new();
        if remote_metadata.template_metadata.disposition == MlxRemoteMetadataDisposition::Refused {
            notes.push(String::from(
                "Ollama prompt/template layers are present but the direct text runtime continues to trust only GGUF-owned template metadata.",
            ));
        }
        if registration.is_none() {
            notes.push(format!(
                "no MLX package loader is registered for architecture `{architecture}`"
            ));
        }
        Ok(MlxCatalogResolutionReport {
            raw_reference: parsed.raw.clone(),
            parsed_reference: parsed,
            source_kind: MlxResolutionSourceKind::LocalOllamaManifest,
            direct_gguf_path: manifest
                .primary_model_layer()
                .map(|layer| layer.blob_path.clone()),
            architecture: Some(architecture),
            architecture_registration: registration,
            remote_metadata,
            conversion_entrypoint: manifest.primary_model_layer().map_or_else(
                || MlxConversionEntryPoint::Unavailable {
                    reason: String::from("ollama manifest carries no primary model blob"),
                },
                |layer| MlxConversionEntryPoint::DirectGguf {
                    path: layer.blob_path.clone(),
                },
            ),
            notes,
            ollama_manifest: Some(MlxOllamaManifestSummary {
                canonical_name: manifest.name.canonical_name(),
                short_name: manifest.name.display_shortest(),
                manifest_path: manifest.manifest_path.clone(),
                manifest_sha256: manifest.manifest_sha256.clone(),
                primary_model_blob: manifest
                    .primary_model_layer()
                    .map(|layer| layer.blob_path.clone()),
            }),
            hugging_face_snapshot: None,
        })
    }

    fn resolve_hugging_face(
        &self,
        parsed: MlxParsedReference,
        policy: &MlxRemoteMetadataPolicy,
    ) -> Result<MlxCatalogResolutionReport, MlxCatalogError> {
        let snapshots = self.discover_hugging_face_snapshots(parsed.value.as_str(), policy)?;
        let selected = match parsed.revision.as_deref() {
            Some(revision) => snapshots
                .into_iter()
                .find(|snapshot| snapshot.revision == revision)
                .ok_or_else(|| MlxCatalogError::MissingHuggingFaceSnapshots {
                    repo_id: parsed.value.clone(),
                    root: hugging_face_repo_root(
                        &self.roots.hugging_face_hub_root,
                        parsed.value.as_str(),
                    ),
                })?,
            None => snapshots
                .into_iter()
                .next()
                .ok_or_else(|| MlxCatalogError::MissingHuggingFaceSnapshots {
                    repo_id: parsed.value.clone(),
                    root: hugging_face_repo_root(
                        &self.roots.hugging_face_hub_root,
                        parsed.value.as_str(),
                    ),
                })?,
        };

        let mut notes = Vec::new();
        let mut architecture = selected.architecture.clone();
        let mut direct_gguf_path = selected.gguf_files.first().cloned();

        if selected.gguf_files.len() > 1 {
            notes.push(format!(
                "multiple GGUF files were discovered in snapshot `{}`; the first stable path was selected for direct load",
                selected.revision
            ));
        }

        if let Some(path) = direct_gguf_path.as_ref() {
            let adapter = GgufDecoderAdapterLoader.load_path(path)?;
            architecture = Some(adapter.family_metadata().architecture.clone());
        }

        let registration = architecture
            .as_deref()
            .and_then(|value| self.architecture_registry.resolve(value))
            .cloned();

        let conversion_entrypoint = if let Some(path) = direct_gguf_path.clone() {
            if selected.remote_metadata.has_refusal() {
                notes.push(String::from(
                    "remote template/processor metadata was refused; direct runtime still uses GGUF-owned metadata only.",
                ));
            }
            MlxConversionEntryPoint::DirectGguf { path }
        } else {
            let readiness = if selected.safetensors_files.is_empty() {
                MlxConversionReadiness::MissingWeights
            } else if registration.is_none() {
                MlxConversionReadiness::UnsupportedArchitecture
            } else if selected.remote_metadata.has_refusal() {
                MlxConversionReadiness::RefusedRemoteMetadata
            } else {
                MlxConversionReadiness::Ready
            };
            MlxConversionEntryPoint::HuggingFaceSnapshot {
                snapshot_path: selected.snapshot_path.clone(),
                safetensors_files: selected.safetensors_files.clone(),
                architecture: architecture.clone(),
                readiness,
                remote_metadata: selected.remote_metadata.clone(),
            }
        };

        if registration.is_none() && architecture.is_some() {
            notes.push(format!(
                "no MLX package loader is registered for architecture `{}`",
                architecture.as_deref().unwrap_or("unknown")
            ));
        }

        Ok(MlxCatalogResolutionReport {
            raw_reference: parsed.raw.clone(),
            parsed_reference: parsed,
            source_kind: MlxResolutionSourceKind::HuggingFaceSnapshot,
            direct_gguf_path: direct_gguf_path.take(),
            architecture,
            architecture_registration: registration,
            remote_metadata: selected.remote_metadata.clone(),
            conversion_entrypoint,
            notes,
            ollama_manifest: None,
            hugging_face_snapshot: Some(selected),
        })
    }
}

fn normalize_architecture_key(value: &str) -> String {
    let mut normalized = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch.to_ascii_lowercase());
        }
    }
    normalized
}

fn parse_reference(reference: &str) -> Result<MlxParsedReference, MlxCatalogError> {
    if let Some(value) = reference.strip_prefix("hf:") {
        let (repo_id, revision) = parse_hugging_face_reference(reference, value)?;
        return Ok(MlxParsedReference {
            raw: reference.to_string(),
            kind: MlxReferenceKind::HuggingFaceRepo,
            value: repo_id,
            revision,
        });
    }

    let path = PathBuf::from(reference);
    if path.exists() || reference.ends_with(".gguf") {
        return Ok(MlxParsedReference {
            raw: reference.to_string(),
            kind: MlxReferenceKind::LocalPath,
            value: path.display().to_string(),
            revision: None,
        });
    }

    Ok(MlxParsedReference {
        raw: reference.to_string(),
        kind: MlxReferenceKind::OllamaModel,
        value: reference.to_string(),
        revision: None,
    })
}

fn parse_hugging_face_reference(
    reference: &str,
    value: &str,
) -> Result<(String, Option<String>), MlxCatalogError> {
    let mut parts = value.split('@');
    let repo_id = parts.next().unwrap_or_default().trim();
    let revision = parts.next().map(str::trim).filter(|value| !value.is_empty());
    if repo_id.is_empty() || !repo_id.contains('/') {
        return Err(MlxCatalogError::InvalidHuggingFaceReference {
            reference: reference.to_string(),
            message: String::from("expected `hf:<owner>/<repo>` or `hf:<owner>/<repo>@<revision>`"),
        });
    }
    if parts.next().is_some() {
        return Err(MlxCatalogError::InvalidHuggingFaceReference {
            reference: reference.to_string(),
            message: String::from("only one `@revision` suffix is allowed"),
        });
    }
    Ok((repo_id.to_string(), revision.map(str::to_string)))
}

fn hugging_face_repo_root(root: &Path, repo_id: &str) -> PathBuf {
    let encoded = repo_id.replace('/', "--");
    root.join(format!("models--{encoded}"))
}

fn inspect_hugging_face_snapshot(
    repo_id: &str,
    snapshot_path: &Path,
    policy: &MlxRemoteMetadataPolicy,
) -> Result<HuggingFaceCacheSnapshot, MlxCatalogError> {
    let mut files = Vec::new();
    collect_files(snapshot_path, &mut files)?;
    files.sort();

    let mut gguf_files = Vec::new();
    let mut safetensors_files = Vec::new();
    let mut config_path = None;
    let mut processor_paths = Vec::new();
    let mut template_paths = Vec::new();
    let mut tokenizer_config_with_template = Vec::new();

    for path in files {
        let name = path.file_name().and_then(|value| value.to_str()).unwrap_or("");
        match name {
            "config.json" => config_path = Some(path.clone()),
            "processor_config.json" | "preprocessor_config.json" => processor_paths.push(path),
            "chat_template.json" | "chat_template.jinja" => template_paths.push(path),
            "tokenizer_config.json" => {
                if tokenizer_config_declares_chat_template(&path)? {
                    tokenizer_config_with_template.push(path);
                }
            }
            _ => {
                if path.extension().and_then(|value| value.to_str()) == Some("gguf") {
                    gguf_files.push(path);
                } else if path.extension().and_then(|value| value.to_str()) == Some("safetensors")
                {
                    safetensors_files.push(path);
                }
            }
        }
    }

    template_paths.extend(tokenizer_config_with_template);
    template_paths.sort();

    let architecture = config_path
        .as_ref()
        .map(|path| read_hugging_face_architecture(path.as_path()))
        .transpose()?
        .flatten();

    let modified_unix_seconds = fs::metadata(snapshot_path)
        .ok()
        .and_then(|metadata| metadata.modified().ok())
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs());

    Ok(HuggingFaceCacheSnapshot {
        repo_id: repo_id.to_string(),
        revision: snapshot_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_string(),
        snapshot_path: snapshot_path.to_path_buf(),
        modified_unix_seconds,
        gguf_files,
        safetensors_files,
        config_path,
        architecture,
        remote_metadata: MlxRemoteMetadataReport {
            processor_metadata: assess_local_metadata_paths(
                processor_paths.as_slice(),
                policy.processor_metadata,
                "processor metadata",
            )?,
            template_metadata: assess_local_metadata_paths(
                template_paths.as_slice(),
                policy.template_metadata,
                "template metadata",
            )?,
        },
    })
}

fn collect_files(root: &Path, files: &mut Vec<PathBuf>) -> Result<(), std::io::Error> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, files)?;
        } else if path.is_file() {
            files.push(path);
        }
    }
    Ok(())
}

fn tokenizer_config_declares_chat_template(path: &Path) -> Result<bool, MlxCatalogError> {
    let bytes = fs::read(path)?;
    let json: Value = serde_json::from_slice(&bytes)?;
    Ok(json.get("chat_template").is_some())
}

fn read_hugging_face_architecture(path: &Path) -> Result<Option<String>, MlxCatalogError> {
    let bytes = fs::read(path)?;
    let json: Value = serde_json::from_slice(&bytes)?;
    if let Some(value) = json.get("model_type").and_then(Value::as_str) {
        return Ok(Some(value.to_string()));
    }
    if let Some(value) = json
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|values| values.first())
        .and_then(Value::as_str)
    {
        return Ok(Some(value.to_string()));
    }
    if let Some(value) = json
        .get("text_config")
        .and_then(|value| value.get("model_type"))
        .and_then(Value::as_str)
    {
        return Ok(Some(value.to_string()));
    }
    Ok(None)
}

fn assess_local_metadata_paths(
    paths: &[PathBuf],
    trust_mode: MlxMetadataTrustMode,
    label: &str,
) -> Result<MlxRemoteMetadataAssessment, MlxCatalogError> {
    if paths.is_empty() {
        return Ok(MlxRemoteMetadataAssessment::absent());
    }
    let mut observations = paths
        .iter()
        .map(|path| {
            Ok(MlxRemoteMetadataObservation {
                path: path.clone(),
                digest: hex::encode(Sha256::digest(fs::read(path)?)),
            })
        })
        .collect::<Result<Vec<_>, std::io::Error>>()?;
    observations.sort_by(|left, right| left.path.cmp(&right.path));

    Ok(match trust_mode {
        MlxMetadataTrustMode::Refuse => MlxRemoteMetadataAssessment {
            disposition: MlxRemoteMetadataDisposition::Refused,
            observations,
            refusal_reason: Some(format!(
                "{label} is only present in remote-derived cache files and current policy refuses trusting it",
            )),
        },
        MlxMetadataTrustMode::AllowDigestBoundLocalCache => MlxRemoteMetadataAssessment {
            disposition: MlxRemoteMetadataDisposition::AllowedDigestBoundLocalCache,
            observations,
            refusal_reason: None,
        },
    })
}

fn inspect_ollama_metadata(
    manifest: &OllamaManifest,
    policy: &MlxRemoteMetadataPolicy,
) -> MlxRemoteMetadataReport {
    let template_layers = manifest
        .layers
        .iter()
        .filter(|layer| {
            matches!(
                layer.kind,
                OllamaLayerKind::Template
                    | OllamaLayerKind::Prompt
                    | OllamaLayerKind::System
                    | OllamaLayerKind::Messages
            )
        })
        .map(|layer| MlxRemoteMetadataObservation {
            path: layer.blob_path.clone(),
            digest: layer.digest.clone(),
        })
        .collect::<Vec<_>>();
    let template_metadata = if template_layers.is_empty() {
        MlxRemoteMetadataAssessment::absent()
    } else {
        match policy.template_metadata {
            MlxMetadataTrustMode::Refuse => MlxRemoteMetadataAssessment {
                disposition: MlxRemoteMetadataDisposition::Refused,
                observations: template_layers,
                refusal_reason: Some(String::from(
                    "ollama prompt/template layers are remote-derived metadata and current policy refuses trusting them for MLX package behavior",
                )),
            },
            MlxMetadataTrustMode::AllowDigestBoundLocalCache => MlxRemoteMetadataAssessment {
                disposition: MlxRemoteMetadataDisposition::AllowedDigestBoundLocalCache,
                observations: template_layers,
                refusal_reason: None,
            },
        }
    };
    MlxRemoteMetadataReport {
        processor_metadata: MlxRemoteMetadataAssessment::absent(),
        template_metadata,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MlxCatalogRoots, MlxCatalogWorkspace, MlxConversionEntryPoint, MlxConversionReadiness,
        MlxMetadataTrustMode, MlxReferenceKind, MlxRemoteMetadataPolicy,
        default_hugging_face_hub_root, default_ollama_models_root, hugging_face_repo_root,
    };
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use psionic_catalog::{ollama_blob_path, ollama_manifest_path};
    use psionic_models::{GgufMetadataValue, GgufTensorType, golden_prompt_fixture};
    use sha2::Digest;
    use tempfile::tempdir;

    #[test]
    fn default_roots_follow_expected_local_conventions() {
        assert!(default_ollama_models_root().ends_with("models"));
        assert!(default_hugging_face_hub_root().ends_with("hub"));
    }

    #[test]
    fn architecture_registry_and_direct_path_resolution_open_text_runtime()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let gguf_path = write_qwen_fixture(temp.path())?;
        let workspace = MlxCatalogWorkspace::new(MlxCatalogRoots {
            ollama_models_root: temp.path().join("ollama"),
            hugging_face_hub_root: temp.path().join("hf"),
        });

        let resolution = workspace.resolve(
            &gguf_path.display().to_string(),
            &MlxRemoteMetadataPolicy::default(),
        )?;
        assert_eq!(resolution.parsed_reference.kind, MlxReferenceKind::LocalPath);
        assert_eq!(resolution.architecture.as_deref(), Some("qwen2"));
        assert_eq!(
            resolution
                .architecture_registration
                .as_ref()
                .map(|value| value.model_family.as_str()),
            Some("qwen")
        );
        assert!(matches!(
            resolution.conversion_entrypoint,
            MlxConversionEntryPoint::DirectGguf { .. }
        ));

        let load_report = workspace.load_text_report(
            &gguf_path.display().to_string(),
            &MlxRemoteMetadataPolicy::default(),
        )?;
        assert_eq!(load_report.descriptor.model.family, "qwen");
        Ok(())
    }

    #[test]
    fn local_ollama_resolution_reports_refused_template_metadata_but_still_opens_text_runtime()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let models_root = temp.path().join("ollama");
        write_ollama_qwen_model(&models_root, "qwen2", Some("{{ .Prompt }}"))?;
        let workspace = MlxCatalogWorkspace::new(MlxCatalogRoots {
            ollama_models_root: models_root.clone(),
            hugging_face_hub_root: temp.path().join("hf"),
        });

        let resolution = workspace.resolve("qwen2", &MlxRemoteMetadataPolicy::default())?;
        assert_eq!(resolution.parsed_reference.kind, MlxReferenceKind::OllamaModel);
        assert!(resolution.ollama_manifest.is_some());
        assert_eq!(resolution.architecture.as_deref(), Some("qwen2"));
        assert_eq!(
            resolution.remote_metadata.template_metadata.refusal_reason.as_deref(),
            Some(
                "ollama prompt/template layers are remote-derived metadata and current policy refuses trusting them for MLX package behavior"
            )
        );

        let load_report = workspace.load_text_report("qwen2", &MlxRemoteMetadataPolicy::default())?;
        assert_eq!(load_report.descriptor.model.family, "qwen");
        Ok(())
    }

    #[test]
    fn hugging_face_snapshot_discovery_reports_direct_gguf_and_refused_metadata()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let repo_id = "mlx-community/tiny-qwen";
        let snapshot_path = write_hugging_face_snapshot(
            &temp.path().join("hub"),
            repo_id,
            "rev-1",
            true,
            true,
            true,
        )?;
        let workspace = MlxCatalogWorkspace::new(MlxCatalogRoots {
            ollama_models_root: temp.path().join("ollama"),
            hugging_face_hub_root: temp.path().join("hub"),
        });

        let snapshots = workspace
            .discover_hugging_face_snapshots(repo_id, &MlxRemoteMetadataPolicy::default())?;
        let snapshot = snapshots.first().expect("snapshot");
        assert_eq!(snapshot.snapshot_path, snapshot_path);
        assert_eq!(snapshot.architecture.as_deref(), Some("qwen2"));
        assert_eq!(snapshot.gguf_files.len(), 1);
        assert_eq!(
            snapshot.remote_metadata.processor_metadata.disposition,
            super::MlxRemoteMetadataDisposition::Refused
        );
        assert_eq!(
            snapshot.remote_metadata.template_metadata.disposition,
            super::MlxRemoteMetadataDisposition::Refused
        );

        let resolution = workspace.resolve(
            &format!("hf:{repo_id}"),
            &MlxRemoteMetadataPolicy::default(),
        )?;
        assert!(matches!(
            resolution.conversion_entrypoint,
            MlxConversionEntryPoint::DirectGguf { .. }
        ));
        Ok(())
    }

    #[test]
    fn hugging_face_conversion_entrypoint_honors_metadata_policy()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let repo_id = "mlx-community/tiny-safetensors";
        write_hugging_face_snapshot(
            &temp.path().join("hub"),
            repo_id,
            "rev-2",
            false,
            true,
            true,
        )?;
        let workspace = MlxCatalogWorkspace::new(MlxCatalogRoots {
            ollama_models_root: temp.path().join("ollama"),
            hugging_face_hub_root: temp.path().join("hub"),
        });

        let refused = workspace.resolve(
            &format!("hf:{repo_id}"),
            &MlxRemoteMetadataPolicy::default(),
        )?;
        let MlxConversionEntryPoint::HuggingFaceSnapshot { readiness, .. } =
            refused.conversion_entrypoint
        else {
            panic!("expected Hugging Face conversion entrypoint");
        };
        assert_eq!(readiness, MlxConversionReadiness::RefusedRemoteMetadata);

        let allowed = workspace.resolve(
            &format!("hf:{repo_id}"),
            &MlxRemoteMetadataPolicy {
                processor_metadata: MlxMetadataTrustMode::AllowDigestBoundLocalCache,
                template_metadata: MlxMetadataTrustMode::AllowDigestBoundLocalCache,
            },
        )?;
        let MlxConversionEntryPoint::HuggingFaceSnapshot {
            readiness,
            remote_metadata,
            ..
        } = allowed.conversion_entrypoint
        else {
            panic!("expected Hugging Face conversion entrypoint");
        };
        assert_eq!(readiness, MlxConversionReadiness::Ready);
        assert_eq!(
            remote_metadata.template_metadata.disposition,
            super::MlxRemoteMetadataDisposition::AllowedDigestBoundLocalCache
        );
        Ok(())
    }

    fn write_hugging_face_snapshot(
        hub_root: &Path,
        repo_id: &str,
        revision: &str,
        include_gguf: bool,
        include_processor: bool,
        include_template: bool,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let repo_root = hugging_face_repo_root(hub_root, repo_id);
        let snapshot_path = repo_root.join("snapshots").join(revision);
        fs::create_dir_all(&snapshot_path)?;
        fs::write(
            snapshot_path.join("config.json"),
            r#"{"model_type":"qwen2","architectures":["Qwen2ForCausalLM"]}"#,
        )?;
        if include_gguf {
            write_qwen_fixture(&snapshot_path)?;
        } else {
            fs::write(snapshot_path.join("model.safetensors"), b"safetensors-fixture")?;
        }
        if include_processor {
            fs::write(
                snapshot_path.join("processor_config.json"),
                r#"{"processor_class":"Qwen2Processor"}"#,
            )?;
        }
        if include_template {
            fs::write(
                snapshot_path.join("tokenizer_config.json"),
                r#"{"chat_template":"{{ messages }}"}"#,
            )?;
        }
        Ok(snapshot_path)
    }

    fn write_ollama_qwen_model(
        models_root: &Path,
        name: &str,
        prompt_template: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let gguf_path = write_qwen_fixture(models_root)?;
        let gguf_bytes = fs::read(&gguf_path)?;
        let digest = format!("sha256:{}", hex::encode(sha2::Sha256::digest(gguf_bytes.as_slice())));
        let blob_path = ollama_blob_path(models_root, digest.as_str())?;
        if let Some(parent) = blob_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&blob_path, &gguf_bytes)?;

        let mut layers = vec![format!(
            r#"{{"mediaType":"application/vnd.ollama.image.model","digest":"{digest}","size":{}}}"#,
            gguf_bytes.len()
        )];
        if let Some(prompt_template) = prompt_template {
            let template_bytes = prompt_template.as_bytes();
            let template_digest = format!(
                "sha256:{}",
                hex::encode(sha2::Sha256::digest(template_bytes))
            );
            let template_blob_path = ollama_blob_path(models_root, template_digest.as_str())?;
            if let Some(parent) = template_blob_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&template_blob_path, template_bytes)?;
            layers.push(format!(
                r#"{{"mediaType":"application/vnd.ollama.image.template","digest":"{template_digest}","size":{}}}"#,
                template_bytes.len()
            ));
        }

        let manifest_path = ollama_manifest_path(
            models_root,
            &psionic_catalog::OllamaModelName::parse(name)?,
        );
        if let Some(parent) = manifest_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(
            &manifest_path,
            format!(
                r#"{{"schemaVersion":2,"mediaType":"application/vnd.docker.distribution.manifest.v2+json","layers":[{}]}}"#,
                layers.join(",")
            ),
        )?;
        Ok(())
    }

    fn write_qwen_fixture(root: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
        fs::create_dir_all(root)?;
        let path = root.join("tiny_qwen2.gguf");
        let template = golden_prompt_fixture("qwen2")
            .and_then(|fixture| fixture.template_variant("qwen2.default"))
            .and_then(|variant| variant.raw_template)
            .expect("qwen2 raw template fixture");
        write_test_gguf(
            &path,
            qwen2_metadata("Tiny Qwen2", Some(template), 128).as_slice(),
            dense_decoder_tensors(true, 2, 3).as_slice(),
        )?;
        Ok(path)
    }

    fn qwen2_metadata(
        name: &str,
        chat_template: Option<&str>,
        context_length: u32,
    ) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = vec![
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
                    GgufMetadataValue::String(String::from("psionic")),
                    GgufMetadataValue::String(String::from("agent")),
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
        ];
        if let Some(chat_template) = chat_template {
            metadata.push((
                String::from("tokenizer.chat_template"),
                GgufMetadataValue::String(chat_template.to_string()),
            ));
        }
        metadata
    }

    fn dense_decoder_tensors(
        include_qkv_bias: bool,
        hello_token_index: usize,
        world_token_index: usize,
    ) -> Vec<TestGgufTensor> {
        let mut tensors = vec![
            dense_tensor(
                "token_embd.weight",
                vec![6, 4],
                token_embedding_values(hello_token_index),
            ),
            dense_tensor("output_norm.weight", vec![4], vec![1.0, 1.0, 1.0, 1.0]),
            dense_tensor(
                "output.weight",
                vec![6, 4],
                output_values(world_token_index),
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
        ];
        if include_qkv_bias {
            tensors.push(dense_tensor("blk.0.attn_q.bias", vec![4], vec![0.0; 4]));
            tensors.push(dense_tensor("blk.0.attn_k.bias", vec![2], vec![0.0; 2]));
            tensors.push(dense_tensor("blk.0.attn_v.bias", vec![2], vec![0.0; 2]));
        }
        tensors
    }

    fn token_embedding_values(hello_token_index: usize) -> Vec<f32> {
        let mut values = vec![0.0; 6 * 4];
        values[hello_token_index.saturating_mul(4)] = 2.0;
        values
    }

    fn output_values(world_token_index: usize) -> Vec<f32> {
        let mut values = vec![0.0; 6 * 4];
        values[world_token_index.saturating_mul(4)] = 1.0;
        values
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
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
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
        let alignment = metadata
            .iter()
            .find(|(key, _)| key == "general.alignment")
            .and_then(|(_, value)| match value {
                GgufMetadataValue::U64(value) => Some(*value as usize),
                GgufMetadataValue::U32(value) => Some(*value as usize),
                _ => None,
            })
            .unwrap_or(32)
            .max(1);

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
                for element in values {
                    push_gguf_array_value(bytes, element)?;
                }
            }
        }
        Ok(())
    }

    fn push_gguf_array_value(
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
            GgufMetadataValue::Array(_) => panic!("nested test arrays are unsupported"),
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
