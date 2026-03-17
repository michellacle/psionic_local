//! Bounded MLX-style multimodal package above Psionic-native prompt and serving
//! surfaces.

use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
};

use psionic_models::{PromptMessage, PromptMessageRole};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Human-readable crate ownership summary.
pub const CRATE_ROLE: &str =
    "bounded MLX-style multimodal package with processor registries and served-request planning above Psionic-native prompt and serving surfaces";

/// Served endpoint targeted by one multimodal request plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxVlmServedEndpoint {
    /// OpenAI-compatible `/v1/responses` request shape.
    Responses,
    /// OpenAI-compatible `/v1/chat/completions` request shape.
    ChatCompletions,
}

/// Supported multimodal media kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxVlmMediaKind {
    /// Still image input.
    Image,
    /// Audio input.
    Audio,
    /// Video input.
    Video,
}

impl MlxVlmMediaKind {
    fn label(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
        }
    }
}

/// Caller-visible source kind for one media attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxVlmMediaSourceKind {
    /// Remote URL reference.
    Url,
    /// Local file path.
    LocalPath,
    /// Inlined data URL.
    DataUrl,
}

impl MlxVlmMediaSourceKind {
    fn label(self) -> &'static str {
        match self {
            Self::Url => "url",
            Self::LocalPath => "local_path",
            Self::DataUrl => "data_url",
        }
    }
}

/// Source reference for one media attachment.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MlxVlmMediaSource {
    /// Remote URL reference.
    Url {
        /// Stable remote URL string.
        url: String,
    },
    /// Local file path.
    LocalPath {
        /// Stable local path.
        path: PathBuf,
    },
    /// Inlined data URL.
    DataUrl {
        /// Raw data URL string.
        data_url: String,
    },
}

impl MlxVlmMediaSource {
    fn kind(&self) -> MlxVlmMediaSourceKind {
        match self {
            Self::Url { .. } => MlxVlmMediaSourceKind::Url,
            Self::LocalPath { .. } => MlxVlmMediaSourceKind::LocalPath,
            Self::DataUrl { .. } => MlxVlmMediaSourceKind::DataUrl,
        }
    }

    fn stable_identifier(&self) -> String {
        match self {
            Self::Url { url } => url.clone(),
            Self::LocalPath { path } => path.display().to_string(),
            Self::DataUrl { data_url } => data_url.clone(),
        }
    }

    fn digest(&self) -> Result<String, MlxVlmError> {
        let bytes = match self {
            Self::Url { url } => url.as_bytes().to_vec(),
            Self::LocalPath { path } => fs::read(path)?,
            Self::DataUrl { data_url } => data_url.as_bytes().to_vec(),
        };
        Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
    }
}

/// One media attachment declared by the caller.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxVlmMediaPart {
    /// Media source.
    pub source: MlxVlmMediaSource,
    /// MIME type when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Optional quality/detail hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl MlxVlmMediaPart {
    /// Creates one media part.
    #[must_use]
    pub fn new(source: MlxVlmMediaSource) -> Self {
        Self {
            source,
            mime_type: None,
            detail: None,
        }
    }

    /// Attaches one MIME type hint.
    #[must_use]
    pub fn with_mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }

    /// Attaches one quality/detail hint.
    #[must_use]
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

/// One multimodal input part in an OpenAI-compatible request shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MlxVlmInputPart {
    /// Plain input text.
    InputText {
        /// Text content.
        text: String,
    },
    /// Image input.
    InputImage {
        /// Media attachment.
        image: MlxVlmMediaPart,
    },
    /// Audio input.
    InputAudio {
        /// Media attachment.
        audio: MlxVlmMediaPart,
    },
    /// Video input.
    InputVideo {
        /// Media attachment.
        video: MlxVlmMediaPart,
    },
}

impl MlxVlmInputPart {
    /// Creates one text input part.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::InputText { text: text.into() }
    }

    fn media_kind(&self) -> Option<MlxVlmMediaKind> {
        match self {
            Self::InputText { .. } => None,
            Self::InputImage { .. } => Some(MlxVlmMediaKind::Image),
            Self::InputAudio { .. } => Some(MlxVlmMediaKind::Audio),
            Self::InputVideo { .. } => Some(MlxVlmMediaKind::Video),
        }
    }

    fn media(&self) -> Option<&MlxVlmMediaPart> {
        match self {
            Self::InputText { .. } => None,
            Self::InputImage { image } => Some(image),
            Self::InputAudio { audio } => Some(audio),
            Self::InputVideo { video } => Some(video),
        }
    }
}

/// Supported multimodal message roles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxVlmMessageRole {
    /// System/developer instruction.
    System,
    /// User input.
    User,
    /// Assistant output.
    Assistant,
    /// Tool result.
    Tool,
}

impl MlxVlmMessageRole {
    fn prompt_role(self) -> PromptMessageRole {
        match self {
            Self::System => PromptMessageRole::System,
            Self::User => PromptMessageRole::User,
            Self::Assistant => PromptMessageRole::Assistant,
            Self::Tool => PromptMessageRole::Tool,
        }
    }
}

/// One multimodal message composed of text and media parts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxVlmMessage {
    /// Message role.
    pub role: MlxVlmMessageRole,
    /// Ordered content parts.
    pub content: Vec<MlxVlmInputPart>,
}

impl MlxVlmMessage {
    /// Creates one multimodal message.
    #[must_use]
    pub fn new(role: MlxVlmMessageRole, content: Vec<MlxVlmInputPart>) -> Self {
        Self { role, content }
    }
}

/// Current processor strategy for the package-owned multimodal lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxVlmProjectionMode {
    /// Media is projected into typed prompt markers plus attachment receipts.
    PromptProjectionOnly,
}

/// One builtin multimodal processor registration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxVlmProcessorRegistration {
    /// Canonical family label.
    pub canonical_family: String,
    /// Accepted aliases.
    pub aliases: Vec<String>,
    /// Media kinds accepted by the family.
    pub accepted_media_kinds: Vec<MlxVlmMediaKind>,
    /// Current processor strategy.
    pub projection_mode: MlxVlmProjectionMode,
    /// Served endpoints exposed through the shared text server.
    pub served_endpoints: Vec<MlxVlmServedEndpoint>,
}

/// Bounded processor registry for multimodal MLX-style packages.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxVlmProcessorRegistry {
    registrations: BTreeMap<String, MlxVlmProcessorRegistration>,
}

impl Default for MlxVlmProcessorRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

impl MlxVlmProcessorRegistry {
    /// Returns the builtin registry.
    #[must_use]
    pub fn builtin() -> Self {
        let mut registry = Self {
            registrations: BTreeMap::new(),
        };
        registry.register(MlxVlmProcessorRegistration {
            canonical_family: String::from("llava"),
            aliases: vec![String::from("llava"), String::from("llava_vision")],
            accepted_media_kinds: vec![MlxVlmMediaKind::Image],
            projection_mode: MlxVlmProjectionMode::PromptProjectionOnly,
            served_endpoints: vec![
                MlxVlmServedEndpoint::Responses,
                MlxVlmServedEndpoint::ChatCompletions,
            ],
        });
        registry.register(MlxVlmProcessorRegistration {
            canonical_family: String::from("qwen2_vl"),
            aliases: vec![
                String::from("qwen2_vl"),
                String::from("qwen2-vl"),
                String::from("qwen_vl"),
            ],
            accepted_media_kinds: vec![MlxVlmMediaKind::Image, MlxVlmMediaKind::Video],
            projection_mode: MlxVlmProjectionMode::PromptProjectionOnly,
            served_endpoints: vec![
                MlxVlmServedEndpoint::Responses,
                MlxVlmServedEndpoint::ChatCompletions,
            ],
        });
        registry.register(MlxVlmProcessorRegistration {
            canonical_family: String::from("omni"),
            aliases: vec![
                String::from("omni"),
                String::from("gpt_oss_omni"),
                String::from("omni_vlm"),
            ],
            accepted_media_kinds: vec![
                MlxVlmMediaKind::Image,
                MlxVlmMediaKind::Audio,
                MlxVlmMediaKind::Video,
            ],
            projection_mode: MlxVlmProjectionMode::PromptProjectionOnly,
            served_endpoints: vec![
                MlxVlmServedEndpoint::Responses,
                MlxVlmServedEndpoint::ChatCompletions,
            ],
        });
        registry
    }

    /// Registers one processor entry and its aliases.
    pub fn register(&mut self, registration: MlxVlmProcessorRegistration) {
        let canonical = normalize_family_key(registration.canonical_family.as_str());
        self.registrations
            .insert(canonical, registration.clone());
        for alias in &registration.aliases {
            self.registrations
                .insert(normalize_family_key(alias.as_str()), registration.clone());
        }
    }

    /// Resolves one family or alias.
    #[must_use]
    pub fn resolve(&self, family: &str) -> Option<&MlxVlmProcessorRegistration> {
        self.registrations.get(&normalize_family_key(family))
    }
}

/// One projected multimodal attachment in prompt-projection mode.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxVlmAttachmentReceipt {
    /// Stable attachment index.
    pub index: usize,
    /// Media kind.
    pub media_kind: MlxVlmMediaKind,
    /// Source kind.
    pub source_kind: MlxVlmMediaSourceKind,
    /// Stable source identifier.
    pub source_identifier: String,
    /// Stable attachment digest.
    pub digest: String,
    /// MIME type when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Detail hint when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Prompt marker inserted into the translated text request.
    pub prompt_marker: String,
}

/// One multimodal prompt-projection report.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxVlmProjectionReport {
    /// Processor registration used for shaping.
    pub processor: MlxVlmProcessorRegistration,
    /// Projected prompt messages consumed by the shared text lane.
    pub projected_messages: Vec<PromptMessage>,
    /// Stable attachment receipts.
    pub attachments: Vec<MlxVlmAttachmentReceipt>,
    /// Honest bounded notes for the current package.
    pub notes: Vec<String>,
}

/// One served request plan translated into the shared text-serving lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxVlmServePlan {
    /// Stable schema version.
    pub schema_version: u32,
    /// Model reference to send to `psionic-mlx-serve`.
    pub model_reference: String,
    /// Target endpoint on the shared server.
    pub target_endpoint: MlxVlmServedEndpoint,
    /// Multimodal projection report.
    pub projection: MlxVlmProjectionReport,
    /// Text-only request JSON for the shared server.
    pub translated_request_json: serde_json::Value,
}

/// Errors returned by the bounded multimodal package.
#[derive(Debug, Error)]
pub enum MlxVlmError {
    /// The caller requested an unknown processor family.
    #[error("unknown MLX multimodal processor family `{family}`")]
    UnknownProcessorFamily {
        /// Requested family string.
        family: String,
    },
    /// The selected family does not accept one media kind.
    #[error("processor family `{family}` does not accept `{media_kind}` inputs")]
    UnsupportedMediaKind {
        /// Canonical family label.
        family: String,
        /// Unsupported media kind label.
        media_kind: String,
    },
    /// Reading one local media file failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Serializing one request plan failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

/// Bounded multimodal workspace.
#[derive(Clone, Debug)]
pub struct MlxVlmWorkspace {
    registry: MlxVlmProcessorRegistry,
}

impl Default for MlxVlmWorkspace {
    fn default() -> Self {
        Self::new(MlxVlmProcessorRegistry::builtin())
    }
}

impl MlxVlmWorkspace {
    /// Creates one multimodal workspace over the provided registry.
    #[must_use]
    pub fn new(registry: MlxVlmProcessorRegistry) -> Self {
        Self { registry }
    }

    /// Returns the active processor registry.
    #[must_use]
    pub fn registry(&self) -> &MlxVlmProcessorRegistry {
        &self.registry
    }

    /// Projects one multimodal conversation into the shared text lane.
    pub fn project_messages(
        &self,
        family: &str,
        messages: &[MlxVlmMessage],
    ) -> Result<MlxVlmProjectionReport, MlxVlmError> {
        let processor = self
            .registry
            .resolve(family)
            .cloned()
            .ok_or_else(|| MlxVlmError::UnknownProcessorFamily {
                family: family.to_string(),
            })?;
        let mut attachments = Vec::new();
        let mut projected_messages = Vec::with_capacity(messages.len());

        for message in messages {
            let mut segments = Vec::new();
            for part in &message.content {
                match part {
                    MlxVlmInputPart::InputText { text } => segments.push(text.clone()),
                    media_part => {
                        let media_kind = media_part
                            .media_kind()
                            .expect("media part should carry media kind");
                        if !processor.accepted_media_kinds.contains(&media_kind) {
                            return Err(MlxVlmError::UnsupportedMediaKind {
                                family: processor.canonical_family.clone(),
                                media_kind: media_kind.label().to_string(),
                            });
                        }
                        let media = media_part.media().expect("media part payload");
                        let receipt =
                            build_attachment_receipt(attachments.len(), media_kind, media)?;
                        segments.push(receipt.prompt_marker.clone());
                        attachments.push(receipt);
                    }
                }
            }
            projected_messages.push(PromptMessage::new(
                message.role.prompt_role(),
                segments.join("\n\n"),
            ));
        }

        Ok(MlxVlmProjectionReport {
            processor,
            projected_messages,
            attachments,
            notes: vec![String::from(
                "Current MLX multimodal support is prompt-projection only: media attachments are carried as typed prompt markers plus digest-bound receipts for the shared text-serving lane, not as a claimed native image/audio/video encoder.",
            )],
        })
    }

    /// Plans one served multimodal request for the shared text-serving lane.
    pub fn plan_request(
        &self,
        family: &str,
        model_reference: &str,
        endpoint: MlxVlmServedEndpoint,
        messages: &[MlxVlmMessage],
    ) -> Result<MlxVlmServePlan, MlxVlmError> {
        let projection = self.project_messages(family, messages)?;
        let translated_request_json = match endpoint {
            MlxVlmServedEndpoint::Responses => responses_request_json(
                model_reference,
                projection.projected_messages.as_slice(),
            )?,
            MlxVlmServedEndpoint::ChatCompletions => chat_request_json(
                model_reference,
                projection.projected_messages.as_slice(),
            )?,
        };
        Ok(MlxVlmServePlan {
            schema_version: 1,
            model_reference: model_reference.to_string(),
            target_endpoint: endpoint,
            projection,
            translated_request_json,
        })
    }
}

fn build_attachment_receipt(
    index: usize,
    media_kind: MlxVlmMediaKind,
    media: &MlxVlmMediaPart,
) -> Result<MlxVlmAttachmentReceipt, MlxVlmError> {
    let digest = media.source.digest()?;
    let source_kind = media.source.kind();
    let prompt_marker = format!(
        "<psionic_media kind=\"{}\" index=\"{}\" source=\"{}\" digest=\"{}\"{}{} />",
        media_kind.label(),
        index,
        source_kind.label(),
        digest,
        media
            .mime_type
            .as_ref()
            .map(|value| format!(" mime=\"{value}\""))
            .unwrap_or_default(),
        media
            .detail
            .as_ref()
            .map(|value| format!(" detail=\"{value}\""))
            .unwrap_or_default(),
    );
    Ok(MlxVlmAttachmentReceipt {
        index,
        media_kind,
        source_kind,
        source_identifier: media.source.stable_identifier(),
        digest,
        mime_type: media.mime_type.clone(),
        detail: media.detail.clone(),
        prompt_marker,
    })
}

fn responses_request_json(
    model_reference: &str,
    messages: &[PromptMessage],
) -> Result<serde_json::Value, MlxVlmError> {
    let input = messages
        .iter()
        .map(|message| {
            serde_json::json!({
                "role": prompt_role_label(message.role),
                "content": [
                    {
                        "type": "input_text",
                        "text": message.content,
                    }
                ]
            })
        })
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "model": model_reference,
        "input": input,
    }))
}

fn chat_request_json(
    model_reference: &str,
    messages: &[PromptMessage],
) -> Result<serde_json::Value, MlxVlmError> {
    let projected = messages
        .iter()
        .map(|message| {
            serde_json::json!({
                "role": prompt_role_label(message.role),
                "content": message.content,
            })
        })
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "model": model_reference,
        "messages": projected,
    }))
}

fn prompt_role_label(role: PromptMessageRole) -> &'static str {
    match role {
        PromptMessageRole::System => "system",
        PromptMessageRole::Developer => "developer",
        PromptMessageRole::User => "user",
        PromptMessageRole::Assistant => "assistant",
        PromptMessageRole::Tool => "tool",
    }
}

fn normalize_family_key(value: &str) -> String {
    value
        .chars()
        .filter(|character| *character != '_' && *character != '-')
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        MlxVlmInputPart, MlxVlmMediaKind, MlxVlmMediaPart, MlxVlmMediaSource, MlxVlmMessage,
        MlxVlmMessageRole, MlxVlmServedEndpoint, MlxVlmWorkspace,
    };
    use std::fs;

    #[test]
    fn builtin_registry_resolves_aliases_and_media_coverage() {
        let workspace = MlxVlmWorkspace::default();
        let llava = workspace
            .registry()
            .resolve("llava_vision")
            .expect("llava alias");
        assert_eq!(llava.canonical_family, "llava");
        assert_eq!(llava.accepted_media_kinds, vec![MlxVlmMediaKind::Image]);

        let qwen = workspace
            .registry()
            .resolve("qwen2-vl")
            .expect("qwen alias");
        assert_eq!(
            qwen.accepted_media_kinds,
            vec![MlxVlmMediaKind::Image, MlxVlmMediaKind::Video]
        );
    }

    #[test]
    fn projection_tracks_image_audio_and_video_receipts() -> Result<(), Box<dyn std::error::Error>>
    {
        let temp = tempfile::tempdir()?;
        let local_video = temp.path().join("clip.mp4");
        fs::write(&local_video, b"video-bytes")?;

        let workspace = MlxVlmWorkspace::default();
        let report = workspace.project_messages(
            "omni",
            &[MlxVlmMessage::new(
                MlxVlmMessageRole::User,
                vec![
                    MlxVlmInputPart::text("Summarize these attachments."),
                    MlxVlmInputPart::InputImage {
                        image: MlxVlmMediaPart::new(MlxVlmMediaSource::Url {
                            url: String::from("https://example.com/image.png"),
                        })
                        .with_mime_type("image/png")
                        .with_detail("high"),
                    },
                    MlxVlmInputPart::InputAudio {
                        audio: MlxVlmMediaPart::new(MlxVlmMediaSource::DataUrl {
                            data_url: String::from("data:audio/wav;base64,AAAA"),
                        })
                        .with_mime_type("audio/wav"),
                    },
                    MlxVlmInputPart::InputVideo {
                        video: MlxVlmMediaPart::new(MlxVlmMediaSource::LocalPath {
                            path: local_video.clone(),
                        })
                        .with_mime_type("video/mp4"),
                    },
                ],
            )],
        )?;

        assert_eq!(report.attachments.len(), 3);
        assert!(report.projected_messages[0].content.contains("psionic_media"));
        assert_eq!(report.attachments[0].media_kind, MlxVlmMediaKind::Image);
        assert_eq!(report.attachments[1].media_kind, MlxVlmMediaKind::Audio);
        assert_eq!(report.attachments[2].media_kind, MlxVlmMediaKind::Video);
        assert!(report.attachments[2].source_identifier.ends_with("clip.mp4"));
        Ok(())
    }

    #[test]
    fn plan_request_emits_translated_responses_json() -> Result<(), Box<dyn std::error::Error>> {
        let workspace = MlxVlmWorkspace::default();
        let plan = workspace.plan_request(
            "llava",
            "ollama:llava",
            MlxVlmServedEndpoint::Responses,
            &[MlxVlmMessage::new(
                MlxVlmMessageRole::User,
                vec![
                    MlxVlmInputPart::InputImage {
                        image: MlxVlmMediaPart::new(MlxVlmMediaSource::Url {
                            url: String::from("https://example.com/cat.png"),
                        }),
                    },
                    MlxVlmInputPart::text("What is in the image?"),
                ],
            )],
        )?;

        assert_eq!(plan.target_endpoint, MlxVlmServedEndpoint::Responses);
        assert_eq!(plan.translated_request_json["model"], "ollama:llava");
        assert_eq!(plan.translated_request_json["input"][0]["role"], "user");
        assert_eq!(
            plan.translated_request_json["input"][0]["content"][0]["type"],
            "input_text"
        );
        assert!(
            plan.translated_request_json["input"][0]["content"][0]["text"]
                .as_str()
                .expect("text")
                .contains("psionic_media")
        );
        Ok(())
    }
}
