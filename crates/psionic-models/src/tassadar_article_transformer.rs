use std::collections::BTreeMap;

use psionic_core::Shape;
use psionic_transformer::{
    ActivationKind, EncoderDecoderTransformer, EncoderDecoderTransformerConfig,
    EncoderDecoderTransformerError, LayerError, LayerNorm, Linear, MultiHeadAttention,
    PositionwiseFeedForward, TransformerDecoderLayer, TransformerEmbeddings,
    TransformerEncoderLayer, TransformerExecutionMode,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::ModelDescriptor;

/// Stable architecture classification for the canonical article Transformer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarArticleTransformerArchitectureVariant {
    AttentionIsAllYouNeedEncoderDecoder,
}

/// Supported embedding and logits-weight sharing strategies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarArticleTransformerEmbeddingStrategy {
    Unshared,
    DecoderInputOutputTied,
    SharedSourceTargetAndOutput,
}

/// Descriptor for the canonical owned article-Transformer model path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarArticleTransformerDescriptor {
    pub model: ModelDescriptor,
    pub source_paper_title: String,
    pub source_paper_ref: String,
    pub architecture_variant: TassadarArticleTransformerArchitectureVariant,
    pub paper_faithful: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub substitution_justification: Option<String>,
    pub embedding_strategy: TassadarArticleTransformerEmbeddingStrategy,
    pub config: EncoderDecoderTransformerConfig,
}

impl TassadarArticleTransformerDescriptor {
    /// Returns a stable descriptor digest.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        stable_digest(b"psionic_tassadar_article_transformer_descriptor|", self)
    }
}

/// One trainable parameter vector exposed by the bounded article route.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TassadarArticleTransformerParameterVector {
    pub parameter_id: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

/// Canonical article-route model wrapper owned by `psionic-models`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TassadarArticleTransformer {
    descriptor: TassadarArticleTransformerDescriptor,
    source_embedding_table: Vec<f32>,
    target_embedding_table: Vec<f32>,
    logits_projection_weight: Vec<f32>,
    logits_projection_bias: Vec<f32>,
    model: EncoderDecoderTransformer,
}

impl TassadarArticleTransformer {
    pub const MODEL_ID: &str = "tassadar-article-transformer-paper-faithful-v0";
    pub const MODEL_FAMILY: &str = "tassadar_article_transformer";
    pub const SOURCE_PAPER_TITLE: &str = "Attention Is All You Need";
    pub const SOURCE_PAPER_REF: &str =
        "~/code/alpha/tassadar/tassadar-research/papers/01-attention-is-all-you-need.pdf";
    pub const SOURCE_EMBEDDING_PARAMETER_ID: &str = "source_embedding_table";
    pub const TARGET_EMBEDDING_PARAMETER_ID: &str = "target_embedding_table";
    pub const SHARED_EMBEDDING_PARAMETER_ID: &str = "shared_embedding_table";
    pub const DECODER_OUTPUT_SHARED_PARAMETER_ID: &str = "decoder_output_shared_table";
    pub const LOGITS_PROJECTION_WEIGHT_PARAMETER_ID: &str = "logits_projection_weight";
    pub const LOGITS_PROJECTION_BIAS_PARAMETER_ID: &str = "logits_projection_bias";

    /// Returns a small canonical reference config used for closure tests.
    #[must_use]
    pub fn tiny_reference_config() -> EncoderDecoderTransformerConfig {
        EncoderDecoderTransformerConfig {
            source_vocab_size: 8,
            target_vocab_size: 8,
            hidden_size: 4,
            feed_forward_size: 8,
            head_count: 2,
            encoder_layer_count: 2,
            decoder_layer_count: 2,
            max_source_positions: 16,
            max_target_positions: 16,
            dropout_probability_bps: 0,
        }
    }

    /// Builds the canonical paper-faithful article route with one explicit
    /// embedding-sharing strategy.
    pub fn paper_faithful_reference(
        config: EncoderDecoderTransformerConfig,
        embedding_strategy: TassadarArticleTransformerEmbeddingStrategy,
    ) -> Result<Self, TassadarArticleTransformerError> {
        let source_embedding_table = seeded_values(
            "article_transformer.source_embedding_table",
            config.source_vocab_size * config.hidden_size,
            0.125,
        );
        let shared_decoder_table = seeded_values(
            "article_transformer.decoder_embedding_table",
            config.target_vocab_size * config.hidden_size,
            0.125,
        );

        let (target_embedding_table, logits_projection_weight) = match embedding_strategy {
            TassadarArticleTransformerEmbeddingStrategy::Unshared => (
                seeded_values(
                    "article_transformer.target_embedding_table",
                    config.target_vocab_size * config.hidden_size,
                    0.125,
                ),
                seeded_values(
                    "article_transformer.logits_projection_weight",
                    config.target_vocab_size * config.hidden_size,
                    0.08,
                ),
            ),
            TassadarArticleTransformerEmbeddingStrategy::DecoderInputOutputTied => (
                shared_decoder_table.clone(),
                shared_decoder_table,
            ),
            TassadarArticleTransformerEmbeddingStrategy::SharedSourceTargetAndOutput => {
                if config.source_vocab_size != config.target_vocab_size {
                    return Err(TassadarArticleTransformerError::InvalidConfiguration {
                        component: "tassadar_article_transformer.embedding_strategy",
                        message: format!(
                            "shared source/target/output weights require matching vocab sizes, found source_vocab_size={} target_vocab_size={}",
                            config.source_vocab_size, config.target_vocab_size
                        ),
                    });
                }
                (source_embedding_table.clone(), source_embedding_table.clone())
            }
        };
        let logits_projection_bias = vec![0.0; config.target_vocab_size];
        Self::from_weight_parts(
            config,
            embedding_strategy,
            source_embedding_table,
            target_embedding_table,
            logits_projection_weight,
            logits_projection_bias,
        )
    }

    /// Returns a small canonical paper-faithful route used by closure reports.
    pub fn canonical_reference() -> Result<Self, TassadarArticleTransformerError> {
        Self::paper_faithful_reference(
            Self::tiny_reference_config(),
            TassadarArticleTransformerEmbeddingStrategy::SharedSourceTargetAndOutput,
        )
    }

    /// Returns the public descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &TassadarArticleTransformerDescriptor {
        &self.descriptor
    }

    /// Returns the embedding-sharing strategy.
    #[must_use]
    pub const fn embedding_strategy(&self) -> TassadarArticleTransformerEmbeddingStrategy {
        self.descriptor.embedding_strategy
    }

    /// Returns the source embedding table.
    #[must_use]
    pub fn source_embedding_table(&self) -> &[f32] {
        &self.source_embedding_table
    }

    /// Returns the target embedding table.
    #[must_use]
    pub fn target_embedding_table(&self) -> &[f32] {
        &self.target_embedding_table
    }

    /// Returns the logits projection weight.
    #[must_use]
    pub fn logits_projection_weight(&self) -> &[f32] {
        &self.logits_projection_weight
    }

    /// Returns the logits projection bias.
    #[must_use]
    pub fn logits_projection_bias(&self) -> &[f32] {
        &self.logits_projection_bias
    }

    /// Returns the bounded trainable parameter vectors for the current sharing strategy.
    #[must_use]
    pub fn trainable_parameter_vectors(&self) -> Vec<TassadarArticleTransformerParameterVector> {
        let config = &self.descriptor.config;
        let mut parameters = match self.embedding_strategy() {
            TassadarArticleTransformerEmbeddingStrategy::Unshared => vec![
                TassadarArticleTransformerParameterVector {
                    parameter_id: String::from(Self::SOURCE_EMBEDDING_PARAMETER_ID),
                    shape: vec![config.source_vocab_size, config.hidden_size],
                    values: self.source_embedding_table.clone(),
                },
                TassadarArticleTransformerParameterVector {
                    parameter_id: String::from(Self::TARGET_EMBEDDING_PARAMETER_ID),
                    shape: vec![config.target_vocab_size, config.hidden_size],
                    values: self.target_embedding_table.clone(),
                },
                TassadarArticleTransformerParameterVector {
                    parameter_id: String::from(Self::LOGITS_PROJECTION_WEIGHT_PARAMETER_ID),
                    shape: vec![config.target_vocab_size, config.hidden_size],
                    values: self.logits_projection_weight.clone(),
                },
            ],
            TassadarArticleTransformerEmbeddingStrategy::DecoderInputOutputTied => vec![
                TassadarArticleTransformerParameterVector {
                    parameter_id: String::from(Self::SOURCE_EMBEDDING_PARAMETER_ID),
                    shape: vec![config.source_vocab_size, config.hidden_size],
                    values: self.source_embedding_table.clone(),
                },
                TassadarArticleTransformerParameterVector {
                    parameter_id: String::from(Self::DECODER_OUTPUT_SHARED_PARAMETER_ID),
                    shape: vec![config.target_vocab_size, config.hidden_size],
                    values: self.target_embedding_table.clone(),
                },
            ],
            TassadarArticleTransformerEmbeddingStrategy::SharedSourceTargetAndOutput => vec![
                TassadarArticleTransformerParameterVector {
                    parameter_id: String::from(Self::SHARED_EMBEDDING_PARAMETER_ID),
                    shape: vec![config.source_vocab_size, config.hidden_size],
                    values: self.source_embedding_table.clone(),
                },
            ],
        };
        parameters.push(TassadarArticleTransformerParameterVector {
            parameter_id: String::from(Self::LOGITS_PROJECTION_BIAS_PARAMETER_ID),
            shape: vec![config.target_vocab_size],
            values: self.logits_projection_bias.clone(),
        });
        parameters
    }

    /// Returns the stable digest over the bounded trainable state.
    #[must_use]
    pub fn trainable_parameter_digest(&self) -> String {
        stable_digest(
            b"psionic_tassadar_article_transformer_trainable_parameters|",
            &self.trainable_parameter_vectors(),
        )
    }

    /// Returns one bounded trainable parameter vector by identifier.
    #[must_use]
    pub fn trainable_parameter_vector(
        &self,
        parameter_id: &str,
    ) -> Option<TassadarArticleTransformerParameterVector> {
        self.trainable_parameter_vectors()
            .into_iter()
            .find(|parameter| parameter.parameter_id == parameter_id)
    }

    /// Rebuilds the model with bounded trainable-parameter overrides.
    pub fn with_parameter_overrides(
        &self,
        overrides: &BTreeMap<String, Vec<f32>>,
    ) -> Result<Self, TassadarArticleTransformerError> {
        let mut source_embedding_table = self.source_embedding_table.clone();
        let mut target_embedding_table = self.target_embedding_table.clone();
        let mut logits_projection_weight = self.logits_projection_weight.clone();
        let mut logits_projection_bias = self.logits_projection_bias.clone();

        let apply_override =
            |parameter_id: &str,
             expected_len: usize,
             target: &mut Vec<f32>,
             overrides: &BTreeMap<String, Vec<f32>>|
             -> Result<(), TassadarArticleTransformerError> {
                if let Some(values) = overrides.get(parameter_id) {
                    if values.len() != expected_len {
                        return Err(TassadarArticleTransformerError::InvalidParameterShape {
                            parameter_id: String::from(parameter_id),
                            expected_len,
                            actual_len: values.len(),
                        });
                    }
                    *target = values.clone();
                }
                Ok(())
            };

        match self.embedding_strategy() {
            TassadarArticleTransformerEmbeddingStrategy::Unshared => {
                apply_override(
                    Self::SOURCE_EMBEDDING_PARAMETER_ID,
                    source_embedding_table.len(),
                    &mut source_embedding_table,
                    overrides,
                )?;
                apply_override(
                    Self::TARGET_EMBEDDING_PARAMETER_ID,
                    target_embedding_table.len(),
                    &mut target_embedding_table,
                    overrides,
                )?;
                apply_override(
                    Self::LOGITS_PROJECTION_WEIGHT_PARAMETER_ID,
                    logits_projection_weight.len(),
                    &mut logits_projection_weight,
                    overrides,
                )?;
            }
            TassadarArticleTransformerEmbeddingStrategy::DecoderInputOutputTied => {
                apply_override(
                    Self::SOURCE_EMBEDDING_PARAMETER_ID,
                    source_embedding_table.len(),
                    &mut source_embedding_table,
                    overrides,
                )?;
                if let Some(values) = overrides.get(Self::DECODER_OUTPUT_SHARED_PARAMETER_ID) {
                    if values.len() != target_embedding_table.len() {
                        return Err(TassadarArticleTransformerError::InvalidParameterShape {
                            parameter_id: String::from(Self::DECODER_OUTPUT_SHARED_PARAMETER_ID),
                            expected_len: target_embedding_table.len(),
                            actual_len: values.len(),
                        });
                    }
                    target_embedding_table = values.clone();
                    logits_projection_weight = values.clone();
                }
            }
            TassadarArticleTransformerEmbeddingStrategy::SharedSourceTargetAndOutput => {
                if let Some(values) = overrides.get(Self::SHARED_EMBEDDING_PARAMETER_ID) {
                    if values.len() != source_embedding_table.len() {
                        return Err(TassadarArticleTransformerError::InvalidParameterShape {
                            parameter_id: String::from(Self::SHARED_EMBEDDING_PARAMETER_ID),
                            expected_len: source_embedding_table.len(),
                            actual_len: values.len(),
                        });
                    }
                    source_embedding_table = values.clone();
                    target_embedding_table = values.clone();
                    logits_projection_weight = values.clone();
                }
            }
        }

        apply_override(
            Self::LOGITS_PROJECTION_BIAS_PARAMETER_ID,
            logits_projection_bias.len(),
            &mut logits_projection_bias,
            overrides,
        )?;

        for parameter_id in overrides.keys() {
            if self
                .trainable_parameter_vector(parameter_id.as_str())
                .is_none()
            {
                return Err(TassadarArticleTransformerError::UnknownParameter {
                    parameter_id: parameter_id.clone(),
                });
            }
        }

        Self::from_weight_parts(
            self.descriptor.config.clone(),
            self.embedding_strategy(),
            source_embedding_table,
            target_embedding_table,
            logits_projection_weight,
            logits_projection_bias,
        )
    }

    /// Runs a paper-faithful encoder-decoder forward pass.
    pub fn forward(
        &self,
        source_index_shape: Shape,
        source_token_ids: &[usize],
        target_index_shape: Shape,
        target_token_ids: &[usize],
        mode: TransformerExecutionMode,
    ) -> Result<
        psionic_transformer::EncoderDecoderTransformerForwardOutput,
        TassadarArticleTransformerError,
    > {
        Ok(self.model.forward(
            source_index_shape,
            source_token_ids,
            target_index_shape,
            target_token_ids,
            mode,
        )?)
    }

    fn from_weight_parts(
        config: EncoderDecoderTransformerConfig,
        embedding_strategy: TassadarArticleTransformerEmbeddingStrategy,
        source_embedding_table: Vec<f32>,
        target_embedding_table: Vec<f32>,
        logits_projection_weight: Vec<f32>,
        logits_projection_bias: Vec<f32>,
    ) -> Result<Self, TassadarArticleTransformerError> {
        let dropout_probability = config.dropout_probability();
        let source_embeddings = TransformerEmbeddings::from_f32_table(
            "tassadar.article_transformer.source_embeddings",
            config.source_vocab_size,
            config.hidden_size,
            config.max_source_positions,
            source_embedding_table.clone(),
            dropout_probability,
        )?;
        let target_embeddings = TransformerEmbeddings::from_f32_table(
            "tassadar.article_transformer.target_embeddings",
            config.target_vocab_size,
            config.hidden_size,
            config.max_target_positions,
            target_embedding_table.clone(),
            dropout_probability,
        )?;
        let encoder_layers = (0..config.encoder_layer_count)
            .map(|layer_index| build_encoder_layer(&config, layer_index))
            .collect::<Result<Vec<_>, _>>()?;
        let decoder_layers = (0..config.decoder_layer_count)
            .map(|layer_index| build_decoder_layer(&config, layer_index))
            .collect::<Result<Vec<_>, _>>()?;
        let logits_projection = Linear::from_f32_parts(
            "tassadar.article_transformer.logits_projection",
            config.hidden_size,
            config.target_vocab_size,
            logits_projection_weight.clone(),
            Some(logits_projection_bias.clone()),
        )?;
        let model = EncoderDecoderTransformer::from_components(
            config.clone(),
            source_embeddings,
            target_embeddings,
            encoder_layers,
            decoder_layers,
            logits_projection,
        )?;
        let descriptor = TassadarArticleTransformerDescriptor {
            model: ModelDescriptor::new(Self::MODEL_ID, Self::MODEL_FAMILY, "v0"),
            source_paper_title: String::from(Self::SOURCE_PAPER_TITLE),
            source_paper_ref: String::from(Self::SOURCE_PAPER_REF),
            architecture_variant:
                TassadarArticleTransformerArchitectureVariant::AttentionIsAllYouNeedEncoderDecoder,
            paper_faithful: true,
            substitution_justification: None,
            embedding_strategy,
            config,
        };
        Ok(Self {
            descriptor,
            source_embedding_table,
            target_embedding_table,
            logits_projection_weight,
            logits_projection_bias,
            model,
        })
    }
}

/// Article-route wrapper construction or forward-pass failure.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum TassadarArticleTransformerError {
    #[error(transparent)]
    EncoderDecoder(#[from] EncoderDecoderTransformerError),
    #[error(transparent)]
    Block(#[from] psionic_transformer::TransformerBlockError),
    #[error(transparent)]
    Layer(#[from] LayerError),
    #[error("component `{component}` invalid configuration: {message}")]
    InvalidConfiguration {
        component: &'static str,
        message: String,
    },
    #[error("unknown bounded trainable parameter `{parameter_id}`")]
    UnknownParameter { parameter_id: String },
    #[error(
        "bounded trainable parameter `{parameter_id}` expected length {expected_len}, found {actual_len}"
    )]
    InvalidParameterShape {
        parameter_id: String,
        expected_len: usize,
        actual_len: usize,
    },
}

fn build_encoder_layer(
    config: &EncoderDecoderTransformerConfig,
    layer_index: usize,
) -> Result<TransformerEncoderLayer, TassadarArticleTransformerError> {
    let module_id = format!("tassadar.article_transformer.encoder_layers.{layer_index}");
    Ok(TransformerEncoderLayer::from_components(
        build_attention(
            format!("{module_id}.self_attention"),
            config.hidden_size,
            config.head_count,
        )?,
        LayerNorm::new(
            format!("{module_id}.self_attention_norm"),
            config.hidden_size,
            1e-5,
        )?,
        build_feed_forward(
            format!("{module_id}.feed_forward"),
            config.hidden_size,
            config.feed_forward_size,
        )?,
        LayerNorm::new(
            format!("{module_id}.feed_forward_norm"),
            config.hidden_size,
            1e-5,
        )?,
    )?)
}

fn build_decoder_layer(
    config: &EncoderDecoderTransformerConfig,
    layer_index: usize,
) -> Result<TransformerDecoderLayer, TassadarArticleTransformerError> {
    let module_id = format!("tassadar.article_transformer.decoder_layers.{layer_index}");
    Ok(TransformerDecoderLayer::from_components(
        build_attention(
            format!("{module_id}.self_attention"),
            config.hidden_size,
            config.head_count,
        )?,
        LayerNorm::new(
            format!("{module_id}.self_attention_norm"),
            config.hidden_size,
            1e-5,
        )?,
        build_attention(
            format!("{module_id}.cross_attention"),
            config.hidden_size,
            config.head_count,
        )?,
        LayerNorm::new(
            format!("{module_id}.cross_attention_norm"),
            config.hidden_size,
            1e-5,
        )?,
        build_feed_forward(
            format!("{module_id}.feed_forward"),
            config.hidden_size,
            config.feed_forward_size,
        )?,
        LayerNorm::new(
            format!("{module_id}.feed_forward_norm"),
            config.hidden_size,
            1e-5,
        )?,
    )?)
}

fn build_attention(
    module_id: String,
    hidden_size: usize,
    head_count: usize,
) -> Result<MultiHeadAttention, TassadarArticleTransformerError> {
    let weight_len = hidden_size * hidden_size;
    Ok(MultiHeadAttention::from_f32_parts(
        module_id.clone(),
        hidden_size,
        head_count,
        seeded_values(&format!("{module_id}.query_projection"), weight_len, 0.06),
        Some(seeded_values(
            &format!("{module_id}.query_bias"),
            hidden_size,
            0.01,
        )),
        seeded_values(&format!("{module_id}.key_projection"), weight_len, 0.06),
        Some(seeded_values(
            &format!("{module_id}.key_bias"),
            hidden_size,
            0.01,
        )),
        seeded_values(&format!("{module_id}.value_projection"), weight_len, 0.06),
        Some(seeded_values(
            &format!("{module_id}.value_bias"),
            hidden_size,
            0.01,
        )),
        seeded_values(&format!("{module_id}.output_projection"), weight_len, 0.06),
        Some(seeded_values(
            &format!("{module_id}.output_bias"),
            hidden_size,
            0.01,
        )),
        0.0,
    )?)
}

fn build_feed_forward(
    module_id: String,
    hidden_size: usize,
    feed_forward_size: usize,
) -> Result<PositionwiseFeedForward, TassadarArticleTransformerError> {
    Ok(PositionwiseFeedForward::from_f32_parts(
        module_id.clone(),
        hidden_size,
        feed_forward_size,
        ActivationKind::Relu,
        seeded_values(
            &format!("{module_id}.feed_forward_input"),
            hidden_size * feed_forward_size,
            0.05,
        ),
        Some(seeded_values(
            &format!("{module_id}.feed_forward_input_bias"),
            feed_forward_size,
            0.01,
        )),
        seeded_values(
            &format!("{module_id}.feed_forward_output"),
            hidden_size * feed_forward_size,
            0.05,
        ),
        Some(seeded_values(
            &format!("{module_id}.feed_forward_output_bias"),
            hidden_size,
            0.01,
        )),
        0.0,
    )?)
}

fn seeded_values(label: &str, len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let mut hasher = Sha256::new();
            hasher.update(b"psionic_tassadar_article_transformer_seed|");
            hasher.update(label.as_bytes());
            hasher.update(index.to_le_bytes());
            let digest = hasher.finalize();
            let sample = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]);
            let centered = (sample as f32 / u32::MAX as f32) * 2.0 - 1.0;
            centered * scale
        })
        .collect()
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
        TassadarArticleTransformer, TassadarArticleTransformerArchitectureVariant,
        TassadarArticleTransformerEmbeddingStrategy, TassadarArticleTransformerError,
    };
    use psionic_core::Shape;
    use psionic_transformer::TransformerExecutionMode;

    #[test]
    fn article_transformer_descriptor_is_paper_faithful() -> Result<(), TassadarArticleTransformerError> {
        let model = TassadarArticleTransformer::canonical_reference()?;
        let descriptor = model.descriptor();

        assert_eq!(
            descriptor.model.model_id,
            TassadarArticleTransformer::MODEL_ID
        );
        assert_eq!(
            descriptor.architecture_variant,
            TassadarArticleTransformerArchitectureVariant::AttentionIsAllYouNeedEncoderDecoder
        );
        assert!(descriptor.paper_faithful);
        assert!(descriptor.substitution_justification.is_none());
        assert_eq!(
            descriptor.source_paper_title,
            TassadarArticleTransformer::SOURCE_PAPER_TITLE
        );
        Ok(())
    }

    #[test]
    fn article_transformer_supports_tied_and_unshared_embedding_options(
    ) -> Result<(), TassadarArticleTransformerError> {
        let config = TassadarArticleTransformer::tiny_reference_config();
        let unshared = TassadarArticleTransformer::paper_faithful_reference(
            config.clone(),
            TassadarArticleTransformerEmbeddingStrategy::Unshared,
        )?;
        let tied = TassadarArticleTransformer::paper_faithful_reference(
            config.clone(),
            TassadarArticleTransformerEmbeddingStrategy::DecoderInputOutputTied,
        )?;
        let shared = TassadarArticleTransformer::paper_faithful_reference(
            config,
            TassadarArticleTransformerEmbeddingStrategy::SharedSourceTargetAndOutput,
        )?;

        assert_ne!(
            unshared.target_embedding_table(),
            unshared.logits_projection_weight()
        );
        assert_eq!(tied.target_embedding_table(), tied.logits_projection_weight());
        assert_eq!(shared.source_embedding_table(), shared.target_embedding_table());
        assert_eq!(shared.target_embedding_table(), shared.logits_projection_weight());
        Ok(())
    }

    #[test]
    fn article_transformer_forward_emits_encoder_decoder_logits() -> Result<(), TassadarArticleTransformerError> {
        let model = TassadarArticleTransformer::canonical_reference()?;
        let output = model.forward(
            Shape::new(vec![1, 2]),
            &[0, 1],
            Shape::new(vec![1, 3]),
            &[0, 1, 2],
            TransformerExecutionMode::Eval,
        )?;

        assert_eq!(output.encoder_hidden_state.dims(), &[1, 2, 4]);
        assert_eq!(output.decoder_hidden_state.dims(), &[1, 3, 4]);
        assert_eq!(output.logits.dims(), &[1, 3, 8]);
        assert_eq!(output.encoder_layer_outputs.len(), 2);
        assert_eq!(output.decoder_layer_outputs.len(), 2);
        Ok(())
    }
}
