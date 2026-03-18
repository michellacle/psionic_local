use psionic_runtime::TassadarExecutorDecodeMode;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Stable routed product identifier for planner-owned executor delegation.
pub const TASSADAR_PLANNER_EXECUTOR_ROUTE_PRODUCT_ID: &str = "psionic.planner_executor_route";
/// Stable delegated executor product identifier.
pub const TASSADAR_EXECUTOR_TRACE_PRODUCT_ID: &str = "psionic.executor_trace";

/// Route posture advertised for one requested decode mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarPlannerExecutorRoutePosture {
    /// Route can stay on a direct executor path for this decode mode.
    DirectGuaranteed,
    /// Route can accept the decode mode but may fall back at runtime.
    FallbackCapable,
}

/// Typed refusal reason surfaced during route negotiation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarPlannerExecutorRouteRefusalReason {
    UnsupportedProduct,
    UnknownModel,
    ProviderNotReady,
    BenchmarkGateMissing,
    DecodeModeUnsupported,
    RuntimeFallbackDisallowed,
    DirectDecodeRequired,
}

/// Public per-decode capability row for one planner / executor route surface.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPlannerExecutorDecodeCapability {
    /// Requested decode mode this row describes.
    pub requested_decode_mode: TassadarExecutorDecodeMode,
    /// Guaranteed-direct or explicit-fallback-capable posture.
    pub route_posture: TassadarPlannerExecutorRoutePosture,
    /// Stable benchmark report backing this route posture.
    pub benchmark_report_ref: String,
    /// Plain-language note describing the support boundary.
    pub note: String,
}

/// Routeable served capability descriptor for one benchmark-gated Tassadar lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPlannerExecutorRouteDescriptor {
    /// Stable route identifier.
    pub route_id: String,
    /// Planner-owned routed product identifier.
    pub product_id: String,
    /// Delegated exact executor product identifier.
    pub executor_product_id: String,
    /// Stable executor model identifier exposed by the route.
    pub model_id: String,
    /// Stable benchmark report backing the served route.
    pub benchmark_report_ref: String,
    /// Stable digest of the served workload capability matrix.
    pub workload_capability_digest: String,
    /// Explicit decode posture published for route negotiation.
    pub decode_capabilities: Vec<TassadarPlannerExecutorDecodeCapability>,
    /// Typed refusal reasons this route may emit during negotiation.
    pub refusal_reasons: Vec<TassadarPlannerExecutorRouteRefusalReason>,
    /// Plain-language note for the route surface.
    pub note: String,
    /// Stable digest over the full route descriptor.
    pub descriptor_digest: String,
}

impl TassadarPlannerExecutorRouteDescriptor {
    /// Creates one routeable served descriptor.
    #[must_use]
    pub fn new(
        route_id: impl Into<String>,
        model_id: impl Into<String>,
        benchmark_report_ref: impl Into<String>,
        workload_capability_digest: impl Into<String>,
        mut decode_capabilities: Vec<TassadarPlannerExecutorDecodeCapability>,
        mut refusal_reasons: Vec<TassadarPlannerExecutorRouteRefusalReason>,
        note: impl Into<String>,
    ) -> Self {
        decode_capabilities.sort_by_key(|capability| capability.requested_decode_mode.as_str());
        refusal_reasons.sort_by_key(|reason| serde_json::to_string(reason).unwrap_or_default());
        refusal_reasons.dedup();
        let mut descriptor = Self {
            route_id: route_id.into(),
            product_id: String::from(TASSADAR_PLANNER_EXECUTOR_ROUTE_PRODUCT_ID),
            executor_product_id: String::from(TASSADAR_EXECUTOR_TRACE_PRODUCT_ID),
            model_id: model_id.into(),
            benchmark_report_ref: benchmark_report_ref.into(),
            workload_capability_digest: workload_capability_digest.into(),
            decode_capabilities,
            refusal_reasons,
            note: note.into(),
            descriptor_digest: String::new(),
        };
        descriptor.descriptor_digest = stable_digest(
            b"psionic_tassadar_planner_executor_route_descriptor|",
            &descriptor,
        );
        descriptor
    }
}

/// Route candidate contributed by one provider / worker pair.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPlannerExecutorRouteCandidate {
    /// Stable provider identifier.
    pub provider_id: String,
    /// Stable worker identifier.
    pub worker_id: String,
    /// Provider backend family label.
    pub backend_family: String,
    /// Whether the provider is ready for routing.
    pub ready: bool,
    /// Public route descriptor exported by the served lane.
    pub route_descriptor: TassadarPlannerExecutorRouteDescriptor,
}

impl TassadarPlannerExecutorRouteCandidate {
    /// Creates one route candidate.
    #[must_use]
    pub fn new(
        provider_id: impl Into<String>,
        worker_id: impl Into<String>,
        backend_family: impl Into<String>,
        ready: bool,
        route_descriptor: TassadarPlannerExecutorRouteDescriptor,
    ) -> Self {
        Self {
            provider_id: provider_id.into(),
            worker_id: worker_id.into(),
            backend_family: backend_family.into(),
            ready,
            route_descriptor,
        }
    }
}

/// One route-negotiation request coming from a planner-facing surface.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPlannerExecutorRouteNegotiationRequest {
    /// Stable request identifier.
    pub request_id: String,
    /// Routed product identifier.
    pub product_id: String,
    /// Optional explicit executor model request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_model_id: Option<String>,
    /// Requested decode mode.
    pub requested_decode_mode: TassadarExecutorDecodeMode,
    /// Whether runtime decode fallback is allowed.
    pub allow_runtime_decode_fallback: bool,
    /// Whether a guaranteed direct path is required.
    pub require_direct_decode: bool,
    /// Whether the route must be benchmark-gated.
    pub require_benchmark_gate: bool,
}

impl TassadarPlannerExecutorRouteNegotiationRequest {
    /// Creates a new route-negotiation request.
    #[must_use]
    pub fn new(
        request_id: impl Into<String>,
        requested_decode_mode: TassadarExecutorDecodeMode,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            product_id: String::from(TASSADAR_PLANNER_EXECUTOR_ROUTE_PRODUCT_ID),
            requested_model_id: None,
            requested_decode_mode,
            allow_runtime_decode_fallback: true,
            require_direct_decode: false,
            require_benchmark_gate: true,
        }
    }

    /// Pins negotiation to one explicit executor model.
    #[must_use]
    pub fn with_requested_model_id(mut self, requested_model_id: impl Into<String>) -> Self {
        self.requested_model_id = Some(requested_model_id.into());
        self
    }

    /// Disallows runtime decode fallback.
    #[must_use]
    pub fn disallow_runtime_decode_fallback(mut self) -> Self {
        self.allow_runtime_decode_fallback = false;
        self
    }

    /// Requires a guaranteed direct route posture.
    #[must_use]
    pub fn require_direct_decode(mut self) -> Self {
        self.require_direct_decode = true;
        self
    }
}

/// Final negotiated route state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarPlannerExecutorNegotiatedRouteState {
    Direct,
    Fallback,
}

/// Successful route-selection result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPlannerExecutorRouteSelection {
    /// Stable request identifier.
    pub request_id: String,
    /// Selected provider identifier.
    pub provider_id: String,
    /// Selected worker identifier.
    pub worker_id: String,
    /// Selected backend family.
    pub backend_family: String,
    /// Selected served route descriptor.
    pub route_descriptor: TassadarPlannerExecutorRouteDescriptor,
    /// Decode capability row used by negotiation.
    pub decode_capability: TassadarPlannerExecutorDecodeCapability,
    /// Direct or fallback-capable route state.
    pub route_state: TassadarPlannerExecutorNegotiatedRouteState,
    /// Plain-language negotiation detail.
    pub detail: String,
}

/// Typed refusal result for route negotiation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPlannerExecutorRouteRefusal {
    /// Stable request identifier.
    pub request_id: String,
    /// Typed refusal reason.
    pub refusal_reason: TassadarPlannerExecutorRouteRefusalReason,
    /// Considered route identifiers in stable order.
    pub considered_route_ids: Vec<String>,
    /// Plain-language refusal detail.
    pub detail: String,
}

/// Route-negotiation result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TassadarPlannerExecutorRouteNegotiationOutcome {
    Selected {
        selection: TassadarPlannerExecutorRouteSelection,
    },
    Refused {
        refusal: TassadarPlannerExecutorRouteRefusal,
    },
}

/// Negotiates one planner / executor route across served candidates.
#[must_use]
pub fn negotiate_tassadar_planner_executor_route(
    candidates: &[TassadarPlannerExecutorRouteCandidate],
    request: &TassadarPlannerExecutorRouteNegotiationRequest,
) -> TassadarPlannerExecutorRouteNegotiationOutcome {
    if request.product_id != TASSADAR_PLANNER_EXECUTOR_ROUTE_PRODUCT_ID {
        return TassadarPlannerExecutorRouteNegotiationOutcome::Refused {
            refusal: TassadarPlannerExecutorRouteRefusal {
                request_id: request.request_id.clone(),
                refusal_reason: TassadarPlannerExecutorRouteRefusalReason::UnsupportedProduct,
                considered_route_ids: route_ids(candidates),
                detail: format!(
                    "unsupported Tassadar planner route product `{}`",
                    request.product_id
                ),
            },
        };
    }

    let mut matching_model_exists = false;
    let mut ready_model_exists = false;
    let mut benchmark_gated_model_exists = false;
    let mut fallback_capable_exists = false;
    let mut decode_mode_exists = false;

    let mut direct_candidates = Vec::new();
    let mut fallback_candidates = Vec::new();

    for candidate in sorted_candidates(candidates) {
        if let Some(requested_model_id) = request.requested_model_id.as_deref()
            && candidate.route_descriptor.model_id != requested_model_id
        {
            continue;
        }
        matching_model_exists = true;

        if !candidate.ready {
            continue;
        }
        ready_model_exists = true;

        if request.require_benchmark_gate
            && candidate
                .route_descriptor
                .benchmark_report_ref
                .trim()
                .is_empty()
        {
            continue;
        }
        benchmark_gated_model_exists = true;

        let Some(decode_capability) = candidate
            .route_descriptor
            .decode_capabilities
            .iter()
            .find(|capability| capability.requested_decode_mode == request.requested_decode_mode)
            .cloned()
        else {
            continue;
        };
        decode_mode_exists = true;

        match decode_capability.route_posture {
            TassadarPlannerExecutorRoutePosture::DirectGuaranteed => {
                direct_candidates.push((candidate, decode_capability));
            }
            TassadarPlannerExecutorRoutePosture::FallbackCapable => {
                fallback_capable_exists = true;
                fallback_candidates.push((candidate, decode_capability));
            }
        }
    }

    if let Some((candidate, decode_capability)) = direct_candidates.into_iter().next() {
        return TassadarPlannerExecutorRouteNegotiationOutcome::Selected {
            selection: TassadarPlannerExecutorRouteSelection {
                request_id: request.request_id.clone(),
                provider_id: candidate.provider_id.clone(),
                worker_id: candidate.worker_id.clone(),
                backend_family: candidate.backend_family.clone(),
                route_descriptor: candidate.route_descriptor.clone(),
                decode_capability,
                route_state: TassadarPlannerExecutorNegotiatedRouteState::Direct,
                detail: format!(
                    "selected direct Tassadar planner / executor route `{}` on provider `{}` worker `{}`",
                    candidate.route_descriptor.route_id, candidate.provider_id, candidate.worker_id
                ),
            },
        };
    }

    if fallback_capable_exists {
        if request.require_direct_decode {
            return refusal(
                request,
                candidates,
                TassadarPlannerExecutorRouteRefusalReason::DirectDecodeRequired,
                format!(
                    "requested decode `{}` is only published as fallback-capable on the available Tassadar planner routes",
                    request.requested_decode_mode.as_str()
                ),
            );
        }
        if !request.allow_runtime_decode_fallback {
            return refusal(
                request,
                candidates,
                TassadarPlannerExecutorRouteRefusalReason::RuntimeFallbackDisallowed,
                format!(
                    "planner policy disallowed runtime decode fallback for `{}`",
                    request.requested_decode_mode.as_str()
                ),
            );
        }
        if let Some((candidate, decode_capability)) = fallback_candidates.into_iter().next() {
            return TassadarPlannerExecutorRouteNegotiationOutcome::Selected {
                selection: TassadarPlannerExecutorRouteSelection {
                    request_id: request.request_id.clone(),
                    provider_id: candidate.provider_id.clone(),
                    worker_id: candidate.worker_id.clone(),
                    backend_family: candidate.backend_family.clone(),
                    route_descriptor: candidate.route_descriptor.clone(),
                    decode_capability,
                    route_state: TassadarPlannerExecutorNegotiatedRouteState::Fallback,
                    detail: format!(
                        "selected fallback-capable Tassadar planner / executor route `{}` on provider `{}` worker `{}`",
                        candidate.route_descriptor.route_id,
                        candidate.provider_id,
                        candidate.worker_id
                    ),
                },
            };
        }
    }

    let (reason, detail) = if request.requested_model_id.is_some() && !matching_model_exists {
        (
            TassadarPlannerExecutorRouteRefusalReason::UnknownModel,
            format!(
                "no served Tassadar planner route matched requested model `{}`",
                request
                    .requested_model_id
                    .as_deref()
                    .unwrap_or("unknown_model")
            ),
        )
    } else if matching_model_exists && !ready_model_exists {
        (
            TassadarPlannerExecutorRouteRefusalReason::ProviderNotReady,
            String::from(
                "matching Tassadar planner routes exist but all providers are currently not ready",
            ),
        )
    } else if request.require_benchmark_gate && ready_model_exists && !benchmark_gated_model_exists
    {
        (
            TassadarPlannerExecutorRouteRefusalReason::BenchmarkGateMissing,
            String::from(
                "matching ready Tassadar planner routes were missing benchmark-gated evidence",
            ),
        )
    } else if matching_model_exists && benchmark_gated_model_exists && !decode_mode_exists {
        (
            TassadarPlannerExecutorRouteRefusalReason::DecodeModeUnsupported,
            format!(
                "no matching Tassadar planner route published decode support for `{}`",
                request.requested_decode_mode.as_str()
            ),
        )
    } else {
        (
            TassadarPlannerExecutorRouteRefusalReason::DecodeModeUnsupported,
            String::from("no eligible Tassadar planner route candidates remained after filtering"),
        )
    };
    refusal(request, candidates, reason, detail)
}

fn refusal(
    request: &TassadarPlannerExecutorRouteNegotiationRequest,
    candidates: &[TassadarPlannerExecutorRouteCandidate],
    refusal_reason: TassadarPlannerExecutorRouteRefusalReason,
    detail: String,
) -> TassadarPlannerExecutorRouteNegotiationOutcome {
    TassadarPlannerExecutorRouteNegotiationOutcome::Refused {
        refusal: TassadarPlannerExecutorRouteRefusal {
            request_id: request.request_id.clone(),
            refusal_reason,
            considered_route_ids: route_ids(candidates),
            detail,
        },
    }
}

fn sorted_candidates(
    candidates: &[TassadarPlannerExecutorRouteCandidate],
) -> Vec<&TassadarPlannerExecutorRouteCandidate> {
    let mut candidates = candidates.iter().collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        (
            left.provider_id.as_str(),
            left.worker_id.as_str(),
            left.route_descriptor.model_id.as_str(),
            left.route_descriptor.route_id.as_str(),
        )
            .cmp(&(
                right.provider_id.as_str(),
                right.worker_id.as_str(),
                right.route_descriptor.model_id.as_str(),
                right.route_descriptor.route_id.as_str(),
            ))
    });
    candidates
}

fn route_ids(candidates: &[TassadarPlannerExecutorRouteCandidate]) -> Vec<String> {
    let mut route_ids = candidates
        .iter()
        .map(|candidate| candidate.route_descriptor.route_id.clone())
        .collect::<Vec<_>>();
    route_ids.sort();
    route_ids.dedup();
    route_ids
}

fn stable_digest<T>(prefix: &[u8], value: &T) -> String
where
    T: Serialize,
{
    let encoded =
        serde_json::to_vec(value).expect("tassadar planner executor route should serialize");
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(encoded);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::{
        TassadarPlannerExecutorDecodeCapability, TassadarPlannerExecutorNegotiatedRouteState,
        TassadarPlannerExecutorRouteCandidate, TassadarPlannerExecutorRouteDescriptor,
        TassadarPlannerExecutorRouteNegotiationOutcome,
        TassadarPlannerExecutorRouteNegotiationRequest, TassadarPlannerExecutorRoutePosture,
        TassadarPlannerExecutorRouteRefusalReason, negotiate_tassadar_planner_executor_route,
    };
    use psionic_runtime::TassadarExecutorDecodeMode;

    fn route_descriptor(
        route_id: &str,
        model_id: &str,
        decode_capabilities: Vec<TassadarPlannerExecutorDecodeCapability>,
    ) -> TassadarPlannerExecutorRouteDescriptor {
        TassadarPlannerExecutorRouteDescriptor::new(
            route_id,
            model_id,
            "fixtures/tassadar/reports/tassadar_article_class_benchmark_report.json",
            "matrix-digest",
            decode_capabilities,
            vec![
                TassadarPlannerExecutorRouteRefusalReason::DecodeModeUnsupported,
                TassadarPlannerExecutorRouteRefusalReason::RuntimeFallbackDisallowed,
                TassadarPlannerExecutorRouteRefusalReason::DirectDecodeRequired,
            ],
            "test route descriptor",
        )
    }

    #[test]
    fn tassadar_route_negotiation_prefers_direct_route() {
        let candidates = vec![
            TassadarPlannerExecutorRouteCandidate::new(
                "provider-a",
                "worker-a",
                "psionic",
                true,
                route_descriptor(
                    "route-a",
                    "article-model",
                    vec![TassadarPlannerExecutorDecodeCapability {
                        requested_decode_mode: TassadarExecutorDecodeMode::ReferenceLinear,
                        route_posture: TassadarPlannerExecutorRoutePosture::DirectGuaranteed,
                        benchmark_report_ref: String::from(
                            "fixtures/tassadar/reports/tassadar_article_class_benchmark_report.json",
                        ),
                        note: String::from("direct dense floor"),
                    }],
                ),
            ),
            TassadarPlannerExecutorRouteCandidate::new(
                "provider-b",
                "worker-b",
                "psionic",
                true,
                route_descriptor(
                    "route-b",
                    "article-model",
                    vec![TassadarPlannerExecutorDecodeCapability {
                        requested_decode_mode: TassadarExecutorDecodeMode::ReferenceLinear,
                        route_posture: TassadarPlannerExecutorRoutePosture::FallbackCapable,
                        benchmark_report_ref: String::from(
                            "fixtures/tassadar/reports/tassadar_article_class_benchmark_report.json",
                        ),
                        note: String::from("fallback capable"),
                    }],
                ),
            ),
        ];

        let outcome = negotiate_tassadar_planner_executor_route(
            candidates.as_slice(),
            &TassadarPlannerExecutorRouteNegotiationRequest::new(
                "request-direct",
                TassadarExecutorDecodeMode::ReferenceLinear,
            )
            .with_requested_model_id("article-model"),
        );
        match outcome {
            TassadarPlannerExecutorRouteNegotiationOutcome::Selected { selection } => {
                assert_eq!(selection.provider_id, "provider-a");
                assert_eq!(
                    selection.route_state,
                    TassadarPlannerExecutorNegotiatedRouteState::Direct
                );
            }
            other => panic!("expected direct route selection, got {other:?}"),
        }
    }

    #[test]
    fn tassadar_route_negotiation_surfaces_fallback_when_allowed() {
        let candidates = vec![TassadarPlannerExecutorRouteCandidate::new(
            "provider-a",
            "worker-a",
            "psionic",
            true,
            route_descriptor(
                "route-a",
                "article-model",
                vec![TassadarPlannerExecutorDecodeCapability {
                    requested_decode_mode: TassadarExecutorDecodeMode::HullCache,
                    route_posture: TassadarPlannerExecutorRoutePosture::FallbackCapable,
                    benchmark_report_ref: String::from(
                        "fixtures/tassadar/reports/tassadar_article_class_benchmark_report.json",
                    ),
                    note: String::from("hull route may fall back"),
                }],
            ),
        )];

        let outcome = negotiate_tassadar_planner_executor_route(
            candidates.as_slice(),
            &TassadarPlannerExecutorRouteNegotiationRequest::new(
                "request-fallback",
                TassadarExecutorDecodeMode::HullCache,
            )
            .with_requested_model_id("article-model"),
        );
        match outcome {
            TassadarPlannerExecutorRouteNegotiationOutcome::Selected { selection } => {
                assert_eq!(
                    selection.route_state,
                    TassadarPlannerExecutorNegotiatedRouteState::Fallback
                );
            }
            other => panic!("expected fallback-capable route selection, got {other:?}"),
        }
    }

    #[test]
    fn tassadar_route_negotiation_refuses_when_direct_path_is_required() {
        let candidates = vec![TassadarPlannerExecutorRouteCandidate::new(
            "provider-a",
            "worker-a",
            "psionic",
            true,
            route_descriptor(
                "route-a",
                "article-model",
                vec![TassadarPlannerExecutorDecodeCapability {
                    requested_decode_mode: TassadarExecutorDecodeMode::SparseTopK,
                    route_posture: TassadarPlannerExecutorRoutePosture::FallbackCapable,
                    benchmark_report_ref: String::from(
                        "fixtures/tassadar/reports/tassadar_article_class_benchmark_report.json",
                    ),
                    note: String::from("sparse route may fall back"),
                }],
            ),
        )];

        let outcome = negotiate_tassadar_planner_executor_route(
            candidates.as_slice(),
            &TassadarPlannerExecutorRouteNegotiationRequest::new(
                "request-direct-required",
                TassadarExecutorDecodeMode::SparseTopK,
            )
            .with_requested_model_id("article-model")
            .require_direct_decode(),
        );
        match outcome {
            TassadarPlannerExecutorRouteNegotiationOutcome::Refused { refusal } => {
                assert_eq!(
                    refusal.refusal_reason,
                    TassadarPlannerExecutorRouteRefusalReason::DirectDecodeRequired
                );
            }
            other => panic!("expected direct-path refusal, got {other:?}"),
        }
    }
}
