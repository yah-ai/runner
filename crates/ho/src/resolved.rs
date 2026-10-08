//! `ResolvedEndpoint` (yah-runner-core's provider resolution output) →
//! [`OpenAiCompatConfig`]. Lives here, not in core, so core never depends on
//! a runner cell (R950-F6).

use crate::openai_compat::{OpenAiCompatConfig, Transport};
use yah_runner_core::resolver::{AuthMaterial, ResolvedEndpoint};

impl From<ResolvedEndpoint> for OpenAiCompatConfig {
    /// The per-endpoint defaults a resolved OpenAI-compat session starts
    /// with. Host policy (quota ledger, identity preamble, rate-limit hint)
    /// is left unset for the host to fill.
    fn from(resolved: ResolvedEndpoint) -> Self {
        let api_key = match resolved.auth {
            AuthMaterial::Bearer(k) => Some(k),
            AuthMaterial::None => None,
        };
        // Ollama-native NDJSON lives behind `/api/chat`; every other
        // OpenAI-compat endpoint we resolve speaks SSE.
        let transport = if resolved.endpoint.contains("/api/chat") {
            Transport::OllamaNdjson
        } else {
            Transport::Sse
        };
        // Per-engine context-window default for the resolver path —
        // matches the factory defaults so sessions started via the
        // bundled connection registry get the same anti-truncation
        // protection as ones started via `OpenAiCompatConfig::ollama_*`
        // directly (R169). Cloud endpoint advertises 32k; local Ollama
        // 8k; OpenAI-Sse leaves the field unset.
        let num_ctx = match transport {
            Transport::OllamaNdjson => {
                if resolved.endpoint.starts_with("https://ollama.com") {
                    Some(32768)
                } else {
                    Some(8192)
                }
            }
            Transport::Sse => None,
        };
        // Cost guard: Ollama Cloud re-encodes the full context on every
        // tool-call iteration (no prompt caching). Guard at 500k cumulative
        // input tokens per turn to prevent runaway spend (R124-B6).
        // SSE endpoints (OpenAI, most gateways) have prompt-caching
        // and get no guard by default. Local Ollama risk is low (own HW).
        let cost_guard_tokens = match transport {
            Transport::OllamaNdjson if resolved.endpoint.starts_with("https://ollama.com") => {
                Some(500_000)
            }
            _ => None,
        };
        // Keep model resident across idle turns (matches factory defaults).
        // Ollama default = 5m; 60m covers typical session-resume gaps.
        let keep_alive = match transport {
            Transport::OllamaNdjson => Some("60m".into()),
            Transport::Sse => None,
        };
        // Pre-flight prompt-size guard window mirrors num_ctx for
        // Ollama-shape backends (the model truncates at num_ctx anyway,
        // so guard early and emit a PromptTrimmed event). For hosted
        // OpenAI-shape backends `None` keeps the legacy "send and let
        // the provider's 400 surface as TurnFailed" behaviour — the host
        // can override per-session via the ConfigResolved.context_tokens
        // path once that's wired through (R418-F2 follow-up).
        let context_window = num_ctx;
        OpenAiCompatConfig {
            provider_label: resolved.provider_label,
            endpoint: resolved.endpoint,
            default_model: resolved.default_model,
            api_key,
            max_output_tokens: None,
            max_tool_iterations: None,
            transport,
            cost_guard_tokens,
            num_ctx,
            keep_alive,
            reasoning: resolved.reasoning,
            context_window,
            prompt_overflow_threshold: None,
            quota_ledger: None,
            rate_limit_hint: None,
            session_preamble: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yah_runner_core::resolver::{ConnectionKind, WireProtocol};
    use yah_runner_core::ReasoningBudget;

    fn resolved(endpoint: &str, auth: AuthMaterial) -> ResolvedEndpoint {
        ResolvedEndpoint {
            endpoint: endpoint.into(),
            auth,
            default_model: "m".into(),
            kind: ConnectionKind::Cloud,
            provider_label: "ollama".into(),
            wire_protocol: WireProtocol::OpenaiWire,
            reasoning: Some(ReasoningBudget::Fast),
        }
    }

    #[test]
    fn ollama_native_endpoints_get_ndjson_and_window_defaults() {
        let cloud = OpenAiCompatConfig::from(resolved(
            "https://ollama.com/api/chat",
            AuthMaterial::Bearer("k".into()),
        ));
        assert!(matches!(cloud.transport, Transport::OllamaNdjson));
        assert_eq!(cloud.num_ctx, Some(32768));
        assert_eq!(cloud.context_window, Some(32768));
        assert_eq!(cloud.cost_guard_tokens, Some(500_000));
        assert_eq!(cloud.keep_alive.as_deref(), Some("60m"));
        assert_eq!(cloud.api_key.as_deref(), Some("k"));
        assert_eq!(cloud.reasoning, Some(ReasoningBudget::Fast));

        let local = OpenAiCompatConfig::from(resolved(
            "http://localhost:11434/api/chat",
            AuthMaterial::None,
        ));
        assert_eq!(local.num_ctx, Some(8192));
        assert_eq!(local.cost_guard_tokens, None);
        assert_eq!(local.api_key, None);
    }

    #[test]
    fn chat_completions_endpoints_get_sse_and_no_window() {
        let cfg = OpenAiCompatConfig::from(resolved(
            "https://api.openai.com/v1/chat/completions",
            AuthMaterial::Bearer("k".into()),
        ));
        assert!(matches!(cfg.transport, Transport::Sse));
        assert_eq!(cfg.num_ctx, None);
        assert_eq!(cfg.context_window, None);
        assert_eq!(cfg.keep_alive, None);
        assert!(cfg.quota_ledger.is_none() && cfg.rate_limit_hint.is_none());
    }
}
