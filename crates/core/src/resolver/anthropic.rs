//! Anthropic resolver: `anthropic` (API key), `anthropic-gc` (OAuth bearer),
//! and `deepseek` (Anthropic-wire compat) presets.
//!
//! ## connection.base_url override
//!
//! All presets honor the subclass's `connection_id` to derive the endpoint
//! from the connection registry (same pattern as OpenAiResolver/GroqResolver).
//! When no `connection_id` is set, the canonical endpoint for each provider
//! is used as the default. This allows pointing the Anthropic-wire runner
//! at any `/v1/messages`-compatible backend (DeepSeek, third-party mirrors,
//! anthropic-proxy, etc.).
//!
//! @yah:relay(R284, "DeepSeek endpoint CI smoke test")
//! @yah:at(2026-05-22T04:00:33Z)
//! @yah:status(review)
//! @yah:gotcha("GHA secrets are repo-level — only the operator can configure; document the secret name in the workflow file")
//! @yah:gotcha("Test must skip gracefully (not fail) when DEEPSEEK_API_KEY is absent, so external contributors' PRs don't red")
//! @arch:see(crates/yah/kg/preroll/connections/deepseek.json)
//! @yah:next("Configure DEEPSEEK_API_KEY secret in GitHub Actions (Settings → Secrets and variables → Actions) — this requires repo-owner access")
//! @yah:handoff("DeepSeek smoke test implemented with #[cfg(not(target_env=\"msvc\"))] guard, skips gracefully when DEEPSEEK_API_KEY absent")
//! @yah:handoff("DeepSeek and Groq now available in Provider Settings UI (packages/yah/ui/src/components/agent/AgentProvidersPanel.tsx)")
//! @yah:handoff("Both providers registered in api-keys-context.tsx for proper keychain integration")
//! @yah:handoff("Rust-side infrastructure verified: deepseek-api-key and groq-api-key slots properly handled by respective resolvers")
//! @yah:handoff("Protocol negotiator complete: DeepSeek resolves wire protocol at session-start time based on subclass runner kind (AnthropicWire → claude_session, OpenaiWire → runner_session). 6 dispatch sites in agent.rs updated.")

use super::{
    secs_to_iso8601, with_query, AuthMaterial, ConnectionKind, CredentialSource,
    EngineResolutionError, ProbeError, ProviderResolver, ProviderSpec, ResolvedEndpoint,
    UsageError, UsageQuery, UsageResult, UsageSnapshot, WireProtocol,
    ANTHROPIC_ADMIN_ENV, ANTHROPIC_ADMIN_SLOT, ANTHROPIC_ENV, ANTHROPIC_OAUTH_ENV,
    ANTHROPIC_OAUTH_SLOT, ANTHROPIC_SLOT,
};
use crate::http::{HttpClient, HttpRequest};
use async_trait::async_trait;

const ANTHROPIC_MESSAGES_ENDPOINT: &str = "https://api.anthropic.com/v1/messages";

/// Resolves Anthropic HTTP auth for the `anthropic` (API key) and
/// `anthropic-gc` (OAuth bearer) providers.
///
/// The shape overlay for `anthropic-gc` (PinnedShape from claude_shape) is
/// a desktop concern applied by the agent.rs call site after resolution.
pub struct AnthropicResolver;

#[async_trait]
impl ProviderResolver for AnthropicResolver {
    fn resolve(
        &self,
        spec: &ProviderSpec,
        credentials: &dyn CredentialSource,
    ) -> Result<ResolvedEndpoint, EngineResolutionError> {
        match spec.provider.as_str() {
            "anthropic" => {
                let key = match spec.inline_key() {
                    Some(k) => Ok(Some(k.clone())),
                    None => credentials.resolve(ANTHROPIC_SLOT, Some(ANTHROPIC_ENV))
                        .map_err(|e| EngineResolutionError::VaultRead { reason: e.to_string() }),
                }?;
                match key {
                    Some(k) if !k.is_empty() => Ok(ResolvedEndpoint {
                        // ProviderRegistry's WireResolver overrides this endpoint from the
                        // ProtocolEntry when a connection_id is set.
                        endpoint: ANTHROPIC_MESSAGES_ENDPOINT.into(),
                        auth: AuthMaterial::Bearer(k),
                        default_model: "claude-sonnet-4-6".into(),
                        kind: ConnectionKind::Cloud,
                        provider_label: "anthropic".into(),
                        wire_protocol: WireProtocol::AnthropicWire,
                        reasoning: spec.reasoning,
                    }),
                    _ => Err(EngineResolutionError::MissingCredential {
                        provider: "anthropic".into(),
                        slot: ANTHROPIC_SLOT.into(),
                        hint: "set one in Settings → Agents (anthropic preset), or pick the \
                               'claude' preset to use the subprocess runner with your existing \
                               Pro/Max subscription"
                            .into(),
                    }),
                }
            }
            "anthropic-gc" => {
                let token = match spec.inline_key() {
                    Some(t) => Ok(Some(t.clone())),
                    None => credentials.resolve(ANTHROPIC_OAUTH_SLOT, Some(ANTHROPIC_OAUTH_ENV))
                        .map_err(|e| EngineResolutionError::VaultRead { reason: e.to_string() }),
                }?;
                match token {
                    Some(t) if !t.is_empty() => Ok(ResolvedEndpoint {
                        endpoint: ANTHROPIC_MESSAGES_ENDPOINT.into(),
                        auth: AuthMaterial::Bearer(t),
                        default_model: "claude-sonnet-4-6".into(),
                        kind: ConnectionKind::Cloud,
                        provider_label: "anthropic-gc".into(),
                        wire_protocol: WireProtocol::AnthropicWire,
                        reasoning: spec.reasoning,
                    }),
                    _ => Err(EngineResolutionError::MissingCredential {
                        provider: "anthropic-gc".into(),
                        slot: ANTHROPIC_OAUTH_SLOT.into(),
                        hint: "paste a `claude setup-token` bearer in Settings → Agents \
                               (anthropic-gc preset)"
                            .into(),
                    }),
                }
            }
            other => Err(EngineResolutionError::InvalidTraits {
                provider: other.into(),
                reason: "AnthropicResolver only handles 'anthropic' and 'anthropic-gc'".into(),
            }),
        }
    }

    async fn fetch_usage(
        &self,
        query: &UsageQuery,
        http: &dyn HttpClient,
        credentials: &dyn CredentialSource,
    ) -> UsageResult {
        let key = credentials.resolve(ANTHROPIC_ADMIN_SLOT, Some(ANTHROPIC_ADMIN_ENV))
            .map_err(|e| UsageError::Transport { reason: e.to_string() })?;
        let key = match key {
            Some(k) if !k.is_empty() => k,
            _ => {
                return Err(UsageError::MissingAdminKey {
                    hint: "create an Admin API key at console.anthropic.com → Settings → \
                           Admin Keys and store it in the 'anthropic-admin-key' slot \
                           (or export ANTHROPIC_ADMIN_KEY). Requires organization membership."
                        .into(),
                })
            }
        };

        let url = with_query(
            "https://api.anthropic.com/v1/organizations/usage_report/messages",
            &[
                ("starting_at", secs_to_iso8601(query.period_start)),
                ("ending_at", secs_to_iso8601(query.period_end)),
                ("bucket_width", "1d".into()),
            ],
        );
        let resp = http
            .send(
                HttpRequest::get(url)
                    .header("x-api-key", &key)
                    .header("anthropic-version", "2023-06-01"),
            )
            .await
            .map_err(|e| UsageError::Transport { reason: e.to_string() })?;

        let status = resp.status;
        let raw_text =
            resp.text().await.map_err(|e| UsageError::Transport { reason: e.to_string() })?;

        if !(200..300).contains(&status) {
            return Err(UsageError::Unexpected {
                status,
                body: raw_text.chars().take(256).collect(),
            });
        }

        let raw: serde_json::Value =
            serde_json::from_str(&raw_text).unwrap_or(serde_json::Value::Null);
        Ok(parse_anthropic_usage(raw, query))
    }

    async fn probe(&self, cfg: &ResolvedEndpoint, http: &dyn HttpClient) -> Result<(), ProbeError> {
        // Derive models URL from the resolved messages endpoint.
        // "https://host/v1/messages" → "https://host/v1/models"
        let models_url = cfg
            .endpoint
            .strip_suffix("/v1/messages")
            .map(|base| format!("{base}/v1/models"))
            .unwrap_or_else(|| {
                // Fallback: assume the endpoint is at the Anthropic default
                // (shouldn't normally hit this since the resolve function
                // always produces a /v1/messages-suffixed endpoint).
                "https://api.anthropic.com/v1/models".into()
            });

        let mut req = HttpRequest::get(models_url).header("anthropic-version", "2023-06-01");
        match &cfg.auth {
            AuthMaterial::Bearer(k) => {
                if cfg.provider_label == "anthropic" {
                    req = req.header("x-api-key", k);
                } else {
                    req = req.header("Authorization", format!("Bearer {k}"));
                }
            }
            AuthMaterial::None => {}
        }
        let resp = http
            .send(req)
            .await
            .map_err(|e| ProbeError::Transport { reason: e.to_string() })?;
        let status = resp.status;
        if resp.is_success() {
            return Ok(());
        }
        let body = resp
            .text()
            .await
            .unwrap_or_default()
            .chars()
            .take(256)
            .collect::<String>();
        match status {
            401 | 403 => Err(ProbeError::AuthRejected { status, body }),
            other => Err(ProbeError::Unexpected { status: other, body }),
        }
    }
}

/// Aggregate token counts across all buckets in a usage_report/messages response.
pub(crate) fn parse_anthropic_usage(raw: serde_json::Value, query: &UsageQuery) -> UsageSnapshot {
    let mut input_tokens = 0u64;
    let mut output_tokens = 0u64;
    let mut cache_read_tokens = 0u64;
    let mut cache_write_tokens = 0u64;

    if let Some(buckets) = raw.get("data").and_then(|d| d.as_array()) {
        for bucket in buckets {
            if let Some(results) = bucket.get("results").and_then(|r| r.as_array()) {
                for r in results {
                    let u64_field = |key: &str| r.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
                    input_tokens += u64_field("input_tokens");
                    output_tokens += u64_field("output_tokens");
                    cache_read_tokens += u64_field("cache_read_input_tokens");
                    cache_write_tokens += u64_field("cache_creation_input_tokens");
                }
            }
        }
    }

    UsageSnapshot {
        provider: "anthropic".into(),
        period_start: Some(secs_to_iso8601(query.period_start)),
        period_end: Some(secs_to_iso8601(query.period_end)),
        input_tokens,
        output_tokens,
        cache_read_tokens,
        cache_write_tokens,
        requests: 0,
        quota: None,
        raw,
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{MapCredentials, StubHttp};
    use super::*;

    #[test]
    fn parse_aggregates_multi_bucket_response() {
        let raw = serde_json::json!({
            "data": [
                {
                    "start_time": "2026-04-09T00:00:00Z",
                    "end_time": "2026-04-10T00:00:00Z",
                    "results": [{
                        "input_tokens": 1000,
                        "output_tokens": 200,
                        "cache_creation_input_tokens": 50,
                        "cache_read_input_tokens": 100
                    }]
                },
                {
                    "start_time": "2026-04-10T00:00:00Z",
                    "end_time": "2026-04-11T00:00:00Z",
                    "results": [{
                        "input_tokens": 500,
                        "output_tokens": 100,
                        "cache_creation_input_tokens": 20,
                        "cache_read_input_tokens": 40
                    }]
                }
            ],
            "has_more": false
        });
        let query = UsageQuery { period_start: 1744156800, period_end: 1746748800, api_key: None };
        let snap = parse_anthropic_usage(raw, &query);

        assert_eq!(snap.input_tokens, 1500);
        assert_eq!(snap.output_tokens, 300);
        assert_eq!(snap.cache_write_tokens, 70);
        assert_eq!(snap.cache_read_tokens, 140);
        assert_eq!(snap.provider, "anthropic");
        assert!(snap.period_start.is_some());
    }

    #[test]
    fn parse_empty_data_yields_zeros() {
        let raw = serde_json::json!({ "data": [], "has_more": false });
        let query = UsageQuery { period_start: 0, period_end: 86400, api_key: None };
        let snap = parse_anthropic_usage(raw, &query);
        assert_eq!(snap.input_tokens, 0);
        assert_eq!(snap.output_tokens, 0);
    }

    #[tokio::test]
    async fn missing_admin_key_returns_error() {
        let query = UsageQuery::last_days(30);
        let result = AnthropicResolver
            .fetch_usage(&query, &StubHttp::new(200, "{}"), &MapCredentials::empty())
            .await;
        assert!(
            matches!(result, Err(UsageError::MissingAdminKey { .. })),
            "expected MissingAdminKey, got: {result:?}"
        );
    }
}
