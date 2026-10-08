//! OpenAI resolver.
//!
//! V2 (R113-T6): when the subclass carries a non-empty `connection_id`, the
//! endpoint is derived from the bundled connection registry (base_url +
//! `/chat/completions`). This allows camp-level overrides (e.g. a vLLM proxy
//! registered as a custom connection) without touching resolver code.
//! Falls back to the canonical `api.openai.com` endpoint when connection_id
//! is absent (legacy records and direct construction).
//!

use super::{
    models_url_from_endpoint, probe_openai_compat, secs_to_iso8601, with_query, AuthMaterial,
    ConnectionKind, CredentialSource, EngineResolutionError, ProbeError, ProviderResolver,
    ProviderSpec, ResolvedEndpoint, UsageError, UsageQuery, UsageResult, UsageSnapshot,
    WireProtocol, OPENAI_ADMIN_ENV, OPENAI_ADMIN_SLOT, OPENAI_ENV,
    OPENAI_SLOT,
};
use crate::http::{HttpClient, HttpRequest};
use async_trait::async_trait;


const OPENAI_DEFAULT_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";
const OPENAI_DEFAULT_MODEL: &str = "gpt-4o";

pub struct OpenAiResolver;

#[async_trait]
impl ProviderResolver for OpenAiResolver {
    fn resolve(
        &self,
        spec: &ProviderSpec,
        credentials: &dyn CredentialSource,
    ) -> Result<ResolvedEndpoint, EngineResolutionError> {
        // ProviderRegistry's WireResolver overrides this endpoint from the ProtocolEntry
        // when a connection_id is set; the default covers direct callers and no-connection paths.
        let endpoint = OPENAI_DEFAULT_ENDPOINT.into();

        let key = match spec.inline_key() {
            Some(k) => Some(k.clone()),
            None => credentials.resolve(OPENAI_SLOT, Some(OPENAI_ENV))
                .map_err(|e| EngineResolutionError::VaultRead { reason: e.to_string() })?
                .filter(|k| !k.is_empty()),
        };

        // When connection_id is set, a missing OPENAI_API_KEY is not an error here —
        // ProviderRegistry::resolve step 2.5 will look up the connection's own credential
        // slot (which may differ from OPENAI_SLOT, e.g. "ollama" for ollama-cloud) and
        // either fill in the auth or report MissingCredential with the correct slot name.
        let has_connection = !spec.connection_id().is_empty();
        if key.is_none() && !has_connection {
            return Err(EngineResolutionError::MissingCredential {
                provider: "openai".into(),
                slot: OPENAI_SLOT.into(),
                hint: "paste one from platform.openai.com/api-keys via Settings → API Keys, \
                       or export OPENAI_API_KEY. ChatGPT Plus does not include API access; \
                       you'll need a separately-billed key."
                    .into(),
            });
        }

        Ok(ResolvedEndpoint {
            endpoint,
            auth: match key {
                Some(k) => AuthMaterial::Bearer(k),
                None => AuthMaterial::None,
            },
            default_model: OPENAI_DEFAULT_MODEL.into(),
            kind: ConnectionKind::Cloud,
            provider_label: "openai".into(),
            wire_protocol: WireProtocol::OpenaiWire,
            reasoning: spec.reasoning,
        })
    }

    async fn probe(&self, cfg: &ResolvedEndpoint, http: &dyn HttpClient) -> Result<(), ProbeError> {
        let models_url = models_url_from_endpoint(&cfg.endpoint);
        let bearer = match &cfg.auth {
            AuthMaterial::Bearer(k) => Some(k.as_str()),
            AuthMaterial::None => None,
        };
        probe_openai_compat(http, &models_url, bearer).await
    }

    async fn fetch_usage(
        &self,
        query: &UsageQuery,
        http: &dyn HttpClient,
        credentials: &dyn CredentialSource,
    ) -> UsageResult {
        let key = credentials.resolve(OPENAI_ADMIN_SLOT, Some(OPENAI_ADMIN_ENV))
            .map_err(|e| UsageError::Transport { reason: e.to_string() })?;
        let key = match key {
            Some(k) if !k.is_empty() => k,
            _ => {
                return Err(UsageError::MissingAdminKey {
                    hint: "create an Admin API key at platform.openai.com → Settings → API Keys \
                           and store it in the 'openai-admin-key' slot \
                           (or export OPENAI_ADMIN_KEY)."
                        .into(),
                })
            }
        };

        let url = with_query(
            "https://api.openai.com/v1/organization/usage/completions",
            &[
                ("start_time", query.period_start.to_string()),
                ("end_time", query.period_end.to_string()),
                ("bucket_width", "1d".into()),
            ],
        );
        let resp = http
            .send(HttpRequest::get(url).bearer_auth(&key))
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
        Ok(parse_openai_usage(raw, query))
    }
}

/// Aggregate token counts across all buckets in an OpenAI usage/completions response.
pub(crate) fn parse_openai_usage(raw: serde_json::Value, query: &UsageQuery) -> UsageSnapshot {
    let mut input_tokens = 0u64;
    let mut output_tokens = 0u64;
    let mut cache_read_tokens = 0u64;
    let mut requests = 0u64;

    if let Some(buckets) = raw.get("data").and_then(|d| d.as_array()) {
        for bucket in buckets {
            if let Some(results) = bucket.get("results").and_then(|r| r.as_array()) {
                for r in results {
                    let u64_field = |key: &str| r.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
                    input_tokens += u64_field("input_tokens");
                    output_tokens += u64_field("output_tokens");
                    cache_read_tokens += u64_field("input_cached_tokens");
                    requests += u64_field("num_model_requests");
                }
            }
        }
    }

    UsageSnapshot {
        provider: "openai".into(),
        period_start: Some(secs_to_iso8601(query.period_start)),
        period_end: Some(secs_to_iso8601(query.period_end)),
        input_tokens,
        output_tokens,
        cache_read_tokens,
        cache_write_tokens: 0,
        requests,
        quota: None,
        raw,
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{MapCredentials, StubHttp};
    use super::super::AuthKind;
    use super::*;

    #[test]
    fn parse_aggregates_multi_bucket_response() {
        let raw = serde_json::json!({
            "data": [
                {
                    "object": "bucket",
                    "start_time": 1744156800_u64,
                    "end_time": 1744243200_u64,
                    "results": [{
                        "input_tokens": 800,
                        "output_tokens": 150,
                        "input_cached_tokens": 30,
                        "num_model_requests": 5
                    }]
                },
                {
                    "object": "bucket",
                    "start_time": 1744243200_u64,
                    "end_time": 1744329600_u64,
                    "results": [{
                        "input_tokens": 200,
                        "output_tokens": 50,
                        "input_cached_tokens": 10,
                        "num_model_requests": 2
                    }]
                }
            ],
            "next_page": null
        });
        let query = UsageQuery { period_start: 1744156800, period_end: 1744329600, api_key: None };
        let snap = parse_openai_usage(raw, &query);

        assert_eq!(snap.input_tokens, 1000);
        assert_eq!(snap.output_tokens, 200);
        assert_eq!(snap.cache_read_tokens, 40);
        assert_eq!(snap.requests, 7);
        assert_eq!(snap.provider, "openai");
    }

    #[test]
    fn parse_empty_data_yields_zeros() {
        let raw = serde_json::json!({ "data": [], "next_page": null });
        let query = UsageQuery { period_start: 0, period_end: 86400, api_key: None };
        let snap = parse_openai_usage(raw, &query);
        assert_eq!(snap.input_tokens, 0);
        assert_eq!(snap.requests, 0);
    }

    /// Resolve with a connection_id pointing at the "openai" connection.
    /// The endpoint is the provider default (api.openai.com) until the
    /// registry's wire resolver overrides it.
    #[test]
    fn connection_id_openai_uses_registry_endpoint() {
        let spec = ProviderSpec {
            auth: Some(AuthKind::Key),
            connection_id: Some("openai".into()),
            ..ProviderSpec::provider("openai")
        };
        let result = OpenAiResolver
            .resolve(&spec, &MapCredentials::with(OPENAI_SLOT, "test-key"))
            .unwrap();
        assert!(result.endpoint.contains("api.openai.com"), "{}", result.endpoint);
        assert!(matches!(result.auth, AuthMaterial::Bearer(_)));
    }

    /// R365-B1 regression: OpenAiResolver with a non-openai connection_id (e.g.
    /// ollama-cloud) and no OPENAI_API_KEY must return Ok(auth: None) rather than
    /// Err(MissingCredential). ProviderRegistry::resolve step 2 then reads the
    /// connection's own slot ("ollama") and either fills in the key or emits the
    /// correct MissingCredential.
    #[test]
    fn non_openai_connection_defers_credential_to_registry() {
        let spec = ProviderSpec {
            auth: Some(AuthKind::Key),
            connection_id: Some("ollama-cloud".into()),
            ..ProviderSpec::provider("openai")
        };
        let result = OpenAiResolver.resolve(&spec, &MapCredentials::empty());
        assert!(result.is_ok(), "expected Ok(auth: None), got Err: {result:?}");
        assert!(
            matches!(result.unwrap().auth, AuthMaterial::None),
            "expected AuthMaterial::None — credential deferred to ProviderRegistry step 2"
        );
    }

    #[tokio::test]
    async fn missing_admin_key_returns_error() {
        let query = UsageQuery::last_days(30);
        let result = OpenAiResolver
            .fetch_usage(&query, &StubHttp::new(200, "{}"), &MapCredentials::empty())
            .await;
        assert!(
            matches!(result, Err(UsageError::MissingAdminKey { .. })),
            "expected MissingAdminKey, got: {result:?}"
        );
    }
}
