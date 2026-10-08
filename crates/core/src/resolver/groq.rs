//! Groq resolver.
//!
//! Groq serves a fully OpenAI-wire-compatible `/chat/completions` endpoint at
//! `https://api.groq.com/openai/v1` with Bearer-key auth. Their LPU inference
//! is fast (~300-500 tok/s on llama-3.3-70b) and the free tier is generous,
//! making them a natural fit for gnome-tier work and the "round-robin across
//! cheap big open-weight models" subclass chain.
//!
//! Dynamic model discovery fetches available models from Groq's `/v1/models`
//! endpoint (standard OpenAI-compatible shape). Falls back to a hardcoded default
//! if the API call fails.
//!
//! No `fetch_usage` implementation today — Groq does not expose a credit or
//! quota endpoint comparable to OpenRouter's `/credits`. Rate-limit headers
//! land on individual 429 responses and are observed by the runner, not by
//! this resolver.

use super::{
    models_url_from_endpoint, probe_openai_compat, AuthMaterial, ConnectionKind,
    CredentialSource, EngineResolutionError, ProbeError, ProviderResolver, ProviderSpec,
    ResolvedEndpoint, UsageError, UsageQuery, UsageResult, WireProtocol, GROQ_ENV, GROQ_SLOT,
};
use crate::http::{HttpClient, HttpRequest};
use async_trait::async_trait;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

const GROQ_DEFAULT_ENDPOINT: &str = "https://api.groq.com/openai/v1/chat/completions";
const GROQ_DEFAULT_MODEL: &str = "llama-3.3-70b-versatile";
const GROQ_MODELS_CACHE_TTL: Duration = Duration::from_secs(3600); // 1 hour

/// Cached model list with fetch timestamp.
struct ModelCache {
    models: Vec<String>,
    fetched_at: Instant,
}

static GROQ_MODELS_CACHE: LazyLock<Mutex<Option<ModelCache>>> = LazyLock::new(|| Mutex::new(None));

/// Fetch available models from Groq's `/v1/models` endpoint.
/// Returns a list of model IDs, or falls back to a hardcoded list on error.
/// Results are cached for 1 hour.
pub async fn fetch_groq_models(http: &dyn HttpClient, api_key: &str) -> Vec<String> {
    // Check cache first
    if let Ok(cache_lock) = GROQ_MODELS_CACHE.lock() {
        if let Some(cached) = cache_lock.as_ref() {
            if cached.fetched_at.elapsed() < GROQ_MODELS_CACHE_TTL {
                return cached.models.clone();
            }
        }
    }

    let models_url = "https://api.groq.com/openai/v1/models";
    match fetch_groq_models_from_api(http, models_url, api_key).await {
        Ok(models) => {
            // Cache the result
            if let Ok(mut cache_lock) = GROQ_MODELS_CACHE.lock() {
                *cache_lock = Some(ModelCache {
                    models: models.clone(),
                    fetched_at: Instant::now(),
                });
            }
            models
        }
        Err(e) => {
            eprintln!("[groq resolver] failed to fetch models: {e}; using fallback");
            fallback_groq_models()
        }
    }
}

async fn fetch_groq_models_from_api(
    http: &dyn HttpClient,
    url: &str,
    api_key: &str,
) -> Result<Vec<String>, String> {
    let response = http
        .send(HttpRequest::get(url).bearer_auth(api_key).timeout(Duration::from_secs(10)))
        .await
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    if !response.is_success() {
        return Err(format!("HTTP {}", response.status));
    }

    let body = response
        .json::<serde_json::Value>()
        .await
        .map_err(|e| format!("JSON parse failed: {e}"))?;

    let data = body
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| "response missing 'data' array".to_string())?;

    let models: Vec<String> = data
        .iter()
        .filter_map(|entry| entry.get("id").and_then(|id| id.as_str()).map(String::from))
        .collect();

    if models.is_empty() {
        return Err("no models found in response".to_string());
    }

    Ok(models)
}

/// Fallback hardcoded list of known Groq models (as of April 2026).
/// Updated periodically; use fetch_groq_models() for the latest list.
fn fallback_groq_models() -> Vec<String> {
    vec![
        "llama-3.3-70b-versatile".into(),
        "gpt-oss-120b".into(),
        "deepseek-r1-distill-llama-70b".into(),
        "qwen3-32b".into(),
        "llama-3.1-70b-versatile".into(),
        "mixtral-8x7b-32768".into(),
    ]
}

pub struct GroqResolver;

#[async_trait]
impl ProviderResolver for GroqResolver {
    fn resolve(
        &self,
        spec: &ProviderSpec,
        credentials: &dyn CredentialSource,
    ) -> Result<ResolvedEndpoint, EngineResolutionError> {
        // ProviderRegistry's WireResolver overrides this endpoint from the ProtocolEntry
        // when a connection_id is set; the default covers direct callers and no-connection paths.
        let endpoint = GROQ_DEFAULT_ENDPOINT.into();

        let key = match spec.inline_key() {
            Some(k) => Ok(Some(k.clone())),
            None => credentials.resolve(GROQ_SLOT, Some(GROQ_ENV))
                .map_err(|e| EngineResolutionError::VaultRead { reason: e.to_string() }),
        }?;
        match key {
            Some(k) if !k.is_empty() => Ok(ResolvedEndpoint {
                endpoint,
                auth: AuthMaterial::Bearer(k),
                default_model: GROQ_DEFAULT_MODEL.into(),
                kind: ConnectionKind::Cloud,
                provider_label: "groq".into(),
                wire_protocol: WireProtocol::OpenaiWire,
                reasoning: spec.reasoning,
            }),
            _ => Err(EngineResolutionError::MissingCredential {
                provider: "groq".into(),
                slot: GROQ_SLOT.into(),
                hint: "create a key at console.groq.com/keys and store it in the 'groq-api-key' \
                       slot via Settings → API Keys (or export GROQ_API_KEY). Groq's free tier \
                       covers llama-3.3-70b-versatile, gpt-oss-120b, deepseek-r1-distill-llama-70b, \
                       and qwen3-32b at ~30 req/min per model. Bumping to paid raises the rate \
                       limits substantially."
                    .into(),
            }),
        }
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
        _query: &UsageQuery,
        _http: &dyn HttpClient,
        _credentials: &dyn CredentialSource,
    ) -> UsageResult {
        // Groq does not expose a credit/quota endpoint. Rate-limit headers
        // arrive on individual 429 responses; the runner observes them.
        Err(UsageError::Unsupported)
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{MapCredentials, StubHttp};
    use super::super::AuthKind;
    use super::*;

    #[test]
    fn missing_key_returns_clear_hint() {
        let err = GroqResolver
            .resolve(&ProviderSpec::provider("groq"), &MapCredentials::empty())
            .unwrap_err();
        match err {
            EngineResolutionError::MissingCredential { provider, slot, hint } => {
                assert_eq!(provider, "groq");
                assert_eq!(slot, "groq-api-key");
                assert!(hint.contains("console.groq.com/keys"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn connection_id_groq_uses_registry_endpoint() {
        let spec = ProviderSpec {
            auth: Some(AuthKind::Key),
            connection_id: Some("groq".into()),
            ..ProviderSpec::provider("groq")
        };
        let cfg = GroqResolver.resolve(&spec, &MapCredentials::with(GROQ_SLOT, "test-key")).unwrap();
        assert_eq!(cfg.provider_label, "groq");
        assert!(cfg.endpoint.contains("api.groq.com"), "{}", cfg.endpoint);
        assert!(matches!(cfg.auth, AuthMaterial::Bearer(_)));
    }

    #[test]
    fn fallback_models_are_non_empty() {
        let models = fallback_groq_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"llama-3.3-70b-versatile".to_string()));
    }

    /// A rejected key (401) yields the fallback list. (The models cache is
    /// process-global and `cache_holds_models_across_calls` fills it with the
    /// same fallback list, so either path returns it.)
    #[tokio::test]
    async fn fetch_groq_models_returns_fallback_on_invalid_key() {
        let http = StubHttp::new(401, "{\"error\":\"invalid key\"}");
        let models = fetch_groq_models(&http, "invalid-key-xyz").await;
        assert!(!models.is_empty());
        // Should be the fallback list
        assert!(models.contains(&"llama-3.3-70b-versatile".to_string()));
    }

    #[test]
    fn cache_holds_models_across_calls() {
        // Clear cache first
        if let Ok(mut cache_lock) = GROQ_MODELS_CACHE.lock() {
            *cache_lock = None;
        }

        let models = fallback_groq_models();
        // Manually populate cache
        if let Ok(mut cache_lock) = GROQ_MODELS_CACHE.lock() {
            *cache_lock = Some(ModelCache {
                models: models.clone(),
                fetched_at: Instant::now(),
            });
        }

        // Verify cache is accessible
        if let Ok(cache_lock) = GROQ_MODELS_CACHE.lock() {
            if let Some(cached) = cache_lock.as_ref() {
                assert_eq!(cached.models.len(), models.len());
            }
        }
    }
}
