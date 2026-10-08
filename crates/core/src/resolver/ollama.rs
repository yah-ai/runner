//! Ollama resolver: distinguishes cloud (keyed) vs local (unauthenticated).
//!
//! **V2 path (R113-T6):** `connection_id` is the primary discriminant.
//! - `connection_id = "ollama-cloud"` → cloud endpoint from registry; `MissingCredential` if key absent.
//! - `connection_id` is any other non-empty string → local endpoint from registry (or localhost fallback).
//! - `connection_id` is empty → legacy fallback path using `AgentSubclass.auth`.
//!
//! **Legacy path (pre-migration records):** uses `AgentSubclass.auth` directly.
//! - `SubclassAuth::Key`  → cloud endpoint; `MissingCredential` if key absent.
//! - `SubclassAuth::None` → local `http://localhost:11434`; no key needed.
//! - Any other value (Subscription/Token) → key-presence fallback (old behaviour).
//!
//! @arch:see(.yah/sessions/session:8e1e1cd4.jsonl)
//!
//! archived: R166 — history: `yah board history oss/runner/crates/core/src/resolver/ollama.rs`

use super::{
    models_url_from_endpoint, probe_openai_compat, AuthKind, AuthMaterial, ConnectionKind,
    CredentialSource, EngineResolutionError, ProbeError, ProviderResolver, ProviderSpec,
    QuotaInfo, ResolvedEndpoint, UsageError, UsageQuery, UsageResult, UsageSnapshot,
    WireProtocol, OLLAMA_ENV, OLLAMA_SLOT,
};
use crate::http::{HttpClient, HttpRequest};
use crate::RunnerError;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

// Native Ollama `/api/chat` — honours `options.num_ctx` and doesn't
// silently truncate the request like `/v1/chat/completions` does at
// num_ctx=4096 (R169).
const OLLAMA_CLOUD_ENDPOINT: &str = "https://ollama.com/api/chat";
const OLLAMA_LOCAL_ENDPOINT: &str = "http://localhost:11434/api/chat";
const OLLAMA_CLOUD_DEFAULT_MODEL: &str = "gpt-oss:20b";
const OLLAMA_LOCAL_DEFAULT_MODEL: &str = "llama3.3";
const MODEL_CACHE_TTL: Duration = Duration::from_secs(30);

/// In-process cache: base_url → (fetched_at, sorted tag list).
static MODEL_CACHE: LazyLock<Mutex<HashMap<String, (Instant, Vec<String>)>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Strip the chat path suffix from an Ollama endpoint to get the native base URL.
/// Handles the native `/api/chat` (canonical post-R169), the OpenAI-compat
/// `/v1/chat/completions` (prefixed path), and the bare `/chat/completions`
/// produced by WireResolver when baseUrl has no `/v1` prefix (e.g. ollama-local.json).
/// `http://localhost:11434/api/chat` → `http://localhost:11434`
fn ollama_base_url(chat_endpoint: &str) -> String {
    for suffix in &["/api/chat", "/v1/chat/completions", "/chat/completions"] {
        if let Some(s) = chat_endpoint.strip_suffix(suffix) {
            return s.to_owned();
        }
    }
    if let Some(s) = chat_endpoint.strip_suffix('/') {
        s.to_owned()
    } else {
        chat_endpoint.to_owned()
    }
}

/// Curated top Ollama cloud models — used when the live `/api/tags` fetch fails
/// or no API key is available yet.
fn ollama_cloud_fallback() -> Vec<String> {
    vec![
        "llama3.3:70b".into(),
        "qwen3:235b".into(),
        "qwen3:32b".into(),
        "qwen3:14b".into(),
        "qwen3:7b".into(),
        "gemma3:27b".into(),
        "gemma3:12b".into(),
        "deepseek-r1:70b".into(),
        "phi4:14b".into(),
        "mistral:7b".into(),
        "llama3.2:3b".into(),
        "gpt-oss:20b".into(),
    ]
}

/// Fetch available models from Ollama Cloud (`https://ollama.com/api/tags`) using
/// a bearer API key. Falls back to [`ollama_cloud_fallback`] if the request fails
/// or no key is provided — the UI can still show a useful picker.
pub async fn list_cloud_models(http: &dyn HttpClient, api_key: Option<&str>) -> Vec<String> {
    let key = match api_key.filter(|k| !k.is_empty()) {
        Some(k) => k,
        None => return ollama_cloud_fallback(),
    };

    let resp = match http.send(HttpRequest::get("https://ollama.com/api/tags").bearer_auth(key)).await {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!("ollama cloud /api/tags request failed: {e}; using fallback list");
            return ollama_cloud_fallback();
        }
    };

    if !resp.is_success() {
        tracing::debug!(
            "ollama cloud /api/tags returned {}; using fallback list",
            resp.status
        );
        return ollama_cloud_fallback();
    }

    let json: serde_json::Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!("ollama cloud /api/tags parse failed: {e}; using fallback list");
            return ollama_cloud_fallback();
        }
    };

    let mut tags: Vec<String> = json
        .get("models")
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("name").and_then(|v| v.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if tags.is_empty() {
        return ollama_cloud_fallback();
    }

    tags.sort();
    tags.dedup();
    tags
}

/// Fetch installed Ollama model tags via `GET /api/tags`, with a 30-second
/// in-process cache keyed by base URL.
///
/// Returns sorted, deduped tag strings (e.g. `"llama3.3:latest"`).
/// An empty `Ok(vec![])` means Ollama is running but no models are pulled.
pub async fn list_local_models(
    http: &dyn HttpClient,
    chat_endpoint: &str,
) -> Result<Vec<String>, RunnerError> {
    let base = ollama_base_url(chat_endpoint);

    {
        let cache = MODEL_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((ts, tags)) = cache.get(&base) {
            if ts.elapsed() < MODEL_CACHE_TTL {
                return Ok(tags.clone());
            }
        }
    }

    let tags_url = format!("{base}/api/tags");
    let resp = http
        .send(HttpRequest::get(tags_url))
        .await
        .map_err(|e| RunnerError::Transport(format!("ollama /api/tags request failed: {e}")))?;

    if !resp.is_success() {
        let status = resp.status;
        let body = resp.text().await.unwrap_or_default();
        return Err(RunnerError::Transport(format!(
            "ollama /api/tags returned {}: {}",
            status,
            body.chars().take(256).collect::<String>(),
        )));
    }

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| RunnerError::Transport(format!("ollama /api/tags parse failed: {e}")))?;

    let mut tags: Vec<String> = json
        .get("models")
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("name").and_then(|v| v.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();
    tags.sort();
    tags.dedup();

    {
        let mut cache = MODEL_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        cache.insert(base, (Instant::now(), tags.clone()));
    }

    Ok(tags)
}

pub struct OllamaResolver;

#[async_trait]
impl ProviderResolver for OllamaResolver {
    fn resolve(
        &self,
        spec: &ProviderSpec,
        credentials: &dyn CredentialSource,
    ) -> Result<ResolvedEndpoint, EngineResolutionError> {
        // Extract subclass-level key override (character > subclass > rig cascade).
        let subclass_key: Option<String> = spec.inline_key().cloned();

        // V2 path: connection_id drives cloud vs local.
        let connection_id = spec.connection_id();
        if !connection_id.is_empty() {
            return self.resolve_from_connection(connection_id, subclass_key, spec, credentials);
        }

        // Legacy path: use auth directly (pre-migration records without connection_id).
        let auth = spec.auth.unwrap_or_default();

        match auth {
            // Typed path: subclass explicitly declares it needs a key → cloud.
            // MissingCredential if the vault slot is empty — no silent local downgrade.
            AuthKind::Key => {
                let key = match &subclass_key {
                    Some(k) => Ok(Some(k.clone())),
                    None => credentials.resolve(OLLAMA_SLOT, Some(OLLAMA_ENV))
                        .map_err(|e| EngineResolutionError::VaultRead { reason: e.to_string() }),
                }?;
                match key {
                    Some(k) if !k.is_empty() => Ok(ResolvedEndpoint {
                        endpoint: OLLAMA_CLOUD_ENDPOINT.into(),
                        auth: AuthMaterial::Bearer(k),
                        default_model: OLLAMA_CLOUD_DEFAULT_MODEL.into(),
                        kind: ConnectionKind::Cloud,
                        provider_label: "ollama".into(),
                        wire_protocol: WireProtocol::OpenaiWire,
                        reasoning: spec.reasoning,
                    }),
                    _ => Err(EngineResolutionError::MissingCredential {
                        provider: "ollama".into(),
                        slot: OLLAMA_SLOT.into(),
                        hint: "set an Ollama Cloud API key in Settings → API Keys, \
                               or switch to an ollama-local subclass for a local Ollama \
                               install (no key required)."
                            .into(),
                    }),
                }
            }

            // Typed path: local subclass → no key, localhost.
            AuthKind::None => Ok(ResolvedEndpoint {
                endpoint: OLLAMA_LOCAL_ENDPOINT.into(),
                auth: AuthMaterial::None,
                default_model: OLLAMA_LOCAL_DEFAULT_MODEL.into(),
                kind: ConnectionKind::Local,
                provider_label: "ollama".into(),
                        wire_protocol: WireProtocol::OpenaiWire,
                        reasoning: spec.reasoning,
            }),

            // Legacy path (Subscription/Token auth, or no subclass):
            // preserve old key-presence behaviour — vault key → cloud, else local.
            _ => {
                let key = match &subclass_key {
                    Some(k) => Ok(Some(k.clone())),
                    None => credentials.resolve(OLLAMA_SLOT, Some(OLLAMA_ENV))
                        .map_err(|e| EngineResolutionError::VaultRead { reason: e.to_string() }),
                }?;
                match key {
                    Some(k) if !k.is_empty() => {
                        tracing::debug!(
                            "ollama: no connection_id, unrecognized auth — \
                             found key in vault, routing to cloud (legacy path)"
                        );
                        Ok(ResolvedEndpoint {
                            endpoint: OLLAMA_CLOUD_ENDPOINT.into(),
                            auth: AuthMaterial::Bearer(k),
                            default_model: OLLAMA_CLOUD_DEFAULT_MODEL.into(),
                            kind: ConnectionKind::Cloud,
                            provider_label: "ollama".into(),
                        wire_protocol: WireProtocol::OpenaiWire,
                        reasoning: spec.reasoning,
                        })
                    }
                    _ => {
                        tracing::debug!(
                            "ollama: no connection_id, no key in vault — \
                             routing to local (legacy path)"
                        );
                        Ok(ResolvedEndpoint {
                            endpoint: OLLAMA_LOCAL_ENDPOINT.into(),
                            auth: AuthMaterial::None,
                            default_model: OLLAMA_LOCAL_DEFAULT_MODEL.into(),
                            kind: ConnectionKind::Local,
                            provider_label: "ollama".into(),
                        wire_protocol: WireProtocol::OpenaiWire,
                        reasoning: spec.reasoning,
                        })
                    }
                }
            }
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

    /// Probe `https://ollama.com/api/account/usage` with the existing Ollama Cloud key.
    ///
    /// The endpoint is community-proposed and not officially documented as of 2026-05-09
    /// (see ollama/ollama#15663). If it 404s, 401s, or 403s we return `Unsupported`
    /// rather than hard-failing — the UI renders "quota data unavailable" instead of an
    /// error strip. Local Ollama (no key) always returns `Unsupported`.
    async fn fetch_usage(
        &self,
        query: &UsageQuery,
        http: &dyn HttpClient,
        credentials: &dyn CredentialSource,
    ) -> UsageResult {
        let key = credentials.resolve(OLLAMA_SLOT, Some(OLLAMA_ENV))
            .map_err(|e| UsageError::Transport { reason: e.to_string() })?;
        let key = match key {
            Some(k) if !k.is_empty() => k,
            _ => return Err(UsageError::Unsupported), // local Ollama — no billing API
        };

        let resp = http
            .send(HttpRequest::get("https://ollama.com/api/account/usage").bearer_auth(&key))
            .await
            .map_err(|e| UsageError::Transport { reason: e.to_string() })?;

        let status = resp.status;
        // Treat 404/401/403 as "endpoint not available yet" — graceful N/A.
        if matches!(status, 401 | 403 | 404) {
            return Err(UsageError::Unsupported);
        }
        if !(200..300).contains(&status) {
            let body = resp.text().await.unwrap_or_default().chars().take(256).collect();
            return Err(UsageError::Unexpected { status, body });
        }

        let raw_text =
            resp.text().await.map_err(|e| UsageError::Transport { reason: e.to_string() })?;
        let raw: serde_json::Value =
            serde_json::from_str(&raw_text).unwrap_or(serde_json::Value::Null);
        Ok(parse_ollama_usage(raw, query))
    }
}

impl OllamaResolver {
    /// V2 resolution path: connection_id drives cloud vs local auth.
    /// ProviderRegistry's WireResolver overrides the endpoint from the ProtocolEntry;
    /// these constants are sensible defaults for direct-resolver callers.
    fn resolve_from_connection(
        &self,
        connection_id: &str,
        subclass_key: Option<String>,
        spec: &ProviderSpec,
        credentials: &dyn CredentialSource,
    ) -> Result<ResolvedEndpoint, EngineResolutionError> {
        let endpoint = if connection_id == "ollama-cloud" {
            OLLAMA_CLOUD_ENDPOINT.into()
        } else {
            OLLAMA_LOCAL_ENDPOINT.into()
        };

        if connection_id == "ollama-cloud" {
            let key = match &subclass_key {
                Some(k) => Ok(Some(k.clone())),
                None => credentials.resolve(OLLAMA_SLOT, Some(OLLAMA_ENV))
                    .map_err(|e| EngineResolutionError::VaultRead { reason: e.to_string() }),
            }?;
            match key {
                Some(k) if !k.is_empty() => Ok(ResolvedEndpoint {
                    endpoint,
                    auth: AuthMaterial::Bearer(k),
                    default_model: OLLAMA_CLOUD_DEFAULT_MODEL.into(),
                    kind: ConnectionKind::Cloud,
                    provider_label: "ollama".into(),
                        wire_protocol: WireProtocol::OpenaiWire,
                        reasoning: spec.reasoning,
                }),
                _ => Err(EngineResolutionError::MissingCredential {
                    provider: "ollama".into(),
                    slot: OLLAMA_SLOT.into(),
                    hint: "set an Ollama Cloud API key in Settings → API Keys, \
                           or switch to an ollama-local subclass for a local Ollama \
                           install (no key required)."
                        .into(),
                }),
            }
        } else {
            // Local or user-authored connection: no auth required.
            let default_model = if connection_id.contains("cloud") {
                OLLAMA_CLOUD_DEFAULT_MODEL
            } else {
                OLLAMA_LOCAL_DEFAULT_MODEL
            };
            Ok(ResolvedEndpoint {
                endpoint,
                auth: AuthMaterial::None,
                default_model: default_model.into(),
                kind: ConnectionKind::Local,
                provider_label: "ollama".into(),
                        wire_protocol: WireProtocol::OpenaiWire,
                        reasoning: spec.reasoning,
            })
        }
    }
}

/// Parse the community-proposed Ollama Cloud account/usage response.
pub(crate) fn parse_ollama_usage(raw: serde_json::Value, _query: &UsageQuery) -> UsageSnapshot {
    let quota = QuotaInfo {
        session_used_pct: raw.pointer("/session/used_percentage").and_then(|v| v.as_f64()),
        session_resets_at: raw
            .pointer("/session/resets_at")
            .and_then(|v| v.as_str())
            .map(str::to_owned),
        weekly_used_pct: raw.pointer("/weekly/used_percentage").and_then(|v| v.as_f64()),
        weekly_resets_at: raw
            .pointer("/weekly/resets_at")
            .and_then(|v| v.as_str())
            .map(str::to_owned),
        plan: raw.get("plan").and_then(|v| v.as_str()).map(str::to_owned),
    };

    UsageSnapshot {
        provider: "ollama".into(),
        period_start: None,
        period_end: None,
        input_tokens: 0,
        output_tokens: 0,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        requests: 0,
        quota: Some(quota),
        raw,
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{MapCredentials, StubHttp};
    use super::*;

    fn cloud_spec() -> ProviderSpec {
        ProviderSpec {
            auth: Some(AuthKind::Key),
            connection_id: Some("ollama-cloud".into()),
            ..ProviderSpec::provider("ollama")
        }
    }

    fn local_spec() -> ProviderSpec {
        ProviderSpec {
            auth: Some(AuthKind::None),
            connection_id: Some("ollama-local".into()),
            ..ProviderSpec::provider("ollama")
        }
    }

    /// A pre-migration record: no connection id, subscription auth.
    fn legacy_spec() -> ProviderSpec {
        ProviderSpec { auth: Some(AuthKind::Subscription), ..ProviderSpec::provider("ollama") }
    }

    /// Core regression test: a cloud spec (AuthKind::Key) with no key in the
    /// credential source must return MissingCredential, never ollama_local().
    #[test]
    fn cloud_subclass_empty_vault_returns_missing_credential() {
        let result = OllamaResolver.resolve(&cloud_spec(), &MapCredentials::empty());
        assert!(
            matches!(result, Err(EngineResolutionError::MissingCredential { .. })),
            "expected MissingCredential, got: {result:?}"
        );
    }

    /// A local spec (AuthKind::None) should resolve to localhost
    /// regardless of vault content — no key read at all.
    #[test]
    fn local_subclass_resolves_to_localhost() {
        let creds = MapCredentials::with(OLLAMA_SLOT, "would-be-cloud-key");
        let result = OllamaResolver.resolve(&local_spec(), &creds).unwrap();
        assert_eq!(result.kind, ConnectionKind::Local);
        assert!(result.endpoint.contains("localhost"));
        assert!(matches!(result.auth, AuthMaterial::None));
    }

    /// Legacy path (no connection_id, unrecognized auth): key present → cloud.
    #[test]
    fn legacy_no_connection_id_no_traits_with_key_routes_to_cloud() {
        let creds = MapCredentials::with(OLLAMA_SLOT, "test-key-123");
        let result = OllamaResolver.resolve(&legacy_spec(), &creds).unwrap();
        assert_eq!(result.kind, ConnectionKind::Cloud);
        assert!(matches!(result.auth, AuthMaterial::Bearer(_)));
    }

    /// Legacy path (no connection_id, unrecognized auth): no key → local (old behaviour preserved).
    #[test]
    fn legacy_no_connection_id_no_traits_without_key_routes_to_local() {
        let result = OllamaResolver.resolve(&legacy_spec(), &MapCredentials::empty()).unwrap();
        assert_eq!(result.kind, ConnectionKind::Local);
    }

    /// Cloud spec WITH a key: should resolve to cloud endpoint with bearer.
    #[test]
    fn cloud_subclass_with_key_resolves_to_cloud() {
        let creds = MapCredentials::with(OLLAMA_SLOT, "my-ollama-cloud-key");
        let result = OllamaResolver.resolve(&cloud_spec(), &creds).unwrap();
        assert_eq!(result.kind, ConnectionKind::Cloud);
        assert!(result.endpoint.contains("ollama.com"));
        assert!(
            matches!(&result.auth, AuthMaterial::Bearer(k) if k == "my-ollama-cloud-key"),
            "expected Bearer auth, got: {:?}",
            result.auth
        );
    }

    // --- fetch_usage tests ---

    #[test]
    fn parse_ollama_usage_extracts_quota_fields() {
        let raw = serde_json::json!({
            "session": { "used_percentage": 42.5, "resets_at": "2026-05-09T17:00:00Z" },
            "weekly":  { "used_percentage": 80.0, "resets_at": "2026-05-15T00:00:00Z" },
            "plan": "pro"
        });
        let query = UsageQuery { period_start: 0, period_end: 0, api_key: None };
        let snap = parse_ollama_usage(raw, &query);

        let quota = snap.quota.expect("quota should be present");
        assert_eq!(quota.session_used_pct, Some(42.5));
        assert_eq!(quota.weekly_used_pct, Some(80.0));
        assert_eq!(quota.plan.as_deref(), Some("pro"));
        assert_eq!(snap.provider, "ollama");
        assert_eq!(snap.input_tokens, 0);
    }

    #[test]
    fn parse_ollama_usage_handles_missing_fields() {
        let raw = serde_json::json!({});
        let query = UsageQuery { period_start: 0, period_end: 0, api_key: None };
        let snap = parse_ollama_usage(raw, &query);

        let quota = snap.quota.expect("quota struct should still be present");
        assert!(quota.session_used_pct.is_none());
        assert!(quota.plan.is_none());
    }

    #[tokio::test]
    async fn local_ollama_returns_unsupported() {
        let query = UsageQuery::last_days(30);
        let result = OllamaResolver
            .fetch_usage(&query, &StubHttp::new(200, "{}"), &MapCredentials::empty())
            .await;
        assert!(
            matches!(result, Err(UsageError::Unsupported)),
            "expected Unsupported for local Ollama, got: {result:?}"
        );
    }

    // --- list_local_models helpers ---

    #[test]
    fn ollama_base_url_strips_chat_suffixes() {
        assert_eq!(ollama_base_url("http://localhost:11434/api/chat"), "http://localhost:11434");
        assert_eq!(ollama_base_url("http://localhost:11434/v1/chat/completions"), "http://localhost:11434");
        // WireResolver produces this when baseUrl has no /v1 prefix (ollama-local.json)
        assert_eq!(ollama_base_url("http://localhost:11434/chat/completions"), "http://localhost:11434");
        assert_eq!(ollama_base_url("http://localhost:11434/"), "http://localhost:11434");
        assert_eq!(ollama_base_url("http://localhost:11434"), "http://localhost:11434");
    }

    /// Parse a mock /api/tags response and verify the tag list is extracted correctly.
    #[test]
    fn parse_api_tags_mock() {
        let raw = serde_json::json!({
            "models": [
                { "name": "llama3.3:latest", "size": 1234 },
                { "name": "qwen3.5:latest", "size": 5678 },
                { "name": "llama3.3:latest" },  // duplicate — should be deduped
            ]
        });

        let mut tags: Vec<String> = raw
            .get("models")
            .and_then(|m| m.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("name").and_then(|v| v.as_str()).map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        tags.sort();
        tags.dedup();

        assert_eq!(tags, vec!["llama3.3:latest", "qwen3.5:latest"]);
    }
}
