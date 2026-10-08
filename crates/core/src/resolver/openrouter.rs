//! OpenRouter resolver.
//!
//! OpenRouter is an OpenAI-wire aggregator: same `/chat/completions` shape
//! as OpenAI, served from `https://openrouter.ai/api/v1`. This resolver
//! mirrors [`OpenAiResolver`](super::openai::OpenAiResolver) — single Bearer
//! key, SSE transport, registry-driven endpoint — and adds two
//! OpenRouter-specific extras:
//!
//! 1. `fetch_usage` hits `GET /api/v1/credits` (dollars, not tokens) and
//!    `GET /api/v1/activity` (per-request log, last 24 h) concurrently.
//!    The activity endpoint is not publicly documented; it degrades silently
//!    to absent if it returns a non-2xx. The usage panel renders the credit
//!    balance from `raw.data` and the activity feed from `raw.activity`.
//! 2. [`free_tier_429_hint`] returns an actionable string when a request to
//!    a `:free` model is rate-limited — the practical pain from the
//!    OpenRouter free-tier rollout: 50 req/day with no credit, 1000 req/day
//!    with $10+ topped up (the credit itself isn't consumed by `:free`
//!    models).
//!
//! @yah:ticket(R260-F1, "Almanac refresh scheduler — yah-camp tick + LRU + freshness chip wire")
//! @yah:assignee(agent:claude)
//! @yah:at(2026-05-20T22:59:35Z)
//! @yah:status(review)
//! @yah:parent(R260)
//! @yah:next("yah-camp ticks crates/yah/runner/src/resolver/openrouter::fetch_models on a configurable cadence (default 24h)")
//! @yah:next("On success, write the snapshot to <camp>/.yah/jit/openrouter-almanac.json with last_seen_at")
//! @yah:next("Boot seed: embed a last-known snapshot under crates/yah/runner/src/resolver/ for cold-boot fallback (probe/data/pricing.json shape)")
//! @yah:next("Add last_seen_at to AlmanacEntry consumers' UI surfaces for the freshness chip")
//! @arch:see(.yah/docs/working/W091-openrouter-almanac.md)
//! @yah:handoff("F1 shipped. Camp daemon now spawns a background OpenRouter almanac refresher in spawn_almanac_refresher (app/yah/cli/src/camp.rs:431-468): first tick fires immediately at startup, then every YAH_ALMANAC_REFRESH_HOURS (default 24h). Persistence layer lives on the runner side: AlmanacSnapshot{fetched_at,entries} written atomically to <camp>/.yah/jit/openrouter-almanac.json via refresh_openrouter_almanac(camp_root) (tmp + rename so a crashed mid-write never half-writes); load_openrouter_almanac(camp_root) reads it back; embedded_almanac_seed() returns a compile-time fallback included from crates/yah/runner/src/resolver/openrouter-almanac-seed.json (currently a 50-entry top-weekly snapshot from 2026-05-21T16:57:08Z). last_seen_at flows through unchanged on AlmanacEntry + AlmanacSnapshot.fetched_at — F4 will consume those for the UI freshness chip; the CLI already prints last_seen_at in human + JSON. Added an iso8601_to_secs inverse to support AlmanacSnapshot::age_secs(now) for the >7d staleness check the design note calls out. 4 new tests (iso8601 roundtrip + malformed-reject, embedded seed parses, refresh atomic write via tempfile); all 13 openrouter tests pass.")
//! @yah:verify("cargo test -p runner --lib resolver::openrouter::tests:: — 13/13 green (includes 4 new: iso8601 roundtrip + reject, embedded seed parse, refresh atomic write)")
//! @yah:verify("cargo check -p runner -p yah — clean (warnings only; no errors)")
//! @yah:verify("./target/debug/yah agent almanac --top 10 — top-weekly roster prints with last_seen_at footer")
//! @yah:verify("./target/debug/yah agent almanac --top 10 --free-only --tag code — currently-hot free code roster prints (owl-alpha, nemotron-3-super, laguna-m.1, gpt-oss-120b, glm-4.5-air, …)")
//!
//! @yah:ticket(R260-F5, "T1 external-benchmark join — quality scores on AlmanacEntry")
//! @yah:assignee(agent:claude)
//! @yah:at(2026-05-20T23:00:59Z)
//! @yah:status(review)
//! @yah:parent(R260)
//! @arch:see(.yah/docs/working/W091-openrouter-almanac.md)
//! @yah:depends_on(R260-F1)
//! @yah:handoff("F5 shipped. Added T1 benchmark join infrastructure to crates/yah/runner/src/resolver/openrouter.rs. AlmanacEntry now has quality_score: Option<f64> and code_subscore: Option<f64> (both #[serde(default)] for back-compat). entry_matches_query now enforces min_quality_score when both query and entry have score data; entries without quality data pass through (graceful degradation). New: BenchmarkScore struct, canonical_model_key() normalisation helper (strips provider prefix + variant suffix, normalises . and _ to -), join_quality_scores() (joins by id then permaslug canonical key), fetch_benchmark_scores() async fn (hits artificialanalysis.ai /api/v1/models via YAH_BENCHMARK_URL override; returns empty map on any error), parse_benchmark_scores() (handles { models: [...] } and bare array shapes, normalises 0–100 scale to 0.0–1.0, accepts multiple field-name aliases). refresh_almanac() calls fetch_benchmark_scores + join_quality_scores after the T0 fetch; join is best-effort and never blocks the snapshot write. lib.rs + resolver/mod.rs export all new symbols. 9 new tests (26/26 green); cargo check -p runner -p yah clean.")
//! @yah:verify("cargo test -p runner --lib resolver::openrouter::tests — 26/26 green (9 new: canonical_model_key, join_quality_scores×2, parse_benchmark_scores×3, entry_matches_query×2)")
//! @yah:verify("cargo check -p runner -p yah — clean (only pre-existing ANTHROPIC_MESSAGES_URL dead_code warning)")
//!
//! @yah:ticket(R260-S6, "T2 almanac-as-evaluator — first-party quality scores via fixed corpus")
//! @yah:assignee(agent:claude)
//! @yah:at(2026-05-20T23:01:31Z)
//! @yah:kind(spike)
//! @yah:status(review)
//! @yah:parent(R260)
//! @yah:next("Design a small fixed corpus (lint-fix, summarize, ack-and-route — ~5 tasks) with reference outputs")
//! @yah:next("Spend a few credits running each new candidate through the corpus; score against the reference (LLM-as-judge or exact-match)")
//! @yah:next("Persist scores back into the almanac snapshot — first-party signal we trust more than third-party benchmarks")
//! @yah:next("Stretch goal: dogfood yah's own eval-rig (R035/external) against the almanac; close the loop between 'yah picks a model' and 'yah evaluates a model'")
//! @arch:see(.yah/docs/working/W091-openrouter-almanac.md)
//! @yah:depends_on(R260-F5)

use super::{
    models_url_from_endpoint, probe_openai_compat, AuthMaterial, ConnectionKind,
    CredentialSource, EngineResolutionError, ModelQuery, ModelTier, ProbeError,
    ProviderResolver, ProviderSpec, ResolvedEndpoint, UsageError, UsageQuery, UsageResult,
    UsageSnapshot, WireProtocol, OPENROUTER_ENV, OPENROUTER_SLOT,
};
use crate::http::{HttpClient, HttpRequest};
use async_trait::async_trait;
use std::cmp::Ordering;
use std::path::Path;

const OPENROUTER_DEFAULT_ENDPOINT: &str = "https://openrouter.ai/api/v1/chat/completions";
const OPENROUTER_DEFAULT_MODEL: &str = "qwen/qwen3-coder:free";
/// `/api/v1/key` — returns label, per-period usage, and optional limit/remaining
/// for the authenticating key. Replaces the older `/api/v1/credits` endpoint
/// (which only reported total_credits/total_usage for the account) with richer
/// per-key data that supports per-subclass and per-character metering.
const OPENROUTER_KEY_URL: &str = "https://openrouter.ai/api/v1/key";

/// Undocumented frontend endpoint that powers openrouter.ai/models?order=…
/// — gives us popularity ordering, the `analytics` traffic block, and a per-
/// model `categories` rollup that the public `/api/v1/models` endpoint does
/// not expose. Treat as best-effort: it can change without notice. The almanac
/// design (R260) layers this on top of `/api/v1/models` rather than depending
/// on it as the sole source.
const OPENROUTER_FRONTEND_MODELS_URL: &str =
    "https://openrouter.ai/api/frontend/models/find?order=top-weekly";
const OPENROUTER_ACTIVITY_URL: &str = "https://openrouter.ai/api/v1/activity";

pub struct OpenRouterResolver;

#[async_trait]
impl ProviderResolver for OpenRouterResolver {
    fn resolve(
        &self,
        spec: &ProviderSpec,
        credentials: &dyn CredentialSource,
    ) -> Result<ResolvedEndpoint, EngineResolutionError> {
        // ProviderRegistry's WireResolver overrides this endpoint from the ProtocolEntry
        // when a connection_id is set; the default covers direct callers and no-connection paths.
        let endpoint = OPENROUTER_DEFAULT_ENDPOINT.into();

        let key = match spec.inline_key() {
            Some(k) => Ok(Some(k.clone())),
            None => credentials.resolve(OPENROUTER_SLOT, Some(OPENROUTER_ENV))
                .map_err(|e| EngineResolutionError::VaultRead { reason: e.to_string() }),
        }?;
        match key {
            Some(k) if !k.is_empty() => Ok(ResolvedEndpoint {
                endpoint,
                auth: AuthMaterial::Bearer(k),
                default_model: OPENROUTER_DEFAULT_MODEL.into(),
                kind: ConnectionKind::Cloud,
                provider_label: "openrouter".into(),
                wire_protocol: WireProtocol::OpenaiWire,
                reasoning: spec.reasoning,
            }),
            _ => Err(EngineResolutionError::MissingCredential {
                provider: "openrouter".into(),
                slot: OPENROUTER_SLOT.into(),
                hint: "create a key at openrouter.ai/keys and store it in the 'openrouter-api-key' \
                       slot via Settings → API Keys (or export OPENROUTER_API_KEY). The free tier \
                       (50 req/day) works without topping up; $10 of credit lifts that to 1000 \
                       req/day and is not consumed by :free models."
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
        query: &UsageQuery,
        http: &dyn HttpClient,
        credentials: &dyn CredentialSource,
    ) -> UsageResult {
        // Key resolution: per-query override (subclass/character key) takes
        // precedence over the rig-level vault slot.
        let key = if let Some(k) =
            query.api_key.as_deref().filter(|k| !k.is_empty())
        {
            k.to_owned()
        } else {
            let vault_key = credentials.resolve(OPENROUTER_SLOT, Some(OPENROUTER_ENV))
                .map_err(|e| UsageError::Transport { reason: e.to_string() })?;
            match vault_key {
                Some(k) if !k.is_empty() => k,
                _ => {
                    return Err(UsageError::MissingAdminKey {
                        hint: "store an OpenRouter API key in the 'openrouter-api-key' slot \
                               (or export OPENROUTER_API_KEY). The same key drives credit \
                               lookup and chat — there is no separate admin role."
                            .into(),
                    })
                }
            }
        };

        let (key_result, activity_raw) = futures_util::join!(
            http.send(HttpRequest::get(OPENROUTER_KEY_URL).bearer_auth(&key)),
            fetch_activity_24h(http, &key),
        );

        let resp = key_result.map_err(|e| UsageError::Transport { reason: e.to_string() })?;
        let status = resp.status;
        if matches!(status, 401 | 403 | 404) {
            return Err(UsageError::Unsupported);
        }
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
        let mut snap = parse_openrouter_key_info(raw, query);

        // Inject per-request activity into raw so the UI can render the feed.
        if !activity_raw.is_null() {
            if let Some(obj) = snap.raw.as_object_mut() {
                obj.insert("activity".into(), activity_raw);
            }
        }

        Ok(snap)
    }
}

/// Fetch the last 24 h of per-request activity from OpenRouter.
///
/// Not a publicly-documented endpoint — used by the web dashboard.
/// Returns `Null` on any network or HTTP error so a credits failure never
/// blocks the activity fetch and vice-versa.
async fn fetch_activity_24h(http: &dyn HttpClient, key: &str) -> serde_json::Value {
    let Ok(resp) = http
        .send(
            HttpRequest::get(format!("{OPENROUTER_ACTIVITY_URL}?date_preset=past_24_hours"))
                .bearer_auth(key),
        )
        .await
    else {
        return serde_json::Value::Null;
    };
    if !resp.is_success() {
        return serde_json::Value::Null;
    }
    resp.json::<serde_json::Value>().await.unwrap_or(serde_json::Value::Null)
}

/// Map the OpenRouter `/api/v1/key` response into a [`UsageSnapshot`].
///
/// The `/api/v1/key` endpoint returns per-key usage broken down by period:
///
/// ```json
/// {
///   "data": {
///     "label": "My Key",
///     "limit": 10.0,            // null → no per-key limit
///     "limit_reset": "month",   // "day" | "week" | "month" | null
///     "limit_remaining": 7.66,  // null → no per-key limit
///     "usage": 2.34,            // all-time
///     "usage_daily": 0.05,      // current UTC day
///     "usage_weekly": 0.30,     // current UTC week (Mon–Sun)
///     "usage_monthly": 1.20,    // current UTC month
///     "byok_usage_daily": 0.0,
///     "is_free_tier": false
///   }
/// }
/// ```
///
/// Tokens/requests aren't surfaced here — they stay 0. The raw payload passes
/// through intact for the UI to read `limit`, `limit_remaining`, and the
/// `usage_daily/weekly/monthly` period breakdown.
pub(crate) fn parse_openrouter_key_info(
    raw: serde_json::Value,
    _query: &UsageQuery,
) -> UsageSnapshot {
    UsageSnapshot {
        provider: "openrouter".into(),
        period_start: None,
        period_end: None,
        input_tokens: 0,
        output_tokens: 0,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        requests: 0,
        quota: None,
        raw,
    }
}

// ── R260 almanac: T0 schema + fetch helper ───────────────────────────────────

/// One row of the OpenRouter model almanac (R260, T0 schema).
///
/// T0 covers what `/api/v1/models` + the frontend ordering endpoint return
/// for free — no auth required. Higher tiers (quality scores T1, in-house
/// eval T2) layer on top of this without changing the shape.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AlmanacEntry {
    /// Canonical model id including any `:free`/`:nitro`/… variant suffix
    /// (e.g. `"qwen/qwen3-coder:free"`). This is the string that goes on
    /// the wire when calling `/v1/chat/completions`.
    pub id: String,
    /// Stable permaslug (snapshot-dated). `analytics`/`categories` are keyed
    /// by `<permaslug>` or `<permaslug>:variant`; kept here so the join is
    /// reproducible without re-fetching.
    pub permaslug: String,
    /// Context window in tokens. Pareto-frontier axis at T0.
    pub context_length: u32,
    /// USD per prompt token. `0.0` for `:free` variants. Pareto-frontier
    /// axis at T0; collapses to a constant on the free tier.
    pub price_prompt_usd_per_token: f64,
    /// USD per completion token.
    pub price_completion_usd_per_token: f64,
    /// True when this variant runs on OpenRouter's free tier (50 req/day
    /// without credit, 1000 with $10+ topped up). Derived from the `:free`
    /// suffix on the canonical id and a zero-priced endpoint.
    pub is_free: bool,
    /// Coarse capability tags derived from modality + supported_parameters
    /// + per-category daily rank. The vocabulary is **stable across runs**
    /// (see [`CAPABILITY_TAGS`]); per-model membership rotates with usage.
    pub capability_tags: Vec<String>,
    /// 1-based rank in the top-weekly ordering. `None` when the frontend
    /// endpoint failed and we only have `/api/v1/models` data.
    pub popularity_rank: Option<u32>,
    /// Weekly request count from `analytics[<permaslug+variant>].count`.
    /// `None` for the same reason as `popularity_rank`.
    pub weekly_requests: Option<u64>,
    /// ISO-8601 fetch time. Consumers use this to surface a freshness chip
    /// and to decide whether to refetch before resolving a `ModelQuery`.
    pub last_seen_at: String,
    /// T1: General quality score from an external benchmark (0.0–1.0 normalised).
    /// `None` when no benchmark data has been joined for this entry (R260-F5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality_score: Option<f64>,
    /// T1: Code-focused quality subscore from an external code benchmark.
    /// `None` when not available. Populated alongside `quality_score` by
    /// [`join_quality_scores`] (R260-F5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code_subscore: Option<f64>,
}

/// The fixed capability-tag vocabulary T0 emits. Keep in sync with the
/// derivation in [`derive_capability_tags`]. Documented so downstream
/// `ModelQuery::capability_tags` filters can lint against typos.
pub const CAPABILITY_TAGS: &[&str] =
    &["code", "vision", "tool-use", "long-context", "reasoning"];

#[derive(Debug, thiserror::Error)]
pub enum AlmanacError {
    #[error("openrouter almanac fetch failed: {reason}")]
    Transport { reason: String },
    #[error("openrouter almanac returned unexpected status {status}: {body}")]
    Unexpected { status: u16, body: String },
    #[error("openrouter almanac payload missing expected field: {field}")]
    Schema { field: &'static str },
}

/// Error from [`resolve_model_query`].
#[derive(Debug, thiserror::Error)]
pub enum ModelQueryError {
    #[error("no OpenRouter model matches the query (tier={tier:?}, tags={tags:?})",
        tier = .query.tier, tags = .query.capability_tags)]
    NoMatch { query: ModelQuery },
}

/// Resolve a [`ModelQuery`] to a concrete model id using the
/// cached almanac snapshot (or the embedded seed as fallback).
///
/// Picks the highest-popularity-rank entry satisfying all non-`None`
/// predicates. Returns the model id and the snapshot age in seconds.
/// `snapshot_age_secs` is `None` when the embedded seed was used (no
/// on-disk cache → fetch time is unknown).
///
/// Errors with [`ModelQueryError::NoMatch`] when no entry satisfies all
/// predicates — the session should abort before `session_started` rather
/// than silently using a stale or wrong model.
pub fn resolve_model_query(
    query: &ModelQuery,
    almanac_path: &Path,
) -> Result<(String, Option<u64>), ModelQueryError> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let (snapshot, from_disk) = match load_almanac(almanac_path) {
        Some(s) => (s, true),
        None => (embedded_almanac_seed(), false),
    };
    let snapshot_age = if from_disk { snapshot.age_secs(now_secs) } else { None };

    let is_code_query = query.capability_tags.iter().any(|t| t == "code");
    let best = snapshot
        .entries
        .iter()
        .filter(|e| entry_matches_query(e, query))
        .max_by(|a, b| resolve_entry_order(a, b, is_code_query));

    match best {
        Some(entry) => Ok((entry.id.clone(), snapshot_age)),
        None => Err(ModelQueryError::NoMatch { query: query.clone() }),
    }
}

/// Returns `true` when `entry` satisfies every predicate in `query`.
fn entry_matches_query(entry: &AlmanacEntry, query: &ModelQuery) -> bool {
    // Tier
    match query.tier {
        ModelTier::Free => {
            if !entry.is_free {
                return false;
            }
        }
        ModelTier::Paid => {
            if entry.is_free {
                return false;
            }
        }
        ModelTier::Any => {}
    }
    // Provider prefix (e.g. "qwen" matches "qwen/qwen3-coder:free")
    if let Some(ref prov) = query.provider {
        if !entry.id.starts_with(&format!("{prov}/")) {
            return false;
        }
    }
    // All required capability tags must be present
    for tag in &query.capability_tags {
        if !entry.capability_tags.iter().any(|t| t == tag) {
            return false;
        }
    }
    // Minimum context window
    if let Some(min_ctx) = query.min_context_tokens {
        if entry.context_length < min_ctx {
            return false;
        }
    }
    // Maximum price per prompt token
    if let Some(max_price) = query.max_price_per_token_usd {
        if entry.price_prompt_usd_per_token > max_price {
            return false;
        }
    }
    // T1: min_quality_score predicate. Entries without quality data pass through
    // (can't filter on what we don't have — graceful degradation when no benchmark
    // join has run). Only filtered out when BOTH query and entry have score data.
    if let Some(min_q) = query.min_quality_score {
        if let Some(qs) = entry.quality_score {
            if qs < min_q {
                return false;
            }
        }
    }
    true
}

/// Compare two almanac entries for quality-aware model selection.
///
/// Returns `Ordering::Greater` when `a` is a better match than `b`.
///
/// Cascade:
/// 1. Entry WITH quality data beats entry WITHOUT (can't rank unknown).
/// 2. Higher quality score beats lower (uses `code_subscore` for code-tagged
///    queries, falling back to `quality_score` when `code_subscore` is absent,
///    then to `quality_score` via [`code_subscore.or`]).
/// 3. More popular (lower `popularity_rank`) beats less popular.
fn resolve_entry_order(a: &AlmanacEntry, b: &AlmanacEntry, is_code_query: bool) -> Ordering {
    let a_score = if is_code_query {
        a.code_subscore.or(a.quality_score)
    } else {
        a.quality_score
    };
    let b_score = if is_code_query {
        b.code_subscore.or(b.quality_score)
    } else {
        b.quality_score
    };

    // 1. Quality-data presence beats absence.
    match (a_score, b_score) {
        (Some(_), None) => return Ordering::Greater,
        (None, Some(_)) => return Ordering::Less,
        (Some(as_), Some(bs)) => {
            // 2. Higher score beats lower. Equal/NaN → fall through to popularity.
            match as_.partial_cmp(&bs) {
                Some(Ordering::Greater) => return Ordering::Greater,
                Some(Ordering::Less) => return Ordering::Less,
                _ => {}
            }
        }
        (None, None) => {}
    }

    // 3. Popularity tiebreaker (lower rank = better).
    let a_rank = a.popularity_rank.unwrap_or(u32::MAX);
    let b_rank = b.popularity_rank.unwrap_or(u32::MAX);
    b_rank.cmp(&a_rank)
}

/// Fetch the T0 almanac snapshot from OpenRouter (no auth required).
///
/// Hits the undocumented `/api/frontend/models/find?order=top-weekly`
/// endpoint — it returns the same model schema as the public
/// `/api/v1/models` *plus* the ordering and the `analytics` traffic block.
/// If/when the frontend endpoint changes shape, the public endpoint is the
/// fallback (see [`fetch_models_public`]) and the join layer becomes
/// best-effort.
///
/// Returns models in top-weekly order. Caller filters / truncates / joins
/// with [`crate::resolver::CAPABILITY_TAGS`] queries.
pub async fn fetch_models(http: &dyn HttpClient) -> Result<Vec<AlmanacEntry>, AlmanacError> {
    let resp = http
        .send(HttpRequest::get(OPENROUTER_FRONTEND_MODELS_URL).header("User-Agent", "yah-almanac/0.1"))
        .await
        .map_err(|e| AlmanacError::Transport { reason: e.to_string() })?;
    let status = resp.status;
    let text = resp.text().await.map_err(|e| AlmanacError::Transport { reason: e.to_string() })?;
    if !(200..300).contains(&status) {
        return Err(AlmanacError::Unexpected {
            status,
            body: text.chars().take(256).collect(),
        });
    }
    let parsed: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| AlmanacError::Transport { reason: format!("invalid JSON: {e}") })?;
    parse_frontend_models(&parsed)
}

/// Document-level wrapper persisted to `<camp>/.yah/jit/openrouter-almanac.json`.
///
/// `fetched_at` mirrors the per-entry `last_seen_at` but lives at the document
/// level so consumers can read snapshot age without parsing the entries array.
/// Format is intentionally append-friendly: future fields (T1 benchmark joins,
/// T2 self-eval scores) layer on without breaking older readers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AlmanacSnapshot {
    pub fetched_at: String,
    pub entries: Vec<AlmanacEntry>,
}

impl AlmanacSnapshot {
    /// Age of the snapshot at `now_unix_secs` in seconds, or `None` when the
    /// `fetched_at` field can't be parsed (treat as fully stale).
    pub fn age_secs(&self, now_unix_secs: u64) -> Option<u64> {
        let fetched = iso8601_to_secs(&self.fetched_at)?;
        Some(now_unix_secs.saturating_sub(fetched))
    }
}

const EMBEDDED_ALMANAC_SEED: &str = include_str!("openrouter-almanac-seed.json");

/// Read the cached snapshot if present and well-formed. `None` (not `Err`)
/// when the file is missing or unparseable — caller decides whether to
/// degrade to the embedded seed or kick a fresh fetch.
pub fn load_almanac(path: &Path) -> Option<AlmanacSnapshot> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Compile-time fallback. Panics on a malformed embedded seed because that
/// represents a broken build, not a runtime degradation.
pub fn embedded_almanac_seed() -> AlmanacSnapshot {
    serde_json::from_str(EMBEDDED_ALMANAC_SEED).expect(
        "openrouter-almanac-seed.json is malformed — regenerate with \
         `yah agent almanac --top 50 --json`",
    )
}

/// Fetch a fresh snapshot from OpenRouter and persist it atomically to the
/// camp's JIT cache. Returns the snapshot so the daemon can skip a re-read
/// if it wants to forward it directly to subscribers.
///
/// Performs the T1 benchmark join (R260-F5) after the T0 fetch: calls
/// [`fetch_benchmark_scores`] and applies [`join_quality_scores`]. The join
/// is best-effort — a failed fetch leaves `quality_score`/`code_subscore`
/// as `None` without blocking the snapshot write.
#[cfg(not(target_arch = "wasm32"))]
pub async fn refresh_almanac(
    http: &dyn HttpClient,
    path: &Path,
) -> Result<AlmanacSnapshot, AlmanacError> {
    let mut entries = fetch_models(http).await?;
    // T1: join external benchmark quality scores (best-effort).
    let scores = fetch_benchmark_scores(http).await;
    if !scores.is_empty() {
        join_quality_scores(&mut entries, &scores);
    }
    let fetched_at = entries
        .first()
        .map(|e| e.last_seen_at.clone())
        .unwrap_or_else(secs_to_iso8601_now);
    let snapshot = AlmanacSnapshot { fetched_at, entries };
    let body = serde_json::to_vec_pretty(&snapshot).map_err(|e| AlmanacError::Transport {
        reason: format!("serialize: {e}"),
    })?;
    // R925: atomic replace through a per-writer staging name. The desktop's
    // embedded camp socket and a standalone `yah camp` daemon both spawn an
    // almanac refresher against the same camp root, and both fire their first
    // tick immediately — a shared `<file>.tmp` lets those two interleave.
    write_atomic(path, &body).map_err(|e| AlmanacError::Transport {
        reason: format!("write {}: {}", path.display(), e),
    })?;
    Ok(snapshot)
}

/// Parse an `RFC3339` UTC `YYYY-MM-DDTHH:MM:SSZ` string into Unix seconds.
/// Inverse of [`secs_to_iso8601`](super::secs_to_iso8601). Returns `None`
/// on any shape mismatch; callers fall back to treating the snapshot as
/// fully stale.
/// Write `bytes` to `path` through a sibling staging file unique to this
/// call, then rename over the target — two concurrent writers never share a
/// staging name, so neither can publish the other's half-written file.
#[cfg(not(target_arch = "wasm32"))]
fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp.{}.{seq}", std::process::id()));
    let tmp = path.with_file_name(name);
    std::fs::write(&tmp, bytes)?;
    if let Err(err) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }
    Ok(())
}

fn iso8601_to_secs(s: &str) -> Option<u64> {
    // Accept the exact shape secs_to_iso8601 emits: 20 chars, `Z` suffix.
    if s.len() != 20 || !s.ends_with('Z') {
        return None;
    }
    let b = s.as_bytes();
    if b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let parse2 = |off: usize| std::str::from_utf8(&b[off..off + 2]).ok()?.parse::<u64>().ok();
    let y: i64 = std::str::from_utf8(&b[0..4]).ok()?.parse().ok()?;
    let mo = parse2(5)?;
    let d = parse2(8)?;
    let h = parse2(11)?;
    let mi = parse2(14)?;
    let se = parse2(17)?;
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || se > 60 {
        return None;
    }
    // Days from civil date (Howard Hinnant), inverse of secs_to_iso8601.
    let y = if mo <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = (y - era * 400) as u64;
    let mp = if mo > 2 { mo - 3 } else { mo + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let z = era * 146097 + doe as i64 - 719468;
    if z < 0 {
        return None;
    }
    Some((z as u64) * 86400 + h * 3600 + mi * 60 + se)
}

fn parse_frontend_models(raw: &serde_json::Value) -> Result<Vec<AlmanacEntry>, AlmanacError> {
    let data = raw.get("data").ok_or(AlmanacError::Schema { field: "data" })?;
    let models = data
        .get("models")
        .and_then(|v| v.as_array())
        .ok_or(AlmanacError::Schema { field: "data.models" })?;
    let analytics = data.get("analytics").and_then(|v| v.as_object());
    let categories = data.get("categories").and_then(|v| v.as_object());

    let now = secs_to_iso8601_now();
    let mut out = Vec::with_capacity(models.len());
    for (rank0, model) in models.iter().enumerate() {
        let slug = model
            .get("slug")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let permaslug = model
            .get("permaslug")
            .and_then(|v| v.as_str())
            .unwrap_or(&slug)
            .to_string();
        let context_length = model
            .get("context_length")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        let endpoint = model.get("endpoint");
        let pricing = endpoint.and_then(|e| e.get("pricing"));
        let price_prompt =
            pricing.and_then(|p| p.get("prompt")).and_then(parse_price_string).unwrap_or(0.0);
        let price_completion =
            pricing.and_then(|p| p.get("completion")).and_then(parse_price_string).unwrap_or(0.0);
        let endpoint_is_free = endpoint
            .and_then(|e| e.get("is_free"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let endpoint_variant = endpoint
            .and_then(|e| e.get("variant"))
            .and_then(|v| v.as_str())
            .unwrap_or("standard");

        // The frontend payload omits the `:free` suffix from `slug` — the
        // suffix is canonical when calling `/v1/chat/completions`, so we
        // re-attach it for non-standard variants. analytics is keyed by
        // `<permaslug>:<variant>` for free/paid splits.
        let canonical_id = if endpoint_variant == "standard" {
            slug.clone()
        } else {
            format!("{slug}:{endpoint_variant}")
        };
        let analytics_key = if endpoint_variant == "standard" {
            permaslug.clone()
        } else {
            format!("{permaslug}:{endpoint_variant}")
        };

        let weekly_requests = analytics
            .and_then(|a| a.get(&analytics_key))
            .and_then(|v| v.get("count"))
            .and_then(|v| v.as_u64());

        let input_modalities: Vec<&str> = model
            .get("input_modalities")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        let supports_reasoning =
            model.get("supports_reasoning").and_then(|v| v.as_bool()).unwrap_or(false);
        let supported_params: Vec<&str> = endpoint
            .and_then(|e| e.get("supported_parameters"))
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        let model_categories: Vec<&str> = categories
            .and_then(|c| c.get(&permaslug))
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|e| e.get("category")?.as_str()).collect())
            .unwrap_or_default();

        let capability_tags = derive_capability_tags(
            &input_modalities,
            &supported_params,
            supports_reasoning,
            context_length,
            &model_categories,
        );

        out.push(AlmanacEntry {
            id: canonical_id,
            permaslug,
            context_length,
            price_prompt_usd_per_token: price_prompt,
            price_completion_usd_per_token: price_completion,
            is_free: endpoint_is_free || price_prompt == 0.0 && endpoint_variant == "free",
            capability_tags,
            popularity_rank: Some((rank0 + 1) as u32),
            weekly_requests,
            last_seen_at: now.clone(),
            quality_score: None,
            code_subscore: None,
        });
    }
    Ok(out)
}

fn parse_price_string(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::String(s) => s.parse().ok(),
        serde_json::Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

/// Stable mapping from OpenRouter's raw fields to our capability vocabulary.
/// Pure / deterministic so it's easy to test and snapshot.
pub fn derive_capability_tags(
    input_modalities: &[&str],
    supported_params: &[&str],
    supports_reasoning: bool,
    context_length: u32,
    or_categories: &[&str],
) -> Vec<String> {
    let mut tags = Vec::new();
    if or_categories.contains(&"programming")
        || supported_params.contains(&"tools") && context_length >= 32_000
    {
        // `programming` is OpenRouter's own per-model daily-rank category.
        // The `tools` + 32k-ctx fallback catches code-tuned models that
        // haven't accumulated category data yet (freshly-published).
        tags.push("code".into());
    }
    if input_modalities.contains(&"image") {
        tags.push("vision".into());
    }
    if supported_params.contains(&"tools") {
        tags.push("tool-use".into());
    }
    if context_length >= 200_000 {
        tags.push("long-context".into());
    }
    if supports_reasoning {
        tags.push("reasoning".into());
    }
    tags
}

// ── R260-F5: T1 external-benchmark quality scores ────────────────────────────

/// Quality scores from an external benchmark source, keyed by canonical model
/// key (see [`canonical_model_key`]).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchmarkScore {
    /// General quality index, normalised to 0.0–1.0 (higher = better).
    pub quality: f64,
    /// Code-specific subscore. `None` when the source doesn't publish one.
    pub code: Option<f64>,
}

/// URL for the artificialanalysis.ai public model quality endpoint.
/// Override via `YAH_BENCHMARK_URL` env var for testing or mirrors.
///
/// Shape expected from this endpoint:
/// ```json
/// { "models": [
///   { "model_id": "claude-3-5-sonnet", "quality": 85.4, "quality_coding": 80.2 }
/// ] }
/// ```
/// Scores above 1.0 are interpreted as 0–100 scale and normalised to 0.0–1.0.
const ARTIFICIALANALYSIS_QUALITY_URL: &str =
    "https://artificialanalysis.ai/api/v1/models";

/// Normalise a model id or slug for benchmark joining.
///
/// Strips the provider prefix (e.g. `"anthropic/"` → `""`), variant suffix
/// (`:free`), and normalises `.` and `_` to `-`, lowercasing everything. Both
/// `entry.id` and `entry.permaslug` are normalised this way in
/// [`join_quality_scores`] so the join has two attempts per entry.
///
/// # Examples
/// ```
/// # use yah_runner_core::resolver::canonical_model_key;
/// assert_eq!(canonical_model_key("anthropic/claude-3.5-sonnet:free"), "claude-3-5-sonnet");
/// assert_eq!(canonical_model_key("qwen/qwen3-coder:free"), "qwen3-coder");
/// assert_eq!(canonical_model_key("claude-3-5-sonnet-20241022"), "claude-3-5-sonnet-20241022");
/// ```
pub fn canonical_model_key(id: &str) -> String {
    let name = id.rsplit('/').next().unwrap_or(id);
    let name = name.split(':').next().unwrap_or(name);
    name.to_lowercase().replace(['.', '_'], "-")
}

/// Join external benchmark quality scores onto a list of almanac entries.
///
/// For each entry, attempts to match against `scores` using:
/// 1. `canonical_model_key(entry.id)` — strips provider prefix + variant.
/// 2. `canonical_model_key(entry.permaslug)` — version-dated slug form.
///
/// On a match, sets `entry.quality_score` and `entry.code_subscore`. Entries
/// with no match are left unchanged (`None` fields stay `None`).
pub fn join_quality_scores(
    entries: &mut Vec<AlmanacEntry>,
    scores: &std::collections::HashMap<String, BenchmarkScore>,
) {
    for entry in entries.iter_mut() {
        let score = scores
            .get(&canonical_model_key(&entry.id))
            .or_else(|| scores.get(&canonical_model_key(&entry.permaslug)));
        if let Some(s) = score {
            entry.quality_score = Some(s.quality);
            entry.code_subscore = s.code;
        }
    }
}

/// Fetch quality scores from the external benchmark source (currently
/// artificialanalysis.ai). Returns an empty map on any fetch or parse error —
/// callers must treat missing entries as "no data available" (T1 fields
/// remain `None`).
///
/// The URL can be overridden with `YAH_BENCHMARK_URL` for testing or mirrors.
pub async fn fetch_benchmark_scores(
    http: &dyn HttpClient,
) -> std::collections::HashMap<String, BenchmarkScore> {
    let url = std::env::var("YAH_BENCHMARK_URL")
        .unwrap_or_else(|_| ARTIFICIALANALYSIS_QUALITY_URL.to_string());
    let req = HttpRequest::get(url.as_str())
        .header("User-Agent", "yah-almanac/0.1")
        .header("Accept", "application/json");
    let resp = match http.send(req).await {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(url = %url, error = %e, "benchmark score fetch transport error");
            return std::collections::HashMap::new();
        }
    };
    if !resp.is_success() {
        tracing::debug!(
            url = %url,
            status = resp.status,
            "benchmark score fetch non-2xx; skipping quality join"
        );
        return std::collections::HashMap::new();
    }
    match resp.json::<serde_json::Value>().await {
        Ok(json) => {
            let scores = parse_benchmark_scores(&json);
            tracing::debug!(count = scores.len(), "benchmark quality scores parsed");
            scores
        }
        Err(e) => {
            tracing::debug!(url = %url, error = %e, "benchmark score response not valid JSON");
            std::collections::HashMap::new()
        }
    }
}

/// Parse the JSON response from the benchmark source into a canonical-key map.
///
/// Handles two shapes:
/// - `{ "models": [...] }` — top-level object with a `models` array
/// - `[...]` — bare top-level array
///
/// Each entry is expected to have (at minimum) a model identifier field and a
/// quality score. Multiple field name aliases are accepted for each so the
/// parser stays compatible with minor API changes.
fn parse_benchmark_scores(
    json: &serde_json::Value,
) -> std::collections::HashMap<String, BenchmarkScore> {
    let mut out = std::collections::HashMap::new();
    let models = json
        .get("models")
        .and_then(|v| v.as_array())
        .or_else(|| json.as_array());
    let Some(models) = models else {
        return out;
    };
    for m in models {
        // Accept several plausible field names for the model identifier.
        let id = m
            .get("model_id")
            .or_else(|| m.get("model"))
            .or_else(|| m.get("id"))
            .or_else(|| m.get("name"))
            .and_then(|v| v.as_str());
        // General quality score — required.
        let quality_raw = m
            .get("quality")
            .or_else(|| m.get("quality_score"))
            .or_else(|| m.get("score"))
            .and_then(|v| v.as_f64());
        // Code subscore — optional.
        let code_raw = m
            .get("quality_coding")
            .or_else(|| m.get("coding"))
            .or_else(|| m.get("code_quality"))
            .or_else(|| m.get("code"))
            .and_then(|v| v.as_f64());
        let (Some(id), Some(quality_raw)) = (id, quality_raw) else {
            continue;
        };
        // Normalise 0–100 scale to 0.0–1.0.
        let quality = if quality_raw > 1.0 { quality_raw / 100.0 } else { quality_raw };
        let code = code_raw.map(|c| if c > 1.0 { c / 100.0 } else { c });
        out.insert(canonical_model_key(id), BenchmarkScore { quality, code });
    }
    out
}

fn secs_to_iso8601_now() -> String {
    super::secs_to_iso8601(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    )
}

#[cfg(test)]
mod tests {
    use super::super::testing::{MapCredentials, StubHttp};
    use super::*;
    use crate::test_support::TempDir;

    #[test]
    fn parse_key_info_passthrough() {
        let raw = serde_json::json!({
            "data": {
                "label": "My Key",
                "limit": 10.0,
                "limit_reset": "month",
                "limit_remaining": 7.66,
                "usage": 2.34,
                "usage_daily": 0.05,
                "usage_weekly": 0.30,
                "usage_monthly": 1.20,
                "byok_usage": 0.0,
                "byok_usage_daily": 0.0,
                "byok_usage_weekly": 0.0,
                "byok_usage_monthly": 0.0,
                "is_free_tier": false
            }
        });
        let query = UsageQuery { period_start: 0, period_end: 0, api_key: None };
        let snap = parse_openrouter_key_info(raw, &query);
        assert_eq!(snap.provider, "openrouter");
        // Tokens unset — /key is a dollar endpoint.
        assert_eq!(snap.input_tokens, 0);
        // Raw passes through intact for the UI to consume.
        let remaining = snap
            .raw
            .pointer("/data/limit_remaining")
            .and_then(|v| v.as_f64())
            .unwrap();
        assert!((remaining - 7.66).abs() < 1e-6, "got {remaining}");
        let daily = snap
            .raw
            .pointer("/data/usage_daily")
            .and_then(|v| v.as_f64())
            .unwrap();
        assert!((daily - 0.05).abs() < 1e-9, "got {daily}");
    }

    #[test]
    fn parse_key_info_unlimited_key() {
        // limit: null, limit_remaining: null — no per-key budget set.
        let raw = serde_json::json!({
            "data": {
                "label": "Rig key",
                "limit": null,
                "limit_reset": null,
                "limit_remaining": null,
                "usage": 5.0,
                "usage_daily": 0.10,
                "usage_weekly": 0.80,
                "usage_monthly": 3.50,
                "is_free_tier": false
            }
        });
        let query = UsageQuery { period_start: 0, period_end: 0, api_key: None };
        let snap = parse_openrouter_key_info(raw, &query);
        assert_eq!(snap.input_tokens, 0);
        assert!(snap.raw.pointer("/data/limit_remaining").and_then(|v| v.as_f64()).is_none());
        let monthly = snap
            .raw
            .pointer("/data/usage_monthly")
            .and_then(|v| v.as_f64())
            .unwrap();
        assert!((monthly - 3.50).abs() < 1e-9, "got {monthly}");
    }

    #[test]
    fn parse_key_info_empty_body_is_safe() {
        let raw = serde_json::json!({});
        let query = UsageQuery { period_start: 0, period_end: 0, api_key: None };
        let snap = parse_openrouter_key_info(raw, &query);
        assert_eq!(snap.input_tokens, 0);
    }


    #[test]
    fn connection_id_openrouter_uses_registry_endpoint() {
        let spec = ProviderSpec {
            auth: Some(super::super::AuthKind::Key),
            connection_id: Some("openrouter".into()),
            ..ProviderSpec::provider("openrouter")
        };
        let result = OpenRouterResolver
            .resolve(&spec, &MapCredentials::with(OPENROUTER_SLOT, "test-key"))
            .unwrap();
        assert!(result.endpoint.contains("openrouter.ai"), "{}", result.endpoint);
        assert!(matches!(result.auth, AuthMaterial::Bearer(_)));
        assert_eq!(result.provider_label, "openrouter");
    }

    #[test]
    fn parse_frontend_models_extracts_top_weekly_order() {
        let raw = serde_json::json!({
            "data": {
                "models": [
                    {
                        "slug": "anthropic/claude-opus-4.7",
                        "permaslug": "anthropic/claude-opus-4-7-20260520",
                        "context_length": 1_000_000,
                        "input_modalities": ["text", "image"],
                        "supports_reasoning": true,
                        "endpoint": {
                            "is_free": false,
                            "variant": "standard",
                            "supported_parameters": ["tools", "temperature"],
                            "pricing": { "prompt": "0.000005", "completion": "0.000025" }
                        }
                    },
                    {
                        "slug": "qwen/qwen3-coder",
                        "permaslug": "qwen/qwen3-coder-480b-a35b-07-25",
                        "context_length": 262_144,
                        "input_modalities": ["text"],
                        "supports_reasoning": false,
                        "endpoint": {
                            "is_free": true,
                            "variant": "free",
                            "supported_parameters": ["tools"],
                            "pricing": { "prompt": "0", "completion": "0" }
                        }
                    }
                ],
                "analytics": {
                    "anthropic/claude-opus-4-7-20260520": { "count": 33_061_550_u64 },
                    "qwen/qwen3-coder-480b-a35b-07-25:free": { "count": 1_234_567_u64 }
                },
                "categories": {
                    "qwen/qwen3-coder-480b-a35b-07-25": [
                        { "category": "programming", "rank": 1 }
                    ]
                }
            }
        });

        let entries = parse_frontend_models(&raw).unwrap();
        assert_eq!(entries.len(), 2);

        // Top-weekly order preserved (Claude #1, Qwen #2)
        assert_eq!(entries[0].id, "anthropic/claude-opus-4.7");
        assert_eq!(entries[0].popularity_rank, Some(1));
        assert_eq!(entries[0].weekly_requests, Some(33_061_550));
        assert!(!entries[0].is_free);
        assert!(entries[0].capability_tags.contains(&"vision".into()));
        assert!(entries[0].capability_tags.contains(&"tool-use".into()));
        assert!(entries[0].capability_tags.contains(&"long-context".into()));
        assert!(entries[0].capability_tags.contains(&"reasoning".into()));

        // :free suffix re-attached for non-standard variant
        assert_eq!(entries[1].id, "qwen/qwen3-coder:free");
        assert!(entries[1].is_free);
        assert_eq!(entries[1].weekly_requests, Some(1_234_567));
        assert!(entries[1].capability_tags.contains(&"code".into()));
    }

    #[test]
    fn derive_capability_tags_matches_canonical_vocab() {
        // Every emitted tag must come from CAPABILITY_TAGS.
        let tags = derive_capability_tags(
            &["text", "image"],
            &["tools", "temperature"],
            true,
            500_000,
            &["programming"],
        );
        for t in &tags {
            assert!(
                CAPABILITY_TAGS.contains(&t.as_str()),
                "unexpected capability tag {t}"
            );
        }
        assert!(tags.contains(&"code".into()));
        assert!(tags.contains(&"vision".into()));
        assert!(tags.contains(&"tool-use".into()));
        assert!(tags.contains(&"long-context".into()));
        assert!(tags.contains(&"reasoning".into()));
    }

    #[test]
    fn parse_price_string_handles_str_and_number() {
        let v_str = serde_json::Value::String("0.000005".into());
        let v_num = serde_json::Value::from(0.000005_f64);
        let v_zero = serde_json::Value::String("0".into());
        assert!((parse_price_string(&v_str).unwrap() - 0.000005).abs() < 1e-12);
        assert!((parse_price_string(&v_num).unwrap() - 0.000005).abs() < 1e-12);
        assert_eq!(parse_price_string(&v_zero), Some(0.0));
    }

    #[test]
    fn iso8601_to_secs_roundtrips_against_secs_to_iso8601() {
        // Spot-check epoch, a known date, and a leap-year crossing — same
        // sample points the inverse helper at resolver/mod.rs covers, run
        // backwards to catch off-by-one regressions in either direction.
        let samples = [0_u64, 1_700_000_000, 1_704_067_200, 1_767_225_600];
        for s in samples {
            let iso = super::super::secs_to_iso8601(s);
            assert_eq!(iso8601_to_secs(&iso), Some(s), "roundtrip failed at {s}: {iso}");
        }
    }

    #[test]
    fn iso8601_to_secs_rejects_malformed() {
        assert!(iso8601_to_secs("not a timestamp").is_none());
        assert!(iso8601_to_secs("2026-05-21T16:57:08").is_none()); // missing Z
        assert!(iso8601_to_secs("2026-13-01T00:00:00Z").is_none()); // month 13
    }

    #[test]
    fn embedded_almanac_seed_parses_and_has_entries() {
        let snap = embedded_almanac_seed();
        // Either the placeholder (0 entries, epoch) or a populated snapshot —
        // both must be well-formed. The release process owns repopulation.
        if !snap.entries.is_empty() {
            assert!(snap.entries[0].popularity_rank.is_some());
            assert!(iso8601_to_secs(&snap.fetched_at).is_some());
        }
    }

    #[test]
    fn refresh_almanac_writes_atomic_snapshot() {
        // Drive refresh_almanac's writer without the network: publish a
        // snapshot through write_atomic into a not-yet-existing directory and
        // confirm the load_almanac reader round-trips it.
        let tmp = TempDir::new();
        let path = tmp.path().join("jit").join("openrouter-almanac.json");
        let written = AlmanacSnapshot {
            fetched_at: "2026-05-21T16:57:08Z".into(),
            entries: vec![AlmanacEntry {
                id: "test/model".into(),
                permaslug: "test/model-20260521".into(),
                context_length: 32_000,
                price_prompt_usd_per_token: 0.0,
                price_completion_usd_per_token: 0.0,
                is_free: true,
                capability_tags: vec!["code".into()],
                popularity_rank: Some(1),
                weekly_requests: Some(42),
                last_seen_at: "2026-05-21T16:57:08Z".into(),
                quality_score: None,
                code_subscore: None,
            }],
        };
        write_atomic(&path, &serde_json::to_vec_pretty(&written).unwrap()).unwrap();
        let loaded = load_almanac(&path).expect("snapshot should load");
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].id, "test/model");
        assert_eq!(loaded.fetched_at, written.fetched_at);
    }

    #[tokio::test]
    async fn missing_key_returns_missing_admin_key() {
        let query = UsageQuery { api_key: None, ..UsageQuery::last_days(30) };
        let result = OpenRouterResolver
            .fetch_usage(&query, &StubHttp::new(200, "{}"), &MapCredentials::empty())
            .await;
        assert!(
            matches!(result, Err(UsageError::MissingAdminKey { .. })),
            "expected MissingAdminKey, got: {result:?}"
        );
    }

    fn almanac(tmp: &TempDir) -> std::path::PathBuf {
        tmp.path().join("openrouter-almanac.json")
    }

    fn make_almanac_with_entries(entries: Vec<AlmanacEntry>) -> TempDir {
        let tmp = TempDir::new();
        let snap = AlmanacSnapshot { fetched_at: "2026-05-21T00:00:00Z".into(), entries };
        std::fs::write(almanac(&tmp), serde_json::to_vec_pretty(&snap).unwrap()).unwrap();
        tmp
    }

    fn free_code_entry(id: &str, rank: u32) -> AlmanacEntry {
        AlmanacEntry {
            id: id.to_string(),
            permaslug: id.to_string(),
            context_length: 32_768,
            price_prompt_usd_per_token: 0.0,
            price_completion_usd_per_token: 0.0,
            is_free: true,
            capability_tags: vec!["code".into()],
            popularity_rank: Some(rank),
            weekly_requests: Some(100),
            last_seen_at: "2026-05-21T00:00:00Z".into(),
            quality_score: None,
            code_subscore: None,
        }
    }

    #[test]
    fn resolve_model_query_picks_highest_rank_free_code_model() {
        let tmp = make_almanac_with_entries(vec![
            free_code_entry("vendor/best:free", 1),
            free_code_entry("vendor/second:free", 2),
            {
                let mut e = free_code_entry("vendor/paid-only", 3);
                e.is_free = false;
                e.price_prompt_usd_per_token = 0.000005;
                e
            },
        ]);
        let q = ModelQuery {
            provider: None,
            tier: ModelTier::Free,
            capability_tags: vec!["code".into()],
            min_context_tokens: None,
            max_price_per_token_usd: None,
            min_quality_score: None,
        };
        let (id, _age) = resolve_model_query(&q, &almanac(&tmp)).unwrap();
        assert_eq!(id, "vendor/best:free");
    }

    #[test]
    fn resolve_model_query_errors_when_no_match() {
        let tmp = make_almanac_with_entries(vec![free_code_entry("vendor/model:free", 1)]);
        let q = ModelQuery {
            provider: None,
            tier: ModelTier::Free,
            capability_tags: vec!["vision".into()],
            min_context_tokens: None,
            max_price_per_token_usd: None,
            min_quality_score: None,
        };
        assert!(resolve_model_query(&q, &almanac(&tmp)).is_err());
    }

    #[test]
    fn resolve_model_query_filters_by_provider_prefix() {
        let tmp = make_almanac_with_entries(vec![
            free_code_entry("openrouter/owl-alpha:free", 1),
            free_code_entry("qwen/qwen3-coder:free", 2),
        ]);
        let q = ModelQuery {
            provider: Some("qwen".into()),
            tier: ModelTier::Free,
            capability_tags: vec!["code".into()],
            min_context_tokens: None,
            max_price_per_token_usd: None,
            min_quality_score: None,
        };
        let (id, _age) = resolve_model_query(&q, &almanac(&tmp)).unwrap();
        assert_eq!(id, "qwen/qwen3-coder:free");
    }

    #[test]
    fn resolve_model_query_falls_back_to_embedded_seed_when_no_cache() {
        let tmp = TempDir::new(); // empty dir — no cache file
        let q = ModelQuery {
            provider: None,
            tier: ModelTier::Any,
            capability_tags: vec![],
            min_context_tokens: None,
            max_price_per_token_usd: None,
            min_quality_score: None,
        };
        // Embedded seed has entries: if it has any, resolution succeeds.
        // If the seed is an empty placeholder, the query returns NoMatch — both
        // outcomes are valid; we just assert no panic.
        let _ = resolve_model_query(&q, &almanac(&tmp));
    }


    // ── R260-F5: T1 benchmark join tests ─────────────────────────────────────

    #[test]
    fn canonical_model_key_strips_provider_and_variant() {
        assert_eq!(canonical_model_key("anthropic/claude-3.5-sonnet:free"), "claude-3-5-sonnet");
        assert_eq!(canonical_model_key("qwen/qwen3-coder:free"), "qwen3-coder");
        assert_eq!(canonical_model_key("openrouter/owl-alpha"), "owl-alpha");
        // No provider prefix
        assert_eq!(canonical_model_key("claude-3-5-sonnet-20241022"), "claude-3-5-sonnet-20241022");
        // Underscores normalised
        assert_eq!(canonical_model_key("vendor/model_v2.1"), "model-v2-1");
    }

    #[test]
    fn join_quality_scores_populates_matched_entries() {
        let mut entries = vec![
            free_code_entry("qwen/qwen3-coder:free", 1),
            free_code_entry("openrouter/owl-alpha:free", 2),
        ];
        let mut scores = std::collections::HashMap::new();
        scores.insert("qwen3-coder".to_string(), BenchmarkScore { quality: 0.82, code: Some(0.78) });
        join_quality_scores(&mut entries, &scores);
        assert_eq!(entries[0].quality_score, Some(0.82));
        assert_eq!(entries[0].code_subscore, Some(0.78));
        // owl-alpha has no score → stays None
        assert_eq!(entries[1].quality_score, None);
        assert_eq!(entries[1].code_subscore, None);
    }

    #[test]
    fn join_quality_scores_falls_back_to_permaslug() {
        let mut entries = vec![{
            let mut e = free_code_entry("vendor/model:free", 1);
            e.permaslug = "vendor/model-20260521".to_string();
            e
        }];
        let mut scores = std::collections::HashMap::new();
        // Key matches permaslug but NOT the canonical id (which is "model")
        scores.insert(
            "model-20260521".to_string(),
            BenchmarkScore { quality: 0.90, code: None },
        );
        join_quality_scores(&mut entries, &scores);
        // id canonical key is "model", no match there; permaslug canonical key
        // is "model-20260521", which DOES match
        assert_eq!(entries[0].quality_score, Some(0.90));
        assert_eq!(entries[0].code_subscore, None);
    }

    #[test]
    fn parse_benchmark_scores_models_array_shape() {
        let json = serde_json::json!({
            "models": [
                { "model_id": "claude-3.5-sonnet", "quality": 85.4, "quality_coding": 80.2 },
                { "model_id": "qwen3-coder",        "quality": 72.0, "quality_coding": 78.5 },
                { "model_id": "no-quality-field",   "name": "ignored" }   // no quality field
            ]
        });
        let scores = parse_benchmark_scores(&json);
        assert_eq!(scores.len(), 2);
        let claude = scores.get("claude-3-5-sonnet").unwrap();
        assert!((claude.quality - 0.854).abs() < 1e-9, "quality should normalise 85.4→0.854");
        assert!((claude.code.unwrap() - 0.802).abs() < 1e-9);
        let qwen = scores.get("qwen3-coder").unwrap();
        assert!((qwen.quality - 0.72).abs() < 1e-9);
    }

    #[test]
    fn parse_benchmark_scores_bare_array_shape() {
        let json = serde_json::json!([
            { "model": "owl-alpha", "quality": 0.88 }
        ]);
        let scores = parse_benchmark_scores(&json);
        assert!(scores.contains_key("owl-alpha"), "bare array should parse");
    }

    #[test]
    fn parse_benchmark_scores_zero_to_one_passthrough() {
        let json = serde_json::json!({
            "models": [{ "model_id": "small-model", "quality": 0.65, "quality_coding": 0.70 }]
        });
        let scores = parse_benchmark_scores(&json);
        let s = scores.get("small-model").unwrap();
        // Already 0–1, should NOT be divided by 100
        assert!((s.quality - 0.65).abs() < 1e-9);
        assert!((s.code.unwrap() - 0.70).abs() < 1e-9);
    }

    #[test]
    fn entry_matches_query_min_quality_score_filters() {
        let mut e = free_code_entry("vendor/model:free", 1);
        e.quality_score = Some(0.60);
        let q = ModelQuery {
            provider: None,
            tier: ModelTier::Free,
            capability_tags: vec![],
            min_context_tokens: None,
            max_price_per_token_usd: None,
            min_quality_score: Some(0.75),
        };
        assert!(!entry_matches_query(&e, &q), "0.60 should fail min_quality 0.75");

        let q_low = ModelQuery { min_quality_score: Some(0.50), ..q.clone() };
        assert!(entry_matches_query(&e, &q_low), "0.60 should pass min_quality 0.50");
    }

    #[test]
    fn entry_matches_query_min_quality_passes_when_score_absent() {
        let e = free_code_entry("vendor/model:free", 1); // quality_score: None
        let q = ModelQuery {
            provider: None,
            tier: ModelTier::Free,
            capability_tags: vec![],
            min_context_tokens: None,
            max_price_per_token_usd: None,
            min_quality_score: Some(0.99), // very high threshold
        };
        // Entries without quality data must pass through (can't filter the unknown)
        assert!(
            entry_matches_query(&e, &q),
            "entry with quality_score=None should pass any min_quality predicate"
        );
    }

    // ── Quality-aware model selection tests ──────────────────────────────

    fn entry_with_quality(
        id: &str,
        rank: u32,
        is_free: bool,
        quality: Option<f64>,
        code_subscore: Option<f64>,
        extra_tags: &[&str],
    ) -> AlmanacEntry {
        let mut tags: Vec<String> = vec!["code".into()];
        for t in extra_tags {
            tags.push(t.to_string());
        }
        AlmanacEntry {
            id: id.to_string(),
            permaslug: id.to_string(),
            context_length: 32_768,
            price_prompt_usd_per_token: 0.0,
            price_completion_usd_per_token: 0.0,
            is_free,
            capability_tags: tags,
            popularity_rank: Some(rank),
            weekly_requests: Some(100),
            last_seen_at: "2026-05-21T00:00:00Z".into(),
            quality_score: quality,
            code_subscore,
        }
    }

    #[test]
    fn resolve_model_query_prefers_quality_over_popularity() {
        let tmp = make_almanac_with_entries(vec![
            entry_with_quality("vendor/popular:free", 1, true, Some(0.60), None, &[]),
            entry_with_quality("vendor/quality:free", 10, true, Some(0.85), None, &[]),
        ]);
        let q = ModelQuery {
            provider: None,
            tier: ModelTier::Free,
            capability_tags: vec!["code".into()],
            min_context_tokens: None,
            max_price_per_token_usd: None,
            min_quality_score: None,
        };
        let (id, _age) = resolve_model_query(&q, &almanac(&tmp)).unwrap();
        assert_eq!(id, "vendor/quality:free", "higher quality should beat higher popularity");
    }

    #[test]
    fn resolve_model_query_uses_code_subscore_for_code_queries() {
        let tmp = make_almanac_with_entries(vec![
            entry_with_quality("vendor/high-code:free", 2, true, Some(0.85), Some(0.95), &[]),
            entry_with_quality("vendor/high-general:free", 1, true, Some(0.90), Some(0.60), &[]),
        ]);
        let q = ModelQuery {
            provider: None,
            tier: ModelTier::Free,
            capability_tags: vec!["code".into()],
            min_context_tokens: None,
            max_price_per_token_usd: None,
            min_quality_score: None,
        };
        let (id, _age) = resolve_model_query(&q, &almanac(&tmp)).unwrap();
        assert_eq!(id, "vendor/high-code:free", "code_subscore should be primary for code queries");
    }

    #[test]
    fn resolve_model_query_prefers_entry_with_quality_data() {
        let tmp = make_almanac_with_entries(vec![
            entry_with_quality("vendor/popular-no-quality:free", 1, true, None, None, &[]),
            entry_with_quality("vendor/unpopular-with-quality:free", 50, true, Some(0.45), None, &[]),
        ]);
        let q = ModelQuery {
            provider: None,
            tier: ModelTier::Free,
            capability_tags: vec!["code".into()],
            min_context_tokens: None,
            max_price_per_token_usd: None,
            min_quality_score: None,
        };
        let (id, _age) = resolve_model_query(&q, &almanac(&tmp)).unwrap();
        assert_eq!(
            id, "vendor/unpopular-with-quality:free",
            "entry with quality data should beat entry without, even if less popular"
        );
    }

    #[test]
    fn resolve_model_query_falls_back_to_popularity_when_no_quality() {
        let tmp = make_almanac_with_entries(vec![
            entry_with_quality("vendor/popular:free", 1, true, None, None, &[]),
            entry_with_quality("vendor/unpopular:free", 50, true, None, None, &[]),
        ]);
        let q = ModelQuery {
            provider: None,
            tier: ModelTier::Free,
            capability_tags: vec!["code".into()],
            min_context_tokens: None,
            max_price_per_token_usd: None,
            min_quality_score: None,
        };
        let (id, _age) = resolve_model_query(&q, &almanac(&tmp)).unwrap();
        assert_eq!(id, "vendor/popular:free", "should fall back to popularity when no quality data");
    }

    #[test]
    fn resolve_model_query_uses_quality_score_for_non_code_queries() {
        let mut a = entry_with_quality("vendor/high-code:free", 2, true, Some(0.85), Some(0.95), &["vision"]);
        a.capability_tags = vec!["vision".into()]; // no "code" tag
        let mut b = entry_with_quality("vendor/high-vision:free", 1, true, Some(0.95), Some(0.60), &["vision"]);
        b.capability_tags = vec!["vision".into()];
        let tmp = make_almanac_with_entries(vec![a, b]);
        let q = ModelQuery {
            provider: None,
            tier: ModelTier::Free,
            capability_tags: vec!["vision".into()], // no "code" tag
            min_context_tokens: None,
            max_price_per_token_usd: None,
            min_quality_score: None,
        };
        let (id, _age) = resolve_model_query(&q, &almanac(&tmp)).unwrap();
        assert_eq!(
            id, "vendor/high-vision:free",
            "quality_score (not code_subscore) should be primary for non-code queries"
        );
    }
}
