//! Provider engine resolution: typed registry + one resolver per provider.
//!
//! Every config decision for an agent session flows through
//! [`ProviderRegistry::resolve`]. Missing credentials are `Err`
//! (`EngineResolutionError::MissingCredential`); there are no silent
//! fallbacks.
//!
//! ## Three-axis dispatch (R359)
//!
//! Resolution draws from three independent sources:
//!
//! 1. **Connection** (`kg/preroll/connections/*.json` or `.yah/connections/`) —
//!    owns `credentialSlot`, `credentialEnv`, `protocol[].baseUrl`.
//! 2. **ProviderResolver** (per-provider impl below) — owns probe/usage
//!    endpoints, default model, and `kind` (local/cloud). Selected by
//!    `engine.provider`; all `OpenaiSdk` connections share `"openai"` since
//!    R359-F3.
//! 3. **WireResolver** (`AnthropicWireResolver` / `OpenAIWireResolver`) — builds
//!    the final chat endpoint from `protocol[].baseUrl`.
//!
//! Adding an OpenAI-compatible backend requires only a Connection JSON — no
//! Rust changes. See `.yah/docs/architecture/A055-connection-keyed-dispatch.md`
//! for the worked example and full resolution-order spec.
//!
//! The registry is consumed by:
//! - session-start code in `app/yah/desktop/src/agent.rs` (F1)
//! - preflight probes before `session_started` is persisted (F2)
//! - `agent_list_models` for endpoint/auth extraction (F1)
//!
//! F2 adds `probe()` to the trait; F1 ships the synchronous `resolve()` path.
//!
//! ## Usage/Billing Endpoint Findings (R114-S1)
//!
//! ### Anthropic
//! - Usage: GET `https://api.anthropic.com/v1/organizations/usage_report/messages`
//! - Cost:  GET `https://api.anthropic.com/v1/organizations/cost_report`
//! - Auth:  `x-api-key: <admin-key>` — Admin API key (`sk-ant-admin…`), NOT the standard
//!   completion key. Only organization members with the admin role can create these via
//!   Console → Settings → Admin Keys. **Unavailable for individual (non-org) accounts.**
//! - Required headers: `anthropic-version: 2023-06-01`
//! - Key params: `starting_at` / `ending_at` (ISO-8601), `bucket_width` (`1m`/`1h`/`1d`),
//!   `group_by[]` (model, workspace_id, api_key_id, service_tier, inference_geo…)
//! - Response: paginated `{ data: [{ start_time, end_time, results: [{ input_tokens,
//!   output_tokens, cache_creation_input_tokens, cache_read_input_tokens, … }] }],
//!   has_more, next_page }`
//! - Cost response: amounts in USD as decimal strings (lowest unit = cents), grouped by
//!   workspace_id or description. Priority Tier costs NOT included in cost endpoint.
//! - Credential slot needed: `anthropic-admin-key` (new slot, distinct from `ANTHROPIC_API_KEY`).
//!   Usage credentials CANNOT piggyback on runtime auth — completely different key type.
//!
//! ### OpenAI
//! - Usage: GET `https://api.openai.com/v1/organization/usage/completions`
//! - Cost:  GET `https://api.openai.com/v1/organization/costs`
//! - Subscription/credits: GET `https://api.openai.com/v1/dashboard/billing/subscription`
//!   and GET `https://api.openai.com/v1/dashboard/billing/credit_grants`
//! - Auth:  `Authorization: Bearer <admin-key>` — Admin API key, distinct from standard
//!   completion key. Obtainable from platform.openai.com settings.
//! - Key params: `start_time` / `end_time` (Unix timestamps in seconds), `bucket_width`
//!   (`1m`/`1h`/`1d`), `group_by[]` (model, project_id, user_id, api_key_id…)
//! - Response: paginated `{ data: [{ object: "bucket", start_time, end_time,
//!   results: [{ input_tokens, output_tokens, input_cached_tokens, num_model_requests,
//!   model, project_id }] }], next_page }`
//! - Cost response: `{ amount: { value: f64, currency: "usd" }, line_item, project_id }`
//! - Credential slot needed: `openai-admin-key` (new slot, distinct from `OPENAI_API_KEY`).
//!
//! ### Local Ollama
//! - No usage/billing API. Local process; no quota or spend tracking. N/A.
//! - fetch_usage() should return `Unsupported`; session tokens from in-memory store only.
//!
//! ### Ollama Cloud (ollama.com)
//! - Auth to inference API: `Authorization: Bearer <OLLAMA_API_KEY>` (existing slot).
//! - Billing model: GPU-time-based. Plans: Free / Pro ($20/mo) / Max ($100/mo).
//! - Session quota resets every 5 hours; weekly quota resets every 7 days. The web UI at
//!   ollama.com shows `{used_pct}% used / Resets in {N} hours` for both windows.
//! - **No official, documented API endpoint** for quota data (as of 2026-05-09).
//!   GitHub issues #15132 (closed DUPLICATE) and #15663 (open) request this feature.
//! - Community-proposed shape (not confirmed to exist server-side):
//!     GET https://ollama.com/api/account/usage
//!     Authorization: Bearer <api_key>
//!     → { session: { used_percentage, resets_at }, weekly: { used_percentage, resets_at }, plan }
//! - F2 decision: probe `GET https://ollama.com/api/account/usage` with the existing key;
//!   if it 404s or 401s, fall back to `Unsupported` and surface session tokens only.
//!   Do NOT hard-fail; render a "quota data unavailable" note in the UI.
//!
//! ### Design decision for F2
//! - Add two new credential slots: `anthropic-admin-key` and `openai-admin-key`.
//! - Extend ProviderResolver with `async fn fetch_usage(&self, ...) -> UsageResult`.
//! - UsageResult should be a Result<UsageSnapshot, UsageError> where UsageError has an
//!   `Unsupported` variant (for local/Ollama) and `MissingAdminKey` variant.
//! - On the UI side (F2/T4): check for `Unsupported` and render "session tokens only"
//!   rather than an amber error strip.
//!
//! archived: R114 — history: `yah board history oss/runner/crates/core/src/resolver/mod.rs`
//!
//! @yah:relay(R205, "OpenRouter backend + free-tier thief bundle")
//! @yah:at(2026-05-16T21:43:27Z)
//! @yah:next("Crack into F1 (bundled connection + thief subclass), F2 (OpenRouterResolver), F3 (per-tier ANTHROPIC_DEFAULT_* env vars in spawn_claude).")
//!
//! @yah:ticket(R205-F2, "OpenRouterResolver — usage + free-tier 429 hint")
//! @yah:assignee(agent:claude)
//! @yah:at(2026-05-16T21:43:53Z)
//! @yah:status(review)
//! @yah:parent(R205)
//! @yah:next("New file crates/yah/runner/src/resolver/openrouter.rs. Mostly clones OpenAiResolver (resolver/openai.rs) — same chat-completions client, same SSE transport. Register in ProviderRegistry at resolver/mod.rs:448 as ('openrouter', Box::new(OpenRouterResolver)).")
//! @yah:next("fetch_usage(): GET https://openrouter.ai/api/v1/credits with Bearer key; map to UsageResult so the credits/usage panel shows OpenRouter balance alongside Anthropic/OpenAI.")
//! @yah:next("Error mapping: on 429 from a :free model, surface a hint distinguishing 'free-tier daily cap, top up $10 for 1k/day' from generic rate limit. Carries the practical pain from the video transcript.")
//! @yah:next("Vault slot 'openrouter-api-key' — follow the pattern at runner/src/resolver/ollama.rs:58.")
//! @yah:verify("cargo test -p runner resolver::openrouter; manual: a session pinned to an openrouter subclass returns credits in the usage panel")
//! @yah:depends_on(R205-F1)
//! @yah:cleanup("Quota-aware fallback + account-pool upgrade path: see .yah/docs/working/W094-quota-aware-subclass-fallback.md. Once basic OpenRouter backend lands, the 429 free-tier handler is the natural entry point for predictive routing — populate a QuotaLedger from X-RateLimit-* headers and let the subclass dispatch chain consult it before sending. Pool/round-robin is the killer-app extension on top.")
//! @yah:handoff("Shipped crates/yah/runner/src/resolver/openrouter.rs — OpenRouterResolver clones the OpenAI shape (registry-driven endpoint, Bearer key, models_url probe). fetch_usage hits https://openrouter.ai/api/v1/credits, parses {total_credits,total_usage} into UsageSnapshot.raw with an injected remaining_credits field (tokens stay 0 — /credits is a dollar endpoint, not a token endpoint). Registered as ('openrouter', OpenRouterResolver) in ProviderRegistry::production. New OPENROUTER_SLOT='openrouter-api-key' / OPENROUTER_ENV='OPENROUTER_API_KEY' constants. Free-tier 429 hint: pub fn free_tier_429_hint(model) returns Some(...) only when model ends with ':free'; wired into openai_compat::issue_chat_completions to emit a tracing::warn (with the 'top up $10 for 1000 req/day' guidance) when provider_label=='openrouter' && model ends with ':free'. No RunnerError API change — error variant stays RateLimited{retry_after_secs}. 6 new tests in resolver::openrouter pass; cargo check -p runner clean.")
//! @yah:next("Structured hint plumbing: add hint: Option<String> to RunnerError::RateLimited (+ ConnectionStatus::RateLimited + ErrorKind::RateLimited) so the UI can show the free-tier message inline, not just in logs. Currently the hint surfaces via tracing::warn only.")
//! @yah:verify("cargo test -p runner --lib resolver::openrouter — 6/6 pass.")
//! @yah:verify("Manual: a session pinned to an openrouter subclass with OPENROUTER_API_KEY set returns credits in the usage panel; a free-tier 429 emits the actionable warn in logs.")
//! @yah:gotcha("OpenRouter /credits returns dollars, not tokens. UsageSnapshot.input_tokens/output_tokens/requests all stay 0 by design; the UI must render snap.raw.data.{total_credits,total_usage,remaining_credits} for this provider rather than the token fields it uses for Anthropic/OpenAI.")
//!
//! @yah:relay(R215, "Subclass probe + value report")
//! @yah:at(2026-05-17T22:10:23Z)
//! @yah:next("Land a probe-suite runner that routes through the existing ResolvedEngineConfig so probes test the real dispatch path with no parallel-shape drift")
//! @yah:next("Two probe families share one harness + budget envelope: health/* (operational) and capability/* (quality); report two scores side by side, never collapsed")
//! @yah:next("Surface as a dedicated screen under the Run tab, entered via deep-link prefill from the Character editor; subclass is the unit of testing")
//! @yah:next("Gnome-as-judge handles subjective scoring so capability probes aren't bottlenecked by the subclass under test")
//! @yah:verify("cargo test -p probe — engine, registry, budget gating, scorers")
//! @yah:verify("Live: probe a cache-capable subclass (anthropic) and a known no-cache subclass (ollama-cloud); verdicts and TTFT deltas distinguish them correctly")
//! @yah:verify("UX: Character editor 'Probe this subclass' → RunProbeScreen prefilled via URL; refresh preserves selection; localStorage untouched")
//!
//! @yah:relay(R310, "Wire-protocol-keyed resolver dispatch")
//! @yah:at(2026-05-24T02:24:38Z)
//! @yah:next("T1 first (data layer — ProtocolEntry + Connection + JSON), no code deps. T2 next (WireResolver trait + registry). T3 last (delete deepseek.rs, simplify per-resolver URL construction). Each ticket is independently reviewable.")
//! @arch:see(crates/yah/kg/preroll/connections/deepseek.json)
//!
//! @yah:ticket(R310-T3, "Delete deepseek.rs, simplify per-resolver URL construction")
//! @yah:at(2026-05-24T02:25:03Z)
//! @yah:status(review)
//! @yah:parent(R310)
//! @yah:depends_on(R310-T2)
//! @yah:verify("cargo check --workspace passes. hao_smoke example compiles. All 7 bundled providers resolve correctly end-to-end (probe + resolve).")
//! @yah:handoff("Deleted deepseek.rs (500+ lines). Added thin inline DeepSeekResolver (~110 lines) in mod.rs: resolve() reads DEEPSEEK_SLOT/DEEPSEEK_ENV, returns AnthropicWire default (WireResolver overrides at registry level); probe() uses /user/balance endpoint (works for both wires); fetch_usage() hits /user/balance. Removed 'deepseek' arm from AnthropicResolver.resolve() and cleaned up DEEPSEEK_* imports. Removed resolve_anthropic_endpoint() helper from anthropic.rs — resolvers now return hardcoded default endpoints that WireResolver overrides from ProtocolEntry when connection_id is set. Same simplification applied to openai.rs, groq.rs, openrouter.rs (removed dead connection_id endpoint-building blocks). ollama.rs: removed endpoint_for_connection() helper, updated resolve_from_connection() to use constants directly. Added /chat/completions to ollama_base_url() strip list (fixes list_local_models for WireResolver-overridden endpoints). hao_smoke.rs: conn.base_url() → conn.url_for_wire(SubclassWireProtocol::OpenaiWire). 58 resolver tests pass, workspace cargo check clean.")
//! @yah:verify("cargo check --workspace # clean")
//! @yah:verify("cargo test -p runner --lib resolver # 58/58 pass (down from 68: 7 deepseek direct-resolver tests removed + 3 anthropic deepseek tests removed, covered by WireResolver tests)")
//! @yah:verify("cargo test -p runner --lib resolver::ollama # 10/10 pass including new ollama_base_url_strips_chat_suffixes test")
//!
//! @yah:ticket(R310-T2, "WireResolver trait + protocol-keyed registry")
//! @yah:assignee(agent:claude)
//! @yah:at(2026-05-24T02:24:55Z)
//! @yah:status(review)
//! @yah:parent(R310)
//! @yah:next("Add WireResolver trait to resolver/mod.rs: fn build_endpoint(entry: &ProtocolEntry) -> String. Register AnthropicWireResolver and OpenAIWireResolver keyed by SubclassWireProtocol (alongside or replacing the current provider-string registry). Update ProviderRegistry dispatch so it: (1) looks up the Connection to negotiate a ProtocolEntry, (2) gets credentials from the provider-keyed credential fn, (3) calls the wire resolver. ResolvedEngineConfig.wire_protocol becomes non-optional (always set from ProtocolEntry.wire).")
//! @yah:gotcha("Provider-specific concerns (vault slot name, default model, fetch_usage endpoint) stay provider-keyed — only endpoint-building and wire-framing move to the wire layer. Don’t collapse credential lookup into WireResolver.")
//! @yah:depends_on(R310-T1)
//! @yah:handoff("WireResolver trait + AnthropicWireResolver + OpenAIWireResolver added to resolver/mod.rs. ProviderRegistry.resolve() now does 3-step dispatch: (1) negotiate_protocol() loads Connection by subclass.divination.connection_id and calls effective_protocol() to select a ProtocolEntry, (2) provider resolver supplies credentials+metadata, (3) wire resolver overrides endpoint (subprocess:// endpoints exempt) and sets wire_protocol from the ProtocolEntry. ResolvedEngineConfig.wire_protocol is now non-optional (SubclassWireProtocol, not Option<>): anthropic resolvers → AnthropicWire, OpenAI/Groq/Ollama/OpenRouter resolvers → OpenaiWire, ClaudeCli → Process. deepseek.rs probe() updated to match non-optional wire_protocol. agent.rs updated at 5 call sites (Some(X)|None → X|Process patterns). Fixed openai.json baseUrl (https://api.openai.com → https://api.openai.com/v1) and deepseek.json openai_wire baseUrl (https://api.deepseek.com → https://api.deepseek.com/v1) so WireResolver builds correct /v1/chat/completions endpoints. 5 new unit tests for WireResolver, 7/7 deepseek tests, 63/63 resolver tests pass. cargo check --workspace clean.")
//! @yah:verify("cargo check --workspace # clean")
//! @yah:verify("cargo test -p runner --lib resolver # 68 pass (63 pre-existing + 5 new wire resolver tests)")
//! @yah:verify("cargo test -p kg --lib party::tests::connection # 3/3 pass")
//! @yah:verify("cargo test -p runner --lib resolver::deepseek # 7/7 pass")
//!
//! @yah:relay(R359, "Connection-keyed credentials (lift credential slot off provider resolver)")
//! @yah:at(2026-05-31T00:22:42Z)
//! @yah:status(review)
//! @yah:handoff("F1–F4 complete. F1: credentialSlot/credentialEnv on Connection JSON + Rust struct, backfilled 6 bundled connections. F2: ProviderRegistry::resolve step 2.5 reads credential from connection slot, bypassed for subscription auth and inline api_key. F3: (RunnerKind::OpenaiSdk, _) band-aid collapsed to \"openai\"; deepseek/groq/openrouter/ollama-cloud all dispatch through the openai arm; groq session-start bug fixed. F4: A055-connection-keyed-dispatch.md + module doc expansion.")
//! @yah:verify("A subclass with connection_id=ollama-cloud + runner=yah + auth=key resolves credentials from the `ollama` slot end-to-end, with no provider-name special-case in engine_ref_from_subclass.")
//! @yah:verify("Adding a custom Connection JSON with a new credentialSlot is sufficient to make that backend usable — no Rust changes required.")
//! @yah:verify("cargo test -p runner --lib resolver passes (per-provider slot fallback tests still cover the direct-EngineRef caller path).")
//! @yah:gotcha("Today's band-aid mapping in engine_ref_from_subclass (ollama-cloud/ollama-local/groq/openrouter alongside the pre-existing deepseek case) is the visible symptom — it collapses once credentials are connection-driven.")
//! @yah:gotcha("anthropic-gc (Subscription auth) reads a different slot than anthropic (Key auth) despite sharing connection_id. Don't drop the auth axis from credential dispatch.")
//! @yah:assumes("Custom connections will be common as yubaba / managed rigs land more endpoints. If they stay rare, the band-aid is fine and this relay isn't worth landing.")
//!
//! @yah:ticket(R359-F2, "Wire connection-keyed credential lookup into ProviderRegistry::resolve")
//! @yah:assignee(agent:claude)
//! @yah:at(2026-05-31T00:23:09Z)
//! @yah:status(review)
//! @yah:parent(R359)
//! @yah:next("Insert a credential-lookup step in ProviderRegistry::resolve between negotiate_protocol and the provider-resolver call: if a Connection is in scope and carries credentialSlot, read keys::get_or_env(slot, env). Pass the resolved key down to the provider resolver — either via a new param on ProviderResolver::resolve or by stashing on a small ResolveContext struct that resolvers consult before their hardcoded slot fallback.")
//! @yah:next("Provider resolvers keep their hardcoded slot constants as a fallback for direct-EngineRef callers (no subclass / no connection in scope) — same shape as today's subclass.divination.api_key precedence.")
//! @yah:next("Subscription auth (anthropic-gc) still routes to anthropic-oauth slot. Keep the auth-axis branch — either special-case in the registry credential step (auth == Subscription → read anthropic-oauth slot instead of connection.credentialSlot) or carry a parallel `oauthSlot` field on Connection. Pick whichever is smaller; the first is fine for now since anthropic-gc is the only subscription backend.")
//! @yah:verify("cargo test -p runner --lib resolver passes (existing per-provider slot fallback tests cover the direct-EngineRef path).")
//! @yah:verify("New test: subclass with connection_id=ollama-cloud + runner=yah + auth=key, with a key in the `ollama` vault slot, resolves successfully through OpenAiResolver (band-aid removed in F3 — for now the test should pass via the band-aid OR the new path).")
//! @yah:verify("Manual: live Kimi-via-ollama-cloud subclass starts a session without the band-aid match in engine_ref_from_subclass (comment out the ollama-cloud arm, re-run).")
//! @yah:depends_on(R359-F1)
//! @yah:handoff("Added connection-keyed credential override in ProviderRegistry::resolve (step 2.5, R359-F2). Refactored negotiate_protocol → negotiate returning (Option<Connection>, Option<ProtocolEntry>) in one registry load. After provider resolver runs: if connection has credential_slot and auth is not subscription and no inline api_key, look up the key and override cfg.auth = Bearer(k). Added 3 unit tests covering: ollama-cloud slot fields, all 6 bundled key-auth connections, anthropic subscription guard. Also fixed pre-existing missing forked_from field in 6 runner test AgentSubclass literals (groq.rs, ollama.rs x2, openai.rs, openrouter.rs, auto_continue.rs). 61/61 resolver tests pass, cargo build -p yah clean.")
//! @yah:verify("cargo build -p yah # clean")
//! @yah:verify("cargo test -- resolver # 61/61 pass")
//! @yah:verify("cargo test -- resolver::tests::negotiate_ # 3/3 pass")
//!
//! @yah:ticket(R359-F4, "Document the three-axis dispatch: Connection / ProviderResolver / WireResolver")
//! @yah:assignee(agent:claude)
//! @yah:at(2026-05-31T00:23:32Z)
//! @yah:status(review)
//! @yah:parent(R359)
//! @yah:verify("Doc at .yah/docs/architecture/A055-connection-keyed-dispatch.md renders cleanly and includes the custom-backend worked example.")
//! @yah:depends_on(R359-F3)
//!
//! @arch:see(crates/yah/kg/preroll/connections/deepseek.json)
//! @arch:see(crates/yah/runner/src/resolver/openai.rs)
//! @arch:see(crates/yah/runner/src/resolver/anthropic.rs)
//!
//! @yah:ticket(R365-B17, "ProviderRegistry: no resolver registered for 'ollama-cloud'")
//! @yah:assignee(agent:claude)
//! @yah:at(2026-06-04T01:59:43Z)
//! @yah:status(review)
//! @yah:parent(R365)
//! @yah:severity(medium)
//! @yah:next("Inspect crates/yah/runner/src/resolver/ollama.rs and check whether it registers 'ollama-cloud' as a distinct entry or only 'ollama'. Add an ollama-cloud entry that builds an OpenAiCompatConfig pointing at the ollama-cloud endpoint (https://ollama.com/api/...) with the appropriate auth header shape.")
//! @yah:next("Mirror the openai/deepseek registration pattern in resolver/mod.rs and add an `register_ollama_cloud(registry)` call.")
//! @yah:next("Add a resolver test that registers ollama-cloud and resolves a kimi-k2.6 subclass to a valid OpenAiCompatConfig.")
//! @yah:verify("subagent_spawn against the Llama custom subclass (sc:custom:1779908933598:1779909200000) returns a SessionId, not 'no resolver registered'")
//! @yah:gotcha("Reproduces against pre-F13 daemon. `subagent_spawn {subclass: \"sc:custom:1779908933598:1779909200000\"}` (Llama: runner=yah, connection=ollama-cloud, model=kimi-k2.6) → daemon returns -32603 'subagent: engine resolution: invalid subclass traits for ollama-cloud: no resolver registered for provider \\'ollama-cloud\\''. The Llama subclass is correctly tagged runner=yah/connectionId=ollama-cloud but ProviderRegistry has no resolver entry for the ollama-cloud connection id (there's an ollama.rs resolver but apparently no ollama-cloud entry).")
//! @yah:handoff("Registered 'ollama-cloud' as a second key for OllamaResolver in ProviderRegistry::production(). camp.rs constructs EngineRef.provider from subclass.divination.connection_id() directly, yielding 'ollama-cloud' — without this entry, InvalidTraits was returned. OllamaResolver.resolve_from_connection already handles the ollama-cloud case; step 2.5 (connection credential override) and step 3 (WireResolver → openai_wire endpoint) both proceed correctly. Added 2 regression tests: registry_resolves_ollama_cloud_via_ollama_cloud_arm (key present → Bearer ok) and registry_reports_ollama_slot_for_missing_key_via_ollama_cloud_arm (missing key → MissingCredential slot=ollama). 66/66 resolver tests pass.")
//! @yah:verify("cargo test -p runner --lib resolver — 66/66 pass incl. 2 new B17 tests")
//!
//!
//! @yah:ticket(R950-F6, "Move resolver/ into core behind ProviderSpec + CredentialSource traits; party AgentSubclass projection and fob key lookup stay in the wrapper")
//! @yah:at(2026-10-05T05:07:39Z)
//! @yah:assignee(agent:bundle-anthropic-ashguard)
//! @yah:parent(R950)
//! @yah:next("W357 lists resolver/ as move-as-is; it is NOT camp-free. Measured party/kg/fob edges: party::{AgentSubclass, Connection, ProtocolEntry, RunnerKind, SubclassAuth, SubclassWireProtocol, Divination, ModelQuery, ModelTier, ModelSpec, ThinkingBudget, load_connections, effective_protocol}, kg::anno::EngineRef (every resolver), fob::get_or_env (24 calls across 6 files), kg::atomic_write::write_atomic (openrouter).")
//! @yah:next("Core gets a wire-neutral ProviderSpec (endpoint, wire protocol, auth kind, model query) and a CredentialSource trait (slot + env var -> secret). Resolvers consume those and return a wire-neutral resolved endpoint; `into_openai_compat()` becomes a fn in ho so core never depends on ho (today resolver/mod.rs imports openai_compat::{OpenAiCompatConfig, Transport}).")
//! @yah:next("Wrapper keeps: AgentSubclass/Connection -> ProviderSpec projection, a fob-backed CredentialSource, load_connections/effective_protocol. openrouter's write_atomic becomes a local helper (or caller-supplied cache dir). Probes go through core HttpClient.")
//! @yah:next("Coordinate R365: resolver bugs there (R365-B15, R365-B17 ollama-cloud) remain R365's; land this move without changing their behaviour.")
//! @yah:next("Tier: Warrior — six resolvers through two new traits; the ProviderSpec shape is the only judgment call and F2 sets its neighbours.")
//! @yah:verify("cd oss/runner && cargo test -p yah-runner-core resolver")
//! @yah:verify("cargo test -p yah-runner resolver")
//! @yah:verify("cargo run -p yah-runner --bin model-drift still runs")
//! @yah:depends_on(R950-F5)
//! @yah:files(oss/runner/crates/core/src/lib.rs)
//! @yah:files(oss/runner/crates/ho/src/lib.rs)
//! @yah:files(crates/yah/runner/src/resolver/mod.rs)
//! @yah:files(crates/yah/runner/src/resolver/anthropic.rs)
//! @yah:files(crates/yah/runner/src/resolver/ollama.rs)
//! @yah:files(crates/yah/runner/src/resolver/openai.rs)
//! @yah:files(crates/yah/runner/src/resolver/openrouter.rs)
//! @yah:files(crates/yah/runner/src/resolver/groq.rs)
//! @yah:files(crates/yah/runner/src/resolver/claude_cli.rs)
//! @yah:files(crates/yah/runner/src/resolver/openrouter-almanac-seed.json)
//! @yah:files(crates/yah/runner/src/launch.rs)
//! @yah:files(crates/yah/runner/src/lib.rs)
//! @yah:status(review)
//! @yah:handoff("CORE (oss/runner/crates/core/src/resolver/, this block moved with mod.rs): ProviderRegistry::production(Arc<dyn CredentialSource>, Arc<dyn HttpClient>) with resolve(&ProviderSpec)/probe/fetch_usage/http(); ProviderResolver::{resolve(spec, &dyn CredentialSource), probe(cfg, &dyn HttpClient), fetch_usage(query, http, creds)}; all six resolver files, the DeepSeek/Kimi/KimiCli/CodexCli inline resolvers, probe cache, slot consts, almanac + seed json. Wire-neutral types: ProviderSpec{provider, inline_key, auth: Option<AuthKind>, connection_id, connection: Option<ConnectionSpec{id, credential_slot, credential_env, protocol: Option<ProtocolEndpoint{wire, base_url}>}>, reasoning}, WireProtocol and AuthKind (variant-for-variant mirrors of SubclassWireProtocol/SubclassAuth), CredentialSource::resolve(slot, env_var: Option<&str>) -> Result<Option<String>, CredentialError>. Result is kept, not the bare Option the brief suggested, so a vault read failure still surfaces as EngineResolutionError::VaultRead. ResolvedEngineConfig was renamed ResolvedEndpoint, and its thinking_budget field became reasoning: Option<ReasoningBudget>. ModelQuery/ModelTier/ModelSpec moved verbatim (inherent impls included); party.rs re-exports them, so there is one definition. ProviderSpec carries no model field because no resolver reads the model.")
//! @yah:handoff("HTTP + FS: every probe, usage query and model list goes through &dyn HttpClient. Groq's 10s timeout is preserved by a new core HttpRequest.timeout field (ReqwestClient applies it). reqwest .query() was replaced by a core with_query form-encoder, tested to match reqwest. OpenRouter's tokio::join! became futures_util::join! (new core dep futures-util; tracing moved from the non-wasm target deps to main deps). kg::atomic_write became a core-local write_atomic under cfg(not wasm32), along with refresh_almanac. load_almanac/refresh_almanac/resolve_model_query take the almanac file path; the camp path .yah/jit/openrouter-almanac.json stays in the wrapper as almanac_cache_path.")
//! @yah:handoff("HO: oss/runner/crates/ho/src/resolved.rs = impl From<ResolvedEndpoint> for OpenAiCompatConfig, the old into_openai_compat body minus with_camp_policy (+2 tests). WRAPPER: crates/yah/runner/src/resolver/ dir deleted, replaced by resolver.rs: pub use yah_runner_core::resolver::* (old paths keep working), FobCredentials over fob::get_or_env, provider_spec(engine, subclass, camp_root) projection (load_connections + effective_protocol; ClaudeCli/KimiCli/CodexCli runners route to their resolver with no connection, which is exactly what the old step-1.5 short-circuit did), wire_protocol()/auth_kind() mappings, CampRegistry (Default + with_camp_root + resolve(engine, subclass)/probe/fetch_usage/http) replacing ProviderRegistry::default()/production(). openai_compat.rs gains camp_openai_compat(resolved) = with_camp_policy(From) and owns free_tier_rate_limit_hint (it returns a ho type, so core cannot hold it).")
//! @yah:handoff("CALL SITES (renames, no shims): agent-tools agent_dispatch_tools.rs, hub in_process.rs, runner launch.rs, desktop agent.rs/probe.rs/backend.rs/dictate.rs, cli camp.rs/serve.rs/mcp/tools.rs/agent.rs, e2e lib.rs. Every resolved.into_openai_compat() is now runner::camp_openai_compat(resolved). Desktop's two resolved.wire_protocol matches use runner::WireProtocol. The list_local_models/list_ollama_cloud_models/openrouter_fetch_models/refresh_openrouter_almanac callers pass &runner::ReqwestClient::default() plus runner::almanac_cache_path(camp_root).")
//! @yah:handoff("TEST ACCOUNTING: old wrapper resolver tests = 70. 57 moved to core and were made deterministic: in-memory MapCredentials + StubHttp replace the env/vault-skip guards and the one live-network groq test. 13 stayed in the wrapper: 11 from mod.rs (the bundled-connection, negotiate, R365-B1/B17 registry and claude_cli short-circuit tests), preroll_openrouter_queries (reads kg preroll data) moved into resolver.rs, and free_tier_hint moved into openai_compat.rs. New tests: 6 in core, 1 in the wrapper (almanac_cache_path), 2 in ho. The wrapper's canonical_model_key doctest moved to core.")
//! @yah:handoff("R365: the R365-B15/B17 code moved as-is with no behaviour change. R365 should re-check oss/runner/crates/core/src/resolver/ollama.rs (resolve + resolve_from_connection, the ollama-cloud arm), mod.rs ProviderRegistry::production/resolve (the ollama-cloud key and the step-2 connection-slot override + guard), and crates/yah/runner/src/resolver.rs provider_spec/negotiate, which now owns connection lookup. Not touched: ho/examples/** and oss/runner/README.md (R950-T12). board.review via MCP returned EAGAIN (os error 35) and did not land, so this block was written by hand.")
//! @yah:verify("cd oss/runner && cargo test -p yah-runner-core -p yah-runner-ho -> core lib 195 (132 + 57 moved + 6 new; 63 under resolver::), core doctest 1 (moved), ho lib 102 (100 + 2), ho e2e 12 / 1 ignored. EXIT=0 /tmp/r950f6-oss-t1.log")
//! @yah:verify("cargo test -p yah-runner -> lib 322 / 2 ignored (378 - 57 moved + 1 new), doctests 0 (the 1 moved to core). EXIT=0 /tmp/r950f6-run-t1.log; cargo test -p yah-runner resolver -> 13 pass EXIT=0")
//! @yah:verify("cargo run -p yah-runner --bin model-drift -> EXIT=0, both drift checks ok (/tmp/r950f6-drift1.log)")
//! @yah:verify("cargo check -p yah-party -p yah -p desktop -p yah-hub -p yah-eval -p probe -p yah-agent-tools -p yah-e2e --tests EXIT=0, no warnings in touched lines (/tmp/r950f6-x-c1.log); cargo check -p yah-runner --tests --bins --examples EXIT=0; cd oss/runner && cargo check --workspace --tests EXIT=0")
//! @yah:verify("cd oss/runner && cargo tree -p yah-runner-core -e normal / -p yah-runner-ho -e normal -> 0 lines matching yah-(board|party|kg|fob|bridge); cargo check -p yah-runner-core -p yah-runner-ho --no-default-features --target wasm32-unknown-unknown EXIT=0 and --target aarch64-apple-ios EXIT=0")


pub mod anthropic;
pub mod claude_cli;
pub mod groq;
pub mod ollama;
pub mod openai;
pub mod openrouter;

pub use anthropic::AnthropicResolver;
pub use claude_cli::ClaudeCliResolver;
pub use groq::{fetch_groq_models, GroqResolver};
pub use ollama::{list_cloud_models as list_ollama_cloud_models, list_local_models, OllamaResolver};
pub use openai::OpenAiResolver;
#[cfg(not(target_arch = "wasm32"))]
pub use openrouter::refresh_almanac as refresh_openrouter_almanac;
pub use openrouter::{
    canonical_model_key, derive_capability_tags, embedded_almanac_seed, fetch_benchmark_scores,
    fetch_models as openrouter_fetch_models, join_quality_scores,
    load_almanac as load_openrouter_almanac, resolve_model_query, AlmanacEntry, AlmanacError,
    AlmanacSnapshot, BenchmarkScore, ModelQueryError, OpenRouterResolver, CAPABILITY_TAGS,
};

use crate::http::{HttpClient, HttpRequest};
use crate::prelude::ReasoningBudget;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Provider spec — what the host tells the registry about one session
// ---------------------------------------------------------------------------

/// Wire protocol a connection speaks. Drives which HTTP framing the runner
/// uses and which [`WireResolver`] builds the final chat endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireProtocol {
    /// Anthropic `/v1/messages` wire format.
    AnthropicWire,
    /// OpenAI `/v1/chat/completions` format. Covers Ollama Cloud, vLLM,
    /// Together, and any OpenAI-shaped proxy.
    OpenaiWire,
    /// MCP JSON-RPC transport.
    McpWire,
    /// Subprocess runner — wire is internal to the binary (`claude`, `codex`).
    Process,
}

/// Credential strategy the host declared for a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum AuthKind {
    /// OAuth / subscription session token.
    #[default]
    Subscription,
    /// Direct API key.
    Key,
    /// MCP connect token.
    Token,
    /// Local, no auth (Ollama).
    None,
}

/// One negotiated protocol of a connection: the wire it speaks and the base
/// URL the [`WireResolver`] appends the chat path to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtocolEndpoint {
    pub wire: WireProtocol,
    pub base_url: String,
}

/// A connection record the host found for the session: where its credential
/// lives and which protocol was negotiated for it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConnectionSpec {
    pub id: String,
    /// Credential slot the [`CredentialSource`] is asked for. Overrides the
    /// provider resolver's own default slot (R359-F2).
    pub credential_slot: Option<String>,
    /// Environment variable consulted when the slot is empty.
    pub credential_env: Option<String>,
    /// The negotiated protocol; `None` when the connection offers none the
    /// session's runner can speak.
    pub protocol: Option<ProtocolEndpoint>,
}

/// Everything the registry needs to resolve one session's engine. Wire-
/// neutral: the host projects its own agent/connection records into this.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProviderSpec {
    /// Provider label — selects the [`ProviderResolver`].
    pub provider: String,
    /// An inline key that beats every credential slot.
    pub inline_key: Option<String>,
    /// Declared credential strategy; `None` when the host has no agent record.
    pub auth: Option<AuthKind>,
    /// The connection id the agent declared, even when no record by that id
    /// was found. Resolvers branch on its presence.
    pub connection_id: Option<String>,
    /// The connection record found for `connection_id`, if any.
    pub connection: Option<ConnectionSpec>,
    pub reasoning: Option<ReasoningBudget>,
}

impl ProviderSpec {
    /// A spec naming only the provider — no agent record, no connection.
    pub fn provider(provider: impl Into<String>) -> Self {
        Self { provider: provider.into(), ..Self::default() }
    }

    fn inline_key(&self) -> Option<&String> {
        self.inline_key.as_ref().filter(|k| !k.is_empty())
    }

    fn connection_id(&self) -> &str {
        self.connection_id.as_deref().unwrap_or("")
    }
}

// ---------------------------------------------------------------------------
// Model selector
// ---------------------------------------------------------------------------

/// Free-vs-paid tier selector for a [`ModelQuery`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelTier {
    /// Only `:free` tier models (zero-price, rate-limited).
    Free,
    /// Only paid models.
    Paid,
    /// Any tier (no filter).
    #[default]
    Any,
}

/// Dynamic model selector resolved at session start against the OpenRouter
/// almanac. Used as the `Query` variant of [`ModelSpec`].
///
/// The resolver picks the highest-popularity-rank almanac entry satisfying
/// all non-`None` predicates. If no entry matches, the session errors before
/// `session_started` (no silent stale fallback). Quality-score predicates
/// (`min_quality_score`) are T1 fields and are silently ignored until
/// R260-F5 populates `AlmanacEntry::quality_score`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelQuery {
    /// Restrict to a specific provider slug prefix (e.g. `"qwen"` matches
    /// `qwen/qwen3-coder:free`). `None` = any provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Free/paid filter. Default `Any`.
    #[serde(default)]
    pub tier: ModelTier,
    /// All listed tags must appear in the entry's `capability_tags` slice.
    /// Vocabulary: `"code"`, `"vision"`, `"tool-use"`, `"long-context"`,
    /// `"reasoning"` — see [`CAPABILITY_TAGS`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capability_tags: Vec<String>,
    /// Minimum context window in tokens (`None` = no constraint).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_context_tokens: Option<u32>,
    /// Maximum prompt price per token in USD (`None` = no constraint).
    /// For `:free` models this is 0 — use `tier: Free` rather than setting
    /// this to `Some(0.0)` explicitly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_price_per_token_usd: Option<f64>,
    /// T1 field: minimum quality score (populated by R260-F5). Silently
    /// ignored until `AlmanacEntry::quality_score` is non-`None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_quality_score: Option<f64>,
}

/// A model specifier on an agent record: either a concrete model id
/// string or a dynamic query resolved against the almanac at session start.
///
/// ## Serde (backward-compat)
///
/// `#[serde(untagged)]` means the JSON wire shape is:
/// - `Literal` → plain JSON string, e.g. `"qwen/qwen3-coder:free"`.
///   Old on-disk records already have this shape — they round-trip as `Literal`
///   with zero migration.
/// - `Query` → JSON object with `ModelQuery` fields, e.g.
///   `{ "tier": "free", "capabilityTags": ["code"] }`.
///
/// ## From impls
///
/// `"foo".into()` and `"foo".to_string().into()` both yield `Literal`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelSpec {
    /// Concrete model id, passed to the runner as-is.
    Literal(String),
    /// Dynamic selector resolved by the almanac resolver at session start.
    Query(ModelQuery),
}

impl ModelSpec {
    /// Returns the model id when this is a `Literal`, or `None` for a `Query`.
    pub fn as_literal_str(&self) -> Option<&str> {
        match self {
            Self::Literal(s) => Some(s.as_str()),
            Self::Query(_) => None,
        }
    }

    /// Returns the model id for `Literal`, or `""` for `Query`.
    ///
    /// Use in legacy call sites that assume a model string is always
    /// present. `Query` variants must be resolved before reaching such
    /// sites; `""` acts as a safe no-op (the resolver will error earlier).
    pub fn literal_str_or_empty(&self) -> &str {
        self.as_literal_str().unwrap_or("")
    }
}

impl Default for ModelSpec {
    fn default() -> Self {
        ModelSpec::Literal(String::new())
    }
}

impl From<String> for ModelSpec {
    fn from(s: String) -> Self {
        ModelSpec::Literal(s)
    }
}

impl<'a> From<&'a str> for ModelSpec {
    fn from(s: &'a str) -> Self {
        ModelSpec::Literal(s.to_string())
    }
}

// ---------------------------------------------------------------------------
// Credential source
// ---------------------------------------------------------------------------

/// A credential store failed to read (not a miss — a miss is `Ok(None)`).
#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub struct CredentialError(pub String);

/// Where secrets come from. The host implements it over its own vault; the
/// registry never reads the environment or a keychain itself.
pub trait CredentialSource: Send + Sync {
    /// Look up `slot`, falling back to `env_var` on a miss. A read failure is
    /// an `Err`, so it surfaces instead of degrading to "no credential".
    fn resolve(&self, slot: &str, env_var: Option<&str>) -> Result<Option<String>, CredentialError>;
}

// ---------------------------------------------------------------------------
// Usage types
// ---------------------------------------------------------------------------

/// Parameters for a usage/billing query.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UsageQuery {
    /// Unix seconds, start of the reporting window.
    pub period_start: u64,
    /// Unix seconds, end of the reporting window.
    pub period_end: u64,
    /// Optional API key override. When set, the resolver uses this key instead
    /// of its default vault slot — enables per-subclass and per-character
    /// metering by passing the key stored in `AgentSubclass.api_key` or
    /// `AgentRecord.api_key`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

impl UsageQuery {
    /// Build a query covering the last `days` days ending now.
    pub fn last_days(days: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        UsageQuery {
            period_start: now.saturating_sub(days * 86400),
            period_end: now,
            api_key: None,
        }
    }
}

/// Aggregated usage/quota data returned by [`ProviderResolver::fetch_usage`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UsageSnapshot {
    pub provider: String,
    /// ISO-8601 period start (absent for quota-only providers like Ollama Cloud).
    pub period_start: Option<String>,
    /// ISO-8601 period end.
    pub period_end: Option<String>,
    /// Aggregated input tokens for the period.
    pub input_tokens: u64,
    /// Aggregated output tokens for the period.
    pub output_tokens: u64,
    /// Cache read tokens (Anthropic only; 0 elsewhere).
    pub cache_read_tokens: u64,
    /// Cache write tokens (Anthropic only; 0 elsewhere).
    pub cache_write_tokens: u64,
    /// Number of API requests in the period.
    pub requests: u64,
    /// Provider-specific quota windows (Ollama Cloud).
    pub quota: Option<QuotaInfo>,
    /// Raw provider response, preserved for Phase-2 agentic interrogation.
    pub raw: serde_json::Value,
}

/// Quota window data for providers that report usage as percentages (Ollama Cloud).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuotaInfo {
    pub session_used_pct: Option<f64>,
    pub session_resets_at: Option<String>,
    pub weekly_used_pct: Option<f64>,
    pub weekly_resets_at: Option<String>,
    pub plan: Option<String>,
}

#[derive(Debug, Error)]
pub enum UsageError {
    /// This provider/config does not support usage reporting.
    #[error("usage reporting is not supported for this provider")]
    Unsupported,
    /// An admin key is required but absent from the vault.
    #[error("admin key required for usage reporting — {hint}")]
    MissingAdminKey { hint: String },
    /// HTTP transport failure.
    #[error("usage fetch transport error: {reason}")]
    Transport { reason: String },
    /// Provider returned an unexpected non-2xx status.
    #[error("usage endpoint returned unexpected status {status}: {body}")]
    Unexpected { status: u16, body: String },
}

pub type UsageResult = Result<UsageSnapshot, UsageError>;

/// Convert a Unix timestamp (seconds since epoch) to an ISO-8601 UTC string.
///
/// Uses Howard Hinnant's civil_from_days algorithm.
pub(crate) fn secs_to_iso8601(secs: u64) -> String {
    let days = secs / 86400;
    let rem = secs % 86400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);

    let z = days as i64 + 719468;
    let era = z.div_euclid(146097);
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };

    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum EngineResolutionError {
    #[error("no {provider} credential in slot '{slot}' — {hint}")]
    MissingCredential {
        provider: String,
        slot: String,
        hint: String,
    },
    #[error("invalid subclass traits for {provider}: {reason}")]
    InvalidTraits { provider: String, reason: String },
    #[error("credential vault read failed: {reason}")]
    VaultRead { reason: String },
}

/// Error returned by a preflight probe.
#[derive(Debug, Error)]
pub enum ProbeError {
    #[error("connection failed: {reason}")]
    Transport { reason: String },
    /// Credentials were rejected (HTTP 401/403).
    #[error("credentials rejected by provider (HTTP {status}): {body}")]
    AuthRejected { status: u16, body: String },
    /// Any other non-success HTTP status.
    #[error("provider returned unexpected status {status}: {body}")]
    Unexpected { status: u16, body: String },
}

// ---------------------------------------------------------------------------
// Resolved endpoint
// ---------------------------------------------------------------------------

/// Credential material resolved for a session.
#[derive(Debug, Clone)]
pub enum AuthMaterial {
    /// Sent as `Authorization: Bearer <token>` (or `x-api-key` for Anthropic keys).
    Bearer(String),
    /// No credential — local endpoints and subprocess runners.
    None,
}

/// Whether the endpoint runs on the user's machine or a hosted service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionKind {
    Local,
    Cloud,
}

/// The output of [`ProviderRegistry::resolve`]: where to send turns, how to
/// authenticate, and which wire to speak. Wire-neutral — a runner cell
/// converts it into its own config (e.g. `yah-runner-ho` turns it into an
/// `OpenAiCompatConfig`).
#[derive(Debug, Clone)]
pub struct ResolvedEndpoint {
    /// Full chat endpoint URL (or `subprocess://<binary>` for CLI runners).
    pub endpoint: String,
    pub auth: AuthMaterial,
    /// Model used when the session does not name one.
    pub default_model: String,
    pub kind: ConnectionKind,
    /// The provider label the runner reports as its engine. Equals the
    /// resolver key the registry dispatched on.
    pub provider_label: String,
    /// Wire protocol the endpoint speaks. Overridden by the negotiated
    /// connection protocol when there is one.
    pub wire_protocol: WireProtocol,
    pub reasoning: Option<ReasoningBudget>,
}

// ---------------------------------------------------------------------------
// Resolver trait
// ---------------------------------------------------------------------------

/// One provider's credential + metadata resolution, preflight probe and
/// usage reporting. Selected by [`ProviderSpec::provider`].
///
/// `resolve` is synchronous: it reads only the [`CredentialSource`] (a local
/// vault / env read), never the network. `probe` and `fetch_usage` do I/O
/// through the [`HttpClient`] the registry was built with.
#[async_trait]
pub trait ProviderResolver: Send + Sync {
    fn resolve(
        &self,
        spec: &ProviderSpec,
        credentials: &dyn CredentialSource,
    ) -> Result<ResolvedEndpoint, EngineResolutionError>;

    async fn probe(&self, cfg: &ResolvedEndpoint, http: &dyn HttpClient) -> Result<(), ProbeError> {
        let _ = (cfg, http);
        Ok(())
    }

    /// Fetch usage/billing for the query window. The default reports the
    /// provider has no usage API.
    async fn fetch_usage(
        &self,
        query: &UsageQuery,
        http: &dyn HttpClient,
        credentials: &dyn CredentialSource,
    ) -> UsageResult {
        let _ = (query, http, credentials);
        Err(UsageError::Unsupported)
    }
}

// ---------------------------------------------------------------------------
// Wire-protocol resolver trait + implementations
// ---------------------------------------------------------------------------

/// Builds the final chat endpoint for one [`WireProtocol`] from a
/// connection's negotiated base URL (R310-T2).
pub trait WireResolver: Send + Sync {
    fn build_endpoint(&self, protocol: &ProtocolEndpoint) -> String;
}

struct AnthropicWireResolver;
impl WireResolver for AnthropicWireResolver {
    fn build_endpoint(&self, protocol: &ProtocolEndpoint) -> String {
        format!("{}/v1/messages", protocol.base_url.trim_end_matches('/'))
    }
}

struct OpenAIWireResolver;
impl WireResolver for OpenAIWireResolver {
    fn build_endpoint(&self, protocol: &ProtocolEndpoint) -> String {
        format!("{}/chat/completions", protocol.base_url.trim_end_matches('/'))
    }
}

// ---------------------------------------------------------------------------
// DeepSeek inline resolver (credentials + usage; endpoint built by
// WireResolver at registry level from the bundled connection JSON)
// ---------------------------------------------------------------------------

const DEEPSEEK_BALANCE_URL: &str = "https://api.deepseek.com/user/balance";

struct DeepSeekResolver;

#[async_trait]
impl ProviderResolver for DeepSeekResolver {
    fn resolve(
        &self,
        spec: &ProviderSpec,
        credentials: &dyn CredentialSource,
    ) -> Result<ResolvedEndpoint, EngineResolutionError> {
        let key = match spec.inline_key() {
            Some(k) => Ok(Some(k.clone())),
            None => credentials
                .resolve(DEEPSEEK_SLOT, Some(DEEPSEEK_ENV))
                .map_err(|e| EngineResolutionError::VaultRead { reason: e.to_string() }),
        }?;
        let key = key.ok_or_else(|| EngineResolutionError::MissingCredential {
            provider: "deepseek".into(),
            slot: DEEPSEEK_SLOT.into(),
            hint: "store a DeepSeek API key in the 'deepseek-api-key' slot \
                   (or export DEEPSEEK_API_KEY). Get one from platform.deepseek.com/api-keys."
                .into(),
        })?;
        Ok(ResolvedEndpoint {
            // Default for direct-resolver callers; ProviderRegistry's WireResolver
            // overrides this from the ProtocolEndpoint when a connection is set.
            endpoint: "https://api.deepseek.com/anthropic/v1/messages".into(),
            auth: AuthMaterial::Bearer(key),
            default_model: "deepseek-chat".into(),
            kind: ConnectionKind::Cloud,
            provider_label: "deepseek".into(),
            wire_protocol: WireProtocol::AnthropicWire,
            reasoning: spec.reasoning,
        })
    }

    async fn probe(&self, cfg: &ResolvedEndpoint, http: &dyn HttpClient) -> Result<(), ProbeError> {
        // Probe via the balance endpoint — works for both wire protocols without
        // needing a /models path on the negotiated base URL.
        let bearer = match &cfg.auth {
            AuthMaterial::Bearer(k) => k.clone(),
            AuthMaterial::None => {
                return Err(ProbeError::AuthRejected {
                    status: 401,
                    body: "no DeepSeek API key configured".into(),
                })
            }
        };
        let resp = http
            .send(HttpRequest::get(DEEPSEEK_BALANCE_URL).bearer_auth(&bearer))
            .await
            .map_err(|e| ProbeError::Transport { reason: e.to_string() })?;
        let status = resp.status;
        match status {
            200..=299 => Ok(()),
            401 | 403 => {
                let body = resp.text().await.unwrap_or_default();
                Err(ProbeError::AuthRejected { status, body })
            }
            other => {
                let body = resp.text().await.unwrap_or_default();
                Err(ProbeError::Unexpected { status: other, body })
            }
        }
    }

    async fn fetch_usage(
        &self,
        query: &UsageQuery,
        http: &dyn HttpClient,
        credentials: &dyn CredentialSource,
    ) -> UsageResult {
        let key = if let Some(k) = query.api_key.as_deref().filter(|k| !k.is_empty()) {
            k.to_owned()
        } else {
            let vault_key = credentials
                .resolve(DEEPSEEK_SLOT, Some(DEEPSEEK_ENV))
                .map_err(|e| UsageError::Transport { reason: e.to_string() })?;
            match vault_key {
                Some(k) if !k.is_empty() => k,
                _ => {
                    return Err(UsageError::MissingAdminKey {
                        hint: "store a DeepSeek API key in the 'deepseek-api-key' slot \
                               (or export DEEPSEEK_API_KEY). Get one from \
                               platform.deepseek.com/api-keys."
                            .into(),
                    })
                }
            }
        };

        let resp = http
            .send(HttpRequest::get(DEEPSEEK_BALANCE_URL).bearer_auth(&key))
            .await
            .map_err(|e| UsageError::Transport { reason: e.to_string() })?;

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
        Ok(UsageSnapshot {
            provider: "deepseek".into(),
            period_start: None,
            period_end: None,
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            requests: 0,
            quota: None,
            raw,
        })
    }
}

// ---------------------------------------------------------------------------
// Kimi inline resolver (Anthropic-wire only; endpoint built by WireResolver
// at registry level from the bundled connection JSON)
// ---------------------------------------------------------------------------

struct KimiResolver;

#[async_trait]
impl ProviderResolver for KimiResolver {
    fn resolve(
        &self,
        spec: &ProviderSpec,
        credentials: &dyn CredentialSource,
    ) -> Result<ResolvedEndpoint, EngineResolutionError> {
        let key = match spec.inline_key() {
            Some(k) => Ok(Some(k.clone())),
            None => credentials
                .resolve(KIMI_SLOT, Some(KIMI_ENV))
                .map_err(|e| EngineResolutionError::VaultRead { reason: e.to_string() }),
        }?;
        let key = key.ok_or_else(|| EngineResolutionError::MissingCredential {
            provider: "kimi".into(),
            slot: KIMI_SLOT.into(),
            hint: "store a Moonshot API key in the 'kimi-platform-api-key' slot \
                   (or export MOONSHOT_API_KEY). Get one from platform.moonshot.cn."
                .into(),
        })?;
        Ok(ResolvedEndpoint {
            // Default for direct-resolver callers; ProviderRegistry's WireResolver
            // overrides this from the ProtocolEndpoint when a connection is set.
            endpoint: "https://api.moonshot.ai/anthropic/v1/messages".into(),
            auth: AuthMaterial::Bearer(key),
            default_model: "kimi-k2.6".into(),
            kind: ConnectionKind::Cloud,
            provider_label: "kimi".into(),
            wire_protocol: WireProtocol::AnthropicWire,
            reasoning: spec.reasoning,
        })
    }
}

// ---------------------------------------------------------------------------
// Shared HTTP helpers (used by individual resolver probe()/fetch_usage() impls)
// ---------------------------------------------------------------------------

/// Derive the OpenAI-compat `/models` URL from a resolved chat endpoint.
pub fn models_url_from_endpoint(chat_endpoint: &str) -> String {
    // Ollama native `/api/chat` lives outside the OpenAI-compat `/v1/*`
    // prefix; the model catalogue still serves at `/v1/models` on the
    // same host, so synthesise the prefix when stripping `/api/chat`.
    if let Some(stripped) = chat_endpoint.strip_suffix("/api/chat") {
        format!("{stripped}/v1/models")
    } else if let Some(stripped) = chat_endpoint.strip_suffix("/chat/completions") {
        format!("{stripped}/models")
    } else if let Some(stripped) = chat_endpoint.strip_suffix('/') {
        format!("{stripped}/models")
    } else {
        format!("{chat_endpoint}/models")
    }
}

/// GET an OpenAI-compat `/models` URL; 2xx = reachable and authorised.
pub(crate) async fn probe_openai_compat(
    http: &dyn HttpClient,
    models_url: &str,
    bearer: Option<&str>,
) -> Result<(), ProbeError> {
    let mut req = HttpRequest::get(models_url);
    if let Some(k) = bearer {
        req = req.bearer_auth(k);
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

/// `url?k=v&...`, form-urlencoding each pair the way `reqwest`'s `.query()`
/// does (serde_urlencoded: unreserved bytes pass, space becomes `+`).
pub(crate) fn with_query(url: &str, pairs: &[(&str, String)]) -> String {
    fn encode(out: &mut String, s: &str) {
        for b in s.bytes() {
            match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'*' | b'-' | b'.' | b'_' => {
                    out.push(b as char)
                }
                b' ' => out.push('+'),
                _ => out.push_str(&format!("%{b:02X}")),
            }
        }
    }
    let mut out = String::from(url);
    for (i, (k, v)) in pairs.iter().enumerate() {
        out.push(if i == 0 { '?' } else { '&' });
        encode(&mut out, k);
        out.push('=');
        encode(&mut out, v);
    }
    out
}

// ---------------------------------------------------------------------------
// Probe cache
// ---------------------------------------------------------------------------

const PROBE_TTL: Duration = Duration::from_secs(300);

static PROBE_CACHE: LazyLock<Mutex<HashMap<(String, String), Instant>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Cache key: (endpoint, first 16 chars of the bearer). The prefix is enough
/// to tell credentials apart without holding a whole secret in the map.
fn probe_cache_key(cfg: &ResolvedEndpoint) -> (String, String) {
    let auth_fp: String = match &cfg.auth {
        AuthMaterial::Bearer(k) => k.chars().take(16).collect(),
        AuthMaterial::None => String::new(),
    };
    (cfg.endpoint.clone(), auth_fp)
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Provider-keyed resolvers plus wire-keyed endpoint builders, bound to the
/// host's [`CredentialSource`] and [`HttpClient`].
///
/// Resolution order (R359): the provider resolver supplies credentials and
/// metadata; a connection's own credential slot then overrides the auth; the
/// negotiated protocol's [`WireResolver`] finally overrides the endpoint.
pub struct ProviderRegistry {
    resolvers: Vec<(&'static str, Box<dyn ProviderResolver>)>,
    wire_resolvers: Vec<(WireProtocol, Box<dyn WireResolver>)>,
    credentials: Arc<dyn CredentialSource>,
    http: Arc<dyn HttpClient>,
}

impl ProviderRegistry {
    /// Every bundled provider resolver.
    pub fn production(credentials: Arc<dyn CredentialSource>, http: Arc<dyn HttpClient>) -> Self {
        Self {
            credentials,
            http,
            resolvers: vec![
                ("anthropic", Box::new(AnthropicResolver)),
                ("anthropic-gc", Box::new(AnthropicResolver)),
                ("openai", Box::new(OpenAiResolver)),
                ("openrouter", Box::new(OpenRouterResolver)),
                ("groq", Box::new(GroqResolver)),
                ("deepseek", Box::new(DeepSeekResolver)),
                ("ollama", Box::new(OllamaResolver)),
                // R365-B17: camp.rs constructs EngineRef.provider from connection_id()
                // directly, so "ollama-cloud" must be registered alongside "ollama".
                // OllamaResolver.resolve_from_connection handles both ids; WireResolver
                // then overrides the endpoint from the ProtocolEndpoint (openai_wire).
                ("ollama-cloud", Box::new(OllamaResolver)),
                ("kimi", Box::new(KimiResolver)),
                ("claude", Box::new(ClaudeCliResolver)),
                ("kimi-cli", Box::new(KimiCliResolver)),
                // CodexCli runner uses provider "codex-cli". The subprocess
                // manages its own auth via ~/.codex/auth.json — no
                // yah-managed credentials.
                ("codex-cli", Box::new(CodexCliResolver)),
                // CodexOauth runner uses provider "openai-oauth" — the
                // golden-cloak direct-HTTP path. The session path reads the
                // keychain token directly, so this resolver's endpoint is
                // only consumed by probe/model-list dispatch.
                ("openai-oauth", Box::new(ClaudeCliResolver)),
            ],
            wire_resolvers: vec![
                (WireProtocol::AnthropicWire, Box::new(AnthropicWireResolver)),
                (WireProtocol::OpenaiWire, Box::new(OpenAIWireResolver)),
            ],
        }
    }

    /// The HTTP substrate probes and usage queries go through.
    pub fn http(&self) -> &Arc<dyn HttpClient> {
        &self.http
    }

    /// Resolve `spec` to a concrete endpoint + credential. Missing
    /// credentials are an `Err`; there is no silent fallback.
    pub fn resolve(&self, spec: &ProviderSpec) -> Result<ResolvedEndpoint, EngineResolutionError> {
        // Step 1: Provider-keyed resolver supplies credentials + metadata.
        let provider = spec.provider.as_str();
        let mut cfg = match self.resolvers.iter().find(|(name, _)| *name == provider) {
            Some((_, resolver)) => resolver.resolve(spec, self.credentials.as_ref())?,
            None => return Err(EngineResolutionError::InvalidTraits {
                provider: provider.to_string(),
                reason: format!("no resolver registered for provider '{provider}'"),
            }),
        };

        // Step 2 (R359-F2): Connection-keyed credential override.
        //
        // If the connection declares a `credential_slot`, look up that slot
        // and replace `cfg.auth`. This decouples credential dispatch from the
        // provider name — any connection can declare its own slot without a
        // dedicated resolver.
        //
        // Guards:
        // - Subscription auth (provider == "anthropic-gc") keeps its own slot
        //   (anthropic-oauth) managed by AnthropicResolver.
        // - An explicit inline key always wins (already baked into
        //   `cfg.auth` by the resolver above; we skip the override).
        let is_subscription =
            spec.provider == "anthropic-gc" || spec.auth == Some(AuthKind::Subscription);
        let has_inline_key = spec.inline_key().is_some();
        if let Some(conn) = &spec.connection {
            if let Some(slot) = &conn.credential_slot {
                if !is_subscription && !has_inline_key {
                    match self.credentials.resolve(slot, conn.credential_env.as_deref()) {
                        Ok(Some(k)) if !k.is_empty() => {
                            cfg.auth = AuthMaterial::Bearer(k);
                        }
                        Ok(_) => {} // slot empty — post-step guard handles auth=None case below
                        Err(e) => {
                            return Err(EngineResolutionError::VaultRead {
                                reason: e.to_string(),
                            });
                        }
                    }
                }
            }
        }

        // Post-step-2 guard: provider resolvers (e.g. OpenAiResolver) may
        // return AuthMaterial::None when a connection id is set, deferring
        // credential lookup to step 2. If the connection slot was also empty,
        // report MissingCredential with the connection's own slot name — not
        // the provider resolver's hardcoded default (e.g. "openai-api-key" for
        // an ollama-cloud connection whose slot is "ollama").
        if matches!(cfg.auth, AuthMaterial::None)
            && !is_subscription
            && !has_inline_key
            && !cfg.endpoint.starts_with("subprocess://")
        {
            if let Some(conn) = &spec.connection {
                if let Some(slot) = &conn.credential_slot {
                    let env = conn.credential_env.as_deref().unwrap_or("");
                    let hint = if env.is_empty() {
                        format!(
                            "store a {} API key in the '{}' slot via Settings → API Keys.",
                            conn.id, slot
                        )
                    } else {
                        format!(
                            "store a {} API key in the '{}' slot (or export {}). \
                             Open Settings → API Keys to configure it.",
                            conn.id, slot, env
                        )
                    };
                    return Err(EngineResolutionError::MissingCredential {
                        provider: conn.id.clone(),
                        slot: slot.clone(),
                        hint,
                    });
                }
            }
        }

        // Step 3: Wire resolver overrides endpoint + sets wire_protocol.
        // Subprocess endpoints are exempt — the subprocess manages its own connection.
        if let Some(protocol) = spec.connection.as_ref().and_then(|c| c.protocol.as_ref()) {
            if !cfg.endpoint.starts_with("subprocess://") {
                if let Some((_, wire_resolver)) =
                    self.wire_resolvers.iter().find(|(w, _)| *w == protocol.wire)
                {
                    cfg.endpoint = wire_resolver.build_endpoint(protocol);
                }
            }
            cfg.wire_protocol = protocol.wire;
        }

        Ok(cfg)
    }

    /// Preflight-probe a resolved endpoint. A success is cached for
    /// [`PROBE_TTL`] per (endpoint, credential prefix), so repeated session
    /// starts don't re-hit the provider.
    pub async fn probe(&self, cfg: &ResolvedEndpoint) -> Result<(), ProbeError> {
        let cache_key = probe_cache_key(cfg);
        {
            let cache = PROBE_CACHE.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(&cached_at) = cache.get(&cache_key) {
                if cached_at.elapsed() < PROBE_TTL {
                    tracing::debug!(
                        endpoint = %cfg.endpoint,
                        "preflight probe: cache hit, skipping I/O"
                    );
                    return Ok(());
                }
            }
        }

        let provider = cfg.provider_label.as_str();
        if let Some((_, resolver)) = self.resolvers.iter().find(|(name, _)| *name == provider) {
            resolver.probe(cfg, self.http.as_ref()).await?;
        }
        // Unknown provider → skip probe silently (same lenient behaviour as resolve returning InvalidTraits)

        {
            let mut cache = PROBE_CACHE.lock().unwrap_or_else(|e| e.into_inner());
            cache.insert(cache_key, Instant::now());
        }
        tracing::debug!(endpoint = %cfg.endpoint, "preflight probe: ok");
        Ok(())
    }

    /// Usage/billing for `provider` over the query window.
    pub async fn fetch_usage(&self, provider: &str, query: &UsageQuery) -> UsageResult {
        match self.resolvers.iter().find(|(name, _)| *name == provider) {
            Some((_, resolver)) => {
                resolver
                    .fetch_usage(query, self.http.as_ref(), self.credentials.as_ref())
                    .await
            }
            None => Err(UsageError::Unsupported),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared key-slot constants (referenced by individual resolvers)
// ---------------------------------------------------------------------------

pub(crate) const ANTHROPIC_SLOT: &str = "anthropic-api-key";
pub(crate) const ANTHROPIC_ENV: &str = "ANTHROPIC_API_KEY";
/// OAuth bearer from `claude setup-token` (anthropic-gc provider).
pub(crate) const ANTHROPIC_OAUTH_SLOT: &str = "anthropic-oauth";
pub(crate) const ANTHROPIC_OAUTH_ENV: &str = "ANTHROPIC_OAUTH_TOKEN";
/// Admin API key for the usage report endpoint (org members only).
pub(crate) const ANTHROPIC_ADMIN_SLOT: &str = "anthropic-admin-key";
pub(crate) const ANTHROPIC_ADMIN_ENV: &str = "ANTHROPIC_ADMIN_KEY";

pub(crate) const OPENAI_SLOT: &str = "openai-api-key";
pub(crate) const OPENAI_ENV: &str = "OPENAI_API_KEY";
/// Admin key for the organization usage endpoint.
pub(crate) const OPENAI_ADMIN_SLOT: &str = "openai-admin-key";
pub(crate) const OPENAI_ADMIN_ENV: &str = "OPENAI_ADMIN_KEY";

pub(crate) const OLLAMA_SLOT: &str = "ollama";
pub(crate) const OLLAMA_ENV: &str = "OLLAMA_API_KEY";

pub(crate) const OPENROUTER_SLOT: &str = "openrouter-api-key";
pub(crate) const OPENROUTER_ENV: &str = "OPENROUTER_API_KEY";

pub(crate) const GROQ_SLOT: &str = "groq-api-key";
pub(crate) const GROQ_ENV: &str = "GROQ_API_KEY";

pub(crate) const DEEPSEEK_SLOT: &str = "deepseek-api-key";
pub(crate) const DEEPSEEK_ENV: &str = "DEEPSEEK_API_KEY";

pub(crate) const KIMI_SLOT: &str = "kimi-platform-api-key";
pub(crate) const KIMI_ENV: &str = "MOONSHOT_API_KEY";

// ---------------------------------------------------------------------------
// Kimi CLI resolver (subprocess — no yah-managed credentials)
// ---------------------------------------------------------------------------

/// Kimi Code CLI subprocess. The binary manages its own OAuth login, so the
/// resolver hands back a `subprocess://` endpoint and no credential.
struct KimiCliResolver;

#[async_trait]
impl ProviderResolver for KimiCliResolver {
    fn resolve(
        &self,
        spec: &ProviderSpec,
        _credentials: &dyn CredentialSource,
    ) -> Result<ResolvedEndpoint, EngineResolutionError> {
        Ok(ResolvedEndpoint {
            endpoint: "subprocess://kimi".into(),
            auth: AuthMaterial::None,
            // `kimi-for-coding` is the subscription model alias the CLI
            // resolves at spawn time. For API-key-based usage, override
            // via subclass model to `kimi-k2.7-code` (etc.).
            default_model: "kimi-for-coding".into(),
            kind: ConnectionKind::Cloud,
            provider_label: "kimi-cli".into(),
            wire_protocol: WireProtocol::Process,
            reasoning: spec.reasoning,
        })
    }
}

/// Codex CLI subprocess. Auth lives in `~/.codex/auth.json`, managed by the
/// binary itself.
struct CodexCliResolver;

#[async_trait]
impl ProviderResolver for CodexCliResolver {
    fn resolve(
        &self,
        spec: &ProviderSpec,
        _credentials: &dyn CredentialSource,
    ) -> Result<ResolvedEndpoint, EngineResolutionError> {
        Ok(ResolvedEndpoint {
            endpoint: "subprocess://codex".into(),
            auth: AuthMaterial::None,
            // `gpt-5.5` is the default codex model; the subprocess resolves
            // the concrete version at spawn time.
            default_model: "gpt-5.5".into(),
            kind: ConnectionKind::Cloud,
            provider_label: "codex-cli".into(),
            wire_protocol: WireProtocol::Process,
            reasoning: spec.reasoning,
        })
    }
}

/// Test doubles shared by every resolver's tests: an in-memory credential
/// store and an HTTP client that answers every request with one canned
/// response, so no test touches a vault, the environment or the network.
#[cfg(test)]
pub(crate) mod testing {
    use super::{CredentialError, CredentialSource};
    use crate::http::{HttpClient, HttpError, HttpRequest, HttpResponse};
    use async_trait::async_trait;
    use bytes::Bytes;
    use std::collections::HashMap;
    use std::pin::Pin;
    use std::sync::Mutex;
    use std::task::{Context, Poll};

    #[derive(Default)]
    pub(crate) struct MapCredentials(pub HashMap<String, String>);

    impl MapCredentials {
        pub(crate) fn empty() -> Self {
            Self::default()
        }

        pub(crate) fn with(slot: &str, secret: &str) -> Self {
            Self(HashMap::from([(slot.to_string(), secret.to_string())]))
        }
    }

    impl CredentialSource for MapCredentials {
        fn resolve(&self, slot: &str, _env_var: Option<&str>) -> Result<Option<String>, CredentialError> {
            Ok(self.0.get(slot).cloned())
        }
    }

    struct Once(Option<Bytes>);

    impl futures_core::Stream for Once {
        type Item = Result<Bytes, HttpError>;
        fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.0.take().map(Ok))
        }
    }

    /// Answers every request with `status` + `body` and records the URLs.
    pub(crate) struct StubHttp {
        pub status: u16,
        pub body: String,
        pub urls: Mutex<Vec<String>>,
    }

    impl StubHttp {
        pub(crate) fn new(status: u16, body: &str) -> Self {
            Self { status, body: body.into(), urls: Mutex::new(Vec::new()) }
        }
    }

    #[async_trait]
    impl HttpClient for StubHttp {
        async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
            self.urls.lock().unwrap().push(request.url);
            Ok(HttpResponse {
                status: self.status,
                headers: Vec::new(),
                body: Box::pin(Once(Some(Bytes::from(self.body.clone())))),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{MapCredentials, StubHttp};
    use super::*;

    fn registry(credentials: MapCredentials) -> ProviderRegistry {
        ProviderRegistry::production(Arc::new(credentials), Arc::new(StubHttp::new(200, "{}")))
    }

    fn connection(id: &str, slot: &str, wire: WireProtocol, base_url: &str) -> ConnectionSpec {
        ConnectionSpec {
            id: id.into(),
            credential_slot: Some(slot.into()),
            credential_env: None,
            protocol: Some(ProtocolEndpoint { wire, base_url: base_url.into() }),
        }
    }

    #[test]
    fn anthropic_wire_resolver_appends_v1_messages() {
        let resolver = AnthropicWireResolver;
        let entry = ProtocolEndpoint {
            wire: WireProtocol::AnthropicWire,
            base_url: "https://api.anthropic.com".into(),
        };
        assert_eq!(resolver.build_endpoint(&entry), "https://api.anthropic.com/v1/messages");
    }

    #[test]
    fn anthropic_wire_resolver_strips_trailing_slash() {
        let resolver = AnthropicWireResolver;
        let entry = ProtocolEndpoint {
            wire: WireProtocol::AnthropicWire,
            base_url: "https://api.deepseek.com/anthropic/".into(),
        };
        assert_eq!(
            resolver.build_endpoint(&entry),
            "https://api.deepseek.com/anthropic/v1/messages"
        );
    }

    #[test]
    fn openai_wire_resolver_appends_chat_completions() {
        let resolver = OpenAIWireResolver;
        let entry = ProtocolEndpoint {
            wire: WireProtocol::OpenaiWire,
            base_url: "https://api.openai.com/v1".into(),
        };
        assert_eq!(
            resolver.build_endpoint(&entry),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn openai_wire_resolver_groq_base_url() {
        let resolver = OpenAIWireResolver;
        let entry = ProtocolEndpoint {
            wire: WireProtocol::OpenaiWire,
            base_url: "https://api.groq.com/openai/v1".into(),
        };
        assert_eq!(
            resolver.build_endpoint(&entry),
            "https://api.groq.com/openai/v1/chat/completions"
        );
    }

    #[test]
    fn openai_wire_resolver_deepseek_base_url() {
        let resolver = OpenAIWireResolver;
        let entry = ProtocolEndpoint {
            wire: WireProtocol::OpenaiWire,
            base_url: "https://api.deepseek.com/v1".into(),
        };
        assert_eq!(
            resolver.build_endpoint(&entry),
            "https://api.deepseek.com/v1/chat/completions"
        );
    }

    /// R359-F2: a connection's own slot fills the auth the provider resolver
    /// deferred, and its negotiated protocol builds the endpoint.
    #[test]
    fn connection_slot_and_protocol_override_the_provider_defaults() {
        let spec = ProviderSpec {
            auth: Some(AuthKind::Key),
            connection_id: Some("ollama-cloud".into()),
            connection: Some(connection(
                "ollama-cloud",
                "ollama",
                WireProtocol::OpenaiWire,
                "https://ollama.com/v1",
            )),
            ..ProviderSpec::provider("openai")
        };
        let cfg = registry(MapCredentials::with("ollama", "k-ollama")).resolve(&spec).unwrap();
        assert!(matches!(&cfg.auth, AuthMaterial::Bearer(k) if k == "k-ollama"));
        assert_eq!(cfg.endpoint, "https://ollama.com/v1/chat/completions");
        assert_eq!(cfg.wire_protocol, WireProtocol::OpenaiWire);
    }

    /// R365-B1: with the connection slot empty too, the error names the
    /// connection's slot, not the provider resolver's default.
    #[test]
    fn empty_connection_slot_reports_the_connection_slot() {
        let spec = ProviderSpec {
            auth: Some(AuthKind::Key),
            connection_id: Some("ollama-cloud".into()),
            connection: Some(connection(
                "ollama-cloud",
                "ollama",
                WireProtocol::OpenaiWire,
                "https://ollama.com/v1",
            )),
            ..ProviderSpec::provider("openai")
        };
        match registry(MapCredentials::empty()).resolve(&spec) {
            Err(EngineResolutionError::MissingCredential { provider, slot, .. }) => {
                assert_eq!(provider, "ollama-cloud");
                assert_eq!(slot, "ollama");
            }
            other => panic!("expected MissingCredential(slot=ollama), got: {other:?}"),
        }
    }

    /// An inline key beats the connection slot, and subscription auth never
    /// reads the connection slot at all.
    #[test]
    fn inline_key_and_subscription_skip_the_connection_slot() {
        let conn = connection("anthropic", "anthropic-api-key", WireProtocol::AnthropicWire, "https://api.anthropic.com");
        let creds = || MapCredentials(HashMap::from([
            ("anthropic-api-key".to_string(), "k-slot".to_string()),
            ("anthropic-oauth".to_string(), "k-oauth".to_string()),
        ]));
        let inline = ProviderSpec {
            inline_key: Some("k-inline".into()),
            auth: Some(AuthKind::Key),
            connection: Some(conn.clone()),
            ..ProviderSpec::provider("anthropic")
        };
        let cfg = registry(creds()).resolve(&inline).unwrap();
        assert!(matches!(&cfg.auth, AuthMaterial::Bearer(k) if k == "k-inline"));

        let gc = ProviderSpec {
            auth: Some(AuthKind::Subscription),
            connection: Some(conn),
            ..ProviderSpec::provider("anthropic-gc")
        };
        let cfg = registry(creds()).resolve(&gc).unwrap();
        assert!(matches!(&cfg.auth, AuthMaterial::Bearer(k) if k == "k-oauth"));
        assert_eq!(cfg.endpoint, "https://api.anthropic.com/v1/messages");
    }

    /// Subprocess endpoints keep their `subprocess://` URL even when a
    /// protocol was negotiated; only the wire is taken from it.
    #[test]
    fn subprocess_endpoint_is_exempt_from_the_wire_override() {
        let spec = ProviderSpec {
            connection: Some(connection("x", "x-slot", WireProtocol::OpenaiWire, "https://x/v1")),
            ..ProviderSpec::provider("claude")
        };
        let cfg = registry(MapCredentials::empty()).resolve(&spec).unwrap();
        assert_eq!(cfg.endpoint, "subprocess://claude");
        assert_eq!(cfg.wire_protocol, WireProtocol::OpenaiWire);
    }

    #[test]
    fn unknown_provider_is_invalid_traits() {
        let err = registry(MapCredentials::empty())
            .resolve(&ProviderSpec::provider("nope"))
            .unwrap_err();
        assert!(matches!(err, EngineResolutionError::InvalidTraits { .. }), "{err:?}");
    }

    #[test]
    fn with_query_form_encodes_like_reqwest() {
        assert_eq!(
            with_query("https://h/p", &[("starting_at", "2026-04-09T00:00:00Z".into()), ("w", "1 d".into())]),
            "https://h/p?starting_at=2026-04-09T00%3A00%3A00Z&w=1+d"
        );
    }
}
