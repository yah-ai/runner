//! Claude CLI resolver: subprocess preset, no yah-managed credentials.
//!
//! The `claude` preset wraps the `claude` CLI as a subprocess and delegates
//! auth entirely to Claude Code's own credential store. No vault slot is
//! read here; the "endpoint" is a placeholder (the subprocess runner ignores
//! it). This resolver exists so `ProviderRegistry` can enumerate all four
//! presets consistently — e.g. for `agent_list_models` dispatch — without
//! special-casing the subprocess path.
//!
//! @yah:ticket(R215-T10, "Subprocess subclass probe driver (claude-cli + codex-cli)")
//! @yah:at(2026-05-21T22:16:28Z)
//! @yah:status(review)
//! @yah:parent(R215)
//! @yah:depends_on(R215-F5)
//! @yah:next("Probe driver dispatch in build_driver (app/yah/desktop/src/probe.rs): detect subprocess:// endpoint → CliSubprocessProbeDriver, HTTP endpoint → existing OpenAiCompatProbeDriver. Both produce the same ProbeCompletion shape so engine/scorers are identical and reports are comparable.")
//! @yah:next("Create CliSubprocessProbeDriver wrapping runner::ClaudeCliRunner — start/send/stop per probe, same session lifecycle as OpenAiCompatProbeDriver.")
//! @yah:next("Cache-field probe for subprocess: the `claude` CLI returns cache metrics in a different schema (cached_bytes / cache_creation vs OpenAI-compat cache_read_tokens). Translate to ProbeCompletion.usage fields or metadata.")
//! @yah:next("health/list-models for subprocess: probe `claude models list` CLI command instead of HTTP GET /v1/models. Return same metadata shape (model_found, models_count) so the ListModelsScorer works identically.")
//! @yah:next("Compare: after T10, a user can probe bundle-openrouter-wizard-primary (HTTP) and bundle-anthropic-wizard-primary (subprocess→claude CLI) side by side and compare TTFT ratios, cache-field presence, and capability scores from the same UI.")
//! @yah:next("Verify: cargo test -p probe; cargo check --workspace; probe sweep against a claude-resolved subclass produces real verdicts (not all-error); TTFT cold/warm and cache-fields differ between subprocess and HTTP subclasses as expected.")
//! @yah:handoff("Subprocess probe driver complete. CliSubprocessProbeDriver at app/yah/desktop/src/subprocess_probe_driver.rs (440 LOC): spawns claude/codex CLI as subprocess per probe, reads stream-json output, measures TTFT as gap between message_start and first text_delta. Probe driver dispatch in build_driver (app/yah/desktop/src/probe.rs:114-133) detects subprocess:// endpoint and routes to CliSubprocessProbeDriver; HTTP endpoints use OpenAiCompatProbeDriver + HealthProbeDriver. Both produce identical ProbeCompletion shape so engine/scorers work unmodified. Cache-field translation (usage_from_stream_json) converts claude CLI schema to TurnUsage. health/list-models implemented via claude models list --json subprocess. TTFT cold/warm pair via run_ttft. 94/94 probe tests + 17/17 desktop probe tests green. cargo check --workspace clean.")
//!
//! @yah:ticket(R575-F9, "Interactive claude-cli re-login (net-new; needs separate UX design)")
//! @yah:status(review)
//! @yah:at(2026-07-15T22:14:55Z)
//! @yah:assignee(agent:bundle-anthropic-ashguard)
//! @yah:phase(P3)
//! @yah:parent(R575)
//! @arch:see(.yah/docs/working/W240-session-recovery-taxonomy.md)
//! @yah:depends_on(R575-T6)
//! @yah:depends_on(R575-F8)
//! @yah:handoff("Shipped FE-only (auth_expired BlockReason already routes through BlockedModal via T6). Decision (open-Q3, user-confirmed): terminal-pane spawn of official 'claude auth login', NOT in-webview OAuth — keeps yah a subprocess wrapper clear of the Pro/Max OAuth 3rd-party TOS ban. Changes: (1) types.ts LocalRecovery gains relogin variant. (2) BlockedModal.tsx: new onRelogin prop; auth_expired renders an ACTIVE 'Re-login to continue' button (was a disabled placeholder) ordered before Retry; auth_expired added to canRetry so operator resumes in one click after login. (3) ChatSurface.tsx wires onRelogin -> navStore.setTerminalIntent(kind:cmd, cmd:'claude auth login') + focus Run>Terminal (same mechanism as RunInTerminalButton). (4) BlockedModal.test.tsx flipped to assert active relogin fires onRelogin + auth_expired offers Retry. W240 open-Q3 + Phase-3 relogin bullet resolved.")
//! @yah:verify("cd packages/yah/ui && bun run typecheck && bun run build — clean")
//! @yah:verify("bun test src/components/agent/party/BlockedModal.test.tsx — 7/7 pass")
//! @yah:verify("Manual (needs desktop rebuild): claude_cli session blocked with auth_expired shows 'Re-login to continue' -> click opens Run>Terminal running 'claude auth login'; after login, Retry resumes the turn.")
//! @yah:gotcha("F9 ships the recovery SURFACE only. The literal claude subprocess path still has no detector that PRODUCES auth_expired from a real 401 frame (health.rs:21 gotcha — no captured claude-cli auth-failure fixture; runner_kind->AuthExpired split is wired+ready but nothing feeds it yet). Subprocess 401->auth_expired detection is a separate follow-up under R575, not this ticket.")
//! @yah:next("FOLLOW-UP (separate ticket, not F9): wire subprocess 401/403 -> auth_expired detection on the literal claude path. The runner_kind->AuthExpired split is ready (block_detector.rs:48) but no detector feeds it a real frame yet (health.rs:21). Needs a captured claude-cli auth-failure result-frame fixture before adding phrase-matching.")

use super::{
    AuthMaterial, ConnectionKind, CredentialSource, EngineResolutionError, ProviderResolver,
    ProviderSpec, ResolvedEndpoint, WireProtocol,
};
use async_trait::async_trait;

pub struct ClaudeCliResolver;

#[async_trait]
impl ProviderResolver for ClaudeCliResolver {
    fn resolve(
        &self,
        spec: &ProviderSpec,
        _credentials: &dyn CredentialSource,
    ) -> Result<ResolvedEndpoint, EngineResolutionError> {
        // Subprocess path: no HTTP endpoint, no yah-managed credential.
        // The `claude` binary handles its own auth from ~/.claude/.credentials.json.
        Ok(ResolvedEndpoint {
            endpoint: "subprocess://claude".into(),
            auth: AuthMaterial::None,
            // `opus` is a `claude --model` alias that resolves to the
            // latest Opus at spawn time. Keeps yah's default model
            // current as Anthropic rotates the underlying version
            // without touching this file.
            default_model: "opus".into(),
            kind: ConnectionKind::Cloud,
            provider_label: "claude".into(),
            wire_protocol: WireProtocol::Process,
            reasoning: spec.reasoning,
        })
    }
}
