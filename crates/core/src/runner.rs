//! The [`Runner`] trait — the surface every backend implements.
//!
//! Kept deliberately narrow and host-free: open a session from a rendered
//! [`Prelude`] and an opaque session key, send a user turn and read back a
//! stream of [`AgentEvent`]s, stop, and resume from history. Anything a
//! particular host layers on top (mid-session prompt mutation, live
//! reasoning-level changes, in-place compaction, slots, steering) belongs in
//! that host's own extension trait over this one.

use crate::agent::SessionId;
use crate::event::AgentEvent;
use crate::prelude::Prelude;
use crate::tool::ToolRegistry;
use async_trait::async_trait;
use futures_core::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use thiserror::Error;

/// Boxed `Stream` of [`AgentEvent`]s. One per [`Runner::send`] call.
///
/// `'static` because the stream typically outlives the borrowed `&self`
/// of `send` — backends own the underlying state in `Arc`s and emit
/// from a spawned task.
pub type EventStream = Pin<Box<dyn Stream<Item = AgentEvent> + Send + 'static>>;

/// Which runner family an implementation belongs to. Lets shared code
/// branch on family (e.g. how an auth failure should be recovered) without
/// matching on trait-object identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RunnerKind {
    /// OpenAI-compatible `/v1/chat/completions` over HTTP (OpenAI, Ollama,
    /// vLLM, Qwen, ...).
    #[default]
    OpenAiCompat,
    /// Anthropic-native `/v1/messages` over HTTP.
    AnthropicNative,
    /// OpenAI Codex Responses API on a ChatGPT-account OAuth credential.
    CodexOauth,
}

/// Errors a [`Runner`] can surface synchronously (i.e. before a turn has
/// been spawned). Mid-turn failures ride [`AgentEvent::TurnFailed`] on
/// the stream — symmetric to [`AgentEvent::TurnEnded`], it carries the
/// accumulated assistant text up to the failure point. The session
/// stays open in either case so the user can retry without re-paying
/// the prelude assembly cost. [`AgentEvent::Error`] is reserved for
/// command-shaped failures with no in-flight turn (e.g. the host's
/// pre-stream `send` itself returning [`RunnerError`]).
#[derive(Debug, Error)]
pub enum RunnerError {
    /// The prelude's model ref names a provider this runner doesn't serve.
    /// Hosts should normally route before invoking the runner; surfacing it
    /// here is defence-in-depth for callers that mint a Runner directly.
    #[error("engine not supported by this runner: {engine}")]
    UnsupportedEngine { engine: String },
    /// Session lookup failed — usually a stale id from the caller.
    #[error("session not found: {0}")]
    SessionNotFound(String),
    /// Backend-specific configuration is missing (API key, endpoint URL,
    /// model name, …). Carries a user-readable hint.
    #[error("configuration missing: {0}")]
    Config(String),
    /// I/O / transport failure during setup. Once a turn is in flight,
    /// transport errors should ride [`AgentEvent::TurnFailed`] on the
    /// stream instead — that variant carries the accumulator so partial
    /// replies aren't lost on a mid-stream disconnect.
    #[error("transport error: {0}")]
    Transport(String),
    /// HTTP 429 rate limit. `retry_after_secs` is parsed from the
    /// `Retry-After` response header when present; `None` means the
    /// endpoint specified no backoff duration.
    #[error("rate limited{}", .retry_after_secs.map(|s| format!(", retry after {s}s")).unwrap_or_default())]
    RateLimited { retry_after_secs: Option<u32> },
    /// Requested model tag is not installed in the local Ollama instance.
    /// `available` is the tag set returned by `/api/tags` at preflight time.
    /// `fallback` names the substitution if the caller declared one; `None`
    /// means no fallback and the caller should block session start with a
    /// pull hint.
    #[error("model '{tag}' is not installed in local Ollama (available: {available:?})")]
    ModelMissing { tag: String, available: Vec<String>, fallback: Option<String> },
}

/// Provider-agnostic agent runner.
///
/// One implementation per provider family. Implementations should be
/// `Arc<Self>`-shareable: hosts hold one runner instance for the process
/// lifetime and dispatch concurrent sessions through it.
#[async_trait]
pub trait Runner: Send + Sync {
    /// Open a session keyed by `session_key` using the rendered `prelude`.
    /// Returns the freshly-minted [`SessionId`]. The implementation
    /// retains whatever state it needs (history, cached prefix, model)
    /// keyed by the session id.
    ///
    /// `session_key` is opaque to the runner — a host passes whatever
    /// identifies the unit of work (a ticket id, a project id, ...).
    ///
    /// `prelude` is consumed because the runner caches the rendered
    /// text plus model/reasoning metadata internally; callers who want to
    /// inspect the prelude after start should clone before calling.
    async fn start(&self, prelude: Prelude, session_key: &str) -> Result<SessionId, RunnerError>;

    /// Append a user turn and return a stream of [`AgentEvent`]s. The
    /// stream is bounded to one turn — implementations close it after
    /// emitting **exactly one** of [`AgentEvent::TurnEnded`] (clean
    /// completion), [`AgentEvent::TurnFailed`] (mid-stream failure
    /// carrying the partial accumulator) or [`AgentEvent::TurnBlocked`].
    /// Stream closure does not close the session; subsequent `send` calls
    /// reuse the same id.
    async fn send(
        &self,
        session_id: &SessionId,
        message: String,
    ) -> Result<EventStream, RunnerError>;

    /// Abort any in-flight turn and tear down the session. Idempotent —
    /// returns `Ok(false)` when no session matched.
    async fn stop(&self, session_id: &SessionId) -> Result<bool, RunnerError>;

    /// Rehydrate a session from its recorded event history.
    ///
    /// Called when waking a paused session. The caller supplies the event
    /// sequence it persisted; the runner reconstructs its internal state
    /// (messages array, model, prelude text) so that subsequent
    /// [`Self::send`] calls succeed as if the session had been `start`-ed
    /// with the accumulated history.
    ///
    /// The default implementation returns `Err(Config(_))` — backends
    /// whose conversation state lives outside this process override this
    /// only if they can meaningfully reconstruct in-process from events.
    async fn resume(
        &self,
        _session_id: &SessionId,
        _history: Vec<AgentEvent>,
        _prelude: Prelude,
    ) -> Result<(), RunnerError> {
        Err(RunnerError::Config("resume not supported by this runner".into()))
    }

    /// Whether this runner reads the reasoning level per request, so a
    /// host-side change takes effect on the next turn. `false` (the
    /// default) means the level was fixed at start.
    fn think_live(&self) -> bool {
        false
    }

    /// Borrow the runner's attached tool registry, when one was wired in at
    /// construction time. A host reaches for this to propagate per-turn
    /// signals (currently the cancel token via
    /// [`ToolRegistry::bind_cancel`]). Default `None` for backends that
    /// don't dispatch tools through a host registry.
    fn tools_registry(&self) -> Option<&Arc<dyn ToolRegistry>> {
        None
    }

    /// The [`RunnerKind`] this implementation embodies. Defaults to
    /// [`RunnerKind::OpenAiCompat`], the most common HTTP backend.
    fn kind(&self) -> RunnerKind {
        RunnerKind::OpenAiCompat
    }
}
