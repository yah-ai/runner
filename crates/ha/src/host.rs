//! The hook a host implements to run Anthropic turns over its own sessions.
//!
//! [`crate::run_anthropic_turn`] holds no session state. Everything it reads
//! (model, reasoning budget, rendered system prompt, history, tools,
//! cancellation) arrives as a [`TurnInput`] from [`HaHost::turn_input`], and
//! everything it would otherwise write into a session goes back out through
//! the other hooks. A host with its own session record (yah's camp) adapts
//! that record; [`crate::AnthropicRunner`] implements it over an in-memory
//! session for everyone else.

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use yah_runner_core::{
    AgentEvent, BlockReason, CacheTtl, Message, Prelude, ReasoningBudget, SessionId, ToolRegistry,
};

use crate::anthropic::{cache_control_value, AnthropicAuth};

/// The `system` blocks a turn sends, already rendered by the host.
///
/// The host decides how many of Anthropic's 4 cache breakpoints it spends
/// here by putting `cache_control` on the blocks it wants cached. The cell
/// spends whatever is left (at most 3) on a rolling window over the last
/// history messages, at [`Self::cache_ttl`].
#[derive(Debug, Clone)]
pub struct SystemPrompt {
    pub blocks: Vec<serde_json::Value>,
    /// TTL for the transcript breakpoints the cell adds behind the system
    /// prompt. Hosts that cache their system blocks should pass the same TTL
    /// they used there.
    pub cache_ttl: CacheTtl,
}

impl SystemPrompt {
    /// The neutral rendering for a host with no composition layer of its
    /// own: the OAuth cloak block when `auth` needs one, then the prelude's
    /// rendered sections as one cached block at the prelude's TTL.
    pub fn from_prelude(prelude: &Prelude, auth: &AnthropicAuth) -> Self {
        let cache_ttl = prelude.cache.ttl;
        let block = |text: &str| {
            serde_json::json!({
                "type": "text",
                "text": text,
                "cache_control": cache_control_value(cache_ttl),
            })
        };
        let mut blocks = Vec::new();
        if let Some(cloak) = auth.cloak_prefix() {
            blocks.push(block(cloak));
        }
        let rendered = prelude.render();
        if !rendered.is_empty() {
            blocks.push(block(&rendered));
        }
        Self { blocks, cache_ttl }
    }
}

/// Everything one provider request is built from.
pub struct TurnInput {
    pub session_id: SessionId,
    pub model: String,
    pub think: Option<ReasoningBudget>,
    pub system: SystemPrompt,
    pub history: Vec<Message>,
    /// Tools to advertise and dispatch. `None` (or an empty registry) sends
    /// no `tools` key and runs no tool loop.
    pub tool_registry: Option<Arc<dyn ToolRegistry>>,
    /// Checked while the turn waits out a pre-stream retry backoff.
    pub cancel: CancellationToken,
}

/// A phase change the turn reports while it streams. Hosts that show a
/// per-session indicator ("typing" vs "running a tool") map these onto
/// their own phase model; the turn's start and settle are the host's to
/// mark, since the host is what starts and settles it.
#[derive(Debug, Clone, PartialEq)]
pub enum HaPhase {
    /// A text block started.
    Emitting,
    /// A `tool_use` block started.
    ToolUse,
    /// The stream ended on a stop reason that blocks the session (context
    /// overflow surfaced as `stop_reason`, not as an HTTP error).
    Blocked(BlockReason),
}

/// The host side of an Anthropic turn.
#[async_trait]
pub trait HaHost: Send + Sync {
    /// The session's provider-facing state. Read once at turn start and again
    /// before each tool-loop re-issue, so a change the host makes mid-turn
    /// reaches the next request exactly as it would have when the cell read
    /// the session directly. `auth` is passed because the system prompt can
    /// depend on it (the OAuth cloak block).
    async fn turn_input(&self, auth: &AnthropicAuth) -> TurnInput;

    /// The turn's assistant message, once, after the stream settles.
    async fn push_history(&self, message: Message);

    /// Called after each turn-text event (`TurnStarted`, `MessageDelta`, and
    /// each terminal event) is emitted to the sink.
    async fn observe(&self, _event: &AgentEvent) {}

    /// Called at block boundaries; see [`HaPhase`].
    async fn set_phase(&self, _phase: HaPhase) {}

    /// The first request of the turn is about to go out with this exact body.
    fn request_built(
        &self,
        _session_id: &SessionId,
        _body: &serde_json::Value,
        _auth: &AnthropicAuth,
        _messages_url: &str,
    ) {
    }

    /// Response headers of each successful stream open, for hosts that
    /// record rate-limit headers.
    async fn response_headers(&self, _headers: &[(String, String)], _auth: &AnthropicAuth) {}
}
