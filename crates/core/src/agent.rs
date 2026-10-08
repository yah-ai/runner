//! Conversation vocabulary shared by every runner and every host.
//!
//! Wire shapes here are load-bearing: hosts persist these to session logs
//! and mirror them in UI types, so serde attributes are part of the contract.

use serde::{Deserialize, Serialize};

/// Stable id for an agent session. Format: `session:<8 hex>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(pub String);

impl SessionId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A turn's content: plain text (every user turn, and an assistant turn on
/// providers/paths that never need block structure preserved) or a raw
/// content-block array captured verbatim off an engine response (assistant
/// turns once a provider needs the blocks round-tripped byte-for-byte —
/// Anthropic compaction blocks, `tool_use` blocks, thinking blocks, ...).
///
/// `#[serde(untagged)]` makes the old plain-string shape and the new
/// block-array shape both valid on read: a JSON string parses as `Text`, a
/// JSON array parses as `Blocks`. This is the R883-T10 migration — every
/// pre-existing serialized `Message` was a bare string, and stays loadable
/// with no version flag or dual-field shim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<serde_json::Value>),
}

impl MessageContent {
    /// Plain-text rendering for readers that don't care about block
    /// structure. `Blocks` concatenates every `type: "text"` block's `text`
    /// field in order; every other block type (`tool_use`, `compaction`,
    /// `thinking`, ...) contributes nothing here — it survives only in the
    /// raw blocks themselves.
    pub fn text(&self) -> std::borrow::Cow<'_, str> {
        match self {
            MessageContent::Text(s) => std::borrow::Cow::Borrowed(s.as_str()),
            MessageContent::Blocks(blocks) => {
                let mut out = String::new();
                for b in blocks {
                    if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                        if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                            out.push_str(t);
                        }
                    }
                }
                std::borrow::Cow::Owned(out)
            }
        }
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        MessageContent::Text(s)
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        MessageContent::Text(s.to_string())
    }
}

/// One conversation turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// Token-counting snapshot for one turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct TurnUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_context_tokens: Option<u32>,
}

impl TurnUsage {
    pub fn accumulate(&mut self, other: &Self) {
        let add = |slot: &mut Option<u32>, n: Option<u32>| {
            if let Some(v) = n {
                *slot = Some(slot.map_or(v, |x| x.saturating_add(v)));
            }
        };
        let max_field = |slot: &mut Option<u32>, n: Option<u32>| {
            if let Some(v) = n {
                *slot = Some(slot.map_or(v, |x: u32| x.max(v)));
            }
        };
        add(&mut self.input_tokens, other.input_tokens);
        add(&mut self.output_tokens, other.output_tokens);
        add(&mut self.cache_read_tokens, other.cache_read_tokens);
        add(&mut self.cache_write_tokens, other.cache_write_tokens);
        add(&mut self.thinking_tokens, other.thinking_tokens);
        max_field(&mut self.peak_context_tokens, other.peak_context_tokens);
    }
}

/// Typed reason a session ended wanting operator recovery (W240 taxonomy).
///
/// Each variant maps to a tailored recovery affordance on the FE. Detection
/// wiring lives in the runner's `block_detector` / `health` / `quota_ledger`
/// modules — see R575-T5/T6/T7. `Unknown { message }` remains the generic
/// fallback → Retry surface.
///
/// Wire form is `{ "kind": "snake_case_variant", ...fields }`. New variants
/// may be added non-breakingly under the `kind`-tagged shape; the FE
/// mirror in `types.ts` must be kept in lockstep.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BlockReason {
    /// Model refused the turn because the prompt exceeded its context window.
    /// Recovery: Compact / Hand off / Unlock-1M (claude-cli only).
    ContextOverflow,
    /// Provider returned 429 with an optional `Retry-After` hint.
    /// Recovery: wait / auto-retry countdown.
    RateLimited {
        #[serde(rename = "retryAfterSecs", default, skip_serializing_if = "Option::is_none")]
        retry_after_secs: Option<u32>,
    },
    /// Claude Max plan's 5-hour subscription window hit — the subprocess
    /// runner stops after emitting `claude_session_limit_hit`. Recovery:
    /// wait until reset / hand off to an API-keyed character.
    UsageLimitHit {
        #[serde(rename = "resetsAt", default, skip_serializing_if = "Option::is_none")]
        resets_at: Option<String>,
    },
    /// Subprocess-runner credentials went stale (401/403 on claude-cli).
    /// Recovery: interactive re-login (Phase 3 UX, R575-F9).
    AuthExpired,
    /// API-runner key is missing or invalid (401/403 on anthropic /
    /// openai-compat). Recovery: re-key in Settings → Agents.
    ApiAuth,
    /// Provider signaled billing/quota exhaustion.
    /// Recovery: top-up / billing link.
    QuotaExhausted,
    /// Catch-all — carries the raw provider message for the FE Retry surface.
    Unknown { message: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConnectionStatus {
    Healthy,
    Degraded {
        #[serde(rename = "sinceSecs")]
        since_secs: u32,
        retries: u32,
    },
    Unhealthy {
        #[serde(rename = "sinceSecs")]
        since_secs: u32,
    },
    RateLimited {
        #[serde(rename = "retryAfterSecs", default, skip_serializing_if = "Option::is_none")]
        retry_after_secs: Option<u32>,
    },
}
