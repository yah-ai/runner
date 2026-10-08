//! Turn-level events a [`crate::Runner`] emits.
//!
//! This is the subset of an agent session's event vocabulary that a runner
//! itself produces or needs back on resume: session open, the user turn,
//! streamed assistant output, tool round-trips, usage, and the three ways a
//! turn ends (ended / failed / blocked). Hosts that track more — slots,
//! dispatch, approvals, inboxes — keep their own superset enum and convert
//! from this one at the seam.
//!
//! The serde shape is kept a strict subset of yah's camp event log (same
//! `kind` tags, same camelCase field names), so a serialized core event
//! parses as the matching camp event.

use crate::agent::{BlockReason, ConnectionStatus, SessionId, TurnUsage};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AgentEvent {
    /// A session opened. `session_key` is the opaque key the host passed to
    /// [`crate::Runner::start`].
    SessionStarted {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        #[serde(rename = "sessionKey")]
        session_key: String,
        engine: String,
        #[serde(rename = "cacheKey")]
        cache_key: String,
    },
    /// The user turn a [`crate::Runner::send`] call appended.
    UserMessage {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        text: String,
    },
    TurnStarted {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
    },
    MessageDelta {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        text: String,
    },
    AssistantText {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        text: String,
    },
    ThinkingDelta {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        text: String,
    },
    TurnUsageDelta {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        usage: TurnUsage,
    },
    ContextCompacted {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        #[serde(rename = "preTokens")]
        pre_tokens: u64,
        #[serde(rename = "postTokens")]
        post_tokens: u64,
    },
    TurnEnded {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        text: String,
        #[serde(rename = "stopReason", skip_serializing_if = "Option::is_none")]
        stop_reason: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<TurnUsage>,
        /// The raw content-block array persisted for this turn when the
        /// provider's reply had block structure that must round-trip
        /// verbatim. `None` for a plain-text turn.
        #[serde(rename = "contentBlocks", default, skip_serializing_if = "Option::is_none")]
        content_blocks: Option<Vec<serde_json::Value>>,
    },
    /// A turn failed mid-stream. `text` is the assistant text accumulated up
    /// to the failure, so a partial reply is not lost.
    TurnFailed {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        text: String,
        message: String,
    },
    /// A turn stopped on a condition that needs recovery (context overflow,
    /// rate limit, auth, quota).
    TurnBlocked {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        text: String,
        reason: BlockReason,
    },
    ConnectionStatusChanged {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        #[serde(rename = "agentId")]
        agent_id: String,
        status: ConnectionStatus,
    },
    Error {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        message: String,
        #[serde(rename = "httpStatus", default, skip_serializing_if = "Option::is_none")]
        http_status: Option<u16>,
        #[serde(rename = "rawBody", default, skip_serializing_if = "Option::is_none")]
        raw_body: Option<String>,
        #[serde(rename = "errorType", default, skip_serializing_if = "Option::is_none")]
        error_type: Option<String>,
    },
    ToolCall {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        args: serde_json::Value,
    },
    ToolResult {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        ok: bool,
        result: serde_json::Value,
    },
    /// The model retried an identical failing tool call; the runner halted
    /// the tool loop rather than spin.
    ToolStuck {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        #[serde(rename = "toolName")]
        tool_name: String,
        args: serde_json::Value,
        result: serde_json::Value,
    },
    /// The runner replaced an oversized tool result in its history to keep
    /// the prompt under the context window.
    PromptTrimmed {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        #[serde(rename = "originalBytes")]
        original_bytes: u64,
        #[serde(rename = "replacementBytes")]
        replacement_bytes: u64,
    },
}

impl AgentEvent {
    /// The session this event belongs to.
    pub fn session_id(&self) -> &SessionId {
        match self {
            AgentEvent::SessionStarted { session_id, .. }
            | AgentEvent::UserMessage { session_id, .. }
            | AgentEvent::TurnStarted { session_id }
            | AgentEvent::MessageDelta { session_id, .. }
            | AgentEvent::AssistantText { session_id, .. }
            | AgentEvent::ThinkingDelta { session_id, .. }
            | AgentEvent::TurnUsageDelta { session_id, .. }
            | AgentEvent::ContextCompacted { session_id, .. }
            | AgentEvent::TurnEnded { session_id, .. }
            | AgentEvent::TurnFailed { session_id, .. }
            | AgentEvent::TurnBlocked { session_id, .. }
            | AgentEvent::ConnectionStatusChanged { session_id, .. }
            | AgentEvent::Error { session_id, .. }
            | AgentEvent::ToolCall { session_id, .. }
            | AgentEvent::ToolResult { session_id, .. }
            | AgentEvent::ToolStuck { session_id, .. }
            | AgentEvent::PromptTrimmed { session_id, .. } => session_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_ended_wire_shape_uses_camel_case_fields() {
        let ev = AgentEvent::TurnEnded {
            session_id: SessionId::new("session:00000001"),
            text: "hi".into(),
            stop_reason: Some("end_turn".into()),
            usage: None,
            content_blocks: None,
        };
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["kind"], "turn_ended");
        assert_eq!(v["sessionId"], "session:00000001");
        assert_eq!(v["stopReason"], "end_turn");
        assert!(v.get("contentBlocks").is_none());
        let back: AgentEvent = serde_json::from_value(v).unwrap();
        assert_eq!(back, ev);
    }
}
