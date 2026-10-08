//! The pv event vocabulary: core [`AgentEvent`]s plus the three claude-cli
//! lifecycle facts that have no provider-agnostic home.
//!
//! Part of R951. Kept out of `yah_runner_core::AgentEvent` on purpose — only
//! the claude `-p` translator produces these, and a host that does not care
//! can collapse them with [`PvEvent::into_core`].

use serde::{Deserialize, Serialize};
use yah_runner_core::{AgentEvent, SessionId};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PvEvent {
    /// A provider-agnostic event, passed through unchanged.
    Core(AgentEvent),
    /// claude reported `system/status: compacting` — a `/compact` (manual or
    /// auto) is running. The matching `ContextCompacted` arrives as a core
    /// event once the `compact_boundary` frame lands.
    CompactingStarted {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
    },
    /// A tool result carried claude's subscription session-limit message.
    /// Always preceded by `Core(TurnBlocked{UsageLimitHit})`; the host is
    /// expected to stop the subprocess — claude will make no further
    /// progress until `resets_at` (claude's own wire text, e.g.
    /// `"3pm (America/Los_Angeles)"`).
    SessionLimitHit {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        #[serde(rename = "resetsAt")]
        resets_at: String,
    },
    /// The turn ended on `terminal_reason: aborted_streaming|aborted_tools`
    /// — an operator interrupt or tool rejection. Neutral, not a failure;
    /// the subprocess stays alive for the next turn.
    TurnAborted {
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        text: String,
    },
}

impl From<AgentEvent> for PvEvent {
    fn from(ev: AgentEvent) -> Self {
        PvEvent::Core(ev)
    }
}

impl PvEvent {
    pub fn session_id(&self) -> &SessionId {
        match self {
            PvEvent::Core(ev) => ev.session_id(),
            PvEvent::CompactingStarted { session_id }
            | PvEvent::SessionLimitHit { session_id, .. }
            | PvEvent::TurnAborted { session_id, .. } => session_id,
        }
    }

    /// The core event, if this is one. The pv-only variants have no core
    /// equivalent and return `None`: `SessionLimitHit` is already preceded by
    /// a core `TurnBlocked`, and `TurnAborted` is mapped by
    /// [`crate::PvRunner`] to a `TurnEnded` with `stop_reason: "aborted"`.
    pub fn into_core(self) -> Option<AgentEvent> {
        match self {
            PvEvent::Core(ev) => Some(ev),
            _ => None,
        }
    }
}
