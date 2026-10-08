//! Session log storage: the [`SessionStore`] trait a runner persists its
//! turn-level [`AgentEvent`]s through, and (off wasm32) [`FsSessionStore`], a
//! JSONL-per-session filesystem default.
//!
//! The store knows nothing about where a host keeps its logs. The filesystem
//! default takes its root directory as a constructor argument; a host decides
//! what that directory is (yah roots it at `<camp>/.yah/sessions`).
//!
//! ## One log, two vocabularies
//!
//! A host whose event vocabulary is a superset of the core [`AgentEvent`]
//! (yah's camp log carries dispatch, inbox and approval events alongside the
//! runner's turns) stores its own event type in the same file:
//! [`FsSessionStore`] is generic over any [`LogEvent`]. The [`SessionStore`]
//! trait is always spoken in core events — appending lifts a core event into
//! the host's type, reading projects the log back down and drops records the
//! core vocabulary has no variant for. The host's own superset reads stay on
//! [`FsSessionStore::replay`].
//!
//! ## Why JSONL
//!
//! - **Append-only** matches how events arrive: SSE deltas land one at a
//!   time, and the natural write is "open, append, close". No re-encoding
//!   cost on each write.
//! - **Line-delimited** means partial writes don't corrupt earlier records —
//!   a crashed turn loses at most the trailing line.
//! - **Self-describing** via the `kind` tag, so a session log parses cleanly
//!   without external metadata.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};
use thiserror::Error;

use crate::agent::{SessionId, TurnUsage};
use crate::event::AgentEvent;

#[cfg(not(target_arch = "wasm32"))]
mod fs;
#[cfg(not(target_arch = "wasm32"))]
pub use fs::{FsSessionStore, JsonReplay, EPHEMERAL_RING_CAP};

/// Errors from session log I/O. Most are surfaced to callers as warnings — a
/// runner shouldn't abort a turn because the log disk is full — but distinct
/// variants keep the failure mode readable in tracing output.
#[derive(Debug, Error)]
pub enum SessionStoreError {
    #[error("io error on session log: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error on session log: {0}")]
    Json(#[from] serde_json::Error),
    #[error("session id contains an unsafe path component: {0}")]
    UnsafeId(String),
}

/// An event type a session log can hold. The store needs exactly three facts
/// about a record beyond its serde shape: whether it carries a turn's final
/// usage (accumulated into the session's spend summary), and whether it opens
/// or closes the session.
pub trait LogEvent: Serialize + DeserializeOwned + Clone + Send + Sync + 'static {
    /// Final usage of a completed turn, if this record is one.
    fn turn_usage(&self) -> Option<&TurnUsage>;
    /// `true` for the record that opens a session.
    fn is_session_started(&self) -> bool;
    /// `true` for the terminal record. Appending it clears the running marker
    /// and stamps the accumulated usage onto the line.
    fn is_session_ended(&self) -> bool;
}

/// The core vocabulary has no terminal event — a runner ends turns, the host
/// ends sessions — so a core-only log never clears its own running marker;
/// the host calls [`SessionStore::clear_running`] when it closes the session.
impl LogEvent for AgentEvent {
    fn turn_usage(&self) -> Option<&TurnUsage> {
        match self {
            AgentEvent::TurnEnded { usage, .. } => usage.as_ref(),
            _ => None,
        }
    }

    fn is_session_started(&self) -> bool {
        matches!(self, AgentEvent::SessionStarted { .. })
    }

    fn is_session_ended(&self) -> bool {
        false
    }
}

/// Where a runner persists a session's events.
///
/// I/O that the filesystem default performs asynchronously (`append`,
/// `read`) is async here; the running-marker operations are synchronous
/// because they must land before any async session work begins — a process
/// killed mid-session has to leave the marker behind.
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Location of `session_id`'s log, for hosts that surface it. `None` for a
    /// store that keeps nothing on disk.
    fn path_for(&self, session_id: &SessionId) -> Result<Option<PathBuf>, SessionStoreError>;

    /// Append one event to the session's log.
    async fn append(&self, session_id: &SessionId, event: &AgentEvent)
        -> Result<(), SessionStoreError>;

    /// Every event in the session's log the core vocabulary can express, in
    /// order. A missing log reads as empty.
    async fn read(&self, session_id: &SessionId) -> Result<Vec<AgentEvent>, SessionStoreError>;

    /// Location of the running marker, which exists from session registration
    /// until the session ends cleanly. A marker left behind tells a reader the
    /// process died mid-session rather than pausing between turns.
    fn running_marker_path(
        &self,
        session_id: &SessionId,
    ) -> Result<Option<PathBuf>, SessionStoreError>;

    /// Write the running marker.
    fn touch_running(&self, session_id: &SessionId) -> Result<(), SessionStoreError>;

    /// Remove the running marker. A missing marker is not an error.
    fn clear_running(&self, session_id: &SessionId) -> Result<(), SessionStoreError>;
}

/// Reject session ids that would escape the store directory or expand to
/// surprising filenames. `session:<hex>` becomes `session:<hex>.jsonl` on
/// disk — the colon is fine on every host filesystem targeted. Only path
/// separators, parent traversal and NUL are rejected.
pub fn sanitize_id(id: &str) -> Result<&str, SessionStoreError> {
    if id.is_empty() || id == "." || id == ".." {
        return Err(SessionStoreError::UnsafeId(id.to_string()));
    }
    if id.contains('/') || id.contains('\\') || id.contains("..") || id.contains('\0') {
        return Err(SessionStoreError::UnsafeId(id.to_string()));
    }
    Ok(id)
}
