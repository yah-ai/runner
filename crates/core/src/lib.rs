//! Core vocabulary and the provider-agnostic [`Runner`] trait for the yah
//! agent runner.
//!
//! Everything here is host-free: no ticket, camp, board or knowledge-graph
//! types, no HTTP client, no host-chosen filesystem paths. A host (yah's camp runner, or
//! any other product) fills a rendered [`Prelude`], opens a session with an
//! opaque session key, and reads a stream of turn-level [`AgentEvent`]s.
//! Tools reach the model through the [`ToolRegistry`] trait the host
//! implements.

pub mod agent;
pub mod block_detector;
pub mod cache;
pub mod event;
pub mod fallback;
pub mod health;
pub mod http;
pub mod model_catalog;
pub mod prelude;
pub mod quota_ledger;
pub mod resolver;
pub mod runner;
pub mod session_id;
pub mod session_store;
pub mod sink;
pub mod structured_output;
pub mod text_filter;
pub mod tool;
pub mod usage_store;

#[cfg(test)]
mod test_support;

pub use agent::{BlockReason, ConnectionStatus, Message, MessageContent, Role, SessionId, TurnUsage};
pub use cache::{CacheControl, CacheTtl};
pub use event::AgentEvent;
pub use http::{ByteStream, HttpClient, HttpError, HttpMethod, HttpRequest, HttpResponse};
#[cfg(feature = "reqwest")]
pub use http::ReqwestClient;
pub use prelude::{ModelRef, Prelude, PreludeSection, ReasoningBudget};
pub use runner::{EventStream, Runner, RunnerError, RunnerKind};
pub use session_id::mint_session_id;
pub use session_store::{LogEvent, SessionStore, SessionStoreError};
pub use sink::{FnSink, NullSink, SessionEventSink};
#[cfg(not(target_arch = "wasm32"))]
pub use session_store::{FsSessionStore, EPHEMERAL_RING_CAP};
pub use tool::{
    build_wire_name_map, sanitize_tool_name_for_wire, tool_result_wire_content, ToolOutcome,
    ToolRegistry, ToolSchema, TOOL_RESULT_CAP_BYTES,
};
pub use usage_store::{IdentitySummary, RateLimitSample, UsageStore};
#[cfg(not(target_arch = "wasm32"))]
pub use usage_store::FsUsageStore;
