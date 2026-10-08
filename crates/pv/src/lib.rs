//! Claude `-p` subprocess cell (pv) for the yah agent runner.
//!
//! Runs `claude --print --input-format stream-json --output-format
//! stream-json` as a long-lived subprocess per session and translates its
//! stdout into [`PvEvent`]s — core [`yah_runner_core::AgentEvent`]s plus three
//! claude-only lifecycle facts. [`PvRunner`] puts the cell behind core's
//! [`yah_runner_core::Runner`] trait.
//!
//! The host seam is [`PvHost`]: config paths, extra env, tool gating and
//! spawn/reap notifications. Persisting, emitting and phase tracking are the
//! host's business — it taps the event stream ([`frame::StreamReader`] for
//! hosts that drive the reader themselves).
//!
//! Tools reach claude only through MCP: the host's
//! [`PvHost::mcp_config_path`] names stdio servers claude spawns itself (see
//! `examples/mcp_tool_server.rs` for one built on `yah_runner_mcp`). pv never
//! links a tool registry.
//!
//! Part of R951 (W357 runner-core OSS split).

pub mod event;
pub mod frame;
pub mod gate;
pub mod host;
pub mod runner;
pub mod spawn;
pub mod transcript;
pub mod translate;

pub use event::PvEvent;
pub use gate::{GateDecider, GateRequest, GateVerdict, GATE_SOCK_ENV};
pub use host::{BareHost, PvHost};
pub use runner::{PvConfig, PvRunner};
pub use spawn::{SpawnSpec, STOP_GRACE};
pub use translate::translate_frame;
