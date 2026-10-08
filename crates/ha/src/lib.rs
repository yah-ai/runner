//! Anthropic-native (`/v1/messages`) cell of the yah agent runner.
//!
//! [`run_anthropic_turn`] is the turn loop, driven over a host-supplied
//! [`HaHost`]; [`AnthropicRunner`] wraps it as a core
//! [`yah_runner_core::Runner`] with in-memory sessions.

pub mod anthropic;
pub mod host;
pub mod runner;

pub use anthropic::{
    anthropic_tools_payload, build_anthropic_body, build_anthropic_body_with_tool_turns,
    cache_control_value, context_editing_beta_header, merge_anthropic_usage,
    model_supports_thinking, run_anthropic_turn, think_budget_tokens, AnthropicAuth, ApiTurnError,
    PinnedShape, ANTHROPIC_CLAUDE_CODE_SYSTEM_PREFIX, ANTHROPIC_MESSAGES_URL,
};
pub use host::{HaHost, HaPhase, SystemPrompt, TurnInput};
pub use runner::{reconstruct_history, AnthropicConfig, AnthropicRunner};
