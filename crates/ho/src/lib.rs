//! yah-runner-ho — the OpenAI-compatible cell of the yah agent runner
//! (R950-F5, W357 §3).
//!
//! [`OpenAiCompatRunner`] implements `yah_runner_core::Runner` for every
//! backend that speaks `POST /v1/chat/completions` with SSE (OpenAI, vLLM,
//! LiteLLM, OpenRouter, Groq, …) and for Ollama's native `/api/chat` NDJSON.
//! All HTTP goes through `yah_runner_core::HttpClient`; the reqwest substrate
//! is the default `reqwest` feature.
//!
//! Host policy arrives as plain [`OpenAiCompatConfig`] fields rather than
//! calls back into a host crate: the quota ledger a 429 is recorded in, the
//! rate-limit hint for a free-tier model, and the per-session preamble a host
//! wants trailing the rendered prelude.

mod openai_compat;
mod resolved;

pub use openai_compat::{
    list_openai_compat_models_with, CompactResult, OpenAiCompatConfig, OpenAiCompatRunner,
    RateLimitHint, ResidentAppend, SessionPreamble, Transport,
};
#[cfg(feature = "reqwest")]
pub use openai_compat::list_openai_compat_models;
