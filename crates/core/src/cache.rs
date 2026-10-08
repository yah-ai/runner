//! Prompt-cache hints a host attaches to a rendered prelude.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    /// Stable 32-char hex hash over the rendered prelude. Same content
    /// = same key = Anthropic prompt cache hit.
    pub key: String,
    pub ttl: CacheTtl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheTtl {
    /// Anthropic default: 5-minute TTL.
    Ephemeral,
    /// Anthropic extended: 1-hour TTL.
    Extended,
}
