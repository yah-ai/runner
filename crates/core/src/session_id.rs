//! Session-id mint helper.
//!
//! Format: `session:<8 hex>` derived from `blake3(now_ms || counter)`.
//! 32 bits of entropy is plenty given sessions live in a per-process
//! map. Hosts that mint ids outside a runner should call this too;
//! sharing the helper keeps log scraping uniform
//! across runners.

use crate::agent::SessionId;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Mint a fresh `SessionId`. Backends are free to mint their own — the
/// contract is just "stable string, unique within the host process" —
/// but using this helper keeps the wire format consistent across
/// runners.
///
/// wasm32-unknown-unknown: this compiles there but `SystemTime::now()` panics
/// at runtime on that target. A wasm host must supply its own id (R950-T11:
/// left uncfg'd deliberately so the API surface is identical on every target).
pub fn mint_session_id() -> SessionId {
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let mut h = blake3::Hasher::new();
    h.update(&now.to_le_bytes());
    h.update(&counter.to_le_bytes());
    let hex = h.finalize().to_hex();
    SessionId::new(format!("session:{}", &hex.as_str()[..8]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn ids_carry_session_prefix_and_eight_hex_chars() {
        let id = mint_session_id();
        assert!(id.as_str().starts_with("session:"));
        assert_eq!(id.as_str().len(), "session:".len() + 8);
        assert!(id
            .as_str()
            .trim_start_matches("session:")
            .chars()
            .all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn rapid_mint_calls_collide_neither_on_clock_nor_counter() {
        // Sequential calls share a clock tick but bump the counter, so the
        // hash diverges. Exercises the counter leg of the uniqueness story.
        let mut seen = HashSet::new();
        for _ in 0..256 {
            assert!(seen.insert(mint_session_id()));
        }
    }
}
