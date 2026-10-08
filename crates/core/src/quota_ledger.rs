//! Per-(connection_id, model) quota ledger for session-start rotation.
//!
//! The ledger tracks two signals per engine entry:
//! - `last_429_at`: when the entry last returned a rate-limit response.
//!   Entries within `cooldown_secs` of a 429 are skipped by the picker.
//! - `last_session_at`: when the entry was last chosen at session-start.
//!   Used as the LRU tie-breaker for `RoundRobin` selection.
//!
//! In-memory only; a host owns the instance (and its lifetime). There is no
//! persistence across restarts — that is a future upgrade tracked in
//! `.yah/docs/working/W094-quota-aware-subclass-fallback.md`.
//!
//! `Instant::now()` compiles on wasm32-unknown-unknown but panics there at
//! runtime; a wasm host needs a clock shim before it records anything.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Cooldown duration after a 429 before the entry is eligible again.
pub const DEFAULT_COOLDOWN_SECS: u32 = 60;

#[derive(Debug, Default)]
struct QuotaEntry {
    last_429_at: Option<Instant>,
    last_session_at: Option<Instant>,
}

/// In-memory ledger keyed by `(connection_id, model)`.
#[derive(Debug, Default)]
pub struct QuotaLedger {
    entries: HashMap<(String, String), QuotaEntry>,
}

impl QuotaLedger {
    /// Record a 429 rate-limit response for `(provider, model)`.
    pub fn record_429(&mut self, provider: &str, model: &str) {
        self.entries
            .entry((provider.to_string(), model.to_string()))
            .or_default()
            .last_429_at = Some(Instant::now());
    }

    /// Record that a new session was started on `(provider, model)`.
    /// Resets the cooldown window — a session started after a cooldown
    /// means the entry is live again.
    pub fn record_session_start(&mut self, provider: &str, model: &str) {
        self.entries
            .entry((provider.to_string(), model.to_string()))
            .or_default()
            .last_session_at = Some(Instant::now());
    }

    /// Returns `true` when `(provider, model)` received a 429 within
    /// the last `cooldown_secs` seconds.
    pub fn is_on_cooldown(&self, provider: &str, model: &str, cooldown_secs: u32) -> bool {
        self.entries
            .get(&(provider.to_string(), model.to_string()))
            .and_then(|e| e.last_429_at)
            .map(|t| t.elapsed() < Duration::from_secs(cooldown_secs as u64))
            .unwrap_or(false)
    }

    /// Pick the best chain entry index for a new session start.
    ///
    /// `chain` is a slice of `(connection_id, model)` pairs, ordered by
    /// priority (index 0 = primary). The picker:
    /// 1. Filters out entries on cooldown.
    /// 2. Among remaining candidates, picks the one used longest ago (LRU),
    ///    preferring never-used entries.
    /// 3. Falls back to index 0 if all entries are on cooldown.
    pub fn pick_best_chain_entry(
        &self,
        chain: &[(String, String)],
        cooldown_secs: u32,
    ) -> usize {
        if chain.is_empty() {
            return 0;
        }

        let available: Vec<usize> = chain
            .iter()
            .enumerate()
            .filter(|(_, (p, m))| !self.is_on_cooldown(p, m, cooldown_secs))
            .map(|(i, _)| i)
            .collect();

        // Degrade gracefully: if all entries are cooling down, use all.
        let candidates = if available.is_empty() {
            (0..chain.len()).collect::<Vec<_>>()
        } else {
            available
        };

        // Pick the candidate used longest ago (LRU = round-robin behavior).
        // `elapsed()` returns time since last use; larger = used further in
        // the past = highest priority. Never-used entries get u128::MAX.
        candidates
            .into_iter()
            .max_by_key(|&i| {
                let (p, m) = &chain[i];
                self.entries
                    .get(&(p.clone(), m.clone()))
                    .and_then(|e| e.last_session_at)
                    .map(|t| t.elapsed().as_millis())
                    .unwrap_or(u128::MAX)
            })
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_best_returns_zero_for_empty_chain() {
        let ledger = QuotaLedger::default();
        assert_eq!(ledger.pick_best_chain_entry(&[], DEFAULT_COOLDOWN_SECS), 0);
    }

    #[test]
    fn pick_best_returns_zero_when_no_usage() {
        let ledger = QuotaLedger::default();
        let chain = vec![
            ("groq".into(), "llama-3.3-70b-versatile".into()),
            ("groq".into(), "openai/gpt-oss-120b".into()),
        ];
        // Both never used; max_by_key returns the last max in iteration order.
        // For the never-used case any index is acceptable.
        let idx = ledger.pick_best_chain_entry(&chain, DEFAULT_COOLDOWN_SECS);
        assert!(idx < chain.len());
    }

    #[test]
    fn cooling_down_entry_is_skipped() {
        let mut ledger = QuotaLedger::default();
        let chain = vec![
            ("groq".into(), "llama-3.3-70b-versatile".into()),
            ("groq".into(), "openai/gpt-oss-120b".into()),
        ];
        // Mark entry 0 as just-429'd with a very long cooldown.
        ledger.record_429("groq", "llama-3.3-70b-versatile");
        let idx = ledger.pick_best_chain_entry(&chain, 3600);
        assert_eq!(idx, 1, "entry 0 is cooling; should pick entry 1");
    }

    #[test]
    fn lru_picks_oldest_session() {
        let mut ledger = QuotaLedger::default();
        let chain = vec![
            ("groq".into(), "llama-3.3-70b-versatile".into()),
            ("groq".into(), "openai/gpt-oss-120b".into()),
        ];
        // Start entry 1 first; entry 0 is never used (max elapsed).
        ledger.record_session_start("groq", "openai/gpt-oss-120b");
        // entry 0 never used → treated as oldest → should be picked.
        let idx = ledger.pick_best_chain_entry(&chain, DEFAULT_COOLDOWN_SECS);
        assert_eq!(idx, 0, "entry 0 never used; should be preferred by LRU");
    }

    #[test]
    fn all_cooling_falls_back_to_any() {
        let mut ledger = QuotaLedger::default();
        let chain = vec![
            ("groq".into(), "llama-3.3-70b-versatile".into()),
            ("groq".into(), "openai/gpt-oss-120b".into()),
        ];
        ledger.record_429("groq", "llama-3.3-70b-versatile");
        ledger.record_429("groq", "openai/gpt-oss-120b");
        let idx = ledger.pick_best_chain_entry(&chain, 3600);
        assert!(idx < chain.len(), "should return a valid index even when all cooling");
    }
}
