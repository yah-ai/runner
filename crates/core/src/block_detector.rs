//! @arch:layer(runner)
//! @arch:role(detector)
//!
//! Typed [`BlockReason`] detection for runner failures (R262-F1). Each
//! backend exposes a detector that pattern-matches on its native error
//! shape and returns `Some(BlockReason)` on a definite block, `None`
//! otherwise so the caller can fall through to the generic error path.
//!
//! The runner-agnostic [`fallback`] inspects a free-form error message
//! against known overflow phrases as a catch-all for runners we haven't
//! typed yet — it emits [`BlockReason::Unknown`] so the UI gates
//! correctly even before per-runner typing lands.
//!
//! @arch:see(.yah/docs/working/W065-blocked-session-state.md)

use serde_json::Value;

use crate::agent::BlockReason;
use crate::health::ErrorKind;

/// Map a classified [`ErrorKind`] to a typed [`BlockReason`] for callers
/// deciding between `TurnBlocked` (typed recovery) and `Error`/`TurnFailed`
/// (generic). Returns `None` for kinds that should ride the generic path.
///
/// Every runner core knows about speaks HTTP, so [`ErrorKind::AuthFailure`]
/// is always a credential the operator re-enters (`ApiAuth`). The one case
/// W240 splits out — a literal `claude` subprocess losing its OAuth session
/// (`AuthExpired`, interactive re-login) — is a host concern: the subprocess
/// is not a core `Runner`, so the host maps it itself.
///
/// R575-T5 wired `RateLimited`; this (T6/T7) adds `AuthFailure` and keeps
/// billing/quota-limit signals in the sibling [`billing`] detector (called
/// separately by message-content, since a billing rejection can ride an
/// `ErrorKind` bucket — 400, or a 429 shared with real rate limits — that
/// doesn't distinguish it from the generic case on status code alone).
pub fn from_error_kind(kind: &ErrorKind) -> Option<BlockReason> {
    match kind {
        ErrorKind::RateLimited { retry_after_secs } => Some(BlockReason::RateLimited {
            retry_after_secs: *retry_after_secs,
        }),
        ErrorKind::AuthFailure => Some(BlockReason::ApiAuth),
        _ => None,
    }
}

/// Runner-agnostic billing/quota-exhaustion detector (R575-T7). Scans a
/// free-form error message for known billing-rejection phrases and emits
/// `QuotaExhausted` on match — called *before* [`from_error_kind`] at each
/// call site so a billing rejection wearing a 429 (OpenAI's
/// `insufficient_quota` is returned with the same HTTP status as a real
/// rate limit) or a 400 (Anthropic's low-credit-balance rejection) isn't
/// misclassified as `RateLimited` / generic `Unknown` first.
///
/// Phrases matched (case-insensitive substring):
/// - `"credit balance is too low"` — Anthropic's documented low-balance
///   rejection body.
/// - `"insufficient_quota"` — OpenAI's `error.code` for exhausted quota.
/// - `"exceeded your current quota"` — OpenAI's paired human-readable text.
/// - `"plans & billing"` — shared closing CTA on both providers' billing
///   rejections; catches phrasing drift the exact codes above miss.
pub fn billing(message: &str) -> Option<BlockReason> {
    let lc = message.to_ascii_lowercase();
    if lc.contains("credit balance is too low")
        || lc.contains("insufficient_quota")
        || lc.contains("exceeded your current quota")
        || lc.contains("plans & billing")
    {
        Some(BlockReason::QuotaExhausted)
    } else {
        None
    }
}

/// claude-cli `result`-frame detector. The wire shape captured in
/// [`crate::sessions`] callers (the `is_error: true` arm of
/// `agent_process::translate_frame`) is:
///
/// ```json
/// {
///   "type": "result",
///   "is_error": true,
///   "subtype": "success",
///   "result": "Prompt is too long",
///   "terminal_reason": "blocking_limit"
/// }
/// ```
///
/// Returns `Some(ContextOverflow)` when the `result` text matches a
/// known overflow phrase, `Some(Unknown{message})` when the frame is
/// otherwise flagged as `blocking_limit`, and `None` for any other
/// shape (so the caller falls through to the generic `TurnFailed`
/// path).
pub fn claude_cli(frame: &Value) -> Option<BlockReason> {
    if frame.get("type").and_then(Value::as_str) != Some("result") {
        return None;
    }
    if frame.get("is_error").and_then(Value::as_bool) != Some(true) {
        return None;
    }
    let result_text = frame.get("result").and_then(Value::as_str).unwrap_or("");
    let terminal_reason = frame.get("terminal_reason").and_then(Value::as_str);

    if matches_overflow(result_text) {
        return Some(BlockReason::ContextOverflow);
    }
    if terminal_reason == Some("blocking_limit") {
        return Some(BlockReason::Unknown {
            message: result_text.to_string(),
        });
    }
    None
}

/// Anthropic `/v1/messages` 400 detector. The provider returns
/// `error.type: "invalid_request_error"` for context-overflow
/// rejections; the message body contains `prompt is too long` or
/// `tokens > N`. Returns `Some(ContextOverflow)` on match. Non-400
/// statuses and non-overflow 400s return `None` so they ride the
/// generic `Error` path.
pub fn anthropic(status: u16, error_type: Option<&str>, message: &str) -> Option<BlockReason> {
    if status != 400 {
        return None;
    }
    // Anthropic always tags these as invalid_request_error, but we
    // tolerate a missing tag (older / mocked responses) and still
    // classify when the message matches.
    let tag_ok = error_type.map_or(true, |t| t == "invalid_request_error");
    if !tag_ok {
        return None;
    }
    if matches_overflow(message) {
        Some(BlockReason::ContextOverflow)
    } else {
        None
    }
}

/// Anthropic SSE stream-end detector for the 4.5+ overflow path.
///
/// Claude 4.5+ models return HTTP 200 and end the SSE stream with
/// `stop_reason: "model_context_window_exceeded"` rather than the
/// legacy 400 rejection. This detector is called with the final
/// `stop_reason` string extracted from the `message_delta` event;
/// it returns `Some(ContextOverflow)` on the magic value and `None`
/// for every other reason (`end_turn`, `tool_use`, `max_tokens`, etc.).
pub fn anthropic_stream_end(stop_reason: &str) -> Option<BlockReason> {
    if stop_reason == "model_context_window_exceeded" {
        Some(BlockReason::ContextOverflow)
    } else {
        None
    }
}

/// Runner-agnostic fallback. Scans the free-form error message for any
/// known overflow phrase and emits `Unknown{message}` on match — so the
/// UI surfaces the recovery modal even for runners we haven't typed
/// yet. Benign transport errors (5xx, timeouts) return `None`; the
/// caller routes those to `Errored` instead of `Blocked`.
pub fn fallback(message: &str) -> Option<BlockReason> {
    if matches_overflow(message) {
        Some(BlockReason::Unknown {
            message: message.to_string(),
        })
    } else {
        None
    }
}

/// Case-insensitive substring match against the known overflow phrases:
///
/// - `prompt is too long` (claude-cli, anthropic)
/// - `context_length_exceeded` (openai error code)
/// - `exceeds the context window` (ollama)
/// - `tokens > <digit>` (anthropic verbose form)
///
/// The `tokens >` check requires a trailing digit so messages like
/// `"tokens > everything"` (English prose, not a token count) don't
/// produce false positives.
fn matches_overflow(message: &str) -> bool {
    let lc = message.to_ascii_lowercase();
    if lc.contains("prompt is too long")
        || lc.contains("context_length_exceeded")
        || lc.contains("exceeds the context window")
    {
        return true;
    }
    if let Some(idx) = lc.find("tokens >") {
        let after = lc[idx + "tokens >".len()..].trim_start();
        if after.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Captured from a real `claude --print` run hitting prompt-too-long
    /// (the same fixture exercised by
    /// `translate_frame_result_is_error_uses_result_field_for_upstream_message`
    /// in `agent_process::tests`). The typed detector must classify
    /// this as `ContextOverflow` — the wire-extraction layer is shared
    /// with `translate_frame` but the *typing* is new.
    #[test]
    fn claude_cli_blocking_limit_with_overflow_text() {
        let frame = json!({
            "type": "result",
            "subtype": "success",
            "is_error": true,
            "result": "Prompt is too long",
            "terminal_reason": "blocking_limit",
        });
        assert_eq!(claude_cli(&frame), Some(BlockReason::ContextOverflow));
    }

    #[test]
    fn claude_cli_blocking_limit_with_unknown_text_emits_unknown() {
        // Future-proofing: claude-cli's `blocking_limit` may carry text
        // we don't recognise yet. Preserve the upstream message so the
        // operator can still see what tripped the runner.
        let frame = json!({
            "type": "result",
            "is_error": true,
            "result": "some new blocking signal",
            "terminal_reason": "blocking_limit",
        });
        assert_eq!(
            claude_cli(&frame),
            Some(BlockReason::Unknown {
                message: "some new blocking signal".into(),
            })
        );
    }

    #[test]
    fn claude_cli_aborted_streaming_is_not_a_block() {
        // Operator-initiated cancel — terminal_reason "aborted_streaming"
        // already routes to TurnAborted in translate_frame. The detector
        // must not steal that arm.
        let frame = json!({
            "type": "result",
            "subtype": "error_during_execution",
            "is_error": true,
            "terminal_reason": "aborted_streaming",
        });
        assert_eq!(claude_cli(&frame), None);
    }

    #[test]
    fn claude_cli_success_returns_none() {
        let frame = json!({
            "type": "result",
            "subtype": "success",
            "is_error": false,
        });
        assert_eq!(claude_cli(&frame), None);
    }

    #[test]
    fn claude_cli_non_result_frame_returns_none() {
        let frame = json!({
            "type": "stream_event",
            "event": { "type": "message_start" },
        });
        assert_eq!(claude_cli(&frame), None);
    }

    #[test]
    fn anthropic_400_invalid_request_with_overflow_message() {
        let r = anthropic(
            400,
            Some("invalid_request_error"),
            "prompt is too long: input length and `max_tokens` exceed context limit",
        );
        assert_eq!(r, Some(BlockReason::ContextOverflow));
    }

    #[test]
    fn anthropic_400_with_tokens_threshold_pattern() {
        let r = anthropic(
            400,
            Some("invalid_request_error"),
            "input length tokens > 200000",
        );
        assert_eq!(r, Some(BlockReason::ContextOverflow));
    }

    #[test]
    fn anthropic_400_non_overflow_returns_none() {
        // 400 invalid_request_error can also mean malformed JSON, bad
        // model name, etc. — those route to Error, not Blocked.
        let r = anthropic(400, Some("invalid_request_error"), "model not found");
        assert_eq!(r, None);
    }

    #[test]
    fn anthropic_500_with_overflow_message_returns_none() {
        // A 5xx is transient — never classify as Blocked even if the
        // body coincidentally mentions an overflow phrase. The session
        // should retry, not raise the recovery modal.
        let r = anthropic(500, Some("api_error"), "prompt is too long");
        assert_eq!(r, None);
    }

    #[test]
    fn openai_context_length_exceeded_via_fallback() {
        // openai isn't typed yet; its `error.code:"context_length_exceeded"`
        // rides as a substring in the raw body passed through fallback.
        let body = r#"{"error":{"code":"context_length_exceeded","message":"This model's maximum context length is 128000 tokens."}}"#;
        match fallback(body) {
            Some(BlockReason::Unknown { message }) => {
                assert!(message.contains("context_length_exceeded"));
            }
            other => panic!("expected Some(Unknown), got {other:?}"),
        }
    }

    #[test]
    fn fallback_ollama_phrase() {
        let r = fallback("prompt token count exceeds the context window");
        assert!(matches!(r, Some(BlockReason::Unknown { .. })));
    }

    #[test]
    fn fallback_benign_5xx_returns_none() {
        assert_eq!(fallback("upstream error: bad gateway"), None);
        assert_eq!(fallback("connection reset by peer"), None);
    }

    #[test]
    fn anthropic_stream_end_overflow_stop_reason() {
        assert_eq!(
            anthropic_stream_end("model_context_window_exceeded"),
            Some(BlockReason::ContextOverflow),
        );
    }

    #[test]
    fn anthropic_stream_end_normal_reasons_return_none() {
        assert_eq!(anthropic_stream_end("end_turn"), None);
        assert_eq!(anthropic_stream_end("tool_use"), None);
        assert_eq!(anthropic_stream_end("max_tokens"), None);
        assert_eq!(anthropic_stream_end("stop_sequence"), None);
    }

    #[test]
    fn from_error_kind_maps_rate_limited() {
        assert_eq!(
            from_error_kind(&ErrorKind::RateLimited {
                retry_after_secs: Some(45)
            }),
            Some(BlockReason::RateLimited {
                retry_after_secs: Some(45)
            })
        );
        assert_eq!(
            from_error_kind(&ErrorKind::RateLimited {
                retry_after_secs: None
            }),
            Some(BlockReason::RateLimited {
                retry_after_secs: None
            })
        );
    }

    #[test]
    fn from_error_kind_untyped_kinds_stay_none() {
        // ServerError/Network/Permanent stay generic — they ride the
        // Error/TurnFailed path so the Retry surface renders the raw message.
        assert_eq!(from_error_kind(&ErrorKind::ServerError), None);
        assert_eq!(from_error_kind(&ErrorKind::Network), None);
        assert_eq!(from_error_kind(&ErrorKind::Permanent), None);
    }

    #[test]
    fn from_error_kind_auth_failure_is_api_auth() {
        // Direct-HTTP Anthropic (API key or pasted setup-token bearer),
        // OpenAI-compat and Codex-OAuth all recover the same way: re-key in
        // Settings -> Agents.
        assert_eq!(
            from_error_kind(&ErrorKind::AuthFailure),
            Some(BlockReason::ApiAuth)
        );
    }

    #[test]
    fn billing_matches_anthropic_low_credit_balance() {
        let r = billing(
            "Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.",
        );
        assert_eq!(r, Some(BlockReason::QuotaExhausted));
    }

    #[test]
    fn billing_matches_openai_insufficient_quota() {
        let body = r#"{"error":{"code":"insufficient_quota","message":"You exceeded your current quota, please check your plan and billing details."}}"#;
        assert_eq!(billing(body), Some(BlockReason::QuotaExhausted));
    }

    #[test]
    fn billing_benign_message_returns_none() {
        assert_eq!(billing("connection reset by peer"), None);
        assert_eq!(billing("openai returned 429: rate limit"), None);
    }

    #[test]
    fn fallback_tokens_pattern_requires_digit() {
        // "tokens > N" matches; English prose with "tokens >" but no
        // trailing digit must not.
        assert!(fallback("input has tokens > 1024 — overflow").is_some());
        assert_eq!(fallback("tokens > everything else here"), None);
    }
}
