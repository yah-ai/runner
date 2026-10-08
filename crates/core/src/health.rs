//! Per-runner connection health: error classification + state machine.
//!
//! [`ErrorKind`] maps a raw transport error or HTTP status code to a
//! health-relevant category. [`RunnerHealth`] is the state machine that
//! translates a sequence of successes and errors into
//! [`ConnectionStatus`] transitions. Callers emit
//! [`kg::agent::AgentEvent::ConnectionStatusChanged`] when either method
//! returns a new status.
//!
//! @arch:see(.yah/docs/working/W240-session-recovery-taxonomy.md)
//!
//! @arch:see(.yah/docs/working/W240-session-recovery-taxonomy.md)

use std::time::Instant;

use crate::agent::ConnectionStatus;

use crate::runner::RunnerError;

/// Coarse category for a runner error, used to drive [`RunnerHealth`].
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorKind {
    /// HTTP 429. May carry a `Retry-After` value parsed from the header.
    RateLimited { retry_after_secs: Option<u32> },
    /// HTTP 5xx — server-side error, potentially transient.
    ServerError,
    /// HTTP 401 / 403 — credential or permission problem.
    AuthFailure,
    /// Network-level failure (DNS, timeout, connection refused, TLS).
    Network,
    /// HTTP 400 / 404 / 422 or `UnsupportedEngine` — won't recover on retry.
    Permanent,
}

/// Classify a [`RunnerError`] into a health-relevant [`ErrorKind`].
pub fn classify_runner_error(err: &RunnerError) -> ErrorKind {
    match err {
        RunnerError::RateLimited { retry_after_secs } => ErrorKind::RateLimited {
            retry_after_secs: *retry_after_secs,
        },
        RunnerError::Transport(msg) => classify_message(msg),
        RunnerError::Config(_) => ErrorKind::AuthFailure,
        RunnerError::UnsupportedEngine { .. }
        | RunnerError::SessionNotFound(_)
        | RunnerError::ModelMissing { .. } => ErrorKind::Permanent,
    }
}

/// Classify a free-form error message string (e.g. `TurnFailed.message`)
/// into a health-relevant [`ErrorKind`]. Parses "returned NNN" / "HTTP NNN"
/// patterns; falls back to [`ErrorKind::Network`] when no HTTP code is found.
///
/// Also parses the `[retry-after: Ns]` annotation embedded by
/// `run_anthropic_turn` into 429 error strings so the retry window
/// survives the `Result<(), String>` boundary.
pub fn classify_message(msg: &str) -> ErrorKind {
    match extract_http_status(msg) {
        Some(429) => ErrorKind::RateLimited {
            retry_after_secs: extract_retry_after_annotation(msg),
        },
        Some(401) | Some(403) => ErrorKind::AuthFailure,
        Some(400) | Some(404) | Some(422) => ErrorKind::Permanent,
        Some(500..=599) => ErrorKind::ServerError,
        _ => ErrorKind::Network,
    }
}

/// Parse `[retry-after: Ns]` from an error message string. Used by the
/// Anthropic runner path which embeds this annotation when the server sends
/// a `Retry-After` header.
fn extract_retry_after_annotation(msg: &str) -> Option<u32> {
    let marker = "[retry-after: ";
    let pos = msg.find(marker)?;
    let rest = &msg[pos + marker.len()..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// Parse an HTTP status code out of common error message patterns:
/// `"… returned NNN: …"` and `"… HTTP NNN: …"`.
fn extract_http_status(msg: &str) -> Option<u16> {
    for keyword in &[" returned ", " HTTP "] {
        if let Some(pos) = msg.find(keyword) {
            let rest = &msg[pos + keyword.len()..];
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = digits.parse::<u16>() {
                return Some(n);
            }
        }
    }
    None
}

/// Tracks connection health for one endpoint. Translates error/success
/// signals into [`ConnectionStatus`] transitions.
///
/// Designed to be embedded in session state (behind the session `Mutex`).
/// Callers emit [`kg::agent::AgentEvent::ConnectionStatusChanged`] when
/// [`on_error`] or [`on_success`] returns a new status.
pub struct RunnerHealth {
    status: ConnectionStatus,
    /// Wall-clock instant when we entered the current non-Healthy state.
    since: Instant,
    /// Consecutive errors since the last success (reset on recovery).
    error_count: u32,
}

impl Default for RunnerHealth {
    fn default() -> Self {
        Self::new()
    }
}

impl RunnerHealth {
    pub fn new() -> Self {
        Self {
            status: ConnectionStatus::Healthy,
            since: Instant::now(),
            error_count: 0,
        }
    }

    /// Current health status (may be stale — update via [`on_error`] /
    /// [`on_success`]).
    pub fn current(&self) -> &ConnectionStatus {
        &self.status
    }

    /// Seconds elapsed since the current non-Healthy state began. Returns `0`
    /// when the connection is [`ConnectionStatus::Healthy`]. Used by callers
    /// building a [`crate::fallback::MetricSnapshot`] for fallback evaluation.
    pub fn since_secs(&self) -> u32 {
        if matches!(self.status, ConnectionStatus::Healthy) {
            0
        } else {
            self.since.elapsed().as_secs() as u32
        }
    }

    /// Call after a successful turn. Returns `Some(Healthy)` when this
    /// success transitions us out of a degraded/unhealthy/rate-limited
    /// state; returns `None` when already healthy (no event needed).
    pub fn on_success(&mut self) -> Option<ConnectionStatus> {
        if matches!(self.status, ConnectionStatus::Healthy) {
            return None;
        }
        self.status = ConnectionStatus::Healthy;
        self.since = Instant::now();
        self.error_count = 0;
        Some(ConnectionStatus::Healthy)
    }

    /// Call after each error. Returns the new [`ConnectionStatus`], which
    /// callers should broadcast via `ConnectionStatusChanged`. Always
    /// returns a value — even if the state didn't change — so callers
    /// can emit without separately tracking prior state (the renderer
    /// deduplicates idempotent transitions).
    pub fn on_error(&mut self, kind: &ErrorKind) -> ConnectionStatus {
        // Start the degradation clock on the first error after a healthy spell.
        if matches!(self.status, ConnectionStatus::Healthy) {
            self.since = Instant::now();
            self.error_count = 0;
        }
        self.error_count = self.error_count.saturating_add(1);
        let since_secs = self.since.elapsed().as_secs() as u32;

        let next = match kind {
            ErrorKind::RateLimited { retry_after_secs } => ConnectionStatus::RateLimited {
                retry_after_secs: *retry_after_secs,
            },
            // Permanent errors (auth, bad-request) are not retryable.
            ErrorKind::AuthFailure | ErrorKind::Permanent => {
                ConnectionStatus::Unhealthy { since_secs }
            }
            // Transient errors degrade before going unhealthy.
            ErrorKind::ServerError | ErrorKind::Network => {
                if self.error_count >= 3 {
                    ConnectionStatus::Unhealthy { since_secs }
                } else {
                    ConnectionStatus::Degraded {
                        since_secs,
                        retries: self.error_count,
                    }
                }
            }
        };
        self.status = next.clone();
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_message_extracts_status_from_returned_pattern() {
        assert_eq!(
            classify_message("openai returned 429: rate limit"),
            ErrorKind::RateLimited {
                retry_after_secs: None
            }
        );
        assert_eq!(classify_message("service returned 503: overloaded"), ErrorKind::ServerError);
        assert_eq!(classify_message("Anthropic returned 401: unauthorized"), ErrorKind::AuthFailure);
        assert_eq!(classify_message("request failed: connection refused"), ErrorKind::Network);
    }

    #[test]
    fn classify_message_extracts_retry_after_annotation() {
        let msg = "Anthropic returned 429: too many requests [retry-after: 30s]";
        assert_eq!(
            classify_message(msg),
            ErrorKind::RateLimited {
                retry_after_secs: Some(30)
            }
        );
    }

    #[test]
    fn health_degrades_then_unhealthy() {
        let mut h = RunnerHealth::new();
        assert!(matches!(h.current(), ConnectionStatus::Healthy));

        h.on_error(&ErrorKind::Network);
        assert!(matches!(h.current(), ConnectionStatus::Degraded { retries: 1, .. }));

        h.on_error(&ErrorKind::ServerError);
        assert!(matches!(h.current(), ConnectionStatus::Degraded { retries: 2, .. }));

        h.on_error(&ErrorKind::Network);
        assert!(matches!(h.current(), ConnectionStatus::Unhealthy { .. }));
    }

    #[test]
    fn health_recovers_on_success() {
        let mut h = RunnerHealth::new();
        h.on_error(&ErrorKind::ServerError);
        assert!(!matches!(h.current(), ConnectionStatus::Healthy));

        let recovered = h.on_success();
        assert_eq!(recovered, Some(ConnectionStatus::Healthy));
        assert!(matches!(h.current(), ConnectionStatus::Healthy));

        // Second success while already healthy is a no-op.
        assert_eq!(h.on_success(), None);
    }

    #[test]
    fn rate_limited_goes_directly_to_rate_limited() {
        let mut h = RunnerHealth::new();
        let status = h.on_error(&ErrorKind::RateLimited {
            retry_after_secs: Some(30),
        });
        assert_eq!(
            status,
            ConnectionStatus::RateLimited {
                retry_after_secs: Some(30)
            }
        );
    }
}
