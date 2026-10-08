//! Event sinks: where a runner's [`AgentEvent`]s go besides its own stream.
//!
//! [`SessionEventSink`] is generic over the event vocabulary so a host with a
//! wider event type than core's can implement the same trait for it, and a
//! cell can push core events into a host sink through a small adapter. The
//! type parameter defaults to core's [`AgentEvent`].

use crate::event::AgentEvent;

/// Receives the events emitted by an agent session.
pub trait SessionEventSink<E = AgentEvent>: Send + Sync {
    fn emit(&self, event: &E);
}

/// No-op sink for tests, dry-runs, and one-shot programmatic invocations
/// that don't care about the event stream. Intentionally not the
/// default for real runners — silent event drops on a real session
/// would be a hard-to-spot bug.
pub struct NullSink;

impl<E> SessionEventSink<E> for NullSink {
    fn emit(&self, _event: &E) {}
}

/// Adapter that fans every event into a closure. Useful for tests
/// (`Arc::new(FnSink::new(move |e: &AgentEvent| tx.send(e.clone())))`) and
/// for a CLI host where "print to stdout" is a one-liner.
pub struct FnSink<F>(pub F);

impl<F> FnSink<F> {
    pub fn new(f: F) -> Self {
        Self(f)
    }
}

impl<E, F: Fn(&E) + Send + Sync> SessionEventSink<E> for FnSink<F> {
    fn emit(&self, event: &E) {
        (self.0)(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::SessionId;
    use std::sync::{Arc, Mutex};

    fn fixture_session_started(id: &str) -> AgentEvent {
        AgentEvent::SessionStarted {
            session_id: SessionId(id.to_string()),
            session_key: "T".to_string(),
            engine: "test:engine".to_string(),
            cache_key: "k".to_string(),
        }
    }

    fn fixture_turn_ended(id: &str) -> AgentEvent {
        AgentEvent::TurnEnded {
            session_id: SessionId(id.to_string()),
            text: String::new(),
            stop_reason: None,
            usage: None,
            content_blocks: None,
        }
    }

    #[test]
    fn null_sink_swallows_events() {
        let sink: Box<dyn SessionEventSink> = Box::new(NullSink);
        sink.emit(&fixture_session_started("test"));
    }

    #[test]
    fn fn_sink_forwards_to_closure() {
        let captured: Arc<Mutex<Vec<AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let cap = Arc::clone(&captured);
        let sink = FnSink::new(move |e: &AgentEvent| cap.lock().unwrap().push(e.clone()));
        sink.emit(&fixture_session_started("a"));
        sink.emit(&fixture_turn_ended("a"));
        let log = captured.lock().unwrap();
        assert_eq!(log.len(), 2);
    }
}
