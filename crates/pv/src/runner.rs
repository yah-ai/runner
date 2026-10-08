//! [`PvRunner`]: the claude `-p` cell behind core's [`Runner`] trait.
//!
//! Part of R951. One long-lived subprocess per session; each
//! [`Runner::send`] writes one stream-json user line and streams events
//! until the turn ends. [`PvRunner::send_pv`] exposes the full [`PvEvent`]
//! vocabulary for hosts that act on the claude-only lifecycle facts.

use crate::event::PvEvent;
use crate::frame::{encode_interrupt_line, encode_user_line, ReadStep, StreamReader};
use crate::host::PvHost;
use crate::spawn::{spawn_claude, stop_child, SpawnSpec};
use async_trait::async_trait;
use futures_core::Stream;
use std::collections::HashMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::io::AsyncWriteExt;
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::{mpsc, Mutex};
use yah_runner_core::{
    mint_session_id, AgentEvent, CacheTtl, EventStream, Prelude, Runner, RunnerError, RunnerKind,
    SessionId,
};

/// Static configuration for a [`PvRunner`].
#[derive(Debug, Clone)]
pub struct PvConfig {
    /// `claude`, or a claude-cli-compatible fork.
    pub cli_binary: String,
    /// Subprocess cwd for every session.
    pub workdir: PathBuf,
    /// Model used when the prelude names none.
    pub default_model: String,
}

struct PvSession {
    child: Child,
    stdin: ChildStdin,
    reader: Arc<Mutex<StreamReader<ChildStdout>>>,
}

pub struct PvRunner {
    host: Arc<dyn PvHost>,
    config: PvConfig,
    sessions: Mutex<HashMap<SessionId, PvSession>>,
}

impl PvRunner {
    pub fn new(host: Arc<dyn PvHost>, config: PvConfig) -> Self {
        Self { host, config, sessions: Mutex::new(HashMap::new()) }
    }

    /// Like [`Runner::send`], but yields the raw [`PvEvent`]s.
    pub async fn send_pv(
        &self,
        session_id: &SessionId,
        message: String,
    ) -> Result<mpsc::Receiver<PvEvent>, RunnerError> {
        let reader = {
            let mut sessions = self.sessions.lock().await;
            let s = sessions
                .get_mut(session_id)
                .ok_or_else(|| RunnerError::SessionNotFound(session_id.as_str().to_string()))?;
            write_line(&mut s.stdin, &encode_user_line(&message)).await?;
            s.reader.clone()
        };
        let (tx, rx) = mpsc::channel(64);
        tokio::spawn(async move {
            let mut reader = reader.lock().await;
            loop {
                match reader.next().await {
                    ReadStep::Frame { events, .. } => {
                        let terminal = events.iter().any(is_turn_terminal);
                        for ev in events {
                            if tx.send(ev).await.is_err() {
                                return;
                            }
                        }
                        if terminal {
                            return;
                        }
                    }
                    ReadStep::Eof => {
                        if let Some(ev) = reader.exit_event(true) {
                            let _ = tx.send(ev.into()).await;
                        }
                        return;
                    }
                    ReadStep::ReadError(ev) => {
                        let _ = tx.send(ev.into()).await;
                        return;
                    }
                }
            }
        });
        Ok(rx)
    }

    /// Soft-cancel the in-flight turn; it ends with [`PvEvent::TurnAborted`].
    pub async fn interrupt(&self, session_id: &SessionId, request_id: &str) -> Result<(), RunnerError> {
        let mut sessions = self.sessions.lock().await;
        let s = sessions
            .get_mut(session_id)
            .ok_or_else(|| RunnerError::SessionNotFound(session_id.as_str().to_string()))?;
        write_line(&mut s.stdin, &encode_interrupt_line(request_id)).await
    }
}

async fn write_line(stdin: &mut ChildStdin, line: &str) -> Result<(), RunnerError> {
    stdin
        .write_all(line.as_bytes())
        .await
        .map_err(|e| RunnerError::Transport(format!("write to claude stdin failed: {e}")))?;
    stdin
        .flush()
        .await
        .map_err(|e| RunnerError::Transport(format!("flush claude stdin failed: {e}")))
}

fn is_turn_terminal(ev: &PvEvent) -> bool {
    matches!(
        ev,
        PvEvent::Core(
            AgentEvent::TurnEnded { .. } | AgentEvent::TurnFailed { .. } | AgentEvent::TurnBlocked { .. }
        ) | PvEvent::TurnAborted { .. }
    )
}

/// Collapse a [`PvEvent`] onto core's vocabulary: an abort is a turn that
/// ended with `stop_reason: "aborted"`; the other pv-only variants carry no
/// information a core consumer lacks (`SessionLimitHit` rides a core
/// `TurnBlocked`) and are dropped.
fn pv_to_core(ev: PvEvent) -> Option<AgentEvent> {
    match ev {
        PvEvent::Core(ev) => Some(ev),
        PvEvent::TurnAborted { session_id, text } => Some(AgentEvent::TurnEnded {
            session_id,
            text,
            stop_reason: Some("aborted".into()),
            usage: None,
            content_blocks: None,
        }),
        PvEvent::CompactingStarted { .. } | PvEvent::SessionLimitHit { .. } => None,
    }
}

struct CoreStream(mpsc::Receiver<PvEvent>);

impl Stream for CoreStream {
    type Item = AgentEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<AgentEvent>> {
        loop {
            match self.0.poll_recv(cx) {
                Poll::Ready(Some(ev)) => {
                    if let Some(core) = pv_to_core(ev) {
                        return Poll::Ready(Some(core));
                    }
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[async_trait]
impl Runner for PvRunner {
    async fn start(&self, prelude: Prelude, _session_key: &str) -> Result<SessionId, RunnerError> {
        let session_id = mint_session_id();
        let model = prelude
            .model
            .as_ref()
            .and_then(|m| m.model.clone())
            .unwrap_or_else(|| self.config.default_model.clone());
        let system = prelude.render();
        let mut spec = SpawnSpec::new(&self.config.cli_binary, &self.config.workdir, &model, &session_id);
        spec.reasoning = prelude.reasoning;
        spec.append_system_prompt = Some(&system);
        spec.extended_cache = prelude.cache.ttl == CacheTtl::Extended;
        let mut child = spawn_claude(&spec, self.host.as_ref())
            .map_err(|e| RunnerError::Transport(format!("spawn {} failed: {e}", self.config.cli_binary)))?;
        let stdin = child.stdin.take().ok_or_else(|| RunnerError::Transport("claude stdin not piped".into()))?;
        let stdout = child.stdout.take().ok_or_else(|| RunnerError::Transport("claude stdout not piped".into()))?;
        let reader = Arc::new(Mutex::new(StreamReader::new(session_id.clone(), stdout)));
        self.sessions
            .lock()
            .await
            .insert(session_id.clone(), PvSession { child, stdin, reader });
        Ok(session_id)
    }

    async fn send(&self, session_id: &SessionId, message: String) -> Result<EventStream, RunnerError> {
        let rx = self.send_pv(session_id, message).await?;
        Ok(Box::pin(CoreStream(rx)))
    }

    async fn stop(&self, session_id: &SessionId) -> Result<bool, RunnerError> {
        let Some(s) = self.sessions.lock().await.remove(session_id) else {
            return Ok(false);
        };
        stop_child(session_id, s.child, Some(s.stdin), self.host.as_ref()).await;
        Ok(true)
    }

    fn think_live(&self) -> bool {
        true
    }

    fn kind(&self) -> RunnerKind {
        RunnerKind::AnthropicNative
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::host::BareHost;
    use yah_runner_core::{CacheControl, Prelude};

    /// A stand-in `claude`: ignores its argv, answers each stdin line with one
    /// stream-json turn, and treats an interrupt as an aborted turn.
    const FAKE_CLAUDE: &str = r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *control_request*)
      echo '{"type":"result","subtype":"error_during_execution","is_error":true,"terminal_reason":"aborted_streaming"}' ;;
    *)
      echo '{"type":"system","subtype":"init"}'
      echo '{"type":"stream_event","event":{"type":"message_start"}}'
      echo '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"pong"}}}'
      echo '{"type":"result","subtype":"success","is_error":false,"result":"pong"}' ;;
  esac
done
"#;

    async fn collect(mut s: EventStream) -> Vec<AgentEvent> {
        let mut out = Vec::new();
        while let Some(ev) = std::future::poll_fn(|cx| s.as_mut().poll_next(cx)).await {
            out.push(ev);
        }
        out
    }

    #[tokio::test]
    async fn pv_runner_round_trips_turns_and_stops() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::TempDir::new().unwrap();
        let bin = dir.path().join("claude");
        std::fs::write(&bin, FAKE_CLAUDE).unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();

        let runner = PvRunner::new(
            Arc::new(BareHost),
            PvConfig {
                cli_binary: bin.to_string_lossy().into_owned(),
                workdir: dir.path().to_path_buf(),
                default_model: "claude-sonnet-4-6".into(),
            },
        );
        let prelude = Prelude::empty(CacheControl { key: "k".into(), ttl: CacheTtl::Ephemeral });
        let sid = runner.start(prelude, "key").await.unwrap();

        for _ in 0..2 {
            let events = collect(runner.send(&sid, "ping".into()).await.unwrap()).await;
            assert!(matches!(events.first(), Some(AgentEvent::TurnStarted { .. })), "{events:?}");
            match events.last() {
                Some(AgentEvent::TurnEnded { text, stop_reason, .. }) => {
                    assert_eq!(text, "pong");
                    assert_eq!(stop_reason.as_deref(), Some("success"));
                }
                other => panic!("expected TurnEnded, got {other:?}"),
            }
        }

        // An interrupt line ends the turn as a neutral abort, surfaced to a
        // core consumer as TurnEnded{stop_reason:"aborted"}.
        // The fake answers the interrupt line before the user line that
        // follows it, so this turn's stream is exactly the abort.
        runner.interrupt(&sid, "req-1").await.unwrap();
        let mut rx = runner.send_pv(&sid, "after".into()).await.unwrap();
        let mut raw = Vec::new();
        while let Some(ev) = rx.recv().await {
            raw.push(ev);
        }
        assert!(matches!(raw[..], [PvEvent::TurnAborted { .. }]), "{raw:?}");

        assert!(runner.stop(&sid).await.unwrap());
        assert!(!runner.stop(&sid).await.unwrap());
        assert!(matches!(
            runner.send(&sid, "x".into()).await,
            Err(RunnerError::SessionNotFound(_))
        ));
    }

    #[test]
    fn aborted_turn_collapses_to_turn_ended_aborted() {
        let sid = SessionId::new("session:x");
        let core = pv_to_core(PvEvent::TurnAborted { session_id: sid.clone(), text: "t".into() });
        assert!(matches!(
            core,
            Some(AgentEvent::TurnEnded { stop_reason: Some(ref r), .. }) if r == "aborted"
        ));
        assert!(pv_to_core(PvEvent::CompactingStarted { session_id: sid }).is_none());
    }
}
