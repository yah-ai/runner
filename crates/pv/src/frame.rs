//! claude stream-json framing: the stdin lines pv writes and the stdout line
//! loop that feeds [`translate_frame`].
//!
//! Part of R951 — copied from `app/yah/desktop/src/agent_process.rs`
//! (`encode_user_line`, `extract_user_frame_text`, `encode_interrupt_line`,
//! and the line loop of `run_reader`). The reader does no I/O beyond stdout:
//! the host decides what to persist, emit and do on each yielded frame.

use crate::event::PvEvent;
use crate::translate::{reader_exit_event, translate_frame};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader, Lines};
use yah_runner_core::{AgentEvent, SessionId};

/// Encode one user turn as a stream-json line. Trailing newline is
/// the frame terminator claude expects on `--input-format stream-json`.
pub fn encode_user_line(text: &str) -> String {
    let payload = serde_json::json!({
        "type": "user",
        "message": {
            "role": "user",
            "content": [{ "type": "text", "text": text }]
        },
        "parent_tool_use_id": null,
    });
    let mut line = payload.to_string();
    line.push('\n');
    line
}

/// Encode a `control_request:interrupt` line for the claude stream-json
/// input protocol. Sent over stdin to soft-cancel the in-flight turn
/// without killing the subprocess; claude ends the turn with
/// `terminal_reason: aborted_streaming`, which [`translate_frame`] maps to
/// [`PvEvent::TurnAborted`]. `request_id` round-trips on the matching
/// `control_response` frame.
pub fn encode_interrupt_line(request_id: &str) -> String {
    let payload = serde_json::json!({
        "type": "control_request",
        "request_id": request_id,
        "request": { "subtype": "interrupt" },
    });
    let mut s = payload.to_string();
    s.push('\n');
    s
}

/// Plain-text content of a `type:"user"` stream frame. A host uses it to
/// recognise a turn it did not write itself (claude-cli's native
/// cross-session `SendMessage` rail) rather than dropping it as an echo.
/// `None` for a tool_result-only frame (no `text` blocks) or a frame with no
/// message content at all.
pub fn extract_user_frame_text(frame: &serde_json::Value) -> Option<String> {
    let content = frame.get("message")?.get("content")?;
    if let Some(s) = content.as_str() {
        return (!s.is_empty()).then(|| s.to_string());
    }
    let blocks: Vec<&str> = content
        .as_array()?
        .iter()
        .filter(|blk| blk.get("type").and_then(|t| t.as_str()) == Some("text"))
        .filter_map(|blk| blk.get("text").and_then(|t| t.as_str()))
        .collect();
    (!blocks.is_empty()).then(|| blocks.join(""))
}

/// True for `{"type":"system","subtype":"compact_boundary"}` — the frame a
/// host waiting on a `/compact` it wrote should be signalled by.
pub fn is_compact_boundary(frame: &serde_json::Value) -> bool {
    frame
        .get("type")
        .and_then(|v| v.as_str())
        .zip(frame.get("subtype").and_then(|v| v.as_str()))
        == Some(("system", "compact_boundary"))
}

/// One step of [`StreamReader::next`].
#[derive(Debug)]
pub enum ReadStep {
    /// One parsed stdout frame and what it translated to. `frame` is kept so
    /// the host can run its own checks on it ([`extract_user_frame_text`],
    /// [`is_compact_boundary`]).
    Frame {
        frame: serde_json::Value,
        events: Vec<PvEvent>,
    },
    /// stdout closed — the subprocess exited. Call
    /// [`StreamReader::exit_event`] for the terminal event, if one is owed.
    Eof,
    /// Reading stdout failed; the reader is finished.
    ReadError(AgentEvent),
}

/// The stdout half of a claude `-p` session: reads stream-json lines and
/// carries the translator's cursors across them.
pub struct StreamReader<R> {
    session_id: SessionId,
    lines: Lines<BufReader<R>>,
    turn_open: bool,
    completed_a_turn: bool,
    accumulated: String,
    accumulated_thinking: String,
    peak_ctx: Option<u32>,
}

impl<R: AsyncRead + Unpin> StreamReader<R> {
    pub fn new(session_id: SessionId, stdout: R) -> Self {
        Self {
            session_id,
            lines: BufReader::new(stdout).lines(),
            turn_open: false,
            completed_a_turn: false,
            accumulated: String::new(),
            accumulated_thinking: String::new(),
            peak_ctx: None,
        }
    }

    /// Read until the next JSON frame (blank and non-JSON lines are skipped),
    /// EOF, or a read error.
    pub async fn next(&mut self) -> ReadStep {
        loop {
            match self.lines.next_line().await {
                Ok(Some(line)) => {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let Ok(frame) = serde_json::from_str::<serde_json::Value>(&line) else {
                        tracing::warn!(line = %line, "claude stream-json: non-JSON line; skipping");
                        continue;
                    };
                    let events = translate_frame(
                        &frame,
                        &self.session_id,
                        &mut self.turn_open,
                        &mut self.accumulated,
                        &mut self.accumulated_thinking,
                        &mut self.peak_ctx,
                    );
                    if events
                        .iter()
                        .any(|e| matches!(e, PvEvent::Core(AgentEvent::TurnEnded { .. })))
                    {
                        self.completed_a_turn = true;
                    }
                    return ReadStep::Frame { frame, events };
                }
                Ok(None) => return ReadStep::Eof,
                Err(e) => {
                    tracing::warn!(error = %e, "claude stdout read failed");
                    return ReadStep::ReadError(AgentEvent::Error {
                        session_id: self.session_id.clone(),
                        message: format!("claude stdout read failed: {e}"),
                        http_status: None,
                        raw_body: None,
                        error_type: None,
                    });
                }
            }
        }
    }

    pub fn turn_open(&self) -> bool {
        self.turn_open
    }

    /// Whether any turn has ended cleanly on this reader.
    pub fn completed_a_turn(&self) -> bool {
        self.completed_a_turn
    }

    /// Close the open turn without a `result` frame (e.g. the host is tearing
    /// the session down on a session limit) and hand back its accumulated
    /// text.
    pub fn abandon_turn(&mut self) -> String {
        self.turn_open = false;
        std::mem::take(&mut self.accumulated)
    }

    /// The terminal event owed when the loop ends. `child_alive` is whether
    /// the host still held a live child (i.e. nobody meant to stop it); see
    /// [`reader_exit_event`].
    pub fn exit_event(&mut self, child_alive: bool) -> Option<AgentEvent> {
        reader_exit_event(
            &self.session_id,
            self.turn_open,
            child_alive && !self.completed_a_turn,
            std::mem::take(&mut self.accumulated),
        )
    }
}

#[cfg(test)]
mod tests {
    //! Part of R951 — copied from the desktop `agent_process.rs` tests
    //! (R951-F3 deletes the originals), plus the reader loop.
    use super::*;
    use serde_json::json;

    fn user_frame(text: &str) -> serde_json::Value {
        serde_json::json!({
            "type": "user",
            "message": { "role": "user", "content": [{ "type": "text", "text": text }] },
        })
    }

    /// The interrupt stdin line must be exactly the shape claude-cli's
    /// stream-json input mode expects: `type:"control_request"`, a
    /// non-empty `request_id`, and `request:{subtype:"interrupt"}`. Drift
    /// here is silent — claude won't ack, ESC stays pressed, and the user
    /// thinks pause is broken again.
    #[test]
    fn encode_interrupt_line_matches_control_request_shape() {
        let line = encode_interrupt_line("yah-int-42");
        assert!(
            line.ends_with('\n'),
            "stdin line must be newline-terminated"
        );
        let parsed: serde_json::Value =
            serde_json::from_str(line.trim_end()).expect("interrupt line must be valid JSON");
        assert_eq!(
            parsed.get("type").and_then(|v| v.as_str()),
            Some("control_request")
        );
        assert_eq!(
            parsed.get("request_id").and_then(|v| v.as_str()),
            Some("yah-int-42")
        );
        assert_eq!(
            parsed.pointer("/request/subtype").and_then(|v| v.as_str()),
            Some("interrupt")
        );
    }

    #[test]
    fn encode_user_line_emits_stream_json_user_envelope() {
        let line = encode_user_line("hello world");
        assert!(line.ends_with('\n'), "frame must terminate with newline");
        let parsed: serde_json::Value =
            serde_json::from_str(line.trim_end()).expect("user line is valid JSON");
        assert_eq!(parsed["type"], "user");
        assert_eq!(parsed["message"]["role"], "user");
        assert_eq!(parsed["message"]["content"][0]["type"], "text");
        assert_eq!(parsed["message"]["content"][0]["text"], "hello world");
    }

    #[test]
    fn extract_user_frame_text_reads_plain_text_block() {
        assert_eq!(
            extract_user_frame_text(&user_frame("hello")),
            Some("hello".to_string())
        );
    }

    #[test]
    fn extract_user_frame_text_none_for_tool_result_only_frame() {
        let frame = serde_json::json!({
            "type": "user",
            "message": { "role": "user", "content": [{ "type": "tool_result", "content": "ok" }] },
        });
        assert_eq!(extract_user_frame_text(&frame), None);
    }

    #[tokio::test]
    async fn stream_reader_skips_junk_and_owes_nothing_after_a_clean_turn() {
        let stdout: &[u8] = b"\n\
not json\n\
{\"type\":\"system\",\"subtype\":\"init\"}\n\
{\"type\":\"stream_event\",\"event\":{\"type\":\"message_start\"}}\n\
{\"type\":\"result\",\"subtype\":\"success\",\"is_error\":false,\"result\":\"ok\"}\n";
        let mut r = StreamReader::new(SessionId::new("session:deadbeef"), stdout);
        let ReadStep::Frame { events, .. } = r.next().await else { panic!("init frame") };
        assert!(events.is_empty());
        let ReadStep::Frame { events, .. } = r.next().await else { panic!("message_start") };
        assert!(matches!(events[..], [PvEvent::Core(AgentEvent::TurnStarted { .. })]));
        assert!(r.turn_open());
        let ReadStep::Frame { events, .. } = r.next().await else { panic!("result") };
        assert!(matches!(events[..], [PvEvent::Core(AgentEvent::TurnEnded { .. })]));
        assert!(r.completed_a_turn());
        assert!(matches!(r.next().await, ReadStep::Eof));
        assert!(r.exit_event(true).is_none(), "a returned child owes no terminal event");
    }

    #[tokio::test]
    async fn stream_reader_owes_a_failure_when_stdout_closes_mid_turn() {
        let stdout: &[u8] = b"{\"type\":\"stream_event\",\"event\":{\"type\":\"message_start\"}}\n\
{\"type\":\"stream_event\",\"event\":{\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"half\"}}}\n";
        let mut r = StreamReader::new(SessionId::new("session:deadbeef"), stdout);
        while let ReadStep::Frame { .. } = r.next().await {}
        match r.exit_event(true) {
            Some(AgentEvent::TurnFailed { text, .. }) => assert_eq!(text, "half"),
            other => panic!("expected TurnFailed, got {other:?}"),
        }
    }

    #[test]
    fn is_compact_boundary_matches_only_the_system_subtype() {
        assert!(is_compact_boundary(&json!({"type": "system", "subtype": "compact_boundary"})));
        assert!(!is_compact_boundary(&json!({"type": "system", "subtype": "init"})));
        assert!(!is_compact_boundary(&json!({"type": "result", "subtype": "compact_boundary"})));
    }
}
