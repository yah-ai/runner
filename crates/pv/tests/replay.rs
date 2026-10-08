//! R951-T4: replay a recorded `claude -p` stream-json turn through
//! `StreamReader` and pin the exact core event sequence. No claude binary.

use yah_runner_core::{AgentEvent, SessionId};
use yah_runner_pv::frame::{ReadStep, StreamReader};
use yah_runner_pv::PvEvent;

const FIXTURE: &[u8] = include_bytes!("fixtures/claude_p_tool_turn.jsonl");

#[tokio::test]
async fn replay_fixture_yields_exact_core_sequence() {
    let sid = SessionId::new("session:replay");
    let mut reader = StreamReader::new(sid.clone(), FIXTURE);
    let mut events = Vec::new();
    loop {
        match reader.next().await {
            ReadStep::Frame { events: evs, .. } => events.extend(evs),
            ReadStep::Eof => break,
            ReadStep::ReadError(e) => panic!("read error: {e:?}"),
        }
    }
    assert!(reader.completed_a_turn());
    assert!(!reader.turn_open());
    assert!(reader.exit_event(true).is_none(), "completed turn owes no exit event");

    let core: Vec<AgentEvent> = events.into_iter().filter_map(PvEvent::into_core).collect();
    let kinds: Vec<&str> = core
        .iter()
        .map(|e| match e {
            AgentEvent::TurnStarted { .. } => "TurnStarted",
            AgentEvent::ThinkingDelta { .. } => "ThinkingDelta",
            AgentEvent::MessageDelta { .. } => "MessageDelta",
            AgentEvent::AssistantText { .. } => "AssistantText",
            AgentEvent::ToolCall { .. } => "ToolCall",
            AgentEvent::ToolResult { .. } => "ToolResult",
            AgentEvent::TurnEnded { .. } => "TurnEnded",
            _ => "other",
        })
        .collect();
    assert_eq!(
        kinds,
        [
            "TurnStarted",
            "ThinkingDelta",
            "MessageDelta",
            "MessageDelta",
            "AssistantText",
            "ToolCall",
            "ToolResult",
            "MessageDelta",
            "AssistantText",
            "TurnEnded",
        ]
    );
    assert!(matches!(&core[1], AgentEvent::ThinkingDelta { text, .. } if text == "I should call echo."));
    match &core[5] {
        AgentEvent::ToolCall { tool_call_id, tool_name, args, .. } => {
            assert_eq!(tool_call_id, "toolu_1");
            assert_eq!(tool_name, "mcp__echo__echo");
            assert_eq!(args, &serde_json::json!({ "text": "hi" }));
        }
        other => panic!("{other:?}"),
    }
    match &core[6] {
        AgentEvent::ToolResult { tool_call_id, ok, result, .. } => {
            assert_eq!(tool_call_id, "toolu_1");
            assert!(*ok);
            assert_eq!(result, &serde_json::json!("hi"));
        }
        other => panic!("{other:?}"),
    }
    match &core[9] {
        AgentEvent::TurnEnded { text, stop_reason, usage: Some(u), .. } => {
            assert_eq!(text, "Calling echo.Done.");
            assert_eq!(stop_reason.as_deref(), Some("success"));
            assert_eq!(
                (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens),
                (Some(100), Some(20), Some(50), Some(5))
            );
        }
        other => panic!("{other:?}"),
    }
}
