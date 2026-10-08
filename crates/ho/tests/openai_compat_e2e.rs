//! End-to-end integration test for the OpenAI-compatible runner.
//!
//! Stands up an in-process TCP server that replays canned SSE frames,
//! points the runner at it, and asserts the full event stream the
//! consumer sees. Exercises the wire path no unit test can: real
//! HTTP/1.1 framing, real streamed response bodies through the default reqwest `HttpClient`, real chunk
//! boundaries chopping deltas mid-frame.

use async_trait::async_trait;
use futures::StreamExt;
use yah_runner_core::{
    AgentEvent, CacheControl, CacheTtl, ModelRef, Prelude, Runner, ToolOutcome, ToolRegistry,
    ToolSchema,
};
use yah_runner_ho::{OpenAiCompatConfig, OpenAiCompatRunner, Transport};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use yah_runner_stub::{InferenceStub, Reply, StubMode};

const CLIENT_TIMEOUT: Duration = Duration::from_secs(5);

/// The shared inference stub (`yah-runner-stub`), preloaded with `replies`:
/// request N is answered with reply N, past the queue the stub's default mode
/// answers. Hold the returned handle for the test's life — dropping it stops
/// the server. The real reqwest `HttpClient` talks to a real HTTP/1.1 server,
/// so chunk boundaries and streaming are exercised, not a hand-fed parser.
async fn scripted(replies: Vec<Reply>) -> (String, InferenceStub) {
    let stub = InferenceStub::builder(StubMode::default())
        .replies(replies)
        .spawn()
        .await
        .expect("spawn stub");
    (stub.completions_url(), stub)
}

/// One SSE reply from raw frames (no trailing blank line).
async fn scripted_sse(frames: Vec<String>) -> (String, InferenceStub) {
    scripted(vec![Reply::Sse(frames)]).await
}

/// `responses.len()` sequential SSE replies — one per tool-call iteration.
async fn scripted_sse_sequence(responses: Vec<Vec<String>>) -> (String, InferenceStub) {
    scripted(responses.into_iter().map(Reply::Sse).collect()).await
}

/// Sequential `(status, retry_after, frames)` replies. 2xx are SSE; anything
/// else carries `frames.join("")` as its JSON error body. Read how many
/// requests the server fielded off the returned stub (`stub.served()`).
async fn scripted_status_sequence(
    responses: Vec<(u16, Option<u32>, Vec<String>)>,
) -> (String, InferenceStub) {
    scripted(
        responses
            .into_iter()
            .map(|(status, retry_after, frames)| {
                if (200..300).contains(&status) {
                    Reply::Sse(frames)
                } else {
                    Reply::Error { status, retry_after, body: frames.join("") }
                }
            })
            .collect(),
    )
    .await
}

fn fake_prelude(provider: &str) -> Prelude {
    let mut prelude = Prelude::empty(CacheControl { key: "0".into(), ttl: CacheTtl::Ephemeral });
    prelude.model = Some(ModelRef::new(provider, Some("fake-model".into())));
    prelude
}

#[tokio::test]
async fn streams_deltas_then_turn_ended_with_stop_reason() {
    let (url, _stub) = scripted_sse(vec![
        r#"data: {"choices":[{"delta":{"content":"hello "},"finish_reason":null}]}"#.into(),
        r#"data: {"choices":[{"delta":{"content":"world"},"finish_reason":null}]}"#.into(),
        r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#.into(),
        "data: [DONE]".into(),
    ])
    .await;

    let runner = OpenAiCompatRunner::new(OpenAiCompatConfig {
        provider_label: "openai".into(),
        endpoint: url,
        default_model: "fake-model".into(),
        api_key: Some("dummy".into()),
        max_output_tokens: None,
        max_tool_iterations: None,
        transport: Transport::Sse,
        num_ctx: None,
        cost_guard_tokens: None,
        keep_alive: None,
        reasoning: None,
        quota_ledger: None,
        rate_limit_hint: None,
        session_preamble: None,
        context_window: None,
        prompt_overflow_threshold: None,
    });

    let id = runner
        .start(fake_prelude("openai"), "T01")
        .await
        .expect("start");
    let stream = runner
        .send(&id, "hi".into())
        .await
        .map_err(|e| format!("{e:?}"))
        .expect("send");

    let events = tokio::time::timeout(CLIENT_TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream completes within timeout");

    assert_eq!(
        events.len(),
        5,
        "expected TurnStarted + 2 deltas + AssistantText + TurnEnded, got {events:?}"
    );
    assert!(matches!(events[0], AgentEvent::TurnStarted { .. }));
    let AgentEvent::MessageDelta { ref text, .. } = events[1] else {
        panic!("expected MessageDelta, got {:?}", events[1]);
    };
    assert_eq!(text, "hello ");
    let AgentEvent::MessageDelta { ref text, .. } = events[2] else {
        panic!("expected MessageDelta, got {:?}", events[2]);
    };
    assert_eq!(text, "world");
    let AgentEvent::AssistantText { ref text, .. } = events[3] else {
        panic!("expected AssistantText, got {:?}", events[3]);
    };
    assert_eq!(text, "hello world");
    let AgentEvent::TurnEnded {
        ref text,
        ref stop_reason,
        ..
    } = events[4]
    else {
        panic!("expected TurnEnded, got {:?}", events[4]);
    };
    assert_eq!(text, "hello world");
    assert_eq!(stop_reason.as_deref(), Some("stop"));
}

#[tokio::test]
async fn error_payload_in_stream_yields_error_event_and_terminates() {
    let (url, _stub) = scripted_sse(vec![
        r#"data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}"#.into(),
        r#"data: {"error":{"message":"context length exceeded"}}"#.into(),
        // Server keeps emitting after the error, but the parser must
        // stop forwarding anything past it (the runner returns from the
        // inner stream as soon as it yields Error).
        r#"data: {"choices":[{"delta":{"content":"unreachable"}}]}"#.into(),
    ])
    .await;

    let runner = OpenAiCompatRunner::new(OpenAiCompatConfig {
        provider_label: "openai".into(),
        endpoint: url,
        default_model: "fake-model".into(),
        api_key: Some("dummy".into()),
        max_output_tokens: None,
        max_tool_iterations: None,
        transport: Transport::Sse,
        num_ctx: None,
        cost_guard_tokens: None,
        keep_alive: None,
        reasoning: None,
        quota_ledger: None,
        rate_limit_hint: None,
        session_preamble: None,
        context_window: None,
        prompt_overflow_threshold: None,
    });

    let id = runner
        .start(fake_prelude("openai"), "T01")
        .await
        .expect("start");
    let stream = runner
        .send(&id, "hi".into())
        .await
        .map_err(|e| format!("{e:?}"))
        .expect("send");

    let events = tokio::time::timeout(CLIENT_TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream completes within timeout");

    // Expect: TurnStarted, MessageDelta(partial), TurnFailed (carrying
    // the partial accumulator). No TurnEnded.
    assert_eq!(events.len(), 3, "got {events:?}");
    assert!(matches!(events[0], AgentEvent::TurnStarted { .. }));
    let AgentEvent::MessageDelta {
        text: ref delta_text,
        ..
    } = events[1]
    else {
        panic!("expected MessageDelta, got {:?}", events[1]);
    };
    let AgentEvent::TurnFailed {
        ref message,
        ref text,
        ..
    } = events[2]
    else {
        panic!("expected TurnFailed, got {:?}", events[2]);
    };
    assert!(message.contains("context length"), "{message}");
    // TurnFailed carries the same accumulated text the renderer streamed.
    assert_eq!(
        text, delta_text,
        "TurnFailed.text should mirror accumulated deltas"
    );
}

#[tokio::test]
async fn upstream_4xx_surfaces_as_runner_error_before_stream_starts() {
    // 401 from the upstream means we never get to stream — `send`
    // should return `RunnerError::Transport` synchronously rather than
    // yielding it on the stream.
    let (url, _stub) = scripted(vec![Reply::Error {
        status: 401,
        retry_after: None,
        body: r#"{"error":{"message":"invalid_api_key"}}"#.into(),
    }])
    .await;

    let runner = OpenAiCompatRunner::new(OpenAiCompatConfig {
        provider_label: "openai".into(),
        endpoint: url,
        default_model: "fake-model".into(),
        api_key: Some("bad-key".into()),
        max_output_tokens: None,
        max_tool_iterations: None,
        transport: Transport::Sse,
        num_ctx: None,
        cost_guard_tokens: None,
        keep_alive: None,
        reasoning: None,
        quota_ledger: None,
        rate_limit_hint: None,
        session_preamble: None,
        context_window: None,
        prompt_overflow_threshold: None,
    });

    let id = runner
        .start(fake_prelude("openai"), "T01")
        .await
        .expect("start");
    match runner.send(&id, "hi".into()).await {
        Err(yah_runner_core::RunnerError::Transport(msg)) => {
            assert!(msg.contains("401"), "{msg}");
            assert!(msg.contains("invalid_api_key"), "{msg}");
        }
        Err(other) => panic!("expected Transport error, got {other:?}"),
        Ok(_) => panic!("expected error for 401 response"),
    }
}

#[tokio::test]
async fn first_request_503_then_200_retries_and_streams() {
    // R237-F12: a transient 5xx on the first attempt is silently retried.
    // The user-visible result is identical to a clean 200 — no TurnFailed,
    // no Error event, full TurnEnded with text.
    let (url, stub) = scripted_status_sequence(vec![
        // Attempt 1: 503 Service Unavailable, no Retry-After (so the
        // backoff is the exponential 1s; we override below).
        (503, None, vec![r#"{"error":{"message":"overloaded"}}"#.into()]),
        // Attempt 2: 200 with normal SSE.
        (
            200,
            None,
            vec![
                r#"data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}"#.into(),
                r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#.into(),
                "data: [DONE]".into(),
            ],
        ),
    ])
    .await;

    let runner = OpenAiCompatRunner::new(OpenAiCompatConfig {
        provider_label: "openai".into(),
        endpoint: url,
        default_model: "fake-model".into(),
        api_key: Some("dummy".into()),
        max_output_tokens: None,
        max_tool_iterations: None,
        transport: Transport::Sse,
        num_ctx: None,
        cost_guard_tokens: None,
        keep_alive: None,
        reasoning: None,
        quota_ledger: None,
        rate_limit_hint: None,
        session_preamble: None,
        context_window: None,
        prompt_overflow_threshold: None,
    });

    let id = runner
        .start(fake_prelude("openai"), "T01")
        .await
        .expect("start");

    // Per-attempt sleep is 1s exponential. Give 5s headroom so the retry
    // path completes well inside the timeout.
    let stream = tokio::time::timeout(Duration::from_secs(5), runner.send(&id, "hi".into()))
        .await
        .expect("send resolves inside backoff window")
        .map_err(|e| format!("{e:?}"))
        .expect("send succeeds on retry");

    let events = tokio::time::timeout(CLIENT_TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream completes within timeout");

    assert_eq!(
        stub.served(),
        2,
        "server should have served both attempts (got {})",
        stub.served(),
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::TurnEnded { .. })),
        "expected TurnEnded after successful retry, got {events:?}",
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::MessageDelta { text, .. } if text == "hello")),
        "expected the post-retry delta, got {events:?}",
    );
    // No TurnFailed — the retry succeeded.
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AgentEvent::TurnFailed { .. })),
        "no TurnFailed should be emitted on a successful retry, got {events:?}",
    );
}

#[tokio::test]
async fn first_request_401_returns_runner_error_without_retry() {
    // R237-F12: 401 is permanent. The retry layer must NOT re-issue —
    // a stale credential won't fix itself by hammering the upstream.
    // Asserted via served-counter (1) plus the synchronous Err.
    let (url, stub) = scripted_status_sequence(vec![(
        401,
        None,
        vec![r#"{"error":{"message":"invalid_api_key"}}"#.into()],
    )])
    .await;

    let runner = OpenAiCompatRunner::new(OpenAiCompatConfig {
        provider_label: "openai".into(),
        endpoint: url,
        default_model: "fake-model".into(),
        api_key: Some("bad-key".into()),
        max_output_tokens: None,
        max_tool_iterations: None,
        transport: Transport::Sse,
        num_ctx: None,
        cost_guard_tokens: None,
        keep_alive: None,
        reasoning: None,
        quota_ledger: None,
        rate_limit_hint: None,
        session_preamble: None,
        context_window: None,
        prompt_overflow_threshold: None,
    });

    let id = runner
        .start(fake_prelude("openai"), "T01")
        .await
        .expect("start");
    // Bound on wall time: a retry would sleep ≥1s before re-issuing, so
    // resolving inside 500ms proves no retry happened.
    let result = tokio::time::timeout(Duration::from_millis(500), runner.send(&id, "hi".into()))
        .await
        .expect("send resolves immediately on permanent failure");
    match result {
        Err(yah_runner_core::RunnerError::Transport(msg)) => {
            assert!(msg.contains("401"), "{msg}");
            assert!(msg.contains("invalid_api_key"), "{msg}");
        }
        Err(other) => panic!("expected Transport error, got {other:?}"),
        Ok(_) => panic!("expected error for 401 response"),
    }
    assert_eq!(
        stub.served(),
        1,
        "401 must not trigger a retry — server should see exactly 1 request",
    );
}

#[tokio::test]
async fn first_request_429_with_retry_after_honors_header_then_succeeds() {
    // R237-F12: 429 with Retry-After is retried; the header value drives
    // the backoff (capped at MAX_RETRY_AFTER_SECS so a stuck provider
    // can't stall the session). Verify (a) we retry, (b) we honor the
    // 1-second header (vs. the default 1s exponential — both happen to
    // land at 1s here, but the assertion that we waited ≥1s proves the
    // sleep ran), and (c) the second attempt succeeds.
    let (url, stub) = scripted_status_sequence(vec![
        // Attempt 1: 429 with Retry-After: 1s.
        (
            429,
            Some(1),
            vec![r#"{"error":{"message":"slow down"}}"#.into()],
        ),
        // Attempt 2: 200 with normal SSE.
        (
            200,
            None,
            vec![
                r#"data: {"choices":[{"delta":{"content":"recovered"},"finish_reason":null}]}"#
                    .into(),
                r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#.into(),
                "data: [DONE]".into(),
            ],
        ),
    ])
    .await;

    // R950-F5: the 429 lands in the host-injected ledger, not a global.
    let ledger = Arc::new(std::sync::Mutex::new(yah_runner_core::quota_ledger::QuotaLedger::default()));
    let runner = OpenAiCompatRunner::new(OpenAiCompatConfig {
        provider_label: "openai".into(),
        endpoint: url,
        default_model: "fake-model".into(),
        api_key: Some("dummy".into()),
        max_output_tokens: None,
        max_tool_iterations: None,
        transport: Transport::Sse,
        num_ctx: None,
        cost_guard_tokens: None,
        keep_alive: None,
        reasoning: None,
        quota_ledger: Some(ledger.clone()),
        rate_limit_hint: None,
        session_preamble: None,
        context_window: None,
        prompt_overflow_threshold: None,
    });

    let id = runner
        .start(fake_prelude("openai"), "T01")
        .await
        .expect("start");
    let start = std::time::Instant::now();
    let stream = tokio::time::timeout(Duration::from_secs(5), runner.send(&id, "hi".into()))
        .await
        .expect("send resolves inside backoff window")
        .map_err(|e| format!("{e:?}"))
        .expect("send succeeds on retry");
    let send_elapsed = start.elapsed();

    let events = tokio::time::timeout(CLIENT_TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream completes within timeout");

    assert!(
        send_elapsed >= Duration::from_millis(900),
        "send should have waited Retry-After=1s before retrying, but only waited {send_elapsed:?}",
    );
    assert_eq!(
        stub.served(),
        2,
        "server should have served both attempts",
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::MessageDelta { text, .. } if text == "recovered")),
        "expected the post-retry delta, got {events:?}",
    );
    assert!(
        ledger.lock().unwrap().is_on_cooldown("openai", "fake-model", 60),
        "the injected ledger must have recorded the 429",
    );
}

/// Stub registry exposing one tool that returns whatever the test
/// hands it. Tracks the call count so assertions can verify the
/// runner dispatched.
struct StubRegistry {
    schemas: Vec<ToolSchema>,
    response: Value,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl ToolRegistry for StubRegistry {
    fn schemas(&self) -> Vec<ToolSchema> {
        self.schemas.clone()
    }
    async fn execute(&self, _name: &str, _args: Value) -> ToolOutcome {
        self.calls.fetch_add(1, Ordering::SeqCst);
        ToolOutcome::ok(self.response.clone())
    }
}

#[tokio::test]
async fn tool_call_loop_dispatches_then_re_issues_for_final_text() {
    // Round-trip 1: provider asks for `read_file({ path: "Cargo.toml" })`.
    // Round-trip 2: provider returns plain text with finish_reason "stop".
    // Expected event order:
    //   TurnStarted → ToolCall → ToolResult → MessageDelta → TurnEnded
    let (url, _stub) = scripted_sse_sequence(vec![
        // First response: streamed tool_call across two chunks + a
        // terminal finish_reason="tool_calls" frame.
        vec![
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":"}}]},"finish_reason":null}]}"#.into(),
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Cargo.toml\"}"}}]},"finish_reason":null}]}"#.into(),
            r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#.into(),
            "data: [DONE]".into(),
        ],
        // Second response: model reasons over the tool result and ends.
        vec![
            r#"data: {"choices":[{"delta":{"content":"It's the workspace Cargo.toml."},"finish_reason":null}]}"#.into(),
            r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#.into(),
            "data: [DONE]".into(),
        ],
    ])
    .await;

    let calls = Arc::new(AtomicUsize::new(0));
    let registry: Arc<dyn ToolRegistry> = Arc::new(StubRegistry {
        schemas: vec![ToolSchema {
            name: "read_file".into(),
            description: "Read a file.".into(),
            input_schema: json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }),
        }],
        response: json!({ "content": "[package]\nname = \"x\"\n", "bytes": 24 }),
        calls: Arc::clone(&calls),
    });

    let runner = OpenAiCompatRunner::with_tools(
        OpenAiCompatConfig {
            provider_label: "openai".into(),
            endpoint: url,
            default_model: "fake-model".into(),
            api_key: Some("dummy".into()),
            max_output_tokens: None,
        max_tool_iterations: None,
        transport: Transport::Sse,
        num_ctx: None,
        cost_guard_tokens: None,
        keep_alive: None,
        reasoning: None,
        quota_ledger: None,
        rate_limit_hint: None,
        session_preamble: None,
        context_window: None,
        prompt_overflow_threshold: None,
        },
        registry,
    );

    let id = runner
        .start(fake_prelude("openai"), "T01")
        .await
        .expect("start");
    let stream = runner
        .send(&id, "what's in Cargo.toml?".into())
        .await
        .map_err(|e| format!("{e:?}"))
        .expect("send");

    let events = tokio::time::timeout(CLIENT_TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream completes within timeout");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "registry must dispatch once"
    );

    // Order: TurnStarted, ToolCall, ToolResult, MessageDelta, AssistantText, TurnEnded.
    assert_eq!(events.len(), 6, "got {events:?}");
    assert!(matches!(events[0], AgentEvent::TurnStarted { .. }));
    let AgentEvent::ToolCall {
        ref tool_name,
        ref args,
        ref tool_call_id,
        ..
    } = events[1]
    else {
        panic!("expected ToolCall, got {:?}", events[1]);
    };
    assert_eq!(tool_name, "read_file");
    assert_eq!(args["path"], "Cargo.toml");
    assert_eq!(tool_call_id, "call_1");
    let AgentEvent::ToolResult {
        ok,
        ref result,
        ref tool_call_id,
        ..
    } = events[2]
    else {
        panic!("expected ToolResult, got {:?}", events[2]);
    };
    assert!(ok);
    assert_eq!(result["bytes"], 24);
    assert_eq!(tool_call_id, "call_1");
    let AgentEvent::MessageDelta { ref text, .. } = events[3] else {
        panic!("expected MessageDelta, got {:?}", events[3]);
    };
    assert_eq!(text, "It's the workspace Cargo.toml.");
    let AgentEvent::AssistantText { ref text, .. } = events[4] else {
        panic!("expected AssistantText, got {:?}", events[4]);
    };
    assert_eq!(text, "It's the workspace Cargo.toml.");
    let AgentEvent::TurnEnded {
        ref text,
        ref stop_reason,
        ..
    } = events[5]
    else {
        panic!("expected TurnEnded, got {:?}", events[5]);
    };
    assert_eq!(text, "It's the workspace Cargo.toml.");
    assert_eq!(stop_reason.as_deref(), Some("stop"));
}

#[tokio::test]
async fn exploration_budget_exhausted_soft_stops_with_turn_ended() {
    // When the runner hits max_tool_iterations on a genuine wide-research
    // turn (different tool args each round, so the loop detector does NOT
    // fire), it must NOT emit TurnFailed — the UI renders that as a red
    // crash banner. Instead it synthesizes a closing assistant message
    // and ends the turn cleanly with stop_reason="exploration_budget".
    // The user can reply "continue" to keep going. R121-F2 soft-stop.
    let (url, _stub) = scripted_sse_sequence(vec![
        // Iter 0: tool_call read_file("a.rs")
        vec![
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.rs\"}"}}]},"finish_reason":null}]}"#.into(),
            r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#.into(),
            "data: [DONE]".into(),
        ],
        // Iter 1: tool_call read_file("b.rs") — different args so the
        // loop detector does NOT fire, only the budget cap does.
        vec![
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_2","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"b.rs\"}"}}]},"finish_reason":null}]}"#.into(),
            r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#.into(),
            "data: [DONE]".into(),
        ],
    ])
    .await;

    let calls = Arc::new(AtomicUsize::new(0));
    let registry: Arc<dyn ToolRegistry> = Arc::new(StubRegistry {
        schemas: vec![ToolSchema {
            name: "read_file".into(),
            description: "Read a file.".into(),
            input_schema: json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }),
        }],
        response: json!({ "content": "ok", "bytes": 2 }),
        calls: Arc::clone(&calls),
    });

    let runner = OpenAiCompatRunner::with_tools(
        OpenAiCompatConfig {
            provider_label: "openai".into(),
            endpoint: url,
            default_model: "fake-model".into(),
            api_key: Some("dummy".into()),
            max_output_tokens: None,
            max_tool_iterations: Some(2),
            transport: Transport::Sse,
            num_ctx: None,
            cost_guard_tokens: None,
            keep_alive: None,
            reasoning: None,
        quota_ledger: None,
        rate_limit_hint: None,
        session_preamble: None,
            context_window: None,
            prompt_overflow_threshold: None,
        },
        registry,
    );

    let id = runner
        .start(fake_prelude("openai"), "T01")
        .await
        .expect("start");
    let stream = runner
        .send(&id, "explore the codebase".into())
        .await
        .map_err(|e| format!("{e:?}"))
        .expect("send");

    let events = tokio::time::timeout(CLIENT_TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream completes within timeout");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "registry must dispatch on both iterations before the cap fires"
    );

    // No TurnFailed event: a budget cap is a soft policy stop, not a crash.
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AgentEvent::TurnFailed { .. })),
        "budget exhaustion must not emit TurnFailed; got {events:?}",
    );

    // Last event is TurnEnded with stop_reason "exploration_budget" and
    // text carrying the synthesized closing message.
    let last = events.last().expect("at least one event");
    let AgentEvent::TurnEnded {
        ref text,
        ref stop_reason,
        ..
    } = last
    else {
        panic!("expected TurnEnded as terminal event, got {last:?}");
    };
    assert_eq!(stop_reason.as_deref(), Some("exploration_budget"));
    assert!(
        text.contains("tool-call budget"),
        "TurnEnded.text must carry the synthesized closing message; got {text:?}",
    );

    // The synthesized message is also emitted as an AssistantText so the
    // chat renderer materializes the bubble (parity with normal text turns).
    let assistant_texts: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::AssistantText { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        assistant_texts.iter().any(|t| t.contains("tool-call budget")),
        "expected an AssistantText carrying the budget notice; got {assistant_texts:?}",
    );
}

#[tokio::test]
async fn tool_call_with_invalid_arguments_surfaces_invalid_args_outcome() {
    // Model produces malformed JSON for `arguments`. The runner must
    // emit ToolCall + a synthetic ToolResult with kind: invalid_arguments
    // (without dispatching to the registry) so the model can correct on
    // the next turn.
    let (url, _stub) = scripted_sse_sequence(vec![
        vec![
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_bad","type":"function","function":{"name":"read_file","arguments":"{ unclosed"}}]}}]}"#.into(),
            r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#.into(),
            "data: [DONE]".into(),
        ],
        vec![
            r#"data: {"choices":[{"delta":{"content":"sorry, retrying"},"finish_reason":null}]}"#.into(),
            r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#.into(),
            "data: [DONE]".into(),
        ],
    ])
    .await;

    let calls = Arc::new(AtomicUsize::new(0));
    let registry: Arc<dyn ToolRegistry> = Arc::new(StubRegistry {
        schemas: vec![ToolSchema {
            name: "read_file".into(),
            description: "Read.".into(),
            input_schema: json!({"type": "object", "properties": {}}),
        }],
        response: json!({}),
        calls: Arc::clone(&calls),
    });

    let runner = OpenAiCompatRunner::with_tools(
        OpenAiCompatConfig {
            provider_label: "openai".into(),
            endpoint: url,
            default_model: "fake-model".into(),
            api_key: Some("dummy".into()),
            max_output_tokens: None,
        max_tool_iterations: None,
        transport: Transport::Sse,
        num_ctx: None,
        cost_guard_tokens: None,
        keep_alive: None,
        reasoning: None,
        quota_ledger: None,
        rate_limit_hint: None,
        session_preamble: None,
        context_window: None,
        prompt_overflow_threshold: None,
        },
        registry,
    );

    let id = runner
        .start(fake_prelude("openai"), "T01")
        .await
        .expect("start");
    let stream = runner
        .send(&id, "go".into())
        .await
        .map_err(|e| format!("{e:?}"))
        .expect("send");
    let events = tokio::time::timeout(CLIENT_TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream completes within timeout");

    // Registry must NOT have been called — args were malformed, so the
    // runner short-circuits with an invalid_arguments result.
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let result_event = events
        .iter()
        .find(|e| matches!(e, AgentEvent::ToolResult { .. }))
        .expect("ToolResult expected");
    let AgentEvent::ToolResult { ok, ref result, .. } = result_event else {
        unreachable!();
    };
    assert!(!ok);
    assert_eq!(result["kind"], "invalid_arguments");
}

/// Stub registry that always returns `ok: false` for any tool.
struct AlwaysFailRegistry {
    schemas: Vec<ToolSchema>,
    error_result: Value,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl ToolRegistry for AlwaysFailRegistry {
    fn schemas(&self) -> Vec<ToolSchema> {
        self.schemas.clone()
    }
    async fn execute(&self, _name: &str, _args: Value) -> ToolOutcome {
        self.calls.fetch_add(1, Ordering::SeqCst);
        ToolOutcome {
            ok: false,
            result: self.error_result.clone(),
        }
    }
}

#[tokio::test]
async fn tool_stuck_emitted_on_identical_failed_retry() {
    // Round-trip 1: model calls `read_arch_doc({ path: "xlb-net" })` → fails.
    // Round-trip 2: model calls the same tool+args again → ToolStuck emitted, stream ends.
    let (url, _stub) = scripted_sse_sequence(vec![
        // First request: model proposes read_arch_doc.
        vec![
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_arch_doc","arguments":"{\"path\":\"xlb-net\"}"}}]},"finish_reason":null}]}"#.into(),
            r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#.into(),
            "data: [DONE]".into(),
        ],
        // Second request: model retries the identical call.
        vec![
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_2","type":"function","function":{"name":"read_arch_doc","arguments":"{\"path\":\"xlb-net\"}"}}]},"finish_reason":null}]}"#.into(),
            r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#.into(),
            "data: [DONE]".into(),
        ],
    ])
    .await;

    let calls = Arc::new(AtomicUsize::new(0));
    let registry: Arc<dyn ToolRegistry> = Arc::new(AlwaysFailRegistry {
        schemas: vec![ToolSchema {
            name: "read_arch_doc".into(),
            description: "Read an arch doc.".into(),
            input_schema: json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }),
        }],
        error_result: json!({ "ok": false, "error": "arch doc not found" }),
        calls: Arc::clone(&calls),
    });

    let runner = OpenAiCompatRunner::with_tools(
        OpenAiCompatConfig {
            provider_label: "openai".into(),
            endpoint: url,
            default_model: "fake-model".into(),
            api_key: Some("dummy".into()),
            max_output_tokens: None,
        max_tool_iterations: None,
        transport: Transport::Sse,
        num_ctx: None,
        cost_guard_tokens: None,
        keep_alive: None,
        reasoning: None,
        quota_ledger: None,
        rate_limit_hint: None,
        session_preamble: None,
        context_window: None,
        prompt_overflow_threshold: None,
        },
        registry,
    );

    let id = runner
        .start(fake_prelude("openai"), "T01")
        .await
        .expect("start");
    let stream = runner
        .send(&id, "read the xlb-net doc".into())
        .await
        .map_err(|e| format!("{e:?}"))
        .expect("send");

    let events = tokio::time::timeout(CLIENT_TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream completes within timeout");

    // Registry must have been called exactly twice (once per round-trip).
    assert_eq!(calls.load(Ordering::SeqCst), 2, "registry called twice");

    // Verify ToolStuck is emitted with the right tool name and args.
    let stuck = events
        .iter()
        .find(|e| matches!(e, AgentEvent::ToolStuck { .. }))
        .expect("ToolStuck expected after identical retry");
    let AgentEvent::ToolStuck {
        ref tool_name,
        ref args,
        ..
    } = stuck
    else {
        unreachable!();
    };
    assert_eq!(tool_name, "read_arch_doc");
    assert_eq!(args["path"], "xlb-net");

    // No TurnEnded — stream terminates on ToolStuck.
    assert!(
        !events.iter().any(|e| matches!(e, AgentEvent::TurnEnded { .. })),
        "TurnEnded must not follow ToolStuck"
    );
}

/// Live HOk validation: drive a real `ollama serve` round-trip end-to-end.
///
/// `#[ignore]` because it depends on a process the dev has to start
/// themselves; CI doesn't bring one up. Run as:
///
/// ```bash
/// ollama serve &  # if not already running
/// cargo test -p yah-runner --test openai_compat_e2e -- --ignored ollama_local_round_trip
/// ```
///
/// Asserts the contract from R028-F3's @yah:next: a Prelude → start →
/// send cycle yields TurnStarted, at least one MessageDelta, and
/// TurnEnded with a stop reason.
#[tokio::test]
#[ignore = "requires `ollama serve` on localhost:11434 with qwen2.5-coder:1.5b pulled"]
async fn ollama_local_round_trip() {
    // Quick liveness probe — if ollama isn't up, skip with a friendly
    // hint rather than a confusing connection-refused panic.
    let probe = tokio::time::timeout(
        Duration::from_secs(2),
        tokio::net::TcpStream::connect("127.0.0.1:11434"),
    )
    .await;
    if !matches!(probe, Ok(Ok(_))) {
        eprintln!("ollama not reachable on :11434 — start `ollama serve` first");
        return;
    }

    let mut prelude = Prelude::empty(CacheControl {
        key: "live-ollama-smoke".into(),
        ttl: CacheTtl::Ephemeral,
    });
    prelude.model = Some(ModelRef::new("ollama", Some("qwen2.5-coder:1.5b".into())));

    let runner = OpenAiCompatRunner::new(OpenAiCompatConfig::ollama_local());

    let id = runner.start(prelude, "T01").await.expect("start");
    let stream = runner
        .send(&id, "Reply with the single word: pong".into())
        .await
        .map_err(|e| format!("{e:?}"))
        .expect("send");

    // Generous timeout — first request after model warmup can be slow.
    let events = tokio::time::timeout(Duration::from_secs(60), stream.collect::<Vec<_>>())
        .await
        .expect("stream completes within 60s");

    assert!(
        matches!(events.first(), Some(AgentEvent::TurnStarted { .. })),
        "expected TurnStarted first, got {events:?}"
    );

    let delta_count = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::MessageDelta { .. }))
        .count();
    assert!(delta_count >= 1, "expected ≥1 MessageDelta, got {events:?}");

    let last = events.last().expect("non-empty event stream");
    let AgentEvent::TurnEnded {
        ref text,
        ref stop_reason,
        ..
    } = last
    else {
        panic!("expected TurnEnded last, got {last:?} — full stream {events:?}");
    };
    assert!(!text.is_empty(), "TurnEnded should carry accumulated text");
    assert!(
        stop_reason.is_some(),
        "TurnEnded should carry a stop reason (got None) — full stream {events:?}"
    );
}

#[tokio::test]
async fn usage_chunk_propagates_to_turn_ended() {
    // OpenAI with stream_options.include_usage=true sends a final
    // chunk carrying token counts. The runner should plumb those
    // through to TurnEnded.usage so the renderer can populate gauges.
    let (url, _stub) = scripted_sse(vec![
        r#"data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}"#.into(),
        r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#.into(),
        r#"data: {"choices":[],"usage":{"prompt_tokens":42,"completion_tokens":17,"prompt_tokens_details":{"cached_tokens":10}}}"#
            .into(),
        "data: [DONE]".into(),
    ])
    .await;

    let runner = OpenAiCompatRunner::new(OpenAiCompatConfig {
        provider_label: "openai".into(),
        endpoint: url,
        default_model: "fake-model".into(),
        api_key: Some("dummy".into()),
        max_output_tokens: None,
        max_tool_iterations: None,
        transport: Transport::Sse,
        num_ctx: None,
        cost_guard_tokens: None,
        keep_alive: None,
        reasoning: None,
        quota_ledger: None,
        rate_limit_hint: None,
        session_preamble: None,
        context_window: None,
        prompt_overflow_threshold: None,
    });
    let id = runner
        .start(fake_prelude("openai"), "T01")
        .await
        .expect("start");
    let stream = runner
        .send(&id, "hi".into())
        .await
        .map_err(|e| format!("{e:?}"))
        .expect("send");
    let events = tokio::time::timeout(CLIENT_TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream completes within timeout");

    let last = events.last().expect("non-empty event stream");
    let AgentEvent::TurnEnded { ref usage, .. } = last else {
        panic!("expected TurnEnded last, got {last:?}");
    };
    let usage = usage.as_ref().expect("usage attached to TurnEnded");
    assert_eq!(usage.input_tokens, Some(42));
    assert_eq!(usage.output_tokens, Some(17));
    assert_eq!(usage.cache_read_tokens, Some(10));
    assert_eq!(usage.cache_write_tokens, None);
    assert_eq!(usage.thinking_tokens, None);
}

#[tokio::test]
async fn turn_ended_usage_none_when_provider_omits_usage_chunk() {
    // Local Ollama deployments today ignore stream_options.include_usage.
    // The runner must still produce a clean TurnEnded — usage just stays
    // None rather than being a zero-filled struct that would mislead the
    // renderer's "is this real" check.
    let (url, _stub) = scripted_sse(vec![
        r#"data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#.into(),
        r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#.into(),
        "data: [DONE]".into(),
    ])
    .await;

    let runner = OpenAiCompatRunner::new(OpenAiCompatConfig {
        provider_label: "openai".into(),
        endpoint: url,
        default_model: "fake-model".into(),
        api_key: Some("dummy".into()),
        max_output_tokens: None,
        max_tool_iterations: None,
        transport: Transport::Sse,
        num_ctx: None,
        cost_guard_tokens: None,
        keep_alive: None,
        reasoning: None,
        quota_ledger: None,
        rate_limit_hint: None,
        session_preamble: None,
        context_window: None,
        prompt_overflow_threshold: None,
    });
    let id = runner
        .start(fake_prelude("openai"), "T01")
        .await
        .expect("start");
    let stream = runner.send(&id, "hi".into()).await.expect("send");
    let events = tokio::time::timeout(CLIENT_TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream completes within timeout");

    let last = events.last().expect("non-empty event stream");
    let AgentEvent::TurnEnded { ref usage, .. } = last else {
        panic!("expected TurnEnded last, got {last:?}");
    };
    assert!(
        usage.is_none(),
        "expected None usage when provider omits the chunk, got {usage:?}"
    );
}

/// R923-F11 (moved here by R923-T13): the runner's deserialize side accepts
/// what `yah_runner_stub::openai_wire` serializes. A script turn exercising
/// every frame kind — content, a split tool call, both finish reasons, usage
/// and `[DONE]` — through the real runner's public `Runner` surface.
#[tokio::test]
async fn runner_parses_every_frame_the_stub_emits() {
    const ROLE: &str = "script:roundtrip";
    let script = yah_runner_stub::Script::from_value(json!({
        "script": "wire-round-trip",
        "roles": {ROLE: [
            {"id": "R1", "when": {"turn_index": 0},
             "emit": [{"text": "Reading."},
                      {"tool_use": {"name": "read_file", "input": {"path": "Cargo.toml"}}}],
             "stop": "tool_use"},
            {"id": "R2", "when": {"after_tool_result": "read_file"},
             "emit": [{"text": "Got $TOOL_RESULT", "chunk_tokens": 1}]}
        ]}
    }))
    .expect("valid script");
    let stub = InferenceStub::spawn_with(StubMode::Script(script)).await.unwrap();

    let registry: Arc<dyn ToolRegistry> = Arc::new(StubRegistry {
        schemas: vec![ToolSchema {
            name: "read_file".into(),
            description: "Read a file.".into(),
            input_schema: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        }],
        response: json!({"echo": "Cargo.toml"}),
        calls: Arc::new(AtomicUsize::new(0)),
    });
    let runner = OpenAiCompatRunner::with_tools(
        OpenAiCompatConfig {
            provider_label: "openai".into(),
            endpoint: stub.completions_url(),
            default_model: ROLE.into(),
            api_key: Some("dummy".into()),
            max_output_tokens: None,
            max_tool_iterations: None,
            transport: Transport::Sse,
            num_ctx: None,
            cost_guard_tokens: None,
            keep_alive: None,
            reasoning: None,
            quota_ledger: None,
            rate_limit_hint: None,
            session_preamble: None,
            context_window: None,
            prompt_overflow_threshold: None,
        },
        registry,
    );
    let mut prelude = fake_prelude("openai");
    prelude.model = Some(ModelRef::new("openai", Some(ROLE.into())));
    let id = runner.start(prelude, "F11").await.expect("start");
    let stream = runner.send(&id, "go".into()).await.map_err(|e| format!("{e:?}")).unwrap();
    let events = tokio::time::timeout(CLIENT_TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("turn completes");

    let call = events
        .iter()
        .find_map(|e| match e {
            AgentEvent::ToolCall { tool_name, args, tool_call_id, .. } => {
                Some((tool_name.clone(), args.clone(), tool_call_id.clone()))
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("no ToolCall in {events:?}"));
    assert_eq!(call, ("read_file".into(), json!({"path": "Cargo.toml"}), "call_R1_0".into()));
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::ToolResult { ok: true, .. })),
        "tool result dispatched: {events:?}"
    );
    let Some(AgentEvent::TurnEnded { text, stop_reason, usage, .. }) = events.last() else {
        panic!("expected TurnEnded last, got {events:?}");
    };
    assert!(text.contains("Got") && text.contains("Cargo.toml"), "final text: {text:?}");
    assert_eq!(stop_reason.as_deref(), Some("stop"));
    let usage = usage.as_ref().expect("usage chunk parsed");
    // The runner sums usage across the turn's two round-trips (10 each).
    assert_eq!(usage.input_tokens, Some(20));
    assert!(usage.output_tokens.is_some_and(|n| n > 0));
    assert_eq!(stub.served(), 2, "one request per round-trip");
}
