//! [`AnthropicRunner`]: the Anthropic cell as a core [`Runner`].
//!
//! Each session lives in memory here and is its own [`HaHost`]: the system
//! prompt is rendered once from the prelude at `start`
//! ([`SystemPrompt::from_prelude`]), history grows by a user and an assistant
//! message per turn, and events reach the caller only through the stream
//! `send` returns. A host that keeps its own session records (yah's camp)
//! calls [`run_anthropic_turn`] with its own [`HaHost`] instead.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use yah_runner_core::{
    mint_session_id, AgentEvent, EventStream, HttpClient, Message, MessageContent, Prelude,
    ReasoningBudget, Role, Runner, RunnerError, RunnerKind, SessionEventSink, SessionId,
    SessionStore, ToolRegistry,
};

use crate::anthropic::{run_anthropic_turn, AnthropicAuth, ApiTurnError};
use crate::host::{HaHost, SystemPrompt, TurnInput};

/// What an [`AnthropicRunner`] needs beyond its HTTP client and tools.
#[derive(Clone)]
pub struct AnthropicConfig {
    /// Model id sent when the prelude names none.
    pub model: String,
    pub auth: AnthropicAuth,
    /// Full `/v1/messages` URL, e.g. [`crate::ANTHROPIC_MESSAGES_URL`].
    pub messages_url: String,
    /// Reasoning budget when the prelude names none.
    pub think: Option<ReasoningBudget>,
    /// Optional session log every turn's events are appended to.
    pub store: Option<Arc<dyn SessionStore>>,
}

/// One in-memory session.
struct LocalSession {
    id: SessionId,
    model: String,
    think: Option<ReasoningBudget>,
    system: SystemPrompt,
    tools: Option<Arc<dyn ToolRegistry>>,
    history: Mutex<Vec<Message>>,
    cancel: std::sync::Mutex<CancellationToken>,
}

#[async_trait]
impl HaHost for LocalSession {
    async fn turn_input(&self, _auth: &AnthropicAuth) -> TurnInput {
        TurnInput {
            session_id: self.id.clone(),
            model: self.model.clone(),
            think: self.think.clone(),
            system: self.system.clone(),
            history: self.history.lock().await.clone(),
            tool_registry: self.tools.clone(),
            cancel: self.cancel.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        }
    }

    async fn push_history(&self, message: Message) {
        self.history.lock().await.push(message);
    }
}

/// Pushes every event into an unbounded channel feeding `send`'s stream.
struct ChannelSink(futures::channel::mpsc::UnboundedSender<AgentEvent>);

impl SessionEventSink for ChannelSink {
    fn emit(&self, event: &AgentEvent) {
        let _ = self.0.unbounded_send(event.clone());
    }
}

/// The Anthropic `/v1/messages` cell behind the core [`Runner`] trait.
pub struct AnthropicRunner {
    config: AnthropicConfig,
    http: Arc<dyn HttpClient>,
    tools: Option<Arc<dyn ToolRegistry>>,
    sessions: Mutex<HashMap<SessionId, Arc<LocalSession>>>,
}

impl AnthropicRunner {
    pub fn with_http(
        config: AnthropicConfig,
        http: Arc<dyn HttpClient>,
        tools: Option<Arc<dyn ToolRegistry>>,
    ) -> Self {
        Self { config, http, tools, sessions: Mutex::new(HashMap::new()) }
    }

    /// [`Self::with_http`] over the default reqwest substrate.
    #[cfg(feature = "reqwest")]
    pub fn new(config: AnthropicConfig, tools: Option<Arc<dyn ToolRegistry>>) -> Self {
        Self::with_http(config, Arc::new(yah_runner_core::ReqwestClient::default()), tools)
    }

    fn open(&self, id: SessionId, prelude: &Prelude, history: Vec<Message>) -> LocalSession {
        LocalSession {
            id,
            model: prelude
                .model
                .as_ref()
                .and_then(|m| m.model.clone())
                .unwrap_or_else(|| self.config.model.clone()),
            think: prelude.reasoning.clone().or_else(|| self.config.think.clone()),
            system: SystemPrompt::from_prelude(prelude, &self.config.auth),
            tools: self.tools.clone(),
            history: Mutex::new(history),
            cancel: std::sync::Mutex::new(CancellationToken::new()),
        }
    }

    async fn session(&self, id: &SessionId) -> Result<Arc<LocalSession>, RunnerError> {
        self.sessions
            .lock()
            .await
            .get(id)
            .cloned()
            .ok_or_else(|| RunnerError::SessionNotFound(id.as_str().to_string()))
    }
}

/// Rebuild an Anthropic message history from a session's event log: each
/// `UserMessage` becomes a user message, each non-empty `TurnEnded` an
/// assistant message — as raw content blocks when the turn preserved them
/// (compaction / thinking blocks, R883-T11), as text otherwise. Tool calls
/// are turn-local and are not replayed (R833-B14).
pub fn reconstruct_history(events: &[AgentEvent]) -> Vec<Message> {
    events
        .iter()
        .filter_map(|ev| match ev {
            AgentEvent::UserMessage { text, .. } => Some(Message {
                role: Role::User,
                content: text.clone().into(),
            }),
            AgentEvent::TurnEnded { text, content_blocks, .. } if !text.is_empty() => {
                Some(Message {
                    role: Role::Assistant,
                    content: match content_blocks {
                        Some(blocks) => MessageContent::Blocks(blocks.clone()),
                        None => text.clone().into(),
                    },
                })
            }
            _ => None,
        })
        .collect()
}

#[async_trait]
impl Runner for AnthropicRunner {
    async fn start(&self, prelude: Prelude, _session_key: &str) -> Result<SessionId, RunnerError> {
        let id = mint_session_id();
        let session = self.open(id.clone(), &prelude, Vec::new());
        self.sessions.lock().await.insert(id.clone(), Arc::new(session));
        Ok(id)
    }

    async fn send(&self, session_id: &SessionId, message: String) -> Result<EventStream, RunnerError> {
        let session = self.session(session_id).await?;
        session.history.lock().await.push(Message {
            role: Role::User,
            content: message.into(),
        });
        let (tx, rx) = futures::channel::mpsc::unbounded();
        let http = Arc::clone(&self.http);
        let store = self.config.store.clone();
        let auth = self.config.auth.clone();
        let url = self.config.messages_url.clone();
        // The turn runs as the stream is polled: the receiver ends once the
        // turn future (which owns the only sender) completes and drops it.
        let turn = async move {
            let sink = ChannelSink(tx);
            let outcome = run_anthropic_turn(
                session.as_ref(),
                &sink,
                store.as_deref(),
                http.as_ref(),
                &auth,
                &url,
            )
            .await;
            if let Err(ApiTurnError { message, http_status, raw_body, error_type, usage }) = outcome {
                if let Some(usage) = usage {
                    sink.emit(&AgentEvent::TurnUsageDelta { session_id: session.id.clone(), usage });
                }
                sink.emit(&AgentEvent::Error {
                    session_id: session.id.clone(),
                    message,
                    http_status,
                    raw_body,
                    error_type,
                });
            }
        };
        let driver = futures::stream::once(turn).filter_map(|()| async { None::<AgentEvent> });
        Ok(Box::pin(futures::stream::select(rx, driver)))
    }

    async fn stop(&self, session_id: &SessionId) -> Result<bool, RunnerError> {
        let Some(session) = self.sessions.lock().await.get(session_id).cloned() else {
            return Ok(false);
        };
        let mut cancel = session.cancel.lock().unwrap_or_else(|e| e.into_inner());
        cancel.cancel();
        *cancel = CancellationToken::new();
        Ok(true)
    }

    async fn resume(
        &self,
        session_id: &SessionId,
        history: Vec<AgentEvent>,
        prelude: Prelude,
    ) -> Result<(), RunnerError> {
        let session = self.open(session_id.clone(), &prelude, reconstruct_history(&history));
        self.sessions.lock().await.insert(session_id.clone(), Arc::new(session));
        Ok(())
    }

    fn tools_registry(&self) -> Option<&Arc<dyn ToolRegistry>> {
        self.tools.as_ref()
    }

    fn kind(&self) -> RunnerKind {
        RunnerKind::AnthropicNative
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yah_runner_core::{
        CacheControl, CacheTtl, HttpError, HttpRequest, HttpResponse, PreludeSection,
    };

    /// Answers every request with the same SSE body and keeps each request
    /// body it was sent.
    struct StubHttp {
        sse: String,
        bodies: std::sync::Mutex<Vec<serde_json::Value>>,
    }

    #[async_trait]
    impl HttpClient for StubHttp {
        async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
            let body = request.body.expect("POST body");
            self.bodies.lock().unwrap().push(serde_json::from_slice(&body).unwrap());
            let chunk = bytes::Bytes::from(self.sse.clone());
            Ok(HttpResponse {
                status: 200,
                headers: Vec::new(),
                body: Box::pin(futures::stream::once(async move { Ok(chunk) })),
            })
        }
    }

    fn text_sse(text: &str) -> String {
        [
            r#"data: {"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":1}}}"#.to_string(),
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#.to_string(),
            format!(r#"data: {{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":"{text}"}}}}"#),
            r#"data: {"type":"content_block_stop","index":0}"#.to_string(),
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}"#.to_string(),
        ]
        .iter()
        .map(|f| format!("{f}\n\n"))
        .collect()
    }

    fn prelude() -> Prelude {
        let mut p = Prelude::empty(CacheControl { key: "k".into(), ttl: CacheTtl::Ephemeral });
        p.sections.push(PreludeSection::new("role", "You are terse."));
        p
    }

    fn runner(http: Arc<StubHttp>) -> AnthropicRunner {
        AnthropicRunner::with_http(
            AnthropicConfig {
                model: "claude-sonnet-4-5".into(),
                auth: AnthropicAuth::ApiKey("k".into()),
                messages_url: "https://example.invalid/v1/messages".into(),
                think: None,
                store: None,
            },
            http,
            None,
        )
    }

    #[tokio::test]
    async fn send_streams_one_turn_and_carries_history_into_the_next() {
        let http = Arc::new(StubHttp { sse: text_sse("17"), bodies: Default::default() });
        let r = runner(Arc::clone(&http));
        assert_eq!(r.kind(), RunnerKind::AnthropicNative);
        let id = r.start(prelude(), "project-1").await.unwrap();

        let events: Vec<AgentEvent> = r.send(&id, "hi".into()).await.unwrap().collect().await;
        assert!(matches!(events.first(), Some(AgentEvent::TurnStarted { .. })));
        match events.last() {
            Some(AgentEvent::TurnEnded { text, stop_reason, .. }) => {
                assert_eq!(text, "17");
                assert_eq!(stop_reason.as_deref(), Some("end_turn"));
            }
            other => panic!("expected TurnEnded last, got {other:?}"),
        }

        let _: Vec<AgentEvent> = r.send(&id, "again".into()).await.unwrap().collect().await;
        let bodies = http.bodies.lock().unwrap();
        assert_eq!(bodies.len(), 2);
        // The prelude rendered into one cached system block (API key: no cloak).
        assert_eq!(bodies[0]["system"].as_array().unwrap().len(), 1);
        assert_eq!(bodies[0]["system"][0]["text"], "You are terse.\n");
        assert_eq!(bodies[0]["model"], "claude-sonnet-4-5");
        // Turn 2 sends user, assistant, user.
        let msgs = bodies[1]["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1]["role"], "assistant");
    }

    #[tokio::test]
    async fn send_to_an_unknown_session_is_session_not_found() {
        let http = Arc::new(StubHttp { sse: text_sse("x"), bodies: Default::default() });
        let r = runner(http);
        let err = r.send(&SessionId::new("nope"), "hi".into()).await.err().expect("error");
        assert!(matches!(err, RunnerError::SessionNotFound(_)));
        assert!(!r.stop(&SessionId::new("nope")).await.unwrap());
    }

    #[test]
    fn reconstruct_history_prefers_preserved_blocks_and_skips_empty_turns() {
        let id = SessionId::new("s");
        let events = vec![
            AgentEvent::UserMessage { session_id: id.clone(), text: "q".into() },
            AgentEvent::TurnEnded {
                session_id: id.clone(),
                text: "a".into(),
                stop_reason: None,
                usage: None,
                content_blocks: Some(vec![serde_json::json!({ "type": "compaction", "content": "s" })]),
            },
            AgentEvent::TurnEnded {
                session_id: id,
                text: String::new(),
                stop_reason: None,
                usage: None,
                content_blocks: None,
            },
        ];
        let history = reconstruct_history(&events);
        assert_eq!(history.len(), 2);
        assert!(matches!(history[1].content, MessageContent::Blocks(_)));
    }
}
