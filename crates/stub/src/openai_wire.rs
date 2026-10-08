//! Serialize-side OpenAI chat-completions STREAMING response types (R923-F11).
//!
//! Every SSE frame [`crate::inference_stub`] emits is built from these
//! structs, so a change to the wire shape is a compile error in the stub
//! rather than a drifted fixture that only an assertion can catch.
//!
//! Self-contained on purpose — `serde` only, no yah types — so the module can
//! move to a shared wire crate beside the runner's deserialize side
//! (`yah-runner-ho`'s `openai_compat`) without a rewrite. Until it does, the
//! runner round-trip test in `inference_stub` is what proves the two agree.

use serde::Serialize;

/// One `data: {...}` payload of a streamed `chat.completion.chunk`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChatCompletionChunk {
    pub choices: Vec<ChunkChoice>,
    /// Present only on the trailing usage chunk (`stream_options.include_usage`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl ChatCompletionChunk {
    /// A single-choice chunk carrying `delta`, still streaming.
    pub fn delta(delta: Delta) -> Self {
        Self { choices: vec![ChunkChoice { delta, finish_reason: None }], usage: None }
    }

    /// The empty-delta chunk that ends a choice with `reason`.
    pub fn finish(reason: FinishReason) -> Self {
        Self {
            choices: vec![ChunkChoice { delta: Delta::default(), finish_reason: Some(reason) }],
            usage: None,
        }
    }

    /// The choice-less trailing chunk that reports token usage.
    pub fn usage(usage: Usage) -> Self {
        Self { choices: Vec::new(), usage: Some(usage) }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChunkChoice {
    pub delta: Delta,
    /// Serialized as `null` while streaming — the wire always carries the key.
    pub finish_reason: Option<FinishReason>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

impl Delta {
    pub fn content(text: impl Into<String>) -> Self {
        Self { content: Some(text.into()), tool_calls: None }
    }

    pub fn tool_call(tc: ToolCallDelta) -> Self {
        Self { content: None, tool_calls: Some(vec![tc]) }
    }
}

/// One fragment of a streamed tool call. The first fragment for an `index`
/// carries `id`, `type` and `function.name`; later ones only append
/// `function.arguments`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolCallDelta {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<ToolCallKind>,
    pub function: FunctionDelta,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallKind {
    Function,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub arguments: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    ToolCalls,
    Length,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// One SSE event on the stream: a chunk, or the `[DONE]` sentinel.
#[derive(Clone, Debug, PartialEq)]
pub enum SseFrame {
    Chunk(ChatCompletionChunk),
    Done,
}

impl SseFrame {
    /// The frame as `data: …\n\n` wire text.
    pub fn to_sse(&self) -> String {
        match self {
            Self::Chunk(c) => {
                let json = serde_json::to_string(c).expect("chunk types always serialize");
                format!("data: {json}\n\n")
            }
            Self::Done => "data: [DONE]\n\n".to_string(),
        }
    }
}

impl From<ChatCompletionChunk> for SseFrame {
    fn from(c: ChatCompletionChunk) -> Self {
        Self::Chunk(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn payload(f: SseFrame) -> Value {
        let s = f.to_sse();
        let body = s.strip_prefix("data: ").and_then(|s| s.strip_suffix("\n\n")).unwrap();
        serde_json::from_str(body).unwrap()
    }

    #[test]
    fn shapes_match_the_openai_stream() {
        assert_eq!(
            payload(ChatCompletionChunk::delta(Delta::content("hi")).into()),
            json!({"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]})
        );
        assert_eq!(
            payload(ChatCompletionChunk::finish(FinishReason::ToolCalls).into()),
            json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        );
        assert_eq!(
            payload(
                ChatCompletionChunk::usage(Usage {
                    prompt_tokens: 1,
                    completion_tokens: 2,
                    total_tokens: 3
                })
                .into()
            ),
            json!({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}})
        );
        let tc = ToolCallDelta {
            index: 0,
            id: Some("call_1".into()),
            kind: Some(ToolCallKind::Function),
            function: FunctionDelta { name: Some("f".into()), arguments: String::new() },
        };
        assert_eq!(
            payload(ChatCompletionChunk::delta(Delta::tool_call(tc)).into()),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                "function": {"name": "f", "arguments": ""}}]}, "finish_reason": null}]})
        );
        assert_eq!(SseFrame::Done.to_sse(), "data: [DONE]\n\n");
    }
}
