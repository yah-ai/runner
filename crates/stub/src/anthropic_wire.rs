//! Serialize-side Anthropic Messages STREAMING response types (R923-T13).
//!
//! The Anthropic twin of [`crate::openai_wire`]: every SSE frame the stub's
//! `/v1/messages` route emits is built from these structs, so a drift in the
//! wire shape is a compile error here rather than a hand-typed fixture that
//! only an assertion can catch. `serde` only, no yah types.
//!
//! The one capability worth naming: a [`Block::ToolUse`] splits its JSON
//! arguments across `fragments` `input_json_delta` events, which is what a
//! real model does and what the runner's accumulator has to reassemble.

use serde::Serialize;
use serde_json::Value;

/// One SSE `data: {...}` event of a streamed message.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    MessageStart { message: MessageStart },
    ContentBlockStart { index: usize, content_block: BlockStart },
    ContentBlockDelta { index: usize, delta: BlockDelta },
    ContentBlockStop { index: usize },
    MessageDelta { delta: MessageDeltaBody, usage: OutputUsage },
    MessageStop,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MessageStart {
    pub usage: StartUsage,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StartUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OutputUsage {
    pub output_tokens: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MessageDeltaBody {
    pub stop_reason: String,
}

/// The `content_block` of a `content_block_start`.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlockStart {
    Text { text: String },
    ToolUse { id: String, name: String },
    /// Opens empty: the summary arrives in ONE `compaction_delta`, never on
    /// the start frame (R883-B13).
    Compaction { content: String },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlockDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    CompactionDelta { content: String },
}

/// One content block of a scripted assistant message.
#[derive(Clone, Debug, PartialEq)]
pub enum Block {
    Text(String),
    /// `input` is serialized then cut into `fragments` `input_json_delta`s.
    ToolUse { id: String, name: String, input: Value, fragments: usize },
    Compaction(String),
}

/// A whole scripted assistant message: blocks, stop reason, usage.
#[derive(Clone, Debug, PartialEq)]
pub struct MessagesTurn {
    pub blocks: Vec<Block>,
    pub stop_reason: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl MessagesTurn {
    /// An empty `end_turn` message; add blocks with the builder methods.
    pub fn new() -> Self {
        Self { blocks: Vec::new(), stop_reason: "end_turn".into(), input_tokens: 0, output_tokens: 0 }
    }

    /// The common case: one text block, `end_turn`.
    pub fn text(text: impl Into<String>) -> Self {
        Self::new().with_text(text)
    }

    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.blocks.push(Block::Text(text.into()));
        self
    }

    /// Append a `tool_use` block (args split across `fragments` deltas) and
    /// switch the stop reason to `tool_use`.
    pub fn with_tool_use(
        mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        input: Value,
        fragments: usize,
    ) -> Self {
        self.blocks.push(Block::ToolUse { id: id.into(), name: name.into(), input, fragments });
        self.stop_reason = "tool_use".into();
        self
    }

    pub fn with_compaction(mut self, summary: impl Into<String>) -> Self {
        self.blocks.push(Block::Compaction(summary.into()));
        self
    }

    /// `message_start` input tokens / `message_delta` output tokens.
    pub fn usage(mut self, input_tokens: u64, output_tokens: u64) -> Self {
        self.input_tokens = input_tokens;
        self.output_tokens = output_tokens;
        self
    }

    /// The typed event sequence for this message.
    pub fn events(&self) -> Vec<Event> {
        let mut out = vec![Event::MessageStart {
            message: MessageStart {
                usage: StartUsage { input_tokens: self.input_tokens, output_tokens: 1 },
            },
        }];
        for (index, block) in self.blocks.iter().enumerate() {
            match block {
                Block::Text(text) => {
                    out.push(Event::ContentBlockStart {
                        index,
                        content_block: BlockStart::Text { text: String::new() },
                    });
                    out.push(Event::ContentBlockDelta {
                        index,
                        delta: BlockDelta::TextDelta { text: text.clone() },
                    });
                }
                Block::ToolUse { id, name, input, fragments } => {
                    out.push(Event::ContentBlockStart {
                        index,
                        content_block: BlockStart::ToolUse { id: id.clone(), name: name.clone() },
                    });
                    for partial_json in split_json(&input.to_string(), *fragments) {
                        out.push(Event::ContentBlockDelta {
                            index,
                            delta: BlockDelta::InputJsonDelta { partial_json },
                        });
                    }
                }
                Block::Compaction(summary) => {
                    out.push(Event::ContentBlockStart {
                        index,
                        content_block: BlockStart::Compaction { content: String::new() },
                    });
                    out.push(Event::ContentBlockDelta {
                        index,
                        delta: BlockDelta::CompactionDelta { content: summary.clone() },
                    });
                }
            }
            out.push(Event::ContentBlockStop { index });
        }
        out.push(Event::MessageDelta {
            delta: MessageDeltaBody { stop_reason: self.stop_reason.clone() },
            usage: OutputUsage { output_tokens: self.output_tokens },
        });
        out.push(Event::MessageStop);
        out
    }

    /// The SSE frames, each a complete `data: {...}\n\n`.
    pub fn frames(&self) -> Vec<String> {
        self.events()
            .iter()
            .map(|e| format!("data: {}\n\n", serde_json::to_string(e).expect("wire event serializes")))
            .collect()
    }
}

impl Default for MessagesTurn {
    fn default() -> Self {
        Self::new()
    }
}

/// Cut `s` into at most `n` non-empty char-aligned pieces (fewer if `s` is
/// short). `n == 0` is treated as one piece.
fn split_json(s: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let n = n.max(1).min(chars.len().max(1));
    let per = chars.len().div_ceil(n);
    chars.chunks(per.max(1)).map(|c| c.iter().collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_use_arguments_split_across_fragments_and_reassemble() {
        let turn = MessagesTurn::text("Looking that up.")
            .with_tool_use("toolu_01", "board_show", json!({"id": "R833"}), 3)
            .usage(100, 40);
        let events = turn.events();
        let frags: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                Event::ContentBlockDelta { delta: BlockDelta::InputJsonDelta { partial_json }, .. } => {
                    Some(partial_json.as_str())
                }
                _ => None,
            })
            .collect();
        assert_eq!(frags.len(), 3, "args cut into three input_json_delta fragments");
        assert_eq!(serde_json::from_str::<Value>(&frags.concat()).unwrap(), json!({"id": "R833"}));
        assert!(matches!(events.last(), Some(Event::MessageStop)));
    }

    #[test]
    fn frames_carry_the_documented_wire_tags() {
        let frames = MessagesTurn::new().with_compaction("s").usage(50, 5).frames().concat();
        for tag in [
            r#""type":"message_start""#,
            r#""type":"compaction""#,
            r#""type":"compaction_delta""#,
            r#""stop_reason":"end_turn""#,
            r#""type":"message_stop""#,
        ] {
            assert!(frames.contains(tag), "missing {tag} in {frames}");
        }
    }
}
