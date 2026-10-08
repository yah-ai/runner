//! One tool, one turn: cargo run -p yah-runner-ho --example hello_ollama
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use yah_runner_core::{cache::*, prelude::*, tool::*, Runner};
use yah_runner_ho::{OpenAiCompatConfig, OpenAiCompatRunner, Transport};
use futures::StreamExt;
struct Calc;
#[async_trait]
impl ToolRegistry for Calc {
    fn schemas(&self) -> Vec<ToolSchema> {
        vec![ToolSchema {
            name: "add".into(),
            description: "Sum integers.".into(),
            input_schema: json!({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}),
        }]
    }
    async fn execute(&self, name: &str, args: Value) -> ToolOutcome {
        if name == "add" {
            let (a, b) = (args.get("a").and_then(|v| v.as_i64()).unwrap_or(0), args.get("b").and_then(|v| v.as_i64()).unwrap_or(0));
            ToolOutcome::ok(json!({"result": a + b}))
        } else { ToolOutcome::fail("unknown tool") }
    }
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3".into());
    let cfg = OpenAiCompatConfig {
        provider_label: "ollama".into(),
        endpoint: "http://localhost:11434/v1/chat/completions".into(),
        default_model: model,
        api_key: None,
        transport: Transport::Sse,
        max_output_tokens: None,
        max_tool_iterations: None,
        cost_guard_tokens: None,
        num_ctx: None,
        keep_alive: None,
        reasoning: None,
        quota_ledger: None,
        rate_limit_hint: None,
        session_preamble: None,
        context_window: None,
        prompt_overflow_threshold: None,
    };
    let runner = OpenAiCompatRunner::with_tools(cfg, Arc::new(Calc));
    let mut pre = Prelude::empty(CacheControl { key: "test".into(), ttl: CacheTtl::Ephemeral });
    pre.sections.push(PreludeSection::new("sys", "You are a calculator; use add."));
    let sid = runner.start(pre, "calc:1").await?;
    let mut ev = runner.send(&sid, "What is 7 + 5?".into()).await?;
    while let Some(e) = ev.next().await { println!("{e:?}"); }
    Ok(())
}
