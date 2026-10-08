//! R951-T4: live smoke — real `claude -p` (needs auth) driving the example MCP
//! tool server through `PvHost::mcp_config_path`.
//!
//! `cargo test -p yah-runner-pv --test live_claude -- --ignored`

use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use yah_runner_core::{AgentEvent, CacheControl, CacheTtl, Prelude, Runner, SessionId};
use yah_runner_pv::{PvConfig, PvHost, PvRunner};

struct ExampleHost(PathBuf);

impl PvHost for ExampleHost {
    fn mcp_config_path(&self, _: &SessionId) -> Option<PathBuf> {
        Some(self.0.clone())
    }
}

/// `target/<profile>/examples/mcp_tool_server`, found from this test binary
/// at `target/<profile>/deps/live_claude-*` (cargo builds examples for tests).
fn example_server() -> PathBuf {
    let deps = std::env::current_exe().unwrap();
    let profile = deps.parent().unwrap().parent().unwrap();
    let exe = profile.join("examples").join("mcp_tool_server");
    assert!(exe.exists(), "build examples first: {}", exe.display());
    exe
}

#[tokio::test]
#[ignore = "needs a logged-in `claude` CLI and network"]
async fn live_claude_calls_the_example_echo_tool() {
    let dir = tempfile::TempDir::new().unwrap();
    let config = dir.path().join("mcp.json");
    let cfg = json!({ "mcpServers": { "echo": { "command": example_server(), "args": ["serve"] } } });
    std::fs::write(&config, serde_json::to_vec(&cfg).unwrap()).unwrap();

    let runner = PvRunner::new(
        Arc::new(ExampleHost(config)),
        PvConfig {
            cli_binary: "claude".into(),
            workdir: dir.path().to_path_buf(),
            default_model: "claude-sonnet-4-6".into(),
        },
    );
    let prelude = Prelude::empty(CacheControl { key: "k".into(), ttl: CacheTtl::Ephemeral });
    let sid = runner.start(prelude, "live").await.unwrap();
    let mut s = runner
        .send(&sid, "Call the mcp__echo__echo tool with text \"pv-live\", then reply done.".into())
        .await
        .unwrap();
    let mut events = Vec::new();
    while let Some(ev) = std::future::poll_fn(|cx| s.as_mut().poll_next(cx)).await {
        events.push(ev);
    }
    runner.stop(&sid).await.unwrap();

    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::ToolCall { tool_name, .. } if tool_name == "mcp__echo__echo")),
        "{events:?}"
    );
    assert!(events.iter().any(|e| matches!(e, AgentEvent::ToolResult { .. })), "{events:?}");
}
