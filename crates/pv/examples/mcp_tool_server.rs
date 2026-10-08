//! How a non-yah consumer hands `claude -p` its own tools.
//!
//! claude reaches tools only through MCP servers named in `--mcp-config`, so a
//! consumer builds a stdio server on `yah_runner_mcp::McpServer` over its core
//! `ToolRegistry`, and its `PvHost` points `mcp_config_path` at a config that
//! launches that binary.
//!
//! ```text
//! cargo run -p yah-runner-pv --example mcp_tool_server          # print the claude argv
//! cargo run -p yah-runner-pv --example mcp_tool_server -- serve # be the MCP server
//! ```
//!
//! Part of R951. Running claude itself is not needed (or done) here.

use async_trait::async_trait;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use yah_runner_core::{SessionId, ToolOutcome, ToolRegistry, ToolSchema};
use yah_runner_mcp::McpServer;
use yah_runner_pv::spawn::claude_fixed_args;
use yah_runner_pv::{PvHost, SpawnSpec};

/// One tool: echo its `text` argument back.
struct EchoTools;

#[async_trait]
impl ToolRegistry for EchoTools {
    fn schemas(&self) -> Vec<ToolSchema> {
        vec![ToolSchema {
            name: "echo".into(),
            description: "Echo `text` back.".into(),
            input_schema: json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"],
            }),
        }]
    }

    async fn execute(&self, name: &str, args: serde_json::Value) -> ToolOutcome {
        match name {
            "echo" => ToolOutcome::ok(args.get("text").cloned().unwrap_or_default()),
            other => ToolOutcome { ok: false, result: json!(format!("unknown tool {other}")) },
        }
    }
}

/// A host whose only injection is the MCP config naming this binary.
struct ExampleHost {
    mcp_config: PathBuf,
}

impl PvHost for ExampleHost {
    fn mcp_config_path(&self, _session_id: &SessionId) -> Option<PathBuf> {
        Some(self.mcp_config.clone())
    }
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    if std::env::args().nth(1).as_deref() == Some("serve") {
        return McpServer::new(Arc::new(EchoTools)).serve_stdio().await;
    }

    // claude's `--mcp-config` shape: `mcpServers.<name>` = a command to spawn.
    // claude publishes this server's tool as `mcp__echo__echo`.
    let exe = std::env::current_exe()?;
    let config = json!({
        "mcpServers": {
            "echo": { "command": exe, "args": ["serve"] }
        }
    });
    let mcp_config = std::env::temp_dir().join("yah-runner-pv-example-mcp.json");
    std::fs::write(&mcp_config, serde_json::to_vec_pretty(&config)?)?;

    let host = ExampleHost { mcp_config };
    let sid = SessionId::new("session:example");
    let cwd = std::env::current_dir()?;
    let spec = SpawnSpec::new("claude", &cwd, "claude-sonnet-4-6", &sid);
    let argv = claude_fixed_args(&spec, &host);
    println!(
        "claude {}",
        argv.iter().map(|a| a.to_string_lossy()).collect::<Vec<_>>().join(" ")
    );
    Ok(())
}
