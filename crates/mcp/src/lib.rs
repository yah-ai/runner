//! @arch:layer(runtime)
//! @arch:role(bridge)
//! @arch:see(.yah/docs/architecture/A028-yah-agent-runtime.md)
//! @arch:see(.yah/docs/architecture/A004-agent-tool-calls.md)
//!
//! MCP (Model Context Protocol) server adapter — exposes any
//! [`runner::ToolRegistry`] as a JSON-RPC 2.0 endpoint speaking the
//! MCP wire shape.
//!
//! The crate is generic over [`ToolRegistry`] so the same protocol
//! adapter serves two consumers without re-binding to a host-specific
//! registry:
//!
//! 1. **PVd (R028-F8)** — the wrapped `claude` CLI. Today PVd only
//!    sees Claude Code's builtin tools; once `yah-mcp` is registered
//!    in `<camp>/.claude/settings.json` (R028-F8-T2) every
//!    `KgToolRegistry` tool flows through `claude`'s MCP client.
//! 2. **HAk+M / HAo+M (R028-F3 mcp-connect)** — once Anthropic GAs the
//!    `mcp_servers` body field on `/v1/messages`, the same crate runs
//!    over TCP so a hosted Anthropic call can route tools/call into
//!    our registry.
//!
//! Today only the stdio transport ships — TCP is parked behind the
//! GA gate. The shared shape is [`McpServer::serve`], which takes any
//! `AsyncRead`/`AsyncWrite` pair; [`McpServer::serve_stdio`] is the
//! convenience wrapper for the PVd subprocess case.
//!
//! ## Wire shape
//!
//! Stdio transport per the MCP spec is newline-delimited JSON — each
//! request/response/notification is one line, terminated by `\n`. There
//! is no Content-Length header (that's the LSP shape, not MCP). Every
//! frame is JSON-RPC 2.0; this crate handles three methods plus a
//! couple of housekeeping notifications:
//!
//! - `initialize` — handshake; the server advertises a `tools` capability
//!   and its `serverInfo`. The client follows up with a
//!   `notifications/initialized` (one-way) which we accept silently.
//! - `tools/list` — returns every [`ToolSchema`] from the registry,
//!   wrapped as MCP's `{ name, description, inputSchema }` shape (the
//!   inputSchema is round-tripped verbatim — same JSON-Schema the runner
//!   already produces).
//! - `tools/call` — dispatches `name + arguments` through
//!   [`ToolRegistry::execute`] and returns the [`ToolOutcome`] as MCP
//!   content blocks. `ok = false` becomes `isError: true` so the LLM
//!   side can tell a tool's own failure from a protocol error.
//! - `ping` — empty result. Common keep-alive across MCP clients.
//!
//! Anything else returns `-32601 method not found`. Malformed JSON gets
//! `-32700 parse error`; missing/wrong-typed params get `-32602 invalid
//! params`.
//!
//! ## Approval gate placement
//!
//! The gate is the registry's concern, not this crate's. When the
//! registry is `KgToolRegistry`, `ToolRegistry::execute` already calls
//! `execute_gated`; this server does not double-gate. A future stdio
//! `ApprovalRouter` shim — which emits a sidecar
//! `AgentEvent::ApprovalRequested` back to the host process — will be
//! plugged into the bin's registry construction, not into this lib.
//! The lib stays a pure protocol adapter.
//!
//! @yah:ticket(R950-T8, "Move crates/yah/mcp lib (lib.rs + jsonrpc.rs) into yah-runner-mcp generic over core ToolRegistry; camp bin stays in crates/yah/mcp")
//! @yah:at(2026-10-05T05:08:22Z)
//! @yah:assignee(agent:bundle-anthropic-miravel)
//! @yah:parent(R950)
//! @yah:next("Measured: the lib half (lib.rs, jsonrpc.rs) touches only runner::ToolRegistry; every agent_tools/party/kg/camp-service use is in main.rs and stdio_router.rs (KgToolRegistry, ApprovalRouter, APPROVAL_SOCK_ENV, CampId, party::Job::from_kebab).")
//! @yah:next("Move lib.rs + jsonrpc.rs to oss/runner/crates/mcp (package yah-runner-mcp; yah-mcp is the camp crate's own name). crates/yah/mcp becomes bin-only (or a thin lib re-export if other crates import it — check with grep before deciding) depending on yah-runner-mcp.")
//! @yah:next("Tier: Cleric — the lib/bin boundary already exists; this is a file move plus manifest edits.")
//! @yah:verify("cd oss/runner && cargo test -p yah-runner-mcp")
//! @yah:verify("cargo build -p yah-mcp and the MCP stdio smoke used by PVd still answers tools/list")
//! @yah:status(review)
//! @yah:handoff("Moved lib.rs + jsonrpc.rs (11 tests) into yah-runner-mcp, generic over yah_runner_core::tool::ToolRegistry; deps: yah-runner-core, async-trait, serde, serde_json, tokio (io-util, io-std, rt).")
//! @yah:handoff("app/yah/cli (yah-image-mcp bin) imports mcp::{McpServer, ServerInfo}, so crates/yah/mcp stays a thin lib (`pub use yah_runner_mcp::*`, lib name `mcp`) plus the camp bin (main.rs, stdio_router.rs untouched); added yah-runner-mcp dep to its Cargo.toml. crates/yah/mcp/src/jsonrpc.rs deleted. This annotation block moved here with the lib.rs copy. app/yah/cli was not built (re-export keeps mcp:: paths identical). board.review via MCP failed twice with EAGAIN, so written by hand.")
//! @yah:verify("cd oss/runner && cargo test -p yah-runner-mcp: 11 passed")
//! @yah:verify("cargo tree -p yah-runner-mcp -e normal: no yah-board/party/kg/fob/bridge")
//! @yah:verify("cargo build -p yah-mcp: EXIT=0")
//! @yah:verify("smoke: (printf initialize + tools/list JSON-RPC lines; sleep 20) | target/debug/yah-mcp /Users/leif/ss/yah -> both answered; sleep keeps stdin open while the camp service starts")
//! @yah:depends_on(R950-F2)
//! @yah:files(oss/runner/crates/mcp/src/lib.rs)
//! @yah:files(crates/yah/mcp/src/lib.rs)
//! @yah:files(crates/yah/mcp/src/jsonrpc.rs)
//! @yah:files(crates/yah/mcp/src/main.rs)
//! @yah:files(crates/yah/mcp/src/stdio_router.rs)
//! @yah:files(crates/yah/mcp/Cargo.toml)

use async_trait::async_trait;
use yah_runner_core::tool::ToolRegistry;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};

mod jsonrpc;

pub use jsonrpc::{ErrorCode, JsonRpcError, JsonRpcRequest, JsonRpcResponse};

/// MCP protocol version this server implements. Clients pass their own
/// version string in `initialize.params.protocolVersion`; per the spec
/// we either echo the client's version (when we support it) or return
/// our latest. The 2024-11-05 revision is the first stable release of
/// the spec — newer revisions add capabilities but stay backwards-
/// compatible at the tools/list + tools/call shape we use.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Identity the server advertises in `initialize.result.serverInfo`.
/// Defaults to `("yah-mcp", crate version)` — the bin and lib hand the
/// same shape to the protocol layer so a future TCP server (R028-F3
/// mcp-connect) reads identically from a client's perspective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

impl Default for ServerInfo {
    fn default() -> Self {
        Self {
            name: "yah-mcp".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Generic over [`ToolRegistry`] so the same protocol code drives both
/// PVd's stdio subprocess and the future TCP server. Hold the registry
/// behind an [`Arc`] because [`ToolRegistry::execute`] is `&self` and
/// the server may dispatch concurrent calls in the future (today's
/// stdio path is sequential, but the abstraction stays clean).
pub struct McpServer<R: ToolRegistry + ?Sized> {
    registry: Arc<R>,
    server_info: ServerInfo,
}

impl<R: ToolRegistry + ?Sized> McpServer<R> {
    pub fn new(registry: Arc<R>) -> Self {
        Self {
            registry,
            server_info: ServerInfo::default(),
        }
    }

    pub fn with_server_info(mut self, info: ServerInfo) -> Self {
        self.server_info = info;
        self
    }

    /// Run the server reading newline-delimited JSON-RPC 2.0 frames
    /// from `reader` and writing responses to `writer`. Returns when
    /// the reader closes (EOF on stdin) or hits an unrecoverable IO
    /// error. Notifications consume input without producing output.
    pub async fn serve<Rd, Wr>(self, reader: Rd, mut writer: Wr) -> std::io::Result<()>
    where
        Rd: AsyncRead + Unpin,
        Wr: AsyncWrite + Unpin,
    {
        let mut lines = BufReader::new(reader).lines();
        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }
            if let Some(response) = self.handle_frame(&line).await {
                let mut buf = serde_json::to_vec(&response).expect("response always serialises");
                buf.push(b'\n');
                writer.write_all(&buf).await?;
                writer.flush().await?;
            }
        }
        Ok(())
    }

    /// Convenience wrapper for the PVd subprocess case — wires
    /// [`tokio::io::stdin`] / [`tokio::io::stdout`] into [`Self::serve`].
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn serve_stdio(self) -> std::io::Result<()> {
        self.serve(tokio::io::stdin(), tokio::io::stdout()).await
    }

    /// Convenience wrapper for tests / TCP transport — splits a single
    /// full-duplex stream (like [`tokio::io::DuplexStream`] or a TCP
    /// stream) into read + write halves and runs [`Self::serve`].
    pub async fn serve_split<S>(self, stream: S) -> std::io::Result<()>
    where
        S: AsyncRead + AsyncWrite + Unpin,
    {
        let (read, write) = tokio::io::split(stream);
        self.serve(read, write).await
    }

    /// Pure dispatch — takes one JSON-RPC frame text, returns `Some`
    /// for requests (a response is required), `None` for notifications
    /// (no response per JSON-RPC 2.0). Exposed for tests so we can
    /// drive the protocol without spinning up real IO.
    pub async fn handle_frame(&self, line: &str) -> Option<JsonRpcResponse> {
        let request: JsonRpcRequest = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                return Some(JsonRpcResponse::error(
                    Value::Null,
                    JsonRpcError::new(
                        ErrorCode::ParseError,
                        format!("invalid JSON-RPC frame: {e}"),
                    ),
                ));
            }
        };
        // Notifications carry no `id`; responses must not be sent for them.
        let is_notification = request.id.is_none();
        let id = request.id.clone().unwrap_or(Value::Null);

        let result = self.dispatch(&request.method, request.params.clone()).await;

        if is_notification {
            return None;
        }

        Some(match result {
            Ok(value) => JsonRpcResponse::ok(id, value),
            Err(err) => JsonRpcResponse::error(id, err),
        })
    }

    async fn dispatch(&self, method: &str, params: Option<Value>) -> Result<Value, JsonRpcError> {
        match method {
            "initialize" => self.handle_initialize(params),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(self.handle_tools_list()),
            "tools/call" => self.handle_tools_call(params).await,
            // Notifications we accept silently — the response value is
            // discarded by the caller because `id` was absent.
            "notifications/initialized" | "notifications/cancelled" => Ok(Value::Null),
            other => Err(JsonRpcError::new(
                ErrorCode::MethodNotFound,
                format!("method '{other}' not implemented by yah-mcp"),
            )),
        }
    }

    fn handle_initialize(&self, _params: Option<Value>) -> Result<Value, JsonRpcError> {
        // Per the spec, a server MAY echo the client's protocolVersion
        // when it supports it; we always advertise our supported version
        // and let the client decide whether to proceed.
        Ok(json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {}
            },
            "serverInfo": self.server_info,
        }))
    }

    fn handle_tools_list(&self) -> Value {
        let tools: Vec<Value> = self
            .registry
            .schemas()
            .into_iter()
            .map(|s| {
                json!({
                    "name": s.name,
                    "description": s.description,
                    "inputSchema": s.input_schema,
                })
            })
            .collect();
        json!({ "tools": tools })
    }

    async fn handle_tools_call(&self, params: Option<Value>) -> Result<Value, JsonRpcError> {
        #[derive(Deserialize)]
        struct CallParams {
            name: String,
            #[serde(default)]
            arguments: Value,
        }
        let p = params.ok_or_else(|| {
            JsonRpcError::new(
                ErrorCode::InvalidParams,
                "tools/call requires { name, arguments } params",
            )
        })?;
        let CallParams { name, arguments } = serde_json::from_value(p).map_err(|e| {
            JsonRpcError::new(
                ErrorCode::InvalidParams,
                format!("tools/call params malformed: {e}"),
            )
        })?;

        // arguments may be Null when the tool takes no args; normalise
        // to {} so registries that always parse an object don't choke.
        let args = if arguments.is_null() {
            json!({})
        } else {
            arguments
        };

        let outcome = self.registry.execute(&name, args).await;
        // MCP shape: result.content is a list of content blocks. We
        // ship one `text` block carrying the tool's JSON serialised as
        // a string — Claude Code (and every other MCP client we've
        // seen) renders this verbatim back to the model. `isError`
        // mirrors `outcome.ok` so an in-band tool failure is
        // distinguishable from a protocol error (the latter never
        // reaches this branch — it would have been an Err result with
        // a JSON-RPC error code).
        let body = serde_json::to_string(&outcome.result)
            .unwrap_or_else(|_| "<<unserialisable tool result>>".to_string());
        Ok(json!({
            "content": [{ "type": "text", "text": body }],
            "isError": !outcome.ok,
        }))
    }
}

/// Convenience: anything implementing [`ToolRegistry`] can be served.
/// The blanket isn't needed because [`McpServer::new`] takes
/// `Arc<R: ToolRegistry>` directly — kept here as a marker trait for
/// future compile-time clarity.
#[async_trait]
pub trait Servable: ToolRegistry {}

#[async_trait]
impl<T: ToolRegistry> Servable for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use yah_runner_core::tool::{ToolOutcome, ToolSchema};

    /// Test stub — returns a fixed schema list and lets each test
    /// pre-program one outcome via `with_outcome`.
    struct StubRegistry {
        schemas: Vec<ToolSchema>,
        outcome: ToolOutcome,
    }

    impl StubRegistry {
        fn empty() -> Self {
            Self {
                schemas: vec![],
                outcome: ToolOutcome::ok(json!({})),
            }
        }

        fn with_one_tool(name: &str) -> Self {
            Self {
                schemas: vec![ToolSchema {
                    name: name.to_string(),
                    description: format!("stub tool {name}"),
                    input_schema: json!({
                        "type": "object",
                        "properties": { "path": { "type": "string" } },
                        "required": ["path"]
                    }),
                }],
                outcome: ToolOutcome::ok(json!({ "stub": true })),
            }
        }

        fn with_outcome(mut self, outcome: ToolOutcome) -> Self {
            self.outcome = outcome;
            self
        }
    }

    #[async_trait]
    impl ToolRegistry for StubRegistry {
        fn schemas(&self) -> Vec<ToolSchema> {
            self.schemas.clone()
        }
        async fn execute(&self, _name: &str, _args: Value) -> ToolOutcome {
            self.outcome.clone()
        }
    }

    fn server_with(registry: StubRegistry) -> McpServer<StubRegistry> {
        McpServer::new(Arc::new(registry))
    }

    #[tokio::test]
    async fn initialize_advertises_tools_capability() {
        let server = server_with(StubRegistry::empty());
        let frame = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let response = server.handle_frame(frame).await.expect("response");
        let result = response.result.expect("ok result");
        assert_eq!(result["protocolVersion"], MCP_PROTOCOL_VERSION);
        assert!(result["capabilities"]["tools"].is_object());
        assert_eq!(result["serverInfo"]["name"], "yah-mcp");
    }

    #[tokio::test]
    async fn tools_list_round_trips_schemas_verbatim() {
        let server = server_with(StubRegistry::with_one_tool("read_file"));
        let frame = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#;
        let response = server.handle_frame(frame).await.expect("response");
        let result = response.result.expect("ok result");
        let tools = result["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "read_file");
        assert_eq!(tools[0]["description"], "stub tool read_file");
        // The MCP-shaped wrapper renames `input_schema` → `inputSchema`
        // but the schema body itself round-trips verbatim — same
        // JSON-Schema the runner emits.
        assert_eq!(tools[0]["inputSchema"]["type"], "object");
        assert_eq!(tools[0]["inputSchema"]["required"][0], "path");
    }

    #[tokio::test]
    async fn tools_call_success_path() {
        let registry = StubRegistry::with_one_tool("read_file")
            .with_outcome(ToolOutcome::ok(json!({ "content": "hi", "bytes": 2 })));
        let server = server_with(registry);
        let frame = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"read_file","arguments":{"path":"foo.txt"}}}"#;
        let response = server.handle_frame(frame).await.expect("response");
        let result = response.result.expect("ok result");
        assert_eq!(result["isError"], false);
        let blocks = result["content"].as_array().expect("content blocks");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "text");
        // The text block carries the tool's JSON serialised; the
        // round-trip preserves the original shape.
        let text = blocks[0]["text"].as_str().expect("text");
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["content"], "hi");
        assert_eq!(parsed["bytes"], 2);
    }

    #[tokio::test]
    async fn tools_call_in_band_failure_marks_iserror() {
        // ToolOutcome::fail is the gate-refusal / sandbox-escape /
        // KG-miss path — the registry returns ok=false and the LLM
        // sees a structured error to reason about. MCP signals this as
        // isError=true so the client can render the failure distinctly.
        let registry =
            StubRegistry::with_one_tool("read_file").with_outcome(ToolOutcome::fail("nope"));
        let server = server_with(registry);
        let frame = r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"missing","arguments":{}}}"#;
        let response = server.handle_frame(frame).await.expect("response");
        let result = response.result.expect("ok result");
        assert_eq!(result["isError"], true);
        let text = result["content"][0]["text"].as_str().expect("text");
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["error"], "nope");
    }

    #[tokio::test]
    async fn tools_call_normalises_null_arguments_to_object() {
        // Some clients send `arguments: null` for zero-arg tools; the
        // registry expects at least an object, so the protocol layer
        // smooths this over.
        let registry = StubRegistry::with_one_tool("ping_tool");
        let server = server_with(registry);
        let frame = r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"ping_tool","arguments":null}}"#;
        let response = server.handle_frame(frame).await.expect("response");
        // Just assert the request didn't error out at the params layer.
        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[tokio::test]
    async fn unknown_method_returns_method_not_found() {
        let server = server_with(StubRegistry::empty());
        let frame = r#"{"jsonrpc":"2.0","id":6,"method":"resources/list"}"#;
        let response = server.handle_frame(frame).await.expect("response");
        let err = response.error.expect("error");
        assert_eq!(err.code, ErrorCode::MethodNotFound as i32);
    }

    #[tokio::test]
    async fn malformed_json_returns_parse_error() {
        let server = server_with(StubRegistry::empty());
        let response = server.handle_frame("{not json").await.expect("response");
        let err = response.error.expect("error");
        assert_eq!(err.code, ErrorCode::ParseError as i32);
    }

    #[tokio::test]
    async fn tools_call_missing_params_returns_invalid_params() {
        let server = server_with(StubRegistry::with_one_tool("read_file"));
        let frame = r#"{"jsonrpc":"2.0","id":7,"method":"tools/call"}"#;
        let response = server.handle_frame(frame).await.expect("response");
        let err = response.error.expect("error");
        assert_eq!(err.code, ErrorCode::InvalidParams as i32);
    }

    #[tokio::test]
    async fn notifications_produce_no_response() {
        let server = server_with(StubRegistry::empty());
        let frame = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        // No `id` → notification → no response per JSON-RPC 2.0.
        assert!(server.handle_frame(frame).await.is_none());
    }

    #[tokio::test]
    async fn ping_returns_empty_object() {
        let server = server_with(StubRegistry::empty());
        let frame = r#"{"jsonrpc":"2.0","id":9,"method":"ping"}"#;
        let response = server.handle_frame(frame).await.expect("response");
        let result = response.result.expect("ok result");
        assert!(result.is_object());
        assert_eq!(result.as_object().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn serve_pumps_requests_and_responses_over_pipes() {
        // Drive the full serve loop over an in-memory duplex so the
        // newline framing / EOF-shutdown path is covered, not just
        // handle_frame. DuplexStream is full-duplex, so we write the
        // request, shutdown the write half (which propagates EOF to
        // the server's read loop), then read the response back from
        // the same DuplexStream.
        let server = server_with(StubRegistry::with_one_tool("read_file"));
        let (mut client, server_io) = tokio::io::duplex(4096);
        let task = tokio::spawn(server.serve_split(server_io));

        client
            .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}\n")
            .await
            .unwrap();
        client.shutdown().await.unwrap();

        let mut buf = Vec::new();
        tokio::io::copy(&mut client, &mut buf).await.unwrap();
        task.await.unwrap().unwrap();

        let line = std::str::from_utf8(&buf).unwrap().trim_end_matches('\n');
        let response: JsonRpcResponse = serde_json::from_str(line).unwrap();
        let tools = response.result.unwrap()["tools"].clone();
        assert_eq!(tools[0]["name"], "read_file");
    }
}
