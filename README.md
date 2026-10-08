# runner

Agent-runtime crates extracted from the yah monorepo. Source of truth is
`oss/runner` in [yah-ai/yah](https://github.com/yah-ai/yah); this repo is a
read-only export mirror.

## Crates

**yah-runner-core** — The `Runner` trait contract and vocabulary (`AgentEvent`,
`Prelude`, `ToolRegistry`). Kept provider-agnostic and host-free: open a session
with a rendered prelude, send user turns, read event streams, stop, and resume.

**yah-runner-ho** — OpenAI-compatible runner for every provider speaking
`/v1/chat/completions` SSE (OpenAI, vLLM, Qwen, Groq, Ollama via `/v1`) and
Ollama's native `/api/chat` NDJSON endpoint. HTTP goes through `yah-runner-core::HttpClient`;
the reqwest substrate is the default `reqwest` feature.

**yah-runner-ha** — Anthropic-native runner for Claude models via `/v1/messages`.
Reserved for the Claude path; `app/yah/desktop/src/agent.rs` wires it for `claude-cli`.

**yah-runner-mcp** — MCP (Model Context Protocol) transport for a runner. Hosts
embed this to serve runners as MCP tools.

## Example

`crates/ho/examples/hello_ollama.rs` — A minimal tool-calling session against a
local Ollama instance. Registers one `add` tool, opens a session with a single
prelude section, sends a user message, and prints every `AgentEvent` from the stream.
Run with `cargo run -p yah-runner-ho --example hello_ollama` (requires Ollama on
`http://localhost:11434` and a tool-capable model; defaults to `qwen3` via `OLLAMA_MODEL` env).

Licensed under MIT OR Apache-2.0.
