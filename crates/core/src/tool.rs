//! Provider-agnostic tool registry surface.
//!
//! The runner sees this thin trait — list schemas, execute by name. *What* a
//! tool does is the host's business: hosts wrap their concrete tools in
//! something that implements [`ToolRegistry`] and hand it to a runner.
//!
//! Function-calling protocol (request body `tools[]`, streamed tool-call
//! deltas, the dispatch loop) is provider-agnostic and lives with the
//! runners; the registry cares only about the string name, the JSON schema
//! passed to the model, and the JSON result passed back. Approval gating and
//! write semantics layer on top on the host side — the runner does not see
//! them.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// One tool's manifest as the LLM will see it. The runner dumps these
/// into the request body's `tools[]` array on every turn.
///
/// `name` must match exactly the string the runner will pass to
/// [`ToolRegistry::execute`] when the LLM emits a `tool_calls[]` entry —
/// it's the identity key for dispatch. `description` is what the LLM
/// reads when deciding which tool to call; keep it short, action-
/// oriented, and surface any sandbox / cost gotchas. `input_schema` is
/// JSON Schema the LLM uses to construct arguments; OpenAI-compat
/// providers wrap it in `{ type: "function", function: { name,
/// description, parameters: input_schema } }` at request time, so
/// implementations should produce a `type: "object"` schema with the
/// argument fields under `properties`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Result of one tool invocation. `ok = false` is *not* a runner error —
/// it rides back to the LLM as the next-turn input so the model can
/// reason about and retry. Runner-level failures (missing tool name,
/// arg-parse failure before dispatch) are the only thing that surface
/// as `AgentEvent::Error`; everything a tool itself reports is in-band.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutcome {
    /// `true` when the tool produced its declared output. `false` when
    /// the tool reached its own error path (file not found, KG node id
    /// missing, sandbox escape, …) — `result` carries a structured
    /// description for the LLM to reason about.
    pub ok: bool,
    /// Tool-specific JSON. Read tools tend to emit
    /// `{ "content": "…", "bytes": N }`; KG tools emit their natural
    /// response shape. Failure path surfaces as e.g.
    /// `{ "error": "…", "kind": "sandbox_escape" }`.
    pub result: serde_json::Value,
}

impl ToolOutcome {
    pub fn ok(result: serde_json::Value) -> Self {
        Self { ok: true, result }
    }

    /// Convenience for the `ok = false` failure path. The LLM sees this
    /// as `{ "error": <message>, ...rest }` so it can correct course.
    pub fn fail(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            result: serde_json::json!({ "error": message.into() }),
        }
    }
}

/// Provider-agnostic tool surface the runner depends on. One
/// implementation per (host context, tool set) — hosts typically build it
/// from whatever handle their tools need (a workspace root, a service
/// client, ...).
///
/// `Send + Sync + 'static` because the runner stores it behind an `Arc`
/// and a session may dispatch tools from spawned tasks.
#[async_trait]
pub trait ToolRegistry: Send + Sync + 'static {
    /// Static manifest the runner injects into every provider request's
    /// `tools[]` array. Must be cheap — called once per turn.
    fn schemas(&self) -> Vec<ToolSchema>;

    /// Invoke a tool by `name` with the LLM-produced JSON `args`. The
    /// runner has no view of approval — write tools must already have
    /// passed the host's gate before this is called (P5 wiring).
    async fn execute(&self, name: &str, args: serde_json::Value) -> ToolOutcome;

    /// Install the per-turn cancel signal long-running tools observe.
    /// The session layer calls this on each send so tools (`bash`,
    /// any future pollers) can clean up side-state (subprocess kills,
    /// in-flight RPCs) instead of getting silently dropped when the
    /// agent task is aborted. Default no-op for registries that have
    /// no cancellable side-effects. R171-B1 P2.
    fn bind_cancel(&self, _token: tokio_util::sync::CancellationToken) {}
}

/// Rewrite a registry tool name into a form every wire we ship to will
/// accept. OpenAI Chat Completions, the Codex Responses API, and
/// Anthropic `/v1/messages` all enforce essentially the same regex on
/// `tools[].name` (`^[a-zA-Z0-9_-]+$`, Anthropic adds a 128-char cap),
/// but yah's registry uses dotted namespaces like `board.show` and
/// `kg.defs`. The single rewrite rule — `.` → `_` — is enough to
/// satisfy every wire; other characters yah uses (letters, digits,
/// underscore) already conform.
///
/// Wires call this once per schema when building the request body, and
/// pair it with [`build_wire_name_map`] so the model's echoed
/// `tool_calls[].name` can be routed back to the original registry
/// name before [`ToolRegistry::execute`] is invoked.
pub fn sanitize_tool_name_for_wire(name: &str) -> String {
    name.replace('.', "_")
}

/// Build a `wire-name → registry-name` map covering only the names
/// that needed rewriting under [`sanitize_tool_name_for_wire`].
///
/// Callers store this alongside the request body and consult it before
/// dispatching tool calls; a miss means the name was already conformant
/// and should pass through unchanged. An empty map means no schema
/// needed rewriting, so the wire can skip the lookup entirely on hot
/// paths if it wants.
pub fn build_wire_name_map(schemas: &[ToolSchema]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for s in schemas {
        let wire = sanitize_tool_name_for_wire(&s.name);
        if wire != s.name {
            map.insert(wire, s.name.clone());
        }
    }
    map
}

/// Largest serialized tool result the runner lets into a transcript — R883-T1
/// (W342 lever 4, "the never-send-it lever").
///
/// Every yah-native wire (`anthropic.rs`, `codex_oauth.rs`, `openai_compat.rs`)
/// used to put `serde_json::to_string(&outcome.result)` on the wire verbatim,
/// and the registry's own tools bound themselves generously: `read_file` and
/// `web_fetch` return up to 256 KB each (~64K tokens) — a quarter of a 272K
/// window for one call, re-sent on every later request of the turn.
///
/// Capping here, where the result *enters* the transcript, is the cheap
/// place: the dropped bytes are never billed, and nothing already in the
/// cached prefix is edited. 32 KB (~8K tokens) is 4× claude-cli's `Read` clamp
/// (`agent-tools/src/read_cap.rs`, `READ_CAP_BYTES`) because this cap covers
/// every tool — structured `board.*` / `kg.*` payloads included — not just
/// file reads, and well under claude-cli's own 25K-token MCP result cap.
///
/// The `AgentEvent::ToolResult` event still carries the full result: the log
/// and renderer are not the context window.
pub const TOOL_RESULT_CAP_BYTES: usize = 32 * 1024;

/// Room kept for the elision marker. Its escaped length stays well under this,
/// which is what lets one pass over a leaf land under the cap.
const ELISION_MARKER_RESERVE_BYTES: usize = 256;

/// A string leaf shorter than this is not worth eliding — the marker would
/// cost about as much as it saves. Results that are over the cap without any
/// leaf this large (thousands of short rows) fall back to a text cut.
const MIN_ELIDABLE_LEAF_BYTES: usize = 1024;

/// Bound on per-leaf passes before falling back to cutting the serialized
/// text; each pass re-serializes the whole value.
const MAX_LEAF_PASSES: usize = 16;

/// Serialize a tool result for the wire, capped at [`TOOL_RESULT_CAP_BYTES`].
///
/// Every wire calls this — the live dispatch path AND any history replay —
/// so a replayed transcript is byte-identical to the live one and stays
/// cache-stable. Pure function of `result`; no env, no clock.
pub fn tool_result_wire_content(result: &serde_json::Value) -> String {
    cap_tool_result(result, TOOL_RESULT_CAP_BYTES)
}

fn cap_tool_result(result: &serde_json::Value, cap: usize) -> String {
    let full = serde_json::to_string(result).unwrap_or_else(|_| "null".to_string());
    if full.len() <= cap {
        return full;
    }

    // Preferred: elide the middle of the largest string leaves, so the result
    // stays valid JSON with every field present — `{content, bytes}` from a
    // read keeps its `bytes`, and the content keeps its head and its tail.
    let mut shrunk = result.clone();
    let mut len = full.len();
    for _ in 0..MAX_LEAF_PASSES {
        if len <= cap {
            break;
        }
        let mut best = None;
        largest_string_leaf(&shrunk, &mut String::new(), &mut best);
        let Some((ptr, leaf_len)) = best else { break };
        if leaf_len < MIN_ELIDABLE_LEAF_BYTES {
            break;
        }
        let Some(serde_json::Value::String(leaf)) = shrunk.pointer_mut(&ptr) else {
            break;
        };
        *leaf = elide_leaf_to_fit(leaf, len - cap, cap);
        let next = serde_json::to_string(&shrunk).map(|s| s.len()).unwrap_or(len);
        if next >= len {
            break;
        }
        len = next;
    }
    if len <= cap {
        if let Ok(s) = serde_json::to_string(&shrunk) {
            return s;
        }
    }

    // Fallback: no dominant leaf (many small rows). Cut the serialized text.
    elide_middle(&full, cap.saturating_sub(ELISION_MARKER_RESERVE_BYTES), cap)
}

/// Elide the middle of `leaf` so its *serialized* form shrinks by at least
/// `over` bytes, or as far as it can. Measured, not estimated: escaping makes
/// raw and serialized lengths diverge (`\n` is two bytes on the wire), so each
/// candidate is cut from the original string and re-measured — never from a
/// previous candidate, which would nest markers and falsify their counts.
fn elide_leaf_to_fit(leaf: &str, over: usize, cap: usize) -> String {
    let wire_len = |s: &str| serde_json::to_string(s).map(|w| w.len()).unwrap_or(s.len());
    let leaf_wire = wire_len(leaf).max(1);
    let target = leaf_wire.saturating_sub(over);
    // First guess in raw bytes, scaled by this leaf's own escape ratio; the
    // loop below only ever shrinks it, so it must not start at zero.
    let mut keep = (target.saturating_sub(ELISION_MARKER_RESERVE_BYTES) as u128
        * leaf.len() as u128
        / leaf_wire as u128) as usize;
    loop {
        let candidate = elide_middle(leaf, keep, cap);
        let got = wire_len(&candidate);
        if got <= target || keep == 0 {
            return candidate;
        }
        // Dropping d more raw bytes drops at least d serialized bytes; the
        // marker only moves by a digit or two, so this converges in a few steps.
        keep = keep.saturating_sub(got - target);
    }
}

/// Depth-first search for the longest string leaf, recorded as a JSON pointer
/// so the caller can re-borrow it mutably via `Value::pointer_mut`.
fn largest_string_leaf(
    value: &serde_json::Value,
    ptr: &mut String,
    best: &mut Option<(String, usize)>,
) {
    use serde_json::Value;
    match value {
        Value::String(s) => {
            if best.as_ref().map_or(true, |(_, n)| s.len() > *n) {
                *best = Some((ptr.clone(), s.len()));
            }
        }
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                let mark = ptr.len();
                ptr.push('/');
                ptr.push_str(&i.to_string());
                largest_string_leaf(item, ptr, best);
                ptr.truncate(mark);
            }
        }
        Value::Object(map) => {
            for (key, item) in map {
                let mark = ptr.len();
                ptr.push('/');
                ptr.push_str(&key.replace('~', "~0").replace('/', "~1"));
                largest_string_leaf(item, ptr, best);
                ptr.truncate(mark);
            }
        }
        _ => {}
    }
}

/// Keep `keep` bytes of `s` — three quarters head, one quarter tail, both on
/// char boundaries — and replace the middle with a marker naming what went.
fn elide_middle(s: &str, keep: usize, cap: usize) -> String {
    if keep >= s.len() {
        return s.to_string();
    }
    let mut head_end = keep * 3 / 4;
    while !s.is_char_boundary(head_end) {
        head_end -= 1;
    }
    let mut tail_start = s.len() - (keep - keep * 3 / 4);
    while !s.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    let tail_start = tail_start.max(head_end);
    let omitted = &s[head_end..tail_start];
    let lines = omitted.matches('\n').count();
    let what = if lines > 0 {
        format!("{lines} lines / {} bytes", omitted.len())
    } else {
        format!("{} bytes", omitted.len())
    };
    format!(
        "{}\n…[{what} omitted · tool result capped at {} KB before entering the transcript · \
         re-call with a narrower range, pattern or limit]…\n{}",
        &s[..head_end],
        cap / 1024,
        &s[tail_start..],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_result_under_the_cap_serializes_verbatim() {
        let result = serde_json::json!({ "content": "hello\nworld\n", "bytes": 12 });
        assert_eq!(
            tool_result_wire_content(&result),
            serde_json::to_string(&result).unwrap(),
        );
    }

    #[test]
    fn oversized_read_keeps_structure_head_and_tail() {
        let body: String = (0..20_000).map(|i| format!("line {i}\n")).collect();
        let result = serde_json::json!({ "content": body, "bytes": body.len() });
        let wire = tool_result_wire_content(&result);

        assert!(wire.len() <= TOOL_RESULT_CAP_BYTES, "{} > cap", wire.len());
        let parsed: serde_json::Value =
            serde_json::from_str(&wire).expect("leaf elision keeps valid JSON");
        assert_eq!(parsed["bytes"], body.len(), "sibling fields survive untouched");
        let content = parsed["content"].as_str().unwrap();
        assert!(content.starts_with("line 0\nline 1\n"));
        assert!(content.ends_with("line 19999\n"));
        assert!(content.contains("lines /"), "marker counts omitted lines: {content:.0}");
        assert!(content.contains("capped at 32 KB"));
    }

    #[test]
    fn many_small_rows_fall_back_to_a_text_cut_under_the_cap() {
        let rows: Vec<String> = (0..5_000).map(|i| format!("row-{i}")).collect();
        let result = serde_json::json!({ "rows": rows });
        let wire = cap_tool_result(&result, 4 * 1024);
        assert!(wire.len() <= 4 * 1024, "{} > cap", wire.len());
        assert!(wire.starts_with("{\"rows\":[\"row-0\""));
        assert!(wire.contains("bytes omitted"));
    }

    #[test]
    fn elision_respects_multibyte_char_boundaries() {
        let body = "é日\n".repeat(20_000);
        let result = serde_json::Value::String(body);
        let wire = cap_tool_result(&result, 2 * 1024);
        assert!(wire.len() <= 2 * 1024, "{} > cap", wire.len());
        let parsed: serde_json::Value = serde_json::from_str(&wire).unwrap();
        assert!(parsed.as_str().unwrap().starts_with("é日\n"));
    }

    #[test]
    fn sanitize_replaces_dots_with_underscores() {
        assert_eq!(sanitize_tool_name_for_wire("board.show"), "board_show");
        assert_eq!(sanitize_tool_name_for_wire("kg.defs"), "kg_defs");
    }

    #[test]
    fn sanitize_is_identity_for_already_conformant_names() {
        assert_eq!(sanitize_tool_name_for_wire("read_file"), "read_file");
        assert_eq!(sanitize_tool_name_for_wire("approve-tool"), "approve-tool");
        assert_eq!(sanitize_tool_name_for_wire("mcp__yah__board_show"), "mcp__yah__board_show");
    }

    #[test]
    fn name_map_carries_only_rewritten_entries() {
        let schemas = vec![
            ToolSchema {
                name: "board.show".into(),
                description: "show ticket".into(),
                input_schema: serde_json::json!({}),
            },
            ToolSchema {
                name: "read_file".into(),
                description: "read file".into(),
                input_schema: serde_json::json!({}),
            },
        ];
        let map = build_wire_name_map(&schemas);
        assert_eq!(map.get("board_show").map(String::as_str), Some("board.show"));
        assert!(!map.contains_key("read_file"), "conformant names skip the map");
    }

    #[test]
    fn name_map_is_empty_when_every_name_already_conforms() {
        let schemas = vec![ToolSchema {
            name: "read_file".into(),
            description: "read file".into(),
            input_schema: serde_json::json!({}),
        }];
        assert!(build_wire_name_map(&schemas).is_empty());
    }

    /// Tripwire: every name the host advertises through the canonical
    /// [`ToolRegistry`] surface must round-trip
    /// `sanitize_tool_name_for_wire`. This guards against someone
    /// adding a tool with `/`, `:`, or whitespace in the name — which
    /// `.` → `_` alone won't rescue and which every wire will 400 on.
    #[test]
    fn sanitized_names_match_the_shared_wire_regex() {
        // Reuse the canonical regex inline so this test is hermetic.
        let ok = |s: &str| {
            !s.is_empty()
                && s.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        };
        let candidates = [
            "read_file",
            "board.show",
            "kg.defs",
            "app.active_tab",
            "mcp__yah__board_show",
            "approve-tool",
        ];
        for name in candidates {
            let wire = sanitize_tool_name_for_wire(name);
            assert!(
                ok(&wire),
                "sanitized name {wire:?} (from {name:?}) doesn't match the shared wire regex",
            );
        }
    }
}
