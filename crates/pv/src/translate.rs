//! claude `-p --output-format stream-json` frame -> event translation.
//!
//! Part of R951 — copied from `app/yah/desktop/src/agent_process.rs`
//! (`translate_frame` and its helpers); the desktop copy is deleted by
//! R951-F3. Everything here is pure: one frame in, zero-or-more [`PvEvent`]s
//! out, with the caller owning every cursor so a reader can carry them
//! across its line loop.
//!
//! Two yah-camp enrichments are deliberately NOT done here and belong to the
//! host's stream tap: stripping `<yah:followups>` blocks (and lifting their
//! hint chips onto `TurnEnded`), and the `bash_ast` `_shape` / `_relocated`
//! summary on `bash` tool calls. Both are yah's own protocol, not claude's.

use crate::event::PvEvent;
use yah_runner_core::{AgentEvent, BlockReason, SessionId, TurnUsage};

/// Event buffer the translator pushes into. Accepts both core
/// [`AgentEvent`]s and the pv-only [`PvEvent`] variants.
#[derive(Default)]
struct Out(Vec<PvEvent>);

impl Out {
    fn push(&mut self, ev: impl Into<PvEvent>) {
        self.0.push(ev.into());
    }
}

/// Map one claude stream-json frame to zero-or-more [`PvEvent`]s.
///
/// `turn_open`, `accumulated`, `accumulated_thinking` and
/// `peak_context_tokens` are reader-owned cursors carried across frames.
pub fn translate_frame(
    frame: &serde_json::Value,
    session_id: &SessionId,
    turn_open: &mut bool,
    accumulated: &mut String,
    accumulated_thinking: &mut String,
    peak_context_tokens: &mut Option<u32>,
) -> Vec<PvEvent> {
    let mut out = Out::default();
    let kind = frame.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match kind {
        // Subprocess init — claude announces session_id, model,
        // cwd, registered tools, etc. The renderer doesn't care
        // about most of this; we already emitted SessionStarted at
        // spawn time so we can drop the frame.
        //
        // Exception: `subtype: "compact_boundary"` (emitted mid-stream
        // after `/compact`). Without surfacing this, claude follows
        // it with a `result` frame whose `num_turns: 0` fires TurnEnded
        // without any matching TurnStarted, and the UI hangs on the
        // "Warming up..." indicator forever. Emit a synthetic
        // assistant bubble carrying pre→post token counts so the
        // lifecycle resolves cleanly and the user can see compact ran.
        "system" => {
            if frame.get("status").and_then(|v| v.as_str()) == Some("compacting") {
                out.push(PvEvent::CompactingStarted {
                    session_id: session_id.clone(),
                });
            } else if frame.get("subtype").and_then(|v| v.as_str()) == Some("compact_boundary") {
                let meta = frame.get("compact_metadata");
                let pre = meta
                    .and_then(|m| m.get("pre_tokens"))
                    .and_then(|n| n.as_u64());
                let post = meta
                    .and_then(|m| m.get("post_tokens"))
                    .and_then(|n| n.as_u64());
                let msg = match (pre, post) {
                    (Some(p), Some(q)) => {
                        format!("Context compacted — {p} → {q} tokens.")
                    }
                    _ => "Context compacted.".to_string(),
                };
                if !*turn_open {
                    *turn_open = true;
                    accumulated.clear();
                    accumulated_thinking.clear();
                    out.push(AgentEvent::TurnStarted {
                        session_id: session_id.clone(),
                    });
                }
                accumulated.push_str(&msg);
                out.push(AgentEvent::MessageDelta {
                    session_id: session_id.clone(),
                    text: msg.clone(),
                });
                out.push(AgentEvent::AssistantText {
                    session_id: session_id.clone(),
                    text: msg,
                });
                // R280-F1: emit structured token-count event for the gauge
                if let (Some(p), Some(q)) = (pre, post) {
                    out.push(AgentEvent::ContextCompacted {
                        session_id: session_id.clone(),
                        pre_tokens: p,
                        post_tokens: q,
                    });
                }
            } else {
                // `system` is where claude puts its extension frames —
                // `init`, `status`, `thinking_tokens`, and the whole
                // `task_*` subagent lifecycle — so an unhandled subtype
                // here is as much a blind spot as an unhandled type.
                note_untranslated_frame("system", frame.get("subtype").and_then(|v| v.as_str()));
            }
        }
        // Streaming partial messages from `--include-partial-messages`.
        // Claude wraps Anthropic's SSE events in a `stream_event`
        // frame; the inner `event.type == "content_block_delta"` is
        // what our HA path also folds into MessageDelta.
        "stream_event" => {
            let event = frame.get("event");
            let inner_type = event
                .and_then(|e| e.get("type"))
                .and_then(|t| t.as_str())
                .unwrap_or("");
            match inner_type {
                "message_start" => {
                    if !*turn_open {
                        *turn_open = true;
                        accumulated.clear();
                        accumulated_thinking.clear();
                        out.push(AgentEvent::TurnStarted {
                            session_id: session_id.clone(),
                        });
                    }
                }
                "content_block_delta" => {
                    // Anthropic distinguishes block kinds via `delta.type`:
                    // `text_delta` → `delta.text`, `thinking_delta` →
                    // `delta.thinking`. We route each to its own AgentEvent
                    // so the renderer can stream them into separate bubbles.
                    let delta = event.and_then(|e| e.get("delta"));
                    let delta_type = delta
                        .and_then(|d| d.get("type"))
                        .and_then(|t| t.as_str())
                        .unwrap_or("");
                    match delta_type {
                        "thinking_delta" => {
                            if let Some(text) = delta
                                .and_then(|d| d.get("thinking"))
                                .and_then(|t| t.as_str())
                            {
                                accumulated_thinking.push_str(text);
                                out.push(AgentEvent::ThinkingDelta {
                                    session_id: session_id.clone(),
                                    text: text.to_string(),
                                });
                            }
                        }
                        // Default to text_delta for backwards compat —
                        // older claude versions emit `delta.text` without
                        // a `delta.type` discriminator.
                        _ => {
                            if let Some(text) =
                                delta.and_then(|d| d.get("text")).and_then(|t| t.as_str())
                            {
                                accumulated.push_str(text);
                                out.push(AgentEvent::MessageDelta {
                                    session_id: session_id.clone(),
                                    text: text.to_string(),
                                });
                            }
                        }
                    }
                }
                // Anthropic's per-chunk running usage (output_tokens
                // primarily; cache/input numbers ride message_start).
                // Surfacing as TurnUsageDelta lets the UI animate the
                // budget gauge without waiting for `result`.
                "message_delta" => {
                    if let Some(usage) = event
                        .and_then(|e| e.get("usage"))
                        .and_then(extract_partial_usage)
                    {
                        out.push(AgentEvent::TurnUsageDelta {
                            session_id: session_id.clone(),
                            usage,
                        });
                    }
                }
                _ => {}
            }
        }
        // Whole-message frame (emitted regardless of
        // --include-partial-messages). Walks the content blocks in
        // order and emits one durable event per block:
        //
        // - `text` → `AssistantText` (persisted to JSONL inline with
        //   `ToolCall` events, so replay reconstructs text-vs-tool
        //   interleaving instead of relying on the aggregate
        //   `TurnEnded.text`). In non-SSE mode (no
        //   `--include-partial-messages`) the live renderer also gets
        //   a fallback `MessageDelta` carrying the whole frame's text
        //   up-front so a chat bubble exists by the time any
        //   `ToolCall` event resets the streaming-event-id.
        // - `thinking` → `ThinkingDelta` (fallback for non-SSE mode,
        //   deduped by the `accumulated_thinking` buffer).
        // - `tool_use` → `ToolCall`. claude dispatches its built-in
        //   tools (Read / Edit / Write / Bash) internally, so this is
        //   the only place the host sees them. Pairs with
        //   `ToolResult` emitted from the matching `user` frame's
        //   `tool_result` blocks.
        "assistant" => {
            let parent_tool_use_id = frame
                .get("parent_tool_use_id")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            // Track the per-API-call context window size for the ring display.
            // Each assistant frame carries message.usage with the tokens for
            // that one Anthropic API call (not accumulated across tool loops).
            // We take the max so TurnEnded.usage.peakContextTokens reflects the
            // largest single-call context seen during the turn.
            if let Some(usage) = frame.get("message").and_then(|m| m.get("usage")) {
                let read_u32 =
                    |k: &str| -> u32 { usage.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as u32 };
                let call_ctx = read_u32("input_tokens")
                    + read_u32("cache_read_input_tokens")
                    + read_u32("cache_creation_input_tokens");
                if call_ctx > 0 {
                    *peak_context_tokens = Some(peak_context_tokens.unwrap_or(0).max(call_ctx));
                }
            }
            let blocks = frame
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_array());
            let Some(blocks) = blocks else {
                return out.0;
            };
            let sse_filled_text = !accumulated.is_empty();
            let sse_filled_thinking = !accumulated_thinking.is_empty();

            // Non-SSE fallback: emit one MessageDelta carrying the
            // whole frame's text up-front so the live renderer paints
            // a bubble before any later `ToolCall` resets the
            // streaming-event-id. SSE mode already drained per-chunk
            // deltas → skip to avoid double-rendering.
            if !sse_filled_text {
                let mut frame_text = String::new();
                for block in blocks {
                    if block.get("type").and_then(|v| v.as_str()) == Some("text") {
                        if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                            frame_text.push_str(t);
                        }
                    }
                }
                if !frame_text.is_empty() {
                    if !*turn_open {
                        *turn_open = true;
                        out.push(AgentEvent::TurnStarted {
                            session_id: session_id.clone(),
                        });
                    }
                    accumulated.push_str(&frame_text);
                    out.push(AgentEvent::MessageDelta {
                        session_id: session_id.clone(),
                        text: frame_text,
                    });
                }
            }

            for block in blocks {
                let kind = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match kind {
                    "text" => {
                        let Some(raw) = block.get("text").and_then(|v| v.as_str()) else {
                            continue;
                        };
                        // `<yah:followups>` stripping is a host tap (module doc).
                        let text = raw.to_string();
                        if text.is_empty() {
                            continue;
                        }
                        if !*turn_open {
                            *turn_open = true;
                            out.push(AgentEvent::TurnStarted {
                                session_id: session_id.clone(),
                            });
                        }
                        out.push(AgentEvent::AssistantText {
                            session_id: session_id.clone(),
                            text,
                        });
                    }
                    "thinking" => {
                        if sse_filled_thinking {
                            continue;
                        }
                        let Some(t) = block.get("thinking").and_then(|v| v.as_str()) else {
                            continue;
                        };
                        if t.is_empty() {
                            continue;
                        }
                        accumulated_thinking.push_str(t);
                        out.push(AgentEvent::ThinkingDelta {
                            session_id: session_id.clone(),
                            text: t.to_string(),
                        });
                    }
                    "tool_use" => {
                        let Some(id) = block.get("id").and_then(|v| v.as_str()) else {
                            continue;
                        };
                        let Some(name) = block.get("name").and_then(|v| v.as_str()) else {
                            continue;
                        };
                        let input = block
                            .get("input")
                            .cloned()
                            .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
                        let (tool_name, mut args) = normalize_claude_tool(name, input);
                        // A tool call made *inside* a Task arrives on the
                        // main stream with `parent_tool_use_id` set to the
                        // Task's own tool_use_id — so without this the
                        // transcript renders a subagent's work as if the
                        // parent had done it. Carried as an args metadata
                        // key alongside `_shape`/`_relocated` rather than a
                        // new `AgentEvent::ToolCall` field: the event is
                        // constructed in a dozen crates and this is display
                        // attribution, not a lifecycle fact. Rendering the
                        // nesting is a separate job; capturing it now means
                        // the events persisted before then aren't lost.
                        if let (Some(ptu), Some(obj)) = (parent_tool_use_id, args.as_object_mut()) {
                            obj.insert(
                                "_parentToolCallId".into(),
                                serde_json::Value::String(ptu.to_string()),
                            );
                        }
                        out.push(AgentEvent::ToolCall {
                            session_id: session_id.clone(),
                            tool_call_id: id.to_string(),
                            tool_name,
                            args,
                        });
                    }
                    _ => {}
                }
            }
        }
        // Echo of the user turn we wrote to stdin, plus tool results
        // claude is feeding back to itself. We surface tool_result
        // blocks as ToolResult breadcrumbs (paired with the ToolCall
        // emitted from the prior assistant frame); plain-text user
        // echoes are dropped — the renderer already drew the user
        // message when the user sent it.
        "user" => {
            let results = extract_user_tool_results(frame);
            // `tool_use_result` is a sibling of `message`, one per frame,
            // with nothing tying it to a particular block — so it is only
            // safe to attribute when the frame carries exactly one result.
            // Batched results (parallel tool calls flushed together) fall
            // back to the text content rather than being misattributed.
            let structured = if results.len() == 1 {
                frame.get("tool_use_result").and_then(claude_structured_result)
            } else {
                None
            };
            for (id, ok, content) in results {
                // R564-F1: session-limit tool results carry the limit message
                // as their content string. Emit the ToolResult breadcrumb
                // first (for JSONL completeness). R575-T7: also emit the
                // typed `TurnBlocked{UsageLimitHit}` so the FE recovery
                // surface (W240) offers a "wait until reset" affordance
                // instead of the session just going dark — emitted BEFORE
                // ClaudeSessionLimitHit because the caller's per-event loop
                // (this file's stream-read loop) reacts to
                // ClaudeSessionLimitHit by stopping the subprocess and
                // `break`ing immediately; ordering TurnBlocked first
                // guarantees the Blocked phase + reason land before that
                // break fires. Return early — remaining tool_results in
                // this batch are duplicates.
                if let Some(resets_at_str) = detect_session_limit(&content) {
                    out.push(AgentEvent::ToolResult {
                        session_id: session_id.clone(),
                        tool_call_id: id,
                        ok,
                        result: content,
                    });
                    out.push(AgentEvent::TurnBlocked {
                        session_id: session_id.clone(),
                        text: String::new(),
                        reason: BlockReason::UsageLimitHit {
                            resets_at: Some(resets_at_str.clone()),
                        },
                    });
                    out.push(PvEvent::SessionLimitHit {
                        session_id: session_id.clone(),
                        resets_at: resets_at_str,
                    });
                    return out.0;
                }
                out.push(AgentEvent::ToolResult {
                    session_id: session_id.clone(),
                    tool_call_id: id,
                    ok,
                    // The structured payload when claude described the
                    // outcome; the model-visible text otherwise.
                    result: match structured.clone() {
                        Some(s) => with_background_output_path(s, &content),
                        None => content,
                    },
                });
            }
        }
        // End-of-turn. `subtype` distinguishes success from various
        // failure modes; `is_error` flips us between TurnEnded and
        // TurnFailed.
        //
        // Why the message field is built carefully: claude-cli has been
        // observed to emit `is_error: true` paired with `subtype: "success"`
        // (typically after API 400s like "tool use concurrency issues"
        // that it absorbs internally before flushing). Naively recording
        // `message = error.unwrap_or(subtype)` then logs `"success"` for
        // a failure, which the renderer can't distinguish from a clean
        // turn. Preserve both signals so the persisted log is debuggable.
        "result" => {
            let is_error = frame
                .get("is_error")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let subtype = frame.get("subtype").and_then(|v| v.as_str()).unwrap_or("");
            let error_str = frame.get("error").and_then(|v| v.as_str());
            // claude puts the human-readable upstream error in `result`
            // (e.g. "Prompt is too long") and a coarse-grained tag in
            // `terminal_reason` (e.g. "blocking_limit"). For first-turn
            // upstream failures the upstream-facing `subtype` is often
            // "success" — see the wild observation at line 2087. Pull
            // both so the rendered message names the actual failure.
            let result_str = frame.get("result").and_then(|v| v.as_str()).and_then(|s| {
                let t = s.trim();
                (!t.is_empty()).then_some(t)
            });
            let terminal_reason = frame
                .get("terminal_reason")
                .and_then(|v| v.as_str())
                .and_then(|s| {
                    let t = s.trim();
                    (!t.is_empty()).then_some(t)
                });
            let text = std::mem::take(accumulated);
            accumulated_thinking.clear();
            *turn_open = false;
            // Operator-initiated soft-cancel: claude responds to a
            // `control_request:interrupt` over stdin by ending the turn with
            // `subtype:"error_during_execution"` + `terminal_reason:"aborted_streaming"`.
            // Tool rejection also emits `aborted_tools` with the same shape — both are
            // neutral user-initiated aborts, not errors. The subprocess stays alive and
            // the next send_process turn rides the same stdin.
            if terminal_reason == Some("aborted_streaming")
                || terminal_reason == Some("aborted_tools")
            {
                out.push(PvEvent::TurnAborted {
                    session_id: session_id.clone(),
                    text,
                });
                return out.0;
            }
            if is_error {
                // R262-F2: typed-block detection runs before the generic
                // TurnFailed path. The wire shape claude-cli emits for
                // context overflow (`is_error: true` + `terminal_reason:
                // "blocking_limit"` + `result: "Prompt is too long"`)
                // surfaces as `TurnBlocked` instead so the FE can show
                // the 3-action recovery modal rather than a red error pill.
                if let Some(reason) = yah_runner_core::block_detector::claude_cli(frame) {
                    out.push(AgentEvent::TurnBlocked {
                        session_id: session_id.clone(),
                        text,
                        reason,
                    });
                    return out.0;
                }
                let useful_subtype =
                    (!subtype.is_empty() && subtype != "success").then_some(subtype);
                let message = match (error_str, useful_subtype, result_str, terminal_reason) {
                    (Some(e), Some(s), _, _) => format!("{s}: {e}"),
                    (Some(e), None, _, _) => e.to_string(),
                    (None, _, Some(r), Some(tr)) => format!("{tr}: {r}"),
                    (None, _, Some(r), None) => r.to_string(),
                    (None, Some(s), None, Some(tr)) => format!("{s} ({tr})"),
                    (None, Some(s), None, None) => s.to_string(),
                    (None, None, None, Some(tr)) => tr.to_string(),
                    // is_error with no useful discriminator — record the
                    // contradiction explicitly rather than fabricating a
                    // misleading "success" message.
                    (None, None, None, None) => {
                        "claude returned is_error with no subtype/error".to_string()
                    }
                };
                out.push(AgentEvent::TurnFailed {
                    session_id: session_id.clone(),
                    text,
                    message,
                });
            } else {
                let usage = extract_result_usage(frame).map(|mut u| {
                    // Embed the per-call peak so sessionStore replay can use
                    // it for the context ring instead of the inflated cumulative
                    // cache_read_tokens total from the result frame.
                    u.peak_context_tokens = *peak_context_tokens;
                    u
                });
                *peak_context_tokens = None;
                out.push(AgentEvent::TurnEnded {
                    session_id: session_id.clone(),
                    text,
                    stop_reason: Some(subtype.to_string()),
                    usage,
                    content_blocks: None,
                });
            }
        }
        // Everything claude emits that we don't translate. This used to
        // be a bare `_ => {}`: four frame types (`rate_limit_event`,
        // `system/task_*`, …) accumulated on the wire unnoticed because
        // dropping them was indistinguishable from them not existing.
        // Log the first sighting of each so a claude-cli upgrade that
        // adds a type leaves a trace instead of silence. Deduped by
        // `type` (and `subtype`, which is where the interesting ones
        // hide) so a per-token frame kind can't flood the log.
        other => note_untranslated_frame(other, frame.get("subtype").and_then(|v| v.as_str())),
    }
    out.0
}

/// R713 (Defect 3): what [`crate::frame::StreamReader`] must emit when its stdout loop ends.
///
/// A terminal event is the ONLY signal a dispatcher gets that a dispatched
/// child is finished — the daemon's `dispatch_result_handle_event`
/// (app/yah/cli/src/camp.rs) banks a `dispatch_result` off `TurnEnded` /
/// `TurnFailed` / `Released` / `SessionEnded` and nothing else. So a
/// subprocess that died WITHOUT an open turn — a missing CLI binary, an auth
/// failure on the first connect, a config error `claude` rejects before it
/// emits any frame — used to end this reader in silence. `party.dispatch` had
/// already returned success with a `pending_collection`, and the collection
/// could never arrive.
///
/// `died_before_returning` is the caller's conjunction of two facts, and both
/// matter:
///
///  - **The exit was unexpected.** the host's stop path takes
///    `ProcessSession::child` under the session lock before it aborts the
///    reader task, and `remove_and_stop` drops the registry row first, so a
///    dead-or-absent child handle means somebody meant to stop us.
///  - **Nothing was ever returned.** A subprocess that completed turns and
///    then went away has already discharged its dispatch obligation via the
///    last `TurnEnded`; synthesizing another terminal event would post a
///    second, contradictory `dispatch_result` (the ledger only dedupes
///    teardown-shaped events, and this one is not).
pub fn reader_exit_event(
    session_id: &SessionId,
    turn_open: bool,
    died_before_returning: bool,
    accumulated: String,
) -> Option<AgentEvent> {
    if turn_open {
        return Some(AgentEvent::TurnFailed {
            session_id: session_id.clone(),
            text: accumulated,
            message: "claude subprocess closed mid-turn".into(),
        });
    }
    if died_before_returning {
        return Some(AgentEvent::TurnFailed {
            session_id: session_id.clone(),
            text: String::new(),
            message: "claude subprocess exited before its first turn — no frames were produced. \
                      Check this subclass's CLI binary and engine credential."
                .into(),
        });
    }
    None
}

/// Pull per-API-call token usage out of one `assistant` entry from claude's
/// own transcript JSONL. The transcript carries the definitive per-call usage
/// block (`message.usage`) — in particular the per-call `input_tokens` before
/// any multi-call accumulation. Returns `None` when the entry has no usage
/// object or every token field is absent (so we don't emit/persist an empty
/// delta). `peak_context_tokens` is the per-call context fill
/// (`input + cache_read + cache_write`) so the live Ring tracks fill mid-turn.
pub fn assistant_entry_usage(v: &serde_json::Value) -> Option<TurnUsage> {
    let u = v.get("message").and_then(|m| m.get("usage"))?;
    let read_u32 = |k: &str| u.get(k).and_then(|x| x.as_u64()).map(|n| n as u32);
    let inp = read_u32("input_tokens");
    let out = read_u32("output_tokens");
    let cr = read_u32("cache_read_input_tokens");
    let cw = read_u32("cache_creation_input_tokens");
    if inp.is_none() && out.is_none() && cr.is_none() && cw.is_none() {
        return None;
    }
    let peak = match (inp, cr, cw) {
        (None, None, None) => None,
        _ => Some(inp.unwrap_or(0) + cr.unwrap_or(0) + cw.unwrap_or(0)),
    };
    Some(TurnUsage {
        input_tokens: inp,
        output_tokens: out,
        cache_read_tokens: cr,
        cache_write_tokens: cw,
        thinking_tokens: None,
        peak_context_tokens: peak,
    })
}

/// One `debug!` per distinct `type[/subtype]` seen, for the lifetime of
/// the process. The set is unbounded in principle but bounded in
/// practice by the wrapped CLIs' frame vocabularies — a few dozen strings.
///
/// Shared by [`translate_frame`] and `translate_codex_frame` (desktop). The latter
/// passes codex's `item.type` as the `subtype`, which is where that wire
/// format hides its variation the same way claude hides it under `system`.
fn note_untranslated_frame(kind: &str, subtype: Option<&str>) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

    let key = match subtype {
        Some(s) => format!("{kind}/{s}"),
        None => kind.to_string(),
    };
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let Ok(mut set) = seen.lock() else { return };
    if set.insert(key.clone()) {
        tracing::debug!(frame = %key, "untranslated claude stream-json frame type");
    }
}

/// Extract token usage from a `result` frame's `usage` object.
/// `result.usage` is the same shape as Anthropic's `/v1/messages`
/// `usage`, modulo extra fields claude adds (server_tool_use,
/// service_tier) that we don't surface.
///
/// Returns `None` only when the frame has no `usage` object at all
/// — partial-usage frames (e.g. only `input_tokens` populated) yield
/// a `TurnUsage` with the missing fields as `None`.
fn extract_result_usage(frame: &serde_json::Value) -> Option<TurnUsage> {
    let usage = frame.get("usage")?;
    let read_u32 = |k: &str| usage.get(k).and_then(|v| v.as_u64()).map(|n| n as u32);
    Some(TurnUsage {
        input_tokens: read_u32("input_tokens"),
        output_tokens: read_u32("output_tokens"),
        cache_read_tokens: read_u32("cache_read_input_tokens"),
        cache_write_tokens: read_u32("cache_creation_input_tokens"),
        // claude doesn't separately expose thinking-token totals on
        // result frames; the budget consumed gauge is approximated
        // from output_tokens + the configured think budget the host
        // already knows about.
        thinking_tokens: None,
        // Caller (result arm of translate_frame) overwrites this with the
        // per-call peak tracked across assistant frames.
        peak_context_tokens: None,
    })
}

/// Per-chunk usage from a `message_delta` SSE event. Same shape as
/// `result.usage` but typically only `output_tokens` is populated —
/// the cache/input numbers ride `message_start` once and don't repeat.
/// Returns `None` when no recognized field is present (so we don't
/// fan a useless empty TurnUsageDelta to the renderer).
fn extract_partial_usage(usage: &serde_json::Value) -> Option<TurnUsage> {
    let read_u32 = |k: &str| usage.get(k).and_then(|v| v.as_u64()).map(|n| n as u32);
    let acc = TurnUsage {
        input_tokens: read_u32("input_tokens"),
        output_tokens: read_u32("output_tokens"),
        cache_read_tokens: read_u32("cache_read_input_tokens"),
        cache_write_tokens: read_u32("cache_creation_input_tokens"),
        thinking_tokens: None,
        peak_context_tokens: None,
    };
    if acc.input_tokens.is_none()
        && acc.output_tokens.is_none()
        && acc.cache_read_tokens.is_none()
        && acc.cache_write_tokens.is_none()
    {
        None
    } else {
        Some(acc)
    }
}

/// Normalize a claude-cli built-in tool name and args to yah-internal canonical form.
/// Maps PascalCase names to lowercase and rewrites arg field names so the renderer
/// sees one consistent shape per tool regardless of which runner emitted the event.
/// Non-claude-cli names (already canonical or user-defined MCP tools) pass through unchanged.
///
/// Mapping table (claude wire → yah-internal canonical):
///   Read → read            file_path → path
///   Edit / MultiEdit → edit  old_string + new_string → diff (synthesised unified diff)
///   Write → write           offset + limit → range: [start, end]
///   Bash → bash             command → cmd
///   Grep → grep             Glob pattern → path (for headline)
///   Glob / LS → list_dir
pub fn normalize_claude_tool(name: &str, mut args: serde_json::Value) -> (String, serde_json::Value) {
    let canonical = match name {
        "Read" => "read",
        "Edit" | "MultiEdit" => "edit",
        "Write" => "write",
        "Bash" => "bash",
        "Grep" => "grep",
        "Glob" | "LS" => "list_dir",
        _ => return (name.to_string(), args),
    };

    if let Some(obj) = args.as_object_mut() {
        // file_path → path (Read, Edit, Write all use file_path in claude's wire shape)
        if let Some(fp) = obj.remove("file_path") {
            obj.entry("path").or_insert(fp);
        }
        // command → cmd (Bash built-in sends `command`; yah uses `cmd`)
        if let Some(cmd) = obj.remove("command") {
            obj.entry("cmd").or_insert(cmd);
        }
        match canonical {
            "read" => {
                // offset + limit → range: [offset, offset+limit] for the ReadResult renderer
                let offset = obj.remove("offset");
                let limit = obj.remove("limit");
                if let Some(off) = offset {
                    let start = off.as_u64().unwrap_or(0);
                    let end = start
                        .saturating_add(limit.as_ref().and_then(|v| v.as_u64()).unwrap_or(200));
                    obj.insert("range".into(), serde_json::json!([start, end]));
                }
            }
            "edit" => {
                // old_string + new_string → diff: synthesised minimal unified diff.
                // Apply remains disabled until R076-F2 supplies full before/after bytes.
                let old = obj.remove("old_string");
                let new = obj.remove("new_string");
                if let (Some(old_v), Some(new_v)) = (old, new) {
                    if let (Some(old_s), Some(new_s)) = (old_v.as_str(), new_v.as_str()) {
                        let old_lines: Vec<&str> = old_s.split('\n').collect();
                        let new_lines: Vec<&str> = new_s.split('\n').collect();
                        let header = format!("@@ -1,{} +1,{} @@", old_lines.len(), new_lines.len());
                        let mut parts = vec![header];
                        parts.extend(old_lines.iter().map(|l| format!("-{l}")));
                        parts.extend(new_lines.iter().map(|l| format!("+{l}")));
                        obj.insert("diff".into(), serde_json::Value::String(parts.join("\n")));
                    }
                }
            }
            // Glob sends `pattern`; LS sends `path`. Surface pattern as path so the
            // headline renders the glob pattern instead of falling back to ".".
            "list_dir" if !obj.contains_key("path") => {
                if let Some(pat) = obj.get("pattern").cloned() {
                    obj.insert("path".into(), pat);
                }
            }
            // `bash`: the yah desktop host attaches its `bash_ast` `_shape` /
            // `_relocated` summary in its stream tap (module doc).
            _ => {}
        }
    }

    (canonical.to_string(), args)
}

/// Longest a genuine limit message runs. The real wire text is ~60 chars; the
/// cap is generous enough for wording drift and far below any tool payload that
/// merely *mentions* the phrase.
const SESSION_LIMIT_MAX_LEN: usize = 160;

/// How far into the string the phrase may start. Covers the pronoun prefix in
/// every apostrophe variant (`You've ` = 7 bytes, `You’ve ` = 9) plus slack.
const SESSION_LIMIT_PHRASE_MAX_OFFSET: usize = 16;

/// R564-F1: detect whether a tool-result content value carries the claude-cli
/// subscription session-limit message. Returns the "resets at" portion of the
/// message (e.g. `"9:20pm (America/Los_Angeles)"`) when matched, or `None`.
///
/// The full wire text is `"You've hit your session limit · resets <time> (<tz>)"`
/// and it arrives as the *entire* tool_result string — never embedded in a
/// larger payload.
///
/// **This matcher used to be self-detonating.** It matched the inner phrase
/// anywhere in the content and accepted any remainder after `" resets "` that
/// contained a `/`. So a `rg "session limit"` over this very file produced a
/// tool result that (a) contained the phrase, from the doc comment and the tests
/// below, and (b) was full of `/Users/…` paths after it — tripping a camp-wide
/// lockout with the tail of the grep output as the bogus "resets at" string. Any
/// agent grepping this module took the whole camp down until the frontend's
/// best-effort clock parse expired the banner.
///
/// The match is therefore anchored end to end. All three guards are load-bearing
/// against the self-grep case; none alone is sufficient:
///
///   * the trimmed content must be short — a real message is ~60 chars;
///   * the phrase must appear at the *start*, not merely somewhere inside;
///   * the remainder after `" resets "` must be exactly a clock time plus a
///     parenthesised zone, with nothing trailing it.
pub fn detect_session_limit(content: &serde_json::Value) -> Option<String> {
    let s = content.as_str()?.trim();
    if s.len() > SESSION_LIMIT_MAX_LEN {
        return None;
    }
    if s.find("hit your session limit")? > SESSION_LIMIT_PHRASE_MAX_OFFSET {
        return None;
    }
    let resets_idx = s.find(" resets ")?;
    let after = s[resets_idx + " resets ".len()..].trim();
    if !is_reset_token(after) {
        return None;
    }
    Some(after.to_string())
}

/// Whole-string match for the reset tail: `9:20pm (America/Los_Angeles)`. The
/// meridiem is optional (24-hour clocks) and the zone must be a slash-separated
/// IANA-style name, which is what keeps a bare `" resets "` in ordinary prose
/// from matching.
fn is_reset_token(s: &str) -> bool {
    // hh:mm
    let (hh, rest) = take_digits(s, 1, 2);
    if hh.is_empty() {
        return false;
    }
    let Some(rest) = rest.strip_prefix(':') else {
        return false;
    };
    let (mm, rest) = take_digits(rest, 2, 2);
    if mm.len() != 2 {
        return false;
    }
    // Optional meridiem.
    let rest = rest.trim_start();
    let rest = if rest.len() >= 2 && matches!(rest[..2].to_ascii_lowercase().as_str(), "am" | "pm") {
        &rest[2..]
    } else {
        rest
    };
    // Parenthesised zone, and nothing after it.
    let rest = rest.trim_start();
    let Some(zone) = rest.strip_prefix('(').and_then(|z| z.strip_suffix(')')) else {
        return false;
    };
    zone.contains('/')
        && !zone.starts_with('/')
        && !zone.ends_with('/')
        && zone
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '+' | '-' | '/'))
}

/// Split off between `min` and `max` leading ASCII digits, returning
/// `(digits, remainder)`. `digits` is empty when fewer than `min` are present.
fn take_digits(s: &str, min: usize, max: usize) -> (&str, &str) {
    let n = s.bytes().take(max).take_while(u8::is_ascii_digit).count();
    if n < min {
        return ("", s);
    }
    s.split_at(n)
}

/// Pull `tool_result` content blocks out of a `user` whole-message
/// frame. Returns `(tool_use_id, ok, content_value)` per block.
/// Anthropic shape:
/// `{"message":{"content":[{"type":"tool_result","tool_use_id":"…","content":…,"is_error":false}, …]}}`.
/// `content` may be a string or an array of content sub-blocks; we
/// pass it through verbatim as a JSON value so the renderer can decide
/// how to fold it.
fn extract_user_tool_results(frame: &serde_json::Value) -> Vec<(String, bool, serde_json::Value)> {
    let Some(blocks) = frame
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_array())
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for block in blocks {
        if block.get("type").and_then(|v| v.as_str()) != Some("tool_result") {
            continue;
        }
        let Some(id) = block.get("tool_use_id").and_then(|v| v.as_str()) else {
            continue;
        };
        let is_error = block
            .get("is_error")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let content = block
            .get("content")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        out.push((id.to_string(), !is_error, content));
    }
    out
}

/// Byte cap on the pre-edit file snapshot forwarded out of
/// `tool_use_result.originalFile`. Nothing renders it today — it exists
/// so DiffCard's Apply (R076-F2) has a real "before" to diff against
/// instead of the synthesised one-hunk approximation. Since it is
/// carried rather than consumed, cap it: past this size the snapshot is
/// dropped and `originalFileTruncated` marks why, because a *partial*
/// before-image is worse than none — Apply would silently reconstruct
/// the wrong file. The diff itself is never truncated; it is bounded by
/// the edit, not the file.
const ORIGINAL_FILE_MAX_BYTES: usize = 512 * 1024;

/// Fold claude's out-of-band `tool_use_result` into the shape yah's
/// renderers already expect.
///
/// claude-cli emits the structured outcome of its own built-in tools as
/// a sibling of `message` on the `user` frame — *not* inside the
/// `tool_result` block, whose `content` is only the text the model gets
/// to read. We had been reading the text and discarding this, which is
/// why `BashOutput` never had a stdout/stderr split and `DiffCard` had
/// to synthesise a fake unified diff out of `old_string`/`new_string`.
///
/// There is no tool name on a `user` frame — only a `tool_use_id` — so
/// the mapping dispatches on the payload's own shape. Each arm returns
/// the key names the corresponding renderer already destructures
/// (`Message.tsx`), so no renderer needs to learn claude's vocabulary.
/// An unrecognised object passes through whole; `None` means "keep the
/// text content", which is the pre-existing behaviour and the fallback
/// for every tool claude doesn't describe (and for every subagent-
/// internal call, where this field arrives null).
fn claude_structured_result(tur: &serde_json::Value) -> Option<serde_json::Value> {
    let obj = tur.as_object()?;

    // Edit / Write / MultiEdit.
    if let Some(hunks) = obj.get("structuredPatch").and_then(|v| v.as_array()) {
        let (diff, additions, deletions) = render_structured_patch(hunks);
        let mut out = serde_json::Map::new();
        out.insert("diff".into(), serde_json::Value::String(diff));
        out.insert("additions".into(), additions.into());
        out.insert("deletions".into(), deletions.into());
        if let Some(p) = obj.get("filePath") {
            out.insert("path".into(), p.clone());
        }
        match obj.get("originalFile").and_then(|v| v.as_str()) {
            Some(s) if s.len() <= ORIGINAL_FILE_MAX_BYTES => {
                out.insert("originalFile".into(), serde_json::Value::String(s.into()));
            }
            Some(_) => {
                out.insert("originalFileTruncated".into(), true.into());
            }
            None => {}
        }
        return Some(serde_json::Value::Object(out));
    }

    // Bash. No exit code rides the wire — `interrupted` is the only
    // failure discriminator claude gives us, and `ok` already carries
    // `is_error`. Leave `exit` absent rather than inventing a 0 that
    // ToolFrame would render as a confident "exited clean".
    //
    // R775-B6: `backgroundTaskId` is not cosmetic. It is the only trace of a
    // command that ran in (or was moved to) the background once this fold has
    // replaced the marker text, and the daemon's return defer keys on it —
    // dropping it here made every backgrounded build invisible to that defer.
    if obj.contains_key("stdout") || obj.contains_key("stderr") {
        let mut out = serde_json::Map::new();
        for k in ["stdout", "stderr", "interrupted", "backgroundTaskId"] {
            if let Some(v) = obj.get(k) {
                out.insert(k.into(), v.clone());
            }
        }
        return Some(serde_json::Value::Object(out));
    }

    // Read. `file.content` is the whole file and no renderer displays it
    // — ReadResult shows the line count and a truncation hint — so it is
    // summarised here rather than forwarded and stripped downstream.
    if let Some(file) = obj.get("file").and_then(|v| v.as_object()) {
        let num = file.get("numLines").and_then(|v| v.as_u64());
        let total = file.get("totalLines").and_then(|v| v.as_u64());
        let mut out = serde_json::Map::new();
        if let Some(n) = num {
            out.insert("lines".into(), n.into());
        }
        if let Some(t) = total {
            out.insert("totalLines".into(), t.into());
        }
        if let Some(c) = file.get("content").and_then(|v| v.as_str()) {
            out.insert("bytes".into(), c.len().into());
        }
        if let (Some(n), Some(t)) = (num, total) {
            out.insert("truncated".into(), (n < t).into());
        }
        return Some(serde_json::Value::Object(out));
    }

    // Task (subagent). `prompt` is already on the ToolCall args and
    // `content` is the transcript the parent model reads; both are large
    // and neither is rendered, so the pass-through drops them and keeps
    // the accounting.
    if obj.contains_key("totalTokens") || obj.contains_key("toolStats") {
        let mut out = obj.clone();
        out.remove("prompt");
        out.remove("content");
        return Some(serde_json::Value::Object(out));
    }

    Some(tur.clone())
}

/// R775-B6: the background task's output path exists only in the model-visible
/// text ("… Output is being written to: <path>. …"), which the structured fold
/// replaces. Lift it onto the folded result as `backgroundOutputPath` so the
/// daemon's liveness check still has a file to stat.
fn with_background_output_path(
    structured: serde_json::Value,
    content: &serde_json::Value,
) -> serde_json::Value {
    let serde_json::Value::Object(mut obj) = structured else {
        return structured;
    };
    if obj.contains_key("backgroundTaskId") && !obj.contains_key("backgroundOutputPath") {
        if let Some(p) = background_task_output_path_from_result(content)
        {
            obj.insert(
                "backgroundOutputPath".into(),
                serde_json::Value::String(p.to_string_lossy().into_owned()),
            );
        }
    }
    serde_json::Value::Object(obj)
}

/// Flatten claude's `structuredPatch` hunks into a unified diff, and
/// count the +/- lines while we're walking them. The hunk shape is
/// `{oldStart, oldLines, newStart, newLines, lines: ["-old", "+new", " ctx"]}`
/// — i.e. everything a `@@` header needs, already parsed.
fn render_structured_patch(hunks: &[serde_json::Value]) -> (String, usize, usize) {
    let mut buf = String::new();
    let (mut additions, mut deletions) = (0usize, 0usize);
    const NO_LINES: &Vec<serde_json::Value> = &Vec::new();
    for h in hunks {
        let n = |k: &str| h.get(k).and_then(|v| v.as_u64()).unwrap_or(0);
        buf.push_str(&format!(
            "@@ -{},{} +{},{} @@\n",
            n("oldStart"),
            n("oldLines"),
            n("newStart"),
            n("newLines")
        ));
        for line in h.get("lines").and_then(|v| v.as_array()).unwrap_or(NO_LINES) {
            let Some(s) = line.as_str() else { continue };
            match s.as_bytes().first() {
                Some(b'+') => additions += 1,
                Some(b'-') => deletions += 1,
                _ => {}
            }
            buf.push_str(s);
            buf.push('\n');
        }
    }
    (buf, additions, deletions)
}


/// Extract the harness output path from a bash `ToolResult` marker:
/// `… Output is being written to: <path>. …`. The path never contains a
/// space in any observed shape; exactly one sentence-terminating `.` is
/// trimmed. Part of R951 — copied from
/// `agent_tools::party_tools::background_task_output_path_from_result`.
fn background_task_output_path_from_result(result: &serde_json::Value) -> Option<std::path::PathBuf> {
    if let Some(p) = result.get("backgroundOutputPath").and_then(serde_json::Value::as_str) {
        return (!p.is_empty()).then(|| std::path::PathBuf::from(p));
    }
    let text = result
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| result.to_string());
    const MARKER: &str = "Output is being written to: ";
    let start = text.find(MARKER)? + MARKER.len();
    let raw: String = text[start..].chars().take_while(|c| !c.is_whitespace()).collect();
    let trimmed = raw.strip_suffix('.').unwrap_or(&raw);
    (!trimmed.is_empty()).then(|| std::path::PathBuf::from(trimmed))
}

#[cfg(test)]
mod tests {
    //! Part of R951 — copied from the desktop `agent_process.rs` tests
    //! (R951-F3 deletes the originals). The `<yah:followups>` and
    //! `bash_ast` tests stay with the desktop host tap that now owns them.
    use super::*;
    use serde_json::json;

    fn sid() -> SessionId {
        SessionId::new("session:deadbeef")
    }

    /// Core-only view of the translator, so the copied assertions read as
    /// they did on the desktop. Tests of the pv-only variants call
    /// `super::translate_frame` directly.
    fn translate_frame(
        frame: &serde_json::Value,
        session_id: &SessionId,
        turn_open: &mut bool,
        accumulated: &mut String,
        accumulated_thinking: &mut String,
        peak_context_tokens: &mut Option<u32>,
    ) -> Vec<AgentEvent> {
        super::translate_frame(
            frame,
            session_id,
            turn_open,
            accumulated,
            accumulated_thinking,
            peak_context_tokens,
        )
        .into_iter()
        .filter_map(PvEvent::into_core)
        .collect()
    }

    /// The regression that motivated R713: `party.dispatch` returns a success
    /// payload carrying `followup.pending_collection`, and the ONLY thing that
    /// can ever discharge that obligation is a terminal event reaching the
    /// daemon's `dispatch_result_handle_event`. A subprocess that dies before
    /// its first frame — wrong CLI binary, revoked credential rejected on the
    /// first connect — has `turn_open == false`, and used to produce nothing at
    /// all. The dispatcher waited forever.
    #[test]
    fn a_subprocess_that_dies_before_its_first_turn_still_reports_a_failure() {
        let ev = reader_exit_event(&sid(), false, true, String::new())
            .expect("early subprocess death must surface a terminal event");
        match ev {
            AgentEvent::TurnFailed {
                session_id,
                message,
                ..
            } => {
                assert_eq!(session_id, sid());
                assert!(
                    message.contains("before its first turn"),
                    "message should name the failure mode: {message}"
                );
            }
            other => panic!("expected TurnFailed, got {other:?}"),
        }
    }

    /// Pre-existing behaviour, pinned: a crash mid-turn carries the partial
    /// text out with it so the return isn't empty.
    #[test]
    fn a_subprocess_that_dies_mid_turn_carries_its_partial_text_out() {
        let ev = reader_exit_event(&sid(), true, true, "half an answ".into())
            .expect("mid-turn death must surface a terminal event");
        match ev {
            AgentEvent::TurnFailed { text, message, .. } => {
                assert_eq!(text, "half an answ");
                assert!(message.contains("mid-turn"), "{message}");
            }
            other => panic!("expected TurnFailed, got {other:?}"),
        }
    }

    /// The other side of the discriminator, and the reason it is a conjunction:
    /// an ORDERLY teardown (`stop_process` took the child handle) and a
    /// subprocess that already returned at least one turn before exiting both
    /// stay silent. Emitting for either would post a second, contradictory
    /// `dispatch_result` for work that already reported.
    #[test]
    fn an_orderly_teardown_or_an_already_returned_child_emits_nothing() {
        assert!(reader_exit_event(&sid(), false, false, String::new()).is_none());
    }

    /// …but a teardown that interrupts a live turn is still a failure the
    /// dispatcher needs to hear about — `turn_open` wins over `child_alive`.
    #[test]
    fn a_teardown_mid_turn_still_reports() {
        assert!(reader_exit_event(&sid(), true, false, String::new()).is_some());
    }

    #[test]
    fn translate_frame_emits_turn_started_on_message_start_stream_event() {
        let frame = json!({
            "type": "stream_event",
            "event": { "type": "message_start" },
        });
        let mut turn_open = false;
        let mut acc = String::new();
        let events = translate_frame(
            &frame,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None,
        );
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], AgentEvent::TurnStarted { .. }));
        assert!(turn_open);
    }

    #[test]
    fn translate_frame_folds_content_block_delta_into_message_delta() {
        let frame = json!({
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": { "type": "text_delta", "text": "hello" },
            },
        });
        let mut turn_open = true;
        let mut acc = String::new();
        let events = translate_frame(
            &frame,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None,
        );
        assert_eq!(events.len(), 1);
        match &events[0] {
            AgentEvent::MessageDelta { text, .. } => assert_eq!(text, "hello"),
            other => panic!("unexpected event {other:?}"),
        }
        assert_eq!(acc, "hello");
    }

    #[test]
    fn translate_frame_result_success_emits_turn_ended_with_subtype() {
        let frame = json!({
            "type": "result",
            "subtype": "success",
            "is_error": false,
        });
        let mut turn_open = true;
        let mut acc = String::from("partial answer");
        let events = translate_frame(
            &frame,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None,
        );
        assert_eq!(events.len(), 1);
        match &events[0] {
            AgentEvent::TurnEnded {
                text,
                stop_reason,
                usage,
                ..
            } => {
                assert_eq!(text, "partial answer");
                assert_eq!(stop_reason.as_deref(), Some("success"));
                assert!(usage.is_none(), "result frame without usage → None");
            }
            other => panic!("unexpected event {other:?}"),
        }
        assert!(!turn_open);
        assert!(acc.is_empty(), "accumulator drained on TurnEnded");
    }

    #[test]
    fn translate_frame_result_carries_usage_from_result_frame() {
        // claude's stream-json result frame carries the full usage
        // object — same shape as Anthropic's /v1/messages, with extra
        // fields we don't surface (server_tool_use, service_tier).
        // We pull the four wire-shaped numbers and ignore the rest.
        let frame = json!({
            "type": "result",
            "subtype": "success",
            "is_error": false,
            "usage": {
                "input_tokens": 1234,
                "output_tokens": 567,
                "cache_read_input_tokens": 1000,
                "cache_creation_input_tokens": 0,
                "server_tool_use": { "web_search_requests": 0 },
            },
        });
        let mut turn_open = true;
        let mut acc = String::from("answer");
        let events = translate_frame(
            &frame,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None,
        );
        assert_eq!(events.len(), 1);
        match &events[0] {
            AgentEvent::TurnEnded { usage: Some(u), .. } => {
                assert_eq!(u.input_tokens, Some(1234));
                assert_eq!(u.output_tokens, Some(567));
                assert_eq!(u.cache_read_tokens, Some(1000));
                assert_eq!(u.cache_write_tokens, Some(0));
                assert!(u.thinking_tokens.is_none());
            }
            other => panic!("expected TurnEnded with Some(usage), got {other:?}"),
        }
    }

    #[test]
    fn translate_frame_result_with_is_error_emits_turn_failed() {
        // Claude flips `is_error` for `error_max_turns`,
        // `error_during_execution`, etc. We map all of those to
        // TurnFailed so the renderer's UI doesn't have to switch
        // on subtype — the carrier event already says "this turn
        // didn't complete cleanly".
        let frame = json!({
            "type": "result",
            "subtype": "error_max_turns",
            "is_error": true,
        });
        let mut turn_open = true;
        let mut acc = String::from("got this far");
        let events = translate_frame(
            &frame,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None,
        );
        assert_eq!(events.len(), 1);
        match &events[0] {
            AgentEvent::TurnFailed { text, message, .. } => {
                assert_eq!(text, "got this far");
                assert_eq!(message, "error_max_turns");
            }
            other => panic!("unexpected event {other:?}"),
        }
    }

    /// Soft-cancel round-trip: when the host writes
    /// `control_request:interrupt` over stdin, claude responds with a result
    /// frame carrying `terminal_reason: "aborted_streaming"` (and
    /// `subtype: "error_during_execution"`, `is_error: true`). Tool rejection
    /// emits `aborted_tools` instead. Both become `TurnAborted` — operator-
    /// initiated, neutral pill — NOT `TurnFailed`, which the UI renders as a
    /// red error with a Retry CTA. Verified empirically against claude-cli
    /// 2.1.143 by the /tmp/spike_followup.sh transcript.
    #[test]
    fn translate_frame_result_aborted_streaming_emits_turn_aborted_not_failed() {
        let frame = json!({
            "type": "result",
            "subtype": "error_during_execution",
            "is_error": true,
            "terminal_reason": "aborted_streaming",
        });
        let mut turn_open = true;
        let mut acc = String::from("partial reply before the cancel landed");
        let events = super::translate_frame(
            &frame,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None,
        );
        assert_eq!(events.len(), 1);
        match &events[0] {
            PvEvent::TurnAborted { text, .. } => {
                assert_eq!(text, "partial reply before the cancel landed");
            }
            other => panic!("expected TurnAborted, got {other:?}"),
        }
        // Accumulator drained (mem::take); turn closed.
        assert!(acc.is_empty());
        assert!(!turn_open);
    }

    #[test]
    fn translate_frame_result_aborted_tools_emits_turn_aborted_not_failed() {
        let frame = json!({
            "type": "result",
            "subtype": "error_during_execution",
            "is_error": true,
            "terminal_reason": "aborted_tools",
        });
        let mut turn_open = true;
        let mut acc = String::from("partial reply before tool was rejected");
        let events = super::translate_frame(
            &frame,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None,
        );
        assert_eq!(events.len(), 1);
        match &events[0] {
            PvEvent::TurnAborted { text, .. } => {
                assert_eq!(text, "partial reply before tool was rejected");
            }
            other => panic!("expected TurnAborted, got {other:?}"),
        }
        assert!(acc.is_empty());
        assert!(!turn_open);
    }

    #[test]
    fn translate_frame_result_is_error_with_success_subtype_records_contradiction() {
        // Observed in the wild: claude-cli has emitted `is_error: true`
        // with `subtype: "success"` after absorbing an API 400 mid-turn.
        // The persisted message must NOT be "success" — that's how the
        // renderer ends up showing successful-looking failures.
        let frame = json!({
            "type": "result",
            "subtype": "success",
            "is_error": true,
        });
        let mut turn_open = true;
        let mut acc = String::from("partial");
        let events = translate_frame(
            &frame,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None,
        );
        assert_eq!(events.len(), 1);
        match &events[0] {
            AgentEvent::TurnFailed { message, .. } => {
                assert_ne!(
                    message, "success",
                    "must not record contradictory \"success\" as the failure reason"
                );
                assert!(!message.is_empty());
            }
            other => panic!("unexpected event {other:?}"),
        }
    }

    #[test]
    fn translate_frame_result_is_error_routes_prompt_too_long_to_turn_blocked() {
        // Captured from a real claude --print run hitting a prompt-too-long
        // upstream rejection: claude sets `is_error: true`, `subtype: "success"`,
        // and puts the human-readable message in `result` plus a coarse tag
        // in `terminal_reason`.
        //
        // R262-F2 typed-block routing: this shape is the canonical
        // context-overflow signal from claude-cli; translate_frame now
        // surfaces it as `TurnBlocked { reason: ContextOverflow }` instead
        // of a generic `TurnFailed`, so the FE can render the 3-action
        // recovery modal (Compact / HandOff / Abandon) rather than a red
        // error pill. Non-overflow `is_error` frames still ride TurnFailed
        // (covered by `translate_frame_result_with_is_error_emits_turn_failed`).
        let frame = json!({
            "type": "result",
            "subtype": "success",
            "is_error": true,
            "result": "Prompt is too long",
            "terminal_reason": "blocking_limit",
        });
        let mut turn_open = true;
        let mut acc = String::new();
        let events = translate_frame(
            &frame,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None,
        );
        assert_eq!(events.len(), 1);
        match &events[0] {
            AgentEvent::TurnBlocked { reason, .. } => {
                assert_eq!(reason, &BlockReason::ContextOverflow);
            }
            other => panic!("expected TurnBlocked for prompt-too-long frame, got {other:?}"),
        }
    }

    #[test]
    fn translate_frame_assistant_full_message_when_no_partial_text() {
        // Without --include-partial-messages claude only emits one
        // assistant frame per reply. We synthesize a fallback
        // MessageDelta so the live renderer paints, plus a durable
        // AssistantText so the JSONL records the block boundary.
        let frame = json!({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    { "type": "text", "text": "non-streamed reply" },
                ],
            },
        });
        let mut turn_open = false;
        let mut acc = String::new();
        let events = translate_frame(
            &frame,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None,
        );
        // TurnStarted + MessageDelta (live fallback) + AssistantText (durable).
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], AgentEvent::TurnStarted { .. }));
        match &events[1] {
            AgentEvent::MessageDelta { text, .. } => {
                assert_eq!(text, "non-streamed reply")
            }
            other => panic!("unexpected event {other:?}"),
        }
        match &events[2] {
            AgentEvent::AssistantText { text, .. } => {
                assert_eq!(text, "non-streamed reply")
            }
            other => panic!("unexpected event {other:?}"),
        }
    }

    #[test]
    fn translate_frame_skips_message_delta_when_partial_already_accumulated() {
        // With --include-partial-messages the whole-message frame is a
        // duplicate of the deltas we already streamed; skip the
        // fallback MessageDelta. AssistantText still fires — it is the
        // durable record persisted to the JSONL, distinct from the
        // per-chunk MessageDeltas which are wire-only.
        let frame = json!({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{ "type": "text", "text": "duplicate" }],
            },
        });
        let mut turn_open = true;
        let mut acc = String::from("duplicate");
        let events = translate_frame(
            &frame,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None,
        );
        assert_eq!(
            events.len(),
            1,
            "no MessageDelta duplicate when acc has text"
        );
        match &events[0] {
            AgentEvent::AssistantText { text, .. } => assert_eq!(text, "duplicate"),
            other => panic!("expected AssistantText, got {other:?}"),
        }
    }

    #[test]
    fn translate_frame_interleaves_assistant_text_with_tool_calls() {
        // Multi-block assistant frames must preserve text-vs-tool order
        // in the AssistantText / ToolCall stream so the JSONL replays
        // with text bubbles in the correct positions. Regression
        // guard for R197-B1: the prior MessageDelta-accumulator-then-
        // ToolCall path glued every interstitial narration into a
        // single TurnEnded.text blob.
        let frame = json!({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    { "type": "text", "text": "first narration" },
                    { "type": "tool_use", "id": "tc_1", "name": "Read", "input": { "file_path": "a.rs" } },
                    { "type": "text", "text": "second narration" },
                    { "type": "tool_use", "id": "tc_2", "name": "Read", "input": { "file_path": "b.rs" } },
                ],
            },
        });
        let mut turn_open = true;
        // SSE-mode: accumulated non-empty so the fallback MessageDelta
        // is skipped; the AssistantText events carry the interleaved
        // record on their own.
        let mut acc = String::from("first narrationsecond narration");
        let events = translate_frame(
            &frame,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None,
        );
        let kinds: Vec<&str> = events
            .iter()
            .map(|e| match e {
                AgentEvent::AssistantText { .. } => "text",
                AgentEvent::ToolCall { .. } => "tool",
                _ => "other",
            })
            .collect();
        assert_eq!(kinds, ["text", "tool", "text", "tool"]);
        match (&events[0], &events[2]) {
            (
                AgentEvent::AssistantText { text: t1, .. },
                AgentEvent::AssistantText { text: t2, .. },
            ) => {
                assert_eq!(t1, "first narration");
                assert_eq!(t2, "second narration");
            }
            other => panic!("expected two AssistantText blocks, got {other:?}"),
        }
    }

    #[test]
    fn translate_frame_drops_system_init_and_user_echo() {
        // The system frame and user echo both predate any meaningful
        // renderer-visible state; dropping them keeps the event log
        // clean and avoids confusing the UI's session list.
        let init = json!({
            "type": "system",
            "subtype": "init",
            "tools": [],
        });
        let user_echo = json!({
            "type": "user",
            "message": { "role": "user", "content": [{ "type": "text", "text": "echo" }] },
        });
        let mut turn_open = false;
        let mut acc = String::new();
        assert!(translate_frame(
            &init,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None
        )
        .is_empty());
        assert!(translate_frame(
            &user_echo,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None
        )
        .is_empty());
    }

    #[test]
    fn translate_frame_compact_boundary_emits_synthetic_turn() {
        // /compact triggers claude to emit a `system/compact_boundary`
        // frame mid-stream, then a `result` frame with `num_turns: 0`.
        // Without surfacing compact_boundary the result's TurnEnded
        // fires without a matching TurnStarted and the renderer hangs
        // on "Warming up...". Verify we emit TurnStarted +
        // MessageDelta + AssistantText + ContextCompacted (R280-F1) so
        // the lifecycle resolves and the UI can display token savings.
        let frame = json!({
            "type": "system",
            "subtype": "compact_boundary",
            "compact_metadata": { "pre_tokens": 18469, "post_tokens": 572 },
        });
        let mut turn_open = false;
        let mut acc = String::new();
        let events = translate_frame(
            &frame,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None,
        );
        assert_eq!(events.len(), 4);
        assert!(matches!(events[0], AgentEvent::TurnStarted { .. }));
        match &events[1] {
            AgentEvent::MessageDelta { text, .. } => {
                assert!(text.contains("18469"));
                assert!(text.contains("572"));
            }
            other => panic!("expected MessageDelta, got {other:?}"),
        }
        assert!(matches!(events[2], AgentEvent::AssistantText { .. }));
        match &events[3] {
            AgentEvent::ContextCompacted {
                pre_tokens,
                post_tokens,
                ..
            } => {
                assert_eq!(*pre_tokens, 18469);
                assert_eq!(*post_tokens, 572);
            }
            other => panic!("expected ContextCompacted, got {other:?}"),
        }
        assert!(turn_open);
        assert!(acc.contains("18469"));
    }

    #[test]
    fn translate_frame_assistant_emits_tool_call_breadcrumbs() {
        // Claude dispatches its built-in tools (Read/Edit/Write/Bash)
        // internally. Surfacing tool_use blocks as ToolCall events
        // gives the UI a "calling read…" breadcrumb without yah
        // dispatching the tool itself. normalize_claude_tool rewrites
        // the PascalCase name and file_path arg before emission.
        let frame = json!({
            "type": "assistant",
            "message": {
                "content": [
                    { "type": "text", "text": "Reading the file." },
                    {
                        "type": "tool_use",
                        "id": "toolu_01ABC",
                        "name": "Read",
                        "input": { "file_path": "/tmp/foo.rs" },
                    },
                ],
            },
        });
        let mut turn_open = true;
        let mut acc = String::new();
        let events = translate_frame(
            &frame,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None,
        );
        // MessageDelta (live fallback) + AssistantText (durable) + ToolCall.
        assert_eq!(events.len(), 3);
        match &events[2] {
            AgentEvent::ToolCall {
                tool_call_id,
                tool_name,
                args,
                ..
            } => {
                assert_eq!(tool_call_id, "toolu_01ABC");
                // normalize_claude_tool maps "Read" → "read"
                assert_eq!(tool_name, "read");
                // file_path is rewritten to path
                assert_eq!(
                    args.get("path").and_then(|v| v.as_str()),
                    Some("/tmp/foo.rs")
                );
                assert!(
                    args.get("file_path").is_none(),
                    "file_path must be renamed to path"
                );
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn normalize_claude_tool_maps_read_name_and_file_path() {
        let args = json!({ "file_path": "src/lib.rs" });
        let (name, out) = normalize_claude_tool("Read", args);
        assert_eq!(name, "read");
        assert_eq!(out.get("path").and_then(|v| v.as_str()), Some("src/lib.rs"));
        assert!(out.get("file_path").is_none());
    }

    #[test]
    fn normalize_claude_tool_read_converts_offset_limit_to_range() {
        let args = json!({ "file_path": "f.rs", "offset": 10u64, "limit": 50u64 });
        let (_, out) = normalize_claude_tool("Read", args);
        let range = out.get("range").and_then(|v| v.as_array()).unwrap();
        assert_eq!(range[0], 10u64);
        assert_eq!(range[1], 60u64);
        assert!(out.get("offset").is_none());
        assert!(out.get("limit").is_none());
    }

    #[test]
    fn normalize_claude_tool_edit_synthesises_diff_from_old_new_string() {
        let args = json!({
            "file_path": "f.rs",
            "old_string": "fn foo() {}",
            "new_string": "fn bar() {}\nfn baz() {}",
        });
        let (name, out) = normalize_claude_tool("Edit", args);
        assert_eq!(name, "edit");
        assert_eq!(out.get("path").and_then(|v| v.as_str()), Some("f.rs"));
        let diff = out.get("diff").and_then(|v| v.as_str()).unwrap();
        assert!(diff.contains("-fn foo()"), "old line prefixed with -");
        assert!(diff.contains("+fn bar()"), "new line prefixed with +");
        assert!(out.get("old_string").is_none());
        assert!(out.get("new_string").is_none());
    }

    #[test]
    fn normalize_claude_tool_bash_maps_command_to_cmd() {
        let args = json!({ "command": "cargo test" });
        let (name, out) = normalize_claude_tool("Bash", args);
        assert_eq!(name, "bash");
        assert_eq!(out.get("cmd").and_then(|v| v.as_str()), Some("cargo test"));
        assert!(out.get("command").is_none());
    }

    #[test]
    fn normalize_claude_tool_glob_maps_to_list_dir_with_path_from_pattern() {
        let args = json!({ "pattern": "**/*.rs" });
        let (name, out) = normalize_claude_tool("Glob", args);
        assert_eq!(name, "list_dir");
        assert_eq!(out.get("path").and_then(|v| v.as_str()), Some("**/*.rs"));
    }

    #[test]
    fn normalize_claude_tool_ls_maps_to_list_dir() {
        let args = json!({ "path": "src/" });
        let (name, out) = normalize_claude_tool("LS", args);
        assert_eq!(name, "list_dir");
        assert_eq!(out.get("path").and_then(|v| v.as_str()), Some("src/"));
    }

    #[test]
    fn normalize_claude_tool_grep_maps_name_preserves_args() {
        let args = json!({ "pattern": "fn run", "glob": "**/*.rs" });
        let (name, out) = normalize_claude_tool("Grep", args);
        assert_eq!(name, "grep");
        assert_eq!(out.get("pattern").and_then(|v| v.as_str()), Some("fn run"));
        assert_eq!(out.get("glob").and_then(|v| v.as_str()), Some("**/*.rs"));
    }

    #[test]
    fn normalize_claude_tool_unknown_names_pass_through_unchanged() {
        let args = json!({ "query": "stuff" });
        let (name, out) = normalize_claude_tool("TodoWrite", args.clone());
        assert_eq!(name, "TodoWrite");
        assert_eq!(out, args);
    }

    /// Helper: a `user` frame carrying one tool_result plus claude's
    /// out-of-band structured outcome. Shapes are transcribed from a
    /// live claude-cli 2.1.252 capture, not invented.
    fn user_frame_with(tur: serde_json::Value) -> serde_json::Value {
        json!({
            "type": "user",
            "message": {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_01ABC",
                    "content": "model-visible text",
                    "is_error": false,
                }],
            },
            "tool_use_result": tur,
        })
    }

    fn translate_one(frame: &serde_json::Value) -> Vec<AgentEvent> {
        translate_frame(
            frame,
            &sid(),
            &mut true,
            &mut String::new(),
            &mut String::new(),
            &mut None,
        )
    }

    fn sole_result(frame: &serde_json::Value) -> serde_json::Value {
        match translate_one(frame).as_slice() {
            [AgentEvent::ToolResult { result, .. }] => result.clone(),
            other => panic!("expected one ToolResult, got {other:?}"),
        }
    }

    /// The whole point of R076-F2: claude ships a real patch and we had
    /// been rendering a synthesised one-hunk approximation instead.
    #[test]
    fn user_tool_use_result_edit_yields_a_real_unified_diff() {
        let result = sole_result(&user_frame_with(json!({
            "filePath": "/tmp/f.txt",
            "oldString": "hello",
            "newString": "goodbye",
            "originalFile": "hello\n",
            "structuredPatch": [{
                "oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 1,
                "lines": ["-hello", "+goodbye"],
            }],
            "userModified": false,
            "replaceAll": false,
        })));
        assert_eq!(
            result["diff"].as_str(),
            Some("@@ -1,1 +1,1 @@\n-hello\n+goodbye\n")
        );
        assert_eq!(result["additions"], 1);
        assert_eq!(result["deletions"], 1);
        assert_eq!(result["path"].as_str(), Some("/tmp/f.txt"));
        assert_eq!(result["originalFile"].as_str(), Some("hello\n"));
    }

    /// Multi-hunk patches keep every hunk's own `@@` header — a single
    /// synthesised header is exactly the fidelity we're replacing.
    #[test]
    fn structured_patch_renders_every_hunk_with_its_own_header() {
        let (diff, adds, dels) = render_structured_patch(&[
            json!({ "oldStart": 3, "oldLines": 2, "newStart": 3, "newLines": 1,
                    "lines": [" ctx", "-gone"] }),
            json!({ "oldStart": 40, "oldLines": 1, "newStart": 39, "newLines": 3,
                    "lines": [" ctx", "+one", "+two"] }),
        ]);
        assert_eq!(
            diff,
            "@@ -3,2 +3,1 @@\n ctx\n-gone\n@@ -40,1 +39,3 @@\n ctx\n+one\n+two\n"
        );
        assert_eq!((adds, dels), (2, 1));
    }

    /// A before-image too large to carry is dropped, not truncated: a
    /// partial original would let Apply reconstruct the wrong file.
    #[test]
    fn oversized_original_file_is_dropped_and_flagged_not_truncated() {
        let big = "x".repeat(ORIGINAL_FILE_MAX_BYTES + 1);
        let result = sole_result(&user_frame_with(json!({
            "filePath": "big.rs",
            "originalFile": big,
            "structuredPatch": [{ "oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 1,
                                  "lines": ["-a", "+b"] }],
        })));
        assert!(result.get("originalFile").is_none());
        assert_eq!(result["originalFileTruncated"], true);
        assert!(result["diff"].as_str().is_some_and(|d| d.contains("+b")));
    }

    #[test]
    fn user_tool_use_result_bash_splits_stdout_and_stderr() {
        let result = sole_result(&user_frame_with(json!({
            "stdout": "ok\n",
            "stderr": "warn\n",
            "interrupted": false,
            "isImage": false,
            "noOutputExpected": false,
        })));
        assert_eq!(result["stdout"].as_str(), Some("ok\n"));
        assert_eq!(result["stderr"].as_str(), Some("warn\n"));
        assert_eq!(result["interrupted"], false);
        // No exit code rides the wire; inventing one would render as a
        // confident "exited clean" on a command that may not have.
        assert!(result.get("exit").is_none());
    }

    /// R775-B6, verbatim shape from claude transcript 3353ec5d (yah
    /// session:a4c54c76): a bash call auto-backgrounded on timeout. The fold
    /// must keep the task id and lift the output path out of the text it
    /// replaces, or the daemon's return defer cannot see the build at all.
    #[test]
    fn user_tool_use_result_bash_keeps_its_background_task() {
        let mut frame = user_frame_with(json!({
            "stdout": "", "stderr": "", "interrupted": false, "isImage": false,
            "noOutputExpected": false, "backgroundTaskId": "bgv37gqyx",
            "timedOutAfterMs": 590000,
        }));
        frame["message"]["content"][0]["content"] = json!(
            "Command did not complete within its 590s timeout and was moved to the background \
             (ID: bgv37gqyx). Output is being written to: /private/tmp/t/tasks/bgv37gqyx.output. \
             You will be notified when it completes."
        );
        let result = sole_result(&frame);
        assert_eq!(result["backgroundTaskId"], "bgv37gqyx");
        assert_eq!(
            result["backgroundOutputPath"],
            "/private/tmp/t/tasks/bgv37gqyx.output"
        );
    }

    #[test]
    fn user_tool_use_result_read_summarises_and_drops_file_content() {
        let result = sole_result(&user_frame_with(json!({
            "type": "text",
            "file": {
                "filePath": "f.rs",
                "content": "a\nb\n",
                "numLines": 2,
                "startLine": 1,
                "totalLines": 9,
            },
        })));
        assert_eq!(result["lines"], 2);
        assert_eq!(result["totalLines"], 9);
        assert_eq!(result["bytes"], 4);
        assert_eq!(result["truncated"], true);
        assert!(
            result.get("content").is_none() && result.get("file").is_none(),
            "the file body is never forwarded: {result}"
        );
    }

    #[test]
    fn user_tool_use_result_task_keeps_accounting_drops_transcript() {
        let result = sole_result(&user_frame_with(json!({
            "status": "completed",
            "prompt": "a very long subagent prompt",
            "agentType": "sniffer",
            "content": [{ "type": "text", "text": "…" }],
            "resolvedModel": "claude-opus-5",
            "totalDurationMs": 8123,
            "totalTokens": 4211,
            "totalToolUseCount": 3,
            "usage": { "output_tokens": 91 },
            "toolStats": { "Read": 2 },
        })));
        assert_eq!(result["totalTokens"], 4211);
        assert_eq!(result["totalToolUseCount"], 3);
        assert_eq!(result["resolvedModel"].as_str(), Some("claude-opus-5"));
        assert_eq!(result["toolStats"]["Read"], 2);
        assert!(result.get("prompt").is_none(), "prompt is already on args");
        assert!(result.get("content").is_none(), "transcript is not rendered");
    }

    /// Subagent-internal calls arrive with `tool_use_result: null`, and
    /// every tool claude doesn't describe has no entry at all. Both must
    /// leave the pre-existing text-content behaviour untouched.
    #[test]
    fn absent_tool_use_result_falls_back_to_the_text_content() {
        for tur in [json!(null), json!("unstructured")] {
            let mut frame = user_frame_with(tur);
            if frame["tool_use_result"].is_null() {
                frame.as_object_mut().unwrap().remove("tool_use_result");
            }
            let result = sole_result(&frame);
            assert_eq!(
                result.as_str(),
                Some("model-visible text"),
                "non-object tool_use_result must not displace the content"
            );
        }
    }

    /// `tool_use_result` is one-per-frame with nothing tying it to a
    /// block, so a batched flush must not staple one tool's outcome onto
    /// another tool's result.
    #[test]
    fn batched_tool_results_are_not_attributed_to_the_single_payload() {
        let frame = json!({
            "type": "user",
            "message": { "content": [
                { "type": "tool_result", "tool_use_id": "toolu_A", "content": "first" },
                { "type": "tool_result", "tool_use_id": "toolu_B", "content": "second" },
            ]},
            "tool_use_result": { "stdout": "only one of these", "stderr": "" },
        });
        let events = translate_one(&frame);
        assert_eq!(events.len(), 2);
        for ev in &events {
            match ev {
                AgentEvent::ToolResult { result, .. } => {
                    assert!(
                        result.is_string(),
                        "batched results keep their own text: {result}"
                    );
                }
                other => panic!("expected ToolResult, got {other:?}"),
            }
        }
    }

    /// A tool call made inside a Task must stay attributable to it.
    #[test]
    fn assistant_tool_use_carries_parent_tool_use_id() {
        let frame = json!({
            "type": "assistant",
            "parent_tool_use_id": "toolu_01PZ8",
            "message": { "content": [{
                "type": "tool_use",
                "id": "toolu_inner",
                "name": "Bash",
                "input": { "command": "ls" },
            }]},
        });
        match translate_one(&frame).as_slice() {
            [AgentEvent::ToolCall { args, .. }] => {
                assert_eq!(args["_parentToolCallId"].as_str(), Some("toolu_01PZ8"));
            }
            other => panic!("expected one ToolCall, got {other:?}"),
        }
    }

    /// …and a top-level call must not grow a null parent, which the FE
    /// would have to special-case as "not really nested".
    #[test]
    fn top_level_tool_use_has_no_parent_key() {
        let frame = json!({
            "type": "assistant",
            "parent_tool_use_id": serde_json::Value::Null,
            "message": { "content": [{
                "type": "tool_use", "id": "t", "name": "Bash", "input": { "command": "ls" },
            }]},
        });
        match translate_one(&frame).as_slice() {
            [AgentEvent::ToolCall { args, .. }] => {
                assert!(args.get("_parentToolCallId").is_none());
            }
            other => panic!("expected one ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn normalize_claude_tool_multiedit_maps_to_edit() {
        let args = json!({ "file_path": "x.rs", "edits": [] });
        let (name, _) = normalize_claude_tool("MultiEdit", args);
        assert_eq!(name, "edit");
    }

    #[test]
    fn translate_frame_user_tool_result_emits_tool_result() {
        // The user frame carries claude's internal tool_result blocks;
        // surface them as ToolResult so the renderer can pair with
        // the prior ToolCall and render success/failure.
        let frame = json!({
            "type": "user",
            "message": {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_01ABC",
                    "content": "file contents here",
                    "is_error": false,
                }],
            },
        });
        let mut turn_open = true;
        let mut acc = String::new();
        let events = translate_frame(
            &frame,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None,
        );
        assert_eq!(events.len(), 1);
        match &events[0] {
            AgentEvent::ToolResult {
                tool_call_id,
                ok,
                result,
                ..
            } => {
                assert_eq!(tool_call_id, "toolu_01ABC");
                assert!(*ok);
                assert_eq!(result.as_str(), Some("file contents here"));
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn translate_frame_user_tool_result_with_is_error_flips_ok_false() {
        let frame = json!({
            "type": "user",
            "message": {
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_X",
                    "content": "permission denied",
                    "is_error": true,
                }],
            },
        });
        let mut turn_open = true;
        let mut acc = String::new();
        let events = translate_frame(
            &frame,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None,
        );
        assert_eq!(events.len(), 1);
        match &events[0] {
            AgentEvent::ToolResult { ok, .. } => assert!(!*ok),
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn translate_frame_message_delta_emits_turn_usage_delta() {
        // Anthropic's per-chunk message_delta carries running
        // output_tokens. Surfacing as TurnUsageDelta lets the UI
        // animate the budget gauge without waiting for the result.
        let frame = json!({
            "type": "stream_event",
            "event": {
                "type": "message_delta",
                "delta": { "stop_reason": null },
                "usage": { "output_tokens": 42 },
            },
        });
        let mut turn_open = true;
        let mut acc = String::new();
        let events = translate_frame(
            &frame,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None,
        );
        assert_eq!(events.len(), 1);
        match &events[0] {
            AgentEvent::TurnUsageDelta { usage, .. } => {
                assert_eq!(usage.output_tokens, Some(42));
            }
            other => panic!("expected TurnUsageDelta, got {other:?}"),
        }
    }

    #[test]
    fn translate_frame_message_delta_without_usage_emits_nothing() {
        // No usage object → no TurnUsageDelta. Empty would just spam
        // the renderer with useless events.
        let frame = json!({
            "type": "stream_event",
            "event": {
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn" },
            },
        });
        let mut turn_open = true;
        let mut acc = String::new();
        let events = translate_frame(
            &frame,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None,
        );
        assert!(events.is_empty());
    }

    #[test]
    fn assistant_entry_usage_extracts_anthropic_shape() {
        // Claude's own JSONL writes per-API-call usage under
        // .message.usage with input/output/cache_read/cache_creation
        // fields. R564-F4 persists these so post-session cost analysis
        // has a granular record.
        let v = json!({
            "type": "assistant",
            "message": {
                "usage": {
                    "input_tokens": 123,
                    "output_tokens": 45,
                    "cache_read_input_tokens": 6789,
                    "cache_creation_input_tokens": 100,
                },
            },
        });
        let u = assistant_entry_usage(&v).expect("usage extracted");
        assert_eq!(u.input_tokens, Some(123));
        assert_eq!(u.output_tokens, Some(45));
        assert_eq!(u.cache_read_tokens, Some(6789));
        assert_eq!(u.cache_write_tokens, Some(100));
        // peak = input + cache_read + cache_write (sum of context-resident
        // fields; output rolls into next turn's cache, see ContextRings).
        assert_eq!(u.peak_context_tokens, Some(123 + 6789 + 100));
    }

    #[test]
    fn assistant_entry_usage_returns_none_when_all_token_fields_missing() {
        let v = json!({ "type": "assistant", "message": { "usage": {} } });
        assert!(assistant_entry_usage(&v).is_none());
    }

    #[test]
    fn assistant_entry_usage_extracts_per_call_tokens() {
        // Shape of one `assistant` line in claude's own transcript JSONL.
        let v = json!({
            "type": "assistant",
            "message": {
                "usage": {
                    "input_tokens": 4,
                    "output_tokens": 120,
                    "cache_read_input_tokens": 18_000,
                    "cache_creation_input_tokens": 200
                }
            }
        });
        let u = assistant_entry_usage(&v).expect("usage present");
        assert_eq!(u.input_tokens, Some(4));
        assert_eq!(u.output_tokens, Some(120));
        assert_eq!(u.cache_read_tokens, Some(18_000));
        assert_eq!(u.cache_write_tokens, Some(200));
        // peak_context_tokens = per-call fill = input + cache_read + cache_write.
        assert_eq!(u.peak_context_tokens, Some(4 + 18_000 + 200));
        assert_eq!(u.thinking_tokens, None);
    }

    #[test]
    fn assistant_entry_usage_none_when_no_usage_object() {
        let v = json!({ "type": "assistant", "message": { "content": [] } });
        assert!(assistant_entry_usage(&v).is_none());
    }

    #[test]
    fn assistant_entry_usage_none_when_all_token_fields_absent() {
        // A usage object with only non-token metadata yields no delta.
        let v =
            json!({ "type": "assistant", "message": { "usage": { "service_tier": "standard" } } });
        assert!(assistant_entry_usage(&v).is_none());
    }

    #[test]
    fn assistant_entry_usage_output_only_still_persists() {
        // Streaming tail-end entries may carry only output_tokens; that's a
        // real (billable) delta and must round-trip.
        let v = json!({ "type": "assistant", "message": { "usage": { "output_tokens": 42 } } });
        let u = assistant_entry_usage(&v).expect("usage present");
        assert_eq!(u.output_tokens, Some(42));
        assert_eq!(u.input_tokens, None);
        // No input/cache fields → no meaningful context-fill peak.
        assert_eq!(u.peak_context_tokens, None);
    }

    #[test]
    fn detect_session_limit_matches_real_wire_text() {
        // Wire text from the 2026-06-20 postmortem (session:8926fab8).
        let content = serde_json::json!(
            "You've hit your session limit \u{00b7} resets 9:20pm (America/Los_Angeles)"
        );
        let result = detect_session_limit(&content);
        assert_eq!(result.as_deref(), Some("9:20pm (America/Los_Angeles)"));
    }

    #[test]
    fn detect_session_limit_requires_timezone_slash() {
        // " resets " without a "/" in the remainder must not match.
        let content = serde_json::json!("hit your session limit, nothing resets today");
        assert!(detect_session_limit(&content).is_none());
    }

    #[test]
    fn detect_session_limit_ignores_a_grep_over_this_very_file() {
        // The regression that motivated the anchored match: an agent runs
        // `rg "session limit"` across the tree, the tool result carries this
        // module's own doc comment plus a pile of absolute paths, and the old
        // loose matcher read the rest of the grep output as a "resets at"
        // string — locking every character in the camp out for hours.
        let content = serde_json::json!(concat!(
            "/Users/leif/ss/yah/app/yah/desktop/src/agent_process.rs:5909:",
            "/// The full wire text is `\"You've hit your session limit ",
            "\u{00b7} resets <time> (<tz>)\"`.\n",
            "/Users/leif/ss/yah/app/yah/desktop/src/agent_process.rs:5914:",
            "fn detect_session_limit(content: &serde_json::Value) -> Option<String> {\n",
            "/Users/leif/ss/yah/app/yah/desktop/src/agent_process.rs:5916:",
            "    if !s.contains(\"hit your session limit\") {\n",
        ));
        assert!(detect_session_limit(&content).is_none());
    }

    #[test]
    fn detect_session_limit_rejects_phrase_buried_mid_string() {
        // Same failure shape at minimum size: the phrase is present and a "/"
        // follows, but the phrase does not start the message.
        let content =
            serde_json::json!("The docs say you hit your session limit · resets a/b");
        assert!(detect_session_limit(&content).is_none());
    }

    #[test]
    fn detect_session_limit_rejects_trailing_junk_after_the_zone() {
        // A real message ends at the closing paren. Anything after it means we
        // are looking at quoted source or prose, not the wire text.
        let content = serde_json::json!(
            "You've hit your session limit · resets 9:20pm (America/Los_Angeles)\"`."
        );
        assert!(detect_session_limit(&content).is_none());
    }

    #[test]
    fn detect_session_limit_accepts_a_24_hour_clock() {
        let content =
            serde_json::json!("You've hit your session limit · resets 21:20 (Europe/London)");
        assert_eq!(
            detect_session_limit(&content).as_deref(),
            Some("21:20 (Europe/London)")
        );
    }

    #[test]
    fn detect_session_limit_ignores_non_string_content() {
        let content = serde_json::json!([{"type": "text", "text": "hit your session limit · resets 9:20pm (America/Los_Angeles)"}]);
        // Array content is not matched — only plain string values.
        assert!(detect_session_limit(&content).is_none());
    }

    #[test]
    fn translate_frame_session_limit_emits_tool_result_then_limit_hit_and_returns_early() {
        // Simulate the wire frame that caused the 2026-06-20 runaway.
        let frame = json!({
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-abc",
                        "content": "You\u{2019}ve hit your session limit \u{00b7} resets 9:20pm (America/Los_Angeles)",
                        "is_error": false
                    },
                    // Second limit result — must NOT appear in output (early return).
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-def",
                        "content": "You\u{2019}ve hit your session limit \u{00b7} resets 9:20pm (America/Los_Angeles)",
                        "is_error": false
                    },
                ]
            }
        });
        let mut turn_open = false;
        let mut acc = String::new();
        let events = super::translate_frame(
            &frame,
            &sid(),
            &mut turn_open,
            &mut acc,
            &mut String::new(),
            &mut None,
        );
        // First event: ToolResult for tool-abc
        assert!(
            matches!(&events[0], PvEvent::Core(AgentEvent::ToolResult { tool_call_id, .. }) if tool_call_id == "tool-abc")
        );
        // Second event: TurnBlocked{UsageLimitHit} (R575-T7) — ordered before
        // SessionLimitHit so the caller's Blocked-phase transition
        // lands before the subprocess is stopped.
        match &events[1] {
            PvEvent::Core(AgentEvent::TurnBlocked { reason, .. }) => match reason {
                BlockReason::UsageLimitHit { resets_at } => {
                    assert_eq!(resets_at.as_deref(), Some("9:20pm (America/Los_Angeles)"));
                }
                other => panic!("expected UsageLimitHit, got {other:?}"),
            },
            other => panic!("expected TurnBlocked, got {other:?}"),
        }
        // Third event: SessionLimitHit
        match &events[2] {
            PvEvent::SessionLimitHit { resets_at, .. } => {
                assert!(
                    resets_at.contains("America/Los_Angeles"),
                    "{resets_at}"
                );
            }
            other => panic!("expected SessionLimitHit, got {other:?}"),
        }
        // Early return: tool-def must NOT appear.
        assert_eq!(
            events.len(),
            3,
            "expected exactly 3 events (early return on first limit hit)"
        );
    }

}
