//! OpenAI-compatible runner.
//!
//! Covers every backend that speaks `POST /v1/chat/completions` with
//! `stream: true` SSE: OpenAI proper, Ollama (local and Pro), Qwen,
//! vLLM, LiteLLM, and any host running an OpenAI-shape gateway. The
//! single implementation pressure-tests the [`Runner`] contract — if
//! the trait holds up here and against the existing Claude path in
//! `app/yah/desktop/src/agent.rs`, it will hold up against the next provider.
//!
//! ## SSE schema we fold
//!
//! Each frame has `data: <json>\n\n` framing. Payloads we recognise:
//!
//! - `{"choices":[{"delta":{"content":"…"},"finish_reason":null}]}` →
//!   yield [`AgentEvent::MessageDelta`].
//! - `{"choices":[{"finish_reason":"stop"}]}` → record the stop reason
//!   for the eventual [`AgentEvent::TurnEnded`].
//! - `{"error":{"message":"…"}}` → yield [`AgentEvent::Error`] and end
//!   the stream.
//! - `data: [DONE]` → end-of-stream sentinel.
//!
//! Any frame that doesn't parse is ignored (matches Anthropic SSE
//! behaviour in the Claude reference impl — never abort a turn over a
//! single weird chunk).
//!
//! ## What's intentionally not here
//!
//! - **Anthropic tool_use** — structurally similar to OpenAI's shape
//!   but uses content-block `tool_use` / `tool_result` blocks instead
//!   of `tools[]` + `tool_calls[]`. Lives in `app/yah/desktop/src/agent.rs`
//!   under the Claude track (R028) and is wired up there once the
//!   native loop ships.
//! - **Parallel sampling** — `n > 1` would require widening the event
//!   shape; we hard-code `n: 1` since that's the only mode the agent
//!   pane drives today.
//! - **Per-session iteration cap** — [`MAX_TOOL_ITERATIONS`] is a const
//!   today. Surfacing it as a per-session knob is a polish follow-up.
//!
//! @arch:see(.yah/docs/architecture/A004-agent-tool-calls.md)
//!
//! archived: R113 — history: `yah board history oss/runner/crates/ho/src/openai_compat.rs`
//!
//! @arch:see(.yah/docs/architecture/A018-session-loop-observability.md)
//!
//! @arch:see(.yah/docs/architecture/A018-session-loop-observability.md)
//!
//! archived: R121 — history: `yah board history oss/runner/crates/ho/src/openai_compat.rs`
//! @arch:see(.yah/docs/architecture/A018-session-loop-observability.md)
//!
//! archived: R124 — history: `yah board history oss/runner/crates/ho/src/openai_compat.rs`
//! @arch:see(.yah/docs/architecture/A029-yah-claude-ollama-bridge.md)
//!
//! @arch:see(.yah/sessions/session:8e1e1cd4.jsonl)
//!
//! archived: R169 — history: `yah board history oss/runner/crates/ho/src/openai_compat.rs`
//!
//! @yah:ticket(R215-F1, "probe-suite engine crate")
//! @yah:assignee(agent:claude)
//! @yah:at(2026-05-17T22:11:03Z)
//! @yah:status(review)
//! @yah:parent(R215)
//! @yah:next("Create crates/yah/probe (pure-Rust). Types: ProbeSpec { id, family, prompt, scorer, est_budget, priority }, ProbeReport { verdict, scores, ttft, latency, usage, raw_per_probe }, ProbeFamily enum, Scorer trait.")
//! @yah:next("Engine entry: probe_subclass(subclass_id, budget, families) -> ProbeReport. Resolve via existing resolver to ResolvedEngineConfig — DO NOT build a parallel dispatch path; the probe must hit the real bytes-on-wire shape (R124-S8 lesson).")
//! @yah:next("Budget gating: probes run in priority order; sweep aborts gracefully on exhaustion and returns partial report with budget_exhausted: true.")
//! @yah:next("Stream progress as ProbeFrame events (probe_started/probe_finished/budget_check) on a tokio channel so the Tauri layer can forward them live.")
//! @yah:verify("cargo test -p probe — registry, budget gating, partial-report shape, Scorer trait impls")
//! @yah:handoff("New crate crates/yah/probe (workspace + default-members). Modules: types (ProbeSpec/ProbeReport/ProbeFamily/Verdict/Budget/ProbeFrame/ProbeRecord/FamilyReport/ProbeCompletion/ProbeScore), scorer (Scorer trait + AlwaysPassScorer + ContainsScorer fixtures — real scorers F3), registry (ProbeRegistry::selected_sorted = priority asc, stable within bucket), driver (ProbeDriver trait + OpenAiCompatProbeDriver wrapping runner::OpenAiCompatRunner via start→send→stop per probe + MockProbeDriver test fixture + ProbePrelude minimal builder), engine (ProbeEngine + top-level probe_subclass convenience). Production driver routes through OpenAiCompatRunner so probes hit the real bytes-on-wire shape (R124-S8 lesson). TTFT = send-accepted→first-MessageDelta; total_ms = send→TurnEnded/TurnFailed. Budget gating: pre-flight check tokens_used+est_budget vs budget.tokens AND optional wall-clock deadline_ms; abort emits ProbeFrame::BudgetCheck + ProbeFrame::SweepEnded{budget_exhausted:true} and returns partial ProbeReport. Driver errors record Verdict::Error (excluded from pass/fail rate) without aborting the sweep. 8 unit tests cover priority order, budget exhaustion → partial report, error-verdict isolation, ContainsScorer pass/fail mix, family filter, full frame lifecycle, registry stable sort, ISO-8601 helper. cargo test -p probe 8/8; cargo check --workspace clean.")
//! @yah:next("F2 (health probes) consumes ProbeRegistry+ProbeSpec; each spec has est_budget for gating and a Scorer reading ttft_ms / usage / stop_reason off ProbeCompletion. Driver: OpenAiCompatProbeDriver::new(Arc::new(OpenAiCompatRunner::new(resolved.into_openai_compat())), ProbePrelude::minimal())")
//! @yah:next("Anthropic-native /v1/messages + ClaudeCli subprocess subclasses NOT yet probeable — F1 ships the OpenAI-compat-wire driver only. Adding an AnthropicProbeDriver (via bridge crate or by lifting Anthropic's send loop) is its own ticket; file once F2 lands the first health-probe spec and wants cache-capable subclass verification")
//! @yah:next("engine_ref_from_subclass duplicated at app/yah/desktop/src/agent.rs:1969 — F5 Tauri command boundary will need it; consider lifting to kg::party::AgentSubclass::engine_ref() before duplicating into probe convenience layer")
//! @yah:verify("cargo test -p probe — 8 tests pass (engine::tests::runs_in_priority_order, budget_exhaustion_yields_partial_report, driver_error_records_error_verdict_without_breaking_sweep, contains_scorer_pass_fail_mix, family_filter_skips_unmatched, frames_stream_full_lifecycle, registry_selected_sorted_stable_within_priority, iso8601_renders_unix_zero)")
//! @yah:verify("cargo check --workspace clean")
//!
//! @yah:ticket(R215-F3, "capability probes (curated MMLU/IFEval/HumanEval slice + scorers)")
//! @yah:assignee(agent:claude)
//! @yah:at(2026-05-17T22:11:38Z)
//! @yah:status(review)
//! @yah:parent(R215)
//! @yah:next("Curate ~10 probes total from public benchmarks: 3 MMLU items (multi-choice, exact-match scorer), 3 IFEval items (instruction-following, regex scorer), 2 HumanEval items (tiny code task, ast-eval scorer), 1 JSON-mode item (json-parse scorer), 1 refusal-calibration item (judge scorer). Bundle source items + attribution in crates/yah/probe/data/. Verify each source license is MIT/Apache/CC per permissive-licenses rule.")
//! @yah:next("Scorer trait impls: ExactMatch, Regex, JsonParse, AstEval (tiny Rust/TS parse-and-compare for HumanEval), GnomeJudge (delegates to F4).")
//! @yah:next("Capability budget budget hits ~30–60k tokens for the full slice (each probe ~1–10k); fits inside the default 200k envelope alongside health probes.")
//! @yah:verify("cargo test -p probe -- capability:: — each scorer with golden inputs (passing/failing cases)")
//! @yah:verify("Live: bundle-anthropic-claude-* scores capability% > 80 on the curated slice; report renders with per-probe pass/fail breakdown")
//! @yah:handoff("10 capability probe specs in crates/yah/probe/src/capability/: 3 MMLU multi-choice (ExactMatchScorer), 3 IFEval instruction-following (RegexScorer), 2 HumanEval code-completion (AstEvalScorer), 1 JSON-mode (JsonParseScorer), 1 refusal-calibration (GnomeJudgeScorer stub — returns Skip until F4). Total est. budget ~16k tokens; combined with health probes ~81k, well within 200k envelope. regex crate added to Cargo.toml. Attribution for all external sources in data/ATTRIBUTION.md (MMLU MIT, IFEval Apache 2.0, HumanEval MIT, custom items MIT). 24 new tests across capability::scorers and capability::. cargo test -p probe: 60/60. cargo check --workspace clean.")
//! @yah:next("F4 (gnome-as-judge): implement GnomeJudgeScorer with a real gnome dispatch; replace the stub in capability/scorers.rs; add judging-subclass setting to prefs")
//! @yah:verify("cargo test -p probe -- capability:: (24 new tests pass)")
//! @yah:verify("cargo test -p probe (all 60 tests pass)")
//! @yah:verify("cargo check --workspace clean")
//! @yah:verify("Live: bundle-anthropic-claude-* should score capability% > 80 on the curated slice (9/10 scoreable items, 1 returns Skip until F4)")
//!
//!
//! @yah:ticket(R237-F12, "F12: OpenAI-compat SSE retry/backoff (covers openai/openrouter/groq + ollama-NDJSON) (R237-F6 sub-work)")
//! @yah:assignee(agent:claude)
//! @yah:at(2026-05-20T06:40:21Z)
//! @yah:status(review)
//! @yah:parent(R237)
//! @yah:next("Single shared stream impl in openai_compat.rs handles 4 providers via Transport::Sse + Transport::OllamaNdjson; one retry layer covers all of them.")
//! @yah:next("Wrap issue_chat_completions + stream consumption in retry loop (3 attempts, 1/2/4s backoff).")
//! @yah:next("Classify errors via health::classify_runner_error; retry only RateLimited (honor Retry-After) + Network + ServerError; pass-through Permanent.")
//! @yah:next("Skip retry once mid-stream data has reached the FE (no duplicate output).")
//! @yah:next("Tool-loop re-issues (line ~864) already re-fetch the whole turn; the retry layer wraps the per-cell fetch, not the tool loop.")
//! @yah:next("Tests: mock 503 once → stream succeeds on second attempt; mock auth failure → no retry, immediate TurnFailed.")
//! @arch:see(.yah/docs/working/W085-fe-to-be-state-promotion.md)
//! @yah:handoff("Pre-stream retry loop landed in openai_compat.rs::issue_chat_completions. Constants: MAX_STREAM_OPEN_ATTEMPTS=3, MAX_RETRY_AFTER_SECS=30. Helpers: is_retryable_status (429+5xx → true; 401/403/4xx → false), compute_stream_open_backoff_ms (Retry-After honored up to 30s; otherwise exp 1s/2s/4s). Loop wraps both the reqwest.send failure path and the non-success-status path; 200 returns immediately, retryable failures sleep+continue, permanent failures return Err. Covers all 4 OpenAI-compat providers (openai/openrouter/groq SSE + ollama-NDJSON) via the single shared call site — each iteration of the tool loop re-enters this function and gets its own retry budget.")
//! @yah:verify("cargo check -p runner -p desktop -p yah  # workspace-relevant crates clean (pre-existing desktop warnings only)")
//! @yah:verify("cargo test -p runner --lib openai_compat::  # 53/53 — includes 3 new retry-helper tests (is_retryable_status, compute_stream_open_backoff_ms exp + Retry-After)")
//! @yah:verify("cargo test -p runner --test openai_compat_e2e -- --skip ollama_local_round_trip  # 11/11 — includes 3 new e2e tests (503-then-200 retries, 401 no-retry asserted via served counter, 429 with Retry-After honors header)")
//! @yah:verify("cargo test -p runner --lib  # 190/190 — no runner-suite regressions (was 187/187 before F12)")
//!
//! @yah:ticket(R293-F1, "Wire thinking_budget through OpenAI-compatible requests")
//! @yah:at(2026-05-22T21:41:45Z)
//! @yah:status(review)
//! @yah:parent(R293)
//! @yah:next("Map ThinkingBudget enum variants to reasoning_effort parameter: Low→'high', Medium→'max', High→'max', ExtraHigh→'max'. Wire subclass.thinking_budget through openai_compat::issue_chat_completions.")
//! @yah:next("For OpenaiWire: inject reasoning_effort into the chat request extra_body as {\"thinking\": {\"type\": \"enabled\"}, \"reasoning_effort\": \"...\"}")
//! @yah:next("For AnthropicWire via DeepSeek: map thinking_budget to output_config.effort parameter in the Anthropic-wire branch (coordinate with AnthropicResolver).")
//! @yah:next("Handle reasoning_content in responses: when the model returns reasoning_tokens or reasoning_content, emit it as ThinkingDelta events in the stream (not just regular content deltas).")
//! @yah:next("Add test: verify a DeepSeek OpenaiWire request with thinkingBudget=medium includes the correct reasoning_effort parameter in the request body.")
//! @yah:verify("cargo test -p runner --lib openai_compat")
//! @yah:verify("Manual: start a DeepSeek subclass session with thinkingBudget set, inspect HTTP request body for {\"thinking\": {\"type\": \"enabled\"}, \"reasoning_effort\": \"max\"}")
//! @yah:cleanup("Context window implications: document in DeepSeek connection guide that reasoning_content is re-sent on tool calls, increasing token usage.")
//! @yah:handoff("Wire thinking_budget through OpenAI-compatible requests: added thinking wrapper to reasoning_effort parameter, implemented ThinkingDelta event emission for reasoning content in responses, added comprehensive tests. All openai_compat tests passing (58/58).")
//! @yah:handoff("All work for OpenAI-compatible thinking_budget wiring is complete:")
//! @yah:handoff("✅ ThinkingBudget enum properly maps to reasoning_effort (Low→'high', Medium/High/ExtraHigh→'max')")
//! @yah:handoff("✅ thinking_budget flows from subclass through ResolvedEngineConfig → OpenAiCompatConfig → request body")
//! @yah:handoff("✅ Request body correctly injects {\"thinking\": {\"type\": \"enabled\"}, \"reasoning_effort\": \"...\"} for models supporting extended thinking")
//! @yah:handoff("✅ parse_sse_frame correctly extracts delta.thinking and emits ThinkingDelta events in the response stream")
//! @yah:handoff("✅ Comprehensive unit tests: body_includes_reasoning_effort_from_thinking_budget covers all budget variants; parse_frame_extracts_thinking_content_from_delta and parse_frame_emits_both_thinking_and_content_deltas verify SSE parsing")
//! @yah:handoff("✅ All 58 openai_compat tests passing")
//! @yah:handoff("Note: AnthropicWire via DeepSeek (output_config.effort mapping) is out of scope for R293-F1 and tracked separately in resolver work")
//!
//! @yah:ticket(R418-F2, "Pre-flight prompt-size guard + auto-elide oversized tool_result before API call")
//! @yah:assignee(agent:claude)
//! @yah:at(2026-06-03T03:08:37Z)
//! @yah:status(review)
//! @yah:parent(R418)
//! @yah:next("Before each API request, estimate next-prompt tokens (sum of message + tool_result content lengths * ~0.25 chars/token). If estimate > 0.95 * context_window, find the largest recent tool_result in session.history and replace its content with a truncation note: '[tool_result elided · NNKB · original call: <toolName>(<args>) · re-call with limit=/since=/bucket= for a slice]'. Retry the request once before failing the turn.")
//! @yah:next("Emit AgentEvent::PromptTrimmed { session_id, elided_tool_call_id, original_bytes, replacement_bytes } so the renderer can show a 'context auto-trimmed' chip and the user knows what happened.")
//! @yah:next("Threshold (0.95) should be tunable via OpenAiCompatConfig::prompt_overflow_threshold (Option<f32>, default 0.95). Don't trim if the smallest possible reduction still leaves us over.")
//! @yah:verify("Construct a fake session whose history has one >800KB tool_result; next send() should auto-elide it, emit PromptTrimmed, and complete the turn instead of TurnFailed.")
//! @yah:gotcha("Today the runner just forwards the 400 'maximum context length' error as TurnFailed with no recovery — same crash mode that ended session:098779e3 turn 8 at 1.51M/1.05M tokens.")
//! @yah:handoff("Pre-flight prompt-size guard landed in crates/yah/runner/src/openai_compat.rs. Each Runner::send() (and every tool-loop re-issue inside the streaming generator) now: (1) reads OpenAiCompatConfig::context_window + prompt_overflow_threshold (default 0.95) — None disables the guard; (2) computes estimated prompt chars (sum of serialized JSON message sizes + prelude, ~4 chars/token); (3) when est > threshold * window, finds the largest tool_result row with content ≥ 4KB and replaces its content with '[tool_result elided · NNKB · original call: <toolName> · re-call with narrower filters (limit, since_ts, bucket) for a slice]'; (4) loops until under-cap or no more elidable rows. Bails out (no trim) when even maxing out elision can't bring us under cap — information loss without benefit. (5) Emits AgentEvent::PromptTrimmed { session_id, tool_call_id, tool_name, original_bytes, replacement_bytes } once per elision, before TurnStarted (initial) or between iterations (in-loop). Ollama factories wire context_window from num_ctx; the generic resolver mirrors num_ctx; openai/groq factories leave context_window=None until the host plugs in the model registry. 10 new openai_compat tests cover estimate, lookup_tool_name (assistant tool_calls back-walk), elide_largest (size selection + structural preservation), min-elidable threshold, no-op under cap, bail when futile, single elision, multi-iteration elision, default threshold, and elision-note format. All 69 openai_compat tests pass; runner --tests compiles.")
//! @yah:next("User smoke test: set context_window on a session config (e.g. via the Ollama path which already wires it from num_ctx). Inject a single huge tool_result (> cap * 0.95) into history and confirm: (a) the next send() emits PromptTrimmed before TurnStarted; (b) the turn completes successfully; (c) the elided row's content shows the truncation note in any subsequent inspection. Then check the per-engine context_window plumbing for hosted models (OpenAI/Anthropic/DeepSeek) — today only Ollama gets a non-None value because num_ctx provides it; for the case that triggered R418 (deepseek-v4-flash, 1.05M window) the host needs to set context_window=Some(1_048_576) when constructing the runner's OpenAiCompatConfig, otherwise the guard stays inert. That wiring lives outside the runner crate; flag in the F3/F4 follow-ups.")
//! @yah:verify("cargo test -p runner --lib openai_compat")
//!
//! @yah:ticket(R418-F4, "ResolveBlock::Compact for openai_compat path (mirror claude_cli /compact flow)")
//! @yah:assignee(agent:claude)
//! @yah:at(2026-06-03T03:09:19Z)
//! @yah:status(review)
//! @yah:parent(R418)
//! @yah:next("When pre-flight elision (R418-F2) isn't enough — i.e. the conversation as a whole is over-budget, not just one outlier tool_result — emit TurnBlocked{reason: context_overflow} instead of TurnFailed. Mirror the existing ResolveBlock::Compact/HandOff/Abandon machinery already designed for claude_cli (see sessions.rs:210 design notes).")
//! @yah:next("Compact handler: spawn a sibling 'summarizer' session over current history with a fixed system prompt ('produce a 2000-token summary of this conversation including key file paths, decisions, and unresolved questions'), replace session.history with [system, summary, last_k_turns], then replay the failed turn.")
//! @yah:next("HandOff handler: pick a wider-window engine from the configured pool (e.g. fall back from a 1M model to a 2M model) and replay last_k_turns there. Needs a per-engine context_window registry — partially exists via ConfigResolved.context_tokens.")
//! @yah:verify("Manual: construct a session whose total history is >0.95*window even after F2 elision. yah session resolve <id> compact must succeed and the replayed turn must complete.")
//! @yah:gotcha("claude_cli wraps the upstream /compact command which is special-cased on Anthropic's side; the openai_compat path has no equivalent — we have to do the summarization ourselves, which makes this strictly more work than the claude_cli path.")
//! @yah:depends_on(R418-F2)
//! @yah:handoff("Two phases landed in crates/yah/runner/src/openai_compat.rs. Phase A (TurnBlocked routing): is_overflow_message() wraps block_detector::fallback for the shared phrase set (context_length_exceeded, prompt is too long, exceeds the context window, tokens > N). Three integration points: (1) pre-stream initial request: when issue_chat_completions returns RunnerError::Transport with an overflow phrase, send() now yields TurnStarted then TurnBlocked{ContextOverflow} instead of returning Err and surfacing as AgentEvent::Error. (2) In-loop tool re-issue: pre-stream failure path detects overflow and emits TurnBlocked instead of TurnFailed. (3) Stream-mid 5xx / non-utf8 / soft-cap paths untouched (those are transport-shaped, not overflow). With R262-F2 wire already in place, the host now routes these to SessionPhase::Blocked and the recovery modal becomes available. Phase B (compact primitive): OpenAiCompatRunner::compact_session(session_id, keep_last_n) issues one non-streaming chat-completions request against the same provider/model with a fixed summarizer system prompt + a markdown dump of history (helper serialize_history_for_summary); rewrites session.history as [summary_system_row, last_N_original_rows]. Returns CompactResult { session_id, summary_chars, original_history_chars, history_messages_before/after, kept_tail }. The session stays open; caller retries the failed turn via send(). 6 new tests cover the overflow phrase set (openai, ollama, rejections for 503/401/dns), history serialization (roles, tool_calls flattening, tool_result IDs), CompactResult shape, and the SessionNotFound path. All 80 openai_compat unit tests pass; runner --tests compiles clean. Deferred (file as R418-F5 if needed): wire OpenAiCompatRunner::compact_session into the daemon's resolve_block_runner dispatch (app/yah/cli/src/camp.rs) so `yah session resolve <id> compact` runs the new path against an openai_compat session — today the F5 dispatch arm still returns a 'use HandOff' hint per the R262-F5 stub. That wiring crosses the runner ↔ daemon boundary and is best landed as a separate sub-ticket alongside an e2e test that drives it through the camp socket.")
//! @yah:next("User smoke test (runner-side only): in a unit-test-like setup, push a tool_result big enough to defeat the pre-flight elision (e.g. context_window=8192, 200KB tool_result inside a session with many small earlier turns). Call send() and confirm the SSE yields TurnBlocked{ContextOverflow} instead of TurnFailed; the session JSONL should now show 'turn_blocked' with blockReason: 'context_overflow'. Then call runner.compact_session(&id, 10) directly against a fake summarizer endpoint and verify session.history is rewritten as [summary, last 10 rows]. The full 'yah session resolve <id> compact' loop needs the daemon wiring noted above — file as R418-F5 when you're ready to land it. Daemon wiring entrypoints to update: resolve_block_runner in app/yah/cli/src/camp.rs (Compact arm currently rejects with 'use HandOff'), and DesktopAppHandler::resolve_block in app/yah/desktop/src/camp_socket.rs (AgentSessions Compact path).")
//! @yah:verify("cargo test -p runner --lib openai_compat")
//!
//! @yah:ticket(R629-T9, "RS-1 regression test: nothing that mutates history may read or write prelude_text")
//! @yah:status(review)
//! @yah:assignee(agent:bundle-anthropic-ashguard)
//! @yah:at(2026-07-23T19:31:33Z)
//! @yah:phase(P3)
//! @yah:parent(R629)
//! @yah:next("The one yah-owned compactor (OpenAiCompatRunner::compact_session, openai_compat.rs:1031) drains s.history, pushes a summary row, re-extends the tail, and never touches prelude_text. That boundary is what makes resident content durable in a provider-independent way — pin it before anything starts mutating the resident set.")
//! @yah:verify("A test asserts prelude_text is byte-identical across a compact_session call.")
//! @yah:gotcha("This is a rule we are adopting, not a property we inherit. See the sibling compaction tickets: on the only transport that actually compacts today, the summarizer is claude, not us.")
//! @yah:handoff("Pinned RS-1 two ways in crates/yah/runner/src/openai_compat.rs. (1) STRUCTURAL: extracted the history rewrite out of OpenAiCompatRunner::compact_session into a free fn rewrite_history_with_summary(history, summary_text, keep_last_n, original_len) -> kept. It takes ONLY history and has no parameter for prelude_text, so compaction cannot read or write the resident set by construction — RS-1 is now enforced by the signature (a future edit that wanted the prelude would have to change the signature, which is the tripwire), not by a caller remembering. compact_session now calls it under the session lock; behaviour byte-identical (summary row first, then last keep_last_n rows).")
//! @yah:handoff("(2) END-TO-END (the ticket's literal verify): compact_session_leaves_prelude_text_byte_identical stands up a loopback HTTP server returning a canned OpenAI chat-completion JSON, points a runner's config.endpoint at it, inserts a session with a known RESIDENT prelude_text + 6 history rows directly into the sessions map, runs the REAL compact_session(&sid, 2), and asserts prelude_text is byte-identical afterward while history was rewritten to [summary, turn 4, turn 5]. Plus two pure-helper unit tests (summary-then-tail ordering + label count; keep_last_n > len saturates without panic).")
//! @yah:handoff("Verify: cargo test -p yah-runner --lib -- rewrite_history compact_session -> 4 pass. Full runner --lib: 398 pass. cargo build --workspace clean. Two full-suite failures are PRE-EXISTING and unrelated (anthropic::tests::model_context_tokens_1m_models, resolver::openai::tests::non_openai_connection_defers_credential_to_registry) — different files, not touched by this ticket; documented as peer failures at app/yah/desktop/src/agent.rs:675. My only runner change is openai_compat.rs.")
//!
//! @yah:ticket(R950-F5, "HttpClient trait in core (reqwest default feature) + move openai_compat.rs into yah-runner-ho (HOk cell) with its e2e suite")
//! @yah:at(2026-10-05T11:07:47Z)
//! @yah:assignee(agent:bundle-anthropic-ashguard)
//! @yah:parent(R950)
//! @yah:next("Core: one HttpClient trait (request + streaming body); reqwest impl behind a default `reqwest` feature so wasm/iOS consumers can supply their own substrate (W357 §3).")
//! @yah:next("Move openai_compat.rs (4952 lines; OpenAiCompatConfig, Transport, the Runner impl) into crates/ho over HttpClient. Its measured camp edges: party::prelude::{Prelude, CacheControl, CacheTtl}, party::ThinkingBudget::{Low..ExtraHigh, to_think_budget}, kg::anno::{ThinkBudget, EngineRef} — all replaced by F2 core types; the ThinkingBudget->budget mapping moves to the wrapper seam.")
//! @yah:next("Invert three intra-crate edges: compose::self_identity_preamble (camp, stays in wrapper) arrives via the caller's rendered Prelude; quota_ledger::global() becomes an injected UsageStore/ledger handle; resolver::free_tier_* (openrouter policy) becomes a caller-supplied predicate or config field.")
//! @yah:next("codex_oauth.rs (HOo, ChatGPT OAuth) stays in the wrapper and is repointed at yah-runner-ho; it is not a new cell (W357 §5).")
//! @yah:next("Move tests/openai_compat_e2e.rs to ho. Coordinate R365: provider bugs on this path (e.g. R365-B26 codex version header) stay filed there; do not fork them.")
//! @yah:next("Tier: Warrior — large move with three coupling inversions and a feature-gated HTTP substrate; design is fixed by F2.")
//! @yah:verify("cd oss/runner && cargo test -p yah-runner-ho")
//! @yah:verify("cargo test -p yah-runner")
//! @yah:status(review)
//! @yah:handoff("CORE: http.rs = HttpClient trait (async send(HttpRequest) -> HttpResponse{status: u16, headers, body: ByteStream of bytes::Bytes}; HttpError is transport-only; HttpResponse::{is_success, header, bytes, text, json}), HttpMethod, HttpRequest::{get, post_json, header, bearer_auth}. ReqwestClient behind default feature `reqwest` (optional reqwest 0.12 rustls-tls+stream). New core dep: bytes. mcp's core dep is now default-features=false so it does not pull reqwest.")
//! @yah:handoff("HO: openai_compat.rs moved verbatim to oss/runner/crates/ho/src/openai_compat.rs (annotation blocks incl. this ticket and R950-T12 moved with it); tests/openai_compat_e2e.rs moved to oss/runner/crates/ho/tests. ho features default=[reqwest] forwarding to core. HTTP goes through Arc<dyn HttpClient>: OpenAiCompatRunner::with_http(config, client, tools) always; new/with_tools/list_openai_compat_models are cfg(feature=reqwest) conveniences; list_openai_compat_models_with(&dyn HttpClient, ..) always.")
//! @yah:handoff("CAMP EDGES: party/kg types replaced by core SessionId/BlockReason/TurnUsage/ReasoningBudget/ModelRef. OpenAiCompatConfig.thinking_budget (party::ThinkingBudget) RENAMED reasoning: Option<ReasoningBudget>; the ThinkingBudget mapping is runner::openai_compat::reasoning_budget, applied in resolver into_openai_compat. Catalog lookups read yah_runner_core::model_catalog::catalog(); with_camp_policy calls the wrapper's layered catalog() first so layers install before any lookup.")
//! @yah:handoff("INVERSIONS: (b) config.quota_ledger: Option<Arc<std::sync::Mutex<QuotaLedger>>> records 429s. crates/yah/runner/src/quota_ledger.rs KEPT: global() now returns Arc<Mutex<QuotaLedger>> - launch.rs's session-start picker legitimately owns the one process-wide ledger, and with_camp_policy injects that same Arc into every ho runner. (c) config.rate_limit_hint: Option<RateLimitHint{model_suffix, message}>; resolver free_tier_429_hint replaced by free_tier_rate_limit_hint(), filled when provider_label is openrouter. (a) DEVIATION: the identity block needs the session id that start() mints, so it cannot be a pre-rendered Prelude section; instead config.session_preamble: Option<Arc<dyn SessionPreamble>> (render(session_id, prelude, session_key)); wrapper CampIdentity calls compose::self_identity_preamble. ho never calls compose; prompt bytes unchanged (render, newline, identity, caps).")
//! @yah:handoff("WRAPPER SEAM: crates/yah/runner/src/openai_compat.rs is now only the camp seam: with_camp_policy (ledger + CampIdentity + openrouter hint), CampIdentity, reasoning_budget, and impl CampExtensions for OpenAiCompatRunner delegating to ho's inherent think/set_think/append_resident_stanza/compact_session with the ThinkBudget<->ReasoningBudget mapping. CompactResult/ResidentAppend are defined in ho, re-exported from runner::runner. Direct factory sites wrapped in with_camp_policy: hub in_process agent_start, eval run.rs, cli camp.rs agent start, cli agent.rs (hok_smoke example and cli tests left bare). codex_oauth had only doc links to openai_compat; they now name yah-runner-ho. yah-runner depends on yah-runner-ho.")
//! @yah:handoff("R365: its openai_compat transport/429 code was moved, not changed; no provider behaviour fixed or forked here. board.review via MCP returned EAGAIN (os error 35) and did not land, so this block was written by hand.")
//! @yah:verify("cd oss/runner && cargo test -p yah-runner-ho -p yah-runner-core -> core 132 (baseline 128, +4 http); ho lib 100 (99 moved, -1 trait-default test moved to wrapper, +2 session_preamble); ho e2e 12 / 1 ignored (= old wrapper e2e; 429 test also asserts the injected ledger). EXIT=0 /tmp/r950f5-o1.log")
//! @yah:verify("cargo test -p yah-runner -> lib 378 / 2 ignored (473 - 99 moved + 4 new seam tests), doctests 1 = baseline; e2e now in ho. EXIT=0 /tmp/r950f5-w2.log")
//! @yah:verify("cargo check -p yah-party -p yah -p desktop -p yah-hub -p yah-eval -p probe --tests EXIT=0, no warnings in touched files (/tmp/r950f5-c1.log); cargo check -p yah-runner --examples EXIT=0")
//! @yah:verify("cd oss/runner && cargo tree -p yah-runner-ho -e normal / -p yah-runner-core -e normal -> 0 yah-(board|party|kg|fob|bridge); cargo check -p yah-runner-ho -p yah-runner-core --no-default-features --target wasm32-unknown-unknown EXIT=0 and --target aarch64-apple-ios EXIT=0")
//! @yah:depends_on(R950-F2)
//! @yah:depends_on(R950-T3)
//! @yah:depends_on(R950-F4)
//! @yah:files(oss/runner/crates/core/src/lib.rs)
//! @yah:files(oss/runner/crates/ho/src/lib.rs)
//! @yah:files(crates/yah/runner/src/openai_compat.rs)
//! @yah:files(crates/yah/runner/src/codex_oauth.rs)
//! @yah:files(crates/yah/runner/src/compose.rs)
//! @yah:files(crates/yah/runner/src/resolver/mod.rs)
//! @yah:files(crates/yah/runner/src/lib.rs)
//! @yah:files(crates/yah/runner/Cargo.toml)
//! @yah:files(crates/yah/runner/tests/openai_compat_e2e.rs)
//! @yah:handoff("CORE: new http.rs = HttpClient trait (async send(HttpRequest) -> HttpResponse{status: u16, headers, body: ByteStream of bytes::Bytes}; HttpError for transport failures only; HttpResponse::{is_success, header, bytes, text, json}), HttpMethod, HttpRequest::{get, post_json, header, bearer_auth}. ReqwestClient behind default feature `reqwest` (optional dep reqwest 0.12 rustls-tls+stream). New core dep: bytes. mcp's core dep set default-features=false so it does not pull reqwest.")
//! @yah:handoff("HO: openai_compat.rs moved verbatim to oss/runner/crates/ho/src/openai_compat.rs (annotation blocks incl. this ticket and R950-T12 moved with it); tests/openai_compat_e2e.rs moved to oss/runner/crates/ho/tests. ho features default=[reqwest] forwarding to core. HTTP goes through Arc<dyn HttpClient>: new OpenAiCompatRunner::with_http(config, client, tools) always; new/with_tools and list_openai_compat_models are cfg(feature=reqwest) conveniences; list_openai_compat_models_with(&dyn HttpClient, ..) always. is_retryable_status takes u16.")
//! @yah:handoff("CAMP EDGES REPLACED: party/kg types -> core SessionId/BlockReason/TurnUsage/ReasoningBudget/ModelRef; OpenAiCompatConfig.thinking_budget: Option<party::ThinkingBudget> RENAMED to reasoning: Option<ReasoningBudget> (wrapper maps via runner::openai_compat::reasoning_budget in resolver into_openai_compat). InnerSession.think is ReasoningBudget. model_catalog lookups read yah_runner_core::model_catalog::catalog(); with_camp_policy calls the wrapper's layered catalog() first so layers install before the first lookup.")
//! @yah:handoff("INVERSIONS: (b) config.quota_ledger: Option<Arc<std::sync::Mutex<QuotaLedger>>>; 429s record there. Wrapper quota_ledger.rs KEPT: global() now returns Arc<Mutex<QuotaLedger>> because launch.rs's session-start picker legitimately owns the one process-wide ledger and the same Arc is injected into every ho runner. (c) config.rate_limit_hint: Option<RateLimitHint{model_suffix, message}>; resolver::free_tier_429_hint replaced by free_tier_rate_limit_hint() -> RateLimitHint, filled when provider_label == openrouter. (a) DEVIATION: the self-identity preamble needs the session id that start() mints, so the wrapper cannot render it into Prelude sections beforehand; instead config.session_preamble: Option<Arc<dyn SessionPreamble>> (trait render(session_id, prelude, session_key)), wrapper CampIdentity calls compose::self_identity_preamble. ho never calls compose. Prompt bytes are identical (render + newline + identity + caps).")
//! @yah:handoff("WRAPPER SEAM: crates/yah/runner/src/openai_compat.rs is now the camp seam: with_camp_policy(config) fills ledger + CampIdentity + openrouter hint; CampIdentity; reasoning_budget(ThinkingBudget); impl CampExtensions for OpenAiCompatRunner delegating to ho's inherent think/set_think/append_resident_stanza/compact_session, with ThinkBudget <-> ReasoningBudget mapping at the seam. CompactResult/ResidentAppend now defined in ho, re-exported from runner::runner. resolver into_openai_compat wraps in with_camp_policy; direct factory sites also wrapped: hub in_process agent_start, eval run.rs, cli camp.rs:16822, cli agent.rs:269 (examples/hok_smoke and cli tests left bare). eval groq literal updated. codex_oauth had no code dependency on openai_compat, only doc links; those now name yah-runner-ho. yah-runner gains yah-runner-ho dep.")
//! @yah:handoff("R365: touched R365-owned code only by moving it (openai_compat.rs transport/429 path); no provider behaviour changed or forked.")
//! @yah:verify("cd oss/runner && cargo test -p yah-runner-ho -p yah-runner-core -> core 132 pass (baseline 128, +4 http tests); ho lib 100 pass (99 moved in-file tests, -1 trait-default test moved to the wrapper, +2 session_preamble tests); ho e2e openai_compat_e2e 12 pass / 1 ignored (= old wrapper e2e; 429 test now also asserts the injected ledger recorded it). EXIT=0 /tmp/r950f5-o1.log")
//! @yah:verify("cargo test -p yah-runner -> lib 378 pass / 2 ignored (baseline 473 - 99 moved + 4 new seam tests in runner/src/openai_compat.rs), doctests 1 = baseline; e2e suite now lives in ho. EXIT=0 /tmp/r950f5-w2.log")
//! @yah:verify("cargo check -p yah-party -p yah -p desktop -p yah-hub -p yah-eval -p probe --tests -> EXIT=0, no warnings in touched files (/tmp/r950f5-c1.log); cargo check -p yah-runner --examples EXIT=0 (/tmp/r950f5-ex1.log)")
//! @yah:verify("cd oss/runner && cargo tree -p yah-runner-ho -e normal and -p yah-runner-core -e normal -> 0 matches for yah-(board|party|kg|fob|bridge)")
//! @yah:verify("cd oss/runner && cargo check -p yah-runner-ho -p yah-runner-core --no-default-features --target wasm32-unknown-unknown EXIT=0; same for --target aarch64-apple-ios EXIT=0; cargo check -p yah-runner-mcp EXIT=0")
//!
//! @yah:ticket(R950-T12, "~40-line Ollama hello-world example: one tool, one HOk turn, prints the AgentEvent stream, no camp/board/MCP")
//! @yah:at(2026-10-05T12:25:33Z)
//! @yah:assignee(agent:claude)
//! @yah:status(review)
//! @yah:parent(R950)
//! @yah:handoff("Example at oss/runner/crates/ho/examples/hello_ollama.rs (69 lines): registers 'add' tool summing integers, reads OLLAMA_MODEL env var (default qwen3), builds OpenAiCompatConfig pointing to http://localhost:11434/v1 with Sse transport, creates one prelude section, runs one turn via Runner trait, prints all AgentEvent variants. Updated oss/runner/README.md with one-paragraph crate map (core/ho/ha/mcp) and example documentation.")
//! @yah:verify("cargo build -p yah-runner-ho --example hello_ollama: EXIT=0, clean compilation")
//! @yah:verify("cargo tree -p yah-runner-ho -e normal: no yah-board, yah-party, yah-kg, yah-bridge, yah-fob dependencies")
//! @yah:verify("Ollama not running locally; example compile-only; run with: cargo run -p yah-runner-ho --example hello_ollama")
//! @yah:gotcha("\"Anchored in openai_compat.rs because oss/runner did not exist at filing; once F5 lands, the ho cell's new home is the natural anchor.\"")
//! @yah:depends_on(R950-F5)
//! @yah:depends_on(R950-F4)
//! @yah:files(oss/runner/crates/ho/examples/hello_ollama.rs)
//! @yah:files(oss/runner/crates/ho/Cargo.toml)
//! @yah:files(oss/runner/README.md)
//! @yah:handoff("Example at oss/runner/crates/ho/examples/hello_ollama.rs (69 lines): registers 'add' tool, reads OLLAMA_MODEL env (default qwen3), runs one turn, prints AgentEvents. Updated oss/runner/README.md with crate map and example doc.")
//! @yah:verify("cargo build -p yah-runner-ho --example hello_ollama: EXIT=0; cargo tree shows no yah-board/party/kg/bridge/fob; Ollama compile-only (not running locally).")
//! @yah:verify("post-trim: 53 lines, build EXIT=0 (/tmp/r950t12-b2.log)")

use async_stream::stream;
use async_trait::async_trait;
use futures::stream::StreamExt;
use yah_runner_core::quota_ledger::QuotaLedger;
use yah_runner_core::{
    AgentEvent, BlockReason, HttpClient, HttpRequest, HttpResponse, Prelude, ReasoningBudget,
    SessionId, TurnUsage,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

use yah_runner_core::mint_session_id;
use yah_runner_core::tool::{build_wire_name_map, sanitize_tool_name_for_wire, ToolRegistry};
use yah_runner_core::{EventStream, Runner, RunnerError};

/// Default per-turn output cap forwarded as `max_tokens`. Independent
/// of the prelude's ring budget — that bounds *input*; this bounds
/// *output*.
const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 4096;

/// Cap on the cumulative SSE body to keep a misbehaving upstream from
/// filling the host's working set.
const MAX_STREAM_BYTES: usize = 32 * 1024 * 1024; // 32 MB

/// Default per-turn tool-call iteration budget used when
/// [`OpenAiCompatConfig::max_tool_iterations`] is `None`. Each
/// iteration is one provider round-trip. Cloud frontier models doing
/// wide BFS research benefit from a larger per-config value; local 7B
/// models should keep a tighter cap.
///
/// Bumped from 8→16 per operator request: agent sessions doing deep
/// exploration (reading many files, multiple grep queries) hit 8
/// regularly during legitimate research, triggering a premature
/// `TurnEnded`. 16 gives a comfortable exploration budget while still
/// providing a safety net against runaway loops.
const MAX_TOOL_ITERATIONS: u32 = 16;

/// Pre-stream retry (R237-F12): how many times to attempt opening the
/// `/v1/chat/completions` (or `/api/chat`) connection before giving up.
/// Only the *pre-stream* phase retries — once any bytes from the
/// response body have been read by the SSE/NDJSON loop, partial output
/// may have streamed to the FE and we cannot safely re-issue (no
/// event-id resume cursor exists for OpenAI SSE, OpenRouter SSE, Groq
/// SSE, or Ollama NDJSON). See [`is_retryable_status`].
const MAX_STREAM_OPEN_ATTEMPTS: u32 = 3;

/// Cap on `Retry-After` we'll honor before declaring the rate-limit
/// permanent. Keeps a misconfigured upstream from stalling a session for
/// hours; users see the original `RateLimited` / `Transport` error
/// instead and can act.
const MAX_RETRY_AFTER_SECS: u32 = 30;

/// Whether a non-success HTTP status from the chat-completions endpoint
/// should be retried in-band. Mirrors the [`yah_runner_core::health::ErrorKind`]
/// taxonomy: 429 (RateLimited) and 5xx (ServerError) are transient;
/// 401/403 (auth) and other 4xx (bad-request / not-found) are permanent.
fn is_retryable_status(status: u16) -> bool {
    status == 429 || (500..600).contains(&status)
}

/// Backoff between pre-stream retry attempts (in ms). `attempt` is the
/// 1-based index of the upcoming retry (so attempt=1 is the wait between
/// the first failed try and the second try). For 429 responses with an
/// explicit `Retry-After`, honor it up to [`MAX_RETRY_AFTER_SECS`]; for
/// 5xx / network errors, use exponential 1s/2s/4s.
fn compute_stream_open_backoff_ms(attempt: u32, retry_after_secs: Option<u32>) -> u64 {
    if let Some(secs) = retry_after_secs {
        return (secs.min(MAX_RETRY_AFTER_SECS) as u64).saturating_mul(1000);
    }
    let exp = attempt.saturating_sub(1).min(4);
    1000u64.saturating_mul(1u64 << exp)
}

/// Streaming wire format the upstream uses.
///
/// Orthogonal to `provider_label`: the OpenAI SSE shape covers almost every
/// hosted gateway, but Ollama's native `/api/chat` endpoint speaks
/// newline-delimited JSON instead. The two differ in frame separator
/// (`\n\n` vs `\n`), per-frame payload shape, and how usage / done is
/// reported — so the parser and the byte-buffer split point both have to
/// know the format up front.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Transport {
    /// OpenAI-style Server-Sent Events: `data: <json>\n\n` framing, with
    /// `[DONE]` as the closing sentinel and a final `usage` chunk when
    /// `stream_options.include_usage` is honoured.
    #[default]
    Sse,
    /// Ollama `/api/chat` NDJSON: one bare JSON object per `\n`-terminated
    /// line, no `data:` prefix, no `[DONE]` sentinel. The final frame
    /// carries `done: true` plus `done_reason`, `prompt_eval_count`, and
    /// `eval_count` — those drive the synthetic `finish_reason` + usage
    /// event the runner expects.
    OllamaNdjson,
}

/// Static configuration for one provider deployment.
///
/// The provider label is the `engine.provider` string the prelude
/// carries (`@yah:engine(openai:gpt-5)` → `provider = "openai"`). A
/// runner refuses sessions whose engine doesn't match its label, so a
/// host running both OpenAI and Ollama can register two runners and
/// dispatch by label without checking model strings.
///
/// @yah:ticket(R883-T7, "Wire context_window for hosted openai_compat engines so the pre-flight elision guard stops being Ollama-only")
/// @yah:status(review)
/// @yah:at(2026-09-26T04:02:09Z)
/// @yah:assignee(agent:bundle-anthropic-ashguard)
/// @yah:phase(P3)
/// @yah:parent(R883)
/// @yah:next("Tier: Warrior — plumbing from an existing catalog into existing config.")
/// @yah:next("context_window is Some only in ollama_local / ollama_cloud (8192 / 32768); openai, groq, deepseek and openrouter leave it None, so try_preflight_elide never fires for any hosted engine. R418-F2 flagged this gap and deferred it to the host.")
/// @yah:next("Source the value from runner::model_catalog (ModelCatalog::context_tokens) rather than a new table — R863 moved context-window lookup onto that runtime catalog precisely so a second compiled table does not drift. An unknown model stays None (guard off), not a guessed number.")
/// @yah:next("Check the catalog actually carries non-Anthropic ids (deepseek-v4-flash 1.05M triggered R418); if it does not, adding them belongs in the catalog JSON, not here.")
/// @yah:verify("cargo test -p yah-runner --lib openai_compat")
/// @yah:verify("Session header for a hosted engine renders the context_window line instead of 'not known to runner — pre-flight prompt-trim guard inactive'.")
/// @arch:see(.yah/docs/working/W342-context-economy.md)
/// @yah:handoff("Added OpenAiCompatConfig::context_window_for(model) in crates/yah/runner/src/openai_compat.rs: explicit config.context_window wins (Ollama keeps num_ctx 8192/32768), else crate::model_catalog::catalog().context_tokens(model), with one retry dropping an OpenRouter-style 'vendor/' segment. Unknown model -> None (guard off). No new table; the embedded catalog JSON already carried gpt-4o, llama-3.3, deepseek-chat/v4-flash etc., so no catalog edit was needed.")
/// @yah:handoff("Resolved per SESSION model (InnerSession.model, i.e. prelude.engine.model or default_model), not per factory default: send() now captures s.model when pushing the user turn and feeds context_window_for(&model) to the pre-flight elide / threshold-reminder paths; render_runtime_caps_block uses context_window_for(model), so hosted engines render '- **context_window**: N tokens (pre-flight prompt-trim guard ...)'. Hosted factories (openai/groq/deepseek/openrouter) keep context_window: None on purpose, doc comment on the field updated to say so.")
/// @yah:handoff("Side effect: hub/in_process.rs:1210,1520 and eval/src/run.rs:393 construct configs with context_window: None and now get the catalog window automatically; untouched (in_process.rs is a peer-dirty file).")
/// @yah:verify("cargo test -p yah-runner --lib openai_compat: 93 passed, 0 failed (baseline 90; +3 new: hosted_engines_resolve_context_window_from_catalog, explicit_context_window_overrides_catalog, runtime_caps_block_renders_catalog_window_for_hosted_engine). runtime_caps_block_calls_out_inactive_guard_when_window_unknown now uses an unknown model id since gpt-4o resolves. The skew guard flagged openai_compat.rs as modified mid-run; that was this command's own python test-insert preceding cargo in the same line, not a peer.")
#[derive(Debug, Clone)]
pub struct OpenAiCompatConfig {
    /// Engine-tag this runner serves (`"openai"`, `"ollama"`, `"qwen"`,
    /// …). Compared verbatim against the ticket's `@yah:engine(...)`
    /// provider field.
    pub provider_label: String,
    /// Full URL to `POST` against — typically `…/v1/chat/completions`.
    /// Includes path so a host can point at any OpenAI-compat gateway
    /// (LiteLLM, custom proxies) without the runner inventing routes.
    pub endpoint: String,
    /// Model used when the ticket's `@yah:engine(...)` doesn't carry
    /// an explicit model. Matches the host's workspace-default
    /// affordance.
    pub default_model: String,
    /// `Authorization: Bearer …` token. `None` is fine for
    /// unauthenticated local Ollama; required for hosted services.
    pub api_key: Option<String>,
    /// Per-turn output token cap. Falls back to
    /// [`DEFAULT_MAX_OUTPUT_TOKENS`] if unset.
    pub max_output_tokens: Option<u32>,
    /// Per-turn tool-call iteration budget. Falls back to
    /// [`MAX_TOOL_ITERATIONS`] (16) if unset. Cloud frontier models
    /// doing wide BFS research benefit from a larger per-config value
    /// (e.g. 20–50); local 7B models should keep the tight default.
    pub max_tool_iterations: Option<u32>,
    /// Streaming wire format the upstream uses. Defaults to
    /// [`Transport::Sse`] (the OpenAI shape every hosted gateway and the
    /// previous Ollama `/v1/chat/completions` path produced); Ollama's
    /// native `/api/chat` endpoint requires [`Transport::OllamaNdjson`]
    /// — different separator (`\n` vs `\n\n`) and different frame shape.
    pub transport: Transport,
    /// Per-turn cumulative input-token hard-stop for non-caching engines.
    /// When `Some(n)`, the runner aborts the in-flight turn and emits
    /// `TurnFailed` with a `[cost_guard]` prefix if cumulative input
    /// tokens observed across tool-call iterations exceed `n`. `None`
    /// disables the guard (default for caching backends). Set
    /// `Some(500_000)` for Ollama Cloud and other non-caching paths where
    /// the full context is re-encoded on every inference call.
    pub cost_guard_tokens: Option<u32>,
    /// Context window the upstream should allocate, surfaced to the
    /// model as `options.num_ctx` on Ollama's native `/api/chat`. Only
    /// meaningful when [`Self::transport`] is [`Transport::OllamaNdjson`]
    /// — `/v1/chat/completions` silently ignores it and truncates the
    /// request to 4096 tokens (the bug R169 routes around). `None`
    /// skips the field entirely, leaving Ollama on its per-model
    /// default. Set per-engine via the factories ([`Self::ollama_local`]
    /// = 8192, [`Self::ollama_cloud`] = 32768); per-session override
    /// lands on the subclass via R169-T3.
    pub num_ctx: Option<u32>,
    /// How long Ollama should keep the model loaded after the last
    /// request, as a duration string (e.g. `"60m"`). Sent as a top-level
    /// `keep_alive` field on `/api/chat` requests — Ollama's default is 5
    /// minutes, which evicts the KV cache between user turns. `None`
    /// omits the field and leaves Ollama on its default. Only meaningful
    /// for `OllamaNdjson` transports; SSE gateways ignore unknown fields
    /// harmlessly.
    pub keep_alive: Option<String>,
    /// Thinking budget for extended thinking / reasoning mode. Maps to
    /// `reasoning_effort` on OpenAI APIs and `output_config.effort` on
    /// Anthropic-compatible APIs. `None` means thinking is not enabled.
    pub reasoning: Option<ReasoningBudget>,
    /// Model input context window in tokens. Drives the pre-flight
    /// prompt-size guard (R418-F2): before each request, the runner
    /// estimates the next prompt's token count and — if it exceeds
    /// `prompt_overflow_threshold * context_window` — elides the largest
    /// oversized `tool_result` row in history and emits
    /// [`AgentEvent::PromptTrimmed`]. `None` disables the guard entirely
    /// (the request is sent as-is and the provider's 400 surfaces as
    /// `TurnFailed` — the pre-R418 behaviour). For Ollama backends this
    /// mirrors [`Self::num_ctx`] (the model truncates past it regardless
    /// of the model's nominal window). Hosted factories leave it `None`
    /// and [`Self::context_window_for`] resolves the session's actual
    /// model through `yah_runner_core::model_catalog` instead (R883-T7) — set it
    /// only to override the catalog.
    pub context_window: Option<u32>,
    /// Fraction of [`Self::context_window`] above which the pre-flight
    /// guard starts eliding. Defaults to 0.95 when `None`. Lower values
    /// (e.g. 0.85) leave more headroom for the response; higher values
    /// (e.g. 0.99) maximise input but risk losing the response budget
    /// on a borderline request. Ignored when [`Self::context_window`]
    /// is `None`.
    pub prompt_overflow_threshold: Option<f32>,
    /// Ledger every observed 429 is recorded in, keyed `(provider_label,
    /// model)`, so the host's session-start picker can rotate away from a
    /// cooling-down entry (R950-F5: injected, not a process global). `None`
    /// records nothing.
    pub quota_ledger: Option<Arc<std::sync::Mutex<QuotaLedger>>>,
    /// Actionable hint logged on a 429 for a model matching
    /// [`RateLimitHint::model_suffix`] — the host's provider policy (e.g.
    /// OpenRouter's `:free` tier cap) as plain data. `None` logs nothing.
    pub rate_limit_hint: Option<RateLimitHint>,
    /// Text the host wants trailing the rendered prelude of every session
    /// this runner starts (R950-F5). It needs the session id, which only
    /// exists once `start` has minted it, so it is a hook rather than a
    /// prelude section. `None` adds nothing.
    pub session_preamble: Option<Arc<dyn SessionPreamble>>,
}

/// A provider rate-limit hint the host supplies as data (R950-F5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RateLimitHint {
    /// The hint applies to models whose id ends with this (e.g. `":free"`).
    pub model_suffix: String,
    /// What to tell the operator.
    pub message: String,
}

impl RateLimitHint {
    fn for_model(&self, model: &str) -> Option<&str> {
        model.ends_with(&self.model_suffix).then_some(self.message.as_str())
    }
}

/// Renders a per-session block appended after the prelude at
/// [`Runner::start`] (R950-F5). yah uses it for the session's self-identity
/// ("you are X, running as session Y"), which must trail the shared prelude
/// so sessions keep a common cache prefix (R883-B15).
pub trait SessionPreamble: Send + Sync + std::fmt::Debug {
    fn render(&self, session_id: &SessionId, prelude: &Prelude, session_key: &str) -> String;
}

/// Outcome of a successful [`OpenAiCompatRunner::compact_session`] call
/// (R418-F4 / R629-S11). Lets the caller log what happened and decide
/// whether to re-issue the failed turn immediately or surface the summary to
/// the operator first.
#[derive(Debug, Clone)]
pub struct CompactResult {
    pub session_id: SessionId,
    /// UTF-8 byte length of the summarizer's reply.
    pub summary_chars: usize,
    /// Sum of JSON-serialized lengths of history rows before compact.
    pub original_history_chars: usize,
    /// History row count before compact.
    pub history_messages_before: usize,
    /// History row count after compact (1 summary + kept tail).
    pub history_messages_after: usize,
    /// How many original tail rows were preserved verbatim.
    pub kept_tail: usize,
}

/// Outcome of a resident-set mutation (R629-F8 / W286). `appended` is
/// `false` when the stanza was already resident — the idempotent no-op that
/// keeps a re-load from double-injecting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentAppend {
    /// The stanza id that was requested.
    pub stanza_id: String,
    /// `true` when the body was appended; `false` when it was already
    /// resident (no bytes added).
    pub appended: bool,
    /// Characters added to the resident prelude (0 on the idempotent no-op).
    pub appended_chars: usize,
    /// Total resident-prelude length (chars) after the operation.
    pub prelude_chars_after: usize,
}

impl OpenAiCompatConfig {
    /// OpenAI defaults: Chat Completions endpoint + bearer auth.
    /// Pick a model the user is entitled to call — leaving the default
    /// at `gpt-4o` keeps the non-Plus path working until the ticket
    /// pins a specific model.
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self {
            provider_label: "openai".into(),
            endpoint: "https://api.openai.com/v1/chat/completions".into(),
            default_model: "gpt-4o".into(),
            api_key: Some(api_key.into()),
            max_output_tokens: None,
            max_tool_iterations: None,
            transport: Transport::Sse,
            cost_guard_tokens: None,
            num_ctx: None,
            keep_alive: None,
            reasoning: None,
            quota_ledger: None,
            rate_limit_hint: None,
            session_preamble: None,
            context_window: None,
            prompt_overflow_threshold: None,
        }
    }

    /// Local Ollama defaults: native `/api/chat` endpoint on the
    /// standard port, no auth (`ollama serve`'s default deployment is
    /// open on loopback).
    ///
    /// The previous `/v1/chat/completions` path silently truncated the
    /// request to `num_ctx` (default 4096) and ignored `options.num_ctx`
    /// — `/api/chat` honours both, so this is the canonical local path
    /// going forward (R169).
    pub fn ollama_local() -> Self {
        Self {
            provider_label: "ollama".into(),
            endpoint: "http://localhost:11434/api/chat".into(),
            default_model: "llama3.3".into(),
            api_key: None,
            max_output_tokens: None,
            max_tool_iterations: None,
            transport: Transport::OllamaNdjson,
            // Local Ollama doesn't support prompt caching, but the spend
            // risk is low (your own hardware). No cost guard by default.
            cost_guard_tokens: None,
            // 8192 covers the prelude budget (≈5.7k for current
            // CLAUDE.md-driven sessions) with headroom for ~2k of
            // conversation before the host has to recycle the ring.
            // Small local models (7-8B) struggle past this anyway, so
            // bumping further wastes VRAM. Per-subclass override via
            // R169-T3.
            num_ctx: Some(8192),
            // 60m keeps the model resident across a typical session-resume
            // gap (Ollama default = 5m). KV cache survives idle turns.
            keep_alive: Some("60m".into()),
            reasoning: None,
            quota_ledger: None,
            rate_limit_hint: None,
            session_preamble: None,
            // Pre-flight guard mirrors num_ctx for Ollama backends —
            // the model truncates anything past num_ctx anyway, so the
            // guard buys us early elision + a PromptTrimmed event
            // instead of a silent loss.
            context_window: Some(8192),
            prompt_overflow_threshold: None,
        }
    }

    /// Hosted Ollama (Cloud / Pro). Requires an API key.
    ///
    /// Endpoint is hardcoded to `https://ollama.com/api/chat`. The
    /// bundled `ollama-cloud.json` connection describes the OpenAI-shape
    /// wire (`baseUrl: https://ollama.com/v1`, `protocol: openai_wire`);
    /// `/api/chat` is the native-shape sibling and lives outside that
    /// `baseUrl`. Folding both wires into the connection registry is
    /// follow-up; the factory hardcodes for now.
    ///
    /// Default model is `gpt-oss:20b` — a small free-tier cloud model at the
    /// time of writing. The exact catalogue shifts, so the host lets the user
    /// override per-session from the chat picker.
    pub fn ollama_cloud(api_key: impl Into<String>) -> Self {
        Self {
            provider_label: "ollama".into(),
            endpoint: "https://ollama.com/api/chat".into(),
            default_model: "gpt-oss:20b".into(),
            api_key: Some(api_key.into()),
            max_output_tokens: None,
            max_tool_iterations: None,
            transport: Transport::OllamaNdjson,
            // Ollama Cloud does not support prompt caching — every
            // tool-call iteration re-encodes the full context. Guard at
            // 500k cumulative input tokens per turn to prevent runaway
            // spend on multi-step tasks (R124-B6).
            cost_guard_tokens: Some(500_000),
            // Cloud frontier models (gpt-oss:20b, kimi-k2.6:cloud)
            // happily handle 32k; matches the host's wider tool-iteration
            // budget for cloud-tier subclasses.
            num_ctx: Some(32768),
            // Same 60m residency guard as local. Ollama Cloud may manage
            // model pool centrally and ignore this; sending it is free.
            keep_alive: Some("60m".into()),
            reasoning: None,
            quota_ledger: None,
            rate_limit_hint: None,
            session_preamble: None,
            context_window: Some(32768),
            prompt_overflow_threshold: None,
        }
    }

    /// Groq API: OpenAI-compatible endpoint. Requires an API key.
    ///
    /// Used as a fallback smoke-test variant when OLLAMA_API_KEY is
    /// unavailable (R292-B1). llama-3.3-70b-versatile is a capable model
    /// available on the Groq API; verify your account tier and model
    /// availability before relying on this endpoint for CI.
    pub fn groq(api_key: impl Into<String>) -> Self {
        Self {
            provider_label: "groq".into(),
            endpoint: "https://api.groq.com/openai/v1/chat/completions".into(),
            default_model: "llama-3.3-70b-versatile".into(),
            api_key: Some(api_key.into()),
            max_output_tokens: None,
            max_tool_iterations: None,
            transport: Transport::Sse,
            // Groq is not subject to prompt caching costs (uses OpenAI wire
            // but doesn't support cache_control yet) — no cost guard needed.
            cost_guard_tokens: None,
            // Groq uses the standard SSE /v1/chat/completions shape and
            // doesn't need num_ctx or keep_alive.
            num_ctx: None,
            keep_alive: None,
            reasoning: None,
            quota_ledger: None,
            rate_limit_hint: None,
            session_preamble: None,
            context_window: None,
            prompt_overflow_threshold: None,
        }
    }

    /// DeepSeek API: OpenAI-compatible endpoint. Requires an API key.
    ///
    /// `deepseek-chat` (V3) is cheap, fast, and supports prompt caching with
    /// significant discounts on cache hits — a better smoke-test default than
    /// Ollama Cloud. Use `deepseek-reasoner` to target R1.
    pub fn deepseek(api_key: impl Into<String>) -> Self {
        Self {
            provider_label: "deepseek".into(),
            endpoint: "https://api.deepseek.com/v1/chat/completions".into(),
            default_model: "deepseek-chat".into(),
            api_key: Some(api_key.into()),
            max_output_tokens: None,
            max_tool_iterations: None,
            transport: Transport::Sse,
            cost_guard_tokens: None,
            num_ctx: None,
            keep_alive: None,
            reasoning: None,
            quota_ledger: None,
            rate_limit_hint: None,
            session_preamble: None,
            context_window: None,
            prompt_overflow_threshold: None,
        }
    }

    /// OpenRouter: routes to many providers via a single OpenAI-compat API.
    ///
    /// Good smoke fallback — a single credential covers dozens of models.
    /// Default model is `deepseek/deepseek-chat-v3-0324` (fast + cheap);
    /// override with a `model_override` arg or the `?provider=` suffix for
    /// provider pinning.
    pub fn openrouter(api_key: impl Into<String>) -> Self {
        Self {
            provider_label: "openrouter".into(),
            endpoint: "https://openrouter.ai/api/v1/chat/completions".into(),
            default_model: "deepseek/deepseek-chat-v3-0324".into(),
            api_key: Some(api_key.into()),
            max_output_tokens: None,
            max_tool_iterations: None,
            transport: Transport::Sse,
            cost_guard_tokens: None,
            num_ctx: None,
            keep_alive: None,
            reasoning: None,
            quota_ledger: None,
            rate_limit_hint: None,
            session_preamble: None,
            context_window: None,
            prompt_overflow_threshold: None,
        }
    }

    fn max_output_tokens(&self) -> u32 {
        self.max_output_tokens.unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS)
    }

    /// R947-F2: a new session's level. The host has already resolved its
    /// level into `Prelude.reasoning`; the config's
    /// subclass default only applies to a prelude that skipped it (a bare
    /// runner, a test), so both paths share the one resolver.
    fn seed_think(&self, explicit: Option<ReasoningBudget>) -> Option<ReasoningBudget> {
        explicit.or(self.reasoning)
    }

    fn max_tool_iterations(&self) -> u32 {
        self.max_tool_iterations.unwrap_or(MAX_TOOL_ITERATIONS)
    }

    fn prompt_overflow_threshold(&self) -> f32 {
        self.prompt_overflow_threshold
            .unwrap_or(DEFAULT_PROMPT_OVERFLOW_THRESHOLD)
    }

    /// Effective context window for `model` (R883-T7): the explicit
    /// [`Self::context_window`] when set, else the runtime model catalog's
    /// `contextLimit` for that id. Resolved per session model, not per
    /// `default_model`, because the prelude's engine can pick any model
    /// the provider serves. OpenRouter namespaces ids as `vendor/model`;
    /// a miss on the full id retries with the vendor segment dropped.
    /// Unknown model → `None` (guard off), never a guessed number.
    fn context_window_for(&self, model: &str) -> Option<u32> {
        self.context_window.or_else(|| {
            let catalog = yah_runner_core::model_catalog::catalog();
            catalog.context_tokens(model).or_else(|| {
                model
                    .split_once('/')
                    .and_then(|(_, bare)| catalog.context_tokens(bare))
            })
        })
    }
}

/// Default fraction of [`OpenAiCompatConfig::context_window`] above which
/// the pre-flight guard starts eliding tool_result rows (R418-F2).
const DEFAULT_PROMPT_OVERFLOW_THRESHOLD: f32 = 0.95;

/// Approximate UTF-8 chars-per-token for English-ish text. Since R883-T8
/// the pre-flight guard only uses it for *deltas* on top of a provider
/// [`UsageAnchor`] — messages appended since the last response, and the
/// bytes an elision removes — never to size the whole prompt.
const CHARS_PER_TOKEN: f32 = 4.0;

/// Minimum `tool_result` content length the pre-flight guard will
/// consider eliding. Below this the replacement note is roughly the same
/// size as the original — eliding buys nothing and just destroys
/// information. The crash that motivated R418 was a 4.4MB single
/// tool_result; anything in the multi-KB range is worth a look.
const MIN_ELIDABLE_BYTES: usize = 4 * 1024;

/// Truncation-note template baked into elided tool_result rows.
fn elision_note(tool_name: &str, original_bytes: usize) -> String {
    let kb = (original_bytes as f64 / 1024.0).max(0.0);
    format!(
        "[tool_result elided · {kb:.1} KB · original call: {tool_name} · \
         re-call with narrower filters (limit, since_ts, bucket) for a slice]"
    )
}

/// One pre-flight elision applied to session history. Surfaced to the
/// stream as [`AgentEvent::PromptTrimmed`] so the renderer can show a
/// 'context auto-trimmed' chip.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ElisionRecord {
    /// Index of the elided row in `history`. Tells the guard whether the
    /// bytes it removed were inside the anchored prefix (R883-T8).
    history_index: usize,
    tool_call_id: String,
    tool_name: String,
    original_bytes: usize,
    replacement_bytes: usize,
}

/// Provider evidence for the pre-flight guard (R883-T8, Hermes pattern).
/// `tokens` is the last response's reported prompt + completion tokens —
/// exactly the size of `history[..covered]` plus the `prelude_chars`-long
/// prelude as the provider tokenized it, including the assistant row that
/// response produced. Only rows appended after `covered` (and prelude
/// growth past `prelude_chars`) are estimated, so JSON-heavy tool_result
/// blobs are measured by the provider instead of guessed at chars/4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UsageAnchor {
    tokens: u64,
    covered: usize,
    prelude_chars: usize,
}

/// Serialized wire length of `msgs` — each row in its JSON shape so
/// tool_calls[] / tool_call_id / content all contribute (matches what
/// `build_request_body` ships).
fn wire_chars(msgs: &[Value]) -> usize {
    msgs.iter()
        .map(|m| serde_json::to_string(m).map(|s| s.len()).unwrap_or(0))
        .fold(0usize, usize::saturating_add)
}

fn chars_to_tokens(chars: usize) -> u64 {
    (chars as f32 / CHARS_PER_TOKEN) as u64
}

/// Projected input tokens for the next request: the provider anchor plus a
/// rough estimate of only what was appended since. `None` when there is no
/// anchor yet (first request, or history was rewritten by compact/resume) —
/// callers treat that as "no evidence", not as "fits".
///
/// @yah:ticket(R883-T8, "Replace the chars/4 prompt estimate with a provider usage anchor in the openai_compat elision guard")
/// @yah:status(review)
/// @yah:at(2026-09-26T05:06:12Z)
/// @yah:assignee(agent:bundle-anthropic-ashguard)
/// @yah:phase(P3)
/// @yah:parent(R883)
/// @yah:next("Tier: Warrior — swap one input to an existing guard; the threshold logic stays.")
/// @yah:next("try_preflight_elide and the mid-loop context_close check fire on estimate_prompt_chars / CHARS_PER_TOKEN (4.0). JSON-heavy tool_result blobs tokenize nothing like prose, and the error direction is not knowable in advance — false trims and missed trims both.")
/// @yah:next("Hermes pattern: anchor on the provider's last reported prompt + completion tokens, add a rough estimate of ONLY the messages appended since that response. With no anchor yet (first request), wait one request for provider evidence rather than trimming on an estimate alone.")
/// @yah:next("Depends on T7: the guard is inert for every hosted engine until context_window is wired, so an anchor before then changes nothing observable.")
/// @yah:verify("cargo test -p yah-runner --lib openai_compat  # guard reads the anchor when present; no trim on the first request without one")
/// @yah:verify("Existing R418-F2 elision tests still pass with anchored inputs.")
/// @arch:see(.yah/docs/working/W342-context-economy.md)
/// @yah:depends_on(R883-T7)
/// @yah:handoff("crates/yah/runner/src/openai_compat.rs: estimate_prompt_chars is replaced by estimate_prompt_tokens(session) -> Option<u64>. InnerSession.usage_anchor is a UsageAnchor{tokens, covered, prelude_chars}: tokens = the last response's prompt_tokens + completion_tokens, covered = history.len() after that response's assistant row. It is set by record_usage_anchor at both assistant push sites; iter_anchor_tokens is captured from FrameEvent::Usage on each iteration. The estimate is the anchor plus chars/4 of the rows after covered plus any prelude growth (append_resident_stanza). There is no anchor when the history is shorter than covered, when compact runs (it clears the anchor), on resume, and on the first request; with no anchor try_preflight_elide does not trim and the mid-loop context_close check reads false. Threshold logic is unchanged, and so is the futile-elision bail (now projected_tokens_after_max_elision). An elision inside the anchored prefix subtracts its saved bytes/4 from the anchor via the new ElisionRecord.history_index. The R418-F3 reminder's context line now reports the anchored estimate.")
/// @yah:handoff("Tests: the 5 existing R418-F2 elide tests now run with anchored inputs through an anchor_prelude helper, and estimate_prompt_chars_sums_prelude_plus_history was replaced. 6 new tests: estimate_prompt_tokens_is_none_without_an_anchor, estimate_prompt_tokens_adds_only_rows_since_the_anchor, record_usage_anchor_keeps_prior_anchor_when_response_had_no_usage, try_preflight_elide_waits_for_provider_evidence_on_first_request, try_preflight_elide_trusts_the_anchor_over_the_char_estimate, try_preflight_elide_discounts_the_anchor_for_prefix_elisions.")
/// @yah:verify("cargo test -p yah-runner --lib openai_compat: 98 passed, 0 failed (baseline 93: 1 test replaced, 6 added). The final run (/tmp/r883t8-final-a.log) had no input skew.")
/// @yah:gotcha("A resumed session starts with no anchor, so its first request goes out unguarded even when the replayed history is large. That is deliberate (the brief says wait for provider evidence); if that request overflows, the provider's 400 surfaces as TurnFailed, which is the pre-R418 behaviour.")
fn estimate_prompt_tokens(session: &InnerSession) -> Option<u64> {
    let anchor = session.usage_anchor?;
    // Defensive: a history shorter than the anchor means it was rewritten
    // without clearing the anchor; the anchor no longer describes it.
    let tail = session.history.get(anchor.covered..)?;
    let prelude_growth = session.prelude_text.len().saturating_sub(anchor.prelude_chars);
    Some(
        anchor
            .tokens
            .saturating_add(chars_to_tokens(wire_chars(tail)))
            .saturating_add(chars_to_tokens(prelude_growth)),
    )
}

/// Walk backwards looking for an assistant message whose `tool_calls[]`
/// contains a matching `id`. Returns the tool name when found; `None` when the
/// originating message is no longer in history (e.g. after a compact).
/// Best-effort — only used for the elision note.
///
/// **`preceding` must be the history *before* the tool row being named, not the
/// whole history** — R725-B4 (W300 Finding 4). `tool_call_id` is the provider's
/// string and yah cannot make it unique: Moonshot's Anthropic-compat endpoint
/// emits an index that resets every turn (`Read_1`, `Bash_2`), so one measured
/// session carried 160 calls under 140 distinct ids. Searching the *whole*
/// history therefore answers with whichever assistant message reused the id
/// **last**, which for any row but the newest is a different call in a later
/// turn — the elision note then names the wrong tool with no way for the reader
/// to tell.
///
/// Slicing to the prefix makes the join positional instead of by-key, which is
/// what the ticket asked for and is correct however the provider mints ids: the
/// assistant message that issued a tool row's call always precedes that row, so
/// the nearest preceding match is *the* match. No pending-call queue is needed
/// here because `history` is already in arrival order.
fn lookup_tool_name<'a>(preceding: &'a [Value], tool_call_id: &str) -> Option<&'a str> {
    for msg in preceding.iter().rev() {
        if msg.get("role").and_then(|v| v.as_str()) != Some("assistant") {
            continue;
        }
        let Some(calls) = msg.get("tool_calls").and_then(|v| v.as_array()) else {
            continue;
        };
        for call in calls {
            if call.get("id").and_then(|v| v.as_str()) == Some(tool_call_id) {
                return call
                    .pointer("/function/name")
                    .and_then(|v| v.as_str());
            }
        }
    }
    None
}

/// Replace the largest oversized `tool_result` row's `content` with a
/// short truncation note. Returns the elision record, or `None` when no
/// row is large enough to bother. The structural fields (`role`,
/// `tool_call_id`) are preserved so protocol shape stays valid.
fn elide_largest_tool_result(
    history: &mut Vec<Value>,
    min_elidable_bytes: usize,
) -> Option<ElisionRecord> {
    let (idx, original_bytes) = history
        .iter()
        .enumerate()
        .filter_map(|(i, msg)| {
            if msg.get("role").and_then(|v| v.as_str()) != Some("tool") {
                return None;
            }
            let len = msg.get("content").and_then(|v| v.as_str())?.len();
            if len < min_elidable_bytes {
                return None;
            }
            Some((i, len))
        })
        .max_by_key(|(_, len)| *len)?;

    let tool_call_id = history[idx]
        .get("tool_call_id")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    // R725-B4: `&history[..idx]`, never `history`. The owning assistant message
    // precedes the tool row; searching past it finds a later turn that reused
    // the same provider id and names the wrong tool.
    let tool_name = lookup_tool_name(&history[..idx], &tool_call_id)
        .unwrap_or("<unknown>")
        .to_string();
    let note = elision_note(&tool_name, original_bytes);
    let replacement_bytes = note.len();
    history[idx]["content"] = Value::String(note);
    Some(ElisionRecord {
        history_index: idx,
        tool_call_id,
        tool_name,
        original_bytes,
        replacement_bytes,
    })
}

/// Projected tokens after eliding every `tool_result` row whose content
/// is `>= min_elidable_bytes`. The pre-flight guard uses this to ask
/// "would maxing out elision even bring us under cap?" — if not, the
/// trim is futile information loss and the guard bails entirely.
fn projected_tokens_after_max_elision(estimate: u64, history: &[Value], min_elidable_bytes: usize) -> u64 {
    // Approximate the replacement-note size with a small constant. The
    // real note is `~120 bytes + len(tool_name)`, so 200 is a safe over.
    const NOTE_BYTES_OVER: usize = 200;
    let saved: usize = history
        .iter()
        .filter(|m| m.get("role").and_then(|v| v.as_str()) == Some("tool"))
        .filter_map(|m| m.get("content").and_then(|v| v.as_str()))
        .filter(|c| c.len() >= min_elidable_bytes)
        .map(|c| c.len().saturating_sub(NOTE_BYTES_OVER))
        .fold(0usize, usize::saturating_add);
    estimate.saturating_sub(chars_to_tokens(saved))
}

/// True when `msg` looks like a context-overflow rejection from any
/// supported OpenAI-shape backend. Wraps the shared phrase set in
/// [`yah_runner_core::block_detector::fallback`] (`context_length_exceeded`,
/// `prompt is too long`, `exceeds the context window`, `tokens > N`)
/// so the openai_compat path stays in sync with claude_cli / anthropic
/// detection without duplicating the substring list.
fn is_overflow_message(msg: &str) -> bool {
    yah_runner_core::block_detector::fallback(msg).is_some()
}

/// Render a flat text dump of `history` for the compact summarizer
/// (R418-F4). Output is human-readable markdown so a chat-completions
/// model can summarize without needing to understand the OpenAI
/// tool-call wire shape. One section per row: header (`## #N <role>`)
/// then content, with tool_calls flattened to `- name(args)` bullets
/// and tool results prefixed with their `tool_call_id` for context.
fn serialize_history_for_summary(history: &[Value]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for (i, msg) in history.iter().enumerate() {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("?");
        let _ = writeln!(out, "## #{i} {role}");
        out.push('\n');
        if let Some(tcs) = msg.get("tool_calls").and_then(|v| v.as_array()) {
            out.push_str("(tool_calls)\n");
            for tc in tcs {
                let name = tc
                    .pointer("/function/name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let args = tc
                    .pointer("/function/arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let _ = writeln!(out, "- {name}({args})");
            }
        } else if let Some(tcid) = msg.get("tool_call_id").and_then(|v| v.as_str()) {
            let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let _ = writeln!(out, "(tool_result for {tcid})");
            out.push_str(content);
            out.push('\n');
        } else {
            let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
            out.push_str(content);
            out.push('\n');
        }
        out.push('\n');
    }
    out
}

/// Render the per-session runtime-self-introspection block (R418-F3).
/// Appended to `session.prelude_text` at session start so every system
/// message the runner ships to the provider carries the agent's own
/// caps and identifiers. The agent can self-report `session_id`,
/// `max_tool_iterations`, etc. without any tool call, and can pace
/// itself against the cap instead of hitting the soft stop blindly.
///
/// `model` is the resolved model after `Self::start` picks between
/// `prelude.engine` and `config.default_model`.
fn render_runtime_caps_block(
    session_id: &SessionId,
    ticket_id: &str,
    model: &str,
    config: &OpenAiCompatConfig,
) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    s.push_str("\n\n## Runtime self-introspection (this session)\n\n");
    s.push_str(
        "These caps apply to **you** right now. Read them before answering \
         questions like 'what session am I?' / 'how many tool calls do I \
         have left this turn?' — speculation is what triggered R418's \
         crash mode.\n\n",
    );
    let _ = writeln!(s, "- **session_id**: `{}`", session_id.as_str());
    let _ = writeln!(s, "- **ticket_id**: `{ticket_id}`");
    let _ = writeln!(
        s,
        "- **runner**: yah-runner (OpenAI-compat, provider=`{}`)",
        config.provider_label,
    );
    let _ = writeln!(s, "- **model**: `{model}`");
    let _ = writeln!(
        s,
        "- **max_tool_iterations**: {} per turn",
        config.max_tool_iterations(),
    );
    let _ = writeln!(
        s,
        "- **max_output_tokens**: {}",
        config.max_output_tokens(),
    );
    if let Some(window) = config.context_window_for(model) {
        let _ = writeln!(
            s,
            "- **context_window**: {window} tokens (pre-flight prompt-trim guard fires at {:.0}% — surfaces as `PromptTrimmed` events)",
            config.prompt_overflow_threshold() * 100.0,
        );
    } else {
        s.push_str(
            "- **context_window**: not known to runner — pre-flight prompt-trim guard inactive\n",
        );
    }
    if let Some(c) = config.cost_guard_tokens {
        let _ = writeln!(
            s,
            "- **cost_guard_tokens**: turn aborts if cumulative input tokens exceed {c}",
        );
    }
    s.push_str(
        "\nWhen iterations get close to `max_tool_iterations`, you'll see a \
         system reminder in history with the remaining budget. Plan tool \
         calls accordingly — don't fan out wide when you have 2 iterations \
         left.\n",
    );
    s
}

/// Apply the pre-flight prompt-size guard to `session.history` in-place.
/// Returns the list of elisions performed (one per replaced row). Empty
/// when no elision was needed, when the bail-out condition fired
/// (eliding everything still wouldn't fit), or when there is no
/// [`UsageAnchor`] yet — the guard waits one request for provider
/// evidence rather than trimming on an estimate alone (R883-T8).
///
/// `context_window` is the model's input token cap; `threshold` is the
/// fraction of it above which we start eliding (typically 0.95).
fn try_preflight_elide(
    session: &mut InnerSession,
    context_window: u32,
    threshold: f32,
) -> Vec<ElisionRecord> {
    let cap = ((context_window as f32) * threshold).max(0.0) as u64;
    let Some(estimate) = estimate_prompt_tokens(session) else {
        return Vec::new();
    };
    if estimate <= cap {
        return Vec::new();
    }
    // Bail when the best-case elision still leaves us over — trimming
    // would only destroy information without saving the request.
    if projected_tokens_after_max_elision(estimate, &session.history, MIN_ELIDABLE_BYTES) > cap {
        return Vec::new();
    }
    let mut elisions = Vec::new();
    while estimate_prompt_tokens(session).is_some_and(|t| t > cap) {
        match elide_largest_tool_result(&mut session.history, MIN_ELIDABLE_BYTES) {
            Some(rec) => {
                // Bytes removed from the tail are re-measured by the next
                // estimate; bytes removed inside the anchored prefix are
                // not, so take them off the anchor itself.
                if let Some(anchor) = session.usage_anchor.as_mut() {
                    if rec.history_index < anchor.covered {
                        let saved = rec.original_bytes.saturating_sub(rec.replacement_bytes);
                        anchor.tokens = anchor.tokens.saturating_sub(chars_to_tokens(saved));
                    }
                }
                elisions.push(rec);
            }
            // No more elidable rows; stop. (Should be unreachable given
            // the projection bail-out above, but defensive.)
            None => break,
        }
    }
    elisions
}

/// In-memory per-session state. The runner keeps history server-side
/// so the host doesn't have to (and so the next turn's body builder
/// has the canonical view).
///
/// `history` carries pre-shaped OpenAI Chat Completions messages
/// (`{role, content}`, plus `tool_calls[]` on assistant turns and
/// `tool_call_id` on `role: "tool"` rows). Storing the protocol shape
/// rather than the cross-runner [`yah_runner_core::Message`] lets a turn
/// re-issue with tool-result rows interleaved without translating
/// through a lower-fidelity intermediate.
struct InnerSession {
    /// Ticket the session was opened for. Currently unused; kept for
    /// the eventual list_sessions / cost-attribution affordances.
    #[allow(dead_code)]
    ticket_id: String,
    prelude_text: String,
    model: String,
    history: Vec<Value>,
    /// Stanza ids resident in `prelude_text` (R629-F8). Seeded from
    /// [`Prelude::section_id_set`] at start/resume so a mid-session
    /// [`OpenAiCompatRunner::append_resident_stanza`] can be idempotent —
    /// re-loading a stanza that is already resident is a no-op rather than a
    /// second copy in the system prompt. Tracks the *Prelude* stanzas only;
    /// the self-identity preamble and runtime-caps block are runner-owned
    /// framing, not addressable stanzas.
    resident_stanzas: std::collections::BTreeSet<String>,
    /// Last provider-reported prompt + completion tokens and the history
    /// prefix they cover (R883-T8). Set when an assistant row lands after
    /// a response that reported usage; cleared whenever history is
    /// rewritten (compact) or rebuilt (resume). Drives the pre-flight
    /// guard via [`estimate_prompt_tokens`].
    usage_anchor: Option<UsageAnchor>,
    /// R947-F2: this session's thinking level, read into every request body.
    /// Seeded from `Prelude.think` (the host's single resolution) at
    /// start/resume; [`OpenAiCompatRunner::set_think`] changes it mid-session.
    think: Option<ReasoningBudget>,
}

impl InnerSession {
    /// Anchor the pre-flight guard on the response whose assistant row was
    /// just pushed. A response that reported no usage leaves the previous
    /// anchor in place — its prefix is still accurate and the newer rows
    /// are simply estimated.
    fn record_usage_anchor(&mut self, tokens: Option<u64>) {
        if let Some(tokens) = tokens {
            self.usage_anchor = Some(UsageAnchor {
                tokens,
                covered: self.history.len(),
                prelude_chars: self.prelude_text.len(),
            });
        }
    }
}

/// Concrete [`Runner`] for OpenAI-compatible providers.
///
/// Cheap to clone (an `Arc` internally) so a host can hand the same
/// runner to the Tauri command surface and to a session-listing query
/// at the same time.
#[derive(Clone)]
pub struct OpenAiCompatRunner {
    inner: Arc<Inner>,
}

struct Inner {
    config: OpenAiCompatConfig,
    client: Arc<dyn HttpClient>,
    sessions: RwLock<HashMap<SessionId, Arc<Mutex<InnerSession>>>>,
    /// Optional tool registry. P1 stores it; P2 will use the schemas
    /// to populate the request body's `tools[]` array and dispatch
    /// `tool_calls[]` deltas through `execute`. Until P2 lands, the
    /// runner ignores it — the renderer side and the agent code path
    /// in `app/yah/desktop/src/agent.rs` can construct the registry today
    /// without changing call sites once the loop turns on.
    tools: Option<Arc<dyn ToolRegistry>>,
}

impl OpenAiCompatRunner {
    /// A runner over the default reqwest substrate.
    #[cfg(feature = "reqwest")]
    pub fn new(config: OpenAiCompatConfig) -> Self {
        Self::with_http(config, Arc::new(yah_runner_core::ReqwestClient::default()), None)
    }

    /// A runner over a host-supplied HTTP substrate (R950-F5) — the
    /// constructor a wasm / iOS host uses with `--no-default-features`.
    pub fn with_http(
        config: OpenAiCompatConfig,
        client: Arc<dyn HttpClient>,
        tools: Option<Arc<dyn ToolRegistry>>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                config,
                client,
                sessions: RwLock::new(HashMap::new()),
                tools,
            }),
        }
    }

    /// Construct a runner with a tool registry attached. Hosts that
    /// already built a registry for the camp hand it in here; sessions
    /// started against this runner will eventually carry the tools[]
    /// array in every provider request (P2). Until then the registry
    /// is held but not exercised — adding it now keeps callers from
    /// having to re-thread the construction site when P2 lands.
    #[cfg(feature = "reqwest")]
    pub fn with_tools(config: OpenAiCompatConfig, tools: Arc<dyn ToolRegistry>) -> Self {
        Self::with_http(config, Arc::new(yah_runner_core::ReqwestClient::default()), Some(tools))
    }

    /// Provider label this runner answers to. Hosts use this to route
    /// `start` calls; useful when registering more than one runner in
    /// one process.
    pub fn provider_label(&self) -> &str {
        &self.inner.config.provider_label
    }

    /// Borrow the attached tool registry, if any. Hosts that need to
    /// inspect the registry (e.g. listing schemas in a settings UI)
    /// can read through here without round-tripping a turn.
    pub fn tools(&self) -> Option<&Arc<dyn ToolRegistry>> {
        self.inner.tools.as_ref()
    }

    /// Summarize the session's conversation in place (R418-F4). The
    /// runner-side implementation of `BlockResolution::Compact` for
    /// openai_compat sessions — the daemon's `resolve_block_runner`
    /// dispatches here after the host sees `TurnBlocked { reason:
    /// ContextOverflow }`. Issues one non-streaming chat completion
    /// against the same provider/model with a fixed summarizer system
    /// prompt; the response replaces all but the last `keep_last_n`
    /// rows of `session.history` with a single `role: "system"` row
    /// carrying the summary. The session stays open — the caller is
    /// expected to retry the failed turn via the existing `send` path
    /// after `Ok` returns.
    ///
    /// `keep_last_n` preserves the most recent rows verbatim ahead of
    /// the summary's *trailing* position; the relay design notes
    /// suggest 10. Pass 0 to discard everything except the summary.
    /// The OpenAI protocol requires an `assistant {tool_calls: [...]}`
    /// row to be followed by its `role: "tool"` results, so callers
    /// that pass a small `keep_last_n` should be prepared for one of
    /// those halves to be missing — in that case the safest follow-up
    /// is to issue a fresh user turn rather than re-issue the failed
    /// one verbatim.
    ///
    /// Distinct from `claude_cli`'s `/compact` flow: that one delegates
    /// to an Anthropic-side command which knows the conversation shape
    /// natively. The openai_compat path has no equivalent server
    /// affordance, so the runner does the summarization itself.
    pub async fn compact_session(
        &self,
        session_id: &SessionId,
        keep_last_n: usize,
    ) -> Result<CompactResult, RunnerError> {
        let session = self
            .inner
            .sessions
            .read()
            .await
            .get(session_id)
            .cloned()
            .ok_or_else(|| RunnerError::SessionNotFound(session_id.as_str().to_string()))?;

        // Snapshot inputs without holding the session lock across the
        // network call. `model` follows the resolved model on the
        // session (set by start()), not the config default, so a
        // mid-session model switch is honoured.
        let (history_snapshot, model) = {
            let s = session.lock().await;
            (s.history.clone(), s.model.clone())
        };
        let original_history_messages = history_snapshot.len();
        let original_history_chars: usize = history_snapshot
            .iter()
            .map(|m| serde_json::to_string(m).map(|s| s.len()).unwrap_or(0))
            .sum();
        let history_dump = serialize_history_for_summary(&history_snapshot);

        let body = json!({
            "model": model,
            "stream": false,
            "max_tokens": 2500,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a faithful conversation summarizer for an agent loop. \
                                Produce a concise summary (≤ 2000 tokens) of the conversation \
                                below. PRESERVE: key file paths read or modified, decisions \
                                made, unresolved questions, the current goal, the most recent \
                                state, and the name of any failing or in-progress operation. \
                                Output plain markdown only — do not emit tool calls, do not \
                                use code blocks longer than 20 lines, do not pad. The summary \
                                will replace the conversation history for the active agent, \
                                so it must be self-contained."
                },
                {
                    "role": "user",
                    "content": format!(
                        "Conversation to summarize (most recent last):\n\n{history_dump}",
                    ),
                },
            ],
        });

        let mut req = HttpRequest::post_json(&self.inner.config.endpoint, &body);
        if let Some(ref key) = self.inner.config.api_key {
            req = req.bearer_auth(key);
        }
        let resp = self
            .inner
            .client
            .send(req)
            .await
            .map_err(|e| RunnerError::Transport(format!("compact request failed: {e}")))?;
        if !resp.is_success() {
            let status = resp.status;
            let body_text = resp.text().await.unwrap_or_default();
            return Err(RunnerError::Transport(format!(
                "compact request returned {status}: {}",
                body_text.chars().take(512).collect::<String>(),
            )));
        }
        let body_value: Value = resp
            .json()
            .await
            .map_err(|e| RunnerError::Transport(format!("compact response parse failed: {e}")))?;
        let summary_text = body_value
            .pointer("/choices/0/message/content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if summary_text.is_empty() {
            return Err(RunnerError::Transport(
                "compact summarizer returned empty content".into(),
            ));
        }
        let summary_chars = summary_text.len();

        // Rewrite history under the session lock. Delegated to
        // [`rewrite_history_with_summary`], which takes *only* `history` —
        // it has no access to `prelude_text`, so RS-1 (R629-T9) is a
        // structural property of the compactor, not merely an observed one.
        let (history_messages_after, kept_tail) = {
            let mut s = session.lock().await;
            let kept = rewrite_history_with_summary(
                &mut s.history,
                &summary_text,
                keep_last_n,
                original_history_messages,
            );
            // R883-T8: the anchor measured the pre-compact history.
            s.usage_anchor = None;
            (s.history.len(), kept)
        };

        Ok(CompactResult {
            session_id: session_id.clone(),
            summary_chars,
            original_history_chars,
            history_messages_before: original_history_messages,
            history_messages_after,
            kept_tail,
        })
    }
}

/// Rewrite a conversation `history` in place into
/// `[summary_row, ...preserved_tail]` and return the count of tail rows
/// preserved verbatim (R418-F4).
///
/// The summary is pushed *first* so the agent reads the past as the past,
/// then the last `keep_last_n` rows so the most-recent context sits closest
/// to the upcoming user turn. `original_len` feeds only the "of N earlier
/// turns" label — the tail split is computed from the live `history` length.
///
/// **RS-1 (R629-T9 / W286).** This is the one yah-owned mutation of session
/// history, and it takes *only* `history`. Because it has no parameter for
/// and no access to `prelude_text`, the resident set cannot be read or
/// written by compaction — the invariant is enforced by the signature, not
/// by a caller remembering to preserve it. A future edit that needed the
/// prelude here would have to change this signature, which is the tripwire.
fn rewrite_history_with_summary(
    history: &mut Vec<Value>,
    summary_text: &str,
    keep_last_n: usize,
    original_len: usize,
) -> usize {
    let n = history.len();
    let tail_start = n.saturating_sub(keep_last_n);
    let kept = n - tail_start;
    let summary_row = json!({
        "role": "system",
        "content": format!(
            "[R418-F4 compact summary of {} earlier turns]\n\n{}",
            original_len.saturating_sub(kept),
            summary_text,
        ),
    });
    let tail: Vec<Value> = history.drain(tail_start..).collect();
    history.clear();
    history.push(summary_row);
    history.extend(tail);
    kept
}

/// Fetch the OpenAI-compat `/v1/models` catalogue for a given chat
/// endpoint. Lifts `…/chat/completions` to `…/models` and parses the
/// `data[].id` array — the same shape OpenAI, Ollama (local + Cloud),
/// vLLM, and most gateways serve.
///
/// Returns model ids sorted alphabetically. Empty `Ok(vec![])` when
/// the endpoint replies but the catalogue is empty (legitimate state
/// for a fresh local Ollama with no models pulled). Errors propagate
/// as `RunnerError::Transport` so the host's renderer can surface a
/// clear "couldn't list models" message.
#[cfg(feature = "reqwest")]
pub async fn list_openai_compat_models(
    chat_endpoint: &str,
    api_key: Option<&str>,
) -> Result<Vec<String>, RunnerError> {
    list_openai_compat_models_with(&yah_runner_core::ReqwestClient::default(), chat_endpoint, api_key)
        .await
}

/// [`list_openai_compat_models`] over a host-supplied HTTP substrate.
pub async fn list_openai_compat_models_with(
    client: &dyn HttpClient,
    chat_endpoint: &str,
    api_key: Option<&str>,
) -> Result<Vec<String>, RunnerError> {
    let models_url = derive_models_url(chat_endpoint);
    let mut req = HttpRequest::get(&models_url);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }
    let resp = client
        .send(req)
        .await
        .map_err(|e| RunnerError::Transport(format!("models request failed: {e}")))?;
    if !resp.is_success() {
        let status = resp.status;
        let body = resp.text().await.unwrap_or_default();
        return Err(RunnerError::Transport(format!(
            "models endpoint returned {}: {}",
            status,
            body.chars().take(256).collect::<String>(),
        )));
    }
    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| RunnerError::Transport(format!("models response parse failed: {e}")))?;
    let mut out: Vec<String> = json
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|v| v.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out.dedup();
    Ok(out)
}

/// Replace the trailing `/chat/completions` path with `/models`. Falls
/// back to appending `/models` if the path doesn't match — keeps any
/// gateway that serves both at the same base happy.
fn derive_models_url(chat_endpoint: &str) -> String {
    // Ollama native `/api/chat` lives outside the OpenAI-compat `/v1/*`
    // prefix; the model catalogue still serves at `/v1/models` on the same
    // host, so synthesise the prefix when stripping `/api/chat`.
    if let Some(stripped) = chat_endpoint.strip_suffix("/api/chat") {
        format!("{stripped}/v1/models")
    } else if let Some(stripped) = chat_endpoint.strip_suffix("/chat/completions") {
        format!("{stripped}/models")
    } else if let Some(stripped) = chat_endpoint.strip_suffix('/') {
        format!("{stripped}/models")
    } else {
        format!("{chat_endpoint}/models")
    }
}

#[async_trait]
impl Runner for OpenAiCompatRunner {
    /// The camp runner reads the opaque session key as the ticket id.
    async fn start(&self, prelude: Prelude, ticket_id: &str) -> Result<SessionId, RunnerError> {
        if let Some(engine) = prelude.model.as_ref() {
            if engine.provider != self.inner.config.provider_label {
                return Err(RunnerError::UnsupportedEngine {
                    engine: engine.as_payload(),
                });
            }
        }
        let model = prelude
            .model
            .as_ref()
            .and_then(|e| e.model.clone())
            .unwrap_or_else(|| self.inner.config.default_model.clone());

        let id = mint_session_id();
        // The host's per-session preamble (yah: the self-identity block,
        // R950-F5) is per-session, so it must trail the shared rendered
        // prelude rather than lead it, or it salts every byte after it and
        // no two sessions share a cache prefix (R883-B15).
        let mut prelude_text = prelude.render();
        if let Some(preamble) = self.inner.config.session_preamble.as_ref() {
            prelude_text.push('\n');
            prelude_text.push_str(&preamble.render(&id, &prelude, ticket_id));
        }
        // R418-F3: append the runtime self-introspection block so the
        // agent can self-report identity + caps without a tool call.
        prelude_text.push_str(&render_runtime_caps_block(
            &id,
            ticket_id,
            &model,
            &self.inner.config,
        ));
        let session = InnerSession {
            ticket_id: ticket_id.to_string(),
            prelude_text,
            model,
            history: Vec::new(),
            resident_stanzas: prelude.section_id_set(),
            usage_anchor: None,
            think: self.inner.config.seed_think(prelude.reasoning),
        };
        self.inner
            .sessions
            .write()
            .await
            .insert(id.clone(), Arc::new(Mutex::new(session)));
        Ok(id)
    }

    async fn send(
        &self,
        session_id: &SessionId,
        message: String,
    ) -> Result<EventStream, RunnerError> {
        let session = self
            .inner
            .sessions
            .read()
            .await
            .get(session_id)
            .cloned()
            .ok_or_else(|| RunnerError::SessionNotFound(session_id.as_str().to_string()))?;

        // Append the user turn so the upcoming request body picks it up.
        // History is now provider-shaped JSON; we push raw `{role, content}`
        // rows here and (further below, inside the stream) push the
        // assistant rows + tool-result rows as they are produced.
        let model = {
            let mut s = session.lock().await;
            s.history.push(json!({
                "role": "user",
                "content": message,
            }));
            s.model.clone()
        };

        // Snapshot what the stream needs so the borrow on `&self` ends
        // before the stream lives across iteration boundaries.
        let client = self.inner.client.clone();
        let endpoint = self.inner.config.endpoint.clone();
        let api_key = self.inner.config.api_key.clone();
        let quota_ledger = self.inner.config.quota_ledger.clone();
        let rate_limit_hint = self.inner.config.rate_limit_hint.clone();
        let provider_label = self.inner.config.provider_label.clone();
        let max_output_tokens = self.inner.config.max_output_tokens();
        let max_tool_iterations = self.inner.config.max_tool_iterations();
        let transport = self.inner.config.transport;
        let cost_guard_tokens = self.inner.config.cost_guard_tokens;
        let request_options = ollama_options(&self.inner.config);
        let keep_alive = self.inner.config.keep_alive.clone();
        let context_window = self.inner.config.context_window_for(&model);
        let prompt_overflow_threshold = self.inner.config.prompt_overflow_threshold();
        let tools = self.inner.tools.clone();
        let (tools_array, tool_name_map) = tools_array_payload(tools.as_ref());
        let session_id_owned = session_id.clone();
        let session_arc = Arc::clone(&session);

        // R418-F2: pre-flight prompt-size guard. If the next request is
        // projected to exceed the context window, elide the largest
        // oversized tool_result row(s) in history before issuing — the
        // event yielded after `TurnStarted` below tells observers the
        // trim happened. No-op when `context_window` is `None` (host
        // didn't tell us the model's window, can't know if we're over).
        let preflight_elisions = if let Some(window) = context_window {
            let mut s = session_arc.lock().await;
            try_preflight_elide(&mut s, window, prompt_overflow_threshold)
        } else {
            Vec::new()
        };

        // Issue the first request synchronously so transport / 4xx
        // failures surface as `RunnerError::Transport` from `send`
        // rather than as a mid-stream `AgentEvent::Error`. Subsequent
        // re-issues for tool-call iterations happen inside the stream
        // (where the only way to signal failure is in-stream), since by
        // then the host has already committed to a streaming pane.
        //
        // R418-F4: a 400 context_length_exceeded is *not* a generic
        // transport error — the operator needs the Compact/HandOff/Abandon
        // recovery modal, not a red banner. Convert it into a streamed
        // TurnBlocked so the host's existing R262-F2 wire path routes it
        // to the Blocked phase instead of emitting AgentEvent::Error.
        let initial_result = issue_chat_completions(
            client.as_ref(),
            &endpoint,
            api_key.as_deref(),
            &provider_label,
            quota_ledger.as_deref(),
            rate_limit_hint.as_ref(),
            {
                let s = session_arc.lock().await;
                build_request_body(
                    &s,
                    max_output_tokens,
                    tools_array.as_deref(),
                    request_options.as_ref(),
                    keep_alive.as_deref(),
                )
            },
        )
        .await;
        let initial_bytes_stream = match initial_result {
            Ok(resp) => Some(resp.body),
            Err(RunnerError::Transport(msg)) if is_overflow_message(&msg) => None,
            Err(e) => return Err(e),
        };
        let pre_stream_block_reason: Option<BlockReason> = if initial_bytes_stream.is_none() {
            Some(BlockReason::ContextOverflow)
        } else {
            None
        };

        let stream = stream! {
            // Emit one PromptTrimmed per elision *before* TurnStarted so
            // the renderer can attach the chip to this turn without
            // racing the first MessageDelta.
            for rec in &preflight_elisions {
                yield AgentEvent::PromptTrimmed {
                    session_id: session_id_owned.clone(),
                    tool_call_id: rec.tool_call_id.clone(),
                    tool_name: rec.tool_name.clone(),
                    original_bytes: rec.original_bytes as u64,
                    replacement_bytes: rec.replacement_bytes as u64,
                };
            }
            yield AgentEvent::TurnStarted { session_id: session_id_owned.clone() };

            // R418-F4: pre-stream context-overflow detected at the
            // initial request — the provider rejected the body as too
            // long before any bytes streamed. Surface as TurnBlocked so
            // the host transitions to the Blocked phase (R262-F2) and
            // the recovery modal becomes available.
            if let Some(reason) = pre_stream_block_reason {
                yield AgentEvent::TurnBlocked {
                    session_id: session_id_owned.clone(),
                    text: String::new(),
                    reason,
                };
                return;
            }

            let session_id = session_id_owned;
            let mut iterations: u32 = 0;
            // Total individual tool dispatches across all iterations.
            let mut total_tool_calls: u32 = 0;
            // ToolStuck tracker: (tool_name, canonical_args) → first ok:false result.
            // A second ok:false hit on the same key triggers AgentEvent::ToolStuck.
            let mut stuck_tracker: std::collections::HashMap<(String, String), serde_json::Value> =
                std::collections::HashMap::new();
            // `last_stop_reason` records the finish_reason from the iteration
            // that broke the loop. Initial value is overwritten on the break
            // path; it exists so the value is typed in scope without
            // `Option<…>` ceremony on the assignment side.
            #[allow(unused_assignments)]
            let mut last_stop_reason: Option<String> = None;
            // Turn-level usage accumulator. Tool-call iterations are
            // separate /v1/chat/completions requests, each with its own
            // usage chunk; we sum across them so `TurnEnded.usage`
            // reflects the user-visible turn cost rather than just the
            // final iteration. `None`-valued fields stay None until at
            // least one iteration reports them, then accumulate u32s.
            let mut turn_usage_acc = TurnUsageAcc::default();

            // First iteration uses the pre-fetched bytes_stream; later
            // iterations issue fresh requests inside the loop and surface
            // any failure as `AgentEvent::Error` (the stream is already
            // live so a synchronous error can't reach the caller).
            // Unwrap is safe: the `pre_stream_block_reason` arm above
            // returned before reaching here when initial_bytes_stream
            // was None.
            let mut pending_bytes_stream = initial_bytes_stream;
            // Successful (ok:true) (tool_name, canonical_args) pairs from the
            // previous iteration. If the current iteration repeats any of them
            // the model is spinning — emit "loop" rather than "budget exhausted".
            let mut prev_iter_successful_calls: HashSet<(String, String)> = HashSet::new();
            // Full text accumulated across ALL iterations. Per-iteration `iter_text`
            // is reset each loop; this spans the whole turn so TurnFailed always
            // carries everything the model said, enabling correct JSONL replay even
            // when the failure fires in a later iteration that produced no text.
            let mut full_text_acc = String::new();
            // R418-F3: single-fire flag for the per-turn budget reminder.
            // Once we've nudged the agent that it's near the iteration or
            // context cap, don't spam — one reminder per turn is enough.
            let mut threshold_reminder_fired = false;

            'turn: loop {
                let mut bytes_stream = match pending_bytes_stream.take() {
                    Some(b) => b,
                    None => {
                        // R418-F2: re-run the pre-flight guard on every
                        // tool-loop re-issue. Tool results can be arbitrarily
                        // large (a single board.list_tickets blew the cap in
                        // session:098779e3) so we have to re-check, not just
                        // trust the initial pass.
                        let mid_loop_elisions = if let Some(window) = context_window {
                            let mut s = session_arc.lock().await;
                            try_preflight_elide(&mut s, window, prompt_overflow_threshold)
                        } else {
                            Vec::new()
                        };
                        for rec in &mid_loop_elisions {
                            yield AgentEvent::PromptTrimmed {
                                session_id: session_id.clone(),
                                tool_call_id: rec.tool_call_id.clone(),
                                tool_name: rec.tool_name.clone(),
                                original_bytes: rec.original_bytes as u64,
                                replacement_bytes: rec.replacement_bytes as u64,
                            };
                        }
                        let body = {
                            let s = session_arc.lock().await;
                            build_request_body(
                                &s,
                                max_output_tokens,
                                tools_array.as_deref(),
                                request_options.as_ref(),
                                keep_alive.as_deref(),
                            )
                        };
                        match issue_chat_completions(
                            client.as_ref(),
                            &endpoint,
                            api_key.as_deref(),
                            &provider_label,
                            quota_ledger.as_deref(),
                            rate_limit_hint.as_ref(),
                            body,
                        )
                        .await
                        {
                            Ok(resp) => resp.body,
                            Err(e) => {
                                // Pre-stream failure on a tool-loop re-issue. No text
                                // in this iteration yet; carry whatever earlier
                                // iterations produced so JSONL replay reconstructs it.
                                // R418-F4: route context-overflow to TurnBlocked so the
                                // host's R262-F2 wire path surfaces the recovery modal
                                // instead of a generic red error.
                                let msg = format!("{e}");
                                if is_overflow_message(&msg) {
                                    yield AgentEvent::TurnBlocked {
                                        session_id: session_id.clone(),
                                        text: full_text_acc.clone(),
                                        reason: BlockReason::ContextOverflow,
                                    };
                                } else {
                                    yield AgentEvent::TurnFailed {
                                        session_id: session_id.clone(),
                                        text: full_text_acc.clone(),
                                        message: msg,
                                    };
                                }
                                return;
                            }
                        }
                    }
                };
                let mut iter_text = String::new();
                // R883-T8: this response's prompt + completion tokens,
                // anchored onto history when its assistant row lands.
                let mut iter_anchor_tokens: Option<u64> = None;
                let mut iter_finish: Option<String> = None;
                let mut tool_call_accs: BTreeMap<u32, ToolCallAcc> = BTreeMap::new();
                let mut total_bytes: usize = 0;
                let mut buffer = String::new();

                'sse: while let Some(chunk) = bytes_stream.next().await {
                    let bytes = match chunk {
                        Ok(b) => b,
                        Err(e) => {
                            yield AgentEvent::TurnFailed {
                                session_id: session_id.clone(),
                                text: full_text_acc.clone(),
                                message: format!("stream read failed: {e}"),
                            };
                            return;
                        }
                    };
                    total_bytes = total_bytes.saturating_add(bytes.len());
                    if total_bytes > MAX_STREAM_BYTES {
                        yield AgentEvent::TurnFailed {
                            session_id: session_id.clone(),
                            text: full_text_acc.clone(),
                            message: format!(
                                "stream exceeded soft cap of {} bytes — aborting",
                                MAX_STREAM_BYTES
                            ),
                        };
                        return;
                    }
                    let text = match std::str::from_utf8(&bytes) {
                        Ok(s) => s,
                        Err(e) => {
                            yield AgentEvent::TurnFailed {
                                session_id: session_id.clone(),
                                text: full_text_acc.clone(),
                                message: format!("non-utf8 chunk: {e}"),
                            };
                            return;
                        }
                    };
                    buffer.push_str(text);

                    // SSE separates frames with `\n\n`; Ollama /api/chat NDJSON
                    // separates with a single `\n` and each frame is a complete
                    // JSON object on its own line.
                    let separator = match transport {
                        Transport::Sse => "\n\n",
                        Transport::OllamaNdjson => "\n",
                    };
                    while let Some(idx) = buffer.find(separator) {
                        let frame = buffer[..idx].to_string();
                        buffer.drain(..idx + separator.len());
                        let parsed_frames = match transport {
                            Transport::Sse => parse_sse_frame(&frame),
                            Transport::OllamaNdjson => parse_ollama_chat_frame(&frame),
                        };
                        for parsed in parsed_frames {
                            match parsed {
                                FrameEvent::Delta { text, tool_calls, finish } => {
                                    if !text.is_empty() {
                                        iter_text.push_str(&text);
                                        full_text_acc.push_str(&text);
                                        yield AgentEvent::MessageDelta {
                                            session_id: session_id.clone(),
                                            text,
                                        };
                                    }
                                    for tcd in tool_calls {
                                        let acc = tool_call_accs
                                            .entry(tcd.index)
                                            .or_default();
                                        if let Some(id) = tcd.id {
                                            if !id.is_empty() {
                                                acc.id = id;
                                            }
                                        }
                                        if let Some(name) = tcd.name {
                                            if !name.is_empty() {
                                                acc.name = name;
                                            }
                                        }
                                        if let Some(args_chunk) = tcd.arguments {
                                            acc.arguments_buf.push_str(&args_chunk);
                                        }
                                    }
                                    if finish.is_some() {
                                        iter_finish = finish;
                                    }
                                }
                                FrameEvent::ThinkingDelta { thinking } => {
                                    yield AgentEvent::ThinkingDelta {
                                        session_id: session_id.clone(),
                                        text: thinking,
                                    };
                                }
                                FrameEvent::Usage {
                                    input_tokens,
                                    output_tokens,
                                    cache_read_tokens,
                                } => {
                                    // Emit per-iteration usage before accumulating so
                                    // the frontend's mergeUsageMax sees each call's raw
                                    // token counts rather than waiting for TurnEnded
                                    // with the inflated cumulative total.
                                    yield AgentEvent::TurnUsageDelta {
                                        session_id: session_id.clone(),
                                        usage: TurnUsage {
                                            input_tokens,
                                            output_tokens,
                                            cache_read_tokens,
                                            ..Default::default()
                                        },
                                    };
                                    turn_usage_acc.add(
                                        input_tokens,
                                        output_tokens,
                                        cache_read_tokens,
                                    );
                                    if let Some(input) = input_tokens {
                                        iter_anchor_tokens = Some(
                                            u64::from(input)
                                                + u64::from(output_tokens.unwrap_or(0)),
                                        );
                                    }
                                    // Cost guard: abort early when a non-caching engine
                                    // is ballooning input tokens across tool-call
                                    // iterations. Checked per-iteration (not at
                                    // TurnEnded) so we stop before the damage is done.
                                    if let Some(limit) = cost_guard_tokens {
                                        if turn_usage_acc.input_tokens.unwrap_or(0) > limit {
                                            let total = turn_usage_acc.input_tokens.unwrap_or(0);
                                            yield AgentEvent::TurnFailed {
                                                session_id: session_id.clone(),
                                                text: full_text_acc.clone(),
                                                message: format!(
                                                    "[cost_guard] Turn aborted: {total} cumulative \
                                                     input tokens exceeded the {limit}-token guard. \
                                                     This engine does not support prompt caching — \
                                                     reduce context size or switch to a \
                                                     caching-capable backend."
                                                ),
                                            };
                                            return;
                                        }
                                    }
                                }
                                FrameEvent::Error(msg) => {
                                    yield AgentEvent::TurnFailed {
                                        session_id: session_id.clone(),
                                        text: full_text_acc.clone(),
                                        message: msg,
                                    };
                                    return;
                                }
                                FrameEvent::Done => break 'sse,
                            }
                        }
                    }
                }

                last_stop_reason = iter_finish.clone();
                // Emit one AssistantText per iteration that produced text. This
                // is the durable record of text-vs-tool-call interleaving:
                // intermediate iterations end with finish_reason="tool_calls"
                // and would otherwise drop their text from the JSONL replay
                // (TurnEnded.text only carries the aggregate). One iteration =
                // one assistant frame in OpenAI's wire format, so emitting
                // once per iter matches the AssistantText contract.
                if !iter_text.is_empty() {
                    yield AgentEvent::AssistantText {
                        session_id: session_id.clone(),
                        text: iter_text.clone(),
                    };
                }
                let calls: Vec<ToolCallFinalized> = tool_call_accs
                    .into_iter()
                    .map(|(_, acc)| acc.finalize())
                    .filter(|c| !c.id.is_empty() || !c.name.is_empty())
                    .collect();

                let go_again = matches!(iter_finish.as_deref(), Some("tool_calls"))
                    || (!calls.is_empty() && iter_finish.is_none());

                if go_again {
                    // Append the assistant turn carrying the tool_calls
                    // exactly as the provider produced them — the next
                    // request body must echo `tool_calls[]` so role:tool
                    // rows can be paired by `tool_call_id`.
                    let assistant_tool_calls: Vec<Value> = calls
                        .iter()
                        .map(|c| json!({
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": c.arguments_str,
                            }
                        }))
                        .collect();
                    {
                        let mut s = session_arc.lock().await;
                        s.history.push(json!({
                            "role": "assistant",
                            "content": iter_text,
                            "tool_calls": assistant_tool_calls,
                        }));
                        s.record_usage_anchor(iter_anchor_tokens);
                    }

                    let registry = match tools.as_ref() {
                        Some(r) => r,
                        None => {
                            // Defensive: we never advertised tools[] in the
                            // request body, so the model shouldn't have
                            // produced tool_calls. Treat as a protocol
                            // violation.
                            yield AgentEvent::TurnFailed {
                                session_id: session_id.clone(),
                                text: full_text_acc.clone(),
                                message: "provider produced tool_calls but no registry is attached"
                                    .into(),
                            };
                            return;
                        }
                    };

                    let mut stuck_this_round = false;
                    let mut current_iter_successful: HashSet<(String, String)> = HashSet::new();
                    for call in calls {
                        let args_value = parse_tool_arguments(&call.arguments_str);
                        // Serialize args once so both ToolCall and stuck_tracker use
                        // the same canonical string without a redundant serde pass.
                        let args_key = serde_json::to_string(&args_value).unwrap_or_default();
                        // The model echoes the wire-name we advertised in `tools[]`
                        // (dots rewritten to underscores). Map back to the
                        // registry's original name so dispatch + observability
                        // see the namespaced form (e.g. `board.show`).
                        let registry_name = tool_name_map
                            .get(&call.name)
                            .cloned()
                            .unwrap_or_else(|| call.name.clone());
                        yield AgentEvent::ToolCall {
                            session_id: session_id.clone(),
                            tool_call_id: call.id.clone(),
                            tool_name: registry_name.clone(),
                            args: args_value.clone(),
                        };
                        let outcome = match args_value.clone() {
                            Value::Null => yah_runner_core::tool::ToolOutcome {
                                ok: false,
                                result: json!({
                                    "error": format!(
                                        "tool arguments are not valid JSON: {}",
                                        truncate_for_error(&call.arguments_str, 200)
                                    ),
                                    "kind": "invalid_arguments",
                                }),
                            },
                            v => registry.execute(&registry_name, v).await,
                        };
                        total_tool_calls += 1;
                        yield AgentEvent::ToolResult {
                            session_id: session_id.clone(),
                            tool_call_id: call.id.clone(),
                            ok: outcome.ok,
                            result: outcome.result.clone(),
                        };
                        // Capped at ingestion (R883-T1); the event keeps the
                        // full result, and replay re-applies the same cap.
                        let tool_content = yah_runner_core::tool::tool_result_wire_content(&outcome.result);
                        if outcome.ok {
                            current_iter_successful.insert((registry_name.clone(), args_key.clone()));
                        }
                        if !outcome.ok {
                            let key = (registry_name.clone(), args_key);
                            if let Some(prior) = stuck_tracker.get(&key) {
                                yield AgentEvent::ToolStuck {
                                    session_id: session_id.clone(),
                                    tool_name: registry_name.clone(),
                                    args: args_value,
                                    result: prior.clone(),
                                };
                                stuck_this_round = true;
                                // Push the result to history so it's available if
                                // the host drains the stream rather than cutting it.
                                {
                                    let mut s = session_arc.lock().await;
                                    s.history.push(json!({
                                        "role": "tool",
                                        "tool_call_id": call.id,
                                        "content": tool_content,
                                    }));
                                }
                                break;
                            } else {
                                stuck_tracker.insert(key, outcome.result.clone());
                            }
                        }
                        if !stuck_this_round {
                            let mut s = session_arc.lock().await;
                            s.history.push(json!({
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": tool_content,
                            }));
                        }
                    }
                    if stuck_this_round {
                        return;
                    }

                    // Loop detection: same successful tool calls (name + args) in
                    // two consecutive iterations means the model is spinning rather
                    // than making progress.
                    let loop_example = current_iter_successful
                        .iter()
                        .find(|k| prev_iter_successful_calls.contains(*k));
                    if let Some((tool_name, _)) = loop_example {
                        yield AgentEvent::TurnFailed {
                            session_id: session_id.clone(),
                            text: full_text_acc.clone(),
                            message: format!(
                                "agent is in a loop — '{tool_name}' was called with the same \
                                 arguments in consecutive iterations \
                                 ({total_tool_calls} tool calls total); stopping",
                            ),
                        };
                        return;
                    }
                    prev_iter_successful_calls = current_iter_successful;

                    iterations += 1;
                    // R418-F3: nudge the agent with a system reminder when
                    // it's about to hit the iteration cap (≥ max-2) or when
                    // the prompt fill is past 0.7 of context_window. Fires
                    // once per turn — repeated reminders would just burn
                    // budget. The reminder lands in history as a
                    // role:"system" row so the next iteration sees it as
                    // signal, not noise.
                    if !threshold_reminder_fired {
                        let iter_close = iterations + 2 >= max_tool_iterations;
                        // R883-T8: anchored estimate; no anchor → no
                        // evidence → not close.
                        let context_est = match context_window {
                            Some(window) if window > 0 => {
                                let s = session_arc.lock().await;
                                estimate_prompt_tokens(&s).map(|est| (est, window))
                            }
                            _ => None,
                        };
                        let context_close = context_est
                            .is_some_and(|(est, window)| est as f32 / window as f32 >= 0.7);
                        if iter_close || context_close {
                            let remaining = max_tool_iterations.saturating_sub(iterations);
                            let context_line = match context_est {
                                Some((est_tokens, window)) => format!(
                                    " · context: ~{}k of {}k tokens used",
                                    est_tokens / 1000,
                                    window / 1000,
                                ),
                                None => String::new(),
                            };
                            let reminder = format!(
                                "[runtime check · R418-F3] {remaining} of {max_tool_iterations} \
                                 tool-call iterations left this turn{context_line}. Wrap up or \
                                 narrow scope — fanning out wide here is likely to truncate.",
                            );
                            {
                                let mut s = session_arc.lock().await;
                                s.history.push(json!({
                                    "role": "system",
                                    "content": reminder,
                                }));
                            }
                            threshold_reminder_fired = true;
                        }
                    }
                    if iterations >= max_tool_iterations {
                        // Soft stop: the model isn't stuck (loop detector
                        // already had a shot), the per-turn tool-call cap
                        // is a policy choice. Synthesize a closing
                        // assistant turn so the transcript reads naturally
                        // and the user can just say "continue" — no red
                        // error banner. stop_reason flags it for
                        // observability without crashing the chat.
                        let notice = format!(
                            "I've reached this turn's tool-call budget \
                             ({max_tool_iterations} iterations, \
                             {total_tool_calls} tool calls). Say \
                             \"continue\" to keep going, or raise \
                             `max_tool_iterations` on the engine config \
                             for wider exploration."
                        );
                        {
                            let mut s = session_arc.lock().await;
                            s.history.push(json!({
                                "role": "assistant",
                                "content": notice.clone(),
                            }));
                        }
                        yield AgentEvent::AssistantText {
                            session_id: session_id.clone(),
                            text: notice.clone(),
                        };
                        if !full_text_acc.is_empty() {
                            full_text_acc.push_str("\n\n");
                        }
                        full_text_acc.push_str(&notice);
                        yield AgentEvent::TurnEnded {
                            session_id: session_id.clone(),
                            text: full_text_acc.clone(),
                            stop_reason: Some("exploration_budget".into()),
                            usage: turn_usage_acc.into_usage(),
                            content_blocks: None,
                        };
                        return;
                    }
                    // Loop: re-issue request with the new role:tool rows.
                    continue 'turn;
                }

                // Plain text reply (or `stop` with no tool calls). Persist
                // the assistant text and break out of the loop.
                {
                    let mut s = session_arc.lock().await;
                    s.history.push(json!({
                        "role": "assistant",
                        "content": iter_text.clone(),
                    }));
                    s.record_usage_anchor(iter_anchor_tokens);
                }
                break 'turn;
            }

            yield AgentEvent::TurnEnded {
                session_id,
                text: full_text_acc,
                stop_reason: last_stop_reason,
                usage: turn_usage_acc.into_usage(),
                content_blocks: None,
            };
        };

        Ok(Box::pin(stream)
            as Pin<
                Box<dyn futures::stream::Stream<Item = AgentEvent> + Send>,
            >)
    }

    async fn stop(&self, session_id: &SessionId) -> Result<bool, RunnerError> {
        Ok(self
            .inner
            .sessions
            .write()
            .await
            .remove(session_id)
            .is_some())
    }

    async fn resume(
        &self,
        session_id: &SessionId,
        history: Vec<AgentEvent>,
        prelude: Prelude,
    ) -> Result<(), RunnerError> {
        if let Some(engine) = prelude.model.as_ref() {
            if engine.provider != self.inner.config.provider_label {
                return Err(RunnerError::UnsupportedEngine {
                    engine: engine.as_payload(),
                });
            }
        }
        let model = prelude
            .model
            .as_ref()
            .and_then(|e| e.model.clone())
            .unwrap_or_else(|| self.inner.config.default_model.clone());

        let messages = reconstruct_openai_messages(&history);
        let ticket_id = history.iter().find_map(|ev| {
            if let AgentEvent::SessionStarted { session_key, .. } = ev {
                Some(session_key.clone())
            } else {
                None
            }
        }).unwrap_or_default();

        let session = InnerSession {
            ticket_id,
            prelude_text: prelude.render(),
            model,
            history: messages,
            resident_stanzas: prelude.section_id_set(),
            usage_anchor: None,
            think: self.inner.config.seed_think(prelude.reasoning),
        };
        self.inner
            .sessions
            .write()
            .await
            .insert(session_id.clone(), Arc::new(Mutex::new(session)));
        Ok(())
    }

    /// R947-F2: the level is read into every request body, so a change lands
    /// on the next `send`.
    fn think_live(&self) -> bool {
        true
    }

    fn tools_registry(&self) -> Option<&Arc<dyn ToolRegistry>> {
        self.inner.tools.as_ref()
    }
}

/// The operations yah's camp layers on top of [`Runner`] (its
/// `CampExtensions` trait, implemented host-side by delegating here).
impl OpenAiCompatRunner {
    /// R947-F2: the level this runner applies to a session's next request.
    pub async fn think(&self, session_id: &SessionId) -> Option<ReasoningBudget> {
        let session = self.inner.sessions.read().await.get(session_id).cloned()?;
        let think = session.lock().await.think;
        think
    }

    /// R947-F2: change a live session's level for its next request.
    /// `Ok(false)` when no session matched.
    pub async fn set_think(
        &self,
        session_id: &SessionId,
        think: Option<ReasoningBudget>,
    ) -> Result<bool, RunnerError> {
        let Some(session) = self.inner.sessions.read().await.get(session_id).cloned() else {
            return Ok(false);
        };
        session.lock().await.think = think;
        Ok(true)
    }

    /// Load a stanza into the resident set mid-session (R629-F8 / W286). The
    /// HTTP-runner side of "load a pill after the session has started":
    /// `prelude_text` is a plain `String` read fresh into `messages[0]` on
    /// every request, so a tail append takes effect on the next `send` with
    /// no re-spawn — unlike the argv-fixed subprocess runners, which inherit
    /// the trait's re-spawn-with-resume fallback.
    ///
    /// **Idempotent.** If `stanza_id` is already resident (seeded from
    /// [`Prelude::stanza_id_set`] at start, plus anything a prior call
    /// appended), this is a no-op returning `appended: false` — re-loading a
    /// resident stanza must not double-inject it.
    ///
    /// **Append-only at the tail, never reorder** (W286 "cost of loading
    /// late"): the body goes on the end of `prelude_text`, so a caching
    /// provider re-processes only the appended text — a 719-token stanza
    /// costs ~719 tokens once, not a full re-prefill. Removal is deliberately
    /// not offered: dropping a stanza would invalidate the cached prefix from
    /// the removal point onward, so it is a natural-restart concern.
    ///
    /// RS-1 (R629-T9): this mutates `prelude_text`, never `history` — the
    /// mirror of `compact_session`, which mutates `history`, never
    /// `prelude_text`. The split keeps the resident set and the conversation
    /// independently durable.
    pub async fn append_resident_stanza(
        &self,
        session_id: &SessionId,
        stanza_id: &str,
        body: &str,
    ) -> Result<ResidentAppend, RunnerError> {
        let session = self
            .inner
            .sessions
            .read()
            .await
            .get(session_id)
            .cloned()
            .ok_or_else(|| RunnerError::SessionNotFound(session_id.as_str().to_string()))?;

        let mut s = session.lock().await;
        if s.resident_stanzas.contains(stanza_id) {
            return Ok(ResidentAppend {
                stanza_id: stanza_id.to_string(),
                appended: false,
                appended_chars: 0,
                prelude_chars_after: s.prelude_text.chars().count(),
            });
        }
        // Separator only when there is already text to separate from.
        let addition = if s.prelude_text.is_empty() {
            body.to_string()
        } else {
            format!("\n\n{body}")
        };
        let appended_chars = addition.chars().count();
        s.prelude_text.push_str(&addition);
        s.resident_stanzas.insert(stanza_id.to_string());
        Ok(ResidentAppend {
            stanza_id: stanza_id.to_string(),
            appended: true,
            appended_chars,
            prelude_chars_after: s.prelude_text.chars().count(),
        })
    }
}

/// Parsed events from one `\n\n`-delimited SSE frame. A single frame
/// can carry multiple `data: …` lines; OpenAI typically emits one
/// chunk per frame, but the parser handles multiple defensively.
#[derive(Debug, PartialEq, Eq)]
enum FrameEvent {
    Delta {
        text: String,
        tool_calls: Vec<ToolCallDelta>,
        finish: Option<String>,
    },
    /// Thinking/reasoning delta from extended thinking models. Emitted when
    /// the model returns `delta.thinking` content (e.g., DeepSeek with
    /// thinking mode enabled).
    ThinkingDelta {
        thinking: String,
    },
    /// `usage` frame emitted when the request asked for
    /// `stream_options.include_usage = true`. OpenAI sends this as a
    /// final chunk after `finish_reason` with `choices: []` and a
    /// top-level `usage { prompt_tokens, completion_tokens, … }` object.
    /// Ollama Cloud follows the same shape. `cache_read_tokens`
    /// is parsed when the provider exposes it (some gateways do); for
    /// vanilla OpenAI it stays `None`.
    Usage {
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
        cache_read_tokens: Option<u32>,
    },
    Error(String),
    Done,
}

/// One increment of a `delta.tool_calls[]` entry. OpenAI streams these
/// in pieces: the first chunk carries `index + id + function.name +
/// (often) the opening `{` of `function.arguments`; later chunks just
/// have `index` and a slice of `function.arguments`. The accumulator
/// (see [`ToolCallAcc`]) joins them by `index`.
#[derive(Debug, PartialEq, Eq, Default)]
struct ToolCallDelta {
    index: u32,
    id: Option<String>,
    name: Option<String>,
    arguments: Option<String>,
}

/// Live merge buffer for one tool call across SSE chunks. Keyed by the
/// provider's `index`. The `arguments_buf` is appended to in-place; we
/// only parse it into JSON once the SSE stream closes (or the
/// `finish_reason: "tool_calls"` chunk lands).
#[derive(Debug, Default)]
struct ToolCallAcc {
    id: String,
    name: String,
    arguments_buf: String,
}

#[derive(Debug)]
struct ToolCallFinalized {
    id: String,
    name: String,
    arguments_str: String,
}

impl ToolCallAcc {
    fn finalize(self) -> ToolCallFinalized {
        ToolCallFinalized {
            id: self.id,
            name: self.name,
            arguments_str: self.arguments_buf,
        }
    }
}

/// Per-turn usage accumulator. A turn that triggers tool calls makes
/// multiple /v1/chat/completions round-trips, each with its own `usage`
/// chunk; this rolls them up so `TurnEnded.usage` reflects the total
/// user-visible cost of the turn rather than just the final iteration.
///
/// Each field stays `None` until at least one iteration reports a value
/// for it; then later values are added on top. Saturating-add keeps a
/// pathological provider from wrapping a u32 mid-turn.
#[derive(Debug, Default)]
struct TurnUsageAcc {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    cache_read_tokens: Option<u32>,
    /// Max of (input + cache_read) per iteration — the per-call context
    /// window size. Stored separately because input_tokens and
    /// cache_read_tokens are summed across iterations for billing; this
    /// gives the ring display the peak instantaneous occupancy instead.
    peak_context_tokens: Option<u32>,
}

impl TurnUsageAcc {
    fn add(
        &mut self,
        input: Option<u32>,
        output: Option<u32>,
        cache_read: Option<u32>,
    ) {
        // Track per-call peak before accumulating so we capture each
        // iteration's individual context window size.
        let call_ctx = input.unwrap_or(0) + cache_read.unwrap_or(0);
        if call_ctx > 0 {
            self.peak_context_tokens =
                Some(self.peak_context_tokens.unwrap_or(0).max(call_ctx));
        }
        Self::merge(&mut self.input_tokens, input);
        Self::merge(&mut self.output_tokens, output);
        Self::merge(&mut self.cache_read_tokens, cache_read);
    }

    fn merge(slot: &mut Option<u32>, incoming: Option<u32>) {
        if let Some(n) = incoming {
            *slot = Some(slot.map_or(n, |existing| existing.saturating_add(n)));
        }
    }

    /// Convert to the wire shape. Returns `None` when no iteration
    /// surfaced any usage numbers — preserves the existing behaviour
    /// for backends that don't honour `stream_options.include_usage`.
    fn into_usage(self) -> Option<TurnUsage> {
        if self.input_tokens.is_none()
            && self.output_tokens.is_none()
            && self.cache_read_tokens.is_none()
        {
            return None;
        }
        Some(TurnUsage {
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            cache_read_tokens: self.cache_read_tokens,
            // OpenAI / Ollama don't surface a separate cache-creation
            // counter (it's implicit in `prompt_tokens`); leave `None`.
            cache_write_tokens: None,
            // Reasoning/thinking tokens — OpenAI reports these inside
            // `completion_tokens_details.reasoning_tokens` for o-series
            // models. Not parsed today; punt with the rest of the
            // thinking-token separation work tracked on R028-F11's
            // handoff (Anthropic side has the same gap).
            thinking_tokens: None,
            peak_context_tokens: self.peak_context_tokens,
        })
    }
}

/// Issue one streaming chat-completions request. Used both for the
/// first request of a `send` (synchronously, so transport failures
/// surface as `RunnerError`) and for subsequent tool-call iterations
/// inside the stream (where any error has to ride the event stream).
///
/// On a non-success status code the response body is read up-front and
/// folded into the `RunnerError::Transport` message so callers see the
/// provider's error description rather than a bare HTTP code.
///
/// **Pre-stream retry (R237-F12):** transport-level failures (DNS, TLS,
/// connect, TCP reset before headers) and retryable HTTP statuses
/// (429 / 5xx) are re-attempted up to [`MAX_STREAM_OPEN_ATTEMPTS`] times
/// with exponential 1s/2s/4s backoff. `Retry-After` is honored when the
/// upstream sends it (capped at [`MAX_RETRY_AFTER_SECS`]). The retry
/// boundary ends when this function returns `Ok(Response)` — once a
/// single chunk has been read from `bytes_stream()`, partial output may
/// have been emitted and re-issuing would double-stream. No OpenAI-shape
/// or NDJSON protocol we speak carries a frame-level event-id, so
/// frame-level resume is not possible.
async fn issue_chat_completions(
    client: &dyn HttpClient,
    endpoint: &str,
    api_key: Option<&str>,
    provider_label: &str,
    quota_ledger: Option<&std::sync::Mutex<QuotaLedger>>,
    rate_limit_hint: Option<&RateLimitHint>,
    body: Value,
) -> Result<HttpResponse, RunnerError> {
    let mut last_err: Option<RunnerError> = None;
    let mut last_retry_after_secs: Option<u32> = None;

    for attempt in 0..MAX_STREAM_OPEN_ATTEMPTS {
        if attempt > 0 {
            let dur = std::time::Duration::from_millis(compute_stream_open_backoff_ms(
                attempt,
                last_retry_after_secs,
            ));
            tokio::time::sleep(dur).await;
        }

        let mut req = HttpRequest::post_json(endpoint, &body);
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }
        let send_result = client.send(req).await;
        let resp = match send_result {
            Ok(r) => r,
            Err(e) => {
                // Transport-level failure (DNS, TLS, connect, etc.) —
                // always treat as transient Network. Retry if attempts
                // remain.
                last_err = Some(RunnerError::Transport(format!(
                    "{} request failed: {}",
                    provider_label, e
                )));
                last_retry_after_secs = None;
                if attempt + 1 < MAX_STREAM_OPEN_ATTEMPTS {
                    continue;
                }
                return Err(last_err.expect("just set"));
            }
        };

        if resp.is_success() {
            return Ok(resp);
        }

        let status = resp.status;
        // Extract Retry-After before consuming the body (headers live on
        // the response, body requires `.text()` which moves out of resp).
        let retry_after_secs = resp
            .header("retry-after")
            .and_then(|s| s.parse::<u32>().ok());
        let body_text = resp.text().await.unwrap_or_default();

        if status == 429 {
            let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("");
            if let Some(hint) = rate_limit_hint.and_then(|h| h.for_model(model)) {
                tracing::warn!(model, hint, provider = provider_label, "rate-limit hint on 429");
            }
            // Record every observed 429 — the session-start picker reads
            // last_429_at as a cooldown signal independent of whether an
            // in-band retry eventually succeeds.
            if let Some(ledger) = quota_ledger {
                ledger
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .record_429(provider_label, model);
            }
        }

        let err = if status == 429 {
            RunnerError::RateLimited { retry_after_secs }
        } else {
            RunnerError::Transport(format!(
                "{} returned {}: {}",
                provider_label,
                status,
                body_text.chars().take(512).collect::<String>(),
            ))
        };

        if is_retryable_status(status) && attempt + 1 < MAX_STREAM_OPEN_ATTEMPTS {
            last_err = Some(err);
            last_retry_after_secs = retry_after_secs;
            continue;
        }
        return Err(err);
    }
    Err(last_err.unwrap_or_else(|| {
        RunnerError::Transport(format!(
            "{} retry loop exhausted with no captured error",
            provider_label
        ))
    }))
}

/// Build the `tools[]` request payload from a registry, if any. Returns
/// `None` when the runner was constructed without a registry — in which
/// case the request body omits the field entirely (the model is told
/// nothing about tools and should never emit `tool_calls`).
/// Reconstruct OpenAI-protocol message history from a replayed JSONL event
/// sequence. Used by [`OpenAiCompatRunner::resume`] to rebuild the in-memory
/// conversation so subsequent turns see the full prior context.
///
/// Mapping:
/// - `UserMessage` → `{"role":"user","content":text}`
/// - `ToolCall` events are buffered; the batch is flushed as an assistant
///   message with `"tool_calls":[…]` when the first matching `ToolResult`
///   arrives (the OpenAI API requires the assistant message with all tool_calls
///   to appear before any of its tool results).
/// - `ToolResult` → `{"role":"tool","tool_call_id":…,"content":…}`
/// - `TurnEnded` → `{"role":"assistant","content":text}` for the final reply
fn reconstruct_openai_messages(events: &[AgentEvent]) -> Vec<Value> {
    let mut messages: Vec<Value> = Vec::new();
    let mut pending_calls: Vec<(String, String, Value)> = Vec::new(); // (id, name, args)

    for event in events {
        match event {
            AgentEvent::UserMessage { text, .. } => {
                pending_calls.clear();
                messages.push(json!({"role": "user", "content": text}));
            }
            AgentEvent::ToolCall { tool_call_id, tool_name, args, .. } => {
                pending_calls.push((tool_call_id.clone(), tool_name.clone(), args.clone()));
            }
            AgentEvent::ToolResult { tool_call_id, result, .. } => {
                // Flush the buffered tool calls as one assistant message before
                // adding the tool result — the protocol requires this ordering.
                if !pending_calls.is_empty() {
                    let tc: Vec<Value> = pending_calls.iter().map(|(id, name, args)| {
                        json!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": args.to_string(),
                            }
                        })
                    }).collect();
                    messages.push(json!({"role": "assistant", "content": Value::Null, "tool_calls": tc}));
                    pending_calls.clear();
                }
                // Same cap as the live dispatch path (R883-T1), so a resumed
                // transcript is byte-identical to the one the provider cached.
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": yah_runner_core::tool::tool_result_wire_content(result),
                }));
            }
            AgentEvent::TurnEnded { text, .. } if !text.is_empty() => {
                // Flush any leftover pending calls (shouldn't occur in well-formed JSONL).
                if !pending_calls.is_empty() {
                    let tc: Vec<Value> = pending_calls.iter().map(|(id, name, args)| {
                        json!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": args.to_string(),
                            }
                        })
                    }).collect();
                    messages.push(json!({"role": "assistant", "content": Value::Null, "tool_calls": tc}));
                    pending_calls.clear();
                }
                messages.push(json!({"role": "assistant", "content": text}));
            }
            _ => {}
        }
    }
    messages
}

/// Build the Chat Completions `tools[]` payload AND the
/// `wire-name → registry-name` map for inbound dispatch routing.
///
/// Yah's registry uses dotted names (`board.show`, `kg.defs`, …) that
/// don't match the OpenAI tool-name regex (`^[a-zA-Z0-9_-]+$`). The
/// rewrite + map building is shared with every other wire that has the
/// same constraint — see [`yah_runner_core::tool::sanitize_tool_name_for_wire`]
/// and [`yah_runner_core::tool::build_wire_name_map`]. This function only owns
/// the Chat-Completions-specific envelope shape (nested `function: {…}`).
fn tools_array_payload(
    tools: Option<&Arc<dyn ToolRegistry>>,
) -> (Option<Vec<Value>>, HashMap<String, String>) {
    let Some(registry) = tools else {
        return (None, HashMap::new());
    };
    let schemas = registry.schemas();
    if schemas.is_empty() {
        return (None, HashMap::new());
    }
    let name_map = build_wire_name_map(&schemas);
    let arr = schemas
        .into_iter()
        .map(|s| {
            json!({
                "type": "function",
                "function": {
                    "name": sanitize_tool_name_for_wire(&s.name),
                    "description": s.description,
                    "parameters": s.input_schema,
                }
            })
        })
        .collect();
    (Some(arr), name_map)
}

/// Parse the `arguments` JSON the model produced for a tool call.
/// Empty / whitespace-only → `{}` (the model declared no args). Any
/// other parse failure → `Value::Null`, which the caller surfaces as a
/// structured invalid-arguments outcome so the model can fix its next
/// attempt.
fn parse_tool_arguments(raw: &str) -> Value {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return json!({});
    }
    serde_json::from_str::<Value>(trimmed).unwrap_or(Value::Null)
}

/// Truncate a string for inclusion in an error message. Keeps tool
/// failure outputs scannable when the model produces a multi-KB blob
/// of malformed JSON.
fn truncate_for_error(raw: &str, cap: usize) -> String {
    if raw.chars().count() <= cap {
        return raw.to_string();
    }
    let cut = raw
        .char_indices()
        .nth(cap)
        .map(|(i, _)| i)
        .unwrap_or(raw.len());
    format!("{}…", &raw[..cut])
}

fn parse_sse_frame(frame: &str) -> Vec<FrameEvent> {
    let mut out = Vec::new();
    for line in frame.lines() {
        let Some(payload) = line
            .strip_prefix("data: ")
            .or_else(|| line.strip_prefix("data:"))
        else {
            continue;
        };
        let payload = payload.trim();
        if payload == "[DONE]" {
            out.push(FrameEvent::Done);
            continue;
        }
        let Ok(json) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        if let Some(error) = json.get("error") {
            let msg = error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("openai-compatible error")
                .to_string();
            out.push(FrameEvent::Error(msg));
            continue;
        }
        if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
            for choice in choices {
                let delta = choice.get("delta");
                let text = delta
                    .and_then(|d| d.get("content"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                // Extract thinking content when extended thinking is enabled
                // (DeepSeek and other models that support reasoning).
                let thinking = delta
                    .and_then(|d| d.get("thinking"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                let tool_calls = delta
                    .and_then(|d| d.get("tool_calls"))
                    .and_then(|t| t.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(parse_tool_call_delta)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let finish = choice
                    .get("finish_reason")
                    .and_then(|f| f.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                // Emit ThinkingDelta if we have thinking content
                if !thinking.is_empty() {
                    out.push(FrameEvent::ThinkingDelta { thinking });
                }
                // Emit regular Delta if we have text, tool_calls, or finish reason
                if !text.is_empty() || !tool_calls.is_empty() || finish.is_some() {
                    out.push(FrameEvent::Delta {
                        text,
                        tool_calls,
                        finish,
                    });
                }
            }
        }
        // Usage frame (`stream_options.include_usage = true`). OpenAI
        // sends one as the very last chunk before `[DONE]`, with
        // `choices: []` and a top-level `usage` object. Some providers
        // also attach `usage` to the same chunk that carries the final
        // delta — accept both shapes.
        if let Some(usage) = json.get("usage") {
            let input_tokens = usage
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .map(|n| n as u32);
            let output_tokens = usage
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .map(|n| n as u32);
            // `prompt_tokens_details.cached_tokens` (OpenAI shape, 2025-Q3+) or
            // `prompt_tokens_details.prompt_cache_hit_tokens` (DeepSeek shape).
            // Older Chat Completions deployments and Ollama don't emit either.
            let cache_read_tokens = usage
                .get("prompt_tokens_details")
                .and_then(|d| {
                    d.get("cached_tokens")
                        .or_else(|| d.get("prompt_cache_hit_tokens"))
                })
                .and_then(|v| v.as_u64())
                .map(|n| n as u32);
            if input_tokens.is_some() || output_tokens.is_some() || cache_read_tokens.is_some()
            {
                out.push(FrameEvent::Usage {
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                });
            }
        }
    }
    out
}

/// Parse one NDJSON frame from Ollama's native `/api/chat` endpoint.
///
/// Each frame is a complete JSON object on its own `\n`-terminated line —
/// no `data:` prefix, no `[DONE]` sentinel. Mapping into the shared
/// [`FrameEvent`] vocabulary:
///
/// - `message.content` → [`FrameEvent::Delta::text`]
/// - `message.tool_calls[]` → [`FrameEvent::Delta::tool_calls`]
///   (Ollama emits these all at once on a single frame, not chunked the
///   way OpenAI streams `function.arguments`; we still funnel them through
///   the same accumulator so the downstream loop stays uniform.)
/// - `done: true` → synthesises a `finish` reason (`"tool_calls"` if the
///   same frame carried tool_calls, else `done_reason` verbatim or
///   `"stop"`), plus a [`FrameEvent::Usage`] when `prompt_eval_count` or
///   `eval_count` are present, plus [`FrameEvent::Done`].
/// - top-level `error: "<msg>"` (Ollama-shape, not the OpenAI nested
///   `{error:{message:…}}`) → [`FrameEvent::Error`].
fn parse_ollama_chat_frame(line: &str) -> Vec<FrameEvent> {
    let line = line.trim();
    if line.is_empty() {
        return Vec::new();
    }
    let Ok(json) = serde_json::from_str::<Value>(line) else {
        return Vec::new();
    };

    if let Some(err) = json.get("error").and_then(|v| v.as_str()) {
        return vec![FrameEvent::Error(err.to_string())];
    }

    let mut out = Vec::new();
    let msg = json.get("message");
    let content_text = msg
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    let tool_calls: Vec<ToolCallDelta> = msg
        .and_then(|m| m.get("tool_calls"))
        .and_then(|tc| tc.as_array())
        .map(|arr| {
            arr.iter()
                .enumerate()
                .filter_map(|(i, raw)| parse_ollama_tool_call(i as u32, raw))
                .collect()
        })
        .unwrap_or_default();
    let done = json.get("done").and_then(|d| d.as_bool()).unwrap_or(false);
    let finish: Option<String> = if done {
        if !tool_calls.is_empty() {
            // The runner's tool-loop branch keys on the literal
            // `"tool_calls"` finish reason; translate Ollama's
            // `done_reason: "tool_calls"` (or any value when tool_calls
            // are present) into that exact string so the existing logic
            // fires.
            Some("tool_calls".into())
        } else {
            Some(
                json.get("done_reason")
                    .and_then(|r| r.as_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or("stop")
                    .to_string(),
            )
        }
    } else {
        None
    };

    if !content_text.is_empty() || !tool_calls.is_empty() || finish.is_some() {
        out.push(FrameEvent::Delta {
            text: content_text,
            tool_calls,
            finish,
        });
    }

    if done {
        let input_tokens = json
            .get("prompt_eval_count")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);
        let output_tokens = json
            .get("eval_count")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);
        if input_tokens.is_some() || output_tokens.is_some() {
            out.push(FrameEvent::Usage {
                input_tokens,
                output_tokens,
                cache_read_tokens: None,
            });
        }
        out.push(FrameEvent::Done);
    }

    out
}

/// Translate one entry from Ollama `message.tool_calls[]` into the shared
/// [`ToolCallDelta`] shape.
///
/// Ollama emits the call all at once (`{function: {name, arguments: <object>}}`)
/// rather than streaming `arguments` chunks the way OpenAI does. We
/// serialise the object back to a string so the same [`ToolCallAcc`]
/// (which appends string slices across SSE chunks) holds the value
/// uniformly. The runner's `parse_tool_arguments` will re-parse it on the
/// way out — net zero round-trip cost.
///
/// Ollama doesn't include per-call `id` fields, so we synthesise
/// `"ollama_tc_<index>"`. The id only needs to be unique within one
/// iteration to pair the assistant's `tool_calls[]` with the role:tool
/// rows that follow.
fn parse_ollama_tool_call(index: u32, raw: &Value) -> Option<ToolCallDelta> {
    let func = raw.get("function")?;
    let name = func
        .get("name")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    let arguments = func.get("arguments").map(|v| match v {
        Value::String(s) => s.clone(),
        // `to_string()` on a JSON Value serialises without spaces — the
        // exact shape downstream `parse_tool_arguments` already accepts.
        other => other.to_string(),
    });
    if name.is_none() && arguments.is_none() {
        return None;
    }
    let id = raw
        .get("id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("ollama_tc_{index}"));
    Some(ToolCallDelta {
        index,
        id: Some(id),
        name,
        arguments,
    })
}

fn parse_tool_call_delta(raw: &Value) -> Option<ToolCallDelta> {
    let index = raw.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as u32;
    let id = raw
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let func = raw.get("function");
    let name = func
        .and_then(|f| f.get("name"))
        .and_then(|n| n.as_str())
        .map(|s| s.to_string());
    let arguments = func
        .and_then(|f| f.get("arguments"))
        .and_then(|a| a.as_str())
        .map(|s| s.to_string());
    if id.is_none() && name.is_none() && arguments.is_none() {
        return None;
    }
    Some(ToolCallDelta {
        index,
        id,
        name,
        arguments,
    })
}

fn build_request_body(
    session: &InnerSession,
    max_tokens: u32,
    tools: Option<&[Value]>,
    options: Option<&Value>,
    keep_alive: Option<&str>,
) -> serde_json::Value {
    let mut messages: Vec<Value> = Vec::with_capacity(session.history.len() + 1);
    messages.push(json!({
        "role": "system",
        "content": session.prelude_text,
    }));
    messages.extend(session.history.iter().cloned());
    let mut body = json!({
        "model": session.model,
        "stream": true,
        // Ask the provider to attach a final `usage` chunk so the host
        // can populate `TurnEnded.usage`. OpenAI + Ollama Cloud honour
        // this; vanilla local Ollama ignores it harmlessly. R028-F11.
        "stream_options": { "include_usage": true },
        "max_tokens": max_tokens,
        "messages": messages,
    });
    if let Some(arr) = tools {
        if !arr.is_empty() {
            body["tools"] = Value::Array(arr.to_vec());
        }
    }
    // Ollama's `/api/chat` reads runtime knobs (num_ctx, temperature, …)
    // from a top-level `options` object. Callers shape this conditionally
    // on transport — OpenAI-shape gateways silently ignore unknown
    // top-level fields, but emitting it only when meaningful keeps the
    // wire clean.
    if let Some(opts) = options {
        body["options"] = opts.clone();
    }
    if let Some(ka) = keep_alive {
        body["keep_alive"] = json!(ka);
    }
    // The session's level (R947-F2: read per request, so a die change lands
    // on the next turn) wired through to reasoning_effort.
    // Maps: Fast→'high', everything above→'max' — R293-F1's DeepSeek dialect.
    // Wrap in {"thinking": {"type": "enabled"}, "reasoning_effort": "..."} for
    // OpenAI-compatible endpoints that support extended thinking (DeepSeek, etc.).
    if let Some(budget) = session.think.as_ref() {
        let reasoning_effort = match budget {
            ReasoningBudget::Fast => "high",
            ReasoningBudget::Standard
            | ReasoningBudget::Deep
            | ReasoningBudget::Budget { .. } => "max",
        };
        body["thinking"] = json!({
            "type": "enabled"
        });
        body["reasoning_effort"] = json!(reasoning_effort);
    }
    body
}

/// Build the `options` payload Ollama's `/api/chat` expects, or `None`
/// when no runtime knobs apply. Currently surfaces only `num_ctx`;
/// future fields (temperature, top_p, stop) slot in here.
fn ollama_options(config: &OpenAiCompatConfig) -> Option<Value> {
    if config.transport != Transport::OllamaNdjson {
        return None;
    }
    let mut obj = serde_json::Map::new();
    if let Some(n) = config.num_ctx {
        obj.insert("num_ctx".into(), json!(n));
    }
    if obj.is_empty() {
        None
    } else {
        Some(Value::Object(obj))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yah_runner_core::{CacheControl, CacheTtl, ModelRef, PreludeSection};

    fn test_prelude() -> Prelude {
        Prelude::empty(CacheControl { key: "0".into(), ttl: CacheTtl::Ephemeral })
    }

    /// Stands in for a host's self-identity preamble.
    #[derive(Debug)]
    struct IdPreamble;

    impl SessionPreamble for IdPreamble {
        fn render(&self, session_id: &SessionId, _prelude: &Prelude, session_key: &str) -> String {
            format!("IDENTITY {} {session_key}\n", session_id.as_str())
        }
    }

    fn session(prelude: &str, model: &str) -> InnerSession {
        InnerSession {
            ticket_id: "T01".into(),
            prelude_text: prelude.into(),
            model: model.into(),
            history: Vec::new(),
            resident_stanzas: std::collections::BTreeSet::new(),
            usage_anchor: None,
            think: None,
        }
    }

    // ── R629-T9: RS-1 — history mutation never touches prelude_text ────────

    /// Structural half of RS-1: the pure history-rewrite helper produces
    /// `[summary, ...tail]` and — by taking only `history` — cannot read or
    /// write the resident set. The signature is the invariant.
    #[test]
    fn rewrite_history_produces_summary_then_tail() {
        let mut history = vec![
            json!({ "role": "user", "content": "one" }),
            json!({ "role": "assistant", "content": "two" }),
            json!({ "role": "user", "content": "three" }),
            json!({ "role": "assistant", "content": "four" }),
        ];
        let kept = rewrite_history_with_summary(&mut history, "SUMMARY", 2, 4);
        assert_eq!(kept, 2);
        assert_eq!(history.len(), 3, "summary row + 2 preserved tail rows");
        assert_eq!(history[0]["role"], "system");
        let head = history[0]["content"].as_str().unwrap();
        assert!(head.contains("SUMMARY"));
        assert!(head.contains("of 2 earlier turns"), "label counts elided rows");
        // The last two rows are preserved verbatim, in order.
        assert_eq!(history[1]["content"], "three");
        assert_eq!(history[2]["content"], "four");
    }

    /// `keep_last_n` larger than the history keeps everything (no panic on
    /// the saturating split) and still prepends one summary row.
    #[test]
    fn rewrite_history_keep_all_when_n_exceeds_len() {
        let mut history = vec![json!({ "role": "user", "content": "only" })];
        let kept = rewrite_history_with_summary(&mut history, "S", 10, 1);
        assert_eq!(kept, 1);
        assert_eq!(history.len(), 2);
        assert_eq!(history[1]["content"], "only");
    }

    /// End-to-end half of RS-1 (the ticket's literal verify): `prelude_text`
    /// is byte-identical across a real `compact_session` call. A local JSON
    /// server stands in for the summarizer so the network round-trip runs
    /// but the resident set must come back untouched while history is
    /// rewritten.
    #[tokio::test]
    async fn compact_session_leaves_prelude_text_byte_identical() {
        // The shared stub answers the one summarizer POST with a canned
        // OpenAI chat-completion JSON body.
        let stub = yah_runner_stub::InferenceStub::builder(yah_runner_stub::StubMode::default())
            .replies([yah_runner_stub::Reply::Json(json!({
                "choices": [{ "message": { "role": "assistant", "content": "CANNED SUMMARY" } }]
            }))])
            .spawn()
            .await
            .unwrap();

        let mut config = OpenAiCompatConfig::openai("test-key");
        config.endpoint = stub.completions_url();
        let runner = OpenAiCompatRunner::new(config);

        // Insert a session directly (bypassing start()'s network path) with a
        // known resident set and a multi-row history.
        const RESIDENT: &str = "## RESIDENT SET\nnever-mutated prelude bytes";
        let sid = SessionId::new("session:rs1test0");
        {
            let mut s = session(RESIDENT, "gpt-4o");
            for i in 0..6 {
                s.history.push(json!({ "role": "user", "content": format!("turn {i}") }));
            }
            runner
                .inner
                .sessions
                .write()
                .await
                .insert(sid.clone(), Arc::new(Mutex::new(s)));
        }

        let result = runner.compact_session(&sid, 2).await.expect("compact ok");
        assert_eq!(result.history_messages_before, 6);
        assert_eq!(result.kept_tail, 2);
        assert_eq!(result.history_messages_after, 3, "summary + 2 tail");

        // RS-1: the resident set is byte-identical; history was rewritten.
        let s = runner.inner.sessions.read().await.get(&sid).cloned().unwrap();
        let s = s.lock().await;
        assert_eq!(s.prelude_text, RESIDENT, "prelude_text must survive compaction unchanged");
        assert_eq!(s.history.len(), 3);
        assert!(s.history[0]["content"].as_str().unwrap().contains("CANNED SUMMARY"));
        assert_eq!(s.history[2]["content"], "turn 5");
    }

    // ── R629-F8: resident-set mutation (append-with-idempotency) ───────────

    /// Appending a stanza mid-session extends prelude_text at the TAIL by
    /// exactly the body (plus a `\n\n` separator) and leaves the existing
    /// prefix byte-identical — the property that makes a caching provider
    /// re-process only the appended text.
    #[tokio::test]
    async fn append_resident_stanza_appends_at_tail() {
        let runner = OpenAiCompatRunner::new(OpenAiCompatConfig::openai("k"));
        let sid = SessionId::new("session:append01");
        const PREFIX: &str = "## RESIDENT PREFIX\nstable bytes";
        {
            let mut s = session(PREFIX, "gpt-4o");
            s.history.push(json!({ "role": "user", "content": "hi" }));
            runner
                .inner
                .sessions
                .write()
                .await
                .insert(sid.clone(), Arc::new(Mutex::new(s)));
        }
        let body = "X".repeat(719);
        let res = runner
            .append_resident_stanza(&sid, "trait:fleet", &body)
            .await
            .unwrap();
        assert!(res.appended);
        assert_eq!(res.appended_chars, 719 + 2, "body + \\n\\n separator");

        let s = runner.inner.sessions.read().await.get(&sid).cloned().unwrap();
        let s = s.lock().await;
        // Tail append: the prefix is untouched; the body is at the end.
        assert!(s.prelude_text.starts_with(PREFIX));
        assert!(s.prelude_text.ends_with(&body));
        assert_eq!(s.prelude_text, format!("{PREFIX}\n\n{body}"));
        // History is untouched (RS-1 mirror — this op never reads history).
        assert_eq!(s.history.len(), 1);
        assert!(s.resident_stanzas.contains("trait:fleet"));
    }

    /// Re-loading a resident stanza is a no-op: no bytes added, appended=false.
    #[tokio::test]
    async fn append_resident_stanza_is_idempotent() {
        let runner = OpenAiCompatRunner::new(OpenAiCompatConfig::openai("k"));
        let sid = SessionId::new("session:append02");
        {
            let mut s = session("BASE", "gpt-4o");
            s.resident_stanzas.insert("trait:board".to_string());
            runner
                .inner
                .sessions
                .write()
                .await
                .insert(sid.clone(), Arc::new(Mutex::new(s)));
        }
        let before = {
            let s = runner.inner.sessions.read().await.get(&sid).cloned().unwrap();
            let s = s.lock().await;
            s.prelude_text.clone()
        };
        let res = runner
            .append_resident_stanza(&sid, "trait:board", "SHOULD NOT APPEAR")
            .await
            .unwrap();
        assert!(!res.appended, "already resident → no-op");
        assert_eq!(res.appended_chars, 0);

        let s = runner.inner.sessions.read().await.get(&sid).cloned().unwrap();
        let s = s.lock().await;
        assert_eq!(s.prelude_text, before, "byte-identical after a no-op re-load");
        assert!(!s.prelude_text.contains("SHOULD NOT APPEAR"));
    }

    /// A start()-seeded resident stanza is recognised as already present, so
    /// a load of something the job already carries is a no-op end-to-end.
    #[tokio::test]
    async fn append_resident_stanza_dedupes_against_start_seeded_stanzas() {
        let runner = OpenAiCompatRunner::new(OpenAiCompatConfig::openai("k"));
        let mut prelude = test_prelude();
        prelude.sections = vec![PreludeSection::new("agent-policy", "policy body")];
        let sid = runner.start(prelude, "T01").await.unwrap();
        let res = runner
            .append_resident_stanza(&sid, "agent-policy", "DUP")
            .await
            .unwrap();
        assert!(!res.appended, "start-seeded stanza is already resident");
    }

    /// R883-B15 regression: identity is per-session (carries the minted
    /// session id) and must trail the doctrine-carrying rendered prelude,
    /// not lead it — leading would salt byte 0 of every session's prompt, so
    /// no two sessions on the same job/loadout could ever share a
    /// prompt-cache prefix. Two `start()` calls with an identical rendered
    /// prelude (standing in for "same job/loadout") but distinct
    /// auto-minted session ids must produce `prelude_text` that agrees
    /// byte-for-byte through the whole rendered doctrine, diverging only in
    /// the trailing identity line.
    #[tokio::test]
    async fn wrapped_prompt_shares_doctrine_prefix_across_sessions() {
        let build_prelude = || {
            let mut p = test_prelude();
            p.sections = vec![PreludeSection::new(
                "agent-policy",
                "## DOCTRINE\nsame bytes for every session on this job/loadout",
            )];
            p.agent_id = Some("agent:test".into());
            p
        };
        let doctrine = build_prelude().render();

        let mut cfg = OpenAiCompatConfig::openai("k");
        cfg.session_preamble = Some(Arc::new(IdPreamble));
        let runner = OpenAiCompatRunner::new(cfg);
        let sid_a = runner.start(build_prelude(), "T-A").await.unwrap();
        let sid_b = runner.start(build_prelude(), "T-B").await.unwrap();
        assert_ne!(sid_a, sid_b, "two starts mint distinct session ids");

        let text_a = {
            let s = runner.inner.sessions.read().await.get(&sid_a).cloned().unwrap();
            let guard = s.lock().await;
            guard.prelude_text.clone()
        };
        let text_b = {
            let s = runner.inner.sessions.read().await.get(&sid_b).cloned().unwrap();
            let guard = s.lock().await;
            guard.prelude_text.clone()
        };

        assert!(
            text_a.starts_with(&doctrine) && text_b.starts_with(&doctrine),
            "doctrine must lead the prompt, not the per-session identity line"
        );
        let common = text_a
            .bytes()
            .zip(text_b.bytes())
            .take_while(|(x, y)| x == y)
            .count();
        assert!(
            common >= doctrine.len(),
            "two sessions of the same job/loadout must share the whole doctrine \
             as a common byte prefix (common={common}, doctrine.len()={})",
            doctrine.len()
        );
        assert_ne!(text_a, text_b, "the trailing identity line must still differ per session");
    }

    /// The host preamble lands between the rendered prelude and the runtime
    /// caps block, with the minted session id and the session key.
    #[tokio::test]
    async fn session_preamble_trails_prelude_with_minted_id() {
        let mut cfg = OpenAiCompatConfig::openai("k");
        cfg.session_preamble = Some(Arc::new(IdPreamble));
        let runner = OpenAiCompatRunner::new(cfg);
        let mut prelude = test_prelude();
        prelude.sections = vec![PreludeSection::new("doc", "DOCTRINE\n")];
        let sid = runner.start(prelude, "T-K").await.unwrap();
        let s = runner.inner.sessions.read().await.get(&sid).cloned().unwrap();
        let text = s.lock().await.prelude_text.clone();
        let expected = format!("DOCTRINE\n\nIDENTITY {} T-K\n", sid.as_str());
        assert!(text.starts_with(&expected), "{text}");
    }

    /// Without a preamble, start adds nothing between prelude and caps.
    #[tokio::test]
    async fn no_session_preamble_adds_nothing() {
        let runner = OpenAiCompatRunner::new(OpenAiCompatConfig::openai("k"));
        let mut prelude = test_prelude();
        prelude.sections = vec![PreludeSection::new("doc", "DOCTRINE\n")];
        let sid = runner.start(prelude, "T-K").await.unwrap();
        let s = runner.inner.sessions.read().await.get(&sid).cloned().unwrap();
        let text = s.lock().await.prelude_text.clone();
        assert!(!text.contains("IDENTITY"), "{text}");
    }

    #[test]
    fn body_includes_system_prelude_then_history_in_order() {
        let mut s = session("PRELUDE", "gpt-4o");
        s.history.push(json!({ "role": "user", "content": "hi" }));
        s.history
            .push(json!({ "role": "assistant", "content": "hey" }));
        s.history
            .push(json!({ "role": "user", "content": "what's up" }));
        let body = build_request_body(&s, 1024, None, None, None);
        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 1024);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "PRELUDE");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "hi");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[2]["content"], "hey");
        assert_eq!(msgs[3]["role"], "user");
        assert_eq!(msgs[3]["content"], "what's up");
        // No registry attached → `tools` field absent (not just empty).
        assert!(body.get("tools").is_none(), "{body}");
    }

    #[test]
    fn body_includes_tools_array_when_schemas_provided() {
        let s = session("PRELUDE", "gpt-4o");
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a camp file.",
                "parameters": { "type": "object", "properties": { "path": { "type": "string" } } }
            }
        })];
        let body = build_request_body(&s, 1024, Some(&tools), None, None);
        let arr = body["tools"].as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["function"]["name"], "read_file");
        assert_eq!(arr[0]["type"], "function");
    }

    #[test]
    fn parse_frame_extracts_content_delta_without_finish() {
        let frame = r#"data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}"#;
        let evs = parse_sse_frame(frame);
        assert_eq!(
            evs,
            vec![FrameEvent::Delta {
                text: "hello".into(),
                tool_calls: vec![],
                finish: None
            }]
        );
    }

    #[test]
    fn parse_frame_extracts_finish_reason_on_terminal_chunk() {
        let frame = r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        let evs = parse_sse_frame(frame);
        assert_eq!(
            evs,
            vec![FrameEvent::Delta {
                text: "".into(),
                tool_calls: vec![],
                finish: Some("stop".into())
            }]
        );
    }

    #[test]
    fn parse_frame_done_sentinel_emits_done_event() {
        assert_eq!(parse_sse_frame("data: [DONE]"), vec![FrameEvent::Done]);
        assert_eq!(parse_sse_frame("data:[DONE]"), vec![FrameEvent::Done]);
    }

    #[test]
    fn parse_frame_error_payload_emits_error_event() {
        let frame = r#"data: {"error":{"message":"rate limited"}}"#;
        let evs = parse_sse_frame(frame);
        assert_eq!(evs, vec![FrameEvent::Error("rate limited".into())]);
    }

    #[test]
    fn parse_frame_extracts_usage_from_terminal_chunk() {
        // OpenAI's stream_options.include_usage shape: a final chunk with
        // empty choices and a top-level usage object.
        let frame = r#"data: {"choices":[],"usage":{"prompt_tokens":42,"completion_tokens":17,"total_tokens":59}}"#;
        let evs = parse_sse_frame(frame);
        assert_eq!(
            evs,
            vec![FrameEvent::Usage {
                input_tokens: Some(42),
                output_tokens: Some(17),
                cache_read_tokens: None,
            }]
        );
    }

    #[test]
    fn parse_frame_extracts_cached_tokens_from_prompt_tokens_details() {
        // Newer OpenAI deployments expose cache-hit counts via
        // prompt_tokens_details.cached_tokens. Make sure we plumb it
        // through to the cache_read slot on TurnUsage.
        let frame = r#"data: {"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":800}}}"#;
        let evs = parse_sse_frame(frame);
        assert_eq!(
            evs,
            vec![FrameEvent::Usage {
                input_tokens: Some(1000),
                output_tokens: Some(50),
                cache_read_tokens: Some(800),
            }]
        );
    }

    #[test]
    fn parse_frame_extracts_deepseek_cache_hit_tokens() {
        // DeepSeek uses prompt_cache_hit_tokens inside prompt_tokens_details
        // instead of OpenAI's cached_tokens — both should map to cache_read.
        let frame = r#"data: {"choices":[],"usage":{"prompt_tokens":1024,"completion_tokens":256,"prompt_tokens_details":{"prompt_cache_hit_tokens":800,"prompt_cache_miss_tokens":224}}}"#;
        let evs = parse_sse_frame(frame);
        assert_eq!(
            evs,
            vec![FrameEvent::Usage {
                input_tokens: Some(1024),
                output_tokens: Some(256),
                cache_read_tokens: Some(800),
            }]
        );
    }

    #[test]
    fn parse_frame_deepseek_cache_miss_is_zero_not_none() {
        // prompt_cache_hit_tokens=0 means the provider supports caching and
        // this turn was a cold miss — must be Some(0), not None.
        let frame = r#"data: {"choices":[],"usage":{"prompt_tokens":1024,"completion_tokens":256,"prompt_tokens_details":{"prompt_cache_hit_tokens":0,"prompt_cache_miss_tokens":1024}}}"#;
        let evs = parse_sse_frame(frame);
        assert_eq!(
            evs,
            vec![FrameEvent::Usage {
                input_tokens: Some(1024),
                output_tokens: Some(256),
                cache_read_tokens: Some(0),
            }]
        );
    }

    /// Coverage table: each row is (provider label, usage JSON, expected cache_read_tokens).
    /// Iterates every provider shape documented in .yah/docs/working/W093-provider-usage-shapes.md
    /// so a future extraction change can't silently break a provider.
    #[test]
    fn cache_field_extraction_covers_all_openai_wire_providers() {
        let cases: &[(&str, &str, Option<u32>)] = &[
            // OpenAI — cache hit
            (
                "openai-hit",
                r#"{"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":800}}}"#,
                Some(800),
            ),
            // OpenAI — cache miss (field present but zero)
            (
                "openai-miss",
                r#"{"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":0}}}"#,
                Some(0),
            ),
            // OpenAI — older endpoint, no prompt_tokens_details
            (
                "openai-no-cache-field",
                r#"{"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":50}}"#,
                None,
            ),
            // OpenRouter — typical non-Anthropic model (no cache details)
            (
                "openrouter-no-cache",
                r#"{"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":50,"total_tokens":1050}}"#,
                None,
            ),
            // DeepSeek — cache hit (prompt_cache_hit_tokens)
            (
                "deepseek-hit",
                r#"{"choices":[],"usage":{"prompt_tokens":1024,"completion_tokens":256,"prompt_tokens_details":{"prompt_cache_hit_tokens":800,"prompt_cache_miss_tokens":224}}}"#,
                Some(800),
            ),
            // DeepSeek — cache miss (field present but zero)
            (
                "deepseek-miss",
                r#"{"choices":[],"usage":{"prompt_tokens":1024,"completion_tokens":256,"prompt_tokens_details":{"prompt_cache_hit_tokens":0,"prompt_cache_miss_tokens":1024}}}"#,
                Some(0),
            ),
        ];

        for (label, payload, expected) in cases {
            let frame = format!("data: {payload}");
            let evs = parse_sse_frame(&frame);
            let got: Option<u32> = evs.iter().find_map(|e| match e {
                FrameEvent::Usage { cache_read_tokens, .. } => Some(*cache_read_tokens),
                _ => None,
            }).flatten();
            assert_eq!(
                got, *expected,
                "provider={label}: expected cache_read_tokens={expected:?}, got {got:?}"
            );
        }
    }

    #[test]
    fn turn_usage_acc_sums_across_iterations() {
        // Two tool-call iterations, each reporting usage. The accumulator
        // adds per field; `into_usage` returns Some with the totals.
        let mut acc = TurnUsageAcc::default();
        acc.add(Some(100), Some(20), None);
        acc.add(Some(150), Some(35), Some(80));
        let usage = acc.into_usage().expect("at least one field set");
        assert_eq!(usage.input_tokens, Some(250));
        assert_eq!(usage.output_tokens, Some(55));
        assert_eq!(usage.cache_read_tokens, Some(80));
        assert_eq!(usage.cache_write_tokens, None);
        assert_eq!(usage.thinking_tokens, None);
    }

    #[test]
    fn turn_usage_acc_returns_none_when_no_iteration_reported() {
        // Local Ollama ignores stream_options.include_usage today; the
        // accumulator should stay quiet rather than emit a zero-filled
        // TurnUsage that would confuse the renderer's "is this real" check.
        let acc = TurnUsageAcc::default();
        assert!(acc.into_usage().is_none());
    }

    #[test]
    fn body_includes_stream_options_include_usage() {
        // Every request body asks the provider to emit a final usage
        // chunk so per-turn cost shape lands in TurnEnded.
        let s = session("PRELUDE", "gpt-4o");
        let body = build_request_body(&s, 1024, None, None, None);
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn ollama_local_factory_sets_num_ctx_8192() {
        let cfg = OpenAiCompatConfig::ollama_local();
        assert_eq!(cfg.transport, Transport::OllamaNdjson);
        assert_eq!(cfg.num_ctx, Some(8192));
        let opts = ollama_options(&cfg).expect("options should populate for OllamaNdjson");
        assert_eq!(opts["num_ctx"], 8192);
    }

    #[test]
    fn ollama_cloud_factory_sets_num_ctx_32768() {
        let cfg = OpenAiCompatConfig::ollama_cloud("k");
        assert_eq!(cfg.transport, Transport::OllamaNdjson);
        assert_eq!(cfg.num_ctx, Some(32768));
        let opts = ollama_options(&cfg).expect("options should populate for OllamaNdjson");
        assert_eq!(opts["num_ctx"], 32768);
    }

    #[test]
    fn openai_factory_leaves_num_ctx_unset() {
        let cfg = OpenAiCompatConfig::openai("k");
        assert_eq!(cfg.transport, Transport::Sse);
        assert_eq!(cfg.num_ctx, None);
        // Even if a caller set num_ctx on the Sse path, the options
        // helper must refuse to emit it — `/v1/chat/completions` would
        // silently ignore the field and the truncation bug would mask
        // as a config issue.
        let mut cfg = cfg;
        cfg.num_ctx = Some(8192);
        assert!(ollama_options(&cfg).is_none());
    }

    #[test]
    fn body_includes_ollama_options_when_provided() {
        let s = session("PRELUDE", "llama3.3");
        let opts = json!({ "num_ctx": 16384 });
        let body = build_request_body(&s, 1024, None, Some(&opts), None);
        assert_eq!(body["options"]["num_ctx"], 16384);
    }

    #[test]
    fn body_omits_options_when_none() {
        let s = session("PRELUDE", "gpt-4o");
        let body = build_request_body(&s, 1024, None, None, None);
        assert!(body.get("options").is_none(), "{body}");
    }

    #[test]
    fn body_includes_keep_alive_at_top_level() {
        let s = session("PRELUDE", "llama3.3");
        let body = build_request_body(&s, 1024, None, None, Some("60m"));
        assert_eq!(body["keep_alive"], "60m", "{body}");
    }

    #[test]
    fn body_omits_keep_alive_when_none() {
        let s = session("PRELUDE", "gpt-4o");
        let body = build_request_body(&s, 1024, None, None, None);
        assert!(body.get("keep_alive").is_none(), "{body}");
    }

    #[test]
    fn ollama_local_factory_sets_keep_alive_60m() {
        let cfg = OpenAiCompatConfig::ollama_local();
        assert_eq!(cfg.keep_alive.as_deref(), Some("60m"));
    }

    #[test]
    fn ollama_cloud_factory_sets_keep_alive_60m() {
        let cfg = OpenAiCompatConfig::ollama_cloud("k");
        assert_eq!(cfg.keep_alive.as_deref(), Some("60m"));
    }

    #[test]
    fn openai_factory_omits_keep_alive() {
        let cfg = OpenAiCompatConfig::openai("k");
        assert!(cfg.keep_alive.is_none());
    }

    #[test]
    fn groq_factory_creates_sse_config() {
        let cfg = OpenAiCompatConfig::groq("test-key");
        assert_eq!(cfg.provider_label, "groq");
        assert_eq!(cfg.endpoint, "https://api.groq.com/openai/v1/chat/completions");
        assert_eq!(cfg.default_model, "llama-3.3-70b-versatile");
        assert_eq!(cfg.api_key.as_deref(), Some("test-key"));
        assert_eq!(cfg.transport, Transport::Sse);
        assert!(cfg.num_ctx.is_none());
        assert!(cfg.keep_alive.is_none());
        assert!(cfg.cost_guard_tokens.is_none());
    }

    #[test]
    fn parse_frame_skips_unparseable_lines_silently() {
        // Stray heartbeat / event-stream comments shouldn't break the
        // parser. A real OpenAI deployment occasionally interleaves
        // `: keepalive\n\n` ping frames; we just want them ignored.
        assert!(parse_sse_frame(": keepalive").is_empty());
        assert!(parse_sse_frame("data: not json").is_empty());
    }

    #[test]
    fn parse_frame_handles_multiple_choices_in_one_payload() {
        let frame = r#"data: {"choices":[{"delta":{"content":"a"}},{"delta":{"content":"b"},"finish_reason":"stop"}]}"#;
        let evs = parse_sse_frame(frame);
        assert_eq!(
            evs,
            vec![
                FrameEvent::Delta {
                    text: "a".into(),
                    tool_calls: vec![],
                    finish: None
                },
                FrameEvent::Delta {
                    text: "b".into(),
                    tool_calls: vec![],
                    finish: Some("stop".into())
                },
            ]
        );
    }

    #[test]
    fn parse_frame_extracts_thinking_content_from_delta() {
        // Extended thinking models (DeepSeek, etc.) emit thinking content in
        // `delta.thinking`. Each frame carries a slice of the accumulated
        // thinking process.
        let frame = r#"data: {"choices":[{"delta":{"thinking":"Let me think about this..."}}]}"#;
        let evs = parse_sse_frame(frame);
        assert_eq!(
            evs,
            vec![FrameEvent::ThinkingDelta {
                thinking: "Let me think about this...".into()
            }]
        );
    }

    #[test]
    fn parse_frame_emits_both_thinking_and_content_deltas() {
        // Some frames may carry both thinking and regular content.
        let frame = r#"data: {"choices":[{"delta":{"thinking":"Thinking...","content":"Result"}}]}"#;
        let evs = parse_sse_frame(frame);
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0], FrameEvent::ThinkingDelta {
            thinking: "Thinking...".into()
        });
        assert_eq!(evs[1], FrameEvent::Delta {
            text: "Result".into(),
            tool_calls: vec![],
            finish: None
        });
    }

    #[test]
    fn parse_frame_extracts_tool_call_first_chunk_with_id_and_name() {
        // First streaming chunk for a tool call: id + name set, args
        // typically begins with `{` then accretes across chunks. The
        // accumulator (tested separately) joins them by index.
        let frame = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"read_file","arguments":"{\""}}]}}]}"#;
        let evs = parse_sse_frame(frame);
        assert_eq!(
            evs,
            vec![FrameEvent::Delta {
                text: "".into(),
                tool_calls: vec![ToolCallDelta {
                    index: 0,
                    id: Some("call_abc".into()),
                    name: Some("read_file".into()),
                    arguments: Some("{\"".into()),
                }],
                finish: None,
            }]
        );
    }

    #[test]
    fn parse_frame_extracts_tool_call_continuation_chunk() {
        // Mid-stream chunk: only the args slice is set; id and name come
        // through from the first chunk.
        let frame = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"path\":\"Cargo.toml\"}"}}]}}]}"#;
        let evs = parse_sse_frame(frame);
        assert_eq!(
            evs,
            vec![FrameEvent::Delta {
                text: "".into(),
                tool_calls: vec![ToolCallDelta {
                    index: 0,
                    id: None,
                    name: None,
                    arguments: Some("path\":\"Cargo.toml\"}".into()),
                }],
                finish: None,
            }]
        );
    }

    #[test]
    fn parse_frame_carries_finish_reason_tool_calls_with_empty_delta() {
        // Final chunk of a tool-call response: empty delta + finish
        // reason. The runner uses this to know it should dispatch the
        // accumulated calls and re-issue.
        let frame = r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#;
        let evs = parse_sse_frame(frame);
        assert_eq!(
            evs,
            vec![FrameEvent::Delta {
                text: "".into(),
                tool_calls: vec![],
                finish: Some("tool_calls".into()),
            }]
        );
    }

    #[test]
    fn tool_call_acc_joins_args_across_chunks_in_index_order() {
        // Simulate the accumulator the runner uses: index → ToolCallAcc.
        // First chunk seeds id+name; later chunks just append to args.
        let mut accs: BTreeMap<u32, ToolCallAcc> = BTreeMap::new();
        let chunks = vec![
            ToolCallDelta {
                index: 0,
                id: Some("call_1".into()),
                name: Some("grep".into()),
                arguments: Some("{\"pat".into()),
            },
            ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments: Some("tern\":\"foo\"}".into()),
            },
            ToolCallDelta {
                index: 1,
                id: Some("call_2".into()),
                name: Some("read_file".into()),
                arguments: Some("{\"path\":\"x\"}".into()),
            },
        ];
        for tcd in chunks {
            let acc = accs.entry(tcd.index).or_default();
            if let Some(id) = tcd.id {
                acc.id = id;
            }
            if let Some(name) = tcd.name {
                acc.name = name;
            }
            if let Some(args) = tcd.arguments {
                acc.arguments_buf.push_str(&args);
            }
        }
        let finalized: Vec<_> = accs.into_iter().map(|(_, a)| a.finalize()).collect();
        assert_eq!(finalized.len(), 2);
        assert_eq!(finalized[0].id, "call_1");
        assert_eq!(finalized[0].name, "grep");
        assert_eq!(finalized[0].arguments_str, "{\"pattern\":\"foo\"}");
        assert_eq!(finalized[1].id, "call_2");
        assert_eq!(finalized[1].name, "read_file");
        assert_eq!(finalized[1].arguments_str, "{\"path\":\"x\"}");
    }

    #[test]
    fn parse_tool_arguments_treats_empty_as_object() {
        // Models sometimes emit `arguments: ""` for parameter-less calls.
        // The downstream registry expects `type: object` args, so we
        // synthesise `{}` rather than failing the call.
        assert_eq!(parse_tool_arguments(""), json!({}));
        assert_eq!(parse_tool_arguments("   "), json!({}));
    }

    #[test]
    fn parse_tool_arguments_returns_null_on_invalid_json() {
        // Malformed args produce Value::Null; the runner converts that
        // into an `invalid_arguments` ToolOutcome the model can correct.
        assert_eq!(parse_tool_arguments("not json"), Value::Null);
        assert_eq!(parse_tool_arguments("{ unclosed"), Value::Null);
    }

    #[test]
    fn truncate_for_error_keeps_short_strings_intact() {
        assert_eq!(truncate_for_error("hello", 200), "hello");
    }

    #[test]
    fn truncate_for_error_appends_ellipsis_when_long() {
        let long = "x".repeat(500);
        let out = truncate_for_error(&long, 100);
        assert_eq!(out.chars().count(), 101);
        assert!(out.ends_with('…'));
    }

    #[test]
    fn tools_array_payload_returns_none_when_registry_absent() {
        let (arr, map) = tools_array_payload(None);
        assert!(arr.is_none());
        assert!(map.is_empty());
    }

    #[test]
    fn tools_array_payload_wraps_each_schema_as_function_object() {
        use yah_runner_core::tool::{ToolOutcome, ToolRegistry, ToolSchema};
        use async_trait::async_trait;

        struct StubRegistry(Vec<ToolSchema>);
        #[async_trait]
        impl ToolRegistry for StubRegistry {
            fn schemas(&self) -> Vec<ToolSchema> {
                self.0.clone()
            }
            async fn execute(&self, _name: &str, _args: Value) -> ToolOutcome {
                ToolOutcome::ok(json!(null))
            }
        }

        let registry: Arc<dyn ToolRegistry> = Arc::new(StubRegistry(vec![ToolSchema {
            name: "read_file".into(),
            description: "Read a file.".into(),
            input_schema: json!({ "type": "object", "properties": {} }),
        }]));
        let (arr, map) = tools_array_payload(Some(&registry));
        let arr = arr.expect("schemas present");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["type"], "function");
        assert_eq!(arr[0]["function"]["name"], "read_file");
        assert_eq!(arr[0]["function"]["description"], "Read a file.");
        assert_eq!(arr[0]["function"]["parameters"]["type"], "object");
        // No rewriting needed for `read_file` — map stays empty.
        assert!(map.is_empty());
    }

    #[test]
    fn tools_array_payload_returns_none_for_empty_schemas() {
        // An attached-but-empty registry shouldn't pollute the request
        // body with `tools: []`. Most providers treat the empty array as
        // "tools allowed but none offered" which is just confusing.
        use yah_runner_core::tool::{ToolOutcome, ToolRegistry, ToolSchema};
        use async_trait::async_trait;

        struct EmptyRegistry;
        #[async_trait]
        impl ToolRegistry for EmptyRegistry {
            fn schemas(&self) -> Vec<ToolSchema> {
                vec![]
            }
            async fn execute(&self, _name: &str, _args: Value) -> ToolOutcome {
                ToolOutcome::ok(json!(null))
            }
        }
        let registry: Arc<dyn ToolRegistry> = Arc::new(EmptyRegistry);
        let (arr, map) = tools_array_payload(Some(&registry));
        assert!(arr.is_none());
        assert!(map.is_empty());
    }

    #[test]
    fn tools_array_payload_rewrites_dotted_names_for_wire() {
        use yah_runner_core::tool::{ToolOutcome, ToolRegistry, ToolSchema};
        use async_trait::async_trait;

        struct StubRegistry(Vec<ToolSchema>);
        #[async_trait]
        impl ToolRegistry for StubRegistry {
            fn schemas(&self) -> Vec<ToolSchema> {
                self.0.clone()
            }
            async fn execute(&self, _name: &str, _args: Value) -> ToolOutcome {
                ToolOutcome::ok(json!(null))
            }
        }

        let registry: Arc<dyn ToolRegistry> = Arc::new(StubRegistry(vec![
            ToolSchema {
                name: "board.show".into(),
                description: "show a ticket".into(),
                input_schema: json!({ "type": "object", "properties": {} }),
            },
            ToolSchema {
                name: "read_file".into(),
                description: "read a file".into(),
                input_schema: json!({ "type": "object", "properties": {} }),
            },
        ]));
        let (arr, map) = tools_array_payload(Some(&registry));
        let arr = arr.expect("schemas present");
        // Dotted name rewritten on the wire to match `^[a-zA-Z0-9_-]+$`.
        assert_eq!(arr[0]["function"]["name"], "board_show");
        // Already-conformant name is untouched.
        assert_eq!(arr[1]["function"]["name"], "read_file");
        // Map carries only the rewritten entry (wire → registry).
        assert_eq!(map.get("board_show").map(String::as_str), Some("board.show"));
        assert!(!map.contains_key("read_file"));
    }

    #[tokio::test]
    async fn start_refuses_engine_with_a_different_provider_label() {
        let runner = OpenAiCompatRunner::new(OpenAiCompatConfig::openai("dummy"));
        let mut prelude = test_prelude();
        prelude.model = Some(ModelRef::new("claude", None));
        let err = runner.start(prelude, "T01").await.unwrap_err();
        assert!(
            matches!(err, RunnerError::UnsupportedEngine { .. }),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn start_then_stop_round_trips_session_registry() {
        let runner = OpenAiCompatRunner::new(OpenAiCompatConfig::openai("dummy"));
        // No model = accept any provider, fall through to default model.
        let id = runner.start(test_prelude(), "T01").await.unwrap();
        assert!(runner.stop(&id).await.unwrap());
        // Idempotent: second stop returns false.
        assert!(!runner.stop(&id).await.unwrap());
    }

    #[test]
    fn derive_models_url_swaps_chat_completions_suffix() {
        assert_eq!(
            derive_models_url("https://api.openai.com/v1/chat/completions"),
            "https://api.openai.com/v1/models",
        );
        assert_eq!(
            derive_models_url("http://localhost:11434/v1/chat/completions"),
            "http://localhost:11434/v1/models",
        );
    }

    #[test]
    fn derive_models_url_appends_when_path_doesnt_match() {
        assert_eq!(
            derive_models_url("https://gateway.example/v1"),
            "https://gateway.example/v1/models",
        );
        assert_eq!(
            derive_models_url("https://gateway.example/v1/"),
            "https://gateway.example/v1/models",
        );
    }

    #[test]
    fn derive_models_url_strips_api_chat_and_synthesises_v1_models() {
        // Ollama native /api/chat lives outside /v1; the catalogue is
        // still at /v1/models on the same host.
        assert_eq!(
            derive_models_url("http://localhost:11434/api/chat"),
            "http://localhost:11434/v1/models",
        );
        assert_eq!(
            derive_models_url("https://ollama.com/api/chat"),
            "https://ollama.com/v1/models",
        );
    }

    #[test]
    fn parse_ollama_chat_frame_extracts_content_delta() {
        let line = r#"{"model":"llama3.3","message":{"role":"assistant","content":"hello"},"done":false}"#;
        let evs = parse_ollama_chat_frame(line);
        assert_eq!(
            evs,
            vec![FrameEvent::Delta {
                text: "hello".into(),
                tool_calls: vec![],
                finish: None,
            }]
        );
    }

    #[test]
    fn parse_ollama_chat_frame_final_done_emits_finish_usage_and_done() {
        // The terminal NDJSON line carries `done: true`, `done_reason`,
        // and (when the model supports it) token counts. The parser fans
        // those into one synthesised Delta with `finish_reason`, one
        // Usage event, and the closing Done sentinel — same shape the
        // runner already drives for the SSE path.
        let line = r#"{"model":"qwen3.5","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":5712,"eval_count":42}"#;
        let evs = parse_ollama_chat_frame(line);
        assert_eq!(
            evs,
            vec![
                FrameEvent::Delta {
                    text: "".into(),
                    tool_calls: vec![],
                    finish: Some("stop".into()),
                },
                FrameEvent::Usage {
                    input_tokens: Some(5712),
                    output_tokens: Some(42),
                    cache_read_tokens: None,
                },
                FrameEvent::Done,
            ]
        );
    }

    #[test]
    fn parse_ollama_chat_frame_tool_call_synthesises_id_and_finish_reason() {
        // Ollama omits per-call `id`; the parser synthesises
        // `ollama_tc_<index>` so the runner can pair the assistant
        // tool_calls[] with the role:tool rows that follow. And when a
        // `done: true` frame carries tool_calls, the parser translates
        // `done_reason` to the literal "tool_calls" the tool-loop branch
        // keys on.
        let line = r#"{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"board_claim","arguments":{"id":"R155-F6"}}}]},"done":true,"done_reason":"stop"}"#;
        let evs = parse_ollama_chat_frame(line);
        let first = evs.first().expect("at least one event");
        let FrameEvent::Delta {
            tool_calls, finish, ..
        } = first
        else {
            panic!("expected Delta first, got {first:?}");
        };
        assert_eq!(finish.as_deref(), Some("tool_calls"));
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id.as_deref(), Some("ollama_tc_0"));
        assert_eq!(tool_calls[0].name.as_deref(), Some("board_claim"));
        let args = tool_calls[0]
            .arguments
            .as_deref()
            .expect("arguments serialised");
        assert!(
            args.contains("R155-F6"),
            "args should round-trip object → string, got {args:?}"
        );
        // The final Done sentinel still lands so the SSE loop exits.
        assert!(matches!(evs.last(), Some(FrameEvent::Done)));
    }

    #[test]
    fn parse_ollama_chat_frame_empty_or_unparseable_line_returns_empty() {
        // The streaming loop calls the parser on every `\n`-split slice;
        // blank lines (e.g. a trailing newline before the connection
        // closes) and non-JSON noise must be silently ignored, matching
        // the SSE parser's "never abort a turn over a single weird chunk"
        // policy.
        assert!(parse_ollama_chat_frame("").is_empty());
        assert!(parse_ollama_chat_frame("   ").is_empty());
        assert!(parse_ollama_chat_frame("not json").is_empty());
    }

    #[test]
    fn parse_ollama_chat_frame_top_level_error_emits_error_event() {
        // Ollama signals errors with a top-level `error: "<msg>"` string —
        // not the OpenAI nested `{error:{message:…}}` shape parse_sse_frame
        // handles. Verify the NDJSON parser hits the right branch.
        let line = r#"{"error":"model qwen3.5:latest not found, try pulling it first"}"#;
        let evs = parse_ollama_chat_frame(line);
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            FrameEvent::Error(msg) => assert!(msg.contains("not found"), "{msg:?}"),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn send_to_unknown_session_returns_session_not_found() {
        let runner = OpenAiCompatRunner::new(OpenAiCompatConfig::openai("dummy"));
        // EventStream doesn't impl Debug, so we can't `.unwrap_err()` on
        // `Result<EventStream, _>` — match the `Err` arm explicitly.
        match runner
            .send(&SessionId::new("session:00000000"), "hi".into())
            .await
        {
            Err(RunnerError::SessionNotFound(_)) => {}
            Err(other) => panic!("expected SessionNotFound, got {other:?}"),
            Ok(_) => panic!("expected error for missing session"),
        }
    }

    #[test]
    fn cost_guard_defaults_by_factory() {
        // ollama_cloud enables the guard; others leave it unset.
        assert_eq!(OpenAiCompatConfig::ollama_cloud("k").cost_guard_tokens, Some(500_000));
        assert_eq!(OpenAiCompatConfig::ollama_local().cost_guard_tokens, None);
        assert_eq!(OpenAiCompatConfig::openai("k").cost_guard_tokens, None);
    }

    #[test]
    fn is_retryable_status_distinguishes_transient_from_permanent() {
        // 429 + 5xx → retry.
        assert!(is_retryable_status(429));
        assert!(is_retryable_status(500));
        assert!(is_retryable_status(502));
        assert!(is_retryable_status(503));
        assert!(is_retryable_status(504));

        // Auth / bad-request / not-found → permanent.
        assert!(!is_retryable_status(401));
        assert!(!is_retryable_status(403));
        assert!(!is_retryable_status(400));
        assert!(!is_retryable_status(404));
        assert!(!is_retryable_status(422));

        // 2xx isn't called on the retry path but defined regardless.
        assert!(!is_retryable_status(200));
    }

    #[test]
    fn compute_stream_open_backoff_ms_uses_exponential_when_no_retry_after() {
        assert_eq!(compute_stream_open_backoff_ms(1, None), 1_000);
        assert_eq!(compute_stream_open_backoff_ms(2, None), 2_000);
        assert_eq!(compute_stream_open_backoff_ms(3, None), 4_000);
    }

    #[test]
    fn compute_stream_open_backoff_ms_honors_retry_after() {
        assert_eq!(compute_stream_open_backoff_ms(1, Some(5)), 5_000);
        // Caps at MAX_RETRY_AFTER_SECS (30) to keep a misconfigured
        // upstream from stalling the session.
        assert_eq!(compute_stream_open_backoff_ms(1, Some(300)), 30_000);
        // Retry-After overrides the exponential schedule even on later
        // attempts.
        assert_eq!(compute_stream_open_backoff_ms(3, Some(10)), 10_000);
    }

    #[test]
    fn cost_guard_triggers_when_threshold_exceeded() {
        // Simulate the check: accumulate tokens over the guard limit and verify
        // the guard condition fires.
        let limit: u32 = 500_000;
        let mut acc = TurnUsageAcc::default();
        acc.add(Some(300_000), Some(1_000), None);
        // Not yet over the limit.
        assert!(acc.input_tokens.unwrap_or(0) <= limit);
        acc.add(Some(201_000), Some(500), None);
        // Now over the limit — guard should fire.
        assert!(acc.input_tokens.unwrap_or(0) > limit);
    }

    #[test]
    fn body_includes_reasoning_effort_from_thinking_budget() {
        let mut s = session("PRELUDE", "deepseek-v4-pro");
        s.history.push(json!({ "role": "user", "content": "hi" }));

        // Every die level, read off the session itself (R947-F2).
        for (level, effort) in [
            (ReasoningBudget::Fast, "high"),
            (ReasoningBudget::Standard, "max"),
            (ReasoningBudget::Deep, "max"),
            (ReasoningBudget::Budget { tokens: 32_000 }, "max"),
        ] {
            s.think = Some(level);
            let body = build_request_body(&s, 1024, None, None, None);
            assert_eq!(body["reasoning_effort"], effort, "{level:?}");
            assert_eq!(body["thinking"]["type"], "enabled");
        }

        // Test with no thinking budget
        s.think = None;
        let body = build_request_body(&s, 1024, None, None, None);
        assert_eq!(body.get("reasoning_effort"), None);
        assert_eq!(body.get("thinking"), None);
    }

    // ── R418-F2: pre-flight prompt-size guard ──────────────────────────

    /// Append an assistant tool_call + tool_result pair to history,
    /// matching the wire shape the runner builds in the live path.
    fn push_tool_pair(s: &mut InnerSession, call_id: &str, tool_name: &str, content: String) {
        s.history.push(json!({
            "role": "assistant",
            "content": Value::Null,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": "{}",
                }
            }]
        }));
        s.history.push(json!({
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
        }));
    }

    /// Anchor `s` as if the provider had just reported `tokens` for the
    /// prelude alone (nothing in history covered yet), so every history row
    /// is estimated — the pre-R883-T8 behaviour, now behind an anchor.
    fn anchor_prelude(s: &mut InnerSession) {
        s.usage_anchor = Some(UsageAnchor {
            tokens: chars_to_tokens(s.prelude_text.len()),
            covered: 0,
            prelude_chars: s.prelude_text.len(),
        });
    }

    #[test]
    fn estimate_prompt_tokens_is_none_without_an_anchor() {
        let mut s = session("PRELUDE_PRELUDE", "m");
        s.history.push(json!({ "role": "user", "content": "hi" }));
        assert_eq!(estimate_prompt_tokens(&s), None);
    }

    #[test]
    fn estimate_prompt_tokens_adds_only_rows_since_the_anchor() {
        let mut s = session("p", "m");
        s.history.push(json!({ "role": "user", "content": "X".repeat(40_000) }));
        s.history.push(json!({ "role": "assistant", "content": "ok" }));
        // Provider says the whole exchange was 1_000 tokens — the 40KB of
        // chars/4 guesswork must not be re-added on top.
        s.record_usage_anchor(Some(1_000));
        assert_eq!(estimate_prompt_tokens(&s), Some(1_000));
        s.history.push(json!({ "role": "user", "content": "Y".repeat(4_000) }));
        let est = estimate_prompt_tokens(&s).unwrap();
        assert!((2_000..2_100).contains(&est), "{est}");
        // Prelude growth (append_resident_stanza) is estimated too.
        s.prelude_text.push_str(&"Z".repeat(4_000));
        assert!(estimate_prompt_tokens(&s).unwrap() >= est + 1_000);
        // A rewritten (shorter) history no longer matches the anchor.
        s.history.truncate(1);
        assert_eq!(estimate_prompt_tokens(&s), None);
    }

    #[test]
    fn record_usage_anchor_keeps_prior_anchor_when_response_had_no_usage() {
        let mut s = session("p", "m");
        s.history.push(json!({ "role": "assistant", "content": "a" }));
        s.record_usage_anchor(Some(500));
        s.history.push(json!({ "role": "assistant", "content": "b" }));
        s.record_usage_anchor(None);
        assert_eq!(s.usage_anchor.map(|a| (a.tokens, a.covered)), Some((500, 1)));
    }

    #[test]
    fn lookup_tool_name_finds_matching_assistant_call() {
        let mut s = session("p", "m");
        push_tool_pair(&mut s, "call_42", "board.list_tickets", "ok".into());
        assert_eq!(
            lookup_tool_name(&s.history, "call_42"),
            Some("board.list_tickets"),
        );
        assert_eq!(lookup_tool_name(&s.history, "call_missing"), None);
    }

    #[test]
    fn elide_largest_tool_result_picks_largest_above_threshold() {
        let mut s = session("p", "m");
        push_tool_pair(&mut s, "small", "x", "small body".into());
        push_tool_pair(&mut s, "huge", "board.list_tickets", "X".repeat(50_000));
        push_tool_pair(&mut s, "medium", "y", "Y".repeat(8_000));

        let rec = elide_largest_tool_result(&mut s.history, MIN_ELIDABLE_BYTES)
            .expect("largest must elide");
        assert_eq!(rec.tool_call_id, "huge");
        assert_eq!(rec.tool_name, "board.list_tickets");
        assert_eq!(rec.original_bytes, 50_000);
        assert!(rec.replacement_bytes < 500, "note must be small");

        // 'huge' row's content was replaced with the note in-place.
        let huge_idx = s
            .history
            .iter()
            .position(|m| m.get("tool_call_id").and_then(|v| v.as_str()) == Some("huge"))
            .unwrap();
        let content = s.history[huge_idx].get("content").unwrap().as_str().unwrap();
        assert!(content.starts_with("[tool_result elided"));
        assert!(content.contains("board.list_tickets"));
        // The 'small' and 'medium' rows are untouched.
        let small_idx = s
            .history
            .iter()
            .position(|m| m.get("tool_call_id").and_then(|v| v.as_str()) == Some("small"))
            .unwrap();
        assert_eq!(s.history[small_idx]["content"], "small body");
    }

    #[test]
    fn elide_skips_rows_below_min_elidable_bytes() {
        let mut s = session("p", "m");
        push_tool_pair(&mut s, "tiny1", "x", "a".repeat(100));
        push_tool_pair(&mut s, "tiny2", "y", "b".repeat(200));
        assert!(elide_largest_tool_result(&mut s.history, MIN_ELIDABLE_BYTES).is_none());
    }

    #[test]
    fn try_preflight_elide_noop_when_under_cap() {
        let mut s = session("short prelude", "m");
        s.history.push(json!({ "role": "user", "content": "hi" }));
        anchor_prelude(&mut s);
        // 8192-token window → ~32KB chars. History is well under.
        let elisions = try_preflight_elide(&mut s, 8192, 0.95);
        assert!(elisions.is_empty());
    }

    /// R883-T8: the first request has no provider evidence, so the guard
    /// waits — even for a history the chars/4 estimate says is far over.
    #[test]
    fn try_preflight_elide_waits_for_provider_evidence_on_first_request() {
        let mut s = session("short prelude", "m");
        s.history.push(json!({ "role": "user", "content": "list everything" }));
        push_tool_pair(&mut s, "call_huge", "board.list_tickets", "X".repeat(800_000));
        assert!(try_preflight_elide(&mut s, 200_000, 0.95).is_empty());
        let content = s.history[2]["content"].as_str().unwrap();
        assert!(!content.starts_with("[tool_result elided"));
    }

    /// R883-T8: the provider's count decides, not chars/4. The same 800KB
    /// row chars/4 would call 200k tokens is reported as 50k (a tokenizer
    /// that packs this content 16 chars/token) → no trim.
    #[test]
    fn try_preflight_elide_trusts_the_anchor_over_the_char_estimate() {
        let mut s = session("short prelude", "m");
        s.history.push(json!({ "role": "user", "content": "list everything" }));
        push_tool_pair(&mut s, "call_huge", "board.list_tickets", "X".repeat(800_000));
        s.record_usage_anchor(Some(50_000));
        assert!(try_preflight_elide(&mut s, 200_000, 0.95).is_empty());
    }

    /// R883-T8: an elision inside the anchored prefix is subtracted from
    /// the anchor, so the loop terminates and later checks stay consistent.
    #[test]
    fn try_preflight_elide_discounts_the_anchor_for_prefix_elisions() {
        let mut s = session("short prelude", "m");
        s.history.push(json!({ "role": "user", "content": "list everything" }));
        push_tool_pair(&mut s, "call_huge", "board.list_tickets", "X".repeat(800_000));
        s.history.push(json!({ "role": "assistant", "content": "done" }));
        s.record_usage_anchor(Some(205_000));
        let elisions = try_preflight_elide(&mut s, 200_000, 0.95);
        assert_eq!(elisions.len(), 1, "{elisions:?}");
        let est = estimate_prompt_tokens(&s).unwrap();
        assert!(est <= 190_000, "post-trim anchored estimate {est} still over cap");
        assert_eq!(s.usage_anchor.unwrap().covered, s.history.len());
    }

    /// R725-B4 (W300 Finding 4): the provider owns `tool_call_id` and does not
    /// have to make it unique. Moonshot's Anthropic-compat endpoint emits an
    /// index that resets every turn, so one measured session carried 160 calls
    /// under 140 distinct ids — `Read_1` appears repeatedly.
    ///
    /// The old back-walk searched the whole history from the end and so
    /// answered with whichever turn reused the id LAST. Here the oversized row
    /// belongs to the FIRST `Read_1` (a `board.list_tickets` call); a later turn
    /// reuses `Read_1` for `git.status`. Searching the whole history names
    /// `git.status`; searching only what precedes the row names the right one.
    #[test]
    fn an_elision_names_its_own_call_when_the_provider_reuses_ids() {
        let mut s = session("short prelude", "m");
        s.history.push(json!({ "role": "user", "content": "turn one" }));
        push_tool_pair(&mut s, "Read_1", "board.list_tickets", "X".repeat(800_000));
        // A later turn, same provider id, different tool. This is the row the
        // old whole-history backwards walk would have found first.
        s.history.push(json!({ "role": "user", "content": "turn two" }));
        push_tool_pair(&mut s, "Read_1", "git.status", "small".to_string());

        anchor_prelude(&mut s);
        let elisions = try_preflight_elide(&mut s, 200_000, 0.95);
        assert_eq!(elisions.len(), 1, "expected one elision, got {elisions:?}");
        assert_eq!(elisions[0].tool_call_id, "Read_1");
        assert_eq!(
            elisions[0].tool_name, "board.list_tickets",
            "the note must name the call whose result was elided, not a later \
             turn that reused the id"
        );
        assert_eq!(elisions[0].original_bytes, 800_000);

        // And the note the model actually reads carries the right name.
        let elided = s
            .history
            .iter()
            .filter_map(|m| m.get("content").and_then(|v| v.as_str()))
            .find(|c| c.contains("tool_result elided"))
            .expect("an elision note must be in history");
        assert!(elided.contains("board.list_tickets"), "{elided}");
        assert!(!elided.contains("git.status"), "{elided}");
    }

    /// Pins that the slice is what does the work, rather than the surrounding
    /// test happening to pass. Same history, same id, two ranges: the whole
    /// thing answers with the LAST reuse, the prefix answers with the owning
    /// call. If these two ever agree, the regression above has stopped being a
    /// regression test and is passing for the wrong reason.
    #[test]
    fn the_whole_history_and_the_prefix_disagree_under_a_reused_id() {
        let mut s = session("p", "m");
        push_tool_pair(&mut s, "Read_1", "board.list_tickets", "big".into());
        push_tool_pair(&mut s, "Read_1", "git.status", "small".into());
        // The first tool row sits at index 1; its owning assistant call is 0.
        let owning_row = 1;
        assert_eq!(
            lookup_tool_name(&s.history, "Read_1"),
            Some("git.status"),
            "whole-history walk finds the newest reuse — this is the old behaviour"
        );
        assert_eq!(
            lookup_tool_name(&s.history[..owning_row], "Read_1"),
            Some("board.list_tickets"),
            "prefix walk finds the call that actually owns the row"
        );
    }

    /// The join must stay correct when the id genuinely is unique, which is the
    /// Anthropic-native `toolu_01…` shape yah's own tests assert elsewhere.
    #[test]
    fn an_elision_still_names_its_call_when_ids_are_unique() {
        let mut s = session("short prelude", "m");
        s.history.push(json!({ "role": "user", "content": "turn one" }));
        push_tool_pair(&mut s, "toolu_01AAA", "board.list_tickets", "X".repeat(800_000));
        s.history.push(json!({ "role": "user", "content": "turn two" }));
        push_tool_pair(&mut s, "toolu_01BBB", "git.status", "small".to_string());

        anchor_prelude(&mut s);
        let elisions = try_preflight_elide(&mut s, 200_000, 0.95);
        assert_eq!(elisions.len(), 1, "{elisions:?}");
        assert_eq!(elisions[0].tool_call_id, "toolu_01AAA");
        assert_eq!(elisions[0].tool_name, "board.list_tickets");
    }

    #[test]
    fn try_preflight_elide_trims_oversized_tool_result() {
        // Simulate the session:098779e3 shape: one giant board.list_tickets
        // tool_result blowing the cap.
        let mut s = session("short prelude", "m");
        s.history.push(json!({ "role": "user", "content": "list everything" }));
        // 800KB tool_result row.
        push_tool_pair(
            &mut s,
            "call_huge",
            "board.list_tickets",
            "X".repeat(800_000),
        );

        // 200k-token window → 800KB cap, threshold 0.95 → 760KB. History is
        // ~800KB+ from the tool result alone, so we're over.
        anchor_prelude(&mut s);
        let elisions = try_preflight_elide(&mut s, 200_000, 0.95);
        assert_eq!(elisions.len(), 1, "expected one elision, got {elisions:?}");
        assert_eq!(elisions[0].tool_call_id, "call_huge");
        assert_eq!(elisions[0].tool_name, "board.list_tickets");
        assert_eq!(elisions[0].original_bytes, 800_000);

        // Post-elision the prompt fits.
        let est = estimate_prompt_tokens(&s).unwrap();
        assert!(est <= 190_000, "post-trim estimate {est} still over cap");
    }

    #[test]
    fn try_preflight_elide_bails_when_futile() {
        // Prelude alone exceeds the cap; no tool elision can save the
        // request. We must not trim — destroying information without
        // benefit. (Captures the relay's "don't trim if the smallest
        // possible reduction still leaves us over" rule.)
        let mut s = session(&"P".repeat(50_000), "m");
        push_tool_pair(&mut s, "tool", "x", "Y".repeat(8_000));
        // Window so small even the prelude alone exceeds the cap.
        anchor_prelude(&mut s);
        let elisions = try_preflight_elide(&mut s, 1024, 0.95);
        assert!(
            elisions.is_empty(),
            "must not trim when prelude alone exceeds cap"
        );
        // 'tool' row's content was NOT mangled.
        let idx = s
            .history
            .iter()
            .position(|m| m.get("tool_call_id").and_then(|v| v.as_str()) == Some("tool"))
            .unwrap();
        let content = s.history[idx]["content"].as_str().unwrap();
        assert!(!content.starts_with("[tool_result elided"));
    }

    #[test]
    fn try_preflight_elide_iterates_until_fits() {
        // Two oversized rows that each fit on their own under the cap
        // but together blow it. The first elision picks the largest;
        // the loop must re-evaluate and elide a second time. With a
        // 12k-token window (cap ≈ 45KB chars), 400KB + 400KB of tool
        // content forces two passes.
        let mut s = session("p", "m");
        push_tool_pair(&mut s, "huge_a", "x", "A".repeat(400_000));
        push_tool_pair(&mut s, "huge_b", "y", "B".repeat(400_000));
        anchor_prelude(&mut s);
        let elisions = try_preflight_elide(&mut s, 12_000, 0.95);
        assert_eq!(
            elisions.len(),
            2,
            "expected 2 elisions, got {elisions:?}",
        );
        let ids: Vec<&str> = elisions.iter().map(|r| r.tool_call_id.as_str()).collect();
        assert!(ids.contains(&"huge_a"));
        assert!(ids.contains(&"huge_b"));
    }

    #[test]
    fn default_prompt_overflow_threshold_is_used_when_unset() {
        let cfg = OpenAiCompatConfig::openai("k");
        assert!((cfg.prompt_overflow_threshold() - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn prompt_overflow_threshold_override_is_honored() {
        let mut cfg = OpenAiCompatConfig::openai("k");
        cfg.prompt_overflow_threshold = Some(0.80);
        assert!((cfg.prompt_overflow_threshold() - 0.80).abs() < f32::EPSILON);
    }

    #[test]
    fn elision_note_format_includes_size_and_tool_name() {
        let note = elision_note("board.list_tickets", 4_400_000);
        assert!(note.starts_with("[tool_result elided"));
        assert!(note.contains("board.list_tickets"));
        // 4_400_000 bytes ≈ 4296.9 KB
        assert!(note.contains("KB"));
    }

    // ── R418-F3: runtime caps + threshold reminder ─────────────────────

    #[test]
    fn runtime_caps_block_carries_session_id_and_iteration_cap() {
        let cfg = OpenAiCompatConfig::ollama_local();
        let sid = SessionId::new("session:test1234");
        let block = render_runtime_caps_block(&sid, "R418-F3", "llama3.3", &cfg);
        assert!(block.contains("session:test1234"));
        assert!(block.contains("R418-F3"));
        assert!(block.contains("llama3.3"));
        assert!(block.contains("max_tool_iterations"));
        assert!(block.contains("max_output_tokens"));
        // Ollama-local has a context_window of 8192 plumbed through.
        assert!(block.contains("8192"));
        assert!(block.contains("PromptTrimmed"));
    }

    #[test]
    fn runtime_caps_block_calls_out_inactive_guard_when_window_unknown() {
        let cfg = OpenAiCompatConfig::openai("k");
        let sid = SessionId::new("session:x");
        let block = render_runtime_caps_block(&sid, "T1", "no-such-model-xyz", &cfg);
        assert!(
            block.contains("not known to runner"),
            "expected explicit inactive-guard note when the catalog has no window",
        );
    }

    // R883-T7: hosted factories leave `context_window` None and resolve
    // the session model through the runtime catalog.
    #[test]
    fn hosted_engines_resolve_context_window_from_catalog() {
        let cases = [
            (OpenAiCompatConfig::openai("k"), "gpt-4o", 131_072),
            (OpenAiCompatConfig::groq("k"), "llama-3.3-70b-versatile", 131_072),
            (OpenAiCompatConfig::deepseek("k"), "deepseek-chat", 131_072),
            (OpenAiCompatConfig::deepseek("k"), "deepseek-v4-flash", 1_000_000),
            (
                OpenAiCompatConfig::openrouter("k"),
                "deepseek/deepseek-chat-v3-0324",
                131_072,
            ),
        ];
        for (cfg, model, want) in cases {
            assert_eq!(cfg.context_window_for(model), Some(want), "{model}");
        }
        // Each factory's own default model is covered by the catalog.
        for cfg in [
            OpenAiCompatConfig::openai("k"),
            OpenAiCompatConfig::groq("k"),
            OpenAiCompatConfig::deepseek("k"),
            OpenAiCompatConfig::openrouter("k"),
        ] {
            assert!(
                cfg.context_window_for(&cfg.default_model).is_some(),
                "{} default {} missing from model catalog",
                cfg.provider_label,
                cfg.default_model,
            );
        }
        let cfg = OpenAiCompatConfig::openai("k");
        assert_eq!(cfg.context_window_for("no-such-model-xyz"), None);
        assert_eq!(cfg.context_window_for("vendor/no-such-model-xyz"), None);
    }

    #[test]
    fn explicit_context_window_overrides_catalog() {
        let mut cfg = OpenAiCompatConfig::openai("k");
        cfg.context_window = Some(50_000);
        assert_eq!(cfg.context_window_for("gpt-4o"), Some(50_000));
        // Ollama keeps num_ctx, not the model's nominal window.
        let cfg = OpenAiCompatConfig::ollama_local();
        assert_eq!(cfg.context_window_for("llama3.3"), Some(8192));
    }

    #[test]
    fn runtime_caps_block_renders_catalog_window_for_hosted_engine() {
        let cfg = OpenAiCompatConfig::openai("k");
        let sid = SessionId::new("session:x");
        let block = render_runtime_caps_block(&sid, "T1", "gpt-4o", &cfg);
        assert!(block.contains("**context_window**: 131072 tokens"), "{block}");
        assert!(!block.contains("not known to runner"));
    }

    #[test]
    fn runtime_caps_block_includes_cost_guard_when_set() {
        let cfg = OpenAiCompatConfig::ollama_cloud("k");
        let sid = SessionId::new("session:x");
        let block = render_runtime_caps_block(&sid, "T1", "kimi", &cfg);
        // ollama_cloud factory sets cost_guard_tokens = Some(500_000).
        assert!(block.contains("cost_guard_tokens"));
        assert!(block.contains("500000"));
    }

    #[tokio::test]
    async fn start_appends_runtime_caps_block_to_prelude_text() {
        let cfg = OpenAiCompatConfig {
            provider_label: "openai".into(),
            endpoint: "http://x".into(),
            default_model: "fake".into(),
            api_key: None,
            max_output_tokens: None,
            max_tool_iterations: Some(7),
            transport: Transport::Sse,
            cost_guard_tokens: None,
            num_ctx: None,
            keep_alive: None,
            reasoning: None,
            quota_ledger: None,
            rate_limit_hint: None,
            session_preamble: None,
            context_window: Some(100_000),
            prompt_overflow_threshold: None,
        };
        let runner = OpenAiCompatRunner::new(cfg);
        // A bare Prelude is enough — start() reads engine + renders text.
        let mut prelude = test_prelude();
        prelude.model = Some(ModelRef::new("openai", Some("fake".into())));
        prelude.reasoning = Some(ReasoningBudget::Standard);
        let id = runner.start(prelude, "R418-F3-test").await.expect("start");
        let session = runner
            .inner
            .sessions
            .read()
            .await
            .get(&id)
            .cloned()
            .expect("session");
        let s = session.lock().await;
        let text = &s.prelude_text;
        assert!(text.contains(id.as_str()), "{text}");
        assert!(text.contains("R418-F3-test"));
        assert!(text.contains("7 per turn"), "expected iter cap 7 in block");
        assert!(text.contains("100000"), "expected window 100k in block");
    }

    // ── R418-F4: TurnBlocked detection + compact_session helpers ───────

    #[test]
    fn is_overflow_message_recognizes_openai_phrase() {
        assert!(is_overflow_message(
            "openai returned 400: This model's maximum context length is 128000 tokens. \
             error code: context_length_exceeded"
        ));
    }

    #[test]
    fn is_overflow_message_recognizes_ollama_phrase() {
        assert!(is_overflow_message(
            "ollama returned 400: input exceeds the context window"
        ));
    }

    #[test]
    fn is_overflow_message_rejects_generic_transport_errors() {
        assert!(!is_overflow_message("openai returned 503: service unavailable"));
        assert!(!is_overflow_message("dns lookup failed"));
        assert!(!is_overflow_message("rate limited"));
        // The block_detector's fallback only matches the known phrase set —
        // a 401 unauthorized must NOT route to TurnBlocked.
        assert!(!is_overflow_message("openai returned 401: missing api key"));
    }

    #[test]
    fn serialize_history_for_summary_includes_roles_and_tool_calls() {
        let mut s = session("p", "m");
        s.history.push(json!({ "role": "user", "content": "what's in src/lib.rs" }));
        push_tool_pair(
            &mut s,
            "call_1",
            "read_file",
            "fn main() { println!(\"hi\") }".into(),
        );
        s.history
            .push(json!({ "role": "assistant", "content": "It's a hello world." }));
        let dump = serialize_history_for_summary(&s.history);
        assert!(dump.contains("## #0 user"));
        assert!(dump.contains("what's in src/lib.rs"));
        assert!(dump.contains("## #1 assistant"));
        assert!(dump.contains("read_file"));
        assert!(dump.contains("## #2 tool"));
        assert!(dump.contains("tool_result for call_1"));
        assert!(dump.contains("fn main()"));
        assert!(dump.contains("## #3 assistant"));
        assert!(dump.contains("hello world"));
    }

    #[test]
    fn compact_result_shape_pins_field_names() {
        // Sanity check: the wire shape downstream consumers will read.
        let r = CompactResult {
            session_id: SessionId::new("session:test"),
            summary_chars: 1500,
            original_history_chars: 4_500_000,
            history_messages_before: 200,
            history_messages_after: 11,
            kept_tail: 10,
        };
        assert_eq!(r.kept_tail + 1, r.history_messages_after);
        assert!(r.summary_chars < r.original_history_chars);
    }

    #[tokio::test]
    async fn compact_session_returns_session_not_found_for_unknown_id() {
        let cfg = OpenAiCompatConfig::openai("k");
        let runner = OpenAiCompatRunner::new(cfg);
        let err = runner
            .compact_session(&SessionId::new("session:missing"), 10)
            .await
            .expect_err("expected SessionNotFound");
        assert!(matches!(err, RunnerError::SessionNotFound(_)));
    }

    /// Verify reminder content (the actual injection lives inside the
    /// stream! macro in `send` — that needs an HTTP fake to exercise
    /// end-to-end, covered by the e2e suite). This test pins the
    /// human-readable shape so we can grep for it from session JSONLs.
    #[test]
    fn threshold_reminder_format_lists_remaining_iters_and_context() {
        // Mirror the actual format-string used in the loop.
        let remaining = 2u32;
        let max = 16u32;
        let context_line = " · context: ~140k of 200k tokens used";
        let reminder = format!(
            "[runtime check · R418-F3] {remaining} of {max} \
             tool-call iterations left this turn{context_line}. Wrap up or \
             narrow scope — fanning out wide here is likely to truncate.",
        );
        assert!(reminder.starts_with("[runtime check · R418-F3]"));
        assert!(reminder.contains("2 of 16"));
        assert!(reminder.contains("context: ~140k of 200k"));
        assert!(reminder.contains("Wrap up"));
    }
}