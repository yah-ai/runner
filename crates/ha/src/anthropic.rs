//! Anthropic-native HTTP cell — the `anthropic` (HAk) and `crab` (HAo)
//! presets in the runtime matrix. Hand-rolled `/v1/messages` SSE folding so
//! we can hold Anthropic's native `tool_use` / `tool_result` content-block
//! protocol and `cache_control: ephemeral` markers without flattening into
//! OpenAI-shaped function-calling.
//!
//! [`run_anthropic_turn`] drives one turn over a core [`HttpClient`]. Every
//! piece of state it reads or writes beyond the wire — the session's model,
//! history, rendered system prompt, phase, and whatever the host wants to
//! observe — goes through the [`HaHost`] hook, so this crate holds no host
//! state. [`crate::AnthropicRunner`] is the [`yah_runner_core::Runner`] built
//! on top of it with a host that keeps its sessions in memory.
//!
//! Lifted out of `app/yah/desktop/src/agent.rs` in R032-T4, and out of
//! `crates/yah/runner` into this crate in R950-F7.
//!
//! archived: R108 — history: `yah board history oss/runner/crates/ha/src/anthropic.rs`
//!
//! archived: R113 — history: `yah board history oss/runner/crates/ha/src/anthropic.rs`
//!
//! @yah:ticket(R237-F11, "F11: Anthropic-native SSE retry/backoff (R237-F6 sub-work)")
//! @yah:assignee(agent:claude)
//! @yah:at(2026-05-20T06:40:12Z)
//! @yah:status(review)
//! @yah:parent(R237)
//! @yah:next("Wrap run_anthropic_turn fetch+stream in a retry loop (3 attempts, exponential backoff 1s/2s/4s).")
//! @yah:next("Catch reqwest::Error / io::Error on bytes_stream().next() and re-issue request with full conversation context (Anthropic SSE has no event-id; cannot resume from frame N).")
//! @yah:next("Only retry transient errors (network / 502 / 503 / 504); classify via health::classify_message; pass-through Permanent (auth, 400).")
//! @yah:next("Skip retry if any text already streamed to FE (avoid duplicate output) — emit Error instead so the user can re-run intentionally.")
//! @yah:next("Acceptance test: kill upstream socket mid-stream (or mock 503 once) and confirm the retry succeeds; unit test with reqwest mock in anthropic.rs::tests.")
//! @arch:see(.yah/docs/working/W085-fe-to-be-state-promotion.md)
//! @yah:handoff("Pre-stream retry loop landed in anthropic.rs::run_anthropic_turn. Constants: MAX_STREAM_OPEN_ATTEMPTS=3, MAX_RETRY_AFTER_SECS=30. Helpers: is_retryable_status (429+5xx → true; 401/403/4xx → false), compute_stream_open_backoff_ms (Retry-After honored up to 30s; otherwise exp 1s/2s/4s). Loop wraps both the reqwest.send failure path and the non-success-status path; success breaks with the Response, retryable failures continue, permanent failures return Err. Cancellation token checked during backoff sleep via tokio::select!.")
//! @yah:handoff("Scope: pre-stream only. Mid-stream socket drops + chunked-encoding failures still emit TurnFailed as before — once any bytes arrive, partial output may have streamed to the FE and re-issuing would double-stream. Anthropic SSE has no event-id, so resume-from-frame is not possible at the protocol layer.")
//! @yah:handoff("Boundary: TurnStarted is emitted ONCE before the retry loop, so the FE sees Phase=Working immediately and stays there across silent retries. A user-visible second connection delay is the only externally-observable difference from today.")
//! @yah:handoff("Tests added (3, all green): is_retryable_status_distinguishes_transient_from_permanent (429/5xx/4xx/auth coverage); compute_stream_open_backoff_ms_uses_exponential_when_no_retry_after (1s/2s/4s); compute_stream_open_backoff_ms_honors_retry_after (5s passes; 300s clamps to 30s; 10s overrides exp schedule on attempt 3). All 16 anthropic-module tests green; full runner --lib 187/187.")
//! @yah:verify("cargo check -p runner -p desktop -p yah  # workspace-relevant crates clean")
//! @yah:verify("cargo test -p runner --lib anthropic::  # the new helpers + 13 existing anthropic tests, 16/16 green")
//! @yah:verify("cargo test -p runner --lib  # 187/187 — no runner-suite regressions")
//! @yah:verify("Manual: kill upstream connectivity briefly mid-handshake (e.g. block api.anthropic.com via Little Snitch for ~2s while sending a turn), confirm the turn succeeds on retry without a visible failure to the user. (Cannot fully automate without the e2e test in cleanup.)")
//! @yah:cleanup("Add end-to-end integration test using the openai_compat_e2e.rs TCP-server pattern: spawn fake server that returns 503 once then success, run a real run_anthropic_turn against it, assert TurnStarted → MessageDelta → TurnEnded with no TurnFailed or Err. ~150 LOC of test scaffolding (fixture session, in-process sink+store, response selector). Pure-helper tests cover retry decisions; e2e would catch wiring regressions.")
//!
//! @yah:ticket(R258-F3, "Split WireSessionPhase::Working into ToolUse + Emitting (BE wire change)")
//! @yah:assignee(agent:claude)
//! @yah:at(2026-05-20T21:20:49Z)
//! @yah:status(review)
//! @yah:parent(R258)
//! @yah:next("Add ToolUse + Emitting variants to WireSessionPhase (kg crate)")
//! @yah:next("Migrate runner's tool_call_started / assistant_text_started arms to emit the appropriate new variant")
//! @yah:next("ts-rs binding regen; FE deriver updates to read the split variants")
//! @yah:verify("cargo test -p kg --lib")
//! @yah:verify("cd packages/yah/ui && bun run typecheck")
//! @arch:see(.yah/docs/working/W097-session-life-state.md)
//! @arch:see(.yah/docs/architecture/A047-yah-runner.md)
//! @yah:handoff("Added ToolUse + Emitting variants to SessionPhase in kg/src/agent.rs (is_busy includes both; Working kept for backward-compat with serialized JSONL). sessions.rs: made update_phase_claude pub(crate); changed both initial Working emits to Emitting; added per-event ToolUse/Emitting transitions in the runner stream loop on ToolCall/AssistantText events. anthropic.rs: imports update_phase_claude + SessionPhase; emits Emitting at content_block_start type=text and ToolUse at type=tool_use. env/types.ts: added tool_use + emitting to WireSessionPhase + updated isPhaseBusy. partyTypes.ts: updated isSlotBusy to include tool_use + emitting. sessionStore.ts: deriveLifecycle handles tool_use/emitting→working; deriveSessionLife handles tool_use + emitting natively, working stays as emitting fallback. cargo test -p kg --lib 271/271, cargo test -p runner --lib 213/213, bun typecheck clean, 43 sessionStore tests green.")
//! @yah:verify("cargo test -p kg --lib")
//! @yah:verify("cargo test -p runner --lib")
//! @yah:verify("cd packages/yah/ui && bun run typecheck && bun test src/components/agent/sessionStore.test.ts")
//! @yah:verify("Manual: start a session and observe the Character row indicator — 'reading foo.rs' (tool_use) vs typing dots (emitting) — once the UI wires Character row to life (future F6)")
//!
//! @yah:ticket(R562-T3, "Direct: rolling transcript cache breakpoint + consolidate system breakpoints for leader jobs")
//! @yah:status(review)
//! @yah:at(2026-06-28T21:53:59Z)
//! @yah:assignee(agent:bundle-anthropic-ashguard)
//! @yah:phase(P2)
//! @yah:parent(R562)
//! @yah:depends_on(R562-T1)
//! @yah:gotcha("build_anthropic_body (anthropic.rs:~369) passes messages through with NO cache_control breakpoint → 100-350k transcript reprocessed at full price EVERY turn. Breakpoint budget = 4 max, all 4 spent on system blocks today; consolidate to (system-prefix tail)+(rolling transcript tail) for leader jobs. 20-block lookback: a turn ingesting >20 tool_use/tool_result blocks can push the prior cache point out of window → silent miss; move the rolling breakpoint forward each turn, accept fan-out-heavy turns re-warm.")
//! @yah:handoff("P2 Direct transcript caching shipped. build_anthropic_body (anthropic.rs) now, for leader jobs, emits the FINAL message as a content-block array carrying a cache_control breakpoint (rolling: moves forward each turn so turn N+1 reads the [system+prior-transcript] prefix from cache). compose_system_blocks gained a `consolidate` param: leader jobs collapse the system prompt to ONE breakpoint on the last block (whole prefix still caches), freeing budget for the transcript breakpoint -> total 2 of the 4-breakpoint budget. One-shot jobs unchanged (per-block system breakpoints, plain-string messages, 5m).")
//! @yah:handoff("Shared cache_control_value(ttl) helper extracted so system + transcript breakpoints emit identical TTL. T1's build_body test updated for the consolidated layout (leader system prompt = 1 breakpoint @1h, not per-block).")
//! @yah:handoff("Tests: compose::consolidate_puts_one_breakpoint_on_the_final_block + anthropic::build_body_adds_rolling_transcript_breakpoint_for_leader_job + build_body_leaves_transcript_uncached_for_short_jobs, all green (runner lib 363 pass; the 2 reds are pre-existing unrelated: model_context_tokens_1m_models + openai resolver credential test).")
//! @yah:verify("cargo test -p runner --lib compose::")
//! @yah:verify("cargo test -p runner --lib anthropic::tests::build_body")
//! @yah:verify("Live (manual, out of unit scope): 2nd-turn cache_read on a leader Direct session should cover the full transcript span, not just the system prefix.")
//!
//! @yah:ticket(R833-F13, "Anthropic-HTTP wire has no tools: send tools[] and run the tool_use/tool_result loop so a direct-API agent can act")
//! @yah:status(review)
//! @yah:at(2026-08-30T02:37:17Z)
//! @yah:assignee(agent:bundle-anthropic-ashguard)
//! @yah:parent(R833)
//! @yah:gotcha("build_anthropic_body at anthropic.rs line 459 emits no tools key in the request body")
//! @yah:gotcha("Stream loop at anthropic.rs 854-866 recognizes a tool_use content_block only to flip SessionPhase::ToolUse for the UI indicator (R258-F3 scope). It never executes the tool and never sends tool_result back. Recognition is not execution.")
//! @yah:gotcha("Module doc at anthropic.rs line 8 advertising the native tool_use/tool_result protocol describes the PARSER, not the caller.")
//! @yah:gotcha("openai_compat.rs and codex_oauth.rs both DO set a tools field on their bodies -- read both as reference implementations; converge on their shape.")
//! @yah:gotcha("crates/yah/runner/src/tool.rs already defines the provider-agnostic ToolRegistry trait (list schemas by name, execute by name, return JSON). The registry exists and is already provider-agnostic; this ticket wires the anthropic path into it and must not invent a new abstraction.")
//! @yah:gotcha("Consequence: a direct-API anthropic session can converse but cannot Read/Edit/Bash or call board.*/party.*. Breaks the anthropic engine for EVERY yah agent, not just remote-camp work.")
//! @yah:next("Serialize the ToolRegistry schemas into a tools[] array on the anthropic request body.")
//! @yah:next("Execute tool_use blocks through ToolRegistry and send tool_result blocks back on the next turn.")
//! @yah:next("Thread the registry to the anthropic session the way start_runner_session already threads tool_registry (app/yah/desktop/src/agent.rs:1942). ClaudeSessionInit currently carries no registry -- that is the plumbing gap.")
//! @yah:next("Mind the cache invariant: anthropic.rs carries a live gotcha about the 4-breakpoint cache_control budget and a 20-block lookback. A turn ingesting more than 20 tool_use/tool_result blocks can push the prior cache point out of window and silently miss.")
//! @yah:handoff("LANDED in the working tree, NOT COMMITTED -- the commit was denied at the approval gate. Files: crates/yah/runner/src/anthropic.rs, crates/yah/runner/src/sessions.rs, crates/yah/runner/src/slot_sim.rs, plus a one-line tool_registry: None in do_resume_claude_session (app/yah/desktop/src/agent.rs).")
//! @yah:handoff("Scope 1 (tools[]): new anthropic_tools_payload serializes registry schemas into Anthropic's FLAT {name, description, input_schema} shape -- not OpenAI's {type: function, function: {..., parameters}} envelope. Reuses the shared sanitize_tool_name_for_wire / build_wire_name_map pair from tool.rs, so board.show goes out as board_show and maps back before execute. Returns None for a missing OR empty registry, so tools: [] never pollutes a body.")
//! @yah:handoff("Scope 2 (the loop): run_anthropic_turn is now an outer loop over provider requests. tool_use blocks are reassembled from content_block_start (id, name) plus streamed input_json_delta.partial_json fragments, parsed at content_block_stop, dispatched through ToolRegistry::execute, and returned as tool_result blocks on the next request. Cap MAX_TOOL_ITERATIONS = 24, closing out with a synthetic assistant notice and stop_reason exploration_budget -- a policy stop, not an error, mirroring openai_compat.")
//! @yah:handoff("Scope 3 (plumbing): AgentSession and ClaudeSessionInit each gained pub tool_registry: Option<Arc<dyn crate::tool::ToolRegistry>>; register_claude_session moves it straight across. Coordinated live with @Ashguard:dragon on R833-F11, who owns the param on start_claude_session (now hoisted into crates/yah/runner/src/launch.rs) and the desktop threading. Their cargo check -p yah -p desktop is green against this field.")
//! @yah:handoff("CACHE INVARIANT, as required by the ticket. The R562-T3 rolling transcript breakpoint stays pinned to the last message of session.history; tool turns append BEHIND it via the new build_anthropic_body_with_tool_turns. Two consequences, both deliberate. (a) Breakpoint COUNT is unchanged -- 2 for leader jobs, <=3 otherwise -- no matter how many tool iterations run, so the loop cannot overrun the 4-breakpoint budget. Marking each tool_result would have blown it on turn one. (b) Every iteration within one turn re-issues the same cached prefix, so iteration N+1 reads it from cache and only the accumulated tool blocks reprocess. A turn pushing more than 20 blocks past the breakpoint still falls out of the 20-block lookback and re-warms -- the known accepted cost, now reachable more often because tool loops exist at all. Test tool_turns_append_behind_the_rolling_cache_breakpoint asserts the count is identical with and without tool turns.")
//! @yah:handoff("ONE MORE CACHE NOTE not in the original gotcha: Anthropic's cacheable prefix is ordered tools -> system -> messages, so introducing the tools key shifts the prefix and costs exactly one cache miss the first time a given session sends it. Stable per session thereafter (schemas are a static manifest), so nothing to amortise past that first turn.")
//! @yah:handoff("USAGE FIX found on the way: merge_anthropic_usage overwrites (later value wins), which is right within one request and wrong across several. Each iteration now merges into a fresh iter_usage, derives that request's peak, and folds it in via TurnUsage::accumulate -- token counts SUM, peak_context_tokens MAXes. Without this a 4-request turn would have reported the sum of four prompts as if one prompt were that large. Single-request turns get byte-identical numbers to before.")
//! @yah:handoff("ALSO FIXED in passing: pre-stream failures (cancel during backoff, transport exhausted, permanent HTTP status) returned Err unconditionally, which the caller maps to AgentEvent::Error. On a tool-loop re-issue text and tool results have already reached the FE, so an Err there would leave the renderer with a turn that started, emitted and never terminated. New bail_pre_stream! returns Err only on the first request and closes out as TurnFailed afterwards. ApiTurnError also gained #[derive(Debug)].")
//! @yah:handoff("TESTS: cargo test -p yah-runner --lib = 488 passed, 0 failed. Pre-change baseline measured on the same tree = 477 passed, 0 failed. +11, no regressions. NOTE the package is yah-runner, not runner -- the ticket's -p runner errors out with 'package ID specification runner did not match any packages'.")
//! @yah:handoff("The 11: 9 unit tests (tools key absent without a registry; flat Anthropic schema shape with no OpenAI envelope; dotted-name rewrite both directions; empty-registry suppression; cache-breakpoint invariance; three PendingToolUse argument-parsing cases including fragmented partial_json, empty buffer as {} not a failure, and unparseable as Null) plus 2 e2e tests against a fake /v1/messages TCP server, following the tests/openai_compat_e2e.rs pattern -- real HTTP framing, real reqwest bytes_stream, real chunk boundaries.")
//! @yah:handoff("The main e2e (tool_use_round_trips_through_the_registry_and_back_onto_the_wire) captures BOTH request bodies and asserts: request 1 advertises board_show with an input_schema; the registry saw board.show with reassembled args {id: R833}; request 2 replays the assistant tool_use block verbatim AND the matching tool_result paired by tool_use_id, and still advertises tools; exactly one TurnStarted and one TurnEnded for the whole loop; usage sums to 300 input / 50 output with peak 200 (the max, not the sum). The second e2e pins the pre-F13 path: a registry-less session makes exactly ONE request with no tools key.")
//! @yah:handoff("ACCEPTANCE, stated precisely. The wire path is proven end to end in-process by the e2e above. I did NOT drive a live API session against api.anthropic.com -- that needs the desktop host and real credentials. @Ashguard:dragon confirmed the dispatch path supplies a real registry (subagent_spawn in app/yah/cli/src/camp.rs builds registry_concrete unconditionally and passes Some(...) through LaunchCharacterParams), so a DISPATCHED anthropic session is wired. Chat/picker sessions are NOT -- see R833-F15.")
//! @yah:handoff("FILED R833-F15 (parent R833): desktop chat/picker anthropic sessions still pass tool_registry: None, because start_runner_session does not RECEIVE a registry -- it builds one inline (~140 lines: tool_ctx, LayeredApprovalStore, FormApprovalRouter, then the base pipeline). Extracting build_session_tool_registry is the fix. Not done here on purpose: those 140 lines are inside app/yah/desktop/src/agent.rs, which @Ashguard:dragon was rewriting wholesale for the R833-F11 hoist during this ticket -- a merge fight for zero acceptance benefit. F15 carries the full diagnosis, the two call sites, and a gotcha to pick it up after F11 lands.")
//! @yah:handoff("R833-B14 (the cross-turn gap) CONFIRMED and narrowed, answering the leader's question: reconstruct_claude_history at sessions.rs:2402 explicitly filters to UserMessage and TurnEnded only -- its own doc says it skips tool call events. So resume does NOT replay ToolCall/ToolResult into history and does NOT narrow the gap. An agent cannot see its own prior tool calls or results across turns, on a fresh session or a resumed one.")
//! @yah:gotcha("NOT COMMITTED: the pathspec-scoped git commit for crates/yah/runner/src/{anthropic,sessions,slot_sim}.rs was DENIED at the approval gate. The change is live in the working tree and green; it needs an operator-approved commit. Deliberately excluded from that pathspec was app/yah/desktop/src/agent.rs, which carries 634 lines of @Ashguard:dragon's in-flight R833-F11 hoist alongside my one-line tool_registry: None -- committing it would have swept their WIP.")
//! @yah:gotcha("Heads-up for whoever does commit sessions.rs: its diff also contains the @yah:ticket(R833-B14, ...) annotation block filed by @Glimmerstone:griffin. That is not my hunk; it shares the file.")
//! @yah:verify("cargo test -p yah-runner --lib  # 488 passed, 0 failed (baseline 477)")
//! @yah:verify("cargo test -p yah-runner --lib anthropic::tool_use_round_trips  # the two-request e2e against a fake /v1/messages server")
//! @yah:verify("cargo check -p desktop --no-default-features  # clean, warnings only")
//! @yah:verify("Live: dispatch an agent onto an anthropic / anthropic-gc engine, have it Read a file and call one board.* tool. Chat-started sessions will NOT work until R833-F15.")
//! @yah:gotcha("COMMIT DENIED TWICE at the approval gate (pathspec-scoped, crates/yah/runner/src/{sessions,anthropic,slot_sim}.rs). Not retried a third time. Handed to @Ashguard:dragon on R833-F11 to carry inside their commit, since our changes are compile-coupled one way -- their launch.rs sets the tool_registry field that only exists in my sessions.rs, so a solo commit from either side leaves HEAD broken. HEAD was 8675e1a0 at handoff.")
//! @yah:gotcha("IF YOU CARRY THIS COMMIT: slot_sim.rs must be in the pathspec. World::build_session (~slot_sim.rs:464) constructs an AgentSession literal, so omitting it fails the yah-runner test build on a missing required field. It is easy to miss -- it appears in neither ticket narrative.")
//! @yah:gotcha("crates/yah/runner/src/lib.rs contains NOTHING of R833-F13 -- verified, zero occurrences of tool_registry. Its only change is @Ashguard:dragon's pub mod launch;.")
//! @yah:verify("Re-measured immediately before the (denied) commit attempt, on the exact tree handed to R833-F11: cargo test -p yah-runner --lib = 488 passed, 0 failed. Baseline on the same tree pre-change = 477 passed, 0 failed.")
//! @yah:verify("UNRELATED PRE-EXISTING RED, do not chase from this ticket: cargo test -p desktop --lib fails commands::audit_ledger_tests::every_loadout_row_the_ledger_can_report_has_a_ceiling_and_fits_it (relay 20262 B over a 20100 B ceiling). That measures rendered trait-loadout bytes against party::prelude::job_resident_ceiling_bytes; crates/yah/party/preroll/traits/shared-tree.json was rewritten 2026-08-30. Neither R833-F13 nor R833-F11 adds a byte of trait prose. Found and filed separately by @Ashguard:dragon.")
//! @yah:gotcha("COMMIT DENIED A THIRD TIME, on @Ashguard:dragon's attempt at the combined eight-file pathspec. Three denials across two sessions on the same action. HEAD is still 8675e1a0; there is no SHA for this ticket. Both halves are live and verified in the working tree only. The exact eight-file pathspec is recorded on R833-F11's gotchas for whoever gets the gate opened.")
//! @yah:gotcha("The audit-ledger red flagged under verify is filed as R660-B20 (parent R660, annotation on crates/yah/party/src/prelude.rs). It carries the measurement, the R660-B18 comparison, and an explicit note that neither R833-F13 nor R833-F11 causes it. Do not chase it from this ticket.")
//! @yah:gotcha("TRANSPORT LIMIT that caps this ticket's value, found by @Ashguard:dragon during R833-F11 acceptance and worth reading before anyone concludes the anthropic engine is fixed everywhere. This tool loop is verified over `yah camp --serve --socket`. It is NOT reachable over `yah camp --stdio`, which routes to serve::run_stdio (app/yah/cli/src/serve.rs:81) and builds a bare CampService with a reduced dispatch table -- no agent./party./subagent. arm, and grep -c DaemonState on that file returns 0. subagent.spawn against it returns -32601 Method not found, reproduced. Since a Remote camp attaches by running exactly `yah camp --stdio --workspace <path>` (crates/yah/rpc-ssh/src/client.rs:100), a REMOTE camp still cannot reach this tool loop. Full detail on R833-F11's gotchas; deliberately not ticketed there because which fix is right is an operator architecture call.")
//! @yah:gotcha("FASTEST LIVE ACCEPTANCE PROBE if someone wants a real tool round trip on the headless road, from @Ashguard:dragon's F11 run: camp-local subclass with runner yah + wire anthropic_wire + connectionId anthropic + auth key, spawned by `subclass` so subagent_spawn's registry_concrete rides along, driven over a real --serve daemon. Their probe reached turn_blocked{api_auth} with a deliberately fake key; a real key should carry it through the loop.")
//!
//!
//! @yah:ticket(R883-T1, "Cap tool output at ingestion on direct-HTTP + codex paths")
//! @yah:status(review)
//! @yah:at(2026-09-15T01:27:15Z)
//! @yah:assignee(agent:bundle-anthropic-ashguard)
//! @yah:parent(R883)
//! @yah:next("Scope as shipped (W342 open item (c)): cap every serialized tool result where it enters the transcript on the three yah-native wires — anthropic.rs, codex_oauth.rs, openai_compat.rs (live dispatch + reconstruct_openai_messages replay). The filing-time bullets claiming 'claude-cli truncates natively' and 'do not build mid-session pruning' were disproved by the 2026-09-09 diligence (see W342 lever 4/6) and are retired.")
//! @yah:handoff("Shipped runner::tool::tool_result_wire_content + TOOL_RESULT_CAP_BYTES (32 KB) in crates/yah/runner/src/tool.rs. Caps each serialized tool result at ingestion: elides the middle of the largest string leaves (JSON stays valid, sibling fields intact, head 3/4 + tail 1/4 kept, marker names omitted lines/bytes), measured on serialized length so escape-heavy text is handled; falls back to cutting the serialized text when no leaf >=1 KB dominates. Pure function, so replay is byte-identical to live.")
//! @yah:handoff("Wired at all four ingestion sites: anthropic.rs tool_result block (~:1318), codex_oauth.rs function_call_output (~:618), openai_compat.rs live tool message (~:1827) AND reconstruct_openai_messages replay (~:2458). AgentEvent::ToolResult still carries the full result (log/renderer are not the context window).")
//! @yah:handoff("Measured first, as asked: all three wires put serde_json::to_string(&outcome.result) on the wire verbatim; registry tools bound themselves at read_file 256 KB, web_fetch 256 KB, yah-tool output 64 KB. openai_compat's pre-flight elision is a separate overflow guard and stays.")
//! @yah:handoff("Discovered + fixed: stale comment at crates/yah/agent-tools/src/tools.rs:2198 claimed the Anthropic loop calls execute_gated directly; it calls ToolRegistry::execute (anthropic.rs ~:1299), and the trait shim is execute_gated's only caller.")
//! @yah:handoff("W342 open item (c) marked SHIPPED in .yah/docs/working/W342-context-economy.md. codex-cli recorded as out of reach by construction: it runs its own shell and feeds its own model; aggregated_output reaches yah's translate (agent_process.rs command_execution arm) only after consumption, so it is display-only.")
//! @yah:verify("cargo test -p yah-runner --lib  # 511 passed, 0 failed (4 new tool::tests: verbatim under cap, oversized read keeps structure/head/tail, many-small-rows text fallback, multibyte boundaries)")
//! @yah:verify("The multibyte test caught a real bug in the first cut (raw-byte keep estimate saturated to 0 on escape-heavy leaves, dropping the whole leaf); fixed by scaling by the leaf's escape ratio before the shrink loop.")
//! @yah:gotcha("W342's other open items are NOT filed as R883 children: (a) wire context_window for hosted openai_compat engines, (b) cap Bash cat / Grep on claude-cli, plus the 'New' rows (context-management + compact betas on anthropic.rs, breakpoints 3-4, non-leader doctrine split, per-job TTL, usage anchor). R883 has only this child, so it will read as done when this archives. Run /refine on W342 before archiving the relay.")
//! @yah:gotcha("One-time cache miss on resume for openai_compat sessions whose event log holds a >32 KB tool result: replay now caps it, so the rebuilt prefix differs from what the provider cached pre-change. Expected, self-healing.")
//! @yah:gotcha("Camp daemon TaskRun store is corrupt: `yah build run` refused with 'turso: I/O error: short read on page 24050: expected 4096 bytes, got 0' (both runs 2026-09-14); builds fell back to local execution. Not from this ticket.")
//!
//! @yah:ticket(R950-F7, "Move anthropic.rs into yah-runner-ha implementing core Runner; camp-host integration becomes a caller-supplied hook trait")
//! @yah:at(2026-10-05T05:08:03Z)
//! @yah:status(review)
//! @yah:handoff("HA CELL (oss/runner/crates/ha, yah-runner-ha): anthropic.rs moved here with the annotation blocks that rode the cell (module header incl. this ticket; R883-S14/F2/F3/B13 on their items). AnthropicAuth::apply is now over core HttpRequest, and the new AnthropicAuth::headers() is what reqwest callers fold. PinnedShape, ANTHROPIC_CLAUDE_CODE_SYSTEM_PREFIX + context_editing_beta_header (now pub), model_supports_thinking, think_budget_tokens(&ReasoningBudget), anthropic_tools_payload, build_anthropic_body(+_with_tool_turns) over TurnInput, cache_control_value (moved from compose.rs, one definition), merge_anthropic_usage, ApiTurnError, run_anthropic_turn(host, core sink, Option<&dyn core SessionStore>, &dyn HttpClient, auth, url). Error text keeps the '429 Too Many Requests' form via http::StatusCode (new dep http = 1).")
//! @yah:handoff("HOOK (ha host.rs): ONE trait HaHost { turn_input(auth) -> TurnInput, read at turn start AND before each tool-loop re-issue (exactly where the old code re-locked the session); push_history; observe(event); set_phase(HaPhase::{Emitting, ToolUse, Blocked(reason)}); request_built(id, body, auth, url) for keepalive capture; response_headers(headers, auth) for usage capture }, the last four default no-op. TurnInput = session_id, model, think, system: SystemPrompt{blocks, cache_ttl}, history, tool_registry, cancel. SystemPrompt::from_prelude(core Prelude, auth) is the neutral rendering. DEVIATIONS from the brief, both to keep one definition: (a) turn gating, slot busy and the Idle phase are not hooks - they already live in sessions::send_claude around the turn, which is the host; (b) history reconstruction is a pure ha fn reconstruct_history(&[core AgentEvent]) that camp reconstruct_claude_history delegates to via core_history, not a hook.")
//! @yah:handoff("RUNNER (ha runner.rs): AnthropicRunner implements core Runner (start/send/stop/resume/tools_registry, kind AnthropicNative) over AnthropicConfig{model, auth, messages_url, think, store}; with_http always, new() under feature reqwest. send() returns a stream that drives the turn as it is polled (no tokio::spawn, so wasm-clean); each in-memory session is its own HaHost.")
//! @yah:handoff("CORE: sink.rs = SessionEventSink<E = core AgentEvent>, NullSink, FnSink, generic over the event type; wrapper sink.rs is only a pub use. Camp sites now name the vocabulary: 71 lines swept dyn SessionEventSink -> dyn SessionEventSink<AgentEvent> (runner sessions.rs/launch.rs, eval run.rs, hub in_process.rs, desktop agent.rs/commands.rs/agent_eval.rs, cli camp.rs incl. the BroadcastSink/TauriCampSink/WatchdogSink impls).")
//! @yah:handoff("WRAPPER (crates/yah/runner/src/anthropic.rs) is the camp seam: camp_system_prompt (job TTL/consolidation gates + compose_system_blocks + the R883-F3 one-shot 2-breakpoint cap, moved out of the body builder), camp_turn_input, CampHaHost{session, sink} (observe -> accumulate_partial_text_claude, set_phase -> update_phase_claude, request_built -> keepalive::global capture, response_headers -> usage_log::capture), CampSink (core -> party lift), and camp_anthropic_turn, the one entry sessions::send_claude calls. model_context_tokens and its R862/R863 blocks stay (layered camp catalog). Prompt bytes unchanged: same blocks, cap, window, TTL, and headers in the same order. Desktop agent.rs has no turn-fn reference; launch.rs only swapped an import path.")
//! @yah:handoff("DISCOVERED + FIXED: AnthropicAuth::apply(reqwest) had 8 callers outside the cell - runner keepalive.rs and btw.rs, sage triage/classify.rs, desktop agent.rs x3 (~9162/9204/9403), dictate.rs:868, resources.rs:204. All now fold auth.headers(); header order and values identical.")
//! @yah:handoff("TEST ACCOUNTING: the old anthropic module had 51 tests (2 ignored live). 37 moved to ha, all 11 R833-F13 tests among them: tool_turns_append_behind_the_rolling_cache_breakpoint re-expressed over host system prompts with 1/2/3 breakpoints instead of camp jobs, the 2 auth-apply tests assert HttpRequest headers, the 5 fake /v1/messages e2e tests run over a TestHost + ReqwestClient. 14 stayed in the wrapper via camp_turn_input: 8 compose-dependent body tests, 4 model_context_tokens, 2 ignored live DeepSeek. sink.rs's 2 tests moved to core. 3 new AnthropicRunner tests in ha runner.rs.")
//! @yah:handoff("A047 STALE LINES for R950-T10 (not edited here): 9-10, 28, 33, 87, 246, 270 still place the Anthropic cell in app/yah/desktop/src/agent.rs and out of the trait; it is now oss/runner/crates/ha, and AnthropicRunner implements Runner. 23 and 101 cite yah-kg/src/agent.rs for AgentEvent (does not exist: core event.rs / party agent.rs). 26 and 285 put SessionEventSink in yah-runner/src/sink.rs (now yah-runner-core sink.rs, generic). board.review via MCP returned EAGAIN (os error 35) and did not land, so this block was written by hand.")
//! @yah:verify("cd oss/runner && cargo test -p yah-runner-ha -> lib 40 pass (37 moved + 3 new), doctests 0. EXIT=0 /tmp/r950f7-ha-t2.log")
//! @yah:verify("cd oss/runner && cargo test -p yah-runner-core -p yah-runner-ho -> core 197 (195 + 2 sink tests moved) + 1 doctest; ho 102; ho e2e 12 / 1 ignored. EXIT=0 /tmp/r950f7-oss-t1.log")
//! @yah:verify("cargo test -p yah-runner -> lib 283 pass / 2 ignored (322 - 37 moved to ha - 2 moved to core). EXIT=0 /tmp/r950f7-run-t1.log")
//! @yah:verify("cargo check -p yah-party -p yah -p desktop -p yah-hub -p yah-eval -p probe -p yah-agent-tools -p sage -p yah-e2e --tests EXIT=0, no skew (/tmp/r950f7-wide-c4.log); cargo check -p yah-runner --tests --bins --examples EXIT=0; cd oss/runner && cargo check --workspace --tests EXIT=0")
//! @yah:verify("cd oss/runner && cargo tree -p yah-runner-ha -e normal -> 0 lines matching yah-(board|party|kg|fob|bridge)")
//! @yah:verify("cd oss/runner && cargo check -p yah-runner-ha -p yah-runner-core -p yah-runner-ho --no-default-features --target wasm32-unknown-unknown EXIT=0; same with --target aarch64-apple-ios EXIT=0")
//! @yah:assignee(agent:bundle-anthropic-glimmerstone)
//! @yah:parent(R950)
//! @yah:next("Move build_anthropic_body (+ _with_tool_turns), run_anthropic_turn, AnthropicAuth, PinnedShape, merge_anthropic_usage, model_supports_thinking, think_budget_tokens into crates/ha over core HttpClient, and make it implement core Runner (it deliberately does not today, anthropic.rs:12).")
//! @yah:next("Measured camp-host edges that must become one HA hook trait the wrapper implements: sessions::{AgentSession, update_phase_claude, reconstruct_claude_history, accumulate_partial_text_claude}, slot_table::SlotTable, turn_gate::TurnGate, keepalive::{DirectKeepalive, global} (keepalive is the only bridge:: user in the crate), compose::{compose_system_blocks, job_is_long_lived, job_wants_extended_cache, RunnerMode, cache_control_value}, usage_log::capture, sink::{SessionEventSink, FnSink, NullSink}, party::agent::SessionPhase::Idle, fob::get_or_env (-> F6 CredentialSource).")
//! @yah:next("Generic pieces of sink.rs (SessionEventSink over core AgentEvent) move to core; compose/keepalive/sessions/slot_table stay in the wrapper (W357 §3).")
//! @yah:next("Desktop/daemon behaviour must not change: app/yah/desktop/src/agent.rs keeps one call site (verified 1 reference to run_anthropic_turn/build_anthropic_body); launch.rs is the R833-F11 launch seam.")
//! @yah:next("Tier: Wizard — the cell was kept outside the trait on purpose; designing the hook that lets it in without moving camp state is the hardest seam in R950.")
//! @yah:verify("cd oss/runner && cargo test -p yah-runner-ha (the 11 anthropic unit+fake-/v1/messages e2e tests move with it)")
//! @yah:verify("cargo test -p yah-runner && cargo check -p yah-desktop -p yah")
//! @yah:gotcha("\"Assumption VERIFIED 2026-10-04 (Glimmerstone): W357 is right, A047 is stale. build_anthropic_body (anthropic.rs:596), build_anthropic_body_with_tool_turns (:694) and run_anthropic_turn (:1063) are defined in crates/yah/runner/src/anthropic.rs; app/yah/desktop/src/agent.rs (13.7k lines) defines neither and references them once — the cell was lifted out in R032-T4 (anthropic.rs header). A047 lines 10/28/33/87/246/270 still point at desktop agent.rs and should be corrected when this lands. Separately: A047 cites crates/yah/kg/src/agent.rs for AgentEvent; that file does not exist — AgentEvent is in crates/yah/party/src/agent.rs:828.\"")
//! @yah:depends_on(R950-F4)
//! @yah:depends_on(R950-F5)
//! @yah:depends_on(R950-F6)
//! @yah:files(oss/runner/crates/ha/src/lib.rs)
//! @yah:files(crates/yah/runner/src/anthropic.rs)
//! @yah:files(crates/yah/runner/src/sessions.rs)
//! @yah:files(crates/yah/runner/src/launch.rs)
//! @yah:files(crates/yah/runner/src/sink.rs)
//! @yah:files(crates/yah/runner/src/keepalive.rs)
//! @yah:files(crates/yah/runner/src/compose.rs)
//! @yah:files(crates/yah/runner/src/lib.rs)
//! @yah:files(app/yah/desktop/src/agent.rs)

use std::fmt;
use std::sync::Arc;

use futures::StreamExt;
use yah_runner_core::{
    build_wire_name_map, sanitize_tool_name_for_wire, AgentEvent, CacheTtl, HttpClient,
    HttpRequest, Message, MessageContent, ReasoningBudget, Role, SessionEventSink, SessionId,
    SessionStore, ToolRegistry, TurnUsage,
};

use crate::host::{HaHost, HaPhase, TurnInput};

/// Default Anthropic API endpoint. Used when no `Connection.base_url` is
/// configured. Callers pass the resolved URL to [`run_anthropic_turn`].
pub const ANTHROPIC_MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// `anthropic-beta` header value the OAuth path requires when no
/// pinned shape is loaded. Comma-joined: `oauth-2025-04-20` enables
/// OAuth-token semantics, `claude-code-20250219` declares the request
/// follows the Claude Code protocol, `context-management-2025-06-27`
/// (R883-F2) unlocks the `context_management` body field, and
/// `compact-2026-01-12` (R883-T10) unlocks the `compact_20260112` edit
/// within that same field. Sending only the first two values still worked
/// before R883-F2; omitting any of the first three triggers shape-mismatch
/// rate-limiting on the Anthropic side, so all four are additive here,
/// never a replacement of the pre-existing set.
const ANTHROPIC_OAUTH_BETA_VALUE: &str =
    "oauth-2025-04-20,claude-code-20250219,context-management-2025-06-27,compact-2026-01-12";

/// Beta flags that unlock the `context_management` request-body field's two
/// edit strategies: `context-management-2025-06-27` (R883-F2, W342 lever 2 —
/// server-side context editing) and `compact-2026-01-12` (R883-T10 —
/// server-side compaction). Both edits ride the same body field and the same
/// gating, so both betas travel together, comma-joined, wherever this
/// constant is sent. Folded directly into [`ANTHROPIC_OAUTH_BETA_VALUE`]'s
/// fixed list for the degraded (no [`PinnedShape`]) OAuth path; added as a
/// standalone header for the `ApiKey` / `BearerToken` paths, which otherwise
/// send no `anthropic-beta` header at all. Never sent on a pinned-shape
/// OAuth request — see [`AnthropicAuth::apply`]'s doc on why
/// [`PinnedShape::headers`] must replay verbatim, which leaves no channel to
/// add it there without mutating the audited signature.
const ANTHROPIC_CONTEXT_MANAGEMENT_BETA: &str =
    "context-management-2025-06-27,compact-2026-01-12";

/// Required system-prompt prefix on OAuth-authed `/v1/messages` calls
/// when no pinned shape is loaded. Anthropic enforces that OAuth-token
/// requests' first system block begins with this exact string — it's
/// the TOS-enforcement hook that distinguishes Claude Code from
/// third-party tools.
pub const ANTHROPIC_CLAUDE_CODE_SYSTEM_PREFIX: &str =
    "You are Claude Code, Anthropic's official CLI for Claude.";

/// Soft cap on the SSE stream's total bytes. A cooperating Anthropic
/// response is bounded by `max_tokens`, but a misbehaving upstream
/// shouldn't be able to fill the host's working set unbounded.
const MAX_STREAM_BYTES: usize = 32 * 1024 * 1024; // 32MB

/// Pre-stream retry: how many times to attempt opening the `/v1/messages`
/// connection before giving up. Only the *pre-stream* phase retries —
/// network failure to send, or a transient HTTP status (5xx / 429). Once
/// the response body starts streaming, partial output may have reached
/// the FE and we cannot safely re-issue. See [`is_retryable_status`].
const MAX_STREAM_OPEN_ATTEMPTS: u32 = 3;

/// Cap on `Retry-After` we'll honor before declaring the rate-limit
/// permanent. Keeps a misconfigured upstream from stalling a session for
/// hours; users see an `Error` instead and can act.
const MAX_RETRY_AFTER_SECS: u32 = 30;

/// R833-F13: how many `tool_use` → `tool_result` round trips one turn may
/// make before the loop closes itself out with a synthetic assistant reply.
/// Mirrors the openai-compat cell's `max_tool_iterations` policy knob; the
/// Anthropic-direct path has no per-engine config surface to read it from
/// yet, so it's a constant. This is a *policy* stop, not an error — the
/// model isn't stuck, it just spent the turn's budget.
const MAX_TOOL_ITERATIONS: usize = 24;

/// Whether a non-success HTTP status from `/v1/messages` should be
/// retried. Mirrors the [`yah_runner_core::health::ErrorKind`] taxonomy: 429 and
/// 5xx are transient; 401/403 (auth) and 4xx (bad-request) are permanent.
fn is_retryable_status(code: u16) -> bool {
    code == 429 || (500..600).contains(&code)
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

/// Per-turn token cap forwarded to the Anthropic API. Independent of
/// the prelude's ring budget — that bounds *input*; this bounds
/// *output*. Both are needed.
const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 4096;

/// Anthropic's documented minimum for `thinking.budget_tokens`. Smaller
/// values reject with 400; we clamp `Budget { tokens: N }` and the
/// `Fast` tier up to this floor.
const ANTHROPIC_THINKING_MIN_BUDGET: u32 = 1024;

/// Authentication mode for the Anthropic-HTTP cell. The host picks
/// between variants by reading its credential store (Tauri keychain,
/// `~/.config/yah/credentials.enc` for `yah-agentd`, etc.) and
/// constructs this enum at the call boundary so the runner side stays
/// host-agnostic.
///
/// The `Oauth` variant carries an optional pinned wire-shape pulled in
/// from a build-time audit (see
/// `.yah/docs/architecture/A030-yah-claude-shape-capture.md`). When present,
/// [`Self::apply`] replays the audited header set verbatim so the
/// request matches Claude Code's signature; when absent (degraded mode)
/// we fall back to the bare `anthropic-beta` + `Authorization` pair.
///
/// `BearerToken` is for third-party Anthropic-wire endpoints (DeepSeek)
/// that use `Authorization: Bearer` but don't need OAuth-specific headers.
///
/// [`PinnedShape`] is intentionally generic over the host's shape
/// representation — the host extracts the relevant headers + system
/// prefix and hands them in as opaque pairs / a string, decoupling
/// runner from the host's audit-artifact loader.
#[derive(Debug, Clone)]
pub enum AnthropicAuth {
    /// `anthropic` (HAk) preset — Console API key. Sent as `x-api-key`.
    ApiKey(String),
    /// `anthropic-gc` (HAo) preset — long-lived OAuth bearer token from
    /// `claude setup-token`. Sent as `Authorization: Bearer …` plus
    /// the pinned-shape header set (or the hardcoded `anthropic-beta`
    /// fallback when no shape is loaded).
    Oauth {
        token: String,
        shape: Option<Arc<PinnedShape>>,
    },
    /// Bearer token for third-party Anthropic-wire endpoints
    /// (e.g. DeepSeek). Sent as `Authorization: Bearer <key>` plus
    /// `anthropic-version`. No OAuth-specific headers are added.
    BearerToken(String),
}

/// Audited wire shape for an OAuth request. The host loads this from
/// its build-time audit artifact; runner consumes only the bits it
/// needs (`headers`, `system_prefix`).
///
/// This deliberately sits in the runner crate (not `kg`) because the
/// audit is an Anthropic-cell concern — other transports won't need
/// it. The host's loader (e.g. `yah-tauri::claude_shape`) constructs
/// instances; runner only reads them.
#[derive(Debug, Clone)]
pub struct PinnedShape {
    /// Header name → value. Names compared case-insensitively at
    /// apply time; the [`AnthropicAuth::apply`] skip-list
    /// (authorization, host, content-length) prevents conflict with
    /// the transport's own bookkeeping.
    pub headers: Vec<(String, String)>,
    /// First system block on OAuth-authed requests. Anthropic enforces
    /// the prefix as the TOS hook that distinguishes Claude Code from
    /// third-party tools.
    pub system_prefix: String,
}

impl AnthropicAuth {
    /// The request headers this auth contributes, in send order. Centralises
    /// the API-key vs Bearer split so call sites (`/v1/messages`,
    /// `/v1/models`) don't each re-derive the header set — whatever HTTP
    /// substrate they build on. Includes `anthropic-version` so callers don't
    /// have to remember.
    ///
    /// On the OAuth path with a loaded [`PinnedShape`], every pinned
    /// header is copied verbatim except the three that the transport
    /// owns (authorization, host, content-length). The authorization slot
    /// is filled with yah's own bearer token.
    pub fn headers(&self) -> Vec<(String, String)> {
        let pair = |k: &str, v: &str| (k.to_string(), v.to_string());
        match self {
            Self::ApiKey(k) => vec![pair("x-api-key", k), pair("anthropic-version", ANTHROPIC_VERSION)],
            Self::Oauth {
                token,
                shape: Some(shape),
            } => {
                let mut out = vec![pair("authorization", &format!("Bearer {token}"))];
                for (name, value) in &shape.headers {
                    if name.eq_ignore_ascii_case("authorization")
                        || name.eq_ignore_ascii_case("host")
                        || name.eq_ignore_ascii_case("content-length")
                    {
                        continue;
                    }
                    out.push((name.clone(), value.clone()));
                }
                out
            }
            Self::Oauth { token, shape: None } => vec![
                pair("authorization", &format!("Bearer {token}")),
                pair("anthropic-beta", ANTHROPIC_OAUTH_BETA_VALUE),
                pair("anthropic-version", ANTHROPIC_VERSION),
            ],
            Self::BearerToken(k) => vec![
                pair("authorization", &format!("Bearer {k}")),
                pair("anthropic-version", ANTHROPIC_VERSION),
            ],
        }
    }

    /// Append [`Self::headers`] to a core [`HttpRequest`].
    pub fn apply(&self, mut req: HttpRequest) -> HttpRequest {
        for (name, value) in self.headers() {
            req = req.header(name, value);
        }
        req
    }

    /// The cloak text to use as block 0 for golden-cloak (OAuth) sessions,
    /// or `None` for ApiKey sessions (no cloak needed).
    ///
    /// Prefers the `PinnedShape.system_prefix` when a shape is loaded; falls
    /// back to the hardcoded [`ANTHROPIC_CLAUDE_CODE_SYSTEM_PREFIX`] for
    /// degraded OAuth (no shape loaded). The empty-prefix case also falls back
    /// to the hardcoded copy — an empty block 0 would invalidate TOS enforcement.
    pub fn cloak_prefix(&self) -> Option<&str> {
        match self {
            Self::ApiKey(_) | Self::BearerToken(_) => None,
            Self::Oauth {
                shape: Some(shape), ..
            } if !shape.system_prefix.is_empty() => Some(shape.system_prefix.as_str()),
            Self::Oauth { .. } => Some(ANTHROPIC_CLAUDE_CODE_SYSTEM_PREFIX),
        }
    }
}

/// Whether the given Anthropic model id supports extended thinking.
/// Extended thinking shipped with `claude-3-7-sonnet` (Feb 2025) and
/// every `claude-4-*` family member ships with it on. Older 3.5/3.0
/// models — including the original Haiku 3 family — silently 400 on
/// the `thinking` field, so we omit it for them.
///
/// DeepSeek v4 models also accept the `thinking` field on the Anthropic
/// wire (live-tested 2026-07-24 on `api.deepseek.com/anthropic`).
pub fn model_supports_thinking(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    m.starts_with("claude-opus-4")
        || m.starts_with("claude-sonnet-4")
        || m.starts_with("claude-haiku-4")
        || m.starts_with("claude-3-7")
        || m.starts_with("deepseek-v4")
        || m.starts_with("kimi-k3")
        || m.starts_with("kimi-k2")
}

/// Whether `model` is a genuine Anthropic-hosted Claude model, as opposed
/// to a third-party Anthropic-wire host (DeepSeek, Kimi/Moonshot) that
/// shares this cell's `/v1/messages` protocol but not necessarily every
/// Anthropic-only beta. `context_management` (R883-F2) has no cross-vendor
/// contract, so it is restricted to genuine `claude-*` ids — same prefix
/// check the catalog-coverage test below uses to skip non-Claude entries.
fn model_supports_context_editing(model: &str) -> bool {
    model.to_ascii_lowercase().starts_with("claude-")
}

/// Whether `model` accepts the `compact_20260112` edit. Anthropic's threshold-
/// compaction compatibility list (platform.claude.com compaction-threshold,
/// read 2026-09-27) is narrower than context editing's: Fable 5/5.1, Mythos,
/// Opus 5/5.5 and 4.6-4.8, Sonnet 5 and 4.6 — NOT Haiku 4.5 or Sonnet 4.5,
/// which a direct-road `anthropic-scout` session runs. Prefix match so a
/// dated snapshot id still counts.
///
/// @yah:ticket(R883-S14, "Live-verify R883 compaction + shared doctrine prefix on the Direct road once a Direct-road Anthropic credential exists (parked from R883-S12)")
/// @yah:status(review)
/// @yah:at(2026-09-27T19:25:01Z)
/// @yah:kind(spike)
/// @yah:assignee(agent:bundle-anthropic-ashguard)
/// @yah:parent(R883)
/// @yah:next("Tier: Warrior. Blocked on operator: host has no anthropic-api-key / anthropic-oauth fob slot and ANTHROPIC_API_KEY unset (checked 2026-09-27); operator chose to park rather than use Claude Code's OAuth token. Unblock = operator runs `yah keys set anthropic-api-key`.")
/// @yah:next("Claim 1 (compaction): yah exposes NO trigger knob (compact_20260112 sent bare => API default 150K; API minimum is 50K). Cheapest proof without inventing one: one Sonnet 4.6 turn whose input exceeds 150K. Verify: request carries anthropic-beta context-management-2025-06-27,compact-2026-01-12; response streams content_block_start{compaction} + compaction_delta; history Blocks carry non-empty content (R883-B13); next request echoes it verbatim; restart replay via TurnEnded.contentBlocks rebuilds Blocks (T11). Note merge_anthropic_usage ignores usage.iterations, so compaction-iteration tokens are invisible to yah's accounting — read raw usage.")
/// @yah:next("Claim 2 (doctrine prefix): operator APPROVED 2026-09-27 minting one roster character via `yah party add --subclass anthropic-smith --name <Name>` (anthropic-smith = runner yah, connection anthropic, auth key, claude-sonnet-4-6). Dispatch two no-tool courier probes ~3 min apart; courier #2 turn 1 must show cache_read ~= doctrine prefix, cache_creation only the ticket tail. TTL per code (R883-F5): courier gets 1h via EXTENDED_TTL_JOBS; confirm from raw usage.cache_creation.ephemeral_1h_input_tokens, which merge_anthropic_usage also drops.")
/// @yah:next("Installed /Applications/yah.app (built 2026-09-27 12:05) predates R883-B13; rebuild/reinstall before dispatching through the daemon or claim 1 will lose the summary.")
/// @yah:depends_on(R883-B13)
/// @yah:next("OPERATOR CALL 2026-09-27: test on DeepSeek's Anthropic-compatible endpoint instead of paying for the Anthropic API (anthropic.rs already routes deepseek-v4-* over the Anthropic wire, :486). First establish from DeepSeek's own docs which of compact_20260112 / clear_tool_uses / cache_control / anthropic-beta it honours. Prove on DeepSeek whatever it supports (at minimum: the direct-road wire round-trips non-text blocks and the doctrine prefix cache-hits across two couriers, read from its usage fields). Anything DeepSeek cannot exercise (likely server-side compaction itself) is recorded as NOT PROVABLE ON DEEPSEEK with the doc citation, not faked. model_supports_compaction (R883-B13) excludes deepseek-*, so do not widen that allowlist unless DeepSeek documents compaction support.")
/// @yah:handoff("(1) DeepSeek docs (api-docs.deepseek.com/guides/anthropic_api + /guides/kv_cache, read 2026-09-27): base https://api.deepseek.com/anthropic; anthropic-beta IGNORED for /messages; cache_control IGNORED on all blocks; thinking supported (budget_tokens ignored); content blocks text/image/thinking/tool_use/tool_result supported, redacted_thinking/document not; NO mention of context_management, clear_tool_uses or compaction. Caching is automatic on-disk prefix caching, best-effort, full-prefix-unit matches only, lifetime hours-days. Observed live: the Anthropic endpoint reports Anthropic-shaped usage (cache_read_input_tokens; cache_creation_input_tokens always 0) and hits land in multiples of 128 tokens. yah sends no context_management to deepseek-* (model_supports_context_editing is claude-only), so none of the context_management edits are even on the DeepSeek wire; model_supports_compaction left unwidened.")
/// @yah:handoff("(2a) PASS - non-text blocks round-trip on the Direct-road wire. Live probe anthropic::tests::live_deepseek_non_text_blocks_round_trip (#[ignore], R883_LIVE=1): real run_anthropic_turn, BearerToken auth (same variant launch::claude_session_auth builds for deepseek), deepseek-v4-flash, thinking on, behind a logging proxy. DeepSeek streams thinking as content_block_start{thinking:'',signature:''} + thinking_delta + signature_delta (signature = the message id); R883-B13's fold_block_delta rebuilt both fields. Run 2 (passing), msg ids 9104fa19-e7be-41a6-9f67-6f178fd61a17 / 2f14c106-4a55-42ca-85d1-e6cb08435c8c / e4c649ca-49e6-4876-9975-d6bc34a8822a, all HTTP 200: (a) turn A history = Blocks[thinking+signature, text]; (b) turn B request 1 echoes it verbatim apart from the T10 windowed cache_control on the last block; (c) the tool-loop re-issue echoes [thinking+signature, tool_use] and DeepSeek accepts it; (d) SessionStore::replay + reconstruct_claude_history rebuilds turn A as identical Blocks (T11). Usage: 245 in/256 read, 290 in/512 read, 255 in/640 read. Harness sessions are in-process fixtures (session:deadbeef), not daemon sessions.")
/// @yah:handoff("(2b) PASS - courier #2 cache-hits courier #1's doctrine prefix. anthropic::tests::live_deepseek_courier_doctrine_prefix_cache: two job=courier Direct-road sessions (session:r883p0 / session:r883p1), one identity (per-run nonce so #1 is cold), prelude = the real 26,977-byte courier doctrine sidecar .yah/sessions/doctrine/675a66b6a65c003f44e001b0a91a57bd.md + distinct ticket tails, prelude_doctrine_bytes = doctrine length, 180 s apart. System blocks identical in both requests: [baseline 1942B cc 1h, identity 50B, doctrine 26977B cc 1h, ticket 52B], so the R883-F4 split and R883-F5 courier TTL=1h are what got SENT. Courier #1 usage: input 6863, cache_read 256 (baseline only). Courier #2 first turn: input 207, cache_read 6912, so the doctrine came from cache. TTL sent = 1h (ttl:'1h' on both breakpoints); DeepSeek ignores cache_control, so this proves prefix byte-stability plus a provider prefix hit, not Anthropic breakpoint/TTL semantics.")
/// @yah:handoff("(3) NOT PROVABLE ON DEEPSEEK: server-side compaction (compact_20260112 / compact-2026-01-12). DeepSeek's Anthropic-compat page lists no context_management and says anthropic-beta is ignored, and yah never sends the field to deepseek-*. Same for clear_tool_uses_20250919, for cache_control breakpoint placement, and for Anthropic TTL honouring (cache_control 'Ignored across all content blocks'). Those still need an Anthropic credential; compaction additionally needs a >150K-token turn (no trigger knob in yah). The compaction wire shape is covered only by R883-B13's doc-derived fixtures.")
/// @yah:handoff("Daemon path not used; everything ran through the in-crate harness on the working tree (includes B13), so no rebuild/reinstall of /Applications/yah.app was needed. Attempted: the installed build's preroll has no deepseek-v4-pro-fast subclass and its daemon caches the roster (a hand-minted ~/.yah/agents record was not picked up without a reload, which is an operator action). The probe record was removed; no roster change remains. The daemon's 'yah party add --subclass' hint (app/yah/cli/src/camp.rs format_character_not_found, R477-F6) names a CLI subcommand that does not exist.")
/// @yah:handoff("Extra work found: R883-B15 (filed, open), the Wrapped road prepends the per-SESSION identity stanza ahead of the doctrine in --append-system-prompt; 8 recent claude-cli slot prompts share only 68 bytes, so the doctrine never cache-shares across sessions on that road. Corrected the stale self_identity_preamble doc (compose.rs), which claimed the Direct road prepends it. Also noted: merge_anthropic_usage drops usage.iterations and the cache_creation 1h/5m split. cargo test -p yah-runner --lib: 570 passed, 0 failed, 2 ignored (the live probes).")
/// @yah:verify("R883_LIVE=1 cargo test -p yah-runner --lib live_deepseek_non_text -- --ignored --nocapture  (PASS 2026-09-27; evidence /tmp/r883s14_live_blocks.json)")
/// @yah:verify("R883_LIVE=1 R883_DOCTRINE=.yah/sessions/doctrine/675a66b6a65c003f44e001b0a91a57bd.md cargo test -p yah-runner --lib live_deepseek_courier -- --ignored --nocapture  (PASS 181.6s; evidence /tmp/r883s14_live_prefix.json)")
/// @yah:verify("cargo test -p yah-runner --lib: 570/0, 2 ignored")
/// @yah:gotcha("DeepSeek prefix cache persists hours to days and is shared per account: re-running the courier probe without a fresh identity nonce would show a warm courier #1. The test salts the identity per run for that reason.")
fn model_supports_compaction(model: &str) -> bool {
    const COMPACTION_MODELS: &[&str] = &[
        "claude-fable-5",
        "claude-mythos-",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
    ];
    let m = model.to_ascii_lowercase();
    COMPACTION_MODELS.iter().any(|p| m.starts_with(p))
}

/// The `anthropic-beta` value to add for `body`, if any — shared by
/// [`run_anthropic_turn`]'s request construction and
/// a host's wake-ahead replay of a captured body (yah's `DirectKeepalive`), since
/// both send whatever `build_anthropic_body_with_tool_turns` produced
/// (yah's replay path passes it through `bridge::build_keepalive_body`
/// byte-identically). Gated on the body itself rather than re-deriving the
/// model/auth check: single source of truth for whether the field is
/// present. `Oauth` is excluded because its two paths either already carry
/// the flag baked into [`ANTHROPIC_OAUTH_BETA_VALUE`] (no shape), or must
/// never gain a second, non-verbatim `anthropic-beta` header (pinned shape).
pub fn context_editing_beta_header(
    auth: &AnthropicAuth,
    body: &serde_json::Value,
) -> Option<&'static str> {
    if body.get("context_management").is_none() {
        return None;
    }
    match auth {
        AnthropicAuth::ApiKey(_) | AnthropicAuth::BearerToken(_) => {
            Some(ANTHROPIC_CONTEXT_MANAGEMENT_BETA)
        }
        AnthropicAuth::Oauth { .. } => None,
    }
}

/// Translate a [`ReasoningBudget`] tier into the `budget_tokens` value
/// Anthropic's `/v1/messages` expects. Tiers are deliberately coarse
/// — a workspace-default-model setting (R027 follow-up) can refine
/// the numeric mapping per model later. `Budget { tokens }` is the
/// escape hatch for callers who need an explicit cap; we still clamp
/// up to Anthropic's documented minimum so the API doesn't 400.
pub fn think_budget_tokens(budget: &ReasoningBudget) -> u32 {
    match budget {
        ReasoningBudget::Deep => 16_000,
        ReasoningBudget::Standard => 4_000,
        ReasoningBudget::Fast => ANTHROPIC_THINKING_MIN_BUDGET,
        ReasoningBudget::Budget { tokens } => (*tokens).max(ANTHROPIC_THINKING_MIN_BUDGET),
    }
}

/// Assemble the JSON body for a single `/v1/messages` turn. Pulled out
/// of [`run_anthropic_turn`] so the `@yah:think` mapping (and the
/// model-thinking-capability check) can be unit-tested without
/// standing up an HTTP server.
///
/// When `input.think` is `Some(_)` and the chosen model supports
/// extended thinking, the body grows a `thinking: { type: "enabled",
/// budget_tokens: N }` field and `max_tokens` is bumped by `N` so the
/// model has headroom for both the thinking trace *and* the actual
/// output (Anthropic requires `max_tokens > budget_tokens`). Otherwise
/// the body matches the pre-think baseline.
pub fn build_anthropic_body(input: &TurnInput, auth: &AnthropicAuth) -> serde_json::Value {
    build_anthropic_body_with_tool_turns(input, auth, &[])
}

/// Build the `tools[]` request payload from a registry, if any — the
/// Anthropic-shaped sibling of `openai_compat::tools_array_payload`.
///
/// Anthropic's `/v1/messages` takes the schema *flat*
/// (`{name, description, input_schema}`) rather than OpenAI's
/// `{type: "function", function: {...}}` envelope, and calls the schema field
/// `input_schema` rather than `parameters`. Everything else — the
/// `^[a-zA-Z0-9_-]+$` name constraint, and therefore the dots-to-underscores
/// rewrite plus the wire→registry name map used to route the model's echoed
/// name back to [`ToolRegistry::execute`] — is shared, so both wires call the
/// same [`sanitize_tool_name_for_wire`] / [`build_wire_name_map`] pair.
///
/// Returns `None` when there is no registry, or when an attached registry
/// advertises nothing: an empty `tools: []` reads to most providers as "tools
/// are allowed but none are offered", which is strictly more confusing than
/// omitting the key. `None` means the model is told nothing about tools and
/// should never emit a `tool_use` block.
///
/// R833-F13.
pub fn anthropic_tools_payload(
    tools: Option<&Arc<dyn ToolRegistry>>,
) -> (
    Option<Vec<serde_json::Value>>,
    std::collections::HashMap<String, String>,
) {
    let Some(registry) = tools else {
        return (None, std::collections::HashMap::new());
    };
    let schemas = registry.schemas();
    if schemas.is_empty() {
        return (None, std::collections::HashMap::new());
    }
    let name_map = build_wire_name_map(&schemas);
    let arr = schemas
        .into_iter()
        .map(|s| {
            serde_json::json!({
                "name": sanitize_tool_name_for_wire(&s.name),
                "description": s.description,
                "input_schema": s.input_schema,
            })
        })
        .collect();
    (Some(arr), name_map)
}

/// [`build_anthropic_body`] plus this turn's in-flight tool round trips.
///
/// `tool_turns` is the turn-local transcript the [`run_anthropic_turn`] tool
/// loop accumulates: alternating `assistant` messages carrying `tool_use`
/// content blocks and `user` messages carrying the matching `tool_result`
/// blocks. They append *after* everything derived from `input.history`,
/// which is what keeps the cache breakpoint layout stable — see the
/// breakpoint note inside.
///
/// Empty slice ⇒ byte-identical to the pre-R833-F13 body except for the
/// `tools[]` key, which is present exactly when the session carries a
/// non-empty [`TurnInput::tool_registry`].
///
/// @yah:ticket(R883-F2, "Send Anthropic server-side context editing + compaction on the direct-HTTP wire (W342 lever 2)")
/// @yah:status(review)
/// @yah:at(2026-09-26T05:06:16Z)
/// @yah:assignee(agent:bundle-anthropic-miravel)
/// @yah:phase(P1)
/// @yah:parent(R883)
/// @yah:next("Tier: Cleric — new provider contract plus a response-preservation rule; a wrong compaction round trip loses state silently.")
/// @yah:next("Add context_management {edits: [{type: clear_tool_uses_20250919}]} with beta context-management-2025-06-27, and beta compact-2026-01-12, to the body built here. Today anthropic.rs sends neither (grep: no context_management, no clear_tool_uses).")
/// @yah:next("Compaction needs response.content appended VERBATIM so compaction blocks survive. run_anthropic_turn pushes only accumulated text into session.history (Message content is a plain String, R833-B14) — that is the lossy extraction W342 warns about. Resolve that seam before enabling compaction; context editing alone does not need it.")
/// @yah:next("OAuth path: ANTHROPIC_OAUTH_BETA_VALUE is a fixed comma-joined anthropic-beta header. Merge new betas into it, do not replace it.")
/// @yah:verify("cargo test -p yah-runner --lib anthropic")
/// @yah:verify("Live anthropic-engine session past the trigger: usage shows cleared tool results / a compaction block, and the next turn still sees the compacted state.")
/// @yah:assumes("Beta names, edit type and the 150K compaction default come from the 2026-09-09 W342 diligence and were not re-read against Anthropic docs at filing. Load the claude-api skill and confirm before implementing.")
/// @arch:see(.yah/docs/working/W342-context-economy.md)
/// @yah:handoff("Compaction (compact-2026-01-12) deliberately NOT enabled: it requires appending response.content back into session.history verbatim so compaction blocks survive the round trip, but party::agent::Message.content (crates/yah/party/src/agent.rs) is a plain String (R833-B14) with no room for structured blocks, and that type is cross-crate (party, not runner) with its own wire-serialization form and every engine/consumer reading session.history as plain text — outside this courier's blast radius. Filed R883-T10 (parent R883) naming the seam and the follow-on work once it's resolved.")
/// @yah:verify("cargo test -p yah-runner --lib anthropic: 56 passed, 0 failed (baseline 49 + 7 new tests covering context_management inclusion/exclusion by model and auth variant, the OAuth beta-value merge, and the context_editing_beta_header helper). cargo build -p yah-runner --lib: clean (pre-existing unrelated warnings only).")
///
/// @yah:ticket(R883-F3, "Spend cache breakpoints 3-4 on a rolling transcript window; give one-shot jobs a transcript breakpoint (W342 lever 3)")
/// @yah:status(review)
/// @yah:at(2026-09-26T05:27:40Z)
/// @yah:assignee(agent:bundle-anthropic-miravel)
/// @yah:phase(P1)
/// @yah:parent(R883)
/// @yah:next("Tier: Cleric — the 4-breakpoint budget and the 20-block lookback interact; an off-by-one here is a silent cache miss, not a failure.")
/// @yah:next("Leader jobs spend 2 of 4 breakpoints (consolidated system + one rolling tail); one-shot jobs (lens/sniffer/gnome) get per-block system breakpoints and NO transcript breakpoint. Move to Hermes system_and_3: system + 3rd-to-last, 2nd-to-last, last message.")
/// @yah:next("Read the gotcha at the top of anthropic.rs (~:102, 20-block lookback) and R833-F13's CACHE INVARIANT handoff first: tool turns deliberately append BEHIND the rolling breakpoint so the count cannot overrun 4. Keep that property.")
/// @yah:next("Depends on F2 only because both edit this body builder; context editing also changes what sits behind the breakpoints.")
/// @yah:verify("cargo test -p yah-runner --lib anthropic  # extend tool_turns_append_behind_the_rolling_cache_breakpoint: count <= 4 for every job, with and without tool turns")
/// @yah:verify("Live: cache_read_input_tokens on a fan-out turn (>20 tool blocks) no longer drops to zero.")
/// @arch:see(.yah/docs/working/W342-context-economy.md)
/// @yah:depends_on(R883-F2)
/// @yah:handoff("Hermes-style multi-point transcript window landed in build_anthropic_body_with_tool_turns (anthropic.rs): system_blocks is now built first, one-shot jobs (!leader_job) cap system breakpoints at 2 (keep first 2, e.g. cloak+baseline or baseline+identity; drop the 3rd/4th block's cache_control), then transcript_breakpoints = min(4-system_breakpoints, 3, history_len) marks that many trailing messages (3rd-to-last/2nd-to-last/last) each with their own cache_control, replacing the old leader-only single last-message breakpoint. Invariant preserved: tool_turns still append strictly behind the computed window so the count is fixed once per body build and can never exceed 4 regardless of tool-loop iterations.")
/// @yah:handoff("compose.rs untouched (no edits) per scope — the doctrine/ticket split stays R883-F4's territory.")
/// @yah:handoff("Tests: extended tool_turns_append_behind_the_rolling_cache_breakpoint to loop over both a leader and a one-shot job asserting count<=4 with and without tool turns; rewrote build_body_adds_rolling_transcript_breakpoint_for_leader_job for the new 2-message window; added build_body_adds_rolling_transcript_breakpoint_for_one_shot_job_too covering the new one-shot transcript breakpoint; updated build_body_leaves_transcript_uncached_for_short_jobs to use empty history (the only case with zero transcript budget now); updated anthropic_body_oauth_path_prepends_claude_code_system_prefix for the OAuth one-shot system cap (2 of 4 blocks keep cache_control).")
/// @yah:verify("cargo test -p yah-runner --lib anthropic: 57 passed, 0 failed (baseline 56 + 1 net new test; one test rewritten, one added, three adjusted for the new breakpoint layout). cargo build -p yah-runner --lib: clean, pre-existing unrelated warnings only.")
pub fn build_anthropic_body_with_tool_turns(
    input: &TurnInput,
    auth: &AnthropicAuth,
    tool_turns: &[serde_json::Value],
) -> serde_json::Value {
    // The host renders the system prompt and decides how much of the
    // 4-breakpoint budget it spends there (yah: the compose layer plus its
    // one-shot cap, R883-F3/F4, R562-T3). Whatever it leaves goes to the
    // rolling transcript window below, at the host's TTL, so turn N+1 reads
    // the [system + prior-transcript] prefix from cache.
    let cache_ttl = input.system.cache_ttl;
    let system_blocks = &input.system.blocks;
    let system_breakpoints = system_blocks
        .iter()
        .filter(|b| b.get("cache_control").is_some())
        .count();

    // Hermes-style rolling transcript window (W342 lever 3): spend whatever
    // of the 4-breakpoint budget the system prompt didn't, on up to the last
    // 3 history messages (3rd-to-last, 2nd-to-last, last) instead of only the
    // last one. `system_breakpoints + transcript_breakpoints` can never
    // exceed 4 by construction, for every job.
    let history_len = input.history.len();
    let transcript_breakpoints = 4usize
        .saturating_sub(system_breakpoints)
        .min(3)
        .min(history_len);
    let transcript_window_start = history_len.saturating_sub(transcript_breakpoints);

    let mut messages: Vec<serde_json::Value> = input
        .history
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let role = match m.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            let windowed = i >= transcript_window_start;
            // R883-T10: a `Blocks` message (e.g. one carrying a preserved
            // compaction or tool_use block) must ride the rolling-window
            // cache_control on its LAST block rather than being wrapped in a
            // synthetic text block — wrapping would nest the whole block
            // array inside a `text` string field. `Text` keeps the pre-T10
            // shape: a plain string outside the window, a single
            // cache_control-bearing text block inside it. Either way exactly
            // one cache_control breakpoint is spent per windowed message, so
            // the ≤4-total invariant (R883-F3) is unaffected.
            let content = match &m.content {
                MessageContent::Text(text) => {
                    if windowed {
                        serde_json::json!([{
                            "type": "text",
                            "text": text,
                            "cache_control": cache_control_value(cache_ttl),
                        }])
                    } else {
                        serde_json::Value::String(text.clone())
                    }
                }
                MessageContent::Blocks(blocks) => {
                    if windowed {
                        let mut blocks = blocks.clone();
                        if let Some(last) = blocks.last_mut().and_then(|b| b.as_object_mut()) {
                            last.insert(
                                "cache_control".to_string(),
                                cache_control_value(cache_ttl),
                            );
                        }
                        serde_json::Value::Array(blocks)
                    } else {
                        serde_json::Value::Array(blocks.clone())
                    }
                }
            };
            serde_json::json!({ "role": role, "content": content })
        })
        .collect();

    // R833-F13, and read the R562-T3 gotcha above before touching this: the
    // rolling transcript window stays pinned to the tail of `input.history`
    // as it stood when this body was built, so this turn's tool round trips
    // append *behind* it rather than displacing it. Two consequences, both
    // deliberate:
    //
    //   1. The breakpoint count is unchanged by tool turns — it is fixed once
    //      `transcript_breakpoints` is computed above — so a tool loop cannot
    //      overrun the 4-breakpoint budget no matter how many iterations it
    //      runs. Marking each tool_result would blow that budget on turn one.
    //   2. Every iteration inside one turn re-issues the SAME cached prefix
    //      (system + history-through-breakpoint), so iteration N+1 reads it
    //      from cache and only the accumulated tool blocks reprocess. If a
    //      fan-out-heavy turn pushes >20 blocks past the last breakpoint it
    //      falls out of the 20-block lookback window and that turn re-warms —
    //      the known, accepted cost the gotcha describes, now reachable more
    //      often because tool loops exist at all.
    messages.extend(tool_turns.iter().cloned());

    let mut body = serde_json::json!({
        "model": input.model,
        "max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "stream": true,
        "system": system_blocks,
        "messages": messages,
    });

    // R833-F13: advertise the session's tools. Omitted entirely when the
    // session carries no registry — that is the pre-F13 body, and the state
    // every anonymous / CLI session is still in.
    //
    // Note for cache accounting: Anthropic's cacheable prefix is ordered
    // tools → system → messages, so introducing this key shifts the prefix and
    // costs exactly one cache miss the first time a given session sends it.
    // It is stable per session thereafter (schemas are a static manifest), so
    // there's nothing to amortise beyond that first turn.
    let (tools_array, _name_map) = anthropic_tools_payload(input.tool_registry.as_ref());
    if let Some(tools) = tools_array {
        body["tools"] = serde_json::Value::Array(tools);
    }

    if let Some(budget) = input.think.as_ref() {
        if model_supports_thinking(&input.model) {
            let budget_tokens = think_budget_tokens(budget);
            body["thinking"] = serde_json::json!({
                "type": "enabled",
                "budget_tokens": budget_tokens,
            });
            // max_tokens must exceed budget_tokens — give the model
            // DEFAULT_MAX_OUTPUT_TOKENS of post-thinking output room.
            body["max_tokens"] =
                serde_json::Value::from(budget_tokens.saturating_add(DEFAULT_MAX_OUTPUT_TOKENS));
        }
    }

    // R883-F2 (W342 lever 2): server-side context editing. Anthropic clears
    // stale tool_use/tool_result blocks server-side as the window fills,
    // rather than yah eating the 20-block-lookback cache miss the R562-T3
    // gotcha above describes.
    //
    // R883-T10: compaction (`compact_20260112`) rides the same field and the
    // same gating now that `Message.content` can hold a raw
    // block array — `run_anthropic_turn` pushes the response's content
    // blocks verbatim into the session history (see the history push near the
    // end of that function), and this same match arm sends them back
    // unchanged on the next request, so a compaction block Anthropic hands
    // back survives the round trip instead of being flattened to text.
    //
    // Excluded on a pinned-shape OAuth request: there is no channel to add
    // the required beta flags without mutating `PinnedShape::headers`, which
    // must replay verbatim (see `AnthropicAuth::apply`'s doc).
    if model_supports_context_editing(&input.model)
        && !matches!(auth, AnthropicAuth::Oauth { shape: Some(_), .. })
    {
        let mut edits = vec![serde_json::json!({ "type": "clear_tool_uses_20250919" })];
        // R883-B13: compaction has a narrower model list than context editing
        // (no Haiku 4.5, no Sonnet 4.5) — see `model_supports_compaction`.
        if model_supports_compaction(&input.model) {
            edits.push(serde_json::json!({ "type": "compact_20260112" }));
        }
        body["context_management"] = serde_json::json!({ "edits": edits });
    }

    body
}

/// Merge Anthropic's `usage` JSON object into a running [`TurnUsage`]
/// accumulator. Anthropic splits the numbers across two SSE events:
/// `message_start.message.usage` carries `input_tokens` +
/// `cache_read_input_tokens` + `cache_creation_input_tokens` (and a
/// placeholder `output_tokens` ≈ 1), while each `message_delta.usage`
/// carries the running cumulative `output_tokens`. We let later
/// values overwrite earlier ones — that means the final
/// `message_delta.usage.output_tokens` wins, as desired.
pub fn merge_anthropic_usage(acc: &mut TurnUsage, usage: &serde_json::Value) {
    let read_u32 = |k: &str| usage.get(k).and_then(|v| v.as_u64()).map(|n| n as u32);
    if let Some(v) = read_u32("input_tokens") {
        acc.input_tokens = Some(v);
    }
    if let Some(v) = read_u32("output_tokens") {
        acc.output_tokens = Some(v);
    }
    if let Some(v) = read_u32("cache_read_input_tokens") {
        acc.cache_read_tokens = Some(v);
    }
    if let Some(v) = read_u32("cache_creation_input_tokens") {
        acc.cache_write_tokens = Some(v);
    }
}

/// Fold one `content_block_delta.delta` into a preserved non-text block
/// (R883-B13). Every string field except `type` is appended to the block's
/// same-named field, a `null` or missing field counting as empty — which covers
/// `compaction_delta.content`, `thinking_delta.thinking` and
/// `signature_delta.signature` without naming each one.
///
/// @yah:ticket(R883-B13, "Direct road drops compaction_delta/thinking_delta payloads (compaction round-trips with empty summary) and sends compact_20260112 to models off the compaction list")
/// @yah:at(2026-09-27T17:58:34Z)
/// @yah:status(review)
/// @yah:assignee(agent:bundle-anthropic-ashguard)
/// @yah:parent(R883)
/// @yah:severity(high)
/// @yah:handoff("Found by R883-S12 static read (no live cred on host). BUG 1: run_anthropic_turn's content_block_delta arm handled only text_delta/input_json_delta, so a preserved non-text block kept only its content_block_start stub. Anthropic's documented compaction wire shape (platform.claude.com compaction-threshold, read 2026-09-27) is start {type:compaction,content:''} + ONE compaction_delta {content:<summary>} — so the compaction block T10/T11 preserve would have round-tripped with an EMPTY summary, and the API drops everything before a compaction block: silent total context loss on the first real compaction. Same defect for thinking blocks (thinking_delta/signature_delta dropped -> unsigned thinking echoed on the tool-loop re-issue). FIX: new fold_block_delta (anthropic.rs, above ApiTurnError) appends every string field of the delta bar `type` onto the stashed block; called from the delta arm for any block type other than text/tool_use.")
/// @yah:handoff("BUG 2: build_anthropic_body_with_tool_turns sent compact_20260112 to every claude-* model; the documented compaction list excludes Haiku 4.5 and Sonnet 4.5 (preroll anthropic-scout = Direct road + Haiku 4.5). FIX: model_supports_compaction prefix allowlist (fable-5, mythos-, opus-5, opus-4-8/7/6, sonnet-5, sonnet-4-6); clear_tool_uses still sent to all claude-*. Beta header unchanged (an unused beta flag is harmless).")
/// @yah:handoff("Test fixtures were wrong about the wire: T10's other_block_frames claimed compaction arrives whole on content_block_start with a `summary` field. Replaced with compaction_block_frames (content + compaction_delta) and fixed compaction_then_tool_use_frames; assertions now on `content`. New tests: anthropic_body_omits_compaction_for_models_off_the_compaction_list, fold_block_delta_concatenates_payload_fields_but_not_type.")
/// @yah:verify("cargo test -p yah-runner --lib anthropic: 63 passed, 0 failed (2026-09-27, camp build run 2f6746a8)")
/// @yah:assumes("Compaction field names and model list come from Anthropic docs, not a live response — R883-S12 live proof is parked pending a Direct-road credential.")
fn fold_block_delta(
    block: &mut serde_json::Map<String, serde_json::Value>,
    delta: &serde_json::Map<String, serde_json::Value>,
) {
    for (k, v) in delta {
        let Some(piece) = v.as_str() else { continue };
        if k == "type" {
            continue;
        }
        let mut s = block.get(k).and_then(|e| e.as_str()).unwrap_or_default().to_string();
        s.push_str(piece);
        block.insert(k.clone(), serde_json::Value::String(s));
    }
}

/// `cache_control` value for a breakpoint at `ttl`: `{type: ephemeral}` for
/// the 5-minute default, plus `ttl: "1h"` for [`CacheTtl::Extended`]. The
/// one spelling every system and transcript breakpoint on this wire uses, so
/// a host composing its own system blocks emits the same bytes this cell
/// puts on the transcript.
pub fn cache_control_value(ttl: CacheTtl) -> serde_json::Value {
    let mut cache_control = serde_json::json!({ "type": "ephemeral" });
    if matches!(ttl, CacheTtl::Extended) {
        cache_control["ttl"] = serde_json::Value::from("1h");
    }
    cache_control
}

/// `"429 Too Many Requests"` for 429: the status code plus its canonical
/// reason phrase, the form error messages on this wire have always carried
/// (and that [`yah_runner_core::health::classify_message`] reads).
fn status_line(status: u16) -> String {
    match http::StatusCode::from_u16(status) {
        Ok(code) => code.to_string(),
        Err(_) => status.to_string(),
    }
}

/// Append `event` to the session log, if the caller keeps one. A failed
/// append is logged and never fails the turn.
async fn log_event(
    store: Option<&dyn SessionStore>,
    session_id: &SessionId,
    event: &AgentEvent,
    what: &str,
) {
    if let Some(store) = store {
        if let Err(e) = store.append(session_id, event).await {
            tracing::warn!(error = %e, "session log: {what} append failed");
        }
    }
}

/// Structured error returned by [`run_anthropic_turn`] when the API call fails.
///
/// Carries the human-readable `message` plus structured fields for session-log
/// telemetry. The caller (sessions.rs) maps this into [`AgentEvent::Error`] so
/// the renderer can show the full error context without re-parsing a string.
///
/// `Debug` is derived so callers can `unwrap`/`expect` a `Result<_,
/// ApiTurnError>` and see the structured fields, not just the `Display`
/// summary — the raw body and `error_type` are usually what identify the
/// failure. R833-F13.
#[derive(Debug)]
pub struct ApiTurnError {
    /// Human-readable summary (same content as was formerly embedded in the
    /// `Result<(), String>` error string). Still forwarded to `message` in the
    /// emitted `AgentEvent::Error` so existing classifier / health logic works
    /// unchanged on the string form.
    pub message: String,
    /// HTTP status code, when the failure was a non-2xx API response.
    pub http_status: Option<u16>,
    /// First 2 KB of the raw response body — enough to capture the full
    /// Anthropic error JSON without risking unbounded allocation.
    pub raw_body: Option<String>,
    /// `error.type` from the provider JSON, e.g. `"context_length_exceeded"`.
    pub error_type: Option<String>,
    /// Usage from the error response body when Anthropic includes it (common
    /// on context-overflow errors). Callers emit this as `TurnUsageDelta`
    /// before the `Error` event so `inputTokens` is visible at failure time.
    pub usage: Option<TurnUsage>,
}

impl fmt::Display for ApiTurnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

/// One `tool_use` content block, reassembled from the SSE frames that carry
/// it. `id` and `name` land whole on `content_block_start`; the arguments
/// arrive as a run of `input_json_delta.partial_json` string fragments that
/// are only valid JSON once concatenated, so they're buffered raw and parsed
/// at `content_block_stop`. R833-F13.
#[derive(Debug, Clone)]
struct PendingToolUse {
    id: String,
    name: String,
    input_json: String,
}

impl PendingToolUse {
    /// The tool's arguments as JSON.
    ///
    /// An empty buffer means the model declared no arguments — Anthropic sends
    /// no `input_json_delta` at all for a zero-arg tool — which is `{}`, not a
    /// parse failure. Anything else that fails to parse rides through as
    /// `Value::Null`; [`ToolRegistry::execute`] implementations
    /// see a null args value and report it in-band, which is what lets the
    /// model correct itself on the next iteration instead of the turn dying.
    fn parsed_input(&self) -> serde_json::Value {
        let trimmed = self.input_json.trim();
        if trimmed.is_empty() {
            return serde_json::json!({});
        }
        serde_json::from_str(trimmed).unwrap_or(serde_json::Value::Null)
    }
}

/// Drive one turn against `messages_url` (a `/v1/messages`-shaped endpoint)
/// with `stream: true`. Folds SSE chunks into [`AgentEvent`]s emitted
/// through `sink` and appended to `store` when one is given. On success,
/// hands the assistant turn to [`HaHost::push_history`]. `auth` selects
/// between the `anthropic` (HAk, `x-api-key`) and `crab` (HAo,
/// `Authorization: Bearer` + `anthropic-beta`) presets via
/// [`AnthropicAuth::apply`].
///
/// Everything the turn needs from its session comes from `host`:
/// [`HaHost::turn_input`] at turn start and again before each tool-loop
/// re-issue, [`HaHost::observe`] after each emitted text/terminal event,
/// [`HaHost::set_phase`] on block boundaries. The host owns the session;
/// this function owns only the turn.
///
/// `messages_url` comes from the session's `Connection.base_url` (suffixed
/// with `/v1/messages`). Pass [`ANTHROPIC_MESSAGES_URL`] for the canonical
/// Anthropic endpoint when no connection override is configured.
///
/// ## Tool use (R833-F13)
///
/// When the turn input carries a [`TurnInput::tool_registry`], the request
/// body advertises its schemas and this function runs the full function-calling
/// loop: a `stop_reason: "tool_use"` response is answered by dispatching every
/// `tool_use` block through [`ToolRegistry::execute`] and re-issuing the
/// request with the matching `tool_result` blocks appended, up to
/// [`MAX_TOOL_ITERATIONS`] round trips. Each dispatch emits
/// [`AgentEvent::ToolCall`] / [`AgentEvent::ToolResult`] so the renderer and
/// the JSONL see the same interleaving the openai-compat path produces.
///
/// The whole loop is ONE turn: `TurnStarted` fires once before the first
/// request and exactly one terminal event (`TurnEnded` / `TurnFailed` /
/// `TurnBlocked`) fires after the last.
///
/// The `tool_use` / `tool_result` blocks live in a turn-local vector, NOT in
/// the host's history — `history` is `Vec<Message>` and
/// `Message.content` is a plain `String`, which cannot hold content blocks.
/// So an agent sees its own tool calls within a turn but not across turns.
/// That gap is **R833-B14**, not an oversight here.
pub async fn run_anthropic_turn(
    host: &dyn HaHost,
    sink: &dyn SessionEventSink,
    store: Option<&dyn SessionStore>,
    http: &dyn HttpClient,
    auth: &AnthropicAuth,
    messages_url: &str,
) -> Result<(), ApiTurnError> {
    let input = host.turn_input(auth).await;
    let session_id = input.session_id.clone();
    let tools = input.tool_registry.clone();
    let cancel = input.cancel.clone();
    let body = build_anthropic_body(&input, auth);
    // Wire→registry name map for routing the model's echoed `tool_use.name`
    // back to the namespaced registry name (`board_show` → `board.show`).
    let (_, tool_name_map) = anthropic_tools_payload(tools.as_ref());

    // R562-F5 Direct road: the host may capture body + auth + URL for a
    // wake-ahead replay (yah's DirectKeepalive). Fired for the FIRST
    // iteration's body only: the replay is a wake-ahead of the user's turn,
    // and re-capturing mid-tool-loop would park a half-finished tool
    // exchange as the thing to replay.
    host.request_built(&session_id, &body, auth, messages_url);

    let turn_started = AgentEvent::TurnStarted {
        session_id: session_id.clone(),
    };
    log_event(store, &session_id, &turn_started, "TurnStarted").await;
    sink.emit(&turn_started);
    host.observe(&turn_started).await;

    // ── Turn-level accumulators (survive across tool-loop iterations) ──
    //
    // Once we're inside the stream loop, any failure carries the partial
    // accumulator and emits as `TurnFailed` (symmetric to the success-path
    // `TurnEnded`). Pre-stream failures above this point return `Err` so the
    // caller emits `Error` — there's no accumulator to flush.
    let mut accumulated = String::new();
    let mut stop_reason: Option<String> = None;
    // R883-T10: the final (non-tool-call) iteration's raw content blocks,
    // captured so the assistant turn pushed to the host's history below can
    // carry them verbatim — this is what lets a compaction block (or a
    // thinking block, or any other block this cell doesn't specially
    // handle) survive into the next request instead of being flattened to
    // `accumulated`'s plain text. Left empty on the synthetic
    // exploration-budget cutoff path (see `MAX_TOOL_ITERATIONS` below),
    // which pushes plain text on purpose — there's no real provider
    // response to preserve blocks from on that path.
    let mut final_blocks: Vec<serde_json::Value> = Vec::new();
    // R883-T10 (follow-up): a compaction block can arrive on ANY iteration
    // of a multi-request tool loop, not just the turn's last one — a long
    // leader/courier turn that calls a tool almost always has one. Track the
    // latest one seen across the whole turn so the history push below can
    // still preserve it even when the turn's real final iteration (the one
    // `final_blocks` captures) never carried one itself.
    let mut latest_compaction_block: Option<serde_json::Value> = None;
    // Soft cap is turn-level, not request-level: a tool loop that re-issues
    // 24 times must not get 24× the byte budget.
    let mut total_bytes: usize = 0;
    // Anthropic reports usage per REQUEST, so a multi-iteration turn's real
    // cost is the sum. `merge_anthropic_usage` overwrites (later value wins),
    // which
    // is right within one request and wrong across them — so each iteration
    // merges into a fresh `iter_usage` and folds that into the turn total via
    // `TurnUsage::accumulate` (which sums tokens and maxes peak_context).
    let mut usage_acc: TurnUsage = TurnUsage::default();
    let mut have_usage = false;
    // R833-F13: this turn's tool round trips, in wire order. See
    // `build_anthropic_body_with_tool_turns`.
    let mut tool_turns: Vec<serde_json::Value> = Vec::new();
    let mut iterations: usize = 0;
    let mut total_tool_calls: u32 = 0;
    let mut body = body;

    macro_rules! turn_failed {
        ($message:expr) => {{
            let ev = AgentEvent::TurnFailed {
                session_id: session_id.clone(),
                text: accumulated.clone(),
                message: $message,
            };
            log_event(store, &session_id, &ev, "TurnFailed").await;
            sink.emit(&ev);
            host.observe(&ev).await;
            return Ok(());
        }};
    }

    // A pre-stream failure on the FIRST request returns `Err` so the caller
    // emits `AgentEvent::Error` — nothing has streamed, so there's no partial
    // turn to close out. On a tool-loop re-issue (iteration > 0) text and tool
    // results have already reached the FE, so the same failure has to close the
    // turn as `TurnFailed` instead; returning `Err` there would leave the
    // renderer with a turn that started, emitted, and never terminated.
    // Mirrors openai_compat's in-loop-vs-initial split.
    macro_rules! bail_pre_stream {
        ($err:expr) => {{
            let err: ApiTurnError = $err;
            if iterations == 0 {
                return Err(err);
            }
            turn_failed!(err.message)
        }};
    }

    // R833-F13: one pass per provider request. Without a registry this runs
    // exactly once and the body below is identical to the pre-F13 one.
    'tool_loop: loop {
    // Pre-stream retry loop (R237-F11): transient network failures and
    // retryable HTTP statuses (429 / 5xx) are re-attempted up to
    // [`MAX_STREAM_OPEN_ATTEMPTS`] times with exponential backoff. The
    // retry boundary is the bytes_stream() opening below — once a single
    // chunk has been read, partial output may have been emitted to the
    // FE and re-issuing would double-stream. Anthropic SSE has no
    // event-id, so frame-level resume is not possible.
    let mut last_retry_after_secs: Option<u32> = None;
    let mut resp_opt: Option<yah_runner_core::HttpResponse> = None;
    let mut final_err: Option<ApiTurnError> = None;
    for attempt in 0..MAX_STREAM_OPEN_ATTEMPTS {
        if attempt > 0 {
            let dur = std::time::Duration::from_millis(compute_stream_open_backoff_ms(
                attempt,
                last_retry_after_secs,
            ));
            tokio::select! {
                _ = tokio::time::sleep(dur) => {}
                _ = cancel.cancelled() => {
                    bail_pre_stream!(final_err.unwrap_or_else(|| ApiTurnError {
                        message: "Anthropic stream-open cancelled during retry backoff".into(),
                        http_status: None,
                        raw_body: None,
                        error_type: None,
                        usage: None,
                    }))
                }
            }
        }

        let mut req = HttpRequest::post_json(messages_url, &body);
        if let Some(beta) = context_editing_beta_header(auth, &body) {
            req = req.header("anthropic-beta", beta);
        }
        let send_result = http.send(auth.apply(req)).await;
        let resp = match send_result {
            Ok(r) => r,
            Err(e) => {
                // Transport-level failure (DNS, TLS, connect, etc.) — always
                // treat as transient Network. Retry if attempts remain.
                let err = ApiTurnError {
                    message: format!("Anthropic request failed: {}", e),
                    http_status: None,
                    raw_body: None,
                    error_type: None,
                    usage: None,
                };
                final_err = Some(err);
                last_retry_after_secs = None;
                if attempt + 1 < MAX_STREAM_OPEN_ATTEMPTS {
                    continue;
                }
                bail_pre_stream!(final_err.expect("just set"))
            }
        };

        if resp.is_success() {
            resp_opt = Some(resp);
            break;
        }

        let status = resp.status;
        let retry_after_secs = resp
            .header("retry-after")
            .and_then(|s| s.parse::<u32>().ok());
        let body_text = resp.text().await.unwrap_or_default();

        let body_json = serde_json::from_str::<serde_json::Value>(&body_text).ok();
        let error_type = body_json
            .as_ref()
            .and_then(|v| v.get("error"))
            .and_then(|e| e.get("type"))
            .and_then(|t| t.as_str())
            .map(|s| s.to_string());
        // F2: Anthropic sometimes includes usage in error responses (e.g.
        // context_length_exceeded). Capture it so the caller can emit
        // TurnUsageDelta before the Error event.
        let error_usage = body_json
            .as_ref()
            .and_then(|v| v.get("usage").cloned())
            .map(|u| {
                let mut acc = TurnUsage::default();
                merge_anthropic_usage(&mut acc, &u);
                acc
            });

        let raw_body: Option<String> = if body_text.is_empty() {
            None
        } else {
            Some(body_text.chars().take(2048).collect())
        };
        let mut msg = format!(
            "Anthropic returned {}: {}",
            status_line(status),
            body_text.chars().take(512).collect::<String>()
        );
        if let Some(secs) = retry_after_secs {
            msg.push_str(&format!(" [retry-after: {secs}s]"));
        }
        let err = ApiTurnError {
            message: msg,
            http_status: Some(status),
            raw_body,
            error_type,
            usage: error_usage,
        };

        if is_retryable_status(status) && attempt + 1 < MAX_STREAM_OPEN_ATTEMPTS {
            final_err = Some(err);
            last_retry_after_secs = retry_after_secs;
            continue;
        }
        bail_pre_stream!(err)
    }
    let resp = resp_opt.expect(
        "loop above breaks with Some(resp) on success or returns Err on permanent failure",
    );

    host.response_headers(&resp.headers, auth).await;

    // ── Per-iteration stream state (reset on every provider request) ──
    let mut buffer = String::new();
    // Anthropic emits usage twice: input + cache numbers ride
    // `message_start.message.usage` (with output_tokens=1 baseline);
    // the running `output_tokens` rides each `message_delta.usage`
    // until end-of-stream. We accumulate both into one struct — per
    // request, then folded into the turn total below.
    let mut iter_usage: TurnUsage = TurnUsage::default();
    let mut text_filter = yah_runner_core::text_filter::TextStreamFilter::new();
    // Per-block tracking for AssistantText persistence. Anthropic's SSE
    // protocol emits content_block_start / content_block_stop boundaries
    // even across text→tool_use→text sequences, so we persist one
    // AssistantText event per text block (at block_stop) to give replay
    // correct text-vs-tool-call interleaving. accumulated keeps the
    // turn-level aggregate for TurnEnded.text convenience.
    let mut current_block_type: Option<String> = None;
    let mut current_text_block = String::new();
    // R883-T10: verbatim `content_block` from the in-flight block's
    // `content_block_start`, for any type this cell has no bespoke field
    // tracking for (compaction, thinking, ...) — see content_block_stop.
    let mut current_other_block: Option<serde_json::Value> = None;
    // R833-F13: this request's assistant content blocks in wire order, plus
    // the tool_use blocks broken out for dispatch. Scoped to THIS request:
    // the assistant message we echo back must reproduce the message the
    // provider actually emitted, not the whole turn's aggregate.
    let mut iter_blocks: Vec<serde_json::Value> = Vec::new();
    let mut iter_calls: Vec<PendingToolUse> = Vec::new();
    let mut current_tool_use: Option<PendingToolUse> = None;

    let mut stream = resp.body;
    while let Some(chunk) = stream.next().await {
        let bytes = match chunk {
            Ok(b) => b,
            Err(e) => turn_failed!(format!("stream read failed: {}", e)),
        };
        total_bytes = total_bytes.saturating_add(bytes.len());
        if total_bytes > MAX_STREAM_BYTES {
            turn_failed!(format!(
                "stream exceeded soft cap of {} bytes — aborting",
                MAX_STREAM_BYTES
            ));
        }
        let text = match std::str::from_utf8(&bytes) {
            Ok(s) => s,
            Err(e) => turn_failed!(format!("non-utf8 chunk: {}", e)),
        };
        buffer.push_str(text);

        // SSE frames are `data: {...}\n\n`-separated. Pull complete
        // frames out of the buffer; leave any half-frame for the next
        // iteration.
        while let Some(idx) = buffer.find("\n\n") {
            let frame = buffer[..idx].to_string();
            buffer.drain(..idx + 2);
            for line in frame.lines() {
                let Some(payload) = line.strip_prefix("data: ") else {
                    continue;
                };
                if payload == "[DONE]" {
                    continue;
                }
                let Ok(event) = serde_json::from_str::<serde_json::Value>(payload) else {
                    continue;
                };
                match event.get("type").and_then(|v| v.as_str()) {
                    Some("message_start") => {
                        if let Some(usage) = event.get("message").and_then(|m| m.get("usage")) {
                            merge_anthropic_usage(&mut iter_usage, usage);
                            have_usage = true;
                        }
                    }
                    Some("content_block_start") => {
                        let block_type = event
                            .get("content_block")
                            .and_then(|b| b.get("type"))
                            .and_then(|t| t.as_str());
                        current_block_type = block_type.map(|s| s.to_string());
                        // R883-T10: stash the block verbatim for any type
                        // this cell has no bespoke handling for (compaction,
                        // thinking, redacted_thinking, ...) so it survives
                        // into `iter_blocks` at content_block_stop instead of
                        // being silently dropped — compaction requires this
                        // block to round-trip byte-for-byte.
                        current_other_block = event.get("content_block").cloned();
                        if block_type == Some("text") {
                            current_text_block.clear();
                            // R258-F3: text block starting → session is emitting.
                            host.set_phase(HaPhase::Emitting).await;
                        } else if block_type == Some("tool_use") {
                            // Flush tail, stripping any trailing "sentence:"
                            let tail = text_filter.on_tool_use_start();
                            if !tail.is_empty() {
                                let ev = AgentEvent::MessageDelta {
                                    session_id: session_id.clone(),
                                    text: tail,
                                };
                                sink.emit(&ev);
                                host.observe(&ev).await;
                            }
                            // R833-F13: open the call. `id` and `name` arrive on
                            // content_block_start; the arguments stream in as
                            // `input_json_delta.partial_json` fragments and are
                            // only parseable once the block stops.
                            let cb = event.get("content_block");
                            current_tool_use = Some(PendingToolUse {
                                id: cb
                                    .and_then(|b| b.get("id"))
                                    .and_then(|v| v.as_str())
                                    .unwrap_or_default()
                                    .to_string(),
                                name: cb
                                    .and_then(|b| b.get("name"))
                                    .and_then(|v| v.as_str())
                                    .unwrap_or_default()
                                    .to_string(),
                                input_json: String::new(),
                            });
                            // R258-F3: tool_use block starting → session is executing a tool.
                            host.set_phase(HaPhase::ToolUse).await;
                        }
                    }
                    Some("content_block_stop") => {
                        let tail = text_filter.flush();
                        if !tail.is_empty() {
                            let ev = AgentEvent::MessageDelta {
                                session_id: session_id.clone(),
                                text: tail,
                            };
                            sink.emit(&ev);
                            host.observe(&ev).await;
                        }
                        if current_block_type.as_deref() == Some("text")
                            && !current_text_block.is_empty()
                        {
                            let text = std::mem::take(&mut current_text_block);
                            iter_blocks.push(serde_json::json!({
                                "type": "text",
                                "text": text,
                            }));
                            let ev = AgentEvent::AssistantText {
                                session_id: session_id.clone(),
                                text,
                            };
                            log_event(store, &session_id, &ev, "AssistantText").await;
                        }
                        if let Some(call) = current_tool_use.take() {
                            iter_blocks.push(serde_json::json!({
                                "type": "tool_use",
                                "id": call.id,
                                "name": call.name,
                                "input": call.parsed_input(),
                            }));
                            iter_calls.push(call);
                        } else if !matches!(current_block_type.as_deref(), Some("text") | None) {
                            // Any other block type (compaction, thinking,
                            // redacted_thinking, ...): preserve verbatim.
                            if let Some(block) = current_other_block.take() {
                                iter_blocks.push(block);
                            }
                        }
                        current_other_block = None;
                        current_block_type = None;
                    }
                    Some("content_block_delta") => {
                        if let Some(text) = event
                            .get("delta")
                            .and_then(|d| d.get("text"))
                            .and_then(|t| t.as_str())
                        {
                            // `accumulated` holds the raw text for session history.
                            // The sink receives the normalized/filtered form.
                            accumulated.push_str(text);
                            current_text_block.push_str(text);
                            let to_emit = text_filter.push_text(text);
                            if !to_emit.is_empty() {
                                let ev = AgentEvent::MessageDelta {
                                    session_id: session_id.clone(),
                                    text: to_emit,
                                };
                                sink.emit(&ev);
                                host.observe(&ev).await;
                            }
                        }
                        // R833-F13: tool arguments stream as JSON fragments on
                        // `input_json_delta`. Concatenate raw; parse at block_stop.
                        if let Some(partial) = event
                            .get("delta")
                            .and_then(|d| d.get("partial_json"))
                            .and_then(|t| t.as_str())
                        {
                            if let Some(call) = current_tool_use.as_mut() {
                                call.input_json.push_str(partial);
                            }
                        }
                        // R883-B13: a preserved non-text block's payload does NOT
                        // arrive on content_block_start — it streams as deltas
                        // (`compaction_delta.content`, `thinking_delta.thinking`,
                        // `signature_delta.signature`). Fold every string field of
                        // the delta (bar `type`) onto the same-named field of the
                        // stashed block, or the block round-trips with an empty
                        // summary / unsigned thinking.
                        if !matches!(current_block_type.as_deref(), Some("text") | Some("tool_use") | None) {
                            if let (Some(block), Some(delta)) = (
                                current_other_block.as_mut().and_then(|b| b.as_object_mut()),
                                event.get("delta").and_then(|d| d.as_object()),
                            ) {
                                fold_block_delta(block, delta);
                            }
                        }
                    }
                    Some("message_delta") => {
                        if let Some(reason) = event
                            .get("delta")
                            .and_then(|d| d.get("stop_reason"))
                            .and_then(|r| r.as_str())
                        {
                            stop_reason = Some(reason.to_string());
                        }
                        // message_delta.usage carries the running
                        // output_tokens count. Final emission before
                        // message_stop replaces our accumulator's
                        // output_tokens with the canonical total.
                        if let Some(usage) = event.get("usage") {
                            merge_anthropic_usage(&mut iter_usage, usage);
                            have_usage = true;
                        }
                    }
                    Some("error") => {
                        let msg = event
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                            .unwrap_or("anthropic error")
                            .to_string();
                        turn_failed!(msg);
                    }
                    _ => {}
                }
            }
        }
    }

    // Flush any tail the filter held back (e.g. stream ended without a
    // content_block_stop).
    {
        let tail = text_filter.flush();
        if !tail.is_empty() {
            let ev = AgentEvent::MessageDelta {
                session_id: session_id.clone(),
                text: tail,
            };
            sink.emit(&ev);
            host.observe(&ev).await;
        }
    }

    // Per-request usage folds into the turn total. Token counts SUM (a turn
    // that made four requests really did pay for four prompts), but the
    // context high-water mark is a MAX — so derive this request's peak here,
    // while `iter_usage` still describes one call, and let
    // `TurnUsage::accumulate` max it in. Deriving it after summing would
    // report the sum of every request's prompt as if one prompt were that
    // large. R833-F13.
    let iter_peak = iter_usage.input_tokens.unwrap_or(0)
        + iter_usage.cache_read_tokens.unwrap_or(0)
        + iter_usage.cache_write_tokens.unwrap_or(0);
    if iter_peak > 0 {
        iter_usage.peak_context_tokens = Some(iter_peak);
    }
    usage_acc.accumulate(&iter_usage);

    // R883-T10 (follow-up): capture this iteration's compaction block, if
    // any, before the tool-dispatch branch below decides what to do with
    // `iter_blocks` — this scan must run unconditionally, once per
    // iteration, whichever path (tool call or natural end) this iteration
    // takes.
    if let Some(compaction) = iter_blocks
        .iter()
        .find(|b| b.get("type").and_then(|t| t.as_str()) == Some("compaction"))
    {
        latest_compaction_block = Some(compaction.clone());
    }

    // ── R833-F13 tool dispatch ──────────────────────────────────────────
    //
    // Anthropic signals "I want to call tools" with `stop_reason: "tool_use"`.
    // Trust the emitted blocks over the stop_reason (a truncated stream can
    // carry blocks without one), but require at least one block either way —
    // a bare `tool_use` stop with nothing to dispatch would spin the loop.
    if !iter_calls.is_empty() {
        let Some(registry) = tools.as_ref() else {
            // Defensive: with no registry we never sent `tools[]`, so the
            // model had nothing to call. Protocol violation, not a retry.
            turn_failed!(
                "provider emitted tool_use blocks but no tool registry is attached to this session"
                    .to_string()
            );
        };

        // Echo the assistant turn back verbatim — Anthropic pairs each
        // `tool_result.tool_use_id` against the `tool_use` block in the
        // preceding assistant message, so it has to be replayed exactly.
        tool_turns.push(serde_json::json!({
            "role": "assistant",
            "content": iter_blocks,
        }));

        let mut result_blocks: Vec<serde_json::Value> = Vec::with_capacity(iter_calls.len());
        for call in &iter_calls {
            // The model echoes the wire name we advertised (dots rewritten to
            // underscores); map back so dispatch and observability both see
            // the namespaced registry name, e.g. `board.show`.
            let registry_name = tool_name_map
                .get(&call.name)
                .cloned()
                .unwrap_or_else(|| call.name.clone());
            let args = call.parsed_input();
            sink.emit(&AgentEvent::ToolCall {
                session_id: session_id.clone(),
                tool_call_id: call.id.clone(),
                tool_name: registry_name.clone(),
                args: args.clone(),
            });
            let outcome = registry.execute(&registry_name, args).await;
            total_tool_calls += 1;
            let result_ev = AgentEvent::ToolResult {
                session_id: session_id.clone(),
                tool_call_id: call.id.clone(),
                ok: outcome.ok,
                result: outcome.result.clone(),
            };
            log_event(store, &session_id, &result_ev, "ToolResult").await;
            sink.emit(&result_ev);
            // `content` is a string on the wire; a failed tool rides back as
            // `is_error: true` so the model can correct rather than treating
            // the JSON as a successful payload. A tool's own failure is never
            // a runner error — see `ToolOutcome`. Capped at ingestion
            // (R883-T1); the event above keeps the full result.
            result_blocks.push(serde_json::json!({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": yah_runner_core::tool_result_wire_content(&outcome.result),
                "is_error": !outcome.ok,
            }));
        }
        tool_turns.push(serde_json::json!({
            "role": "user",
            "content": result_blocks,
        }));

        iterations += 1;
        if iterations >= MAX_TOOL_ITERATIONS {
            // Soft stop, not an error: the model isn't stuck, it spent the
            // turn's budget. Synthesize a closing assistant turn so the
            // transcript reads naturally and the user can just say "continue".
            // Mirrors openai_compat's `exploration_budget` close-out.
            let notice = format!(
                "I've reached this turn's tool-call budget ({MAX_TOOL_ITERATIONS} iterations, \
                 {total_tool_calls} tool calls). Say \"continue\" to keep going."
            );
            let ev = AgentEvent::AssistantText {
                session_id: session_id.clone(),
                text: notice.clone(),
            };
            log_event(store, &session_id, &ev, "AssistantText").await;
            sink.emit(&ev);
            if !accumulated.is_empty() {
                accumulated.push_str("\n\n");
            }
            accumulated.push_str(&notice);
            stop_reason = Some("exploration_budget".into());
            break 'tool_loop;
        }

        // Re-issue with the tool results appended.
        body = {
            let input = host.turn_input(auth).await;
            build_anthropic_body_with_tool_turns(&input, auth, &tool_turns)
        };
        continue 'tool_loop;
    }

    // Natural end: this iteration's response is the turn's real final
    // assistant message. Capture its raw blocks (text, and anything else
    // this cell preserved verbatim — never tool_use, since
    // !iter_calls.is_empty() would have taken the branch above) for the
    // history push below.
    final_blocks = std::mem::take(&mut iter_blocks);
    break 'tool_loop;
    }

    // R883-T10: a single-request turn (no tool_use round trip) pushes the
    // final iteration's raw blocks verbatim — the common case, and the one
    // whose whole content compaction can safely replace.
    //
    // A turn that made ≥1 tool call already collapses its several requests'
    // text into `accumulated` (R833-B14: intermediate tool_use/tool_result
    // exchanges are turn-local, never persisted — a tool_use block
    // persisted without its paired tool_result would be invalid history on
    // replay), so pushing just the last iteration's blocks there would
    // silently drop every earlier iteration's visible text. But long
    // leader/courier turns are exactly the ones that both call tools AND
    // run long enough to hit compaction, so dropping a compaction block on
    // this path would mean it almost never survives in practice. Persist it
    // explicitly instead: [the latest compaction block seen across any
    // iteration of this turn] + [one text block holding the full
    // accumulated text] — never the tool_use/thinking blocks, which stay
    // turn-local as before.
    let history_content = if iterations == 0 && !final_blocks.is_empty() {
        MessageContent::Blocks(final_blocks.clone())
    } else if let Some(compaction) = latest_compaction_block.clone() {
        MessageContent::Blocks(vec![
            compaction,
            serde_json::json!({ "type": "text", "text": accumulated.clone() }),
        ])
    } else {
        MessageContent::Text(accumulated.clone())
    };
    // R883-T11: carry the same blocks onto the durable TurnEnded event so
    // reconstruct_claude_history can rebuild Blocks verbatim on restart
    // instead of collapsing to Text.
    let content_blocks = match &history_content {
        MessageContent::Blocks(blocks) => Some(blocks.clone()),
        MessageContent::Text(_) => None,
    };

    host.push_history(Message {
        role: Role::Assistant,
        content: history_content,
    })
    .await;

    // Anthropic 4.5+ models surface context overflow via stop_reason rather
    // than a 400. Detect it here and emit TurnBlocked so the FE shows the
    // recovery modal instead of parking the session as waiting-for-input.
    if let Some(block_reason) = stop_reason
        .as_deref()
        .and_then(yah_runner_core::block_detector::anthropic_stream_end)
    {
        let blocked_ev = AgentEvent::TurnBlocked {
            session_id: session_id.clone(),
            text: accumulated,
            reason: block_reason.clone(),
        };
        log_event(store, &session_id, &blocked_ev, "TurnBlocked").await;
        sink.emit(&blocked_ev);
        host.observe(&blocked_ev).await;
        host.set_phase(HaPhase::Blocked(block_reason)).await;
        return Ok(());
    }

    let turn_ended = AgentEvent::TurnEnded {
        session_id: session_id.clone(),
        text: accumulated,
        stop_reason,
        // `peak_context_tokens` was set per request inside the loop above and
        // maxed across iterations — a single-request turn gets exactly the
        // pre-R833-F13 value (input + cache_read + cache_write).
        usage: have_usage.then_some(usage_acc),
        content_blocks,
    };
    log_event(store, &session_id, &turn_ended, "TurnEnded").await;
    sink.emit(&turn_ended);
    host.observe(&turn_ended).await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::SystemPrompt;
    use tokio::sync::Mutex;
    use yah_runner_stub::anthropic_wire::MessagesTurn;
    use yah_runner_core::{Message, Role, SessionId, ToolOutcome, ToolSchema};

    /// A one-block system prompt cached at the 5m default — the shape a host
    /// with no composition layer sends.
    fn fixture_system(breakpoints: usize) -> SystemPrompt {
        let blocks = (0..breakpoints.max(1))
            .map(|i| {
                let mut b = serde_json::json!({ "type": "text", "text": format!("system block {i}") });
                if i < breakpoints {
                    b["cache_control"] = cache_control_value(CacheTtl::Ephemeral);
                }
                b
            })
            .collect();
        SystemPrompt { blocks, cache_ttl: CacheTtl::Ephemeral }
    }

    fn fixture_input(model: &str, think: Option<ReasoningBudget>) -> TurnInput {
        TurnInput {
            session_id: SessionId::new("session:deadbeef"),
            model: model.into(),
            think,
            system: fixture_system(1),
            history: vec![Message {
                role: Role::User,
                content: "hi".into(),
            }],
            tool_registry: None,
            cancel: tokio_util::sync::CancellationToken::new(),
        }
    }

    /// In-memory [`HaHost`] over a [`TurnInput`]: hands back the same input
    /// (with whatever history the turn has pushed so far) on every call.
    struct TestHost {
        input: TurnInput,
        history: Mutex<Vec<Message>>,
    }

    impl TestHost {
        fn new(input: TurnInput) -> Self {
            let history = Mutex::new(input.history.clone());
            Self { input, history }
        }
    }

    #[async_trait::async_trait]
    impl HaHost for TestHost {
        async fn turn_input(&self, _auth: &AnthropicAuth) -> TurnInput {
            TurnInput {
                session_id: self.input.session_id.clone(),
                model: self.input.model.clone(),
                think: self.input.think.clone(),
                system: self.input.system.clone(),
                history: self.history.lock().await.clone(),
                tool_registry: self.input.tool_registry.clone(),
                cancel: self.input.cancel.clone(),
            }
        }
        async fn push_history(&self, message: Message) {
            self.history.lock().await.push(message);
        }
    }

    async fn run_turn(host: &TestHost, sink: &dyn SessionEventSink, url: &str) -> Result<(), ApiTurnError> {
        let http = yah_runner_core::ReqwestClient::default();
        run_anthropic_turn(host, sink, None, &http, &AnthropicAuth::ApiKey("k".into()), url).await
    }

    /// Registry whose `schemas()` is whatever the test hands it and whose
    /// `execute()` records the calls it saw, so a test can assert both what
    /// went out on the wire and what came back through dispatch.
    struct StubRegistry {
        schemas: Vec<ToolSchema>,
        calls: std::sync::Mutex<Vec<(String, serde_json::Value)>>,
    }

    impl StubRegistry {
        fn new(schemas: Vec<ToolSchema>) -> Self {
            Self {
                schemas,
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl ToolRegistry for StubRegistry {
        fn schemas(&self) -> Vec<ToolSchema> {
            self.schemas.clone()
        }
        async fn execute(
            &self,
            name: &str,
            args: serde_json::Value,
        ) -> ToolOutcome {
            self.calls
                .lock()
                .unwrap()
                .push((name.to_string(), args.clone()));
            ToolOutcome::ok(serde_json::json!({ "echo": args }))
        }
    }

    fn stub_schema(name: &str) -> ToolSchema {
        ToolSchema {
            name: name.into(),
            description: format!("{name} does a thing"),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
            }),
        }
    }

    #[test]
    fn build_body_omits_tools_when_session_has_no_registry() {
        // The pre-R833-F13 shape, and what every anonymous / CLI session
        // still sends: no `tools` key at all, so the model is told nothing
        // about tools and should never emit a tool_use block.
        let session = fixture_input("claude-sonnet-4-5", None);
        let body = build_anthropic_body(&session, &AnthropicAuth::ApiKey("k".into()));
        assert!(
            body.get("tools").is_none(),
            "a registry-less session must not carry a tools key"
        );
    }

    #[test]
    fn build_body_serializes_registry_schemas_into_anthropic_shaped_tools() {
        let mut session = fixture_input("claude-sonnet-4-5", None);
        session.tool_registry = Some(Arc::new(StubRegistry::new(vec![
            stub_schema("read_file"),
            stub_schema("board.show"),
        ])));
        let body = build_anthropic_body(&session, &AnthropicAuth::ApiKey("k".into()));

        let tools = body["tools"].as_array().expect("tools array present");
        assert_eq!(tools.len(), 2);
        // Anthropic takes the schema flat, with `input_schema` — NOT OpenAI's
        // `{type: "function", function: {..., parameters}}` envelope.
        assert_eq!(tools[0]["name"], "read_file");
        assert_eq!(tools[0]["description"], "read_file does a thing");
        assert_eq!(tools[0]["input_schema"]["type"], "object");
        assert!(tools[0].get("type").is_none(), "no OpenAI function envelope");
        assert!(tools[0].get("parameters").is_none(), "field is input_schema");
        // Dotted registry names are rewritten to satisfy ^[a-zA-Z0-9_-]+$.
        assert_eq!(tools[1]["name"], "board_show");
    }

    #[test]
    fn tools_payload_maps_rewritten_wire_names_back_to_the_registry() {
        // The model echoes the wire name; dispatch needs the registry name.
        let registry: Arc<dyn ToolRegistry> = Arc::new(StubRegistry::new(vec![
            stub_schema("board.show"),
            stub_schema("read_file"),
        ]));
        let (arr, map) = anthropic_tools_payload(Some(&registry));
        assert_eq!(arr.expect("schemas present").len(), 2);
        assert_eq!(map.get("board_show").map(String::as_str), Some("board.show"));
        assert!(
            !map.contains_key("read_file"),
            "already-conformant names stay out of the map"
        );
    }

    #[test]
    fn tools_payload_is_absent_for_an_attached_but_empty_registry() {
        // `tools: []` reads to providers as "tools allowed, none offered",
        // which is worse than omitting the key.
        let registry: Arc<dyn ToolRegistry> = Arc::new(StubRegistry::new(vec![]));
        let (arr, map) = anthropic_tools_payload(Some(&registry));
        assert!(arr.is_none());
        assert!(map.is_empty());
    }

    #[test]
    fn tool_turns_append_behind_the_rolling_cache_breakpoint() {
        // R562-T3's breakpoint budget is 4. The tool loop must not spend any
        // of it: tool turns append AFTER the marked history tail, so the
        // breakpoint count is identical with and without them. R883-F3
        // extends this to a Hermes-style multi-message window, so re-check
        // the invariant for both a leader job (system caps at ≤3) and a
        // one-shot job (system now caps at 2, R883-F3) — the count must stay
        // <= 4 either way, with and without tool turns.
        let count_breakpoints = |body: &serde_json::Value| -> usize {
            let sys = body["system"]
                .as_array()
                .expect("system blocks")
                .iter()
                .filter(|b| b.get("cache_control").is_some())
                .count();
            let msgs = body["messages"]
                .as_array()
                .expect("messages")
                .iter()
                .filter(|m| {
                    m["content"].as_array().is_some_and(|blocks| {
                        blocks.iter().any(|b| b.get("cache_control").is_some())
                    })
                })
                .count();
            sys + msgs
        };

        // The host decides the system prompt's share of the budget (yah's
        // leader and one-shot layouts spend 2-3); re-check the invariant for
        // each share the cell can be handed.
        for sys_bp in [1usize, 2, 3] {
            let mut session = fixture_input("claude-sonnet-4-5", None);
            session.system = fixture_system(sys_bp);
            session.history.push(Message {
                role: Role::Assistant,
                content: "sure".into(),
            });
            let auth = AnthropicAuth::ApiKey("k".into());

            let plain = build_anthropic_body(&session, &auth);
            let tool_turns = vec![
                serde_json::json!({ "role": "assistant", "content": [
                    { "type": "tool_use", "id": "tu_1", "name": "read_file", "input": {} },
                ]}),
                serde_json::json!({ "role": "user", "content": [
                    { "type": "tool_result", "tool_use_id": "tu_1", "content": "{}", "is_error": false },
                ]}),
            ];
            let with_tools = build_anthropic_body_with_tool_turns(&session, &auth, &tool_turns);

            let before = count_breakpoints(&plain);
            assert_eq!(
                before,
                count_breakpoints(&with_tools),
                "tool turns must not add or move a cache breakpoint (system breakpoints={sys_bp})"
            );
            assert!(before <= 4, "breakpoint budget (system breakpoints={sys_bp})");
            assert!(count_breakpoints(&with_tools) <= 4, "breakpoint budget with tool turns (system breakpoints={sys_bp})");

            // The breakpoint still sits on the last HISTORY message, with the
            // tool turns strictly behind it.
            let msgs = with_tools["messages"].as_array().unwrap();
            assert_eq!(msgs.len(), session.history.len() + 2);
            let marked = msgs[session.history.len() - 1]["content"][0]["cache_control"].clone();
            assert!(!marked.is_null(), "history tail still carries the breakpoint (system breakpoints={sys_bp})");
            assert_eq!(msgs[session.history.len()]["role"], "assistant");
            assert_eq!(msgs[session.history.len() + 1]["role"], "user");
        }
    }

    #[test]
    fn pending_tool_use_parses_streamed_argument_fragments() {
        // Anthropic streams tool args as partial_json fragments that are only
        // valid JSON once concatenated.
        let mut call = PendingToolUse {
            id: "tu_1".into(),
            name: "read_file".into(),
            input_json: String::new(),
        };
        for frag in [r#"{"pa"#, r#"th": "/tm"#, r#"p/x"}"#] {
            call.input_json.push_str(frag);
        }
        assert_eq!(call.parsed_input(), serde_json::json!({ "path": "/tmp/x" }));
    }

    #[test]
    fn pending_tool_use_treats_an_empty_buffer_as_no_arguments() {
        // A zero-arg tool gets no input_json_delta at all. That's `{}`, not a
        // parse failure — sending Null would make every zero-arg call fail.
        let call = PendingToolUse {
            id: "tu_1".into(),
            name: "camp_roster".into(),
            input_json: "   ".into(),
        };
        assert_eq!(call.parsed_input(), serde_json::json!({}));
    }

    #[test]
    fn pending_tool_use_surfaces_unparseable_arguments_as_null() {
        // Null rides through to execute(), which reports it in-band so the
        // model can correct itself rather than the turn dying.
        let call = PendingToolUse {
            id: "tu_1".into(),
            name: "read_file".into(),
            input_json: "{not json".into(),
        };
        assert_eq!(call.parsed_input(), serde_json::Value::Null);
    }

    #[tokio::test]
    async fn registry_dispatch_routes_the_wire_name_back_to_the_registry_name() {
        // The dispatch half of the rewrite: the model echoes `board_show`,
        // ToolRegistry::execute must see `board.show`.
        let stub = Arc::new(StubRegistry::new(vec![stub_schema("board.show")]));
        let registry: Arc<dyn ToolRegistry> = stub.clone();
        let (_, name_map) = anthropic_tools_payload(Some(&registry));

        let echoed = "board_show";
        let registry_name = name_map
            .get(echoed)
            .cloned()
            .unwrap_or_else(|| echoed.to_string());
        assert_eq!(registry_name, "board.show");

        let outcome = registry
            .execute(&registry_name, serde_json::json!({ "id": "R833" }))
            .await;
        assert!(outcome.ok);
        let calls = stub.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "board.show");
        assert_eq!(calls[0].1, serde_json::json!({ "id": "R833" }));
    }

    /// The shared inference stub (`yah-runner-stub`) preloaded with one scripted
    /// `/v1/messages` turn per expected request, so this exercises real reqwest
    /// streaming and real chunk boundaries rather than a hand-fed parser. Read
    /// the bodies it received off `stub.requests()`; hold the handle for the
    /// test's life — dropping it stops the server.
    ///
    /// R833-F13: the tool loop's whole contract is "what goes out on request
    /// N+1 given what came back on request N", which no pure-function test can
    /// reach.
    async fn scripted_messages(turns: Vec<MessagesTurn>) -> (String, yah_runner_stub::InferenceStub) {
        let stub = yah_runner_stub::InferenceStub::builder(yah_runner_stub::StubMode::default())
            .replies(turns)
            .spawn()
            .await
            .expect("spawn stub");
        (stub.messages_url(), stub)
    }

    /// One sentence of text, then one `tool_use` block whose arguments arrive
    /// split across three `input_json_delta` fragments, stopping with `tool_use`.
    fn tool_use_turn(wire_name: &str) -> MessagesTurn {
        MessagesTurn::text("Looking that up.")
            .with_tool_use("toolu_01", wire_name, serde_json::json!({"id": "R833"}), 3)
            .usage(100, 40)
    }

    /// A plain text response that ends the turn.
    fn final_text_turn(text: &str) -> MessagesTurn {
        MessagesTurn::text(text).usage(200, 10)
    }

    #[tokio::test]
    async fn tool_use_round_trips_through_the_registry_and_back_onto_the_wire() {
        // The R833-F13 acceptance path in miniature: a direct-API anthropic
        // turn advertises its tools, the model calls one, the registry runs
        // it, the result goes back on the next request, and the turn ends
        // with the model's follow-up text.
        let (url, stub_server) = scripted_messages(vec![
            tool_use_turn("board_show"),
            final_text_turn("R833 is in progress."),
        ])
        .await;

        let stub = Arc::new(StubRegistry::new(vec![stub_schema("board.show")]));
        let mut session = fixture_input("claude-sonnet-4-5", None);
        session.tool_registry = Some(stub.clone());
        let session = TestHost::new(session);

        let seen: Arc<std::sync::Mutex<Vec<AgentEvent>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = {
            let seen = Arc::clone(&seen);
            yah_runner_core::FnSink::new(move |e: &AgentEvent| seen.lock().unwrap().push(e.clone()))
        };

        run_turn(&session, &sink, &url)
        .await
        .expect("turn should succeed");

        // ── the wire ──
        let bodies = stub_server.requests();
        assert_eq!(bodies.len(), 2, "one request, one tool re-issue");

        // Request 1 advertises the tool under its sanitized wire name.
        let tools = bodies[0]["tools"].as_array().expect("tools advertised");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "board_show");
        assert!(tools[0]["input_schema"].is_object());

        // Request 2 replays the assistant's tool_use block verbatim, then the
        // matching tool_result — pairing is by tool_use_id, so a mismatch here
        // is a 400 from the real API.
        let msgs = bodies[1]["messages"].as_array().unwrap();
        let echo = &msgs[msgs.len() - 2];
        assert_eq!(echo["role"], "assistant");
        assert_eq!(echo["content"][0]["type"], "text");
        assert_eq!(echo["content"][0]["text"], "Looking that up.");
        assert_eq!(echo["content"][1]["type"], "tool_use");
        assert_eq!(echo["content"][1]["id"], "toolu_01");
        assert_eq!(echo["content"][1]["name"], "board_show");
        // Fragmented input_json_delta reassembled into real JSON.
        assert_eq!(echo["content"][1]["input"], serde_json::json!({"id": "R833"}));

        let result = &msgs[msgs.len() - 1];
        assert_eq!(result["role"], "user");
        assert_eq!(result["content"][0]["type"], "tool_result");
        assert_eq!(result["content"][0]["tool_use_id"], "toolu_01");
        assert_eq!(result["content"][0]["is_error"], false);
        // Request 2 still advertises the tools — dropping them mid-loop makes
        // the model unable to call anything on its follow-up turn.
        assert!(bodies[1]["tools"].is_array());

        // ── dispatch ──
        let calls = stub.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].0, "board.show",
            "wire name must be mapped back to the registry name before execute"
        );
        assert_eq!(calls[0].1, serde_json::json!({ "id": "R833" }));

        // ── events ──
        let seen = seen.lock().unwrap();
        let tool_calls: Vec<_> = seen
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolCall { tool_name, .. } => Some(tool_name.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(tool_calls, vec!["board.show".to_string()]);
        assert_eq!(
            seen.iter()
                .filter(|e| matches!(e, AgentEvent::ToolResult { ok: true, .. }))
                .count(),
            1
        );
        // Exactly one TurnStarted and one terminal event for the whole loop.
        assert_eq!(
            seen.iter()
                .filter(|e| matches!(e, AgentEvent::TurnStarted { .. }))
                .count(),
            1
        );
        let ended: Vec<_> = seen
            .iter()
            .filter_map(|e| match e {
                AgentEvent::TurnEnded {
                    text,
                    stop_reason,
                    usage,
                    ..
                } => Some((text.clone(), stop_reason.clone(), usage.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(ended.len(), 1, "one terminal event for the whole tool loop");
        let (text, stop_reason, usage) = ended[0].clone();
        assert_eq!(text, "Looking that up.R833 is in progress.");
        assert_eq!(stop_reason.as_deref(), Some("end_turn"));
        // Usage sums across both requests; peak context is the max of the two,
        // not their sum (100 and 200 in the fixtures).
        let usage = usage.expect("usage reported");
        assert_eq!(usage.input_tokens, Some(300));
        assert_eq!(usage.output_tokens, Some(50));
        assert_eq!(usage.peak_context_tokens, Some(200));

        // The turn appends one assistant Message carrying the aggregate text;
        // the tool blocks are turn-local by design (R833-B14). A turn that
        // made a tool call keeps the pre-T10 Text shape (see the comment at
        // the history push in run_anthropic_turn) rather than Blocks.
        let s = session.turn_input(&AnthropicAuth::ApiKey("k".into())).await;
        assert_eq!(s.history.len(), 2);
        assert!(matches!(s.history[1].role, Role::Assistant));
        assert_eq!(
            s.history[1].content,
            MessageContent::Text("Looking that up.R833 is in progress.".into()),
        );
    }

    #[tokio::test]
    async fn a_registry_less_session_makes_exactly_one_request_with_no_tools() {
        // The pre-R833-F13 path must be untouched: no tools key, no loop.
        let (url, stub_server) = scripted_messages(vec![final_text_turn("no tools here")]).await;

        let session = fixture_input("claude-sonnet-4-5", None);
        let session = TestHost::new(session);

        run_turn(&session, &yah_runner_core::NullSink, &url)
        .await
        .expect("turn should succeed");

        let bodies = stub_server.requests();
        assert_eq!(bodies.len(), 1);
        assert!(bodies[0].get("tools").is_none());
    }

    /// A single-block response carrying a real Anthropic `compaction` block,
    /// in its documented wire shape (R883-B13): the block opens with an empty
    /// `content` and the whole summary arrives in ONE `compaction_delta`.
    fn compaction_block_turn(summary: &str) -> MessagesTurn {
        MessagesTurn::new().with_compaction(summary).usage(50, 5)
    }

    #[tokio::test]
    async fn compaction_block_survives_into_history_and_round_trips_verbatim() {
        // R883-T10 acceptance path: a single-request turn whose only content
        // block is one this cell has no bespoke field-tracking for (standing
        // in for Anthropic's `compaction` block) must land in
        // `session.history` as `Blocks`, not be silently dropped or
        // flattened to empty text — and the next request must replay it
        // byte-for-byte.
        let (url, _stub_server) =
            scripted_messages(vec![compaction_block_turn("earlier work summarized")]).await;

        let session = fixture_input("claude-sonnet-4-5", None);
        let session = TestHost::new(session);

        run_turn(&session, &yah_runner_core::NullSink, &url)
        .await
        .expect("turn should succeed");

        let s = session.turn_input(&AnthropicAuth::ApiKey("k".into())).await;
        assert_eq!(s.history.len(), 2);
        assert!(matches!(s.history[1].role, Role::Assistant));
        assert_eq!(
            s.history[1].content,
            MessageContent::Blocks(vec![serde_json::json!({
                "type": "compaction",
                "content": "earlier work summarized",
            })]),
            "the whole-block response must survive verbatim, not collapse to Text",
        );
        assert_eq!(s.history[1].content.text(), "", "a non-text block contributes nothing to text()");

        // Round trip: the next request must send this block back unchanged.
        let body = build_anthropic_body_with_tool_turns(&s, &AnthropicAuth::ApiKey("k".into()), &[]);
        let msgs = body["messages"].as_array().unwrap();
        let echoed = &msgs[1]["content"][0];
        assert_eq!(echoed["type"], "compaction");
        assert_eq!(echoed["content"], "earlier work summarized");
    }

    /// A response carrying a compaction block AHEAD of a text block and a
    /// tool_use block, stopping with `tool_use` — the shape a long
    /// tool-calling turn actually produces when compaction fires mid-loop
    /// rather than on the turn's final (tool-free) iteration.
    fn compaction_then_tool_use_turn(wire_name: &str, summary: &str, text: &str) -> MessagesTurn {
        MessagesTurn::new()
            .with_compaction(summary)
            .with_text(text)
            .with_tool_use("toolu_01", wire_name, serde_json::json!({"id": "R883"}), 1)
            .usage(9000, 40)
    }

    #[tokio::test]
    async fn compaction_block_seen_mid_tool_loop_still_survives_into_history() {
        // R883-T10 follow-up: a long leader/courier turn that both calls a
        // tool AND runs long enough to hit compaction is the common case,
        // not the edge case — the first iteration carries a compaction block
        // alongside its tool_use, the second iteration is a plain follow-up
        // text with no blocks worth preserving. The persisted assistant
        // Message must still carry the compaction block (not silently drop
        // it because the turn made a tool call), paired with one text block
        // holding the full accumulated text — never the tool_use block,
        // which stays turn-local per R833-B14.
        let (url, stub_server) = scripted_messages(vec![
            compaction_then_tool_use_turn("board_show", "earlier work summarized", "Looking that up."),
            final_text_turn("R833 is in progress."),
        ])
        .await;

        let stub = Arc::new(StubRegistry::new(vec![stub_schema("board.show")]));
        let mut session = fixture_input("claude-sonnet-4-5", None);
        session.tool_registry = Some(stub.clone());
        let session = TestHost::new(session);

        run_turn(&session, &yah_runner_core::NullSink, &url)
        .await
        .expect("turn should succeed");

        // The re-issued request (iteration 2) must echo iteration 1's
        // assistant content verbatim, including the compaction block, ahead
        // of the tool_result — this is the in-flight round trip, separate
        // from the persisted-history round trip asserted below.
        let bodies = stub_server.requests();
        assert_eq!(bodies.len(), 2);
        let msgs2 = bodies[1]["messages"].as_array().unwrap();
        let echo = &msgs2[msgs2.len() - 2];
        assert_eq!(echo["role"], "assistant");
        assert_eq!(echo["content"][0]["type"], "compaction");
        assert_eq!(echo["content"][0]["content"], "earlier work summarized");
        assert_eq!(echo["content"][1]["type"], "text");
        assert_eq!(echo["content"][2]["type"], "tool_use");

        let s = session.turn_input(&AnthropicAuth::ApiKey("k".into())).await;
        assert_eq!(s.history.len(), 2);
        assert!(matches!(s.history[1].role, Role::Assistant));
        assert_eq!(
            s.history[1].content,
            MessageContent::Blocks(vec![
                serde_json::json!({ "type": "compaction", "content": "earlier work summarized" }),
                serde_json::json!({ "type": "text", "text": "Looking that up.R833 is in progress." }),
            ]),
            "a compaction block seen mid-loop must survive, paired with the turn's full text \
             — not the tool_use block, and not just the final iteration's text",
        );

        // Persisted-history round trip: the NEXT external turn's body must
        // still send this compaction block back verbatim.
        let body = build_anthropic_body_with_tool_turns(&s, &AnthropicAuth::ApiKey("k".into()), &[]);
        let msgs = body["messages"].as_array().unwrap();
        let echoed = &msgs[1]["content"];
        assert_eq!(echoed[0]["type"], "compaction");
        assert_eq!(echoed[0]["content"], "earlier work summarized");
        assert_eq!(echoed[1]["type"], "text");
        assert_eq!(echoed[1]["text"], "Looking that up.R833 is in progress.");
    }

    #[test]
    fn is_retryable_status_distinguishes_transient_from_permanent() {
        // 429 + 5xx → retry
        assert!(is_retryable_status(429));
        assert!(is_retryable_status(500));
        assert!(is_retryable_status(502));
        assert!(is_retryable_status(503));
        assert!(is_retryable_status(504));

        // Auth / bad-request → permanent
        assert!(!is_retryable_status(401));
        assert!(!is_retryable_status(403));
        assert!(!is_retryable_status(400));
        assert!(!is_retryable_status(404));
        assert!(!is_retryable_status(422));

        // 2xx isn't called on retry path but defined regardless
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
        // Caps at MAX_RETRY_AFTER_SECS (30) to keep a misconfigured upstream
        // from stalling the session.
        assert_eq!(compute_stream_open_backoff_ms(1, Some(300)), 30_000);
        // Retry-After overrides the exponential schedule even on later attempts.
        assert_eq!(compute_stream_open_backoff_ms(3, Some(10)), 10_000);
    }

    #[test]
    fn model_supports_thinking_recognises_4x_and_3_7() {
        assert!(model_supports_thinking("claude-opus-4-7"));
        assert!(model_supports_thinking("claude-sonnet-4-6"));
        assert!(model_supports_thinking("claude-haiku-4-5"));
        assert!(model_supports_thinking("claude-3-7-sonnet-20250219"));
        assert!(model_supports_thinking("deepseek-v4-pro"));
        assert!(model_supports_thinking("kimi-k3"));
        assert!(model_supports_thinking("kimi-k2.6"));
        assert!(model_supports_thinking("kimi-k2.7-code"));
        assert!(!model_supports_thinking("claude-3-5-sonnet-20241022"));
        assert!(!model_supports_thinking("claude-3-haiku-20240307"));
        assert!(!model_supports_thinking("claude-2.1"));
    }

    #[test]
    fn think_budget_tokens_clamps_to_anthropic_minimum() {
        assert_eq!(think_budget_tokens(&ReasoningBudget::Fast), 1024);
        assert_eq!(
            think_budget_tokens(&ReasoningBudget::Budget { tokens: 200 }),
            1024,
            "values below the floor must round up to avoid 400 from Anthropic",
        );
        assert_eq!(
            think_budget_tokens(&ReasoningBudget::Budget { tokens: 8000 }),
            8000
        );
        assert_eq!(think_budget_tokens(&ReasoningBudget::Standard), 4000);
        assert_eq!(think_budget_tokens(&ReasoningBudget::Deep), 16_000);
    }

    #[test]
    fn anthropic_body_omits_thinking_when_session_has_no_think_budget() {
        let s = fixture_input("claude-opus-4-7", None);
        let body = build_anthropic_body(&s, &AnthropicAuth::ApiKey("k".into()));
        assert!(body.get("thinking").is_none());
        assert_eq!(body["max_tokens"], DEFAULT_MAX_OUTPUT_TOKENS);
        assert_eq!(body["model"], "claude-opus-4-7");
        assert_eq!(body["stream"], true);
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn anthropic_body_adds_thinking_field_and_bumps_max_tokens_on_supported_model() {
        let s = fixture_input("claude-opus-4-7", Some(ReasoningBudget::Standard));
        let body = build_anthropic_body(&s, &AnthropicAuth::ApiKey("k".into()));
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 4000);
        assert_eq!(body["max_tokens"], 4000 + DEFAULT_MAX_OUTPUT_TOKENS);
    }

    #[test]
    fn anthropic_body_drops_thinking_field_on_unsupported_model() {
        let s = fixture_input("claude-3-5-sonnet-20241022", Some(ReasoningBudget::Deep));
        let body = build_anthropic_body(&s, &AnthropicAuth::ApiKey("k".into()));
        assert!(body.get("thinking").is_none());
        assert_eq!(body["max_tokens"], DEFAULT_MAX_OUTPUT_TOKENS);
    }

    #[test]
    fn anthropic_body_omits_thinking_field_when_session_think_is_none() {
        let s = fixture_input("claude-opus-4-7", None);
        let body = build_anthropic_body(&s, &AnthropicAuth::ApiKey("k".into()));
        assert!(
            body.get("thinking").is_none(),
            "no thinking dial set → no `thinking` key on wire (no surprise extended-thinking spend)",
        );
        assert_eq!(body["max_tokens"], DEFAULT_MAX_OUTPUT_TOKENS);
    }

    #[test]
    fn anthropic_body_carries_each_named_tier_on_supported_model() {
        // Pin the contract for every named tier so adding a fifth (Max) or
        // remapping budgets shows up here, not in an integration smoke.
        for (tier, expected_budget) in [
            (ReasoningBudget::Fast, ANTHROPIC_THINKING_MIN_BUDGET),
            (ReasoningBudget::Standard, 4_000),
            (ReasoningBudget::Deep, 16_000),
        ] {
            let s = fixture_input("claude-opus-4-7", Some(tier));
            let body = build_anthropic_body(&s, &AnthropicAuth::ApiKey("k".into()));
            assert_eq!(body["thinking"]["type"], "enabled", "tier {tier:?}");
            assert_eq!(
                body["thinking"]["budget_tokens"], expected_budget,
                "tier {tier:?} budget_tokens",
            );
            assert_eq!(
                body["max_tokens"],
                expected_budget + DEFAULT_MAX_OUTPUT_TOKENS,
                "tier {tier:?} max_tokens = budget + headroom",
            );
        }
    }

    #[test]
    fn anthropic_body_clamps_sub_minimum_explicit_budget_up_to_anthropic_floor() {
        // Anthropic /v1/messages rejects budget_tokens < 1024 with 400.
        // The runner clamps Budget { tokens: N < 1024 } up to the floor so
        // a user-set tiny budget doesn't make the whole turn fail.
        let s = fixture_input(
            "claude-opus-4-7",
            Some(ReasoningBudget::Budget { tokens: 256 }),
        );
        let body = build_anthropic_body(&s, &AnthropicAuth::ApiKey("k".into()));
        assert_eq!(body["thinking"]["budget_tokens"], ANTHROPIC_THINKING_MIN_BUDGET);
    }

    #[test]
    fn anthropic_body_adds_context_management_for_claude_model_on_api_key() {
        let s = fixture_input("claude-opus-4-7", None);
        let body = build_anthropic_body(&s, &AnthropicAuth::ApiKey("k".into()));
        assert_eq!(
            body["context_management"]["edits"][0]["type"],
            "clear_tool_uses_20250919",
        );
        assert_eq!(
            body["context_management"]["edits"][1]["type"],
            "compact_20260112",
            "R883-T10: compaction rides the same edits list as context editing",
        );
    }

    #[test]
    fn anthropic_body_omits_compaction_for_models_off_the_compaction_list() {
        // R883-B13: Haiku 4.5 / Sonnet 4.5 get context editing but not
        // compact_20260112 — it is not on Anthropic's compaction model list.
        for model in ["claude-haiku-4-5-20251001", "claude-sonnet-4-5"] {
            let s = fixture_input(model, None);
            let body = build_anthropic_body(&s, &AnthropicAuth::ApiKey("k".into()));
            let edits = body["context_management"]["edits"].as_array().unwrap();
            assert_eq!(edits.len(), 1, "{model}");
            assert_eq!(edits[0]["type"], "clear_tool_uses_20250919", "{model}");
        }
        for model in ["claude-sonnet-4-6", "claude-sonnet-5", "claude-opus-5-5", "claude-fable-5-1"] {
            assert!(model_supports_compaction(model), "{model}");
        }
    }

    #[test]
    fn fold_block_delta_concatenates_payload_fields_but_not_type() {
        // R883-B13: the payload of a preserved block streams on deltas.
        let mut compaction = serde_json::json!({ "type": "compaction", "content": "" });
        fold_block_delta(
            compaction.as_object_mut().unwrap(),
            serde_json::json!({ "type": "compaction_delta", "content": "the summary" }).as_object().unwrap(),
        );
        assert_eq!(compaction, serde_json::json!({ "type": "compaction", "content": "the summary" }));

        let mut thinking = serde_json::json!({ "type": "thinking", "thinking": "", "signature": "" });
        for d in [
            serde_json::json!({ "type": "thinking_delta", "thinking": "a" }),
            serde_json::json!({ "type": "thinking_delta", "thinking": "b" }),
            serde_json::json!({ "type": "signature_delta", "signature": "sig" }),
        ] {
            fold_block_delta(thinking.as_object_mut().unwrap(), d.as_object().unwrap());
        }
        assert_eq!(
            thinking,
            serde_json::json!({ "type": "thinking", "thinking": "ab", "signature": "sig" })
        );
    }

    #[test]
    fn anthropic_body_omits_context_management_for_non_claude_model() {
        // DeepSeek/Kimi share this cell's wire protocol but have no contract
        // for an Anthropic-only beta field.
        let s = fixture_input("deepseek-v4-pro", None);
        let body = build_anthropic_body(&s, &AnthropicAuth::ApiKey("k".into()));
        assert!(body.get("context_management").is_none());
    }

    #[test]
    fn anthropic_body_omits_context_management_on_pinned_oauth_shape() {
        // PinnedShape::headers must replay verbatim (TOS-signature match),
        // so there's no channel to add the beta flag this field needs —
        // sending the field without it would just 400.
        let s = fixture_input("claude-opus-4-7", None);
        let shape = Arc::new(PinnedShape {
            headers: Vec::new(),
            system_prefix: "p".into(),
        });
        let body = build_anthropic_body(
            &s,
            &AnthropicAuth::Oauth {
                token: "t".into(),
                shape: Some(shape),
            },
        );
        assert!(body.get("context_management").is_none());
    }

    #[test]
    fn anthropic_body_adds_context_management_on_degraded_oauth() {
        // No pinned shape → the fallback header path, which carries the beta
        // baked into ANTHROPIC_OAUTH_BETA_VALUE.
        let s = fixture_input("claude-opus-4-7", None);
        let body = build_anthropic_body(
            &s,
            &AnthropicAuth::Oauth {
                token: "t".into(),
                shape: None,
            },
        );
        assert_eq!(
            body["context_management"]["edits"][0]["type"],
            "clear_tool_uses_20250919",
        );
        assert_eq!(
            body["context_management"]["edits"][1]["type"],
            "compact_20260112",
        );
    }

    #[test]
    fn oauth_beta_value_merges_context_management_without_dropping_prior_betas() {
        assert!(ANTHROPIC_OAUTH_BETA_VALUE.contains("oauth-2025-04-20"));
        assert!(ANTHROPIC_OAUTH_BETA_VALUE.contains("claude-code-20250219"));
        assert!(ANTHROPIC_OAUTH_BETA_VALUE.contains(ANTHROPIC_CONTEXT_MANAGEMENT_BETA));
        assert!(ANTHROPIC_OAUTH_BETA_VALUE.contains("compact-2026-01-12"));
    }

    #[test]
    fn context_management_beta_carries_both_edit_flags() {
        assert!(ANTHROPIC_CONTEXT_MANAGEMENT_BETA.contains("context-management-2025-06-27"));
        assert!(ANTHROPIC_CONTEXT_MANAGEMENT_BETA.contains("compact-2026-01-12"));
    }

    #[test]
    fn context_editing_beta_header_adds_flag_for_api_key_and_bearer_when_field_present() {
        let body = serde_json::json!({ "context_management": { "edits": [] } });
        assert_eq!(
            context_editing_beta_header(&AnthropicAuth::ApiKey("k".into()), &body),
            Some(ANTHROPIC_CONTEXT_MANAGEMENT_BETA),
        );
        assert_eq!(
            context_editing_beta_header(&AnthropicAuth::BearerToken("k".into()), &body),
            Some(ANTHROPIC_CONTEXT_MANAGEMENT_BETA),
        );
    }

    #[test]
    fn context_editing_beta_header_is_none_without_the_field_or_on_oauth() {
        let with_field = serde_json::json!({ "context_management": { "edits": [] } });
        let without_field = serde_json::json!({});
        assert_eq!(
            context_editing_beta_header(&AnthropicAuth::ApiKey("k".into()), &without_field),
            None,
            "nothing to opt into if the body never carries the field",
        );
        assert_eq!(
            context_editing_beta_header(
                &AnthropicAuth::Oauth { token: "t".into(), shape: None },
                &with_field,
            ),
            None,
            "degraded OAuth already carries the flag baked into ANTHROPIC_OAUTH_BETA_VALUE",
        );
        assert_eq!(
            context_editing_beta_header(
                &AnthropicAuth::Oauth {
                    token: "t".into(),
                    shape: Some(Arc::new(PinnedShape {
                        headers: Vec::new(),
                        system_prefix: "p".into(),
                    })),
                },
                &with_field,
            ),
            None,
            "pinned shape must never gain a second, non-verbatim anthropic-beta header",
        );
    }

    /// The header a request built by [`AnthropicAuth::apply`] carries for
    /// `name`, matched case-insensitively. `None` when absent; panics when
    /// sent twice, since a duplicated auth header is itself a bug.
    fn header<'a>(req: &'a HttpRequest, name: &str) -> Option<&'a str> {
        let mut hits = req.headers.iter().filter(|(k, _)| k.eq_ignore_ascii_case(name));
        let first = hits.next().map(|(_, v)| v.as_str());
        assert!(hits.next().is_none(), "header {name} sent twice");
        first
    }

    #[test]
    fn anthropic_auth_apply_skips_transport_owned_headers_on_pinned_oauth() {
        let headers = vec![
            ("Authorization".into(), "Bearer should-be-replaced".into()),
            ("HOST".into(), "evil.example.com".into()),
            ("Content-Length".into(), "999".into()),
            ("anthropic-beta".into(), "pin-test".into()),
            ("user-agent".into(), "claude-cli/test".into()),
        ];
        let shape = Arc::new(PinnedShape {
            headers,
            system_prefix: "p".into(),
        });
        let auth = AnthropicAuth::Oauth {
            token: "real-token".into(),
            shape: Some(shape),
        };
        let req = auth.apply(HttpRequest::post_json("https://example.invalid/", &serde_json::json!({})));
        assert_eq!(header(&req, "authorization"), Some("Bearer real-token"));
        assert_eq!(header(&req, "anthropic-beta"), Some("pin-test"));
        assert_eq!(header(&req, "user-agent"), Some("claude-cli/test"));
        assert_eq!(header(&req, "host"), None, "host must not come from the pin");
        assert_eq!(header(&req, "content-length"), None, "content-length is the transport's");
    }

    #[test]
    fn anthropic_auth_apply_falls_back_when_oauth_shape_missing() {
        let auth = AnthropicAuth::Oauth {
            token: "real-token".into(),
            shape: None,
        };
        let req = auth.apply(HttpRequest::post_json("https://example.invalid/", &serde_json::json!({})));
        assert_eq!(header(&req, "authorization"), Some("Bearer real-token"));
        assert_eq!(header(&req, "anthropic-beta"), Some(ANTHROPIC_OAUTH_BETA_VALUE));
        assert_eq!(header(&req, "anthropic-version"), Some(ANTHROPIC_VERSION));
    }

    #[test]
    fn merge_anthropic_usage_takes_input_from_message_start_and_output_from_message_delta() {
        let mut acc = TurnUsage::default();
        let start_usage = serde_json::json!({
            "input_tokens": 1234,
            "cache_read_input_tokens": 1000,
            "cache_creation_input_tokens": 0,
            "output_tokens": 1,
        });
        merge_anthropic_usage(&mut acc, &start_usage);
        assert_eq!(acc.input_tokens, Some(1234));
        assert_eq!(acc.cache_read_tokens, Some(1000));
        assert_eq!(acc.cache_write_tokens, Some(0));
        assert_eq!(acc.output_tokens, Some(1));

        let delta_usage = serde_json::json!({ "output_tokens": 567 });
        merge_anthropic_usage(&mut acc, &delta_usage);
        assert_eq!(
            acc.output_tokens,
            Some(567),
            "later message_delta output_tokens overwrites the placeholder",
        );
        assert_eq!(acc.input_tokens, Some(1234));
    }

}
