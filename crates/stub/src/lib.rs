//! Inference stub — tiny axum server returning canned OpenAI-compat SSE
//! responses keyed off `blake3(url)`.
//!
//! Point `GROQ_BASE_URL=http://127.0.0.1:<port>` at a running stub to let
//! eval tickets that test plumbing (T1 schema, T5 deploy) run without a
//! real provider.  T2/T3 still need real inference — the stub only covers
//! the subset where LLM output quality does not matter.
//!
//! ## Wire shape
//!
//! `POST /v1/chat/completions` — accepts any OpenAI-compat body.
//! Extracts the first `http(s)://…` URL from the message contents, hashes
//! it with blake3, and returns a deterministic two-sentence summary + three
//! tags in a single SSE chunk followed by `finish_reason: stop`, a usage
//! chunk, and `[DONE]`.
//!
//! `GET /v1/models` — returns a one-model catalogue so runner model-listing
//! calls don't 404.
//!
//! ## Modes
//!
//! [`StubMode::CannedSummary`] is the above and the default. [`StubMode::Lorem`]
//! (R267-F9) instead streams N deterministic lorem words in frames of
//! `chunk_tokens`, which is what makes this usable as a **memory workload**: a
//! renderer's cost tracks SSE-frame count, not byte count, so `chunk_tokens: 1`
//! reproduces a real token-by-token stream while the same text in one frame
//! would exercise almost none of it. Spawn with [`InferenceStub::spawn_with`].
//!
//! [`StubMode::Script`] (R923-F6, W354 §2) answers from a [`Script`]: rules
//! per role, where the role is the request's `model` and every predicate is
//! read off the request body (`turn_index` = assistant messages in it,
//! `after_tool_result` / `tool_result_contains` = the trailing tool results).
//! The conversation IS the state — the stub never counts — so a script is
//! deterministic under any interleaving. First match wins. No match is an
//! HTTP 500 carrying the conversation tail, never a fallthrough to lorem.
//! The fired-rule sequence per conversation is the scenario's oracle, on
//! `GET /control/state` under `conversations`.
//!
//! ## Barriers and the control plane (R923-F2)
//!
//! A response is a sequence of [`Step`]s — SSE frames and `Await(barrier)`
//! holds — streamed by [`step_body`]. A barrier is a property of the SERVER,
//! not of any script format, so it works in every mode today; R923-F6's
//! `await: barrier:<name>` lowers to the same [`Step::Await`].
//!
//! - **Arming** (how a mode with no script gets a hold): a [`BarrierArm`] says
//!   "the next `count` completions matching `model` / `contains` hold at
//!   `barrier` before their first frame". `POST /control/arm` or
//!   [`InferenceStub::arm`].
//! - **Release** is latched: `POST /control/release {"barrier": …}` or
//!   [`InferenceStub::release`] frees every response parked there AND every
//!   later arrival, so a scenario that releases before the request lands does
//!   not deadlock.
//! - **Rendezvous**: [`InferenceStub::until_waiting`] resolves once N responses
//!   are parked at a barrier — event-driven, never a sleep.
//! - **State**: `GET /control/state` / [`InferenceStub::state`] — served count,
//!   every completion request (model, turn index, barrier it was armed with),
//!   barriers currently waiting and barriers released.
//! - **`delay_ms`** ([`StubBuilder::delay_ms`]) sleeps between frames. Pacing for
//!   streaming / busy-state rendering only — it is NOT a rendezvous.
//!
//! Each held response parks on its own barrier's `watch` channel; the shared
//! registry is a `std::sync::Mutex` that is never held across an `.await`, so
//! one held conversation cannot stall any other. With no arms and
//! `delay_ms == 0` the body is the same single buffer as before, which keeps
//! `memory_growth.rs` (the perf lane, W354 §4) on exactly its old path.
//!
//! @yah:ticket(R267-F9, "Lorem-ipsum mock agent + N-concurrent booking driver: one harness for the memory workload and the slot-allocation bugs")
//! @yah:status(review)
//! @yah:at(2026-09-02T18:24:03Z)
//! @yah:assignee(agent:bundle-anthropic-ashguard)
//! @yah:phase(P1)
//! @yah:parent(R267)
//! @arch:see(.yah/docs/working/W336-desktop-memory-campaign-retrospective.md)
//! @yah:next("Extend the existing axum stub (this file) with a lorem-ipsum turn of configurable token length. Do NOT build a new engine: the runner reaches it with zero runner changes via a camp-local subclass with wire=openai_wire and base_url=http://127.0.0.1:<port>, and crates/yah/bridge already translates Anthropic<->OpenAI for an anthropic-wire subclass.")
//! @yah:next("Driver that books N (8-16) concurrent agents off the board and streams into them. This is the memory workload AND the reproduction harness for the booking bugs - same code path, two payoffs.")
//! @yah:next("Reuse it from R267-F8's memory_growth.rs: the 30-min empty-camp run went FLAT (2.3 MB/min), so live streaming concurrency is the leading candidate for the missing variable.")
//! @yah:next("Tool-use scripting (tier 2) is NOT needed for memory - only for interaction-correctness tests. Real Haiku (tier 3) stays opt-in: non-deterministic, costs money, needs network, and a flaky memory gate is worse than none.")
//! @yah:gotcha("Operator reports live slot-allocation bugs this harness should catch (rare but real): a booking sometimes 'goes into the ether', and sometimes gets scheduled but never pushed to the working set. Both happen BEFORE the model emits anything, so tier-1 lorem output is sufficient to exercise them - no tool-use mocking required.")
//! @yah:handoff("TIER 1 STUB LANDED in crates/yah/eval/src/inference_stub.rs. Added StubMode {CannedSummary, Lorem{tokens, chunk_tokens}} + InferenceStub::spawn_with(mode); spawn() keeps the old canned behaviour so every pre-existing eval caller is untouched. build_lorem_sse streams `tokens` words in frames of `chunk_tokens` with the same frame shape as the canned path (finish_reason stop + usage + [DONE]). chunk_tokens is the load-bearing knob: the renderer's cost tracks EVENT count, not byte count, so one 2000-word frame exercises almost nothing while 2000 one-word frames reproduce a real token stream. lorem_offset() seeds from a hash of the request body rather than a shared counter, so content is deterministic per conversation but differs between conversations (N agents must not send byte-identical transcripts - interning would make the workload unrealistically cheap). Verify: cargo test -p yah-eval inference_stub -> 8/8 pass (3 new + 5 pre-existing). Also fixed the file's own stale verify command: the crate is yah-eval, `eval` is only the bin name, so the `-p eval` it carried fails with 'did not match any packages'.")
//! @yah:gotcha("Fixture camps must seed their OWN .yah/subclasses/<id>.json (camp-local wins over shared). Do NOT rely on ~/.yah/subclasses - agent-tools/src/tools.rs:234 already records real test failures 'polluted by real ~/.yah/subclasses on this dev machine', which is exactly the trap an e2e booking driver would fall into: it would pass on this rig and fail everywhere else, or silently book a REAL provider subclass instead of the stub.")
//! @yah:handoff("CHECKPOINT. Tier 1 (the mock agent) is COMPLETE and tested; the N-booking driver half was split out to R267-F10 so this ticket's scope now matches what actually shipped. Nothing is half-built: StubMode::Lorem is usable today by anything that can point a subclass at a base_url.")
//! @yah:handoff("Tree anchor at handoff: 570fe0eae15cf8d5fb626a532cd7b6c07e43ba14 — the shared tree as I left it. Diff against it (`git diff 570fe0eae15cf8d5fb626a532cd7b6c07e43ba14..HEAD`) to see what landed under you, and quote this SHA rather than 'HEAD' in any revert/restore instruction.")
//! @yah:next("The booking driver moved to R267-F10 - pick that up next, not this. It carries the subclass-seeding steps and the ~/.yah/subclasses pollution trap.")
//! @yah:next("W336 section 0a holds the open question this feeds: an empty idle camp went FLAT over 30 min (2.3 MB/min), so live streaming concurrency is the leading candidate for the missing variable. R267-F10 is what tests it.")
//! @yah:verify("cargo test -p yah-eval inference_stub -> 8 passed (3 new lorem tests + 5 pre-existing, no regressions)")
//! @yah:handoff("TIER 1 COMPLETE. StubMode::Lorem + InferenceStub::spawn_with land in crates/yah/eval/src/inference_stub.rs; the N-booking half shipped as R267-F10 and is now consuming this from app/yah/e2e (LoremStub). R267-F10 also added InferenceStub::served() - a completions counter on the router state - because a booking driver otherwise cannot tell 'N agents streamed' from 'N agents were booked and never spoke', and a memory scenario measuring the second while claiming the first passes vacuously (the memProbe.ts:20 failure this doc keeps citing). cargo test -p yah-eval inference_stub -> 8/8 before the counter; re-verified after.")
//! @yah:verify("cargo test -p yah-eval inference_stub")
//!
//! @yah:ticket(R923-F2, "Barriers + /control plane on the inference stub: hold a turn in flight")
//! @yah:status(review)
//! @yah:at(2026-10-05T05:18:25Z)
//! @yah:assignee(agent:bundle-anthropic-ashguard)
//! @yah:phase(P1)
//! @yah:parent(R923)
//! @yah:next("Tier: Warrior — the endpoint is trivial; holding an HTTP response open across a barrier while N other conversations stream is where this either works or deadlocks. Budget for that, not for the route.")
//! @yah:next("Ships: an `await: barrier:<name>` step, `delay_ms` inter-frame delay, `POST /control/release {barrier}` and `GET /control/state` (rules fired per session, barriers currently waiting) on the same axum router. yah-eval and yah-e2e are test-only surfaces, so no production surface grows.")
//! @yah:next("USABLE FROM StubMode::Lorem IMMEDIATELY — that is why it precedes the script engine. A barrier is a property of the SERVER, not of the script format, so phase 1's scenarios get their pacing primitive without any of phase 2 landing. This is what makes R921-T14 ('character xN running', i.e. N turns simultaneously in flight) expressible.")
//! @yah:next("delay_ms is for streaming and busy-state rendering only. It is NOT a rendezvous — racing an assertion against a sleep is the flake the e2e README forbids ('No retries. A flake is a bug.'). Barriers are the rendezvous.")
//! @yah:verify("cargo test -p yah-eval inference_stub — the existing 8 tests stay green, plus new ones for hold/release and for a second conversation streaming to completion while the first is held.")
//! @yah:verify("N concurrent conversations, one held at a barrier: the other N-1 complete. Proves the hold is per-conversation and not a router-wide lock.")
//! @yah:gotcha("Do not let a barrier and a perf run mix. memory_growth.rs measures a real slope with real timing; a held response corrupts the measurement. W354 section 4 makes this two lanes on purpose.")
//! @yah:handoff("LANDED in crates/yah/eval/src/inference_stub.rs (+ tokio 'sync','time' features in crates/yah/eval/Cargo.toml). API: a response is a VecDeque<Step> where pub enum Step { Frame(String), Await(String) }, streamed by step_body() via futures::stream::unfold + Body::from_stream. R923-F6 should lower its emit lists into the same Step (text/tool_use -> Frame, await: barrier:<name> -> Await) and reuse step_body unchanged.")
//! @yah:handoff("Arming (how Lorem/CannedSummary get a hold without a script): pub struct BarrierArm { barrier, model: Option<String> (exact match on body.model), contains: Option<String> (substring of any message content), count: usize default 1 }, builders BarrierArm::next(b).contains(..).model(..).count(n). Arms are first-match-wins in arm order, consumed per matching request; a matched request parks at the barrier BEFORE its first frame (headers are already sent, so the client sees a live in-flight stream).")
//! @yah:handoff("Barriers are LATCHED: release frees everything parked AND every later arrival, so releasing before the request lands cannot deadlock. Each barrier is its own tokio watch<bool>; the registry is a std Mutex only ever held for short critical sections, never across an await, so one held conversation cannot stall others. A Parked drop guard decrements the waiting count on release or client hangup.")
//! @yah:handoff("Control plane on the same router: POST /control/arm (BarrierArm JSON) -> {armed}; POST /control/release {barrier} -> {barrier, was_waiting}; GET /control/state -> {served, requests:[{model, turn_index (assistant msgs in body), barrier}], armed:[...], waiting:{name:n}, released:[names]}. Rust mirrors on InferenceStub: arm(), release(), waiting(), state(), and until_waiting(barrier, n).await, an event-driven rendezvous (an epoch watch bumped on every park/unpark/release, no polling interval) for scenarios to assert behind.")
//! @yah:handoff("delay_ms: InferenceStub::builder(mode).delay_ms(ms).spawn(); spawn_with(mode) is now builder(mode).spawn(). This sleeps between consecutive frames and is pacing only, not a rendezvous. Perf-lane safety: with no matching arm and delay_ms == 0 the handler takes the old single-buffer Body::from(frames.concat()) fast path, so memory_growth.rs / app/yah/e2e LoremStub get byte-identical behaviour (no caller changes, app/yah/e2e untouched). build_sse_response/build_lorem_sse are now #[cfg(test)] concat wrappers over canned_frames/lorem_frames.")
//! @yah:handoff("Not done here, F6 territory: 'rules fired per session' in /control/state. Lorem has no rules, so state records per-request {model, turn_index, barrier}; F6 should add the fired rule id to RequestRecord.")
//! @yah:verify("cargo test -p yah-eval inference_stub -> 14 passed, 0 failed (8 pre-existing + 6 new barrier_tests: held_response_waits_until_released, other_conversations_stream_while_one_is_held [N=8, multi_thread, delay_ms 1], release_before_arrival_does_not_hold, arm_count_is_consumed, control_plane_over_http, delay_ms_paces_frames). Run had no input skew per the camp build rail.")
//!
//! @yah:ticket(R923-F6, "StubMode::Script with request-keyed rule matching")
//! @yah:status(review)
//! @yah:at(2026-10-05T12:00:00Z)
//! @yah:assignee(agent:bundle-anthropic-ashguard)
//! @yah:phase(P2)
//! @yah:parent(R923)
//! @yah:next("Tier: Warrior — the format is specified in W354 section 2; the judgment left is the matcher's edge cases. Unit-testable in yah-eval without a desktop, same as the existing 8 inference_stub tests.")
//! @yah:next("THE KEYING RULE, and it is the whole design: read the ordinal off the REQUEST, never count it in the stub. The handler already receives the full body as Json(body): Json<Value> (:203) and already reads body.get('messages') (:319). Two axes — role from body['model'] (each fixture character gets a divination whose model is a script role id like 'script:leader'; the OpenAI wire guarantees model is present, so roles are addressed without scraping the system prompt), and position from the message array (turn_index = count of assistant messages in THIS body, after_tool_result = name of the trailing tool result, tool_result_contains = substring test on it). The conversation IS the state, so the stub is stateless per-conversation and deterministic under any interleaving.")
//! @yah:next("WHY NOT the existing ordinal format: spawn_fake_messages_server (anthropic.rs:1749) takes Vec<Vec<String>> keyed by request ordinal and already round-trips a real tool loop, but one stub + one counter + N agents does not survive concurrency — courier 2's first request can arrive before courier 1's, and a retry (reconnect, 429) consumes an ordinal. That is an order-dependent flake, which the e2e README forbids outright.")
//! @yah:next("Rule order is significant, FIRST MATCH WINS. That is how a scenario distinguishes 'both returns landed' from 'one did' inside the script, without the test asserting on prose.")
//! @yah:next("THE SCRIPT IS ALSO THE ORACLE: record the fired-rule sequence per session and expose it on GET /control/state. A scenario asserting 'leader fired L1, then L2, then L3-collect-both' states 'it dispatched twice and then collected both returns' once, in one place, instead of as five brittle DOM assertions. spawn_fake_messages_server hands back every request body (anthropic.rs:1751) for the same reason; this is that idea named.")
//! @yah:verify("cargo test -p yah-eval inference_stub — existing 8 green, plus matcher tests incl. first-match-wins ordering and a no-match case asserting 500 with the conversation tail in the body.")
//! @yah:verify("Two conversations interleaved request-by-request produce the same fired-rule sequences as when run serially.")
//! @yah:gotcha("FALLBACK MUST BE LOUD: if no rule matches, return HTTP 500 with the unmatched conversation tail. NEVER fall through to lorem. This is the single most important rule in the format and the precedent is in-tree — served() exists (inference_stub.rs:48) because a driver otherwise cannot tell 'N agents streamed' from 'N agents were booked and never spoke', and a scenario measuring the second while claiming the first passes vacuously. A silent lorem fallback reintroduces exactly that.")
//! @yah:gotcha("Sigils in a script must come from party::SIGIL_ORDER (party.rs:963-980: polaris, griffin, eclipse, spade, libra, dove, coffee, dragon, hydra, blade, rune, vortex, golem, rose, citadel, nautilus). seed_stub_characters assigns them in that order capped at the pool size, so a scenario's addressable sigils are the first N of that list and nothing else. W354's first draft used 'moon', which is not one.")
//! @yah:depends_on(R923-S5)
//! @yah:gotcha("W354 section 2's script example targets `party_dispatch` with `target: {sigil: griffin}`. That is wrong: party.dispatch's input schema has NO sigil key; target accepts only character/subclass/class/job/camp (agent-tools/src/agent_dispatch_tools.rs, PartyDispatch::schema). Operator call 2026-10-04: scripts target by CHARACTER, never sigil and never bare (a bare dispatch auto-picks the first eligible subclass, which may be a real provider). R923-S5's tests/agent_dispatch.rs uses {character: Lorem0}.")
//! @yah:verify("cargo test -p yah-eval inference_stub -> 20 passed, 0 failed (baseline 15: 14 kept + the S5 test replaced by 6 script_tests). yah-desktop-e2e-functional (native, run 0f83d7af-d237-49b0-9e7e-07110afedd30) -> Success: r921 4/4, restart 1/1, terminal_tile 5/5, agent_dispatch 1/1.")
//! @yah:handoff("LANDED StubMode::Script in crates/yah/eval/src/inference_stub.rs; it SUBSUMES and DELETES R923-S5's ToolUseRule / StubBuilder::tool_use / tool_use_frames (pre-1.0, no shim). Script API: pub struct Script {script: String, roles: BTreeMap<model, Vec<Rule>>}; Rule {id (unique across the script), when: When, emit: Vec<Emit>, stop: Option<Stop end_turn|tool_use>}; When {turn_index: Option<usize>, after_tool_result: Option<String>, tool_result_contains: string-or-array} (ANDed, absent = any); Emit untagged = {text, chunk_tokens?, delay_ms?} | {tool_use: {name (WIRE name e.g. party_dispatch), input}} | {await: 'barrier:<name>'}. All structs deny_unknown_fields. Build with Script::from_value(json) / Script::from_json(str) (both validate); spawn via InferenceStub::spawn_with(StubMode::Script(s)) (builder.spawn() re-validates).")
//! @yah:handoff("Matching: role = body.model exactly; turn_index = assistant messages in the body; trailing tool results = the trailing RUN of role:tool messages (a 2-call turn returns 2), each named by resolving tool_call_id against assistant tool_calls[].function.name (fallback: the message's own name); after_tool_result matches any of them; every tool_result_contains needle must appear in one. First match wins. $TOOL_RESULT in text = last trailing result. Lowering: text -> content frames (chunk_tokens words each, Step::Sleep(delay_ms) between), tool_use -> 3 frames at increasing index with id call_<rule>_<n>, await -> Step::Await, then finish_reason tool_calls iff any tool_use else stop. Added Step::Sleep; the single-buffer fast path is kept when steps are all frames, no arm, delay_ms 0.")
//! @yah:handoff("No match (unknown role or no rule) -> HTTP 500 application/json {error:{type:script_no_match, message, model, turn_index, after_tool_result, role_rules (null if role unknown), tail: last 4 messages}}; the request is logged with unmatched:true and arms are not consumed. ORACLE: GET /control/state gains conversations:[{conversation, model, fired:[rule ids]}] and each request record gains conversation. Conversation key = blake3(model + first user message)[..12], read off the request, no counter: two same-role conversations opening with an identical prompt share a key, so give scenario conversations distinct prompts.")
//! @yah:handoff("Script::validate refuses: duplicate rule ids; stop inconsistent with emit; party_dispatch/party_assist tool_use without input.target.character (the character-not-sigil operator call, now enforced). Refactor: content_frame/finish_frames shared by canned/lorem/script paths.")
//! @yah:handoff("app/yah/e2e migrated: LoremStub::start_with_tool_use deleted -> LoremStub::start_script(Script); start() goes through private start_mode. tests/agent_dispatch.rs runs a full script (L1-dispatch, L2-dispatched, L3-woken for a dispatch_result wake, child lorem-1 C1-reply) and asserts the leader fired sequence starts [L1-dispatch, L2-dispatched] and that NO request went unmatched.")
//! @yah:handoff("Board note: board.review (MCP and CLI) was refused with os error 35 (EAGAIN) three times; board.update worked, so @yah:status(review) was asserted in source by hand.")
//!
//! @yah:ticket(R923-F11, "Emit frames from typed wire structs, not string literals")
//! @yah:status(review)
//! @yah:at(2026-10-05T17:03:40Z)
//! @yah:assignee(agent:bundle-anthropic-ashguard)
//! @yah:phase(P4)
//! @yah:parent(R923)
//! @yah:next("Tier: Warrior — the first real decision in it is where the new types live, and that is a crate-boundary call, not a mechanical port.")
//! @yah:next("THE UNKNOWN IS SETTLED AND THE ANSWER IS NO — diligence 2026-09-17. openai_compat.rs carries ZERO Serialize derives, and its only wire-shaped types (ToolCallDelta :2214, ToolCallAcc :2226, ToolCallFinalized :2233, TurnUsageAcc :2258) are PRIVATE, deserialize-side accumulators living in crates/yah/runner while the stub lives in yah-eval. So this ticket AUTHORS serialize-side response types and picks their home; it does not reach for existing ones.")
//! @yah:next("The problem being fixed: tool_use_frames / final_text_frames (anthropic.rs:1820-1850) are raw r#\"data: {...}\"# literals, so a production wire change breaks an ASSERTION rather than a COMPILE. That is the mockParty.ts risk in miniature — a fixture drifting from the thing it depicts because nothing can contradict it.")
//! @yah:verify("Change a field name on the response type: the stub fails to COMPILE, not to assert.")
//! @yah:depends_on(R923-F6)
//! @yah:handoff("LANDED: crates/yah/eval/src/openai_wire.rs (new, pub mod in lib.rs): Serialize-only, serde-only (no yah types, so it can move to a shared wire crate) ChatCompletionChunk{choices, usage?} / ChunkChoice{delta, finish_reason: Option<FinishReason> serialized as null} / Delta{content?, tool_calls?} / ToolCallDelta{index, id?, type: ToolCallKind?, function: FunctionDelta{name?, arguments}} / FinishReason{Stop,ToolCalls,Length} / Usage{prompt,completion,total_tokens} + SseFrame{Chunk,Done}::to_sse(). Constructors ChatCompletionChunk::delta/finish/usage, Delta::content/tool_call.")
//! @yah:handoff("inference_stub.rs: content_frame, finish_frames (now takes FinishReason, not &str), tool_call_frames (StubMode::Script tool_use) and the [DONE] sentinel all build through openai_wire; sse_frame(Value) is deleted, so no SSE frame literal remains. The 500 unmatched-script error doc and /v1/models listing are not stream frames and stay json!.")
//! @yah:handoff("Round-trip: new test mod runner_round_trip drives the real OpenAiCompatRunner (public Runner start/send + runner::lift_stream) against StubMode::Script emitting text + split tool call + tool_calls/stop finish + usage + [DONE]; asserts ToolCall name/args/id, ToolResult ok, TurnEnded stop_reason=stop, usage parsed (input_tokens 20 = summed over the two round-trips). The runner's parse_sse_frame is private to oss/runner/crates/ho, so no direct parse entry point exists; the full Runner surface is the public seam used. crates/yah/runner untouched (R950).")
//! @yah:handoff("Side edits: crates/yah/eval/src/run.rs build_prelude -> pub(crate) (reused by the test); crates/yah/eval/Cargo.toml dev-dep async-trait 0.1 (ToolRegistry impl).")
//! @yah:verify("cargo test -p yah-eval --lib -- inference_stub openai_wire -> 22 passed, 0 failed (20 baseline + openai_wire::tests::shapes_match_the_openai_stream + runner_round_trip::runner_parses_every_frame_the_stub_emits).")
//! @yah:verify("Compile-break demo: renamed Usage::completion_tokens -> completion_tokenz; cargo check -p yah-eval failed with E0560 at inference_stub.rs:1273 (finish_frames); restored, cargo check -p yah-eval --lib --tests EXIT=0.")
//!
//! @yah:ticket(R923-F12, "Record mode: a real turn lowered into a script")
//! @yah:status(review)
//! @yah:at(2026-10-05T17:33:51Z)
//! @yah:assignee(agent:bundle-anthropic-glimmerstone)
//! @yah:phase(P4)
//! @yah:parent(R923)
//! @yah:next("Tier: Wizard — this is what makes the corpus recorded rather than authored, and the lowering is where the judgment is.")
//! @yah:next("Run a scenario against a real model once, capture the turn sequence off the real .yah/sessions/<id>.jsonl (SessionStore::append, session.rs:318-360), and lower it mechanically into a script. That makes the first script a RECORDING, honouring R722-F2's 'hand-authored fixtures lie; recorded ones cannot' directly, and makes drift detectable by re-recording.")
//! @yah:next("Strongest of the four answers to mockParty.ts and the most work. DO NOT let it gate earlier phases — the other three answers (script is upstream of every validator; an invented tool name does not resolve; typed frames) already hold without it.")
//! @yah:next("Format reuse where it is free: recorded turns are the same AgentEvent-shaped .yah/sessions/*.jsonl that agent.history.replay and env/fixture.ts already consume. Fixture freezing stays a manual, human-reviewed step — that rule transfers from R657 and is not up for renegotiation here.")
//! @yah:verify("A recorded script replays to the same fired-rule sequence as the recording it came from.")
//! @yah:depends_on(R923-F6)
//! @yah:depends_on(R923-F11)
//! @yah:handoff("LANDED crates/yah/eval/src/lower.rs (pub mod lower): lower_file / lower_jsonl / lower_events(events, role) -> Lowered {script: Script, fired: Vec<String> (the recording's own rule sequence = replay oracle), exchanges: Vec<Exchange{rule, turn_index, user, text, calls: Vec<RecordedCall{name, input, ok, result}>}>}. One rule per model RESPONSE (a response ends at its first tool_result; the next text/call opens a new one), id t<N>, when.turn_index = session-wide response ordinal, when.after_tool_result = wire name of the last call answered by the previous response, emit = text then tool_use in recorded order (dots -> _ via the runner's sanitizer rule), stop derived. tool_result_contains deliberately NOT emitted (replay tools run in another camp). Final-response text: assistant_text if recorded, else turn_ended.text minus earlier assistant_text of the turn. Refuses turn_failed/turn_aborted, mid-turn EOF, unanswered calls, and (via Script::validate) a recorded dispatch by sigil. Events are read as serde_json::Value keyed on AgentEvent kind names, not the typed enum, so old logs with since-changed fields still lower.")
//! @yah:handoff("CLI: `yah-eval lower <session.jsonl> --role <model>` prints the script JSON (checked against the fixture: t0/t1/t2, after_tool_result Bash, stop tool_use/tool_use/end_turn).")
//! @yah:handoff("REAL-MODEL STEP per leader decision: no new paid run. Lowered an EXISTING real transcript, .yah/sessions/session:1fcb68e1.jsonl (claude-haiku-4-5, one user turn, two Bash tool calls, text answer), copied VERBATIM to crates/yah/eval/recorded/session-1fcb68e1.jsonl. Checked: no keys/tokens/personal paths (one result is a jettison stub). Kept out of fixtures/ because build.rs tars that tree.")
//! @yah:handoff("OPERATOR REVIEW NEEDED (R657): crates/yah/eval/recorded/session-1fcb68e1.jsonl and the script it lowers to are PRODUCED, NOT CANONICAL. Freezing them as a scenario fixture is the operator's call.")
//! @yah:verify("cargo test -p yah-eval -> lib 35 passed, 0 failed (baseline 32: +3 lower::tests). EXIT=0 /tmp/r923f12-run3.log, no input skew. Includes inference_stub + openai_wire 22 green.")
//! @yah:verify("lower::tests::recorded_session_replays_to_its_own_fired_sequence: the recorded session lowers, replays through a live StubMode::Script stub (recorded user msg + recorded tool results fed back), every response matches the recording's text and calls, no request unmatched, and the stub's fired sequence == lowered.fired == [t0,t1,t2]. multi_turn_lowering_replays covers two user turns, parallel calls, dotted names and accumulated turn text the same way.")
//!
//! @yah:ticket(R923-T13, "Consolidate the fake wire servers onto one")
//! @yah:status(review)
//! @yah:at(2026-10-05T18:08:09Z)
//! @yah:assignee(agent:bundle-anthropic-miravel)
//! @yah:phase(P4)
//! @yah:parent(R923)
//! @yah:next("Tier: Cleric — real debt, low priority. Sequence it LAST so the winner is the one this relay proved out rather than the one that happened to be biggest.")
//! @yah:next("Four known: inference_stub.rs, spawn_fake_messages_server (anthropic.rs:1749), spawn_fake_sse_server (runner/tests/openai_compat_e2e.rs:35), and hao_smoke.rs's stub (crates/yah/runner/examples/hao_smoke.rs). Pre-1.0, so one should win OUTRIGHT rather than four drifting — see the workspace's 'break it, don't tape it' rule; do not leave a compatibility shim beside the winner.")
//! @yah:next("spawn_fake_messages_server is the one with real capability to preserve: it already round-trips a live tool loop (tool_use_frames at :1820 splits arguments across three input_json_delta fragments, the real registry runs the tool, the result returns on request N+1). Whatever wins must keep that, and its request-body capture (:1751).")
//! @yah:verify("cargo test --workspace — every suite that used a retired stub still passes against the survivor, with no shim left behind.")
//! @yah:gotcha("COUNT THEM YOURSELF BEFORE CONSOLIDATING. The figure four is inherited from W354, which explicitly did NOT sweep for a fifth, and per .yah/CLAUDE.md a clean-looking negative is the least trustworthy result in this camp. Confirm the tool ran, the output was not truncated by annotation prose, and the exit code came from the command you think it did.")
//! @yah:depends_on(R923-F6)
//! @yah:depends_on(R923-F12)
//! @yah:depends_on(R923-T10)
//! @yah:handoff("Winner = the R923 inference stub, MOVED from crates/yah/eval to a new test-support crate oss/runner/crates/stub (yah-runner-stub, publish=false, no yah deps). Why: the ha/ho suites now live in the standalone oss/runner workspace (R950) and cannot depend on yah-eval (non-exported, depends on yah-runner => cycle/mirror break). yah-eval, app/yah/e2e and crates/yah/runner (hao_smoke) take a path dep on it. No re-export shim: eval::inference_stub/openai_wire deleted, call sites rewritten.")
//! @yah:handoff("Added to the winner: typed Anthropic Messages wire (anthropic_wire.rs, MessagesTurn with tool_use args split across N input_json_delta fragments + compaction blocks), POST /v1/messages route, StubBuilder::replies(Vec<Reply>) (Sse / Json / Error{status,retry_after}) queue, requests() body capture, hits() route log, messages_url().")
//! @yah:handoff("Retired and deleted: spawn_fake_messages_server (oss/runner ha anthropic.rs), spawn_fake_sse_server + spawn_multi_request_sse_server + spawn_status_sequence_server + inline 401 server (ho tests/openai_compat_e2e.rs), compact_session inline JSON server (ho openai_compat.rs), hao_smoke axum stub. runner_round_trip test moved from the stub into ho e2e (runner_parses_every_frame_the_stub_emits).")
//! @yah:handoff("Counted myself (comment-stripped grep): the '4 known' was stale. R950 had already moved the ha/ho servers into oss/runner and crates/yah/runner/tests/openai_compat_e2e.rs no longer exists; ho alone had 4 raw-socket servers + 1 in openai_compat.rs. NOT consolidated: three keepalive fake /v1/messages servers (crates/yah/runner/src/keepalive.rs:261, crates/yah/bridge/src/keepalive.rs:499, app/yah/desktop/src/keepalive_subsystem.rs:499) which assert on request HEADERS the stub does not capture; and anthropic.rs:757 spawn_logging_proxy, a pass-through to the real upstream (not a fake). Filing as followup is the leader's call.")
//! @yah:handoff("Edited R950-dirty files minimally: crates/yah/runner/Cargo.toml (+1 dev-dep line) and examples/hao_smoke.rs; @Mable:griffin (session:3b56cf7a) was idle awaiting_user, not notified.")
//! @yah:verify("oss/runner: cargo test -p yah-runner-stub 23 pass; -p yah-runner-ho 102 lib + 13/1 ignored e2e (baseline 102 + 12/1, +1 moved round-trip test); -p yah-runner-ha 40 pass (baseline 40). yah workspace: cargo test -p yah-eval 13 pass; cargo build -p yah-runner --example hao_smoke OK; cargo check -p yah-e2e --tests OK (e2e scenarios not executed).")
//! @yah:handoff("Winner = R923 inference stub, MOVED from crates/yah/eval to new test-support crate oss/runner/crates/stub (yah-runner-stub, publish=false, no yah deps) because ha/ho suites live in standalone oss/runner (R950) and cannot depend on yah-eval. eval, app/yah/e2e, crates/yah/runner take a path dep; eval::inference_stub deleted, no shim.")
//! @yah:handoff("Added to winner: typed Anthropic wire (anthropic_wire.rs; tool_use args split across N input_json_delta; compaction), POST /v1/messages, StubBuilder::replies(Reply::Sse/Json/Error), requests() body capture, hits(), messages_url().")
//! @yah:handoff("Retired: spawn_fake_messages_server (ha), spawn_fake_sse_server/multi_request/status_sequence/inline 401 (ho e2e), compact_session inline server (ho openai_compat.rs), hao_smoke axum stub. Round-trip test moved into ho e2e.")
//! @yah:handoff("Recount: the 'four' was stale (R950 moved them; runner/tests/openai_compat_e2e.rs gone; ho alone had 5). NOT consolidated: 3 keepalive fake /v1/messages servers asserting on request HEADERS (crates/yah/runner/src/keepalive.rs:261, crates/yah/bridge/src/keepalive.rs:499, app/yah/desktop/src/keepalive_subsystem.rs:499) - stub would need header capture; anthropic.rs:757 logging proxy is a real-upstream pass-through, not a fake.")
//! @yah:handoff("Verified: stub 23 pass; ho 102 + 13/1ign (baseline 102 + 12/1, +1 moved test); ha 40 (baseline 40); yah-eval 13; hao_smoke builds; yah-e2e --tests checks (scenarios not run). Touched R950-dirty crates/yah/runner/Cargo.toml (+1 dev-dep) and examples/hao_smoke.rs.")
//! @yah:handoff("E2E lane run: yah qed run yah-desktop-e2e-functional --where local --runtime native -> all steps green (build-ui, build-desktop, sync-ui, r921, restart, terminal-tile, agent-dispatch, approval, turn-zero, leader-two-couriers) against the moved crate.")
//! @yah:handoff("R950 acceptance (@Mable:griffin, recorded on R950-T13): ho/ha dev-deps on yah-runner-stub must stay version-less path deps; stub ships in the mirror so keeps MIT OR Apache-2.0 and no yah deps. oss/runner has no .yah/camp.toml, so the stub crate's board annotations stay visible to this camp; no anchors moved.")
//! @yah:handoff("Keepalive fakes left out deliberately: they assert only on request headers/body of a single max_tokens:0, stream-stripped replay POST and never stream inference (crates/yah/runner/src/keepalive.rs, crates/yah/bridge/src/keepalive.rs, app/yah/desktop/src/keepalive_subsystem.rs).")

pub mod anthropic_wire;
pub mod openai_wire;

use axum::{
    body::Body,
    extract::{Json, State},
    http::{HeaderValue, StatusCode},
    response::Response,
    routing::{get, post},
    Router,
};
use crate::openai_wire::{
    ChatCompletionChunk, Delta, FinishReason, FunctionDelta, SseFrame, ToolCallDelta,
    ToolCallKind, Usage,
};
use serde::{Deserialize, Serialize};
use crate::anthropic_wire::MessagesTurn;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::watch;

/// What a stub answers with.
///
/// [`StubMode::CannedSummary`] is the original eval behaviour and stays the
/// default so existing callers are untouched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StubMode {
    /// Two-sentence summary + tags, keyed off `blake3(url)`. For eval tickets
    /// that exercise plumbing where output quality does not matter.
    CannedSummary,
    /// Deterministic lorem, `tokens` words per turn, streamed `chunk_tokens`
    /// words per SSE frame (R267-F9).
    ///
    /// `chunk_tokens` is the load-bearing knob and the reason this is not just
    /// "return a long string": the renderer's cost is driven by **event count**,
    /// not byte count. One 2000-word frame produces a single `MessageDelta`;
    /// 2000 one-word frames produce 2000 of them, which is what a real
    /// token-by-token stream does to `sessionStore` and the turn renderer. A
    /// stub that returned the same text in one frame would look like a
    /// realistic workload and exercise almost none of the path under suspicion.
    Lorem { tokens: usize, chunk_tokens: usize },
    /// Answer from a request-keyed rule program (R923-F6). See [`Script`].
    Script(Script),
}

impl Default for StubMode {
    fn default() -> Self {
        Self::CannedSummary
    }
}

/// Deterministic word pool. Real lorem, not repeated filler: a single repeated
/// word would let the engine's string interning make the workload
/// unrealistically cheap, which is the opposite of what this stub is for.
const LOREM_WORDS: &[&str] = &[
    "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit", "sed", "do",
    "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore", "magna", "aliqua", "enim",
    "ad", "minim", "veniam", "quis", "nostrud", "exercitation", "ullamco", "laboris", "nisi",
    "aliquip", "ex", "ea", "commodo", "consequat", "duis", "aute", "irure", "in",
    "reprehenderit", "voluptate", "velit", "esse", "cillum", "fugiat", "nulla", "pariatur",
];

const CANNED_SUMMARIES: &[&str] = &[
    "A resource providing technical documentation and guides.",
    "An online platform for sharing information and resources.",
    "A web service offering data and connectivity features.",
    "A reference site with detailed specifications and examples.",
    "A community-driven repository of knowledge and articles.",
];

const CANNED_TAGS: &[&[&str]] = &[
    &["tech", "docs", "reference"],
    &["platform", "web", "service"],
    &["api", "data", "integration"],
    &["spec", "guide", "example"],
    &["community", "knowledge", "open"],
];

/// A running inference stub server.
///
/// The background task is aborted when this handle is dropped.
pub struct InferenceStub {
    pub port: u16,
    /// `http://127.0.0.1:<port>` — pass as `GROQ_BASE_URL` to the runner.
    pub base_url: String,
    /// Completions requests answered since spawn — see [`Self::served`].
    served: Arc<AtomicU64>,
    /// Barriers, arms and the request log — shared with the router.
    control: Arc<Control>,
    /// The scripted-reply queue and the captured request bodies / hits.
    wire: Arc<WireLog>,
    handle: tokio::task::JoinHandle<()>,
}

/// Spawn-time configuration for an [`InferenceStub`].
///
/// Everything here is opt-in: [`InferenceStub::spawn_with`] is
/// `builder(mode).spawn()`, so a caller that never touches the builder gets
/// the pre-R923 wire behaviour byte for byte.
pub struct StubBuilder {
    mode: StubMode,
    delay_ms: u64,
    replies: Vec<Reply>,
}

impl StubBuilder {
    /// Sleep this long between consecutive SSE frames of every response.
    ///
    /// For streaming and busy-state rendering only. It is NOT a rendezvous —
    /// an assertion racing a sleep is the flake the e2e README forbids; use a
    /// barrier ([`InferenceStub::arm`]) to hold a turn in flight.
    pub fn delay_ms(mut self, ms: u64) -> Self {
        self.delay_ms = ms;
        self
    }

    /// Answer the first `replies.len()` inference requests (either wire) with
    /// these, in order, ahead of the mode — for tests that script a tool loop
    /// turn by turn, or an error status on the wire. Once the queue is empty
    /// the mode (or, on `/v1/messages`, a default `stub-ok` message) answers.
    pub fn replies(mut self, replies: impl IntoIterator<Item = impl Into<Reply>>) -> Self {
        self.replies = replies.into_iter().map(Into::into).collect();
        self
    }

    pub async fn spawn(self) -> anyhow::Result<InferenceStub> {
        if let StubMode::Script(script) = &self.mode {
            script.validate()?;
        }
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        let base_url = format!("http://127.0.0.1:{port}");

        let served = Arc::new(AtomicU64::new(0));
        let control = Arc::new(Control::default());
        let wire = Arc::new(WireLog {
            replies: Mutex::new(self.replies.into()),
            bodies: Mutex::new(Vec::new()),
            hits: Mutex::new(Vec::new()),
        });
        let app = Router::new()
            .route("/v1/chat/completions", post(handle_completions))
            .route("/v1/messages", post(handle_messages))
            .route("/v1/models", get(handle_models))
            .route("/control/arm", post(handle_arm))
            .route("/control/release", post(handle_release))
            .route("/control/state", get(handle_state))
            .with_state(StubState {
                mode: self.mode,
                delay_ms: self.delay_ms,
                served: Arc::clone(&served),
                control: Arc::clone(&control),
                wire: Arc::clone(&wire),
            });
        let app = app.layer(axum::middleware::from_fn_with_state(Arc::clone(&wire), log_hit));

        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });

        Ok(InferenceStub {
            port,
            base_url,
            served,
            control,
            wire,
            handle,
        })
    }
}

impl InferenceStub {
    /// Spawn a stub on an OS-assigned free port, answering with canned
    /// summaries. Unchanged behaviour for every pre-R267-F9 caller.
    pub async fn spawn() -> anyhow::Result<Self> {
        Self::spawn_with(StubMode::default()).await
    }

    /// Spawn a stub in an explicit [`StubMode`].
    ///
    /// One stub serves any number of concurrent sessions — axum handles each
    /// request independently and the mode is immutable, so N booked agents can
    /// share a single port rather than needing a stub each.
    pub async fn spawn_with(mode: StubMode) -> anyhow::Result<Self> {
        Self::builder(mode).spawn().await
    }

    /// Configure a stub before spawning it (`delay_ms`, R923-F2).
    pub fn builder(mode: StubMode) -> StubBuilder {
        StubBuilder { mode, delay_ms: 0, replies: Vec::new() }
    }

    /// Hold the next matching completion(s) at a barrier. Same as
    /// `POST /control/arm`.
    pub fn arm(&self, arm: BarrierArm) {
        self.control.arm(arm);
    }

    /// Release a barrier (latched). Returns how many responses were parked at
    /// it. Same as `POST /control/release`.
    pub fn release(&self, barrier: &str) -> usize {
        self.control.barriers.release(barrier)
    }

    /// Responses currently parked at `barrier`.
    pub fn waiting(&self, barrier: &str) -> usize {
        self.control.barriers.waiting(barrier)
    }

    /// Resolve once at least `n` responses are parked at `barrier`.
    ///
    /// The rendezvous a scenario asserts behind: it wakes on every barrier
    /// state change, so there is no polling interval to race. Wrap it in a
    /// timeout — a response that never arrives waits forever.
    pub async fn until_waiting(&self, barrier: &str, n: usize) {
        self.control.barriers.until_waiting(barrier, n).await
    }

    /// The `GET /control/state` document.
    pub fn state(&self) -> Value {
        self.control.state(self.served())
    }

    /// Completions endpoint URL for direct use in tests.
    pub fn completions_url(&self) -> String {
        format!("{}/v1/chat/completions", self.base_url)
    }

    /// `http://127.0.0.1:<port>/v1/messages` — the Anthropic Messages route.
    pub fn messages_url(&self) -> String {
        format!("{}/v1/messages", self.base_url)
    }

    /// Every JSON body POSTed to either inference route, in arrival order.
    /// The tool loop's whole contract is "what goes out on request N+1 given
    /// what came back on request N"; this is how a test reads request N+1.
    pub fn requests(&self) -> Vec<Value> {
        self.wire.bodies.lock().unwrap().clone()
    }

    /// `"METHOD /path"` of every request the server saw, including routes it
    /// does not model (those 404) — for a probe that only needs to know
    /// whether a client reached the wire at all.
    pub fn hits(&self) -> Vec<String> {
        self.wire.hits.lock().unwrap().clone()
    }

    /// How many completions requests this stub has answered.
    ///
    /// The liveness oracle for a driver that books agents: zero means nothing
    /// reached the model, however many sessions the daemon reports minting.
    pub fn served(&self) -> u64 {
        self.served.load(Ordering::Relaxed)
    }
}

impl Drop for InferenceStub {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

/// What the router carries: the immutable mode plus the completions counter.
///
/// R267-F10 added the counter because a booking driver cannot otherwise tell
/// "N agents streamed" from "N agents were booked and never spoke" — and a
/// memory scenario that measures the second while claiming the first passes
/// vacuously, which is exactly the failure `memProbe.ts:20` records.
#[derive(Clone)]
struct StubState {
    wire: Arc<WireLog>,
    mode: StubMode,
    delay_ms: u64,
    served: Arc<AtomicU64>,
    control: Arc<Control>,
}

/// One scripted answer from [`StubBuilder::replies`], either wire.
#[derive(Clone, Debug)]
pub enum Reply {
    /// 200 `text/event-stream`; each entry is one frame WITHOUT its trailing
    /// blank line (the server appends `\n\n`).
    Sse(Vec<String>),
    /// 200 `application/json` — a non-streaming completion.
    Json(Value),
    /// A non-2xx answer: `body` verbatim as `application/json`, plus an
    /// optional `Retry-After` in seconds.
    Error { status: u16, retry_after: Option<u32>, body: String },
}

impl From<MessagesTurn> for Reply {
    fn from(turn: MessagesTurn) -> Self {
        Reply::Sse(turn.frames().into_iter().map(|f| f.trim_end().to_string()).collect())
    }
}

impl Reply {
    fn into_response(self) -> Response<Body> {
        let (status, content_type, retry_after, body) = match self {
            Reply::Sse(frames) => (200, "text/event-stream", None, frames.iter().map(|f| format!("{f}\n\n")).collect()),
            Reply::Json(v) => (200, "application/json", None, v.to_string()),
            Reply::Error { status, retry_after, body } => (status, "application/json", retry_after, body),
        };
        let mut resp = Response::builder()
            .status(StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR))
            .header("content-type", content_type);
        if let Some(secs) = retry_after {
            resp = resp.header("retry-after", secs.to_string());
        }
        resp.body(Body::from(body)).unwrap()
    }
}

/// The scripted-reply queue plus everything the server saw on the wire.
struct WireLog {
    replies: Mutex<VecDeque<Reply>>,
    bodies: Mutex<Vec<Value>>,
    hits: Mutex<Vec<String>>,
}

async fn log_hit(
    State(wire): State<Arc<WireLog>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response<Body> {
    wire.hits.lock().unwrap().push(format!("{} {}", req.method(), req.uri().path()));
    next.run(req).await
}

/// `POST /v1/messages` — the Anthropic wire. Answers the next queued
/// [`Reply`], else a default `stub-ok` message (SSE when the request asked
/// for `stream`, a plain JSON message otherwise).
async fn handle_messages(
    State(state): State<StubState>,
    Json(body): Json<Value>,
) -> Response<Body> {
    state.served.fetch_add(1, Ordering::Relaxed);
    state.wire.bodies.lock().unwrap().push(body.clone());
    if let Some(reply) = state.wire.replies.lock().unwrap().pop_front() {
        return reply.into_response();
    }
    if body["stream"].as_bool().unwrap_or(false) {
        return Reply::from(MessagesTurn::text("stub-ok").usage(1, 1)).into_response();
    }
    Reply::Json(json!({
        "id": "msg_stub",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "stub-ok"}],
        "model": "stub",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }))
    .into_response()
}

async fn handle_completions(
    State(state): State<StubState>,
    Json(body): Json<Value>,
) -> Response<Body> {
    state.served.fetch_add(1, Ordering::Relaxed);
    state.wire.bodies.lock().unwrap().push(body.clone());
    if let Some(reply) = state.wire.replies.lock().unwrap().pop_front() {
        return reply.into_response();
    }
    let (steps, fired) = match &state.mode {
        StubMode::CannedSummary => {
            let url = extract_url_from_messages(&body);
            let (summary, tags) = canned_response(url.as_deref());
            let frames = canned_frames(&format!("{summary} Tags: {}", tags.join(", ")));
            (frames.into_iter().map(Step::Frame).collect(), None)
        }
        StubMode::Lorem {
            tokens,
            chunk_tokens,
        } => {
            let frames = lorem_frames(*tokens, *chunk_tokens, lorem_offset(&body));
            (frames.into_iter().map(Step::Frame).collect(), None)
        }
        StubMode::Script(script) => match script.select(&body) {
            Some(rule) => (rule.steps(&trailing_tool_results(&body)), Some(rule.id.clone())),
            None => {
                state.control.record(&body, None, None, true);
                return unmatched_response(script, &body);
            }
        },
    };
    let barrier = state.control.admit(&body, fired);

    // Fast path: nothing to pace and nothing to hold → one buffer, exactly
    // the pre-R923 wire behaviour. The perf lane (memory_growth.rs) lives
    // here and must not pay for a stream it never asked for.
    let frames: Option<Vec<&str>> = steps
        .iter()
        .map(|s| match s {
            Step::Frame(f) => Some(f.as_str()),
            _ => None,
        })
        .collect();
    let body = match frames {
        Some(frames) if barrier.is_none() && state.delay_ms == 0 => Body::from(frames.concat()),
        _ => step_body(
            barrier.map(Step::Await).into_iter().chain(steps).collect(),
            Arc::clone(&state.control.barriers),
            Duration::from_millis(state.delay_ms),
        ),
    };

    Response::builder()
        .status(StatusCode::OK)
        .header(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        )
        .header("cache-control", HeaderValue::from_static("no-cache"))
        .body(body)
        .unwrap()
}

/// One unit of a streamed response.
///
/// The shape R923-F6's script engine lowers its `emit` lists into: `text` /
/// `tool_use` become `Frame`s, `await: barrier:<name>` becomes `Await`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// A complete `data: …\n\n` SSE frame.
    Frame(String),
    /// Park the response until the named barrier is released.
    Await(String),
    /// Sleep before the next step — a script text's own `delay_ms`.
    Sleep(Duration),
}

/// Stream `steps` as a response body: frames are written as they are reached,
/// `delay` apart; an `Await` parks only THIS response on its barrier.
fn step_body(steps: VecDeque<Step>, barriers: Arc<Barriers>, delay: Duration) -> Body {
    let stream = futures::stream::unfold(
        (steps, barriers, false),
        move |(mut steps, barriers, mut sent_any)| async move {
            loop {
                match steps.pop_front()? {
                    Step::Await(name) => barriers.wait(&name).await,
                    Step::Sleep(d) => tokio::time::sleep(d).await,
                    Step::Frame(frame) => {
                        if sent_any && !delay.is_zero() {
                            tokio::time::sleep(delay).await;
                        }
                        sent_any = true;
                        return Some((Ok::<_, Infallible>(frame), (steps, barriers, sent_any)));
                    }
                }
            }
        },
    );
    Body::from_stream(stream)
}

/// "Hold the next `count` completions matching this at `barrier`."
///
/// How a mode with no script of its own (Lorem, CannedSummary) gets a turn
/// held in flight. Matchers are ANDed; an absent one matches everything, so
/// `{"barrier": "b"}` alone holds the very next completion, whoever sends it.
/// Arms are consumed in the order they were armed — first match wins.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BarrierArm {
    pub barrier: String,
    /// Exact match on the request's `model` (the role axis R923-F6 keys on).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Substring of any message's string `content` — how a scenario picks out
    /// the conversation it poked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contains: Option<String>,
    /// How many matching completions to hold. Default 1.
    #[serde(default = "one")]
    pub count: usize,
}

fn one() -> usize {
    1
}

impl BarrierArm {
    /// Hold the next completion, from any conversation, at `barrier`.
    pub fn next(barrier: impl Into<String>) -> Self {
        Self {
            barrier: barrier.into(),
            model: None,
            contains: None,
            count: 1,
        }
    }

    pub fn contains(mut self, needle: impl Into<String>) -> Self {
        self.contains = Some(needle.into());
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn count(mut self, n: usize) -> Self {
        self.count = n;
        self
    }

    fn matches(&self, body: &Value) -> bool {
        if let Some(m) = &self.model {
            if body.get("model").and_then(Value::as_str) != Some(m.as_str()) {
                return false;
            }
        }
        if let Some(needle) = &self.contains {
            let hit = body
                .get("messages")
                .and_then(Value::as_array)
                .is_some_and(|ms| {
                    ms.iter().any(|m| {
                        m.get("content")
                            .and_then(Value::as_str)
                            .is_some_and(|c| c.contains(needle.as_str()))
                    })
                });
            if !hit {
                return false;
            }
        }
        true
    }
}

/// R923-F6 — a scenario's model side as data (W354 §2): rules per role, each
/// a predicate over ONE request plus what to stream when it matches.
///
/// JSON shape, one file per scenario:
///
/// ```json
/// {"script": "leader-dispatches",
///  "roles": {
///    "script:leader": [
///      {"id": "L1", "when": {"turn_index": 0},
///       "emit": [{"text": "Dispatching."},
///                {"tool_use": {"name": "party_dispatch",
///                              "input": {"target": {"character": "Lorem0"}, "prompt": "…"}}}],
///       "stop": "tool_use"},
///      {"id": "L2-both", "when": {"after_tool_result": "inbox_read",
///                                 "tool_result_contains": ["rs-count", "ts-count"]},
///       "emit": [{"await": "barrier:collect"}, {"text": "Got both: $TOOL_RESULT"}]},
///      {"id": "L2-partial", "when": {"after_tool_result": "inbox_read"},
///       "emit": [{"text": "PARTIAL"}]}]}}
/// ```
///
/// - **Role** = the request's `model`, exactly. A fixture character's
///   subclass carries the role id as its model, so roles are addressed
///   without scraping prompts.
/// - **Rules are tried in order; first match wins.** An absent `when` key
///   matches anything, so put the general rule after the specific one.
/// - **No match is an HTTP 500** with the conversation tail — never a
///   fallthrough. A silent fallback is how a scenario passes vacuously.
/// - Tool names are the WIRE names (`party_dispatch`, not `party.dispatch`:
///   the runner rewrites dots, `sanitize_tool_name_for_wire`).
/// - A `party_dispatch` / `party_assist` call must target
///   `{"character": …}` (operator call 2026-10-04): the schema has no `sigil`
///   key, and a bare dispatch auto-picks the first eligible subclass, which
///   may be a real provider. [`Script::validate`] refuses anything else.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Script {
    /// Scenario name — appears in validation errors.
    #[serde(default)]
    pub script: String,
    /// Role (request `model`) → its rules, in match order.
    pub roles: BTreeMap<String, Vec<Rule>>,
}

/// One script rule. `id` is what the fired-rule oracle records, so it must
/// be unique across the whole script.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rule {
    pub id: String,
    #[serde(default)]
    pub when: When,
    #[serde(default)]
    pub emit: Vec<Emit>,
    /// Optional, and checked against `emit`: `tool_use` iff the rule emits a
    /// tool call. The wire's `finish_reason` is derived from `emit` either way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Stop>,
}

/// A rule's predicate — every key is read off the request, all are ANDed,
/// and an absent key matches anything.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct When {
    /// Assistant messages already in the body (see [`turn_index`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_index: Option<usize>,
    /// The body ends in tool results and one of them answers a call to this
    /// (wire) tool name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_tool_result: Option<String>,
    /// Every needle appears in one of the trailing tool results. A string or
    /// an array of strings.
    #[serde(
        default,
        deserialize_with = "one_or_many",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub tool_result_contains: Vec<String>,
}

/// One `emit` entry. Exactly one of `text`, `tool_use`, `await`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Emit {
    Text(TextStep),
    ToolUse(ToolUseStep),
    Await(AwaitStep),
}

/// Streamed content. `$TOOL_RESULT` is replaced by the last trailing tool
/// result. `chunk_tokens` splits it into frames of that many words (default:
/// one frame); `delay_ms` sleeps between those frames.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextStep {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delay_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolUseStep {
    pub tool_use: ToolCall,
}

/// A tool call the model makes. `name` is the wire name.
///
/// A string anywhere in `input` that reads `$TOOL_RESULT:<json-pointer>` is
/// replaced by that pointer into the last trailing tool result parsed as JSON
/// (R923-T9) — how a script addresses an id only known at run time, e.g.
/// `{"session_id": "$TOOL_RESULT:/session_id"}` off a dispatch receipt. A
/// pointer that misses leaves the string as written, so the call fails loudly
/// downstream instead of silently addressing something else.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCall {
    pub name: String,
    #[serde(default = "empty_object")]
    pub input: Value,
}

/// Park the response at a barrier (`"barrier:<name>"`; the prefix is
/// optional) until `POST /control/release` frees it. Frames emitted before
/// it have already streamed, so the client sees a live turn in flight.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AwaitStep {
    #[serde(rename = "await")]
    pub barrier: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stop {
    EndTurn,
    ToolUse,
}

fn empty_object() -> Value {
    json!({})
}

fn one_or_many<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(String),
        Many(Vec<String>),
    }
    Ok(match OneOrMany::deserialize(d)? {
        OneOrMany::One(s) => vec![s],
        OneOrMany::Many(v) => v,
    })
}

/// Wire names of the tools that book a session — the ones a script must
/// target by character.
const DISPATCH_TOOLS: &[&str] = &["party_dispatch", "party_assist"];

impl Script {
    /// Parse and [`validate`](Self::validate) a script document.
    pub fn from_value(v: Value) -> anyhow::Result<Self> {
        let script: Self = serde_json::from_value(v)?;
        script.validate()?;
        Ok(script)
    }

    /// [`Self::from_value`] from JSON text (a scenario's fixture file).
    pub fn from_json(s: &str) -> anyhow::Result<Self> {
        Self::from_value(serde_json::from_str(s)?)
    }

    /// Refuse a script that would make the oracle ambiguous or book a real
    /// provider. [`StubBuilder::spawn`] runs this too.
    pub fn validate(&self) -> anyhow::Result<()> {
        let name = &self.script;
        let mut ids = std::collections::HashSet::new();
        for rules in self.roles.values() {
            for rule in rules {
                anyhow::ensure!(
                    ids.insert(rule.id.as_str()),
                    "script {name:?}: rule id {:?} appears twice — the fired-rule oracle could not tell them apart",
                    rule.id
                );
                let calls = rule.emit.iter().any(|e| matches!(e, Emit::ToolUse(_)));
                if let Some(stop) = rule.stop {
                    anyhow::ensure!(
                        (stop == Stop::ToolUse) == calls,
                        "script {name:?}: rule {:?} says stop {stop:?} but {} a tool call",
                        rule.id,
                        if calls { "emits" } else { "does not emit" }
                    );
                }
                for e in &rule.emit {
                    let Emit::ToolUse(ToolUseStep { tool_use }) = e else {
                        continue;
                    };
                    if DISPATCH_TOOLS.contains(&tool_use.name.as_str()) {
                        anyhow::ensure!(
                            tool_use.input.pointer("/target/character").and_then(Value::as_str).is_some(),
                            "script {name:?}: rule {:?} calls {} without target.character — scripts \
                             target by CHARACTER, never sigil and never bare (a bare dispatch may book \
                             a real provider)",
                            rule.id,
                            tool_use.name
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// The first rule of this request's role whose `when` holds.
    fn select(&self, body: &Value) -> Option<&Rule> {
        let model = body.get("model").and_then(Value::as_str)?;
        let turn = turn_index(body);
        let results = trailing_tool_results(body);
        self.roles
            .get(model)?
            .iter()
            .find(|r| r.when.matches(turn, &results))
    }
}

impl When {
    fn matches(&self, turn: usize, results: &[ToolResult]) -> bool {
        self.turn_index.is_none_or(|t| t == turn)
            && self
                .after_tool_result
                .as_deref()
                .is_none_or(|n| results.iter().any(|r| r.name.as_deref() == Some(n)))
            && self
                .tool_result_contains
                .iter()
                .all(|needle| results.iter().any(|r| r.content.contains(needle.as_str())))
    }
}

impl Rule {
    /// Lower `emit` into wire steps: text and tool calls become frames,
    /// `await` a hold, and the finish reason follows from whether any tool
    /// call was emitted.
    fn steps(&self, results: &[ToolResult]) -> VecDeque<Step> {
        let last = results.last().map_or("", |r| r.content.as_str());
        let mut out = VecDeque::new();
        let mut calls = 0usize;
        let mut words = 0usize;
        for e in &self.emit {
            match e {
                Emit::Text(t) => {
                    let text = t.text.replace("$TOOL_RESULT", last);
                    words += text.split_whitespace().count();
                    for (i, chunk) in text_chunks(&text, t.chunk_tokens).into_iter().enumerate() {
                        if let (true, Some(ms)) = (i > 0, t.delay_ms) {
                            out.push_back(Step::Sleep(Duration::from_millis(ms)));
                        }
                        out.push_back(Step::Frame(content_frame(&chunk)));
                    }
                }
                Emit::ToolUse(ToolUseStep { tool_use }) => {
                    let id = format!("call_{}_{calls}", self.id);
                    let call = ToolCall {
                        name: tool_use.name.clone(),
                        input: fill_tool_result(&tool_use.input, last),
                    };
                    out.extend(tool_call_frames(calls, &id, &call).into_iter().map(Step::Frame));
                    calls += 1;
                }
                Emit::Await(AwaitStep { barrier }) => {
                    let name = barrier.strip_prefix("barrier:").unwrap_or(barrier);
                    out.push_back(Step::Await(name.to_string()));
                }
            }
        }
        let reason = if calls > 0 { FinishReason::ToolCalls } else { FinishReason::Stop };
        out.extend(finish_frames(reason, words + 20 * calls).into_iter().map(Step::Frame));
        out
    }
}

/// `text` as frames of `chunk_tokens` words, whitespace preserved so the
/// frames re-join to exactly `text`. `None` (or 0) is one frame.
fn text_chunks(text: &str, chunk_tokens: Option<usize>) -> Vec<String> {
    let Some(n) = chunk_tokens.filter(|n| *n > 0) else {
        return vec![text.to_string()];
    };
    let words: Vec<&str> = text.split_inclusive(char::is_whitespace).collect();
    if words.is_empty() {
        return vec![String::new()];
    }
    words.chunks(n).map(|c| c.concat()).collect()
}

/// One tool call at `index`, streamed the way OpenAI does: the first
/// `tool_calls` delta carries `id` + `name`, the arguments arrive split
/// across two more. Splitting is deliberate — a stub that sent the arguments
/// whole would never exercise the runner's per-index accumulator.
/// Resolve `$TOOL_RESULT:<json-pointer>` strings in a tool input (see [`ToolCall`]).
fn fill_tool_result(input: &Value, last: &str) -> Value {
    match input {
        Value::String(s) => s
            .strip_prefix("$TOOL_RESULT:")
            .and_then(|ptr| serde_json::from_str::<Value>(last).ok()?.pointer(ptr).cloned())
            .unwrap_or_else(|| input.clone()),
        Value::Array(xs) => Value::Array(xs.iter().map(|x| fill_tool_result(x, last)).collect()),
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, v)| (k.clone(), fill_tool_result(v, last)))
                .collect(),
        ),
        _ => input.clone(),
    }
}

fn tool_call_frames(index: usize, id: &str, call: &ToolCall) -> Vec<String> {
    let args = call.input.to_string();
    let mid = args.len() / 2;
    let mid = (mid..=args.len()).find(|i| args.is_char_boundary(*i)).unwrap_or(args.len());
    let (a, b) = args.split_at(mid);
    let frame = |tc: ToolCallDelta| {
        SseFrame::from(ChatCompletionChunk::delta(Delta::tool_call(tc))).to_sse()
    };
    let fragment = |arguments: &str| ToolCallDelta {
        index,
        id: None,
        kind: None,
        function: FunctionDelta { name: None, arguments: arguments.to_string() },
    };
    vec![
        frame(ToolCallDelta {
            index,
            id: Some(id.to_string()),
            kind: Some(ToolCallKind::Function),
            function: FunctionDelta { name: Some(call.name.clone()), arguments: String::new() },
        }),
        frame(fragment(a)),
        frame(fragment(b)),
    ]
}

/// The 500 a script answers when no rule matches: loud, and carrying enough
/// of the conversation to write the missing rule from.
fn unmatched_response(script: &Script, body: &Value) -> Response<Body> {
    let model = body.get("model").and_then(Value::as_str);
    let results = trailing_tool_results(body);
    let rules: Option<Vec<&str>> = model
        .and_then(|m| script.roles.get(m))
        .map(|rs| rs.iter().map(|r| r.id.as_str()).collect());
    let messages = body.get("messages").and_then(Value::as_array);
    let tail: Vec<&Value> = messages
        .map(|ms| ms.iter().skip(ms.len().saturating_sub(4)).collect())
        .unwrap_or_default();
    let doc = json!({"error": {
        "type": "script_no_match",
        "message": format!(
            "inference stub: script {:?} has no rule matching model {model:?} at turn_index {} — \
             scripts never fall through",
            script.script,
            turn_index(body)
        ),
        "model": model,
        "turn_index": turn_index(body),
        "after_tool_result": results.iter().map(|r| r.name.as_deref()).collect::<Vec<_>>(),
        "role_rules": rules,
        "tail": tail,
    }});
    Response::builder()
        .status(StatusCode::INTERNAL_SERVER_ERROR)
        .header("content-type", HeaderValue::from_static("application/json"))
        .body(Body::from(doc.to_string()))
        .unwrap()
}

/// Assistant messages already in the body — the turn ordinal read off the
/// request (W354 §2), never counted by the stub.
fn turn_index(body: &Value) -> usize {
    body.get("messages")
        .and_then(Value::as_array)
        .map(|ms| {
            ms.iter()
                .filter(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
                .count()
        })
        .unwrap_or(0)
}

/// One `role: tool` message at the end of a body.
struct ToolResult {
    /// The wire name of the tool it answers — resolved through its
    /// `tool_call_id` against the assistant `tool_calls`, else its own `name`.
    name: Option<String>,
    content: String,
}

/// The body's trailing run of `role: tool` messages, in order. Empty unless
/// the body ends in tool results. A run, not just the last message, because
/// a turn that made two calls comes back with two results.
fn trailing_tool_results(body: &Value) -> Vec<ToolResult> {
    let Some(ms) = body.get("messages").and_then(Value::as_array) else {
        return Vec::new();
    };
    let is_tool = |m: &Value| m.get("role").and_then(Value::as_str) == Some("tool");
    let start = ms.len() - ms.iter().rev().take_while(|m| is_tool(m)).count();
    let call_name = |id: &str| {
        ms.iter()
            .filter_map(|m| m.get("tool_calls").and_then(Value::as_array))
            .flatten()
            .find(|c| c.get("id").and_then(Value::as_str) == Some(id))
            .and_then(|c| c.pointer("/function/name").and_then(Value::as_str))
            .map(str::to_string)
    };
    ms[start..]
        .iter()
        .map(|m| ToolResult {
            name: m
                .get("tool_call_id")
                .and_then(Value::as_str)
                .and_then(call_name)
                .or_else(|| m.get("name").and_then(Value::as_str).map(str::to_string)),
            content: match m.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
                None => String::new(),
            },
        })
        .collect()
}

/// Which conversation a request belongs to, read off the request: its model
/// plus its first user message, which every later turn re-sends unchanged.
/// Two conversations of one role opened with the identical prompt share a
/// key — give scenario conversations distinct prompts.
fn conversation_key(body: &Value) -> String {
    let model = body.get("model").and_then(Value::as_str).unwrap_or("");
    let first_user = body
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|ms| {
            ms.iter()
                .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
        })
        .and_then(|m| m.get("content"))
        .map(Value::to_string)
        .unwrap_or_default();
    let h = blake3::hash(format!("{model}\0{first_user}").as_bytes());
    h.to_hex()[..12].to_string()
}

/// One completion request as `GET /control/state` reports it.
#[derive(Clone, Debug, Serialize)]
struct RequestRecord {
    model: Option<String>,
    /// See [`conversation_key`].
    conversation: String,
    /// See [`turn_index`].
    turn_index: usize,
    barrier: Option<String>,
    /// The script [`Rule`] that answered this request, if any.
    fired: Option<String>,
    /// A script had no rule for it and it was answered 500.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    unmatched: bool,
    /// What the client's last tool call returned — how a scenario reads a
    /// tool's result (e.g. a `party.dispatch` receipt) as the model saw it.
    tool_result: Option<String>,
}

/// The oracle (W354 §2): one conversation's fired rules, in arrival order.
#[derive(Clone, Debug, Serialize)]
struct ConversationRecord {
    conversation: String,
    model: Option<String>,
    fired: Vec<String>,
}

/// Router-shared control state. Every lock here is a short `std` critical
/// section; none is held across an `.await`.
#[derive(Default)]
struct Control {
    barriers: Arc<Barriers>,
    arms: Mutex<Vec<BarrierArm>>,
    requests: Mutex<Vec<RequestRecord>>,
    conversations: Mutex<Vec<ConversationRecord>>,
}

impl Control {
    fn arm(&self, arm: BarrierArm) {
        if arm.count > 0 {
            self.arms.lock().unwrap().push(arm);
        }
    }

    /// Log the request and consume the first matching arm, if any.
    fn admit(&self, body: &Value, fired: Option<String>) -> Option<String> {
        let barrier = {
            let mut arms = self.arms.lock().unwrap();
            arms.iter().position(|a| a.matches(body)).map(|i| {
                let name = arms[i].barrier.clone();
                arms[i].count -= 1;
                if arms[i].count == 0 {
                    arms.remove(i);
                }
                name
            })
        };
        self.record(body, barrier.clone(), fired, false);
        barrier
    }

    /// Log a request, and append its fired rule to its conversation.
    fn record(&self, body: &Value, barrier: Option<String>, fired: Option<String>, unmatched: bool) {
        let model = body.get("model").and_then(Value::as_str).map(str::to_string);
        let conversation = conversation_key(body);
        if let Some(id) = &fired {
            let mut convs = self.conversations.lock().unwrap();
            match convs.iter_mut().find(|c| c.conversation == conversation) {
                Some(c) => c.fired.push(id.clone()),
                None => convs.push(ConversationRecord {
                    conversation: conversation.clone(),
                    model: model.clone(),
                    fired: vec![id.clone()],
                }),
            }
        }
        self.requests.lock().unwrap().push(RequestRecord {
            model,
            conversation,
            turn_index: turn_index(body),
            barrier,
            fired,
            unmatched,
            tool_result: trailing_tool_results(body).pop().map(|r| r.content),
        });
    }

    fn state(&self, served: u64) -> Value {
        let (waiting, released) = self.barriers.snapshot();
        json!({
            "served": served,
            "requests": *self.requests.lock().unwrap(),
            "conversations": *self.conversations.lock().unwrap(),
            "armed": *self.arms.lock().unwrap(),
            "waiting": waiting,
            "released": released,
        })
    }
}

/// Named, latched barriers. Each has its own `watch` channel, so a parked
/// response waits on that channel alone — never on the registry lock.
#[derive(Default)]
struct Barriers {
    slots: Mutex<HashMap<String, BarrierSlot>>,
    /// Bumped on every park / unpark / release, for [`Self::until_waiting`].
    epoch: watch::Sender<u64>,
}

struct BarrierSlot {
    open: watch::Sender<bool>,
    waiting: usize,
}

impl BarrierSlot {
    fn new() -> Self {
        Self {
            open: watch::channel(false).0,
            waiting: 0,
        }
    }
}

impl Barriers {
    fn bump(&self) {
        self.epoch.send_modify(|e| *e += 1);
    }

    async fn wait(self: &Arc<Self>, name: &str) {
        let mut open = {
            let mut slots = self.slots.lock().unwrap();
            let slot = slots.entry(name.to_string()).or_insert_with(BarrierSlot::new);
            if *slot.open.borrow() {
                return;
            }
            slot.waiting += 1;
            slot.open.subscribe()
        };
        self.bump();
        // Unpark on release AND on cancellation (client hung up), so
        // `waiting` never counts a response nobody is holding any more.
        let _parked = Parked {
            barriers: Arc::clone(self),
            name: name.to_string(),
        };
        let _ = open.wait_for(|o| *o).await;
    }

    fn release(&self, name: &str) -> usize {
        let was_waiting = {
            let mut slots = self.slots.lock().unwrap();
            let slot = slots.entry(name.to_string()).or_insert_with(BarrierSlot::new);
            slot.open.send_replace(true);
            slot.waiting
        };
        self.bump();
        was_waiting
    }

    fn waiting(&self, name: &str) -> usize {
        self.slots
            .lock()
            .unwrap()
            .get(name)
            .map_or(0, |s| s.waiting)
    }

    async fn until_waiting(&self, name: &str, n: usize) {
        // Subscribe BEFORE checking, so a park between the check and the
        // await still marks the receiver changed.
        let mut epoch = self.epoch.subscribe();
        while self.waiting(name) < n {
            if epoch.changed().await.is_err() {
                return;
            }
        }
    }

    /// `(waiting: name → parked count, released: names)`, both sorted.
    fn snapshot(&self) -> (BTreeMap<String, usize>, Vec<String>) {
        let slots = self.slots.lock().unwrap();
        let waiting = slots
            .iter()
            .filter(|(_, s)| s.waiting > 0)
            .map(|(n, s)| (n.clone(), s.waiting))
            .collect();
        let mut released: Vec<String> = slots
            .iter()
            .filter(|(_, s)| *s.open.borrow())
            .map(|(n, _)| n.clone())
            .collect();
        released.sort();
        (waiting, released)
    }
}

/// Drop guard for a parked response.
struct Parked {
    barriers: Arc<Barriers>,
    name: String,
}

impl Drop for Parked {
    fn drop(&mut self) {
        if let Some(slot) = self.barriers.slots.lock().unwrap().get_mut(&self.name) {
            slot.waiting -= 1;
        }
        self.barriers.bump();
    }
}

async fn handle_arm(State(state): State<StubState>, Json(arm): Json<BarrierArm>) -> Json<Value> {
    state.control.arm(arm.clone());
    Json(json!({ "armed": arm }))
}

#[derive(Deserialize)]
struct ReleaseReq {
    barrier: String,
}

async fn handle_release(
    State(state): State<StubState>,
    Json(req): Json<ReleaseReq>,
) -> Json<Value> {
    let was_waiting = state.control.barriers.release(&req.barrier);
    Json(json!({ "barrier": req.barrier, "was_waiting": was_waiting }))
}

async fn handle_state(State(state): State<StubState>) -> Json<Value> {
    Json(state.control.state(state.served.load(Ordering::Relaxed)))
}

async fn handle_models() -> Json<Value> {
    Json(json!({
        "data": [{"id": "stub-1b", "object": "model"}]
    }))
}

/// Build the four-frame SSE body the runner's `parse_sse_frame` consumes.
#[cfg(test)]
fn build_sse_response(content: &str) -> String {
    canned_frames(content).concat()
}

/// The canned path's four frames, one `data: …\n\n` string each.
fn canned_frames(content: &str) -> Vec<String> {
    let mut out = vec![content_frame(content)];
    out.extend(finish_frames(FinishReason::Stop, 20));
    out
}

fn content_frame(text: &str) -> String {
    SseFrame::from(ChatCompletionChunk::delta(Delta::content(text))).to_sse()
}

/// The tail every response ends with: `finish_reason`, usage, `[DONE]`.
/// Without these the runner never ends the turn.
fn finish_frames(reason: FinishReason, completion_tokens: usize) -> [String; 3] {
    let usage = Usage {
        prompt_tokens: 10,
        completion_tokens,
        total_tokens: 10 + completion_tokens,
    };
    [
        SseFrame::from(ChatCompletionChunk::finish(reason)).to_sse(),
        SseFrame::from(ChatCompletionChunk::usage(usage)).to_sse(),
        SseFrame::Done.to_sse(),
    ]
}

/// Where in [`LOREM_WORDS`] this request's text starts.
///
/// Derived from the request body rather than a counter so the stub stays
/// deterministic under concurrency: N agents hitting one port interleave in
/// arbitrary order, and a shared `AtomicUsize` would make each one's content
/// depend on that ordering. Hashing the body means "same conversation, same
/// text" regardless of who else is streaming — while still differing *between*
/// sessions, so no two agents send byte-identical transcripts.
fn lorem_offset(body: &Value) -> usize {
    let seed = body
        .get("messages")
        .and_then(Value::as_array)
        .map(|m| {
            // Length plus the last message's content: cheap, and it advances as
            // a conversation grows, so successive turns in one session differ.
            let tail = m
                .last()
                .and_then(|msg| msg.get("content"))
                .and_then(Value::as_str)
                .unwrap_or("");
            format!("{}:{tail}", m.len())
        })
        .unwrap_or_default();
    let h = blake3::hash(seed.as_bytes());
    u16::from_le_bytes([h.as_bytes()[0], h.as_bytes()[1]]) as usize % LOREM_WORDS.len()
}

/// Stream `tokens` lorem words in frames of `chunk_tokens`.
///
/// Frame shape matches `build_sse_response` exactly — the runner's
/// `parse_sse_frame` is the consumer either way; only the number of content
/// frames differs.
#[cfg(test)]
fn build_lorem_sse(tokens: usize, chunk_tokens: usize, offset: usize) -> String {
    lorem_frames(tokens, chunk_tokens, offset).concat()
}

/// [`build_lorem_sse`] as individual frames, so a paced or held response can
/// stream them one at a time.
fn lorem_frames(tokens: usize, chunk_tokens: usize, offset: usize) -> Vec<String> {
    let chunk = chunk_tokens.max(1);
    let mut out = Vec::new();

    let mut emitted = 0usize;
    while emitted < tokens {
        let take = chunk.min(tokens - emitted);
        let mut text = String::new();
        for i in 0..take {
            // Leading space on every word but the very first, so the
            // concatenation of all frames reads as prose — a consumer that
            // joins deltas naively gets sensible text.
            if emitted + i > 0 {
                text.push(' ');
            }
            text.push_str(LOREM_WORDS[(offset + emitted + i) % LOREM_WORDS.len()]);
        }
        out.push(content_frame(&text));
        emitted += take;
    }

    out.extend(finish_frames(FinishReason::Stop, tokens));
    out
}

/// Extract the first `http(s)://…` URL from the message contents.
fn extract_url_from_messages(body: &Value) -> Option<String> {
    let messages = body.get("messages")?.as_array()?;
    // Skip, don't bail. `?` here would abandon the whole scan on the first
    // message with a non-string `content` or no URL — which made this a loop
    // that could never reach its second iteration (clippy `never_loop`, and a
    // real mismatch with the doc comment: a URL in message #2 was unreachable
    // whenever message #1 was a plain system prompt).
    for msg in messages {
        let Some(content) = msg.get("content").and_then(Value::as_str) else {
            continue;
        };
        let Some(start) = content.find("https://").or_else(|| content.find("http://")) else {
            continue;
        };
        let rest = &content[start..];
        let end = rest
            .find(|c: char| {
                c.is_whitespace() || matches!(c, '"' | '\'' | ')' | ']' | '>' | '\n')
            })
            .unwrap_or(rest.len());
        return Some(rest[..end].to_string());
    }
    None
}

/// Return deterministic `(summary, tags)` keyed off `blake3(url)`.
/// Falls back to slot 0 when no URL was found in the request.
fn canned_response(url: Option<&str>) -> (&'static str, &'static [&'static str]) {
    let idx = if let Some(u) = url {
        (blake3::hash(u.as_bytes()).as_bytes()[0] as usize) % CANNED_SUMMARIES.len()
    } else {
        0
    };
    (CANNED_SUMMARIES[idx], CANNED_TAGS[idx])
}

// ─────────────────────────────────────────────────────────────────────────
//  Top-level test — run with:
//  `cargo test -p yah-eval inference_stub::round_trips`
//
//  (The crate is `yah-eval`; `eval` is only the bin name, so the `-p eval`
//  this comment used to carry fails with "did not match any packages".)
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[tokio::test]
async fn round_trips() {
    let stub = InferenceStub::spawn().await.expect("stub spawn");

    // Same URL twice → identical content.
    let a = test_helpers::call_completions(&stub, "https://example.com/page").await;
    let b = test_helpers::call_completions(&stub, "https://example.com/page").await;
    assert_eq!(a, b, "same URL must produce identical content");
    assert!(!a.is_empty(), "response must not be empty");
    assert!(a.contains("Tags:"), "response must include Tags: marker");

    // No URL in body → slot 0 fallback, still non-empty.
    let no_url =
        test_helpers::call_completions_raw(&stub, "Just some text without a URL.").await;
    assert!(!no_url.is_empty(), "no-URL request must still return content");

    // Model listing round-trips.
    let models = test_helpers::call_models(&stub).await;
    let ids: Vec<&str> = models["data"]
        .as_array()
        .expect("data array")
        .iter()
        .filter_map(|m| m["id"].as_str())
        .collect();
    assert!(!ids.is_empty(), "models endpoint must return at least one model");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `chunk_tokens` must actually control frame count, not just text length.
    ///
    /// This is the assertion that makes the stub worth having for R267: the
    /// renderer's cost tracks event count, so a regression that coalesced
    /// frames would silently turn a 2000-event workload into a 1-event one and
    /// every memory measurement taken with it would quietly become meaningless.
    #[test]
    fn lorem_chunking_controls_frame_count() {
        let one_at_a_time = build_lorem_sse(40, 1, 0);
        let ten_at_a_time = build_lorem_sse(40, 10, 0);

        let content_frames = |s: &str| s.matches("\"content\"").count();
        assert_eq!(content_frames(&one_at_a_time), 40);
        assert_eq!(content_frames(&ten_at_a_time), 4);

        // Same words either way — only the framing differs.
        let words = |s: &str| {
            s.lines()
                .filter_map(|l| l.strip_prefix("data: "))
                .filter_map(|p| serde_json::from_str::<Value>(p).ok())
                .filter_map(|v| {
                    v["choices"][0]["delta"]["content"]
                        .as_str()
                        .map(str::to_string)
                })
                .collect::<Vec<_>>()
                .join("")
        };
        assert_eq!(words(&one_at_a_time), words(&ten_at_a_time));
        assert_eq!(words(&one_at_a_time).split_whitespace().count(), 40);
    }

    /// Deterministic per conversation, distinct between conversations.
    ///
    /// Both halves matter: determinism keeps a failing memory run reproducible,
    /// and distinctness keeps N concurrent agents from sending byte-identical
    /// transcripts that string interning could make unrealistically cheap.
    #[test]
    fn lorem_offset_is_deterministic_but_varies_by_conversation() {
        let a = json!({"messages": [{"role": "user", "content": "session one"}]});
        let b = json!({"messages": [{"role": "user", "content": "session two"}]});

        assert_eq!(lorem_offset(&a), lorem_offset(&a));
        assert_ne!(
            build_lorem_sse(20, 1, lorem_offset(&a)),
            build_lorem_sse(20, 1, lorem_offset(&b)),
            "two different conversations must not stream identical text"
        );
    }

    /// A tail frame with `finish_reason: stop` and a `[DONE]` sentinel, same as
    /// the canned path — without these the runner never ends the turn.
    #[test]
    fn lorem_terminates_the_stream() {
        let sse = build_lorem_sse(5, 2, 0);
        assert!(sse.contains(r#""finish_reason":"stop""#));
        assert!(sse.trim_end().ends_with("data: [DONE]"));
        assert!(sse.contains(r#""completion_tokens":5"#));
    }

    #[test]
    fn extract_url_finds_https() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Summarize https://example.com/page please"}
            ]
        });
        assert_eq!(
            extract_url_from_messages(&body),
            Some("https://example.com/page".to_string())
        );
    }

    #[test]
    fn extract_url_returns_none_when_absent() {
        let body = json!({
            "messages": [{"role": "user", "content": "no url here"}]
        });
        assert!(extract_url_from_messages(&body).is_none());
    }

    #[test]
    fn canned_response_is_deterministic() {
        let (s1, t1) = canned_response(Some("https://example.com"));
        let (s2, t2) = canned_response(Some("https://example.com"));
        assert_eq!(s1, s2);
        assert_eq!(t1, t2);
    }

    #[test]
    fn build_sse_response_contains_expected_frames() {
        let sse = build_sse_response("hello tags: a, b");
        assert!(sse.contains("\"content\":\"hello tags: a, b\""));
        assert!(sse.contains("\"finish_reason\":\"stop\""));
        assert!(sse.contains("[DONE]"));
    }
}

/// R923-F2 — barriers and the `/control` plane.
#[cfg(test)]
mod barrier_tests {
    use super::*;
    use std::time::Instant;
    use tokio::time::timeout;

    /// Generous: these bound a hang, they are not the thing under test.
    const BOUND: Duration = Duration::from_secs(10);

    const LOREM: StubMode = StubMode::Lorem {
        tokens: 12,
        chunk_tokens: 1,
    };

    fn body(conv: &str) -> Value {
        json!({
            "model": "stub-1b",
            "stream": true,
            "messages": [{"role": "user", "content": conv}]
        })
    }

    /// POST a completion and return the joined delta text plus whether the
    /// stream reached `[DONE]`.
    async fn complete(url: String, body: Value) -> (String, bool) {
        let text = reqwest::Client::new()
            .post(url)
            .json(&body)
            .send()
            .await
            .expect("completions request")
            .text()
            .await
            .expect("completions body");
        let mut out = String::new();
        let mut done = false;
        for payload in text.lines().filter_map(|l| l.strip_prefix("data: ")) {
            if payload == "[DONE]" {
                done = true;
                break;
            }
            if let Ok(v) = serde_json::from_str::<Value>(payload) {
                if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                    out.push_str(c);
                }
            }
        }
        (out, done)
    }

    #[tokio::test]
    async fn held_response_waits_until_released() {
        let stub = InferenceStub::spawn_with(LOREM).await.unwrap();
        stub.arm(BarrierArm::next("b"));

        let req = tokio::spawn(complete(stub.completions_url(), body("held")));
        timeout(BOUND, stub.until_waiting("b", 1))
            .await
            .expect("request never parked at the barrier");

        assert!(!req.is_finished(), "a held response must not complete");
        let state = stub.state();
        assert_eq!(state["waiting"]["b"], 1);
        assert_eq!(state["requests"][0]["barrier"], "b");
        assert_eq!(state["served"], 1);

        assert_eq!(stub.release("b"), 1);
        let (text, done) = timeout(BOUND, req).await.expect("release did not free it").unwrap();
        assert!(done, "released stream must reach [DONE]");
        assert_eq!(text.split_whitespace().count(), 12);
        assert_eq!(stub.waiting("b"), 0);
        assert_eq!(stub.state()["released"], json!(["b"]));
    }

    /// The deadlock this ticket was budgeted for: one conversation parked must
    /// not stall any other. N-1 stream to completion while the first is held.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn other_conversations_stream_while_one_is_held() {
        const N: usize = 8;
        let stub = InferenceStub::builder(LOREM).delay_ms(1).spawn().await.unwrap();
        stub.arm(BarrierArm::next("hold").contains("conv-0"));

        let held = tokio::spawn(complete(stub.completions_url(), body("conv-0")));
        timeout(BOUND, stub.until_waiting("hold", 1))
            .await
            .expect("conv-0 never parked");

        let others: Vec<_> = (1..N)
            .map(|i| tokio::spawn(complete(stub.completions_url(), body(&format!("conv-{i}")))))
            .collect();
        for o in others {
            let (text, done) = timeout(BOUND, o)
                .await
                .expect("a free conversation was stalled by the held one")
                .unwrap();
            assert!(done);
            assert_eq!(text.split_whitespace().count(), 12);
        }

        assert!(!held.is_finished(), "conv-0 must still be held");
        assert_eq!(stub.waiting("hold"), 1);
        assert_eq!(stub.served(), N as u64);

        stub.release("hold");
        let (_, done) = timeout(BOUND, held).await.expect("held never released").unwrap();
        assert!(done);
    }

    /// Release is latched: a scenario that releases before the request lands
    /// must not deadlock it.
    #[tokio::test]
    async fn release_before_arrival_does_not_hold() {
        let stub = InferenceStub::spawn_with(LOREM).await.unwrap();
        assert_eq!(stub.release("early"), 0);
        stub.arm(BarrierArm::next("early"));
        let (_, done) = timeout(BOUND, complete(stub.completions_url(), body("x")))
            .await
            .expect("latched barrier still held the response");
        assert!(done);
    }

    /// `count` holds exactly that many, then arms run out and requests flow.
    #[tokio::test]
    async fn arm_count_is_consumed() {
        let stub = InferenceStub::spawn_with(LOREM).await.unwrap();
        stub.arm(BarrierArm::next("two").count(2));

        let a = tokio::spawn(complete(stub.completions_url(), body("a")));
        let b = tokio::spawn(complete(stub.completions_url(), body("b")));
        timeout(BOUND, stub.until_waiting("two", 2)).await.expect("both park");

        let (_, done) = timeout(BOUND, complete(stub.completions_url(), body("c")))
            .await
            .expect("third request must not be held");
        assert!(done);
        assert_eq!(stub.state()["armed"], json!([]));

        stub.release("two");
        for h in [a, b] {
            assert!(timeout(BOUND, h).await.unwrap().unwrap().1);
        }
    }

    /// The same flow over the HTTP control plane a desktop e2e scenario uses.
    #[tokio::test]
    async fn control_plane_over_http() {
        let stub = InferenceStub::spawn_with(LOREM).await.unwrap();
        let http = reqwest::Client::new();
        let base = stub.base_url.clone();

        let armed: Value = http
            .post(format!("{base}/control/arm"))
            .json(&json!({"barrier": "b", "model": "stub-1b"}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(armed["armed"]["count"], 1);

        let req = tokio::spawn(complete(stub.completions_url(), body("http")));
        timeout(BOUND, stub.until_waiting("b", 1)).await.unwrap();

        let state: Value = http
            .get(format!("{base}/control/state"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(state["waiting"], json!({"b": 1}));
        assert_eq!(state["requests"][0]["model"], "stub-1b");
        assert_eq!(state["requests"][0]["turn_index"], 0);

        let released: Value = http
            .post(format!("{base}/control/release"))
            .json(&json!({"barrier": "b"}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(released["was_waiting"], 1);
        assert!(timeout(BOUND, req).await.unwrap().unwrap().1);
    }

    /// `delay_ms` paces frames. Lower bound only — an upper bound would be a
    /// timing race, which is exactly what delay_ms must not be used for.
    #[tokio::test]
    async fn delay_ms_paces_frames() {
        // 4 content frames + stop + usage + [DONE] = 7 frames, 6 gaps.
        let mode = StubMode::Lorem {
            tokens: 4,
            chunk_tokens: 1,
        };
        let stub = InferenceStub::builder(mode).delay_ms(25).spawn().await.unwrap();
        let start = Instant::now();
        let (text, done) = complete(stub.completions_url(), body("paced")).await;
        assert!(done);
        assert_eq!(text.split_whitespace().count(), 4);
        assert!(start.elapsed() >= Duration::from_millis(6 * 25));
    }
}

#[cfg(test)]
mod script_tests {
    use super::*;
    use tokio::time::timeout;

    const BOUND: Duration = Duration::from_secs(10);

    /// The W354 §2 leader shape, character-targeted: dispatch twice, then
    /// collect — `L3-both` before `L3-partial` is first-match-wins doing the
    /// branching.
    fn leader_script() -> Script {
        Script::from_value(json!({
            "script": "leader-dispatches-two",
            "roles": {
                "script:leader": [
                    {"id": "L1-open", "when": {"turn_index": 0},
                     "emit": [
                         {"text": "Two counts. Dispatching both."},
                         {"tool_use": {"name": "party_dispatch",
                                       "input": {"target": {"character": "Lorem0"}, "prompt": "count rs"}}},
                         {"tool_use": {"name": "party_dispatch",
                                       "input": {"target": {"character": "Lorem1"}, "prompt": "count ts"}}}],
                     "stop": "tool_use"},
                    {"id": "L2-collect", "when": {"after_tool_result": "party_dispatch", "turn_index": 1},
                     "emit": [{"tool_use": {"name": "inbox_read", "input": {}}}]},
                    {"id": "L3-both", "when": {"after_tool_result": "inbox_read",
                                               "tool_result_contains": ["rs-count", "ts-count"]},
                     "emit": [{"text": "Got both: $TOOL_RESULT"}], "stop": "end_turn"},
                    {"id": "L3-partial", "when": {"after_tool_result": "inbox_read"},
                     "emit": [{"text": "PARTIAL"}]}
                ],
                "lorem-1": [
                    {"id": "C1-any", "emit": [{"text": "child reply", "chunk_tokens": 1}]}
                ]
            }
        }))
        .expect("valid script")
    }

    async fn spawn(script: Script) -> InferenceStub {
        InferenceStub::spawn_with(StubMode::Script(script)).await.unwrap()
    }

    /// POST and return `(status, data frames as JSON)`; a non-200 body comes
    /// back as one JSON value.
    async fn post(stub: &InferenceStub, body: &Value) -> (u16, Vec<Value>) {
        let resp = reqwest::Client::new()
            .post(stub.completions_url())
            .json(body)
            .send()
            .await
            .expect("completions request");
        let status = resp.status().as_u16();
        let text = resp.text().await.expect("completions body");
        if status != 200 {
            return (status, vec![serde_json::from_str(&text).expect("error body is JSON")]);
        }
        let frames = text
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter(|p| *p != "[DONE]")
            .map(|p| serde_json::from_str(p).expect("frame is JSON"))
            .collect();
        (status, frames)
    }

    /// Re-assemble streamed tool calls into `(id, name, parsed arguments)`.
    fn tool_calls(frames: &[Value]) -> Vec<(String, String, Value)> {
        let mut acc: BTreeMap<u64, (String, String, String)> = BTreeMap::new();
        for tc in frames
            .iter()
            .filter_map(|f| f.pointer("/choices/0/delta/tool_calls/0"))
        {
            let e = acc.entry(tc["index"].as_u64().unwrap()).or_default();
            if let Some(id) = tc["id"].as_str() {
                e.0 = id.to_string();
            }
            if let Some(n) = tc.pointer("/function/name").and_then(Value::as_str) {
                e.1 = n.to_string();
            }
            if let Some(a) = tc.pointer("/function/arguments").and_then(Value::as_str) {
                e.2.push_str(a);
            }
        }
        acc.into_values()
            .map(|(id, n, a)| (id, n, serde_json::from_str(&a).unwrap()))
            .collect()
    }

    fn text_of(frames: &[Value]) -> String {
        frames
            .iter()
            .filter_map(|f| f.pointer("/choices/0/delta/content").and_then(Value::as_str))
            .collect()
    }

    fn finish(frames: &[Value]) -> Option<String> {
        frames
            .iter()
            .find_map(|f| f.pointer("/choices/0/finish_reason").and_then(Value::as_str))
            .map(str::to_string)
    }

    /// Drives one leader conversation through the whole script like a
    /// runner would: append the assistant turn, answer every tool call with
    /// `results(name)`, send again. Returns the final turn's text.
    struct Convo {
        body: Value,
    }

    impl Convo {
        fn new(prompt: &str) -> Self {
            Self {
                body: json!({"model": "script:leader", "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": prompt}]}),
            }
        }

        /// One request; returns its frames. The turn ended iff they carry no
        /// tool calls.
        async fn step(&mut self, stub: &InferenceStub, inbox: &str) -> Vec<Value> {
            let (status, frames) = post(stub, &self.body).await;
            assert_eq!(status, 200, "unexpected no-match: {frames:?}");
            let calls = tool_calls(&frames);
            let msgs = self.body["messages"].as_array_mut().unwrap();
            msgs.push(json!({"role": "assistant", "content": text_of(&frames),
                "tool_calls": calls.iter().map(|(id, n, a)| json!({"id": id, "type": "function",
                    "function": {"name": n, "arguments": a.to_string()}})).collect::<Vec<_>>()}));
            for (id, name, _) in &calls {
                let content = if name == "inbox_read" { inbox.to_string() } else { format!("{{\"receipt\":\"{id}\"}}") };
                msgs.push(json!({"role": "tool", "tool_call_id": id, "content": content}));
            }
            frames
        }

        /// Step until the turn ends; returns the final frames.
        async fn run(&mut self, stub: &InferenceStub, inbox: &str) -> Vec<Value> {
            loop {
                let frames = self.step(stub, inbox).await;
                if tool_calls(&frames).is_empty() {
                    return frames;
                }
            }
        }
    }

    /// Fired sequences keyed by conversation id — order-independent.
    fn fired(stub: &InferenceStub) -> BTreeMap<String, Vec<String>> {
        stub.state()["conversations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                let ids = c["fired"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string());
                (c["conversation"].as_str().unwrap().to_string(), ids.collect())
            })
            .collect()
    }

    /// A rule's emit lowers to the wire: text first, then two tool calls at
    /// indices 0 and 1 whose fragments re-join into the script's input, then
    /// `finish_reason: tool_calls`; the next turn branches on what came back.
    #[tokio::test]
    async fn emit_lowers_to_text_then_indexed_tool_calls() {
        let stub = spawn(leader_script()).await;
        let mut c = Convo::new("count things");
        let inbox = "rs-count=3 ts-count=4";
        let frames = c.step(&stub, inbox).await;
        assert_eq!(text_of(&frames), "Two counts. Dispatching both.");
        let calls = tool_calls(&frames);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, "call_L1-open_0");
        assert_eq!(calls[1].1, "party_dispatch");
        assert_eq!(calls[1].2, json!({"target": {"character": "Lorem1"}, "prompt": "count ts"}));
        assert_eq!(finish(&frames).as_deref(), Some("tool_calls"));

        // Both receipts back → L2 asks for the inbox; it holds both counts →
        // L3-both, and $TOOL_RESULT is the inbox result as the model saw it.
        let frames = c.step(&stub, inbox).await;
        assert_eq!(tool_calls(&frames)[0].1, "inbox_read");
        let last = c.step(&stub, inbox).await;
        assert_eq!(text_of(&last), "Got both: rs-count=3 ts-count=4");
        assert_eq!(finish(&last).as_deref(), Some("stop"));

        let reqs = stub.state()["requests"].clone();
        assert_eq!(reqs[1]["fired"], "L2-collect");
        assert_eq!(reqs[1]["turn_index"], 1);
        assert_eq!(reqs[1]["tool_result"], "{\"receipt\":\"call_L1-open_1\"}");
        assert_eq!(reqs[2]["fired"], "L3-both");
    }

    /// First match wins: with only one count back, `L3-both` fails its
    /// `tool_result_contains` and the general `L3-partial` answers; with both,
    /// `L3-both` shadows `L3-partial` even though both match.
    #[tokio::test]
    async fn first_match_wins() {
        let stub = spawn(leader_script()).await;
        for (prompt, inbox, want) in [
            ("a", "rs-count=3", "PARTIAL"),
            ("b", "rs-count=3 ts-count=4", "Got both: rs-count=3 ts-count=4"),
        ] {
            let frames = Convo::new(prompt).run(&stub, inbox).await;
            assert_eq!(text_of(&frames), want);
        }
        let seqs: Vec<Vec<String>> = fired(&stub).into_values().collect();
        assert!(seqs.contains(&vec!["L1-open".into(), "L2-collect".into(), "L3-partial".into()]));
        assert!(seqs.contains(&vec!["L1-open".into(), "L2-collect".into(), "L3-both".into()]));
    }

    /// No rule (unknown role, or a known role in an unscripted position) is a
    /// LOUD 500 carrying the conversation tail — never a fallthrough.
    #[tokio::test]
    async fn no_match_is_500_with_the_tail() {
        let stub = spawn(leader_script()).await;

        // Known role, unscripted position: a plain user follow-up at turn 1.
        let body = json!({"model": "script:leader", "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "an unscripted follow-up"}]});
        let (status, doc) = post(&stub, &body).await;
        assert_eq!(status, 500);
        let err = &doc[0]["error"];
        assert_eq!(err["type"], "script_no_match");
        assert_eq!(err["turn_index"], 1);
        assert_eq!(err["tail"][2]["content"], "an unscripted follow-up");
        assert_eq!(err["role_rules"], json!(["L1-open", "L2-collect", "L3-both", "L3-partial"]));

        // Unknown role.
        let (status, doc) = post(&stub, &json!({"model": "real-model", "messages": [
            {"role": "user", "content": "hi"}]}))
        .await;
        assert_eq!(status, 500);
        assert_eq!(doc[0]["error"]["role_rules"], Value::Null);

        let state = stub.state();
        assert_eq!(state["served"], 2);
        assert_eq!(state["requests"][0]["unmatched"], true);
        assert_eq!(state["requests"][0]["fired"], Value::Null);
        assert_eq!(state["conversations"], json!([]));
    }

    /// The keying rule's proof: two conversations interleaved request by
    /// request fire exactly the sequences they fire run serially — the stub
    /// holds no per-conversation counter to be confused.
    #[tokio::test]
    async fn interleaved_conversations_fire_the_same_sequences_as_serial() {
        let inbox = |p: &str| if p == "A" { "rs-count=1 ts-count=2" } else { "rs-count=1" };

        let serial = spawn(leader_script()).await;
        for p in ["A", "B"] {
            Convo::new(p).run(&serial, inbox(p)).await;
        }

        let interleaved = spawn(leader_script()).await;
        let (mut a, mut b) = (Convo::new("A"), Convo::new("B"));
        // B goes first, so arrival order differs from the serial run too.
        let (mut a_done, mut b_done) = (false, false);
        while !(a_done && b_done) {
            if !b_done {
                b_done = tool_calls(&b.step(&interleaved, inbox("B")).await).is_empty();
            }
            if !a_done {
                a_done = tool_calls(&a.step(&interleaved, inbox("A")).await).is_empty();
            }
        }

        let want = fired(&serial);
        assert_eq!(want.len(), 2);
        assert_eq!(fired(&interleaved), want);
        assert!(want.values().any(|s| s.last().unwrap() == "L3-both"));
        assert!(want.values().any(|s| s.last().unwrap() == "L3-partial"));
    }

    /// An `await` emit holds the response mid-stream until released; the
    /// text before it has already gone out. `chunk_tokens` frames the text.
    #[tokio::test]
    async fn await_emit_holds_until_released() {
        let script = Script::from_value(json!({"roles": {"r": [
            {"id": "R1", "emit": [
                {"text": "one two three", "chunk_tokens": 1},
                {"await": "barrier:go"},
                {"text": "after"}]}]}}))
        .unwrap();
        let stub = spawn(script).await;
        let body = json!({"model": "r", "messages": [{"role": "user", "content": "x"}]});
        let url = stub.completions_url();
        let req = tokio::spawn(async move {
            reqwest::Client::new().post(url).json(&body).send().await.unwrap().text().await.unwrap()
        });
        timeout(BOUND, stub.until_waiting("go", 1)).await.expect("never parked");
        assert!(!req.is_finished());
        stub.release("go");
        let text = timeout(BOUND, req).await.unwrap().unwrap();
        let frames: Vec<Value> = text
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter(|p| *p != "[DONE]")
            .map(|p| serde_json::from_str(p).unwrap())
            .collect();
        let contents: Vec<&str> = frames
            .iter()
            .filter_map(|f| f.pointer("/choices/0/delta/content").and_then(Value::as_str))
            .collect();
        assert_eq!(contents, ["one ", "two ", "three", "after"]);
    }

    /// Validation refuses what would make the oracle ambiguous or book a
    /// real provider.
    #[test]
    fn validate_refuses_bad_scripts() {
        let dup = json!({"roles": {"a": [{"id": "X"}], "b": [{"id": "X"}]}});
        assert!(Script::from_value(dup).unwrap_err().to_string().contains("appears twice"));

        let stop = json!({"roles": {"a": [{"id": "X", "emit": [{"text": "t"}], "stop": "tool_use"}]}});
        assert!(Script::from_value(stop).is_err());

        for target in [json!({"sigil": "griffin"}), json!(null)] {
            let s = json!({"roles": {"a": [{"id": "X", "emit": [{"tool_use": {
                "name": "party_dispatch", "input": {"target": target, "prompt": "p"}}}]}]}});
            let err = Script::from_value(s).unwrap_err().to_string();
            assert!(err.contains("target by CHARACTER"), "{err}");
        }

        let unknown_key = json!({"roles": {"a": [{"id": "X", "when": {"turn": 0}}]}});
        assert!(Script::from_value(unknown_key).is_err());
    }

    /// R923-T9: `$TOOL_RESULT:<pointer>` in a tool input resolves against the
    /// last tool result; a miss leaves the string as written.
    #[test]
    fn tool_input_templates_resolve_against_the_last_tool_result() {
        let input = json!({"target": {"session_id": "$TOOL_RESULT:/session_id"},
                           "keep": ["$TOOL_RESULT:/missing", 3]});
        let filled = fill_tool_result(&input, r#"{"session_id": "session:abc"}"#);
        assert_eq!(
            filled,
            json!({"target": {"session_id": "session:abc"}, "keep": ["$TOOL_RESULT:/missing", 3]})
        );
        assert_eq!(fill_tool_result(&input, "not json"), input);
    }
}


#[cfg(test)]
mod test_helpers {
    use super::*;
    use serde_json::Value;

    pub async fn call_completions(stub: &InferenceStub, target_url: &str) -> String {
        let body = json!({
            "model": "stub-1b",
            "stream": true,
            "messages": [
                {"role": "system", "content": "You are a URL summarizer."},
                {"role": "user", "content": format!("Summarize this URL: {target_url}")}
            ]
        });
        first_content_from_sse(&stub.completions_url(), body).await
    }

    pub async fn call_completions_raw(stub: &InferenceStub, user_content: &str) -> String {
        let body = json!({
            "model": "stub-1b",
            "stream": true,
            "messages": [{"role": "user", "content": user_content}]
        });
        first_content_from_sse(&stub.completions_url(), body).await
    }

    pub async fn call_models(stub: &InferenceStub) -> Value {
        reqwest::Client::new()
            .get(format!("{}/v1/models", stub.base_url))
            .send()
            .await
            .expect("models request")
            .json()
            .await
            .expect("models JSON")
    }

    /// POST the body and return the first non-empty `content` field from SSE.
    async fn first_content_from_sse(url: &str, body: Value) -> String {
        let text = reqwest::Client::new()
            .post(url)
            .json(&body)
            .send()
            .await
            .expect("completions request")
            .text()
            .await
            .expect("completions body");

        for line in text.lines() {
            let Some(payload) = line.strip_prefix("data: ") else {
                continue;
            };
            if payload == "[DONE]" {
                break;
            }
            if let Ok(val) = serde_json::from_str::<Value>(payload) {
                if let Some(c) = val["choices"][0]["delta"]["content"].as_str() {
                    if !c.is_empty() {
                        return c.to_string();
                    }
                }
            }
        }
        String::new()
    }
}
