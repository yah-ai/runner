//! The host seam for the claude `-p` cell.
//!
//! Part of R951. [`PvHost`] carries only what a host must *inject* into a
//! spawn (config paths, env, tool gating) and the two process-lifecycle
//! notifications. Persisting, emitting, phase tracking and turn permits are
//! deliberately NOT hooks: the reader yields [`crate::PvEvent`]s and the host
//! taps that stream.
//!
//! @yah:ticket(R950-F14, "Replace --permission-prompt-tool mcp__yah__approve_tool with a tool.check mod shipped by yah-runner-pv")
//! @yah:status(review)
//! @yah:at(2026-10-05T21:03:17Z)
//! @yah:assignee(agent:bundle-anthropic-ashguard)
//! @yah:parent(R950)
//! @yah:next("Spike PASSED 2026-10-05 on claude 2.1.290 (session:a1e1cc8c; throwaway at /tmp/yah-mod-spike). A plugin loaded with `claude -p --plugin-dir` registers an on('tool.check') hook. That hook sees every tool call, including core-allowed ones, subagent calls (with agentId) and MCP tools. It POSTs over a Unix socket via $.http.fetch({ socketPath }) and returns {decision, reason}. A deny reaches the model as 'Permission to use X denied by plugin <name>: <reason>'. Stream-json emits system/permission_denied (decision_reason_type: hook) and system/ui_log frames.")
//! @yah:next("Operator call 2026-10-05: semantics B. The hook awaits next(e) (core's verdict). An allow or deny from core stands; only decision === 'ask' goes to yah's gate. Today's behaviour is preserved, without the MCP hop.")
//! @yah:next("$.http.fetch hard-aborts at 30000ms, and HttpInit has no timeoutMs. The hook's 10s budget does NOT count $ calls in flight. So the host endpoint must long-poll: hold each request under 30s, answer {decision:'pending'}, and the mod re-POSTs with the same tool_use_id until decided. Verified: a 240s approval took 10 polls and then allowed. Check next.signal.aborted between polls.")
//! @yah:next("Neither existing socket speaks HTTP: YAH_MCP_APPROVAL_SOCK (ApprovalProtocolFrame, approval.rs) and the camp daemon socket are both NDJSON. Add a per-session HTTP-over-UDS listener on the pv host side. Desktop implements it in-process by calling the ApproveTool logic (consult_gate, then form mint and wait). Socket path must stay under ~100B, in a 0700 dir.")
//! @yah:next("PvHost: replace permission_prompt_tool() with plugin_dirs(). Embed the mod (plugin.json, hooks.json, register.ts) in yah-runner-pv via include_str!. Write it to a per-camp dir at spawn, pass --plugin-dir, and export the socket path in extra_env (the mod reads it with a literal $.env.get).")
//! @yah:next("Delete, don't keep beside it: the approve_tool MCP tool (tools.rs + form_tools.rs ApproveTool schema), the --permission-prompt-tool argv and its tests, and the R318-B3 stash_and_return_allow / restore_pending_payload path. tool.check returns a verdict only and never echoes updatedInput, so the oversized-input drop that path worked around cannot occur. Keep the R206-F2 AskUserQuestion hint-deny and the R186-B4 schema-validation deny as server-side deny reasons.")
//! @yah:verify("claude plugin validate <embedded mod dir> passes, and claude plugin test covers allow/deny/pending-loop")
//! @yah:verify("Fixture: a -p run with --permission-prompts none and a Bash call core says 'ask' for resolves through the yah endpoint (not auto-denied)")
//! @yah:verify("grep -c 'permission-prompt-tool\\|approve_tool' over app/ crates/ oss/ code (comment-stripped) -> 0")
//! @yah:gotcha("Fail closed: wrap the hook in .catch(...) returning {decision:'deny'}. Without a catch, a thrown or overrun hook is skipped and core's 'ask' falls to the headless decider. A .catch has only 1000ms of grace.")
//! @yah:gotcha("The hot-reload consent prompt is off under -p; that is not a blocker. A -p run loads --plugin-dir mods fresh each spawn. Long-lived stream-json sessions only hot-reload with CLAUDE_CODE_PLUGIN_DIR_WATCH=1, which we don't need.")
//! @yah:depends_on(R951-F3)
//! @yah:files(oss/runner/crates/pv/src/host.rs)
//! @yah:files(oss/runner/crates/pv/src/spawn.rs)
//! @yah:files(app/yah/desktop/src/agent_process.rs)
//! @yah:files(crates/yah/agent-tools/src/form_tools.rs)
//! @yah:files(app/yah/cli/src/mcp/tools.rs)
//! @yah:handoff("Landed (unverified build pending): oss/runner/crates/pv/mod/yah-gate (plugin.json, hooks.json, register.ts; register.test.ts not embedded) — claude plugin validate passes, claude plugin test 7/7 (allow-stands, ask->endpoint, deny reason, pending re-poll same id, no-socket/500/throw fail closed). pv src/gate.rs: install_gate_mod (include_str!), bind_gate_socket (0700 parent), serve_gate HTTP/1.1-over-UDS long-poll (DEFAULT_HOLD 20s, one decide per tool_use_id), GateDecider trait; PvHost::permission_prompt_tool -> plugin_dirs; cargo test -p yah-runner-pv 92 pass. agent-tools: ApproveTool + stash_and_return_allow deleted -> pub async gate_tool_check(ToolCheckCaller, ..) -> ToolCheckVerdict; TriageSource::ApproveTool renamed ToolCheck; stamp_character_metadata takes (camp_root, subclass_id). cli mcp/tools.rs: approve_tool schema+dispatch, restore_pending_payload and its 6 tests deleted; Meta set is now 3. desktop agent_process.rs: GateEndpoint (Drop aborts+unlinks) at $TMPDIR/yah-gate/<id>.sock, DesktopGateDecider, CampPvHost.gate_sock + plugin_dirs (.yah/tmp/mods/yah-gate) + YAH_GATE_SOCK env; probe driver flag dropped; prelude.rs + .yah/CLAUDE.md quirk line updated.")
//! @yah:handoff("Shipped: yah-gate claude mod (oss/runner/crates/pv/mod/yah-gate) embedded in yah-runner-pv (gate.rs: install_gate_mod, bind_gate_socket 0700, serve_gate HTTP-over-UDS long-poll, GateDecider). Semantics B: the mod awaits next(e), core allow/deny stands, only 'ask' POSTs to $YAH_GATE_SOCK; pending re-polls with the same tool_use_id; .catch fails closed. PvHost::permission_prompt_tool replaced by plugin_dirs. Desktop: GateEndpoint per claude session at $TMPDIR/yah-gate/<id>.sock (Drop aborts and unlinks), DesktopGateDecider -> agent_tools::form_tools::gate_tool_check (the former ApproveTool body: R206-F2 hint-deny, R186-B4 schema deny, consult_gate, form + 10m cap, R459-F12 hash gate). Deleted: approve_tool MCP tool (schema, dispatch, form_tools registration), stash_and_return_allow, restore_pending_payload + 6 cli tests, --permission-prompt-tool argv + tests, the probe driver's flag. Renamed TriageSource::ApproveTool -> ToolCheck. Extra: prelude.rs host-quirk line + .yah/CLAUDE.md updated; form tool-count tests 5 -> 4; MCP Meta set is now 3 (ask_user, session_context, yah_help).")
//! @yah:verify("claude plugin validate oss/runner/crates/pv/mod/yah-gate: passed. claude plugin test: 7/7 (allow stands, ask -> endpoint, deny reason, pending loop, no-socket / 500 / throw fail closed)\ncargo run -p yah-runner-pv --example gate_fixture (real claude -p 2.1.290, haiku): the Bash ask reached the endpoint, was allowed, and ran -> PASS\ncargo test: yah-runner-pv 92 ok; yah-agent-tools form_tools+triage 96 ok; desktop spawn/argv/env/approval 39 ok; yah-party prelude 148 ok; yah mcp::tools 158 ok. cargo check --tests for the 4 crates: 0 errors\nComment-stripped grep for 'permission-prompt-tool|approve_tool' over app/crates/oss .rs/.ts: 0 code hits")
//! @yah:assumes("The desktop path was not exercised end to end in a running yah.app; the fixture proves the pv endpoint and mod against real claude, and the unit tests cover the desktop wiring")
//! @yah:cleanup("Comments in other files (gate_watch.rs, rpc, agent.rs, event_bridge.rs, approval.rs docs) still name approve_tool historically; they're prose only, not code")
//! @yah:handoff("Operator live-verified 2026-10-05 in yah.app after restart (session:a1e1cc8c). The respawned sessions carry --plugin-dir .yah/tmp/mods/yah-gate, and none still carries --permission-prompt-tool. With yolo off, a non-allowlisted `cp` raised a Continue/Revise card. It was approved from mobile and the command ran. An allowlisted `touch` (settings.local.json Bash(touch:*)) went through without a card, which is correct per semantics B.")
//! @yah:gotcha("Answer-queue live push lag, observed during verification: after one approval was answered, the next card (`cp`) did not appear on mobile until a manual refresh. The bug is in the queue's push transport, not in yah-gate: the card was minted and answerable once it appeared. Handed to session:4d3fed76, which is on the realtime-push work.")
//! @yah:cleanup("From R955 closing gate (2026-10-06): removing approve_tool (deliberate, under this ticket) left `restore_pending_payload` in app/yah/cli/src/mcp/tools.rs (~:6466) with no caller. Delete it, or wire it into the yah-gate tool.check path if it is still needed. Forensics are on R955-B17's handoff.")

use std::ffi::OsString;
use std::path::PathBuf;
use yah_runner_core::SessionId;

/// Host-supplied spawn inputs and lifecycle callbacks. Every method has an
/// empty default, so `struct Bare; impl PvHost for Bare {}` spawns a plain
/// `claude -p` with no MCP servers, no hooks and no plugins.
pub trait PvHost: Send + Sync {
    /// MCP config file passed as `--mcp-config <path> --strict-mcp-config`.
    /// This is how tools reach claude: it spawns the stdio servers the file
    /// names (e.g. a binary built on `yah_runner_mcp::McpServer`).
    fn mcp_config_path(&self, _session_id: &SessionId) -> Option<PathBuf> {
        None
    }

    /// Settings file passed as `--settings <path>` (claude-cli reads its
    /// `hooks` block). `None` omits the flag — required for claude-cli forks
    /// that do not recognise it.
    fn settings_path(&self, _session_id: &SessionId) -> Option<PathBuf> {
        None
    }

    /// Plugin directories passed as `--plugin-dir`, one flag each. The
    /// approval gate rides here: install [`crate::gate::install_gate_mod`]
    /// and export [`crate::gate::GATE_SOCK_ENV`] from [`Self::extra_env`].
    fn plugin_dirs(&self, _session_id: &SessionId) -> Vec<PathBuf> {
        Vec::new()
    }

    /// Extra environment for the subprocess (approval sockets, host ids,
    /// build env). Applied after pv's own `ANTHROPIC_*` /
    /// context-window envs, so a host value wins on a collision.
    fn extra_env(&self, _session_id: &SessionId) -> Vec<(OsString, OsString)> {
        Vec::new()
    }

    /// Built-in claude tools to deny via `--disallowedTools`.
    fn disallowed_tools(&self, _session_id: &SessionId) -> Vec<String> {
        Vec::new()
    }

    /// Last word on the command before it spawns — called after pv has set
    /// args, stdio, process group and env. A host that registers a `pre_exec`
    /// hook that replaces the exec (yah's macOS TCC disclaim, R940-B1) does it
    /// here, so it lands after every other hook.
    fn prepare_command(&self, _cmd: &mut std::process::Command) -> std::io::Result<()> {
        Ok(())
    }

    /// The subprocess was spawned as the leader of process group `pgid`.
    fn on_spawned(&self, _session_id: &SessionId, _pgid: u32) {}

    /// The subprocess (and its process group) has been reaped.
    fn on_reaped(&self, _session_id: &SessionId) {}
}

/// A host that injects nothing.
#[derive(Debug, Default, Clone, Copy)]
pub struct BareHost;

impl PvHost for BareHost {}
