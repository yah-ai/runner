//! Spawning and stopping a `claude -p` stream-json subprocess.
//!
//! Part of R951 — copied from `app/yah/desktop/src/agent_process.rs`
//! (`claude_fixed_args`, `claude_spawn_envs`, `spawn_claude`, the
//! process-group SIGKILL of its stop path, `STOP_GRACE`, `one_m_variant`).
//! The camp-specific inputs it used to reach for (settings path, approval
//! socket, `YAH_*` envs, conductor tool gating) arrive through [`PvHost`].

use crate::host::PvHost;
use crate::transcript::claude_session_uuid;
use std::ffi::OsString;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::{Child, ChildStdin, Command};
use yah_runner_core::model_catalog::model_context_tokens;
use yah_runner_core::{ReasoningBudget, SessionId};

/// Grace window between stdin close and SIGKILL on stop. Claude Code
/// flushes its result on EOF; if it's still alive after this it's
/// almost certainly stuck and a kill is warranted.
pub const STOP_GRACE: Duration = Duration::from_millis(1500);

/// claude-cli's override for the context window it assumes for a model it
/// does not recognise (its own unknown-model notice names it).
pub const CLAUDE_MAX_CONTEXT_TOKENS_ENV: &str = "CLAUDE_CODE_MAX_CONTEXT_TOKENS";

/// The Claude Code `--model` string for the 1M-context variant of `model`,
/// or `None` when `model` has no distinct 1M tier to upgrade to.
///
/// Only Sonnet 4.6 exposes a separate 1M selection under subscription today;
/// Opus 4.x already runs at 1M, and older models have no 1M tier. The 1M
/// selection is addressed by the `[1m]` model suffix — the one line to change
/// if a future Claude Code build expects a different form.
pub fn one_m_variant(model: &str) -> Option<String> {
    if model.starts_with("claude-sonnet-4-6") && !model.contains("[1m]") {
        Some(format!("{model}[1m]"))
    } else {
        None
    }
}

/// `--effort` level for a [`ReasoningBudget`].
pub fn claude_effort(budget: ReasoningBudget) -> &'static str {
    match budget {
        ReasoningBudget::Fast => "low",
        ReasoningBudget::Standard => "medium",
        ReasoningBudget::Deep => "high",
        ReasoningBudget::Budget { tokens } => match tokens {
            0..=2048 => "low",
            2049..=8000 => "medium",
            8001..=20_000 => "high",
            _ => "xhigh",
        },
    }
}

/// Everything about one spawn that is not host policy.
#[derive(Debug, Clone)]
pub struct SpawnSpec<'a> {
    /// `claude`, or a claude-cli-compatible fork.
    pub cli_binary: &'a str,
    /// The subprocess cwd — also what claude keys its transcript dir off.
    pub workdir: &'a Path,
    pub model: &'a str,
    pub session_id: &'a SessionId,
    /// `--resume <uuid>` instead of `--session-id <uuid>`.
    pub resume: Option<&'a str>,
    pub reasoning: Option<ReasoningBudget>,
    pub append_system_prompt: Option<&'a str>,
    /// `ANTHROPIC_BASE_URL` — a non-Anthropic upstream speaking its API.
    pub base_url: Option<&'a str>,
    /// `ANTHROPIC_AUTH_TOKEN` for `base_url`.
    pub auth_token: Option<&'a str>,
    /// Pins every claude tier alias (sonnet/haiku/opus) to one upstream model.
    pub tier_model: Option<&'a str>,
    /// Force claude-cli's assumed context window. When `None`, a non-`claude-`
    /// model falls back to the core model catalog's window.
    pub num_ctx: Option<u32>,
    /// `ENABLE_PROMPT_CACHING_1H=1` (a long-lived session's prelude cache).
    pub extended_cache: bool,
    /// `--chrome`: attach claude-cli's Claude-in-Chrome integration (the
    /// `mcp__claude-in-chrome__*` tools). claude-cli proper only. A fork that
    /// lacks the flag would fail the spawn, so the host refuses that pairing
    /// before it gets here. It survives `--strict-mcp-config`: the tools come
    /// from the flag, not from the MCP config.
    pub chrome: bool,
}

impl<'a> SpawnSpec<'a> {
    pub fn new(cli_binary: &'a str, workdir: &'a Path, model: &'a str, session_id: &'a SessionId) -> Self {
        Self {
            cli_binary,
            workdir,
            model,
            session_id,
            resume: None,
            reasoning: None,
            append_system_prompt: None,
            base_url: None,
            auth_token: None,
            tier_model: None,
            num_ctx: None,
            extended_cache: false,
            chrome: false,
        }
    }
}

/// The argv after the binary name.
pub fn claude_fixed_args(spec: &SpawnSpec<'_>, host: &dyn PvHost) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec![
        "--print".into(),
        "--input-format".into(),
        "stream-json".into(),
        "--output-format".into(),
        "stream-json".into(),
        "--include-partial-messages".into(),
        "--verbose".into(),
        "--model".into(),
        spec.model.into(),
    ];
    for dir in host.plugin_dirs(spec.session_id) {
        args.push("--plugin-dir".into());
        args.push(dir.into_os_string());
    }
    if let Some(mcp) = host.mcp_config_path(spec.session_id) {
        args.push("--mcp-config".into());
        args.push(mcp.into_os_string());
        args.push("--strict-mcp-config".into());
    }
    if let Some(settings) = host.settings_path(spec.session_id) {
        args.push("--settings".into());
        args.push(settings.into_os_string());
    }
    // claude-cli rejects `--session-id X --resume Y` ("--session-id can only be
    // used with --continue or --resume if --fork-session is also specified"),
    // so the two flags are mutually exclusive. On resume the uuid is the one
    // the original start pinned, since claude_session_uuid is deterministic.
    match spec.resume {
        Some(uuid) => {
            args.push("--resume".into());
            args.push(uuid.into());
        }
        None => {
            args.push("--session-id".into());
            args.push(claude_session_uuid(spec.session_id).into());
        }
    }
    if let Some(budget) = spec.reasoning {
        args.push("--effort".into());
        args.push(claude_effort(budget).into());
    }
    if let Some(prompt) = spec.append_system_prompt {
        if !prompt.is_empty() {
            args.push("--append-system-prompt".into());
            args.push(prompt.into());
        }
    }
    if spec.chrome {
        args.push("--chrome".into());
    }
    // claude-cli accepts space- or comma-separated lists; one comma-joined
    // value keeps the argv compact.
    let disallowed = host.disallowed_tools(spec.session_id);
    if !disallowed.is_empty() {
        args.push("--disallowedTools".into());
        args.push(disallowed.join(",").into());
    }
    args
}

/// The subprocess environment: pv's own upstream / context-window envs,
/// then [`PvHost::extra_env`].
pub fn claude_spawn_envs(spec: &SpawnSpec<'_>, host: &dyn PvHost) -> Vec<(OsString, OsString)> {
    let mut envs: Vec<(OsString, OsString)> = Vec::new();
    if let Some(url) = spec.base_url {
        envs.push(("ANTHROPIC_BASE_URL".into(), url.into()));
    }
    if let Some(token) = spec.auth_token {
        envs.push(("ANTHROPIC_AUTH_TOKEN".into(), token.into()));
    }
    if let Some(m) = spec.tier_model {
        envs.push(("ANTHROPIC_DEFAULT_SONNET_MODEL".into(), m.into()));
        envs.push(("ANTHROPIC_DEFAULT_HAIKU_MODEL".into(), m.into()));
        envs.push(("ANTHROPIC_DEFAULT_OPUS_MODEL".into(), m.into()));
    }
    if spec.extended_cache {
        envs.push(("ENABLE_PROMPT_CACHING_1H".into(), "1".into()));
    }
    // Only a caller that wrote a window down gets one forced; claude-* models
    // otherwise keep claude-cli's own per-model table, which is authoritative
    // for models it knows. A non-claude model the CLI cannot resolve gets the
    // catalog's window instead of the CLI's flat 200K fallback.
    if let Some(tokens) = spec.num_ctx.filter(|n| *n > 0) {
        envs.push((CLAUDE_MAX_CONTEXT_TOKENS_ENV.into(), tokens.to_string().into()));
    } else if !spec.model.starts_with("claude-") {
        if let Some(tokens) = model_context_tokens(spec.model).filter(|n| *n > 0) {
            envs.push((CLAUDE_MAX_CONTEXT_TOKENS_ENV.into(), tokens.to_string().into()));
        }
    }
    envs.extend(host.extra_env(spec.session_id));
    envs
}

/// Spawn the subprocess with piped stdio, as the leader of its own process
/// group (unix), and report the pgid via [`PvHost::on_spawned`].
pub fn spawn_claude(spec: &SpawnSpec<'_>, host: &dyn PvHost) -> std::io::Result<Child> {
    let mut cmd = Command::new(spec.cli_binary);
    cmd.current_dir(spec.workdir);
    cmd.args(claude_fixed_args(spec, host));
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    // Own process group, so a stop can take claude's tool children (MCP
    // servers, Bash subshells) down with it.
    #[cfg(unix)]
    cmd.process_group(0);
    for (k, v) in claude_spawn_envs(spec, host) {
        cmd.env(k, v);
    }
    host.prepare_command(cmd.as_std_mut())?;
    let child = cmd.spawn()?;
    if let Some(pgid) = child.id() {
        host.on_spawned(spec.session_id, pgid);
    }
    Ok(child)
}

/// SIGKILL the child's whole process group (unix), then the child itself.
pub async fn kill_process_group(child: &mut Child) {
    #[cfg(unix)]
    if let Some(pgid) = child.id() {
        // SAFETY: kill(2) with a negative pid signals the process group;
        // no memory is touched.
        unsafe { libc::kill(-(pgid as libc::pid_t), libc::SIGKILL) };
    }
    let _ = child.start_kill();
    let _ = child.wait().await;
}

/// Cooperative stop: drop stdin (claude flushes and exits on EOF), wait
/// [`STOP_GRACE`], then kill the process group. Reports
/// [`PvHost::on_reaped`].
pub async fn stop_child(
    session_id: &SessionId,
    mut child: Child,
    stdin: Option<ChildStdin>,
    host: &dyn PvHost,
) {
    drop(stdin);
    if tokio::time::timeout(STOP_GRACE, child.wait()).await.is_err() {
        kill_process_group(&mut child).await;
    }
    host.on_reaped(session_id);
}

#[cfg(test)]
mod tests {
    //! Part of R951 — ported from the desktop `agent_process.rs` argv/env
    //! tests (R951-F3 deletes the originals). The camp-only ones (`YAH_*`
    //! envs, build env, job-driven 1h cache, conductor gating) stay with the
    //! desktop's `PvHost` impl, which now owns those inputs.
    use super::*;
    use crate::host::BareHost;
    use std::path::PathBuf;

    fn sid() -> SessionId {
        SessionId::new("session:spawntest")
    }

    fn argv(spec: &SpawnSpec<'_>, host: &dyn PvHost) -> Vec<String> {
        claude_fixed_args(spec, host)
            .into_iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    fn env_of(envs: &[(OsString, OsString)], key: &str) -> Option<String> {
        envs.iter()
            .rev()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.to_string_lossy().into_owned())
    }

    struct FullHost;

    impl PvHost for FullHost {
        fn mcp_config_path(&self, _: &SessionId) -> Option<PathBuf> {
            Some(PathBuf::from("/camp/.claude/settings.json"))
        }
        fn settings_path(&self, _: &SessionId) -> Option<PathBuf> {
            Some(PathBuf::from("/camp/.claude/settings.json"))
        }
        fn plugin_dirs(&self, _: &SessionId) -> Vec<PathBuf> {
            vec![PathBuf::from("/camp/mods/yah-gate"), PathBuf::from("/camp/mods/other")]
        }
        fn extra_env(&self, _: &SessionId) -> Vec<(OsString, OsString)> {
            vec![("HOST_VAR".into(), "1".into()), ("ANTHROPIC_BASE_URL".into(), "host-wins".into())]
        }
        fn disallowed_tools(&self, _: &SessionId) -> Vec<String> {
            vec!["Agent".into(), "Workflow".into()]
        }
    }

    #[test]
    fn one_m_variant_only_upgrades_sonnet_4_6() {
        assert_eq!(one_m_variant("claude-sonnet-4-6"), Some("claude-sonnet-4-6[1m]".to_string()));
        assert_eq!(
            one_m_variant("claude-sonnet-4-6-20260101"),
            Some("claude-sonnet-4-6-20260101[1m]".to_string())
        );
        assert_eq!(one_m_variant("claude-opus-4-7"), None);
        assert_eq!(one_m_variant("claude-sonnet-4-5"), None);
        assert_eq!(one_m_variant("claude-sonnet-4-6[1m]"), None);
    }

    #[test]
    fn bare_host_argv_is_stream_json_with_no_host_flags() {
        let sid = sid();
        let a = argv(&SpawnSpec::new("claude", Path::new("/w"), "claude-sonnet-4-6", &sid), &BareHost);
        for flag in ["--print", "--include-partial-messages", "--verbose"] {
            assert!(a.iter().any(|x| x == flag), "{flag} missing: {a:?}");
        }
        assert_eq!(a.windows(2).filter(|w| w == &["--input-format", "stream-json"]).count(), 1);
        assert_eq!(a.windows(2).filter(|w| w == &["--output-format", "stream-json"]).count(), 1);
        for absent in [
            "--mcp-config",
            "--strict-mcp-config",
            "--settings",
            "--plugin-dir",
            "--disallowedTools",
            "--append-system-prompt",
            "--effort",
            "--resume",
            "--chrome",
        ] {
            assert!(!a.iter().any(|x| x == absent), "{absent} must be absent: {a:?}");
        }
    }

    #[test]
    fn chrome_flag_is_emitted_iff_the_spec_asks() {
        let sid = sid();
        let mut spec = SpawnSpec::new("claude", Path::new("/w"), "claude-sonnet-4-6", &sid);
        assert!(!argv(&spec, &FullHost).iter().any(|x| x == "--chrome"));
        spec.chrome = true;
        let a = argv(&spec, &FullHost);
        assert_eq!(a.iter().filter(|x| *x == "--chrome").count(), 1, "{a:?}");
    }

    #[test]
    fn host_flags_come_from_pv_host() {
        let sid = sid();
        let a = argv(&SpawnSpec::new("claude", Path::new("/w"), "claude-sonnet-4-6", &sid), &FullHost);
        let after = |flag: &str| {
            let i = a.iter().position(|x| x == flag).unwrap_or_else(|| panic!("{flag} missing: {a:?}"));
            a[i + 1].clone()
        };
        assert_eq!(after("--mcp-config"), "/camp/.claude/settings.json");
        assert!(a.iter().any(|x| x == "--strict-mcp-config"));
        assert_eq!(after("--settings"), "/camp/.claude/settings.json");
        let plugin_dirs: Vec<_> = a.windows(2).filter(|w| w[0] == "--plugin-dir").map(|w| w[1].clone()).collect();
        assert_eq!(plugin_dirs, ["/camp/mods/yah-gate", "/camp/mods/other"]);
        assert_eq!(after("--disallowedTools"), "Agent,Workflow");
    }

    #[test]
    fn argv_includes_deterministic_session_id_or_resume_never_both() {
        let sid = sid();
        let mut spec = SpawnSpec::new("claude", Path::new("/w"), "m", &sid);
        let a = argv(&spec, &BareHost);
        let i = a.iter().position(|x| x == "--session-id").expect("--session-id");
        assert_eq!(a[i + 1], claude_session_uuid(&sid));
        assert!(!a.iter().any(|x| x == "--resume"));

        spec.resume = Some("11111111-2222-4333-8444-555555555555");
        let a = argv(&spec, &BareHost);
        let i = a.iter().position(|x| x == "--resume").expect("--resume");
        assert_eq!(a[i + 1], "11111111-2222-4333-8444-555555555555");
        assert!(!a.iter().any(|x| x == "--session-id"));
    }

    #[test]
    fn argv_includes_effort_for_reasoning_budgets() {
        let sid = sid();
        for (budget, expected) in [
            (ReasoningBudget::Fast, "low"),
            (ReasoningBudget::Standard, "medium"),
            (ReasoningBudget::Deep, "high"),
            (ReasoningBudget::Budget { tokens: 64_000 }, "xhigh"),
        ] {
            let mut spec = SpawnSpec::new("claude", Path::new("/w"), "m", &sid);
            spec.reasoning = Some(budget);
            let a = argv(&spec, &BareHost);
            let i = a.iter().position(|x| x == "--effort").expect("--effort");
            assert_eq!(a[i + 1], expected, "{budget:?}");
        }
    }

    #[test]
    fn argv_includes_append_system_prompt_only_when_non_empty() {
        let sid = sid();
        let mut spec = SpawnSpec::new("claude", Path::new("/w"), "m", &sid);
        spec.append_system_prompt = Some("be brief");
        let a = argv(&spec, &BareHost);
        let i = a.iter().position(|x| x == "--append-system-prompt").expect("flag");
        assert_eq!(a[i + 1], "be brief");
        spec.append_system_prompt = Some("");
        assert!(!argv(&spec, &BareHost).iter().any(|x| x == "--append-system-prompt"));
    }

    #[test]
    fn envs_force_a_declared_context_window() {
        let sid = sid();
        let mut spec = SpawnSpec::new("claude", Path::new("/w"), "kimi-k2.7-code", &sid);
        spec.num_ctx = Some(262_144);
        let envs = claude_spawn_envs(&spec, &BareHost);
        assert_eq!(env_of(&envs, CLAUDE_MAX_CONTEXT_TOKENS_ENV).as_deref(), Some("262144"));
    }

    #[test]
    fn envs_leave_claude_models_on_the_clis_own_window() {
        let sid = sid();
        let spec = SpawnSpec::new("claude", Path::new("/w"), "claude-sonnet-4-6", &sid);
        let envs = claude_spawn_envs(&spec, &BareHost);
        assert_eq!(env_of(&envs, CLAUDE_MAX_CONTEXT_TOKENS_ENV), None);
    }

    #[test]
    fn envs_fall_back_to_the_catalog_for_non_claude_models() {
        let sid = sid();
        let model = "deepseek-v4-pro";
        let spec = SpawnSpec::new("claude", Path::new("/w"), model, &sid);
        let envs = claude_spawn_envs(&spec, &BareHost);
        assert_eq!(
            env_of(&envs, CLAUDE_MAX_CONTEXT_TOKENS_ENV),
            model_context_tokens(model).filter(|n| *n > 0).map(|n| n.to_string()),
        );
    }

    #[test]
    fn envs_set_tier_models_and_upstream_only_when_given() {
        let sid = sid();
        let mut spec = SpawnSpec::new("claude", Path::new("/w"), "m", &sid);
        let envs = claude_spawn_envs(&spec, &BareHost);
        for k in ["ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ENABLE_PROMPT_CACHING_1H"] {
            assert_eq!(env_of(&envs, k), None, "{k}");
        }
        spec.base_url = Some("https://up.example");
        spec.auth_token = Some("tok");
        spec.tier_model = Some("glm-5");
        spec.extended_cache = true;
        let envs = claude_spawn_envs(&spec, &BareHost);
        assert_eq!(env_of(&envs, "ANTHROPIC_BASE_URL").as_deref(), Some("https://up.example"));
        assert_eq!(env_of(&envs, "ANTHROPIC_AUTH_TOKEN").as_deref(), Some("tok"));
        for k in ["ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL"] {
            assert_eq!(env_of(&envs, k).as_deref(), Some("glm-5"), "{k}");
        }
        assert_eq!(env_of(&envs, "ENABLE_PROMPT_CACHING_1H").as_deref(), Some("1"));
    }

    #[test]
    fn host_extra_env_is_applied_last_and_wins() {
        let sid = sid();
        let mut spec = SpawnSpec::new("claude", Path::new("/w"), "m", &sid);
        spec.base_url = Some("spec");
        let envs = claude_spawn_envs(&spec, &FullHost);
        assert_eq!(env_of(&envs, "HOST_VAR").as_deref(), Some("1"));
        // Last occurrence wins in `Command::env`; env_of reads the last one.
        assert_eq!(env_of(&envs, "ANTHROPIC_BASE_URL").as_deref(), Some("host-wins"));
    }

    /// Spawns `cat` (a stand-in that exits on stdin EOF) in its own process
    /// group and stops it cooperatively, checking both host notifications.
    #[cfg(unix)]
    #[tokio::test]
    async fn stop_child_reaps_a_cooperative_child_and_notifies_the_host() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        #[derive(Default)]
        struct Counting {
            spawned: AtomicU32,
            reaped: AtomicU32,
        }
        impl PvHost for Arc<Counting> {
            fn on_spawned(&self, _: &SessionId, pgid: u32) {
                assert!(pgid > 0);
                self.spawned.fetch_add(1, Ordering::SeqCst);
            }
            fn on_reaped(&self, _: &SessionId) {
                self.reaped.fetch_add(1, Ordering::SeqCst);
            }
        }

        let host = Arc::new(Counting::default());
        let sid = sid();
        let mut cmd = Command::new("cat");
        cmd.stdin(Stdio::piped()).stdout(Stdio::null()).kill_on_drop(true);
        cmd.process_group(0);
        let mut child = cmd.spawn().expect("spawn cat");
        host.on_spawned(&sid, child.id().unwrap());
        let stdin = child.stdin.take();
        let started = std::time::Instant::now();
        stop_child(&sid, child, stdin, &host).await;
        assert!(started.elapsed() < STOP_GRACE, "cat exits on EOF, no SIGKILL needed");
        assert_eq!(host.spawned.load(Ordering::SeqCst), 1);
        assert_eq!(host.reaped.load(Ordering::SeqCst), 1);
    }
}
