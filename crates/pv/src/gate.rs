//! The `yah-gate` claude mod and the host endpoint it asks.
//!
//! pv ships a claude plugin (`mod/yah-gate`, embedded here) whose
//! `tool.check` hook lets core decide first and puts core's `ask` AND
//! `allow` verdicts to the host, tagged with [`CoreDecision`]; only a core
//! deny stands on its own. It replaces `--permission-prompt-tool`: no MCP
//! hop, no `updatedInput` echo, and it sees subagent calls too. Forwarding
//! the allows is what lets the host apply the camp boundary (R824-B10), so
//! the boundary has one owner here instead of a PreToolUse hook `ask` beside
//! it.
//!
//! The mod reaches the host over HTTP on a Unix socket named by
//! [`GATE_SOCK_ENV`]. `$.http.fetch` hard-aborts at 30s and a person can take
//! minutes, so the endpoint long-polls: [`serve_gate`] holds each request up
//! to `hold`, answers `{"decision":"pending"}` if the [`GateDecider`] has not
//! finished, and the mod re-POSTs with the same `tool_use_id`. The decision
//! itself runs once per `tool_use_id`, detached from any one request.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::watch;

/// The env var the mod reads the endpoint's socket path from. The mod names
/// it as a literal, so changing it means changing `register.ts` too.
pub const GATE_SOCK_ENV: &str = "YAH_GATE_SOCK";

/// The plugin's name, which is also the directory [`install_gate_mod`] writes.
pub const GATE_MOD_NAME: &str = "yah-gate";

/// How long [`serve_gate`] holds one request before answering `pending`.
/// Comfortably under the mod's 30s fetch abort.
pub const DEFAULT_HOLD: Duration = Duration::from_secs(20);

const MOD_FILES: &[(&str, &str)] = &[
    (
        ".claude-plugin/plugin.json",
        include_str!("../mod/yah-gate/.claude-plugin/plugin.json"),
    ),
    ("hooks/hooks.json", include_str!("../mod/yah-gate/hooks/hooks.json")),
    ("hooks/register.ts", include_str!("../mod/yah-gate/hooks/register.ts")),
];

/// Write the embedded mod under `parent/yah-gate` and return that directory,
/// the value for `--plugin-dir`. Files already holding the right bytes are
/// left alone, so concurrent spawns don't rewrite a mod claude is loading.
pub fn install_gate_mod(parent: &Path) -> std::io::Result<PathBuf> {
    let dir = parent.join(GATE_MOD_NAME);
    for (rel, body) in MOD_FILES {
        let path = dir.join(rel);
        if std::fs::read(&path).ok().as_deref() == Some(body.as_bytes()) {
            continue;
        }
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        // Write-then-rename so a concurrent reader never sees a torn file.
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&tmp, body)?;
        std::fs::rename(&tmp, &path)?;
    }
    Ok(dir)
}

/// What claude-cli's own permission layer said before the mod forwarded the
/// call. The host runs its full gate on an `Ask` and only the camp-boundary
/// check on an `Allow`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum CoreDecision {
    #[default]
    Ask,
    Allow,
}

/// One tool call core did not deny, as the mod posts it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GateRequest {
    pub tool: String,
    #[serde(default)]
    pub input: serde_json::Value,
    pub tool_use_id: String,
    /// The subagent loop the call was made in; absent on the main loop.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Core's own verdict. A mod older than this field only forwarded asks,
    /// which is what the default says.
    #[serde(default)]
    pub core_decision: CoreDecision,
    /// Why core said `ask`, when it said.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub core_reason: Option<String>,
}

/// The host's answer. A deny's `reason` is what the model reads.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "decision", rename_all = "lowercase")]
pub enum GateVerdict {
    Allow {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    Deny { reason: String },
}

/// Host policy for an `ask`. May take as long as a person does.
#[async_trait]
pub trait GateDecider: Send + Sync {
    async fn decide(&self, req: GateRequest) -> GateVerdict;
}

type Inflight = Arc<Mutex<HashMap<String, watch::Receiver<Option<GateVerdict>>>>>;

/// Bind the endpoint's socket, creating its parent 0700 and clearing a stale
/// socket file. Keep `path` short: macOS caps a socket path near 104 bytes.
pub fn bind_gate_socket(path: &Path) -> std::io::Result<UnixListener> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700))?;
        }
    }
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    UnixListener::bind(path)
}

/// Serve the mod until the listener fails. Run it as a task and abort it on
/// session stop.
pub async fn serve_gate(listener: UnixListener, decider: Arc<dyn GateDecider>, hold: Duration) {
    let inflight: Inflight = Arc::default();
    loop {
        let stream = match listener.accept().await {
            Ok((s, _)) => s,
            Err(e) => {
                tracing::warn!("yah-gate endpoint accept failed: {e}");
                return;
            }
        };
        let decider = decider.clone();
        let inflight = inflight.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_conn(stream, decider, inflight, hold).await {
                tracing::debug!("yah-gate endpoint connection: {e}");
            }
        });
    }
}

async fn handle_conn(
    stream: UnixStream,
    decider: Arc<dyn GateDecider>,
    inflight: Inflight,
    hold: Duration,
) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream);
    // One request per connection; the mod's fetch does not pipeline.
    let mut request_line = String::new();
    reader.read_line(&mut request_line).await?;
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).await? == 0 {
            break;
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
    }
    const MAX_BODY: usize = 16 << 20;
    if !request_line.starts_with("POST ") {
        return respond(reader.into_inner(), 405, r#"{"error":"POST only"}"#).await;
    }
    if content_length > MAX_BODY {
        return respond(reader.into_inner(), 413, r#"{"error":"body too large"}"#).await;
    }
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body).await?;
    let req: GateRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            let msg = serde_json::json!({ "error": e.to_string() }).to_string();
            return respond(reader.into_inner(), 400, &msg).await;
        }
    };
    let out = match poll(&req, &decider, &inflight, hold).await {
        Some(v) => serde_json::to_string(&v).expect("verdict serializes"),
        None => r#"{"decision":"pending"}"#.to_string(),
    };
    respond(reader.into_inner(), 200, &out).await
}

/// Start (or join) the decision for `req.tool_use_id` and wait up to `hold`.
/// `None` is `pending`. A decided id is forgotten once it has been answered.
async fn poll(
    req: &GateRequest,
    decider: &Arc<dyn GateDecider>,
    inflight: &Inflight,
    hold: Duration,
) -> Option<GateVerdict> {
    let key = req.tool_use_id.clone();
    let mut rx = {
        let mut map = inflight.lock().expect("gate inflight lock");
        map.entry(key.clone())
            .or_insert_with(|| {
                let (tx, rx) = watch::channel(None);
                let decider = decider.clone();
                let req = req.clone();
                tokio::spawn(async move {
                    let v = decider.decide(req).await;
                    let _ = tx.send(Some(v));
                });
                rx
            })
            .clone()
    };
    let decided = match tokio::time::timeout(hold, rx.wait_for(|v| v.is_some())).await {
        Err(_elapsed) => return None,
        Ok(Ok(v)) => v.clone(),
        // The deciding task died (panic) without sending. Answering `pending`
        // here left the entry in place, so every re-poll joined the same dead
        // channel and got `pending` back instantly: a hot loop between the mod
        // and this endpoint until the operator interrupted. Fail closed with a
        // reason the model can read instead.
        Ok(Err(_closed)) => {
            // The desktop's stderr is /dev/null, so this line is the only
            // trace the panic leaves.
            tracing::error!(
                tool = %req.tool,
                tool_use_id = %key,
                "yah-gate decider died before answering (panic?); denying the call"
            );
            Some(GateVerdict::Deny {
                reason: "yah-gate: the approval decider failed before answering; retry the call"
                    .into(),
            })
        }
    };
    inflight.lock().expect("gate inflight lock").remove(&key);
    decided
}

async fn respond(mut stream: UnixStream, status: u16, body: &str) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        _ => "Error",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(body.as_bytes()).await?;
    stream.shutdown().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Slow {
        calls: AtomicUsize,
        delay: Duration,
    }

    #[async_trait]
    impl GateDecider for Slow {
        async fn decide(&self, req: GateRequest) -> GateVerdict {
            self.calls.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(self.delay).await;
            if req.tool == "Bash" {
                GateVerdict::Allow { reason: None }
            } else {
                GateVerdict::Deny { reason: "no".into() }
            }
        }
    }

    async fn post(sock: &Path, body: &str) -> (u16, serde_json::Value) {
        let mut s = UnixStream::connect(sock).await.unwrap();
        let req = format!(
            "POST /tool-check HTTP/1.1\r\nhost: yah\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
            body.len()
        );
        s.write_all(req.as_bytes()).await.unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).await.unwrap();
        let status = out[9..12].parse().unwrap();
        let json = out.split("\r\n\r\n").nth(1).unwrap();
        (status, serde_json::from_str(json).unwrap())
    }

    fn start(delay: Duration, hold: Duration) -> (tempfile::TempDir, PathBuf, Arc<Slow>) {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("g/gate.sock");
        let listener = bind_gate_socket(&sock).unwrap();
        let decider = Arc::new(Slow { calls: AtomicUsize::new(0), delay });
        tokio::spawn(serve_gate(listener, decider.clone(), hold));
        (dir, sock, decider)
    }

    #[tokio::test]
    async fn answers_a_quick_decision_in_one_request() {
        let (_d, sock, _) = start(Duration::ZERO, Duration::from_secs(5));
        let (st, v) = post(&sock, r#"{"tool":"Bash","input":{"command":"ls"},"tool_use_id":"t1"}"#).await;
        assert_eq!(st, 200);
        assert_eq!(v, serde_json::json!({"decision":"allow"}));
        let (_, v) = post(&sock, r#"{"tool":"Write","tool_use_id":"t2"}"#).await;
        assert_eq!(v, serde_json::json!({"decision":"deny","reason":"no"}));
    }

    #[tokio::test]
    async fn long_poll_answers_pending_then_the_verdict_deciding_once() {
        let (_d, sock, decider) = start(Duration::from_millis(400), Duration::from_millis(100));
        let body = r#"{"tool":"Bash","tool_use_id":"same"}"#;
        let mut polls = 0;
        let verdict = loop {
            polls += 1;
            let (_, v) = post(&sock, body).await;
            if v["decision"] != "pending" {
                break v;
            }
            assert!(polls < 20, "never decided");
        };
        assert_eq!(verdict["decision"], "allow");
        assert!(polls >= 3, "expected pending polls first, got {polls}");
        assert_eq!(decider.calls.load(Ordering::SeqCst), 1);
    }

    struct Panics;

    #[async_trait]
    impl GateDecider for Panics {
        async fn decide(&self, _req: GateRequest) -> GateVerdict {
            panic!("decider blew up")
        }
    }

    #[tokio::test]
    async fn a_dead_decider_denies_once_instead_of_pending_forever() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("g/gate.sock");
        let listener = bind_gate_socket(&sock).unwrap();
        tokio::spawn(serve_gate(listener, Arc::new(Panics), Duration::from_secs(5)));
        let body = r#"{"tool":"Read","tool_use_id":"dies"}"#;
        let (_, v) = post(&sock, body).await;
        assert_eq!(v["decision"], "deny", "got {v}");
        assert!(v["reason"].as_str().unwrap().contains("decider failed"));
    }

    #[test]
    fn core_decision_defaults_to_ask_for_an_older_mod() {
        let r: GateRequest = serde_json::from_str(r#"{"tool":"Read","tool_use_id":"t"}"#).unwrap();
        assert_eq!(r.core_decision, CoreDecision::Ask);
        let r: GateRequest =
            serde_json::from_str(r#"{"tool":"Read","tool_use_id":"t","core_decision":"allow"}"#)
                .unwrap();
        assert_eq!(r.core_decision, CoreDecision::Allow);
    }

    #[tokio::test]
    async fn rejects_bad_requests() {
        let (_d, sock, _) = start(Duration::ZERO, Duration::from_secs(1));
        let (st, _) = post(&sock, "not json").await;
        assert_eq!(st, 400);
    }

    #[test]
    fn bind_makes_the_parent_owner_only() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("p/gate.sock");
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let _l = rt.block_on(async { bind_gate_socket(&sock).unwrap() });
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(sock.parent().unwrap()).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o700);
        }
    }

    #[test]
    fn install_writes_the_mod_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let m = install_gate_mod(dir.path()).unwrap();
        assert_eq!(m, dir.path().join("yah-gate"));
        let reg = std::fs::read_to_string(m.join("hooks/register.ts")).unwrap();
        assert!(reg.contains(GATE_SOCK_ENV), "mod must read the env var pv exports");
        assert!(m.join(".claude-plugin/plugin.json").is_file());
        let before = std::fs::metadata(m.join("hooks/hooks.json")).unwrap().modified().unwrap();
        install_gate_mod(dir.path()).unwrap();
        let after = std::fs::metadata(m.join("hooks/hooks.json")).unwrap().modified().unwrap();
        assert_eq!(before, after);
    }
}
