//! Live check of the `yah-gate` mod against a real `claude -p` (R950-F14).
//!
//! Installs the embedded mod, serves the endpoint with a decider that records
//! every request and allows it, then asks claude to run a Bash command core
//! would `ask` about. Exits non-zero unless the Bash call reached the endpoint
//! and its output came back.
//!
//! `cargo run -p yah-runner-pv --example gate_fixture` (spends one small turn).

use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use yah_runner_pv::gate::{
    bind_gate_socket, install_gate_mod, serve_gate, GateDecider, GateRequest, GateVerdict,
    DEFAULT_HOLD, GATE_SOCK_ENV,
};

#[derive(Default)]
struct Recorder(Mutex<Vec<GateRequest>>);

#[async_trait]
impl GateDecider for Recorder {
    async fn decide(&self, req: GateRequest) -> GateVerdict {
        eprintln!("gate_fixture: endpoint asked about {} {}", req.tool, req.input);
        self.0.lock().unwrap().push(req);
        GateVerdict::Allow { reason: Some("gate_fixture allows".into()) }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let dir = tempfile::tempdir().unwrap();
    let mod_dir = install_gate_mod(&dir.path().join("mods")).unwrap();
    let sock = dir.path().join("g/gate.sock");
    let listener = bind_gate_socket(&sock).unwrap();
    let rec = Arc::new(Recorder::default());
    tokio::spawn(serve_gate(listener, rec.clone(), DEFAULT_HOLD));

    let marker = "gate-fixture-7c41";
    let out = tokio::process::Command::new("claude")
        .current_dir(dir.path())
        .env(GATE_SOCK_ENV, &sock)
        .args(["-p", "--output-format", "text", "--model", "claude-haiku-4-5-20251001"])
        .arg("--plugin-dir")
        .arg(&mod_dir)
        .arg(format!(
            "Use the Bash tool to run exactly: echo {marker} > out.txt && cat out.txt. Then reply with its output only."
        ))
        .output()
        .await
        .expect("claude on PATH");
    let stdout = String::from_utf8_lossy(&out.stdout);
    eprintln!("gate_fixture: claude exit {:?}\n{stdout}", out.status.code());
    let asked_bash = rec.0.lock().unwrap().iter().any(|r| r.tool == "Bash");
    let wrote = std::fs::read_to_string(dir.path().join("out.txt")).unwrap_or_default();
    if asked_bash && wrote.contains(marker) {
        println!("PASS: Bash ask resolved through the yah-gate endpoint and ran");
    } else {
        println!("FAIL: asked_bash={asked_bash} file={wrote:?}");
        std::process::exit(1);
    }
}
