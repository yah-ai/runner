//! claude-cli's own on-disk transcript: the deterministic session uuid, the
//! `~/.claude/projects/<encoded-cwd>/<uuid>.jsonl` layout, fork / rewind /
//! pre-compact snapshot surgery on it, and a tailer that yields per-API-call
//! usage from it.
//!
//! Part of R951 — copied from `app/yah/desktop/src/agent_process.rs` and
//! `crates/yah/party/src/agent.rs` (`claude_session_uuid`); R951-F3 makes this
//! the single definition. Every path argument here is the subprocess **cwd** —
//! claude-cli keys its project directory off it and nothing else.

use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio::sync::mpsc;
use yah_runner_core::{AgentEvent, SessionId};

use crate::translate::assistant_entry_usage;

/// Deterministic claude-cli session uuid for a [`SessionId`]: blake3 of the
/// id, shaped as an RFC 4122 v4 uuid. Spawn (`--session-id`) and every later
/// reader (`--resume`, the transcript path, a hook's `${session_id}`) derive
/// the same value without persisting it.
pub fn claude_session_uuid(session_id: &SessionId) -> String {
    let digest = blake3::hash(session_id.as_str().as_bytes());
    let mut b = [0u8; 16];
    b.copy_from_slice(&digest.as_bytes()[..16]);
    // RFC 4122: version 4 in the high nibble of byte 6, variant '10' in
    // the high two bits of byte 8. Bit-twiddling keeps the rest random.
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15],
    )
}

/// `<claude-proj>/<uuid>.jsonl` for a subprocess running in `cwd`.
pub fn claude_jsonl_path(cwd: &Path, claude_uuid: &str) -> PathBuf {
    claude_project_dir(cwd).join(format!("{claude_uuid}.jsonl"))
}

/// Claude Code stores per-project JSONL transcripts under
/// `~/.claude/projects/<encoded-cwd>/<uuid>.jsonl`. The directory name
/// is the absolute cwd path with every `/` and `.` mapped to `-`
/// (e.g. `/Users/user/ss/yah/.claude/x` → `-Users-user-ss-yah--claude-x`).
pub fn claude_project_dir(cwd: &Path) -> PathBuf {
    let home = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"));
    claude_project_dir_under(&home, cwd)
}

/// Inner form of [`claude_project_dir`] that takes an explicit home root.
/// Lets tests target a TempDir without mutating the process-global `HOME`.
fn claude_project_dir_under(home: &Path, cwd: &Path) -> PathBuf {
    let encoded: String = cwd
        .to_string_lossy()
        .chars()
        .map(|c| if c == '/' || c == '.' { '-' } else { c })
        .collect();
    home.join(".claude").join("projects").join(encoded)
}

#[cfg(test)]
fn claude_jsonl_path_under(home: &Path, cwd: &Path, claude_uuid: &str) -> PathBuf {
    claude_project_dir_under(home, cwd).join(format!("{claude_uuid}.jsonl"))
}

/// True iff `entry` is a `type:"user"` JSONL line whose message content is a
/// user-authored text turn (NOT a tool_result wrapper). Used by
/// [`fork_claude_jsonl`] to count conversational user turns past which to
/// truncate. Tool result lines carry `content: [{type:"tool_result",...}]`
/// and are passed through as part of the prior assistant turn.
fn is_user_text_entry(entry: &serde_json::Value) -> bool {
    let Some(msg) = entry.get("message") else {
        return false;
    };
    let Some(content) = msg.get("content") else {
        return false;
    };
    if content.is_string() {
        return true;
    }
    if let Some(arr) = content.as_array() {
        return arr
            .iter()
            .any(|blk| blk.get("type").and_then(|t| t.as_str()) == Some("text"));
    }
    false
}

/// Copy `<claude-proj>/<old_uuid>.jsonl` to `<claude-proj>/<new_uuid>.jsonl`
/// truncated after `keep_user_turns` user-text entries (1-indexed), with
/// `sessionId` rewritten to `new_uuid` on every line. Used by the fork-from-
/// message path so `claude --resume <new_uuid>` loads a snipped transcript
/// (R202-F3).
///
/// A user-text turn is a `type:"user"` line whose `message.content` is plain
/// text — see [`is_user_text_entry`]. `tool_result` user lines, assistant
/// blocks, and claude metadata (`queue-operation`, `attachment`, `ai-title`,
/// `last-prompt`) ride along with the preceding turn. Once the
/// `keep_user_turns + 1`-th user-text line appears we stop — that line is
/// the prompt for the turn AFTER the fork point and must NOT be in the new
/// transcript or claude would replay it.
///
/// Malformed JSON lines are skipped. Empty source file is treated as a
/// no-op (still creates a zero-byte destination so callers see Ok).
pub fn fork_claude_jsonl(
    cwd: &Path,
    old_uuid: &str,
    new_uuid: &str,
    keep_user_turns: usize,
) -> std::io::Result<()> {
    let home = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"));
    fork_claude_jsonl_under(&home, cwd, old_uuid, new_uuid, keep_user_turns)
}

/// Truncate `<claude-proj>/<uuid>.jsonl` in place after `keep_user_turns`
/// user-text entries (1-indexed). SessionId/uuid is preserved (rewind
/// keeps the same `--resume <uuid>`). Truncated content lands in a
/// sibling temp file and is atomically renamed onto the source so
/// `claude --resume` never observes a partial write. R250-F2.
pub fn truncate_claude_jsonl(
    cwd: &Path,
    uuid: &str,
    keep_user_turns: usize,
) -> std::io::Result<PathBuf> {
    let home = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"));
    truncate_claude_jsonl_under(&home, cwd, uuid, keep_user_turns)
}

/// Inner form: explicit home root for tests. See [`truncate_claude_jsonl`].
fn truncate_claude_jsonl_under(
    home: &Path,
    cwd: &Path,
    uuid: &str,
    keep_user_turns: usize,
) -> std::io::Result<PathBuf> {
    use std::io::{BufRead, BufReader, BufWriter, Write};

    let path = claude_project_dir_under(home, cwd).join(format!("{uuid}.jsonl"));
    if !path.exists() {
        return Ok(path);
    }
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::other("claude jsonl has no parent dir")
    })?;
    let tmp = parent.join(format!(".{uuid}.rewind.tmp"));

    let in_file = std::fs::File::open(&path)?;
    let out_file = std::fs::File::create(&tmp)?;
    let reader = BufReader::new(in_file);
    let mut writer = BufWriter::new(out_file);

    let mut user_text_count: usize = 0;
    for line_result in reader.lines() {
        let line = line_result?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if entry.get("type").and_then(|t| t.as_str()) == Some("user") && is_user_text_entry(&entry)
        {
            user_text_count += 1;
            if user_text_count > keep_user_turns {
                break;
            }
        }
        writeln!(writer, "{}", serde_json::to_string(&entry)?)?;
    }
    writer.flush()?;
    drop(writer);
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

/// Save a verbatim copy of the current claude JSONL as a `.pre-compact.jsonl`
/// sidecar in the same directory. Called immediately before `/compact` is
/// written to the subprocess stdin so the snapshot reflects the pre-compact
/// conversation. Returns the sidecar path. No-op (returns `Ok`) when the
/// source JSONL doesn't exist yet (e.g. very first turn). R283.
pub fn snapshot_pre_compact_claude_jsonl(cwd: &Path, uuid: &str) -> std::io::Result<PathBuf> {
    let home = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"));
    snapshot_pre_compact_claude_jsonl_under(&home, cwd, uuid)
}

/// Inner form: explicit home root for tests.
fn snapshot_pre_compact_claude_jsonl_under(
    home: &Path,
    cwd: &Path,
    uuid: &str,
) -> std::io::Result<PathBuf> {
    let proj_dir = claude_project_dir_under(home, cwd);
    let src = proj_dir.join(format!("{uuid}.jsonl"));
    let dst = proj_dir.join(format!("{uuid}.pre-compact.jsonl"));
    if src.exists() {
        std::fs::copy(&src, &dst)?;
    }
    Ok(dst)
}

/// Copy the pre-compact snapshot sidecar (`<old_uuid>.pre-compact.jsonl`) to a
/// new UUID's JSONL path, rewriting every `sessionId` field from `old_uuid` to
/// `new_uuid`. All lines are preserved (no truncation — the snapshot already
/// ends at the pre-compact boundary). Returns an error when the snapshot
/// doesn't exist (compact was never triggered or the sidecar was deleted). R283.
pub fn fork_claude_from_compact_snapshot(
    cwd: &Path,
    old_uuid: &str,
    new_uuid: &str,
) -> std::io::Result<()> {
    let home = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"));
    fork_claude_from_compact_snapshot_under(&home, cwd, old_uuid, new_uuid)
}

fn fork_claude_from_compact_snapshot_under(
    home: &Path,
    cwd: &Path,
    old_uuid: &str,
    new_uuid: &str,
) -> std::io::Result<()> {
    use std::io::{BufRead, BufReader, BufWriter, Write};
    let proj_dir = claude_project_dir_under(home, cwd);
    let src = proj_dir.join(format!("{old_uuid}.pre-compact.jsonl"));
    let dst = proj_dir.join(format!("{new_uuid}.jsonl"));
    if !src.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("pre-compact snapshot not found: {}", src.display()),
        ));
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let in_file = std::fs::File::open(&src)?;
    let out_file = std::fs::File::create(&dst)?;
    let reader = BufReader::new(in_file);
    let mut writer = BufWriter::new(out_file);
    for line_result in reader.lines() {
        let line = line_result?;
        if line.trim().is_empty() {
            continue;
        }
        let mut entry: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(obj) = entry.as_object_mut() {
            if obj.contains_key("sessionId") {
                obj.insert(
                    "sessionId".into(),
                    serde_json::Value::String(new_uuid.to_string()),
                );
            }
        }
        writeln!(writer, "{}", serde_json::to_string(&entry)?)?;
    }
    writer.flush()?;
    Ok(())
}

/// Inner form: explicit home root for tests. See [`fork_claude_jsonl`].
fn fork_claude_jsonl_under(
    home: &Path,
    cwd: &Path,
    old_uuid: &str,
    new_uuid: &str,
    keep_user_turns: usize,
) -> std::io::Result<()> {
    use std::io::{BufRead, BufReader, BufWriter, Write};

    let src = claude_project_dir_under(home, cwd).join(format!("{old_uuid}.jsonl"));
    let dst = claude_project_dir_under(home, cwd).join(format!("{new_uuid}.jsonl"));

    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let in_file = std::fs::File::open(&src)?;
    let out_file = std::fs::File::create(&dst)?;
    let reader = BufReader::new(in_file);
    let mut writer = BufWriter::new(out_file);

    let mut user_text_count: usize = 0;
    for line_result in reader.lines() {
        let line = line_result?;
        if line.trim().is_empty() {
            continue;
        }
        let mut entry: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if entry.get("type").and_then(|t| t.as_str()) == Some("user") && is_user_text_entry(&entry)
        {
            user_text_count += 1;
            if user_text_count > keep_user_turns {
                break;
            }
        }
        if let Some(obj) = entry.as_object_mut() {
            if obj.contains_key("sessionId") {
                obj.insert(
                    "sessionId".into(),
                    serde_json::Value::String(new_uuid.to_string()),
                );
            }
        }
        writeln!(writer, "{}", serde_json::to_string(&entry)?)?;
    }
    writer.flush()?;
    Ok(())
}

/// Tail claude's own JSONL transcript and send a `TurnUsageDelta` for each
/// new assistant entry. The transcript carries the definitive per-API-call
/// usage block — in particular the per-call `input_tokens` before any
/// multi-call accumulation. Polling at 500 ms balances ring-fill freshness
/// against per-session watcher overhead.
///
/// Waits up to 10 s for the file to appear (claude creates it on its first
/// write), then seeks to EOF so historical entries aren't replayed. Returns
/// when `tx`'s receiver is dropped, or when the file never appears; a host
/// that wants it gone sooner aborts the task.
pub async fn run_jsonl_watcher(
    session_id: SessionId,
    jsonl_path: PathBuf,
    tx: mpsc::Sender<AgentEvent>,
) {
    let mut file = None;
    for _ in 0u32..40 {
        match tokio::fs::File::open(&jsonl_path).await {
            Ok(f) => {
                file = Some(f);
                break;
            }
            Err(_) => tokio::time::sleep(Duration::from_millis(250)).await,
        }
    }
    let Some(mut file) = file else {
        tracing::debug!(path = %jsonl_path.display(), "jsonl watcher: file never appeared, exiting");
        return;
    };

    if file.seek(std::io::SeekFrom::End(0)).await.is_err() {
        return;
    }

    let mut tail = String::new();
    let mut chunk = vec![0u8; 8192];
    loop {
        tokio::time::sleep(Duration::from_millis(500)).await;
        if tx.is_closed() {
            return;
        }

        loop {
            match file.read(&mut chunk).await {
                Ok(0) => break,
                Ok(n) => match std::str::from_utf8(&chunk[..n]) {
                    Ok(s) => tail.push_str(s),
                    Err(_) => break,
                },
                Err(_) => break,
            }
        }

        while let Some(nl) = tail.find('\n') {
            let line = tail[..nl].trim();
            if !line.is_empty() {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                    if v.get("type").and_then(|t| t.as_str()) == Some("assistant") {
                        if let Some(usage) = assistant_entry_usage(&v) {
                            let ev = AgentEvent::TurnUsageDelta {
                                session_id: session_id.clone(),
                                usage,
                            };
                            if tx.send(ev).await.is_err() {
                                return;
                            }
                        }
                    }
                }
            }
            tail = tail[nl + 1..].to_string();
        }
    }
}

#[cfg(test)]
mod tests {
    //! Part of R951 — copied from the desktop `agent_process.rs` tests
    //! (R951-F3 deletes the originals).
    use super::*;
    use serde_json::{json, Value};
    use tempfile::TempDir;

    fn sid() -> SessionId {
        SessionId::new("session:deadbeef")
    }

    /// Build a minimal claude-shaped JSONL transcript for fork-truncation
    /// tests. Three user-text turns, one assistant + tool_result entry
    /// between them, plus a queue-operation metadata line up front.
    fn write_claude_jsonl_fixture(home: &Path, camp_root: &Path, uuid: &str) {
        let dir = super::claude_project_dir_under(home, camp_root);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{uuid}.jsonl"));
        let entries: Vec<Value> = vec![
            json!({"type": "queue-operation", "operation": "enqueue", "sessionId": uuid}),
            json!({"type": "user", "sessionId": uuid, "uuid": "u0",
                   "message": {"role": "user", "content": [{"type": "text", "text": "turn 1"}]}}),
            json!({"type": "assistant", "sessionId": uuid, "uuid": "a0", "parentUuid": "u0",
                   "message": {"role": "assistant", "content": [{"type": "tool_use", "name": "Bash"}]}}),
            json!({"type": "user", "sessionId": uuid, "uuid": "tr0", "parentUuid": "a0",
                   "message": {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]}}),
            json!({"type": "assistant", "sessionId": uuid, "uuid": "a1", "parentUuid": "tr0",
                   "message": {"role": "assistant", "content": [{"type": "text", "text": "done 1"}]}}),
            json!({"type": "user", "sessionId": uuid, "uuid": "u1", "parentUuid": "a1",
                   "message": {"role": "user", "content": [{"type": "text", "text": "turn 2"}]}}),
            json!({"type": "assistant", "sessionId": uuid, "uuid": "a2", "parentUuid": "u1",
                   "message": {"role": "assistant", "content": [{"type": "text", "text": "done 2"}]}}),
            json!({"type": "user", "sessionId": uuid, "uuid": "u2", "parentUuid": "a2",
                   "message": {"role": "user", "content": [{"type": "text", "text": "turn 3"}]}}),
            json!({"type": "assistant", "sessionId": uuid, "uuid": "a3", "parentUuid": "u2",
                   "message": {"role": "assistant", "content": [{"type": "text", "text": "done 3"}]}}),
        ];
        let body = entries
            .iter()
            .map(|v| serde_json::to_string(v).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        std::fs::write(path, body).unwrap();
    }

    fn read_jsonl(path: &Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .collect()
    }

    #[test]
    fn fork_claude_jsonl_truncates_after_first_user_turn() {
        let home = TempDir::new().unwrap();
        let camp_root = home.path().join("camp");
        std::fs::create_dir_all(&camp_root).unwrap();
        let old_uuid = "old-uuid-1111";
        let new_uuid = "new-uuid-2222";
        write_claude_jsonl_fixture(home.path(), &camp_root, old_uuid);

        super::fork_claude_jsonl_under(home.path(), &camp_root, old_uuid, new_uuid, 1).unwrap();

        let dst = super::claude_jsonl_path_under(home.path(), &camp_root, new_uuid);
        let entries = read_jsonl(&dst);
        // Should include: queue-op, user u0, assistant a0, user tr0 (tool_result),
        // assistant a1 — everything before the second user-text entry u1.
        let kinds: Vec<&str> = entries
            .iter()
            .map(|e| e["type"].as_str().unwrap())
            .collect();
        assert_eq!(
            kinds,
            vec!["queue-operation", "user", "assistant", "user", "assistant"]
        );
        // u1 (second user-text turn) MUST NOT appear — otherwise claude would
        // replay the prompt for the turn after the fork point.
        for e in &entries {
            assert_ne!(e.get("uuid").and_then(|v| v.as_str()), Some("u1"));
        }
        // Every entry's sessionId rewritten to the new uuid.
        for e in &entries {
            if let Some(sid) = e.get("sessionId").and_then(|v| v.as_str()) {
                assert_eq!(sid, new_uuid, "sessionId not rewritten on {e:?}");
            }
        }
    }

    #[test]
    fn fork_claude_jsonl_keep_all_turns_clones_entire_transcript() {
        let home = TempDir::new().unwrap();
        let camp_root = home.path().join("camp");
        std::fs::create_dir_all(&camp_root).unwrap();
        let old_uuid = "old-uuid-3333";
        let new_uuid = "new-uuid-4444";
        write_claude_jsonl_fixture(home.path(), &camp_root, old_uuid);

        // 3 user-text turns in fixture; asking for ≥3 keeps everything.
        super::fork_claude_jsonl_under(home.path(), &camp_root, old_uuid, new_uuid, 99).unwrap();

        let dst = super::claude_jsonl_path_under(home.path(), &camp_root, new_uuid);
        let entries = read_jsonl(&dst);
        // Fixture has 9 lines.
        assert_eq!(entries.len(), 9);
    }

    #[test]
    fn fork_claude_jsonl_skips_tool_result_user_lines_in_turn_count() {
        // A `user` line carrying only a tool_result MUST NOT increment the
        // user-text count — otherwise truncating after N "turns" would cut
        // mid-turn (between an assistant tool_use and its tool_result).
        let home = TempDir::new().unwrap();
        let camp_root = home.path().join("camp");
        std::fs::create_dir_all(&camp_root).unwrap();
        let old_uuid = "old-uuid-5555";
        let new_uuid = "new-uuid-6666";
        write_claude_jsonl_fixture(home.path(), &camp_root, old_uuid);

        super::fork_claude_jsonl_under(home.path(), &camp_root, old_uuid, new_uuid, 2).unwrap();

        let dst = super::claude_jsonl_path_under(home.path(), &camp_root, new_uuid);
        let entries = read_jsonl(&dst);
        // Should reach through u1 + its assistant turn a2, stop before u2.
        let last = entries.last().unwrap();
        assert_eq!(last["uuid"].as_str(), Some("a2"));
        // u2 must not appear.
        for e in &entries {
            assert_ne!(e.get("uuid").and_then(|v| v.as_str()), Some("u2"));
        }
    }

    #[test]
    fn claude_project_dir_encodes_slashes_and_dots() {
        let home = Path::new("/home/user");
        let dir = super::claude_project_dir_under(home, Path::new("/Users/user/ss/yah"));
        assert_eq!(
            dir,
            Path::new("/home/user/.claude/projects/-Users-user-ss-yah")
        );
        let dir = super::claude_project_dir_under(home, Path::new("/a/b/.c/d"));
        assert_eq!(dir, Path::new("/home/user/.claude/projects/-a-b--c-d"));
    }

    #[test]
    fn snapshot_pre_compact_creates_sidecar_copy() {
        let home = TempDir::new().unwrap();
        let camp_root = home.path().join("camp");
        let uuid = "snap-uuid-0001";
        write_claude_jsonl_fixture(home.path(), &camp_root, uuid);

        let proj_dir = super::claude_project_dir_under(home.path(), &camp_root);
        let src = proj_dir.join(format!("{uuid}.jsonl"));
        let snap = proj_dir.join(format!("{uuid}.pre-compact.jsonl"));

        assert!(src.exists(), "fixture must exist before snapshot");
        assert!(
            !snap.exists(),
            "sidecar must not exist before snapshot call"
        );

        super::snapshot_pre_compact_claude_jsonl_under(home.path(), &camp_root, uuid).unwrap();

        assert!(snap.exists(), "sidecar must be created after snapshot call");
        let src_content = std::fs::read_to_string(&src).unwrap();
        let snap_content = std::fs::read_to_string(&snap).unwrap();
        assert_eq!(
            src_content, snap_content,
            "sidecar must be byte-identical to source"
        );
    }

    #[test]
    fn fork_claude_from_compact_snapshot_rewrites_session_id() {
        let home = TempDir::new().unwrap();
        let camp_root = home.path().join("camp");
        let old_uuid = "snap-uuid-0002";
        let new_uuid = "snap-new-0003";
        write_claude_jsonl_fixture(home.path(), &camp_root, old_uuid);

        // Create the sidecar manually (simulating what snapshot_pre_compact_claude_jsonl does).
        let proj_dir = super::claude_project_dir_under(home.path(), &camp_root);
        std::fs::copy(
            proj_dir.join(format!("{old_uuid}.jsonl")),
            proj_dir.join(format!("{old_uuid}.pre-compact.jsonl")),
        )
        .unwrap();

        super::fork_claude_from_compact_snapshot_under(home.path(), &camp_root, old_uuid, new_uuid)
            .unwrap();

        let dst = super::claude_jsonl_path_under(home.path(), &camp_root, new_uuid);
        assert!(dst.exists(), "forked JSONL must be created");
        let entries = read_jsonl(&dst);
        // All 9 lines from the fixture (no truncation — snapshot is used as-is).
        assert_eq!(entries.len(), 9, "all lines from snapshot must be copied");
        for e in &entries {
            if let Some(sid) = e.get("sessionId").and_then(|v| v.as_str()) {
                assert_eq!(sid, new_uuid, "sessionId must be rewritten to new_uuid");
            }
        }
    }

    #[test]
    fn fork_claude_from_compact_snapshot_errors_when_no_snapshot() {
        let home = TempDir::new().unwrap();
        let camp_root = home.path().join("camp");
        std::fs::create_dir_all(&camp_root).unwrap();

        let result = super::fork_claude_from_compact_snapshot_under(
            home.path(),
            &camp_root,
            "no-snapshot-uuid",
            "new-uuid-x",
        );
        assert!(
            result.is_err(),
            "must error when pre-compact snapshot is absent"
        );
        assert!(
            result.unwrap_err().kind() == std::io::ErrorKind::NotFound,
            "error kind must be NotFound"
        );
    }

    #[test]
    fn claude_session_uuid_is_a_stable_v4_shaped_hash() {
        let uuid = claude_session_uuid(&SessionId::new("session:abc12345"));
        // Pinned to the same vector as party's copy (crates/yah/party/src/agent.rs)
        // so the two cannot drift before R951-F3 makes this the only one.
        assert_eq!(uuid, "615df1c0-45a5-4bca-9894-6d66a104227d");
        assert_eq!(&uuid[14..15], "4");
        assert!(matches!(&uuid[19..20], "8" | "9" | "a" | "b"));
    }

    #[tokio::test]
    async fn jsonl_watcher_yields_usage_for_entries_appended_after_open() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("t.jsonl");
        // A pre-existing entry must NOT be replayed (watcher seeks to EOF).
        std::fs::write(
            &path,
            "{\"type\":\"assistant\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n",
        )
        .unwrap();
        let (tx, mut rx) = mpsc::channel(4);
        let task = tokio::spawn(run_jsonl_watcher(sid(), path.clone(), tx));
        tokio::time::sleep(Duration::from_millis(200)).await;
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
            writeln!(
                f,
                "{{\"type\":\"assistant\",\"message\":{{\"usage\":{{\"input_tokens\":7,\"output_tokens\":3}}}}}}"
            )
            .unwrap();
        }
        let ev = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("watcher should yield within 5s")
            .expect("channel open");
        match ev {
            AgentEvent::TurnUsageDelta { usage, .. } => {
                assert_eq!(usage.input_tokens, Some(7));
                assert_eq!(usage.output_tokens, Some(3));
            }
            other => panic!("expected TurnUsageDelta, got {other:?}"),
        }
        drop(rx);
        task.await.unwrap();
    }
}
