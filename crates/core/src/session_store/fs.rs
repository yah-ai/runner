//! [`FsSessionStore`]: one `<root>/<session_id>.jsonl` per session, plus
//! per-session sidecars (`<id>.terminated`, and whatever extension a host
//! asks [`FsSessionStore::sidecar_path`] for) in the same directory.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

use super::{sanitize_id, LogEvent, SessionStore, SessionStoreError};
use crate::agent::{SessionId, TurnUsage};
use crate::event::AgentEvent;

/// Maximum events an ephemeral store's ring buffer retains; the oldest drops
/// on overflow.
pub const EPHEMERAL_RING_CAP: usize = 200;

const MS_PER_DAY: u64 = 86_400_000;

/// Tool-result bodies whose serialized length exceeds this are replaced with a
/// `{"_jettisoned":true,"bytes":N}` stub by
/// [`FsSessionStore::sweep_jettison_tool_bodies`] (R588-F4). Smaller results
/// are cheap and still useful in replay, so they're kept.
const JETTISON_MIN_BYTES: usize = 512;

/// A session's tool-result bodies aren't jettisoned until its JSONL has been
/// idle (unmodified) at least this long — so a just-ended or about-to-resume
/// session keeps its bodies through the live window (R588-F4).
const JETTISON_GRACE_MS: u64 = MS_PER_DAY;

/// Persists `E` records to per-session JSONL files under a root directory.
///
/// Cheap to clone (an `Arc` internally) so a runner can stash one in its
/// session record and keep wrapping streams.
pub struct FsSessionStore<E> {
    inner: Arc<Inner<E>>,
}

impl<E> Clone for FsSessionStore<E> {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

struct Inner<E> {
    /// Directory holding the logs. `None` for an ephemeral store that skips
    /// disk writes entirely.
    root: Option<PathBuf>,
    /// One write-mutex per store keeps concurrent appends from interleaving
    /// lines on overlapping flushes. Per-session locks would scale better, but
    /// each session writes from one task, so a single mutex is the simpler
    /// shape.
    write_lock: Mutex<()>,
    /// Running cumulative token totals per session. Populated on each
    /// turn-usage append; drained into the session-ended record so every
    /// JSONL ends with a single-line spend summary.
    usage_acc: std::sync::Mutex<HashMap<String, TurnUsage>>,
    /// Optional in-memory ring for ephemeral stores, shared with the host so
    /// events outlive the session. Capped at [`EPHEMERAL_RING_CAP`].
    ring: Option<Arc<std::sync::Mutex<VecDeque<E>>>>,
}

impl<E: LogEvent> FsSessionStore<E> {
    fn build(root: Option<PathBuf>, ring: Option<Arc<std::sync::Mutex<VecDeque<E>>>>) -> Self {
        Self {
            inner: Arc::new(Inner {
                root,
                write_lock: Mutex::new(()),
                usage_acc: std::sync::Mutex::new(HashMap::new()),
                ring,
            }),
        }
    }

    /// A store whose logs live directly in `root`. Does not create the
    /// directory eagerly — that happens on the first write — so a read-only
    /// consumer can construct one without side effects.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self::build(Some(root.into()), None)
    }

    /// A store that discards every event — no JSONL, no running marker.
    pub fn ephemeral() -> Self {
        Self::build(None, None)
    }

    /// An ephemeral store that additionally captures events into a shared
    /// ring buffer, so a host can replay recent exchanges without writing
    /// files.
    pub fn ephemeral_with_ring(ring: Arc<std::sync::Mutex<VecDeque<E>>>) -> Self {
        Self::build(None, Some(ring))
    }

    /// `true` when this store discards events.
    pub fn is_ephemeral(&self) -> bool {
        self.inner.root.is_none()
    }

    /// The directory logs live in. `None` for an ephemeral store.
    pub fn root(&self) -> Option<&Path> {
        self.inner.root.as_deref()
    }

    /// `<root>/<session_id>.<ext>`. The one place a per-session file name is
    /// formed, so every sidecar gets the same id sanitizing. `None` for an
    /// ephemeral store.
    pub fn sidecar_path(
        &self,
        session_id: &SessionId,
        ext: &str,
    ) -> Result<Option<PathBuf>, SessionStoreError> {
        match &self.inner.root {
            None => Ok(None),
            Some(root) => {
                let safe = sanitize_id(session_id.as_str())?;
                Ok(Some(root.join(format!("{safe}.{ext}"))))
            }
        }
    }

    /// `<root>/<session_id>.jsonl`. `None` for an ephemeral store.
    pub fn path_for(&self, session_id: &SessionId) -> Result<Option<PathBuf>, SessionStoreError> {
        self.sidecar_path(session_id, "jsonl")
    }

    /// Append one event. Creates the parent directory + log file on first
    /// write. Holds an internal mutex so concurrent appends from two tasks
    /// can't interleave bytes on the same flush.
    ///
    /// Two things happen beyond plain serialization:
    /// - A `"ts"` field (Unix epoch milliseconds) is injected into every
    ///   record so raw-JSONL analysis tools can filter/sort by time.
    /// - Turn usage is accumulated in-memory; when the session-ended record
    ///   arrives the cumulative total is written into it as `"usage"`, giving
    ///   each JSONL a single-line session spend summary.
    pub async fn append(&self, session_id: &SessionId, event: &E) -> Result<(), SessionStoreError> {
        let path = match self.path_for(session_id)? {
            None => {
                // Ephemeral: push to ring buffer if present, then discard.
                if let Some(ring) = &self.inner.ring {
                    let mut guard = ring.lock().unwrap();
                    if guard.len() >= EPHEMERAL_RING_CAP {
                        guard.pop_front();
                    }
                    guard.push_back(event.clone());
                }
                return Ok(());
            }
            Some(p) => p,
        };

        // Accumulate per-turn usage before serializing so the running total
        // is always up-to-date when the session ends.
        if let Some(u) = event.turn_usage() {
            let mut acc = self.inner.usage_acc.lock().unwrap();
            acc.entry(session_id.as_str().to_owned()).or_default().accumulate(u);
        }

        let ended = event.is_session_ended();
        let mut obj = serde_json::to_value(event)?;
        if let Some(map) = obj.as_object_mut() {
            // Timestamp: milliseconds since Unix epoch, injected at write
            // time so the JSONL is sortable without a wrapper.
            let ts_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            map.insert("ts".into(), ts_ms.into());

            // Cumulative session spend injected into the terminal record.
            if ended {
                let cumulative = self.inner.usage_acc.lock().unwrap().remove(session_id.as_str());
                if let Some(usage) = cumulative {
                    if let Ok(v) = serde_json::to_value(&usage) {
                        map.insert("usage".into(), v);
                    }
                }
            }
        }

        let mut line = serde_json::to_vec(&obj)?;
        line.push(b'\n');

        let _guard = self.inner.write_lock.lock().await;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;
        file.write_all(&line).await?;
        file.flush().await?;

        // Clear the running marker when the terminal event lands so readers
        // see the session as cleanly ended.
        if ended {
            if let Err(e) = self.clear_running(session_id) {
                tracing::warn!(
                    session_id = %session_id.as_str(),
                    error = %e,
                    "session log: failed to clear .terminated marker on session_ended",
                );
            }
        }

        Ok(())
    }

    /// Path of the running marker for `session_id`: `<root>/<id>.terminated`.
    /// The file exists while the session is live; it is created by
    /// [`Self::touch_running`] at session registration and deleted by
    /// [`Self::clear_running`] when the session-ended record is appended. If
    /// the process is killed before that record lands, the file remains on
    /// disk so readers can distinguish "paused cleanly between turns" from
    /// "killed mid-stream."
    pub fn running_marker_path(
        &self,
        session_id: &SessionId,
    ) -> Result<Option<PathBuf>, SessionStoreError> {
        self.sidecar_path(session_id, "terminated")
    }

    /// Write the running marker synchronously. Must be called before any
    /// async session work begins so a SIGKILL mid-session leaves the marker
    /// on disk. Best-effort — caller logs on error and continues; a missing
    /// marker degrades to "ambiguous" not "incorrect." No-op for ephemeral
    /// stores.
    pub fn touch_running(&self, session_id: &SessionId) -> Result<(), SessionStoreError> {
        let path = match self.running_marker_path(session_id)? {
            None => return Ok(()),
            Some(p) => p,
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, b"")?;
        Ok(())
    }

    /// Remove the running marker. Best-effort — a missing file is not an
    /// error. Called automatically by [`Self::append`] when the session-ended
    /// record is persisted. No-op for ephemeral stores.
    pub fn clear_running(&self, session_id: &SessionId) -> Result<(), SessionStoreError> {
        let path = match self.running_marker_path(session_id)? {
            None => return Ok(()),
            Some(p) => p,
        };
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    /// Raw log text, or `None` when the store is ephemeral or the log does
    /// not exist yet.
    async fn read_raw(&self, session_id: &SessionId) -> Result<Option<String>, SessionStoreError> {
        let path = match self.path_for(session_id)? {
            None => return Ok(None),
            Some(p) => p,
        };
        match tokio::fs::read_to_string(&path).await {
            Ok(s) => Ok(Some(s)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Read the full session log. Records that fail to parse (corrupt
    /// trailing line from an aborted write, schema drift in an old log) are
    /// skipped with a `tracing::warn` rather than aborting — a partly
    /// replayable log is more useful than no log at all.
    pub async fn replay(&self, session_id: &SessionId) -> Result<Vec<E>, SessionStoreError> {
        let Some(raw) = self.read_raw(session_id).await? else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for (idx, line) in raw.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<E>(trimmed) {
                Ok(ev) => out.push(ev),
                Err(e) => tracing::warn!(
                    session_id = %session_id.as_str(),
                    line = idx + 1,
                    error = %e,
                    "skipping unparseable session log line",
                ),
            }
        }
        Ok(out)
    }

    /// Same records as [`Self::replay`], but as raw JSON objects rather than
    /// typed events — preserving the `"ts"` (and the terminal record's
    /// `"usage"`) fields [`Self::append`] injects at write time, which the
    /// event type's own `Deserialize` has no field for and would silently
    /// drop. Callers that hand replayed events to a frontend that stamps each
    /// event's display time from `ts` must use this, not `replay`; internal
    /// reconstruction (history injection, tool-index, crash-tail detection)
    /// doesn't care about wall-clock time and should keep using `replay`.
    pub async fn replay_json(
        &self,
        session_id: &SessionId,
    ) -> Result<Vec<serde_json::Value>, SessionStoreError> {
        let Some(raw) = self.read_raw(session_id).await? else {
            return Ok(Vec::new());
        };
        Ok(Self::parse_json_lines(session_id, &raw))
    }

    /// [`Self::replay_json`] resumed from a cursor, so a reader on a slow
    /// link fetches only what was appended since its last read rather than
    /// the whole log again.
    ///
    /// `since` is the `next` of a previous call ("" for the start), and is
    /// opaque to callers: a byte offset into the log plus a hash of the
    /// record that ends there. Only newline-terminated records are consumed,
    /// so `next` never lands inside a record the writer is still appending;
    /// that record comes back on the following call instead.
    ///
    /// The log is not strictly append-only — [`Self::truncate`] rewinds it —
    /// so an offset alone could land on a boundary of the rewritten log and
    /// silently skip the rewind. The hash catches that: a `since` whose
    /// record is no longer the one at that offset (rewound, replaced, or
    /// gone) replays from the start with `reset: true`, and the reader must
    /// discard what it holds rather than append.
    pub async fn replay_json_since(
        &self,
        session_id: &SessionId,
        since: &str,
    ) -> Result<JsonReplay, SessionStoreError> {
        let raw = self.read_raw(session_id).await?.unwrap_or_default();
        let complete = &raw[..raw.rfind('\n').map_or(0, |i| i + 1)];
        let start = resume_offset(complete, since);
        Ok(JsonReplay {
            events: Self::parse_json_lines(session_id, &complete[start.unwrap_or(0)..]),
            next: replay_cursor(complete),
            reset: start.is_none(),
        })
    }

    fn parse_json_lines(session_id: &SessionId, raw: &str) -> Vec<serde_json::Value> {
        let mut out = Vec::new();
        for (idx, line) in raw.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: serde_json::Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        session_id = %session_id.as_str(),
                        line = idx + 1,
                        error = %e,
                        "skipping unparseable session log line",
                    );
                    continue;
                }
            };
            // Validate against the typed shape so a corrupt/schema-drifted
            // line is skipped exactly as `replay` would skip it, instead of
            // handing the frontend an event kind it can't reduce.
            if let Err(e) = serde_json::from_value::<E>(value.clone()) {
                tracing::warn!(
                    session_id = %session_id.as_str(),
                    line = idx + 1,
                    error = %e,
                    "skipping unparseable session log line",
                );
                continue;
            }
            out.push(value);
        }
        out
    }

    /// Truncate the session log to keep only the first `keep_lines` non-empty
    /// lines. Used for R283 Rewind recovery — drops a failed turn and
    /// everything after it. No-op for an ephemeral store. Returns the number
    /// of lines kept.
    pub async fn truncate(
        &self,
        session_id: &SessionId,
        keep_lines: usize,
    ) -> Result<usize, SessionStoreError> {
        let Some(path) = self.path_for(session_id)? else {
            return Ok(0);
        };
        let Some(raw) = self.read_raw(session_id).await? else {
            return Ok(0);
        };

        let mut lines_kept = 0;
        let mut out = String::new();
        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                // Preserve blank lines in the output
                out.push('\n');
                continue;
            }
            if lines_kept < keep_lines {
                out.push_str(line);
                out.push('\n');
                lines_kept += 1;
            } else {
                break;
            }
        }

        // Write back the truncated content. Hold the write lock to ensure
        // atomicity with concurrent appends.
        let _guard = self.inner.write_lock.lock().await;
        tokio::fs::write(&path, &out).await?;

        Ok(lines_kept)
    }

    /// Ids of every session with a running marker on disk, paired with whether
    /// its log already holds the session-ended record.
    async fn marked_sessions(&self, context: &str) -> Vec<(String, bool)> {
        let Some(root) = self.inner.root.clone() else {
            return Vec::new();
        };
        let entries = match std::fs::read_dir(&root) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
            Err(e) => {
                tracing::warn!(error = %e, "{context}: failed to read sessions dir");
                return Vec::new();
            }
        };
        let mut out = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(true, |ext| ext != "terminated") {
                continue;
            }
            let Some(stem) = path.file_stem() else {
                continue;
            };
            let session_id_str = stem.to_string_lossy().into_owned();
            let sid = SessionId::new(&session_id_str);
            match self.replay(&sid).await {
                Ok(events) => {
                    let ended = events.iter().any(LogEvent::is_session_ended);
                    out.push((session_id_str, ended));
                }
                Err(e) => tracing::warn!(
                    session_id = %session_id_str,
                    error = %e,
                    "{context}: could not read JSONL; skipping"
                ),
            }
        }
        out
    }

    /// Scan for running markers whose log already contains the session-ended
    /// record (stale orphaned marker) and remove those markers. Returns the
    /// session ids whose stale markers were cleared.
    ///
    /// Sessions with a marker but *no* session-ended record were interrupted
    /// mid-run and are left untouched — they are candidates for resume and are
    /// discoverable via [`Self::find_interrupted_sessions`].
    pub async fn reconcile_aborted_sessions(&self) -> Vec<String> {
        let mut reconciled = Vec::new();
        for (session_id_str, already_ended) in self.marked_sessions("session reconciliation").await
        {
            if !already_ended {
                // Truly interrupted — leave the marker so
                // find_interrupted_sessions can surface this session.
                tracing::info!(
                    session_id = %session_id_str,
                    "cold-boot reconciliation: interrupted session kept for resume"
                );
                continue;
            }
            if let Err(e) = self.clear_running(&SessionId::new(&session_id_str)) {
                tracing::warn!(
                    session_id = %session_id_str,
                    error = %e,
                    "session reconciliation: could not clear stale .terminated marker"
                );
            }
            tracing::info!(
                session_id = %session_id_str,
                "cold-boot reconciliation: stale marker cleared"
            );
            reconciled.push(session_id_str);
        }
        reconciled
    }

    /// Sessions that were interrupted mid-run: a running marker present with
    /// no session-ended record in the log. Candidates for resume. Does not
    /// modify any files.
    pub async fn find_interrupted_sessions(&self) -> Vec<String> {
        self.marked_sessions("find_interrupted_sessions")
            .await
            .into_iter()
            .filter(|(_, ended)| !ended)
            .map(|(id, _)| id)
            .collect()
    }

    /// The session ids of every `<id>.jsonl` in the store directory (R588-F2).
    /// Empty for an ephemeral store or a missing directory.
    pub async fn list_session_ids(&self) -> Vec<SessionId> {
        let Some(root) = self.inner.root.clone() else {
            return Vec::new();
        };
        let entries = match std::fs::read_dir(&root) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
            Err(e) => {
                tracing::warn!(error = %e, "list_session_ids: failed to read sessions dir");
                return Vec::new();
            }
        };
        let mut ids = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(true, |ext| ext != "jsonl") {
                continue;
            }
            if let Some(stem) = path.file_stem() {
                ids.push(SessionId::new(stem.to_string_lossy().into_owned()));
            }
        }
        ids
    }

    /// Read just the session-started record, stopping after the first match.
    /// `None` when the file is absent or holds no such record.
    pub async fn read_session_started(
        &self,
        session_id: &SessionId,
    ) -> Result<Option<E>, SessionStoreError> {
        let Some(raw) = self.read_raw(session_id).await? else {
            return Ok(None);
        };
        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(ev) = serde_json::from_str::<E>(trimmed) {
                if ev.is_session_started() {
                    return Ok(Some(ev));
                }
            }
        }
        Ok(None)
    }

    /// Write the full assembled system prompt for `slot_id` to
    /// `<root>/<slot_id>.prompt.txt`, overwriting on each call so the file
    /// reflects the most recently used prompt. No-op for ephemeral stores.
    pub fn write_slot_prompt(&self, slot_id: &str, text: &str) -> Result<(), SessionStoreError> {
        let Some(root) = &self.inner.root else {
            return Ok(());
        };
        let safe = sanitize_id(slot_id)?;
        let path = root.join(format!("{safe}.prompt.txt"));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, text)?;
        Ok(())
    }

    /// R660-F16: write the doctrine text a `doctrineKey` names to
    /// `<root>/doctrine/<key>.md`, content-addressed and **write-once** — the
    /// key is a hash of the content, so a second write could only reproduce
    /// the same bytes. That makes this cheap to call on every session start
    /// (one `exists` check on the steady-state path) and makes the directory
    /// a complete, self-describing catalog of every prompt generation run:
    /// without it a `doctrineKey` is an opaque hash six months on.
    ///
    /// No-op for ephemeral stores.
    pub fn write_doctrine(&self, key: &str, text: &str) -> Result<(), SessionStoreError> {
        let Some(root) = &self.inner.root else {
            return Ok(());
        };
        let safe = sanitize_id(key)?;
        let dir = root.join("doctrine");
        let path = dir.join(format!("{safe}.md"));
        if path.exists() {
            return Ok(());
        }
        std::fs::create_dir_all(&dir)?;
        std::fs::write(&path, text)?;
        Ok(())
    }

    /// Jettison bulky `tool_result` bodies from session logs idle past
    /// [`JETTISON_GRACE_MS`] (R588-F4). Each large `result` payload is
    /// replaced in place with a `{"_jettisoned":true,"bytes":N}` stub; the
    /// `tool_call` metadata (tool name + args, incl. written file paths) and
    /// the `ok` / `toolCallId` fields are kept, so attribution and metadata
    /// search stay intact while the ephemeral response bodies stop bloating
    /// the log. Idempotent — already-stubbed results are skipped. Live /
    /// unreconciled sessions (a running marker present) and sessions still
    /// within the grace window are left alone. Returns the session ids whose
    /// logs were rewritten.
    pub async fn sweep_jettison_tool_bodies(&self, now_ms: u64) -> Vec<String> {
        let Some(root) = self.inner.root.clone() else {
            return Vec::new();
        };
        let entries = match std::fs::read_dir(&root) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
            Err(e) => {
                tracing::warn!(error = %e, dir = %root.display(), "jettison: read sessions dir failed");
                return Vec::new();
            }
        };

        let mut jettisoned = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(true, |ext| ext != "jsonl") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            // Skip live / unreconciled (resume-candidate) sessions.
            if root.join(format!("{stem}.terminated")).exists() {
                continue;
            }
            let Some(mtime_ms) = mtime_ms(&path) else {
                continue;
            };
            if now_ms < mtime_ms.saturating_add(JETTISON_GRACE_MS) {
                continue; // still within the live grace window
            }

            let raw = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(session_id = stem, error = %e, "jettison: read jsonl failed");
                    continue;
                }
            };
            let Some((new_text, stubbed, reclaimed)) = jettison_tool_bodies(&raw, JETTISON_MIN_BYTES)
            else {
                continue; // nothing bulky to shed (or already stubbed)
            };

            // Hold the write lock so we don't race a stray append. Preserve the
            // JSONL mtime afterward so this rewrite doesn't reset an expiry
            // clock a host's janitor reads off it.
            let _guard = self.inner.write_lock.lock().await;
            if let Err(e) = std::fs::write(&path, &new_text) {
                tracing::warn!(session_id = stem, error = %e, "jettison: write jsonl failed");
                continue;
            }
            let _ = set_mtime_ms(&path, mtime_ms);
            drop(_guard);

            tracing::info!(
                session_id = stem,
                stubbed,
                reclaimed_bytes = reclaimed,
                "janitor: jettisoned tool-result bodies"
            );
            jettisoned.push(stem.to_string());
        }
        jettisoned
    }
}

/// Every log this store writes is also readable through the core trait: a core
/// event lifts into `E` on append, and on read each record projects back down,
/// dropping the ones the core vocabulary has no variant for (a host's
/// camp-only events). For `E = AgentEvent` both conversions are the identity.
#[async_trait]
impl<E> SessionStore for FsSessionStore<E>
where
    E: LogEvent + From<AgentEvent>,
    AgentEvent: TryFrom<E>,
{
    fn path_for(&self, session_id: &SessionId) -> Result<Option<PathBuf>, SessionStoreError> {
        FsSessionStore::path_for(self, session_id)
    }

    async fn append(
        &self,
        session_id: &SessionId,
        event: &AgentEvent,
    ) -> Result<(), SessionStoreError> {
        let lifted = E::from(event.clone());
        FsSessionStore::append(self, session_id, &lifted).await
    }

    async fn read(&self, session_id: &SessionId) -> Result<Vec<AgentEvent>, SessionStoreError> {
        Ok(FsSessionStore::replay(self, session_id)
            .await?
            .into_iter()
            .filter_map(|ev| AgentEvent::try_from(ev).ok())
            .collect())
    }

    fn running_marker_path(
        &self,
        session_id: &SessionId,
    ) -> Result<Option<PathBuf>, SessionStoreError> {
        FsSessionStore::running_marker_path(self, session_id)
    }

    fn touch_running(&self, session_id: &SessionId) -> Result<(), SessionStoreError> {
        FsSessionStore::touch_running(self, session_id)
    }

    fn clear_running(&self, session_id: &SessionId) -> Result<(), SessionStoreError> {
        FsSessionStore::clear_running(self, session_id)
    }
}

/// A file's mtime as Unix-epoch milliseconds; `None` when unreadable or before
/// the epoch.
fn mtime_ms(path: &Path) -> Option<u64> {
    let modified = path.metadata().and_then(|m| m.modified()).ok()?;
    let since = modified.duration_since(std::time::UNIX_EPOCH).ok()?;
    Some(since.as_millis() as u64)
}

/// Best-effort set of a file's mtime (ms since epoch) via a second open +
/// `set_modified`. Used by [`FsSessionStore::sweep_jettison_tool_bodies`] so a
/// body-shedding rewrite doesn't reset an expiry clock.
fn set_mtime_ms(path: &Path, mtime_ms: u64) -> std::io::Result<()> {
    let when = std::time::UNIX_EPOCH + std::time::Duration::from_millis(mtime_ms);
    let f = std::fs::OpenOptions::new().write(true).open(path)?;
    f.set_modified(when)
}

/// Rewrite a session JSONL's text, replacing each `tool_result` event's bulky
/// `result` payload (serialized length > `min_bytes`) with a
/// `{"_jettisoned":true,"bytes":N}` stub. Every other event — and every
/// tool_result whose body is small or already stubbed — is passed through
/// verbatim (no reserialization, so key order is preserved). Returns
/// `Some((new_text, stubbed_count, bytes_reclaimed))` when at least one body
/// was stubbed, else `None` (so the caller can skip the write).
fn jettison_tool_bodies(raw: &str, min_bytes: usize) -> Option<(String, u32, u64)> {
    let mut out = String::with_capacity(raw.len());
    let mut stubbed = 0u32;
    let mut reclaimed = 0u64;
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            out.push('\n');
            continue;
        }
        let mut v = match serde_json::from_str::<serde_json::Value>(trimmed) {
            Ok(v) => v,
            Err(_) => {
                out.push_str(line);
                out.push('\n');
                continue;
            }
        };
        let is_tool_result = v.get("kind").and_then(|k| k.as_str()) == Some("tool_result");
        let already_stubbed = v
            .get("result")
            .and_then(|r| r.get("_jettisoned"))
            .and_then(|b| b.as_bool())
            == Some(true);
        if is_tool_result && !already_stubbed {
            let len = v
                .get("result")
                .map(|r| serde_json::to_string(r).map(|s| s.len()).unwrap_or(0))
                .unwrap_or(0);
            if len > min_bytes {
                if let Some(obj) = v.as_object_mut() {
                    obj.insert("result".into(), serde_json::json!({ "_jettisoned": true, "bytes": len }));
                }
                stubbed += 1;
                reclaimed += len as u64;
                out.push_str(&v.to_string());
                out.push('\n');
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    if stubbed == 0 {
        None
    } else {
        Some((out, stubbed, reclaimed))
    }
}

/// What [`FsSessionStore::replay_json_since`] read.
#[derive(Debug, Clone, PartialEq)]
pub struct JsonReplay {
    /// Records after `since`, oldest first — or every record when `reset`.
    pub events: Vec<serde_json::Value>,
    /// Pass back as the next call's `since`.
    pub next: String,
    /// `since` no longer names a record of this log; `events` is the whole
    /// log and replaces, rather than extends, what the reader holds.
    pub reset: bool,
}

/// The resume cursor after `consumed` (whole newline-terminated records):
/// `<byte offset>.<fnv-1a of the record ending there>`, or "" for none.
fn replay_cursor(consumed: &str) -> String {
    let Some(body) = consumed.strip_suffix('\n') else {
        return String::new();
    };
    let last = &body[body.rfind('\n').map_or(0, |i| i + 1)..];
    let hash = last
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325_u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3));
    format!("{}.{hash:016x}", consumed.len())
}

/// Where in `complete` a reader holding `since` resumes, or `None` when that
/// cursor does not name a record boundary of this log.
fn resume_offset(complete: &str, since: &str) -> Option<usize> {
    if since.is_empty() {
        return Some(0);
    }
    let offset: usize = since.split_once('.')?.0.parse().ok()?;
    let at_boundary =
        offset > 0 && offset <= complete.len() && complete.as_bytes()[offset - 1] == b'\n';
    (at_boundary && replay_cursor(&complete[..offset]) == since).then_some(offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TempDir;
    use serde::{Deserialize, Serialize};

    /// A host vocabulary for the generic store: a strict superset of the core
    /// turn events plus a terminal record and one host-only kind, serialized
    /// with the same `kind` tags. Stands in for yah's camp log.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum Ev {
        TurnStarted {
            #[serde(rename = "sessionId")]
            session_id: SessionId,
        },
        MessageDelta {
            #[serde(rename = "sessionId")]
            session_id: SessionId,
            text: String,
        },
        TurnEnded {
            #[serde(rename = "sessionId")]
            session_id: SessionId,
            text: String,
            #[serde(default, skip_serializing_if = "Option::is_none")]
            usage: Option<TurnUsage>,
        },
        ToolCall {
            #[serde(rename = "sessionId")]
            session_id: SessionId,
            #[serde(rename = "toolCallId")]
            tool_call_id: String,
            #[serde(rename = "toolName")]
            tool_name: String,
            args: serde_json::Value,
        },
        ToolResult {
            #[serde(rename = "sessionId")]
            session_id: SessionId,
            #[serde(rename = "toolCallId")]
            tool_call_id: String,
            ok: bool,
            result: serde_json::Value,
        },
        SessionEnded {
            #[serde(rename = "sessionId")]
            session_id: SessionId,
            #[serde(default, skip_serializing_if = "Option::is_none")]
            usage: Option<TurnUsage>,
        },
        HostOnly {
            #[serde(rename = "sessionId")]
            session_id: SessionId,
        },
    }

    impl LogEvent for Ev {
        fn turn_usage(&self) -> Option<&TurnUsage> {
            match self {
                Ev::TurnEnded { usage, .. } => usage.as_ref(),
                _ => None,
            }
        }
        fn is_session_started(&self) -> bool {
            false
        }
        fn is_session_ended(&self) -> bool {
            matches!(self, Ev::SessionEnded { .. })
        }
    }

    impl From<AgentEvent> for Ev {
        fn from(ev: AgentEvent) -> Self {
            match ev {
                AgentEvent::TurnStarted { session_id } => Ev::TurnStarted { session_id },
                AgentEvent::MessageDelta { session_id, text } => Ev::MessageDelta { session_id, text },
                AgentEvent::TurnEnded { session_id, text, usage, .. } => {
                    Ev::TurnEnded { session_id, text, usage }
                }
                other => panic!("test vocabulary does not lift {other:?}"),
            }
        }
    }

    impl TryFrom<Ev> for AgentEvent {
        type Error = Ev;
        fn try_from(ev: Ev) -> Result<Self, Ev> {
            match ev {
                Ev::TurnStarted { session_id } => Ok(AgentEvent::TurnStarted { session_id }),
                Ev::MessageDelta { session_id, text } => {
                    Ok(AgentEvent::MessageDelta { session_id, text })
                }
                Ev::TurnEnded { session_id, text, usage } => Ok(AgentEvent::TurnEnded {
                    session_id,
                    text,
                    stop_reason: None,
                    usage,
                    content_blocks: None,
                }),
                other => Err(other),
            }
        }
    }

    fn store(tmp: &TempDir) -> FsSessionStore<Ev> {
        FsSessionStore::new(tmp.path().join("sessions"))
    }

    fn started(id: &SessionId) -> Ev {
        Ev::TurnStarted { session_id: id.clone() }
    }

    fn ended(id: &SessionId) -> Ev {
        Ev::SessionEnded { session_id: id.clone(), usage: None }
    }

    fn delta(id: &SessionId, text: &str) -> Ev {
        Ev::MessageDelta { session_id: id.clone(), text: text.to_string() }
    }

    fn jsonl_mtime_ms(store: &FsSessionStore<Ev>, id: &SessionId) -> u64 {
        mtime_ms(&store.path_for(id).unwrap().unwrap()).unwrap()
    }

    /// A reader on a slow link resumes from its cursor and receives only new
    /// records; a rewind it did not see resets it instead of being appended
    /// onto the stale copy it holds.
    #[tokio::test]
    async fn replay_json_since_resumes_and_resets_on_a_rewind() {
        let tmp = TempDir::new();
        let store = store(&tmp);
        let id = SessionId::new("session:5e5e5e5e");
        store.append(&id, &started(&id)).await.unwrap();
        store.append(&id, &delta(&id, "a")).await.unwrap();

        let first = store.replay_json_since(&id, "").await.unwrap();
        assert_eq!((first.events.len(), first.reset), (2, false));

        store.append(&id, &delta(&id, "b")).await.unwrap();
        let path = store.path_for(&id).unwrap().unwrap();
        // A record the writer has not finished is not consumed yet.
        let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        std::io::Write::write_all(&mut f, b"{\"kind\":\"message_de").unwrap();
        let tail = store.replay_json_since(&id, &first.next).await.unwrap();
        assert_eq!((tail.events.len(), tail.reset), (1, false));
        assert_eq!(tail.events[0]["text"], "b");
        let caught_up = store.replay_json_since(&id, &tail.next).await.unwrap();
        assert_eq!((caught_up.events.len(), caught_up.reset, &caught_up.next), (0, false, &tail.next));

        // Rewind to one record, then grow back past the old cursor's offset.
        std::fs::write(&path, std::fs::read_to_string(&path).unwrap().lines().next().unwrap().to_owned() + "\n").unwrap();
        store.append(&id, &delta(&id, "c")).await.unwrap();
        store.append(&id, &delta(&id, "d")).await.unwrap();
        let after_rewind = store.replay_json_since(&id, &tail.next).await.unwrap();
        assert!(after_rewind.reset, "a rewound log must reset the reader");
        assert_eq!(after_rewind.events.len(), 3);
    }

    /// R660-F16: the doctrine catalog is content-addressed and write-once, so
    /// every session start can call it unconditionally. Also pins that the
    /// `doctrine/` subdirectory doesn't disturb the session walkers, which all
    /// filter on a file extension.
    #[tokio::test]
    async fn write_doctrine_is_write_once_and_survives_the_session_walkers() {
        let tmp = TempDir::new();
        let store = store(&tmp);
        let key = "0123456789abcdef0123456789abcdef";

        store.write_doctrine(key, "first write").unwrap();
        // A hash cannot address two different texts; a second write with the
        // same key can only be the same bytes, so it is a deliberate no-op.
        store.write_doctrine(key, "second write").unwrap();
        let path = tmp.path().join("sessions/doctrine").join(format!("{key}.md"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "first write");

        assert!(store.list_session_ids().await.is_empty(), "doctrine/ is not a session");
        assert!(store.find_interrupted_sessions().await.is_empty());

        // Traversal is refused, not silently rewritten into the base dir.
        assert!(store.write_doctrine("../escape", "nope").is_err());
    }

    #[tokio::test]
    async fn append_then_replay_round_trips_event_order() {
        let tmp = TempDir::new();
        let store = store(&tmp);
        let id = SessionId::new("session:abc12345");

        store.append(&id, &started(&id)).await.unwrap();
        store.append(&id, &delta(&id, "hello ")).await.unwrap();
        store.append(&id, &delta(&id, "world")).await.unwrap();
        store
            .append(&id, &Ev::TurnEnded { session_id: id.clone(), text: "hello world".into(), usage: None })
            .await
            .unwrap();

        let replay = store.replay(&id).await.unwrap();
        assert_eq!(replay.len(), 4);
        assert!(matches!(replay[0], Ev::TurnStarted { .. }));
        assert!(matches!(replay[3], Ev::TurnEnded { .. }));
    }

    #[tokio::test]
    async fn replay_returns_empty_when_log_does_not_exist() {
        let tmp = TempDir::new();
        let store = store(&tmp);
        let out = store.replay(&SessionId::new("session:00000000")).await.unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn replay_skips_unparseable_lines_without_aborting() {
        let tmp = TempDir::new();
        let store = store(&tmp);
        let id = SessionId::new("session:11111111");
        // Append one good event, then poke a corrupt line in by hand.
        store.append(&id, &started(&id)).await.unwrap();
        let path = store.path_for(&id).unwrap().unwrap();
        let mut existing = std::fs::read_to_string(&path).unwrap();
        existing.push_str("not json at all\n");
        existing.push_str(&serde_json::to_string(&ended(&id)).unwrap());
        existing.push('\n');
        std::fs::write(&path, existing).unwrap();

        let replay = store.replay(&id).await.unwrap();
        assert_eq!(replay.len(), 2);
        assert!(matches!(replay[0], Ev::TurnStarted { .. }));
        assert!(matches!(replay[1], Ev::SessionEnded { .. }));
    }

    #[tokio::test]
    async fn unsafe_session_ids_refuse_path_resolution() {
        let tmp = TempDir::new();
        let store = store(&tmp);
        for bad in ["..", "../escape", "a/b", "a\\b", "with\0null", ""] {
            let id = SessionId::new(bad);
            assert!(matches!(store.path_for(&id), Err(SessionStoreError::UnsafeId(_))));
        }
    }

    #[tokio::test]
    async fn touch_running_creates_terminated_marker() {
        let tmp = TempDir::new();
        let store = store(&tmp);
        let id = SessionId::new("session:aabbccdd");
        let marker = store.running_marker_path(&id).unwrap().unwrap();
        assert!(!marker.exists(), "marker should not exist before touch");
        store.touch_running(&id).unwrap();
        assert!(marker.exists(), "marker should exist after touch");
    }

    #[tokio::test]
    async fn clear_running_removes_marker_and_is_idempotent() {
        let tmp = TempDir::new();
        let store = store(&tmp);
        let id = SessionId::new("session:11223344");
        store.touch_running(&id).unwrap();
        let marker = store.running_marker_path(&id).unwrap().unwrap();
        assert!(marker.exists());
        store.clear_running(&id).unwrap();
        assert!(!marker.exists(), "marker should be gone after clear");
        // Clearing again must not error.
        store.clear_running(&id).unwrap();
    }

    #[tokio::test]
    async fn append_session_ended_clears_running_marker() {
        let tmp = TempDir::new();
        let store = store(&tmp);
        let id = SessionId::new("session:deadd00d");
        store.touch_running(&id).unwrap();
        let marker = store.running_marker_path(&id).unwrap().unwrap();
        assert!(marker.exists());
        // A non-terminal event must NOT clear the marker.
        store.append(&id, &started(&id)).await.unwrap();
        assert!(marker.exists(), "non-terminal event must not clear marker");
        // The terminal record must clear it.
        store.append(&id, &ended(&id)).await.unwrap();
        assert!(!marker.exists(), "marker must be cleared after SessionEnded append");
    }

    #[tokio::test]
    async fn reconcile_skips_truly_interrupted_sessions() {
        let tmp = TempDir::new();
        let store = store(&tmp);
        let id = SessionId::new("session:reconcile1");

        // Interrupted session: marker present, no terminal record.
        store.touch_running(&id).unwrap();
        store.append(&id, &started(&id)).await.unwrap();
        let marker = store.running_marker_path(&id).unwrap().unwrap();
        assert!(marker.exists(), "marker must exist before reconcile");

        let reconciled = store.reconcile_aborted_sessions().await;
        assert!(reconciled.is_empty(), "interrupted session must not be in reconciled list");
        assert!(marker.exists(), "marker must still exist; session is resumable");

        let replay = store.replay(&id).await.unwrap();
        assert!(
            !replay.iter().any(|e| matches!(e, Ev::SessionEnded { .. })),
            "no synthetic SessionEnded should be appended"
        );
    }

    #[tokio::test]
    async fn find_interrupted_sessions_returns_open_sessions() {
        let tmp = TempDir::new();
        let store = store(&tmp);
        let id = SessionId::new("session:interrupted1");
        store.touch_running(&id).unwrap();
        store.append(&id, &started(&id)).await.unwrap();
        assert_eq!(store.find_interrupted_sessions().await, vec!["session:interrupted1"]);
    }

    #[tokio::test]
    async fn find_interrupted_sessions_ignores_ended_sessions() {
        let tmp = TempDir::new();
        let store = store(&tmp);
        let id = SessionId::new("session:ended1");
        // Ended cleanly — the marker is cleared by append.
        store.touch_running(&id).unwrap();
        store.append(&id, &ended(&id)).await.unwrap();
        assert!(
            store.find_interrupted_sessions().await.is_empty(),
            "ended session must not appear as interrupted"
        );
    }

    #[tokio::test]
    async fn reconcile_already_ended_clears_stale_marker_only() {
        let tmp = TempDir::new();
        let store = store(&tmp);
        let id = SessionId::new("session:reconcile2");

        store.touch_running(&id).unwrap();
        let marker = store.running_marker_path(&id).unwrap().unwrap();
        store.append(&id, &ended(&id)).await.unwrap();
        // The append cleared the marker; recreate it to simulate a stale orphan.
        std::fs::write(&marker, b"").unwrap();
        assert!(marker.exists(), "stale marker present");

        let reconciled = store.reconcile_aborted_sessions().await;
        assert_eq!(reconciled, vec!["session:reconcile2"]);
        assert!(!marker.exists(), "stale marker cleared");

        let replay = store.replay(&id).await.unwrap();
        let ended_count = replay.iter().filter(|e| matches!(e, Ev::SessionEnded { .. })).count();
        assert_eq!(ended_count, 1, "no duplicate SessionEnded appended");
    }

    #[tokio::test]
    async fn reconcile_empty_dir_returns_empty_vec() {
        let tmp = TempDir::new();
        let store = store(&tmp);
        // No sessions directory exists yet.
        assert!(store.reconcile_aborted_sessions().await.is_empty());
    }

    #[tokio::test]
    async fn reconcile_skips_non_terminated_files() {
        let tmp = TempDir::new();
        let store = store(&tmp);
        let id = SessionId::new("session:skipped1");
        // Only a JSONL (no .terminated marker).
        store.append(&id, &ended(&id)).await.unwrap();
        let reconciled = store.reconcile_aborted_sessions().await;
        assert!(reconciled.is_empty(), "no .terminated file → nothing to reconcile");
    }

    // ─── Tool-body jettison (R588-F4) ────────────────────────────────────

    #[test]
    fn jettison_stubs_bulky_result_keeps_small_and_metadata() {
        let big = "x".repeat(2000);
        let raw = format!(
            "{}\n{}\n{}\n{}\n",
            r#"{"kind":"session_started","sessionId":"s","ticketId":"R1","engine":"e","cacheKey":"k","estimatedTokens":0,"ringDepth":0.0}"#,
            r#"{"kind":"tool_call","sessionId":"s","toolCallId":"c1","toolName":"Read","args":{"path":"src/a.rs"}}"#,
            format!(r#"{{"kind":"tool_result","sessionId":"s","toolCallId":"c1","ok":true,"result":"{big}"}}"#),
            r#"{"kind":"tool_result","sessionId":"s","toolCallId":"c2","ok":true,"result":"small"}"#,
        );
        let (out, stubbed, reclaimed) = jettison_tool_bodies(&raw, JETTISON_MIN_BYTES).unwrap();
        assert_eq!(stubbed, 1, "only the bulky result is stubbed");
        assert!(reclaimed > 2000);
        // tool_call metadata + args survive verbatim.
        assert!(out.contains(r#""toolName":"Read""#));
        assert!(out.contains(r#""path":"src/a.rs""#));
        // bulky body replaced with a stub; the raw blob is gone.
        assert!(out.contains(r#""_jettisoned":true"#));
        assert!(!out.contains(&big));
        // small result kept untouched.
        assert!(out.contains(r#""result":"small""#));
    }

    #[test]
    fn jettison_is_idempotent_and_noop_when_nothing_bulky() {
        let small = format!(
            "{}\n",
            r#"{"kind":"tool_result","sessionId":"s","toolCallId":"c1","ok":true,"result":"tiny"}"#,
        );
        assert!(jettison_tool_bodies(&small, JETTISON_MIN_BYTES).is_none());

        let big = "y".repeat(2000);
        let raw = format!(
            "{}\n",
            format!(r#"{{"kind":"tool_result","sessionId":"s","toolCallId":"c1","ok":true,"result":"{big}"}}"#),
        );
        let (once, _, _) = jettison_tool_bodies(&raw, JETTISON_MIN_BYTES).unwrap();
        // Re-running over already-stubbed content is a no-op.
        assert!(jettison_tool_bodies(&once, JETTISON_MIN_BYTES).is_none());
    }

    #[tokio::test]
    async fn sweep_jettison_stubs_bulky_bodies_after_grace() {
        let tmp = TempDir::new();
        let store = store(&tmp);
        let id = SessionId::new("session:jett1");
        store
            .append(
                &id,
                &Ev::ToolCall {
                    session_id: id.clone(),
                    tool_call_id: "c1".into(),
                    tool_name: "Read".into(),
                    args: serde_json::json!({"path": "src/a.rs"}),
                },
            )
            .await
            .unwrap();
        store
            .append(
                &id,
                &Ev::ToolResult {
                    session_id: id.clone(),
                    tool_call_id: "c1".into(),
                    ok: true,
                    result: serde_json::json!("z".repeat(3000)),
                },
            )
            .await
            .unwrap();

        let mtime = jsonl_mtime_ms(&store, &id);
        // now = mtime + 2d → past the 1-day grace window.
        let jettisoned = store.sweep_jettison_tool_bodies(mtime + 2 * MS_PER_DAY).await;
        assert_eq!(jettisoned, vec!["session:jett1".to_string()]);

        let events = store.replay(&id).await.unwrap();
        // tool_call survives; tool_result body is stubbed.
        assert!(events.iter().any(|e| matches!(e, Ev::ToolCall { .. })));
        let stub = events
            .iter()
            .find_map(|e| match e {
                Ev::ToolResult { result, .. } => Some(result),
                _ => None,
            })
            .unwrap();
        assert_eq!(stub.get("_jettisoned").and_then(|b| b.as_bool()), Some(true));
    }

    #[tokio::test]
    async fn sweep_jettison_respects_grace_and_terminated() {
        let tmp = TempDir::new();
        let store = store(&tmp);
        let id = SessionId::new("session:jett2");
        store
            .append(
                &id,
                &Ev::ToolResult {
                    session_id: id.clone(),
                    tool_call_id: "c1".into(),
                    ok: true,
                    result: serde_json::json!("w".repeat(3000)),
                },
            )
            .await
            .unwrap();
        let mtime = jsonl_mtime_ms(&store, &id);

        // Within grace → untouched.
        assert!(store.sweep_jettison_tool_bodies(mtime).await.is_empty());

        // Past grace but a running marker present (live/resume-candidate) → skipped.
        store.touch_running(&id).unwrap();
        assert!(store.sweep_jettison_tool_bodies(mtime + 2 * MS_PER_DAY).await.is_empty());
    }

    // ─── Core trait over a host vocabulary (R950-F4) ─────────────────────

    /// Through `dyn SessionStore` a host-vocabulary log reads back as core
    /// events only: host-only records stay on disk for the host's own
    /// `replay`, and the core reader never sees them.
    #[tokio::test]
    async fn core_trait_reads_drop_host_only_records() {
        let tmp = TempDir::new();
        let store = store(&tmp);
        let id = SessionId::new("session:core1");
        let dyn_store: &dyn SessionStore = &store;

        dyn_store.append(&id, &AgentEvent::TurnStarted { session_id: id.clone() }).await.unwrap();
        store.append(&id, &Ev::HostOnly { session_id: id.clone() }).await.unwrap();
        dyn_store
            .append(&id, &AgentEvent::MessageDelta { session_id: id.clone(), text: "hi".into() })
            .await
            .unwrap();

        assert_eq!(store.replay(&id).await.unwrap().len(), 3, "host replay sees the superset");
        let core = dyn_store.read(&id).await.unwrap();
        assert_eq!(core.len(), 2);
        assert!(matches!(core[0], AgentEvent::TurnStarted { .. }));
        assert!(matches!(core[1], AgentEvent::MessageDelta { .. }));
        assert_eq!(dyn_store.path_for(&id).unwrap(), store.path_for(&id).unwrap());
    }

    /// The store over the core vocabulary itself: both conversions are the
    /// identity, and turn usage accumulates exactly as for a host vocabulary.
    #[tokio::test]
    async fn core_vocabulary_store_round_trips_through_the_trait() {
        let tmp = TempDir::new();
        let store: FsSessionStore<AgentEvent> = FsSessionStore::new(tmp.path());
        let id = SessionId::new("session:core2");
        let dyn_store: &dyn SessionStore = &store;

        dyn_store.touch_running(&id).unwrap();
        assert!(dyn_store.running_marker_path(&id).unwrap().unwrap().exists());
        let usage = TurnUsage { input_tokens: Some(7), ..Default::default() };
        dyn_store
            .append(
                &id,
                &AgentEvent::TurnEnded {
                    session_id: id.clone(),
                    text: "done".into(),
                    stop_reason: None,
                    usage: Some(usage.clone()),
                    content_blocks: None,
                },
            )
            .await
            .unwrap();
        let read = dyn_store.read(&id).await.unwrap();
        assert!(matches!(&read[..], [AgentEvent::TurnEnded { usage: Some(u), .. }] if *u == usage));
        // No terminal event in the core vocabulary: the host clears the marker.
        assert!(dyn_store.running_marker_path(&id).unwrap().unwrap().exists());
        dyn_store.clear_running(&id).unwrap();
        assert!(!dyn_store.running_marker_path(&id).unwrap().unwrap().exists());
    }

    #[tokio::test]
    async fn ephemeral_ring_caps_and_writes_nothing() {
        let ring = Arc::new(std::sync::Mutex::new(VecDeque::new()));
        let store: FsSessionStore<Ev> = FsSessionStore::ephemeral_with_ring(ring.clone());
        let id = SessionId::new("session:ring");
        assert!(store.is_ephemeral());
        assert!(store.path_for(&id).unwrap().is_none());
        for i in 0..EPHEMERAL_RING_CAP + 5 {
            store.append(&id, &delta(&id, &i.to_string())).await.unwrap();
        }
        let guard = ring.lock().unwrap();
        assert_eq!(guard.len(), EPHEMERAL_RING_CAP);
        assert!(matches!(&guard[0], Ev::MessageDelta { text, .. } if text == "5"));
        drop(guard);
        assert!(store.replay(&id).await.unwrap().is_empty());
    }
}
