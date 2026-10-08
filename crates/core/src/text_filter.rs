//! Streaming text filter for `MessageDelta` events.
//!
//! Two transforms applied before text reaches the sink:
//!
//! 1. **Path normalization** — `path/to/file.rs#L42` → `path/to/file.rs:42`
//!    and `[label](path/to/file.rs#L42)` → `path/to/file.rs:42`. Models
//!    frequently produce GitHub `#L42` anchors and markdown links; the yah UI
//!    expects bare `path:line` chips.
//!
//! 2. **Colon-before-tool-call stripping** — when a `tool_use` content block
//!    starts immediately after a text block, the model often ends with a
//!    sentence like "Let me read the file:" that the system prompt forbids.
//!    `on_tool_use_start` strips the trailing colon-sentence from the tail
//!    buffer so it never reaches the sink.
//!
//! The filter holds a tail buffer (`TAIL_WINDOW` bytes) so it can act on the
//! end of the text stream when a tool call arrives. Text before the window is
//! flushed eagerly so streaming latency is bounded.

/// Tail buffer size. The colon-sentence pattern is at most ~200 chars; 512
/// gives headroom without adding noticeable streaming delay.
const TAIL_WINDOW: usize = 512;

/// Streaming text filter. One instance per turn; reset between turns via
/// `flush()` at `content_block_stop` or turn end.
pub struct TextStreamFilter {
    tail: String,
}

impl Default for TextStreamFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl TextStreamFilter {
    pub fn new() -> Self {
        Self {
            tail: String::with_capacity(TAIL_WINDOW + 64),
        }
    }

    /// Process a raw text delta. Applies path normalization, appends to the
    /// tail buffer, and returns any text that spills past the window (ready
    /// to emit immediately).
    pub fn push_text(&mut self, raw: &str) -> String {
        let normalized = normalize_paths(raw);
        self.tail.push_str(&normalized);

        if self.tail.len() > TAIL_WINDOW {
            let spill = self.tail.len() - TAIL_WINDOW;
            // Respect char boundaries.
            let split_at = self.tail.floor_char_boundary(spill);
            let to_emit = self.tail[..split_at].to_string();
            self.tail.drain(..split_at);
            to_emit
        } else {
            String::new()
        }
    }

    /// Called when a `tool_use` content block starts. Strips any trailing
    /// "sentence:" colon pattern from the tail (the colon-before-tool-call
    /// anti-pattern) and returns whatever remains to emit.
    pub fn on_tool_use_start(&mut self) -> String {
        strip_colon_tail(&mut self.tail);
        self.flush()
    }

    /// Flush the tail buffer as-is (text block ended normally, no tool call
    /// following). Returns the full tail; the filter is empty afterward.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.tail)
    }
}

// ── Colon stripping ──────────────────────────────────────────────────────────

/// Remove a trailing "sentence ending in colon" from `s`.
///
/// Only fires when the text literally ends with `:` (after trimming trailing
/// whitespace). Skips code fences, indented blocks, and list items so we
/// don't silently eat intentional colons.
fn strip_colon_tail(s: &mut String) {
    let trimmed = s.trim_end();
    if !trimmed.ends_with(':') {
        return;
    }
    let last_nl = trimmed.rfind('\n').map(|i| i + 1).unwrap_or(0);
    let sentence = &trimmed[last_nl..];
    // Leave colons that are part of code blocks, indented text, or list items.
    if sentence.starts_with("```")
        || sentence.starts_with("    ")
        || sentence.starts_with("- ")
        || sentence.starts_with("* ")
        || sentence.starts_with("+ ")
        || sentence.len() < 3
    {
        return;
    }
    s.truncate(last_nl);
    while s.ends_with(|c: char| c.is_whitespace()) {
        s.pop();
    }
}

// ── Path normalization ───────────────────────────────────────────────────────

/// Normalize file-path references in a text chunk.
///
/// - `[label](path/to/file.rs#L42)` → `path/to/file.rs:42`
/// - `path/to/file.rs#L42`          → `path/to/file.rs:42`
///
/// Applied eagerly to each incoming delta. Patterns that span a chunk
/// boundary are handled naturally by the tail buffer (both passes look for
/// complete tokens, so a split `#L` would just pass through unchanged until
/// the next push flushes it).
pub fn normalize_paths(text: &str) -> String {
    let s = strip_markdown_link_wrappers(text);
    normalize_hash_line_anchors(&s)
}

/// `[label](path/to/file.rs#L42)` → `path/to/file.rs:42`
fn strip_markdown_link_wrappers(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(open) = rest.find('[') {
        let after_open = &rest[open + 1..];
        if let Some(rel_bracket) = after_open.find("](") {
            let close_bracket = open + 1 + rel_bracket; // index of ']' in `rest`
            let url_start = close_bracket + 2; // index of char after '('
            if let Some(rel_close) = rest[url_start..].find(')') {
                let url = &rest[url_start..url_start + rel_close];
                if let Some(chip) = extract_path_line(url) {
                    out.push_str(&rest[..open]);
                    out.push_str(&chip);
                    rest = &rest[url_start + rel_close + 1..];
                    continue;
                }
            }
        }
        // Not a file-path link — advance past '[' and keep scanning.
        out.push_str(&rest[..open + 1]);
        rest = &rest[open + 1..];
    }

    out.push_str(rest);
    out
}

/// `path/to/file.rs#L42` → `path/to/file.rs:42` (bare anchor form).
fn normalize_hash_line_anchors(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(pos) = rest.find("#L") {
        let after = &rest[pos + 2..];
        let digits_end = after
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(after.len());

        if digits_end == 0 {
            out.push_str(&rest[..pos + 2]);
            rest = after;
            continue;
        }

        let digits = &after[..digits_end];
        let before = &rest[..pos]; // everything before '#'

        // Count path-like chars working backwards from '#'.
        let path_char_count = before
            .bytes()
            .rev()
            .take_while(|&b| {
                b.is_ascii_alphanumeric()
                    || b == b'/'
                    || b == b'.'
                    || b == b'_'
                    || b == b'-'
            })
            .count();

        if path_char_count > 0 {
            let path_start = before.len() - path_char_count;
            let path = &before[path_start..];
            if path.contains('/') || path.contains('.') {
                out.push_str(&before[..path_start]);
                out.push_str(path);
                out.push(':');
                out.push_str(digits);
                rest = &after[digits_end..];
                continue;
            }
        }

        // Not a file path; skip past `#L`.
        out.push_str(&rest[..pos + 2]);
        rest = after;
    }

    out.push_str(rest);
    out
}

/// Extract `"path:line"` from a URL of the form `path/to/file.rs#L42`.
/// Also handles `yah://file/path.rs#L42` by stripping the scheme prefix.
fn extract_path_line(url: &str) -> Option<String> {
    let url = url
        .trim()
        .trim_start_matches("yah://file/")
        .trim_start_matches("yah://arch/");
    let hash_pos = url.find("#L")?;
    let path = &url[..hash_pos];
    let after = &url[hash_pos + 2..];
    let line_len = after
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after.len());
    if line_len == 0 || (!path.contains('/') && !path.contains('.')) {
        return None;
    }
    Some(format!("{}:{}", path, &after[..line_len]))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Path normalization

    #[test]
    fn normalizes_bare_hash_anchor() {
        assert_eq!(
            normalize_paths("see crates/yah/runner/src/lib.rs#L42"),
            "see crates/yah/runner/src/lib.rs:42"
        );
    }

    #[test]
    fn normalizes_markdown_path_link() {
        assert_eq!(
            normalize_paths("[lib.rs](crates/yah/runner/src/lib.rs#L42)"),
            "crates/yah/runner/src/lib.rs:42"
        );
    }

    #[test]
    fn normalizes_yah_scheme_link() {
        assert_eq!(
            normalize_paths("[foo](yah://file/crates/yah/runner/src/lib.rs#L10)"),
            "crates/yah/runner/src/lib.rs:10"
        );
    }

    #[test]
    fn does_not_touch_plain_urls() {
        let text = "see https://example.com/path for details";
        assert_eq!(normalize_paths(text), text);
    }

    #[test]
    fn does_not_touch_plain_hash() {
        let text = "color: #FF0000";
        assert_eq!(normalize_paths(text), text);
    }

    #[test]
    fn normalizes_bare_anchor_only_with_preceding_path() {
        // "#L42" alone (no preceding path) must not be touched.
        let text = "check #L42 for the spec";
        assert_eq!(normalize_paths(text), text);
    }

    // Colon stripping

    #[test]
    fn strips_colon_tail_on_tool_use() {
        let mut f = TextStreamFilter::new();
        let eager = f.push_text("Let me read the file:");
        assert!(eager.is_empty()); // held in tail
        let tail = f.on_tool_use_start();
        assert!(
            tail.is_empty(),
            "colon-sentence should be stripped, got: {:?}",
            tail
        );
    }

    #[test]
    fn preserves_non_colon_tail_on_tool_use() {
        let mut f = TextStreamFilter::new();
        let _ = f.push_text("Here is what I found.");
        let tail = f.on_tool_use_start();
        assert_eq!(tail.trim(), "Here is what I found.");
    }

    #[test]
    fn flush_emits_colon_tail_unchanged() {
        // Colon stripping only on tool_use_start, not on normal flush.
        let mut f = TextStreamFilter::new();
        let _ = f.push_text("The answer is:");
        let tail = f.flush();
        assert!(tail.contains("The answer is:"));
    }

    #[test]
    fn does_not_strip_code_fence_colon() {
        let mut f = TextStreamFilter::new();
        let _ = f.push_text("example:\n```");
        let tail = f.on_tool_use_start();
        // "```" line is a code fence — should NOT be stripped.
        assert!(tail.contains("```"), "code fence should not be stripped");
    }

    #[test]
    fn does_not_strip_list_item_colon() {
        let mut f = TextStreamFilter::new();
        let _ = f.push_text("items:\n- foo:");
        let tail = f.on_tool_use_start();
        // "- foo:" is a list item — should NOT be stripped.
        assert!(tail.contains("- foo:"), "list item should not be stripped");
    }

    #[test]
    fn spill_flushes_eagerly_beyond_window() {
        let mut f = TextStreamFilter::new();
        // Push enough text to exceed TAIL_WINDOW.
        let big = "x".repeat(TAIL_WINDOW + 100);
        let eager = f.push_text(&big);
        assert!(
            !eager.is_empty(),
            "overflow beyond TAIL_WINDOW should emit eagerly"
        );
        assert_eq!(f.tail.len(), TAIL_WINDOW);
    }

    #[test]
    fn strips_sentence_before_preceding_text() {
        let mut f = TextStreamFilter::new();
        let _ = f.push_text("Here is my plan.\n\nLet me check the code:");
        let tail = f.on_tool_use_start();
        assert!(
            tail.contains("Here is my plan."),
            "preceding text should be kept"
        );
        assert!(
            !tail.contains("Let me check"),
            "colon-sentence should be stripped"
        );
    }
}
