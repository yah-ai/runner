//! Rate-limit sample storage: the [`RateLimitSample`] record, the
//! [`UsageStore`] trait, and (off wasm32) [`FsUsageStore`], a JSONL-per-identity
//! filesystem default rooted at a caller-supplied directory.
//!
//! Parsing samples off a provider's response headers is the host's job (it
//! owns the HTTP client); this module stores and reads what the host parsed.
//!
//! Per-sample shape (JSONL, one line per turn, internal tag `k`):
//! - OAuth:  `{"k":"o","t":<unix>,"h5":<f32>,"d7":<f32>,"h5r":<u32>,"d7r":<u32>}`
//! - ApiKey: `{"k":"a","t":<unix>,"r":<f32>,"tk":<f32>,"it":<f32>,"ot":<f32>,"rr":<u32>,"tr":<u32>}`
//!
//! File layout: `<dir>/<identity_fingerprint>.jsonl`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Drop samples older than this when reading. Keeps the on-read view bounded
/// while leaving the file itself untouched (best-effort: a compaction pass can
/// prune later if size ever bites).
pub const RETENTION_SECS: i64 = 30 * 24 * 60 * 60;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "k")]
pub enum RateLimitSample {
    /// Subscription schema: one utilization fraction per 5h / 7d window.
    #[serde(rename = "o")]
    Oauth {
        #[serde(rename = "t")]
        ts_unix: i64,
        #[serde(rename = "h5")]
        h5_util: f32,
        #[serde(rename = "d7")]
        d7_util: f32,
        #[serde(rename = "h5r")]
        h5_reset: u32,
        #[serde(rename = "d7r")]
        d7_reset: u32,
    },
    /// API-key schema: per-resource utilization (`1 - remaining/limit`).
    #[serde(rename = "a")]
    ApiKey {
        #[serde(rename = "t")]
        ts_unix: i64,
        #[serde(rename = "r")]
        requests_util: f32,
        #[serde(rename = "tk")]
        tokens_util: f32,
        #[serde(rename = "it")]
        input_util: f32,
        #[serde(rename = "ot")]
        output_util: f32,
        #[serde(rename = "rr")]
        requests_reset: u32,
        #[serde(rename = "tr")]
        tokens_reset: u32,
    },
}

impl RateLimitSample {
    pub fn ts_unix(&self) -> i64 {
        match self {
            Self::Oauth { ts_unix, .. } | Self::ApiKey { ts_unix, .. } => *ts_unix,
        }
    }
}

/// One identity's sample stream, as [`UsageStore::list_identities`] reports it
/// so a UI can render an identity picker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdentitySummary {
    /// 12-hex fingerprint (the filename without `.jsonl`).
    pub fingerprint: String,
    /// `"oauth"` or `"apikey"` — sourced from the first sample's tag.
    /// `None` if the file is empty or carries no parseable samples.
    pub kind: Option<String>,
    /// Newest sample timestamp (unix seconds). Used for "most recent
    /// identity" tie-breaking when no identity is explicitly selected.
    pub last_ts: i64,
    /// Total sample count after the [`RETENTION_SECS`] filter.
    pub samples: usize,
}

/// Where rate-limit samples are kept, one stream per identity fingerprint.
/// Telemetry must never fail an in-flight turn, so nothing here returns an
/// error: write failures are logged and dropped, read failures read as empty.
#[async_trait]
pub trait UsageStore: Send + Sync {
    /// Append one sample to `fingerprint`'s stream.
    async fn append(&self, fingerprint: &str, sample: &RateLimitSample);

    /// `fingerprint`'s samples from the last [`RETENTION_SECS`] before
    /// `now_unix`, oldest first. Malformed records are skipped.
    async fn load(&self, fingerprint: &str, now_unix: i64) -> Vec<RateLimitSample>;

    /// Every identity with at least one retained sample, most recent first.
    async fn list_identities(&self, now_unix: i64) -> Vec<IdentitySummary>;
}

#[cfg(not(target_arch = "wasm32"))]
pub use fs::FsUsageStore;

#[cfg(not(target_arch = "wasm32"))]
mod fs {
    use std::path::{Path, PathBuf};

    use async_trait::async_trait;
    use tokio::io::AsyncWriteExt;

    use super::{IdentitySummary, RateLimitSample, UsageStore, RETENTION_SECS};

    /// `<dir>/<fingerprint>.jsonl`, one line per sample.
    #[derive(Debug, Clone)]
    pub struct FsUsageStore {
        dir: PathBuf,
    }

    impl FsUsageStore {
        /// A store rooted at `dir`. The directory is created on first append.
        pub fn new(dir: impl Into<PathBuf>) -> Self {
            Self { dir: dir.into() }
        }

        pub fn dir(&self) -> &Path {
            &self.dir
        }
    }

    #[async_trait]
    impl UsageStore for FsUsageStore {
        async fn append(&self, fingerprint: &str, sample: &RateLimitSample) {
            let dir = &self.dir;
            if let Err(e) = tokio::fs::create_dir_all(dir).await {
                tracing::warn!(error = %e, dir = %dir.display(), "usage_log: mkdir failed");
                return;
            }
            let path = dir.join(format!("{fingerprint}.jsonl"));
            let line = match serde_json::to_string(sample) {
                Ok(mut s) => {
                    s.push('\n');
                    s
                }
                Err(e) => {
                    tracing::warn!(error = %e, "usage_log: serialize failed");
                    return;
                }
            };
            let file = tokio::fs::OpenOptions::new().create(true).append(true).open(&path).await;
            match file {
                Ok(mut f) => {
                    if let Err(e) = f.write_all(line.as_bytes()).await {
                        tracing::warn!(error = %e, path = %path.display(), "usage_log: write failed");
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, path = %path.display(), "usage_log: open failed");
                }
            }
        }

        async fn load(&self, fingerprint: &str, now_unix: i64) -> Vec<RateLimitSample> {
            let path = self.dir.join(format!("{fingerprint}.jsonl"));
            let body = match tokio::fs::read_to_string(&path).await {
                Ok(s) => s,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
                Err(e) => {
                    tracing::warn!(error = %e, path = %path.display(), "usage_log: read failed");
                    return Vec::new();
                }
            };
            let cutoff = now_unix - RETENTION_SECS;
            body.lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .filter_map(|line| serde_json::from_str::<RateLimitSample>(line).ok())
                .filter(|s| s.ts_unix() >= cutoff)
                .collect()
        }

        async fn list_identities(&self, now_unix: i64) -> Vec<IdentitySummary> {
            let mut rd = match tokio::fs::read_dir(&self.dir).await {
                Ok(r) => r,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
                Err(e) => {
                    tracing::warn!(error = %e, dir = %self.dir.display(), "usage_log: readdir failed");
                    return Vec::new();
                }
            };
            let mut out = Vec::new();
            while let Ok(Some(entry)) = rd.next_entry().await {
                let path = entry.path();
                let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                    continue;
                };
                if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                    continue;
                }
                let samples = self.load(stem, now_unix).await;
                if samples.is_empty() {
                    continue;
                }
                let last_ts = samples.last().map(|s| s.ts_unix()).unwrap_or(0);
                let kind = samples.first().map(|s| match s {
                    RateLimitSample::Oauth { .. } => "oauth".to_string(),
                    RateLimitSample::ApiKey { .. } => "apikey".to_string(),
                });
                out.push(IdentitySummary {
                    fingerprint: stem.to_string(),
                    kind,
                    last_ts,
                    samples: samples.len(),
                });
            }
            out.sort_by(|a, b| b.last_ts.cmp(&a.last_ts));
            out
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::test_support::TempDir;

        #[tokio::test]
        async fn append_creates_dir_and_writes_line() {
            let tmp = TempDir::new();
            let store = FsUsageStore::new(tmp.path());
            let sample = RateLimitSample::Oauth {
                ts_unix: 1747800000,
                h5_util: 0.42,
                d7_util: 0.18,
                h5_reset: 1747818000,
                d7_reset: 1748404800,
            };
            store.append("deadbeefcafe", &sample).await;
            store.append("deadbeefcafe", &sample).await;

            let path = tmp.path().join("deadbeefcafe.jsonl");
            let body = tokio::fs::read_to_string(&path).await.unwrap();
            let lines: Vec<&str> = body.lines().collect();
            assert_eq!(lines.len(), 2);
            let parsed: RateLimitSample = serde_json::from_str(lines[0]).unwrap();
            assert_eq!(parsed, sample);
        }

        #[tokio::test]
        async fn append_creates_nested_dir() {
            let tmp = TempDir::new();
            let nested = tmp.path().join("does/not/exist/yet");
            let sample =
                RateLimitSample::Oauth { ts_unix: 1, h5_util: 0.0, d7_util: 0.0, h5_reset: 0, d7_reset: 0 };
            FsUsageStore::new(&nested).append("abc", &sample).await;
            assert!(nested.join("abc.jsonl").exists());
        }

        #[tokio::test]
        async fn load_samples_filters_stale() {
            let tmp = TempDir::new();
            let store = FsUsageStore::new(tmp.path());
            let now = 100 * 24 * 60 * 60; // day 100
            let fresh = RateLimitSample::Oauth {
                ts_unix: now - 60,
                h5_util: 0.5,
                d7_util: 0.2,
                h5_reset: 0,
                d7_reset: 0,
            };
            let stale = RateLimitSample::Oauth {
                ts_unix: now - RETENTION_SECS - 1,
                h5_util: 0.9,
                d7_util: 0.9,
                h5_reset: 0,
                d7_reset: 0,
            };
            store.append("fp1", &stale).await;
            store.append("fp1", &fresh).await;
            let got = store.load("fp1", now).await;
            assert_eq!(got.len(), 1);
            assert_eq!(got[0], fresh);
        }

        #[tokio::test]
        async fn load_samples_skips_malformed_lines() {
            let tmp = TempDir::new();
            let path = tmp.path().join("fp2.jsonl");
            tokio::fs::write(
                &path,
                r#"{"k":"o","t":1,"h5":0.1,"d7":0.0,"h5r":0,"d7r":0}
not json at all
{"k":"o","t":2,"h5":0.2,"d7":0.0,"h5r":0,"d7r":0}
"#,
            )
            .await
            .unwrap();
            let got = FsUsageStore::new(tmp.path()).load("fp2", 100).await;
            assert_eq!(got.len(), 2);
        }

        #[tokio::test]
        async fn load_samples_returns_empty_when_file_missing() {
            let tmp = TempDir::new();
            let got = FsUsageStore::new(tmp.path()).load("no-such-fp", 0).await;
            assert!(got.is_empty());
        }

        #[tokio::test]
        async fn list_identities_returns_sorted_by_recency() {
            let tmp = TempDir::new();
            let store = FsUsageStore::new(tmp.path());
            let now = 100 * 24 * 60 * 60;
            let old = RateLimitSample::Oauth {
                ts_unix: now - 3600,
                h5_util: 0.1,
                d7_util: 0.0,
                h5_reset: 0,
                d7_reset: 0,
            };
            let new = RateLimitSample::ApiKey {
                ts_unix: now - 60,
                requests_util: 0.2,
                tokens_util: 0.1,
                input_util: 0.0,
                output_util: 0.0,
                requests_reset: 0,
                tokens_reset: 0,
            };
            store.append("olderid", &old).await;
            store.append("newerid", &new).await;

            let ids = store.list_identities(now).await;
            assert_eq!(ids.len(), 2);
            assert_eq!(ids[0].fingerprint, "newerid");
            assert_eq!(ids[0].kind.as_deref(), Some("apikey"));
            assert_eq!(ids[1].fingerprint, "olderid");
            assert_eq!(ids[1].kind.as_deref(), Some("oauth"));
        }
    }
}
