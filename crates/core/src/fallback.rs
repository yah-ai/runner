//! Declarative fallback rule evaluator for agent subclasses.
//!
//! Evaluates [`kg::FallbackRule`]s against a [`MetricSnapshot`] before each
//! turn. When the first matching rule's debounce window expires, it returns
//! [`FallbackSwitch::Triggered`]; when no rule matches after an active fallback,
//! it returns [`FallbackSwitch::Recovered`].
//!
//! ## Design
//!
//! - **Pure evaluation**: [`evaluate_condition`] takes a snapshot and a
//!   condition; no I/O, no state. Unit-testable in isolation.
//! - **Stateful debounce**: [`FallbackEvaluator`] owns per-rule onset timers.
//!   Embed one in each session record.
//! - **Metric sourcing**: [`MetricSnapshot`] is assembled by the caller before
//!   each turn. Health fields come from [`crate::health::RunnerHealth`];
//!   usage fields (weekly_tokens, session_usage_pct, etc.) come from the
//!   host's usage tracker (supplied as `None` until that tracker lands).
//!

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

// ── Probe conditions ─────────────────────────────────────────────────────────

/// Declarative condition for a fallback rule. Evaluated by the daemon's
/// rule evaluator (R037-F17). `metric op threshold` reads as e.g.
/// "session_usage_pct > 50".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeCondition {
    pub metric: Metric,
    pub op: ComparisonOp,
    pub threshold: MetricValue,
    /// Optional debounce: condition must hold for this many seconds
    /// before the rule fires.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sustained_for_secs: Option<u32>,
}

/// Runtime metrics the rule evaluator can probe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Metric {
    /// Pro/Max 5-hour rolling usage window, 0–100.
    SessionUsagePct,
    /// Rolling 7-day total input+output tokens.
    WeeklyTokens,
    /// Daily input+output tokens.
    DailyTokens,
    /// Daily cost in USD.
    DailyCostUsd,
    /// Seconds the connection has been returning 4xx/5xx.
    ConnectionUnhealthyFor,
    /// Seconds the connection has been 429 rate-limited.
    RateLimitedFor,
    /// Boolean: `1.0` when auth token has expired, `0.0` otherwise.
    AuthExpired,
    /// Hour of day (0–23) in the given timezone. Useful for "use local
    /// model outside business hours" rules.
    HourOfDay {
        tz: String,
    },
}

/// Comparison operators for probe conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonOp {
    Gt,
    Gte,
    Lt,
    Lte,
    Eq,
}

/// Numeric value in a probe condition threshold. Stored as `f64` so
/// integer and fractional thresholds round-trip cleanly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricValue(pub f64);

// ── MetricSnapshot + evaluate_condition ──────────────────────────────────────

/// Point-in-time metric values for fallback rule evaluation.
///
/// All fields are `Option<f64>` — `None` means "not measurable right now".
/// Rules that probe an absent metric evaluate to `false` (safe: keeps the
/// primary subclass). Moved here from `runner::fallback` (R287) so the
/// stateless `kg::resolver::resolve` algorithm can consume it without a
/// dependency on the runner crate.
#[derive(Debug, Clone, Default)]
pub struct MetricSnapshot {
    /// Pro/Max 5-hour rolling window, 0.0–100.0.
    pub session_usage_pct: Option<f64>,
    /// Rolling 7-day total input+output tokens.
    pub weekly_tokens: Option<f64>,
    /// Daily input+output tokens.
    pub daily_tokens: Option<f64>,
    /// Daily cost in USD.
    pub daily_cost_usd: Option<f64>,
    /// Seconds the connection has been returning 4xx/5xx.
    pub connection_unhealthy_for_secs: Option<f64>,
    /// Seconds the connection has been 429 rate-limited.
    pub rate_limited_for_secs: Option<f64>,
    /// `1.0` when auth token has expired, `0.0` otherwise.
    pub auth_expired: Option<f64>,
    /// Hour of day (0.0–23.0) per named IANA timezone string.
    pub hour_of_day: HashMap<String, f64>,
}

fn sample_metric(metric: &Metric, snapshot: &MetricSnapshot) -> Option<f64> {
    match metric {
        Metric::SessionUsagePct => snapshot.session_usage_pct,
        Metric::WeeklyTokens => snapshot.weekly_tokens,
        Metric::DailyTokens => snapshot.daily_tokens,
        Metric::DailyCostUsd => snapshot.daily_cost_usd,
        Metric::ConnectionUnhealthyFor => snapshot.connection_unhealthy_for_secs,
        Metric::RateLimitedFor => snapshot.rate_limited_for_secs,
        Metric::AuthExpired => snapshot.auth_expired,
        Metric::HourOfDay { tz } => snapshot.hour_of_day.get(tz).copied(),
    }
}

/// Evaluate one [`ProbeCondition`] against `snapshot` (no debounce).
///
/// Returns `false` when the metric value is unavailable, so rules that
/// probe un-wired metrics never fire inadvertently.
pub fn evaluate_condition(condition: &ProbeCondition, snapshot: &MetricSnapshot) -> bool {
    let Some(value) = sample_metric(&condition.metric, snapshot) else {
        return false;
    };
    let t = condition.threshold.0;
    match condition.op {
        ComparisonOp::Gt => value > t,
        ComparisonOp::Gte => value >= t,
        ComparisonOp::Lt => value < t,
        ComparisonOp::Lte => value <= t,
        ComparisonOp::Eq => (value - t).abs() < f64::EPSILON,
    }
}

// ── Rule seam ────────────────────────────────────────────────────────────────

/// What [`FallbackEvaluator`] needs from a host's rule type: the probe
/// condition and, when the rule's outcome is "switch to subclass X", that id.
/// The host's rule container (it carries camp subclass types in its `then`)
/// implements this; rules whose outcome is not a plain subclass switch return
/// `None` and are skipped by mid-session evaluation.
pub trait FallbackRuleLike {
    fn condition(&self) -> &ProbeCondition;
    fn subclass_id(&self) -> Option<&str>;
}

// ── FallbackSwitch ────────────────────────────────────────────────────────────

/// Outcome returned by [`FallbackEvaluator::evaluate`].
///
/// The caller maps this to a `ConfigSwitchReason` and emits a
/// `AgentEvent::ConfigSwitch` when the variant is not
/// [`Self::Unchanged`].
#[derive(Debug, Clone, PartialEq)]
pub enum FallbackSwitch {
    /// Active subclass is unchanged — no event needed.
    Unchanged,
    /// A fallback rule fired (first time, or switched to a different one).
    /// Caller emits `ConfigSwitch{FallbackTriggered { subclass_id }}`.
    Triggered { subclass_id: String },
    /// No rule fires anymore; primary subclass is restored.
    /// Caller emits `ConfigSwitch{FallbackRecovered}`.
    Recovered,
}

// ── FallbackEvaluator ─────────────────────────────────────────────────────────

/// Stateful evaluator for a subclass's fallback rules.
///
/// Embed one in each session record alongside the agent's `fallback_rules`.
/// Call [`evaluate`] before each turn — it updates debounce timers, computes
/// the new decision, and returns whether the active subclass changed.
///
/// Rules are evaluated in order; the **first** rule whose condition has held
/// for at least `sustained_for_secs` (or immediately when that field is
/// absent) wins. Lower-priority rules accumulate their own onset timers
/// independently, so they "pre-warm" while a higher-priority rule is still
/// debouncing.
pub struct FallbackEvaluator {
    /// Rule index → wall-clock instant when the condition first became true.
    condition_onset: HashMap<usize, Instant>,
    /// Currently active fallback subclass id, if any.
    active_fallback: Option<String>,
}

impl Default for FallbackEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl FallbackEvaluator {
    pub fn new() -> Self {
        Self {
            condition_onset: HashMap::new(),
            active_fallback: None,
        }
    }

    /// Returns the currently active fallback subclass id, if any.
    pub fn active_fallback(&self) -> Option<&str> {
        self.active_fallback.as_deref()
    }

    /// Evaluate `rules` against `snapshot`, updating debounce timers.
    ///
    /// Returns [`FallbackSwitch::Unchanged`] when the active subclass didn't
    /// change, [`FallbackSwitch::Triggered`] when a rule fires for the first
    /// time or switches to a different fallback, and
    /// [`FallbackSwitch::Recovered`] when the last active fallback cleared.
    pub fn evaluate<R: FallbackRuleLike>(
        &mut self,
        rules: &[R],
        snapshot: &MetricSnapshot,
    ) -> FallbackSwitch {
        let now = Instant::now();
        let mut winning: Option<&str> = None;

        for (i, rule) in rules.iter().enumerate() {
            if evaluate_condition(rule.condition(), snapshot) {
                let onset = self.condition_onset.entry(i).or_insert(now);
                let held_secs = now.duration_since(*onset).as_secs() as u32;
                let required = rule.condition().sustained_for_secs.unwrap_or(0);
                if held_secs >= required {
                    if let Some(id) = rule.subclass_id() {
                        winning = Some(id);
                        break;
                    }
                    // PartialConfig / ClassDefault — not handled here; skip.
                }
                // Still debouncing — fall through to lower-priority rules.
            } else {
                // Condition no longer met; reset its onset so debounce restarts
                // if the condition becomes true again.
                self.condition_onset.remove(&i);
            }
        }

        let prev = self.active_fallback.take();
        let next = winning.map(str::to_string);

        let switch = match (&prev, &next) {
            (None, None) => FallbackSwitch::Unchanged,
            (None, Some(sc)) => FallbackSwitch::Triggered { subclass_id: sc.clone() },
            (Some(_), None) => FallbackSwitch::Recovered,
            (Some(p), Some(n)) if p == n => FallbackSwitch::Unchanged,
            (Some(_), Some(sc)) => FallbackSwitch::Triggered { subclass_id: sc.clone() },
        };

        self.active_fallback = next;
        switch
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Stand-in for the host's rule container (`party::FallbackRule`).
    struct FallbackRule {
        when: ProbeCondition,
        target: String,
    }

    impl FallbackRuleLike for FallbackRule {
        fn condition(&self) -> &ProbeCondition {
            &self.when
        }
        fn subclass_id(&self) -> Option<&str> {
            Some(&self.target)
        }
    }

    fn rule(metric: Metric, op: ComparisonOp, threshold: f64, target: &str) -> FallbackRule {
        FallbackRule {
            when: ProbeCondition {
                metric,
                op,
                threshold: MetricValue(threshold),
                sustained_for_secs: None,
            },
            target: target.to_string(),
        }
    }

    fn rule_debounced(
        metric: Metric,
        op: ComparisonOp,
        threshold: f64,
        target: &str,
        secs: u32,
    ) -> FallbackRule {
        FallbackRule {
            when: ProbeCondition {
                metric,
                op,
                threshold: MetricValue(threshold),
                sustained_for_secs: Some(secs),
            },
            target: target.to_string(),
        }
    }

    // ── evaluate_condition ────────────────────────────────────────────────────

    #[test]
    fn condition_gt_fires_when_above_threshold() {
        let cond = ProbeCondition {
            metric: Metric::ConnectionUnhealthyFor,
            op: ComparisonOp::Gt,
            threshold: MetricValue(30.0),
            sustained_for_secs: None,
        };
        let snap = MetricSnapshot {
            connection_unhealthy_for_secs: Some(31.0),
            ..Default::default()
        };
        assert!(evaluate_condition(&cond, &snap));
    }

    #[test]
    fn condition_gt_does_not_fire_at_threshold() {
        let cond = ProbeCondition {
            metric: Metric::ConnectionUnhealthyFor,
            op: ComparisonOp::Gt,
            threshold: MetricValue(30.0),
            sustained_for_secs: None,
        };
        let snap = MetricSnapshot {
            connection_unhealthy_for_secs: Some(30.0),
            ..Default::default()
        };
        assert!(!evaluate_condition(&cond, &snap));
    }

    #[test]
    fn condition_returns_false_for_missing_metric() {
        let cond = ProbeCondition {
            metric: Metric::WeeklyTokens,
            op: ComparisonOp::Gt,
            threshold: MetricValue(1_000_000.0),
            sustained_for_secs: None,
        };
        let snap = MetricSnapshot::default(); // weekly_tokens = None
        assert!(!evaluate_condition(&cond, &snap));
    }

    #[test]
    fn condition_lte_fires_at_threshold() {
        let cond = ProbeCondition {
            metric: Metric::SessionUsagePct,
            op: ComparisonOp::Lte,
            threshold: MetricValue(50.0),
            sustained_for_secs: None,
        };
        let snap = MetricSnapshot {
            session_usage_pct: Some(50.0),
            ..Default::default()
        };
        assert!(evaluate_condition(&cond, &snap));
    }

    #[test]
    fn condition_eq_fires_exactly() {
        let cond = ProbeCondition {
            metric: Metric::AuthExpired,
            op: ComparisonOp::Eq,
            threshold: MetricValue(1.0),
            sustained_for_secs: None,
        };
        let snap_expired = MetricSnapshot { auth_expired: Some(1.0), ..Default::default() };
        let snap_ok = MetricSnapshot { auth_expired: Some(0.0), ..Default::default() };
        assert!(evaluate_condition(&cond, &snap_expired));
        assert!(!evaluate_condition(&cond, &snap_ok));
    }

    #[test]
    fn condition_hour_of_day_by_timezone() {
        let cond = ProbeCondition {
            metric: Metric::HourOfDay { tz: "America/New_York".to_string() },
            op: ComparisonOp::Lt,
            threshold: MetricValue(9.0),
            sustained_for_secs: None,
        };
        let mut snap = MetricSnapshot::default();
        snap.hour_of_day.insert("America/New_York".to_string(), 8.0);
        assert!(evaluate_condition(&cond, &snap));
        snap.hour_of_day.insert("America/New_York".to_string(), 10.0);
        assert!(!evaluate_condition(&cond, &snap));
    }

    // ── FallbackEvaluator ─────────────────────────────────────────────────────

    #[test]
    fn no_rules_always_unchanged() {
        let mut ev = FallbackEvaluator::new();
        let snap = MetricSnapshot {
            connection_unhealthy_for_secs: Some(999.0),
            ..Default::default()
        };
        assert_eq!(ev.evaluate::<FallbackRule>(&[], &snap), FallbackSwitch::Unchanged);
        assert_eq!(ev.active_fallback(), None);
    }

    #[test]
    fn rule_fires_immediately_when_no_debounce() {
        let rules = vec![rule(
            Metric::RateLimitedFor,
            ComparisonOp::Gt,
            0.0,
            "cheap-fallback",
        )];
        let mut ev = FallbackEvaluator::new();
        let snap = MetricSnapshot {
            rate_limited_for_secs: Some(1.0),
            ..Default::default()
        };
        let sw = ev.evaluate(&rules, &snap);
        assert_eq!(
            sw,
            FallbackSwitch::Triggered { subclass_id: "cheap-fallback".into() }
        );
        assert_eq!(ev.active_fallback(), Some("cheap-fallback"));
    }

    #[test]
    fn rule_recovers_when_condition_clears() {
        let rules = vec![rule(
            Metric::ConnectionUnhealthyFor,
            ComparisonOp::Gt,
            30.0,
            "fallback-sc",
        )];
        let mut ev = FallbackEvaluator::new();

        let snap_bad = MetricSnapshot {
            connection_unhealthy_for_secs: Some(60.0),
            ..Default::default()
        };
        let snap_ok = MetricSnapshot::default();

        // First call: triggers.
        assert!(matches!(ev.evaluate(&rules, &snap_bad), FallbackSwitch::Triggered { .. }));
        // Still bad: unchanged.
        assert_eq!(ev.evaluate(&rules, &snap_bad), FallbackSwitch::Unchanged);
        // Recovered: emit Recovered.
        assert_eq!(ev.evaluate(&rules, &snap_ok), FallbackSwitch::Recovered);
        // Healthy again: unchanged.
        assert_eq!(ev.evaluate(&rules, &snap_ok), FallbackSwitch::Unchanged);
    }

    #[test]
    fn debounce_does_not_fire_before_sustained_window() {
        // Rule requires condition held for 60s. In unit tests time doesn't
        // advance so the very first call won't satisfy the debounce.
        let rules = vec![rule_debounced(
            Metric::SessionUsagePct,
            ComparisonOp::Gte,
            80.0,
            "heavy-fallback",
            60,
        )];
        let mut ev = FallbackEvaluator::new();
        let snap = MetricSnapshot {
            session_usage_pct: Some(90.0),
            ..Default::default()
        };
        // First evaluate: onset is set to `now`, held_secs == 0, required == 60.
        assert_eq!(ev.evaluate(&rules, &snap), FallbackSwitch::Unchanged);
        assert_eq!(ev.active_fallback(), None);
    }

    #[test]
    fn debounce_fires_once_threshold_elapsed() {
        // Bypass wall-clock by using a 0-second debounce — the onset is set
        // on the first call and immediately satisfies held_secs >= 0.
        let rules = vec![rule_debounced(
            Metric::SessionUsagePct,
            ComparisonOp::Gte,
            80.0,
            "heavy-fallback",
            0,
        )];
        let mut ev = FallbackEvaluator::new();
        let snap = MetricSnapshot {
            session_usage_pct: Some(90.0),
            ..Default::default()
        };
        // With sustained_for_secs=0, the first evaluate immediately satisfies.
        let sw = ev.evaluate(&rules, &snap);
        assert_eq!(sw, FallbackSwitch::Triggered { subclass_id: "heavy-fallback".into() });
    }

    #[test]
    fn debounce_onset_resets_when_condition_clears() {
        // A 0-second debounce: onset set on call 1, then condition clears on
        // call 2 (removing onset), then fires again on call 3.
        let rules = vec![rule_debounced(
            Metric::RateLimitedFor,
            ComparisonOp::Gt,
            0.0,
            "rate-fallback",
            0,
        )];
        let mut ev = FallbackEvaluator::new();
        let snap_bad = MetricSnapshot { rate_limited_for_secs: Some(1.0), ..Default::default() };
        let snap_ok = MetricSnapshot::default();

        // fires
        assert!(matches!(ev.evaluate(&rules, &snap_bad), FallbackSwitch::Triggered { .. }));
        // recovers
        assert_eq!(ev.evaluate(&rules, &snap_ok), FallbackSwitch::Recovered);
        // fires again
        assert!(matches!(ev.evaluate(&rules, &snap_bad), FallbackSwitch::Triggered { .. }));
    }

    #[test]
    fn lower_priority_rule_fires_when_higher_debouncing() {
        // Rule 0: session_usage_pct > 80, debounce 60s (won't satisfy in test).
        // Rule 1: rate_limited_for > 0, no debounce (fires immediately).
        let rules = vec![
            rule_debounced(Metric::SessionUsagePct, ComparisonOp::Gt, 80.0, "usage-sc", 60),
            rule(Metric::RateLimitedFor, ComparisonOp::Gt, 0.0, "rate-sc"),
        ];
        let mut ev = FallbackEvaluator::new();
        let snap = MetricSnapshot {
            session_usage_pct: Some(90.0),
            rate_limited_for_secs: Some(5.0),
            ..Default::default()
        };
        let sw = ev.evaluate(&rules, &snap);
        // Rule 0 debouncing, so rule 1 wins.
        assert_eq!(sw, FallbackSwitch::Triggered { subclass_id: "rate-sc".into() });
    }

    #[test]
    fn switching_between_fallbacks_emits_triggered_not_recovered() {
        // Start with rule 0 active, then switch to rule 1 (both conditions true,
        // rule 0 becomes false).
        let rules = vec![
            rule(Metric::RateLimitedFor, ComparisonOp::Gt, 0.0, "rate-sc"),
            rule(Metric::ConnectionUnhealthyFor, ComparisonOp::Gt, 0.0, "unhealthy-sc"),
        ];
        let mut ev = FallbackEvaluator::new();

        // rate-limited fires, unhealthy doesn't.
        let snap1 = MetricSnapshot { rate_limited_for_secs: Some(5.0), ..Default::default() };
        assert_eq!(
            ev.evaluate(&rules, &snap1),
            FallbackSwitch::Triggered { subclass_id: "rate-sc".into() }
        );

        // Now unhealthy fires, rate-limit clears.
        let snap2 = MetricSnapshot {
            connection_unhealthy_for_secs: Some(5.0),
            ..Default::default()
        };
        assert_eq!(
            ev.evaluate(&rules, &snap2),
            FallbackSwitch::Triggered { subclass_id: "unhealthy-sc".into() }
        );
    }
}
