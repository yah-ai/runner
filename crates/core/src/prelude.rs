//! The rendered system prompt a host hands to [`crate::Runner::start`].
//!
//! Core never assembles a prelude — it has no notion of tickets, doctrine or
//! knowledge graphs. The host renders its own sections and fills this type;
//! the runner concatenates them, keys its prompt cache off [`CacheControl`],
//! and picks the model and reasoning level from the optional refs.

use crate::cache::CacheControl;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// One rendered block of the system prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreludeSection {
    /// Stable identity of this block. Runners record the set of ids as the
    /// session's resident set, so a host can tell whether a block is already
    /// loaded. Ids are host-defined and opaque to core.
    pub id: String,
    /// Rendered markdown for the block.
    pub markdown: String,
}

impl PreludeSection {
    pub fn new(id: impl Into<String>, markdown: impl Into<String>) -> Self {
        Self { id: id.into(), markdown: markdown.into() }
    }
}

/// Which provider and model a session should run on. `model: None` means the
/// runner's configured default.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelRef {
    pub provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl ModelRef {
    pub fn new(provider: impl Into<String>, model: Option<String>) -> Self {
        Self { provider: provider.into(), model }
    }

    /// Canonical `provider:model` (or bare `provider`) string form.
    pub fn as_payload(&self) -> String {
        match &self.model {
            Some(m) => format!("{}:{}", self.provider, m),
            None => self.provider.clone(),
        }
    }
}

/// How much extended reasoning a turn may spend. The named tiers are mapped
/// to a concrete budget by each runner for its provider; `Budget` is an
/// explicit token cap the runner clamps to what the model supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ReasoningBudget {
    Deep,
    Standard,
    Fast,
    Budget { tokens: u32 },
}

/// A rendered prelude: the host's sections plus the metadata a runner needs
/// to open a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prelude {
    pub sections: Vec<PreludeSection>,
    pub cache: CacheControl,
    /// Provider/model to run on. A runner rejects a provider it does not
    /// serve with [`crate::RunnerError::UnsupportedEngine`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelRef>,
    /// Reasoning level the session starts at. `None` leaves it to the
    /// runner's configured default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningBudget>,
    /// Display name of the agent running this session, when the host has
    /// one. Runners may surface it to the model as self-identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Host-defined key/value metadata. Core never reads it; it rides along
    /// so a host's own runner extensions can see values the host set at
    /// start without a side channel.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub host: std::collections::BTreeMap<String, String>,
}

impl Prelude {
    /// A prelude with no sections and no model/reasoning preference.
    pub fn empty(cache: CacheControl) -> Self {
        Self {
            sections: Vec::new(),
            cache,
            model: None,
            reasoning: None,
            agent_id: None,
            host: Default::default(),
        }
    }

    /// Concatenate the sections into the system prompt text. Sections are
    /// joined by a blank line and each is newline-terminated.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for (i, section) in self.sections.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.push_str(&section.markdown);
            if !section.markdown.ends_with('\n') {
                out.push('\n');
            }
        }
        out
    }

    /// The set of section ids — the session's initial resident set.
    pub fn section_id_set(&self) -> BTreeSet<String> {
        self.sections.iter().map(|s| s.id.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CacheTtl;

    fn cache() -> CacheControl {
        CacheControl { key: "k".into(), ttl: CacheTtl::Ephemeral }
    }

    #[test]
    fn render_joins_sections_newline_terminated() {
        let mut p = Prelude::empty(cache());
        p.sections.push(PreludeSection::new("a", "alpha"));
        p.sections.push(PreludeSection::new("b", "beta\n"));
        assert_eq!(p.render(), "alpha\n\nbeta\n");
        assert_eq!(
            p.section_id_set().into_iter().collect::<Vec<_>>(),
            vec!["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn model_ref_payload_round_trips_shape() {
        assert_eq!(ModelRef::new("openai", Some("gpt".into())).as_payload(), "openai:gpt");
        assert_eq!(ModelRef::new("ollama", None).as_payload(), "ollama");
    }
}
