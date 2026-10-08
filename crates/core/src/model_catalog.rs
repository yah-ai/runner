//! @arch:layer(kg_store)
//! @arch:role(registry)
//!
//! Runtime model catalog — the one schema every layer reads for
//! per-model metadata (label, thinking support, context limit, pricing
//! cliff, size tier).
//!
//! Before R863-T1 the same facts were hand-copied across four places:
//! `app/yah/desktop/resources/model-catalog.json` (id/label/thinking),
//! `packages/yah/ui/src/env/models.ts` (contextLimit/pricingCliff/tier),
//! `packages/yah/ui/src/env/browser.ts` (a literal mirror of the JSON for
//! browser dev mode), and a `PREFIXES` const compiled into
//! `model_context_tokens`. Drift between them was
//! invisible until something rendered a wrong context meter — R862 caught
//! exactly that by chance. Now `resources/model-catalog.json` in *this*
//! crate is the single source; the others read it.
//!
//! ## Layering
//!
//! [`EMBEDDED`] is compiled in and always present. A host may overlay more
//! layers (a shared cache, a camp-local override) and hand the result to
//! [`install_catalog`]; layers merge **field-wise** ([`Catalog::overlay`]) so
//! one that only knows some fields never blanks out what a lower layer knew.
//! Reading files is the host's job — core has no filesystem access.
//!
//! ## Lookup
//!
//! [`Catalog::resolve`] matches an exact `id` first, then falls back to the
//! ordered `prefixes` array (first match wins, so longer/more specific
//! prefixes are listed first). Fields from the exact entry are layered over
//! the prefix match, so a date-stamped id (`claude-3-5-sonnet-20241022`)
//! and an alias (`opus`) both resolve without needing an entry each.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::Deserialize;

/// The compiled-in catalog. Also the byte source the desktop's resource
/// manifest hands out for the `"model-catalog"` resource, so both sides
/// start from the same default.
pub const EMBEDDED: &str = include_str!("../resources/model-catalog.json");

/// Everything the catalog knows about one model. Every field is optional:
/// a layer contributes what it knows and leaves the rest alone.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelInfo {
    pub label: Option<String>,
    pub supports_thinking: Option<bool>,
    /// Hard context-window cap in tokens.
    pub context_limit: Option<u32>,
    /// First pricing-tier boundary, or `None` for single-tier models.
    pub pricing_cliff: Option<u32>,
    /// Coarse size tier: `"slim"` / `"mid"` / `"full"`.
    pub tier: Option<String>,
}

impl ModelInfo {
    /// Layer `other` on top of `self` — every field `other` actually
    /// carries wins; the rest survive.
    fn overlay(&mut self, other: &ModelInfo) {
        if other.label.is_some() {
            self.label = other.label.clone();
        }
        if other.supports_thinking.is_some() {
            self.supports_thinking = other.supports_thinking;
        }
        if other.context_limit.is_some() {
            self.context_limit = other.context_limit;
        }
        if other.pricing_cliff.is_some() {
            self.pricing_cliff = other.pricing_cliff;
        }
        if other.tier.is_some() {
            self.tier = other.tier.clone();
        }
    }
}

/// A parsed catalog: exact-id entries plus the ordered prefix fallbacks.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    entries: BTreeMap<String, ModelInfo>,
    prefixes: Vec<(String, ModelInfo)>,
}

impl Catalog {
    /// Parse one catalog document. Presets are flattened — the runner only
    /// needs id → metadata; which preset offered an id is the picker's
    /// concern, not this lookup's.
    pub fn parse(json: &str) -> Result<Self, String> {
        let raw: RawCatalog = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let mut cat = Catalog::default();
        for preset in raw.presets.values() {
            preset.collect_into(&mut cat.entries);
        }
        cat.prefixes = raw
            .prefixes
            .into_iter()
            .map(|p| {
                (
                    p.prefix,
                    ModelInfo {
                        label: None,
                        supports_thinking: p.supports_thinking,
                        context_limit: p.context_limit,
                        pricing_cliff: p.pricing_cliff,
                        tier: p.tier,
                    },
                )
            })
            .collect();
        Ok(cat)
    }

    /// Merge `other` on top of `self`. Entries merge field-wise by id;
    /// `other`'s prefixes are *prepended* so an override can shadow a
    /// lower layer's prefix without having to restate the whole table.
    pub fn overlay(&mut self, other: Catalog) {
        for (id, info) in other.entries {
            self.entries.entry(id).or_default().overlay(&info);
        }
        let mut prefixes = other.prefixes;
        prefixes.append(&mut self.prefixes);
        self.prefixes = prefixes;
    }

    /// Everything known about `model`: the ordered prefix match, with the
    /// exact-id entry layered on top. Empty [`ModelInfo`] when nothing
    /// matches — callers read the field they care about and treat `None`
    /// as "unknown".
    pub fn resolve(&self, model: &str) -> ModelInfo {
        let mut info = self
            .prefixes
            .iter()
            .find(|(prefix, _)| model.starts_with(prefix.as_str()))
            .map(|(_, info)| info.clone())
            .unwrap_or_default();
        if let Some(exact) = self.entries.get(model) {
            info.overlay(exact);
        }
        info
    }

    /// Context-window cap for `model`, or `None` when unknown.
    pub fn context_tokens(&self, model: &str) -> Option<u32> {
        self.resolve(model).context_limit
    }

    /// Whether the catalog carries an explicit entry for this exact id
    /// (as opposed to resolving through a prefix). The drift guards use
    /// this to assert every id the picker offers is spelled out.
    pub fn has_entry(&self, model: &str) -> bool {
        self.entries.contains_key(model)
    }
}

// ── process-wide catalog ─────────────────────────────────────────────────────

static CATALOG: OnceLock<Catalog> = OnceLock::new();

/// Install the process-wide catalog. The host builds it (embedded plus
/// whatever file layers it knows about — core touches no filesystem) and
/// calls this once at startup; `true` when this call won. After the first
/// [`catalog`] read the embedded default is fixed and a later install is
/// refused.
pub fn install_catalog(catalog: Catalog) -> bool {
    CATALOG.set(catalog).is_ok()
}

/// The process-wide catalog: whatever the host installed, else
/// [`EMBEDDED`] alone.
pub fn catalog() -> &'static Catalog {
    CATALOG.get_or_init(|| Catalog::parse(EMBEDDED).unwrap_or_default())
}

/// Context-window cap for `model` from the process-wide [`catalog`], or
/// `None` when unknown. The one lookup every layer shares (it replaced a
/// hand-copied `PREFIXES` table in the Anthropic runner, R863-T1).
pub fn model_context_tokens(model: &str) -> Option<u32> {
    catalog().context_tokens(model)
}

// ── wire shapes ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct RawCatalog {
    #[serde(default)]
    presets: BTreeMap<String, RawPreset>,
    #[serde(default)]
    prefixes: Vec<RawPrefix>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawPreset {
    #[serde(default)]
    models: Vec<RawModel>,
    #[serde(default)]
    by_endpoint_id: BTreeMap<String, RawPreset>,
}

impl RawPreset {
    fn collect_into(&self, out: &mut BTreeMap<String, ModelInfo>) {
        for m in &self.models {
            out.entry(m.id.clone()).or_default().overlay(&ModelInfo {
                label: m.label.clone(),
                supports_thinking: m.supports_thinking,
                context_limit: m.context_limit,
                pricing_cliff: m.pricing_cliff,
                tier: m.tier.clone(),
            });
        }
        for endpoint in self.by_endpoint_id.values() {
            endpoint.collect_into(out);
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawModel {
    id: String,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    supports_thinking: Option<bool>,
    #[serde(default)]
    context_limit: Option<u32>,
    #[serde(default)]
    pricing_cliff: Option<u32>,
    #[serde(default)]
    tier: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawPrefix {
    prefix: String,
    #[serde(default)]
    supports_thinking: Option<bool>,
    #[serde(default)]
    context_limit: Option<u32>,
    #[serde(default)]
    pricing_cliff: Option<u32>,
    #[serde(default)]
    tier: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedded() -> Catalog {
        Catalog::parse(EMBEDDED).expect("embedded catalog parses")
    }

    #[test]
    fn embedded_catalog_parses_and_covers_the_picker() {
        let cat = embedded();
        // Every concrete id the anthropic preset offers must carry an
        // explicit context limit — the drift guard R862 wanted.
        for id in [
            "claude-fable-5-1",
            "claude-opus-5",
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-sonnet-5",
            "claude-sonnet-4-6",
            "claude-sonnet-4-6[1m]",
            "claude-haiku-4-5-20251001",
        ] {
            assert!(cat.has_entry(id), "catalog has no entry for `{id}`");
            assert!(
                cat.context_tokens(id).is_some(),
                "catalog entry `{id}` carries no contextLimit",
            );
        }
    }

    #[test]
    fn exact_entries_carry_tier_and_cliff() {
        let cat = embedded();
        let sonnet_1m = cat.resolve("claude-sonnet-4-6[1m]");
        assert_eq!(sonnet_1m.context_limit, Some(1_000_000));
        assert_eq!(sonnet_1m.pricing_cliff, Some(200_000));
        assert_eq!(sonnet_1m.tier.as_deref(), Some("mid"));

        let haiku = cat.resolve("claude-haiku-4-5-20251001");
        assert_eq!(haiku.tier.as_deref(), Some("slim"));
        assert_eq!(haiku.supports_thinking, Some(false));
        assert_eq!(haiku.pricing_cliff, None);
    }

    #[test]
    fn bare_aliases_resolve_through_entries() {
        let cat = embedded();
        // "opus"/"sonnet"/"haiku" have no prefix row — they only resolve
        // because the picker presets list them as ids.
        assert_eq!(cat.context_tokens("opus"), Some(1_000_000));
        assert_eq!(cat.context_tokens("sonnet"), Some(200_000));
        assert_eq!(cat.context_tokens("haiku"), Some(200_000));
    }

    #[test]
    fn unlisted_ids_fall_back_to_the_prefix_table() {
        let cat = embedded();
        // Date-stamped and undated variants nobody lists explicitly.
        assert!(!cat.has_entry("claude-3-5-sonnet-20241022"));
        assert_eq!(cat.context_tokens("claude-3-5-sonnet-20241022"), Some(200_000));
        assert_eq!(cat.context_tokens("claude-haiku-4-5"), Some(200_000));
        assert_eq!(cat.resolve("claude-haiku-4-5").tier.as_deref(), Some("slim"));
        assert_eq!(cat.context_tokens("kimi-k2.5"), Some(262_144));
        assert_eq!(cat.context_tokens("unknown-model-xyz"), None);
    }

    #[test]
    fn prefix_order_is_first_match_wins() {
        let cat = embedded();
        // gpt-4o must beat the bare gpt-4 (8K) row; the [1m] sonnet must
        // beat the 200K claude-sonnet-4 row.
        assert_eq!(cat.context_tokens("gpt-4o-2024-08-06"), Some(131_072));
        assert_eq!(cat.context_tokens("gpt-4-0613"), Some(8_192));
        assert_eq!(cat.context_tokens("claude-sonnet-4-6[1m]"), Some(1_000_000));
        assert_eq!(cat.context_tokens("claude-sonnet-4-6"), Some(200_000));
    }

    #[test]
    fn overlay_merges_field_wise_and_never_blanks_a_lower_layer() {
        let mut cat = embedded();
        // An overlay that knows only the label — the schema an older cache
        // (written before contextLimit existed) would produce.
        cat.overlay(
            Catalog::parse(
                r#"{"presets":{"anthropic":{"models":[
                     {"id":"claude-opus-5","label":"Renamed Opus","supportsThinking":true}
                   ]}}}"#,
            )
            .unwrap(),
        );
        let info = cat.resolve("claude-opus-5");
        assert_eq!(info.label.as_deref(), Some("Renamed Opus"));
        assert_eq!(
            info.context_limit,
            Some(1_000_000),
            "a label-only overlay must not erase the embedded contextLimit",
        );
    }

    #[test]
    fn overlay_can_raise_a_limit_and_add_an_unknown_model() {
        let mut cat = embedded();
        cat.overlay(
            Catalog::parse(
                r#"{"presets":{"local":{"models":[
                     {"id":"claude-opus-5","contextLimit":2000000},
                     {"id":"my-local-model","contextLimit":65536,"tier":"slim"}
                   ]}},
                   "prefixes":[{"prefix":"my-local-","contextLimit":8192}]}"#,
            )
            .unwrap(),
        );
        assert_eq!(cat.context_tokens("claude-opus-5"), Some(2_000_000));
        assert_eq!(cat.context_tokens("my-local-model"), Some(65_536));
        assert_eq!(cat.resolve("my-local-model").tier.as_deref(), Some("slim"));
        // Overlay prefixes are prepended, so they shadow the base table.
        assert_eq!(cat.context_tokens("my-local-other"), Some(8_192));
    }
}
