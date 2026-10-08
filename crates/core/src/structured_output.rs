//! Structured-output enforcement for `party.dispatch` (R534-F2).
//!
//! When a dispatch carries a `schema` (a JSON Schema object), the child is
//! steered to answer with a single JSON value matching that schema, and the
//! parent validates the reply before handing it back. This mirrors the
//! Workflow `agent({schema})` contract: the caller gets a *validated object*
//! in place of free text, with a bounded retry-on-mismatch loop.
//!
//! Why prompt-steering + parse-and-validate rather than a forced
//! `StructuredOutput` tool call: yah dispatches across five runner families
//! (anthropic SDK, claude-cli subprocess, openai-compat, codex-oauth,
//! yah-runner), and `tool_choice`-style forcing is not uniformly available
//! across them. Steering the child to emit JSON and validating the reply is
//! engine-agnostic and rides the existing first-turn `final_response` /
//! `subagent.send` substrate with no daemon changes. The validation gate is
//! the load-bearing part — the prompt is just how we ask.
//!
//! The validator implements the practical JSON Schema subset that
//! agent-authored schemas use: `type` (incl. union arrays + `integer`),
//! `required`, `properties`, `items`, `enum`, and `additionalProperties:
//! false`. Unknown keywords are ignored (treated as "no constraint") rather
//! than erroring, so a richer schema still validates on the parts we
//! understand instead of being rejected wholesale.

use serde_json::{json, Value};

/// Build the child's first-turn prompt: the caller's prompt followed by a
/// structured-output directive embedding the JSON Schema. Kept terse — the
/// schema itself carries the field-level contract.
pub fn build_structured_prompt(prompt: &str, schema: &Value) -> String {
    let pretty = serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string());
    format!(
        "{prompt}\n\n\
         ---\n\
         Respond with a SINGLE JSON value that validates against this JSON Schema, \
         and nothing else — no prose, no explanation, no markdown fences around it:\n\n\
         {pretty}"
    )
}

/// R775-F2 (W321 §5): every caller-declared schema gets a free-form
/// `unexpected` property, mandatory, because a rigid schema silently drops
/// whatever the caller didn't know to ask for. Idempotent — a schema that
/// already declares `unexpected` is left alone. No-op on a non-object schema
/// (an array/string/enum return has nowhere to hang a sibling field); those
/// are rare enough among caller-declared return schemas that inventing a
/// wrapper would cost more clarity than the hatch is worth.
pub fn with_escape_hatch(schema: &Value) -> Value {
    let mut schema = schema.clone();
    let Some(obj) = schema.as_object_mut() else {
        return schema;
    };
    if obj.get("type").and_then(Value::as_str).is_some_and(|t| t != "object") {
        return schema;
    }
    let props = obj.entry("properties").or_insert_with(|| json!({}));
    if let Some(props) = props.as_object_mut() {
        props.entry("unexpected").or_insert_with(|| {
            json!({
                "type": "string",
                "description": "Anything worth reporting that this schema didn't ask for — \
                                 a finding, a caveat, a surprise. Free-form; never leave a \
                                 discovery out because there was no field for it."
            })
        });
    }
    schema
}

/// Async-path prompt injection (R775-F2, W321 §1). Unlike
/// [`build_structured_prompt`] — which demands the very NEXT reply be raw
/// JSON, correct only when the dispatcher is blocked waiting on it — a
/// `wait: false` dispatch lets the child work normally across turns and
/// declares the schema as an OBLIGATION due at hand-back time via
/// `subagent.release_to_parent { result }`.
pub fn build_async_schema_note(prompt: &str, schema: &Value) -> String {
    let pretty = serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string());
    format!(
        "{prompt}\n\n\
         ---\n\
         Your dispatcher declared a return schema for this task. When you are done, call \
         `subagent.release_to_parent` with a `result` argument set to a JSON value matching \
         this schema — fill every field you can, and put anything the schema didn't \
         anticipate in `unexpected` rather than leaving it out:\n\n\
         {pretty}\n\n\
         If you end the task on an ordinary closing turn instead of calling that tool, your \
         dispatcher only receives your raw text — calling it with `result` is the contract \
         it is actually waiting on."
    )
}

/// Corrective re-prompt sent when the child's reply failed validation. Names
/// the specific mismatch so the next turn can repair it, and re-states the
/// JSON-only contract.
pub fn correction_prompt(error: &str, schema: &Value) -> String {
    let pretty = serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string());
    format!(
        "Your previous reply did not match the required schema: {error}\n\n\
         Reply again with a SINGLE valid JSON value (no prose, no markdown fences) \
         matching:\n\n{pretty}"
    )
}

/// Pull a JSON value out of an assistant reply. Tries, in order: the whole
/// trimmed text; the contents of the first ```json / ``` fenced block; the
/// first balanced `{…}` or `[…]` span. Returns `None` when nothing parses.
pub fn extract_json(text: &str) -> Option<Value> {
    let trimmed = text.trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return Some(v);
    }
    if let Some(body) = fenced_block(trimmed) {
        if let Ok(v) = serde_json::from_str::<Value>(body.trim()) {
            return Some(v);
        }
    }
    if let Some(span) = balanced_span(trimmed) {
        if let Ok(v) = serde_json::from_str::<Value>(span) {
            return Some(v);
        }
    }
    None
}

/// Extract the contents of the first ```…``` fenced block, dropping an
/// optional language tag on the opening fence (e.g. ```json).
fn fenced_block(text: &str) -> Option<&str> {
    let start = text.find("```")?;
    let after = &text[start + 3..];
    // Skip an optional language tag up to the first newline.
    let body_start = after.find('\n').map(|n| n + 1).unwrap_or(0);
    let body = &after[body_start..];
    let end = body.find("```")?;
    Some(&body[..end])
}

/// Find the first balanced `{…}` or `[…]` span, honoring string literals so
/// braces inside strings don't throw off the depth count.
fn balanced_span(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let open = bytes.iter().position(|&b| b == b'{' || b == b'[')?;
    let (open_ch, close_ch) = match bytes[open] {
        b'{' => (b'{', b'}'),
        _ => (b'[', b']'),
    };
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(open) {
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            x if x == open_ch => depth += 1,
            x if x == close_ch => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[open..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Validate `value` against `schema`, returning `Ok(())` or a human-readable
/// `path: reason` describing the first failure. Implements the supported
/// JSON Schema subset (see module docs); unknown keywords are skipped.
pub fn validate(value: &Value, schema: &Value) -> Result<(), String> {
    validate_at("$", value, schema)
}

fn validate_at(path: &str, value: &Value, schema: &Value) -> Result<(), String> {
    let Some(obj) = schema.as_object() else {
        // A non-object schema (e.g. `true`) imposes no constraint.
        return Ok(());
    };

    if let Some(t) = obj.get("type") {
        check_type(path, value, t)?;
    }

    if let Some(Value::Array(allowed)) = obj.get("enum") {
        if !allowed.iter().any(|a| a == value) {
            return Err(format!("{path}: value not in enum"));
        }
    }

    match value {
        Value::Object(map) => {
            if let Some(Value::Array(required)) = obj.get("required") {
                for r in required {
                    if let Some(key) = r.as_str() {
                        match map.get(key) {
                            None | Some(Value::Null) => {
                                return Err(format!("{path}: missing required property '{key}'"));
                            }
                            _ => {}
                        }
                    }
                }
            }
            let props = obj.get("properties").and_then(Value::as_object);
            if let Some(props) = props {
                for (k, v) in map {
                    if let Some(sub) = props.get(k) {
                        validate_at(&format!("{path}.{k}"), v, sub)?;
                    }
                }
            }
            if obj.get("additionalProperties") == Some(&Value::Bool(false)) {
                if let Some(props) = props {
                    for k in map.keys() {
                        if !props.contains_key(k) {
                            return Err(format!("{path}: unexpected property '{k}'"));
                        }
                    }
                } else if !map.is_empty() {
                    return Err(format!("{path}: no additional properties allowed"));
                }
            }
        }
        Value::Array(items) => {
            if let Some(item_schema) = obj.get("items") {
                for (i, item) in items.iter().enumerate() {
                    validate_at(&format!("{path}[{i}]"), item, item_schema)?;
                }
            }
        }
        _ => {}
    }

    Ok(())
}

/// Check a value against a `type` keyword — either a single type name or an
/// array of acceptable names (union).
fn check_type(path: &str, value: &Value, t: &Value) -> Result<(), String> {
    let ok = match t {
        Value::String(name) => matches_type(value, name),
        Value::Array(names) => names
            .iter()
            .filter_map(Value::as_str)
            .any(|name| matches_type(value, name)),
        _ => true, // malformed type keyword — don't constrain
    };
    if ok {
        Ok(())
    } else {
        Err(format!(
            "{path}: expected type {t}, got {}",
            json_kind(value)
        ))
    }
}

fn matches_type(value: &Value, name: &str) -> bool {
    match name {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "number" => value.is_number(),
        // JSON Schema `integer` accepts an integral number (incl. 2.0).
        "integer" => {
            value.is_i64() || value.is_u64() || value.as_f64().is_some_and(|f| f.fract() == 0.0)
        }
        _ => true, // unknown type name — don't constrain
    }
}

fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extract_plain_json_object() {
        let v = extract_json("{\"a\": 1}").unwrap();
        assert_eq!(v, json!({"a": 1}));
    }

    #[test]
    fn extract_from_json_fence() {
        let text = "Here you go:\n```json\n{\"a\": 1}\n```\nthanks";
        assert_eq!(extract_json(text).unwrap(), json!({"a": 1}));
    }

    #[test]
    fn extract_from_bare_fence() {
        let text = "```\n[1, 2, 3]\n```";
        assert_eq!(extract_json(text).unwrap(), json!([1, 2, 3]));
    }

    #[test]
    fn extract_balanced_span_with_surrounding_prose() {
        let text = "Sure! {\"a\": {\"b\": 2}} is the answer.";
        assert_eq!(extract_json(text).unwrap(), json!({"a": {"b": 2}}));
    }

    #[test]
    fn extract_ignores_braces_inside_strings() {
        let text = "prefix {\"a\": \"}{\"} suffix";
        assert_eq!(extract_json(text).unwrap(), json!({"a": "}{"}));
    }

    #[test]
    fn extract_none_when_no_json() {
        assert!(extract_json("just prose, no json here").is_none());
    }

    #[test]
    fn validate_accepts_matching_object() {
        let schema = json!({
            "type": "object",
            "properties": { "name": {"type": "string"}, "count": {"type": "integer"} },
            "required": ["name"]
        });
        assert!(validate(&json!({"name": "x", "count": 3}), &schema).is_ok());
    }

    #[test]
    fn validate_rejects_missing_required() {
        let schema = json!({"type": "object", "required": ["name"]});
        let err = validate(&json!({"other": 1}), &schema).unwrap_err();
        assert!(err.contains("missing required property 'name'"), "{err}");
    }

    #[test]
    fn validate_treats_null_required_as_missing() {
        let schema = json!({"type": "object", "required": ["name"]});
        assert!(validate(&json!({"name": null}), &schema).is_err());
    }

    #[test]
    fn validate_rejects_wrong_type() {
        let schema = json!({"type": "object", "properties": {"count": {"type": "integer"}}});
        let err = validate(&json!({"count": "three"}), &schema).unwrap_err();
        assert!(err.contains("$.count"), "{err}");
    }

    #[test]
    fn validate_integer_accepts_integral_float() {
        let schema = json!({"type": "integer"});
        assert!(validate(&json!(2.0), &schema).is_ok());
        assert!(validate(&json!(2.5), &schema).is_err());
    }

    #[test]
    fn validate_enum() {
        let schema = json!({"enum": ["a", "b"]});
        assert!(validate(&json!("a"), &schema).is_ok());
        assert!(validate(&json!("c"), &schema).is_err());
    }

    #[test]
    fn validate_type_union() {
        let schema = json!({"type": ["string", "null"]});
        assert!(validate(&json!("x"), &schema).is_ok());
        assert!(validate(&json!(null), &schema).is_ok());
        assert!(validate(&json!(5), &schema).is_err());
    }

    #[test]
    fn validate_array_items() {
        let schema = json!({"type": "array", "items": {"type": "integer"}});
        assert!(validate(&json!([1, 2, 3]), &schema).is_ok());
        let err = validate(&json!([1, "two"]), &schema).unwrap_err();
        assert!(err.contains("$[1]"), "{err}");
    }

    #[test]
    fn validate_additional_properties_false() {
        let schema = json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "additionalProperties": false
        });
        assert!(validate(&json!({"a": 1}), &schema).is_ok());
        let err = validate(&json!({"a": 1, "b": 2}), &schema).unwrap_err();
        assert!(err.contains("unexpected property 'b'"), "{err}");
    }

    #[test]
    fn validate_ignores_unknown_keywords() {
        // minLength is not in our supported subset — it should be skipped,
        // not error, so the rest of the schema still validates.
        let schema = json!({"type": "string", "minLength": 100});
        assert!(validate(&json!("short"), &schema).is_ok());
    }

    #[test]
    fn build_prompt_embeds_schema_and_prompt() {
        let schema = json!({"type": "object"});
        let p = build_structured_prompt("do the thing", &schema);
        assert!(p.contains("do the thing"));
        assert!(p.contains("SINGLE JSON value"));
        assert!(p.contains("\"type\""));
    }

    #[test]
    fn escape_hatch_adds_unexpected_property() {
        let schema = json!({"type": "object", "properties": {"count": {"type": "integer"}}});
        let hatched = with_escape_hatch(&schema);
        assert_eq!(hatched["properties"]["unexpected"]["type"], "string");
        assert_eq!(hatched["properties"]["count"]["type"], "integer");
    }

    #[test]
    fn escape_hatch_does_not_clobber_existing_unexpected() {
        let schema = json!({
            "type": "object",
            "properties": {"unexpected": {"type": "array"}}
        });
        let hatched = with_escape_hatch(&schema);
        assert_eq!(hatched["properties"]["unexpected"]["type"], "array");
    }

    #[test]
    fn escape_hatch_survives_additional_properties_false() {
        let schema = json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "additionalProperties": false
        });
        let hatched = with_escape_hatch(&schema);
        // The hatch itself must validate now that it's a declared property.
        assert!(validate(&json!({"a": 1, "unexpected": "found a thing"}), &hatched).is_ok());
    }

    #[test]
    fn escape_hatch_is_noop_on_non_object_schema() {
        let schema = json!({"type": "array", "items": {"type": "integer"}});
        assert_eq!(with_escape_hatch(&schema), schema);
    }

    #[test]
    fn async_schema_note_embeds_release_to_parent_contract() {
        let schema = json!({"type": "object"});
        let note = build_async_schema_note("go build it", &schema);
        assert!(note.contains("go build it"));
        assert!(note.contains("subagent.release_to_parent"));
        assert!(note.contains("\"type\""));
    }
}
