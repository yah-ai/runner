//! JSON-RPC 2.0 wire types.
//!
//! Just the request / response / error shapes — dispatch lives in
//! [`crate::McpServer`]. We model `id` as `Option<Value>` so a missing
//! id (notifications) is distinguishable from `id: null` (response to
//! a parse-error per the JSON-RPC spec) without bespoke deserializers.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Inbound frame. `params` is optional — `tools/list` and `ping`
/// legitimately omit it.
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcRequest {
    #[serde(rename = "jsonrpc")]
    pub _jsonrpc: String,
    /// Absent for notifications. `serde` distinguishes "no field" from
    /// "field set to null", which matters here — a notification has no
    /// id at all (so no response should be sent), while an
    /// otherwise-malformed request without an id still gets an error
    /// response with `id: null`.
    #[serde(default)]
    pub id: Option<Value>,
    pub method: String,
    pub params: Option<Value>,
}

/// Outbound frame. Either `result` or `error` is populated, never both.
/// `jsonrpc` is owned (not `&'static str`) so the type round-trips
/// through `serde_json::from_str` without forcing the input lifetime
/// to `'static` — useful for tests that deserialize server responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    pub fn ok(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }
    pub fn error(id: Value, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }
}

/// JSON-RPC 2.0 error object. The MCP spec layers no extra error
/// codes on top of the standard set used here, so we keep the shape
/// minimal — `data` exists for future structured detail (e.g. a
/// sidecar approval-channel reply with the user's choice text) but is
/// `None` for everything we emit today.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code: code as i32,
            message: message.into(),
            data: None,
        }
    }
}

/// Standard JSON-RPC 2.0 error codes. Anything in `-32000..=-32099`
/// is reserved for application use; we don't carve out any custom
/// codes today.
#[derive(Debug, Copy, Clone)]
#[repr(i32)]
pub enum ErrorCode {
    ParseError = -32700,
    InvalidRequest = -32600,
    MethodNotFound = -32601,
    InvalidParams = -32602,
    InternalError = -32603,
}
