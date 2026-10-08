//! The HTTP substrate a provider cell runs over (R950-F5, W357 §3).
//!
//! One trait, [`HttpClient`]: send a [`HttpRequest`], get back a status, the
//! response headers and a streaming byte body. Nothing in it is
//! reqwest-shaped, so a wasm or iOS host can hand a runner its own substrate
//! (`fetch`, `URLSession`) and build core with `--no-default-features`.
//!
//! [`ReqwestClient`] is the default implementation, behind the default
//! `reqwest` feature.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures_core::Stream;

/// HTTP method. Only the verbs a provider cell actually issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
}

/// One outgoing request. The body is fully buffered — every provider
/// request is a JSON document; only responses stream.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: HttpMethod,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<Bytes>,
    /// Whole-request deadline. `None` leaves it to the client's default.
    pub timeout: Option<Duration>,
}

impl HttpRequest {
    pub fn get(url: impl Into<String>) -> Self {
        Self {
            method: HttpMethod::Get,
            url: url.into(),
            headers: Vec::new(),
            body: None,
            timeout: None,
        }
    }

    /// A `POST` with `body` serialized as JSON and the matching
    /// `content-type` header.
    pub fn post_json(url: impl Into<String>, body: &serde_json::Value) -> Self {
        Self {
            method: HttpMethod::Post,
            url: url.into(),
            headers: vec![("content-type".into(), "application/json".into())],
            body: Some(Bytes::from(body.to_string())),
            timeout: None,
        }
    }

    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn bearer_auth(self, token: &str) -> Self {
        self.header("authorization", format!("Bearer {token}"))
    }
}

/// A transport failure: DNS, TLS, connect, a reset mid-body. HTTP error
/// statuses are not errors at this layer — they arrive as an
/// [`HttpResponse`] with that status.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct HttpError(pub String);

/// The response body, chunk by chunk as the substrate delivers it.
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, HttpError>> + Send>>;

/// A response whose headers have arrived; the body has not been read.
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: ByteStream,
}

impl std::fmt::Debug for HttpResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpResponse")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .finish_non_exhaustive()
    }
}

impl HttpResponse {
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// First value of header `name`, compared case-insensitively.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Read the whole body.
    pub async fn bytes(self) -> Result<Vec<u8>, HttpError> {
        let mut body = self.body;
        let mut out = Vec::new();
        while let Some(chunk) = NextChunk(&mut body).await {
            out.extend_from_slice(&chunk?);
        }
        Ok(out)
    }

    /// Read the whole body as (lossy) UTF-8.
    pub async fn text(self) -> Result<String, HttpError> {
        let bytes = self.bytes().await?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Read the whole body and parse it as JSON.
    pub async fn json<T: serde::de::DeserializeOwned>(self) -> Result<T, HttpError> {
        let bytes = self.bytes().await?;
        serde_json::from_slice(&bytes).map_err(|e| HttpError(format!("invalid JSON body: {e}")))
    }
}

/// `StreamExt::next` without pulling futures-util into core.
struct NextChunk<'a>(&'a mut ByteStream);

impl Future for NextChunk<'_> {
    type Output = Option<Result<Bytes, HttpError>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.0.as_mut().poll_next(cx)
    }
}

/// The HTTP substrate. Implementations must be cheap to share: a runner
/// holds one as `Arc<dyn HttpClient>` for every session it serves.
#[async_trait]
pub trait HttpClient: Send + Sync {
    /// Send `request`. Resolves once the status line and headers are in;
    /// the body streams from the returned [`HttpResponse::body`].
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError>;
}

#[cfg(feature = "reqwest")]
pub use self::reqwest_client::ReqwestClient;

#[cfg(feature = "reqwest")]
mod reqwest_client {
    use super::*;

    /// [`HttpClient`] over a `reqwest::Client`.
    #[derive(Debug, Clone, Default)]
    pub struct ReqwestClient {
        client: reqwest::Client,
    }

    impl ReqwestClient {
        pub fn new(client: reqwest::Client) -> Self {
            Self { client }
        }
    }

    #[async_trait]
    impl HttpClient for ReqwestClient {
        async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
            let method = match request.method {
                HttpMethod::Get => reqwest::Method::GET,
                HttpMethod::Post => reqwest::Method::POST,
            };
            let mut req = self.client.request(method, &request.url);
            for (name, value) in &request.headers {
                req = req.header(name, value);
            }
            if let Some(body) = request.body {
                req = req.body(body);
            }
            if let Some(timeout) = request.timeout {
                req = req.timeout(timeout);
            }
            let resp = req.send().await.map_err(|e| HttpError(e.to_string()))?;
            let status = resp.status().as_u16();
            let headers = resp
                .headers()
                .iter()
                .filter_map(|(k, v)| v.to_str().ok().map(|v| (k.as_str().to_string(), v.to_string())))
                .collect();
            let body = MapErr(Box::pin(resp.bytes_stream()));
            Ok(HttpResponse {
                status,
                headers,
                body: Box::pin(body),
            })
        }
    }

    type ReqwestStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

    /// Maps reqwest's body errors onto [`HttpError`].
    struct MapErr(ReqwestStream);

    impl Stream for MapErr {
        type Item = Result<Bytes, HttpError>;

        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            self.0
                .as_mut()
                .poll_next(cx)
                .map(|item| item.map(|r| r.map_err(|e| HttpError(e.to_string()))))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Chunks(Vec<Result<Bytes, HttpError>>);

    impl Stream for Chunks {
        type Item = Result<Bytes, HttpError>;
        fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Ready(if self.0.is_empty() { None } else { Some(self.0.remove(0)) })
        }
    }

    fn response(chunks: Vec<Result<Bytes, HttpError>>) -> HttpResponse {
        HttpResponse {
            status: 429,
            headers: vec![("Retry-After".into(), "7".into())],
            body: Box::pin(Chunks(chunks)),
        }
    }

    #[test]
    fn header_lookup_is_case_insensitive() {
        let resp = response(vec![]);
        assert_eq!(resp.header("retry-after"), Some("7"));
        assert!(!resp.is_success());
    }

    #[tokio::test]
    async fn json_concatenates_chunks() {
        let resp = response(vec![Ok(Bytes::from_static(b"{\"a\":")), Ok(Bytes::from_static(b"1}"))]);
        let v: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(v["a"], 1);
    }

    #[tokio::test]
    async fn body_error_surfaces() {
        let resp = response(vec![Ok(Bytes::from_static(b"x")), Err(HttpError("reset".into()))]);
        assert_eq!(resp.text().await.unwrap_err().0, "reset");
    }

    #[test]
    fn post_json_sets_content_type_and_bearer() {
        let req = HttpRequest::post_json("http://x", &serde_json::json!({"k": 1})).bearer_auth("t");
        assert_eq!(req.method, HttpMethod::Post);
        assert!(req.headers.contains(&("authorization".into(), "Bearer t".into())));
        assert_eq!(req.body.as_deref(), Some(&b"{\"k\":1}"[..]));
    }
}
