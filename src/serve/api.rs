use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::sampler::SamplerConfig;
use crate::tokenizer::Tokenizer;

use super::engine::{IncomingRequest, ServerEvent};

// ── State ──────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub tx: tokio::sync::mpsc::Sender<IncomingRequest>,
    pub tokenizer: Arc<Tokenizer>,
}

// ── OpenAI types ───────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
    pub seed: Option<u64>,
}

#[derive(Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
struct ChatResponse {
    id: &'static str,
    object: &'static str,
    choices: Vec<Choice>,
}

#[derive(Serialize)]
struct Choice {
    index: usize,
    message: Option<ResponseMessage>,
    delta: Option<Delta>,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct ResponseMessage {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

// ── Routes ─────────────────────────────────────────────────────────────

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, StatusCode> {
    let tokenizer = &state.tokenizer;

    // Format messages as LLaMA 3 chat template
    let mut text = String::new();
    for msg in &req.messages {
        text.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            msg.role, msg.content
        ));
    }
    text.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");

    let mut prompt_tokens = vec![tokenizer.bos_id];
    prompt_tokens.extend(tokenizer.encode(&text));

    let mut stop_tokens = vec![tokenizer.eos_id];
    if let Some(eot) = tokenizer.eot_id {
        stop_tokens.push(eot);
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    let incoming = IncomingRequest {
        prompt_tokens,
        sampler_config: SamplerConfig {
            temperature: req.temperature.unwrap_or(0.7),
            top_k: 40,
            top_p: req.top_p.unwrap_or(0.9),
            seed: req.seed.unwrap_or(42),
        },
        max_tokens: req.max_tokens.unwrap_or(2048),
        stop_tokens,
        tx,
    };

    state.tx.try_send(incoming).map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?;

    if req.stream {
        Ok(Sse::new(EventStream { rx, done: false }).into_response())
    } else {
        // Collect all tokens
        let text = collect_response(rx).await;
        Ok(Json(ChatResponse {
            id: "cmpl",
            object: "chat.completion",
            choices: vec![Choice {
                index: 0,
                message: Some(ResponseMessage { role: "assistant", content: text }),
                delta: None,
                finish_reason: Some("stop"),
            }],
        }).into_response())
    }
}

pub async fn list_models() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "data": [{"id": "default", "object": "model"}]
    }))
}

// ── SSE stream ─────────────────────────────────────────────────────────

struct EventStream {
    rx: tokio::sync::mpsc::UnboundedReceiver<ServerEvent>,
    done: bool,
}

impl futures_core::Stream for EventStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done { return Poll::Ready(None); }
        match self.rx.poll_recv(cx) {
            Poll::Ready(Some(ServerEvent::Token(bytes))) => {
                let content = String::from_utf8_lossy(&bytes).into_owned();
                let chunk = serde_json::json!({
                    "id": "cmpl",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": null}]
                });
                Poll::Ready(Some(Ok(Event::default().data(chunk.to_string()))))
            }
            Poll::Ready(Some(ServerEvent::Done)) => {
                self.done = true;
                let chunk = serde_json::json!({
                    "id": "cmpl",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                });
                Poll::Ready(Some(Ok(Event::default().data(chunk.to_string()))))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

async fn collect_response(mut rx: tokio::sync::mpsc::UnboundedReceiver<ServerEvent>) -> String {
    let mut text = String::new();
    while let Some(event) = rx.recv().await {
        match event {
            ServerEvent::Token(bytes) => text.push_str(&String::from_utf8_lossy(&bytes)),
            ServerEvent::Done => break,
        }
    }
    text
}
