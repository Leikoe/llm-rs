mod api;
mod engine;
mod scheduler;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};

use crate::backend::Backend;
use crate::kv_pool::PagedKVPool;
use crate::model::Model;
use crate::tokenizer::Tokenizer;

pub use engine::ServerEvent;

pub fn start<B, M>(
    model: M,
    backend: B,
    pool: PagedKVPool<B>,
    tokenizer: Arc<Tokenizer>,
    port: u16,
) where
    B: Backend + Send + 'static,
    B::Buffer: Send,
    M: Model<B> + Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel::<engine::IncomingRequest>(256);

    let tok = tokenizer.clone();
    std::thread::spawn(move || {
        engine::run(model, backend, pool, tok, rx);
    });

    let state = api::AppState { tx, tokenizer };

    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let app = Router::new()
            .route("/v1/chat/completions", post(api::chat_completions))
            .route("/v1/models", get(api::list_models))
            .with_state(state);
        let addr = SocketAddr::from(([0, 0, 0, 0], port));
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
        eprintln!("Serving on http://{addr}");
        axum::serve(listener, app).await.unwrap();
    });
}
