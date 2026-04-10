//! Per-architecture loaders. Each module is a declarative recipe that
//! reads tensors from a GGUF file and assembles a `Transformer<B>`.

mod llama;
mod qwen3;

use crate::backend::Backend;
use crate::gguf::GgufFile;
use crate::model::{ConfigError, ModelConfig, Transformer};

pub fn load<B: Backend>(gguf: &GgufFile, backend: &B) -> Result<Transformer<B>, ConfigError> {
    let config = ModelConfig::from_gguf_metadata(&gguf.metadata)?;

    eprintln!(
        "{} model: dim={}, layers={}, heads={}, kv_heads={}, vocab={}",
        config.architecture,
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size,
    );

    let upload_start = std::time::Instant::now();
    let mut loader = Loader::new(gguf, backend);

    let model = match config.architecture.as_str() {
        "llama" => llama::build(&mut loader, config),
        "qwen3" => qwen3::build(&mut loader, config),
        other => return Err(ConfigError::unsupported_arch(other)),
    };

    let elapsed = upload_start.elapsed().as_secs_f64();
    let gb = loader.bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "Uploaded model weights to backend: {:.2} GB in {:.3}s ({:.1} GB/s)",
        gb, elapsed, gb / elapsed,
    );

    Ok(model)
}

/// Tensor upload helper. Not generic over architecture — every recipe
/// pulls weights through the same two methods.
pub struct Loader<'a, B: Backend> {
    gguf: &'a GgufFile,
    backend: &'a B,
    bytes: usize,
}

impl<'a, B: Backend> Loader<'a, B> {
    fn new(gguf: &'a GgufFile, backend: &'a B) -> Self {
        Loader { gguf, backend, bytes: 0 }
    }

    pub fn opt(&mut self, name: &str) -> Option<B::Buffer> {
        self.gguf.tensor_view(name).ok().map(|tv| {
            self.bytes += tv.data.len();
            self.backend.upload_tensor(&tv)
        })
    }

    pub fn req(&mut self, name: &str) -> B::Buffer {
        self.opt(name).unwrap_or_else(|| panic!("missing tensor: {name}"))
    }
}
