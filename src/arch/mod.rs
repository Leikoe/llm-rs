//! Per-architecture model loaders. Each module defines a concrete model type
//! that implements `Model<B>`.

pub mod llama;
pub mod qwen3;

use std::fmt;

use crate::backend::Backend;
use crate::gguf::GgufFile;

#[derive(Debug)]
pub struct LoadError(String);

impl LoadError {
    pub fn missing(key: &str) -> Self {
        LoadError(format!("missing metadata key: {key}"))
    }
    pub fn unsupported_arch(arch: &str) -> Self {
        LoadError(format!("unsupported architecture: {arch}"))
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for LoadError {}

/// Read the architecture string from GGUF metadata.
pub fn arch_name(gguf: &GgufFile) -> Result<&str, LoadError> {
    gguf.metadata.get("general.architecture")
        .and_then(|v| v.as_str())
        .ok_or_else(|| LoadError::missing("general.architecture"))
}

/// Tensor upload helper used by all architecture loaders.
pub struct Loader<'a, B: Backend> {
    gguf: &'a GgufFile,
    backend: &'a B,
    bytes: usize,
    start: std::time::Instant,
}

impl<'a, B: Backend> Loader<'a, B> {
    pub fn new(gguf: &'a GgufFile, backend: &'a B) -> Self {
        Loader { gguf, backend, bytes: 0, start: std::time::Instant::now() }
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

    pub fn print_stats(&self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let gb = self.bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        eprintln!("Uploaded model weights: {gb:.2} GB in {elapsed:.3}s ({:.1} GB/s)", gb / elapsed);
    }
}
