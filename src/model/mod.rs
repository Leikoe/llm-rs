pub mod llama;

use std::collections::HashMap;

use crate::gguf::metadata::MetadataValue;

/// Model hyperparameters parsed from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
}

impl ModelConfig {
    pub fn from_gguf_metadata(metadata: &HashMap<String, MetadataValue>) -> Self {
        let architecture = metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("llama")
            .to_string();

        let arch = &architecture;

        // GGUF doesn't store vocab_size directly; derive it from the token table length.
        // GGUF doesn't store vocab_size directly; derive it from the token table length.
        let vocab_size = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_string_array())
            .expect("missing tokenizer.ggml.tokens")
            .len();

        let dim = get_u32(metadata, &format!("{arch}.embedding_length"))
            .expect("missing embedding_length") as usize;

        let n_layers = get_u32(metadata, &format!("{arch}.block_count"))
            .expect("missing block_count") as usize;

        let n_heads = get_u32(metadata, &format!("{arch}.attention.head_count"))
            .expect("missing head_count") as usize;

        let n_kv_heads = get_u32(metadata, &format!("{arch}.attention.head_count_kv"))
            .unwrap_or(n_heads as u32) as usize;

        // Qwen3 stores head_dim explicitly (key_length); LLaMA derives it from dim/n_heads.
        let head_dim = get_u32(metadata, &format!("{arch}.attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(dim / n_heads);

        let hidden_dim = get_u32(metadata, &format!("{arch}.feed_forward_length"))
            .expect("missing feed_forward_length") as usize;

        let norm_eps = get_f32(metadata, &format!("{arch}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);

        let rope_theta = get_f32(metadata, &format!("{arch}.rope.freq_base"))
            .unwrap_or(10000.0);

        // Cap context length to avoid huge KV cache allocation during development.
        // The full context can be used once we have paged attention.
        let max_seq_len = (get_u32(metadata, &format!("{arch}.context_length"))
            .unwrap_or(4096) as usize)
            .min(4096);

        ModelConfig {
            architecture,
            vocab_size,
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            hidden_dim,
            norm_eps,
            rope_theta,
            max_seq_len,
        }
    }
}

fn get_u32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Option<u32> {
    metadata.get(key).and_then(|v| match v {
        MetadataValue::U32(n) => Some(*n),
        MetadataValue::I32(n) => Some(*n as u32),
        MetadataValue::U64(n) => Some(*n as u32),
        MetadataValue::I64(n) => Some(*n as u32),
        _ => None,
    })
}

fn get_f32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Option<f32> {
    metadata.get(key).and_then(|v| match v {
        MetadataValue::F32(n) => Some(*n),
        MetadataValue::F64(n) => Some(*n as f32),
        _ => None,
    })
}
