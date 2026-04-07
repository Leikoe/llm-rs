//! LLaMA family (LLaMA 2/3, Mistral): GQA + RoPE + SwiGLU, no QK-norm.

use crate::arch::Loader;
use crate::backend::Backend;
use crate::model::{Ffn, Layer, Mixer, ModelConfig, Transformer};

pub fn build<B: Backend>(l: &mut Loader<B>, config: ModelConfig) -> Transformer<B> {
    let token_embedding = l.req("token_embd.weight");
    let output_weight = l.opt("output.weight").unwrap_or_else(|| l.req("token_embd.weight"));
    let output_norm = l.req("output_norm.weight");

    let layers = (0..config.n_layers)
        .map(|i| Layer {
            attn_norm: l.req(&format!("blk.{i}.attn_norm.weight")),
            mixer: Mixer::Gqa {
                wq: l.req(&format!("blk.{i}.attn_q.weight")),
                wk: l.req(&format!("blk.{i}.attn_k.weight")),
                wv: l.req(&format!("blk.{i}.attn_v.weight")),
                wo: l.req(&format!("blk.{i}.attn_output.weight")),
                q_norm: None,
                k_norm: None,
            },
            ffn_norm: l.req(&format!("blk.{i}.ffn_norm.weight")),
            ffn: Ffn::SwiGlu {
                w1: l.req(&format!("blk.{i}.ffn_gate.weight")),
                w2: l.req(&format!("blk.{i}.ffn_down.weight")),
                w3: l.req(&format!("blk.{i}.ffn_up.weight")),
            },
        })
        .collect();

    Transformer { config, token_embedding, output_norm, output_weight, layers }
}
