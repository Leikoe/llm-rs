use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "llm-rs", about = "High-performance LLM inference")]
pub struct Cli {
    /// Path to GGUF model file
    #[arg(short, long)]
    pub model: PathBuf,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Single-shot text completion
    Complete {
        /// Input prompt
        #[arg(short, long)]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(short = 'n', long, default_value = "256")]
        max_tokens: usize,

        #[command(flatten)]
        sampling: SamplingArgs,
    },
    /// Interactive chat mode
    Chat {
        /// System prompt
        #[arg(short, long)]
        system: Option<String>,

        #[command(flatten)]
        sampling: SamplingArgs,
    },
    /// Start OpenAI-compatible HTTP server
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },
}

#[derive(clap::Args)]
pub struct SamplingArgs {
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,
    #[arg(long, default_value = "40")]
    pub top_k: usize,
    #[arg(long, default_value = "0.9")]
    pub top_p: f32,
    #[arg(long, default_value = "42")]
    pub seed: u64,
}

impl SamplingArgs {
    pub fn to_config(&self) -> crate::sampler::SamplerConfig {
        crate::sampler::SamplerConfig {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            seed: self.seed,
        }
    }
}
