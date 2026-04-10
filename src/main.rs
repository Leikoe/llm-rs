use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;

use llm_rs::arch;
use llm_rs::backend::Backend;
use llm_rs::backend::metal::MetalBackend;
use llm_rs::batch::Batch;
use llm_rs::cli::{Cli, Command};
use llm_rs::gguf::GgufFile;
use llm_rs::kv_pool::{self, PagedKVPool};
use llm_rs::model::Model;
use llm_rs::sampler::{Sampler, SamplerConfig};
use llm_rs::tokenizer::Tokenizer;

fn main() {
    if let Err(e) = run_main() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run_main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let gguf = GgufFile::open(&cli.model)?;
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata);
    let backend = MetalBackend::new();
    let (n_layers, kv_dim) = kv_config(&gguf);
    let weight_bytes = gguf.file_size();

    match arch::arch_name(&gguf)? {
        "llama" => {
            let model = arch::llama::LlamaModel::load(&gguf, &backend)?;
            let pool = PagedKVPool::new(&backend, n_layers, kv_dim, weight_bytes);
            dispatch(backend, model, pool, tokenizer, cli);
        }
        "qwen3" => {
            let model = arch::qwen3::Qwen3Model::load(&gguf, &backend)?;
            let pool = PagedKVPool::new(&backend, n_layers, kv_dim, weight_bytes);
            dispatch(backend, model, pool, tokenizer, cli);
        }
        other => return Err(format!("unsupported architecture: {other}").into()),
    }
    Ok(())
}

fn kv_config(gguf: &GgufFile) -> (usize, usize) {
    let m = &gguf.metadata;
    let arch = arch::arch_name(gguf).unwrap_or("llama");
    let get = |key: &str| m.get(key).and_then(|v| v.as_u32()).unwrap_or(0) as usize;
    let n_layers = get(&format!("{arch}.block_count"));
    let n_kv_heads = get(&format!("{arch}.attention.head_count_kv"));
    let n_heads = get(&format!("{arch}.attention.head_count"));
    let dim = get(&format!("{arch}.embedding_length"));
    let head_dim = m.get(&format!("{arch}.attention.key_length"))
        .and_then(|v| v.as_u32()).map(|v| v as usize)
        .unwrap_or_else(|| dim / n_heads.max(1));
    let kv_dim = n_kv_heads.max(1) * head_dim;
    (n_layers, kv_dim)
}

fn dispatch<B: Backend + Send + 'static, M: Model<B> + Send + 'static>(
    backend: B,
    mut model: M,
    mut pool: PagedKVPool<B>,
    tokenizer: Tokenizer,
    cli: Cli,
) where B::Buffer: Send {
    match cli.command {
        Command::Complete { prompt, max_tokens, sampling } => {
            run_completion(&backend, &mut model, &mut pool, &tokenizer, &prompt, max_tokens, sampling.to_config());
        }
        Command::Chat { system, sampling } => {
            run_chat(&backend, &mut model, &mut pool, &tokenizer, system.as_deref(), sampling.to_config());
        }
        Command::Serve { port } => {
            llm_rs::serve::start(model, backend, pool, Arc::new(tokenizer), port);
        }
    }
}

fn run_completion<B: Backend, M: Model<B>>(
    backend: &B,
    model: &mut M,
    pool: &mut PagedKVPool<B>,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    sampler_config: SamplerConfig,
) {
    let mut sampler = Sampler::new(sampler_config);
    let mut block_table: Vec<u32> = Vec::new();
    let mut pos: usize = 0;

    let mut tokens = vec![tokenizer.bos_id];
    tokens.extend(tokenizer.encode(prompt));
    eprintln!("Prompt tokens: {}", tokens.len());

    // Prefill
    kv_pool::ensure_blocks(&mut block_table, pool, pos + tokens.len());
    let batch = Batch::single(backend, &tokens, pos, &block_table, pool.block_size);
    let start = Instant::now();
    model.forward(backend, pool, &batch);
    backend.sync();
    let prefill_time = start.elapsed();
    eprintln!(
        "Prefill: {} tokens in {:.2}s ({:.1} tok/s)",
        tokens.len(), prefill_time.as_secs_f64(),
        tokens.len() as f64 / prefill_time.as_secs_f64()
    );
    pos += tokens.len();

    // Decode
    backend.reset_gpu_secs();
    let gen_start = Instant::now();
    let mut generated = 0;

    loop {
        let next_token = sampler.sample(backend, model.logits());

        if next_token == tokenizer.eos_id
            || tokenizer.eot_id == Some(next_token)
            || generated >= max_tokens
        {
            break;
        }

        let piece = tokenizer.decode_token(next_token);
        io::stdout().write_all(&piece).unwrap();
        io::stdout().flush().unwrap();

        generated += 1;
        kv_pool::ensure_blocks(&mut block_table, pool, pos + 1);
        let batch = Batch::single(backend, &[next_token], pos, &block_table, pool.block_size);
        model.forward(backend, pool, &batch);
        pos += 1;
    }

    backend.sync();
    println!();
    let gen_time = gen_start.elapsed();
    let wall = gen_time.as_secs_f64();
    let gpu = backend.gpu_secs_total();
    eprintln!(
        "Generated: {} tokens in {:.2}s ({:.1} tok/s)  [gpu={:.2}s overhead={:.2}s = {:.1}%]",
        generated, wall, generated as f64 / wall,
        gpu, wall - gpu, (wall - gpu) / wall * 100.0,
    );
}

fn run_chat<B: Backend, M: Model<B>>(
    backend: &B,
    model: &mut M,
    pool: &mut PagedKVPool<B>,
    tokenizer: &Tokenizer,
    system_prompt: Option<&str>,
    sampler_config: SamplerConfig,
) {
    let mut sampler = Sampler::new(sampler_config);
    let mut block_table: Vec<u32> = Vec::new();
    let mut pos: usize = 0;
    let stdin = io::stdin();

    let mut warmup = vec![tokenizer.bos_id];
    if let Some(sys) = system_prompt {
        let sys_text = format!("<|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>");
        warmup.extend(tokenizer.encode(&sys_text));
    }
    kv_pool::ensure_blocks(&mut block_table, pool, pos + warmup.len());
    let batch = Batch::single(backend, &warmup, pos, &block_table, pool.block_size);
    model.forward(backend, pool, &batch);
    pos += warmup.len();

    eprintln!("Chat mode. Type your message and press Enter. Ctrl+D to exit.\n");

    loop {
        eprint!("> ");
        io::stderr().flush().unwrap();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).unwrap() == 0 {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        let user_text = format!(
            "<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        let user_tokens = tokenizer.encode(&user_text);
        kv_pool::ensure_blocks(&mut block_table, pool, pos + user_tokens.len());
        let batch = Batch::single(backend, &user_tokens, pos, &block_table, pool.block_size);
        model.forward(backend, pool, &batch);
        pos += user_tokens.len();

        loop {
            let next_token = sampler.sample(backend, model.logits());

            if next_token == tokenizer.eos_id || tokenizer.eot_id == Some(next_token) {
                break;
            }

            let piece = tokenizer.decode_token(next_token);
            io::stdout().write_all(&piece).unwrap();
            io::stdout().flush().unwrap();

            kv_pool::ensure_blocks(&mut block_table, pool, pos + 1);
            let batch = Batch::single(backend, &[next_token], pos, &block_table, pool.block_size);
            model.forward(backend, pool, &batch);
            pos += 1;
        }

        println!();
    }
}
