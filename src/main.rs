use std::io::{self, BufRead, Write};
use std::time::Instant;

use clap::Parser;

#[cfg(not(all(target_os = "macos", feature = "metal")))]
use llm_rs::backend::cpu::CpuBackend;
#[cfg(all(target_os = "macos", feature = "metal"))]
use llm_rs::backend::metal::MetalBackend;
use llm_rs::backend::Backend;
use llm_rs::cli::{Cli, Command};
use llm_rs::gguf::GgufFile;
use llm_rs::model::llama::LlamaModel;
use llm_rs::sampler::{Sampler, SamplerConfig};

use llm_rs::tokenizer::Tokenizer;

fn main() {
    let cli = Cli::parse();

    let gguf = GgufFile::open(&cli.model).expect("failed to open GGUF file");
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata);

    #[cfg(all(target_os = "macos", feature = "metal"))]
    let backend = MetalBackend::new();
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    let backend = CpuBackend::new();

    let mut model = LlamaModel::from_gguf(&gguf, &backend);

    match cli.command {
        Command::Complete { prompt, max_tokens, sampling } => {
            run_completion(&backend, &mut model, &tokenizer, &prompt, max_tokens, sampling.to_config());
        }
        Command::Chat { system, sampling } => {
            run_chat(&backend, &mut model, &tokenizer, system.as_deref(), sampling.to_config());
        }
    }
}

fn run_completion(
    backend: &dyn Backend,
    model: &mut LlamaModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    sampler_config: SamplerConfig,
) {
    let mut sampler = Sampler::new(sampler_config);

    let mut tokens = vec![tokenizer.bos_id];
    tokens.extend(tokenizer.encode(prompt));
    eprintln!("Prompt tokens: {}", tokens.len());

    // Prefill: batch all prompt tokens except the last through GEMM.
    // Then forward the last token to get the first set of logits.
    let start = Instant::now();
    if tokens.len() > 1 {
        model.forward_prefill(backend, &tokens[..tokens.len() - 1], 0);
    }
    let last_prompt_pos = tokens.len() - 1;
    let mut logits = model.forward(backend, tokens[last_prompt_pos], last_prompt_pos);
    let mut pos = tokens.len();

    let prefill_time = start.elapsed();
    eprintln!(
        "Prefill: {} tokens in {:.2}s ({:.1} tok/s)",
        tokens.len(),
        prefill_time.as_secs_f64(),
        tokens.len() as f64 / prefill_time.as_secs_f64()
    );

    // Generate
    let gen_start = Instant::now();
    let mut generated = 0;

    loop {
        let next_token = sampler.sample(&mut logits);

        if next_token == tokenizer.eos_id || tokenizer.eot_id == Some(next_token) || generated >= max_tokens {
            break;
        }

        let piece = tokenizer.decode_token(next_token);
        io::stdout().write_all(&piece).unwrap();
        io::stdout().flush().unwrap();

        generated += 1;
        logits = model.forward(backend, next_token, pos);
        pos += 1;
    }

    println!();
    let gen_time = gen_start.elapsed();
    eprintln!(
        "Generated: {} tokens in {:.2}s ({:.1} tok/s)",
        generated,
        gen_time.as_secs_f64(),
        generated as f64 / gen_time.as_secs_f64()
    );
}

fn run_chat(
    backend: &dyn Backend,
    model: &mut LlamaModel,
    tokenizer: &Tokenizer,
    system_prompt: Option<&str>,
    sampler_config: SamplerConfig,
) {
    let mut sampler = Sampler::new(sampler_config);
    let stdin = io::stdin();
    let mut pos = 0;

    // BOS token
    model.forward_kv_only(backend, tokenizer.bos_id, pos);
    pos += 1;

    // System prompt
    if let Some(sys) = system_prompt {
        let sys_text = format!(
            "<|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>"
        );
        let sys_tokens = tokenizer.encode(&sys_text);
        for &token in &sys_tokens {
            model.forward_kv_only(backend, token, pos);
            pos += 1;
        }
    }

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

        // Format with LLaMA 3 chat template
        let user_text = format!(
            "<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        let user_tokens = tokenizer.encode(&user_text);

        // Prefill user message (all but last token) with batched GEMM
        if user_tokens.len() > 1 {
            model.forward_prefill(backend, &user_tokens[..user_tokens.len() - 1], pos);
            pos += user_tokens.len() - 1;
        }

        // Forward last user token to get initial logits
        let mut logits = model.forward(backend, *user_tokens.last().unwrap(), pos);
        pos += 1;

        // Generate response
        loop {
            let next_token = sampler.sample(&mut logits);

            if next_token == tokenizer.eos_id || tokenizer.eot_id == Some(next_token) {
                break;
            }

            let piece = tokenizer.decode_token(next_token);
            io::stdout().write_all(&piece).unwrap();
            io::stdout().flush().unwrap();

            logits = model.forward(backend, next_token, pos);
            pos += 1;
        }

        println!();
    }
}
