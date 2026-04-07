use std::io::{self, BufRead, Write};
use std::time::Instant;

use clap::Parser;

use llm_rs::backend::Backend;
use llm_rs::backend::metal::MetalBackend;
use llm_rs::cli::{Cli, Command};
use llm_rs::gguf::GgufFile;
use llm_rs::arch;
use llm_rs::model::{Session, Transformer};
use llm_rs::sampler::{Sampler, SamplerConfig};

use llm_rs::tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let gguf = GgufFile::open(&cli.model)?;
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata);

    // One match-and-dispatch at startup. Future backends slot in here.
    let backend = MetalBackend::new();
    run(&backend, &gguf, &tokenizer, cli);
    Ok(())
}

fn run<B: Backend>(backend: &B, gguf: &GgufFile, tokenizer: &Tokenizer, cli: Cli) {
    let model = arch::load(gguf, backend);

    match cli.command {
        Command::Complete {
            prompt,
            max_tokens,
            sampling,
        } => {
            let mut session = Session::new(backend, &model.config);
            run_completion(
                backend,
                &model,
                &mut session,
                tokenizer,
                &prompt,
                max_tokens,
                sampling.to_config(),
            );
        }
        Command::Chat { system, sampling } => {
            let mut session = Session::new(backend, &model.config);
            run_chat(
                backend,
                &model,
                &mut session,
                tokenizer,
                system.as_deref(),
                sampling.to_config(),
            );
        }
    }
}

fn run_completion<B: Backend>(
    backend: &B,
    model: &Transformer<B>,
    session: &mut Session<B>,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    sampler_config: SamplerConfig,
) {
    let mut sampler = Sampler::new(sampler_config);

    let mut tokens = vec![tokenizer.bos_id];
    tokens.extend(tokenizer.encode(prompt));
    eprintln!("Prompt tokens: {}", tokens.len());

    // Prefill all prompt tokens in one call; logits come back for the last token.
    let start = Instant::now();
    model.forward(backend, session, &tokens, true);

    let prefill_time = start.elapsed();
    eprintln!(
        "Prefill: {} tokens in {:.2}s ({:.1} tok/s)",
        tokens.len(),
        prefill_time.as_secs_f64(),
        tokens.len() as f64 / prefill_time.as_secs_f64()
    );

    // Generate
    backend.reset_gpu_secs();
    let gen_start = Instant::now();
    let mut generated = 0;

    loop {
        let next_token = sampler.sample(backend, session.logits());

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
        model.forward(backend, session, &[next_token], true);
    }

    println!();
    let gen_time = gen_start.elapsed();
    let wall = gen_time.as_secs_f64();
    let gpu = backend.gpu_secs_total();
    eprintln!(
        "Generated: {} tokens in {:.2}s ({:.1} tok/s)  [gpu={:.2}s overhead={:.2}s = {:.1}%]",
        generated,
        wall,
        generated as f64 / wall,
        gpu,
        wall - gpu,
        (wall - gpu) / wall * 100.0,
    );
}

fn run_chat<B: Backend>(
    backend: &B,
    model: &Transformer<B>,
    session: &mut Session<B>,
    tokenizer: &Tokenizer,
    system_prompt: Option<&str>,
    sampler_config: SamplerConfig,
) {
    let mut sampler = Sampler::new(sampler_config);
    let stdin = io::stdin();

    // BOS + optional system prompt, prefilled in one call.
    let mut warmup = vec![tokenizer.bos_id];
    if let Some(sys) = system_prompt {
        let sys_text = format!("<|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>");
        warmup.extend(tokenizer.encode(&sys_text));
    }
    model.forward(backend, session, &warmup, false);

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

        // Prefill user message in one call; logits come back for the last token.
        model.forward(backend, session, &user_tokens, true);

        // Generate response
        loop {
            let next_token = sampler.sample(backend, session.logits());

            if next_token == tokenizer.eos_id || tokenizer.eot_id == Some(next_token) {
                break;
            }

            let piece = tokenizer.decode_token(next_token);
            io::stdout().write_all(&piece).unwrap();
            io::stdout().flush().unwrap();

            model.forward(backend, session, &[next_token], true);
        }

        println!();
    }
}
