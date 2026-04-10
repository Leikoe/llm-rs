/// Throughput benchmark: sends concurrent requests to a running llm-rs server
/// and measures aggregate token throughput.
///
/// Start the server first:
///   cargo run --release -- -m models/Llama-3.2-1B-Instruct-BF16.gguf serve
///
/// Then run the benchmark:
///   cargo run --release --example bench_throughput
///   cargo run --release --example bench_throughput -- --concurrent 16 --max-tokens 64
///   cargo run --release --example bench_throughput -- --sweep

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let url = arg(&args, "--url").unwrap_or("http://localhost:8080".into());
    let max_tokens: usize = arg(&args, "--max-tokens").unwrap_or("32".into()).parse().unwrap();
    let sweep = args.iter().any(|a| a == "--sweep");

    if sweep {
        eprintln!("Sweeping concurrency levels...\n");
        eprintln!("{:>10} {:>10} {:>12} {:>12}", "concurrent", "tokens", "wall (s)", "agg tok/s");
        eprintln!("{}", "-".repeat(48));
        for &c in &[1, 2, 4, 8, 16, 32, 64, 128] {
            let (total_tokens, wall) = run_bench(&url, c, max_tokens).await;
            let tps = total_tokens as f64 / wall;
            eprintln!("{:>10} {:>10} {:>12.2} {:>12.1}", c, total_tokens, wall, tps);
        }
    } else {
        let concurrent: usize = arg(&args, "--concurrent").unwrap_or("8".into()).parse().unwrap();
        let (total_tokens, wall) = run_bench(&url, concurrent, max_tokens).await;
        let tps = total_tokens as f64 / wall;
        eprintln!(
            "\nconcurrent={concurrent}  tokens={total_tokens}  wall={wall:.2}s  throughput={tps:.1} tok/s"
        );
    }
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
}

async fn run_bench(url: &str, concurrent: usize, max_tokens: usize) -> (usize, f64) {
    let total_tokens = Arc::new(AtomicUsize::new(0));
    let client = reqwest::Client::new();
    let endpoint = format!("{url}/v1/chat/completions");

    let start = Instant::now();
    let mut tasks = Vec::new();

    for i in 0..concurrent {
        let client = client.clone();
        let endpoint = endpoint.clone();
        let total_tokens = total_tokens.clone();

        tasks.push(tokio::spawn(async move {
            let prompt = format!("Count from {i} to 1000. Go:");
            let body = serde_json::json!({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": true
            });

            let resp = match client.post(&endpoint).json(&body).send().await {
                Ok(r) => r,
                Err(e) => { eprintln!("Request {i} failed: {e}"); return; }
            };

            let mut tokens = 0usize;
            let text = resp.text().await.unwrap_or_default();
            for line in text.lines() {
                if !line.starts_with("data: ") { continue; }
                let data = &line["data: ".len()..];
                if data == "[DONE]" { break; }
                if data.contains("\"content\"") {
                    tokens += 1;
                }
            }
            total_tokens.fetch_add(tokens, Ordering::Relaxed);
        }));
    }

    for t in tasks { let _ = t.await; }
    let wall = start.elapsed().as_secs_f64();
    (total_tokens.load(Ordering::Relaxed), wall)
}
