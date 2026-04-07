/// Xorshift64 PRNG -- minimal, no external dependency.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng {
            state: if seed == 0 { 0xDEADBEEFCAFE } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Random f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            seed: 42,
        }
    }
}

use crate::backend::Backend;

pub struct Sampler {
    config: SamplerConfig,
    rng: Rng,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        let rng = Rng::new(config.seed);
        Sampler { config, rng }
    }

    /// Sample the next token id from on-device logits. Greedy (temp=0) stays
    /// fully on device — only 4 bytes come back. Stochastic paths still need
    /// the full vector CPU-side.
    pub fn sample<B: Backend>(&mut self, backend: &B, logits: &B::Buffer) -> u32 {
        if self.config.temperature == 0.0 {
            return backend.argmax(logits);
        }
        let mut logits = backend.read_to_vec_f32(logits);
        self.sample_cpu(&mut logits)
    }

    fn sample_cpu(&mut self, logits: &mut [f32]) -> u32 {

        // Temperature scaling
        for v in logits.iter_mut() {
            *v /= self.config.temperature;
        }

        // Top-k: zero out everything outside the top k
        if self.config.top_k > 0 && self.config.top_k < logits.len() {
            let mut indices: Vec<usize> = (0..logits.len()).collect();
            indices.select_nth_unstable_by(self.config.top_k, |&a, &b| {
                logits[b].partial_cmp(&logits[a]).unwrap()
            });
            for &i in &indices[self.config.top_k..] {
                logits[i] = f32::NEG_INFINITY;
            }
        }

        // Softmax
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in logits.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in logits.iter_mut() {
            *v /= sum;
        }

        // Top-p (nucleus sampling)
        if self.config.top_p < 1.0 {
            let mut sorted: Vec<(usize, f32)> =
                logits.iter().copied().enumerate().collect();
            sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let mut cumsum = 0.0f32;
            let mut cutoff_idx = sorted.len();
            for (i, &(_, prob)) in sorted.iter().enumerate() {
                cumsum += prob;
                if cumsum >= self.config.top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }

            // Zero out tokens beyond the nucleus
            for &(idx, _) in &sorted[cutoff_idx..] {
                logits[idx] = 0.0;
            }

            // Re-normalize
            let sum: f32 = logits.iter().sum();
            if sum > 0.0 {
                for v in logits.iter_mut() {
                    *v /= sum;
                }
            }
        }

        // Sample from the distribution
        let r = self.rng.next_f32();
        let mut cumsum = 0.0f32;
        for (i, &prob) in logits.iter().enumerate() {
            cumsum += prob;
            if r < cumsum {
                return i as u32;
            }
        }

        // Fallback: return last non-zero
        (logits.len() - 1) as u32
    }
}

