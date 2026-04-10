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
    indices: Vec<usize>,
    sorted: Vec<(usize, f32)>,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        let rng = Rng::new(config.seed);
        Sampler { config, rng, indices: Vec::new(), sorted: Vec::new() }
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
        let inv_temp = 1.0 / self.config.temperature;
        for v in logits.iter_mut() {
            *v *= inv_temp;
        }

        // Top-k
        if self.config.top_k > 0 && self.config.top_k < logits.len() {
            let k = self.config.top_k;
            self.indices.clear();
            self.indices.extend(0..logits.len());
            self.indices.select_nth_unstable_by(k, |&a, &b| {
                logits[b].partial_cmp(&logits[a]).unwrap()
            });
            for &i in &self.indices[k..] {
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
        let inv_sum = 1.0 / sum;
        for v in logits.iter_mut() {
            *v *= inv_sum;
        }

        // Top-p (nucleus sampling)
        if self.config.top_p < 1.0 {
            self.sorted.clear();
            self.sorted.extend(logits.iter().copied().enumerate());
            self.sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let mut cumsum = 0.0f32;
            let mut cutoff = self.sorted.len();
            for (i, &(_, prob)) in self.sorted.iter().enumerate() {
                cumsum += prob;
                if cumsum >= self.config.top_p {
                    cutoff = i + 1;
                    break;
                }
            }

            for &(idx, _) in &self.sorted[cutoff..] {
                logits[idx] = 0.0;
            }

            let sum: f32 = logits.iter().sum();
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for v in logits.iter_mut() {
                    *v *= inv;
                }
            }
        }

        // Sample
        let r = self.rng.next_f32();
        let mut cumsum = 0.0f32;
        for (i, &prob) in logits.iter().enumerate() {
            cumsum += prob;
            if r < cumsum {
                return i as u32;
            }
        }
        (logits.len() - 1) as u32
    }
}
