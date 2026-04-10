use std::sync::Arc;

use crate::backend::Backend;
use crate::batch::Batch;
use crate::kv_pool::PagedKVPool;
use crate::model::Model;
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::Tokenizer;

use super::scheduler::Scheduler;

// ── Types ──────────────────────────────────────────────────────────────

pub type ReqId = usize;

pub enum ServerEvent {
    Token(Vec<u8>),
    Done,
}

pub struct IncomingRequest {
    pub prompt_tokens: Vec<u32>,
    pub sampler_config: SamplerConfig,
    pub max_tokens: usize,
    pub stop_tokens: Vec<u32>,
    pub tx: tokio::sync::mpsc::UnboundedSender<ServerEvent>,
}

pub struct Request {
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub computed: usize,
    pub block_table: Vec<u32>,
    pub sampler: Sampler,
    pub max_tokens: usize,
    pub stop_tokens: Vec<u32>,
    tx: tokio::sync::mpsc::UnboundedSender<ServerEvent>,
}

impl Request {
    fn from_incoming(req: IncomingRequest) -> Self {
        Request {
            prompt_tokens: req.prompt_tokens,
            generated_tokens: Vec::new(),
            computed: 0,
            block_table: Vec::new(),
            sampler: Sampler::new(req.sampler_config),
            max_tokens: req.max_tokens,
            stop_tokens: req.stop_tokens,
            tx: req.tx,
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens.len()
    }

    fn tokens_for_step(&self, n: usize) -> Vec<u32> {
        if self.computed < self.prompt_tokens.len() {
            self.prompt_tokens[self.computed..self.computed + n].to_vec()
        } else {
            vec![*self.generated_tokens.last().unwrap()]
        }
    }
}

// ── Request Table (stable indices, MRV2 pattern) ───────────────────────

pub struct RequestTable {
    pub slots: Vec<Option<Request>>,
    free: Vec<usize>,
}

impl RequestTable {
    fn new(capacity: usize) -> Self {
        RequestTable {
            slots: (0..capacity).map(|_| None).collect(),
            free: (0..capacity).rev().collect(),
        }
    }

    fn add(&mut self, req: Request) -> ReqId {
        let id = self.free.pop().expect("request table full");
        self.slots[id] = Some(req);
        id
    }

    fn remove(&mut self, id: ReqId) {
        self.slots[id] = None;
        self.free.push(id);
    }
}

// ── Batch building ─────────────────────────────────────────────────────

fn build_batch<B: Backend>(
    scheduled: &[(ReqId, usize)],
    reqs: &RequestTable,
    pool: &PagedKVPool<B>,
    backend: &B,
) -> Batch<B> {
    let mut tokens = Vec::new();
    let mut positions = Vec::new();
    let mut query_starts = vec![0u32];
    let mut seq_lens = Vec::new();
    let mut slot_mapping = Vec::new();
    let mut logit_indices = Vec::new();
    let mut max_blocks: usize = 0;

    for &(req_id, n) in scheduled {
        let r = reqs.slots[req_id].as_ref().unwrap();
        let start = r.computed;

        tokens.extend_from_slice(&r.tokens_for_step(n));
        for t in 0..n {
            positions.push((start + t) as u32);
        }
        let prev = *query_starts.last().unwrap();
        query_starts.push(prev + n as u32);
        seq_lens.push((start + n) as u32);
        for t in 0..n {
            let pos = start + t;
            let block_idx = pos / pool.block_size;
            let block_off = pos % pool.block_size;
            slot_mapping.push(r.block_table[block_idx] * pool.block_size as u32 + block_off as u32);
        }
        logit_indices.push(prev + n as u32 - 1);
        max_blocks = max_blocks.max(r.block_table.len());
    }

    // Pad block tables to uniform width
    let mut block_tables = Vec::new();
    for &(req_id, _) in scheduled {
        let r = reqs.slots[req_id].as_ref().unwrap();
        block_tables.extend_from_slice(&r.block_table);
        block_tables.resize(block_tables.len() + max_blocks - r.block_table.len(), 0);
    }

    let num_tokens = tokens.len();
    let num_requests = scheduled.len();

    Batch {
        tokens,
        positions: backend.upload_u32(&positions),
        query_starts: backend.upload_u32(&query_starts),
        seq_lens: backend.upload_u32(&seq_lens),
        block_tables: backend.upload_u32(&block_tables),
        slot_mapping: backend.upload_u32(&slot_mapping),
        logit_indices: backend.upload_u32(&logit_indices),
        num_tokens,
        num_requests,
        max_blocks_per_seq: max_blocks.max(1),
    }
}

// ── Engine loop ────────────────────────────────────────────────────────

const MAX_REQUESTS: usize = 1024;
const MAX_TOKENS_PER_STEP: usize = 2048;

pub fn run<B: Backend, M: Model<B>>(
    mut model: M,
    backend: B,
    mut pool: PagedKVPool<B>,
    tokenizer: Arc<Tokenizer>,
    mut rx: tokio::sync::mpsc::Receiver<IncomingRequest>,
) {
    let mut requests = RequestTable::new(MAX_REQUESTS);
    let mut scheduler = Scheduler::new(MAX_TOKENS_PER_STEP);

    loop {
        // Non-blocking drain
        while let Ok(req) = rx.try_recv() {
            let id = requests.add(Request::from_incoming(req));
            scheduler.enqueue(id);
        }

        if scheduler.is_idle() {
            match rx.blocking_recv() {
                Some(req) => {
                    let id = requests.add(Request::from_incoming(req));
                    scheduler.enqueue(id);
                }
                None => break,
            }
        }

        // Schedule
        let scheduled = scheduler.schedule(&mut requests, &mut pool);
        if scheduled.is_empty() { continue; }

        // Build batch and forward
        let batch = build_batch(&scheduled, &requests, &pool, &backend);
        let batch_size = scheduled.len();
        model.forward(&backend, &mut pool, &batch);
        backend.sync();

        // Read all logits to CPU
        let all_logits = backend.read_to_vec_f32(model.logits());
        let vocab = all_logits.len() / batch_size;

        // Sample + emit for each request
        let mut finished = Vec::new();
        for (i, &(req_id, num_toks)) in scheduled.iter().enumerate() {
            let r = requests.slots[req_id].as_mut().unwrap();
            r.computed += num_toks;

            // Only sample after prefill is complete
            if r.computed < r.prompt_tokens.len() { continue; }

            let mut logits_slice = all_logits[i * vocab..(i + 1) * vocab].to_vec();
            let token = r.sampler.sample_from(&mut logits_slice);

            if r.stop_tokens.contains(&token) || r.generated_tokens.len() >= r.max_tokens {
                let _ = r.tx.send(ServerEvent::Done);
                finished.push(req_id);
            } else {
                let text = tokenizer.decode_token(token);
                let _ = r.tx.send(ServerEvent::Token(text));
                r.generated_tokens.push(token);
            }
        }

        for req_id in finished {
            let r = requests.slots[req_id].as_ref().unwrap();
            pool.free_blocks(&r.block_table);
            scheduler.finish(req_id);
            requests.remove(req_id);
        }
    }
}
