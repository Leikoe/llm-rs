use std::collections::VecDeque;

use crate::kv_pool::{self, PagedKVPool};
use crate::backend::Backend;

use super::engine::{ReqId, RequestTable};

pub struct Scheduler {
    running: Vec<ReqId>,
    waiting: VecDeque<ReqId>,
    budget: usize,
}

impl Scheduler {
    pub fn new(budget: usize) -> Self {
        Scheduler { running: Vec::new(), waiting: VecDeque::new(), budget }
    }

    pub fn enqueue(&mut self, id: ReqId) {
        self.waiting.push_back(id);
    }

    pub fn finish(&mut self, id: ReqId) {
        self.running.retain(|&r| r != id);
    }

    pub fn is_idle(&self) -> bool {
        self.running.is_empty() && self.waiting.is_empty()
    }

    /// Unified token budget scheduler (vLLM V1 pattern).
    /// Skips requests that can't allocate KV blocks — they retry next step
    /// when blocks may have been freed by completed requests.
    pub fn schedule<B: Backend>(
        &mut self,
        reqs: &mut RequestTable,
        pool: &mut PagedKVPool<B>,
    ) -> Vec<(ReqId, usize)> {
        let mut budget = self.budget;
        let mut out = Vec::new();

        // 1. Running requests first (decode = 1 token each)
        for &id in &self.running {
            if budget == 0 { break; }
            let r = reqs.slots[id].as_mut().unwrap();
            let needed = r.total_tokens() - r.computed;
            if needed == 0 { continue; }
            let n = needed.min(budget);
            if !kv_pool::ensure_blocks(&mut r.block_table, pool, r.computed + n) {
                continue; // no blocks available, skip this step
            }
            out.push((id, n));
            budget -= n;
        }

        // 2. Waiting requests (prefill, chunked if budget < prompt)
        let mut retry = VecDeque::new();
        while budget > 0 {
            let Some(id) = self.waiting.pop_front() else { break };
            let r = reqs.slots[id].as_mut().unwrap();
            let needed = r.prompt_tokens.len() - r.computed;
            let n = needed.min(budget);
            if !kv_pool::ensure_blocks(&mut r.block_table, pool, r.computed + n) {
                retry.push_back(id); // put back, try next step
                break; // if one doesn't fit, later ones won't either
            }
            out.push((id, n));
            budget -= n;
            self.running.push(id);
        }
        // Put unattempted + failed requests back at the front
        retry.extend(self.waiting.drain(..));
        self.waiting = retry;

        out
    }
}
