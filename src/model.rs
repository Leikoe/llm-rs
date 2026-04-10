use crate::backend::Backend;
use crate::batch::Batch;
use crate::kv_pool::PagedKVPool;

pub trait Model<B: Backend> {
    fn forward(&mut self, backend: &B, pool: &mut PagedKVPool<B>, batch: &Batch<B>);
    fn logits(&self) -> &B::Buffer;
}
