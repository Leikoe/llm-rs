use crate::backend::Backend;

/// A loaded model that can run forward passes.
pub trait Model<B: Backend> {
    type Session;
    fn new_session(&self, backend: &B) -> Self::Session;
    fn forward(&self, backend: &B, session: &mut Self::Session, tokens: &[u32], want_logits: bool);
    fn logits<'a>(&self, session: &'a Self::Session) -> &'a B::Buffer;
}
