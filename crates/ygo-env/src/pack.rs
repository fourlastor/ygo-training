//! A decision as the arrays the policy reads, and batches of them.
//!
//! The one definition of the model's input, shared with whatever else seats
//! a checkpoint: it lives in `native/ygo-model-input` (`input/src/pack.rs`).
//! The trainer builds none of these arrays itself.

pub use ygo_model_input::pack::*;
