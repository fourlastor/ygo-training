//! A decision as the trainer receives it.
//!
//! The view and what makes one from a seat's question are shared with
//! whatever else seats a checkpoint: they live in `native/ygo-model-input`
//! (`input/src/view.rs` for the data, `seat/src/view.rs` for its making).

pub use ygo_model_seat::view::*;
