//! What a seat keeps of its own play, and what a decision is to a network.
//!
//! An episode keeps a trail for each of its seats, and so does a seat at an
//! EDOPro table: one code, shared with whatever else seats a checkpoint.  It
//! lives in `native/ygo-model-input` (`seat/src/trail.rs`).

pub use ygo_model_seat::trail::*;
