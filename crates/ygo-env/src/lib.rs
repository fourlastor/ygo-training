//! Duels for the trainer.
//!
//! The duels are run by the `ygo-policies` crates: their engine build, their
//! card data, their decks and their pilots.  This crate adds what training
//! needs on top:
//!
//! * [`seat`]: a seat that hands its decisions to the caller instead of
//!   asking a policy, built from the pieces the policies' own seat is made of;
//! * [`view`]: a decision as the trainer receives it, derived from what a
//!   pilot is given at the same point;
//! * [`episode`]: one duel, with each seat played by a pilot, by the caller
//!   or from a record;
//! * [`features`] and [`pack`]: a decision as the arrays the policy reads,
//!   and batches of them;
//! * [`trail`]: what a seat keeps of its own play;
//! * [`follow`]: what becomes of the cards a seat sees at a decision, the
//!   labels of a head that learns how cards work;
//! * [`pool`]: many duels taken forward together, their decisions in batches;
//! * [`edopro`]: the caller's seat in a room of EDOPro, against a person;
//! * a C interface over JSON, which the Python trainer loads.
//!
//! The view, the arrays, the trail and the policy of a caller's seat are
//! shared with whatever else seats a checkpoint: they are the crates of
//! `native/ygo-model-input`, a repository of its own, and the modules of
//! those names here pass them on.

pub mod edopro;
pub mod episode;
pub mod features;
mod ffi;
pub mod follow;
pub mod pack;
pub mod pool;
pub mod seat;
pub mod trail;
pub mod view;

pub use crate::episode::{Control, Entry, Environment, Episode, Event, Options, Over, SeatSetup, Setup};
pub use crate::pool::Pool;
pub use crate::view::View;

/// Everything here fails with a message.
pub type Result<T> = std::result::Result<T, String>;
