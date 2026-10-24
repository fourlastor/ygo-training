//! A seat that hands its decisions to the caller.
//!
//! `ygo_policies_ocgcore::Seat` asks its policy inside `feed`.  The trainer
//! has to stop at a decision, show it to a network in another process and
//! come back with the answer; it also labels decisions and plays recorded
//! ones again.  So this seat returns what it is asked instead of answering.
//! It is built from the same public pieces as `Seat` (redaction, projection,
//! `decision::prompt`, `Sequential`), used in the same order: a policy asked
//! through [`Asked`] sees exactly what it would see seated in a `Seat`.

use std::sync::Arc;

use ygo_policies::cards::CardDatabase;
use ygo_policies::model::{CardRef, CardView, Decision, Location, Observation, Position};
use ygo_policies_ocgcore::decision::{self, Sequential, Step};
use ygo_policies_ocgcore::{message, redact, wire, Message, Projection};

use crate::follow::Follower;
use crate::Result;

/// One player's view of the duel, rebuilt from the messages it is fed.
pub struct Table {
    db: Arc<dyn CardDatabase>,
    projection: Projection,
    selection_hint: u64,
}

impl Table {
    /// The view of `seat` (0 goes first).
    pub fn new(db: Arc<dyn CardDatabase>, seat: u8) -> Self {
        Table { db, projection: Projection::new(Some(seat)), selection_hint: 0 }
    }

    /// Feed one engine message (id first), every message of the duel in
    /// order.  When it asks this seat to decide, the selection comes back.
    /// The cards `followers` follow are moved as the message moves them,
    /// as the seat sees it, before the projection is.
    pub fn feed(&mut self, message: &[u8], followers: &mut [Follower]) -> Result<Option<Asked>> {
        let failed = |e: wire::ProtocolError| e.to_string();
        let Some(visible) = redact::redact(message, self.projection.seat).map_err(failed)? else { return Ok(None) };
        let parsed = message::parse(&visible).map_err(failed)?;
        for follower in followers.iter_mut() {
            follower.apply(&parsed, &self.projection);
        }
        self.projection.apply(&parsed);
        let Some(me) = self.projection.seat else { return Ok(None) };
        if let Message::Hint { kind: wire::hint::SELECTMSG, player, data } = &parsed {
            if *player == me {
                self.selection_hint = *data;
                return Ok(None);
            }
        }
        if parsed.responder() != Some(me) {
            return Ok(None);
        }
        let observation = self.projection.observation(me, &*self.db);
        let hint = std::mem::take(&mut self.selection_hint);
        let Some(prompt) = decision::prompt(&parsed, &observation, hint, &*self.db).map_err(failed)? else { return Ok(None) };
        Asked::new(observation, prompt).map(Some)
    }

    /// A refresh of the field that was decoded once for both seats: what
    /// [`Table::feed`] makes of the same message, which asks nobody anything.
    /// The library's own seats take an update this way
    /// (`ygo_policies_ocgcore::Seat::feed_update_pair`).
    pub fn update(&mut self, mut update: Message, followers: &mut [Follower]) -> Result<()> {
        redact::filter_update(&mut update, self.projection.seat).map_err(|e| e.to_string())?;
        for follower in followers.iter_mut() {
            follower.apply(&update, &self.projection);
        }
        self.projection.apply(&update);
        Ok(())
    }

    /// What the seat sees of the duel now, as a pilot would be shown it.
    pub fn observation(&self) -> Observation {
        self.projection.observation(self.projection.seat.unwrap_or(0), &*self.db)
    }
}

/// One player's seat with nothing hidden but the Decks' order: a projection
/// fed the engine's messages as the engine wrote them, which knows every
/// card a message names.  It asks no one anything.  It is what a critic that
/// sees the hidden cards reads in training, beside the seat's own view; a
/// seat that plays never reads it.
pub struct Sight {
    db: Arc<dyn CardDatabase>,
    projection: Projection,
}

impl Sight {
    /// The duel from `seat`'s side (0 goes first).
    pub fn new(db: Arc<dyn CardDatabase>, seat: u8) -> Self {
        Sight { db, projection: Projection::new(Some(seat)) }
    }

    /// Feed one engine message (id first), every message of the duel in
    /// order, unredacted.
    pub fn feed(&mut self, message: &[u8]) -> Result<()> {
        let parsed = message::parse(message).map_err(|e| e.to_string())?;
        self.projection.apply(&parsed);
        Ok(())
    }

    /// A refresh of the field, decoded once for both seats, unfiltered.
    pub fn update(&mut self, update: &Message) {
        self.projection.apply(update);
    }

    /// The duel as `me` would see it with every card it knows shown, and
    /// the next cards of each player's Deck, top first, where `decks` names
    /// them (none where it names none).
    pub fn observation(&self, me: u8, decks: &[Vec<u32>; 2]) -> Observation {
        let mut observation = self.projection.observation_whole(me, &*self.db);
        for (player, next) in decks.iter().enumerate() {
            for (sequence, code) in next.iter().enumerate() {
                let printed = self.db.card(*code);
                observation.cards.push(CardView {
                    at: CardRef { controller: player as u8, location: Location::Deck, sequence: sequence as u32 },
                    code: Some(*code),
                    position: Position { face_up: false, attack: false },
                    attack: printed.map_or(0, |card| card.attack),
                    defense: printed.map_or(0, |card| card.defense),
                    level: printed.map_or(0, |card| card.level),
                    can_attack: false,
                    battles: 0,
                    counters: 0,
                    coin_effect: None,
                    negated: false,
                });
            }
        }
        observation
    }
}

/// What a selection is made of: one decision whose choices are the engine's
/// answers, or cards picked one at a time.
#[derive(Clone)]
enum Selection {
    Plain { responses: Vec<Vec<u8>> },
    Sequential { sequential: Sequential, picked: Vec<usize>, steps: Vec<Step> },
}

/// A selection the engine asks of one seat, answered step by step.
///
/// Most selections are one decision.  Card and sum selections are several:
/// one card at a time, then finish.  [`Asked::decision`] is the step to
/// answer now and [`Asked::answer`] takes one of its choices; the engine's
/// response comes back with the last step.
#[derive(Clone)]
pub struct Asked {
    /// What the seat sees.  It is the same for every step of the selection.
    pub observation: Observation,
    decision: Decision,
    selection: Selection,
}

impl Asked {
    fn new(observation: Observation, prompt: decision::Prompt) -> Result<Self> {
        match prompt.sequential {
            Some(sequential) => {
                let (decision, steps) = sequential.step(&[]);
                if steps.is_empty() {
                    return Err("a selection offers no legal answer".into());
                }
                Ok(Asked { observation, decision, selection: Selection::Sequential { sequential, picked: Vec::new(), steps } })
            }
            None if prompt.responses.is_empty() => Err("a selection offers no legal answer".into()),
            None => Ok(Asked { observation, decision: prompt.decision, selection: Selection::Plain { responses: prompt.responses } }),
        }
    }

    /// The decision to answer now.
    pub fn decision(&self) -> &Decision {
        &self.decision
    }

    /// Take choice `index` of the current decision (one past the last choice
    /// counts as the last, as in `Seat`).  The engine's response comes back
    /// once the selection is complete; until then [`Asked::decision`] is the
    /// next step.
    pub fn answer(&mut self, index: usize) -> Option<Vec<u8>> {
        let index = index.min(self.decision.choices.len() - 1);
        match &mut self.selection {
            Selection::Plain { responses } => Some(responses[index.min(responses.len() - 1)].clone()),
            Selection::Sequential { sequential, picked, steps } => match steps[index] {
                Step::Cancel => Some(Sequential::cancel_response()),
                Step::Finish => Some(sequential.response(picked)),
                Step::Pick(card) => {
                    picked.push(card);
                    let (decision, next) = sequential.step(picked);
                    // Nothing left to pick: the selection is complete.
                    if !next.iter().any(|step| matches!(step, Step::Pick(_))) {
                        return Some(sequential.response(picked));
                    }
                    self.decision = decision;
                    *steps = next;
                    None
                }
            },
        }
    }

    /// Whether two answers to this selection, each in a form the engine
    /// reads, are one answer: the same bytes, or for a card selection the
    /// same cards, which a person's client and a policy write differently.
    pub fn same(&self, first: &[u8], second: &[u8]) -> bool {
        if first == second {
            return true;
        }
        let Selection::Sequential { sequential, .. } = &self.selection else { return false };
        let cards = |response: &[u8]| {
            let mut named = cards_of(response, sequential.candidates.len())?;
            named.sort_unstable();
            Some(named)
        };
        matches!((cards(first), cards(second)), (Some(first), Some(second)) if first == second)
    }

    /// The choice of the current decision that leads to `response`, an
    /// answer of the engine's own form to this selection: for a card
    /// selection the next of its cards in the engine's order, then finish.
    /// `None` when `response` is not an answer to this selection.
    pub fn toward(&self, response: &[u8]) -> Option<usize> {
        match &self.selection {
            Selection::Plain { responses } => responses.iter().position(|r| r == response),
            Selection::Sequential { sequential, picked, steps } => {
                if response == Sequential::cancel_response() {
                    return steps.iter().position(|step| *step == Step::Cancel);
                }
                let wanted = cards_of(response, sequential.candidates.len())?;
                if picked.iter().any(|card| !wanted.contains(card)) {
                    return None;
                }
                match wanted.iter().find(|card| !picked.contains(card)) {
                    Some(card) => steps.iter().position(|step| *step == Step::Pick(*card)),
                    None => steps.iter().position(|step| *step == Step::Finish),
                }
            }
        }
    }
}

/// The candidates, of `candidates`, that a card or sum selection response
/// names, in any of the forms the engine reads (`parse_response_cards` in
/// OCGCore's `playerop.cpp`).  An `i32` says which: 0, 1 or 2 for a `u32`
/// count and as many indices of 32, 16 or 8 bits, and 3 for one bit a
/// candidate, the lowest of a byte first.  `Sequential` writes the first for
/// cards and the third for sums; EDOPro's own client, which a person
/// answers through, writes the last.
fn cards_of(response: &[u8], candidates: usize) -> Option<Vec<usize>> {
    let kind = i32::from_le_bytes(response.get(0..4)?.try_into().ok()?);
    let named: Vec<usize> = match kind {
        3 => (0..candidates)
            .filter(|index| response.get(4 + index / 8).is_some_and(|byte| byte >> (index % 8) & 1 == 1))
            .collect(),
        0..=2 => {
            let count = u32::from_le_bytes(response.get(4..8)?.try_into().ok()?) as usize;
            let width = [4, 2, 1][kind as usize];
            let body = &response[8..];
            if body.len() != width * count {
                return None;
            }
            let index = |bytes: &[u8]| bytes.iter().rev().fold(0usize, |value, byte| value << 8 | *byte as usize);
            body.chunks(width).map(index).collect()
        }
        _ => return None,
    };
    named.iter().all(|index| *index < candidates).then_some(named)
}

#[cfg(test)]
mod tests {
    use super::cards_of;

    #[test]
    fn a_selection_is_read_in_every_form_the_engine_reads() {
        // The second and the eleventh of twelve candidates.
        let wide = [0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 10, 0, 0, 0];
        let short = [1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 10, 0];
        let bytes = [2, 0, 0, 0, 2, 0, 0, 0, 1, 10];
        let bits = [3, 0, 0, 0, 0b0000_0010, 0b0000_0100];
        for response in [&wide[..], &short[..], &bytes[..], &bits[..]] {
            assert_eq!(cards_of(response, 12), Some(vec![1, 10]), "{response:?}");
        }
        // A candidate that is not there, a form that is none, a count that
        // does not hold, and nothing picked.
        assert_eq!(cards_of(&wide, 10), None);
        assert_eq!(cards_of(&[4, 0, 0, 0, 1], 12), None);
        assert_eq!(cards_of(&[0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0], 12), None);
        assert_eq!(cards_of(&[3, 0, 0, 0], 12), Some(vec![]));
        assert_eq!(cards_of(&[3, 0, 0, 0, 0b1000_0000], 8), Some(vec![7]));
    }
}
