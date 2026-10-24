//! What becomes of the cards a seat sees at a decision, by a later point of
//! the duel: the labels of a head that learns how cards work
//! (experiments/2026-09-08-engine-transition-prediction.md).
//!
//! A [`Follower`] takes the cards of a decision's state, in the order the
//! network reads them (the observation's), and moves each as the seat's
//! projection moves it: a message is applied to the follower first, while
//! the projection still holds the piles as they were, and the follower
//! takes a card out of a pile and puts one in as the projection does. A
//! card the seat cannot follow is lost and has no label: one of a pile the
//! engine deals again (the other player's hand shuffled, set cards
//! shuffled), or one a refresh of the field drops. At the horizon the
//! follower tells of each card where it is as the seat then sees it.
//! Nothing of it is read by the network: it is a target of training.

use serde::Serialize;
use ygo_policies::model::{CardView, Location, Observation};
use ygo_policies_ocgcore::wire::{location, Loc};
use ygo_policies_ocgcore::{Message, Projection};

use crate::pack::MAX_CARDS;

/// The places a card can be found at: on its own seat's side, then the
/// other's ([`SIDE`] apart), and out of the duel.
pub const PLACES: usize = 23;
const SIDE: i64 = 11;
const HAND: i64 = 0;
const ATTACK: i64 = 1;
const DEFENSE: i64 = 2;
const FACE_DOWN: i64 = 3;
const SPELL_FACE_UP: i64 = 4;
const SPELL_FACE_DOWN: i64 = 5;
const GRAVE: i64 = 6;
const BANISHED: i64 = 7;
const DECK: i64 = 8;
const EXTRA: i64 = 9;
const MATERIAL: i64 = 10;
const GONE: i64 = 22;
/// The locations a seat's cards are counted in, for each seat.
pub const COUNTED: usize = 7;

/// Where a followed card is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Place {
    /// In a pile or a zone, by OCGCore's location bits.
    At { controller: u8, location: u8, sequence: u32 },
    /// In a Deck, or under a monster as its material: of that controller.
    Deck(u8),
    Material(u8),
    /// Out of the duel: a token that left the field.
    Gone,
    /// Out of what the seat can follow.
    Lost,
}

/// What became of a decision's cards by its horizon.
#[derive(Clone, Debug, Serialize)]
pub struct Label {
    /// The decision followed: its place among the duel's engine answers,
    /// and its step in the selection.
    pub answer: usize,
    pub step: usize,
    /// Of each card of the decision's state, where it is: one of
    /// [`PLACES`], or -1 for a card that was lost.
    pub places: Vec<i64>,
    /// Of each card on the field face-up at the decision and at the
    /// horizon, the change of its ATK and DEF, scaled as the input scales
    /// them; not a number for the others.
    pub changes: Vec<[f32; 2]>,
    /// The change of the seat's life points and of the other's, scaled as
    /// the input scales them.
    pub life: [f32; 2],
    /// The change of each seat's cards in each location counted, the seat's
    /// own first: Deck, hand, monster zones, spell and trap zones,
    /// Graveyard, banished, Extra Deck; a quarter of the count.
    pub counts: [f32; 2 * COUNTED],
    /// Of the cards whose identity the seat knew at the decision and knows
    /// where they were followed to, how many it was checked for, and how
    /// many were found there with that identity.
    pub checked: u32,
    pub agreed: u32,
}

/// The cards of one decision's state, followed through the duel.
pub struct Follower {
    pub answer: usize,
    pub step: usize,
    /// The seat's transaction the decision is of.
    pub transaction: u32,
    me: u8,
    places: Vec<Place>,
    /// Each card as the decision showed it.
    seen: Vec<CardView>,
    life: [i32; 2],
    counts: [[i64; COUNTED]; 2],
}

/// OCGCore's location bits of a location of the policies' vocabulary.
fn bits(location: Location) -> u8 {
    match location {
        Location::Deck => location::DECK,
        Location::Hand => location::HAND,
        Location::MonsterZone => location::MZONE,
        Location::SpellTrapZone => location::SZONE,
        Location::Graveyard => location::GRAVE,
        Location::Banished => location::REMOVED,
        Location::Extra => location::EXTRA,
        Location::Overlay => location::OVERLAY,
        Location::Other => 0,
    }
}

fn is_pile(location: u8) -> bool {
    matches!(location, location::HAND | location::GRAVE | location::REMOVED | location::EXTRA)
}

fn is_zone(location: u8) -> bool {
    matches!(location, location::MZONE | location::SZONE)
}

/// How many cards each pile of each player holds, as the projection held
/// them before a message; kept up to date through its steps.
struct Piles([[usize; 4]; 2]);

impl Piles {
    fn of(projection: &Projection) -> Piles {
        let lengths = |p: usize| {
            let side = &projection.sides[p];
            [side.hand.len(), side.grave.len(), side.banished.len(), side.extra.len()]
        };
        Piles([lengths(0), lengths(1)])
    }

    fn index(location: u8) -> usize {
        match location {
            location::HAND => 0,
            location::GRAVE => 1,
            location::REMOVED => 2,
            _ => 3,
        }
    }

    fn length(&mut self, controller: u8, location: u8) -> &mut usize {
        &mut self.0[controller as usize][Piles::index(location)]
    }
}

/// Of each seat, relative to `me`, the cards in each location counted.
fn counted(observation: &Observation, me: u8) -> [[i64; COUNTED]; 2] {
    let mut counts = [[0; COUNTED]; 2];
    for (player, location, size) in &observation.pile_sizes {
        let slot = match location {
            Location::Deck => 0,
            Location::Hand => 1,
            Location::Graveyard => 4,
            Location::Banished => 5,
            Location::Extra => 6,
            _ => continue,
        };
        counts[usize::from(*player != me)][slot] = i64::from(*size);
    }
    for card in &observation.cards {
        let slot = match card.at.location {
            Location::MonsterZone => 2,
            Location::SpellTrapZone => 3,
            _ => continue,
        };
        counts[usize::from(card.at.controller != me)][slot] += 1;
    }
    counts
}

impl Follower {
    /// Follow the cards of the decision whose state is `observation`'s, the
    /// decision of `answer` and `step`, of the seat's `transaction`.
    pub fn new(observation: &Observation, answer: usize, step: usize, transaction: u32) -> Follower {
        let seen = observation.cards[..observation.cards.len().min(MAX_CARDS)].to_vec();
        let places = seen
            .iter()
            .map(|card| Place::At { controller: card.at.controller, location: bits(card.at.location), sequence: card.at.sequence })
            .collect();
        Follower {
            answer,
            step,
            transaction,
            me: observation.me,
            places,
            seen,
            life: observation.life_points,
            counts: counted(observation, observation.me),
        }
    }

    /// Move the cards followed as `message` moves them, before the seat's
    /// `projection` is given it.
    pub fn apply(&mut self, message: &Message, projection: &Projection) {
        let mut piles = Piles::of(projection);
        match message {
            Message::Move { from, to, .. } => {
                let taken = if from.is_none() { None } else { self.take(*from, &mut piles) };
                if to.is_none() {
                    if let Some(card) = taken {
                        self.places[card] = Place::Gone;
                    }
                } else {
                    self.put(*to, taken, &mut piles);
                }
            }
            Message::Swap { first, second } => {
                let one = self.take(first.1, &mut piles);
                let other = self.take(second.1, &mut piles);
                self.put(second.1, one, &mut piles);
                self.put(first.1, other, &mut piles);
            }
            // Where each set card went the seat is not told.
            Message::ShuffleSetCard { from, .. } => {
                for at in from {
                    self.lose(at.controller, at.location, at.sequence);
                }
            }
            Message::ShuffleHand { player, codes } => self.dealt(*player, location::HAND, codes),
            Message::ShuffleExtra { player, codes } => self.dealt(*player, location::EXTRA, codes),
            // A refresh lists a pile whole, or every zone of a location.
            Message::UpdateData { player, location: at, cards } => {
                if is_pile(*at) {
                    for place in &mut self.places {
                        if let Place::At { controller, location, sequence } = *place {
                            if controller == *player && location == *at && sequence as usize >= cards.len() {
                                *place = Place::Lost;
                            }
                        }
                    }
                } else if is_zone(*at) {
                    for (sequence, card) in cards.iter().enumerate() {
                        if card.is_none() {
                            self.lose(*player, *at, sequence as u32);
                        }
                    }
                }
            }
            Message::UpdateCard { player, location: at, sequence, card: None } if is_zone(*at) => {
                self.lose(*player, *at, u32::from(*sequence));
            }
            Message::Start { .. } => self.places.fill(Place::Lost),
            _ => {}
        }
    }

    /// The card followed at a place, if one is.
    fn found(&self, controller: u8, location: u8, sequence: u32) -> Option<usize> {
        self.places.iter().position(|place| *place == Place::At { controller, location, sequence })
    }

    fn lose(&mut self, controller: u8, location: u8, sequence: u32) {
        if let Some(card) = self.found(controller, location, sequence) {
            self.places[card] = Place::Lost;
        }
    }

    /// What the projection's `take` takes from `at`: the card followed
    /// there, if any, which is on its way until it is put somewhere.
    fn take(&mut self, at: Loc, piles: &mut Piles) -> Option<usize> {
        let (controller, location) = (at.controller, at.location);
        if controller > 1 || location & location::OVERLAY != 0 {
            return None;
        }
        let taken = if is_pile(location) {
            let length = piles.length(controller, location);
            if *length == 0 {
                return None;
            }
            // The card at the sequence, or the last where there is none.
            let index = (at.sequence as usize).min(*length - 1);
            *length -= 1;
            let taken = self.found(controller, location, index as u32);
            for place in &mut self.places {
                if let Place::At { controller: c, location: l, sequence } = place {
                    if *c == controller && *l == location && *sequence as usize > index {
                        *sequence -= 1;
                    }
                }
            }
            taken
        } else if is_zone(location) {
            self.found(controller, location, at.sequence)
        } else {
            None
        };
        if let Some(card) = taken {
            self.places[card] = Place::Lost;
        }
        taken
    }

    /// What the projection's `put` does at `at`, with the card `taken`.
    fn put(&mut self, at: Loc, taken: Option<usize>, piles: &mut Piles) {
        let (controller, location) = (at.controller, at.location);
        let place = if controller > 1 {
            Place::Lost
        } else if location == location::DECK {
            Place::Deck(controller)
        } else if location & location::OVERLAY != 0 {
            Place::Material(controller)
        } else if is_pile(location) {
            let length = piles.length(controller, location);
            let index = (at.sequence as usize).min(*length);
            *length += 1;
            for place in &mut self.places {
                if let Place::At { controller: c, location: l, sequence } = place {
                    if *c == controller && *l == location && *sequence as usize >= index {
                        *sequence += 1;
                    }
                }
            }
            Place::At { controller, location, sequence: index as u32 }
        } else if is_zone(location) {
            // A card the projection puts over another leaves none there.
            self.lose(controller, location, at.sequence);
            Place::At { controller, location, sequence: at.sequence }
        } else {
            Place::Lost
        };
        if let Some(card) = taken {
            self.places[card] = place;
        }
    }

    /// A pile of `player`'s dealt again: of the seat's own, each card found
    /// again by its identity; of the other's, the seat cannot tell which.
    fn dealt(&mut self, player: u8, pile: u8, codes: &[u32]) {
        let mut placed = vec![false; codes.len()];
        for card in 0..self.places.len() {
            let Place::At { controller, location, .. } = self.places[card] else { continue };
            if controller != player || location != pile {
                continue;
            }
            let code = self.seen[card].code.filter(|_| player == self.me);
            let at = code.and_then(|code| (0..codes.len()).find(|&index| !placed[index] && codes[index] == code));
            self.places[card] = match at {
                Some(index) => {
                    placed[index] = true;
                    Place::At { controller, location, sequence: index as u32 }
                }
                None => Place::Lost,
            };
        }
    }

    /// Where each card is at the horizon, and what changed, as the seat sees
    /// the duel there in `horizon`.
    pub fn finish(&self, horizon: &Observation) -> Label {
        let me = self.me;
        let side = |player: u8| i64::from(player != me) * SIDE;
        let mut places = Vec::with_capacity(self.places.len());
        let mut changes = Vec::with_capacity(self.places.len());
        let (mut checked, mut agreed) = (0, 0);
        for (card, seen) in self.places.iter().zip(&self.seen) {
            let (place, now) = match *card {
                Place::At { controller, location: at, sequence } => {
                    let now = horizon
                        .cards
                        .iter()
                        .find(|card| card.at.controller == controller && card.at.sequence == sequence && bits(card.at.location) == at);
                    let place = match at {
                        location::HAND => Some(HAND),
                        location::GRAVE => Some(GRAVE),
                        location::REMOVED => Some(BANISHED),
                        location::EXTRA => Some(EXTRA),
                        location::MZONE => now.map(|card| match (card.position.face_up, card.position.attack) {
                            (true, true) => ATTACK,
                            (true, false) => DEFENSE,
                            (false, _) => FACE_DOWN,
                        }),
                        location::SZONE => now.map(|card| if card.position.face_up { SPELL_FACE_UP } else { SPELL_FACE_DOWN }),
                        _ => None,
                    };
                    (place.map(|place| side(controller) + place), now)
                }
                Place::Deck(controller) => (Some(side(controller) + DECK), None),
                Place::Material(controller) => (Some(side(controller) + MATERIAL), None),
                Place::Gone => (Some(GONE), None),
                Place::Lost => (None, None),
            };
            if let (Some(code), Some(later)) = (seen.code, now.and_then(|card| card.code)) {
                checked += 1;
                agreed += u32::from(later == code);
            }
            places.push(place.unwrap_or(-1));
            let fielded = |card: &CardView| card.at.location == Location::MonsterZone && card.position.face_up && card.code.is_some();
            changes.push(match now {
                Some(now) if fielded(seen) && fielded(now) => [
                    (now.attack - seen.attack) as f32 / 4000.0,
                    (now.defense - seen.defense) as f32 / 4000.0,
                ],
                _ => [f32::NAN; 2],
            });
        }
        let life = |player: usize| (horizon.life_points[player] - self.life[player]) as f32 / 8000.0;
        let later = counted(horizon, me);
        let mut counts = [0.0; 2 * COUNTED];
        for (index, count) in counts.iter_mut().enumerate() {
            let (player, slot) = (index / COUNTED, index % COUNTED);
            *count = (later[player][slot] - self.counts[player][slot]) as f32 / 4.0;
        }
        Label {
            answer: self.answer,
            step: self.step,
            places,
            changes,
            life: [life(me as usize), life(1 - me as usize)],
            counts,
            checked,
            agreed,
        }
    }
}
