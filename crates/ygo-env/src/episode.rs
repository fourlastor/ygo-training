//! One duel, run for the trainer.
//!
//! The duel is the one `policy-bench` plays: the same deal from a seed, the
//! same messages to the same seats, pilots seeded the same way.  What this
//! adds is who answers each seat and what is reported:
//!
//! * a **pilot** plays its seat, as in the bench;
//! * the **caller** decides (a network): [`Episode::advance`] stops at each
//!   decision and [`Episode::choose`] answers it, while the deck's pilot,
//!   asked alongside, gives the answer to imitate;
//! * a **record** is played again, answer by answer;
//! * a **search** plays its seat: the policies' staged search, which plays
//!   its worlds on the duel itself while it answers, its answers given as a
//!   record's are.
//!
//! Decisions that are not a network's to take never reach the caller: where
//! a card is placed is the pilot's choice, and a decision with one answer
//! takes it.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use ygo_policies::agent::Policy;
use ygo_policies::cards::CardDatabase;
use ygo_policies::registry;
use ygo_policies_duel::{Core, Deck, Duel, DuelOptions, Recorded, State};
use ygo_policies_ocgcore::wire::{location, msg, query};
use ygo_policies_ocgcore::{message, Message, SqliteCards};
use ygo_policies_search::{LocalPilots, SearchOptions, SearchSeat, SeatStats};

use crate::features::{self, Features};

use crate::follow::{Follower, Label};
use crate::pack::{self, Batch, Packed};
use crate::seat::{Asked, Sight, Table};
use crate::trail::{standing, Trail};
use crate::view::{self, View};
use crate::Result;

/// A deck of the pool, under the id of the policy written for it.
#[derive(Serialize, Clone, Debug)]
pub struct Entry {
    pub id: String,
    /// The name of its list.
    pub deck: String,
    pub main: Vec<u32>,
    pub extra: Vec<u32>,
}

/// Who answers a seat.
#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Control {
    Pilot,
    Caller,
    Record,
    Search,
}

#[derive(Deserialize, Clone, Debug)]
pub struct SeatSetup {
    pub control: Control,
    /// The policy that plays the seat, or is asked alongside it.  The one
    /// written for the seat's deck when left out.  A search's seat is
    /// piloted by it, the generic policy when left out.
    #[serde(default)]
    pub policy: Option<String>,
    /// For a search's seat: the policy the other player is taken to be, the
    /// generic one when left out.
    #[serde(default)]
    pub model: Option<String>,
    /// For a search's seat: what it changes of the settings upstream
    /// measured (`SearchOptions::from_json`).
    #[serde(default)]
    pub search: Option<serde_json::Value>,
    /// For a caller's or a recorded seat: no pilot alongside.
    #[serde(default)]
    pub alone: bool,
    /// Report the seat's decisions (a caller is shown its own anyway).
    #[serde(default)]
    pub observe: bool,
}

/// What is kept from a network: the zone a card goes to, a decision with a
/// single answer, and all but its last answers.
pub use ygo_model_seat::Options;

fn default_limit() -> usize {
    4096
}

/// A duel to run.
#[derive(Deserialize, Clone, Debug)]
pub struct Setup {
    /// The duel of `policy-bench` with this seed: the deal, the engine's
    /// random stream and the pilots' tie-breaking all come from it.
    pub seed: u64,
    /// The deck of seat 0, who goes first, and of seat 1, by policy id.
    pub decks: [String; 2],
    /// A deck list read from a `.ydk` file in place of a seat's deck's own:
    /// a list no policy was written for.  The seat's pilot, or the one asked
    /// alongside it, is still its deck's.
    #[serde(default)]
    pub lists: [Option<PathBuf>; 2],
    pub seats: [SeatSetup; 2],
    /// Engine answers after which the duel is stopped without a winner.
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub life_points: Option<[u32; 2]>,
    /// A recorded duel (the JSON form of `Recorded`) for its recorded seats.
    #[serde(default)]
    pub record: Option<serde_json::Value>,
    /// Return the record of the duel when it ends.  One the engine stops
    /// returns its record whatever this says.
    #[serde(default)]
    pub keep_record: bool,
    #[serde(default)]
    pub options: Options,
    /// Pack beside each of a caller's decisions the same decision with
    /// nothing hidden but the Decks' order ([`Event::whole`]): what a critic
    /// that sees the hidden cards reads in training.  The caller's own view
    /// is the same with it or without.
    #[serde(default)]
    pub whole: bool,
    /// With `whole`, the next so many cards of each player's Deck as well,
    /// top first: the draws to come, known.
    #[serde(default)]
    pub whole_deck: usize,
    /// Follow the cards of each of a caller's transactions, at its last
    /// decision, to the horizon ([`Event::labels`]): what a head that learns
    /// how cards work is taught in training.  Nothing of it reaches a view.
    #[serde(default)]
    pub follow: bool,
}

/// How a duel ended.
#[derive(Serialize, Clone, Debug)]
pub struct Over {
    /// The seat that won, 2 for a draw; `None` when the limit stopped the duel.
    pub winner: Option<u8>,
    /// The engine's reason: 1 Life Points, 2 deck-out, 16 and up a card's own.
    pub reason: Option<u8>,
    pub turns: u32,
    /// Engine answers given by both seats.
    pub decisions: usize,
    pub limit: bool,
    /// `policy-bench`'s digest of every question and answer.
    pub digest: String,
    /// Selections of a recorded seat that its pilot would have answered otherwise.
    pub departures: usize,
    pub record: Option<serde_json::Value>,
    /// What each search's seat did: its questions, those it searched, the
    /// answers it changed from its pilot's and the time its searches took.
    pub search: Option<serde_json::Value>,
    /// What the engine reported, where it stopped the duel: a card's script
    /// that failed, or the duel's arena found full.  Such a duel ends as one
    /// the limit stops, its record with it, to be played again.
    pub failure: Option<String>,
}

/// What [`Episode::advance`] returns.
#[derive(Serialize, Debug)]
pub struct Event {
    /// Decisions of observed seats answered since the last call.
    pub samples: Vec<View>,
    /// A decision for the caller: [`Episode::choose`] answers it.
    pub decision: Option<View>,
    /// The same decision with nothing hidden but the Decks' order, where the
    /// setup asks for it: never a seat's to read.
    pub whole: Option<View>,
    /// What became of the cards of decisions shown before, where the setup
    /// asks to follow them ([`Setup::follow`]): each at its horizon, the
    /// seat's next decision of another transaction with no chain open, the
    /// end of the turn or of the duel, whichever came first.
    pub labels: Vec<Label>,
    /// The end of the duel.
    pub over: Option<Over>,
}

/// The engine with its cards and scripts, and the decks of the pool.
/// One per thread, as the `Core` it holds.
pub struct Environment {
    pub(crate) core: Core,
    /// The card database, and where it was read from.
    pub(crate) db: Arc<dyn CardDatabase>,
    pub(crate) cards: PathBuf,
    pub(crate) decks: HashMap<String, Deck>,
    catalog: Vec<Entry>,
    /// Printed card data.
    features: Features,
}

impl Environment {
    /// `cards` is the card database, `scripts` the card scripts and `decks`
    /// the folder of the policies' deck lists.
    pub fn open(cards: &Path, scripts: &Path, decks: &Path) -> Result<Self> {
        let core = Core::open(None, cards, scripts)?;
        let db: Arc<dyn CardDatabase> =
            Arc::new(SqliteCards::open(cards).map_err(|e| format!("{}: {e}", cards.display()))?);
        let features = features::open(cards)?;
        let mut lists = HashMap::new();
        let mut catalog = Vec::new();
        for entry in registry::POLICIES {
            let deck = Deck::load(&decks.join(format!("{}.ydk", entry.deck)))?;
            catalog.push(Entry {
                id: entry.id.to_owned(),
                deck: entry.deck.to_owned(),
                main: deck.main.clone(),
                extra: deck.extra.clone(),
            });
            lists.insert(entry.id.to_owned(), deck);
        }
        Ok(Environment { core, db, cards: cards.to_owned(), decks: lists, catalog, features })
    }

    /// A decision as the arrays the policy reads.
    pub fn pack(&self, view: &View) -> Result<Packed> {
        pack::pack(view, &self.features)
    }

    /// Packed decisions as one batch: the tensors the model is given.
    pub fn batch(&self, decisions: &[&Packed]) -> Result<Batch> {
        pack::batch(decisions)
    }

    /// The decks of the pool.
    pub fn catalog(&self) -> &[Entry] {
        &self.catalog
    }

    fn deck(&self, id: &str) -> Result<Deck> {
        self.decks.get(id).cloned().ok_or_else(|| format!("no deck {id:?} in the pool"))
    }

    /// The list seat `player` of `setup` plays: the one read from its file,
    /// or its deck's own.
    fn list(&self, setup: &Setup, player: usize) -> Result<Deck> {
        let own = self.deck(&setup.decks[player])?;
        match &setup.lists[player] {
            Some(path) => Deck::load(path),
            None => Ok(own),
        }
    }

    /// Start a duel.  Nothing is played until [`Episode::advance`].
    pub fn start(&self, setup: &Setup) -> Result<Episode<'_>> {
        let lists = [self.list(setup, 0)?, self.list(setup, 1)?];
        let record = setup.record.as_ref().map(Recorded::from_json).transpose()?;
        let options = DuelOptions {
            life_points: setup.life_points.unwrap_or([DuelOptions::LIFE_POINTS; 2]),
            ..DuelOptions::seeded(setup.seed)
        };
        let duel = match &record {
            Some(record) => record.duel(&self.core)?,
            None => self.core.deal(&options, &lists)?,
        };
        // Shared with a search's seat, which plays its worlds on it.
        let duel = Rc::new(RefCell::new(duel));
        // Kept whatever the setup asks, for a duel the engine stops.
        let kept = match &record {
            Some(record) => Recorded { responses: Vec::new(), ..record.clone() },
            None => Recorded::dealt(&options, &lists),
        };
        let mut answers = [VecDeque::new(), VecDeque::new()];
        for (player, answer) in record.into_iter().flat_map(|record| record.responses) {
            answers.get_mut(player as usize).ok_or("a recorded answer names no seat")?.push_back(answer);
        }
        let mut seats = Vec::new();
        for (player, seat) in setup.seats.iter().enumerate() {
            if seat.control == Control::Record && setup.record.is_none() {
                return Err(format!("seat {player} plays a record and none is given"));
            }
            // Asked alongside a search's seat, the deck's own.
            let id = match seat.control {
                Control::Search => setup.decks[player].as_str(),
                _ => seat.policy.as_deref().unwrap_or(&setup.decks[player]),
            };
            let pilot = if seat.control == Control::Pilot || !seat.alone {
                // The bench seeds seat `p` with `seed + p`.
                let seed = setup.seed.wrapping_add(player as u64);
                Some(registry::create_seeded(id, self.db.clone(), seed).ok_or_else(|| format!("no policy {id:?}"))?)
            } else {
                None
            };
            let start = duel.borrow().start_message(player as u8);
            let mut table = Table::new(self.db.clone(), player as u8);
            table.feed(&start, &mut [])?;
            let sight = if setup.whole && seat.control == Control::Caller {
                let mut sight = Sight::new(self.db.clone(), player as u8);
                sight.feed(&start)?;
                Some(sight)
            } else {
                None
            };
            let search = if seat.control == Control::Search {
                let options = match &seat.search {
                    Some(options) => SearchOptions::from_json(options)?,
                    None => SearchOptions::MEASURED,
                };
                let pilots = LocalPilots { cards: self.db.clone() };
                let pilot = seat.policy.as_deref().unwrap_or("generic");
                let mut search =
                    SearchSeat::with_pilots(duel.clone(), pilots, player as u8, setup.seed, options, pilot, seat.model.as_deref())?;
                search.feed(&start)?;
                Some(search)
            } else {
                None
            };
            seats.push(Seat {
                table,
                control: seat.control,
                pilot,
                observe: seat.observe,
                trail: Trail::default(),
                sight,
                followers: Vec::new(),
                search,
            });
        }
        let seats: [Seat; 2] = seats.try_into().map_err(|_| "a duel has two seats".to_owned())?;
        Ok(Episode {
            duel,
            seats,
            options: setup.options.clone(),
            limit: setup.limit,
            answers,
            kept,
            keep_record: setup.keep_record,
            pending: None,
            samples: Vec::new(),
            whole: None,
            whole_deck: setup.whole_deck,
            follow: setup.follow,
            labels: Vec::new(),
            decisions: 0,
            turns: 0,
            digest: 0xcbf29ce484222325,
            departures: 0,
            over: None,
            refreshes: Refreshes::default(),
        })
    }
}

struct Seat<'a> {
    table: Table,
    control: Control,
    /// Plays a pilot's seat; is asked alongside a caller's or a recorded one.
    pilot: Option<Box<dyn Policy>>,
    observe: bool,
    /// Its own earlier answers and the transaction it is in.
    trail: Trail,
    /// The duel from its side with nothing hidden but the Decks' order, for
    /// a caller's seat whose setup asks for it.
    sight: Option<Sight>,
    /// The cards of its decisions followed to their horizon, where the
    /// setup asks for it: of a transaction, its last decision's.
    followers: Vec<Follower>,
    /// The staged search that answers a search's seat.
    search: Option<SearchSeat<'a, LocalPilots>>,
}

/// The field as it was last refreshed, a location at a time: the bytes of its
/// `MSG_UPDATE_DATA` and what they decode to.  A refresh is most often the
/// one before it byte for byte (84% of those before a selection in games
/// between pilots), and is then not decoded again.
#[derive(Default)]
struct Refreshes {
    last: Vec<(Vec<u8>, Message)>,
}

impl Refreshes {
    /// What a refresh decodes to, as `message::parse` decodes it.
    fn decode(&mut self, bytes: &[u8]) -> Result<Message> {
        // The refreshes of a location begin with the same three bytes: the
        // message's id, the player and the location.
        let location = |kept: &[u8]| kept.len() >= 3 && bytes.len() >= 3 && kept[..3] == bytes[..3];
        let at = self.last.iter().position(|(kept, _)| location(kept));
        if let Some(at) = at {
            if self.last[at].0 == bytes {
                return Ok(self.last[at].1.clone());
            }
        }
        let decoded = message::parse(bytes).map_err(|e| e.to_string())?;
        if bytes.first() == Some(&msg::UPDATE_DATA) && bytes.len() >= 3 {
            let kept = (bytes.to_vec(), decoded.clone());
            match at {
                Some(at) => self.last[at] = kept,
                None => self.last.push(kept),
            }
        }
        Ok(decoded)
    }
}

/// A selection being answered.
struct Pending {
    player: usize,
    asked: Asked,
    /// The selection message, for the digest.
    message: Vec<u8>,
    step: usize,
    /// The choices of the step the caller was shown and has not answered.
    shown: Option<Vec<usize>>,
}

/// One duel.  [`Episode::advance`] runs it until the caller has to decide or
/// the duel ends.
pub struct Episode<'a> {
    duel: Rc<RefCell<Duel<'a>>>,
    seats: [Seat<'a>; 2],
    options: Options,
    limit: usize,
    /// The recorded answers each seat has yet to give.
    answers: [VecDeque<Vec<u8>>; 2],
    kept: Recorded,
    keep_record: bool,
    pending: Option<Pending>,
    samples: Vec<View>,
    /// The decision the caller is shown, with nothing hidden, and how many
    /// of each Deck's next cards that shows.
    whole: Option<View>,
    whole_deck: usize,
    /// Whether the cards of a caller's decisions are followed, and the
    /// labels of those that reached their horizon, to hand out.
    follow: bool,
    labels: Vec<Label>,
    decisions: usize,
    turns: u32,
    digest: u64,
    departures: usize,
    over: Option<Over>,
    refreshes: Refreshes,
}

impl Episode<'_> {
    /// Run the duel until the caller has to decide, or to its end.  The
    /// decisions of observed seats answered on the way come with it.
    pub fn advance(&mut self) -> Result<Event> {
        loop {
            if let Some(over) = &self.over {
                let samples = std::mem::take(&mut self.samples);
                let labels = std::mem::take(&mut self.labels);
                return Ok(Event { samples, decision: None, whole: None, labels, over: Some(over.clone()) });
            }
            if self.pending.is_none() {
                self.run()?;
                continue;
            }
            if let Some(decision) = self.settle()? {
                let samples = std::mem::take(&mut self.samples);
                let labels = std::mem::take(&mut self.labels);
                return Ok(Event { samples, decision: Some(decision), whole: self.whole.take(), labels, over: None });
            }
        }
    }

    /// Answer the decision [`Episode::advance`] returned with one of its choices.
    pub fn choose(&mut self, index: usize) -> Result<()> {
        let pending = self.pending.as_mut().ok_or("no decision is waiting for an answer")?;
        let open = pending.shown.take().ok_or("no decision is waiting for an answer")?;
        let Some(&choice) = open.get(index) else {
            let offered = open.len();
            pending.shown = Some(open);
            return Err(format!("choice {index} of a decision that offers {offered}"));
        };
        let trail = &mut self.seats[pending.player].trail;
        let decision = pending.asked.decision();
        trail.remember(&decision.choices[choice], self.options.history);
        trail.note(decision, choice);
        pending.step += 1;
        if let Some(response) = pending.asked.answer(choice) {
            self.respond(response);
        }
        Ok(())
    }

    /// Let the engine run until it asks a seat or the duel ends.
    fn run(&mut self) -> Result<()> {
        let stepped = self.duel.borrow_mut().step();
        let step = match stepped {
            Ok(step) => step,
            // An error the engine reported: a card's script failed, or the
            // duel's arena is full.  The duel cannot go on, and ends there as
            // one the limit stops, with what was reported and its record.
            // Any other error is of the messages read, and stops the caller.
            Err(message) if !self.duel.borrow().errors().is_empty() => {
                self.finish_followers();
                self.over = Some(Over {
                    winner: None,
                    reason: None,
                    turns: self.turns,
                    decisions: self.decisions,
                    limit: true,
                    digest: format!("{:016x}", self.digest),
                    departures: self.departures,
                    record: Some(self.kept.to_json()),
                    search: self.searched(),
                    failure: Some(message),
                });
                return Ok(());
            }
            Err(message) => return Err(message),
        };
        let mut pending = None;
        for sent in &step.messages {
            // A search's seat is fed every message, the field's refreshes
            // with the others, and answers a question of its own as it is fed
            // it, its worlds played meanwhile: its answer is then given as a
            // recorded one is.
            for (player, seat) in self.seats.iter_mut().enumerate() {
                if let Some(search) = &mut seat.search {
                    if let Some(answer) = search.feed(&sent.bytes)? {
                        self.answers[player].push_back(answer);
                    }
                }
            }
            if !sent.refresh {
                match sent.bytes.first() {
                    Some(&msg::RETRY) => return Err("the engine rejected an answer".into()),
                    Some(&msg::NEW_TURN) => {
                        // The end of a turn is the horizon of every decision
                        // still followed: seen as the turn ended.
                        self.finish_followers();
                        self.turns += 1;
                    }
                    _ => {}
                }
            }
            if matches!(sent.bytes.first(), Some(&msg::UPDATE_DATA) | Some(&msg::UPDATE_CARD)) {
                // Most of what a seat is fed is the field's refresh, which a
                // seat on its own would redact and decode for itself: it is
                // decoded once, or not at all when it is the refresh of its
                // location before it, and each seat shown its part of it.
                let update = self.refreshes.decode(&sent.bytes)?;
                let [first, second] = &mut self.seats;
                for sight in [&mut first.sight, &mut second.sight].into_iter().flatten() {
                    sight.update(&update);
                }
                first.table.update(update.clone(), &mut first.followers)?;
                second.table.update(update, &mut second.followers)?;
                continue;
            }
            for (player, seat) in self.seats.iter_mut().enumerate() {
                if let Some(sight) = &mut seat.sight {
                    sight.feed(&sent.bytes)?;
                }
                if let Some(asked) = seat.table.feed(&sent.bytes, &mut seat.followers)? {
                    if pending.is_some() {
                        return Err("two seats are asked in one batch of messages".into());
                    }
                    pending = Some(Pending { player, asked, message: sent.bytes.clone(), step: 0, shown: None });
                }
            }
        }
        if matches!(step.state, State::Over(_)) || self.decisions >= self.limit {
            self.finish_followers();
            let (winner, reason) = match step.state {
                State::Over(Some(outcome)) => (Some(outcome.winner), Some(outcome.reason)),
                State::Over(None) => return Err("the engine stopped without naming a winner".into()),
                _ => (None, None),
            };
            self.over = Some(Over {
                winner,
                reason,
                turns: self.turns,
                decisions: self.decisions,
                limit: winner.is_none(),
                digest: format!("{:016x}", self.digest),
                departures: self.departures,
                record: self.keep_record.then(|| self.kept.to_json()),
                search: self.searched(),
                failure: None,
            });
            return Ok(());
        }
        match pending {
            None if step.state == State::Awaiting => Err("the engine waits and no seat was asked".into()),
            None => Ok(()),
            Some(pending) => {
                self.seats[pending.player].trail.asked(pending.asked.decision());
                self.pending = Some(pending);
                Ok(())
            }
        }
    }

    /// Answer the pending selection as far as it goes without the caller.
    /// The decision comes back when the caller has to take it.
    fn settle(&mut self) -> Result<Option<View>> {
        let pending = self.pending.as_mut().expect("a selection is pending");
        if pending.shown.is_some() {
            return Err("the decision has not been answered".into());
        }
        let seat = &mut self.seats[pending.player];
        let response = match seat.control {
            Control::Pilot => loop {
                let pilot = seat.pilot.as_mut().expect("a pilot's seat has its pilot");
                let choice = ask(pilot.as_mut(), &pending.asked);
                report(seat, pending, &self.options, self.decisions, choice, Some(choice), &mut self.samples);
                if let Some(response) = pending.asked.answer(choice) {
                    break response;
                }
            },
            Control::Record | Control::Search => {
                let player = pending.player;
                let recorded = self.answers[player].pop_front().ok_or(match seat.control {
                    Control::Search => "the search gave no answer to its seat's question",
                    _ => "the record ends before the duel does",
                })?;
                // The pilot answers the selection its own way first, as it
                // did when the duel was played: it stays in step with the
                // recorded one, and a selection it answers as recorded is
                // walked through in its own order of picks.
                let own = seat.pilot.as_mut().map(|pilot| {
                    let mut asked = pending.asked.clone();
                    let mut path = Vec::new();
                    loop {
                        let choice = ask(pilot.as_mut(), &asked);
                        path.push(choice);
                        if let Some(response) = asked.answer(choice) {
                            break (path, response);
                        }
                    }
                });
                let followed = own.as_ref().is_some_and(|(_, response)| pending.asked.same(response, &recorded));
                if own.is_some() && !followed {
                    self.departures += 1;
                }
                let mut walked = 0;
                loop {
                    let choice = match &own {
                        Some((path, _)) if followed => path[walked],
                        _ => pending
                            .asked
                            .toward(&recorded)
                            .ok_or("a recorded answer does not fit the selection it answers")?,
                    };
                    // Where the record leaves the pilot, its own answer is
                    // known for the step both were asked: the first.
                    let pilot = match &own {
                        Some(_) if followed => Some(choice),
                        Some((path, _)) if walked == 0 => Some(path[0]),
                        _ => None,
                    };
                    report(seat, pending, &self.options, self.decisions, choice, pilot, &mut self.samples);
                    walked += 1;
                    if let Some(response) = pending.asked.answer(choice) {
                        if !pending.asked.same(&response, &recorded) {
                            return Err("a recorded answer cannot be rebuilt from its selection".into());
                        }
                        // The engine is given the answer as it was given
                        // then: the same cards may be written otherwise.
                        break recorded;
                    }
                }
            }
            Control::Caller => loop {
                let decision = pending.asked.decision();
                let standing = seat.trail.offered(&pending.asked.observation, decision, &self.options);
                let label = seat.pilot.as_mut().map(|pilot| ask(pilot.as_mut(), &pending.asked));
                if standing.real {
                    let mut shown = view::view(&pending.asked.observation, decision, &standing.open);
                    seat.trail.stamp(&mut shown, &pending.asked.observation, pending.step, self.decisions);
                    shown.label = label.and_then(|label| standing.open.iter().position(|index| *index == label));
                    shown.pilot = shown.label;
                    if self.follow {
                        let observation = &pending.asked.observation;
                        // A decision of a later transaction with no chain
                        // open is the horizon of those followed before it.
                        if observation.chain.is_empty() {
                            let (reached, going): (Vec<_>, Vec<_>) = std::mem::take(&mut seat.followers)
                                .into_iter()
                                .partition(|follower| follower.transaction != shown.transaction);
                            self.labels.extend(reached.iter().map(|follower| follower.finish(observation)));
                            seat.followers = going;
                        }
                        // Of a transaction, its last decision is followed.
                        seat.followers.retain(|follower| follower.transaction != shown.transaction);
                        seat.followers.push(Follower::new(observation, shown.answer, shown.step, shown.transaction));
                    }
                    // The same decision with nothing hidden, stamped with the
                    // seat's own trail: its cards are found where the
                    // observation with nothing hidden has them.
                    self.whole = match &seat.sight {
                        None => None,
                        Some(sight) => {
                            let decks = match self.whole_deck {
                                0 => Default::default(),
                                count => {
                                    let duel = self.duel.borrow();
                                    [next_cards(&duel, 0, count)?, next_cards(&duel, 1, count)?]
                                }
                            };
                            let observation = sight.observation(pending.player as u8, &decks);
                            let mut whole = view::view(&observation, decision, &standing.open);
                            seat.trail.stamp(&mut whole, &observation, pending.step, self.decisions);
                            whole.label = shown.label;
                            whole.pilot = shown.pilot;
                            Some(whole)
                        }
                    };
                    pending.shown = Some(standing.open);
                    return Ok(Some(shown));
                }
                let choice = match label {
                    Some(label) if standing.placement => label,
                    _ => standing.open.first().copied().unwrap_or(0),
                };
                seat.trail.note(decision, choice);
                pending.step += 1;
                if let Some(response) = pending.asked.answer(choice) {
                    break response;
                }
            },
        };
        self.respond(response);
        Ok(None)
    }

    /// Every decision still followed has its horizon here: its label, as
    /// its seat sees the duel now.
    fn finish_followers(&mut self) {
        for seat in &mut self.seats {
            if seat.followers.is_empty() {
                continue;
            }
            let observation = seat.table.observation();
            self.labels.extend(seat.followers.drain(..).map(|follower| follower.finish(&observation)));
        }
    }

    /// What each search's seat did, where the duel has one.
    fn searched(&self) -> Option<serde_json::Value> {
        let told: Vec<_> = self
            .seats
            .iter()
            .enumerate()
            .filter_map(|(player, seat)| seat.search.as_ref().map(|search| search_summary(player, search.stats())))
            .collect();
        (!told.is_empty()).then(|| serde_json::Value::Array(told))
    }

    /// Give the engine the answer to the pending selection.
    fn respond(&mut self, response: Vec<u8>) {
        let pending = self.pending.take().expect("a selection is pending");
        for byte in pending.message.iter().chain(&response) {
            self.digest = (self.digest ^ *byte as u64).wrapping_mul(0x100000001b3);
        }
        self.duel.borrow_mut().respond(&response);
        self.kept.responses.push((pending.player as u8, response));
        self.decisions += 1;
        self.seats[1 - pending.player].trail.interrupt();
    }
}

/// What a search's seat did, its answers changed counted and its searches'
/// time summed.
fn search_summary(player: usize, stats: &SeatStats) -> serde_json::Value {
    let mut told = stats.to_json();
    told["player"] = player.into();
    told["changed"] = stats.deviations.len().into();
    told["seconds"] = (stats.search_ms.iter().sum::<f64>() / 1000.0).into();
    if let Some(told) = told.as_object_mut() {
        told.remove("deviations");
        told.remove("search_decision_ms");
    }
    told
}

/// The codes of the next `count` cards of `player`'s Deck, top first: the
/// engine's own, never a seat's to read.
fn next_cards(duel: &Duel, player: u8, count: usize) -> Result<Vec<u32>> {
    let deck = u32::from(location::DECK);
    let size = duel.count(player, deck)?;
    (0..size.min(count as u32))
        .map(|from_top| {
            let bytes = duel.card(query::CODE, player, deck, size - 1 - from_top)?;
            queried_code(&bytes).ok_or_else(|| format!("the engine names no card at {from_top} from the top of a Deck"))
        })
        .collect()
}

/// The code a card query of the engine holds: fields of a `u16` size, a
/// `u32` flag and the value.
fn queried_code(bytes: &[u8]) -> Option<u32> {
    let mut at = 0;
    while at + 6 <= bytes.len() {
        let size = u16::from_le_bytes([bytes[at], bytes[at + 1]]) as usize;
        if size < 4 || at + 2 + size > bytes.len() {
            return None;
        }
        let flag = u32::from_le_bytes(bytes[at + 2..at + 6].try_into().ok()?);
        if flag == query::CODE && size >= 8 {
            return Some(u32::from_le_bytes(bytes[at + 6..at + 10].try_into().ok()?));
        }
        at += 2 + size;
    }
    None
}

/// What `pilot` answers to the current step of `asked`.
fn ask(pilot: &mut dyn Policy, asked: &Asked) -> usize {
    pilot.choose(&asked.observation, asked.decision()).min(asked.decision().choices.len() - 1)
}

/// A step of a pilot's or a recorded seat, answered with `choice`: report
/// it when the seat is observed, and remember it.  Steps a network would
/// not be asked are passed over, as is an answer a network could not give
/// (a cancelled selection).
fn report(
    seat: &mut Seat,
    pending: &mut Pending,
    options: &Options,
    decisions: usize,
    choice: usize,
    pilot: Option<usize>,
    samples: &mut Vec<View>,
) {
    let decision = pending.asked.decision();
    let standing = standing(decision, options);
    if standing.real {
        if let Some(chosen) = standing.open.iter().position(|index| *index == choice) {
            if seat.observe {
                let mut shown = view::view(&pending.asked.observation, decision, &standing.open);
                seat.trail.stamp(&mut shown, &pending.asked.observation, pending.step, decisions);
                shown.label = Some(chosen);
                shown.chosen = Some(chosen);
                shown.pilot = pilot.and_then(|pilot| standing.open.iter().position(|index| *index == pilot));
                samples.push(shown);
            }
            seat.trail.remember(&decision.choices[choice], options.history);
        }
    }
    seat.trail.note(decision, choice);
    pending.step += 1;
}

#[cfg(test)]
mod tests {
    use super::Refreshes;
    use ygo_policies_ocgcore::message;
    use ygo_policies_ocgcore::wire::{msg, query};

    /// A location's refresh: the cards' codes, face-up in Attack Position.
    fn refresh(player: u8, location: u8, codes: &[u32]) -> Vec<u8> {
        let mut records = Vec::new();
        for code in codes {
            for (flag, value) in [(query::CODE, *code), (query::POSITION, 1)] {
                records.extend_from_slice(&8u16.to_le_bytes());
                records.extend_from_slice(&flag.to_le_bytes());
                records.extend_from_slice(&value.to_le_bytes());
            }
            records.extend_from_slice(&4u16.to_le_bytes());
            records.extend_from_slice(&query::END.to_le_bytes());
        }
        let mut bytes = vec![msg::UPDATE_DATA, player, location];
        bytes.extend_from_slice(&(records.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&records);
        bytes
    }

    #[test]
    fn a_refresh_kept_decodes_as_the_parser_decodes_it() {
        let mut refreshes = Refreshes::default();
        for bytes in [
            refresh(0, 2, &[1, 2]),
            refresh(0, 2, &[1, 2]),
            refresh(1, 2, &[1, 2]),
            refresh(0, 2, &[2]),
            refresh(0, 16, &[]),
            refresh(0, 2, &[2]),
            refresh(1, 2, &[1, 2]),
            refresh(0, 2, &[1, 2]),
        ] {
            assert_eq!(refreshes.decode(&bytes).unwrap(), message::parse(&bytes).unwrap());
        }
        // One kept a location: each player's hand and a Graveyard.
        assert_eq!(refreshes.last.len(), 3);
    }
}
