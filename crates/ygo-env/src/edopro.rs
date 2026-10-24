//! The caller's seat at an EDOPro table.
//!
//! `ygo-policies-edopro` joins a room hosted in EDOPro and pipes the duel's
//! messages through a seat of the policies.  The policy seated here is the
//! caller of an episode: what a network is not asked in a duel of the
//! trainer is answered as an episode answers it (the deck's pilot places the
//! cards, a single answer is taken), and what it is asked is shown as an
//! episode shows it, with the seat's own trail through the duel.  That
//! policy and what it reads from the messages (that a selection begins, and
//! that the other seat was asked one) are `ygo_model_seat::caller`, shared
//! with whatever else seats a checkpoint at a table of EDOPro's protocol.
//!
//! The room is joined on a thread of its own.  The caller answers on the
//! thread it called from, and is asked now and then whether to go on while
//! nothing happens: a session waits on a person.

use std::net::{Shutdown, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, sync_channel, RecvTimeoutError, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use ygo_model_seat::{Caller, Progress};
use ygo_policies::cards::CardDatabase;
use ygo_policies::registry;
use ygo_policies_duel::{Deck, Recorded, State};
use ygo_policies_edopro::log::{read_trace, unhex, DuelLog, Entry, Names};
use ygo_policies_edopro::{play, protocol, ydk, Config, Observer, Outcome};
use ygo_policies_ocgcore::message;
use ygo_policies_ocgcore::wire::msg;
use ygo_policies_ocgcore::Seat;

use crate::episode::{Environment, Options};
use crate::view::View;
use crate::Result;

/// How long nothing may happen before the caller is asked whether to go on.
const STILL: Duration = Duration::from_millis(250);

/// A room to join, and how.
#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Room {
    pub host: String,
    pub port: u16,
    pub password: String,
    pub room_id: u32,
    /// The player's name in the room.
    pub name: String,
    /// EDOPro's `ClientVersion`.
    pub version: u32,
    /// Duels to play, joining the room again for each.
    pub duels: u32,
    /// Take the first turn when the seat wins rock-paper-scissors.
    pub go_first: bool,
    /// Seconds to keep trying to reach the room.
    pub wait: u64,
    /// Where each duel's log and trace are written: nowhere when left out.
    pub log_dir: Option<PathBuf>,
    /// What the logs are named after, behind the time.
    pub tag: String,
    /// The seed of the pilot beside the seat: the time when left out.
    pub seed: Option<u64>,
}

impl Default for Room {
    fn default() -> Self {
        Room {
            host: "127.0.0.1".into(),
            port: 7911,
            password: String::new(),
            room_id: 0,
            name: "[AI] model".into(),
            version: protocol::DEFAULT_VERSION,
            duels: 1,
            go_first: true,
            wait: 60,
            log_dir: None,
            tag: "model".into(),
            seed: None,
        }
    }
}

/// How a duel in a room ended.
#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
pub struct Played {
    /// `won`, `lost` or `draw`, or why the duel ended without a result.
    pub result: String,
    /// The seat the caller had: 0 went first.
    pub seat: Option<u8>,
    /// The engine's reason for the result.
    pub reason: Option<u8>,
    /// Decisions the caller took.
    pub decisions: usize,
    /// The duel's log, when one is kept: its `.log` and `.trace` files.
    pub log: Option<String>,
}

/// A trace fed to the seat again.
#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
pub struct Replayed {
    /// Selections the seat answered.
    pub answers: usize,
    /// Decisions the caller took for them.
    pub decisions: usize,
    /// Answers that are not the recorded ones.
    pub differ: usize,
}

/// A duel as EDOPro keeps one in a replay (a `.yrp`): how it began, each
/// player's cards in the order the engine was given them, and every answer
/// in order.  Who gave an answer is not kept: the duel played again tells.
#[derive(Deserialize, Clone, Debug)]
pub struct Kept {
    /// The engine's random seed.
    pub seed: [u64; 4],
    /// OCGCore's duel flags.
    pub flags: u64,
    pub start_lp: u32,
    pub start_hand: u32,
    pub draw_count: u32,
    /// The cards of the player who went first, then the other's.
    pub decks: [Cards; 2],
    /// Each answer's bytes, in hex.
    pub responses: Vec<String>,
}

/// A player's cards as a replay lists them.
#[derive(Deserialize, Clone, Debug)]
pub struct Cards {
    pub main: Vec<u32>,
    pub extra: Vec<u32>,
}

/// A kept duel played again.
#[derive(Serialize, Clone, Debug)]
pub struct Reread {
    /// The duel as a record of the policies' form, which an episode plays
    /// again: each answer with the player who gave it.
    pub record: serde_json::Value,
    /// The seat that won, 2 for a draw; `None` when the replay stops before
    /// the duel ends, as one does that a player gave up.
    pub winner: Option<u8>,
    pub reason: Option<u8>,
    pub turns: u32,
}

/// What the caller is told or asked.
pub enum Event<'a> {
    /// Nothing happened for a while.
    Still,
    /// How the session goes on.
    Note(&'a str),
    /// A decision: the answer is one of its choices.
    Decision(&'a View),
}

/// The caller: it answers a decision with one of its choices, and whatever
/// it is told or asked, an error from it ends the session.
pub type Ask<'a> = dyn FnMut(Event<'_>) -> Result<usize> + 'a;

/// What reaches the caller's thread from the one at the table.
enum Request {
    Note(String),
    Decision(Box<View>, SyncSender<usize>),
}

/// Tells the seat's progress of the duel's messages, from which it reads
/// what an episode knows from running the duel, and keeps the duel's log.
struct Watch {
    progress: Progress,
    log: Option<DuelLog>,
    requests: Sender<Request>,
}

impl Observer for Watch {
    fn received(&mut self, message: &[u8]) {
        self.progress.received(message);
        if let Some(log) = &mut self.log {
            log.received(message);
        }
    }

    fn answered(&mut self, seat: &Seat, response: &[u8]) {
        self.progress.answered();
        if let Some(log) = &mut self.log {
            log.answered(seat, response);
        }
    }

    fn note(&mut self, text: &str) {
        if let Some(log) = &mut self.log {
            log.note(text);
        }
        let _ = self.requests.send(Request::Note(text.to_owned()));
    }
}

/// The seats of a session: each duel a new one, its trail empty and the
/// pilot beside it seeded anew.
struct Seats {
    deck: String,
    db: Arc<dyn CardDatabase>,
    progress: Progress,
    requests: Sender<Request>,
    seed: u64,
}

impl Seats {
    fn new(deck: &str, db: Arc<dyn CardDatabase>, requests: Sender<Request>, seed: Option<u64>) -> Result<Self> {
        if registry::find(deck).is_none() {
            return Err(format!("no policy {deck:?}"));
        }
        let now = || SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |elapsed| elapsed.as_nanos() as u64);
        Ok(Seats { deck: deck.to_owned(), db, progress: Progress::default(), requests, seed: seed.unwrap_or_else(now) })
    }

    /// The policy of the seat is the caller of an episode, as
    /// `Episode::settle` treats it, with the deck's pilot asked alongside:
    /// its decisions go to the caller's thread.
    fn seat(&mut self) -> Seat {
        self.seed = self.seed.wrapping_add(1);
        self.progress.restart();
        let pilot = registry::create_seeded(&self.deck, self.db.clone(), self.seed).expect("the deck has a pilot");
        let requests = self.requests.clone();
        let ask = move |shown: View| {
            let (reply, answer) = sync_channel(1);
            let asked = requests.send(Request::Decision(Box::new(shown), reply)).is_ok();
            // No answer when the caller has stopped: the session is ending.
            answer.recv().ok().filter(|_| asked)
        };
        Seat::new(Box::new(Caller::new(pilot, Options::default(), self.progress.clone(), ask)), self.db.clone(), None)
    }

    fn watch(&self, log: Option<DuelLog>) -> Watch {
        Watch { progress: self.progress.clone(), log, requests: self.requests.clone() }
    }

    fn decisions(&self) -> usize {
        self.progress.decisions()
    }
}

/// What the thread at the table shares with the caller's.
struct Table<'a> {
    requests: Sender<Request>,
    /// The connection to the room, for the caller's thread to cut.
    live: &'a Mutex<Option<TcpStream>>,
    stopping: &'a AtomicBool,
}

impl Table<'_> {
    fn note(&self, text: String) {
        let _ = self.requests.send(Request::Note(text));
    }
}

/// Run `work` on a thread of its own and serve what it asks of the caller.
fn serve<T: Send>(ask: &mut Ask<'_>, work: impl FnOnce(Table<'_>) -> Result<T> + Send) -> Result<T> {
    let (requests, asked) = channel();
    let live = Mutex::new(None);
    let stopping = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let table = Table { requests, live: &live, stopping: &stopping };
        let working = scope.spawn(move || work(table));
        let mut failure: Option<String> = None;
        let heard = |told: Result<usize>, failure: &mut Option<String>| {
            if let Err(message) = told {
                *failure = Some(message);
                stopping.store(true, Ordering::SeqCst);
                if let Some(stream) = live.lock().expect("the connection is not poisoned").as_ref() {
                    let _ = stream.shutdown(Shutdown::Both);
                }
            }
        };
        loop {
            match asked.recv_timeout(STILL) {
                Ok(_) if failure.is_some() => {}
                Ok(Request::Note(text)) => heard(ask(Event::Note(&text)), &mut failure),
                Ok(Request::Decision(shown, reply)) => match ask(Event::Decision(&shown)) {
                    Ok(index) if index < shown.choices.len() => {
                        let _ = reply.send(index);
                    }
                    Ok(index) => {
                        let offered = shown.choices.len();
                        heard(Err(format!("choice {index} of a decision that offers {offered}")), &mut failure);
                    }
                    told => heard(told, &mut failure),
                },
                Err(RecvTimeoutError::Timeout) if failure.is_none() => heard(ask(Event::Still), &mut failure),
                Err(RecvTimeoutError::Timeout) => {}
                // The work is done: nothing holds the other end any more.
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        let done = working.join().map_err(|_| "the session stopped on a panic".to_owned())?;
        match failure {
            Some(message) => Err(message),
            None => done,
        }
    })
}

/// The duels of a room, one after the other: `edopro-bot`'s own loop around
/// `play`, with the caller's seat.
fn session(
    table: Table<'_>,
    room: &Room,
    deck: &str,
    list: ydk::Deck,
    db: Arc<dyn CardDatabase>,
    cards: &Path,
) -> Result<Vec<Played>> {
    let config = Config {
        name: room.name.clone(),
        room_id: room.room_id,
        password: room.password.clone(),
        version: room.version,
        deck: list,
        go_first: room.go_first,
    };
    let mut seats = Seats::new(deck, db, table.requests.clone(), room.seed)?;
    let stopped = || table.stopping.load(Ordering::SeqCst);
    let mut played = Vec::new();
    for duel in 1..=room.duels {
        let started = Instant::now();
        let mut told = false;
        let mut stream = loop {
            if stopped() {
                return Ok(played);
            }
            match TcpStream::connect((room.host.as_str(), room.port)) {
                Ok(stream) => break stream,
                Err(error) if started.elapsed() < Duration::from_secs(room.wait) => {
                    if !std::mem::replace(&mut told, true) {
                        table.note(format!("waiting for EDOPro at {}:{} ({error})", room.host, room.port));
                    }
                    std::thread::sleep(STILL);
                }
                Err(error) => return Err(format!("cannot reach EDOPro at {}:{}: {error}", room.host, room.port)),
            }
        };
        let _ = stream.set_nodelay(true);
        *table.live.lock().expect("the connection is not poisoned") = stream.try_clone().ok();
        table.note(format!("duel {duel} of {}: at {}:{} with {deck}", room.duels, room.host, room.port));
        let mut stem = None;
        let log = match &room.log_dir {
            Some(folder) => {
                std::fs::create_dir_all(folder).map_err(|e| format!("{}: {e}", folder.display()))?;
                let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |elapsed| elapsed.as_secs());
                let path = folder.join(format!("{stamp}-{}", room.tag));
                let log = DuelLog::create(&path, Names::open(cards).unwrap_or_default()).map_err(|e| e.to_string())?;
                stem = Some(path.display().to_string());
                Some(log)
            }
            None => None,
        };
        let mut watch = seats.watch(log);
        let outcome = play(&mut stream, &config, &mut || seats.seat(), &mut watch);
        drop(watch);
        *table.live.lock().expect("the connection is not poisoned") = None;
        if stopped() {
            return Ok(played);
        }
        let (result, seat, reason) = match outcome.map_err(|e| format!("connection lost: {e}"))? {
            Outcome::Finished { winner: 2, seat, reason } => ("draw".to_owned(), seat, Some(reason)),
            Outcome::Finished { winner, seat: Some(seat), reason } => {
                (if seat == winner { "won" } else { "lost" }.to_owned(), Some(seat), Some(reason))
            }
            Outcome::Finished { winner, seat: None, reason } => (format!("player {winner} won"), None, Some(reason)),
            Outcome::Ended(why) => (why, None, None),
        };
        table.note(format!("duel {duel}: {result}"));
        played.push(Played { result, seat, reason, decisions: seats.decisions(), log: stem });
    }
    Ok(played)
}

/// A trace of `edopro-bot` fed to the caller's seat: `edopro-bot --replay`,
/// with the caller asked what a pilot was asked then.  What it answers is
/// not played: the trace goes on as it was recorded.
fn replay(table: Table<'_>, deck: &str, trace: &Path, db: Arc<dyn CardDatabase>) -> Result<Replayed> {
    let entries = read_trace(trace).map_err(|e| format!("{}: {e}", trace.display()))?;
    let mut seats = Seats::new(deck, db, table.requests.clone(), Some(0))?;
    let mut watch = seats.watch(None);
    let mut seat: Option<Seat> = None;
    let mut pending: Option<Vec<u8>> = None;
    let mut found = Replayed { answers: 0, decisions: 0, differ: 0 };
    for entry in entries {
        if table.stopping.load(Ordering::SeqCst) {
            break;
        }
        match entry {
            Entry::Received(message) => {
                watch.received(&message);
                if message.first() == Some(&msg::START) {
                    found.decisions += seats.decisions();
                    seat = Some(seats.seat());
                }
                let Some(seat) = seat.as_mut() else { continue };
                if let Some(response) = seat.feed(&message).map_err(|e| e.to_string())? {
                    watch.answered(seat, &response);
                    found.answers += 1;
                    pending = Some(response);
                }
            }
            Entry::Sent(recorded) => {
                if pending.take().is_some_and(|ours| ours != recorded) {
                    found.differ += 1;
                }
            }
        }
    }
    found.decisions += seats.decisions();
    Ok(found)
}

impl Environment {
    /// Play a duel EDOPro kept again, to tell who gave each of its answers.
    /// It is the same duel only on the engine and the card scripts it was
    /// played on: an answer the engine does not take, or answers left when
    /// the duel ends, say that it is another.
    pub fn kept(&self, kept: &Kept) -> Result<Reread> {
        // A replay lists the cards as the engine was given them, and a
        // record gives the engine the last card of its lists first.
        let deck = |cards: &Cards| Deck {
            main: cards.main.iter().rev().copied().collect(),
            extra: cards.extra.iter().rev().copied().collect(),
        };
        let mut recorded = Recorded {
            seed: kept.seed,
            flags: kept.flags,
            life_points: [kept.start_lp; 2],
            hand: kept.start_hand,
            per_turn: kept.draw_count,
            decks: [deck(&kept.decks[0]), deck(&kept.decks[1])],
            responses: Vec::new(),
        };
        let mut duel = recorded.duel(&self.core)?;
        let mut answers = kept.responses.iter();
        let (mut asked, mut turns) = (None, 0);
        let outcome = loop {
            let step = duel.step()?;
            for sent in step.messages.iter().filter(|sent| !sent.refresh) {
                match sent.bytes.first() {
                    Some(&msg::RETRY) => {
                        let given = recorded.responses.len();
                        return Err(format!(
                            "the engine does not take answer {given} of the replay: not the same duel"
                        ));
                    }
                    Some(&msg::NEW_TURN) => turns += 1,
                    _ => {}
                }
                if let Some(player) = message::parse(&sent.bytes).map_err(|e| e.to_string())?.responder() {
                    asked = Some(player);
                }
            }
            match step.state {
                State::Over(outcome) => break outcome,
                State::Running => {}
                State::Awaiting => {
                    // No answer left: the duel was not played to its end.
                    let Some(answer) = answers.next() else { break None };
                    let player = asked.ok_or("the engine waits and has asked no player")?;
                    let bytes =
                        unhex(answer).ok_or_else(|| format!("an answer of the replay is not hex: {answer:?}"))?;
                    duel.respond(&bytes);
                    recorded.responses.push((player, bytes));
                }
            }
        };
        if answers.len() > 0 {
            return Err(format!(
                "{} answers of the replay are left when the duel ends: not the same duel",
                answers.len()
            ));
        }
        Ok(Reread {
            record: recorded.to_json(),
            winner: outcome.map(|outcome| outcome.winner),
            reason: outcome.map(|outcome| outcome.reason),
            turns,
        })
    }

    /// Join an EDOPro room with the deck `deck` and play its duels, `ask`
    /// deciding what the caller of an episode decides.
    pub fn edopro(&self, deck: &str, room: &Room, ask: &mut Ask<'_>) -> Result<Vec<Played>> {
        let found = self.decks.get(deck).ok_or_else(|| format!("no deck {deck:?} in the pool"))?;
        let list = ydk::Deck { main: found.main.clone(), extra: found.extra.clone(), side: Vec::new() };
        let (db, cards) = (self.db.clone(), self.cards.clone());
        serve(ask, move |table| session(table, room, deck, list, db, &cards))
    }

    /// Feed a trace of `edopro-bot` to the seat of `deck`, `ask` deciding
    /// what the caller of an episode decides.
    pub fn edopro_replay(&self, deck: &str, trace: &Path, ask: &mut Ask<'_>) -> Result<Replayed> {
        let db = self.db.clone();
        serve(ask, move |table| replay(table, deck, trace, db))
    }
}
