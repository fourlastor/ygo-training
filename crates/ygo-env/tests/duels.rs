//! Real duels on the policies' engine, cards, scripts and decks.
//!
//! The reference is the duel `policy-bench` plays: two policies seated in
//! `ygo_policies_ocgcore::Seat`, fed and answered as the bench does it,
//! with the bench's digest of every question and answer.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use ygo_env::{Control, Environment, Options, Over, SeatSetup, Setup, View};
use ygo_policies::{cards::CardDatabase, registry};
use ygo_policies_duel::{Core, Deck, DuelOptions, Recorded, State};
use ygo_policies_ocgcore::{wire::msg, Seat, SqliteCards};
use ygo_policies_search::{LocalPilots, SearchOptions, SearchSeat};

const IDLE: u8 = 1;
const BATTLE: u8 = 2;
const CHAIN: u8 = 3;
const PLACE: u8 = 10;
const ATTACK_TARGET: u8 = 22;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../native/ygo-policies")
}

fn cards() -> PathBuf {
    root().join("vendor/BabelCdb/cards.cdb")
}

fn environment() -> Environment {
    Environment::open(&cards(), &root().join("vendor/CardScripts"), &root().join("decks")).unwrap()
}

/// What tells two duels apart: the digest, the winner, the turns and the answers.
type Played = (String, Option<u8>, u32, usize);

fn played(over: &Over) -> Played {
    (over.digest.clone(), over.winner, over.turns, over.decisions)
}

/// The duel of `policy-bench` with this seed, between the policies `ids`.
fn bench(environment: &Environment, seed: u64, ids: [&str; 2]) -> Played {
    let core = Core::open(None, &cards(), &root().join("vendor/CardScripts")).unwrap();
    let db: Arc<dyn CardDatabase> = Arc::new(SqliteCards::open(cards()).unwrap());
    let list = |id: &str| {
        let entry = environment.catalog().iter().find(|entry| entry.id == id).unwrap();
        Deck { main: entry.main.clone(), extra: entry.extra.clone() }
    };
    let mut duel = core.deal(&DuelOptions::seeded(seed), &[list(ids[0]), list(ids[1])]).unwrap();
    let mut seats = Vec::new();
    for (player, id) in ids.iter().enumerate() {
        let policy = registry::create_seeded(id, db.clone(), seed + player as u64).unwrap();
        let mut seat = Seat::new(policy, db.clone(), Some(player as u8));
        seat.feed(&duel.start_message(player as u8)).unwrap();
        seats.push(seat);
    }
    let (mut decisions, mut turns, mut digest) = (0, 0u32, 0xcbf29ce484222325u64);
    loop {
        let step = duel.step().unwrap();
        let mut response = None;
        for sent in &step.messages {
            if !sent.refresh && sent.bytes[0] == msg::NEW_TURN {
                turns += 1;
            }
            for seat in seats.iter_mut() {
                if let Some(answer) = seat.feed(&sent.bytes).unwrap() {
                    for byte in sent.bytes.iter().chain(&answer) {
                        digest = (digest ^ *byte as u64).wrapping_mul(0x100000001b3);
                    }
                    response = Some(answer);
                }
            }
        }
        if let State::Over(outcome) = step.state {
            return (format!("{digest:016x}"), outcome.map(|outcome| outcome.winner), turns, decisions);
        }
        if let Some(answer) = response {
            duel.respond(&answer);
            decisions += 1;
        }
    }
}

/// A search's settings for a test: a few worlds and candidates, so that a
/// duel searched at every decision is played in seconds.
const QUICK: &str = r#"{"stages": [2, 2, 4], "kept": 1, "candidates": 3}"#;

/// The duel of `policy-bench generic --method staged` with this seed: the
/// staged search in seat `searcher`, piloted by `pilot` and taking the other
/// player to be `model`, its seat fed every message as the bench feeds it,
/// against the pilot of the other deck.
fn searched_bench(environment: &Environment, seed: u64, ids: [&str; 2], searcher: usize, pilot: &str, model: Option<&str>) -> Played {
    let core = Core::open(None, &cards(), &root().join("vendor/CardScripts")).unwrap();
    let db: Arc<dyn CardDatabase> = Arc::new(SqliteCards::open(cards()).unwrap());
    let list = |id: &str| {
        let entry = environment.catalog().iter().find(|entry| entry.id == id).unwrap();
        Deck { main: entry.main.clone(), extra: entry.extra.clone() }
    };
    let duel = Rc::new(RefCell::new(core.deal(&DuelOptions::seeded(seed), &[list(ids[0]), list(ids[1])]).unwrap()));
    let other = 1 - searcher;
    let policy = registry::create_seeded(ids[other], db.clone(), seed + other as u64).unwrap();
    let mut seat = Seat::new(policy, db.clone(), Some(other as u8));
    seat.feed(&duel.borrow().start_message(other as u8)).unwrap();
    let options = SearchOptions::from_json(&serde_json::from_str(QUICK).unwrap()).unwrap();
    let pilots = LocalPilots { cards: db.clone() };
    let mut search = SearchSeat::with_pilots(duel.clone(), pilots, searcher as u8, seed, options, pilot, model).unwrap();
    let start = duel.borrow().start_message(searcher as u8);
    search.feed(&start).unwrap();
    let (mut decisions, mut turns, mut digest) = (0, 0u32, 0xcbf29ce484222325u64);
    loop {
        let step = duel.borrow_mut().step().unwrap();
        let mut response = None;
        for sent in &step.messages {
            if !sent.refresh && sent.bytes[0] == msg::NEW_TURN {
                turns += 1;
            }
            let answers = [search.feed(&sent.bytes).unwrap(), seat.feed(&sent.bytes).unwrap()];
            for answer in answers.into_iter().flatten() {
                for byte in sent.bytes.iter().chain(&answer) {
                    digest = (digest ^ *byte as u64).wrapping_mul(0x100000001b3);
                }
                response = Some(answer);
            }
        }
        if let State::Over(outcome) = step.state {
            return (format!("{digest:016x}"), outcome.map(|outcome| outcome.winner), turns, decisions);
        }
        if let Some(answer) = response {
            duel.borrow_mut().respond(&answer);
            decisions += 1;
        }
    }
}

fn seat(control: Control, alone: bool, observe: bool) -> SeatSetup {
    SeatSetup { control, policy: None, model: None, search: None, alone, observe }
}

fn setup(seed: u64, ids: [&str; 2], seats: [SeatSetup; 2]) -> Setup {
    Setup {
        seed,
        decks: ids.map(str::to_owned),
        lists: [None, None],
        seats,
        limit: 4096,
        life_points: None,
        record: None,
        keep_record: false,
        options: Options::default(),
        whole: false,
        whole_deck: 0,
        follow: false,
    }
}

/// Run a duel to its end, answering the caller's decisions with `decide`:
/// the observed decisions, the caller's own, and how it ended.
fn play(environment: &Environment, setup: &Setup, mut decide: impl FnMut(&View) -> usize) -> (Vec<View>, Vec<View>, Over) {
    let mut episode = environment.start(setup).unwrap();
    let (mut samples, mut decisions) = (Vec::new(), Vec::new());
    loop {
        let event = episode.advance().unwrap();
        samples.extend(event.samples);
        if let Some(decision) = event.decision {
            episode.choose(decide(&decision)).unwrap();
            decisions.push(decision);
        }
        if let Some(over) = event.over {
            return (samples, decisions, over);
        }
    }
}

const DUELS: [(u64, [&str; 2]); 4] =
    [(7, ["blackwing", "monarch"]), (8, ["monarch", "blackwing"]), (11, ["heroes", "spellcaster"]), (12, ["lightsworn", "gladiator"])];

#[test]
fn pilots_play_the_duel_of_the_bench() {
    let environment = environment();
    for (seed, ids) in DUELS {
        let pilots = [seat(Control::Pilot, false, false), seat(Control::Pilot, false, false)];
        let (samples, decisions, over) = play(&environment, &setup(seed, ids, pilots), |_| unreachable!());
        assert_eq!(played(&over), bench(&environment, seed, ids), "seed {seed}, {ids:?}");
        assert!(samples.is_empty() && decisions.is_empty());
        assert!(over.winner.is_some());
    }
}

#[test]
fn a_search_plays_its_seat_as_the_bench_seats_it() {
    let environment = environment();
    // The generic pilot, the other player taken to be it; then the deck's
    // own specialist, the other player taken to be the other deck's.
    let searches = [(7, ["blackwing", "monarch"], 0, "generic", None), (8, ["monarch", "blackwing"], 1, "blackwing", Some("monarch"))];
    for (seed, ids, searcher, pilot, model) in searches {
        let mut seats = [seat(Control::Pilot, false, false), seat(Control::Pilot, false, false)];
        seats[searcher] = SeatSetup {
            control: Control::Search,
            policy: Some(pilot.to_owned()),
            model: model.map(str::to_owned),
            search: Some(serde_json::from_str(QUICK).unwrap()),
            alone: true,
            observe: false,
        };
        let (samples, decisions, over) = play(&environment, &setup(seed, ids, seats), |_| unreachable!());
        assert_eq!(played(&over), searched_bench(&environment, seed, ids, searcher, pilot, model), "seed {seed}, {ids:?}");
        assert!(samples.is_empty() && decisions.is_empty());
        let told = &over.search.as_ref().expect("the search says what it did")[0];
        assert_eq!(told["player"], searcher);
        assert!(told["searched"].as_u64().unwrap() > 0, "{told}");
    }
}

#[test]
fn a_caller_that_follows_its_pilot_plays_the_same_duel() {
    let environment = environment();
    let mut targets = 0;
    for (seed, ids) in DUELS {
        for caller in 0..2 {
            let mut seats = [seat(Control::Pilot, false, false), seat(Control::Pilot, false, false)];
            seats[caller] = seat(Control::Caller, false, false);
            let (_, decisions, over) = play(&environment, &setup(seed, ids, seats), |decision| {
                decision.label.expect("the pilot's answer is one a network may give")
            });
            assert_eq!(played(&over), bench(&environment, seed, ids), "seed {seed}, {ids:?}, caller {caller}");
            assert!(!decisions.is_empty());
            for decision in &decisions {
                assert_eq!(decision.player as usize, caller);
                assert!(decision.choices.len() > 1, "a decision with one answer is not asked");
                assert_ne!(decision.question.kind, PLACE, "the zone is the pilot's choice");
                assert!(decision.history.len() <= 32);
                // A command has no card before it; the target of an attack
                // is asked with the attacker named.
                if matches!(decision.question.kind, IDLE | BATTLE | CHAIN) {
                    assert!(decision.acting.is_none());
                }
                if decision.question.hint == ATTACK_TARGET {
                    let attacker = decision.acting.as_ref().expect("an attack target follows an attack");
                    assert!(attacker.code != 0 && attacker.card >= 0);
                    targets += 1;
                }
            }
            // Steps of one answer share its transaction, and a command opens one.
            for pair in decisions.windows(2) {
                if pair[0].answer == pair[1].answer {
                    assert_eq!(pair[0].transaction, pair[1].transaction);
                    assert_eq!(pair[0].step + 1, pair[1].step);
                } else if matches!(pair[1].question.kind, IDLE | BATTLE | CHAIN) {
                    assert!(pair[1].transaction > pair[0].transaction);
                }
            }
        }
    }
    assert!(targets > 0, "no duel asked for an attack target");
}

/// The file of the list the deck `id` plays.
fn list_file(environment: &Environment, id: &str) -> PathBuf {
    let entry = environment.catalog().iter().find(|entry| entry.id == id).unwrap();
    root().join("decks").join(format!("{}.ydk", entry.deck))
}

#[test]
fn a_list_read_from_a_file_is_played_by_its_decks_pilot() {
    let environment = environment();
    let pilots = || [seat(Control::Pilot, false, false), seat(Control::Pilot, false, false)];
    // Each seat's list read from its own file: the duel of the bench.
    for (seed, ids) in DUELS {
        let mut read = setup(seed, ids, pilots());
        read.lists = ids.map(|id| Some(list_file(&environment, id)));
        let (_, _, over) = play(&environment, &read, |_| unreachable!());
        assert_eq!(played(&over), bench(&environment, seed, ids), "seed {seed}, {ids:?}");
    }
    // A list no policy was written for, in Blackwing's seat: Blackwing's
    // pilot plays it, and is asked alongside a caller in the same seat.
    let variant = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/decks/darker-than-black.ydk");
    let (seed, ids) = DUELS[0];
    let mut piloted = setup(seed, ids, pilots());
    piloted.lists[0] = Some(variant.clone());
    let (_, _, over) = play(&environment, &piloted, |_| unreachable!());
    assert_ne!(over.digest, bench(&environment, seed, ids).0, "another list, another duel");
    let mut called = setup(seed, ids, [seat(Control::Caller, false, false), seat(Control::Pilot, false, false)]);
    called.lists[0] = Some(variant);
    let (_, decisions, followed) =
        play(&environment, &called, |decision| decision.label.expect("the pilot's answer is one a network may give"));
    assert_eq!(played(&followed), played(&over));
    assert!(!decisions.is_empty());
    // A list that cannot be read is told by its file.
    let mut missing = setup(seed, ids, pilots());
    missing.lists[1] = Some(root().join("decks/no such list.ydk"));
    let error = environment.start(&missing).err().expect("the list cannot be read");
    assert!(error.contains("no such list.ydk"), "{error}");
}

/// What two reports of the same decisions have in common.
fn outline(views: &[View]) -> Vec<(usize, usize, u32, usize, Option<usize>, Option<usize>)> {
    views.iter().map(|view| (view.answer, view.step, view.transaction, view.choices.len(), view.label, view.chosen)).collect()
}

#[test]
fn a_record_plays_again_with_the_same_decisions() {
    let environment = environment();
    for (seed, ids) in DUELS {
        let mut first = setup(seed, ids, [seat(Control::Pilot, false, true), seat(Control::Pilot, false, false)]);
        first.keep_record = true;
        let (samples, _, over) = play(&environment, &first, |_| unreachable!());
        assert!(!samples.is_empty());
        assert!(samples.iter().all(|sample| sample.player == 0 && sample.label == sample.chosen && sample.pilot == sample.label));

        let mut again = setup(seed, ids, [seat(Control::Record, false, true), seat(Control::Record, true, false)]);
        again.record = over.record.clone();
        let (replayed, _, end) = play(&environment, &again, |_| unreachable!());
        assert_eq!(played(&end), played(&over), "seed {seed}, {ids:?}");
        assert_eq!(end.departures, 0);
        assert_eq!(outline(&replayed), outline(&samples));
        // What the seat is shown is the same too.
        let shown = |views: &[View]| views.iter().map(|view| serde_json::to_string(view).unwrap()).collect::<Vec<_>>();
        assert_eq!(shown(&replayed), shown(&samples));
    }
}

/// A card or sum selection answered as `Sequential` writes it (an `i32` 0
/// with indices of 32 bits, or 2 with indices of 8), written again as
/// EDOPro's own client writes one: an `i32` 3 and one bit a candidate.
fn as_a_client_writes_it(response: &str) -> Option<String> {
    let byte = |at: usize| u8::from_str_radix(&response[2 * at..2 * at + 2], 16).unwrap();
    let bytes: Vec<u8> = (0..response.len() / 2).map(byte).collect();
    if bytes.len() < 8 {
        return None;
    }
    let number = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    let picked: Vec<usize> = match (number(0), number(4)) {
        (0, count) if bytes.len() == 8 + 4 * count => (0..count).map(|index| number(8 + 4 * index)).collect(),
        (2, count) if bytes.len() == 8 + count => bytes[8..].iter().map(|index| *index as usize).collect(),
        _ => return None,
    };
    let mut bits = vec![0u8; picked.iter().max().map_or(1, |last| last / 8 + 1)];
    for index in picked {
        bits[index / 8] |= 1 << (index % 8);
    }
    Some([3u8, 0, 0, 0].iter().chain(&bits).map(|byte| format!("{byte:02x}")).collect())
}

#[test]
fn a_record_with_a_persons_answers_plays_again() {
    // A duel a person played in EDOPro comes back with the cards of a
    // selection named by bits, where a policy names them by number.
    let environment = environment();
    let mut rewritten = 0;
    for (seed, ids) in DUELS {
        let mut first = setup(seed, ids, [seat(Control::Pilot, false, true), seat(Control::Pilot, false, true)]);
        first.keep_record = true;
        let (samples, _, over) = play(&environment, &first, |_| unreachable!());

        let mut record = over.record.clone().unwrap();
        for answer in record["responses"].as_array_mut().unwrap() {
            if let Some(written) = as_a_client_writes_it(answer[1].as_str().unwrap()) {
                answer[1] = written.into();
                rewritten += 1;
            }
        }
        let mut again = setup(seed, ids, [seat(Control::Record, false, true), seat(Control::Record, false, true)]);
        again.record = Some(record);
        let (replayed, _, end) = play(&environment, &again, |_| unreachable!());
        // The same duel, decision for decision, and the pilots beside the
        // seats find their own answers in it.  Its digest is another: it
        // holds the answers as they are written.
        assert_eq!((end.winner, end.turns, end.decisions), (over.winner, over.turns, over.decisions), "seed {seed}");
        assert_eq!(end.departures, 0);
        assert_eq!(outline(&replayed), outline(&samples));
    }
    assert!(rewritten > 20, "{rewritten} selections of cards in these duels");
}

#[test]
fn a_record_shows_where_it_left_the_pilot() {
    let environment = environment();
    for (seed, ids) in DUELS {
        // A caller that follows its pilot, except at its tenth command.
        let mut first = setup(seed, ids, [seat(Control::Caller, false, false), seat(Control::Pilot, false, false)]);
        first.keep_record = true;
        let (mut commands, mut left) = (0, None);
        let (_, _, over) = play(&environment, &first, |decision| {
            let label = decision.label.expect("the pilot's answer is one a network may give");
            if matches!(decision.question.kind, IDLE | BATTLE | CHAIN) {
                commands += 1;
                if commands == 10 {
                    left = Some(decision.answer);
                    return (label + 1) % decision.choices.len();
                }
            }
            label
        });
        let left = left.expect("the duel has ten commands");

        let mut again = setup(seed, ids, [seat(Control::Record, false, true), seat(Control::Record, true, false)]);
        again.record = over.record.clone();
        let (samples, _, end) = play(&environment, &again, |_| unreachable!());
        assert_eq!(played(&end), played(&over), "seed {seed}, {ids:?}");
        assert_eq!(end.departures, 1);
        let departed: Vec<&View> = samples.iter().filter(|sample| sample.label != sample.pilot).collect();
        assert_eq!(departed.len(), 1);
        assert_eq!((departed[0].answer, departed[0].step), (left, 0));
        assert!(departed[0].pilot.is_some());
    }
}

/// The policies' scripts with that of Blackwing - Sirocco the Dawn made to
/// fail from the 20th call of any of its functions on: the card plays as
/// written until then, which comes at one point of a seeded duel.  A folder
/// of links under the tests' own, the failing script at its root, which the
/// engine reads before `official/`.
fn scripts_failing_at_sirocco() -> PathBuf {
    let scripts = root().join("vendor/CardScripts");
    let folder = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("a-script-that-fails");
    if folder.exists() {
        std::fs::remove_dir_all(&folder).unwrap();
    }
    std::fs::create_dir_all(&folder).unwrap();
    for entry in std::fs::read_dir(&scripts).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|extension| extension == "lua") {
            std::os::unix::fs::symlink(&path, folder.join(path.file_name().unwrap())).unwrap();
        }
    }
    for kept in ["official", "pre-errata"] {
        std::os::unix::fs::symlink(scripts.join(kept), folder.join(kept)).unwrap();
    }
    let script = std::fs::read_to_string(scripts.join("official/c75498415.lua")).unwrap();
    let failing = "
local calls = 0
for name, f in pairs(s) do
	if type(f) == \"function\" and name ~= \"initial_effect\" then
		s[name] = function(...)
			calls = calls + 1
			if calls >= 20 then error(\"a script that fails, for a test\") end
			return f(...)
		end
	end
end
";
    std::fs::write(folder.join("c75498415.lua"), script + failing).unwrap();
    folder
}

#[test]
fn a_duel_whose_card_script_fails_ends_cut_off_with_what_failed_and_its_record() {
    let pilots = [seat(Control::Pilot, false, false), seat(Control::Pilot, false, false)];
    let duel = setup(7, ["blackwing", "monarch"], pilots);
    // As written, the duel ends and reports nothing.
    let (_, _, over) = play(&environment(), &duel, |_| 0);
    assert!(over.failure.is_none() && over.record.is_none());
    // With Sirocco's script failing it is cut off, as the limit cuts a duel
    // off, with the engine's error and the record of the duel so far.
    let scripts = scripts_failing_at_sirocco();
    let failing = Environment::open(&cards(), &scripts, &root().join("decks")).unwrap();
    let (_, _, over) = play(&failing, &duel, |_| 0);
    assert!(over.limit && over.winner.is_none());
    let failure = over.failure.expect("what the engine reported");
    assert!(failure.contains("c75498415.lua") && failure.contains("a script that fails, for a test"), "{failure}");
    let record = Recorded::from_json(&over.record.expect("the duel's record")).unwrap();
    assert!(over.decisions > 0, "the script fails in the course of the duel");
    assert_eq!(record.responses.len(), over.decisions);
    // The record plays the duel again to the same failure, on the same scripts.
    let core = Core::open(None, &cards(), &scripts).unwrap();
    let mut answers = 0;
    let again = core.replay(&record, None, &mut |shown| {
        if let ygo_policies_duel::Replayed::Answer(..) = shown {
            answers += 1;
        }
    });
    assert_eq!(answers, over.decisions);
    assert_eq!(again, Err(failure));
}
