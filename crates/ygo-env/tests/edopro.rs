//! At an EDOPro table the caller is shown what it is shown in an episode.
//!
//! A duel is played as an episode, the caller answering by a rule.  Then a
//! scripted EDOPro room serves the same duel to the client: the engine's
//! messages for the seat, `MSG_WAITING` where the other seat is asked, and
//! that seat's recorded answers.  The seat there has to give the answers of
//! the episode's, from the same arrays.

use std::collections::VecDeque;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;

use ygo_env::edopro::{Event, Kept, Room};
use ygo_env::pack::{Array, Packed};
use ygo_env::{Control, Environment, Options, SeatSetup, Setup, View};
use ygo_policies_duel::{Core, Recorded, State};
use ygo_policies_edopro::protocol::{ctos, read_packet, stoc, write_packet};
use ygo_policies_ocgcore::message::parse;
use ygo_policies_ocgcore::wire::msg;

fn paths() -> [PathBuf; 3] {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../native/ygo-policies");
    [root.join("vendor/BabelCdb/cards.cdb"), root.join("vendor/CardScripts"), root.join("decks")]
}

fn environment() -> Environment {
    let [cards, scripts, decks] = paths();
    Environment::open(&cards, &scripts, &decks).unwrap()
}

/// What a test's caller answers: its pilot's answer mostly, and now and then
/// another, told from what it is shown alone.
fn rule(view: &View) -> usize {
    let mut hash = 0xcbf29ce484222325u64;
    for part in [
        view.turn as u64,
        view.phase as u64,
        view.step as u64,
        view.cards.len() as u64,
        view.history.len() as u64,
        view.choices.len() as u64,
        view.life_points[0] as u64,
        view.life_points[1] as u64,
    ] {
        hash = (hash ^ part).wrapping_mul(0x100000001b3);
    }
    match view.label {
        Some(label) if !hash.is_multiple_of(4) => label,
        _ => (hash / 4) as usize % view.choices.len(),
    }
}

/// A decision as the caller is shown it: the arrays the policy reads, and
/// where the seat's trail puts it, which the arrays do not all hold (the
/// transaction it belongs to, the step of its selection, and whether a card
/// is acting).
#[derive(Debug, PartialEq)]
struct Shown {
    arrays: Vec<(String, std::sync::Arc<Array>)>,
    trail: (u32, usize, bool),
    /// The kind of the question.
    kind: u8,
}

fn shown(environment: &Environment, view: &View) -> Shown {
    let packed: Packed = environment.pack(view).unwrap();
    Shown {
        arrays: packed.arrays,
        trail: (view.transaction, view.step, view.acting.is_some()),
        kind: view.question.kind,
    }
}

/// The kinds of question that are commands: they open a transaction
/// whatever came before.
fn commands() -> Vec<u8> {
    let vocabulary = ygo_env::view::vocabulary();
    vocabulary["decision"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|pair| ["Idle", "Battle", "Chain"].contains(&pair[0].as_str().unwrap()))
        .map(|pair| pair[1].as_u64().unwrap() as u8)
        .collect()
}

/// The duel of `seed` between `decks` as an episode, `caller` deciding by
/// [`rule`]: each decision it took as it was shown it, the record, and who
/// won.
fn episode(
    environment: &Environment,
    seed: u64,
    decks: [&str; 2],
    caller: usize,
) -> (Vec<Shown>, serde_json::Value, Option<u8>) {
    let seat = |control| SeatSetup { control, policy: None, model: None, search: None, alone: false, observe: false };
    let mut seats = [seat(Control::Pilot), seat(Control::Pilot)];
    seats[caller] = seat(Control::Caller);
    let setup = Setup {
        seed,
        decks: decks.map(str::to_owned),
        lists: [None, None],
        seats,
        limit: 4096,
        life_points: None,
        record: None,
        keep_record: true,
        options: Options::default(),
        whole: false,
        whole_deck: 0,
        follow: false,
    };
    let mut episode = environment.start(&setup).unwrap();
    let mut seen = Vec::new();
    loop {
        let event = episode.advance().unwrap();
        if let Some(view) = event.decision {
            seen.push(shown(environment, &view));
            episode.choose(rule(&view)).unwrap();
        }
        if let Some(over) = event.over {
            assert!(!over.limit, "the duel ran into its limit");
            return (seen, over.record.unwrap(), over.winner);
        }
    }
}

fn expect(stream: &mut TcpStream, opcode: u8) -> Option<Vec<u8>> {
    let (got, payload) = read_packet(stream).ok()?;
    assert_eq!(got, opcode, "unexpected client packet {got:#x}");
    Some(payload)
}

/// A room of EDOPro that serves the recorded duel to the client seated as
/// `ours`.  It comes back with the answers the client gave that are the
/// recorded ones, and whether the duel was served to its end.
fn room(listener: TcpListener, record: serde_json::Value, ours: u8) -> (usize, bool) {
    let [cards, scripts, _] = paths();
    let core = Core::open(None, &cards, &scripts).unwrap();
    let recorded = Recorded::from_json(&record).unwrap();
    let mut duel = recorded.duel(&core).unwrap();
    let mut answers = [VecDeque::new(), VecDeque::new()];
    for (player, answer) in recorded.responses {
        answers[player as usize].push_back(answer);
    }
    let (mut stream, _) = listener.accept().unwrap();
    stream.set_nodelay(true).unwrap();
    let game = |stream: &mut TcpStream, message: &[u8]| write_packet(stream, stoc::GAME_MSG, message).is_ok();

    // The lobby, as EDOPro goes through it.
    expect(&mut stream, ctos::PLAYER_INFO).unwrap();
    expect(&mut stream, ctos::JOIN_GAME).unwrap();
    write_packet(&mut stream, stoc::JOIN_GAME, &[0; 20]).unwrap();
    expect(&mut stream, ctos::UPDATE_DECK).unwrap();
    write_packet(&mut stream, stoc::TYPE_CHANGE, &[0x01]).unwrap();
    expect(&mut stream, ctos::HS_READY).unwrap();
    write_packet(&mut stream, stoc::DUEL_START, &[]).unwrap();
    write_packet(&mut stream, stoc::SELECT_HAND, &[]).unwrap();
    expect(&mut stream, ctos::HAND_RESULT).unwrap();
    write_packet(&mut stream, stoc::SELECT_TP, &[]).unwrap();
    expect(&mut stream, ctos::TP_RESULT).unwrap();

    let mut given = 0;
    game(&mut stream, &duel.start_message(ours));
    loop {
        let step = duel.step().unwrap();
        let mut asked = None;
        for sent in &step.messages {
            match parse(&sent.bytes).ok().and_then(|message| message.responder()) {
                // A client is sent its own prompts, and is told to wait
                // while the other player answers one.
                Some(player) if player != ours => {
                    game(&mut stream, &[msg::WAITING]);
                    asked = Some(player);
                }
                responder => {
                    game(&mut stream, &sent.bytes);
                    asked = responder.or(asked);
                }
            }
        }
        match (step.state, asked) {
            (State::Over(_), _) => return (given, true),
            (_, Some(player)) if player == ours => {
                let Some(response) = expect(&mut stream, ctos::RESPONSE) else { return (given, false) };
                let recorded =
                    answers[player as usize].pop_front().expect("the record has an answer for every question");
                assert_eq!(response, recorded, "answer {given} of the seat is not the episode's");
                given += 1;
                duel.respond(&response);
            }
            (_, Some(player)) => {
                let recorded =
                    answers[player as usize].pop_front().expect("the record has an answer for every question");
                duel.respond(&recorded);
            }
            (_, None) => {}
        }
    }
}

/// The duel served to the caller's seat by a scripted room: each decision
/// the caller took as it was shown it, and how the client says it ended.
fn at_the_table(
    environment: &Environment,
    seed: u64,
    decks: [&str; 2],
    caller: usize,
    record: serde_json::Value,
    stop_after: Option<usize>,
) -> (Vec<Shown>, ygo_env::Result<Vec<ygo_env::edopro::Played>>, (usize, bool)) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let host = std::thread::spawn(move || room(listener, record, caller as u8));
    // An episode seeds the pilot beside seat `p` with `seed + p`, and a
    // table's seat takes the seed after the one it is given.
    let settings =
        Room { port: address.port(), seed: Some((seed + caller as u64).wrapping_sub(1)), wait: 5, ..Room::default() };
    let mut seen = Vec::new();
    let mut notes = Vec::new();
    let played = environment.edopro(decks[caller], &settings, &mut |event| match event {
        Event::Decision(_) if stop_after == Some(seen.len()) => Err("enough".to_owned()),
        Event::Decision(view) => {
            seen.push(shown(environment, view));
            Ok(rule(view))
        }
        Event::Note(text) => {
            notes.push(text.to_owned());
            Ok(0)
        }
        Event::Still => Ok(0),
    });
    let served = host.join().unwrap();
    assert!(notes.iter().any(|note| note.starts_with("duel 1 of 1")), "{notes:?}");
    (seen, played, served)
}

fn same(first: &[Shown], second: &[Shown]) {
    assert_eq!(first.len(), second.len(), "as many decisions");
    for (index, (a, b)) in first.iter().zip(second).enumerate() {
        assert_eq!(a.trail, b.trail, "decision {index} stands elsewhere on the seat's trail");
        assert_eq!(a.arrays, b.arrays, "decision {index} is not shown as the episode shows it");
    }
}

#[test]
fn the_seat_at_a_table_is_the_caller_of_an_episode() {
    let environment = environment();
    // Either seat, and decks that ask for targets, costs and zones.
    let commands = commands();
    assert_eq!(commands.len(), 3);
    let mut after_the_other = 0;
    for (seed, decks, caller) in [
        (52_001, ["blackwing", "monarch"], 0),
        (52_002, ["heroes", "blackwing"], 1),
        (52_003, ["blackwing", "gladiator"], 0),
        (52_004, ["lightsworn", "blackwing"], 1),
        (52_005, ["blackwing", "spellcaster"], 0),
        (52_006, ["burn", "blackwing"], 1),
        (52_007, ["blackwing", "blackwing"], 0),
        (52_008, ["dragunity", "blackwing"], 1),
    ] {
        let (shown, record, winner) = episode(&environment, seed, decks, caller);
        assert!(shown.len() > 20, "a duel of {} decisions tells little", shown.len());
        // Selections that are no command and open a transaction all the
        // same: the other seat answered something since the last, which is
        // known at a table from `MSG_WAITING` alone.
        after_the_other += shown
            .windows(2)
            .filter(|pair| !commands.contains(&pair[1].kind) && pair[1].trail.0 > pair[0].trail.0)
            .count();
        let (at_table, played, (given, to_the_end)) = at_the_table(&environment, seed, decks, caller, record, None);
        assert!(to_the_end, "the room served the duel of seed {seed} to its end");
        assert!(given > shown.len(), "placements and single answers are answered beside the caller's");
        same(&shown, &at_table);
        let played = played.unwrap();
        assert_eq!(played.len(), 1);
        let result = match winner {
            Some(2) => "draw",
            Some(seat) if seat as usize == caller => "won",
            _ => "lost",
        };
        assert_eq!((played[0].result.as_str(), played[0].seat), (result, Some(caller as u8)));
        assert_eq!(played[0].decisions, shown.len());
    }
    assert!(after_the_other >= 10, "{after_the_other} selections follow an answer of the other seat");
}

#[test]
fn a_caller_that_stops_ends_the_session() {
    let environment = environment();
    let (seed, decks, caller) = (52_001, ["blackwing", "monarch"], 0);
    let (shown, record, _) = episode(&environment, seed, decks, caller);
    let (at_table, played, (_, to_the_end)) = at_the_table(&environment, seed, decks, caller, record, Some(5));
    assert_eq!(played, Err("enough".to_owned()));
    // What it was shown until then is what the episode showed it.
    same(&shown[..5], &at_table);
    assert!(!to_the_end, "the room was left before the duel ended");
}

/// A record as EDOPro would keep the duel: each list in the order the engine
/// was given its cards, and the answers without who gave them.
fn as_kept(record: &serde_json::Value) -> Kept {
    let turned = |cards: &serde_json::Value| -> Vec<u64> {
        cards.as_array().unwrap().iter().rev().map(|code| code.as_u64().unwrap()).collect()
    };
    let decks: Vec<_> = record["decks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|deck| serde_json::json!({"main": turned(&deck["main"]), "extra": turned(&deck["extra"])}))
        .collect();
    let answers: Vec<_> = record["responses"].as_array().unwrap().iter().map(|answer| answer[1].clone()).collect();
    serde_json::from_value(serde_json::json!({
        "seed": record["seed"],
        "flags": record["flags"],
        "start_lp": record["start_lp"],
        "start_hand": record["start_hand"],
        "draw_count": record["draw_count"],
        "decks": decks,
        "responses": answers,
    }))
    .unwrap()
}

#[test]
fn a_kept_duel_played_again_tells_who_gave_each_answer() {
    let environment = environment();
    let (_, record, winner) = episode(&environment, 52_002, ["heroes", "blackwing"], 1);
    let kept = as_kept(&record);

    let reread = environment.kept(&kept).unwrap();
    assert_eq!(reread.record, record);
    assert_eq!(reread.winner, winner);
    assert!(reread.turns > 2);

    // A duel a player gave up stops short: what was answered is told.
    let mut short = kept.clone();
    short.responses.truncate(40);
    let reread = environment.kept(&short).unwrap();
    assert_eq!(reread.winner, None);
    assert_eq!(reread.record["responses"].as_array().unwrap()[..], record["responses"].as_array().unwrap()[..40]);

    // Another duel than the one its answers were given in is said to be.
    let mut other = kept.clone();
    other.decks.swap(0, 1);
    assert!(environment.kept(&other).unwrap_err().contains("not the same duel"));
    let mut longer = kept.clone();
    longer.responses.push("ffffffff".to_owned());
    assert!(environment.kept(&longer).unwrap_err().contains("are left when the duel ends"));
}
