//! The arrays of real decisions.
//!
//! What the arrays say of a decision is held by `tests/test_packed.py` of
//! the trainer, and what they are, digest for digest, by the vectors of
//! `native/ygo-model-input`.  Here: that every decision of real duels
//! packs, into batches of the declared shapes, and how long that takes
//! (`--nocapture` shows it).

use std::path::PathBuf;
use std::time::Instant;

use ygo_env::pack::{Data, MAX_ACTION_MEMBERS, MAX_CARDS, MAX_HISTORY};
use ygo_env::{Control, Environment, Options, SeatSetup, Setup, View};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../native/ygo-policies")
}

fn environment() -> Environment {
    let root = root();
    Environment::open(&root.join("vendor/BabelCdb/cards.cdb"), &root.join("vendor/CardScripts"), &root.join("decks"))
        .unwrap()
}

/// The decisions of both seats of pilot-played duels.
fn decisions(environment: &Environment, games: u64) -> Vec<View> {
    let ids: Vec<String> = environment.catalog().iter().map(|entry| entry.id.clone()).collect();
    let mut found = Vec::new();
    for game in 0..games {
        let decks = [ids[game as usize % ids.len()].clone(), ids[(game as usize * 5 + 3) % ids.len()].clone()];
        let seat = || SeatSetup { control: Control::Pilot, policy: None, model: None, search: None, alone: false, observe: true };
        let setup = Setup {
            seed: 9_000 + game,
            decks: decks.clone(),
            lists: [None, None],
            seats: [seat(), seat()],
            limit: 4096,
            life_points: None,
            record: None,
            keep_record: false,
            options: Options::default(),
            whole: false,
            whole_deck: 0,
            follow: false,
        };
        let mut episode = environment.start(&setup).unwrap();
        loop {
            let event = episode.advance().unwrap();
            found.extend(event.samples);
            if event.over.is_some() {
                break;
            }
        }
    }
    found
}

#[test]
fn real_decisions_pack_into_batches_of_the_declared_shapes() {
    let environment = environment();
    let decisions = decisions(&environment, 40);
    assert!(decisions.len() > 2_000);

    let started = Instant::now();
    let packed: Vec<_> =
        decisions.iter().map(|shown| environment.pack(shown).unwrap()).collect();
    let packing = started.elapsed();

    let started = Instant::now();
    let mut batches = 0;
    for chunk in packed.chunks(256) {
        let batch = environment.batch(&chunk.iter().collect::<Vec<_>>()).unwrap();
        batches += 1;
        let shape = |name: &str| -> Vec<usize> {
            batch.arrays.iter().find(|(key, _)| key == name).unwrap_or_else(|| panic!("no {name}")).1.shape.clone()
        };
        assert_eq!(batch.arrays.len(), 45);
        for (name, array) in &batch.arrays {
            assert_eq!(array.shape[0], chunk.len(), "{name}");
            assert_eq!(array.shape.iter().product::<usize>(), array.data.len(), "{name}");
        }
        let (cards, actions) = (shape("state_card")[1], shape("action_kind")[1]);
        let members = shape("action_member_card")[2];
        assert!((1..=MAX_CARDS).contains(&cards));
        assert!((1..=MAX_ACTION_MEMBERS).contains(&members));
        assert!((1..=MAX_HISTORY).contains(&shape("history_card")[1]));
        assert_eq!(shape("state_status"), [chunk.len(), cards, 13]);
        assert_eq!(shape("context_numeric"), [chunk.len(), 7, 17]);
        assert_eq!(shape("action_member_categories"), [chunk.len(), actions, members, 8]);
        assert_eq!(shape("action_mask"), [chunk.len(), actions]);
        assert_eq!(shape("scalars"), [chunk.len(), 24]);
        // Every decision offers what its mask says, and at least one answer.
        let (_, mask) = batch.arrays.iter().find(|(key, _)| key == "action_mask").unwrap();
        let Data::Bool(mask) = &mask.data else { panic!("the mask is of booleans") };
        for (row, shown) in mask.chunks(actions).zip(&decisions[(batches - 1) * 256..]) {
            assert_eq!(row.iter().map(|on| *on as usize).sum::<usize>(), shown.choices.len());
        }
    }
    let batching = started.elapsed();
    println!(
        "{} decisions: packing {:.1} us a decision, batches of 256 {:.1} us a decision",
        decisions.len(),
        packing.as_secs_f64() * 1e6 / decisions.len() as f64,
        batching.as_secs_f64() * 1e6 / decisions.len() as f64,
    );
}
