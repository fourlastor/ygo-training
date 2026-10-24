//! Many duels taken forward together are the duels played one at a time.

use std::collections::BTreeMap;
use std::path::PathBuf;

use ygo_env::pack::{Array, Batch, Data};
use ygo_env::pool::Round;
use ygo_env::{Control, Environment, Options, Pool, SeatSetup, Setup, View};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../native/ygo-policies")
}

fn paths() -> [PathBuf; 3] {
    let root = root();
    [root.join("vendor/BabelCdb/cards.cdb"), root.join("vendor/CardScripts"), root.join("decks")]
}

fn pool(threads: usize, slots: usize) -> Pool {
    let [cards, scripts, decks] = paths();
    Pool::open(&cards, &scripts, &decks, threads, slots, false).unwrap()
}

/// A pool whose threads draw their duels from one queue.
fn sharing(threads: usize, slots: usize) -> Pool {
    let [cards, scripts, decks] = paths();
    Pool::open(&cards, &scripts, &decks, threads, slots, true).unwrap()
}

/// Duels where the seat `duel % 2` decides for itself, its pilot asked
/// alongside, against the pilot of another deck.
fn setups(count: u64) -> Vec<Setup> {
    let ids = ["blackwing", "monarch", "heroes", "gladiator", "burn", "lightsworn"];
    (0..count)
        .map(|duel| {
            let seat = |control| SeatSetup { control, policy: None, model: None, search: None, alone: false, observe: false };
            let caller = (duel % 2) as usize;
            let mut seats = [seat(Control::Pilot), seat(Control::Pilot)];
            seats[caller] = seat(Control::Caller);
            Setup {
                seed: 31_000 + duel,
                decks: [ids[duel as usize % ids.len()].to_owned(), ids[(duel as usize * 5 + 1) % ids.len()].to_owned()],
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
        })
        .collect()
}

fn numbers(arrays: &[(String, Array)], name: &str) -> Vec<i64> {
    let (_, array) = arrays.iter().find(|(key, _)| key == name).unwrap_or_else(|| panic!("no {name}"));
    match &array.data {
        Data::I64(values) => values.clone(),
        _ => panic!("{name} is not of whole numbers"),
    }
}

fn whole(batch: &Batch, name: &str) -> Vec<i64> {
    numbers(&batch.arrays, name)
}

/// The arrays of a batch that are the model's input.
fn input(batch: &Batch) -> Vec<&(String, Array)> {
    batch.arrays.iter().filter(|(name, _)| !name.starts_with("meta_")).collect()
}

/// How many answers each decision of a batch offers.
fn offered(batch: &Batch) -> Vec<usize> {
    let (_, mask) = batch.arrays.iter().find(|(key, _)| key == "action_mask").unwrap();
    let Data::Bool(values) = &mask.data else { panic!("the mask is of booleans") };
    values.chunks(mask.shape[1]).map(|row| row.iter().map(|on| *on as usize).sum()).collect()
}

/// An answer that depends on the decision alone: the pilot's on every third
/// decision of a duel, else one of the others.
fn answer(place: i64, label: i64, choices: usize) -> u32 {
    if place % 3 == 0 && label >= 0 {
        label as u32
    } else {
        (place as usize * 7 + 1) as u32 % choices as u32
    }
}

/// How each duel went: its digest, winner and turns, and the answers given.
type Played = BTreeMap<usize, (String, Option<u8>, u32, Vec<u32>)>;

fn through_a_pool(threads: usize, slots: usize, count: u64) -> (Played, usize, usize) {
    let (ended, rounds) = through(pool(threads, slots), threads * slots, count);
    let largest = rounds.iter().map(Vec::len).max().unwrap_or(0);
    (ended, rounds.len() + 1, largest)
}

/// The duels through `pool`, and the duels each of its rounds asked of.
fn through(mut pool: Pool, capacity: usize, count: u64) -> (Played, Vec<Vec<i64>>) {
    assert_eq!(pool.submit(setups(count), false, false), 0);
    let mut played: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    let mut ended = Played::new();
    let (mut answers, mut rounds) = (Vec::new(), Vec::new());
    loop {
        let round = pool.step(&answers).unwrap();
        for finished in round.finished {
            let given = played.remove(&finished.duel).unwrap_or_default();
            ended.insert(finished.duel, (finished.over.digest, finished.over.winner, finished.over.turns, given));
        }
        let Some(batch) = &round.batch else {
            assert_eq!(round.in_flight, 0);
            break;
        };
        // The decisions come in the order of their duels, no more than fit.
        let duels = whole(batch, "meta_duel");
        assert_eq!(duels, round.duels.iter().map(|duel| *duel as i64).collect::<Vec<_>>());
        assert!(duels.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(duels.len() <= capacity);
        let (labels, places, choices) = (whole(batch, "meta_label"), whole(batch, "meta_answer"), offered(batch));
        answers = (0..duels.len()).map(|row| answer(places[row], labels[row], choices[row])).collect();
        for (duel, given) in duels.iter().zip(&answers) {
            played.entry(*duel as usize).or_default().push(*given);
        }
        rounds.push(duels);
    }
    assert!(played.is_empty());
    (ended, rounds)
}

fn one_at_a_time(count: u64) -> Played {
    let [cards, scripts, decks] = paths();
    let environment = Environment::open(&cards, &scripts, &decks).unwrap();
    let mut ended = Played::new();
    for (duel, setup) in setups(count).iter().enumerate() {
        let mut episode = environment.start(setup).unwrap();
        let mut given = Vec::new();
        let over = loop {
            let event = episode.advance().unwrap();
            if let Some(over) = event.over {
                break over;
            }
            let shown = event.decision.unwrap();
            let label = shown.label.map_or(-1, |label| label as i64);
            let chosen = answer(shown.answer as i64, label, shown.choices.len());
            episode.choose(chosen as usize).unwrap();
            given.push(chosen);
        };
        ended.insert(duel, (over.digest, over.winner, over.turns, given));
    }
    ended
}

#[test]
fn a_pool_plays_the_duels_played_one_at_a_time() {
    let alone = one_at_a_time(24);
    assert!(alone.values().map(|(_, _, _, given)| given.len()).sum::<usize>() > 400);
    // One duel at a time on one thread, and many on several: the same duels,
    // answer for answer.
    let (single, rounds, largest) = through_a_pool(1, 1, 24);
    assert_eq!(single, alone);
    assert_eq!(largest, 1);
    let (many, fewer, widest) = through_a_pool(4, 3, 24);
    assert_eq!(many, alone);
    assert!(widest > 6 && fewer < rounds / 4, "{widest} decisions at most a round, {fewer} rounds for {rounds}");
}

#[test]
fn threads_that_draw_from_one_queue_play_the_same_duels() {
    let alone = one_at_a_time(30);
    // Three threads of two duels and thirty duels: every slot is taken over
    // and over, by whichever thread has it free first.
    let (shared, rounds) = through(sharing(3, 2), 6, 30);
    assert_eq!(shared, alone);
    // A round asks of the same duels every time, whichever thread has them:
    // as many start as slots come free, in the order they were submitted.
    for _ in 0..3 {
        assert_eq!(through(sharing(3, 2), 6, 30), (alone.clone(), rounds.clone()));
    }
    // Not the rounds of duels dealt in advance, and no more of them: no slot
    // is left empty while a duel waits to start.
    let (dealt, their_rounds) = through(pool(3, 2), 6, 30);
    assert_eq!(dealt, alone);
    assert_ne!(their_rounds, rounds);
    assert!(rounds.len() <= their_rounds.len(), "{} rounds against {}", rounds.len(), their_rounds.len());
    let full = |rounds: &[Vec<i64>]| rounds.iter().filter(|round| round.len() == 6).count();
    assert!(full(&rounds) >= full(&their_rounds));

    // Duels the pilots play end in one round either way, and what is kept of
    // them is the same, row for row.
    let kept = |mut pool: Pool| {
        assert_eq!(pool.submit(observed(11), true, false), 0);
        let round = pool.step(&[]).unwrap();
        assert!(round.batch.is_none() && round.finished.len() == 11);
        let rows: Vec<u64> = (0..pool.kept() as u64).collect();
        (round.samples, pool.batch(&rows).unwrap().arrays)
    };
    let (samples, arrays) = kept(sharing(3, 2));
    assert!(numbers(&samples, "sample_row").len() > 200);
    assert!(kept(pool(3, 2)) == (samples, arrays));
}

#[test]
fn duels_can_be_added_while_others_are_in_flight() {
    let mut pool = pool(2, 2);
    assert_eq!(pool.submit(setups(3), false, false), 0);
    let first = pool.step(&[]).unwrap();
    assert_eq!((first.duels.clone(), first.in_flight), (vec![0, 1, 2], 3));
    assert_eq!(pool.submit(setups(2), false, false), 3);
    assert_eq!(pool.in_flight(), 5);
    let batch = first.batch.unwrap();
    let next = pool.step(&offered(&batch).iter().map(|choices| *choices as u32 - 1).collect::<Vec<_>>()).unwrap();
    // Two threads of two duels: the fourth duel has a slot, the fifth waits.
    assert!(next.duels.contains(&3) && !next.duels.contains(&4));
}

#[test]
fn what_is_not_an_answer_to_the_round_is_refused() {
    let mut pool = pool(2, 2);
    pool.submit(setups(2), false, false);
    let round = pool.step(&[]).unwrap();
    assert_eq!(round.duels, [0, 1]);
    assert!(pool.step(&[0]).unwrap_err().contains("1 answers for the 2 decisions"));
    // An answer the decision does not offer stops the pool for good: its
    // duels are no longer in step.
    assert!(pool.step(&[0, 4000]).unwrap_err().contains("duel 1: choice 4000"));
    assert!(pool.step(&[0, 0]).unwrap_err().contains("an earlier round"));
    assert!(Pool::open(&paths()[0], &paths()[1], &paths()[2], 0, 1, false).is_err());
}

/// Duels the pilots play, the seat `duel % 2` observed.
fn observed(count: u64) -> Vec<Setup> {
    let mut setups = setups(count);
    for (duel, setup) in setups.iter_mut().enumerate() {
        setup.seats[duel % 2] = SeatSetup { control: Control::Pilot, policy: None, model: None, search: None, alone: false, observe: true };
    }
    setups
}

#[test]
fn the_decisions_of_kept_duels_are_given_back_as_batches() {
    let [cards, scripts, decks] = paths();
    let environment = Environment::open(&cards, &scripts, &decks).unwrap();

    // Duels the pilots play: one round takes them all to their end, and the
    // observed seat's decisions stay behind, in the order of the duels and
    // of play.
    let mut pool = pool(3, 2);
    assert_eq!(pool.submit(observed(7), true, false), 0);
    let round = pool.step(&[]).unwrap();
    assert!(round.batch.is_none() && round.finished.len() == 7 && round.in_flight == 0);
    let rows = numbers(&round.samples, "sample_row");
    assert!(rows.len() > 150 && pool.kept() == rows.len());
    assert_eq!(rows, (0..rows.len() as i64).collect::<Vec<_>>());
    let mut expected = Vec::new();
    let mut views = Vec::new();
    for (duel, setup) in observed(7).iter().enumerate() {
        let mut episode = environment.start(setup).unwrap();
        let event = episode.advance().unwrap();
        assert!(event.over.is_some());
        for sample in event.samples {
            let (me, label) = (sample.player as usize, sample.label.unwrap() as i64);
            assert_eq!((me, sample.chosen, sample.pilot), (duel % 2, sample.label, sample.label));
            expected.push((duel as i64, label, me as i64, i64::from(sample.transaction), sample.answer as i64));
            views.push((environment.pack(&sample).unwrap(), duel));
        }
    }
    let of = |name: &str| numbers(&round.samples, name);
    let (duels, labels, players) = (of("sample_duel"), of("sample_label"), of("sample_player"));
    let (transactions, answers) = (of("sample_transaction"), of("sample_answer"));
    let told: Vec<_> =
        (0..rows.len()).map(|row| (duels[row], labels[row], players[row], transactions[row], answers[row])).collect();
    assert_eq!(told, expected);
    assert_eq!((of("sample_pilot"), of("sample_chosen")), (labels.clone(), labels));

    // Any of them, in any order, as the batch built from the views themselves.
    let picked: Vec<u64> = (0..rows.len() as u64).rev().step_by(7).collect();
    let alone: Vec<_> = picked.iter().map(|row| &views[*row as usize].0).collect();
    assert_eq!(pool.batch(&picked).unwrap().arrays, environment.batch(&alone).unwrap().arrays);
    assert!(pool.batch(&[rows.len() as u64]).unwrap_err().contains("no decision is kept under row"));

    // What the caller answers is kept too, under the rows its rounds name:
    // a round of kept decisions is the batch of its rows.
    let follow = |batch: &Batch| -> Vec<u32> {
        let pick = |(label, choices): (&i64, usize)| if *label >= 0 { *label as u32 } else { choices as u32 - 1 };
        whole(batch, "meta_label").iter().zip(offered(batch)).map(pick).collect()
    };
    let before = pool.kept();
    let first = pool.submit(setups(4), true, false);
    let (mut answers, mut asked) = (Vec::new(), Vec::new());
    loop {
        let round = pool.step(&answers).unwrap();
        let Some(batch) = &round.batch else { break };
        assert!(round.samples.is_empty());
        let rows: Vec<u64> = whole(batch, "meta_row").iter().map(|row| *row as u64).collect();
        assert_eq!(pool.batch(&rows).unwrap().arrays.iter().collect::<Vec<_>>(), input(batch));
        asked.extend(rows);
        answers = follow(batch);
    }
    assert!(asked.len() > 150, "{} decisions", asked.len());
    assert_eq!(asked.iter().copied().collect::<std::collections::BTreeSet<_>>().len(), asked.len());
    assert_eq!(pool.kept(), before + asked.len());

    // Duels that are not kept leave nothing, beside others that are.
    pool.submit(setups(2), false, false);
    pool.submit(setups(2), true, false);
    let (mut answers, mut more) = (Vec::new(), 0);
    loop {
        let round = pool.step(&answers).unwrap();
        let Some(batch) = &round.batch else { break };
        for (duel, row) in whole(batch, "meta_duel").iter().zip(whole(batch, "meta_row")) {
            assert_eq!(row >= 0, (*duel as usize) >= first + 6, "duel {duel} under row {row}");
            more += usize::from(row >= 0);
        }
        answers = follow(batch);
    }
    assert!(more > 50 && pool.kept() == before + asked.len() + more);

    // Forgotten, the rows are not given again and no longer answer.
    assert_eq!(pool.submit(setups(1), true, false), first + 8);
    let waiting = pool.step(&[]).unwrap();
    assert!(pool.forget().unwrap_err().contains("1 duels are in flight"));
    let mut round = waiting;
    while let Some(batch) = &round.batch {
        let answers: Vec<u32> = offered(batch).iter().map(|choices| *choices as u32 - 1).collect();
        round = pool.step(&answers).unwrap();
    }
    let held = pool.kept();
    pool.forget().unwrap();
    assert_eq!(pool.kept(), 0);
    assert!(pool.batch(&[0]).unwrap_err().contains("no decision is kept under row 0"));
    pool.submit(observed(1), true, false);
    let next = pool.step(&[]).unwrap();
    assert_eq!(numbers(&next.samples, "sample_row")[0], held as i64);
}

#[test]
fn chosen_rows_are_let_go_of_while_duels_are_in_flight() {
    // Kept duels taken a few rounds forward, every decision answered with
    // its last choice: the rows of their decisions so far.
    let mut pool = pool(2, 2);
    pool.submit(setups(4), true, false);
    let (mut rows, mut answers) = (Vec::new(), Vec::new());
    for _ in 0..6 {
        let round = pool.step(&answers).unwrap();
        let batch = round.batch.as_ref().expect("the duels are in flight");
        rows.extend(whole(batch, "meta_row").iter().map(|row| *row as u64));
        answers = offered(batch).iter().map(|choices| *choices as u32 - 1).collect();
    }
    assert!(pool.in_flight() > 0 && rows.len() >= 8, "{} rows", rows.len());
    let kept = pool.kept();

    // Every other one let go of: the others still answer, those do not.
    let gone: Vec<u64> = rows.iter().copied().step_by(2).collect();
    let left: Vec<u64> = rows.iter().copied().skip(1).step_by(2).collect();
    let before = pool.batch(&left).unwrap().arrays;
    pool.forget_rows(&gone).unwrap();
    assert_eq!(pool.kept(), kept - gone.len());
    assert_eq!(pool.batch(&left).unwrap().arrays, before);
    assert!(pool.batch(&gone[1..2]).unwrap_err().contains("no decision is kept under row"));

    // A row let go of already, or named twice, is refused, and nothing is
    // let go of then.
    assert!(pool.forget_rows(&[left[0], gone[0]]).unwrap_err().contains("no decision is kept"));
    assert!(pool.forget_rows(&[left[0], left[0]]).unwrap_err().contains("named twice"));
    assert_eq!(pool.batch(&left).unwrap().arrays, before);

    // The duels go on, their decisions under rows that follow the last.
    let round = pool.step(&answers).unwrap();
    let batch = round.batch.as_ref().expect("the duels are in flight");
    let next: Vec<u64> = whole(batch, "meta_row").iter().map(|row| *row as u64).collect();
    let last = *rows.iter().max().unwrap();
    assert!(next.iter().all(|row| *row > last));

    // All of them let go of, none is kept; the next rows follow the last.
    pool.forget_rows(&left).unwrap();
    pool.forget_rows(&next).unwrap();
    assert_eq!(pool.kept(), 0);
    let answers: Vec<u32> = offered(batch).iter().map(|choices| *choices as u32 - 1).collect();
    let round = pool.step(&answers).unwrap();
    if let Some(batch) = &round.batch {
        assert!(whole(batch, "meta_row").iter().all(|row| *row as u64 > *next.iter().max().unwrap()));
    }
}

#[test]
fn a_duel_that_shows_its_decisions_gives_the_views_they_were_packed_from() {
    let [cards, scripts, decks] = paths();
    let environment = Environment::open(&cards, &scripts, &decks).unwrap();
    let json = |view: &View| serde_json::to_value(view).unwrap();

    // Of observed seats: each decision kept comes with the view an episode
    // of the same duel gives, where its duel shows them.
    let mut observing = pool(3, 2);
    observing.submit(observed(2), true, false);
    observing.submit(observed(3), true, true);
    let round = observing.step(&[]).unwrap();
    let mut expected = Vec::new();
    for (duel, setup) in observed(2).iter().chain(&observed(3)).enumerate() {
        let mut episode = environment.start(setup).unwrap();
        let event = episode.advance().unwrap();
        expected.extend(event.samples.iter().map(|sample| (duel >= 2).then(|| json(sample))));
    }
    assert!(expected.iter().filter(|view| view.is_some()).count() > 50);
    assert_eq!(numbers(&round.samples, "sample_duel").len(), expected.len());
    assert_eq!(round.sample_views, expected);
    assert!(round.views.is_empty());

    // Of the caller's: the views of the waiting decisions, in the order of
    // the batch, kept or not.
    let mut asking = pool(2, 2);
    asking.submit(setups(1), false, false);
    asking.submit(setups(2), false, true);
    let mut shown: BTreeMap<usize, Vec<serde_json::Value>> = BTreeMap::new();
    let mut answers = Vec::new();
    loop {
        let round = asking.step(&answers).unwrap();
        let Some(batch) = &round.batch else { break };
        assert_eq!(round.views.len(), round.duels.len());
        for (duel, view) in round.duels.iter().zip(&round.views) {
            assert_eq!(view.is_some(), *duel > 0, "duel {duel}");
            if let Some(view) = view {
                shown.entry(*duel).or_default().push(view.clone());
            }
        }
        let (labels, places, choices) = (whole(batch, "meta_label"), whole(batch, "meta_answer"), offered(batch));
        answers = (0..round.duels.len()).map(|row| answer(places[row], labels[row], choices[row])).collect();
    }
    for (duel, setup) in setups(2).iter().enumerate() {
        let mut episode = environment.start(setup).unwrap();
        let mut views = Vec::new();
        while let Some(view) = episode.advance().unwrap().decision {
            let label = view.label.map_or(-1, |label| label as i64);
            episode.choose(answer(view.answer as i64, label, view.choices.len()) as usize).unwrap();
            views.push(json(&view));
        }
        assert!(views.len() > 20);
        assert_eq!(shown[&(duel + 1)], views, "duel {}", duel + 1);
    }
}

/// A pool whose threads' duels are shared out among `groups`.
fn grouped(threads: usize, slots: usize, groups: usize) -> Pool {
    let [cards, scripts, decks] = paths();
    Pool::grouped(&cards, &scripts, &decks, threads, slots, false, groups).unwrap()
}

/// The duels through `pool` a group at a time, every group set on its way
/// before the first is waited for; and all of them together from the round
/// on at which the groups' last rounds hold `over` duels or fewer between
/// them.  How each duel went, and the duels each round asked of, with its
/// group (none for a round of all of them).
fn by_groups(mut pool: Pool, groups: usize, count: u64, over: usize) -> (Played, Vec<(Option<usize>, Vec<i64>)>) {
    assert_eq!(pool.submit(setups(count), false, false), 0);
    let mut played: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    let mut ended = Played::new();
    let mut rounds = Vec::new();
    // What a round leaves: the duels that ended in it are noted, and its
    // decisions answered; none where no duel waits.
    let mut taken = |round: Round, group: Option<usize>| -> Option<Vec<u32>> {
        for finished in round.finished {
            let given = played.remove(&finished.duel).unwrap_or_default();
            ended.insert(finished.duel, (finished.over.digest, finished.over.winner, finished.over.turns, given));
        }
        let batch = round.batch.as_ref()?;
        let duels = whole(batch, "meta_duel");
        assert!(duels.windows(2).all(|pair| pair[0] < pair[1]));
        let (labels, places, choices) = (whole(batch, "meta_label"), whole(batch, "meta_answer"), offered(batch));
        let answers: Vec<u32> = (0..duels.len()).map(|row| answer(places[row], labels[row], choices[row])).collect();
        for (duel, given) in duels.iter().zip(&answers) {
            played.entry(*duel as usize).or_default().push(*given);
        }
        rounds.push((group, duels));
        Some(answers)
    };
    let mut going: Vec<usize> = (0..groups).collect();
    for group in &going {
        pool.begin(*group, &[]).unwrap();
    }
    let mut sizes: BTreeMap<usize, usize> = BTreeMap::new();
    let mut together: Option<Vec<u32>> = None;
    let mut left = usize::MAX;
    'groups: while !going.is_empty() {
        for group in going.clone() {
            let round = pool.finish(group).unwrap();
            let in_flight = round.in_flight;
            let Some(answers) = taken(round, Some(group)) else {
                going.retain(|other| *other != group);
                sizes.remove(&group);
                left = in_flight;
                continue;
            };
            sizes.insert(group, answers.len());
            if sizes.len() == going.len() && sizes.values().sum::<usize>() <= over {
                // The other groups are waited for, and all go on together:
                // their answers group after group.
                let mut waiting = BTreeMap::from([(group, answers)]);
                for other in going.iter().copied().filter(|other| *other != group) {
                    let round = pool.finish(other).unwrap();
                    if let Some(answers) = taken(round, Some(other)) {
                        waiting.insert(other, answers);
                    }
                }
                together = Some(waiting.into_values().flatten().collect());
                break 'groups;
            }
            pool.begin(group, &answers).unwrap();
        }
    }
    if let Some(mut answers) = together {
        loop {
            let round = pool.step(&answers).unwrap();
            let in_flight = round.in_flight;
            match taken(round, None) {
                Some(next) => answers = next,
                None => {
                    left = in_flight;
                    break;
                }
            }
        }
    }
    assert_eq!(left, 0);
    assert!(played.is_empty());
    (ended, rounds)
}

#[test]
fn groups_taken_forward_one_at_a_time_play_the_same_duels() {
    let alone = one_at_a_time(30);
    // Three threads of four duels in two groups: a group's round holds two
    // duels of every thread at most, and the engines of one group run while
    // the other is answered.
    let (two, rounds) = by_groups(grouped(3, 4, 2), 2, 30, 0);
    assert_eq!(two, alone);
    assert!(rounds.iter().all(|(group, duels)| group.is_some() && duels.len() <= 6));
    assert!(rounds.iter().any(|(group, _)| *group == Some(0)) && rounds.iter().any(|(group, _)| *group == Some(1)));
    // A duel stays in the group whose slot it took.
    let mut of: BTreeMap<i64, Option<usize>> = BTreeMap::new();
    for (group, duels) in &rounds {
        for duel in duels {
            assert_eq!(*of.entry(*duel).or_insert(*group), *group, "duel {duel} changed its group");
        }
    }
    // The rounds are the same every time, whatever the threads' pace.
    for _ in 0..3 {
        assert_eq!(by_groups(grouped(3, 4, 2), 2, 30, 0), (alone.clone(), rounds.clone()));
    }
    // Four groups of one duel a thread, and the groups of a pool taken
    // forward together: the same duels again.
    assert_eq!(by_groups(grouped(3, 4, 4), 4, 30, 0).0, alone);
    assert_eq!(through(grouped(3, 4, 2), 12, 30).0, alone);
}

#[test]
fn groups_whose_rounds_have_grown_small_go_on_together() {
    let alone = one_at_a_time(30);
    // By groups while their rounds hold more than eight duels between them,
    // and together from there: the same duels, in fewer rounds than by
    // groups to the end.
    let by_groups_only = by_groups(grouped(3, 4, 2), 2, 30, 0).1;
    let (mixed, rounds) = by_groups(grouped(3, 4, 2), 2, 30, 8);
    assert_eq!(mixed, alone);
    let first = rounds.iter().position(|(group, _)| group.is_none()).expect("the groups never went on together");
    assert!(first > 2, "the groups went on together after {first} rounds");
    assert!(rounds[..first].iter().all(|(group, _)| group.is_some()));
    assert!(rounds[first..].iter().all(|(group, duels)| group.is_none() && duels.len() <= 8));
    // The rounds before it are the rounds the groups have to the end.
    assert_eq!(rounds[..first], by_groups_only[..first]);
    assert!(rounds.len() < by_groups_only.len());
    // The same rounds every time.
    assert_eq!(by_groups(grouped(3, 4, 2), 2, 30, 8), (alone.clone(), rounds));
    // Together as soon as every group has had a round, the first of them
    // waited for once more: with duels still to start, which take the free
    // slots of either group.
    let (early, rounds) = by_groups(grouped(3, 4, 2), 2, 30, 12);
    assert_eq!(early, alone);
    let groups: Vec<Option<usize>> = rounds.iter().map(|(group, _)| *group).take(4).collect();
    assert_eq!(groups, [Some(0), Some(1), Some(0), None]);
    assert!(rounds[3..].iter().all(|(group, _)| group.is_none()));
    assert!(rounds.iter().any(|(group, duels)| group.is_none() && duels.len() == 12));
}

#[test]
fn a_group_is_begun_once_and_finished_once() {
    let mut pool = grouped(2, 2, 2);
    pool.submit(setups(6), false, false);
    assert!(pool.finish(0).unwrap_err().contains("group 0 was not begun"));
    assert!(pool.begin(2, &[]).unwrap_err().contains("group 2 of a pool of 2"));
    pool.begin(0, &[]).unwrap();
    assert!(pool.begin(0, &[]).unwrap_err().contains("on its way already"));
    // The duels wait a group at a time from here on.
    let round = pool.finish(0).unwrap();
    assert_eq!(round.duels.len(), 2);
    assert!(pool.step(&[]).unwrap_err().contains("0 answers for the 2 decisions of the last round"));
    assert!(pool.begin(0, &[0]).unwrap_err().contains("1 answers for the 2 decisions of group 0"));
    // What was refused left the group as it was.
    pool.begin(1, &[]).unwrap();
    // All of them are taken on together only when none is on its way.
    assert!(pool.step(&[0, 0]).unwrap_err().contains("group 1 is on its way: finish waits for it"));
    pool.begin(0, &[0, 0]).unwrap();
    assert_eq!(pool.finish(1).unwrap().duels.len(), 2);
    assert!(pool.finish(0).is_ok());

    let [cards, scripts, decks] = paths();
    let refused = |slots, shared, groups| Pool::grouped(&cards, &scripts, &decks, 2, slots, shared, groups).err().unwrap();
    assert!(refused(3, false, 2).contains("not shared out evenly"));
    assert!(refused(4, false, 0).contains("not shared out evenly"));
    assert!(refused(4, true, 2).contains("do not go together"));

    // Duels that wait together are taken on together.
    let mut whole = grouped(2, 2, 2);
    whole.submit(setups(4), false, false);
    assert_eq!(whole.step(&[]).unwrap().duels.len(), 4);
    assert!(whole.begin(0, &[]).unwrap_err().contains("wait together"));
}
