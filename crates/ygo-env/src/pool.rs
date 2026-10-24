//! Many duels at once.
//!
//! A network answers a batch of decisions in little more time than one, and
//! a duel that waits for its answer keeps a processor idle.  A [`Pool`] holds
//! many duels and takes them forward in rounds: every duel in flight is
//! answered, each then runs to its next decision or to its end, and the
//! decisions come back together, as the arrays the model reads.
//!
//! * The duels are spread over threads.  A thread has its own
//!   [`Environment`] (the engine underneath is one a thread, as in
//!   `policy-bench`) and its duels stay on it.  Which thread a duel goes to
//!   is one of two things ([`Pool::open`]): the thread of its number, dealt
//!   in advance, or whichever thread has a slot free first, from one queue.
//! * A round is the same whatever the threads' pace: its decisions are in
//!   the order of their duels, and a duel's own course depends on nothing
//!   but its setup and its answers.
//! * A thread's duels can be shared out among groups ([`Pool::grouped`]),
//!   which are taken forward one at a time ([`Pool::begin`], then
//!   [`Pool::finish`]): while the model answers the decisions of one group,
//!   the engines of another run.  A group's round is as much the same every
//!   time as a pool's: which duels are in a group follows from the order
//!   the groups are begun in, not from anyone's pace.  Groups that wait can
//!   be taken on together from any round on ([`Pool::step`]): a call of the
//!   model costs much the same whatever it holds, so that small rounds are
//!   better answered in one.
//! * Training reads a decision again after its duel, more than once.  The
//!   decisions of duels submitted to be kept stay in the pool, each under a
//!   row, and any of them are given back as a batch ([`Pool::batch`]) until
//!   they are forgotten ([`Pool::forget`]): the ones the caller answered and
//!   the ones of observed seats, which a pilot or a record answered.  Those
//!   of chosen rows can be let go of while duels are in flight
//!   ([`Pool::forget_rows`]): duels kept going from one update to the next
//!   hold their decisions until each ends and is trained on.
//! * A duel can be submitted to show its decisions as well
//!   ([`Pool::submit`]): each comes with the view it was packed from, for
//!   what is read of a decision beside the model's input.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::episode::{Environment, Episode, Event, Over, Setup};
use crate::follow::{Label, COUNTED};
use crate::pack::{self, Array, Batch, Data, Packed};
use crate::view::View;
use crate::Result;

/// A decision of a duel: one that waits for its answer, or one of an
/// observed seat that was answered on the way.
struct Shown {
    duel: usize,
    packed: Packed,
    /// The answer to imitate, among the decision's choices.
    label: Option<usize>,
    /// The accompanying pilot's own, and the one that was played.
    pilot: Option<usize>,
    chosen: Option<usize>,
    player: u8,
    transaction: u32,
    /// The place of the decision among the duel's engine answers, and its
    /// step in the selection.
    answer: usize,
    step: usize,
    /// Whether the duel's decisions are kept, and its cards followed.
    keep: bool,
    follow: bool,
    /// The decision as the seat was shown it, where the duel shows them.
    view: Option<serde_json::Value>,
    /// The same decision with nothing hidden, where the duel's setup asks
    /// for it ([`Setup::whole`](crate::episode::Setup::whole)).
    whole: Option<Packed>,
}

/// A duel that ended in a round.
#[derive(Debug)]
pub struct Finished {
    pub duel: usize,
    pub over: Over,
}

/// What a round leaves: the decisions that wait, and the duels that ended.
#[derive(Debug)]
pub struct Round {
    /// The waiting decisions as one batch, in the order of their duels, with
    /// what tells them apart under `meta_` names; none when no duel waits.
    pub batch: Option<Batch>,
    /// The duels of the batch's rows.
    pub duels: Vec<usize>,
    /// The decisions of observed seats answered in the round, of the duels
    /// that are kept: one number a decision under each of `sample_` names,
    /// in the order of their duels and, within a duel, of play.
    pub samples: Vec<(String, Array)>,
    /// The views the waiting decisions were packed from, in the order of the
    /// batch, and those of the observed seats' decisions, in the order of the
    /// samples: of the duels that show them, and none for the others.
    pub views: Vec<Option<serde_json::Value>>,
    pub sample_views: Vec<Option<serde_json::Value>>,
    pub finished: Vec<Finished>,
    /// Duels submitted and not yet ended.
    pub in_flight: usize,
    /// How long the round took: the threads (their engines and pilots, and
    /// the packing of each decision), then the building of the batch.
    pub seconds: [f64; 2],
}

/// A duel to start when a slot is free.
#[derive(Clone)]
struct Queued {
    duel: usize,
    setup: Setup,
    keep: bool,
    show: bool,
}

enum Command {
    /// Duels to queue, then the answers to this thread's waiting duels: those
    /// of `group`, or all of them.
    Run { group: Option<usize>, queued: Vec<Queued>, answers: Vec<(usize, u32)> },
    Stop,
}

#[derive(Default)]
struct Report {
    waiting: Vec<Shown>,
    samples: Vec<Shown>,
    /// What became of the cards of decisions kept before, by their duel.
    labels: Vec<(usize, Label)>,
    finished: Vec<Finished>,
    /// How long the thread took over the round.
    seconds: f64,
}

struct Running<'a> {
    duel: usize,
    /// The group whose slot the duel has.
    group: usize,
    episode: Episode<'a>,
    keep: bool,
    show: bool,
    follow: bool,
}

/// A decision as the pool passes it on, and the same with nothing hidden
/// where there is that.
fn shown(environment: &Environment, entry: &Running, view: &View, whole: Option<&View>) -> Result<Shown> {
    let packed = environment.pack(view).map_err(|e| format!("duel {}: {e}", entry.duel))?;
    let whole = whole.map(|whole| environment.pack(whole)).transpose().map_err(|e| format!("duel {}: {e}", entry.duel))?;
    let shown = entry.show.then(|| serde_json::to_value(view)).transpose().map_err(|e| e.to_string())?;
    Ok(Shown {
        duel: entry.duel,
        packed,
        label: view.label,
        pilot: view.pilot,
        chosen: view.chosen,
        player: view.player,
        transaction: view.transaction,
        answer: view.answer,
        step: view.step,
        keep: entry.keep,
        follow: entry.follow,
        view: shown,
        whole,
    })
}

/// The duels every thread draws from, where they are not dealt in advance.
type Shared = Arc<Mutex<VecDeque<Queued>>>;

/// Take a thread's duels one round forward: those of `group`, or all of
/// them.  A group has `slots` of the thread's duels.  A free slot of a group
/// that is taken forward takes its next duel from `shared` where there is
/// one, and from the thread's own `queue` otherwise; the groups are filled
/// in their order.
fn round<'a>(
    environment: &'a Environment,
    queue: &mut VecDeque<Queued>,
    shared: Option<&Shared>,
    running: &mut Vec<Running<'a>>,
    slots: usize,
    groups: usize,
    group: Option<usize>,
    answers: &[(usize, u32)],
) -> Result<Report> {
    let taken = |entry: &Running| group.map_or(true, |group| entry.group == group);
    let waiting = running.iter().filter(|entry| taken(entry)).count();
    if answers.len() != waiting {
        return Err(format!("{} answers for {waiting} waiting duels", answers.len()));
    }
    for (duel, answer) in answers {
        let entry = running
            .iter_mut()
            .find(|entry| entry.duel == *duel && taken(entry))
            .ok_or(format!("duel {duel} does not wait"))?;
        entry.episode.choose(*answer as usize).map_err(|e| format!("duel {duel}: {e}"))?;
    }
    let mut report = Report::default();
    let mut index = 0;
    loop {
        if index == running.len() {
            // Everything in flight waits again: start what a free slot allows.
            let free = group.map_or(0..groups, |group| group..group + 1).find(|group| {
                running.iter().filter(|entry| entry.group == *group).count() < slots
            });
            let Some(free) = free else { break };
            let next = match shared {
                Some(shared) => shared.lock().map_err(|_| "the queue of duels is lost")?.pop_front(),
                None => queue.pop_front(),
            };
            let Some(Queued { duel, setup, keep, show }) = next else { break };
            let episode = environment.start(&setup).map_err(|e| format!("duel {duel}: {e}"))?;
            running.push(Running { duel, group: free, episode, keep, show, follow: setup.follow });
        }
        if !taken(&running[index]) {
            index += 1;
            continue;
        }
        let entry = &mut running[index];
        let duel = entry.duel;
        let Event { samples, decision, whole, labels, over } =
            entry.episode.advance().map_err(|e| format!("duel {duel}: {e}"))?;
        if entry.keep {
            for sample in &samples {
                report.samples.push(shown(environment, entry, sample, None)?);
            }
            report.labels.extend(labels.into_iter().map(|label| (duel, label)));
        }
        if let Some(over) = over {
            report.finished.push(Finished { duel, over });
            running.remove(index);
            continue;
        }
        let waiting = decision.ok_or(format!("duel {duel} neither ended nor asked"))?;
        report.waiting.push(shown(environment, entry, &waiting, whole.as_ref())?);
        index += 1;
    }
    Ok(report)
}

/// A thread of the pool: its own engine, and the duels given to it.
fn work(
    index: usize,
    paths: [PathBuf; 3],
    slots: usize,
    groups: usize,
    shared: Option<Shared>,
    commands: Receiver<Command>,
    reports: Sender<Told>,
) {
    let opened = Environment::open(&paths[0], &paths[1], &paths[2]);
    let mut queue = VecDeque::new();
    let mut running = Vec::new();
    while let Ok(Command::Run { group, queued, answers }) = commands.recv() {
        queue.extend(queued);
        let started = Instant::now();
        let report = match &opened {
            Ok(environment) => {
                round(environment, &mut queue, shared.as_ref(), &mut running, slots, groups, group, &answers)
            }
            Err(message) => Err(message.clone()),
        };
        let report = report.map(|report| Report { seconds: started.elapsed().as_secs_f64(), ..report });
        if reports.send((index, group, report)).is_err() {
            break;
        }
    }
}

/// A thread's report of a round: the thread, the group it took forward (none
/// for all of them) and what came of it.
type Told = (usize, Option<usize>, Result<Report>);

struct Worker {
    commands: Sender<Command>,
    thread: Option<JoinHandle<()>>,
}

/// One number a decision, under a name.
fn column(name: &str, values: Vec<i64>) -> (String, Array) {
    (name.to_owned(), Array { shape: vec![values.len()], data: Data::I64(values) })
}

/// A choice of a decision, or -1 where there is none.
fn choice(choice: Option<usize>) -> i64 {
    choice.map_or(-1, |choice| choice as i64)
}

/// Duels spread over threads, taken forward together.
pub struct Pool {
    workers: Vec<Worker>,
    reports: Receiver<Told>,
    /// Reports that came for another round than the one being gathered.
    early: Vec<Told>,
    /// The groups a thread's duels are shared out among, and for each of
    /// them, where the duels are taken forward a group at a time: the duels
    /// of its last round's decisions with the threads they are on, and since
    /// when it is on its way, begun and not yet gathered.
    groups: usize,
    asked_of: Vec<Vec<usize>>,
    owners_of: Vec<Vec<usize>>,
    moving: Vec<Option<Instant>>,
    /// Duels submitted since the last round, by the thread that gets them.
    queued: Vec<Vec<Queued>>,
    /// Or the one queue all the threads draw from.
    shared: Option<Shared>,
    submitted: usize,
    in_flight: usize,
    /// The duels of the last round's decisions, in the order of its batch,
    /// and the thread each of them is on.
    asked: Vec<usize>,
    owners: Vec<usize>,
    /// A round failed: the duels are no longer in step with one another.
    broken: bool,
    /// The decisions kept, the first of them under the row `forgotten`;
    /// none where one was let go of before those ahead of it, and `held` of
    /// them there.
    kept: VecDeque<Option<Packed>>,
    /// Beside each, the same decision with nothing hidden, where its duel's
    /// setup asked for it: under the same row.
    kept_whole: VecDeque<Option<Packed>>,
    /// Beside each, what became of its cards by its horizon, where its
    /// duel's setup asked to follow them and it was its transaction's last:
    /// filled when its label comes, a round or more after it.  The rows of
    /// the decisions whose labels are still to come, by their duel, answer
    /// and step.
    kept_labels: VecDeque<Option<Label>>,
    awaiting: HashMap<(usize, usize, usize), u64>,
    forgotten: usize,
    held: usize,
}

impl Pool {
    /// `threads` engines over the card database `cards`, the card scripts
    /// `scripts` and the deck lists of `decks`, each with up to `slots` duels
    /// in flight.
    ///
    /// A duel goes to the thread of its number, dealt when it is submitted;
    /// or, with `shared`, to whichever thread has a slot free first, all of
    /// them drawing from one queue, so that no thread stands idle while
    /// another has duels left to start.  Either way a round holds the same
    /// duels every time: as many start as slots come free, in the order they
    /// were submitted.  The two ways do not hold the same duels in a round.
    pub fn open(
        cards: &Path,
        scripts: &Path,
        decks: &Path,
        threads: usize,
        slots: usize,
        shared: bool,
    ) -> Result<Pool> {
        Pool::grouped(cards, scripts, decks, threads, slots, shared, 1)
    }

    /// [`Pool::open`], with every thread's `slots` shared out evenly among
    /// `groups`.  The groups are taken forward one at a time with
    /// [`Pool::begin`] and [`Pool::finish`], so that the engines of one run
    /// while the decisions of another are answered; [`Pool::step`] still
    /// takes all of them forward together, and takes over from the groups
    /// where their rounds have grown too small to be worth a call of the
    /// model each.  A free slot of a group takes the
    /// thread's next duel when that group is taken forward, which one queue
    /// for all the threads would leave to their pace: the two do not go
    /// together.
    pub fn grouped(
        cards: &Path,
        scripts: &Path,
        decks: &Path,
        threads: usize,
        slots: usize,
        shared: bool,
        groups: usize,
    ) -> Result<Pool> {
        if threads == 0 || slots == 0 {
            return Err("a pool has at least one thread and one duel a thread".into());
        }
        if groups == 0 || slots % groups != 0 {
            return Err(format!("{slots} duels a thread are not shared out evenly among {groups} groups"));
        }
        if groups > 1 && shared {
            return Err("groups of duels and one queue for all the threads do not go together".into());
        }
        let shared: Option<Shared> = shared.then(Default::default);
        let (report, reports) = channel();
        let mut workers = Vec::with_capacity(threads);
        for index in 0..threads {
            let (commands, receiver) = channel();
            let paths = [cards.to_owned(), scripts.to_owned(), decks.to_owned()];
            let report = report.clone();
            let queue = shared.clone();
            let thread = std::thread::Builder::new()
                .name(format!("duels-{index}"))
                .spawn(move || work(index, paths, slots / groups, groups, queue, receiver, report))
                .map_err(|e| format!("cannot start a thread: {e}"))?;
            workers.push(Worker { commands, thread: Some(thread) });
        }
        Ok(Pool {
            workers,
            reports,
            early: Vec::new(),
            groups,
            asked_of: vec![Vec::new(); groups],
            owners_of: vec![Vec::new(); groups],
            moving: vec![None; groups],
            queued: vec![Vec::new(); threads],
            shared,
            submitted: 0,
            in_flight: 0,
            asked: Vec::new(),
            owners: Vec::new(),
            broken: false,
            kept: VecDeque::new(),
            kept_whole: VecDeque::new(),
            kept_labels: VecDeque::new(),
            awaiting: HashMap::new(),
            forgotten: 0,
            held: 0,
        })
    }

    /// Queue duels; they start as slots are free.  The first of them is the
    /// duel of the number returned, the others follow it.  With `keep`,
    /// their decisions stay in the pool for [`Pool::batch`]; with `show`,
    /// each of them comes in its round with the view it was packed from
    /// ([`Round::views`]).
    pub fn submit(&mut self, setups: Vec<Setup>, keep: bool, show: bool) -> usize {
        let first = self.submitted;
        // A thread that panicked with the queue in hand leaves it as it was:
        // the next round fails on its own account.
        let mut shared = self.shared.as_ref().map(|shared| shared.lock().unwrap_or_else(|lost| lost.into_inner()));
        for setup in setups {
            let duel = self.submitted;
            let queued = Queued { duel, setup, keep, show };
            match shared.as_mut() {
                Some(shared) => shared.push_back(queued),
                None => self.queued[duel % self.workers.len()].push(queued),
            }
            self.submitted += 1;
            self.in_flight += 1;
        }
        first
    }

    /// Duels submitted and not yet ended.
    pub fn in_flight(&self) -> usize {
        self.in_flight
    }

    /// Give the last round's decisions their answers, each one of its
    /// decision's choices, in the order of that round's batch; then take
    /// every duel to its next decision or to its end.
    ///
    /// Duels that were taken forward a group at a time are taken on together
    /// from here, once no group is on its way: the answers are then those to
    /// the groups' last rounds, group after group.  A duel's course is its
    /// own either way.
    pub fn step(&mut self, answers: &[u32]) -> Result<Round> {
        if self.broken {
            return Err("an earlier round of this pool failed".into());
        }
        if let Some(group) = self.moving.iter().position(Option::is_some) {
            return Err(format!("group {group} is on its way: finish waits for it"));
        }
        let waiting = self.asked.len() + self.asked_of.iter().map(Vec::len).sum::<usize>();
        if answers.len() != waiting {
            return Err(format!("{} answers for the {waiting} decisions of the last round", answers.len()));
        }
        for group in 0..self.groups {
            self.asked.append(&mut self.asked_of[group]);
            self.owners.append(&mut self.owners_of[group]);
        }
        self.broken = true;
        let started = Instant::now();
        let (asked, owners) = (std::mem::take(&mut self.asked), std::mem::take(&mut self.owners));
        self.send(None, &asked, &owners, answers)?;
        let reports = self.gather(None)?;
        self.broken = false;
        let (round, owners) = self.told(reports, started.elapsed().as_secs_f64())?;
        self.asked = round.duels.clone();
        self.owners = owners;
        Ok(round)
    }

    /// Give the decisions of the last round of `group` their answers, in the
    /// order of that round's batch, and set its duels on their way to their
    /// next decision or their end.  It does not wait for them:
    /// [`Pool::finish`] does, and other groups can be begun and finished
    /// meanwhile.
    pub fn begin(&mut self, group: usize, answers: &[u32]) -> Result<()> {
        if self.broken {
            return Err("an earlier round of this pool failed".into());
        }
        if group >= self.groups {
            return Err(format!("group {group} of a pool of {}", self.groups));
        }
        if !self.asked.is_empty() {
            return Err("the duels wait together: step takes them on".into());
        }
        if self.moving[group].is_some() {
            return Err(format!("group {group} is on its way already"));
        }
        if answers.len() != self.asked_of[group].len() {
            return Err(format!(
                "{} answers for the {} decisions of group {group}",
                answers.len(),
                self.asked_of[group].len()
            ));
        }
        let (asked, owners) = (std::mem::take(&mut self.asked_of[group]), std::mem::take(&mut self.owners_of[group]));
        self.send(Some(group), &asked, &owners, answers)?;
        self.moving[group] = Some(Instant::now());
        Ok(())
    }

    /// Wait for the duels of `group`, which was begun, and give its round:
    /// what [`Pool::step`] gives, of that group's duels alone.  Its seconds
    /// are those the slowest thread took over it, whatever else went on
    /// meanwhile.
    pub fn finish(&mut self, group: usize) -> Result<Round> {
        if self.broken {
            return Err("an earlier round of this pool failed".into());
        }
        if self.moving.get(group).copied().flatten().is_none() {
            return Err(format!("group {group} was not begun"));
        }
        self.broken = true;
        let reports = self.gather(Some(group))?;
        self.broken = false;
        self.moving[group] = None;
        let slowest = reports.iter().map(|(_, report)| report.seconds).fold(0.0, f64::max);
        let (round, owners) = self.told(reports, slowest)?;
        self.asked_of[group] = round.duels.clone();
        self.owners_of[group] = owners;
        Ok(round)
    }

    /// Send a round to the threads: the duels of `group`, or all of them,
    /// each with its answer.
    fn send(&mut self, group: Option<usize>, asked: &[usize], owners: &[usize], answers: &[u32]) -> Result<()> {
        let mut given: Vec<Vec<(usize, u32)>> = vec![Vec::new(); self.workers.len()];
        for ((duel, owner), answer) in asked.iter().zip(owners).zip(answers) {
            given[*owner].push((*duel, *answer));
        }
        for ((worker, queued), answers) in self.workers.iter().zip(&mut self.queued).zip(given) {
            worker
                .commands
                .send(Command::Run { group, queued: std::mem::take(queued), answers })
                .map_err(|_| "a thread of the pool has stopped")?;
        }
        Ok(())
    }

    /// Every thread's report of the round of `group`, with the thread it
    /// came from.  A report of another group's round is kept for its turn.
    fn gather(&mut self, group: Option<usize>) -> Result<Vec<(usize, Report)>> {
        let mut reports = Vec::with_capacity(self.workers.len());
        let mut failure: Option<String> = None;
        let (mine, others): (Vec<Told>, Vec<Told>) =
            std::mem::take(&mut self.early).into_iter().partition(|(_, of, _)| *of == group);
        self.early = others;
        let mut came = mine;
        while came.len() < self.workers.len() {
            let told = self.reports.recv().map_err(|_| "a thread of the pool has stopped")?;
            if told.1 == group {
                came.push(told);
            } else {
                self.early.push(told);
            }
        }
        for (thread, _, report) in came {
            match report {
                Ok(report) => reports.push((thread, report)),
                Err(message) => failure = failure.or(Some(message)),
            }
        }
        match failure {
            Some(message) => Err(message),
            None => Ok(reports),
        }
    }

    /// What the threads' reports of a round come to, the threads having
    /// taken `seconds` over it; and the thread each waiting duel is on, in
    /// the order of the batch: its answer goes back there.
    fn told(&mut self, reports: Vec<(usize, Report)>, seconds: f64) -> Result<(Round, Vec<usize>)> {
        let started = Instant::now();
        let mut waiting = Vec::new();
        let mut samples = Vec::new();
        let mut labels = Vec::new();
        let mut finished = Vec::new();
        let mut on: HashMap<usize, usize> = HashMap::new();
        for (thread, report) in reports {
            on.extend(report.waiting.iter().map(|decision| (decision.duel, thread)));
            waiting.extend(report.waiting);
            samples.extend(report.samples);
            labels.extend(report.labels);
            finished.extend(report.finished);
        }
        // A duel's own decisions come from one thread, in the order of play,
        // and these sorts leave them so.
        waiting.sort_by_key(|decision| decision.duel);
        samples.sort_by_key(|decision| decision.duel);
        finished.sort_by_key(|ended| ended.duel);
        self.in_flight -= finished.len();
        let duels: Vec<usize> = waiting.iter().map(|decision| decision.duel).collect();
        let owners = duels.iter().map(|duel| on[duel]).collect();

        // Rows for what is kept: the observed decisions, then the waiting ones.
        let each = |shown: &[Shown], value: fn(&Shown) -> i64| shown.iter().map(value).collect::<Vec<_>>();
        let described = |prefix: &str, shown: &[Shown], rows: Vec<i64>| {
            vec![
                column(&format!("{prefix}row"), rows),
                column(&format!("{prefix}duel"), each(shown, |decision| decision.duel as i64)),
                column(&format!("{prefix}label"), each(shown, |decision| choice(decision.label))),
                column(&format!("{prefix}pilot"), each(shown, |decision| choice(decision.pilot))),
                column(&format!("{prefix}chosen"), each(shown, |decision| choice(decision.chosen))),
                column(&format!("{prefix}player"), each(shown, |decision| i64::from(decision.player))),
                column(&format!("{prefix}transaction"), each(shown, |decision| i64::from(decision.transaction))),
                column(&format!("{prefix}answer"), each(shown, |decision| decision.answer as i64)),
            ]
        };
        let mut keep = |decision: &mut Shown| {
            if !decision.keep {
                return -1;
            }
            self.kept.push_back(Some(std::mem::take(&mut decision.packed)));
            self.kept_whole.push_back(decision.whole.take());
            self.kept_labels.push_back(None);
            self.held += 1;
            let row = (self.forgotten + self.kept.len() - 1) as i64;
            if decision.follow {
                self.awaiting.insert((decision.duel, decision.answer, decision.step), row as u64);
            }
            row
        };
        let sampled: Vec<i64> = samples.iter_mut().map(&mut keep).collect();
        let rows: Vec<i64> = waiting.iter_mut().map(&mut keep).collect();
        // A label is of a decision kept in an earlier round: the decision's
        // duel went on to its horizon only once it was answered.  One whose
        // decision was let go of meanwhile is dropped.
        for (duel, label) in labels {
            let Some(row) = self.awaiting.remove(&(duel, label.answer, label.step)) else { continue };
            if let Some(place) = (row as usize).checked_sub(self.forgotten) {
                if self.kept.get(place).is_some_and(Option::is_some) {
                    self.kept_labels[place] = Some(label);
                }
            }
        }
        let views = waiting.iter_mut().map(|decision| decision.view.take()).collect();
        let sample_views = samples.iter_mut().map(|decision| decision.view.take()).collect();
        let batch = if waiting.is_empty() {
            None
        } else {
            let decisions: Vec<&Packed> = waiting
                .iter()
                .zip(&rows)
                .map(|(decision, row)| match row {
                    row if *row < 0 => Ok(&decision.packed),
                    row => self.kept[*row as usize - self.forgotten]
                        .as_ref()
                        .ok_or_else(|| format!("the decision of row {row} is let go of")),
                })
                .collect::<Result<_>>()?;
            let mut batch = pack::batch(&decisions)?;
            batch.arrays.extend(described("meta_", &waiting, rows));
            Some(batch)
        };
        let samples = if samples.is_empty() { Vec::new() } else { described("sample_", &samples, sampled) };
        let seconds = [seconds, started.elapsed().as_secs_f64()];
        let round = Round { batch, duels, samples, views, sample_views, finished, in_flight: self.in_flight, seconds };
        Ok((round, owners))
    }

    /// How many decisions are kept.
    pub fn kept(&self) -> usize {
        self.held
    }

    /// Where the decision of `row` is kept, if it is.
    fn place(&self, row: u64) -> Result<usize> {
        (row as usize)
            .checked_sub(self.forgotten)
            .filter(|place| self.kept.get(*place).is_some_and(Option::is_some))
            .ok_or_else(|| {
                format!(
                    "no decision is kept under row {row}: the rows kept are {} to {}, less those let go of",
                    self.forgotten,
                    self.forgotten + self.kept.len()
                )
            })
    }

    /// The kept decisions of `rows` as one batch, in that order.
    pub fn batch(&self, rows: &[u64]) -> Result<Batch> {
        let mut decisions = Vec::with_capacity(rows.len());
        for row in rows {
            let place = self.place(*row)?;
            decisions.push(self.kept[place].as_ref().expect("a kept place holds its decision"));
        }
        pack::batch(&decisions)
    }

    /// The decisions of `rows` with nothing hidden, as one batch in that
    /// order: each kept, and of a duel whose setup asked for it.
    pub fn batch_whole(&self, rows: &[u64]) -> Result<Batch> {
        let mut decisions = Vec::with_capacity(rows.len());
        for row in rows {
            let place = self.place(*row)?;
            let whole = self.kept_whole[place].as_ref();
            decisions.push(whole.ok_or_else(|| format!("the decision of row {row} is kept without its whole view"))?);
        }
        pack::batch(&decisions)
    }

    /// What became of the cards of the decisions of `rows` by their
    /// horizons, as one batch in that order: each decision kept.  A row
    /// without a label (one of a duel whose cards were not followed, or not
    /// its transaction's last decision) is `labelled` false, its places -1.
    ///
    /// * `card_place`: of each card of the decision's state, its place at
    ///   the horizon (`follow::PLACES` of them), -1 where none: padded to the
    ///   longest state of the rows labelled;
    /// * `card_change`: the change of each card's ATK and DEF, scaled, not a
    ///   number where there is none;
    /// * `life_change`, `count_change`: the change of each seat's life
    ///   points, and of its cards in each location counted;
    /// * `label_checked`, `label_agreed`: the cards whose identity the seat
    ///   knew both at the decision and at the horizon, and those of them
    ///   found where they were followed to.
    pub fn batch_labels(&self, rows: &[u64]) -> Result<Batch> {
        let mut labels = Vec::with_capacity(rows.len());
        for row in rows {
            let place = self.place(*row)?;
            labels.push(self.kept_labels[place].as_ref());
        }
        let width = labels.iter().flatten().map(|label| label.places.len()).max().unwrap_or(0).max(1);
        let count = rows.len();
        let mut place = vec![-1i64; count * width];
        let mut change = vec![f32::NAN; count * width * 2];
        let mut life = vec![0f32; count * 2];
        let mut counts = vec![0f32; count * 2 * COUNTED];
        let (mut labelled, mut checked, mut agreed) = (vec![0u8; count], vec![0i64; count], vec![0i64; count]);
        for (row, label) in labels.iter().enumerate() {
            let Some(label) = label else { continue };
            labelled[row] = 1;
            checked[row] = i64::from(label.checked);
            agreed[row] = i64::from(label.agreed);
            place[row * width..row * width + label.places.len()].copy_from_slice(&label.places);
            for (card, [attack, defense]) in label.changes.iter().enumerate() {
                change[(row * width + card) * 2] = *attack;
                change[(row * width + card) * 2 + 1] = *defense;
            }
            life[row * 2..row * 2 + 2].copy_from_slice(&label.life);
            counts[row * 2 * COUNTED..(row + 1) * 2 * COUNTED].copy_from_slice(&label.counts);
        }
        let array = |shape: Vec<usize>, data: Data| Array { shape, data };
        Ok(Batch {
            arrays: vec![
                ("labelled".to_owned(), array(vec![count], Data::Bool(labelled))),
                ("card_place".to_owned(), array(vec![count, width], Data::I64(place))),
                ("card_change".to_owned(), array(vec![count, width, 2], Data::F32(change))),
                ("life_change".to_owned(), array(vec![count, 2], Data::F32(life))),
                ("count_change".to_owned(), array(vec![count, 2 * COUNTED], Data::F32(counts))),
                ("label_checked".to_owned(), array(vec![count], Data::I64(checked))),
                ("label_agreed".to_owned(), array(vec![count], Data::I64(agreed))),
            ],
        })
    }

    /// Let go of the decisions kept.  Rows are not given again: the next
    /// decision kept follows the last one.
    pub fn forget(&mut self) -> Result<()> {
        if self.in_flight > 0 {
            return Err(format!("{} duels are in flight", self.in_flight));
        }
        self.forgotten += self.kept.len();
        self.kept.clear();
        self.kept_whole.clear();
        self.kept_labels.clear();
        self.awaiting.clear();
        self.held = 0;
        Ok(())
    }

    /// Let go of the decisions of `rows`, each kept and named once, whatever
    /// is in flight: all of them, or none where one is not.  Rows are not
    /// given again.
    pub fn forget_rows(&mut self, rows: &[u64]) -> Result<()> {
        let mut places = rows.iter().map(|row| self.place(*row)).collect::<Result<Vec<_>>>()?;
        places.sort_unstable();
        if places.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("a row is named twice".into());
        }
        for place in places {
            self.kept[place] = None;
            self.kept_whole[place] = None;
            self.kept_labels[place] = None;
            self.held -= 1;
        }
        while self.kept.front().is_some_and(Option::is_none) {
            self.kept.pop_front();
            self.kept_whole.pop_front();
            self.kept_labels.pop_front();
            self.forgotten += 1;
        }
        Ok(())
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        for worker in &self.workers {
            let _ = worker.commands.send(Command::Stop);
        }
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }
    }
}
