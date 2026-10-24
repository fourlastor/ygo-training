//! The C interface the Python trainer loads.
//!
//! Text goes both ways as NUL-terminated UTF-8, structured values as JSON.
//! A returned string belongs to the handle it came from and lasts until the
//! next call on that handle.  A call that fails returns null (or -1) and
//! leaves its message for [`ygo_env_last_error`], which is per thread.

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

use serde::Deserialize;

use crate::edopro::{Event, Kept, Room};
use crate::episode::{Environment, Episode, Setup};
use crate::pack::{Array, Batch};
use crate::pool::{Pool, Round};
use crate::{view, Result};

thread_local! {
    static ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn text(value: String) -> CString {
    CString::new(value.replace('\0', " ")).expect("no NUL is left in the text")
}

fn guarded<T>(fallback: T, body: impl FnOnce() -> Result<T>) -> T {
    let message = match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(value)) => return value,
        Ok(Err(message)) => message,
        Err(panic) => match panic.downcast_ref::<&str>() {
            Some(message) => format!("panic: {message}"),
            None => match panic.downcast_ref::<String>() {
                Some(message) => format!("panic: {message}"),
                None => "panic".to_owned(),
            },
        },
    };
    ERROR.with(|error| *error.borrow_mut() = text(message));
    fallback
}

unsafe fn read<'a>(value: *const c_char, what: &str) -> Result<&'a str> {
    if value.is_null() {
        return Err(format!("{what} is null"));
    }
    CStr::from_ptr(value).to_str().map_err(|e| format!("{what}: {e}"))
}

/// An [`Environment`] and the last text it returned.
pub struct Env {
    environment: Environment,
    text: CString,
}

/// A duel of an [`Env`] and the last text it returned.
pub struct Session {
    // Borrowed from its `Env`, which the caller keeps open until the
    // session is finished.
    episode: Episode<'static>,
    text: CString,
}

/// The policies of the pool, each with the name of its deck list, as JSON:
/// what can be seated, known without opening the engine.  The string lasts
/// until the next call on this thread.
#[no_mangle]
pub extern "C" fn ygo_env_pool() -> *const c_char {
    thread_local! {
        static POOL: RefCell<CString> = RefCell::new(CString::default());
    }
    let pool: Vec<_> = ygo_policies::registry::POLICIES
        .iter()
        .map(|entry| serde_json::json!({"id": entry.id, "deck": entry.deck}))
        .collect();
    POOL.with(|kept| {
        *kept.borrow_mut() = text(serde_json::Value::from(pool).to_string());
        kept.borrow().as_ptr()
    })
}

/// Open the engine: the card database, the card scripts and the folder of
/// deck lists.
///
/// # Safety
/// The three arguments are NUL-terminated strings.
#[no_mangle]
pub unsafe extern "C" fn ygo_env_open(cards: *const c_char, scripts: *const c_char, decks: *const c_char) -> *mut Env {
    guarded(std::ptr::null_mut(), || {
        let environment = Environment::open(
            Path::new(read(cards, "cards")?),
            Path::new(read(scripts, "scripts")?),
            Path::new(read(decks, "decks")?),
        )?;
        Ok(Box::into_raw(Box::new(Env { environment, text: CString::default() })))
    })
}

/// The decks of the pool (`decks`) and the names behind the numbers of a
/// decision (`vocabulary`), as JSON.
///
/// # Safety
/// `env` comes from [`ygo_env_open`].
#[no_mangle]
pub unsafe extern "C" fn ygo_env_describe(env: *mut Env) -> *const c_char {
    guarded(std::ptr::null(), || {
        let env = env.as_mut().ok_or("env is null")?;
        let described = serde_json::json!({"decks": env.environment.catalog(), "vocabulary": view::vocabulary()});
        env.text = text(described.to_string());
        Ok(env.text.as_ptr())
    })
}

/// Start a duel from the JSON form of a [`Setup`].
///
/// # Safety
/// `env` comes from [`ygo_env_open`] and stays open until the session is
/// finished; `setup` is a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ygo_env_start(env: *mut Env, setup: *const c_char) -> *mut Session {
    guarded(std::ptr::null_mut(), || {
        let env = env.as_ref().ok_or("env is null")?;
        let setup: Setup = serde_json::from_str(read(setup, "setup")?).map_err(|e| format!("setup: {e}"))?;
        let episode = env.environment.start(&setup)?;
        // SAFETY: the caller keeps `env` open for as long as the session lives.
        let episode = std::mem::transmute::<Episode<'_>, Episode<'static>>(episode);
        Ok(Box::into_raw(Box::new(Session { episode, text: CString::default() })))
    })
}

/// Run the duel until the caller has to decide or the duel ends: the JSON
/// form of an [`Event`](crate::Event).
///
/// # Safety
/// `session` comes from [`ygo_env_start`].
#[no_mangle]
pub unsafe extern "C" fn ygo_env_advance(session: *mut Session) -> *const c_char {
    guarded(std::ptr::null(), || {
        let session = session.as_mut().ok_or("session is null")?;
        let event = session.episode.advance()?;
        session.text = text(serde_json::to_string(&event).map_err(|e| e.to_string())?);
        Ok(session.text.as_ptr())
    })
}

/// Answer the pending decision with one of its choices.  0, or -1 on failure.
///
/// # Safety
/// `session` comes from [`ygo_env_start`].
#[no_mangle]
pub unsafe extern "C" fn ygo_env_choose(session: *mut Session, index: u32) -> i32 {
    guarded(-1, || {
        let session = session.as_mut().ok_or("session is null")?;
        session.episode.choose(index as usize)?;
        Ok(0)
    })
}

/// End a duel, finished or not.
///
/// # Safety
/// `session` comes from [`ygo_env_start`] and is not used again.
#[no_mangle]
pub unsafe extern "C" fn ygo_env_finish(session: *mut Session) {
    if !session.is_null() {
        drop(Box::from_raw(session));
    }
}

/// Close the engine, once every session of it is finished.
///
/// # Safety
/// `env` comes from [`ygo_env_open`] and is not used again.
#[no_mangle]
pub unsafe extern "C" fn ygo_env_close(env: *mut Env) {
    if !env.is_null() {
        drop(Box::from_raw(env));
    }
}

/// Decisions to turn into one batch.
#[derive(Deserialize)]
struct Decisions {
    views: Vec<view::View>,
}

/// A batch, with the description of its arrays.
pub struct Tensors {
    batch: Batch,
    layout: CString,
}

/// The name, the type and the shape of an array.
fn described(name: &str, array: &Array) -> serde_json::Value {
    serde_json::json!({"name": name, "dtype": array.data.kind(), "shape": array.shape})
}

impl Tensors {
    fn new(batch: Batch) -> Tensors {
        let layout: Vec<_> = batch.arrays.iter().map(|(name, array)| described(name, array)).collect();
        Tensors { batch, layout: text(serde_json::Value::from(layout).to_string()) }
    }
}

/// The decisions of `request` as the batch the model is given: the JSON
/// form of the views, as [`ygo_env_advance`] returned them.  Its arrays
/// are read with [`ygo_env_batch_layout`] and
/// [`ygo_env_batch_data`] and last until [`ygo_env_batch_free`].
///
/// # Safety
/// `env` comes from [`ygo_env_open`]; `request` is a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ygo_env_batch(env: *mut Env, request: *const c_char) -> *mut Tensors {
    guarded(std::ptr::null_mut(), || {
        let env = env.as_ref().ok_or("env is null")?;
        let request: Decisions =
            serde_json::from_str(read(request, "request")?).map_err(|e| format!("request: {e}"))?;
        let mut packed = Vec::with_capacity(request.views.len());
        for shown in &request.views {
            packed.push(env.environment.pack(shown)?);
        }
        let batch = env.environment.batch(&packed.iter().collect::<Vec<_>>())?;
        Ok(Box::into_raw(Box::new(Tensors::new(batch))))
    })
}

/// One decision as the arrays the model reads before a batch pads them: the
/// JSON form of its view, as [`ygo_env_advance`] returned it.  They are read
/// and given back as a batch's are.
///
/// # Safety
/// `env` comes from [`ygo_env_open`]; `view` is a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ygo_env_pack(env: *mut Env, view: *const c_char) -> *mut Tensors {
    guarded(std::ptr::null_mut(), || {
        let env = env.as_ref().ok_or("env is null")?;
        let shown: view::View = serde_json::from_str(read(view, "view")?).map_err(|e| format!("view: {e}"))?;
        let packed = env.environment.pack(&shown)?;
        let arrays = packed.arrays.iter().map(|(name, array)| (name.clone(), Array::clone(array))).collect();
        Ok(Box::into_raw(Box::new(Tensors::new(Batch { arrays }))))
    })
}

/// The arrays of a batch, in order: the name, the type (`int64`, `float32`
/// or `bool`) and the shape of each, as JSON.
///
/// # Safety
/// `batch` comes from [`ygo_env_batch`].
#[no_mangle]
pub unsafe extern "C" fn ygo_env_batch_layout(batch: *const Tensors) -> *const c_char {
    guarded(std::ptr::null(), || Ok(batch.as_ref().ok_or("batch is null")?.layout.as_ptr()))
}

/// The numbers of the array at `index` of the layout, in row-major order.
///
/// # Safety
/// `batch` comes from [`ygo_env_batch`].
#[no_mangle]
pub unsafe extern "C" fn ygo_env_batch_data(batch: *const Tensors, index: u32) -> *const u8 {
    guarded(std::ptr::null(), || {
        let tensors = batch.as_ref().ok_or("batch is null")?;
        let (_, array) = tensors.batch.arrays.get(index as usize).ok_or("the batch has no such array")?;
        Ok(array.data.as_ptr())
    })
}

/// Give a batch back.
///
/// # Safety
/// `batch` comes from [`ygo_env_batch`] and is not used again.
#[no_mangle]
pub unsafe extern "C" fn ygo_env_batch_free(batch: *mut Tensors) {
    if !batch.is_null() {
        drop(Box::from_raw(batch));
    }
}

/// A round of a pool, with its description.
pub struct Stepped {
    round: Round,
    layout: CString,
}

/// The arrays of a round, in the order of its layout.
fn arrays(round: &Round) -> impl Iterator<Item = &(String, Array)> {
    round.batch.iter().flat_map(|batch| &batch.arrays).chain(&round.samples)
}

/// Open a pool of `threads` engines, each with up to `slots` duels in
/// flight, over the card database, the card scripts and the folder of deck
/// lists.  With `shared` other than 0 the threads draw their duels from one
/// queue, where each is otherwise dealt its own ([`Pool::open`]).  A
/// thread's duels are shared out among `groups`, which [`ygo_pool_begin`]
/// and [`ygo_pool_finish`] take forward one at a time ([`Pool::grouped`]);
/// one group is a pool as [`ygo_pool_step`] alone takes it forward.
///
/// # Safety
/// The first three arguments are NUL-terminated strings.
#[no_mangle]
pub unsafe extern "C" fn ygo_pool_open(
    cards: *const c_char,
    scripts: *const c_char,
    decks: *const c_char,
    threads: u32,
    slots: u32,
    shared: u32,
    groups: u32,
) -> *mut Pool {
    guarded(std::ptr::null_mut(), || {
        let pool = Pool::grouped(
            Path::new(read(cards, "cards")?),
            Path::new(read(scripts, "scripts")?),
            Path::new(read(decks, "decks")?),
            threads as usize,
            slots as usize,
            shared != 0,
            groups as usize,
        )?;
        Ok(Box::into_raw(Box::new(pool)))
    })
}

/// Queue duels: a JSON list of [`Setup`]s.  The number of the first, the
/// others following it; -1 on failure.  With `keep` other than 0 their
/// decisions stay in the pool, for [`ygo_pool_batch`]; with `show` other
/// than 0 each comes in its round with the view it was packed from.
///
/// # Safety
/// `pool` comes from [`ygo_pool_open`]; `setups` is a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ygo_pool_submit(pool: *mut Pool, setups: *const c_char, keep: u32, show: u32) -> i64 {
    guarded(-1, || {
        let pool = pool.as_mut().ok_or("pool is null")?;
        let setups: Vec<Setup> = serde_json::from_str(read(setups, "setups")?).map_err(|e| format!("setups: {e}"))?;
        Ok(pool.submit(setups, keep != 0, show != 0) as i64)
    })
}

/// Give the last round's decisions their answers, in the order of its
/// batch, and take every duel to its next decision or to its end.  The
/// round is read with [`ygo_pool_round_layout`] and [`ygo_pool_round_data`]
/// and lasts until [`ygo_pool_round_free`].
///
/// # Safety
/// `pool` comes from [`ygo_pool_open`]; `answers` holds `count` numbers.
#[no_mangle]
pub unsafe extern "C" fn ygo_pool_step(pool: *mut Pool, answers: *const u32, count: usize) -> *mut Stepped {
    guarded(std::ptr::null_mut(), || {
        let pool = pool.as_mut().ok_or("pool is null")?;
        stepped(pool.step(given(answers, count)?)?)
    })
}

/// Give the decisions of the last round of `group` their answers, in the
/// order of its batch, and set its duels on their way; it does not wait for
/// them ([`Pool::begin`]).  0, or -1 on failure.
///
/// # Safety
/// `pool` comes from [`ygo_pool_open`]; `answers` holds `count` numbers.
#[no_mangle]
pub unsafe extern "C" fn ygo_pool_begin(pool: *mut Pool, group: u32, answers: *const u32, count: usize) -> i32 {
    guarded(-1, || {
        let pool = pool.as_mut().ok_or("pool is null")?;
        pool.begin(group as usize, given(answers, count)?)?;
        Ok(0)
    })
}

/// Wait for the duels of `group`, which was begun, and give its round, as
/// [`ygo_pool_step`] gives the round of all the duels ([`Pool::finish`]).
///
/// # Safety
/// `pool` comes from [`ygo_pool_open`].
#[no_mangle]
pub unsafe extern "C" fn ygo_pool_finish(pool: *mut Pool, group: u32) -> *mut Stepped {
    guarded(std::ptr::null_mut(), || {
        let pool = pool.as_mut().ok_or("pool is null")?;
        stepped(pool.finish(group as usize)?)
    })
}

/// The answers a caller hands over.
///
/// # Safety
/// `answers` holds `count` numbers.
unsafe fn given<'a>(answers: *const u32, count: usize) -> Result<&'a [u32]> {
    if count == 0 {
        Ok(&[][..])
    } else if answers.is_null() {
        Err("answers is null".into())
    } else {
        Ok(std::slice::from_raw_parts(answers, count))
    }
}

/// A round with its description, for the caller to read and free.
fn stepped(mut round: Round) -> Result<*mut Stepped> {
    let arrays: Vec<_> = arrays(&round).map(|(name, array)| described(name, array)).collect();
    let mut finished = Vec::with_capacity(round.finished.len());
    for ended in &round.finished {
        let mut over = serde_json::to_value(&ended.over).map_err(|e| e.to_string())?;
        over["duel"] = ended.duel.into();
        finished.push(over);
    }
    let seconds = serde_json::json!({"threads": round.seconds[0], "batch": round.seconds[1]});
    let layout = serde_json::json!({
        "arrays": arrays,
        "finished": finished,
        "in_flight": round.in_flight,
        "seconds": seconds,
        "views": std::mem::take(&mut round.views),
        "sample_views": std::mem::take(&mut round.sample_views),
    });
    Ok(Box::into_raw(Box::new(Stepped { round, layout: text(layout.to_string()) })))
}

/// A round as JSON: `arrays`, the name, type and shape of each of its
/// arrays (the model's input for the waiting decisions, what tells them
/// apart under `meta_` names, then one number for each decision of an
/// observed seat under `sample_` names); `finished`, the duels that ended,
/// each with its number and how it ended; `in_flight`, the duels not yet
/// ended; `seconds`, how long the `threads` took and then the `batch`;
/// `views` and `sample_views`, the view of each waiting decision and of
/// each observed one, null where its duel does not show them.
///
/// # Safety
/// `round` comes from [`ygo_pool_step`].
#[no_mangle]
pub unsafe extern "C" fn ygo_pool_round_layout(round: *const Stepped) -> *const c_char {
    guarded(std::ptr::null(), || Ok(round.as_ref().ok_or("round is null")?.layout.as_ptr()))
}

/// The numbers of the array at `index` of a round's layout.
///
/// # Safety
/// `round` comes from [`ygo_pool_step`].
#[no_mangle]
pub unsafe extern "C" fn ygo_pool_round_data(round: *const Stepped, index: u32) -> *const u8 {
    guarded(std::ptr::null(), || {
        let stepped = round.as_ref().ok_or("round is null")?;
        let (_, array) = arrays(&stepped.round).nth(index as usize).ok_or("the round has no such array")?;
        Ok(array.data.as_ptr())
    })
}

/// The kept decisions of the `count` rows at `rows` as one batch, in that
/// order: read and given back as the batch of [`ygo_env_batch`] is.
///
/// # Safety
/// `pool` comes from [`ygo_pool_open`]; `rows` holds `count` numbers.
#[no_mangle]
pub unsafe extern "C" fn ygo_pool_batch(pool: *const Pool, rows: *const u64, count: usize) -> *mut Tensors {
    guarded(std::ptr::null_mut(), || {
        let pool = pool.as_ref().ok_or("pool is null")?;
        if rows.is_null() || count == 0 {
            return Err("cannot batch an empty list of decisions".into());
        }
        let batch = pool.batch(std::slice::from_raw_parts(rows, count))?;
        Ok(Box::into_raw(Box::new(Tensors::new(batch))))
    })
}

/// The decisions a pool keeps under `rows` with nothing hidden, as one
/// batch in that order, read as [`ygo_pool_batch`]'s is: each of a duel
/// whose setup asked for it (`whole`).  Null on failure.
///
/// # Safety
/// `pool` comes from [`ygo_pool_open`]; `rows` holds `count` numbers.
#[no_mangle]
pub unsafe extern "C" fn ygo_pool_batch_whole(pool: *const Pool, rows: *const u64, count: usize) -> *mut Tensors {
    guarded(std::ptr::null_mut(), || {
        let pool = pool.as_ref().ok_or("pool is null")?;
        if rows.is_null() || count == 0 {
            return Err("cannot batch an empty list of decisions".into());
        }
        let batch = pool.batch_whole(std::slice::from_raw_parts(rows, count))?;
        Ok(Box::into_raw(Box::new(Tensors::new(batch))))
    })
}

/// What became of the cards of the kept decisions of `rows` by their
/// horizons, as [`Pool::batch_labels`] gives them; null on failure.
///
/// # Safety
/// `pool` comes from [`ygo_pool_open`]; `rows` holds `count` numbers.
#[no_mangle]
pub unsafe extern "C" fn ygo_pool_batch_labels(pool: *const Pool, rows: *const u64, count: usize) -> *mut Tensors {
    guarded(std::ptr::null_mut(), || {
        let pool = pool.as_ref().ok_or("pool is null")?;
        if rows.is_null() || count == 0 {
            return Err("cannot batch an empty list of decisions".into());
        }
        let batch = pool.batch_labels(std::slice::from_raw_parts(rows, count))?;
        Ok(Box::into_raw(Box::new(Tensors::new(batch))))
    })
}

/// How many decisions a pool keeps; -1 on failure.
///
/// # Safety
/// `pool` comes from [`ygo_pool_open`].
#[no_mangle]
pub unsafe extern "C" fn ygo_pool_kept(pool: *const Pool) -> i64 {
    guarded(-1, || Ok(pool.as_ref().ok_or("pool is null")?.kept() as i64))
}

/// Let go of the decisions a pool keeps: 0, or -1 on failure.
///
/// # Safety
/// `pool` comes from [`ygo_pool_open`].
#[no_mangle]
pub unsafe extern "C" fn ygo_pool_forget(pool: *mut Pool) -> i32 {
    guarded(-1, || {
        pool.as_mut().ok_or("pool is null")?.forget()?;
        Ok(0)
    })
}

/// Let go of the decisions a pool keeps under `rows`, each kept and named
/// once, while duels are in flight or not ([`Pool::forget_rows`]): 0, or -1
/// on failure, when none is let go of.
///
/// # Safety
/// `pool` comes from [`ygo_pool_open`]; `rows` holds `count` numbers.
#[no_mangle]
pub unsafe extern "C" fn ygo_pool_forget_rows(pool: *mut Pool, rows: *const u64, count: usize) -> i32 {
    guarded(-1, || {
        let pool = pool.as_mut().ok_or("pool is null")?;
        if rows.is_null() && count > 0 {
            return Err("rows is null".into());
        }
        let rows = if count == 0 { &[][..] } else { std::slice::from_raw_parts(rows, count) };
        pool.forget_rows(rows)?;
        Ok(0)
    })
}

/// Give a round back.
///
/// # Safety
/// `round` comes from [`ygo_pool_step`] and is not used again.
#[no_mangle]
pub unsafe extern "C" fn ygo_pool_round_free(round: *mut Stepped) {
    if !round.is_null() {
        drop(Box::from_raw(round));
    }
}

/// Close a pool and stop its threads.
///
/// # Safety
/// `pool` comes from [`ygo_pool_open`] and is not used again.
#[no_mangle]
pub unsafe extern "C" fn ygo_pool_close(pool: *mut Pool) {
    if !pool.is_null() {
        drop(Box::from_raw(pool));
    }
}

/// Why the last call on this thread failed.
#[no_mangle]
pub extern "C" fn ygo_env_last_error() -> *const c_char {
    ERROR.with(|error| error.borrow().as_ptr())
}

/// What the caller of [`ygo_env_edopro`] is told or asked.  Null: nothing
/// has happened for a while.  Else JSON: `{"note": text}`, how the session
/// goes on, or `{"decision": view}`, a decision as an episode shows it.  It
/// returns the choice it takes of a decision and 0 otherwise; a negative
/// number ends the session.
pub type Asked = extern "C" fn(*const c_char) -> i32;

#[derive(Deserialize)]
struct Table {
    /// The deck the caller plays, by policy id: its pilot sits alongside.
    deck: String,
    /// The room to join.
    #[serde(default)]
    room: Option<Room>,
    /// Or a trace of `edopro-bot` to feed the seat again.
    #[serde(default)]
    trace: Option<String>,
}

/// Seat the caller in a room of EDOPro, or before a trace of one, and play:
/// the JSON form of a [`Table`] in, `{"played": [...]}` or `{"replayed":
/// {...}}` out.  `ask` is called on this thread until the session is over.
///
/// # Safety
/// `env` comes from [`ygo_env_open`]; `table` is a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ygo_env_edopro(env: *mut Env, table: *const c_char, ask: Option<Asked>) -> *const c_char {
    guarded(std::ptr::null(), || {
        let env = env.as_mut().ok_or("env is null")?;
        let ask = ask.ok_or("no caller is given")?;
        let table: Table = serde_json::from_str(read(table, "table")?).map_err(|e| format!("table: {e}"))?;
        let mut asking = |event: Event<'_>| {
            let told = match event {
                Event::Still => None,
                Event::Note(note) => Some(serde_json::json!({"note": note})),
                Event::Decision(decision) => Some(serde_json::json!({"decision": decision})),
            };
            let answer = match told {
                None => ask(std::ptr::null()),
                Some(told) => ask(text(told.to_string()).as_ptr()),
            };
            usize::try_from(answer).map_err(|_| "the caller ended the session".to_owned())
        };
        let done = match (&table.room, &table.trace) {
            (Some(room), None) => {
                serde_json::json!({"played": env.environment.edopro(&table.deck, room, &mut asking)?})
            }
            (None, Some(trace)) => {
                let replayed = env.environment.edopro_replay(&table.deck, Path::new(trace), &mut asking)?;
                serde_json::json!({"replayed": replayed})
            }
            _ => return Err("a table is a room or a trace: one of the two".into()),
        };
        env.text = text(done.to_string());
        Ok(env.text.as_ptr())
    })
}

/// Play a duel EDOPro kept again, to tell who gave each of its answers: the
/// JSON form of a [`Kept`] in, of a `Reread` out.
///
/// # Safety
/// `env` comes from [`ygo_env_open`]; `kept` is a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ygo_env_kept(env: *mut Env, kept: *const c_char) -> *const c_char {
    guarded(std::ptr::null(), || {
        let env = env.as_mut().ok_or("env is null")?;
        let kept: Kept = serde_json::from_str(read(kept, "kept")?).map_err(|e| format!("kept: {e}"))?;
        let reread = env.environment.kept(&kept)?;
        env.text = text(serde_json::to_string(&reread).map_err(|e| e.to_string())?);
        Ok(env.text.as_ptr())
    })
}
