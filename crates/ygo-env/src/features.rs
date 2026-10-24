//! Printed card data as the policy reads it.
//!
//! The arithmetic is shared with whatever else seats a checkpoint and lives
//! in `native/ygo-model-input` (`input/src/features.rs`); what is here reads
//! the card database the trainer's duels are played with.

use std::path::Path;

pub use ygo_model_input::features::*;

use crate::Result;

/// Every card of the database.
pub fn open(database: &Path) -> Result<Features> {
    let failed = |e: rusqlite::Error| format!("{}: {e}", database.display());
    let db = rusqlite::Connection::open_with_flags(
        database,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(failed)?;
    let mut statement =
        db.prepare("SELECT id, type, atk, def, level, race, attribute, setcode FROM datas").map_err(failed)?;
    let rows = statement
        .query_map([], |entry| {
            let mut row = [0i64; 8];
            for (index, value) in row.iter_mut().enumerate() {
                *value = entry.get(index)?;
            }
            Ok(row)
        })
        .map_err(failed)?;
    let rows = rows.collect::<std::result::Result<Vec<_>, _>>().map_err(failed)?;
    Ok(Features::from_rows(rows))
}
