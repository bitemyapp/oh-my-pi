//! Shared SQLite initialization for independently launched inference clients.

use std::{
	thread,
	time::{Duration, Instant},
};

use rusqlite::{Connection, ErrorCode};

/// WAL transition can bypass SQLite's busy handler. Retry only transient
/// contention, bounded by the same startup budget as ordinary database writes.
pub(crate) fn configure_journal(
	connection: &Connection,
	timeout: Duration,
) -> rusqlite::Result<()> {
	let started = Instant::now();
	loop {
		let result: rusqlite::Result<()> = (|| {
			let mode: String =
				connection.pragma_query_value(None, "journal_mode", |row| row.get(0))?;
			if !mode.eq_ignore_ascii_case("wal") {
				connection.pragma_update(None, "journal_mode", "WAL")?;
			}
			Ok(())
		})();
		match result {
			// Concurrent first-open journal transitions can return BUSY without
			// invoking SQLite's busy handler. Bound this transition retry too.
			Err(error)
				if matches!(
					error.sqlite_error_code(),
					Some(ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked)
				) && started.elapsed() < timeout =>
			{
				thread::sleep(Duration::from_millis(10).min(timeout.saturating_sub(started.elapsed())));
			},
			result => return result,
		}
	}
}

#[cfg(test)]
mod tests {
	use rusqlite::hooks::{AuthAction, AuthContext, Authorization};

	use super::*;

	#[test]
	fn journal_transition_retries_busy_until_reader_releases() {
		let directory = tempfile::tempdir().expect("directory");
		let path = directory.path().join("store.sqlite");
		let reader = Connection::open(&path).expect("reader");
		reader
			.execute_batch("CREATE TABLE fixture (id INTEGER); BEGIN; SELECT * FROM fixture;")
			.expect("hold read lease");
		let peer = Connection::open(&path).expect("peer");
		peer
			.busy_timeout(Duration::ZERO)
			.expect("immediate SQLite busy errors");
		let (attempt_tx, attempt_rx) = flume::bounded(1);
		peer
			.authorizer(Some(move |context: AuthContext<'_>| {
				if matches!(context.action, AuthAction::Pragma {
					pragma_name:  "journal_mode",
					pragma_value: Some(_),
				}) {
					let _ = attempt_tx.try_send(());
				}
				Authorization::Allow
			}))
			.expect("transition scheduling hook");
		let (finished_tx, finished_rx) = flume::bounded(1);
		let worker = thread::spawn(move || {
			finished_tx
				.send(configure_journal(&peer, Duration::from_secs(1)))
				.expect("transition result")
		});
		attempt_rx
			.recv_timeout(Duration::from_secs(1))
			.expect("peer tries WAL transition");
		assert!(matches!(
			finished_rx.recv_timeout(Duration::from_millis(30)),
			Err(flume::RecvTimeoutError::Timeout)
		));
		reader.execute_batch("ROLLBACK").expect("release reader");
		finished_rx
			.recv_timeout(Duration::from_secs(1))
			.expect("transition finishes")
			.expect("WAL transition succeeds");
		worker.join().expect("transition worker");
		let mode: String = Connection::open(&path)
			.expect("reopen database")
			.pragma_query_value(None, "journal_mode", |row| row.get(0))
			.expect("journal mode");
		assert_eq!(mode, "wal");
	}

	#[test]
	fn journal_transition_contention_is_bounded() {
		let directory = tempfile::tempdir().expect("directory");
		let path = directory.path().join("store.sqlite");
		let reader = Connection::open(&path).expect("reader");
		reader
			.execute_batch("CREATE TABLE fixture (id INTEGER); BEGIN; SELECT * FROM fixture;")
			.expect("hold read lease");
		let peer = Connection::open(&path).expect("peer");
		peer
			.busy_timeout(Duration::ZERO)
			.expect("immediate SQLite busy errors");
		let timeout = Duration::from_millis(30);
		let started = Instant::now();
		let error = configure_journal(&peer, timeout).expect_err("contention expires");
		assert_eq!(error.sqlite_error_code(), Some(ErrorCode::DatabaseBusy));
		assert!(started.elapsed() >= timeout);
		assert!(started.elapsed() < Duration::from_secs(1));
	}
}
