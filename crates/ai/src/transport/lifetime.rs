//! The bounds on one transport attempt's lifetime.
//!
//! An attempt's `timeout` bounds only its *start*: opening the request body,
//! the response headers, and the first frame. Once a response has begun the
//! provider is working, and a reasoning model is silent for as long as it
//! reasons, so the start bound no longer applies. From then on a wait may end
//! only at the route's idle bound between frames, re-armed on every frame, or
//! at the caller's limit on the whole attempt; an attempt with neither is
//! owned by its caller through cancellation.

use tokio::time::{Instant, sleep_until};

use crate::codec::TransportAttempt;

/// Which bound ends a wait on a response that has started.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum StartedBound {
	/// The route's idle bound: no frame for `idle_timeout` since the last.
	Idle,
	/// The caller's limit on the whole attempt, from its start.
	Elapsed,
}

/// The instant a wait beginning now on a started response ends, and which
/// bound ends it; `None` when the attempt has no bound past its start.
pub(super) fn started_bound(
	attempt: &TransportAttempt,
	started: Instant,
) -> Option<(Instant, StartedBound)> {
	let idle = attempt
		.idle_timeout
		.map(|idle| (Instant::now() + idle, StartedBound::Idle));
	let elapsed = attempt
		.elapsed_limit
		.map(|limit| (started + limit, StartedBound::Elapsed));
	match (idle, elapsed) {
		(Some(idle), Some(elapsed)) => Some(if elapsed.0 <= idle.0 { elapsed } else { idle }),
		(idle, elapsed) => idle.or(elapsed),
	}
}

/// Which bound ends the wait for the first visible event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FirstEventBound {
	/// The route's first-event watchdog.
	Watchdog,
	/// The caller's limit on the whole attempt, from its start.
	Elapsed,
}

/// The instant the wait for the first *visible* event ends: the route's
/// watchdog when it has one, and the caller's limit — never the start bound,
/// which the first frame satisfied. Silence between frames meanwhile is the
/// body's own wait, bounded by [`started_bound`].
pub(super) fn first_event_bound(
	attempt: &TransportAttempt,
	started: Instant,
) -> Option<(Instant, FirstEventBound)> {
	let watchdog = attempt
		.first_event_timeout
		.map(|timeout| (Instant::now() + timeout, FirstEventBound::Watchdog));
	let elapsed = attempt
		.elapsed_limit
		.map(|limit| (started + limit, FirstEventBound::Elapsed));
	match (watchdog, elapsed) {
		(Some(watchdog), Some(elapsed)) => Some(if elapsed.0 <= watchdog.0 {
			elapsed
		} else {
			watchdog
		}),
		(watchdog, elapsed) => watchdog.or(elapsed),
	}
}

/// Sleeps until `deadline`, or forever when there is none: a wait its caller
/// ends only through cancellation.
pub(super) async fn wait_until(deadline: Option<Instant>) {
	match deadline {
		Some(deadline) => sleep_until(deadline).await,
		None => std::future::pending().await,
	}
}
