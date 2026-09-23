//! Shared inference admission and usage for one user-requested task tree.
use std::sync::{
	Arc,
	atomic::{AtomicU64, Ordering},
};

use omp_ai::{ChatEvent, ChatStream, Error, ErrorDetail, ErrorKind, ExecutionReceipt};
use omp_core::Str;
use tokio_util::sync::CancellationToken;

/// One allowance shared by the parent, children, and auxiliary inference.
/// A new explicit user request gets a new allowance; descendants cannot reset
/// it.
#[derive(Debug)]
pub struct TreeBudget {
	max_requests:        u64,
	max_uncached_tokens: u64,
	requests:            AtomicU64,
	uncached_tokens:     AtomicU64,
	cancellation:        CancellationToken,
	failure:             parking_lot::Mutex<Option<Error>>,
}
impl TreeBudget {
	/// Creates a finite shared allowance. Zero admits no requests.
	pub fn new(max_requests: u64, max_uncached_tokens: u64) -> Self {
		Self {
			max_requests,
			max_uncached_tokens,
			requests: AtomicU64::new(0),
			uncached_tokens: AtomicU64::new(0),
			cancellation: CancellationToken::new(),
			failure: parking_lot::Mutex::new(None),
		}
	}

	/// Stops the entire task tree, including requests already in flight.
	pub fn stop(&self) {
		self.cancellation.cancel();
	}

	/// Cancels descendants even if their owning future is aborted or unwinds.
	pub fn cancel_on_drop(&self) -> tokio_util::sync::DropGuard {
		self.cancellation.clone().drop_guard()
	}

	/// Records the first provider or allowance failure, then cancels
	/// descendants.
	pub fn fail(&self, error: &Error) {
		self.failure.lock().get_or_insert_with(|| error.clone());
		self.stop();
	}

	/// The original failure, retained so cancellation does not hide its reason.
	pub fn failure(&self) -> Option<Error> {
		self.failure.lock().clone()
	}

	/// Whether an ancestor failed or the allowance was exhausted.
	pub fn is_stopped(&self) -> bool {
		self.cancellation.is_cancelled()
	}

	/// Resolves when all descendants must stop.
	pub async fn cancelled(&self) {
		self.cancellation.cancelled().await;
	}

	/// Atomically reserves a request before contacting a provider.
	pub fn admit(&self) -> Result<(), Error> {
		let used = self.uncached_tokens.load(Ordering::Acquire);
		if self.is_stopped() {
			return Err(self.failure().unwrap_or_else(|| {
				Error::new(
					ErrorKind::Cancelled,
					omp_ai::ErrorPhase::Planning,
					omp_ai::RetryAction::Never,
					ExecutionReceipt::default(),
				)
			}));
		}
		if used >= self.max_uncached_tokens {
			return Err(self.exhausted(
				"task-tree uncached input tokens",
				self.max_uncached_tokens,
				used,
			));
		}
		self
			.requests
			.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
				(n < self.max_requests).then(|| n + 1)
			})
			.map(|_| ())
			.map_err(|used| self.exhausted("task-tree model requests", self.max_requests, used))
	}

	fn exhausted(&self, dimension: &'static str, limit: u64, observed: u64) -> Error {
		let error = Error::planning(
			ErrorKind::BudgetExhausted,
			ErrorDetail::Budget {
				dimension: Str::new_static(dimension),
				limit:     u128::from(limit),
				observed:  u128::from(observed),
			},
			ExecutionReceipt::default(),
		);
		self.fail(&error);
		error
	}

	/// Charges the provider's uncached input once per completed response.
	/// Cached input is a separate dimension and is never double counted.
	pub fn record(&self, tokens: u64) {
		let _ = self
			.uncached_tokens
			.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| Some(n.saturating_add(tokens)));
	}

	/// Preserves the provider stream and its control channel while observing
	/// accounting.
	pub fn observe(self: Arc<Self>, stream: ChatStream) -> ChatStream {
		stream.inspect(move |event| match event {
			Ok(ChatEvent::Completed(completion)) => self.record(completion.usage.input_tokens),
			Err(error)
				if matches!(
					error.kind,
					ErrorKind::RateLimited | ErrorKind::Authentication | ErrorKind::BudgetExhausted
				) =>
			{
				self.fail(error)
			},
			_ => {},
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	#[tokio::test]
	async fn dropping_the_parent_task_cancels_its_tree() {
		let budget = Arc::new(TreeBudget::new(20, 1000));
		let child = budget.clone();
		let (started, ready) = tokio::sync::oneshot::channel();
		let parent = tokio::spawn(async move {
			let _lifetime = child.cancel_on_drop();
			started.send(()).unwrap();
			std::future::pending::<()>().await;
		});
		ready.await.unwrap();
		parent.abort();
		assert!(parent.await.unwrap_err().is_cancelled());
		assert!(budget.is_stopped());
		assert_eq!(budget.admit().unwrap_err().kind, ErrorKind::Cancelled);
	}

	#[tokio::test]
	async fn provider_failure_cancels_siblings_and_preserves_the_actual_error() {
		use futures::StreamExt;
		let budget = Arc::new(TreeBudget::new(20, 1000));
		let error = Error::new(
			ErrorKind::RateLimited,
			omp_ai::ErrorPhase::Planning,
			omp_ai::RetryAction::Never,
			ExecutionReceipt::default(),
		);
		let mut stream = budget
			.clone()
			.observe(ChatStream::ordinary(Box::pin(futures::stream::iter([Err(error)]))));
		assert!(stream.next().await.unwrap().is_err());
		tokio::time::timeout(std::time::Duration::from_millis(100), budget.cancelled())
			.await
			.unwrap();
		assert_eq!(budget.admit().unwrap_err().kind, ErrorKind::RateLimited);
		assert_eq!(budget.failure().unwrap().kind, ErrorKind::RateLimited);
	}

	#[test]
	fn concurrent_children_cannot_each_spend_the_parent_allowance() {
		let budget = Arc::new(TreeBudget::new(20, 1_000));
		let admitted = Arc::new(AtomicU64::new(0));
		std::thread::scope(|scope| {
			for _ in 0..8 {
				let budget = budget.clone();
				let admitted = admitted.clone();
				scope.spawn(move || {
					for _ in 0..10 {
						if budget.admit().is_ok() {
							admitted.fetch_add(1, Ordering::Relaxed);
						}
					}
				});
			}
		});
		assert_eq!(admitted.load(Ordering::Relaxed), 20);
		assert!(budget.is_stopped());
	}
	#[test]
	fn usage_and_parent_failure_stop_descendants_but_a_new_request_is_independent() {
		let budget = TreeBudget::new(100, 1000);
		budget.admit().unwrap();
		budget.record(800);
		budget.admit().unwrap();
		budget.record(200);
		assert!(budget.admit().is_err());
		let next = TreeBudget::new(100, 1000);
		assert!(next.admit().is_ok());
		next.stop();
		assert!(next.admit().is_err());
	}
}
