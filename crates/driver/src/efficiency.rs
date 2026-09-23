//! Bounded defaults for native hosts that explicitly enable efficient
//! inference.
use omp_con::Ctx;

use crate::headless::kernel::PromptOverrides;
/// Whether the native host opted into the finite task-tree policy.
pub fn enabled() -> bool {
	std::env::var_os("OMP_TOKEN_EFFICIENT").is_some_and(|v| v == "1")
}
/// Applies host policy after persisted session settings have been restored.
pub fn apply(ctx: &Ctx, prompt: &mut PromptOverrides) -> Result<(), omp_con::ConError> {
	use crate::subagent::settings::*;
	SV_TASK_MAX_CONCURRENCY.set(ctx, 2)?;
	SV_TASK_MAX_RECURSION_DEPTH.set(ctx, 1)?;
	SV_TASK_SOFT_REQUEST_BUDGET.set(ctx, 40)?;
	SV_TASK_SOFT_REQUEST_BUDGET_NOTICE.set(ctx, true)?;
	omp_ai::settings::AI_COMPACTION_ENABLED.set(ctx, true)?;
	omp_ai::settings::AI_COMPACTION_MID_TURN_ENABLED.set(ctx, true)?;
	omp_ai::settings::AI_COMPACTION_THRESHOLD_TOKENS.set(ctx, 48_000)?;
	omp_ai::settings::AI_COMPACTION_KEEP_RECENT_TOKENS.set(ctx, 8_000)?;
	const GUIDANCE: &str =
		"Token efficiency: The task shares 160 inference calls and 1,000,000 uncached input tokens \
		 across all agents; each worker has 40 calls plus a final handoff. Batch independent reads \
		 and shell work. Keep tool output and coordination concise. Delegate only concrete \
		 independent work; avoid overlapping assignments. Workers cannot delegate further. Read the \
		 actual working directory before editing. Owned background jobs and workers deliver results \
		 automatically: do other work or end your reply and completion will wake you. Do not poll \
		 with sleep, process checks, or repeated log reads. Use hub wait only for explicit \
		 coordination, with timeoutMs 30000 or longer. Consolidate peer messages and drain the \
		 inbox in batches. Do not repeatedly read unchanged files or poll statuses. Respect the \
		 shared task allowance; give a concise handoff when work cannot fit.";
	let existing = prompt.append_prompt.take().unwrap_or_default();
	prompt.append_prompt = Some(omp_core::sf!("{existing}\n\n{GUIDANCE}"));
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;
	#[test]
	fn old_session_settings_cannot_restore_unbounded_fanout() {
		use crate::subagent::settings::*;
		let ctx = Ctx::new();
		SV_TASK_MAX_CONCURRENCY.set(&ctx, 32).unwrap();
		SV_TASK_MAX_RECURSION_DEPTH.set(&ctx, 2).unwrap();
		let mut prompt = PromptOverrides::default();
		apply(&ctx, &mut prompt).unwrap();
		assert_eq!(SV_TASK_MAX_CONCURRENCY.get(&ctx), 2);
		assert_eq!(SV_TASK_MAX_RECURSION_DEPTH.get(&ctx), 1);
		assert_eq!(omp_ai::settings::AI_COMPACTION_THRESHOLD_TOKENS.get(&ctx), 48_000);
	}
}
