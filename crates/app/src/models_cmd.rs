//! Model catalog commands with inference-routed runtime discovery refresh.

use std::{
	collections::{BTreeMap, BTreeSet},
	fs,
	path::Path,
	sync::Arc,
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use miette::{IntoDiagnostic as _, miette};
use omp_ai::{
	Client, Registry,
	call::{CallMeta, DiscoveryRequest, Target},
	discovery::{DiscoveryCacheKey, DiscoveryStore, ProviderDiscoveryState, ProviderLifecycle},
	id::RequestId,
	receipt::ExecutionBudget,
	router,
};
use omp_catalog::{DiscoveredModel, ModelSpec, ProviderId, snapshot::Catalog};
use omp_core::Str;

use crate::cli::{LaunchExtensions, ModelRole, ModelsArgs, ModelsCommand};

const DISCOVERY_CACHE_TTL: Duration = Duration::from_secs(2 * 60 * 60);

/// The cache identity a provider's listing is kept under: scoped to the
/// account the listing was fetched for when one is pinned, since a backend
/// tailors what it offers (models, processing tiers) to the subscription.
pub(crate) fn provider_cache_key(
	provider: &ProviderId<str>,
	principal: Option<&str>,
) -> DiscoveryCacheKey {
	match principal {
		Some(principal) => DiscoveryCacheKey::credential(provider.to_owned(), principal),
		None => DiscoveryCacheKey::provider(provider.to_owned()),
	}
}

/// Returns a provider's current account-verified model selectors from the
/// discovery cache. Embedded catalog rows are intentionally excluded.
pub(crate) fn fresh_provider_models(
	data_dir: &Path,
	provider: &ProviderId<str>,
	principal: Option<&str>,
) -> miette::Result<Option<Vec<String>>> {
	let path = data_dir.join("models.db");
	if !path.exists() {
		return Ok(None);
	}
	let store = DiscoveryStore::open(&path).into_diagnostic()?;
	let Some(cached) = store
		.load_fresh(&provider_cache_key(provider, principal), discovery_now_ms()?)
		.into_diagnostic()?
	else {
		return Ok(None);
	};
	// Cached rows outlive the binary that decoded them. Older Anthropic
	// bootstrap decoders omitted the endpoint's implicit Chat capability, and
	// older Codex decoders left the route's tool support unknown. Treat those
	// rows as stale so the current decoder replaces them before a caller
	// exposes a selector that planning cannot use.
	let stale = match provider.as_str() {
		"anthropic" => cached.rows.iter().any(|row| {
			!discovered_model_supports_chat(row) || row.wire_model.as_str().ends_with("[1m]")
		}),
		"openai-codex" => cached.rows.iter().any(|row| {
			row.declared_capabilities
				.as_ref()
				.and_then(|capabilities| capabilities.chat.as_ref())
				.is_none_or(|chat| chat.tools.is_unknown())
		}),
		_ => false,
	};
	if stale {
		return Ok(None);
	}
	Ok(Some(discovered_selectors(&cached.rows)))
}

/// Discovers and atomically caches the models advertised by a provider's
/// authenticated discovery endpoint. Some providers return only additions to
/// their client-owned base catalog rather than a complete model list.
pub(crate) async fn refresh_provider(
	registry: &Registry,
	data_dir: &Path,
	provider: &ProviderId<str>,
	pins: Option<&omp_ai::AccountPins>,
) -> miette::Result<Vec<String>> {
	let routes = provider_discovery_routes(registry.catalog(), provider);
	if routes.is_empty() {
		return Err(miette!("provider `{provider}` does not expose model discovery"));
	}
	// A pinned account answers the listing for itself, and the rows are
	// cached for that account alone.
	let principal = pins.and_then(|pins| {
		pins
			.snapshot()
			.get(provider)
			.map(|principal| Str::new(principal.as_str()))
	});

	let now_ms = discovery_now_ms()?;
	let mut rows = Vec::new();
	for route in routes {
		let planner = router::Router::new(registry.clone(), Duration::from_secs(30));
		let meta = CallMeta {
			id:             RequestId::from(format!("omp-model-refresh-{}", provider.as_str())),
			target:         Target::RouteService(route.clone()),
			deadline:       None,
			budget:         ExecutionBudget::default(),
			session:        None,
			debug_session:  None,
			response_hooks: Default::default(),
		};
		let mut cursor = None;
		loop {
			let mut client = Client::new(registry.service(), planner.clone(), meta.clone());
			if let Some(pins) = pins {
				client = client.with_account_pins(pins.clone());
			}
			let page = client
				.execute(DiscoveryRequest {
					provider:  Some(provider.to_owned()),
					route:     Some(route.clone()),
					cursor:    cursor.clone(),
					page_size: 500,
					operation: None,
				})
				.await
				.map_err(|error| miette!("{provider} model discovery failed: {error}"))?;
			rows.extend(
				page
					.models
					.iter()
					.filter_map(|model| discovered(model, provider, &route, now_ms)),
			);
			cursor = page.next_cursor;
			if cursor.is_none() {
				break;
			}
		}
	}
	if rows.is_empty() {
		return Err(miette!("{provider} model discovery returned no models"));
	}

	fs::create_dir_all(data_dir).into_diagnostic()?;
	DiscoveryStore::open(&data_dir.join("models.db"))
		.into_diagnostic()?
		.publish(
			&provider_cache_key(provider, principal.as_deref()),
			&rows,
			now_ms,
			DISCOVERY_CACHE_TTL,
		)
		.into_diagnostic()?;
	Ok(discovered_selectors(&rows))
}

fn provider_discovery_routes(
	catalog: &Catalog,
	provider: &ProviderId<str>,
) -> Vec<omp_catalog::RouteId> {
	let mut seen_specs = BTreeSet::new();
	catalog
		.routes()
		.iter()
		.filter(|route| route.provider == *provider)
		.filter_map(|route| {
			let discovery = route.discovery.as_ref()?;
			seen_specs
				.insert(discovery.clone())
				.then(|| route.id.clone())
		})
		.collect()
}

fn discovered_selectors(rows: &[DiscoveredModel]) -> Vec<String> {
	let mut selectors = rows
		.iter()
		.filter(|row| discovered_model_supports_chat(row))
		.map(|row| format!("{}/{}", row.provider, row.wire_model))
		.collect::<Vec<_>>();
	selectors.sort();
	selectors.dedup();
	selectors
}

fn discovered_model_supports_chat(row: &DiscoveredModel) -> bool {
	row.declared_operations
		.contains_kind(omp_catalog::OperationKind::Chat)
		|| row
			.declared_capabilities
			.as_ref()
			.is_some_and(|capabilities| {
				capabilities
					.operations
					.contains_kind(omp_catalog::OperationKind::Chat)
			})
}

fn discovery_now_ms() -> miette::Result<u64> {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.into_diagnostic()?
		.as_millis()
		.try_into()
		.map_err(|_| miette!("system clock exceeds discovery timestamp range"))
}

/// Runs a model catalog operation. Refresh travels through the same inference
/// routes and credentials used at call time, then atomically updates only the
/// runtime discovery cache.
pub async fn run(args: &ModelsArgs, extensions: &LaunchExtensions) -> miette::Result<()> {
	let data_dir = omp_core::dirs::data_dir(None).into_diagnostic()?;
	let catalog = composed_catalog(&data_dir, extensions).await?;
	match args.command.as_ref() {
		None => print_rows(&select(catalog.as_ref(), args.filter.as_deref(), args.role), args.json),
		Some(ModelsCommand::List { filter, json, role }) => {
			print_rows(&select(catalog.as_ref(), filter.as_deref(), *role), *json)
		},
		Some(ModelsCommand::Find { pattern, json }) => {
			print_rows(&select(catalog.as_ref(), Some(pattern), None), *json)
		},
		Some(ModelsCommand::Refresh) => refresh().await,
	}
}

async fn composed_catalog(
	data_dir: &Path,
	_launch: &LaunchExtensions,
) -> miette::Result<Arc<Catalog>> {
	omp_driver::registry::production_catalog(data_dir).into_diagnostic()
}

async fn refresh() -> miette::Result<()> {
	let data_dir = omp_core::dirs::data_dir(None).into_diagnostic()?;
	fs::create_dir_all(&data_dir).into_diagnostic()?;
	let credentials = omp_driver::registry::open_credential_store(data_dir.join("credentials.db"))
		.into_diagnostic()?;
	let registry = omp_driver::registry::production_registry(&data_dir, credentials)
		.await
		.into_diagnostic()?;
	let catalog = registry.catalog();
	let mut routes = BTreeMap::<ProviderId, Vec<_>>::new();
	for route in catalog
		.routes()
		.iter()
		.filter(|route| route.discovery.is_some())
	{
		routes
			.entry(route.provider.clone())
			.or_default()
			.push(route.id.clone());
	}
	let store = DiscoveryStore::open(&data_dir.join("models.db")).into_diagnostic()?;
	let now_ms = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.into_diagnostic()?
		.as_millis()
		.try_into()
		.map_err(|_| miette!("system clock exceeds discovery timestamp range"))?;
	store.prune_expired(now_ms).into_diagnostic()?;
	let loaded_config =
		omp_driver::discovery::models::load_or_import_legacy(&data_dir).into_diagnostic()?;
	let mut refreshed = refresh_local_providers(
		&store,
		catalog,
		loaded_config.as_ref().map(|loaded| &loaded.config),
		now_ms,
	)
	.await?;
	let mut failures = Vec::new();
	for (provider, provider_routes) in routes {
		let mut rows = Vec::new();
		for route in provider_routes {
			let planner = router::Router::new(registry.clone(), Duration::from_secs(30));
			let meta = CallMeta {
				id:             RequestId::from(format!("omp-model-refresh-{}", provider.as_str())),
				// Pin the route the page is requested on, as `refresh_provider`
				// does: a provider-level target lets the planner pick a sibling
				// route, and the answer then fails page validation.
				target:         Target::RouteService(route.clone()),
				deadline:       None,
				budget:         ExecutionBudget::default(),
				session:        None,
				debug_session:  None,
				response_hooks: Default::default(),
			};
			let mut cursor = None;
			loop {
				let page = Client::new(registry.service(), planner.clone(), meta.clone())
					.execute(DiscoveryRequest {
						provider:  Some(provider.clone()),
						route:     Some(route.clone()),
						cursor:    cursor.clone(),
						page_size: 500,
						operation: None,
					})
					.await;
				let page = match page {
					Ok(page) => page,
					Err(error) => {
						tracing::warn!(
							provider = %provider,
							route = %route,
							%error,
							"model discovery refresh failed"
						);
						failures.push(format!("{}: {error}", provider.as_str()));
						break;
					},
				};
				rows.extend(
					page
						.models
						.iter()
						.filter_map(|model| discovered(model, &provider, &route, now_ms)),
				);
				cursor = page.next_cursor;
				if cursor.is_none() {
					break;
				}
			}
		}
		if !rows.is_empty() {
			store
				.publish(
					&DiscoveryCacheKey::provider(provider.clone()),
					&rows,
					now_ms,
					DISCOVERY_CACHE_TTL,
				)
				.into_diagnostic()?;
			refreshed = refreshed.saturating_add(rows.len());
		}
	}
	for failure in failures {
		eprintln!("warning: discovery refresh failed for {failure}");
	}
	println!("refreshed {refreshed} runtime model row(s); configured catalog models remain visible");
	Ok(())
}

async fn refresh_local_providers(
	store: &DiscoveryStore,
	catalog: &Catalog,
	config: Option<&omp_driver::discovery::models::ModelsConfig>,
	now_ms: u64,
) -> miette::Result<usize> {
	let probes =
		omp_driver::discovery::models::discovery_probes(config, catalog).into_diagnostic()?;
	let http = omp_envd::model_discovery::ModelDiscoveryHttpHost::new();
	let mut refreshed = 0_usize;
	for probe in probes {
		let provider = probe.provider.clone();
		let key = DiscoveryCacheKey::endpoint(provider.clone(), &probe.endpoint);
		store
			.set_lifecycle(&ProviderLifecycle {
				provider:       provider.clone(),
				cache_scope:    key.credential_scope.clone(),
				state:          ProviderDiscoveryState::Probing,
				error_code:     None,
				observed_at_ms: now_ms,
				retry_at_ms:    None,
			})
			.into_diagnostic()?;
		match probe
			.probe(&http, tokio_util::sync::CancellationToken::new())
			.await
		{
			Ok(mut rows) => {
				omp_driver::discovery::models::apply_runtime_discovery_overrides(&probe, &mut rows);
				for row in &mut rows {
					row.observed_at_ms = Some(now_ms);
				}
				store
					.publish(&key, &rows, now_ms, DISCOVERY_CACHE_TTL)
					.into_diagnostic()?;
				refreshed = refreshed.saturating_add(rows.len());
			},
			Err(error) => {
				let error_code: &'static str = error.into();
				store
					.set_lifecycle(&ProviderLifecycle {
						provider:       provider.clone(),
						cache_scope:    key.credential_scope.clone(),
						state:          ProviderDiscoveryState::Failed,
						error_code:     Some(Str::new_static(error_code)),
						observed_at_ms: now_ms,
						retry_at_ms:    Some(now_ms.saturating_add(5 * 60 * 1000)),
					})
					.into_diagnostic()?;
				eprintln!("warning: local model discovery failed for {}: {error}", provider.as_str());
			},
		}
	}
	Ok(refreshed)
}

fn discovered(
	model: &ModelSpec,
	provider: &ProviderId<str>,
	route: &omp_catalog::RouteId<str>,
	now_ms: u64,
) -> Option<DiscoveredModel> {
	// A bundled model comes back from the mixed projector as its own spec,
	// served on its per-model route rather than the discovery route. It is
	// still worth a row: the live listing is the only source of what this
	// account is offered on it (the Codex "Fast" tier, say), and hydration
	// applies only that to a bundled key.
	let wire_model = model
		.wire_ids
		.iter()
		.find_map(|(candidate, wire)| (candidate == route).then(|| wire.clone()))
		.or_else(|| {
			model.wire_ids.iter().find_map(|(candidate, wire)| {
				(candidate.as_str() != route.as_str()).then(|| wire.clone())
			})
		})?;
	let mut declared_operations = model.capabilities.operations;
	// The Claude subscription bootstrap endpoint is itself an authoritative
	// list of chat-selectable models. The mixed discovery projector may replace
	// a compact response row with conservative catalog metadata, so retain this
	// endpoint-level fact when materializing the durable cache row.
	if provider.as_str() == "anthropic" {
		declared_operations.insert_kind(omp_catalog::OperationKind::Chat);
	}
	Some(DiscoveredModel {
		provider: provider.to_owned(),
		route: route.to_owned(),
		wire_model,
		aliases: Box::new([]),
		display_name: Some(model.display_name.clone()),
		declared_class: Some(model.class.clone()),
		declared_operations,
		declared_capabilities: Some(model.capabilities.clone()),
		declared_limits: Some(model.limits.clone()),
		declared_pricing: Box::new([]),
		extended_context_mode: None,
		availability: Some(model.availability.clone()),
		source: Str::new_static("runtime-inference-discovery"),
		observed_at_ms: Some(now_ms),
		updated_at_ms: None,
		deprecated: None,
	})
}

fn select<'a>(
	catalog: &'a Catalog,
	filter: Option<&str>,
	role: Option<ModelRole>,
) -> Vec<&'a ModelSpec> {
	let needle = filter.unwrap_or_default().to_ascii_lowercase();
	let mut rows = catalog
		.models()
		.iter()
		.filter(|model| {
			needle.is_empty()
				|| model.key.as_str().to_ascii_lowercase().contains(&needle)
				|| model
					.display_name
					.as_str()
					.to_ascii_lowercase()
					.contains(&needle)
				|| model
					.routes
					.iter()
					.filter_map(|id| catalog.routes().iter().find(|route| route.id == *id))
					.any(|route| {
						route
							.provider
							.as_str()
							.to_ascii_lowercase()
							.contains(&needle)
					})
		})
		.collect::<Vec<_>>();
	if let Some(role) = role {
		let index = match role {
			ModelRole::Primary => 0,
			ModelRole::Smol => 1,
			ModelRole::Slow => 2,
			ModelRole::Plan => 3,
		};
		rows = rows
			.get(index % rows.len().max(1))
			.into_iter()
			.copied()
			.collect();
	}
	rows
}

fn print_rows(rows: &[&ModelSpec], json: bool) -> miette::Result<()> {
	if json {
		println!("{}", serde_json::to_string_pretty(rows).into_diagnostic()?);
		return Ok(());
	}
	for model in rows {
		println!(
			"{}\t{}\tcontext={}\tmax_output={}\tthinking={}",
			model.key,
			model.display_name,
			model
				.limits
				.context_window
				.map_or_else(|| "?".into(), |value| value.to_string()),
			model
				.limits
				.maximum_output_tokens
				.map_or_else(|| "?".into(), |value| value.to_string()),
			model.thinking.as_ref().map_or("no", |_| "yes")
		);
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn anthropic_cache_projection_retains_endpoint_chat_capability() {
		let catalog = Catalog::embedded();
		let provider = ProviderId::from("anthropic");
		let route = omp_catalog::RouteId::from("anthropic/primary");
		let mut model = catalog
			.model(omp_catalog::ModelKey::from_ref("anthropic/claude-fable-5"))
			.expect("bundled Anthropic model")
			.clone();
		model.capabilities = omp_catalog::unknown_capabilities();
		let row = discovered(&model, &provider, &route, 1).expect("primary route wire id");
		assert!(discovered_model_supports_chat(&row));
	}

	#[test]
	fn codex_cache_without_route_tool_evidence_is_refreshed() {
		let data_dir = tempfile::tempdir().expect("data dir");
		let catalog = Catalog::embedded();
		let provider = ProviderId::from("openai-codex");
		let route = omp_catalog::RouteId::from("openai-codex/primary");
		let mut model = catalog
			.model(omp_catalog::ModelKey::from_ref("openai-codex/gpt-5.6-sol"))
			.expect("bundled Codex model")
			.clone();
		model.capabilities.chat.as_mut().expect("chat").tools = omp_catalog::Availability::Unknown;
		let mut row = discovered(&model, &provider, &route, 1).expect("primary route wire id");
		let now_ms = discovery_now_ms().expect("clock");
		DiscoveryStore::open(&data_dir.path().join("models.db"))
			.expect("store")
			.publish(
				&DiscoveryCacheKey::provider(provider.clone()),
				std::slice::from_ref(&row),
				now_ms,
				DISCOVERY_CACHE_TTL,
			)
			.expect("publish");

		assert_eq!(
			fresh_provider_models(data_dir.path(), &provider, None).expect("cache read"),
			None
		);

		row.declared_capabilities
			.as_mut()
			.and_then(|capabilities| capabilities.chat.as_mut())
			.expect("chat")
			.tools = omp_catalog::Availability::Native(omp_catalog::ToolCapabilities {
			features:      omp_catalog::ToolFeatureBits::empty(),
			maximum_tools: None,
		});
		DiscoveryStore::open(&data_dir.path().join("models.db"))
			.expect("store")
			.publish(
				&DiscoveryCacheKey::provider(provider.clone()),
				std::slice::from_ref(&row),
				now_ms,
				DISCOVERY_CACHE_TTL,
			)
			.expect("publish");
		assert!(
			fresh_provider_models(data_dir.path(), &provider, None)
				.expect("cache read")
				.is_some()
		);
	}

	#[test]
	fn provider_discovery_routes_deduplicate_shared_specs() {
		let catalog = Catalog::embedded();
		let routes = provider_discovery_routes(catalog, ProviderId::from_ref("anthropic"));
		assert_eq!(routes, [omp_catalog::RouteId::from("anthropic/primary")]);
	}

	#[test]
	fn finds_a_model_by_case_insensitive_key_or_display_name() {
		let catalog = Catalog::embedded();
		let first = catalog.models().first().expect("embedded model");
		let prefix = &first.key.as_str()[..3.min(first.key.as_str().len())];
		assert!(select(catalog, Some(&prefix.to_ascii_uppercase()), None).contains(&first));
	}
}
