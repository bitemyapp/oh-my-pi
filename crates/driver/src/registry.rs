//! Production inference and credential-service composition.

use std::{
	collections::{BTreeMap, BTreeSet},
	env,
	env::consts,
	fs,
	future::Future,
	io,
	io::IsTerminal as _,
	path::{Path, PathBuf},
	sync::{Arc, LazyLock},
	time::Duration,
};

#[cfg(target_os = "macos")]
use omp_ai::auth::FallbackKeySource;
#[cfg(feature = "local-applefm")]
use omp_ai::provider::builtin::LocalRouteBackend;
#[cfg(feature = "local-applefm")]
use omp_ai::receipt::ReasonId;
use omp_ai::{
	AccountPins, Registry,
	account::{
		AccountPool, AccountStateStore, AccountStateStoreError, RefreshCoordinator, RefreshPolicy,
	},
	auth::{
		AlibabaTokenPlanLoginEngine, AlibabaTokenPlanShaper, AuthControlHandle, AuthLoginEngine,
		AuthManager, AuthManagerBuildError, AwsCredentialSource, CredentialAcquisitionLoginEngine,
		CredentialAcquisitionLoginEngineError, CredentialAffinityResolver, CredentialBroker,
		CredentialBrokerEngines, CredentialShaperRegistry, CredentialStore, FileCredentialKeySource,
		FileKeyError, GithubCopilotShaper, KeyError, KeySource, OAuthCustomDispatcher,
		OAuthLoginEngine, OAuthLoginEngineError, OsCredentialKeySource, ProviderShaper,
		RefreshingCredentialSource, SecretLoginEngine, SecretLoginEngineError, StoreError,
		StoredOAuthRefreshEngine, SystemOAuthClock, SystemOAuthHttpClient, UnavailableKeySource,
		oauth::OAuthCustomDispatchError, prune_credentialless_accounts,
	},
	call::AuthMethod,
	codec::google_cca::{
		AntigravityFingerprint, AntigravityPolicy, CcaHeaders, DEFAULT_ANTIGRAVITY_ARCH,
		DEFAULT_ANTIGRAVITY_CL, DEFAULT_ANTIGRAVITY_OS, DEFAULT_ANTIGRAVITY_VERSION,
	},
	id::AccountId,
	layer::{admission::AdmissionController, stack::BuiltinConfig},
	operation::usage::{
		ConsoleUsageFetcher, ConsoleUsageManager, UsageFetcherRegistry,
		alibaba_token_plan::AlibabaTokenPlanUsageFetcher,
		claude::ClaudeUsageFetcher,
		cursor::CursorUsageFetcher,
		gemini::GeminiUsageFetcher,
		github_copilot::GithubCopilotUsageFetcher,
		google_antigravity::GoogleAntigravityUsageFetcher,
		kimi::KimiUsageFetcher,
		minimax_code::MiniMaxCodeUsageFetcher,
		ollama::OllamaUsageFetcher,
		openai_codex::{CodexRedemption, CodexRedemptionReason, OpenAiCodexUsageFetcher},
		opencode_go::OpenCodeGoUsageFetcher,
		synthetic::SyntheticUsageFetcher,
		umans::UmansUsageFetcher,
		xai_oauth::XaiOauthUsageFetcher,
		zai::ZaiUsageFetcher,
	},
	provider::builtin::{
		AuthApplicationConfig, AzureEndpointConfig, GoogleCcaConfig, ProductionDependencies,
		discover_antigravity_version,
	},
	session::{ConversationError, ConversationSessionPlanner},
	transport::{http::HttpTransport, websocket_transport::WebSocketTransport},
};
use omp_catalog::{
	CatalogOverlay, ContextStrategy, DiscoveryDefaults, DiscoveryNormalizer, OverlaySource,
	OverlayStack, Pricing, ProvenanceKind, ProvenanceSource, UnsafeTrustScope,
	provider::AuthSpecKind, snapshot,
};
use omp_core::{Hash32, SecretString, Str, sf};
use omp_envd::browser_fetch::BrowserFetchAdapter;
use omp_serve::inference::InferenceRpc;
use tokio::time;

use crate::{auth_backend, auth_backend::GithubCredentialAuthority};

/// Credential database encryption-key source selected at startup.
#[derive(
	Clone,
	Copy,
	Debug,
	Default,
	Eq,
	PartialEq,
	serde::Deserialize,
	serde::Serialize,
	strum::Display,
	strum::EnumString,
	strum::IntoStaticStr,
	strum::VariantNames,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case", ascii_case_insensitive)]
pub enum CredentialKeySourceSetting {
	/// Select the local file only for an interactive owner.
	#[default]
	Auto,
	/// Refuse durable secret reads and writes.
	Unavailable,
	/// Use an owner-only file beside the credential database.
	LocalFile,
	/// Use the operating-system credential service.
	OsKeychain,
}

omp_con::con_enum!(CredentialKeySourceSetting);

omp_con::var! {
	/// Encryption-key source for the durable credential database.
	pub static SV_CREDENTIAL_KEY_SOURCE = sv_credential_key_source: CredentialKeySourceSetting {
		default: CredentialKeySourceSetting::Auto,
		flags: archive,
	};
}

const KEY_SOURCE_ENV: &str = "OMP_LLM_KEY_SOURCE";
const KEYCHAIN_SERVICE: &str = "dev.omp.llm";
const KEYCHAIN_ACCOUNT: &str = "credential-store-master";
const ANTIGRAVITY_VERSION_ENV: &str = "OMP_ANTIGRAVITY_VERSION";
const ANTIGRAVITY_CL_ENV: &str = "OMP_ANTIGRAVITY_CL";
const ANTIGRAVITY_OS_ENV: &str = "OMP_ANTIGRAVITY_OS";
const ANTIGRAVITY_ARCH_ENV: &str = "OMP_ANTIGRAVITY_ARCH";
const ANTIGRAVITY_VERSION_CACHE_FILE: &str = "antigravity-version";
const ANTIGRAVITY_VERSION_FETCH_TIMEOUT: Duration = Duration::from_secs(5);
const MODEL_DISCOVERY_CACHE_TTL: Duration = Duration::from_secs(2 * 60 * 60);
const AZURE_BASE_URL_ENV: &str = "OMP_AZURE_OPENAI_BASE_URL";
const AZURE_RESOURCE_NAME_ENV: &str = "OMP_AZURE_OPENAI_RESOURCE_NAME";
const AZURE_DEPLOYMENT_ENV: &str = "OMP_AZURE_OPENAI_DEPLOYMENT";
const AZURE_API_VERSION_ENV: &str = "OMP_AZURE_OPENAI_API_VERSION";

/// Production inference-registry or credential-state construction failure.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
	/// Durable state directory could not be prepared.
	#[error("could not prepare inference state directory")]
	PrepareState(#[source] io::Error),
	/// The checked-in catalog snapshot is invalid.
	#[error("embedded catalog snapshot is invalid")]
	Catalog(#[source] &'static omp_catalog::snapshot::SnapshotError),
	/// Native configuration or discovery cache could not be composed.
	#[error("live catalog composition failed")]
	CatalogComposition(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),
	/// Another process held the discovery-cache writer lease past its deadline.
	#[error("model discovery cache {path} is busy; retry startup after model refresh finishes")]
	DiscoveryCacheBusy {
		/// Cache whose writer lease was unavailable.
		path:   PathBuf,
		/// Typed cause retained without printing untrusted SQLite text.
		#[source]
		source: omp_ai::discovery::DiscoveryStoreError,
	},
	/// Discovery-cache SQLite access failed with a non-content diagnostic.
	#[error("could not read model discovery cache {path} (SQLite code {code})")]
	DiscoveryCacheSqlite {
		/// Cache that could not be read.
		path:   PathBuf,
		/// SQLite extended result code.
		code:   i32,
		/// Typed cause retained without printing untrusted SQLite text.
		#[source]
		source: omp_ai::discovery::DiscoveryStoreError,
	},
	/// A cached model row could not be decoded.
	#[error("could not decode model discovery cache {path}")]
	DiscoveryCacheDecode {
		/// Cache that could not be decoded.
		path:   PathBuf,
		/// Typed cause retained without printing model or credential contents.
		#[source]
		source: omp_ai::discovery::DiscoveryStoreError,
	},
	/// Registry construction or route service failed.
	#[error(transparent)]
	Inference(#[from] Box<omp_ai::Error>),
	/// Encrypted credential state could not be opened.
	#[error(transparent)]
	CredentialStore(#[from] StoreError),
	/// Credential encryption key provisioning failed.
	#[error(transparent)]
	CredentialKey(#[from] KeyError),
	/// Owner-only credential key file provisioning failed.
	#[error(transparent)]
	CredentialKeyFile(#[from] FileKeyError),
	/// Console policy could not be resolved.
	#[error(transparent)]
	Console(#[from] omp_con::ConError),
	/// Durable account state could not be opened.
	#[error(transparent)]
	AccountState(#[from] AccountStateStoreError),
	/// A static secret login engine was invalid.
	#[error(transparent)]
	SecretLogin(#[from] SecretLoginEngineError),
	/// A credential-acquisition engine was invalid.
	#[error(transparent)]
	CredentialAcquisitionLogin(#[from] CredentialAcquisitionLoginEngineError),
	/// An OAuth login engine was invalid.
	#[error(transparent)]
	OAuthLogin(#[from] OAuthLoginEngineError),
	/// A custom OAuth exchange handler could not be registered.
	#[error(transparent)]
	OAuthCustom(#[from] OAuthCustomDispatchError),
	/// Refresh coordination policy was invalid.
	#[error(transparent)]
	RefreshPolicy(#[from] omp_ai::account::RefreshPolicyError),
	/// Catalog authentication could not be assembled.
	#[error(transparent)]
	AuthManager(#[from] AuthManagerBuildError),
	/// Durable conversation state could not be opened.
	#[error(transparent)]
	Conversation(#[from] ConversationError),
}

impl From<omp_ai::Error> for RegistryError {
	fn from(error: omp_ai::Error) -> Self {
		Self::Inference(Box::new(error))
	}
}

/// Selection of the credential encryption-key source.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CredentialKeyMode {
	/// Fail closed without accessing persistent encryption-key material.
	#[default]
	Unavailable,
	/// Use an owner-only key file beside the credential database.
	LocalFile,
	/// Use the operating-system credential service after explicit opt-in.
	OsKeychain,
}

impl CredentialKeyMode {
	/// Selects the key source from an explicit environment override followed by
	/// the typed settings value. Malformed values fail closed; the `auto`
	/// default uses an owner-only local key file as the filesystem security
	/// boundary for interactive processes and fails closed for unattended
	/// ones.
	pub fn from_configuration(configured: CredentialKeySourceSetting) -> Self {
		let interactive = io::stdin().is_terminal() && io::stderr().is_terminal();
		Self::resolve(env::var(KEY_SOURCE_ENV).ok().as_deref(), configured, interactive)
	}

	fn resolve(
		explicit: Option<&str>,
		configured: CredentialKeySourceSetting,
		interactive: bool,
	) -> Self {
		let auto = if interactive {
			Self::LocalFile
		} else {
			Self::Unavailable
		};
		match explicit.map(str::trim) {
			Some("local-file") => Self::LocalFile,
			Some("os-keychain") => Self::OsKeychain,
			Some("auto") => auto,
			Some("unavailable") | Some(_) => Self::Unavailable,
			None => match configured {
				CredentialKeySourceSetting::Auto => auto,
				CredentialKeySourceSetting::Unavailable => Self::Unavailable,
				CredentialKeySourceSetting::LocalFile => Self::LocalFile,
				CredentialKeySourceSetting::OsKeychain => Self::OsKeychain,
			},
		}
	}
}

fn placeholder_affinity_key() -> &'static str {
	static KEY: LazyLock<String> = LazyLock::new(|| match omp_cache::secret_key::load_or_create() {
		Ok(key) => key,
		Err(error) => {
			tracing::warn!(%error, "could not persist credential-affinity key; using process-local identity");
			omp_core::Ulid::generate().to_string()
		},
	});
	KEY.as_str()
}

/// Opens the encrypted credential database using default console policy.
pub fn open_credential_store(
	database: impl AsRef<Path>,
) -> Result<Arc<CredentialStore>, RegistryError> {
	open_credential_store_from_con(database, &omp_con::Ctx::new())
}

/// Opens the encrypted credential database using the effective console policy.
pub fn open_credential_store_from_con(
	database: impl AsRef<Path>,
	ctx: &omp_con::Ctx,
) -> Result<Arc<CredentialStore>, RegistryError> {
	let configured = SV_CREDENTIAL_KEY_SOURCE.get(ctx);
	open_credential_store_with_mode(
		database.as_ref(),
		CredentialKeyMode::from_configuration(configured),
	)
}

fn open_credential_store_with_mode(
	database: &Path,
	mode: CredentialKeyMode,
) -> Result<Arc<CredentialStore>, RegistryError> {
	match mode {
		CredentialKeyMode::Unavailable => {
			open_credential_store_with_key_source(database, Arc::new(UnavailableKeySource))
		},
		CredentialKeyMode::LocalFile => open_local_file_credential_store(database.as_ref()),
		CredentialKeyMode::OsKeychain => {
			let key_source = OsCredentialKeySource::new(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT);
			if key_source.active_key().is_err() {
				key_source.rotate()?;
			}
			open_credential_store_with_key_source(database, Arc::new(key_source))
		},
	}
}

fn open_local_file_credential_store(
	database: &Path,
) -> Result<Arc<CredentialStore>, RegistryError> {
	let file = FileCredentialKeySource::open(database.with_extension("key"))?;
	#[cfg(target_os = "macos")]
	{
		// One-time clean cutover for credentials written by the old interactive
		// default. The fallback is consulted only for legacy key identifiers;
		// after this transaction all rows use the file key and later rebuilds
		// never contact Keychain. Denial aborts the transaction without losing
		// the existing encrypted records.
		let source = FallbackKeySource::new(
			file,
			OsCredentialKeySource::new(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT),
		);
		let store = open_credential_store_with_key_source(database, Arc::new(source))?;
		store.rotate_keys()?;
		Ok(store)
	}
	#[cfg(not(target_os = "macos"))]
	{
		open_credential_store_with_key_source(database, Arc::new(file))
	}
}

/// Opens encrypted credential state with an explicitly supplied non-secret key
/// source.
pub fn open_credential_store_with_key_source(
	database: impl AsRef<Path>,
	key_source: Arc<dyn KeySource>,
) -> Result<Arc<CredentialStore>, RegistryError> {
	Ok(Arc::new(CredentialStore::open(database.as_ref(), key_source)?))
}

/// Returns the immutable production catalog with configured and fresh
/// runtime-discovery layers materialized in explicit precedence order.
/// The account each provider's listing was fetched for in this process, set
/// by whoever pins accounts before the launch catalog is composed. Hydration
/// prefers that account's cached listing over the provider-wide one, because
/// what a subscription is offered is the account's fact.
static DISCOVERY_SCOPES: parking_lot::RwLock<BTreeMap<omp_catalog::ProviderId, Str>> =
	parking_lot::RwLock::new(BTreeMap::new());

/// Records which account's listing hydrates each provider in this process.
pub fn set_discovery_scopes(scopes: BTreeMap<omp_catalog::ProviderId, Str>) {
	*DISCOVERY_SCOPES.write() = scopes;
}

/// The account whose listing hydrates `provider` in this process, if any.
pub fn discovery_scope(provider: &omp_catalog::ProviderId<str>) -> Option<Str> {
	DISCOVERY_SCOPES.read().get(provider).cloned()
}

pub fn production_catalog(data_dir: &Path) -> Result<Arc<snapshot::Catalog>, RegistryError> {
	let bundled = snapshot::Catalog::try_embedded()
		.map_err(RegistryError::Catalog)?
		.clone();
	let loaded = if data_dir.exists() {
		crate::discovery::models::load_or_import_legacy(data_dir).map_err(catalog_composition)?
	} else {
		None
	};
	let user_overlay = loaded
		.as_ref()
		.map(|loaded| crate::discovery::models::lower_user_overlay(&loaded.config))
		.transpose()
		.map_err(catalog_composition)?;
	let configured = if let Some(overlay) = &user_overlay {
		bundled
			.with_overlay_stack(
				&OverlayStack::from_layers([(OverlaySource::UserConfig, overlay.clone())]),
				UnsafeTrustScope::ALL,
			)
			.map_err(catalog_composition)?
	} else {
		bundled
	};
	let cache_path = data_dir.join("models.db");
	if !cache_path.exists() {
		return Ok(Arc::new(configured));
	}
	let store = omp_ai::discovery::DiscoveryStore::open(&cache_path)
		.map_err(|source| catalog_cache_error(&cache_path, source))?;
	let now_ms = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.unwrap_or_default()
		.as_millis()
		.try_into()
		.unwrap_or(u64::MAX);
	let probes = crate::discovery::models::discovery_probes(
		loaded.as_ref().map(|loaded| &loaded.config),
		&configured,
	)
	.map_err(catalog_composition)?;
	let mut cache_keys = BTreeSet::new();
	for route in configured.routes() {
		if route.discovery.is_some() {
			cache_keys.insert(omp_ai::discovery::DiscoveryCacheKey::provider(route.provider.clone()));
		}
	}
	for probe in probes {
		cache_keys
			.insert(omp_ai::discovery::DiscoveryCacheKey::endpoint(probe.provider, &probe.endpoint));
	}
	let mut normalized = Vec::new();
	let scopes = DISCOVERY_SCOPES.read().clone();
	for key in cache_keys {
		// The pinned account's own listing wins over the provider-wide one.
		let scoped = key
			.credential_scope
			.is_none()
			.then(|| scopes.get(&key.provider))
			.flatten()
			.map(|principal| {
				omp_ai::discovery::DiscoveryCacheKey::credential(
					key.provider.clone(),
					principal.clone(),
				)
			});
		// Account listings remain usable offline after the refresh TTL. Never
		// substitute a different account's listing for an explicitly pinned one.
		let cache_key = scoped.as_ref().unwrap_or(&key);
		let cached = if matches!(key.provider.as_str(), "anthropic" | "openai-codex") {
			store.load_latest(cache_key)
		} else {
			store.load_fresh(cache_key, now_ms)
		}
		.map_err(|source| catalog_cache_error(&cache_path, source))?;
		let Some(cached) = cached else {
			continue;
		};
		let Some(provider) = configured.provider(&key.provider) else {
			continue;
		};
		let defaults = configured
			.discovery_defaults(&key.provider)
			.cloned()
			.unwrap_or_else(|| DiscoveryDefaults {
				wire_policy:          provider.wire_policy.clone(),
				extended_wire_policy: None,
				context:              ContextStrategy::Replay,
				thinking:             None,
				pricing:              Pricing::default(),
			});
		let explicit = loaded
			.as_ref()
			.and_then(|loaded| loaded.config.providers.get(key.provider.as_str()));
		// A cached row for a model the bundle already knows never replaces
		// it: the bundle keeps its pricing, thinking policy, and routes, and
		// the live listing contributes only which processing tiers this
		// account is offered (the Codex "Fast" tier). Such rows skip the
		// normalizer altogether so a shape it dislikes cannot cost the
		// catalog. Everything else is a genuinely new model.
		let mut fresh_rows = Vec::new();
		for row in &cached.rows {
			let bundled_key =
				omp_catalog::ModelKey::from(format!("{}/{}", key.provider, row.wire_model));
			match configured.model(&bundled_key) {
				Some(bundled) => normalized.extend(offered_tiers_patch(bundled, row, now_ms)),
				None => fresh_rows.push(row.clone()),
			}
		}
		for record in DiscoveryNormalizer::new(defaults)
			.normalize_batch(&fresh_rows)
			.map_err(catalog_composition)?
		{
			let explicitly_configured = explicit.is_some_and(|provider| {
				provider.models.contains_key(record.model.key.as_str())
					|| provider
						.model_overrides
						.contains_key(record.model.key.as_str())
			});
			if !explicitly_configured {
				// The normalizer keys discovered rows by their bare logical
				// model; every bundled key is `provider/model`, and selectors
				// resolve by key, so prefix the provider here.
				let mut record = record;
				if key.provider.as_str() == "anthropic" && record.model.thinking.is_none() {
					record.model.thinking = discovered_anthropic_thinking(&configured, &record.model);
				}
				if !record.model.key.as_str().contains('/') {
					let prefixed =
						omp_catalog::ModelKey::from(format!("{}/{}", key.provider, record.model.key));
					record.model.key = prefixed.clone();
					for alias in record.aliases.iter_mut() {
						alias.target = prefixed.clone();
					}
				}
				normalized.push(record.into_catalog_overlay());
			}
		}
	}
	if normalized.is_empty() {
		return Ok(Arc::new(configured));
	}
	let overlay = CatalogOverlay::combined(
		ProvenanceSource {
			kind:           ProvenanceKind::Discovered,
			origin:         Str::new_static("models.db"),
			revision:       None,
			confidence:     omp_catalog::EvidenceConfidence::Inferred,
			observed_at_ms: Some(now_ms),
		},
		normalized,
	);
	let catalog = configured
		.with_overlay_stack(
			&OverlayStack::from_layers([(OverlaySource::DiskCache, overlay)]),
			UnsafeTrustScope::NONE,
		)
		.map_err(catalog_composition)?;
	Ok(Arc::new(catalog))
}

/// Reuse a supported wire contract by reported capability, never by model ID.
/// Anthropic models must explicitly report adaptive thinking and effort
/// support. Unknown contracts remain unknown instead of silently borrowing
/// model specs.
fn discovered_anthropic_thinking(
	catalog: &snapshot::Catalog,
	model: &omp_catalog::ModelSpec,
) -> Option<omp_catalog::ThinkingPolicyId> {
	use omp_catalog::{Availability, ReasoningFeatureBits, ThinkingMode};
	let Availability::Native(reasoning) = &model.capabilities.chat.as_ref()?.reasoning else {
		return None;
	};
	if !reasoning.features.contains(ReasoningFeatureBits::EFFORT)
		|| !reasoning.features.contains(ReasoningFeatureBits::ADAPTIVE)
	{
		return None;
	}
	catalog.models().iter().find_map(|known| {
		if !known.key.as_str().starts_with("anthropic/") {
			return None;
		}
		let id = known.thinking.as_ref()?;
		let policy = catalog.thinking_policy(id)?;
		(policy.mode == ThinkingMode::AnthropicAdaptive
			&& policy
				.efforts
				.iter()
				.copied()
				.map(omp_catalog::ReasoningEffort::from)
				.eq(reasoning.efforts.iter().copied()))
		.then(|| id.clone())
	})
}

/// The one fact a live listing may change on a bundled model: the service
/// tiers this account is offered on it. `None` when the listing said nothing
/// or agrees with the bundle.
fn offered_tiers_patch(
	bundled: &omp_catalog::ModelSpec,
	row: &omp_catalog::DiscoveredModel,
	now_ms: u64,
) -> Option<CatalogOverlay> {
	let offered = row
		.declared_capabilities
		.as_ref()
		.and_then(|capabilities| capabilities.chat.as_ref())
		.map(|chat| &chat.service_tiers)
		.filter(|tiers| !matches!(tiers, omp_catalog::Availability::Unknown))?;
	let mut capabilities = bundled.capabilities.clone();
	let chat = capabilities.chat.as_mut()?;
	if &chat.service_tiers == offered {
		return None;
	}
	chat.service_tiers = offered.clone();
	let (provider, _) = bundled.key.as_str().split_once('/')?;
	let overlay = omp_catalog::ModelOverlay {
		selector: omp_catalog::ExactSelector {
			provider: omp_catalog::ProviderId::from(provider),
			model:    bundled.key.clone(),
		},
		added:    None,
		patch:    omp_catalog::ModelPatch { capabilities: Some(capabilities), ..Default::default() },
	};
	Some(
		omp_catalog::CatalogOverlayBuilder::new(ProvenanceSource {
			kind:           ProvenanceKind::Discovered,
			origin:         Str::new_static("models.db"),
			revision:       None,
			confidence:     omp_catalog::EvidenceConfidence::Declared,
			observed_at_ms: Some(now_ms),
		})
		.with_model(overlay)
		.build(),
	)
}

fn catalog_cache_error(
	path: &Path,
	source: omp_ai::discovery::DiscoveryStoreError,
) -> RegistryError {
	if let omp_ai::discovery::DiscoveryStoreError::Sqlite(error) = &source {
		if let Some(error) = error.sqlite_error() {
			if matches!(
				error.code,
				rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
			) {
				return RegistryError::DiscoveryCacheBusy { path: path.to_path_buf(), source };
			}
			return RegistryError::DiscoveryCacheSqlite {
				path: path.to_path_buf(),
				code: error.extended_code,
				source,
			};
		}
	}
	RegistryError::DiscoveryCacheDecode { path: path.to_path_buf(), source }
}

fn catalog_composition(source: impl std::error::Error + Send + Sync + 'static) -> RegistryError {
	RegistryError::CatalogComposition(Box::new(source))
}

async fn refresh_model_discovery_cache(
	data_dir: &Path,
	catalog: Arc<snapshot::Catalog>,
) -> Result<Arc<snapshot::Catalog>, RegistryError> {
	use omp_ai::discovery::{
		DiscoveryCacheKey, DiscoveryStore, DiscoveryStoreError, ProviderDiscoveryState,
		ProviderLifecycle,
	};

	let loaded =
		crate::discovery::models::load_or_import_legacy(data_dir).map_err(catalog_composition)?;
	let probes = crate::discovery::models::discovery_probes(
		loaded.as_ref().map(|loaded| &loaded.config),
		&catalog,
	)
	.map_err(catalog_composition)?;
	if probes.is_empty() {
		return Ok(catalog);
	}
	let cache_path = data_dir.join("models.db");
	let store = Arc::new(
		DiscoveryStore::open(&cache_path)
			.map_err(|source| catalog_cache_error(&cache_path, source))?,
	);
	let now_ms = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.unwrap_or_default()
		.as_millis()
		.try_into()
		.unwrap_or(u64::MAX);
	store
		.prune_expired(now_ms)
		.map_err(|source| catalog_cache_error(&cache_path, source))?;
	let http = omp_envd::model_discovery::ModelDiscoveryHttpHost::new();
	let mut pending = Vec::new();
	for probe in probes {
		let key = DiscoveryCacheKey::endpoint(probe.provider.clone(), &probe.endpoint);
		if store
			.load_fresh(&key, now_ms)
			.map_err(|source| catalog_cache_error(&cache_path, source))?
			.is_some()
		{
			continue;
		}
		if store
			.lifecycle(&key)
			.map_err(|source| catalog_cache_error(&cache_path, source))?
			.is_some_and(|state| {
				state.state == ProviderDiscoveryState::Failed
					&& state.retry_at_ms.is_some_and(|retry| retry > now_ms)
			}) {
			continue;
		}
		store
			.set_lifecycle(&ProviderLifecycle {
				provider:       probe.provider.clone(),
				cache_scope:    key.credential_scope.clone(),
				state:          ProviderDiscoveryState::Probing,
				error_code:     None,
				observed_at_ms: now_ms,
				retry_at_ms:    None,
			})
			.map_err(|source| catalog_cache_error(&cache_path, source))?;
		let store = Arc::clone(&store);
		let http = http.clone();
		pending.push(async move {
			let provider = probe.provider.clone();
			match probe
				.probe(&http, tokio_util::sync::CancellationToken::new())
				.await
			{
				Ok(mut rows) => {
					crate::discovery::models::apply_runtime_discovery_overrides(&probe, &mut rows);
					for row in &mut rows {
						row.observed_at_ms = Some(now_ms);
					}
					store.publish(&key, &rows, now_ms, MODEL_DISCOVERY_CACHE_TTL)?;
					Ok::<bool, DiscoveryStoreError>(true)
				},
				Err(error) => {
					let error_code: &'static str = error.into();
					tracing::debug!(
						provider = %provider,
						error_code,
						"bounded model discovery probe was unavailable"
					);
					store.set_lifecycle(&ProviderLifecycle {
						provider,
						cache_scope: key.credential_scope.clone(),
						state: ProviderDiscoveryState::Failed,
						error_code: Some(Str::new_static(error_code)),
						observed_at_ms: now_ms,
						retry_at_ms: Some(now_ms.saturating_add(5 * 60 * 1000)),
					})?;
					Ok::<bool, DiscoveryStoreError>(false)
				},
			}
		});
	}
	let mut changed = false;
	for result in futures::future::join_all(pending).await {
		changed |= result.map_err(|source| catalog_cache_error(&cache_path, source))?;
	}
	if changed {
		production_catalog(data_dir)
	} else {
		Ok(catalog)
	}
}

/// Builds the production inference registry over durable daemon state.
pub async fn production_registry(
	data_dir: &Path,
	credential_store: Arc<CredentialStore>,
) -> Result<Registry, RegistryError> {
	production_registry_from_con(data_dir, credential_store, &omp_con::Ctx::new()).await
}

/// Builds the production inference registry from the effective console context.
pub async fn production_registry_from_con(
	data_dir: &Path,
	credential_store: Arc<CredentialStore>,
	ctx: &omp_con::Ctx,
) -> Result<Registry, RegistryError> {
	production_assembly_for_session(
		data_dir,
		credential_store,
		None,
		UsageFetcherRegistry::default(),
		inference_settings(ctx, None),
	)
	.await
	.map(|(registry, ..)| registry)
}
/// Builds the production console-usage authority over the canonical
/// credential and account stores.
pub async fn production_usage_manager(
	data_dir: &Path,
) -> Result<ConsoleUsageManager, RegistryError> {
	let credential_store = open_credential_store(data_dir.join("credentials.db"))?;
	production_assembly(data_dir, credential_store)
		.await
		.map(|(_, _, _, _, _, usage, ..)| usage)
}

/// Redeems one saved Codex reset for an exact durable account.
pub async fn redeem_codex_reset(
	data_dir: &Path,
	account: &AccountId<str>,
) -> Result<Option<bool>, RegistryError> {
	let Some(service) = production_codex_redemption(data_dir)? else {
		return Ok(None);
	};
	Ok(Some(
		service
			.redeem_account(CodexRedemptionReason::Restore, account)
			.await,
	))
}

fn production_codex_redemption(data_dir: &Path) -> Result<Option<CodexRedemption>, RegistryError> {
	let catalog = snapshot::Catalog::try_embedded().map_err(RegistryError::Catalog)?;
	let credential_store = open_credential_store(data_dir.join("credentials.db"))?;
	let stored = Arc::new(auth_backend::combined_authority(credential_store));
	let credentials = CredentialBroker::system(catalog, CredentialBrokerEngines {
		stored: Some(stored),
		..CredentialBrokerEngines::default()
	})
	.map_err(|_| {
		RegistryError::Inference(Box::new(omp_ai::Error::planning(
			omp_ai::ErrorKind::InvalidRequest,
			omp_ai::ErrorDetail::target(sf!("catalog-credential-broker-invalid")),
			Default::default(),
		)))
	})?;
	let accounts = AccountPool::with_store(Arc::new(AccountStateStore::open(
		&data_dir.join("credentials.db"),
	)?))?;
	let http = Arc::new(SystemOAuthHttpClient::new());
	Ok(CodexRedemption::from_catalog(catalog, credentials, accounts, http))
}

/// Builds the production inference registry and exposes a clone of its one
/// authentication manager to the stdio RPC host.
pub async fn production_rpc_registry(
	data_dir: &Path,
	credential_store: Arc<CredentialStore>,
) -> Result<(Registry, AuthManager), RegistryError> {
	production_rpc_registry_from_con(data_dir, credential_store, &omp_con::Ctx::new(), None).await
}

/// Builds the RPC registry from the effective console context.
pub async fn production_rpc_registry_from_con(
	data_dir: &Path,
	credential_store: Arc<CredentialStore>,
	ctx: &omp_con::Ctx,
	project_root: Option<&Path>,
) -> Result<(Registry, AuthManager), RegistryError> {
	production_rpc_registry_with_usage_from_con(data_dir, credential_store, ctx, project_root)
		.await
		.map(|(registry, auth, _)| (registry, auth))
}

/// As [`production_rpc_registry_from_con`], also exposing the console usage
/// manager so the RPC host can report each stored account's plan and quota
/// windows.
pub async fn production_rpc_registry_with_usage_from_con(
	data_dir: &Path,
	credential_store: Arc<CredentialStore>,
	ctx: &omp_con::Ctx,
	project_root: Option<&Path>,
) -> Result<(Registry, AuthManager, ConsoleUsageManager), RegistryError> {
	production_assembly_for_session(
		data_dir,
		credential_store,
		None,
		UsageFetcherRegistry::default(),
		inference_settings(ctx, project_root),
	)
	.await
	.map(|(registry, _, _, _, auth, usage, ..)| (registry, auth, usage))
}

/// Invocation-owned inference values that must not enter agent or durable
/// state.
#[derive(Default)]
pub struct InferenceSessionOverrides {
	/// Provider pinned by an invocation API-key lease.
	pub provider:                Option<omp_catalog::ProviderId>,
	/// Generic API key held only by the session's credential broker overlay.
	pub api_key:                 Option<SecretString>,
	/// Opaque prompt-cache identity lowered by compatible codecs.
	pub prompt_cache_affinity:   Option<Str>,
	/// Host-owned per-provider account pins shared with the inference facade.
	pub account_pins:            Option<AccountPins>,
	/// Shared extension-host usage registry allocated before inference assembly.
	pub usage_fetchers:          Option<UsageFetcherRegistry>,
	/// Session-owned provider response hook sink.
	pub provider_response_hooks: Option<omp_ai::ProviderResponseHooks>,
	/// Catalog composed with frozen extension providers before model selection.
	pub catalog:                 Option<Arc<snapshot::Catalog>>,
	/// Effective console context for the session.
	pub con:                     Option<Arc<omp_con::Ctx>>,
}

/// Session-owned production inference authorities assembled from one credential
/// owner.
pub struct ProductionInference {
	/// Immutable registry used by direct chat and provider CONTROL projection.
	pub registry:             Registry,
	/// Cloneable route composition retained for atomic provider registry
	/// rebuilds.
	pub builtins:             BuiltinConfig,
	/// RPC facade sharing the registry's route services and conversation owner.
	pub rpc:                  InferenceRpc,
	/// Narrow GitHub URL credential projection over the canonical encrypted
	/// store.
	pub credential_authority: Arc<dyn omp_envd::github_url::CredentialAuthority>,
	/// Same encrypted authority used for MCP native-key import and OAuth leases.
	pub mcp_authority:        Arc<auth_backend::CombinedAuthAuthority>,
	/// MCP OAuth coordinator over that exact authority.
	pub mcp_oauth:            Arc<omp_envd::mcp::oauth::McpOAuth>,
	/// Authentication owner assembled into the registry's production route
	/// stack.
	pub auth_manager:         AuthManager,
	/// Lifecycle CONTROL view of that exact authentication owner.
	pub auth_control:         AuthControlHandle,
	/// Shared provider usage registry accepting extension-scoped overlays.
	pub usage_fetchers:       UsageFetcherRegistry,
}

/// Builds the production inference RPC authority used by the gateway and chat.
pub async fn production_inference(
	data_dir: &Path,
	tool_registry: Arc<omp_tool::Registry>,
	project_root: Option<&Path>,
) -> Result<ProductionInference, RegistryError> {
	production_inference_from_con(
		data_dir,
		tool_registry,
		project_root,
		Arc::new(omp_con::Ctx::new()),
	)
	.await
}

/// Builds the production inference RPC authority from the process console.
pub async fn production_inference_from_con(
	data_dir: &Path,
	tool_registry: Arc<omp_tool::Registry>,
	project_root: Option<&Path>,
	ctx: Arc<omp_con::Ctx>,
) -> Result<ProductionInference, RegistryError> {
	production_inference_for_session(
		data_dir,
		tool_registry,
		project_root,
		InferenceSessionOverrides { con: Some(ctx), ..Default::default() },
	)
	.await
}

/// Builds a session-owned production inference stack with ephemeral overrides.
#[tracing::instrument(
	level = "debug",
	skip_all,
	fields(
		data_dir = %data_dir.display(),
		project_root = ?project_root,
		provider = ?overrides.provider.as_ref(),
		catalog_override = overrides.catalog.is_some(),
	)
)]
pub async fn production_inference_for_session(
	data_dir: &Path,
	tool_registry: Arc<omp_tool::Registry>,
	project_root: Option<&Path>,
	overrides: InferenceSessionOverrides,
) -> Result<ProductionInference, RegistryError> {
	let ctx = overrides
		.con
		.as_ref()
		.map_or_else(|| Arc::new(omp_con::Ctx::new()), Arc::clone);
	let credential_store =
		open_credential_store_from_con(data_dir.join("credentials.db"), ctx.as_ref())?;
	let provider = overrides.provider.clone();
	let provider_override = provider.is_some();
	let catalog = overrides.catalog.clone();
	let catalog_override = catalog.is_some();
	let usage_fetchers = overrides.usage_fetchers.unwrap_or_default();
	let provider_response_hooks = overrides.provider_response_hooks.unwrap_or_default();
	let invocation_key = match (provider.as_ref(), overrides.api_key) {
		(Some(provider), Some(secret)) => Some((provider.clone(), secret)),
		(None, None) => None,
		(Some(_), None) | (None, Some(_)) => {
			return Err(RegistryError::Inference(Box::new(omp_ai::Error::planning(
				omp_ai::ErrorKind::InvalidRequest,
				omp_ai::ErrorDetail::target(sf!("invocation-credential-override-incomplete")),
				Default::default(),
			))));
		},
	};
	let inference_settings = inference_settings(ctx.as_ref(), project_root);
	let (registry, sessions, authority, mcp_authority, auth_manager, usage_manager, builtins) =
		production_assembly_with_catalog(
			data_dir,
			credential_store,
			invocation_key,
			usage_fetchers,
			inference_settings,
			catalog,
		)
		.await?;
	let usage_fetchers = usage_manager.fetchers();
	let search_settings = omp_ai::search_settings::WebSearchSettings::from_con(ctx.as_ref());
	let rpc = InferenceRpc::new(registry.clone(), sessions, tool_registry)
		.with_session_overrides(provider, overrides.prompt_cache_affinity)
		.with_account_pins(overrides.account_pins.unwrap_or_default())
		.with_provider_response_hooks(provider_response_hooks.clone())
		.with_search_settings(search_settings);
	auth_manager.bind_provider_hooks(provider_response_hooks);
	let auth_control = auth_manager.control_handle();
	let mcp_oauth = Arc::new(omp_envd::mcp::oauth::McpOAuth::new(
		Arc::new(SystemOAuthHttpClient::new()),
		Arc::clone(&mcp_authority),
		Arc::new(omp_envd::mcp::oauth::SystemBrowserLauncher),
	));
	let inference = ProductionInference {
		registry,
		builtins,
		rpc,
		credential_authority: authority,
		mcp_authority,
		mcp_oauth,
		auth_manager,
		auth_control,
		usage_fetchers,
	};
	tracing::debug!(provider_override, catalog_override, "production inference stack composed");
	Ok(inference)
}

fn inference_settings(
	ctx: &omp_con::Ctx,
	project_root: Option<&Path>,
) -> omp_ai::InferenceSettings {
	let cwd = project_root
		.map(Path::to_path_buf)
		.or_else(|| env::current_dir().ok())
		.unwrap_or_default();
	let home = env::var_os("HOME").map_or_else(|| cwd.clone(), std::path::PathBuf::from);
	omp_ai::InferenceSettings {
		retry:                     omp_ai::settings::RetrySettings::from_con(ctx),
		sampling:                  omp_ai::settings::SamplingSettings::from_con(ctx),
		providers:                 omp_ai::settings::ProviderRuntimeSettings::from_con(ctx),
		model:                     omp_catalog::settings::ModelSettings::from_con(ctx)
			.resolve_path_scopes(&cwd, &home),
		context_promotion_enabled: omp_ai::settings::AI_CONTEXT_PROMOTION_ENABLED.get(ctx),
	}
}

async fn production_assembly(
	data_dir: &Path,
	credential_store: Arc<CredentialStore>,
) -> Result<
	(
		Registry,
		ConversationSessionPlanner,
		Arc<dyn omp_envd::github_url::CredentialAuthority>,
		Arc<auth_backend::CombinedAuthAuthority>,
		AuthManager,
		ConsoleUsageManager,
		BuiltinConfig,
	),
	RegistryError,
> {
	production_assembly_for_session(
		data_dir,
		credential_store,
		None,
		UsageFetcherRegistry::default(),
		omp_ai::InferenceSettings::default(),
	)
	.await
}

async fn production_assembly_for_session(
	data_dir: &Path,
	credential_store: Arc<CredentialStore>,
	invocation_key: Option<(omp_catalog::ProviderId, SecretString)>,
	usage_fetchers: UsageFetcherRegistry,
	inference_settings: omp_ai::InferenceSettings,
) -> Result<
	(
		Registry,
		ConversationSessionPlanner,
		Arc<dyn omp_envd::github_url::CredentialAuthority>,
		Arc<auth_backend::CombinedAuthAuthority>,
		AuthManager,
		ConsoleUsageManager,
		BuiltinConfig,
	),
	RegistryError,
> {
	production_assembly_with_catalog(
		data_dir,
		credential_store,
		invocation_key,
		usage_fetchers,
		inference_settings,
		None,
	)
	.await
}

async fn production_assembly_with_catalog(
	data_dir: &Path,
	credential_store: Arc<CredentialStore>,
	invocation_key: Option<(omp_catalog::ProviderId, SecretString)>,
	usage_fetchers: UsageFetcherRegistry,
	inference_settings: omp_ai::InferenceSettings,
	catalog: Option<Arc<snapshot::Catalog>>,
) -> Result<
	(
		Registry,
		ConversationSessionPlanner,
		Arc<dyn omp_envd::github_url::CredentialAuthority>,
		Arc<auth_backend::CombinedAuthAuthority>,
		AuthManager,
		ConsoleUsageManager,
		BuiltinConfig,
	),
	RegistryError,
> {
	fs::create_dir_all(data_dir).map_err(RegistryError::PrepareState)?;
	let catalog = match catalog {
		Some(catalog) => catalog,
		None => {
			let catalog = production_catalog(data_dir)?;
			refresh_model_discovery_cache(data_dir, catalog).await?
		},
	};
	#[cfg(feature = "local-applefm")]
	let apple_routes = catalog
		.routes()
		.iter()
		.filter(|route| {
			route.codec_profile == omp_catalog::CodecProfile::AppleFm
				&& route.transport == omp_catalog::TransportKind::Local
		})
		.map(|route| route.id.clone())
		.collect::<Vec<_>>();
	let stored = Arc::new(auth_backend::combined_authority(credential_store.clone()));
	let database = data_dir.join("credentials.db");
	let accounts = AccountPool::with_store(Arc::new(AccountStateStore::open(&database)?))?;
	// Stored route sets predate this catalog whenever per-model route hashes
	// moved; rebinding keeps every account selectable for the live routes.
	match accounts.rebind_routes(&catalog) {
		Ok(0) => {},
		Ok(rebound) => tracing::debug!(rebound, "account routes rebound to the current catalog"),
		Err(error) => tracing::warn!(%error, "account route rebinding failed"),
	}
	// A logout that left the durable record behind would otherwise hydrate
	// an account with no credential; it ranks first for selection and fails
	// every call and every discovery, so it stays out of this process's pool
	// until the user signs in again.
	let credentialless = prune_credentialless_accounts(&catalog, &credential_store, &accounts);
	if !credentialless.is_empty() {
		tracing::warn!(
			accounts = ?credentialless,
			"accounts without a stored credential were left out of the pool; sign in again to \
			 use them"
		);
	}
	let oauth_http = Arc::new(SystemOAuthHttpClient::new());
	// Resolve the Antigravity client version concurrently with the remaining
	// assembly: route codecs freeze their headers at construction, so the
	// bounded manifest probe must settle before `GoogleCcaConfig` is built.
	let antigravity_version = antigravity_version_task(data_dir, oauth_http.clone());
	let oauth_clock = Arc::new(SystemOAuthClock);
	let oauth_custom =
		Arc::new(OAuthCustomDispatcher::builtin(oauth_http.clone(), oauth_clock.clone())?);
	let refresh_coordinator =
		Arc::new(RefreshCoordinator::new("omp-auth-refresh", RefreshPolicy::default())?);
	let acquisition_credentials = CredentialBroker::system(&catalog, CredentialBrokerEngines {
		stored: Some(stored.clone()),
		..CredentialBrokerEngines::default()
	})
	.map_err(|_| {
		RegistryError::Inference(Box::new(omp_ai::Error::planning(
			omp_ai::ErrorKind::InvalidRequest,
			omp_ai::ErrorDetail::target(sf!("catalog-credential-broker-invalid")),
			Default::default(),
		)))
	})?;
	let login_engines: Vec<Arc<dyn AuthLoginEngine>> = vec![
		// Provider-scoped engines must precede generic engines for the same method.
		Arc::new(AlibabaTokenPlanLoginEngine::new(
			catalog.clone(),
			credential_store.clone(),
			accounts.clone(),
			oauth_http.clone(),
		)),
		Arc::new(SecretLoginEngine::new(
			AuthMethod::ApiKey,
			sf!("api-key"),
			catalog.clone(),
			credential_store.clone(),
			accounts.clone(),
		)?),
		Arc::new(SecretLoginEngine::new(
			AuthMethod::SessionToken,
			sf!("session-token"),
			catalog.clone(),
			credential_store.clone(),
			accounts.clone(),
		)?),
		Arc::new(CredentialAcquisitionLoginEngine::new(
			AuthMethod::ApplicationDefault,
			sf!("application-default"),
			catalog.clone(),
			acquisition_credentials.clone(),
			accounts.clone(),
		)?),
		Arc::new(CredentialAcquisitionLoginEngine::new(
			AuthMethod::AwsCredentialChain,
			sf!("aws-credential-chain"),
			catalog.clone(),
			acquisition_credentials.clone(),
			accounts.clone(),
		)?),
		Arc::new(OAuthLoginEngine::new(
			AuthMethod::OAuthPkce,
			catalog.clone(),
			credential_store.clone(),
			accounts.clone(),
			oauth_http.clone(),
			oauth_clock.clone(),
			oauth_custom.clone(),
		)?),
		Arc::new(OAuthLoginEngine::new(
			AuthMethod::OAuthDevice,
			catalog.clone(),
			credential_store.clone(),
			accounts.clone(),
			oauth_http.clone(),
			oauth_clock.clone(),
			oauth_custom.clone(),
		)?),
	];
	let refresh = Arc::new(StoredOAuthRefreshEngine::new(
		catalog.clone(),
		credential_store.clone(),
		accounts.clone(),
		oauth_http.clone(),
		oauth_clock,
		oauth_custom,
		refresh_coordinator,
	));
	// Rejections reported through the broker demote the credential in the one
	// shared pool, so selection stops offering a revoked account.
	let refreshing = Arc::new(
		RefreshingCredentialSource::new(stored.clone(), refresh.clone()).with_pool(accounts.clone()),
	);
	let aws = AwsCredentialSource::system();
	let invocation_supplies_aws_bearer = invocation_key.as_ref().is_some_and(|(provider, _)| {
		catalog.provider(provider).is_some_and(|provider| {
			provider.auth.iter().any(|auth| {
				catalog
					.auth_spec(auth)
					.is_some_and(|auth| auth.kind == AuthSpecKind::AwsSigv4)
			})
		})
	});
	let mut aws_availability = aws.registry_availability().await;
	if invocation_supplies_aws_bearer {
		aws_availability = aws_availability.map(|availability| availability.with_bearer_override());
	}
	let credentials = CredentialBroker::system(&catalog, CredentialBrokerEngines {
		stored: Some(refreshing),
		aws: Some(Arc::new(aws)),
		..CredentialBrokerEngines::default()
	})
	.map_err(|_| {
		RegistryError::Inference(Box::new(omp_ai::Error::planning(
			omp_ai::ErrorKind::InvalidRequest,
			omp_ai::ErrorDetail::target(sf!("catalog-credential-broker-invalid",)),
			Default::default(),
		)))
	})?;
	let credentials = match invocation_key {
		Some((provider, secret)) => credentials
			.with_api_key_override(&catalog, &provider, secret)
			.map_err(|_| {
				RegistryError::Inference(Box::new(omp_ai::Error::planning(
					omp_ai::ErrorKind::InvalidRequest,
					omp_ai::ErrorDetail::target(sf!("invocation-credential-override-invalid")),
					Default::default(),
				)))
			})?,
		None => credentials,
	};
	let auth_manager = AuthManager::new(
		catalog.clone(),
		credential_store,
		credentials.clone(),
		accounts.clone(),
		login_engines,
		refresh,
	)?
	.with_affinity_resolver(CredentialAffinityResolver::new(
		Hash32::sum(placeholder_affinity_key().as_bytes()).into_bytes(),
	));
	let exposed_auth_manager = auth_manager.clone();
	usage_fetchers.install_builtins([
		Arc::new(AlibabaTokenPlanUsageFetcher::new(oauth_http.clone()))
			as Arc<dyn ConsoleUsageFetcher>,
		Arc::new(ClaudeUsageFetcher::new(oauth_http.clone())),
		Arc::new(OpenAiCodexUsageFetcher::new(oauth_http.clone())),
		Arc::new(GithubCopilotUsageFetcher::new(oauth_http.clone())),
		Arc::new(CursorUsageFetcher::new(oauth_http.clone())),
		Arc::new(XaiOauthUsageFetcher::new(oauth_http.clone())),
		Arc::new(GoogleAntigravityUsageFetcher::new(oauth_http.clone())),
		Arc::new(GeminiUsageFetcher::new(oauth_http.clone())),
		Arc::new(KimiUsageFetcher::new(oauth_http.clone())),
		Arc::new(ZaiUsageFetcher::new(oauth_http.clone())),
		Arc::new(MiniMaxCodeUsageFetcher::new(oauth_http.clone())),
		Arc::new(MiniMaxCodeUsageFetcher::china(oauth_http.clone())),
		Arc::new(UmansUsageFetcher::new(oauth_http.clone())),
		Arc::new(SyntheticUsageFetcher::new(oauth_http.clone())),
		Arc::new(OpenCodeGoUsageFetcher::new(oauth_http.clone())),
		Arc::new(OllamaUsageFetcher::new()),
		Arc::new(OllamaUsageFetcher::cloud()),
	]);
	let usage_manager = ConsoleUsageManager::new(
		catalog.clone(),
		credentials.clone(),
		accounts.clone(),
		usage_fetchers,
	);
	let exposed_usage_manager = usage_manager.clone();
	let mut credential_shapers = CredentialShaperRegistry::new();
	credential_shapers
		.register(ProviderShaper::AlibabaTokenPlan(AlibabaTokenPlanShaper::new()))
		.expect("Alibaba Token Plan credential shaper registered once");
	credential_shapers
		.register(ProviderShaper::GithubCopilot(GithubCopilotShaper::new(oauth_http)))
		.expect("GitHub Copilot credential shaper registered once");
	let sessions = ConversationSessionPlanner::open(&database, catalog.clone())?;
	let aws_region = aws_availability
		.as_ref()
		.map_or_else(|_| sf!("us-east-1"), |availability| Str::new(availability.region()));
	let auth_application = AuthApplicationConfig::for_catalog(&catalog, aws_region);
	let antigravity_fingerprint = AntigravityFingerprint {
		version: antigravity_version.await,
		cl:      env_override(ANTIGRAVITY_CL_ENV).unwrap_or_else(|| sf!(DEFAULT_ANTIGRAVITY_CL)),
		os:      env_override(ANTIGRAVITY_OS_ENV).unwrap_or_else(|| sf!(DEFAULT_ANTIGRAVITY_OS)),
		arch:    env_override(ANTIGRAVITY_ARCH_ENV).unwrap_or_else(|| sf!(DEFAULT_ANTIGRAVITY_ARCH)),
	};
	let google_cca = GoogleCcaConfig {
		gemini_cli_platform: Str::from(consts::OS),
		gemini_cli_arch:     Str::from(consts::ARCH),
		antigravity_headers: CcaHeaders::antigravity(&antigravity_fingerprint, false, None),
		antigravity_policy:  AntigravityPolicy::default(),
	};
	let dependencies = ProductionDependencies::new(
		credentials,
		auth_manager,
		accounts,
		sessions.clone(),
		WebSocketTransport::new(),
		google_cca,
		HttpTransport::new().with_browser_fetch(BrowserFetchAdapter),
		auth_application,
		AdmissionController::new(32, 128),
		Duration::from_secs(60),
		Arc::new(BTreeMap::new()),
		Arc::new(credential_shapers),
	)
	.with_settings(inference_settings)
	.with_aws_registry_availability(aws_availability)
	.with_azure_endpoint(production_azure_endpoint()?);
	let dependencies = dependencies.with_usage_manager(usage_manager);
	#[cfg(feature = "local-applefm")]
	let dependencies = {
		use omp_ai::local::applefm::{AppleFmCodec, AppleFmTransport, FRAMEWORK_TIMEOUT};
		match AppleFmTransport::new() {
			Ok(transport) => {
				let backend =
					LocalRouteBackend::new(Arc::new(AppleFmCodec), transport, FRAMEWORK_TIMEOUT);
				dependencies.with_local_routes(
					apple_routes
						.into_iter()
						.map(|route| (route, backend.clone())),
				)
			},
			Err(evidence) => {
				let route_count = apple_routes.len();
				if route_count > 0 {
					tracing::warn!(
						provider = "applefm",
						state = %evidence.state.code(),
						route_count,
						"local provider initialization failed; routes are unavailable"
					);
				}
				let reason = ReasonId(Str::from(evidence.state.code()));
				dependencies.with_local_unavailable(
					apple_routes
						.into_iter()
						.map(|route| (route, reason.clone())),
				)
			},
		}
	};
	let builtins = BuiltinConfig::production(dependencies);
	let registry = Registry::builder(catalog)
		.with_builtins(builtins.clone())?
		.build()?;
	let authority: Arc<dyn omp_envd::github_url::CredentialAuthority> =
		Arc::new(GithubCredentialAuthority::new(Arc::clone(&stored)));
	Ok((
		registry,
		sessions,
		authority,
		stored,
		exposed_auth_manager,
		exposed_usage_manager,
		builtins,
	))
}

/// Resolves the Antigravity client version without blocking assembly work:
/// explicit `OMP_ANTIGRAVITY_VERSION` override → bounded update-manifest
/// discovery → last discovered release persisted in the data directory →
/// pinned reference fallback.
fn antigravity_version_task(
	data_dir: &Path,
	client: Arc<SystemOAuthHttpClient>,
) -> impl Future<Output = Str> {
	let override_version = env_override(ANTIGRAVITY_VERSION_ENV);
	let cache_path = data_dir.join(ANTIGRAVITY_VERSION_CACHE_FILE);
	let fetch = override_version.is_none().then(|| {
		tokio::spawn(async move {
			time::timeout(
				ANTIGRAVITY_VERSION_FETCH_TIMEOUT,
				discover_antigravity_version(client.as_ref()),
			)
			.await
			.ok()
			.flatten()
		})
	});
	async move {
		if let Some(version) = override_version {
			return version;
		}
		if let Some(fetch) = fetch
			&& let Ok(Some(version)) = fetch.await
		{
			// Best-effort persistence so offline boots keep the discovered release.
			let _ = fs::write(&cache_path, version.as_str());
			return version;
		}
		// Discovery failed: prefer the persisted release over the pinned default
		// only when it is actually newer (a stale cache must not undo a shipped
		// fallback bump).
		let cached = fs::read_to_string(&cache_path).ok().and_then(|raw| {
			let raw = raw.trim();
			release_ordinal(raw).map(|ordinal| (Str::from(raw), ordinal))
		});
		let pinned = release_ordinal(DEFAULT_ANTIGRAVITY_VERSION).unwrap_or_default();
		match cached {
			Some((version, ordinal)) if ordinal > pinned => {
				tracing::warn!(
					provider = "google_antigravity",
					fallback = "cached",
					"provider version discovery failed; using fallback"
				);
				version
			},
			_ => {
				tracing::warn!(
					provider = "google_antigravity",
					fallback = "pinned",
					"provider version discovery failed; using fallback"
				);
				sf!(DEFAULT_ANTIGRAVITY_VERSION)
			},
		}
	}
}

/// Parses a `major.minor.patch` release into an orderable key; any other
/// shape is rejected.
fn release_ordinal(version: &str) -> Option<[u64; 3]> {
	let mut ordinal = [0_u64; 3];
	let mut parts = version.split('.');
	for slot in &mut ordinal {
		let part = parts.next()?;
		if part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()) {
			return None;
		}
		*slot = part.parse().ok()?;
	}
	parts.next().is_none().then_some(ordinal)
}

/// Reads a non-empty trimmed environment override.
fn env_override(name: &str) -> Option<Str> {
	env::var(name).ok().and_then(|value| {
		let value = value.trim();
		(!value.is_empty()).then(|| Str::from(value))
	})
}
fn production_azure_endpoint() -> Result<Option<AzureEndpointConfig>, RegistryError> {
	let base = match (env_override(AZURE_BASE_URL_ENV), env_override(AZURE_RESOURCE_NAME_ENV)) {
		(Some(base), _) => Some(base),
		(None, Some(resource))
			if resource
				.chars()
				.all(|character| character.is_ascii_alphanumeric() || character == '-') =>
		{
			Some(Str::from(format!("https://{}.openai.azure.com", resource.as_str())))
		},
		(None, Some(_)) => {
			return Err(RegistryError::CatalogComposition(Box::new(io::Error::new(
				io::ErrorKind::InvalidInput,
				"OMP_AZURE_OPENAI_RESOURCE_NAME is invalid",
			))));
		},
		(None, None) => None,
	};
	let Some(base) = base else {
		return Ok(None);
	};
	AzureEndpointConfig::new(
		base,
		env_override(AZURE_DEPLOYMENT_ENV),
		Arc::new(BTreeMap::new()),
		env_override(AZURE_API_VERSION_ENV),
	)
	.map(Some)
	.map_err(|code| {
		RegistryError::CatalogComposition(Box::new(io::Error::new(io::ErrorKind::InvalidInput, code)))
	})
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn catalog_cache_diagnostics_identify_file_and_sqlite_kind_without_raw_text() {
		let path = Path::new("/scratch/models.db");
		for (code, expected) in [(5, "is busy"), (1, "SQLite code 1"), (26, "SQLite code 26")] {
			let error = catalog_cache_error(
				path,
				omp_ai::discovery::DiscoveryStoreError::Sqlite(rusqlite::Error::SqliteFailure(
					rusqlite::ffi::Error::new(code),
					Some("private credential-bearing diagnostic".into()),
				)),
			);
			let rendered = error.to_string();
			assert!(rendered.contains("/scratch/models.db"));
			assert!(rendered.contains(expected));
			assert!(!rendered.contains("private"));
			assert!(std::error::Error::source(&error).is_some());
		}
	}

	#[test]
	fn catalog_cache_corruption_has_an_actionable_startup_diagnostic() {
		let directory = tempfile::tempdir().expect("state");
		fs::write(directory.path().join("models.db"), b"not a SQLite database")
			.expect("broken cache");
		let error = production_catalog(directory.path()).expect_err("cache must fail explicitly");
		assert!(matches!(error, RegistryError::DiscoveryCacheSqlite { code: 26, .. }));
		assert!(error.to_string().contains("models.db"));
	}

	#[test]
	fn an_unbundled_anthropic_model_survives_cache_expiry_with_live_limits() {
		use omp_catalog::{DiscoveredModel, ModelKey, ProviderId, RouteId, WireModelId};
		let data_dir = tempfile::tempdir().expect("data dir");
		let bundled = snapshot::Catalog::embedded();
		let template = bundled
			.model(ModelKey::from_ref("anthropic/claude-opus-5"))
			.unwrap();
		let key = ModelKey::from("anthropic/claude-opus-99-7");
		assert!(bundled.model(&key).is_none());
		let mut capabilities = template.capabilities.clone();
		if let omp_catalog::Availability::Native(reasoning) =
			&mut capabilities.chat.as_mut().unwrap().reasoning
		{
			reasoning.features =
				omp_catalog::ReasoningFeatureBits::EFFORT | omp_catalog::ReasoningFeatureBits::ADAPTIVE;
		}
		let mut limits = template.limits;
		limits.maximum_input_tokens = Some(987_654);
		let row = DiscoveredModel {
			provider:              ProviderId::from("anthropic"),
			route:                 RouteId::from("anthropic/primary"),
			wire_model:            WireModelId::from("claude-opus-99-7"),
			aliases:               Box::new([]),
			display_name:          Some(Str::new_static("Future model")),
			declared_class:        Some(template.class.clone()),
			declared_operations:   capabilities.operations,
			declared_capabilities: Some(capabilities),
			declared_limits:       Some(limits),
			declared_pricing:      Box::new([]),
			extended_context_mode: None,
			availability:          Some(omp_catalog::ModelAvailability::Available),
			source:                Str::new_static("runtime-anthropic-models-v1"),
			observed_at_ms:        Some(1),
			updated_at_ms:         None,
			deprecated:            None,
		};
		omp_ai::discovery::DiscoveryStore::open(&data_dir.path().join("models.db"))
			.unwrap()
			.publish(
				&omp_ai::discovery::DiscoveryCacheKey::provider("anthropic"),
				&[row],
				1,
				Duration::from_secs(1),
			)
			.unwrap();
		let catalog = production_catalog(data_dir.path()).expect("catalog hydrates offline");
		let model = catalog
			.model(&key)
			.expect("unbundled model remains selectable");
		assert_eq!(model.limits.maximum_input_tokens, Some(987_654));
		assert_eq!(model.wire_ids[0].1.as_str(), "claude-opus-99-7");
		assert!(model.thinking.is_some(), "observed adaptive effort contract is retained");
		assert_eq!(model.pricing, omp_catalog::Pricing::default(), "no fabricated price");
	}

	#[test]
	fn a_cached_row_for_a_bundled_model_changes_only_its_offered_tiers() {
		use omp_catalog::{
			Availability, DiscoveredModel, ModelKey, OperationBits, OperationKind, ProviderFamily,
			ProviderId, RouteId, ServiceTier, WireModelId,
		};
		let data_dir = tempfile::tempdir().expect("data dir");
		let bundled = snapshot::Catalog::embedded();
		let key = ModelKey::from("openai-codex/gpt-5.6-sol");
		let before = bundled.model(&key).expect("bundled sol").clone();
		let fast = ServiceTier::for_family(ProviderFamily::OpenAi, "priority").expect("priority");
		let mut capabilities = before.capabilities.clone();
		capabilities.chat.as_mut().expect("chat").service_tiers =
			Availability::Native(vec![fast.clone()].into_boxed_slice());
		let mut operations = OperationBits::empty();
		operations.insert_kind(OperationKind::Chat);
		let row = DiscoveredModel {
			provider:              ProviderId::from("openai-codex"),
			route:                 RouteId::from("openai-codex/primary"),
			wire_model:            WireModelId::from("gpt-5.6-sol"),
			aliases:               Box::new([]),
			display_name:          Some(Str::new_static("GPT-5.6 Sol")),
			declared_class:        Some(before.class.clone()),
			declared_operations:   operations,
			declared_capabilities: Some(capabilities),
			declared_limits:       Some(before.limits.clone()),
			declared_pricing:      Box::new([]),
			extended_context_mode: None,
			availability:          Some(before.availability.clone()),
			source:                Str::new_static("test"),
			observed_at_ms:        Some(1),
			updated_at_ms:         None,
			deprecated:            None,
		};
		let now_ms = std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.expect("clock")
			.as_millis()
			.try_into()
			.expect("ms");
		omp_ai::discovery::DiscoveryStore::open(&data_dir.path().join("models.db"))
			.expect("store")
			.publish(
				&omp_ai::discovery::DiscoveryCacheKey::provider(ProviderId::from("openai-codex")),
				&[row],
				now_ms,
				std::time::Duration::from_secs(3600),
			)
			.expect("publish");
		let catalog = production_catalog(data_dir.path()).expect("catalog hydrates");
		let after = catalog.model(&key).expect("sol survives hydration");
		assert_eq!(
			after
				.capabilities
				.chat
				.as_ref()
				.expect("chat")
				.service_tiers,
			Availability::Native(vec![fast].into_boxed_slice()),
			"the live listing's tiers reach the bundled model"
		);
		assert_eq!(after.pricing, before.pricing, "the bundle keeps its pricing");
		assert_eq!(after.thinking, before.thinking, "the bundle keeps its thinking policy");
		assert_eq!(after.routes, before.routes, "the bundle keeps its routes");
		assert_eq!(after.wire_ids, before.wire_ids, "the bundle keeps its wire ids");
	}

	#[test]
	fn hydration_prefers_the_pinned_accounts_listing_over_the_providers() {
		use omp_catalog::{
			Availability, DiscoveredModel, ModelKey, OperationBits, OperationKind, ProviderFamily,
			ProviderId, RouteId, ServiceTier, WireModelId,
		};
		let data_dir = tempfile::tempdir().expect("data dir");
		let bundled = snapshot::Catalog::embedded();
		let key = ModelKey::from("openai-codex/gpt-5.6-luna");
		let before = bundled.model(&key).expect("bundled luna").clone();
		let fast = ServiceTier::for_family(ProviderFamily::OpenAi, "priority").expect("priority");
		let row = |tiers: Availability<Box<[ServiceTier]>>| {
			let mut capabilities = before.capabilities.clone();
			capabilities.chat.as_mut().expect("chat").service_tiers = tiers;
			let mut operations = OperationBits::empty();
			operations.insert_kind(OperationKind::Chat);
			DiscoveredModel {
				provider:              ProviderId::from("openai-codex"),
				route:                 RouteId::from("openai-codex/primary"),
				wire_model:            WireModelId::from("gpt-5.6-luna"),
				aliases:               Box::new([]),
				display_name:          None,
				declared_class:        Some(before.class.clone()),
				declared_operations:   operations,
				declared_capabilities: Some(capabilities),
				declared_limits:       Some(before.limits.clone()),
				declared_pricing:      Box::new([]),
				extended_context_mode: None,
				availability:          Some(before.availability.clone()),
				source:                Str::new_static("test"),
				observed_at_ms:        Some(1),
				updated_at_ms:         None,
				deprecated:            None,
			}
		};
		let now_ms = std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.expect("clock")
			.as_millis()
			.try_into()
			.expect("ms");
		let store = omp_ai::discovery::DiscoveryStore::open(&data_dir.path().join("models.db"))
			.expect("store");
		let ttl = std::time::Duration::from_secs(3600);
		// The provider's default account is offered no tier on Luna...
		store
			.publish(
				&omp_ai::discovery::DiscoveryCacheKey::provider(ProviderId::from("openai-codex")),
				&[row(Availability::Unsupported)],
				now_ms,
				ttl,
			)
			.expect("provider-wide rows");
		// ...while the account the host pinned is.
		store
			.publish(
				&omp_ai::discovery::DiscoveryCacheKey::credential(
					ProviderId::from("openai-codex"),
					"principal-pro",
				),
				&[row(Availability::Native(vec![fast.clone()].into_boxed_slice()))],
				now_ms,
				ttl,
			)
			.expect("account rows");
		set_discovery_scopes(BTreeMap::new());
		let unpinned = production_catalog(data_dir.path()).expect("catalog hydrates");
		assert!(matches!(
			unpinned
				.model(&key)
				.expect("luna")
				.capabilities
				.chat
				.as_ref()
				.expect("chat")
				.service_tiers,
			Availability::Unsupported
		));
		set_discovery_scopes(BTreeMap::from([(
			omp_catalog::ProviderId::from("openai-codex"),
			Str::new_static("principal-pro"),
		)]));
		let pinned = production_catalog(data_dir.path()).expect("catalog hydrates");
		assert_eq!(
			pinned
				.model(&key)
				.expect("luna")
				.capabilities
				.chat
				.as_ref()
				.expect("chat")
				.service_tiers,
			Availability::Native(vec![fast].into_boxed_slice())
		);
		set_discovery_scopes(BTreeMap::new());
	}

	#[test]
	fn default_context_declares_credential_key_source() {
		let ctx = omp_con::Ctx::new();
		assert_eq!(SV_CREDENTIAL_KEY_SOURCE.get(&ctx), CredentialKeySourceSetting::Auto);
	}

	#[test]
	fn credential_key_mode_requires_deliberate_configuration() {
		assert_eq!(
			CredentialKeyMode::resolve(None, CredentialKeySourceSetting::Unavailable, true),
			CredentialKeyMode::Unavailable,
		);
		assert_eq!(
			CredentialKeyMode::resolve(None, CredentialKeySourceSetting::LocalFile, false),
			CredentialKeyMode::LocalFile,
		);
		assert_eq!(
			CredentialKeyMode::resolve(None, CredentialKeySourceSetting::OsKeychain, false),
			CredentialKeyMode::OsKeychain,
		);
	}
	#[test]
	fn auto_uses_a_local_key_file_only_for_interactive_processes() {
		assert_eq!(
			CredentialKeyMode::resolve(None, CredentialKeySourceSetting::Auto, true),
			CredentialKeyMode::LocalFile,
		);
		assert_eq!(
			CredentialKeyMode::resolve(None, CredentialKeySourceSetting::Auto, false),
			CredentialKeyMode::Unavailable,
		);
		assert_eq!(
			CredentialKeyMode::resolve(Some("auto"), CredentialKeySourceSetting::Unavailable, true),
			CredentialKeyMode::LocalFile,
		);
	}

	#[test]
	fn explicit_environment_selection_precedes_config_and_invalid_values_fail_closed() {
		assert_eq!(
			CredentialKeyMode::resolve(
				Some("local-file"),
				CredentialKeySourceSetting::Unavailable,
				false,
			),
			CredentialKeyMode::LocalFile,
		);
		assert_eq!(
			CredentialKeyMode::resolve(
				Some("os-keychain"),
				CredentialKeySourceSetting::LocalFile,
				false,
			),
			CredentialKeyMode::OsKeychain,
		);
		assert_eq!(
			CredentialKeyMode::resolve(Some("typo"), CredentialKeySourceSetting::LocalFile, true),
			CredentialKeyMode::Unavailable,
		);
	}
	#[test]
	fn frozen_snapshot_projects_model_policy_into_inference_composition() {
		let ctx = omp_con::Ctx::new();
		ctx.run("ai_default_thinking high")
			.expect("thinking setting");
		ctx.run("ai_provider_order [anthropic openai]")
			.expect("provider order setting");
		ctx.run("ai_openai_websockets off")
			.expect("websocket setting");
		ctx.run("ai_cache_retention long").expect("cache setting");
		let settings = inference_settings(&ctx, None);
		assert_eq!(settings.model.default_thinking, omp_catalog::ThinkingEffort::High,);
		assert_eq!(
			settings
				.model
				.provider_order
				.iter()
				.map(Str::as_str)
				.collect::<Vec<_>>(),
			["anthropic", "openai"],
		);
		assert_eq!(settings.model.openai_websockets, omp_catalog::settings::WireToggle::Off,);
		assert_eq!(
			settings.model.cache_retention,
			omp_catalog::settings::CacheRetentionSetting::Long,
		);
	}
}
