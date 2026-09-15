//! Host-facing account, login, and account-model discovery authority for
//! `omp rpc`.
//!
//! The journal-first RPC server composes its kernel from launch arguments and
//! leaves credentials to the inference stack. Embedding hosts such as native
//! desktop shells additionally need to list stored accounts, pin one account
//! per provider for the live session, run OAuth logins over the protocol, and
//! learn which subscription models an account can actually use. This module
//! owns that surface so the server loop only dispatches to it.

use std::{
	collections::{BTreeMap, HashMap, HashSet},
	path::{Path, PathBuf},
	sync::Arc,
	time::Duration,
};

use miette::miette;
use omp_ai::{
	AccountPins, Registry,
	answer::{AccountState, AccountSummary, AuthAnswer, AuthEvent, AuthPromptKind, AuthSession},
	auth::manager::AuthManager,
	call::{AuthInput, AuthMethod, AuthRequest, LoginRequest},
	id::LoginSessionId,
};
use omp_catalog::{ProviderId, settings::ModelSettings};
use omp_core::{ExposeSecret as _, SecretString, Str};
use omp_rpc::protocol::{
	OAuthProvider, RpcAccount, RpcAccountState, RpcAuthAccount, RpcAuthAnswerFrame, RpcAuthEvent,
	RpcAuthEventFrame, RpcAuthInputKind, RpcAuthMethod, RpcAuthPromptKind, RpcAuthTerminalFrame,
	RpcAuthTerminalOutcome, RpcErrorCode,
};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio::time;
use tokio_util::sync::CancellationToken;

/// Bound on one provider's live account-model discovery at startup or
/// refresh; a stalled endpoint must not hold the host's ready handshake.
const ACCOUNT_MODEL_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(8);
const ANTHROPIC_PROVIDER: &str = "anthropic";
const CODEX_PROVIDER: &str = "openai-codex";
/// Providers whose authenticated discovery endpoint reports account-specific
/// models beyond the bundled catalog.
const ACCOUNT_DISCOVERY_PROVIDERS: [&str; 2] = [ANTHROPIC_PROVIDER, CODEX_PROVIDER];
/// Claude subscription models every authenticated account can select; the
/// authenticated bootstrap response adds account-specific options.
const ANTHROPIC_SUBSCRIPTION_MODELS: [&str; 3] =
	["anthropic/claude-fable-5", "anthropic/claude-opus-5", "anthropic/claude-sonnet-5"];

/// Emits one protocol frame to the connected host.
pub(crate) type Notify = Arc<dyn Fn(Value) + Send + Sync>;

/// Structured failure returned to the host as an RPC error response.
#[derive(Debug)]
pub(crate) struct CommandError {
	pub code:    RpcErrorCode,
	pub message: String,
}

impl CommandError {
	fn new(code: &'static str, message: impl Into<String>) -> Self {
		Self { code: RpcErrorCode::new(code), message: message.into() }
	}
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AccountParams {
	account_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LoginParams {
	provider_id: String,
	#[serde(default)]
	method:      Option<RpcAuthMethod>,
}

struct PendingAuth {
	cancellation: CancellationToken,
	prompt:       Option<(String, RpcAuthPromptKind)>,
}

#[derive(Default)]
struct Inner {
	providers:        Vec<OAuthProvider>,
	/// Account id pinned per provider for the live session.
	pinned:           BTreeMap<String, String>,
	/// Anthropic selectors the authenticated subscription can use.
	anthropic_models: HashSet<String>,
	pending_auth:     HashMap<String, PendingAuth>,
}

/// Credential, account, and discovery authority shared by the RPC server
/// loop and the login forwarders it spawns.
pub(crate) struct AccountAuthority {
	auth:     AuthManager,
	registry: Registry,
	data_dir: PathBuf,
	settings: ModelSettings,
	/// Host-owned per-provider pins attached to every inference call.
	pub pins: AccountPins,
	inner:    Mutex<Inner>,
}

impl AccountAuthority {
	/// Composes the credential authority, runs account-model discovery for
	/// every authenticated provider (from the cache when fresh), and records
	/// which providers the host may log in to.
	///
	/// This must run before the launch catalog is composed: discovery
	/// publishes into `models.db`, and `production_catalog` hydrates that
	/// cache so newly discovered models become selectable.
	pub(crate) async fn compose(
		data_dir: &Path,
		ctx: &omp_con::Ctx,
		project: &Path,
		home: &Path,
		pinned_accounts: &[Str],
	) -> miette::Result<Arc<Self>> {
		let store = omp_driver::registry::open_credential_store(data_dir.join("credentials.db"))
			.map_err(|error| miette!(error))?;
		let (registry, auth) = omp_driver::registry::production_rpc_registry_from_con(
			data_dir,
			store,
			ctx,
			Some(project),
		)
		.await
		.map_err(|error| miette!(error))?;
		let settings = ModelSettings::from_con(ctx).resolve_path_scopes(project, home);
		let accounts = stored_accounts(&auth).await.unwrap_or_default();
		let authenticated = active_providers(&accounts);
		// An account the host pins at launch answers discovery for its
		// provider, and the launch catalog hydrates from that account's
		// listing: what a subscription is offered (models, the "Fast" tier)
		// is the account's fact, not the provider's.
		let pins = AccountPins::default();
		for pinned in pinned_accounts {
			let Some(account) = accounts
				.iter()
				.find(|account| account.account.as_str() == pinned.as_str())
			else {
				eprintln!("warning: pinned account `{pinned}` is not stored; ignoring");
				continue;
			};
			match &account.principal {
				Some(principal) => pins.set(account.provider.clone(), principal.clone()),
				None => eprintln!("warning: pinned account `{pinned}` has no principal; ignoring"),
			}
		}
		omp_driver::registry::set_discovery_scopes(
			pins
				.snapshot()
				.into_iter()
				.map(|(provider, principal)| (provider, Str::new(principal.as_str())))
				.collect(),
		);
		let discovered =
			discover_account_models(&registry, data_dir, &authenticated, false, &pins).await;
		let anthropic_models = anthropic_models_for(&authenticated, &discovered);
		let providers = registry
			.catalog()
			.providers()
			.iter()
			.filter(|entry| settings.provider_allowed(entry.id.as_str()))
			.map(|entry| OAuthProvider {
				id:            entry.id.as_str().to_owned(),
				name:          entry.name.to_string(),
				available:     true,
				authenticated: authenticated.contains(entry.id.as_str()),
			})
			.collect();
		Ok(Arc::new(Self {
			auth,
			registry,
			data_dir: data_dir.to_path_buf(),
			settings,
			pins,
			inner: Mutex::new(Inner {
				providers,
				pinned: BTreeMap::new(),
				anthropic_models,
				pending_auth: HashMap::new(),
			}),
		}))
	}

	/// Whether a catalog model key should be offered to the host: model
	/// policy applies everywhere, and Anthropic models are limited to the
	/// ones the authenticated subscription reports.
	pub(crate) fn model_visible(&self, key: &str) -> bool {
		let Some((provider, model)) = key.split_once('/') else {
			return false;
		};
		if provider == ANTHROPIC_PROVIDER && !self.inner.lock().anthropic_models.contains(key) {
			return false;
		}
		self.settings.model_allowed(provider, model)
	}

	/// `get_login_providers`: every provider the host may authenticate,
	/// with its current authentication state.
	pub(crate) fn login_providers(&self) -> Value {
		json!({ "providers": self.inner.lock().providers })
	}

	/// Account ids pinned per provider, for `get_state`.
	pub(crate) fn pinned_accounts(&self) -> Value {
		json!(self.inner.lock().pinned)
	}

	async fn stored_accounts(&self) -> Result<Vec<AccountSummary>, CommandError> {
		stored_accounts(&self.auth).await
	}

	/// Re-derives every provider's `authenticated` flag from the live account
	/// list so the flags agree with what `list_accounts` just reported.
	fn sync_provider_authentication(&self, accounts: &[AccountSummary]) {
		let mut inner = self.inner.lock();
		for provider in &mut inner.providers {
			provider.authenticated = accounts.iter().any(|account| {
				account.provider.as_str() == provider.id && account.state == AccountState::Active
			});
		}
	}

	/// `list_accounts`: non-secret summaries of every stored account.
	pub(crate) async fn list_accounts(&self) -> Result<Value, CommandError> {
		let accounts = self.stored_accounts().await?;
		self.sync_provider_authentication(&accounts);
		let accounts = accounts.into_iter().map(rpc_account).collect::<Vec<_>>();
		Ok(json!({ "accounts": accounts }))
	}

	/// `refresh_models`: re-runs account model discovery for every
	/// authenticated provider, bypassing the cache. The live kernel's catalog
	/// was composed at launch, so the host must relaunch the child before the
	/// widened list becomes selectable.
	pub(crate) async fn refresh_models(&self) -> Result<Value, CommandError> {
		let accounts = self.stored_accounts().await?;
		let authenticated = active_providers(&accounts);
		let discovered =
			discover_account_models(&self.registry, &self.data_dir, &authenticated, true, &self.pins)
				.await;
		let anthropic_models = anthropic_models_for(&authenticated, &discovered);
		self.inner.lock().anthropic_models = anthropic_models;
		let mut models = discovered.values().flatten().cloned().collect::<Vec<_>>();
		models.sort();
		models.dedup();
		let mut refreshed = discovered.keys().cloned().collect::<Vec<_>>();
		refreshed.sort();
		Ok(json!({ "models": models, "refreshed": refreshed, "restartRequired": true }))
	}

	/// `set_account`: pins one stored account for its provider on every later
	/// inference call of this session.
	pub(crate) async fn set_account(
		&self,
		params: &Map<String, Value>,
	) -> Result<Value, CommandError> {
		let params = parse_params::<AccountParams>(params)?;
		let account_id = params.account_id;
		if account_id.is_empty() {
			return Err(CommandError::new("invalid_params", "accountId must not be empty"));
		}
		let accounts = self.stored_accounts().await?;
		let account = accounts
			.iter()
			.find(|candidate| candidate.account.as_str() == account_id)
			.ok_or_else(|| {
				CommandError::new("account_not_found", format!("unknown account `{account_id}`"))
			})?;
		if !matches!(account.state, AccountState::Active | AccountState::RefreshRequired) {
			return Err(CommandError::new(
				"account_unavailable",
				format!("account `{account_id}` cannot serve requests"),
			));
		}
		let Some(principal) = account.principal.clone() else {
			return Err(CommandError::new(
				"account_principal_unknown",
				format!("account `{account_id}` has no principal to pin"),
			));
		};
		// The pin is read by every later planned call.
		self.pins.set(account.provider.clone(), principal.clone());
		self
			.inner
			.lock()
			.pinned
			.insert(account.provider.as_str().to_owned(), account_id.clone());
		// The launch catalog was hydrated from the account pinned at launch.
		// A different account may be offered other models and tiers, so its
		// listing is fetched now (into its own cache scope) and the host is
		// told a relaunch would show it.
		let launch_scope = omp_driver::registry::discovery_scope(&account.provider);
		let restart_required = launch_scope.as_deref() != Some(principal.as_str());
		if restart_required {
			let _ = discover_provider_models(
				&self.registry,
				&self.data_dir,
				account.provider.as_str(),
				false,
				&self.pins,
			)
			.await;
		}
		Ok(json!({
			"accountId": account_id,
			"providerId": account.provider.as_str(),
			"principal": principal.as_str(),
			"restartRequired": restart_required,
		}))
	}

	/// `logout_account`: forgets one stored account and clears its pin.
	pub(crate) async fn logout_account(
		&self,
		params: &Map<String, Value>,
	) -> Result<Value, CommandError> {
		let params = parse_params::<AccountParams>(params)?;
		let account_id = params.account_id;
		if account_id.is_empty() {
			return Err(CommandError::new("invalid_params", "accountId must not be empty"));
		}
		let accounts = self.stored_accounts().await?;
		let account = accounts
			.iter()
			.find(|candidate| candidate.account.as_str() == account_id)
			.cloned()
			.ok_or_else(|| {
				CommandError::new("account_not_found", format!("unknown account `{account_id}`"))
			})?;
		self
			.auth
			.execute(AuthRequest::Logout { account: account.account.clone() })
			.await
			.map_err(|error| CommandError::new("auth_failed", error.to_string()))?;
		let remaining = self.stored_accounts().await?;
		self.sync_provider_authentication(&remaining);
		let provider = account.provider.as_str().to_owned();
		let was_pinned = self
			.inner
			.lock()
			.pinned
			.get(&provider)
			.is_some_and(|pinned| *pinned == account_id);
		if was_pinned {
			self.pins.clear(&account.provider);
			self.inner.lock().pinned.remove(&provider);
		}
		let provider_authenticated = remaining.iter().any(|candidate| {
			candidate.provider == account.provider && candidate.state == AccountState::Active
		});
		Ok(json!({
			"accountId": account_id,
			"providerId": provider,
			"providerAuthenticated": provider_authenticated,
		}))
	}

	/// `login`: starts an authentication session for one provider and
	/// forwards its events to the host as `auth_event` and `auth_terminal`
	/// frames until it completes.
	pub(crate) async fn login(
		self: &Arc<Self>,
		params: &Map<String, Value>,
		notify: Notify,
	) -> Result<Value, CommandError> {
		let params = parse_params::<LoginParams>(params)?;
		if !self
			.inner
			.lock()
			.providers
			.iter()
			.any(|candidate| candidate.id == params.provider_id)
		{
			return Err(CommandError::new(
				"provider_not_found",
				format!("unknown provider `{}`", params.provider_id),
			));
		}
		let provider = ProviderId::from(params.provider_id.as_str());
		let method = params.method.map(auth_method);
		let session = match self
			.auth
			.execute(AuthRequest::Login(LoginRequest { provider, method }))
			.await
		{
			Ok(AuthAnswer::Session(session)) => session,
			Ok(_) => {
				return Err(CommandError::new(
					"invalid_auth_answer",
					"authentication manager returned a non-session login answer",
				));
			},
			Err(error) => return Err(CommandError::new("auth_failed", error.to_string())),
		};
		let session_id = session.id.as_str().to_owned();
		{
			let mut inner = self.inner.lock();
			if inner.pending_auth.contains_key(&session_id) {
				return Err(CommandError::new(
					"auth_session_exists",
					"authentication session is already active",
				));
			}
			inner.pending_auth.insert(session_id.clone(), PendingAuth {
				cancellation: CancellationToken::new(),
				prompt:       None,
			});
		}
		self.start_auth_forwarder(session, notify);
		Ok(json!({
			"providerId": params.provider_id,
			"sessionId": session_id,
			"outcome": "started",
		}))
	}

	fn start_auth_forwarder(self: &Arc<Self>, session: AuthSession, notify: Notify) {
		let session_id = session.id.as_str().to_owned();
		let cancellation = self
			.inner
			.lock()
			.pending_auth
			.get(&session_id)
			.map(|pending| pending.cancellation.clone())
			.expect("authentication session inserted before forwarding");
		let authority = Arc::clone(self);
		tokio::spawn(async move {
			loop {
				let event = tokio::select! {
					() = cancellation.cancelled() => {
						let _ = authority.auth.execute(AuthRequest::Submit {
							session: LoginSessionId::from(session_id.as_str()),
							input: AuthInput::Cancel,
						}).await;
						notify(auth_terminal(&session_id, RpcAuthTerminalOutcome::Cancelled, None, None));
						break;
					},
					event = session.events.recv_async() => event,
				};
				let event = match event {
					Ok(Ok(event)) => event,
					Ok(Err(error)) => {
						notify(auth_terminal(
							&session_id,
							RpcAuthTerminalOutcome::Failed,
							None,
							Some(error.to_string()),
						));
						break;
					},
					Err(_) => {
						notify(auth_terminal(
							&session_id,
							RpcAuthTerminalOutcome::Failed,
							None,
							Some("authentication event stream closed before completion".into()),
						));
						break;
					},
				};
				match event {
					AuthEvent::OpenUrl { url, launch } => {
						notify(auth_event(&session_id, RpcAuthEvent::OpenUrl {
							url:    url.to_string(),
							launch: launch.map(|url| url.to_string()),
						}));
					},
					AuthEvent::ShowDeviceCode { code, verification_url } => {
						notify(auth_event(&session_id, RpcAuthEvent::DeviceCode {
							code:             code.expose_secret().to_owned(),
							verification_url: verification_url.to_string(),
						}));
					},
					AuthEvent::Prompt(prompt) => {
						let input = auth_prompt_kind(prompt.input);
						if let Some(pending) = authority.inner.lock().pending_auth.get_mut(&session_id) {
							pending.prompt = Some((prompt.id.to_string(), input));
						}
						notify(auth_event(&session_id, RpcAuthEvent::Prompt {
							prompt_id: prompt.id.to_string(),
							message: prompt.message.to_string(),
							input,
						}));
					},
					AuthEvent::Waiting => {
						notify(auth_event(&session_id, RpcAuthEvent::Waiting));
					},
					AuthEvent::Complete(account) => {
						if let Some(provider) = authority
							.inner
							.lock()
							.providers
							.iter_mut()
							.find(|provider| provider.id == account.provider.as_str())
						{
							provider.authenticated = true;
						}
						notify(auth_terminal(
							&session_id,
							RpcAuthTerminalOutcome::Completed,
							Some(auth_account(account)),
							None,
						));
						break;
					},
				}
			}
			authority.inner.lock().pending_auth.remove(&session_id);
		});
	}

	/// `auth_answer`: forwards the host's answer to the pending prompt of an
	/// authentication session, or cancels the session.
	pub(crate) async fn answer_auth(&self, answer: RpcAuthAnswerFrame) -> Result<(), CommandError> {
		let expected = self
			.inner
			.lock()
			.pending_auth
			.get(&answer.session_id)
			.map(|pending| pending.prompt.clone())
			.ok_or_else(|| {
				CommandError::new("auth_session_not_found", "authentication session is not active")
			})?;
		if answer.input != RpcAuthInputKind::Cancel {
			let (prompt_id, prompt_kind) = expected.ok_or_else(|| {
				CommandError::new(
					"auth_prompt_not_pending",
					"authentication prompt is not awaiting input",
				)
			})?;
			if answer.prompt_id.as_deref() != Some(prompt_id.as_str()) {
				return Err(CommandError::new(
					"stale_auth_prompt",
					"authentication answer does not match the pending prompt",
				));
			}
			if !auth_input_matches(answer.input, prompt_kind) {
				return Err(CommandError::new(
					"invalid_auth_input",
					"authentication answer kind does not match the pending prompt",
				));
			}
		}
		let input = rpc_auth_input(answer.input, answer.value)?;
		self
			.auth
			.execute(AuthRequest::Submit {
				session: LoginSessionId::from(answer.session_id.as_str()),
				input,
			})
			.await
			.map_err(|error| CommandError::new("auth_failed", error.to_string()))?;
		let mut inner = self.inner.lock();
		if answer.input == RpcAuthInputKind::Cancel {
			if let Some(pending) = inner.pending_auth.get(&answer.session_id) {
				pending.cancellation.cancel();
			}
		} else if let Some(pending) = inner.pending_auth.get_mut(&answer.session_id) {
			pending.prompt = None;
		}
		Ok(())
	}
}

async fn stored_accounts(auth: &AuthManager) -> Result<Vec<AccountSummary>, CommandError> {
	match auth
		.execute(AuthRequest::ListAccounts { provider: None })
		.await
	{
		Ok(AuthAnswer::Accounts(accounts)) => Ok(accounts),
		Ok(_) => Err(CommandError::new(
			"invalid_auth_answer",
			"credential authority returned an unexpected account response",
		)),
		Err(error) => Err(CommandError::new("auth_failed", error.to_string())),
	}
}

fn active_providers(accounts: &[AccountSummary]) -> HashSet<String> {
	accounts
		.iter()
		.filter(|account| account.state == AccountState::Active)
		.map(|account| account.provider.as_str().to_owned())
		.collect()
}

fn anthropic_models_for(
	authenticated: &HashSet<String>,
	discovered: &HashMap<String, Vec<String>>,
) -> HashSet<String> {
	if !authenticated.contains(ANTHROPIC_PROVIDER) {
		return HashSet::new();
	}
	anthropic_subscription_models(
		discovered
			.get(ANTHROPIC_PROVIDER)
			.cloned()
			.unwrap_or_default(),
	)
}

/// Account-verified model selectors per provider: the fresh discovery cache
/// when one exists (unless `force_refresh`), otherwise one bounded live
/// discovery. Providers are probed concurrently so a stalled endpoint costs
/// one timeout, and failures degrade to no additions so startup never stalls.
async fn discover_account_models(
	registry: &Registry,
	data_dir: &Path,
	authenticated: &HashSet<String>,
	force_refresh: bool,
	pins: &AccountPins,
) -> HashMap<String, Vec<String>> {
	let probes = ACCOUNT_DISCOVERY_PROVIDERS
		.into_iter()
		.filter(|provider_id| authenticated.contains(*provider_id))
		.map(|provider_id| async move {
			let models =
				discover_provider_models(registry, data_dir, provider_id, force_refresh, pins).await;
			(provider_id.to_owned(), models)
		});
	futures::future::join_all(probes)
		.await
		.into_iter()
		.collect()
}

async fn discover_provider_models(
	registry: &Registry,
	data_dir: &Path,
	provider_id: &str,
	force_refresh: bool,
	pins: &AccountPins,
) -> Vec<String> {
	let provider = ProviderId::from(provider_id);
	let principal = pins
		.snapshot()
		.get(&provider)
		.map(|principal| principal.as_str().to_owned());
	let cached = if force_refresh {
		None
	} else {
		crate::models_cmd::fresh_provider_models(data_dir, &provider, principal.as_deref())
			.unwrap_or_else(|error| {
				eprintln!("warning: {provider_id} model cache is unreadable: {error}");
				None
			})
	};
	if let Some(models) = cached {
		return models;
	}
	match time::timeout(
		ACCOUNT_MODEL_DISCOVERY_TIMEOUT,
		crate::models_cmd::refresh_provider(registry, data_dir, &provider, Some(pins)),
	)
	.await
	{
		Ok(Ok(models)) => models,
		Ok(Err(error)) => {
			eprintln!("warning: {provider_id} account model discovery failed: {error}");
			Vec::new()
		},
		Err(_) => {
			eprintln!(
				"warning: {provider_id} account model discovery exceeded {} seconds",
				ACCOUNT_MODEL_DISCOVERY_TIMEOUT.as_secs()
			);
			Vec::new()
		},
	}
}

fn anthropic_subscription_models(discovered: Vec<String>) -> HashSet<String> {
	ANTHROPIC_SUBSCRIPTION_MODELS
		.into_iter()
		.map(str::to_owned)
		.chain(discovered)
		.collect()
}

fn parse_params<P>(params: &Map<String, Value>) -> Result<P, CommandError>
where
	P: for<'de> Deserialize<'de>,
{
	serde_json::from_value(Value::Object(params.clone()))
		.map_err(|error| CommandError::new("invalid_params", error.to_string()))
}

fn auth_event(session_id: &str, event: RpcAuthEvent) -> Value {
	let frame =
		RpcAuthEventFrame { kind: "auth_event".into(), session_id: session_id.to_owned(), event };
	serde_json::to_value(frame).unwrap_or(Value::Null)
}

fn auth_terminal(
	session_id: &str,
	outcome: RpcAuthTerminalOutcome,
	account: Option<RpcAuthAccount>,
	error: Option<String>,
) -> Value {
	let frame = RpcAuthTerminalFrame {
		kind: "auth_terminal".into(),
		session_id: session_id.to_owned(),
		outcome,
		account,
		code: error.as_ref().map(|_| RpcErrorCode::new("auth_failed")),
		error,
	};
	serde_json::to_value(frame).unwrap_or(Value::Null)
}

fn auth_method(method: RpcAuthMethod) -> AuthMethod {
	match method {
		RpcAuthMethod::ApiKey => AuthMethod::ApiKey,
		RpcAuthMethod::OauthPkce => AuthMethod::OAuthPkce,
		RpcAuthMethod::OauthDevice => AuthMethod::OAuthDevice,
		RpcAuthMethod::ApplicationDefault => AuthMethod::ApplicationDefault,
		RpcAuthMethod::AwsCredentialChain => AuthMethod::AwsCredentialChain,
		RpcAuthMethod::SessionToken => AuthMethod::SessionToken,
	}
}

fn auth_prompt_kind(kind: AuthPromptKind) -> RpcAuthPromptKind {
	match kind {
		AuthPromptKind::AuthorizationCode => RpcAuthPromptKind::AuthorizationCode,
		AuthPromptKind::ApiKey => RpcAuthPromptKind::ApiKey,
		AuthPromptKind::SessionToken => RpcAuthPromptKind::SessionToken,
		AuthPromptKind::PlainText => RpcAuthPromptKind::PlainText,
		AuthPromptKind::OptionalSecret => RpcAuthPromptKind::OptionalSecret,
		AuthPromptKind::Confirmation => RpcAuthPromptKind::Confirmation,
	}
}

/// Whether a host's answer fits the prompt that is waiting for it.
///
/// A PKCE login whose loopback server is listening asks for a confirmation,
/// yet the exchange also completes from a pasted callback URL: the browser
/// that finished the login may be on another machine, where the redirect to
/// `localhost` went nowhere, and the user carries the address it landed on
/// back by hand. The engine accepts that URL, so the seam does too.
fn auth_input_matches(input: RpcAuthInputKind, prompt: RpcAuthPromptKind) -> bool {
	matches!(
		(input, prompt),
		(RpcAuthInputKind::AuthorizationCode, RpcAuthPromptKind::AuthorizationCode)
			| (RpcAuthInputKind::ApiKey, RpcAuthPromptKind::ApiKey)
			| (RpcAuthInputKind::SessionToken, RpcAuthPromptKind::SessionToken)
			| (RpcAuthInputKind::PlainText, RpcAuthPromptKind::PlainText)
			| (RpcAuthInputKind::OptionalSecret, RpcAuthPromptKind::OptionalSecret)
			| (RpcAuthInputKind::DeviceConfirmed, RpcAuthPromptKind::Confirmation)
			| (RpcAuthInputKind::CallbackUrl, RpcAuthPromptKind::AuthorizationCode)
			| (RpcAuthInputKind::CallbackUrl, RpcAuthPromptKind::Confirmation)
	)
}

fn rpc_auth_input(
	kind: RpcAuthInputKind,
	value: Option<String>,
) -> Result<AuthInput, CommandError> {
	let required = |value: Option<String>| {
		value.ok_or_else(|| {
			CommandError::new("invalid_auth_input", "authentication input requires a value")
		})
	};
	match kind {
		RpcAuthInputKind::AuthorizationCode => {
			Ok(AuthInput::AuthorizationCode(SecretString::from(required(value)?)))
		},
		RpcAuthInputKind::ApiKey => Ok(AuthInput::ApiKey(SecretString::from(required(value)?))),
		RpcAuthInputKind::SessionToken => {
			Ok(AuthInput::SessionToken(SecretString::from(required(value)?)))
		},
		RpcAuthInputKind::CallbackUrl => {
			Ok(AuthInput::CallbackUrl(SecretString::from(required(value)?)))
		},
		RpcAuthInputKind::PlainText => Ok(AuthInput::PlainText(Str::from(required(value)?))),
		RpcAuthInputKind::OptionalSecret => {
			Ok(AuthInput::OptionalSecret(SecretString::from(value.unwrap_or_default())))
		},
		RpcAuthInputKind::DeviceConfirmed => {
			if value.is_some() {
				return Err(CommandError::new(
					"invalid_auth_input",
					"device confirmation does not accept a value",
				));
			}
			Ok(AuthInput::DeviceConfirmed)
		},
		RpcAuthInputKind::Cancel => {
			if value.is_some() {
				return Err(CommandError::new(
					"invalid_auth_input",
					"authentication cancellation does not accept a value",
				));
			}
			Ok(AuthInput::Cancel)
		},
	}
}

fn auth_account(account: AccountSummary) -> RpcAuthAccount {
	RpcAuthAccount {
		account_id:  account.account.as_str().to_owned(),
		provider_id: account.provider.as_str().to_owned(),
		principal:   account
			.principal
			.map(|principal| principal.as_str().to_owned()),
		label:       account.label.map(|label| label.to_string()),
	}
}

fn rpc_account(account: AccountSummary) -> RpcAccount {
	RpcAccount {
		account_id:  account.account.as_str().to_owned(),
		provider_id: account.provider.as_str().to_owned(),
		principal:   account
			.principal
			.map(|principal| principal.as_str().to_owned()),
		label:       account.label.map(|label| label.to_string()),
		state:       match account.state {
			AccountState::Active => RpcAccountState::Active,
			AccountState::RefreshRequired => RpcAccountState::RefreshRequired,
			AccountState::Disabled => RpcAccountState::Disabled,
			AccountState::LoggedOut => RpcAccountState::LoggedOut,
		},
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn anthropic_subscription_models_merge_the_baseline_with_discovery() {
		let models = anthropic_subscription_models(vec![
			"anthropic/claude-fable-5-1".into(),
			"anthropic/claude-sonnet-5".into(),
		]);
		assert!(models.contains("anthropic/claude-fable-5-1"));
		assert!(models.contains("anthropic/claude-opus-5"));
		assert_eq!(models.len(), 4);
	}

	#[test]
	fn anthropic_models_require_an_authenticated_account() {
		let discovered =
			HashMap::from([("anthropic".to_owned(), vec!["anthropic/claude-fable-5-1".to_owned()])]);
		assert!(anthropic_models_for(&HashSet::new(), &discovered).is_empty());
		let authenticated = HashSet::from(["anthropic".to_owned()]);
		assert!(
			anthropic_models_for(&authenticated, &discovered).contains("anthropic/claude-fable-5-1")
		);
	}

	#[test]
	fn auth_answers_must_match_the_pending_prompt_kind() {
		assert!(auth_input_matches(
			RpcAuthInputKind::CallbackUrl,
			RpcAuthPromptKind::AuthorizationCode
		));
		// A browser on another machine cannot reach the loopback server; the
		// callback URL it landed on is carried back by hand.
		assert!(auth_input_matches(RpcAuthInputKind::CallbackUrl, RpcAuthPromptKind::Confirmation));
		assert!(!auth_input_matches(
			RpcAuthInputKind::AuthorizationCode,
			RpcAuthPromptKind::Confirmation
		));
		assert!(!auth_input_matches(RpcAuthInputKind::ApiKey, RpcAuthPromptKind::PlainText));
		assert!(rpc_auth_input(RpcAuthInputKind::DeviceConfirmed, Some("x".into())).is_err());
		assert!(rpc_auth_input(RpcAuthInputKind::Cancel, None).is_ok());
	}
}
