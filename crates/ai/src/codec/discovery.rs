//! Standalone typed model-discovery codecs selected by catalog discovery kind.

use bytes::Bytes;
use omp_catalog::{
	Availability, ChatCapabilities, ClassId, DiscoveredModel, DiscoveryKind, DiscoveryPagination,
	DiscoverySpec, ImageDecoderFamily, ImageInputCapabilities, ImageInputFormatBits, ModalityBits,
	ModelAvailability, ModelLimits, OperationBits, OperationKind, ProviderId, ReasoningCapabilities,
	ReasoningEffort, ReasoningFeatureBits, RouteId, StructuredOutputBits, ToolCapabilities,
	ToolFeatureBits, WireModelId, unknown_capabilities,
};
use omp_core::Str;
use serde::Deserialize;
use url::Url;

use crate::{
	auth::AuthScheme,
	body::BodySource,
	call::{DiscoveryRequest, OperationCall},
	codec::{
		Codec, DecodeContext, Decoder, DecoderState, EncodeContext, EncodedRequest, RawEvent,
		RequestHeader, RequestMethod, SizeBounds, ollama,
	},
	error::{Error, ErrorKind, ErrorPhase, RetryAction},
	receipt::ExecutionReceipt,
	transport::{Frame, FramingProtocol},
};

const MAX_FRAME_BYTES: u64 = 16 * 1024 * 1024;
const MAX_RESPONSE_BYTES: u64 = 256 * 1024 * 1024;
const ANTHROPIC_PROVIDER: &str = "anthropic";
const ANTHROPIC_OAUTH_BETA: &str = "oauth-2025-04-20";
const ANTHROPIC_API_VERSION: &str = "2023-06-01";
const CLAUDE_CODE_USER_AGENT: &str = "claude-code/2.1.258";
const CLAUDE_CODE_UPDATE_SENTINEL_PREFIX: &str = "cc-update-required-";

/// Standalone OpenAI-compatible `{ "data": [...] }` model discovery codec.
#[derive(Clone, Debug)]
pub struct OpenAiModelsDiscoveryCodec {
	core: DiscoveryCore,
}

impl OpenAiModelsDiscoveryCodec {
	/// Constructs a codec from an exact OpenAI-models discovery specification.
	pub fn from_spec(spec: &DiscoverySpec) -> Result<Self, Error> {
		DiscoveryCore::new(spec, DiscoveryKind::OpenAiModels).map(|core| Self { core })
	}
}

/// Standalone Google Generative Language `{ "models": [...] }` discovery codec.
#[derive(Clone, Debug)]
pub struct GoogleModelsDiscoveryCodec {
	core: DiscoveryCore,
}

impl GoogleModelsDiscoveryCodec {
	/// Constructs a codec from an exact Google-models discovery specification.
	pub fn from_spec(spec: &DiscoverySpec) -> Result<Self, Error> {
		DiscoveryCore::new(spec, DiscoveryKind::GoogleModels).map(|core| Self { core })
	}
}

/// Standalone account-scoped model discovery codec.
#[derive(Clone, Debug)]
pub struct AccountModelsDiscoveryCodec {
	core: DiscoveryCore,
}

impl AccountModelsDiscoveryCodec {
	/// Constructs a codec from an exact account-models discovery specification.
	pub fn from_spec(spec: &DiscoverySpec) -> Result<Self, Error> {
		// Older bundled catalogs point at a client bootstrap whose optional
		// model list is incomplete. The authenticated Models API is paginated
		// and describes every model available to the account.
		if matches!(
			spec.path.as_str().split('?').next(),
			Some("/api/claude_cli/bootstrap" | "/api/claude_code/bootstrap")
		) {
			let mut spec = spec.clone();
			spec.path = Str::new_static("/v1/models?limit=1000");
			spec.pagination =
				DiscoveryPagination::Cursor { query_parameter: Str::new_static("after_id") };
			spec.label = Str::new_static("anthropic-models-v1");
			return DiscoveryCore::new(&spec, DiscoveryKind::AccountModels).map(|core| Self { core });
		}
		DiscoveryCore::new(spec, DiscoveryKind::AccountModels).map(|core| Self { core })
	}
}

/// Standalone Ollama `/api/tags` discovery codec.
#[derive(Clone, Debug)]
pub struct OllamaTagsDiscoveryCodec {
	core: DiscoveryCore,
}

impl OllamaTagsDiscoveryCodec {
	/// Constructs a codec from an exact Ollama-tags discovery specification.
	pub fn from_spec(spec: &DiscoverySpec) -> Result<Self, Error> {
		DiscoveryCore::new(spec, DiscoveryKind::OllamaTags).map(|core| Self { core })
	}
}

macro_rules! impl_discovery_codec {
	($codec:ty, $flavor:expr) => {
		impl Codec for $codec {
			fn encode(
				&self,
				context: &EncodeContext<'_>,
				operation: &OperationCall,
			) -> Result<EncodedRequest, Error> {
				self.core.encode(context, operation)
			}

			fn decoder(&self, context: &DecodeContext<'_>) -> Result<DecoderState, Error> {
				self.core.decoder(context, $flavor)
			}
		}
	};
}

impl_discovery_codec!(OpenAiModelsDiscoveryCodec, DiscoveryFlavor::OpenAi);
impl_discovery_codec!(GoogleModelsDiscoveryCodec, DiscoveryFlavor::Google);
impl_discovery_codec!(AccountModelsDiscoveryCodec, DiscoveryFlavor::Account);
impl_discovery_codec!(OllamaTagsDiscoveryCodec, DiscoveryFlavor::Ollama);

#[derive(Clone, Debug)]
struct DiscoveryCore {
	spec: DiscoverySpec,
}

impl DiscoveryCore {
	fn new(spec: &DiscoverySpec, expected: DiscoveryKind) -> Result<Self, Error> {
		if spec.kind != expected {
			return Err(invalid_request());
		}
		Ok(Self { spec: spec.clone() })
	}

	fn encode(
		&self,
		context: &EncodeContext<'_>,
		operation: &OperationCall,
	) -> Result<EncodedRequest, Error> {
		let OperationCall::DiscoverModels(request) = operation else {
			return Err(invalid_request());
		};
		let uri = discovery_uri(
			context.route.endpoint.base_url.as_str(),
			self.spec.path.as_str(),
			&self.spec.pagination,
			request,
		)?;
		Ok(EncodedRequest::new(
			OperationKind::DiscoverModels,
			RequestMethod::Get,
			uri,
			discovery_headers(&context.route.provider, context.auth_scheme, self.spec.path.as_str()),
			BodySource::Bytes(Bytes::new()),
			FramingProtocol::Raw,
			SizeBounds {
				request_body: 0,
				frame:        MAX_FRAME_BYTES,
				response:     MAX_RESPONSE_BYTES,
			},
		))
	}

	fn decoder(
		&self,
		context: &DecodeContext<'_>,
		flavor: DiscoveryFlavor,
	) -> Result<DecoderState, Error> {
		if context.operation != OperationKind::DiscoverModels
			|| context.operation_call.kind() != OperationKind::DiscoverModels
		{
			return Err(invalid_request());
		}
		let OperationCall::DiscoverModels(request) = context.operation_call else {
			return Err(invalid_request());
		};
		Ok(Box::new(DiscoveryDecoder {
			provider: context.provider.clone(),
			route: context.route.clone(),
			source: self.spec.label.clone(),
			pagination: self.spec.pagination.clone(),
			request_cursor: request.cursor.clone(),
			flavor,
			done: false,
		}))
	}
}

fn discovery_headers(
	provider: &ProviderId<str>,
	auth_scheme: Option<AuthScheme>,
	path: &str,
) -> Box<[RequestHeader]> {
	if provider.as_str() != ANTHROPIC_PROVIDER {
		return Box::new([]);
	}
	let version = (path.split('?').next() == Some("/v1/models"))
		.then(|| RequestHeader::new_static("anthropic-version", ANTHROPIC_API_VERSION));
	if auth_scheme == Some(AuthScheme::OAuth) {
		return [
			RequestHeader::new_static("accept", "application/json, text/plain, */*"),
			RequestHeader::new_static("content-type", "application/json"),
			RequestHeader::new_static("anthropic-beta", ANTHROPIC_OAUTH_BETA),
			RequestHeader::new_static("user-agent", CLAUDE_CODE_USER_AGENT),
		]
		.into_iter()
		.chain(version)
		.collect();
	}
	version.into_iter().collect()
}

fn discovery_uri(
	base: &str,
	path: &str,
	pagination: &DiscoveryPagination,
	request: &DiscoveryRequest,
) -> Result<Str, Error> {
	let mut joined = String::with_capacity(base.len() + path.len() + 1);
	joined.push_str(base.trim_end_matches('/'));
	if !path.starts_with('/') {
		joined.push('/');
	}
	joined.push_str(path);
	let mut uri = Url::parse(&joined).map_err(|_| invalid_request())?;
	match pagination {
		DiscoveryPagination::SinglePage => {
			if request.cursor.is_some() {
				return Err(invalid_request());
			}
		},
		DiscoveryPagination::Cursor { query_parameter } => {
			if let Some(cursor) = &request.cursor {
				uri.query_pairs_mut()
					.append_pair(query_parameter.as_str(), cursor.as_str());
			}
		},
		DiscoveryPagination::PageNumber { query_parameter, first_page } => {
			let page = request
				.cursor
				.as_ref()
				.map(|cursor| {
					cursor
						.as_str()
						.parse::<u32>()
						.map_err(|_| invalid_request())
				})
				.transpose()?
				.unwrap_or(*first_page);
			uri.query_pairs_mut()
				.append_pair(query_parameter.as_str(), &page.to_string());
		},
	}
	Ok(Str::new(&uri))
}

#[derive(Clone, Copy, Debug)]
enum DiscoveryFlavor {
	OpenAi,
	Google,
	Account,
	Ollama,
}

struct DiscoveryDecoder {
	provider:       ProviderId,
	route:          RouteId,
	source:         Str,
	pagination:     DiscoveryPagination,
	request_cursor: Option<Str>,
	flavor:         DiscoveryFlavor,
	done:           bool,
}

impl Decoder for DiscoveryDecoder {
	fn push(&mut self, frame: Frame, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		if self.done {
			return Err(protocol_error());
		}
		let Frame::Raw(payload) = frame else {
			return Err(protocol_error());
		};
		let (rows, provider_cursor) = match self.flavor {
			DiscoveryFlavor::OpenAi => self.decode_openai(&payload)?,
			DiscoveryFlavor::Google => self.decode_google(&payload)?,
			DiscoveryFlavor::Account => self.decode_account(&payload)?,
			DiscoveryFlavor::Ollama => self.decode_ollama(&payload)?,
		};
		let next_cursor = self.next_cursor(provider_cursor)?;
		emit(RawEvent::DiscoveredModels { rows, next_cursor });
		self.done = true;
		Ok(())
	}

	fn finish(&mut self, _emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		if self.done {
			Ok(())
		} else {
			Err(protocol_error())
		}
	}
}

impl DiscoveryDecoder {
	fn decode_openai(
		&self,
		payload: &[u8],
	) -> Result<(Vec<DiscoveredModel>, ProviderCursor), Error> {
		let envelope: OpenAiEnvelope =
			serde_json::from_slice(payload).map_err(|_| protocol_error())?;
		let cursor = envelope.cursor();
		let rows = envelope
			.data
			.into_iter()
			.map(|model| self.openai_row(model))
			.collect();
		Ok((rows, cursor))
	}

	fn openai_row(&self, model: OpenAiModel) -> DiscoveredModel {
		let context_length = model.context_length.filter(|tokens| *tokens > 0);
		let max_output_tokens = model.max_output_tokens.filter(|tokens| *tokens > 0);
		let limits =
			(context_length.is_some() || max_output_tokens.is_some()).then_some(ModelLimits {
				context_window:        context_length,
				maximum_input_tokens:  None,
				maximum_output_tokens: max_output_tokens,
				maximum_batch:         None,
			});
		let rich_chat_metadata = model.input_modalities.is_some() || model.supports_tools.is_some();
		let mut row = self.row(model.id, model.display_name, None, None);
		row.declared_limits = limits;
		if rich_chat_metadata {
			let modalities = model
				.input_modalities
				.map_or(ModalityBits::empty(), |values| {
					values
						.into_iter()
						.fold(ModalityBits::empty(), |mut bits, modality| {
							bits.insert(match modality {
								OpenAiInputModality::Text => ModalityBits::TEXT,
								OpenAiInputModality::Image => ModalityBits::IMAGE,
							});
							bits
						})
				});
			let tools = if model.supports_tools == Some(false) {
				Availability::Unsupported
			} else {
				// OMP's enriched model-list contract omits this field when
				// tools are usable and emits it only for explicit `false`.
				Availability::Native(ToolCapabilities {
					features:      ToolFeatureBits::empty(),
					maximum_tools: None,
				})
			};
			let mut capabilities = unknown_capabilities();
			capabilities.operations = OperationBits::for_kind(OperationKind::Chat);
			capabilities.chat = Some(openai_chat_capabilities(modalities, tools));
			row.declared_operations = OperationBits::for_kind(OperationKind::Chat);
			row.declared_capabilities = Some(capabilities);
		}
		row
	}

	fn decode_google(
		&self,
		payload: &[u8],
	) -> Result<(Vec<DiscoveredModel>, ProviderCursor), Error> {
		let envelope: GoogleModelsEnvelope =
			serde_json::from_slice(payload).map_err(|_| protocol_error())?;
		let mut rows = Vec::with_capacity(envelope.models.len());
		for model in envelope.models {
			let name: &str = model.name.as_str();
			let wire_model = name.strip_prefix("models/").unwrap_or(name);
			if wire_model.is_empty() {
				return Err(protocol_error());
			}
			let mut operations = OperationBits::empty();
			for method in model.supported_generation_methods {
				match method.as_str() {
					"generateContent" | "streamGenerateContent" => {
						operations.insert_kind(OperationKind::Chat);
					},
					"countTokens" => operations.insert_kind(OperationKind::CountTokens),
					"embedContent" | "batchEmbedContents" => {
						operations.insert_kind(OperationKind::Embed);
					},
					_ => {},
				}
			}
			let limits = (model.input_token_limit.is_some() || model.output_token_limit.is_some())
				.then_some(ModelLimits {
					context_window:        None,
					maximum_input_tokens:  model.input_token_limit,
					maximum_output_tokens: model.output_token_limit,
					maximum_batch:         None,
				});
			let mut row = self.row(Str::new(wire_model), model.display_name, None, None);
			if let Some(base_model) = model.base_model_id {
				let alias = base_model
					.as_str()
					.strip_prefix("models/")
					.unwrap_or(base_model.as_str());
				if !alias.is_empty() && alias != wire_model {
					row.aliases = vec![WireModelId::from(alias)].into_boxed_slice();
				}
			}
			row.declared_operations = operations;
			row.declared_limits = limits;
			rows.push(row);
		}
		Ok((rows, ProviderCursor { next: envelope.next_page_token, ..ProviderCursor::default() }))
	}

	fn decode_account(
		&self,
		payload: &[u8],
	) -> Result<(Vec<DiscoveredModel>, ProviderCursor), Error> {
		let envelope: AccountEnvelope =
			serde_json::from_slice(payload).map_err(|_| protocol_error())?;
		let cursor = envelope.cursor();
		if cursor.has_more == Some(true) && cursor.next.is_none() && cursor.last_id.is_none() {
			return Err(protocol_error());
		}
		let mut rows = Vec::with_capacity(envelope.models.len());
		for model in envelope.models {
			let wire_model = model
				.model
				.or(model.slug)
				.or(model.id)
				.ok_or_else(protocol_error)?;
			if wire_model.starts_with(CLAUDE_CODE_UPDATE_SENTINEL_PREFIX) {
				continue;
			}
			let availability = match model.visibility.as_ref().map(Str::as_str) {
				Some("visible" | "available") => Some(ModelAvailability::Available),
				Some("hide" | "hidden" | "disabled") => Some(ModelAvailability::Disabled),
				_ => None,
			};
			let mut row =
				self.row(wire_model, model.display_name, model.family.map(ClassId::from), availability);
			if self.provider.as_str() == ANTHROPIC_PROVIDER {
				let maximum_input_tokens = model.max_input_tokens.filter(|tokens| *tokens > 0);
				let maximum_output_tokens = model.max_tokens.filter(|tokens| *tokens > 0);
				row.declared_limits = (maximum_input_tokens.is_some()
					|| maximum_output_tokens.is_some())
				.then_some(ModelLimits {
					context_window: None,
					maximum_input_tokens,
					maximum_output_tokens,
					maximum_batch: None,
				});
				// Both the Models API and compact bootstrap list selectable
				// Messages models. Missing optional metadata is not rejection
				// evidence for chat or ordinary client-side tool use.
				row.declared_operations.insert_kind(OperationKind::Chat);
				let mut capabilities = unknown_capabilities();
				capabilities.operations = row.declared_operations;
				capabilities.chat = Some(anthropic_chat_capabilities(model.capabilities.as_ref()));
				row.declared_capabilities = Some(capabilities);
			}
			rows.push(row);
		}
		Ok((rows, cursor))
	}

	fn decode_ollama(
		&self,
		payload: &[u8],
	) -> Result<(Vec<DiscoveredModel>, ProviderCursor), Error> {
		let envelope = ollama::decode_tags(payload)?;
		let rows = envelope
			.models
			.into_iter()
			.map(|model| {
				let wire_model = model.model.unwrap_or_else(|| model.name.clone());
				let declared_class = model
					.details
					.and_then(|details| details.family)
					.map(ClassId::from);
				self.row(wire_model, Some(model.name), declared_class, None)
			})
			.collect();
		Ok((rows, ProviderCursor::default()))
	}

	fn row(
		&self,
		wire_model: Str,
		display_name: Option<Str>,
		declared_class: Option<ClassId>,
		availability: Option<ModelAvailability>,
	) -> DiscoveredModel {
		DiscoveredModel {
			provider: self.provider.clone(),
			route: self.route.clone(),
			wire_model: WireModelId::from(wire_model),
			aliases: Box::new([]),
			display_name,
			declared_class,
			declared_operations: OperationBits::empty(),
			declared_capabilities: None,
			declared_limits: None,
			declared_pricing: Box::new([]),
			extended_context_mode: None,
			availability,
			source: self.source.clone(),
			observed_at_ms: None,
			updated_at_ms: None,
			deprecated: None,
		}
	}

	fn next_cursor(&self, provider: ProviderCursor) -> Result<Option<Str>, Error> {
		match &self.pagination {
			DiscoveryPagination::SinglePage => Ok(None),
			DiscoveryPagination::Cursor { .. } => {
				let next = provider.next.or_else(|| {
					provider
						.has_more
						.and_then(|more| more.then_some(provider.last_id).flatten())
				});
				if provider.has_more == Some(true) && next.as_deref().is_none_or(str::is_empty) {
					return Err(protocol_error());
				}
				Ok(next)
			},
			DiscoveryPagination::PageNumber { first_page, .. } => {
				if provider.has_more != Some(true) {
					return Ok(None);
				}
				let current = self
					.request_cursor
					.as_ref()
					.map(|value| value.as_str().parse::<u32>().map_err(|_| protocol_error()))
					.transpose()?
					.unwrap_or(*first_page);
				current
					.checked_add(1)
					.map(|page| Some(Str::new(page.to_string())))
					.ok_or_else(protocol_error)
			},
		}
	}
}

fn anthropic_chat_capabilities(metadata: Option<&AnthropicCapabilities>) -> ChatCapabilities {
	let mut chat = openai_chat_capabilities(
		ModalityBits::TEXT,
		Availability::Native(ToolCapabilities {
			features:      ToolFeatureBits::empty(),
			maximum_tools: None,
		}),
	);
	let Some(metadata) = metadata else {
		// A native bitset is a complete modality declaration; without image
		// metadata, a text-only set would falsely reject image inputs.
		chat.input_modalities = Availability::Unknown;
		return chat;
	};
	match metadata
		.image_input
		.as_ref()
		.and_then(|image| image.supported)
	{
		Some(true) => {
			chat.input_modalities = Availability::Native(ModalityBits::TEXT | ModalityBits::IMAGE);
			chat.image_input = Availability::Native(ImageInputCapabilities {
				formats: ImageInputFormatBits::ALL,
				decoder: ImageDecoderFamily::Native,
			});
		},
		Some(false) => chat.image_input = Availability::Unsupported,
		None => chat.input_modalities = Availability::Unknown,
	}
	chat.structured_output = match metadata
		.structured_outputs
		.as_ref()
		.and_then(|structured| structured.supported)
	{
		Some(true) => Availability::Native(StructuredOutputBits::JSON_SCHEMA),
		Some(false) => Availability::Unsupported,
		None => Availability::Unknown,
	};
	chat.reasoning = metadata.reasoning();
	chat
}

const fn openai_chat_capabilities(
	input_modalities: ModalityBits,
	tools: Availability<ToolCapabilities>,
) -> ChatCapabilities {
	ChatCapabilities {
		roles: Availability::Unknown,
		mid_session_roles: Availability::Unknown,
		tools,
		structured_output: Availability::Unknown,
		grammar: Availability::Unknown,
		text_verbosity: Availability::Unknown,
		reasoning: Availability::Unknown,
		input_modalities: Availability::Native(input_modalities),
		image_input: Availability::Unknown,
		hosted_tools: Availability::Unknown,
		prompt_caching: Availability::Unknown,
		service_tiers: Availability::Unknown,
		sampling: Availability::Unknown,
		safety: Availability::Unknown,
		determinism: Availability::Unknown,
		server_state: Availability::Unknown,
		logprobs: Availability::Unknown,
	}
}

#[derive(Default)]
struct ProviderCursor {
	next:     Option<Str>,
	has_more: Option<bool>,
	last_id:  Option<Str>,
}

#[derive(Deserialize)]
struct OpenAiEnvelope {
	data:        Vec<OpenAiModel>,
	#[serde(default)]
	next:        Option<Str>,
	#[serde(default)]
	next_cursor: Option<Str>,
	#[serde(default)]
	has_more:    Option<bool>,
	#[serde(default)]
	last_id:     Option<Str>,
}

impl OpenAiEnvelope {
	fn cursor(&self) -> ProviderCursor {
		ProviderCursor {
			next:     self.next_cursor.clone().or_else(|| self.next.clone()),
			has_more: self.has_more,
			last_id:  self.last_id.clone(),
		}
	}
}

#[derive(Deserialize)]
struct OpenAiModel {
	id:                Str,
	#[serde(default)]
	display_name:      Option<Str>,
	#[serde(default)]
	context_length:    Option<u64>,
	#[serde(default)]
	max_output_tokens: Option<u64>,
	#[serde(default)]
	input_modalities:  Option<Vec<OpenAiInputModality>>,
	#[serde(default)]
	supports_tools:    Option<bool>,
	#[serde(default, rename = "created")]
	_created:          Option<u64>,
	#[serde(default, rename = "owned_by")]
	_owned_by:         Option<Str>,
	#[serde(default, rename = "object")]
	_object:           Option<Str>,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum OpenAiInputModality {
	Text,
	Image,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleModelsEnvelope {
	models:          Vec<GoogleModel>,
	#[serde(default)]
	next_page_token: Option<Str>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleModel {
	name: Str,
	#[serde(default)]
	display_name: Option<Str>,
	#[serde(default)]
	base_model_id: Option<Str>,
	#[serde(default)]
	_version: Option<Str>,
	#[serde(default)]
	_description: Option<Str>,
	#[serde(default)]
	supported_generation_methods: Vec<Str>,
	#[serde(default)]
	input_token_limit: Option<u64>,
	#[serde(default)]
	output_token_limit: Option<u64>,
}

#[derive(Deserialize)]
struct AccountEnvelope {
	#[serde(default, alias = "additional_model_options", alias = "data")]
	models:      Vec<AccountModel>,
	#[serde(default)]
	next:        Option<Str>,
	#[serde(default)]
	next_cursor: Option<Str>,
	#[serde(default)]
	has_more:    Option<bool>,
	#[serde(default)]
	last_id:     Option<Str>,
}

impl AccountEnvelope {
	fn cursor(&self) -> ProviderCursor {
		ProviderCursor {
			next:     self.next_cursor.clone().or_else(|| self.next.clone()),
			has_more: self.has_more,
			last_id:  self.last_id.clone(),
		}
	}
}

#[derive(Deserialize)]
struct AccountModel {
	#[serde(default)]
	model:            Option<Str>,
	#[serde(default)]
	slug:             Option<Str>,
	#[serde(default)]
	id:               Option<Str>,
	#[serde(default, alias = "name")]
	display_name:     Option<Str>,
	#[serde(default)]
	family:           Option<Str>,
	#[serde(default)]
	visibility:       Option<Str>,
	#[serde(default)]
	max_input_tokens: Option<u64>,
	#[serde(default)]
	max_tokens:       Option<u64>,
	#[serde(default)]
	capabilities:     Option<AnthropicCapabilities>,
}

#[derive(Deserialize)]
struct AnthropicCapabilities {
	image_input:        Option<AnthropicSupport>,
	structured_outputs: Option<AnthropicSupport>,
	thinking:           Option<AnthropicThinking>,
	effort:             Option<AnthropicEffort>,
}

impl AnthropicCapabilities {
	fn reasoning(&self) -> Availability<ReasoningCapabilities> {
		let thinking = self.thinking.as_ref();
		let supports_thinking = thinking.and_then(|thinking| thinking.supported);
		let effort = self.effort.as_ref();
		let supports_effort = effort.and_then(|effort| effort.supported) == Some(true);
		if supports_thinking != Some(true) && !supports_effort {
			return if supports_thinking == Some(false) {
				Availability::Unsupported
			} else {
				Availability::Unknown
			};
		}
		let mut features = ReasoningFeatureBits::empty();
		if supports_thinking == Some(true) {
			features.insert(ReasoningFeatureBits::VISIBLE | ReasoningFeatureBits::SIGNATURES);
			if thinking
				.and_then(|thinking| thinking.types.as_ref())
				.and_then(|types| types.adaptive.as_ref())
				.and_then(|support| support.supported)
				== Some(true)
			{
				features.insert(ReasoningFeatureBits::ADAPTIVE);
			}
			if thinking
				.and_then(|thinking| thinking.types.as_ref())
				.and_then(|types| types.enabled.as_ref())
				.and_then(|enabled| enabled.supported)
				== Some(true)
			{
				features.insert(ReasoningFeatureBits::BUDGET);
			}
		}
		let efforts: Box<[ReasoningEffort]> = if let Some(effort) = effort.filter(|_| supports_effort)
		{
			features.insert(ReasoningFeatureBits::EFFORT);
			[
				(ReasoningEffort::Low, &effort.low),
				(ReasoningEffort::Medium, &effort.medium),
				(ReasoningEffort::High, &effort.high),
				(ReasoningEffort::Xhigh, &effort.xhigh),
				(ReasoningEffort::Max, &effort.max),
			]
			.into_iter()
			.filter_map(|(level, support)| {
				(support.as_ref().and_then(|support| support.supported) == Some(true)).then_some(level)
			})
			.collect()
		} else {
			Box::new([])
		};
		Availability::Native(ReasoningCapabilities {
			features,
			efforts,
			minimum_budget_tokens: None,
			maximum_budget_tokens: None,
		})
	}
}

#[derive(Deserialize)]
struct AnthropicSupport {
	supported: Option<bool>,
}

#[derive(Deserialize)]
struct AnthropicThinking {
	supported: Option<bool>,
	types:     Option<AnthropicThinkingTypes>,
}

#[derive(Deserialize)]
struct AnthropicThinkingTypes {
	enabled:  Option<AnthropicSupport>,
	adaptive: Option<AnthropicSupport>,
}

#[derive(Deserialize)]
struct AnthropicEffort {
	supported: Option<bool>,
	low:       Option<AnthropicSupport>,
	medium:    Option<AnthropicSupport>,
	high:      Option<AnthropicSupport>,
	xhigh:     Option<AnthropicSupport>,
	max:       Option<AnthropicSupport>,
}

fn invalid_request() -> Error {
	Error::new(
		ErrorKind::InvalidRequest,
		ErrorPhase::Encoding,
		RetryAction::Never,
		ExecutionReceipt::default(),
	)
}

fn protocol_error() -> Error {
	Error::new(
		ErrorKind::Protocol,
		ErrorPhase::Discovery,
		RetryAction::Never,
		ExecutionReceipt::default(),
	)
}

#[cfg(test)]
mod tests {

	use std::time;

	use omp_core::sf;

	use super::*;

	fn request(cursor: Option<&str>) -> DiscoveryRequest {
		DiscoveryRequest {
			provider:  None,
			route:     None,
			cursor:    cursor.map(Str::new),
			page_size: 100,
			operation: None,
		}
	}

	fn decoder(flavor: DiscoveryFlavor, pagination: DiscoveryPagination) -> DiscoveryDecoder {
		DiscoveryDecoder {
			provider: ProviderId::from("provider"),
			route: RouteId::from("route"),
			source: sf!("fixture"),
			pagination,
			request_cursor: None,
			flavor,
			done: false,
		}
	}

	fn discovered(
		flavor: DiscoveryFlavor,
		pagination: DiscoveryPagination,
		fixture: &'static [u8],
	) -> (Vec<DiscoveredModel>, Option<Str>) {
		discovered_for_provider("provider", flavor, pagination, fixture)
	}

	fn discovered_for_provider(
		provider: &str,
		flavor: DiscoveryFlavor,
		pagination: DiscoveryPagination,
		fixture: &'static [u8],
	) -> (Vec<DiscoveredModel>, Option<Str>) {
		let mut decoder = decoder(flavor, pagination);
		decoder.provider = ProviderId::from(provider);
		let mut output = None;
		decoder
			.push(Frame::Raw(Bytes::from_static(fixture)), &mut |event| {
				if let RawEvent::DiscoveredModels { rows, next_cursor } = event {
					output = Some((rows, next_cursor));
				}
			})
			.expect("fixture decodes");
		decoder.finish(&mut |_| {}).expect("decoder completes");
		output.expect("discovery event")
	}
	#[test]
	fn legacy_anthropic_route_migrates_to_paginated_models_api() {
		let spec = DiscoverySpec {
			id:            omp_catalog::DiscoverySpecId::from("fixture"),
			kind:          DiscoveryKind::AccountModels,
			label:         sf!("bootstrap"),
			path:          sf!("/api/claude_cli/bootstrap?model=old-client-default"),
			pagination:    DiscoveryPagination::SinglePage,
			authoritative: false,
			interval:      None,
		};
		let migrated = AccountModelsDiscoveryCodec::from_spec(&spec)
			.unwrap()
			.core
			.spec;
		assert_eq!(migrated.path.as_str(), "/v1/models?limit=1000");
		assert_eq!(
			discovery_uri(
				"https://api.anthropic.com",
				&migrated.path,
				&migrated.pagination,
				&request(Some("page-two"))
			)
			.unwrap()
			.as_str(),
			"https://api.anthropic.com/v1/models?limit=1000&after_id=page-two"
		);
		for auth in [AuthScheme::OAuth, AuthScheme::ApiKey] {
			assert!(
				discovery_headers(&ProviderId::from("anthropic"), Some(auth), &migrated.path)
					.contains(&RequestHeader::new_static("anthropic-version", "2023-06-01"))
			);
		}
		let mut custom = spec;
		custom.path = sf!("/custom/catalog");
		assert_eq!(
			AccountModelsDiscoveryCodec::from_spec(&custom)
				.unwrap()
				.core
				.spec
				.path,
			custom.path
		);
	}

	#[test]
	fn anthropic_models_api_discovers_future_models_with_reported_metadata() {
		let (rows, next) = discovered_for_provider("anthropic", DiscoveryFlavor::Account,
			DiscoveryPagination::Cursor { query_parameter: sf!("after_id") }, br#"{
				"data":[{"id":"claude-opus-99-7","display_name":"Future Opus",
				"max_input_tokens":1000000,"max_tokens":128000,
				"capabilities":{"image_input":{"supported":true},"structured_outputs":{"supported":true},
				"thinking":{"supported":true,"types":{"enabled":{"supported":false},"adaptive":{"supported":true}}},
				"effort":{"supported":true,"low":{"supported":true},"high":{"supported":true},"max":{"supported":false}}}}],
				"has_more":true,"last_id":"claude-opus-99-7"
			}"#);
		assert_eq!(next.as_deref(), Some("claude-opus-99-7"));
		assert_eq!(rows[0].wire_model.as_str(), "claude-opus-99-7");
		let limits = rows[0].declared_limits.unwrap();
		assert_eq!(limits.maximum_input_tokens, Some(1000000));
		assert_eq!(limits.maximum_output_tokens, Some(128000));
		let chat = rows[0]
			.declared_capabilities
			.as_ref()
			.unwrap()
			.chat
			.as_ref()
			.unwrap();
		assert!(matches!(chat.tools, Availability::Native(_)));
		assert!(matches!(chat.image_input, Availability::Native(_)));
		let Availability::Native(reasoning) = &chat.reasoning else {
			panic!("reasoning")
		};
		assert_eq!(&*reasoning.efforts, &[ReasoningEffort::Low, ReasoningEffort::High]);
		assert!(reasoning.features.contains(ReasoningFeatureBits::ADAPTIVE));
		assert!(!reasoning.features.contains(ReasoningFeatureBits::BUDGET));
		assert!(rows[0].declared_pricing.is_empty());
	}

	#[test]
	fn anthropic_nullable_metadata_stays_unknown_and_missing_page_cursor_fails() {
		let (rows, next) = discovered_for_provider(
			"anthropic",
			DiscoveryFlavor::Account,
			DiscoveryPagination::Cursor { query_parameter: sf!("after_id") },
			br#"{"data":[{"id":"future","capabilities":null,"max_tokens":null}],"has_more":false}"#,
		);
		assert!(next.is_none());
		assert!(rows[0].declared_limits.is_none());
		let chat = rows[0]
			.declared_capabilities
			.as_ref()
			.unwrap()
			.chat
			.as_ref()
			.unwrap();
		assert!(chat.reasoning.is_unknown());
		assert!(chat.image_input.is_unknown());
		let decoder = decoder(DiscoveryFlavor::Account, DiscoveryPagination::Cursor {
			query_parameter: sf!("after_id"),
		});
		assert!(
			decoder
				.decode_account(br#"{"data":[{"id":"future"}],"has_more":true}"#)
				.is_err()
		);
	}

	#[test]
	fn standalone_constructors_require_their_exact_discovery_kind() {
		let mut spec = DiscoverySpec {
			id:            omp_catalog::DiscoverySpecId::from("fixture"),
			kind:          DiscoveryKind::OpenAiModels,
			label:         sf!("fixture"),
			path:          sf!("/models"),
			pagination:    DiscoveryPagination::SinglePage,
			authoritative: false,
			interval:      Some(time::Duration::from_secs(5)),
		};
		OpenAiModelsDiscoveryCodec::from_spec(&spec).expect("OpenAI models kind");
		assert_eq!(
			AccountModelsDiscoveryCodec::from_spec(&spec)
				.expect_err("account codec rejects OpenAI kind")
				.kind,
			ErrorKind::InvalidRequest,
		);
		spec.kind = DiscoveryKind::AccountModels;
		AccountModelsDiscoveryCodec::from_spec(&spec).expect("account models kind");
		spec.kind = DiscoveryKind::OllamaTags;
		OllamaTagsDiscoveryCodec::from_spec(&spec).expect("Ollama tags kind");
		spec.kind = DiscoveryKind::Specialized;
		assert_eq!(
			OllamaTagsDiscoveryCodec::from_spec(&spec)
				.expect_err("specialized Ollama discovery remains codec-owned")
				.kind,
			ErrorKind::InvalidRequest,
		);
	}

	#[test]
	fn anthropic_oauth_discovery_uses_claude_code_request_headers() {
		let headers = discovery_headers(
			&ProviderId::from("anthropic"),
			Some(AuthScheme::OAuth),
			"/api/claude_cli/bootstrap",
		);
		assert_eq!(headers.as_ref(), &[
			RequestHeader::new_static("accept", "application/json, text/plain, */*"),
			RequestHeader::new_static("content-type", "application/json"),
			RequestHeader::new_static("anthropic-beta", "oauth-2025-04-20"),
			RequestHeader::new_static("user-agent", "claude-code/2.1.258"),
		]);
		assert!(
			discovery_headers(
				&ProviderId::from("anthropic"),
				Some(AuthScheme::ApiKey),
				"/api/claude_cli/bootstrap"
			)
			.is_empty()
		);
		assert!(
			discovery_headers(&ProviderId::from("openai"), Some(AuthScheme::OAuth), "/v1/models")
				.is_empty()
		);
	}

	#[test]
	fn openai_models_fixture_preserves_only_declared_raw_identity() {
		let (rows, next) = discovered(
			DiscoveryFlavor::OpenAi,
			DiscoveryPagination::SinglePage,
			include_bytes!("../../../../fixtures/llm-oracle/openai/chat/response.models.json"),
		);
		assert_eq!(next, None);
		assert_eq!(rows.len(), 2);
		assert_eq!(rows[0].wire_model.as_str(), "gpt-4.1");
		assert_eq!(rows[1].wire_model.as_str(), "gpt-4.1-mini");
		assert_eq!(rows[0].provider.as_str(), "provider");
		assert_eq!(rows[0].route.as_str(), "route");
		assert_eq!(rows[0].source.as_str(), "fixture");
		assert!(rows[0].declared_operations.is_empty());
		assert_eq!(rows[0].declared_class, None);
		assert_eq!(rows[0].declared_capabilities, None);
	}

	#[test]
	fn openai_model_list_preserves_enriched_gateway_card_fields() {
		let (rows, next) = discovered(
			DiscoveryFlavor::OpenAi,
			DiscoveryPagination::SinglePage,
			br#"{"object":"list","data":[{
				"id":"openai/gpt-5.6-sol",
				"object":"model",
				"owned_by":"openai",
				"display_name":"GPT-5.6 Sol",
				"context_length":1000000,
				"max_output_tokens":128000,
				"input_modalities":["text","image"],
				"supports_tools":false
			}]}"#,
		);
		assert_eq!(next, None);
		let [row] = rows.as_slice() else {
			panic!("one enriched model row");
		};
		assert_eq!(row.wire_model.as_str(), "openai/gpt-5.6-sol");
		assert_eq!(row.display_name.as_deref(), Some("GPT-5.6 Sol"));
		assert_eq!(
			row.declared_limits,
			Some(ModelLimits {
				context_window:        Some(1_000_000),
				maximum_input_tokens:  None,
				maximum_output_tokens: Some(128_000),
				maximum_batch:         None,
			})
		);
		assert!(row.declared_operations.contains_kind(OperationKind::Chat));
		let chat = row
			.declared_capabilities
			.as_ref()
			.and_then(|capabilities| capabilities.chat.as_ref())
			.expect("enriched row has chat evidence");
		assert!(matches!(&chat.tools, Availability::Unsupported));
		assert!(matches!(
			&chat.input_modalities,
			Availability::Native(modalities)
				if modalities.contains(ModalityBits::TEXT | ModalityBits::IMAGE)
		));
	}

	#[test]
	fn google_constructor_requires_exact_kind_and_page_token_query_is_opaque() {
		let mut spec = DiscoverySpec {
			id:            omp_catalog::DiscoverySpecId::from("google-models"),
			kind:          DiscoveryKind::GoogleModels,
			label:         "google-list-models".into(),
			path:          "/v1beta/models".into(),
			pagination:    DiscoveryPagination::Cursor { query_parameter: "pageToken".into() },
			authoritative: false,
			interval:      Some(time::Duration::from_secs(5)),
		};
		GoogleModelsDiscoveryCodec::from_spec(&spec).expect("Google discovery kind constructs");
		spec.kind = DiscoveryKind::OpenAiModels;
		assert_eq!(
			GoogleModelsDiscoveryCodec::from_spec(&spec)
				.expect_err("wrong discovery kind is rejected")
				.kind,
			ErrorKind::InvalidRequest,
		);
		let uri = discovery_uri(
			"https://generativelanguage.googleapis.com",
			"/v1beta/models",
			&DiscoveryPagination::Cursor { query_parameter: "pageToken".into() },
			&request(Some("opaque +/% token")),
		)
		.expect("Google cursor URI");
		assert_eq!(
			uri.as_str(),
			"https://generativelanguage.googleapis.com/v1beta/models?pageToken=opaque+%2B%2F%25+token",
		);
	}

	#[test]
	fn google_list_models_fixture_preserves_declared_operations_limits_and_cursor() {
		let (rows, next) = discovered(
			DiscoveryFlavor::Google,
			DiscoveryPagination::Cursor { query_parameter: sf!("pageToken") },
			include_bytes!("fixtures/google_list_models.json"),
		);
		assert_eq!(next.as_ref().map(Str::as_str), Some("page-token-REDACTED"));
		assert_eq!(rows.len(), 2);
		assert_eq!(rows[0].wire_model.as_str(), "gemini-1.5-flash-001");
		assert_eq!(rows[0].display_name.as_ref().map(Str::as_str), Some("Gemini 1.5 Flash"),);
		assert_eq!(
			rows[0]
				.aliases
				.iter()
				.map(WireModelId::as_str)
				.collect::<Vec<_>>(),
			vec!["gemini-1.5-flash"],
		);
		assert!(
			rows[0]
				.declared_operations
				.contains_kind(OperationKind::Chat)
		);
		assert!(
			rows[0]
				.declared_operations
				.contains_kind(OperationKind::CountTokens)
		);
		assert!(
			!rows[0]
				.declared_operations
				.contains_kind(OperationKind::Embed)
		);
		assert_eq!(
			rows[0].declared_limits,
			Some(ModelLimits {
				context_window:        None,
				maximum_input_tokens:  Some(1_048_576),
				maximum_output_tokens: Some(8_192),
				maximum_batch:         None,
			}),
		);
		assert_eq!(rows[1].wire_model.as_str(), "text-embedding-004");
		assert!(
			rows[1]
				.declared_operations
				.contains_kind(OperationKind::Embed)
		);
		assert!(
			!rows[1]
				.declared_operations
				.contains_kind(OperationKind::Chat)
		);
		assert_eq!(rows[0].declared_class, None);
		assert_eq!(rows[0].declared_capabilities, None);
		assert_eq!(rows[0].availability, None);
	}

	#[test]
	fn account_models_fixture_preserves_provider_fields_and_cursor() {
		let (rows, next) = discovered_for_provider(
			"anthropic",
			DiscoveryFlavor::Account,
			DiscoveryPagination::Cursor { query_parameter: sf!("after") },
			include_bytes!("../../../../fixtures/llm-oracle/openai/chat/response.account_models.json"),
		);
		assert_eq!(next.as_ref().map(Str::as_str), Some("retired-account-model"));
		assert_eq!(rows[0].wire_model.as_str(), "gpt-5.2-codex");
		assert_eq!(rows[0].display_name.as_ref().map(Str::as_str), Some("GPT-5.2 Codex"));
		assert_eq!(rows[0].declared_class.as_ref().map(ClassId::as_str), Some("gpt-5-codex"));
		assert_eq!(rows[0].availability, Some(ModelAvailability::Available));
		assert_eq!(rows[1].availability, Some(ModelAvailability::Disabled));
	}

	#[test]
	fn ollama_tags_fixture_uses_exact_wire_model_and_declared_class() {
		let (rows, next) = discovered(
			DiscoveryFlavor::Ollama,
			DiscoveryPagination::SinglePage,
			include_bytes!("../../../../fixtures/llm-oracle/openai/chat/response.ollama_tags.json"),
		);
		assert_eq!(next, None);
		assert_eq!(rows.len(), 1);
		assert_eq!(rows[0].wire_model.as_str(), "qwen3:8b");
		assert_eq!(rows[0].display_name.as_ref().map(Str::as_str), Some("qwen3:8b"));
		assert_eq!(rows[0].declared_class.as_ref().map(ClassId::as_str), Some("qwen3"));
	}

	#[test]
	fn pagination_lowers_exact_query_and_single_page_rejects_cursor() {
		let cursor_uri = discovery_uri(
			"https://example.test/v1",
			"models",
			&DiscoveryPagination::Cursor { query_parameter: sf!("after") },
			&request(Some("opaque +/% cursor")),
		)
		.expect("cursor URI");
		assert_eq!(
			cursor_uri.as_str(),
			"https://example.test/v1/models?after=opaque+%2B%2F%25+cursor"
		);
		let page_uri = discovery_uri(
			"https://example.test",
			"/models",
			&DiscoveryPagination::PageNumber { query_parameter: sf!("page"), first_page: 1 },
			&request(Some("7")),
		)
		.expect("page URI");
		assert_eq!(page_uri.as_str(), "https://example.test/models?page=7");
		let error = discovery_uri(
			"https://example.test",
			"/models",
			&DiscoveryPagination::SinglePage,
			&request(Some("unexpected")),
		)
		.expect_err("single-page discovery rejects a cursor");
		assert_eq!(error.kind, ErrorKind::InvalidRequest);
	}

	#[test]
	fn anthropic_bootstrap_models_use_server_ids_and_names() {
		let uri = discovery_uri(
			"https://api.anthropic.com",
			"/api/claude_cli/bootstrap?model=claude-sonnet-5",
			&DiscoveryPagination::SinglePage,
			&request(None),
		)
		.expect("Anthropic bootstrap URI");
		assert_eq!(
			uri.as_str(),
			"https://api.anthropic.com/api/claude_cli/bootstrap?model=claude-sonnet-5",
		);
		let (rows, next) = discovered_for_provider(
			"anthropic",
			DiscoveryFlavor::Account,
			DiscoveryPagination::SinglePage,
			br#"{
				"client_data": {"opaque": true},
				"additional_model_options": [
					{"model":"claude-fable-5-1","name":"Claude Fable 5.1","description":""},
					{"model":"claude-opus-5","name":"Claude Opus 5","description":""}
				]
			}"#,
		);
		assert_eq!(next, None);
		assert_eq!(rows.len(), 2);
		assert_eq!(rows[0].wire_model.as_str(), "claude-fable-5-1");
		assert_eq!(rows[0].display_name.as_ref().map(Str::as_str), Some("Claude Fable 5.1"),);
		assert!(
			rows[0]
				.declared_operations
				.contains_kind(OperationKind::Chat)
		);
		assert_eq!(rows[1].wire_model.as_str(), "claude-opus-5");
	}

	#[test]
	fn anthropic_bootstrap_omits_client_update_control_rows() {
		let (rows, next) = discovered(
			DiscoveryFlavor::Account,
			DiscoveryPagination::SinglePage,
			br#"{
				"additional_model_options": [
					{"model":"claude-fable-5-1[1m]","name":"Fable 5.1"},
					{"model":"cc-update-required-1","name":"Update Claude Code"}
				]
			}"#,
		);
		assert_eq!(next, None);
		assert_eq!(rows.len(), 1);
		assert_eq!(rows[0].wire_model.as_str(), "claude-fable-5-1[1m]");
	}

	#[test]
	fn known_model_field_with_wrong_type_is_rejected() {
		let mut decoder = decoder(DiscoveryFlavor::OpenAi, DiscoveryPagination::SinglePage);
		let error = decoder
			.push(
				Frame::Raw(Bytes::from_static(br#"{"data":[{"id":7,"created":"yesterday"}]}"#)),
				&mut |_| {},
			)
			.expect_err("typed model fields reject wrong types");
		assert_eq!(error.kind, ErrorKind::Protocol);
		assert_eq!(error.phase, ErrorPhase::Discovery);
	}
}
