//! Typed `ChatGPT` Codex Responses Lite, WebSocket, continuation, and
//! discovery shapes.

use omp_core::{Str, sf};
use serde::{Deserialize, Serialize};

use super::{
	Codec, DecodeContext, Decoder, DecoderState, EncodeContext, EncodedRequest, RawEvent,
	RequestHeader, RequestMethod, SizeBounds,
	openai_responses::{
		OpenAiResponsesCodec, OpenAiResponsesDecoder, OpenAiResponsesOptions, ResponsesContentKind,
		ResponsesEncodeError, ResponsesInputContent, ResponsesInputItem, ResponsesInputItemKind,
		ResponsesMetadata, ResponsesMetadataValue, ResponsesNamedToolKind, ResponsesOutputItem,
		ResponsesOutputItemKind, ResponsesReasoning, ResponsesRequest, ResponsesRole,
		ResponsesStreamEvent, ResponsesStreamEventKind, ResponsesStreamOptions, ResponsesTool,
		ResponsesToolChoice, ResponsesToolChoiceMode, ResponsesToolKind, hosted_image_body,
	},
};
use crate::{
	call::{AccountRoutingContext as CallAccountRoutingContext, OperationCall},
	catalog::{
		CodexTransportPreference, ModelCapabilities as CatalogModelCapabilities, OperationBits,
		OperationKind, ProviderId, RouteId,
	},
	error::Error,
	transport::Frame,
};

/// Codex Desktop protocol version shared by chat, discovery, and live voice.
/// Codex CLI version presented to `chatgpt.com`. The `/codex/models` listing is
/// tailored to this value: the backend withholds models it considers too new
/// for the declared client (GPT-6 Astra appeared only from 0.153.x), so keep it
/// close to the current Codex CLI release when new subscription models fail to
/// show up.
pub const CODEX_CLIENT_VERSION: &str = "0.153.1";
const CODEX_ORIGINATOR: &str = "omp";
const CODEX_DISCOVERY_SOURCE: &str = "openai_codex_models";
const CODEX_RESIDENCY_HEADER: &str = "x-openai-internal-codex-residency";

/// Codex wire transport selected for one attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodexWireTransport {
	/// HTTP POST with SSE response framing.
	Http,
	/// Reused WebSocket carrying `response.create` frames.
	WebSocket,
}

/// Stable session and turn identity supplied outside the codec.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CodexRequestIdentity {
	/// Stable conversation identity.
	pub session_id: Str,
	/// Stable turn identity; retries reuse it.
	pub turn_id:    Str,
	/// Optional account identity that contains no bearer bytes.
	pub account_id: Option<Str>,
	/// Optional originator label.
	pub originator: Option<Str>,
}

/// Typed Codex request options.
#[derive(Clone, Debug, Default)]
pub struct OpenAiCodexOptions {
	/// Route-selected Responses Lite mode.
	pub responses_lite:       bool,
	/// Selected HTTP or WebSocket transport.
	pub transport:            Option<CodexWireTransport>,
	/// Stable request identity.
	pub identity:             Option<CodexRequestIdentity>,
	/// Opaque caller-owned turn metadata with typed JSON scalar values.
	pub turn_metadata:        ResponsesMetadata,
	/// Deliver concurrent reasoning summaries using Codex's sequential-cutoff
	/// contract.
	pub concurrent_summaries: bool,
	/// Base Responses options.
	pub responses:            OpenAiResponsesOptions,
}

/// Responses Lite transformation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CodexRequestError {
	/// A forced tool does not exist in the request.
	UnknownForcedTool,
	/// Continuation is not a strict append of the authoritative prior request
	/// and output.
	StaleContinuation,
	/// A native item required for replay lacks a stable identity.
	MissingItemIdentity,
}

fn apply_all_turns_context_policy(request: &mut ResponsesRequest, supported: bool) {
	if !supported
		&& let Some(reasoning) = request.reasoning.as_mut()
		&& reasoning.context.as_deref() == Some("all_turns")
	{
		reasoning.context = None;
	}
}

/// Applies the Codex full Responses contract and optional Responses Lite
/// rewrite.
pub fn transform_codex_request(
	request: &mut ResponsesRequest,
	responses_lite: bool,
	concurrent_summaries: bool,
) -> Result<(), CodexRequestError> {
	request.store = false;
	request.stream = true;
	if !request
		.include
		.iter()
		.any(|value| value == "reasoning.encrypted_content")
	{
		request.include.push(sf!("reasoning.encrypted_content"));
	}
	if let Some(reasoning) = request.reasoning.as_mut()
		&& reasoning.effort.is_some()
		&& reasoning.summary.is_none()
	{
		reasoning.summary = Some(Some(sf!("auto")));
	}
	if concurrent_summaries {
		request.stream_options = Some(ResponsesStreamOptions {
			include_obfuscation:        None,
			reasoning_summary_delivery: Some(sf!("sequential_cutoff")),
		});
	}
	for item in &mut request.input {
		if item.role != Some(ResponsesRole::Assistant) {
			continue;
		}
		if let Some(parts) = item.content.parts_mut() {
			for content in parts {
				if content.kind == ResponsesContentKind::InputText {
					content.kind = ResponsesContentKind::OutputText;
				}
			}
		}
	}
	for tool in &mut request.tools {
		if tool.kind == ResponsesToolKind::Function {
			tool.strict = Some(false);
		}
	}
	if !responses_lite {
		return Ok(());
	}

	request.temperature = None;
	request.top_p = None;
	request.presence_penalty = None;
	request.frequency_penalty = None;
	request.max_output_tokens = None;
	request.parallel_tool_calls = Some(false);
	request.prompt_cache_retention = None;
	if let Some(reasoning) = request.reasoning.as_mut() {
		reasoning.context = Some(sf!("all_turns"));
	} else {
		request.reasoning = Some(ResponsesReasoning {
			effort:  None,
			summary: None,
			mode:    None,
			context: Some(sf!("all_turns")),
		});
	}
	for item in &mut request.input {
		if item.kind != Some(ResponsesInputItemKind::ComputerCall) {
			item.id = None;
		}
		if let Some(parts) = item.content.parts_mut() {
			for content in parts {
				content.detail = None;
			}
		}
	}
	request
		.input
		.retain(|item| item.kind != Some(ResponsesInputItemKind::ItemReference));

	let instructions = request.instructions.take();
	let selected = match request.tool_choice.as_ref() {
		Some(ResponsesToolChoice::Named(choice)) => Some((choice.kind, choice.name.clone())),
		_ => None,
	};
	let mut additional_tools = Vec::new();
	if let Some((kind, name)) = selected {
		additional_tools.extend(
			request
				.tools
				.iter()
				.filter(|tool| tool_matches_choice(tool, kind, name.as_ref()))
				.cloned(),
		);
		if additional_tools.is_empty()
			&& matches!(
				kind,
				ResponsesNamedToolKind::Function
					| ResponsesNamedToolKind::Custom
					| ResponsesNamedToolKind::Computer
			) {
			return Err(CodexRequestError::UnknownForcedTool);
		}
	} else {
		additional_tools.extend(
			request
				.tools
				.iter()
				.filter(|tool| {
					matches!(
						tool.kind,
						ResponsesToolKind::Function
							| ResponsesToolKind::Custom
							| ResponsesToolKind::Computer
					)
				})
				.cloned(),
		);
	}
	request.tools.clear();
	request.additional_tools.clear();
	if !additional_tools.is_empty() {
		request.input.insert(0, ResponsesInputItem {
			kind: Some(ResponsesInputItemKind::AdditionalTools),
			id: None,
			role: Some(ResponsesRole::Developer),
			content: ResponsesInputContent::default(),
			name: None,
			call_id: None,
			arguments: None,
			input: None,
			output: None,
			summary: None,
			encrypted_content: None,
			actions: Vec::new(),
			pending_safety_checks: Vec::new(),
			acknowledged_safety_checks: Vec::new(),
			status: None,
			tools: additional_tools,
			cache_control: None,
			metadata: ResponsesMetadata::new(),
		});
	}
	if let Some(instructions) = instructions {
		let position = usize::from(
			!request.input.is_empty()
				&& request.input[0].kind == Some(ResponsesInputItemKind::AdditionalTools),
		);
		let mut message = ResponsesInputItem::message(ResponsesRole::Developer, vec![
			super::openai_responses::ResponsesContent::input_text(instructions),
		]);
		message.kind = Some(ResponsesInputItemKind::Message);
		request.input.insert(position, message);
	}
	request.tool_choice = Some(match request.tool_choice.take() {
		Some(ResponsesToolChoice::Mode(
			mode @ (ResponsesToolChoiceMode::None | ResponsesToolChoiceMode::Required),
		)) => ResponsesToolChoice::Mode(mode),
		Some(ResponsesToolChoice::Named(choice))
			if matches!(
				choice.kind,
				ResponsesNamedToolKind::Function
					| ResponsesNamedToolKind::Custom
					| ResponsesNamedToolKind::Computer
			) =>
		{
			ResponsesToolChoice::Mode(ResponsesToolChoiceMode::Required)
		},
		_ => ResponsesToolChoice::Mode(ResponsesToolChoiceMode::Auto),
	});
	Ok(())
}

fn tool_matches_choice(
	tool: &ResponsesTool,
	kind: ResponsesNamedToolKind,
	name: Option<&Str>,
) -> bool {
	match kind {
		ResponsesNamedToolKind::Function => {
			tool.kind == ResponsesToolKind::Function && tool.name.as_ref() == name
		},
		ResponsesNamedToolKind::Custom => {
			tool.kind == ResponsesToolKind::Custom && tool.name.as_ref() == name
		},
		ResponsesNamedToolKind::Computer => tool.kind == ResponsesToolKind::Computer,
		ResponsesNamedToolKind::WebSearch => tool.kind == ResponsesToolKind::WebSearch,
		ResponsesNamedToolKind::FileSearch => tool.kind == ResponsesToolKind::FileSearch,
		ResponsesNamedToolKind::CodeInterpreter => tool.kind == ResponsesToolKind::CodeInterpreter,
	}
}

/// Applies stable Codex identity and WebSocket-mirrored session metadata.
pub fn apply_codex_client_metadata(
	request: &mut ResponsesRequest,
	identity: &CodexRequestIdentity,
	transport: CodexWireTransport,
	responses_lite: bool,
	turn_state: Option<&str>,
	turn_metadata: &ResponsesMetadata,
) {
	let mut metadata = turn_metadata.clone();
	metadata.insert(sf!("session_id"), ResponsesMetadataValue::String(identity.session_id.clone()));
	metadata.insert(sf!("turn_id"), ResponsesMetadataValue::String(identity.turn_id.clone()));
	// The server's client_metadata contract requires string values, including
	// flags.
	metadata.insert(sf!("responses_lite"), ResponsesMetadataValue::String(sf!("{responses_lite}")));
	if let Some(account) = &identity.account_id {
		metadata.insert(sf!("account_id"), ResponsesMetadataValue::String(account.clone()));
	}
	if let Some(originator) = &identity.originator {
		metadata.insert(sf!("originator"), ResponsesMetadataValue::String(originator.clone()));
	}
	if transport == CodexWireTransport::WebSocket {
		metadata.insert(sf!("transport"), ResponsesMetadataValue::String(sf!("websocket")));
		if let Some(state) = turn_state {
			metadata.insert(sf!("turn_state"), ResponsesMetadataValue::String(Str::new(state)));
		}
	}
	request.client_metadata = Some(metadata);
}
/// Adds a region-pinned workspace's residency to Codex request headers
/// without replacing a caller-supplied value.
pub fn apply_codex_residency_header(
	headers: &mut Vec<RequestHeader>,
	account: Option<&CallAccountRoutingContext>,
) {
	if headers
		.iter()
		.any(|header| header.name.eq_ignore_ascii_case(CODEX_RESIDENCY_HEADER))
	{
		return;
	}
	let Some(residency) = account.and_then(|account| account.region.as_ref()) else {
		return;
	};
	headers.push(RequestHeader {
		name:  Str::new_static(CODEX_RESIDENCY_HEADER),
		value: Str::new(residency.as_str()),
	});
}

/// Resolves a configured endpoint to `/codex/responses` without duplicating the
/// suffix.
pub fn resolve_codex_responses_url(base_url: &str) -> Str {
	let base = base_url.trim_end_matches('/');
	if base.ends_with("/codex/responses") {
		Str::new(base)
	} else if base.ends_with("/codex") {
		sf!("{base}/responses")
	} else {
		sf!("{base}/codex/responses")
	}
}

/// Converts a Codex HTTP endpoint to a WebSocket endpoint.
pub fn codex_websocket_url(url: &str) -> Result<Str, CodexWebSocketProtocolError> {
	if let Some(rest) = url.strip_prefix("https://") {
		Ok(sf!("wss://{rest}"))
	} else if let Some(rest) = url.strip_prefix("http://") {
		Ok(sf!("ws://{rest}"))
	} else if url.starts_with("wss://") || url.starts_with("ws://") {
		Ok(Str::new(url))
	} else {
		Err(CodexWebSocketProtocolError::InvalidEndpoint)
	}
}

/// WebSocket request discriminator.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CodexResponseCreateKind {
	/// Create a response.
	#[serde(rename = "response.create")]
	ResponseCreate,
}

/// Typed Codex WebSocket request frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodexResponseCreate {
	/// Frame discriminator.
	#[serde(rename = "type")]
	pub kind:    CodexResponseCreateKind,
	/// Flattened typed Responses request.
	#[serde(flatten)]
	pub request: ResponsesRequest,
}

/// Protocol violation that makes continued socket reuse unsafe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodexWebSocketProtocolError {
	/// Endpoint had no HTTP or WebSocket scheme.
	InvalidEndpoint,
	/// A second response was interleaved into the active request.
	InterleavedResponse,
	/// Sequence numbers regressed or repeated.
	RegressingSequence,
	/// Continuation was not a strict input append.
	StaleContinuation,
}

/// Result of routing one inbound WebSocket frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodexFrameDisposition {
	/// Ignore a stale, pre-handshake, or post-terminal frame.
	Drop,
	/// Accept a correlated nonterminal frame.
	Accept,
	/// Accept the one authoritative terminal frame.
	Terminal,
}

/// Per-request WebSocket frame correlator.
#[derive(Clone, Debug, Default)]
pub struct CodexFrameRouter {
	active_response_id: Option<Str>,
	last_sequence:      Option<u64>,
	terminal:           bool,
}

impl CodexFrameRouter {
	/// Routes a typed Responses frame while enforcing response and sequence
	/// correlation.
	pub fn route(
		&mut self,
		event: &ResponsesStreamEvent,
	) -> Result<CodexFrameDisposition, CodexWebSocketProtocolError> {
		if self.terminal {
			return Ok(CodexFrameDisposition::Drop);
		}
		let response_id = event
			.response
			.as_ref()
			.and_then(|response| response.id.as_ref());
		if self.active_response_id.is_none() {
			if event.kind != ResponsesStreamEventKind::Created {
				return Ok(CodexFrameDisposition::Drop);
			}
			let Some(response_id) = response_id else {
				return Ok(CodexFrameDisposition::Drop);
			};
			self.active_response_id = Some(response_id.clone());
		} else if let Some(response_id) = response_id
			&& self.active_response_id.as_ref() != Some(response_id)
		{
			return Err(CodexWebSocketProtocolError::InterleavedResponse);
		}
		if let Some(sequence) = event.sequence_number {
			if self
				.last_sequence
				.is_some_and(|previous| sequence <= previous)
			{
				return Err(CodexWebSocketProtocolError::RegressingSequence);
			}
			self.last_sequence = Some(sequence);
		}
		if is_terminal(event.kind) {
			self.terminal = true;
			Ok(CodexFrameDisposition::Terminal)
		} else {
			Ok(CodexFrameDisposition::Accept)
		}
	}

	/// Borrows the active response identity.
	pub fn active_response_id(&self) -> Option<&str> {
		self.active_response_id.as_deref()
	}
}

const fn is_terminal(kind: ResponsesStreamEventKind) -> bool {
	matches!(
		kind,
		ResponsesStreamEventKind::Completed
			| ResponsesStreamEventKind::Incomplete
			| ResponsesStreamEventKind::Done
			| ResponsesStreamEventKind::Failed
			| ResponsesStreamEventKind::Cancelled
			| ResponsesStreamEventKind::Error
	)
}

/// Successful continuation state retained after an authoritative terminal
/// response.
#[derive(Clone, Debug)]
pub struct CodexContinuationState {
	request:        ResponsesRequest,
	response_id:    Str,
	response_items: Vec<ResponsesOutputItem>,
}

impl CodexContinuationState {
	/// Captures authoritative successful state.
	pub const fn new(
		request: ResponsesRequest,
		response_id: Str,
		response_items: Vec<ResponsesOutputItem>,
	) -> Self {
		Self { request, response_id, response_items }
	}

	/// Builds a strict delta-only `response.create` frame. The per-turn service
	/// tier does not break an otherwise identical continuation chain.
	pub fn continuation_frame(
		&self,
		current: &ResponsesRequest,
	) -> Result<CodexResponseCreate, CodexRequestError> {
		let previous_len = self.request.input.len();
		let replayed_len = self.response_items.len();
		let mut previous_shape = self.request.clone();
		previous_shape.input.clear();
		previous_shape.client_metadata = None;
		previous_shape.service_tier = None;
		let mut current_shape = current.clone();
		current_shape.input.clear();
		current_shape.client_metadata = None;
		current_shape.service_tier = None;
		if current_shape != previous_shape
			|| current.input.len() <= previous_len.saturating_add(replayed_len)
		{
			return Err(CodexRequestError::StaleContinuation);
		}
		if current.input[..previous_len] != self.request.input[..] {
			return Err(CodexRequestError::StaleContinuation);
		}
		for (position, output) in self.response_items.iter().enumerate() {
			let Some(current_item) = current.input.get(previous_len + position) else {
				return Err(CodexRequestError::StaleContinuation);
			};
			if !output_matches_replay(output, current_item) {
				return Err(CodexRequestError::StaleContinuation);
			}
		}
		let mut request = current.clone();
		request.input = current.input[previous_len + replayed_len..].to_vec();
		request.previous_response_id = Some(self.response_id.clone());
		Ok(CodexResponseCreate { kind: CodexResponseCreateKind::ResponseCreate, request })
	}
}

fn output_matches_replay(output: &ResponsesOutputItem, input: &ResponsesInputItem) -> bool {
	match output.kind {
		ResponsesOutputItemKind::FunctionCall => {
			input.kind == Some(ResponsesInputItemKind::FunctionCall)
				&& input.call_id == output.call_id
				&& input.name == output.name
				&& input.arguments == output.arguments
		},
		ResponsesOutputItemKind::CustomToolCall => {
			input.kind == Some(ResponsesInputItemKind::CustomToolCall)
				&& input.call_id == output.call_id
				&& input.name == output.name
				&& input.input == output.input
		},
		ResponsesOutputItemKind::ComputerCall => {
			input.kind == Some(ResponsesInputItemKind::ComputerCall)
				&& input.call_id == output.call_id
				&& input.actions == output.actions
		},
		_ => false,
	}
}

/// Produces a full-context HTTP fallback body by removing WebSocket-only
/// continuation state.
pub fn codex_http_fallback_body(full_request: &ResponsesRequest) -> ResponsesRequest {
	let mut request = full_request.clone();
	request.previous_response_id = None;
	request
}

/// WebSocket failure class used only to report replay/fallback evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodexWebSocketFailure {
	/// Caller cancellation.
	Cancelled,
	/// Socket cannot be reused and HTTP fallback may be considered.
	ConnectionFatal,
	/// Retryable empty transport attempt.
	RetryableTransport,
	/// Retryable provider signal that must not silently switch transport.
	RetryableProvider,
	/// Nonretryable provider failure.
	Provider,
}

/// Delivery facts required by the outer retry layer.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CodexReplaySafety {
	/// Ordinary output was delivered.
	pub committed:           bool,
	/// A tool call was delivered to the consumer.
	pub delivered_tool_call: bool,
	/// An authoritative terminal event was delivered.
	pub terminal:            bool,
	/// Request body evidence permits a fresh replay.
	pub replayable_body:     bool,
}

/// Evidence-only fallback recommendation; this module never performs retry or
/// I/O.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodexFallbackEvidence {
	/// Outer policy may reconnect WebSocket within its explicit budget.
	ReconnectWebSocket,
	/// Outer policy may replay the authoritative full body over HTTP.
	ReplayOverHttp,
	/// Surface the original failure.
	Surface,
	/// Surface cancellation.
	Cancelled,
}

/// Classifies fallback evidence without performing retry.
pub const fn classify_codex_fallback(
	failure: CodexWebSocketFailure,
	safety: CodexReplaySafety,
	retries: u32,
	retry_budget: u32,
) -> CodexFallbackEvidence {
	if matches!(failure, CodexWebSocketFailure::Cancelled) {
		return CodexFallbackEvidence::Cancelled;
	}
	if safety.committed || safety.delivered_tool_call || safety.terminal || !safety.replayable_body {
		return CodexFallbackEvidence::Surface;
	}
	match failure {
		CodexWebSocketFailure::ConnectionFatal => CodexFallbackEvidence::ReplayOverHttp,
		CodexWebSocketFailure::RetryableTransport if retries < retry_budget => {
			CodexFallbackEvidence::ReconnectWebSocket
		},
		CodexWebSocketFailure::RetryableTransport => CodexFallbackEvidence::ReplayOverHttp,
		CodexWebSocketFailure::RetryableProvider | CodexWebSocketFailure::Provider => {
			CodexFallbackEvidence::Surface
		},
		CodexWebSocketFailure::Cancelled => CodexFallbackEvidence::Cancelled,
	}
}

/// Codex codec combining Responses lowering with Lite and WebSocket shaping.
#[derive(Clone, Debug, Default)]
pub struct OpenAiCodexCodec {
	responses: OpenAiResponsesCodec,
	options:   OpenAiCodexOptions,
}

impl OpenAiCodexCodec {
	/// Constructs a typed Codex codec.
	pub fn new(options: OpenAiCodexOptions) -> Self {
		Self { responses: OpenAiResponsesCodec::new(options.responses.clone()), options }
	}

	/// Borrows the underlying Responses codec.
	pub const fn responses(&self) -> &OpenAiResponsesCodec {
		&self.responses
	}

	/// Constructs a fresh shared Responses event decoder.
	pub fn responses_decoder(&self) -> OpenAiResponsesDecoder {
		OpenAiResponsesDecoder::default()
	}

	/// Borrows typed Codex options.
	pub const fn options(&self) -> &OpenAiCodexOptions {
		&self.options
	}
}

#[derive(Deserialize)]
#[serde(untagged)]
enum CodexModelsPayload {
	Rows(Vec<CodexModelRow>),
	Envelope(CodexModelsEnvelope),
}

#[derive(Deserialize)]
struct CodexModelsEnvelope {
	#[serde(default)]
	models: Option<Box<CodexModelsPayload>>,
	#[serde(default)]
	data:   Option<Box<CodexModelsPayload>>,
	#[serde(default)]
	result: Option<Box<CodexModelsPayload>>,
	#[serde(default)]
	items:  Option<Box<CodexModelsPayload>>,
}

impl CodexModelsPayload {
	fn into_rows(self) -> Option<Vec<CodexModelRow>> {
		match self {
			Self::Rows(rows) => Some(rows),
			Self::Envelope(envelope) => {
				[envelope.data, envelope.models, envelope.result, envelope.items]
					.into_iter()
					.flatten()
					.find_map(|payload| payload.into_rows())
			},
		}
	}
}

#[derive(Deserialize)]
struct CodexModelRow {
	#[serde(default)]
	slug: Option<Str>,
	#[serde(default)]
	id: Option<Str>,
	#[serde(default)]
	display_name: Option<Str>,
	#[serde(default)]
	visibility: Option<Str>,
	#[serde(default)]
	context_window: Option<u64>,
	#[serde(default)]
	default_reasoning_level: Option<Str>,
	#[serde(default)]
	supported_reasoning_levels: Vec<CodexReasoningLevel>,
	#[serde(default)]
	input_modalities: Vec<Str>,
	/// Processing tiers the backend offers for this model on this account.
	/// The ChatGPT client's "Fast" switch is the `priority` tier. An
	/// absent field says nothing; an empty list says the model has none.
	#[serde(default)]
	service_tiers: Option<Vec<CodexServiceTier>>,
}

/// One advertised processing tier: `{"id":"priority","name":"Fast",…}`.
#[derive(Deserialize)]
struct CodexServiceTier {
	id: Str,
}

/// One supported reasoning level. The live endpoint sends presets shaped
/// `{"effort":"high","description":"…"}`; older payloads and fixtures use the
/// bare effort name.
#[derive(Deserialize)]
#[serde(untagged)]
enum CodexReasoningLevel {
	Named(Str),
	Preset { effort: Str },
}

impl CodexReasoningLevel {
	fn effort(&self) -> &str {
		match self {
			Self::Named(effort) | Self::Preset { effort } => effort.as_str(),
		}
	}
}

struct CodexModelsDecoder {
	provider: ProviderId,
	route:    RouteId,
	done:     bool,
}

impl CodexModelsDecoder {
	fn protocol_error(&self, code: &'static str) -> Error {
		use crate::{
			error::{Error, ErrorKind, ErrorPhase, RetryAction},
			receipt::ExecutionReceipt,
		};
		Error::new(
			ErrorKind::Protocol,
			ErrorPhase::Discovery,
			RetryAction::Never,
			ExecutionReceipt::default(),
		)
		.provider(self.provider.clone())
		.route(self.route.clone())
		.code(Str::new(code))
	}

	fn capabilities(row: &CodexModelRow) -> CatalogModelCapabilities {
		use crate::catalog::{
			Availability, ChatCapabilities, ModalityBits, ModelCapabilities, OperationBits,
			OperationKind, ProviderFamily, ReasoningCapabilities, ReasoningEffort,
			ReasoningFeatureBits, ServiceTier, ToolCapabilities, ToolFeatureBits,
		};
		let service_tiers = match &row.service_tiers {
			None => Availability::Unknown,
			Some(tiers) => {
				let tiers: Vec<ServiceTier> = tiers
					.iter()
					.filter_map(|tier| ServiceTier::for_family(ProviderFamily::OpenAi, &tier.id))
					.collect();
				if tiers.is_empty() {
					Availability::Unsupported
				} else {
					Availability::Native(tiers.into_boxed_slice())
				}
			},
		};
		let mut efforts = Vec::new();
		for level in row.default_reasoning_level.as_deref().into_iter().chain(
			row.supported_reasoning_levels
				.iter()
				.map(CodexReasoningLevel::effort),
		) {
			let effort = match level {
				"none" | "off" => Some(ReasoningEffort::Off),
				"minimal" => Some(ReasoningEffort::Minimal),
				"low" => Some(ReasoningEffort::Low),
				"medium" => Some(ReasoningEffort::Medium),
				"high" => Some(ReasoningEffort::High),
				"xhigh" => Some(ReasoningEffort::Xhigh),
				"max" => Some(ReasoningEffort::Max),
				_ => None,
			};
			if let Some(effort) = effort
				&& !efforts.contains(&effort)
			{
				efforts.push(effort);
			}
		}
		let reasoning_observed = row
			.default_reasoning_level
			.as_deref()
			.is_some_and(|level| !matches!(level, "none" | "off"))
			|| !row.supported_reasoning_levels.is_empty();
		let modalities = row
			.input_modalities
			.iter()
			.fold(ModalityBits::empty(), |bits, modality| {
				bits
					| match modality.as_str() {
						"text" => ModalityBits::TEXT,
						"image" => ModalityBits::IMAGE,
						"audio" => ModalityBits::AUDIO,
						"video" => ModalityBits::VIDEO,
						"document" | "pdf" => ModalityBits::DOCUMENT,
						_ => ModalityBits::empty(),
					}
			});
		ModelCapabilities {
			operations:    OperationBits::for_kind(OperationKind::Chat),
			chat:          Some(ChatCapabilities {
				roles: Availability::Unknown,
				mid_session_roles: Availability::Unknown,
				// The authenticated Codex Responses route accepts caller-provided
				// tools for every model it lists. The models payload has no per-model
				// tools field, so this support comes from the route contract rather
				// than from an absent response property.
				tools: Availability::Native(ToolCapabilities {
					features:      ToolFeatureBits::empty(),
					maximum_tools: None,
				}),
				structured_output: Availability::Unknown,
				grammar: Availability::Unknown,
				text_verbosity: Availability::Unknown,
				reasoning: if reasoning_observed {
					Availability::Native(ReasoningCapabilities {
						features:              ReasoningFeatureBits::EFFORT,
						efforts:               efforts.into_boxed_slice(),
						minimum_budget_tokens: None,
						maximum_budget_tokens: None,
					})
				} else {
					Availability::Unknown
				},
				input_modalities: if modalities.bits() == 0 {
					Availability::Unknown
				} else {
					Availability::Native(modalities)
				},
				hosted_tools: Availability::Unknown,
				image_input: Availability::Unknown,
				prompt_caching: Availability::Unknown,
				service_tiers,
				sampling: Availability::Unknown,
				safety: Availability::Unknown,
				determinism: Availability::Unknown,
				server_state: Availability::Unknown,
				logprobs: Availability::Unknown,
			}),
			embeddings:    None,
			image:         None,
			video:         None,
			speech:        None,
			transcription: None,
			realtime:      None,
			search:        None,
			tokenization:  None,
		}
	}
}

impl Decoder for CodexModelsDecoder {
	fn push(&mut self, frame: Frame, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		use crate::catalog::{DiscoveredModel, ModelAvailability, ModelLimits, WireModelId};
		if self.done {
			return Ok(());
		}
		let Frame::Raw(payload) = frame else {
			return Err(self.protocol_error("codex_discovery_wrong_framing"));
		};
		let payload: CodexModelsPayload = serde_json::from_slice(&payload)
			.map_err(|_| self.protocol_error("codex_discovery_invalid_json"))?;
		let raw_rows = payload
			.into_rows()
			.ok_or_else(|| self.protocol_error("codex_discovery_missing_models"))?;
		let mut rows = Vec::new();
		for raw in raw_rows {
			if raw
				.visibility
				.as_deref()
				.is_some_and(|visibility| matches!(visibility, "hide" | "hidden"))
			{
				continue;
			}
			let Some(wire_model) = raw
				.slug
				.clone()
				.or_else(|| raw.id.clone())
				.filter(|id| !id.is_empty())
			else {
				continue;
			};
			let context_window = raw.context_window.unwrap_or(272_000);
			let declared_operations = OperationBits::for_kind(OperationKind::Chat)
				| omp_catalog::model_operation_overrides("openai-codex", wire_model.as_str());
			tracing::debug!(
				model = %wire_model,
				service_tiers = ?raw
					.service_tiers
					.as_ref()
					.map(|tiers| tiers.iter().map(|tier| tier.id.as_str()).collect::<Vec<_>>()),
				"codex listing row"
			);
			rows.push(DiscoveredModel {
				provider: self.provider.clone(),
				route: self.route.clone(),
				wire_model: WireModelId::from(wire_model.clone()),
				aliases: Box::new([]),
				display_name: raw
					.display_name
					.clone()
					.filter(|name| !name.is_empty())
					.or(Some(wire_model)),
				declared_class: None,
				declared_operations,
				declared_capabilities: Some(Self::capabilities(&raw)),
				declared_limits: Some(ModelLimits {
					context_window:        Some(context_window),
					maximum_input_tokens:  None,
					maximum_output_tokens: Some(context_window.min(128_000)),
					maximum_batch:         None,
				}),
				declared_pricing: Box::new([]),
				extended_context_mode: None,
				availability: Some(ModelAvailability::Available),
				source: Str::new(CODEX_DISCOVERY_SOURCE),
				observed_at_ms: None,
				updated_at_ms: None,
				deprecated: None,
			});
		}
		rows.sort_by(|left, right| left.wire_model.cmp(&right.wire_model));
		emit(RawEvent::DiscoveredModels { rows, next_cursor: None });
		self.done = true;
		Ok(())
	}

	fn finish(&mut self, _emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		if self.done {
			Ok(())
		} else {
			Err(self.protocol_error("codex_discovery_response_missing"))
		}
	}
}

impl Codec for OpenAiCodexCodec {
	fn encode(
		&self,
		context: &EncodeContext<'_>,
		operation: &OperationCall,
	) -> Result<EncodedRequest, Error> {
		use bytes::Bytes;

		use crate::{
			body::BodySource,
			error::{Error, ErrorKind, ErrorPhase, RetryAction},
			receipt::ExecutionReceipt,
			transport::FramingProtocol,
		};
		let fail = |code: &'static str| {
			Error::new(
				ErrorKind::InvalidRequest,
				ErrorPhase::Encoding,
				RetryAction::Never,
				ExecutionReceipt::default(),
			)
			.code(Str::new(code))
		};
		if let OperationCall::DiscoverModels(request) = operation {
			if request.cursor.is_some() {
				return Err(fail("codex_discovery_cursor_unsupported"));
			}
			let base = context
				.route
				.endpoint
				.base_url
				.as_str()
				.trim_end_matches('/');
			let uri = sf!("{base}/codex/models?client_version={CODEX_CLIENT_VERSION}");
			let mut headers = vec![
				super::RequestHeader { name: sf!("accept"), value: sf!("application/json") },
				super::RequestHeader {
					name:  sf!("openai-beta"),
					value: sf!("responses=experimental"),
				},
				super::RequestHeader { name: sf!("originator"), value: Str::new(CODEX_ORIGINATOR) },
				super::RequestHeader { name: sf!("version"), value: Str::new(CODEX_CLIENT_VERSION) },
			];
			let account_id = context
				.account
				.and_then(|account| account.account.as_ref())
				.map(|account| Str::new(account.as_str()))
				.or_else(|| {
					self
						.options
						.identity
						.as_ref()
						.and_then(|identity| identity.account_id.clone())
				});
			if let Some(account_id) = account_id {
				headers.push(RequestHeader { name: sf!("chatgpt-account-id"), value: account_id });
			}
			apply_codex_residency_header(&mut headers, context.account);
			return Ok(EncodedRequest {
				operation: OperationKind::DiscoverModels,
				method: RequestMethod::Get,
				uri,
				headers: headers.into_boxed_slice(),
				body: BodySource::Bytes(Bytes::new()),
				framing: FramingProtocol::Raw,
				bounds: SizeBounds {
					request_body: 0,
					frame:        16 * 1024 * 1024,
					response:     256 * 1024 * 1024,
				},
				sealed_body: None,
				adjustments: Vec::new(),
			});
		}
		if let OperationCall::GenerateImage(request) = operation {
			let target = context
				.target
				.ok_or_else(|| fail("missing_codex_image_wire_target"))?;
			let endpoint = resolve_codex_responses_url(target.endpoint.base_url.as_str());
			let mut headers = vec![
				super::RequestHeader { name: sf!("content-type"), value: sf!("application/json") },
				super::RequestHeader { name: sf!("accept"), value: sf!("text/event-stream") },
			];
			let account_id = context
				.account
				.and_then(|account| account.account.as_ref())
				.map(|account| Str::new(account.as_str()))
				.or_else(|| {
					self
						.options
						.identity
						.as_ref()
						.and_then(|identity| identity.account_id.clone())
				});
			if let Some(account_id) = account_id {
				headers.push(RequestHeader { name: sf!("chatgpt-account-id"), value: account_id });
			}
			apply_codex_residency_header(&mut headers, context.account);
			return Ok(EncodedRequest {
				operation:   OperationKind::GenerateImage,
				method:      RequestMethod::Post,
				uri:         endpoint,
				headers:     headers.into_boxed_slice(),
				body:        BodySource::Bytes(hosted_image_body(context, request)?),
				framing:     FramingProtocol::Sse,
				bounds:      SizeBounds {
					request_body: 64 * 1024 * 1024,
					frame:        16 * 1024 * 1024,
					response:     256 * 1024 * 1024,
				},
				sealed_body: None,
				adjustments: Vec::new(),
			});
		}
		let OperationCall::Chat(chat) = operation else {
			return Err(fail("codex_chat_only"));
		};
		let target = context
			.target
			.ok_or_else(|| fail("missing_codex_wire_target"))?;
		let encoded = self
			.responses
			.encode_chat(context, chat)
			.map_err(|error| match error {
				ResponsesEncodeError::MismatchedProviderProof => {
					fail("codex_mismatched_responses_provider_proof")
				},
				ResponsesEncodeError::UnreplayableProviderProof => {
					fail("codex_unreplayable_responses_provider_proof")
				},
				ResponsesEncodeError::MissingWireTarget => fail("codex_missing_responses_wire_target"),
				ResponsesEncodeError::MissingCallIdentity => {
					fail("codex_missing_responses_call_identity")
				},
				ResponsesEncodeError::UnresolvedStoredMedia => fail("codex_unresolved_responses_media"),
				ResponsesEncodeError::UnsupportedOutputFormat => {
					fail("codex_unsupported_responses_output_format")
				},
				ResponsesEncodeError::UnsupportedComputerUse => {
					fail("codex_unsupported_responses_computer_use")
				},
				ResponsesEncodeError::UnsupportedComputerUseConfig => {
					fail("codex_unsupported_responses_computer_use_config")
				},
				ResponsesEncodeError::MalformedServerState => {
					fail("codex_malformed_responses_server_state")
				},
				ResponsesEncodeError::MismatchedServerState => {
					fail("codex_mismatched_responses_server_state")
				},
				ResponsesEncodeError::UnreplayableImageFileReference { .. } => {
					fail("codex_unreplayable_responses_image_file_reference")
				},
				_ => fail("codex_responses_encoding"),
			})?;
		if !encoded.adjustments.is_empty() {
			return Err(fail("codex_adjustment_requires_planning"));
		}
		let mut request = encoded.request;
		let responses_lite = context
			.route
			.use_responses_lite
			.unwrap_or(self.options.responses_lite);
		transform_codex_request(&mut request, responses_lite, self.options.concurrent_summaries)
			.map_err(|_| fail("invalid_codex_responses_lite_request"))?;
		if !responses_lite {
			apply_all_turns_context_policy(
				&mut request,
				context.policy.reasoning.supports_all_turns_context == Some(true),
			);
		}
		let transport = self
			.options
			.transport
			.unwrap_or(match context.route.codex_transport {
				CodexTransportPreference::WebsocketPreferred => CodexWireTransport::Http,
				CodexTransportPreference::HttpOnly => CodexWireTransport::Http,
			});
		// Codec options pin an identity for hosts that own the session; a bare
		// call still names its session through the call affinity, with the
		// request id as the retry-stable turn.
		let affinity_identity =
			context
				.affinity
				.provider_session
				.clone()
				.map(|session_id| CodexRequestIdentity {
					session_id,
					turn_id: Str::new(context.request_id.as_str()),
					account_id: None,
					originator: Some(Str::new(CODEX_ORIGINATOR)),
				});
		if let Some(identity) = self
			.options
			.identity
			.as_ref()
			.or(affinity_identity.as_ref())
		{
			apply_codex_client_metadata(
				&mut request,
				identity,
				transport,
				responses_lite,
				None,
				&self.options.turn_metadata,
			);
		}
		let endpoint = resolve_codex_responses_url(target.endpoint.base_url.as_str());
		let (uri, body, framing, accept, method) = match transport {
			CodexWireTransport::Http => (
				endpoint,
				serde_json::to_vec(&request).map(Bytes::from),
				FramingProtocol::Sse,
				"text/event-stream",
				RequestMethod::Post,
			),
			CodexWireTransport::WebSocket => (
				codex_websocket_url(&endpoint).map_err(|_| fail("invalid_codex_websocket_endpoint"))?,
				serde_json::to_vec(&CodexResponseCreate {
					kind: CodexResponseCreateKind::ResponseCreate,
					request,
				})
				.map(Bytes::from),
				FramingProtocol::WebSocket,
				"application/json",
				RequestMethod::Get,
			),
		};
		let body = body.map_err(|_| fail("codex_request_serialization"))?;
		crate::codec::debug_capture_request("openai-codex", &body);
		let mut headers = vec![
			super::RequestHeader { name: sf!("content-type"), value: sf!("application/json") },
			super::RequestHeader { name: sf!("accept"), value: Str::new(accept) },
		];
		apply_codex_residency_header(&mut headers, context.account);
		Ok(EncodedRequest {
			operation: OperationKind::Chat,
			method,
			uri,
			headers: headers.into_boxed_slice(),
			body: BodySource::Bytes(body),
			framing,
			bounds: SizeBounds {
				request_body: 64 * 1024 * 1024,
				frame:        16 * 1024 * 1024,
				response:     256 * 1024 * 1024,
			},
			sealed_body: None,
			adjustments: Vec::new(),
		})
	}

	fn decoder(&self, context: &DecodeContext<'_>) -> Result<DecoderState, Error> {
		context.debug_assert_valid();
		if context.operation == OperationKind::DiscoverModels {
			if !matches!(context.operation_call, crate::call::OperationCall::DiscoverModels(_)) {
				return Err(
					CodexModelsDecoder {
						provider: context.provider.clone(),
						route:    context.route.clone(),
						done:     false,
					}
					.protocol_error("codex_discovery_operation_mismatch"),
				);
			}
			return Ok(Box::new(CodexModelsDecoder {
				provider: context.provider.clone(),
				route:    context.route.clone(),
				done:     false,
			}));
		}
		<OpenAiResponsesCodec as Codec>::decoder(&self.responses, context)
	}
}

#[cfg(test)]
mod tests {
	use omp_core::Str;
	use serde::Deserialize;

	use super::{
		CodexContinuationState, CodexFallbackEvidence, CodexFrameDisposition, CodexFrameRouter,
		CodexModelsDecoder, CodexReplaySafety, CodexResponseCreate, CodexWebSocketFailure,
		CodexWebSocketProtocolError, apply_all_turns_context_policy,
		apply_codex_residency_header as super_apply_codex_residency_header, classify_codex_fallback,
		transform_codex_request,
	};
	use crate::{
		call::AccountRoutingContext,
		codec::{
			RequestHeader,
			openai_responses::{ResponsesOutputItem, ResponsesRequest, ResponsesStreamEvent},
		},
		id::RegionId,
	};

	#[test]
	fn discovered_codex_hosted_image_models_retain_image_generation() {
		for model in ["gpt-5.5", "o3", "o3-pro"] {
			assert!(
				omp_catalog::model_operation_overrides("openai-codex", model)
					.contains_kind(omp_catalog::OperationKind::GenerateImage)
			);
		}
		assert!(
			!omp_catalog::model_operation_overrides("openai-codex", "codex-mini-latest")
				.contains_kind(omp_catalog::OperationKind::GenerateImage)
		);
	}

	#[test]
	fn codex_client_metadata_serializes_protocol_strings() {
		let identity = super::CodexRequestIdentity {
			session_id: Str::new_static("session-1"),
			turn_id:    Str::new_static("turn-1"),
			account_id: Some(Str::new_static("account-1")),
			originator: Some(Str::new_static("omperor")),
		};
		for transport in [super::CodexWireTransport::Http, super::CodexWireTransport::WebSocket] {
			for responses_lite in [false, true] {
				let mut request: ResponsesRequest = serde_json::from_str(include_str!(
					"../../../../fixtures/llm-oracle/openai/codex/request.responses_lite.json"
				))
				.expect("typed request fixture");
				super::apply_codex_client_metadata(
					&mut request,
					&identity,
					transport,
					responses_lite,
					Some("state-1"),
					&super::ResponsesMetadata::new(),
				);
				let wire = serde_json::to_value(&request).expect("wire request");
				let metadata = wire["client_metadata"]
					.as_object()
					.expect("metadata object");
				assert!(metadata.values().all(serde_json::Value::is_string));
				assert_eq!(metadata["responses_lite"], responses_lite.to_string());
				assert_eq!(metadata["session_id"], "session-1");
				assert_eq!(metadata["turn_id"], "turn-1");
			}
		}
	}

	#[test]
	fn responses_lite_fixture_is_an_exact_typed_rewrite() {
		let mut request: ResponsesRequest = serde_json::from_str(include_str!(
			"../../../../fixtures/llm-oracle/openai/codex/request.responses_lite.json"
		))
		.expect("typed request fixture");
		let expected: ResponsesRequest = serde_json::from_str(include_str!(
			"../../../../fixtures/llm-oracle/openai/codex/expect.responses_lite.json"
		))
		.expect("typed expected fixture");
		transform_codex_request(&mut request, true, false).expect("Responses Lite rewrite");
		assert_eq!(request, expected);
	}
	#[test]
	fn supports_all_turns_reasoning_context_matches_pi_request_shape() {
		let mut request: ResponsesRequest = serde_json::from_str(include_str!(
			"../../../../fixtures/llm-oracle/openai/codex/request.responses_lite.json"
		))
		.expect("typed request fixture");
		transform_codex_request(&mut request, true, false).expect("Responses Lite rewrite");
		let reasoning = request.reasoning.as_mut().expect("reasoning fixture");
		reasoning.context = Some(Str::new_static("all_turns"));
		apply_all_turns_context_policy(&mut request, false);
		assert_eq!(
			request
				.reasoning
				.as_ref()
				.and_then(|value| value.context.as_deref()),
			None
		);
		request
			.reasoning
			.as_mut()
			.expect("reasoning fixture")
			.context = Some(Str::new_static("all_turns"));
		apply_all_turns_context_policy(&mut request, true);
		assert_eq!(
			request
				.reasoning
				.as_ref()
				.and_then(|value| value.context.as_deref()),
			Some("all_turns")
		);
	}

	#[test]
	fn assistant_history_uses_codex_output_content() {
		let original: ResponsesRequest = serde_json::from_value(serde_json::json!({
			"model": "gpt-5.6-luna",
			"input": [
				{
					"role": "assistant",
					"content": [{"type": "input_text", "text": "prior answer"}]
				},
				{
					"role": "user",
					"content": [{"type": "input_text", "text": "next question"}]
				}
			]
		}))
		.expect("typed request");

		for responses_lite in [false, true] {
			let mut request = original.clone();
			transform_codex_request(&mut request, responses_lite, false)
				.expect("Codex request rewrite");
			let encoded = serde_json::to_value(request).expect("request JSON");
			assert_eq!(encoded["input"][0]["content"][0]["type"], "output_text");
			assert_eq!(encoded["input"][1]["content"][0]["type"], "input_text");
		}
	}
	#[test]
	fn function_tools_use_permissive_codex_schemas() {
		let original: ResponsesRequest = serde_json::from_value(serde_json::json!({
			"model": "gpt-5.6-luna",
			"input": [],
			"tools": [{
				"type": "function",
				"name": "ask",
				"parameters": {
					"type": "object",
					"properties": {"optional": {"type": "string"}}
				},
				"strict": true
			}]
		}))
		.expect("typed request");

		for responses_lite in [false, true] {
			let mut request = original.clone();
			transform_codex_request(&mut request, responses_lite, false)
				.expect("Codex request rewrite");
			let tool = if responses_lite {
				&request.input[0].tools[0]
			} else {
				&request.tools[0]
			};
			assert_eq!(tool.strict, Some(false));
		}
	}
	#[test]
	fn residency_header_is_applied_only_for_claimed_regions() {
		let account = AccountRoutingContext {
			region: Some(RegionId::new("us")),
			..AccountRoutingContext::default()
		};
		let mut headers = Vec::new();
		super_apply_codex_residency_header(&mut headers, Some(&account));
		assert_eq!(headers, vec![RequestHeader {
			name:  Str::new_static("x-openai-internal-codex-residency"),
			value: Str::new_static("us"),
		}],);

		let mut configured = vec![RequestHeader {
			name:  Str::new_static("X-OpenAI-Internal-Codex-Residency"),
			value: Str::new_static("eu"),
		}];
		super_apply_codex_residency_header(&mut configured, Some(&account));
		assert_eq!(configured[0].value.as_str(), "eu");

		let mut absent = Vec::new();
		super_apply_codex_residency_header(&mut absent, Some(&AccountRoutingContext::default()));
		assert!(absent.is_empty());
	}

	#[test]
	fn websocket_continuation_fixture_sends_only_strict_delta() {
		#[derive(Deserialize)]
		struct Fixture {
			previous_request:        ResponsesRequest,
			previous_response_id:    Str,
			previous_response_items: Vec<ResponsesOutputItem>,
			current_request:         ResponsesRequest,
			expected_frame:          CodexResponseCreate,
		}
		let fixture: Fixture = serde_json::from_str(include_str!(
			"../../../../fixtures/llm-oracle/openai/codex/websocket.continuation.json"
		))
		.expect("continuation fixture");
		let state = CodexContinuationState::new(
			fixture.previous_request,
			fixture.previous_response_id,
			fixture.previous_response_items,
		);
		assert_eq!(
			state
				.continuation_frame(&fixture.current_request)
				.expect("strict delta"),
			fixture.expected_frame
		);
	}

	#[test]
	fn websocket_continuation_survives_service_tier_changes() {
		#[derive(Deserialize)]
		struct Fixture {
			previous_request:        ResponsesRequest,
			previous_response_id:    Str,
			previous_response_items: Vec<ResponsesOutputItem>,
			current_request:         ResponsesRequest,
		}
		let mut fixture: Fixture = serde_json::from_str(include_str!(
			"../../../../fixtures/llm-oracle/openai/codex/websocket.continuation.json"
		))
		.expect("continuation fixture");
		fixture.previous_request.service_tier = Some(Str::new("default"));
		fixture.current_request.service_tier = Some(Str::new("priority"));
		let state = CodexContinuationState::new(
			fixture.previous_request,
			fixture.previous_response_id.clone(),
			fixture.previous_response_items,
		);
		let frame = state
			.continuation_frame(&fixture.current_request)
			.expect("service tier is a per-turn nonsemantic option");
		assert_eq!(frame.request.previous_response_id, Some(fixture.previous_response_id));
		assert_eq!(frame.request.service_tier.as_deref(), Some("priority"));
		assert!(frame.request.input.len() < fixture.current_request.input.len());
	}

	#[test]
	fn websocket_router_replays_interleaved_fixture_once() {
		let frames: Vec<ResponsesStreamEvent> = serde_json::from_str(include_str!(
			"../../../../fixtures/llm-oracle/openai/codex/websocket.interleaved.json"
		))
		.expect("interleaved fixture");
		let expected = [
			CodexFrameDisposition::Drop,
			CodexFrameDisposition::Accept,
			CodexFrameDisposition::Accept,
			CodexFrameDisposition::Accept,
			CodexFrameDisposition::Accept,
			CodexFrameDisposition::Accept,
			CodexFrameDisposition::Terminal,
			CodexFrameDisposition::Drop,
		];
		let mut router = CodexFrameRouter::default();
		let actual = frames
			.iter()
			.map(|frame| router.route(frame).expect("correlated frame"))
			.collect::<Vec<_>>();
		assert_eq!(actual, expected);
		assert_eq!(router.active_response_id(), Some("resp_new"));
	}

	#[test]
	fn active_foreign_response_is_typed_interleaving_failure() {
		let frames: Vec<ResponsesStreamEvent> = serde_json::from_str(
			r#"[{"type":"response.created","response":{"id":"resp_a"}},{"type":"response.completed","response":{"id":"resp_b"}}]"#,
		).expect("typed frames");
		let mut router = CodexFrameRouter::default();
		assert_eq!(router.route(&frames[0]), Ok(CodexFrameDisposition::Accept));
		assert_eq!(router.route(&frames[1]), Err(CodexWebSocketProtocolError::InterleavedResponse));
	}

	#[test]
	fn fallback_classification_never_performs_post_commit_replay() {
		let safe = CodexReplaySafety { replayable_body: true, ..CodexReplaySafety::default() };
		assert_eq!(
			classify_codex_fallback(CodexWebSocketFailure::RetryableTransport, safe, 0, 5),
			CodexFallbackEvidence::ReconnectWebSocket,
		);
		assert_eq!(
			classify_codex_fallback(CodexWebSocketFailure::RetryableTransport, safe, 5, 5),
			CodexFallbackEvidence::ReplayOverHttp,
		);
		let committed = CodexReplaySafety {
			committed: true,
			replayable_body: true,
			..CodexReplaySafety::default()
		};
		assert_eq!(
			classify_codex_fallback(CodexWebSocketFailure::ConnectionFatal, committed, 0, 5),
			CodexFallbackEvidence::Surface,
		);
	}

	#[test]
	fn codex_discovery_decodes_typed_recursive_envelope_without_name_inference() {
		use bytes::Bytes;

		use crate::{
			catalog::{OperationKind, ProviderId, RouteId},
			codec::{Decoder as _, RawEvent},
			transport::Frame,
		};
		let mut decoder = CodexModelsDecoder {
			provider: ProviderId::from("openai-codex"),
			route:    RouteId::from("openai-codex"),
			done:     false,
		};
		let mut output = None;
		decoder
			.push(
				Frame::Raw(Bytes::from_static(
					br#"{"data":{"models":[
				{"slug":"gpt-5.2-codex","display_name":"GPT-5.2 Codex","visibility":"visible",
				 "context_window":400000,"default_reasoning_level":"medium",
				 "input_modalities":["text","image"]},
				{"slug":"hidden-model","visibility":"hidden"},
				{"id":"opaque-model","supported_reasoning_levels":["low","high"]}
			]}}"#,
				)),
				&mut |event| output = Some(event),
			)
			.expect("typed Codex discovery response");
		let Some(RawEvent::DiscoveredModels { rows, next_cursor: None }) = output else {
			panic!("expected discovered rows");
		};
		assert_eq!(rows.len(), 2);
		assert_eq!(rows[0].wire_model.as_str(), "gpt-5.2-codex");
		assert_eq!(rows[0].display_name.as_deref(), Some("GPT-5.2 Codex"));
		assert_eq!(
			rows[0]
				.declared_limits
				.expect("native limits")
				.context_window,
			Some(400_000)
		);
		assert!(
			rows[0]
				.declared_operations
				.contains_kind(OperationKind::Chat)
		);
		assert_eq!(rows[1].wire_model.as_str(), "opaque-model");
		assert!(rows[1].declared_class.is_none());
	}

	#[test]
	fn decodes_live_codex_models_payload_with_reasoning_presets() {
		use bytes::Bytes;

		use crate::{
			catalog::{Availability, ProviderId, ReasoningEffort, RouteId},
			codec::{Decoder as _, RawEvent},
			transport::Frame,
		};
		let mut decoder = CodexModelsDecoder {
			provider: ProviderId::from("openai-codex"),
			route:    RouteId::from("openai-codex"),
			done:     false,
		};
		let mut output = None;
		decoder
			.push(
				Frame::Raw(Bytes::from_static(
					br#"{"models":[
				{"slug":"gpt-6-astra","display_name":"GPT-6-Astra","visibility":"list","priority":1,
				 "default_reasoning_level":"medium",
				 "supported_reasoning_levels":[
				   {"effort":"low","description":"Fast responses with lighter reasoning"},
				   {"effort":"medium","description":"Balances speed and reasoning depth"},
				   {"effort":"high","description":"Greater reasoning depth"},
				   {"effort":"xhigh","description":"Extra high reasoning"}],
				 "service_tiers":[{"id":"priority","name":"Fast","description":"2x speed"}],
				 "context_window":272000,"max_context_window":872000,
				 "input_modalities":["text","image"]},
				{"slug":"gpt-daybreak-blue-latest","display_name":"Daybreak Blue","visibility":"list","priority":2,
				 "service_tiers":[]},
				{"slug":"gpt-old","display_name":"GPT-Old","visibility":"list","priority":4},
				{"slug":"gpt-reserve","display_name":"GPT-Reserve","visibility":"hide","priority":3}
			]}"#,
				)),
				&mut |event| output = Some(event),
			)
			.expect("live Codex discovery response");
		let Some(RawEvent::DiscoveredModels { rows, next_cursor: None }) = output else {
			panic!("expected discovered rows");
		};
		assert_eq!(rows.len(), 3, "hidden models are skipped");
		assert_eq!(rows[0].wire_model.as_str(), "gpt-6-astra");
		// The tiers a model advertises are what the ChatGPT client's "Fast"
		// switch stands on: Astra offers `priority`, Daybreak Blue says it
		// has none, and a payload without the field says nothing.
		let tiers = |row: usize| {
			rows[row]
				.declared_capabilities
				.as_ref()
				.and_then(|capabilities| capabilities.chat.as_ref())
				.map(|chat| chat.service_tiers.clone())
				.expect("chat capabilities")
		};
		let Availability::Native(astra_tiers) = tiers(0) else {
			panic!("advertised tiers should be native");
		};
		assert_eq!(
			astra_tiers
				.iter()
				.map(|tier| tier.name.as_str())
				.collect::<Vec<_>>(),
			["priority"]
		);
		assert!(matches!(tiers(1), Availability::Unsupported));
		assert!(matches!(tiers(2), Availability::Unknown));
		let chat = rows[0]
			.declared_capabilities
			.as_ref()
			.and_then(|capabilities| capabilities.chat.as_ref())
			.expect("chat capabilities");
		assert!(matches!(chat.tools, Availability::Native(_)));
		let Availability::Native(reasoning) = &chat.reasoning else {
			panic!("reasoning presets should be recorded as native reasoning");
		};
		assert_eq!(reasoning.efforts.as_ref(), [
			ReasoningEffort::Medium,
			ReasoningEffort::Low,
			ReasoningEffort::High,
			ReasoningEffort::Xhigh
		]);
	}
}
