//! DeepSeek HTTP client and ModelProvider implementation.

use crate::dto::{
    DeepSeekChatRequest, DeepSeekChatResponse, DeepSeekMessage, DeepSeekStreamOptions,
    DeepSeekThinking,
};
use crate::sse::parse_sse_frame_choices;
use crate::tool_calls::{decode_deepseek_tool_call, decode_usage, encode_tool_specs};
use crate::{ChatChunk, ChatRequest, ChatResponse, ChatStream, ModelProfile, ModelProvider};
use async_trait::async_trait;
use bytes::Bytes;
use futures_util::stream::{self, BoxStream};
use futures_util::StreamExt;
use seekcode_common::{ChatRole, SeekCodeError, SeekCodeResult};
use std::collections::VecDeque;
use std::time::Duration;

/// Default model context window in tokens when none is configured.
pub const DEFAULT_CONTEXT_WINDOW: u32 = 1_000_000;

/// Configuration for DeepSeek API access.
#[derive(Clone, Debug)]
pub struct DeepSeekConfig {
    /// Base URL for the DeepSeek OpenAI-compatible API.
    pub base_url: String,
    /// Optional API key loaded from the configured secret store.
    pub api_key: Option<String>,
    /// Default model used when the caller does not choose one.
    pub default_model: String,
    /// Model context window in tokens, used for context-compression decisions.
    pub context_window: u32,
    /// HTTP timeout for provider requests.
    pub timeout: Duration,
}

impl Default for DeepSeekConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.deepseek.com".to_string(),
            api_key: None,
            default_model: "deepseek-v4-pro".to_string(),
            context_window: DEFAULT_CONTEXT_WINDOW,
            timeout: Duration::from_secs(120),
        }
    }
}

/// DeepSeek provider implementation.
#[derive(Clone)]
pub struct DeepSeekClient {
    config: DeepSeekConfig,
    http: reqwest::Client,
}

impl DeepSeekClient {
    /// Creates a new DeepSeek client.
    pub fn new(config: DeepSeekConfig) -> SeekCodeResult<Self> {
        let http = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|error| SeekCodeError::ModelProvider(error.to_string()))?;

        Ok(Self { config, http })
    }

    /// Returns the active client configuration.
    pub fn config(&self) -> &DeepSeekConfig {
        &self.config
    }

    /// Returns the underlying HTTP client for future request implementations.
    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    fn chat_completions_url(&self) -> String {
        format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        )
    }

    fn api_key(&self) -> SeekCodeResult<&str> {
        self.config
            .api_key
            .as_deref()
            .filter(|key| !key.trim().is_empty())
            .ok_or_else(|| {
                SeekCodeError::Validation("DeepSeek API key is not configured".to_string())
            })
    }

    fn build_request(
        &self,
        request: ChatRequest,
        stream: bool,
    ) -> SeekCodeResult<DeepSeekChatRequest> {
        let tools = encode_tool_specs(&request.tools)?;
        let model = if request.model.trim().is_empty() {
            self.config.default_model.clone()
        } else {
            request.model
        };
        let messages = request
            .messages
            .into_iter()
            .map(DeepSeekMessage::from)
            .collect::<Vec<_>>();
        log_deepseek_request(&model, stream, &messages, tools.len());

        Ok(DeepSeekChatRequest {
            model,
            messages,
            stream,
            tools,
            stream_options: stream.then_some(DeepSeekStreamOptions {
                include_usage: true,
            }),
            thinking: DeepSeekThinking {
                kind: if request.thinking {
                    "enabled".to_string()
                } else {
                    "disabled".to_string()
                },
            },
            reasoning_effort: request
                .thinking
                .then_some(request.reasoning_effort)
                .flatten(),
        })
    }

    async fn send_chat_request(
        &self,
        request: DeepSeekChatRequest,
    ) -> SeekCodeResult<reqwest::Response> {
        // tracing::debug!("Sending chat request: {:?}", request);
        let response = self
            .http
            .post(self.chat_completions_url())
            .bearer_auth(self.api_key()?)
            .json(&request)
            .send()
            .await
            .map_err(|error| SeekCodeError::ModelProvider(error.to_string()))?;

        ensure_success(response).await
    }
}

#[async_trait]
impl ModelProvider for DeepSeekClient {
    fn stream_chat(&self, request: ChatRequest) -> SeekCodeResult<ChatStream> {
        let request = self.build_request(request, true)?;
        let model = request.model.clone();
        let _ = self.api_key()?;
        let state = StreamState::Init {
            client: self.clone(),
            request,
            model,
        };

        Ok(Box::pin(stream::unfold(state, next_stream_item)))
    }

    async fn complete_chat(&self, request: ChatRequest) -> SeekCodeResult<ChatResponse> {
        let request = self.build_request(request, false)?;
        let model = request.model.clone();
        let response = self.send_chat_request(request).await?;
        let status = response.status();
        let bytes = response.bytes().await.map_err(|error| {
            tracing::error!(
                target: "seekcode_deepseek_client::response",
                model,
                %status,
                %error,
                "failed to read non-streaming model response body"
            );
            SeekCodeError::ModelProvider(error.to_string())
        })?;
        let response: DeepSeekChatResponse = serde_json::from_slice(&bytes).map_err(|error| {
            let body_preview = response_body_preview(&bytes);
            tracing::error!(
                target: "seekcode_deepseek_client::response",
                model,
                %status,
                %error,
                body_preview,
                "failed to decode non-streaming model response body"
            );
            SeekCodeError::ModelProvider(format!("invalid response body: {error}"))
        })?;
        let choice = response.choices.into_iter().next().ok_or_else(|| {
            SeekCodeError::ModelProvider("DeepSeek returned no choices".to_string())
        })?;
        let tool_calls = choice
            .message
            .tool_calls
            .iter()
            .map(decode_deepseek_tool_call)
            .collect::<SeekCodeResult<Vec<_>>>()?;

        Ok(ChatResponse {
            content: choice.message.content.unwrap_or_default(),
            reasoning_content: choice.message.reasoning_content,
            tool_calls,
            usage: response.usage.map(decode_usage),
        })
    }

    async fn model_profile(&self, model: &str) -> SeekCodeResult<ModelProfile> {
        Ok(ModelProfile {
            id: model.to_string(),
            context_window: self.config.context_window,
            supports_tools: true,
            supports_thinking: true,
        })
    }
}

impl From<seekcode_common::ChatMessage> for DeepSeekMessage {
    fn from(message: seekcode_common::ChatMessage) -> Self {
        let seekcode_common::ChatMessage {
            role,
            content,
            reasoning_content,
            tool_calls,
            tool_call_id,
            ..
        } = message;
        let role_name = match role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "tool",
        };
        let content = match role {
            ChatRole::Assistant if content.is_empty() && tool_calls.is_empty() => Some(content),
            ChatRole::Tool => Some(content),
            _ if content.is_empty() => None,
            _ => Some(content),
        };

        Self {
            role: role_name.to_string(),
            content,
            reasoning_content,
            tool_calls,
            tool_call_id: tool_call_id.map(|id| id.to_string()),
        }
    }
}

fn log_deepseek_request(
    model: &str,
    stream: bool,
    messages: &[DeepSeekMessage],
    tool_count: usize,
) {
    tracing::debug!(
        target: "seekcode_deepseek_client::request",
        model,
        stream,
        message_count = messages.len(),
        tool_count,
        "building DeepSeek chat request"
    );

    for (index, message) in messages.iter().enumerate() {
        let serialized = serde_json::to_string(message)
            .unwrap_or_else(|error| format!("serialize error: {error}"));
        tracing::debug!(
                target: "seekcode_deepseek_client::request",
                index,
                role = %message.role,
                content_present = message.content.is_some(),
                content_len = message.content.as_deref().map(str::len).unwrap_or(0),
                reasoning_present = message.reasoning_content.is_some(),
                reasoning_len = message.reasoning_content.as_deref().map(str::len).unwrap_or(0),
                tool_call_count = message.tool_calls.len(),
                tool_call_id = ?message.tool_call_id,
                message_json = %serialized,
                "DeepSeek request message"
        );
    }
}

fn response_body_preview(bytes: &[u8]) -> String {
    const MAX_PREVIEW_CHARS: usize = 2_000;
    String::from_utf8_lossy(bytes)
        .chars()
        .take(MAX_PREVIEW_CHARS)
        .collect()
}

async fn ensure_success(response: reqwest::Response) -> SeekCodeResult<reqwest::Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let body = response
        .text()
        .await
        .unwrap_or_else(|error| format!("failed to read error body: {error}"));

    Err(SeekCodeError::ModelProvider(format!(
        "DeepSeek request failed with {status}: {body}"
    )))
}

enum StreamState {
    Init {
        client: DeepSeekClient,
        request: DeepSeekChatRequest,
        model: String,
    },
    Running {
        model: String,
        status: reqwest::StatusCode,
        bytes: BoxStream<'static, Result<Bytes, reqwest::Error>>,
        utf8_buffer: Vec<u8>,
        buffer: String,
        pending: VecDeque<ChatChunk>,
        done: bool,
    },
    Done,
}

async fn next_stream_item(
    mut state: StreamState,
) -> Option<(SeekCodeResult<ChatChunk>, StreamState)> {
    loop {
        match state {
            StreamState::Init {
                client,
                request,
                model,
            } => match client.send_chat_request(request).await {
                Ok(response) => {
                    let status = response.status();
                    state = StreamState::Running {
                        model,
                        status,
                        bytes: response.bytes_stream().boxed(),
                        utf8_buffer: Vec::new(),
                        buffer: String::new(),
                        pending: VecDeque::new(),
                        done: false,
                    };
                }
                Err(error) => {
                    tracing::error!(
                        target: "seekcode_deepseek_client::response",
                        model,
                        %error,
                        "failed to start streaming model response"
                    );
                    return Some((Err(error), StreamState::Done));
                }
            },
            StreamState::Running {
                model,
                status,
                mut bytes,
                mut utf8_buffer,
                mut buffer,
                mut pending,
                mut done,
            } => {
                if let Some(chunk) = pending.pop_front() {
                    return Some((
                        Ok(chunk),
                        StreamState::Running {
                            model,
                            status,
                            bytes,
                            utf8_buffer,
                            buffer,
                            pending,
                            done,
                        },
                    ));
                }

                if done {
                    return None;
                }

                match bytes.next().await {
                    Some(Ok(next_bytes)) => {
                        if let Err(error) =
                            push_stream_text(&mut buffer, &mut utf8_buffer, &next_bytes)
                        {
                            return Some((Err(error), StreamState::Done));
                        }

                        while let Some(frame) = take_next_sse_frame(&mut buffer) {
                            match frame_data(&frame) {
                                Some(data) if data == "[DONE]" => {
                                    pending.push_back(ChatChunk::Finished);
                                    done = true;
                                }
                                Some(data) => match parse_sse_frame_choices(&data) {
                                    Ok(chunks) => pending.extend(chunks),
                                    Err(error) => {
                                        tracing::error!(
                                            target: "seekcode_deepseek_client::response",
                                            model,
                                            %status,
                                            %error,
                                            frame = %data.chars().take(2_000).collect::<String>(),
                                            "failed to decode streaming model response frame"
                                        );
                                        return Some((Err(error), StreamState::Done));
                                    }
                                },
                                None => {}
                            }
                        }

                        state = StreamState::Running {
                            model,
                            status,
                            bytes,
                            utf8_buffer,
                            buffer,
                            pending,
                            done,
                        };
                    }
                    Some(Err(error)) => {
                        tracing::error!(
                            target: "seekcode_deepseek_client::response",
                            model,
                            %status,
                            %error,
                            "failed to read streaming model response body"
                        );
                        return Some((
                            Err(SeekCodeError::ModelProvider(error.to_string())),
                            StreamState::Done,
                        ));
                    }
                    None => {
                        if utf8_buffer.is_empty() {
                            return None;
                        }
                        return Some((
                            Err(SeekCodeError::ModelProvider(
                                "DeepSeek stream ended with incomplete UTF-8 data".to_string(),
                            )),
                            StreamState::Done,
                        ));
                    }
                }
            }
            StreamState::Done => return None,
        }
    }
}

fn push_stream_text(
    buffer: &mut String,
    utf8_buffer: &mut Vec<u8>,
    next_bytes: &[u8],
) -> SeekCodeResult<()> {
    utf8_buffer.extend_from_slice(next_bytes);
    match std::str::from_utf8(utf8_buffer) {
        Ok(text) => {
            push_normalized_stream_text(buffer, text);
            utf8_buffer.clear();
            Ok(())
        }
        Err(error) => {
            let valid_up_to = error.valid_up_to();
            if valid_up_to > 0 {
                let text = std::str::from_utf8(&utf8_buffer[..valid_up_to]).map_err(|error| {
                    SeekCodeError::ModelProvider(format!(
                        "DeepSeek stream returned invalid UTF-8 prefix: {error}"
                    ))
                })?;
                push_normalized_stream_text(buffer, text);
                utf8_buffer.drain(..valid_up_to);
            }

            if error.error_len().is_some() {
                Err(SeekCodeError::ModelProvider(
                    "DeepSeek stream returned invalid UTF-8 data".to_string(),
                ))
            } else {
                Ok(())
            }
        }
    }
}

fn push_normalized_stream_text(buffer: &mut String, text: &str) {
    buffer.push_str(&text.replace("\r\n", "\n"));
}

fn take_next_sse_frame(buffer: &mut String) -> Option<String> {
    let split_at = buffer.find("\n\n")?;
    let frame = buffer[..split_at].to_string();
    buffer.drain(..split_at + 2);
    Some(frame)
}

fn frame_data(frame: &str) -> Option<String> {
    let lines = frame
        .lines()
        .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
        .collect::<Vec<_>>();

    (!lines.is_empty()).then(|| lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use seekcode_common::{ChatMessage, ChatRole};
    use serde_json::json;

    #[test]
    fn assistant_reasoning_only_message_sets_empty_content() {
        let mut message = ChatMessage::new(ChatRole::Assistant, "");
        message.reasoning_content = Some("thinking".to_string());

        let encoded = DeepSeekMessage::from(message);

        assert_eq!(encoded.role, "assistant");
        assert_eq!(encoded.content.as_deref(), Some(""));
        assert_eq!(encoded.reasoning_content.as_deref(), Some("thinking"));
    }

    #[test]
    fn assistant_tool_call_message_omits_empty_content() {
        let mut message = ChatMessage::new(ChatRole::Assistant, "");
        message.tool_calls = vec![json!({
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": "{}"
            }
        })];

        let encoded = DeepSeekMessage::from(message);

        assert_eq!(encoded.role, "assistant");
        assert_eq!(encoded.content, None);
        assert_eq!(encoded.tool_calls.len(), 1);
    }

    #[test]
    fn empty_tool_result_message_keeps_empty_content() {
        let message = ChatMessage::new(ChatRole::Tool, "");

        let encoded = DeepSeekMessage::from(message);

        assert_eq!(encoded.role, "tool");
        assert_eq!(encoded.content.as_deref(), Some(""));
    }

    #[test]
    fn stream_text_keeps_multibyte_character_split_across_chunks() {
        let text = "它系统提示词";
        let split_at = 2;
        let mut buffer = String::new();
        let mut utf8_buffer = Vec::new();

        push_stream_text(&mut buffer, &mut utf8_buffer, &text.as_bytes()[..split_at])
            .expect("partial UTF-8 prefix is buffered");
        assert_eq!(buffer, "");
        assert_eq!(utf8_buffer, text.as_bytes()[..split_at]);

        push_stream_text(&mut buffer, &mut utf8_buffer, &text.as_bytes()[split_at..])
            .expect("remaining UTF-8 bytes complete the text");
        assert_eq!(buffer, text);
        assert!(utf8_buffer.is_empty());
    }
}
