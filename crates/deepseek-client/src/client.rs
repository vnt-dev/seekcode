//! DeepSeek HTTP client and ModelProvider implementation.

use crate::dto::{
    DeepSeekChatRequest, DeepSeekChatResponse, DeepSeekMessage, DeepSeekStreamOptions,
};
use crate::sse::{parse_sse_frame_with_accumulator, ToolCallAccumulator};
use crate::tool_calls::{decode_deepseek_tool_call, decode_usage, encode_tool_specs};
use async_trait::async_trait;
use bytes::Bytes;
use futures_util::stream::{self, BoxStream};
use futures_util::StreamExt;
use seekcode_common::{ChatRole, SeekCodeError, SeekCodeResult};
use seekcode_model_provider::{ChatRequest, ChatResponse, ChatStream, ModelProfile, ModelProvider};
use std::collections::VecDeque;
use std::time::Duration;

/// Configuration for DeepSeek API access.
#[derive(Clone, Debug)]
pub struct DeepSeekConfig {
    /// Base URL for the DeepSeek OpenAI-compatible API.
    pub base_url: String,
    /// Optional API key loaded from the configured secret store.
    pub api_key: Option<String>,
    /// Default model used when the caller does not choose one.
    pub default_model: String,
    /// HTTP timeout for provider requests.
    pub timeout: Duration,
}

impl Default for DeepSeekConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.deepseek.com".to_string(),
            api_key: None,
            default_model: "deepseek-v4-pro".to_string(),
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

        Ok(DeepSeekChatRequest {
            model: if request.model.trim().is_empty() {
                self.config.default_model.clone()
            } else {
                request.model
            },
            messages: request
                .messages
                .into_iter()
                .map(DeepSeekMessage::from)
                .collect(),
            stream,
            tools,
            stream_options: stream.then_some(DeepSeekStreamOptions {
                include_usage: true,
            }),
        })
    }

    async fn send_chat_request(
        &self,
        request: DeepSeekChatRequest,
    ) -> SeekCodeResult<reqwest::Response> {
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
        let _ = self.api_key()?;
        let state = StreamState::Init {
            client: self.clone(),
            request: Some(request),
        };

        Ok(Box::pin(stream::unfold(state, next_stream_item)))
    }

    async fn complete_chat(&self, request: ChatRequest) -> SeekCodeResult<ChatResponse> {
        let request = self.build_request(request, false)?;
        let response = self.send_chat_request(request).await?;
        let response: DeepSeekChatResponse = response
            .json()
            .await
            .map_err(|error| SeekCodeError::ModelProvider(error.to_string()))?;
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
            context_window: 64_000,
            supports_tools: true,
            supports_thinking: true,
        })
    }
}

impl From<seekcode_common::ChatMessage> for DeepSeekMessage {
    fn from(message: seekcode_common::ChatMessage) -> Self {
        let role = match message.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "tool",
        };

        Self {
            role: role.to_string(),
            content: Some(message.content),
            reasoning_content: message.reasoning_content,
            tool_call_id: None,
        }
    }
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
        request: Option<DeepSeekChatRequest>,
    },
    Running {
        bytes: BoxStream<'static, Result<Bytes, reqwest::Error>>,
        buffer: String,
        pending: VecDeque<seekcode_model_provider::ChatChunk>,
        accumulator: ToolCallAccumulator,
        done: bool,
    },
    Done,
}

async fn next_stream_item(
    mut state: StreamState,
) -> Option<(
    SeekCodeResult<seekcode_model_provider::ChatChunk>,
    StreamState,
)> {
    loop {
        match state {
            StreamState::Init {
                client,
                mut request,
            } => {
                let request = request.take().expect("stream request is present");
                match client.send_chat_request(request).await {
                    Ok(response) => {
                        state = StreamState::Running {
                            bytes: response.bytes_stream().boxed(),
                            buffer: String::new(),
                            pending: VecDeque::new(),
                            accumulator: ToolCallAccumulator::default(),
                            done: false,
                        };
                    }
                    Err(error) => return Some((Err(error), StreamState::Done)),
                }
            }
            StreamState::Running {
                mut bytes,
                mut buffer,
                mut pending,
                mut accumulator,
                mut done,
            } => {
                if let Some(chunk) = pending.pop_front() {
                    return Some((
                        Ok(chunk),
                        StreamState::Running {
                            bytes,
                            buffer,
                            pending,
                            accumulator,
                            done,
                        },
                    ));
                }

                if done {
                    return None;
                }

                match bytes.next().await {
                    Some(Ok(next_bytes)) => {
                        buffer
                            .push_str(&String::from_utf8_lossy(&next_bytes).replace("\r\n", "\n"));

                        while let Some(frame) = take_next_sse_frame(&mut buffer) {
                            match frame_data(&frame) {
                                Some(data) if data == "[DONE]" => {
                                    pending.push_back(seekcode_model_provider::ChatChunk::Finished);
                                    done = true;
                                }
                                Some(data) => {
                                    match parse_sse_frame_with_accumulator(&data, &mut accumulator)
                                    {
                                        Ok(chunks) => pending.extend(chunks),
                                        Err(error) => {
                                            return Some((Err(error), StreamState::Done));
                                        }
                                    }
                                }
                                None => {}
                            }
                        }

                        state = StreamState::Running {
                            bytes,
                            buffer,
                            pending,
                            accumulator,
                            done,
                        };
                    }
                    Some(Err(error)) => {
                        return Some((
                            Err(SeekCodeError::ModelProvider(error.to_string())),
                            StreamState::Done,
                        ));
                    }
                    None => return None,
                }
            }
            StreamState::Done => return None,
        }
    }
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
