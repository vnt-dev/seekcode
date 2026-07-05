//! Background task runner that drives the model/tool execution loop.

use futures_util::StreamExt;
use parking_lot::RwLock;
use seekcode_common::{ChatMessage, ChatRole, SeekCodeError, SeekCodeResult, SessionId, TaskId};
use seekcode_deepseek_client::{ChatChunk, ChatRequest, ModelProvider, ToolCall};
use seekcode_tool_system::{ToolContext, ToolRegistry};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

use crate::agent::AgentToolContext;
use crate::config::AgentConfig;
use crate::context::{AgentContext, AgentContextPreparer, AgentTaskContext};
use crate::event::{publish, publish_state, tool_call_display, tool_finished_event, AgentEvent};
use crate::task::{set_task_state, task_state, AgentState, AgentTask};

const MAX_MODEL_REQUEST_RETRIES: u32 = 5;

/// Percentage of the context window at which the in-loop compaction triggers.
const IN_LOOP_COMPACTION_TRIGGER_PERCENT: i64 = 95;

/// Owns the state required to run a single agent task to completion.
pub(crate) struct AgentTaskRunner {
    pub(crate) config: AgentConfig,
    pub(crate) provider: Arc<dyn ModelProvider>,
    pub(crate) tools: Arc<ToolRegistry>,
    pub(crate) tasks: Arc<RwLock<HashMap<TaskId, AgentTask>>>,
    pub(crate) events: mpsc::UnboundedSender<AgentEvent>,
    pub(crate) context_preparer: Option<Arc<dyn AgentContextPreparer>>,
    pub(crate) tool_context: AgentToolContext,
}

/// Input required to run one accepted agent task.
pub(crate) struct AgentRunRequest {
    pub(crate) task_id: TaskId,
    pub(crate) session_id: SessionId,
    pub(crate) model: String,
    pub(crate) thinking: bool,
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) prompt: String,
    pub(crate) context: AgentContext,
}

/// Result of executing one tool call, kept to append back into the context.
pub(crate) struct ToolCallRunResult {
    pub(crate) tool_call: ToolCall,
    pub(crate) result_content: String,
}

/// Data assembled from one successful model request attempt.
struct ModelRoundAttemptResult {
    round_content: String,
    round_reasoning: String,
    tool_results: Vec<ToolCallRunResult>,
    /// Input token count reported by the provider for this attempt, if any.
    input_tokens: Option<i64>,
}

/// Distinguishes retryable provider failures from fatal local runner failures.
enum ModelRoundAttemptError {
    Retryable(SeekCodeError),
    Fatal(SeekCodeError),
}

/// Outcome of one model request attempt.
enum ModelRoundAttemptOutcome {
    Finished(ModelRoundAttemptResult),
    Canceled,
}

impl AgentTaskRunner {
    /// Runs the task loop and reports failure as a terminal event.
    pub(crate) async fn run(self, request: AgentRunRequest) {
        let task_id = request.task_id;
        let session_id = request.session_id;
        let result = self.run_inner(request).await;
        if let Err(error) = result {
            tracing::error!(
                target: "seekcode_agent_core::runner",
                %session_id,
                %task_id,
                %error,
                "agent task failed"
            );
            let _ = set_task_state(&self.tasks, task_id, AgentState::Failed).await;
            publish_state(&self.events, session_id, task_id, AgentState::Failed);
            publish(
                &self.events,
                AgentEvent::Failed {
                    session_id,
                    task_id,
                    error: error.to_string(),
                },
            );
        }
    }

    /// Streams model rounds, executing tools until the model stops requesting them.
    async fn run_inner(&self, request: AgentRunRequest) -> SeekCodeResult<()> {
        let AgentRunRequest {
            task_id,
            session_id,
            model,
            thinking,
            reasoning_effort,
            prompt,
            context,
        } = request;
        let tool_specs = self.tools.tool_specs(self.config.strict_tools);
        let mut task_context = context.task_context.clone();
        if let Some(preparer) = &self.context_preparer {
            let precheck = preparer
                .precheck_context(task_id, session_id, &model, &prompt, &task_context)
                .await?;
            if precheck.should_compact_context {
                publish(
                    &self.events,
                    AgentEvent::ContextCompactionStarted {
                        session_id,
                        task_id,
                    },
                );

                let prepared = preparer
                    .prepare_context(
                        task_id,
                        session_id,
                        &model,
                        &prompt,
                        &task_context,
                        precheck,
                    )
                    .await?;
                task_context = prepared.context;

                if let Some(compaction) = prepared.compaction {
                    publish(
                        &self.events,
                        AgentEvent::ContextCompactionFinished {
                            session_id,
                            task_id,
                            compacted_rounds: compaction.compacted_rounds,
                            compacted_through_turn: compaction.compacted_through_turn,
                            summary_chars: compaction.summary_chars,
                        },
                    );
                } else {
                    publish(
                        &self.events,
                        AgentEvent::ContextCompactionCanceled {
                            session_id,
                            task_id,
                        },
                    );
                }
            }
        }

        // Messages appended during the loop (assistant turns and tool results);
        // the immutable per-round base comes from `task_context.messages()`.
        let mut appended_messages: Vec<ChatMessage> = Vec::new();

        let in_loop_compaction_threshold =
            i64::from(self.provider.model_profile(&model).await?.context_window)
                * IN_LOOP_COMPACTION_TRIGGER_PERCENT
                / 100;
        let mut in_loop_compacted = false;
        let mut latest_input_tokens: Option<i64> = None;

        for round_id in 1..=64 {
            // Fold the compacted-context + history portion once when the live input
            // token count crosses the trigger; skipped on round 1 (no usage yet).
            if !in_loop_compacted
                && latest_input_tokens.is_some_and(|tokens| tokens >= in_loop_compaction_threshold)
            {
                self.compact_running_context(task_id, session_id, &model, &mut task_context)
                    .await?;
                in_loop_compacted = true;
            }

            // Rebuild the request context from the (possibly compacted) base plus
            // the messages accumulated so far this run.
            let mut messages = task_context.messages();
            messages.extend(appended_messages.iter().cloned());

            let mut retry_count = 0;
            let attempt_result = loop {
                let chat_request = ChatRequest {
                    model: model.clone(),
                    messages: messages.clone(),
                    tools: tool_specs.clone(),
                    thinking,
                    reasoning_effort: reasoning_effort.clone(),
                    strict_tools: self.config.strict_tools,
                };

                publish(
                    &self.events,
                    AgentEvent::ModelRequestStarted {
                        session_id,
                        task_id,
                        round_id,
                        model: model.clone(),
                        message_count: chat_request.messages.len(),
                        tool_count: tool_specs.len(),
                    },
                );

                tracing::debug!(
                    target: "seekcode_agent_core::runner",
                    %session_id,
                    %task_id,
                    round_id,
                    attempt = retry_count + 1,
                    model = %model,
                    message_count = chat_request.messages.len(),
                    tool_count = tool_specs.len(),
                    "starting streaming model request"
                );

                match self
                    .run_model_round_attempt(task_id, session_id, round_id, chat_request)
                    .await
                {
                    Ok(ModelRoundAttemptOutcome::Finished(result)) => break result,
                    Ok(ModelRoundAttemptOutcome::Canceled) => return Ok(()),
                    Err(ModelRoundAttemptError::Fatal(error)) => return Err(error),
                    Err(ModelRoundAttemptError::Retryable(error)) => {
                        if retry_count >= MAX_MODEL_REQUEST_RETRIES {
                            return Err(error);
                        }

                        retry_count += 1;
                        tracing::warn!(
                            target: "seekcode_agent_core::runner",
                            %session_id,
                            %task_id,
                            round_id,
                            retry_count,
                            max_retries = MAX_MODEL_REQUEST_RETRIES,
                            %error,
                            "streaming model request failed; retrying round"
                        );
                        publish(
                            &self.events,
                            AgentEvent::ModelRequestRetrying {
                                session_id,
                                task_id,
                                round_id,
                                retry_count,
                                max_retries: MAX_MODEL_REQUEST_RETRIES,
                                error: error.to_string(),
                            },
                        );
                    }
                }
            };
            let ModelRoundAttemptResult {
                round_content,
                round_reasoning,
                tool_results,
                input_tokens,
            } = attempt_result;

            // Track the most recent reported input tokens for the next round's
            // compaction check; keep the prior value if this round had no usage.
            if let Some(input_tokens) = input_tokens {
                latest_input_tokens = Some(input_tokens);
            }

            if task_state(&self.tasks, task_id).await? == AgentState::Canceled {
                return Ok(());
            }

            if tool_results.is_empty() {
                set_task_state(&self.tasks, task_id, AgentState::Completed).await?;
                publish_state(&self.events, session_id, task_id, AgentState::Completed);
                publish(
                    &self.events,
                    AgentEvent::Finished {
                        session_id,
                        task_id,
                    },
                );
                return Ok(());
            }

            tracing::debug!(
                target: "seekcode_agent_core::runner",
                %session_id,
                %task_id,
                round_id,
                tool_result_count = tool_results.len(),
                request_message_count = messages.len(),
                "appending tool results before next streaming model request"
            );
            append_tool_results_to_context(
                &mut appended_messages,
                round_content,
                round_reasoning,
                tool_results,
            );
            tracing::debug!(
                target: "seekcode_agent_core::runner",
                %session_id,
                %task_id,
                current_round_id = round_id,
                next_round_id = round_id + 1,
                appended_message_count = appended_messages.len(),
                "tool results appended; next round will send updated context"
            );
        }

        Err(SeekCodeError::ModelProvider(
            "model requested too many tool rounds".to_string(),
        ))
    }

    /// Folds the compacted-context and history portion into a single summary.
    ///
    /// Publishes the compaction lifecycle events and, on success, replaces the
    /// compacted context and clears the expanded history in `task_context`.
    async fn compact_running_context(
        &self,
        task_id: TaskId,
        session_id: SessionId,
        model: &str,
        task_context: &mut AgentTaskContext,
    ) -> SeekCodeResult<()> {
        let Some(preparer) = &self.context_preparer else {
            return Ok(());
        };

        publish(
            &self.events,
            AgentEvent::ContextCompactionStarted {
                session_id,
                task_id,
            },
        );

        // All in-loop rounds persist under this task's single turn, so the summary
        // is recorded as covering the highest already-persisted prior turn only.
        let prior_max_turn = task_context
            .history_messages
            .iter()
            .map(|history| history.turn_sequence)
            .max()
            .unwrap_or(0);
        let messages_to_compact = task_context.compactable_messages();
        let compaction = preparer
            .compact_running_context(
                task_id,
                session_id,
                model,
                &messages_to_compact,
                prior_max_turn,
            )
            .await?;

        if let Some(compaction) = compaction {
            task_context.replace_compactable_with_summary(vec![compaction.summary_message]);
            publish(
                &self.events,
                AgentEvent::ContextCompactionFinished {
                    session_id,
                    task_id,
                    compacted_rounds: compaction.outcome.compacted_rounds,
                    compacted_through_turn: compaction.outcome.compacted_through_turn,
                    summary_chars: compaction.outcome.summary_chars,
                },
            );
        } else {
            publish(
                &self.events,
                AgentEvent::ContextCompactionCanceled {
                    session_id,
                    task_id,
                },
            );
        }

        Ok(())
    }

    /// Runs one streaming model request attempt for a round.
    async fn run_model_round_attempt(
        &self,
        task_id: TaskId,
        session_id: SessionId,
        round_id: u32,
        chat_request: ChatRequest,
    ) -> Result<ModelRoundAttemptOutcome, ModelRoundAttemptError> {
        let mut stream = self
            .provider
            .stream_chat(chat_request)
            .map_err(ModelRoundAttemptError::Retryable)?;
        let mut final_usage = None;
        let mut tool_calls = ToolCallAccumulator::default();
        let mut tool_results = Vec::new();
        let mut round_content = String::new();
        let mut round_reasoning = String::new();

        while let Some(chunk) = stream.next().await {
            if task_state(&self.tasks, task_id)
                .await
                .map_err(ModelRoundAttemptError::Fatal)?
                == AgentState::Canceled
            {
                return Ok(ModelRoundAttemptOutcome::Canceled);
            }

            match chunk.map_err(ModelRoundAttemptError::Retryable)? {
                ChatChunk::Choice(mut choice) => {
                    let content_delta = choice.delta.content.clone();
                    let reasoning_delta = choice.delta.reasoning_content.clone();
                    if let Some(text) = &choice.delta.content {
                        round_content.push_str(text);
                    }
                    if let Some(text) = &choice.delta.reasoning_content {
                        round_reasoning.push_str(text);
                    }
                    tool_calls.apply_choice_delta(&mut choice);
                    let should_run_tools = choice.finish_reason.as_deref() == Some("tool_calls");
                    if content_delta.is_some() || reasoning_delta.is_some() {
                        publish(
                            &self.events,
                            AgentEvent::AssistantMessageDelta {
                                session_id,
                                task_id,
                                round_id,
                                content: content_delta,
                                reasoning_content: reasoning_delta,
                            },
                        );
                    }

                    if should_run_tools {
                        for tool_call in tool_calls
                            .take_completed()
                            .map_err(ModelRoundAttemptError::Fatal)?
                        {
                            tool_results.push(
                                self.run_tool_call(task_id, session_id, round_id, tool_call)
                                    .await
                                    .map_err(ModelRoundAttemptError::Fatal)?,
                            );
                        }
                    }
                }
                ChatChunk::Content(text) => {
                    round_content.push_str(&text);
                    publish(
                        &self.events,
                        AgentEvent::AssistantMessageDelta {
                            session_id,
                            task_id,
                            round_id,
                            content: Some(text),
                            reasoning_content: None,
                        },
                    );
                }
                ChatChunk::Reasoning(text) => {
                    round_reasoning.push_str(&text);
                    publish(
                        &self.events,
                        AgentEvent::AssistantMessageDelta {
                            session_id,
                            task_id,
                            round_id,
                            content: None,
                            reasoning_content: Some(text),
                        },
                    );
                }
                ChatChunk::Usage(usage) => {
                    final_usage = Some(usage);
                }
                ChatChunk::Finished => {
                    let (assistant_message, tool_messages) = build_model_round_messages(
                        round_content.clone(),
                        round_reasoning.clone(),
                        &tool_results,
                    );
                    publish(
                        &self.events,
                        AgentEvent::ModelRoundFinished {
                            session_id,
                            task_id,
                            round_id,
                            assistant_message,
                            tool_messages,
                            usage: final_usage.clone(),
                        },
                    );
                    break;
                }
            }
        }

        Ok(ModelRoundAttemptOutcome::Finished(
            ModelRoundAttemptResult {
                round_content,
                round_reasoning,
                tool_results,
                input_tokens: final_usage
                    .as_ref()
                    .map(|usage| i64::from(usage.prompt_tokens)),
            },
        ))
    }

    /// Executes a single tool call, publishing lifecycle events and tracing.
    async fn run_tool_call(
        &self,
        task_id: TaskId,
        session_id: SessionId,
        round_id: u32,
        tool_call: ToolCall,
    ) -> SeekCodeResult<ToolCallRunResult> {
        set_task_state(&self.tasks, task_id, AgentState::RunningTool).await?;
        publish_state(&self.events, session_id, task_id, AgentState::RunningTool);

        let tool_call_id = tool_call.id.clone();
        let name = tool_call.name.clone();
        let arguments = tool_call.arguments.clone();
        let display = tool_call_display(&name, &arguments);
        let started_at = Instant::now();
        tracing::debug!(
            target: "seekcode_agent_core::tools",
            %session_id,
            %task_id,
            round_id,
            %tool_call_id,
            tool_name = %name,
            arguments = %arguments,
            "tool call started"
        );
        publish(
            &self.events,
            AgentEvent::ToolCallStarted {
                session_id,
                task_id,
                round_id,
                tool_call_id: tool_call_id.clone(),
                name: name.clone(),
                arguments,
                display,
            },
        );

        let result = self
            .tools
            .execute(
                &tool_call.name,
                ToolContext {
                    task_id,
                    workspace_id: self.tool_context.workspace_id,
                    workspace_root: self.tool_context.workspace_root.clone(),
                },
                tool_call.arguments.clone(),
            )
            .await;
        match &result {
            Ok(output) => tracing::debug!(
                target: "seekcode_agent_core::tools",
                %session_id,
                %task_id,
                round_id,
                %tool_call_id,
                tool_name = %name,
                elapsed_ms = started_at.elapsed().as_millis(),
                output_summary = %output.summary,
                output_content = %output.content,
                "tool call finished successfully"
            ),
            Err(error) => tracing::warn!(
                target: "seekcode_agent_core::tools",
                %session_id,
                %task_id,
                round_id,
                %tool_call_id,
                tool_name = %name,
                elapsed_ms = started_at.elapsed().as_millis(),
                %error,
                "tool call failed"
            ),
        }
        let result_content = match result {
            Ok(output) => {
                let content = output.content.to_string();
                publish(
                    &self.events,
                    tool_finished_event(
                        task_id,
                        session_id,
                        round_id,
                        tool_call_id.clone(),
                        name,
                        Ok(output),
                    ),
                );
                content
            }
            Err(error) => {
                // Return a structured failure payload to the model so it can recover or retry.
                let content = tool_error_result_content(&error);
                publish(
                    &self.events,
                    tool_finished_event(
                        task_id,
                        session_id,
                        round_id,
                        tool_call_id.clone(),
                        name,
                        Err(error),
                    ),
                );
                content
            }
        };

        set_task_state(&self.tasks, task_id, AgentState::Thinking).await?;
        publish_state(&self.events, session_id, task_id, AgentState::Thinking);

        Ok(ToolCallRunResult {
            tool_call,
            result_content,
        })
    }
}

/// Formats a failed tool execution as model-facing tool result content.
fn tool_error_result_content(error: &SeekCodeError) -> String {
    serde_json::json!({
        "ok": false,
        "error": error.to_string(),
    })
    .to_string()
}

/// Appends the assistant turn and tool results back into the running context.
pub(crate) fn append_tool_results_to_context(
    messages: &mut Vec<ChatMessage>,
    content: String,
    reasoning_content: String,
    tool_results: Vec<ToolCallRunResult>,
) {
    let (assistant, tool_messages) =
        build_model_round_messages(content, reasoning_content, &tool_results);

    messages.push(assistant);
    messages.extend(tool_messages);
}

/// Builds the assistant message and tool result messages for one completed model API call.
pub(crate) fn build_model_round_messages(
    content: String,
    reasoning_content: String,
    tool_results: &[ToolCallRunResult],
) -> (ChatMessage, Vec<ChatMessage>) {
    let mut assistant = ChatMessage::new(ChatRole::Assistant, content);
    if !reasoning_content.is_empty() {
        assistant.reasoning_content = Some(reasoning_content);
    }
    assistant.tool_calls = tool_results
        .iter()
        .map(|result| {
            serde_json::json!({
                "id": result.tool_call.id.to_string(),
                "type": "function",
                "function": {
                    "name": result.tool_call.name,
                    "arguments": result.tool_call.arguments.to_string()
                }
            })
        })
        .collect();

    let mut tool_messages = Vec::with_capacity(tool_results.len());
    for result in tool_results {
        let mut message = ChatMessage::new(ChatRole::Tool, result.result_content.clone());
        message.tool_call_id = Some(result.tool_call.id.clone());
        tool_messages.push(message);
    }

    (assistant, tool_messages)
}

/// Reassembles streamed tool-call deltas into complete tool calls.
#[derive(Default)]
struct ToolCallAccumulator {
    partials: BTreeMap<u32, PartialToolCall>,
}

impl ToolCallAccumulator {
    /// Merges the tool-call deltas from one choice chunk into the partial state.
    fn apply_choice_delta(&mut self, choice: &mut seekcode_deepseek_client::ChatChoiceChunk) {
        for delta in &mut choice.delta.tool_calls {
            let partial = self
                .partials
                .entry(delta.index)
                .or_insert_with(|| PartialToolCall {
                    id: String::new(),
                    name: String::new(),
                    arguments: String::new(),
                });
            if let Some(id) = &delta.id {
                partial.id.push_str(id);
            }
            if !partial.id.is_empty() {
                delta.id = Some(partial.id.clone());
            }

            if let Some(name) = &delta.name {
                partial.name.push_str(name);
            }
            if let Some(arguments) = &delta.arguments {
                partial.arguments.push_str(arguments);
            }
        }
    }

    /// Drains the accumulated partials into fully-formed tool calls.
    fn take_completed(&mut self) -> SeekCodeResult<Vec<ToolCall>> {
        let partials = std::mem::take(&mut self.partials);
        partials
            .into_values()
            .map(|partial| {
                if partial.id.is_empty() {
                    return Err(SeekCodeError::ModelProvider(
                        "missing streamed tool call id".to_string(),
                    ));
                }
                if partial.name.is_empty() {
                    return Err(SeekCodeError::ModelProvider(
                        "missing streamed tool call name".to_string(),
                    ));
                }
                let arguments = if partial.arguments.trim().is_empty() {
                    serde_json::Value::Object(Default::default())
                } else {
                    serde_json::from_str(&partial.arguments).map_err(|error| {
                        SeekCodeError::ModelProvider(format!(
                            "invalid streamed tool arguments: {error}"
                        ))
                    })?
                };

                Ok(ToolCall {
                    id: partial.id,
                    name: partial.name,
                    arguments,
                })
            })
            .collect()
    }
}

/// Partial tool call assembled across streamed deltas.
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}
