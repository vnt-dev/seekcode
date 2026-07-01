//! Background task runner that drives the model/tool execution loop.

use futures_util::StreamExt;
use parking_lot::RwLock;
use seekcode_common::{
    ChatMessage, ChatRole, SeekCodeError, SeekCodeResult, SessionId, TaskId, ToolCallId,
};
use seekcode_deepseek_client::{ChatChunk, ChatRequest, ModelProvider, ToolCall};
use seekcode_tool_system::{ToolContext, ToolRegistry};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

use crate::agent::AgentToolContext;
use crate::config::AgentConfig;
use crate::context::{AgentContext, AgentContextPreparer};
use crate::event::{publish, publish_state, tool_call_display, tool_finished_event, AgentEvent};
use crate::task::{set_task_state, task_state, AgentState, AgentTask};

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

/// Result of executing one tool call, kept to append back into the context.
pub(crate) struct ToolCallRunResult {
    pub(crate) tool_call: ToolCall,
    pub(crate) result_content: String,
}

impl AgentTaskRunner {
    /// Runs the task loop and reports failure as a terminal event.
    pub(crate) async fn run(
        self,
        task_id: TaskId,
        session_id: SessionId,
        model: String,
        thinking: bool,
        reasoning_effort: Option<String>,
        prompt: String,
        context: AgentContext,
    ) {
        let result = self
            .run_inner(
                task_id,
                session_id,
                model,
                thinking,
                reasoning_effort,
                prompt,
                context,
            )
            .await;
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
    async fn run_inner(
        &self,
        task_id: TaskId,
        session_id: SessionId,
        model: String,
        thinking: bool,
        reasoning_effort: Option<String>,
        prompt: String,
        context: AgentContext,
    ) -> SeekCodeResult<()> {
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

        let mut messages = task_context.messages();
        for round_id in 1..=64 {
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

            let mut stream = self.provider.stream_chat(chat_request)?;
            let mut final_usage = None;
            let mut sent_round_finished = false;
            let mut tool_calls = ToolCallAccumulator::default();
            let mut tool_results = Vec::new();
            let mut round_content = String::new();
            let mut round_reasoning = String::new();

            while let Some(chunk) = stream.next().await {
                if task_state(&self.tasks, task_id).await? == AgentState::Canceled {
                    return Ok(());
                }

                match chunk? {
                    ChatChunk::Choice(mut choice) => {
                        if let Some(text) = &choice.delta.content {
                            round_content.push_str(text);
                        }
                        if let Some(text) = &choice.delta.reasoning_content {
                            round_reasoning.push_str(text);
                        }
                        tool_calls.apply_choice_delta(&mut choice);
                        let should_run_tools =
                            choice.finish_reason.as_deref() == Some("tool_calls");
                        publish(
                            &self.events,
                            AgentEvent::ModelChoice {
                                session_id,
                                task_id,
                                round_id,
                                choice,
                            },
                        );

                        if should_run_tools {
                            for tool_call in tool_calls.take_completed()? {
                                tool_results.push(
                                    self.run_tool_call(task_id, session_id, round_id, tool_call)
                                        .await?,
                                );
                            }
                        }
                    }
                    ChatChunk::Content(text) => {
                        round_content.push_str(&text);
                        publish(
                            &self.events,
                            AgentEvent::AssistantToken {
                                session_id,
                                task_id,
                                round_id,
                                text,
                            },
                        );
                    }
                    ChatChunk::Reasoning(text) => {
                        round_reasoning.push_str(&text);
                        publish(
                            &self.events,
                            AgentEvent::AssistantReasoning {
                                session_id,
                                task_id,
                                round_id,
                                text,
                            },
                        );
                    }
                    ChatChunk::Usage(usage) => {
                        final_usage = Some(usage.clone());
                        publish(
                            &self.events,
                            AgentEvent::ModelRoundFinished {
                                session_id,
                                task_id,
                                round_id,
                                usage: Some(usage),
                            },
                        );
                        sent_round_finished = true;
                    }
                    ChatChunk::Finished => {
                        if !sent_round_finished {
                            publish(
                                &self.events,
                                AgentEvent::ModelRoundFinished {
                                    session_id,
                                    task_id,
                                    round_id,
                                    usage: final_usage,
                                },
                            );
                        }
                        break;
                    }
                }
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

            append_tool_results_to_context(
                &mut messages,
                round_content,
                round_reasoning,
                tool_results,
            );
        }

        Err(SeekCodeError::ModelProvider(
            "model requested too many tool rounds".to_string(),
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

        let tool_call_id = tool_call.id;
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
                tool_call_id,
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
                    workspace_id: self.tool_context.workspace_id.clone(),
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
                let content = serde_json::to_string(&output.content).unwrap_or_default();
                publish(
                    &self.events,
                    tool_finished_event(
                        task_id,
                        session_id,
                        round_id,
                        tool_call_id,
                        name,
                        Ok(output),
                    ),
                );
                content
            }
            Err(error) => {
                publish(
                    &self.events,
                    tool_finished_event(
                        task_id,
                        session_id,
                        round_id,
                        tool_call_id,
                        name,
                        Err(error),
                    ),
                );
                set_task_state(&self.tasks, task_id, AgentState::Thinking).await?;
                publish_state(&self.events, session_id, task_id, AgentState::Thinking);
                return Err(SeekCodeError::ToolExecution(
                    "tool execution failed".to_string(),
                ));
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

/// Appends the assistant turn and tool results back into the running context.
pub(crate) fn append_tool_results_to_context(
    messages: &mut Vec<ChatMessage>,
    content: String,
    reasoning_content: String,
    tool_results: Vec<ToolCallRunResult>,
) {
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
                    "arguments": serde_json::to_string(&result.tool_call.arguments).unwrap_or_default()
                }
            })
        })
        .collect();
    messages.push(assistant);

    for result in tool_results {
        let mut message = ChatMessage::new(ChatRole::Tool, result.result_content);
        message.tool_call_id = Some(result.tool_call.id);
        messages.push(message);
    }
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
                    id: ToolCallId::new(),
                    name: None,
                    arguments: String::new(),
                });
            delta.id = Some(partial.id);

            if let Some(name) = &delta.name {
                partial.name = Some(name.clone());
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
                let name = partial.name.ok_or_else(|| {
                    SeekCodeError::ModelProvider("missing streamed tool call name".to_string())
                })?;
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
                    name,
                    arguments,
                })
            })
            .collect()
    }
}

/// Partial tool call assembled across streamed deltas.
struct PartialToolCall {
    id: ToolCallId,
    name: Option<String>,
    arguments: String,
}
