use crate::kernel::SessionService;
use seekcode_agent_core::AgentEvent;
use seekcode_common::{ModelCallLogId, SessionId, TaskId, TokenUsage};
use seekcode_storage::{local_now_text, NewModelCallLog};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

pub(crate) fn spawn_session_agent_event_bridge(
    sessions: Arc<SessionService>,
    mut events: mpsc::UnboundedReceiver<AgentEvent>,
    ui_events: mpsc::UnboundedSender<AgentEvent>,
    task_id: TaskId,
    session_id: SessionId,
    turn_sequence: i64,
) {
    tokio::spawn(async move {
        let mut content = String::new();
        let mut reasoning = String::new();
        let model_provider = sessions
            .get_session(session_id)
            .await
            .map(|session| session.model_provider)
            .unwrap_or_else(|_| "deepseek".to_string());
        let mut active_model_call: Option<PendingModelCallLog> = None;

        loop {
            match events.recv().await {
                Some(AgentEvent::ModelChoice {
                    session_id: event_session_id,
                    task_id: event_task_id,
                    round_id,
                    choice,
                    ..
                }) if event_task_id == task_id && event_session_id == session_id => {
                    if let Some(text) = &choice.delta.content {
                        content.push_str(text);
                    }
                    if let Some(text) = &choice.delta.reasoning_content {
                        reasoning.push_str(text);
                    }
                    sessions
                        .persist_assistant_choice(session_id, turn_sequence, &choice)
                        .await;
                    if let Err(error) = ui_events.send(AgentEvent::ModelChoice {
                        session_id: event_session_id,
                        task_id: event_task_id,
                        round_id,
                        choice,
                    }) {
                        tracing::warn!(
                            target: "seekcode_app_kernel::agent_events",
                            %session_id,
                            %task_id,
                            %error,
                            "failed to forward model choice event"
                        );
                    }
                }
                Some(AgentEvent::AssistantToken {
                    task_id: event_task_id,
                    session_id: event_session_id,
                    round_id,
                    text,
                    ..
                }) if event_task_id == task_id && event_session_id == session_id => {
                    content.push_str(&text);
                    if let Err(error) = ui_events.send(AgentEvent::AssistantToken {
                        session_id: event_session_id,
                        task_id: event_task_id,
                        round_id,
                        text,
                    }) {
                        tracing::warn!(
                            target: "seekcode_app_kernel::agent_events",
                            %session_id,
                            %task_id,
                            %error,
                            "failed to forward assistant token event"
                        );
                    }
                }
                Some(AgentEvent::AssistantReasoning {
                    task_id: event_task_id,
                    session_id: event_session_id,
                    round_id,
                    text,
                    ..
                }) if event_task_id == task_id && event_session_id == session_id => {
                    reasoning.push_str(&text);
                    if let Err(error) = ui_events.send(AgentEvent::AssistantReasoning {
                        session_id: event_session_id,
                        task_id: event_task_id,
                        round_id,
                        text,
                    }) {
                        tracing::warn!(
                            target: "seekcode_app_kernel::agent_events",
                            %session_id,
                            %task_id,
                            %error,
                            "failed to forward assistant reasoning event"
                        );
                    }
                }
                Some(event)
                    if agent_event_task_id(&event) == Some(task_id)
                        && agent_event_session_id(&event) == Some(session_id)
                        && is_terminal_agent_event(&event) =>
                {
                    match &event {
                        AgentEvent::Failed { error, .. } => {
                            tracing::error!(
                                target: "seekcode_app_kernel::agent_events",
                                %session_id,
                                %task_id,
                                %error,
                                "agent task finished with failure"
                            );
                        }
                        AgentEvent::Canceled { .. } => {
                            tracing::warn!(
                                target: "seekcode_app_kernel::agent_events",
                                %session_id,
                                %task_id,
                                "agent task was canceled"
                            );
                        }
                        AgentEvent::Finished { .. } => {}
                        _ => {}
                    }
                    if let Some(call) = active_model_call.take() {
                        let success = matches!(event, AgentEvent::Finished { .. });
                        sessions
                            .append_model_call_log(call.into_new_log(
                                model_provider.clone(),
                                session_id,
                                success,
                                None,
                            ))
                            .await;
                    }
                    if let Err(error) = ui_events.send(event) {
                        tracing::warn!(
                            target: "seekcode_app_kernel::agent_events",
                            %session_id,
                            %task_id,
                            %error,
                            "failed to forward terminal agent event"
                        );
                    }
                    break;
                }
                Some(event)
                    if agent_event_task_id(&event) == Some(task_id)
                        && agent_event_session_id(&event) == Some(session_id) =>
                {
                    match &event {
                        AgentEvent::ModelRequestStarted {
                            round_id, model, ..
                        } => {
                            active_model_call = Some(PendingModelCallLog {
                                round_id: *round_id,
                                model: model.clone(),
                                started_at: Instant::now(),
                                called_at: local_now_text(),
                            });
                        }
                        AgentEvent::ModelRoundFinished {
                            round_id, usage, ..
                        } => {
                            if let Some(usage) = usage {
                                // Record the latest input token count so the next
                                // task can decide whether to compact the context.
                                sessions
                                    .update_last_input_tokens(
                                        session_id,
                                        i64::from(usage.prompt_tokens),
                                    )
                                    .await;
                            }
                            if active_model_call
                                .as_ref()
                                .is_some_and(|call| call.round_id == *round_id)
                            {
                                let call = active_model_call.take().expect("active call exists");
                                sessions
                                    .append_model_call_log(call.into_new_log(
                                        model_provider.clone(),
                                        session_id,
                                        true,
                                        usage.clone(),
                                    ))
                                    .await;
                            }
                        }
                        AgentEvent::ToolCallFinished {
                            tool_call_id,
                            output,
                            error,
                            ..
                        } => {
                            sessions
                                .persist_tool_result(
                                    session_id,
                                    turn_sequence,
                                    *tool_call_id,
                                    output.clone(),
                                    error.clone(),
                                )
                                .await;
                        }
                        _ => {}
                    }
                    if let Err(error) = ui_events.send(event) {
                        tracing::warn!(
                            target: "seekcode_app_kernel::agent_events",
                            %session_id,
                            %task_id,
                            %error,
                            "failed to forward agent event"
                        );
                    }
                }
                Some(_) => {}
                None => {
                    tracing::warn!(
                        target: "seekcode_app_kernel::agent_events",
                        %session_id,
                        %task_id,
                        "agent event channel closed before terminal event"
                    );
                    break;
                }
            }
        }

        sessions.finish_running_task(session_id);
    });
}

struct PendingModelCallLog {
    round_id: u32,
    model: String,
    started_at: Instant,
    called_at: String,
}

impl PendingModelCallLog {
    fn into_new_log(
        self,
        model_provider: String,
        session_id: SessionId,
        success: bool,
        usage: Option<TokenUsage>,
    ) -> NewModelCallLog {
        let usage = usage.unwrap_or_default();
        NewModelCallLog {
            id: ModelCallLogId::new(),
            model_provider,
            model: self.model,
            session_id,
            input_tokens: i64::from(usage.prompt_tokens),
            output_tokens: i64::from(usage.completion_tokens),
            cache_hit_tokens: i64::from(usage.cached_tokens),
            elapsed_ms: self.started_at.elapsed().as_millis().min(i64::MAX as u128) as i64,
            success,
            called_at: self.called_at,
        }
    }
}

fn agent_event_task_id(event: &AgentEvent) -> Option<TaskId> {
    match event {
        AgentEvent::TaskStarted { task_id, .. }
        | AgentEvent::StateChanged { task_id, .. }
        | AgentEvent::ModelRequestStarted { task_id, .. }
        | AgentEvent::ModelChoice { task_id, .. }
        | AgentEvent::AssistantToken { task_id, .. }
        | AgentEvent::AssistantReasoning { task_id, .. }
        | AgentEvent::ToolCallStarted { task_id, .. }
        | AgentEvent::ToolCallFinished { task_id, .. }
        | AgentEvent::ModelRoundFinished { task_id, .. }
        | AgentEvent::Finished { task_id, .. }
        | AgentEvent::Failed { task_id, .. }
        | AgentEvent::Canceled { task_id, .. }
        | AgentEvent::ContextCompactionStarted { task_id, .. }
        | AgentEvent::ContextCompactionCanceled { task_id, .. }
        | AgentEvent::ContextCompactionFinished { task_id, .. } => Some(*task_id),
    }
}

fn agent_event_session_id(event: &AgentEvent) -> Option<SessionId> {
    match event {
        AgentEvent::TaskStarted { session_id, .. }
        | AgentEvent::StateChanged { session_id, .. }
        | AgentEvent::ModelRequestStarted { session_id, .. }
        | AgentEvent::ModelChoice { session_id, .. }
        | AgentEvent::AssistantToken { session_id, .. }
        | AgentEvent::AssistantReasoning { session_id, .. }
        | AgentEvent::ToolCallStarted { session_id, .. }
        | AgentEvent::ToolCallFinished { session_id, .. }
        | AgentEvent::ModelRoundFinished { session_id, .. }
        | AgentEvent::Finished { session_id, .. }
        | AgentEvent::Failed { session_id, .. }
        | AgentEvent::Canceled { session_id, .. }
        | AgentEvent::ContextCompactionStarted { session_id, .. }
        | AgentEvent::ContextCompactionCanceled { session_id, .. }
        | AgentEvent::ContextCompactionFinished { session_id, .. } => Some(*session_id),
    }
}

fn is_terminal_agent_event(event: &AgentEvent) -> bool {
    matches!(
        event,
        AgentEvent::Finished { .. } | AgentEvent::Failed { .. } | AgentEvent::Canceled { .. }
    )
}
