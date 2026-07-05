use crate::session_service::SessionService;
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
        let model_provider = sessions
            .get_session(session_id)
            .await
            .map(|session| session.model_provider)
            .unwrap_or_else(|_| "deepseek".to_string());
        let mut active_model_call: Option<PendingModelCallLog> = None;

        loop {
            match events.recv().await {
                Some(AgentEvent::AssistantMessageDelta {
                    session_id: event_session_id,
                    task_id: event_task_id,
                    round_id,
                    content,
                    reasoning_content,
                    ..
                }) if event_task_id == task_id && event_session_id == session_id => {
                    if let Err(error) = ui_events.send(AgentEvent::AssistantMessageDelta {
                        session_id: event_session_id,
                        task_id: event_task_id,
                        round_id,
                        content,
                        reasoning_content,
                    }) {
                        tracing::warn!(
                            target: "seekcode_app_kernel::agent_events",
                            %session_id,
                            %task_id,
                            %error,
                            "failed to forward assistant message delta event"
                        );
                    }
                }
                Some(event)
                    if event.task_id() == task_id
                        && event.session_id() == session_id
                        && event.is_terminal() =>
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
                Some(event) if event.task_id() == task_id && event.session_id() == session_id => {
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
                        AgentEvent::ModelRequestRetrying {
                            round_id, error, ..
                        } => {
                            tracing::warn!(
                                target: "seekcode_app_kernel::agent_events",
                                %session_id,
                                %task_id,
                                round_id,
                                %error,
                                "model request attempt failed and will be retried"
                            );
                            if active_model_call
                                .as_ref()
                                .is_some_and(|call| call.round_id == *round_id)
                            {
                                if let Some(call) = active_model_call.take() {
                                    sessions
                                        .append_model_call_log(call.into_new_log(
                                            model_provider.clone(),
                                            session_id,
                                            false,
                                            None,
                                        ))
                                        .await;
                                }
                            }
                        }
                        AgentEvent::ModelRoundFinished {
                            round_id,
                            assistant_message,
                            tool_messages,
                            usage,
                            ..
                        } => {
                            sessions
                                .persist_model_round_messages(
                                    session_id,
                                    turn_sequence,
                                    assistant_message,
                                    tool_messages,
                                )
                                .await;
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
                                if let Some(call) = active_model_call.take() {
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
                        }
                        AgentEvent::ToolCallFinished { .. } => {
                            // Tool results are persisted with the completed model round.
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
