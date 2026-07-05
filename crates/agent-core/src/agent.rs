//! Provider-backed agent runtime and task lifecycle management.

use parking_lot::RwLock;
use seekcode_common::{SeekCodeError, SeekCodeResult, SessionId, TaskId, WorkspaceId};
use seekcode_deepseek_client::ModelProvider;
use seekcode_tool_system::ToolRegistry;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::AbortHandle;

use crate::config::AgentConfig;
use crate::context::{AgentContext, AgentContextPreparer, AgentTaskContext};
use crate::event::{publish, publish_state, AgentEvent};
use crate::runner::{AgentRunRequest, AgentTaskRunner};
use crate::task::{AgentState, AgentTask, StartTaskRequest};

/// Provider-backed agent runtime.
pub struct Agent {
    config: AgentConfig,
    provider: Arc<RwLock<Arc<dyn ModelProvider>>>,
    _tools: Arc<ToolRegistry>,
    tasks: Arc<RwLock<HashMap<TaskId, AgentTask>>>,
    task_controls: Arc<RwLock<HashMap<TaskId, TaskControl>>>,
}

/// Workspace data passed to tools for one agent task.
#[derive(Clone, Debug, Default)]
pub struct AgentToolContext {
    /// Workspace identifier associated with the task.
    pub workspace_id: Option<WorkspaceId>,
    /// Workspace root used to resolve relative tool paths.
    pub workspace_root: Option<PathBuf>,
}

impl AgentToolContext {
    /// Creates tool context for a task running inside a workspace.
    pub fn workspace(workspace_id: WorkspaceId, workspace_root: impl Into<PathBuf>) -> Self {
        Self {
            workspace_id: Some(workspace_id),
            workspace_root: Some(workspace_root.into()),
        }
    }
}

/// Handle used to abort the background task once it is running.
struct TaskControl {
    abort_handle: AbortHandle,
}

/// Emits terminal cancellation events when a running task is dropped mid-flight.
struct RunCompletionGuard {
    task_id: TaskId,
    session_id: SessionId,
    tasks: Arc<RwLock<HashMap<TaskId, AgentTask>>>,
    task_controls: Arc<RwLock<HashMap<TaskId, TaskControl>>>,
    events: mpsc::UnboundedSender<AgentEvent>,
}

impl Drop for RunCompletionGuard {
    fn drop(&mut self) {
        self.task_controls.write().remove(&self.task_id);

        let is_canceled = self
            .tasks
            .read()
            .get(&self.task_id)
            .map(|task| task.state == AgentState::Canceled)
            .unwrap_or(false);
        if !is_canceled {
            return;
        }

        publish_state(
            &self.events,
            self.session_id,
            self.task_id,
            AgentState::Canceled,
        );
        publish(
            &self.events,
            AgentEvent::Canceled {
                session_id: self.session_id,
                task_id: self.task_id,
            },
        );
    }
}

impl Agent {
    /// Creates a new agent runtime.
    pub fn new(
        config: AgentConfig,
        provider: Arc<dyn ModelProvider>,
        tools: Arc<ToolRegistry>,
    ) -> Self {
        Self {
            config,
            provider: Arc::new(RwLock::new(provider)),
            _tools: tools,
            tasks: Arc::new(RwLock::new(HashMap::new())),
            task_controls: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Returns the active agent configuration.
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Replaces the model provider used by newly-started tasks.
    pub fn set_provider(&self, provider: Arc<dyn ModelProvider>) {
        *self.provider.write() = provider;
    }

    /// Starts a new task with optional application-level context preparation.
    pub async fn start_task_with_messages_tools_and_context_preparer(
        &self,
        request: StartTaskRequest,
        context: AgentTaskContext,
        tools: Arc<ToolRegistry>,
        events: mpsc::UnboundedSender<AgentEvent>,
        context_preparer: Option<Arc<dyn AgentContextPreparer>>,
    ) -> SeekCodeResult<AgentTask> {
        self.start_task_with_messages_tools_tool_context_and_context_preparer(
            request,
            context,
            tools,
            AgentToolContext::default(),
            events,
            context_preparer,
        )
        .await
    }

    /// Starts a new task with tool workspace context and optional context preparation.
    pub async fn start_task_with_messages_tools_tool_context_and_context_preparer(
        &self,
        request: StartTaskRequest,
        context: AgentTaskContext,
        tools: Arc<ToolRegistry>,
        tool_context: AgentToolContext,
        events: mpsc::UnboundedSender<AgentEvent>,
        context_preparer: Option<Arc<dyn AgentContextPreparer>>,
    ) -> SeekCodeResult<AgentTask> {
        let prompt = request.prompt.trim();
        if prompt.is_empty() {
            return Err(SeekCodeError::Validation(
                "agent task prompt cannot be empty".to_string(),
            ));
        }
        let prompt = prompt.to_string();

        let task_id = TaskId::new();
        let session_id = request.session_id;
        self.insert_task(AgentTask {
            id: task_id,
            state: AgentState::Queued,
        })
        .await;
        self.set_task_state(task_id, AgentState::Thinking).await?;

        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.config.default_model.clone());
        let thinking = request.thinking.unwrap_or(self.config.thinking);
        let reasoning_effort = request.reasoning_effort.clone();
        let context = AgentContext::new(task_id, session_id, context);

        publish(
            &events,
            AgentEvent::TaskStarted {
                session_id,
                task_id,
                model: model.clone(),
            },
        );
        publish_state(&events, session_id, task_id, AgentState::Thinking);

        let runner = AgentTaskRunner {
            config: self.config.clone(),
            provider: self.provider.read().clone(),
            tools,
            tasks: self.tasks.clone(),
            events: events.clone(),
            context_preparer,
            tool_context,
        };
        let task_controls = self.task_controls.clone();
        let completion_guard = RunCompletionGuard {
            task_id,
            session_id,
            tasks: self.tasks.clone(),
            task_controls: task_controls.clone(),
            events: events.clone(),
        };
        let handle = tokio::spawn(async move {
            let _completion_guard = completion_guard;
            runner
                .run(AgentRunRequest {
                    task_id,
                    session_id,
                    model,
                    thinking,
                    reasoning_effort,
                    prompt,
                    context,
                })
                .await;
        });
        self.task_controls.write().insert(
            task_id,
            TaskControl {
                abort_handle: handle.abort_handle(),
            },
        );
        if matches!(
            self.task_state(task_id).await?,
            AgentState::Completed | AgentState::Canceled | AgentState::Failed
        ) {
            self.task_controls.write().remove(&task_id);
        }

        let task = AgentTask {
            id: task_id,
            state: AgentState::Thinking,
        };
        Ok(task)
    }

    /// Cancels a running task.
    pub async fn cancel_task(&self, task_id: TaskId) -> SeekCodeResult<()> {
        {
            let mut tasks = self.tasks.write();
            let task = tasks
                .get_mut(&task_id)
                .ok_or_else(|| SeekCodeError::NotFound(format!("agent task '{task_id}'")))?;

            if matches!(
                task.state,
                AgentState::Completed | AgentState::Canceled | AgentState::Failed
            ) {
                return Ok(());
            }

            task.state = AgentState::Canceled;
        }

        if let Some(control) = self.task_controls.write().remove(&task_id) {
            control.abort_handle.abort();
        }

        Ok(())
    }

    /// Resumes a paused or persisted task.
    pub async fn resume_task(&self, task_id: TaskId) -> SeekCodeResult<AgentTask> {
        let mut tasks = self.tasks.write();
        let task = tasks
            .get_mut(&task_id)
            .ok_or_else(|| SeekCodeError::NotFound(format!("agent task '{task_id}'")))?;

        if matches!(task.state, AgentState::Canceled | AgentState::Failed) {
            task.state = AgentState::Queued;
        }

        Ok(task.clone())
    }

    pub(crate) async fn insert_task(&self, task: AgentTask) {
        self.tasks.write().insert(task.id, task);
    }

    async fn set_task_state(&self, task_id: TaskId, state: AgentState) -> SeekCodeResult<()> {
        let mut tasks = self.tasks.write();
        let task = tasks
            .get_mut(&task_id)
            .ok_or_else(|| SeekCodeError::NotFound(format!("agent task '{task_id}'")))?;
        task.state = state;
        Ok(())
    }

    pub(crate) async fn task_state(&self, task_id: TaskId) -> SeekCodeResult<AgentState> {
        self.tasks
            .read()
            .get(&task_id)
            .map(|task| task.state.clone())
            .ok_or_else(|| SeekCodeError::NotFound(format!("agent task '{task_id}'")))
    }
}
