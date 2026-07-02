//! Tool registry, schemas, and executor traits.

use async_trait::async_trait;
use schemars::schema::RootSchema;
use seekcode_common::{SeekCodeError, SeekCodeResult, TaskId, WorkspaceId};
use seekcode_deepseek_client::ToolSpec;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;

/// Name for the run command tool.
pub const RUN_COMMAND_TOOL: &str = "run_command";

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Raw JSON input passed to a tool.
pub type ToolInput = Value;

/// Tool output returned to the agent loop.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolOutput {
    /// Machine-readable result payload.
    pub content: Value,
    /// Short human-readable summary.
    pub summary: String,
}

/// Context available to a tool execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolContext {
    /// Current task identifier.
    pub task_id: TaskId,
    /// Current workspace identifier.
    pub workspace_id: Option<WorkspaceId>,
    /// Absolute workspace root available to local system tools.
    pub workspace_root: Option<PathBuf>,
}

/// Persistable tool execution record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolExecution {
    /// Tool call identifier.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Raw input.
    pub input: ToolInput,
    /// Output if execution succeeded.
    pub output: Option<ToolOutput>,
    /// Error message if execution failed.
    pub error: Option<String>,
}

/// Runtime configuration for workspace-scoped system tools.
#[derive(Clone, Debug)]
pub struct SystemToolConfig {
    /// Maximum file size read by text tools.
    pub max_file_bytes: u64,
    /// Maximum command output retained per stream.
    pub max_command_output_bytes: usize,
    /// Default command timeout.
    pub command_timeout: Duration,
    /// Maximum search matches returned by default.
    pub max_search_results: usize,
}

impl SystemToolConfig {
    /// Creates a system tool config with default limits.
    pub fn new() -> Self {
        Self {
            max_file_bytes: 10 * 1024 * 1024,
            max_command_output_bytes: 512 * 1024,
            command_timeout: Duration::from_secs(120),
            max_search_results: 200,
        }
    }
}

impl Default for SystemToolConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Registers all built-in workspace-scoped system tools.
pub fn register_system_tools(
    registry: &mut ToolRegistry,
    config: SystemToolConfig,
) -> SeekCodeResult<()> {
    registry.register(RunCommandTool::new(config))?;
    Ok(())
}

/// Creates a registry preloaded with all built-in system tools.
pub fn system_tool_registry(config: SystemToolConfig) -> SeekCodeResult<ToolRegistry> {
    let mut registry = ToolRegistry::new();
    register_system_tools(&mut registry, config)?;
    Ok(registry)
}

/// Tool that runs a non-interactive command in the workspace.
pub struct RunCommandTool {
    config: SystemToolConfig,
}

impl RunCommandTool {
    /// Creates a run command tool.
    pub fn new(config: SystemToolConfig) -> Self {
        Self { config }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RunCommandInput {
    /// Command line to execute.
    command: String,
    /// Working directory for the command. Defaults to the turn cwd.
    #[serde(default)]
    cwd: Option<PathBuf>,
    /// Timeout in milliseconds.
    #[serde(default)]
    timeout_ms: Option<u64>,
}

#[derive(Debug, Serialize)]
struct CommandResult {
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
    timed_out: bool,
    truncated: bool,
}

/// Executable tool exposed to the agent.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Returns the stable tool name.
    fn name(&self) -> &'static str;

    /// Returns a concise model-facing description.
    fn description(&self) -> &'static str;

    /// Returns the JSON schema for tool input.
    fn input_schema(&self) -> RootSchema;

    /// Executes the tool with validated JSON input.
    async fn execute(&self, ctx: ToolContext, input: ToolInput) -> SeekCodeResult<ToolOutput>;
}

/// Registry for all tools available to an agent.
#[derive(Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Creates an empty tool registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a tool and rejects duplicate names.
    pub fn register<T>(&mut self, tool: T) -> SeekCodeResult<()>
    where
        T: Tool + 'static,
    {
        let name = tool.name().to_string();
        if self.tools.contains_key(&name) {
            return Err(SeekCodeError::Validation(format!(
                "tool '{name}' is already registered"
            )));
        }

        self.tools.insert(name, Arc::new(tool));
        Ok(())
    }

    /// Returns a registered tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Converts registered tools into provider-facing tool specs.
    pub fn tool_specs(&self, strict: bool) -> Vec<ToolSpec> {
        self.tools
            .values()
            .map(|tool| ToolSpec {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                input_schema: serde_json::to_value(tool.input_schema()).unwrap_or(Value::Null),
                strict,
            })
            .collect()
    }

    /// Executes a registered tool.
    pub async fn execute(
        &self,
        name: &str,
        ctx: ToolContext,
        input: ToolInput,
    ) -> SeekCodeResult<ToolOutput> {
        let tool = self
            .get(name)
            .ok_or_else(|| SeekCodeError::NotFound(format!("tool '{name}'")))?;

        tool.execute(ctx, input).await
    }
}

/// Input schema marker for tools that accept arbitrary JSON.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AnyToolInput {
    /// Arbitrary JSON payload.
    pub value: Value,
}

#[async_trait]
impl Tool for RunCommandTool {
    fn name(&self) -> &'static str {
        RUN_COMMAND_TOOL
    }

    fn description(&self) -> &'static str {
        run_command_description()
    }

    fn input_schema(&self) -> RootSchema {
        schemars::schema_for!(RunCommandInput)
    }

    async fn execute(&self, ctx: ToolContext, input: ToolInput) -> SeekCodeResult<ToolOutput> {
        let workspace_root = workspace_root_from_context(&ctx)?;
        let input: RunCommandInput = parse_input(input)?;
        if input.command.trim().is_empty() {
            return Err(SeekCodeError::Validation(
                "command cannot be empty".to_string(),
            ));
        }

        let cwd = input.cwd.unwrap_or_else(|| PathBuf::from("."));
        let cwd = resolve_existing_workspace_path(&workspace_root, &cwd)?;
        if !cwd.is_dir() {
            return Err(SeekCodeError::Shell(format!(
                "command cwd is not a directory: {}",
                relative_display(&workspace_root, &cwd)
            )));
        }

        let timeout = input
            .timeout_ms
            .map(Duration::from_millis)
            .unwrap_or(self.config.command_timeout);
        let result = run_command_line(&self.config, &input.command, &cwd, timeout).await?;

        Ok(ToolOutput {
            content: serde_json::to_value(&result)
                .map_err(|error| SeekCodeError::Internal(error.to_string()))?,
            summary: format!(
                "Command exited with {}",
                result
                    .exit_code
                    .map(|code| code.to_string())
                    .unwrap_or_else(|| "unknown status".to_string())
            ),
        })
    }
}

fn parse_input<T>(input: ToolInput) -> SeekCodeResult<T>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_value(input).map_err(|error| SeekCodeError::Validation(error.to_string()))
}

/// Returns the canonical workspace root supplied for this tool execution.
fn workspace_root_from_context(ctx: &ToolContext) -> SeekCodeResult<PathBuf> {
    let root = ctx.workspace_root.as_ref().ok_or_else(|| {
        SeekCodeError::Workspace("workspace root was not provided to the tool".to_string())
    })?;
    let root = root
        .canonicalize()
        .map_err(|error| SeekCodeError::Workspace(error.to_string()))?;
    if !root.is_dir() {
        return Err(SeekCodeError::Workspace(format!(
            "workspace root is not a directory: {}",
            root.display()
        )));
    }

    Ok(root)
}

/// Joins an input path against the workspace root. Absolute paths are used
/// as-is; relative paths are resolved against the workspace root. Tools are not
/// restricted to the workspace root.
fn join_workspace_path(root: &Path, input: &Path) -> PathBuf {
    if input.is_absolute() {
        input.to_path_buf()
    } else {
        root.join(input)
    }
}

/// Resolves an existing path for reading or traversal.
fn resolve_existing_workspace_path(root: &Path, input: &Path) -> SeekCodeResult<PathBuf> {
    join_workspace_path(root, input)
        .canonicalize()
        .map_err(|error| SeekCodeError::Workspace(error.to_string()))
}

fn relative_display(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Builds the base shell command for the current platform.
fn build_command(command_line: &str) -> Command {
    #[cfg(windows)]
    {
        let mut command = Command::new("powershell");
        command.args(["-NoProfile", "-NonInteractive", "-Command", command_line]);
        command.creation_flags(CREATE_NO_WINDOW);
        command
    }

    #[cfg(not(windows))]
    {
        let mut command = Command::new("sh");
        command.args(["-c", command_line]);
        command
    }
}

#[cfg(windows)]
fn run_command_description() -> &'static str {
    "Runs a Powershell command (Windows) and returns its output.\n\nExamples of valid command strings:\n\n- ls -a (show hidden): \"Get-ChildItem -Force\"\n- recursive find by name: \"Get-ChildItem -Recurse -Filter *.py\"\n- recursive grep: \"Get-ChildItem -Path C:\\\\myrepo -Recurse | Select-String -Pattern 'TODO' -CaseSensitive\"\n- ps aux | grep python: \"Get-Process | Where-Object { $_.ProcessName -like '*python*' }\"\n- setting an env var: \"$env:FOO='bar'; echo $env:FOO\"\n- running an inline Python script: \"@'\nprint('Hello, world!')\n'@ | python -\"\n\nWindows safety rules:\n- Do not compose destructive filesystem commands across shells. Do not enumerate paths in PowerShell and then pass them to `cmd /c`, batch builtins, or another shell for deletion or moving. Use one shell end-to-end, prefer native PowerShell cmdlets such as `Remove-Item` / `Move-Item` with `-LiteralPath`, and avoid string-built shell commands for file operations.\n- Before any recursive delete or move on Windows, verify the resolved absolute target paths stay within the intended workspace or explicitly named target directory. Never issue a recursive delete or move against a computed path if the final target has not been checked.\n- When using `Start-Process` to launch a background helper or service, pass `-WindowStyle Hidden` unless the user explicitly asked for a visible interactive window. Use visible windows only for interactive tools the user needs to see or control."
}

#[cfg(not(windows))]
fn run_command_description() -> &'static str {
    "Run a non-interactive command line through sh. The working directory may be absolute or relative to the workspace root."
}

async fn run_command_line(
    config: &SystemToolConfig,
    command_line: &str,
    cwd: &Path,
    timeout: Duration,
) -> SeekCodeResult<CommandResult> {
    let mut command = build_command(command_line);
    command
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let child = command
        .spawn()
        .map_err(|error| SeekCodeError::Shell(error.to_string()))?;

    let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(output) => output.map_err(|error| SeekCodeError::Shell(error.to_string()))?,
        Err(_) => {
            return Ok(CommandResult {
                exit_code: None,
                stdout: String::new(),
                stderr: format!("command timed out after {} ms", timeout.as_millis()),
                timed_out: true,
                truncated: false,
            });
        }
    };

    let (stdout, stdout_truncated) = truncate_utf8(
        &String::from_utf8_lossy(&output.stdout),
        config.max_command_output_bytes,
    );
    let (stderr, stderr_truncated) = truncate_utf8(
        &String::from_utf8_lossy(&output.stderr),
        config.max_command_output_bytes,
    );

    Ok(CommandResult {
        exit_code: output.status.code(),
        stdout,
        stderr,
        timed_out: false,
        truncated: stdout_truncated || stderr_truncated,
    })
}

fn truncate_utf8(value: &str, max_bytes: usize) -> (String, bool) {
    if value.len() <= max_bytes {
        return (value.to_string(), false);
    }

    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    (format!("{}...[truncated]", &value[..end]), true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct DummyTool;

    #[async_trait]
    impl Tool for DummyTool {
        fn name(&self) -> &'static str {
            "dummy"
        }

        fn description(&self) -> &'static str {
            "Dummy tool used in tests."
        }

        fn input_schema(&self) -> RootSchema {
            schemars::schema_for!(AnyToolInput)
        }

        async fn execute(
            &self,
            _ctx: ToolContext,
            _input: ToolInput,
        ) -> SeekCodeResult<ToolOutput> {
            Ok(ToolOutput {
                content: Value::Null,
                summary: "ok".to_string(),
            })
        }
    }

    #[test]
    fn duplicate_tool_names_are_rejected() {
        let mut registry = ToolRegistry::new();

        registry
            .register(DummyTool)
            .expect("first register succeeds");
        let error = registry.register(DummyTool).expect_err("duplicate fails");

        assert!(matches!(error, SeekCodeError::Validation(_)));
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn system_run_command_executes_via_powershell_on_windows() {
        let root = make_temp_workspace();
        let config = SystemToolConfig::new();
        let tool = RunCommandTool::new(config);

        let output = tool
            .execute(
                ToolContext {
                    task_id: TaskId::new(),
                    workspace_id: None,
                    workspace_root: Some(root.clone()),
                },
                serde_json::json!({ "command": "cmd /c echo seekcode" }),
            )
            .await
            .expect("command runs");

        assert!(output.content["stdout"]
            .as_str()
            .expect("stdout is a string")
            .contains("seekcode"));
        assert_eq!(output.content["exit_code"], 0);
        let _ = std::fs::remove_dir_all(root);
    }

    fn make_temp_workspace() -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time is after epoch")
            .as_nanos();
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "seekcode-tool-system-test-{}-{suffix}-{counter}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).expect("create temp workspace");
        root
    }
}
