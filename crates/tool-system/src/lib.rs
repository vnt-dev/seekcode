//! Tool registry, schemas, and executor traits.

use async_trait::async_trait;
use schemars::schema::RootSchema;
use seekcode_common::{SeekCodeError, SeekCodeResult, TaskId, ToolCallId, WorkspaceId};
use seekcode_model_provider::ToolSpec;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use walkdir::WalkDir;

/// Name for the read file tool.
pub const READ_FILE_TOOL: &str = "read_file";
/// Name for the write file tool.
pub const WRITE_FILE_TOOL: &str = "write_file";
/// Name for the list files tool.
pub const LIST_FILES_TOOL: &str = "list_files";
/// Name for the text search tool.
pub const SEARCH_TEXT_TOOL: &str = "search_text";
/// Name for the apply patch tool.
pub const APPLY_PATCH_TOOL: &str = "apply_patch";
/// Name for the run command tool.
pub const RUN_COMMAND_TOOL: &str = "run_command";
/// Name for the git status tool.
pub const GIT_STATUS_TOOL: &str = "git_status";
/// Name for the git diff tool.
pub const GIT_DIFF_TOOL: &str = "git_diff";

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
    pub id: ToolCallId,
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
    /// Canonical absolute workspace root.
    pub workspace_root: PathBuf,
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
    /// Creates a system tool config for an existing workspace root.
    pub fn new(workspace_root: impl Into<PathBuf>) -> SeekCodeResult<Self> {
        let workspace_root = workspace_root.into();
        let workspace_root = workspace_root
            .canonicalize()
            .map_err(|error| SeekCodeError::Workspace(error.to_string()))?;
        if !workspace_root.is_dir() {
            return Err(SeekCodeError::Workspace(format!(
                "workspace root is not a directory: {}",
                workspace_root.display()
            )));
        }

        Ok(Self {
            workspace_root,
            max_file_bytes: 2 * 1024 * 1024,
            max_command_output_bytes: 512 * 1024,
            command_timeout: Duration::from_secs(120),
            max_search_results: 200,
        })
    }
}

/// Registers all built-in workspace-scoped system tools.
pub fn register_system_tools(
    registry: &mut ToolRegistry,
    config: SystemToolConfig,
) -> SeekCodeResult<()> {
    registry.register(ReadFileTool::new(config.clone()))?;
    registry.register(WriteFileTool::new(config.clone()))?;
    registry.register(ListFilesTool::new(config.clone()))?;
    registry.register(SearchTextTool::new(config.clone()))?;
    registry.register(ApplyPatchTool::new(config.clone()))?;
    registry.register(RunCommandTool::new(config.clone()))?;
    registry.register(GitStatusTool::new(config.clone()))?;
    registry.register(GitDiffTool::new(config))?;
    Ok(())
}

/// Creates a registry preloaded with all built-in system tools.
pub fn system_tool_registry(config: SystemToolConfig) -> SeekCodeResult<ToolRegistry> {
    let mut registry = ToolRegistry::new();
    register_system_tools(&mut registry, config)?;
    Ok(registry)
}

/// Tool that reads a UTF-8 file from the workspace.
pub struct ReadFileTool {
    config: SystemToolConfig,
}

impl ReadFileTool {
    /// Creates a read file tool.
    pub fn new(config: SystemToolConfig) -> Self {
        Self { config }
    }
}

/// Tool that writes UTF-8 content to a workspace file.
pub struct WriteFileTool {
    config: SystemToolConfig,
}

impl WriteFileTool {
    /// Creates a write file tool.
    pub fn new(config: SystemToolConfig) -> Self {
        Self { config }
    }
}

/// Tool that lists workspace files.
pub struct ListFilesTool {
    config: SystemToolConfig,
}

impl ListFilesTool {
    /// Creates a list files tool.
    pub fn new(config: SystemToolConfig) -> Self {
        Self { config }
    }
}

/// Tool that searches UTF-8 files by substring.
pub struct SearchTextTool {
    config: SystemToolConfig,
}

impl SearchTextTool {
    /// Creates a text search tool.
    pub fn new(config: SystemToolConfig) -> Self {
        Self { config }
    }
}

/// Tool that applies a unified diff with git apply.
pub struct ApplyPatchTool {
    config: SystemToolConfig,
}

impl ApplyPatchTool {
    /// Creates an apply patch tool.
    pub fn new(config: SystemToolConfig) -> Self {
        Self { config }
    }
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

/// Tool that runs git status.
pub struct GitStatusTool {
    config: SystemToolConfig,
}

impl GitStatusTool {
    /// Creates a git status tool.
    pub fn new(config: SystemToolConfig) -> Self {
        Self { config }
    }
}

/// Tool that runs git diff.
pub struct GitDiffTool {
    config: SystemToolConfig,
}

impl GitDiffTool {
    /// Creates a git diff tool.
    pub fn new(config: SystemToolConfig) -> Self {
        Self { config }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ReadFileInput {
    /// Workspace-relative file path.
    path: PathBuf,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct WriteFileInput {
    /// Workspace-relative file path.
    path: PathBuf,
    /// UTF-8 content to write.
    content: String,
    /// Whether missing parent directories may be created.
    #[serde(default)]
    create_dirs: bool,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ListFilesInput {
    /// Workspace-relative directory path.
    #[serde(default)]
    path: Option<PathBuf>,
    /// Maximum traversal depth.
    #[serde(default)]
    max_depth: Option<usize>,
    /// Maximum entries returned.
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SearchTextInput {
    /// Substring pattern to search for.
    pattern: String,
    /// Workspace-relative directory or file path to search.
    #[serde(default)]
    path: Option<PathBuf>,
    /// Maximum matches returned.
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ApplyPatchInput {
    /// Unified diff accepted by git apply.
    patch: String,
    /// Whether to run git apply --check before applying.
    #[serde(default = "default_true")]
    check: bool,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RunCommandInput {
    /// Program to execute without a shell.
    program: String,
    /// Program arguments.
    #[serde(default)]
    args: Vec<String>,
    /// Workspace-relative working directory.
    #[serde(default)]
    cwd: Option<PathBuf>,
    /// Extra environment variables.
    #[serde(default)]
    env: BTreeMap<String, String>,
    /// Timeout in milliseconds.
    #[serde(default)]
    timeout_ms: Option<u64>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct GitDiffInput {
    /// Show staged changes instead of unstaged changes.
    #[serde(default)]
    staged: bool,
    /// Optional workspace-relative path filter.
    #[serde(default)]
    path: Option<PathBuf>,
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
impl Tool for ReadFileTool {
    fn name(&self) -> &'static str {
        READ_FILE_TOOL
    }

    fn description(&self) -> &'static str {
        "Read a UTF-8 text file from the current workspace."
    }

    fn input_schema(&self) -> RootSchema {
        schemars::schema_for!(ReadFileInput)
    }

    async fn execute(&self, _ctx: ToolContext, input: ToolInput) -> SeekCodeResult<ToolOutput> {
        let input: ReadFileInput = parse_input(input)?;
        let path = resolve_existing_workspace_path(&self.config.workspace_root, &input.path)?;
        let metadata = tokio::fs::metadata(&path)
            .await
            .map_err(|error| SeekCodeError::Workspace(error.to_string()))?;
        if metadata.len() > self.config.max_file_bytes {
            return Err(SeekCodeError::Workspace(format!(
                "file is too large to read: {} bytes",
                metadata.len()
            )));
        }

        let content = tokio::fs::read_to_string(&path)
            .await
            .map_err(|error| SeekCodeError::Workspace(error.to_string()))?;
        Ok(ToolOutput {
            content: json!({
                "path": relative_display(&self.config.workspace_root, &path),
                "content": content,
                "bytes": metadata.len(),
            }),
            summary: format!(
                "Read {}",
                relative_display(&self.config.workspace_root, &path)
            ),
        })
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &'static str {
        WRITE_FILE_TOOL
    }

    fn description(&self) -> &'static str {
        "Write UTF-8 content to a file inside the current workspace."
    }

    fn input_schema(&self) -> RootSchema {
        schemars::schema_for!(WriteFileInput)
    }

    async fn execute(&self, _ctx: ToolContext, input: ToolInput) -> SeekCodeResult<ToolOutput> {
        let input: WriteFileInput = parse_input(input)?;
        let path = resolve_new_workspace_path(&self.config.workspace_root, &input.path)?;
        if let Some(parent) = path.parent() {
            if input.create_dirs {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|error| SeekCodeError::Workspace(error.to_string()))?;
            } else if !parent.exists() {
                return Err(SeekCodeError::Workspace(format!(
                    "parent directory does not exist: {}",
                    relative_display(&self.config.workspace_root, parent)
                )));
            }
        }

        tokio::fs::write(&path, input.content.as_bytes())
            .await
            .map_err(|error| SeekCodeError::Workspace(error.to_string()))?;
        Ok(ToolOutput {
            content: json!({
                "path": relative_display(&self.config.workspace_root, &path),
                "bytes": input.content.len(),
            }),
            summary: format!(
                "Wrote {} bytes to {}",
                input.content.len(),
                relative_display(&self.config.workspace_root, &path)
            ),
        })
    }
}

#[async_trait]
impl Tool for ListFilesTool {
    fn name(&self) -> &'static str {
        LIST_FILES_TOOL
    }

    fn description(&self) -> &'static str {
        "List files and directories inside the current workspace."
    }

    fn input_schema(&self) -> RootSchema {
        schemars::schema_for!(ListFilesInput)
    }

    async fn execute(&self, ctx: ToolContext, input: ToolInput) -> SeekCodeResult<ToolOutput> {
        let started_at = Instant::now();
        tracing::debug!(
            target: "seekcode_tool_system::list_files",
            task_id = %ctx.task_id,
            workspace_id = ?ctx.workspace_id,
            workspace_root = %self.config.workspace_root.display(),
            raw_input = %input,
            "list_files tool started"
        );
        let input: ListFilesInput = parse_input(input)?;
        let base = input.path.unwrap_or_else(|| PathBuf::from("."));
        let base = resolve_existing_workspace_path(&self.config.workspace_root, &base)?;
        let max_depth = input.max_depth.unwrap_or(4);
        let limit = input.limit.unwrap_or(500);
        let mut entries = Vec::new();
        tracing::debug!(
            target: "seekcode_tool_system::list_files",
            task_id = %ctx.task_id,
            base = %base.display(),
            max_depth,
            limit,
            "list_files resolved input"
        );

        for entry in WalkDir::new(&base)
            .max_depth(max_depth)
            .into_iter()
            .filter_entry(|entry| !is_ignored_name(entry.path()))
        {
            let entry = entry.map_err(|error| SeekCodeError::Workspace(error.to_string()))?;
            if entry.path() == base {
                continue;
            }
            let metadata = entry
                .metadata()
                .map_err(|error| SeekCodeError::Workspace(error.to_string()))?;
            entries.push(json!({
                "path": relative_display(&self.config.workspace_root, entry.path()),
                "is_dir": metadata.is_dir(),
                "size": metadata.is_file().then_some(metadata.len()),
            }));
            if entries.len() >= limit {
                tracing::debug!(
                    target: "seekcode_tool_system::list_files",
                    task_id = %ctx.task_id,
                    entries = entries.len(),
                    elapsed_ms = started_at.elapsed().as_millis(),
                    "list_files reached limit"
                );
                break;
            }
        }

        tracing::debug!(
            target: "seekcode_tool_system::list_files",
            task_id = %ctx.task_id,
            entries = entries.len(),
            elapsed_ms = started_at.elapsed().as_millis(),
            "list_files tool finished"
        );

        Ok(ToolOutput {
            content: json!({ "entries": entries }),
            summary: format!("Listed {} entries", entries.len()),
        })
    }
}

#[async_trait]
impl Tool for SearchTextTool {
    fn name(&self) -> &'static str {
        SEARCH_TEXT_TOOL
    }

    fn description(&self) -> &'static str {
        "Search UTF-8 workspace files by substring."
    }

    fn input_schema(&self) -> RootSchema {
        schemars::schema_for!(SearchTextInput)
    }

    async fn execute(&self, _ctx: ToolContext, input: ToolInput) -> SeekCodeResult<ToolOutput> {
        let input: SearchTextInput = parse_input(input)?;
        if input.pattern.is_empty() {
            return Err(SeekCodeError::Validation(
                "search pattern cannot be empty".to_string(),
            ));
        }

        let base = input.path.unwrap_or_else(|| PathBuf::from("."));
        let base = resolve_existing_workspace_path(&self.config.workspace_root, &base)?;
        let limit = input.limit.unwrap_or(self.config.max_search_results);
        let mut matches = Vec::new();

        for file in collect_search_files(&self.config, &base)? {
            let content = match tokio::fs::read_to_string(&file).await {
                Ok(content) => content,
                Err(_) => continue,
            };
            for (index, line) in content.lines().enumerate() {
                if line.contains(&input.pattern) {
                    matches.push(json!({
                        "path": relative_display(&self.config.workspace_root, &file),
                        "line": index + 1,
                        "text": line,
                    }));
                    if matches.len() >= limit {
                        return Ok(ToolOutput {
                            content: json!({ "matches": matches }),
                            summary: format!("Found {} matches", limit),
                        });
                    }
                }
            }
        }

        Ok(ToolOutput {
            content: json!({ "matches": matches }),
            summary: format!("Found {} matches", matches.len()),
        })
    }
}

#[async_trait]
impl Tool for ApplyPatchTool {
    fn name(&self) -> &'static str {
        APPLY_PATCH_TOOL
    }

    fn description(&self) -> &'static str {
        "Apply a unified diff to the workspace using git apply."
    }

    fn input_schema(&self) -> RootSchema {
        schemars::schema_for!(ApplyPatchInput)
    }

    async fn execute(&self, _ctx: ToolContext, input: ToolInput) -> SeekCodeResult<ToolOutput> {
        let input: ApplyPatchInput = parse_input(input)?;
        if input.patch.trim().is_empty() {
            return Err(SeekCodeError::Validation(
                "patch cannot be empty".to_string(),
            ));
        }

        if input.check {
            let check = run_program_with_stdin(
                &self.config,
                "git",
                &["apply", "--check", "--whitespace=nowarn"],
                &self.config.workspace_root,
                &input.patch,
                self.config.command_timeout,
                BTreeMap::new(),
            )
            .await?;
            if check.exit_code != Some(0) {
                return Err(SeekCodeError::Patch(format!(
                    "git apply --check failed: {}{}",
                    check.stdout, check.stderr
                )));
            }
        }

        let result = run_program_with_stdin(
            &self.config,
            "git",
            &["apply", "--whitespace=nowarn"],
            &self.config.workspace_root,
            &input.patch,
            self.config.command_timeout,
            BTreeMap::new(),
        )
        .await?;
        if result.exit_code != Some(0) {
            return Err(SeekCodeError::Patch(format!(
                "git apply failed: {}{}",
                result.stdout, result.stderr
            )));
        }

        Ok(ToolOutput {
            content: serde_json::to_value(&result)
                .map_err(|error| SeekCodeError::Internal(error.to_string()))?,
            summary: "Applied patch".to_string(),
        })
    }
}

#[async_trait]
impl Tool for RunCommandTool {
    fn name(&self) -> &'static str {
        RUN_COMMAND_TOOL
    }

    fn description(&self) -> &'static str {
        "Run a non-interactive command inside the current workspace without a shell."
    }

    fn input_schema(&self) -> RootSchema {
        schemars::schema_for!(RunCommandInput)
    }

    async fn execute(&self, _ctx: ToolContext, input: ToolInput) -> SeekCodeResult<ToolOutput> {
        let input: RunCommandInput = parse_input(input)?;
        if input.program.trim().is_empty() {
            return Err(SeekCodeError::Validation(
                "command program cannot be empty".to_string(),
            ));
        }

        let cwd = input.cwd.unwrap_or_else(|| PathBuf::from("."));
        let cwd = resolve_existing_workspace_path(&self.config.workspace_root, &cwd)?;
        if !cwd.is_dir() {
            return Err(SeekCodeError::Shell(format!(
                "command cwd is not a directory: {}",
                relative_display(&self.config.workspace_root, &cwd)
            )));
        }

        let timeout = input
            .timeout_ms
            .map(Duration::from_millis)
            .unwrap_or(self.config.command_timeout);
        let arg_refs = input.args.iter().map(String::as_str).collect::<Vec<_>>();
        let result = run_program(
            &self.config,
            &input.program,
            &arg_refs,
            &cwd,
            timeout,
            input.env,
        )
        .await?;

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

#[async_trait]
impl Tool for GitStatusTool {
    fn name(&self) -> &'static str {
        GIT_STATUS_TOOL
    }

    fn description(&self) -> &'static str {
        "Show git status for the current workspace."
    }

    fn input_schema(&self) -> RootSchema {
        schemars::schema_for!(AnyToolInput)
    }

    async fn execute(&self, _ctx: ToolContext, _input: ToolInput) -> SeekCodeResult<ToolOutput> {
        let result = run_program(
            &self.config,
            "git",
            &["status", "--short", "--branch"],
            &self.config.workspace_root,
            self.config.command_timeout,
            BTreeMap::new(),
        )
        .await?;

        Ok(ToolOutput {
            summary: "Read git status".to_string(),
            content: serde_json::to_value(&result)
                .map_err(|error| SeekCodeError::Internal(error.to_string()))?,
        })
    }
}

#[async_trait]
impl Tool for GitDiffTool {
    fn name(&self) -> &'static str {
        GIT_DIFF_TOOL
    }

    fn description(&self) -> &'static str {
        "Show git diff for the current workspace."
    }

    fn input_schema(&self) -> RootSchema {
        schemars::schema_for!(GitDiffInput)
    }

    async fn execute(&self, _ctx: ToolContext, input: ToolInput) -> SeekCodeResult<ToolOutput> {
        let input: GitDiffInput = parse_input(input)?;
        let mut args = vec!["diff"];
        if input.staged {
            args.push("--staged");
        }

        let path_filter = if let Some(path) = input.path {
            Some(resolve_existing_workspace_path(
                &self.config.workspace_root,
                &path,
            )?)
        } else {
            None
        };
        if path_filter.is_some() {
            args.push("--");
        }
        let path_string = path_filter
            .as_ref()
            .map(|path| relative_display(&self.config.workspace_root, path));
        if let Some(path) = path_string.as_deref() {
            args.push(path);
        }

        let result = run_program(
            &self.config,
            "git",
            &args,
            &self.config.workspace_root,
            self.config.command_timeout,
            BTreeMap::new(),
        )
        .await?;

        Ok(ToolOutput {
            summary: "Read git diff".to_string(),
            content: serde_json::to_value(&result)
                .map_err(|error| SeekCodeError::Internal(error.to_string()))?,
        })
    }
}

fn default_true() -> bool {
    true
}

fn parse_input<T>(input: ToolInput) -> SeekCodeResult<T>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_value(input).map_err(|error| SeekCodeError::Validation(error.to_string()))
}

fn reject_unsafe_relative_path(path: &Path) -> SeekCodeResult<()> {
    if path.as_os_str().is_empty() {
        return Ok(());
    }

    for component in path.components() {
        match component {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(SeekCodeError::PolicyDenied(format!(
                    "path must stay inside workspace: {}",
                    path.display()
                )));
            }
        }
    }

    Ok(())
}

fn resolve_existing_workspace_path(root: &Path, relative: &Path) -> SeekCodeResult<PathBuf> {
    reject_unsafe_relative_path(relative)?;
    let path = root.join(relative);
    let canonical = path
        .canonicalize()
        .map_err(|error| SeekCodeError::Workspace(error.to_string()))?;
    if !canonical.starts_with(root) {
        return Err(SeekCodeError::PolicyDenied(format!(
            "path escapes workspace: {}",
            relative.display()
        )));
    }

    Ok(canonical)
}

fn resolve_new_workspace_path(root: &Path, relative: &Path) -> SeekCodeResult<PathBuf> {
    reject_unsafe_relative_path(relative)?;
    let path = root.join(relative);
    if path.exists() {
        return resolve_existing_workspace_path(root, relative);
    }

    let parent = path
        .parent()
        .ok_or_else(|| SeekCodeError::Workspace("file path has no parent".to_string()))?;
    let parent_to_check = nearest_existing_parent(parent)?;
    let canonical_parent = parent_to_check
        .canonicalize()
        .map_err(|error| SeekCodeError::Workspace(error.to_string()))?;
    if !canonical_parent.starts_with(root) {
        return Err(SeekCodeError::PolicyDenied(format!(
            "path escapes workspace: {}",
            relative.display()
        )));
    }

    Ok(path)
}

fn nearest_existing_parent(path: &Path) -> SeekCodeResult<PathBuf> {
    let mut current = path;
    loop {
        if current.exists() {
            return Ok(current.to_path_buf());
        }
        current = current
            .parent()
            .ok_or_else(|| SeekCodeError::Workspace("no existing parent directory".to_string()))?;
    }
}

fn relative_display(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn is_ignored_name(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| matches!(name, ".git" | "target" | "node_modules"))
        .unwrap_or(false)
}

fn collect_search_files(config: &SystemToolConfig, base: &Path) -> SeekCodeResult<Vec<PathBuf>> {
    if base.is_file() {
        return Ok(vec![base.to_path_buf()]);
    }

    let mut files = Vec::new();
    for entry in WalkDir::new(base)
        .into_iter()
        .filter_entry(|entry| !is_ignored_name(entry.path()))
    {
        let entry = entry.map_err(|error| SeekCodeError::Workspace(error.to_string()))?;
        if entry.path().is_file()
            && entry
                .metadata()
                .map(|metadata| metadata.len() <= config.max_file_bytes)
                .unwrap_or(false)
        {
            files.push(entry.path().to_path_buf());
        }
    }

    Ok(files)
}

async fn run_program(
    config: &SystemToolConfig,
    program: &str,
    args: &[&str],
    cwd: &Path,
    timeout: Duration,
    env: BTreeMap<String, String>,
) -> SeekCodeResult<CommandResult> {
    run_program_inner(config, program, args, cwd, None, timeout, env).await
}

async fn run_program_with_stdin(
    config: &SystemToolConfig,
    program: &str,
    args: &[&str],
    cwd: &Path,
    stdin: &str,
    timeout: Duration,
    env: BTreeMap<String, String>,
) -> SeekCodeResult<CommandResult> {
    run_program_inner(config, program, args, cwd, Some(stdin), timeout, env).await
}

async fn run_program_inner(
    config: &SystemToolConfig,
    program: &str,
    args: &[&str],
    cwd: &Path,
    stdin: Option<&str>,
    timeout: Duration,
    env: BTreeMap<String, String>,
) -> SeekCodeResult<CommandResult> {
    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if stdin.is_some() {
        command.stdin(Stdio::piped());
    }
    for (key, value) in env {
        command.env(key, value);
    }

    let mut child = command
        .spawn()
        .map_err(|error| SeekCodeError::Shell(error.to_string()))?;
    if let Some(stdin) = stdin {
        let mut child_stdin = child
            .stdin
            .take()
            .ok_or_else(|| SeekCodeError::Shell("failed to open child stdin".to_string()))?;
        child_stdin
            .write_all(stdin.as_bytes())
            .await
            .map_err(|error| SeekCodeError::Shell(error.to_string()))?;
        drop(child_stdin);
    }

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

    #[tokio::test]
    async fn system_read_file_reads_workspace_file() {
        let root = make_temp_workspace();
        tokio::fs::write(root.join("hello.txt"), "hello world")
            .await
            .expect("write fixture");
        let config = SystemToolConfig::new(&root).expect("config builds");
        let tool = ReadFileTool::new(config);

        let output = tool
            .execute(
                ToolContext {
                    task_id: TaskId::new(),
                    workspace_id: None,
                    workspace_root: Some(root.clone()),
                },
                json!({ "path": "hello.txt" }),
            )
            .await
            .expect("read file");

        assert_eq!(output.content["content"], "hello world");
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn system_read_file_rejects_parent_escape() {
        let root = make_temp_workspace();
        let config = SystemToolConfig::new(&root).expect("config builds");
        let tool = ReadFileTool::new(config);

        let error = tool
            .execute(
                ToolContext {
                    task_id: TaskId::new(),
                    workspace_id: None,
                    workspace_root: Some(root.clone()),
                },
                json!({ "path": "../outside.txt" }),
            )
            .await
            .expect_err("escape is rejected");

        assert!(matches!(error, SeekCodeError::PolicyDenied(_)));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn system_tools_register_all_builtin_tools() {
        let root = make_temp_workspace();
        let config = SystemToolConfig::new(&root).expect("config builds");
        let registry = system_tool_registry(config).expect("registry builds");

        for name in [
            READ_FILE_TOOL,
            WRITE_FILE_TOOL,
            LIST_FILES_TOOL,
            SEARCH_TEXT_TOOL,
            APPLY_PATCH_TOOL,
            RUN_COMMAND_TOOL,
            GIT_STATUS_TOOL,
            GIT_DIFF_TOOL,
        ] {
            assert!(registry.get(name).is_some(), "{name} should be registered");
        }

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
