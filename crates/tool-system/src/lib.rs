//! Tool registry, schemas, and executor traits.

use async_trait::async_trait;
use schemars::schema::RootSchema;
use seekcode_common::{SeekCodeError, SeekCodeResult, TaskId, ToolCallId, WorkspaceId};
use seekcode_deepseek_client::ToolSpec;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use walkdir::WalkDir;

/// Name for the read file tool.
pub const READ_FILE_TOOL: &str = "read_file";
/// Name for the write file tool.
pub const WRITE_FILE_TOOL: &str = "write_file";
/// Name for the insert lines tool.
pub const INSERT_LINES_TOOL: &str = "insert_lines";
/// Name for the text search tool.
pub const SEARCH_TEXT_TOOL: &str = "search_text";
/// Name for the run command tool.
pub const RUN_COMMAND_TOOL: &str = "run_command";

const MAX_READ_FILE_LINES: usize = 5_000;

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
    registry.register(ReadFileTool::new(config.clone()))?;
    registry.register(WriteFileTool::new(config.clone()))?;
    registry.register(InsertLinesTool::new(config.clone()))?;
    registry.register(SearchTextTool::new(config.clone()))?;
    registry.register(RunCommandTool::new(config))?;
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
pub struct WriteFileTool;

impl WriteFileTool {
    /// Creates a write file tool.
    pub fn new(_config: SystemToolConfig) -> Self {
        Self
    }
}

/// Tool that inserts UTF-8 content after a line in a workspace file.
pub struct InsertLinesTool {
    config: SystemToolConfig,
}

impl InsertLinesTool {
    /// Creates an insert lines tool.
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
struct ReadFileInput {
    /// Absolute path, or a path relative to the workspace root.
    path: PathBuf,
    /// 1-based line number to start reading from.
    #[serde(default = "default_start_line")]
    start_line: usize,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct WriteFileInput {
    /// Absolute path, or a path relative to the workspace root.
    path: PathBuf,
    /// UTF-8 content to write.
    content: String,
    /// Whether missing parent directories may be created.
    #[serde(default)]
    create_dirs: bool,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct InsertLinesInput {
    /// Absolute path, or a path relative to the workspace root.
    path: PathBuf,
    /// 1-based line number after which content is inserted. Use 0 to insert at the beginning.
    line: usize,
    /// UTF-8 content to insert.
    content: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SearchTextInput {
    /// Substring pattern to search for.
    pattern: String,
    /// Absolute directory or file path, or a path relative to the workspace root.
    #[serde(default)]
    path: Option<PathBuf>,
    /// Maximum matches returned.
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RunCommandInput {
    /// Command line to execute.
    command: String,
    /// Absolute working directory, or a path relative to the workspace root.
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
impl Tool for ReadFileTool {
    fn name(&self) -> &'static str {
        READ_FILE_TOOL
    }

    fn description(&self) -> &'static str {
        "Read up to 5000 lines from a UTF-8 text file by absolute path or a path relative to the workspace root. Reading starts at start_line, defaulting to 1."
    }

    fn input_schema(&self) -> RootSchema {
        schemars::schema_for!(ReadFileInput)
    }

    async fn execute(&self, ctx: ToolContext, input: ToolInput) -> SeekCodeResult<ToolOutput> {
        let workspace_root = workspace_root_from_context(&ctx)?;
        let input: ReadFileInput = parse_input(input)?;
        if input.start_line == 0 {
            return Err(SeekCodeError::Validation(
                "read start_line must be at least 1".to_string(),
            ));
        }

        let path = resolve_existing_workspace_path(&workspace_root, &input.path)?;
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
        let (content, line_count, total_lines) =
            read_line_window(&content, input.start_line, MAX_READ_FILE_LINES);
        let end_line = (line_count > 0).then_some(input.start_line + line_count - 1);
        let truncated = end_line
            .map(|end_line| end_line < total_lines)
            .unwrap_or(false);
        Ok(ToolOutput {
            content: json!({
                "path": relative_display(&workspace_root, &path),
                "content": content,
                "bytes": metadata.len(),
                "start_line": input.start_line,
                "end_line": end_line,
                "line_count": line_count,
                "total_lines": total_lines,
                "truncated": truncated,
            }),
            summary: format!(
                "Read {} lines from {}",
                line_count,
                relative_display(&workspace_root, &path)
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
        "Write UTF-8 content to a file by absolute path or a path relative to the workspace root."
    }

    fn input_schema(&self) -> RootSchema {
        schemars::schema_for!(WriteFileInput)
    }

    async fn execute(&self, ctx: ToolContext, input: ToolInput) -> SeekCodeResult<ToolOutput> {
        let workspace_root = workspace_root_from_context(&ctx)?;
        let input: WriteFileInput = parse_input(input)?;
        let path = resolve_new_workspace_path(&workspace_root, &input.path)?;
        if let Some(parent) = path.parent() {
            if input.create_dirs {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|error| SeekCodeError::Workspace(error.to_string()))?;
            } else if !parent.exists() {
                return Err(SeekCodeError::Workspace(format!(
                    "parent directory does not exist: {}",
                    relative_display(&workspace_root, parent)
                )));
            }
        }

        tokio::fs::write(&path, input.content.as_bytes())
            .await
            .map_err(|error| SeekCodeError::Workspace(error.to_string()))?;
        Ok(ToolOutput {
            content: json!({
                "path": relative_display(&workspace_root, &path),
                "bytes": input.content.len(),
            }),
            summary: format!(
                "Wrote {} bytes to {}",
                input.content.len(),
                relative_display(&workspace_root, &path)
            ),
        })
    }
}

#[async_trait]
impl Tool for InsertLinesTool {
    fn name(&self) -> &'static str {
        INSERT_LINES_TOOL
    }

    fn description(&self) -> &'static str {
        "Insert UTF-8 content into a file after a 1-based line number. Use line 0 to insert at the beginning."
    }

    fn input_schema(&self) -> RootSchema {
        schemars::schema_for!(InsertLinesInput)
    }

    async fn execute(&self, ctx: ToolContext, input: ToolInput) -> SeekCodeResult<ToolOutput> {
        let workspace_root = workspace_root_from_context(&ctx)?;
        let input: InsertLinesInput = parse_input(input)?;
        let path = resolve_existing_workspace_path(&workspace_root, &input.path)?;
        let metadata = tokio::fs::metadata(&path)
            .await
            .map_err(|error| SeekCodeError::Workspace(error.to_string()))?;
        if metadata.len() > self.config.max_file_bytes {
            return Err(SeekCodeError::Workspace(format!(
                "file is too large to edit: {} bytes",
                metadata.len()
            )));
        }

        let content = tokio::fs::read_to_string(&path)
            .await
            .map_err(|error| SeekCodeError::Workspace(error.to_string()))?;
        let total_lines = count_text_lines(&content);
        let insert_at = line_end_byte_index(&content, input.line).ok_or_else(|| {
            SeekCodeError::Validation(format!(
                "insert line {} is outside the file line range 0..={}",
                input.line, total_lines
            ))
        })?;
        let mut updated = String::with_capacity(content.len() + input.content.len());
        updated.push_str(&content[..insert_at]);
        updated.push_str(&input.content);
        updated.push_str(&content[insert_at..]);

        tokio::fs::write(&path, updated.as_bytes())
            .await
            .map_err(|error| SeekCodeError::Workspace(error.to_string()))?;
        Ok(ToolOutput {
            content: json!({
                "path": relative_display(&workspace_root, &path),
                "line": input.line,
                "inserted_bytes": input.content.len(),
                "bytes": updated.len(),
            }),
            summary: format!(
                "Inserted {} bytes into {} after line {}",
                input.content.len(),
                relative_display(&workspace_root, &path),
                input.line
            ),
        })
    }
}

#[async_trait]
impl Tool for SearchTextTool {
    fn name(&self) -> &'static str {
        SEARCH_TEXT_TOOL
    }

    fn description(&self) -> &'static str {
        "Search UTF-8 files by substring at an absolute path or a path relative to the workspace root."
    }

    fn input_schema(&self) -> RootSchema {
        schemars::schema_for!(SearchTextInput)
    }

    async fn execute(&self, ctx: ToolContext, input: ToolInput) -> SeekCodeResult<ToolOutput> {
        let workspace_root = workspace_root_from_context(&ctx)?;
        let input: SearchTextInput = parse_input(input)?;
        if input.pattern.is_empty() {
            return Err(SeekCodeError::Validation(
                "search pattern cannot be empty".to_string(),
            ));
        }

        let base = input.path.unwrap_or_else(|| PathBuf::from("."));
        let base = resolve_existing_workspace_path(&workspace_root, &base)?;
        let limit = input.limit.unwrap_or(self.config.max_search_results);
        let mut matches = Vec::new();

        // A single file target is searched directly; otherwise the tree is walked
        // lazily so files are searched one at a time instead of being collected up
        // front. Each file is streamed line by line and we stop as soon as the
        // match limit is reached.
        if base.is_file() {
            search_file_lines(&base, &input.pattern, &workspace_root, limit, &mut matches).await?;
        } else {
            for entry in WalkDir::new(&base)
                .into_iter()
                .filter_entry(|entry| !is_ignored_name(entry.path()))
            {
                let entry = entry.map_err(|error| SeekCodeError::Workspace(error.to_string()))?;
                if !entry.file_type().is_file() {
                    continue;
                }
                let too_large = entry
                    .metadata()
                    .map(|metadata| metadata.len() > self.config.max_file_bytes)
                    .unwrap_or(true);
                if too_large {
                    continue;
                }

                let limit_reached = search_file_lines(
                    entry.path(),
                    &input.pattern,
                    &workspace_root,
                    limit,
                    &mut matches,
                )
                .await?;
                if limit_reached {
                    break;
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

fn default_start_line() -> usize {
    1
}

fn read_line_window(content: &str, start_line: usize, max_lines: usize) -> (String, usize, usize) {
    let mut selected = String::new();
    let mut selected_lines = 0usize;
    let mut total_lines = 0usize;

    for (index, line) in content.split_inclusive('\n').enumerate() {
        total_lines += 1;
        let line_number = index + 1;
        if line_number >= start_line && selected_lines < max_lines {
            selected.push_str(line);
            selected_lines += 1;
        }
    }

    (selected, selected_lines, total_lines)
}

fn count_text_lines(content: &str) -> usize {
    content.split_inclusive('\n').count()
}

fn line_end_byte_index(content: &str, line: usize) -> Option<usize> {
    if line == 0 {
        return Some(0);
    }

    let mut end = 0usize;
    for (index, text_line) in content.split_inclusive('\n').enumerate() {
        end += text_line.len();
        if index + 1 == line {
            return Some(end);
        }
    }

    None
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

/// Resolves a target path for a new file, returning the existing path when it
/// already exists.
fn resolve_new_workspace_path(root: &Path, input: &Path) -> SeekCodeResult<PathBuf> {
    let path = join_workspace_path(root, input);
    if path.exists() {
        return resolve_existing_workspace_path(root, input);
    }

    Ok(path)
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

/// Searches one file line by line, appending matches until the limit is hit.
/// Returns `true` once the match limit has been reached. Files that cannot be
/// opened or are not valid UTF-8 are skipped.
async fn search_file_lines(
    path: &Path,
    pattern: &str,
    workspace_root: &Path,
    limit: usize,
    matches: &mut Vec<Value>,
) -> SeekCodeResult<bool> {
    let file = match tokio::fs::File::open(path).await {
        Ok(file) => file,
        Err(_) => return Ok(false),
    };

    let mut lines = BufReader::new(file).lines();
    let mut line_number = 0usize;
    loop {
        // Read one line at a time so large files never load fully into memory.
        let line = match lines.next_line().await {
            Ok(Some(line)) => line,
            Ok(None) => break,
            Err(_) => return Ok(false),
        };
        line_number += 1;

        if line.contains(pattern) {
            matches.push(json!({
                "path": relative_display(workspace_root, path),
                "line": line_number,
                "text": line,
            }));
            if matches.len() >= limit {
                return Ok(true);
            }
        }
    }

    Ok(false)
}

/// Builds the base shell command for the current platform.
fn build_command(command_line: &str) -> Command {
    #[cfg(windows)]
    {
        let mut command = Command::new("powershell");
        command.args(["-NoProfile", "-NonInteractive", "-Command", command_line]);
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
    "Run a non-interactive command line through PowerShell. The working directory may be absolute or relative to the workspace root."
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

    #[tokio::test]
    async fn system_read_file_reads_workspace_file() {
        let root = make_temp_workspace();
        tokio::fs::write(root.join("hello.txt"), "hello world")
            .await
            .expect("write fixture");
        let config = SystemToolConfig::new();
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
    async fn system_read_file_uses_workspace_from_context() {
        let first_root = make_temp_workspace();
        let second_root = make_temp_workspace();
        tokio::fs::write(first_root.join("hello.txt"), "first workspace")
            .await
            .expect("write first fixture");
        tokio::fs::write(second_root.join("hello.txt"), "second workspace")
            .await
            .expect("write second fixture");
        let tool = ReadFileTool::new(SystemToolConfig::new());

        let first_output = tool
            .execute(
                ToolContext {
                    task_id: TaskId::new(),
                    workspace_id: None,
                    workspace_root: Some(first_root.clone()),
                },
                json!({ "path": "hello.txt" }),
            )
            .await
            .expect("read first workspace");
        let second_output = tool
            .execute(
                ToolContext {
                    task_id: TaskId::new(),
                    workspace_id: None,
                    workspace_root: Some(second_root.clone()),
                },
                json!({ "path": "hello.txt" }),
            )
            .await
            .expect("read second workspace");

        assert_eq!(first_output.content["content"], "first workspace");
        assert_eq!(second_output.content["content"], "second workspace");
        let _ = std::fs::remove_dir_all(first_root);
        let _ = std::fs::remove_dir_all(second_root);
    }

    #[tokio::test]
    async fn system_read_file_starts_at_requested_line_and_limits_output() {
        let root = make_temp_workspace();
        let content = (1..=5_010)
            .map(|line| format!("line {line}\n"))
            .collect::<String>();
        tokio::fs::write(root.join("many.txt"), content)
            .await
            .expect("write fixture");
        let config = SystemToolConfig::new();
        let tool = ReadFileTool::new(config);

        let output = tool
            .execute(
                ToolContext {
                    task_id: TaskId::new(),
                    workspace_id: None,
                    workspace_root: Some(root.clone()),
                },
                json!({ "path": "many.txt", "start_line": 3 }),
            )
            .await
            .expect("read file");

        let content = output.content["content"]
            .as_str()
            .expect("content is a string");
        assert!(content.starts_with("line 3\n"));
        assert!(content.ends_with("line 5002\n"));
        assert_eq!(content.lines().count(), MAX_READ_FILE_LINES);
        assert_eq!(output.content["start_line"], 3);
        assert_eq!(output.content["end_line"], 5002);
        assert_eq!(output.content["line_count"], MAX_READ_FILE_LINES);
        assert_eq!(output.content["total_lines"], 5_010);
        assert_eq!(output.content["truncated"], true);
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn system_read_file_reads_outside_workspace_via_absolute_path() {
        // Tools are not restricted to the workspace root: an absolute path that
        // points outside the workspace must resolve and read successfully.
        let root = make_temp_workspace();
        let outside = make_temp_workspace();
        let outside_file = outside.join("outside.txt");
        tokio::fs::write(&outside_file, "outside content")
            .await
            .expect("write fixture");
        let config = SystemToolConfig::new();
        let tool = ReadFileTool::new(config);

        let output = tool
            .execute(
                ToolContext {
                    task_id: TaskId::new(),
                    workspace_id: None,
                    workspace_root: Some(root.clone()),
                },
                json!({ "path": outside_file.to_string_lossy() }),
            )
            .await
            .expect("read outside file");

        assert_eq!(output.content["content"], "outside content");
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[tokio::test]
    async fn system_read_file_reads_parent_relative_path() {
        // A relative path that walks above the workspace root is now allowed.
        let root = make_temp_workspace();
        let parent = root.parent().expect("temp dir has a parent");
        let sibling = parent.join(format!("seekcode-sibling-{}.txt", std::process::id()));
        tokio::fs::write(&sibling, "sibling content")
            .await
            .expect("write fixture");
        let relative =
            PathBuf::from("..").join(sibling.file_name().expect("sibling has a file name"));
        let config = SystemToolConfig::new();
        let tool = ReadFileTool::new(config);

        let output = tool
            .execute(
                ToolContext {
                    task_id: TaskId::new(),
                    workspace_id: None,
                    workspace_root: Some(root.clone()),
                },
                json!({ "path": relative.to_string_lossy() }),
            )
            .await
            .expect("read sibling file");

        assert_eq!(output.content["content"], "sibling content");
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_file(sibling);
    }

    #[tokio::test]
    async fn system_insert_lines_inserts_content_after_requested_line() {
        let root = make_temp_workspace();
        let path = root.join("hello.txt");
        tokio::fs::write(&path, "alpha\nomega\n")
            .await
            .expect("write fixture");
        let config = SystemToolConfig::new();
        let tool = InsertLinesTool::new(config);

        let output = tool
            .execute(
                ToolContext {
                    task_id: TaskId::new(),
                    workspace_id: None,
                    workspace_root: Some(root.clone()),
                },
                json!({ "path": "hello.txt", "line": 1, "content": "beta\n" }),
            )
            .await
            .expect("insert content");

        let updated = tokio::fs::read_to_string(&path)
            .await
            .expect("read updated file");
        assert_eq!(updated, "alpha\nbeta\nomega\n");
        assert_eq!(output.content["line"], 1);
        assert_eq!(output.content["inserted_bytes"], 5);
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn system_insert_lines_supports_inserting_at_beginning() {
        let root = make_temp_workspace();
        let path = root.join("hello.txt");
        tokio::fs::write(&path, "omega\n")
            .await
            .expect("write fixture");
        let config = SystemToolConfig::new();
        let tool = InsertLinesTool::new(config);

        tool.execute(
            ToolContext {
                task_id: TaskId::new(),
                workspace_id: None,
                workspace_root: Some(root.clone()),
            },
            json!({ "path": "hello.txt", "line": 0, "content": "alpha\n" }),
        )
        .await
        .expect("insert content");

        let updated = tokio::fs::read_to_string(&path)
            .await
            .expect("read updated file");
        assert_eq!(updated, "alpha\nomega\n");
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn system_search_text_streams_matches_and_skips_ignored_dirs() {
        let root = make_temp_workspace();
        tokio::fs::write(root.join("a.txt"), "alpha\nbeta needle here\ngamma")
            .await
            .expect("write a.txt");
        tokio::fs::create_dir_all(root.join("sub"))
            .await
            .expect("create sub dir");
        tokio::fs::write(root.join("sub").join("b.txt"), "needle at top\nno hit")
            .await
            .expect("write b.txt");
        // Files under ignored directories must not be searched.
        tokio::fs::create_dir_all(root.join("target"))
            .await
            .expect("create target dir");
        tokio::fs::write(root.join("target").join("c.txt"), "needle ignored")
            .await
            .expect("write c.txt");
        let config = SystemToolConfig::new();
        let tool = SearchTextTool::new(config);

        let output = tool
            .execute(
                ToolContext {
                    task_id: TaskId::new(),
                    workspace_id: None,
                    workspace_root: Some(root.clone()),
                },
                json!({ "pattern": "needle" }),
            )
            .await
            .expect("search runs");

        let matches = output.content["matches"]
            .as_array()
            .expect("matches is an array");
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().all(|item| item["text"]
            .as_str()
            .expect("text is a string")
            .contains("needle")));
        assert!(matches
            .iter()
            .any(|item| item["path"] == "a.txt" && item["line"] == 2));
        assert!(matches
            .iter()
            .any(|item| item["path"] == "sub/b.txt" && item["line"] == 1));

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn system_search_text_respects_match_limit() {
        let root = make_temp_workspace();
        tokio::fs::write(root.join("a.txt"), "needle\nneedle\nneedle")
            .await
            .expect("write a.txt");
        let config = SystemToolConfig::new();
        let tool = SearchTextTool::new(config);

        let output = tool
            .execute(
                ToolContext {
                    task_id: TaskId::new(),
                    workspace_id: None,
                    workspace_root: Some(root.clone()),
                },
                json!({ "pattern": "needle", "limit": 2 }),
            )
            .await
            .expect("search runs");

        assert_eq!(
            output.content["matches"]
                .as_array()
                .expect("matches is an array")
                .len(),
            2
        );

        let _ = std::fs::remove_dir_all(root);
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
                json!({ "command": "cmd /c echo seekcode" }),
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
