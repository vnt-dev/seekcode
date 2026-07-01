use chrono::Local;
use seekcode_common::{ChatMessage, ChatRole, MessageId, SessionId};
use seekcode_storage::SessionMessageRecord;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Returns the shell the run_command tool executes through on this platform.
/// Windows commands run through PowerShell; other systems run through sh.
fn context_shell() -> &'static str {
    if cfg!(windows) {
        "powershell"
    } else {
        "sh"
    }
}
pub(crate) const SKILLS_SYSTEM_PREFIX: &str = "Skills\nA skill is a set of instructions provided through a `SKILL.md` source. Below is the list of skills that can be used. Each entry includes a name, description, and source locator. `file` locators are on the host filesystem, `environment resource` locators are owned by an execution environment, `orchestrator resource` locators are opaque non-filesystem resources, and `custom resource` locators use their provider's access mechanism.\n### Available skills";

pub(crate) fn build_skills_system_message() -> Option<String> {
    seekcode_skills_dir().and_then(|skills_dir| build_skills_system_message_for_dir(&skills_dir))
}

pub(crate) fn build_agents_instructions_message(workspace_path: &str) -> Option<String> {
    let agents_path = Path::new(workspace_path).join("AGENTS.md");
    if !agents_path.is_file() {
        return None;
    }

    let content = std::fs::read_to_string(agents_path).ok()?;
    Some(format!(
        "# AGENTS.md instructions for {workspace_path}\n\n<INSTRUCTIONS>\n{content}\n</INSTRUCTIONS>"
    ))
}

fn seekcode_skills_dir() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .map(|home| home.join(".seekcode").join("skills"))
}

pub(crate) fn build_skills_system_message_for_dir(skills_dir: &Path) -> Option<String> {
    let mut skill_paths = Vec::new();
    collect_skill_paths(skills_dir, &mut skill_paths);
    skill_paths.sort();

    if skill_paths.is_empty() {
        return None;
    }

    let mut message = SKILLS_SYSTEM_PREFIX.to_string();
    for path in skill_paths {
        message.push('\n');
        message.push_str(&skill_entry_from_path(&path));
    }

    Some(message)
}

fn collect_skill_paths(path: &Path, skill_paths: &mut Vec<PathBuf>) {
    let Ok(metadata) = std::fs::metadata(path) else {
        return;
    };

    if metadata.is_file() {
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.eq_ignore_ascii_case("skill.md"))
        {
            skill_paths.push(path.to_path_buf());
        }
        return;
    }

    if !metadata.is_dir() {
        return;
    }

    let Ok(entries) = std::fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        collect_skill_paths(&entry.path(), skill_paths);
    }
}

fn skill_entry_from_path(path: &Path) -> String {
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let name = extract_frontmatter_value(&content, "name")
        .or_else(|| {
            path.parent()
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| "unknown".to_string());
    let description = extract_frontmatter_value(&content, "description")
        .unwrap_or_else(|| "No description provided.".to_string());
    let source = path.to_string_lossy();

    format!("- {name}: {description} (file: {source})")
}

fn extract_frontmatter_value(content: &str, key: &str) -> Option<String> {
    let mut lines = content.lines();
    if lines.next()?.trim() != "---" {
        return None;
    }

    let prefix = format!("{key}:");
    for line in lines {
        let line = line.trim();
        if line == "---" {
            break;
        }
        if let Some(value) = line.strip_prefix(&prefix) {
            let value = value.trim();
            if !value.is_empty() {
                return Some(unquote_frontmatter_value(value));
            }
        }
    }

    None
}

fn unquote_frontmatter_value(value: &str) -> String {
    let value = value.trim();
    if value.len() >= 2 {
        let first = value.as_bytes()[0] as char;
        let last = value.as_bytes()[value.len() - 1] as char;
        if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
            return value[1..value.len() - 1].to_string();
        }
    }

    value.to_string()
}

pub(crate) fn build_environment_context(cwd: &str) -> String {
    let cwd = escape_environment_context_text(cwd);
    let os = escape_environment_context_text(std::env::consts::OS);
    let shell = escape_environment_context_text(context_shell());
    let now = Local::now();
    let current_date = now.format("%Y-%m-%d").to_string();
    let timezone = escape_environment_context_text(&current_timezone_name(&now));

    format!(
        "<environment_context>\n  <cwd>{cwd}</cwd>\n  <os>{os}</os>\n  <shell>{shell}</shell>\n  <current_date>{current_date}</current_date>\n  <timezone>{timezone}</timezone>\n  <filesystem><workspace_roots><root>{cwd}</root></workspace_roots></filesystem>\n</environment_context>"
    )
}

pub(crate) fn current_timezone_name(now: &chrono::DateTime<Local>) -> String {
    iana_time_zone::get_timezone().unwrap_or_else(|_| now.format("%:z").to_string())
}

fn escape_environment_context_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

pub(crate) fn push_record_as_context_message(
    record: SessionMessageRecord,
    messages: &mut Vec<ChatMessage>,
) {
    let mut message = ChatMessage::new(record.role, record.content);
    message.id = record.id;
    message.reasoning_content = record.reasoning_content;
    message.tool_calls = record.tool_calls;
    message.tool_call_id = record.tool_call_id;
    messages.push(message);
}

pub(crate) fn push_turn_records_as_context_messages(
    session_id: SessionId,
    records: Vec<SessionMessageRecord>,
    messages: &mut Vec<ChatMessage>,
) {
    let include_reasoning = records.iter().any(record_has_tool_activity);
    let mut pending_assistant = PendingAssistantContext::default();
    let mut turn_messages = Vec::new();

    for record in records {
        tracing::debug!(
            target: "seekcode_app_kernel::context",
            %session_id,
            message_id = %record.id,
            turn_sequence = record.turn_sequence,
            role = ?record.role,
            content_len = record.content.len(),
            reasoning_len = record.reasoning_content.as_deref().map(str::len).unwrap_or(0),
            tool_call_count = record.tool_calls.len(),
            tool_call_id = ?record.tool_call_id,
            tool_calls = %serde_json::Value::Array(record.tool_calls.clone()),
            include_reasoning,
            "adding persisted message to agent context"
        );

        if record.role == ChatRole::Assistant {
            pending_assistant.apply_record(session_id, record, include_reasoning);
            continue;
        }

        pending_assistant.flush(session_id, &mut turn_messages);
        push_record_as_context_message(record, &mut turn_messages);
    }

    pending_assistant.flush(session_id, &mut turn_messages);
    append_valid_tool_context_messages(session_id, turn_messages, messages);
}

fn record_has_tool_activity(record: &SessionMessageRecord) -> bool {
    !record.tool_calls.is_empty()
        || (record.role == ChatRole::Tool && record.tool_call_id.is_some())
}

#[derive(Default)]
pub(crate) struct PendingAssistantContext {
    content: String,
    reasoning_content: String,
    tool_calls: PersistedToolCallAccumulator,
}

impl PendingAssistantContext {
    pub(crate) fn apply_record(
        &mut self,
        session_id: SessionId,
        record: SessionMessageRecord,
        include_reasoning: bool,
    ) {
        self.content.push_str(&record.content);
        if include_reasoning {
            if let Some(reasoning_content) = record.reasoning_content {
                self.reasoning_content.push_str(&reasoning_content);
            }
        }
        if !record.tool_calls.is_empty() {
            self.tool_calls
                .apply(session_id, record.id, record.tool_calls);
        }
    }

    pub(crate) fn flush(&mut self, session_id: SessionId, messages: &mut Vec<ChatMessage>) {
        let tool_calls = self.tool_calls.take_completed(session_id);
        if self.content.is_empty() && self.reasoning_content.is_empty() && tool_calls.is_empty() {
            return;
        }

        tracing::debug!(
            target: "seekcode_app_kernel::context",
            %session_id,
            content_len = self.content.len(),
            reasoning_len = self.reasoning_content.len(),
            tool_call_count = tool_calls.len(),
            tool_calls = %serde_json::Value::Array(tool_calls.clone()),
            "flushing persisted assistant deltas into context message"
        );

        let mut message = ChatMessage::new(ChatRole::Assistant, std::mem::take(&mut self.content));
        if !self.reasoning_content.is_empty() {
            message.reasoning_content = Some(std::mem::take(&mut self.reasoning_content));
        }
        message.tool_calls = tool_calls;
        messages.push(message);
    }
}

fn append_valid_tool_context_messages(
    session_id: SessionId,
    turn_messages: Vec<ChatMessage>,
    messages: &mut Vec<ChatMessage>,
) {
    let mut allowed_tool_call_ids = BTreeSet::new();

    for (index, message) in turn_messages.iter().enumerate() {
        match message.role {
            ChatRole::Assistant if !message.tool_calls.is_empty() => {
                let expected_tool_call_ids = tool_call_ids_from_assistant(message);
                let following_tool_call_ids = following_tool_call_ids(&turn_messages, index + 1);
                if expected_tool_call_ids.is_empty()
                    || !expected_tool_call_ids
                        .iter()
                        .all(|id| following_tool_call_ids.contains(id))
                {
                    tracing::warn!(
                        target: "seekcode_app_kernel::context",
                        %session_id,
                        expected_tool_call_ids = ?expected_tool_call_ids,
                        following_tool_call_ids = ?following_tool_call_ids,
                        "dropping assistant message with unanswered tool calls"
                    );
                    allowed_tool_call_ids.clear();
                    continue;
                }

                allowed_tool_call_ids = expected_tool_call_ids;
                messages.push(message.clone());
            }
            ChatRole::Tool => {
                let Some(tool_call_id) = message.tool_call_id.map(|id| id.to_string()) else {
                    tracing::warn!(
                        target: "seekcode_app_kernel::context",
                        %session_id,
                        message_id = %message.id,
                        "dropping tool message without tool_call_id"
                    );
                    continue;
                };

                if allowed_tool_call_ids.remove(&tool_call_id) {
                    messages.push(message.clone());
                } else {
                    tracing::warn!(
                        target: "seekcode_app_kernel::context",
                        %session_id,
                        message_id = %message.id,
                        %tool_call_id,
                        "dropping orphan tool message without matching assistant tool call"
                    );
                }
            }
            _ => {
                allowed_tool_call_ids.clear();
                messages.push(message.clone());
            }
        }
    }
}

fn tool_call_ids_from_assistant(message: &ChatMessage) -> BTreeSet<String> {
    message
        .tool_calls
        .iter()
        .filter_map(tool_call_id_from_value)
        .collect()
}

fn following_tool_call_ids(messages: &[ChatMessage], start: usize) -> BTreeSet<String> {
    messages
        .iter()
        .skip(start)
        .take_while(|message| message.role == ChatRole::Tool)
        .filter_map(|message| message.tool_call_id.map(|id| id.to_string()))
        .collect()
}

fn tool_call_id_from_value(value: &serde_json::Value) -> Option<String> {
    value
        .get("id")
        .and_then(serde_json::Value::as_str)
        .filter(|id| !id.is_empty())
        .map(ToOwned::to_owned)
}

#[derive(Default)]
struct PersistedToolCallAccumulator {
    partials: BTreeMap<String, PersistedToolCallPartial>,
}

impl PersistedToolCallAccumulator {
    fn apply(
        &mut self,
        session_id: SessionId,
        message_id: MessageId,
        tool_calls: Vec<serde_json::Value>,
    ) {
        for tool_call in tool_calls {
            let Some(id) = tool_call
                .get("id")
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
            else {
                tracing::warn!(
                    target: "seekcode_app_kernel::context",
                    %session_id,
                    %message_id,
                    tool_call = %tool_call,
                    "skipping persisted tool call delta without id"
                );
                continue;
            };

            let partial =
                self.partials
                    .entry(id.clone())
                    .or_insert_with(|| PersistedToolCallPartial {
                        id,
                        kind: "function".to_string(),
                        name: None,
                        arguments: String::new(),
                    });

            if let Some(kind) = tool_call
                .get("type")
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.is_empty())
            {
                partial.kind = kind.to_string();
            }

            let function = tool_call
                .get("function")
                .and_then(serde_json::Value::as_object);
            if let Some(name) = function
                .and_then(|value| value.get("name"))
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.is_empty())
            {
                partial.name = Some(name.to_string());
            }
            if let Some(arguments) = function
                .and_then(|value| value.get("arguments"))
                .and_then(serde_json::Value::as_str)
            {
                partial.arguments.push_str(arguments);
            }
        }
    }

    fn take_completed(&mut self, session_id: SessionId) -> Vec<serde_json::Value> {
        if self.partials.is_empty() {
            return Vec::new();
        }

        let mut tool_calls = Vec::new();
        for partial in std::mem::take(&mut self.partials).into_values() {
            let Some(name) = partial.name else {
                tracing::warn!(
                    target: "seekcode_app_kernel::context",
                    %session_id,
                    tool_call_id = %partial.id,
                    arguments_len = partial.arguments.len(),
                    "dropping incomplete persisted tool call without function name"
                );
                continue;
            };

            tool_calls.push(serde_json::json!({
                "id": partial.id,
                "type": partial.kind,
                "function": {
                    "name": name,
                    "arguments": partial.arguments
                }
            }));
        }

        tool_calls
    }
}

struct PersistedToolCallPartial {
    id: String,
    kind: String,
    name: Option<String>,
    arguments: String,
}
