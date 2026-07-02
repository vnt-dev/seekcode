//! Reconstructs tool-call `display` metadata for persisted session messages so
//! the UI can render historical tool calls the same way as live ones.

use seekcode_agent_core::tool_call_display;
use seekcode_storage::SessionMessageRecord;
use serde_json::Value;
use std::collections::HashMap;

pub(crate) fn hydrate_session_message_tool_call_displays(records: &mut [SessionMessageRecord]) {
    let displays = persisted_tool_call_displays(records);
    for record in records {
        for tool_call in &mut record.tool_calls {
            attach_tool_call_display(tool_call, &displays);
        }
    }
}

fn attach_tool_call_display(tool_call: &mut Value, displays: &HashMap<String, Value>) {
    let display = tool_call
        .get("id")
        .and_then(Value::as_str)
        .and_then(|id| displays.get(id))
        .cloned()
        .or_else(|| persisted_tool_call_display(tool_call));

    let Some(display) = display else {
        return;
    };

    if let Value::Object(call) = tool_call {
        call.entry("display").or_insert(display);
    }
}

fn persisted_tool_call_displays(records: &[SessionMessageRecord]) -> HashMap<String, Value> {
    let mut partials = HashMap::<String, PersistedToolCallDisplayPartial>::new();

    for record in records {
        for tool_call in &record.tool_calls {
            let Some(id) = tool_call
                .get("id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
            else {
                continue;
            };

            let partial = partials.entry(id.to_string()).or_default();
            if let Some(name) = tool_call_name(tool_call) {
                partial.name = Some(name.to_string());
            }
            if let Some(arguments) = tool_call_arguments_text(tool_call) {
                partial.arguments.push_str(arguments);
            }
        }
    }

    partials
        .into_iter()
        .filter_map(|(id, partial)| {
            let name = partial.name?;
            let arguments = serde_json::from_str::<Value>(&partial.arguments).ok()?;
            let display = tool_call_display(&name, &arguments)?;
            serde_json::to_value(display)
                .ok()
                .map(|display| (id, display))
        })
        .collect()
}

fn persisted_tool_call_display(tool_call: &Value) -> Option<Value> {
    let name = tool_call_name(tool_call)?;
    let arguments = tool_call_arguments_value(tool_call)?;
    let display = tool_call_display(name, &arguments)?;
    serde_json::to_value(display).ok()
}

fn tool_call_name(tool_call: &Value) -> Option<&str> {
    tool_call
        .get("function")
        .and_then(Value::as_object)
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
}

fn tool_call_arguments_text(tool_call: &Value) -> Option<&str> {
    tool_call
        .get("function")
        .and_then(Value::as_object)
        .and_then(|function| function.get("arguments"))
        .and_then(Value::as_str)
}

fn tool_call_arguments_value(tool_call: &Value) -> Option<Value> {
    let arguments = tool_call
        .get("function")
        .and_then(Value::as_object)
        .and_then(|function| function.get("arguments"))?;
    match arguments {
        Value::String(value) => serde_json::from_str(value).ok(),
        Value::Object(_) => Some(arguments.clone()),
        _ => None,
    }
}

#[derive(Default)]
struct PersistedToolCallDisplayPartial {
    name: Option<String>,
    arguments: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::test_message_record;
    use seekcode_common::{ChatRole, SessionId};

    #[test]
    fn hydrate_session_messages_adds_backend_tool_display_to_history() {
        let session_id = SessionId::new();
        let mut records = vec![
            test_message_record(
                session_id,
                1,
                ChatRole::Assistant,
                vec![serde_json::json!({
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": seekcode_tool_system::RUN_COMMAND_TOOL,
                        "arguments": "{\"command\":\"cargo "
                    }
                })],
            ),
            test_message_record(
                session_id,
                1,
                ChatRole::Assistant,
                vec![serde_json::json!({
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "arguments": "test\\ncargo check\"}"
                    }
                })],
            ),
        ];

        hydrate_session_message_tool_call_displays(&mut records);
        let first_display = records[0].tool_calls[0].get("display").expect("display");
        let second_display = records[1].tool_calls[0].get("display").expect("display");

        assert_eq!(
            first_display.get("title").and_then(Value::as_str),
            Some("Shell")
        );
        assert_eq!(
            first_display.get("preview").and_then(Value::as_str),
            Some("cargo test")
        );
        assert_eq!(first_display, second_display);
    }
}
