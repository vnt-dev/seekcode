use crate::time::{local_now_text, LOCAL_TIME_FORMAT};
use crate::{
    MigrationRunner, ModelCallLogStore, NewSession, NewSessionMessage, NewWorkspace,
    SessionContextStore, SessionStore, SqliteStorage, WorkspaceStore,
};
use chrono::NaiveDateTime;
use seekcode_common::{ChatMessage, ChatRole, ModelCallLogId, SessionId, WorkspaceId};
use sqlx::sqlite::SqlitePoolOptions;

#[tokio::test]
async fn migrations_create_requested_tables() {
    let storage = test_storage().await;

    let table_names: Vec<String> = sqlx::query_scalar(
        r#"
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name IN ('workspaces', 'sessions', 'session_messages', 'model_call_logs')
        ORDER BY name
        "#,
    )
    .fetch_all(storage.pool())
    .await
    .expect("table names");

    assert_eq!(
        table_names,
        vec![
            "model_call_logs".to_string(),
            "session_messages".to_string(),
            "sessions".to_string(),
            "workspaces".to_string()
        ]
    );
}

#[tokio::test]
async fn stores_workspace_session_and_messages_with_local_time_text() {
    let storage = test_storage().await;
    let workspace_id = WorkspaceId::new();
    let session_id = SessionId::new();

    let workspace = storage
        .create_workspace(NewWorkspace {
            id: workspace_id,
            name: "SeekCode".to_string(),
            absolute_path: "D:\\rust\\tmp\\seekcode".to_string(),
            is_visible: true,
        })
        .await
        .expect("create workspace");

    assert_eq!(workspace.id, workspace_id);
    assert!(workspace.is_visible);
    assert_local_time_text(&workspace.created_at);
    assert_local_time_text(&workspace.updated_at);

    let session = storage
        .create_session(NewSession {
            id: session_id,
            workspace_id,
            name: "Initial chat".to_string(),
            model_provider: "deepseek".to_string(),
            model: "deepseek-chat".to_string(),
            thinking_enabled: true,
            reasoning_effort: Some("medium".to_string()),
        })
        .await
        .expect("create session");

    assert_eq!(session.workspace_id, workspace_id);
    assert_eq!(session.name, "Initial chat");
    assert!(session.thinking_enabled);
    assert_local_time_text(&session.created_at);
    assert_local_time_text(&session.updated_at);

    let renamed = storage
        .rename_session(session_id, "Generated title".to_string())
        .await
        .expect("rename session");
    assert_eq!(renamed.name, "Generated title");

    let updated_model = storage
        .update_session_model(
            session_id,
            "deepseek".to_string(),
            "deepseek-v4-flash".to_string(),
            false,
            Some("max".to_string()),
        )
        .await
        .expect("update session model");
    assert_eq!(updated_model.model_provider, "deepseek");
    assert_eq!(updated_model.model, "deepseek-v4-flash");
    assert!(!updated_model.thinking_enabled);
    assert_eq!(updated_model.reasoning_effort.as_deref(), Some("max"));

    storage
        .append_session_message(NewSessionMessage {
            session_id,
            turn_sequence: 1,
            role: ChatRole::User,
            content: "Hello".to_string(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            created_at: local_now_text(),
        })
        .await
        .expect("append user message");
    storage
        .append_message(
            session_id,
            ChatMessage::new(ChatRole::Assistant, "Hi, what should we build?"),
        )
        .await
        .expect("append chat message");

    let sessions = storage
        .list_workspace_sessions(workspace_id)
        .await
        .expect("list workspace sessions");
    let messages = storage
        .list_session_messages(session_id)
        .await
        .expect("list session messages");

    assert_eq!(sessions.len(), 1);
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[1].id, messages[0].id + 1);
    assert_eq!(messages[0].turn_sequence, 1);
    assert_eq!(messages[1].turn_sequence, 2);
    assert_eq!(messages[1].role, ChatRole::Assistant);
    assert_local_time_text(&messages[0].created_at);
    assert_local_time_text(&messages[1].created_at);

    storage
        .append_model_call_log(crate::NewModelCallLog {
            id: ModelCallLogId::new(),
            model_provider: "deepseek".to_string(),
            model: "deepseek-v4-flash".to_string(),
            session_id,
            input_tokens: 12,
            output_tokens: 4,
            cache_hit_tokens: 3,
            elapsed_ms: 120,
            success: true,
            called_at: local_now_text(),
        })
        .await
        .expect("append model call log");

    storage
        .delete_session(session_id)
        .await
        .expect("delete session");
    let message_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM session_messages WHERE session_id = ?1")
            .bind(session_id.to_string())
            .fetch_one(storage.pool())
            .await
            .expect("message count");
    assert_eq!(message_count, 0);
    let model_call_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM model_call_logs WHERE session_id = ?1")
            .bind(session_id.to_string())
            .fetch_one(storage.pool())
            .await
            .expect("model call count");
    assert_eq!(model_call_count, 1);
}

#[tokio::test]
async fn lists_session_messages_by_recent_turn_pages() {
    let storage = test_storage().await;
    let workspace_id = WorkspaceId::new();
    let session_id = SessionId::new();
    storage
        .create_workspace(NewWorkspace {
            id: workspace_id,
            name: "SeekCode".to_string(),
            absolute_path: "D:\\rust\\tmp\\seekcode".to_string(),
            is_visible: true,
        })
        .await
        .expect("create workspace");
    storage
        .create_session(NewSession {
            id: session_id,
            workspace_id,
            name: "Paged chat".to_string(),
            model_provider: "deepseek".to_string(),
            model: "deepseek-v4-pro".to_string(),
            thinking_enabled: true,
            reasoning_effort: None,
        })
        .await
        .expect("create session");

    for turn_sequence in 1..=25 {
        storage
            .append_session_message(NewSessionMessage {
                session_id,
                turn_sequence,
                role: ChatRole::User,
                content: format!("Question {turn_sequence}"),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                created_at: local_now_text(),
            })
            .await
            .expect("append message");
    }
    storage
        .append_session_message(NewSessionMessage {
            session_id,
            turn_sequence: 25,
            role: ChatRole::Assistant,
            content: "Answer 25".to_string(),
            reasoning_content: Some("Reasoning 25".to_string()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            created_at: local_now_text(),
        })
        .await
        .expect("append second record in latest turn");

    let latest = storage
        .list_session_messages_page(session_id, None, 20)
        .await
        .expect("latest page");
    assert_eq!(latest.first().map(|message| message.turn_sequence), Some(6));
    assert_eq!(latest.last().map(|message| message.turn_sequence), Some(25));
    assert_eq!(latest.len(), 21);
    assert_eq!(
        latest
            .iter()
            .map(|message| message.turn_sequence)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        20
    );

    let older = storage
        .list_session_messages_page(session_id, Some(6), 20)
        .await
        .expect("older page");
    assert_eq!(older.len(), 5);
    assert_eq!(older.first().map(|message| message.turn_sequence), Some(1));
    assert_eq!(older.last().map(|message| message.turn_sequence), Some(5));

    let range = storage
        .list_session_messages_in_turn_range(session_id, 10, Some(15))
        .await
        .expect("turn range");
    assert_eq!(range.len(), 4);
    assert_eq!(range.first().map(|message| message.turn_sequence), Some(11));
    assert_eq!(range.last().map(|message| message.turn_sequence), Some(14));
}

#[tokio::test]
async fn lists_session_messages_in_insert_order_within_same_turn() {
    let storage = test_storage().await;
    let workspace_id = WorkspaceId::new();
    let session_id = SessionId::new();
    storage
        .create_workspace(NewWorkspace {
            id: workspace_id,
            name: "SeekCode".to_string(),
            absolute_path: "D:\\rust\\tmp\\seekcode".to_string(),
            is_visible: true,
        })
        .await
        .expect("create workspace");
    storage
        .create_session(NewSession {
            id: session_id,
            workspace_id,
            name: "Ordered chat".to_string(),
            model_provider: "deepseek".to_string(),
            model: "deepseek-v4-pro".to_string(),
            thinking_enabled: true,
            reasoning_effort: None,
        })
        .await
        .expect("create session");

    for (role, content) in [
        (ChatRole::User, "question"),
        (ChatRole::Assistant, "tool call"),
        (ChatRole::Tool, "tool result"),
        (ChatRole::Assistant, "answer"),
    ] {
        storage
            .append_session_message(NewSessionMessage {
                session_id,
                turn_sequence: 1,
                role,
                content: content.to_string(),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                created_at: "2026-01-01 00:00:00".to_string(),
            })
            .await
            .expect("append message");
    }

    let messages = storage
        .list_session_messages(session_id)
        .await
        .expect("messages list");
    assert_eq!(
        messages
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>(),
        vec!["question", "tool call", "tool result", "answer"]
    );
}

#[tokio::test]
async fn session_context_state_appends_and_session_stores_token_watermark() {
    let storage = test_storage().await;
    let workspace_id = WorkspaceId::new();
    let session_id = SessionId::new();
    storage
        .create_workspace(NewWorkspace {
            id: workspace_id,
            name: "SeekCode".to_string(),
            absolute_path: "D:\\rust\\tmp\\seekcode".to_string(),
            is_visible: true,
        })
        .await
        .expect("create workspace");
    storage
        .create_session(NewSession {
            id: session_id,
            workspace_id,
            name: "Chat".to_string(),
            model_provider: "deepseek".to_string(),
            model: "deepseek-v4-pro".to_string(),
            thinking_enabled: true,
            reasoning_effort: None,
        })
        .await
        .expect("create session");

    assert!(storage
        .get_session_context_state(session_id)
        .await
        .expect("get missing state")
        .is_none());

    storage
        .update_session_last_input_tokens(session_id, 4_096)
        .await
        .expect("update tokens");
    let session = storage.get_session(session_id).await.expect("get session");
    assert_eq!(session.last_input_tokens, 4_096);
    assert!(storage
        .get_session_context_state(session_id)
        .await
        .expect("get missing state after token update")
        .is_none());

    storage
        .save_session_compaction(session_id, "summary text".to_string(), 5)
        .await
        .expect("save compaction");
    let state = storage
        .get_session_context_state(session_id)
        .await
        .expect("get state")
        .expect("state exists");
    assert_eq!(state.summary, "summary text");
    assert_eq!(state.compacted_through_turn, 5);

    storage
        .save_session_compaction(session_id, "newer summary".to_string(), 8)
        .await
        .expect("save second compaction");
    let state = storage
        .get_session_context_state(session_id)
        .await
        .expect("get latest state")
        .expect("state exists");
    assert_eq!(state.summary, "newer summary");
    assert_eq!(state.compacted_through_turn, 8);
    assert_local_time_text(&state.updated_at);

    let row_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM session_context_state WHERE session_id = ?1")
            .bind(session_id.to_string())
            .fetch_one(storage.pool())
            .await
            .expect("context state count");
    assert_eq!(row_count, 2);

    storage
        .update_session_last_input_tokens(session_id, 9_000)
        .await
        .expect("update tokens again");
    let session = storage.get_session(session_id).await.expect("get session");
    assert_eq!(session.last_input_tokens, 9_000);
}

#[tokio::test]
async fn session_model_call_stats_aggregates_rows() {
    let storage = test_storage().await;
    let workspace_id = WorkspaceId::new();
    let session_id = SessionId::new();
    storage
        .create_workspace(NewWorkspace {
            id: workspace_id,
            name: "SeekCode".to_string(),
            absolute_path: "D:\\rust\\tmp\\seekcode".to_string(),
            is_visible: true,
        })
        .await
        .expect("create workspace");
    storage
        .create_session(NewSession {
            id: session_id,
            workspace_id,
            name: "Chat".to_string(),
            model_provider: "deepseek".to_string(),
            model: "deepseek-v4-pro".to_string(),
            thinking_enabled: true,
            reasoning_effort: None,
        })
        .await
        .expect("create session");

    let empty = storage
        .session_model_call_stats(session_id)
        .await
        .expect("empty stats");
    assert_eq!(empty.call_count, 0);
    assert_eq!(empty.input_tokens, 0);

    for (input, output, cache) in [(100, 40, 30), (200, 60, 50)] {
        storage
            .append_model_call_log(crate::NewModelCallLog {
                id: ModelCallLogId::new(),
                model_provider: "deepseek".to_string(),
                model: "deepseek-v4-pro".to_string(),
                session_id,
                input_tokens: input,
                output_tokens: output,
                cache_hit_tokens: cache,
                elapsed_ms: 10,
                success: true,
                called_at: local_now_text(),
            })
            .await
            .expect("append model call log");
    }

    let stats = storage
        .session_model_call_stats(session_id)
        .await
        .expect("aggregated stats");
    assert_eq!(stats.call_count, 2);
    assert_eq!(stats.input_tokens, 300);
    assert_eq!(stats.output_tokens, 100);
    assert_eq!(stats.cache_hit_tokens, 80);
}

fn assert_local_time_text(value: &str) {
    NaiveDateTime::parse_from_str(value, LOCAL_TIME_FORMAT).expect("local time text format");
    assert_eq!(value.len(), "yyyy-MM-dd HH:mm:ss".len());
}

async fn test_storage() -> SqliteStorage {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .expect("connect in-memory sqlite");
    MigrationRunner::run(&pool).await.expect("run migrations");
    SqliteStorage::new(pool)
}
