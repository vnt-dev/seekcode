use crate::time::{local_now_text, LOCAL_TIME_FORMAT};
use crate::{
    MigrationRunner, ModelCallLogStore, NewSession, NewSessionMessage, NewWorkspace, SessionStore,
    SqliteStorage, WorkspaceStore,
};
use chrono::NaiveDateTime;
use seekcode_common::{ChatMessage, ChatRole, MessageId, ModelCallLogId, SessionId, WorkspaceId};
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
        )
        .await
        .expect("update session model");
    assert_eq!(updated_model.model_provider, "deepseek");
    assert_eq!(updated_model.model, "deepseek-v4-flash");

    storage
        .append_session_message(NewSessionMessage {
            id: MessageId::new(),
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
