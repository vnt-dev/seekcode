use crate::models::{NewSession, NewSessionMessage, SessionMessageRecord, SessionRecord};
use crate::rows::{
    bool_to_i64, chat_role_from_str, chat_role_to_str, ensure_affected, i64_to_bool, not_found,
    parse_id, row_get, storage_error,
};
use crate::sqlite::SqliteStorage;
use crate::time::{local_now_text, utc_to_local_text};
use crate::traits::SessionStore;
use async_trait::async_trait;
use seekcode_common::{ChatMessage, SeekCodeResult, SessionId, ToolCallId, WorkspaceId};
use serde_json::Value;
use sqlx::{Row, SqlitePool};

#[async_trait]
impl SessionStore for SqliteStorage {
    async fn create_session(&self, session: NewSession) -> SeekCodeResult<SessionRecord> {
        let now = local_now_text();

        sqlx::query(
            r#"
            INSERT INTO sessions (
                id,
                workspace_id,
                name,
                model_provider,
                model,
                thinking_enabled,
                reasoning_effort,
                created_at,
                updated_at
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            "#,
        )
        .bind(session.id.to_string())
        .bind(session.workspace_id.to_string())
        .bind(&session.name)
        .bind(&session.model_provider)
        .bind(&session.model)
        .bind(bool_to_i64(session.thinking_enabled))
        .bind(&session.reasoning_effort)
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await
        .map_err(storage_error)?;

        get_session(&self.pool, session.id).await
    }

    async fn get_session(&self, session_id: SessionId) -> SeekCodeResult<SessionRecord> {
        get_session(&self.pool, session_id).await
    }

    async fn rename_session(
        &self,
        session_id: SessionId,
        name: String,
    ) -> SeekCodeResult<SessionRecord> {
        let result = sqlx::query(
            r#"
            UPDATE sessions
            SET name = ?1, updated_at = ?2
            WHERE id = ?3
            "#,
        )
        .bind(name)
        .bind(local_now_text())
        .bind(session_id.to_string())
        .execute(&self.pool)
        .await
        .map_err(storage_error)?;

        ensure_affected(result.rows_affected(), "session", session_id)?;
        get_session(&self.pool, session_id).await
    }

    async fn update_session_model(
        &self,
        session_id: SessionId,
        model_provider: String,
        model: String,
        thinking_enabled: bool,
        reasoning_effort: Option<String>,
    ) -> SeekCodeResult<SessionRecord> {
        let result = sqlx::query(
            r#"
            UPDATE sessions
            SET model_provider = ?1,
                model = ?2,
                thinking_enabled = ?3,
                reasoning_effort = ?4,
                updated_at = ?5
            WHERE id = ?6
            "#,
        )
        .bind(model_provider)
        .bind(model)
        .bind(bool_to_i64(thinking_enabled))
        .bind(reasoning_effort)
        .bind(local_now_text())
        .bind(session_id.to_string())
        .execute(&self.pool)
        .await
        .map_err(storage_error)?;

        ensure_affected(result.rows_affected(), "session", session_id)?;
        get_session(&self.pool, session_id).await
    }

    async fn list_sessions(&self) -> SeekCodeResult<Vec<SessionRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, workspace_id, name, model_provider, model, thinking_enabled, reasoning_effort, last_input_tokens, created_at, updated_at
            FROM sessions
            ORDER BY updated_at DESC, created_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(storage_error)?;

        rows.into_iter().map(session_from_row).collect()
    }

    async fn list_workspace_sessions(
        &self,
        workspace_id: WorkspaceId,
    ) -> SeekCodeResult<Vec<SessionRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, workspace_id, name, model_provider, model, thinking_enabled, reasoning_effort, last_input_tokens, created_at, updated_at
            FROM sessions
            WHERE workspace_id = ?1
            ORDER BY updated_at DESC, created_at DESC
            "#,
        )
        .bind(workspace_id.to_string())
        .fetch_all(&self.pool)
        .await
        .map_err(storage_error)?;

        rows.into_iter().map(session_from_row).collect()
    }

    async fn delete_session(&self, session_id: SessionId) -> SeekCodeResult<()> {
        let result = sqlx::query("DELETE FROM sessions WHERE id = ?1")
            .bind(session_id.to_string())
            .execute(&self.pool)
            .await
            .map_err(storage_error)?;

        ensure_affected(result.rows_affected(), "session", session_id)
    }

    async fn delete_workspace_sessions(&self, workspace_id: WorkspaceId) -> SeekCodeResult<()> {
        sqlx::query("DELETE FROM sessions WHERE workspace_id = ?1")
            .bind(workspace_id.to_string())
            .execute(&self.pool)
            .await
            .map_err(storage_error)?;

        Ok(())
    }

    async fn append_session_message(
        &self,
        message: NewSessionMessage,
    ) -> SeekCodeResult<SessionMessageRecord> {
        let result = sqlx::query(
            r#"
            INSERT INTO session_messages (
                session_id,
                turn_sequence,
                role,
                content,
                reasoning_content,
                tool_calls,
                tool_call_id,
                created_at
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            "#,
        )
        .bind(message.session_id.to_string())
        .bind(message.turn_sequence)
        .bind(chat_role_to_str(&message.role))
        .bind(&message.content)
        .bind(&message.reasoning_content)
        .bind(serde_json::to_string(&message.tool_calls).map_err(storage_error)?)
        .bind(message.tool_call_id.map(|id| id.to_string()))
        .bind(&message.created_at)
        .execute(&self.pool)
        .await
        .map_err(storage_error)?;

        touch_session(&self.pool, message.session_id).await?;
        get_session_message(&self.pool, result.last_insert_rowid()).await
    }

    async fn list_session_messages(
        &self,
        session_id: SessionId,
    ) -> SeekCodeResult<Vec<SessionMessageRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, turn_sequence, role, content, reasoning_content, tool_calls, tool_call_id, created_at
            FROM session_messages
            WHERE session_id = ?1
            ORDER BY turn_sequence ASC, id ASC
            "#,
        )
        .bind(session_id.to_string())
        .fetch_all(&self.pool)
        .await
        .map_err(storage_error)?;

        rows.into_iter().map(session_message_from_row).collect()
    }

    async fn list_session_messages_in_turn_range(
        &self,
        session_id: SessionId,
        after_turn_sequence: i64,
        before_turn_sequence: Option<i64>,
    ) -> SeekCodeResult<Vec<SessionMessageRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, turn_sequence, role, content, reasoning_content, tool_calls, tool_call_id, created_at
            FROM session_messages
            WHERE session_id = ?1
              AND turn_sequence > ?2
              AND (?3 IS NULL OR turn_sequence < ?3)
            ORDER BY turn_sequence ASC, id ASC
            "#,
        )
        .bind(session_id.to_string())
        .bind(after_turn_sequence)
        .bind(before_turn_sequence)
        .fetch_all(&self.pool)
        .await
        .map_err(storage_error)?;

        rows.into_iter().map(session_message_from_row).collect()
    }

    async fn list_session_messages_page(
        &self,
        session_id: SessionId,
        before_turn_sequence: Option<i64>,
        turn_limit: i64,
    ) -> SeekCodeResult<Vec<SessionMessageRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, turn_sequence, role, content, reasoning_content, tool_calls, tool_call_id, created_at
            FROM session_messages
            WHERE session_id = ?1
              AND turn_sequence IN (
                SELECT turn_sequence
                FROM (
                  SELECT DISTINCT turn_sequence
                  FROM session_messages
                  WHERE session_id = ?1
                    AND (?2 IS NULL OR turn_sequence < ?2)
                  ORDER BY turn_sequence DESC
                  LIMIT ?3
                )
              )
            ORDER BY turn_sequence ASC, id ASC
            "#,
        )
        .bind(session_id.to_string())
        .bind(before_turn_sequence)
        .bind(turn_limit.max(1))
        .fetch_all(&self.pool)
        .await
        .map_err(storage_error)?;

        rows.into_iter().map(session_message_from_row).collect()
    }

    async fn next_session_turn_sequence(&self, session_id: SessionId) -> SeekCodeResult<i64> {
        next_turn_sequence(&self.pool, session_id).await
    }

    async fn append_message(
        &self,
        session_id: SessionId,
        message: ChatMessage,
    ) -> SeekCodeResult<()> {
        let turn_sequence = next_turn_sequence(&self.pool, session_id).await?;
        let new_message = NewSessionMessage {
            session_id,
            turn_sequence,
            role: message.role,
            content: message.content,
            reasoning_content: message.reasoning_content,
            tool_calls: message.tool_calls,
            tool_call_id: message.tool_call_id,
            created_at: utc_to_local_text(message.created_at),
        };

        self.append_session_message(new_message).await?;

        Ok(())
    }

    async fn update_session_last_input_tokens(
        &self,
        session_id: SessionId,
        last_input_tokens: i64,
    ) -> SeekCodeResult<()> {
        let result = sqlx::query(
            r#"
            UPDATE sessions
            SET last_input_tokens = ?1,
                updated_at = ?2
            WHERE id = ?3
            "#,
        )
        .bind(last_input_tokens)
        .bind(local_now_text())
        .bind(session_id.to_string())
        .execute(&self.pool)
        .await
        .map_err(storage_error)?;

        ensure_affected(result.rows_affected(), "session", session_id)
    }
}

async fn get_session(pool: &SqlitePool, session_id: SessionId) -> SeekCodeResult<SessionRecord> {
    let row = sqlx::query(
        r#"
        SELECT id, workspace_id, name, model_provider, model, thinking_enabled, reasoning_effort, last_input_tokens, created_at, updated_at
        FROM sessions
        WHERE id = ?1
        "#,
    )
    .bind(session_id.to_string())
    .fetch_optional(pool)
    .await
    .map_err(storage_error)?
    .ok_or_else(|| not_found("session", session_id))?;

    session_from_row(row)
}

async fn get_session_message(
    pool: &SqlitePool,
    message_id: i64,
) -> SeekCodeResult<SessionMessageRecord> {
    let row = sqlx::query(
        r#"
        SELECT id, session_id, turn_sequence, role, content, reasoning_content, tool_calls, tool_call_id, created_at
        FROM session_messages
        WHERE id = ?1
        "#,
    )
    .bind(message_id)
    .fetch_optional(pool)
    .await
    .map_err(storage_error)?
    .ok_or_else(|| not_found("session message", message_id))?;

    session_message_from_row(row)
}

async fn touch_session(pool: &SqlitePool, session_id: SessionId) -> SeekCodeResult<()> {
    let result = sqlx::query(
        r#"
        UPDATE sessions
        SET updated_at = ?1
        WHERE id = ?2
        "#,
    )
    .bind(local_now_text())
    .bind(session_id.to_string())
    .execute(pool)
    .await
    .map_err(storage_error)?;

    ensure_affected(result.rows_affected(), "session", session_id)
}

async fn next_turn_sequence(pool: &SqlitePool, session_id: SessionId) -> SeekCodeResult<i64> {
    let row = sqlx::query(
        r#"
        SELECT COALESCE(MAX(turn_sequence), 0) + 1 AS next_turn_sequence
        FROM session_messages
        WHERE session_id = ?1
        "#,
    )
    .bind(session_id.to_string())
    .fetch_one(pool)
    .await
    .map_err(storage_error)?;

    row.try_get("next_turn_sequence").map_err(storage_error)
}

fn session_from_row(row: sqlx::sqlite::SqliteRow) -> SeekCodeResult<SessionRecord> {
    Ok(SessionRecord {
        id: parse_id(row_get::<String>(&row, "id")?)?,
        workspace_id: parse_id(row_get::<String>(&row, "workspace_id")?)?,
        name: row_get(&row, "name")?,
        model_provider: row_get(&row, "model_provider")?,
        model: row_get(&row, "model")?,
        thinking_enabled: i64_to_bool(row_get(&row, "thinking_enabled")?),
        reasoning_effort: row_get(&row, "reasoning_effort")?,
        last_input_tokens: row_get(&row, "last_input_tokens")?,
        created_at: row_get(&row, "created_at")?,
        updated_at: row_get(&row, "updated_at")?,
    })
}

fn session_message_from_row(row: sqlx::sqlite::SqliteRow) -> SeekCodeResult<SessionMessageRecord> {
    let role = chat_role_from_str(row_get::<String>(&row, "role")?)?;
    let tool_calls = parse_tool_calls(row_get::<String>(&row, "tool_calls")?)?;
    let tool_call_id = row_get::<Option<String>>(&row, "tool_call_id")?
        .map(parse_id::<ToolCallId>)
        .transpose()?;

    Ok(SessionMessageRecord {
        id: row_get(&row, "id")?,
        session_id: parse_id(row_get::<String>(&row, "session_id")?)?,
        turn_sequence: row_get(&row, "turn_sequence")?,
        role,
        content: row_get(&row, "content")?,
        reasoning_content: row_get(&row, "reasoning_content")?,
        tool_calls,
        tool_call_id,
        created_at: row_get(&row, "created_at")?,
    })
}

fn parse_tool_calls(value: String) -> SeekCodeResult<Vec<Value>> {
    serde_json::from_str(&value).map_err(storage_error)
}
