use crate::rows::storage_error;
use seekcode_common::SeekCodeResult;
use sqlx::SqlitePool;

/// Database migration runner.
pub struct MigrationRunner;

impl MigrationRunner {
    /// Runs storage migrations.
    pub async fn run(pool: &SqlitePool) -> SeekCodeResult<()> {
        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(pool)
            .await
            .map_err(storage_error)?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS workspaces (
                id TEXT PRIMARY KEY NOT NULL,
                name TEXT NOT NULL,
                absolute_path TEXT NOT NULL UNIQUE,
                is_visible INTEGER NOT NULL DEFAULT 1 CHECK (is_visible IN (0, 1)),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            "#,
        )
        .execute(pool)
        .await
        .map_err(storage_error)?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY NOT NULL,
                workspace_id TEXT NOT NULL,
                name TEXT NOT NULL,
                model_provider TEXT NOT NULL,
                model TEXT NOT NULL,
                thinking_enabled INTEGER NOT NULL DEFAULT 0 CHECK (thinking_enabled IN (0, 1)),
                reasoning_effort TEXT,
                last_input_tokens INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(pool)
        .await
        .map_err(storage_error)?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS session_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_sequence INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                reasoning_content TEXT,
                tool_calls TEXT NOT NULL DEFAULT '[]',
                tool_call_id TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(pool)
        .await
        .map_err(storage_error)?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS model_call_logs (
                id TEXT PRIMARY KEY NOT NULL,
                model_provider TEXT NOT NULL,
                model TEXT NOT NULL,
                session_id TEXT NOT NULL,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                cache_hit_tokens INTEGER NOT NULL DEFAULT 0,
                elapsed_ms INTEGER NOT NULL DEFAULT 0,
                success INTEGER NOT NULL DEFAULT 0 CHECK (success IN (0, 1)),
                called_at TEXT NOT NULL
            )
            "#,
        )
        .execute(pool)
        .await
        .map_err(storage_error)?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS session_context_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                compacted_through_turn INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(pool)
        .await
        .map_err(storage_error)?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_sessions_workspace_id ON sessions(workspace_id)",
        )
        .execute(pool)
        .await
        .map_err(storage_error)?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_session_messages_session_id_turn ON session_messages(session_id, turn_sequence)",
        )
        .execute(pool)
        .await
        .map_err(storage_error)?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_session_messages_session_id_turn_order ON session_messages(session_id, turn_sequence, id)",
        )
        .execute(pool)
        .await
        .map_err(storage_error)?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_model_call_logs_session_called_at ON model_call_logs(session_id, called_at)",
        )
        .execute(pool)
        .await
        .map_err(storage_error)?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_session_context_state_session_id_id ON session_context_state(session_id, id DESC)",
        )
        .execute(pool)
        .await
        .map_err(storage_error)?;

        Ok(())
    }
}
