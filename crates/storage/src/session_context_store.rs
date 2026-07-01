use crate::models::SessionContextStateRecord;
use crate::rows::{parse_id, row_get, storage_error};
use crate::sqlite::SqliteStorage;
use crate::time::local_now_text;
use crate::traits::SessionContextStore;
use async_trait::async_trait;
use seekcode_common::{SeekCodeResult, SessionId};

#[async_trait]
impl SessionContextStore for SqliteStorage {
    async fn get_session_context_state(
        &self,
        session_id: SessionId,
    ) -> SeekCodeResult<Option<SessionContextStateRecord>> {
        let row = sqlx::query(
            r#"
            SELECT session_id, summary, compacted_through_turn, updated_at
            FROM session_context_state
            WHERE session_id = ?1
            ORDER BY id DESC
            LIMIT 1
            "#,
        )
        .bind(session_id.to_string())
        .fetch_optional(&self.pool)
        .await
        .map_err(storage_error)?;

        row.map(session_context_state_from_row).transpose()
    }

    async fn save_session_compaction(
        &self,
        session_id: SessionId,
        summary: String,
        compacted_through_turn: i64,
    ) -> SeekCodeResult<()> {
        sqlx::query(
            r#"
            INSERT INTO session_context_state (
                session_id,
                summary,
                compacted_through_turn,
                updated_at
            )
            VALUES (?1, ?2, ?3, ?4)
            "#,
        )
        .bind(session_id.to_string())
        .bind(summary)
        .bind(compacted_through_turn)
        .bind(local_now_text())
        .execute(&self.pool)
        .await
        .map_err(storage_error)?;

        Ok(())
    }
}

fn session_context_state_from_row(
    row: sqlx::sqlite::SqliteRow,
) -> SeekCodeResult<SessionContextStateRecord> {
    Ok(SessionContextStateRecord {
        session_id: parse_id(row_get::<String>(&row, "session_id")?)?,
        summary: row_get(&row, "summary")?,
        compacted_through_turn: row_get(&row, "compacted_through_turn")?,
        updated_at: row_get(&row, "updated_at")?,
    })
}
