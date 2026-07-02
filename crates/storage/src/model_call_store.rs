use crate::models::{ModelCallLogRecord, NewModelCallLog, SessionModelCallStats};
use crate::rows::{bool_to_i64, i64_to_bool, parse_id, row_get, storage_error};
use crate::sqlite::SqliteStorage;
use crate::traits::ModelCallLogStore;
use async_trait::async_trait;
use seekcode_common::{SeekCodeResult, SessionId};

#[async_trait]
impl ModelCallLogStore for SqliteStorage {
    async fn append_model_call_log(
        &self,
        log: NewModelCallLog,
    ) -> SeekCodeResult<ModelCallLogRecord> {
        sqlx::query(
            r#"
            INSERT INTO model_call_logs (
                id,
                model_provider,
                model,
                session_id,
                input_tokens,
                output_tokens,
                cache_hit_tokens,
                elapsed_ms,
                success,
                called_at
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            "#,
        )
        .bind(log.id.to_string())
        .bind(&log.model_provider)
        .bind(&log.model)
        .bind(log.session_id.to_string())
        .bind(log.input_tokens)
        .bind(log.output_tokens)
        .bind(log.cache_hit_tokens)
        .bind(log.elapsed_ms)
        .bind(bool_to_i64(log.success))
        .bind(&log.called_at)
        .execute(self.pool())
        .await
        .map_err(storage_error)?;

        get_model_call_log(self.pool(), log.id).await
    }

    async fn session_model_call_stats(
        &self,
        session_id: SessionId,
    ) -> SeekCodeResult<SessionModelCallStats> {
        let row = sqlx::query(
            r#"
            SELECT
                (
                    SELECT COUNT(*)
                    FROM model_call_logs
                    WHERE session_id = ?1
                ) AS call_count,
                (
                    SELECT COALESCE(SUM(input_tokens), 0)
                    FROM model_call_logs
                    WHERE session_id = ?1
                ) AS input_tokens,
                (
                    SELECT COALESCE(SUM(output_tokens), 0)
                    FROM model_call_logs
                    WHERE session_id = ?1
                ) AS output_tokens,
                (
                    SELECT COALESCE(SUM(cache_hit_tokens), 0)
                    FROM model_call_logs
                    WHERE session_id = ?1
                ) AS cache_hit_tokens,
                (
                    SELECT COALESCE(CAST(ROUND(AVG(elapsed_ms)) AS INTEGER), 0)
                    FROM model_call_logs
                    WHERE session_id = ?1
                ) AS average_call_elapsed_ms,
                (
                    SELECT COALESCE(CAST(ROUND(AVG(turn_elapsed_ms)) AS INTEGER), 0)
                    FROM (
                        SELECT
                            (julianday(MAX(created_at)) - julianday(MIN(created_at))) * 86400000
                                AS turn_elapsed_ms
                        FROM session_messages
                        WHERE session_id = ?1
                        GROUP BY turn_sequence
                        HAVING COUNT(*) > 1
                    )
                ) AS average_turn_elapsed_ms
            "#,
        )
        .bind(session_id.to_string())
        .fetch_one(self.pool())
        .await
        .map_err(storage_error)?;

        Ok(SessionModelCallStats {
            call_count: row_get(&row, "call_count")?,
            input_tokens: row_get(&row, "input_tokens")?,
            output_tokens: row_get(&row, "output_tokens")?,
            cache_hit_tokens: row_get(&row, "cache_hit_tokens")?,
            average_call_elapsed_ms: row_get(&row, "average_call_elapsed_ms")?,
            average_turn_elapsed_ms: row_get(&row, "average_turn_elapsed_ms")?,
        })
    }
}

async fn get_model_call_log(
    pool: &sqlx::SqlitePool,
    id: seekcode_common::ModelCallLogId,
) -> SeekCodeResult<ModelCallLogRecord> {
    let row = sqlx::query(
        r#"
        SELECT id, model_provider, model, session_id, input_tokens, output_tokens, cache_hit_tokens, elapsed_ms, success, called_at
        FROM model_call_logs
        WHERE id = ?1
        "#,
    )
    .bind(id.to_string())
    .fetch_one(pool)
    .await
    .map_err(storage_error)?;

    Ok(ModelCallLogRecord {
        id: parse_id(row_get(&row, "id")?)?,
        model_provider: row_get(&row, "model_provider")?,
        model: row_get(&row, "model")?,
        session_id: parse_id(row_get(&row, "session_id")?)?,
        input_tokens: row_get(&row, "input_tokens")?,
        output_tokens: row_get(&row, "output_tokens")?,
        cache_hit_tokens: row_get(&row, "cache_hit_tokens")?,
        elapsed_ms: row_get(&row, "elapsed_ms")?,
        success: i64_to_bool(row_get(&row, "success")?),
        called_at: row_get(&row, "called_at")?,
    })
}
