use crate::models::{ModelCallLogRecord, NewModelCallLog};
use crate::rows::{bool_to_i64, i64_to_bool, parse_id, row_get, storage_error};
use crate::sqlite::SqliteStorage;
use crate::traits::ModelCallLogStore;
use async_trait::async_trait;
use seekcode_common::SeekCodeResult;

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
