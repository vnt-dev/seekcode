use seekcode_common::{ChatRole, SeekCodeError, SeekCodeResult};
use sqlx::{Decode, Row, Sqlite, Type};
use std::str::FromStr;

pub(crate) fn row_get<T>(row: &sqlx::sqlite::SqliteRow, column: &'static str) -> SeekCodeResult<T>
where
    for<'r> T: Decode<'r, Sqlite> + Type<Sqlite>,
{
    row.try_get(column).map_err(storage_error)
}

pub(crate) fn parse_id<T>(value: String) -> SeekCodeResult<T>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    value.parse::<T>().map_err(storage_error)
}

pub(crate) fn chat_role_to_str(role: &ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
        ChatRole::Tool => "tool",
    }
}

pub(crate) fn chat_role_from_str(value: String) -> SeekCodeResult<ChatRole> {
    match value.as_str() {
        "system" => Ok(ChatRole::System),
        "user" => Ok(ChatRole::User),
        "assistant" => Ok(ChatRole::Assistant),
        "tool" => Ok(ChatRole::Tool),
        _ => Err(SeekCodeError::Storage(format!(
            "unknown chat role in storage: {value}"
        ))),
    }
}

pub(crate) fn bool_to_i64(value: bool) -> i64 {
    i64::from(value)
}

pub(crate) fn i64_to_bool(value: i64) -> bool {
    value != 0
}

pub(crate) fn ensure_affected(
    rows_affected: u64,
    entity: &'static str,
    id: impl std::fmt::Display,
) -> SeekCodeResult<()> {
    if rows_affected == 0 {
        Err(not_found(entity, id))
    } else {
        Ok(())
    }
}

pub(crate) fn not_found(entity: &'static str, id: impl std::fmt::Display) -> SeekCodeError {
    SeekCodeError::NotFound(format!("{entity}: {id}"))
}

pub(crate) fn storage_error(error: impl std::fmt::Display) -> SeekCodeError {
    SeekCodeError::Storage(error.to_string())
}
