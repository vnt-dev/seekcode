use crate::models::AuditLogRecord;
use crate::sqlite::SqliteStorage;
use crate::traits::AuditStore;
use async_trait::async_trait;
use seekcode_common::{SeekCodeError, SeekCodeResult};

#[async_trait]
impl AuditStore for SqliteStorage {
    async fn write_audit_log(&self, _record: AuditLogRecord) -> SeekCodeResult<()> {
        Err(SeekCodeError::NotImplemented(
            "audit log storage is not wired yet",
        ))
    }
}
