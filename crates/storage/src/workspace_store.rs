use crate::models::{NewWorkspace, WorkspaceRecord};
use crate::rows::{
    bool_to_i64, ensure_affected, i64_to_bool, not_found, parse_id, row_get, storage_error,
};
use crate::sqlite::SqliteStorage;
use crate::time::local_now_text;
use crate::traits::WorkspaceStore;
use async_trait::async_trait;
use seekcode_common::{SeekCodeResult, WorkspaceId};
use sqlx::SqlitePool;

#[async_trait]
impl WorkspaceStore for SqliteStorage {
    async fn create_workspace(&self, workspace: NewWorkspace) -> SeekCodeResult<WorkspaceRecord> {
        let now = local_now_text();

        sqlx::query(
            r#"
            INSERT INTO workspaces (id, name, absolute_path, is_visible, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            "#,
        )
        .bind(workspace.id.to_string())
        .bind(&workspace.name)
        .bind(&workspace.absolute_path)
        .bind(bool_to_i64(workspace.is_visible))
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await
        .map_err(storage_error)?;

        get_workspace(&self.pool, workspace.id).await
    }

    async fn get_workspace(&self, workspace_id: WorkspaceId) -> SeekCodeResult<WorkspaceRecord> {
        get_workspace(&self.pool, workspace_id).await
    }

    async fn find_workspace_by_path(
        &self,
        absolute_path: &str,
    ) -> SeekCodeResult<Option<WorkspaceRecord>> {
        let row = sqlx::query(
            r#"
            SELECT id, name, absolute_path, is_visible, created_at, updated_at
            FROM workspaces
            WHERE absolute_path = ?1
            "#,
        )
        .bind(absolute_path)
        .fetch_optional(&self.pool)
        .await
        .map_err(storage_error)?;

        row.map(workspace_from_row).transpose()
    }

    async fn list_workspaces(&self) -> SeekCodeResult<Vec<WorkspaceRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, name, absolute_path, is_visible, created_at, updated_at
            FROM workspaces
            ORDER BY updated_at DESC, created_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(storage_error)?;

        rows.into_iter().map(workspace_from_row).collect()
    }

    async fn list_visible_workspaces(&self) -> SeekCodeResult<Vec<WorkspaceRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, name, absolute_path, is_visible, created_at, updated_at
            FROM workspaces
            WHERE is_visible = 1
            ORDER BY updated_at DESC, created_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(storage_error)?;

        rows.into_iter().map(workspace_from_row).collect()
    }

    async fn set_workspace_visibility(
        &self,
        workspace_id: WorkspaceId,
        is_visible: bool,
    ) -> SeekCodeResult<()> {
        let result = sqlx::query(
            r#"
            UPDATE workspaces
            SET is_visible = ?1, updated_at = ?2
            WHERE id = ?3
            "#,
        )
        .bind(bool_to_i64(is_visible))
        .bind(local_now_text())
        .bind(workspace_id.to_string())
        .execute(&self.pool)
        .await
        .map_err(storage_error)?;

        ensure_affected(result.rows_affected(), "workspace", workspace_id)
    }
}

pub(crate) async fn get_workspace(
    pool: &SqlitePool,
    workspace_id: WorkspaceId,
) -> SeekCodeResult<WorkspaceRecord> {
    let row = sqlx::query(
        r#"
        SELECT id, name, absolute_path, is_visible, created_at, updated_at
        FROM workspaces
        WHERE id = ?1
        "#,
    )
    .bind(workspace_id.to_string())
    .fetch_optional(pool)
    .await
    .map_err(storage_error)?
    .ok_or_else(|| not_found("workspace", workspace_id))?;

    workspace_from_row(row)
}

fn workspace_from_row(row: sqlx::sqlite::SqliteRow) -> SeekCodeResult<WorkspaceRecord> {
    Ok(WorkspaceRecord {
        id: parse_id(row_get::<String>(&row, "id")?)?,
        name: row_get(&row, "name")?,
        absolute_path: row_get(&row, "absolute_path")?,
        is_visible: i64_to_bool(row_get(&row, "is_visible")?),
        created_at: row_get(&row, "created_at")?,
        updated_at: row_get(&row, "updated_at")?,
    })
}
