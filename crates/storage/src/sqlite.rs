use crate::migrations::MigrationRunner;
use crate::rows::storage_error;
use seekcode_common::SeekCodeResult;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::SqlitePool;
use std::path::Path;
use std::str::FromStr;

/// SQLite storage implementation.
pub struct SqliteStorage {
    pub(crate) pool: SqlitePool,
}

impl SqliteStorage {
    /// Opens a SQLite database and runs migrations.
    pub async fn connect(database_url: &str) -> SeekCodeResult<Self> {
        let options = SqliteConnectOptions::from_str(database_url)
            .map_err(storage_error)?
            .create_if_missing(true)
            .foreign_keys(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await
            .map_err(storage_error)?;

        MigrationRunner::run(&pool).await?;

        Ok(Self::new(pool))
    }

    /// Opens a SQLite database file and runs migrations.
    pub async fn connect_path(path: impl AsRef<Path>) -> SeekCodeResult<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(storage_error)?;
        }

        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .foreign_keys(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await
            .map_err(storage_error)?;

        MigrationRunner::run(&pool).await?;

        Ok(Self::new(pool))
    }

    /// Creates a storage wrapper from a SQLite pool.
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Returns the underlying SQLite pool.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}
