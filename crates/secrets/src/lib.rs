//! API key storage abstraction with an in-memory placeholder implementation.

use async_trait::async_trait;
use seekcode_common::{redact_secret, SeekCodeResult};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tokio::sync::RwLock;

/// Secret lookup key.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct SecretKey(pub String);

/// Secret value wrapper.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SecretValue(pub String);

/// Secret storage boundary.
#[async_trait]
pub trait SecretStore: Send + Sync {
    /// Reads a secret by key.
    async fn get(&self, key: &SecretKey) -> SeekCodeResult<Option<SecretValue>>;

    /// Writes a secret value.
    async fn set(&self, key: SecretKey, value: SecretValue) -> SeekCodeResult<()>;

    /// Deletes a secret by key.
    async fn delete(&self, key: &SecretKey) -> SeekCodeResult<()>;

    /// Redacts a secret for logs and UI.
    fn redact(&self, value: &SecretValue) -> String {
        redact_secret(&value.0)
    }
}

/// In-memory secret store for early development and tests.
#[derive(Default)]
pub struct InMemorySecretStore {
    values: RwLock<BTreeMap<SecretKey, SecretValue>>,
}

impl InMemorySecretStore {
    /// Creates an empty in-memory secret store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl SecretStore for InMemorySecretStore {
    async fn get(&self, key: &SecretKey) -> SeekCodeResult<Option<SecretValue>> {
        Ok(self.values.read().await.get(key).cloned())
    }

    async fn set(&self, key: SecretKey, value: SecretValue) -> SeekCodeResult<()> {
        self.values.write().await.insert(key, value);
        Ok(())
    }

    async fn delete(&self, key: &SecretKey) -> SeekCodeResult<()> {
        self.values.write().await.remove(key);
        Ok(())
    }
}
