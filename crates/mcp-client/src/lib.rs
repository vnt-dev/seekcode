//! Optional MCP client boundary reserved for future tool integrations.

use seekcode_common::SeekCodeResult;
use serde::{Deserialize, Serialize};

/// MCP server configuration placeholder.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Human-readable server name.
    pub name: String,
    /// Launch command or transport endpoint.
    pub endpoint: String,
}

/// MCP client placeholder.
pub struct McpClient;

impl McpClient {
    /// Connects to an MCP server.
    pub async fn connect(_config: McpServerConfig) -> SeekCodeResult<Self> {
        todo!("connect MCP client")
    }
}
